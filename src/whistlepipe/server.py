"""Web demo server: the page (web/) plus the inference API.     python whistle.py serve  ->  http://127.0.0.1:8000

    GET  /api/status                    model files present/loaded, limits, Table 3 conditions
    POST /api/reload-model              re-read the trained files (after dropping them into place)
    POST /api/analyze                   multipart upload (file, channel, n_samples, seed)        -> {"job": id}
    POST /api/analyze-example           {"example": id, "seed": 0}   (web/data/examples.json)    -> {"job": id}
    POST /api/generate                  {"n": 8, "seed": 0, "condition": null, "fc_hz": null}  -> {"job": id}
    GET  /api/jobs/{id}                 state, stage, progress, error, result URL
    GET  /api/results/{id}/{file}       result.json, PNG, WAV of a finished job
    GET  /api/results/{id}.zip          everything of a job in one file
    GET  /                              the page itself (web/, with HTTP Range for the videos)

Jobs run one at a time on a single worker thread (the GPU holds one model). Each job writes runs/infer/<id>/;
the newest ``keep_jobs`` job folders are kept. Binds to 127.0.0.1 by default (configs/inference.toml [server]).
"""
from __future__ import annotations

import io
import json
import queue
import re
import shutil
import threading
import traceback
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audio import InputError
from .config import PROJECT_ROOT
from .infer import Engine, new_job_id, resolve

JOB_ID = re.compile(r"^(gen_)?\d{8}T\d{6}Z_[0-9a-f]{6}$")
FILE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
WEB_DIR = PROJECT_ROOT / "web"


class GenerateRequest(BaseModel):
    n: int = 8
    seed: int = 0
    condition: int | None = None
    fc_hz: float | None = None


class ExampleRequest(BaseModel):
    example: str
    seed: int = 0
    n_samples: int | None = None


class Jobs:
    def __init__(self, engine: Engine, out_root: Path, keep_jobs: int = 200):
        self.engine, self.out_root, self.keep = engine, out_root, keep_jobs
        self.jobs: dict[str, dict] = {}
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._work, daemon=True, name="inference-worker").start()

    def folder(self, kind: str) -> tuple[str, Path]:
        jid = ("gen_" if kind == "generate" else "") + new_job_id()
        d = self.out_root / jid
        d.mkdir(parents=True)
        return jid, d

    def submit(self, jid: str, kind: str, fn) -> dict:
        self.jobs[jid] = {"id": jid, "kind": kind, "state": "queued", "stage": None, "frac": None, "error": None}
        self.q.put((jid, fn))
        self._prune()
        return {"job": jid, "position": self.q.qsize()}

    def _work(self):
        while True:
            jid, fn = self.q.get()
            job = self.jobs[jid]
            job["state"] = "running"

            def progress(stage, frac=None):
                job["stage"], job["frac"] = stage, frac

            try:
                with self.engine.lock:
                    fn(progress)
                job.update(state="done", result=f"/api/results/{jid}/result.json")
            except InputError as e:
                job.update(state="error", error=str(e))
            except Exception as e:  # noqa: BLE001  (shown as an internal error; the traceback goes to the job folder)
                (self.out_root / jid / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                job.update(state="error", error=f"internal error ({type(e).__name__}); see runs/infer/{jid}/error.txt")

    def _prune(self):
        # oldest first: folders from earlier server runs (by name: a timestamp), then this run's jobs in submission
        # order (ids within one second differ only by their random suffix, so names alone do not order them)
        order = {jid: k for k, jid in enumerate(self.jobs)}
        dirs = sorted((p for p in self.out_root.iterdir() if p.is_dir() and JOB_ID.match(p.name)),
                      key=lambda p: (p.name in order, order.get(p.name, 0), p.name.removeprefix("gen_")))
        for p in dirs[:-self.keep] if len(dirs) > self.keep else []:
            if self.jobs.get(p.name, {}).get("state") not in ("queued", "running"):
                shutil.rmtree(p, ignore_errors=True)
                self.jobs.pop(p.name, None)


def create_app(engine: Engine | None = None, web_dir: Path = WEB_DIR, keep_jobs: int = 200) -> FastAPI:
    engine = engine or Engine()
    icfg = engine.icfg
    out_root = resolve(icfg["output"]["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = Jobs(engine, out_root, keep_jobs)
    app = FastAPI(title="Whistle VAE-GMM", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.engine, app.state.jobs = engine, jobs

    def examples() -> dict:
        p = web_dir / "data" / "examples.json"
        return {e["id"]: e for e in json.loads(p.read_text(encoding="utf-8"))["examples"]} if p.exists() else {}

    @app.get("/api/status")
    def status():
        return {**engine.status(), "queue": jobs.q.qsize(), "examples": len(examples())}

    @app.post("/api/reload-model")
    def reload_model():
        return engine.reload_model()

    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...), channel: int | None = Form(None), n_samples: int | None = Form(None),
                      seed: int = Form(0)):
        name = Path(file.filename or "upload").name[:120]
        ext = Path(name).suffix.lower()
        if ext not in icfg["input"]["formats"]:
            raise HTTPException(400, f"{name}: unsupported file type (use {', '.join(icfg['input']['formats'])})")
        limit = int(icfg["input"]["max_upload_mb"]) * 1024 * 1024
        jid, d = jobs.folder("analyze")
        dst, size = d / f"input{ext}", 0
        with open(dst, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    f.close()
                    shutil.rmtree(d, ignore_errors=True)
                    raise HTTPException(413, f"{name}: larger than {icfg['input']['max_upload_mb']} MB")
                f.write(chunk)
        ns = None if n_samples is None else max(0, min(int(icfg["output"]["n_samples"]) * 2, n_samples))
        return jobs.submit(jid, "analyze", lambda progress: engine.analyze(dst, d, channel, ns, seed, name, progress))

    @app.post("/api/analyze-example")
    def analyze_example(req: ExampleRequest):
        ex = examples().get(req.example)
        if ex is None:
            raise HTTPException(404, f"no example {req.example!r}")
        src = web_dir / ex["audio"]
        jid, d = jobs.folder("analyze")
        return jobs.submit(jid, "analyze", lambda progress: engine.analyze(src, d, None, req.n_samples, req.seed,
                                                                           Path(ex["audio"]).name, progress))

    @app.post("/api/generate")
    def generate(req: GenerateRequest):
        if not engine.model.status["ready"]["generate"]:
            raise HTTPException(409, engine.model.status["message"])
        jid, d = jobs.folder("generate")
        return jobs.submit(jid, "generate", lambda progress: engine.generate(d, req.n, req.seed, req.condition, req.fc_hz, progress))

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        j = jobs.jobs.get(jid)
        if j is None:
            raise HTTPException(404, "unknown job")
        return {k: v for k, v in j.items()}

    def job_dir(jid: str) -> Path:
        d = out_root / jid
        if not JOB_ID.match(jid) or not d.is_dir():
            raise HTTPException(404, "unknown job")
        return d

    @app.get("/api/results/{jid}.zip")
    def results_zip(jid: str):
        d = job_dir(jid)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(d.iterdir()):
                if p.is_file() and not p.name.startswith("input"):
                    z.write(p, f"{jid}/{p.name}")
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{jid}.zip"'})

    @app.get("/api/results/{jid}/{name}")
    def result_file(jid: str, name: str):
        d = job_dir(jid)
        if not FILE_NAME.match(name) or not (d / name).is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(d / name)

    @app.exception_handler(InputError)
    async def input_error(_, e: InputError):
        return JSONResponse({"detail": str(e)}, status_code=400)

    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def serve(host: str | None = None, port: int | None = None, device: str | None = None) -> None:
    import uvicorn

    engine = Engine(device=device)
    s = engine.icfg["server"]
    host, port = host or s["host"], int(port or s["port"])
    print(f"model: {engine.model.status['message']}")
    print(f"serving web/ and the API on http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run(create_app(engine), host=host, port=port, log_level="warning")
