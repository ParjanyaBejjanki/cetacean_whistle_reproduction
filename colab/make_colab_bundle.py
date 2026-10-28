"""Pack the code needed on Colab into one zip (uploaded to Google Drive; the notebook unpacks it every session).

    python colab/make_colab_bundle.py [--out PATH]

Contents: src/whistlepipe/ (without __pycache__), configs/, whistle.py, CODE_VERSION.json. CODE_VERSION.json records the
git commit, whether src/configs/whistle.py had uncommitted changes, and the SHA-256 of every packed file, so runs on
Colab (no .git there) keep their code lineage (storage.code_version reads it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "colab" / "dist" / "whistle_code.zip"))
    args = ap.parse_args()
    files = sorted(p for p in (ROOT / "src" / "whistlepipe").rglob("*.py") if "__pycache__" not in p.parts)
    files += sorted((ROOT / "configs").glob("*.toml")) + [ROOT / "whistle.py"]
    version = {"git_commit": git("rev-parse", "HEAD"),
               "git_uncommitted_code_changes": bool(git("status", "--porcelain", "--", "src", "configs", "whistle.py")),
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "files_sha256": {str(f.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, "whistle_code/" + str(f.relative_to(ROOT)).replace("\\", "/"))
        z.writestr("whistle_code/CODE_VERSION.json", json.dumps(version, indent=2))
    print(f"{out}  ({out.stat().st_size / 1e6:.2f} MB, {len(files)} files, commit {version['git_commit'][:7]}, "
          f"uncommitted changes: {version['git_uncommitted_code_changes']})")


if __name__ == "__main__":
    main()
