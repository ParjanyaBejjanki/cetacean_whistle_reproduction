"""Entry point:  python whistle.py <command> ...   (adds src/ to sys.path; no installation needed)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from whistlepipe.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
