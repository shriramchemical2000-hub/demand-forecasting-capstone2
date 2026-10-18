#!/usr/bin/env python
"""Point every notebook at your GitHub repository (run once after you create it).

    python scripts/set_repo_url.py https://github.com/<your-username>/demand-forecasting-capstone.git
"""
import re
import sys
from pathlib import Path


def main(url: str) -> None:
    m = re.match(r"https://github\.com/([^/]+)/([^/.]+?)(\.git)?/?$", url)
    if not m:
        raise SystemExit("Give the repository address like https://github.com/<username>/<repo>.git")
    user, repo = m.group(1), m.group(2)
    n = 0
    for nb in sorted(Path(__file__).resolve().parents[1].glob("notebooks/*.ipynb")):
        s = nb.read_text()
        s2 = re.sub(r"https://github\.com/[^\"\\\\]*?\.git", f"https://github.com/{user}/{repo}.git", s)
        s2 = re.sub(r"colab\.research\.google\.com/github/[^/]+/[^/]+/blob", f"colab.research.google.com/github/{user}/{repo}/blob", s2)
        if s2 != s:
            nb.write_text(s2)
            n += 1
    print(f"Updated {n} notebooks to {user}/{repo}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
