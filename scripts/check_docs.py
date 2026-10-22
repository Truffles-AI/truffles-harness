"""Check local Markdown links and compile Python snippets without making API calls."""

import ast
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "examples/README.md"]
paths += [ROOT / "apps/pfm/README.md", *sorted((ROOT / "apps/pfm/docs").glob("*.md"))]
snippets = 0
for path in paths:
    text = path.read_text()
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        target = unquote(target.split("#", 1)[0])
        if not (path.parent / target).exists():
            raise SystemExit(f"Broken link in {path.relative_to(ROOT)}: {target}")
    for index, snippet in enumerate(re.findall(r"```python\n(.*?)\n```", text, re.DOTALL), 1):
        ast.parse(snippet, filename=f"{path.relative_to(ROOT)}:snippet-{index}")
        snippets += 1
print(f"Checked {len(paths)} Markdown files and {snippets} Python snippets.")
