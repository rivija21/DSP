"""
Build the self-contained study guide from guide/template.html.

    python guide/build_guide.py

Placeholders in the template
    <!--ANNOT path::func--> <ol class="notes">..</ol> <!--/ANNOT-->
                         -> source of `func` (extracted with ast) beside its notes
    <!--FIG name.png|caption-->   -> figure from examples/figures (embedded as data URI)
    <!--EXERCISES-->              -> EXERCISES.md rendered to HTML
    <!--FILES_JSON-->, <!--ZIP_B64-->, <!--ZIPSIZE-->  -> embedded package files + zip

Outputs (guide/dist/)
    study_guide.html   complete HTML document, opens offline from disk
    artifact.html      same page without <html>/<head>/<body> (for hosting as a claude.ai artifact)
"""
import ast
import base64
import html
import io
import json
import os
import re
import zipfile

import markdown
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIST = os.path.join(HERE, "dist")
FIGS = os.path.join(ROOT, "examples", "figures")

SKIP_DIRS = {"__pycache__", ".pytest_cache", "dist", ".venv", ".git"}
SKIP_FILES = {".DS_Store"}


def func_source(relpath, name):
    src = open(os.path.join(ROOT, relpath)).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(f"{name} not found in {relpath}")


_png_cache = {}


def png_b64(path):
    """Palette-quantised PNG (much smaller, visually identical for line plots)."""
    if path not in _png_cache:
        im = Image.open(path).convert("RGB")
        q = im.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        buf = io.BytesIO()
        q.save(buf, format="PNG", optimize=True)
        _png_cache[path] = base64.b64encode(buf.getvalue()).decode()
    return _png_cache[path]


def render_annot(m):
    relpath, name = m.group(1).split("::")
    notes = m.group(2).strip()
    code = html.escape(func_source(relpath, name))
    return (f'<div class="annot"><div class="src"><div class="bar"><span>{relpath} &middot; {name}()</span>'
            f'<button class="copy" type="button">Copy</button></div>'
            f'<pre><code class="language-python">{code}</code></pre></div>{notes}</div>')


def render_fig(m):
    name, caption = m.group(1).split("|", 1)
    b64 = png_b64(os.path.join(FIGS, name.strip()))
    alt = re.sub("<[^>]+>", "", caption)[:160]
    return (f'<figure><div class="frame"><img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}"></div>'
            f'<figcaption>{caption}</figcaption></figure>')


def render_exercises():
    md = open(os.path.join(ROOT, "EXERCISES.md")).read()
    md = re.sub(r"^# .*\n", "", md, count=1)                 # chapter heading comes from the template
    out = markdown.markdown(md, extensions=["sane_lists"])
    out = out.replace("<h2>", "<h3>").replace("</h2>", "</h3>")
    return out


def package_files():
    items = []
    for d, dirs, fs in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(fs):
            if f in SKIP_FILES or f.endswith(".pyc"):
                continue
            items.append(os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/"))
    order = {"": 0, "dspkit": 1, "tests": 2, "examples": 3, "examples/figures": 4, "guide": 5}
    items.sort(key=lambda p: (order.get(os.path.dirname(p), 9), os.path.dirname(p), p))
    return items


def build():
    tpl = open(os.path.join(HERE, "template.html")).read()
    tpl = re.sub(r"<!--ANNOT (\S+)-->(.*?)<!--/ANNOT-->", render_annot, tpl, flags=re.S)
    tpl = re.sub(r"<!--FIG (.*?)-->", render_fig, tpl, flags=re.S)
    tpl = tpl.replace("<!--EXERCISES-->", render_exercises())

    paths = package_files()
    files, zbuf = [], io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            full = os.path.join(ROOT, p)
            z.write(full, "dsp-stage1/" + p)
            if p.endswith(".png"):
                files.append({"path": p, "kind": "image", "size": os.path.getsize(full), "b64": png_b64(full)})
            else:
                text = open(full, encoding="utf-8").read()
                files.append({"path": p, "kind": "text", "size": len(text.encode()), "text": text})
    zip_bytes = zbuf.getvalue()
    # Escape every "<" so file texts containing "</script>" or "<!--" cannot end or
    # re-mode the <script> data block (HTML script-data parsing rules).
    files_json = json.dumps(files).replace("<", "\\u003c")
    # Data placeholders go in LAST and once each: the embedded file texts (this script,
    # template.html) contain the placeholder strings themselves.
    tpl = tpl.replace("<!--ZIPSIZE-->", f"{len(paths)} files, {len(zip_bytes) / 1024:.0f} KB", 1)
    tpl = tpl.replace("<!--ZIP_B64-->", base64.b64encode(zip_bytes).decode(), 1)
    head, _, tail = tpl.rpartition("<!--FILES_JSON-->")
    tpl = head + files_json + tail

    os.makedirs(DIST, exist_ok=True)
    with open(os.path.join(DIST, "artifact.html"), "w") as f:
        f.write(tpl)
    full_doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                '</head>\n<body>\n' + tpl + "\n</body>\n</html>\n")
    with open(os.path.join(DIST, "study_guide.html"), "w") as f:
        f.write(full_doc)
    with open(os.path.join(DIST, "dsp-stage1.zip"), "wb") as f:
        f.write(zip_bytes)
    page = tpl[: tpl.rfind('<script id="files-data"')]
    leftover = re.findall(r"<!--(ANNOT|FIG|EXERCISES|FILES_JSON|ZIP)", page)
    print(f"files: {len(paths)}, zip {len(zip_bytes)/1024:.0f} KB, page {len(full_doc)/1024:.0f} KB, "
          f"unreplaced placeholders: {leftover}")


if __name__ == "__main__":
    build()
