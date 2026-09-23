#!/usr/bin/env python3
"""Build versioned Sphinx docs from a checkout without compiling torchTT.

Run this script from the workflow checkout, pointing --source at the exact
commit to document (which can predate this script). Install docs/requirements.txt,
CPU-only PyTorch and pandoc first; no torchTT installation is needed.
"""

from __future__ import annotations

import argparse
import html
import importlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", required=True, help="latest or a release tag")
    parser.add_argument("--sha", required=True, help="Full source commit SHA")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "conf.py").is_file() or not (source / "torchtt").is_dir():
        parser.error("--source must be a torchTT source checkout with conf.py")
    if output == source or source in output.parents:
        parser.error("--output must be outside the source checkout")
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be empty to avoid retaining stale documentation")
    if args.channel != "latest" and not re.fullmatch(r"v[0-9][A-Za-z0-9.+-]*", args.channel):
        parser.error("--channel must be latest or a version tag such as v0.5.0")
    if not re.fullmatch(r"[0-9a-f]{40,64}", args.sha):
        parser.error("--sha must be a full commit SHA")

    # Put the selected source ahead of any installed package or workflow checkout.
    sys.path.insert(0, str(source))
    os.environ["PYTHONPATH"] = str(source)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    module = importlib.import_module("torchtt")
    if Path(module.__file__).resolve() != source / "torchtt" / "__init__.py":
        raise RuntimeError("torchtt was imported from outside the selected checkout")

    from sphinx.application import Sphinx

    label = (
        f"development (main @ {args.sha[:7]})"
        if args.channel == "latest"
        else args.channel
    )

    def add_version_links(app, pagename, templatename, context, doctree):
        # Each channel occupies one directory beneath the complete Pages site.
        prefix = "../" * (pagename.count("/") + 1)
        context["body"] = (
            '<nav aria-label="Documentation versions" '
            'style="margin-bottom:1.5rem;padding:.75rem;background:#edf2f7">'
            f"<strong>torchTT {html.escape(label)}</strong><br>"
            f'<a href="{prefix}stable/index.html">Stable release</a> · '
            f'<a href="{prefix}latest/index.html">Development (main)</a>'
            "</nav>"
            + context.get("body", "")
        )

    def fix_source_links(app, docname, content):
        if docname == "docs/install":
            content[0] = content[0].replace(
                "`tests/ <tests/>`_",
                f"`tests/ <https://github.com/ion-g-ion/torchTT/tree/{args.sha}/tests>`_",
            )

    def fix_legacy_docstrings(app, what, name, obj, options, lines):
        if name == "torchtt.cat":
            lines[:] = [line for line in lines if line.strip() != "`"]
        # v0.5.0 predates two Sphinx docstring fixes. Apply presentation-only
        # corrections while documenting the exact released Python source.
        if args.channel == "v0.5.0":
            if name == "torchtt":
                lines[:] = [line for line in lines if line.strip() != ".. include:: INTRO.md"]
            elif name == "torchtt.nn.TTDensityLayer":
                lines[:] = [line.replace("|det J_T(x)|", "``|det J_T(x)|``") for line in lines]

    failures = []

    class BuildErrors(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR or getattr(record, "type", "") == "autodoc":
                failures.append(record.getMessage())

    with tempfile.TemporaryDirectory(prefix="torchtt-sphinx-doctrees-") as doctrees:
        app = Sphinx(
            srcdir=str(source),
            confdir=str(source),
            outdir=str(output),
            doctreedir=doctrees,
            buildername="html",
            confoverrides={
                "release": label,
                "version": label,
                "html_title": f"torchTT {label} documentation",
                "nbsphinx_execute": "never",
                "exclude_patterns": ["_build", "Thumbs.db", ".DS_Store", ".venv", "venv"],
            },
            freshenv=True,
        )
        # Historical style warnings remain visible, but failed imports and
        # malformed content must never result in a published documentation tree.
        error_handler = BuildErrors()
        sphinx_logger = logging.getLogger("sphinx")
        sphinx_logger.addHandler(error_handler)
        app.connect("html-page-context", add_version_links)
        app.connect("source-read", fix_source_links)
        app.connect("autodoc-process-docstring", fix_legacy_docstrings)
        try:
            app.build(force_all=True)
        finally:
            sphinx_logger.removeHandler(error_handler)
        if failures:
            raise RuntimeError("Sphinx documentation errors: " + "; ".join(failures))
        if app.statuscode:
            return app.statuscode

    # Autodoc failures are warnings to Sphinx. Do not publish a nominally
    # successful build with the central package API missing.
    api = (output / "docs" / "torchtt.html").read_text(encoding="utf-8")
    required = [
        "torchtt.TT",
        "torchtt.solvers.amen_solve",
        "module-torchtt.interpolate",
        "module-torchtt.grad",
        "module-torchtt.manifold",
        "module-torchtt.nn",
    ]
    missing = [anchor for anchor in required if f'id="{anchor}"' not in api]
    if missing:
        raise RuntimeError(f"Incomplete Sphinx API documentation: {', '.join(missing)}")
    (output / "build-info.json").write_text(
        json.dumps({"channel": args.channel, "source_sha": args.sha}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
