#!/usr/bin/env python3
"""Assemble a versioned GitHub Pages site without deleting older documentation.

Examples (the workflow serializes commits to the docs-site branch)::

    python scripts/docs_site.py --site site --html _build/html --channel latest --sha FULL_SHA
    python scripts/docs_site.py --site site --html _build/html --channel release \
        --tag v0.5.0 --sha FULL_SHA --stable-eligible

Only pass --stable-eligible after verifying the tag has a published, non-prerelease
GitHub release. Numbered snapshots are immutable, including on a same-SHA rebuild.
"""

import argparse
import html
import json
from pathlib import Path
import re
import shutil

from packaging.version import InvalidVersion, Version


MANIFEST = "docs-versions.json"
MODULES = ("torchtt", "cpp", "errors", "grad", "interpolate", "manifold", "nn", "solvers")


def release_version(tag):
    """Accept only PEP 440 versions which are safe, single path components."""
    if not isinstance(tag, str) or not re.fullmatch(r"v?[0-9][A-Za-z0-9._+-]*", tag):
        raise ValueError(f"Invalid release tag: {tag!r}")
    try:
        return Version(tag)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid release tag: {tag!r}") from exc


def validate_sha(sha):
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Source SHA must be a full, lowercase, 40-character Git commit SHA")


def read_manifest(site):
    path = site / MANIFEST
    if not path.exists():
        return {"schema": 1, "stable": None, "versions": {}}
    if path.is_symlink():
        raise ValueError("The documentation manifest must not be a symlink")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(data.get("versions"), dict):
        raise ValueError("Invalid documentation manifest")
    for name, entry in data["versions"].items():
        if name != "latest":
            release_version(name)
        validate_sha(entry["sha"])
        if not (site / name / "index.html").is_file() or (site / name).is_symlink():
            raise ValueError(f"Missing or unsafe documentation snapshot: {name}")
        if any(path.is_symlink() for path in (site / name).rglob("*")):
            raise ValueError(f"Documentation snapshot contains symlinks: {name}")
    stable = data.get("stable")
    if stable is not None:
        version = release_version(stable)
        if stable not in data["versions"] or version.is_prerelease or version.is_devrelease:
            raise ValueError("Invalid stable documentation version")
    return data


def redirect(target, *, api_target=None):
    """Keep query strings and old pdoc API anchors when following a redirect."""
    escaped = html.escape(target, quote=True)
    # pdoc used torchtt.torchtt.TT; Sphinx exposes the public torchtt.TT API.
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Documentation moved</title>
<link rel="canonical" href="{escaped}">
<script>
const target = {json.dumps(target)};
const apiTarget = {json.dumps(api_target)};
const hash = window.location.hash.replace(/^#torchtt\\.torchtt\\./, "#torchtt.");
const url = new URL(apiTarget && hash.startsWith("#torchtt.") ? apiTarget : target, window.location.href);
url.search = window.location.search;
if (hash) url.hash = hash;
window.location.replace(url.href);
</script>
<noscript><meta http-equiv="refresh" content="0; url={escaped}"></noscript>
</head><body><p>Continue to <a href="{escaped}">the documentation</a>.</p></body></html>
'''


def replace_tree(source, destination):
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def assemble(site, built, channel, sha, tag=None, stable_eligible=False):
    """Merge one build into SITE. Validate all inputs before modifying the site."""
    site, built = Path(site), Path(built)
    validate_sha(sha)
    if channel not in ("latest", "release"):
        raise ValueError("Channel must be latest or release")
    if channel == "latest" and (tag is not None or stable_eligible):
        raise ValueError("Latest builds cannot specify a release tag or stable eligibility")
    name = tag if channel == "release" else "latest"
    version = release_version(tag) if channel == "release" else None
    if site.is_symlink() or built.is_symlink():
        raise ValueError("Site and build directories must not be symlinks")
    if site.resolve() == built.resolve() or site.resolve() in built.resolve().parents or built.resolve() in site.resolve().parents:
        raise ValueError("Site and build directories must not overlap")
    for required in ("index.html", "docs/torchtt.html"):
        if not (built / required).is_file():
            raise ValueError(f"Incomplete Sphinx build: missing {required}")
    if any(path.is_symlink() for path in built.rglob("*")):
        raise ValueError("The built site must not contain symlinks")
    for managed in (name, "stable", "torchtt", "index.html", ".nojekyll"):
        if (site / managed).is_symlink():
            raise ValueError(f"Unsafe documentation path: {managed}")
    for filename in ("index", *MODULES):
        if (site / "torchtt" / f"{filename}.html").is_symlink():
            raise ValueError(f"Unsafe legacy documentation path: {filename}")
    manifest = read_manifest(site)
    versions = manifest["versions"]
    previous = versions.get(name)
    if channel == "release":
        if previous and previous["sha"] != sha:
            raise ValueError(f"Release {tag} is already published from a different commit")
        if not previous and (site / name).exists():
            raise ValueError(f"Release directory {tag} already exists without manifest provenance")
        if any(other != "latest" and other != tag and release_version(other) == version for other in versions):
            raise ValueError(f"Another tag already represents release version {version}")
    stable = manifest["stable"]
    advance_stable = (channel == "release" and stable_eligible
                      and not version.is_prerelease and not version.is_devrelease
                      and (stable is None or version > release_version(stable)))
    # A same-tag, same-SHA rerun keeps the originally published HTML unchanged.
    site.mkdir(parents=True, exist_ok=True)
    if channel == "latest" or previous is None:
        replace_tree(built, site / name)
        versions[name] = {"sha": sha, "label": f"main ({sha[:7]})" if channel == "latest" else tag}
    if advance_stable:
        replace_tree(site / name, site / "stable")
        manifest["stable"] = name
    elif stable and not (site / "stable" / "index.html").is_file():
        replace_tree(site / stable, site / "stable")
    default = "stable" if manifest["stable"] else "latest"
    # A prerelease-only bootstrap still needs a useful homepage.
    if default == "latest" and "latest" not in versions:
        default = name
    (site / "index.html").write_text(redirect(f"{default}/index.html"), encoding="utf-8")
    legacy = site / "torchtt"
    legacy.mkdir(exist_ok=True)
    api = f"../{default}/docs/torchtt.html"
    (legacy / "index.html").write_text(redirect(f"../{default}/index.html", api_target=api), encoding="utf-8")
    for module in MODULES:
        anchor = "torchtt" if module == "torchtt" else f"torchtt.{module}"
        (legacy / f"{module}.html").write_text(redirect(f"{api}#module-{anchor}"), encoding="utf-8")
    (site / ".nojekyll").touch()
    (site / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", type=Path, required=True, help="Checkout of the docs-site branch")
    parser.add_argument("--html", type=Path, required=True, help="Fresh Sphinx HTML directory")
    parser.add_argument("--channel", choices=("latest", "release"), required=True)
    parser.add_argument("--sha", required=True, help="Full source commit SHA")
    parser.add_argument("--tag", help="Exact release tag, required for the release channel")
    parser.add_argument("--stable-eligible", action="store_true", help="Tag is a published, non-prerelease GitHub release")
    args = parser.parse_args()
    try:
        manifest = assemble(args.site, args.html, args.channel, args.sha, args.tag, args.stable_eligible)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
