"""Regression tests for docs publishing; no torch import or network required."""

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "docs_site.py"
SPEC = importlib.util.spec_from_file_location("docs_site", SCRIPT)
docs_site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docs_site)
SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture
def dirs(tmp_path):
    built = tmp_path / "html"
    (built / "docs").mkdir(parents=True)
    (built / "_static").mkdir()
    (built / "index.html").write_text("release homepage")
    (built / "docs/torchtt.html").write_text("API docs")
    (built / "_static/style.css").write_text("body { color: red; }")
    return tmp_path / "site", built


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def publish(dirs, tag="v0.5.0", sha=SHA_A, eligible=True):
    return docs_site.assemble(*dirs, "release", sha, tag, eligible)


def test_latest_updates_preserve_stable_and_all_release_files(dirs):
    site, built = dirs
    publish(dirs)
    stable = files(site / "stable")
    release = files(site / "v0.5.0")
    (site / "CNAME").write_text("docs.example.org")
    (built / "index.html").write_text("new development homepage")
    docs_site.assemble(site, built, "latest", SHA_B)
    assert files(site / "stable") == stable
    assert files(site / "v0.5.0") == release
    assert (site / "latest/index.html").read_text() == "new development homepage"
    assert (site / "CNAME").read_text() == "docs.example.org"
    assert (site / "stable/_static/style.css").is_file()
    manifest = json.loads((site / docs_site.MANIFEST).read_text())
    assert manifest["stable"] == "v0.5.0"
    assert manifest["versions"]["latest"] == {"sha": SHA_B, "label": "main (bbbbbbb)"}


def test_latest_before_release_has_working_homepage(dirs):
    site, built = dirs
    docs_site.assemble(site, built, "latest", SHA_A)
    assert 'href="latest/index.html"' in (site / "index.html").read_text()
    publish(dirs)
    assert 'href="stable/index.html"' in (site / "index.html").read_text()
    assert (site / "latest/index.html").is_file()


def test_stable_uses_version_order_and_does_not_regress(dirs):
    site, built = dirs
    publish(dirs, "v0.9.0")
    (built / "index.html").write_text("version ten")
    publish(dirs, "v0.10.0", SHA_B)
    publish(dirs, "v0.5.0")
    assert json.loads((site / docs_site.MANIFEST).read_text())["stable"] == "v0.10.0"
    assert (site / "stable/index.html").read_text() == "version ten"
    assert all((site / tag / "index.html").is_file() for tag in ("v0.5.0", "v0.9.0", "v0.10.0"))


@pytest.mark.parametrize("tag,eligible", [("v1.0.0rc1", True), ("v1.0.0.dev1", True), ("v1.0.0", False)])
def test_unreleased_or_prerelease_builds_do_not_advance_stable(dirs, tag, eligible):
    site, _ = dirs
    publish(dirs)
    manifest = publish(dirs, tag, SHA_B, eligible)
    assert manifest["stable"] == "v0.5.0"
    assert (site / tag / "index.html").is_file()


def test_only_prerelease_has_working_homepage_but_no_stable(dirs):
    site, _ = dirs
    manifest = publish(dirs, "v0.6.0rc1")
    assert manifest["stable"] is None
    assert not (site / "stable").exists()
    assert 'href="v0.6.0rc1/index.html"' in (site / "index.html").read_text()


def test_release_rerun_preserves_original_snapshot(dirs):
    site, built = dirs
    publish(dirs)
    before = files(site)
    (built / "index.html").write_text("different generated build output")
    publish(dirs)
    assert files(site) == before


def test_same_tag_different_commit_rejected_without_mutation(dirs):
    site, _ = dirs
    publish(dirs)
    before = files(site)
    with pytest.raises(ValueError, match="different commit"):
        publish(dirs, sha=SHA_B)
    assert files(site) == before


def test_snapshot_can_be_promoted_when_release_is_published(dirs):
    site, built = dirs
    publish(dirs, eligible=False)
    (built / "index.html").write_text("later build output must be ignored")
    assert publish(dirs)["stable"] == "v0.5.0"
    assert (site / "stable/index.html").read_text() == "release homepage"


@pytest.mark.parametrize("tag", ["../bad", "v0.5.0/../../bad", "/tmp/bad", "latest", "stable", "v1..2", "", None])
def test_invalid_tag_cannot_mutate_site(dirs, tag):
    site, _ = dirs
    with pytest.raises(ValueError, match="Invalid release tag"):
        publish(dirs, tag)
    assert not site.exists()


def test_unknown_existing_release_directory_is_not_overwritten(dirs):
    site, _ = dirs
    (site / "v0.5.0").mkdir(parents=True)
    (site / "v0.5.0/index.html").write_text("untracked original")
    with pytest.raises(ValueError, match="without manifest provenance"):
        publish(dirs)
    assert (site / "v0.5.0/index.html").read_text() == "untracked original"


def test_alias_tag_for_existing_version_is_rejected(dirs):
    publish(dirs)
    with pytest.raises(ValueError, match="Another tag"):
        publish(dirs, "0.5.0")


def test_missing_sphinx_output_cannot_publish(dirs):
    site, built = dirs
    (built / "docs/torchtt.html").unlink()
    with pytest.raises(ValueError, match="Incomplete Sphinx"):
        publish(dirs)
    assert not site.exists()


def test_symlink_in_build_is_rejected(dirs, tmp_path):
    site, built = dirs
    (built / "outside").symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError, match="symlinks"):
        publish(dirs)
    assert not site.exists()


def test_legacy_redirect_symlink_is_rejected(dirs, tmp_path):
    site, _ = dirs
    secret = tmp_path / "secret"
    secret.write_text("must not be changed")
    (site / "torchtt").mkdir(parents=True)
    (site / "torchtt/solvers.html").symlink_to(secret)
    with pytest.raises(ValueError, match="Unsafe legacy"):
        publish(dirs)
    assert secret.read_text() == "must not be changed"


def test_overlapping_build_and_site_are_rejected(dirs):
    _, built = dirs
    with pytest.raises(ValueError, match="overlap"):
        docs_site.assemble(built, built, "latest", SHA_A)


def test_old_pypi_module_links_target_stable_sphinx_api(dirs):
    site, _ = dirs
    publish(dirs)
    for module in docs_site.MODULES:
        anchor = "torchtt" if module == "torchtt" else f"torchtt.{module}"
        page = (site / "torchtt" / f"{module}.html").read_text()
        assert f'href="../stable/docs/torchtt.html#module-{anchor}"' in page
    assert 'href="../stable/index.html"' in (site / "torchtt/index.html").read_text()
    assert (site / ".nojekyll").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is only needed to execute browser redirect JavaScript")
@pytest.mark.parametrize("old_path,expected_path", [
    ("torchtt/index.html", "stable/index.html"),
    ("torchtt/solvers.html", "stable/docs/torchtt.html#module-torchtt.solvers"),
    ("torchtt/solvers.html?source=pypi#torchtt.solvers.amen_solve", "stable/docs/torchtt.html?source=pypi#torchtt.solvers.amen_solve"),
    ("torchtt/torchtt.html#torchtt.torchtt.TT", "stable/docs/torchtt.html#torchtt.TT"),
    ("torchtt/index.html#torchtt.TT.full", "stable/docs/torchtt.html#torchtt.TT.full"),
])
def test_redirect_executes_with_old_urls_and_fragments(dirs, old_path, expected_path):
    site, _ = dirs
    publish(dirs)
    filename = old_path.split("?")[0].split("#")[0]
    script = re.search(r"<script>(.*?)</script>", (site / filename).read_text(), re.S).group(1)
    base = "https://ion-g-ion.github.io/torchTT/"
    harness = "const location = new URL(" + json.dumps(base + old_path) + ");\n"
    harness += "location.replace = (url) => console.log(url);\nconst window = {location};\n" + script
    result = subprocess.run(["node", "-e", harness], check=True, text=True, capture_output=True)
    assert result.stdout.strip() == base + expected_path
