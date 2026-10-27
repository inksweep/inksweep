"""``inksweep doctor``: every check on faked probe results (no GPU, no vesuvius package needed)."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from inksweep import doctor as D


def fake_probe(answers):
    """A probe answering by the first key found in the probed code (each check's code has a unique marker)."""
    def probe(code, timeout=0.0):
        for key, value in answers.items():
            if key in code:
                return value
        raise AssertionError(f"unexpected probe: {code[:80]!r}")
    return probe


def ok_json(obj):
    return 0, "some library chatter\n" + D.MARK + json.dumps(obj) + "\n", ""


TORCH, CV2, VESUVIUS = "torch.version.cuda", "cv2.__version__", "build_network_from_config"


# ---------------------------------------------------------------- python and packages
def test_python_release_checks():
    assert D.check_python((3, 14, 10, "final", 0), "3.14.10", "/v/python")[0].status == "ok"
    pre = D.check_python((3, 14, 0, "alpha", 6), "3.14.0a6", "/v/python")[0]
    assert pre.status == "fail" and "pre-release" in pre.detail and "final release" in pre.fix
    old = D.check_python((3, 12, 3, "final", 0), "3.12.3", "/v/python")[0]
    assert old.status == "warn" and "3.14" in old.detail


def test_packages_check():
    versions = {"numpy": "2.5.3", "zarr": "3.4.0"}

    def version(name):
        if name not in versions:
            raise D.importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    ok = D.check_packages(("numpy", "zarr"), version=version)[0]
    assert ok.status == "ok" and "numpy 2.5.3" in ok.detail
    bad = D.check_packages(("numpy", "tifffile"), version=version)[0]
    assert bad.status == "fail" and "tifffile" in bad.detail and bad.fix == "pip install tifffile"


# ---------------------------------------------------------------- torch and CUDA
def test_torch_crash_is_reported_not_fatal():
    checks, info = D.check_torch(fake_probe({TORCH: (-11, "", "")}))
    assert info is None
    assert checks[0].status == "fail" and "SIGSEGV" in checks[0].detail and "pre-release" in checks[0].fix
    assert checks[1].name == "cuda" and checks[1].status == "skip"


def test_torch_missing():
    err = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'\n"
    checks, _ = D.check_torch(fake_probe({TORCH: (1, "", err)}))
    assert checks[0].detail == "torch is not installed" and "download.pytorch.org" in checks[0].fix


def test_torch_with_gpus():
    info = dict(torch="2.13.0+cu129", cuda_build="12.9", cuda_available=True, torchvision="0.28.0+cu129",
                devices=[dict(index=0, name="GPU A", total=80 * 2**30, free=79 * 2**30, capability="9.0"),
                         dict(index=1, name="GPU B", total=24 * 2**30, free=23 * 2**30, capability="8.9")])
    checks, got = D.check_torch(fake_probe({TORCH: ok_json(info)}))
    assert got == info
    by = {c.name: c for c in checks}
    assert by["torch"].status == "ok" and "0.28.0+cu129" in by["torch"].detail
    assert by["cuda"].status == "ok"
    assert "GPU A, 80.0 GiB (79.0 free)" in by["cuda"].detail and "GPU B" in by["cuda"].detail
    assert "GPU memory" in by["cuda"].detail                       # a card below 40 GiB points to the guideline


def test_torch_cpu_build_and_no_device():
    info = dict(torch="2.13.0+cpu", cuda_build=None, cuda_available=False, torchvision="0.28.0", devices=[])
    by = {c.name: c for c in D.check_torch(fake_probe({TORCH: ok_json(info)}))[0]}
    assert by["torch"].status == "fail" and "CPU-only" in by["torch"].detail
    assert by["cuda"].status == "fail" and "nvidia-smi" in by["cuda"].fix


def test_a_crash_after_import_torch_is_put_on_its_step():
    torch_only = dict(done=["torch"], torch="2.13.0+cu129", cuda_build="12.9", devices=[])
    gpus = dict(torch_only, done=["torch", "cuda"], cuda_available=True,
                devices=[dict(index=0, name="GPU", total=80 * 2**30, free=80 * 2**30, capability="9.0")])
    lines = lambda *infos: "".join(D.MARK + json.dumps(i) + "\n" for i in infos)      # noqa: E731

    checks, info = D.check_torch(fake_probe({TORCH: (-11, lines(torch_only), "")}))  # crashed querying the GPUs
    by = {c.name: c for c in checks}
    assert info is not None and by["torch"].status == "ok" and "not checked" in by["torch"].detail
    assert by["cuda"].status == "fail" and "querying the GPUs crashed (SIGSEGV)" in by["cuda"].detail

    checks, _ = D.check_torch(fake_probe({TORCH: (-11, lines(torch_only, gpus), "")}))  # crashed in torchvision
    by = {c.name: c for c in checks}
    assert by["torch"].status == "warn" and "crashed on import (SIGSEGV)" in by["torch"].detail
    assert by["cuda"].status == "ok"

    failed = dict(gpus, done=["torch", "cuda", "torchvision"], cuda_available=False, devices=[], torchvision="0.28.0",
                  cuda_error="RuntimeError: CUDA-capable device(s) is/are busy or unavailable")
    c = {c.name: c for c in D.check_torch(fake_probe({TORCH: ok_json(failed)}))[0]}["cuda"]
    assert c.status == "fail" and "busy or unavailable" in c.detail and "exclusive" in c.fix


def test_torchvision_mismatch_is_a_warning():
    info = dict(torch="2.13.0+cu129", cuda_build="12.9", cuda_available=True, torchvision=None,
                torchvision_error="RuntimeError: operator torchvision::nms does not exist",
                devices=[dict(index=0, name="GPU", total=80 * 2**30, free=80 * 2**30, capability="9.0")])
    t = D.check_torch(fake_probe({TORCH: ok_json(info)}))[0][0]
    assert t.status == "warn" and "torchvision::nms" in t.detail and "same index" in t.fix


# ---------------------------------------------------------------- OpenCV
def test_opencv_fips_abort_suggests_the_manylinux2014_wheel():
    c = D.check_opencv(fake_probe({CV2: (-6, "", "FIPS self test failure\n")}), fips=True)[0]
    assert c.status == "fail" and "FIPS" in c.detail and "SIGABRT" in c.detail
    assert "manylinux2014" in c.fix and D.OPENCV_WHEEL_SHA256 in c.fix and "--force-reinstall" in c.fix


def test_opencv_other_failures():
    assert D.check_opencv(fake_probe({CV2: ok_json(dict(version="5.0.0", file="/x"))}), fips=False)[0].status == "ok"
    missing = D.check_opencv(fake_probe({CV2: (1, "", "ModuleNotFoundError: No module named 'cv2'")}), fips=False)[0]
    assert missing.status == "fail" and missing.fix == "pip install opencv-python-headless"
    gl = D.check_opencv(fake_probe({CV2: (1, "", "ImportError: libGL.so.1: cannot open shared object file")}),
                        fips=False)[0]
    assert "headless" in gl.fix


def test_opencv_probe_in_a_real_subprocess():
    pytest.importorskip("cv2")
    c = D.check_opencv(fips=False)[0]                               # this environment's own cv2
    assert c.status == "ok" and "subprocess" in c.detail


# ---------------------------------------------------------------- vesuvius
def test_vesuvius_missing_or_incomplete():
    err = "ModuleNotFoundError: No module named 'vesuvius'\n"
    c = D.check_vesuvius(fake_probe({VESUVIUS: (1, "", err)}), required=True)[0]
    assert c.status == "fail" and D.VILLA_COMMIT in c.fix and "--no-deps" in c.fix and "subdirectory=vesuvius" in c.fix
    assert D.check_vesuvius(fake_probe({VESUVIUS: (1, "", err)}), required=False)[0].status == "warn"
    dep = D.check_vesuvius(fake_probe({VESUVIUS: (1, "", "No module named 'timm.layers'")}), required=True)[0]
    assert "'timm'" in dep.detail and "inksweep[villa]" in dep.fix
    old = D.check_vesuvius(fake_probe({VESUVIUS: (1, "", "No module named 'vesuvius.ink_detection'")}))[0]
    assert "another villa version" in old.detail and D.VILLA_COMMIT in old.fix


def test_vesuvius_commit(monkeypatch):
    probe = fake_probe({VESUVIUS: ok_json(dict(file="/x/vesuvius/__init__.py"))})
    monkeypatch.setattr(D, "villa_source", lambda f: dict(kind="pip from git", where="u", commit=D.VILLA_COMMIT))
    assert D.check_vesuvius(probe)[0].status == "ok"
    monkeypatch.setattr(D, "villa_source", lambda f: dict(kind="git checkout", where="/v", commit="b" * 40))
    c = D.check_vesuvius(probe)[0]
    assert c.status == "warn" and D.VILLA_COMMIT[:8] in c.detail and "bbbbbbbb" in c.detail
    monkeypatch.setattr(D, "villa_source", lambda f: dict(kind="installed", where="/s", commit=None))
    assert D.check_vesuvius(probe)[0].status == "warn"


def test_villa_source_git_checkout(tmp_path):
    root = tmp_path / "villa"
    init = root / "vesuvius" / "src" / "vesuvius" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text("")
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text(D.VILLA_COMMIT + "\n")            # detached, as setup_inksweep.sh leaves it
    src = D.villa_source(str(init))
    assert src["kind"] == "git checkout" and src["commit"] == D.VILLA_COMMIT and src["where"] == str(root.resolve())
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    assert D.villa_source(str(init))["commit"] == "a" * 40


def test_villa_source_pip_from_git(tmp_path, monkeypatch):
    init = tmp_path / "a" / "b" / "site-packages" / "vesuvius" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text("")
    record = {"url": "https://github.com/ScrollPrize/villa.git", "subdirectory": "vesuvius",
              "vcs_info": {"vcs": "git", "commit_id": D.VILLA_COMMIT, "requested_revision": D.VILLA_COMMIT}}
    monkeypatch.setattr(D, "direct_url", lambda dist: record)
    src = D.villa_source(str(init))
    assert src == dict(kind="pip from git", where=record["url"], commit=D.VILLA_COMMIT)
    monkeypatch.setattr(D, "direct_url", lambda dist: None)
    assert D.villa_source(str(init))["commit"] is None


# ---------------------------------------------------------------- checkpoints and disk
def test_checkpoint_check(tmp_path):
    from inksweep.readers.checkpoints import Checkpoint, sha256_file

    ck = tmp_path / "ck"
    good = ck / "a" / "w.pth"
    good.parent.mkdir(parents=True)
    good.write_bytes(b"weights")
    reg = {"good": Checkpoint("good", "https://example.invalid/w", sha256_file(good), "a/w.pth", 7),
           "bad": Checkpoint("bad", "https://example.invalid/b", "0" * 64, "a/w.pth", 7),
           "short": Checkpoint("short", "https://example.invalid/s", sha256_file(good), "a/w.pth", 8),
           "missing": Checkpoint("missing", "https://example.invalid/m", "0" * 64, "b/m.pth", 1000)}
    checks, missing = D.check_checkpoints({"r1": ["good"], "r2": ["bad"], "r3": ["missing"], "r4": ["short"],
                                           "r5": []}, str(ck), registry=reg)
    by = {c.detail.split(":")[0]: c for c in checks}
    assert by["r1"].status == "ok" and "sha256 ok" in by["r1"].detail
    assert by["r2"].status == "fail" and by["r2"].fix.startswith("rm ") and "inksweep fetch --readers r2" in by["r2"].fix
    assert by["r3"].status == "fail" and "missing" in by["r3"].detail and by["r3"].fix.startswith("inksweep fetch")
    assert by["r4"].status == "fail" and "8" in by["r4"].detail
    assert by["r5"].status == "skip"
    assert missing == 1000
    unhashed, _ = D.check_checkpoints({"r2": ["bad"]}, str(ck), hash_files=False, registry=reg)
    assert unhashed[0].status == "ok" and "not checked" in unhashed[0].detail


def test_disk_check(tmp_path):
    low = D.check_disk(str(tmp_path / "ck" / "new"), 2 * 2**30, str(tmp_path), 50,
                       usage=lambda p: SimpleNamespace(free=2**30))
    assert [c.status for c in low] == ["fail", "warn"] and "--no-raw" in low[1].fix
    high = D.check_disk(str(tmp_path / "ck"), 0, str(tmp_path), 50, usage=lambda p: SimpleNamespace(free=100 * 2**30))
    assert [c.status for c in high] == ["ok", "ok"]


# ---------------------------------------------------------------- report and command
def test_run_checks_skips_what_needs_torch(tmp_path):
    probe = fake_probe({TORCH: (-11, "", ""), CV2: ok_json(dict(version="5", file="/x")),
                        VESUVIUS: (1, "", "No module named 'torch'")})
    checks = D.run_checks(ckpt_dir=str(tmp_path), out_dir=str(tmp_path), probe=probe)
    by = {}
    for c in checks:
        by.setdefault(c.name, c)
    assert by["torch"].status == "fail" and by["checkpoint"].status == "skip"
    assert by["vesuvius"].status == "fail" and "torch line" in by["vesuvius"].detail
    text = D.format_checks(checks)
    assert "FAIL  torch" in text and "fix: " in text and text.splitlines()[-1].startswith("summary: ")


def test_doctor_command(monkeypatch, capsys, tmp_path):
    from inksweep import cli

    good = [D.Check("python", "ok", "3.14.10")]
    monkeypatch.setattr(D, "run_checks", lambda **kw: good)
    assert cli.main(["doctor", "--json", "--ckpt-dir", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["checks"][0]["name"] == "python" and out["checks"][0]["status"] == "ok"
    monkeypatch.setattr(D, "run_checks", lambda **kw: good + [D.Check("opencv", "fail", "aborted", "do this")])
    assert cli.main(["doctor"]) == 1
    text = capsys.readouterr().out
    assert "FAIL  opencv" in text and "fix: do this" in text


def test_command_line_does_not_import_torch_at_start():
    code = "import sys, inksweep.cli as c; c.build_parser(); print('torch' in sys.modules)"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "False"


def test_pins_match_the_readers_and_the_node_setup():
    from inksweep.readers.villa import VILLA_COMMIT

    assert D.VILLA_COMMIT == VILLA_COMMIT
    node = Path(__file__).resolve().parents[1] / "node"
    if not (node / "setup_inksweep.sh").is_file():
        pytest.skip("node/ is not in this tree")
    setup = (node / "setup_inksweep.sh").read_text()
    assert f"OPENCV_WHEEL_URL={D.OPENCV_WHEEL_URL}\n" in setup
    assert f"OPENCV_WHEEL_SHA256={D.OPENCV_WHEEL_SHA256}\n" in setup
    assert f"VILLA_COMMIT={D.VILLA_COMMIT}" in (node / "common.sh").read_text()
