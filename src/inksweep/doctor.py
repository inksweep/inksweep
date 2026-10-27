"""``inksweep doctor``: check the environment, one line per check, with a fix for every problem found.

Checks (nothing is loaded onto a GPU beyond a CUDA context, nothing is written):

* ``python``: a final CPython release (torch's wheels crash on pre-releases), 3.14 for the organisers' ``vesuvius``
  package;
* ``packages``: inksweep's own dependencies are installed (read from their metadata, nothing imported);
* ``torch`` and ``cuda``: torch imports, has a CUDA build and sees the GPUs (names, total and free memory), torchvision
  matches it;
* ``opencv``: ``import cv2`` works (OpenCV's manylinux_2_28 wheels abort the process on FIPS-mode kernels);
* ``vesuvius``: the organisers' package imports with every module the villa-family readers use, from the villa commit
  of the PHerc. 343 maps;
* ``checkpoint``: every pinned file of the requested readers is in ``--ckpt-dir``, with its size and sha256;
* ``disk``: room for the checkpoints still to download and for the outputs.

Every import that can crash the interpreter (torch, cv2, vesuvius) runs in a subprocess of the same Python, so a crash
is reported, not fatal; this module itself imports neither torch nor numpy. The exit status is 1 when a check failed.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import __version__

#: ScrollPrize/villa commit of every PHerc. 343 ink map (``inksweep.readers.villa.VILLA_COMMIT``; repeated here so that
#: the doctor never imports torch)
VILLA_COMMIT = "0fb38c45b488e192566ea83ae36369cb6db92652"
VILLA_PIP = ('pip install --no-deps "vesuvius @ git+https://github.com/ScrollPrize/villa.git@'
             f'{VILLA_COMMIT}#subdirectory=vesuvius"')
#: OpenCV's manylinux2014 wheel, as pinned by node/setup_inksweep.sh: it does not bundle the OpenSSL whose FIPS
#: self-test aborts ``import cv2`` on FIPS-mode kernels
OPENCV_WHEEL_URL = ("https://files.pythonhosted.org/packages/2b/97/"
                    "8170e9819764c47e436c130d3ff6cfb73b58f923eae9d3a03d8982b04aec/"
                    "opencv_python_headless-5.0.0.93-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl")
OPENCV_WHEEL_SHA256 = "09a872a157c1376ab922a69bbf22f9a95bcc7b658a9d8b436a60212b02b2eeb4"
OPENCV_FIPS_FIX = (f'pip install --no-deps --force-reinstall "opencv-python-headless @ {OPENCV_WHEEL_URL}'
                   f'#sha256={OPENCV_WHEEL_SHA256}"  (Linux x86_64)')
TORCH_PIP = ("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129  (or the index for your "
             "CUDA driver, see pytorch.org)")
RECIPE_READERS = ("dnative", "hecate", "ink9_s42", "ink9_s43")
CORE_PACKAGES = ("numpy", "scipy", "zarr", "numcodecs", "tifffile", "imagecodecs", "safetensors")
#: import name -> pip name, for the hints
PIP_NAMES = {"cv2": "opencv-python-headless", "yaml": "pyyaml", "PIL": "pillow", "nrrd": "pynrrd",
             "huggingface_hub": "huggingface-hub", "torch": "torch"}
MARK = "INKSWEEP_DOCTOR "
GIB = float(1 << 30)

Probe = Callable[[str, float], "tuple[int, str, str]"]


@dataclass
class Check:
    """One line of the report."""

    name: str
    status: str                  # ok | warn | fail | skip
    detail: str
    fix: str = ""


# ----------------------------------------------------------------------------------------------- helpers
def _probe(code: str, timeout: float = 300.0) -> tuple[int, str, str]:
    """Run ``code`` in a fresh interpreter of this environment: (return code, stdout, stderr).

    A negative return code is the signal that killed it (``-11`` = SIGSEGV, ``-6`` = SIGABRT). ``-P`` keeps the
    current directory off ``sys.path``, as for the ``inksweep`` command itself.
    """
    try:
        p = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True, errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout:.0f} s"
    except OSError as exc:
        return 127, "", f"cannot start {sys.executable}: {exc}"
    return p.returncode, p.stdout, p.stderr


def _parse(stdout: str) -> Optional[dict]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARK):
            try:
                return json.loads(line[len(MARK):])
            except ValueError:
                return None
    return None


def _signal_name(rc: int) -> str:
    try:
        return signal.Signals(-rc).name
    except ValueError:
        return f"signal {-rc}"


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1][:300] if lines else "(no message)"


def _missing_module(stderr: str) -> Optional[str]:
    """The module of the last ``No module named 'x.y'`` in a traceback (dotted, as printed), if any."""
    key = "No module named '"
    i = stderr.rfind(key)
    if i < 0:
        return None
    return stderr[i + len(key):].split("'", 1)[0] or None


def fips_enabled() -> bool:
    """True when the Linux kernel runs in FIPS mode."""
    try:
        return Path("/proc/sys/crypto/fips_enabled").read_text().strip() == "1"
    except OSError:
        return False


def direct_url(dist: str) -> Optional[dict]:
    """PEP 610 ``direct_url.json`` of an installed distribution (pip and uv write it for URL, VCS and editable installs)."""
    try:
        text = importlib.metadata.distribution(dist).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def git_head(root: Path) -> Optional[str]:
    """The commit checked out in the git work tree ``root`` (``.git/HEAD`` and its ref, else ``git rev-parse``)."""
    git = Path(root) / ".git"
    try:
        if git.is_dir():
            head = (git / "HEAD").read_text().strip()
            if not head.startswith("ref:"):
                return head or None
            ref = head[4:].strip()
            if (git / ref).is_file():
                return (git / ref).read_text().strip() or None
            packed = git / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split()[0]
    except OSError:
        pass
    try:
        p = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip() or None


def villa_source(vesuvius_file: str) -> dict:
    """Where the ``vesuvius`` package was installed from and its villa commit, when known.

    ``kind`` is ``"git checkout"`` (an editable install from a villa clone: the commit is the clone's HEAD), ``"pip
    from git"`` (``pip install git+...``: the commit pip recorded in ``direct_url.json``) or ``"installed"``.
    """
    f = Path(vesuvius_file).resolve()
    parents = f.parents
    root = parents[3] if len(parents) > 3 else None             # villa/vesuvius/src/vesuvius/__init__.py -> villa
    if root is not None and (root / ".git").exists():
        return dict(kind="git checkout", where=str(root), commit=git_head(root))
    d = direct_url("vesuvius") or {}
    vcs = d.get("vcs_info") or {}
    if vcs.get("commit_id"):
        return dict(kind="pip from git", where=str(d.get("url", "")), commit=str(vcs["commit_id"]))
    return dict(kind="installed", where=str(f.parent), commit=None)


# ----------------------------------------------------------------------------------------------- checks
def check_python(version_info: Optional[Sequence] = None, version: Optional[str] = None,
                 executable: Optional[str] = None) -> list[Check]:
    vi = tuple(version_info if version_info is not None else sys.version_info)
    ver = version or platform.python_version()
    exe = executable or sys.executable
    if vi[3] != "final":
        return [Check("python", "fail", f"{ver} at {exe} is a pre-release ({vi[3]})",
                      "torch's wheels can crash on pre-release interpreters: install a final release (uv >= 0.9: "
                      "uv python install 3.14; or python.org) and recreate the venv")]
    if tuple(vi[:2]) != (3, 14):
        return [Check("python", "warn", f"{ver} at {exe}: the organisers' vesuvius package (villa-family readers) "
                                        "requires Python 3.14",
                      "use a final CPython 3.14 for dnative, ink9_*, readerv2, readerv3 and d9v4c")]
    return [Check("python", "ok", f"{ver} (final) at {exe}")]


def check_packages(packages: Sequence[str] = CORE_PACKAGES,
                   version: Callable[[str], str] = importlib.metadata.version) -> list[Check]:
    found, missing = [], []
    for name in packages:
        try:
            found.append(f"{name} {version(name)}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    detail = f"inksweep {__version__}; " + ", ".join(found)
    if missing:
        return [Check("packages", "fail", detail + "; missing: " + ", ".join(missing), "pip install " + " ".join(missing))]
    return [Check("packages", "ok", detail)]


TORCH_CODE = r'''
import json
info = {"done": []}
def report(step):                       # a progress line after every step: a later crash keeps what was found
    info["done"].append(step)
    print("INKSWEEP_DOCTOR " + json.dumps(info), flush=True)
import torch
info["torch"] = torch.__version__
info["cuda_build"] = torch.version.cuda
report("torch")
info["devices"] = []
try:
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info["devices"].append(dict(index=i, name=p.name, total=int(total), free=int(free),
                                        capability="%d.%d" % (p.major, p.minor)))
except Exception as exc:
    info["cuda_error"] = "%s: %s" % (type(exc).__name__, exc)
report("cuda")
try:
    import torchvision
    info["torchvision"] = torchvision.__version__
except Exception as exc:
    info["torchvision"] = None
    info["torchvision_error"] = "%s: %s" % (type(exc).__name__, exc)
report("torchvision")
'''
TORCH_STEPS = ("torch", "cuda", "torchvision")


def check_torch(probe: Optional[Probe] = None) -> tuple[list[Check], Optional[dict]]:
    """The ``torch`` and ``cuda`` lines, and the probe's findings (``None`` if torch does not import).

    The probe reports after each step (torch imported, GPUs queried, torchvision imported), so a crash after
    ``import torch`` is put on the step that crashed rather than on torch.
    """
    rc, out, err = (probe or _probe)(TORCH_CODE, 300.0)
    info = _parse(out)                                # the last progress line; None if torch did not import
    if info is None:
        if rc < 0:
            check = Check("torch", "fail", f"`import torch` crashed ({_signal_name(rc)})",
                          "a crash on import usually means a pre-release Python (see the python line) or a broken "
                          "wheel: on a final Python, " + TORCH_PIP)
        elif _missing_module(err) == "torch":
            check = Check("torch", "fail", "torch is not installed", TORCH_PIP)
        else:
            check = Check("torch", "fail", f"`import torch` failed: {_last_line(err)}", TORCH_PIP)
        return [check, Check("cuda", "skip", "needs torch")], None
    done = info.get("done") or list(TORCH_STEPS)
    pending = [s for s in TORCH_STEPS if s not in done]
    crashed_at = pending[0] if pending and rc != 0 else None
    how = _signal_name(rc) if rc < 0 else f"exit status {rc}: {_last_line(err)}"
    checks = []
    tv = info.get("torchvision")
    if "torchvision" in done:
        tv_text = tv or "not importable"
    else:
        tv_text = f"crashed on import ({how})" if crashed_at == "torchvision" else "not checked"
    detail = f"{info.get('torch')} (CUDA build {info.get('cuda_build') or 'none'}), torchvision {tv_text}"
    if not info.get("cuda_build"):
        checks.append(Check("torch", "fail", detail + ": a CPU-only build", TORCH_PIP))
    elif crashed_at == "torchvision" or ("torchvision" in done and tv is None):
        error = info.get("torchvision_error")
        checks.append(Check("torch", "warn", detail + (f" ({error})" if error else ""),
                            "timm, which villa's model builder imports, needs a torchvision built for this torch: "
                            "install both from the same index: " + TORCH_PIP))
    else:
        checks.append(Check("torch", "ok", detail))
    devices = info.get("devices") or []
    if crashed_at == "cuda":
        checks.append(Check("cuda", "fail", f"torch imports, but querying the GPUs crashed ({how})",
                            "check nvidia-smi: a faulted GPU, or one held by another process in exclusive mode, "
                            "does this"))
    elif info.get("cuda_error"):
        checks.append(Check("cuda", "fail", f"torch imports, but querying the GPUs failed: {info['cuda_error']}",
                            "check nvidia-smi and CUDA_VISIBLE_DEVICES (a GPU in exclusive-process mode that another "
                            "job holds fails like this)"))
    elif not info.get("cuda_available") or not devices:
        checks.append(Check("cuda", "fail", "torch sees no CUDA device",
                            "check nvidia-smi and CUDA_VISIBLE_DEVICES; the driver must support CUDA "
                            f"{info.get('cuda_build') or '(a CUDA build of torch)'}, or install torch for an older "
                            "CUDA"))
    else:
        parts = [f"{d['index']}: {d['name']}, {d['total'] / GIB:.1f} GiB ({d['free'] / GIB:.1f} free), "
                 f"compute {d['capability']}" for d in devices]
        note = ""
        if min(d["total"] for d in devices) < 40 * GIB:
            note = " (below 40 GiB: see docs/guide.md, GPU memory, for --batch)"
        checks.append(Check("cuda", "ok", f"{len(devices)} GPU(s): " + "; ".join(parts) + note))
    return checks, info


CV2_CODE = ('import json, cv2\n'
            'print("INKSWEEP_DOCTOR " + json.dumps(dict(version=cv2.__version__, file=cv2.__file__)))\n')


def check_opencv(probe: Optional[Probe] = None, fips: Optional[bool] = None) -> list[Check]:
    fips = fips_enabled() if fips is None else fips
    rc, out, err = (probe or _probe)(CV2_CODE, 120.0)
    info = _parse(out) if rc == 0 else None
    if info is not None:
        return [Check("opencv", "ok", f"cv2 {info.get('version')} (imported in a subprocess)"
                                      + ("; the kernel is in FIPS mode" if fips else ""))]
    if rc < 0:
        if fips or "fips" in err.lower():
            return [Check("opencv", "fail", f"`import cv2` aborted ({_signal_name(rc)}) on a FIPS-mode kernel: this "
                                            "OpenCV wheel bundles an OpenSSL whose FIPS self-test fails",
                          "install OpenCV's manylinux2014 wheel: " + OPENCV_FIPS_FIX)]
        return [Check("opencv", "fail", f"`import cv2` crashed ({_signal_name(rc)})",
                      "pip install --force-reinstall opencv-python-headless; on a FIPS-mode kernel: " + OPENCV_FIPS_FIX)]
    if _missing_module(err) == "cv2":
        return [Check("opencv", "fail", "OpenCV is not installed", "pip install opencv-python-headless")]
    if "libGL" in err:
        return [Check("opencv", "fail", f"`import cv2` failed: {_last_line(err)}",
                      "the desktop OpenCV wheel needs libGL: pip uninstall -y opencv-python && "
                      "pip install opencv-python-headless")]
    return [Check("opencv", "fail", f"`import cv2` failed: {_last_line(err)}",
                  "pip install --force-reinstall opencv-python-headless")]


VESUVIUS_CODE = r'''
import json
import vesuvius
import vesuvius.ink_detection.config
import vesuvius.ink_detection.models.checkpoint
import vesuvius.ink_detection.models.model
import vesuvius.ink_detection.inference.inference_runtime
import vesuvius.models.build.build_network_from_config
print("INKSWEEP_DOCTOR " + json.dumps(dict(file=vesuvius.__file__)))
'''
VILLA_READERS = "dnative, ink9_*, ink9_soup42, readerv2, readerv3, d9v4c"


def check_vesuvius(probe: Optional[Probe] = None, required: bool = True) -> list[Check]:
    """The organisers' package with the modules the villa-family readers import, and its villa commit."""
    rc, out, err = (probe or _probe)(VESUVIUS_CODE, 300.0)
    info = _parse(out) if rc == 0 else None
    if info is None:
        bad = "fail" if required else "warn"
        if rc < 0:
            return [Check("vesuvius", bad, f"importing vesuvius crashed ({_signal_name(rc)})", VILLA_PIP)]
        missing = _missing_module(err)
        top = missing.split(".")[0] if missing else None
        if missing == "vesuvius":
            return [Check("vesuvius", bad, f"the organisers' vesuvius package is not installed (needed by {VILLA_READERS})",
                          VILLA_PIP)]
        if top == "vesuvius":
            return [Check("vesuvius", bad, f"this vesuvius package has no module {missing!r} (another villa version?)",
                          VILLA_PIP)]
        if top == "torch":
            return [Check("vesuvius", bad, "vesuvius needs torch (see the torch line)", TORCH_PIP)]
        if top:
            return [Check("vesuvius", bad, f"vesuvius is installed, but its dependency {top!r} is missing",
                          f'pip install "inksweep[villa]"  (or: pip install {PIP_NAMES.get(top, top)})')]
        return [Check("vesuvius", bad, f"importing vesuvius failed: {_last_line(err)}",
                      f'pip install "inksweep[villa]", then {VILLA_PIP}')]
    src = villa_source(info["file"])
    commit = src.get("commit")
    where = f"{src['kind']}, {src['where']}"
    if not commit:
        return [Check("vesuvius", "warn", f"vesuvius imports ({where}), villa commit unknown",
                      f"the PHerc. 343 maps were made with villa {VILLA_COMMIT[:8]}: {VILLA_PIP}")]
    if commit != VILLA_COMMIT:
        return [Check("vesuvius", "warn", f"vesuvius from villa {commit[:8]} ({where}); the PHerc. 343 maps were made "
                                          f"with {VILLA_COMMIT[:8]}, other commits can change the villa readers' maps",
                      VILLA_PIP)]
    return [Check("vesuvius", "ok", f"villa {commit[:8]} ({where}): the villa-family readers can be built")]


def reader_checkpoint_keys(readers: Sequence[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Pinned checkpoint keys and family of each reader (imports the reader registry, hence torch)."""
    from .runner import make_specs

    specs = make_specs(list(readers))
    return {s.name: list(s.checkpoints) for s in specs}, {s.name: s.family for s in specs}


def check_checkpoints(keys_by_reader: dict[str, list[str]], ckpt_dir: str, hash_files: bool = True,
                      registry: Optional[dict] = None,
                      sha256: Optional[Callable[[Path], str]] = None) -> tuple[list[Check], int]:
    """One line per reader: every pinned file present, of its pinned size and sha256. Also returns the bytes still to
    download (pinned sizes of the missing files)."""
    if registry is None:
        from .readers.checkpoints import REGISTRY as registry
    if sha256 is None:
        from .readers.checkpoints import sha256_file as sha256
    checks: list[Check] = []
    missing_bytes = 0
    for reader, keys in keys_by_reader.items():
        if not keys:
            checks.append(Check("checkpoint", "skip", f"{reader}: no pinned checkpoint (inksweep run --checkpoint "
                                                      f"{reader}=PATH)"))
            continue
        problems, bad, total = [], [], 0
        for key in keys:
            c = registry[key]
            path = Path(ckpt_dir) / c.filename
            if not path.is_file():
                problems.append(f"{c.filename} missing")
                missing_bytes += int(c.size or 0)
                continue
            size = path.stat().st_size
            total += size
            if c.size is not None and size != c.size:
                problems.append(f"{c.filename} has {size} bytes, pinned {c.size}")
                bad.append(str(path))
            elif hash_files and sha256(path) != c.sha256:
                problems.append(f"{c.filename}: sha256 is not the pinned {c.sha256[:12]}")
                bad.append(str(path))
        if problems:
            fix = f"inksweep fetch --readers {reader} --ckpt-dir {ckpt_dir}"
            if bad:
                fix = "rm " + " ".join(bad) + " && " + fix
            checks.append(Check("checkpoint", "fail", f"{reader}: " + "; ".join(problems), fix))
        else:
            how = "sha256 ok" if hash_files else "sizes ok (sha256 not checked)"
            checks.append(Check("checkpoint", "ok", f"{reader}: {len(keys)} file(s), {total / 2**20:.0f} MiB, {how}"))
    return checks, missing_bytes


def _existing(path: Path) -> Path:
    p = Path(path).absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def check_disk(ckpt_dir: str, missing_bytes: int, out_dir: str, min_free_gb: float,
               usage: Callable = shutil.disk_usage) -> list[Check]:
    p = _existing(Path(ckpt_dir))
    free = usage(p).free
    ok = free >= missing_bytes + GIB
    checks = [Check("disk", "ok" if ok else "fail",
                    f"checkpoints ({p}): {free / GIB:.1f} GiB free, {missing_bytes / GIB:.2f} GiB still to download",
                    "" if ok else "free some space or give another --ckpt-dir")]
    q = _existing(Path(out_dir))
    free_out = usage(q).free
    ok = free_out >= min_free_gb * GIB
    checks.append(Check("disk", "ok" if ok else "warn", f"outputs ({q}): {free_out / GIB:.1f} GiB free",
                        "" if ok else f"less than {min_free_gb:g} GiB free: a sweep writes two float32 maps per "
                                      "reader, depth and orientation (--no-raw keeps one)"))
    return checks


def run_checks(ckpt_dir: str = "checkpoints", readers: Sequence[str] = RECIPE_READERS, out_dir: str = ".",
               hash_files: bool = True, min_free_gb: float = 50.0, probe: Optional[Probe] = None) -> list[Check]:
    """Every check, in report order."""
    checks = check_python() + check_packages()
    torch_checks, torch_info = check_torch(probe)
    checks += torch_checks
    checks += check_opencv(probe)
    keys = families = None
    if torch_info is not None:                       # torch imports: the reader registry can be loaded here
        try:
            keys, families = reader_checkpoint_keys(readers)
        except Exception as exc:
            checks.append(Check("readers", "fail", f"cannot resolve readers {','.join(readers)}: {exc}",
                                "inksweep readers  (lists the built-in names)"))
    required = families is None or any(f == "villa" for f in families.values())
    checks += check_vesuvius(probe, required=required)
    missing = 0
    if keys is not None:
        found, missing = check_checkpoints(keys, ckpt_dir, hash_files)
        checks += found
    else:
        checks.append(Check("checkpoint", "skip", "needs a working torch (see above)"))
    checks += check_disk(ckpt_dir, missing, out_dir, min_free_gb)
    return checks


LABEL = {"ok": "ok", "warn": "WARN", "fail": "FAIL", "skip": "skip"}


def format_checks(checks: Sequence[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"{LABEL.get(c.status, c.status):4s}  {c.name:10s}  {c.detail}")
        if c.fix and c.status in ("warn", "fail"):
            lines.append(f"{'':4s}  {'':10s}  fix: {c.fix}")
    n = {s: sum(c.status == s for c in checks) for s in LABEL}
    lines.append(f"summary: {n['ok']} ok, {n['warn']} warnings, {n['fail']} failed, {n['skip']} skipped")
    return "\n".join(lines)
