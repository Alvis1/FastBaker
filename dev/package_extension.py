#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Package FastBake into per-platform, per-Python-generation extension zips
with the bundled ``autoremesher_core`` wheels.

Blender rejects an extension that lists any wheel its Python cannot load, so a
single universal zip cannot carry both the cp311 (Blender 4.5-5.0, Python 3.11)
and the cp312 stable-ABI (Blender 5.1+, Python 3.13) wheels. We therefore emit
one zip per (platform x Python generation), each with the manifest patched to
the matching wheel and Blender version range.

Run in CI (.github/workflows/build-wheels.yml) after cibuildwheel has produced
the wheels; also runnable locally once you have wheels in a directory:

    python dev/package_extension.py --wheel-dir wheelhouse --python-tag cp311
    python dev/package_extension.py --wheel-dir wheelhouse --python-tag cp312

Wheels-less builds (QuadriFlow-only, works everywhere) are still produced the
normal way with ``blender --command extension build --source-dir fastbake``.
"""

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADDON_DIR = ROOT / "fastbake"
MANIFEST = ADDON_DIR / "blender_manifest.toml"

# cibuildwheel platform tag -> Blender platform id.
WHEEL_PLATFORM_TO_BLENDER = [
    (re.compile(r"win_amd64"), "windows-x64"),
    (re.compile(r"win_arm64"), "windows-arm64"),
    (re.compile(r"macosx_[\d_]+_arm64"), "macos-arm64"),
    (re.compile(r"macosx_[\d_]+_x86_64"), "macos-x64"),
    (re.compile(r"manylinux.*x86_64"), "linux-x64"),
]

# Each zip carries exactly one Python generation, scoped to the Blender
# releases shipping that Python. FastBake's floor is 4.5; cp311 covers up to
# (but excluding) 5.1, where Blender switches to Python 3.13.
PYTHON_TAG_BLENDER_RANGE = {
    "cp311": ("4.5.0", "5.1.0"),  # Python 3.11; version_max is exclusive
    "cp312": ("5.1.0", None),     # stable-ABI wheel, Python 3.12+ (3.13 host)
}

# Every platform the wheel CI is expected to produce (plan §9 matrix, §12
# decision 4 includes macos-x64). Missing ones are reported, not fatal: a
# release can legitimately ship after a single arch leg was re-run by hand.
EXPECTED_PLATFORMS = ("linux-x64", "macos-arm64", "macos-x64", "windows-x64")

# Third-party code that is statically linked into the autoremesher_core
# wheel and therefore only present in the wheel-bearing zips. The committed
# manifest (universal, QuadriFlow-only zip) bundles none of it, so these
# credits are injected here rather than living in blender_manifest.toml.
# All four licenses are GPL-compatible (plan §9 "License").
WHEEL_COPYRIGHT = [
    "2026 INRIA / Bruno Levy (Geogram, BSD-3-Clause)",
    "2026 Eigen contributors (Eigen, MPL-2.0)",
    "2026 UXL Foundation / Intel (oneTBB, Apache-2.0)",
]


def blender_platform(wheel_name):
    for pattern, platform in WHEEL_PLATFORM_TO_BLENDER:
        if pattern.search(wheel_name):
            return platform
    raise SystemExit("Unrecognized wheel platform tag: " + wheel_name)


def patch_copyright(manifest):
    """Append the bundled-wheel dependency credits to the copyright list.

    The extension platform requires every bundled third party to be credited;
    Geogram/Eigen/TBB ship inside the autoremesher_core wheel, so only the
    wheel-bearing zips must (and do) name them."""
    match = re.search(r"^copyright = \[.*?^\]$", manifest, re.M | re.S)
    if match is None:
        raise SystemExit("could not find the copyright list in the manifest")
    body = match.group(0)
    extra = "".join('  "%s",\n' % line for line in WHEEL_COPYRIGHT
                    if line not in body)
    if not extra:
        return manifest
    patched = body[:body.rindex("]")] + extra + "]"
    return manifest[:match.start()] + patched + manifest[match.end():]


def patch_manifest(manifest, platform, wheel_names, python_tag):
    """Insert platforms/wheels, tighten the Blender version range and credit
    the code that is linked into the bundled wheel.

    FastBake's committed manifest has no platforms/wheels keys (its default
    build is a universal QuadriFlow-only zip); we inject them here, keeping
    them top-level (before the first [table] section)."""
    version_min, version_max = PYTHON_TAG_BLENDER_RANGE[python_tag]
    wheels = ", ".join('"./wheels/%s"' % name for name in sorted(wheel_names))
    block = ['blender_version_min = "%s"' % version_min]
    if version_max is not None:
        block.append('blender_version_max = "%s"' % version_max)
    block.append('platforms = ["%s"]' % platform)
    block.append("wheels = [%s]" % wheels)
    patched, n = re.subn(r'^blender_version_min = .*$', "\n".join(block),
                         manifest, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("could not find blender_version_min in the manifest")
    return patch_copyright(patched)


def addon_files():
    for path in sorted(ADDON_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "blender_manifest.toml":
            continue
        parts = path.relative_to(ADDON_DIR).parts
        if "__pycache__" in parts or parts[0] == "wheels":
            continue
        yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", default=ROOT / "wheelhouse", type=Path)
    parser.add_argument("--out-dir", default=ROOT / "dist", type=Path)
    parser.add_argument("--python-tag", choices=sorted(PYTHON_TAG_BLENDER_RANGE),
                        default="cp312",
                        help="wheel generation to package (default: cp312)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wheels = sorted(args.wheel_dir.glob(
        "autoremesher_core-*-%s*.whl" % args.python_tag))
    if not wheels:
        raise SystemExit("No %s autoremesher_core wheels in %s"
                         % (args.python_tag, args.wheel_dir))

    manifest_template = MANIFEST.read_text()
    version = re.search(r'^version = "([^"]+)"', manifest_template, re.M).group(1)
    files = list(addon_files())

    by_platform = {}
    for wheel in wheels:
        by_platform.setdefault(blender_platform(wheel.name), []).append(wheel)

    # cp311 (Blender 4.5-5.0) zips carry a filename marker; the extensions
    # platform expects them uploaded as a separate download from the cp312 one.
    marker = "" if args.python_tag == "cp312" else "-blender45"
    for platform, platform_wheels in sorted(by_platform.items()):
        zip_path = args.out_dir / ("fastbake-%s%s-%s.zip" % (
            version, marker, platform.replace("-", "_")))
        manifest = patch_manifest(manifest_template, platform,
                                  [w.name for w in platform_wheels], args.python_tag)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("blender_manifest.toml", manifest)
            for path in files:
                zf.write(path, path.relative_to(ADDON_DIR))
            for wheel in platform_wheels:
                zf.write(wheel, "wheels/" + wheel.name)
        print("wrote %s: %s" % (zip_path.name,
                                ", ".join(w.name for w in platform_wheels)))

    missing = [p for p in EXPECTED_PLATFORMS if p not in by_platform]
    if missing:
        print("WARNING: no %s wheel found, so no zip for: %s"
              % (args.python_tag, ", ".join(missing)))


if __name__ == "__main__":
    main()
