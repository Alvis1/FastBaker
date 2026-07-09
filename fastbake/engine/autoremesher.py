# AutoRemesher CLI backend: binary discovery + subprocess job with an OBJ
# round-trip through ctx.tmpdir.
#
# AutoRemesher 1.0 (2026-07) CLI:
#   autoremesher --input in.obj --output out.obj [--report r.txt]
#     --target-quads N --sharp-edge DEG --adaptivity 0..1
# The headless mode still boots Qt, so display-less Linux needs
# QT_QPA_PLATFORM=offscreen. No granular progress on stdout — progress is
# faked from elapsed time.

import glob
import os
import shutil
import subprocess
import sys
import time

import bpy
import bmesh

from .. import prefs

_COMMON_PATHS = (
    "/Applications/autoremesher.app/Contents/MacOS/autoremesher",
    "/Applications/AutoRemesher.app/Contents/MacOS/autoremesher",
    "~/Applications/autoremesher.app/Contents/MacOS/autoremesher",
    "/usr/local/bin/autoremesher",
    "/opt/autoremesher/autoremesher",
    "~/.local/bin/autoremesher",
    "C:\\Program Files\\autoremesher\\autoremesher.exe",
    "C:\\Program Files\\AutoRemesher\\autoremesher.exe",
)

_APPIMAGE_GLOBS = (
    "~/Applications/autoremesher*.AppImage",
    "~/.local/bin/autoremesher*.AppImage",
)


def find_binary():
    """AutoRemesher executable path, or None. Preference wins over PATH."""
    p = prefs.get_prefs()
    if p and p.autoremesher_path:
        path = bpy.path.abspath(p.autoremesher_path)
        if os.path.isfile(path):
            return path
    path = shutil.which("autoremesher")
    if path:
        return path
    for cand in _COMMON_PATHS:
        cand = os.path.expanduser(cand)
        if os.path.isfile(cand):
            return cand
    for pattern in _APPIMAGE_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        if hits:
            return hits[-1]
    return None


def available():
    return find_binary() is not None


# --- minimal OBJ i/o ----------------------------------------------------------

def write_obj(obj, path):
    """Write a minimal triangulated OBJ of obj's mesh (object-local coords).

    Triangulates a bmesh copy — the object's mesh is untouched. Returns the
    triangle count (used for the pseudo-progress time estimate).
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.index_update()
    with open(path, "w") as f:
        for v in bm.verts:
            f.write("v %.8f %.8f %.8f\n" % v.co[:])
        for face in bm.faces:
            f.write("f %d %d %d\n" % tuple(v.index + 1 for v in face.verts))
    tris = len(bm.faces)
    bm.free()
    return tris


def read_obj(path):
    """Parse v/f lines of an OBJ into (verts, faces).

    Faces may have 3..n indices in 'f 1 2 3' or 'f 1/1/1 ...' form — split on
    '/' and keep the first int. Negative indices are relative to the vertex
    count so far, per the OBJ spec.
    """
    verts = []
    faces = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:  # skip a truncated final line
                    verts.append((float(parts[1]), float(parts[2]),
                                  float(parts[3])))
            elif line.startswith("f "):
                idx = []
                for part in line.split()[1:]:
                    i = int(part.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                if len(idx) >= 3:
                    faces.append(tuple(idx))
    return verts, faces


# --- subprocess job -----------------------------------------------------------

class AutoRemesherJob:
    """One remesh subprocess. start(obj, ...) launches it; the stage polls
    running()/progress(); after success result(name) yields a new bpy mesh.
    All temp files live in ctx.tmpdir."""

    def __init__(self, exe, ctx):
        self.exe = exe
        self.ctx = ctx
        self.proc = None
        self.started_at = 0.0
        self.expected = 10.0
        self.in_path = os.path.join(ctx.tmpdir, "retopo_in.obj")
        self.out_path = os.path.join(ctx.tmpdir, "retopo_out.obj")
        self.report_path = os.path.join(ctx.tmpdir, "retopo_report.txt")
        self.stdout_path = os.path.join(ctx.tmpdir, "autoremesher_stdout.txt")
        self.stderr_path = os.path.join(ctx.tmpdir, "autoremesher_stderr.txt")
        self._stdout = None
        self._stderr = None

    def start(self, obj, target_quads, sharp_edge):
        # Remove stale outputs from a previous attempt (a gate Redo reuses
        # the same tmpdir) so succeeded() only sees this run's file.
        for path in (self.out_path, self.report_path):
            if os.path.isfile(path):
                os.remove(path)
        tris = write_obj(obj, self.in_path)
        self.expected = max(10.0, tris / 20000.0)
        args = [self.exe,
                "--input", self.in_path,
                "--output", self.out_path,
                "--report", self.report_path,
                "--target-quads", str(int(target_quads)),
                "--sharp-edge", "%.1f" % sharp_edge,
                "--adaptivity", "1.0"]
        env = os.environ.copy()
        if sys.platform.startswith("linux") and not env.get("QT_QPA_PLATFORM"):
            # Force offscreen in --background even when DISPLAY is set (a
            # stale value would block the Qt boot on a dead X server).
            if (bpy.app.background
                    or (not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"))):
                env["QT_QPA_PLATFORM"] = "offscreen"
        self._stdout = open(self.stdout_path, "w")
        self._stderr = open(self.stderr_path, "w")
        self.ctx.log("autoremesher: " + " ".join(args))
        self.proc = subprocess.Popen(args, stdout=self._stdout,
                                     stderr=self._stderr, env=env,
                                     cwd=self.ctx.tmpdir)
        self.started_at = time.monotonic()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def progress(self):
        elapsed = time.monotonic() - self.started_at
        return min(0.9, elapsed / self.expected)

    def timed_out(self):
        """Wall-clock watchdog: True once the process has run far beyond the
        time estimate — treat as hung, cancel() it and fall back."""
        if not self.running():
            return False
        elapsed = time.monotonic() - self.started_at
        return elapsed > max(self.expected * 60.0, 900.0)

    def succeeded(self):
        return (self.proc is not None and self.proc.returncode == 0
                and os.path.isfile(self.out_path))

    def close(self):
        for f in (self._stdout, self._stderr):
            if f and not f.closed:
                f.close()

    def cancel(self):
        if self.running():
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.close()

    def error_excerpt(self):
        """Last stdout/stderr lines, for the log after a failure."""
        lines = []
        for path in (self.stdout_path, self.stderr_path):
            try:
                with open(path, "r", errors="replace") as f:
                    lines += [ln.rstrip() for ln in f.readlines()[-5:]]
            except OSError:
                pass
        return " | ".join(ln for ln in lines if ln)

    def result(self, mesh_name):
        """Parse the output OBJ into a new bpy mesh datablock.

        Raises ValueError when the output is empty/unusable — the caller
        falls back to QuadriFlow.
        """
        verts, faces = read_obj(self.out_path)
        if not verts or not faces:
            raise ValueError("output OBJ has no geometry")
        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(verts, [], faces)
        mesh.validate()
        mesh.update()
        if len(mesh.polygons) == 0:
            bpy.data.meshes.remove(mesh)
            raise ValueError("output OBJ contained no usable faces")
        return mesh


classes = ()
