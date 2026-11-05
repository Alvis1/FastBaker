# In-process AutoRemesher backend: the bundled `autoremesher_core` wheel
# (nanobind module, no Qt, no external binary) driven through a subprocess of
# Blender's own Python so a native crash can never take Blender down and the
# modal driver stays responsive with real progress.
#
# CoreRemeshJob is deliberately duck-compatible with
# engine/autoremesher.py::AutoRemesherJob (same running()/progress()/
# timed_out()/succeeded()/close()/cancel()/result()/proc surface), so
# RetopoStage drives either backend with the same poll() loop.
#
# The wheel is produced by CI (.github/workflows/build-wheels.yml) from the
# MIT AutoRemesher core and bundled via blender_manifest.toml `wheels`. When
# it is absent (e.g. a wheels-less dev build, or an unsupported platform),
# core_available() returns False and retopo falls back to the external binary
# or QuadriFlow — the add-on always works, it is just faster/better with the
# wheel present.

import json
import os
import subprocess
import sys
import time

import bpy
import numpy as np

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "remesh_worker.py")

# AutoRemesher's target is approximate and biased low: a sweep of targets
# 300–8000 on 4k- and 16k-triangle spheres landed at 65–85 % of the request
# every time (measured on the 1.0.0 core, 2026-09-17). Asking for ~30 % more
# puts the first attempt inside RetopoStage's 25 % tolerance in the common
# case, so the corrective retry (a second full remesh) rarely has to run.
TARGET_BIAS = 1.3


def _core_module():
    """The imported autoremesher_core module, or None if not bundled."""
    try:
        import autoremesher_core
        return autoremesher_core
    except Exception:
        return None


def core_available():
    return _core_module() is not None


# --- mesh <-> array helpers ---------------------------------------------------

def mesh_to_arrays(me):
    """(vertices float64 (n,3), triangles uint32 (m,3)) from a bpy mesh."""
    if hasattr(me, "calc_loop_triangles"):
        me.calc_loop_triangles()
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    m = len(me.loop_triangles)
    tri = np.empty(m * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    return (co.reshape(-1, 3).astype(np.float64),
            tri.reshape(-1, 3).astype(np.uint32))


def mesh_from_result(name, vertices, face_indices, face_sizes):
    """Build a bpy mesh from the worker's ngon-aware output arrays.

    Raises ValueError when there is no usable geometry (caller falls back)."""
    face_sizes = np.asarray(face_sizes, dtype=np.int32)
    face_indices = np.asarray(face_indices, dtype=np.int32)
    if face_sizes.shape[0] == 0 or face_indices.shape[0] == 0:
        raise ValueError("remesh output has no faces")
    verts = np.asarray(vertices, dtype=np.float32)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(verts.shape[0])
    mesh.vertices.foreach_set("co", verts.ravel())
    mesh.loops.add(face_indices.shape[0])
    mesh.loops.foreach_set("vertex_index", face_indices)
    mesh.polygons.add(face_sizes.shape[0])
    loop_start = np.zeros(face_sizes.shape[0], dtype=np.int32)
    np.cumsum(face_sizes[:-1], out=loop_start[1:])
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", face_sizes)
    mesh.validate()
    mesh.update(calc_edges=True)
    if len(mesh.polygons) == 0:
        bpy.data.meshes.remove(mesh)
        raise ValueError("remesh output produced no usable faces")
    return mesh


# --- subprocess job -----------------------------------------------------------

class CoreRemeshJob:
    """One remesh subprocess over the bundled wheel. Same lifecycle surface as
    AutoRemesherJob: start(obj, ...) launches it; the stage polls
    running()/progress()/timed_out(); result(name) yields a new bpy mesh."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.proc = None
        self.started_at = 0.0
        self.expected = 10.0
        self.in_path = os.path.join(ctx.tmpdir, "retopo_in.npz")
        self.out_path = os.path.join(ctx.tmpdir, "retopo_out.npz")
        self.stdout_path = os.path.join(ctx.tmpdir, "core_stdout.txt")
        self.stderr_path = os.path.join(ctx.tmpdir, "core_stderr.txt")
        self._stdout = None
        self._stderr = None

    def start(self, obj, target_quads, sharp_edge, hard_surface=False):
        # A gate Redo reuses the tmpdir; drop a stale result so succeeded()
        # only ever sees this run's output.
        if os.path.isfile(self.out_path):
            os.remove(self.out_path)
        verts, tris = mesh_to_arrays(obj.data)
        if tris.shape[0] == 0:
            raise ValueError("work mesh has no faces to remesh")
        self.expected = max(10.0, tris.shape[0] / 20000.0)
        params = {
            "target_quad_count": int(round(target_quads * TARGET_BIAS)),
            "sharp_edge_degrees": float(sharp_edge),
            "hard_surface": bool(hard_surface),
            "adaptivity": 1.0,
            "scaling": 1.0,
            "smooth_normal_degrees": 0.0,
            "island_detail_spans": 10,
            "feature_size_factor": 1.0,
        }
        np.savez(self.in_path, vertices=verts, triangles=tris,
                 params_json=json.dumps(params))

        core = _core_module()
        env = os.environ.copy()
        if core is not None:
            core_dir = os.path.dirname(os.path.abspath(core.__file__))
            env["PYTHONPATH"] = core_dir + os.pathsep + env.get("PYTHONPATH", "")
        args = [sys.executable, WORKER, self.in_path, self.out_path]
        self._stdout = open(self.stdout_path, "w")
        # the core chatters on stderr (~50 KB/s): keep it out of the file the
        # progress reader tails every timer tick
        self._stderr = open(self.stderr_path, "w")
        self.ctx.log("core remesh: %d tris -> %d quads target"
                     % (tris.shape[0], int(target_quads)))
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            args, stdout=self._stdout, stderr=self._stderr, env=env,
            cwd=self.ctx.tmpdir, creationflags=creationflags)
        self.started_at = time.monotonic()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def progress(self):
        """Real progress parsed from the worker's last PROGRESS line, with a
        time-based floor so the bar always creeps forward before the first
        line is written."""
        floor = min(0.9, (time.monotonic() - self.started_at) / self.expected)
        frac = self._last_progress()
        return max(floor, frac) if frac is not None else floor

    def _last_progress(self):
        # only the tail is read: the modal timer calls this ten times a second
        try:
            with open(self.stdout_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                lines = f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            parts = line.rstrip().split(" ", 2)
            if parts[0] == "PROGRESS" and len(parts) >= 2:
                try:
                    return min(1.0, float(parts[1]))
                except ValueError:
                    continue
        return None

    def timed_out(self):
        if not self.running():
            return False
        elapsed = time.monotonic() - self.started_at
        return elapsed > max(self.expected * 60.0, 900.0)

    def succeeded(self):
        return (self.proc is not None and self.proc.returncode == 0
                and os.path.isfile(self.out_path))

    def close(self):
        for handle in (self._stdout, self._stderr):
            if handle and not handle.closed:
                handle.close()

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
        lines = []
        for path in (self.stderr_path, self.stdout_path):
            try:
                with open(path, "r", errors="replace") as f:
                    lines += [ln.rstrip() for ln in f.readlines()[-6:]]
            except OSError:
                pass
        return " | ".join(ln for ln in lines if ln)

    def result(self, mesh_name):
        """Parse the worker's out.npz into a new bpy mesh datablock.

        Raises ValueError when the output is empty/unusable — the caller
        falls back to the external binary or QuadriFlow."""
        data = np.load(self.out_path, allow_pickle=False)
        return mesh_from_result(mesh_name, data["vertices"],
                                data["face_indices"], data["face_sizes"])


classes = ()
