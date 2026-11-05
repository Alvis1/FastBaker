# SPDX-License-Identifier: GPL-3.0-or-later
#
# Subprocess entry point for the bundled AutoRemesher core wheel.
#
# Runs in a *subprocess* of Blender's own Python (never in the Blender
# process), so a native crash in the remeshing core becomes an error report
# instead of taking Blender down, and Esc/cancel genuinely stops it by
# terminating the process.
#
# Protocol (kept deliberately dumb so pipeline/retopo.py can drive it with
# plain file reads and no threads on the Blender side):
#   argv:   remesh_worker.py <in.npz> <out.npz>
#   in.npz:  vertices (float64 (n,3)), triangles (uint32 (m,3)),
#            params_json (str)
#   stdout:  repeated "PROGRESS <fraction 0..1> <status text>" lines
#            (retopo.py reads the last one from the redirected stdout file)
#   out.npz: vertices (float64 (n,3)), face_indices (uint32 flat),
#            face_sizes (uint32 per-face vertex count, 3..7, mostly 4)
#   exit:    0 success, 2 remesh returned no geometry, anything else = crash
#
# Adapted from jakeblakeley/autoremesher-blender (GPL-3.0-or-later);
# autoremesher_core itself is MIT (github.com/huxingyi/autoremesher).

import json
import sys
import threading
import time

import numpy as np


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path, allow_pickle=False)
    params = json.loads(str(data["params_json"]))

    import autoremesher_core  # provided by the bundled wheel, via PYTHONPATH

    remesher = autoremesher_core.Remesher(
        np.ascontiguousarray(data["vertices"], dtype=np.float64),
        np.ascontiguousarray(data["triangles"], dtype=np.uint32),
    )
    remesher.target_quad_count = int(params["target_quad_count"])
    remesher.island_detail_spans = int(params.get("island_detail_spans", 10))
    remesher.feature_size_factor = float(params.get("feature_size_factor", 1.0))
    remesher.scaling = float(params.get("scaling", 1.0))
    remesher.adaptivity = float(params.get("adaptivity", 1.0))
    remesher.sharp_edge_degrees = float(params.get("sharp_edge_degrees", 90.0))
    remesher.smooth_normal_degrees = float(params.get("smooth_normal_degrees", 0.0))
    remesher.hard_surface = bool(params.get("hard_surface", False))

    result = {}

    def run():
        # run() blocks with the GIL released; progress/status stay readable.
        result["ok"] = remesher.run()

    thread = threading.Thread(target=run)  # subprocess-local, not in Blender
    thread.start()
    while thread.is_alive():
        print("PROGRESS %.4f %s" % (remesher.progress, remesher.status),
              flush=True)
        time.sleep(0.25)
    thread.join()

    if not result.get("ok", False):
        print("PROGRESS 1.0 Failed", flush=True)
        print("REMESH FAILED", flush=True)
        return 2

    np.savez(out_path, vertices=remesher.vertices(),
             face_indices=remesher.face_indices(),
             face_sizes=remesher.face_sizes())
    print("PROGRESS 1.0 Done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
