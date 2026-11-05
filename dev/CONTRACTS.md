# FastBake module contracts (v0.2)

Shared spec for all pipeline modules. The spine files are authoritative — read
them before writing code: `fastbake/pipeline/context.py` (PipelineContext +
Stage), `fastbake/pipeline/state.py` (STATE), `fastbake/pipeline/driver.py`
(how stages are consumed), `fastbake/props.py` (settings keys),
`fastbake/compat.py` (version seams), `fastbake/naming.py`.

## Hard rules
1. **Never store bpy ID references across stage boundaries or gates** — store
   names (strings) on the context; re-resolve with `ctx.obj(name)` /
   `bpy.data.images.get(...)`.
2. **No `threading`.** Long work = subprocess (list-form args only) or Cycles
   background jobs. Synchronous seconds-scale work may block inside
   `start()`.
3. **Background-mode rule:** when `bpy.app.background` is True, use blocking
   operator calls (`EXEC_DEFAULT`) instead of `INVOKE_DEFAULT` jobs; skip HUD,
   popups, `status_text_set`, `tag_redraw`.
4. Version-dependent API goes through `compat.py` ONLY (compositor tree,
   media_type, boolean solver, bake handlers). If a new seam is required,
   note it in your summary so it can be added centrally.
5. Expected failures: `raise StageError("user-readable message")`. Track every
   created datablock: `ctx.track_object/track_image/track_material` —
   rollback and redo depend on it.
6. Operators use idname prefix `fastbake.`; classes `FASTBAKE_OT_*`,
   `FASTBAKE_PT_*`; each module exposes `classes = (...)` (may be empty) —
   only modules listed in `fastbake/__init__.py::_modules()` are registered.
7. Temp files under `ctx.tmpdir` only — it is created lazily under
   `compat.extension_user_dir("tmp")` (plan §9 review compliance: temp/cache
   exclusively in the extension user dir). Never call `tempfile` anywhere
   else. Output textures to `ctx.output_dir()`, which raises `StageError`
   when the path cannot be resolved (unsaved .blend + `//`-relative path);
   there is deliberately NO temp-dir fallback, because textures the user
   cannot find are a silent failure.
8. Use `bpy.context.temp_override(...)` for operator context; assume the 3D
   viewport may not exist (background mode).
9. Python: bundled 3.11 (Blender 4.5–5.0) and 3.13 (5.1+). numpy is bundled —
   import freely. No other third-party imports.
10. **Linked IDs are ignored** wherever names are scanned for collisions or
    manifests are read: filter `id.library is None` (preflight's existing-name
    and prior-result scans, retopo's mesh-name list, material.py lookups). A
    linked library's objects are not ours to collide with or overwrite.
11. **`cancel(ctx)` is non-blocking.** No `time.sleep` loops, no freeing of
    images or other datablocks a running job may still write to: remove
    handlers, mark cancelled, return. The driver is what waits.
12. Every artifact name comes from `ctx.tex_name(key)` / `ctx.mat_name()` —
    never from `naming.tex_name(...)` / `naming.mat_name(...)` directly, so
    the naming templates are honoured uniformly.

## Stage protocol (context.py)
- `stage_id`, `label` (progress caption), `gate` (bool), `adjust_props`
  (tuple of `FastBakeAdvanced` prop names for the Adjust dialog).
- `start(ctx)` — begin (or complete) the work.
- `poll(ctx) -> float | None` — progress while running, None when done.
  Synchronous stages: do everything in `start()`, inherit `poll`.
- `cancel(ctx)` — kill subprocesses etc. (best effort).
- `summary(ctx) -> str` — one line for the review gate.
- The driver sets `stage._snapshot = ctx.snapshot_counts()` before `start`
  and uses `ctx.rollback_to(snapshot)` for the Redo button.

## PipelineContext fields (context.py is authoritative)
`settings` (plain dict: quads, tex_size(str), roughness_to_alpha,
review_stages, advanced{output_dir, margin_px, ssaa(str), normal_y, engine,
merge_detail, model_type, sharp_edge, protect_view, naming_style,
naming_template, material_template, lm_size, lm_samples, lm_gamma,
lm_denoise}), `src_names`, `active_name`, `base`, `run`,
`work_name`, `result_name`, `images{key->img name}`, `files{key->abspath}`,
`material_name`, `created_*`, `src_visibility{name->(hv,hr)}`, `flags`,
`notices`, `report_lines`, `tmpdir` (lazy property), `undo_hooks`.

Methods: `obj(name)`, `log`, `notice`, `track_*`, `snapshot_counts`,
`rollback_to`, `tmpdir_if_created()`, `output_path()` (absolute dir or None),
`output_dir(create=True)` (raises `StageError` instead of returning None),
`tex_name(key)`, `mat_name()`.

`flags` keys used across modules: `source_uses_alpha` (bool),
`has_metallic` (bool), `already_baked` (str message, set by preflight),
`force` (bool, set by re-bake confirm), `engine_used` (str),
`actual_quads` (int), `lightmap` (dict of the dialog values),
`uv_stats` (dict: terminals, seam_edges, fold_seams, islands, stretch, relief,
sector_fix, overlap_fix, flipped, overlaps, bad_faces, qa, fallback, kept —
`fallback` is True only when the DELIVERED UVs are Smart UV Project's, `kept`
is "solver" or "fallback", `bad_faces` = flipped + overlapping or None when
the gate could not count them).

## Per-stage assignments
- **preflight.py** — `PreflightStage` (stage_id `preflight`, gate False):
  validate selection was already done in `make_context`; here: material
  analysis (alpha/metallic flags + notice when alpha overrides packing),
  output dir check, save `ctx.src_visibility`. Also `ApplyTransformsStage`
  (stage_id `transforms`, gate False): apply rotation+scale on source
  originals (make single-user when mesh data is shared; add notice), location
  untouched. Module also provides
  `make_context(context) -> (PipelineContext|None, error_str|None)`:
  collects selected MESH objects, active name, builds settings snapshot
  (props.settings_to_dict), computes `ctx.base` (naming.base_name of active),
  `ctx.run` (naming.next_run over bpy images+materials+output dir file stems),
  and sets `ctx.flags['already_baked']` message when an unchanged prior
  result exists (check `obj["fastbake"]` manifests of scene objects vs a
  settings+source hash; count existing textures for the message). And
  `LightmapPreflightStage`: active object must be a mesh with UVs.
- **shell.py** — `extract_outer_shell(ctx, work, voxel, probe_percent,
  max_bridged_percent, scale)`: merge mode `OUTER_SHELL` (settings advanced
  `merge_mode`, `shell_probe`, `shell_max_bridged`). Called by
  `MergeStage._outer_shell` from `_build_scaffold` on the voxel scaffold, in
  the stage's INTERNAL space (`scale` converts reported distances back to
  world units). Morphological closing by the probe radius (Geometry Nodes SDF
  grid offset when `gn_available()` and the band is ≤ 16 voxels, else Displace
  + voxel remesh) with a re-mesh between dilate and erode so sealed voids turn
  solid; drops enclosed components (ray parity) and slivers; measures the
  bridged area fraction against a BVH of the sources; raises `StageError` above
  the limit, else records `ctx.flags['outer_shell']` and a notice. Synchronous.
- **merge.py** — `MergeStage` (stage_id `merge`, gate True, adjust_props
  ('merge_detail', 'merge_mode', 'shell_probe', 'shell_max_bridged')):
  snapshot originals via evaluated depsgraph
  (`object_eval.to_mesh()`), join copies into one work object named
  `ctx.base + "_fbwork"`; diagnose (boundary edges, non-manifold, normals);
  repair small defects on the copies before the path choice
  (`_repair_openings`, limits in sizes of a provisional assume-closed voxel):
  weld coincident boundary vertices (`SEAM_WELD_FRACTION` x diag), delete
  wire edges and loose vertices, delete open flaps up to `FLAP_VOXELS`^2 in
  area that hang off non-manifold edges, fill boundary loops up to
  `HOLE_FILL_VOXELS` around whose surrounding surface beats the
  isoperimetric bound (`HOLE_SURFACE_RATIO`); then split the connected parts
  still open into `<name>_open_fbtmp` objects and re-run the census, so only
  those are solidified and only their area counts twice. Solidify offset:
  inward (against outward normals) when boundary_length^2/area <
  `MOSTLY_CLOSED_RATIO` and the signed volume is real, else centred;
  fast path boolean self-union (compat.boolean_solver), robust path
  solidify-open-parts + voxel remesh (`mesh.remesh_voxel_size`, operator
  `bpy.ops.object.voxel_remesh` with temp_override); loose-parts cleanup
  (voxel mode has NO remove-disconnected — do it manually, drop parts with
  bbox < 2% of main); validate; retry voxel*0.7 then *1.5; set
  `ctx.work_name`. Single clean manifold source → may skip (still duplicate,
  set work_name). Voxel default diag/300 (LOW 150 / MED 300 / HIGH 600).
- **retopo.py: smooth result** — `_install_result` smooth-shades every
  result (`Mesh.shade_smooth()`, which also clears sharp edges) before the
  UV and bake stages: the tangent-space normal map is baked against the
  low-poly's shading normals, and both engines return flat faces.
- **retopo.py: AutoRemesher input density and target bias** — the
  out-of-process engines can only place quads where the input has vertices,
  so before an AutoRemesher launch a scaffold with fewer than
  `ASYNC_MIN_TRIS_PER_QUAD` (8) triangles per target quad is re-voxelised to
  ~`ASYNC_TARGET_TRIS_PER_QUAD` (30) per quad (`_densify_for_async`, capped
  by `MAX_RETOPO_INPUT_FACES`); QuadriFlow input is left untouched. The core
  wheel also asks for `engine/core.TARGET_BIAS` (1.3x) quads because the
  1.0.0 core lands at 65–85 % of its request; `_retry_target` still reasons
  in the user's target. The first async result is kept across the corrective
  retry and the retry is adopted only when it lands closer (`_closer_of`); if
  the retry fails the first result is installed instead of falling back to
  QuadriFlow. Every engine result goes through `_finite_or_trimmed`:
  QuadriFlow intermittently returns NaN positions for tiny disconnected
  pieces; up to `NONFINITE_DROP_FRACTION` (2 %) of the vertices are dropped
  with a notice, anything wider is treated as a failed attempt.
- **retopo.py + engine/autoremesher.py** — `RetopoStage` (stage_id `retopo`,
  gate True, adjust_props ('engine','model_type','sharp_edge')): async.
  Before dispatching to either engine, cap input density
  (`_cap_input_density`, ceiling `MAX_RETOPO_INPUT_FACES` = 300k tris): both
  QuadriFlow (field solve) and AutoRemesher (OBJ round-trip) blow up on
  raw multi-million-poly input, so collapse-decimate + weld first (notice on
  reduction, best-effort — logs and continues at full res on failure).
  Engine dispatch order (AUTO/AUTOREMESHER): (1) bundled `autoremesher_core`
  wheel via `engine/core.py` (`core_available()`, `CoreRemeshJob` — writes
  vertices/triangles to an .npz, runs `engine/remesh_worker.py` in a
  subprocess of Blender's own Python, streams `PROGRESS <frac> <status>` to a
  redirected stdout file that `progress()` tails, reads back an ngon-aware
  vertices/face_indices/face_sizes .npz; no external binary, no Qt,
  crash-isolated); (2) external AutoRemesher binary (`engine/autoremesher.py`,
  `find_binary()` / prefs override); (3) QuadriFlow. `CoreRemeshJob` is
  duck-compatible with `AutoRemesherJob` (same running/progress/timed_out/
  succeeded/close/cancel/result/proc surface) so `poll()` drives either. The
  wheel is CI-built (`.github/workflows/build-wheels.yml`) from the pinned MIT
  core and bundled per-platform by `dev/package_extension.py`; absent it, the
  add-on still works via (2)/(3).
  engine/autoremesher.py: `find_binary()` (prefs.autoremesher_path, then
  shutil.which('autoremesher'), then common install paths incl.
  /Applications/autoremesher.app/Contents/MacOS/autoremesher), `available()`,
  `AutoRemesherJob` class: writes triangulated OBJ of work mesh to
  ctx.tmpdir (own minimal writer), `Popen([exe, '--input', ..., '--output',
  ..., '--report', ..., '--target-quads', str(q), '--sharp-edge', ...,
  '--adaptivity', '1.0'])`, `poll()` (time-based pseudo-progress capped 0.9),
  `cancel()` (terminate), `result()` parses output OBJ (v/f lines, quads)
  into a new mesh. RetopoStage: choose engine per settings+availability;
  QuadriFlow fallback = `bpy.ops.object.quadriflow_remesh(target_faces=quads,
  mode='FACES', use_mesh_symmetry=False)` on a duplicate of work mesh
  (blocking, fine). On AutoRemesher failure → log + notice + QuadriFlow.
  Result object named `naming.next_mesh_name(ctx.base, existing)`; copy
  scale/rotation? (originals had transforms applied — result inherits work
  mesh transform); `ctx.result_name` set; hide work object (it is tracked,
  will be removed by material stage cleanup — actually: remove work object
  here after success, keep tracked). Set flags engine_used, actual_quads.
  `summary` mentions engine + quad count.
- **uvs.py** — `UVStage` (stage_id `uv`, gate True, adjust_props
  ('protect_view',)): the seam solver per plan §5 stage 3. Pure
  bmesh+numpy+mathutils. Steps: (1) per-vertex angle-deficit curvature;
  terminals = top extremes (count scales ~ sqrt(faces)/2, min 4); (2)
  per-vertex AO via BVHTree.FromBMesh hemisphere sampling (~24 rays); (3)
  edge cost = length * exp(-2*(1-AO)) * concavity_factor (convex dihedral
  expensive, concave cheap) * loop_bonus(0.7 when continuing straight);
  protect_view: multiply cost by 4 for edges whose faces face the view
  direction (region_3d.view_rotation @ -Z) — skip in background; (4) greedy
  Steiner: Dijkstra-connect terminals into one seam tree; mark edge.use_seam;
  (5) `uv.unwrap(method='MINIMUM_STRETCH', iterations=50, no_flip=True,
  margin=0.001)` via temp_override on edit mode; measure stretch (3D vs UV
  area ratio per face, normalized); if worst 1% faces stretch > 4x median →
  add relief seam path from worst vertex to nearest seam, re-unwrap (max 2
  extra rounds); (6) `uv.average_islands_scale()`, `uv.pack_islands(
  shape_method='CONCAVE', rotate=True, margin_method='FRACTION',
  margin=4/tex_size)`; (7) hard QA gate — ZERO flipped faces AND ZERO
  overlapping faces (overlaps counted in FACE select mode; faces with UV area
  < 1e-9 may be ignored as rounding slivers), else fall back to
  `uv.smart_project(angle_limit=radians(66))` + pack; when the fallback fails
  the gate too, keep whichever set has fewer flipped + overlapping faces
  (contract decision 19). Packing asks for
  `rotate_method='CARDINAL'` when the operator accepts it. UV selection state
  goes through `compat.uv_select_sync(tool_settings)`. All bpy.ops.uv.* calls need edit mode + a temp_override with a
  VIEW_3D area when available; in background use
  `bpy.ops.object.mode_set` + direct call (works headless — verify).
- **bake.py** — `BakeStage` (stage_id `bake`, gate True, adjust_props
  ('margin_px','ssaa','normal_y')): bake normal, color, roughness from
  src_names (sources selected, result active, selected_to_active). Auto ray
  distance: BVH on joined sources' eval mesh; max distance from result verts
  ×1.5 → cage_extrusion, ×2 → max_ray_distance (min epsilon 1e-4). Images:
  size = tex_size × ssaa, float_buffer for normal; colorspace: color=sRGB,
  normal/rough='Non-Color'; names via naming.tex_name(base, key, run, style)
  (SSAA images created at big size then downscaled in pack stage — create
  with final names from the start). Per material slot of EVERY source: add
  temp image node, deselect all nodes, node.select=True, nodes.active=node
  (5.0 rule); restore/cleanup after. Result object needs at least one
  material slot with the image node too?? — No: selected-to-active bakes
  into the ACTIVE object's materials; result has no material yet → give it a
  temporary bake material with the image node (this satisfies the rule).
  Normal: type='NORMAL', normal_space='TANGENT', swizzle per normal_y
  (DIRECTX → normal_g='NEG_Y'). Color: preflight flag has_metallic →
  emission-rewire (wrap each source material: connect Base Color source to
  temporary Emission, bake EMIT) else type='DIFFUSE',
  pass_filter={'COLOR'}. Roughness: type='ROUGHNESS'. Alpha: when
  `flags['source_uses_alpha']`, bake an extra map with key `"alpha"` by
  rewiring each source material's Principled *Alpha* into an Emission and
  baking EMIT into a Non-Color 8-bit image (same mechanism as the metallic
  colour rewire); pack.py folds it into the colour alpha and drops the
  datablock, so `ctx.images` has no `"alpha"` key afterwards. Add a
  Triangulate modifier (quad_method='FIXED') on the result during bake and
  KEEP it on the success path, with `ctx.notice("Result keeps a
  fixed-diagonal Triangulate modifier so exported tangents match the bake")`
  — error paths delete the result object anyway. `cancel` only removes
  handlers and marks itself cancelled (hard rule 11). cycles.samples=2, adaptive off, margin
  scaled: margin_px * (tex/1024) with margin_type='ADJACENT_FACES'.
  Interactive: run each map via INVOKE_DEFAULT + poll compat.bake_job_running,
  handlers for completion; background: EXEC_DEFAULT blocking sequential.
  poll() returns per-map fraction (map_index + job_progress_guess)/n_maps.
- **pack.py** — `PackStage` (stage_id `pack`, gate False): numpy
  foreach_get/foreach_set; downscale SSAA (box mean over 2x2) for all maps;
  roughness→alpha per rules (settings.roughness_to_alpha AND NOT
  flags.source_uses_alpha); set alpha_mode='CHANNEL_PACKED' on color; save
  all images: color PNG 8bit RGBA, normal PNG 16bit RGB, roughness (if
  separate) PNG 8bit BW/RGB — via image.save_render or
  image.filepath_raw + file_format + image.save(); use
  compat.set_image_format when going through scene render settings; simplest
  reliable: img.filepath_raw = path; img.file_format='PNG'; img.save().
  16-bit: use scene.render.image_settings copy + save_render. Record
  ctx.files. If roughness folds into alpha, remove the standalone big
  roughness image datablock after packing (keep only if separate).
- **material.py** — `MaterialStage` (stage_id `material`, gate False):
  create the `ctx.mat_name()` material (leave the Triangulate modifier in
  place; link colour *Alpha* → Principled *Alpha* and set the surface render
  method to blended/dithered — feature-detect with `hasattr` — when the
  sources used alpha), Principled: color tex (sRGB) →
  Base Color; roughness: color image alpha → Roughness when packed, else
  separate tex; normal tex (Non-Color) → Normal Map node → Normal. Assign to
  result (replace bake temp material). Write provenance ID props:
  result["fastbake"] = dict(version=1, sources=list, run=int, quads=int,
  tex=int, maps=list, settings_hash=str, date=iso); each source:
  "fastbake_source_of" append result name. Hide originals (save vis in
  ctx.src_visibility BEFORE hiding — preflight already saved it): hide via
  hide_set(True) + hide_render=True. Move originals into collection
  "FastBake_Sources" (create if missing). Remove the work scaffold object.
  Select+activate result.
- **lightmap.py** — `FASTBAKE_OT_lightmap` operator (invoke_props_dialog
  showing lm_size, lm_samples, lm_gamma, lm_denoise from scene settings
  advanced; on OK → preflight-lite + driver.launch(build_lightmap_stages));
  `LightmapStage` (stage_id `lightmap`, gate False, adjust_props
  ('lm_size','lm_samples','lm_gamma','lm_denoise')): active mesh object with
  UVs; bake type='DIFFUSE', pass_filter={'DIRECT','INDIRECT'},
  use_clear=False on float image (float_buffer=True, colorspace scene
  linear); samples=lm_samples; denoise via compositor:
  compat.ensure_compositor_tree, CompositorNodeImage → CompositorNodeDenoise
  → (no gamma node — apply gamma in numpy after) → Composite; set render res
  to image size; bpy.ops.render.render(write_still=False) then read
  'Render Result'?? — simpler proven route: render write_still=True to a
  temp EXR then load back. Snapshot+restore every scene render/compositor
  setting touched. Needs a camera: create temp camera if none. Gamma: numpy
  rgb **= (1/gamma) on pixels when gamma != 1.0. Save EXR
  (set_image_format 'OPEN_EXR', float). Name `ctx.tex_name('lightmap')`; `make_lightmap_context` sets `ctx.run`
  from the active result's `obj["fastbake"]["run"]` when present, else
  `next_run` over lightmap stems only. Neutralise the scene render region
  (`use_border`, `use_crop_to_border`) for the denoise render and restore it;
  detect bake completion through the `object_bake_complete/cancel` handlers
  plus a `_seen_running` flag (never a fixed grace period). ctx.files['lightmap']. In background skip nothing (denoise
  works headless).
- **ui/hud.py** — `enable()/disable()`: SpaceView3D.draw_handler_add
  (POST_PIXEL) drawing STATE['message'], STATE['detail'], big percent via
  blf (sizes ~28/64), themed offsets; guard double-add; no-op in background.
- **tests** — all headless, all driving the real pipeline; run with
  `blender --background --factory-startup --python-exit-code 1 --python <file>`
  (except the first, which is plain Python):
  * `tests/test_naming.py` — naming engine unit tests (`python3 tests/...`).
  * `tests/bl_smoke.py` — integration suite: build scene, full bake through
    the driver's blocking path, re-bake guard + forced re-bake, lightmap.
  * `tests/bl_torture.py` — the plan §11 M1 torture set: 24 procedural meshes
    (open shells, non-manifold, kitbash, thin sheets, debris, degenerate
    faces, tiny real-world scale, high genus). Asserts every merge scaffold is
    watertight via `merge._diagnose` and that the pipeline completes. Takes
    `-- --only <substr> --quads N`.
  * `tests/bl_uv_bench.py` — plan §11 M3: UV stage vs. Smart UV Project on six
    meshes (island count, median stretch), with hard zero-overlap and
    zero-flipped-face gates on the solver output. Measures with its own code,
    not the solver's helpers.
  * `tests/bl_uv_thin.py` — thin folded parts (contract decision 19): a hat
    with a ruffled knife-edge brim (clean solver UVs, every fold a seam; the
    overlap repair alone with fold seams off), the keep-better choice in both
    directions with UVs and seams restored, and a smooth closed part beside a
    busy one getting its own cut.
  * `tests/bl_guard_presets.py` — re-bake fingerprint (a shape-changing edit
    must clear the guard) and the factory/user preset round-trips.
  CI: `.github/workflows/tests.yml` runs `test_naming.py` and `bl_smoke.py`
  on Blender 4.5 LTS / 5.0 / 5.1 / 5.2 (official Linux tarballs) on every push
  — this is the §10 mitigation that keeps the `compat.py` seams honest. The
  heavier suites are run on demand. `.github/workflows/build-wheels.yml` is
  the 4-arch x 2-Python wheel matrix (see §9 and the file's own comments).

## Key bpy facts (verified 2026-07-09 — do not re-research)
- Bake operator args: type, pass_filter, margin, margin_type
  ('ADJACENT_FACES'/'EXTEND'), use_selected_to_active, max_ray_distance,
  cage_extrusion, normal_space, normal_r/g/b, target, use_clear, uv_layer.
- 5.0+: bake writes only to image nodes that are BOTH selected and active.
- Handler name is `object_bake_cancel` (not `..._canceled`).
- `bpy.app.is_job_running('OBJECT_BAKE')` exists 3.3+.
- Cycles can't denoise non-COMBINED bakes → compositor OIDN post-pass.
- `uv.unwrap(method='MINIMUM_STRETCH', iterations=, no_flip=)` exists 4.3+.
- `pack_islands(shape_method='CONCAVE', margin_method='FRACTION')` exists.
- Voxel remesh: no Remove-Disconnected in VOXEL mode; flat input needs
  Solidify; output is manifold.
- Boolean modifier: Operand Type COLLECTION with empty collection + EXACT
  solver = self-union. 4.5+ has solver='MANIFOLD' (manifold input only).
  5.0 renamed FAST→FLOAT (FastBake doesn't use it).
- `layout.progress(factor=, type='BAR', text=)` exists 4.0+.
- quadriflow_remesh: needs manifold input, object mode, active object;
  args include target_faces, mode='FACES', seed.


## Contract decisions (2026-09 fix pass)

These amend the per-stage assignments above; where they differ, these win.

1. **Temp files** — `ctx.tmpdir` is lazily created under
   `compat.extension_user_dir("tmp")`; stale `run_*` dirs are purged on
   startup. Modules keep treating `ctx.tmpdir` as a path string.
2. **Unsaved .blend** — the BAKE operator asks for an output folder when the
   .blend is unsaved and the output path is `//`-relative. `ctx.output_dir()`
   never falls back to a temp dir; it raises `StageError`.
3. **Cancel during a Cycles bake** — the driver never rolls back while
   `compat.bake_job_running()` is True: it keeps polling with the label
   "Waiting for Cycles to finish…" and no time cap. `BakeStage.cancel` /
   `LightmapStage.cancel` are non-blocking (hard rule 11).
4. **Source alpha** — see bake.py / pack.py / material.py above. `ctx.images`
   has no `"alpha"` key after pack; roughness stays a separate texture.
5. **Triangulate kept for export** — the result keeps the fixed-diagonal
   Triangulate modifier on success, with a notice.
6. **Naming templates** — `FastBakeAdvanced.naming_template` (default
   `"T_{base}_{map}{run}"`) and `material_template` (default `"M_{base}{run}"`)
   are in `_SNAP_ADV`. naming.py expands `{base}`, `{map}` (style suffix
   table) and `{run}` ("" for run 0, else `_NN`); `next_run` recognises names
   produced by the active template. `ctx.tex_name/mat_name` read them.
7. **Lightmap run number** — from the active result's manifest when present
   (see lightmap.py above).
8. **Cycles availability** — preflight raises
   `StageError("Cycles render engine is not available")`; `LightmapStage`
   guards the same way.
9. **compat seams** — `compat.uv_select_sync(tool_settings)` is a context
   manager forcing UV sync selection + FACE mesh select mode and restoring
   both. Every `hasattr(scene, "compositing_node_group")`, media_type and
   `CompositorNodeComposite` branch lives in compat.py, not in lightmap.py.
10. **Linked IDs** — hard rule 10.
11. **Machine-level defaults** — prefs gain `output_root` (DIR_PATH) and
    `review_default`; a `load_post` handler initialises a scene's settings
    from prefs the first time that scene is seen (hidden `initialized`
    BoolProperty), and `make_context` uses `output_root` when the scene output
    dir is still the default `//textures/`.
12. **Presets** — factory presets are written COMPLETE (full
    `settings_to_dict` of defaults + overrides), `FACTORY` is deep-copied and
    never mutated, `dict_to_settings` returns the rejected keys and the
    load/import operators report them as warnings, and preset names are
    sanitised symmetrically (the sanitised stem is the name everywhere).
13. **Failure popup** — on error, `ctx.report_lines` + traceback go to a Text
    datablock `FastBake.log`; the popup shows the message and a "Copy log"
    button (operator writing `window_manager.clipboard`).
14. **Wedge-proofing** — `FASTBAKE_OT_run_pipeline.cancel()` tears down; a
    `load_pre` handler calls `driver.shutdown()`; `launch()` resets state when
    the modal operator does not return `RUNNING_MODAL`; an undo push
    "FastBake: start" happens at launch (interactive only).
15. **Re-bake fingerprint** — `settings_hash` includes a digest of each
    source's vertex coordinates (numpy `foreach_get("co")` → sha1 of bytes)
    and its modifier stack (type + name), so shape edits invalidate the
    result. Covered by `tests/bl_guard_presets.py` A2.
16. **Merge/retopo** — `_diagnose`, `_tri_count` and `weld_short_edges` keep
    their names and signatures (retopo and `tests/bl_torture.py` use them).
    Added: per-component survival validation; scale normalisation (or an
    equivalent floor) so tiny real-world-scale inputs survive QuadriFlow's
    1e-4 edge epsilon; voxel size coupled to the quad target and lowered by a
    thin-shell estimate (plan §5 stage 1); one retry with a scaled target when
    the actual quad count misses by more than 25 % (plan §5 stage 2).
17. **UV** — hard QA gate, see uvs.py above.
18. **Lightmap** — render-region neutralisation + handler-based completion
    detection, see lightmap.py above.
19. **UV on thin parts (2026-11 fix pass)** — measured on a 700k-triangle car
    whose ~1 cm parts the retopo folds into knife edges (8 low-polys,
    QuadriFlow + AutoRemesher, 3k-8k quads: 1163 -> 4-8 flipped + overlapping
    faces summed, all on the 8.5k-face one). In `uvs.py`:
    * fold seams: every edge whose face normals are more than
      `FOLD_SEAM_ANGLE` (150 deg) apart is cut after the Steiner tree and the
      genus loops, before the sector repair. Not applied to the Smart UV
      fallback (its 66-deg grouping already splits there; measured no change);
    * every CLOSED part gets at least two terminals
      (`_closed_part_terminals`) — a seamless closed part unwraps into
      garbage (527 one-face islands on the car);
    * the overlap repair cuts one flap per cluster of bad faces (overlapping
      or flipped) per round, up to `OVERLAP_ROUNDS` / `OVERLAP_TIME_BUDGET`,
      then cuts what is left loose face by face, and always ends on the best
      state it measured; it only declines (hand-over to the fallback) above
      `OVERLAP_REPAIR_FRACTION` of the faces. The flap source keeps the
      BFS-last-reached tie-break (measured better than index order);
    * lone faces (an island of one) that SLIM still lays out flipped or as a
      bow-tie get their own planar projection, wound like the layout, before
      averaging and packing (`_relay_lone_faces`); faces that cross
      themselves even in their own plane are left alone;
    * keep-better: when both gates fail, the solver's UVs + seams
      (snapshotted before the fallback) are restored if their bad-face count
      is <= the fallback's (a tie goes to the solver), re-measured, and the
      notice says which set was kept and both verdicts. Only a COUNTED solver
      verdict competes (not "seams did not take", unusable UVs or an
      unavailable overlap test); unusable fallback UVs lose to any counted
      solver set instead of raising.
    Cost: the repair re-unwraps the whole mesh each round, so the stage took
    6-16 s instead of 6-13 s on the 3-7k-face car layouts and up to ~36 s on
    the 8.5k-face one (repair at its time cap, then the fallback).
