# FastBake module contracts (v0.1)

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
7. Temp files under `ctx.tmpdir` only. Output textures to `ctx.output_dir()`.
8. Use `bpy.context.temp_override(...)` for operator context; assume the 3D
   viewport may not exist (background mode).
9. Python: bundled 3.11 (Blender 4.5–5.0) and 3.13 (5.1+). numpy is bundled —
   import freely. No other third-party imports.

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
merge_detail, model_type, sharp_edge, protect_view, naming_style, lm_size,
lm_samples, lm_gamma, lm_denoise}), `src_names`, `active_name`, `base`, `run`,
`work_name`, `result_name`, `images{key->img name}`, `files{key->abspath}`,
`material_name`, `created_*`, `src_visibility{name->(hv,hr)}`, `flags`,
`notices`, `report_lines`, `tmpdir`.

`flags` keys used across modules: `source_uses_alpha` (bool),
`has_metallic` (bool), `already_baked` (str message, set by preflight),
`force` (bool, set by re-bake confirm), `engine_used` (str),
`actual_quads` (int), `lightmap` (dict of the dialog values).

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
- **merge.py** — `MergeStage` (stage_id `merge`, gate True, adjust_props
  ('merge_detail',)): snapshot originals via evaluated depsgraph
  (`object_eval.to_mesh()`), join copies into one work object named
  `ctx.base + "_fbwork"`; diagnose (boundary edges, non-manifold, normals);
  fast path boolean self-union (compat.boolean_solver), robust path
  solidify-if-open + voxel remesh (`mesh.remesh_voxel_size`, operator
  `bpy.ops.object.voxel_remesh` with temp_override); loose-parts cleanup
  (voxel mode has NO remove-disconnected — do it manually, drop parts with
  bbox < 2% of main); validate; retry voxel*0.7 then *1.5; set
  `ctx.work_name`. Single clean manifold source → may skip (still duplicate,
  set work_name). Voxel default diag/300 (LOW 150 / MED 300 / HIGH 600).
- **retopo.py + engine/autoremesher.py** — `RetopoStage` (stage_id `retopo`,
  gate True, adjust_props ('engine','model_type','sharp_edge')): async.
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
  margin=4/tex_size)`; (7) QA overlap check — fallback
  `uv.smart_project(angle_limit=radians(66))` + pack when solver output is
  invalid. All bpy.ops.uv.* calls need edit mode + a temp_override with a
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
  pass_filter={'COLOR'}. Roughness: type='ROUGHNESS'. Add a Triangulate
  modifier (quad_method='FIXED') on result during bake, remove after (keep
  note for export consistency). cycles.samples=2, adaptive off, margin
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
  create naming.mat_name(base, run) material, Principled: color tex (sRGB) →
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
  (set_image_format 'OPEN_EXR', float). Name naming.tex_name(base,'lightmap',
  run(style)). ctx.files['lightmap']. In background skip nothing (denoise
  works headless).
- **ui/hud.py** — `enable()/disable()`: SpaceView3D.draw_handler_add
  (POST_PIXEL) drawing STATE['message'], STATE['detail'], big percent via
  blf (sizes ~28/64), themed offsets; guard double-add; no-op in background.
- **tests** — `tests/bl_smoke.py`: headless integration suite (see file
  docstring scaffold): builds scene, runs pipeline via driver blocking path,
  asserts results; run with
  `blender --background --factory-startup --python tests/bl_smoke.py`.

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
