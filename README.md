# FastBake

One click from high-poly to a baked, game-ready quad asset.

FastBake is a Blender add-on (4.5+) that automates the sculpt-to-asset pipeline: it merges your high-poly objects, quad-remeshes them with [AutoRemesher](https://github.com/huxingyi/autoremesher), unwraps UVs, bakes textures, and builds a clean material on the result.

Supported Blender releases: **4.5 LTS, 5.0, 5.1, 5.2**. Every one of them is exercised by CI (see [Tests](#tests)).

## Installation

Two kinds of build:

- **Universal (`fastbake-0.2.1.zip` in `dist/`)** — works on every platform with **no external dependencies**. On its own it retopologises with Blender's built-in QuadriFlow; add the AutoRemesher extension (below) to get AutoRemesher.
- **Platform builds (`fastbake-<ver>[-blender45]-<platform>.zip` in `dist/`, built by CI)** — bundle the `autoremesher_core` wheel, so retopology uses AutoRemesher **with no separate install and no Qt**. Pick the zip matching your OS and Blender (the `-blender45` marker = Blender 4.5–5.0 / Python 3.11; the unmarked one = Blender 5.1+ / Python 3.13).

Platform zips are built for `windows_x64`, `linux_x64`, `macos_x64` and `macos_arm64`.

Then in Blender: **Edit → Preferences → Add-ons → Install from Disk…**, select the zip, and enable **FastBake**.

### AutoRemesher with the universal build

FastBake's AutoRemesher engine is the `autoremesher_core` module from [jakeblakeley/autoremesher-blender](https://github.com/jakeblakeley/autoremesher-blender) (FastBake's remesh worker is adapted from that add-on). Its release page has a Blender extension per platform: `autoremesher-1.0.0-<platform>.zip` for Blender 5.1+ and `autoremesher-1.0.0-blender42-<platform>.zip` for Blender 4.2–5.0. Install it the same way and keep it **enabled**: Blender then puts its `autoremesher_core` wheel in the extensions' shared Python folder, and FastBake's **Auto** engine uses it with no other setup. The line under **Advanced → Retopo engine** says which engine will run. Disabling the AutoRemesher extension removes the wheel again, and FastBake falls back to QuadriFlow.

## Retopology engine

FastBake picks the best available quad remesher automatically, in this order:

1. **AutoRemesher core** (`autoremesher_core`), bundled in a platform build or provided by the enabled AutoRemesher extension — preferred; runs in a crash-isolated subprocess of Blender's own Python with live progress.
2. **External AutoRemesher binary** — if you point *Preferences → FastBake → AutoRemesher executable* at one.
3. **QuadriFlow** — Blender's built-in fallback; always present.

AutoRemesher places quads by curvature, so small features such as mirrors survive at game budgets. QuadriFlow spreads its quads evenly and needs roughly four times the quad count to keep the same features.

Dense inputs are automatically decimated to a safe ceiling before remeshing so a multi-million-poly mesh can't exhaust memory. The low-poly is smooth-shaded before the bake, so the normal map is baked against the shading it is displayed with.

### Open and broken meshes

The merge stage needs a closed surface. Before it decides how to merge, it repairs small defects: coincident vertices along split seams are welded, wire edges and small flaps hanging off non-manifold edges are removed, and pin holes are filled. "Small" is measured in voxels of the merge grid: holes up to 8 voxels around and flaps up to 4 voxels across. The repairs touch only FastBake's internal copy, never your source.

Parts that are still open afterwards, such as single-sided glass or a scan with an open bottom, are split off and given thickness, and nothing else is. A part that has an outside grows its wall inward, so its outer surface stays where the source has it; a flat sheet grows evenly on both sides.

## Usage

1. Select your high-poly object(s) — the **active** one gives the asset its name.
2. Open the **FastBake** panel in the 3D Viewport sidebar (N-panel).
3. Set the target quad count, texture size, whether roughness is packed into the colour texture's alpha, and whether to pause for review after each stage. Everything else lives under **Advanced**: output folder, retopo engine, merge detail, surface type, sharp-edge angle, view protection, bake margin, supersampling, normal-Y convention, the suffix style and the texture/material name templates (`T_{base}_{map}{run}` and `M_{base}{run}` by default).
4. Click **B A K E** — you get a new quad mesh with baked textures and a ready-to-use material.

There is no map selection: FastBake always bakes **normal, colour and roughness** (roughness either packed into the colour alpha or written as its own texture). Colour alpha is used for the source material's own alpha when it has one, in which case roughness stays separate.

If the .blend has never been saved and the output folder is still the default `//textures/`, that path has no location on disk — FastBake asks for an output folder before the run starts instead of guessing (nothing is ever written to a temp directory you would not find again). A machine-wide default output root can be set in *Preferences → FastBake*.

### Outer shell (buildings, vehicles, anything with an interior)

Set **Advanced → Merge mode** to **Outer shell** to bake only the exterior of a complex model: a house with rooms and furniture, a car with a cabin, a scan with debris inside. FastBake voxelises the selection, seals every opening narrower than twice the **Probe radius** (a percentage of the model's bounding diagonal; 3 % by default — for a building with doors, raise it until the widest opening you want closed is smaller than twice the radius), then discards every surface that ends up enclosed. Closed interiors disappear on their own; open ones need the probe to seal their openings first.

Every sealed opening is invented surface. The completion report states how much of the hull was bridged and the widest sealed span, and if more than **Max bridged** (15 % by default) of the hull would be invented — a missing facade, a dollhouse cut-away — the run cancels with that number instead of baking a blob. On Blender builds with SDF grid nodes the closing is exact; older builds use an offset-and-remesh approximation that stays within one voxel.

### Lightmap

The **Lightmap** button bakes a denoised DIFFUSE direct+indirect lightmap for the active object into an EXR. `Gamma` defaults to `1.0`, which keeps the EXR **scene-linear** — what every engine expects. Any other value bakes that curve into the file, so the result is deliberately **not** a linear lightmap any more; only change it if your target pipeline asks for it.

### Presets

Factory presets ship with the add-on: **Props 2k** (5k quads / 2048), **Hero 4k** (20k / 4096), **Mobile 1k** (1.5k / 1024, letter suffixes) and **Unreal** (letter suffixes + DirectX green channel). Presets are plain JSON in Blender's extension user directory, so they survive upgrades and are removed on uninstall; the panel's preset menu saves, loads, deletes, imports and exports them.

## Development

- Add-on source lives in `fastbake/`.
- Internal module contracts are documented in `dev/CONTRACTS.md` — read it before touching a pipeline module.

### Tests

All test suites are headless. `tests/test_naming.py` is plain Python; everything else needs Blender, because it drives the real pipeline:

```
python3 tests/test_naming.py

blender --background --factory-startup --python-exit-code 1 --python tests/bl_smoke.py
blender --background --factory-startup --python-exit-code 1 --python tests/bl_torture.py
blender --background --factory-startup --python-exit-code 1 --python tests/bl_uv_bench.py
blender --background --factory-startup --python-exit-code 1 --python tests/bl_uv_thin.py
blender --background --factory-startup --python-exit-code 1 --python tests/bl_guard_presets.py
blender --background --factory-startup --python-exit-code 1 --python tests/bl_alpha.py
blender --background --factory-startup --python-exit-code 1 --python tests/bl_outer_shell.py
blender --background --factory-startup --python-exit-code 1 --python tests/bl_merge_repair.py
```

| suite | what it covers |
|---|---|
| `tests/test_naming.py` | naming engine: base names, run numbers, suffix styles, templates |
| `tests/bl_smoke.py` | full bake → re-bake guard → forced re-bake → lightmap on a small scene |
| `tests/bl_torture.py` | 24 procedural torture meshes (open shells, non-manifold, kitbash, thin sheets, debris, degenerate faces, tiny scale, high genus); asserts every **merge scaffold is watertight** and the whole pipeline completes |
| `tests/bl_uv_bench.py` | UV solver vs. Smart UV Project on six meshes: island count, median stretch, and hard zero-overlap / zero-flip gates |
| `tests/bl_uv_thin.py` | thin folded parts: a hat with a ruffled knife-edge brim must unwrap with zero overlaps and every fold cut; when both QA gates fail, the UV set with fewer bad faces is kept; a smooth closed part beside a busy one still gets a seam |
| `tests/bl_guard_presets.py` | re-bake fingerprint (a sculpt edit must clear the guard) and the factory/user preset round-trips |
| `tests/bl_alpha.py` | a source whose material drives Principled Alpha: the colour PNG's alpha must carry the source alpha, roughness stays separate, the material is wired for transparency |
| `tests/bl_outer_shell.py` | the **Outer shell** merge mode on procedural houses: closed interiors dissolve, a door is sealed when the probe is large enough, a cut-away facade cancels the run |
| `tests/bl_merge_repair.py` | the merge repair pass: a stray flap, a pin hole and a split seam leave a closed model unthickened and counted once; a loose sheet is thickened on its own; an open-bottomed box keeps its outer surface; the low-poly is smooth-shaded |

`bl_torture.py` and `bl_uv_bench.py` take command-line arguments after `--`, e.g. `-- --only "thin sheet"`.

CI (`.github/workflows/tests.yml`) runs the naming tests, `bl_smoke.py`, `bl_alpha.py` and `bl_outer_shell.py` on **Blender 4.5 LTS, 5.0, 5.1 and 5.2** on every push, which is what keeps the dual-API code in `fastbake/compat.py` honest. The heavier suites are run on demand.

### Demo asset

`demo/FastBake_demo.blend` is a procedural sculpt-like rock (~27k faces, two materials — bare stone and moss — plus two intersecting chunks) with the sources already selected and a camera and sun set up. Open it and press **B A K E**.

It is generated, not hand-modelled; regenerate or tweak it with:

```
blender --background --factory-startup --python demo/make_demo.py
blender --background --factory-startup --python demo/make_demo.py -- --subdivisions 5
```

### Building the bundled-AutoRemesher wheels

The `autoremesher_core` wheels are compiled from the MIT AutoRemesher core (pinned in `.github/workflows/build-wheels.yml`) — FastBake owns this pipeline so it controls the compiler flags. Upstream's own 1.0 build targets `x86-64-v3` and SIGILLs on pre-AVX2 CPUs; FastBake's do not:

- **Linux x64 and macOS x64** are built with `-march=x86-64-v2` (SSE4.2/POPCNT — Nehalem, 2008).
- **macOS arm64** has a single fixed baseline, so it carries no ISA flags (an x86 `-march` value does not even compile for AArch64).
- **Windows x64** passes no `/arch` flag: MSVC's x64 default baseline is SSE2, which is *below* `x86-64-v2` and therefore safe on everything that baseline covers. MSVC has no `/arch:SSE4.2` switch, and `/arch:AVX` would raise the requirement to Sandy Bridge.

1. Run the **build-wheels** GitHub Actions workflow (manually, or push a `v*` tag). It builds cp311 + cp312 wheels on four legs — `linux-x64`, `windows-x64`, `macos-x64` (Intel runner) and `macos-arm64` — i.e. the plan's 8 binaries per release.
2. `dev/package_extension.py` then bundles them into per-platform / per-Python `fastbake-*.zip` extension packages (downloadable as the `fastbake-extensions` artifact). It also appends the bundled wheel's dependency credits to each zip's manifest.

To bundle wheels locally once you have them in `wheelhouse/`:

```
python dev/package_extension.py --wheel-dir wheelhouse --python-tag cp311
python dev/package_extension.py --wheel-dir wheelhouse --python-tag cp312
```

The universal (QuadriFlow-only) zip is built with `blender --command extension build --source-dir fastbake --output-dir dist`.

## License

GPL-3.0-or-later. The platform builds bundle AutoRemesher (MIT) © Dust3D Project, and with it Geogram (BSD-3-Clause), Eigen (MPL-2.0) and TBB (Apache-2.0) — all GPL-compatible, and each credited in the manifest of the zip that ships them.
