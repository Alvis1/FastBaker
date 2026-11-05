# FastBake

One click from high-poly to a baked, game-ready quad asset. FastBake merges the selected objects, quad-remeshes them, unwraps UVs, bakes normal, colour and roughness, and builds the material. Blender 4.5 LTS, 5.0, 5.1 and 5.2.

## Install

Pick a zip from [`dist/`](dist/), then **Edit → Preferences → Add-ons → Install from Disk…**

| Zip | Use for |
|---|---|
| `fastbake-0.2.1-<platform>.zip` | Blender 5.1+, AutoRemesher built in |
| `fastbake-0.2.1-blender45-<platform>.zip` | Blender 4.5–5.0, AutoRemesher built in |
| `fastbake-0.2.1.zip` | any platform, QuadriFlow only |

Platforms: `windows_x64`, `linux_x64`, `macos_x64`, `macos_arm64`. The universal zip also uses AutoRemesher if [jakeblakeley's AutoRemesher extension](https://github.com/jakeblakeley/autoremesher-blender) is enabled. **Advanced → Retopo engine** shows which engine will run.

## Use

1. Select the high-poly object(s). The active one names the asset.
2. In the **FastBake** sidebar tab, set quad count and texture size.
3. Press **B A K E**.

- **Outer shell** (Advanced → Merge mode): bakes only the exterior of models with interiors, such as houses or cars. It cancels if more than *Max bridged* of the hull would be invented.
- **Lightmap**: bakes a denoised, scene-linear EXR lightmap for the active object.
- **Presets**: Props 2k, Hero 4k, Mobile 1k and Unreal. You can save your own.
- Small mesh defects like split seams, flaps and pin holes are repaired automatically. Parts that stay open are thickened.

## Develop

Source is in `fastbake/`, and the module contracts are in `dev/CONTRACTS.md`. Tests run headless:

```
python3 tests/test_naming.py
blender --background --factory-startup --python-exit-code 1 --python tests/bl_smoke.py
```

The other Blender suites in `tests/` run the same way. They are `bl_torture`, `bl_uv_bench`, `bl_uv_thin`, `bl_guard_presets`, `bl_alpha`, `bl_outer_shell` and `bl_merge_repair`. CI runs the naming and smoke tests on all four Blender versions on every push.

Builds:
- **Platform zips:** run the `build-wheels` workflow, or push a `v*` tag. It compiles AutoRemesher for each platform and packages the zips with `dev/package_extension.py`.
- **Universal zip:** `blender --command extension build --source-dir fastbake --output-dir dist`.
- **Demo scene:** `demo/make_demo.py` regenerates `demo/FastBake_demo.blend`.

## License

GPL-3.0-or-later. The platform zips bundle AutoRemesher (MIT, Dust3D Project), Geogram (BSD-3), Eigen (MPL-2.0) and TBB (Apache-2.0).
