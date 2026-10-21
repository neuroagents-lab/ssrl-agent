# RoboKitchen asset provenance

These assets are the model closure required by the LEXA RoboKitchen
environment. They were copied from:

- Repository: <https://github.com/orybkin/lexa-benchmark>
- Commit: `674d56ea7454f7a3804a7ffb987930f64d6133e6`

Only files transitively referenced by the controller XML are retained, plus
licenses and these provenance records. `ASSET_MANIFEST.sha256` covers all 65
runtime XML, mesh, and texture files after the compatibility edits. Its own
SHA-256 is
`3d746c79d77faa1963223fea78a6359e0d7ec501678d6dcd57bb96683b1c00f9`.

The pristine upstream controller XML has SHA-256
`9c4be5887590b1cdb52aa306d7e75a418578f82e3e99becbe22d63dddd26c701`
and Git blob ID `f6cbf524b16b73e5987bc5a3d1de5243781a9d3c` at the source commit.
The pristine kettle defaults XML has SHA-256
`8b210794408eb18726c39eb6a5e448c379105c3078961590603e56b9978b9ab6`
and Git blob ID `dbe8e9bf6b12d734d257168b0d38f6358dc1bdc7`.
The copied D'Suite and Franka licenses have SHA-256 values
`185bf0a2c9f1c7036b5f6c4b8eca248bf1c3dfc698d87864ba586a58e28cbf4a`
and `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`,
respectively.

## Local MuJoCo 3 changes

Two XML-only compatibility changes were made; geometry, dynamics parameters,
keyframes, textures, and meshes are otherwise unchanged:

1. `kettle_asset.xml` nests its named `kettle` default below an anonymous
   top-level default. MuJoCo 3 reserves the implicit top-level class `main` and
   rejects the legacy file otherwise.
2. `franka_kitchen_ee_ctrl.xml` gives the mocap-to-link7 weld the explicit
   identity relation `relpose="0 0 0 1 0 0 0"`. This preserves the identity
   weld that the reference Python code installed at runtime.

The Python environment and controller in `envs/robokitchen.py` are a fresh
native-MuJoCo implementation; no LEXA or D4RL Python source is vendored here.
