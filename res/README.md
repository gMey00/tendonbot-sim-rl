# Resources

[← Back to project root](../README.md)

Simulation assets used by the Isaac Sim / Isaac Lab environments.
All paths referenced in the Python robot configurations point into this directory.

## Contents

| Directory | Description |
|-----------|-------------|
| `Props/` | Environment props — plastic drum, T-shirt (cloth simulation demo) |
| `Scenes/` | Pre-built USD scenes (shirt physics demo, showcase) |
| `Tensegrity/` | Tensegrity robot: URDF sources, USD models, mesh files, joint configs |
| `UR10/` | UR10 reference robot assets (USD model, gripper script) |

## Notes

- Robot USD files are generated via Isaac Sim's URDF importer; see
  [`Tensegrity/README.md`](Tensegrity/README.md) for the import procedure.
- The `PROJ_ASSETS_PATH` constant in the robot config modules points to
  this directory (`res/`).

## Related

- [Robot specification](Tensegrity/README.md) — full kinematic chain, joint limits, tendon arrangement, actuation modes
- [Source code](../src/tensegrity_pick/README.md) — Isaac Lab extension that references these assets