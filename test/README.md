# Tests

[← Back to project root](../README.md)

Test suite for the tensegrity simulation and RL environment code.  Uses
**pytest** with a custom `requires_isaac` marker for tests that need the full
Isaac Sim runtime.

## Test Modules

| Module | Scope | Isaac Sim? |
|---|---|---|
| `test_tendon_actuator.py` | Jacobian shape, torque mapping, affine tension conversion, pseudo-inverse, physical plausibility | No |
| `test_robot_configs.py` | `ArticulationCfg` structure: joint names, actuator groups, USD paths, effort limits, PD vs. tendon | Yes |
| `test_environments.py` | Smoke tests (`reset` / `step` loop) for all 9 registered gymnasium environments | Yes |
| `test_step_response.py` | PID gain values, tension limits, gravity compensation, Jacobian consistency with actuator module | Yes |

## Running Tests

```bash
# Pure-math tests only (no GPU required)
conda activate env_isaaclab
pytest test/test_tendon_actuator.py -v

# Full suite (Isaac Sim tests are skipped automatically when the runtime is unavailable)
pytest test/ -v
```

## Markers

- **`requires_isaac`** — Test needs the Omniverse Kit runtime (`omni.timeline`).
  Detected automatically via a subprocess probe in `conftest.py`; tests are
  skipped when the runtime is not reachable.

## Related

- [Robot configs under test](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/) — `ArticulationCfg` definitions and `TendonEffortAction`
- [Step response script](../src/tensegrity_pick/scripts/diagnostics/step_response_test.py) — PID validation tested by `test_step_response.py`
- [Source code & tasks](../src/tensegrity_pick/README.md) — extension overview