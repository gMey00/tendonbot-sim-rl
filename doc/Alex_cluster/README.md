# Alex Cluster Documentation

[← Back to documentation index](../README.md) · [Project root](../../README.md)

This section documents everything specific to the **Alex** NHR@FAU GPGPU cluster: getting
started, verified hardware/software facts for the RTX PRO 6000 partition, and the Tier3
access application.

## Contents

| Document | Description |
|----------|-------------|
| [alex_quickstart.md](alex_quickstart.md) | Cluster conventions, helper scripts (`tools/` + `.config/env_vars.sh`), and everyday Slurm commands |
| [alex_rtx6000pro_hardware.md](alex_rtx6000pro_hardware.md) | Verified hardware and software environment of the `rtxpro6k` partition nodes |
| [alex_tier3_application_form_georg_meyer.md](alex_tier3_application_form_georg_meyer.md) | Drop-in field values for the FAU Tier3 access application to Alex |

## Related

- [Documentation index](../README.md) — all project documentation
- [Rendering scripts](../../src/tensegrity_pick/scripts/rendering/README.md) — why trained policies cannot be rendered on Alex
- `tools/train_alex.sh`, `tools/train_reach_alex.sh` — Slurm training launchers referenced in the quickstart
