# Organisation

[← Back to project root](README.md)

Project management and organisational information.

## Calendar

👉 [Open the project calendar](https://kalender.digital/8a3e2b7f89063ab12eeb)

## Project Milestones

| Phase | Status | Description |
|-------|--------|-------------|
| Environment setup | Done | Isaac Sim 5.1.0, Isaac Lab, conda environment, remote desktop |
| Robot import | Done | URDF → USD import, 3-DOF and 5-DOF configurations |
| Base scene | Done | Ground plane, conveyor, drum, lighting |
| Pick task | Done | Cube sorting with 8 green + 8 red cubes |
| Reach task | Done | End-effector pose reaching with FK-sampled targets |
| Place task | Done | Green cube placement with colour-discrimination curriculum |
| Tendon simulation | Done | Custom `TendonEffortAction`, Jacobian-transpose mapping |
| Tendon reach / place | Done | Tendon-driven variants of reach and place tasks |
| Step response test | Done | PID controller validation against Klein (2023) methodology |
| Training & evaluation | In progress | PPO training, PD vs. tendon comparison |
