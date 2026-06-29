"""Fast standalone PBD-cloth stability sandbox.

Minimal scene (ground + box at belt height + the shirt). Sweeps cloth params
via CLI, applies the real apply_cloth_startup_event, settles, and logs per-step
max|pos| so we can see exactly when/if it diverges. Writes results to a file.
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--flatten", type=float, default=0.05)
p.add_argument("--stretch", type=float, default=300.0)
p.add_argument("--bend", type=float, default=0.8)
p.add_argument("--shear", type=float, default=10.0)
p.add_argument("--damping", type=float, default=0.4)
p.add_argument("--selfcoll", type=int, default=0)
p.add_argument("--iters", type=int, default=16)
p.add_argument("--matdamp", type=float, default=0.3)
p.add_argument("--vclamp", type=float, default=0.0)  # 0 = off; else clamp |vel| each step
p.add_argument("--substeps", type=int, default=1)
p.add_argument("--mass", type=float, default=0.2)
p.add_argument("--rest", type=float, default=-1.0)
p.add_argument("--contact", type=float, default=-1.0)
p.add_argument("--pcontact", type=float, default=-1.0)
p.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args()
a.headless = True
app = AppLauncher(a).app

import numpy as np, torch  # noqa
from types import SimpleNamespace  # noqa
import isaaclab.sim as sim_utils  # noqa
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa
from isaaclab.sim import SimulationContext, SimulationCfg  # noqa
from isaaclab.assets import AssetBaseCfg  # noqa
from isaaclab.utils import configclass  # noqa

from tensegrity_pick.tasks.manager_based.shared.cloth_object import (  # noqa
    ClothObject, apply_cloth_startup_event, ClothObjectCfg, PBDClothParams,
)
from tensegrity_pick.tasks.manager_based.shirt_place.shirt_place_scene_cfg import (  # noqa
    TSHIRT_USD_PATH, TSHIRT_MESH_PRIM,
)

RES = "/tmp/claude-1000/-home-robot-studentische-arbeiten/1975d042-f83b-4129-ac13-dfddbc8df8fa/scratchpad/sandbox_result.txt"
f = open(RES, "w")
def w(*x): print(*x, file=f, flush=True)

BELT = 0.8

@configclass
class MinCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))
    box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 0.4), visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8, restitution=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, BELT - 0.2)),
    )

try:
    w(f"PARAMS flatten={a.flatten} stretch={a.stretch} bend={a.bend} shear={a.shear} "
      f"damping={a.damping} selfcoll={a.selfcoll}")
    sim_cfg = SimulationCfg(dt=0.01)
    sim_cfg.physx.gpu_collision_stack_size = 2**31 - 1
    sim_cfg.physx.solver_type = 1
    sim_cfg.physx.enable_stabilization = True
    sim_cfg.physx.bounce_threshold_velocity = 0.2
    sim_cfg.physx.gpu_max_rigid_contact_count = 2**21
    sim_cfg.physx.gpu_max_rigid_patch_count = 2**19
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(MinCfg(num_envs=1, env_spacing=5.0, replicate_physics=False))

    params = PBDClothParams(
        flatten_scale=a.flatten, stretch_stiffness=a.stretch, bend_stiffness=a.bend,
        shear_stiffness=a.shear, spring_damping=a.damping,
        global_self_collision=bool(a.selfcoll),
        solver_position_iterations=a.iters, damping=a.matdamp,
        mass_kg=a.mass,
        rest_offset=(a.rest if a.rest > 0 else None),
        contact_offset=(a.contact if a.contact > 0 else None),
        particle_contact_offset=(a.pcontact if a.pcontact > 0 else None),
    )
    cfg = ClothObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shirt", usd_path=TSHIRT_USD_PATH,
        mesh_prim_path=TSHIRT_MESH_PRIM, pbd_params=params,
        init_pos=(0.0, 0.0, BELT + 0.05),
    )
    apply_cloth_startup_event(SimpleNamespace(sim=sim, scene=scene), None, cfg)
    sim.reset()
    scene.update(0.01)
    cloth = ClothObject(cfg, num_envs=1, device=sim.device)
    cloth.update()
    p0 = cloth.nodal_pos_w[0]
    w(f"spawn: thickness={float(p0[:,2].max()-p0[:,2].min()):.4f} "
      f"minZ={float(p0[:,2].min()):.4f} maxabs={float(p0.abs().max()):.3f}")

    e32 = torch.tensor([0], device=sim.device, dtype=torch.int32)
    diverged_at = None
    for i in range(a.steps):
        for _ in range(a.substeps):
            scene.write_data_to_sim(); sim.step(render=False)
        scene.update(0.01)
        if a.vclamp > 0:
            cloth.update()
            v = cloth.nodal_vel_w[0]
            sp = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            vc = v * (sp.clamp(max=a.vclamp) / sp)
            cloth._vel_flat[0] = vc.reshape(-1)
            cloth._cloth_view.set_velocities(cloth._vel_flat, e32)
        if (i+1) % 20 == 0 or i < 5:
            cloth.update()
            mab = float(cloth.nodal_pos_w[0].abs().max())
            z = cloth.nodal_pos_w[0, :, 2]
            w(f"  step {i+1:3d}: maxabs={mab:.3f} thick={float(z.max()-z.min()):.4f} "
              f"minZ={float(z.min()):.4f}")
            if mab > 100 and diverged_at is None:
                diverged_at = i+1
                w(f"  >>> DIVERGED at step {diverged_at}"); break
    cloth.update()
    z = cloth.nodal_pos_w[0, :, 2]
    pp = cloth.nodal_pos_w[0]
    xy = (pp[:,:2].max(0).values - pp[:,:2].min(0).values).tolist()
    w(f"FINAL thickness={float(z.max()-z.min()):.4f} minZ(rel belt)={float(z.min())-BELT:+.4f} "
      f"footprint=[{xy[0]:.3f},{xy[1]:.3f}] diverged={diverged_at}")
    w("OK")
except Exception as e:
    import traceback; w("EXC "+repr(e)); w(traceback.format_exc())
finally:
    f.flush(); f.close(); app.close()
