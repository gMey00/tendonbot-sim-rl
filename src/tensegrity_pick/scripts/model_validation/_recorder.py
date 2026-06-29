"""Tiny camera-frame recorder for the cloth validation scripts.

Injects a ``CameraCfg`` into a scene config (picked up by ``InteractiveScene``
because it iterates ``cfg.__dict__``), captures RGB frames during a run, and
writes an MP4 plus a few key PNG frames.
"""

from __future__ import annotations

import os

import numpy as np


def add_record_camera(
    scene_cfg,
    prim_path: str = "/World/envs/env_0/RecordCam",
    width: int = 960,
    height: int = 540,
    name: str = "record_cam",
):
    """Attach a pinhole camera to ``scene_cfg`` and return its scene key."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    cam_cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
        offset=CameraCfg.OffsetCfg(pos=(1.0, -1.0, 1.4), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    setattr(scene_cfg, name, cam_cfg)
    return name


class FrameRecorder:
    """Collects camera frames and saves an MP4 + key PNGs.

    If a ``tracker`` callable is given (returns a world-space ``[3]`` target or
    ``None``), the camera is re-aimed at that target with ``eye_offset`` before
    each capture — a close follow-shot that keeps a moving subject framed.
    """

    def __init__(self, camera, every: int = 5, tracker=None,
                 eye_offset=(0.55, -0.55, 0.40)):
        self._cam = camera
        self._every = max(1, every)
        self._frames: list[np.ndarray] = []
        self._i = 0
        self._tracker = tracker
        self._eye_offset = eye_offset

    def _reaim(self) -> None:
        if self._tracker is None:
            return
        import torch
        tgt = self._tracker()
        if tgt is None:
            return
        tgt = tgt.reshape(1, 3)
        eye = tgt + torch.tensor([self._eye_offset], device=tgt.device, dtype=tgt.dtype)
        self._cam.set_world_poses_from_view(eye, tgt)

    def maybe_capture(self) -> None:
        self._i += 1
        if self._i % self._every == 0:
            self._reaim()
            self.capture()

    def capture(self) -> None:
        rgb = self._cam.data.output["rgb"]
        if rgb is None:
            return
        if rgb.ndim == 4:
            rgb = rgb[0]
        frame = rgb[..., :3].detach().cpu().numpy().astype(np.uint8)
        self._frames.append(frame)

    def save(self, out_dir: str, name: str, fps: int = 20, n_keys: int = 4) -> str:
        os.makedirs(out_dir, exist_ok=True)
        if not self._frames:
            print(f"[recorder] no frames captured for '{name}'")
            return ""
        mp4 = os.path.join(out_dir, f"{name}.mp4")
        try:
            import imageio.v2 as imageio
            imageio.mimsave(mp4, self._frames, fps=fps, macro_block_size=None)
        except Exception as exc:  # pragma: no cover
            print(f"[recorder] mp4 encode failed ({exc})")
            mp4 = ""
        # Evenly spaced key frames as PNGs.
        try:
            from PIL import Image
            idxs = np.linspace(0, len(self._frames) - 1, n_keys).astype(int)
            for j, idx in enumerate(idxs):
                Image.fromarray(self._frames[idx]).save(
                    os.path.join(out_dir, f"{name}_key{j}.png")
                )
        except Exception as exc:  # pragma: no cover
            print(f"[recorder] png save failed ({exc})")
        print(f"[recorder] saved {len(self._frames)} frames -> {mp4} (+{n_keys} key PNGs)")
        return mp4
