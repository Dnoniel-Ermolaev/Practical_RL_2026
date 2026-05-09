"""Auxiliary Atari Pong wrapper for ES / CEM / policy-gradient experiments.

Adapted to modern `gymnasium` + `ale-py` (the original code targeted the now-removed
OpenAI gym + scipy.misc.imresize). Provides:
    - make_pong()  -> a 4-frame, 42x42, grayscale Pong env with 4-tuple step()/reset()
                     so the rest of the seminar code can stay unchanged.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.core import Wrapper
from gymnasium.spaces.box import Box

try:
    import cv2  # cheap and fast resize
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    from PIL import Image
    _HAS_CV2 = False

# Make ALE envs discoverable through gymnasium.make
gym.register_envs(ale_py)


def _resize(img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Resize an HxWx3 uint8 image to (h, w, 3). Backend: cv2 if present, else PIL."""
    h, w = size_hw
    if _HAS_CV2:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(img)
    return np.array(pil.resize((w, h), Image.BILINEAR))


def make_pong():
    """Create a preprocessed Atari Pong environment with a stack of `n_frames` frames.

    Returns the framebuffer as a (n_frames, height, width) float32 array in [0, 1].
    The wrapper exposes a legacy 4-tuple `step(a) -> (obs, r, done, info)` and a
    `reset() -> obs` (without the new gymnasium info dict) for compatibility with
    the seminar/project code.
    """
    base = gym.make("ALE/Pong-v5", frameskip=4, full_action_space=False)
    return PreprocessAtari(base)


class PreprocessAtari(Wrapper):
    def __init__(self, env, height=42, width=42,
                 crop=lambda img: img[34:34 + 160], n_frames=4):
        super().__init__(env)
        self.img_size = (height, width)
        self.crop = crop
        self.n_frames = n_frames
        self.observation_space = Box(0.0, 1.0, [n_frames, height, width], dtype=np.float32)
        self.framebuffer = np.zeros([n_frames, height, width], dtype=np.float32)

    def reset(self, **kwargs):
        self.framebuffer = np.zeros_like(self.framebuffer)
        obs, _info = self.env.reset(**kwargs)
        self._update_buffer(obs)
        return self.framebuffer.copy()

    def step(self, action):
        new_img, r, term, trunc, info = self.env.step(action)
        self._update_buffer(new_img)
        done = bool(term or trunc)
        return self.framebuffer.copy(), r, done, info

    # ---------- image processing ----------
    def _update_buffer(self, img: np.ndarray) -> None:
        img = self._preproc_image(img)
        self.framebuffer = np.vstack([img[None], self.framebuffer[:-1]])

    def _preproc_image(self, img: np.ndarray) -> np.ndarray:
        img = self.crop(img)
        img = _resize(img, self.img_size).mean(-1)
        return img.astype("float32") / 255.0
