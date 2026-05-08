"""Dark matter halo: NFW-distributed particle cloud with diffuse Gaussian sprites.

Real cold-dark-matter halos follow the Navarro-Frenk-White density profile,
    ρ(r) = ρ_s / [(r / r_s) · (1 + r / r_s)²],
which is sharply peaked at the center and falls off as r⁻³ at large radii.
Modern N-body simulations (Aquarius, Millennium, IllustrisTNG) show those
halos as diffuse, lumpy clouds — not hard-edged shells.

We sample N particles from the NFW radial mass distribution via inverse-CDF,
optionally triaxial via per-axis scaling, with optional Gaussian "subhalo"
clusters for substructure. Each particle renders as a soft Gaussian sprite
with additive blending to produce the volumetric-glow look. Tint is applied
once per frame via BLEND_RGBA_MULT instead of per-particle so we don't need
many copies of the sprite.
"""

import math
import pygame
import numpy as np

from astronim.utils.tools import get_2d, Vec3, distance
from astronim.utils import constants


class DarkMatter:
    static = True

    # Camera state populated by set_camera() each frame
    camera = Vec3(0.0, 0.0, 0.0)
    rx = 0.0
    ry = 0.0

    # Class-level sprite cache (shared across all halos that pick the same size)
    _SPRITE_CACHE: dict = {}

    def __init__(
        self,
        pos: Vec3,
        r_scale: float = 100.0,
        r_virial: float = 400.0,
        n_particles: int = 6000,
        color: tuple = (150, 180, 255),
        intensity: float = 1.0,
        axes: Vec3 = None,
        subhalo_count: int = 0,
        subhalo_radius: float = None,
        sprite_radius: int = 28,
        seed: int = None,
        # Back-compat: the old ctor was DarkMatter(pos, base_radius)
        base_radius: float = None,
    ):
        # Back-compat: if someone passes the old `base_radius` positional arg,
        # treat it as r_virial and pick a sensible r_scale.
        if base_radius is not None:
            r_virial = float(base_radius)
            r_scale = r_virial / 4.0

        self.pos = pos
        self.r_scale = float(r_scale)
        self.r_virial = float(r_virial)
        self.n_particles = int(n_particles)
        self.color = tuple(int(c) for c in color[:3])
        self.intensity = float(intensity)
        self.axes = axes if axes is not None else Vec3(1.0, 1.0, 1.0)
        self.subhalo_count = int(subhalo_count)
        self.subhalo_radius = (
            float(subhalo_radius) if subhalo_radius is not None
            else 0.15 * self.r_virial
        )
        self.sprite_radius = max(4, int(sprite_radius))
        self.seed = seed

        self._build_particles()
        self._sprites = self._get_sprites(self.sprite_radius)
        self._scratch_surface = None

    # ---------- Particle sampling ----------

    def _build_particles(self):
        rng = np.random.default_rng(self.seed)
        rs = self.r_scale
        rv = self.r_virial

        # Inverse-CDF sample of the NFW radial mass distribution
        # dM/dr ∝ r² · ρ(r) = r² / [(r/rs) · (1 + r/rs)²]
        # Floor the inner radius at 0.6·r_s so the NFW central spike
        # doesn't pile up enough particles to clip the additive buffer.
        r_grid = np.linspace(rs * 0.6, rv, 4000)
        rho = 1.0 / ((r_grid / rs) * (1.0 + r_grid / rs) ** 2)
        dM_dr = rho * r_grid * r_grid
        cdf = np.cumsum(dM_dr)
        cdf = cdf / cdf[-1]

        n = self.n_particles
        u = rng.uniform(0.0, 1.0, n)
        radii = np.interp(u, cdf, r_grid)

        # Uniform points on a sphere
        phi = rng.uniform(0.0, 2.0 * math.pi, n)
        cos_theta = rng.uniform(-1.0, 1.0, n)
        sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, 0.0, 1.0))

        ax, ay, az = self.axes.x, self.axes.y, self.axes.z
        x = radii * sin_theta * np.cos(phi) * ax
        y = radii * sin_theta * np.sin(phi) * ay
        z = radii * cos_theta * az

        # Optional Gaussian subhalo clusters embedded in the host halo
        if self.subhalo_count > 0:
            extras = []
            n_per = 120
            for _ in range(self.subhalo_count):
                u_sh = rng.uniform(0.25, 0.92)
                r_sh = float(np.interp(u_sh, cdf, r_grid))
                phi_sh = rng.uniform(0.0, 2.0 * math.pi)
                cth = rng.uniform(-0.7, 0.7)
                sth = math.sqrt(max(0.0, 1.0 - cth * cth))
                cx = r_sh * sth * math.cos(phi_sh) * ax
                cy = r_sh * sth * math.sin(phi_sh) * ay
                cz = r_sh * cth * az
                sigma = self.subhalo_radius
                sx = rng.normal(cx, sigma, n_per)
                sy = rng.normal(cy, sigma, n_per)
                sz = rng.normal(cz, sigma, n_per)
                extras.append(np.stack([sx, sy, sz], axis=-1))
            if extras:
                extra = np.concatenate(extras, axis=0)
                pts = np.stack([x, y, z], axis=-1)
                pts = np.concatenate([pts, extra], axis=0)
                x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        self._positions_local = np.stack([x, y, z], axis=-1)

        # All particles share one sprite intensity. The NFW radial
        # distribution of particles already produces the bright-center /
        # diffuse-outskirts look via accumulation — boosting individual
        # particles by their density on top of that just clips the center
        # to white. Keep one sprite, let count do the work.
        self._intensity_bucket = np.zeros(len(x), dtype=np.int32)

    # ---------- Sprite cache ----------

    @classmethod
    def _get_sprites(cls, radius: int):
        key = radius
        if key in cls._SPRITE_CACHE:
            return cls._SPRITE_CACHE[key]
        sprites = [cls._bake_sprite(radius, intensity)
                   for intensity in _INTENSITY_LEVELS]
        cls._SPRITE_CACHE[key] = sprites
        return sprites

    @staticmethod
    def _bake_sprite(radius: int, intensity: float):
        size = radius * 2 + 1
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        sigma = max(2.0, radius / 2.4)
        peak = int(255 * intensity)
        if peak <= 0:
            return surf
        # Concentric stamps: outer faint → inner bright. Pre-multiplied RGB
        # (alpha unused under BLEND_RGBA_ADD).
        step = max(1, radius // 12)
        for r in range(radius, 0, -step):
            f = math.exp(-((r / sigma) ** 2) / 2.0)
            v = int(peak * f)
            if v < 2:
                continue
            pygame.draw.circle(surf, (v, v, v, 255), (radius, radius), r)
        return surf

    # ---------- Scratch surface ----------

    def _get_scratch(self):
        w, h = constants.WIDTH, constants.HEIGHT
        if (self._scratch_surface is None
                or self._scratch_surface.get_width() != w
                or self._scratch_surface.get_height() != h):
            self._scratch_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        return self._scratch_surface

    def on_resize(self, w, h):
        self._scratch_surface = None

    # ---------- Drawing ----------

    def draw(self, screen):
        # Project all particle positions in one batched pass.
        # World positions = self.pos + local offsets - camera
        cam = DarkMatter.camera
        offset = np.array([
            self.pos.x - cam.x,
            self.pos.y - cam.y,
            self.pos.z - cam.z,
        ])
        rel = self._positions_local + offset

        x = rel[:, 0]; y = rel[:, 1]; z = rel[:, 2]
        cos_rx, sin_rx = math.cos(DarkMatter.rx), math.sin(DarkMatter.rx)
        cos_ry, sin_ry = math.cos(DarkMatter.ry), math.sin(DarkMatter.ry)
        # (x,z) ← R(rx) (x,z); (y,z) ← R(ry) (y,z)
        x2 = cos_rx * x - sin_rx * z
        z = sin_rx * x + cos_rx * z
        x = x2
        y2 = cos_ry * y - sin_ry * z
        z = sin_ry * y + cos_ry * z
        y = y2

        valid = z > 0.1
        if not valid.any():
            return
        z_safe = np.where(valid, z, 1.0)
        sx = x * constants.DEPTH / z_safe + constants.WIDTH / 2.0
        sy = constants.HEIGHT / 2.0 - y * constants.DEPTH / z_safe

        scratch = self._get_scratch()
        scratch.fill((0, 0, 0, 0))

        sprites = self._sprites
        sprite_r = self.sprite_radius
        buckets = self._intensity_bucket

        # On-screen particle coords (rounded), bucket index, and validity
        sx_int = sx.astype(np.int32)
        sy_int = sy.astype(np.int32)
        # Fast Python loop — sprites are small and pygame.blit is the bottleneck
        for i in np.flatnonzero(valid):
            scratch.blit(
                sprites[buckets[i]],
                (int(sx_int[i]) - sprite_r, int(sy_int[i]) - sprite_r),
                special_flags=pygame.BLEND_RGBA_ADD,
            )

        # Tint the accumulated grayscale glow once
        r, g, b = self.color
        intensity = self.intensity
        tr = int(np.clip(r * intensity, 0, 255))
        tg = int(np.clip(g * intensity, 0, 255))
        tb = int(np.clip(b * intensity, 0, 255))
        scratch.fill((tr, tg, tb, 255), special_flags=pygame.BLEND_RGBA_MULT)

        # Composite onto the screen additively (so it brightens the background
        # like a glowing cloud rather than occluding stars behind it)
        screen.blit(scratch, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    @classmethod
    def set_camera(cls, camera, rx, ry):
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry


# One very faint sprite. Many overlapping copies near the NFW center build
# up into a bright core; sparse copies at the outskirts stay diffuse. With
# the 0.6·r_s inner floor, a typical inner pixel sees ~25 sprites; with a
# peak of 0.04 (≈10/255 each) that gives ~250 — bright but unclipped.
_INTENSITY_LEVELS = (0.04,)
N_LEVELS = len(_INTENSITY_LEVELS)
