"""Planetary nebula: multi-color emission shell around a hot white dwarf.

A planetary nebula is the ejected envelope of a dying sun-like star, lit
from within by the exposed stellar core (a white dwarf whose UV output
ionizes the gas). The light is emission-line radiation, not a continuum,
so the colors map to specific ions:

    blue-green / cyan  — doubly ionized oxygen [OIII], filling the inner
                         cavity closest to the hot star
    red / orange       — hydrogen (Hα) and ionized nitrogen [NII] in the
                         cooler shell and outskirts

Structurally this follows DarkMatter's additive-sprite recipe (batched
numpy projection, soft Gaussian sprites, BLEND_RGBA_ADD accumulation on a
scratch surface) but with two key differences:

  * particles are sampled in a SHELL (Gaussian ring around r_shell), not a
    centrally-peaked profile — projecting a thin shell gives the bright
    limb-brightened rim seen in JWST images (e.g. the Southern Ring,
    NGC 3132) for free, because the line of sight tangent to the shell
    passes through the most material;
  * four color layers (cavity / shell / rim knots / halo) render in one
    pass using sprites with the tint baked in (premultiplied RGB), rather
    than DarkMatter's grayscale-accumulate-then-tint. Where the cyan
    cavity and orange shell overlap in projection the additive sum turns
    yellow-white — exactly what the real rim looks like.

Morphology defaults to spherical but supports the shapes carved by binary
companions or magnetic fields: shape="ring" (torus) and shape="hourglass"
(bipolar / butterfly lobes), plus arbitrary orientation and triaxial
stretch.

shape="eagle" is a different beast: a faithful model of the Eagle Nebula
(M16, an H II star-forming region) sampled from embedded, cleaned Hubble
reference imagery, with a high-detail Pillars of Creation sub-model at
their true location, subtractive dust (columns read as dark silhouettes
carved out of the nebula's own glow — they cannot occlude objects behind
the nebula), and the NGC 6611 cluster instead of a central white dwarf.
"""

import math
import pygame
import numpy as np

from astronim.utils.tools import Vec3, distance, get_2d, get_camera_roll
from astronim.utils import constants


class PlanetaryNebula:
    static = True

    # Camera state populated by set_camera() each frame
    camera = Vec3(0.0, 0.0, 0.0)
    rx = 0.0
    ry = 0.0

    # (radius, color, peak_milli) -> pre-tinted premultiplied sprite
    _SPRITE_CACHE: dict = {}
    # (bucket, spikes, color) -> star glow sprite (small, FIFO-capped)
    _STAR_CACHE: dict = {}

    # Per-layer sprite base radii at scale 1.0 (cavity, shell, rim, halo).
    # Cavity/halo sprites are bigger + softer (diffuse haze); rim sprites
    # smaller + brighter (knots).
    _LAYER_BASE_RADII = (30, 18, 14, 26)
    # Sprite peak brightness per layer: the fraction of full-scale (255)
    # contributed by ONE sprite center, so the expected overlap count sets
    # the on-screen brightness. Unlike DarkMatter these sprites carry the
    # tint baked in, so 8-bit quantization eats dim values — peaks are set
    # high enough that each sprite's colored core survives rounding, and
    # the shell tangent lines (~25-35 overlapping sprites) drive the rim
    # into the luminous 200+ range while broad areas stay unclipped.
    _LAYER_PEAKS = (0.07, 0.095, 0.15, 0.045)

    LAYER_CAVITY = 0
    LAYER_SHELL = 1
    LAYER_RIM = 2
    LAYER_HALO = 3

    def __init__(
        self,
        pos: Vec3,
        r_shell: float = 120.0,
        shell_width: float = None,      # default 0.12 * r_shell
        r_halo: float = None,           # default 2.2 * r_shell
        shape: str = "spherical",       # "spherical" | "ring" | "hourglass"
                                        # | "eagle" (image-derived M16 model)
        waist: float = 0.30,            # hourglass pinch, fraction of r_shell
        ring_power: float = 8.0,        # torus concentration: sin^k(theta)
        equatorial_density: float = 0.0,  # 0..1 extra equator weight (spherical)
        rotation: Vec3 = None,          # Euler radians, applied at build time
        axes: Vec3 = None,              # triaxial stretch, default (1,1,1)
        n_particles: int = None,        # default 6000 (20000 for "eagle")
        clumpiness: float = 0.35,       # 0..1 knottiness of the rim
        n_streaks: int = 24,            # radial wisp directions in the halo
        intensity: float = 1.0,
        cavity_color: tuple = (60, 230, 200),   # [OIII] cyan-teal
        shell_color: tuple = (255, 96, 48),     # Hα + [NII] red-orange
        rim_color: tuple = (255, 200, 130),     # knotty yellow-white rim
        halo_color: tuple = (200, 70, 50),      # faint rusty outer wisps
        star_color: tuple = (200, 220, 255),    # white dwarf, blue-white
        star_glow: float = 1.0,
        spikes: int = 4,                # diffraction spikes: 0 (off), 4 or 6
        sprite_radius: int = None,      # scales all layer sprites; default 18
        seed: int = None,
    ):
        if shape not in ("spherical", "ring", "hourglass", "eagle"):
            raise ValueError(
                f"shape must be 'spherical', 'ring', 'hourglass' or 'eagle',"
                f" got {shape!r}")

        self.pos = Vec3(pos.x, pos.y, pos.z)
        self.r_shell = float(r_shell)
        self.shell_width = (float(shell_width) if shell_width is not None
                            else 0.12 * self.r_shell)
        self.r_halo = (float(r_halo) if r_halo is not None
                       else 2.2 * self.r_shell)
        self.shape = shape
        self.waist = float(waist)
        self.ring_power = float(ring_power)
        self.equatorial_density = float(np.clip(equatorial_density, 0.0, 1.0))
        self.rotation = rotation
        self.axes = axes if axes is not None else Vec3(1.0, 1.0, 1.0)
        # Eagle is an image-derived model spanning a large area; it needs a
        # bigger default budget (12000 is a usable fast preset).
        self.n_particles = (int(n_particles) if n_particles is not None
                            else (20000 if shape == "eagle" else 6000))
        self.clumpiness = float(np.clip(clumpiness, 0.0, 1.0))
        self.n_streaks = int(n_streaks)
        self.intensity = float(intensity)
        self.cavity_color = tuple(int(c) for c in cavity_color[:3])
        self.shell_color = tuple(int(c) for c in shell_color[:3])
        self.rim_color = tuple(int(c) for c in rim_color[:3])
        self.halo_color = tuple(int(c) for c in halo_color[:3])
        self.star_color = tuple(int(c) for c in star_color[:3])
        self.star_glow = float(star_glow)
        self.spikes = int(spikes)
        self.sprite_radius = (float(sprite_radius) if sprite_radius is not None
                              else 18.0)
        self.seed = seed

        self._scale_max = 2.5
        self._build_particles()
        self._scratch_surface = None

    # ---------- Particle sampling ----------

    def _build_particles(self):
        rng = np.random.default_rng(self.seed)
        if self.shape == "eagle":
            self._build_eagle(rng)
            return
        n = self.n_particles
        n_cav = int(0.25 * n)
        n_shell = int(0.50 * n)
        n_halo = n - n_cav - n_shell
        rs = self.r_shell
        w = self.shell_width

        # Layer 0 — inner [OIII] cavity: uniform-density filled ball.
        # r = R * u^(1/3) is the exact inverse-CDF; a uniform ball projects
        # as a smooth centrally-brightest haze.
        d_cav = self._sample_directions(rng, n_cav)
        s_cav = self._shape_scale(d_cav[:, 1])
        r_cav = 0.92 * rs * rng.uniform(0.0, 1.0, n_cav) ** (1.0 / 3.0) * s_cav
        pts_cav = d_cav * r_cav[:, None]

        # Layer 1 — main Hα/[NII] shell: Gaussian ring around r_shell.
        # The bright rim is emergent from projecting this thin shell.
        d_sh = self._sample_directions(rng, n_shell)
        s_sh = self._shape_scale(d_sh[:, 1])
        # Shapes that shrink the shell radius (hourglass waist) pack the
        # same particle count into less area, inflating the additive
        # overlap ~1/s² — normalize the sprite peaks by mean(s)² so the
        # waist glows without clipping a broad region to white.
        self._density_norm = float(np.clip(np.mean(s_sh) ** 2, 0.25, 1.0))
        r_sh = np.clip(rng.normal(rs, w, n_shell), 0.4 * rs, self.r_halo)
        # log-normal radius jitter softens the ring into filaments
        r_sh = r_sh * s_sh * np.exp(rng.normal(0.0, 0.05, n_shell))
        pts_shell = d_sh * r_sh[:, None]
        layer_shell = np.full(n_shell, self.LAYER_SHELL, dtype=np.int8)

        # Layer 2 — rim knots: RELOCATE a fraction of shell particles into
        # compact clumps (flux-conserving, so the nebula stays "less clumpy"
        # than DarkMatter's additive subhalos). Clump centers are drawn from
        # the shell distribution itself, so knots sit on the rim.
        n_reloc = int(0.35 * self.clumpiness * n_shell)
        if n_reloc > 0:
            n_clumps = int(10 + 30 * self.clumpiness)
            centers = pts_shell[rng.choice(n_shell, n_clumps, replace=False)]
            idx = rng.choice(n_shell, n_reloc, replace=False)
            which = rng.integers(0, n_clumps, n_reloc)
            pts_shell[idx] = (centers[which]
                              + rng.normal(0.0, 0.5 * w, (n_reloc, 3)))
            layer_shell[idx] = self.LAYER_RIM

        # Layer 3 — outer halo: rho ∝ r^-2 means dM/dr is constant, so the
        # radius is simply uniform in [r_shell + w, r_halo]. Half the
        # particles snap to a few radial streak directions for the faint
        # spoke texture seen outside real rims.
        d_halo = self._sample_directions(rng, n_halo, soften=True)
        n_snap = n_halo // 2
        if self.n_streaks > 0 and n_snap > 0:
            streak_dirs = rng.normal(0.0, 1.0, (self.n_streaks, 3))
            streak_dirs /= np.linalg.norm(streak_dirs, axis=1, keepdims=True)
            v = (streak_dirs[rng.integers(0, self.n_streaks, n_snap)]
                 + rng.normal(0.0, 0.05, (n_snap, 3)))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            d_halo[:n_snap] = v
        # mild morphology blend so wisps follow the shape without hard edges
        s_halo = 0.5 + 0.5 * self._shape_scale(d_halo[:, 1])
        r_h = rng.uniform(rs + w, self.r_halo, n_halo) * s_halo
        pts_halo = d_halo * r_h[:, None]

        pts = np.concatenate([pts_cav, pts_shell, pts_halo], axis=0)
        layers = np.concatenate([
            np.full(n_cav, self.LAYER_CAVITY, dtype=np.int8),
            layer_shell,
            np.full(n_halo, self.LAYER_HALO, dtype=np.int8),
        ])

        # ~30 points marking the central star, used only by morph_particles
        self._star_local = rng.normal(0.0, 0.01 * rs, (30, 3))

        if self.shape == "ring":
            # compress the symmetry axis for a proper toroidal cross-section
            pts[:, 1] *= 0.6
            self._star_local[:, 1] *= 0.6

        # Triaxial stretch, then build-time Euler rotation (same axis order
        # as Galaxy.rotate_vector: x, then y, then z)
        ax = np.array([self.axes.x, self.axes.y, self.axes.z])
        pts *= ax
        self._star_local *= ax
        if self.rotation is not None:
            pts = self._rotate_points(pts, self.rotation)
            self._star_local = self._rotate_points(self._star_local,
                                                   self.rotation)

        self._positions_local = pts
        self._layer = layers

        # Per-instance layer spec (colors / peaks / radii / blend / zoom
        # response) — the classic four layers verbatim; "eagle" builds a
        # palette-driven spec of its own in _build_eagle.
        self._layer_colors = [self.cavity_color, self.shell_color,
                              self.rim_color, self.halo_color]
        self._layer_peaks = list(self._LAYER_PEAKS)
        self._layer_radii = list(self._LAYER_BASE_RADII)
        self._layer_flags = [pygame.BLEND_RGBA_ADD] * 4
        self._layer_peak_scale = [0.0] * 4

        # Morph stand-in colors, precomputed so morph_particles is
        # shape-agnostic (dimmed roughly by each layer's rendered brightness)
        factors = {
            self.LAYER_CAVITY: 0.55, self.LAYER_SHELL: 0.80,
            self.LAYER_RIM: 1.00, self.LAYER_HALO: 0.35,
        }
        cols = np.empty((len(pts), 3))
        for li, f in factors.items():
            cols[layers == li] = (np.array(self._layer_colors[li], dtype=float)
                                  * f)
        self._morph_colors = cols
        self._morph_index = np.arange(len(pts))

    def _sample_directions(self, rng, n, soften=False):
        """Shape-aware unit vectors (n,3), local y = symmetry axis.

        "spherical" (with equatorial_density) and "ring" use vectorized
        rejection on the polar angle; "hourglass" keeps uniform directions
        (its morphology lives in the radial scale, _shape_scale). soften=True
        halves the rejection contrast — used for the halo so the outer wisps
        only loosely follow the morphology.
        """
        if n <= 0:
            return np.zeros((0, 3))
        chunks = []
        got = 0
        while got < n:
            m = max(64, (n - got) * 3)
            phi = rng.uniform(0.0, 2.0 * math.pi, m)
            cos_t = rng.uniform(-1.0, 1.0, m)
            sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, 1.0))
            if self.shape == "ring":
                p = sin_t ** self.ring_power
            elif self.shape == "spherical" and self.equatorial_density > 0.0:
                e = self.equatorial_density
                p = (1.0 - e) + e * sin_t ** 2
            else:
                p = None
            if p is None:
                keep = slice(None)
            else:
                if soften:
                    p = 0.5 + 0.5 * p
                keep = rng.uniform(0.0, 1.0, m) < p
            pts = np.stack([sin_t * np.cos(phi), cos_t,
                            sin_t * np.sin(phi)], axis=-1)[keep]
            chunks.append(pts)
            got += len(pts)
        return np.concatenate(chunks, axis=0)[:n]

    def _shape_scale(self, cos_theta):
        """Radial scale factor s(θ) as a function of polar angle."""
        if self.shape == "hourglass":
            # r ∝ |cosθ| is two tangent lobes meeting at the origin; the
            # waist floor keeps the pinch a narrow bright ring instead of a
            # point (the shell is densest there, so the waist glows).
            return self.waist + (1.0 - self.waist) * np.abs(cos_theta)
        return np.ones_like(cos_theta)

    @staticmethod
    def _rotate_points(pts, rotation):
        """Vectorized Galaxy.rotate_vector: Euler x, then y, then z."""
        x, y, z = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()
        cx, sx = math.cos(rotation.x), math.sin(rotation.x)
        y, z = cx * y - sx * z, sx * y + cx * z
        cy, sy = math.cos(rotation.y), math.sin(rotation.y)
        z, x = cy * z - sy * x, sy * z + cy * x
        cz, sz = math.cos(rotation.z), math.sin(rotation.z)
        x, y = cz * x - sz * y, sz * x + cz * y
        return np.stack([x, y, z], axis=-1)

    # ---------- Eagle Nebula (M16) ----------

    def _build_eagle(self, rng):
        """Image-derived model of the Eagle Nebula, sampled from embedded,
        cleaned Hubble reference imagery (WikiSky wide field + the 2014 HST
        Pillars of Creation close-up).

        Layers: 12 wide-field palette buckets + 6 pillars palette buckets
        (all additive), 2 SUBTRACTIVE dust layers (pillar columns + the dark
        ridges at top — they carve darkness out of the nebula's own glow,
        giving dark silhouettes with bright rims; they cannot occlude other
        scene objects behind the nebula), and 2 star layers (NGC 6611).

        The rng draw order below is a seeded contract — reordering any call
        changes every seeded build.
        """
        wide = _eagle_image("wide")
        Hm, Wm = wide.shape[:2]
        lum_w = (0.299 * wide[..., 0] + 0.587 * wide[..., 1]
                 + 0.114 * wide[..., 2])

        ext_h = 2.0 * self.r_shell
        ext_w = ext_h * _EAGLE_ASPECT

        n = self.n_particles
        n_gas = int(0.58 * n)
        n_pgas = int(0.30 * n)
        n_pdust = int(0.08 * n)
        n_wdust = max(0, n - n_gas - n_pgas - n_pdust)

        # -- 1. wide-field gas: luminance-weighted sample of the gas map.
        # gamma < 1 because the palette color already carries the pixel's
        # brightness; density x color gives a net response ~ lum^1.8.
        # Inside the pillars box the dedicated sub-model supplies most of
        # the material, so the wide map's contribution is thinned there to
        # avoid a double-counted bright square.
        u0, v0, u1, v1 = _EAGLE_PILLARS_BOX
        p = (lum_w / 255.0) ** _EAGLE_GAMMA
        uu_map = (np.arange(Wm) + 0.5) / Wm
        vv_map = (np.arange(Hm) + 0.5) / Hm
        in_box = ((uu_map[None, :] >= u0) & (uu_map[None, :] <= u1)
                  & (vv_map[:, None] >= v0) & (vv_map[:, None] <= v1))
        p[in_box] *= 0.45
        p = p.ravel()
        p = p / p.sum()
        idx = rng.choice(Hm * Wm, n_gas, p=p)
        rows, cols = idx // Wm, idx % Wm
        u = (cols + rng.uniform(0.0, 1.0, n_gas)) / Wm
        v = (rows + rng.uniform(0.0, 1.0, n_gas)) / Hm
        pix_gas = wide[rows, cols]
        pal_w = np.asarray(_EAGLE_PALETTE_WIDE, dtype=float)
        lay_gas = np.argmin(
            ((pix_gas[:, None, :] - pal_w[None]) ** 2).sum(-1),
            axis=1).astype(np.int8)

        # -- 2. depth: low-frequency undulation + luminance-scaled jitter
        # (bright emission = physically deeper column)
        grid = rng.normal(0.0, 1.0, (9, 9))
        z_base = 0.10 * ext_h * self._eagle_noise(grid, u, v)
        lum_frac = lum_w[rows, cols] / 255.0
        z_gas = z_base + (rng.normal(0.0, 1.0, n_gas)
                          * (0.02 + 0.05 * lum_frac) * ext_h)
        pts_gas = np.stack(
            [(u - 0.5) * ext_w, (0.5 - v) * ext_h, z_gas], axis=-1)

        # -- pillars geometry: the composite reference has the HST pillars
        # inset pasted at its true sky position; that box places the
        # high-detail sub-model (x1.1 because the inset crops the pillars
        # tight). Slightly camera-side of the mid-plane.
        pill = _eagle_image("pillars")
        Hp, Wp = pill.shape[:2]
        plum = (0.299 * pill[..., 0] + 0.587 * pill[..., 1]
                + 0.114 * pill[..., 2])
        # Feather the sub-model's sampling weights to zero at the box
        # border so it dissolves into the wide field without a visible seam
        pfu = (np.arange(Wp) + 0.5) / Wp
        pfv = (np.arange(Hp) + 0.5) / Hp
        feather = np.clip(np.minimum(
            np.minimum(pfu[None, :], 1.0 - pfu[None, :]) / 0.12,
            np.minimum(pfv[:, None], 1.0 - pfv[:, None]) / 0.12), 0.0, 1.0)
        bx = (0.5 * (u0 + u1) - 0.5) * ext_w
        by = (0.5 - 0.5 * (v0 + v1)) * ext_h
        bh = (v1 - v0) * ext_h * 1.1
        bw = bh * Wp / Hp
        z_c = -0.06 * ext_h

        # -- 3. pillars emission --
        w_p = (np.where(plum > 45.0, (plum / 255.0) ** 1.5, 0.0)
               * feather).ravel()
        w_p = w_p / w_p.sum()
        idx = rng.choice(Hp * Wp, n_pgas, p=w_p)
        prow, pcol = idx // Wp, idx % Wp
        pu = (pcol + rng.uniform(0.0, 1.0, n_pgas)) / Wp
        pv = (prow + rng.uniform(0.0, 1.0, n_pgas)) / Hp
        pix_pgas = pill[prow, pcol]
        pal_p = np.asarray(_EAGLE_PALETTE_PILLARS, dtype=float)
        lay_pgas = (12 + np.argmin(
            ((pix_pgas[:, None, :] - pal_p[None]) ** 2).sum(-1),
            axis=1)).astype(np.int8)
        pts_pgas = np.stack(
            [bx + (pu - 0.5) * bw, by + (0.5 - pv) * bh,
             z_c + rng.normal(0.0, 0.03 * bh, n_pgas)], axis=-1)

        # -- 4. pillars dust: dark warm column pixels, subtractive. Same z
        # distribution as their rims so the parallax stays coherent.
        w_d = (np.where((plum < 70.0) & (pill[..., 0] > pill[..., 2]),
                        70.0 - plum, 0.0) * feather).ravel()
        w_d = w_d / w_d.sum()
        idx = rng.choice(Hp * Wp, n_pdust, p=w_d)
        drow, dcol = idx // Wp, idx % Wp
        du = (dcol + rng.uniform(0.0, 1.0, n_pdust)) / Wp
        dv = (drow + rng.uniform(0.0, 1.0, n_pdust)) / Hp
        pts_pdust = np.stack(
            [bx + (du - 0.5) * bw, by + (0.5 - dv) * bh,
             z_c + rng.normal(0.0, 0.03 * bh, n_pdust)], axis=-1)
        lay_pdust = np.full(n_pdust, 18, dtype=np.int8)

        # -- 5. wide dust: the dark rust "eagle" ridges across the top --
        v_rows = ((np.arange(Hm) + 0.5) / Hm)[:, None]
        ridge = ((v_rows < 0.4) & (lum_w < 55.0)
                 & (wide[..., 0] > wide[..., 2]))
        w_r = np.where(ridge, 55.0 - lum_w, 0.0).ravel()
        if w_r.sum() > 0.0 and n_wdust > 0:
            w_r = w_r / w_r.sum()
            idx = rng.choice(Hm * Wm, n_wdust, p=w_r)
            rrow, rcol = idx // Wm, idx % Wm
            ru = (rcol + rng.uniform(0.0, 1.0, n_wdust)) / Wm
            rv = (rrow + rng.uniform(0.0, 1.0, n_wdust)) / Hm
            pts_wdust = np.stack(
                [(ru - 0.5) * ext_w, (0.5 - rv) * ext_h,
                 0.10 * ext_h * self._eagle_noise(grid, ru, rv)], axis=-1)
        else:
            n_wdust = 0
            pts_wdust = np.zeros((0, 3))
        lay_wdust = np.full(n_wdust, 19, dtype=np.int8)

        # -- 6. NGC 6611 cluster stars (positions from the reference) --
        stars = np.asarray(_EAGLE_STARS, dtype=float)
        sz = rng.normal(0.0, 0.05 * ext_h, len(stars))
        pts_star = np.stack(
            [(stars[:, 0] - 0.5) * ext_w, (0.5 - stars[:, 1]) * ext_h, sz],
            axis=-1)
        pink = (stars[:, 2] - stars[:, 4]) > 10.0
        lay_star = np.where(pink, 21, 20).astype(np.int8)

        # Assemble — array order IS blit order: emission, then subtractive
        # dust, then stars on top (never carved by the dust).
        pts = np.concatenate(
            [pts_gas, pts_pgas, pts_pdust, pts_wdust, pts_star])
        layers = np.concatenate(
            [lay_gas, lay_pgas, lay_pdust, lay_wdust, lay_star])

        ax = np.array([self.axes.x, self.axes.y, self.axes.z])
        pts *= ax
        if self.rotation is not None:
            pts = self._rotate_points(pts, self.rotation)

        self._positions_local = pts
        self._layer = layers
        self._star_local = np.zeros((0, 3))
        self._density_norm = 1.0

        # Morph stand-in: emission + stars only (dust would darken the cloud)
        i0 = n_gas + n_pgas
        i1 = i0 + n_pdust + n_wdust
        self._morph_index = np.concatenate(
            [np.arange(0, i0), np.arange(i1, len(pts))])
        self._morph_colors = np.concatenate([
            pix_gas * 0.85,
            pix_pgas * 0.85,
            np.full((len(stars), 3), 255.0),
        ])

        self._build_eagle_layer_spec()

    def _build_eagle_layer_spec(self):
        """Colors / sprite radii / peaks / blend flags for the 22 eagle
        layers. Brighter palette buckets get smaller, hotter sprites; the
        pillars layers have peak_scale=1 so they brighten as the camera
        closes in (offsets the 1/zoom² dilution of additive overlap)."""
        colors, peaks, radii, flags, pscale = [], [], [], [], []
        for c in _EAGLE_PALETTE_WIDE:
            lk = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0
            colors.append(tuple(int(x) for x in c))
            radii.append(int(np.clip(np.round(30 - 16 * lk), 14, 30)))
            peaks.append(0.085 + 0.07 * lk)
            flags.append(pygame.BLEND_RGBA_ADD)
            pscale.append(0.0)
        # Pillars layers are dense (30% of the budget in ~4% of the area):
        # low base peaks keep them subtle at the wide view; peak_scale
        # brings them up as the camera closes in, offsetting the 1/zoom²
        # dilution of additive overlap.
        for c in _EAGLE_PALETTE_PILLARS:
            lk = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0
            # k-means averaging mutes the HST rims' golden glow — push the
            # chroma back up so the rims read warm against the teal
            mean = (c[0] + c[1] + c[2]) / 3.0
            sat = tuple(int(np.clip(mean + (ch - mean) * 1.45, 0, 255))
                        for ch in c)
            colors.append(sat)
            radii.append(int(np.clip(np.round(11 - 5 * lk), 5, 11)))
            peaks.append(0.03 + 0.024 * lk)
            flags.append(pygame.BLEND_RGBA_ADD)
            pscale.append(1.5)
        # Subtractive dust. Subtracting more blue than red leaves a warm
        # rust residue where columns overlay teal glow — the HST look.
        # Tight sprites so the carving hugs the column shapes and doesn't
        # swallow the glowing rims next to them.
        colors += [(18, 28, 42), (20, 26, 36)]
        radii += [5, 16]
        peaks += [0.038, 0.05]
        flags += [pygame.BLEND_RGBA_SUB, pygame.BLEND_RGBA_SUB]
        pscale += [1.3, 0.0]
        # NGC 6611 stars: white + pink
        colors += [(245, 245, 255), (255, 180, 225)]
        radii += [4, 4]
        peaks += [0.85, 0.70]
        flags += [pygame.BLEND_RGBA_ADD, pygame.BLEND_RGBA_ADD]
        pscale += [0.0, 0.0]

        self._layer_colors = colors
        self._layer_peaks = peaks
        self._layer_radii = radii
        self._layer_flags = flags
        self._layer_peak_scale = pscale

    @staticmethod
    def _eagle_noise(grid, u, v):
        """Bilinear interpolation of a coarse noise grid at (u, v) in [0,1]."""
        gh, gw = grid.shape
        gx = np.clip(u, 0.0, 1.0) * (gw - 1)
        gy = np.clip(v, 0.0, 1.0) * (gh - 1)
        x0 = np.minimum(gx.astype(np.int64), gw - 2)
        y0 = np.minimum(gy.astype(np.int64), gh - 2)
        fx = gx - x0
        fy = gy - y0
        return ((grid[y0, x0] * (1 - fx) + grid[y0, x0 + 1] * fx) * (1 - fy)
                + (grid[y0 + 1, x0] * (1 - fx)
                   + grid[y0 + 1, x0 + 1] * fx) * fy)

    # ---------- Sprite baking ----------

    @classmethod
    def _get_sprite(cls, radius, color, peak):
        key = (radius, color, int(round(peak * 1000)))
        sprite = cls._SPRITE_CACHE.get(key)
        if sprite is None:
            sprite = cls._bake_sprite(radius, color, peak)
            cls._SPRITE_CACHE[key] = sprite
            if len(cls._SPRITE_CACHE) > 512:
                cls._SPRITE_CACHE.pop(next(iter(cls._SPRITE_CACHE)))
        return sprite

    @staticmethod
    def _bake_sprite(radius, color, peak):
        # Pre-tinted premultiplied Gaussian stamp: the falloff AND the color
        # both live in the RGB values (alpha is ignored by BLEND_RGBA_ADD),
        # so all four layers can accumulate on one scratch surface.
        size = radius * 2 + 1
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        sigma = max(2.0, radius / 2.4)
        step = max(1, radius // 12)
        for r in range(radius, 0, -step):
            f = peak * math.exp(-((r / sigma) ** 2) / 2.0)
            col = (int(color[0] * f), int(color[1] * f), int(color[2] * f))
            if max(col) < 1:
                continue
            pygame.draw.circle(surf, (*col, 255), (radius, radius), r)
        return surf

    def _get_layer_sprites(self, scale):
        """The layer sprites (and radii) at the current distance bucket."""
        base = self.sprite_radius / 18.0
        sprites, radii = [], []
        for i in range(len(self._layer_colors)):
            r = max(2, int(round(self._layer_radii[i] * base * scale)))
            # peak_scale > 0 makes a layer brighten as the camera closes in
            # (offsets the 1/zoom² dilution of additive overlap); it is
            # exactly 0.0 for the classic shapes, so scale ** 0.0 == 1.0 and
            # their sprite cache keys are unchanged.
            peak = (self._layer_peaks[i] * self.intensity * self._density_norm
                    * (scale ** self._layer_peak_scale[i]))
            sprites.append(self._get_sprite(r, self._layer_colors[i], peak))
            radii.append(r)
        return sprites, radii

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
        cam = PlanetaryNebula.camera
        offset = np.array([
            self.pos.x - cam.x,
            self.pos.y - cam.y,
            self.pos.z - cam.z,
        ])
        rel = self._positions_local + offset

        x = rel[:, 0]; y = rel[:, 1]; z = rel[:, 2]
        cos_rx, sin_rx = math.cos(PlanetaryNebula.rx), math.sin(PlanetaryNebula.rx)
        cos_ry, sin_ry = math.cos(PlanetaryNebula.ry), math.sin(PlanetaryNebula.ry)
        # (x,z) ← R(rx) (x,z); (y,z) ← R(ry) (y,z)
        x2 = cos_rx * x - sin_rx * z
        z = sin_rx * x + cos_rx * z
        x = x2
        y2 = cos_ry * y - sin_ry * z
        z = sin_ry * y + cos_ry * z
        y = y2

        roll = get_camera_roll()
        if roll != 0.0:
            cos_rz, sin_rz = math.cos(roll), math.sin(roll)
            x2 = cos_rz * x - sin_rz * y
            y = sin_rz * x + cos_rz * y
            x = x2

        valid = z > 0.1
        if valid.any():
            z_safe = np.where(valid, z, 1.0)
            sx = x * constants.DEPTH / z_safe + constants.WIDTH / 2.0
            sy = constants.HEIGHT / 2.0 - y * constants.DEPTH / z_safe

            # One global sprite scale per frame from the object-center
            # distance, quantized to 0.25 steps so the cache stays tiny.
            dist = max(1.0, distance(self.pos, cam))
            scale = min(self._scale_max, max(0.5, 2.2 * self.r_shell / dist))
            scale = round(scale * 4.0) / 4.0
            sprites, radii = self._get_layer_sprites(scale)

            # Off-screen cull (matters when flying close: most particles
            # leave the frame and their blits would be wasted)
            margin = max(radii) + 2
            valid &= ((sx > -margin) & (sx < constants.WIDTH + margin)
                      & (sy > -margin) & (sy < constants.HEIGHT + margin))

        if valid.any():
            scratch = self._get_scratch()
            scratch.fill((0, 0, 0, 0))

            sx_int = sx.astype(np.int32)
            sy_int = sy.astype(np.int32)
            layer = self._layer
            # Array order is blit order: for "eagle" the subtractive dust
            # layers sit after all emission layers, so they carve darkness
            # out of the accumulated glow before the final screen blit.
            flags = self._layer_flags
            for i in np.flatnonzero(valid):
                li = layer[i]
                scratch.blit(
                    sprites[li],
                    (int(sx_int[i]) - radii[li], int(sy_int[i]) - radii[li]),
                    special_flags=flags[li],
                )

            # Additive composite: the gas brightens what's behind it rather
            # than occluding it — sprites are already tinted, no MULT pass.
            screen.blit(scratch, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        # Star last, on top of its own gas, so it reads as the source
        self._draw_star(screen)

    # ---------- Central white dwarf ----------

    def _draw_star(self, screen):
        if self.shape == "eagle":
            # M16 is lit by the NGC 6611 cluster (rendered as particle
            # layers), not a central white dwarf.
            return
        try:
            center = get_2d(self.pos - PlanetaryNebula.camera,
                            PlanetaryNebula.rx, PlanetaryNebula.ry)
        except ValueError:
            return
        if center is None:
            return

        dist = max(1.0, distance(self.pos, PlanetaryNebula.camera))
        core_r = max(1, min(6, int(1.5 * 500 / dist)))
        glow_r = int(0.12 * self.r_shell * constants.DEPTH / dist
                     * self.star_glow)
        glow_r = max(10, min(300, glow_r))
        bucket = max(16, (glow_r // 8) * 8)

        sprite = self._get_star_sprite(bucket)
        half = sprite.get_width() // 2
        screen.blit(sprite, (center[0] - half, center[1] - half),
                    special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.circle(screen, (255, 255, 255), center, core_r)

    def _get_star_sprite(self, bucket):
        key = (bucket, self.spikes, self.star_color)
        sprite = PlanetaryNebula._STAR_CACHE.get(key)
        if sprite is None:
            sprite = self._bake_star(bucket, self.spikes, self.star_color)
            PlanetaryNebula._STAR_CACHE[key] = sprite
            if len(PlanetaryNebula._STAR_CACHE) > 8:
                PlanetaryNebula._STAR_CACHE.pop(
                    next(iter(PlanetaryNebula._STAR_CACHE)))
        return sprite

    @staticmethod
    def _bake_star(radius, spikes, color):
        # Glow and diffraction spikes baked into one premultiplied surface.
        # The Gaussian is kept tight (sigma = r/3) so additive clipping to
        # white stays confined to the core — which is what a star should do.
        spike_len = int(3.5 * radius) if spikes > 0 else radius
        half = spike_len
        surf = pygame.Surface((half * 2, half * 2), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        sigma = max(3.0, radius / 3.0)
        peak = 0.55
        step = max(1, radius // 60)
        for r in range(radius, 0, -step):
            f = peak * math.exp(-((r / sigma) ** 2) / 2.0)
            col = (int(color[0] * f), int(color[1] * f), int(color[2] * f))
            if max(col) < 1:
                continue
            pygame.draw.circle(surf, (*col, 255), (half, half), r)

        if spikes > 0:
            # Thin rays with linearly decaying brightness, JWST-style
            # orientation: 4 spikes -> X cross at 45°, 6 -> 60° steps.
            offset = math.radians(45.0 if spikes == 4 else 30.0)
            n_seg = 28
            for k in range(spikes):
                ang = offset + k * (2.0 * math.pi / spikes)
                ca, sa = math.cos(ang), math.sin(ang)
                for s in range(n_seg):
                    t0 = s / n_seg
                    t1 = (s + 1) / n_seg
                    f = 0.5 * peak * (1.0 - t0)
                    col = (int(color[0] * f), int(color[1] * f),
                           int(color[2] * f))
                    if max(col) < 1:
                        break
                    pygame.draw.line(
                        surf, (*col, 255),
                        (half + ca * t0 * spike_len,
                         half + sa * t0 * spike_len),
                        (half + ca * t1 * spike_len,
                         half + sa * t1 * spike_len), 1)
        return surf

    # ---------- Particle stand-in (used by scene-transition morphs) ----------

    def morph_particles(self):
        """Return an (M, 6) array [x, y, z, r, g, b] of the nebula as a
        particle cloud in world coordinates, respecting the layer colors
        (dimmed roughly by each layer's rendered brightness) plus a small
        white cluster for the central star."""
        origin = np.array([self.pos.x, self.pos.y, self.pos.z])
        pts = self._positions_local[self._morph_index] + origin
        cols = self._morph_colors

        star_pts = self._star_local + origin
        star_cols = np.full((len(star_pts), 3), 255.0)

        return np.concatenate([
            np.concatenate([pts, cols], axis=-1),
            np.concatenate([star_pts, star_cols], axis=-1),
        ], axis=0)

    @classmethod
    def set_camera(cls, camera, rx, ry):
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry


# ======================================================================
# Eagle Nebula embedded data
#
# Generated once by a preprocessing script from the reference imagery
# (WikiSky 4xHubble Eagle Nebula composite + the 2014 HST Pillars of
# Creation close-up): pasted-inset artifacts harmonized, point sources
# stripped from the gas maps, k-means palettes and a bright-star
# catalogue extracted, images re-encoded as small JPEGs. Decoded lazily
# at first eagle build via pygame — no pillow, no filesystem access.
# ======================================================================

import io
import base64

_EAGLE_GAMMA = 1.0      # luminance exponent for position sampling

_EAGLE_IMG_CACHE = {}


def _eagle_image(which):
    """Decode an embedded reference map to a (H, W, 3) float32 array."""
    if which not in _EAGLE_IMG_CACHE:
        b64 = _EAGLE_WIDE_B64 if which == "wide" else _EAGLE_PILLARS_B64
        surf = pygame.image.load(io.BytesIO(base64.b64decode(b64)))
        arr = pygame.surfarray.array3d(surf)          # (W, H, 3)
        _EAGLE_IMG_CACHE[which] = np.transpose(
            arr, (1, 0, 2)).astype(np.float32)        # (H, W, 3)
    return _EAGLE_IMG_CACHE[which]

# Generated by preprocess_eagle.py — Eagle Nebula embedded data
_EAGLE_WIDE_WH = (212, 225)
_EAGLE_ASPECT = 0.942222
_EAGLE_PILLARS_BOX = (0.5613, 0.4489, 0.7453, 0.6844)

_EAGLE_PALETTE_WIDE = (
    (27, 23, 10),
    (85, 63, 24),
    (88, 78, 78),
    (83, 89, 119),
    (128, 99, 51),
    (91, 111, 154),
    (127, 107, 95),
    (167, 120, 65),
    (103, 145, 204),
    (133, 144, 151),
    (170, 151, 110),
    (171, 168, 182),
)

_EAGLE_PALETTE_PILLARS = (
    (64, 71, 87),
    (101, 79, 61),
    (67, 93, 114),
    (122, 112, 101),
    (82, 126, 149),
    (170, 172, 160),
)

_EAGLE_STARS = (
    (0.5684, 0.7133, 255, 253, 255),
    (0.3160, 0.5400, 255, 255, 244),
    (0.6014, 0.6889, 254, 255, 244),
    (0.3679, 0.5244, 251, 255, 250),
    (0.5425, 0.3556, 254, 251, 255),
    (0.2170, 0.8422, 255, 250, 255),
    (0.3821, 0.6222, 255, 255, 227),
    (0.6769, 0.8467, 248, 253, 255),
    (0.4929, 0.6556, 247, 255, 246),
    (0.9245, 0.8422, 255, 249, 255),
    (0.5307, 0.2911, 255, 249, 255),
    (0.6321, 0.3467, 252, 250, 255),
    (0.2335, 0.4778, 255, 254, 225),
    (0.3585, 0.9844, 255, 248, 255),
    (0.6792, 0.7622, 245, 253, 255),
    (0.3184, 0.5978, 255, 253, 227),
    (0.9693, 0.9600, 255, 247, 255),
    (0.9434, 0.9533, 248, 250, 255),
    (0.7123, 0.7578, 240, 254, 255),
    (0.5873, 0.9444, 255, 246, 255),
    (0.8844, 0.9267, 255, 246, 255),
    (0.9528, 0.4089, 255, 246, 255),
    (0.7429, 0.8578, 255, 245, 255),
    (0.8396, 0.8556, 255, 245, 255),
    (0.4505, 0.4822, 255, 245, 255),
    (0.6344, 0.3844, 255, 245, 255),
    (0.9458, 0.8067, 247, 249, 255),
    (0.8939, 0.6756, 242, 251, 255),
    (0.8892, 0.6289, 240, 252, 255),
    (0.2925, 0.8333, 255, 244, 255),
    (0.4387, 0.7956, 255, 244, 255),
    (0.5189, 0.3733, 255, 244, 255),
    (0.3703, 0.6333, 255, 247, 239),
    (0.5566, 0.7356, 254, 247, 241),
    (0.6297, 0.9444, 255, 243, 255),
    (0.8797, 0.5733, 255, 243, 255),
    (0.9410, 0.5489, 255, 243, 255),
    (0.7453, 0.7956, 237, 252, 255),
    (0.4528, 0.7333, 255, 243, 254),
    (0.8042, 0.7089, 254, 243, 255),
    (0.8632, 0.7644, 230, 255, 255),
    (0.2901, 0.6978, 255, 243, 250),
    (0.5542, 0.6311, 235, 252, 255),
    (0.5047, 0.4778, 229, 255, 255),
    (0.3774, 0.6689, 255, 244, 243),
    (0.2170, 0.1356, 255, 242, 253),
    (0.5094, 0.6844, 246, 248, 243),
    (0.3679, 0.7200, 255, 243, 245),
    (0.4410, 0.8689, 255, 241, 255),
    (0.8443, 0.6356, 255, 241, 255),
    (0.9929, 0.5778, 255, 241, 255),
    (0.9670, 0.4489, 255, 241, 255),
    (0.5920, 0.3933, 255, 241, 255),
    (0.8349, 0.2978, 255, 241, 255),
    (0.9410, 0.8311, 246, 245, 255),
    (0.4670, 0.4600, 230, 253, 255),
    (0.2099, 0.1711, 255, 246, 225),
    (0.5236, 0.9156, 255, 240, 255),
    (0.1627, 0.8489, 255, 240, 255),
    (0.2736, 0.8356, 255, 240, 255),
    (0.9693, 0.8756, 231, 252, 255),
    (0.6132, 0.0333, 255, 245, 225),
    (0.9434, 0.6778, 255, 239, 255),
    (0.9575, 0.5089, 255, 239, 255),
    (0.8561, 0.2644, 255, 239, 255),
    (0.4175, 0.1022, 255, 239, 255),
    (0.2406, 0.4533, 255, 250, 198),
    (0.3019, 0.8733, 255, 241, 244),
    (0.4976, 0.7889, 255, 241, 244),
    (0.1392, 0.8089, 255, 244, 228),
    (0.0991, 0.8689, 255, 242, 236),
    (0.4906, 0.9667, 255, 239, 250),
    (0.7618, 0.9644, 255, 238, 255),
    (0.2146, 0.7378, 255, 238, 255),
    (0.2995, 0.6911, 255, 238, 255),
    (0.4127, 0.1667, 255, 238, 255),
    (0.8986, 0.9844, 245, 243, 255),
    (0.5519, 0.9844, 255, 248, 202),
    (0.7830, 0.7244, 221, 255, 255),
    (0.6887, 0.7222, 221, 255, 255),
    (0.4458, 0.8978, 255, 242, 232),
    (0.5189, 0.3311, 244, 243, 255),
    (0.7972, 0.6800, 232, 249, 255),
    (0.5472, 0.5222, 220, 255, 255),
    (0.5684, 0.0778, 255, 237, 255),
    (0.4104, 0.4422, 245, 242, 255),
    (0.5142, 0.5067, 219, 255, 255),
    (0.3184, 0.1600, 255, 238, 248),
    (0.5425, 0.0800, 255, 238, 248),
    (0.0189, 0.0956, 255, 242, 226),
    (0.3019, 0.1600, 251, 241, 240),
    (0.2406, 0.9222, 255, 236, 255),
    (0.8774, 0.8111, 239, 244, 255),
    (0.3255, 0.0111, 255, 244, 212),
    (0.2547, 0.6689, 255, 236, 253),
    (0.0920, 0.0978, 255, 241, 227),
    (0.5731, 0.2733, 252, 237, 255),
    (0.0755, 0.4889, 255, 235, 255),
    (0.5873, 0.3667, 255, 235, 255),
    (0.0896, 0.0600, 255, 241, 224),
    (0.3090, 0.6311, 255, 240, 228),
    (0.1439, 0.1089, 255, 238, 238),
    (0.4623, 0.5622, 219, 253, 255),
    (0.2618, 0.8711, 255, 236, 248),
    (0.3514, 0.5778, 244, 248, 215),
    (0.8255, 0.0356, 255, 238, 237),
    (0.5849, 0.8200, 234, 245, 255),
    (0.3703, 0.7489, 255, 235, 251),
    (0.6722, 0.4356, 255, 234, 255),
    (0.8349, 0.2444, 255, 234, 255),
    (0.1132, 0.8422, 255, 238, 234),
    (0.7783, 0.1200, 255, 236, 244),
    (0.0519, 0.7978, 255, 240, 223),
    (0.5967, 0.0200, 255, 239, 228),
    (0.4575, 0.9822, 255, 242, 212),
    (0.7524, 0.1622, 255, 239, 225),
    (0.0920, 0.8422, 255, 238, 230),
    (0.4788, 0.9489, 255, 233, 255),
    (0.3090, 0.7244, 255, 233, 255),
    (0.8656, 0.2467, 255, 233, 255),
)

_EAGLE_WIDE_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEP"
    "ERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4e"
    "Hh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCADhANQDASIA"
    "AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQA"
    "AAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3"
    "ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWm"
    "p6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEA"
    "AwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSEx"
    "BhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElK"
    "U1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3"
    "uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDC0qFb"
    "dpBEUEBXHA+bPofassxzNdSxrKN+88quCOuOe1WluDFBtZFGByCOtEc2+IkkB88ZH1r5R4RLXodc"
    "Zts2tOKJZpFyXP3mJzzzT7mWeCTIJwe6moNHJMTeaNr+q96uSo2w5bJHqO1c6pR5mmerGElTTKqa"
    "jPE+ZJgOch1OPzqCfUVlDrvLAt97I+XpUN4N0eX4yRjjOKzrdImnCszqc8ZPFdMaSXQ4pyltfQm1"
    "O1t9QjK3kcjTbR5Uy55wMDIqGPwoJ4WJnEZAPzL0xz/D3HPPtV2JbiO62SSq4QfKCegrTW9SLbGw"
    "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"
)

_EAGLE_PILLARS_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAQDAwMDAgQDAwMEBAQFBgoGBgUFBgwICQcKDgwPDg4M"
    "DQ0PERYTDxAVEQ0NExoTFRcYGRkZDxIbHRsYHRYYGRj/2wBDAQQEBAYFBgsGBgsYEA0QGBgYGBgY"
    "GBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBj/wAARCAEFAPoDASIA"
    "AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQA"
    "AAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3"
    "ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWm"
    "p6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEA"
    "AwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSEx"
    "BhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElK"
    "U1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3"
    "uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD5DjS4"
    "WOSHjbIoV/lUnAYNwSODkDkc9uhNPhgJ+4CAT+NWpY3Vl8ol/lBbPADHqPw6e/WmltuOATjvkYOT"
    "xz/T1roVkejyyfUtQ2iMWyuCw6e/9KmNuYsBcYx1Jqkl0TkIgJGOc1ZjMlxGd/TPX1qoSRNSndaj"
    "mlfoD07iowp35yckcnP61eV5RpctlGkJDyLIWMa7gQCAA2MgfMcgHBwCegquieW2S+D0xWy8zjnp"
    "sXrHexwH2j1PethbPCGUfLnisyxIRDzjgAcdK6bcrxkqwIx1Bpt8trK5VOHMnd2t+JTGBhETBPHA"
    "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"
    "yyOcL0TtWj9htxxsoopSZrTWh//Z"
)
