"""Photoreal-ish billboarded black hole renderer.

Painter's order layers (back to front):
  1. Halo bloom (Gaussian additive falloff)
  2. Far half of accretion disk (filled annulus quads)
  3. Lensed top arc (far disk wrapping over the shadow at the photon ring)
  4. Event horizon shadow (solid black sphere silhouette)
  5. Photon ring (thin bright rim with bloom)
  6. Near half of accretion disk
  7. Lensed bottom arc

All geometry derives from one master scalar `r_shadow`. Disk inclination is
controlled per-BH via `spin_axis` (in the camera-billboarded view frame, so
the BH is camera-agnostic) plus an additional `tilt_angle` for top-down view.

The disk is shaded via a thermal radial gradient with relativistic Doppler
beaming applied as a brightness factor (no hue shift, to keep additive
blending clean).
"""

import math
import pygame
import numpy as np

from astronim.utils.tools import get_2d, Vec3, distance
from astronim.utils import constants


_C_LIGHT = 2.998e8
_G_NEWTON = 6.674e-11
# Calibrated so mass=1e36 kg → r_shadow ≈ 6.0 model units
_MASS_TO_R_SHADOW_SCALE = 1.55e-9


class BlackHole:
    # Class-level camera state (populated by set_camera each frame)
    camera: Vec3 = Vec3(0.0, 0.0, 0.0)
    rx: float = 0.0
    ry: float = 0.0
    _cos_rx: float = 1.0
    _sin_rx: float = 0.0
    _cos_ry: float = 1.0
    _sin_ry: float = 0.0
    _R_cam_inv = np.eye(3)

    def __init__(self, pos: Vec3, vel: Vec3, mass: float,
                 *,
                 r_shadow: float = None,
                 r_outer: float = None,
                 spin_axis: Vec3 = None,
                 tilt_angle: float = math.radians(15),
                 color=(255, 200, 100),
                 doppler_strength: float = 0.35,
                 glow: bool = True,
                 trail: bool = False,
                 N_radial: int = 20,
                 N_azim: int = 72,
                 # Back-compat kwargs (unused but accepted)
                 radius: float = 0.01,
                 rotation: Vec3 = None):
        # Physics state
        self.pos = pos
        self.velocity = [vel.x, vel.y, vel.z]
        self.mass = mass
        # Visual params
        self.color = color
        self.tilt_angle = tilt_angle
        self.spin_axis = spin_axis if spin_axis is not None else Vec3(0.0, 1.0, 0.0)
        self.doppler_strength = doppler_strength
        self.glow_enabled = glow
        # Trails
        self.trail = trail
        self.trail_list = []
        self.trail_length = 50
        # Back-compat
        self.base_radius = radius
        self.rotation = rotation if rotation is not None else Vec3(0.0, 0.0, 0.0)
        # Master scale & derived radii
        self.r_shadow = r_shadow if r_shadow is not None else self._mass_to_r_shadow(mass)
        self.r_outer = r_outer if r_outer is not None else 6.0 * self.r_shadow
        self.r_inner = 1.5 * self.r_shadow
        self.r_photon = 1.05 * self.r_shadow
        self.halo_extent = 8.0 * self.r_shadow
        # Geometry tunables
        self.N_R = max(4, int(N_radial))
        self.N_T = max(8, int(N_azim))
        # Build static geometry & disk basis once
        self._build_disk_geometry()
        self._M_disk = self._compute_disk_matrix()
        self._halo_cache: dict = {}
        # Backing surface
        self.surface = pygame.Surface(
            (constants.WIDTH, constants.HEIGHT), pygame.SRCALPHA
        )

    # ---------- Geometry / scale ----------

    @staticmethod
    def _mass_to_r_shadow(mass: float) -> float:
        return (
            2.6 * 2.0 * _G_NEWTON * mass / (_C_LIGHT ** 2)
            * _MASS_TO_R_SHADOW_SCALE
        )

    def _compute_disk_matrix(self) -> np.ndarray:
        """3×3 = R_tilt · R_disk where R_disk maps (0,1,0) onto spin_axis.

        spin_axis is interpreted in the billboard view frame, so the BH's
        appearance is invariant to camera rotation.
        """
        s = self.spin_axis
        n = math.sqrt(s.x * s.x + s.y * s.y + s.z * s.z)
        if n < 1e-12:
            R_disk = np.eye(3)
        else:
            sx, sy, sz = s.x / n, s.y / n, s.z / n
            if abs(sy - 1.0) < 1e-9:
                R_disk = np.eye(3)
            elif abs(sy + 1.0) < 1e-9:
                R_disk = np.diag([1.0, -1.0, -1.0])
            else:
                # axis k = cross((0,1,0), s) = (sz, 0, -sx)
                kx, kz = sz, -sx
                kl = math.sqrt(kx * kx + kz * kz)
                kx, kz = kx / kl, kz / kl
                cos_t = sy
                sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
                oc = 1.0 - cos_t
                R_disk = np.array([
                    [cos_t + kx * kx * oc, -kz * sin_t,         kx * kz * oc],
                    [kz * sin_t,            cos_t,             -kx * sin_t],
                    [kx * kz * oc,          kx * sin_t,         cos_t + kz * kz * oc],
                ])
        ct, st = math.cos(self.tilt_angle), math.sin(self.tilt_angle)
        R_tilt = np.array([
            [1.0, 0.0, 0.0],
            [0.0, ct,  -st],
            [0.0, st,   ct],
        ])
        return R_tilt @ R_disk

    def _build_disk_geometry(self):
        """Vertex grid for the disk annulus + per-quad (R, θ) for shading."""
        Rs = np.linspace(self.r_inner, self.r_outer, self.N_R + 1)
        Ts = np.linspace(0.0, 2.0 * math.pi, self.N_T + 1)
        R_grid, T_grid = np.meshgrid(Rs, Ts, indexing='ij')
        X = R_grid * np.cos(T_grid)
        Z = R_grid * np.sin(T_grid)
        Y = np.zeros_like(X)
        self._verts_local = np.stack([X, Y, Z], axis=-1)  # (N_R+1, N_T+1, 3)
        # quad-center (R, θ) for shader sampling
        self._quad_R = 0.25 * (
            R_grid[:-1, :-1] + R_grid[1:, :-1]
            + R_grid[:-1, 1:] + R_grid[1:, 1:]
        )
        self._quad_T = 0.25 * (
            T_grid[:-1, :-1] + T_grid[1:, :-1]
            + T_grid[:-1, 1:] + T_grid[1:, 1:]
        )

    def on_resize(self, w, h):
        self.surface = pygame.Surface((w, h), pygame.SRCALPHA)
        self._halo_cache.clear()

    @classmethod
    def set_camera(cls, camera, rx, ry):
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry
        cos_rx = math.cos(rx); sin_rx = math.sin(rx)
        cos_ry = math.cos(ry); sin_ry = math.sin(ry)
        cls._cos_rx = cos_rx; cls._sin_rx = sin_rx
        cls._cos_ry = cos_ry; cls._sin_ry = sin_ry
        # Forward camera in get_2d:
        #   (x,z) ← R(rx)·(x,z)  [rotation about y-axis]
        #   (y,z) ← R(ry)·(y,z)  [rotation about x-axis]
        # So R_cam_full = R_y_about_x(ry) · R_x_about_y(rx)
        # Inverse applied in reverse order with negated angles.
        Rx_inv = np.array([
            [cos_rx, 0.0,  sin_rx],
            [0.0,    1.0,  0.0   ],
            [-sin_rx, 0.0, cos_rx],
        ])
        Ry_inv = np.array([
            [1.0,  0.0,    0.0   ],
            [0.0,  cos_ry, sin_ry],
            [0.0, -sin_ry, cos_ry],
        ])
        cls._R_cam_inv = Rx_inv @ Ry_inv

    # ---------- Drawing ----------

    def draw(self, screen):
        center = get_2d(self.pos - BlackHole.camera, BlackHole.rx, BlackHole.ry)
        if center is None:
            return
        cx, cy = center
        dist = distance(self.pos, BlackHole.camera)

        # The BH is an emission object, so everything except the shadow
        # composites *additively* onto the screen using pre-multiplied alpha.
        # The shadow uses BLEND_RGBA_SUB to subtract brightness — exactly
        # what an opaque sphere does to the surrounding light.

        # Layer 1: halo glow
        if self.glow_enabled:
            self._blit_halo_additive(screen, cx, cy, dist)

        # Compute disk vertex transforms once
        verts_view = self._verts_local @ self._M_disk.T
        verts_world = verts_view @ BlackHole._R_cam_inv.T
        verts_screen, valid = self._project_grid(verts_world)

        view_z = verts_view[..., 2]
        quad_view_z = 0.25 * (
            view_z[:-1, :-1] + view_z[1:, :-1]
            + view_z[:-1, 1:] + view_z[1:, 1:]
        )
        quad_valid = (
            valid[:-1, :-1] & valid[1:, :-1]
            & valid[:-1, 1:] & valid[1:, 1:]
        )
        rgba = self._disk_shader(self._quad_R, self._quad_T)

        back_mask = quad_valid & (quad_view_z > 0.0)
        front_mask = quad_valid & (quad_view_z <= 0.0)

        # Layer 2: far half of disk (additive, pre-multiplied)
        self._blit_disk_additive(screen, verts_screen, rgba, back_mask)

        # Layer 3: lensed top arc (additive)
        self._blit_lensed_arc_additive(screen, cx, cy, dist, side='top')

        # Layer 4: event horizon shadow (subtractive — removes brightness
        # from halo and any back-disk that snuck through behind it)
        self._blit_shadow_subtractive(screen, cx, cy, dist)

        # Layer 5: photon ring (additive bloom + sharp inner edge)
        self._blit_photon_ring_additive(screen, cx, cy, dist)

        # Layer 6: near half of disk (additive)
        self._blit_disk_additive(screen, verts_screen, rgba, front_mask)

        # Layer 7: lensed bottom arc (additive)
        self._blit_lensed_arc_additive(screen, cx, cy, dist, side='bottom')

    def _project_grid(self, world_off):
        """Project an (..., 3) array of world-space offsets through the
        camera. Returns (screen_xy (..., 2), valid_mask)."""
        offset = np.array([
            self.pos.x - BlackHole.camera.x,
            self.pos.y - BlackHole.camera.y,
            self.pos.z - BlackHole.camera.z,
        ])
        rel = world_off + offset
        x = rel[..., 0]; y = rel[..., 1]; z = rel[..., 2]
        cos_rx, sin_rx = BlackHole._cos_rx, BlackHole._sin_rx
        cos_ry, sin_ry = BlackHole._cos_ry, BlackHole._sin_ry
        # (x,z) ← R(rx) (x,z)
        x2 = cos_rx * x - sin_rx * z
        z = sin_rx * x + cos_rx * z
        x = x2
        # (y,z) ← R(ry) (y,z)
        y2 = cos_ry * y - sin_ry * z
        z = sin_ry * y + cos_ry * z
        y = y2
        valid = z > 0.1
        z_safe = np.where(valid, z, 1.0)
        sx = x * constants.DEPTH / z_safe + constants.WIDTH / 2.0
        sy = constants.HEIGHT / 2.0 - y * constants.DEPTH / z_safe
        return np.stack([sx, sy], axis=-1), valid

    # ---------- Shader ----------

    _PALETTE_T = np.array([0.0, 0.25, 0.6, 1.0])
    _PALETTE_C = np.array([
        [255.0, 245.0, 220.0],   # white-hot ISCO
        [255.0, 200.0, 100.0],   # bright orange
        [230.0, 100.0,  30.0],   # deep orange/red
        [ 60.0,  15.0,   3.0],   # outer red, fading
    ])

    def _disk_shader(self, R, T):
        """Return (..., 4) pre-multiplied RGBA for disk quads. RGB is already
        scaled by the falloff factor so additive blits compose correctly
        regardless of the alpha channel (which BLEND_RGBA_ADD ignores)."""
        t_r = (R - self.r_inner) / max(self.r_outer - self.r_inner, 1e-9)
        t_r = np.clip(t_r, 0.0, 1.0)
        rgb = np.empty((*t_r.shape, 3), dtype=float)
        for i in range(3):
            rgb[..., i] = np.interp(t_r, self._PALETTE_T, self._PALETTE_C[:, i])

        # Doppler beaming
        M20 = self._M_disk[2, 0]
        M22 = self._M_disk[2, 2]
        approach = np.sin(T) * M20 - np.cos(T) * M22
        dop = 1.0 + self.doppler_strength * approach
        dop = np.clip(dop, 0.35, 2.4)
        rgb = rgb * (dop[..., None] ** 3)

        # Brightness envelope: peak near the inner edge, soft outer falloff.
        inner_fade = np.clip(t_r * 5.0, 0.0, 1.0)
        outer_fade = np.clip((1.0 - t_r) ** 1.3, 0.0, 1.0)
        envelope = inner_fade * outer_fade
        rgb = rgb * envelope[..., None]
        rgb = np.clip(rgb, 0.0, 255.0)

        # Pre-multiplied alpha = 255 (we encode visibility entirely in RGB)
        alpha = np.full((*t_r.shape, 1), 255.0)
        rgba = np.concatenate([rgb, alpha], axis=-1)
        return rgba.astype(np.int32)

    # ---------- Layer renderers ----------

    def _temp_surface(self):
        """A scratch full-viewport SRCALPHA surface for staging additive
        polygon batches. Reused across calls in a single frame."""
        if (self.surface.get_width() != constants.WIDTH
                or self.surface.get_height() != constants.HEIGHT):
            self.surface = pygame.Surface(
                (constants.WIDTH, constants.HEIGHT), pygame.SRCALPHA
            )
        return self.surface

    def _blit_disk_additive(self, screen, verts_screen, rgba, mask):
        if not mask.any():
            return
        surf = self._temp_surface()
        surf.fill((0, 0, 0, 0))
        idxs = np.argwhere(mask)
        sx = verts_screen[..., 0]
        sy = verts_screen[..., 1]
        for i, j in idxs:
            xs = (sx[i, j], sx[i, j+1], sx[i+1, j+1], sx[i+1, j])
            ys = (sy[i, j], sy[i, j+1], sy[i+1, j+1], sy[i+1, j])
            if any(math.isnan(v) for v in xs):
                continue
            color = (
                int(rgba[i, j, 0]),
                int(rgba[i, j, 1]),
                int(rgba[i, j, 2]),
                255,
            )
            pts = [(int(xs[k]), int(ys[k])) for k in range(4)]
            pygame.draw.polygon(surf, color, pts)
        screen.blit(surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _blit_shadow_subtractive(self, screen, cx, cy, dist):
        radius = max(2, min(5000, int(self.r_shadow * constants.DEPTH / dist)))
        # Subtract the full RGB range so the halo + any back-disk that shines
        # through the shadow's location is removed (an opaque sphere blocks
        # everything behind it).
        size = radius * 2 + 4
        cutout = pygame.Surface((size, size), pygame.SRCALPHA)
        cutout.fill((0, 0, 0, 0))
        pygame.draw.circle(
            cutout, (255, 255, 255, 255),
            (radius + 2, radius + 2), radius,
        )
        screen.blit(
            cutout, (int(cx) - radius - 2, int(cy) - radius - 2),
            special_flags=pygame.BLEND_RGBA_SUB,
        )

    def _blit_photon_ring_additive(self, screen, cx, cy, dist):
        rp = max(3, int(self.r_photon * constants.DEPTH / dist))
        surf = self._temp_surface()
        surf.fill((0, 0, 0, 0))
        # Faint wide bloom → bright sharp inner ring (pre-multiplied colors)
        for factor, intensity, width in (
            (1.20, 30, 3),
            (1.12, 70, 3),
            (1.06, 130, 2),
            (1.00, 240, 2),
        ):
            r = max(2, int(rp * factor))
            i = max(2, intensity)
            color = (i, int(i * 0.97), int(i * 0.88), 255)
            pygame.draw.circle(
                surf, color, (int(cx), int(cy)), r, width=width,
            )
        screen.blit(surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _blit_lensed_arc_additive(self, screen, cx, cy, dist, side):
        rp = self.r_photon * constants.DEPTH / dist
        # Thin arc — the lensed image is concentrated near the photon ring.
        thickness = (self.r_inner - self.r_photon) * 0.9 * constants.DEPTH / dist
        if rp < 3 or thickness < 2:
            return

        n_pts = 48
        if side == 'top':
            phis = np.linspace(0.0, math.pi, n_pts)
            theta_local = math.pi
        else:
            phis = np.linspace(math.pi, 2.0 * math.pi, n_pts)
            theta_local = 0.0

        envelope = np.sin(phis if side == 'top' else (phis - math.pi))
        envelope = np.clip(envelope, 0.0, 1.0) ** 1.4  # peak narrower
        outer_r = rp + thickness * envelope

        cos_phi = np.cos(phis)
        sin_phi = np.sin(phis)
        inner_x = cx + rp * cos_phi
        inner_y = cy - rp * sin_phi
        outer_x = cx + outer_r * cos_phi
        outer_y = cy - outer_r * sin_phi

        polygon = [(int(outer_x[i]), int(outer_y[i])) for i in range(n_pts)]
        polygon += [(int(inner_x[i]), int(inner_y[i]))
                    for i in range(n_pts - 1, -1, -1)]

        rgb = self._arc_color(theta_local)
        opacity = 0.45
        pre = (
            int(rgb[0] * opacity),
            int(rgb[1] * opacity),
            int(rgb[2] * opacity),
            255,
        )
        surf = self._temp_surface()
        surf.fill((0, 0, 0, 0))
        pygame.draw.polygon(surf, pre, polygon)
        screen.blit(surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _arc_color(self, theta_local):
        """Color sampled from the disk shader at r ≈ 1.4·r_inner, theta_local."""
        R = np.array([1.4 * self.r_inner])
        T = np.array([theta_local])
        rgba = self._disk_shader(R, T)
        return (int(rgba[0, 0]), int(rgba[0, 1]), int(rgba[0, 2]))

    def _blit_halo_additive(self, target, cx, cy, dist):
        max_halo = max(constants.WIDTH, constants.HEIGHT)
        halo_r = max(20, min(max_halo, int(self.halo_extent * constants.DEPTH / dist)))
        bucket = max(20, (halo_r // 12) * 12)
        if bucket not in self._halo_cache:
            self._halo_cache[bucket] = self._bake_halo(bucket)
            if len(self._halo_cache) > 8:
                self._halo_cache.pop(next(iter(self._halo_cache)))
        halo = self._halo_cache[bucket]
        target.blit(
            halo, (int(cx) - bucket, int(cy) - bucket),
            special_flags=pygame.BLEND_RGBA_ADD,
        )

    @staticmethod
    def _bake_halo(radius):
        # Note: the final blit uses BLEND_RGBA_ADD which adds raw RGB and
        # ignores src alpha — so the falloff has to live in the RGB values
        # themselves (pre-multiplied), not in the alpha channel.
        size = radius * 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        sigma = max(8.0, radius / 2.6)
        peak = 0.45  # halo intensity at center (0..1)
        step = max(1, radius // 90)
        base = (255.0, 200.0, 130.0)
        for r in range(radius, 0, -step):
            f = peak * math.exp(-((r / sigma) ** 2) / 2)
            rr = int(base[0] * f)
            gg = int(base[1] * f)
            bb = int(base[2] * f)
            if rr < 2 and gg < 2 and bb < 2:
                continue
            pygame.draw.circle(surf, (rr, gg, bb, 255),
                               (radius, radius), r)
        return surf

    # ---------- Renderer-facing utilities ----------

    def screen_occludes(self, star_pos):
        bh_2d = get_2d(self.pos - BlackHole.camera, BlackHole.rx, BlackHole.ry)
        star_2d = get_2d(star_pos - BlackHole.camera,
                         BlackHole.rx, BlackHole.ry)
        if bh_2d is None or star_2d is None:
            return False
        if (distance(star_pos, BlackHole.camera)
                <= distance(self.pos, BlackHole.camera)):
            return False
        dist = distance(self.pos, BlackHole.camera)
        radius = max(2, int(self.r_shadow * constants.DEPTH / dist))
        dx = star_2d[0] - bh_2d[0]
        dy = star_2d[1] - bh_2d[1]
        return dx * dx + dy * dy < radius * radius

    def draw_trail(self, screen):
        path = [
            pt for pt in (
                get_2d(Vec3(*p) - BlackHole.camera,
                       BlackHole.rx, BlackHole.ry)
                for p in self.trail_list
            ) if pt
        ]
        if len(path) > 1:
            pygame.draw.lines(screen, (255, 255, 255), False, path, 1)

    def draw_trail_color(self, trail_surface, speed_max, speed_min):
        if len(self.trail_list) < 2:
            return
        recent = self.trail_list[-25:]
        path = []
        for p in recent:
            pt = get_2d(Vec3(*p) - BlackHole.camera,
                        BlackHole.rx, BlackHole.ry)
            if pt is not None:
                path.append(pt)
        if len(path) < 2:
            return
        vx, vy, vz = self.velocity
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed_max > speed_min:
            rel = max(0.0, min(1.0,
                               (speed - speed_min) / (speed_max - speed_min)))
        else:
            rel = 0.0
        alpha = int(20 + rel * (255 - 20))
        if rel < 0.5:
            t = rel / 0.5
            rgb = self._lerp_color((0, 128, 255), (255, 255, 255), t)
        else:
            t = (rel - 0.5) / 0.5
            rgb = self._lerp_color((255, 255, 255), (255, 165, 0), t)
        pygame.draw.lines(trail_surface, (*rgb, alpha), False, path, 1)

    @staticmethod
    def _lerp_color(c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )
