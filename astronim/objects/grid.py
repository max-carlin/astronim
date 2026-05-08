"""Spacetime embedding-diagram grid.

A horizontal mesh whose vertices live at fixed (x, z) positions on a
`center.y` plane and whose y-coordinate is dynamically displaced each frame
by the summed Newtonian potential of every massive source in the scene:

    Δy(x, z) = − Σᵢ k · Mᵢ / √((x − xᵢ)² + (z − zᵢ)² + ε²)

This is the Newtonian-limit shape of Flamm's paraboloid — the familiar
"trampoline with bowling balls" picture of curved spacetime.

The strength k is auto-calibrated each frame so the deepest well in the
scene reaches `well_depth` model units, regardless of absolute mass scale,
so the visualization works whether the masses are stellar (~1e30 kg) or
galactic (~1e36+ kg).
"""

import math
import pygame
import numpy as np

from astronim.utils.tools import get_2d, Vec3


class Grid:
    """A live spacetime-embedding mesh.

    Parameters
    ----------
    extent : float
        Full edge length of the (square) plane, in model units.
    spacing : int
        Vertices per side. ``spacing × spacing`` total vertices.
    center : Vec3
        Center of the plane. The rest height of the sheet is ``center.y``.
    color : tuple
        Base RGB line color.
    well_depth : float
        Maximum vertical dip (model units) at the deepest well in the
        current frame. The grid auto-rescales each frame so the heaviest
        source produces a well of this depth.
    softening : float
        Inner radius (model units) capping the central spike of each well.
    sources : list | tuple | object | None
        Where to read masses from each frame. Pass a live list reference
        (e.g. ``u.simulation.star_objects``) so the grid picks up new
        objects automatically as they're added. Tuples of lists are
        flattened. A single object is wrapped. ``None`` disables the
        dynamic deformation.
    line_width : int
    """

    static = True

    # Camera state (populated by set_camera each frame)
    camera = Vec3(0.0, 0.0, 0.0)
    rx = 0.0
    ry = 0.0

    def __init__(
        self,
        extent: float = 600.0,
        # Second positional captures the legacy `m` extent; the new code is
        # square so we just take the larger of (extent, m). Keyword-only
        # arguments below.
        m: float = None,
        center: Vec3 = None,
        *,
        spacing: int = 32,
        color=(90, 130, 200),
        well_depth: float = 50.0,
        softening: float = 8.0,
        sources=None,
        line_width: int = 1,
        # Back-compat kwargs
        n: float = None,
        curvature=None,
        rotation: Vec3 = None,
    ):
        # Legacy: Grid(n, m, center, spacing=..., curvature=...)
        if n is not None:
            extent = float(n)
        if m is not None:
            extent = max(float(extent), float(m))

        self.extent = float(extent)
        self.spacing = max(2, int(spacing))
        self.center = center if center is not None else Vec3(0.0, 0.0, 0.0)
        self.color = tuple(int(c) for c in color[:3])
        self.well_depth = float(well_depth)
        self.softening = float(softening)
        self.line_width = max(1, int(line_width))
        # rotation kwarg is accepted but not honored — the embedding diagram is
        # canonically horizontal; users who want a tilt can rotate the camera.
        self._rotation = rotation

        self._sources = self._coerce_sources(sources, curvature)
        self._build_rest_grid()

        # Pre-allocated working arrays so per-frame draw() doesn't churn.
        n_vert = self.spacing * self.spacing
        self._scratch_y = np.empty(n_vert, dtype=float)

    # ---------- Construction helpers ----------

    @staticmethod
    def _coerce_sources(sources, curvature):
        """Build a single live-reference list of source-providers.

        The returned object is a list of *containers* (lists/tuples) that we
        iterate each frame. Storing the containers (not their contents)
        keeps the live-update behavior: the user's append to the original
        list is reflected next frame.
        """
        out = []
        if sources is not None:
            if isinstance(sources, (list, tuple)):
                # Either a list of objects or a tuple of lists. Probe the
                # first element to decide.
                if len(sources) > 0 and isinstance(sources[0], (list, tuple)):
                    out.extend(sources)
                else:
                    out.append(sources)
            else:
                # Single object — wrap so iteration yields it
                out.append([sources])
        if curvature is not None:
            out.append([curvature])
        return out

    def _build_rest_grid(self):
        n = self.spacing
        half = self.extent / 2.0
        xs = np.linspace(self.center.x - half, self.center.x + half, n)
        zs = np.linspace(self.center.z - half, self.center.z + half, n)
        Xg, Zg = np.meshgrid(xs, zs, indexing='xy')  # shape (n, n)
        self._rest_x = Xg.astype(float)
        self._rest_z = Zg.astype(float)
        # Flat (M, 2) for vectorized potential math
        self._rest_xz = np.stack([Xg.ravel(), Zg.ravel()], axis=-1)

    # ---------- Per-frame deformation ----------

    def _gather_sources(self):
        """Walk the live containers and yield (mass, x, z) tuples."""
        out = []
        for container in self._sources:
            try:
                it = iter(container)
            except TypeError:
                continue
            for obj in it:
                mass = getattr(obj, 'mass', None)
                pos = getattr(obj, 'pos', None)
                if mass is None or pos is None:
                    continue
                try:
                    out.append((float(mass), float(pos.x), float(pos.z)))
                except (AttributeError, TypeError, ValueError):
                    continue
        return out

    def _compute_dy(self):
        """Compute Δy at every grid vertex, vectorized.

        Returns a flat (M,) array of dy values, where M = spacing².
        """
        sources = self._gather_sources()
        m_total = self._rest_xz.shape[0]
        if not sources:
            return np.zeros(m_total, dtype=float)

        # (S, 3) — mass, x, z
        src = np.asarray(sources, dtype=float)
        masses = src[:, 0]                     # (S,)
        max_mass = masses.max()
        if max_mass <= 0:
            return np.zeros(m_total, dtype=float)

        # Auto-calibrate strength: well at the heaviest source's center
        # reaches `well_depth` (with softening floor).
        #   y_center = -k * M_max / sqrt(0 + ε²) = -k * M_max / ε  =  -well_depth
        #   ⇒ k = well_depth · ε / M_max
        eps = self.softening
        k = self.well_depth * eps / max_mass

        # Broadcast: (M, 1) - (1, S) → (M, S) per axis
        dx = self._rest_xz[:, 0:1] - src[:, 1][None, :]
        dz = self._rest_xz[:, 1:2] - src[:, 2][None, :]
        dist = np.sqrt(dx * dx + dz * dz + eps * eps)        # (M, S)
        contrib = masses[None, :] / dist                      # (M, S)
        dy = -k * contrib.sum(axis=1)                          # (M,)
        return dy

    # ---------- Drawing ----------

    def draw(self, screen):
        n = self.spacing
        dy = self._compute_dy().reshape(n, n)
        rest_y = self.center.y

        # Build vertex (x, y, z) world positions in flat (n, n, 3) form
        verts_x = self._rest_x
        verts_y = rest_y + dy
        verts_z = self._rest_z

        # Project all vertices via the camera. Inline the get_2d math here
        # rather than calling it n² times — same convention as
        # astronim/utils/tools.py:45.
        cam = Grid.camera
        cos_rx = math.cos(Grid.rx); sin_rx = math.sin(Grid.rx)
        cos_ry = math.cos(Grid.ry); sin_ry = math.sin(Grid.ry)

        x = verts_x - cam.x
        y = verts_y - cam.y
        z = verts_z - cam.z

        x2 = cos_rx * x - sin_rx * z
        z = sin_rx * x + cos_rx * z
        x = x2

        y2 = cos_ry * y - sin_ry * z
        z = sin_ry * y + cos_ry * z
        y = y2

        from astronim.utils import constants
        valid = z > 0.1
        z_safe = np.where(valid, z, 1.0)
        sx = (x * constants.DEPTH / z_safe + constants.WIDTH / 2.0)
        sy = (constants.HEIGHT / 2.0 - y * constants.DEPTH / z_safe)

        # Per-vertex depth tint: deeper wells get a brighter line color.
        max_dy = float(np.abs(dy).max()) if dy.size else 0.0
        if max_dy > 1e-9:
            tint = np.clip(np.abs(dy) / max_dy, 0.0, 1.0)
        else:
            tint = np.zeros((n, n), dtype=float)

        # Draw row strips (constant z index) and column strips (constant x).
        self._draw_strips(screen, sx, sy, valid, tint, by_row=True)
        self._draw_strips(screen, sx, sy, valid, tint, by_row=False)

    def _draw_strips(self, screen, sx, sy, valid, tint, by_row):
        n = self.spacing
        base = self.color
        lw = self.line_width
        for i in range(n):
            # Pull a row or column of vertices
            if by_row:
                row_sx = sx[i, :]
                row_sy = sy[i, :]
                row_valid = valid[i, :]
                row_tint = tint[i, :]
            else:
                row_sx = sx[:, i]
                row_sy = sy[:, i]
                row_valid = valid[:, i]
                row_tint = tint[:, i]

            # Walk consecutive valid runs; emit each as a polyline.
            run = []
            run_tint_max = 0.0
            for j in range(n):
                if row_valid[j]:
                    run.append((float(row_sx[j]), float(row_sy[j])))
                    if row_tint[j] > run_tint_max:
                        run_tint_max = float(row_tint[j])
                else:
                    if len(run) >= 2:
                        self._emit_polyline(screen, run, run_tint_max, base, lw)
                    run = []
                    run_tint_max = 0.0
            if len(run) >= 2:
                self._emit_polyline(screen, run, run_tint_max, base, lw)

    @staticmethod
    def _emit_polyline(screen, pts, tint_max, base, lw):
        # Brighten the line proportional to its deepest dip, with a floor so
        # the un-deformed outer mesh is still faintly visible.
        floor = 0.35
        f = floor + (1.0 - floor) * tint_max
        color = (
            int(base[0] * f),
            int(base[1] * f),
            int(base[2] * f),
        )
        if lw == 1:
            pygame.draw.aalines(screen, color, False, pts)
        else:
            pygame.draw.lines(screen, color, False, pts, lw)

    @classmethod
    def set_camera(cls, camera, rx, ry):
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry
