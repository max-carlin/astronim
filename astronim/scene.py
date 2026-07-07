"""Scenes and particle-morph transitions for AstroAnim.

A `Scene` pairs a build callback with a duration. A `Transition` placed
between two `Scene`s in the `run_scenes` list produces a particle morph:
the outgoing scene's geometry is sampled as a point cloud, the incoming
scene's geometry is sampled as another point cloud, and for the duration
of the transition we lerp positions and colors between them. The camera
also lerps from the source scene's final pose to the target scene's pose.

Public API:
    Scene(build, duration, name="")
    Transition(duration, ease="smoothstep", name="", max_particles=8000)

Internals (for `Universe.run_scenes` to call):
    _extract_particles(simulation, max_particles) -> (N, 6) ndarray
    _sandbox_build(u, build_fn) -> (Simulation, camera_tuple)
    _MorphRenderer(...) — static-list-compatible renderable
"""

from dataclasses import dataclass
from typing import Callable
import math
import pygame
import numpy as np

from astronim.utils.tools import Vec3
from astronim.utils import constants


# ---------- Public dataclasses ----------

@dataclass
class Scene:
    build: Callable          # build(universe) -> None
    duration: float          # seconds (frames if recording, wall-clock otherwise)
    name: str = ""


@dataclass
class Transition:
    """A particle-morph cut between two adjacent Scenes in run_scenes."""
    duration: float
    ease: str = "smoothstep"        # "linear" | "smoothstep" | "ease_in" | "ease_out"
    name: str = ""
    max_particles: int = 8000


# ---------- Easing ----------

def _ease(t: float, kind: str) -> float:
    t = max(0.0, min(1.0, float(t)))
    if kind == "linear":
        return t
    if kind == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    return t


# ---------- Particle extraction ----------

def _text_particles(text_obj) -> np.ndarray:
    """Sample a Text object as a point cloud by rasterizing its glyphs to
    a temp surface and emitting one particle per opaque pixel. Each
    pixel maps to a local (x, y, 0) offset (1 world-unit per pixel),
    centered on text_obj.pos. Returns an (N, 6) array."""
    if not pygame.font.get_init():
        pygame.font.init()
    msg = text_obj.current_message if (
        getattr(text_obj, 'type_out', False) and text_obj.current_message
    ) else text_obj.message
    if not msg:
        return np.zeros((0, 6), dtype=float)
    font = pygame.font.SysFont('Times New Roman', int(text_obj.base_size))
    color = tuple(int(c) for c in text_obj.color[:3])
    surf = font.render(msg, True, color)
    w, h = surf.get_size()
    if w == 0 or h == 0:
        return np.zeros((0, 6), dtype=float)

    # Pixel mask: antialiased text returns SRCALPHA; otherwise treat
    # any pixel with non-zero RGB as a glyph pixel.
    if surf.get_flags() & pygame.SRCALPHA:
        mask = pygame.surfarray.array_alpha(surf) > 16
    else:
        rgb = pygame.surfarray.array3d(surf)
        mask = rgb.sum(axis=-1) > 16

    xs_idx, ys_idx = np.where(mask)
    if xs_idx.size == 0:
        return np.zeros((0, 6), dtype=float)

    cx, cy = w / 2.0, h / 2.0
    local_x = xs_idx.astype(float) - cx
    # screen-y goes down, world-y goes up — flip so the text isn't upside-down
    local_y = cy - ys_idx.astype(float)

    rgb = np.tile(np.asarray(color, dtype=float), (xs_idx.size, 1))
    return np.stack([
        local_x + text_obj.pos.x,
        local_y + text_obj.pos.y,
        np.full_like(local_x, text_obj.pos.z),
        rgb[:, 0], rgb[:, 1], rgb[:, 2],
    ], axis=-1)


def _extract_particles(simulation, max_particles: int) -> np.ndarray:
    """Walk simulation.star_objects + .static_objects and produce a single
    (N, 6) float array of [x, y, z, r, g, b] rows. Capped at max_particles
    via random subsampling. Unknown types are silently skipped."""
    # Local imports to avoid cycles at module-load time
    from astronim.objects.galaxy import Galaxy
    from astronim.objects.darkmatterhalo import DarkMatter
    from astronim.objects.grid import Grid
    from astronim.objects.star import Star
    from astronim.objects.blackhole import BlackHole
    from astronim.objects.text import Text

    chunks: list = []

    objs = list(simulation.star_objects) + list(simulation.static_objects)
    for obj in objs:
        if isinstance(obj, BlackHole):
            chunks.append(obj.morph_particles())

        elif isinstance(obj, Text):
            text_pts = _text_particles(obj)
            if text_pts.size:
                chunks.append(text_pts)

        elif isinstance(obj, Galaxy):
            stars = getattr(obj, 'positions', None) or getattr(obj, 'stars', None)
            if stars:
                rows = []
                ox, oy, oz = obj.pos.x, obj.pos.y, obj.pos.z
                for star in stars:
                    p, c = star[0], star[1]
                    rows.append((p.x + ox, p.y + oy, p.z + oz,
                                 float(c[0]), float(c[1]), float(c[2])))
                chunks.append(np.asarray(rows, dtype=float))

        elif isinstance(obj, DarkMatter):
            local = obj._positions_local
            offset = np.array([obj.pos.x, obj.pos.y, obj.pos.z])
            r, g, b = obj.color
            tint = obj.intensity
            n = local.shape[0]
            xyz = local + offset
            color = np.tile(
                np.array([r * tint * 0.5, g * tint * 0.5, b * tint * 0.5]),
                (n, 1),
            )
            chunks.append(np.concatenate([xyz, color], axis=-1))

        elif isinstance(obj, Grid):
            n = obj.spacing
            xs = obj._rest_x.ravel()
            zs = obj._rest_z.ravel()
            ys = np.full_like(xs, obj.center.y)
            color = np.tile(np.asarray(obj.color, dtype=float), (xs.size, 1))
            chunks.append(np.stack([xs, ys, zs,
                                    color[:, 0], color[:, 1], color[:, 2]], axis=-1))

        elif isinstance(obj, Star):
            c = getattr(obj, 'color', (255, 255, 255))
            chunks.append(np.array([[
                obj.pos.x, obj.pos.y, obj.pos.z,
                float(c[0]), float(c[1]), float(c[2])
            ]], dtype=float))

        # else: unknown renderable; skip

    if not chunks:
        return np.zeros((0, 6), dtype=float)
    pts = np.concatenate(chunks, axis=0)

    if pts.shape[0] > max_particles:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_particles, replace=False)
        pts = pts[idx]
    return pts


# ---------- Sandbox-build helper ----------

def _sandbox_build(u, build_fn):
    """Run `build_fn(u)` against a fresh, throwaway Simulation so we can
    capture the would-be next scene's particle state without disturbing
    the live one. Restores u.simulation and the camera before returning.

    Returns (sandbox_simulation, target_camera_tuple) where the camera
    tuple is (camera, rx, ry, roll).
    """
    from astronim.simulation import Simulation
    from astronim.utils.tools import get_camera_roll, set_camera_roll
    saved_sim = u.simulation
    saved_cam = (
        Vec3(u.renderer.camera.x, u.renderer.camera.y, u.renderer.camera.z),
        u.renderer.rx,
        u.renderer.ry,
        get_camera_roll(),
    )
    u.simulation = Simulation()
    try:
        build_fn(u)
        sandboxed = u.simulation
        target_cam = (
            Vec3(u.renderer.camera.x, u.renderer.camera.y, u.renderer.camera.z),
            u.renderer.rx,
            u.renderer.ry,
            get_camera_roll(),
        )
    finally:
        u.simulation = saved_sim
        u.renderer.camera = saved_cam[0]
        u.renderer.rx = saved_cam[1]
        u.renderer.ry = saved_cam[2]
        set_camera_roll(saved_cam[3])
    return sandboxed, target_cam


# ---------- Transition setup (captures source/target visuals + particles) ----------

def setup_morph_transition(u, transition, next_scene):
    """Construct a fully-configured `_MorphRenderer` for a transition into
    `next_scene`. Captures the source scene's most recent rendered frame
    *and* renders the target scene to a temp surface, so the morph can
    crossfade actual scene visuals around its particle interpolation
    rather than starting/ending on a sparse particle cloud that doesn't
    match either side.

    Caller (Universe.run_scenes) is responsible for clearing the
    simulation and adding the returned morph to it.
    """
    from astronim.utils.tools import get_camera_roll, set_camera_roll

    # 1. Capture source state BEFORE any side-effects: particles and the
    #    last rendered frame the user just saw.
    src_particles = _extract_particles(u.simulation, transition.max_particles)
    src_cam = (
        Vec3(u.renderer.camera.x, u.renderer.camera.y, u.renderer.camera.z),
        u.renderer.rx, u.renderer.ry,
        get_camera_roll(),
    )
    source_surface = u.renderer.screen.copy()

    # 2. Sandbox-build the target scene. Capture target particles + render a
    #    target preview frame so the morph has both endpoints visually.
    sandbox, tgt_cam = _sandbox_build(u, next_scene.build)
    tgt_particles = _extract_particles(sandbox, transition.max_particles)

    # Render the sandbox target to a separate surface using the target's
    # camera. We can't use u.renderer.draw() (that mutates camera state we
    # already restored); easiest is to swap simulation + camera, render
    # once, and swap back.
    saved_sim = u.simulation
    saved_cam = (
        Vec3(u.renderer.camera.x, u.renderer.camera.y, u.renderer.camera.z),
        u.renderer.rx, u.renderer.ry,
        get_camera_roll(),
    )
    u.simulation = sandbox
    u.renderer.camera = tgt_cam[0]
    u.renderer.rx = tgt_cam[1]
    u.renderer.ry = tgt_cam[2]
    set_camera_roll(tgt_cam[3])
    try:
        u.renderer.draw(sandbox)
        target_surface = u.renderer.screen.copy()
    finally:
        u.simulation = saved_sim
        u.renderer.camera = saved_cam[0]
        u.renderer.rx = saved_cam[1]
        u.renderer.ry = saved_cam[2]
        set_camera_roll(saved_cam[3])
        # Restore the source frame on the live screen so the next tick
        # blends from it cleanly (the target render just above clobbered it).
        u.renderer.screen.blit(source_surface, (0, 0))

    return _MorphRenderer(
        src_particles, tgt_particles,
        src_cam, tgt_cam,
        duration=transition.duration, ease=transition.ease,
        renderer=u.renderer,
        source_surface=source_surface,
        target_surface=target_surface,
    )


# ---------- Morph renderer ----------

class _MorphRenderer:
    """Draws an eased lerp between two precomputed (N, 6) particle arrays.

    Adheres to the static-object protocol: `static = True`, has a
    `set_camera` classmethod the renderer calls each frame (we ignore the
    args — we control the camera ourselves via the `renderer` ref).

    Lifecycle: created by `Universe.run_scenes` when it encounters a
    `Transition`. Lives in `simulation.static_objects` for `duration`
    seconds, then the runner clears the simulation and proceeds to the
    real next scene.
    """

    static = True

    # Class-level camera state (touched by renderer's set_camera dispatch,
    # but not actually used by us — we drive the renderer's camera each
    # frame instead).
    camera = Vec3(0, 0, 0)
    rx = 0.0
    ry = 0.0

    # Position is needed by the renderer's depth sort (see renderer.py).
    # Use the midpoint of source and target camera "look at" so we sort
    # to a sensible depth.
    pos: Vec3

    def __init__(self, src_particles, tgt_particles,
                 src_camera, tgt_camera,
                 duration, ease="smoothstep", renderer=None,
                 source_surface=None, target_surface=None):
        # Both clouds end up at n = max(len(src), len(tgt)) so each
        # endpoint fully describes its scene. The denser side is used
        # as-is; the sparser side is sampled WITH REPLACEMENT to
        # inflate to n. The upstream cap (transition.max_particles in
        # _extract_particles) bounds n.
        n_src = len(src_particles)
        n_tgt = len(tgt_particles)
        n = max(n_src, n_tgt)
        # Either side empty → can't oversample from nothing. Fall back
        # to no particle layer; the source/target backdrop crossfade
        # still handles the visual transition.
        if n == 0 or n_src == 0 or n_tgt == 0:
            self.src = np.zeros((0, 6), dtype=float)
            self.tgt = np.zeros((0, 6), dtype=float)
        else:
            rng = np.random.default_rng(0)
            src = src_particles
            tgt = tgt_particles
            if n_src < n:
                src = src[rng.choice(n_src, n, replace=True)]
            if n_tgt < n:
                tgt = tgt[rng.choice(n_tgt, n, replace=True)]
            tgt = tgt[rng.permutation(n)]
            self.src = src
            self.tgt = tgt

        self._src_pos = self.src[:, :3] if self.src.size else np.zeros((0, 3))
        self._src_col = self.src[:, 3:6] if self.src.size else np.zeros((0, 3))
        self._tgt_pos = self.tgt[:, :3] if self.tgt.size else np.zeros((0, 3))
        self._tgt_col = self.tgt[:, 3:6] if self.tgt.size else np.zeros((0, 3))

        self._src_cam = src_camera
        self._tgt_cam = tgt_camera
        # Roll is the 4th element of the camera tuple. Tolerate older
        # 3-tuples (treat as roll=0) just in case anything builds the
        # renderer manually with the old shape.
        self._src_roll = src_camera[3] if len(src_camera) > 3 else 0.0
        self._tgt_roll = tgt_camera[3] if len(tgt_camera) > 3 else 0.0
        self._duration = max(1e-6, float(duration))
        self._ease = ease
        self._renderer = renderer
        self._start_t = None
        self._frames_so_far = 0
        # When set (by run_scenes during recording), pace progress by frame
        # count over an exact budget. When None (interactive mode), pace by
        # wall-clock so the morph completes regardless of tick rate.
        self._total_frames = None

        # Set self.pos to the midpoint between cameras' look-at — a
        # reasonable depth-sort proxy that puts us neither in front of nor
        # behind the morphing cloud.
        self.pos = Vec3(
            0.5 * (src_camera[0].x + tgt_camera[0].x),
            0.5 * (src_camera[0].y + tgt_camera[0].y),
            0.5 * (src_camera[0].z + tgt_camera[0].z),
        )

        # Pre-bake a soft Gaussian sprite for the additive draw
        self._sprite = _bake_morph_sprite(radius=10)

        # Backdrop surfaces for crossfading the actual scene visuals around
        # the particle morph. Without these, t=0 and t=1 look like a sparse
        # particle cloud — *very* different from the procedural rendering
        # of the source/target scenes.
        self._source_surface = source_surface
        self._target_surface = target_surface

    @classmethod
    def set_camera(cls, camera, rx, ry):
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry

    def _progress(self):
        """Return eased t ∈ [0, 1] for the current frame.

        Two pacing modes:
          - RECORDING (`_total_frames` set by run_scenes): frame-count over
            the recorder's exact frame budget. The output mp4 has a uniform
            morph regardless of how slow ticks are at high resolution.
          - INTERACTIVE (`_total_frames` is None): wall-clock pacing, so
            the morph completes within `duration` real seconds even when
            the actual tick rate is well below 60 fps."""
        if self._total_frames is not None:
            t = self._frames_so_far / max(1, self._total_frames)
            self._frames_so_far += 1
        else:
            now_ms = pygame.time.get_ticks()
            if self._start_t is None:
                self._start_t = now_ms
            elapsed_s = (now_ms - self._start_t) / 1000.0
            t = elapsed_s / self._duration
        return _ease(min(1.0, max(0.0, t)), self._ease)

    def draw(self, screen):
        from astronim.utils.tools import set_camera_roll
        e = self._progress()

        # Camera lerp — write back to the live renderer
        if self._renderer is not None:
            sa = self._src_cam[0]; sb = self._tgt_cam[0]
            self._renderer.camera = Vec3(
                (1 - e) * sa.x + e * sb.x,
                (1 - e) * sa.y + e * sb.y,
                (1 - e) * sa.z + e * sb.z,
            )
            self._renderer.rx = (1 - e) * self._src_cam[1] + e * self._tgt_cam[1]
            self._renderer.ry = (1 - e) * self._src_cam[2] + e * self._tgt_cam[2]
            # Roll is the third rotation axis; same lerp as rx/ry.
            set_camera_roll((1 - e) * self._src_roll + e * self._tgt_roll)

        # Crossfade schedule. Quick handoffs at the start and end so we
        # don't dwell in dim states; at every moment at least ONE of
        # {source, particles, target} is at >= 50% opacity:
        #
        #   source:    full [0.00, 0.15], fades to 0 over [0.15, 0.30]
        #   particles: ramp [0.15, 0.30] (matches source fade-out),
        #              hold full [0.30, 0.70],
        #              fade [0.70, 0.85] (matches target ramp-in)
        #   target:    0 until 0.70, ramps to full over [0.70, 0.85],
        #              holds full to end
        #
        # Crossover invariants:
        #   - At e=0.225 (mid of source-out / particles-in): each at ~50%
        #   - At e=0.775 (mid of particles-out / target-in): each at ~50%
        #   - Source briefly held at full at the start [0, 0.15]
        #   - Target briefly held at full at the end [0.85, 1.0]
        def _fade(t, start, end):
            if end <= start:
                return 1.0 if t >= end else 0.0
            return max(0.0, min(1.0, (t - start) / (end - start)))

        # Particles ramp from 0.65 baseline to full by e=0.10 — they
        # appear immediately on top of the source so source + particles
        # are both visible together. Source then holds at full until
        # e=0.45, after which it fades out alone over [0.45, 0.60]
        # while particles stay at full. From e=0.60 to 0.70 the screen
        # is just particles. Tail end [0.70, 0.85] is unchanged: target
        # ramps in additively and particles fade out.
        target_factor = _fade(e, 0.70, 0.85)
        source_factor = 1.0 - _fade(e, 0.45, 0.60)
        particle_factor = min(
            0.65 + 0.35 * _fade(e, 0.0, 0.10),
            1.0 - _fade(e, 0.70, 0.85),
        )
        # stash for the particle-rendering block below
        self._frame_particle_factor = particle_factor

        if self._source_surface is not None and source_factor > 0.0:
            src_a = int(255 * source_factor)
            if src_a > 0:
                surf = self._source_surface
                surf.set_alpha(src_a)
                screen.blit(surf, (0, 0))

        if self._src_pos.shape[0] == 0:
            # No particles to render; still composite the target backdrop.
            if self._target_surface is not None and target_factor > 0.0:
                tgt_a = int(255 * target_factor)
                if tgt_a > 0:
                    surf = self._target_surface
                    surf.set_alpha(tgt_a)
                    screen.blit(surf, (0, 0))
            return

        # Lerp positions and colors
        pos = (1 - e) * self._src_pos + e * self._tgt_pos
        col = (1 - e) * self._src_col + e * self._tgt_col
        col = np.clip(col, 0.0, 255.0).astype(int)

        # Project all positions in batch using the (now lerped) renderer cam
        cam = self._renderer.camera if self._renderer is not None else _MorphRenderer.camera
        rx = self._renderer.rx if self._renderer is not None else _MorphRenderer.rx
        ry = self._renderer.ry if self._renderer is not None else _MorphRenderer.ry

        cos_rx = math.cos(rx); sin_rx = math.sin(rx)
        cos_ry = math.cos(ry); sin_ry = math.sin(ry)

        x = pos[:, 0] - cam.x
        y = pos[:, 1] - cam.y
        z = pos[:, 2] - cam.z

        x2 = cos_rx * x - sin_rx * z
        z = sin_rx * x + cos_rx * z
        x = x2

        y2 = cos_ry * y - sin_ry * z
        z = sin_ry * y + cos_ry * z
        y = y2

        # Camera roll around the view axis — match the rest of the
        # rendering pipeline (get_2d in tools.py, DarkMatter's inline
        # batch). Without this the morph particles project unrolled
        # while the source/target backdrops already have roll baked
        # in, causing an apparent orientation flip.
        from astronim.utils.tools import get_camera_roll
        roll = get_camera_roll()
        if roll != 0.0:
            cos_rz = math.cos(roll); sin_rz = math.sin(roll)
            x2 = cos_rz * x - sin_rz * y
            y = sin_rz * x + cos_rz * y
            x = x2

        valid = z > 0.1
        if not valid.any():
            return
        z_safe = np.where(valid, z, 1.0)
        sx = x * constants.DEPTH / z_safe + constants.WIDTH / 2.0
        sy = constants.HEIGHT / 2.0 - y * constants.DEPTH / z_safe

        # Render: per-particle colored Gaussian sprite, additive
        sprite = self._sprite
        sr = sprite.get_width() // 2
        sx_int = sx.astype(int)
        sy_int = sy.astype(int)
        # Particle visibility (set in the crossfade-schedule block above):
        # 0 at the endpoints (where the source/target backdrop dominates)
        # and full across the midpoint plateau. Without this, the
        # additively-blended particles drown out the dim backdrop frames
        # and the transition appears to start mid-morph.
        particle_brightness = self._frame_particle_factor
        if particle_brightness > 0.01:
            for i in np.flatnonzero(valid):
                r = int(col[i, 0] * particle_brightness)
                g = int(col[i, 1] * particle_brightness)
                b = int(col[i, 2] * particle_brightness)
                if r < 2 and g < 2 and b < 2:
                    continue
                tinted = sprite.copy()
                tinted.fill((r, g, b, 255), special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(
                    tinted,
                    (int(sx_int[i]) - sr, int(sy_int[i]) - sr),
                    special_flags=pygame.BLEND_RGBA_ADD,
                )

        # Crossfade the target backdrop IN. Drawn on top of the particle
        # layer. Critically: composited ADDITIVELY (with RGB pre-multiplied
        # by target_factor) so the BLACK regions of the target render
        # don't darken the source/particles underneath. With plain
        # alpha-blending, the target's empty sky pulled the screen toward
        # black wherever target_factor < 1 — that was the dark gap.
        if self._target_surface is not None and target_factor > 0.0:
            tgt_a = int(255 * target_factor)
            if tgt_a > 0:
                tinted = self._target_surface.copy()
                tinted.fill((tgt_a, tgt_a, tgt_a, 255),
                            special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(tinted, (0, 0),
                            special_flags=pygame.BLEND_RGBA_ADD)


def _bake_morph_sprite(radius: int = 10) -> pygame.Surface:
    size = radius * 2 + 1
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    sigma = max(1.5, radius / 2.4)
    step = 1
    for r in range(radius, 0, -step):
        f = math.exp(-((r / sigma) ** 2) / 2)
        v = int(255 * f)
        if v < 2:
            continue
        pygame.draw.circle(surf, (v, v, v, 255), (radius, radius), r)
    return surf
