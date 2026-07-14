import os
import runpy
import time
import traceback

import pygame
from astronim.simulation import Simulation
from astronim.renderer import Renderer
from astronim.recorder import Recorder
from astronim.utils import constants
from astronim.utils.tools import Vec3
import numpy as np


class Universe:
    """Handles the core loop of astronim. 

        -Initializes pygame and the rendering window. 
        -Manages the simulation and updates bodies. 
        -Runs the renderer (camera, drawing, depth sorting)
        -Records frames to an output .mp4 file using recorder. 
        -Handles user input to move around scene. 

        Attributes
        ----------
        screen : pygame.Surface
            The pygame screen that the simulation will draw on. 

        simulation : Simulation
            Contains all the objects in the scene and update logic

        renderer : Renderer
            Contains draw order and handles drawing logic. 

        recorder : Recorder
            Captures screen frame-by-frame and uses ffmpeg to save video to a specified output file. 

        running : bool
            Main loop will run while true. 

        clock : pygame.time.Clock
            Controls frame timing

        speed : float
            Controls fly control speeds. 

        shift_speed : float
            Controls fly control speeds while holding down right shift. 

        Methods
        -------
        handle_events(): 
            Processes pygame events. 

        main_loop(): 
            Runs the simulation, rendering, and recording. Must be called in any project file. 

        controls():
            Handles all the controls for moving through the scene (WASD, space, ctrl, shift)

    """
    def __init__(self, width: int = 1920, height: int = 1080,
                 output_file: str =  "output", on_tab: callable = None):


        pygame.init()
        self.width = width
        self.height = height
        # Visible window is freely resizable; the scene is drawn into an
        # offscreen surface at the requested render resolution and blit-scaled
        # onto the window each frame. The recorder captures the offscreen
        # surface, so recordings are always at (width, height) regardless of
        # how the user has resized the on-screen window.
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.render_surface = pygame.Surface((width, height))
        pygame.display.set_caption('Astronim')

        constants.WIDTH = width
        constants.HEIGHT = height

        self.simulation = Simulation()
        self.renderer = Renderer(self.render_surface, width, height)
        self.recorder = Recorder()

        self.running = True
        self.clock = pygame.time.Clock()

        self.speed = 0.2
        self.shift_speed_factor = 10
        self.rotation_speed = np.radians(1)

        # Camera smoothing — exponential lerp toward a target velocity each frame.
        # Higher = snappier (1.0 is instant). Lower = more inertia.
        self.move_smoothing = 0.15
        self.rot_smoothing = 0.2

        # Distance-scaled speed: when the camera is within `distance_reference`
        # AU of the nearest star, translation speed scales down linearly toward
        # `min_distance_scale`. Far away, scale is capped at 1.0.
        self.distance_reference = 5.0
        self.min_distance_scale = 0.02

        self._move_vel = np.zeros(3)
        self._rx_vel = 0.0
        self._ry_vel = 0.0

        self.output_file = output_file
        self.static_mouse = False

        self.on_tab = on_tab

        # Register with the interactive REPL pump, if it's listening. Import
        # lazily so non-interactive scripts don't pay any import cost.
        from astronim import repl as _repl
        _repl._set_active(self)


    def handle_events(self):
        '''
        Handles all of the pygame events, quits running on window close. 
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.pause_sim()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                if self.on_tab is not None:
                    self.on_tab()



    def tick(self):
        '''
        Run exactly one frame: events, controls, physics, render, record, present.

        Extracted from ``main_loop`` so external drivers (e.g. the interactive
        REPL pump) can advance the scene one frame at a time while keeping
        pygame on the main thread.
        '''
        self.dt = 0.1 * 86400  # seconds per frame
        self.handle_events()

        keys = pygame.key.get_pressed()
        self.controls(keys)

        self.simulation.update(self.dt)
        self.renderer.draw(self.simulation)
        self.recorder.save_frame(self.render_surface)

        win_w, win_h = self.screen.get_size()
        render_w, render_h = self.render_surface.get_size()
        if (win_w, win_h) == (render_w, render_h):
            self.screen.blit(self.render_surface, (0, 0))
        else:
            # Fit the render surface into the window preserving aspect ratio;
            # letterbox/pillarbox the remainder so objects don't squish when
            # the user drags to a different aspect.
            scale = min(win_w / render_w, win_h / render_h)
            scaled_w = max(1, int(render_w * scale))
            scaled_h = max(1, int(render_h * scale))
            scaled = pygame.transform.smoothscale(
                self.render_surface, (scaled_w, scaled_h)
            )
            self.screen.fill((0, 0, 0))
            self.screen.blit(
                scaled, ((win_w - scaled_w) // 2, (win_h - scaled_h) // 2)
            )
        pygame.display.flip()

    def main_loop(self, watch=None):
        '''
        The main loop that updates our simulation, draws to the screen, and records the scene.

        If ``watch`` is a path to a script defining ``build(u)``, the loop polls its
        mtime and re-executes ``build(u)`` in place whenever the file changes, so
        edits take effect without closing the pygame window.
        '''
        watch_mtime = self._watch_mtime(watch) if watch else None
        poll_every = 10
        frame = 0

        while self.running:
            if watch and frame % poll_every == 0:
                current = self._watch_mtime(watch)
                if current is not None and current != watch_mtime:
                    watch_mtime = current
                    self._reload_scene(watch)

            self.tick()
            frame += 1

        pygame.quit()
        self._finalize_recorder()

    def _finalize_recorder(self):
        '''Stop the recorder and write the final mp4. Idempotent — safe to
        call even if the recorder was never started.'''
        if not self.recorder.recording:
            return
        if self.output_file.endswith('.mp4'):
            out = self.output_file
        else:
            out = self.output_file + '.mp4'
        self.recorder.stop(output_file=out)

    def run_scenes(self, scenes):
        '''Play a sequence of `Scene`s back-to-back. `Transition` items
        between Scenes produce a particle-morph cross-cut.

        Per-Scene durations are interpreted as:
          - RECORDING: frames (duration · 60), so the final mp4 has each
            scene at exactly the requested length regardless of tick speed.
          - INTERACTIVE: wall-clock seconds.

        The recorder is NOT auto-started; call `u.recorder.start()` before
        `run_scenes` if you want a video.
        '''
        from .scene import Transition, setup_morph_transition
        target_fps = 60

        i = 0
        while i < len(scenes):
            if not self.running:
                break
            item = scenes[i]

            if isinstance(item, Transition):
                # Find the next non-Transition item to morph TOWARD
                next_scene = next(
                    (s for s in scenes[i + 1:] if not isinstance(s, Transition)),
                    None,
                )
                if next_scene is None:
                    i += 1
                    continue   # trailing transition with nothing to morph into

                # Deregister the outgoing scene's camera animation BEFORE
                # morph setup: setup renders the target scene for its
                # backdrop, and a still-live callback would fire on that
                # draw — steering the camera (and, for callbacks that
                # mutate it in place, corrupting captured poses) while the
                # morph is being constructed.
                self.renderer.clear_camera_animation()

                try:
                    morph = setup_morph_transition(self, item, next_scene)
                except Exception:
                    label = f" {item.name!r}" if item.name else ""
                    print(f"[astronim] morph setup failed in transition{label}:")
                    traceback.print_exc()
                    i += 1
                    continue

                self.simulation.clear()
                self.renderer.trail_surface.fill((0, 0, 0, 0))
                # Pace by frame count when recording (exact mp4 timing) or
                # by wall-clock when interactive (so the morph completes
                # within `duration` regardless of actual tick rate).
                if self.recorder.recording:
                    morph._total_frames = max(1, int(item.duration * target_fps))
                self.simulation.add(morph)
                self._tick_for(item.duration, target_fps)
                i += 1
                continue

            # Regular Scene
            self._run_one_scene(item, target_fps)
            i += 1

        pygame.quit()
        self._finalize_recorder()

    def _run_one_scene(self, scene, target_fps):
        self.simulation.clear()
        self.renderer.trail_surface.fill((0, 0, 0, 0))
        # Fresh camera-animation slate: without this, a callback registered
        # by the previous scene (e.g. an orbit) keeps firing every frame and
        # steers the camera through scenes that never asked for it.
        self.renderer.clear_camera_animation()
        try:
            scene.build(self)
        except Exception:
            label = f" {scene.name!r}" if scene.name else ""
            print(f"[astronim] build raised in scene{label}:")
            traceback.print_exc()
            return
        self._tick_for(scene.duration, target_fps)

    def _tick_for(self, duration, target_fps):
        if self.recorder.recording:
            frames = max(1, int(duration * target_fps))
            for _ in range(frames):
                if not self.running:
                    break
                self.tick()
        else:
            start = time.monotonic()
            while self.running and (time.monotonic() - start) < duration:
                self.tick()


    
    def controls(self, keys):
        '''
        Handles WASD/shift/ctrl translation and Q/E/P/L rotation with velocity
        smoothing and distance-scaled speed (camera auto-slows near objects).
        '''
        forward = np.array([np.sin(self.renderer.rx), 0.0, np.cos(self.renderer.rx)])
        right = np.array([np.sin(self.renderer.rx + np.pi / 2), 0.0,
                          np.cos(self.renderer.rx + np.pi / 2)])
        up = np.array([0.0, 1.0, 0.0])

        target = np.zeros(3)
        if keys[pygame.K_w]:
            target += forward
        if keys[pygame.K_s]:
            target -= forward
        if keys[pygame.K_d]:
            target += right
        if keys[pygame.K_a]:
            target -= right
        if keys[pygame.K_LSHIFT]:
            target += up
        if keys[pygame.K_LCTRL]:
            target -= up

        shift_mult = self.shift_speed_factor if keys[pygame.K_RSHIFT] else 1.0
        target *= self.speed * shift_mult * self._distance_scale()

        self._move_vel += (target - self._move_vel) * self.move_smoothing
        self.renderer.camera.x += float(self._move_vel[0])
        self.renderer.camera.y += float(self._move_vel[1])
        self.renderer.camera.z += float(self._move_vel[2])

        target_rx = 0.0
        target_ry = 0.0
        if keys[pygame.K_q]:
            target_rx -= self.rotation_speed
        if keys[pygame.K_e]:
            target_rx += self.rotation_speed
        if keys[pygame.K_p]:
            target_ry += self.rotation_speed
        if keys[pygame.K_l]:
            target_ry -= self.rotation_speed

        self._rx_vel += (target_rx - self._rx_vel) * self.rot_smoothing
        self._ry_vel += (target_ry - self._ry_vel) * self.rot_smoothing
        delta_rx = float(self._rx_vel)
        delta_ry = float(self._ry_vel)
        self.renderer.rx += delta_rx
        self.renderer.ry += delta_ry

        # Q/E (yaw) and P/L (pitch) both orbit the camera around the scene's
        # centroid so the look-at point stays in frame instead of swinging
        # away. The relationship: when forward rotates by +Δ, the camera
        # offset around the pivot rotates by −Δ (opposite sense). Empty
        # scenes or camera-on-pivot fall back to in-place rotation.
        if abs(delta_rx) > 1e-9 or abs(delta_ry) > 1e-9:
            pivot = self._scene_pivot()
            if pivot is not None:
                cam = self.renderer.camera
                if abs(delta_rx) > 1e-9:
                    dx = cam.x - pivot[0]
                    dz = cam.z - pivot[2]
                    if dx * dx + dz * dz > 1e-6:
                        cos_d = np.cos(delta_rx)
                        sin_d = np.sin(delta_rx)
                        cam.x = pivot[0] + cos_d * dx + sin_d * dz
                        cam.z = pivot[2] - sin_d * dx + cos_d * dz
                if abs(delta_ry) > 1e-9:
                    dy = cam.y - pivot[1]
                    dz = cam.z - pivot[2]
                    if dy * dy + dz * dz > 1e-6:
                        cos_d = np.cos(delta_ry)
                        sin_d = np.sin(delta_ry)
                        cam.y = pivot[1] + cos_d * dy + sin_d * dz
                        cam.z = pivot[2] - sin_d * dy + cos_d * dz

    def _scene_pivot(self):
        """Mass-weighted centroid of dynamic objects, or unweighted mean of
        static objects' positions if there are no dynamic ones. Returns
        ``None`` for an empty scene."""
        sim = self.simulation
        if sim.star_objects:
            masses = np.asarray(sim.star_masses, dtype=float)
            positions = np.asarray(sim.star_positions, dtype=float)
            total = masses.sum()
            if total > 0:
                return (positions * masses[:, None]).sum(axis=0) / total
            return positions.mean(axis=0)
        if sim.static_objects:
            pts = [
                [o.pos.x, o.pos.y, o.pos.z]
                for o in sim.static_objects
                if hasattr(o, 'pos')
            ]
            if pts:
                return np.array(pts, dtype=float).mean(axis=0)
        return None

    def _distance_scale(self):
        '''Scale factor for translation speed based on distance to nearest star.

        Full speed (1.0) when beyond ``distance_reference`` AU; linearly scales
        down to ``min_distance_scale`` as the camera approaches a star.
        '''
        stars = self.simulation.star_objects
        if not stars:
            return 1.0
        cam = self.renderer.camera
        nearest_sq = float("inf")
        for obj in stars:
            dx = obj.pos.x - cam.x
            dy = obj.pos.y - cam.y
            dz = obj.pos.z - cam.z
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < nearest_sq:
                nearest_sq = d2
        d = np.sqrt(nearest_sq)
        scale = d / self.distance_reference
        return max(self.min_distance_scale, min(1.0, scale))

    # def pause_sim(self):
    #     for obj in self.simulation.star_objects:
    #         obj.static = True
    def pause_sim(self):
        self.paused = not getattr(self, "paused", False)
        for obj in self.simulation.star_objects:
            obj.static = self.paused

    @staticmethod
    def _watch_mtime(path):
        try:
            return os.path.getmtime(path)
        except (FileNotFoundError, OSError):
            return None

    def _reload_scene(self, watch_path):
        '''Re-execute the watched script and rebuild the scene in place.'''
        cam = self.renderer.camera
        rx, ry = self.renderer.rx, self.renderer.ry
        try:
            ns = runpy.run_path(watch_path, run_name="__reload__")
        except Exception:
            print(f"[astronim] reload failed to execute {watch_path}:")
            traceback.print_exc()
            return

        build = ns.get("build")
        if not callable(build):
            print(f"[astronim] {watch_path} defines no build(u); keeping current scene")
            return

        self.simulation.clear()
        try:
            build(self)
        except Exception:
            print(f"[astronim] build(u) raised during reload:")
            traceback.print_exc()
        finally:
            self.renderer.camera = cam
            self.renderer.rx = rx
            self.renderer.ry = ry

        




