import pygame 
import numpy as np
import math
from astronim.utils.tools import get_2d, Vec3, distance, rotation_matrix
from astronim.utils import constants


class BlackHole:
    def __init__(self, pos:Vec3, vel: Vec3, mass: float,
                 radius: float = 0.01, color = (255, 255, 255),
                 trail = False, rotation : Vec3 = Vec3(0, 0, 0),
                 tilt_angle: float = math.radians(15)):

        self.velocity = [vel.x, vel.y, vel.z]
        self.pos = pos
        self.mass = mass
        self.base_radius = radius
        self.color = color
        self.rotation = rotation
        self.tilt_angle = tilt_angle

        self.num_lines = 30

        self.trail = trail
        self.trail_list = []
        self.trail_length = 50



        # self.circle_lines = self.glow_circle(radius=5, num_lines=30)
        self.parabola_lines = self.glow_parabolas(num_lines=self.num_lines, a=0.08)
        self.ellipse_lines = self.glow_ellipse(num_lines=self.num_lines, a=8, b=6)

        self.circle_lines = []
        self.circle_lines.extend(self.parabola_lines)
        self.circle_lines.extend(self.ellipse_lines)


        if any([self.rotation.x, self.rotation.y, self.rotation.z]):
            self.circle_lines = [
                ([self.rotate_vector(p) for p in points], color)
                for points, color in self.circle_lines
            ]
        
        self.surface = pygame.Surface((constants.WIDTH, constants.HEIGHT), pygame.SRCALPHA)



    def on_resize(self, w, h):
        self.surface = pygame.Surface((w, h), pygame.SRCALPHA)


    def draw(self, screen):
        self.surface.fill((0, 0, 0, 0))
        # Painter's order: back rings first so the cutout punches them out
        # (otherwise the spherical horizon doesn't occlude what's behind it).
        self._draw_ellipse_segments(front_only=False)
        self.draw_event_horizon_cutout(radius_px=6.0)
        self._draw_ellipse_segments(front_only=True)
        self._draw_parabolas()

        screen.blit(self.surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_ellipse_segments(self, front_only):
        for idx, (points, color) in enumerate(self.ellipse_lines):
            view_points = self._tilt_points(points)
            back_segs, front_segs = self.split_ellipse_by_depth(view_points)
            segs = front_segs if front_only else back_segs
            r, g, b, a = color
            radial_factor = np.exp(- (idx / self.num_lines) ** 2 * 2.5)
            N_GLOW = 5

            for seg in segs:
                seg = self._to_world(seg)
                path = []
                for p in seg:
                    pt = get_2d((self.pos + p) - BlackHole.camera,
                                BlackHole.rx, BlackHole.ry)
                    if pt:
                        path.append(pt)

                if len(path) < 2:
                    continue

                for j in range(N_GLOW):
                    t = j / N_GLOW
                    width = int(2 + t * 10)
                    alpha = int(a * radial_factor * np.exp(-3 * t))
                    cool = int(30 * t)

                    pygame.draw.lines(
                        self.surface,
                        (min(255, r + cool),
                         min(255, g + cool),
                         min(255, b + cool),
                         alpha),
                        False,
                        path,
                        width
                    )

    def _draw_parabolas(self):
        for idx, (points, color) in enumerate(self.parabola_lines):
            points = self._orient_points(points)
            path = []

            for p in points:
                pt = get_2d((self.pos + p) - BlackHole.camera,
                            BlackHole.rx, BlackHole.ry)
                if pt:
                    path.append(pt)

            if len(path) < 2:
                continue
            r, g, b, a = color
            radial_factor = np.exp(- (idx / self.num_lines) ** 2 * 2.5)

            N_GLOW = 5
            for j in range(N_GLOW):
                t = j / N_GLOW
                width = int(2 + t * 10)
                alpha = int(a * radial_factor * np.exp(-3 * t))
                cool = int(30 * t)

                pygame.draw.lines(
                    self.surface,
                    (min(255, r + cool),
                     min(255, g + cool),
                     min(255, b + cool),
                     alpha),
                    False,
                    path,
                    width
                )


    def glow_circle(self, num_lines, radius):

        circles = []
        start_radius = radius
        end_radius = radius + 2


        for idx, i in enumerate(np.linspace(start_radius, end_radius, num_lines)):
            circle = self.circle_config(i)

            alpha = max(50, int(255 * np.exp(-0.025 * idx)))
            red, green, blue = self.get_color(idx)

            color = (red, green, blue, alpha)

            circles.append([circle, color])
        
        return circles
    
    def glow_ellipse(self, num_lines, a, b):

        ellipses = []
        start_radius = (a, b)
        end_radius = (a + 4, b+4)

        a_range = np.linspace(start_radius[0], end_radius[0], num_lines)
        b_range = np.linspace(start_radius[1], end_radius[1], num_lines)

        for idx, (a_i, b_i) in enumerate(zip(a_range, b_range)) :

            ellipse = self.ellipse_config(a_i, b_i)

            alpha = max(50, int(255 * np.exp(-0.025 * idx)))
            red, green, blue = self.get_color(idx)

            color = (red, green, blue, alpha)

            ellipses.append([ellipse, color])
        
        return ellipses

    def glow_parabolas(self, num_lines, a=0.08):
        curves = []
        a_vals = np.linspace(a, a * 1.8, num_lines)

        y_target = 0.0  # where all parabolas terminate visually

        for idx, a_i in enumerate(a_vals):
            x_extent = np.sqrt((y_target + 7) / a_i)

            upper = self.parabola_config(a=a_i, sign=1,
                                        offset=-7, x_extent=x_extent)
            lower = self.parabola_config(a=a_i, sign=-1,
                                        offset=-7, x_extent=x_extent)

            alpha = max(50, int(255 * np.exp(-0.05 * idx)))

            red, green, blue = self.get_color(idx)
            color = (red, green, blue, alpha)

            curves.append([upper, color])
            curves.append([lower, color])

        return curves

    def circle_config(self, radius):

        circle_points = []
        center = Vec3(0, 0, 0)

        for i in np.linspace(0, 2*np.pi, 50):

            x = radius * np.cos(i)
            y = radius * np.sin(i)

            z = center.z

            circle_points.append(Vec3(x, y, z))


        return circle_points
    
    def ellipse_config(self, a, b):
        ellipse_points = []
        center = Vec3(0, 0, 0)
        # a = self.base_radius
        # b = (6/8) * self.base_radius


        for i in np.linspace(0, 2*np.pi, 50):

            x = a * np.cos(i)
            z = b * np.sin(i)

            y = center.y

            ellipse_points.append(Vec3(x, y, z))

        return ellipse_points
    

    def parabola_config(self, a=0.08, x_extent=8, num_pts=60, sign=1, offset=2.0):
        points = []
        # x_extent = self.base_radius
        for idx, x in enumerate(np.linspace(-x_extent, x_extent, num_pts)):
            y = sign * (a * x * x) + sign * offset
            z = 0
            points.append(Vec3(x, y, z))
        return points




    def get_color(self, i):
        deep_orange = (255, 208, 0)
        white = (255, 255, 255)

        red_range = np.linspace(deep_orange[0], white[0], self.num_lines)
        green_range = np.linspace(deep_orange[1], white[1], self.num_lines)
        blue_range = np.linspace(deep_orange[2], white[2], self.num_lines)

        red = red_range[i]
        blue = blue_range[i]
        green = green_range[i]

        return red, green, blue 
    
    def rotate_vector(self, v: Vec3):
        """Rotate vector v by self.rotation (in radians) around x, y, z axes."""
        x, y, z = v.x, v.y, v.z
        rx, ry, rz = self.rotation.x, self.rotation.y, self.rotation.z

        # Around x-axis
        y, z = np.matmul(rotation_matrix(rx), np.array([y, z]))

        # Around y-axis
        z, x = np.matmul(rotation_matrix(ry), np.array([z, x]))

        # Around z-axis
        x, y = np.matmul(rotation_matrix(rz), np.array([x, y]))

        return Vec3(x, y, z)
    
    def _tilt_points(self, points):
        """Apply only R_tilt. Returns view-space points (z is depth from
        the camera relative to BH center, since the billboard makes view
        orientation = R_tilt · p_model). Use this for any depth-sort or
        front/back split."""
        cos_t, sin_t = math.cos(self.tilt_angle), math.sin(self.tilt_angle)
        out = []
        for p in points:
            x, y, z = p.x, p.y, p.z
            y, z = cos_t * y - sin_t * z, sin_t * y + cos_t * z
            out.append(Vec3(x, y, z))
        return out

    def _to_world(self, points):
        """Rotate view-space points back into world space by R_cam^-1, so
        get_2d's forward R_cam projects them to their intended view
        location. Call this after any depth-based splitting."""
        cos_rx, sin_rx = BlackHole._cos_rx, BlackHole._sin_rx
        cos_ry, sin_ry = BlackHole._cos_ry, BlackHole._sin_ry
        out = []
        for p in points:
            x, y, z = p.x, p.y, p.z
            y, z = cos_ry * y + sin_ry * z, -sin_ry * y + cos_ry * z
            x, z = cos_rx * x + sin_rx * z, -sin_rx * x + cos_rx * z
            out.append(Vec3(x, y, z))
        return out

    def _orient_points(self, points):
        """Convenience: full billboard transform R_cam^-1 · R_tilt."""
        return self._to_world(self._tilt_points(points))

    def split_ellipse_by_depth(self, points):
        """
        Splits ellipse polyline into back-facing (z < 0)
        and front-facing (z >= 0) segments. At each z=0 crossover the
        previous point is carried into the new segment so the bridging
        edge gets drawn — otherwise every ring has a visible radial gap
        at the two crossover angles.
        """
        back, front = [], []

        current = []
        current_is_front = None

        for p in points:
            is_front = p.z >= 0

            if current_is_front is None:
                current = [p]
                current_is_front = is_front
            elif is_front == current_is_front:
                current.append(p)
            else:
                if current_is_front:
                    front.append(current)
                else:
                    back.append(current)

                current = [current[-1], p]
                current_is_front = is_front

        if current:
            if current_is_front:
                front.append(current)
            else:
                back.append(current)

        return back, front

    def draw_event_horizon_cutout(self, radius_px=6.0):
        """Spherical event horizon silhouette: a circle in screen space
        (perpendicular to the disk). Radius is a model-space value and
        scales with 1/dist through the usual perspective factor."""
        center = get_2d(self.pos - BlackHole.camera, BlackHole.rx, BlackHole.ry)
        if not center:
            return
        dist = distance(self.pos, BlackHole.camera)
        radius = max(2, min(5000, int(radius_px * constants.DEPTH / dist)))

        cutout = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
        cutout.fill((0, 0, 0, 0))
        pygame.draw.circle(cutout, (255, 255, 255, 255), (radius + 1, radius + 1), radius)

        top_left = (center[0] - radius - 1, center[1] - radius - 1)
        self.surface.blit(cutout, top_left, special_flags=pygame.BLEND_ALPHA_SDL2)
        self.surface.blit(cutout, top_left, special_flags=pygame.BLEND_RGBA_SUB)

    def screen_occludes(self, star_pos):
        """
        Returns True if a star at star_pos is hidden behind the BH.
        """
        bh_2d = get_2d(self.pos - BlackHole.camera, BlackHole.rx, BlackHole.ry)
        star_2d = get_2d(star_pos - BlackHole.camera, BlackHole.rx, BlackHole.ry)

        if bh_2d is None or star_2d is None:
            return False

        # star must be farther than BH
        if distance(star_pos, BlackHole.camera) <= distance(self.pos, BlackHole.camera):
            return False

        # screen-space radius
        dist = distance(self.pos, BlackHole.camera)
        radius = max(2, int(7.0 * constants.DEPTH / dist))

        dx = star_2d[0] - bh_2d[0]
        dy = star_2d[1] - bh_2d[1]

        return dx*dx + dy*dy < radius*radius

    def draw_trail(self, screen): 
        obj_path_2d = [
                pt for pt in (get_2d(Vec3(*p) - BlackHole.camera, BlackHole.rx, BlackHole.ry) for p in self.trail_list) if pt ]
        
        if len(obj_path_2d) >1:
            pygame.draw.lines(screen, (255, 255, 255), False, obj_path_2d, 1)

    def draw_trail_color(self, trail_surface, speed_max, speed_min):
        if len(self.trail_list) < 2:
            return
        
        # Use only recent points so trails don't lag
        RECENT = 25
        points = self.trail_list[-RECENT:]

        obj_path_2d = []
        for p in points:
            pt = get_2d(Vec3(*p) - BlackHole.camera, BlackHole.rx, BlackHole.ry)
            if pt is not None:
                obj_path_2d.append(pt)

        if len(obj_path_2d) < 2:
            return

        # alpha based on velocity (much bigger scaling)
        vx, vy, vz = self.velocity
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)

        # --- Normalize speed into [0, 1] based on global min/max ---
        if speed_max > speed_min:
            rel = (speed - speed_min) / (speed_max - speed_min)
        else:
            rel = 0.0  # fallback if speeds identical

        # clamp exactly
        rel = max(0.0, min(1.0, rel))

        # --- Map rel → alpha (0 → 20, 1 → 255) ---
        alpha = int(20 + rel * (255 - 20))



        # Colorbar: blue (slow) → white (mid) → orange (fast)
        if rel < 0.5:
            # Map 0 → blue to 0.5 → white
            t = rel / 0.5
            rgb = self.lerp_color((0, 128, 255), (255, 255, 255), t)
        else:
            # Map 0.5 → white to 1 → orange
            t = (rel - 0.5) / 0.5
            rgb = self.lerp_color((255, 255, 255), (255, 165, 0), t)

        color = (*rgb, alpha)

        pygame.draw.lines(trail_surface, color, False, obj_path_2d, 1)

    def lerp_color(self, c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )
    
    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called in main_loop in astronim"""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry
        cls._cos_rx = math.cos(rx)
        cls._sin_rx = math.sin(rx)
        cls._cos_ry = math.cos(ry)
        cls._sin_ry = math.sin(ry)



