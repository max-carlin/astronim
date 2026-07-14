import pygame 
import numpy as np
import math
from astronim.utils.tools import distance, get_2d, Vec3
from astronim.utils.constants import DEPTH

class Star:
    def __init__(self, pos:Vec3, vel: Vec3, mass: float, radius: float = 0.01, color = (255, 255, 255), trail = False, static: bool = False):
        self.pos = pos
        self.velocity = [vel.x, vel.y, vel.z]
        self.mass = mass
        self.base_radius = radius
        self.color = color
        self.trail_list = []
        self.trail_length = 50
        self.trail = trail
        # Only set the attribute when True: Simulation.add treats a missing
        # `static` attribute as "dynamic" (AttributeError path), but an
        # object carrying static == False would fall through BOTH of add()'s
        # branches and be silently dropped from the simulation.
        if static:
            self.static = True

        

    def draw_glow_circle(self, surface, color, center, radius, glow_radius):
        
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)

        # multiple circles with decreasing alpha for glow
        for i in np.arange(glow_radius, radius, -1):
            t =  (i - radius) / (glow_radius - radius)
            alpha = int(255 * np.exp(-4 * t))
            pygame.draw.circle(glow_surf, (*color, alpha), (glow_radius, glow_radius), i)


        glow_rect = glow_surf.get_rect(center=center)
        surface.blit(glow_surf, glow_rect, special_flags = pygame.BLEND_ALPHA_SDL2 )

        pygame.draw.circle(surface, color, center, radius)




    def draw(self, screen):
        try:
            obj_pos_2d = get_2d(self.pos - Star.camera, Star.rx, Star.ry)
        except ValueError: 
            return
        if obj_pos_2d:
            dist = distance(self.pos, Star.camera)
            self.radius = max(2, min(20, int(self.base_radius * DEPTH / dist)))

            self.draw_glow_circle(
                screen, self.color, obj_pos_2d,
                radius=self.radius,
                glow_radius=self.radius * 8
            )
            pygame.draw.circle(screen, self.color, obj_pos_2d, 0.01)

    def draw_trail(self, screen):
        obj_path_2d = []
        for p in self.trail_list:
            try:
                pt = get_2d(Vec3(*p) - Star.camera, Star.rx, Star.ry)
                if pt is not None:
                    obj_path_2d.append(pt)
            except ValueError:
                continue
        
        if len(obj_path_2d) >1:
            pygame.draw.lines(screen, (255, 255, 255), False, obj_path_2d, 1)

    # def draw_trail(self, screen):
    #     obj_path_2d = []
    #     for p in self.trail_list:
    #         try:
    #             pt = get_2d(Vec3(*p) - Star.camera, Star.rx, Star.ry)
    #             if pt is not None:
    #                 obj_path_2d.append(pt)
    #         except ValueError:
    #             continue

    #     if len(obj_path_2d) > 1:

    #         trail_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    #         color = (167, 227, 252, 80)  # 80/255 alpha 

    #         pygame.draw.lines(trail_surf, color, False, obj_path_2d, 1)
    #         screen.blit(trail_surf, (0, 0))



    def draw_trail_color(self, trail_surface, speed_max, speed_min):
        if len(self.trail_list) < 2:
            return

        obj_path_2d = []
        for p in self.trail_list:
            pt = get_2d(Vec3(*p) - Star.camera, Star.rx, Star.ry)
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

