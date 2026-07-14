import random
from bisect import bisect_right

import pygame
from astronim.utils.tools import distance, get_2d, Vec3
from astronim.utils import constants


class Text:
    """A 3D-positioned text label, optionally typed out one character at
    a time.

    Type-out is paced by frame count (one draw call = one output frame
    at 60 fps), not wall-clock, so a recording at 4K types at the same
    apparent speed as one at 1080p. Per-character delays carry
    lognormal jitter for human-like rhythm.
    """

    # Class-level camera state (renderer's set_camera dispatch updates these)
    camera = Vec3(0, 0, 0)
    rx = 0.0
    ry = 0.0

    _FPS = 60.0                  # frame-counted scene-time conversion
    _CURSOR_HZ = 2.0             # blinks per second
    _CURSOR_LINGER_SEC = 1.0     # cursor stays this long after typing finishes
    _CURSOR_GLYPH = "|"

    def __init__(
        self,
        message: str,
        pos: Vec3 = Vec3(0, 0, 50),
        base_size: int = 32,
        color=(255, 255, 255),
        type_out: bool = False,
        wpm: float = 80.0,
        typing_jitter: float = 0.4,
        cursor: bool = True,
        fade_in: float = 0.0,      # seconds to fade from invisible to full
        fade_delay: float = 0.0,   # seconds to wait before the fade starts
    ):
        self.pos = pos
        self.message = message
        self.base_size = base_size
        self.color = color
        self.type_out = type_out
        self.wpm = wpm
        self.typing_jitter = typing_jitter
        self.cursor = cursor
        self.fade_in = float(fade_in)
        self.fade_delay = float(fade_delay)
        self.static = True

        # Pre-bake the cumulative reveal schedule. cum[i] = scene-time
        # (seconds) at which character i becomes visible. Lognormal
        # jitter scaled by typing_jitter gives realistic rhythm.
        if type_out and message:
            base = 60.0 / max(1.0, float(wpm) * 5.0)   # sec/char from wpm (5 ch/word)
            sigma = max(0.0, float(typing_jitter)) * 0.6
            cum = [0.0]
            for _ in range(len(message)):
                jitter = random.lognormvariate(0.0, sigma) if sigma > 0 else 1.0
                cum.append(cum[-1] + base * jitter)
            self._cum = cum
            self._total_duration = cum[-1]
        else:
            self._cum = [0.0]
            self._total_duration = 0.0

        # Per-instance frame counter, advances on each draw().
        self._frames_drawn = 0
        self._completed_at_frame = None

    @property
    def current_message(self):
        """The substring visible at this moment based on the reveal
        schedule. For non-type_out text, the full message."""
        if not self.type_out or len(self._cum) <= 1:
            return self.message
        scene_t = self._frames_drawn / self._FPS
        n = max(0, bisect_right(self._cum, scene_t) - 1)
        return self.message[:min(n, len(self.message))]

    def draw(self, screen):
        # Frame-counted scene time
        scene_t = self._frames_drawn / self._FPS

        # How many characters are visible this frame
        if self.type_out and len(self._cum) > 1:
            chars_visible = max(0, bisect_right(self._cum, scene_t) - 1)
            chars_visible = min(chars_visible, len(self.message))
        else:
            chars_visible = len(self.message)

        if (
            self.type_out
            and chars_visible >= len(self.message)
            and self._completed_at_frame is None
        ):
            self._completed_at_frame = self._frames_drawn

        visible = self.message[:chars_visible]
        cursor_glyph = self._cursor_glyph_this_frame(scene_t, chars_visible)

        # Bump the frame counter for the next draw
        self._frames_drawn += 1

        # Fade-in envelope (frame-paced like everything else)
        alpha = 255
        if self.fade_in > 0.0 or self.fade_delay > 0.0:
            t = scene_t - self.fade_delay
            if t <= 0.0:
                return
            if self.fade_in > 0.0:
                alpha = int(255 * min(1.0, t / self.fade_in))

        if not visible and not cursor_glyph:
            return

        # Distance-scaled font size (same as before)
        dist = distance(self.pos, Text.camera)
        size = max(2, min(5000, int(self.base_size * constants.DEPTH / dist)))
        font = pygame.font.SysFont('Times New Roman', size)

        # Project to 2D. If the text would be behind the camera, get_2d returns None.
        text_pos = get_2d(self.pos - Text.camera, Text.rx, Text.ry)
        if not text_pos:
            return

        if self.type_out:
            # Left-align the typed portion at where the FULL string's left
            # edge will end up, so characters appear in place rather than
            # the whole label expanding from its center as it types.
            full_surface = font.render(self.message, True, self.color)
            full_w, full_h = full_surface.get_size()
            partial_surface = font.render(visible + cursor_glyph, True, self.color)
            if alpha < 255:
                partial_surface.set_alpha(alpha)
            left = int(text_pos[0] - full_w / 2)
            top = int(text_pos[1] - full_h / 2)
            screen.blit(partial_surface, (left, top))
        else:
            text_surface = font.render(self.message + cursor_glyph, True, self.color)
            if alpha < 255:
                text_surface.set_alpha(alpha)
            rect = text_surface.get_rect()
            rect.center = text_pos
            screen.blit(text_surface, rect)

    def _cursor_glyph_this_frame(self, scene_t: float, chars_visible: int) -> str:
        if not self.cursor or not self.type_out:
            return ""
        typing_done = chars_visible >= len(self.message)
        if typing_done and self._completed_at_frame is not None:
            elapsed_post = (self._frames_drawn - self._completed_at_frame) / self._FPS
            if elapsed_post >= self._CURSOR_LINGER_SEC:
                return ""
        # Blink at _CURSOR_HZ (50% duty cycle)
        phase = scene_t * self._CURSOR_HZ
        return self._CURSOR_GLYPH if (phase - int(phase)) < 0.5 else ""

    def attach_to(self):
        """Attaches the text object to another object."""
        pass

    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called by the renderer once per frame."""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry
