"""Glowing retro figlet-banner text on a fixed 3D plane.

`AsciiText` renders a string as large FIGLET-style letterforms composed of
ASCII characters (/ \\ | _ ( ) ...), placed on a fixed plane in world space —
perspective changes as the camera moves, like `Grid`; it is not billboarded.
The look is CRT/neon: each figlet letter is drawn as one additive glowing
sprite (its ASCII rows rendered as monospace text with a blurred halo
underneath), with per-frame flicker, screen-space scanlines, and an
optional type-on reveal that pops in whole letters sequentially (same
frame-paced schedule as `Text`).

Scale: with the 6-row 'standard' figlet font, one source character is
roughly 8-10 cells wide, so a 10-char message spans ~95 columns —
about `base_size * 0.6 * 95` world units wide.

The letterform table below is generated once with pyfiglet (font
'standard') and embedded, so there is no runtime dependency.
"""

import math
import random
from bisect import bisect_right

import numpy as np
import pygame

from astronim.utils.tools import Vec3, get_camera_roll
from astronim.utils import constants


_FONT_HEIGHT = 6

# Figlet 'standard' letterforms, generated once with pyfiglet and
# embedded so there is no runtime dependency. Each glyph is a clean
# rectangle: uniform row width, exactly _FONT_HEIGHT rows.
_FONT = {
    'A': [
        '    _    ',
        '   / \\   ',
        '  / _ \\  ',
        ' / ___ \\ ',
        '/_/   \\_\\',
        '         ',
    ],
    'B': [
        ' ____  ',
        '| __ ) ',
        '|  _ \\ ',
        '| |_) |',
        '|____/ ',
        '       ',
    ],
    'C': [
        '  ____ ',
        ' / ___|',
        '| |    ',
        '| |___ ',
        ' \\____|',
        '       ',
    ],
    'D': [
        ' ____  ',
        '|  _ \\ ',
        '| | | |',
        '| |_| |',
        '|____/ ',
        '       ',
    ],
    'E': [
        ' _____ ',
        '| ____|',
        '|  _|  ',
        '| |___ ',
        '|_____|',
        '       ',
    ],
    'F': [
        ' _____ ',
        '|  ___|',
        '| |_   ',
        '|  _|  ',
        '|_|    ',
        '       ',
    ],
    'G': [
        '  ____ ',
        ' / ___|',
        '| |  _ ',
        '| |_| |',
        ' \\____|',
        '       ',
    ],
    'H': [
        ' _   _ ',
        '| | | |',
        '| |_| |',
        '|  _  |',
        '|_| |_|',
        '       ',
    ],
    'I': [
        ' ___ ',
        '|_ _|',
        ' | | ',
        ' | | ',
        '|___|',
        '     ',
    ],
    'J': [
        '     _ ',
        '    | |',
        ' _  | |',
        '| |_| |',
        ' \\___/ ',
        '       ',
    ],
    'K': [
        ' _  __',
        '| |/ /',
        "| ' / ",
        '| . \\ ',
        '|_|\\_\\',
        '      ',
    ],
    'L': [
        ' _     ',
        '| |    ',
        '| |    ',
        '| |___ ',
        '|_____|',
        '       ',
    ],
    'M': [
        ' __  __ ',
        '|  \\/  |',
        '| |\\/| |',
        '| |  | |',
        '|_|  |_|',
        '        ',
    ],
    'N': [
        ' _   _ ',
        '| \\ | |',
        '|  \\| |',
        '| |\\  |',
        '|_| \\_|',
        '       ',
    ],
    'O': [
        '  ___  ',
        ' / _ \\ ',
        '| | | |',
        '| |_| |',
        ' \\___/ ',
        '       ',
    ],
    'P': [
        ' ____  ',
        '|  _ \\ ',
        '| |_) |',
        '|  __/ ',
        '|_|    ',
        '       ',
    ],
    'Q': [
        '  ___  ',
        ' / _ \\ ',
        '| | | |',
        '| |_| |',
        ' \\__\\_\\',
        '       ',
    ],
    'R': [
        ' ____  ',
        '|  _ \\ ',
        '| |_) |',
        '|  _ < ',
        '|_| \\_\\',
        '       ',
    ],
    'S': [
        ' ____  ',
        '/ ___| ',
        '\\___ \\ ',
        ' ___) |',
        '|____/ ',
        '       ',
    ],
    'T': [
        ' _____ ',
        '|_   _|',
        '  | |  ',
        '  | |  ',
        '  |_|  ',
        '       ',
    ],
    'U': [
        ' _   _ ',
        '| | | |',
        '| | | |',
        '| |_| |',
        ' \\___/ ',
        '       ',
    ],
    'V': [
        '__     __',
        '\\ \\   / /',
        ' \\ \\ / / ',
        '  \\ V /  ',
        '   \\_/   ',
        '         ',
    ],
    'W': [
        '__        __',
        '\\ \\      / /',
        ' \\ \\ /\\ / / ',
        '  \\ V  V /  ',
        '   \\_/\\_/   ',
        '            ',
    ],
    'X': [
        '__  __',
        '\\ \\/ /',
        ' \\  / ',
        ' /  \\ ',
        '/_/\\_\\',
        '      ',
    ],
    'Y': [
        '__   __',
        '\\ \\ / /',
        ' \\ V / ',
        '  | |  ',
        '  |_|  ',
        '       ',
    ],
    'Z': [
        ' _____',
        '|__  /',
        '  / / ',
        ' / /_ ',
        '/____|',
        '      ',
    ],
    '0': [
        '  ___  ',
        ' / _ \\ ',
        '| | | |',
        '| |_| |',
        ' \\___/ ',
        '       ',
    ],
    '1': [
        ' _ ',
        '/ |',
        '| |',
        '| |',
        '|_|',
        '   ',
    ],
    '2': [
        ' ____  ',
        '|___ \\ ',
        '  __) |',
        ' / __/ ',
        '|_____|',
        '       ',
    ],
    '3': [
        ' _____ ',
        '|___ / ',
        '  |_ \\ ',
        ' ___) |',
        '|____/ ',
        '       ',
    ],
    '4': [
        ' _  _   ',
        '| || |  ',
        '| || |_ ',
        '|__   _|',
        '   |_|  ',
        '        ',
    ],
    '5': [
        ' ____  ',
        '| ___| ',
        '|___ \\ ',
        ' ___) |',
        '|____/ ',
        '       ',
    ],
    '6': [
        '  __   ',
        ' / /_  ',
        "| '_ \\ ",
        '| (_) |',
        ' \\___/ ',
        '       ',
    ],
    '7': [
        ' _____ ',
        '|___  |',
        '   / / ',
        '  / /  ',
        ' /_/   ',
        '       ',
    ],
    '8': [
        '  ___  ',
        ' ( _ ) ',
        ' / _ \\ ',
        '| (_) |',
        ' \\___/ ',
        '       ',
    ],
    '9': [
        '  ___  ',
        ' / _ \\ ',
        '| (_) |',
        ' \\__, |',
        '   /_/ ',
        '       ',
    ],
    '.': [
        '   ',
        '   ',
        '   ',
        ' _ ',
        '(_)',
        '   ',
    ],
    ',': [
        '   ',
        '   ',
        '   ',
        ' _ ',
        '( )',
        '|/ ',
    ],
    '!': [
        ' _ ',
        '| |',
        '| |',
        '|_|',
        '(_)',
        '   ',
    ],
    '?': [
        ' ___ ',
        '|__ \\',
        '  / /',
        ' |_| ',
        ' (_) ',
        '     ',
    ],
    ':': [
        '   ',
        ' _ ',
        '(_)',
        ' _ ',
        '(_)',
        '   ',
    ],
    '-': [
        '       ',
        '       ',
        ' _____ ',
        '|_____|',
        '       ',
        '       ',
    ],
    "'": [
        ' _ ',
        '( )',
        '|/ ',
        '   ',
        '   ',
        '   ',
    ],
    '(': [
        '  __',
        ' / /',
        '| | ',
        '| | ',
        '| | ',
        ' \\_\\',
    ],
    ')': [
        '__  ',
        '\\ \\ ',
        ' | |',
        ' | |',
        ' | |',
        '/_/ ',
    ],
    '+': [
        '       ',
        '   _   ',
        ' _| |_ ',
        '|_   _|',
        '  |_|  ',
        '       ',
    ],
    ' ': [
        '    ',
        '    ',
        '    ',
        '    ',
        '    ',
        '    ',
    ],
}


class AsciiText:
    """A glowing, retro ASCII-banner text object.

    Each figlet letter is baked as one sprite — its 6 rows rendered as
    monospace text so the ASCII characters connect exactly like a
    terminal — with a soft neon halo underneath. Letters are projected
    individually on the fixed text plane and scaled by their own depth.

    Parameters
    ----------
    message : str
        Text to render. Uppercase-folded for glyph lookup; unknown
        characters fall back to '?'. '\\n' starts a new banner line.
    pos : Vec3
        Center of the text plane in world units.
    color : tuple
        Base RGB color of the glow. Default is phosphor green.
    base_size : float
        World-unit height of one character cell (a figlet letter is
        _FONT_HEIGHT cells tall).
    letter_spacing / line_spacing : int
        Blank cell columns between letters / rows between lines.
    flicker : bool
        Per-frame CRT/neon brightness jitter (deterministic, so
        recordings are reproducible). `flicker_depth` sets the range.
    scanlines : bool
        Faint screen-space horizontal darkening over the text.
    glow : float
        Halo intensity multiplier (0 disables the bloom).
    type_out / wpm / typing_jitter / cursor
        Same frame-paced reveal machinery as `Text`, but whole figlet
        letters appear sequentially; the cursor is a solid glowing block.
    """

    # Class-level camera state (renderer's set_camera dispatch updates these)
    camera = Vec3(0.0, 0.0, 0.0)
    rx = 0.0
    ry = 0.0

    _FPS = 60.0
    _CURSOR_HZ = 2.0
    _CURSOR_LINGER_SEC = 1.0
    _CURSOR_COLS = 3

    # Shared caches. Sprites are grayscale (tint applied per-frame per
    # instance) so keys never include color or flicker state.
    _SPRITE_CACHE = {}         # (char|None, bucket, glow) -> premultiplied sprite
    _SPRITE_CACHE_MAX = 256
    _FONT_CACHE = {}           # bucket -> pygame Font
    _SCANLINE_CACHE = {}       # (w, h, strength) -> stripe surface
    # Geometric ~1.2x steps (cell height in px) so a camera dolly re-bakes
    # each letter at most len(_SIZE_BUCKETS) times, and one frame spans
    # only a few buckets.
    _SIZE_BUCKETS = np.array(
        [3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 24, 29,
         35, 42, 50, 60, 72, 86, 104, 125, 150, 180]
    )

    def __init__(
        self,
        message: str,
        pos: Vec3 = None,
        color=(51, 255, 102),
        base_size: float = 6.0,
        letter_spacing: int = 1,
        line_spacing: int = 1,
        flicker: bool = True,
        flicker_depth: float = 0.12,
        scanlines: bool = True,
        scanline_strength: float = 0.35,
        glow: float = 1.0,
        type_out: bool = False,
        wpm: float = 80.0,
        typing_jitter: float = 0.4,
        cursor: bool = True,
    ):
        pos = pos if pos is not None else Vec3(0.0, 0.0, 120.0)
        self.pos = Vec3(pos.x, pos.y, pos.z)
        self.message = message
        self.color = tuple(int(c) for c in color[:3])
        self.base_size = float(base_size)
        self.cell_h = self.base_size
        self.cell_w = 0.6 * self.base_size      # monospace aspect
        self.flicker = flicker
        self.scanlines = scanlines
        self.scanline_strength = float(scanline_strength)
        self.glow = round(float(glow), 2)
        self.type_out = type_out
        self.wpm = wpm
        self.typing_jitter = typing_jitter
        self.cursor = cursor
        self.static = True

        self._build_grid(message, max(0, int(letter_spacing)),
                         max(0, int(line_spacing)))

        # Deterministic flicker table: mostly near full brightness with a
        # quadratic bias, plus occasional deeper "neon stutter" dips smeared
        # over 2-3 consecutive frames so they read at 60 fps. Fixed seed so
        # re-recording a scene produces the identical video.
        rng = random.Random(0xA5C11)
        table = []
        while len(table) < 1024:
            if rng.random() < 0.02:
                depth = rng.uniform(0.55, 0.75)
                table.extend([depth] * rng.randint(2, 3))
            else:
                u = rng.random()
                table.append(1.0 - float(flicker_depth) * u * u)
        self._flicker_table = table[:1024]

        # Frame-paced type-on reveal schedule (same scheme as Text):
        # cum[i] = scene-time at which source character i becomes visible.
        if type_out and message:
            base = 60.0 / max(1.0, float(wpm) * 5.0)
            sigma = max(0.0, float(typing_jitter)) * 0.6
            cum = [0.0]
            for _ in range(len(message)):
                jitter = random.lognormvariate(0.0, sigma) if sigma > 0 else 1.0
                cum.append(cum[-1] + base * jitter)
            self._cum = cum
        else:
            self._cum = [0.0]

        self._frames_drawn = 0
        self._completed_at_frame = None
        self._scratch = None

    # ---------- Geometry ----------

    def _build_grid(self, message, letter_spacing, line_spacing):
        """Lay the message out as a banner grid. Two products:

        - Per-LETTER records for drawing: (src_index, glyph char, local
          center offset), sorted by src_index so the type-on visible set
          is a prefix slice.
        - Per-CELL arrays for scene morphs: every non-space banner cell
          as a local offset (one particle per lit cell).
        """
        fallback = _FONT.get('?')
        lines = message.split("\n")

        # Per line: glyph rows + per-source-char column spans
        line_grids = []       # (rows list[str], [(src_idx, glyph_ch, col_start, width)])
        flat_idx = 0
        for line in lines:
            rows = [""] * _FONT_HEIGHT
            spans = []
            for ch in line:
                key = ch.upper()
                glyph = _FONT.get(key)
                if glyph is None:
                    key = '?'
                    glyph = fallback
                spans.append((flat_idx, key, len(rows[0]), len(glyph[0])))
                for i in range(_FONT_HEIGHT):
                    rows[i] += glyph[i] + " " * letter_spacing
                flat_idx += 1
            line_grids.append((rows, spans))
            flat_idx += 1     # the '\n' itself counts as a source char

        total_cols = max((len(rows[0]) for rows, _ in line_grids), default=0)
        n_lines = len(line_grids)
        total_rows = n_lines * _FONT_HEIGHT + max(0, n_lines - 1) * line_spacing
        cx = (total_cols - 1) / 2.0
        cy = (total_rows - 1) / 2.0
        self._grid_center = (cx, cy)

        letters = []          # (src_idx, glyph_ch, center_row, center_col)
        cells = []            # (src_idx, row, col)
        # Cursor slots: local-grid (row_offset, col) of the block cursor
        # when the NEXT character to reveal is source index n.
        slots = {}
        for li, (rows, spans) in enumerate(line_grids):
            row_off = li * (_FONT_HEIGHT + line_spacing)
            col_off = (total_cols - len(rows[0])) // 2 if rows[0] else 0
            center_row = row_off + (_FONT_HEIGHT - 1) / 2.0
            for src_idx, glyph_ch, c0, width in spans:
                slots[src_idx] = (row_off, col_off + c0)
                lit = False
                for gi in range(_FONT_HEIGHT):
                    row_str = rows[gi]
                    for gj in range(c0, c0 + width):
                        if row_str[gj] != " ":
                            cells.append((src_idx, row_off + gi, col_off + gj))
                            lit = True
                if lit:       # spaces bake no sprite
                    letters.append((src_idx, glyph_ch, center_row,
                                    col_off + c0 + (width - 1) / 2.0))
            end_idx = spans[-1][0] + 1 if spans else (
                sum(len(l) + 1 for l in lines[:li]))
            slots[end_idx] = (row_off, col_off + len(rows[0]))
        if len(message) not in slots:
            slots[len(message)] = (
                (n_lines - 1) * (_FONT_HEIGHT + line_spacing), total_cols)
        self._cursor_slots = slots

        # Letter records (drawing)
        letters.sort(key=lambda l: l[0])
        self._letter_src = np.array([l[0] for l in letters], dtype=int)
        self._letter_chars = [l[1] for l in letters]
        self._letter_local = np.zeros((len(letters), 3), dtype=float)
        for i, (_, _, r, c) in enumerate(letters):
            self._letter_local[i, 0] = (c - cx) * self.cell_w
            self._letter_local[i, 1] = (cy - r) * self.cell_h

        # Cell arrays (morph particles)
        cells.sort(key=lambda c: c[0])
        self._src_index = np.array([c[0] for c in cells], dtype=int)
        self._local = np.zeros((len(cells), 3), dtype=float)
        if cells:
            rows_arr = np.array([c[1] for c in cells], dtype=float)
            cols_arr = np.array([c[2] for c in cells], dtype=float)
            self._local[:, 0] = (cols_arr - cx) * self.cell_w
            self._local[:, 1] = (cy - rows_arr) * self.cell_h

    def _cursor_center_local(self, next_src_idx):
        """(3,) local offset of the block cursor's center when the next
        character to reveal is source index `next_src_idx`."""
        slot = self._cursor_slots.get(next_src_idx)
        if slot is None:
            return None
        row_off, col = slot
        cx, cy = self._grid_center
        return np.array([
            (col + (self._CURSOR_COLS - 1) / 2.0 - cx) * self.cell_w,
            (cy - (row_off + (_FONT_HEIGHT - 1) / 2.0)) * self.cell_h,
            0.0,
        ])

    # ---------- Reveal state ----------

    @property
    def current_message(self):
        """The substring visible at this moment based on the reveal
        schedule. For non-type_out text, the full message."""
        if not self.type_out or len(self._cum) <= 1:
            return self.message
        scene_t = self._frames_drawn / self._FPS
        n = max(0, bisect_right(self._cum, scene_t) - 1)
        return self.message[:min(n, len(self.message))]

    def _visible_cell_count(self):
        return int(np.searchsorted(self._src_index,
                                   len(self.current_message)))

    # ---------- Drawing ----------

    def draw(self, screen):
        scene_t = self._frames_drawn / self._FPS

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

        frame_no = self._frames_drawn
        self._frames_drawn += 1

        k = int(np.searchsorted(self._letter_src, chars_visible))
        cursor_center = self._cursor_center_this_frame(scene_t, chars_visible)
        if k == 0 and cursor_center is None:
            return

        pts = self._letter_local[:k]
        if cursor_center is not None:
            pts = np.concatenate([pts, cursor_center[None, :]], axis=0)

        # Batched projection — same inline pattern as DarkMatter and Grid:
        # rx rotates (x, z), ry rotates (y, z), camera roll rotates the
        # projection-plane (x, y), then perspective divide.
        cam = AsciiText.camera
        cos_rx = math.cos(AsciiText.rx); sin_rx = math.sin(AsciiText.rx)
        cos_ry = math.cos(AsciiText.ry); sin_ry = math.sin(AsciiText.ry)

        x = pts[:, 0] + (self.pos.x - cam.x)
        y = pts[:, 1] + (self.pos.y - cam.y)
        z = pts[:, 2] + (self.pos.z - cam.z)

        x2 = cos_rx * x - sin_rx * z
        z = sin_rx * x + cos_rx * z
        x = x2

        y2 = cos_ry * y - sin_ry * z
        z = sin_ry * y + cos_ry * z
        y = y2

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
        w, h = constants.WIDTH, constants.HEIGHT
        sx = x * constants.DEPTH / z_safe + w / 2.0
        sy = h / 2.0 - y * constants.DEPTH / z_safe

        # Per-letter cell height in px -> size bucket
        h_px = self.cell_h * constants.DEPTH / z_safe
        buckets = AsciiText._SIZE_BUCKETS
        bucket_idx = np.minimum(np.searchsorted(buckets, h_px),
                                len(buckets) - 1)

        scratch = self._get_scratch()
        scratch.fill((0, 0, 0, 0))

        drew_any = False
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        drawn_h = []
        for i in range(pts.shape[0]):
            if not valid[i] or h_px[i] < 3:
                continue
            bucket = int(buckets[bucket_idx[i]])
            ch = self._letter_chars[i] if i < k else None   # None → cursor
            spr = self._letter_sprite(ch, bucket)
            sw, sh = spr.get_size()
            left = int(sx[i]) - sw // 2
            top = int(sy[i]) - sh // 2
            if left + sw < 0 or left > w or top + sh < 0 or top > h:
                continue
            scratch.blit(spr, (left, top),
                         special_flags=pygame.BLEND_RGBA_ADD)
            drew_any = True
            drawn_h.append(h_px[i])
            if left < min_x: min_x = left
            if top < min_y: min_y = top
            if left + sw > max_x: max_x = left + sw
            if top + sh > max_y: max_y = top + sh
        if not drew_any:
            return

        # Tint + flicker in one pass. Sprites are grayscale; this fill maps
        # them to the instance color scaled by the flicker multiplier.
        m = self._flicker_table[frame_no % 1024] if self.flicker else 1.0
        r, g, b = self.color
        scratch.fill((int(r * m), int(g * m), int(b * m), 255),
                     special_flags=pygame.BLEND_RGBA_MULT)

        # Screen-space scanlines over the text's bounding box only. The
        # stripe surface multiplies text pixels; everything else on the
        # scratch is (0,0,0,0) already. Skipped when the glyphs project
        # too small (moiré).
        if self.scanlines and float(np.median(drawn_h)) >= 5.0:
            bbox = pygame.Rect(int(min_x), int(min_y),
                               int(max_x - min_x), int(max_y - min_y))
            bbox = bbox.clip(pygame.Rect(0, 0, w, h))
            if bbox.width > 0 and bbox.height > 0:
                stripes = self._get_scanline_surface(w, h)
                scratch.blit(stripes, bbox.topleft, area=bbox,
                             special_flags=pygame.BLEND_RGBA_MULT)

        screen.blit(scratch, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _cursor_center_this_frame(self, scene_t, chars_visible):
        if not self.cursor or not self.type_out:
            return None
        typing_done = chars_visible >= len(self.message)
        if typing_done and self._completed_at_frame is not None:
            elapsed = (self._frames_drawn - 1 - self._completed_at_frame) / self._FPS
            if elapsed >= self._CURSOR_LINGER_SEC:
                return None
        phase = scene_t * self._CURSOR_HZ
        if (phase - int(phase)) >= 0.5:
            return None
        return self._cursor_center_local(min(chars_visible, len(self.message)))

    def _get_scratch(self):
        w, h = constants.WIDTH, constants.HEIGHT
        if self._scratch is None or self._scratch.get_size() != (w, h):
            self._scratch = pygame.Surface((w, h), pygame.SRCALPHA)
        return self._scratch

    # ---------- Sprite baking ----------

    def _letter_sprite(self, ch, bucket):
        """A premultiplied grayscale glow sprite for one whole figlet
        letter (`ch` is the message character) or, for `ch=None`, the
        solid block cursor. The letter's 6 rows are rendered as monospace
        text stacked at the font's line height, so the ASCII characters
        connect exactly as they would in a terminal; a smoothscale-blurred
        copy underneath provides the neon halo. Cached class-wide; keys
        exclude color/flicker by design."""
        key = (ch, bucket, self.glow)
        cache = AsciiText._SPRITE_CACHE
        spr = cache.get(key)
        if spr is not None:
            return spr

        core = self._render_core(ch, bucket)
        cw, ch_px = core.get_size()

        if self.glow > 0 and bucket >= 5:
            pad = max(2, int(bucket * 0.8))
            size = (cw + 2 * pad, ch_px + 2 * pad)
            spr = pygame.Surface(size, pygame.SRCALPHA)
            spr.fill((0, 0, 0, 0))
            # Halo: blur the core cheaply by scaling down 3x and back up,
            # then dim. Premultiplied RGB carries all the falloff —
            # BLEND_RGBA_ADD ignores source alpha downstream.
            halo = pygame.Surface(size, pygame.SRCALPHA)
            halo.fill((0, 0, 0, 0))
            halo.blit(core, (pad, pad))
            small = pygame.transform.smoothscale(
                halo, (max(1, size[0] // 3), max(1, size[1] // 3)))
            halo = pygame.transform.smoothscale(small, size)
            v = max(0, min(255, int(130 * self.glow)))
            halo.fill((v, v, v, 255), special_flags=pygame.BLEND_RGBA_MULT)
            spr.blit(halo, (0, 0))
            spr.blit(core, (pad, pad),
                     special_flags=pygame.BLEND_RGBA_ADD)
        else:
            spr = core

        if len(cache) >= AsciiText._SPRITE_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = spr
        return spr

    def _render_core(self, ch, bucket):
        """Sharp white core, premultiplied. For a letter: its glyph rows
        rendered line by line with a monospace font. For `ch=None` (the
        block cursor) or on any font failure: a solid rectangle."""
        font = self._get_font(bucket)
        if ch is not None:
            rows = _FONT.get(ch, _FONT['?'])
            try:
                line_h = font.get_linesize()
                row_surfs = [font.render(r, True, (255, 255, 255))
                             for r in rows]
                width = max(s.get_width() for s in row_surfs)
                core = pygame.Surface(
                    (max(1, width), max(1, line_h * len(rows))),
                    pygame.SRCALPHA)
                core.fill((0, 0, 0, 0))
                for i, s in enumerate(row_surfs):
                    core.blit(s, (0, i * line_h))
                return core.premul_alpha()
            except Exception:
                pass
        block_h = font.get_linesize() * _FONT_HEIGHT
        block = pygame.Surface(
            (max(1, int(bucket * 0.6 * self._CURSOR_COLS)), max(1, block_h)),
            pygame.SRCALPHA)
        block.fill((255, 255, 255, 255))
        return block

    @classmethod
    def _get_font(cls, bucket):
        font = cls._FONT_CACHE.get(bucket)
        if font is None:
            # Bold: at typical projected sizes (6-20 px) regular-weight
            # strokes are a single antialiased pixel and disappear.
            font = pygame.font.SysFont(
                'menlo,couriernew,courier,monospace', bucket, bold=True)
            cls._FONT_CACHE[bucket] = font
        return font

    def _get_scanline_surface(self, w, h):
        """Full-screen stripe surface: 3-px vertical period with one darkened
        row, multiplied over the text's bbox for the CRT banding."""
        key = (w, h, round(self.scanline_strength, 3))
        cache = AsciiText._SCANLINE_CACHE
        surf = cache.get(key)
        if surf is None:
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            surf.fill((255, 255, 255, 255))
            v = max(0, min(255, int(255 * (1.0 - self.scanline_strength))))
            for y in range(0, h, 3):
                surf.fill((v, v, v, 255), rect=pygame.Rect(0, y, w, 1))
            if len(cache) >= 4:
                cache.pop(next(iter(cache)))
            cache[key] = surf
        return surf

    def morph_particles(self):
        """(N, 6) [x, y, z, r, g, b] rows for scene-morph transitions —
        one particle per currently-visible lit banner cell.

        A type-out text that hasn't revealed anything yet (e.g. a freshly
        built scene being sampled as a morph TARGET) falls back to the
        FULL letter grid — same behavior as Text via _text_particles —
        so the morph flies particles into the complete word rather than
        into nothing."""
        k = self._visible_cell_count()
        if k == 0 and self.type_out:
            k = len(self._local)
        if k == 0:
            return np.zeros((0, 6), dtype=float)
        xyz = self._local[:k] + np.array(
            [self.pos.x, self.pos.y, self.pos.z])
        col = np.tile(np.asarray(self.color, dtype=float), (k, 1))
        return np.concatenate([xyz, col], axis=-1)

    @classmethod
    def set_camera(cls, camera, rx, ry):
        """Called by the renderer once per object per frame."""
        cls.camera = camera
        cls.rx = rx
        cls.ry = ry
