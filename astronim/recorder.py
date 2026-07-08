import os
import shutil
import pygame
import tempfile
import subprocess


class Recorder:
    '''Records the scene frame-by-frame and streams it to ffmpeg as an mp4 file.

    Frames are piped as raw RGB bytes into a long-lived ffmpeg subprocess, so
    encoding happens while the scene renders. No intermediate image files are
    written and pygame's optional image encoders are not used.

    Attributes
    ----------
    recording : bool
        Records the scene when True.
    frame_count : int
        The number of frames recorded. Initially zero.
    '''
    def __init__(self):
        self.recording = False
        self.frame_count = 0
        self.proc = None
        self.tmpfile = None

    def start(self):
        '''Begins recording the scene. The encoder is spawned lazily on the
        first save_frame(), when the surface size is known.
        '''
        self.recording = True

    def _spawn(self, width, height):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "Recording requires ffmpeg on PATH. Install it (e.g. `brew install ffmpeg`) and retry."
            )
        fd, self.tmpfile = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        self.proc = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-framerate", "60",
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            self.tmpfile,
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def save_frame(self, screen):
        '''Saves each frame in the scene.

        params
        ------
        screen : pygame.Surface
            The screen to capture.

        '''
        if not self.recording:
            return
        if self.proc is None:
            self._spawn(screen.get_width(), screen.get_height())
        self.proc.stdin.write(pygame.image.tobytes(screen, "RGB"))
        self.frame_count += 1

    def stop(self, output_file = "output.mp4"):
        '''Stops recording, finalizes the encode, and moves the video to
        output_file.

        params
        ------
        output_file : str, optional
            The file name to save to
        '''
        self.recording = False
        if self.proc is None:
            return
        self.proc.stdin.close()
        self.proc.wait()
        self.proc = None
        shutil.move(self.tmpfile, output_file)
        self.tmpfile = None
