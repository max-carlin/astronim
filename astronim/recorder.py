import os
import pygame
import tempfile
import subprocess

class Recorder: 
    def __init__(self):
        self.recording = False
        self.frame_count = 0
        self.tmpdir = None

    def start(self): 
        self.recording = True
        self.tmpdir = tempfile.TemporaryDirectory()

    def save_frame(self, screen): 
        if not self.recording: 
            return
        filename = os.path.join(self.tmpdir.name, f"frame_{self.frame_count:05d}.png")
        pygame.image.save(screen, filename)
        self.frame_count += 1

    def stop(self, output_file = "output.mp4"): 
        self.recording = False
        if self.tmpdir:
            frame_pattern = os.path.join(self.tmpdir.name, "frame_%05d.png")
            subprocess.run([
                "ffmpeg", "-framerate", "60", "-i", frame_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", output_file
            ])
            self.tmpdir.cleanup()
            self.tmpdir = None