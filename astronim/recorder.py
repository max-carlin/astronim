import os
import pygame
import tempfile
import subprocess


class Recorder:
    '''Records the scene frame-by-frame and compiles it into an mp4 file using ffmpeg. 

    Attributes
    ----------
    recording : bool
        Records the scene when True. 
    frame_count : int
        The number of frames recorded. Initially zero. 
    tmpdir : tempfile.TemporaryDirectory or None
        Temporary directory used to store image frames while recording. 
    ''' 
    def __init__(self):
        self.recording = False
        self.frame_count = 0
        self.tmpdir = None

    def start(self): 
        '''Begins recording the scene. Creates a temp directory. 
        '''
        self.recording = True
        self.tmpdir = tempfile.TemporaryDirectory()

    def save_frame(self, screen): 
        '''Saves each frame in the scene. 

        params
        ------
        screen : pygame.Surface 
            The screen to capture. 
        
        '''
        if not self.recording: 
            return
        filename = os.path.join(self.tmpdir.name, f"frame_{self.frame_count:05d}.png")
        pygame.image.save(screen, filename)
        self.frame_count += 1

    def stop(self, output_file = "output.mp4"): 
        '''Stops recording and compiles into video. 

        params
        ------
        output_file : str, optional
            The file name to save to
        '''
        self.recording = False
        if self.tmpdir:
            frame_pattern = os.path.join(self.tmpdir.name, "frame_%05d.png")
            subprocess.run([
                "ffmpeg", "-framerate", "60", "-i", frame_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", output_file
            ])
            self.tmpdir.cleanup()
            self.tmpdir = None