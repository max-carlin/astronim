"""Scene dataclass for sequencing multi-scene videos.

A `Scene` pairs a build callback with a duration. `Universe.run_scenes`
plays them back-to-back as hard cuts onto a single recorded mp4 — see
`astronim/universe.py` for the runner.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Scene:
    build: Callable          # build(universe) -> None — populates simulation, optionally sets camera
    duration: float          # seconds of final video to spend in this scene (60 fps)
    name: str = ""           # identifier for log output / debugging
