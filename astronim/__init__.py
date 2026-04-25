import os

# Must be set before pygame is imported (which happens via .universe below).
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

_BANNER = r"""
    _    ____ _____ ____   ___  _   _ ___ __  __
   / \  / ___|_   _|  _ \ / _ \| \ | |_ _|  \/  |
  / _ \ \___ \ | | | |_) | | | |  \| || || |\/| |
 / ___ \ ___) || | |  _ <| |_| | |\  || || |  | |
/_/   \_\____/ |_| |_| \_\\___/|_| \_|___|_|  |_|
© Max Carlin 2026. 
"""

if os.environ.get("ASTRONIM_HIDE_BANNER") != "1":
    print(_BANNER)

from .universe import Universe
from .objects import *
from .objects import __all__ as _object_names
from .utils.tools import Vec3
from .repl import interactive


__all__ = ["Universe", "Vec3", "interactive", *_object_names]
