"""Interactive REPL mode for AstroAnim.

`astronim.interactive()` enters a faux-REPL where typed Python statements
execute against a live pygame scene. The REPL runs on a daemon thread; the
pygame main loop runs on the main thread (required by SDL/Cocoa on macOS),
drains a queue of compiled statements each frame, and ticks whichever
``Universe`` was most recently constructed.
"""

import __main__
import code
import queue
import sys
import threading
import time
import traceback
import weakref

import pygame

_cmd_queue: "queue.Queue" = queue.Queue()
_active_ref: "weakref.ReferenceType | None" = None
_repl_alive = threading.Event()


def _set_active(universe):
    """Record the most recently constructed Universe so the pump can tick it."""
    global _active_ref
    _active_ref = weakref.ref(universe)


def _get_active():
    if _active_ref is None:
        return None
    return _active_ref()


class _QueueingConsole(code.InteractiveConsole):
    """InteractiveConsole that ships compiled code objects to the main thread
    instead of executing them locally. The console still handles line
    buffering, prompts (``>>>``/``...``), and syntax-error display."""

    def runcode(self, code_obj):
        # Called by the base class after compile() succeeds.
        _cmd_queue.put(code_obj)


def _repl_thread(namespace):
    console = _QueueingConsole(locals=namespace)
    try:
        console.interact(banner="")
    except SystemExit:
        pass
    finally:
        _repl_alive.clear()


def _drain_queue(namespace):
    while True:
        try:
            code_obj = _cmd_queue.get_nowait()
        except queue.Empty:
            return
        try:
            exec(code_obj, namespace)
        except SystemExit:
            _repl_alive.clear()
            return
        except BaseException:
            traceback.print_exc()


def interactive():
    """Enter interactive REPL mode. Blocks on the main thread forever."""
    namespace = __main__.__dict__

    import astronim  # avoid top-level cycle
    for name in astronim.__all__:
        namespace.setdefault(name, getattr(astronim, name))

    _repl_alive.set()
    thread = threading.Thread(
        target=_repl_thread, args=(namespace,), daemon=True, name="astronim-repl"
    )
    thread.start()

    frame_period = 1.0 / 60.0
    try:
        while _repl_alive.is_set():
            frame_start = time.monotonic()
            _drain_queue(namespace)

            u = _get_active()
            if u is not None and getattr(u, "running", False):
                try:
                    u.tick()
                except BaseException:
                    traceback.print_exc()
                    u.running = False
            # No Universe yet (or window was closed): stay idle; do NOT call
            # pygame.event.pump() because pygame.display hasn't been
            # initialized until the first Universe is constructed.

            elapsed = time.monotonic() - frame_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
    except BaseException:
        traceback.print_exc()
    finally:
        try:
            if pygame.get_init():
                pygame.quit()
        except Exception:
            pass
        sys.exit(0)
