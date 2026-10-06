"""Run dreamerv3-torch training without Windows suspending it underneath you.

    python scripts/train_dreamer.py --logdir ../dreamerv3-torch/logdir/walker_long -- \\
        --configs dmc_vision --task dmc_walker_walk --steps 500000 --eval_every 10000 \\
        --eval_episode_num 1 --log_every 5000 --envs 1 --video_pred_log False --train_ratio 128

Everything after `--` goes to the repo's dreamer.py unchanged. Resuming is automatic:
dreamer.py reloads `latest.pt` and the episodes in the logdir and carries on, so
rerunning the same command after an interruption continues where it stopped (losing
at most --eval_every env steps of weight updates).

Why this exists, learned the hard way: on a Modern Standby laptop, Windows suspends
background desktop processes whenever the display goes off, and this silently
stalled training for hours twice. `ES_SYSTEM_REQUIRED` alone does not prevent it
(a first version of this launcher used only that, and also ran training in a child
process). What does:

  * a power request of type *ExecutionRequired*, which exempts a process from that
    suspension — but it only covers the process that makes the request, so training
    runs *inside this process* (runpy) rather than as a child;
  * plus SystemRequired, and DisplayRequired so the screen never turns off and
    triggers standby in the first place (opt out with --allow-display-off).

No system setting is changed; every request is released when this process exits.
Keep the laptop plugged in. Closing the lid can still sleep it, depending on your
power plan.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import runpy
import sys
from ctypes import wintypes
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parent.parent.parent / "dreamerv3-torch"

# winnt.h POWER_REQUEST_TYPE
POWER_REQUEST_DISPLAY_REQUIRED = 0
POWER_REQUEST_SYSTEM_REQUIRED = 1
POWER_REQUEST_EXECUTION_REQUIRED = 3
POWER_REQUEST_CONTEXT_VERSION = 0
POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


class _ReasonContext(ctypes.Structure):
    # REASON_CONTEXT with the SimpleReasonString arm of its union.
    _fields_ = [("Version", wintypes.ULONG), ("Flags", wintypes.DWORD), ("SimpleReasonString", wintypes.LPWSTR)]


class _KeepAwake:
    """Holds Windows power requests for as long as it is open. No-op off Windows."""

    def __init__(self, allow_display_off: bool):
        self._handle = None
        self._active: list[int] = []
        self._kernel32 = None
        if sys.platform != "win32":
            return
        k = self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k.PowerCreateRequest.restype = wintypes.HANDLE
        k.PowerCreateRequest.argtypes = [ctypes.POINTER(_ReasonContext)]
        k.PowerSetRequest.restype = wintypes.BOOL
        k.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
        k.PowerClearRequest.restype = wintypes.BOOL
        k.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.SetThreadExecutionState.restype = wintypes.DWORD
        k.SetThreadExecutionState.argtypes = [wintypes.DWORD]

        context = _ReasonContext(POWER_REQUEST_CONTEXT_VERSION, POWER_REQUEST_CONTEXT_SIMPLE_STRING,
                                 "worldoptbench: Dreamer training is running")
        handle = k.PowerCreateRequest(ctypes.byref(context))
        if not handle or handle == wintypes.HANDLE(-1).value:
            print(f"WARNING: PowerCreateRequest failed (error {ctypes.get_last_error()}); relying on SetThreadExecutionState only", flush=True)
        else:
            self._handle = handle
            wanted = [POWER_REQUEST_EXECUTION_REQUIRED, POWER_REQUEST_SYSTEM_REQUIRED]
            if not allow_display_off:
                wanted.append(POWER_REQUEST_DISPLAY_REQUIRED)
            names = {0: "display", 1: "system", 3: "execution"}
            for kind in wanted:
                ok = bool(k.PowerSetRequest(handle, kind))
                if ok:
                    self._active.append(kind)
                print(f"power request '{names[kind]}-required': {'held' if ok else f'FAILED (error {ctypes.get_last_error()})'}", flush=True)
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | (0 if allow_display_off else ES_DISPLAY_REQUIRED)
        k.SetThreadExecutionState(flags)

    def release(self) -> None:
        k = self._kernel32
        if k is None:
            return
        for kind in self._active:
            k.PowerClearRequest(self._handle, kind)
        if self._handle:
            k.CloseHandle(self._handle)
        k.SetThreadExecutionState(ES_CONTINUOUS)
        self._active, self._handle = [], None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="dreamerv3-torch clone")
    ap.add_argument("--logdir", required=True, type=Path)
    ap.add_argument("--allow-display-off", action="store_true",
                    help="don't hold the display on (training is then protected only by the execution request)")
    ap.add_argument("dreamer_args", nargs=argparse.REMAINDER, help="after `--`: arguments for dreamer.py")
    args = ap.parse_args()
    extra = args.dreamer_args[1:] if args.dreamer_args[:1] == ["--"] else args.dreamer_args

    repo = args.repo.resolve()
    script = repo / "dreamer.py"
    if not script.exists():
        print(f"no dreamer.py in {repo}", file=sys.stderr)
        return 2

    awake = _KeepAwake(args.allow_display_off)
    try:
        # In-process (not a subprocess): the execution-required request only protects
        # the process that made it.
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        sys.argv = [str(script), *extra, "--logdir", str(args.logdir.resolve())]
        print("running dreamer.py in-process:", " ".join(sys.argv[1:]), flush=True)
        runpy.run_path(str(script), run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code or 0)
    finally:
        awake.release()


if __name__ == "__main__":
    raise SystemExit(main())
