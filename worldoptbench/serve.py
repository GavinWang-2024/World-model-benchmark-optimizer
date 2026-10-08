"""worldserve: hold a world model plus an optimization stack in memory and serve generation over HTTP.

    worldserve --model wan --stack recommended --port 8000
    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/generate -d '{"prompt": "A red ball rolls down a ramp.", "horizon": 2, "seed": 0}'

Design, and what it deliberately is not:

* Standard library only (`http.server`), so it runs wherever the benchmark runs; no web-framework
  dependency. It is a small, correct server for one GPU, not a production gateway (no auth, no TLS,
  no streaming): put a proxy in front for that.
* One worker thread owns the model (a GPU runs one generation at a time). Requests queue in a bounded
  queue; a full queue answers 429 instead of growing without limit.
* Micro-batching: when the model has `generate_batch` (Dreamer) and `max_batch > 1`, queued requests
  that carry `init_video` and `actions` are grouped with `scheduling.bucket_requests` (identical shapes
  only) and run in one batched call. Text-to-video models run one request per call.
* Optional result cache (`cache_size > 0`) keyed by the request, the model and the active stack, so a
  repeated request returns the stored result and a different stack cannot return a stale one. Batched
  results are not cached: a batch draws its noise jointly, so a request's output depends on its batch.
* The "profile" returned with each result is latency, queue time, batch size and the active stack. A
  PAES score needs a baseline and a reference, which an arbitrary request does not have; compute it
  offline with the benchmark runner.

Request JSON (all optional except what the model needs): prompt, horizon, seed, init_frame, init_video,
actions, context_actions, params (extra model kwargs). Arrays travel as {"npy_b64": "<base64 of np.save>"}
(init_video: a list of them, or one stacked array). Response frames are base64 PNGs ("frames") when
Pillow is installed, else one base64 .npy ("frames_npy_b64"); pass "format": "npy" to force the latter.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import queue
import statistics
import threading
import time
from collections import deque
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

_MAX_BODY_BYTES = 64 * 1024 * 1024
_KNOWN_FIELDS = {"prompt", "horizon", "seed", "init_frame", "init_video", "actions", "context_actions", "params", "format"}


class ServerBusy(RuntimeError):
    """The request queue is full."""


class ServerStopped(RuntimeError):
    """The server is not running."""


# ---- wire encoding -------------------------------------------------------------------------------------


def decode_array(value: Any) -> Any:
    """{"npy_b64": ...} -> ndarray; a list is decoded element-wise; anything else is returned as is."""
    if isinstance(value, dict) and "npy_b64" in value:
        return np.load(io.BytesIO(base64.b64decode(value["npy_b64"])), allow_pickle=False)
    if isinstance(value, list) and value and all(isinstance(v, dict) and "npy_b64" in v for v in value):
        return [decode_array(v) for v in value]
    return value


def encode_array(array: Any) -> dict[str, str]:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return {"npy_b64": base64.b64encode(buffer.getvalue()).decode("ascii")}


def encode_frames(frames: list[Any], fmt: str | None = None) -> dict[str, Any]:
    """Frames as base64 PNGs (needs Pillow) or one base64 .npy; `fmt` "npy" forces the latter."""
    if fmt != "npy":
        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is not None:
            encoded = []
            for frame in frames:
                buffer = io.BytesIO()
                Image.fromarray(np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))).save(buffer, format="PNG")
                encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
            return {"frames": encoded, "frame_format": "png"}
    return {"frames_npy_b64": encode_array(np.stack([np.asarray(f) for f in frames]))["npy_b64"], "frame_format": "npy"}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


# ---- the server -----------------------------------------------------------------------------------------


class _Job:
    __slots__ = ("fmt", "future", "key", "kwargs", "request", "submitted")

    def __init__(self, request: dict, kwargs: dict, fmt: str | None, key: str | None):
        self.request, self.kwargs, self.fmt, self.key = request, kwargs, fmt, key
        self.future: Future = Future()
        self.submitted = time.perf_counter()


class WorldModelServer:
    """Loads nothing itself: pass an already-built model. `modules` / `module_kwargs` are an optimization
    stack (names from the library), or `recommended=True` for `defaults.recommended_config`."""

    def __init__(
        self,
        model: Any,
        modules: list[str] | None = None,
        module_kwargs: dict[str, dict[str, Any]] | None = None,
        *,
        stack: list[str] | None = None,
        recommended: bool = False,
        max_queue: int = 16,
        max_batch: int = 1,
        batch_wait: float = 0.01,
        cache_size: int = 0,
        constraints: Any = None,
        baseline_seconds_per_video_second: float | None = None,
        warmup: list[dict] | None = None,
    ):
        """`stack` is the outline's name for `modules` (give one or the other). `constraints` (a
        `worldoptbench.constraints.Constraints`) is reported by `info()`; at serve time only the speedup
        target can be checked, and only when `baseline_seconds_per_video_second` (the unoptimized latency per
        generated second, from a benchmark on this machine) is known: with it every response carries a
        `speedup`, and `profile()` says whether the running mean meets the target. The physics and VRAM limits
        cannot be verified per request (no reference, no per-request VRAM tracking) and are reported as such.

        `warmup` is a list of request dicts (same fields as /generate) run once, in order, inside `start()` before the
        server accepts traffic, so CUDA-graph capture, TensorRT engine builds and kernel autotuning happen at startup and
        not in the first real request's latency. Give one request per distinct shape you expect (each horizon captures its
        own graph). They are not counted in the profile, and an error in one stops `start()` with that error."""
        if stack is not None:
            if modules:
                raise ValueError("pass the module list as `modules` or as `stack`, not both")
            modules = list(stack)
        if max_queue < 1 or max_batch < 1 or batch_wait < 0 or cache_size < 0:
            raise ValueError("max_queue and max_batch must be >= 1, batch_wait and cache_size >= 0")
        if baseline_seconds_per_video_second is not None and baseline_seconds_per_video_second <= 0:
            raise ValueError("baseline_seconds_per_video_second must be positive")
        self._warmup = [self._normalize(r)[0] for r in (warmup or [])]
        self.constraints = constraints
        self.baseline_seconds_per_video_second = baseline_seconds_per_video_second
        self._speedups: deque[float] = deque(maxlen=1000)
        from worldoptbench.stack import OptimizationStack

        if recommended:
            from worldoptbench.defaults import recommended_config

            modules, module_kwargs = recommended_config(model)
        self._stack = OptimizationStack(model, modules or [], module_kwargs=module_kwargs)
        self.model = self._stack.apply()
        self.max_batch, self.batch_wait = max_batch, batch_wait
        self._queue: queue.Queue[_Job | None] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._started_at = time.time()
        self._latencies: deque[float] = deque(maxlen=1000)
        self._counts = {"requests": 0, "errors": 0, "cache_hits": 0, "batches": 0}
        self._counts_lock = threading.Lock()
        self._cache: dict[str, dict] | None = {} if cache_size else None
        self._cache_order: deque[str] = deque()
        self._cache_size = cache_size
        self._namespace = f"{self.model.get_info().name}|{self._stack.name}"

    # -- lifecycle --

    def start(self) -> WorldModelServer:
        if self._thread is None:
            for kwargs in self._warmup:  # before any request is accepted; not counted in the profile
                self.model.generate(**kwargs)
            self._stopped.clear()
            self._thread = threading.Thread(target=self._work, name="worldserve-worker", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stops accepting work, fails whatever is still queued, restores the model."""
        self._stopped.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is not None and not job.future.done():
                job.future.set_exception(ServerStopped("server stopped before this request ran"))
        self._stack.restore()

    # -- requests --

    def _normalize(self, request: dict) -> tuple[dict, str | None]:
        if not isinstance(request, dict):
            raise TypeError("request must be a JSON object")
        unknown = set(request) - _KNOWN_FIELDS
        if unknown:
            raise ValueError(f"unknown request fields {sorted(unknown)}; allowed: {sorted(_KNOWN_FIELDS)}")
        kwargs: dict[str, Any] = dict(request.get("params") or {})
        for field in ("prompt", "horizon", "seed", "init_frame", "init_video", "actions", "context_actions"):
            if field in request and request[field] is not None:
                kwargs[field] = decode_array(request[field])
        if "horizon" in kwargs:
            kwargs["horizon"] = float(kwargs["horizon"])
        if "seed" in kwargs:
            kwargs["seed"] = int(kwargs["seed"])
        return kwargs, request.get("format")

    def _cache_key(self, request: dict) -> str:
        return hashlib.sha1((self._namespace + "|" + json.dumps(request, sort_keys=True, default=str)).encode()).hexdigest()

    def submit(self, request: dict) -> Future:
        """Queues a request; the future resolves to the response dict. Raises ServerBusy if the queue is full."""
        if self._thread is None or self._stopped.is_set():
            raise ServerStopped("server is not running (call start())")
        kwargs, fmt = self._normalize(request)
        key = self._cache_key(request) if self._cache is not None else None
        if key is not None and key in self._cache:
            with self._counts_lock:
                self._counts["requests"] += 1
                self._counts["cache_hits"] += 1
            future: Future = Future()
            future.set_result({**self._cache[key], "cached": True, "queue_seconds": 0.0})
            return future
        job = _Job(request, kwargs, fmt, key)
        try:
            self._queue.put_nowait(job)
        except queue.Full as error:
            raise ServerBusy(f"request queue is full ({self._queue.maxsize})") from error
        return job.future

    def generate(self, request: dict, timeout: float | None = None) -> dict:
        return self.submit(request).result(timeout)

    # -- the worker --

    def _sync(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001, S110  (no torch, or no CUDA: nothing to wait for)
            pass

    def _batchable(self, job: _Job) -> bool:
        return (
            self.max_batch > 1 and hasattr(self.model, "generate_batch")
            and "init_video" in job.kwargs and "actions" in job.kwargs and "horizon" in job.kwargs
        )

    def _work(self) -> None:
        while not self._stopped.is_set():
            job = self._queue.get()
            if job is None:
                break
            group = [job]
            if self._batchable(job):
                deadline = time.perf_counter() + self.batch_wait
                while len(group) < self.max_batch:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        extra = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if extra is None:
                        self._stopped.set()
                        break
                    group.append(extra)
            self._run_group(group)

    def _run_group(self, group: list[_Job]) -> None:
        batches: list[list[_Job]] = []
        batchable = [j for j in group if self._batchable(j)]
        loose = [j for j in group if not self._batchable(j)]
        if batchable:
            from worldoptbench.scheduling import bucket_requests

            for indices in bucket_requests([j.kwargs for j in batchable], max_batch=self.max_batch):
                batches.append([batchable[i] for i in indices])
        batches.extend([j] for j in loose)
        for batch in batches:
            started = time.perf_counter()
            try:
                self._sync()
                if len(batch) > 1:
                    seed = batch[0].kwargs.get("seed", 0)
                    rollouts = self.model.generate_batch(
                        [{k: v for k, v in j.kwargs.items() if k != "seed"} for j in batch], seed=seed
                    )
                else:
                    rollouts = [self.model.generate(**batch[0].kwargs)]
                self._sync()
                elapsed = time.perf_counter() - started
            except Exception as error:  # noqa: BLE001  (reported to the caller, never kills the worker)
                with self._counts_lock:
                    self._counts["errors"] += len(batch)
                for job in batch:
                    job.future.set_exception(error)
                continue
            with self._counts_lock:
                self._counts["requests"] += len(batch)
                self._counts["batches"] += 1
                self._latencies.append(elapsed)
            for job, rollout in zip(batch, rollouts):
                response = {
                    **encode_frames(rollout.frames, job.fmt),
                    "fps": rollout.fps,
                    "num_frames": len(rollout.frames),
                    "latency_seconds": elapsed,
                    "queue_seconds": started - job.submitted,
                    "batch_size": len(batch),
                    "speedup": self._speedup(elapsed / len(batch), job.request.get("horizon", rollout.metadata.get("horizon"))),
                    "cached": False,
                    "metadata": _json_safe(rollout.metadata),
                    "stack": self.stack_report(),
                }
                if job.key is not None and len(batch) == 1:
                    self._remember(job.key, response)
                job.future.set_result(response)

    def _speedup(self, latency: float, horizon: Any) -> float | None:
        """baseline / measured latency per generated second, or None if no baseline or horizon is known."""
        if self.baseline_seconds_per_video_second is None or not horizon or latency <= 0:
            return None
        value = self.baseline_seconds_per_video_second / (latency / float(horizon))
        self._speedups.append(value)
        return value

    def _remember(self, key: str, response: dict) -> None:
        assert self._cache is not None
        if key not in self._cache:
            self._cache_order.append(key)
            while len(self._cache_order) > self._cache_size:
                self._cache.pop(self._cache_order.popleft(), None)
        self._cache[key] = response

    # -- introspection --

    def stack_report(self) -> dict[str, Any]:
        return {
            "applied": [m.name for m in self._stack.modules],
            "skipped": [{"name": s.name, "reason": s.reason} for s in self._stack.skipped],
        }

    def constraints_status(self) -> dict[str, Any] | None:
        """What can be said about the constraints while serving: the speedup target against the running mean
        speedup; the physics and VRAM limits as `unverifiable here`."""
        c = self.constraints
        if c is None:
            return None
        status: dict[str, Any] = {}
        if c.target_speedup is not None:
            mean = sum(self._speedups) / len(self._speedups) if self._speedups else None
            status["target_speedup"] = {
                "required": c.target_speedup, "running_mean": mean,
                "met": None if mean is None else mean >= c.target_speedup,
            }
        for name in ("min_physics_score", "max_vram_gb"):
            if getattr(c, name) is not None:
                status[name] = {"required": getattr(c, name), "met": None, "note": "cannot be verified per request; check offline with the benchmark"}
        return status

    def serve(self, host: str = "127.0.0.1", port: int = 8000, block: bool = True, timeout: float = 600.0):
        """Starts the worker and an HTTP server on host:port (the outline's `server.serve(port=8000)`). With
        `block=True` (default) it runs until interrupted, then shuts down; with `block=False` it returns the
        running `ThreadingHTTPServer` (call `.shutdown()` and `.server_close()` when done; `stop()` the server too)."""
        self.start()
        httpd = make_http_server(self, host, port, timeout=timeout)
        if not block:
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            return httpd
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
            self.stop()
        return None

    def info(self) -> dict[str, Any]:
        info = self.model.get_info()
        return {
            "constraints": None if self.constraints is None else {k: v for k, v in vars(self.constraints).items() if v is not None},
            "model": info.name, "architecture": info.architecture, "param_count": info.param_count,
            "stack": self.stack_report(), "max_batch": self.max_batch, "max_queue": self._queue.maxsize,
        }

    def profile(self) -> dict[str, Any]:
        with self._counts_lock:
            latencies = sorted(self._latencies)
            counts = dict(self._counts)

        def percentile(q: float) -> float | None:
            return latencies[min(len(latencies) - 1, int(q * len(latencies)))] if latencies else None

        return {
            **counts,
            "speedup_mean": (sum(self._speedups) / len(self._speedups)) if self._speedups else None,
            "constraints": self.constraints_status(),
            "queue_depth": self._queue.qsize(),
            "uptime_seconds": time.time() - self._started_at,
            "latency_mean_seconds": statistics.fmean(latencies) if latencies else None,
            "latency_p50_seconds": percentile(0.5),
            "latency_p95_seconds": percentile(0.95),
        }


# ---- HTTP -----------------------------------------------------------------------------------------------


def make_http_server(server: WorldModelServer, host: str = "127.0.0.1", port: int = 8000, timeout: float = 600.0):
    """A ThreadingHTTPServer in front of `server` (call `.serve_forever()`; port 0 picks a free port)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            routes = {"/health": lambda: {"status": "ok", "queue_depth": server._queue.qsize()},
                      "/info": server.info, "/profile": server.profile}
            handler = routes.get(self.path.split("?")[0])
            self._send(200, handler()) if handler else self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:
            if self.path.split("?")[0] != "/generate":
                return self._send(404, {"error": f"no route {self.path}"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > _MAX_BODY_BYTES:
                return self._send(413, {"error": f"body larger than {_MAX_BODY_BYTES} bytes"})
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as error:
                return self._send(400, {"error": f"invalid JSON: {error}"})
            try:
                self._send(200, server.generate(request, timeout=timeout))
            except ServerBusy as error:
                self._send(429, {"error": str(error)})
            except ServerStopped as error:
                self._send(503, {"error": str(error)})
            except (ValueError, TypeError, KeyError) as error:
                self._send(400, {"error": f"{type(error).__name__}: {error}"})
            except Exception as error:  # noqa: BLE001
                self._send(500, {"error": f"{type(error).__name__}: {error}"})

    return ThreadingHTTPServer((host, port), Handler)


# ---- command line ---------------------------------------------------------------------------------------


def build_model(name: str, kwargs: dict[str, Any]) -> Any:
    if name == "wan":
        from worldoptbench.models.wan_video import WanVideo

        return WanVideo(**kwargs)
    if name == "dreamer":
        from worldoptbench.models.dreamer import DreamerRepo, DreamerWorldModel

        kwargs = dict(kwargs)
        checkpoint = kwargs.pop("checkpoint_dir")
        return DreamerWorldModel(DreamerRepo(kwargs.pop("repo_path"), **kwargs), checkpoint)
    raise ValueError(f"unknown model {name!r}; choose wan or dreamer")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="worldserve", description="Serve a world model with an optimization stack over HTTP.")
    ap.add_argument("--model", required=True, choices=["wan", "dreamer"])
    ap.add_argument("--model-kwargs", default="{}", help='JSON; dreamer needs {"repo_path": ..., "checkpoint_dir": ...}')
    ap.add_argument("--stack", default="recommended", help="'recommended', 'none', or comma-separated module names")
    ap.add_argument("--stack-kwargs", default="{}", help="JSON: per-module constructor kwargs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-queue", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--cache-size", type=int, default=0)
    args = ap.parse_args(argv)

    model = build_model(args.model, json.loads(args.model_kwargs))
    recommended = args.stack == "recommended"
    modules = [] if args.stack in ("none", "recommended") else args.stack.split(",")
    server = WorldModelServer(
        model, modules, json.loads(args.stack_kwargs), recommended=recommended,
        max_queue=args.max_queue, max_batch=args.max_batch, cache_size=args.cache_size,
    ).start()
    httpd = make_http_server(server, args.host, args.port)
    print(f"worldserve: {server.info()['model']} with stack {server.stack_report()['applied']} on http://{args.host}:{httpd.server_address[1]}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
