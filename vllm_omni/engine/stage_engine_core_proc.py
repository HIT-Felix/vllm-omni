"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import signal
import time
from contextlib import contextmanager
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import (
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    SignalCallback,
    get_engine_zmq_addresses,
)
from vllm.v1.utils import shutdown

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor

logger = init_logger(__name__)

_WORKER_STARTUP_BREAKDOWN: dict[str, float | int] = {}
_WORKER_PROBES_INSTALLED = False


def _record_worker_startup_metric(name: str, elapsed_ms: float) -> None:
    _WORKER_STARTUP_BREAKDOWN[name] = float(_WORKER_STARTUP_BREAKDOWN.get(name, 0.0)) + elapsed_ms
    count_key = f"{name}_count"
    _WORKER_STARTUP_BREAKDOWN[count_key] = int(_WORKER_STARTUP_BREAKDOWN.get(count_key, 0)) + 1


def _reset_worker_startup_breakdown() -> None:
    _WORKER_STARTUP_BREAKDOWN.clear()


def _get_worker_startup_breakdown() -> dict[str, float | int]:
    return dict(_WORKER_STARTUP_BREAKDOWN)


def _install_worker_startup_probes() -> None:
    """Install one-shot Worker method probes inside the stage subprocess.

    These hooks let us split the current kv_cache_init bucket into more
    actionable pieces for cold-start analysis on vLLM 0.19.x.
    """
    global _WORKER_PROBES_INSTALLED
    if _WORKER_PROBES_INSTALLED:
        return

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:
        logger.warning(
            "[StageEngineCoreProc] Failed to import vllm.v1.worker.gpu_worker.Worker; "
            "kv_cache_init_breakdown will be unavailable."
        )
        return

    def _wrap_timed_method(method_name: str, metric_name: str) -> None:
        if not hasattr(Worker, method_name):
            logger.warning(
                "[StageEngineCoreProc] Worker.%s missing; %s will be unavailable.",
                method_name,
                metric_name,
            )
            return

        original = getattr(Worker, method_name)

        def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            try:
                return original(self, *args, **kwargs)
            finally:
                _record_worker_startup_metric(metric_name, (time.monotonic() - start_time) * 1000.0)

        setattr(Worker, method_name, _wrapped)

    _wrap_timed_method("load_model", "weight_load_ms")
    _wrap_timed_method("determine_available_memory", "determine_available_memory_ms")
    _wrap_timed_method("profile", "profile_ms")
    _wrap_timed_method("execute_dummy_batch", "dummy_batch_ms")
    _wrap_timed_method("initialize_from_config", "kv_cache_alloc_ms")
    _wrap_timed_method("compile_or_warm_up_model", "cudagraph_capture_ms")

    _WORKER_PROBES_INSTALLED = True


def _round_nested_metrics(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _round_nested_metrics(nested_value) for key, nested_value in sorted(value.items())}
    return value


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._omni_handshake_metrics: dict[str, Any] = {}
        self._omni_init_received_monotonic: float | None = None
        super().__init__(*args, **kwargs)

    def _initialize_kv_caches(self, vllm_config: Any) -> Any:
        if self._omni_init_received_monotonic is not None:
            self._omni_handshake_metrics["pre_kv_cache_init_ms"] = (
                time.monotonic() - self._omni_init_received_monotonic
            ) * 1000.0

        kv_cache_init_start = time.monotonic()
        kv_cache_config = super()._initialize_kv_caches(vllm_config)
        self._omni_handshake_metrics["kv_cache_init_ms"] = (time.monotonic() - kv_cache_init_start) * 1000.0
        worker_breakdown = _get_worker_startup_breakdown()
        if worker_breakdown:
            self._omni_handshake_metrics["kv_cache_init_breakdown"] = worker_breakdown
        return kv_cache_config

    def startup_handshake(
        self,
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: Any = None,
    ) -> EngineZmqAddresses:
        hello_send_start = time.monotonic()
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )
        self._omni_handshake_metrics["hello_send_ms"] = (time.monotonic() - hello_send_start) * 1000.0

        wait_for_init_start = time.monotonic()
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=600 * 1000):
            raise RuntimeError("Did not receive response from front-end process within 600 minutes")
        init_bytes = handshake_socket.recv()
        self._omni_handshake_metrics["wait_for_init_msg_ms"] = (time.monotonic() - wait_for_init_start) * 1000.0
        self._omni_init_received_monotonic = time.monotonic()

        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(init_bytes, type=EngineHandshakeMetadata)
        logger.debug("Received init message: %s", init_message)
        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: Any,
        parallel_config_to_update: Any = None,
    ):
        handshake_socket = ctx.socket(zmq.DEALER)
        try:
            handshake_socket.setsockopt(zmq.IDENTITY, identity)
            handshake_socket.setsockopt(zmq.LINGER, 5000)
            handshake_socket.connect(handshake_address)
            addresses = self.startup_handshake(
                handshake_socket,
                local_client,
                headless,
                parallel_config_to_update,
            )
            yield addresses

            if self._omni_init_received_monotonic is not None:
                init_done_monotonic = time.monotonic()
                self._omni_handshake_metrics["engine_core_init_ms"] = (
                    init_done_monotonic - self._omni_init_received_monotonic
                ) * 1000.0
                pre_kv_cache_ms = self._omni_handshake_metrics.get("pre_kv_cache_init_ms", 0.0)
                kv_cache_init_ms = self._omni_handshake_metrics.get("kv_cache_init_ms", 0.0)
                post_kv_cache_ms = max(
                    0.0,
                    self._omni_handshake_metrics["engine_core_init_ms"] - pre_kv_cache_ms - kv_cache_init_ms,
                )
                self._omni_handshake_metrics["post_kv_cache_init_ms"] = post_kv_cache_ms

            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            ready_msg: dict[str, Any] = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "handshake_breakdown": _round_nested_metrics(self._omni_handshake_metrics),
            }
            if hasattr(self, "frontend_stats_publish_address"):
                ready_msg["dp_stats_address"] = self.frontend_stats_publish_address
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = vllm_config.parallel_config.compute_hash()

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))
        finally:
            handshake_socket.close(linger=5000)

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageEngineCoreProc busy loop in background process."""
        signal_callback: SignalCallback | None = None
        maybe_register_config_serialize_by_value()
        _reset_worker_startup_breakdown()
        _install_worker_startup_probes()

        engine_core: StageEngineCoreProc | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config = vllm_config.parallel_config

            set_process_title(f"StageEngineCoreProc_DP{dp_rank}")
            decorate_logs()

            # the current vllm-omni does not support data parallelism,
            # so we set the data parallel size to 1.
            # [TODO] support data parallelism in the future.
            # https://github.com/vllm-project/vllm-omni/issues/984
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            parallel_config.data_parallel_index = dp_rank

            engine_core = StageEngineCoreProc(
                *args,
                engine_index=dp_rank,
                **kwargs,
            )

            def wakeup_engine() -> None:
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum: int, frame: Any) -> None:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageEngineCoreProc exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("StageEngineCoreProc failed to start.")
            else:
                logger.exception("StageEngineCoreProc encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()


def spawn_stage_core(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool = False,
) -> tuple[EngineZmqAddresses, BaseProcess, str]:
    """Spawn a *StageEngineCoreProc* subprocess without performing the handshake.

    Must be called while the correct device env vars are set (e.g. under
    the stage-launch lock).  Call ``complete_stage_handshake`` afterwards.

    Returns ``(addresses, process, handshake_address)``.
    """
    addresses = get_engine_zmq_addresses(vllm_config)
    handshake_address = get_open_zmq_ipc_path()

    ctx = get_mp_context()
    proc = ctx.Process(
        target=StageEngineCoreProc.run_stage_core,
        name="StageEngineCoreProc",
        kwargs={
            "vllm_config": vllm_config,
            "local_client": True,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "dp_rank": 0,
            "local_dp_rank": 0,
        },
    )
    proc.start()
    return addresses, proc, handshake_address


def complete_stage_handshake(
    proc: BaseProcess,
    handshake_address: str,
    addresses: EngineZmqAddresses,
    vllm_config: VllmConfig,
    handshake_timeout: int,
) -> dict[str, Any]:
    """Perform the HELLO/INIT/READY handshake with an already-spawned proc.

    On failure the process is terminated before re-raising.
    """
    try:
        return _perform_handshake(proc, handshake_address, addresses, vllm_config, handshake_timeout)
    except Exception:
        shutdown([proc])
        raise


def _perform_handshake(
    proc: BaseProcess,
    handshake_address: str,
    addresses: EngineZmqAddresses,
    vllm_config: VllmConfig,
    handshake_timeout: int,
) -> dict[str, Any]:
    """Run the HELLO / INIT / READY handshake with the subprocess."""
    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        poller = zmq.Poller()
        poller.register(handshake_socket, zmq.POLLIN)
        poller.register(proc.sentinel, zmq.POLLIN)

        hello_wait_start = time.monotonic()
        identity, msg = _recv(poller, handshake_socket, proc, "HELLO", handshake_timeout)
        if msg.get("status") != "HELLO":
            raise RuntimeError(f"Expected HELLO, got: {msg}")
        hello_wait_ms = (time.monotonic() - hello_wait_start) * 1000.0

        init_payload = EngineHandshakeMetadata(
            addresses=addresses,
            parallel_config={},
        )
        init_to_ready_start = time.monotonic()
        handshake_socket.send_multipart([identity, msgspec.msgpack.encode(init_payload)])

        identity, msg = _recv(poller, handshake_socket, proc, "READY", handshake_timeout)
        if msg.get("status") != "READY":
            raise RuntimeError(f"Expected READY, got: {msg}")
        num_gpu_blocks = msg.get("num_gpu_blocks")
        if num_gpu_blocks is not None:
            vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        ready_wait_ms = (time.monotonic() - init_to_ready_start) * 1000.0

        handshake_breakdown = dict(msg.get("handshake_breakdown") or {})
        handshake_breakdown["wait_for_hello_ms"] = round(hello_wait_ms, 3)
        handshake_breakdown["init_to_ready_ms"] = round(ready_wait_ms, 3)
        return handshake_breakdown


def _recv(
    poller: zmq.Poller,
    handshake_socket: zmq.Socket,
    proc: BaseProcess,
    expected: str,
    timeout_s: int = 600,
) -> tuple[bytes, dict]:
    """Wait for one handshake message; raise if the process dies first."""
    timeout_ms = timeout_s * 1000
    while True:
        events = dict(poller.poll(timeout=timeout_ms))
        if not events:
            raise TimeoutError(
                f"Timed out waiting for {expected} from StageEngineCoreProc after {timeout_s}s. "
                f"This typically indicates model loading or initialization is taking too long. "
                f"Consider increasing `stage_init_timeout` for large models."
            )
        if handshake_socket in events:
            identity, raw = handshake_socket.recv_multipart()
            return identity, msgspec.msgpack.decode(raw)
        if proc.exitcode is not None:
            raise RuntimeError(f"StageEngineCoreProc died during {expected} (exit code {proc.exitcode})")
