import ctypes
import json
import os
import queue
import resource
import threading
import time
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("RKLLM_MODEL_DIR", "/home/orangepi/rkllm_gemma")
MODEL_PATH = os.environ.get(
    "RKLLM_MODEL_PATH",
    os.path.join(MODEL_DIR, "gemma-3-1b-it_w8a8_g128_rk3588.rkllm"),
)
LIB_PATH = os.environ.get(
    "RKLLM_LIB_PATH",
    "/home/orangepi/rkllm_gemma/lib/librkllmrt.so",
)
HOST = os.environ.get("WEBCHAT_HOST", "0.0.0.0")
PORT = int(os.environ.get("WEBCHAT_PORT", "8080"))


app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))


MODEL_CATALOG = [
    {
        "id": "gemma-3-270m-it",
        "label": "Gemma 3 270M",
        "path": os.path.join(MODEL_DIR, "gemma-3-270m-it_w8a8_g128_rk3588.rkllm"),
    },
    {
        "id": "gemma-3-1b-it",
        "label": "Gemma 3 1B",
        "path": os.path.join(MODEL_DIR, "gemma-3-1b-it_w8a8_g128_rk3588.rkllm"),
    },
    {
        "id": "gemma-3n-e2b",
        "label": "Gemma 3n E2B IT (experimental)",
        "path": os.path.join(MODEL_DIR, "gemma-3n-E2B-it-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm"),
    },
]


class LLMCallState:
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3


class RKLLMInputType:
    RKLLM_INPUT_PROMPT = 0


class RKLLMInferMode:
    RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


class RKLLMEngine:
    def __init__(self, model_path: str, lib_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(lib_path)

        self.model_path = model_path
        self.lib = ctypes.CDLL(lib_path)
        self.handle = ctypes.c_void_p()
        self._queue: Optional[queue.Queue] = None
        self._state_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._active_done = False
        self._active_error = None
        self._active_perf = None
        self._stop_requested = False

        self.callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(RKLLMResult),
            ctypes.c_void_p,
            ctypes.c_int,
        )
        self.callback = self.callback_type(self._callback_impl)

        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(RKLLMParam),
            self.callback_type,
        ]
        self.lib.rkllm_init.restype = ctypes.c_int
        self.lib.rkllm_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        self.lib.rkllm_run.restype = ctypes.c_int
        self.lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rkllm_destroy.restype = ctypes.c_int
        self.lib.rkllm_clear_kv_cache.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.rkllm_clear_kv_cache.restype = ctypes.c_int
        self.lib.rkllm_abort.argtypes = [ctypes.c_void_p]
        self.lib.rkllm_abort.restype = ctypes.c_int

        self._init_model()

    def _build_param(self) -> RKLLMParam:
        param = RKLLMParam()
        param.model_path = self.model_path.encode("utf-8")
        param.max_context_len = 4096
        param.max_new_tokens = 1024
        param.top_k = 1
        param.n_keep = -1
        param.top_p = 0.9
        param.temperature = 0.8
        param.repeat_penalty = 1.1
        param.frequency_penalty = 0.0
        param.presence_penalty = 0.0
        param.mirostat = 0
        param.mirostat_tau = 5.0
        param.mirostat_eta = 0.1
        param.skip_special_token = True
        param.is_async = False
        param.img_start = b""
        param.img_end = b""
        param.img_content = b""
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.n_batch = 1
        param.extend_param.use_cross_attn = 0
        param.extend_param.enabled_cpus_num = 4
        param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        return param

    def _init_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)
        param = self._build_param()
        ret = self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), self.callback)
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed: {ret}")

    def _reinit_model(self) -> None:
        if self.handle:
            self.lib.rkllm_destroy(self.handle)
        self.handle = ctypes.c_void_p()
        self._init_model()

    def _callback_impl(self, result_ptr, _userdata, state) -> int:
        with self._state_lock:
            if state == LLMCallState.RKLLM_RUN_NORMAL and self._queue is not None:
                piece = result_ptr.contents.text
                if piece:
                    self._queue.put(piece.decode("utf-8", errors="ignore"))
            elif state == LLMCallState.RKLLM_RUN_FINISH:
                perf = result_ptr.contents.perf
                self._active_perf = {
                    "prefill_tokens": int(perf.prefill_tokens),
                    "generate_tokens": int(perf.generate_tokens),
                    "prefill_time_ms": float(perf.prefill_time_ms),
                    "generate_time_ms": float(perf.generate_time_ms),
                    "memory_usage_mb": float(perf.memory_usage_mb),
                }
                self._active_done = True
            elif state == LLMCallState.RKLLM_RUN_ERROR:
                self._active_done = True
                self._active_error = "generation stopped" if self._stop_requested else "rkllm_run error"
        return 0

    def reset(self) -> None:
        with self._run_lock:
            ret = self.lib.rkllm_clear_kv_cache(self.handle, 1, None, None)
            if ret != 0:
                self._reinit_model()

    def abort(self) -> None:
        with self._state_lock:
            self._stop_requested = True
        self.lib.rkllm_abort(self.handle)

    def switch_model(self, model_path: str) -> None:
        with self._run_lock:
            self.model_path = model_path
            self._reinit_model()

    def stream_chat(self, prompt: str):
        with self._run_lock:
            token_queue: queue.Queue = queue.Queue()
            with self._state_lock:
                self._queue = token_queue
                self._active_done = False
                self._active_error = None
                self._active_perf = None
                self._stop_requested = False

            infer = RKLLMInferParam()
            infer.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
            infer.lora_params = None
            infer.prompt_cache_params = None
            infer.keep_history = 1

            rkllm_input = RKLLMInput()
            rkllm_input.role = b"user"
            rkllm_input.enable_thinking = False
            rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
            rkllm_input.input_data.prompt_input = prompt.encode("utf-8")

            def runner():
                ret = self.lib.rkllm_run(
                    self.handle,
                    ctypes.byref(rkllm_input),
                    ctypes.byref(infer),
                    None,
                )
                if ret != 0:
                    with self._state_lock:
                        self._active_done = True
                        self._active_error = f"rkllm_run returned {ret}"

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()

            try:
                while True:
                    try:
                        yield {"type": "token", "text": token_queue.get(timeout=0.05)}
                        continue
                    except queue.Empty:
                        pass

                    with self._state_lock:
                        done = self._active_done
                        error = self._active_error
                        perf = self._active_perf

                    if done and token_queue.empty():
                        if error:
                            raise RuntimeError(error)
                        if perf is not None:
                            generate_time_s = perf["generate_time_ms"] / 1000.0
                            yield {
                                "type": "meta",
                                "stats": {
                                    "tokens": perf["generate_tokens"],
                                    "tokens_per_second": (
                                        perf["generate_tokens"] / generate_time_s
                                        if generate_time_s > 0
                                        else 0.0
                                    ),
                                    "prefill_tokens": perf["prefill_tokens"],
                                    "memory_usage_mb": perf["memory_usage_mb"],
                                },
                            }
                        break
            finally:
                with self._state_lock:
                    self._queue = None


resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))
engine = RKLLMEngine(MODEL_PATH, LIB_PATH)


def get_available_models():
    items = []
    for model in MODEL_CATALOG:
        item = dict(model)
        item["available"] = os.path.exists(model["path"])
        item["active"] = os.path.abspath(model["path"]) == os.path.abspath(engine.model_path)
        items.append(item)
    return items


@app.get("/")
def index():
    active_model = next((model for model in get_available_models() if model["active"]), None)
    return render_template(
        "index.html",
        model_name=(active_model["label"] if active_model else os.path.basename(engine.model_path)),
    )


@app.get("/api/status")
def status():
    return jsonify(
        {
            "ok": True,
            "model": os.path.basename(engine.model_path),
            "models": get_available_models(),
        }
    )


@app.get("/api/models")
def models():
    return jsonify({"models": get_available_models()})


@app.post("/api/reset")
def reset_chat():
    engine.reset()
    return jsonify({"ok": True})


@app.post("/api/model")
def switch_model():
    data = request.get_json(silent=True) or {}
    model_id = data.get("model_id")
    target = next((item for item in MODEL_CATALOG if item["id"] == model_id), None)
    if target is None:
        return jsonify({"error": "unknown model"}), 404
    if not os.path.exists(target["path"]):
        return jsonify({"error": "model file not found"}), 404

    try:
        engine.switch_model(target["path"])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"ok": True, "model": target["id"], "models": get_available_models()})


@app.post("/api/stop")
def stop_chat():
    engine.abort()
    return jsonify({"ok": True})


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("message") or "").strip()
    if not prompt:
        return jsonify({"error": "message is required"}), 400

    def generate():
        try:
            for chunk in engine.stream_chat(prompt):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=True, debug=False)
