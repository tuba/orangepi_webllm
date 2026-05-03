# Orange Pi WebLLM

Flask web chat for RKLLM models running on Orange Pi RK3588.

## Features

- Streaming token output in the browser
- Conversation reset
- Stop generation
- Per-message stats: tokens, tok/s, prefill, memory
- Model switcher for installed RKLLM models

## Run

```bash
python3 app.py
```

Environment variables:

- `RKLLM_MODEL_DIR`
- `RKLLM_MODEL_PATH`
- `RKLLM_LIB_PATH`
- `WEBCHAT_HOST`
- `WEBCHAT_PORT`
