# Webchat Plan

## Current State

- RKLLM webchat is running on Orange Pi RK3588
- Streaming chat UI is implemented
- Model switcher is implemented
- Memory and model-load telemetry are visible in the UI
- `terminal.exec` MCP-style tool loop is implemented behind a UI toggle
- Terminal tool parsing now accepts bare commands like `pwd` and `ls -l`
- Internal tool calls now avoid polluting chat history with `keep_history=0`
- Context-size and active-model questions are answered directly by the backend

## Next Steps

1. Harden terminal access with an explicit command allowlist and denylist.
2. Force post-tool answers to match the user's language more reliably.
3. Add per-tool-call history in the sidebar with timestamps and durations.
4. Isolate terminal execution and model inference into separate worker processes.
5. Replace the internal tool loop with a real MCP server/client integration.
6. Add structured tests for model switching, SSE streaming, and terminal tool calls.

## Notes

- `Gemma 3 1B` is the current stable default.
- `Gemma 3n E2B` is available as experimental and can crash the process after generation.
- Terminal access is intentionally disabled by default and should stay opt-in.
- The local 1B model still sometimes answers awkwardly after tool output, even though the execution path now works.
