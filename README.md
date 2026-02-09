<p align="center">
  <h1 align="center">UniClaudeProxy</h1>
  <p align="center">
    <strong>Use Any LLM with Claude Code — The Universal Anthropic API Proxy</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#features">Features</a> &bull;
    <a href="#configuration">Configuration</a> &bull;
    <a href="#supported-providers">Providers</a> &bull;
    <a href="#how-it-works">Architecture</a>
  </p>
</p>

---

**UniClaudeProxy** is a lightweight, high-performance FastAPI proxy that lets [Claude Code](https://docs.anthropic.com/en/docs/claude-code) talk to **any LLM backend** — OpenAI-compatible APIs, Google Gemini, DeepSeek, GLM, Ollama, or Anthropic passthrough. Drop it in as your API endpoint and use whatever model you want with Claude Code's full tool-calling capabilities.

> I wanted a quick way to use all the models in Claude Code without being locked to a single provider. So I built one of the best proxies out there — fast, modular, and packed with features that just work.

```
Claude Code CLI  -->  UniClaudeProxy (localhost:9223)  -->  Any LLM Provider
     ^                        |
     +---- Anthropic SSE <----+
```

## Why UniClaudeProxy?

- **Use any model with Claude Code** — DeepSeek, GLM, Ollama, Gemini, or any OpenAI-compatible API
- **Zero changes to Claude Code** — just point the API URL to `localhost:9223`
- **Full tool calling support** — native function calling + ReAct XML fallback for models without it
- **Streaming first** — real-time SSE streaming with proper Anthropic event format
- **Production ready** — hot-reload config, image support, thinking/reasoning blocks, custom headers

---

## Features

### Provider Support
| Provider Type | Protocol | Endpoints | Status |
|---|---|---|---|
| **OpenAI-compatible** | Chat Completions | `/v1/chat/completions` | :white_check_mark: |
| **OpenAI-compatible** | Responses API | `/v1/responses` | :white_check_mark: |
| **Google Gemini** | Native Gemini API | `generateContent` / `streamGenerateContent` | :white_check_mark: |
| **Anthropic Passthrough** | Messages API | `/v1/messages` | :white_check_mark: |

### Core Features

- **Automatic API Translation** — Converts Anthropic Messages API requests to OpenAI, Gemini, or Claude passthrough format and back, seamlessly
- **Full Streaming Support** — Real-time SSE streaming with proper `message_start`, `content_block_delta`, `message_delta`, and `message_stop` events
- **Native Tool Calling** — Full function calling support across all provider types with automatic ID conversion (`toolu_` <-> `fc_` <-> Gemini `functionCall`)
- **ReAct XML Tool Calling** — For models without native function calling (like local Ollama models), injects XML tool descriptions into the system prompt and parses `<tool_call>` XML responses back into proper Anthropic tool_use blocks
- **Thinking / Reasoning Blocks** — Supports `<think>` tag extraction, OpenAI reasoning summaries, and Gemini `thought` parts — all converted to Anthropic `thinking` content blocks
- **System Prompt Replacement** — Replace identity-specific strings in the system prompt per-model (e.g., strip "Claude Code" references for models that refuse to role-play)
- **Automatic Hot Reload** — Edit `config.json` and the proxy picks up changes instantly via filesystem watcher — no restart needed
- **Image Support** — Three modes: `input_image` (inline base64), `save_and_ref` (save to disk + reference), or `strip` (remove images for text-only models)
- **Custom Headers** — Per-provider custom headers for authentication, routing, or any other need
- **Extra OpenAI Parameters** — Pass `reasoning`, `truncation`, `text`, `parallel_tool_calls`, and other provider-specific parameters per-model
- **Tool Name Mapping** — Map upstream tool names to Claude Code names (e.g., `shell_call` -> `Bash`)
- **Force Stream Mode** — For providers that always return SSE, consume internally and return as non-streaming when needed
- **Gemini thoughtSignature Round-Trip** — Properly encodes and decodes Gemini's `thoughtSignature` through tool call IDs for multi-turn thinking conversations
- **Parameter Auto-Fix** — Automatically corrects camelCase/snake_case parameter mismatches from Gemini function calls

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/vibheksoni/UniClaudeProxy.git
cd UniClaudeProxy
pip install -r requirements.txt
```

### 2. Create your config

```bash
cp config.example.json config.json
```

Edit `config.json` with your API keys and model mappings. See [Configuration](#configuration) for details.

### 3. Start the proxy

**Windows:**
```bash
Run.bat
```

**Linux / macOS:**
```bash
chmod +x Run.sh
./Run.sh
```

**Or directly:**
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 9223
```

### 4. Point Claude Code to the proxy

The recommended way is to create a **Claude Code profile** in `~/.claude/settings.json`. This keeps your proxy config isolated and stable:

```json
{
  "profiles": {
    "cc-proxy": {
      "env": {
        "ANTHROPIC_AUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9223",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "API_TIMEOUT_MS": "3000000",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "50000",
        "CLAUDE_BASH_NO_LOGIN": "1"
      },
      "permissions": {
        "allow": [],
        "deny": []
      }
    }
  }
}
```

Then launch Claude Code with the profile:

```bash
claude --profile cc-proxy
```

**What each variable does:**

| Variable | Purpose |
|---|---|
| `ANTHROPIC_AUTH_TOKEN` | Empty — proxy handles auth, no token needed client-side |
| `ANTHROPIC_BASE_URL` | Routes all API traffic through UniClaudeProxy |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | Prevents Claude Code from making background requests that bypass the proxy |
| `API_TIMEOUT_MS` | 50-minute timeout — prevents disconnects on long-running tool calls |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | Allows up to 50k output tokens per response |
| `CLAUDE_BASH_NO_LOGIN` | Skips login shell for bash commands — faster execution |

That's it. Claude Code now routes through UniClaudeProxy to whatever backend you configured. Run `/status` inside Claude Code to verify the endpoint is active.

---

## Configuration

UniClaudeProxy uses a single `config.json` file with three sections:

```json
{
  "server": { "host": "127.0.0.1", "port": 9223 },
  "models": {
    "<anthropic-model-name>": "<provider-name>/<model-id>"
  },
  "providers": {
    "<provider-name>": {
      "provider_type": "openai | gemini | claude",
      "api_key": "your-api-key",
      "base_url": "https://api.example.com",
      "headers": {},
      "models": {
        "<model-id>": { ... }
      }
    }
  }
}
```

### How Routing Works

1. Claude Code sends a request with `model: "claude-sonnet-4-5-20250929"`
2. `config.models` maps it to `"deepseek/deepseek-chat"`
3. The proxy splits on `/` — provider = `deepseek`, model_id = `deepseek-chat`
4. Looks up `config.providers.deepseek.models["deepseek-chat"]` for model-specific settings
5. Converts the request to the provider's format, sends it, converts the response back

### Model Config Options

Each model entry under a provider supports these fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | Human-readable display name |
| `upstream_model_id` | string | null | Override model ID sent upstream (when config key differs from actual model name) |
| `responses` | bool | false | Use OpenAI `/v1/responses` endpoint instead of `/v1/chat/completions` |
| `use_react` | bool | false | Enable ReAct XML tool calling for models without native function calling |
| `inject_context` | bool | false | Inject system prompt + tool summary as developer message |
| `force_stream` | bool | false | Provider always returns SSE; consume internally for non-stream requests |
| `upstream_system` | bool | false | Provider forces its own system prompt/tools (skip forwarding ours) |
| `tool_mapping` | object | {} | Map upstream tool names to Claude Code names (e.g. `{"shell_call": "Bash"}`) |
| `reasoning` | object | {} | Reasoning config (e.g. `{"effort": "high", "summary": "auto"}`) |
| `truncation` | string | null | Truncation strategy (`"auto"` or `"disabled"`) |
| `text` | object | {} | Text config (e.g. `{"verbosity": "low"}`) |
| `max_output_tokens` | int | null | Override max output tokens |
| `parallel_tool_calls` | bool | null | Enable parallel tool calls |
| `image_mode` | string | `"input_image"` | `"input_image"`, `"save_and_ref"`, or `"strip"` |
| `image_dir` | string | null | Custom directory for saved images |
| `system_replacements` | object | {} | String replacements on system prompt (key=target, value=replacement) |

### System Prompt Replacement

Some models refuse to operate when they see identity claims like "You are Claude Code" in the system prompt. Use `system_replacements` to fix this per-model:

```json
"system_replacements": {
  "You are Claude Code, Anthropic's official CLI for Claude.": "You are an advanced AI coding assistant integrated into a CLI tool.",
  "Claude Code": "the coding assistant"
}
```

Replacements are applied universally before any provider dispatch — works with OpenAI, Gemini, Claude passthrough, and ReAct paths.

---

## Supported Providers

### OpenAI-Compatible (`provider_type: "openai"`)

Any API that speaks the OpenAI Chat Completions or Responses protocol. This includes:

- **DeepSeek** — DeepSeek V3, DeepSeek R1
- **GLM (Zhipu AI)** — GLM-4, GLM-4 Plus
- **Ollama** — Any local model (Llama, Qwen, Mistral, etc.)
- **LM Studio** — Local models via OpenAI-compatible API
- **vLLM** — Self-hosted inference
- **Together AI**, **Groq**, **Fireworks** — Cloud inference
- Any other OpenAI-compatible endpoint

```json
{
  "provider_type": "openai",
  "api_key": "your-key",
  "base_url": "https://api.deepseek.com",
  "models": {
    "deepseek-chat": {
      "name": "DeepSeek V3",
      "max_output_tokens": 8192
    }
  }
}
```

### Google Gemini (`provider_type: "gemini"`)

Native Gemini API with full support for thinking, function calling, and `thoughtSignature` round-tripping.

```json
{
  "provider_type": "gemini",
  "api_key": "your-gemini-key",
  "base_url": "https://generativelanguage.googleapis.com/v1beta",
  "headers": {
    "x-goog-api-key": "your-gemini-key"
  },
  "models": {
    "gemini-2.5-pro-preview-06-05": {
      "name": "Gemini 2.5 Pro"
    }
  }
}
```

### Anthropic Passthrough (`provider_type: "claude"`)

For upstream Anthropic-compatible APIs. No conversion — raw body forwarded as-is.

```json
{
  "provider_type": "claude",
  "api_key": "your-key",
  "base_url": "https://api.anthropic.com/v1",
  "headers": {
    "anthropic-version": "2023-06-01"
  },
  "models": {
    "claude-sonnet-4-5-20250929": {
      "name": "Claude Sonnet 4.5"
    }
  }
}
```

---

## How It Works

```
Claude Code CLI  -->  UniClaudeProxy (localhost:9223)  -->  Any LLM Provider
(Anthropic fmt)       Route + Convert + Stream              (OpenAI/Gemini/etc)
     ^                        |
     +---- Anthropic SSE <----+
```

### Request Flow

1. Claude Code sends an Anthropic Messages API request
2. The proxy resolves the model name to a provider via `config.json`
3. `system_replacements` are applied to the system prompt
4. If `use_react` is enabled, tools are injected as XML into the system prompt
5. The request is converted to the target provider's format
6. The response streams back, converted to Anthropic SSE in real-time
7. Tool calls, thinking blocks, and content are all properly mapped back

---

## ReAct XML Tool Calling

For models without native function calling (like local Ollama models), enable `"use_react": true` in the model config. The proxy will:

1. Strip native `tools` from the request body
2. Inject XML tool descriptions with full parameter schemas into the system prompt
3. Convert `tool_use`/`tool_result` history to XML format
4. Add `</tool_call>` as a stop sequence
5. Parse the model's XML output back into proper Anthropic `tool_use` blocks
6. Extract `<think>` blocks into Anthropic `thinking` content blocks

The model outputs tool calls like this:

```xml
<tool_call>
<name>Bash</name>
<parameters>
{"command": "ls -la"}
</parameters>
</tool_call>
```

And receives results as:

```xml
<observation>
total 42
drwxr-xr-x  5 user user 4096 Feb  9 12:00 .
...
</observation>
```

---

## Project Structure

```
UniClaudeProxy/
├── config.example.json              # Example configuration (copy to config.json)
├── config.json                      # Your config (gitignored)
├── requirements.txt                 # Python dependencies
├── Run.bat                          # Windows launcher
├── Run.sh                           # Linux/macOS launcher
├── app/
│   ├── main.py                      # FastAPI app, POST /v1/messages, GET /health
│   ├── config.py                    # Config loader, route resolver, hot reload
│   ├── models.py                    # Pydantic models for Anthropic API types
│   ├── watcher.py                   # Filesystem watcher for config hot reload
│   ├── converters/
│   │   ├── anthropic_to_openai.py   # Anthropic -> OpenAI (Chat + Responses API)
│   │   ├── openai_to_anthropic.py   # OpenAI -> Anthropic (non-stream + stream)
│   │   ├── anthropic_to_gemini.py   # Anthropic -> Gemini native format
│   │   └── gemini_to_anthropic.py   # Gemini -> Anthropic (non-stream + stream)
│   ├── providers/
│   │   ├── openai_provider.py       # HTTP client for OpenAI-compatible APIs
│   │   ├── gemini_provider.py       # HTTP client for Gemini native API
│   │   └── anthropic_provider.py    # HTTP client for Anthropic passthrough
│   ├── react/
│   │   ├── prompt.py                # ReAct system prompt + XML tool formatting
│   │   ├── parser.py                # XML <tool_call> parsing
│   │   └── transform.py             # Request/response/stream transformation
│   └── utils/
│       └── images.py                # Image handling (detect, save, convert)
```

---

## Debugging

All debug output goes to `debug.log` in the project root. Useful log patterns:

```bash
# Check route resolution
grep "Request: model=" debug.log | tail -5

# Check tool calls working
grep "REACT STREAM\|tool_use=True" debug.log | tail -10

# Check for errors
grep "ERROR\|Traceback" debug.log | tail -10

# See what's being sent upstream
grep "OUTGOING REQUEST" debug.log | tail -5
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Model says "I cannot assume the role of Claude Code" | Identity claims in system prompt | Add `system_replacements` to strip "Claude Code" / "Anthropic" |
| Model narrates tool calls instead of executing | Missing ReAct prompt or no few-shot example | Enable `use_react: true` and check prompt template |
| Tool calls loop on same tool | Schema too compact for model to construct params | Check that parameter schemas include nested properties |
| Empty responses | Provider returned non-JSON or empty body | Check `debug.log` for upstream errors |
| Connection refused | Proxy not running or wrong port | Verify `Run.bat`/`Run.sh` started successfully |

---

## Contributing

Contributions are welcome. Open an issue or submit a pull request.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built for developers who want freedom to use any model with Claude Code.</sub>
</p>
