# lspeak

<div align="center">
  
  **Text-to-speech that doesn't talk over itself**
  
  [GitHub](https://github.com/nickpending/lspeak) | [Issues](https://github.com/nickpending/lspeak/issues) | [Roadmap](#roadmap)

  [![Status](https://img.shields.io/badge/Status-Alpha-orange?style=flat)](#status-alpha)
  [![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python)](https://python.org)
  [![uv](https://img.shields.io/badge/uv-0.7.15+-5A67D8?style=flat)](https://github.com/astral-sh/uv)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

**lspeak** is a Unix pipe for text-to-speech that queues speech from parallel processes, caches similar phrases semantically, and keeps models loaded for instant responses.

```bash
echo "Deploy complete" | lspeak
# Speaks through ElevenLabs or system TTS, caches semantically similar phrases
```

## What is lspeak?

lspeak is a Unix-first CLI tool that makes text-to-speech actually usable in your terminal. It's not a TTS engine - it's the smart orchestration layer that:

- **Semantically caches** similar phrases (save 90%+ on API costs)
- **Pre-loads models** in daemon mode (sub-second responses)
- **Manages audio queues** (no overlapping speech from parallel processes)
- **Provides clean abstractions** over ElevenLabs API and system TTS

Think of it as **infrastructure for voice-enabled tools**. You can:
- **Pipe to it**: Any tool's output becomes speech
- **Import it**: Python library for sophisticated integrations
- **Build on it**: Semantic caching means your tool's common phrases are instant

This is how [clarvis](https://github.com/nickpending/clarvis) gives Claude Code a voice - it calls `lspeak.speak()` for each sentence, leveraging the cache so "Sir, project X is complete" variants all share audio.

## Status: Alpha

**This is early software that works but has rough edges.** It's been in daily use for a month, handling 2,800+ unique phrases, but expect quirks.

## Installation

Since lspeak isn't on PyPI yet, install directly from GitHub using uv:

```bash
# Install as a tool (for CLI usage)
uv tool install git+https://github.com/nickpending/lspeak.git

# Or clone and install for development/library usage
git clone https://github.com/nickpending/lspeak.git
cd lspeak
uv sync  # Installs in editable mode for development
```

For high-quality voices via ElevenLabs:
```bash
export ELEVENLABS_API_KEY=your_key_here
```

No API key? Uses system TTS (`say` on macOS, `espeak` on Linux).

## Quick Start

```bash
# Speak text directly
lspeak "Build successful"

# Pipe from any command
git status | lspeak
cat README.md | lspeak

# Read a file
lspeak -f notes.txt

# Save to audio file
lspeak -o alert.mp3 "System is down"

# Use specific voice
lspeak -v Rachel "Hello world"

# List available voices
lspeak --list-voices
```

## Why a Daemon?

Two critical problems needed solving:

1. **Model loading takes 27 seconds** - Sentence-transformers (for semantic similarity) has huge startup time
2. **Parallel processes create audio chaos** - Multiple scripts speaking simultaneously is unintelligible

The daemon solves both: keeps models loaded for instant response AND queues speech so only one process talks at a time.

## The Key Innovation: Semantic Caching

Traditional caching matches exact text. lspeak understands **meaning**:

```bash
lspeak "Deploy complete"        # First call: hits API
lspeak "Deployment complete"     # Cached! Semantically similar
lspeak "Deploy finished"         # Also cached! 95% similarity
```

This isn't just string matching - it uses sentence-transformers to understand that these phrases mean the same thing. Your API costs drop by 90%+ for typical CLI usage.

## Daemon Mode: Instant Response

The first call loads ML models for semantic similarity. Keep them in memory for instant subsequent responses:

```bash
# First call - loads models, starts daemon automatically
lspeak "Hello"  # ~27 seconds (one-time model loading)

# Every subsequent call - daemon already running
lspeak "Build complete"  # <100ms response
```

The daemon also manages a speech queue, ensuring multiple processes don't create audio chaos.

## Python Library Usage

When installed via `uv sync` (development mode) or as a dependency:

```python
from lspeak import speak
import asyncio

async def notify_user():
    # Simple usage
    await speak("Tests passed")

    # With options
    await speak(
        "Deploy complete",
        voice="Rachel",
        provider="elevenlabs",  # or "system" for local TTS
        cache=True,  # Semantic caching enabled
        cache_threshold=0.95  # 95% similarity threshold
    )

asyncio.run(notify_user())
```

To use lspeak as a library in your project:
```bash
# Add to your project's dependencies
cd your-project
uv add git+https://github.com/nickpending/lspeak.git
```

## HTTP API Usage

The lspeak daemon can optionally expose an HTTP API for remote access or integration with non-Python tools:

```bash
# Start daemon with HTTP API on port 7777
LSPEAK_HTTP_PORT=7777 lspeak --daemon-start

# Optional: Secure with API key authentication
LSPEAK_HTTP_PORT=7777 LSPEAK_API_KEY=your_secret_key lspeak --daemon-start
```

### Using curl

```bash
# Speak text
curl -X POST http://localhost:7777/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Build complete", "provider": "system"}'

# With API key authentication
curl -X POST http://localhost:7777/speak \
  -H "X-API-Key: your_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Build complete", "provider": "system"}'

# Check daemon status
curl http://localhost:7777/status

# Check queue status
curl http://localhost:7777/queue
```

### Using Python httpx

```python
import httpx
import asyncio

async def speak_via_http():
    async with httpx.AsyncClient() as client:
        # Simple speak request
        response = await client.post(
            "http://localhost:7777/speak",
            json={"text": "Deploy complete", "provider": "system"}
        )
        result = response.json()
        print(f"Success: {result['success']}")

        # With API key authentication
        response = await client.post(
            "http://localhost:7777/speak",
            headers={"X-API-Key": "your_secret_key"},
            json={
                "text": "Build successful",
                "provider": "elevenlabs",
                "voice": "Rachel",
                "cache": True
            }
        )

        # Check queue status
        status = await client.get("http://localhost:7777/queue")
        print(status.json())

asyncio.run(speak_via_http())
```

**API Endpoints:**
- `POST /speak` - Queue speech synthesis (returns immediately)
- `GET /status` - Get daemon status (uptime, models loaded)
- `GET /queue` - Get current queue status
- `GET /docs` - Interactive API documentation (FastAPI auto-generated)

**Authentication:**
- No auth required by default (localhost only)
- Set `LSPEAK_API_KEY` env var to require `X-API-Key` header on all requests
- Returns 401 for missing/invalid API keys when auth enabled

## Real-World Usage

### CI/CD Notifications
```yaml
# .github/workflows/notify.yml
- name: Notify build status
  run: |
    if [ "${{ job.status }}" == "success" ]; then
      echo "Build successful for ${{ github.ref }}" | lspeak
    else
      echo "Build failed on ${{ github.ref }}" | lspeak
    fi
```

### Monitoring Scripts
```bash
#!/bin/bash
while true; do
    if ! curl -s https://api.example.com/health > /dev/null; then
        echo "API is down" | lspeak
    fi
    sleep 60
done
```

### With Claude Code (via [clarvis](https://github.com/nickpending/clarvis))

Get Jarvis-style voice updates from your AI pair programmer. In `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "cat | clarvis"
      }]
    }]
  }
}
```

Now Claude Code speaks through clarvis → lspeak:
- "Sir, project auth has encountered an error. Database connection failed."
- "Sir, deployment is complete. All tests passed."

Clarvis uses lspeak's library interface for maximum cache efficiency.

## How It Works

lspeak orchestrates several components:

1. **Text Input** → Standard Unix pipes, args, or files
2. **Semantic Analysis** → Generate embeddings via sentence-transformers
3. **Cache Lookup** → FAISS similarity search for cached audio
4. **TTS Provider** → ElevenLabs API or system TTS if cache miss
5. **Audio Output** → pygame for cross-platform playback
6. **Queue Management** → Serial playback via daemon

```
┌─────────┐     ┌──────────┐     ┌───────────┐
│  Text   │────▶│ Semantic │────▶│   Cache   │
│  Input  │     │ Embedding│     │  (FAISS)  │
└─────────┘     └──────────┘     └───────────┘
                                        │
                                   Cache Hit?
                                    ┌───┴───┐
                                    │       │
                                   Yes      No
                                    │       │
                              Use Cached    Call TTS API
                                Audio       (ElevenLabs/System)
                                    │       │
                                    └───┬───┘
                                        │
                                  ┌─────▼─────┐
                                  │   Audio   │
                                  │  Playback │
                                  └───────────┘
```

## Providers

### ElevenLabs (Premium)
- High-quality AI voices
- 29+ languages
- Requires API key and internet
- ~1-2 second latency per unique phrase

### System (Free)
- Uses `say` (macOS) or `espeak` (Linux)
- No API key needed
- Works offline
- Instant but robotic

## Advanced Features

### Cache Control

**When to use caching (default):**
- Status messages: "Build complete", "Tests passed", "Deploy failed"
- Notifications: "Process finished", "Error occurred", "Warning: low memory"
- Common phrases: "Hello", "Goodbye", "Thank you"
- Repeated alerts: "API is down", "Server unreachable"

**When to disable caching (`--no-cache`):**
- Timestamps: "The time is 3:45 PM"
- Dynamic data: "CPU usage is 47%"
- Unique IDs: "Order #ABC123 processed"
- Personal info: "Hello, John Smith"
- One-time messages: Random quotes, news headlines

```bash
# Good for caching - will likely repeat
lspeak "Build successful"
lspeak "Deployment failed"

# Bad for caching - too unique
lspeak --no-cache "Current time: $(date)"
lspeak --no-cache "Random UUID: $(uuidgen)"
lspeak --no-cache "Temperature is 72.5°F"

# Adjust similarity threshold for fuzzy matching
lspeak --cache-threshold 0.98 "Deploy complete"  # Stricter matching
lspeak --cache-threshold 0.90 "Build done"       # Looser matching

# Debug mode - see cache hits/misses
lspeak --debug "Hello world"
```

### Queue Management
```bash
# Check queue status
lspeak --queue-status

# Skip queue for immediate playback
lspeak --no-queue "Urgent alert"
```

### Daemon Control
```bash
# Check daemon status
lspeak --daemon-status

# Restart daemon
lspeak --daemon-restart

# Stop daemon (frees memory)
lspeak --daemon-stop
```

## Performance

With semantic caching and daemon mode:

- **First call**: ~27 seconds (loading ML models)
- **Cached phrases**: <100ms (semantic similarity match)
- **New phrases**: 1-2 seconds (TTS API call + caching)
- **Cache hit rate**: 70-95% for typical CLI usage
- **Cost reduction**: 90%+ via semantic caching

## Architecture Decisions

**Why semantic caching?**  
Exact string matching is useless for natural speech. "Build complete" and "Build finished" should share cached audio.

**Why a daemon?**  
Loading sentence-transformers takes 27 seconds. Keeping models in memory enables instant responses.

**Why Unix pipes?**  
Composability. Any tool that outputs text can now speak.

**Why Python?**  
Best ecosystem for ML (sentence-transformers, FAISS) and audio (pygame).

## Platform Support

- **macOS**: ✅ Tested daily (uses built-in `say` command)
- **Linux**: 🤷 Should work (uses `espeak` - untested)
- **Windows**: 🤷 Might work (SAPI support exists - untested)

## Requirements

- Python 3.13+
- macOS (only tested platform)
- pygame for audio playback (~15MB, installed automatically via pip/uv)
- ElevenLabs API key (optional, for premium voices)
- Internet connection (for ElevenLabs)

## FAQ

**Q: What makes this different from just calling the ElevenLabs API directly?**  
A: Semantic caching. You'll hit the same phrases repeatedly in CLI usage. We cache similar meanings, not exact strings, reducing API calls by 90%+.

**Q: Why is the first call slow?**  
A: Loading sentence-transformers for semantic similarity. This happens once - the daemon keeps models in memory.

**Q: Can I use this offline?**  
A: System provider works offline. Cached phrases work offline. New phrases with ElevenLabs need internet.

**Q: Is my text being sent to the cloud?**  
A: Only with ElevenLabs provider. System provider is local. Embeddings for caching are generated locally.

## Known Issues & Limitations

**Current limitations in v0.1.0:**

- **No daemon logs** - Debug output goes to /dev/null (fix coming in v0.2)
- **Cache grows unbounded** - No built-in cleanup (manual: `rm -rf ~/.cache/lspeak/audio/`)
- **Daemon can get stuck** - Restart with `lspeak --daemon-restart`
- **No cache metrics** - Can't see hit rate without code changes
- **Code duplication** - TTS pipeline implemented twice (daemon vs direct)

**What works well:**
- Semantic caching (2,800+ cached phrases in production)
- Queue management (no overlapping audio)
- Sub-second responses with daemon
- ElevenLabs and system TTS providers

## Roadmap

**v0.2.0** (Next release):
- [ ] Daemon logging to file
- [ ] Cache management commands
- [ ] Fix code duplication
- [ ] Error recovery

**v0.3.0** (Future):
- [ ] Cache metrics dashboard
- [ ] Queue visibility/control
- [ ] PyPI package
- [ ] Comprehensive tests

See [improvement-tasks.md](https://github.com/nickpending/lspeak/blob/main/docs/improvement-tasks.md) for detailed plans.

## Contributing

PRs welcome, especially for items on the roadmap. Core principles:

- Keep the Unix philosophy - do one thing well
- Don't add features that complicate the core mission
- Test with real audio and real models
- Cache aggressively - API calls are expensive

**Note**: The codebase needs refactoring (see roadmap). Major changes should wait for v0.2.0's cleanup.

## License

MIT

## Credits

Built on top of excellent tools:
- **ElevenLabs** for outstanding TTS API
- **sentence-transformers** for semantic similarity
- **FAISS** for lightning-fast similarity search
- **pygame** for reliable cross-platform audio

lspeak doesn't do text-to-speech - it makes existing TTS tools usable in the Unix pipeline through intelligent caching and orchestration.