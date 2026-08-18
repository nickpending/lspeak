---
type: project
domain: technical
status: active
started: 2025-09-25
---
# lspeak - Core Idea

## The Problem

**What specific problem does this solve?**

Modern AI voice synthesis has no clean unix tool. Developers need text-to-speech in their pipelines - for automation alerts, accessibility, voice interfaces, or just making their terminal talk. But existing solutions are broken: espeak sounds like a 1980s robot, old tools like ospeak are abandoned with outdated dependencies, and modern TTS services require clicking through web UIs or writing custom API wrappers. There's no simple way to pipe text to high-quality speech.

**Who has this problem?**

- CLI power users who want speech output from any command (`git log | lspeak`)
- Developers building voice features (Claude Code hooks, monitoring alerts, CI notifications)
- DevOps engineers who need audio alerts from scripts and automation
- Accessibility users who deserve better than espeak's robotic voice
- Anyone building terminal-based assistants or automation that needs to speak

**How do they solve it today?**

Badly. They either:

- Use espeak (sounds terrible, but it works)
- Write one-off Python scripts hitting TTS APIs
- Copy text to web UIs manually
- Use abandoned tools that break with "dependency not found"
- Build overengineered solutions with config files and frameworks
- Give up and just read the text themselves

## The Solution

**Core Value Proposition**

A unix-first TTS tool that just works: pipe any text to `lspeak` and hear it spoken in a natural AI voice. No configs, no frameworks - just high-quality speech that fits naturally in your command pipeline.

**Key Differentiators**

- **True unix philosophy**: Reads stdin, speaks audio. One tool, one job, done well.
- **Modern AI voices**: ElevenLabs quality, not 1980s robots (with free local TTS fallback)
- **Zero configuration**: API key in env var for premium voices, or use system TTS free
- **Just works**: `pip install lspeak` and you're talking
- **Library + CLI**: Import in Python or use from command line
- **Semantic caching**: Find similar phrases by meaning, not exact match (save $$$)

## System Flow (Initial Sketch)

> **Note**: This is a preliminary sketch of system operation. The actual workflow will evolve significantly during development.

1. Receive text input (stdin, file, or command arg)
2. Generate embedding and search semantic cache
3. If similarity > threshold, use cached audio
4. Otherwise, call TTS provider (ElevenLabs/system)
5. Cache result with embedding for future matches
6. Play audio through speakers or save to file

## User Experience Vision

**Primary User Journey**

1. Install with pip: `pip install lspeak`
2. Choose your setup:
   - Premium voices: `export ELEVENLABS_API_KEY=...`
   - Free voices: Works immediately with system TTS
3. Pipe any text: `echo "Hello world" | lspeak`
4. Hear speech immediately

**Core User Workflows**

- **Quick speech**: `lspeak "Deploy complete"`
- **Pipe from commands**: `git status | lspeak`
- **Read files**: `lspeak < notes.txt` or `lspeak -f notes.txt`
- **Voice selection**: `lspeak --list-voices` then `lspeak -v Rachel "Hello"`
- **Save output**: `lspeak -o alert.mp3 "System is down"`
- **Free local TTS**: `lspeak --provider=system "No API needed"`
- **Library usage**: `from lspeak import speak; speak("Hello", cache=True)`

**Success Criteria**

- Users can go from install to speech in under 60 seconds
- Works reliably in scripts and automation
- Speech quality makes people say "whoa, that's not espeak"
- Becomes the go-to tool for CLI text-to-speech

## MVP Definition

**What is the absolute minimum viable version?**

A command-line tool that takes text input and speaks it using ElevenLabs voices. Must handle basic unix patterns (stdin, args) and just work out of the box with minimal configuration.

**MVP Scope**

- Read text from stdin or command arguments
- Call ElevenLabs API with default voice
- Play audio through speakers using pygame
- Basic voice selection (list and choose)
- Save to file with `-o` option

**MVP Constraints**

- No caching (every call hits API)
- Default voice only unless specified
- No streaming (wait for full audio)
- No text preprocessing (speak as-is)
- Basic error messages only

**Post-MVP Evolution (Iteration 2)**

- Library interface for Python imports
- Semantic caching system (embeddings for similarity)
- Local TTS provider option (free alternative)
- Better error handling and recovery

## Features Status

**Status Legend:**

- 📋 **Planned** - Feature defined and ready for iteration planning
- 🔄 **In Progress** - Feature currently being developed
- ✅ **Built** - Feature completed and shipped

**Current Features:**

- ✅ **Text input handling** - Accept text from stdin, args, or files (iteration-1)
- ✅ **ElevenLabs integration** - API calls for text-to-speech conversion (iteration-1)
- ✅ **Audio playback** - Cross-platform playback via pygame (iteration-1)
- ✅ **Voice selection** - List and choose ElevenLabs voices (iteration-1)
- ✅ **File output** - Save audio to file instead of playing (iteration-1)
- ✅ **Environment config** - API key from environment variable (iteration-1)
- ✅ **Error handling** - Comprehensive error messages with debug mode (iteration-1)
- ✅ **Library interface** - Import and use from Python code (iteration-2)
- ✅ **Semantic caching** - Embeddings-based cache for similar phrases (iteration-2)
- ✅ **Local TTS provider** - System TTS for free usage (iteration-2)
- ✅ **Daemon architecture** - Sub-second responses via pre-loaded models (iteration-3)
- 🔄 **Speech queue** - Serial TTS playback for multiple processes (iteration-4)

**Future Features:**

- 🔄 **HTTP API Interface** - Broad integration beyond CLI (iteration-5)
- 📋 **OpenAI TTS** - Alternative premium provider
- 📋 **Advanced caching** - Semantic similarity matching
- 📋 **Text preprocessing** - Technical content transformation (moved to clarvis)
- 📋 **Streaming support** - Start playback before full download
- 📋 **Voice cloning** - Custom voice creation and management

## Technical Approach

**Architecture Decision**

- [X] **Single Tool/Application** - Integrated solution, focused functionality
- [ ] **Composed Tool Ecosystem** - Multiple tools with clean interfaces

**Why this approach?**

Users want one command that just works. A single `lspeak` binary that handles everything from API calls to audio playback. No need to pipe between multiple tools or manage complex configurations. 

**Design Philosophy**:
- lspeak stays simple - text in, speech out
- No complex text processing or intelligence
- Tools like clarvis handle chunking, preprocessing, queueing
- lspeak is called once per chunk/sentence for optimal caching
- Library interface allows direct integration without subprocess overhead

**Design Principles (from ClaudeX standards)**

Following composition-first architecture principles:

1. **Do One Thing Well**: Convert text to speech - nothing more, nothing less
2. **Expect Tool Composition**: Works as a filter in pipelines (`cat file | lspeak`)
3. **Clean Data Interfaces**: Clear input (text) and output (audio) contracts
4. **Fail Fast, Fail Clear**: Validate inputs early, report specific errors with fixes
5. **Scriptable by Default**: Minimal interactive prompts, predictable output
6. **Progressive Enhancement**: Basic usage simple, advanced features optional

**Dependencies & Prerequisites**

- ElevenLabs API key (user provides)
- Python 3.8+ (standard on most systems)
- pygame for audio playback
- Internet connection for API calls

**Integration Requirements**

- Environment variable for API key
- Standard unix pipes and file I/O
- System audio output (handled by pygame)
- Python library interface for tools like clarvis
- Simple text input only (no preprocessing)

**Data Requirements**

- Input text (ephemeral, single sentences/chunks)
- Text embeddings (stored with cached audio)
- Audio data (cached to avoid repeated API calls)
- Voice list (cached after first fetch)
- Cache storage (~/.cache/lspeak/ with embeddings + audio)
- FAISS index for fast similarity search

**Key Technical Constraints**

- API rate limits (user's responsibility)
- Audio format compatibility (MP3/WAV)
- Cross-platform audio playback
- Package size (pygame adds ~15MB)

**Configuration Management (ClaudeX Standards)**

Following environment-first configuration:

- **Environment Variables**: 
  - `ELEVENLABS_API_KEY` - Required, no default
  - `LSPEAK_DEFAULT_VOICE` - Optional voice ID
  - `LSPEAK_CACHE_DIR` - Optional cache location
- **Configuration Validation**: Fail fast on startup if API key missing
- **No Config Files**: Environment variables only (unix philosophy)
- **Secrets Management**: Never log API keys, sanitize in errors

**Error Handling Strategy (ClaudeX Standards)**

Following fail-fast and explicit error principles:

- **API Errors**: 
  - Rate limit (429): Exponential backoff with jitter
  - Auth failure (401): Clear message about API key
  - Server errors (5xx): Retry with circuit breaker
- **Audio Errors**:
  - Playback failure: Fallback to file save with warning
  - Format issues: Convert or report specific codec
- **Input Errors**:
  - Empty text: Exit code 1 with usage hint
  - File not found: Specific path in error message
- **Resource Cleanup**: Always close audio streams, temp files

## Technical Architecture (Tentative)

> **Note**: This section captures current technical thinking and design exploration. All architectural decisions and implementation details are subject to change based on validation, iteration learnings, and practical constraints discovered during development.

**Data Design (Draft)**

Following ClaudeX type safety and validation principles:

```python
from typing import Union, TextIO, Iterator
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Input contracts with validation
@dataclass
class TextInput:
    """Text input from various sources"""
    source: Union[str, TextIO]
    
    def __post_init__(self):
        # Validation at boundaries
        if isinstance(self.source, str) and not self.source.strip():
            raise ValueError("Empty text input")

@dataclass 
class VoiceConfig:
    """Voice configuration with defaults"""
    voice_id: str = "default"
    stability: float = 0.5
    similarity_boost: float = 0.75
    
    def __post_init__(self):
        # Validate ranges
        if not 0 <= self.stability <= 1:
            raise ValueError("Stability must be 0-1")
        if not 0 <= self.similarity_boost <= 1:
            raise ValueError("Similarity boost must be 0-1")

# Output contracts
AudioData = bytes  # MP3 or WAV format

# Cache entry for iteration 2
@dataclass
class CacheEntry:
    text: str  # Original text for reference
    embedding: list[float]  # Text embedding for similarity
    audio_path: Path
    voice_id: str
    timestamp: datetime
    file_size: int  # For cache eviction
```

**Component Architecture (Working Model)**

Following ClaudeX Python src/ layout (prevents import issues):

```
lspeak/
├── src/
│   └── lspeak/
│       ├── __init__.py
│       ├── __main__.py      # CLI entry point (uv run python -m lspeak)
│       ├── cli.py           # Typer-based CLI definition
│       ├── core.py          # Core business logic (separate from CLI)
│       ├── tts/
│       │   ├── __init__.py
│       │   ├── client.py    # ElevenLabs API client
│       │   ├── models.py    # Data models with validation
│       │   └── errors.py    # Custom exceptions
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── player.py    # Cross-platform playback
│       │   └── writer.py    # File output handling
│       ├── cache/           # Iteration 2: Simple caching
│       │   ├── __init__.py
│       │   └── manager.py   # Hash-based cache
│       ├── providers/       # Iteration 2: Multiple TTS
│       │   ├── __init__.py
│       │   ├── elevenlabs.py
│       │   └── system.py    # Local TTS
│       └── embeddings/      # Iteration 2: Semantic cache
│           ├── __init__.py
│           └── openai.py    # Generate embeddings
├── tests/
│   ├── __init__.py
│   ├── unit/               # Isolated component tests
│   └── integration/        # Real API/audio tests
├── pyproject.toml          # Modern Python packaging
├── README.md
└── .gitignore
```

**Integration Points (Planned)**

- stdin/stdout for unix pipeline compatibility
- ElevenLabs REST API for TTS
- pygame.mixer for cross-platform audio
- Python library interface: `from lspeak import speak`
- System TTS via native OS commands (say/espeak/SAPI)
- OpenAI embeddings API for semantic cache
- FAISS for fast similarity search
- File-based cache in ~/.cache/lspeak/

**Tool/Technology Stack (Current Thinking)**

Based on ClaudeX Python standards:

- **Language**: Python 3.13+ (latest stable with JIT compilation and enhanced interpreter)
- **Package Management**: uv 0.7.15 (replaces pip, poetry, virtualenv, pyenv - 10-100x faster)
- **CLI Framework**: Typer (modern type-hinted wrapper over Click)
- **HTTP Client**: httpx (async-first, modern replacement for requests, HTTP/2 support)
- **Audio Playback**: pygame (most reliable cross-platform, despite being heavyweight)
- **Testing**: pytest 8.3+ with real integration tests (no mocking internal code)
- **Code Quality**: ruff 0.9+ (10x faster than pylint, includes formatting)
- **Type Checking**: mypy 1.15+ (catch bugs before runtime)
- **Semantic Cache**: OpenAI embeddings + FAISS for similarity search (CRITICAL for cost)
- **System TTS**: Native OS commands for local TTS (say on macOS, espeak on Linux, SAPI on Windows)
- **Library Interface**: Clean Python API for direct integration

## Implementation Strategy (Subject to Change)

> **Note**: This section explores potential implementation approaches and operational considerations. These represent current thinking and will evolve significantly based on prototyping results, user feedback, and practical constraints discovered during development.

**Iteration Priorities (Draft)**

1. **Core CLI + TTS**: Get basic speech working
2. **Voice management**: List and select voices
3. **Polish + packaging**: Error handling, PyPI release
4. **Semantic caching**: Add intelligence to reduce API calls
5. **Provider abstraction**: Support multiple TTS services

**Deployment/Operations (Initial Thoughts)**

- PyPI package: `pip install lspeak` (though we develop with uv)
- Single entry point: `lspeak` command  
- No daemon, no service, no state (except future cache)
- User manages their own API keys and limits

**Development Workflow (ClaudeX Python Standards)**

Setup and daily commands:

```bash
# Initial setup
uv init lspeak --python 3.13
cd lspeak
uv add httpx typer pygame elevenlabs  # Core deps
uv add --group dev pytest ruff mypy pytest-cov  # Dev tools

# Daily development
uv sync                    # Sync environment when switching projects
uv run python -m lspeak    # Run the tool
uv run pytest -xvs         # Run tests with verbose output
uv run ruff check .        # Lint and format
uv run mypy src/           # Type checking

# Before committing
uv run pytest --cov=lspeak --cov-report=term-missing
uv run ruff check . --fix  # Auto-fix issues
```

**Data Flow (Conceptual)**

```
Text Input → Validation → API Call → Audio Data → Playback/Save
                              ↓
                        (Future: Cache Check)
```

**Decision Logic (Draft)**

- If stdin has data, use it
- Else if args provided, join as text
- Else if -f specified, read file
- If -o specified, save instead of play
- If cache hit (future), skip API call

**Automation/Orchestration (Exploration)**

- Designed for shell scripts and pipelines
- Exit codes for scripting (0=success, 1=error)  
- Quiet mode for automation
- JSON output for voice listing (scriptable)

**Testing Philosophy (ClaudeX Standards)**

Following ClaudeX testing principles - NO MOCKING INTERNAL CODE:

**Unit Tests (Isolated Component Testing)**:
- Test individual functions in complete isolation
- Mock ONLY external dependencies (ElevenLabs API)
- Fast execution (<100ms per test)
- High coverage (>90%) with edge cases
- Use hypothesis for property-based testing

**Integration Tests (Real System Testing)**:
- Test with REAL audio playback (pygame)
- Test with REAL file I/O (temporary directories)
- Test with REAL ElevenLabs API (test account)
- Test actual command-line execution
- Verify data flows end-to-end

**What We Mock**:
- ✅ ElevenLabs API calls (expensive, rate-limited)
- ✅ Destructive operations (if any)
- ❌ NEVER mock our own TTS client
- ❌ NEVER mock audio playback logic
- ❌ NEVER mock file operations

**Logging & Observability (ClaudeX Standards)**

Structured logging with appropriate levels:

```python
import structlog

logger = structlog.get_logger()

# Log levels by context:
logger.debug("api_request", url=url, voice_id=voice_id)  # Dev only
logger.info("audio_generated", duration=duration, size=size)  # Normal ops
logger.warning("fallback_to_file", error=str(e))  # Degraded mode
logger.error("api_failed", status_code=429, retry_after=60)  # Failures

# NEVER log:
# - API keys or tokens
# - Full text content (privacy)
# - User personal information
```

**Code Organization Principles (ClaudeX Standards)**

Following single responsibility and dependency injection:

```python
# ✅ GOOD: Separate concerns, inject dependencies
class TTSClient:
    def __init__(self, api_key: str, http_client: httpx.Client):
        self._api_key = api_key
        self._client = http_client  # Injected, not created
    
    def synthesize(self, text: str, voice: VoiceConfig) -> AudioData:
        # Single responsibility: API communication only
        pass

# ❌ BAD: Mixed concerns, creates dependencies
class TTSClient:
    def __init__(self):
        self._api_key = os.getenv("API_KEY")  # Reads env
        self._client = httpx.Client()  # Creates dependency
        pygame.mixer.init()  # Initializes audio?!
```

## Learning and Evolution

**Key Learnings**

- Python audio ecosystem is a mess (simpleaudio abandoned, everything else needs ffmpeg)
- pygame is heavyweight but most reliable for cross-platform audio
- ElevenLabs API is straightforward and well-documented
- Users value simplicity over features for CLI tools

**Evolution Notes**

- Started exploring espeak-style naming (elspeak, elsay)
- Settled on lspeak (L for eLevenlabs, follows espeak pattern)
- Dropped streaming for MVP (complexity not worth it yet)
- Iteration 1 complete: Full MVP with ElevenLabs integration
- Iteration 2 planning: Library interface, caching, local TTS option
- Clarvis identified as separate project for Claude Code voice features
- Text preprocessing moving to clarvis (domain-specific needs)

## Open Questions

**User/Market Questions**

- How much are users willing to pay for API calls?
- Is voice selection important or do most stick with defaults?
- Do users want multiple provider support or is ElevenLabs enough?

**Technical Questions**

- Optimal similarity threshold for cache hits (0.95? 0.98?)
- Best embedding model (text-embedding-3-small?)
- Cache eviction strategy when storage fills up?

**Operational Questions**

- PyPI package naming (is lspeak available?)
- How to handle API key security in examples?
- Documentation hosting (just README or full docs?)

## Success Metrics

**Primary Metrics**

- GitHub stars and PyPI downloads
- Time from install to first speech
- User feedback: "replaced espeak with lspeak"

**Learning Metrics**

- API call patterns (cache hit rate potential)
- Common voice selections
- Error types and frequencies

## Risks and Assumptions

**Key Assumptions**

- Users want high-quality voices OR free local TTS
- pygame audio playback works reliably across platforms
- Text is generally short (not reading novels)
- Python projects will benefit from library interface

**Primary Risks**

- ElevenLabs API changes or becomes unavailable
- pygame installation issues on some systems
- API costs discourage usage
- Another tool solves this first

**Mitigation Strategies**

- Local TTS fallback for zero-cost option
- Clear pygame installation docs
- Semantic caching dramatically reduces API costs
- Library interface enables custom integrations

---

## Next Steps

1. ✅ MVP Complete (iteration-1)
2. Add library interface for Python imports
3. Implement caching system
4. Add local TTS provider option
5. Update documentation
6. Test with clarvis integration

**This Document Should:**

- Guide MVP development with clear scope
- Track feature status through iterations
- Acknowledge technical uncertainties
- Focus on shipping working software fast
- Evolve based on real user feedback