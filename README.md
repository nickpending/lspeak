# lspeak

A small Go command that speaks text. Pipe text in, or pass it as arguments; the call returns at once, and sentences play in the order they were sent, one at a time, even when many processes call lspeak together.

```bash
echo "Deploy complete" | lspeak
lspeak -- "Build finished"
lspeak -v bf_emma "Good morning"
lspeak -o briefing.mp3 "Today's briefing"
```

macOS only: playback and the system provider use `afplay`, `say`, and `afconvert`, and the service runs under launchd.

## How it works

- **lspeak** is a Go client. It does no synthesis itself.
- **Kokoro-FastAPI** is the speech engine: a long-lived local service on `127.0.0.1:8880`, run as a LaunchAgent (`com.lspeak.kokoro`) with the model kept loaded.
- **Speak mode** (no `-o`) writes the sentence, with its provider and voice, to `~/.local/state/lspeak/queue/`, starts a detached drainer, and exits 0. The drainer holds an exclusive lock on `~/.local/state/lspeak/drain.lock`, synthesizes each item in enqueue order, plays it with `afplay`, and exits when the queue is empty. Failures are written to `~/.local/state/lspeak/lspeak.log` and the item is dropped; the caller never sees them.
- **Output mode** (`-o PATH`) synthesizes in the calling process, writes a temp file beside PATH, renames it onto PATH, and only then prints `Audio saved to PATH`. A relative PATH lands in the caller's working directory. On any failure it exits non-zero and leaves no file.

There is no cache: every sentence is synthesized fresh.

## Install

Prerequisites: Go, and Kokoro-FastAPI cloned to `~/.local/share/kokoro-fastapi` with its `.venv` and model downloaded.

```bash
./install.sh
```

This retires an old Python `uv tool` install of lspeak and `~/.cache/lspeak`, builds the client to `~/.local/bin/lspeak`, writes `~/Library/LaunchAgents/com.lspeak.kokoro.plist` (RunAtLoad, KeepAlive, loopback only, logs under `~/.local/state/kokoro-fastapi/`), and loads it with launchctl. It is safe to re-run.

## Usage

```
lspeak [-p kokoro|system] [-v voice] [-o file] [--no-cache] [--cache-threshold v] [--] [text...]
```

| Flag | Meaning |
|---|---|
| `--` | Everything after it is text. |
| `-p`, `--provider` | `kokoro` (default) or `system` (macOS `say`). |
| `-v`, `--voice` | Voice. Kokoro default is the config voice or `af_heart`; `system` passes `-v` to `say` only when given. |
| `-o`, `--output` | Write audio to a file instead of speaking. Kokoro picks the format from the extension (mp3, wav, opus, flac, aac; otherwise wav). The system provider always writes WAVE. |
| `--no-cache`, `--cache-threshold` | Accepted and ignored, so older callers keep working. |

With no text arguments, lspeak reads stdin. Any other flag exits 2 and names the flag. `--model` and `-p elevenlabs` are refused with a message that they were dropped.

## Configuration

`~/.config/lspeak/config.toml` is optional.

```toml
[tts]
provider = "kokoro"
voice = "bf_emma"

[tts.pronunciation]
Rudy = "ɹˈudi"      # Kokoro only; whole word, case-sensitive, applied in file order

[kokoro]
url = "http://127.0.0.1:8880"   # default
```

Old `[http]`, `[cache]`, and `device` settings are ignored.

## Development

```bash
go vet ./... && staticcheck ./... && gofmt -l . && go test ./...
```

## License

MIT
