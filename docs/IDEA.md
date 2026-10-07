---
type: project
domain: technical
status: active
started: 2025-09-25
---
# lspeak - Core Idea

## The Problem

Modern AI voice synthesis has no clean unix tool. Developers want text-to-speech in their pipelines: automation alerts, Claude Code hooks, accessibility, voice interfaces. espeak sounds like a 1980s robot, and good TTS engines need custom API wrappers or web UIs. There is no simple way to pipe text to high-quality speech.

A second problem shows up the moment several programs speak at once: their audio overlaps unless something serializes it.

## The Solution

A unix-first command: pipe any text to `lspeak` and hear it in a natural local voice. No cloud, no API keys, no cost.

- **Returns at once.** Speaking never blocks the caller, so a hook that speaks pays only process start, a file write, and a fork.
- **Never talks over itself.** Sentences queue on disk and one drainer plays them in the order they were sent.
- **Synchronous file output.** `-o PATH` synthesizes in the calling process and the file exists, complete, when the command exits.

## Shape

lspeak is a small Go client. The speech engine is Kokoro-FastAPI, a long-lived local service on `127.0.0.1:8880` that keeps the model loaded and is run by a launchd LaunchAgent that `install.sh` writes. Keeping the engine in its own process means the client carries no ML stack, starts in milliseconds, and cannot crash on a native library conflict.

```
caller -> lspeak -> queue dir -> drainer (flock) -> Kokoro-FastAPI -> afplay
                \-> -o PATH: Kokoro-FastAPI (or say + afconvert) -> temp file -> rename
```

Providers: `kokoro` (default) and `system` (macOS `say`, converted to WAVE with `afconvert` for `-o`).

## Callers and the contract

Every live caller keeps working flag for flag: `lspeak -- <sentence>` (bench and sable hooks), `lspeak <line>`, `lspeak --no-cache --provider kokoro --voice <v> --cache-threshold <n> <sentence>`, stdin, and `lspeak --provider system --no-cache -o <file> <script>` (prismis). Anything else is refused with exit 2 naming the flag.

## Decisions and sacrifices

- No semantic cache: a warm Kokoro service synthesizes a sentence in about 0.2 s, so every sentence is synthesized fresh.
- Speak mode reports no error to its caller; a down service or failed item appears only in `~/.local/state/lspeak/lspeak.log`, and the item is dropped, not retried.
- Playback starts after a whole sentence is synthesized; there is no streaming into the player.
- Pronunciation overrides from `[tts.pronunciation]` rewrite whole-word, case-sensitive matches to Kokoro's inline `[word](/ipa/)` syntax, for the Kokoro provider only.
- macOS only.

## Out of scope

ElevenLabs, `--model`, `--file`, `--list-voices`, queue and daemon controls, an HTTP API, and any waiting for the service to finish booting. Moving playback to a house-wide speech processor would change only the client's playback step.
