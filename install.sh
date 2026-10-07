#!/bin/bash
# Install lspeak on this Mac: build the Go client into ~/.local/bin and run the
# Kokoro-FastAPI service (already cloned to ~/.local/share/kokoro-fastapi) as a
# LaunchAgent on 127.0.0.1:8880.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KOKORO="$HOME/.local/share/kokoro-fastapi"
LABEL="com.lspeak.kokoro"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGS="$HOME/.local/state/kokoro-fastapi"
BIN="$HOME/.local/bin/lspeak"

[ -x "$KOKORO/.venv/bin/uvicorn" ] || {
	echo "install.sh: $KOKORO/.venv/bin/uvicorn not found; set up Kokoro-FastAPI there first" >&2
	exit 1
}

# Retire the Python install and its semantic cache. This runs before the build:
# uv removes the executable it linked into ~/.local/bin.
if command -v uv >/dev/null 2>&1 && uv tool list 2>/dev/null | grep -q '^lspeak '; then
	uv tool uninstall lspeak
fi
rm -rf "$HOME/.cache/lspeak"

# Client.
mkdir -p "$HOME/.local/bin" "$LOGS" "$HOME/Library/LaunchAgents"
(cd "$REPO" && go build -o "$BIN.new" .)
rm -f "$BIN"
mv "$BIN.new" "$BIN"

# Service: the start-gpu_mac.sh environment with absolute paths, bound to loopback.
cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$KOKORO/.venv/bin/uvicorn</string>
        <string>api.src.main:app</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>8880</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$KOKORO</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>USE_GPU</key>
        <string>true</string>
        <key>PYTHONPATH</key>
        <string>$KOKORO:$KOKORO/api</string>
        <key>MODEL_DIR</key>
        <string>$KOKORO/api/src/models</string>
        <key>VOICES_DIR</key>
        <string>$KOKORO/api/src/voices/v1_0</string>
        <key>WEB_PLAYER_PATH</key>
        <string>$KOKORO/web</string>
        <key>DEVICE_TYPE</key>
        <string>mps</string>
        <key>PYTORCH_ENABLE_MPS_FALLBACK</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOGS/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$LOGS/stderr.log</string>
    <key>ThrottleInterval</key>
    <integer>30</integer>
</dict>
</plist>
EOF

DOMAIN="gui/$(id -u)"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
# bootout returns before the service is gone; bootstrap fails until it is.
for _ in $(seq 1 50); do
	launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || break
	sleep 0.2
done
launchctl bootstrap "$DOMAIN" "$PLIST"

echo "lspeak installed at $BIN; Kokoro service $LABEL loaded (http://127.0.0.1:8880)"
