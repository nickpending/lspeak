package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// TestBuiltBinarySpeakReturnsFast runs the real binary, with the real
// detached drainer fork, ten times back to back (the Stop-hook path) and
// requires each call to exit 0 in under 250 ms. The Kokoro URL points at a
// closed port, so nothing plays; the drainer logs and drops each item.
func TestBuiltBinarySpeakReturnsFast(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "lspeak")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	home := t.TempDir()
	cfgDir := filepath.Join(home, ".config", "lspeak")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cfgDir, "config.toml"), []byte("[kokoro]\nurl = \"http://127.0.0.1:1\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// The first exec of a fresh binary pays macOS's one-time scan; the
	// installed binary has already paid it.
	_ = exec.Command(bin, "--bogus").Run()
	for i := 0; i < 10; i++ {
		cmd := exec.Command(bin, "--", "timing check")
		cmd.Env = append(os.Environ(), "HOME="+home, "LSPEAK_DRAIN=")
		start := time.Now()
		out, err := cmd.CombinedOutput()
		elapsed := time.Since(start)
		if err != nil {
			t.Fatalf("call %d: %v\n%s", i, err, out)
		}
		t.Logf("call %d: %v", i, elapsed)
		if elapsed >= 250*time.Millisecond {
			t.Fatalf("call %d took %v, want under 250ms", i, elapsed)
		}
	}
}
