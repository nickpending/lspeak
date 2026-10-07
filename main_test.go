package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// setup points HOME at a temp dir and stops the drainer from starting.
func setup(t *testing.T) (home string, spawned *int) {
	t.Helper()
	home = t.TempDir()
	t.Setenv("HOME", home)
	n := 0
	old := spawnDrainer
	spawnDrainer = func() error { n++; return nil }
	t.Cleanup(func() { spawnDrainer = old })
	return home, &n
}

func runArgs(args []string, stdin string, tty bool) (code int, stdout, stderr string) {
	var out, errb bytes.Buffer
	code = run(args, strings.NewReader(stdin), tty, &out, &errb)
	return code, out.String(), errb.String()
}

func queued(t *testing.T, home string) []item {
	t.Helper()
	dir := filepath.Join(home, ".local", "state", "lspeak", "queue")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		t.Fatal(err)
	}
	var items []item
	for _, e := range entries {
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		var it item
		if err := json.Unmarshal(data, &it); err != nil {
			t.Fatal(err)
		}
		items = append(items, it)
	}
	return items
}

func TestRefusals(t *testing.T) {
	cases := []struct {
		name  string
		args  []string
		stdin string
		tty   bool
		want  string
	}{
		{"unknown flag", []string{"--bogus", "hi"}, "", false, "--bogus"},
		{"model", []string{"--model", "eleven_turbo_v2_5", "hi"}, "", false, "--model was dropped"},
		{"elevenlabs", []string{"-p", "elevenlabs", "hi"}, "", false, "elevenlabs"},
		{"list voices", []string{"--list-voices"}, "", false, "--list-voices"},
		{"no text tty", nil, "", true, "no text"},
		{"no text empty stdin", nil, "  \n", false, "no text"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			home, spawned := setup(t)
			code, _, stderr := runArgs(c.args, c.stdin, c.tty)
			if code == 0 {
				t.Fatalf("exit 0, want non-zero")
			}
			if !strings.Contains(stderr, c.want) {
				t.Fatalf("stderr %q lacks %q", stderr, c.want)
			}
			if len(queued(t, home)) != 0 || *spawned != 0 {
				t.Fatalf("queue touched")
			}
		})
	}
}

func TestCallerArgv(t *testing.T) {
	cases := []struct {
		name  string
		args  []string
		stdin string
		want  item
	}{
		{"dash dash", []string{"--", "Hello", "there."}, "", item{"Hello there.", "kokoro", "af_heart"}},
		{"no dash dash", []string{"one line of text"}, "", item{"one line of text", "kokoro", "af_heart"}},
		{"momentum", []string{"--no-cache", "--provider", "kokoro", "--voice", "bf_emma", "--cache-threshold", "0.95", "A sentence."}, "", item{"A sentence.", "kokoro", "bf_emma"}},
		{"stdin", nil, "text from stdin\n", item{"text from stdin", "kokoro", "af_heart"}},
		{"dashed text after --", []string{"--", "-hello"}, "", item{"-hello", "kokoro", "af_heart"}},
		{"system no voice", []string{"-p", "system", "hi"}, "", item{"hi", "system", ""}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			home, spawned := setup(t)
			code, _, stderr := runArgs(c.args, c.stdin, false)
			if code != 0 {
				t.Fatalf("exit %d: %s", code, stderr)
			}
			got := queued(t, home)
			if len(got) != 1 || got[0] != c.want {
				t.Fatalf("queue = %+v, want [%+v]", got, c.want)
			}
			if *spawned != 1 {
				t.Fatalf("drainer spawned %d times", *spawned)
			}
		})
	}
}

func TestConfigVoiceUsedWhenNoFlag(t *testing.T) {
	home, _ := setup(t)
	cfgDir := filepath.Join(home, ".config", "lspeak")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cfgDir, "config.toml"), []byte("[tts]\nvoice = \"bf_emma\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if code, _, e := runArgs([]string{"--", "hi"}, "", false); code != 0 {
		t.Fatal(e)
	}
	if got := queued(t, home); len(got) != 1 || got[0].Voice != "bf_emma" {
		t.Fatalf("queue = %+v", got)
	}
}

func TestOutputToMissingDirFails(t *testing.T) {
	setup(t)
	missing := filepath.Join(t.TempDir(), "nope", "x.wav")
	code, stdout, stderr := runArgs([]string{"-o", missing, "text"}, "", false)
	if code == 0 || strings.Contains(stdout, "Audio saved") || !strings.Contains(stderr, "no such file") {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}
