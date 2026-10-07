package main

import (
	"os"
	"path/filepath"
	"testing"
)

func writeConfig(t *testing.T, body string) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	dir := filepath.Join(home, ".config", "lspeak")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.toml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestConfigMissingFileDefaults(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cfg, err := loadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Provider != "kokoro" || cfg.Voice != "af_heart" || cfg.KokoroURL != "http://127.0.0.1:8880" || len(cfg.Pronunciations) != 0 {
		t.Fatalf("cfg = %+v", cfg)
	}
}

func TestConfigFileOrderAndIgnoredTables(t *testing.T) {
	writeConfig(t, `
[tts]
provider = "system"
voice = "bf_emma"
device = "auto"

[tts.pronunciation]
Zed = "z1"
Alpha = "a1"
"Zoë" = "z2"

[http]
host = "127.0.0.1"

[cache]
enabled = true

[kokoro]
url = "http://127.0.0.1:9999"
`)
	cfg, err := loadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Provider != "system" || cfg.Voice != "bf_emma" || cfg.KokoroURL != "http://127.0.0.1:9999" {
		t.Fatalf("cfg = %+v", cfg)
	}
	want := []pronunciation{{"Zed", "z1"}, {"Alpha", "a1"}, {"Zoë", "z2"}}
	if len(cfg.Pronunciations) != len(want) {
		t.Fatalf("pronunciations = %+v", cfg.Pronunciations)
	}
	for i := range want {
		if cfg.Pronunciations[i] != want[i] {
			t.Fatalf("pronunciations = %+v, want %+v", cfg.Pronunciations, want)
		}
	}
}

func TestConfigMalformedIsAnError(t *testing.T) {
	writeConfig(t, "[tts\n")
	if _, err := loadConfig(); err == nil {
		t.Fatal("want error")
	}
}
