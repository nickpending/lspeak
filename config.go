package main

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

const (
	defaultProvider = "kokoro"
	defaultVoice    = "af_heart"
	defaultKokoro   = "http://127.0.0.1:8880"
)

// pronunciation is one [tts.pronunciation] entry.
type pronunciation struct {
	Word string
	IPA  string
}

// config is the subset of ~/.config/lspeak/config.toml the client reads.
type config struct {
	Provider       string
	Voice          string
	KokoroURL      string
	Pronunciations []pronunciation // in file order
}

type configFile struct {
	TTS struct {
		Provider      string            `toml:"provider"`
		Voice         string            `toml:"voice"`
		Pronunciation map[string]string `toml:"pronunciation"`
	} `toml:"tts"`
	Kokoro struct {
		URL string `toml:"url"`
	} `toml:"kokoro"`
}

func homeDir() (string, error) {
	h := os.Getenv("HOME")
	if h == "" {
		return "", errors.New("HOME is not set")
	}
	return h, nil
}

func configPath() (string, error) {
	h, err := homeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(h, ".config", "lspeak", "config.toml"), nil
}

// loadConfig reads the config file; a missing file yields the defaults.
// The old [http] and [cache] tables and unknown keys are ignored.
func loadConfig() (config, error) {
	cfg := config{Provider: defaultProvider, Voice: defaultVoice, KokoroURL: defaultKokoro}
	path, err := configPath()
	if err != nil {
		return cfg, err
	}
	var raw configFile
	md, err := toml.DecodeFile(path, &raw)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return cfg, nil
		}
		return cfg, fmt.Errorf("reading %s: %w", path, err)
	}
	if raw.TTS.Provider != "" {
		cfg.Provider = raw.TTS.Provider
	}
	if raw.TTS.Voice != "" {
		cfg.Voice = raw.TTS.Voice
	}
	if raw.Kokoro.URL != "" {
		cfg.KokoroURL = raw.Kokoro.URL
	}
	for _, k := range md.Keys() {
		if len(k) == 3 && k[0] == "tts" && k[1] == "pronunciation" {
			if ipa, ok := raw.TTS.Pronunciation[k[2]]; ok {
				cfg.Pronunciations = append(cfg.Pronunciations, pronunciation{Word: k[2], IPA: ipa})
			}
		}
	}
	return cfg, nil
}
