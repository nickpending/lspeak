package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var kokoroClient = &http.Client{
	Timeout:   60 * time.Second,
	Transport: &http.Transport{Proxy: nil},
}

// synthKokoro asks the Kokoro-FastAPI service for audio in the given format.
// Pronunciation overrides are applied here, for Kokoro text only.
func synthKokoro(cfg config, text, voice, format string) ([]byte, error) {
	body, err := json.Marshal(map[string]string{
		"model":           "kokoro",
		"input":           applyPronunciations(text, cfg.Pronunciations),
		"voice":           voice,
		"response_format": format,
	})
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(cfg.KokoroURL, "/") + "/v1/audio/speech"
	resp, err := kokoroClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("kokoro service %s unreachable: %w", cfg.KokoroURL, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("kokoro service %s: reading response: %w", cfg.KokoroURL, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("kokoro service %s returned %d: %s", cfg.KokoroURL, resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return data, nil
}

func sayArgs(voice string, extra ...string) []string {
	args := append([]string{}, extra...)
	if voice != "" {
		args = append(args, "-v", voice)
	}
	return args
}

// synthSystemToFile renders text with say and converts it to WAVE at dest.
func synthSystemToFile(text, voice, dest string) error {
	dir, err := os.MkdirTemp("", "lspeak-say-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	aiff := filepath.Join(dir, "say.aiff")
	args := append(sayArgs(voice, "-o", aiff), "--", text)
	if out, err := exec.Command("say", args...).CombinedOutput(); err != nil {
		return fmt.Errorf("say failed: %w: %s", err, strings.TrimSpace(string(out)))
	}
	if out, err := exec.Command("afconvert", "-f", "WAVE", "-d", "LEI16", aiff, dest).CombinedOutput(); err != nil {
		return fmt.Errorf("afconvert failed: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

// kokoroFormat picks the response_format from the output path's extension.
func kokoroFormat(path string) string {
	switch ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(path), ".")); ext {
	case "mp3", "wav", "opus", "flac", "aac":
		return ext
	}
	return "wav"
}

// writeOutput synthesizes in this process and atomically places the audio at
// path, which is resolved against the caller's cwd. The file appears
// complete or not at all.
func writeOutput(cfg config, provider, voice, text, path string) error {
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(abs), ".lspeak-*.tmp")
	if err != nil {
		return fmt.Errorf("cannot write %s: %w", path, err)
	}
	tmpName := tmp.Name()
	tmp.Close()
	done := false
	defer func() {
		if !done {
			os.Remove(tmpName)
		}
	}()

	switch provider {
	case "kokoro":
		data, err := synthKokoro(cfg, text, voice, kokoroFormat(path))
		if err != nil {
			return err
		}
		if err := os.WriteFile(tmpName, data, 0o644); err != nil {
			return fmt.Errorf("cannot write %s: %w", path, err)
		}
	case "system":
		if err := synthSystemToFile(text, voice, tmpName); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown provider %q", provider)
	}
	if err := os.Chmod(tmpName, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmpName, abs); err != nil {
		return fmt.Errorf("cannot write %s: %w", path, err)
	}
	done = true
	return nil
}
