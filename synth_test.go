package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// fakeKokoro serves /v1/audio/speech and records each request body.
func fakeKokoro(t *testing.T, status int, reply string) (url string, bodies *[]map[string]string) {
	t.Helper()
	var got []map[string]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/speech" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		data, _ := io.ReadAll(r.Body)
		var m map[string]string
		if err := json.Unmarshal(data, &m); err != nil {
			t.Errorf("bad body: %v", err)
		}
		got = append(got, m)
		w.WriteHeader(status)
		io.WriteString(w, reply)
	}))
	t.Cleanup(srv.Close)
	return srv.URL, &got
}

func TestSynthKokoroRequestAndPronunciation(t *testing.T) {
	url, bodies := fakeKokoro(t, 200, "AUDIO")
	cfg := config{KokoroURL: url, Pronunciations: []pronunciation{{"Rudy", "ɹˈudi"}}}
	data, err := synthKokoro(cfg, "Hi Rudy", "bf_emma", "mp3")
	if err != nil || string(data) != "AUDIO" {
		t.Fatalf("data=%q err=%v", data, err)
	}
	want := map[string]string{"model": "kokoro", "input": "Hi [Rudy](/ɹˈudi/)", "voice": "bf_emma", "response_format": "mp3"}
	got := (*bodies)[0]
	for k, v := range want {
		if got[k] != v {
			t.Fatalf("body[%s]=%q want %q", k, got[k], v)
		}
	}
}

func TestSynthKokoroErrorNamesStatusAndURL(t *testing.T) {
	url, _ := fakeKokoro(t, 500, "boom")
	_, err := synthKokoro(config{KokoroURL: url}, "x", "v", "wav")
	if err == nil || !strings.Contains(err.Error(), "500") || !strings.Contains(err.Error(), "boom") || !strings.Contains(err.Error(), url) {
		t.Fatalf("err = %v", err)
	}
}

func TestKokoroFormat(t *testing.T) {
	for path, want := range map[string]string{"a.mp3": "mp3", "a.WAV": "wav", "a.opus": "opus", "a.flac": "flac", "a.aac": "aac", "a.txt": "wav", "a": "wav"} {
		if got := kokoroFormat(path); got != want {
			t.Errorf("%s: %s, want %s", path, got, want)
		}
	}
}

func TestWriteOutputKokoroRelativeAtomic(t *testing.T) {
	url, bodies := fakeKokoro(t, 200, "MP3DATA")
	dir := t.TempDir()
	t.Chdir(dir)
	if err := writeOutput(config{KokoroURL: url}, "kokoro", "af_heart", "hello", "out.mp3"); err != nil {
		t.Fatal(err)
	}
	if data, err := os.ReadFile(filepath.Join(dir, "out.mp3")); err != nil || string(data) != "MP3DATA" {
		t.Fatalf("data=%q err=%v", data, err)
	}
	if (*bodies)[0]["response_format"] != "mp3" {
		t.Fatalf("format = %q", (*bodies)[0]["response_format"])
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 1 {
		t.Fatalf("leftovers: %v", entries)
	}
}

func TestWriteOutputFailureLeavesNothing(t *testing.T) {
	url, _ := fakeKokoro(t, 500, "boom")
	dir := t.TempDir()
	err := writeOutput(config{KokoroURL: url}, "kokoro", "v", "hello", filepath.Join(dir, "y.mp3"))
	if err == nil {
		t.Fatal("want error")
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Fatalf("leftovers: %v", entries)
	}
}

func TestWriteOutputServiceDownLeavesNothing(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	dir := t.TempDir()
	err := writeOutput(config{KokoroURL: url}, "kokoro", "v", "hello", filepath.Join(dir, "y.mp3"))
	if err == nil || !strings.Contains(err.Error(), url) {
		t.Fatalf("err = %v", err)
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Fatalf("leftovers: %v", entries)
	}
}

func TestWriteOutputSystemIsWave(t *testing.T) {
	for _, tool := range []string{"say", "afconvert", "afinfo"} {
		if _, err := exec.LookPath(tool); err != nil {
			t.Skipf("%s not available", tool)
		}
	}
	dir := t.TempDir()
	t.Chdir(dir)
	if err := writeOutput(config{}, "system", "", "hello there", "briefing.mp3"); err != nil {
		t.Fatal(err)
	}
	wav := filepath.Join(dir, "copy.wav")
	data, err := os.ReadFile(filepath.Join(dir, "briefing.mp3"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(wav, data, 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("afinfo", wav).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "WAVE") {
		t.Fatalf("afinfo: %v\n%s", err, out)
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 2 {
		t.Fatalf("leftovers: %v", entries)
	}
}
