package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeAfplay puts an afplay stand-in first on PATH. It appends "start <file>"
// and "end <file>" to a trace file, sleeping between them.
func fakeAfplay(t *testing.T) (trace string) {
	t.Helper()
	bin := t.TempDir()
	trace = filepath.Join(t.TempDir(), "trace")
	script := fmt.Sprintf("#!/bin/sh\necho \"start $1\" >> %s\nsleep 0.15\necho \"end $1\" >> %s\n", trace, trace)
	if err := os.WriteFile(filepath.Join(bin, "afplay"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	return trace
}

func readTrace(t *testing.T, trace string) []string {
	t.Helper()
	data, err := os.ReadFile(trace)
	if err != nil {
		t.Fatal(err)
	}
	return strings.Fields(strings.ReplaceAll(string(data), "\n", " "))
}

func TestDrainPlaysInOrderNeverOverlapping(t *testing.T) {
	home, _ := setup(t)
	trace := fakeAfplay(t)
	url, bodies := fakeKokoro(t, 200, "WAV")
	cfgDir := filepath.Join(home, ".config", "lspeak")
	os.MkdirAll(cfgDir, 0o755)
	os.WriteFile(filepath.Join(cfgDir, "config.toml"), []byte("[kokoro]\nurl = \""+url+"\"\n"), 0o644)

	for i := 0; i < 3; i++ {
		if err := enqueue(item{Text: fmt.Sprintf("sentence %d", i), Provider: "kokoro", Voice: "af_heart"}); err != nil {
			t.Fatal(err)
		}
	}
	// Two drainers start at once; exactly one may play.
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); drain() }()
	}
	wg.Wait()

	for i, b := range *bodies {
		if b["input"] != fmt.Sprintf("sentence %d", i) {
			t.Fatalf("synth order wrong: %v", *bodies)
		}
	}
	fields := readTrace(t, trace)
	if len(fields) != 12 { // 3 x (start path, end path)
		t.Fatalf("trace = %v", fields)
	}
	var last string
	for i := 0; i < len(fields); i += 4 {
		if fields[i] != "start" || fields[i+2] != "end" || fields[i+1] != fields[i+3] {
			t.Fatalf("overlap or disorder: %v", fields)
		}
		if fields[i+1] <= last {
			t.Fatalf("not in enqueue order: %v", fields)
		}
		last = fields[i+1]
	}
	if got := queued(t, home); len(got) != 0 {
		t.Fatalf("queue not empty: %v", got)
	}
}

func TestDrainRecheckPlaysItemEnqueuedBeforeUnlock(t *testing.T) {
	home, _ := setup(t)
	trace := fakeAfplay(t)
	url, _ := fakeKokoro(t, 200, "WAV")
	cfgDir := filepath.Join(home, ".config", "lspeak")
	os.MkdirAll(cfgDir, 0o755)
	os.WriteFile(filepath.Join(cfgDir, "config.toml"), []byte("[kokoro]\nurl = \""+url+"\"\n"), 0o644)

	fired := false
	beforeUnlock = func() {
		if fired {
			return
		}
		fired = true
		if err := enqueue(item{Text: "late", Provider: "kokoro", Voice: "v"}); err != nil {
			t.Error(err)
		}
	}
	t.Cleanup(func() { beforeUnlock = func() {} })

	if err := enqueue(item{Text: "early", Provider: "kokoro", Voice: "v"}); err != nil {
		t.Fatal(err)
	}
	if err := drain(); err != nil {
		t.Fatal(err)
	}
	if n := len(readTrace(t, trace)); n != 8 {
		t.Fatalf("trace has %d fields, want both items played", n)
	}
	if got := queued(t, home); len(got) != 0 {
		t.Fatalf("late item stranded: %v", got)
	}
}

func TestDrainRecheckPicksUpLateItem(t *testing.T) {
	home, _ := setup(t)
	trace := fakeAfplay(t)
	url, _ := fakeKokoro(t, 200, "WAV")
	cfgDir := filepath.Join(home, ".config", "lspeak")
	os.MkdirAll(cfgDir, 0o755)
	os.WriteFile(filepath.Join(cfgDir, "config.toml"), []byte("[kokoro]\nurl = \""+url+"\"\n"), 0o644)

	if err := enqueue(item{Text: "first", Provider: "kokoro", Voice: "v"}); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() { drain(); close(done) }()
	time.Sleep(60 * time.Millisecond) // first is playing
	if err := enqueue(item{Text: "second", Provider: "kokoro", Voice: "v"}); err != nil {
		t.Fatal(err)
	}
	<-done
	if n := len(readTrace(t, trace)); n != 8 {
		t.Fatalf("trace has %d fields, want both items played", n)
	}
}

func TestDrainFailureLogsURLAndTextAndDropsItem(t *testing.T) {
	home, _ := setup(t)
	fakeAfplay(t)
	cfgDir := filepath.Join(home, ".config", "lspeak")
	os.MkdirAll(cfgDir, 0o755)
	url := "http://127.0.0.1:1"
	os.WriteFile(filepath.Join(cfgDir, "config.toml"), []byte("[kokoro]\nurl = \""+url+"\"\n"), 0o644)

	if err := enqueue(item{Text: "unreachable", Provider: "kokoro", Voice: "v"}); err != nil {
		t.Fatal(err)
	}
	if err := drain(); err != nil {
		t.Fatal(err)
	}
	if got := queued(t, home); len(got) != 0 {
		t.Fatalf("item retried or kept: %v", got)
	}
	log, err := os.ReadFile(filepath.Join(home, ".local", "state", "lspeak", "lspeak.log"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(log), url) || !strings.Contains(string(log), `"unreachable"`) || !strings.Contains(string(log), "ERROR") {
		t.Fatalf("log = %q", log)
	}
}

func TestFirstQueuedSkipsDotfilesAndSortsByName(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{".00000000000000000001-1.tmp", "00000000000000000009-1", "00000000000000000002-5"} {
		os.WriteFile(filepath.Join(dir, n), nil, 0o644)
	}
	if name, ok := firstQueued(dir); !ok || name != "00000000000000000002-5" {
		t.Fatalf("name=%q ok=%v", name, ok)
	}
}
