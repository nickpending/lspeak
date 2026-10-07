package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"
)

// item is one queued sentence; it carries its provider and voice.
type item struct {
	Text     string `json:"text"`
	Provider string `json:"provider"`
	Voice    string `json:"voice"`
}

func stateDir() (string, error) {
	h, err := homeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(h, ".local", "state", "lspeak"), nil
}

func queueDir() (string, error) {
	s, err := stateDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(s, "queue"), nil
}

// spawnDrainer starts this executable detached with LSPEAK_DRAIN=1.
// It is a variable so tests can keep speech from playing.
var spawnDrainer = func() error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	cmd := exec.Command(exe)
	cmd.Env = append(os.Environ(), "LSPEAK_DRAIN=1")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return err
	}
	return cmd.Process.Release()
}

// enqueue writes the item atomically (dot-temp then rename) into the queue
// and starts a drainer. Names are zero-padded nanoseconds, so name order is
// time order.
func enqueue(it item) error {
	dir, err := queueDir()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(it)
	if err != nil {
		return err
	}
	name := fmt.Sprintf("%020d-%d", time.Now().UnixNano(), os.Getpid())
	tmp := filepath.Join(dir, "."+name+".tmp")
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, filepath.Join(dir, name)); err != nil {
		os.Remove(tmp)
		return err
	}
	return spawnDrainer()
}

func logError(format string, args ...any) {
	s, err := stateDir()
	if err != nil {
		return
	}
	os.MkdirAll(s, 0o755)
	f, err := os.OpenFile(filepath.Join(s, "lspeak.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "%s ERROR %s\n", time.Now().Format(time.RFC3339), fmt.Sprintf(format, args...))
}

func firstQueued(dir string) (string, bool) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", false
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		names = append(names, e.Name())
	}
	if len(names) == 0 {
		return "", false
	}
	sort.Strings(names)
	return names[0], true
}

// play synthesizes and plays one item; it returns when playback ends.
func play(cfg config, name string, it item) error {
	switch it.Provider {
	case "kokoro":
		data, err := synthKokoro(cfg, it.Text, it.Voice, "wav")
		if err != nil {
			return err
		}
		tmp := filepath.Join(os.TempDir(), "lspeak-"+name+".wav")
		if err := os.WriteFile(tmp, data, 0o600); err != nil {
			return err
		}
		defer os.Remove(tmp)
		if out, err := exec.Command("afplay", tmp).CombinedOutput(); err != nil {
			return fmt.Errorf("afplay failed: %w: %s", err, strings.TrimSpace(string(out)))
		}
	case "system":
		args := append(sayArgs(it.Voice), "--", it.Text)
		if out, err := exec.Command("say", args...).CombinedOutput(); err != nil {
			return fmt.Errorf("say failed: %w: %s", err, strings.TrimSpace(string(out)))
		}
	default:
		return fmt.Errorf("unknown provider %q", it.Provider)
	}
	return nil
}

// beforeUnlock runs after the drainer's final empty listing and before it
// unlocks: the window in which a late enqueue is only seen by the re-check.
// It is a variable so tests can enqueue inside that window.
var beforeUnlock = func() {}

// drain plays queued items one at a time, in name order, under an exclusive
// non-blocking flock; a second drainer exits at once. After unlocking it
// re-lists once so an item enqueued during shutdown is not stranded.
func drain() error {
	s, err := stateDir()
	if err != nil {
		return err
	}
	dir := filepath.Join(s, "queue")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	lock, err := os.OpenFile(filepath.Join(s, "drain.lock"), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer lock.Close()
	for {
		if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
			return nil // another drainer owns the queue
		}
		drainLocked(dir)
		beforeUnlock()
		syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)
		if _, more := firstQueued(dir); !more {
			return nil
		}
	}
}

func drainLocked(dir string) {
	for {
		name, ok := firstQueued(dir)
		if !ok {
			return
		}
		path := filepath.Join(dir, name)
		data, err := os.ReadFile(path)
		os.Remove(path)
		if err != nil {
			logError("queue item %s unreadable: %v", name, err)
			continue
		}
		var it item
		if err := json.Unmarshal(data, &it); err != nil {
			logError("queue item %s malformed: %v", name, err)
			continue
		}
		cfg, err := loadConfig()
		if err != nil {
			logError("config: %v (text %q)", err, it.Text)
			continue
		}
		if err := play(cfg, name, it); err != nil {
			logError("%v (text %q)", err, it.Text)
		}
	}
}
