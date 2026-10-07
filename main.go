package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

type options struct {
	output   string
	voice    string
	provider string
	text     string
	hasText  bool
}

const usage = "usage: lspeak [-p kokoro|system] [-v voice] [-o file] [--no-cache] [--cache-threshold v] [--] [text...]"

func parseArgs(args []string) (options, error) {
	var o options
	var words []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" {
			words = append(words, args[i+1:]...)
			break
		}
		if !strings.HasPrefix(a, "-") || a == "-" {
			words = append(words, a)
			continue
		}
		name, inline, hasInline := strings.Cut(a, "=")
		get := func() (string, error) {
			if hasInline {
				return inline, nil
			}
			i++
			if i >= len(args) {
				return "", fmt.Errorf("flag %s needs a value", name)
			}
			return args[i], nil
		}
		var err error
		switch name {
		case "-o", "--output":
			o.output, err = get()
		case "-v", "--voice":
			o.voice, err = get()
		case "-p", "--provider":
			o.provider, err = get()
		case "--cache-threshold":
			_, err = get()
		case "--no-cache":
		case "--model":
			return o, errors.New("--model was dropped: lspeak no longer takes a model (elevenlabs is gone)")
		default:
			return o, fmt.Errorf("unknown flag %s", name)
		}
		if err != nil {
			return o, err
		}
	}
	o.text = strings.Join(words, " ")
	o.hasText = len(words) > 0
	switch o.provider {
	case "", "kokoro", "system":
	case "elevenlabs":
		return o, errors.New("provider elevenlabs was dropped: use kokoro or system")
	default:
		return o, fmt.Errorf("unknown provider %q: use kokoro or system", o.provider)
	}
	return o, nil
}

func run(args []string, stdin io.Reader, stdinIsTTY bool, stdout, stderr io.Writer) int {
	o, err := parseArgs(args)
	if err != nil {
		fmt.Fprintf(stderr, "lspeak: %v\n%s\n", err, usage)
		return 2
	}
	text := o.text
	if !o.hasText {
		if stdinIsTTY {
			fmt.Fprintf(stderr, "lspeak: no text given\n%s\n", usage)
			return 2
		}
		data, err := io.ReadAll(stdin)
		if err != nil {
			fmt.Fprintf(stderr, "lspeak: reading stdin: %v\n", err)
			return 1
		}
		text = strings.TrimSpace(string(data))
	}
	if strings.TrimSpace(text) == "" {
		fmt.Fprintf(stderr, "lspeak: no text given\n%s\n", usage)
		return 2
	}
	cfg, err := loadConfig()
	if err != nil {
		fmt.Fprintf(stderr, "lspeak: %v\n", err)
		return 1
	}
	provider := o.provider
	if provider == "" {
		provider = cfg.Provider
	}
	if provider != "kokoro" && provider != "system" {
		fmt.Fprintf(stderr, "lspeak: provider %q in config is not supported (elevenlabs was dropped): use kokoro or system\n", provider)
		return 2
	}
	voice := o.voice
	if voice == "" && provider == "kokoro" {
		voice = cfg.Voice
	}
	if o.output != "" {
		if err := writeOutput(cfg, provider, voice, text, o.output); err != nil {
			fmt.Fprintf(stderr, "lspeak: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "Audio saved to %s\n", o.output)
		return 0
	}
	if err := enqueue(item{Text: text, Provider: provider, Voice: voice}); err != nil {
		fmt.Fprintf(stderr, "lspeak: %v\n", err)
		return 1
	}
	return 0
}

func stdinIsTerminal() bool {
	fi, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

func main() {
	if os.Getenv("LSPEAK_DRAIN") == "1" {
		os.Unsetenv("LSPEAK_DRAIN")
		if err := drain(); err != nil {
			logError("drainer: %v", err)
			os.Exit(1)
		}
		return
	}
	os.Exit(run(os.Args[1:], os.Stdin, stdinIsTerminal(), os.Stdout, os.Stderr))
}
