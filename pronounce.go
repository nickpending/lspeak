package main

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

func isWordRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r)
}

// boundaryAt reports whether Python's \b holds at byte offset i of s:
// exactly one side of i is a word rune.
func boundaryAt(s string, i int) bool {
	before, after := false, false
	if i > 0 {
		r, _ := utf8.DecodeLastRuneInString(s[:i])
		before = isWordRune(r)
	}
	if i < len(s) {
		r, _ := utf8.DecodeRuneInString(s[i:])
		after = isWordRune(r)
	}
	return before != after
}

// replaceWord rewrites each case-sensitive occurrence of word that sits on
// \b boundaries at both ends to [word](/ipa/).
func replaceWord(text, word, ipa string) string {
	if word == "" {
		return text
	}
	repl := "[" + word + "](/" + ipa + "/)"
	var b strings.Builder
	pos := 0
	for pos <= len(text) {
		idx := strings.Index(text[pos:], word)
		if idx < 0 {
			break
		}
		start := pos + idx
		end := start + len(word)
		if boundaryAt(text, start) && boundaryAt(text, end) {
			b.WriteString(text[pos:start])
			b.WriteString(repl)
			pos = end
			continue
		}
		_, size := utf8.DecodeRuneInString(text[start:])
		b.WriteString(text[pos : start+size])
		pos = start + size
	}
	b.WriteString(text[pos:])
	return b.String()
}

// applyPronunciations applies the entries in file order. Kokoro only.
func applyPronunciations(text string, entries []pronunciation) string {
	for _, e := range entries {
		text = replaceWord(text, e.Word, e.IPA)
	}
	return text
}
