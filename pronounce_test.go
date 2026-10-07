package main

import "testing"

func TestApplyPronunciations(t *testing.T) {
	cases := []struct {
		name    string
		entries []pronunciation
		in, out string
	}{
		{"whole word case sensitive", []pronunciation{{"Rudy", "ɹˈudi"}},
			"Rudy, Rudyard and rudy met Rudy.", "[Rudy](/ɹˈudi/), Rudyard and rudy met [Rudy](/ɹˈudi/)."},
		{"non-ascii key", []pronunciation{{"Zoë", "zˈoʊ"}},
			"Zoë and Zoëy and éZoë", "[Zoë](/zˈoʊ/) and Zoëy and éZoë"},
		{"unicode neighbour blocks match", []pronunciation{{"Rudy", "x"}}, "Rudyé Rudy", "Rudyé [Rudy](/x/)"},
		{"underscore and digits are word chars", []pronunciation{{"Rudy", "x"}}, "Rudy_1 2Rudy", "Rudy_1 2Rudy"},
		{"file order, later sees earlier output", []pronunciation{{"a", "b"}, {"b", "c"}},
			"a", "[a](/[b](/c/)/)"},
		{"no entries", nil, "Rudy", "Rudy"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := applyPronunciations(c.in, c.entries); got != c.out {
				t.Fatalf("got %q, want %q", got, c.out)
			}
		})
	}
}
