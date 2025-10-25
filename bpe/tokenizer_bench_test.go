package bpe

import (
	"strings"
	"testing"
)

// Generate sample text of varying sizes
func generateText(size int) []byte {
	// Create text with some patterns to make merging interesting
	patterns := []string{
		"the quick brown fox jumps over the lazy dog ",
		"hello world this is a test ",
		"byte pair encoding is used for tokenization ",
		"machine learning models need tokenizers ",
	}

	var builder strings.Builder
	for builder.Len() < size {
		for _, p := range patterns {
			builder.WriteString(p)
			if builder.Len() >= size {
				break
			}
		}
	}

	return []byte(builder.String()[:size])
}

func BenchmarkTrain_1KB_Vocab300(b *testing.B) {
	text := generateText(1024) // 1KB
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		tokenizer := New()
		tokenizer.Train(text, 300)
	}
}

func BenchmarkTrain_10KB_Vocab300(b *testing.B) {
	text := generateText(10 * 1024) // 10KB
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		tokenizer := New()
		tokenizer.Train(text, 300)
	}
}

func BenchmarkTrain_10KB_Vocab500(b *testing.B) {
	text := generateText(10 * 1024) // 10KB
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		tokenizer := New()
		tokenizer.Train(text, 500)
	}
}

func BenchmarkTrain_100KB_Vocab500(b *testing.B) {
	text := generateText(100 * 1024) // 100KB
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		tokenizer := New()
		tokenizer.Train(text, 500)
	}
}

func BenchmarkTrain_100KB_Vocab1000(b *testing.B) {
	text := generateText(100 * 1024) // 100KB
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		tokenizer := New()
		tokenizer.Train(text, 1000)
	}
}

func BenchmarkEncode_1KB(b *testing.B) {
	text := generateText(1024)
	tokenizer := New()
	tokenizer.Train(text, 400)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.Encode(text)
	}
}

func BenchmarkEncode_10KB(b *testing.B) {
	text := generateText(10 * 1024)
	tokenizer := New()
	tokenizer.Train(text, 400)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.Encode(text)
	}
}

func BenchmarkDecode_1KB(b *testing.B) {
	text := generateText(1024)
	tokenizer := New()
	tokenizer.Train(text, 400)
	tokens := tokenizer.Encode(text)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.Decode(tokens)
	}
}
