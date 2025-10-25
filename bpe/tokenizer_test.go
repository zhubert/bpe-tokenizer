package bpe

import (
	"bytes"
	"testing"
)

func TestNew(t *testing.T) {
	tokenizer := New()

	// Should have 256 base tokens (one per byte)
	if tokenizer.VocabSize != 256 {
		t.Errorf("Expected vocab size 256, got %d", tokenizer.VocabSize)
	}

	// Check that vocabulary contains all bytes
	if len(tokenizer.Vocabulary) != 256 {
		t.Errorf("Expected 256 vocabulary entries, got %d", len(tokenizer.Vocabulary))
	}

	// Verify a few byte mappings
	for i := 0; i < 256; i++ {
		if len(tokenizer.Vocabulary[i]) != 1 || tokenizer.Vocabulary[i][0] != byte(i) {
			t.Errorf("Vocabulary entry %d is incorrect", i)
		}
	}
}

func TestEncodeDecodeWithoutTraining(t *testing.T) {
	tokenizer := New()
	text := []byte("Hello, World!")

	// Encode
	tokens := tokenizer.Encode(text)

	// Without training, each byte should be its own token
	if len(tokens) != len(text) {
		t.Errorf("Expected %d tokens, got %d", len(text), len(tokens))
	}

	// Decode
	decoded := tokenizer.Decode(tokens)

	if !bytes.Equal(decoded, text) {
		t.Errorf("Decoded text doesn't match original.\nExpected: %s\nGot: %s", text, decoded)
	}
}

func TestTrainSimple(t *testing.T) {
	tokenizer := New()
	text := []byte("aaabdaaabac")

	// Train to merge a few pairs
	err := tokenizer.Train(text, 260)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Should have learned 4 merges (260 - 256)
	if len(tokenizer.Merges) != 4 {
		t.Errorf("Expected 4 merges, got %d", len(tokenizer.Merges))
	}

	// Vocabulary size should be 260
	if tokenizer.VocabSize != 260 {
		t.Errorf("Expected vocab size 260, got %d", tokenizer.VocabSize)
	}
}

func TestTrainAndEncode(t *testing.T) {
	tokenizer := New()
	trainText := []byte("low lower lowest")

	// Train
	err := tokenizer.Train(trainText, 270)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Encode the same text
	tokens := tokenizer.Encode(trainText)

	// Should be fewer tokens than bytes due to merges
	if len(tokens) >= len(trainText) {
		t.Errorf("Expected fewer tokens than bytes after training. Bytes: %d, Tokens: %d", len(trainText), len(tokens))
	}

	// Decode should give us back the original text
	decoded := tokenizer.Decode(tokens)
	if !bytes.Equal(decoded, trainText) {
		t.Errorf("Decoded text doesn't match original.\nExpected: %s\nGot: %s", trainText, decoded)
	}
}

func TestTrainAndEncodeNew(t *testing.T) {
	tokenizer := New()
	trainText := []byte("aaabdaaabac")

	// Train
	err := tokenizer.Train(trainText, 260)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Encode new text with similar patterns
	newText := []byte("aaab")
	tokens := tokenizer.Encode(newText)

	// Decode should give us back the text
	decoded := tokenizer.Decode(tokens)
	if !bytes.Equal(decoded, newText) {
		t.Errorf("Decoded text doesn't match original.\nExpected: %s\nGot: %s", newText, decoded)
	}

	// Should use fewer tokens than bytes if "aa" or "aaa" was learned
	if len(tokens) >= len(newText) {
		t.Logf("Note: Encoding didn't compress the text (tokens=%d, bytes=%d)", len(tokens), len(newText))
	}
}

func TestTargetVocabSizeTooSmall(t *testing.T) {
	tokenizer := New()
	text := []byte("test")

	// Target vocab size must be > 256
	err := tokenizer.Train(text, 256)
	if err == nil {
		t.Error("Expected error for target vocab size <= 256")
	}

	err = tokenizer.Train(text, 100)
	if err == nil {
		t.Error("Expected error for target vocab size < 256")
	}
}

func TestEmptyText(t *testing.T) {
	tokenizer := New()
	text := []byte("")

	// Should handle empty text gracefully
	tokens := tokenizer.Encode(text)
	if len(tokens) != 0 {
		t.Errorf("Expected 0 tokens for empty text, got %d", len(tokens))
	}

	decoded := tokenizer.Decode(tokens)
	if !bytes.Equal(decoded, text) {
		t.Errorf("Decoded empty text should be empty")
	}
}

func TestSingleByte(t *testing.T) {
	tokenizer := New()
	text := []byte("a")

	err := tokenizer.Train(text, 260)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Single byte should encode to single token
	tokens := tokenizer.Encode(text)
	if len(tokens) != 1 {
		t.Errorf("Expected 1 token, got %d", len(tokens))
	}

	decoded := tokenizer.Decode(tokens)
	if !bytes.Equal(decoded, text) {
		t.Errorf("Decoded text doesn't match original")
	}
}

func TestRepeatedPattern(t *testing.T) {
	tokenizer := New()
	// Lots of repetition should create effective merges
	text := []byte("ababababab")

	err := tokenizer.Train(text, 260)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	tokens := tokenizer.Encode(text)

	// Should compress well due to repeated "ab" pattern
	if len(tokens) >= len(text) {
		t.Logf("Warning: Expected compression for repeated pattern. Bytes: %d, Tokens: %d", len(text), len(tokens))
	}

	decoded := tokenizer.Decode(tokens)
	if !bytes.Equal(decoded, text) {
		t.Errorf("Decoded text doesn't match original.\nExpected: %s\nGot: %s", text, decoded)
	}
}

func TestDecodeInvalidToken(t *testing.T) {
	tokenizer := New()

	// Token ID that doesn't exist in vocabulary
	tokens := []int{999999}

	decoded := tokenizer.Decode(tokens)

	// Should return empty for invalid token
	if len(decoded) != 0 {
		t.Errorf("Expected empty result for invalid token, got %d bytes", len(decoded))
	}
}

func TestMergeOrder(t *testing.T) {
	tokenizer := New()
	text := []byte("aaa")

	err := tokenizer.Train(text, 258)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Should have learned 2 merges
	if len(tokenizer.Merges) != 2 {
		t.Errorf("Expected 2 merges, got %d", len(tokenizer.Merges))
	}

	// First merge should be 'a' + 'a' (both should be 97, the ASCII code for 'a')
	if tokenizer.Merges[0].First != 97 || tokenizer.Merges[0].Second != 97 {
		t.Errorf("First merge should be 'a'+'a' (97+97), got %d+%d",
			tokenizer.Merges[0].First, tokenizer.Merges[0].Second)
	}
}
