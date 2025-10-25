package bpe

import (
	"fmt"
)

// Tokenizer represents a BPE tokenizer with learned merge rules
type Tokenizer struct {
	// Vocabulary maps token IDs to their byte representations
	Vocabulary map[int][]byte

	// Merges stores the merge rules in the order they were learned
	// Each merge is a pair of token IDs that should be merged
	Merges []Merge

	// VocabSize is the current size of the vocabulary
	VocabSize int
}

// Merge represents a single merge rule
type Merge struct {
	First  int // First token ID
	Second int // Second token ID
	Result int // Resulting merged token ID
}

// New creates a new BPE tokenizer initialized with byte-level vocabulary
func New() *Tokenizer {
	vocab := make(map[int][]byte)

	// Initialize with all possible byte values (0-255)
	for i := 0; i < 256; i++ {
		vocab[i] = []byte{byte(i)}
	}

	return &Tokenizer{
		Vocabulary: vocab,
		Merges:     []Merge{},
		VocabSize:  256,
	}
}

// Train learns BPE merges from the training text
// targetVocabSize is the desired final vocabulary size
func (t *Tokenizer) Train(text []byte, targetVocabSize int) error {
	if targetVocabSize <= 256 {
		return fmt.Errorf("target vocabulary size must be > 256")
	}

	// Start with each byte as a separate token
	tokens := make([]int, len(text))
	for i, b := range text {
		tokens[i] = int(b)
	}

	// Learn merges until we reach target vocabulary size
	for t.VocabSize < targetVocabSize {
		// Find the most frequent pair
		pair, count := t.findMostFrequentPair(tokens)
		if count == 0 {
			// No more pairs to merge
			break
		}

		// Create new token for this merge
		newTokenID := t.VocabSize

		// Add to vocabulary (concatenate the two tokens)
		firstBytes := t.Vocabulary[pair[0]]
		secondBytes := t.Vocabulary[pair[1]]
		newBytes := append([]byte{}, firstBytes...)
		newBytes = append(newBytes, secondBytes...)
		t.Vocabulary[newTokenID] = newBytes

		// Record the merge
		t.Merges = append(t.Merges, Merge{
			First:  pair[0],
			Second: pair[1],
			Result: newTokenID,
		})

		// Apply the merge to tokens
		tokens = t.applyMerge(tokens, pair[0], pair[1], newTokenID)

		t.VocabSize++
	}

	return nil
}

// Encode converts text into token IDs using the learned merges
func (t *Tokenizer) Encode(text []byte) []int {
	// Start with byte-level tokens
	tokens := make([]int, len(text))
	for i, b := range text {
		tokens[i] = int(b)
	}

	// Apply each merge in order
	for _, merge := range t.Merges {
		tokens = t.applyMerge(tokens, merge.First, merge.Second, merge.Result)
	}

	return tokens
}

// Decode converts token IDs back into text
func (t *Tokenizer) Decode(tokens []int) []byte {
	result := []byte{}
	for _, tokenID := range tokens {
		if bytes, ok := t.Vocabulary[tokenID]; ok {
			result = append(result, bytes...)
		}
	}
	return result
}

// findMostFrequentPair finds the most frequently occurring adjacent pair of tokens
func (t *Tokenizer) findMostFrequentPair(tokens []int) ([2]int, int) {
	// Count all pairs
	pairCounts := make(map[[2]int]int)

	for i := 0; i < len(tokens)-1; i++ {
		pair := [2]int{tokens[i], tokens[i+1]}
		pairCounts[pair]++
	}

	// Find the most frequent pair
	var mostFrequentPair [2]int
	maxCount := 0

	for pair, count := range pairCounts {
		if count > maxCount {
			maxCount = count
			mostFrequentPair = pair
		}
	}

	return mostFrequentPair, maxCount
}

// applyMerge replaces all occurrences of (first, second) with merged token
func (t *Tokenizer) applyMerge(tokens []int, first, second, merged int) []int {
	result := []int{}

	i := 0
	for i < len(tokens) {
		// Check if we have a pair to merge
		if i < len(tokens)-1 && tokens[i] == first && tokens[i+1] == second {
			result = append(result, merged)
			i += 2 // Skip both tokens
		} else {
			result = append(result, tokens[i])
			i++
		}
	}

	return result
}
