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

	// Build initial pair counts (only done once!)
	pairCounts := t.countPairs(tokens)

	// Learn merges until we reach target vocabulary size
	for t.VocabSize < targetVocabSize {
		// Find the most frequent pair from our maintained counts
		pair, count := t.findMaxPair(pairCounts)
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

		// Apply the merge to tokens AND update pair counts incrementally
		tokens = t.applyMergeIncremental(tokens, pair[0], pair[1], newTokenID, pairCounts)

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

// countPairs builds initial pair counts from tokens
// This is only called once at the start of training
func (t *Tokenizer) countPairs(tokens []int) map[[2]int]int {
	pairCounts := make(map[[2]int]int)

	for i := 0; i < len(tokens)-1; i++ {
		pair := [2]int{tokens[i], tokens[i+1]}
		pairCounts[pair]++
	}

	return pairCounts
}

// findMaxPair finds the most frequent pair from the counts map
func (t *Tokenizer) findMaxPair(pairCounts map[[2]int]int) ([2]int, int) {
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

// applyMergeIncremental replaces all occurrences of (first, second) with merged token
// and updates the pairCounts map incrementally (the key optimization!)
func (t *Tokenizer) applyMergeIncremental(tokens []int, first, second, merged int, pairCounts map[[2]int]int) []int {
	result := []int{}

	i := 0
	for i < len(tokens) {
		// Check if we have a pair to merge
		if i < len(tokens)-1 && tokens[i] == first && tokens[i+1] == second {
			// Found a merge location - update counts for affected pairs

			// 1. Update left neighbor pair (if exists)
			if len(result) > 0 {
				leftNeighbor := result[len(result)-1]
				// Decrement old pair (leftNeighbor, first)
				t.decrementPair(pairCounts, [2]int{leftNeighbor, first})
				// Increment new pair (leftNeighbor, merged)
				pairCounts[[2]int{leftNeighbor, merged}]++
			}

			// 2. Decrement the pair we're merging
			t.decrementPair(pairCounts, [2]int{first, second})

			// 3. Update right neighbor pair (if exists)
			if i+2 < len(tokens) {
				rightNeighbor := tokens[i+2]
				// Decrement old pair (second, rightNeighbor)
				t.decrementPair(pairCounts, [2]int{second, rightNeighbor})
				// Increment new pair (merged, rightNeighbor)
				pairCounts[[2]int{merged, rightNeighbor}]++
			}

			result = append(result, merged)
			i += 2 // Skip both tokens
		} else {
			result = append(result, tokens[i])
			i++
		}
	}

	return result
}

// decrementPair decrements a pair count and removes it if it reaches zero
func (t *Tokenizer) decrementPair(pairCounts map[[2]int]int, pair [2]int) {
	pairCounts[pair]--
	if pairCounts[pair] <= 0 {
		delete(pairCounts, pair)
	}
}

// applyMerge replaces all occurrences of (first, second) with merged token
// Used by Encode() which doesn't need incremental counting
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
