# CLAUDE.md - Developer Context

This document provides context for AI assistants (like Claude) working on this codebase.

## Project Overview

This is a **Byte Pair Encoding (BPE) tokenizer** implemented in Go. BPE is a data compression technique adapted for subword tokenization in NLP. It's used by many modern language models (GPT, BERT variants, etc.) to handle vocabulary efficiently.

**Current status**: Working implementation with optimized training algorithm

## Architecture

### File Structure

```
bpe-tokenizer/
├── bpe/
│   ├── tokenizer.go           # Core implementation
│   ├── tokenizer_test.go      # Unit tests
│   └── tokenizer_bench_test.go # Performance benchmarks
├── go.mod
└── README.md
```

### Core Components (tokenizer.go)

1. **Tokenizer struct** (`tokenizer.go:8-18`)
   - `Vocabulary`: Maps token IDs to byte sequences
   - `Merges`: Ordered list of learned merge rules
   - `VocabSize`: Current vocabulary size

2. **Merge struct** (`tokenizer.go:21-25`)
   - Represents a single merge rule: (first, second) → result

3. **Key Methods**:
   - `New()`: Initializes with 256 base byte tokens
   - `Train()`: Learns merge rules from corpus
   - `Encode()`: Applies merge rules to convert text to tokens
   - `Decode()`: Converts tokens back to text

## Key Implementation Details

### Critical Optimization: Incremental Pair Counting

The main algorithmic optimization is in `applyMergeIncremental()` (`tokenizer.go:149-190`):

**Problem**: Naive BPE implementations recount all pairs after each merge, making training O(n² × m) where n is corpus size and m is number of merges.

**Solution**: When applying a merge, only update counts for pairs affected by that specific merge:

1. **Left neighbor pair**: Decrement `(leftNeighbor, first)`, increment `(leftNeighbor, merged)`
2. **Merged pair**: Decrement `(first, second)`
3. **Right neighbor pair**: Decrement `(second, rightNeighbor)`, increment `(merged, rightNeighbor)`

This reduces per-merge complexity from O(n) to O(k) where k is the number of times the pair appears.

### Training Flow

1. **Initialize tokens** (`tokenizer.go:51-54`): Convert each byte to its token ID
2. **Build initial pair counts** (`tokenizer.go:57`): Count all adjacent pairs once
3. **Merge loop** (`tokenizer.go:60-89`):
   - Find most frequent pair from maintained counts
   - Create new vocabulary entry (concatenate byte sequences)
   - Record merge rule
   - Apply merge and update counts incrementally
   - Increment vocabulary size

### Encoding vs Training

Two different `applyMerge` implementations:

- **`applyMergeIncremental()`** (`tokenizer.go:149`): Used during training, updates pair counts
- **`applyMerge()`** (`tokenizer.go:202`): Used during encoding, no count updates needed

Both do the same token replacement but with different side effects.

## Testing Strategy

### Unit Tests (tokenizer_test.go)

- **Initialization tests**: Verify base vocabulary setup
- **Encode/decode roundtrip**: Ensure lossless transformation
- **Training validation**: Check merge learning, vocabulary growth
- **Edge cases**: Empty text, single bytes, invalid tokens, repeated patterns
- **Error handling**: Invalid vocabulary sizes

### Benchmarks (tokenizer_bench_test.go)

Test scenarios across different scales:
- Corpus sizes: 1KB, 10KB, 100KB
- Vocabulary targets: 300, 500, 1000
- Operations: Train, Encode, Decode

## Design Decisions

### Why Byte-Level?

Starting with 256 base tokens (one per byte value) ensures:
- Universal encoding: Can handle any text/binary data
- No unknown tokens: Everything can be represented
- Consistent with modern tokenizers (GPT-2, GPT-3, etc.)

### Why Incremental Counting?

Previous commit history shows this was added as an optimization:
- Commit `e27d203`: "Implement incremental pair counting optimization"

This dramatically improves training performance for large corpora.

### Merge Order Matters

Merges must be applied in the order they were learned. This ensures:
- Deterministic encoding
- Correctness of the learned vocabulary
- The most frequent patterns are merged first (greedy approach)

## Common Development Tasks

### Adding New Features

Consider these areas for enhancement:
1. **Serialization**: Save/load trained tokenizers to disk
2. **Vocabulary pruning**: Remove infrequent tokens
3. **Parallel training**: Multi-threaded pair counting
4. **Special tokens**: Add support for `[PAD]`, `[UNK]`, etc.
5. **Pretokenization**: Add whitespace/punctuation splitting before BPE

### Performance Optimization

Current bottlenecks:
- `findMaxPair()` (`tokenizer.go:135`): O(n) scan of pair counts map
  - Could use a heap/priority queue for O(log n) extraction
- Memory allocations in `applyMerge*()`: Creates new slice each time
  - Could use in-place updates or buffer pools

### Testing New Changes

Always run:
```bash
go test -v ./bpe          # Unit tests
go test -bench=. ./bpe    # Benchmarks
```

Compare benchmark results before/after changes to verify performance impact.

## Debugging Tips

### Common Issues

1. **Merges not reducing token count**: Check if target vocab size allows enough merges
2. **Decode doesn't match original**: Likely invalid token IDs or corrupted vocabulary
3. **Training is slow**: Profile to ensure incremental counting is working correctly

### Useful Debug Points

- After `countPairs()`: Check initial pair frequencies
- Inside merge loop: Log most frequent pair and its count
- After `applyMergeIncremental()`: Verify pair counts are updated correctly

## Code Style

- Standard Go formatting (gofmt)
- Comments explain "why" not "what"
- Descriptive variable names (pairCounts, mostFrequentPair, etc.)
- Test functions named `Test<Functionality>`
- Benchmark functions named `Benchmark<Operation>_<Size>_<Config>`

## Related Concepts

- **Subword tokenization**: BPE, WordPiece, Unigram
- **Compression**: BPE originated as a compression algorithm
- **Language models**: Modern LLMs use BPE or variants (GPT series, LLaMA, etc.)
- **Vocabulary size tradeoffs**: Larger = better compression but more model parameters

## Future Directions

Potential enhancements:
- [ ] Add regex-based pretokenization (GPT-2 style)
- [ ] Implement vocabulary merging for multi-corpus training
- [ ] Add support for special tokens and token types
- [ ] Create CLI tool for training and encoding files
- [ ] Add JSON serialization for trained models
- [ ] Benchmark against other BPE implementations
- [ ] Add fuzzing tests for robustness

## Questions to Ask When Modifying

1. Does this change affect merge order or determinism?
2. Will this impact training performance (complexity)?
3. Are edge cases (empty text, single byte) still handled?
4. Does encode/decode remain lossless?
5. Are benchmarks still representative?

## Related Resources

- [BPE Paper (Sennrich et al.)](https://arxiv.org/abs/1508.07909)
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) - Production-grade implementation
- [SentencePiece](https://github.com/google/sentencepiece) - Alternative approach with BPE variant
