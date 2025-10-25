# BPE Tokenizer

A high-performance Byte Pair Encoding (BPE) tokenizer implementation in Go with optimized incremental pair counting.

## Overview

This library implements the Byte Pair Encoding algorithm, a subword tokenization technique widely used in natural language processing and machine learning models. BPE iteratively merges the most frequent pairs of bytes (or tokens) in the training corpus to build a vocabulary of subword units.

### Key Features

- **Byte-level encoding**: Starts with a base vocabulary of all 256 possible byte values
- **Incremental pair counting**: Optimized training algorithm that updates pair frequencies incrementally rather than recounting after each merge
- **Simple API**: Clean and intuitive interface for training, encoding, and decoding
- **Comprehensive tests**: Full test coverage with unit tests and benchmarks

## Installation

```bash
go get github.com/zhubert/bpe-tokenizer
```

## Usage

### Basic Example

```go
package main

import (
    "fmt"
    "github.com/zhubert/bpe-tokenizer/bpe"
)

func main() {
    // Create a new tokenizer
    tokenizer := bpe.New()

    // Training text
    text := []byte("low lower lowest")

    // Train the tokenizer with a target vocabulary size of 300
    err := tokenizer.Train(text, 300)
    if err != nil {
        panic(err)
    }

    // Encode some text
    tokens := tokenizer.Encode([]byte("low"))
    fmt.Printf("Tokens: %v\n", tokens)

    // Decode back to original text
    decoded := tokenizer.Decode(tokens)
    fmt.Printf("Decoded: %s\n", decoded)
}
```

### Training

The `Train` method learns merge rules from your training data:

```go
tokenizer := bpe.New()

// Train with target vocabulary size of 500
// (must be > 256 since base vocabulary is all bytes)
err := tokenizer.Train(trainingData, 500)
if err != nil {
    // Handle error
}
```

The training process:
1. Initializes each byte as a separate token
2. Finds the most frequent adjacent token pair
3. Merges that pair into a new token
4. Updates pair frequencies incrementally
5. Repeats until reaching the target vocabulary size

### Encoding

Convert text into token IDs:

```go
text := []byte("Hello, World!")
tokens := tokenizer.Encode(text)
// tokens is []int containing token IDs
```

### Decoding

Convert token IDs back into text:

```go
tokens := []int{72, 101, 108, 108, 111}
text := tokenizer.Decode(tokens)
// text is []byte containing the original bytes
```

## API Reference

### Types

#### `Tokenizer`

Main tokenizer struct with the following fields:

- `Vocabulary map[int][]byte` - Maps token IDs to their byte representations
- `Merges []Merge` - Ordered list of merge rules learned during training
- `VocabSize int` - Current vocabulary size

#### `Merge`

Represents a single merge rule:

- `First int` - First token ID in the pair
- `Second int` - Second token ID in the pair
- `Result int` - Resulting merged token ID

### Methods

#### `New() *Tokenizer`

Creates a new BPE tokenizer initialized with byte-level vocabulary (tokens 0-255).

#### `Train(text []byte, targetVocabSize int) error`

Learns BPE merge rules from training text.

- `text`: Training corpus as bytes
- `targetVocabSize`: Desired final vocabulary size (must be > 256)
- Returns error if target size is invalid

#### `Encode(text []byte) []int`

Converts text into token IDs using learned merge rules.

- `text`: Input text as bytes
- Returns slice of token IDs

#### `Decode(tokens []int) []byte`

Converts token IDs back into text.

- `tokens`: Slice of token IDs
- Returns original text as bytes (invalid token IDs are skipped)

## Performance

This implementation uses an optimized incremental pair counting algorithm that dramatically improves training performance compared to naive implementations.

### Key Optimization

Instead of recounting all pairs after each merge (O(n) per merge), the algorithm updates only the affected pair counts incrementally (O(k) where k is the number of merge locations). This reduces the overall training complexity significantly for large corpora.

### Running Benchmarks

```bash
cd bpe
go test -bench=. -benchmem
```

Benchmark results show efficient performance across various corpus sizes and vocabulary targets:

```
BenchmarkTrain_1KB_Vocab300
BenchmarkTrain_10KB_Vocab300
BenchmarkTrain_100KB_Vocab1000
BenchmarkEncode_1KB
BenchmarkDecode_1KB
```

## Testing

Run the test suite:

```bash
cd bpe
go test -v
```

The test suite includes:
- Basic functionality tests (initialization, encoding, decoding)
- Edge cases (empty text, single bytes, invalid tokens)
- Training validation (merge order, vocabulary growth)
- Pattern recognition tests (repeated patterns)

## Algorithm Details

### BPE Training Algorithm

1. **Initialize**: Start with base vocabulary of 256 byte values
2. **Count pairs**: Build initial frequency map of all adjacent token pairs
3. **Repeat until target vocabulary size**:
   - Find most frequent pair
   - Create new token for this pair
   - Apply merge to training data and update pair counts incrementally
   - Record merge rule

### Encoding Process

Apply learned merge rules in the order they were learned:
1. Start with byte-level tokens
2. For each merge rule, replace all occurrences of (first, second) with merged token
3. Return final token sequence

### Decoding Process

Concatenate the byte sequences corresponding to each token ID.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[Add your license here]

## References

- [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) - Wikipedia
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - Original BPE paper by Sennrich et al.
