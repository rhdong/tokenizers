#!/usr/bin/env python3
"""
Simplified BPE (Byte Pair Encoding) implementation demo
For understanding BPE merge rules and working principles
Now includes real LLaMA tokenizer comparison
"""

from typing import List, Tuple, Dict, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json
import os


class SimpleBPE:
    """Simplified BPE implementation with only core merge logic"""
    
    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]]):
        """
        Initialize BPE model
        
        Args:
            vocab: Vocabulary mapping tokens to IDs
            merges: List of merge rules, sorted by priority
        """
        self.vocab = vocab
        self.vocab_r = {v: k for k, v in vocab.items()}  # Reverse vocabulary
        # Convert merges to (pair -> (rank, new_token)) mapping
        self.merge_map = {}
        for rank, (a, b) in enumerate(merges):
            merged = a + b
            if merged in vocab:
                self.merge_map[(a, b)] = (rank, merged, vocab[merged])
    
    def _byte_encode_char(self, char: str) -> List[str]:
        """
        Encode a character as byte tokens (similar to GPT-2's byte-level BPE)
        Returns a list of tokens representing the bytes of the character
        """
        try:
            # Encode character to UTF-8 bytes
            char_bytes = char.encode('utf-8')
            byte_tokens = []
            
            # Try to find byte representations in vocabulary
            for byte in char_bytes:
                # GPT-2 uses special byte tokens like 'Ġ' for space (0x20 + 0x100)
                # and other mappings for bytes 0-255
                byte_token = None
                
                # Check various possible representations
                # Direct byte value
                if chr(byte) in self.vocab:
                    byte_token = chr(byte)
                # Byte as string number
                elif str(byte) in self.vocab:
                    byte_token = str(byte)
                # Special GPT-2 style byte token (byte + 256 offset for some chars)
                elif byte >= 32 and chr(byte + 256) in self.vocab:
                    byte_token = chr(byte + 256)
                # Hex representation
                elif f"<0x{byte:02X}>" in self.vocab:
                    byte_token = f"<0x{byte:02X}>"
                
                if byte_token:
                    byte_tokens.append(byte_token)
                else:
                    # If we can't find the byte in vocab, return None
                    return None
                    
            return byte_tokens if byte_tokens else None
        except:
            return None
    
    def tokenize_word(self, word: str) -> List[Tuple[int, str]]:
        """
        Tokenize a single word using BPE (simplified version)
        
        Args:
            word: Word to tokenize
            
        Returns:
            List of (token_id, token_str) tuples
        """
        # Handle empty string
        if not word:
            return []
            
        # Step 1: Split word into character-level tokens
        tokens = []
        for char in word:
            if char in self.vocab:
                tokens.append(char)
            else:
                # For unknown characters, try byte-level fallback
                # This mimics GPT-2's byte-level BPE behavior
                byte_tokens = self._byte_encode_char(char)
                if byte_tokens:
                    tokens.extend(byte_tokens)
                else:
                    # Final fallback: use <unk>
                    tokens.append('<unk>')
        
        # If only one token, return directly
        if len(tokens) == 1:
            token = tokens[0]
            if token in self.vocab:
                return [(self.vocab[token], token)]
            else:
                return [(self.vocab.get('<unk>', 0), '<unk>')]
        
        # Step 2: Iteratively merge
        while len(tokens) > 1:
            # Find all adjacent pairs
            pairs = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_map:
                    rank, _, _ = self.merge_map[pair]
                    pairs.append((i, pair, rank))
            
            if not pairs:
                break  # No mergeable pairs
            
            # Find highest priority (lowest rank) pair
            pairs.sort(key=lambda x: x[2])
            pos, (a, b), _ = pairs[0]
            
            # Execute merge
            _, merged_token, _ = self.merge_map[(a, b)]
            new_tokens = tokens[:pos] + [merged_token] + tokens[pos + 2:]
            tokens = new_tokens
            
            # Only print when merging, not every time
            if len(tokens) <= 10:  # Avoid too long output
                print(f"  Merge: '{a}' + '{b}' -> '{merged_token}'")
                print(f"  Current: {tokens}")
        
        # Convert to (id, token) format
        result = []
        for token in tokens:
            if token in self.vocab:
                result.append((self.vocab[token], token))
            else:
                result.append((self.vocab.get('<unk>', 0), '<unk>'))
        
        return result
    
    def tokenize(self, text: str) -> List[Tuple[int, str]]:
        """Tokenize text (split by spaces and process word by word)"""
        # Simple space splitting (real BPE would use more complex pre-tokenizers)
        # This is a simplified handling of spaces
        if not text:
            return []
            
        words = text.split()
        all_tokens = []
        
        for i, word in enumerate(words):
            if not word:  # Skip empty words
                continue
                
            print(f"\nProcessing word: '{word}'")
            tokens = self.tokenize_word(word)
            all_tokens.extend(tokens)
            
            # Add space if not last word (if space is in vocab)
            if i < len(words) - 1 and ' ' in self.vocab:
                all_tokens.append((self.vocab[' '], ' '))
        
        return all_tokens
    
    def apply_bpe_to_tokens(self, tokens: List[str]) -> List[Tuple[int, str]]:
        """
        Apply BPE merges to a list of already pre-tokenized tokens.
        This allows us to use HuggingFace's pre-tokenizer for consistent input.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            List of (token_id, token_str) tuples after BPE merges
        """
        # If empty, return empty
        if not tokens:
            return []
        
        # Process each pre-tokenized element
        all_results = []
        for token in tokens:
            # If token is directly in vocab, use it
            if token in self.vocab:
                all_results.append((self.vocab[token], token))
                continue
                
            # Otherwise, apply BPE merges to this token
            # First split into characters that are in vocab
            char_tokens = []
            for char in token:
                if char in self.vocab:
                    char_tokens.append(char)
                else:
                    # Skip characters not in vocab for this comparison
                    pass
            
            if not char_tokens:
                continue
                
            # Apply BPE merges
            merged_tokens = char_tokens[:]
            while len(merged_tokens) > 1:
                # Find best merge
                best_pair = None
                best_rank = float('inf')
                best_idx = -1
                
                for i in range(len(merged_tokens) - 1):
                    pair = (merged_tokens[i], merged_tokens[i + 1])
                    if pair in self.merge_map:
                        rank, _, _ = self.merge_map[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            best_idx = i
                
                # No more merges possible
                if best_pair is None:
                    break
                
                # Apply merge
                _, merged_token, _ = self.merge_map[best_pair]
                merged_tokens = merged_tokens[:best_idx] + [merged_token] + merged_tokens[best_idx + 2:]
            
            # Convert to (id, token) pairs
            for t in merged_tokens:
                if t in self.vocab:
                    all_results.append((self.vocab[t], t))
        
        return all_results


def aligned_bpe_test():
    """Test BPE merge algorithm using HuggingFace pre-tokenizer for consistent input"""
    print("\n\n=== Aligned BPE Merge Test ===")
    print("Using HuggingFace pre-tokenizer for consistent input, comparing only BPE merges\n")
    
    try:
        from tokenizers import Tokenizer, pre_tokenizers
        import tempfile
        
        print("Loading GPT-2 tokenizer...")
        tokenizer = Tokenizer.from_pretrained("gpt2")
        
        # Extract vocabulary and merges
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            tokenizer.save(temp_file)
        
        with open(temp_file, 'r') as f:
            tokenizer_data = json.load(f)
        
        os.unlink(temp_file)
        
        model_data = tokenizer_data['model']
        vocab = model_data['vocab']
        raw_merges = model_data['merges']
        
        merges = []
        for merge in raw_merges:
            if isinstance(merge, str):
                parts = merge.split()
                if len(parts) == 2:
                    merges.append(tuple(parts))
            elif isinstance(merge, list) and len(merge) == 2:
                merges.append(tuple(merge))
        
        print(f"\nTokenizer info:")
        print(f"  Vocabulary size: {len(vocab)}")
        print(f"  Number of merges: {len(merges)}")
        
        # Create SimpleBPE with the same vocabulary and merges
        simple_bpe = SimpleBPE(vocab, merges)
        
        # Test cases
        test_texts = [
            ("hello world", "basic greeting"),
            ("artificial intelligence", "AI term"),
            ("machine learning", "ML term"),
            ("Hello World", "capitalized"),
            ("OpenAI GPT-3", "model name"),
            ("transformer", "architecture"),
            ("tokenization", "NLP term"),
            ("hello@example.com", "email"),
            ("https://example.com", "URL"),
            ("The quick brown fox", "pangram"),
            ("function()", "code"),
            ("hello_world", "snake_case"),
            ("helloWorld", "camelCase"),
            ("123.456", "decimal"),
            ("$100", "currency"),
            # Long sentences from tokenizers tests
            ("The quick brown fox jumps over the lazy dog", "full pangram"),
            ("Hello! How are you? I'm fine, thank you.", "conversation"),
            ("Hey there dear friend!", "greeting"),
            ("A sentence 🤗", "with emoji"),
            ("My name is John", "introduction"),
            ("Call 911!", "emergency"),
            ("Héllò hôw are ü?", "with accents"),
            # Technical sentences
            ("Artificial intelligence and machine learning are transforming the technology landscape", "tech sentence"),
            ("The development of large language models has revolutionized natural language processing and understanding", "LLM sentence"),
            ("Python is a high-level, interpreted programming language with dynamic semantics and clear syntax", "programming desc"),
            ("This function implements the byte pair encoding algorithm which iteratively merges the most frequent pairs of tokens in the vocabulary until reaching the desired vocabulary size or no more merges are possible", "code comment"),
            # Mixed content from tests
            ("The model achieved 95.2% accuracy on the test set with a learning rate of 0.001 and batch size of 32 after training for 50 epochs", "model metrics"),
            ("To install the package, first ensure you have Python 3.7 or higher installed on your system, then run pip install tokenizers in your terminal or command prompt, and finally verify the installation by importing the library in a Python script", "install guide"),
            # Complex URLs and technical strings
            ("https://api.openai.com/v1/chat/completions?model=gpt-4&temperature=0.7&max_tokens=150", "API URL"),
            ("data/wikitext-103-raw/wiki.train.raw", "file path"),
            ("[CLS] $A [SEP] $B:1 [SEP]:1", "template"),
            # Very long sentence
            ("The history of artificial intelligence dates back to ancient times when philosophers contemplated the nature of human reasoning and attempted to describe human thinking as a mechanical manipulation of symbols, but the field as we know it today was formally founded in 1956 at the Dartmouth Conference where researchers gathered to discuss the possibility of creating machines that could simulate human intelligence", "AI history"),
            # Multiple languages and special characters
            ("UnigramTrainer can only train a Unigram", "error message"),
            ("normalizer.normalize_str('Héllò hôw are ü?')", "code example"),
            # Edge cases from tests
            ("", "empty string"),
            ("a", "single char"),
            ("123", "numbers only"),
            ("@#$%", "special chars"),
            # Real tokenizer test cases
            ("BPE is a subword tokenization algorithm that starts with a character-level vocabulary and iteratively merges the most frequent pairs of adjacent tokens", "BPE explanation"),
            ("The tokenizer.encode() method converts text to token IDs while tokenizer.decode() converts IDs back to text", "tokenizer methods"),
        ]
        
        print("\n\nComparing BPE merge results (using HF pre-tokenizer):")
        print("-" * 100)
        print(f"{'Text':<25} {'Description':<20} {'HF Tokens':<30} {'SimpleBPE':<30} {'Match'}")
        print("-" * 100)
        
        passed = 0
        total = 0
        
        for text, description in test_texts:
            total += 1
            
            # Get pre-tokenized output from HuggingFace
            encoding = tokenizer.encode(text)
            
            # Also get the pre-tokenized version before BPE
            # We'll simulate this by using the tokenizer's pre-tokenizer
            pre_tok_output = tokenizer.pre_tokenizer.pre_tokenize_str(text)
            pre_tokens = [token for token, _ in pre_tok_output]
            
            # Apply our BPE to the pre-tokenized output
            simple_result = simple_bpe.apply_bpe_to_tokens(pre_tokens)
            simple_tokens = [token for _, token in simple_result]
            simple_ids = [id for id, _ in simple_result]
            
            # Compare with HuggingFace result
            hf_tokens = encoding.tokens
            hf_ids = encoding.ids
            
            # Check if they match
            match = (simple_tokens == hf_tokens)
            if match:
                passed += 1
                status = "✅"
            else:
                status = "❌"
            
            # Format output
            hf_str = str(hf_tokens[:3]) + "..." if len(hf_tokens) > 3 else str(hf_tokens)
            simple_str = str(simple_tokens[:3]) + "..." if len(simple_tokens) > 3 else str(simple_tokens)
            
            print(f"{text:<25} {description:<20} {hf_str:<30} {simple_str:<30} {status}")
            
            if not match:
                print(f"  Pre-tokens: {pre_tokens}")
                print(f"  HF result: {hf_tokens}")
                print(f"  Simple result: {simple_tokens}")
        
        print("-" * 100)
        print(f"\nResults: {passed}/{total} passed ({passed/total*100:.1f}%)")
        
        if passed < total:
            print("\nNote: This test isolates the BPE merge algorithm by using")
            print("HuggingFace's pre-tokenizer for both implementations.")
            print("Any differences indicate issues in the merge logic itself.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


# llama_tokenizer_test function removed - only keeping aligned test


def edge_case_tests():
    """Test edge cases and error handling"""
    print("\n\n=== Edge Case Tests ===")
    
    # Minimal vocabulary
    minimal_vocab = {
        'a': 0,
        'b': 1,
        'ab': 2,
        '<unk>': 3
    }
    
    minimal_merges = [
        ('a', 'b'),  # Merge to 'ab'
    ]
    
    simple_bpe = SimpleBPE(minimal_vocab, minimal_merges)
    
    edge_tests = [
        ("", "empty string"),
        ("a", "single known character"),
        ("b", "another known character"),
        ("ab", "mergeable character pair"),
        ("aba", "partially mergeable"),
        ("abab", "multiple merges"),
        ("c", "unknown character"),
        ("abc", "contains unknown character"),
        ("aaa", "repeated characters"),
        ("aaabbb", "multiple repeated characters"),
        ("abcdef", "multiple unknown characters"),
        ("a b", "contains space (unknown)"),
        ("a🌟b", "contains emoji"),
        ("a\nb", "contains newline"),
        ("a\tb", "contains tab"),
        ("aaaaaaaaaa", "very long repetition"),
    ]
    
    for text, description in edge_tests:
        print(f"\nTest: '{repr(text)}' ({description})")
        try:
            tokens = simple_bpe.tokenize_word(text)
            print(f"Result: {tokens}")
            # Validate results
            for token_id, token_str in tokens:
                if token_id not in simple_bpe.vocab_r:
                    print(f"  ⚠️  Warning: token_id {token_id} not in reverse vocabulary!")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def manual_example():
    """Manually constructed simple example to understand merge process"""
    print("\n\n=== Manual Example ===")
    
    # Build a simple vocabulary and merge rules
    vocab = {
        'h': 0, 'e': 1, 'l': 2, 'o': 3,
        'll': 4,      # Result of merging 'l' + 'l'
        'he': 5,      # Result of merging 'h' + 'e'  
        'llo': 6,     # Result of merging 'll' + 'o'
        'hello': 7,   # Result of merging 'he' + 'llo'
        '<unk>': 8
    }
    
    # Merge rules (sorted by priority)
    merges = [
        ('l', 'l'),      # First priority: merge 'll'
        ('h', 'e'),      # Second priority: merge 'he'
        ('ll', 'o'),     # Third priority: merge 'llo'
        ('he', 'llo'),   # Fourth priority: merge 'hello'
    ]
    
    simple_bpe = SimpleBPE(vocab, merges)
    
    # Demonstrate tokenization process for "hello"
    print("\nDetailed tokenization process for 'hello':")
    result = simple_bpe.tokenize_word('hello')
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            # Run all tests
            manual_example()
            edge_case_tests()
            aligned_bpe_test()
        elif sys.argv[1] == "aligned":
            # Run aligned BPE test
            aligned_bpe_test()
        elif sys.argv[1] == "basic":
            # Run basic tests
            manual_example()
            edge_case_tests()
        else:
            print("Usage: python simple_bpe_demo.py [all|aligned|basic]")
            print("  (default) - Run aligned BPE test with HF pre-tokenizer")
            print("  all       - Run all tests")
            print("  aligned   - Test BPE merges with HF pre-tokenizer (same as default)")
            print("  basic     - Run basic manual examples and edge case tests")
    else:
        # Default: run aligned BPE test
        aligned_bpe_test()
