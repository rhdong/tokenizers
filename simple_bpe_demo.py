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


def compare_with_huggingface():
    """Compare simplified implementation with HuggingFace standard implementation"""
    
    # Create simple training data
    corpus = [
        "hello world",
        "hello there", 
        "how are you",
        "the quick brown fox",
        "the lazy dog",
        "hello hello hello",
        "the the the"
    ]
    
    # Train a small BPE model using HuggingFace tokenizers
    print("=== Training BPE Model ===")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=100,  # Increase vocab size to cover more characters
        min_frequency=1,
        show_progress=False,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
    )
    
    tokenizer.train_from_iterator(corpus, trainer)
    
    # Extract vocabulary and merge rules
    # Save to temp file to get model data
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        tokenizer.save(temp_file)
    
    with open(temp_file, 'r') as f:
        tokenizer_data = json.load(f)
    
    os.unlink(temp_file)
    
    model_dict = tokenizer_data['model']
    vocab = model_dict['vocab']
    merges = [tuple(pair.split() if isinstance(pair, str) else pair) for pair in model_dict['merges']]
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Number of merge rules: {len(merges)}")
    print(f"\nFirst 10 merge rules:")
    for i, (a, b) in enumerate(merges[:10]):
        print(f"  {i}: '{a}' + '{b}' -> '{a}{b}'")
    
    # Create simplified BPE
    simple_bpe = SimpleBPE(vocab, merges)
    
    # Test texts - only test word-level since we use Whitespace pre-tokenizer
    test_texts = [
        # Single word tests
        ("hello", "single word test"),
        ("the", "common word"),
        ("world", "another common word"),
        ("quick", "test merge"),
        # Unseen words
        ("unknown", "unseen word"),
        ("test", "test word"),
        # Case tests
        ("HELLO", "uppercase word"),
        ("World", "capitalized word"),
        # Number tests
        ("123", "pure numbers"),
        ("hello123", "alphanumeric"),
    ]
    
    print("\n\n=== Comparison Test ===")
    print("Note: Since we use Whitespace pre-tokenizer, we only test single word tokenization\n")
    
    # Statistics
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for text, description in test_texts:
        total_tests += 1
        print(f"\nTest {total_tests}: '{text}' ({description})")
        print("-" * 50)
        
        # HuggingFace results - encode single word only
        hf_encoding = tokenizer.encode(text)
        hf_tokens = hf_encoding.tokens
        hf_ids = hf_encoding.ids
        print(f"HuggingFace results:")
        print(f"  Tokens: {hf_tokens}")
        print(f"  IDs:    {hf_ids}")
        
        # Simplified version results - directly call tokenize_word for single word
        print(f"\nSimplified version:")
        # Don't print detailed process to avoid too much output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            simple_result = simple_bpe.tokenize_word(text)
            simple_ids = [id for id, _ in simple_result]
            simple_tokens = [token for _, token in simple_result]
        finally:
            sys.stdout = old_stdout
            
        print(f"  Tokens: {simple_tokens}")
        print(f"  IDs:    {simple_ids}")
        
        # Compare results
        if hf_ids == simple_ids:
            print("\nStatus: ✅ PASS")
            passed_tests += 1
        else:
            print("\nStatus: ❌ FAIL")
            print(f"Difference details:")
            print(f"  - HuggingFace: {list(zip(hf_tokens, hf_ids))}")
            print(f"  - Simplified:  {list(zip(simple_tokens, simple_ids))}")
            failed_tests.append((text, description))
    
    # Summary report
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"✅ Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"❌ Failed: {len(failed_tests)} ({len(failed_tests)/total_tests*100:.1f}%)")
    
    if failed_tests:
        print("\nFailed tests:")
        for text, desc in failed_tests:
            print(f"  - '{text}' ({desc})")

def llama_tokenizer_test():
    """Test with real LLaMA tokenizer vocabulary and merge rules"""
    print("\n\n=== Real LLaMA Tokenizer Test ===")
    print("Testing with a real-world tokenizer configuration\n")
    
    try:
        # Try to load a pre-trained tokenizer
        # We'll use GPT-2 as it's a well-known BPE tokenizer that's easily accessible
        # (LLaMA uses SentencePiece which is slightly different)
        from tokenizers import Tokenizer
        import tempfile
        
        print("Loading GPT-2 tokenizer (BPE-based, similar to LLaMA)...")
        # Load GPT-2 tokenizer directly
        tokenizer = Tokenizer.from_pretrained("gpt2")
        
        # Extract vocabulary and merges by temporarily saving
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            tokenizer.save(temp_file)
        
        with open(temp_file, 'r') as f:
            tokenizer_data = json.load(f)
        
        # Clean up temp file
        os.unlink(temp_file)
        
        model_data = tokenizer_data['model']
        vocab = model_data['vocab']
        
        # Handle different merge formats
        raw_merges = model_data['merges']
        merges = []
        for merge in raw_merges:
            if isinstance(merge, str):
                # Format: "a b"
                parts = merge.split()
                if len(parts) == 2:
                    merges.append(tuple(parts))
            elif isinstance(merge, list) and len(merge) == 2:
                # Format: ["a", "b"]
                merges.append(tuple(merge))
        
        print(f"\nTokenizer info:")
        print(f"  Vocabulary size: {len(vocab)}")
        print(f"  Number of merges: {len(merges)}")
        print(f"  Model type: {model_data.get('type', 'Unknown')}")
        print(f"\nFirst 10 vocabulary items:")
        for i, (token, id) in enumerate(list(vocab.items())[:10]):
            print(f"  {repr(token)}: {id}")
        
        # Create our simplified BPE with the real vocabulary
        simple_bpe = SimpleBPE(vocab, merges)
        
        # Test cases - real-world examples
        test_texts = [
            # Common English words
            ("hello", "common greeting"),
            ("world", "common noun"),
            ("artificial", "technical term"),
            ("intelligence", "technical term"),
            # Programming terms
            ("function", "programming term"),
            ("variable", "programming term"),
            ("algorithm", "CS term"),
            # Mixed case and special
            ("Hello", "capitalized"),
            ("WORLD", "uppercase"),
            ("hello123", "alphanumeric"),
            ("hello_world", "snake_case"),
            ("helloWorld", "camelCase"),
            # Common phrases
            ("machine", "ML term"),
            ("learning", "ML term"),
            ("neural", "AI term"),
            ("network", "AI term"),
            # Edge cases
            ("a", "single letter"),
            ("I", "single capital"),
            ("123", "numbers"),
            ("!", "punctuation"),
            # More complex examples
            ("OpenAI", "company name"),
            ("GPT-3", "model name"),
            ("transformer", "architecture"),
            ("tokenization", "NLP term"),
            ("embeddings", "ML concept"),
            # Special characters
            ("hello@world", "email-like"),
            ("$100", "currency"),
            ("3.14", "decimal"),
            ("http://example.com", "URL"),
            # Emojis and Unicode (will likely fail)
            ("😊", "emoji"),
            ("你好", "Chinese"),
        ]
        
        print("\n\nTesting with real tokenizer vocabulary:")
        print("-" * 90)
        print(f"{'Text':<20} {'Description':<25} {'Tokenizer':<25} {'SimpleBPE':<25} {'Match'}")
        print("-" * 90)
        
        passed = 0
        total = 0
        
        for text, description in test_texts:
            total += 1
            
            # Tokenize with the real tokenizer
            encoding = tokenizer.encode(text)
            real_tokens = encoding.tokens
            real_ids = encoding.ids
            
            # Tokenize with our implementation
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                # For GPT-2, we need to handle the tokenization differently
                # as it uses byte-level BPE
                simple_result = simple_bpe.tokenize_word(text)
                simple_ids = [id for id, _ in simple_result]
                simple_tokens = [token for _, token in simple_result]
            finally:
                sys.stdout = old_stdout
            
            # Compare results
            match = real_ids == simple_ids
            if match:
                passed += 1
                status = "✅"
            else:
                status = "❌"
            
            # Format output
            real_str = f"{real_tokens[:2]}..." if len(real_tokens) > 2 else str(real_tokens)
            simple_str = f"{simple_tokens[:2]}..." if len(simple_tokens) > 2 else str(simple_tokens)
            
            print(f"{text:<20} {description:<25} {real_str:<25} {simple_str:<25} {status}")
        
        # Summary
        print("-" * 90)
        print(f"\nResults: {passed}/{total} passed ({passed/total*100:.1f}%)")
        
        if passed < total:
            print("\nNote: Differences are expected because:")
            print("- GPT-2 uses byte-level BPE with special byte token mappings")
            print("- Our implementation has simplified byte-level fallback")
            print("- Real tokenizers have additional pre/post-processing steps")
            print("- Some characters may not have byte representations in the vocabulary")
            
            # Show detailed comparison for failed cases
            print("\n\nDetailed analysis of differences:")
            for text, description in test_texts:
                encoding = tokenizer.encode(text)
                real_tokens = encoding.tokens
                real_ids = encoding.ids
                
                # Suppress output for simple tokenization
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    simple_result = simple_bpe.tokenize_word(text)
                    simple_ids = [id for id, _ in simple_result]
                    simple_tokens = [token for _, token in simple_result]
                finally:
                    sys.stdout = old_stdout
                
                if real_ids != simple_ids:
                    print(f"\n'{text}' ({description}):")
                    print(f"  Tokenizer: {real_tokens} -> {real_ids}")
                    print(f"  SimpleBPE: {simple_tokens} -> {simple_ids}")
                    
                    # Check if it's because of unknown tokens
                    has_unk = any(t == '<unk>' for t in simple_tokens)
                    if has_unk:
                        print(f"  Issue: Contains characters not in vocabulary (byte-level fallback failed)")
            
    except Exception as e:
        print(f"Error setting up real tokenizer test: {e}")
        print("You may need to install: pip install tokenizers")
        import traceback
        traceback.print_exc()


def accurate_comparison_test():
    """Accurate comparison test - using exactly the same vocabulary and merge rules"""
    print("\n\n=== Accurate Comparison Test ===")
    print("Using HuggingFace trained model, ensuring vocabulary is exactly the same\n")
    
    # Training data
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "hello world hello there",
        "testing testing one two three",
        "this is a test sentence",
        "machine learning is amazing"
    ]
    
    # Train HuggingFace BPE
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=200,
        min_frequency=1,
        show_progress=False,
        special_tokens=["<unk>"]
    )
    
    tokenizer.train_from_iterator(corpus, trainer)
    
    # Extract model data
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        tokenizer.save(temp_file)
    
    with open(temp_file, 'r') as f:
        tokenizer_data = json.load(f)
    
    os.unlink(temp_file)
    
    model_dict = tokenizer_data['model']
    vocab = model_dict['vocab']
    merges = [tuple(pair.split() if isinstance(pair, str) else pair) for pair in model_dict['merges']]
    
    # Create simplified BPE (using same vocabulary)
    simple_bpe = SimpleBPE(vocab, merges)
    
    # Test cases
    test_cases = [
        # Words from training set
        "the", "quick", "hello", "world", "test",
        # Phrase combinations from training set
        "brown", "fox", "lazy", "dog",
        # New word tests
        "new", "word", "unseen",
        # Contains numbers
        "test123", "456test",
        # Case variations
        "Hello", "WORLD", "Test"
    ]
    
    results = []
    
    for word in test_cases:
        # HuggingFace encoding
        hf_encoding = tokenizer.encode(word)
        hf_tokens = hf_encoding.tokens
        hf_ids = hf_encoding.ids
        
        # Simplified version encoding (silent mode)
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            simple_result = simple_bpe.tokenize_word(word)
            simple_ids = [id for id, _ in simple_result]
            simple_tokens = [token for _, token in simple_result]
        finally:
            sys.stdout = old_stdout
        
        # Record results
        is_match = hf_ids == simple_ids
        results.append({
            'word': word,
            'hf_tokens': hf_tokens,
            'hf_ids': hf_ids,
            'simple_tokens': simple_tokens,
            'simple_ids': simple_ids,
            'match': is_match
        })
    
    # Display results table
    print(f"{'Word':<15} {'HuggingFace':<30} {'Simplified':<30} {'Match':<10}")
    print("-" * 85)
    
    passed = 0
    for r in results:
        hf_str = str(list(zip(r['hf_tokens'], r['hf_ids'])))
        simple_str = str(list(zip(r['simple_tokens'], r['simple_ids'])))
        status = "✅" if r['match'] else "❌"
        if r['match']:
            passed += 1
        
        print(f"{r['word']:<15} {hf_str:<30} {simple_str:<30} {status:<10}")
    
    # Summary
    print("\n" + "=" * 85)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed} ({passed/len(results)*100:.1f}%)")
    print(f"Failed: {len(results)-passed} ({(len(results)-passed)/len(results)*100:.1f}%)")
    
    # Analyze failures
    failed_results = [r for r in results if not r['match']]
    if failed_results:
        print("\nFailure analysis:")
        for r in failed_results:
            print(f"\n'{r['word']}':")
            print(f"  HuggingFace: {r['hf_tokens']} -> {r['hf_ids']}")
            print(f"  Simplified:  {r['simple_tokens']} -> {r['simple_ids']}")
            
            # Check vocabulary
            for token in r['simple_tokens']:
                if token not in vocab and token != '<unk>':
                    print(f"  ⚠️  Token '{token}' not in vocabulary")


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
            compare_with_huggingface()
            accurate_comparison_test()
            llama_tokenizer_test()
        elif sys.argv[1] == "llama":
            # Run real tokenizer test
            llama_tokenizer_test()
        else:
            print("Usage: python simple_bpe_demo.py [all|llama]")
            print("  all   - Run all tests")
            print("  llama - Test with real tokenizer (GPT-2/LLaMA-like)")
    else:
        # Default: run accurate comparison test
        print("Running accurate comparison test")
        print("Use 'python simple_bpe_demo.py all' to run all tests")
        print("Use 'python simple_bpe_demo.py llama' to test with real tokenizer\n")
        accurate_comparison_test()
