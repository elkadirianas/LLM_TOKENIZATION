import time
import sys
import os

from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

def measure_tokenizer_performance(tokenizer_cls, text, n_runs=10):
    tokenizer = tokenizer_cls()
    # Warm-up
    tokenizer.encode(text)
    tokenizer.decode(tokenizer.encode(text))
    # Measure encoding time
    start = time.perf_counter()
    for _ in range(n_runs):
        ids = tokenizer.encode(text)
    encode_time = (time.perf_counter() - start) / n_runs * 1000  # ms
    # Measure decoding time
    start = time.perf_counter()
    for _ in range(n_runs):
        decoded = tokenizer.decode(ids)
    decode_time = (time.perf_counter() - start) / n_runs * 1000  # ms
    num_tokens = len(ids)
    return encode_time, decode_time, num_tokens

if __name__ == "__main__":
    if "--demo" in sys.argv:
        sample = (
    "The Internet is the global system of interconnected computer networks that "
    "uses the Internet protocol suite (TCP/IP) to communicate between networks and devices. "
    "It is a network of networks that consists of private, public, academic, business, and government networks "
    "of local to global scope, linked by a broad array of electronic, wireless, and optical networking technologies. "
    "The Internet carries a vast range of information resources and services, such as the inter-linked hypertext documents "
    "and applications of the World Wide Web (WWW), electronic mail, telephony, and file sharing.\n"
) * 25  # Repeat 25 times to ensure a long sample (about 4000+ words)
        print(f"Sample text (first 60 chars): {sample[:60]!r}... (len={len(sample)})\n")
        print(f"{'Tokenizer':<16} | {'Encode ms':>9} | {'Decode ms':>9} | {'# Tokens':>8}")
        print("-" * 52)
        for tok_cls in [BasicTokenizer, RegexTokenizer, GPT4Tokenizer]:
            try:
                encode_ms, decode_ms, num_tokens = measure_tokenizer_performance(tok_cls, sample)
                print(f"{tok_cls.__name__:<16} | {encode_ms:9.2f} | {decode_ms:9.2f} | {num_tokens:8}")
            except Exception as e:
                print(f"{tok_cls.__name__:<16} | ERROR: {e}")
        print("\nInterpretation:")
        print("- Fewer tokens means more efficient tokenization.")
        print("- Lower encode/decode time means faster performance.")
        print("- GPT4Tokenizer may be slower but more efficient.")
    else:
        import pytest
        pytest.main()
