"""
Instrumented SHA-256 Implementation
===================================

Full implementation of SHA-256 from FIPS 180-4 specification,
with hooks to capture internal state at any round.

This is NOT using hashlib - we need access to intermediate states.

Author: Bee Davis
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# SHA-256 Constants (first 32 bits of fractional parts of cube roots of first 64 primes)
K = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
], dtype=np.uint32)

# Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
H0 = np.array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
], dtype=np.uint32)


@dataclass
class SHA256State:
    """State capture at a specific round."""
    round_num: int                    # 0-64 (0 = initial, 64 = after last round)
    working_vars: np.ndarray          # shape (8,) dtype uint32 [a,b,c,d,e,f,g,h]
    state_vector: np.ndarray          # shape (256,) dtype uint8 (bit-expanded)
    block_index: int = 0
    round_in_block: int = 0
    
    def to_bytes(self) -> bytes:
        """Convert working variables to 32-byte representation."""
        return b''.join(int(v).to_bytes(4, 'big') for v in self.working_vars)


@dataclass  
class SHA256Trajectory:
    """Complete trajectory through SHA-256 computation."""
    input_message: bytes              # original input (before padding)
    padded_message: bytes             # after SHA-256 padding
    states: List[SHA256State]         # states at sampled rounds
    final_hash: bytes                 # 256-bit output
    
    @property
    def num_blocks(self) -> int:
        """Number of 512-bit blocks processed."""
        return len(self.padded_message) // 64

    def state_at(self, round_in_block: int, block_index: int = 0) -> SHA256State:
        """Select an explicit block/local-round pair, including block boundaries."""
        matches = [s for s in self.states if s.block_index == block_index
                   and s.round_in_block == round_in_block]
        if len(matches) != 1:
            raise ValueError(f"Expected one capture for block {block_index}, round {round_in_block}; found {len(matches)}")
        return matches[0]


def _rotr(x: np.uint32, n: int) -> np.uint32:
    """Right rotate a 32-bit integer by n bits."""
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def _shr(x: np.uint32, n: int) -> np.uint32:
    """Right shift a 32-bit integer by n bits."""
    return x >> n


def _ch(e: np.uint32, f: np.uint32, g: np.uint32) -> np.uint32:
    """SHA-256 Ch function: (e AND f) XOR ((NOT e) AND g)."""
    return (e & f) ^ ((~e) & g) & 0xFFFFFFFF


def _maj(a: np.uint32, b: np.uint32, c: np.uint32) -> np.uint32:
    """SHA-256 Maj function: (a AND b) XOR (a AND c) XOR (b AND c)."""
    return (a & b) ^ (a & c) ^ (b & c)


def _sigma0(x: np.uint32) -> np.uint32:
    """SHA-256 Σ0: ROTR²(x) XOR ROTR¹³(x) XOR ROTR²²(x)."""
    return _rotr(x, 2) ^ _rotr(x, 13) ^ _rotr(x, 22)


def _sigma1(x: np.uint32) -> np.uint32:
    """SHA-256 Σ1: ROTR⁶(x) XOR ROTR¹¹(x) XOR ROTR²⁵(x)."""
    return _rotr(x, 6) ^ _rotr(x, 11) ^ _rotr(x, 25)


def _gamma0(x: np.uint32) -> np.uint32:
    """SHA-256 σ0: ROTR⁷(x) XOR ROTR¹⁸(x) XOR SHR³(x)."""
    return _rotr(x, 7) ^ _rotr(x, 18) ^ _shr(x, 3)


def _gamma1(x: np.uint32) -> np.uint32:
    """SHA-256 σ1: ROTR¹⁷(x) XOR ROTR¹⁹(x) XOR SHR¹⁰(x)."""
    return _rotr(x, 17) ^ _rotr(x, 19) ^ _shr(x, 10)


def _expand_to_bits(working_vars: np.ndarray) -> np.ndarray:
    """
    Expand 8 x 32-bit working variables to 256-bit vector.
    
    Args:
        working_vars: shape (8,) uint32 array [a,b,c,d,e,f,g,h]
    
    Returns:
        shape (256,) uint8 array with individual bits
    """
    bits = np.zeros(256, dtype=np.uint8)
    for i, var in enumerate(working_vars):
        var_int = int(var) & 0xFFFFFFFF
        for bit_pos in range(32):
            # Big-endian bit ordering (MSB first within each word)
            bits[i * 32 + bit_pos] = (var_int >> (31 - bit_pos)) & 1
    return bits


def _pad_message(message: bytes) -> bytes:
    """
    Apply SHA-256 padding to message.
    
    Padding: append 1 bit, then 0 bits until length ≡ 448 (mod 512),
    then append 64-bit big-endian length.
    """
    msg_len = len(message)
    msg_len_bits = msg_len * 8
    
    # Append the bit '1' to message (as 0x80 byte)
    padded = bytearray(message)
    padded.append(0x80)
    
    # Append zeros until length ≡ 56 (mod 64) bytes = 448 (mod 512) bits
    while len(padded) % 64 != 56:
        padded.append(0x00)
    
    # Append original length as 64-bit big-endian integer
    padded.extend(msg_len_bits.to_bytes(8, 'big'))
    
    return bytes(padded)


def _parse_block(block: bytes) -> np.ndarray:
    """Parse a 512-bit (64-byte) block into 16 x 32-bit words."""
    assert len(block) == 64, f"Block must be 64 bytes, got {len(block)}"
    words = np.zeros(16, dtype=np.uint32)
    for i in range(16):
        words[i] = int.from_bytes(block[i*4:(i+1)*4], 'big')
    return words


def _prepare_message_schedule(block_words: np.ndarray) -> np.ndarray:
    """
    Prepare the message schedule W[0..63] from the 16-word block.
    
    W[i] = M[i] for i = 0..15
    W[i] = σ1(W[i-2]) + W[i-7] + σ0(W[i-15]) + W[i-16] for i = 16..63
    """
    W = np.zeros(64, dtype=np.uint32)
    W[:16] = block_words
    
    for i in range(16, 64):
        W[i] = (int(_gamma1(W[i-2])) + int(W[i-7]) + 
                int(_gamma0(W[i-15])) + int(W[i-16])) & 0xFFFFFFFF
    
    return W


class InstrumentedSHA256:
    """
    SHA-256 implementation with state trajectory capture.
    
    This implements SHA-256 exactly per FIPS 180-4, but exposes internal
    state at configurable round intervals for geometric analysis.
    """
    
    def __init__(self, sample_rounds: Optional[List[int]] = None):
        """
        Args:
            sample_rounds: Which rounds to capture state (0-64).
                          Default captures every 8 rounds.
                          0 = initial state, 64 = after last round.
        """
        if sample_rounds is None:
            self.sample_rounds = [0, 8, 16, 24, 32, 40, 48, 56, 64]
        else:
            self.sample_rounds = sorted(sample_rounds)
    
    def _capture_state(self, round_num: int, working_vars: np.ndarray) -> SHA256State:
        """Capture current state if this round is in sample_rounds."""
        return SHA256State(
            round_num=round_num,
            working_vars=working_vars.copy(),
            state_vector=_expand_to_bits(working_vars)
        )
    
    def _process_block(
        self, 
        block: bytes, 
        H: np.ndarray
    ) -> tuple[np.ndarray, List[SHA256State]]:
        """
        Process a single 512-bit block.
        
        Returns:
            (updated_H, list of captured states)
        """
        states = []
        
        # Parse block and prepare message schedule
        M = _parse_block(block)
        W = _prepare_message_schedule(M)
        
        # Initialize working variables
        a, b, c, d, e, f, g, h = [np.uint32(x) for x in H]
        
        # Capture initial state (round 0)
        working = np.array([a, b, c, d, e, f, g, h], dtype=np.uint32)
        if 0 in self.sample_rounds:
            states.append(self._capture_state(0, working))
        
        # 64 rounds
        for i in range(64):
            # Compute round function
            S1 = _sigma1(e)
            ch = _ch(e, f, g)
            temp1 = (int(h) + int(S1) + int(ch) + int(K[i]) + int(W[i])) & 0xFFFFFFFF
            S0 = _sigma0(a)
            maj = _maj(a, b, c)
            temp2 = (int(S0) + int(maj)) & 0xFFFFFFFF
            
            # Update working variables
            h = g
            g = f
            f = e
            e = np.uint32((int(d) + temp1) & 0xFFFFFFFF)
            d = c
            c = b
            b = a
            a = np.uint32((temp1 + temp2) & 0xFFFFFFFF)
            
            # Capture state after this round (round i+1)
            round_num = i + 1
            if round_num in self.sample_rounds:
                working = np.array([a, b, c, d, e, f, g, h], dtype=np.uint32)
                states.append(self._capture_state(round_num, working))
        
        # Compute new hash value
        new_H = np.array([
            (int(H[0]) + int(a)) & 0xFFFFFFFF,
            (int(H[1]) + int(b)) & 0xFFFFFFFF,
            (int(H[2]) + int(c)) & 0xFFFFFFFF,
            (int(H[3]) + int(d)) & 0xFFFFFFFF,
            (int(H[4]) + int(e)) & 0xFFFFFFFF,
            (int(H[5]) + int(f)) & 0xFFFFFFFF,
            (int(H[6]) + int(g)) & 0xFFFFFFFF,
            (int(H[7]) + int(h)) & 0xFFFFFFFF,
        ], dtype=np.uint32)
        
        return new_H, states
    
    def hash_with_trajectory(self, message: bytes) -> SHA256Trajectory:
        """
        Compute SHA-256 hash and return full trajectory of sampled states.
        
        Args:
            message: Input message (any length)
        
        Returns:
            SHA256Trajectory with input, states at sampled rounds, and final hash
        """
        # Pad message
        padded = _pad_message(message)
        
        # Initialize hash values
        H = H0.copy()
        
        # Process each 512-bit block
        all_states = []
        num_blocks = len(padded) // 64
        
        for block_idx in range(num_blocks):
            block = padded[block_idx * 64 : (block_idx + 1) * 64]
            H, block_states = self._process_block(block, H)
            # Add block index to state round numbers for multi-block messages
            for state in block_states:
                state.block_index = block_idx
                state.round_in_block = state.round_num
                state.round_num += block_idx * 64
            all_states.extend(block_states)
        
        # Compute final hash
        final_hash = b''.join(int(h).to_bytes(4, 'big') for h in H)
        
        return SHA256Trajectory(
            input_message=message,
            padded_message=padded,
            states=all_states,
            final_hash=final_hash
        )
    
    def hash_batch(self, messages: List[bytes]) -> List[SHA256Trajectory]:
        """
        Batch processing for parallelization.
        
        Args:
            messages: List of input messages
        
        Returns:
            List of SHA256Trajectory objects
        """
        return [self.hash_with_trajectory(msg) for msg in messages]
    
    def hash(self, message: bytes) -> bytes:
        """
        Simple hash function - just returns the digest.
        
        Args:
            message: Input message
        
        Returns:
            32-byte SHA-256 digest
        """
        return self.hash_with_trajectory(message).final_hash


def validate_implementation():
    """
    Validate SHA-256 implementation against FIPS 180-4 test vectors.
    
    Returns:
        True if all tests pass
    
    Raises:
        AssertionError if any test fails
    """
    sha = InstrumentedSHA256()
    
    # Test vector 1: empty string
    expected1 = bytes.fromhex('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
    result1 = sha.hash(b'')
    assert result1 == expected1, f"Empty string failed: got {result1.hex()}"
    
    # Test vector 2: "abc"
    expected2 = bytes.fromhex('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
    result2 = sha.hash(b'abc')
    assert result2 == expected2, f"'abc' failed: got {result2.hex()}"
    
    # Test vector 3: longer message (448 bits = 56 bytes, fits in one block after padding)
    msg3 = b'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'
    expected3 = bytes.fromhex('248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1')
    result3 = sha.hash(msg3)
    assert result3 == expected3, f"Long message failed: got {result3.hex()}"
    
    print("✓ All SHA-256 test vectors pass")
    return True


if __name__ == "__main__":
    validate_implementation()
    
    # Demo trajectory capture
    sha = InstrumentedSHA256(sample_rounds=[0, 16, 32, 48, 64])
    trajectory = sha.hash_with_trajectory(b"Hello, geometric cryptanalysis!")
    
    print(f"\nTrajectory for 'Hello, geometric cryptanalysis!':")
    print(f"  Input length: {len(trajectory.input_message)} bytes")
    print(f"  Padded length: {len(trajectory.padded_message)} bytes ({trajectory.num_blocks} block(s))")
    print(f"  Final hash: {trajectory.final_hash.hex()}")
    print(f"  Captured {len(trajectory.states)} states:")
    for state in trajectory.states:
        print(f"    Round {state.round_num}: vars={[hex(v) for v in state.working_vars[:4]]}...")
