"""
Input Generator for SHA-256 Experiments
========================================

Generates controlled input sets for heat kernel experiments:
- Random messages for baseline analysis
- Hamming-1 pairs for avalanche analysis
- Structured sets for control experiments
- Adversarial sets for stress testing

Author: Bee Davis
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HammingPair:
    """A pair of messages differing by exactly one bit."""
    message_a: bytes
    message_b: bytes
    flipped_bit: int  # Which bit position was flipped (0-511 for 64-byte message)


class InputGenerator:
    """
    Generate controlled input sets for SHA-256 experiments.
    
    All methods generate pre-padded 512-bit (64-byte) messages unless
    otherwise specified, ensuring single-block SHA-256 processing.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
    
    def random_batch(
        self, 
        n: int, 
        message_size: int = 55
    ) -> List[bytes]:
        """
        Generate n random messages.
        
        Args:
            n: Number of messages to generate
            message_size: Size in bytes before padding (max 55 for single block)
        
        Returns:
            List of random byte strings
        """
        return [
            self.rng.bytes(message_size)
            for _ in range(n)
        ]
    
    def random_prepadded(self, n: int) -> List[bytes]:
        """
        Generate n random pre-padded 512-bit messages.
        
        These are raw 64-byte blocks with proper SHA-256 padding structure,
        ready for direct processing.
        
        Returns:
            List of 64-byte messages with SHA-256 padding
        """
        messages = []
        for _ in range(n):
            # Random content (55 bytes max for single block)
            content_len = self.rng.integers(1, 56)
            content = bytes(self.rng.bytes(content_len))
            
            # Apply SHA-256 padding
            padded = self._pad_message(content)
            messages.append(padded)
        
        return messages
    
    def _pad_message(self, message: bytes) -> bytes:
        """Apply SHA-256 padding."""
        msg_len = len(message)
        msg_len_bits = msg_len * 8
        
        padded = bytearray(message)
        padded.append(0x80)
        
        while len(padded) % 64 != 56:
            padded.append(0x00)
        
        padded.extend(msg_len_bits.to_bytes(8, 'big'))
        return bytes(padded)
    
    def hamming_pairs(
        self, 
        n: int, 
        bit_positions: Optional[List[int]] = None,
        message_size: int = 55
    ) -> List[HammingPair]:
        """
        Generate n pairs of messages differing by exactly 1 bit.
        
        Args:
            n: Number of pairs to generate
            bit_positions: Which bit positions to flip. If None, sample uniformly.
            message_size: Size of base message in bytes
        
        Returns:
            List of HammingPair objects
        """
        total_bits = message_size * 8
        pairs = []
        
        for _ in range(n):
            # Generate base message
            base = bytearray(self.rng.bytes(message_size))
            
            # Choose bit to flip
            if bit_positions is not None:
                bit_pos = self.rng.choice(bit_positions)
            else:
                bit_pos = self.rng.integers(0, total_bits)
            
            # Create flipped version
            flipped = bytearray(base)
            byte_idx = bit_pos // 8
            bit_idx = 7 - (bit_pos % 8)  # Big-endian bit ordering
            flipped[byte_idx] ^= (1 << bit_idx)
            
            pairs.append(HammingPair(
                message_a=bytes(base),
                message_b=bytes(flipped),
                flipped_bit=bit_pos
            ))
        
        return pairs
    
    def hamming_sweep(
        self,
        base_message: bytes,
        bit_positions: Optional[List[int]] = None
    ) -> List[HammingPair]:
        """
        Generate all Hamming-1 neighbors of a base message.
        
        Args:
            base_message: The base message
            bit_positions: Which bits to flip (default: all)
        
        Returns:
            List of HammingPair objects, one per bit position
        """
        total_bits = len(base_message) * 8
        
        if bit_positions is None:
            bit_positions = list(range(total_bits))
        
        pairs = []
        for bit_pos in bit_positions:
            flipped = bytearray(base_message)
            byte_idx = bit_pos // 8
            bit_idx = 7 - (bit_pos % 8)
            flipped[byte_idx] ^= (1 << bit_idx)
            
            pairs.append(HammingPair(
                message_a=base_message,
                message_b=bytes(flipped),
                flipped_bit=bit_pos
            ))
        
        return pairs
    
    def structured_set(
        self, 
        pattern: str, 
        n: int,
        message_size: int = 55
    ) -> List[bytes]:
        """
        Generate messages with specific structure for control experiments.
        
        Args:
            pattern: Type of structure:
                'zeros': all zeros
                'ones': all ones (0xFF bytes)
                'counter': sequential integers
                'low_entropy': limited alphabet (just 4 distinct byte values)
                'alternating': alternating 0xAA/0x55 pattern
                'gradient': gradually increasing byte values
            n: Number of messages
            message_size: Size in bytes
        
        Returns:
            List of structured messages
        """
        messages = []
        
        if pattern == 'zeros':
            for _ in range(n):
                messages.append(b'\x00' * message_size)
        
        elif pattern == 'ones':
            for _ in range(n):
                messages.append(b'\xff' * message_size)
        
        elif pattern == 'counter':
            for i in range(n):
                # Embed counter in first 8 bytes, rest random
                counter_bytes = i.to_bytes(8, 'big')
                remaining = message_size - 8
                if remaining > 0:
                    msg = counter_bytes + bytes(self.rng.bytes(remaining))
                else:
                    msg = counter_bytes[:message_size]
                messages.append(msg)
        
        elif pattern == 'low_entropy':
            # Only use 4 distinct byte values
            alphabet = [0x00, 0x55, 0xAA, 0xFF]
            for _ in range(n):
                indices = self.rng.integers(0, 4, size=message_size)
                msg = bytes([alphabet[i] for i in indices])
                messages.append(msg)
        
        elif pattern == 'alternating':
            for i in range(n):
                if i % 2 == 0:
                    msg = bytes([0xAA if j % 2 == 0 else 0x55 for j in range(message_size)])
                else:
                    msg = bytes([0x55 if j % 2 == 0 else 0xAA for j in range(message_size)])
                messages.append(msg)
        
        elif pattern == 'gradient':
            for i in range(n):
                offset = i * 7 % 256  # Vary starting point
                msg = bytes([(j + offset) % 256 for j in range(message_size)])
                messages.append(msg)
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        return messages
    
    def adversarial_set(self, n: int) -> List[bytes]:
        """
        Generate messages designed to stress-test SHA-256 assumptions.
        
        Includes:
        - Near-collision attempts (similar internal states)
        - Weak message schedules
        - Symmetry-inducing inputs
        
        Args:
            n: Number of adversarial messages
        
        Returns:
            List of adversarial messages
        """
        messages = []
        
        # Type 1: Highly repetitive patterns (weak message schedules)
        for _ in range(n // 4):
            # Repeating short pattern
            pattern_len = self.rng.integers(1, 8)
            pattern = bytes(self.rng.bytes(pattern_len))
            msg = (pattern * (55 // pattern_len + 1))[:55]
            messages.append(msg)
        
        # Type 2: Palindromic messages (potential symmetry exploitation)
        for _ in range(n // 4):
            half_len = 27
            half = bytes(self.rng.bytes(half_len))
            msg = half + bytes([self.rng.integers(0, 256)]) + half[::-1]
            messages.append(msg)
        
        # Type 3: Messages with many zeros/ones (extreme Hamming weight)
        for _ in range(n // 4):
            if self.rng.random() < 0.5:
                # Sparse: mostly zeros with few ones
                msg = bytearray(55)
                num_ones = self.rng.integers(1, 20)
                for _ in range(num_ones):
                    pos = self.rng.integers(0, 55)
                    msg[pos] = self.rng.integers(1, 256)
            else:
                # Dense: mostly ones with few zeros  
                msg = bytearray(b'\xff' * 55)
                num_zeros = self.rng.integers(1, 20)
                for _ in range(num_zeros):
                    pos = self.rng.integers(0, 55)
                    msg[pos] = self.rng.integers(0, 255)
            messages.append(bytes(msg))
        
        # Type 4: Messages designed to create similar message schedules
        remaining = n - len(messages)
        for _ in range(remaining):
            # Use specific byte patterns that might cause W[] collisions
            base = bytes([i % 256 for i in range(55)])
            # Small random perturbation
            msg = bytearray(base)
            num_changes = self.rng.integers(1, 5)
            for _ in range(num_changes):
                pos = self.rng.integers(0, 55)
                msg[pos] ^= self.rng.integers(1, 256)
            messages.append(bytes(msg))
        
        return messages[:n]
    
    def collision_candidates(
        self, 
        target_hash: bytes, 
        n: int,
        similarity_bits: int = 32
    ) -> List[bytes]:
        """
        Generate messages that might produce hashes close to a target.
        
        This is for experimental analysis only - not actual collision finding.
        
        Args:
            target_hash: 32-byte hash to approach
            n: Number of candidates
            similarity_bits: How many leading bits to match (for filtering)
        
        Returns:
            List of candidate messages
        """
        # This is a placeholder - real collision finding would require
        # much more sophisticated techniques
        return self.random_batch(n)


if __name__ == "__main__":
    # Demo input generation
    gen = InputGenerator(seed=42)
    
    print("Random batch (5 messages):")
    for i, msg in enumerate(gen.random_batch(5)):
        print(f"  {i}: {msg[:20].hex()}... ({len(msg)} bytes)")
    
    print("\nHamming pairs (3 pairs):")
    for pair in gen.hamming_pairs(3):
        print(f"  Bit {pair.flipped_bit}: {pair.message_a[:10].hex()} -> {pair.message_b[:10].hex()}")
    
    print("\nStructured sets:")
    for pattern in ['zeros', 'counter', 'low_entropy']:
        msgs = gen.structured_set(pattern, 2)
        print(f"  {pattern}: {msgs[0][:20].hex()}...")
    
    print("\nAdversarial set (4 messages):")
    for msg in gen.adversarial_set(4):
        print(f"  {msg[:20].hex()}... ({len(msg)} bytes)")
