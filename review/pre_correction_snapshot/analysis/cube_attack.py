import struct

# --- 1. Compact Pure Python SHA-256 for Reduced Rounds ---
def rotr(x, n):
    return (x >> n) | (x << (32 - n)) & 0xFFFFFFFF

def sha256_reduced(message_bytes, rounds=16):
    """
    SHA-256 implementation that can stop at 'rounds'.
    """
    K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]

    h_init = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    ]

    # Padding
    length = len(message_bytes) * 8
    message_bytes += b'\x80'
    while (len(message_bytes) * 8 + 64) % 512 != 0:
        message_bytes += b'\x00'
    message_bytes += struct.pack('>Q', length)

    # Process blocks (assuming 1 block for this test)
    chunk = message_bytes[:64]
    w = [0] * 64
    for i in range(16):
        w[i] = struct.unpack('>I', chunk[i*4:(i+1)*4])[0]

    # Minimal Message Schedule
    for i in range(16, rounds): # Only compute needed schedule
        s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
        w[i] = (w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF

    a, b, c, d, e, f, g, h = h_init

    # Compression Loop
    for i in range(rounds):
        s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        temp1 = (h + s1 + ch + K[i] + w[i]) & 0xFFFFFFFF
        s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (s0 + maj) & 0xFFFFFFFF

        h = g
        g = f
        f = e
        e = (d + temp1) & 0xFFFFFFFF
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xFFFFFFFF

    # Return the XOR sum of the state (simple reduction for the cube test)
    return [a, b, c, d, e, f, g, h]

# --- 2. The Sigma-Rotational Cube Attack ---

def run_cube_attack(target_round, cube_indices):
    """
    Sums outputs over a boolean cube defined by cube_indices.
    If the final XOR sum is 0, the algebraic degree is low (Distinguisher found).
    """
    print(f"\n[*] Running Cube Attack on Round {target_round}")
    print(f"[*] Cube Indices (Message Bits): {cube_indices}")
    print(f"[*] Cube Dimension: {len(cube_indices)}")
    
    # Initialize XOR sum accumulator
    state_sum = [0] * 8
    
    # Iterate through all 2^d possibilities
    for i in range(1 << len(cube_indices)):
        # Construct message
        msg_int = 0
        
        # Apply the cube bits
        for bit_idx, cube_pos in enumerate(cube_indices):
            if (i >> bit_idx) & 1:
                msg_int |= (1 << cube_pos)
        
        # Add some fixed constants to avoid trivial zero-inputs
        msg_int |= (0xDEADBEEF << 64) 
        
        # Convert to bytes
        msg_bytes = msg_int.to_bytes(64, 'big') # 512-bit block
        
        # Run Reduced SHA-256
        output_state = sha256_reduced(msg_bytes, rounds=target_round)
        
        # XOR into accumulator
        for j in range(8):
            state_sum[j] ^= output_state[j]

    return state_sum

# --- 3. Configuration based on User's Log ---

# Cube Bits Selection:
# Based on your log: "Bit 288: message bit 32" and "Bit 295: message bit 39"
# Plus we target the "carry" bits near them to encourage interaction.
# We also include bits that hit the Sigma constants (2, 13, 22) in the first word.
target_bits = [
    2, 13, 22,      # The Sigma_0 constants (Word 0)
    32, 39,         # The explicit slow bits from your log (Word 1)
    32+2, 32+13,    # Neighbors of the slow bits
]

# Run the attack on the "Transition Cliff"
# Your log says Isotropic at 16. If we get [0,0,0...] at 16, we broke it.
for r in [15, 16, 17]:
    result = run_cube_attack(target_round=r, cube_indices=target_bits)
    
    # Check if broken
    is_zero = all(x == 0 for x in result)
    status = "BROKEN (Distinguisher Found!)" if is_zero else "Random-like"
    print(f"--> Round {r} Result: {result}")
    print(f"--> Status: {status}")