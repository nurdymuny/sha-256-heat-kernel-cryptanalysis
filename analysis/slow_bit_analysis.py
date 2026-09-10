"""Map input-message bit positions; descriptive counts, not register inference."""
from collections import Counter

def bit_to_word(bit_position):
    if not 0 <= bit_position < 440:
        raise ValueError("Expected an MSB-first bit in a 55-byte message")
    return f"W{bit_position//32}", bit_position%32

def analyze_slow_bits(slow_bits):
    mappings=[]
    for bit in slow_bits:
        word,offset=bit_to_word(int(bit))
        mappings.append({'input_bit':int(bit),'message_word':word,
                         'msb_first_offset':offset,'lsb_number':31-offset})
    return {'mappings':mappings,'counts':dict(Counter(m['message_word'] for m in mappings)),
            'interpretation':'selected input positions; selection is not a significance test'}

def print_analysis(slow_bits,label="Selected input bits"):
    result=analyze_slow_bits(slow_bits)
    print(label,result)
    return result
