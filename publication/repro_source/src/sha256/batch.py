"""Vectorized raw-block compression for controlled measurements.

Input coordinates are MSB-first within each byte. Captures are working
registers after r rounds, before feed-forward; they are never called digests.
This implementation is cross-checked against the instrumented scalar core.
"""
import numpy as np
from .core import K, H0, _pad_message


def padded_blocks(messages):
    blocks=[_pad_message(bytes(m)) for m in messages]
    if any(len(b)!=64 for b in blocks):
        raise ValueError('This batch API requires messages of at most 55 bytes')
    return np.frombuffer(b''.join(blocks),dtype=np.uint8).reshape(-1,64).copy()


def compress_blocks(blocks, rounds=(64,)):
    blocks=np.asarray(blocks,dtype=np.uint8)
    if blocks.ndim!=2 or blocks.shape[1]!=64:
        raise ValueError('Expected (n,64) raw blocks')
    rounds=sorted(set(rounds))
    if not rounds or min(rounds)<0 or max(rounds)>64:
        raise ValueError('Rounds must be between 0 and 64')
    n=len(blocks)
    w=np.zeros((max(16,max(rounds)),n),dtype=np.uint32)
    w[:16]=np.ascontiguousarray(blocks).view('>u4').astype(np.uint32).T
    def rr(x,r): return (x>>r)|(x<<np.uint32(32-r))
    for t in range(16,max(rounds)):
        x=w[t-15]; y=w[t-2]
        s0=rr(x,7)^rr(x,18)^(x>>3)
        s1=rr(y,17)^rr(y,19)^(y>>10)
        w[t]=w[t-16]+s0+w[t-7]+s1
    v=np.repeat(H0[:,None],n,axis=1)
    result={0:v.T.copy()} if 0 in rounds else {}
    for t in range(max(rounds)):
        a,b,c,d,e,f,g,h=v
        t1=h+(rr(e,6)^rr(e,11)^rr(e,25))+((e&f)^((~e)&g))+K[t]+w[t]
        t2=(rr(a,2)^rr(a,13)^rr(a,22))+((a&b)^(a&c)^(b&c))
        v=np.array([t1+t2,a,b,c,d+t1,e,f,g],dtype=np.uint32)
        if t+1 in rounds: result[t+1]=v.T.copy()
    return result


def state_bits(words):
    return np.unpackbits(np.asarray(words,dtype='>u4').view(np.uint8).reshape(-1,32),axis=1)


def flip_input_bits(blocks, positions):
    """Flip one declared MSB-first message position per row."""
    out=np.asarray(blocks,dtype=np.uint8).copy()
    pos=np.broadcast_to(np.asarray(positions,dtype=int),(len(out),))
    if np.any((pos<0)|(pos>=512)): raise ValueError('Bit out of raw-block range')
    out[np.arange(len(out)),pos//8]^=(1<<(7-pos%8)).astype(np.uint8)
    return out


def cube_states(base_blocks, positions, rounds):
    """Return XOR of working words for each base, round; all cube bits canonical."""
    positions=list(positions)
    if len(set(positions))!=len(positions) or not positions:
        raise ValueError('Cube positions must be nonempty and distinct')
    corners=1<<len(positions)
    base=np.asarray(base_blocks,dtype=np.uint8)
    expanded=np.repeat(base,corners,axis=0)
    indices=np.tile(np.arange(corners),len(base))
    for j,pos in enumerate(positions):
        if not 0<=pos<512: raise ValueError('Cube bit outside block')
        expanded[:,pos//8]^=(((indices>>j)&1)<<(7-pos%8)).astype(np.uint8)
    captures=compress_blocks(expanded,rounds)
    return {r:np.bitwise_xor.reduce(v.reshape(len(base),corners,8),axis=1) for r,v in captures.items()}


def gf2_rank(matrix):
    a=np.asarray(matrix,dtype=np.uint8).copy()%2
    row=0
    for col in range(a.shape[1]):
        pivots=np.flatnonzero(a[row:,col])
        if not len(pivots): continue
        p=row+pivots[0]; a[[row,p]]=a[[p,row]]
        mask=np.flatnonzero(a[:,col]); mask=mask[mask!=row]
        a[mask]^=a[row]; row+=1
        if row==a.shape[0]: break
    return row
