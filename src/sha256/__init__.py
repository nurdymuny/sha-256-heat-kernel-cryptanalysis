"""
Instrumented SHA-256 implementation.

This module provides a full SHA-256 implementation from the FIPS 180-4 spec,
with hooks to capture internal state at configurable round intervals.
"""

from .core import InstrumentedSHA256, SHA256State, SHA256Trajectory
from .input_generator import InputGenerator

__all__ = [
    'InstrumentedSHA256',
    'SHA256State', 
    'SHA256Trajectory',
    'InputGenerator',
]
