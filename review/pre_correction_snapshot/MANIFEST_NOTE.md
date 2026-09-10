# Snapshot manifest note

`review/pre_correction_snapshot/` preserves the December 2025 code before the September 2026 corrections.
On 2026-09-09, in the five files below, docstring or message lines that carried a confidentiality classification or an
unsupported claim of a cryptographic break were replaced by neutral lines of identical line count, so that line references in
`review/SHA256_SUBMISSION_REVIEW.md` remain valid. No executable logic changed. The original SHA-256 hashes are recorded here;
`manifest.json` carries the current hashes.

| File | Original SHA-256 | Current SHA-256 |
|---|---|---|
| `analysis/davis_manifold_relaxation.py` | `681d61884e64133a7602b096f7fb217fe0d9e774f59b26c2d64c306184953ddb` | `80a66f0105bf5927727bee6c0f3ee382ec40e530943d2646d6ec3721b0d98678` |
| `analysis/sha256_distinguisher_test.py` | `11d93f4f4b65afb13f805113d0d50da3a0d51124c116ef3eabf2066d7eec722c` | `63a184e5d55b1a2f16521a2241636ba2fb33d7481c721e89d14cdbb64420196e` |
| `analysis/sha256_geometric_analysis.py` | `480c35e781cd9fd598d4edc7b896f9c76dd319709c30df656c33612b0b50fd3c` | `13398f72675eddfb53a0015f1ad720f57cb2f561e61f14e0472f6116e16bd386` |
| `analysis/sha256_geometric_attacks.py` | `7a9bfd6ec7e9c6824da0ad75805309181f81fb28c29b11eaa02cb646159f8391` | `4feb63168bedbce43adba90dad123353496b715c11e06439a8ed603ac7702241` |
| `analysis/distinguisher.py` | `ee6a9105f380b7b99f4cde7f5cce9b402f68fb1b5e5ffeb649b6cff9b466d7fc` | `ff3d830a64911c337d0c68aa9573feb9d4d1eb19175e70150a67c5d0eea5133d` |
