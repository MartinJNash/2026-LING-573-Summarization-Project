# Token Limit Analysis: biobart-large-lora

Comparison of model outputs across `max_new_tokens` settings vs. gold (human-written) summaries.
All stats computed on the full test set (n=3,396).

## Summary Table

| | Avg length | Median length | Max length | Truncated | Compression (avg) | Compression (median) |
|---|---:|---:|---:|---:|---:|---:|
| **gold** | 98w | 90w | 637w | — | 21.6% | 18.9% |
| **default (256 tokens)** | 127w | 139w | 201w | 1,297 (38.2%) | 28.5% | 24.7% |
| **384 tokens** | 142w | 135w | 295w | 361 (10.6%) | 31.7% | 26.1% |
| **512 tokens** | 144w | 136w | 369w | 79 (2.3%) | 32.1% | 26.2% |

**Compression** = predicted length / input length. Gold references compress to ~19% of the input; the model compresses to ~26% across all token limits, meaning it consistently over-generates by ~46 words relative to gold.

## Key Takeaways

- **512 tokens reduces truncation to a negligible 2.3%** (vs. 38% at default and 11% at 384), with almost no change in average output length (+2w over 384) — the extra headroom completes sentences rather than generating more text.
- **All settings over-generate relative to gold**: model outputs compress to ~26% of the input across all token limits, compared to ~19% for human-written summaries (~46 extra words per summary on average). This is a model behavior issue independent of the token limit.
