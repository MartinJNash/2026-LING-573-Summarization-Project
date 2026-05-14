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

- **Truncation drops sharply** with each limit increase: 38% → 11% → 2.3%.
- **384 → 512 has negligible length impact** (+2w avg): the model isn't generating longer text, it's completing sentences that were previously cut off.
- **The default (256) ceiling is visible in the data**: median (139w) exceeds mean (127w) because many outputs pile up against the hard token cap, skewing the distribution.
- **All settings over-generate vs. gold**: even at 512 tokens, outputs average +46w more than human summaries. This is a model behavior issue, not a token limit issue.
- **79 outputs remain truncated at 512**: these are edge cases averaging 183w (max 368w); 72 were also truncated at 384. They can reasonably be filtered from eval.
