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

## Sentence-Level Analysis

Sentence counts computed by splitting on `.`, `!`, `?` followed by whitespace.

| | Mean sentences | Median | 1–3 sent | 4–6 sent | 7–9 sent | 10+ sent |
|---|---:|---:|---:|---:|---:|---:|
| **gold** | 5.5 | 5 | 856 (25%) | 1,477 (43%) | 777 (23%) | 286 (8%) |
| **default (256 tokens)** | 7.2 | 7 | 455 (13%) | 949 (28%) | 1,179 (35%) | 813 (24%) |
| **384 tokens** | 8.0 | 7 | 454 (13%) | 953 (28%) | 851 (25%) | 1,138 (34%) |
| **512 tokens** | 8.1 | 7 | 460 (14%) | 945 (28%) | 848 (25%) | 1,143 (34%) |

Gold summaries concentrate in the **4–6 sentence range** (43%), with a median of 5 sentences. Model outputs across all token limits skew toward 7–10+ sentences — 59% of 512-token predictions exceed 6 sentences vs. 31% of gold. Sentence count stabilizes between 384 and 512 (median 7 at both), confirming the verbosity is model behavior rather than a truncation artifact.

## Key Takeaways

- **512 tokens reduces truncation to a negligible 2.3%** (vs. 38% at default and 11% at 384), with almost no change in average output length (+2w over 384) — the extra headroom completes sentences rather than generating more text.
- **All settings over-generate relative to gold**: model outputs compress to ~26% of the input across all token limits, compared to ~19% for human-written summaries (~46 extra words per summary on average). This is a model behavior issue independent of the token limit.
