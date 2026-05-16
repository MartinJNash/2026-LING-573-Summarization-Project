import sys
import os

# Make the files visible
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from noun_phrase_extractor import NounPhraseExtractor
from mlm_scorer import MLMScorer

TEXT = "The stock market experienced significant volatility last week as investors reacted to rising inflation data and uncertainty about future interest rate decisions by the Federal Reserve "

extractor = NounPhraseExtractor()
scorer = MLMScorer()

noun_phrases = extractor.extract_noun_phrases(TEXT)

print(f"Input: {TEXT}\n")
print(f"{'Noun Phrase':<45} {'MLM Score':>10}")
print("-" * 57)

results = []
for span in noun_phrases:
    score = scorer.mlm_score(TEXT, span.start, span.end)
    results.append((TEXT[span.start:span.end], score))

for phrase, score in sorted(results, key=lambda x: x[1] or 0, reverse=True):
    score_str = f"{score:.4f}" if score is not None else "N/A"
    print(f"{phrase:<45} {score_str:>10}")
