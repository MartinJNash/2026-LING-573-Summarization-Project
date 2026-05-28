import argparse
import json
from src.mlm_scorer import MLMScorer
from src.noun_phrase_extractor import NounPhraseExtractor, SubstringSpan
from dataclasses import dataclass, asdict

@dataclass
class SpanScore:
    text: str
    start: int
    end: int
    score: float

MIN_MLM_SCORE = 8

class MLMScorePipeline:
    """
    Read inference file from --input
    Calculate MLM scores for noun phrases in "pred" field
    Write JSON file with "mlm-scores" dictionary for each prediction
    """

    def __init__(self):
        self.extractor = NounPhraseExtractor()
        self.scorer = MLMScorer()

    def compute_replacements(self, prediction: str) -> list[SpanScore]:
        replacements: list[SpanScore] = []
        phrases = self.extractor.extract_noun_phrases(prediction)
        for phrase_span in phrases:
            score = self.scorer.mlm_score(
                text=phrase_span.text, 
                span_start=phrase_span.start, 
                span_end=phrase_span.end
            )

            # Skip low scores
            if score is None or score < MIN_MLM_SCORE:
                continue
            
            replacements.append(SpanScore(
                text=phrase_span.substring(),
                start=phrase_span.start,
                end=phrase_span.end,
                score=score,
            ))

        return replacements

def work_on_file(input: str, output: str):
    with open(input, "r") as f:
        data = json.load(f)

    scorer = MLMScorePipeline()
    for ex in data["examples"]:
        prediction = ex["pred"]
        replacements = scorer.compute_replacements(prediction=prediction)
        replacement_json = replacement_as_json(replacements=replacements)
        print(f"{replacement_json}\n\n")
        ex["mlm-scores"] = replacement_json 

    with open(output, 'w') as f:
        json.dump(data, f, indent=2)

def replacement_as_json(replacements: list[SpanScore]) -> list[dict[str, str | float | int]]:
    return [ asdict(span) for span in replacements ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    work_on_file(input=args.input, output=args.output)

if __name__ == "__main__":
    main()

