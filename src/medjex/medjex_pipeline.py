import argparse
import json

from mlm_scorer import MLMScorer
from noun_phrase_extractor import NounPhraseExtractor, SubstringSpan
from wiki_definition_lookup import WikiDefinitionLookup

MIN_MLM_SCORE = 8

class Medjex:

    def __init__(self):
        self.extractor = NounPhraseExtractor()
        self.scorer = MLMScorer()
        self.wiki_lookup = WikiDefinitionLookup()

    def compute_replacements(self, prediction: str) -> list[tuple[SubstringSpan, str]]:
        replacements: list[tuple[SubstringSpan, str]] = []
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
            
            wiki_definition = self.wiki_lookup.lookup(phrase_span.substring())
            if wiki_definition:
                replacements.append(
                    (phrase_span, wiki_definition)
                )

        return replacements

    def apply_replacements(self, original_text: str, replacements: list[tuple[SubstringSpan, str]]) -> str:
        edited = original_text
        backward_replacements = sorted(replacements, key=lambda x: x[0].start, reverse=True)
        for span, replacement in backward_replacements:
            annotation = f"{span.substring()} ({replacement})"
            edited = edited[:span.start] + annotation + edited[span.end:]

        return edited


def work_on_file(input: str, output: str):
    with open(input, "r") as f:
        data = json.load(f)

    medjex = Medjex()
    for ex in data["examples"]:
        prediction = ex["pred"]
        replacements = medjex.compute_replacements(prediction=prediction)
        sans_jargon = medjex.apply_replacements(original_text=prediction, replacements=replacements)
        ex["post-medjex"] = sans_jargon        
        ex["medjex-replacements"] = replcement_as_json(replacements=replacements)

        print(f"\n\n{ex['medjex-replacements']}")

    with open(output, 'w') as f:
        json.dump(data, f, indent=2)

def replcement_as_json(replacements: list[tuple[SubstringSpan, str]]) -> dict[str, str]:
    results: dict[str, str] = dict()
    for span, replacement in replacements:
        results[span.substring()] = replacement
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    work_on_file(input=args.input, output=args.output)

if __name__ == "__main__":
    main()

