import argparse
import json
from src.mlm_scorer import MLMScorer
from src.noun_phrase_extractor import NounPhraseExtractor
from src.wiki_definition_lookup import WikiDefinitionLookup


MIN_MLM_SCORE = 8

class Medjex:

    def __init__(self):
        self.extractor = NounPhraseExtractor()
        self.scorer = MLMScorer()
        self.wiki_lookup = WikiDefinitionLookup()
    
    def work_on_prediction(self, prediction: str):
        phrases = self.extractor.extract_noun_phrases(prediction)
        for phrase in phrases:
            score = self.scorer.mlm_score(
                text=phrase.text, 
                span_start=phrase.start, 
                span_end=phrase.end
            )

            if score is not None and score > MIN_MLM_SCORE:
                print(f"{phrase.substring()} --> {score}")



def work_on_file(input: str):
    with open(input, "r") as f:
        data = json.load(f)

    medjex = Medjex()
    examples = data["examples"]
    for ex in examples:
        prediction = ex["pred"]
        medjex.work_on_prediction(prediction=prediction)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    work_on_file(input=args.input)

if __name__ == "__main__":
    main()

