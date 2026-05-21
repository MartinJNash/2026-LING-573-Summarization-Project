import spacy
from dataclasses import dataclass

@dataclass
class SubstringSpan:
   text: str
   start: int
   end: int

   def substring(self) -> str:
      return self.text[self.start : self.end]

class NounPhraseExtractor:
  def __init__(self):
    self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

  def _should_include(self, span_text: str) -> bool:
    # short tokens
    if len(span_text) < 4:
        return False
    
    # pure numbers
    if span_text.replace(".", "").replace(",", "").isnumeric():
        return False

    # Already has parenthesis
    if span_text.startswith("("):
        return False

    return True

  def extract_noun_phrases(self, text: str) -> list[SubstringSpan]:

    # Parse text with spaCy to find noun chunks
    doc = self.nlp(text)
    candidates: list[SubstringSpan] = []

    for chunk in doc.noun_chunks:
        span_text = chunk.text.strip()
        start = chunk.start_char
        end = chunk.end_char

        # Filter candidates
        if not self._should_include(span_text):
            continue
        
        candidates.append(SubstringSpan(text, start, end))

    return candidates
