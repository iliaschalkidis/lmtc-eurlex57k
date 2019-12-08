import spacy


def tokenize_doc_en(doc):

    tokens = []
    for token in doc:
        if '\n' in token.text:
            tokens.append(token)
        elif token.tag_ != '_SP' and token.text.strip(' '):
            tokens.append(token)

    return tokens


class Tagger(object):

    def __init__(self):
        self._spacy_tagger = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def tokenize_text(self, text: str):

        return [t for t in self._spacy_tagger(text)]
