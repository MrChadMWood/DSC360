from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import math
import nltk
from nltk.corpus import wordnet
from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd
import re
import spacy
import unicodedata

try:
    nltk.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')


class Normalizer:
    def __init__(self, lines_series):
        self.series = lines_series
        self.collapsed = []

    def collapse_rebuild(self, lines_series=None, collapse=False, rebuild=False):
        # function collapses series into single string for computationally
        # intense processes, like instantiating BeautifulSoup. Then rebuilds series.

        # Matches linebreaks, words+linebreaks, and words
        pattern = r"\n|\S+\n|\S+"

        if collapse:
            # Collapses input series to long string
            text = ' '.join(self)
            self.collapsed = text
            return self.collapsed

        if rebuild:
            # Rebuilds series as nested lists using word counts
            # Elements still seperated by regex pattern
            text_list = re.findall(pattern, self)
            total_words = 0
            new_list = []
            for line in lines_series:
                line_word_count = len(re.findall(pattern, line))
                temp = text_list[total_words: total_words + line_word_count]
                new_list.append(temp)
                total_words += line_word_count

            # Joins each nested list into single string
            for i in range(len(new_list)):
                new_list[i] = ' '.join(new_list[i])

            return pd.Series(new_list)

    # Strips HTML from each row. Performance optimized via collapse_rebuild
    def strip_html(self):
        # Collapses text to reduce instances of BeautifulSoup
        text = Normalizer.collapse_rebuild(self, collapse=True)

        soup = BeautifulSoup(text, 'html.parser')
        [s.decompose() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()

        # Rebuilds the collapsed text into series
        stripped_series = Normalizer.collapse_rebuild(stripped_text, self, rebuild=True)
        return stripped_series.apply(lambda x: x.replace(r'[\r|\n|\r\n]+', '\n'))

    # Removes all accented characters from each row
    def remove_accented_chars(self):
        cleansed_lines = self.apply(lambda x: unicodedata.normalize('NFKD', x))
        cleansed_lines = cleansed_lines.apply(lambda x: x.encode('ascii', 'ignore').decode('utf-8', 'ignore'))
        return cleansed_lines

    # Expands contractions in each row
    def expand_contractions(self, contraction_mapping=CONTRACTION_MAP):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        # Performs matching and construction of expansion
        def expand_match(contraction_regex):
            contraction = contraction_regex.group(0)
            # Preserves first character for casing
            first_char = contraction[0]
            # Contraction_mapping lacks first characters
            expanded_contraction = contraction_mapping.get(contraction) \
                if contraction_mapping.get(contraction) \
                else contraction_mapping.get(contraction.lower())
            # Combines first character with expansion
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        expanded_lines = self.apply(lambda x: contractions_pattern.sub(expand_match, x))
        expanded_lines = expanded_lines.apply(lambda x: re.sub("'", "", x))
        return expanded_lines

    # Removes specieal characters from each row
    def remove_special_chars(self, digits=False):
        # Creates reference pattern for what characters to keep
        pattern = r'[^a-zA-z0-9\s]' if not digits else r'[^a-zA-z\s]'
        cleansed_lines = self.apply(lambda x: re.sub(pattern, '', x))
        return cleansed_lines

    # Removes repeat characters from each row
    def remove_repeat_chars(self):
        # Creates pattern for recognizing repeated chars
        pattern = re.compile(r'(\w*)(\w)\2(\w*)')
        substitute = r'\1\2\3'

        # This should be updated for accuracy
        # Noticeable errors in 'hello' and 'ebook'...
        def replace(old_word):
            if wordnet.synsets(old_word):
                return old_word
            new_word = pattern.sub(substitute, old_word)
            return replace(new_word) if new_word != old_word else new_word

        correct_words = self.apply(lambda x: replace(x))
        return correct_words

    # Lemmatizes the text in each row. Performance optimized via collapse_rebuild
    def lemmatize_text(self):
        # The function is designed to optimize SpaCy nlp usage
        # By calling nlp for each max value of chars nlp takes

        # Uses %delim% to track rows within series
        text = ' '.join(self + '%delim%')

        # Runs nlp on max 100000 characters
        nlp_list = []

        def nlp_pipe(text):
            root_text = nlp(text)
            root_text = ' '.join([word.lemma_
                                  if word.lemma_ != '-PRON-'
                                  else word.text for word in root_text])
            nlp_list.append(root_text)

        # Depth tracks # of characters from text sent to nlp_pipe
        # Cycles mathmatically determines how many pass-throughs required
        depth = 0
        cycles = math.ceil(len(text) / 100000)

        # For each 100000 chars, sends to nlp_pipe
        for i in range(cycles):
            start = depth
            depth += 100000
            # Prevents loop from breaking a word by useing space as delimiter
            while text[start:depth][-1:] != ' ':
                depth -= 1
                if depth < start:
                    print('Something went wrong during lemmatization...')

            nlp_pipe(text[start:depth])

        # Completes remainder of text (if text did not end 0n space)
        if text[-1:] != ' ':
            nlp_pipe(text[depth:])

        root_list = ''.join(nlp_list).split('% delim%')
        return pd.Series(root_list)

    # Removes the stopwords from each row
    def remove_stopwords(self, is_lower_case=False):
        tokens_series = self.apply(lambda x: tokenizer.tokenize(x))

        stripped_tokens = []
        for tokens_list in tokens_series:
            stripped_tokens.append([token.strip() for token in tokens_list])

        # Performs filter using NLTK stopword_list for reference
        filtered_list = []
        for tokens_list in tokens_series:
            if is_lower_case:
                filtered_tokens = [token for token in tokens_list if token not in stopword_list]
            else:
                filtered_tokens = [token for token in tokens_list if token.lower() not in stopword_list]

            # Rejoins words as str due to tokenizing resulting in list of words
            filtered_list.append(' '.join(filtered_tokens))
        return pd.Series(filtered_list)

    # Performs all specified normalizations
    def normalize(self, strip_html=True, remove_accented_chars=True,
                  expand_contractions=True, remove_special_chars=True,
                  remove_repeat_chars=True, lemmatize_text=True,
                  remove_stopwords=True, remove_digits=True):

        if strip_html:
            self.series = Normalizer.strip_html(self.series)
        if remove_accented_chars:
            self.series = Normalizer.remove_accented_chars(self.series)
        if expand_contractions:
            self.series = Normalizer.expand_contractions(self.series)
        if remove_special_chars:
            self.series = Normalizer.remove_special_chars(self.series, digits=remove_digits)
        if remove_repeat_chars:
            self.series = Normalizer.remove_repeat_chars(self.series)
        if lemmatize_text:
            self.series = Normalizer.lemmatize_text(self.series)
        if remove_stopwords:
            self.series = Normalizer.remove_stopwords(self.series)

        return self.series
