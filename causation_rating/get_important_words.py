import nltk
from word_forms.word_forms import get_word_forms
import pandas as pd


nltk.download('wordnet')
# First, we will import a list of ``important words'' from manually annotated data and get a list of all possible forms of these words.

# Expand your list of important words
def get_direct_derivations(word: str) -> set:
    if not isinstance(word, str):
        return set()  # Return empty set if input is not a string

    word = word.lower()  # Ensure lowercase
    forms = get_word_forms(word) # get a dictionary of forms
    derivations = set()
    for key in forms:
        derivations.update(forms[key])
    return derivations

def get_important_words(path: str) -> list:
    # Base important words
    important_base_words = pd.read_csv(path)['complete_unique'].tolist()
    # Expand each word using relevant forms
    expanded_important_words = set()
    for word in important_base_words:
        expanded_important_words.update(get_direct_derivations(word))

    expanded_important_words = list(expanded_important_words)
    return expanded_important_words
