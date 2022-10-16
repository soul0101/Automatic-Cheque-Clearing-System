from fileinput import close
import math
import functools
import numpy as np
from utils.validation_utils import get_alpha
from rapidfuzz.process import extractOne
from rapidfuzz.distance.Levenshtein import distance as levenshtein_distance
from rapidfuzz.fuzz import ratio

amount_dictionary = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "lakh", "lack", "lacks", "lakhs", "crore"]

class FuzzyCorrection():
    def __init__(self, dictionary=None):
        if dictionary is None:
            self.dictionary = amount_dictionary
        else:
            self.dictionary = dictionary

    def scorer(self, word):
        if(len(word) < 3):
            return -50

        if word in amount_dictionary:
            return 10*len(word) # (So that sixty > six)
        else:
            res = extractOne(word, amount_dictionary, scorer=levenshtein_distance)
            score = len(word) - res[1]
            if(score < 2):
                return 0
            else:
                return score

    def wordSeqFitness(self, words):
        return functools.reduce(lambda x,y: x+y,
            (self.scorer(w) for w in words))

    def splitPairs(self, word):
        return [(word[:i+1], word[i+1:]) for i in range(len(word))]

    @functools.cache
    def segment(self, word):

        if not word: return []
        allSegmentations = [([first] + self.segment(rest))
                            for (first, rest) in self.splitPairs(word)]

        return max(allSegmentations, key = self.wordSeqFitness)

    def get_closest_match(self, sentence):
        sentence = get_alpha(sentence)
        if len(sentence) == 0:
            return sentence
        closest_words = self.segment(sentence)
        return " ".join([extractOne(word, amount_dictionary, scorer=ratio)[0] for word in closest_words])

if __name__ == "__main__":
    corrector_instance = FuzzyCorrection()
    test = "Two crore four lakh"
    print(corrector_instance.get_closest_match(test)) 


