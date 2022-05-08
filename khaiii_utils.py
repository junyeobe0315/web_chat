from khaiii import KhaiiiApi
from pykospacing import Spacing
from hanspell import spell_checker
import numpy as np


def make_no_space(sentence):
    no_space_sentence = sentence.replace(" ", "")
    return no_space_sentence


def spacing(no_space_sentence):
    spacing = Spacing()
    space_sentence = spacing(no_space_sentence)
    return space_sentence


def spell_check(space_sentence):
    spelled_sentence = spell_checker.check(space_sentence)
    spell_sentence = spelled_sentence.checked
    return spell_sentence


def tokenize(spell_sentence):
    api = KhaiiiApi()
    tokenized_sentence = []
    for word in api.analyze(spell_sentence):
        for morph in word.morphs:
            tokenized_sentence.append(morph)
    return tokenized_sentence


def pos_tagging(spell_sentence):
    api = KhaiiiApi()
    token_list = []
    for word in api.analyze(spell_sentence):
        for morph in word.morphs:
            token_list.append((morph.lex, morph.tag))
    return token_list


def bag_of_words(tokenized_sentence, words):
    sentence_words = [word for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, word in enumerate(words):
        if word in sentence_words:
            bag[idx] = 1
    return bag
