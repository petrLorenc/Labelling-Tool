#!/usr/bin/python

import sys
import numpy as np


def load_data_and_labels(filename):
    """
    Loads data and label from a file.
    """
    sents, marks, labels = [], [], []
    with open(filename) as f:
        words, signs, tags = [], [], []
        for line in f:
            try:
                line = line.rstrip()
                if len(line) == 0 or line.startswith('-DOCSTART-'):
                    if len(words) != 0:
                        sents.append(words)
                        marks.append(signs)
                        labels.append(tags)
                        words, signs, tags = [], [], []
                else:
                    word, sign, tag = line.split('\t')
                    words.append(word)
                    signs.append(sign)
                    tags.append(tag)
            except:
                print(line)

    return np.asarray(sents), np.asarray(labels)


def update_vocabulary(path_to_vocabulary, manually_labelled, unlabeled_data):
    vocabulary = [word.strip() for word in open(path_to_vocabulary, "r").readlines()
                  if len(word) >= 1 and word.isalpha()]
    sents_old, _ = load_data_and_labels(manually_labelled)
    sents_new, _ = load_data_and_labels(unlabeled_data)

    log_counter = 0
    log_counter_added = 0

    for sent in np.concatenate([sents_old, sents_new]):
        for word in sent:
            log_counter +=1
            if word not in vocabulary and word.isalpha():
                log_counter_added += 1
                vocabulary.append(word)

            if log_counter % 10000 == 0:
                print("processed {} words, added {} new words".format(log_counter,log_counter_added))

    if log_counter_added > 0:
        with open(path_to_vocabulary, "w") as f:
            for word in vocabulary:
                f.write(word + "\n")


def main():
    assert isinstance(sys.argv[1], str)
    assert isinstance(sys.argv[2], str)
    assert isinstance(sys.argv[3], str)

    update_vocabulary(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
