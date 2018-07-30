import numpy as np

import torch


class ModelUtils:
    """
    Group of function to help work with embeddings and models.
    """

    @staticmethod
    def load_glove_mapping(path):
        """
        creates a dictionary mapping words to vectors from a file in glove format.
        """
        with open(path, ) as f:
            all_words = f.readlines()

            glove_embeddings = []
            mapping = {}

            for index, line in enumerate(all_words):
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')

                glove_embeddings.append(vector)
                mapping[word] = index

            return mapping, glove_embeddings

    @staticmethod
    def load_glove(path):
        """
        creates a dictionary of words with theirs vectors from a file in glove format.
        """
        embeddings_index = {}

        with open(path, ) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        return embeddings_index

    @staticmethod
    def add_categories_for_model(names_of_categories):
        """
        For returning corresponding tags for categories
        :param names_of_categories: List of categories
        :return: New list of categories with B/I prefix
        """
        new_names_of_categories = ["0"]
        for category in names_of_categories:
            new_names_of_categories.append("B-" + category[0])
            new_names_of_categories.append("I-" + category[0])

        return new_names_of_categories
