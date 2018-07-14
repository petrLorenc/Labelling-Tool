import numpy as np

import torch


class ModelUtils:
    """
    Interface to work with a model
    Loading, saving a manipulating with him
    """
    @staticmethod
    def load_model(model, optimizer, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'".format(path))

    @staticmethod
    def save_model(model, optimizer, path):
        print("=> saving checkpoint '{}'".format(path))
        state = {
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, path)
        print("=> saved checkpoint '{}'".format(path))


    @staticmethod
    def load_glove(path):
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
    def add_categories_for_model(names_of_categories):
        new_names_of_categories = ["0"]
        for category in names_of_categories:
            new_names_of_categories.append("B-" + category[0])
            new_names_of_categories.append("I-" + category[0])

        return new_names_of_categories
