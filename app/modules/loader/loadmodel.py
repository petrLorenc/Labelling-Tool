from ..model.LSTMNet import LSTMnet

from pathlib import Path
import numpy as np
import sklearn.metrics

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


class ModelInterface:
    """
    Interface to work with a model
    Loading, saving a manipulating with him
    """

    def __init__(self):
        pass

    @staticmethod
    def load_model(model, optimizer, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'".format(path))
        return model, optimizer

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
    def create_model(embedding_path,
                     vocabulary, classes,
                     use_saved_if_found, path_to_saved_model,
                     hidden_dim, embedding_dim=50):
        '''
        Create model and according the tag load saved one
        :param embedding_path:
        :param vocabulary: For adding OOV words
        :param classes:
        :param use_saved_if_found:
        :param path_to_saved_model:
        :param hidden_dim:
        :return:
        '''

        mapping, embedding_data = ModelInterface.load_glove(embedding_path)

        # addind OOV words with random embeddings (maybe zeros would be more convenient)
        for word in vocabulary:
            if word not in mapping:
                # print (word)
                mapping[word] = len(mapping)  # last index
                embedding_data.append(np.zeros(embedding_dim))

        # mapping for tags
        tag_to_class = {}
        index = 0
        for category in ModelInterface.add_categories_for_model(classes):
            if category not in tag_to_class:
                tag_to_class[category] = index
                index += 1

        model = LSTMnet(tag_to_class, mapping, np.array(embedding_data), hidden_dim=hidden_dim)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        if use_saved_if_found:
            my_file = Path("./app/static/data/models/actual_model.pth.tar")
            if my_file.is_file():
                model, optimizer = ModelInterface.load_model(model, optimizer, path_to_saved_model)

        return model, loss_function, optimizer

    @staticmethod
    def train_model(model, loss_function, optimizer, X_train, y_train, epochs=1, batch_size=32):
        losses = []
        accumulated_loss = 0

        for epoch in range(epochs):
            for i in range(0, X_train.shape[0], batch_size):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                model.hidden = model.init_hidden()

                sentences_in = model.prepare_sentence(X_train[i:i + batch_size], batch=True)
                labels = model.prepare_targets(y_train[i:i + batch_size], batch=True)

                # Step 3. Run our forward pass.
                tag_scores = model(sentences_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, labels)
                accumulated_loss += int(loss)

                loss.backward()
                optimizer.step()

            losses.append(accumulated_loss / float(X_train.shape[0]))
            accumulated_loss = 0
            print ("Epoch end")

        return accumulated_loss

    @staticmethod
    def get_confidence(output_of_model):

        def multiply(numbers):
            total = 1
            for x in numbers:
                total *= x
            return total

        x = [np.max(x.numpy()) for x in F.softmax(output_of_model, dim=1)]
        return np.power(multiply(x),
                        1 / len(output_of_model.numpy()))  # nth root of multiplied probabilities -> NORMALIZATION

    @staticmethod
    def get_indexes_less_confident(model, test_data):
        confidences = []
        for index in range(len(test_data)):
            with torch.no_grad():
                sentence_input = [x[0] for x in test_data[index]]
                print (sentence_input)
                inputs = model.prepare_sentence(sentence_input)
                print (inputs)
                tag_scores = model(inputs)
                conf = ModelInterface.get_confidence(tag_scores)
                confidences.append((index, conf))
        return sorted(confidences, key=lambda elem: elem[1], reverse=True)

    @staticmethod
    def load_glove(path, embedding_dim=50):
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

    @staticmethod
    def test_model(model, X_test, y_test):
        with torch.no_grad():
            inputs = model.prepare_sentence(X_test, batch=True)
            tag_scores = model(inputs)
            tag_scores = [np.argmax(x.numpy()) for x in tag_scores]
            report = sklearn.metrics.classification_report(model.prepare_targets(y_test, batch=True), tag_scores)
            report = str([x.split() for x in report.split("\n")])
            # return sklearn.metrics.accuracy_score(model.prepare_targets(y_test, batch=True), tag_scores)
            return report
