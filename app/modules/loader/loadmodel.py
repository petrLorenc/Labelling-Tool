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
    def create_model(embedding_path,
                     vocabulary, classes,
                     use_saved_if_found, path_to_saved_model,
                     hidden_dim, embedding_dim=50):
        '''
        Create model and according the tag load saved one
        :param embedding_dim:
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
        # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
        # It is useful when training a classification problem with C classes.
        # If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes.
        # This is particularly useful when you have an unbalanced training set.
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        if use_saved_if_found:
            my_file = Path(path_to_saved_model)
            if my_file.is_file():
                ModelInterface.load_model(model, optimizer, path_to_saved_model)

        return model, loss_function, optimizer

    @staticmethod
    def train_model(model, loss_function, optimizer, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        losses = []
        key_errors = 0

        for epoch in range(epochs):
            accumulated_loss_train = 0.0
            correct = 0
            total = 0
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
                accumulated_loss_train += float(loss_function(tag_scores, labels))

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    outputs = model(sentences_in)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accumulated_loss_train /= X_train.shape[0]

            train_accuracy = 100 * (correct / float(total))

            correct = 0
            total = 0
            sentences_in = model.prepare_sentence(X_test, batch=True)
            labels = model.prepare_targets(y_test, batch=True)
            with torch.no_grad():
                outputs = model(sentences_in)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on {} examples in the testset : {} %%'.format(len(X_test),100 * correct / total))

            test_accuracy = 100 * (correct / float(total))

            losses.append([epoch, accumulated_loss_train, train_accuracy, test_accuracy])
            print("Epoch {} end".format(epoch))

        print("There was {} errors".format(key_errors))
        print(losses)

        epochs = [x[0] for x in losses]
        train_loss = [x[1] for x in losses]
        train_acc = [x[2] for x in losses]
        test_acc = [x[3] for x in losses]

        return epochs, train_loss, train_acc, test_acc

    @staticmethod
    def get_confidence(output_of_model):

        def multiply(numbers):
            total = 1
            for x in numbers:
                total *= x
            return total

        prediction_of_model = F.softmax(output_of_model, dim=1)
        hard_prediction = [np.argmax(x.numpy()) for x in prediction_of_model]
        prediction = [np.max(x.numpy()) for x in prediction_of_model]
        return hard_prediction, np.power(multiply(prediction),
                                         1.0 / len(prediction))  # nth root of multiplied probabilities -> NORMALIZATION

    @staticmethod
    def get_indexes_less_confident(model, test_data):
        confidences = []
        for index in range(len(test_data)):
            with torch.no_grad():
                sentence_input = [x[0] for x in test_data[index]]
                inputs = model.prepare_sentence(sentence_input)
                tag_scores = model(inputs)
                prediction, conf = ModelInterface.get_confidence(tag_scores)
                confidences.append((index, conf, model.return_class_from_target(prediction)))
        return sorted(confidences, key=lambda elem: elem[1], reverse=True)

    @staticmethod
    def get_sentence_based_on_model(sentence, sorted_examples):
        new_sentence = []
        for i, data in enumerate(sentence): #word, tag, label
            new_sentence.append([data[0], "1" if sorted_examples[2][i] != "0" else "0", sorted_examples[2][i]])
        return new_sentence


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

    @staticmethod
    def test_model(model, X_test, y_test):
        with torch.no_grad():
            inputs = model.prepare_sentence(X_test, batch=True)
            tag_scores = model(inputs)
            tag_scores = [np.argmax(x.numpy()) for x in tag_scores]
            raw_report = sklearn.metrics.classification_report(model.prepare_targets(y_test, batch=True), tag_scores)

            report_data = []
            lines = raw_report.split('\n')
            for line in lines[2:-3]:
                row = {}
                print (line)
                row_data = line.split('      ')
                if len(row_data) > 0:
                    row['class'] = model.return_class_from_target([int(row_data[1])])[0]
                    row['precision'] = float(row_data[2])
                    row['recall'] = float(row_data[3])
                    row['f1_score'] = float(row_data[4])
                    row['support'] = float(row_data[5])
                    report_data.append(row)

            return report_data
