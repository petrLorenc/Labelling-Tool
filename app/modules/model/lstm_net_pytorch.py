from .base.abstract_model import AbstractModel
from ..loader.model_utils import ModelUtils
from .base.pytorch_lstm_net import PytorchLstmNetModel

import numpy as np
import sklearn.metrics
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class PytorchLstmNet(AbstractModel):

    def load(self, app, classes, vocabulary, zero_marker):

        embedding_path = app.path_to_embeddings
        embedding_dim = app.embedding_dim
        use_saved_if_found = app.using_saved_model
        path_to_saved_model = app.path_to_saved_model
        hidden_dim = app.hidden_dim
        use_gpu = app.use_gpu

        mapping, embedding_data = ModelUtils.load_glove_mapping(embedding_path)

        # addind OOV words with random embeddings (maybe zeros would be more convenient)
        for word in vocabulary:
            if word not in mapping:
                mapping[word] = len(mapping)  # last index
                embedding_data.append(np.zeros(embedding_dim))

        # mapping for tags
        tag_to_class = {}
        index = 0
        for category in ModelUtils.add_categories_for_model(classes):
            if category not in tag_to_class:
                tag_to_class[category] = index
                index += 1

        model = PytorchLstmNetModel(tag_to_class, mapping, np.array(embedding_data), hidden_dim=hidden_dim, use_gpu=use_gpu)
        # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
        # It is useful when training a classification problem with C classes.
        # If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes.
        # This is particularly useful when you have an unbalanced training set.
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        if use_saved_if_found:
            my_file = Path(path_to_saved_model)
            if my_file.is_file():
                self.load_file(path_to_saved_model)

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self, X, y, epochs=100, batch_size=32):
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1)
        losses = []
        key_errors = 0
        for epoch in range(epochs):
            accumulated_loss_train = 0.0
            correct = 0
            total = 0
            for i in range(0, X_train.shape[0], batch_size):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.model.hidden = self.model.init_hidden()

                sentences_in = self.model.prepare_sentence(X_train[i:i + batch_size], batch=True)
                labels = self.model.prepare_targets(y_train[i:i + batch_size], batch=True)

                # Step 3. Run our forward pass.
                tag_scores = self.model(sentences_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = self.loss_function(tag_scores, labels)
                accumulated_loss_train += float(self.loss_function(tag_scores, labels))

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    outputs = self.model(sentences_in)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accumulated_loss_train /= X_train.shape[0]

            train_accuracy = 100 * (correct / float(total))

            correct = 0
            total = 0
            sentences_in = self.model.prepare_sentence(X_test, batch=True)
            labels = self.model.prepare_targets(y_test, batch=True)
            with torch.no_grad():
                outputs = self.model(sentences_in)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on {} examples in the testset : {} %%'.format(len(X_test),
                                                                                         100 * correct / total))

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

    def predict(self, X):
        with torch.no_grad():
            sentence_input = [x[0] for x in X]
            inputs = self.model.prepare_sentence(sentence_input)
            outputs = self.model(inputs)

            prediction_of_model = F.softmax(outputs, dim=1)
            prediction = [np.argmax(x.cpu().numpy()) for x in prediction_of_model]

            prediction = self.model.return_class_from_target(prediction)

        new_sentence = []
        for i, data in enumerate(X):
            new_sentence.append([data[0], "1" if prediction[i] != "0" else "0", prediction[i]])
        return new_sentence

    def test(self, X_test, y_test):
        with torch.no_grad():
            inputs = self.model.prepare_sentence(X_test, batch=True)
            tag_scores = self.model(inputs)
            tag_scores = [np.argmax(x.cpu().numpy()) for x in tag_scores]
            raw_report = sklearn.metrics.classification_report(self.model.prepare_targets(y_test, batch=True), tag_scores)

            report_data = []
            lines = raw_report.split('\n')
            for line in lines[2:-3]:
                row = {}
                print (line)
                row_data = line.split('      ')
                if len(row_data) > 0:
                    row['class'] = self.model.return_class_from_target([int(row_data[1])])[0]
                    row['precision'] = float(row_data[2])
                    row['recall'] = float(row_data[3])
                    row['f1_score'] = float(row_data[4])
                    row['support'] = float(row_data[5])
                    report_data.append(row)

            return report_data

    def save_file(self, path):
        print("=> saving checkpoint '{}'".format(path))
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, path)
        print("=> saved checkpoint '{}'".format(path))

    def load_file(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'".format(path))