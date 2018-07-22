from .base.abstract_model import AbstractModel
from ..loader.load_data import LoadData
from ..loader.model_utils import ModelUtils

from pathlib import Path
import numpy as np
import sklearn.metrics
from keras.models import model_from_json, load_model

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


class KerasLstmNet(AbstractModel):
    '''
        Keras LSTM neural network, last layer is projected with fully connected neural netowork
    '''

    def load(self, app, classes, vocabulary, zero_marker):

        embedding_path = app.path_to_embeddings
        embedding_dim = app.embedding_dim
        manually_labelled = app.path_to_manually_labeled_data
        unlabeled_data = app.path_to_unlabeled_data
        use_saved_if_found = app.using_saved_model
        path_to_saved_model = app.path_to_saved_model
        hidden_dim = app.hidden_dim
        use_gpu = app.use_gpu

        sents, _, _ = LoadData.load_data_and_labels(unlabeled_data)
        X, _, y = LoadData.load_data_and_labels(manually_labelled)

        words = list(set([word.strip() for words in sents for word in words] +
                         [word.strip() for words in X for word in words]))
        words.append("ENDPAD")

        embedding_index = ModelUtils.load_glove_keras(embedding_path)

        embedding_matrix = np.zeros((len(words) + 1, embedding_dim))
        for i, word in enumerate(words):
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # mapping for tags
        tag_to_class = {}
        index = 0
        for category in ModelUtils.add_categories_for_model(classes):
            if category not in tag_to_class:
                tag_to_class[category] = index
                index += 1

        self.max_len = 20
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.tag2idx = tag_to_class
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        input = Input(shape=(self.max_len,))

        model = Embedding(input_dim=len(self.word2idx) + 1, output_dim=embedding_dim, weights=[embedding_matrix],
                          input_length=self.max_len, mask_zero=True, trainable=False)(input)

        model = Bidirectional(LSTM(units=hidden_dim, return_sequences=True,
                                   recurrent_dropout=0.1))(model)  # variational biLSTM
        model = TimeDistributed(Dense(hidden_dim, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        model = Dense(hidden_dim, activation="relu")(model)
        out = Dense(len(self.tag2idx), activation="softmax")(model)

        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        if use_saved_if_found:
            my_file = Path(path_to_saved_model)
            if my_file.is_file():
                self.load_file(path_to_saved_model)

        self.model = model

        X = [[self.word2idx[w.strip()] for w in s] for s in X]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.word2idx["ENDPAD"])

        y = [[self.tag2idx[w] for w in s] for s in y]
        y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=self.tag2idx["0"])
        y = [to_categorical(i, num_classes=len(self.tag2idx)) for i in y]

        self.model.fit(X, np.array(y), batch_size=32, epochs=3, validation_split=0.1, verbose=1)

    def train(self, X, y, epochs=10, batch_size=32):
        X = [[self.word2idx[w.strip()] for w in s] for s in X]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.word2idx["ENDPAD"])

        y = [[self.tag2idx[w] for w in s] for s in y]
        y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=self.tag2idx["0"])
        y = [to_categorical(i, num_classes=len(self.tag2idx)) for i in y]

        history = self.model.fit(X, np.array(y), batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)
        return list(range(epochs)), history.history["loss"], history.history['acc'], history.history['val_acc']

    def predict(self, input):
        X = [[self.word2idx[w[0].strip()] for w in input]]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.word2idx["ENDPAD"])
        y_pred = [self.idx2tag[np.argmax(tag)] for pred in self.model.predict(X) for tag in pred]

        new_sentence = []
        for i, data in enumerate(input):
            new_sentence.append([data[0], "1" if y_pred[i] != "0" else "0", y_pred[i]])
        return new_sentence

    def test(self, X_test, y_test):
        X = [[self.word2idx[w[0].strip()] for w in X_test]]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.word2idx["ENDPAD"])

        p = self.model.predict(X)
        y_pred = np.argmax(p, axis=-1)
        idx2tag = {v: k for k, v in self.tag2idx.ietms()}

        y_t = [idx2tag[np.argmax(tag)] for sentence in y_test for tag in sentence]
        y_p = [idx2tag[tag] for pred in y_pred for tag in pred]

        raw_report = sklearn.metrics.classification_report(y_pred=y_p, y_true=y_t)

        report_data = []
        lines = raw_report.split('\n')
        for line in lines[2:-3]:
            row = {}
            print(line)
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
        self.model.save(path)
        # just because of curiosity
        with open(path + '_model_architecture.json', 'w') as f:
            f.write(self.model.to_json())

    def load_file(self, path):
        self.model = load_model(path)