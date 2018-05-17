import numpy as np


class LoadData:

    def __init__(self, path, path_to_marker="./app/static/data/marker.txt"):
        # path to marker/number of already processed sentences
        # need to be in separate file because there is possibility to label one sentence several times (in a case of
        # not clear definition), so the number cannot be deduce afterwards
        self.path_to_marker = path_to_marker
        # number of already processed sentences
        self.marker = 0

        self.data = self.load_without_labels(path)

    def load_without_labels(self, path):
        """
        Loaded data are saved into this object. Sentences are divided by the keyword END.
        Example input file (newline around each example, tab separated):

        ...

        how	0	0
        you	0	0
        doing	0	0
        END	0	0

        ...

        Marker file store value of already labelled sentences

        :param path:
        :return:
        """
        with open(path) as f:
            data = "".join(f.readlines()).split("\n\n")

        with open(self.path_to_marker, "r") as f:
            self.marker = int(f.readline())

        data = [word.split("\t") for line in data[self.marker:] for word in line.split("\n")]
        processed_data = []

        sentence = []
        for word in data:
            if word[0] != "END":
                sentence.append(word)
            else:
                processed_data.append(sentence)
                sentence = []

        return processed_data

    def generate_data(self):
        for sentence in self.data:
            self.marker = self.marker + 1
            with open(self.path_to_marker, "w") as f:
                f.write(str(self.marker))
            yield sentence

    @staticmethod
    def load_data_and_labels(filename):
        """Loads data and label from a file.
        Args:
            filename (str): path to the file.
            The file format is tab-separated values.
            A blank line is required at the end of a sentence.
            For example:
            ```
            EU	B-ORG
            rejects	O
            German	B-MISC
            call	O
            to	O
            boycott	O
            British	B-MISC
            lamb	O
            .	O
            Peter	B-PER
            Blackburn	I-PER
            ...
            ```
        Returns:
            tuple(numpy array, numpy array): data and labels.
        Example:
            >>> filename = 'conll2003/en/ner/train.txt'
            >>> data, labels = load_data_and_labels(filename)
        """
        sents, marks, labels = [], [], []
        with open(filename) as f:
            words, signs, tags = [], [], []
            for line in f:
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

        return np.asarray(sents), np.asanyarray(signs), np.asarray(labels)
