from abc import ABC, abstractmethod


class AbstractModel(ABC):

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.word2idx = None
        self.idx2word = None
        self.tag2idx = None
        self.idx2tag = None
        self.max_len = None
        super().__init__()

    @abstractmethod
    def load(self, app, classes, vocabulary, zero_marker: bool) -> None:
        pass

    @abstractmethod
    def train(self, X, y, epochs=100, batch_size=32):
        pass

    @abstractmethod
    def predict(self, input):
        pass

    @abstractmethod
    def test(self, X_test, y_test):
        pass

    @abstractmethod
    def save_file(self, path):
        pass

    @abstractmethod
    def load_file(self, path):
        pass
