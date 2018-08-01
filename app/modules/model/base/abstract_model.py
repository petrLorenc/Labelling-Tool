from abc import ABC, abstractmethod


class AbstractModel(ABC):

    def __init__(self):
        self.model = None

        self.word2idx = None
        self.idx2word = None
        self.tag2idx = None
        self.idx2tag = None

        self.max_len = None
        self.vocabulary = None
        self.categories = None

        self.load_workflow = []
        self.load_param = []
        super().__init__()

    def load(self):
        for method, param in zip(self.load_workflow, self.load_param):
            print(method)
            if method is not None and callable(method):
                method(*param)

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
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    def load_vocabulary(self, path_to_vocabulary):
        self.vocabulary = [word.strip() for word in open(path_to_vocabulary, "r").readlines() if len(word) >= 1]

    def load_categories(self, path_to_categories):
        self.categories = [category.split("\t") for category in open(path_to_categories, "r").readlines() if len(category) > 3]
