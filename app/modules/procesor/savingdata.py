class SavingData(object):
    def __init__(self, path):
        self.path = path

    def add_item_to_file(self, sentence):
        with open(self.path, "a") as f:
            for word in sentence:
                f.write(word[0] + "\t" + word[1] + "\t" + word[2] + "\n")
            f.write("\n")

