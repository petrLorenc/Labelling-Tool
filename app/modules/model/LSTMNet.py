import torch
from torch import nn, Tensor
from torch.autograd import Variable
import torch.nn.functional as F


class LSTMnet(nn.Module):
    '''
        LSTM neural network (not bidirectional), last layer is projected with fully connected neural netowork
    '''

    def __init__(self, tag_to_class, mapping, embedding_data, hidden_dim):
        super(LSTMnet, self).__init__()
        self.tag_to_class = tag_to_class
        self.hidden_dim = hidden_dim
        self.mapping, self.embedding_data = mapping, embedding_data

        self.embedding_data = torch.from_numpy(self.embedding_data).float()
        self.vocab_size = self.embedding_data.size(0)
        self.embedding_dim = self.embedding_data.size(1)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(self.embedding_data)
        # self.embedding.weight.requires_grad = False # to not train them afterwards

        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, len(self.tag_to_class))
        self.hidden = self.init_hidden()

    def prepare_sentence(self, sentences, batch=False):
        if batch:
            idx = [self.mapping[w] if w in self.mapping else self.mapping["UNK"] for example in sentences for w in example]
        else:
            idx = [self.mapping[w] if w in self.mapping else self.mapping["UNK"] for w in sentences]

        return torch.tensor(idx, dtype=torch.long)

    def prepare_targets(self, tags, batch=False):
        if batch:
            idx = [self.tag_to_class[t] for example in tags for t in example]
        else:
            idx = [self.tag_to_class[t] for t in tags]
        return torch.tensor(idx, dtype=torch.long)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.embedding(Variable(torch.LongTensor(sentence)))

        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)

        tag_space = self.fc(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
