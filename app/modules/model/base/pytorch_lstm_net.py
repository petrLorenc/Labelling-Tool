import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

torch.backends.cudnn.enabled = False


class PytorchLstmNetModel(nn.Module):
    '''
        Pytorch LSTM neural network (not bidirectional), last layer is projected with fully connected neural netowork
    '''

    def __init__(self, tag2idx, word2emb_idx, embedding_data, hidden_dim, use_gpu):
        super(PytorchLstmNetModel, self).__init__()

        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.tag2idx = tag2idx
        self.idx2tag = {v: k for k, v in tag2idx.items()}

        self.hidden_dim = hidden_dim
        self.word2emb_idx, self.embedding_data = word2emb_idx, embedding_data

        if self.use_gpu:
            self.embedding_data = torch.from_numpy(self.embedding_data).float().to('cuda')
            self.vocab_size = self.embedding_data.size(0)
            self.embedding_dim = self.embedding_data.size(1)

            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to('cuda')
            self.embedding.weight = nn.Parameter(self.embedding_data)
            # self.embedding.weight.requires_grad = False # to not train them afterwards

            self.lstm = nn.LSTM(self.embedding_dim, hidden_dim).to('cuda')

            self.fc = nn.Linear(hidden_dim, len(self.tag2idx)).to('cuda')

        else:
            self.embedding_data = torch.from_numpy(self.embedding_data).float()
            self.vocab_size = self.embedding_data.size(0)
            self.embedding_dim = self.embedding_data.size(1)

            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
            self.embedding.weight = nn.Parameter(self.embedding_data)
            # self.embedding.weight.requires_grad = False # to not train them afterwards

            self.lstm = nn.LSTM(self.embedding_dim, hidden_dim)

            self.fc = nn.Linear(hidden_dim, len(self.tag2idx))

        self.hidden = self.init_hidden()

    def prepare_sentence(self, sentences, batch=False):
        if batch:
            idx = [self.word2emb_idx[w] if w in self.word2emb_idx else self.word2emb_idx["UNK"] for example in sentences for w in example]
        else:
            idx = [self.word2emb_idx[w] if w in self.word2emb_idx else self.word2emb_idx["UNK"] for w in sentences]

        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def prepare_targets(self, tags, batch=False):
        if batch:
            idx = [self.tag2idx[t] for example in tags for t in example]
        else:
            idx = [self.tag2idx[t] for t in tags]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def return_class_from_target(self, target, batch=False):
        if batch:
            classes = [self.idx2tag[t] for example in target for t in example]
        else:
            classes = [self.idx2tag[t] for t in target]
        return classes

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim, device=self.device),
                torch.zeros(1, 1, self.hidden_dim, device=self.device))

    def forward(self, sentence):
        if self.use_gpu:
            embeds = self.embedding(Variable(torch.cuda.LongTensor(sentence)).to('cuda'))
        else:
            embeds = self.embedding(Variable(torch.LongTensor(sentence)))

        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)

        tag_space = self.fc(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
