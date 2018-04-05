import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class LSTMclassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMclassifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))

    def forward(self, inputs):
        lstm_out, self.hidden = self.lstm(inputs.view(1, 1, -1), self.hidden)
        output = self.hidden2tag(lstm_out.view(1, -1))
        output = F.log_softmax(output, dim=1)
        return output

    # def train(label_tensor, input_tensor):
    #     rnn.zero_grad()
    #     rnn.hidden = self.init_hidden()
    #
    #     for i in range(input_tensor.size()[0]):
    #         output = self.forward(input_tensor[i])
    #
    #     print (output)
    #     loss = loss_function(output, label_tensor.view(1, 1, -1))
    #     loss.backward()
    #
    #     optimizer.step()
    #
    #     return output, loss.data[0]
