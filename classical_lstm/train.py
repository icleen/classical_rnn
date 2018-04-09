from __future__ import unicode_literals, print_function, division
import torch
from model import *
import time
import math
import sys
from data import * # get_hotones(folder), get_notes(file)
import matplotlib.pyplot as plt


def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_genres[category_i], category_i

def train(genre_tensor, song_tensor):
    rnn.zero_grad()
    rnn.hidden = rnn.init_hidden()

    for i in range(song_tensor.size()[0]):
        output = rnn(song_tensor[i])

    loss = loss_function(output, genre_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == '__main__':
    genres, all_genres = get_data()

    n_epochs = 10000 # 10,000
    learning_rate = 0.001 # If you set this too high, it might explode. If too low, it might not learn
    n_hidden = 128
    savefile = 'saves/classical_lstm_ten_lr001_h128.pt'
    if len(sys.argv) > 1 and sys.argv[1] == '-R':
        rnn = torch.load(sys.argv[2])
        savefile = sys.argv[3]
        if len(sys.argv) > 4:
            n_epochs = int(sys.argv[4])
        if len(sys.argv) > 5:
            learning_rate = float(sys.argv[5])
    else:
        if len(sys.argv) > 1:
            n_epochs = int(sys.argv[1])
        if len(sys.argv) > 2:
            learning_rate = float(sys.argv[2])
        if len(sys.argv) > 3:
            savefile = sys.argv[3]
        if len(sys.argv) > 4:
            n_hidden = int(sys.argv[4])
        rnn = LSTMclassifier(note_range, n_hidden, len(all_genres), 3)

    graphfile = savefile.split('/')[-1]
    graphfile = 'graphs/' + graphfile.split('.')[0] + '.png'

    print_every = n_epochs / 100

    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()

    # Keep track of losses for plotting
    cum_loss = 0.0
    all_losses = []

    start = time.time()

    for epoch in range(1, n_epochs + 1):
        genre, song, genre_tensor, song_tensor = randomTrainingPair(genres, all_genres)
        output, loss = train(genre_tensor, song_tensor)
        cum_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = 'c' if guess == genre else 'f (%s)' % genre
            print('%d %d%% (%s) %.4f / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, guess, correct))
            print('Average Loss: %f' % (cum_loss / print_every))
            # Add current loss avg to list of losses
            all_losses.append(cum_loss / print_every)
            cum_loss = 0.0

    torch.save(rnn, savefile)
    print (savefile)
    plt.plot(all_losses)
    plt.savefig(graphfile)
    print (graphfile)
    plt.show()
