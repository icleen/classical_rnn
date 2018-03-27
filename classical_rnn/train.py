from __future__ import unicode_literals, print_function, division
import torch
from model import *
import time
import math
from data import * # get_hotones(folder), get_notes(file)

n_hidden = 128
n_epochs = 500
print_every = n_epochs / 10
plot_every = n_epochs / 5
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_artists[category_i], category_i

rnn = RNN(note_range, n_hidden, len(all_artists))
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(artist_tensor, track_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(track_tensor.size()[0]):
        output, hidden = rnn(track_tensor[i], hidden)

    loss = criterion(output, artist_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    artist, track, artist_tensor, track_tensor = randomTrainingPair()
    output, loss = train(artist_tensor, track_tensor)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = 'c' if guess == artist else 'f (%s)' % artist
        print('%d %d%% (%s) %.4f / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'classical_music_classification.pt')
