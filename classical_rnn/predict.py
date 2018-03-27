from model import *
from data import *
import sys

rnn = torch.load('classical_music_classification.pt')

# Just return an output given a line
def evaluate(track_tensor):
    hidden = rnn.initHidden()

    for i in range(track_tensor.size()[0]):
        output, hidden = rnn(track_tensor[i], hidden)

    return output

def predict(track, n_predictions=2):
    output = evaluate(Variable(track))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        artist_index = topi[0][i]
        print('(%.2f) %s' % (value, all_artists[artist_index]))
        predictions.append([value, all_artists[artist_index]])

    return predictions

if __name__ == '__main__':
    tensor = get_notes(sys.argv[1])
    predict(tensor)
