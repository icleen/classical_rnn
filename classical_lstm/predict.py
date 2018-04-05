from model import *
from data import *
from evaluate import evaluate
import sys


def predict(track, n_predictions=2):
    output = evaluate(Variable(track))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        genre_index = topi[0][i]
        print('(%.2f) %s' % (value, all_genres[genre_index]))
        predictions.append([value, all_genres[genre_index]])

    return predictions

if __name__ == '__main__':
    if len(sys.argv) > 2:
        rnn = torch.load(sys.argv[2])
    else:
        rnn = torch.load('classical_music_classification.pt')

    tensor = get_notes(sys.argv[1])
    predict(tensor, rnn)
