from model import *
from data import *
import sys

# Just return an output given a line
def evaluate(track_tensor, rnn):
    rnn.hidden = rnn.init_hidden()

    for i in range(track_tensor.size()[0]):
        output = rnn(track_tensor[i])

    return output

if __name__ == '__main__':
    all_genres = get_genres()
    print all_genres
    rnn = torch.load(sys.argv[1])
    test_dir = sys.argv[2]
    tensors = get_hotones(test_dir)
    correct_genre = test_dir.strip('/').split('/')[-1]
    print('correct_genre: ' + correct_genre)
    correct = 0
    for tensor in tensors:
        output = evaluate(Variable(tensor), rnn)
        topv, topi = output.data.topk(1, 1, True)
        if correct_genre == all_genres[topi[0][0]]:
            correct += 1

    print('Accuracy: {}'.format(float(float(correct) / len(tensors))))
