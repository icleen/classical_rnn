from model import *
from data import *
import sys
import matplotlib.pyplot as plt
import numpy as np

# Just return an output given a line
def evaluate(track_tensor, rnn):
    hidden = rnn.initHidden()

    for i in range(track_tensor.size()[0]):
        output, hidden = rnn(track_tensor[i], hidden)

    return output

def eval_set(weight_file, test_dir=None):
    rnn = torch.load(weight_file)
    if test_dir is not None:
        genres, all_genres = get_data(test_dir)
    else:
        genres, all_genres = get_data()
    print all_genres

    confusion = np.zeros((len(all_genres), len(all_genres)))
    correct = 0
    total_correct = 0
    count = 0
    for g, gen in enumerate(all_genres):
        count += len(genres[gen])
        for tensor in genres[gen]:
            output = evaluate(Variable(tensor), rnn)
            topv, topi = output.data.topk(1, 1, True)
            confusion[g, topi[0][0]] += 1
            if gen == all_genres[topi[0][0]]:
                correct += 1
                total_correct += 1
        if len(genres[gen]) > 0:
            print('{} Accuracy: {}'.format(gen, float(correct) / len(genres[gen])))
            correct = 0
            confusion[g] /= np.sum(confusion[g])

    print('Total Accuracy: {}'.format(float(total_correct) / count))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(all_genres))
    plt.xticks(tick_marks, all_genres, rotation=45)
    plt.yticks(tick_marks, all_genres)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    graphfile = weight_file.split('/')[-1]
    graphfile = 'graphs/confusion_' + graphfile.split('.')[0] + '.png'
    plt.savefig(graphfile)
    print('Saved Figure: {}'.format(graphfile))

def eval_folder(weight_file, test_dir):
    all_genres = get_genres()
    print all_genres
    rnn = torch.load(weight_file)
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

if __name__ == '__main__':
    if len(sys.argv) > 2:
        eval_set(sys.argv[1], sys.argv[2])
    else:
        eval_set(sys.argv[1])
