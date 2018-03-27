import sys
import os
import numpy as np
import json
import torch
import random
from torch.autograd import Variable

note_num = 30
note_range = 95

def get_notes(midi_file):
    with open(midi_file, 'r') as f:
        midi = json.load(f)
    notes = midi['tracks'][1]['notes']
    mnotes = np.asarray([int(note['midi']) for note in notes])
    mnotes -= 24
    tensor = torch.zeros(note_num, 1, note_range)
    for ni in range(note_num):
        tensor[ni][0][mnotes[ni]] = 1
    return tensor

def get_hotones(folder):
    artists = {}
    for file in os.listdir(folder):
        if '.js' in file:
            artist = file.split('_')[0]
            if artist not in artists:
                artists[artist] = []

            track = get_notes(os.path.join(folder,file))
            artists[artist].append(track)

    return artists

def randomTrainingPair():
    artist = randomChoice(all_artists)
    track = randomChoice(artist_tracks[artist])
    artist_tensor = Variable(torch.LongTensor([all_artists.index(artist)]))
    track_tensor = Variable(track)
    return artist, track, artist_tensor, track_tensor

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

artist_tracks = get_hotones('../midi_data/train')
all_artists = [key for key in artist_tracks]

if __name__ == "__main__":
    # midi_folder = sys.argv[1]
    # artist_tracks = get_hotones('midi_data')
    # all_artists = [key for key in artist_tracks]
    artist, track, artist_tensor, track_tensor = randomTrainingPair()
