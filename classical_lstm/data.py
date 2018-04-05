import sys
import os
import json
import numpy as np
import json
import torch
import random
from torch.autograd import Variable

note_num = 30
# note_range = 95
note_range = 95 + 2 #added 2 for velocity and durations
velocity = note_range - 2 #might be off by one errors
duration = note_range - 1

def get_notes(midi_file):
    with open(midi_file, 'r') as f:
        midi = json.load(f)

    times = {}
    for track in midi['tracks']:
        if len(track['notes']) > 1:
            notes = track['notes']
            for note in notes:
                if note['time'] not in note:
                    times[note['time']] = torch.zeros(note_range)
                times[note['time']][int(note['midi'])-24] += 1
                times[note['time']][velocity] = note['velocity']
                times[note['time']][duration] = note['duration']

    keylist = times.keys()
    keylist.sort()
    # tensor = torch.zeros(len(keylist), 1, note_range)
    # for ki, key in enumerate(keylist):
    #     tensor[ki][0] += times[key]
    tensor = torch.zeros(note_num, 1, note_range)
    for ni in range(note_num):
        tensor[ni][0] += times[keylist[ni]]
    return tensor

def get_hotones(folder):
    # print folder
    songs = []
    for file in os.listdir(folder):
        if '.json' in file:
            songs.append( get_notes(os.path.join(folder,file)) )
    return songs

def randomTrainingPair(genres, all_genres):
    genre = randomChoice(all_genres)
    song = randomChoice(genres[genre])
    genre_tensor = Variable(torch.LongTensor([all_genres.index(genre)]))
    song_tensor = Variable(song)
    return genre, song, genre_tensor, song_tensor

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def get_data(base_dir='../data'):
    genres = {}
    all_genres = []
    for folder in os.listdir(base_dir):
        all_genres.append(folder)
        genres[folder] = get_hotones(os.path.join(base_dir, folder))
    # with open('all_genres.json', 'w') as f:
    #     json.dump(all_genres, f)
    print ('got data')
    return genres, all_genres,

def get_genres(filename='all_genres.json'):
    with open(filename, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    # midi_dir = sys.argv[1]
    # genres = {}
    # all_genres = []
    # for folder in os.listdir(midi_dir):
    #     all_genres.append(folder)
    #     genres[folder] = get_hotones(os.path.join(midi_dir, folder))
    # genre, song, genre_tensor, song_tensor = randomTrainingPair()
    # print (genre)

    get_data()
