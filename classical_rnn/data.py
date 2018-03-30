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


def combine_tracks(filename):
    file = open(filename, 'r')

    data = file.read()
    name = file.name
    newName = name[:name.rfind('.')] + "_combined" + name[name.rfind('.'):]
    file.close()

    jsonData = json.loads(data)

    # combines all tracks from entire json
    tracks = jsonData['tracks']
    combined = tracks[0]
    for track in tracks[1:]:
        if track['duration'] > combined['duration']:
            combined['duration'] = track['duration']
        if track['id'] > combined['id']:
            combined['id'] = track['id'] + 1
        for note in track['notes']:
            combined['notes'].append(note)
        combined['length'] = len(combined['notes'])

    print (combined)

    # combined is now a single track containing all notes.
    # This should work with fugue tracks too.
    # This will probably break if we try to include instrumentation later
    # jsonWrite = json.dumps(combined, indent=4)
    # newFile = open(newName, 'w')
    # newFile.write(jsonWrite)
    # newFile.close()


artist_tracks = get_hotones('../BaroqueJSON')
all_artists = [key for key in artist_tracks]

if __name__ == "__main__":
    # midi_folder = sys.argv[1]
    # artist_tracks = get_hotones('midi_data')
    # all_artists = [key for key in artist_tracks]
    # artist, track, artist_tensor, track_tensor = randomTrainingPair()
    for file in os.listdir('../BaroqueJSON'):
        if '.js' in file:
            combine_tracks(os.path.join('../BaroqueJSON', file))
