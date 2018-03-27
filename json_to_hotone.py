import sys
import os
import numpy as np
import json
# import random
# import argparse
# import time

note_num = 30
note_range = 95

def get_notes(midi_file):
    with open(midi_file, 'r') as f:
        midi = json.load(f)
    notes = midi['tracks'][1]['notes']
    # print( 'number of notes: {}'.format(len(notes)) )
    mnotes = np.asarray([int(note['midi']) for note in notes])
    # print( 'mean: {}, min: {}, max: {}'.format(np.mean(mnotes), np.min(mnotes), np.max(mnotes)) )
    mnotes -= 24 # this trims off the first two rows of the midi table
    # see: https://www.midikits.net/midi_analyser/midi_note_numbers_for_octaves.htm
    # print mnotes[0]
    onehot = np.zeros((note_num, note_range))
    onehot[np.arange(note_num), mnotes[:note_num]] = 1
    onehot = onehot.reshape(note_num, 1, note_range)
    # print np.argmax(onehot[0])
    # print np.max(onehot[0])
    # print onehot.shape
    return onehot

def get_hotones(folder):
    tracks = {}
    for file in os.listdir(folder):
        if '.js' in file:
            key = file.split('_')[0]
            if key not in tracks:
                tracks[key] = []

            notes = get_notes(os.path.join(folder,file))
            tracks[key].append(notes)

    return tracks


if __name__ == "__main__":
    midi_folder = sys.argv[1]

    tracks = get_hotones(midi_folder)
