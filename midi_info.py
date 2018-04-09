import json
import torch
import numpy as np
import sys
import os

def get_midi(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def get_tracks(midi):
    return [t for t in midi['tracks']]

def print_info(arr):
    for t in arr:
        # for key in t:
            # if key is not 'notes':
            #     print key + t[key]
        if 'name' in t:
            print 'name: ' + t['name']
        if 'instrument' in t:
            print 'instrument: ' + str(t['instrument'])
        if 'startTime' in t:
            print 'startTime: ' + str(t['startTime'])
        if 'length' in t:
            print 'length: ' + str(t['length'])
        if 'duration' in t:
            print 'duration: ' + str(t['duration'])
        if 'notes' in t:
            print 'notes: ' + str(len(t['notes']))
        print '-----------------------------'

def count_instruments(instr, arr):
    for t in arr:
        if 'instrument' in t:
            if t['instrument'] not in instr:
                instr[t['instrument']] = 0
            instr[t['instrument']] += 1
    return instr

def get_notes(midi_file):
    note_num = 30
    note_range = 95
    with open(midi_file, 'r') as f:
        midi = json.load(f)
    notes = midi['tracks'][1]['notes']
    mnotes = np.asarray([int(note['midi']) for note in notes])
    mnotes -= 24
    tensor = torch.zeros(note_num, 1, note_range)
    for ni in range(note_num):
        tensor[ni][0][mnotes[ni]] = 1
    return tensor


folder = sys.argv[1]
instr = {}
for f in os.listdir(folder):
    if '.json' in f:
        midi = get_midi(os.path.join(folder,f))
        # print '-------------One_File----------------'
        tracks = get_tracks(midi)
        instr = count_instruments(instr, tracks)
        # print_info(tracks)

for key in instr:
    print 'Instrument: {}, Frequency: {}'.format(key, instr[key])
# print len(tracks)
# print tracks[1]['name']
# print tracks[1]['instrument']
# print tracks[1]['notes'][0]
# print tracks[1]['notes'][1]
# print tracks[2]['name']
# print tracks[2]['instrument']
# print tracks[2]['notes'][0]
# print tracks[2]['notes'][1]
# print tracks[2]['notes'][2]
# print tracks[2]['notes'][3]
# print tracks[2]['notes'][4]
# print tracks[2]['notes'][5]

# tensor = get_notes('midi_data/train/bach_846.js')
# print tensor[0].numpy()
# print tensor[1].numpy()
