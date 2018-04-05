
# coding: utf-8

# In[47]:


import json


file = open('beethoven_opus_10_1.json', 'r')


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
    
# combined is now a single track containing all notes. 
# This should work with fugue tracks too.
# This will probably break if we try to include instrumentation later
    
    
jsonWrite = json.dumps(combined, indent=4)
newFile = open(newName, 'w')
newFile.write(jsonWrite)
newFile.close()

        
#print(combined)

