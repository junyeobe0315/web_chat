import json

with open('intents.json', 'r') as f:
    intents = json.load(f)
print(intents)

tags = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

print(tags)