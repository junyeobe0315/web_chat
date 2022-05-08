import random
import json

import torch

from model import NeuralNet
from khaiii_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"


def get_response(sentence):
    sentence = make_no_space(sentence)
    sentence = spacing(sentence)
    sentence = spell_check(sentence)
    tokenized_sentence = tokenize(sentence)

    X = bag_of_words(tokenized_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tags = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "제가 잘 이해하지 못했어요 다시 한번 말해주세요..."


if __name__ == "__main__":
    print("채팅을 시작해봅시다!( 'quit'을 입력하면 종료됩니다.)")
    while True:
        sentence = input("당신 : ")
        if sentence == 'quit':
            break

        response = get_response(sentence)
        print(response)
