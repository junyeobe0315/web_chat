import json

with open('ko_wiki_v1_squad.json', 'r') as f:
    data = json.load(f)

qna = []
questions = []
print(data["data"][0]["paragraphs"][0]["qas"][0]["question"])
print(data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"])
for i in range(70000):
    try:
        qna.append(data["data"][i]["paragraphs"][0]["qas"][0])
    except:
        pass
