import json
with open("public.json", 'r') as f:
    data = json.load(f)
for i, d in enumerate(data):
    del data[i]["relevant"]
    del data[i]["answers"]
with open("public_masked.json", 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
