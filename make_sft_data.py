import json
from utiles import split_to_sharegpt
file_name = "jarvis_chat_v1.jsonl"
data = []
with open(file_name,'r',) as f:
    files = f.readlines()
    i = 1
    for file in files:
        try:
            a = split_to_sharegpt(file)
            data.extend(a)
        except:
            print(i,'this line contains error')
            continue
        i += 1
    with open('jarvis_chat_sharegpt.json','w') as ff:
        json.dump(data, ff, ensure_ascii=False,indent=4)



