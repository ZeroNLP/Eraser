import json
import numpy as np

file_path = ""
scores = []
with open(file_path, 'r') as file:
    for line in file.readlines():
        data = json.loads(line)
        scores.append(int(data['score']))

print("================================================")
print(f"[AVG SCORE]: {np.average(scores)}")
accs = [s>=3 for s in scores]
print(f"[AVG ACC]: {np.average(accs)}")
print("================================================")