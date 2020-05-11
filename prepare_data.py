import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

all_text = []
for fl in os.scandir("../data/"):
    if fl.name.endswith(".txt") and "text" in fl.name:
        print(f'treating {fl.name}')
        df = pd.read_csv(fl, names=["text"], delimiter="\n", header=None)
        texts = df.text.tolist()
        texts = [t for t in tqdm(texts) if isinstance(t, str)]
        all_text.extend(texts)

train, test = train_test_split(all_text, test_size=0.1)

with open("../data/train.txt", "w") as f:
    for line in train:
        f.write(line + "\n")

with open("../data/test.txt", "w") as f:
    for line in test:
        f.write(line + "\n")