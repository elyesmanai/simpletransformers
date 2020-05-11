import pandas as pd
from sklearn.model_selection import train_test_split
import os

all_text = []
for fl in os.scandir("../data/"):
    if fl.name.endswith(".txt") and "text" in fl.name:
        df = pd.read_csv(fl, delimiter="\n", header=None)
        df.columns = ["text"]
        texts = df.text.tolist()
        texts = [t for t in texts if isinstance(t, str)]
        all_text.extend(texts)

train, test = train_test_split(all_text, test_size=0.1)

with open("../data/train.txt", "w") as f:
    for line in train:
        f.write(line + "\n")

with open("../dada/test.txt", "w") as f:
    for line in test:
        f.write(line + "\n")