from google.colab import drive
drive.mount('/content/gdrive')

with open("/content/gdrive/My Drive/thesis/pretraining.txt",'r') as f:
  dataset = f.readlines()

with open("data/pretraining.txt", "w") as f:
  f.write('\n'.join(final_dataset))
