import random
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# STEP 0
for dirname, _, filenames in os.walk('/data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# STEP 1,2,3
labels_txt = open("data/fma_genre_spectrograms/train_labels.txt")
fileList = labels_txt.readlines()
labels_txt.close()

chosen_files = [random.randint(0, len(fileList)) for i in range(2)]
chosen_files = [fileList[chosen_files[i]].split('\t')[0][0:-3] for i in range(2)]
chosen_files = ["45506.fused.full.npy","29933.fused.full.npy"]
spec1 = np.load('data/fma_genre_spectrograms/train/{}'.format(chosen_files[0]))
spec2 = np.load('data/fma_genre_spectrograms/train/{}'.format(chosen_files[1]))

mel1, chroma1 = spec1[:128], spec1[128:]
mel2, chroma2 = spec2[:128], spec2[128:]

mels = [mel1,mel2]
chromas = [chroma1, chroma2]

#Beat
spec1_b = np.load('data/fma_genre_spectrograms_beat/train/{}'.format(chosen_files[0]))
spec2_b = np.load('data/fma_genre_spectrograms_beat/train/{}'.format(chosen_files[1]))

mel1_b, chroma1_b = spec1_b[:128], spec1_b[128:]
mel2_b, chroma2_b = spec2_b[:128], spec2_b[128:]

mels_b = [mel1_b, mel2_b]
chromas_b = [chroma1_b, chroma2_b]

# Plot the spectrogram and the chromagram
for i in range(2):
    print("Spectrogram Dimensions:")
    print(mels[i].shape)  # (frequencies x time steps)

    # Spectrogram
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mels[i], x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Spectrogram' + " {}".format(chosen_files[i]))
    fig.colorbar(img, ax=ax, format="%+2.f dB")

for i in range(2):
    print("Beat-Synced Spectrogram Dimensions:")
    print(mels_b[i].shape)  # (frequencies x time steps)

    # Beat-synced spectorgram
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mels_b[i], x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Beat-Synced Spectrogram' + " {}".format(chosen_files[i]))
    fig.colorbar(img, ax=ax, format="%+2.f dB")

for i in range(2):
    print("Chromogram Dimensions:")
    print(chromas[i].shape)  # (frequencies x time steps)

    # Chromogram
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chromas[i], y_axis='chroma', x_axis='time', ax=ax)
    ax.set(title='Chromogram'  + " {}".format(chosen_files[i]))
    fig.colorbar(img, ax=ax)

for i in range(2):
    print("Beat-Synced Chromogram Dimensions:")
    print(chromas_b[i].shape)  # (frequencies x time steps)

    # Beat-synced Chromogram
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chromas_b[i], y_axis='chroma', x_axis='time', ax=ax)
    ax.set(title='Beat-Synced Chromagram'  + " {}".format(chosen_files[i]))
    fig.colorbar(img, ax=ax)

plt.show()