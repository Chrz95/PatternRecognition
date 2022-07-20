import os
from glob import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split('\\')[1].split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers


def extract_features(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames


def split_free_digits(frames, ids, speakers, labels):
    X_train, X_test, y_train, y_test, spk_train, spk_test = train_test_split(frames, labels, speakers,test_size = 0.33, random_state=42)
    # print(X_train,y_train,spk_train,"+++++++++++++++++",X_test,y_test,spk_test,sep='\n')
    return X_train, X_test, y_train, y_test, spk_train, spk_test

def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    #print("Normalization will be performed using mean: {}".format(scaler.mean_))
    #print("Normalization will be performed using std: {}".format(scaler.scale_))
    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled
    return scale

def parser(directory, n_mfcc=6):
    import sys

    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(frames, ids, speakers, y)

    X_train, X_dev, y_train, y_dev,spk_train,spk_dev = train_test_split(X_train, y_train,spk_train,stratify = y_train, test_size = 0.20, random_state=42)
    scale_fn = make_scale_fn(X_train)
    X_train = scale_fn(X_train)
    X_dev = scale_fn(X_dev)
    X_test = scale_fn(X_test)

    return X_train,X_dev, X_test,y_train,y_dev , y_test, spk_train,spk_dev, spk_test


if __name__ == "__main__":
    pass

