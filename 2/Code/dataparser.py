import os,sys
from glob import glob
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm # A Fast, Extensible Progress Bar for Python and CLI
from statistics import mean,stdev
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    #print(files)

    fnames = [f.split("\\")[1].split(".")[0] for f in files]
    #print(fnames)

    ids = [''.join([i for i in ini_string if not i.isdigit()]) for ini_string in fnames] # Digits spoken at each file
    speakers = [''.join([i for i in ini_string if i.isdigit()]) for ini_string in fnames] # Speaker of each file
    WordToNumDict = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
    y = [WordToNumDict[digit] for digit in ids]

    # Read all wavs
    wavs = [librosa.core.load(f, sr=None) for f in files]

    # Print dataset info
    _, Fs = librosa.core.load(files[0], sr=None)
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, speakers, ids, Fs , y

def extract_features(wavs, n_mfcc=13, Fs=16000):
    # Extract MFCCs for all wavs
    window = 25 * Fs // 1000
    step = 10 * Fs // 1000
    wavs = np.array(wavs)

    frames = [librosa.feature.mfcc(wav[0], Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc).T for wav in tqdm(wavs, desc="Extracting mfcc features...")]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    deltas = [librosa.feature.delta(frame,order=1) for frame in frames]
    delta_deltas = [librosa.feature.delta(frame,order=2) for frame in frames]

    return frames,deltas,delta_deltas


def split_free_digits(features, ids, speakers, labels):
    X_train,X_test,y_train,y_test,spk_train,spk_test = train_test_split(features, labels,speakers,test_size=0.30, random_state=42)
    #print(X_train,y_train,spk_train,"+++++++++++++++++",X_test,y_test,spk_test,sep='\n')
    return X_train, X_test, y_train, y_test, spk_train, spk_test

def make_histograms_nfcc(frames,y,speakers,ids,nfccs_idx,digits):
    digits_idx = {digit:[] for digit in digits} # Get indexes (file number) of each digit
    for digit in digits:
        for idx,i in enumerate(y):
            if (i == digit):
                digits_idx[digit].append(idx)

    #print(digits_idx)

    pltcnt = 0
    try:
        files = glob.glob("Plots/4_1/*")
        for f in files:
            os.remove(f)
    except:
        pass

    all_frames = {digit:[] for digit in digits}

    for digit in digits: # For every digit [3,2]
        for file in digits_idx[digit]:  # For every file this digit is spoken
            for lst in frames[file]:
                all_frames[digit].append(lst)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('First two NFFCs for digits 2,3')

    X_count = 0
    Y_count = 0
    for digit in digits:
        Y_count = 0
        for nfcc in nfccs_idx: # For nfccs [0,1] (1st and 2nd)
            x = list(map(itemgetter(nfcc), all_frames[digit]))
            #myCount = [1 for i in x if (-800 <i <-700)]
            #print(sum(myCount))
            axs[X_count, Y_count].hist(x)
            axs[X_count, Y_count].set_title("Digit {} - NFCC {}".format(digit,nfcc))
            axs[X_count, Y_count].set_xlabel('nFCC')
            Y_count = Y_count + 1

        X_count = X_count + 1

    plt.savefig("Plots/4_1/NFCCs for digits 3,2.png")
    plt.clf()

def show_correlations_nsfc_nfcc(wavs,frames,y,speakers,ids,digits):
    Fs = 16000
    window = 25 * Fs // 1000
    step = 10 * Fs // 1000
    wavs = np.array(wavs)

    try:
        files = glob.glob("Plots/4_2/*")
        for f in files:
            os.remove(f)
    except:
        pass

    frames_nfsc = [librosa.feature.melspectrogram(wav[0], Fs, n_fft=window, hop_length=window - step, n_mels=13).T for wav in tqdm(wavs, desc="Extracting mfsc features...")]

    digits_idx = {digit: [] for digit in digits}  # Get indexes (file number) of each digit
    for digit in digits:
        for idx, i in enumerate(y):
            if (i == digit):
                digits_idx[digit].append(idx)

    for key,idx in digits_idx.items(): # We want only two files for each digit
        digits_idx[key] = digits_idx[key][0:2]

    NFCCs_dict = {(i,j):frames[j] for i in digits_idx.keys() for j in digits_idx[i]}
    NFSCs_dict = {(i,j):frames_nfsc[j] for i in digits_idx.keys() for j in digits_idx[i]}

    # NFCCs
    for x,y in NFCCs_dict.items():
        #print("Digit: ",x[0],", Speaker: ",speakers[x[1]])
        corrs = np.corrcoef(y,rowvar = False)
        #print(corrs)
        #print(len(corrs),len(corrs[0]))

        # Depict correlations
        ax = sns.heatmap(corrs)
        plt.title('Correlation of NFCCs (Digit:{},Speaker:{})'.format(x[0],speakers[x[1]]))
        plt.savefig("Plots/4_2/{}_{}_NFCC.png".format(speakers[x[1]], x[0]))
        plt.clf()

    # NFSCs
    for x,y in NFSCs_dict.items():
        #print("Digit: ",x[0],", Speaker: ",speakers[x[1]])
        corrs = np.corrcoef(y,rowvar = False)
        #print(corrs)
        #print(len(corrs),len(corrs[0]))

        # Depict correlations
        ax = sns.heatmap(corrs)
        plt.title('Correlation of NFSCs (Digit:{},Speaker:{})'.format(x[0],speakers[x[1]]))
        plt.savefig("Plots/4_2/{}_{}_NFSC.png".format(speakers[x[1]],x[0]))
        plt.clf()

def make_scatterplots_first_two(final_features_mean,final_features_std,y,speakers):

    # print(final_features_mean)
    # print(len(final_features_mean))
    # print(len(final_features_mean[0]))

    Markers = ["d", "4", "<", "P", "*", "H", "_", "8", "s", "+", ".", "v", "1", "2", "3"]
    Colors = ["red","green","yellow","orange","black","blue","purple","cyan","magenta"]
    MarkersDict = {i+1:Markers[i] for i in range(0,9)}
    ColorsDict = {i+1:Colors[i] for i in range(0,9)}


    ####### Scatterplot for mean #######

    first_dim_mean = list(map(itemgetter(0), final_features_mean))
    second_dim_mean = list(map(itemgetter(1), final_features_mean))

    y_labels = list(map(str,y))
    data = pd.DataFrame(list(zip(first_dim_mean, second_dim_mean,y_labels)),columns =['1st NFCC','2nd NFCC','digit'])
    data = data.sort_values("digit")
    MyPlot = sns.scatterplot(data=data, x="1st NFCC", y="2nd NFCC", hue='digit', style='digit').set_title('Mean')
    fig = MyPlot.get_figure()
    fig.set_size_inches(17, 9)
    fig.savefig('Plots/5_mean.png')
    plt.clf()

    ####### Scatterplot for std #######

    first_dim_std = list(map(itemgetter(0), final_features_std))
    second_dim_std = list(map(itemgetter(1), final_features_std))

    y_labels = list(map(str, y))
    data = pd.DataFrame(list(zip(first_dim_std, second_dim_std, y_labels)), columns=['1st NFCC', '2nd NFCC', 'digit'])
    data = data.sort_values("digit")
    MyPlot = sns.scatterplot(data=data, x="1st NFCC", y="2nd NFCC", hue='digit', style='digit').set_title('Standard Deviation')
    fig = MyPlot.get_figure()
    fig.set_size_inches(17, 9)
    fig.savefig('Plots/5_std.png')
    plt.clf()

def make_scatterplots_PCA_2D(final_features_mean,final_features_std,y,speakers):
    n_components = 2

    pca = PCA(n_components=n_components)
    final_features_mean = pca.fit_transform(final_features_mean)
    final_features_std = pca.fit_transform(final_features_std)

    #print(final_features_mean)
    #print(len(final_features_mean))
    #print(len(final_features_mean[0]))

    Markers = ["d", "4", "<", "P", "*", "H", "_", "8", "s", "+", ".", "v", "1", "2", "3"]
    Colors = ["red", "green", "yellow", "orange", "black", "blue", "purple", "cyan", "magenta"]
    MarkersDict = {i + 1: Markers[i] for i in range(0, 15)}
    ColorsDict = {i + 1: Colors[i] for i in range(0, 9)}

    ####### Scatterplot for mean #######

    first_dim_mean = list(map(itemgetter(0), final_features_mean))
    second_dim_mean = list(map(itemgetter(1), final_features_mean))

    y_labels = list(map(str, y))
    data = pd.DataFrame(list(zip(first_dim_mean, second_dim_mean, y_labels)), columns=['1st PCA', '2nd PCA', 'digit'])
    data = data.sort_values("digit")
    MyPlot = sns.scatterplot(data=data, x="1st PCA", y="2nd PCA", hue='digit', style='digit').set_title('Mean')
    fig = MyPlot.get_figure()
    fig.set_size_inches(17, 9)
    fig.savefig('Plots/6_2D_mean.png')
    plt.clf()

    ####### Scatterplot for std #######

    first_dim_std = list(map(itemgetter(0), final_features_std))
    second_dim_std = list(map(itemgetter(1), final_features_std))

    y_labels = list(map(str, y))
    data = pd.DataFrame(list(zip(first_dim_std, second_dim_std, y_labels)), columns=['1st PCA', '2nd PCA', 'digit'])
    data = data.sort_values("digit")
    MyPlot = sns.scatterplot(data=data, x="1st PCA", y="2nd PCA", hue='digit', style='digit').set_title(
        'Standard Deviation')
    fig = MyPlot.get_figure()
    fig.set_size_inches(17, 9)
    fig.savefig('Plots/6_2D_std.png')
    plt.clf()

def make_scatterplots_PCA_3D(final_features_mean,final_features_std,y,speakers):
    n_components = 3

    pca = PCA(n_components=n_components)
    final_features_mean = pca.fit_transform(final_features_mean)
    final_features_std = pca.fit_transform(final_features_std)

    #print(final_features_mean)
    #print(len(final_features_mean))
    #print(len(final_features_mean[0]))

    Markers = ["d", "4", "<", "P", "*", "H", "_", "8", "s", "+", ".", "v", "1", "2", "3"]
    Colors = ["red", "green", "yellow", "orange", "black", "blue", "purple", "cyan", "magenta"]
    MarkersDict = {i + 1: Markers[i] for i in range(0, 15)}
    ColorsDict = {i + 1: Colors[i] for i in range(0, 9)}
    y_labels = list(map(str, y))

    ####### Scatterplot for mean #######

    first_dim_mean = list(map(itemgetter(0), final_features_mean))
    second_dim_mean = list(map(itemgetter(1), final_features_mean))
    third_dim_mean = list(map(itemgetter(2), final_features_mean))

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    fig.set_size_inches(17, 9)
    legend_elements = []

    # Design each point with its respective color and marker
    for i in range(len(first_dim_mean)):
        legend_elements.append(ax.scatter3D(first_dim_mean[i], second_dim_mean[i],third_dim_mean[i], c=ColorsDict[y[i]], marker=MarkersDict[y[i]], label=y_labels[i]))
        # print(MarkersDict[int(speakers[i])],speakers[i])

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    df = pd.DataFrame(list(zip(by_label.values(), by_label.keys())),columns=['Values','Keys'])
    df = df.sort_values('Keys')
    plt.legend(df['Values'],df['Keys'])

    plt.title('Mean')
    plt.xlabel('First feature')
    plt.ylabel('Second feature')
    ax.set_zlabel('Third feature')
    plt.savefig('Plots/6_3D_mean.png')

    ####### Scatterplot for std #######

    first_dim_std = list(map(itemgetter(0), final_features_std))
    second_dim_std = list(map(itemgetter(1), final_features_std))
    third_dim_std = list(map(itemgetter(2), final_features_std))

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    fig.set_size_inches(17, 9)
    legend_elements = []

    # Design each point with its respective color and marker
    for i in range(len(first_dim_std)):
        legend_elements.append(
            ax.scatter3D(first_dim_std[i], second_dim_std[i], third_dim_std[i], c=ColorsDict[y[i]],
                         marker=MarkersDict[y[i]], label=y_labels[i]))

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    df = pd.DataFrame(list(zip(by_label.values(), by_label.keys())), columns=['Values', 'Keys'])
    df = df.sort_values('Keys')
    plt.legend(df['Values'], df['Keys'])

    plt.title('Standard Deviation')
    plt.xlabel('First feature')
    plt.ylabel('Second feature')
    ax.set_zlabel('Third feature')
    plt.savefig('Plots/6_3D_std.png')
    plt.clf()

def digit_mean(X, y): # Calculates the mean for all instances of a specific digit
    n_features = len(X[0])
    Classes = sorted(list(set(y)))
    Means = [[np.mean(X[y == i, j]) for j in range(n_features)] for i in Classes]
    #print(len(Means),len(Means[0]))
    #print(Means,sep='\n\n')
    return Means

def digit_variance(X, y): # Calculates the variance for all instances of a specific digit
    n_features = len(X[0])
    Classes = sorted(list(set(y)))
    Vars = [[np.var(X[y == i, j]) for j in range(n_features)] for i in Classes]
    # print(len(Vars),len(Vars[0]))
    # print(Vars,sep='\n\n')
    return Vars

def calculate_priors(X, y):
    y = np.array(y)
    return np.array([len(y[y == i])/len(y) for i in range(1,10)])

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance

    def fit(self, X, y):
        self.NumOfClasses = len(set(y))
        self.NumOfFeatures = X.shape[1]

        self.Apriori = calculate_priors(X, y)
        #print(self.Apriori)
        #print(sum(self.Apriori))
        self.X_mean_ = digit_mean(X, y)
        self.X_var_ = digit_variance(X, y)

        return self

    def predict(self, X):
        NumOfTestSamples = len(X)
        y_pred = np.zeros(NumOfTestSamples,dtype=int)

        for i in range(0,NumOfTestSamples):
            probabilities = [np.prod((1 / (np.sqrt(2 * np.pi) * np.sqrt(self.X_var_[j]))) * np.exp(-((X[i] - self.X_mean_[j]) ** 2 / (2 * np.sqrt(self.X_var_[j]) ** 2)))) * self.Apriori[j] for j in range(self.NumOfClasses)]
            y_pred[i] = probabilities.index(max(probabilities))

        return y_pred

    def score(self, X, y):  # We present the accuracy of our prediction
        y_pred = self.predict(X)
        y_pred = [y+1 for y in y_pred]
        return accuracy_score(y, y_pred)

def getFeaturesFromFrames(frames,deltas,delta_deltas):
    final_features_mean = []
    final_features_std = []

    features = [[]] * len(deltas)

    for i in range(len(deltas)): # Merge mfccs,delta,delta-deltas
        features[i] = []
        # print(len(deltas[i]))
        for j in range(len(deltas[i])):
            # print(frames[i][j])
            features[i].append(frames[i][j].tolist() + deltas[i][j].tolist() + delta_deltas[i][j].tolist())
            # print(features[i][j])
            # print(len(features[i][j]))

    for i, file in enumerate(features): # Get mean and std vectors
        final_features_mean.append([])
        final_features_std.append([])
        for val in range(len(file[0])):
            feature = list(map(itemgetter(val), file))
            final_features_mean[i].append(mean(feature))
            final_features_std[i].append(stdev(feature))

    return final_features_mean,final_features_std

def parser(directory,choice, n_mfcc=13):

    wavs,speakers,ids,Fs,y = parse_free_digits(directory)
    frames,deltas,delta_deltas = extract_features(wavs)
    # frames : 133x50 (approximately) x13

    make_histograms_nfcc(frames,y,speakers,ids,[0,1],[3,2])
    show_correlations_nsfc_nfcc(wavs,frames,y,speakers,ids,[3,2])

    final_features_mean,final_features_std = getFeaturesFromFrames(frames, deltas, delta_deltas)

    make_scatterplots_first_two(final_features_mean,final_features_std,y,speakers)
    make_scatterplots_PCA_2D(final_features_mean,final_features_std, y,speakers)
    make_scatterplots_PCA_3D(final_features_mean,final_features_std, y, speakers)

    features = []
    if (choice == 1):
        features = final_features_mean
        #print(features)
        #print(len(features))
        #print(len(features[0]))
    elif (choice == 2):
        features = [sublist[0:13] for sublist in final_features_mean]
        #print(features)
        #print(len(features))
        #print(len(features[0]))
    elif (choice == 3):
        features = [sublist[0:26] for sublist in final_features_mean]
    elif (choice == 4):
        pca = PCA(n_components=2)
        features = pca.fit_transform(final_features_mean)
    elif (choice == 5):
        pca = PCA(n_components=2)
        features = pca.fit_transform([sublist[0:13] for sublist in final_features_mean])
    elif (choice == 6):
        pca = PCA(n_components=2)
        features = pca.fit_transform([sublist[0:26] for sublist in final_features_mean])
    elif (choice == 7):
        features = final_features_std
    elif (choice == 8):
        features = [sublist[0:13] for sublist in final_features_std]
    elif (choice == 9):
        features = [sublist[0:26] for sublist in final_features_std]
    elif (choice == 10):
        pca = PCA(n_components=2)
        features = pca.fit_transform(final_features_std)
    elif (choice == 11):
        pca = PCA(n_components=2)
        features = pca.fit_transform([sublist[0:13] for sublist in final_features_std])
    elif (choice == 12):
        pca = PCA(n_components=2)
        features = pca.fit_transform([sublist[0:26] for sublist in final_features_std])
    else:
        pass

    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(features, ids, speakers, y)
    return X_train, X_test, y_train, y_test, spk_train, spk_test

def classify(X_train,X_test,y_train,y_test):

    Classifiers = [CustomNBClassifier(),GaussianNB(),SVC(kernel='linear'),KNeighborsClassifier(),RandomForestClassifier()]

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    for i,clf in enumerate(Classifiers):
        clf.fit(X_train,y_train)
        print("{}. {}:  ".format(i+1,type(clf).__name__))
        print("\tAccuracy on test dataset is {:.2f} %".format(clf.score(X_test, y_test) * 100))

if __name__ == "__main__":

    # Create different types of features
    for choice in range(1,13):
        print("Choice",choice)
        X_train, X_test, y_train, y_test, spk_train, spk_test = parser("digits",choice,13)

        # Trying normalization on different subsets of data
        for norm in range(0,3):
            print("Normalization",norm)
            if (norm == 0):
                print("If using all data to calculate normalization statistics")
                X = np.concatenate((X_train , X_test))
                scaler = StandardScaler().fit(X)
                #print(X.shape)
                #print("Normalization will be performed using mean: {}".format(scaler.mean_))
                #print("Normalization will be performed using std: {}".format(scaler.scale_))
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            elif (norm == 1):
                print("If using X_train to calculate normalization statistics")
                X = X_train
                scaler = StandardScaler().fit(X)
                # print("Normalization will be performed using mean: {}".format(scaler.mean_))
                # print("Normalization will be performed using std: {}".format(scaler.scale_))
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            else: # Dont normalize
                pass

            classify(X_train, X_test, y_train, y_test)
            print("======================================")