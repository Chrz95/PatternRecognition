from sklearn.base import BaseEstimator, ClassifierMixin
import math
import matplotlib.pyplot as plt
from random import randrange
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def plot_clf(clf, X, y, labels):

    fig, ax = plt.subplots()

    # title for the plots
    title = ('Decision surface of Classifier')

    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1] # Plot based on the two first features
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1

    # numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None) # Return evenly spaced values within a given interval.
    # numpy.meshgrid(*xi, copy=True, sparse=False, indexing='xy')[source] # Return coordinate matrices from coordinate vectors.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),np.arange(y_min, y_max, .05))
    #print(xx,yy,sep = '\n\n')

    #print('\n\n')

    # np.c_[xx.ravel(), yy.ravel()] # Creates (x,y) coordinate pairs
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #print(xx,yy,Z,sep='\n\n')

    # Decision surface
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # We draw the samples from every class
    colors = ['blue','red','green','cyan','magenta','yellow','black','white','purple','brown']
    for i in range (0,10): # Place the samples
        ax.scatter(X0[y == i], X1[y == i],
        c=colors[i], label=labels[i],
        s=60, alpha=0.9, edgecolors='k')

    ax.set_ylabel(labels[1])
    ax.set_xlabel(labels[0])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 1)):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def ListSimilarity(list1,list2):
    length = len(list1)
    if (length != len(list2)):
        print ("Different Length - Cannot compare!")
        return

    same = 0
    for i in range(length):
        if (list1[i] == list2[i]):
            same += 1

    return (same/length) * 100

class DigitsDataset(Dataset):
    def __init__(self,X,y,trans = None):
        self.data = list(zip(X, y))
        self.trans = trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.trans is not None:
            return self.trans(self.data[idx])
        else:
            return self.data[idx]

class DigitsDataset_1(Dataset): # For test samples in PyTorchNN Classifier
    def __init__(self,X,trans = None):
        self.data = list(X)
        self.trans = trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.trans is not None:
            return self.trans(self.data[idx])
        else:
            return self.data[idx]

class LinearWActivation(nn.Module):  # always inherit from nn.Module
    def __init__(self, in_features, out_features, activation='sigmoid'):
        super(LinearWActivation, self).__init__()
        self.f = nn.Linear(in_features, out_features)
        if activation == 'sigmoid':
            self.a = nn.Sigmoid()
        else:
            self.a = nn.ReLU()

    def forward(self, x):  # the forward pass of info through the net
        return self.a(self.f(x))


class MyNN(nn.Module):  # again we inherit from nn.Module
    def __init__(self, layers, n_features, n_classes, activation='sigmoid'):
        super(MyNN, self).__init__()
        layers_in = [n_features] + layers  # list concatenation
        layers_out = layers + [n_classes]
        # loop through layers_in and layers_out lists
        self.f = nn.Sequential(*[
            LinearWActivation(in_feats, out_feats, activation=activation)
            for in_feats, out_feats in zip(layers_in, layers_out)
        ])
        self.clf = nn.Linear(n_classes, n_classes)

    def forward(self, x):  # again the forward pass
        y = self.f(x)
        return self.clf(y)
# ===========================================

def show_sample(X, index): # Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index
    sample_digit = X[index, :].reshape(16,16)  # Resize features to 16x16 pixel table so that we can print it as an image
    plt.imshow(sample_digit)
    plt.title("Sample {}".format(index))
    plt.show()

def plot_digits_samples(X, y): # Takes a dataset and selects one example from each label and plots it in subplots
    labels = list(range(0,10))
    samples = []
    sample_labels = []

    while(labels):
        sampleidx = randrange(0, len(y)) # Get a random sample
        if (y[sampleidx] in labels): # If we have yet to get a sample from this label
            labels.remove(y[sampleidx]) # So that we get 10 samples, ONE from each label
            samples.append(list(X[sampleidx]))
            sample_labels.append(y[sampleidx])
            #print(y[sampleidx])

    fig, axs = plt.subplots(2, 5) # 10 subplots
    axs = axs.ravel()

    for i in range(len(sample_labels)):
        axs[i].imshow(np.asarray(samples[i]).reshape(16, 16))
        axs[i].set_title("Digit " + str(sample_labels[i]))

    plt.show()


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)): # Calculates the mean for all instances of a specific digit at a pixel location
    Values = [X[i].reshape(16, 16)[pixel] for i in range(len(y)) if (y[i] == digit)]
    return np.mean(Values)

def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)): # Calculates the variance for all instances of a specific digit at a pixel location
    Values = [X[i].reshape(16, 16)[pixel] for i in range(len(y)) if (y[i] == digit)]
    #print(all(ele == Values[0] for ele in Values))
    var = np.var(Values)
    if (var == 0): # If variance is 0 (the pixel has the same value for every sample), then set it to a very small value
        var = 4.308e-10
    return var

def digit_mean(X, y, digit): # Calculates the mean for all instances of a specific digit
    Keys = [(i, j) for i in range(0, 16) for j in range(0, 16)]  # List of tuples with contain matrix coordinates (x,y)
    Means = np.zeros((16, 16))

    for j in Keys:
        Means[j[0]][j[1]] = digit_mean_at_pixel(X,y,digit,pixel=j)

    # print(Vars,sep='\n\n')
    return Means

def digit_variance(X, y, digit): # Calculates the variance for all instances of a specific digit
    Keys = [(i, j) for i in range(0, 16) for j in range(0, 16)]  # List of tuples with contain matrix coordinates (x,y)
    Vars = np.zeros((16, 16))

    for j in Keys:
        Vars[j[0]][j[1]] = digit_variance_at_pixel(X, y, digit, pixel=j)

    #print(Vars)
    return Vars

def euclidean_distance(s, m): # Calculates the euclidean distance between a sample s and a mean template m
    return math.sqrt(sum((v1-v2)**2 for v1, v2 in zip(s,m)))

def euclidean_distance_classifier(X, X_mean): # Classifiece based on the euclidean distance between samples in X and template vectors in X_mean
    NumOfTestSamples = len(X)
    y_pred = np.zeros(NumOfTestSamples,dtype=int)
    for i in range(0,NumOfTestSamples):
        distance = [euclidean_distance(X[i,:], X_mean[j]) for j in range(0, 10)]
        min_index = distance.index(min(distance))
        y_pred[i] = min_index

    return y_pred

class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin): # Classify samples based on the distance from the mean feature value

    def __init__(self):
        pass

    def fit(self, X, y): # We pass the training samples, so that our classifier 'learns'

        self.NumOfClasses = len(set(y))
        #self.NumOfClasses = 10
        self.X_mean_ = np.zeros((self.NumOfClasses, 256))

        for i in range(0, self.NumOfClasses):
            Means = digit_mean(X, y, i)  # Mean of every pixel for all samples
            self.X_mean_[i] = Means.reshape(1, 256)

        return self

    def predict(self, X): # We predict that class that each sample belongs to
        NumOfTestSamples = len(X)
        y_pred = np.zeros(NumOfTestSamples,dtype=int)
        for i in range(0,NumOfTestSamples):
            distance = [euclidean_distance(X[i,:], self.X_mean_[j]) for j in range(0, self.NumOfClasses)]
            min_index = distance.index(min(distance))
            y_pred[i] = min_index

        return y_pred

    def score(self, X, y): # We present the accuracy of our prediction
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def evaluate_classifier(clf, X, y, folds=5): # Returns the 5-fold accuracy for classifier clf on X and y
    scores = cross_val_score(clf, X, y,cv=KFold(n_splits=folds, random_state=42,shuffle=True),scoring="accuracy")
    accuracy = np.mean(scores)

    return accuracy

# ========================================

def calculate_priors(X, y):
    return np.array([len(y[y == i])/len(y) for i in range(0,10)])

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance

    def fit(self, X, y):
        self.NumOfClasses = len(set(y))
        self.NumOfFeatures = X.shape[1]

        self.Apriori = calculate_priors(X, y)
        self.X_mean_ = np.ones((self.NumOfClasses, self.NumOfFeatures))
        self.X_var_ = np.ones((self.NumOfClasses, self.NumOfFeatures))

        for i in range(0, self.NumOfClasses):
            Means = digit_mean(X, y, i)  # Mean of every pixel for all samples
            self.X_mean_[i] = Means.reshape(1, self.NumOfFeatures)
            if (not self.use_unit_variance):
                Variance = digit_variance(X, y, i)  # Variance of every pixel for all samples
                self.X_var_[i] = Variance.reshape(1, self.NumOfFeatures)

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
        return accuracy_score(y, y_pred)

class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        # TODO: initialize model, criterion and optimizer
        self.EPOCHS = args[0]  # more epochs means more training on the given data. IS this good ??
        self.BATCH_SZ = args[1]  # the mini-batch size, usually a power of 2 but not restrictive rule in general
        self.HLayerNeuronNum = args[2]  # Number of neurons of hidden layers
        self.NumberOfHLayers = args[3]  # Number of hidden layers
        self.activation = args[4]
        self.NumOfFeatures = args[5]
        self.NumOfClasses = args[6]
        self.lr = args[7]

        self.model = MyNN([self.HLayerNeuronNum] * self.NumberOfHLayers, self.NumOfFeatures, self.NumOfClasses, activation=self.activation)
        self.criterion = nn.CrossEntropyLoss() # define the loss function in which case is CrossEntropy Loss

        if (args[8] == 'Adam'):
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # feed the optimizer with the network parameters
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)  # feed the optimizer with the network parameters

    def fit(self, X, y):
        # TODO: split X, y in train and validation set and wrap in pytorch dataloaders
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)
        train_data = DigitsDataset(X_train.astype(float), y_train)
        val_data = DigitsDataset(X_val.astype(float), y_val)

        self.train_dl = DataLoader(train_data, batch_size=self.BATCH_SZ, shuffle=True)
        self.val_dl = DataLoader(val_data, batch_size=self.BATCH_SZ, shuffle=True)

        # TODO: Train model
        self.model.train()  # gradients "on"
        loss = math.inf
        for epoch in range(self.EPOCHS):  # loop through dataset
            running_average_loss = 0
            for i, data in enumerate(self.train_dl):  # loop thorugh batches
                X_batch, y_batch = data  # get the features and labels
                y_pred = self.model(X_batch.float())  # forward pass
                loss = self.criterion(y_pred, y_batch.long())  # compute per batch loss
                self.optimizer.zero_grad()  # ALWAYS USE THIS!!
                loss.backward()  # compute gradients based on the loss function
                self.optimizer.step()  # update weights
                running_average_loss += loss.detach().item()
                if i % 100 == 0:
                    print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i, float(running_average_loss) / (i + 1)))

        # Evaluate on validation set
        self.model.eval()  # turns off batchnorm/dropout ...
        acc = 0
        n_samples = 0
        with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
            for i, data in enumerate(self.val_dl):
                # print(data)
                X_batch, y_batch = data  # test data and labels
                out = self.model(X_batch.float())  # get net's predictions
                val, y_pred = out.max(1)  # argmax since output is a prob distribution
                acc += (y_batch == y_pred).sum().detach().item()  # get accuracy
                n_samples += self.BATCH_SZ

        print("Accuracy on validation set:",(acc/n_samples)*100,'%')

    def predict(self, X): # WARNING: Make sure predict returns the expected (nsamples) numpy array not a torch tensor.
        # TODO: wrap X in a test loader and evaluate
        test_data = DigitsDataset_1(X.astype(float))
        test_dl = DataLoader(test_data, batch_size=self.BATCH_SZ)

        # Evaluate on test set
        self.model.eval()  # turns off batchnorm/dropout ...
        acc = 0
        n_samples = 0
        y_pred_total = []
        with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
            for i, data in enumerate(test_dl):
                # print(data)
                X_batch = data  # test data and labels
                out = self.model(X_batch.float())  # get net's predictions
                val, y_pred = out.max(1)  # argmax since output is a prob distribution
                y_pred_total += y_pred.tolist()

        return np.array(y_pred_total)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)