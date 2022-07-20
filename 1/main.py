from lib import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier
import warnings
from itertools import combinations
import itertools
from sklearn import tree
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    train_path = 'train.txt'
    test_path = 'test.txt'

    # 1.Read Data
    train = pd.read_csv(train_path, sep=" ", header=None,dtype=np.float64)
    train = train.drop([257],axis=1) # Remove NaN column
    NumOfFeatures = train.shape[1] - 1
    #print(train)

    test = pd.read_csv(test_path, sep=" ", header=None,dtype=np.float64)
    test = test.drop([257],axis=1)
    #print(test)

    X_train = train.iloc[:,1:].to_numpy() # Get features
    y_train = train.iloc[:,0].to_numpy(dtype=int)  # Get labels
    #print(X_train,y_train,sep='\n\n')

    #print('\n')

    X_test = test.iloc[:,1:].to_numpy()
    y_test = test.iloc[:,0].to_numpy(dtype=int)
    #print(X_test,y_test,sep='\n\n')

    NumOfClasses = len(set(y_train))  # 10 digits

    # 2. Design digit 131
    show_sample(X_train,131)

    # 3. Design random sample from each label
    plot_digits_samples(X_train, y_train)

    # 4. Mean of pixel (10,10) for digit '0'
    print("Mean of digit {} at {} is {}".format(0, (10,10), digit_mean_at_pixel(X_train, y_train, 0)))

    # 5. Variance of pixel (10,10) for digit '0'
    print("Variance of digit {} at {} is {}".format(0, (10,10), digit_variance_at_pixel(X_train, y_train, 0)))

    # 6,7,8 Calculate mean and variance of digit '0' at every pixel

    fig, axs = plt.subplots(1, 2)  # 20 subplots
    axs = axs.ravel()

    Means = digit_mean(X_train, y_train, 0)  # Mean and Variance of every pixel for all samples
    Vars = digit_variance(X_train, y_train, 0)

    axs[0].imshow(Means)
    axs[0].set_title("Digit {} - Means".format(0))
    axs[1].imshow(Vars)
    axs[1].set_title("Digit {} - Variances".format(0))
    plt.show()

    # 9 Calculate mean and variance of all digits at every pixel

    fig, axs = plt.subplots(2, NumOfClasses)  # 20 subplots
    axs = axs.ravel()
    cnt = 0 # Subplot counter

    AllMeans = np.zeros((NumOfClasses,256))

    for i in range(0,NumOfClasses):

        Means  = digit_mean(X_train,y_train,i) # Mean and Variance of every pixel for all samples
        Vars = digit_variance(X_train,y_train,i)

        AllMeans[i] = Means.reshape(1,256) # 16x16 to 1x256

        axs[cnt].imshow(Means)
        axs[cnt].set_title("Digit {} - Means".format(i))

        cnt += 1

        axs[cnt].imshow(Vars)
        axs[cnt].set_title("Digit {} - Variances".format(i))

        cnt += 1

    plt.show()

    # 10 Classify sample 101
    digit_101 = X_test[101,:]
    distance = [euclidean_distance(digit_101,AllMeans[i]) for i in range(0,NumOfClasses)]
    min_index = distance.index(min(distance)) # Find the digit from which we have the minimum distance
    print("The digit {} is classified as {} - {}".format(y_test[101],min_index,y_test[101]==min_index))

    # 11. Classify all samples
    y_pred = euclidean_distance_classifier(X_test,AllMeans)
    accuracy = accuracy_score(y_test, y_pred)
    print("Euclidean accuracy (test) is {} %".format(accuracy*100))

    # 12 Build a scikit-learn like classifier class
    MyClassifier = EuclideanDistanceClassifier()
    MyClassifier.fit(X_train,y_train)
    print("Euclidean clf accuracy (train) is {} %".format(MyClassifier.score(X_train, y_train) * 100))
    print("Euclidean clf accuracy (test) is {} %".format(MyClassifier.score(X_test, y_test) * 100))

    # 13
    # a - 5-Fold Cross Validation
    print("Euclidean accuracy after 5-fold Cross-Validation is {} %".format(evaluate_classifier(MyClassifier,X_train, y_train) * 100))

    # b - Plot decision surface of classifier
    # Reduce features from 256 to 2
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_pca = pca.transform(X_train)

    ClassNames = [str(i) for i in range(0,10)]
    plot_clf(MyClassifier,X_pca,y_train,ClassNames)

    # c - Plot learning curve of classifier
    MyClassifier = EuclideanDistanceClassifier()
    with warnings.catch_warnings():
        train_sizes, train_scores, test_scores = learning_curve(MyClassifier, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    #print(train_sizes,train_scores,test_scores,sep = '\n\n')
    plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0.75, 0.9))
    plt.show()

    # 14 - Calculate apriori probabilities for every class
    Apriori = calculate_priors(X_train,y_train)
    print ("Apriori probabilities for every class: \n",Apriori)

    # 15 - NaiveBayes Classifier
    # a
    CNBClassifier = CustomNBClassifier()
    CNBClassifier.fit(X_train,y_train)
    print("Test samples are sorted in the following classes: ",CNBClassifier.predict(X_test))

    #b
    print("Custom NB accuracy (train) is {} %".format(CNBClassifier.score(X_train, y_train) * 100))
    print("5-CV: ", evaluate_classifier(CNBClassifier, X_train, y_train)*100,"%")
    print("Custom NB accuracy (test) is {} %".format(CNBClassifier.score(X_test,y_test) * 100))

    # c
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    print("Scikit-Learn NB accuracy (train) is {} %".format(clf.score(X_train, y_train) * 100))
    print("5-CV: ", evaluate_classifier(clf, X_train, y_train)*100,"%")
    print("Scikit-Learn NB accuracy (test) is {} %".format(clf.score(X_test,y_test) * 100))

    # 16
    # a
    CNBClassifier1 = CustomNBClassifier(True)
    CNBClassifier1.fit(X_train, y_train)
    print(CNBClassifier1.predict(X_test))

    # b
    print("Custom NB accuracy (Var = 1) (train) is {} %".format(CNBClassifier1.score(X_train, y_train) * 100))
    print("5-CV: ", evaluate_classifier(CNBClassifier1, X_train, y_train)*100,"%")
    print("Custom NB accuracy (Var = 1) (test) is {} %".format(CNBClassifier1.score(X_test, y_test) * 100))

    # 17
    print("\nScikit Learn Classifier Comparison")

    # Naive Bayes
    GNB = GaussianNB()
    GNB.fit(X_train, y_train)
    print("Scikit-Learn NB accuracy is {} %".format(GNB.score(X_train, y_train) * 100))
    print("5-CV: ", evaluate_classifier(GNB, X_train, y_train)*100,"%")
    print("Scikit-Learn NB accuracy (test) is {} %".format(GNB.score(X_test, y_test) * 100))

    # Nearest Neighbours
    KNClf = KNeighborsClassifier(n_neighbors=3)
    KNClf.fit(X_train, y_train)
    print("Nearest Neighbours accuracy is {} %".format(KNClf.score(X_train, y_train) * 100))
    print("5-CV: ", evaluate_classifier(KNClf, X_train, y_train)*100,"%")
    print("Nearest Neighbours accuracy (test) is {} %".format(KNClf.score(X_test, y_test) * 100))

    # SVM
    SVM = SVC(kernel="linear")
    SVM.fit(X_train, y_train)
    print("SVM (linear) accuracy is {} %".format(SVM.score(X_train, y_train) * 100))
    print("5-CV: ", evaluate_classifier(SVM, X_train, y_train)*100,"%")
    print("SVM (linear) accuracy (test) is {} %".format(SVM.score(X_test, y_test) * 100))

    SVM_1 = SVC(kernel="rbf")
    SVM_1.fit(X_train, y_train)
    print("SVM (rbf) accuracy is {} %".format(SVM_1.score(X_train, y_train) * 100))
    print("5-CV: ", evaluate_classifier(SVM_1, X_train, y_train)*100,"%")
    print("SVM (rbf) accuracy (test) is {} %".format(SVM_1.score(X_test, y_test) * 100))

    SVM_2 = SVC(kernel="poly")
    SVM_2.fit(X_train, y_train)
    print("SVM (poly) accuracy is {} %".format(SVM_2.score(X_train, y_train) * 100))
    print("5-CV: ", evaluate_classifier(SVM_2, X_train, y_train)*100,"%")
    print("SVM (poly) accuracy (test) is {} %".format(SVM_2.score(X_test, y_test) * 100))

    SVM_3 = SVC(kernel="sigmoid")
    SVM_3.fit(X_train, y_train)
    print("SVM (sigmoid) accuracy is {} %".format(SVM_3.score(X_train, y_train) * 100))
    print("5-CV: ",evaluate_classifier(SVM_3, X_train, y_train)*100,"%")
    print("SVM (sigmoid) accuracy (test) is {} %".format(SVM_3.score(X_test, y_test) * 100))

    # 18    
    MyClassifier = EuclideanDistanceClassifier()
    MyClassifier.fit(X_train, y_train)
    Classifiers = [MyClassifier,CNBClassifier,CNBClassifier1,GNB,SVM_3]
    ClfNames = ['Euclidean','Custom Naive Bayes','Custom Naive Bayes (1)','Gaussian Naive Bayes','SVM with sigmoid kernel']
    predictions = [(i,clf.predict(X_train)) for i,clf in enumerate(Classifiers)]
    Combs = list(combinations(predictions, 2))
    Similarities = [ListSimilarity(comb[0][1],comb[1][1]) for comb in Combs]

    # Find which classifiers make the same mistakes
    Sim_Results = [(ClfNames[Combs[i][0][0]],ClfNames[Combs[i][1][0]],sim) for i,sim in enumerate(Similarities)]
    for i in Sim_Results:
        print(i)

    # Exclude combinations with classifiers that make the same choices
    Exclude_Pairs = [sim for sim in Sim_Results if sim[2] > 90] # If similarity between the two classifiers is too large, there is no point in test a combination that includes them both
    Combinations = list(combinations(ClfNames, 3))
    Exclude_idx = [i for excl in Exclude_Pairs for i,comb in enumerate(Combinations) if ((excl[0] in comb) and (excl[1] in comb))]
    Final_Combinations = [comb for i,comb in enumerate(Combinations) if (i not in Exclude_idx)]
    #for i in Final_Combinations:
        #print(i)

    # a
    Classifiers = [EuclideanDistanceClassifier(), CustomNBClassifier(), CustomNBClassifier(True), GaussianNB(),
                   SVC(kernel="sigmoid")]    
    Clf_dict = {ClfNames[i]: Classifiers[i] for i in range(len(Classifiers))}

    print("Hard Voting")
    for comb in Final_Combinations:
        estimators = [(comb[i], Clf_dict[comb[i]]) for i in range(len(comb))]
        eclf1 = VotingClassifier(estimators=estimators, voting='hard')
        eclf1.fit(X_train, y_train)
        print("Train: ", comb, eclf1.score(X_train, y_train) * 100, "%")
        print("5-CV: ", comb, evaluate_classifier(eclf1, X_train, y_train) * 100, "%")
    
    print("Soft Voting")
    estimators = [('KN',KNeighborsClassifier(n_neighbors=3)),('GNB',GaussianNB()),('SVC (sigmoid))',SVC(kernel="sigmoid",probability=True))]
    eclf1 = VotingClassifier(estimators=estimators, voting='soft')
    eclf1.fit(X_train, y_train)
    print("Train: (KN,GNB,SVC (sigmoid))" ,eclf1.score(X_train, y_train) * 100, "%")
    print("5-CV: (KN,GNB,SVC (sigmoid))",evaluate_classifier(eclf1, X_train, y_train) * 100, "%")

    estimators = [('KN', KNeighborsClassifier(n_neighbors=3)), ('GNB', GaussianNB()),
                  ('SVC (linear))', SVC(kernel="linear", probability=True))]
    eclf1 = VotingClassifier(estimators=estimators, voting='soft')
    eclf1.fit(X_train, y_train)
    print("Train: (KN,GNB,SVC (linear))", eclf1.score(X_train, y_train) * 100, "%")
    print("5-CV: (KN,GNB,SVC (linear))", evaluate_classifier(eclf1, X_train, y_train) * 100, "%")

    # b
    clf = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    print("Train: ", clf.score(X_train, y_train) * 100, "%")
    print("5-CV: ", evaluate_classifier(clf, X_train, y_train) * 100, "%")

    clf = BaggingClassifier(base_estimator=SVC(kernel="sigmoid", probability=True), n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    print("Train: ", clf.score(X_train, y_train) * 100, "%")
    print("5-CV: ", evaluate_classifier(clf, X_train, y_train) * 100, "%")

    # 19
    # a
    # Parameters
    run_tests = False
    if (run_tests):

        EPOCHS_L = [30,60]
        BATCH_SZ_L = [32,64]
        HLayerNeuronNum_L = [100,500]
        NumberOfHLayers_L = [2,3]
        LR_L = [1e-4,1e-2]
        Optimizers = ['Adam','SGD']
        Activations = ['sigmoid','ReLU']
        criterion = nn.CrossEntropyLoss() # define the loss function in which case is CrossEntropy Loss

        Parameters = list(itertools.product(EPOCHS_L,BATCH_SZ_L,HLayerNeuronNum_L,NumberOfHLayers_L,Activations,LR_L,Optimizers))
        Results = {param:[] for param in Parameters}

        fields = ['Epochs', 'Batch Size', 'Number of Neurons (Hidden Layers)', 'Number of Hidden Layers','Activation','Learning Rate','Optimizer',"Starting loss",'Final loss',"Training accuracy",'Testing accuracy']
        with open('Results.txt', 'w') as f:
            f.write(";".join(fields))

        #print(Parameters)

        for cnt,param in enumerate(Parameters):
            EPOCHS = param[0]  # more epochs means more training on the given data. IS this good ??
            BATCH_SZ = param[1] # the mini-batch size, usually a power of 2 but not restrictive rule in general
            HLayerNeuronNum = param[2] # Number of neurons of hidden layers
            NumberOfHLayers = param[3] # Number of hidden layers
            activation = param[4]

            net = MyNN([HLayerNeuronNum] * NumberOfHLayers, NumOfFeatures, NumOfClasses,activation=activation)
            #print(f"The network architecture is: \n {net}")

            # define the optimizer which will be used to update the network parameters
            lr = param[5]
            if (param[6] == 'Adam'):
                optimizer = optim.Adam(net.parameters(), lr=lr)  # feed the optimizer with the network parameters
            else:
                optimizer = optim.SGD(net.parameters(), lr=lr)  # feed the optimizer with the network parameters

            train_data = DigitsDataset(X_train.astype(float),y_train)
            test_data = DigitsDataset(X_test.astype(float),y_test)

            train_dl = DataLoader(train_data, batch_size=BATCH_SZ,shuffle=True)
            test_dl = DataLoader(test_data, batch_size=BATCH_SZ,shuffle=True)

            # b
            # Train neural network
            net.train()  # gradients "on"
            loss = math.inf
            for epoch in range(EPOCHS):  # loop through dataset
                running_average_loss = 0
                for i, data in enumerate(train_dl):  # loop thorugh batches
                    X_batch, y_batch = data  # get the features and labels

                    y_pred = net(X_batch.float())  # forward pass

                    #print (X_batch,len(X_batch))
                    #print('\n')
                    #print(y_batch,len(y_batch)) # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                    loss = criterion(y_pred, y_batch.long())  # compute per batch loss
                    optimizer.zero_grad()  # ALWAYS USE THIS!!
                    loss.backward()  # compute gradients based on the loss function
                    optimizer.step()  # update weights

                    running_average_loss += loss.detach().item()
                    #if i % 100 == 0:
                        #print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i, float(running_average_loss) / (i + 1)))

                if (epoch == 0) or (epoch == EPOCHS - 1):
                    Results[param].append(float(loss))

            #print('Evaluation on training samples')
            net.eval()  # turns off batchnorm/dropout ...
            acc = 0
            n_samples = 0
            with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
                for i, data in enumerate(train_dl):
                    # print(data)
                    X_batch, y_batch = data  # test data and labels
                    out = net(X_batch.float())  # get net's predictions
                    val, y_pred = out.max(1)  # argmax since output is a prob distribution
                    acc += (y_batch == y_pred).sum().detach().item()  # get accuracy
                    n_samples += BATCH_SZ

            #print("Accuracy on training samples: ", (acc / n_samples) * 100)
            Results[param].append(float(acc / n_samples) * 100)
            train_res = float(acc / n_samples) * 100

            # print('Evaluation on test samples')
            net.eval()  # turns off batchnorm/dropout ...
            acc = 0
            n_samples = 0
            with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
                for i, data in enumerate(test_dl):
                    # print(data)
                    X_batch, y_batch = data  # test data and labels
                    out = net(X_batch.float())  # get net's predictions
                    val, y_pred = out.max(1)  # argmax since output is a prob distribution
                    acc += (y_batch == y_pred).sum().detach().item()  # get accuracy
                    n_samples += BATCH_SZ

            Results[param].append(float(acc / n_samples) * 100)
            test_res = float(acc / n_samples) * 100

            #Export results
            with open('Results.txt', 'a+') as f:
                f.write("\n")
                NewList = list(param) + [train_res,test_res]
                NewList = [str(z) for z in NewList]
                f.write(";".join(NewList))

    # c,d
    # Parameter values
    EPOCHS = [30,30,60]  # more epochs means more training on the given data. IS this good ??
    BATCH_SZ = [32,32,64]  # the mini-batch size, usually a power of 2 but not restrictive rule in general
    HLayerNeuronNum = [100,100,100]  # Number of neurons of hidden layers
    NumberOfHLayers = [3,3,2]  # Number of hidden layers
    activation = ['sigmoid','sigmoid','ReLu']
    lr = [0.0001,0.01,0.0001]
    optimizer = ['SGD','Adam','Adam']

    for i in range(0,3):
        clf = PytorchNNModel(EPOCHS[i], BATCH_SZ[i], HLayerNeuronNum[i], NumberOfHLayers[i], activation[i], NumOfFeatures,
                             NumOfClasses, lr[i], optimizer[i])
        clf.fit(X_train, y_train)
        print("Evaluate PyTorchNN Classfier:")
        print("On train data: ", clf.score(X_train, y_train) * 100, '%')
        print("On test data: ", clf.score(X_test, y_test) * 100, '%')