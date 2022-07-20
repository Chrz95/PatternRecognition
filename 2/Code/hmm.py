import numpy as np
import pandas as pd
from pomegranate import *
from scripts.parser import parser

# digit: digit for which the model will be created
# n_mixtures: the number of Gaussians
# n_states: the number of HMM states
def CreateGMM_HMM_Model(X_train,y_train,digit,n_mixtures = 2, n_states = 2,N_tier = 10):
    #print("Creating GMM-HMM model for digit", digit, "with", n_mixtures, "mixtures and", n_states, "HMM states")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    digit_data = X_train[np.where(y_train == digit)] # Get all (training) samples for this digit

    # Convert data to list of lists
    digit_data = digit_data.tolist()
    for i in range(len(digit_data)):
        digit_data[i] = digit_data[i].tolist()

    """
    print(len(digit_data))
    print(len(digit_data[0]))
    print(len(digit_data[0][0]))
    print(type(digit_data))
    print(type(digit_data[0]))
    print(type(digit_data[0][0]))
    """

    # Remove first dim from digit_data, new_data: Total number of frames for this digit * n_features (13)
    new_data = []
    for i in range(len(digit_data)):
        new_data = new_data + digit_data[i]
    new_data = np.array(new_data)

    #print(new_data.shape)

    dists = [] # list of probability distributions for the HMM states
    for i in range(n_states):
        if (n_mixtures > 1):
            a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_mixtures, new_data)
        else:
            a = MultivariateGaussianDistribution.from_samples(new_data)

        dists.append(a)

    trans_mat = np.array([])

    if (n_states == 1):
        trans_mat = numpy.array([[1.0]])  # your transition matrix (n_states*n_states)
    elif (n_states == 2):
        trans_mat = numpy.array([[0.4, 0.6],
                                 [0.0, 1]])  # your transition matrix (n_states*n_states)
    elif (n_states == 3):
        trans_mat = numpy.array([[0.4, 0.6, 0.0],
                                 [0.0, 0.3, 0.7],
                                 [0.0, 0.0, 1]])  # your transition matrix (n_states*n_states)
    elif (n_states == 4):
        trans_mat = numpy.array([[0.4, 0.6, 0.0, 0.0],
                                 [0.0, 0.3, 0.7, 0.0],
                                 [0.0, 0.0, 0.6, 0.4],
                                 [0.0, 0.0, 0.0, 1]])  # your transition matrix (n_states*n_states)

    starts = np.array([1.0] + [0.0]*(n_states-1)) # your starting probability matrix
    ends = np.array([0.0]*(n_states-1) + [1.0]) # your ending probability matrix

    data = digit_data # your data: must be a Python list that contains: 2D lists with the sequences (so its dimension would be num_sequences x seq_length x feature_dimension)
              # But be careful, it is not a numpy array, it is a Python list (so each sequence can have different length)

    # Define the GMM-HMM
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])

    # Fit the model
    model.fit(data, max_iterations=N_tier)

    # Predict a sequence
    #sample = data[0] # a sample sequence
    #logp, _ = model.viterbi(sample) # Run viterbi algorithm and return log-probability
    #print("Log-probability = ",logp)

    return model

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from collections import Counter
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Split dataset
    X_train, X_dev, X_test, y_train, y_dev, y_test, spk_train, spk_dev, spk_test = parser("recordings", 13)

    #Test every model on validation data to get find parameters (n_mixtures,n_states)
    # Generate GMM-HMM models, one for every digit
    acc_dict = {}
    count_dict = {}
    pred_dict = {}
    for n_mixtures in range(1,6):
        for n_states in range(1,5):
            myModels = [CreateGMM_HMM_Model(X_train, y_train, digit, n_mixtures, n_states) for digit in range(10)]
            y_dev_pred = []
            for sample in X_dev:
                log_probs = [model.viterbi(sample)[0] for model in myModels]
                y_dev_pred.append(log_probs.index(max(log_probs)))

            acc_dict[(n_mixtures, n_states)] = accuracy_score(y_dev, y_dev_pred)*100
            pred_dict[(n_mixtures, n_states)] = y_dev_pred
            count_dict[(n_mixtures, n_states)] = Counter(y_dev_pred)
            #print("Accuracy for GMM-HMM (n_mixtures, n_states)=({},{}) is {}%".format(n_mixtures, n_states,acc_dict[(n_mixtures, n_states)] ))

    #print(acc_dict)
    df = pd.DataFrame({'Parameters': list(acc_dict.keys()),'Counts': list(count_dict.values()), 'Accuracy': list(acc_dict.values())})
    df_sorted = df.sort_values("Accuracy",ascending=False)
    print("Counts and accuracy on validation set are:")
    print(df_sorted)
    print("===============================")

    n_mixtures , n_states = df_sorted.iat[0,0] # Best parameters

     #Use best models to make predictions on test data
    BestModels = [CreateGMM_HMM_Model(X_train, y_train, digit, n_mixtures, n_states) for digit in range(10)]
    y_test_pred = []
    for sample in X_test:
        log_probs = [model.viterbi(sample)[0] for model in BestModels]
        y_test_pred.append(log_probs.index(max(log_probs)))

    print("Counts for best GMM-HMM (n_mixtures, n_states) =({},{}) on test samples is {}%".format(n_mixtures, n_states,Counter(y_test_pred)))
    acc = accuracy_score(y_test, y_test_pred) * 100
    print("Accuracy for best GMM-HMM (n_mixtures, n_states) =({},{}) on test samples is {}%".format(n_mixtures, n_states, acc))

    # Confusion matrices
    conf_best1 = confusion_matrix(y_dev, pred_dict[df_sorted.iat[0, 0]])
    print("Confusion matrix for best model on validation data is:")
    print(conf_best1)

    conf_best2 = confusion_matrix(y_test, y_test_pred)
    print("Confusion matrix for best model on test data is:")
    print(conf_best2)

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Confusion matrices')
    sns.heatmap(conf_best1, ax=axs[0])
    axs[0].set_title("Best model - validation set - conf. matrix")
    sns.heatmap(conf_best2, ax=axs[1])
    axs[1].set_title("Best model - test set - conf. matrix")
    plt.show()