import numpy as np
import torch,math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from scripts.parser import parser
import torch.optim as optim
import sys

class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """

        self.lengths = [len(frame) for frame in feats] # Find the lengths
        #print(self.lengths)
        #print(max(self.lengths))
        #print("========================")

        self.feats = self.zero_pad_and_stack(feats)
        #print([len(frame) for frame in self.feats])
        #print('\n')
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('int64')

    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        max_sequence_length = max(self.lengths)
        for i in range(len(x)):
            row_to_be_added = np.array([0]*(13))
            # Adding row to numpy array
            for j in range(max_sequence_length - x[i].shape[0]):
                x[i] = np.vstack((x[i], row_to_be_added))

        padded = np.array(x)
        #print(padded)
        #print(padded.shape)
        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False,Dropout_factor = 0.2):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=rnn_size ,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(Dropout_factor)
        self.fc = nn.Linear(self.feature_size,output_dim)

    def forward(self, x, lengths):
        """
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        
        # --------------- Insert your code here ---------------- #
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network

        out, _ = self.lstm(x.float())
        last_outputs = self.last_timestep(out, lengths, self.bidirectional)
        out = self.dropout(last_outputs)
        out = self.fc(out)
        return out

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)
        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()



def save_checkpoint(model,save_path,optimizer,valid_loss):
    if (save_path == None):
        print("Not a valid save path!")
        return

    torch.save({'model_state_dict' :model.state_dict(),'optimizer_state_dict' :optimizer.state_dict(),'valid_loss' :valid_loss},save_path)

def load_checkpoint(model, load_path, optimizer, valid_loss):
    if (load_path == None):
        print("Not a valid load path!")
        return

    state_dict = torch.load(load_path)

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    X_train, X_dev, X_test, y_train, y_dev, y_test, spk_train, spk_dev, spk_test = parser("recordings",13)

    train_data = FrameLevelDataset(X_train, y_train)
    val_data = FrameLevelDataset(X_dev, y_dev)
    test_data = FrameLevelDataset(X_test, y_test)

    input_size = 13
    output_size = 10
    hidden_size = 100
    numofLayers = 3
    epochs = 100
    perc = 10

    BATCH_SZ = len(train_data) // perc
    train_dl = DataLoader(train_data, batch_size=BATCH_SZ, shuffle=True)

    BATCH_SZ_val = len(val_data) // perc
    dev_dl = DataLoader(val_data, batch_size=BATCH_SZ_val, shuffle=True)

    BATCH_SZ_test = len(test_data) // perc
    test_dl = DataLoader(test_data, batch_size=BATCH_SZ_test, shuffle=True)

    # """

    # Estimate best parameters for model by calculatiνγ accuracy on validation set

    learning_rates = [1e-4,1e-2]

    parameters_dict = {}

    file = open("Train_val_loss_acc.txt", 'w')
    file.close()

    file = open("Train_val_loss_acc.txt",'a+')

    for lr in learning_rates:
            for Dropout_factor in [0, 0.2, 0.5]:
                for weight_decay in [0,1e-3, 1e-1]:
                        for patience in [math.inf,1,3,7]:
                            for bidirectional in [False, True]:
                                    #print("Layers={},Batch%={},LR={},Epochs={},Dropout={},L2={},BiDir={},Patience={}".format(num_layers, perc, lr, epochs, Dropout_factor, weight_decay,                                          bidirectional, patience))
                                    #sys.stdout.flush()
                                    file.write("\nLR={},Dropout={},L2={},BiDir={},Patience={}\n".format(lr, Dropout_factor, weight_decay,
                                            bidirectional, patience))
                                    file.flush()

                                    model = BasicLSTM(input_size, hidden_size, output_size, numofLayers, bidirectional, Dropout_factor)
                                    criterion = torch.nn.CrossEntropyLoss()
                                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # feed the optimizer with the network parameters

                                    # Train model
                                    val_losses = []
                                    stop_cnt = 0
                                    prev_val_loss = math.inf
                                    for epoch in range(epochs):
                                        #print("EPOCH {}".format(epoch))
                                        #sys.stdout.flush()
                                        file.write("EPOCH {} \n".format(epoch))
                                        file.flush()

                                        # Train model with training samples
                                        running_average_loss = 0
                                        for i, (data, label,length) in enumerate(train_dl):
                                            # Forward Data #
                                            pred = model(data,length)
                                            #print(pred)

                                            # Calculate Loss #
                                            train_loss = criterion(pred, label)

                                            # Initialize Optimizer, Back Propagation and Update #
                                            optimizer.zero_grad()
                                            train_loss.backward()
                                            optimizer.step()
                                            running_average_loss += train_loss.detach().item()

                                        train_actual_loss = float(running_average_loss) / len(train_dl)
                                        file.write("Train loss: {}\n".format(train_actual_loss))
                                        file.flush()

                                        # Eval model on validation data
                                        model.eval()  # turns off batchnorm/dropout ...
                                        acc = 0
                                        n_samples = 0
                                        running_average_loss = 0
                                        with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
                                            for i, (data, label,length) in enumerate(dev_dl):
                                                # print(data)
                                                out = model(data.float(),length)  # get net's predictions
                                                valid_loss = criterion(out, label)
                                                val, y_pred = out.max(1)  # argmax since output is a prob distribution
                                                acc += (label == y_pred).sum().detach().item()  # get accuracy
                                                n_samples += BATCH_SZ_val
                                                running_average_loss += valid_loss.detach().item()

                                        val_actual_loss = float(running_average_loss) / (len(dev_dl))
                                        val_actual_acc = float((acc / n_samples) * 100)
                                        parameters_dict[(lr, Dropout_factor, weight_decay,
                                                         bidirectional,patience)] = val_actual_acc

                                        # Early Stopping
                                        if (prev_val_loss < val_actual_loss):
                                            stop_cnt = stop_cnt + 1
                                            if (stop_cnt > patience):
                                                file.write("EARLY STOP: Valid loss: {},Valid acc: {}%\n".format(val_actual_loss, val_actual_acc))
                                                file.flush()
                                                break
                                        else:
                                            stop_cnt = 0

                                        prev_val_loss =  val_actual_loss
                                        file.write("Valid loss: {},Valid acc: {}%\n".format(val_actual_loss, val_actual_acc))
                                        file.flush()

    file.close()
    df = pd.DataFrame({'Parameters': list(parameters_dict.keys()), 'Accuracy': list(parameters_dict.values())})
    df_sorted = df.sort_values("Accuracy", ascending=False)
    print(df_sorted)
    best_param = df_sorted.iat[0, 0]
    print("The best parameters are {} with accuracy {}% on validation data".format(best_param,df_sorted.iat[0, 1]))  # Best parameters
    lr, Dropout_factor, weight_decay,bidirectional,patience = best_param

    #"""

    model = BasicLSTM(input_size, hidden_size, output_size, numofLayers, bidirectional, Dropout_factor)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)  # feed the optimizer with the network parameters

    # Train BEST model
    print("Training best model")
    stop_cnt = 0
    prev_val_loss = math.inf
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        #print("EPOCH {}".format(epoch))
        sys.stdout.flush()

        # Train model with training samples
        running_average_loss = 0
        for i, (data, label, length) in enumerate(train_dl):
            # Forward Data #
            pred = model(data, length)
            # print(pred)

            # Calculate Loss #
            train_loss = criterion(pred, label)

            # Initialize Optimizer, Back Propagation and Update #
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            running_average_loss += train_loss.detach().item()

        train_actual_loss = float(running_average_loss) / len(train_dl)
        train_losses.append(train_actual_loss)

        # Eval model on validation data
        model.eval()  # turns off batchnorm/dropout ...
        acc = 0
        n_samples = 0
        running_average_loss = 0
        with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
            for i, (data, label, length) in enumerate(dev_dl):
                # print(data)
                out = model(data.float(), length)  # get net's predictions
                valid_loss = criterion(out, label)
                val, y_pred = out.max(1)  # argmax since output is a prob distribution
                acc += (label == y_pred).sum().detach().item()  # get accuracy
                n_samples += BATCH_SZ_val
                running_average_loss += valid_loss.detach().item()

        val_actual_loss = float(running_average_loss) / (len(dev_dl))
        val_losses.append(val_actual_loss)

        # Early Stopping
        if (prev_val_loss < val_actual_loss):
            stop_cnt = stop_cnt + 1
            if (stop_cnt >= patience):
                break
        else:
            stop_cnt = 0

        prev_val_loss = val_actual_loss

    # Plot train and validation losses
    print(train_losses)
    print(val_losses)
    Xaxis = list(range(epochs))
    plt.plot(Xaxis, train_losses,label = "Train loss")
    plt.plot(Xaxis, val_losses,label = "Validation loss")
    plt.legend()
    plt.show()

    # Save and load best model (CHECKPOINT)

    save_checkpoint(model,"Best_model.model",optimizer,val_actual_loss)

    #model = BasicLSTM(input_size, hidden_size, output_size, numofLayers, bidirectional,Dropout_factor)
    #criterion = torch.nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)  # feed the optimizer with the network parameters

    #val_actual_loss = load_checkpoint(model,"Best_model.model",optimizer,val_actual_loss)

    # Test model on validation data
    model.eval()  # turns off batchnorm/dropout ...
    acc = 0
    n_samples = 0
    running_average_loss = 0
    y_pred_val = []
    y_real_val = []
    with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
        for i, (data, label, length) in enumerate(dev_dl):
            # print(data)
            out = model(data.float(), length)  # get net's predictions
            valid_loss = criterion(out, label)
            val, y_pred = out.max(1)  # argmax since output is a prob distribution
            acc += (label == y_pred).sum().detach().item()  # get accuracy
            n_samples += BATCH_SZ_val
            running_average_loss += valid_loss.detach().item()
            y_pred_val = y_pred_val + y_pred.flatten().tolist()
            y_real_val = y_real_val + label.flatten().tolist()

    # Test model on test data
    model.eval()  # turns off batchnorm/dropout ...
    acc = 0
    n_samples = 0
    running_average_loss = 0
    y_pred_test = []
    y_real_test = []
    with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
        for i, (data, label, length) in enumerate(test_dl):
            # print(data)
            out = model(data.float(), length)  # get net's predictions
            valid_loss = criterion(out, label)
            val, y_pred = out.max(1)  # argmax since output is a prob distribution
            acc += (label == y_pred).sum().detach().item()  # get accuracy
            n_samples += BATCH_SZ_test
            running_average_loss += valid_loss.detach().item()

            y_pred_test = y_pred_test + y_pred.flatten().tolist()
            y_real_test = y_real_test + label.flatten().tolist()

    test_actual_loss = float(running_average_loss) / (len(test_dl))
    test_actual_acc = float((acc / n_samples) * 100)
    print("Test accuracy of best model is: {}%".format(test_actual_acc))

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    val_conf = confusion_matrix(y_real_val, y_pred_val)
    print("Confusion matrix for best model on validation data is:")
    print(val_conf)

    test_conf = confusion_matrix(y_real_test, y_pred_test)
    print("Confusion matrix for best model on test data is:")
    print(test_conf)

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Confusion matrices')
    sns.heatmap(val_conf, ax=axs[0])
    axs[0].set_title("Best model - validation set - conf. matrix")
    sns.heatmap(test_conf, ax=axs[1])
    axs[1].set_title("Best model - test set - conf. matrix")
    plt.show()