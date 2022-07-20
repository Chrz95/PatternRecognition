import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

if __name__ == "__main__":
    # sampling freq
    fs = 1000

    # message freq
    fm = 40

    # discrete frequency
    w = 2 * np.pi * fm / fs

    # time period
    N = 1000

    time = np.arange(0, N, 1)
    amplitude = np.sin(w * time)

    X = []
    for i in range(0, len(amplitude), 10):
        X.append(list(amplitude[i:i + 10]))

    plt.figure(3)
    amplitude = np.cos(w * time)

    y = []
    for i in range(0, len(amplitude), 10):
        y.append(list(amplitude[i:i + 10]))

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.30, random_state=42,shuffle=False)
    X_train= np.array(X_train)
    X_test= np.array(X_test)
    y_train= np.array(y_train)
    y_test = np.array(y_test)

    # Train our model
    num_epochs = 200
    lr = 0.01

    input_size = 10
    hidden_size = 2
    num_layers = 2
    num_classes = 1
    output_size = 10

    model = RNN(input_size, hidden_size, num_layers, output_size)

    criterion = torch.nn.MSELoss()

    # Optimizer #
    opt = 1
    if (opt == 1):
        optimizer = optim.Adam(model.parameters(), lr=lr)  # feed the optimizer with the network parameters
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)  # feed the optimizer with the network parameters

    train_data = DigitsDataset(X_train, y_train)
    test_data = DigitsDataset(X_test, y_test)

    BATCH_SZ = N//10
    BATCH_SZ_test = len(test_data)//10

    train_dl = DataLoader(train_data, batch_size=BATCH_SZ)
    test_dl = DataLoader(test_data, batch_size=BATCH_SZ_test)

    # Train model
    for epoch in range(num_epochs):
        running_average_loss = 0
        for i, (data, label) in enumerate(train_dl):
            # Forward Data #
            #print(data.unsqueeze(0).float().shape)
            data = data.unsqueeze(1).float()
            label = label.unsqueeze(1).float()
            pred = model(data)

            #print(pred.shape)
            pred = pred.unsqueeze(1).float()
            # Calculate Loss #
            train_loss = criterion(pred.float(), label)

            """
            for j in range(len(pred)):
                print(pred[j],label[j])
                print("==========================")
            """

            # Initialize Optimizer, Back Propagation and Update #
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            running_average_loss += train_loss.detach().item()
            print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i, float(running_average_loss) / (i + 1)))

    # Test model
    model.eval()
    acc = 0
    n_samples = 0
    all_predictions = []
    real = []
    with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
        for i, data in enumerate(test_dl):
            X_batch, y_batch = data  # test data and labels
            X_batch = X_batch.unsqueeze(1).float()
            out = model(X_batch)  # get net's predictions
            real.append(y_batch.tolist())
            all_predictions.append(out.tolist())


    all_predictions = [item for sublist in all_predictions for item in sublist]
    all_predictions = [item for sublist in all_predictions for item in sublist]
    real = [item for sublist in real for item in sublist]
    real = [item for sublist in real for item in sublist]
    test_time = np.arange(0, len(real), 1)

    plt.figure(2)

    plt.plot(test_time, real, label='Cos')
    plt.plot(test_time, all_predictions, label="Prediction")
    plt.legend()
    plt.show()
    plt.savefig("plots/TorchPractice.png",dpi = 700)

