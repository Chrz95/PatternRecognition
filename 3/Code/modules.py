import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math,os
import torch.optim as optim
from dataset import SpectrogramDataset,torch_train_val_split,SPECT_DIR,CLASS_MAPPING
from torch.utils.data import DataLoader
from feeling import MULTI_SPECT_DIR , torch_train_val_test_split

class LSTMBackbone(nn.Module):
    def __init__(self, input_dim, rnn_size, num_layers, bidirectional=False, Dropout_factor=0.2):
        super(LSTMBackbone, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=rnn_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(Dropout_factor)

    def forward(self, x, lengths):
        """
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """

        out, _ = self.lstm(x.float())
        last_outputs = self.last_timestep(out, lengths, self.bidirectional)
        out = self.dropout(last_outputs)
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
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0), outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()


class CNNBackbone(nn.Module):
    def __init__(self,case = 4,kernel = 5,output = 10):
        super().__init__()
        self.output = output
        self.feature_size = 84
        self.l1 = nn.Conv2d(1, 1, kernel)
        self.l2 = nn.BatchNorm2d(1)
        self.l4 = nn.MaxPool2d(4, 4)

        if (case == 1):
            self.fc1 = nn.Linear(1* 322* 31, 120)
        elif(case == 2):
            self.fc1 = nn.Linear(1 * 62 * 62, 120)
        elif (case == 3):
            self.fc1 = nn.Linear(1 * 322 * 2, 120)
        elif (case == 4):
            self.fc1 = nn.Linear(1 * 322 * 34, 120)
        else:
            self.fc1 = nn.Linear(1 * 323 * 35, 120)

        self.fc2 = nn.Linear(120, self.feature_size)

    def forward(self, x):
        x = torch.unsqueeze(x,1).float()
        x = self.l1(x)
        x = self.l2(x)
        x = self.l4(F.relu(x))
        Shape = list(x.shape)
        #print(Shape)
        x = x.view(-1, Shape[2] * Shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def load_backbone_from_checkpoint(model, checkpoint_path):
    if (checkpoint_path == None):
        print("Not a valid load path!")
        return

    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['model_state_dict'])

    return model


def save_backbone_to_checkpoint(model,optimizer, checkpoint_path = ""):
    if (checkpoint_path == None):
        print("Not a valid save path!")
        return

    if (isinstance(model,Classifier)):
        checkpoint_path = "classifier.pt"
    else:
        checkpoint_path = "regressor.pt"

    torch.save({'model_state_dict': model.backbone.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

def train_lstm_clf(train_dl,dev_dl,parameters,overfit_batch):

    input_size, val_size, BATCH_SZ, BATCH_SZ_val, hidden_size, numofLayers, epochs, bidirectional,Dropout_factor, weight_decay, lr, patience = parameters
    output_size = 1

    # Define the model
    model = Classifier(LSTMBackbone(input_size, hidden_size, numofLayers, bidirectional, Dropout_factor),10)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # feed the optimizer with the network parameters

    #Start training
    model.train()
    if (overfit_batch):
        MyTrainBatches = []
        for i, (data, label, length) in enumerate(train_dl):
            MyTrainBatches.append((data, label, length))
            if (i == 3):
                break

    if (overfit_batch):
        MyValBatches = []
        for i, (data, label, length) in enumerate(dev_dl):
            MyValBatches.append((data, label, length))
            if (i == 3):
                break

    # Train model
    train_losses = []
    val_losses = []
    stop_cnt = 0
    prev_val_loss = math.inf
    min_val_loss = math.inf
    for epoch in range(epochs):
        model.train()
        #print("EPOCH {}".format(epoch))
        # sys.stdout.flush()

        # Train model with training samples
        running_average_loss = 0
        if (overfit_batch):  # Train for a few batches
            for i,Batch  in enumerate(MyTrainBatches):

                data, label, length = Batch

                data = data.cuda()
                label = label.cuda()
                length = length.cuda()

                train_loss , pred = model(data.float(),label, length)

                # Initialize Optimizer, Back Propagation and Update #
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                running_average_loss += train_loss.detach().item()
        else:
            for i, (data, label, length) in enumerate(train_dl):
                # Forward Data #
                data = data.cuda()
                label = label.cuda()
                length = length.cuda()

                train_loss , pred = model(data.float(),label, length)

                # Initialize Optimizer, Back Propagation and Update #
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                running_average_loss += train_loss.detach().item()

        if (overfit_batch):  # Train for a few batches
            train_actual_loss = float(running_average_loss) / len(MyTrainBatches)
        else:
            train_actual_loss = float(running_average_loss) / len(train_dl)
        train_losses.append(train_actual_loss)
        #print("Training loss: {}".format(train_actual_loss))

        # Eval model on validation data
        model.eval()  # turns off batchnorm/dropout ...
        acc = 0
        n_samples = 0
        running_average_loss = 0
        with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
            if (overfit_batch):  # Train for a few batches
                for i, (data, label, length) in enumerate(MyValBatches):
                    # print(data)
                    data = data.cuda()
                    label = label.cuda()
                    length = length.cuda()

                    valid_loss, out = model(data.float(), label, length)
                    val, y_pred = out.max(1)  # argmax since output is a prob distribution
                    acc += (label == y_pred).sum().detach().item()  # get accuracy
                    n_samples += BATCH_SZ_val
                    running_average_loss += valid_loss.detach().item()
            else:
                for i, (data, label, length) in enumerate(dev_dl):
                    # print(data)
                    data = data.cuda()
                    label = label.cuda()
                    length = length.cuda()

                    valid_loss, out = model(data.float(), label, length)
                    val, y_pred = out.max(1)  # argmax since output is a prob distribution
                    acc += (label == y_pred).sum().detach().item()  # get accuracy
                    n_samples += BATCH_SZ_val
                    running_average_loss += valid_loss.detach().item()

        val_actual_acc = float((acc / n_samples) * 100)
        if (overfit_batch):  # Train for a few batches
            val_actual_loss = float(running_average_loss) / (len(MyValBatches))
        else:
            val_actual_loss = float(running_average_loss) / (len(dev_dl))
        val_losses.append(val_actual_loss)
        #print("Validation loss: {}".format(val_actual_loss))
        #print("Validation Accuracy: {}%".format(val_actual_acc))

        if (not overfit_batch):
            if (train_actual_loss < 0.1):
                break

        # Early Stopping
        if (prev_val_loss < val_actual_loss):
            stop_cnt = stop_cnt + 1
            if (stop_cnt >= patience):
                break
        else:
            stop_cnt = 0

        # Each time we encounter a minimum validation loss, save model
        if (val_actual_loss < min_val_loss):
            min_val_loss = val_actual_loss
            save_backbone_to_checkpoint(model,optimizer)
            print("SAVED!")

        prev_val_loss = val_actual_loss

    print("Final loss,Final Accuracy - Validation Data:")
    val_actual_acc = float((acc / n_samples) * 100)
    print(val_actual_loss, val_actual_acc)

    return model

class Classifier(nn.Module):
    def __init__(self, backbone, num_classes, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        num_classes (int): The number of classes
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Classifier, self).__init__()
        self.backbone = backbone # An LSTMBackbone or CNNBackbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()  # Loss function for classification

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        logits = self.output_layer(feats)
        loss = self.criterion(logits, targets)
        return loss, logits


class Regressor(nn.Module):
    def __init__(self, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        num_classes (int): The number of classes
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Regressor, self).__init__()
        self.backbone = backbone  # An LSTMBackbone or CNNBackbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, 1)
        self.criterion = torch.nn.MSELoss()  # Loss function for regression

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        logits = self.output_layer(feats)
        loss = self.criterion(logits, targets)
        return loss, logits


class MultitaskRegressor(nn.Module):
    def __init__(self, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        num_classes (int): The number of classes
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(MultitaskRegressor, self).__init__()
        self.backbone = backbone  # An LSTMBackbone or CNNBackbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)

        self.output_layer_valence = nn.Linear(self.backbone.feature_size, 1)
        self.output_layer_energy = nn.Linear(self.backbone.feature_size, 1)
        self.output_layer_dance = nn.Linear(self.backbone.feature_size, 1)

        self.criterion = MultiTaskLossWrapper()  # Loss function for regression

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)

        valence = self.output_layer_valence(feats)
        energy = self.output_layer_energy(feats)
        dance = self.output_layer_dance(feats)
        logits = [valence,energy,dance]

        targets = torch.transpose(targets, 0, 1)

        loss = self.criterion(logits, targets[0],targets[1],targets[2])
        return loss, logits

# Multi-task loss function
class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super(MultiTaskLossWrapper, self).__init__()

    def forward(self, preds, valence, energy, danceability):
        mse1,mse2,mse3= torch.nn.MSELoss(), torch.nn.MSELoss(),torch.nn.MSELoss()

        valence = valence.unsqueeze(1)
        energy = energy.unsqueeze(1)
        danceability = danceability.unsqueeze(1)

        #print(preds[0].shape,valence.shape)
        # One MSELoss function for each axis
        loss0 = mse1(preds[0], valence)
        loss1 = mse2(preds[1], energy)
        loss2 = mse3(preds[2], danceability)
        #print("Valence: {}, Energy:{} , Danceability:{}".format(loss0,loss1,loss2))

        w0,w1,w2 = 1,1,1

        loss0 = w0 * loss0
        loss1 = w1 * loss1
        loss2 = w2 * loss2

        return loss0 + loss1 + loss2

def train_lstm_rgr(train_dl, dev_dl,parameters,overfit_batch,multi= False):
    input_size, val_size, BATCH_SZ, BATCH_SZ_val, hidden_size, numofLayers, epochs, bidirectional, Dropout_factor, weight_decay, lr, patience = parameters

    # Define the model
    backbone = LSTMBackbone(input_size, hidden_size, numofLayers, bidirectional, Dropout_factor)

    if (multi):
        model = MultitaskRegressor(backbone)
    else:
        model = Regressor(backbone, "classifier.pt")
        print("Loaded backbone from checkpoint!")

    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)  # feed the optimizer with the network parameters

    #Start training
    model.train()
    if (overfit_batch):
        MyTrainBatches = []
        for i, (data, label, length) in enumerate(train_dl):
            MyTrainBatches.append((data, label, length))
            if (i == 3):
                break

    if (overfit_batch):
        MyValBatches = []
        for i, (data, label, length) in enumerate(dev_dl):
            MyValBatches.append((data, label, length))
            if (i == 3):
                break

    # Train model
    train_losses = []
    val_losses = []
    final_epochs = []
    stop_cnt = 0
    prev_val_loss = math.inf
    for epoch in range(epochs):
        model.train()
        print("EPOCH {}".format(epoch))
        # sys.stdout.flush()

        # Train model with training samples
        running_average_loss = 0
        if (overfit_batch):  # Train for a few batches
            for i,Batch in enumerate(MyTrainBatches):

                data, label, length = Batch

                data = data.cuda()
                label = label.cuda()
                length = length.cuda()
                #print(label)

                train_loss, pred = model(data.float(), label.float(), length)

                # Initialize Optimizer, Back Propagation and Update #
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                running_average_loss += train_loss.detach().item()
        else:
            for i, (data, label, length) in enumerate(train_dl):
                # Forward Data #
                data = data.cuda()
                label = label.cuda()
                length = length.cuda()

                train_loss,pred = model(data.float(),label.float(), length)

                # Initialize Optimizer, Back Propagation and Update #
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                running_average_loss += train_loss.detach().item()

        if (overfit_batch):  # Train for a few batches
            train_actual_loss = float(running_average_loss) / len(MyTrainBatches)
        else:
            train_actual_loss = float(running_average_loss) / len(train_dl)
        train_losses.append(train_actual_loss)
        print("Training loss: {}".format(train_actual_loss))

        # Eval model on validation data
        model.eval()  # turns off batchnorm/dropout ...
        acc = 0
        n_samples = 0
        running_average_loss = 0
        with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
            if (overfit_batch):  # Train for a few batches
                for i, (data, label, length) in enumerate(MyValBatches):
                    # print(data)
                    data = data.cuda()
                    label = label.cuda()
                    length = length.cuda()

                    valid_loss, out = model(data.float(), label.float(), length)

                    n_samples += BATCH_SZ_val
                    running_average_loss += valid_loss.detach().item()
            else:
                for i, (data, label, length) in enumerate(dev_dl):
                    # print(data)
                    data = data.cuda()
                    label = label.cuda()
                    length = length.cuda()

                    valid_loss, out = model(data.float(), label.float(), length)

                    n_samples += BATCH_SZ_val
                    running_average_loss += valid_loss.detach().item()

        val_actual_acc = float((acc / n_samples) * 100)
        if (overfit_batch):  # Train for a few batches
            val_actual_loss = float(running_average_loss) / (len(MyValBatches))
        else:
            val_actual_loss = float(running_average_loss) / (len(dev_dl))
        val_losses.append(val_actual_loss)
        print("Validation loss: {}".format(val_actual_loss))

        final_epochs.append(epoch)

        # Early Stopping
        if (prev_val_loss < val_actual_loss):
            stop_cnt = stop_cnt + 1
            if (stop_cnt >= patience):
                break
        else:
            stop_cnt = 0

        prev_val_loss = val_actual_loss

    return model

def test_lstm(test_dl,model,multi= False):
    from scipy import stats

    # Test model on test data
    model.eval()  # turns off batchnorm/dropout ...
    n_samples = 0
    running_average_loss = 0
    if (not multi):
        total_pred = []
        total_data = []
    else:
        pred1 = []
        pred2 = []
        pred3 = []
        label1 = []
        label2 = []
        label3 = []

    with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
        for i, (data, label, length) in enumerate(test_dl):
            data = data.cuda()
            label = label.cuda()
            length = length.cuda()

            test_loss, out = model(data.float(), label.float(), length)
            n_samples += BATCH_SZ_val
            running_average_loss += test_loss.detach().item()

            if (not multi):
                total_pred += out.flatten().tolist()
                total_data += label.flatten().tolist()
            else:
                labels = np.array(label.tolist())

                label1 += [row[0] for row in labels]
                label2 += [row[1] for row in labels]
                label3 += [row[2] for row in labels]

                out1 = torch.transpose(out[0], 0, 1)
                out2 = torch.transpose(out[1], 0, 1)
                out3 = torch.transpose(out[2], 0, 1)
                pred1 += out1.flatten().tolist()
                pred2 += out2.flatten().tolist()
                pred3 += out3.flatten().tolist()

    if (not multi):
        print("LSTM: Spearman correlation is {}".format(stats.spearmanr(total_pred, total_data)))
    else:
        #print(pred1, label1)
        #print(len(pred1),len(label1))
        print("LSTM (Valence) Spearman correlation is {}".format(stats.spearmanr(pred1, label1)))
        print("LSTM (Energy) Spearman correlation is {}".format(stats.spearmanr(pred2, label2)))
        print("LSTM (Danceability) Spearman correlation is {}".format(stats.spearmanr(pred3, label3)))

if __name__ == "__main__":
    # Parameters
    val_size = 0.20
    BATCH_SZ = 100
    BATCH_SZ_val = 50
    hidden_size = 128
    numofLayers = 3
    epochs = 100
    bidirectional = True
    Dropout_factor = 0
    weight_decay = 0
    lr = 1e-4
    patience = 2
    overfit_batch = False

    ############# Train LSTM (Spectrograms Only) #############
    # print("############### LSTM ###############")
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=True, chroma=False, All=False)

    parameters_lstm = [dataset[10][0].shape[1], val_size, BATCH_SZ, BATCH_SZ_val, hidden_size, numofLayers, epochs, bidirectional,
                       Dropout_factor, weight_decay, lr, patience]
    

    train_dl, dev_dl = torch_train_val_split(dataset, BATCH_SZ, BATCH_SZ_val, val_size=val_size,
                                                           shuffle=True, seed=420)

    print("Training classifier on fma_genre_spectrograms")
    model = train_lstm_clf(train_dl, dev_dl, parameters_lstm, overfit_batch)
    model = model.cuda()

    axis = 3
    dataset = SpectrogramDataset(MULTI_SPECT_DIR, class_mapping=None, train=True, chroma=False, All=False, regression=axis)
    train_dl, dev_dl, test_dl = torch_train_val_test_split(dataset, BATCH_SZ, BATCH_SZ_val, val_size=val_size,
                                                           shuffle=True, seed=420)
    # Define the model
    epochs = 50
    patience = 5
    parameters_lstm = [dataset[10][0].shape[1], val_size, BATCH_SZ, BATCH_SZ_val, hidden_size, numofLayers, epochs,
                       bidirectional,
                       Dropout_factor, weight_decay, lr, patience]

    model = train_lstm_rgr(train_dl, dev_dl, parameters_lstm, overfit_batch)
    model = model.cuda()
    print("Testing regressor on multitask dataset ({})".format(axis))
    test_lstm(test_dl,model)

    ###### STEP 10 #######
    dataset = SpectrogramDataset(MULTI_SPECT_DIR, class_mapping=None, train=True, chroma=False, All=False,regression=1,multi = True)
    train_dl, dev_dl, test_dl = torch_train_val_test_split(dataset, BATCH_SZ, BATCH_SZ_val, val_size=val_size,
                                                           shuffle=True, seed=420)
    # Define the model
    parameters_lstm = [dataset[10][0].shape[1], val_size, BATCH_SZ, BATCH_SZ_val, hidden_size, numofLayers, epochs,
                       bidirectional,
                       Dropout_factor, weight_decay, lr, patience]

    model = train_lstm_rgr(train_dl, dev_dl, parameters_lstm, overfit_batch,multi = True)
    model = model.cuda()
    test_lstm(test_dl, model,multi = True)