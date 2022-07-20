import torch.nn.functional as F
from dataset import *


def train_model(dataset,parameters,overfit_batch,case):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(str(device) + " - " + torch.cuda.get_device_name(0))
    output, val_size, BATCH_SZ, BATCH_SZ_val, kernel, epochs, weight_decay, lr, patience = parameters

    train_dl, dev_dl = torch_train_val_split(dataset, BATCH_SZ, BATCH_SZ_val, val_size=val_size, shuffle=True, seed=420)

    # Define the model
    model = BasicCNN(case,kernel,output)
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # feed the optimizer with the network parameters

    #Start training
    # Create batches for short training
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

                # Forward Data #
                pred = model(data.float())
                # print(pred)

                # Calculate Loss #
                train_loss = criterion(pred, label)

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

                pred = model(data.float())
                # print(pred)

                # Calculate Loss #
                train_loss = criterion(pred, label)

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

                    out = model(data.float())  # get net's predictions
                    # print(out.shape, label.shape)
                    valid_loss = criterion(out, label)
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

                    out = model(data.float())  # get net's predictions
                    # print(out.shape, label.shape)
                    valid_loss = criterion(out, label)
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
            if (stop_cnt > patience):
                break
        else:
            stop_cnt = 0

        prev_val_loss = val_actual_loss

    print("Final loss,Final Accuracy - Validation Data:")
    val_actual_acc = float((acc / n_samples) * 100)
    print(val_actual_loss, val_actual_acc)

    if (overfit_batch):
        plt.plot(list(range(epochs)),train_losses,label='Training Loss')
        plt.plot(list(range(epochs)),val_losses,label='Validation Loss')
        plt.title("Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    return model, criterion

def test_model(dataset,model,criterion):
    BATCH_SZ_test = len(dataset) // 10
    test_dl = DataLoader(dataset, batch_size=BATCH_SZ_test, shuffle=True)

    # Test model on test data
    model.eval()  # turns off batchnorm/dropout ...
    acc = 0
    n_samples = 0
    running_average_loss = 0
    y_pred_test = []
    y_real_test = []
    with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
        for i, (data, label, length) in enumerate(test_dl):
            data = data.cuda()
            label = label.cuda()
            length = length.cuda()

            # print(data)
            out = model(data.float())  # get net's predictions
            valid_loss = criterion(out, label)
            val, y_pred = out.max(1)  # argmax since output is a prob distribution
            acc += (label == y_pred).sum().detach().item()  # get accuracy
            n_samples += BATCH_SZ_test
            running_average_loss += valid_loss.detach().item()

            y_pred_test = y_pred_test + y_pred.flatten().tolist()
            y_real_test = y_real_test + label.flatten().tolist()

    print(classification_report(y_real_test, y_pred_test))

class BasicCNN(nn.Module):
    def __init__(self,case = 4,kernel = 5,output = 10):
        super().__init__()
        self.output = output
        self.l1 = nn.Conv2d(1, 1, kernel)
        self.l2 = nn.BatchNorm2d(1)
        self.l4 = nn.MaxPool2d(4)

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

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output)

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
        x = self.fc3(x)
        #print(x)

        if (self.output == 1):
            x = torch.squeeze(x)

        return x

if __name__ == "__main__":

    # Parameters
    output = 10
    val_size = 0.20
    BATCH_SZ = 100
    BATCH_SZ_val = 50
    kernel = 5
    epochs = 100
    weight_decay = 0
    lr = 1e-4
    patience = math.inf

    parameters = [output,val_size, BATCH_SZ, BATCH_SZ_val, kernel, epochs, weight_decay, lr, patience]
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=True, chroma=False,
                                 All=False)

    # Check if the model can train
    #print("Test training: \n")
    #overfit_batch = True
    #model, criterion = train_model(dataset, parameters, overfit_batch)
    #print("Model trained successfully!!")

    # Start real training
    overfit_batch = False

    ############# Train CNN (Spectrograms Only) #############
    model, criterion = train_model(dataset, parameters, overfit_batch,1)
    # Test model
    model = model.cuda()
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=False, chroma=False, All=False)
    print("Spectrograms: \n")
    test_model(dataset, model, criterion)

    """
    #############  Train CNN (Beat-synced Spectrograms Only) #############
    dataset = SpectrogramDataset(BEAT_SPECT_DIR, class_mapping=CLASS_MAPPING, train=True, chroma=False,
                                 All=False)
    model, criterion = train_model(dataset, parameters, overfit_batch,2)
    model = model.cuda()
    # Test model
    dataset = SpectrogramDataset(BEAT_SPECT_DIR, class_mapping=CLASS_MAPPING, train=False, chroma=False,
                                 All=False)
    print("Beat-Synced Spectrograms: \n")
    test_model(dataset, model, criterion)
    """

    #############  Train CNN (Chromograms Only) #############
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=True,
                                 chroma=True,
                                 All=False)
    model, criterion = train_model(dataset, parameters, overfit_batch,3)
    model = model.cuda()
    # Test model
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=False,
                                 chroma=True,
                                 All=False)
    print("Chromograms: \n")
    test_model(dataset, model, criterion)

    ############# Train CNN (Spectograms + Chromograms) #############
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=True,
                                 chroma=False,
                                 All=True)
    model, criterion = train_model(dataset, parameters, overfit_batch,4)
    model = model.cuda()
    # Test model
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=False,
                                 chroma=False,
                                 All=True)
    print("Spectograms + Chromograms: \n")
    test_model(dataset, model, criterion)