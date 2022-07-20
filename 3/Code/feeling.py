from convolution import BasicCNN
from dataset import *

MULTI_SPECT_DIR = "data/multitask_dataset"

# Split dataset into train, val AND test
def torch_train_val_test_split(dataset, batch_train, batch_eval, val_size=0.2, shuffle=True, seed=420):

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices, test_indices = np.split(indices, [int(len(indices) * 0.8), int(len(indices) * 0.9)])

    #print(sorted(train_indices))
    #print(sorted(val_indices))
    #print(sorted(test_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_eval, sampler=test_sampler)
    return train_loader, val_loader, test_loader


def train_lstm(train_dl, dev_dl,parameters,overfit_batch):

    val_size, BATCH_SZ, BATCH_SZ_val, hidden_size, numofLayers, epochs, bidirectional,Dropout_factor, weight_decay, lr, patience = parameters

    input_size = dataset[10][0].shape[1]
    output_size = 1

    # Define the model
    model = BasicLSTM(input_size, hidden_size, output_size, numofLayers, bidirectional, Dropout_factor)
    model = model.cuda()
    criterion = torch.nn.MSELoss()
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
    final_epochs = []
    stop_cnt = 0
    prev_val_loss = math.inf
    for epoch in range(epochs):
        model.train()
        #print("EPOCH {}".format(epoch))
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

                # Forward Data #
                pred = model(data.float(), length)
                # print(pred)

                # Calculate Loss #
                label = label.unsqueeze(1)

                """
                for i in range(len(pred.float())):
                    print("Train:{}".format(abs(pred.float().flatten().tolist()[i] -
                                               label.float().flatten().tolist()[i])))
                """

                train_loss = criterion(pred.float(), label.float())

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

                pred = model(data.float(), length)
                # print(pred)

                # Calculate Loss #
                label = label.unsqueeze(1)

                train_loss = criterion(pred.float(), label.float())

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

                    out = model(data.float(), length)  # get net's predictions
                    # print(out.shape, label.shape)
                    label = label.unsqueeze(1)
                    """
                    for i in range(len(out.float())):
                        print("Val:{}".format(abs(out.float().flatten().tolist()[i]-
                                                   label.float().flatten().tolist()[i])))
                    """
                    valid_loss = criterion(out, label)
                    n_samples += BATCH_SZ_val
                    running_average_loss += valid_loss.detach().item()
            else:
                for i, (data, label, length) in enumerate(dev_dl):
                    # print(data)
                    data = data.cuda()
                    label = label.cuda()
                    length = length.cuda()

                    out = model(data.float(), length)  # get net's predictions
                    # print(out.shape, label.shape)
                    label = label.unsqueeze(1)
                    valid_loss = criterion(out, label)
                    n_samples += BATCH_SZ_val
                    running_average_loss += valid_loss.detach().item()

        val_actual_acc = float((acc / n_samples) * 100)
        if (overfit_batch):  # Train for a few batches
            val_actual_loss = float(running_average_loss) / (len(MyValBatches))
        else:
            val_actual_loss = float(running_average_loss) / (len(dev_dl))
        val_losses.append(val_actual_loss)
        #print("Validation loss: {}".format(val_actual_loss))

        final_epochs.append(epoch)

        # Early Stopping
        if (prev_val_loss < val_actual_loss):
            stop_cnt = stop_cnt + 1
            if (stop_cnt > patience):
                break
        else:
            stop_cnt = 0

        prev_val_loss = val_actual_loss

    if (overfit_batch):
        plt.plot(final_epochs,train_losses,label='Training Loss')
        plt.plot(final_epochs,val_losses,label='Validation Loss')
        plt.title("Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    return model, criterion

def test_lstm(test_dl,model,criterion):
    from scipy import stats

    # Test model on test data
    model.eval()  # turns off batchnorm/dropout ...
    n_samples = 0
    running_average_loss = 0
    total_pred = []
    total_data = []
    with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
        for i, (data, label, length) in enumerate(test_dl):
            data = data.cuda()
            label = label.cuda()
            length = length.cuda()

            # print(data)
            out = model(data.float(), length)  # get net's predictions
            label = label.unsqueeze(1)
            valid_loss = criterion(out, label)
            n_samples += BATCH_SZ_val
            running_average_loss += valid_loss.detach().item()

            total_pred += out.flatten().tolist()
            total_data += label.flatten().tolist()

    print("LSTM: Spearman correlation is {}".format(stats.spearmanr(total_pred, total_data)))

def train_CNN(train_dl, dev_dl ,parameters,overfit_batch):

    output, val_size, BATCH_SZ, BATCH_SZ_val, kernel, epochs, weight_decay, lr, patience = parameters

    # Define the model
    model = BasicCNN(4,kernel,output)
    model = model.cuda()
    criterion = torch.nn.MSELoss()
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
    final_epochs = []
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
                train_loss = criterion(pred.float(), label.float())

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

                # Calculate Loss
                train_loss = criterion(pred.float(), label.float())

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
                    n_samples += BATCH_SZ_val
                    running_average_loss += valid_loss.detach().item()

        val_actual_acc = float((acc / n_samples) * 100)
        if (overfit_batch):  # Train for a few batches
            val_actual_loss = float(running_average_loss) / (len(MyValBatches))
        else:
            val_actual_loss = float(running_average_loss) / (len(dev_dl))
        val_losses.append(val_actual_loss)
        #print("Validation loss: {}".format(val_actual_loss))

        final_epochs.append(epoch)

        # Early Stopping
        if (prev_val_loss < val_actual_loss):
            stop_cnt = stop_cnt + 1
            if (stop_cnt > patience):
                break
        else:
            stop_cnt = 0

        prev_val_loss = val_actual_loss

    if (overfit_batch):
        plt.plot(final_epochs,train_losses,label='Training Loss')
        plt.plot(final_epochs,val_losses,label='Validation Loss')
        plt.title("Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    return model, criterion

def test_CNN(test_dl,model,criterion):
    from scipy import stats

    # Test model on test data
    model.eval()  # turns off batchnorm/dropout ...
    n_samples = 0
    running_average_loss = 0
    total_pred = []
    total_data = []
    with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
        for i, (data, label, length) in enumerate(test_dl):
            data = data.cuda()
            label = label.cuda()
            length = length.cuda()

            # print(data)
            out = model(data.float())  # get net's predictions
            valid_loss = criterion(out, label)
            n_samples += BATCH_SZ_val
            running_average_loss += valid_loss.detach().item()
            total_pred += out.flatten().tolist()
            total_data += label.flatten().tolist()

    print("CNN: Spearman correlation is {}".format(stats.spearmanr(total_pred, total_data)))

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

    output = 1
    kernel = 5

    parameters_lstm = [val_size, BATCH_SZ, BATCH_SZ_val, hidden_size, numofLayers, epochs, bidirectional, Dropout_factor, weight_decay, lr, patience]
    parameters_cnn = [output, val_size, BATCH_SZ, BATCH_SZ_val, kernel, epochs, weight_decay, lr, patience]
    overfit_batch = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(str(device) + " - " + torch.cuda.get_device_name(0))

    for i in range(1, 4):
        print("############## Label {} ################".format(i))

        ############# Train LSTM (Spectrograms Only) #############
        # print("############### LSTM ###############")
        dataset = SpectrogramDataset(MULTI_SPECT_DIR, class_mapping=None, train=True, chroma=False, All=False, regression=i)
        train_dl, dev_dl, test_dl = torch_train_val_test_split(dataset, BATCH_SZ, BATCH_SZ_val, val_size=val_size,
                                                               shuffle=True, seed=420)

        model, criterion = train_lstm(train_dl, dev_dl, parameters_lstm, overfit_batch)
        # Test model
        model = model.cuda()
        test_lstm(test_dl, model, criterion)

        ############# Train CNN (Spectrograms + Chromograms) #############
        dataset = SpectrogramDataset(MULTI_SPECT_DIR, class_mapping=None, train=True, chroma=False, All=True, regression=i)
        train_dl, dev_dl, test_dl = torch_train_val_test_split(dataset, BATCH_SZ, BATCH_SZ_val, val_size=val_size,
                                                               shuffle=True, seed=420)
        # print("\n############### CNN ###############")
        model, criterion = train_CNN(train_dl, dev_dl, parameters_cnn, overfit_batch)
        # Test model
        model = model.cuda()
        test_CNN(test_dl, model, criterion)