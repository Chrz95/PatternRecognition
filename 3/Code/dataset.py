import copy
import os, math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from lstm import *
#from lstm_lab import *

# HINT: Use this class mapping to merge similar classes and ignore classes that do not work very well
CLASS_MAPPING = {
    "Rock": "Rock",
    "Psych-Rock": "Rock",
    "Indie-Rock": None,
    "Post-Rock": "Rock",
    "Psych-Folk": "Folk",
    "Folk": "Folk",
    "Metal": "Metal",
    "Punk": "Metal",
    "Post-Punk": None,
    "Trip-Hop": "Trip-Hop",
    "Pop": "Pop",
    "Electronic": "Electronic",
    "Hip-Hop": "Hip-Hop",
    "Classical": "Classical",
    "Blues": "Blues",
    "Chiptune": "Electronic",
    "Jazz": "Jazz",
    "Soundtrack": None,
    "International": None,
    "Old-Time": None,
}

SPECT_DIR = "data/fma_genre_spectrograms"
BEAT_SPECT_DIR = "data/fma_genre_spectrograms_beat"

def torch_train_val_split(dataset, batch_train, batch_eval, val_size=0.2, shuffle=True, seed=420):

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    return train_loader, val_loader


def read_spectrogram(spectrogram_file, chroma=False,All = False):
    # with open(spectrogram_file, "r") as f:
    spectrogram_file = spectrogram_file.replace("beatsync.fused", "fused.full")
    spectrogram_file = spectrogram_file.replace("full.fused", "fused.full")

    spectrograms = np.load(spectrogram_file)
    # spectrograms contains a fused mel spectrogram and chromagram
    # Decompose as follows

    if (All):
        return spectrograms.T
    else:
        mel, chroma_set = spectrograms[:128], spectrograms[128:]

        if (chroma):
            return chroma_set.T
        else:
            return mel.T


class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[: self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1


class SpectrogramDataset(Dataset):
    def __init__(
        self, path, class_mapping=None, train=True, max_length=-1, regression=None,chroma = False,All = False,multi=False
    ):
        t = "train" if train else "test"
        p = os.path.join(path, t)
        self.regression = regression
        self.multi = multi
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spectrogram(os.path.join(p, f),chroma=chroma,All=All) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)): # Here the labels are encoded to integers
            if not regression:
                self.labels = np.array(
                    self.label_transformer.fit_transform(labels)
                ).astype("int64")
            else:
                self.labels = np.array(labels).astype("float64")

    def get_files_labels(self, txt, class_mapping):
        with open(txt, "r") as fd:
            lines = [l.rstrip().split("\t") for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            if self.regression:
                if (self.multi):
                    l = l[0].split(",")
                    files.append(l[0] + ".fused.full.npy")
                    labels.append([l[1],l[2],l[3]])
                    continue
                else:
                    l = l[0].split(",")
                    files.append(l[0] + ".fused.full.npy")
                    labels.append(l[self.regression])
                    continue
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            fname = l[0]
            if fname.endswith(".gz"):
                fname = ".".join(fname.split(".")[:-1])
            files.append(fname)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        length = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], length

    def __len__(self):
        return len(self.labels)

def train_model(dataset,parameters,overfit_batch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(str(device) + " - " + torch.cuda.get_device_name(0))
    val_size, BATCH_SZ, BATCH_SZ_val, hidden_size, numofLayers, epochs, bidirectional,Dropout_factor, weight_decay, lr, patience = parameters

    # Create train and validation dataloaders
    train_dl, dev_dl = torch_train_val_split(dataset, BATCH_SZ, BATCH_SZ_val, val_size=val_size, shuffle=True, seed=420)

    input_size = dataset[10][0].shape[1]
    output_size = len(set(dataset.labels))

    # Define the model
    model = BasicLSTM(input_size, hidden_size, output_size, numofLayers, bidirectional, Dropout_factor)
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # feed the optimizer with the network parameters

    #Start training
    model.train()

    # Create the batches for the fast training
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
        print("EPOCH {}".format(epoch))
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
                pred = model(data.float(), length)
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

                pred = model(data.float(), length)
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

                    out = model(data.float(), length)  # get net's predictions
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

                    out = model(data.float(), length)  # get net's predictions
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
        print("Validation loss: {}".format(val_actual_loss))
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

    # Plot training loss and validation loss for short training
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
            out = model(data.float(), length)  # get net's predictions
            valid_loss = criterion(out, label)
            val, y_pred = out.max(1)  # argmax since output is a prob distribution
            acc += (label == y_pred).sum().detach().item()  # get accuracy
            n_samples += BATCH_SZ_test
            running_average_loss += valid_loss.detach().item()

            y_pred_test = y_pred_test + y_pred.flatten().tolist()
            y_real_test = y_real_test + label.flatten().tolist()

    print(classification_report(y_real_test, y_pred_test))

if __name__ == "__main__":
    # Without mapping
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=None, train=True,chroma=False,All=False)
    #print(set(dataset.labels))

    bins = np.arange(0, max(dataset.labels) + 1.5) - 0.5

    # then you plot away
    fig, ax = plt.subplots()
    _ = ax.hist(dataset.labels, bins)
    ax.set_xticks(bins + 0.5)
    ax.set_title("Sample distribution to labels (Before mapping)")
    ax.set_xlabel("Labels")
    ax.set_ylabel("Samples per label")

    # With mapping
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=True,chroma=False,All=False)

    bins = np.arange(0, max(dataset.labels) + 1.5) - 0.5

    # then you plot away
    fig, ax = plt.subplots()
    _ = ax.hist(dataset.labels, bins)
    ax.set_xticks(bins + 0.5)
    ax.set_title("Sample distribution to labels (After mapping)")
    ax.set_xlabel("Labels")
    ax.set_ylabel("Samples per label")
    plt.show()

    #print(dataset[10])
    print(f"Input: {dataset[10][0].shape}")
    print(f"Label: {dataset[10][1]}")
    print(f"Original length: {dataset[10][2]}")

    # Parameters
    val_size = 0.20
    BATCH_SZ = 100
    BATCH_SZ_val = 50
    hidden_size = 128
    numofLayers = 3
    epochs = 200
    bidirectional = True
    Dropout_factor = 0
    weight_decay = 0
    lr = 1e-4
    patience = math.inf

    #BATCH_SZ = int(len(dataset)*(1-val_size))//10
    #BATCH_SZ_val = int(len(dataset)*val_size)//10
    parameters = [val_size, BATCH_SZ, BATCH_SZ_val, hidden_size, numofLayers, epochs, bidirectional, Dropout_factor,
                  weight_decay, lr, patience]

    # Check if the model can train
    print("Test training: \n")
    overfit_batch = True
    model, criterion = train_model(dataset,parameters,overfit_batch)
    print("Model trained successfully!!")

    # Start real training
    overfit_batch = False

    ############# Train LSTM (Spectrograms Only) #############
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=True, chroma=False,
                                 All=False)

    model, criterion = train_model(dataset, parameters,overfit_batch)
    # Test model
    model = model.cuda()
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=False, chroma=False,All=False)
    print("Spectrograms: \n")
    test_model(dataset, model, criterion)

    #############  Train LSTM (Beat-synced Spectrograms Only) #############
    dataset = SpectrogramDataset(BEAT_SPECT_DIR, class_mapping=CLASS_MAPPING, train=True, chroma=False,
                                 All=False)
    model, criterion = train_model(dataset, parameters,overfit_batch)
    model = model.cuda()
    # Test model
    dataset = SpectrogramDataset(BEAT_SPECT_DIR, class_mapping=CLASS_MAPPING, train=False, chroma=False,
                                 All=False)
    print("Beat-Synced Spectrograms: \n")
    test_model(dataset, model, criterion)

    #############  Train LSTM (Chromograms Only) #############
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=True,
                                 chroma=True,
                                 All=False)
    model, criterion = train_model(dataset, parameters,overfit_batch)
    model = model.cuda()
    # Test model
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=False,
                                 chroma=True,
                                 All=False)
    print("Chromograms: \n")
    test_model(dataset, model, criterion)

    ############# Train LSTM (Spectograms + Chromograms) #############
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=True,
                                 chroma=False,
                                 All=True)
    model, criterion = train_model(dataset, parameters,overfit_batch)
    model = model.cuda()
    # Test model
    dataset = SpectrogramDataset(SPECT_DIR, class_mapping=CLASS_MAPPING, train=False,
                                 chroma=False,
                                 All=True)
    print("Spectograms + Chromograms: \n")
    test_model(dataset, model, criterion)