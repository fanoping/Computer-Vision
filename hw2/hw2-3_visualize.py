from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from scipy.misc import imread
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import numpy as np
import torch
import os


class MnistDataset(Dataset):
    def __init__(self, image_directory, mode):
        assert(mode == 'train' or mode == 'valid')
        self.image_directory = image_directory
        self.mode = mode

        self.image = []
        self.label = []
        self.__read()

    def __read(self):
        print("Reading {} Image...".format(self.mode.title()))

        base = os.path.join(self.image_directory, self.mode)
        files = [classes for classes in sorted(os.listdir(base)) if classes.startswith('class')]

        for file in files:
            for imagefile in sorted(os.listdir(os.path.join(base, file))):
                image = imread(os.path.join(base, file, imagefile))
                image = image / 255.0
                label = int(file[-1])

                self.image.append(np.expand_dims(image, axis=0))
                self.label.append(label)
        print("{} Image Loaded!".format(self.mode.title()))

    def __getitem__(self, item):
        return self.image[item], self.label[item]

    def __len__(self):
        return len(self.image)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, image):
        feature = self.conv(image)
        output = self.classifier(feature.view(feature.size(0), -1))
        return output


def tsne(image, label, idx):
    image = image.reshape(image.shape[0], -1)
    t = TSNE(n_components=2, random_state=30, verbose=1, n_iter=2500)
    embedded = t.fit_transform(image)

    plt.figure(figsize=(12, 12))
    colors = iter(cm.rainbow(np.linspace(0, 1, 10)))
    for lab in np.unique(label):
        c = next(colors)
        mask = label == lab
        plt.scatter(embedded[mask, 0], embedded[mask, 1], s=30, c=c, label=lab)


    plt.legend(loc="best")
    plt.title('t-SNE for {} convolution layer'.format(idx))
    plt.savefig("tsne_{}_conv.png".format(idx))


def main():
    """
    checkpoint = torch.load('checkpoints/epoch100_checkpoint.pth.tar', map_location=lambda storage, loc: storage)

    print("Plot Learning Curve...")
    epoch = range(1, 101)
    acc = checkpoint['accuracy']
    val_acc = checkpoint['val_accuracy']
    loss = checkpoint['loss']
    val_loss = checkpoint['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Train/Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epoch, loss, 'b', label='train loss')
    plt.plot(epoch, val_loss, 'r', label='valid loss')
    plt.legend(loc="best")

    plt.subplot(122)
    plt.title('Train/Validation Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(epoch, acc, 'b', label='train accuracy')
    plt.plot(epoch, val_acc, 'r', label='valid accuracy')
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig("curve.png")
    """

    # tsne result
    checkpoint = torch.load('checkpoints/best_checkpoint.pth.tar', map_location=lambda storage, loc: storage)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CNN().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    data = MnistDataset('hw2_data/hw2-3_data', 'valid')
    image_100, label_100 = [], []
    for i in range(10):
        for j in range(100):
            image_100.append(data.image[1000 * i + j])
            label_100.append(data.label[1000 * i + j])
    image_100, label_100 = np.array(image_100), np.array(label_100)
    image_100 = torch.tensor(image_100, dtype=torch.float, device=device)
    for i, layer in enumerate(model.conv):
        image_100 = layer(image_100)
        if i == 0:
            tsne(image_100.cpu().detach().numpy(), label_100, "first")
        if i == 3:
            tsne(image_100.cpu().detach().numpy(), label_100, "last")

if __name__ == '__main__':
    main()
