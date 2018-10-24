from torch.utils.data import Dataset, DataLoader
from scipy.misc import imread
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import sys
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


class Trainer:
    def __init__(self, directory, epoch, batch_size):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using CUDA") if torch.cuda.is_available() else print("Using CPU")

        self.epoch = epoch
        self.batch_size = batch_size

        self.train_dataset = MnistDataset(directory, 'train')
        self.valid_dataset = MnistDataset(directory, 'valid')
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           shuffle=True,
                                           batch_size=batch_size)
        self.valid_dataloader = DataLoader(dataset=self.valid_dataset,
                                           shuffle=False,
                                           batch_size=len(self.valid_dataset))

        # model description
        self.__build_model()

        # checkpoint data
        self.loss_list, self.acc_list = [], []
        self.val_loss_list, self.val_acc_list = [], []
        self.max_acc = 0

    def __build_model(self):
        self.model = CNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.0001)

    def train(self, valid=True):
        for epoch in range(1, self.epoch+1):
            self.model.train()
            total_loss, total_acc = 0, 0
            for batch_idx, (image, label) in enumerate(self.train_dataloader):
                image = torch.tensor(image, dtype=torch.float, device=self.device)
                label = torch.tensor(label, dtype=torch.long, device=self.device)

                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                result = torch.max(output, dim=1)[1]
                accuracy = np.mean((result == label).cpu().data.numpy())

                total_loss += loss.item()
                total_acc += accuracy
                print('Epoch: {}/{} [{}/{} ({:.0f}%)] loss: {:.6f}, acc: {:.6f}'.format(
                       epoch,
                       self.epoch,
                       (batch_idx + 1) * self.train_dataloader.batch_size,
                       len(self.train_dataset),
                       100.0 * (batch_idx + 1) * self.train_dataloader.batch_size / len(self.train_dataset),
                       loss.item(),
                       accuracy
                ), end='\r')
                sys.stdout.write('\033[K')
            print("Epoch: {}/{} loss:{:.6f} acc:{:.6f}".format(epoch,
                                                               self.epoch,
                                                               total_loss / len(self.train_dataloader),
                                                               total_acc / len(self.train_dataloader)), end=' ')

            self.loss_list.append(total_loss / len(self.train_dataloader))
            self.acc_list.append(total_acc / len(self.train_dataloader))

            if valid:
                _, val_acc = self.__valid()
                self.__save_checkpoint(epoch, val_acc)
            else:
                print()
                self.__save_checkpoint(epoch, total_acc / len(self.train_dataloader))

    def __valid(self):
        with torch.no_grad():
            self.model.eval()
            total_loss, total_acc = 0, 0
            for batch_idx, (image, label) in enumerate(self.valid_dataloader):
                image = torch.tensor(image, dtype=torch.float, device=self.device)
                label = torch.tensor(label, dtype=torch.long, device=self.device)

                output = self.model(image)
                loss = self.criterion(output, label)

                result = torch.max(output, dim=1)[1]
                accuracy = np.mean((result == label).cpu().data.numpy())

                total_loss += loss.item()
                total_acc += accuracy

            print('val_loss: {:.6f} val_acc: {:.6f}'.format(total_loss / len(self.valid_dataloader),
                                                            total_acc / len(self.valid_dataloader)))
            self.val_loss_list.append(total_loss / len(self.train_dataloader))
            self.val_acc_list.append(total_acc / len(self.train_dataloader))

        return total_loss / len(self.valid_dataloader), total_acc / len(self.valid_dataloader)

    def __save_checkpoint(self, epoch, current_acc=None):
        state = {
            'model': 'CNN',
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_list,
            'accuracy': self.acc_list,
            'val_loss': self.val_loss_list,
            'val_accuracy': self.val_acc_list
        }

        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        filename = os.path.join("checkpoints", "epoch{}_checkpoint.pth.tar".format(epoch))
        torch.save(state, f=filename)

        best_filename = os.path.join("checkpoints", "best_checkpoint.pth.tar")
        if self.max_acc < current_acc:
            torch.save(state, f=best_filename)
            print("Saving Epoch: {}, Updating acc {:.6f} to {:.6f}".format(epoch, self.max_acc, current_acc))
            self.max_acc = current_acc


def main(directory):
    trainer = Trainer(directory=directory,
                      epoch=100,
                      batch_size=64)
    trainer.train(valid=True)


if __name__ == '__main__':
    # file directory from argv
    directory = sys.argv[1]
    main(directory)
