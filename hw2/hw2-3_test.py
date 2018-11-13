from scipy.misc import imread
import numpy as np
import torch.nn as nn
import torch
import csv
import sys
import os


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


def main(directory, output_csv):
    checkpoint = torch.load('best_checkpoint.pth.tar', map_location=lambda storage, loc: storage)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CNN().to(device)
    model.load_state_dict(checkpoint['state_dict'])

    print("Selected checkpoint training acc: {:.6f} validation acc: {:.6f}".format(
        checkpoint['accuracy'][checkpoint['epoch'] - 1],
        checkpoint['val_accuracy'][checkpoint['epoch'] - 1])
    )

    with torch.no_grad():
        model.eval()
        with open(output_csv, "w") as f:
            s = csv.writer(f, delimiter=',', lineterminator='\n')
            s.writerow(["id", "label"])
            for imagefile in sorted(os.listdir(directory)):
                image = imread(os.path.join(directory, imagefile))
                image = np.expand_dims(image, axis=0)
                image = torch.tensor(image, dtype=torch.float, device=device).unsqueeze(0)

                output = model(image)
                result = torch.max(output, dim=1)[1]

                idx = os.path.splitext(os.path.basename(imagefile))[0]
                s.writerow([idx, result.item()])


if __name__ == '__main__':
    directory = sys.argv[1]
    output_csv = sys.argv[2]
    main(directory, output_csv)
