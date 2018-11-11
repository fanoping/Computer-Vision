from scipy.misc import imsave
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch


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


class CNNFilterVisualization:
    def __init__(self, state_dict):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = CNN().to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.selected_layer = 0
        self.selected_filter = 0
        self.init_image = np.zeros((1, 28, 28))

    def visualize(self, selected_layer, selected_filter):
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.process = torch.tensor(self.init_image, dtype=torch.float, device=self.device).unsqueeze(0)

        # model optimizer
        optimizer = Adam(params=[self.process], lr=0.001, weight_decay=1e-6)

        for idx in range(1, 101):
            optimizer.zero_grad()
            x = self.process
            # get to the selected layer
            for i, layer in enumerate(self.model.conv):
                x = layer(x)
                if i == self.selected_layer:
                    break

            selected_output = x[0, self.selected_filter]
            loss = -torch.mean(selected_output)
            loss.backward()
            optimizer.step()

            print(self.process)
            input()
            print("Epoch {} loss: {:.6f}".format(idx, loss.item()))
            imsave("ep{}_layer{}_filter{}.png".format(idx, self.selected_layer, self.selected_filter),
                   (self.process.squeeze().cpu().numpy()))


def main():
    checkpoint = torch.load('checkpoints/best_checkpoint.pth.tar')
    visual = CNNFilterVisualization(checkpoint['state_dict'])
    visual.visualize(1, 50)


if __name__ == '__main__':
    main()
