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

    def gradient_ascend(self, selected_layer, selected_filter):
        print(selected_layer, selected_filter)
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.process = torch.tensor(np.zeros((1, 28, 28)),
                                    dtype=torch.float,
                                    device=self.device).unsqueeze(0)
        self.process.requires_grad = True

        # model optimizer
        optimizer = Adam(params=[self.process], lr=0.1, weight_decay=1e-6)

        for idx in range(1, 1001):
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

            print("Epoch {} loss: {:.6f}".format(idx, loss.item()))

        self.process = self.process.squeeze().cpu().detach().numpy()
        self.process -= self.process.mean()
        self.process /= self.process.std()
        self.process *= 0.1

        self.process += 0.5
        self.process = np.clip(self.process, 0, 1)
        self.process *= 255

        imsave("layer{}_filter{}.png".format(self.selected_layer, self.selected_filter), self.process)
        print("Done!")
        print()



def main():
    checkpoint = torch.load('best_checkpoint.pth.tar', map_location=lambda storage, loc: storage)
    visual = CNNFilterVisualization(checkpoint['state_dict'])
    for idx in [1, 2, 4, 8, 16, 32]:
        visual.gradient_ascend(0, idx)
        visual.gradient_ascend(3, idx)


if __name__ == '__main__':
    main()
