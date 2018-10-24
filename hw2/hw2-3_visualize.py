import matplotlib.pyplot as plt
import torch


def main():
    checkpoint = torch.load('checkpoints/epoch100_checkpoint.pth.tar')

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


if __name__ == '__main__':
    main()
