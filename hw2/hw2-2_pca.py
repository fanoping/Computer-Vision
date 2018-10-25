from scipy.misc import imread, imsave
import numpy as np
import sys
import os


def plot(image, filename):
    imsave(filename, image)


def mse(image_a, image_b):
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err = err / (float(len(image_a)))
    return err


def manual_pca(images):
    # mean face
    mu = images.mean(axis=0)

    ma = images - mu
    u, s, v = np.linalg.svd(ma.T, full_matrices=False)

    weights = np.dot(ma, u)
    return u, weights, mu


def reconstruct(img_idx, u, weight, mu, n):
    recon = mu + np.dot(weight[img_idx, :n], u[:, :n].T)
    return recon


def main(im_directory):
    # read image
    train_images = []
    test_images = []

    im_shape = imread(os.path.join(im_directory, '1_1.png')).shape
    print("Image shape:", im_shape)

    for i in range(1, 41):
        for j in range(1, 8):
            image = imread(os.path.join(im_directory, '{}_{}.png'.format(i, j)))
            train_images.append(image.reshape(-1,))
        for k in range(8, 11):
            image = imread(os.path.join(im_directory, '{}_{}.png'.format(i, k)))
            test_images.append(image.reshape(-1,))

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    print(len(train_images), "train images and", len(test_images), "test images loaded!")

    eigen, weights, mu = manual_pca(train_images)
    # plot(mu.reshape(im_shape), "mean.png")

    # eigenfaces
    # for i in range(5):
    #    plot(eigen[:, i].reshape(im_shape), "eigenface_{}.png".format(i + 1))

    # reconstruct person_8_image_6
    for n in [5, 50, 150, len(train_images)]:
        recon = reconstruct(54, eigen, weights, mu, n)
    #    plot(recon.reshape(im_shape), "reconstruct_{}.png".format(n))
        error = mse(recon, train_images[54])
        print("MSE with n eigenface:", error)


if __name__ == '__main__':
    directory = sys.argv[1]
    main(directory)
