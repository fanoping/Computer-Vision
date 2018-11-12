from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.misc import imread, imsave
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
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
    recon = mu + np.dot(weight[img_idx, :n], u[:, :n].T)   # (1 * n) * (n * 2576) 2576 pixels
    return recon


def tsne_visual(weight, label):
    tsne = TSNE(n_components=2, random_state=30, verbose=1, n_iter=2500)
    embedded = tsne.fit_transform(weight)

    plt.figure(figsize=(12, 12))
    colors = iter(cm.rainbow(np.linspace(0, 1, 40)))
    for lab in np.unique(label):
        c = next(colors)
        mask = label == lab
        plt.scatter(embedded[mask, 0], embedded[mask, 1], s=30, c=c, label=lab)
    #for x, y, v in zip(embedded[:, 0], embedded[:, 1], label):
    #    plt.text(x, y, v, fontsize=8)

    plt.legend(loc="best")
    plt.title('t-SNE for PCA embedding')
    plt.savefig("tsne_pca.png")


def main(im_directory, testing_image, output_testing_image):
    # read image
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    im_shape = imread(os.path.join(im_directory, '1_1.png')).shape
    print("Image shape:", im_shape)

    for i in range(1, 41):
        for j in range(1, 8):
            image = imread(os.path.join(im_directory, '{}_{}.png'.format(i, j)))
            train_images.append(image.reshape(-1,))
            train_labels.append(i)
        for k in range(8, 11):
            image = imread(os.path.join(im_directory, '{}_{}.png'.format(i, k)))
            test_images.append(image.reshape(-1,))
            test_labels.append(i)

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
        print("MSE with {:3} eigenfaces:\t{}".format(n, error))

    # t-SNE visualize
    # embed = test_images - mu
    # embed = np.dot(embed, eigen[:, :100])
    # tsne_visual(embed, test_labels)

    # reconstruct testing images
    print("Testing Image:", testing_image)
    print("Reconstructed Image File:", output_testing_image)

    testing_image = imread(testing_image).reshape(-1,)
    mean = testing_image - mu
    embed = np.dot(mean, eigen[:, :])
    inverse = np.dot(embed, eigen[:, :].T) + mu
    plot(inverse.reshape(im_shape), output_testing_image)

    # KNN result
    k_near = [1, 3, 5]
    dim = [3, 10, 39]

    print("Cross Validation Score by KNN")
    for k in k_near:
        for n in dim:
            mean = train_images - mu
            embed = np.dot(mean, eigen[:, :n])
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(embed, train_labels)
            scores = cross_val_score(knn, embed, train_labels, cv=3, scoring="accuracy")
            print("K = {}, N = {}:\tcross validation acc = {:.4f} / {:.4f} / {:.4f}".format(
                k, n, scores[0], scores[1], scores[2]), end=" "
            )

            test_m = test_images - mu
            test_em = np.dot(test_m, eigen[:, :n])
            print("Test score: {:.5f}".format(knn.score(test_em, test_labels)))

    print("Done!")


if __name__ == '__main__':
    directory = sys.argv[1]
    test_image = sys.argv[2]
    output_image = sys.argv[3]
    main(directory, test_image, output_image)
