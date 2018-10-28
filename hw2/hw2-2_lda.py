from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.misc import imread, imsave
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def plot(image, filename):
    imsave(filename, image)


def pca(images):
    # mean face
    mu = images.mean(axis=0)

    ma = images - mu
    u, s, v = np.linalg.svd(ma.T, full_matrices=False)

    weights = np.dot(ma, u)
    return u, weights, mu


def lda(embedded_images):
    # S_B part
    s_w = np.zeros(shape=(240, 240))
    for i in range(40):
        for j in range(7):
            # mean of every classes C_i; i = 1, 2, 3, ..., 40
            mu_i = embedded_images[7*i:7*(i+1), :].mean(axis=0)
            sw_i = embedded_images[7 * i + j, :].reshape(-1, 1) - mu_i.reshape(-1, 1)
            sw_i = np.multiply(sw_i, sw_i.T)
            s_w += sw_i

    s_b = np.zeros(shape=(240, 240))
    mu = embedded_images.mean(axis=0)
    for i in range(40):
        mu_i = embedded_images[7 * i:7 * (i + 1), :].mean(axis=0)
        sb_i = mu_i.reshape(-1, 1) - mu.reshape(-1, 1)
        sb_i = np.multiply(sb_i, sb_i.T)
        s_b += sb_i

    inv_sw = np.linalg.inv(s_w)
    matrix = np.multiply(inv_sw, s_b)
    print(matrix.shape)

    lda_mu = matrix.mean(axis=0)
    lda_ma = matrix - lda_mu
    u, s, v = np.linalg.svd(lda_ma.T, full_matrices=False)
    weights = np.dot(lda_ma, u)

    return u, weights, lda_mu, matrix


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
    plt.legend(loc="best")
    plt.title('t-SNE for LDA embedding')
    plt.savefig("tsne_lda.png")


def main(im_directory, output_testing_image):
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

    eigen, weights, mu = pca(train_images)

    mean = train_images - mu
    embed = np.dot(mean, eigen[:, :240])   # project onto (N-C) eigenvectors N = 280, C = 40

    eigen_lda, weights_lda, mu_lda, matrix_lda = lda(embed)

    for i in range(5):
        fisher = mu_lda + np.dot(weights_lda[:, i], eigen_lda[:, i].T)
        inv = np.dot(fisher, eigen[:, :240].T) + mu
        plot(inv.reshape(im_shape), "result/fisherface_{}.png".format(i+1))

    # t-SNE visualize
    # mean = train_images - mu
    # embed = np.dot(mean, eigen[:, :240])
    # embed_mean = embed - mu_lda
    # embed_lda = np.dot(embed_mean, eigen_lda[:, :30])
    # tsne_visual(embed_lda, train_labels)

    print("Reconstructed Image File:", output_testing_image)
    recon = mu_lda + np.dot(weights_lda[:, 0], eigen_lda[:, 0].T)
    inverse = np.dot(recon, eigen[:, :240].T) + mu
    plot(inverse.reshape(im_shape), output_testing_image)

    # KNN result
    k_near = [1, 3, 5]
    dim = [3, 10, 39]

    print("Cross Validation Score by KNN")
    for k in k_near:
        for n in dim:
            mean = test_images - mu
            embed = np.dot(mean, eigen[:, :240])
            embed_mean = embed - mu_lda
            embed_lda = np.dot(embed_mean, eigen_lda[:, :n])
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, embed_lda, test_labels, cv=3, scoring="accuracy")
            print("K = {}, N = {}:\tcross validation acc = {:.3f} / {:.3f} / {:.3f}".format(
                k, n, scores[0], scores[1], scores[2])
            )


if __name__ == '__main__':
    directory = sys.argv[1]
    output_image = sys.argv[2]
    main(directory, output_image)
