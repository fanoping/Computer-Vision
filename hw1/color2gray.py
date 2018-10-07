from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import argparse
import time
import cv2
import sys
import os


def candidate():
    for w_r in range(11):
        for w_g in range(11):
            if w_r + w_g > 10:
                break
            else:
                yield w_r/10, w_g/10, abs(round(1.0-w_r/10-w_g/10, 1))


def plot(image, file):
    return cv2.imwrite(file, image)


class Position:
    def __init__(self, w_r, w_g, w_b):
        self.w_r = w_r
        self.w_g = w_g
        self.w_b = w_b
        self.votes = 0

    def __eq__(self, other):
        if self.w_r == other.w_r and self.w_g == other.w_g and self.w_b == other.w_b:
            return True
        else:
            return False

    def __str__(self):
        return "({}, {}, {})".format(self.w_r, self.w_g, self.w_b)

    def __hash__(self):
        return int(10 * self.w_r + 1000 * self.w_g + 10000 * self.w_b)

    def __isneighbor__(self, other):
        if round(abs(self.w_r - other.w_r) + abs(self.w_g - other.w_g) + abs(self.w_b - other.w_b), 1) == 0.2:
            return True
        else:
            return False

    def vote(self):
        self.votes += 1


class JointBilateralFilter:
    def __init__(self, args, image):
        self.args = args

        # image definition
        self.image = image / 255.0
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        # filtering factors
        self.sigma_s = None
        self.sigma_r = None

        # initialize
        self.bilateral_image = None
        self.joint_bilateral_image = dict()

    def filtered_image_gen(self):
        # saved path
        filename = os.path.join(self.args.output,
                                "s{}r{}".format(self.sigma_s, self.sigma_r),
                                os.path.splitext(os.path.basename(self.args.input))[0])
        if not os.path.exists(filename):
            os.makedirs(filename)

        # for verifying
        # test = cv2.bilateralFilter(self.image.astype(dtype=np.float32), d=7, sigmaSpace=1, sigmaColor=100)
        # plot(test*255.0, os.path.join(filename, "bilateral_bycv2.png"))

        # bilateral image
        self.bilateral_image = self.__filter(guide=None)

        if self.args.plot:
            print("Plotting bilateral image.")
            self.bilateral_image *= 255.0
            plot(self.bilateral_image, os.path.join(filename, "origin_bilateral.png"))

        # candidates (joint bilateral image)
        for w_r, w_g, w_b in candidate():
            print("Joint Bilateral Filtered Image: w_r:{} w_g:{} w_b:{}".format(w_r, w_g, w_b), end='\r')
            sys.stdout.write('\033[K')

            y = w_r * self.image[:, :, 2] + w_g * self.image[:, :, 1] + w_b * self.image[:, :, 0]
            filtered = self.__filter(guide=y)
            if self.args.plot:
                filtered *= 255.0
                plot(filtered, os.path.join(filename, "w_r_{}_w_g_{}_w_b_{}.png".format(w_r, w_g, w_b)))

            pos = Position(w_r, w_g, w_b)
            self.joint_bilateral_image[pos] = filtered

        print("Finish filtering {} images".format(len(self.joint_bilateral_image)))

    def __filter(self, guide=None):
        filtered = np.zeros(self.image.shape)
        radius = 3 * self.sigma_s

        for x in range(self.width):
            for y in range(self.height):
                # edges
                y_bottom = np.maximum(0, y - radius)
                y_top = np.minimum(self.height, y + radius + 1)
                x_left = np.maximum(0, x - radius)
                x_right = np.minimum(self.width, x + radius + 1)

                # h_space: size = (2r + 1) x (2r + 1) (window size)
                h_space = [[i**2 + j**2 for i in range(x_left-x, x_right-x)] for j in range(y_bottom-y, y_top-y)]
                h_space = np.exp(-np.array(h_space) / (2 * self.sigma_s ** 2))

                # for single-channel image
                if guide is not None:
                    center_value = guide[y][x]
                    power = (guide[y_bottom:y_top, x_left:x_right] - center_value) ** 2
                    h_range = np.exp(-power / (2 * (self.sigma_r ** 2)))

                    # add together
                    im = self.image[y_bottom:y_top, x_left:x_right]
                    multi = np.multiply(h_space, h_range)
                    filtered[y_bottom:y_top, x_left:x_right, 0] += np.multiply(multi, im[:, :, 0]) / np.sum(multi)
                    filtered[y_bottom:y_top, x_left:x_right, 1] += np.multiply(multi, im[:, :, 1]) / np.sum(multi)
                    filtered[y_bottom:y_top, x_left:x_right, 2] += np.multiply(multi, im[:, :, 2]) / np.sum(multi)

                else:
                    # for self bilateral (rgb image)
                    center_value = self.image[y][x]

                    power_b = (self.image[y_bottom:y_top, x_left:x_right, 0] - center_value[0]) ** 2
                    power_g = (self.image[y_bottom:y_top, x_left:x_right, 1] - center_value[1]) ** 2
                    power_r = (self.image[y_bottom:y_top, x_left:x_right, 2] - center_value[2]) ** 2
                    h_range_b = np.exp(-power_b / (2 * (self.sigma_r ** 2)))
                    h_range_g = np.exp(-power_g / (2 * (self.sigma_r ** 2)))
                    h_range_r = np.exp(-power_r / (2 * (self.sigma_r ** 2)))

                    # add together
                    im = self.image[y_bottom:y_top, x_left:x_right]
                    multi_b = np.multiply(h_space, h_range_b)
                    multi_g = np.multiply(h_space, h_range_g)
                    multi_r = np.multiply(h_space, h_range_r)

                    filtered[y_bottom:y_top, x_left:x_right, 0] += np.multiply(multi_b, im[:, :, 0]) / np.sum(multi_b)
                    filtered[y_bottom:y_top, x_left:x_right, 1] += np.multiply(multi_g, im[:, :, 1]) / np.sum(multi_g)
                    filtered[y_bottom:y_top, x_left:x_right, 2] += np.multiply(multi_r, im[:, :, 2]) / np.sum(multi_r)

        return filtered

    def __cost(self, image):
        return np.sum(abs(image - self.bilateral_image))

    def __is_local_min(self, pos):
        for neigh in self.joint_bilateral_image:
            if pos == neigh:
                continue
            elif pos.__isneighbor__(neigh):
                if self.__cost(self.joint_bilateral_image[neigh]) < self.__cost(self.joint_bilateral_image[pos]):
                    return False
        return True

    def vote(self):
        for pos in self.joint_bilateral_image:
            if self.__is_local_min(pos):
                pos.vote()

    def print_result(self):
        print("===========Result=============")
        count = 0
        for pos in self.joint_bilateral_image:
            print(pos, pos.votes, end='\t')
            count += 1
            if count % 3 == 0:
                print()

    def plot_result(self, filename):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.set_xlabel("w_r")
        ax.set_ylabel("w_g")
        ax.set_zlabel("w_b")

        x, y, z, v = [], [], [], []
        for pos, _ in self.joint_bilateral_image.items():
            x.append(pos.w_r)
            y.append(pos.w_g)
            z.append(pos.w_b)
            v.append(pos.votes)

        p = ax.scatter(x, y, z, c=v, cmap=cm.YlOrRd, s=30, vmin=0, vmax=9)
        for xs, ys, zs, vs in zip(x, y, z, v):
            ax.text(xs, ys, zs, vs, fontsize=8)
        cb = plt.colorbar(p)
        cb.set_label("votes")
        ax.view_init(30, 60)
        plt.savefig(fname=filename)

    def topn_result(self, n):
        top_n_pos = [k for k, _ in self.joint_bilateral_image.items()]
        top_n_pos = sorted(top_n_pos, key=lambda k: k.votes, reverse=True)
        top_n_pos = top_n_pos[:n]

        print("Top {} positions: (RGB order)")
        for i, pos in enumerate(top_n_pos):
            print("[{}]:\t({}, {}, {})\tvotes: {}".format(i, pos.w_r, pos.w_g, pos.w_b, k.votes))
            if self.args.plot:
                filename = os.path.splitext(os.path.basename(self.args.input))[0] + '_y' + str(i) + '.png'
                if not os.path.exists(os.path.join(self.args.output, "results")):
                    os.mkdir(os.path.join(self.args.output, "results"))

                output = pos.w_r * self.image[:, :, 2] + pos.w_g * self.image[:, :, 1] + pos.w_b * self.image[:, :, 0]
                output = output * 255.0
                plot(np.expand_dims(output, axis=2),
                     os.path.join(self.args.output, "results", filename))


def main(args):
    image = cv2.imread(os.path.join(args.input))  # BGR image

    # Conventional (BGR->YUV)
    if args.mode == "c":
        y = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
        filename = os.path.splitext(os.path.basename(args.input))[0] + '_y.png'
        plot(np.expand_dims(y, axis=2), os.path.join(args.output, filename))

    # Advanced (BGR->YUV)
    elif args.mode == "a":
        bilateral_filter = JointBilateralFilter(args, image)
        for sigma_s in [1, 2, 3]:
            for sigma_r in [0.05, 0.1, 0.2]:
                print("Progress: sigma_s={}, sigma_r={}".format(sigma_s, sigma_r))
                start = time.time()
                bilateral_filter.sigma_s = sigma_s
                bilateral_filter.sigma_r = sigma_r
                bilateral_filter.filtered_image_gen()
                bilateral_filter.vote()
                bilateral_filter.print_result()
                print("Time Elapsed:", time.time() - start)
                print("===========================================================")

        bilateral_filter.plot_result(os.path.join(args.output,
                                                  os.path.splitext(os.path.basename(args.input))[0] + '_vote.png'))
        bilateral_filter.topn_result(n=3)
        print("Done!")

    else:
        raise NotImplementedError("Please specify the mode \"a\" for advanced or \"c\" for conventional method.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="testdata/0a.png",
                        help="rgb input image")
    parser.add_argument("-o", "--output", default="advanced",
                        help="output directory")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="plot for every candidate while initializing")
    parser.add_argument("--mode", default="a",
                        help="c: conventional; a: advanced")
    main(parser.parse_args())
