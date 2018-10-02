import argparse
import cv2
import numpy as np
import os
import time


def candidate():
    for w_r in range(11):
        for w_g in range(11):
            if w_r + w_g > 10:
                continue
            else:
                yield w_r/10, w_g/10, round(1.0-w_r/10-w_g/10, 1)


def plot(image, file):
    return cv2.imwrite(file, image)


class JointBilateralFilter:
    def __init__(self, image, guide, sigma_s, sigma_r, r_factor=3):
        assert(image.shape[0] == guide.shape[0])
        assert(image.shape[1] == guide.shape[1])

        # image definition
        self.image = image / 255.0
        self.guide = guide / 255.0
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.filtered = None

        # filtering factors
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.radius = r_factor * sigma_s

    def filter(self):
        self.filtered = np.zeros(self.image.shape)
        for x in range(self.width):
            for y in range(self.height):
                # edges
                y_bottom = np.maximum(0, y - self.radius)
                y_top = np.minimum(self.height, y + self.radius + 1)
                x_left = np.maximum(0, x - self.radius)
                x_right = np.minimum(self.width, x + self.radius + 1)

                # h_space: size = (2r + 1) x (2r + 1) (window size)
                h_space = [[i**2 + j**2 for i in range(x_left-x, x_right-x)] for j in range(y_bottom-y, y_top-y)]
                h_space = np.exp(-np.array(h_space) / (2 * self.sigma_s ** 2))

                # for single-channel image
                center_value = self.guide[y][x]
                h_range = np.exp(-(self.guide[y_bottom:y_top, x_left:x_right] - center_value) ** 2 / (2 * (self.sigma_r ** 2)))

                im = self.image[y_bottom:y_top, x_left:x_right]
                multi = np.multiply(h_space, h_range)
                self.filtered[y_bottom:y_top, x_left:x_right, 0] += np.multiply(multi, im[:, :, 0]) / np.sum(multi)
                self.filtered[y_bottom:y_top, x_left:x_right, 1] += np.multiply(multi, im[:, :, 1]) / np.sum(multi)
                self.filtered[y_bottom:y_top, x_left:x_right, 2] += np.multiply(multi, im[:, :, 2]) / np.sum(multi)

        self.filtered = self.filtered * 255



"""
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
fig = plt.figure(6)
ax = plt.axes(projection='3d')
ax.view_init(30, 30)
ax.set_xlim3d(0.0,1.0)
ax.set_ylim3d(0.0,1.0)
ax.set_zlim3d(0.0,1.0)
for x,y,z in candidate():
    ax.scatter(x,y,z, c="y")
plt.show()
"""

def main(args):
    try:
        image = cv2.imread(os.path.join(args.input))  # BGR image

        if not os.path.exists(args.output):
            os.makedirs(args.output)
        # Conventional (BGR->YUV)
        if args.mode == "c":
            y = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
            filename = os.path.splitext(os.path.basename(args.input))[0] + '_y.png'
            plot(np.expand_dims(y, axis=2), os.path.join(args.output, filename))

        # Advanced (BGR->YUV)
        elif args.mode == "a":
            image_candidate = dict()
            start = time.time()
            for w_r, w_g, w_b in candidate():
                y = w_r * image[:, :, 2] + w_g * image[:, :, 1] + w_b * image[:, :, 0]
                bilateral_filter = JointBilateralFilter(image, y, args.sigma_s, args.sigma_r)
                bilateral_filter.filter()
                print("Plotting filtered image: w_r:{} w_g:{} w_b:{}".format(w_r, w_g, w_b))
                image_candidate[(w_r, w_g, w_b)] = bilateral_filter.filtered
                plot(bilateral_filter.filtered,
                     os.path.join(args.output, "w_r_{}_w_g_{}_w_b_{}.png".format(w_r, w_g, w_b)))
            print("Done! Time elapsed:", time.time() - start)

            # calculate cost (local minimum)


        else:
            raise NotImplementedError("Please specify the mode \"a\" for advanced or \"c\" for conventional method.")

    except TypeError:
        print("Please input correct image directory.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="testdata/0c.png",
                        help="rgb input image")
    parser.add_argument("-o", "--output", default="advanced/s1r0.05/0c",
                        help="gray scale output directory")
    parser.add_argument("-s", "--sigma_s", default=1,
                        help="sigma for spatial kernel")
    parser.add_argument("-r", "--sigma_r", default=0.1,
                        help="sigma for range(color) kernel")
    parser.add_argument("--mode", default="a",
                        help="c: conventional; a: advanced")
    main(parser.parse_args())
