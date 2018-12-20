import numpy as np
import cv2
import time
import math
import os
from scipy.ndimage.filters import median_filter, sobel
DEBUG = 1
SAVE_DIR = "./result_test_bilateral_0.02_0.4_verifying"


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape

    if DEBUG:
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)

    print('* Bilateral Filtering')
    def bilateral_filter(image, texture):
        r = 9
        sigma_s, sigma_r = 0.02, 0.4*255
        image_pad = np.pad(image, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float64)
        texture_pad = np.pad(texture, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.int32)

        output = np.zeros_like(image)
        scale_factor_s = 1 / (2 * sigma_s * sigma_s)
        scale_factor_r = 1 / (2 * sigma_r * sigma_r)
        # range kernel
        table = np.exp(-np.arange(256) * np.arange(256) * scale_factor_r)

        x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
        kernel_s = np.exp(-(x * x + y * y) * scale_factor_s)

        for y in range(r, r+h):
            for x in range(r, r+w):
                weight = table[abs(texture_pad[y - r: y + r + 1, x - r: x + r + 1, 0] - texture_pad[y, x, 0])] * \
                         table[abs(texture_pad[y - r: y + r + 1, x - r: x + r + 1, 1] - texture_pad[y, x, 1])] * \
                         table[abs(texture_pad[y - r: y + r + 1, x - r: x + r + 1, 2] - texture_pad[y, x, 2])] * \
                         kernel_s
                cum_weight = np.sum(weight)
                output[y - r, x - r, 0] = np.sum(weight * image_pad[y - r: y + r + 1, x - r:x + r + 1, 0]) / cum_weight
                output[y - r, x - r, 1] = np.sum(weight * image_pad[y - r: y + r + 1, x - r:x + r + 1, 1]) / cum_weight
                output[y - r, x - r, 2] = np.sum(weight * image_pad[y - r: y + r + 1, x - r:x + r + 1, 2]) / cum_weight
        return output

    Il = bilateral_filter(Il, Il)
    Ir = bilateral_filter(Ir, Ir)

    if DEBUG:
        cv2.imwrite(os.path.join(SAVE_DIR, '{}_bilateral_filter_left.png'.format(name)), Il)
        cv2.imwrite(os.path.join(SAVE_DIR, '{}_bilateral_filter_right.png'.format(name)), Ir)

    Il = Il.astype(np.float64)
    Ir = Ir.astype(np.float64)

    tao_1, tao_2 = 20, 6
    l1, l2 = 34, 17
    def arm_check(p_y, p_x, q_y, q_x, edge_term, img):
        # basic check, window size > pixel size
        if not (0 <= q_y < h and 0 <= q_x < w): return False
        if abs(q_y - p_y) == 1 or abs(q_x - p_x) == 1: return True

        # Rule 1: Extend the arm such that the difference of the color intensity is smaller than the threshold tao_1
        if abs(img[p_y, p_x, 0] - img[q_y, q_x, 0]) >= tao_1: return False
        if abs(img[p_y, p_x, 1] - img[q_y, q_x, 1]) >= tao_1: return False
        if abs(img[p_y, p_x, 2] - img[q_y, q_x, 2]) >= tao_1: return False

        if abs(img[q_y, q_x, 0] - img[q_y + edge_term[0], q_x + edge_term[1], 0]) >= tao_1: return False
        if abs(img[q_y, q_x, 1] - img[q_y + edge_term[0], q_x + edge_term[1], 1]) >= tao_1: return False
        if abs(img[q_y, q_x, 2] - img[q_y + edge_term[0], q_x + edge_term[1], 2]) >= tao_1: return False

        # Rule 2: Aggregation window < L1
        if abs(q_y - p_y) >= l1 or abs(q_x - p_x) >= l1: return False

        # Rule 3: Local information constraint
        if abs(q_y - p_y) >= l2 or abs(q_x - p_x) >= l2:
            if abs(img[p_y, p_x, 0] - img[q_y, q_x, 0]) >= tao_2: return False
            if abs(img[p_y, p_x, 1] - img[q_y, q_x, 1]) >= tao_2: return False
            if abs(img[p_y, p_x, 2] - img[q_y, q_x, 2]) >= tao_2: return False

        return True

    # check aggregated window
    def window_def(img):
        window = np.empty((h, w, 4), dtype=np.int)
        for y in range(h):
            for x in range(w):
                window[y, x, 0] = y - 1
                window[y, x, 1] = y + 1
                window[y, x, 2] = x - 1
                window[y, x, 3] = x + 1
                while arm_check(y, x, window[y, x, 0], x, [1, 0], img):
                    window[y, x, 0] -= 1
                while arm_check(y, x, window[y, x, 1], x, [-1, 0], img):
                    window[y, x, 1] += 1
                while arm_check(y, x, y, window[y, x, 2], [0, 1], img):
                    window[y, x, 2] -= 1
                while arm_check(y, x, y, window[y, x, 3], [0, -1], img):
                    window[y, x, 3] += 1
        return window

    def cross_based_cost_aggregation(wind_l, wind_r, prev_cost, vertical=True):
        after_cost = np.empty_like(prev_cost)
        for disp in range(max_disp):
            for y in range(h):
                for x in range(w):
                    if x - disp < 0:
                        after_cost[disp, y, x] = prev_cost[disp, y, x]
                        continue
                    p_cost, pixel_cnt = 0, 0
                    if vertical:
                        left_edge = max(wind_l[y, x, 2], wind_r[y, x - disp, 2] + disp) + 1
                        right_edge = min(wind_l[y, x, 3], wind_r[y, x - disp, 3] + disp)
                        for hori in range(left_edge, right_edge):
                            up_edge = max(wind_l[y, hori, 0], wind_r[y, hori - disp, 0]) + 1
                            down_edge = min(wind_l[y, hori, 1], wind_r[y, hori - disp, 1])
                            for verti in range(up_edge, down_edge):
                                p_cost += prev_cost[disp, verti, hori]
                                pixel_cnt += 1
                    else:
                        up_edge = max(wind_l[y, x, 0], wind_r[y, x - disp, 0]) + 1
                        down_edge = min(wind_l[y, x, 1], wind_r[y, x - disp, 1])
                        for verti in range(up_edge, down_edge):
                            left_edge = max(wind_l[verti, x, 2], wind_r[verti, x - disp, 2] + disp) + 1
                            right_edge = min(wind_l[verti, x, 3], wind_r[verti, x - disp, 3] + disp)
                            for hori in range(left_edge, right_edge):
                                p_cost += prev_cost[disp, verti, hori]
                                pixel_cnt += 1
                    assert (pixel_cnt > 0)
                    after_cost[disp, y, x] = p_cost / pixel_cnt
        return after_cost

    def cost_matching(img_l, img_r, reverse=False):
        # >>> Cost computation
        print("* Cost computation (initialization)")
        tic = time.time()

        # initial cost computation
        left_census = np.zeros((h, w, 63 * 3), dtype=np.int)
        right_census = np.zeros((h, w, 63 * 3), dtype=np.int)
        np.random.seed(1)
        random_int = np.random.randint(2**31)

        cost_ad = np.zeros((max_disp, h, w), dtype=np.float64)
        cost_census = np.ones((max_disp, h, w), dtype=np.float64) * np.inf

        for y in range(h):
            for x in range(w):
                # initial cost computation C_AD
                for disp in range(max_disp):
                    if x - disp < 0:
                        cost_ad[disp, y, x] = math.inf
                    else:
                        cost_ad[disp, y, x] += abs(img_l[y, x, 0] - img_r[y, x - disp, 0])
                        cost_ad[disp, y, x] += abs(img_l[y, x, 1] - img_r[y, x - disp, 1])
                        cost_ad[disp, y, x] += abs(img_l[y, x, 2] - img_r[y, x - disp, 2])
                        cost_ad[disp, y, x] /= 3

                # initial census cost
                index = 0
                for i in range(y - 3, y + 4):
                    for j in range(x - 4, x + 5):
                        for k in range(3):
                            if 0 <= i < h and 0 <= j < w:
                                left_census[y, x, index] = img_l[i, j, k] < img_l[y, x, k]
                                right_census[y, x, index] = img_r[i, j, k] < img_r[y, x, k]
                            else:
                                left_census[y, x, index] = random_int
                                right_census[y, x, index] = random_int
                            index += 1

        for disp in range(max_disp):
            cost_census[disp, :, disp:] = np.sum(left_census[:, disp:] != right_census[:, :w - disp], 2)

        cost_census /= 3

        lambda_census = 30
        lambda_ad = 10

        norm_ad = 1 - np.exp(-cost_ad / lambda_ad)
        norm_census = 1 - np.exp(-cost_census / lambda_census)
        cost_total = norm_census + norm_ad

        if DEBUG:
            label = cost_total.argmin(0).astype(np.float64)
            filename = os.path.join(SAVE_DIR, '{}_adcost.png'.format(name)) if not reverse \
                       else os.path.join(SAVE_DIR, '{}_adcost_r.png'.format(name))
            cv2.imwrite(filename, np.uint8(label * scale_factor))

        toc = time.time()
        print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

        # >>> Cost aggregation
        tic = time.time()
        # TODO: Refine cost by aggregate nearby costs
        # cross-based aggregation method

        window_l = window_def(img_l)
        window_r = window_def(img_r)

        for idx in range(1, 3):
            # do horizontal and vertical aggregation alternatively
            print('* Cost aggregation (horizontal {})'.format(idx))
            cost_total = cross_based_cost_aggregation(window_l, window_r, cost_total, False)
            if DEBUG:
                label = cost_total.argmin(0).astype(np.float64)
                filename = os.path.join(SAVE_DIR, '{}_adcost_hori_{}.png'.format(name, idx)) if not reverse \
                    else os.path.join(SAVE_DIR, '{}_adcost_hori_{}_r.png'.format(name, idx))
                cv2.imwrite(filename, np.uint8(label * scale_factor))

            print('* Cost aggregation (vertical {})'.format(idx))
            cost_total = cross_based_cost_aggregation(window_l, window_r, cost_total, True)
            if DEBUG:
                label = cost_total.argmin(0).astype(np.float64)
                filename = os.path.join(SAVE_DIR, '{}_adcost_verti_{}.png'.format(name, idx)) if not reverse \
                    else os.path.join(SAVE_DIR, '{}_adcost_verti_{}_r.png'.format(name, idx))
                cv2.imwrite(filename, np.uint8(label * scale_factor))

        toc = time.time()
        print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

        if DEBUG:
            label = cost_total.argmin(0).astype(np.float64)
            filename = os.path.join(SAVE_DIR, '{}_adcost_agg.png'.format(name)) if not reverse \
                else os.path.join(SAVE_DIR, '{}_adcost_agg_r.png'.format(name))
            cv2.imwrite(filename, np.uint8(label * scale_factor))

        # >>> Disparity optimization
        tic = time.time()
        # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
        # Scanline optimization: semi-global matching
        print('* Scanline optimization: (semi-global matching)')
        tao_so = 15
        pi_1, pi_2 = 1., 3.

        # left-right scanline optimization
        result = np.empty_like(cost_total)
        prev_min = 0
        for y in range(h):
            for x in range(w):
                curr_min = np.inf
                for disp in range(max_disp):
                    if x - disp - 1 < 0:
                        result[disp, y, x] = cost_total[disp, y, x]
                    else:
                        # difference between neighboring
                        d_1 = max(abs(img_l[y, x, 0] - img_l[y, x - 1, 0]),
                                  abs(img_l[y, x, 1] - img_l[y, x - 1, 1]),
                                  abs(img_l[y, x, 2] - img_l[y, x - 1, 2]))
                        d_2 = max(abs(img_r[y, x - disp, 0] - img_r[y, x - disp - 1, 0]),
                                  abs(img_r[y, x - disp, 1] - img_r[y, x - disp - 1, 1]),
                                  abs(img_r[y, x - disp, 2] - img_r[y, x - disp - 1, 2]))

                        # constraint
                        if d_1 < tao_so and d_2 < tao_so:
                            p_1, p_2 = pi_1, pi_2
                        elif d_1 < tao_so and d_2 >= tao_so:
                            p_1, p_2 = pi_1/4., pi_2/4.
                        elif d_1 >= tao_so and d_2 < tao_so:
                            p_1, p_2 = pi_1/4., pi_2/4.
                        else:
                            p_1, p_2 = pi_1/10., pi_2/10.

                        result[disp, y, x] = cost_total[disp, y, x] - prev_min + min(
                            result[disp, y, x - 1],
                            result[disp - 1, y, x - 1] + p_1 if disp - 1 >= 0 else np.inf,
                            result[disp + 1, y, x - 1] + p_1 if disp + 1 < max_disp else np.inf,
                            prev_min + p_2
                        )

                    if result[disp, y, x] < curr_min:
                        curr_min = result[disp, y, x]
                prev_min = curr_min
        left_right = result

        # right-left scanline optimization
        result = np.empty_like(cost_total)
        for y in range(h):
            for x in range(w-1, -1, -1):
                curr_min = np.inf
                for disp in range(max_disp):
                    if x + 1 >= w or x - disp < 0:
                        result[disp, y, x] = cost_total[disp, y, x]
                    else:
                        # difference between neighboring
                        d_1 = max(abs(img_l[y, x, 0] - img_l[y, x + 1, 0]),
                                  abs(img_l[y, x, 1] - img_l[y, x + 1, 1]),
                                  abs(img_l[y, x, 2] - img_l[y, x + 1, 2]))
                        d_2 = max(abs(img_r[y, x - disp, 0] - img_r[y, x - disp + 1, 0]),
                                  abs(img_r[y, x - disp, 1] - img_r[y, x - disp + 1, 1]),
                                  abs(img_r[y, x - disp, 2] - img_r[y, x - disp + 1, 2]))

                        # constraint
                        if d_1 < tao_so and d_2 < tao_so:
                            p_1, p_2 = pi_1, pi_2
                        elif d_1 < tao_so and d_2 >= tao_so:
                            p_1, p_2 = pi_1/4., pi_2/4.
                        elif d_1 >= tao_so and d_2 < tao_so:
                            p_1, p_2 = pi_1/4., pi_2/4.
                        else:
                            p_1, p_2 = pi_1/10., pi_2/10.

                        result[disp, y, x] = cost_total[disp, y, x] - prev_min + min(
                            result[disp, y, x + 1],
                            result[disp - 1, y, x + 1] + p_1 if disp - 1 >= 0 else np.inf,
                            result[disp + 1, y, x + 1] + p_1 if disp + 1 < max_disp else np.inf,
                            prev_min + p_2
                        )

                    if result[disp, y, x] < curr_min:
                        curr_min = result[disp, y, x]
                prev_min = curr_min
        right_left = result

        # up-down scanline optimization
        result = np.empty_like(cost_total)
        for x in range(w):
            for y in range(h):
                curr_min = np.inf
                for disp in range(max_disp):
                    if y - 1 < 0 or x - disp < 0:
                        result[disp, y, x] = cost_total[disp, y, x]
                    else:
                        # difference between neighboring
                        d_1 = max(abs(img_l[y, x, 0] - img_l[y - 1, x, 0]),
                                  abs(img_l[y, x, 1] - img_l[y - 1, x, 1]),
                                  abs(img_l[y, x, 2] - img_l[y - 1, x, 2]))
                        d_2 = max(abs(img_r[y, x - disp, 0] - img_r[y - 1, x - disp, 0]),
                                  abs(img_r[y, x - disp, 1] - img_r[y - 1, x - disp, 1]),
                                  abs(img_r[y, x - disp, 2] - img_r[y - 1, x - disp, 2]))

                        # constraint
                        if d_1 < tao_so and d_2 < tao_so:
                            p_1, p_2 = pi_1, pi_2
                        elif d_1 < tao_so and d_2 >= tao_so:
                            p_1, p_2 = pi_1 / 4., pi_2 / 4.
                        elif d_1 >= tao_so and d_2 < tao_so:
                            p_1, p_2 = pi_1 / 4., pi_2 / 4.
                        else:
                            p_1, p_2 = pi_1 / 10., pi_2 / 10.

                        result[disp, y, x] = cost_total[disp, y, x] - prev_min + min(
                            result[disp, y - 1, x],
                            result[disp - 1, y - 1, x] + p_1 if disp - 1 >= 0 else np.inf,
                            result[disp + 1, y - 1, x] + p_1 if disp + 1 < max_disp else np.inf,
                            prev_min + p_2
                        )

                    if result[disp, y, x] < curr_min:
                        curr_min = result[disp, y, x]
                prev_min = curr_min
        up_down = result

        # down-up scanline optimization
        result = np.empty_like(cost_total)
        for x in range(w):
            for y in range(h - 1, -1, -1):
                curr_min = np.inf
                for disp in range(max_disp):
                    if y + 1 >= h or x - disp < 0:
                        result[disp, y, x] = cost_total[disp, y, x]
                    else:
                        # difference between neighboring
                        d_1 = max(abs(img_l[y, x, 0] - img_l[y + 1, x, 0]),
                                  abs(img_l[y, x, 1] - img_l[y + 1, x, 1]),
                                  abs(img_l[y, x, 2] - img_l[y + 1, x, 2]))
                        d_2 = max(abs(img_r[y, x - disp, 0] - img_r[y + 1, x - disp, 0]),
                                  abs(img_r[y, x - disp, 1] - img_r[y + 1, x - disp, 1]),
                                  abs(img_r[y, x - disp, 2] - img_r[y + 1, x - disp, 2]))

                        # constraint
                        if d_1 < tao_so and d_2 < tao_so:
                            p_1, p_2 = pi_1, pi_2
                        elif d_1 < tao_so and d_2 >= tao_so:
                            p_1, p_2 = pi_1 / 4., pi_2 / 4.
                        elif d_1 >= tao_so and d_2 < tao_so:
                            p_1, p_2 = pi_1 / 4., pi_2 / 4.
                        else:
                            p_1, p_2 = pi_1 / 10., pi_2 / 10.

                        result[disp, y, x] = cost_total[disp, y, x] - prev_min + min(
                            result[disp, y + 1, x],
                            result[disp - 1, y + 1, x] + p_1 if disp - 1 >= 0 else np.inf,
                            result[disp + 1, y + 1, x] + p_1 if disp + 1 < max_disp else np.inf,
                            prev_min + p_2
                        )

                    if result[disp, y, x] < curr_min:
                        curr_min = result[disp, y, x]
                prev_min = curr_min
        down_up = result

        cost_total = (left_right + right_left + up_down + down_up) / 4.

        if DEBUG:
            label = cost_total.argmin(0).astype(np.float64)
            filename = os.path.join(SAVE_DIR, '{}_adcost_optim.png'.format(name)) if not reverse \
                else os.path.join(SAVE_DIR, '{}_adcost_optim_r.png'.format(name))
            cv2.imwrite(filename, np.uint8(label * scale_factor))

        toc = time.time()
        print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

        return cost_total, window_l

    cost_total, window_l = cost_matching(Il, Ir)
    cost_total_r, _ = cost_matching(Ir[:, ::-1], Il[:, ::-1], True)
    cost_total_r = cost_total_r[:, :, ::-1]

    def outlier_detection(label, label_r):
        outlier = np.empty_like(label)
        # 0: not an outlier, 1: mismatch point, 2: occlusion point
        for y in range(h):
            for x in range(w):
                if x - label[y, x] < 0:
                    outlier[y, x] = 2
                elif abs(label[y, x] - label_r[y, x - label[y, x]]) < 1.1:
                    outlier[y, x] = 0
                else:
                    for disp in range(max_disp):
                        if x - disp > 0 and abs(disp - label_r[y, x - disp]) < 1.1:
                            outlier[y, x] = 1
                            break
                        else:
                            outlier[y, x] = 2
        return outlier

    tao_s, tao_h = 20, 0.4

    def iterative_region_voting(window, label, outlier):
        histogram = np.empty(max_disp, dtype=int)
        result = np.empty_like(label)
        outlier_result = np.empty_like(outlier)
        for y in range(h):
            for x in range(w):
                disp = label[y, x]
                result[y, x] = disp
                outlier_result[y, x] = outlier[y, x]
                if outlier[y, x] == 0:
                    continue
                for k in range(max_disp):
                    histogram[k] = 0

                not_outlier_cnt = 0
                for verti in range(window[y, x, 0] + 1, window[y, x, 1]):
                    for hori in range(window[verti, x, 2] + 1, window[verti, x, 3]):
                        if outlier[verti, hori] == 0:
                            histogram[label[verti, hori]] += 1
                            not_outlier_cnt += 1
                disp = histogram.argmax()

                if not_outlier_cnt > tao_s and float(histogram[disp]) / not_outlier_cnt > tao_h:
                    outlier_result[y, x] = 0
                    result[y, x] = disp
        return result, outlier_result

    def proper_interpolation(img, label, outlier):
        # outliers are filled with interpolation
        # find 16 different directions
        # if p is an occlusion point -> the pixel with lowest disparity is selected (p comes from background)
        # if p is mismatch point -> the pixel with similar color is selected

        all_dir = np.array([
            [0, 1],
            [-0.5, 1],
            [-1, 1],
            [-1, 0.5],
            [-1, 0],
            [-1, -0.5],
            [-1, -1],
            [-0.5, -1],
            [0, -1],
            [0.5, -1],
            [1, -1],
            [1, -0.5],
            [1, 0],
            [1, 0.5],
            [1, 1],
            [0.5, 1]
        ])
        result = np.empty_like(label)
        for y in range(h):
            for x in range(w):
                result[y, x] = label[y, x]
                if outlier[y, x] != 0:
                    min_distance = np.inf
                    min_disp = -1
                    for direction in range(16):
                        dir_y, dir_x = all_dir[direction, 0], all_dir[direction, 1]
                        verti, hori = y, x
                        verti_iter, hori_iter = int(round(verti)), int(round(hori))
                        while 0 <= verti_iter < h and 0 <= hori_iter < w and outlier[verti_iter, hori_iter] != 0:
                            verti += dir_y
                            hori += dir_x
                            verti_iter, hori_iter = int(round(verti)), int(round(hori))
                        if 0 <= verti_iter < h and 0 <= hori_iter < w:
                            assert(outlier[verti_iter, hori_iter] == 0)
                            if outlier[y, x] == 1:
                                curr_dist = max(abs(img[y, x] - img[verti_iter, hori_iter]))
                            else:
                                curr_dist = label[verti_iter, hori_iter]

                            if curr_dist < min_distance:
                                min_distance = curr_dist
                                min_disp = label[verti_iter, hori_iter]
                    assert(min_disp != -1)
                    result[y, x] = min_disp

        return result

    def depth_discontinuity_adjustment(label, cost):
        # Detect edges and replace the label with lower cost
        # horizontal
        result = np.empty_like(label)
        sobel_filter = sobel(label, axis=0)
        for y in range(h):
            for x in range(w):
                result[y, x] = label[y, x]
                if sobel_filter[y, x] > 10 and 1 <= x < w - 1:
                    disp = label[y, x]
                    if cost[label[y, x - 1], y, x] < cost[disp, y, x]:
                        disp = label[y, x - 1]
                    if cost[label[y, x + 1], y, x] < cost[disp, y, x]:
                        disp = label[y, x + 1]
                    result[y, x] = disp
        # vertical
        label = result
        result = np.empty_like(label)
        sobel_filter = sobel(label, axis=1)
        for y in range(h):
            for x in range(w):
                result[y, x] = label[y, x]
                if sobel_filter[y, x] > 10 and 1 <= y < h - 1:
                    disp = label[y, x]
                    if cost[label[y - 1, x], y, x] < cost[disp, y, x]:
                        disp = label[y - 1, x]
                    if cost[label[y + 1, x], y, x] < cost[disp, y, x]:
                        disp = label[y + 1, x]
                    result[y, x] = disp
        return result

    def subpixel_enhancement(label, cost):
        result = np.empty((h, w))
        for y in range(h):
            for x in range(w):
                disp = label[y, x]
                result[y, x] = disp
                if 1 <= disp < max_disp - 1:
                    cn = cost[disp - 1, y, x]
                    cz = cost[disp, y, x]
                    cp = cost[disp + 1, y, x]
                    denominator = 2 * (cp + cn - 2 * cz)
                    if denominator > 1e-5:
                        result[y, x] = disp - min(1, max(-1, (cp - cn) / denominator))
        return result

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    labels = cost_total.argmin(0)
    labels_r = cost_total_r.argmin(0)

    print('* Outlier Detection')
    outlier = outlier_detection(labels, labels_r)
    if DEBUG:
        output = np.copy(Il)
        output[outlier != 0] = 0
        output[outlier == 1, 0] = 255
        output[outlier == 2, 1] = 255
        cv2.imwrite(os.path.join(SAVE_DIR, '{}_outlier.png'.format(name)), np.uint8(output))

    # 5 times iterative region voting
    print('* Iterative region voting')
    for _ in range(6):
        labels, outlier = iterative_region_voting(window_l, labels, outlier)

    if DEBUG:
        cv2.imwrite(os.path.join(SAVE_DIR, '{}_iter_region_voting.png'.format(name)), np.uint8(labels * scale_factor))
        output = np.copy(Il)
        output[outlier != 0] = 0
        output[outlier == 1, 0] = 255
        output[outlier == 2, 1] = 255
        cv2.imwrite(os.path.join(SAVE_DIR, '{}_outlier_after_voting.png'.format(name)), np.uint8(output))

    print('* Proper interpolation')
    labels = proper_interpolation(Il, labels, outlier)
    if DEBUG:
        cv2.imwrite(os.path.join(SAVE_DIR, '{}_proper_interpolation.png'.format(name)), np.uint8(labels * scale_factor))

    print('* Depth discontinuity adjustment')
    labels = depth_discontinuity_adjustment(labels, cost_total)
    if DEBUG:
        cv2.imwrite(os.path.join(SAVE_DIR, '{}_depth_discont_adjust.png'.format(name)), np.uint8(labels * scale_factor))

    print('* Subpixel enhancement')
    labels = subpixel_enhancement(labels, cost_total)
    if DEBUG:
        cv2.imwrite(os.path.join(SAVE_DIR, '{}_subpixel_enhancement.png'.format(name)), np.uint8(labels * scale_factor))

    labels = median_filter(labels, size=3, mode='constant')

    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return labels


def main():
    global name, scale_factor
    name = 'tsukuba'
    print('Tsukuba')  # (288, 384, 3)
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 16
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite(os.path.join(SAVE_DIR, 'tsukuba.png'), np.uint8(labels * scale_factor))

    name = 'venus'
    print('Venus')  # (383, 434, 3)
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite(os.path.join(SAVE_DIR, 'venus.png'), np.uint8(labels * scale_factor))

    name = 'teddy'
    print('Teddy')  # (375, 450, 3)
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite(os.path.join(SAVE_DIR, 'teddy.png'), np.uint8(labels * scale_factor))

    name = 'cones'
    print('Cones')  # (375, 450, 3)
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite(os.path.join(SAVE_DIR, 'cones.png'), np.uint8(labels * scale_factor))


if __name__ == '__main__':
    main()
