import numpy as np
import cv2
import time
import math
DEBUG = 1

"""
    * TO BE UPDATED.....
    run time and bad pixel ratio results (initialization / aggregation / optimization / refinement / bad pixel ratio)
    Tsukuba: 50.49/1160.8/59.18/-/4.77
    Venus: 83.50/2198.3/124.57/-/4.43
    Teddy: 141.51/3024.0/-/-/17.76
    Cones: 144.93/1268.0/-/-/14.26
"""


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    Il = Il.astype(np.float64)
    Ir = Ir.astype(np.float64)

    if DEBUG:
        import os
        if not os.path.exists('./results'):
            os.mkdir('./results')
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
                    cost_ad[disp, y, x] = np.sum(abs(Il[y, x] - Ir[y, x - disp])) / 3.

            # initial census cost
            index = 0
            for i in range(y - 3, y + 4):
                for j in range(x - 4, x + 5):
                    for k in range(3):
                        if 0 <= i < h and 0 <= j < w:
                            left_census[y, x, index] = Il[i, j, k] < Il[y, x, k]
                            right_census[y, x, index] = Ir[i, j, k] < Ir[y, x, k]
                        else:
                            left_census[y, x, index] = random_int
                            right_census[y, x, index] = random_int
                        index += 1

    for disp in range(max_disp):
        cost_census[disp, :, disp:] = np.sum(left_census[:, disp:] != right_census[:, :w - disp], 2)

    lambda_census = 30.
    lambda_ad = 10.

    norm_ad = 1 - np.exp(-cost_ad / lambda_ad)
    norm_census = 1 - np.exp(-cost_census / lambda_census)
    cost_total = norm_census + norm_ad

    if DEBUG:
        label = cost_total.argmin(0).astype(np.float64)
        cv2.imwrite('./results/{}_adcost.png'.format(name), np.uint8(label * scale_factor))

    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    # cross-based aggregation method
    tao_1, tao_2 = 20, 6
    l1, l2 = 34, 17

    def arm_check(p_y, p_x, q_y, q_x, edge_term, img):
        # basic check, window size > pixel size
        if not (0 <= q_y < h and 0 <= q_x < w):
            return False
        if abs(q_y - p_y) == 1 or abs(q_x - p_x) == 1:
            return True

        # Rule 1: Extend the arm such that the difference of the color intensity is smaller than the threshold tao_1
        if max(abs(img[p_y, p_x] - img[q_y, q_x])) >= tao_1:
            return False
        if max(abs(img[q_y, q_x] - img[q_y + edge_term[0], q_x + edge_term[1]])) >= tao_1:
            return False

        # Rule 2: Aggregation window < L1
        if abs(q_y - p_y) >= l1 or abs(q_x - p_x) >= l1:
            return False

        # Rule 3: Local information constraint
        if abs(q_y - p_y) >= l2 or abs(q_x - p_x) >= l2:
            if max(abs(img[p_y, p_x] - img[q_y, q_x])) >= tao_2:
                return False

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

    def cross_based_cost_aggrgation(wind_l, wind_r, prev_cost, vertical=True):
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
                            up_edge = max(wind_l[y, x, 0], wind_r[y, x - disp, 0]) + 1
                            down_edge = min(wind_l[y, x, 1], wind_r[y, x - disp, 1])
                            for verti in range(up_edge, down_edge):
                                p_cost += prev_cost[disp, verti, hori]
                                pixel_cnt += 1
                    else:
                        up_edge = max(wind_l[y, x, 0], wind_r[y, x - disp, 0]) + 1
                        down_edge = min(wind_l[y, x, 1], wind_r[y, x - disp, 1])
                        for verti in range(up_edge, down_edge):
                            left_edge = max(wind_l[y, x, 2], wind_r[y, x - disp, 2] + disp) + 1
                            right_edge = min(wind_l[y, x, 3], wind_r[y, x - disp, 3] + disp)
                            for hori in range(left_edge, right_edge):
                                p_cost += prev_cost[disp, verti, hori]
                                pixel_cnt += 1
                    assert(pixel_cnt > 0)
                    after_cost[disp, y, x] = p_cost / pixel_cnt
        return after_cost

    window_l = window_def(Il)
    window_r = window_def(Ir)

    for idx in range(1, 3):
        # do horizontal and vertical aggregation alternatively
        print('* Cost aggregation ({})'.format(idx))
        cost_total = cross_based_cost_aggrgation(window_l, window_r, cost_total, False)
        if DEBUG:
            label = cost_total.argmin(0).astype(np.float64)
            cv2.imwrite('./results/{}_adcost_hori_{}.png'.format(name, idx), np.uint8(label * scale_factor))

        cost_total = cross_based_cost_aggrgation(window_l, window_r, cost_total, True)
        if DEBUG:
            label = cost_total.argmin(0).astype(np.float64)
            cv2.imwrite('./results/{}_adcost_verti_{}.png'.format(name, idx), np.uint8(label * scale_factor))

    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    if DEBUG:
        label = cost_total.argmin(0).astype(np.float64)
        cv2.imwrite('./results/{}_adcost_agg.png'.format(name), np.uint8(label * scale_factor))

    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    # Scanline optimization: semi-global matching
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
                    d_1 = max(abs(Il[y, x] - Il[y, x - 1]))
                    d_2 = max(abs(Ir[y, x - disp] - Ir[y, x - disp - 1]))

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
                    d_1 = max(abs(Il[y, x] - Il[y, x + 1]))
                    d_2 = max(abs(Ir[y, x - disp] - Ir[y, x - disp + 1]))

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
                if y - 1 < 0 or y - disp < 0:
                    result[disp, y, x] = cost_total[disp, y, x]
                else:
                    # difference between neighboring
                    d_1 = max(abs(Il[y, x] - Il[y - 1, x]))
                    d_2 = max(abs(Ir[y, x - disp] - Ir[y - 1, x - disp]))

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
                if y + 1 >= h or y - disp < 0:
                    result[disp, y, x] = cost_total[disp, y, x]
                else:
                    # difference between neighboring
                    d_1 = max(abs(Il[y, x] - Il[y + 1, x]))
                    d_2 = max(abs(Ir[y, x - disp] - Ir[y + 1, x - disp]))

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
        cv2.imwrite('./results/{}_adcost_optim.png'.format(name), np.uint8(label * scale_factor))

    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    labels = cost_total.argmin(0).astype(np.float64)
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
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))

    name = 'venus'
    print('Venus')  # (383, 434, 3)
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))

    name = 'teddy'
    print('Teddy')  # (375, 450, 3)
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    name = 'cones'
    print('Cones')  # (375, 450, 3)
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))


if __name__ == '__main__':
    main()
