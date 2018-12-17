import numpy as np
import cv2
import time
import math
"""
    r = 5
    # TODO: Compute matching cost from Il and Ir
    for y in range(r, h-r):
        print("y", y)
        for x in range(r, w-r):
            best_label = 0
            curr_distance = math.inf

            for disp in range(max_disp):
                dist = 0

                for v in range(-r, r):
                    for u in range(-r, r):
                        d = int(Il[y+v, x+u]) - int(Ir[y+v, x+u - disp])
                        dist += d * d
                if dist < curr_distance:
                    curr_distance = dist
                    best_label = disp
            labels[y, x] = best_label
    """

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost computation
    print("* Cost computation (initialization)")
    tic = time.time()

    # initial cost computation
    r_y, r_x = 7, 9
    cost_census = np.ones((h, w, max_disp + 1), dtype=np.float32) * math.inf
    cost_ad = np.ones((h, w, max_disp + 1), dtype=np.float32) * math.inf

    for y in range(r_y // 2, h- (r_y // 2)):
        for x in range(r_x // 2, w- (r_x //2)):
            for disp in range(max_disp + 1):
                if x - (r_x // 2) - disp < 0:
                    break
                # initial cost computation C_AD
                c_ad = np.sum(abs(Il[y, x] - Ir[y, x - disp])) / 3.
                cost_ad[y, x, disp] = c_ad

                # initial cost computation C_CENSUS
                # left image window
                c_census_l = Il[y - (r_y // 2): y + (r_y // 2) + 1, x - (r_x // 2): x + (r_x // 2) + 1] > Il[y, x]
                # right image window
                c_census_r = Ir[y - (r_y // 2): y + (r_y // 2) + 1, x - (r_x // 2) - disp: x + (r_x // 2) + 1 - disp] > Ir[y, x]
                # Hamming distance calculation
                c_census = (~c_census_l ^ ~c_census_r) & (c_census_l ^ c_census_r)
                c_census = float(np.sum(c_census)) / 3.
                cost_census[y, x, disp] = c_census

    lambda_census = 30.
    lambda_ad = 10.

    norm_ad = 1 - np.exp(-cost_ad / lambda_ad)
    norm_census = 1 - np.exp(-cost_census / lambda_census)
    cost_total = norm_census + norm_ad

    label = np.array([[np.argmin(cost_total[y, x]) for x in range(w)] for y in range(h)])
    cv2.imwrite('test_cd.png', np.uint8(label * 16))

    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    # cross-based aggregation method
    tao_1, tao_2 = 20, 6
    l1, l2 = 34, 17



    for idx in range(1, 3):
        print('* Cost aggregation ({})'.format(idx))

        # pre-compute aggregated cost horizontally
        #cummulated_cost_horizontal = np.zeros_like(cost_total)
        #for y, line in enumerate(cost_total):
        #    for idx in range(len(line)):
        #        cummulated_cost_horizontal[y, idx] = np.sum(cost_total[y, 0: idx + 1], axis=0)

        def arm_check(p_y, p_x, q_y, q_x, edge_term):
            """
                Rule 1:
                Extend the arm such that the difference of the color intensity is smaller than the threshold tao_1
            """
            if max(abs(Il[p_y, p_x] - Il[q_y, q_x])) >= tao_1:
                return False
            if max(abs(Il[q_y, q_x] - Il[q_y + edge_term[0], q_x + edge_term[1]])) >= tao_1:
                return False
            """
                Rule 2:
                Aggregation window < L1
            """
            if abs(q_y - p_y) >= l1 or abs(q_x - p_x) >= l1:
                return False
            """
                Rule 3:
                Local information constraint
            """
            if abs(q_y - p_y) > l2 or abs(q_x - p_x) > l2:
                if max(abs(Il[p_y, p_x] - Il[q_y, q_x])) >= tao_2:
                    return False
            return True

        cost_total_aggregated = np.copy(cost_total)
        for y in range(r_y // 2, h - (r_y // 2)):
            for x in range(r_x // 2, w - (r_x // 2)):
                print(y, x)
                up_arm, lower_arm = 0, 0
                while arm_check(y, x, y - up_arm - 1, x, [1, 0]):
                    up_arm += 1
                while arm_check(y, x, y + lower_arm + 1, x, [-1, 0]):
                    lower_arm += 1
                    if y + lower_arm + 1 >= h:
                        break

                p_cost = 0
                for vertical in range(-up_arm, lower_arm + 1):
                    left_arm, right_arm = 0, 0
                    while arm_check(y + vertical, x, y + vertical, x - left_arm - 1, [0, 1]):
                        left_arm += 1
                        if x - left_arm - 1 < 0:
                            break
                    while arm_check(y + vertical, x, y + vertical, x + right_arm + 1, [0, -1]):
                        right_arm += 1
                        if x + right_arm + 1 >= w:
                            break

                    for horizontal in range(-left_arm, right_arm + 1):
                        p_cost += cost_total[y + vertical, x + horizontal]

                cost_total_aggregated[y ,x] = p_cost

        cost_total_aggregated_v = np.copy(cost_total_aggregated)
        for y in range(r_y // 2, h - (r_y // 2)):
            for x in range(r_x // 2, w - (r_x // 2)):
                left_arm, right_arm = 0, 0
                while arm_check(y, x, y, x - left_arm - 1, [0, 1]):
                    left_arm += 1
                    if x - left_arm - 1 < 0:
                        break
                while arm_check(y, x, y, x + right_arm + 1, [0, -1]):
                    right_arm += 1
                    if x + right_arm + 1 >= w:
                        break

                p_cost = 0
                for horizontal in range(-left_arm, right_arm + 1):
                    up_arm, lower_arm = 0, 0
                    while arm_check(y, x + horizontal, y - up_arm - 1, x + horizontal, [1, 0]):
                        up_arm += 1
                    while arm_check(y, x + horizontal, y + lower_arm + 1, x + horizontal, [-1, 0]):
                        lower_arm += 1
                        if y + lower_arm + 1 >= h:
                            break

                    for vertical in range(-up_arm, lower_arm + 1):
                        p_cost += cost_total_aggregated[y + vertical, x + horizontal]

                cost_total_aggregated_v[y ,x] = p_cost


        cost_total = np.copy(cost_total_aggregated_v)
        label = np.array([[np.argmin(cost_total[y, x]) for x in range(w)] for y in range(h)])
        cv2.imwrite('test_c{}.png'.format(idx), np.uint8(label * 16))

    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    labels = np.array([[np.argmin(cost_total[y, x]) for x in range(w)] for y in range(h)])

    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return labels


def main():

    print('Tsukuba')  # (288, 384, 3)
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('tsukuba1.png', np.uint8(labels * scale_factor))

    print('Venus')  # (383, 434, 3)
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus1.png', np.uint8(labels * scale_factor))

    print('Teddy')  # (375, 450, 3)
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy1.png', np.uint8(labels * scale_factor))

    print('Cones')  # (375, 450, 3)
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones1.png', np.uint8(labels * scale_factor))


if __name__ == '__main__':
    main()
