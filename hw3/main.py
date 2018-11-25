import numpy as np
import cv2


# u, v are N-by-2 matrices (x, y), representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    # A = np.zeros((2*N, 8))
    # if you take solution 2:
    # A = np.zeros((2*N, 9))
    # b = np.zeros((2*N, 1))
    # H = np.zeros((3, 3))
    # TODO: compute H from A and b
    A = np.array([[u[0][0], u[0][1], 1, 0, 0, 0, -u[0][0] * v[0][0], -u[0][1] * v[0][0]],
                  [0, 0, 0, u[0][0], u[0][1], 1, -u[0][0] * v[0][1], -u[0][1] * v[0][1]],
                  [u[1][0], u[1][1], 1, 0, 0, 0, -u[1][0] * v[1][0], -u[1][1] * v[1][0]],
                  [0, 0, 0, u[1][0], u[1][1], 1, -u[1][0] * v[1][1], -u[1][1] * v[1][1]],
                  [u[2][0], u[2][1], 1, 0, 0, 0, -u[2][0] * v[2][0], -u[2][1] * v[2][0]],
                  [0, 0, 0, u[2][0], u[2][1], 1, -u[2][0] * v[2][1], -u[2][1] * v[2][1]],
                  [u[3][0], u[3][1], 1, 0, 0, 0, -u[3][0] * v[3][0], -u[3][1] * v[3][0]],
                  [0, 0, 0, u[3][0], u[3][1], 1, -u[3][0] * v[3][1], -u[3][1] * v[3][1]]])
    b = np.array([[v[0][0]],
                  [v[0][1]],
                  [v[1][0]],
                  [v[1][1]],
                  [v[2][0]],
                  [v[2][1]],
                  [v[3][0]],
                  [v[3][1]]])

    h = np.dot(np.linalg.inv(A), b)

    H = np.array([[h[0, 0], h[1, 0], h[2, 0]],
                  [h[3, 0], h[4, 0], h[5, 0]],
                  [h[6, 0], h[7, 0], 1]])
    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    # TODO: some magic
    img_corner = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    homography_matrix = solve_homography(img_corner, corners)

    for y in range(h):
        for x in range(w):
            new_pos = np.dot(homography_matrix, np.array([[x, y, 1]]).T)
            new_x, new_y = int(new_pos[0, 0] / new_pos[2, 0]), int(new_pos[1, 0] / new_pos[2, 0])
            canvas[new_y, new_x] = img[y, x]
    return canvas


def backward_warpping(img, output, corners):
    # TODO: some magic
    h, w, ch = output.shape
    img_corner = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    homography_matrix = solve_homography(img_corner, corners)
    for y in range(h):
        for x in range(w):
            new_pos = np.dot(homography_matrix, np.array([[x, y, 1]]).T)
            new_x, new_y = new_pos[0][0] / new_pos[2][0], new_pos[1][0] / new_pos[2][0]
            inter_pixel = interpolate(img, new_x, new_y)
            output[y, x] = inter_pixel

    return output


def interpolate(src_img, new_x, new_y):
    x_fraction = round(new_x - int(new_x), 3)
    y_fraction = round(new_y - int(new_y), 3)

    pixel = np.zeros((3,))
    pixel += (1 - x_fraction) * (1 - y_fraction) * src_img[int(new_y), int(new_x)]
    pixel += (1 - x_fraction) * y_fraction * src_img[int(new_y) + 1, int(new_x)]
    pixel += x_fraction * (1 - y_fraction) * src_img[int(new_y), int(new_x) + 1]
    pixel += x_fraction * y_fraction * src_img[int(new_y) + 1, int(new_x) + 1]

    return pixel


def main():
    part1, part2, part3 = False, True, False

    # Part 1
    if part1:
        print("========Part 1========")
        canvas = cv2.imread('./input/times_square.jpg')  # (1050, 1680, 3)
        img1 = cv2.imread('./input/wu.jpg')              # (348, 348, 3)
        img2 = cv2.imread('./input/ding.jpg')            # (960, 960, 3)
        img3 = cv2.imread('./input/yao.jpg')             # (774, 774, 3)
        img4 = cv2.imread('./input/kp.jpg')              # (900, 900, 3)
        img5 = cv2.imread('./input/lee.jpg')             # (962, 1442, 3)

        corners1 = np.array([[818, 352], [884, 352], [818, 407], [885, 408]])
        corners2 = np.array([[311, 14], [402, 150], [157, 152], [278, 315]])
        corners3 = np.array([[364, 674], [430, 725], [279, 864], [369, 885]])
        corners4 = np.array([[808, 495], [892, 495], [802, 609], [896, 609]])
        corners5 = np.array([[1024, 608], [1118, 593], [1032, 664], [1134, 651]])

        # TODO: some magic
        print("Transform ./input/wu.jpg...")
        canvas = transform(img1, canvas, corners1)
        print("Transform ./input/ding.jpg...")
        canvas = transform(img2, canvas, corners2)
        print("Transform ./input/yao.jpg...")
        canvas = transform(img3, canvas, corners3)
        print("Transform ./input/kp.jpg...")
        canvas = transform(img4, canvas, corners4)
        print("Transform ./input/lee.jpg...")
        canvas = transform(img5, canvas, corners5)
        print("Plot part1.png...")
        cv2.imwrite('part1.png', canvas)

    # Part 2
    if part2:
        print("========Part 2========")
        img = cv2.imread('./input/screen.jpg')  # 2000 * 1500
        output = np.zeros((300, 300, 3))

        # find position first
        print("Backward Warping ./input/screen.jpg...")
        corners = np.array([[1041, 370], [1100, 396], [984, 552], [1036, 599]])
        output = backward_warpping(img, output, corners)

        print("Plot part2.png...")
        cv2.imwrite('part2.png', output)

    # Part 3
    if part3:
        print("========Part 3========")
        img_front = cv2.imread('./input/crosswalk_front.jpg')   # 725 * 400
        output_img = np.zeros((400, 400, 3))

        print("Backward Warping ./input/crosswalk_front.jpg...")
        corners = np.array([[160, 129], [563, 129], [0, 286], [723, 286]])
        output_img = backward_warpping(img_front, output_img, corners)

        print("Plot part3.png...")
        cv2.imwrite('part3.png', output_img)

    print("======================")
    print("Process Done!")


if __name__ == '__main__':
    main()
