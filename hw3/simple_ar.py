import numpy as np
import cv2
import argparse


def main(args):
    """
    ******************************** Useful links ********************************
    VideoCapture:
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    feature matching:
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    keypoint object:
        https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html
    DMatch object:
        https://docs.opencv.org/3.4.3/d4/de0/classcv_1_1DMatch.html
    VideoWrite:
        https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html
    ******************************************************************************
    """

    # read source files
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, size)

    marker = cv2.imread(args.marker)  # 410 * 410 * 3
    image = cv2.imread(args.image)

    # feature detector
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.BFMatcher()

    # marker features
    marker_keypoint, marker_description = sift.detectAndCompute(marker, None)

    # start reading video and matching position
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame features
        frame_keypoint, frame_description = sift.detectAndCompute(frame, None)
        # find matching descriptor
        matches = matcher.knnMatch(marker_description, frame_description, k=2)

        # find good matches
        goodmatches = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                goodmatches.append([m])

        # match the coordinate with larger value by multiplying a constant factor
        contraction_factor = 2.5
        # check match > min count of match features
        if len(goodmatches) > 10:
            marker_pts = np.array([marker_keypoint[ma[0].queryIdx].pt for ma in goodmatches]).reshape(-1, 1, 2)
            frame_pts = np.array([frame_keypoint[ma[0].trainIdx].pt for ma in goodmatches]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(marker_pts * contraction_factor, frame_pts, cv2.RANSAC, 5.0)

            if args.detect:
                h, w, _ = marker.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frames
                dst = cv2.perspectiveTransform(pts, homography)
                # connect them with lines
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            else:
                h, w, _ = image.shape
                # project image to video
                edge = 20
                for y in range(h):
                    for x in range(w):
                        new_pos = np.dot(homography, np.array([[x + edge, y + edge, 1]]).T)
                        new_x, new_y = int(new_pos[0, 0] / new_pos[2, 0]), int(new_pos[1, 0] / new_pos[2, 0])

                        # fill out points surrounded (new_x, new_y)
                        frame[new_y, new_x] = image[y, x]
                        frame[new_y + 1, new_x] = image[y, x]
                        frame[new_y + 1, new_x + 1] = image[y, x]
                        frame[new_y, new_x + 1] = image[y, x]

        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, default="./input/ar_marker.mp4")
    parser.add_argument("-m", "--marker", type=str, default="./input/marker.png")
    parser.add_argument("-i", "--image", type=str, default="./input/wu.jpg")
    parser.add_argument("-o", "--output", type=str, default="output.mp4")

    # other feature
    parser.add_argument("-d", "--detect", action="store_true",
                        help="For detect corresponding image in the video")
    main(parser.parse_args())
