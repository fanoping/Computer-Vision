import numpy as np
import cv2
import argparse


def main(args):
    """
    # video reader ref: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    # feature matching ref: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # keypoint object ref: https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html
    # DMatch object ref: https://docs.opencv.org/3.4.3/d4/de0/classcv_1_1DMatch.html
    """

    # read source files
    cap = cv2.VideoCapture(args.video)
    marker = cv2.imread(args.marker)
    marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
    image = cv2.imread(args.image, 0)

    # feature detector
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # marker features
    marker_keypoint, marker_description = orb.detectAndCompute(marker, None)

    # start reading video and matching position
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # frame features
        frame_keypoint, frame_description = orb.detectAndCompute(frame, None)
        # find matching descriptor
        matches = matcher.match(marker_description, frame_description)
        # sort them in order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # check match > min count of match features
        if len(matches) > 10:
            marker_pts = np.array([marker_keypoint[ma.queryIdx].pt for ma in matches]).reshape(-1, 1, 2)
            frame_pts = np.array([frame_keypoint[ma.trainIdx].pt for ma in matches]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(marker_pts, frame_pts, cv2.RANSAC, 5.0)

            h, w = marker.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # project corners into frame
            dst = cv2.perspectiveTransform(pts, homography)
            # connect them with lines
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            frame = cv2.drawMatches(marker, marker_keypoint, frame, frame_keypoint, matches[:10], 0, flags=2)

        cv2.imshow('frame', frame)
        input()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, default="./input/ar_marker.mp4")
    parser.add_argument("-m", "--marker", type=str, default="./input/marker.png")
    parser.add_argument("-i", "--image", type=str, default="./input/wu.jpg")
    main(parser.parse_args())
