# Importing Statements
# importing Open CV Library
import cv2
# importing Numpy Library
import numpy as np
# importing stats from Scipy Library for implementing Linear Regression
from scipy import stats
# <-------------------------------------------------------------------------------------------------->
# importing Video File Stored in Testing_data folder with OPENCV
lane_video = cv2.VideoCapture('./Testing_data/solidYellowLeft.mp4')
if not lane_video.isOpened():
    # Error Handling In case of Corrupted Video
    print("Error In Opening Vide Source File")
while lane_video.isOpened():
    # Splitting Frames Of The Video
    ret, image_of_car = lane_video.read()
    if ret:
# def hello():
        # image_of_car: None = cv2.imread('./Testing_data/solidYellowLeft.jpg')
        # Making Copy Of original frame of video to be user later onn
        original_image = np.copy(image_of_car)
        # converting the RGB image to Grayscale with OPENCV function
        image_of_car = cv2.cvtColor(image_of_car, cv2.COLOR_RGB2GRAY)
        # image_of_car = other_image_of_car[1::]
        # Define a kernel size and apply Gaussian smoothing
        kernel_size = 5
        image_of_car = cv2.GaussianBlur(image_of_car, (kernel_size, kernel_size), 0)
        # Define our parameters for Canny and apply
        low_threshold = 66.67
        high_threshold = 200
        image_of_car = cv2.Canny(image_of_car, low_threshold, high_threshold)
        # APPLYING MASKING IN LOWER TRIANGLE
        image_of_mask = np.copy(image_of_car)
        image_of_mask[:, :] = 0
        a = 960
        b = 540
        pt1 = (0, 540)
        pt2 = (960, 540)
        pt3 = (480, 300)
        triangle_cnt = np.array([pt1, pt2, pt3])
        cv2.drawContours(image_of_mask, [triangle_cnt], 0, (255, 255, 255), -1)
        image_of_car = cv2.bitwise_and(image_of_mask, image_of_car)
        # Define the Hough transform parameters
        rho = 1
        theta = np.pi / 180
        threshold = 1
        min_line_length = 10
        max_line_gap = 1
        # Make a blank the same size as our image to draw on
        lane_line_image = np.copy(image_of_car) * 0
        # Finding The Straight lines Using Hough Transform taking min_line_length as 10 and max_line_gap as 1
        lines = cv2.HoughLinesP(image_of_car, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        # Converting Image To BGR from GRAY do apply Create Our Final Image
        lane_line_image = cv2.cvtColor(lane_line_image, cv2.COLOR_GRAY2RGB)
        image_of_car1 = cv2.cvtColor(image_of_car, cv2.COLOR_GRAY2RGB)
        # Declaring Variable To Store Coordinates for applying Linear Regression
        xl, yl, xr, yr = [], [], [], []
        # Iterate over the output "lines" and draw lines on the blank
        for line in lines:
            # Dividing The image into two parts for both left and right lane
            for x1, y1, x2, y2 in line:
                if x1 >= a/2:
                    xr.append(x1)
                    yr.append(y1)
                else:
                    xl.append(x1)
                    yl.append(y1)
                if x2 >= a/2:
                    xr.append(x2)
                    yr.append(y2)
                else:
                    xl.append(x2)
                    yl.append(y2)

        # Applying Linear Regression And Finding Initial and Final Coordinates Of both lane markings
        res1 = stats.linregress(xl, yl)
        res2 = stats.linregress(xr, yr)
        # # FOR LANE NO 1
        li = 310*np.sin(np.arctan(res1.slope))
        q = 310*np.cos(np.arctan(res1.slope))
        x_start1 = (540 - res1.intercept) / res1.slope
        y_start1 = 540
        x_end1 = x_start1 + q
        y_end1 = y_start1 + li
        # # FOR LANE NO 2
        li = 310*np.sin(np.arctan(res2.slope))
        q = 310*np.cos(np.arctan(res2.slope))
        x_start = (540 - res2.intercept) / res2.slope
        y_start = 540
        x_end = x_start - q
        y_end = y_start - li
        # src = np.float32(
        #     [
        #         [y_start1, x_start1],
        #         [y_end1, x_end1],
        #         [y_start, x_start],
        #         [y_end, x_end]
        #     ]
        # )
        # dist = np.float32(
        #     [
        #         [y_start1, x_start1],
        #         [y_start1 - 310, x_start1],
        #         [y_start1, x_start1 + 400],
        #         [y_start1 - 310, x_start1 + 400]
        #     ]
        # )
        src = np.float32(
            [
                [x_start1, y_start1],
                [x_end1, y_end1],
                [x_start, y_start],
                [x_end, y_end]
            ]
        )
        dist = np.float32(
            [
                [x_start1, y_start1],
                [x_start1, y_start1 - 310],
                [x_start1 + 400, y_start1],
                [x_start1 + 400, y_start1 - 310]
            ]
        )
        img_size = {original_image.shape[1], original_image.shape[0]}
        M = cv2.getPerspectiveTransform(src, dist)
        warped = cv2.warpPerspective(original_image, M, tuple(img_size), flags=cv2.INTER_LINEAR)
        cv2.imshow('frame', warped)
        cv2.waitKey(0)

    else:
        break
