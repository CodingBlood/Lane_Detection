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
    print(image_of_car.shape)
    if ret:
        # Making Copy Of original frame of video to be user later onn
        original_image = np.copy(image_of_car)
        # converting the RGB image to Grayscale with OPENCV function
        image_of_car = cv2.cvtColor(image_of_car, cv2.COLOR_RGB2GRAY)
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
        x_start = (540 - res1.intercept) / res1.slope
        y_start = 540
        x_end = x_start + q
        y_end = y_start + li
        # Drawing The Lane Line 1 on The Masked image
        lane_line_image = cv2.line(lane_line_image, (int(x_start), int(y_start)),
                                   (int(x_end), int(y_end)), (150, 10, 255), 10)
        # # FOR LANE NO 2
        li = 310*np.sin(np.arctan(res2.slope))
        q = 310*np.cos(np.arctan(res2.slope))
        x_start = (540 - res2.intercept) / res2.slope
        y_start = 540
        x_end = x_start - q
        y_end = y_start - li
        # Drawing The Lane Line 2 on The Masked image
        lane_line_image = cv2.line(lane_line_image, (int(x_start), int(y_start)),
                                   (int(x_end), int(y_end)), (150, 10, 255), 10)

        # Combining Both masked image with lane markings with original image Initially Stored
        combo = cv2.addWeighted(lane_line_image, 1, original_image, 1, 0)
        # Displaying The Final Image
        cv2.imshow('Lane Detection V1.0', combo)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# <--------------------------------------------------------------------------------------------------->

# <----OLDER VERSion----------------------------------------------------------------------------------------------?
# lane_video = cv2.VideoCapture('./Testing_data/solidWhiteRight.mp4')
# if not lane_video.isOpened():
#     print("Error In Opening Vide Source File")
# while lane_video.isOpened():
#     ret, image_of_car = lane_video.read()
#     if ret:
#         # Reading The Image Stored in Testing_data folder with OPENCV
#         original_image = np.copy(image_of_car)
#         # print(original_image.shape)
#         # converting the RGB image to Grayscale with OPENCV function
#         image_of_car = cv2.cvtColor(image_of_car, cv2.COLOR_BGR2GRAY)
#         # Define a kernel size and apply Gaussian smoothing
#         kernel_size = 5
#         image_of_car = cv2.GaussianBlur(image_of_car, (kernel_size, kernel_size), 0)
#         # Define our parameters for Canny and apply
#         low_threshold = 66.67
#         high_threshold = 200
#         image_of_car = cv2.Canny(image_of_car, low_threshold, high_threshold)
#         # APPLYING MASKING IN LOWER TRIANGLE
#         image_of_mask = np.copy(image_of_car)
#         image_of_mask[:, :] = 0
#         # print("this is the shape of mask image")
#         # a = image_of_mask.shape[0]
#         # b = image_of_mask.shape[1]
#         # For Video USE THIS REMEMBER IMPORTANT
#         # a = 1280
#         # b = 720
#         # pt1 = (0, 720)
#         # pt2 = (1280, 720)
#         # pt3 = (640, 360)
#         # triangle_cnt = np.array([pt1, pt2, pt3])
#         a = 960
#         b = 540
#         pt1 = (0, 540)
#         pt2 = (960, 540)
#         pt3 = (480, 300)
#         triangle_cnt = np.array([pt1, pt2, pt3])
#         cv2.drawContours(image_of_mask, [triangle_cnt], 0, (255, 255, 255), -1)
#         image_of_car = cv2.bitwise_and(image_of_mask, image_of_car)
#         # cv2.imshow('masked',image_of_car)
#         # cv2.waitKey(0)
#         # Define the Hough transform parameters
#         rho = 1
#         theta = np.pi / 180
#         threshold = 1
#         min_line_length = 200
#         max_line_gap = 120
#         # #<<--actual---->>
#         # min_line_length = 10
#         # max_line_gap = 1
#         # Make a blank the same size as our image to draw on
#         lane_line_image = np.copy(image_of_car) * 0
#         lines = cv2.HoughLinesP(image_of_car, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
#         # Iterate over the output "lines" and draw lines on the blank
#         lane_line_image = cv2.cvtColor(lane_line_image, cv2.COLOR_GRAY2BGR)
#         image_of_car1 = cv2.cvtColor(image_of_car, cv2.COLOR_GRAY2BGR)
#         x = []
#         y = []
#         for line in lines:
#             print(line)
#             for x1, y1, x2, y2 in line:
#                 # x.append(x1)
#                 # x.append(x2)
#                 # y.append(-y1)
#                 # y.append(-y2)
#                 lane_line_image = cv2.line(lane_line_image, (x1, y1), (x2, y2), (150, 10, 255), 10)
#         # Draw the lines on the edge image
#         # res = stats.linregress(x, y)
#         # li = np.sin(np.arctan(res.slope))
#         # q = np.cos(np.arctan(res.slope))
#         # x_start = (540 - res.intercept) / res.slope
#         # y_start = 540
#         # x_end = x_start + q
#         # y_end = y_start + li
#         # print(-x_start, y_start, -x_end, y_end)
#         # lane_line_image = cv2.line(lane_line_image, (int(x_start), int(y_start)),
#                                     (int(x_end), int(y_end)), (150, 10, 255), 10)
#         combo = cv2.addWeighted(lane_line_image, 1, original_image, 1, 0)
#         cv2.imshow('FRAME', combo)
#         # cv2.waitKey(0)
#         # Press Q on keyboard to  exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else:
#         break

# <--------------------------------------------------------------------------------------------------->
# # Reading The Image Stored in Testing_data folder with OPENCV
# image_of_car = cv2.imread('./Testing_data/solidYellowCurve2.jpg')
# original_image = np.copy(image_of_car)
# # converting the RGB image to Grayscale with OPENCV function
# image_of_car = cv2.cvtColor(image_of_car, cv2.COLOR_RGB2GRAY)
# # Define a kernel size and apply Gaussian smoothing
# kernel_size = 5
# image_of_car = cv2.GaussianBlur(image_of_car, (kernel_size, kernel_size), 0)
# # Define our parameters for Canny and apply
# low_threshold = 100
# high_threshold = 200
# image_of_car = cv2.Canny(image_of_car, low_threshold, high_threshold)
# image_of_mask = np.copy(image_of_car)
# image_of_mask[:, :] = 0
# print("this is the shape of mask image")
# a = image_of_mask.shape[0]
# b = image_of_mask.shape[1]
# # For Video USE THIS REMEMBER IMPORTANT
# # pt1 = (0, 720)
# # pt2 = (1280, 720)
# # pt3 = (640, 360)
# pt1 = (0, 540)
# pt2 = (960, 540)
# pt3 = (480, 270)
# triangle_cnt = np.array([pt1, pt2, pt3])
# cv2.drawContours(image_of_mask, [triangle_cnt], 0, (255, 255, 255), -1)
# image_of_car = cv2.bitwise_and(image_of_mask,image_of_car)
# # Define the Hough transform parameters
# rho = 1
# theta = np.pi / 180
# threshold = 1
# min_line_length = 9
# max_line_gap = 3
# # #<<--actual---->>
# # min_line_length = 10
# # max_line_gap = 1
# # Make a blank the same size as our image to draw on
# lane_line_image = np.copy(image_of_car) * 0
# lines = cv2.HoughLinesP(image_of_car, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
# # Iterate over the output "lines" and draw lines on the blank
# lane_line_image = cv2.cvtColor(lane_line_image, cv2.COLOR_GRAY2RGB)
# image_of_car1 = cv2.cvtColor(image_of_car, cv2.COLOR_GRAY2RGB)
# for line in lines:
#     print(line)
#     for x1, y1, x2, y2 in line:
#         lane_line_image = cv2.line(lane_line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
#
# # Draw the lines on the edge image
# combo = cv2.addWeighted(lane_line_image, 0.8, original_image, 1, 0)
#
# # cv2.imshow('road', image_of_car)
# # cv2.waitKey(0)
# # cv2.imshow('road1', lane_line_image)
# # cv2.waitKey(0)
# cv2.imshow('road2', combo)
# cv2.waitKey(0)
