import cv2
import numpy as np


# ===========================================================
# Declaring a function to Make Undistorted Image
def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)
# ===========================================================

# ===========================================================
# Setting Up Calibration Points via ChesBoard of size 9,6
nx = 9
ny = 6
objpoints = []
imgpoints = []
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
# Finding in mtx, dst
img = cv2.imread('camera_cal/calibration2.jpg')
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
# If found, draw corners
if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
undistorted = undistort_image(img, mtx, dist)
# ===========================================================

# ===========================================================
# Creating a function to apply Absolute Sobel On Image
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    scaled_sobel = None
    # Sobel x
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Sobel y
    else:
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  # Take the derivative in y
        abs_sobely = np.absolute(sobely)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    # Threshold x gradient
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return grad_binary


# ===========================================================

# ===========================================================
# Applying Sobel and color Thresholding
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    abs_magnitude = np.absolute(magnitude)
    scaled_magnitude = np.uint8(255 * abs_magnitude / np.max(abs_magnitude))
    mag_binary = np.zeros_like(scaled_magnitude)
    mag_binary[(scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    arctan = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(arctan)
    dir_binary[(arctan >= thresh[0]) & (arctan <= thresh[1])] = 1
    return dir_binary


def combined_s_gradient_thresholds(img, show=False):
    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(20, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.4))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    # Threshold color channel
    s_thresh_min = 150
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1
    return combined_binary


# ===========================================================
# importing Video File Stored in Testing_data folder with OPENCV
lane_video = cv2.VideoCapture('./Testing_data/challenge_video.mp4')
# Declaring a variable to choose method of lane detection
i = 0

if not lane_video.isOpened():
    # Error Handling In case of Corrupted Video
    print("Error In Opening Vide Source File")
while lane_video.isOpened():
    # Splitting Frames Of The Video
    ret, image_of_car = lane_video.read()
    image_of_car = cv2.cvtColor(image_of_car, cv2.COLOR_BGR2RGB)
    # image_of_car = cv2.cvtColor(image_of_car, cv2.COLOR_RGB2HLS)
    print(image_of_car.shape)
    if ret:

        try:
            img = np.copy(image_of_car)
            # Grab the x and y size and make a copy of the image
            ysize = img.shape[0]
            xsize = img.shape[1]
            # Define our color selection criteria
            red_threshold = 160
            green_threshold = 100
            blue_threshold = 0
            rgb_threshold = [red_threshold, green_threshold, blue_threshold]
            # Identify pixels below the threshold
            thresholds = (img[:, :, 0] < rgb_threshold[0]) \
                         | (img[:, :, 1] < rgb_threshold[1]) \
                         | (img[:, :, 2] < rgb_threshold[2])
            img[thresholds] = [0, 0, 0]

            combined_binary = combined_s_gradient_thresholds(img, True)

            # Applying Transform to Image
            # Grab the image shape
            img_size = (combined_binary.shape[1], combined_binary.shape[0])
            leftupperpoint = [568, 470]
            rightupperpoint = [717, 470]
            leftlowerpoint = [260, 680]
            rightlowerpoint = [1043, 680]
            src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
            dst = np.float32([[200, 0], [200, 680], [1000, 0], [1000, 680]])
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # Warp the image using OpenCV warpPerspective()
            warped_img = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_NEAREST)


            # Defining a function to locate lines using 9 windows which are later used to calculate the curve
            def locate_lines(binary_warped, nwindows=9, margin=100, minpix=50):
                # Assuming you have created a warped binary image called "binary_warped"
                # Take a histogram of the bottom half of the image
                histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

                # Find the peak of the left and right halves of the histogram
                # These will be the starting point for the left and right lines
                midpoint = int(histogram.shape[0] / 2)
                leftx_base = np.argmax(histogram[:midpoint])
                rightx_base = np.argmax(histogram[midpoint:]) + midpoint

                # Set height of windows
                window_height = int(binary_warped.shape[0] / nwindows)
                # Identify the x and y positions of all nonzero pixels in the image
                nonzero = binary_warped.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                # Current positions to be updated for each window
                leftx_current = leftx_base
                rightx_current = rightx_base

                # Create empty lists to receive left and right lane pixel indices
                left_lane_inds = []
                right_lane_inds = []

                # Create an image to draw on and an image to show the selection window
                out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

                # Step through the windows one by one
                for window in range(nwindows):
                    # Identify window boundaries in x and y (and right and left)
                    win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                    win_y_high = binary_warped.shape[0] - window * window_height
                    win_xleft_low = leftx_current - margin
                    win_xleft_high = leftx_current + margin
                    win_xright_low = rightx_current - margin
                    win_xright_high = rightx_current + margin
                    # Draw the windows on the visualization image
                    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                                  (0, 255, 0), 2)
                    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                                  (0, 255, 0), 2)
                    # Identify the nonzero pixels in x and y within the window
                    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                    # Append these indices to the lists
                    left_lane_inds.append(good_left_inds)
                    right_lane_inds.append(good_right_inds)
                    # If you found > minpix pixels, recenter next window on their mean position
                    if len(good_left_inds) > minpix:
                        leftx_current = int(np.mean(nonzerox[good_left_inds]))
                    if len(good_right_inds) > minpix:
                        rightx_current = int(np.mean(nonzerox[good_right_inds]))

                # Concatenate the arrays of indices
                left_lane_inds = np.concatenate(left_lane_inds)
                right_lane_inds = np.concatenate(right_lane_inds)

                # Extract left and right line pixel positions
                leftx = nonzerox[left_lane_inds]
                lefty = nonzeroy[left_lane_inds]
                rightx = nonzerox[right_lane_inds]
                righty = nonzeroy[right_lane_inds]

                # Fit a second order polynomial to each
                left_fit = np.polyfit(lefty, leftx, 2)
                right_fit = np.polyfit(righty, rightx, 2)

                return left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy


            # Defining a function to locate lines using output of the previous function just less costly
            def locate_line_further(left_fit, right_fit, binary_warped):
                nonzero = binary_warped.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                margin = 50
                left_lane_inds = (
                        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin))
                        & (nonzerox < (
                        left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
                right_lane_inds = ((nonzerox > (
                        right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin))
                                   & (nonzerox < (
                                right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

                # Again, extract left and right line pixel positions
                leftx = nonzerox[left_lane_inds]
                lefty = nonzeroy[left_lane_inds]
                rightx = nonzerox[right_lane_inds]
                righty = nonzeroy[right_lane_inds]

                # Fit a second order polynomial to each
                if len(leftx) == 0:
                    left_fit_new = []
                else:
                    left_fit_new = np.polyfit(lefty, leftx, 2)

                if len(rightx) == 0:
                    right_fit_new = []
                else:
                    right_fit_new = np.polyfit(righty, rightx, 2)

                return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds, nonzerox, nonzeroy


            # Defining a function to display the lane lines for the calculation of curvature and position
            def visulizeLanes(left_fit, right_fit, left_lane_inds, right_lane_inds, binary_warped, nonzerox, nonzeroy,
                              margin=100):
                # Generate x and y values for plotting
                ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
                left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
                right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

                # Create an image to draw on and an image to show the selection window
                out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
                window_img = np.zeros_like(out_img)
                # Color in left and right line pixels
                out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
                out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

                # Generate a polygon to illustrate the search window area
                # And recast the x and y points into usable format for cv2.fillPoly()
                left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
                left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                                ploty])))])
                left_line_pts = np.hstack((left_line_window1, left_line_window2))
                right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
                right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                                 ploty])))])
                right_line_pts = np.hstack((right_line_window1, right_line_window2))

                # Draw the lane onto the warped blank image
                cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
                cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
                result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


            # calling the costly lane finding function every fifth time increase processing speed and increase accuracy
            left_fit1 = ()
            right_fit1 = ()
            if i % 5 == 0:
                left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = locate_lines(warped_img)
                left_fit1 = left_fit
                right_fit1 = right_fit
            else:
                left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = locate_line_further(
                    left_fit1, right_fit1, warped_img)
            # calling lane visualizing function to process image into bird eye perspective and calculate lane curvature
            visulizeLanes(left_fit, right_fit, left_lane_inds, right_lane_inds, warped_img, nonzerox, nonzeroy,
                          margin=100)


            # Function to calculate the radius of curvature
            def radius_curvature(binary_warped, left_fit, right_fit):
                ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
                left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
                right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

                # Define conversions in x and y from pixels space to meters
                ym_per_pix = 30 / 720  # meters per pixel in y dimension
                xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
                y_eval = np.max(ploty)

                # Fit new polynomials to x,y in world space
                left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
                right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

                # Calculate the new radii of curvature
                left_curvature = ((1 + (
                        2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                    2 * left_fit_cr[0])
                right_curvature = ((1 + (
                        2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                    2 * right_fit_cr[0])

                # Calculate vehicle center
                # left_lane and right lane bottom in pixels
                left_lane_bottom = (left_fit[0] * y_eval) ** 2 + left_fit[0] * y_eval + left_fit[2]
                right_lane_bottom = (right_fit[0] * y_eval) ** 2 + right_fit[0] * y_eval + right_fit[2]

                # Lane center as mid of left and right lane bottom
                lane_center = (left_lane_bottom + right_lane_bottom) / 2.
                center_image = 640
                center = (lane_center - center_image) * xm_per_pix  # Convert to meters
                position = "left" if center < 0 else "right"
                center = "Vehicle is {:.2f}m {}".format(center, position)

                # Now our radius of curvature is in meters
                return left_curvature, right_curvature, center


            # Calling the radius of curvature function and getting the required values
            left_curvature, right_curvature, center = radius_curvature(warped_img, left_fit, right_fit)


            # Function to draw the above calculate values on the image

            def draw_on_image(undist, warped_img, left_fit, right_fit, M, left_curvature, right_curvature, center,
                              show_values=False):
                ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
                left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
                right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

                # Create an image to draw the lines on
                warp_zero = np.zeros_like(warped_img).astype(np.uint8)
                color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

                # Recast the x and y points into usable format for cv2.fillPoly()x
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
                pts = np.hstack((pts_left, pts_right))

                # Draw the lane onto the warped blank image
                cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
                Minv = np.linalg.inv(M)
                # Warp the blank back to original image space using inverse perspective matrix (Minv)
                newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], img.shape[0]))
                # Combine the result with the original image
                result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

                cv2.putText(result, 'Left curvature: {:.0f} m'.format(left_curvature), (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1,
                            (255, 255, 255), 2)
                cv2.putText(result, 'Right curvature: {:.0f} m'.format(right_curvature), (50, 100),
                            cv2.FONT_HERSHEY_DUPLEX, 1,
                            (255, 255, 255), 2)
                cv2.putText(result, '{}'.format(center), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                return result


            # CConverting Image Back To RGB Format And Displaying it
            image_of_car = cv2.cvtColor(image_of_car, cv2.COLOR_BGR2RGB)
            img = draw_on_image(image_of_car, warped_img, left_fit, right_fit, M, left_curvature, right_curvature,
                                center, True)
            cv2.imshow('final', img)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # In the exception block we removed the color thresholding for considering shadow
        except:
            img = np.copy(image_of_car)
            combined_binary = combined_s_gradient_thresholds(img, True)

            # Applying Transform to Image
            # Grab the image shape
            img_size = (combined_binary.shape[1], combined_binary.shape[0])
            leftupperpoint = [568, 470]
            rightupperpoint = [717, 470]
            leftlowerpoint = [260, 680]
            rightlowerpoint = [1043, 680]
            src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
            dst = np.float32([[200, 0], [200, 680], [1000, 0], [1000, 680]])
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # Warp the image using OpenCV warpPerspective()
            warped_img = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_NEAREST)


            # Defining a function to locate lines using 9 windows which are later used to calculate the curve
            def locate_lines(binary_warped, nwindows=9, margin=100, minpix=50):
                # Assuming you have created a warped binary image called "binary_warped"
                # Take a histogram of the bottom half of the image
                histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

                # Find the peak of the left and right halves of the histogram
                # These will be the starting point for the left and right lines
                midpoint = int(histogram.shape[0] / 2)
                leftx_base = np.argmax(histogram[:midpoint])
                rightx_base = np.argmax(histogram[midpoint:]) + midpoint

                # Set height of windows
                window_height = int(binary_warped.shape[0] / nwindows)
                # Identify the x and y positions of all nonzero pixels in the image
                nonzero = binary_warped.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                # Current positions to be updated for each window
                leftx_current = leftx_base
                rightx_current = rightx_base

                # Create empty lists to receive left and right lane pixel indices
                left_lane_inds = []
                right_lane_inds = []

                # Create an image to draw on and an image to show the selection window
                out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

                # Step through the windows one by one
                for window in range(nwindows):
                    # Identify window boundaries in x and y (and right and left)
                    win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                    win_y_high = binary_warped.shape[0] - window * window_height
                    win_xleft_low = leftx_current - margin
                    win_xleft_high = leftx_current + margin
                    win_xright_low = rightx_current - margin
                    win_xright_high = rightx_current + margin
                    # Draw the windows on the visualization image
                    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                                  (0, 255, 0), 2)
                    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                                  (0, 255, 0), 2)
                    # Identify the nonzero pixels in x and y within the window
                    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                    # Append these indices to the lists
                    left_lane_inds.append(good_left_inds)
                    right_lane_inds.append(good_right_inds)
                    # If you found > minpix pixels, recenter next window on their mean position
                    if len(good_left_inds) > minpix:
                        leftx_current = int(np.mean(nonzerox[good_left_inds]))
                    if len(good_right_inds) > minpix:
                        rightx_current = int(np.mean(nonzerox[good_right_inds]))

                # Concatenate the arrays of indices
                left_lane_inds = np.concatenate(left_lane_inds)
                right_lane_inds = np.concatenate(right_lane_inds)

                # Extract left and right line pixel positions
                leftx = nonzerox[left_lane_inds]
                lefty = nonzeroy[left_lane_inds]
                rightx = nonzerox[right_lane_inds]
                righty = nonzeroy[right_lane_inds]

                # Fit a second order polynomial to each
                left_fit = np.polyfit(lefty, leftx, 2)
                right_fit = np.polyfit(righty, rightx, 2)

                return left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy


            # Defining a function to locate lines using output of the previous function just less costly
            def locate_line_further(left_fit, right_fit, binary_warped):
                nonzero = binary_warped.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                margin = 50
                left_lane_inds = (
                        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin))
                        & (nonzerox < (
                        left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
                right_lane_inds = ((nonzerox > (
                        right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin))
                                   & (nonzerox < (
                                right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

                # Again, extract left and right line pixel positions
                leftx = nonzerox[left_lane_inds]
                lefty = nonzeroy[left_lane_inds]
                rightx = nonzerox[right_lane_inds]
                righty = nonzeroy[right_lane_inds]

                # Fit a second order polynomial to each
                if len(leftx) == 0:
                    left_fit_new = []
                else:
                    left_fit_new = np.polyfit(lefty, leftx, 2)

                if len(rightx) == 0:
                    right_fit_new = []
                else:
                    right_fit_new = np.polyfit(righty, rightx, 2)

                return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds, nonzerox, nonzeroy


            # Defining a function to display the lane lines for the calculation of curvature and position
            def visulizeLanes(left_fit, right_fit, left_lane_inds, right_lane_inds, binary_warped, nonzerox, nonzeroy,
                              margin=100):
                # Generate x and y values for plotting
                ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
                left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
                right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

                # Create an image to draw on and an image to show the selection window
                out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
                window_img = np.zeros_like(out_img)
                # Color in left and right line pixels
                out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
                out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

                # Generate a polygon to illustrate the search window area
                # And recast the x and y points into usable format for cv2.fillPoly()
                left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
                left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                                ploty])))])
                left_line_pts = np.hstack((left_line_window1, left_line_window2))
                right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
                right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                                 ploty])))])
                right_line_pts = np.hstack((right_line_window1, right_line_window2))

                # Draw the lane onto the warped blank image
                cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
                cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
                result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


            # Locating the lane lines using costlier function Only
            left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = locate_lines(warped_img)
            visulizeLanes(left_fit, right_fit, left_lane_inds, right_lane_inds, warped_img, nonzerox, nonzeroy,
                          margin=100)


            # Function to calculate the radius of curvature
            def radius_curvature(binary_warped, left_fit, right_fit):
                ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
                left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
                right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

                # Define conversions in x and y from pixels space to meters
                ym_per_pix = 30 / 720  # meters per pixel in y dimension
                xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
                y_eval = np.max(ploty)

                # Fit new polynomials to x,y in world space
                left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
                right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

                # Calculate the new radii of curvature
                left_curvature = ((1 + (
                        2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                    2 * left_fit_cr[0])
                right_curvature = ((1 + (
                        2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                    2 * right_fit_cr[0])

                # Calculate vehicle center
                # left_lane and right lane bottom in pixels
                left_lane_bottom = (left_fit[0] * y_eval) ** 2 + left_fit[0] * y_eval + left_fit[2]
                right_lane_bottom = (right_fit[0] * y_eval) ** 2 + right_fit[0] * y_eval + right_fit[2]

                # Lane center as mid of left and right lane bottom
                lane_center = (left_lane_bottom + right_lane_bottom) / 2.
                center_image = 640
                center = (lane_center - center_image) * xm_per_pix  # Convert to meters
                position = "left" if center < 0 else "right"
                center = "Vehicle is {:.2f}m {}".format(center, position)

                # Now our radius of curvature is in meters
                return left_curvature, right_curvature, center


            # Calling the radius of curvature function and getting the required values
            left_curvature, right_curvature, center = radius_curvature(warped_img, left_fit, right_fit)


            # Function to draw the above calculate values on the image

            def draw_on_image(undist, warped_img, left_fit, right_fit, M, left_curvature, right_curvature, center,
                              show_values=False):
                ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
                left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
                right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

                # Create an image to draw the lines on
                warp_zero = np.zeros_like(warped_img).astype(np.uint8)
                color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

                # Recast the x and y points into usable format for cv2.fillPoly()
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
                pts = np.hstack((pts_left, pts_right))

                # Draw the lane onto the warped blank image
                cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
                Minv = np.linalg.inv(M)
                # Warp the blank back to original image space using inverse perspective matrix (Minv)
                newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], img.shape[0]))
                # Combine the result with the original image
                result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

                cv2.putText(result, 'Left curvature: {:.0f} m'.format(left_curvature), (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1,
                            (255, 255, 255), 2)
                cv2.putText(result, 'Right curvature: {:.0f} m'.format(right_curvature), (50, 100),
                            cv2.FONT_HERSHEY_DUPLEX, 1,
                            (255, 255, 255), 2)
                cv2.putText(result, '{}'.format(center), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                return result


            # CConverting Image Back To RGB Format And Displaying it
            image_of_car = cv2.cvtColor(image_of_car, cv2.COLOR_BGR2RGB)
            img = draw_on_image(image_of_car, warped_img, left_fit, right_fit, M, left_curvature, right_curvature,
                                center, True)
            cv2.imshow('final', img)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    else:
        break
