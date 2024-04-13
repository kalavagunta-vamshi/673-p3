#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the size of each square in millimeters
square_size = 21.5

# Define the number of inner corners on the chessboard
inner_corners = (9, 6)

# Define the termination criteria for the iterative algorithm
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create an array to hold the object points and the image points
object_points = []
image_points = []

# Define the object points for the chessboard corners in 3D space
object_points_array = np.zeros((inner_corners[0] * inner_corners[1], 3), np.float32)
object_points_array[:, :2] = np.mgrid[0:inner_corners[0], 0:inner_corners[1]].T.reshape(-1, 2) * square_size

# Loop over all calibration images
folder_path = "calibration_images"
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        image = cv.imread(image_path)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, inner_corners, None)

        # If corners are found, add the object points and image points to the respective arrays
        if ret:
            object_points.append(object_points_array)
            corners_acc = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners_acc)

            # Draw the corners on the image
            cv.drawChessboardCorners(image, inner_corners, corners_acc, ret)
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            plt.show()

# Calculate the camera matrix, distortion coefficients, rotation and translation vectors
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# Compute the reprojection error for each image
mean_error = 0
for i in range(len(object_points)):
    image_points_reprojected, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(image_points[i], image_points_reprojected, cv.NORM_L2) / len(image_points_reprojected)
    print("Reprojection error for image {}: {}".format(i+1, error))
    mean_error += error

# Calculate the mean reprojection error
print("Mean reprojection error: {}".format(mean_error / len(object_points)))

# Display the camera matrix
print("Camera matrix (K):\n", mtx)


# In[ ]:





# In[ ]:





# In[ ]:




