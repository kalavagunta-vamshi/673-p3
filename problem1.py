#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.linalg import rq

"""Function to calculate projection matrix P"""

def calculate_projection_matrix(image_points, world_points):
    # Append 1 to each image point and world point to form homogeneous coordinates
    img_c = np.hstack((image_points, np.ones((8, 1))))
    world_c = np.hstack((world_points, np.ones((8, 1))))
    # Create an empty matrix A
    A = np.zeros((16, 12))
    # Loop over all image points and construct matrix A
    for i in range(8):
        x, y, w = img_c[i]
        x2, y2, z2, w2 = world_c[i]
        A[2 * i] = [-x2, -y2, -z2, -w2, 0, 0, 0, 0, x * x2, x * y2, x * z2, x * w2]
        A[2 * i + 1] = [0, 0, 0, 0, -x2, -y2, -z2, -w2, y * x2, y * y2, y * z2, y * w2]
    # Compute SVD of matrix A
    U, S, Vt = np.linalg.svd(A)
    # The last row of Vt is the nullspace of A and corresponds to the projection matrix P
    P = -Vt[-1].reshape((3, 4))
    # Normalize P such that the last element of the last row is 1
    P = P / P[2, 3]
    return P

"""Function to decompose projection matrix P into intrinsic matrix K, rotation matrix R, and translation vector t"""
def decompose_projection_matrix(P):
    # Perform RQ decomposition on the first three columns of P
    K, R = rq(P[:, :3], mode='full')
    # Normalize K such that its bottom-right element is 1
    K = K / K[2, 2]
    # Compute the translation vector t as the inverse of K times the last column of P
    t = np.dot(np.linalg.inv(K), P[:, 3])
    return K, R, t

"""Function to compute reprojection error given image points, world points, and projection matrix P"""
def compute_reprojection_error(image_points, world_points, P):
    reprojection_error = np.zeros(8)
    a = 0
    for i in range(8):
        world_point = np.hstack((world_points[i], 1))
        projected_point = np.dot(P, world_point)
        projected_point /= projected_point[2]
        reprojection_error[i] = np.linalg.norm(projected_point[:2] - image_points[i])
        a = a + reprojection_error[i]
    # Compute mean reprojection error
    mean_reprojection_error = a / 8
    return reprojection_error, mean_reprojection_error

"""Loading the image points and world points as (x,y) and (x,y,z) respectively"""
image_points = np.array([[757, 213], [758, 415], [758, 686], [759, 966], [1190, 172], [329, 1041], [1204, 850], [340, 159]])
world_points = np.array([[0, 0, 0], [0, 3, 0], [0, 7, 0], [0, 11, 0], [7, 1, 0], [0, 11, 7], [7, 9, 0], [0, 1, 7]])

P = calculate_projection_matrix(image_points, world_points)
K, R, t = decompose_projection_matrix(P)
reprojection_error, mean_reprojection_error = compute_reprojection_error(image_points, world_points, P)

print("Projection matrix P:\n", P)
print("\n-----------------------------------------------------------------------\n")
print("\nIntrinsic matrix K:\n", K)
print("\n-----------------------------------------------------------------------\n")
print("Rotation matrix R:\n", R, )
print("\n-----------------------------------------------------------------------\n")
print("Translation vector t:\n", t)
print("\n-----------------------------------------------------------------------\n")
for i in range(8):
    print(f"The reprojection error for point ({image_points[i, 0]}, {image_points[i, 1]}) is {reprojection_error[i]:.4f}")
print("\n-----------------------------------------------------------------------\n")
print("Mean reprojection error is:", mean_reprojection_error)
print("\n-----------------------------------------------------------------------\n")


# In[ ]:





# In[ ]:





# In[ ]:




