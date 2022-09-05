import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import plot_3d_keypoint

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2]) -Done
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. -Done
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices) - Done
    (4) As a reference, our solution's best error is around ~2000 on the 3D points. 
'''
def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):
    N = temple_pts1.shape[0]
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    temple_pts2 = np.zeros((N,2))

    for i in range(N):
        # corresponding points found
        temple_pts2[i,0], temple_pts2[i,1] = epipolarCorrespondence(im1,im2,F,temple_pts1[i,0],temple_pts1[i,1]) 

    M2,C2,P = findM2(F,temple_pts1,temple_pts2,intrinsics)
    #print(M2)
    return M2,C2,P

if __name__ == "__main__":

    temple_coords_path = np.load('data/templeCoords.npz')
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    #print(F)

    pts1_x = temple_coords_path["x1"]
    pts1_y = temple_coords_path["y1"]
    pts1 = np.hstack([pts1_x,pts1_y])
    M2,C2,P = compute3D_pts(pts1,intrinsics,F,im1,im2)
    # print(P.shape) - getting 288,3
    M1 = np.hstack((np.eye(3),np.zeros((3,1))))
    #print(M1)
    C1 = K1@M1

    np.savez("submission/q4_2.npz",F=F,M1=M1,C1=C1,M2=M2,C2=C2)
    
    # 3D plot - gfg
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ticks_x = np.arange(-0.7, 0.7, 0.2)
    ax.set_xticks(ticks_x)
    ticks_y = np.arange(-0.45, 0.45, 0.2)
    ax.set_yticks(ticks_y)
    ticks_z = np.arange(3.2, 4.5, 0.2)
    ax.set_zticks(ticks_z)
    ax.set_xlim3d(-0.7,0.7)
    ax.set_ylim3d(-0.45,0.45)
    ax.set_zlim3d(3.2,4.5)

    ax.scatter3D(P[:,0], P[:,1], P[:,2], color = "blue")
    plt.title("q4_2_visualize")
    ax.view_init(90, 0) # yshows problem
    plt.show()
