import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF

# Insert your package here

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T. - Done
    (2) Setup the eight point algorithm's equation. - Done
    (3) Solve for the least square solution using SVD. - Done
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. #Refine already has it (Done)
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. - Done
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    N = pts1.shape[0]
    pts1s = pts1/M #scaled1 
    pts2s = pts2/M #scaled2

    points = []
    for i in range(N):
        pt = [ (pts1s[i,0]*pts2s[i,0]) , (pts1s[i,0]*pts2s[i,1]) , pts1s[i,0] ,
                            (pts1s[i,1]*pts2s[i,0]) , (pts1s[i,1]*pts2s[i,1]) , pts1s[i,1] ,
                            pts2s[i,0] , pts2s[i,1] , 1]
        points.append(pt)
    
    A = np.asarray(points)
    u,s,vt = np.linalg.svd(A)
    F = vt[-1,:].reshape((3,3)).T 

    F = refineF(F,pts1s,pts2s)
    F = F/F[2,2]
    T = np.diag([1/M,1/M,1])
    us_F = T.T @ F @ T
    #print(us_F)
    return us_F


if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    # Q2.1
    M=np.max([*im1.shape, *im2.shape])
    np.savez("submission/q2_1.npz",F=F,M=M)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)
    #calc_epi_error(pts1_homogenous,pts2_homogenous,F)
    print("M = ",M)
    print("F =")
    print(F)
    displayEpipolarF(im1,im2,F) #markersize changed

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)