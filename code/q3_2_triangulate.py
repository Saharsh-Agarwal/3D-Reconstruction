import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2 - Done
    (2) Solve for the least square solution using np.linalg.svd - Done
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones) - Done
    (4) Keep track of the 3D points and projection error, and continue to next point - Done
    (5) You do not need to follow the exact procedure above. 
'''
def triangulate(C1, pts1, C2, pts2):
    w , err = 0,0
    N = pts1.shape[0]
    A = np.zeros((4,4))
    e = 0 #error
    P = np.zeros((N,3))

    for i in range(N):
        A[0,:] = (pts1[i,1]*C1[2,:])-C1[1,:]
        A[1,:] = -(pts1[i,0]*C1[2,:])+C1[0,:]
        A[2,:] = (pts2[i,1]*C2[2,:])-C2[1,:]
        A[3,:] = -(pts2[i,0]*C2[2,:])+C2[0,:]
        u,s,vh = np.linalg.svd(A)
        X = vh[-1,:]
        X = X/X[3]
        x = C1@X.T
        x = x/x[2]
        x_ = C2@X.T
        x_ = x_/x_[2]
        e = e + (np.linalg.norm(pts1[i]-x[0:2])**2 + np.linalg.norm(pts2[i]-x_[0:2])**2)
        P[i,:] = X[0:3]

    return P, e

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def findM2(F, pts1, pts2, intrinsics, filename = 'q3_3.npz'):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)
    
    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    '''
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F,K1,K2)
    M1 = np.hstack((np.eye(3),np.zeros((3,1))))
    #print(M1)
    C1 = K1@M1
    M2s = camera2(E)
    e = 10000
    M2_best = None
    C2_best = None
    P_best = None
    
    for i in range(M2s.shape[-1]): #essentially 4 times 
        M2 = M2s[:,:,i]
        C2 = K2@M2
        P_trial, err = triangulate(C1, pts1, C2, pts2)
        # print("P_trial", i,P_trial[:,-1])
        ### print(np.all(P_trial[:,-1]>0))
        if (np.all(P_trial[:,-1]>0) and err<e):
            e = err
            M2_best = M2
            C2_best = C2
            P_best = P_trial

    print("Best Error : ",e)
    ##print(M2_best, P_best[:,-1])
    return M2_best, C2_best, P_best

if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)
    np.savez("submission/q3_3.npz",M2=M2,C2=C2,P=P)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert(err < 500)