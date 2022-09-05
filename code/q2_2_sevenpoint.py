import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize

# Insert your package here
import numpy.polynomial.polynomial as npp

'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M. - Done
    (2) Setup the seven point algorithm's equation. - Done
    (3) Solve for the least square solution using SVD. - Done
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2) - Done
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots - Done
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):

    Farray = []
    
    # ----- TODO -----
    # YOUR CODE HERE
    N = 7 
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
    F1 = vt[-1,:].reshape((3,3)).T
    F2 = vt[-2,:].reshape((3,3)).T

    #det(F(a)) = det(aF1 + (1-a)F2) = 0 (irrespective of a) = c0 + c1.a + c2.a^2 +c3.a^3

    c0 = np.linalg.det(F2) #putting a=0
    c2 = (np.linalg.det(F1)+np.linalg.det((2*F2)-F1)-(2*c0))/2 #a = 1 and a = -1 : both are added
    c1_c3 = (np.linalg.det(F1)-np.linalg.det((2*F2)-F1))*0.5 # when the above is subtracted
    c3 = (1/12)*(np.linalg.det(2*F1-F2)-np.linalg.det(3*F2-2*F1)-2*(np.linalg.det(F1) - np.linalg.det(2*F2-F1)))
    c1 = c1_c3-c3

    roots = npp.polyroots([c0,c1,c2,c3]) # computes roots of the polynomials with coefficients c
    T = np.diag([1/M,1/M,1])

    for i in roots:
        if i.imag == 0:
            a = i.real
            F = a*F1 + (1-a)*F2
            F = T.T @ F @ T # unscaling - as not refined, thus call singularise by hand 
            F = _singularize(F)
            F = F/F[2,2] # making the last value 1
            Farray.append(F)
    Farray = np.asarray(Farray)
    #print(Farray.shape)
    return Farray

if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')
    
    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])
    np.savez("submission/q2_2.npz",Fs=Fs,F=F,Err=ress[min_idx],M=M)
    print("F = ")
    print(F)
    displayEpipolarF(im1,im2,F)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)