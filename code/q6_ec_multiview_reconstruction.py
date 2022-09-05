from ast import Mult
import numpy as np
import matplotlib.pyplot as plt

import os

from pyparsing import col
from q2_1_eightpoint import eightpoint

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate

# Insert your package here

'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres = 100):
    N = pts1.shape[0]
    P = np.zeros((N,3))
    e = np.zeros(N)

    for i in range(N):
        
        if(pts1[i,2]<=pts2[i,2] and pts1[i,2]<=pts3[i,2]): #least confidence in camera 1 
            Ptemp, etemp = triangulate(C2, np.expand_dims(pts2[i,:2],axis=0), C3, np.expand_dims(pts3[i,:2],axis=0))

        elif(pts2[i,2]<pts3[i,2] and pts2[i,2]<pts1[i,2]):
            Ptemp, etemp = triangulate(C3, np.expand_dims(pts3[i,:2],axis=0), C1, np.expand_dims(pts1[i,:2],axis=0))

        else:
            Ptemp, etemp = triangulate(C1, np.expand_dims(pts1[i,:2],axis=0), C2, np.expand_dims(pts2[i,:2],axis=0))

        #### print(Ptemp.shape)
        P[i,:] = Ptemp
        e[i] = etemp
    
    #print(P.shape)
    return P,e

'''
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
'''
def plot_3d_keypoint_video(pts_3d_video):
    fig = plt.figure()
    num_points = pts_3d_video.shape[1]
    ax = plt.axes(projection='3d')

    for i in range(pts_3d_video.shape[0]):
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d_video[i,index0,0], pts_3d_video[i,index1,0]]
            yline = [pts_3d_video[i,index0,1], pts_3d_video[i,index1,1]]
            zline = [pts_3d_video[i,index0,2], pts_3d_video[i,index1,2]]
            ax.plot(xline,yline,zline,color=colors[j])

        np.set_printoptions(threshold=1e6, suppress=True)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    plt.show()


#Extra Credit
if __name__ == "__main__":
         
    pts_3d_video = []
    frame = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join('data/q6/','time'+str(loop)+'.npz')
        image1_path = os.path.join('data/q6/','cam1_time'+str(loop)+'.jpg')
        image2_path = os.path.join('data/q6/','cam2_time'+str(loop)+'.jpg')
        image3_path = os.path.join('data/q6/','cam3_time'+str(loop)+'.jpg')

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data['pts1']
        pts2 = data['pts2']
        pts3 = data['pts3']

        K1 = data['K1']
        K2 = data['K2']
        K3 = data['K3']

        M1 = data['M1']
        M2 = data['M2']
        M3 = data['M3']

        #Note - Press 'Escape' key to exit img preview and loop further 
        #img = visualize_keypoints(im2, pts2)
        C1 = K1@M1
        C2 = K2@M2
        C3 = K3@M3

        #print(pts2.sum(axis=0))
        #print(pts2)
        #F12 = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
        #F23 = eightpoint(pts2, pts3, M=np.max([*im2.shape, *im3.shape]))
        #F31 = eightpoint(pts3, pts1, M=np.max([*im3.shape, *im1.shape]))

        P, err = MultiviewReconstruction(C1,pts1,C2,pts2,C3,pts3)
        frame.append(P)

    #print(frame[0][0])
    #plot_3d_keypoint(frame[0])
    np.savez("submission/q6_1.npz",P = frame[0])
    
    frames = np.asarray(frame)
    #print(frames.shape)
    plot_3d_keypoint_video(frames)