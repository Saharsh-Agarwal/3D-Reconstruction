from unittest.util import _MIN_DIFF_LEN
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    pts1 = [] #modified
    pts2 = [] #modified
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2) #mouse_stop = 2, middle button

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        pts1.append([xc,yc]) #modified
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        pts2.append([x2,y2])
        ax2.plot(x2, y2, 'ro', markersize=8, linewidth=2)
        plt.draw()
    
    #modified 
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    np.savez("submission/q4_1.npz",F=F,pts1=pts1,pts2=pts2)


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2 - Done
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    x = np.asarray([int(x1),int(y1),1])
    line = F@x.T #search along this line
    window = 50 # varied #for 4.2 changed from 10
    c = window//2 #centre se dist 

    patch = im1[y1-c:y1+c+1,x1-c:x1+c+1] #patch in image1 
    #print(patch.shape)
    # gaussian - gfg
    ss = 2
    xx,yy = np.meshgrid(np.linspace(-ss,ss,window+1), np.linspace(-ss,ss,window+1)) #(-1,1) can be changed
    dist = (xx**2 + yy**2)**0.5         #print(dist.shape)
    sigma = 3
    gauss = np.exp(-((dist)**2 / (2.0*(sigma**2))))
    gauss = gauss/np.sum(gauss)
    #gauss = np.eye(gauss.shape[0])
    #print(gauss.shape)

    if len(patch.shape)>2 : #image has channels
        for i in range(patch.shape[-1]):
            img_result = gauss*patch[:,:,i]
    else:
        img_result = gauss*patch

    min_dist = 10000
    ## print(im2.shape[0]-window)
    for i in range(im2.shape[0]-window): #image goes columnwise
        y2 = i+c
        x2 = int(-(line[1]*y2+line[2])/line[0])
        patch2 = im2[y2-c:y2+c+1,x2-c:x2+c+1]
        ## print(patch2.shape, y2-c,y2+c+1,y2,c,i,i+c)
        if len(patch2.shape)>2 : #image2 has channels
            for j in range(patch2.shape[-1]):
                img2_result = gauss*patch2[:,:,j]
        else:
            img2_result = gauss*patch2

        distpoint = np.linalg.norm(np.asarray([x1-x2,y1-y2]))
        diff = np.linalg.norm(img2_result-img_result)
        if min_dist>diff and distpoint<50:
            best_x2 = x2
            best_y2 = y2
            min_dist = diff

    return best_x2,best_y2


if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')
    
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    
    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    print(x2)

    epipolarMatchGUI(im1,im2,F)
    print("F:")
    print(F)
    assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)