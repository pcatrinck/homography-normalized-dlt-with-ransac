import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import my_ransac
import my_dlt

MIN_MATCH_COUNT = 10
img1 = cv.imread('imgs\edu_vingadores.png',0)          # queryImage
img2 = cv.imread('imgs\edu.png',0)                     # trainImage
img6 = cv.imread('imgs\cat_placeholder.jpeg')          # subplot placeholder

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN stands for Fast Library for Approximate Nearest Neighbors.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# The variable search_params specifies the number of times the trees in the index should
# be recursively traversed. Higher values gives better precision, but also takes more time.
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])

    # my homography
    best_pts1_in, best_pts2_in, iteractions = my_ransac.ransac(src_pts, dst_pts,4,6)
    H = my_dlt.my_homography(best_pts1_in, best_pts2_in)
    img4 = cv.warpPerspective(img1, H, (img1.shape[1],img1.shape[0])) #, None) #, flags[, borderMode[, borderValue]]]]	)
    
    # cv homography to comparasion
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,6.0)
    img5 = cv.warpPerspective(img1, M, (img1.shape[1],img1.shape[0])) #, None) #, flags[, borderMode[, borderValue]]]]	)

matchesMask = None
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

# Display the images side by side for comparison
fig, axs = plt.subplots(2, 3, figsize=(20, 13))

# Original images
axs[0, 0].imshow(img1, cmap='gray')
axs[0, 0].set_title('Original Image 1')

axs[0, 1].imshow(img2, cmap='gray')
axs[0, 1].set_title('Original Image 2')

# points correlated
axs[0, 2].imshow(img3, cmap='gray')
axs[0, 2].set_title('Points correlated')

# Images transformed using your homography (H)
axs[1, 0].imshow(img4, cmap='gray')
axs[1, 0].set_title(f'Image 1 Transformed (Custom Homography)\n{iteractions} iterations')

# Images transformed using OpenCV's homography (M)
axs[1, 1].imshow(img5, cmap='gray')
axs[1, 1].set_title('Image 1 Transformed (OpenCV Homography)')

# empty space
axs[1, 2].imshow(img6, cmap='gray')
axs[1, 2].set_title('this place was empty\nplease take this cat placeholder')

# Adjust spacing between subplots
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)

# Set axis labels
for ax in axs.flat:
    ax.set(xlabel='X-axis', ylabel='Y-axis')

# Remove the x and y ticks for the plots
for ax in axs.flat:
    ax.label_outer()

plt.show()