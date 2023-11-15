import numpy as np
import my_dlt
import math

def ransac(pts1, pts2, sample_size, threshold):
    
    # warm up for loop
    inliers = [0]    # len == 1, len(inliers)/len(pts1) < 0.88
    iteractions = 0
    num_points = len(pts1)
    best_inliers_num = 0
    best_pts1_in = None
    best_pts2_in = None

    # Turn on N, to loop over
    i = np.random.choice(num_points, sample_size, replace=False)
    sample_pts1 = pts1[i]
    sample_pts2 = pts2[i]
    H = my_dlt.my_homography(sample_pts1, sample_pts2)
    inliers = find_inliers(pts1, pts2, H, threshold)

    e = 0.001   #e = len(pts1) - len (inliers)
    p = 0.999
    s = sample_size

    N = math.log(1 - p) / math.log(1 - (1 - e)**s)

    while (iteractions < N) and ((len(inliers) / len(pts1) > 0.88)):
        # select random samples
        sample = np.random.choice(num_points, size=sample_size, replace=False)
        sample_pts1 = pts1[sample]
        sample_pts2 = pts2[sample]

        # fit model to the samples
        H = my_dlt.my_homography(sample_pts1, sample_pts2)

        # find inliers
        inliers = find_inliers(pts1, pts2, H, threshold)

        # update the best model if the current one is better
        if np.sum(inliers) > best_inliers_num:
            best_inliers_num = np.sum(inliers)
            best_pts1_in = pts1[inliers]
            best_pts2_in = pts2[inliers]
            print(f"inliers={inliers},best_inliers_num{best_inliers_num},best_pts1_in={best_pts1_in}")

        # e = len(best_pts1_in) - len (inliers)
        # print(f"e={e},s={s}")
        # N = math.log(1 - p) / math.log(1 - (1 - e)**s)
        # print(f"N={N}") 
        iteractions+=1

    return best_pts1_in, best_pts2_in, iteractions

def find_inliers(pts1, pts2, H, threshold):

    pts1_homogeneous = np.column_stack((pts1, np.ones(len(pts1))))
    pts2_transformed = np.dot(H, pts1_homogeneous.T).T
    pts2_transformed /= pts2_transformed[:, 2][:, np.newaxis]
    pts2_transformed = pts2_transformed[:, :2]

    distances = np.linalg.norm(pts2 - pts2_transformed, axis=1)

    inliers = distances < threshold
    
    return inliers