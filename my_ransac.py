import numpy as np
import my_dlt
import math

class Ransac():
    
    def __init__(self,sample_size,max,threshold):
        self.sample_size = sample_size      # Number of data points randomly sampled in each iteration of RANSAC
        self.max = max                      # Maximum number of iterations
        self.threshold = threshold          # Limit to consider a point as an inliner
        
    def ransac(self, pts1, pts2):
        num_points = len(pts1)
        max_inliers = 0
        best_pts1_in = None
        best_pts2_in = None

        # Turn on N, to loop over
        i = np.random.choice(num_points, size=self.sample_size, replace=False)
        sample_pts1 = pts1[i]
        sample_pts2 = pts2[i]
        H = my_dlt.my_homography(sample_pts1, sample_pts2)
        inliers = self.find_inliers(pts1, pts2, H)

        #e = len(pts1) - len (inliers)
        e = 0.001
        p = 0.999
        s = self.sample_size

        print(f"e={e},s={s}")
        N = math.log(1 - p) / math.log(1 - (1 - e)**s)
        print(f"N={N}")

        # warm up for loop
        inliers = [0]    # len == 1, len(inliers)/len(pts1) < 0.88
        iterations = 0

        #while iterations < self.max:
        #while (iterations < self.max) and (len(inliers)/len(pts1) < 0.88):
        while (iterations < N) and (len(inliers)/len(pts1) < 0.88):
            i = np.random.choice(num_points, size=self.sample_size, replace=False)
            sample_pts1 = pts1[i]
            sample_pts2 = pts2[i]

            H = my_dlt.my_homography(sample_pts1, sample_pts2)

            inliers = self.find_inliers(pts1, pts2, H)

            if np.sum(inliers) > max_inliers:
                max_inliers = np.sum(inliers)
                best_pts1_in = pts1[inliers]
                best_pts2_in = pts2[inliers]

            iterations+=1

        H_final = my_dlt.my_homography(best_pts1_in, best_pts2_in)

        return H_final, best_pts1_in, best_pts2_in

    def find_inliers(self,pts1, pts2, H):

        pts1_homogeneous = np.column_stack((pts1, np.ones(len(pts1))))
        pts2_transformed = np.dot(H, pts1_homogeneous.T).T
        pts2_transformed /= pts2_transformed[:, 2][:, np.newaxis]
        pts2_transformed = pts2_transformed[:, :2]

        distances = np.linalg.norm(pts2 - pts2_transformed, axis=1)

        inliers = distances < self.threshold
        return inliers