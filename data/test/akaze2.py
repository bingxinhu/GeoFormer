import numpy as np
import cv2

# 读取图像
img1 = cv2.imread('frame_000402.png')  # 事件图像
img2 = cv2.imread('000403.png')        # RGB图像

# 相机标定参数
rgb_camera_matrix = np.array([[2564.19310444823, 0, 739.387821416966],
                              [0, 2563.41510197018, 537.540801537576],
                              [0, 0, 1]])
rgb_dist_coeffs = np.array([-0.161037226394270, 0.128949733352634, 0, 0, 0])

dvs_camera_matrix = np.array([[1671.20555532487, 0, 644.067031753728],
                              [0, 1670.49054323916, 373.313702644126],
                              [0, 0, 1]])
dvs_dist_coeffs = np.array([-0.0707315665225943, 0.290205090753698, 0, 0, 0])

# 图像去畸变
undistorted_img1 = cv2.undistort(img1, dvs_camera_matrix, dvs_dist_coeffs)
undistorted_img2 = cv2.undistort(img2, rgb_camera_matrix, rgb_dist_coeffs)

# 将图像转换为灰度图
gray1 = cv2.cvtColor(undistorted_img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(undistorted_img2, cv2.COLOR_BGR2GRAY)

# init AKAZE detector
akaze = cv2.AKAZE_create()

# Find the keypoints and descriptors with AKAZE
kp1, des1 = akaze.detectAndCompute(gray1, None)
kp2, des2 = akaze.detectAndCompute(gray2, None)

print(f"Keypoints in img1: {len(kp1)}, Keypoints in img2: {len(kp2)}")

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test and collect good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        good_matches.append(m)

print(f"Number of good matches: {len(good_matches)}")

# Sort the good matches by their distance (quality of match)
good_matches = sorted(good_matches, key=lambda x: x.distance)

# Select top N matches (e.g., top 50 or less if fewer matches)
N = min(100, len(good_matches))
best_matches = good_matches[:N]

# Calculate slopes and filter matches
slopes = []
for i, match in enumerate(best_matches):
    x1, y1 = kp1[match.queryIdx].pt
    x2, y2 = kp2[match.trainIdx].pt
    
    # Calculate slope (y2 - y1) / (x2 - x1)
    if x2 - x1 == 0:  # Handle vertical line case
        slope = 10
    else:
        slope = (y2 - y1) / (x2 - x1)
    # Calculate the distance between (x1, y1) and (x2, y2)
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if  500 < distance < 700 :
        print("No.", i, x1, y1, x2, y2, distance, slope)
        slopes.append((slope, match))  # Store slope and corresponding match

# Sort matches based on slope
slopes_sorted = sorted(slopes, key=lambda x: x[0])

# Extract the sorted best matches
sorted_best_matches = [match for _, match in slopes_sorted]

# Optionally, select top N matches from the sorted list
N = min(10, len(sorted_best_matches))
best_matches = sorted_best_matches[:N]

print(best_matches)
# Select good matched keypoints from top matches
ref_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
sensed_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Draw matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('sorted_matches.jpg', img3)

# 读取稀疏图像
sparse_img = cv2.imread('events_000402.png', cv2.IMREAD_UNCHANGED)

# 确保H矩阵已计算
if len(ref_matched_kpts) >= 4 and len(sensed_matched_kpts) >= 4:
    # Compute homography using RANSAC
    H, status = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)
    print(H)
    # Warp sparse image using the computed homography
    warped_sparse_image = cv2.warpPerspective(sparse_img, H, (img2.shape[1], img2.shape[0]))

    # 显示结果
    cv2.imshow('Warped Sparse Image', warped_sparse_image)
    cv2.waitKey(9000)
    cv2.destroyAllWindows()
else:
    print("Not enough matches to compute homography.")
