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
    if m.distance < 0.79 * n.distance:
        good_matches.append(m)

print(f"Number of good matches: {len(good_matches)}")

# Sort the good matches by their distance (quality of match)
good_matches = sorted(good_matches, key=lambda x: x.distance)

# Select top N matches (e.g., top 50 or less if fewer matches)
N = min(50, len(good_matches))
best_matches = good_matches[:N]

# Select good matched keypoints from top matches
ref_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
sensed_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Calculate slopes and filter matches
slopes = []
for i, match in enumerate(best_matches):
    x1, y1 = kp1[match.queryIdx].pt
    x2, y2 = kp2[match.trainIdx].pt
    
    # Calculate slope (y2 - y1) / (x2 - x1)
    if x2 - x1 == 0:  # Handle vertical line case
        slope = 0
    else:
        slope = abs(y2 - y1) / abs(x2 - x1)
    
    slopes.append((slope, match))  # Store slope and corresponding match

# Sort matches based on slope
slopes_sorted = sorted(slopes, key=lambda x: x[0])

# Extract the sorted best matches
sorted_best_matches = [match for _, match in slopes_sorted]

# Optionally, select top N matches from the sorted list
N = min(4, len(sorted_best_matches))
best_matches = sorted_best_matches[:N]

# Draw matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('sorted_matches.jpg', img3)

# Ensure at least 4 points for homography computation
if len(ref_matched_kpts) >= 4 and len(sensed_matched_kpts) >= 4:
    # Compute homography using RANSAC
    H, status = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)
    print(H)
    # Warp image using the computed homography
    warped_image = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    
    cv2.imwrite('warped.jpg', warped_image)
else:
    print("Not enough matches to compute homography.")
#----------------------------------------------------------
# Slope: 0.078, Distance: 55.01
# Points: (398.47, 160.26) to (453.31, 164.53)

# Slope: 0.174, Distance: 117.45
# Points: (1027.57, 92.65) to (911.88, 72.48)

# Slope: 0.332, Distance: 233.10
# Points: (766.75, 324.68) to (987.95, 398.19)

# Slope: 0.442, Distance: 256.94
# Points: (783.01, 393.74) to (1018.00, 497.65)

# Slope: 4.134, Distance: 189.59
# Points: (635.33, 206.22) to (679.90, 390.49)

# Slope: 6.084, Distance: 382.25
# Points: (713.88, 59.14) to (651.89, 436.33)