import numpy as np
import open3d as o3d
import cv2

# 读取图像
img1 = cv2.imread('000402.png')  
img2 = cv2.imread('000403.png')

# 相机标定参数
rgb_camera_matrix = np.array([[2564.19310444823, 0, 739.387821416966],
                               [0, 2563.41510197018, 537.540801537576],
                               [0, 0, 1]])
rgb_dist_coeffs = np.array([-0.161037226394270, 0.128949733352634, 0, 0, 0])

dvs_camera_matrix = np.array([[1671.20555532487, 0, 644.067031753728],
                              [0, 1670.49054323916, 373.313702644126],
                              [0, 0, 1]])
dvs_dist_coeffs = np.array([-0.0707315665225943, 0.290205090753698, 0, 0, 0])
# 去畸变
undistorted_img1 = cv2.undistort(img1, rgb_camera_matrix, rgb_dist_coeffs)
undistorted_img2 = cv2.undistort(img2, rgb_camera_matrix, rgb_dist_coeffs)

# 转换为灰度图
gray1 = cv2.cvtColor(undistorted_img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(undistorted_img2, cv2.COLOR_BGR2GRAY)

# Resize the RGB image to match the DVS event image resolution
gray1_resized = cv2.resize(gray1, (gray2.shape[1], gray2.shape[0]))

# 特征点检测和描述
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1_resized, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# 特征匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# 选取匹配点
points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])

# 计算单应性矩阵
if len(matches) >= 4:
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    print(H)
else:
    print("Not enough matches to compute homography.")

# 计算差异并生成点云
difference = cv2.absdiff(gray1_resized, gray2)
_, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

# 获取变化区域的坐标
y_indices, x_indices = np.where(thresh > 0)

# 生成 2D 点云 (x, y, intensity)
intensity = gray2[y_indices, x_indices]
points = np.array([[x, y] for x, y in zip(x_indices, y_indices)], dtype=np.float32)

# 创建 Open3D 2D 点云对象
point_cloud_gray = o3d.geometry.PointCloud()
point_cloud_gray.points = o3d.utility.Vector3dVector(np.hstack((points, np.zeros((points.shape[0], 1)))))
point_cloud_gray.colors = o3d.utility.Vector3dVector(np.tile(intensity[:, None] / 255.0, (1, 3)))

# DVS事件数据处理
dvs_events = np.load("000402.npy")
points_dvs = []
colors_dvs = []
for event in dvs_events:
    x, y, t, polarity = event
    x_rounded = int(np.round(1.125 * x))
    y_rounded = int(np.round(1.5 * y))
    points_dvs.append([x_rounded, y_rounded, 0])
    color = [0.0, 0.0, 1.0] if polarity else [0.0, 0.0, 0.0]
    colors_dvs.append(color)

# 创建DVS点云
point_cloud_dvs = o3d.geometry.PointCloud()
point_cloud_dvs.points = o3d.utility.Vector3dVector(points_dvs)
point_cloud_dvs.colors = o3d.utility.Vector3dVector(colors_dvs)

# 处理 RGB 图像以生成点云
rgb_image_np = np.asarray(undistorted_img2)
height, width, _ = rgb_image_np.shape
points_rgb = []
colors_rgb = []

# 创建 RGB 点云
for y in range(height):
    for x in range(width):
        points_rgb.append([x, y, 0])  # Z设置为0
        colors_rgb.append(rgb_image_np[y, x] / 255.0)  # RGB颜色值归一化

# 创建RGB点云对象
point_cloud_rgb = o3d.geometry.PointCloud()
point_cloud_rgb.points = o3d.utility.Vector3dVector(points_rgb)
point_cloud_rgb.colors = o3d.utility.Vector3dVector(colors_rgb)

# 使用单应性矩阵进行点云配准
if H is not None:
    H_4x4 = np.eye(4)  # 创建一个4x4单位矩阵
    H_4x4[:3, :3] = H  # 将3x3的单应性矩阵赋值给4x4矩阵的左上角
    point_cloud_gray.transform(H_4x4)  # 应用变换

# 显示结果
o3d.visualization.draw_geometries([point_cloud_gray, point_cloud_dvs],
                                   window_name='Registered Point Cloud Visualization',
                                   width=1280, height=720)

