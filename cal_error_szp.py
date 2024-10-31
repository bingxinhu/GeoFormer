import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def p2p_homo(input_points, H, num=4):
    assert input_points.shape[1] == 2
    ev_space = np.zeros((3, num))
    ev_space[0, :] = input_points[:, 0]  # col
    ev_space[1, :] = input_points[:, 1]  # row
    ev_space[2, :] = np.ones((1, num))
    img_space = np.matmul(H, ev_space)
    img_space = img_space / img_space[2, :]  # 归一化！！！很重要！！！
    output_points = np.zeros_like(input_points, dtype=np.float32)
    output_points[:, 0] = img_space[0, :]
    output_points[:, 1] = img_space[1, :]
    output_points = np.round(output_points, 0).astype(np.int32)  # 四舍五入，然后转换成int类型
    return output_points


def visual():  # 可视化离群值
    data = np.load("/media/HDD1/dual_modality_datasets/24_01_28/valid_seq/24_01_28_output_03/points_list.npy", allow_pickle=True).item()
    pt1_list = []
    for key in data.keys():
        pt1_list.append(data[key][0])
    points = np.array(pt1_list)

    # 使用 DBSCAN 进行离群值检测
    db = DBSCAN(eps=2, min_samples=2).fit(points)
    outliers = points[db.labels_ == -1]  # 标签为 -1 的点是离群点
    inliers = points[db.labels_ != -1]  # 标签为 -1 的点是离群点
    print(len(points), len(outliers))
    # print("离群点：", outliers)

    plt.scatter(inliers[:, 0], inliers[:, 1], s=8, color='blue', label="Inliers")
    plt.scatter(outliers[:, 0], outliers[:, 1], s=8, color='red', label="Outliers")
    plt.scatter([-237], [-101], s=20, color='green', label="thre_0.1")
    plt.scatter([-130], [-68], s=20, color='cyan', label="thre_0.5")

    # 添加标题和标签
    plt.title("Point Distribution with Outliers")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.xlim(-300, 200)  # x 轴范围
    plt.ylim(-300, 200)  # y 轴范围
    plt.legend()
    plt.savefig("distribution.png")
    plt.show()

    # # 检测findHomography是否可以剔除离群值
    # import cv2
    # import numpy as np

    # # 示例匹配点
    # src_pts = np.array([[0, 0], [5, 2], [4, 7], [13, 4]])
    # dst_pts = np.array([[0, 0], [50, 20], [40, 70], [130, 40]])
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC) 
    # print(H)

    # src_pts = np.array([[0, 0], [5, 2], [4, 7], [13, 4], [15, 7]])
    # dst_pts = np.array([[0, 0], [50, 20], [40, 70], [130, 40], [190, 300]])
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)  # 四个正确点，一个错误点，计算错误
    # print(H)

    # src_pts = np.array([[0, 0], [5, 2], [4, 7], [13, 4], [21, 13], [15, 7]])
    # dst_pts = np.array([[0, 0], [50, 20], [40, 70], [130, 40], [210, 13], [190, 300]])
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)  # 五个正确点，一个错误点，计算正确
    # print(H)

    # src_pts = np.array([[0, 0], [5, 2], [4, 7], [13, 4], [21, 13], [15, 7], [20, 20]])
    # dst_pts = np.array([[0, 0], [50, 20], [40, 70], [130, 40], [210, 13], [190, 300], [170, 250]])
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)  # 五个正确点，两个错误点，计算错误
    # print(H)


def check_point_distribution():
    root_path = "/media/HDD1/dual_modality_datasets/24_01_28/output_03"
    file_list = sorted(glob.glob(os.path.join(root_path, "homo_matrix/homography_inv_*.npy")))
    img_list = sorted(glob.glob(os.path.join(root_path, "mapped_images/*.jpg")))
    assert len(file_list) == len(img_list)
    points_dict = {}
    for img_name, file in zip(img_list, file_list):
        rev_H = np.load(file)
        input_points = np.array([[0, 0], [1439, 0], [1439, 1079], [0, 1079]])
        output_points = p2p_homo(input_points, rev_H)
        points_dict[img_name.split('/')[-1]] = output_points
    np.save("/media/HDD1/dual_modality_datasets/24_01_28/output_03/points_list.npy", np.array(points_dict))

    rev_H = np.load("/media/HDD1/dual_modality_datasets/24_01_28/valid_seq/24_01_28_output_03/homography_inv_0.5.npy")
    input_points = np.array([[0, 0], [1439, 0], [1439, 1079], [0, 1079]])
    output_points = p2p_homo(input_points, rev_H)
    print(output_points)


def cal_homo_from_points():
    pt1_list = np.load("/media/HDD1/dual_modality_datasets/24_01_28/valid_seq/24_01_28_output_03/test_homo/pt1_list_thre_0.5.npy", allow_pickle=True)
    pt1_list = np.concatenate(pt1_list, axis=0)
    pt2_list = np.load("/media/HDD1/dual_modality_datasets/24_01_28/valid_seq/24_01_28_output_03/test_homo/pt2_list_thre_0.5.npy", allow_pickle=True)
    pt2_list = np.concatenate(pt2_list, axis=0)
    homography, _ = cv2.findHomography(pt1_list, pt2_list, cv2.RANSAC)  # cv2.RANSAC会剔除错误点
    homography_inv, _ = cv2.findHomography(pt2_list, pt1_list, cv2.RANSAC)
    np.save("/media/HDD1/dual_modality_datasets/24_01_28/valid_seq/24_01_28_output_03/homo_align/H.npy", homography_inv)
    np.save("/media/HDD1/dual_modality_datasets/24_01_28/valid_seq/24_01_28_output_03/homo_align/rev_H.npy", homography)

if __name__ == "__main__":
    pass


