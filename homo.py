import os
import argparse
import torch
import numpy as np
import cv2

from model.loftr_src.loftr.utils.cvpr_ds_config import default_cfg
from model.full_model import GeoFormer as GeoFormer_
from eval_tool.immatch.utils.data_io import load_gray_scale_tensor_cv
from model.geo_config import default_cfg as geoformer_cfg

class GeoFormer():
    def __init__(self, imsize, match_threshold, no_match_upscale=False, ckpt=None, device='cuda'):
        self.device = device
        self.imsize = imsize
        self.match_threshold = match_threshold
        self.no_match_upscale = no_match_upscale

        # Load model
        conf = dict(default_cfg)
        conf['match_coarse']['thr'] = self.match_threshold
        geoformer_cfg['coarse_thr'] = self.match_threshold
        self.model = GeoFormer_(conf)
        ckpt_dict = torch.load(ckpt, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt_dict:
            ckpt_dict = ckpt_dict['state_dict']
        self.model.load_state_dict(ckpt_dict, strict=False)
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = ckpt.split('/')[-1].split('.')[0]
        self.name = f'GeoFormer_{self.ckpt_name}'
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name}')

    def change_device(self, device):
        self.device = device
        self.model.to(device)

    def load_im(self, im_path, enhanced=False):
        return load_gray_scale_tensor_cv(
            im_path, self.device, imsize=self.imsize, dfactor=8, enhanced=enhanced, value_to_scale=min
        )

    def match_inputs_(self, gray1, gray2, is_draw=False):
        batch = {'image0': gray1, 'image1': gray2}
        with torch.no_grad():
            batch = self.model(batch)
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()

        def draw():
            #import matplotlib.pyplot as plt
            #plt.figure(dpi=200)
            kp0 = kpts1
            kp1 = kpts2
            kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
            kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
            matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in range(len(kp0))]
            show = cv2.drawMatches((gray1.cpu()[0][0].numpy() * 255).astype(np.uint8), kp0,
                                   (gray2.cpu()[0][0].numpy() * 255).astype(np.uint8), kp1, matches, None)
            #cv2.imwrite('matched_image.jpg', show)
            #plt.imshow(show)
            #plt.show()

        if is_draw:
            draw()

        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path, cpu=False, is_draw=False, npy_file=None, output_file=None, h_dir=None):
        torch.cuda.empty_cache()
        tmp_device = self.device
        if cpu:
            self.change_device('cpu')
        gray1, sc1 = self.load_im(im1_path)
        gray2, sc2 = self.load_im(im2_path)

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2, is_draw)

        # Sort by scores
        indices = np.argsort(scores)[::-1]
        sorted_matches = matches[indices]
        sorted_kpts1 = kpts1[indices]
        sorted_kpts2 = kpts2[indices]
        sorted_scores = scores[indices]

        threshold = 0.5
        best_matches = sorted_matches[sorted_scores > threshold]
        best_kpts1 = sorted_kpts1[sorted_scores > threshold]
        best_kpts2 = sorted_kpts2[sorted_scores > threshold]

        if len(best_kpts1) >= 4:
            homography, _ = cv2.findHomography(best_kpts1, best_kpts2, cv2.RANSAC)
            homography_inv, _ = cv2.findHomography(best_kpts2, best_kpts1, cv2.RANSAC)
            # Extract base names for saving
            im1_base = os.path.basename(im1_path).split('.')[0]
            im2_base = os.path.basename(im2_path).split('.')[0]

            # Save homography matrices with paired file names in the specified directory
            np.save(os.path.join(h_dir, f'homography_{im1_base}_{im2_base}.npy'), homography)
            np.save(os.path.join(h_dir, f'homography_inv_{im1_base}_{im2_base}.npy'), homography_inv)
        else:
            homography = None
            homography_inv = None

        if self.no_match_upscale:
            if npy_file and output_file and homography_inv is not None:
                self.map_points_and_save_image(npy_file, im1_path, homography_inv, output_file)
            return best_matches, best_kpts1, best_kpts2, sorted_scores, upscale.squeeze(0), homography, homography_inv

        best_matches = upscale * best_matches
        best_kpts1 = sc1 * best_kpts1
        best_kpts2 = sc2 * best_kpts2

        if cpu:
            self.change_device(tmp_device)

        if npy_file and output_file and homography_inv is not None:
            self.map_points_and_save_image(npy_file, im1_path, homography_inv, output_file)

        return best_matches, best_kpts1, best_kpts2, sorted_scores, homography, homography_inv

    def map_points_and_save_image(self, npy_file, image_file, H, output_file):
        dvs_points = np.load(npy_file, allow_pickle=True)
        x_coords = dvs_points['x'].astype(np.float32)
        y_coords = dvs_points['y'].astype(np.float32)
        p_values = dvs_points['p'].astype(np.float32)

        x_coords = np.round(x_coords).astype(np.float32)
        y_coords = np.round(y_coords).astype(np.float32)

        dvs_points_extracted = np.column_stack((x_coords, y_coords)).astype(np.float32)

        mapped_points = cv2.perspectiveTransform(dvs_points_extracted.reshape(-1, 1, 2), H)
        target_image = cv2.imread(image_file)

        for point, p in zip(mapped_points, p_values):
            x, y = int(point[0][0]), int(point[0][1])
            if 0 <= x < target_image.shape[1] and 0 <= y < target_image.shape[0]:
                color = (255, 0, 0) if p == 1 else (0, 0, 225)
                cv2.circle(target_image, (x, y), 1, color, -1)
        cv2.imwrite(output_file, target_image)

def process_directory(base_dir):
    # 创建 mapped_images 目录（如果不存在的话）
    output_dir = os.path.join(base_dir, 'mapped_images')
    os.makedirs(output_dir, exist_ok=True)
    h_dir = os.path.join(base_dir, 'homo_matrix')
    os.makedirs(h_dir, exist_ok=True)

    raw_frame_dir = os.path.join(base_dir, 'raw_frame')
    reconstruction_dir = os.path.join(base_dir, 'ImageRecons/reconstruction')
    time_split_dir = os.path.join(base_dir, 'time_split')

    image_files = sorted(os.listdir(raw_frame_dir))
    recon_files = sorted(os.listdir(reconstruction_dir))
    npy_files = sorted(os.listdir(time_split_dir))

    for i in range(len(image_files)):
        if image_files[i].endswith('.png') and recon_files[i].endswith('.png') and npy_files[i-1].endswith('.npy'):
            im1_path = os.path.join(raw_frame_dir, image_files[i])
            im2_path = os.path.join(reconstruction_dir, recon_files[i])
            npy_file = os.path.join(time_split_dir, npy_files[i-1])
            output_file = os.path.join(base_dir, 'mapped_images', f'mapped_image_{i:03d}.jpg')
            print(i, image_files[i], recon_files[i], npy_files[i-1], output_file)
            g.match_pairs(im1_path, im2_path, is_draw=True, npy_file=npy_file, output_file=output_file, h_dir=h_dir)

def main():
    parser = argparse.ArgumentParser(description='Process image pairs and DVS data.')
    parser.add_argument('base_dir', type=str, help='Base directory containing the data.')
    args = parser.parse_args()

    global g
    g = GeoFormer(1440, 0.2, no_match_upscale=False, ckpt='saved_ckpt/geoformer.ckpt', device='cuda')
    process_directory(args.base_dir)

if __name__ == '__main__':
    main()
