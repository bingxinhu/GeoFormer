from argparse import Namespace
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

    def change_deivce(self, device):
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
            import matplotlib.pyplot as plt
            import cv2
            import numpy as np
            plt.figure(dpi=200)
            kp0 = kpts1
            kp1 = kpts2
            # if len(kp0) > 0:
            kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
            kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
            matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in
                       range(len(kp0))]

            show = cv2.drawMatches((gray1.cpu()[0][0].numpy() * 255).astype(np.uint8), kp0,
                                   (gray2.cpu()[0][0].numpy() * 255).astype(np.uint8), kp1, matches,
                                   None)
            #zhaifang add 
            cv2.imwrite('matched_image.jpg', show)
            plt.imshow(show)
            plt.show()
        if is_draw:
            draw()
        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path, cpu=False, is_draw=False):
        torch.cuda.empty_cache()
        tmp_device = self.device
        if cpu:
            self.change_deivce('cpu')
        gray1, sc1 = self.load_im(im1_path)
        gray2, sc2 = self.load_im(im2_path)

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2, is_draw)
        # Sort by scores
        indices = np.argsort(scores)[::-1]  # Sort in descending order
        sorted_matches = matches[indices]
        sorted_kpts1 = kpts1[indices]
        sorted_kpts2 = kpts2[indices]
        sorted_scores = scores[indices]

        # Determine a threshold for the best matches
        threshold = 0.5  # Adjust based on your needs
        best_matches = sorted_matches[sorted_scores > threshold]
        best_kpts1 = sorted_kpts1[sorted_scores > threshold]
        best_kpts2 = sorted_kpts2[sorted_scores > threshold]

        if len(best_kpts1) >= 4:
            # Calculate homography matrix
            homography, _ = cv2.findHomography(best_kpts1, best_kpts2, cv2.RANSAC)
            homography_inv, _ = cv2.findHomography(best_kpts2, best_kpts1, cv2.RANSAC)
        else:
            homography = None
            homography_inv = None

        if self.no_match_upscale:
            return best_matches, best_kpts1, best_kpts2, sorted_scores, upscale.squeeze(0), homography, homography_inv

        # Upscale matches & kpts
        best_matches = upscale * best_matches
        best_kpts1 = sc1 * best_kpts1
        best_kpts2 = sc2 * best_kpts2

        if cpu:
            self.change_deivce(tmp_device)
    
        print("Best matches:", best_matches)
        print("KeyPoints 1:", best_kpts1)
        print("KeyPoints 2:", best_kpts2)
        print("Scores:", sorted_scores)
        print("Homography matrix:", homography)
        print("Homography matrix inv:", homography_inv)
        return best_matches, best_kpts1, best_kpts2, sorted_scores, homography, homography_inv

g = GeoFormer(1440, 0.2, no_match_upscale=False, ckpt='saved_ckpt/geoformer.ckpt', device='cuda')
g.match_pairs('./data/test/000403.png', './data/test/frame_000402.png', is_draw=True)
