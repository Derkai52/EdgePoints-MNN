import copy
import os
import cv2
import glob
import logging
import argparse
import numpy as np
from tqdm import tqdm
from EdgePoint import EdgePoint
import matplotlib.pyplot as plt
from soft_detect import DKD
import torch
import time
import math



def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.FM_RANSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    print('inlier ratio: ', np.sum(mask)/len(mask))

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches



def mnn_mather(desc1, desc2):
    sim = desc1 @ desc2.transpose()
    sim[sim < 0.9] = 0
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    return matches.transpose()


class EdgePointInterface(EdgePoint):
    def __init__(self,
                 # ================================== feature encoder
                 params,
                 # ================================== detect parameters
                 radius: int = 2,
                 top_k: int = 500, scores_th: float = 0.01,
                 n_limit: int = 5000,
                 device: str = 'cuda',
                 model_path: str = ''
                 ):
        super().__init__(params)
        self.radius = radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th
        self.dkd = DKD(radius=self.radius, top_k=self.top_k,
                       scores_th=self.scores_th, n_limit=self.n_limit)
        self.device = device

        if model_path != '':
            state_dict = torch.load(model_path, self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)
            self.eval()
            logging.info(f'Loaded model parameters from {model_path}')
            logging.info(
                f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e3}KB")

    def extract_dense_map(self, image, ret_dict=False):
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        b, c, h, w = image.shape
        # print(b, c, h, w)
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w

        if h_ != h:
            h_padding = torch.zeros(b, c, h_ - h, w, device=device)
            image = torch.cat([image, h_padding], dim=2)
        if w_ != w:
            w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
            image = torch.cat([image, w_padding], dim=3)
        # ====================================================

        scores_map, descriptor_map = super().forward(image)
        # print(scores_map)
        # ====================================================
        if h_ != h or w_ != w:
            descriptor_map = descriptor_map[:, :, :h, :w]
            scores_map = scores_map[:, :, :h, :w]  # Bx1xHxW
        # ====================================================
        # print(descriptor_map.shape)

        # BxCxHxW
        descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2, dim=1)

        if ret_dict:
            return {'descriptor_map': descriptor_map, 'scores_map': scores_map, }
        else:
            return descriptor_map, scores_map

    def forward(self, img, image_size_max=99999, sort=False, sub_pixel=False):
        """
        :param img: np.array HxWx3, RGB
        :param image_size_max: maximum image size, otherwise, the image will be resized
        :param sort: sort keypoints by scores
        :param sub_pixel: whether to use sub-pixel accuracy
        :return: a dictionary with 'keypoints', 'descriptors', 'scores', and 'time'
        """
        H, W, three = img.shape
        assert three == 3, "input image shape should be [HxWx3]"

        # ==================== image size constraint
        image = copy.deepcopy(img)
        max_hw = max(H, W)
        if max_hw > image_size_max:
            ratio = float(image_size_max / max_hw)
            image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

        # ==================== convert image to tensor
        image = torch.from_numpy(image).to(self.device).to(torch.float32).permute(2, 0, 1)[None] / 255.0
        # ==================== extract keypoints
        start = time.time()
        with torch.no_grad():
            descriptor_map, scores_map = self.extract_dense_map(image)

            keypoints, descriptors, scores, _ = self.dkd(scores_map, descriptor_map,
                                                         sub_pixel=sub_pixel)
            
            keypoints, descriptors, scores = keypoints[0], descriptors[0], scores[0]
            keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W - 1, H - 1]])
        if sort:
            indices = torch.argsort(scores, descending=True)
            keypoints = keypoints[indices]
            descriptors = descriptors[indices]
            scores = scores[indices]

        end = time.time()

        # return {'keypoints': keypoints,
        #         'descriptors': descriptors,
        #         'scores': scores,
        #         'scores_map': scores_map,
        #         'time': end - start, }
        return {'keypoints': keypoints.cpu().numpy(),
                'descriptors': descriptors.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'scores_map': scores_map.cpu().numpy(),
                'time': end - start, }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EdgePoint Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera0" (for webcam0).')
    parser.add_argument('--model', default="../model/EdgePoint.pt",
                        help="The model path (default: ../model/EdgePoint.pt).")
    parser.add_argument('--device', type=str, default='cuda', help="Running device (default: cuda).")
    parser.add_argument('--top_k', type=int, default=1000,
                        help='Detect top K keypoints. -1 for threshold based mode, >0 for top K mode. (default: -1)')
    parser.add_argument('--scores_th', type=float, default=0.01,
                        help='Detector score threshold (default: 0.2).')
    parser.add_argument('--n_limit', type=int, default=2000,
                        help='Maximum number of keypoints to be detected (default: 5000).')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--no_sub_pixel', action='store_true',
                        help='Do not detect sub-pixel keypoints (default: False).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    param = {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64}
    model = EdgePointInterface(param,
                  device=args.device,
                  top_k=args.top_k,
                  scores_th=args.scores_th,
                  n_limit=args.n_limit,
                  model_path=args.model)
    runtime = []


    im1 = cv2.imread('/home/emnavi/image1.png')
    im2 = cv2.imread('/home/emnavi/image2.png')
    im1 = cv2.resize(im1, (640, 480)) 
    im2 = cv2.resize(im2, (640, 480)) 
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    output0 = model(im1, sub_pixel=not args.no_sub_pixel)
    kpts1 = output0['keypoints']
    print(kpts1)
    desc1 = output0['descriptors']
    runtime.append(output0['time'])

    # output1 = model(im2, sub_pixel=not args.no_sub_pixel)
    # kpts2 = output1['keypoints']
    # desc2 = output1['descriptors']
    # runtime.append(output1['time'])

    #Update with image resolution (required)
    # output0.update({'image_size': (im1.shape[1], im1.shape[0])})
    # output1.update({'image_size': (im2.shape[1], im2.shape[0])})
    # mkpts_0, mkpts_1, _e = match_lighterglue(output0, output1)
    # canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
    # plt.figure(figsize=(12,12))
    # plt.imshow(canvas[..., ::-1]), plt.show()
    # out, N_matches = tracker.update(img, kpts, desc)

    # matches = mnn_mather(desc1, desc2)
    # mpts1, mpts2 = kpts1[matches[:, 0]], kpts2[matches[:, 1]]
    # N_matches = len(matches)
    # canvas = warp_corners_and_draw_matches(mpts1, mpts2, im1, im2)
    # plt.figure(figsize=(12,12))
    # plt.imshow(canvas[..., ::-1]), plt.show()
    # out = copy.deepcopy(im1)
    # for pt1, pt2 in zip(mpts1, mpts2):
    #     p1 = (int(round(pt1[0])), int(round(pt1[1])))
    #     p2 = (int(round(pt2[0])), int(round(pt2[1])))
    #     print(p1)
    #     print(p2)
    #     cv2.line(out, p1, p2, (0, 255, 0), lineType=16)
    #     cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)
    # cv2.imshow(args.model, out)
    # cv2.waitKey(0)


    # ave_fps = (1. / np.stack(runtime)).mean()
    # status = f"Fps:{ave_fps:.1f}, Keypoints/Matches: {len(kpts)}/{N_matches}"
    # progress_bar.set_description(status)

        # if not args.no_display:
        #     cv2.setWindowTitle(args.model, args.model + ': ' + status)
        #     cv2.imshow(args.model, out)
        #     if cv2.waitKey(1) == ord('q'):
        #         break
    # logging.info('Finished!')
    # if not args.no_display:
    #     logging.info('Press any key to exit!')
    #     cv2.waitKey()


