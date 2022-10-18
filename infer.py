import argparse
from data.dataloader import get_data_loader
from models.associator import FruitletAssociator
import torch
import torch.optim as optim
import torch.nn as nn
import util
import numpy as np
import os
import cv2
import json

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--match_thresh', type=float, default=0.6)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_assoc_dir', default=None)
    parser.add_argument('--vis_dir', default=None)

    args = parser.parse_args()
    return args

def get_model(opt):
    config = {'match_threshold': opt.match_thresh}
    model = FruitletAssociator(config)
    model = model.to(util.device)

    return model

def vis(indices0, indices1, detection_ids_0, detection_ids_1, is_tag_0, is_tag_1, M, assoc_path, vis_path):

    assoc = util.read_dict(assoc_path)

    im_0_path = assoc['image_0']
    im_1_path = assoc['image_1']

    im_0 = cv2.imread(im_0_path).copy()
    im_1 = cv2.imread(im_1_path).copy()

    im = np.zeros((im_0.shape[0], 2*im_0.shape[1], im_0.shape[2]))
    im[:, 0:im_0.shape[1], :] = im_0
    im[:, im_0.shape[1]:, :] = im_1

    gt_im = im.copy()

    det_id0_dict = {}
    for det in assoc['annotations_0']:
        if 'fruitlet' not in det['label']:
            continue

        x0 = det['left']
        x1 = x0 + det['width']
        y0 = det['top']
        y1 = y0 + det['height']

        cy = (y0 + y1) / 2
        cx = (x0 + x1) / 2

        det_id0_dict[det['detection_id']] = [cy, cx]

    det_id1_dict = {}
    for det in assoc['annotations_1']:
        if 'fruitlet' not in det['label']:
            continue

        x0 = det['left']
        x1 = x0 + det['width']
        y0 = det['top']
        y1 = y0 + det['height']

        cy = (y0 + y1) / 2
        cx = (x0 + x1) / 2

        det_id1_dict[det['detection_id']] = [cy, cx]
        
    for i in range(indices0.shape[0]):
        ind0 = indices0[i]
        if ind0 == -1:
            continue

        assert(indices1[ind0] == i)

        if is_tag_0[i]:
            continue
        if is_tag_1[ind0]:
            continue

        if (i in M[:, 0]) or (ind0 in M[:, 1]):
            i_ind = np.where(M[:, 0] == i)[0]
            if (i_ind.shape[0] == 0) or (M[i_ind[0], 1] != ind0):
                colour = (255, 0, 255)
            else:
                colour = (0, 0, 255)
        else:
            continue
            #colour = (0, 165, 255)

        det_id_0 = detection_ids_0[i]
        det_id_1 = detection_ids_1[ind0]

        cy0, cx0 = det_id0_dict[det_id_0]
        cy1, cx1 = det_id1_dict[det_id_1]

        cv2.line(im, (int(cx0), int(cy0)), (int(cx1) + im_0.shape[1], int(cy1)), colour, 2) 

    for i in range(M.shape[0]):
        ind0, ind1 = M[i]

        det_id_0 = detection_ids_0[ind0]
        det_id_1 = detection_ids_1[ind1]

        cy0, cx0 = det_id0_dict[det_id_0]
        cy1, cx1 = det_id1_dict[det_id_1]
        
        cv2.line(gt_im, (int(cx0), int(cy0)), (int(cx1) + im_0.shape[1], int(cy1)), (0, 0, 255), 2)
    
    full_im = np.zeros((2*im.shape[0], im.shape[1], im.shape[2]))
    full_im[0:im.shape[0], :, :] = im
    full_im[im.shape[0]:, :, :] = gt_im 
    cv2.imwrite(vis_path, full_im)


def infer(opt):
    dataloader = get_data_loader(opt, opt.feature_dir, False)
    model = get_model(opt)
    model.load_state_dict(torch.load(opt.checkpoint_path))
    model.eval()

    num_correct = 0
    num_incorrect = 0
    num_true_pos = 0
    num_total = 0

    with torch.no_grad():
        for batch in dataloader:
            descs, kpts, width, height, is_tag, assoc_scores, M, detection_ids, feature_path = batch

            descs = [[descs[0][0]], [descs[1][0]]]
            kpts = [[kpts[0][0]], [kpts[1][0]]]
            is_tag = [[is_tag[0][0]], [is_tag[1][0]]]
            assoc_scores = [[assoc_scores[0][0]], [assoc_scores[1][0]]]
            M = [M[0]]

            data = {}
            data['descriptors'] = descs
            data['keypoints'] = kpts
            data['is_tag'] = is_tag
            data['assoc_scores'] = assoc_scores
            data['M'] = M
            data['return_matches'] = True

            res = model(data)

            indices0 = res['matches0'][0].detach().cpu().numpy()
            indices1 = res['matches1'][0].detach().cpu().numpy()

            match_dict_0 = {}
            match_dict_1 = {}
            M = M[0].detach().cpu().numpy()
            for j in range(M.shape[0]):
                ind_0, ind_1 = M[j]
                match_dict_0[ind_0] = ind_1
                match_dict_1[ind_1] = ind_0

            for j in range(indices0.shape[0]):
                if indices0[j] == -1:
                    if j in match_dict_0:
                        num_incorrect += 1
                        num_total += 1
                    else:
                        #don't care
                        pass
                else:
                    if indices1[indices0[j]] != j:
                        raise RuntimeError('Why here, should not happen')
                    if j in match_dict_0:
                        if indices0[j] != match_dict_0[j]:
                            num_incorrect += 1
                        else:
                            num_correct += 1
                        num_total += 1
                    else:
                        #don't care
                        pass

            for j in range(indices1.shape[0]):
                if indices1[j] == -1:
                    if j in match_dict_1:
                        num_incorrect += 1
                        num_total += 1
                    else:
                        #don't care
                        pass
            
            if opt.vis:
                detection_ids_0 = detection_ids[0][0].detach().cpu().numpy()
                detection_ids_1 = detection_ids[1][0].detach().cpu().numpy()

                is_tag_0 = is_tag[0][0].detach().cpu().numpy()
                is_tag_1 = is_tag[1][0].detach().cpu().numpy()

                basename = os.path.basename(feature_path[0]).split('.pkl')[0]
                assoc_path = os.path.join(opt.vis_assoc_dir, basename + '.json')
                
                vis_path = os.path.join(opt.vis_dir, basename + '.png')
                vis(indices0, indices1, detection_ids_0, detection_ids_1, is_tag_0, is_tag_1, M, assoc_path, vis_path)
    
    accuracy = num_correct / num_total
    print('Accuracy is: ', accuracy)

if __name__ == "__main__":
    opt = parse_args()

    infer(opt)