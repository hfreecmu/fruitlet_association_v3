from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import Boxes
import json
import numpy as np
import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

device = 'cuda'

box_transform = T.Resize((64, 64))

disps_mean = 138.27248
disps_std = 83.86233
rows_mean = 539.5
rows_std  = 311.76901171647364
cols_mean = 719.5
cols_std = 415.69209358209673

#TODO 
#A lot of repeating create_cfgs, maybe move to one place
#in case a setting changes
#Or create your own cfg as before and read that file with CFG path
def create_cfg(model_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.INPUT.MAX_SIZE_TEST = 1440

    return cfg

def read_dict(path_to_read):
    with open(path_to_read, 'r') as f:
        dict_to_read = json.loads(f.read())

    return dict_to_read

def write_dict(path_to_write, json_dict):
    with open(path_to_write, 'w') as f:
        json.dump(json_dict, f)

def save_checkpoint(epoch, checkpoint_dir, model, accuracy=None, loss=None, is_best=False):
    if not is_best:
        path = os.path.join(checkpoint_dir, 'epoch_%d.pth' % epoch)
        print('Saving checkpoint: ' + path)
    else:
        print('updating best')
        path = os.path.join(checkpoint_dir, 'best.pth')

        is_best_json = {'save_epoch': epoch, 'loss': loss, 'accuracy': accuracy}
        json_path = os.path.join(checkpoint_dir, 'best.json')
        write_dict(json_path, is_best_json)
    
    torch.save(model.state_dict(), path)

def plot_losses(plot_loss_dir, losses):
    np_path = os.path.join(plot_loss_dir, 'loss.npy')
    plot_path = os.path.join(plot_loss_dir, 'loss.png')
    print('Plotting loss: ' + plot_path)

    np.save(np_path, losses)

    epochs = np.arange(losses.shape[0])
    plt.plot(epochs, losses, 'b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(plot_path)

    plt.clf()

def plot_val_losses(plot_loss_dir, val_losses, val_accs, val_epochs):
    loss_np_path = os.path.join(plot_loss_dir, 'val_loss.npy')
    loss_plot_path = os.path.join(plot_loss_dir, 'val_loss.png')
    acc_np_path = os.path.join(plot_loss_dir, 'val_acc.npy')
    acc_plot_path = os.path.join(plot_loss_dir, 'val_acc.png')
    print('Plotting validation')

    np.save(loss_np_path, val_losses)
    np.save(acc_np_path, val_accs)

    plt.plot(val_epochs, val_losses, 'b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(loss_plot_path)

    plt.clf()

    plt.plot(val_epochs, val_accs, 'r')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(acc_plot_path)

    plt.clf()


def get_boxes(associations, box_seg_im, tag_seg_im, disparities, rows, cols, force_assoc):
    out_boxes = []
    is_tag = []
    keypoint_vecs = []
    detection_ids = []
    assoc_scores = []
    assoc_ids = []

    for i in range(len(associations)):
        box = associations[i]

        label = box['label']

        x0 = box['left']
        x1 = x0 + box['width']
        y0 = box['top']
        y1 = y0 + box['height']

        area = (x1-x0)*(y1-y0)
        if area == 0:
            continue

        if label in ['foreground_fruitlet', 'background_fruitlet']:
            is_tag.append(False)
            
            enc_im_id = 255 - box['detection_id'] 
            box_seg = np.zeros((box_seg_im.shape))
            box_seg[box_seg_im == enc_im_id] = 1.0
            box_seg = box_seg[y0:y1, x0:x1]
        elif label == 'tag':
            #still ignore tags we don't want
            if not box['is_assoc']:
                continue
            is_tag.append(True)
            
            enc_im_id = 255 - box['detection_id'] 
            box_seg = np.zeros((tag_seg_im.shape))
            box_seg[tag_seg_im == enc_im_id] = 1.0
            box_seg = box_seg[y0:y1, x0:x1]
        else:
            raise RuntimeError('Invalid label round: ' + label)
        
        box_seg = torch.as_tensor(box_seg, dtype=torch.float32).cuda()

        box_disp = disparities[y0:y1, x0:x1]
        box_row = rows[y0:y1, x0:x1]
        box_col = cols[y0:y1, x0:x1]
        keypoint_vec = torch.stack((box_disp, box_row, box_col, box_seg))
        keypoint_vec = box_transform(keypoint_vec)
        
        out_boxes.append([x0, y0, x1, y1])
        keypoint_vecs.append(keypoint_vec)

        detection_ids.append(box['detection_id'])
        assoc_scores.append(box['assoc_score'])

        if (force_assoc) or ('assoc_id' in box):
            assoc_ids.append(box['assoc_id'])
               
    out_boxes = np.vstack(out_boxes)
    out_boxes = torch.as_tensor(out_boxes, dtype=torch.float32).cuda()
    keypoint_vecs = torch.stack(keypoint_vecs)

    return Boxes(out_boxes), is_tag, keypoint_vecs, detection_ids, assoc_scores, assoc_ids

def extract_descriptors(associations, image_0_path, image_1_path, tag_seg_image_0_path, tag_seg_image_1_path,  
                        box_seg_im_0_path, box_seg_im_1_path, disparity_0_path, disparity_1_path, feature_predictor, force_assoc=True):
    tag_seg_im_0 = cv2.imread(tag_seg_image_0_path, cv2.IMREAD_GRAYSCALE).copy()
    tag_seg_im_1 = cv2.imread(tag_seg_image_1_path, cv2.IMREAD_GRAYSCALE).copy()
    box_seg_im_0 = cv2.imread(box_seg_im_0_path, cv2.IMREAD_GRAYSCALE).copy()
    box_seg_im_1 = cv2.imread(box_seg_im_1_path, cv2.IMREAD_GRAYSCALE).copy()

    im_0 = cv2.imread(associations['image_0']).copy()
    im_1 = cv2.imread(associations['image_1']).copy()
    disparities_0 = np.load(disparity_0_path)
    disparities_1 = np.load(disparity_1_path)

    disparities_0 = torch.as_tensor(disparities_0, dtype=torch.float32).cuda()
    disparities_1 = torch.as_tensor(disparities_1, dtype=torch.float32).cuda()
    row_mat, col_mat = torch.meshgrid(torch.arange(1080).float().cuda(), torch.arange(1440).float().cuda())

    disparities_0 = (disparities_0 - disps_mean) / disps_std
    disparities_1 = (disparities_1 - disps_mean) / disps_std
    row_mat = (row_mat - rows_mean) / rows_std
    col_mat = (col_mat - cols_mean) / cols_std 

    boxes_0, is_tag_0, keypoint_vecs_0, detection_ids_0, assoc_scores_0, assoc_ids_0 = get_boxes(associations['annotations_0'], box_seg_im_0, tag_seg_im_0, disparities_0, row_mat, col_mat, force_assoc)
    boxes_1, is_tag_1, keypoint_vecs_1, detection_ids_1, assoc_scores_1, assoc_ids_1 = get_boxes(associations['annotations_1'], box_seg_im_1, tag_seg_im_1, disparities_1, row_mat, col_mat, force_assoc)

    if np.sum(is_tag_0) != 1:
        raise RuntimeError('Not 1 tag: ' + associations['image_0'])

    if np.sum(is_tag_1) != 1:
        raise RuntimeError('Not 1 tag: ' + associations['image_1'])
    
    with torch.no_grad():
        box_features_0 = feature_predictor(original_image=im_0, boxes=boxes_0)
        box_features_1 = feature_predictor(original_image=im_1, boxes=boxes_1)

    width = im_0.shape[1]
    height = im_0.shape[0]

    is_tag_0 = torch.as_tensor(is_tag_0, dtype=torch.long).cuda()
    is_tag_1 = torch.as_tensor(is_tag_1, dtype=torch.long).cuda()

    assoc_scores_0 = torch.as_tensor(assoc_scores_0, dtype=torch.float).cuda()
    assoc_scores_1 = torch.as_tensor(assoc_scores_1, dtype=torch.float).cuda()

    box_features = [box_features_0, box_features_1]
    keypoint_vecs = [keypoint_vecs_0, keypoint_vecs_1]
    is_tag = [is_tag_0, is_tag_1]
    detection_ids = [detection_ids_0, detection_ids_1]
    assoc_scores = [assoc_scores_0, assoc_scores_1]
    assoc_ids = [assoc_ids_0, assoc_ids_1]

    return box_features, keypoint_vecs, width, height, is_tag, detection_ids, assoc_scores, assoc_ids
