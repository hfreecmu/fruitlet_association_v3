import argparse
import os
import torch

from models.feature_predictor import FeaturePredictor
import util

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--associations_dir', required=True)
    parser.add_argument('--tag_segment_dir', require=True)
    parser.add_argument('--box_segment_dir', require=True)
    parser.add_argument('--disparity_dir', require=True)
    parser.add_argument('--model_path', require=True)
    parser.add_argument('--output_dir', require=True)
    parser.add_argument('--num_rand', type=int, default=0)

    args = parser.parse_args()
    return args

def extract_descriptors(associations_dir, tag_segment_dir, box_segment_dir, disparity_dir, model_path, output_dir, num_rand=0):
    cfg = util.create_cfg(model_path)
    feature_predictor = FeaturePredictor(cfg)
    feature_predictor.eval()

    for filename in os.listdir(associations_dir):
        if not filename.endswith('.json'):
            continue

        associations_path = os.path.join(associations_dir, filename)

        associations = util.read_dict(associations_path)

        image_0_path = associations['image_0']
        basename_0 = os.path.basename(image_0_path).replace('.png', '')
        tag_seg_image_0_path = os.path.join(tag_segment_dir, basename_0 + '.png')
        box_seg_im_0_path = os.path.join(box_segment_dir, basename_0 + '.png')
        disparity_0_path = os.path.join(disparity_dir, basename_0 + '.npy')
        file_checks_0 = [image_0_path, tag_seg_image_0_path, box_seg_im_0_path, disparity_0_path, output_dir]

        image_1_path = associations['image_1']
        basename_1 = os.path.basename(image_1_path).replace('.png', '')
        tag_seg_image_1_path = os.path.join(tag_segment_dir, basename_1 + '.png')
        box_seg_im_1_path = os.path.join(box_segment_dir, basename_1 + '.png')
        disparity_1_path = os.path.join(disparity_dir, basename_1 + '.npy')
        file_checks_1 = [image_1_path, tag_seg_image_1_path, box_seg_im_1_path, disparity_1_path, output_dir]

        for file_path in file_checks_0:
            if not os.path.exists(file_path):
                raise RuntimeError('Path does not exist: ' + file_path)

        for file_path in file_checks_1:
            if not os.path.exists(file_path):
                raise RuntimeError('Path does not exist: ' + file_path)

        descs, kpts, width, height, is_tag, detection_ids, assoc_scores, assoc_ids = \
            util.extract_descriptors(associations, image_0_path, image_1_path, tag_seg_image_0_path, tag_seg_image_1_path,  
            box_seg_im_0_path, box_seg_im_1_path, disparity_0_path, disparity_1_path, feature_predictor)

        data = {}
        data['descriptors'] = descs
        data['keypoints'] = kpts
        data['width'] = width
        data['height'] = height
        data['is_tag'] = is_tag
        data['detection_ids'] = detection_ids
        data['assoc_scores'] = assoc_scores
        data['assoc_ids'] = assoc_ids

        output_path = os.path.join(output_dir, filename.replace('.json', '.pkl'))
        torch.save(data, output_path)

        for i in range(num_rand):
            descs, kpts, width, height, is_tag, detection_ids, assoc_scores, assoc_ids = \
                util.extract_descriptors(associations, image_0_path, image_1_path, tag_seg_image_0_path, tag_seg_image_1_path,  
                box_seg_im_0_path, box_seg_im_1_path, disparity_0_path, disparity_1_path, feature_predictor, rand_shift=True)

            data = {}
            data['descriptors'] = descs
            data['keypoints'] = kpts
            data['width'] = width
            data['height'] = height
            data['is_tag'] = is_tag
            data['detection_ids'] = detection_ids
            data['assoc_scores'] = assoc_scores
            data['assoc_ids'] = assoc_ids

            output_path = os.path.join(output_dir, filename.replace('.json', '_rand_shift_' + str(i) + '.pkl'))
            torch.save(data, output_path)

    print('done')
    