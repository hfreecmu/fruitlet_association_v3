import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

def rand_flip(descs, kpts):
    rand_var = np.random.uniform()
    if rand_var < 0.5:
        descs = torch.fliplr(descs)
        kpts = torch.fliplr(kpts)

    return descs, kpts

#TODO can make faster to optimize
def rand_drop(descs, kpts, is_tag, assoc_scores, detection_ids, assoc_ids):
    should_keep = [None]*descs.shape[0]

    for i in range(descs.shape[0]):
        if is_tag[i]:
            should_keep[i]= True
        elif is_assoc[i]:
            prob_drop = np.random.uniform(0, 0.1)
            rand_var = np.random.uniform()
            should_keep[i] = rand_var >= prob_drop
        else:
            prob_drop = np.random.uniform(0.2, 0.6)
            rand_var = np.random.uniform()
            should_keep[i] = rand_var >= prob_drop

    descs = descs[should_keep]
    kpts = kpts[should_keep]
    is_tag = is_tag[should_keep]
    assoc_scores = assoc_scores[should_keep]
    detection_ids = detection_ids[should_keep]
    assoc_ids = assoc_ids[should_keep]

    return descs, kpts, is_tag, assoc_scores, detection_ids, assoc_ids

def rand_remove_assoc(assoc_scores, assoc_ids):
    should_keep = [None]*descs.shape[0]

    rand_var = np.random.uniform()
    if rand_var > 0.08:
        return

    inds = np.where(assoc_ids > -1)
    print(inds.shape)
    sfd

    rand_ind = np.random.randint(inds.shape[0])
    assoc_inds[rand_ind] = -1
    assoc_scores[rand_ind] = np.random.uniform(0.0, 0.2)

    return descs, kpts, is_tag, assoc_scores, detection_ids, assoc_ids

def rand_assoc_scores(is_tag, assoc_scores):
    rand_add = (np.random.uniform(size=assoc_scores.shape[0]) - 0.5)*0.02
    assoc_scores[is_tag == 0] = assoc_scores[is_tag == 0] + rand_add[is_tag == 0]
    assoc_scores[assoc_scores > 1] = 1
    assoc_scores[assoc_scores < 0] = 0

    return is_tag, assoc_scores 

#TODO add shift in extract_feature_descriptors
def augment(descs, kpts, is_tag, assoc_scores, detection_ids, assoc_ids):
    for i in range(2):
        descs[i], kpts[i] = rand_flip(descs[i], kpts[i])
        #assoc_scores[i], assoc_ids[i] = rand_remove_assoc[assoc_scores[i], assoc_ids[i]]
        #descs[i], kpts[i], is_tag[i], assoc_scores[i], detection_ids[i], assoc_ids[i] = rand_drop(descs[i], kpts[i], is_tag[i], assoc_scores[i], detection_ids[i], assoc_ids[i])
        #is_tag[i], assoc_scores[i] = rand_assoc_scores[is_tag[i], assoc_scores[i]]

    return descs, kpts, is_tag, assoc_scores, detection_ids, assoc_ids

#WARNING WARNING WARNING
#TODO
#Normalize descriptors?
class AssociationDataSet(Dataset):
    """Load images under folders"""
    def __init__(self, feature_dir, augment):
        self.features = self.get_features(feature_dir)
        self.augment = augment

    def get_features(self, feature_dir):
        feature_dict = {}
        for filename in os.listdir(feature_dir):
            if not filename.endswith('.pkl'):
                continue
        
            basename = filename

            if basename not in feature_dict:
                feature_dict[basename] = []

            feature_dict[basename].append(os.path.join(feature_dir, filename))

        return list(feature_dict.values())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        img_locs = self.features[idx]

        assert len(img_locs) == 1

        if len(img_locs) > 1:
            rand_ind = np.random.randint(len(img_locs))
            img_loc = img_locs[rand_ind]
        else:
            img_loc = img_locs[0]

        feature_dict = torch.load(img_loc)

        descs = feature_dict['descriptors']
        kpts = feature_dict['keypoints']
        width = feature_dict['width']
        height = feature_dict['height']
        is_tag = feature_dict['is_tag']
        assoc_scores = feature_dict['assoc_scores']
        detection_ids_0, detection_ids_1 = feature_dict['detection_ids']
        assoc_ids = feature_dict['assoc_ids']

        detection_ids = [np.array(detection_ids_0), np.array(detection_ids_1)]

        if self.augment:
            descs, kpts, is_tag, assoc_scores, detection_ids, assoc_ids = augment(descs, kpts, is_tag, assoc_scores, detection_ids, assoc_ids)

        assoc_ids_0, assoc_ids_1 = assoc_ids

        M = []
        found_matched_inds = set()
        for i in range(descs[0].shape[0]):
            assoc_id = assoc_ids_0[i]
            if assoc_id == -1:
                found = False
            elif assoc_id < 0:
                raise RuntimeError('why here?')
            else:
                try:
                    match_ind = assoc_ids_1.index(assoc_id)
                    found = True
                except:
                    found = False
            
            if found:
                M.append([i, match_ind])
                
        M = torch.as_tensor(M, dtype=torch.long).cuda()

        return descs, kpts, width, height, is_tag, assoc_scores, M, detection_ids, img_loc

def get_data_loader(opt, feature_dir, is_train, augment=False):

    dataset = AssociationDataSet(feature_dir, augment)
    dloader = DataLoader(dataset=dataset, batch_size=1, shuffle=is_train)

    return dloader
    