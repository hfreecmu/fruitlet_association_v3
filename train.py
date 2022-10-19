import argparse
from data.dataloader import get_data_loader
from models.associator import FruitletAssociator
import torch
import torch.optim as optim
import torch.nn as nn
import util
import numpy as np

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', required=True)
    parser.add_argument('--num_epochs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val', type=int, default=1)
    parser.add_argument('--val_feature_dir', default=None)
    parser.add_argument('--checkpoint', type=int, default=10)
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--plot_loss', type=int, default=1)
    parser.add_argument('--plot_loss_dir', default='./loss_plots')
    parser.add_argument('--match_thresh', type=float, default=0.6)

    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()
    return args

def get_model(opt):
    config = {'match_threshold': opt.match_thresh}
    model = FruitletAssociator(config)
    model = model.to(util.device)

    lr = opt.lr

    return model, lr

def train(opt):
    dataloader = get_data_loader(opt, opt.feature_dir, True, opt.augment)
    data_size = len(dataloader)

    if opt.val_feature_dir is not None:
        best_loss = None
        val_dataloader = get_data_loader(opt, opt.val_feature_dir, False, False)
        val_loss_array = []
        val_acc_array = []
        val_epochs = []

    gamma = 0.1
    milestones = [20, 40]

    model, lr = get_model(opt)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    loss_array = []
    for epoch in range(0, opt.num_epochs):
        losses = 0
        num_losses = 0

        batch_descs_0 = []
        batch_kpts_0 = []
        batch_is_tag_0 = []
        batch_assoc_scores_0 = []
        max_num_bbox_0 = 0

        batch_descs_1 = []
        batch_kpts_1 = []
        batch_is_tag_1 = []
        batch_assoc_scores_1 = []
        max_num_bbox_1 = 0

        batch_M = []

        for i, data in enumerate(dataloader):
            descs, kpts, width, height, is_tag, assoc_scores, M, _, _ = data

            descs = [descs[0][0], descs[1][0]]
            kpts = [kpts[0][0], kpts[1][0]]
            is_tag = [is_tag[0][0], is_tag[1][0]]
            assoc_scores = [assoc_scores[0][0], assoc_scores[1][0]]
            M = M[0]

            if len(descs[0]) > max_num_bbox_0:
                max_num_bbox_0 = len(descs[0])

            if len(descs[1]) > max_num_bbox_1:
                max_num_bbox_1 = len(descs[1])

            batch_descs_0.append(descs[0])
            batch_kpts_0.append(kpts[0])
            batch_is_tag_0.append(is_tag[0])
            batch_assoc_scores_0.append(assoc_scores[0])

            batch_descs_1.append(descs[1])
            batch_kpts_1.append(kpts[1])
            batch_is_tag_1.append(is_tag[1])
            batch_assoc_scores_1.append(assoc_scores[1])

            batch_M.append(M)

            if ((i +1) % opt.batch_size == 0) or (i == data_size - 1):
                num_adds_0 = []
                for j in range(len(batch_descs_0)):
                    db_0 = batch_descs_0[j]
                    kp_0 = batch_kpts_0[j]
                    is_tag_j_0 = batch_is_tag_0[j]
                    assoc_scores_0 = batch_assoc_scores_0[j]
                    
                    if db_0.shape[0] > max_num_bbox_0:
                        raise RuntimeError('Why here?')
                    elif db_0.shape[0] == max_num_bbox_0:
                        num_adds_0.append(0)
                        continue

                    num_adds_0.append(max_num_bbox_0 - db_0.shape[0])
                    
                    zero_desc_append = torch.zeros(max_num_bbox_0 - db_0.shape[0], db_0.shape[1], db_0.shape[2], db_0.shape[3], dtype=db_0.dtype).cuda()
                    batch_descs_0[j] = torch.cat([db_0, zero_desc_append], dim=0)

                    zero_kpts_append = torch.zeros(max_num_bbox_0 - kp_0.shape[0], kp_0.shape[1], kp_0.shape[2], kp_0.shape[3], dtype=kp_0.dtype).cuda()
                    batch_kpts_0[j] = torch.cat([kp_0, zero_kpts_append], dim=0)

                    zero_is_tag_append = torch.zeros(max_num_bbox_0 - is_tag_j_0.shape[0], dtype=is_tag_j_0.dtype).cuda()
                    batch_is_tag_0[j] = torch.cat([is_tag_j_0, zero_is_tag_append], dim=0)

                    zero_assoc_scors_append = torch.zeros(max_num_bbox_0 - assoc_scores_0.shape[0], dtype=assoc_scores_0.dtype).cuda()
                    batch_assoc_scores_0[j] = torch.cat([assoc_scores_0, zero_assoc_scors_append], dim=0)

                num_adds_1 = []
                for j in range(len(batch_descs_1)):
                    db_1 = batch_descs_1[j]
                    kp_1 = batch_kpts_1[j]
                    is_tag_j_1 = batch_is_tag_1[j]
                    assoc_scores_1 = batch_assoc_scores_1[j]
                    
                    if db_1.shape[0] > max_num_bbox_1:
                        raise RuntimeError('Why here?')
                    elif db_1.shape[0] == max_num_bbox_1:
                        num_adds_1.append(0)
                        continue

                    num_adds_1.append(max_num_bbox_1 - db_1.shape[0])
                    
                    zero_desc_append = torch.zeros(max_num_bbox_1 - db_1.shape[0], db_1.shape[1], db_1.shape[2], db_1.shape[3], dtype=db_1.dtype).cuda()
                    batch_descs_1[j] = torch.cat([db_1, zero_desc_append], dim=0)

                    zero_kpts_append = torch.zeros(max_num_bbox_1 - kp_1.shape[0], kp_1.shape[1], kp_1.shape[2], kp_1.shape[3], dtype=kp_1.dtype).cuda()
                    batch_kpts_1[j] = torch.cat([kp_1, zero_kpts_append], dim=0)

                    zero_is_tag_append = torch.zeros(max_num_bbox_1 - is_tag_j_1.shape[0], dtype=is_tag_j_1.dtype).cuda()
                    batch_is_tag_1[j] = torch.cat([is_tag_j_1, zero_is_tag_append], dim=0)

                    zero_assoc_scors_append = torch.zeros(max_num_bbox_1 - assoc_scores_1.shape[0], dtype=assoc_scores_1.dtype).cuda()
                    batch_assoc_scores_1[j] = torch.cat([assoc_scores_1, zero_assoc_scors_append], dim=0)
                
                data = {}
                data['descriptors'] = [batch_descs_0, batch_descs_1]
                data['keypoints'] = [batch_kpts_0, batch_kpts_1]
                data['is_tag'] = [batch_is_tag_0, batch_is_tag_1]
                data['assoc_scores'] = [batch_assoc_scores_0, batch_assoc_scores_1]
                data['M'] = batch_M
                data['return_matches'] = False
                data['return_losses'] = True

                res = model(data)

                #don't need to worry num_adds about separating because no dustbin
                optimizer.zero_grad()
                loss = torch.mean(res['losses'])
                loss.backward()
                optimizer.step()

                losses += loss.item()
                num_losses += 1

                batch_descs_0 = []
                batch_kpts_0 = []
                batch_is_tag_0 = []
                batch_assoc_scores_0 = []
                max_num_bbox_0 = 0

                batch_descs_1 = []
                batch_kpts_1 = []
                batch_is_tag_1 = []
                batch_assoc_scores_1 = []
                max_num_bbox_1 = 0

                batch_M = []

        scheduler.step()

        losses = losses / num_losses
        loss_array.append(losses)
        print('Epoch [{:4d}] | loss: {:6.4f}'.format(
                epoch, losses))

        save_epoch = epoch + 1
        if (save_epoch % opt.checkpoint == 0) or (save_epoch == opt.num_epochs):
            util.save_checkpoint(save_epoch, opt.checkpoint_dir, model)

        if (opt.val_feature_dir is not None) and ((save_epoch % opt.val == 0) or (save_epoch == opt.num_epochs)):
            print('beginning validation')
            model.eval()

            with torch.no_grad():
                val_losses = 0
                num_correct = 0
                num_incorrect = 0
                num_total = 0
                for batch in val_dataloader:
                    descs, kpts, width, height, is_tag, assoc_scores, M, _, _ = batch

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
                    data['return_losses'] = True

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

                    val_loss = torch.mean(res['losses'])
                    val_losses += val_loss.item()                

                val_losses = val_losses / len(val_dataloader)
                accuracy = num_correct / num_total

                val_loss_array.append(val_losses)
                val_acc_array.append(accuracy)
                val_epochs.append(epoch)

            util.plot_val_losses(opt.plot_loss_dir, np.array(val_loss_array), np.array(val_acc_array), np.array(val_epochs))

            if (best_loss is None) or (val_losses < best_loss):
                best_loss = val_losses
                util.save_checkpoint(save_epoch, opt.checkpoint_dir, model, accuracy=accuracy, loss=val_losses, is_best=True)

            model.train()
            print('done validation')

        if (save_epoch % opt.plot_loss == 0) or (save_epoch == opt.num_epochs):
            util.plot_losses(opt.plot_loss_dir, np.array(loss_array))


if __name__ == "__main__":
    opt = parse_args()

    train(opt)
