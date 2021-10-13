from Cocodataset.build_dataset import Build_Dataset
import paddle.nn as nn
import paddle
import os
from config import cfg
#from fasterrcnn.utils.visualization import *
import warnings
from models.rcnn import RCNN
import numpy as np
from tqdm import trange
from utils.utils import *
from paddle.vision.transforms import Normalize

normalize = Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        data_format='CHW')

N_CLASS = 20

warnings.filterwarnings('ignore')

def load_data(data_root, mode):
    if mode == 'train':
        npz = np.load('./dataset/train0.npz',allow_pickle=True)
        train_imgs = npz['train_imgs']
        train_img_info = npz['train_img_info']
        train_roi = npz['train_roi']
        train_cls = npz['train_cls']
        train_tbbox = npz['train_tbbox']
        train_imgs = normalize(train_imgs)
        train_imgs = paddle.to_tensor(train_imgs)
        Ntotal = train_imgs.shape[0]
        Ntrain = int(Ntotal * 0.8)
        pm = np.random.permutation(Ntotal)
        train_set = pm[:Ntrain]
        val_set = pm[Ntrain:]
        return train_imgs, train_roi, train_cls, train_tbbox, Ntotal, Ntrain, train_set, val_set
    elif mode =='test':
        npz = np.load('dataset/test.npz', allow_pickle = True)
        test_imgs = npz['test_imgs']
        test_img_info = npz['test_img_info']
        test_roi = npz['test_roi']
        test_orig_roi = npz['test_orig_roi']
        test_imgs = normalize(test_imgs)
        test_imgs = paddle.to_tensor(test_imgs)
        return test_imgs, test_img_info, test_roi, test_orig_roi



def train_batch(model, optimizer, img, rois, ridx, gt_cls, gt_tbbox, is_val=False):
    #print(ridx)       
    #print(rois)
    #[2, 224, 3, 224]  (4,1)  (69,4)  [48]  [48,4] 
    # print(img.shape)
    #img = paddle.transpose(img, perm=[0, 2, 1, 3])
    sc, r_bbox = model(img, rois, ridx )
    loss, loss_sc, loss_loc = model.calc_loss(sc, r_bbox, gt_cls, gt_tbbox)
    fl = loss.cuda().numpy()[0]
    fl_sc = loss_sc.cuda().numpy()[0]
    fl_loc = loss_loc.cuda().numpy()[0]

    if not is_val:
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
    return fl, fl_sc, fl_loc

def train_epoch(model, optimizer, run_set, is_val=False):
    I = 2
    B = 64
    POS = int(B * 0.25)
    NEG = B - POS
    Nimg = len(run_set)
    perm = np.random.permutation(Nimg)
    perm = run_set[perm]

    # if is_val:
        # rcnn.eval()
    # else:
        # rcnn.train()

    losses = []
    losses_sc = []
    losses_loc = []
    for i in trange(0, Nimg, I):
        #print(Nimg)
        lb = i
        rb = min(i+I, Nimg)
        if((rb-lb)!=2):
            continue
        torch_seg = paddle.to_tensor(perm[lb:rb])
        #print("lb:{}, rb:{}".format(lb,rb))
        #print(torch_seg)
        img = train_imgs[torch_seg].cuda()
        ridx = []
        glo_ids = []

        for j in range(lb, rb):
            info = train_img_info[perm[j]]
            pos_idx = info['pos_idx']
            neg_idx = info['neg_idx']
            ids = []

            if len(pos_idx) > 0:
                ids.append(np.random.choice(pos_idx, size=POS))
            if len(neg_idx) > 0:
                ids.append(np.random.choice(neg_idx, size=NEG))
            if len(ids) == 0:
                continue
            ids = np.concatenate(ids, axis=0)
            glo_ids.append(ids)
            ridx += [j-lb] * ids.shape[0]

        if len(ridx) == 0:
            continue
        glo_ids = np.concatenate(glo_ids, axis=0)
        ridx = np.array(ridx)

        rois = train_roi[glo_ids]
        gt_cls = paddle.to_tensor(train_cls[glo_ids]).cuda()
        gt_tbbox = paddle.to_tensor(train_tbbox[glo_ids]).cuda()
        #print(gt_cls)
        loss, loss_sc, loss_loc = train_batch(model, optimizer, img, rois, ridx, gt_cls, gt_tbbox, is_val=is_val)
        losses.append(loss)
        losses_sc.append(loss_sc)
        losses_loc.append(loss_loc)

    avg_loss = np.mean(losses)
    avg_loss_sc = np.mean(losses_sc)
    avg_loss_loc = np.mean(losses_loc)
    print(f'Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')

def start_training(n_epoch=5):
    rcnn = RCNN()
    optimizer = paddle.optimizer.Adam(learning_rate=0.0001,
        parameters=rcnn.parameters())
    for i in range(n_epoch):
        print(f'===========================================')
        print(f'[Training Epoch {i+1}]')
        train_epoch(rcnn, optimizer, train_set, False)
        print(f'[Validation Epoch {i+1}]')
        train_epoch(rcnn, optimizer, val_set, True)



def test_image(model, img, img_size, rois, orig_rois):
    nroi = rois.shape[0]
    ridx = np.zeros(nroi).astype(int)
    sc, tbbox = model(img, rois, ridx)
    sc = nn.functional.softmax(sc)
    sc = sc.cpu().numpy()
    tbbox = tbbox.cpu().numpy()
    bboxs = reg_to_bbox(img_size, tbbox, orig_rois)

    res_bbox = []
    res_cls = []

    for c in range(1, N_CLASS+1):
        c_sc = sc[:,c]
        c_bboxs = bboxs[:,c,:]

        boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0.3, score_threshold=0.6)
        res_bbox.extend(boxes)
        res_cls.extend([c] * len(boxes))

    if len(res_cls) == 0:
        for c in range(1, N_CLASS+1):
            c_sc = sc[:,c]
            c_bboxs = bboxs[:,c,:]

            boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0.3, score_threshold=0.3)
            res_bbox.extend(boxes)
            res_cls.extend([c] * len(boxes))
        res_bbox = res_bbox[:1]
        res_cls = res_cls[:1]

    print(res_cls)

    return np.array(res_bbox), np.array(res_cls)

def test_epoch():
    Nimg = test_imgs.shape[0]
    # rcnn.eval()
    Nc = Nimg // 10

    perm = np.random.permutation(Nimg)[:Nc]

    bbox_preds = []
    bbox_cls = []

    for i in range(Nimg):
        bbox_preds.append(np.ndarray((0, 4)))
        bbox_cls.append(np.ndarray((0, 1)))

    for i in trange(Nc):
        pi = perm[i]
        img = test_imgs[pi:pi+1]
        ridx = []
        glo_ids = []

        info = test_img_info[pi]
        img_size = info['img_size']
        idxs = info['idxs']

        idxs = np.array(idxs)
        rois = test_roi[idxs]
        orig_rois = test_orig_roi[idxs]

        res_bbox, res_cls = test_image(img, img_size, rois, orig_rois)
        bbox_preds[pi] = res_bbox
        bbox_cls[pi] = res_cls
        print(bbox_preds)
        print(bbox_cls)
#     evaluate.evaluate(bbox_preds, bbox_cls)

#     print('Test complete')


start_training(n_epoch=3)

test_epoch()

