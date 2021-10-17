import numpy as np
import pickle
from PIL import Image
import tqdm
from tqdm import trange
from utils.utils import *
from utils.build_dataset import Build_Dataset, DataTest
from config.config import cfg

root = cfg['data_cfg']['data_root']

BD = Build_Dataset()

def build_data(cfg):
    #print("root:{}, train_pipeline:{}, kwarge_loader:{}, BD:{}".format(root, train_pipeline, kwargs_loader, BD))
    train_pipeline = cfg['data_cfg']['train_pipeline0']
    dataset,rects = BD.build_train_data(root, train_pipeline)
    return dataset, rects

def build_data_0_3(cfg):
    #print("root:{}, train_pipeline:{}, kwarge_loader:{}, BD:{}".format(root, train_pipeline, kwargs_loader, BD))
    train_pipeline = cfg['data_cfg']['train_pipeline0']
    dataset,rects = BD.build_train_data_0_3(root, train_pipeline)
    return dataset, rects

def build_data_3_6(cfg):
    #print("root:{}, train_pipeline:{}, kwarge_loader:{}, BD:{}".format(root, train_pipeline, kwargs_loader, BD))
    train_pipeline = cfg['data_cfg']['train_pipeline1']
    dataset,rects = BD.build_train_data_3_6(root, train_pipeline)
    return dataset, rects

def build_data_6_9(cfg):
    #print("root:{}, train_pipeline:{}, kwarge_loader:{}, BD:{}".format(root, train_pipeline, kwargs_loader, BD))
    train_pipeline = cfg['data_cfg']['train_pipeline2']
    dataset,rects = BD.build_train_data_6_9(root, train_pipeline)
    return dataset, rects

def build_data_9_11(cfg):
    #print("root:{}, train_pipeline:{}, kwarge_loader:{}, BD:{}".format(root, train_pipeline, kwargs_loader, BD))
    train_pipeline = cfg['data_cfg']['train_pipeline3']
    dataset,rects = BD.build_train_data_9_11(root, train_pipeline)
    return dataset, rects

def build_test_data(cfg):
    # 为构建单张数据及进行实列化
    root = cfg['data_cfg']['test_data_root']
    test_pipeline = cfg['data_cfg']['test_pipeline']
    datasets_test = BD.build_test_data(root, test_pipeline)
    return datasets_test

### Train
def traindt_to_npz(cfg, flag):
    train_imgs = []
    train_img_info = []
    train_roi = []
    train_cls = []
    train_tbbox = []

    if (flag==0):
        data, rects = build_data_0_3(cfg)
    elif (flag==1):
        data, rects = build_data_3_6(cfg)
    elif (flag==2):
        data, rects = build_data_6_9(cfg)
    elif (flag==3):
        data, rects = build_data_9_11(cfg)  
    else:
        data, rects = build_data(cfg)    
    N_train = len(data)
    #print(N_train)

    N_train = len(data)
    #print(N_train)
    for i in trange(N_train):
        img_path = data[i]['img_root']
        gt_boxs = data[i]['gt_bboxes'].numpy()
        #print(gt_boxs)
        # print(img_path)
        gt_classes = data[i]['gt_labels']
        bboxs = rects[i].numpy()
        nroi = len(bboxs)
        img = data[i]['img']
        img_size = data[i]['ori_shape']
        rbboxs = rel_bbox(img_size, bboxs)
        ious = calc_ious(bboxs, gt_boxs)
        #print(ious)
        max_ious = ious.max(axis=1)
        max_idx = ious.argmax(axis=1)
        #print(max_ious)
        tbbox = bbox_transform(bboxs, gt_boxs[max_idx])
        pos_idx = []
        neg_idx = []
        # print(bboxs)
        # print(1)
        # print(gt_boxs[max_idx])
        # print(1)
        # print(tbbox)  #(1,4)
        for j in range(nroi):
            if max_ious[j] < 0.1:
                continue

            gid = len(train_roi)
            train_roi.append(rbboxs[j])
            train_tbbox.append(tbbox[j])

            if max_ious[j] >= 0.5:
                pos_idx.append(gid)
                a = max_idx[j]
                #print(gt_classes[int(a)])
                train_cls.append(gt_classes[int(a)])
                #print(max_ious[j])
            else:
                neg_idx.append(gid)
                train_cls.append(0)
        pos_idx = np.array(pos_idx)
        neg_idx = np.array(neg_idx)
        train_imgs.append(img)
        train_img_info.append({
            'img_size': img_size,
            'pos_idx': pos_idx,
            'neg_idx': neg_idx,
        })
        # print(len(pos_idx), len(neg_idx))
    train_imgs = np.array(train_imgs)
    train_img_info = np.array(train_img_info)
    train_roi = np.array(train_roi)
    train_cls = np.array(train_cls).astype(np.int)
    train_tbbox = np.array(train_tbbox).astype(np.float32)
    np.savez(open('./dataset/train{}.npz'.format(flag), 'wb'), 
            train_imgs=train_imgs, train_img_info=train_img_info,
            train_roi=train_roi, train_cls=train_cls, train_tbbox=train_tbbox)

    # print("train_imgs.shape:{}".format(train_imgs.shape))
    # print("train_cls:{}".format(train_cls))
    # print("train_roi:{}".format(train_roi))
    # print("train_tbbox:{}".format(train_tbbox))

## Test
def testdt_to_npz(cfg):
    #测试集测试，待debug
    test_data, rects = build_test_data(cfg)
    
    #一张图测试
    # test_pipeline = cfg['data_cfg']['test_pipeline']
    # test_data = DataTest(test_pipeline)
    # test_data = test_data.data_test('/home/aistudio/work/fast-rcnn/dataset/coco/val2017/green000.jpg')
    
    N_test = len(test_data)

    test_imgs = []
    test_img_info = []
    test_roi = []
    test_orig_roi = []
    #print(test_data.parse_test(0))
    for i in trange(N_test):
        bboxs = rects[i].numpy()
        #print(bboxs)
        nroi = len(bboxs)
        img = test_data[i]['img']
        img_size = test_data[i]['ori_shape']
        rbboxs = rel_bbox(img_size, bboxs)
        idxs = []

        for j in range(nroi):
            gid = len(test_roi)
            test_roi.append(rbboxs[j])
            test_orig_roi.append(bboxs[j])
            idxs.append(gid)

        idxs = np.array(idxs)
        test_imgs.append(img)
        test_img_info.append({
            'img_size': img_size,
            'idxs': idxs
        })
        # print(len(idxs))

    test_imgs = np.array(test_imgs)
    test_img_info = np.array(test_img_info)
    test_roi = np.array(test_roi)
    test_orig_roi = np.array(test_orig_roi)

    # print(test_imgs.shape)
    # print(test_roi.shape)
    # print(test_orig_roi.shape)

    np.savez(open('./dataset/test.npz', 'wb'), 
            test_imgs=test_imgs, test_img_info=test_img_info, test_roi=test_roi, test_orig_roi=test_orig_roi)

# test
if __name__ == "__main__":
    # for Train
    #traindt_to_npz(cfg,3)
    np0 = np.load('./dataset/train0.npz')
    np1 = np.load('./dataset/train1.npz')
    np2 = np.load('./dataset/train2.npz')
    np3 = np.load('./dataset/train3.npz')
    npall = [np0,np1,np2,np3]
    np.savez('./dataset/train.npz',npall)

    # #for Test
    # testdt_to_npz(cfg)