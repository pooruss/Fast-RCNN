import numpy as np
import sys
import cv2
import paddle


def ss(img_root):
    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(16);
 
    # read image
    im = cv2.imread(img_root)
    # resize image
    #im = cv2.resize(im, (224, 224))    
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # set input image on which we will run segmentation
    # im = np.array(paddle.transpose(im, perm=[1,2,0]))
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    # run selective search segmentation on input image
    rects = paddle.to_tensor(ss.process())
    rects[:,2] = rects[:,0] + rects[:,2]
    rects[:,3] = rects[:,1] + rects[:,3]
    #print(rects)
    #print('Total Number of Region Proposals: {}'.format(len(rects)))
    # print(rects) # rects[numbers_of_RP, 4]
    return rects


def rel_bbox(size, bbox):
    bbox = bbox.astype(np.float32)
    print(bbox[:,1])
    print(size[0])
    bbox[:,0] /= size[1]
    bbox[:,1] /= size[0]
    # bbox[:,2] += 1
    bbox[:,2] /= size[1]
    # bbox[:,3] += 1
    bbox[:,3] /= size[0]
    return bbox

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:,2] - ex_rois[:,0] + 1.0
    ex_heights = ex_rois[:,3] - ex_rois[:,1] + 1.0
    ex_ctr_x = ex_rois[:,0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:,1] + 0.5 * ex_heights

    gt_widths = gt_rois[:,2] - gt_rois[:,0] + 1.0
    gt_heights = gt_rois[:,3] - gt_rois[:,1] + 1.0
    gt_ctr_x = gt_rois[:,0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:,1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.array([targets_dx, targets_dy, targets_dw, targets_dh]).T
    return targets

def calc_ious(ex_rois, gt_rois):
    ex_area = (1. + ex_rois[:,2] - ex_rois[:,0]) * (1. + ex_rois[:,3] - ex_rois[:,1])
    gt_area = (1. + gt_rois[:,2] - gt_rois[:,0]) * (1. + gt_rois[:,3] - gt_rois[:,1])
    #print(ex_area)
    #print(gt_area)
    area_sum = ex_area.reshape((-1, 1)) + gt_area.reshape((1, -1))
    lb = np.maximum(ex_rois[:,0].reshape((-1, 1)), gt_rois[:,0].reshape((1, -1)))
    rb = np.minimum(ex_rois[:,2].reshape((-1, 1)), gt_rois[:,2].reshape((1, -1)))
    tb = np.maximum(ex_rois[:,1].reshape((-1, 1)), gt_rois[:,1].reshape((1, -1)))
    ub = np.minimum(ex_rois[:,3].reshape((-1, 1)), gt_rois[:,3].reshape((1, -1)))

    width = np.maximum(1. + rb - lb, 0.)
    height = np.maximum(1. + ub - tb, 0.)
    area_i = width * height
    #print(area_sum)
    #print(1)
    #print(area_i)
    area_u = area_sum - area_i
    ious = area_i / area_u
    return ious

def reg_to_bbox(img_size, reg, box):
    img_width = img_size[0]
    img_height = img_size[1]
    bbox_width = box[:,2] - box[:,0] + 1.0
    bbox_height = box[:,3] - box[:,1] + 1.0
    bbox_ctr_x = box[:,0] + 0.5 * bbox_width
    bbox_ctr_y = box[:,1] + 0.5 * bbox_height

    bbox_width = bbox_width[:,np.newaxis]
    bbox_height = bbox_height[:,np.newaxis]
    bbox_ctr_x = bbox_ctr_x[:,np.newaxis]
    bbox_ctr_y = bbox_ctr_y[:,np.newaxis]

    out_ctr_x = reg[:,:,0] * bbox_width + bbox_ctr_x
    out_ctr_y = reg[:,:,1] * bbox_height + bbox_ctr_y

    out_width = bbox_width * np.exp(reg[:,:,2])
    out_height = bbox_height * np.exp(reg[:,:,3])

    return np.array([
        np.maximum(0, out_ctr_x - 0.5 * out_width),
        np.maximum(0, out_ctr_y - 0.5 * out_height),
        np.minimum(img_width, out_ctr_x + 0.5 * out_width),
        np.minimum(img_height, out_ctr_y + 0.5 * out_height)
    ]).transpose([1, 2, 0])

def non_maximum_suppression(sc, bboxs, iou_threshold=0.7, score_threshold=0.6):
    nroi = sc.shape[0]
    idx = np.argsort(sc)[::-1]
    rb = 0
    while rb < nroi and sc[idx[rb]] >= score_threshold:
        rb += 1
    if rb == 0:
        return []
    idx = idx[:rb]
    sc = sc[idx]
    bboxs = bboxs[idx,:]
    ious = calc_ious(bboxs, bboxs)

    res = []
    for i in range(rb):
        if i == 0 or ious[i, :i].max() < iou_threshold:
            res.append(bboxs[i])

    return res

