import paddle.nn as nn
import paddle
import numpy
import numpy as np
from rawvgg import VGG
from roipool import SlowROIPool

N_CLASS = 20

class RCNN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.seq = VGG(num_classes = N_CLASS)
        # self.roipool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.roipool = SlowROIPool(output_size=(7, 7))
        self.feature = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        _x = np.zeros((1,3,224,224))
        _x = paddle.to_tensor(_x,dtype=paddle.get_default_dtype())
        _r = np.array([[0., 0., 1., 1.]],dtype=int)
        _ri = np.array([0])
        _x = self.seq(_x)
        _x = self.roipool(_x, _r, _ri)
        _x = paddle.reshape(_x, [1,-1])
        _x = self.feature(_x)
        # print("sss:{}".format(_x))  [1,4096]
        feature_dim = _x.shape[1]
        self.cls_score = nn.Linear(feature_dim, N_CLASS+1)
        self.bbox = nn.Linear(feature_dim, 4*(N_CLASS+1))
        
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, inp, rois, ridx):
        res = inp
        res = self.seq(res)
        # print(res) # [2,512,14,14]
        res = self.roipool(res, rois, ridx)
        res = res.detach()
        res = paddle.reshape(res, [0, -1])
        # print(res.shape) #[1204224, 1]
        feat = self.feature(res)

        cls_score = self.cls_score(feat)
        bbox = paddle.reshape(self.bbox(feat),[-1,N_CLASS+1,4])
        return cls_score, bbox

    def calc_loss(self, probs, bbox, labels, gt_bbox):
        loss_sc = self.cel(probs, labels)
        #print(labels)
        lbl = paddle.reshape(labels, [-1,1,1])
        #print(labels.shape[0])
        lbl = paddle.expand(lbl, shape = [labels.shape[0], 1, 4])
        mask = (labels!=0)
        mask = paddle.reshape(mask, [-1,1])
        mask = paddle.expand(mask, shape = [labels.shape[0], 4])
        
        a = paddle.gather(bbox,labels,1)
        for i in range(labels.shape[0]):
            if i == 0:
                a_correct = paddle.reshape(a[i][i],[-1,4])
                continue
            a_correct = paddle.concat( [a_correct, paddle.reshape(a[i][i],[-1,4])], axis=0)
        # print("a_correct:{}".format(a_correct.shape))
        # a = a.squeeze(1)
        # print("bbox:{}".format(bbox.shape))
        # print("lbl:{}".format(lbl.shape))
        # print("a:{}".format(a.shape))
        # print("mask:{}".format(mask.shape))
        # print("gtbbox:{}".format(gt_bbox.shape))
        # b = paddle.transpose(a, perm=[1,0,2])
        c = a_correct * mask
        #print(mask)
        loss_loc = self.sl1(c, gt_bbox * mask)
        lmb = 1.0
        loss = loss_sc + lmb * loss_loc
        return loss, loss_sc, loss_loc


# test
if __name__ == "__main__":
    rcnn = RCNN()
    print(rcnn)