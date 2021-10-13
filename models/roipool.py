import paddle.nn as nn
import paddle
import numpy
import numpy as np


class SlowROIPool(nn.Layer):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2D(output_size)
        self.size = output_size

    def forward(self, images, rois, roi_idx):
        n = rois.shape[0]
        # print(n)
        # print(rois.shape) #(1,4) 
        # print(roi_idx[0])
        h = images.shape[2]  #images:[1,512,14,14]
        w = images.shape[3]
        x1 = rois[:,0]
        y1 = rois[:,1]
        x2 = rois[:,2]
        y2 = rois[:,3]

        x1 = np.floor(x1 * w).astype(int)
        x2 = np.ceil(x2 * w).astype(int)
        y1 = np.floor(y1 * h).astype(int)
        y2 = np.ceil(y2 * h).astype(int)
        
        # print("x1:{}".format(x1))
        # print("y1:{}".format(y1))
        # print("x2:{}".format(x2))
        # print("y2:{}".format(y2))
        
        res = []
        for i in range(n):
            #print(images[0])  #(512,14,14)
            #print(roi_idx[i]) #0
            img = images[int(roi_idx[i])].unsqueeze(0)
            # if y1[i]>=56:
            #     print("y1:{}".format(y1[i]))
            # if x1[i]>=56 or y1[i]>=56 or x2[i]>=56 or y2[i]>=56:
            #     print("x1:{}".format(x1[i]))
            #     print("y1:{}".format(y1[i]))
            #     print("x2:{}".format(x2[i]))
            #     print("y2:{}".format(y2[i]))
            # if y2[i]>=56:
            #     print("y2:{}".format(y2[i]))
            # if x2[i]>=56:
            #     print("x2:{}".format(x2[i]))
            img = img[:, :, int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
            img = self.maxpool(img)
            res.append(img)
        res = paddle.concat(res, axis=0)
        #print(res) #(2*48,512,7,7)
        #res = paddle.reshape(res,[-1])
        #print(res.shape) #(1,512,7,7)
        return res  #(1*512*7*7)