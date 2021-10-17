# Fast-RCNN

1. 参考代码：

   https://github.com/rbgirshick/fast-rcnn/tree/master（caffe），

   https://github.com/gary1346aa/Fast-RCNN-Object-Detection-Pytorch/blob/master/README.ipynb（pytorch）. 

   

2. 对齐打卡

   ​	由于所给原参考代码使用的是caffe框架，复现难度有所增加，考虑到相关打卡和对齐方法都是针对torch和paddle，目前也还未有caffe与paddle的对应接口教程，个人对caffe也不熟悉，所以没有去跑通caffe代码。**目前网上包括github、paper with code等能找到基于pytorch复现fast-rcnn只有个别不完整不规范版本**，**所以此复现无法涉及五个打卡点diff的计算。**另外，对于个人自己找的相对较完整，但不规范代码https://github.com/gary1346aa/Fast-RCNN-Object-Detection-Pytorch/blob/master/README.ipynb，群里老师也未给予确定性的可根据此代码作为参考打卡点代码的回复。

   ​	个人根据阅读论文、分析caffe源码和相关torch代码，在paddle上结构层面上尽量对齐：

   **模型结构**与**损失函数**：

   ```
   aistudio@jupyter-585642-2443378:~/work/fast-rcnn/models$ python rcnn.py
   W1013 22:16:52.692131 12378 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
   W1013 22:16:52.697325 12378 device_context.cc:422] device: 0, cuDNN Version: 7.6.
   /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:641: UserWarning: When training, we now always track global mean and variance.
     "When training, we now always track global mean and variance.")
   RCNN(
     (seq): VGG(
       (extract_feature): Sequential(
         (0): Conv2D(3, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (1): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
         (2): ReLU()
         (3): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (4): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
         (5): ReLU()
         (6): MaxPool2D(kernel_size=2, stride=2, padding=0)
         (7): Conv2D(64, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (8): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
         (9): ReLU()
         (10): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (11): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
         (12): ReLU()
         (13): MaxPool2D(kernel_size=2, stride=2, padding=0)
         (14): Conv2D(128, 256, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (15): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
         (16): ReLU()
         (17): Conv2D(256, 256, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (18): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
         (19): ReLU()
         (20): Conv2D(256, 256, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (21): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
         (22): ReLU()
         (23): MaxPool2D(kernel_size=2, stride=2, padding=0)
         (24): Conv2D(256, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (25): BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
         (26): ReLU()
         (27): Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (28): BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
         (29): ReLU()
         (30): Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (31): BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
         (32): ReLU()
         (33): MaxPool2D(kernel_size=2, stride=2, padding=0)
         (34): Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (35): BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
         (36): ReLU()
         (37): Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (38): BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
         (39): ReLU()
         (40): Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
         (41): BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
         (42): ReLU()
         (43): MaxPool2D(kernel_size=2, stride=2, padding=0)
       )
     )
     (roipool): SlowROIPool(
       (maxpool): AdaptiveMaxPool2D(output_size=(7, 7), return_mask=False)
     )
     (feature): Sequential(
       (0): Linear(in_features=25088, out_features=4096, dtype=float32)
       (1): ReLU()
       (2): Dropout(p=0.5, axis=None, mode=upscale_in_train)
       (3): Linear(in_features=4096, out_features=4096, dtype=float32)
       (4): ReLU()
       (5): Dropout(p=0.5, axis=None, mode=upscale_in_train)
     )
     (cls_score): Linear(in_features=4096, out_features=21, dtype=float32)
     (bbox): Linear(in_features=4096, out_features=84, dtype=float32)
     (cel): CrossEntropyLoss()
     (sl1): SmoothL1Loss()
   )
   ```

   **验证/测试集数据读取对齐**：

   ​	未找到相关torch版本的数据读取，无法对齐。

   **评估指标对齐：**

   ​	未找到相关torch版本的评估指标实现，无法对齐。

   **反向对齐与训练对齐：**

   ​	未找到相关torch版本的完整模型代码，无法对齐。

   

3. selectivesearch候选框：

   ​	由于国内暂时无法从dropbox下载voc数据集的selectivesearch数据，使用cv2的switchToSelectiveSearchFast()生成rois，并对voc数据进行预处理，结果存储为.npz文件。训练时直接读取npz文件作为训练/验证/测试数据。目前各基于caffe复现的代码都是直接从dropbox直接下载selectivsearch.mat作为候选框构建训练数据。



4. 总结：

​	目前模型已可进行正常完整的训练，使用的是voc2012trainval数据，自己生成的selectivsearch候选框。尚未对训练结果进行测试评估。也由于没有某份torch版本源代码能提供一个参考去对齐（个人自己找的相对完整代码未得到老师的许可），所以对自己的复现工作进度比较模糊，质量可能也不太令人满意。



待完善：数据的读取方式、selectivesearch数据的获取、模型各参数对齐调整、权重初始化、结果评估evaluate.py、

