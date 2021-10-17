from utils.cocodata import CocoDataSet, DataTest
from paddle.io import DataLoader
from utils.utils import *
import paddle

class Build_Dataset():
    def __init__(self):
        self.mode = None

    def build_train_data(self, data_root, train_pipelines, mode='train'):
        # data_root文件夹中必须包含cocojson文件，如训练为 train.json,暂时不开接口
        datasets_train = CocoDataSet(data_root, train_pipelines, mode=mode)
        # RP
        rects = []
        self.mode = mode
        # print("cat2class:{}".format(datasets_train.cat2class) #cat2class:{1: 'green', 2: 'red'}
        # )
        # print("cat2ids:{}".format(datasets_train.cat_ids) #[1, 2]
        # )
        # print("img_ids:{}".format(datasets_train.img_ids) #20190000001
        # )
        # print("img_infos:{}".format(datasets_train.img_infos[0]) #{'file_name': 'red189.jpg', 'height': 376, 'width': 672, 'id': 20190000069, 'img_root': '/home/aistudio/work/fast-rcnn/dataset/coco/val2017/red189.jpg'}]
        # )
        # print("parestrain:{}".format(datasets_train.parse_train(0)) #gt_labels, img:tensor[3,224,224], gt_bboxes:[1,4],img_shape(3,224,224), 
        # )#'pad_shape': (224, 224, 3), 'cat_weight': {1: 1.0, 2: 1.0}, 'ori_shape': (376, 672, 3), 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}, 'img_info': {'gt_labels', 'img', 'gt_bboxes'}, 'img_meta': {'img_shape', 'img_root', 'gt_bboxes', 'pad_shape', 'gt_labels', 'cat_weight', 'ori_shape', 'img_norm_cfg'}
        #print(datasets_train.__len__())
        a = 0
        dataset_small = []
        for i in datasets_train:  
            if a < 10:
                rects.append(ss(i['img_root']))
                a = a + 1
                dataset_small.append(i)
                print(a)
            else:
                break
        return dataset_small, rects

    def build_train_data_0_3(self, data_root, train_pipelines, mode='train'):
        # data_root文件夹中必须包含cocojson文件，如训练为 train.json,暂时不开接口
        #print(train_pipelines)
        datasets_train = CocoDataSet(data_root, train_pipelines, mode=mode)
        # RP
        rects = []
        self.mode = mode
        a = 0
        dataset_small = []
        for i in datasets_train:  
            if a < 3000:
                rects.append(ss(i['img_root']))
                dataset_small.append(i)
                print(a)
                a = a + 1
            else:
                break
        return dataset_small, rects
        
    def build_train_data_3_6(self, data_root, train_pipelines, mode='train'):
        # data_root文件夹中必须包含cocojson文件，如训练为 train.json,暂时不开接口
        #print(train_pipelines)
        datasets_train = CocoDataSet(data_root, train_pipelines, mode=mode)
        # RP
        rects = []
        self.mode = mode
        a = 0
        dataset_small = []
        for i in datasets_train:  
            if (a < 6000) and (a >= 3000):
                rects.append(ss(i['img_root'])) 
                dataset_small.append(i)
                print(a)
                a = a + 1
            elif a >= 6000:
                break
            else:
                a = a + 1
        return dataset_small, rects

    def build_train_data_6_9(self, data_root, train_pipelines, mode='train'):
        # data_root文件夹中必须包含cocojson文件，如训练为 train.json,暂时不开接口
        datasets_train = CocoDataSet(data_root, train_pipelines, mode=mode)
        # RP
        rects = []
        self.mode = mode
        a = 0
        dataset_small = []
        for i in datasets_train:  
            if (a < 9000) and (a >= 6000):
                rects.append(ss(i['img_root'])) 
                dataset_small.append(i)
                print(a)
                a = a + 1
            elif a >= 9000:
                break
            else:
                a = a + 1
        return dataset_small, rects
    
    def build_train_data_9_11(self, data_root, train_pipelines, mode='train'):
        # data_root文件夹中必须包含cocojson文件，如训练为 train.json,暂时不开接口
        datasets_train = CocoDataSet(data_root, train_pipelines, mode=mode)
        # RP
        rects = []
        self.mode = mode
        a = 0
        dataset_small = []
        for i in datasets_train:  
            if (a < 11539) and (a >= 9000):
                rects.append(ss(i['img_root'])) 
                dataset_small.append(i)
                print(a)
            elif a >= 11539:
                break
            else:
                a = a + 1
        return dataset_small, rects
    

    def build_val_data(self, data_root, val_pipelines, mode='val'):
        datasets_val = CocoDataSet(data_root, val_pipelines, mode=mode)
        self.mode = mode
        return datasets_val

    def build_test_data(self, data_root, test_pipelines, mode='test'):
        datasets_test = CocoDataSet(data_root, test_pipelines, mode=mode)
        self.mode = mode
        rects = []
        datasets_test_small = []
        a = 0
        for i in datasets_test:  
            if a < 10:
                rects.append(ss(i['img_root']))
                datasets_test_small.append(i)
                a = a + 1
                print(a)
            else:
                break
        return datasets_test_small, rects

    def build_img_data(self, test_pipelines, mode='test'):
        # 构建一张图测试的类初始化
        data_img_test = DataTest(test_pipelines, mode=mode)
        self.mode = mode
        return data_img_test

    def collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        #print(batch)
        num_batch = len(batch)
        img_info_keys = batch[0]['img_info']
        img_meta_keys = batch[0]['img_meta']
        data = {}
        for key in img_info_keys:
            key_lst = []
            for i in range(num_batch):
                key_lst.append(batch[i][key])
            if key == 'img':
                key_lst = paddle.stack(key_lst, 0)
            data[key] = key_lst
        img_meta = []
        for i in range(num_batch):
            key_dict = {}
            for key in img_meta_keys:
                key_dict[key] = batch[i][key]
            img_meta.append(key_dict)

        return [data, img_meta]

    def build_dataloader(self, datasets, **kwargs):

        batch_size = kwargs.get('batch_size', 1)
        num_worksers = kwargs.get('num_workers', 2)
        shuffle = kwargs.get('shffle', True)
        drop_last = kwargs.get('drop_last', True)

        dataloader = DataLoader(datasets,
                                batch_size=batch_size,
                                num_workers=num_worksers,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                collate_fn=self.collate
                                )

        return dataloader



if __name__ == '__main__':
    test_pipeline = [
        dict(type='Resize', img_scale=(224, 224)),
        dict(type='Normalize'),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])

    ]

    train_pipeline = [
        dict(type='LoadImageFromFile'),  # 载入图像
        dict(type='LoadAnnotations', with_bbox=True),  # 载入annotations

        dict(type='Resize', img_scale=(224, 224)),  #(512,512)有的图像不行！
        #dict(type='RandomFlip', flip_ratio=0.5),
        #dict(type='RandomFlip', direction='vertical', flip_ratio=0.5),

        #dict(type='RandomCropResize', crop_size=(426, 426), crop_ratio=1.1),

        # 加载数据处理模块#

        dict(type='Normalize'),
        #dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
             img_meta_keys=['cat2class']),  # 在results中需要提取的结果
    ]

    dataset_json = True
    if dataset_json:
        # Cocojson数据处理过程
        root = r'/home/aistudio/work/fast-rcnn/dataset/coco/train2017small'
        BD = Build_Dataset()
        dataset = BD.build_train_data(root, train_pipeline)
        dataloader = BD.build_dataloader(dataset)
        # for data, img_meta in dataloader:
        #     print(11111111111111111111)
            # print("shape:{}".format(data['img'].shape))
            # print(img_meta)
        print("data loaded!")
    else:
        img_root = r'/home/aistudio/work/fast-rcnn/dataset/coco/test/val2017/green332.jpg'
        BD = Build_Dataset()
        dataset_img = BD.build_img_data(test_pipeline)
        result_img = dataset_img.data_test(img_root)
        print(result_img)
