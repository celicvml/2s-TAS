'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
from grid_sampler import GridSampler, TimeWarpLayer

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path_RGB, features_path_FLOW, sample_rate):
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path_RGB = features_path_RGB
        self.features_path_FLOW = features_path_FLOW
        self.sample_rate = sample_rate

        self.timewarp_layer = TimeWarpLayer()

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]    # 读取视频文件名列表
        file_ptr.close()
        # 构建标签文件路径
        self.gts = [self.gt_path + vid for vid in self.list_of_examples]
        # 构建RGB特征文件路径
        self.features_RGB = [self.features_path_RGB + vid.split('.')[0] + '.npy' for vid in self.list_of_examples]
        # 构建Optical Flow特征文件路径
        self.features_FLOW = [self.features_path_FLOW + vid.split('.')[0] + '.npy' for vid in self.list_of_examples]
        self.my_shuffle()

    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features_RGB)
        random.seed(randnum)
        random.shuffle(self.features_FLOW)

    def warp_video(self, batch_input_tensor, batch_target_tensor):
        '''
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        '''
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid, mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()  # obtain the same shape

        return warped_batch_input_tensor, warped_batch_target_tensor

    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features

        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))

    # 读取下一批次的数据，并将数据整理成适合模型训练的格式。
    def next_batch(self, batch_size, if_warp=False): # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features_RGB = self.features_RGB[self.index:self.index + batch_size]
        batch_features_FLOW = self.features_FLOW[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input_RGB = []
        batch_input_FLOW = []
        batch_target_RGB = []
        batch_target_FLOW = []
        for idx, vid in enumerate(batch):
            features_RGB = np.load(batch_features_RGB[idx]).T
            features_FLOW = np.load(batch_features_FLOW[idx])
            file_ptr = open(batch_gts[idx], 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes_RGB = np.zeros(min(np.shape(features_RGB)[1], len(content)))
            classes_FLOW = np.zeros(min(np.shape(features_FLOW)[1], len(content)))
            for i in range(len(classes_RGB)):
                classes_RGB[i] = self.actions_dict[content[i]]

            for i in range(len(classes_FLOW)):
                classes_FLOW[i] = self.actions_dict[content[i]]

            feature_RGB = features_RGB[:, ::self.sample_rate]
            feature_FLOW = features_FLOW[:, ::self.sample_rate]
            target_RGB = classes_RGB[::self.sample_rate]
            target_FLOW = classes_FLOW[::self.sample_rate]
            batch_input_RGB.append(feature_RGB)
            batch_input_FLOW.append(feature_FLOW)
            batch_target_RGB.append(target_RGB)
            batch_target_FLOW.append(target_FLOW)

        length_of_sequences_RGB = list(map(len, batch_target_RGB))
        length_of_sequences_FLOW = list(map(len, batch_target_FLOW))

        batch_input_tensor_RGB = torch.zeros(len(batch_input_RGB), np.shape(batch_input_RGB[0])[0], max(length_of_sequences_RGB), dtype=torch.float)  # bs, C_in,
        batch_input_tensor_FLOW = torch.zeros(len(batch_input_FLOW), np.shape(batch_input_FLOW[0])[0],max(length_of_sequences_FLOW), dtype=torch.float)
        batch_target_tensor_RGB = torch.ones(len(batch_input_RGB), max(length_of_sequences_RGB), dtype=torch.long) * (-100)
        batch_target_tensor_FLOW = torch.ones(len(batch_input_FLOW), max(length_of_sequences_FLOW), dtype=torch.long) * (-100)

        mask_RGB = torch.zeros(len(batch_input_RGB), self.num_classes, max(length_of_sequences_RGB), dtype=torch.float)
        mask_FLOW = torch.zeros(len(batch_input_FLOW), self.num_classes, max(length_of_sequences_FLOW), dtype=torch.float)

        for i in range(len(batch_input_RGB)):
            if if_warp:
                warped_input_RGB, warped_target_RGB = self.warp_video(torch.from_numpy(batch_input_RGB[i]).unsqueeze(0), torch.from_numpy(batch_target_RGB[i]).unsqueeze(0))
                batch_input_tensor_RGB[i, :, :np.shape(batch_input_RGB[i])[1]], batch_target_tensor_RGB[i, :np.shape(batch_target_RGB[i])[0]] = warped_input_RGB.squeeze(0), warped_target_RGB.squeeze(0)
            else:
                batch_input_tensor_RGB[i, :, :np.shape(batch_input_RGB[i])[1]] = torch.from_numpy(batch_input_RGB[i])
                batch_target_tensor_RGB[i, :np.shape(batch_target_RGB[i])[0]] = torch.from_numpy(batch_target_RGB[i])
            mask_RGB[i, :, :np.shape(batch_target_RGB[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target_RGB[i])[0])

        for i in range(len(batch_input_FLOW)):
            if if_warp:
                warped_input_FLOW, warped_target_FLOW = self.warp_video(torch.from_numpy(batch_input_FLOW[i]).unsqueeze(0),torch.from_numpy(batch_target_FLOW[i]).unsqueeze(0))
                batch_input_tensor_FLOW[i, :, :np.shape(batch_input_FLOW[i])[1]], batch_target_tensor_FLOW[i,:np.shape(batch_target_FLOW[i])[0]] = warped_input_FLOW.squeeze(0), warped_target_FLOW.squeeze(0)
            else:
                batch_input_tensor_FLOW[i, :, :np.shape(batch_input_FLOW[i])[1]] = torch.from_numpy(batch_input_FLOW[i])
                batch_target_tensor_FLOW[i, :np.shape(batch_target_FLOW[i])[0]] = torch.from_numpy(batch_target_FLOW[i])
            mask_FLOW[i, :, :np.shape(batch_target_FLOW[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target_FLOW[i])[0])

        return batch_input_tensor_RGB, batch_input_tensor_FLOW, batch_target_tensor_RGB, batch_target_tensor_FLOW, mask_RGB, mask_FLOW, batch


if __name__ == '__main__':
    pass