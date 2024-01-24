from tool import DataProcessing as DP
from tool import ConfigSemanticKITTI as cfg
from os.path import join
import numpy as np
import os, pickle
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

class SemanticKITTI(Dataset):
    def __init__(self, test_id, labeled_point, gen_pseudo, retrain):
        self.name = 'SemanticKITTI'
        self.dataset_path = '/data/dataset/SemanticKitti/dataset/sequences_0.06'
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])
        self.gen_pseudo = gen_pseudo
        self.retrain = retrain
        
        
        self.use_val = True  # whether use validation set or not
        self.val_split = '08'

        self.seq_list = np.sort(os.listdir(self.dataset_path))     
        self.test_scan_number = str(test_id)        # 只有这个序列进行测试（并不是用11~21所有的序列进行测试）
        self.train_list, self.val_list, self.test_list = DP.get_file_list(self.dataset_path,            # 默认序列08作为验证集
                                                           self.test_scan_number,
                                                           self.gen_pseudo)
        
        self.train_list = DP.shuffle_list(self.train_list)
        self.val_list = DP.shuffle_list(self.val_list)
    
    


        self.possibility = []
        self.min_possibility = []

        self.num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                       240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                       9833174, 129609852, 4506626, 1168181])
        
        cfg.class_weights = DP.get_class_weights(self.num_per_class, 'sqrt')
        
        
        if '%' in labeled_point:
            r = float(labeled_point[:-1]) / 100
            self.num_with_anno_per_batch = max(int(cfg.num_points * r), 1)
        else:
            self.num_with_anno_per_batch = cfg.num_classes
            
        self.labeled_point = labeled_point    
                

        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        
        
    def __getitem__(self, index):
        pass    
    
    
    def __len__(self):
        pass



class SemanticKITTILoader(SemanticKITTI):
    def __init__(self, test_id, labeled_point="0.1%", gen_pseudo=False, retrain=False, split='training'):
        super().__init__(test_id, labeled_point, gen_pseudo, retrain)
        
        self.split = split
        
        if split == 'training':
            self.num_per_epoch = int(len(self.train_list) / cfg.batch_size) * cfg.batch_size
            self.path_list = self.train_list
        elif split == 'validation':
            self.num_per_epoch = int(len(self.val_list) / cfg.val_batch_size) * cfg.val_batch_size
            cfg.val_steps = int(len(self.val_list) / cfg.batch_size)
            self.path_list = self.val_list
        elif split == 'test':
            self.num_per_epoch = int(len(self.test_list) / cfg.val_batch_size) * cfg.val_batch_size * 4
            self.path_list = self.test_list
            for test_file_name in self.path_list:
                points = np.load(test_file_name)
                self.possibility += [np.random.rand(points.shape[0]) * 1e-3]
                self.min_possibility += [float(np.min(self.possibility[-1]))]
            



    def __len__(self):
        return self.num_per_epoch


    def __getitem__(self, item):

        selected_pc, selected_labels, selected_idx, cloud_ind, xyz_with_anno, labels_with_anno = self.spatially_regular_gen(item)
        return selected_pc, selected_labels, selected_idx, cloud_ind, xyz_with_anno, labels_with_anno



    def spatially_regular_gen(self, item):
        # Generator loop
        while(1):
            
            if self.split != 'test':
                cloud_ind = item
                pc_path = self.path_list[cloud_ind]
                pc, tree, labels = self.get_data(pc_path)           # 返回的label已经被处理好了，大多数点的label为0，只有少部分有label 这里返回的pc某个sequence下的某帧
                # crop a small point cloud
                pick_idx = np.random.choice(len(pc), 1)             # 在该帧下的所有点中选出一个索引，该索引对应的点作为中心点取knn
                selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)       # 这里选出中心点附近的45056个点，点数固定，方便后续传入网络
                
                
                if self.split == 'training':        # training 数据没有使用概率取点的方式，在semantickitti数据集中训练数据是按顺序选取的
                    unique_label_value = np.unique(selected_labels)
                    if len(unique_label_value) <= 1: 
                        continue
                    else:
                        # ================================================================== #
                        #            Keep the same number of labeled points per batch        #
                        # ================================================================== #
                        idx_with_anno = np.where(selected_labels != self.ignored_labels[0])[0]
                        num_with_anno = len(idx_with_anno)
                        if num_with_anno > self.num_with_anno_per_batch:
                            idx_with_anno = np.random.choice(idx_with_anno, self.num_with_anno_per_batch,
                                                                replace=False)
                        elif num_with_anno < self.num_with_anno_per_batch:
                            dup_idx = np.random.choice(idx_with_anno,
                                                        self.num_with_anno_per_batch - len(idx_with_anno))
                            idx_with_anno = np.concatenate([idx_with_anno, dup_idx], axis=0)
                        xyz_with_anno = selected_pc[idx_with_anno]
                        labels_with_anno = selected_labels[idx_with_anno]
                        
                else:   # 验证集
                    xyz_with_anno = selected_pc
                    labels_with_anno = selected_labels
            
            else:                                                   # 测试集
                cloud_ind = int(np.argmin(self.min_possibility))
                pick_idx = np.argmin(self.possibility[cloud_ind])
                pc_path = self.path_list[cloud_ind]
                pc, tree, labels = self.get_data(pc_path)
                selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)

                # update the possibility of the selected pc
                dists = np.sum(np.square((selected_pc - pc[pick_idx]).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[cloud_ind][selected_idx] += delta
                self.min_possibility[cloud_ind] = np.min(self.possibility[cloud_ind])
                xyz_with_anno = selected_pc
                labels_with_anno = selected_labels            
                        
            break                
                

            # if self.split != 'test':
            #     cloud_ind = item
            #     pc_path = self.data_list[cloud_ind]
            #     pc, tree, labels = self.get_data(pc_path)
            #     # crop a small point cloud
            #     pick_idx = np.random.choice(len(pc), 1)
            #     selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)
            # else:
            #     cloud_ind = int(np.argmin(self.min_possibility))
            #     pick_idx = np.argmin(self.possibility[cloud_ind])
            #     pc_path = self.path_list[cloud_ind]
            #     pc, tree, labels = self.get_data(pc_path)
            #     selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)

            #     # update the possibility of the selected pc
            #     dists = np.sum(np.square((selected_pc - pc[pick_idx]).astype(np.float32)), axis=1)
            #     delta = np.square(1 - dists / np.max(dists))
            #     self.possibility[cloud_ind][selected_idx] += delta
            #     self.min_possibility[cloud_ind] = np.min(self.possibility[cloud_ind])

        return selected_pc.astype(np.float32), selected_labels.astype(np.int32), selected_idx.astype(np.int32), np.array([cloud_ind], dtype=np.int32),\
               xyz_with_anno.astype(np.float32), labels_with_anno.astype(np.int32)

    def get_data(self, file_path):                  # 从文件中读取点云数据
        seq_id = file_path.split('/')[-3]
        frame_id = file_path.split('/')[-1][:-4]
        kd_tree_path = join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
        # Read pkl with search tree
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        # Load labels
        if int(seq_id) >= 11:   # 测试集
            labels = np.zeros(np.shape(points)[0], dtype=np.uint8)          # 测试集没有label，直接初始化一个全零的矩阵
        else:
            labeled_point = self.labeled_point
            label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
            labels = np.squeeze(np.load(label_path))
            if int(seq_id) != 8:                                    # 训练集 如果是验证集08,则不需要随机稀疏标注,因为08需要推测全部的点,需要全部点的坐标信息
                # ======================================== #
                #          Random Sparse Annotation        #
                # ======================================== #
                if not self.gen_pseudo:             # 训练集才要屏蔽大多数点的标签，验证集不用
                    if '%' in labeled_point:
                        new_labels = np.zeros_like(labels, dtype=np.int32)
                        num_pts = len(labels)                       # 点的个数
                        r = float(labeled_point[:-1]) / 100
                        num_with_anno = max(int(num_pts * r), 1)    # 有标注的点的个数
                        valid_idx = np.where(labels)[0]             # 非零元素的索引，也就是有效的标签的索引？
                        idx_with_anno = np.random.choice(valid_idx, num_with_anno, replace=False)
                        new_labels[idx_with_anno] = labels[idx_with_anno]
                        labels = new_labels
                    else:
                        for i in range(self.num_classes):
                            ind_per_class = np.where(labels == i)[0]  # index of points belongs to a specific class
                            num_per_class = len(ind_per_class)
                            if num_per_class > 0:
                                num_with_anno = int(labeled_point)
                                num_without_anno = num_per_class - num_with_anno
                                idx_without_anno = np.random.choice(ind_per_class, num_without_anno, replace=False)
                                labels[idx_without_anno] = 0
                    # =================================================================== #
                    #            retrain the model with predicted pseudo labels           #
                    # =================================================================== #
                    if self.retrain:
                        pseudo_label_path = './test/sequences'
                        temp = np.load(join(pseudo_label_path, seq_id, 'predictions', frame_id + '.npy'))
                        pseudo_label = np.squeeze(temp)
                        pseudo_label_ratio = 0.01
                        pseudo_label[labels != 0] = labels[labels != 0]
                        labels = pseudo_label
                        self.num_with_anno_per_batch = int(cfg.num_points * pseudo_label_ratio)           
            
        return points, search_tree, labels

    @staticmethod
    def crop_pc(points, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
        select_idx = DP.shuffle_idx(select_idx)     # 打乱了索引，用于随机采样
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx

    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx, batch_xyz_with_anno, batch_labels_with_anno):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx, batch_xyz_with_anno, batch_labels_with_anno]

        return input_list

    def collate_fn(self,batch):

        selected_pc, selected_labels, selected_idx, cloud_ind, xyz_with_anno, labels_with_anno = [],[],[],[],[],[]
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])
            xyz_with_anno.append(batch[i][4])
            labels_with_anno.append(batch[i][5])

        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)
        xyz_with_anno = np.stack(xyz_with_anno)
        labels_with_anno = np.stack(labels_with_anno)
        

        flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind, xyz_with_anno, labels_with_anno)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        # inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1,2).float()
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()
        
        inputs['batch_xyz_anno'] = torch.from_numpy(flat_inputs[4 * num_layers + 4]).float()            # 改成了float
        inputs['batch_label_anno'] = torch.from_numpy(flat_inputs[4 * num_layers + 5]).long()
        

        return inputs           # 这个return的值就是遍历 dataloader 时的那个


if __name__ == '__main__':

    train_dataset = SemanticKITTILoader(14, "0.1%", False, False, 'training')
    validation_dataset = SemanticKITTILoader(14, "0.1%", False, False, 'validation')
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=train_dataset.collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=6, shuffle=True, collate_fn=validation_dataset.collate_fn)
    for data in train_dataloader:
        print(len(data['xyz']))
        print(data['xyz'][0].shape)
        print(data['features'].shape)
        print(data['labels'].shape)
        print(data['input_inds'].shape)
        print(data['cloud_inds'].shape)
        print(data['batch_xyz_anno'].shape)
        print(data['batch_label_anno'].shape)
        break
    for data in validation_dataloader:
        print(len(data['xyz']))
        print(data['xyz'][0].shape)
        print(data['features'].shape)
        print(data['labels'].shape)
        print(data['input_inds'].shape)
        print(data['cloud_inds'].shape)
        print(data['batch_xyz_anno'].shape)
        print(data['batch_label_anno'].shape)
        break