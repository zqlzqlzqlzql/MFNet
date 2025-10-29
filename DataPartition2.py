import numpy as np
import torch
import argparse
from scipy.io import loadmat
from torch.utils import data
import matplotlib
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Houston2013','Trento'], default='Trento', help='dataset to use')
parser.add_argument('--patch_size', default=11, help='dataset to use')
parser.add_argument('--batch_size', default=32, help='dataset to use')
parser.add_argument('--num_classes', default=6, help='dataset to use')
parser.add_argument('--partition_percent', default=0.1, help='dataset to use')
parser.add_argument('--partition_least', default=5, help='dataset to use')

args = parser.parse_args()


def indexToAssignment(index_, pad_length, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, ex_len, pos_row, pos_col):
    # print(matrix.shape)
    selected_rows = matrix[:,range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, :, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(proptionTR, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_train = int(proptionTR * len(indices)) #类别i在测试集的样本总数
        if nb_train < 5:
            nb_train = args.partition_least
        train[i] = indices[:nb_train] #字典：key为类别，values为每个类别的数量
        test[i] = indices[nb_train:]
    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
        train_indices += train[i] #索引字典的键，得到values
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return whole_indices, train_indices, test_indices


def zeroPadding_3D(old_matrix, pad_length, pad_depth=0):
    depth, height, width = old_matrix.shape
    new_height = ((height + 2 * pad_length - 1) // args.patch_size) * args.patch_size
    new_width = ((width + 2 * pad_length - 1) // args.patch_size) * args.patch_size
    pad_height = max(0, (new_height - height) // 2)
    pad_width = max(0, (new_width - width) // 2)
    if (height + 2 * pad_height) % args.patch_size != 0:
        pad_height += 1
    if (width + 2 * pad_width) % args.patch_size != 0:
        pad_width += 1
    new_matrix = np.pad(old_matrix,
                        ((pad_depth, pad_depth),
                         (pad_height, pad_height),
                         (pad_width, pad_width)),
                        mode='constant', constant_values=0)
    return new_matrix


class HSIDataset(data.Dataset):
    def __init__(self, list_IDs, samples, labels):
        self.list_IDs = list_IDs
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.samples[ID]
        y = self.labels[ID]

        return X, y



'''load dataset'''
if args.dataset == 'Houston2013':
    data_hsi = loadmat('F:\pythonProject\datasets\Multisource\data\Houston2013\HSI.mat')
    data_HSI = data_hsi['HSI']
    data_lidar = loadmat('F:\pythonProject\datasets\Multisource\data\Houston2013\LiDAR.mat')
    data_liDAR = data_lidar['LiDAR']
    groundtruth = loadmat('F:\pythonProject\datasets\Multisource\data\Houston2013\gt.mat')
    groundtruth = groundtruth['gt']
    color_mat = loadmat('F:\pythonProject\datasets\Multisource\data\Houston2013\Houston2013Colormap.mat')
elif args.dataset == 'Trento':
    data_hsi = loadmat('F:\pythonProject\datasets\Multisource\data\Trento\HSI.mat')
    data_HSI = data_hsi['HSI']
    data_lidar = loadmat('F:\pythonProject\datasets\Multisource\data\Trento\LiDAR.mat')
    data_liDAR = data_lidar['LiDAR']
    groundtruth = loadmat('F:\pythonProject\datasets\Multisource\data\Trento\gt.mat')
    groundtruth = groundtruth['gt']
    color_mat = loadmat('F:\pythonProject\datasets\Multisource\data\Trento\TrentoColormap.mat')
else:
    raise ValueError('Unknow dataset')

color_mat_list = list(color_mat)
color_matrix = color_mat[color_mat_list[3]] #(16,3)
color_matrix = color_matrix[0:args.num_classes+1,:]

# print ('The shape of HSI:\n',data_HSI.shape)
# print ('The shape of liDAR:\n',data_liDAR.shape)
# print('We have {} classes! They are {}.'.format(np.max(groundtruth),np.unique(groundtruth)))

'''obtain shape of multimodal remote image'''
data_HSI2 = data_HSI
data_HSI = np.reshape(data_HSI, (-1, data_HSI.shape[2]))
pca = PCA(n_components=30, whiten=True)
data_HSI = pca.fit_transform(data_HSI)
data_HSI1 = np.reshape(data_HSI, (data_HSI2.shape[0], data_HSI2.shape[1], 30))
height, width, band_HSI = data_HSI1.shape #(349,1905,144)
height, width = data_liDAR.shape #(349,1905)
band_liDAR = 1
MAX_HSI = data_HSI.max()
data_HSI = np.transpose(data_HSI1, (2,0,1))

data_HSI = data_HSI - np.mean(data_HSI, axis=(1,2), keepdims=True)
data_HSI = data_HSI / MAX_HSI
data_liDAR = (data_liDAR - data_liDAR.min()) / (data_liDAR.max() - data_liDAR.min())

data = data_HSI.reshape(np.prod(data_HSI.shape[:1]),np.prod(data_HSI.shape[1:]))
data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data)
new_gt = groundtruth
gt = new_gt.reshape(np.prod(new_gt.shape[:2]),)

whole_data = data.reshape(data_HSI.shape[0], data_HSI.shape[1],data_HSI.shape[2])
whole_data = whole_data - np.mean(whole_data, axis=(1,2), keepdims=True)
PATCH_LENGTH = int((args.patch_size-1)/2)
padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH) #(144,355,1911)
padded_data_liDAR = zeroPadding_3D(np.expand_dims(data_liDAR, axis=0), PATCH_LENGTH)



'''obtain index of training set, testing set, and val set'''
all_indices, train_indices, test_indices = sampling(args.partition_percent, gt)

'''number of training samples, testing samples, and total samples'''
TRAIN_SIZE = len(train_indices)
TEST_SIZE = len(test_indices)
# TOTAL_SIZE = TRAIN_SIZE + TEST_SIZE

'''hyperspectral image'''
train_data_HSI = np.zeros((TRAIN_SIZE, band_HSI, args.patch_size, args.patch_size))
test_data_HSI = np.zeros((TEST_SIZE, band_HSI, args.patch_size, args.patch_size))
# all_data_HSI = np.zeros((TOTAL_SIZE, band_HSI, args.patch_size, args.patch_size))

'''liDAR image'''
train_data_liDAR = np.zeros((TRAIN_SIZE, band_liDAR, args.patch_size, args.patch_size))
test_data_liDAR = np.zeros((TEST_SIZE, band_liDAR, args.patch_size, args.patch_size))
# all_data_liDAR = np.zeros((TOTAL_SIZE, band_liDAR, args.patch_size, args.patch_size))

'''groundtruth'''
y_train = gt[train_indices]
y_test = gt[test_indices]
# y_all = gt[all_indices]


train_assign = indexToAssignment(train_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
for i in range(len(train_assign)):
    if (train_assign[i][0] - PATCH_LENGTH >= 0 and train_assign[i][0] + PATCH_LENGTH < padded_data.shape[1] and
            train_assign[i][1] - PATCH_LENGTH >= 0 and train_assign[i][1] + PATCH_LENGTH < padded_data.shape[2]):
        train_data_HSI[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, train_assign[i][0], train_assign[i][1])
        train_data_liDAR[i] = selectNeighboringPatch(padded_data_liDAR, PATCH_LENGTH, train_assign[i][0], train_assign[i][1])

test_assign = indexToAssignment(test_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
for i in range(len(test_assign)):
    if (test_assign[i][0] - PATCH_LENGTH >= 0 and test_assign[i][0] + PATCH_LENGTH < padded_data.shape[1] and
            test_assign[i][1] - PATCH_LENGTH >= 0 and test_assign[i][1] + PATCH_LENGTH < padded_data.shape[2]):
        test_data_HSI[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, test_assign[i][0], test_assign[i][1])
        test_data_liDAR[i] = selectNeighboringPatch(padded_data_liDAR, PATCH_LENGTH, test_assign[i][0], test_assign[i][1])
# #
all_assign = indexToAssignment(all_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
# for i in range(len(all_assign)):
#     if (all_assign[i][0] - PATCH_LENGTH >= 0 and all_assign[i][0] + PATCH_LENGTH < padded_data.shape[1] and
#             all_assign[i][1] - PATCH_LENGTH >= 0 and all_assign[i][1] + PATCH_LENGTH < padded_data.shape[2]):
#         all_data_HSI[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, all_assign[i][0], all_assign[i][1])
#
# # Parameters
# params = {'batch_size': args.batch_size,
#           'shuffle': True,
#           'num_workers': 0}
# #
# train_data_HSI = torch.from_numpy(train_data_HSI)
# test_data_HSI = torch.from_numpy(test_data_HSI)
# all_data_HSI = torch.from_numpy(all_data_HSI)
# #
# # # Generators
# training_set = HSIDataset(range(len(train_indices)), train_data_HSI, y_train)
# training_generator = torch.utils.data.DataLoader(training_set, **params)
# #
# validation_set = HSIDataset(range(len(test_indices)), test_data_HSI, y_test)
# validation_generator = torch.utils.data.DataLoader(validation_set, **params)
# #
# all_set = HSIDataset(range(len(all_indices)), all_data_HSI, y_all)
# all_generator = torch.utils.data.DataLoader(all_set, **params)
#
#
# trainloader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
#
# validationloader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
# allloader = torch.utils.data.DataLoader(all_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
#
# # load data
# y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
# y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
height, width = 166, 600
y_train_2d = np.zeros((height, width))
y_test_2d = np.zeros((height, width))

# Fill y_train_2d and y_test_2d based on train_indices and test_indices
for idx, value in zip(test_indices, y_test):
    row, col = divmod(idx, width)
    y_test_2d[row, col] = value

for idx, value in zip(train_indices, y_train):
    row, col = divmod(idx, width)
    y_train_2d[row, col] = value

"""main"""
#-------------------------------------------------------------------------------

# output classification maps
"""all"""
# plt.subplot(1,1,1)
# plt.imshow(groundtruth, colors.ListedColormap(color_matrix))
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# """train"""
# prediction_matrix = np.zeros((height, width), dtype=float)
# for i in range(len(train_assign)):
#     prediction_matrix[train_assign[i][0], train_assign[i][1]] = y_train[i] + 1
# plt.subplot(1, 1, 1)
# plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# """test"""
# prediction_matrix = np.zeros((height, width), dtype=float)
# for i in range(len(test_assign)):
#     prediction_matrix[test_assign[i][0], test_assign[i][1]] = y_test[i] + 1
# plt.subplot(1, 1, 1)
# plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
# plt.xticks([])
# plt.yticks([])
# plt.show()
# test_data_HSI = np.transpose(test_data_HSI, (0, 2, 3, 1))
# train_data_HSI = np.transpose(train_data_HSI, (0, 2, 3, 1))
# test_data_liDAR = np.transpose(test_data_liDAR, (0, 2, 3, 1))
# train_data_liDAR = np.transpose(train_data_liDAR, (0, 2, 3, 1))
# y_train = y_train.reshape(1, -1)
# y_test = y_test.reshape(1, -1)
