import sys
sys.path.append("./../")
import re
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import time
import torch.utils.data as dataf
import torch.nn as nn
import torch
import numpy as np
import os
cudnn.deterministic = True
cudnn.benchmark = False
import DataPartition1
from net7 import MFNet

DATASETS_WITH_HSI_PARTS = ['Berlin', 'Augsburg']
DATA2_List = ['SAR','DSM','MS']
os.environ["CUDA_VISIBLE_DEVICES"]="0"
datasetNames = ["Augsburg"]

patchsize = 11
batchsize = 32
testSizeNumber = 500
EPOCH = 200
BandSize = 1
LR = 5e-4
FM = 16
HSIOnly = False
FileName = 'MFT'


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(xtest, xtest2, ytest, name, model):
    pred_y = np.empty((len(ytest)), dtype=np.float32)
    number = len(ytest) // testSizeNumber
    for i in range(number):
        temp = xtest[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
        temp = temp.cuda()
        temp1 = xtest2[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
        temp1 = temp1.cuda()

        temp2 = model(temp, temp1)

        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i * testSizeNumber:(i + 1) * testSizeNumber] = temp3.cpu()
        del temp, temp2, temp3, temp1

    if (i + 1) * testSizeNumber < len(ytest):
        temp = xtest[(i + 1) * testSizeNumber:len(ytest), :, :]
        temp = temp.cuda()
        temp1 = xtest2[(i + 1) * testSizeNumber:len(ytest), :, :]
        temp1 = temp1.cuda()

        temp2 = model(temp, temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * testSizeNumber:len(ytest)] = temp3.cpu()
        del temp, temp2, temp3, temp1

    pred_y = torch.from_numpy(pred_y).long()

    if name == 'Houston':
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass'
            , 'Trees', 'Soil', 'Water',
                        'Residential', 'Commercial', 'Road', 'Highway',
                        'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court',
                        'Running Track']
    elif name == 'Trento':
        target_names = ['Apples', 'Buildings', 'Ground', 'Woods', 'Vineyard',
                        'Roads']
    elif name == 'MUUFL' or name == 'MUUFLS' or name == 'MUUFLSR':
        target_names = ['Trees', 'Grass_Pure', 'Grass_Groundsurface', 'Dirt_And_Sand', 'Road_Materials', 'Water',
                        "Buildings'_Shadow",
                        'Buildings', 'Sidewalk', 'Yellow_Curb', 'ClothPanels']
    elif name == 'Augsburg':
        target_names = ['Forest', 'Residential-Area', 'Industrial-Area', 'Low-Plants',
                        'Allotment',
                        'Commercial-Area', 'Water']

    #     classification = classification_report(ytest, pred_y, target_names=target_names)
    oa = accuracy_score(ytest, pred_y)
    confusion = confusion_matrix(ytest, pred_y)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(ytest, pred_y)

    return confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train():
    for BandSize in [1]:
        for datasetName in datasetNames:
            print("----------------------------------Training for ", datasetName,
                  " ---------------------------------------------")
            try:
                os.makedirs(datasetName)
            except FileExistsError:
                pass
            data1Name = ''
            data2Name = ''
            if datasetName in ["Houston", "Trento", "MUUFL"]:
                data1Name = datasetName
                data2Name = "LIDAR"
            else:
                for dataName in DATA2_List:
                    dataNameToCheck = re.compile(dataName)
                    matchObj = dataNameToCheck.search(datasetName)
                    if matchObj:
                        data1Name = datasetName.replace(dataName, "")
                        data2Name = dataName

            HSI = DataPartition1.train_data_HSI
            TrainPatch = HSI
            TrainPatch = TrainPatch.astype(np.float32)
            NC = TrainPatch.shape[3]  # NC is number of bands

            LIDAR = DataPartition1.train_data_liDAR
            TrainPatch2 = LIDAR
            TrainPatch2 = TrainPatch2.astype(np.float32)
            NCLIDAR = TrainPatch2.shape[3]  # NC is number of bands

            label = DataPartition1.y_train
            TrLabel = label

            # Test data
            HSI = DataPartition1.test_data_HSI
            TestPatch = HSI
            TestPatch = TestPatch.astype(np.float32)

            LIDAR = DataPartition1.test_data_liDAR
            TestPatch2 = LIDAR
            TestPatch2 = TestPatch2.astype(np.float32)

            label = DataPartition1.y_test
            TsLabel = label

            TrainPatch1 = torch.from_numpy(TrainPatch).to(torch.float32)
            TrainPatch2 = torch.from_numpy(TrainPatch2).to(torch.float32)
            TrainLabel1 = torch.from_numpy(TrLabel) - 1
            TrainLabel1 = TrainLabel1.long()
            TrainLabel1 = TrainLabel1.reshape(-1)

            TestPatch1 = torch.from_numpy(TestPatch).to(torch.float32)
            TestPatch2 = torch.from_numpy(TestPatch2).to(torch.float32)
            TestLabel1 = torch.from_numpy(TsLabel) - 1
            TestLabel1 = TestLabel1.long()
            TestLabel1 = TestLabel1.reshape(-1)

            Classes = len(np.unique(TrainLabel1))
            dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel1)

            train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0)
            print("HSI Train data shape = ", TrainPatch1.shape)
            print(data2Name + " Train data shape = ", TrainPatch2.shape)
            print("Train label shape = ", TrainLabel1.shape)

            print("HSI Test data shape = ", TestPatch1.shape)
            print(data2Name + " Test data shape = ", TestPatch2.shape)
            print("Test label shape = ", TestLabel1.shape)

            print("Number of Classes = ", Classes)
            KAPPA = []
            OA = []
            AA = []
            ELEMENT_ACC = np.zeros((5, Classes))  # 存储五次运行的分类精度

            for run in range(1):  # 运行五次
                print(f"\n=================== Run {run + 1}/5 ===================")
                set_seed(42 + run)
                model = MFNet(FM, NC, Classes, dropout=0.1, in_channels=1).cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-3)
                loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
                BestAcc = 0

                torch.cuda.synchronize()
                start = time.time()
                # train and test the designed model
                for epoch in range(EPOCH):
                    model.train()
                    total_loss = 0  # 用于累积每个 epoch 的训练损失

                    for step, (b_x1, b_x2, b_y) in enumerate(train_loader):
                        # move train data to GPU
                        b_x1 = b_x1.cuda()
                        b_y = b_y.cuda()
                        if HSIOnly:
                            out1 = model(b_x1, b_x2)
                            loss = loss_func(out1, b_y)
                        else:
                            b_x2 = b_x2.cuda()
                            out = model(b_x1, b_x2)
                            loss = loss_func(out, b_y)

                        optimizer.zero_grad()  # clear gradients for this training step
                        loss.backward()  # backpropagation, compute gradients
                        optimizer.step()  # apply gradients

                        total_loss += loss.item()  # 累积损失

                    # 计算平均训练损失
                    avg_loss = total_loss / len(train_loader)

                    # 每个 epoch 结束后评估测试集
                    model.eval()
                    pred_y = np.empty((len(TestLabel1)), dtype='float32')
                    number = len(TestLabel1) // testSizeNumber
                    for i in range(number):
                        temp = TestPatch1[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
                        temp = temp.cuda()
                        temp1 = TestPatch2[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
                        temp1 = temp1.cuda()
                        if HSIOnly:
                            temp2 = model(temp, temp1)
                            temp3 = torch.max(temp2, 1)[1].squeeze()
                            pred_y[i * testSizeNumber:(i + 1) * testSizeNumber] = temp3.cpu()
                            del temp, temp2, temp3
                        else:
                            temp2 = model(temp, temp1)
                            temp3 = torch.max(temp2, 1)[1].squeeze()
                            pred_y[i * testSizeNumber:(i + 1) * testSizeNumber] = temp3.cpu()
                            del temp, temp1, temp2, temp3

                    if (i + 1) * testSizeNumber < len(TestLabel1):
                        temp = TestPatch1[(i + 1) * testSizeNumber:len(TestLabel1), :, :]
                        temp = temp.cuda()
                        temp1 = TestPatch2[(i + 1) * testSizeNumber:len(TestLabel1), :, :]
                        temp1 = temp1.cuda()
                        if HSIOnly:
                            temp2 = model(temp, temp1)
                            temp3 = torch.max(temp2, 1)[1].squeeze()
                            pred_y[(i + 1) * testSizeNumber:len(TestLabel1)] = temp3.cpu()
                            del temp, temp2, temp3
                        else:
                            temp2 = model(temp, temp1)
                            temp3 = torch.max(temp2, 1)[1].squeeze()
                            pred_y[(i + 1) * testSizeNumber:len(TestLabel1)] = temp3.cpu()
                            del temp, temp1, temp2, temp3

                    pred_y = torch.from_numpy(pred_y).long()
                    accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)

                    print('Epoch: ', epoch, '| train loss: %.4f' % avg_loss,
                          '| test accuracy: %.4f' % (accuracy * 100))

                    # save the parameters in network
                    if accuracy > BestAcc:
                        BestAcc = accuracy
                        torch.save(model.state_dict(), datasetName + '/net_params_' + FileName + f'_run{run}.pkl')

                    scheduler.step()

                torch.cuda.synchronize()
                end = time.time()
                print(end - start)
                Train_time = end - start

                # load the saved parameters
                model.load_state_dict(torch.load(datasetName + '/net_params_' + FileName + f'_run{run}.pkl'))

                model.eval()
                confusion, oa, each_acc, aa, kappa = reports(TestPatch1, TestPatch2, TestLabel1, datasetName, model)
                print(f"Each class accuracy: {each_acc}")
                KAPPA.append(kappa)
                OA.append(oa)
                AA.append(aa)
                ELEMENT_ACC[run, :] = each_acc
                torch.save(model, datasetName + '/best_model_' + FileName + '_BandSize' + str(BandSize) + '_Iter' + str(
                    run) + '.pt')
                print("OA = ", oa)
                print("Average accuracy (AA): ", aa)
                print("KAPPA:", kappa)

            # 计算五次运行的平均结果
            avg_OA = np.mean(OA)
            avg_AA = np.mean(AA)
            avg_Kappa = np.mean(KAPPA)
            avg_ELEMENT_ACC = np.mean(ELEMENT_ACC, axis=0)

            # 输出平均结果
            print("\n=================== Final Average Results ===================")
            print(f"Average OA: {avg_OA:.4f} ± {np.std(OA):.4f}")
            print(f"Average AA: {avg_AA:.4f} ± {np.std(AA):.4f}")
            print(f"Average Kappa: {avg_Kappa:.4f} ± {np.std(KAPPA):.4f}")
            print("Average Class Accuracy:", avg_ELEMENT_ACC)
            print("=============================================================")

            print("----------" + datasetName + " Training Finished -----------")


train()