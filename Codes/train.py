from Network.NetworkModule import GazeNet
from Network.Config import Methods, Configure
import pickle,cv2
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from  openpyxl import Workbook
import os, sklearn
from keras.utils import plot_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def LoadDataset():
    pickleFile = open(Configure.PATH_CUSTOMIZED_GT,'rb')
    dict = pickle.load(pickleFile)

    x_images = []
    x_poseInfos = []

    y = []

    for filename in dict:
        img = cv2.imread(Configure.DIR_CUSTOMIZED_DATASET+filename)

        Info = dict[filename]
        face = Info['face']
        eye1 = Info["eye1"]
        eye2 = Info['eye2']
        gt = Info['gt']

        pose = face + eye1 + eye2

        x_images.append(img)
        x_poseInfos.append(pose)
        y.append((gt[0]/Configure.resolution[0], gt[1]/Configure.resolution[1]))  # 输出除以resolution.

    x_images = np.array(x_images,dtype='float')
    x_poseInfos = np.array(x_poseInfos, dtype='float')
    y = np.array(y, dtype='float')

    x_images /=255.
    x_images -= .5


    print("数据集-输入: ", x_images.shape, x_poseInfos.shape)
    print("数据集-输出: ", y.shape)
    return x_images, x_poseInfos, y




if __name__ == '__main__':
    x_images, x_poseInfos, y_gt = LoadDataset()

    model = GazeNet.BuildNetwork()
    model.compile(loss='mse', optimizer=Adam(Configure.Learning_rate))
    # plot_model(model, to_file='./network_structure.png', show_shapes=True)

    image_num = x_images.shape[0]
    batch_amount = image_num // Configure.Batch_size
    rest_size = image_num - (batch_amount * Configure.Batch_size)
    history_epoch = []
    history_batch = []
    iter = 0
    while iter < Configure.Epoch:

        # shuffle the training set.
        index = [i for i in range(image_num)]
        np.random.shuffle(index)

        for bid in range(batch_amount):

            # using the shuffled index
            x_batch1 = x_images   [index[bid * Configure.Batch_size: (bid + 1) * Configure.Batch_size]]
            x_batch2 = x_poseInfos[index[bid * Configure.Batch_size: (bid + 1) * Configure.Batch_size]]
            y_batch  = y_gt        [index[bid * Configure.Batch_size: (bid + 1) * Configure.Batch_size]]

            loss = model.train_on_batch(x=[x_batch1, x_batch2], y=y_batch)
            history_batch.append((iter,bid,loss))

            print("ITER %d/%d - Batch %d : %s Loss = %.6f" % (iter, Configure.Epoch, bid,
                                                              Methods.GetProcessBar(bid,batch_amount), loss))

        epoch_loss = model.evaluate(x=[x_images, x_poseInfos], y=y_gt,verbose=0,batch_size=Configure.Batch_size)
        history_epoch.append((iter,epoch_loss))
        print("------------EPOCH {} MSE = {}\n".format(iter,epoch_loss))

        iter += 1


    model.save("model_{}.h5".format(history_epoch[-1][-1]))

    # Save the training history information.
    wb = Workbook()
    ws1 = wb.active  # MSE/MLAE
    ws1.title = "EPOCH History"
    ws2 = wb.create_sheet("BATCH History")  # batch loss

    ws1.append(["Epoch ID", "MSE Loss"])
    ws2.append(["Epoch ID", "Batch ID", "MSE Loss"])

    for i in range(len(history_epoch)):
        ws1.append(history_epoch[i])
    for i in range(len(history_batch)):
        ws2.append(history_batch[i])
    wb.save("train_info.xlsx")

    # Save the predicted results
    y_pred= model.predict(x=[x_images,x_poseInfos],batch_size=Configure.Batch_size)

    with open("./results.txt","w") as f:
        for index in range(y_pred.shape[0]):
            f.write("%.4f\t%.4f  ==  %.4f  %.4f\n" % (y_gt[index,0],y_gt[index,1], y_pred[index,0],y_pred[index,1]))
            f.write("{}\t{}  ==  {}  {}\n".format(int(y_gt[index, 0] * Configure.resolution[0]), int(y_gt[index, 1]*Configure.resolution[1]),
                                                  int(y_pred[index, 0]*Configure.resolution[0]), int(y_pred[index, 1]*Configure.resolution[1])))

    # history = model.fit(x=[x_images,x_poseInfos],y=y_gt, batch_size=Configure.Batch_size, epochs=Configure.Epoch)
    # print(history.history)
    # print(history.epoch)
    #
    # model.save_weights("./weight")

    # model.load_weights("./logs/best_weight")


    # y_pred= model.predict(x=[x_images[:50],x_poseInfos[:50]])
    #
    # print("mse loss:", sklearn.metrics.mean_squared_error(y_gt[:50], y_pred))
    # print(y_gt[:10])
    # print("------")
    # print(y_pred)
    # print("------")
    # print(x_poseInfos[:10])
    #
    # with open("./results.txt","w") as f:
    #     for index in range(y_pred.shape[0]):
    #         f.write("{}\t{}  ==  {}  {}\n".format(y_gt[index,0],y_gt[index,1], y_pred[index,0],y_pred[index,1]))
