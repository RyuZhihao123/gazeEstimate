import numpy, os, shutil

class Configure:
    # Target Path
    TARGET_ROOT_PATH = "E:\\workspace\\Dataset_Gaze\\"
    # raw data
    DIR_RAW_DATA = TARGET_ROOT_PATH + "Datasets-lsm\\dataset\\images\\"
    PATH_RAW_GT =  TARGET_ROOT_PATH + "Datasets-lsm\\dataset\\data.txt"

    # customized data
    DIR_CUSTOMIZED__ROOT = TARGET_ROOT_PATH +"Customized_Dataset\\"
    DIR_CUSTOMIZED_DATASET = DIR_CUSTOMIZED__ROOT + "dataset\\"
    PATH_CUSTOMIZED_GT =     TARGET_ROOT_PATH +"Customized_Dataset\\gt.p"

    #
    image_size = (640,360)          # size of the input image
    resolution = (1920., 1080.)     # resolution of the output ratio.


    # Training setup.
    Epoch = 40
    Learning_rate = 0.001
    Batch_size = 6




class Methods:
    @staticmethod
    def GetAllFilesIn(filepath):
        filepaths = []
        filenames = []
        for root, dirs, files in os.walk(filepath):
            for f in files:
                filepaths.append(os.path.join(root, f))
                filenames.append(f)
        print("加载文件: {} ".format(len(filepaths)))

        return filepaths, filenames

    @staticmethod
    def ReadGroundTruth(filepath):
        dict = {}

        with open(filepath, 'r') as f:
            list1 = f.readlines()

            for line in list1:
                line = line.split()

                if len(line) != 3:
                    continue

                dict[line[0]] = (int(line[1]), int(line[2]))
        return dict

    @staticmethod
    def ClearDir(path):
        if os.path.exists(path):
            print("Resetting the folder.....", path)
            shutil.rmtree(path=path)
        os.mkdir(path)

    @staticmethod
    def MakeDir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def RemoveDir(path):
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def GetProcessBar(bid, batch_num, dot_num=40):
        ratio = (bid + 1) / batch_num
        delta = 40 - (int(ratio * dot_num) + int((1 - ratio) * dot_num))
        return '[' + '=' * int(ratio * dot_num) + '>' + "." * int((1 - ratio) * dot_num + delta) + ']'

