from Network.Config import Configure
from Network.Config import Methods
import cv2
import pickle
import numpy as np
if __name__ == '__main__':

    filepaths, filenames = Methods.GetAllFilesIn(Configure.DIR_RAW_DATA)
    dict = Methods.ReadGroundTruth(Configure.PATH_RAW_GT)

    dataSize = len(filepaths)

    # init output dataset
    Methods.ClearDir(Configure.DIR_CUSTOMIZED__ROOT)
    Methods.ClearDir(Configure.DIR_CUSTOMIZED_DATASET)

    face_cascade = cv2.CascadeClassifier('./Data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./Data/haarcascade_eye.xml')

    final_gts = {}

    for i in range(dataSize):

        inputImage = cv2.resize(cv2.imread(filepaths[i]), Configure.image_size)

        face_Rects = face_cascade.detectMultiScale(inputImage, 1.3, 5)

        if len(face_Rects) > 0:
            x, y, w, h = face_Rects[0]  # first face position.

            face_roi = inputImage[y:y+h, x:x+w]  # face region of interest

            eyes = eye_cascade.detectMultiScale(face_roi)

            if(len(eyes) != 2):
                continue

            targetImage = np.ones(shape=(Configure.image_size[1], Configure.image_size[0],3)).astype(np.int) *255

            cur_info = {}
            cur_info["face"] = (x,y,w,h)


            cv2.rectangle(targetImage,(x,y), (x+w,y+h),(255,0,0), 2)
            count = 1
            for (ex, ey, ew, eh) in eyes:
                ex = x+ex
                ey = y+ey
                print(ex,ey,ew,eh, inputImage.shape, targetImage.shape)
                targetImage[ ey:ey+eh, ex:ex+ew] = inputImage[ ey:ey+eh, ex:ex+ew]
                cv2.rectangle(targetImage, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cur_info["eye{}".format(count)] = (ex,ey,ew,eh)
                count +=1

            cv2.imwrite(Configure.DIR_CUSTOMIZED_DATASET+filenames[i], targetImage)
            cur_info['gt']= dict[filenames[i]]


            final_gts[filenames[i]] = cur_info

            # cv2.imshow("aadas",inputImage)
            # cv2.waitKey(0)

    pickle.dump(final_gts, open(Configure.PATH_CUSTOMIZED_GT,'wb'))






