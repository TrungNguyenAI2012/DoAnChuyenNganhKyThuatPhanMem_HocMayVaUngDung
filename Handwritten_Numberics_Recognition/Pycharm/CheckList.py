import numpy as np
import cv2
import os
import pickle

######################################################################################################
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

######################################################################################################
path_check = 'Data/H1'
path_model = 'Model/H3.p'
threshold = 0
pickle_in = open(path_model, "rb")
model = pickle.load(pickle_in)

count = trueImg = totalImg = 0
myList = os.listdir(path_check)

for x in range(0, len(myList)):
    myPicList = os.listdir(path_check + "/" + str(count))
    print('Lớp', count)
    for y in myPicList:
        curImg = cv2.imread(path_check + "/" + str(count) + "/" + y)
        curImg = cv2.resize(curImg, (32, 32))
        image = cv2.resize(curImg, (320, 320))
        img = np.asarray(image)
        img = cv2.resize(img, (32, 32))
        img = preProcessing(img)
        img = img.reshape(1, 32, 32, 1)
        # Predict
        classIndex = int(model.predict_classes(img))
        predictions = model.predict(img)
        probVal = np.amax(predictions)
        print("Số " + str(classIndex) + ": " + str(round(round(probVal, 2) * 100)) + "%")
        totalImg += 1
        if count == classIndex:
            trueImg += 1
    count += 1
    print('')
print('Tổng cộng có', totalImg, 'hình được kiểm tra')
print('Đúng', trueImg, 'hình. Chiếm', round((trueImg / totalImg * 100), 2),'%')