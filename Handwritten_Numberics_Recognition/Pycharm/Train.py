import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle

######################################################################################################
###### Hàm tiền xủ lý ảnh
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

###### Hàm mô hình huấn luyện
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(32, 32, 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

######################################################################################################
###### Khai báo biến
path_data = 'Data/H3'
path_model = 'Model/H3.p'
testRatio = 0.2
valRatio = 0.2
batchSizeVal = 32
epochsVal = 100

###### Import the data: Đưa dữ liệu vào máy
count = 0
images = []
classNo = []
myList = os.listdir(path_data)
noOfClasses = len(myList)
print("Nhập dữ liệu các lớp...")
print("Nhập thành công lớp: ", end=" ")
for x in range(0, noOfClasses):
    myPicList = os.listdir(path_data + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path_data + "/" + str(count) + "/" + y)
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
print("Tổng số hình là: ", len(images))
images = np.array(images)
classNo = np.array(classNo)

###### Spliting the data: Chia dữ liệu thành các tập [Huấn luyện, Kiểm tra, Tối ưu]
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)
print("Số hình dùng để huấn luyện: ", X_train.shape[0])
print("Số hình dùng để thử nghiệm: ", X_test.shape[0])
print("Số hình dùng để tối ưu: ", X_validation.shape[0])
numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))
plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

###### Tiền xủ lý ảnh
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

###### Huấn luyện
model = myModel()
print(model.summary())
history = model.fit_generator(dataGen.flow(X_train,
                                           y_train,
                                           batch_size=batchSizeVal),
                              epochs=epochsVal,
                              validation_data=(X_validation, y_validation),
                              shuffle=1)

###### Hiện bảng hàm mất mát
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

###### Hiện bảng hàm chính xác
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

###### Điểm số thử huấn luyện
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])

###### Lưu mô hình đã huấn luyện vào file
pickle_out = open(path_model, "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
