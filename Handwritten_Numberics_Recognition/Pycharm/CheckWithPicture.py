import numpy as np
import cv2
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

######################################################################################################
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def chooseImg():
    Tk().withdraw()
    file_path = askopenfilename()
    recognitionNumberByImage(file_path)

def recognitionNumberByImage(file_path):
    image = cv2.imread(file_path, 1)
    image = cv2.resize(image, (320, 320))
    while True:
        cv2.imshow('Image', image)
        img = np.asarray(image)
        img = cv2.resize(img, (32, 32))
        img = preProcessing(img)
        img = img.reshape(1, 32, 32, 1)
        # Predict
        classIndex = int(model.predict_classes(img))
        predictions = model.predict(img)
        probVal = np.amax(predictions)
        print(classIndex, probVal)
        if probVal > threshold:
            probVal = round(round(probVal, 2) * 100)
            cv2.putText(image,
                        str(classIndex) + ": " + str(probVal) + "%",
                        (0, 25),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        1)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.destroyAllWindows()
            chooseImg()
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

######################################################################################################
path_model = 'Model/H3.p'
threshold = 0
pickle_in = open(path_model, "rb")
model = pickle.load(pickle_in)

chooseImg()
