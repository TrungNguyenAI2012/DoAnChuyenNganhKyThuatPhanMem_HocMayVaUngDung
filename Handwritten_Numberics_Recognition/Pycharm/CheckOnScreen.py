from PIL import ImageGrab
import numpy as np
import cv2
import pickle

######################################################################################################
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def openScreenAndRecognitionNumberByImage():
    while True:
        imgOriginal = ImageGrab.grab(bbox=(470, 350, 945, 830))
        imgOriginal = np.asarray(imgOriginal)
        img = cv2.resize(imgOriginal, (32, 32))
        img = preProcessing(img)
        img = img.reshape(1, 32, 32, 1)
        # Predict
        classIndex = int(model.predict_classes(img))
        predictions = model.predict(img)
        probVal = np.amax(predictions)
        print(classIndex, probVal)
        if probVal > threshold:
            cv2.putText(imgOriginal,
                        str(classIndex) + ": " + str(round(round(probVal, 2) * 100)) + "%",
                        (0, 25),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        1)
        cv2.imshow('Screen', imgOriginal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

######################################################################################################
path_model = 'Model/H3.p'
threshold = 0
pickle_in = open(path_model, "rb")
model = pickle.load(pickle_in)

openScreenAndRecognitionNumberByImage()
