import pytesseract #ADD Tesseract-OCR binaries --> path variables  
import cv2
import matplotlib.pyplot as plt
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
img = cv2.imread(r'C:/Users/Antoine/Pictures/SingleDigitrecognition.jpg')
#?string = pytesseract.image_to_string(img, config="-psm 6")
#string = pytesseract.image_to_string(img)
string = pytesseract.image_to_string(img) #config='digits')
print(string)


#! look for image to data function
import sys
import cv2
import numpy as np
import picture


class DigitRecognizerTraining:
    """Class used to train digits on an image"""

    def __init__(self):
        #!self.training_pics = [picture.Pic(), picture.Pic(pic_name="ocr_insurance_card_train_2.jpg", contour_dimension_from_h=21, contour_dimension_to_h=28)]
        pass
    def train(self):
        """Method to train digits"""
        # Loop all images to train
        for training_pic in self.training_pics:
            im = cv2.imread(training_pic.pic_name)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 1)
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            samples = np.empty((0, 100))
            responses = []
            keys = [i for i in range(48, 58)]

            for cnt in contours:
                if cv2.contourArea(cnt) > (training_pic.contour_dimension_to_h * 2):
                    [x, y, w, h] = cv2.boundingRect(cnt)
                    # print("contour:" w, h)
                    if h > training_pic.contour_dimension_from_h and h < training_pic.contour_dimension_to_h:
                        cv2.rectangle(im, (x - 1, y - 1), (x + 1 + w, y + 1 + h), (0, 0, 255), 1)
                        roi = thresh[y:y + h, x:x + w]
                        roismall = cv2.resize(roi, (10, 10))
                        cv2.imshow('Training: Enter digits displayed in the red rectangle!', im)
                        key = cv2.waitKey(0)

                        if key == 27:  # (escape to quit)
                            self.save_data(samples, responses)
                            cv2.destroyAllWindows()
                            sys.exit()
                        elif key in keys:  # (append data)
                            responses.append(int(chr(key)))
                            sample = roismall.reshape((1, 100))
                            samples = np.append(samples, sample, 0)
        # Save collected data
        self.save_data(samples, responses)

    @staticmethod
    def save_data(samples, responses):
        """Method to save trained data"""
        responses = np.array(responses, np.float32)
        responses = responses.reshape((responses.size, 1))
        np.savetxt('ocr_training.data', samples)
        np.savetxt('ocr_responses.data', responses)
        print("training complete")

# Start the training process
if __name__ == '__main__':
    DigitRecognizerTraining().train()

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/Antoine/Pictures/SingleDigitrecognition.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.find_nearest(test,k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(accuracy)