import cv2
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.keras.models import load_model
import pandas as pd


class HandwritingRecognizer:
    def __init__(self):
        self.model = load_model('model.h5')
        self.strFinal = str('')
        self.max_gap = 50
        self.last_x = 0
        self.last_y = 0
        self.contourNumber = 0
        self.mapp = pd.read_csv("Data/emnist-balanced-mapping.txt", delimiter=' ', index_col=0, header=None,
                                squeeze=True)

    def readImage(self, image_uri, user_interface):
        self.strFinal = ""
        self.image = cv2.imread(image_uri)
        grey = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(grey.copy(), 85, 255, cv2.THRESH_BINARY_INV)

        contours, b = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        # sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1] * self.image.shape[1] + cv2.boundingRect(ctr)[0])

        # Sort Contours left to right and top to bottom
        cList = []
        for i in contours:
            cList.append(cv2.boundingRect(i))

        nparray = np.array(cList)

        maxHeight = nparray.max(axis=0)[3]
        nearest = maxHeight * 1.4

        sorted_contours = sorted(contours,
                                 key=lambda ctr: [int(nearest * round(float(cv2.boundingRect(ctr)[1]) / nearest)),
                                                  cv2.boundingRect(ctr)[0]])

        counter = 0
        sumGap = 0
        lastX = 0
        lastY = 0
        for c in sorted_contours:
            x, y, w, h = cv2.boundingRect(c)
            if counter == 0:
                counter += 1

            elif y > lastY:
                pass
            else:
                sumGap += x - lastX
                counter += 1

            lastX = x + w
            lastY = y + h

        self.max_gap = int(sumGap / (counter - 1))
        print(f"Max Gap : {self.max_gap} : Counter {counter}")

        is_first = True
        for c in sorted_contours:
            x, y, w, h = cv2.boundingRect(c)

            if self.contourNumber != 0 and x - self.last_x > self.max_gap:
                self.strFinal += ' '
                is_first = True

            if self.contourNumber != 0 and y > self.last_y:
                self.strFinal += '\n'
                is_first = True

            self.last_x = x + w
            self.last_y = y + h

            # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
            cv2.rectangle(self.image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = thresh[y:y + h, x:x + w]

            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit, (20, 20))

            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((4, 4), (4, 4)), "constant", constant_values=0)

            padded_digit = padded_digit.astype('float32') / 255

            # Predicting
            prediction = self.model.predict(padded_digit.reshape(1, 28, 28, 1))

            # cv2.putText(self.image, str(chr(ord('A') + np.argmax(prediction))), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.6, (36, 255, 12), 2)

            cv2.putText(self.image, str(chr(self.mapp.iloc[np.argmax(prediction)])), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
            cv2.putText(self.image, "%" + str(round(prediction[0][np.argmax(prediction)] * 100, 1)), (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

            # self.strFinal += str(chr(ord('A') + np.argmax(prediction)))

            asciCode = self.mapp.iloc[np.argmax(prediction)]

            if is_first:
                if ord('z') > asciCode > ord('a'):
                    self.strFinal += str(chr(asciCode - ord('a') + ord('A')))
                else:
                    self.strFinal += str(chr(asciCode))
            else:
                if ord('Z') > asciCode > ord('A'):
                    self.strFinal += str(chr(asciCode - ord('A') + ord('a')))
                else:
                    self.strFinal += str(chr(asciCode))

            plt.imshow(self.image, cmap="gray")
            plt.show()

            self.contourNumber += 1
            is_first = False

        print(f"{self.strFinal}")
        user_interface.setStr(self.strFinal)
        user_interface.setImage(self.image)
