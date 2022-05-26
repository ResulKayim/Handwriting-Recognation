from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtGui, QtCore
import sys
import cv2

from HandwritingRecognizer import HandwritingRecognizer

import os


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Handwriting Recognition')
        self.setGeometry(400, 100, 1200, 900)

        self.imageUri = ""
        self.strFinal = ""

        # Widgets
        self.label = QLabel(parent=self)
        self.selectedImage = cv2.imread('openImage.jpeg')
        self.shownImage = cv2.imread('openImage.jpeg')

        self.openImage = QPushButton(text='Open Image To Read', parent=self)
        self.readImage = QPushButton(text='Read Image', parent=self)
        self.saveImage = QPushButton(text='Save Result', parent=self)

        self.openImage.setFont(QFont('verdana', 10))
        self.readImage.setFont(QFont('verdana', 10))
        self.saveImage.setFont(QFont('verdana', 10))

        # Initialize model

        self.recognizer = HandwritingRecognizer()
        self.setImage(self.shownImage)
        self.initialize()

    def setStr(self, strFinal):
        self.strFinal = strFinal

    def initialize(self):
        self.openImage.move(350, 850)
        self.readImage.move(600, 850)
        self.saveImage.move(750, 850)

        self.label.setGeometry(0, 0, 1200, 800)

        self.openImage.clicked.connect(self.openImageFun)
        self.readImage.clicked.connect(self.readImageFun)
        self.saveImage.clicked.connect(self.saveImageFun)

        self.show()

    def setImage(self, shown_image):
        self.shownImage = shown_image
        shownImage = cv2.resize(self.shownImage, (1200, 800), interpolation=cv2.INTER_NEAREST)

        shownImage = cv2.cvtColor(shownImage, cv2.COLOR_BGR2BGRA)
        shownImage = QtGui.QImage(shownImage, shownImage.shape[1], shownImage.shape[0],
                                  QtGui.QImage.Format_RGB32).rgbSwapped()
        self.label.setPixmap(QtGui.QPixmap.fromImage(shownImage))
        self.label.setAlignment(QtCore.Qt.AlignCenter)

    def openImageFun(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open Image File', r"",
                                                "Image files (*.jpg *.jpeg *.png)")

        if file_name[0] != "":
            self.selectedImage = cv2.imread(file_name[0])
            self.setWindowTitle(file_name[1])
            self.setImage(self.selectedImage)
            self.imageUri = file_name[0]

    def readImageFun(self):
        if self.imageUri != "":
            self.recognizer.readImage(self.imageUri, self)

    def saveImageFun(self):
        if self.imageUri != "":
            try:
                os.mkdir(self.imageUri + "_result")
            except OSError:
                print("Creation of the directory failed")
            else:
                print("Successfully created the directory")

            cv2.imwrite(self.imageUri + "_result\\" + "resultImage.jpeg", self.shownImage)
            txtFile = open(self.imageUri + "_result\\" + "resultImage.txt", "w+")
            txtFile.write(self.strFinal)


app = QApplication(sys.argv)
window = Window()
sys.exit(app.exec_())
