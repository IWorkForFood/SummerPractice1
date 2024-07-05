import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import design
from PyQt5.QtWidgets import QFileDialog, QDial
from PyQt5.QtGui import QImage, QPixmap, QTransform
import cv2
import imutils
from PyQt5.QtCore import Qt
import numpy as np

import os


class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):

        super().__init__()
        self.setupUi(self)
        self.filename = None
        self.tmp = None
        self.pixmap = None

        self.pushButton.clicked.connect(self.loadImage)

        #задаем параметры ползунка
        self.dial.setRange(0, 360)
        self.dial.setValue(0)
        self.dial.setWrapping(False)
        self.dial.setSingleStep(1)
        self.dial.valueChanged.connect(self.update_angle)

        self.spinBox.setMinimum(0)
        self.spinBox_2.setMinimum(1)
        self.spinBox_3.setMinimum(0)
        self.spinBox_4.setMinimum(1)
        self.pushButton_8.clicked.connect(self.cutImg)

        self.radioButton.toggled.connect(self.addCircle)

        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider_3.setMinimum(0)
        self.horizontalSlider.valueChanged.connect(self.change_center_coord)
        self.horizontalSlider_2.valueChanged.connect(self.change_center_coord)
        self.horizontalSlider_3.valueChanged.connect(self.change_radius)

        self.R = 10
        self.coords = (10, 10)

        self.pushButton_3.clicked.connect(self.get_red_ch)
        self.pushButton_4.clicked.connect(self.get_green_ch)
        self.pushButton_5.clicked.connect(self.get_blue_ch)
        self.pushButton_6.clicked.connect(self.get_norm_ch)

        self.blue = self.green = self.red = self.a_c = None

        self.pushButton_2.clicked.connect(self.make_photo)

    def loadImage(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, 'Открыть файл', '/home', 'Изображения (*.png *.xpm *.jpg)')
        if self.fname:

            self.pixmap = QPixmap(self.fname)
            scaled_pixmap = self.pixmap.scaled(700, self.pixmap.height(), Qt.KeepAspectRatio)
            self.label.setPixmap(scaled_pixmap)
            self.spinBoxValues()
            self.set_slider_m_values()

        print(type(self.fname), self.fname)
        file_path = self.fname
        stream = open(file_path, 'rb')
        bytes = bytearray(stream.read())
        array = np.asarray(bytes, dtype=np.uint8)
        self.bgrImage = self.image_copy = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

    def make_photo(self):
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self.label.setText("Не удается подключиться к камере")
        succ, frame = capture.read()
        if succ:
            self.bgrImage = self.image_copy = frame
            self.from_cv_to_pq(frame)
            self.spinBoxValues()
            self.set_slider_m_values()
        else:
            self.label.setText("Ошибка захвата изображения")


    def update_angle(self):
        if not self.pixmap:
            return
        angle = self.dial.value()
        transform = QTransform().rotate(angle)
        rotated_image = self.pixmap.transformed(transform, Qt.SmoothTransformation)
        self.label.setPixmap(rotated_image.scaled(700, self.pixmap.height(), Qt.KeepAspectRatio))

    def cutImg(self):
        if not self.pixmap:
            return
        cropped_image = self.image_copy[self.spinBox_3.value():self.spinBox_4.value(), self.spinBox.value():self.spinBox_2.value()]
        cv2.imshow('Cutted', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def addCircle(self):
        if not self.pixmap:
            return

        if self.radioButton.isChecked():
            self.image_copy = self.bgrImage.copy()
            cv2.circle(self.image_copy, self.coords, self.R, (0, 0, 255), -1)
            image_rgb = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch*w
            q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            print(self.image_copy.shape)
            self.pixmap = QPixmap(QtGui.QPixmap.fromImage(q_image))
            self.label.setPixmap(self.pixmap.scaled(700, self.pixmap.height(), Qt.KeepAspectRatio))
        else:
            mask = np.zeros_like(self.bgrImage)
            cv2.circle(mask, self.coords, self.R, (255, 255, 255), -1)
            inverse_mask = cv2.bitwise_not(mask)
            circle_removed = cv2.bitwise_and(self.image_copy, inverse_mask)

            final_image = cv2.bitwise_or(circle_removed, cv2.bitwise_and(self.bgrImage, mask))
            self.image_copy = final_image
            rgb_img = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QtGui.QPixmap.fromImage(q_image)
            self.label.setPixmap(self.pixmap.scaled(700, self.pixmap.height(), Qt.KeepAspectRatio))
        self.update_angle()

    def change_center_coord(self):
        if not self.pixmap:
            return
        self.coords = (self.horizontalSlider.value(), self.horizontalSlider_2.value())
        self.addCircle()
        self.update_angle()
    def change_radius(self):
        if not self.pixmap:
            return
        self.R = self.horizontalSlider_3.value()
        self.addCircle()
        self.update_angle()
    def spinBoxValues(self):

        self.spinBox.setMaximum(self.pixmap.width()-1)
        self.spinBox_2.setMaximum(self.pixmap.width())
        self.spinBox_3.setMaximum(self.pixmap.height()-1)
        self.spinBox_4.setMaximum(self.pixmap.height())
        self.update_angle()

    def set_slider_m_values(self):

        self.horizontalSlider.setMaximum(self.pixmap.width())
        self.horizontalSlider_2.setMaximum(self.pixmap.height())
        self.horizontalSlider_3.setMaximum(min(self.pixmap.height()//2, self.pixmap.height()//2))
        self.update_angle()

    def get_red_ch(self):
        if not self.pixmap:
            return
        if len(cv2.split(self.image_copy)) < 3:
            self.image_copy = cv2.merge((self.blue, self.green, self.red))
        if len(cv2.split(self.bgrImage)) < 3:
            self.bgrImage = cv2.merge((self.bgr_b, self.bgr_g, self.bgr_r))
        self.blue, self.green, self.red = cv2.split(self.image_copy)[:3]
        self.bgr_b, self.bgr_g, self.bgr_r = cv2.split(self.bgrImage)[:3]
        self.bgrImage = self.bgr_r
        self.image_copy = self.red
        self.from_cv_to_pq(self.red)
        self.addCircle()
        self.update_angle()

    def get_green_ch(self):
        if not self.pixmap:
            return
        if len(cv2.split(self.image_copy)) < 3:
            self.image_copy = cv2.merge((self.blue, self.green, self.red))
        if len(cv2.split(self.bgrImage)) < 3:
            self.bgrImage = cv2.merge((self.bgr_b, self.bgr_g, self.bgr_r))
        self.blue, self.green, self.red = cv2.split(self.image_copy)[:3]
        self.bgr_b, self.bgr_g, self.bgr_r = cv2.split(self.bgrImage)[:3]
        self.bgrImage = self.bgr_g
        self.image_copy = self.green
        self.from_cv_to_pq(self.green)
        self.addCircle()
        self.update_angle()

    def get_blue_ch(self):
        if not self.pixmap:
            return
        if len(cv2.split(self.image_copy)) < 3:
            self.image_copy = cv2.merge((self.blue, self.green, self.red))
        if len(cv2.split(self.bgrImage)) < 3:
            self.bgrImage = cv2.merge((self.bgr_b, self.bgr_g, self.bgr_r))
        self.blue, self.green, self.red = cv2.split(self.image_copy)[:3]
        self.bgr_b, self.bgr_g, self.bgr_r = cv2.split(self.bgrImage)[:3]
        self.bgrImage = self.bgr_b
        self.image_copy = self.blue
        self.from_cv_to_pq(self.blue)
        self.addCircle()
        self.update_angle()

    def get_norm_ch(self):
        if not self.pixmap:
            return
        self.norm_ch = cv2.merge((self.red, self.green, self.blue))
        self.bgrImage = cv2.merge((self.bgr_b, self.bgr_g, self.bgr_r))
        self.image_copy = self.bgrImage
        self.from_cv_to_pq(self.norm_ch[:,:,::-1])
        self.addCircle()
        self.update_angle()
    def from_cv_to_pq(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(q_image)
        self.label.setPixmap(self.pixmap.scaled(700, self.pixmap.height(), Qt.KeepAspectRatio))
        self.update_angle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()