import requests
from PyQt5 import QtGui, QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPainter, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QPushButton, QWidget, QDialog, QLabel
import sys
from googletrans import Translator

import numpy as np
import cv2

from mss import mss


from PIL import Image
import pytesseract

def detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('image', img)
    mask = cv2.inRange(hsv, (100,55,55), (110,255, 255))
    blank_image = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    blank_image[:,:,0] = 255
    blank_image[:,:,1] = 255
    blank_image[:,:,2] = 255
    dst1 = cv2.bitwise_and(blank_image, blank_image, mask=mask)
    dst1 = cv2.bitwise_not(dst1)
    cv2.imwrite('dst1.png', dst1)
    tessdata_dir_config = r'--tessdata-dir "."'
    return dst1, pytesseract.image_to_string(dst1, lang='chi_sim',  config=tessdata_dir_config)

target_rect = {'top': 0, 'left': 0, 'width': 10, 'height': 10}

class SettingWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setGeometry(
            QtWidgets.QStyle.alignedRect(
                QtCore.Qt.LeftToRight, QtCore.Qt.AlignCenter,
                QtCore.QSize(220, 32),
                QtWidgets.qApp.desktop().availableGeometry()
        ))
        self.setWindowOpacity(0.3)
        
    def mousePressEvent(self, event):
        target_rect['top'] = self.pos().y()
        target_rect['left'] = self.pos().x()
        target_rect['width'] = self.frameGeometry().width()
        target_rect['height'] = self.frameGeometry().height()
        print(target_rect)
        self.close()


class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()

# while True:
#     x,y,w,h = cv2.getWindowImageRect('Frame')
#
#     cv2.imshow('image', np.array(sct_img))
#     # print(detect(np.array(sct_img)))
#     if (cv2.waitKey(1) & 0xFF) == ord('q'):
#         cv2.destroyAllWindows()
#         break


app = QApplication(sys.argv)

layout = QVBoxLayout()
button = QPushButton('start')
text = QLabel()
text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
text2 = QLabel()
img = ImageWidget()

layout.addWidget(button)
layout.addWidget(img)
layout.addWidget(text)
layout.addWidget(text2)


def display_image(img, display, scale=4):
    disp_size = img.shape[1]//scale, img.shape[0]//scale
    disp_bpl = disp_size[0] * 3
    if scale > 1:
        img = cv2.resize(img, disp_size,
                         interpolation=cv2.INTER_CUBIC)
    qimg = QImage(img.data, disp_size[0], disp_size[1],
                  disp_bpl, QImage.Format_RGB888)
    display.setImage(qimg)

sct = mss()

translator = Translator()

def start():
    sct_img = sct.grab(target_rect)
    x, y = detect(np.array(sct_img))
    y = y.replace(" ", "")
    text.setText(y.replace(" ", ""))
    text2.setText(get_nmt_translate(y))
    display_image(x, img)

CLIENT_ID = "neOi2CcEC1DiSvoBE03f"
CLIENT_SECRET = "5A5IBd1uMF"

def get_nmt_translate(context):
    try:
        url = "https://openapi.naver.com/v1/papago/n2mt"
        headers= {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
        params = {"source": "zh-CN", "target": "ko", "text": context}
        response = requests.post(url, headers=headers, data=params)
        res = response.json()
        return res['message']['result']['translatedText']
    except:
        return "번역 실패"

button.clicked.connect(start)

window = QWidget()
window.setLayout(layout)

window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint |     QtCore.Qt.X11BypassWindowManagerHint)
window.show()

helpwindow = SettingWindow()
helpwindow.show()

app.exec_()

