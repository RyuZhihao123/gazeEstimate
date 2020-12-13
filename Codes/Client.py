from PyQt5 import QtWidgets
from ui.MainWindow import Ui_MainWindow


class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.btnPredict.clicked.connect(self.ButtonClickedEvent)

    def ButtonClickedEvent(self):
        print("adasdas")

        import cv2
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()
        cv2.imshow("adasdas",frame)
        cv2.waitKey(0)
        print(frame.shape)




if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())