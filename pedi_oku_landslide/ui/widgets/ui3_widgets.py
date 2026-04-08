from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class KeyboardOnlySpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class KeyboardOnlyDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()
