from PyQt5.QtWidgets import QComboBox, QHBoxLayout


def HBox():
    return QHBoxLayout()


class NoWheelComboBox(QComboBox):
    """Ignore mouse wheel to avoid accidental role changes while scrolling."""

    def wheelEvent(self, event):
        event.ignore()
