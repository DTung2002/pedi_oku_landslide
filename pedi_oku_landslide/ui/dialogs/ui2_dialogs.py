from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class AutoLineDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Line Generation")

        layout = QVBoxLayout(self)
        grp = QGroupBox("Automatic Lines")
        grid = QGridLayout(grp)

        self.main_num = QSpinBox()
        self.main_num.setRange(0, 200)
        self.main_num.setSingleStep(2)
        self.main_num.setValue(4)

        self.main_off = QDoubleSpinBox()
        self.main_off.setRange(0.0, 1e6)
        self.main_off.setDecimals(2)
        self.main_off.setValue(20.0)

        self.cross_num = QSpinBox()
        self.cross_num.setRange(0, 200)
        self.cross_num.setSingleStep(2)
        self.cross_num.setValue(4)

        self.cross_off = QDoubleSpinBox()
        self.cross_off.setRange(0.0, 1e6)
        self.cross_off.setDecimals(2)
        self.cross_off.setValue(20.0)

        grid.addWidget(QLabel("Main lines:"), 0, 0)
        grid.addWidget(QLabel("Line number:"), 0, 1)
        grid.addWidget(self.main_num, 0, 2)
        grid.addWidget(QLabel("Offset (m):"), 0, 3)
        grid.addWidget(self.main_off, 0, 4)

        grid.addWidget(QLabel("Cross lines:"), 1, 0)
        grid.addWidget(QLabel("Line number:"), 1, 1)
        grid.addWidget(self.cross_num, 1, 2)
        grid.addWidget(QLabel("Offset (m):"), 1, 3)
        grid.addWidget(self.cross_off, 1, 4)

        layout.addWidget(grp)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
