from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QFileDialog

class FilePicker(QWidget):
    def __init__(self, title: str, filter_text: str, parent=None):
        super().__init__(parent)
        self.filter_text = filter_text
        self._title = title
        self.path = ""  # dùng string thay vì None

        self.btn = QPushButton(title)
        self.lbl = QLabel("No file selected")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.btn)
        lay.addWidget(self.lbl, 1)

        self.btn.clicked.connect(self.pick)

    def set_path(self, p: str) -> None:
        """Set path + update label UI."""
        self.path = (p or "")
        self.lbl.setText(self.path if self.path else "No file selected")

    def clear(self) -> None:
        """Reset về trạng thái ban đầu."""
        self.set_path("")

    def pick(self):
        p, _ = QFileDialog.getOpenFileName(self, self.btn.text(), "", self.filter_text)
        if p:
            self.set_path(p)   # dùng set_path để đồng bộ UI
