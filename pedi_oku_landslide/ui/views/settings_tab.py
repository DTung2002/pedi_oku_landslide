from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox
from project.settings_store import load_settings, save_settings

class SettingsTab(QWidget):
    SCALES = [50, 75, 100, 125, 150, 200]

    def __init__(self, base_dir: str, on_scale_changed, parent=None):
        super().__init__(parent)
        self.base_dir = base_dir
        self.on_scale_changed = on_scale_changed
        self.settings = load_settings(base_dir)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("UI Scale (%)"))

        self.combo = QComboBox()
        for p in self.SCALES:
            self.combo.addItem(f"{p}%", p)

        idx = self.combo.findData(self.settings.ui_scale_percent)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

        self.combo.currentIndexChanged.connect(self._on_change)
        layout.addWidget(self.combo)

    def _on_change(self):
        val = int(self.combo.currentData())
        self.settings.ui_scale_percent = val
        save_settings(self.base_dir, self.settings)
        # notify main window to apply immediately
        if callable(self.on_scale_changed):
            self.on_scale_changed(val)
