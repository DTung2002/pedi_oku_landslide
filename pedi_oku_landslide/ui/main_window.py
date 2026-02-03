# pedi_oku_landslide/ui/main_window.py
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QLinearGradient, QPainter, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QTabWidget, QToolBar, QAction, QMessageBox,
    QDialog, QScrollArea
)
from PyQt5.QtGui import QFont, QFontDatabase
from .views.analyze_tab import AnalyzeTab
from .views.section_tab import SectionSelectionTab
from .views.curve_tab import CurveAnalyzeTab
from .views.settings_dialog import SettingsDialog
from pedi_oku_landslide.project.settings_store import load_settings
from PyQt5.QtGui import QIcon


def get_app_icon(base_dir: str) -> QIcon:
    candidates = [
        os.path.join(base_dir, "assets", "OKUYAMA Boring.png"),
        os.path.join(base_dir, "assets", "icons", "OKUYAMA Boring.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[Icon] Using window icon: {p}")
            return QIcon(p)
    return QIcon()

class CustomTitleBar(QWidget):
    def __init__(self, parent=None, base_dir: str = None):
        super().__init__(parent)
        self.setWindowIcon(get_app_icon(base_dir))
        self.parent = parent
        self.base_dir = base_dir
        self.setFixedHeight(60)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)

        # Logo bên trái
        self.logo_label = QLabel()

        # đường dẫn logo
        candidate_paths = []

        if self.base_dir:
            candidate_paths.append(
                os.path.join(self.base_dir, "assets", "OKUYAMA Boring.png")
            )
            candidate_paths.append(
                os.path.join(self.base_dir, "pedi_oku_landslide", "assets", "OKUYAMA Boring.png")
            )

        assets_dir_legacy = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "assets")
        )
        candidate_paths.append(os.path.join(assets_dir_legacy, "OKUYAMA Boring.png"))

        logo_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                logo_path = p
                print(f"[Icon] Using header logo: {p}")
                break

        if logo_path:
            pm = QPixmap(logo_path)
            if not pm.isNull():
                self.logo_label.setPixmap(
                    pm.scaledToHeight(50, Qt.SmoothTransformation)
                )
        else:
            self.logo_label.setText("PEDI OKU Landslide")
            self.logo_label.setStyleSheet(
                "color: #001F3F; font-size: 16px; font-weight: 700;"
            )

        layout.addWidget(self.logo_label)

        layout.addStretch(1)

        #Nút control
        btn_style = """
        QPushButton {
            background-color: #001F3F;
            color: white;
            border: none;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #023d7a;
        }
        QPushButton:pressed {
            background-color: #000c1f;
        }
        """

        btn_min = QPushButton("–")
        btn_min.setFixedSize(32, 24)
        btn_min.clicked.connect(lambda: parent.showMinimized())
        btn_min.setStyleSheet(btn_style)
        layout.addWidget(btn_min)

        btn_max = QPushButton("□")
        btn_max.setFixedSize(32, 24)
        btn_max.clicked.connect(self.toggle_max)
        btn_max.setStyleSheet(btn_style)
        layout.addWidget(btn_max)

        btn_close = QPushButton("✕")
        btn_close.setFixedSize(32, 24)
        btn_close.clicked.connect(parent.close)
        btn_close.setStyleSheet(btn_style)
        layout.addWidget(btn_close)


    # Gradient background
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ffffff"))  # header trắng

    # Cho phép kéo cửa sổ
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.parent.move(self.parent.pos() + event.globalPos() - self.dragPos)
            self.dragPos = event.globalPos()

    # Double-click để maximize / restore
    def mouseDoubleClickEvent(self, event):
        self.toggle_max()

    def toggle_max(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()

def load_inter_fonts(base_dir: str):
    from PyQt5.QtGui import QFontDatabase, QFont
    import os

    # Các vị trí có thể chứa fonts
    candidate_dirs = [
        os.path.join(base_dir, "assets", "fonts"),
        os.path.join(base_dir, "pedi_oku_landslide", "assets", "fonts"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts"),
    ]

    font_dir = None
    for d in candidate_dirs:
        if os.path.isdir(d):
            font_dir = d
            print(f"[Font] Using font dir: {d}")
            break

    if font_dir is None:
        print("[Font] WARNING: Inter fonts directory not found. Using default font.")
        return QFont().defaultFamily()

    font_db = QFontDatabase()
    family_name = None

    for fname in os.listdir(font_dir):
        if not fname.lower().endswith((".otf", ".ttf")):
            continue
        path = os.path.join(font_dir, fname)
        fid = font_db.addApplicationFont(path)
        if fid != -1:
            families = font_db.applicationFontFamilies(fid)
            if families:
                family_name = families[0]
                print(f"[Font] Loaded: {fname} -> {family_name}")
        else:
            print(f"[Font] Failed to load font: {fname}")

    if not family_name:
        print("[Font] WARNING: no Inter fonts loaded, using default.")
        return QFont().defaultFamily()

    return family_name or "Inter"

class MainWindow(QMainWindow):
    def __init__(self, app_root: str, internal_root: str, app: QApplication):
        super().__init__()
        self.app_root = app_root
        self.internal_root = internal_root
        self.app = app
        self.settings = load_settings(self.app_root)
        self.setWindowIcon(get_app_icon(self.internal_root))
        # Root central
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)  # vẫn giữ nền trắng
        self.setCentralWidget(central)
        titlebar = CustomTitleBar(self, base_dir=self.internal_root)
        root_layout.addWidget(titlebar)

        #Header (white)
        header = QWidget(objectName="headerBar")
        h = QHBoxLayout(header)
        h.setContentsMargins(8, 8, 8, 8)
        h.addStretch(1)

        # Nút New Session
        self.btn_new_session = QPushButton("New Session")
        self.btn_new_session.setFixedHeight(28)
        self.btn_new_session.clicked.connect(self._on_new_session_clicked)
        h.addWidget(self.btn_new_session)

        # Nút Settings
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.setFixedHeight(28)
        self.btn_settings.clicked.connect(self._on_open_settings_dialog)
        h.addWidget(self.btn_settings)

        root_layout.addWidget(header)

        # Bọc mỗi tab bằng QScrollArea
        self.tabs = QTabWidget()

        # Analyze tab
        self.analyze_tab = AnalyzeTab(self.app_root)
        analyze_scroll = QScrollArea()
        analyze_scroll.setWidget(self.analyze_tab)
        analyze_scroll.setWidgetResizable(True)
        self.tabs.addTab(analyze_scroll, "Input Analyze")

        # Section tab
        self.section_tab = SectionSelectionTab(self.app_root)
        section_scroll = QScrollArea()
        section_scroll.setWidget(self.section_tab)
        section_scroll.setWidgetResizable(True)
        self.tabs.addTab(section_scroll, "Section Selection")

        # Curve tab
        self.curve_tab = CurveAnalyzeTab(self.app_root)
        curve_scroll = QScrollArea()
        curve_scroll.setWidget(self.curve_tab)
        curve_scroll.setWidgetResizable(True)
        self.tabs.addTab(curve_scroll, "Curve Analyze")

        root_layout.addWidget(self.tabs)

        # Lưu index để enable/disable
        self._idx_analyze = self.tabs.indexOf(analyze_scroll)
        self._idx_section = self.tabs.indexOf(section_scroll)
        self._idx_curve = self.tabs.indexOf(curve_scroll)

        # Khóa tab 2–3 khi mở app:
        self.tabs.setCurrentIndex(self._idx_analyze)

        # Kết nối tín hiệu giữa các tab
        if hasattr(self.analyze_tab, "vectors_rendered"):
            self.analyze_tab.vectors_rendered.connect(self._on_vectors_ready)
        else:
            self.analyze_tab.vectors_ready.connect(self._on_vectors_ready)

        if hasattr(self.section_tab, "sections_confirmed"):
            self.section_tab.sections_confirmed.connect(self._on_sections_confirmed)

        self.setWindowTitle("PEDI Landslide Analyzer")
        self.resize(1000, 700)
        font_family = load_inter_fonts(self.internal_root)
        self.font_family = font_family
        self.apply_scale(self.settings.ui_scale_percent)

    #  Flow handlers
    def _on_vectors_ready(self, project: str, run_label: str, run_dir: str) -> None:
        try:
            if getattr(self, "section_tab", None):
                # Lấy step/scale hiện tại từ AnalyzeTab (UI1)
                vec_step = None
                vec_scale = None
                if hasattr(self.analyze_tab, "spin_vec_step"):
                    try:
                        vec_step = int(self.analyze_tab.spin_vec_step.value())
                    except Exception:
                        pass
                if hasattr(self.analyze_tab, "spin_vec_scale"):
                    try:
                        vec_scale = float(self.analyze_tab.spin_vec_scale.value())
                    except Exception:
                        pass

                self.section_tab.set_context(
                    project,
                    run_label,
                    run_dir,
                    vec_step=vec_step,
                    vec_scale=vec_scale,
                )

            if getattr(self, "curve_tab", None):
                self.curve_tab.set_context(project, run_label, run_dir)

            if hasattr(self, "_idx_section"):
                self.tabs.setCurrentIndex(self._idx_section)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot open Section tab:\n{e}")


    def _on_sections_confirmed(self, project: str, run_label: str, run_dir: str) -> None:
        try:
            if getattr(self, "curve_tab", None):
                self.curve_tab.set_context(project, run_label, run_dir)
            # Chuyển sang Curve Analyze ngay sau khi confirm
            if hasattr(self, "_idx_curve"):
                self.tabs.setCurrentIndex(self._idx_curve)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot open Curve tab:\n{e}")

    def _on_new_session_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "New Session",
            "Clear current session and start a new one?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._reset_session()

    # Settings / New Session
    def _on_open_settings_dialog(self):
        dlg = SettingsDialog(self.app_root, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        # Scale
        self.apply_scale(dlg.selected_scale())


    def _reset_session(self) -> None:
        if hasattr(self, "analyze_tab"):
            self.analyze_tab.reset_session()
        if hasattr(self, "section_tab"):
            self.section_tab.reset_session()
        if hasattr(self, "curve_tab"):
            self.curve_tab.reset_session()
        self.tabs.setCurrentIndex(self._idx_analyze)

    #  UI scale & stylesheet
    def apply_scale(self, percent: int) -> None:
        base_pt = 10
        pt = max(8, int(base_pt * percent / 100.0))

        self.app.setFont(QFont(self.font_family, pt))

        self.app.setStyleSheet(f"""
            /* ================== BASE TYPOGRAPHY ================== */
            QMainWindow, QWidget {{
                font-size: {pt}pt;
                background-color: #ffffff;
            }}

            /* Toàn bộ text: Inter, mảnh (300) */
            QWidget, QLabel, QGroupBox, QTabBar::tab {{
                font-family: "{self.font_family}";
                font-weight: 300;
            }}

            /* ================== HEADER ================== */
            #headerBar {{
                background-color: #ffffff;
            }}

            QLabel#headerTitle {{
                background: #ffffff;
                color: #444;
                font-weight: 400;
            }}

            QAbstractScrollArea,
            QScrollArea,
            QScrollArea > QWidget {{
                background-color: #ffffff;
            }}

            /* ================== TABS (Input / Section / Curve) ================== */
            QTabWidget::pane {{
                border: none;
                top: 0px;
            }}

            QTabBar::tab {{
                background: #056832;
                color: #ffffff;
                padding: 6px 22px;
                margin-right: 4px;
                min-width: 120px;
                border: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                font-weight: 400;
                font-size: {pt + 1}px;
            }}

            QTabBar::tab:selected {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f5a623, stop:1 #f76b1c
                );
                color: #ffffff;
            }}

            QTabBar::tab:hover:!selected {{
                background-color: #7bb455;
            }}

            QTabBar::tab:!selected {{
                background-color: #649540;
                color: #f0f0f0;
            }}

            /* ================== GROUPBOX = cards ================== */
            QGroupBox {{
                background: #f8f9fb;
                border: 1px solid #e0e4ec;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                margin-left: 4px;
                margin-top: 2px;
                background: transparent;
                color: #001F3F;
                font-weight: 400;
            }}

            /* ================== FORM FIELDS (macOS-style) ================== */
            QLineEdit,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox,
            QPlainTextEdit,
            QTextEdit {{
                border: 1px solid #d0d6e2;
                border-radius: 6px;
                padding: 4px 6px;
                background: #ffffff;
                font-weight: 300;
            }}

            QLineEdit:focus,
            QComboBox:focus,
            QSpinBox:focus,
            QDoubleSpinBox:focus,
            QPlainTextEdit:focus,
            QTextEdit:focus {{
                border: 1px solid #056832;
                box-shadow: 0 0 0 2px rgba(5, 104, 50, 0.15);
            }}

            /* ================== TABLES ================== */
            QTableView {{
                gridline-color: #dde2ec;
                selection-background-color: rgba(5, 104, 50, 0.12);
                selection-color: #001F3F;
                background: #ffffff;
            }}

            QTableView::item {{
                padding: 2px 4px;
            }}

            QTableView::item:alternate {{
                background: #f9fafc;
            }}

            QHeaderView::section {{
                background: #f1f3f7;
                color: #001F3F;
                border: none;
                border-bottom: 1px solid #dde2ec;
                padding: 4px 6px;
                font-weight: 400;
            }}

            /* ================== GRAPHICS VIEW ================== */
            QGraphicsView {{
                border: 1px solid #dde2ec;
                border-radius: 8px;
                background: #ffffff;   /* nền gần đen cho hillshade */
            }}

            /* ================== BUTTONS (gradient xanh) ================== */
            QPushButton,
            QToolButton {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6fa34a,
                    stop:1 #056832
                );
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 400;           /* đậm hơn body chút xíu */
                font-size: {pt + 2}px;        /* chữ nút to hơn */
            }}

            QPushButton:hover,
            QToolButton:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8bcf6a,
                    stop:1 #079c4f
                );
            }}

            QPushButton:pressed,
            QToolButton:pressed {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4f7a34,
                    stop:1 #034d26
                );
            }}

            QPushButton:disabled,
            QToolButton:disabled {{
                background: #c8d8bf;
                color: #eeeeee;
            }}

            /* ================== TOOLBAR / MENUBAR ================== */
            QToolBar, QMenuBar {{
                background: #ffffff;
                border: none;
            }}

            /* ================== SCROLLBAR – macOS style A (navy gradient) ================== */

            /* Vertical */
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                margin: 2px 0 2px 0;
            }}

            QScrollBar::handle:vertical {{
                min-height: 20px;
                border-radius: 3px;
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #023d7a,
                    stop:1 #001F3F
                );
            }}

            QScrollBar::handle:vertical:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0356b0,
                    stop:1 #023d7a
                );
            }}

            QScrollBar::handle:vertical:pressed {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #001F3F,
                    stop:1 #000c1f
                );
            }}

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
                background: transparent;
                border: none;
            }}

            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
            }}

            /* Horizontal */
            QScrollBar:horizontal {{
                background: transparent;
                height: 6px;
                margin: 0 2px 0 2px;
            }}

            QScrollBar::handle:horizontal {{
                min-width: 20px;
                border-radius: 3px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #023d7a,
                    stop:1 #001F3F
                );
            }}

            QScrollBar::handle:horizontal:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0356b0,
                    stop:1 #023d7a
                );
            }}

            QScrollBar::handle:horizontal:pressed {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #001F3F,
                    stop:1 #000c1f
                );
            }}

            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {{
                background: transparent;
                width: 0px;
                border: none;
            }}

            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {{
                background: transparent;
            }}
        """)


# Entry point
def run_ui(app_root: str, internal_root: str):
    import sys
    app = QApplication(sys.argv)
    win = MainWindow(app_root, internal_root, app)
    win.showMaximized()
    sys.exit(app.exec_())
