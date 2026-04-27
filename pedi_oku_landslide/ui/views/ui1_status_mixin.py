from datetime import datetime

from PyQt5.QtWidgets import QMessageBox


class UI1StatusMixin:
    @staticmethod
    def _status_brief(msg: str, fallback: str) -> str:
        skip_prefixes = (
            "project:",
            "run:",
            "output:",
            "folder:",
            "outputs:",
            "- dx:",
            "- dy:",
            "- dz:",
            "- mask:",
            "- polygons:",
            "- dX:",
            "- dY:",
            "- dZ:",
        )
        for raw in str(msg or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if any(low.startswith(prefix.lower()) for prefix in skip_prefixes):
                continue
            if "\\" in line or "/" in line:
                if ":" in line:
                    line = line.split(":", 1)[0].strip()
                else:
                    continue
            return line
        return fallback

    def _append_status(self, text: str) -> None:
        """
        Ghi 1 dòng vào khung Status; nếu chưa có self.status_text thì in ra console.
        Tự động cuộn xuống cuối.
        """
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        box = getattr(self, "status_text", None)
        if box is not None:
            try:
                box.append(line)
                box.moveCursor(box.textCursor().End)
                return
            except Exception:
                pass
        print(line)

    def _info(self, msg: str) -> None:
        return

    def _ok(self, msg: str) -> None:
        self._append_status(f"OK: {self._status_brief(msg, 'Completed.')}")

    def _warn(self, msg: str) -> None:
        self._append_status(f"ERROR: {self._status_brief(msg, 'Action required.')}")
        QMessageBox.warning(self, "Warning", msg)

    def _err(self, msg: str) -> None:
        self._append_status(f"ERROR: {self._status_brief(msg, 'Error.')}")
        QMessageBox.critical(self, "Error", msg)
