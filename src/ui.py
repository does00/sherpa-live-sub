"""实时字幕窗口：深色大字显示，支持置顶悬浮在浏览器视频上。"""

import os
import sys
import json
import threading
import time

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QProgressDialog, QPushButton, QSlider, QStatusBar,
    QTextEdit, QVBoxLayout, QWidget,
)

from . import audio as audio_mod
from . import __version__
from .engine import StreamEngine
from .models import MODELS, ensure_models, models_ready


def project_dir() -> str:
    """项目目录（源码运行就是解压出来的 sherpa-live-sub 文件夹）。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_MODEL_DIR_CACHE = ""


def model_base_dir() -> str:
    """模型根目录：项目目录下的 models 文件夹，整个文件夹拷走即走。

    项目目录不可写时（极少数情况），回退到用户目录的老位置。
    """
    global _MODEL_DIR_CACHE
    if _MODEL_DIR_CACHE:
        return _MODEL_DIR_CACHE
    if getattr(sys, "frozen", False):  # 打包后的 exe：模型放 exe 旁边的 data 目录
        d = os.path.join(os.path.dirname(sys.executable), "data")
    else:
        d = os.path.join(project_dir(), "models")
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            d = ""
        if not d or not os.access(d, os.W_OK):
            d = os.path.join(os.path.expanduser("~"), ".sherpa-live-sub")
    _MODEL_DIR_CACHE = d
    return d


def app_data_dir() -> str:
    """兼容老代码，等同于 model_base_dir()。"""
    return model_base_dir()


def _settings_path() -> str:
    return os.path.join(project_dir(), "settings.json")


def load_settings() -> dict:
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(s: dict):
    try:
        with open(_settings_path(), "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def subtitle_dir() -> str:
    """字幕保存目录：用户自选 > 默认项目目录下的 subtitles。"""
    custom = load_settings().get("subtitle_dir", "")
    if custom and os.path.isdir(custom):
        return custom
    return os.path.join(project_dir(), "subtitles")


def log(msg: str):
    """运行日志：记到项目目录的 debug.log，方便排查一闪而过的问题。"""
    try:
        d = project_dir()
        with open(os.path.join(d, "debug.log"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


class Bridge(QObject):
    """工作线程 -> UI 线程的信号桥。"""

    partial = Signal(str)
    final = Signal(str, float, float)  # 文本，开始秒，结束秒
    status = Signal(str)
    stopped = Signal()  # 引擎线程真正退出
    error = Signal(str)  # 工作线程异常死亡（比如音频设备被拔掉）


class MainWindow(QMainWindow):
    MAX_LINES = 200

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"实时字幕 v{__version__}")
        self.resize(640, 480)

        self._engine: StreamEngine | None = None
        self._bridge = Bridge()
        self._bridge.partial.connect(self._on_partial)
        self._bridge.final.connect(self._on_final)
        self._bridge.status.connect(self._on_status)
        self._bridge.stopped.connect(self._on_stopped)
        self._bridge.error.connect(self._on_error)
        self._stopping = False
        self._t0 = 0.0
        self._dl_pending = None
        self._subfile = None
        self._subfile_path = ""
        self._subfile_index = 0

        self._build_ui()
        self._set_running(False)

    # ---------- 界面 ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(6)

        # 当前正在说的一句（实时刷新）
        self.partial_label = QLabel("点击「开始字幕」，把网页的声音变成实时字幕。")
        self.partial_label.setWordWrap(True)
        self.partial_label.setStyleSheet(
            "background:#1c1c1e; color:#ffd60a; border-radius:8px; padding:10px;"
        )
        self.partial_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.partial_label.setMinimumHeight(64)
        layout.addWidget(self.partial_label)

        # 定稿字幕历史
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.history.setStyleSheet(
            "background:#101012; color:#f2f2f2; border-radius:8px; padding:6px;"
        )
        layout.addWidget(self.history, 1)

        # 控制条
        bar = QHBoxLayout()
        self.source_combo = QComboBox()
        for kind, name in audio_mod.available_sources():
            self.source_combo.addItem(name, kind)
        bar.addWidget(QLabel("声音来源:"))
        bar.addWidget(self.source_combo)

        self.start_btn = QPushButton("开始字幕")
        self.start_btn.clicked.connect(self._toggle)
        bar.addWidget(self.start_btn)

        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self._clear)
        bar.addWidget(clear_btn)

        self.pin_check = QCheckBox("窗口置顶")
        self.pin_check.toggled.connect(self._toggle_pin)
        bar.addWidget(self.pin_check)

        bar.addWidget(QLabel("字号:"))
        self.font_slider = QSlider(Qt.Horizontal)
        self.font_slider.setRange(12, 40)
        self.font_slider.setValue(20)
        self.font_slider.setFixedWidth(110)
        self.font_slider.valueChanged.connect(self._apply_font_size)
        bar.addWidget(self.font_slider)
        bar.addStretch(1)
        layout.addLayout(bar)

        # 字幕保存位置
        save_bar = QHBoxLayout()
        save_bar.addWidget(QLabel("字幕保存到:"))
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setReadOnly(True)
        self.save_path_edit.setText(subtitle_dir())
        save_bar.addWidget(self.save_path_edit, 1)
        change_btn = QPushButton("更改…")
        change_btn.clicked.connect(self._choose_subtitle_dir)
        save_bar.addWidget(change_btn)
        open_btn = QPushButton("打开文件夹")
        open_btn.clicked.connect(self._open_subtitle_dir)
        save_bar.addWidget(open_btn)
        layout.addLayout(save_bar)

        self.setStatusBar(QStatusBar())
        self._apply_font_size(20)

    def _choose_subtitle_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择字幕保存位置",
                                             self.save_path_edit.text())
        if d:
            self.save_path_edit.setText(d)
            s = load_settings()
            s["subtitle_dir"] = d
            save_settings(s)
            log(f"字幕保存位置改为：{d}")

    def _open_subtitle_dir(self):
        import subprocess
        d = self.save_path_edit.text()
        os.makedirs(d, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(d)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "打不开", f"无法打开文件夹：\n{e}")

    def _apply_font_size(self, size: int):
        f = self.partial_label.font()
        f.setPointSize(size)
        self.partial_label.setFont(f)
        hf = self.history.font()
        hf.setPointSize(max(10, size - 4))
        self.history.setFont(hf)

    def _toggle_pin(self, on: bool):
        flags = self.windowFlags()
        if on:
            self.setWindowFlags(flags | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags & ~Qt.WindowStaysOnTopHint)
        self.show()

    def _clear(self):
        self.history.clear()
        self.partial_label.setText("")

    # ---------- 字幕 ----------
    def _stamp(self) -> str:
        s = int(time.time() - self._t0)
        return f"[{s // 60:02d}:{s % 60:02d}]"

    def _on_partial(self, text: str):
        self.partial_label.setText(text if text else "…")

    def _on_final(self, text: str, start_sec: float, end_sec: float):
        self.history.append(f"<b>{self._stamp()}</b>  {text}")
        doc = self.history.document()
        while doc.blockCount() > self.MAX_LINES:
            cursor = self.history.textCursor()
            cursor.movePosition(cursor.MoveMode.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        bar = self.history.verticalScrollBar()
        bar.setValue(bar.maximum())
        # 同步写入字幕文本文件
        if self._subfile is not None:
            try:
                self._subfile_index += 1
                s = int(start_sec)
                self._subfile.write(f"[{s // 60:02d}:{s % 60:02d}] {text}\n")
                self._subfile.flush()
            except Exception as e:  # noqa: BLE001
                log(f"字幕写入失败：{e}")

    def _on_status(self, msg: str):
        self.statusBar().showMessage(msg)

    # ---------- 开始 / 停止 ----------
    def _toggle(self):
        if self._stopping:
            return  # 正在停止中，忽略重复点击
        if self._engine and self._engine.running:
            self._stop_engine()
        else:
            self._start_engine()

    def _set_running(self, running: bool):
        self.start_btn.setText("停止字幕" if running else "开始字幕")
        self.source_combo.setEnabled(not running)

    def _start_engine(self):
        data_dir = app_data_dir()
        log(f"点击开始字幕，模型齐全={models_ready(data_dir)}")
        if not models_ready(data_dir):
            if not self._download_models(data_dir):
                log("模型准备未完成，放弃启动")
                return
        kind = self.source_combo.currentData()
        try:
            src = audio_mod.make_source(kind)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "音频源不可用", f"无法打开音频源：\n{e}")
            return
        self._engine = StreamEngine(data_dir)
        self._engine.on_partial = lambda t: self._bridge.partial.emit(t)
        self._engine.on_final = lambda t, s, e: self._bridge.final.emit(t, s, e)
        self._engine.on_error = self._bridge.error.emit
        try:
            self._open_subfile()
            self._engine.start(src)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "启动失败", f"识别引擎启动失败：\n{e}")
            self._close_subfile()
            self._engine = None
            return
        self._t0 = time.time()
        self._set_running(True)
        self._on_status("正在监听…")
        self.partial_label.setText("…")

    def _stop_engine(self):
        """异步停止：界面不卡，引擎线程退出后才真正收尾。"""
        if self._stopping:
            return
        self._stopping = True
        self.start_btn.setEnabled(False)
        self._on_status("正在停止…")
        log("开始停止引擎")
        eng = self._engine

        def do_stop():
            if eng is not None:
                eng.stop()
            self._bridge.stopped.emit()

        threading.Thread(target=do_stop, daemon=True).start()

    def _on_stopped(self):
        """引擎线程已退出（UI 线程）。"""
        self._engine = None
        self._stopping = False
        self._close_subfile()
        self._set_running(False)
        self.start_btn.setEnabled(True)
        self._on_status("已停止")
        self.partial_label.setText("已停止。点击「开始字幕」继续。")
        log("引擎已停止")

    def _on_error(self, msg: str):
        """工作线程异常死亡（UI 线程）：多见于音频设备变动/休眠唤醒。"""
        log(f"识别线程异常死亡：{msg}")
        self._engine = None
        self._stopping = False
        self._close_subfile()
        self._set_running(False)
        self.start_btn.setEnabled(True)
        self._on_status(f"识别中断（{msg}），点击「开始字幕」可重新启动")
        self.partial_label.setText("识别中断，点击「开始字幕」重新启动。")

    def _open_subfile(self):
        """打开本次的字幕文本文件（每行 [分:秒] + 内容）。"""
        subdir = subtitle_dir()
        os.makedirs(subdir, exist_ok=True)
        self._subfile_path = os.path.join(
            subdir, time.strftime("字幕_%Y%m%d_%H%M%S.txt"))
        self._subfile = open(self._subfile_path, "w", encoding="utf-8")
        self._subfile_index = 0
        log(f"字幕文件：{self._subfile_path}")

    def _close_subfile(self):
        if self._subfile is not None:
            try:
                self._subfile.close()
            except Exception:
                pass
            self._subfile = None
            if self._subfile_index > 0:
                self._on_status(f"字幕已保存（{self._subfile_index}条）：{self._subfile_path}")
                log(f"字幕已保存（{self._subfile_index}条）：{self._subfile_path}")

    def _download_models(self, data_dir: str) -> bool:
        from .models import DownloadCancelled

        total_models = len(MODELS)
        dlg = QProgressDialog("正在准备识别模型…", "取消", 0, 100, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumWidth(460)
        # 关键：关掉 Qt 进度框"到100%自动隐藏"的默认行为，否则 3 个模型
        # 之间对话框会一闪一闪；0 延迟立刻显示，不等默认的 4 秒
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)
        dlg.show()
        cancelled = {"v": False}
        finished = {"v": False, "err": None, "cancelled": False}
        dlg.canceled.connect(lambda: cancelled.__setitem__("v", True))
        log("下载对话框打开")

        def worker():
            def progress(idx, total, label, done, total_bytes):
                # 下载线程里只记录，UI 线程定时取
                if not cancelled["v"]:
                    self._dl_pending = (idx, total, label, done, total_bytes)

            try:
                ensure_models(data_dir, progress_cb=progress,
                              should_stop=lambda: cancelled["v"])
            except DownloadCancelled:
                finished["cancelled"] = True
            except Exception as e:  # noqa: BLE001
                finished["err"] = e
                log(f"下载失败：{e}")
            finally:
                finished["v"] = True

        threading.Thread(target=worker, daemon=True).start()
        mb = 1048576
        while not finished["v"]:
            QApplication.processEvents()
            if cancelled["v"]:
                log("用户取消下载")
                dlg.close()
                log("下载对话框关闭（取消）")
                return False
            if self._dl_pending is not None:
                idx, total, label, done, total_bytes = self._dl_pending
                self._dl_pending = None
                if done is None or done < 0:
                    dlg.setLabelText(f"正在解压{label}…")
                    dlg.setValue(100)
                else:
                    have = f"{done // mb}MB" if total_bytes else "…"
                    want = f"{total_bytes // mb}MB" if total_bytes else "…"
                    dlg.setLabelText(
                        f"首次运行，正在下载模型 ({idx + 1}/{total}){label}：{have} / {want}\n"
                        f"共约 490MB，一次就好，之后完全离线。"
                    )
                    dlg.setValue(int(done * 100 / total_bytes) if total_bytes else 0)
            time.sleep(0.05)
        dlg.close()
        if finished["cancelled"]:
            log("下载对话框关闭（取消）")
            return False
        log("下载对话框关闭（结束）")
        if finished["err"]:
            QMessageBox.warning(
                self, "下载失败", f"模型下载失败：\n{finished['err']}\n请检查网络后重试。"
            )
            return False
        log("模型就绪")
        return True

    def closeEvent(self, event):
        # 先同步关字幕文件，保证落盘；引擎线程是 daemon，随进程退出
        try:
            if self._subfile is not None:
                self._subfile.close()
                self._subfile = None
        except Exception:
            pass
        self._stop_engine()
        super().closeEvent(event)


def run():
    app = QApplication(sys.argv)
    app.setApplicationName("实时字幕")
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        QMessageBox.critical(None, "缺少依赖", "没有找到 sherpa_onnx，请先安装依赖。")
        return 1
    w = MainWindow()
    w.show()
    return app.exec()
