"""音频源：系统声音（Windows loopback）/ 麦克风 / 文件（测试用）。"""

import time
import warnings
import wave

import numpy as np


def to_16k_mono(data: np.ndarray, src_rate: int) -> np.ndarray:
    """任意采样率立体声/单声道 -> 16k 单声道 float32。"""
    data = np.asarray(data, dtype=np.float32)
    mono = data.mean(axis=1) if data.ndim == 2 else data
    if src_rate == 16000:
        return mono.astype(np.float32, copy=False)
    n_out = int(len(mono) * 16000 / src_rate)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(len(mono), dtype=np.float64)
    x_new = np.linspace(0, len(mono) - 1, n_out)
    return np.interp(x_new, x_old, mono).astype(np.float32)


class LoopbackSource:
    """Windows 系统声音：抓取浏览器/网页正在播放的声音（WASAPI loopback）。

    原理：把系统当前播放的声音当成“麦克风”录进来，无需装虚拟声卡。
    """

    NAME = "系统声音（浏览器/网页播放的声音）"

    def open(self):
        import soundcard as sc

        speaker = sc.default_speaker()
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        self._rec = mic.recorder(samplerate=48000, channels=2)
        self._rec.__enter__()
        self._src_rate = 48000

    def read(self, num_frames_16k: int) -> np.ndarray:
        want = int(num_frames_16k * self._src_rate / 16000)
        # 屏蔽 soundcard 偶发的 "data discontinuity" 警告（短暂丢帧提示，不影响识别）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self._rec.record(numframes=want)
        return to_16k_mono(data, self._src_rate)

    def close(self):
        try:
            self._rec.__exit__(None, None, None)
        except Exception:
            pass


class MicSource:
    """麦克风（备用：想识别自己说话时用）。"""

    NAME = "麦克风"

    def open(self):
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", blocksize=3200
        )
        self._stream.start()

    def read(self, num_frames_16k: int) -> np.ndarray:
        data, _ = self._stream.read(num_frames_16k)
        return np.asarray(data[:, 0], dtype=np.float32)

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class FileSource:
    """从 wav 文件模拟实时音频流（开发测试用）。"""

    NAME = "测试文件"

    def __init__(self, path: str):
        self._path = path

    def open(self):
        with wave.open(self._path, "rb") as f:
            assert f.getframerate() == 16000 and f.getnchannels() == 1, \
                "测试文件必须是 16k 单声道 wav"
            raw = f.readframes(f.getnframes())
        self._samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        self._pos = 0

    def read(self, num_frames_16k: int) -> np.ndarray:
        if self._pos >= len(self._samples):
            return np.zeros(0, dtype=np.float32)
        chunk = self._samples[self._pos:self._pos + num_frames_16k]
        self._pos += len(chunk)
        time.sleep(len(chunk) / 16000.0)  # 模拟实时速度
        return chunk

    def close(self):
        pass


def available_sources():
    """返回本机可用的音频源列表。"""
    import platform

    sources = []
    if platform.system() == "Windows":
        sources.append(("loopback", LoopbackSource.NAME))
    sources.append(("mic", MicSource.NAME))
    return sources


def make_source(kind: str, **kwargs):
    if kind == "loopback":
        return LoopbackSource()
    if kind == "mic":
        return MicSource()
    if kind == "file":
        return FileSource(kwargs["path"])
    raise ValueError(f"未知音频源: {kind}")
