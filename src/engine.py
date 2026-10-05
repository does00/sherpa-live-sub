"""流式识别引擎：在后台线程里跑。

实时字幕（快）：流式 zipformer 边听边出 partial，同一行不断刷新。
定稿精修（准）：一句话说完（endpoint），用 SenseVoice 离线重识别这段
  音频，再加标点，输出最终字幕。精修不可用时回退到流式结果+标点。
"""

import threading
import time

import numpy as np
import sherpa_onnx

from .models import model_paths


class StreamEngine:
    SAMPLE_RATE = 16000
    CHUNK = int(SAMPLE_RATE * 0.2)  # 每次喂 0.2 秒音频
    MIN_POLISH_SEC = 0.6  # 短于此时长的片段不值得精修，直接用流式结果

    def __init__(self, base_dir: str, num_threads: int = 2):
        self._base_dir = base_dir
        self._num_threads = num_threads
        self._stop = threading.Event()
        self._quick_tail = False  # 停止时尾段快速收尾，不做精修
        self._run_token = None    # 作废令牌：stop 后旧线程的回调全部丢弃
        self._active_src = None
        self._thread: threading.Thread | None = None
        self.on_partial = None  # fn(text)
        self.on_final = None    # fn(text, start_sec, end_sec)
        self.on_error = None    # fn(msg)：工作线程异常死亡时调用

    # ---------- 模型 ----------
    def _build(self):
        p = model_paths(self._base_dir, "asr_stream")
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=p["tokens"],
            encoder=p["encoder"],
            decoder=p["decoder"],
            joiner=p["joiner"],
            num_threads=self._num_threads,
            sample_rate=self.SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.0,
            rule2_min_trailing_silence=1.0,
            rule3_min_utterance_length=25.0,
            decoding_method="greedy_search",
        )
        # 精修 + 标点（可选，失败则回退）
        self._polish_rec = None
        self._punct = None
        try:
            pp = model_paths(self._base_dir, "asr_polish")
            self._polish_rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=pp["model"], tokens=pp["tokens"],
                use_itn=False, num_threads=self._num_threads,
            )
        except Exception:
            self._polish_rec = None
        try:
            pu = model_paths(self._base_dir, "punct")
            self._punct = sherpa_onnx.OfflinePunctuation(
                sherpa_onnx.OfflinePunctuationConfig(
                    model=sherpa_onnx.OfflinePunctuationModelConfig(
                        ct_transformer=pu["model"]
                    )
                )
            )
        except Exception:
            self._punct = None

    # ---------- 运行 ----------
    def start(self, audio_source):
        if self._thread and self._thread.is_alive():
            return
        self._build()
        self._stop.clear()
        self._quick_tail = False
        self._run_token = object()
        self._t0 = time.time()
        self._thread = threading.Thread(
            target=self._run, args=(audio_source,), daemon=True, name="asr"
        )
        self._thread.start()

    def _run(self, src):
        """线程入口：包一层异常捕获，音频设备变动等导致线程死亡时通知 UI。"""
        token = self._run_token
        try:
            self._run_guarded(src)
        except Exception as e:  # noqa: BLE001
            # stop() 之后才抛的异常不算故障（比如关闭音频源时捅醒了 record()）
            if token is not None and token is self._run_token and self.on_error:
                try:
                    self.on_error(f"{type(e).__name__}: {e}")
                except Exception:
                    pass

    def _run_guarded(self, src):
        token = self._run_token
        self._active_src = src

        def _alive():
            # 只有当前这轮运行的回调才有效；stop 后旧线程直接静默
            return token is not None and token is self._run_token

        rec = self._rec
        s = rec.create_stream()
        src.open()
        last = ""
        seg_audio: list = []
        seg_start: float | None = None
        try:
            while not self._stop.is_set():
                samples = src.read(self.CHUNK)
                if samples is None or len(samples) == 0:
                    time.sleep(0.05)
                    continue
                if seg_start is None:
                    seg_start = time.time()
                seg_audio.append(samples)
                s.accept_waveform(self.SAMPLE_RATE, samples)
                while rec.is_ready(s):
                    rec.decode_stream(s)
                text = rec.get_result(s)
                if text != last:
                    last = text
                    if _alive() and self.on_partial:
                        self.on_partial(text)
                if rec.is_endpoint(s):
                    if last.strip() and _alive() and self.on_final:
                        end = time.time()
                        start = seg_start if seg_start is not None else end
                        self.on_final(self._finalize(seg_audio, last.strip()),
                                      start - self._t0, end - self._t0)
                    rec.reset(s)
                    last = ""
                    seg_audio = []
                    seg_start = None
        finally:
            self._active_src = None
            try:
                src.close()
            except Exception:
                pass
        # 收尾：最后没说完的半句也定稿（停止时快速收尾，不精修）
        try:
            s.input_finished()
            while rec.is_ready(s):
                rec.decode_stream(s)
            tail = rec.get_result(s).strip()
            if tail and _alive() and self.on_final:
                end = time.time()
                start = seg_start if seg_start is not None else end
                self.on_final(self._finalize(seg_audio, tail, fast=self._quick_tail),
                              start - self._t0, end - self._t0)
        except Exception:
            pass

    def _finalize(self, seg_audio: list, fallback: str, fast: bool = False) -> str:
        """定稿：精修重识别 + 标点。fast=True 时跳过精修（停止收尾用）。"""
        text = fallback
        if not fast and self._polish_rec is not None and seg_audio:
            try:
                audio = np.concatenate(seg_audio).astype(np.float32)
                if len(audio) >= self.SAMPLE_RATE * self.MIN_POLISH_SEC:
                    ps = self._polish_rec.create_stream()
                    ps.accept_waveform(self.SAMPLE_RATE, audio)
                    self._polish_rec.decode_stream(ps)
                    refined = ps.result.text.strip()
                    if refined:
                        text = refined
            except Exception:
                pass
        if self._punct is not None and text.strip():
            try:
                text = self._punct.add_punctuation(text.strip())
            except Exception:
                pass
        return text

    def stop(self):
        # 先作废令牌：即使工作线程卡在 record() 里稍后才醒来，
        # 它的任何回调也会被直接丢弃，界面上不会再冒字幕
        self._run_token = None
        self._quick_tail = True
        self._stop.set()
        # 试着关闭音频源，把可能卡住的 record() 捅醒
        src, self._active_src = self._active_src, None
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
