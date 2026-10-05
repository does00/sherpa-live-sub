"""模型定义与自动下载。

三个模型：
- asr_stream：流式识别（zipformer 中英双语 int8），负责实时出字，`快`
- asr_polish：离线精修（SenseVoice int8），每句话定稿时重识别一遍，准
- punct：标点恢复（ct-transformer int8），给定时稿加标点

下载是脆弱环节（大文件 + 杀毒软件扫描 + 用户重复点击 + 国内网络），所以：
- 每个文件有多个镜像源（国内优先：ModelScope / hf-mirror），逐个尝试
- 镜像都失败再整包下载兜底（gh-proxy 加速 → GitHub 直连）
- 每个模型有独立文件锁：同一时间只下一份，重复点击/僵尸线程不会抢文件
- 所有文件 I/O（删除、改名）失败自动重试：Windows Defender 扫描大文件
  时会短暂锁定，等几秒就好
- 下载校验 Content-Length，不完整自动换源重下
"""

import os
import tarfile
import threading
import time
import urllib.request


class DownloadCancelled(Exception):
    """用户取消下载。"""


_GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


def _tarball_mirrors(path: str) -> list:
    """整包下载的镜像顺序：gh-proxy 加速优先，GitHub 直连兜底。"""
    url = f"{_GH}/{path}"
    return [
        "https://ghproxy.cn/" + url,
        "https://mirror.ghproxy.com/" + url,
        url,
    ]


MODELS = {
    "asr_stream": {
        "dirname": "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        "label": "实时识别模型",
        "files": {
            "encoder": {
                "filename": "encoder-epoch-99-avg-1.int8.onnx",
                "urls": [
                    "https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
                    "/resolve/main/encoder-epoch-99-avg-1.int8.onnx",
                ],
            },
            "decoder": {
                "filename": "decoder-epoch-99-avg-1.onnx",
                "urls": [
                    "https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
                    "/resolve/main/decoder-epoch-99-avg-1.onnx",
                ],
            },
            "joiner": {
                "filename": "joiner-epoch-99-avg-1.int8.onnx",
                "urls": [
                    "https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
                    "/resolve/main/joiner-epoch-99-avg-1.int8.onnx",
                ],
            },
            "tokens": {
                "filename": "tokens.txt",
                "urls": [
                    "https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
                    "/resolve/main/tokens.txt",
                ],
            },
        },
        "tarball_urls": _tarball_mirrors(
            "asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"),
    },
    "asr_polish": {
        "dirname": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        "label": "精修识别模型",
        "files": {
            "model": {
                "filename": "model.int8.onnx",
                "urls": [
                    "https://modelscope.cn/models/pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue"
                    "/resolve/master/model.int8.onnx",
                    "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
                    "/resolve/main/model.int8.onnx",
                ],
            },
            "tokens": {
                "filename": "tokens.txt",
                "urls": [
                    "https://modelscope.cn/models/pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue"
                    "/resolve/master/tokens.txt",
                    "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
                    "/resolve/main/tokens.txt",
                ],
            },
        },
        "tarball_urls": _tarball_mirrors(
            "asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"),
    },
    "punct": {
        "dirname": "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
        "label": "标点模型",
        "files": {
            # 该模型暂无国内单文件镜像，走整包下载
            "model": {"filename": "model.int8.onnx", "urls": []},
        },
        "tarball_urls": _tarball_mirrors(
            "punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2"),
    },
}


# ---------- 并发与 I/O 健壮性 ----------

_DL_LOCKS: dict = {}
_DL_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _DL_LOCKS_GUARD:
        return _DL_LOCKS.setdefault(key, threading.Lock())


def _io_retry(fn, tries: int = 6, what: str = "文件操作"):
    """I/O 出错自动重试（Windows 杀毒软件扫描时会短暂锁文件）。"""
    last = None
    for i in range(tries):
        try:
            return fn()
        except FileNotFoundError:
            return None
        except OSError as e:  # noqa: BLE001
            last = e
            time.sleep(1 + i)
    raise last


def _try_remove(path: str) -> bool:
    """尽力删除，删不掉也不报错（比如被杀毒软件锁住，过会儿就好）。"""
    try:
        _io_retry(lambda: os.remove(path), tries=4)
        return True
    except OSError:
        return False


def _sleep_interruptible(seconds: float, should_stop=None):
    end = time.time() + seconds
    while time.time() < end:
        if should_stop and should_stop():
            raise DownloadCancelled()
        time.sleep(min(0.5, max(0.0, end - time.time())))


# ---------- 模型管理 ----------

def _model_dir(base_dir: str, key: str) -> str:
    return os.path.join(base_dir, MODELS[key]["dirname"])


def model_paths(base_dir: str, key: str) -> dict:
    """返回某模型各文件的绝对路径。"""
    d = _model_dir(base_dir, key)
    return {name: os.path.join(d, info["filename"])
            for name, info in MODELS[key]["files"].items()}


def _ready(base_dir: str, key: str) -> bool:
    return all(os.path.isfile(p) and os.path.getsize(p) > 0
               for p in model_paths(base_dir, key).values())


def models_ready(base_dir: str) -> bool:
    return all(_ready(base_dir, k) for k in MODELS)


def _migrate_legacy(base_dir: str) -> int:
    """把老位置（~/.sherpa-live-sub）已下好的文件搬到新位置，省得重下。

    逐文件搬：以前下到一半的，只要单个文件完整就能继续用。
    同盘符下是秒级改名；搬失败（被占用等）就跳过，下次或重下兜底。
    """
    import shutil
    legacy = os.path.join(os.path.expanduser("~"), ".sherpa-live-sub")
    if os.path.abspath(legacy) == os.path.abspath(base_dir):
        return 0
    if not os.path.isdir(legacy):
        return 0
    moved = 0
    for key, spec in MODELS.items():
        for name, info in spec["files"].items():
            src = os.path.join(legacy, spec["dirname"], info["filename"])
            dst = os.path.join(base_dir, spec["dirname"], info["filename"])
            if os.path.isfile(dst) and os.path.getsize(dst) > 0:
                continue
            if not (os.path.isfile(src) and os.path.getsize(src) > 0):
                continue
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                moved += 1
            except OSError:
                pass
    return moved


def ensure_models(base_dir: str, progress_cb=None, retries: int = 3,
                  should_stop=None) -> str:
    """确保三个模型就位。

    progress_cb(idx, total, label, done_bytes, total_bytes)：
    下载中 done_bytes/total_bytes 为字节数；解压时 done_bytes 为 -1。
    """
    os.makedirs(base_dir, exist_ok=True)
    _migrate_legacy(base_dir)  # 老版本下好的文件搬过来，不重复下载
    keys = list(MODELS.keys())
    for i, key in enumerate(keys):
        if _ready(base_dir, key):
            continue
        # 同一模型同一时间只下一份：防重复点击、僵尸线程抢文件
        with _lock_for(key):
            if _ready(base_dir, key):  # 等锁时别人下好了
                continue
            _ensure_one(base_dir, key, i, len(keys), progress_cb, retries, should_stop)
    return base_dir


def _ensure_one(base_dir: str, key: str, idx: int, total: int,
                progress_cb, retries: int, should_stop) -> None:
    spec = MODELS[key]
    d = _model_dir(base_dir, key)
    os.makedirs(d, exist_ok=True)

    # 1) 逐文件走国内镜像（失败就转整包兜底）
    for name, info in spec["files"].items():
        if should_stop and should_stop():
            raise DownloadCancelled()
        dst = os.path.join(d, info["filename"])
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            continue
        if info["urls"]:
            label = f"{spec['label']}·{info['filename']}"
            cb = (lambda done, total_b, _l=label:  # noqa: B023
                  progress_cb(idx, total, _l, done, total_b)) if progress_cb else None
            if _download_with_mirrors(info["urls"], dst, cb, should_stop):
                continue
        # 该文件镜像全部失败（或无镜像）：整包兜底
        _ensure_tarball(base_dir, key, idx, total, progress_cb, retries, should_stop)
        return

    if not _ready(base_dir, key):
        # 极少数情况：文件都在但校验不过，走整包重下
        _ensure_tarball(base_dir, key, idx, total, progress_cb, retries, should_stop)


def _download_with_mirrors(urls: list, dst: str, progress_cb, should_stop) -> bool:
    """按顺序尝试每个镜像，成功返回 True；全部失败返回 False。"""
    for url in urls:
        if should_stop and should_stop():
            raise DownloadCancelled()
        try:
            _download(url, dst, progress_cb, should_stop)
            return True
        except DownloadCancelled:
            raise
        except Exception:  # noqa: BLE001
            _try_remove(dst + ".part")
            continue
    return False


def _ensure_tarball(base_dir: str, key: str, idx: int, total: int,
                    progress_cb, retries: int, should_stop) -> None:
    """整包下载兜底：轮换 gh-proxy 镜像与 GitHub 直连。"""
    spec = MODELS[key]
    label = spec["label"]
    archive = os.path.join(base_dir, spec["dirname"] + ".tar.bz2")
    urls = spec["tarball_urls"]
    last_err = None
    for attempt in range(retries + 1):
        if should_stop and should_stop():
            raise DownloadCancelled()
        url = urls[attempt % len(urls)]  # 每次重试换一个镜像
        try:
            _download(url, archive,
                      lambda d, t: progress_cb(idx, total, label + "（整包）", d, t)
                      if progress_cb else None,
                      should_stop=should_stop)
            if not _is_bz2(archive):
                raise IOError("下载到的不是有效压缩包（镜像可能返回了错误页面），换源重试")
            if progress_cb:
                progress_cb(idx, total, label, -1, -1)  # 解压中
            with tarfile.open(archive, "r:bz2") as tf:
                tf.extractall(base_dir, filter="data")
            if not _ready(base_dir, key):
                raise RuntimeError("解压后文件不全")
            _try_remove(archive)  # 删不掉也认：文件齐了就能用
            return
        except DownloadCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            for f in (archive, archive + ".part"):
                _try_remove(f)
            if attempt < retries:
                _sleep_interruptible(2 * (attempt + 1), should_stop)
    raise RuntimeError(f"{label}下载失败（国内镜像与官方源都试过了）：{last_err}")


def _is_bz2(path: str) -> bool:
    """检查文件头是否为 bzip2（防镜像返回 HTML 错误页面）。"""
    try:
        with open(path, "rb") as f:
            return f.read(3) == b"BZh"
    except OSError:
        return False


def _download(url: str, dst: str, progress_cb=None, should_stop=None):
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "sherpa-live-sub"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0) or 0)
            done = 0
            while True:
                if should_stop and should_stop():
                    raise DownloadCancelled()
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress_cb:
                    progress_cb(done, total)
    except DownloadCancelled:
        _try_remove(tmp)
        raise
    # 校验完整性：服务器给了长度就必须对上；没给长度则要求文件非空
    if total and done != total:
        _try_remove(tmp)
        raise IOError(f"下载不完整（{done}/{total} 字节），网络中断")
    if not total and done == 0:
        _try_remove(tmp)
        raise IOError("下载得到空文件")
    _io_retry(lambda: os.replace(tmp, dst), tries=6)
