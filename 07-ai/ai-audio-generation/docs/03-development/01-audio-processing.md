# 音声処理パイプライン実装ガイド

> 前処理、特徴抽出、ノイズ除去、フォーマット変換、リサンプリングなど、音声AIシステムの入出力を支える音声処理パイプラインの設計と実装を体系的に解説する。

---

## この章で学ぶこと

1. **音声信号の基礎**（サンプリングレート、ビット深度、チャンネル）を理解し、適切な前処理を設計できる
2. **特徴抽出**（MFCC、メルスペクトログラム、F0等）の原理と実装を習得し、AI入力を最適化できる
3. **ノイズ除去・正規化・フォーマット変換**の実装パターンを学び、本番品質のパイプラインを構築できる

---

## 1. 音声信号の基礎

### 1.1 デジタル音声の構造

```
アナログ音声波形 → サンプリング → 量子化 → デジタル音声

時間軸 ──────────────────────────────────────>
          ┌─┐
      ┌─┐ │ │ ┌─┐
  ┌─┐ │ │ │ │ │ │ ┌─┐
──┤ ├─┤ ├─┤ ├─┤ ├─┤ ├──  ← サンプル値（振幅）
  └─┘ └─┘ └─┘ └─┘ └─┘
  t0  t1  t2  t3  t4       ← サンプリング間隔

サンプリングレート: 1秒あたりのサンプル数 (Hz)
  - 8,000 Hz  : 電話音声品質
  - 16,000 Hz : 音声認識標準
  - 22,050 Hz : AM放送品質
  - 44,100 Hz : CD品質
  - 48,000 Hz : プロフェッショナル/動画標準
  - 96,000 Hz : ハイレゾオーディオ

ビット深度: 各サンプルの量子化ビット数
  - 8 bit  : 256段階（低品質）
  - 16 bit : 65,536段階（CD品質）
  - 24 bit : 16,777,216段階（プロ品質）
  - 32 bit float : 機械学習標準

ナイキスト周波数 = サンプリングレート / 2
  - 16kHz → 最大 8kHz の音声周波数を再現可能
  - 44.1kHz → 最大 22.05kHz（人間の可聴域をカバー）
```

### 1.2 音声フォーマット比較表

| フォーマット | 拡張子 | 圧縮 | 用途 | ビットレート例 |
|------------|--------|------|------|--------------|
| WAV | .wav | 非圧縮 | 編集・処理用 | 1,411 kbps (16bit/44.1kHz) |
| FLAC | .flac | 可逆圧縮 | アーカイブ | ~900 kbps |
| MP3 | .mp3 | 非可逆 | 配信・再生 | 128-320 kbps |
| AAC | .m4a | 非可逆 | 配信・モバイル | 96-256 kbps |
| OGG/Opus | .ogg | 非可逆 | WebRTC・低遅延 | 32-128 kbps |
| PCM | .raw | 非圧縮 | API入力 | 256 kbps (16bit/16kHz) |
| AIFF | .aiff | 非圧縮 | macOS/Logic Pro | 1,411 kbps (16bit/44.1kHz) |
| WebM | .webm | 非可逆 | Web配信 | 64-256 kbps |

### 1.3 音声ファイルのメタデータ取得

```python
import soundfile as sf
import librosa
from pathlib import Path

def get_audio_info(file_path: str) -> dict:
    """音声ファイルの詳細情報を取得"""
    path = Path(file_path)

    # soundfileで基本情報を取得
    info = sf.info(file_path)

    # librosaで追加分析
    y, sr = librosa.load(file_path, sr=None, mono=False)

    result = {
        "file_name": path.name,
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_sec": info.duration,
        "bit_depth": info.subtype,
        # 信号分析
        "peak_amplitude": float(abs(y).max()),
        "rms_level_db": float(20 * __import__("numpy").log10(
            __import__("numpy").sqrt(__import__("numpy").mean(y ** 2)) + 1e-10
        )),
        "is_mono": info.channels == 1,
        "is_stereo": info.channels == 2,
    }

    return result


def batch_audio_info(directory: str) -> list[dict]:
    """ディレクトリ内の全音声ファイルの情報を一括取得"""
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}
    results = []

    for path in Path(directory).iterdir():
        if path.suffix.lower() in audio_extensions:
            try:
                info = get_audio_info(str(path))
                results.append(info)
            except Exception as e:
                results.append({"file_name": path.name, "error": str(e)})

    return results
```

---

## 2. 音声処理パイプラインの全体設計

### 2.1 パイプラインアーキテクチャ

```
┌──────────────────────────────────────────────────────────┐
│                   音声処理パイプライン                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  入力音声 ──> [フォーマット変換] ──> [リサンプリング]       │
│                                         │                │
│                                         v                │
│               [チャンネル変換] <── [正規化]               │
│                    │                                     │
│                    v                                     │
│  [VAD (音声区間検出)] ──> [ノイズ除去] ──> [トリミング]    │
│                                              │           │
│                                              v           │
│              [特徴抽出] ──> [MFCC/メルスペクトログラム]    │
│                                   │                      │
│                                   v                      │
│                            [AI モデル入力]                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 2.2 パイプラインの基本実装

```python
# 音声処理パイプライン基盤
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

@dataclass
class AudioData:
    """音声データの統一表現"""
    samples: np.ndarray          # 波形データ (float32, -1.0 ~ 1.0)
    sample_rate: int             # サンプリングレート
    channels: int = 1            # チャンネル数
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """秒単位の長さ"""
        return len(self.samples) / self.sample_rate

    @property
    def rms(self) -> float:
        """RMS（平均二乗平方根）レベル"""
        return float(np.sqrt(np.mean(self.samples ** 2)))

    @property
    def peak(self) -> float:
        """ピーク振幅"""
        return float(np.max(np.abs(self.samples)))

    @property
    def rms_db(self) -> float:
        """RMSレベル（dB）"""
        rms = self.rms
        if rms == 0:
            return -float('inf')
        return float(20 * np.log10(rms))

    @property
    def crest_factor(self) -> float:
        """クレストファクター（ピーク/RMS比）"""
        rms = self.rms
        if rms == 0:
            return float('inf')
        return self.peak / rms

    def get_channel(self, channel: int) -> 'AudioData':
        """特定チャンネルのデータを取得"""
        if self.channels == 1:
            return self
        return AudioData(
            samples=self.samples[channel],
            sample_rate=self.sample_rate,
            channels=1,
            metadata={**self.metadata, "channel": channel},
        )

class AudioPipeline:
    """拡張可能な音声処理パイプライン"""

    def __init__(self):
        self.steps: list[tuple[str, Callable]] = []
        self._logs: list[dict] = []

    def add_step(self, name: str, func: Callable[[AudioData], AudioData]):
        """処理ステップを追加"""
        self.steps.append((name, func))
        return self  # メソッドチェーン対応

    def remove_step(self, name: str):
        """処理ステップを削除"""
        self.steps = [(n, f) for n, f in self.steps if n != name]
        return self

    def process(self, audio: AudioData, verbose: bool = True) -> AudioData:
        """全ステップを順次実行"""
        self._logs = []

        for name, func in self.steps:
            try:
                prev_duration = audio.duration
                prev_rms = audio.rms
                audio = func(audio)
                log_entry = {
                    "step": name,
                    "status": "OK",
                    "duration": audio.duration,
                    "sample_rate": audio.sample_rate,
                    "rms": audio.rms,
                }
                self._logs.append(log_entry)

                if verbose:
                    print(f"  [OK] {name}: {audio.duration:.2f}s, "
                          f"SR={audio.sample_rate}, RMS={audio.rms:.4f}")
            except Exception as e:
                self._logs.append({
                    "step": name,
                    "status": "FAILED",
                    "error": str(e),
                })
                if verbose:
                    print(f"  [NG] {name}: {e}")
                raise
        return audio

    def get_logs(self) -> list[dict]:
        """処理ログを取得"""
        return self._logs

    @staticmethod
    def load(path: str, target_sr: Optional[int] = None) -> AudioData:
        """音声ファイルを読み込み"""
        samples, sr = librosa.load(path, sr=target_sr, mono=False)
        if samples.ndim == 1:
            channels = 1
        else:
            channels = samples.shape[0]
            samples = samples.mean(axis=0)  # モノラル化
        return AudioData(
            samples=samples.astype(np.float32),
            sample_rate=sr,
            channels=channels,
            metadata={"source": path},
        )

    @staticmethod
    def save(audio: AudioData, path: str, format: str = "wav"):
        """音声データを保存"""
        sf.write(path, audio.samples, audio.sample_rate, format=format)

    @staticmethod
    def load_raw(path: str, sample_rate: int, dtype: str = "int16") -> AudioData:
        """RAW PCMファイルを読み込み"""
        dtype_map = {"int16": np.int16, "float32": np.float32}
        np_dtype = dtype_map.get(dtype, np.int16)

        raw_data = np.fromfile(path, dtype=np_dtype)

        if np_dtype == np.int16:
            samples = raw_data.astype(np.float32) / 32768.0
        else:
            samples = raw_data

        return AudioData(
            samples=samples,
            sample_rate=sample_rate,
            channels=1,
            metadata={"source": path, "raw_dtype": dtype},
        )
```

### 2.3 ストリーミング対応パイプライン

```python
import numpy as np
from collections import deque
from typing import Iterator

class StreamingAudioPipeline:
    """ストリーミング対応の音声処理パイプライン"""

    def __init__(
        self,
        chunk_size: int = 4096,
        sample_rate: int = 16000,
        overlap: int = 0,
    ):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.steps: list[tuple[str, Callable]] = []
        self._buffer = np.array([], dtype=np.float32)
        self._overlap_buffer = np.array([], dtype=np.float32)

    def add_step(self, name: str, func: Callable):
        self.steps.append((name, func))
        return self

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """1チャンクを処理"""
        # オーバーラップ処理
        if len(self._overlap_buffer) > 0:
            chunk = np.concatenate([self._overlap_buffer, chunk])

        if self.overlap > 0:
            self._overlap_buffer = chunk[-self.overlap:]
        else:
            self._overlap_buffer = np.array([], dtype=np.float32)

        audio = AudioData(
            samples=chunk.astype(np.float32),
            sample_rate=self.sample_rate,
        )

        for name, func in self.steps:
            audio = func(audio)

        return audio.samples

    def process_stream(
        self, audio_stream: Iterator[np.ndarray]
    ) -> Iterator[np.ndarray]:
        """音声ストリームを処理"""
        for chunk in audio_stream:
            processed = self.process_chunk(chunk)
            yield processed

    def reset(self):
        """内部状態をリセット"""
        self._buffer = np.array([], dtype=np.float32)
        self._overlap_buffer = np.array([], dtype=np.float32)


def create_microphone_stream(
    sample_rate: int = 16000,
    chunk_size: int = 1600,  # 100ms
) -> Iterator[np.ndarray]:
    """マイク入力からのストリーム生成"""
    import pyaudio

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            yield np.frombuffer(data, dtype=np.float32)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
```

---

## 3. 前処理モジュール

### 3.1 リサンプリング

```python
# リサンプリング: サンプリングレート変換
import librosa
import numpy as np

def resample(audio: AudioData, target_sr: int = 16000) -> AudioData:
    """サンプリングレートを変換する"""
    if audio.sample_rate == target_sr:
        return audio

    resampled = librosa.resample(
        audio.samples,
        orig_sr=audio.sample_rate,
        target_sr=target_sr,
        res_type="kaiser_best",  # 高品質リサンプリング
        # 他の選択肢:
        #   "kaiser_fast"  - 高速だが品質やや低下
        #   "scipy"        - scipy.signal.resample
        #   "polyphase"    - ポリフェーズフィルタ
        #   "fft"          - FFTベース（周期信号向け）
        #   "soxr_hq"      - SoXリサンプラー高品質
        #   "soxr_vhq"     - SoXリサンプラー最高品質
    )

    return AudioData(
        samples=resampled.astype(np.float32),
        sample_rate=target_sr,
        channels=audio.channels,
        metadata={**audio.metadata, "resampled_from": audio.sample_rate},
    )


def resample_batch(
    audio_files: list[str],
    target_sr: int = 16000,
    output_dir: str = "./resampled",
) -> list[str]:
    """複数ファイルの一括リサンプリング"""
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results = []

    for file_path in audio_files:
        audio = AudioPipeline.load(file_path)
        resampled = resample(audio, target_sr)

        out_file = output_path / Path(file_path).name
        AudioPipeline.save(resampled, str(out_file))
        results.append(str(out_file))

    return results
```

### 3.2 正規化

```python
# 音声正規化: ピーク正規化とラウドネス正規化
import numpy as np

def peak_normalize(audio: AudioData, target_peak: float = 0.95) -> AudioData:
    """ピーク正規化: 最大振幅を指定値に合わせる"""
    current_peak = np.max(np.abs(audio.samples))
    if current_peak == 0:
        return audio

    gain = target_peak / current_peak
    normalized = audio.samples * gain

    return AudioData(
        samples=normalized.astype(np.float32),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        metadata={**audio.metadata, "peak_normalized": True, "gain": gain},
    )

def rms_normalize(
    audio: AudioData, target_db: float = -20.0
) -> AudioData:
    """RMS正規化: 平均音量を指定dBに合わせる"""
    current_rms = np.sqrt(np.mean(audio.samples ** 2))
    if current_rms == 0:
        return audio

    target_rms = 10 ** (target_db / 20)
    gain = target_rms / current_rms

    normalized = audio.samples * gain
    # クリッピング防止
    normalized = np.clip(normalized, -1.0, 1.0)

    return AudioData(
        samples=normalized.astype(np.float32),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        metadata={**audio.metadata, "rms_normalized": True},
    )

def lufs_normalize(
    audio: AudioData, target_lufs: float = -14.0
) -> AudioData:
    """
    LUFS正規化: ITU-R BS.1770準拠のラウドネス正規化
    ストリーミング配信（Spotify: -14LUFS、YouTube: -14LUFS）向け
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(audio.sample_rate)

    # 現在のラウドネスを計測
    current_lufs = meter.integrated_loudness(audio.samples)

    if current_lufs == -float('inf'):
        return audio

    # ゲイン計算
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)

    normalized = audio.samples * gain_linear
    # ピーク制限（True Peakが-1dBTPを超えないように）
    peak = np.max(np.abs(normalized))
    if peak > 0.891:  # -1dBTP ≈ 0.891
        normalized = normalized * (0.891 / peak)

    return AudioData(
        samples=normalized.astype(np.float32),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        metadata={
            **audio.metadata,
            "lufs_normalized": True,
            "original_lufs": current_lufs,
            "target_lufs": target_lufs,
        },
    )
```

### 3.3 チャンネル変換

```python
import numpy as np

def to_mono(audio: AudioData, method: str = "mean") -> AudioData:
    """ステレオをモノラルに変換"""
    if audio.channels == 1:
        return audio

    if audio.samples.ndim == 1:
        return audio

    if method == "mean":
        # 平均値（標準的な方法）
        mono = np.mean(audio.samples, axis=0)
    elif method == "left":
        mono = audio.samples[0]
    elif method == "right":
        mono = audio.samples[1]
    elif method == "side":
        # サイド信号（L-R）: 残響成分の抽出に有用
        mono = audio.samples[0] - audio.samples[1]
    elif method == "mid":
        # ミッド信号（L+R）: ボーカル抽出に有用
        mono = audio.samples[0] + audio.samples[1]
    else:
        raise ValueError(f"未対応の変換方法: {method}")

    return AudioData(
        samples=mono.astype(np.float32),
        sample_rate=audio.sample_rate,
        channels=1,
        metadata={**audio.metadata, "mono_method": method},
    )

def to_stereo(audio: AudioData) -> AudioData:
    """モノラルをステレオに変換"""
    if audio.channels == 2:
        return audio

    stereo = np.stack([audio.samples, audio.samples], axis=0)

    return AudioData(
        samples=stereo.astype(np.float32),
        sample_rate=audio.sample_rate,
        channels=2,
        metadata={**audio.metadata, "converted_to_stereo": True},
    )
```

### 3.4 トリミングとパディング

```python
import numpy as np
import librosa

def trim_silence(
    audio: AudioData,
    top_db: float = 30,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> AudioData:
    """無音区間をトリミング"""
    trimmed, index = librosa.effects.trim(
        audio.samples,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    trim_start = index[0] / audio.sample_rate
    trim_end = index[1] / audio.sample_rate

    return AudioData(
        samples=trimmed.astype(np.float32),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        metadata={
            **audio.metadata,
            "trimmed": True,
            "trim_start_sec": trim_start,
            "trim_end_sec": trim_end,
            "original_duration": len(audio.samples) / audio.sample_rate,
        },
    )

def pad_to_duration(
    audio: AudioData,
    target_duration: float,
    pad_mode: str = "constant",
) -> AudioData:
    """指定した長さにパディング"""
    target_samples = int(target_duration * audio.sample_rate)
    current_samples = len(audio.samples)

    if current_samples >= target_samples:
        # 長い場合は切り詰め
        return AudioData(
            samples=audio.samples[:target_samples],
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            metadata={**audio.metadata, "padded": False, "truncated": True},
        )

    pad_length = target_samples - current_samples

    if pad_mode == "constant":
        padded = np.pad(audio.samples, (0, pad_length), mode="constant")
    elif pad_mode == "wrap":
        # ループ（繰り返し）パディング
        padded = np.pad(audio.samples, (0, pad_length), mode="wrap")
    elif pad_mode == "reflect":
        # 反転パディング（自然なフェードアウト風）
        padded = np.pad(audio.samples, (0, pad_length), mode="reflect")
    else:
        padded = np.pad(audio.samples, (0, pad_length), mode="constant")

    return AudioData(
        samples=padded.astype(np.float32),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        metadata={
            **audio.metadata,
            "padded": True,
            "pad_mode": pad_mode,
            "pad_duration": pad_length / audio.sample_rate,
        },
    )

def split_audio(
    audio: AudioData,
    chunk_duration: float = 30.0,
    overlap: float = 0.0,
) -> list[AudioData]:
    """音声を指定秒数のチャンクに分割"""
    chunk_samples = int(chunk_duration * audio.sample_rate)
    overlap_samples = int(overlap * audio.sample_rate)
    step = chunk_samples - overlap_samples

    chunks = []
    start = 0

    while start < len(audio.samples):
        end = min(start + chunk_samples, len(audio.samples))
        chunk = audio.samples[start:end]

        # 最後のチャンクが短すぎる場合はパディング
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")

        chunks.append(AudioData(
            samples=chunk.astype(np.float32),
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            metadata={
                **audio.metadata,
                "chunk_index": len(chunks),
                "chunk_start_sec": start / audio.sample_rate,
                "chunk_end_sec": end / audio.sample_rate,
            },
        ))

        start += step

    return chunks
```

---

## 4. ノイズ除去

### 4.1 ノイズ除去手法の比較

```
┌──────────────────────────────────────────────────┐
│              ノイズ除去手法の分類                    │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌────────────────┐    ┌────────────────────┐   │
│  │  統計的手法     │    │  スペクトル手法      │   │
│  ├────────────────┤    ├────────────────────┤   │
│  │ ウィーナーフィルタ│    │ スペクトラルゲート   │   │
│  │ カルマンフィルタ │    │ スペクトル減算       │   │
│  └────────────────┘    └────────────────────┘   │
│                                                  │
│  ┌────────────────┐    ┌────────────────────┐   │
│  │  深層学習手法    │    │  適応フィルタ       │   │
│  ├────────────────┤    ├────────────────────┤   │
│  │ RNNoise        │    │  LMS適応フィルタ    │   │
│  │ DTLN           │    │  NLMS適応フィルタ   │   │
│  │ DeepFilterNet  │    │  RLS適応フィルタ    │   │
│  └────────────────┘    └────────────────────┘   │
└──────────────────────────────────────────────────┘
```

| 手法 | 計算コスト | 品質 | リアルタイム | 適用場面 |
|------|-----------|------|------------|---------|
| スペクトラルゲート | 低 | 中 | 可能 | 定常ノイズ |
| スペクトル減算 | 低 | 中 | 可能 | 背景ノイズ |
| ウィーナーフィルタ | 中 | 中~高 | 可能 | 汎用 |
| RNNoise | 中 | 高 | 可能 | 汎用 |
| DTLN | 中 | 高 | 可能 | リアルタイム |
| DeepFilterNet | 高 | 非常に高 | 可能(GPU) | 高品質要求 |

### 4.2 スペクトラルゲートによるノイズ除去

```python
# スペクトラルゲートによるノイズ除去
import numpy as np
import librosa

def spectral_gate_denoise(
    audio: AudioData,
    noise_sample_duration: float = 0.5,
    threshold_factor: float = 1.5,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> AudioData:
    """
    スペクトラルゲーティングによるノイズ除去
    冒頭のノイズサンプルからノイズプロファイルを推定し、閾値以下を抑制
    """
    samples = audio.samples
    sr = audio.sample_rate

    # STFT(短時間フーリエ変換)
    stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # ノイズプロファイル推定（冒頭の無音区間から）
    noise_frames = int(noise_sample_duration * sr / hop_length)
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # スペクトラルゲート適用
    threshold = noise_profile * threshold_factor
    mask = magnitude > threshold
    # ソフトマスク（滑らかな遷移）
    soft_mask = np.clip(
        (magnitude - threshold) / (threshold + 1e-10), 0.0, 1.0
    )

    cleaned_magnitude = magnitude * soft_mask

    # 逆STFT
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    cleaned_samples = librosa.istft(
        cleaned_stft, hop_length=hop_length, length=len(samples)
    )

    return AudioData(
        samples=cleaned_samples.astype(np.float32),
        sample_rate=sr,
        channels=audio.channels,
        metadata={**audio.metadata, "denoised": "spectral_gate"},
    )
```

### 4.3 noisereduceライブラリによるノイズ除去

```python
import noisereduce as nr
import numpy as np

def noisereduce_denoise(
    audio: AudioData,
    stationary: bool = True,
    prop_decrease: float = 0.75,
    n_fft: int = 2048,
    noise_clip: Optional[np.ndarray] = None,
) -> AudioData:
    """
    noisereduceライブラリによるノイズ除去

    Parameters:
        stationary: True=定常ノイズ仮定, False=非定常ノイズ対応
        prop_decrease: ノイズ削減率（0.0-1.0）
        noise_clip: ノイズサンプル（Noneの場合は自動推定）
    """
    reduced = nr.reduce_noise(
        y=audio.samples,
        sr=audio.sample_rate,
        y_noise=noise_clip,
        stationary=stationary,
        prop_decrease=prop_decrease,
        n_fft=n_fft,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
    )

    return AudioData(
        samples=reduced.astype(np.float32),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        metadata={
            **audio.metadata,
            "denoised": "noisereduce",
            "stationary": stationary,
            "prop_decrease": prop_decrease,
        },
    )
```

### 4.4 DeepFilterNetによる高品質ノイズ除去

```python
def deepfilternet_denoise(audio: AudioData) -> AudioData:
    """
    DeepFilterNetによる高品質ノイズ除去（GPU推奨）
    - DNNベースで非定常ノイズにも対応
    - 音声品質の劣化が最小
    """
    from df.enhance import enhance, init_df, load_audio, save_audio
    import torch

    model, df_state, _ = init_df()

    # 音声をモデル入力形式に変換
    audio_tensor = torch.tensor(
        audio.samples, dtype=torch.float32
    ).unsqueeze(0)

    # ノイズ除去実行
    enhanced = enhance(model, df_state, audio_tensor)

    return AudioData(
        samples=enhanced.squeeze().numpy().astype(np.float32),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        metadata={**audio.metadata, "denoised": "deepfilternet"},
    )
```

### 4.5 エコーキャンセレーション

```python
import numpy as np
from scipy import signal

def echo_cancellation(
    audio: AudioData,
    reference: AudioData,
    filter_length: int = 1024,
    step_size: float = 0.01,
) -> AudioData:
    """
    NLMSアルゴリズムによるエコーキャンセレーション

    Parameters:
        audio: エコーが含まれるマイク入力
        reference: スピーカー出力（参照信号）
        filter_length: 適応フィルタの長さ
        step_size: 適応ステップサイズ（μ）
    """
    x = reference.samples  # 参照信号
    d = audio.samples      # エコー混入信号
    n = len(d)

    # フィルタ係数の初期化
    w = np.zeros(filter_length)
    y = np.zeros(n)  # エコー推定
    e = np.zeros(n)  # エラー（エコー除去後）

    for i in range(filter_length, n):
        # 参照信号のウィンドウ
        x_window = x[i - filter_length:i][::-1]

        # エコー推定
        y[i] = np.dot(w, x_window)

        # エラー計算
        e[i] = d[i] - y[i]

        # NLMS更新
        norm = np.dot(x_window, x_window) + 1e-10
        w += step_size * e[i] * x_window / norm

    return AudioData(
        samples=e.astype(np.float32),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        metadata={**audio.metadata, "echo_cancelled": True},
    )
```

---

## 5. 特徴抽出

### 5.1 MFCC（メル周波数ケプストラム係数）

```python
# MFCC特徴抽出
import librosa
import numpy as np

def extract_mfcc(
    audio: AudioData,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    include_delta: bool = True,
) -> np.ndarray:
    """
    MFCC特徴量を抽出する

    Parameters:
        n_mfcc: MFCC係数の数（通常13 or 40）
        include_delta: デルタ/デルタデルタを含めるか

    Returns:
        shape: (n_features, n_frames) のnumpy配列
        - include_delta=False: n_features = n_mfcc
        - include_delta=True:  n_features = n_mfcc * 3
    """
    mfcc = librosa.feature.mfcc(
        y=audio.samples,
        sr=audio.sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    if not include_delta:
        return mfcc

    # デルタ（1次微分）: 時間変化を捉える
    delta = librosa.feature.delta(mfcc, order=1)
    # デルタデルタ（2次微分）: 加速度を捉える
    delta2 = librosa.feature.delta(mfcc, order=2)

    # 結合: (n_mfcc*3, n_frames)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return features
```

### 5.2 メルスペクトログラム

```python
def extract_mel_spectrogram(
    audio: AudioData,
    n_mels: int = 80,
    n_fft: int = 2048,
    hop_length: int = 512,
    to_db: bool = True,
) -> np.ndarray:
    """
    メルスペクトログラムを抽出する
    Whisper等のモデルはメルスペクトログラムを直接入力として使用
    """
    mel = librosa.feature.melspectrogram(
        y=audio.samples,
        sr=audio.sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)

    return mel
```

### 5.3 ピッチ（F0）抽出

```python
import librosa
import numpy as np

def extract_pitch(
    audio: AudioData,
    fmin: float = 50.0,
    fmax: float = 500.0,
    method: str = "pyin",
) -> dict:
    """
    ピッチ（基本周波数 F0）を抽出する

    Parameters:
        fmin: 最小周波数（男性声: 50Hz, 女性声: 100Hz）
        fmax: 最大周波数（男性声: 300Hz, 女性声: 500Hz）
        method: "pyin" or "crepe"
    """
    if method == "pyin":
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio.samples,
            sr=audio.sample_rate,
            fmin=fmin,
            fmax=fmax,
        )
    elif method == "yin":
        f0 = librosa.yin(
            audio.samples,
            sr=audio.sample_rate,
            fmin=fmin,
            fmax=fmax,
        )
        voiced_flag = f0 > 0
        voiced_probs = None
    else:
        raise ValueError(f"未対応のメソッド: {method}")

    # NaNを0で埋める
    f0_clean = np.nan_to_num(f0, nan=0.0)

    # 統計量
    voiced_f0 = f0_clean[f0_clean > 0]

    return {
        "f0": f0_clean,
        "voiced_flag": voiced_flag,
        "voiced_probs": voiced_probs,
        "statistics": {
            "mean_f0": float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0,
            "std_f0": float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0,
            "min_f0": float(np.min(voiced_f0)) if len(voiced_f0) > 0 else 0,
            "max_f0": float(np.max(voiced_f0)) if len(voiced_f0) > 0 else 0,
            "voiced_ratio": float(np.sum(f0_clean > 0) / len(f0_clean)),
        },
    }
```

### 5.4 スペクトル特徴量

```python
import librosa
import numpy as np

def extract_spectral_features(
    audio: AudioData,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict:
    """
    スペクトル特徴量を一括抽出する
    音声分類・感情分析等に有用
    """
    y = audio.samples
    sr = audio.sample_rate

    # スペクトル重心（音の「明るさ」）
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    # スペクトル帯域幅（音の「広がり」）
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    # スペクトルロールオフ（エネルギーの85%がある周波数）
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    # スペクトルフラットネス（トーナル vs ノイズ）
    spectral_flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length
    )[0]

    # ゼロクロッシングレート
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop_length
    )[0]

    # クロマグラム（調性情報）
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )

    return {
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "spectral_flatness": spectral_flatness,
        "zero_crossing_rate": zcr,
        "chroma": chroma,
        "statistics": {
            "mean_centroid": float(np.mean(spectral_centroid)),
            "mean_bandwidth": float(np.mean(spectral_bandwidth)),
            "mean_rolloff": float(np.mean(spectral_rolloff)),
            "mean_flatness": float(np.mean(spectral_flatness)),
            "mean_zcr": float(np.mean(zcr)),
        },
    }
```

### 5.5 音声区間検出（VAD）

```python
# Voice Activity Detection (VAD)
import numpy as np

def energy_based_vad(
    audio: AudioData,
    frame_length: int = 2048,
    hop_length: int = 512,
    energy_threshold_db: float = -40.0,
    min_speech_duration: float = 0.3,
    min_silence_duration: float = 0.2,
) -> list[tuple[float, float]]:
    """
    エネルギーベースのVAD（音声区間検出）

    Returns:
        [(start_sec, end_sec), ...] 音声区間のリスト
    """
    samples = audio.samples
    sr = audio.sample_rate

    # フレームごとのエネルギー計算
    frames = librosa.util.frame(
        samples, frame_length=frame_length, hop_length=hop_length
    )
    energy = np.sum(frames ** 2, axis=0)
    energy_db = 10 * np.log10(energy + 1e-10)

    # 閾値判定
    is_speech = energy_db > energy_threshold_db

    # 最小持続時間フィルタリング
    min_speech_frames = int(min_speech_duration * sr / hop_length)
    min_silence_frames = int(min_silence_duration * sr / hop_length)

    segments = []
    in_speech = False
    start_frame = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            start_frame = i
            in_speech = True
        elif not speech and in_speech:
            duration_frames = i - start_frame
            if duration_frames >= min_speech_frames:
                start_sec = start_frame * hop_length / sr
                end_sec = i * hop_length / sr
                segments.append((start_sec, end_sec))
            in_speech = False

    # 末尾処理
    if in_speech:
        duration_frames = len(is_speech) - start_frame
        if duration_frames >= min_speech_frames:
            start_sec = start_frame * hop_length / sr
            end_sec = len(samples) / sr
            segments.append((start_sec, end_sec))

    # 近接セグメント統合
    merged = _merge_close_segments(segments, min_silence_duration)
    return merged

def _merge_close_segments(
    segments: list[tuple[float, float]],
    min_gap: float,
) -> list[tuple[float, float]]:
    """近接するセグメントを統合"""
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_end = merged[-1][1]
        if start - prev_end < min_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def silero_vad(
    audio: AudioData,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> list[tuple[float, float]]:
    """Silero VAD（高精度なニューラルネットワークベースのVAD）"""
    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )
    get_speech_timestamps = utils[0]

    audio_tensor = torch.tensor(audio.samples, dtype=torch.float32)

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=audio.sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    segments = [
        (ts["start"] / audio.sample_rate, ts["end"] / audio.sample_rate)
        for ts in speech_timestamps
    ]

    return segments
```

---

## 6. フォーマット変換

### 6.1 ffmpegを使った堅牢な変換

```python
# ffmpegベースの堅牢なフォーマット変換
import subprocess
import tempfile
from pathlib import Path

class AudioConverter:
    """ffmpegを使った音声フォーマット変換"""

    # AI API向け推奨設定
    API_PRESETS = {
        "whisper": {"format": "wav", "sr": 16000, "channels": 1, "bit_depth": 16},
        "google_stt": {"format": "flac", "sr": 16000, "channels": 1, "bit_depth": 16},
        "azure_stt": {"format": "wav", "sr": 16000, "channels": 1, "bit_depth": 16},
        "polly_output": {"format": "mp3", "sr": 22050, "channels": 1, "bitrate": "128k"},
        "elevenlabs": {"format": "mp3", "sr": 44100, "channels": 1, "bitrate": "192k"},
        "deepgram": {"format": "wav", "sr": 16000, "channels": 1, "bit_depth": 16},
    }

    @staticmethod
    def convert(
        input_path: str,
        output_path: str,
        sample_rate: int = 16000,
        channels: int = 1,
        bit_depth: int = 16,
        output_format: str = "wav",
    ) -> str:
        """汎用フォーマット変換"""
        codec_map = {
            "wav": "pcm_s16le" if bit_depth == 16 else "pcm_s24le",
            "flac": "flac",
            "mp3": "libmp3lame",
            "ogg": "libopus",
            "aac": "aac",
        }

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-acodec", codec_map.get(output_format, "pcm_s16le"),
            output_path,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg変換失敗: {result.stderr}")

        return output_path

    @classmethod
    def convert_for_api(
        cls, input_path: str, api: str, output_dir: str = "/tmp"
    ) -> str:
        """API向けプリセットで変換"""
        preset = cls.API_PRESETS.get(api)
        if not preset:
            raise ValueError(f"未知のAPIプリセット: {api}")

        stem = Path(input_path).stem
        ext = preset["format"]
        output_path = f"{output_dir}/{stem}_{api}.{ext}"

        return cls.convert(
            input_path=input_path,
            output_path=output_path,
            sample_rate=preset["sr"],
            channels=preset["channels"],
            bit_depth=preset.get("bit_depth", 16),
            output_format=ext,
        )

    @staticmethod
    def get_audio_info_ffprobe(file_path: str) -> dict:
        """ffprobeで音声ファイル情報を取得"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe失敗: {result.stderr}")

        import json
        return json.loads(result.stdout)

    @staticmethod
    def extract_segment(
        input_path: str,
        output_path: str,
        start_sec: float,
        end_sec: float,
    ) -> str:
        """音声ファイルから指定区間を抽出"""
        duration = end_sec - start_sec
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", str(start_sec),
            "-t", str(duration),
            "-c", "copy",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg区間抽出失敗: {result.stderr}")

        return output_path
```

---

## 7. データ拡張（Data Augmentation）

### 7.1 音声データ拡張テクニック

```python
import numpy as np
import librosa

class AudioAugmentor:
    """音声データ拡張ツール（学習データ増強用）"""

    @staticmethod
    def add_noise(
        audio: AudioData,
        noise_level: float = 0.005,
        noise_type: str = "gaussian",
    ) -> AudioData:
        """ノイズを付加"""
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_level, len(audio.samples))
        elif noise_type == "uniform":
            noise = np.random.uniform(-noise_level, noise_level, len(audio.samples))
        elif noise_type == "pink":
            # ピンクノイズ（1/fノイズ）
            white = np.random.randn(len(audio.samples))
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(len(white))
            freqs[0] = 1  # DC成分の0除算回避
            fft = fft / np.sqrt(freqs)
            noise = np.fft.irfft(fft, n=len(white)) * noise_level
        else:
            noise = np.random.normal(0, noise_level, len(audio.samples))

        augmented = audio.samples + noise.astype(np.float32)
        augmented = np.clip(augmented, -1.0, 1.0)

        return AudioData(
            samples=augmented,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            metadata={**audio.metadata, "augmentation": f"noise_{noise_type}"},
        )

    @staticmethod
    def time_stretch(
        audio: AudioData,
        rate: float = 1.0,
    ) -> AudioData:
        """タイムストレッチ（速度変更、ピッチ維持）"""
        stretched = librosa.effects.time_stretch(audio.samples, rate=rate)

        return AudioData(
            samples=stretched.astype(np.float32),
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            metadata={**audio.metadata, "augmentation": f"time_stretch_{rate}"},
        )

    @staticmethod
    def pitch_shift(
        audio: AudioData,
        n_steps: float = 0.0,
    ) -> AudioData:
        """ピッチシフト（音の高さ変更）"""
        shifted = librosa.effects.pitch_shift(
            audio.samples,
            sr=audio.sample_rate,
            n_steps=n_steps,
        )

        return AudioData(
            samples=shifted.astype(np.float32),
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            metadata={**audio.metadata, "augmentation": f"pitch_shift_{n_steps}"},
        )

    @staticmethod
    def add_reverb(
        audio: AudioData,
        decay: float = 0.3,
        delay_ms: float = 50.0,
    ) -> AudioData:
        """簡易リバーブの付加"""
        delay_samples = int(delay_ms * audio.sample_rate / 1000)
        impulse = np.zeros(delay_samples + 1)
        impulse[0] = 1.0
        impulse[-1] = decay

        reverbed = np.convolve(audio.samples, impulse, mode="full")[:len(audio.samples)]

        return AudioData(
            samples=reverbed.astype(np.float32),
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            metadata={**audio.metadata, "augmentation": "reverb"},
        )

    @staticmethod
    def random_crop(
        audio: AudioData,
        crop_duration: float = 5.0,
    ) -> AudioData:
        """ランダムクロップ"""
        crop_samples = int(crop_duration * audio.sample_rate)
        if len(audio.samples) <= crop_samples:
            return audio

        start = np.random.randint(0, len(audio.samples) - crop_samples)
        cropped = audio.samples[start:start + crop_samples]

        return AudioData(
            samples=cropped.astype(np.float32),
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            metadata={
                **audio.metadata,
                "augmentation": "random_crop",
                "crop_start_sec": start / audio.sample_rate,
            },
        )

    @staticmethod
    def spec_augment(
        spectrogram: np.ndarray,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ) -> np.ndarray:
        """
        SpecAugment: スペクトログラムへのマスキング拡張
        音声認識モデルの学習で精度を大幅に改善する手法
        """
        augmented = spectrogram.copy()
        n_freq, n_time = augmented.shape

        # 周波数マスク
        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, n_freq - f)
            augmented[f0:f0 + f, :] = 0

        # 時間マスク
        for _ in range(num_time_masks):
            t = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, n_time - t)
            augmented[:, t0:t0 + t] = 0

        return augmented
```

---

## 8. パイプライン統合例

### 8.1 STT用パイプライン

```python
# 完全なパイプライン構成例
def build_stt_pipeline(target_sr: int = 16000) -> AudioPipeline:
    """音声認識用の標準パイプラインを構築"""
    pipeline = AudioPipeline()
    pipeline.add_step("resample", lambda a: resample(a, target_sr))
    pipeline.add_step("normalize", lambda a: rms_normalize(a, target_db=-20))
    pipeline.add_step("denoise", lambda a: spectral_gate_denoise(a))
    pipeline.add_step("peak_norm", lambda a: peak_normalize(a, target_peak=0.95))
    return pipeline

# 使用例
audio = AudioPipeline.load("input.wav")
pipeline = build_stt_pipeline()
processed = pipeline.process(audio)
AudioPipeline.save(processed, "output_processed.wav")

features = extract_mfcc(processed, n_mfcc=40, include_delta=True)
print(f"特徴量形状: {features.shape}")
```

### 8.2 TTS後処理パイプライン

```python
def build_tts_postprocess_pipeline(
    target_sr: int = 22050,
    target_lufs: float = -14.0,
) -> AudioPipeline:
    """TTS出力の後処理パイプラインを構築"""
    pipeline = AudioPipeline()
    pipeline.add_step("resample", lambda a: resample(a, target_sr))
    pipeline.add_step("denoise", lambda a: noisereduce_denoise(a, prop_decrease=0.5))
    pipeline.add_step("lufs_norm", lambda a: lufs_normalize(a, target_lufs))
    pipeline.add_step("trim", lambda a: trim_silence(a, top_db=40))
    return pipeline
```

### 8.3 バッチ処理パイプライン

```python
import concurrent.futures
from pathlib import Path

def batch_process_audio(
    input_dir: str,
    output_dir: str,
    pipeline: AudioPipeline,
    max_workers: int = 4,
    extensions: set = {".wav", ".mp3", ".flac"},
) -> dict:
    """ディレクトリ内の音声ファイルを一括処理"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in extensions
    ]

    results = {"success": 0, "failed": 0, "errors": []}

    def process_file(file_path: Path) -> dict:
        try:
            audio = AudioPipeline.load(str(file_path))
            processed = pipeline.process(audio, verbose=False)
            out_file = output_path / file_path.name
            AudioPipeline.save(processed, str(out_file))
            return {"file": file_path.name, "status": "OK"}
        except Exception as e:
            return {"file": file_path.name, "status": "FAILED", "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result["status"] == "OK":
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(result)

    return results
```

---

## 9. アンチパターン

### 9.1 アンチパターン：リサンプリングなしでAPIに送信

```python
# NG: 44.1kHzの音声をそのままAPIに送信
audio_44k = load_audio("music_quality.wav")  # 44,100Hz
result = stt_api.transcribe(audio_44k)  # APIは16kHz想定

# OK: API仕様に合わせてリサンプリング
audio_44k = load_audio("music_quality.wav")
audio_16k = resample(audio_44k, target_sr=16000)
result = stt_api.transcribe(audio_16k)
```

**問題点**: APIが想定するサンプリングレートと異なる音声を送ると、認識精度が大幅に低下する。APIのドキュメントを確認し、必ず適切なレートに変換する。

### 9.2 アンチパターン：ノイズ除去の過剰適用

```python
# NG: 閾値を極端に高く設定してノイズ除去
cleaned = spectral_gate_denoise(
    audio, threshold_factor=5.0  # 高すぎる閾値
)
# → 音声成分まで除去され「ロボット声」になる

# OK: 適度な閾値でノイズ除去
cleaned = spectral_gate_denoise(
    audio, threshold_factor=1.5  # 適度な閾値
)
# 必ず聴取して品質を確認する
```

**問題点**: ノイズ除去を強くかけすぎると音声の自然さが失われる（アーティファクト発生）。閾値は控えめに設定し、必ず人間の耳で品質を確認する。

### 9.3 アンチパターン：メモリ管理の欠如

```python
# NG: 大量の音声ファイルを全てメモリに保持
all_audios = [load_audio(f) for f in glob("*.wav")]  # メモリ爆発

# OK: ジェネレータでストリーミング処理
def process_files(file_list):
    for f in file_list:
        audio = load_audio(f)
        result = pipeline.process(audio)
        yield result
        del audio, result  # 明示的な解放
```

### 9.4 アンチパターン：正規化の順序間違い

```python
# NG: ノイズ除去前に正規化（ノイズも増幅される）
pipeline = AudioPipeline()
pipeline.add_step("normalize", peak_normalize)  # ノイズごと増幅
pipeline.add_step("denoise", spectral_gate_denoise)  # 増幅されたノイズは除去が困難

# OK: ノイズ除去後に正規化
pipeline = AudioPipeline()
pipeline.add_step("denoise", spectral_gate_denoise)  # まずノイズ除去
pipeline.add_step("normalize", peak_normalize)  # クリーンな信号を正規化
```

**問題点**: 正規化とノイズ除去の順序を間違えると、ノイズが増幅されて除去が困難になる。前処理の順序は「リサンプリング→ノイズ除去→正規化」が基本。

---

## 10. FAQ

### Q1: librosaとsoundfileの使い分けは？

**A**: `librosa` は分析・特徴抽出が得意で、読み込み時に自動リサンプリングやモノラル変換が可能。`soundfile` は高速なI/Oに特化し、大容量ファイルの読み書きに向く。前処理・分析には `librosa`、最終出力の保存には `soundfile` という使い分けが一般的。

### Q2: リアルタイム処理でのバッファサイズはどう決めるか？

**A**: バッファサイズはレイテンシとスループットのトレードオフ。音声認識の場合、`chunk_size = sample_rate * 0.1`（100ms）が一般的。WebRTCでは20msが標準。バッファが小さすぎると処理オーバーヘッドが増加し、大きすぎると応答遅延が増える。

### Q3: GPU vs CPU、音声処理ではどちらを使うべきか？

**A**: 前処理（リサンプリング、FFT、ノイズ除去）はCPUで十分高速。GPUが有利なのはディープラーニングベースのノイズ除去（RNNoise、DeepFilterNet）や、大規模バッチの特徴抽出。リアルタイム処理ではCPU処理の方がレイテンシが安定する場合が多い。

### Q4: 音声データ拡張はどの程度効果があるか？

**A**: SpecAugmentは音声認識のWERを相対10-20%改善することが報告されている。ノイズ付加は実環境のロバスト性を大幅に向上させる。ただし、過剰な拡張は学習を不安定にするため、元データの2-5倍程度に留めるのが一般的。時間伸縮（0.8-1.2倍）とピッチシフト（-2~+2半音）の組み合わせが効果的。

### Q5: ラウドネス正規化のターゲット値はどう決めるか？

**A**: 配信プラットフォームごとに推奨値が異なる。Spotify/YouTube: -14 LUFS、Apple Music: -16 LUFS、TV/ラジオ: -24 LUFS（EBU R128）。ポッドキャスト: -16 to -14 LUFS。ターゲットに合わせて `lufs_normalize` を使用し、True Peakが-1dBTPを超えないよう制限する。

---

## 11. まとめ

| カテゴリ | ポイント |
|---------|---------|
| 基本設定 | 音声認識用は16kHz/16bit/モノラルが標準 |
| 前処理 | リサンプリング→ノイズ除去→正規化の順序を守る |
| 特徴抽出 | MFCC(13-40次元)+デルタが汎用的、Whisper系はメルスペクトログラム |
| ノイズ除去 | スペクトラルゲートが基本、高品質要求にはDNN系 |
| VAD | エネルギーベースが高速、Silero VADが高精度 |
| フォーマット | ffmpegで堅牢に変換、APIプリセットで統一 |
| データ拡張 | SpecAugment + ノイズ付加 + 時間伸縮が効果的 |
| ストリーミング | オーバーラップ付きチャンク処理でリアルタイム対応 |
| パイプライン | ステップを分離・合成可能に設計し、テスタビリティを確保 |

---

## 次に読むべきガイド

- [00-audio-apis.md](./00-audio-apis.md) — 音声AI APIの比較・統合
- [02-real-time-audio.md](./02-real-time-audio.md) — リアルタイム音声処理
- [../00-fundamentals/03-stt-technologies.md](../00-fundamentals/03-stt-technologies.md) — STT技術の詳細

---

## 参考文献

1. librosa 公式ドキュメント — https://librosa.org/doc/
2. Jurafsky & Martin, "Speech and Language Processing" — https://web.stanford.edu/~jurafsky/slp3/
3. soundfile ドキュメント — https://python-soundfile.readthedocs.io/
4. ffmpeg 公式ドキュメント — https://ffmpeg.org/documentation.html
5. RNNoise: Learning Noise Suppression — https://jmvalin.ca/demo/rnnoise/
6. Park, D.S., et al. (2019). "SpecAugment: A Simple Data Augmentation Method for ASR" — Google Brainによるデータ拡張手法
7. Schroeter, H., et al. (2022). "DeepFilterNet: A Low Complexity Speech Enhancement Framework" — 高品質DNN系ノイズ除去
