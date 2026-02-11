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

ビット深度: 各サンプルの量子化ビット数
  - 8 bit  : 256段階（低品質）
  - 16 bit : 65,536段階（CD品質）
  - 24 bit : 16,777,216段階（プロ品質）
  - 32 bit float : 機械学習標準
```

### 1.2 音声フォーマット比較表

| フォーマット | 拡張子 | 圧縮 | 用途 | ビットレート例 |
|------------|--------|------|------|--------------|
| WAV | .wav | 非圧縮 | 編集・処理用 | 1,411 kbps (16bit/44.1kHz) |
| FLAC | .flac | 可逆圧縮 | アーカイブ | 〜900 kbps |
| MP3 | .mp3 | 非可逆 | 配信・再生 | 128-320 kbps |
| AAC | .m4a | 非可逆 | 配信・モバイル | 96-256 kbps |
| OGG/Opus | .ogg | 非可逆 | WebRTC・低遅延 | 32-128 kbps |
| PCM | .raw | 非圧縮 | API入力 | 256 kbps (16bit/16kHz) |

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

class AudioPipeline:
    """拡張可能な音声処理パイプライン"""

    def __init__(self):
        self.steps: list[tuple[str, Callable]] = []

    def add_step(self, name: str, func: Callable[[AudioData], AudioData]):
        """処理ステップを追加"""
        self.steps.append((name, func))
        return self  # メソッドチェーン対応

    def process(self, audio: AudioData) -> AudioData:
        """全ステップを順次実行"""
        for name, func in self.steps:
            try:
                audio = func(audio)
                print(f"  [OK] {name}: {audio.duration:.2f}s, "
                      f"SR={audio.sample_rate}, RMS={audio.rms:.4f}")
            except Exception as e:
                print(f"  [NG] {name}: {e}")
                raise
        return audio

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
```

---

## 3. 前処理モジュール

### 3.1 リサンプリング

```python
# リサンプリング: サンプリングレート変換
import librosa

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
    )

    return AudioData(
        samples=resampled.astype(np.float32),
        sample_rate=target_sr,
        channels=audio.channels,
        metadata={**audio.metadata, "resampled_from": audio.sample_rate},
    )
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
| ウィーナーフィルタ | 中 | 中〜高 | 可能 | 汎用 |
| RNNoise | 中 | 高 | 可能 | 汎用 |
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

### 5.2 音声区間検出（VAD）

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
```

---

## 7. パイプライン統合例

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

---

## 8. アンチパターン

### 8.1 アンチパターン：リサンプリングなしでAPIに送信

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

### 8.2 アンチパターン：ノイズ除去の過剰適用

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

### 8.3 アンチパターン：メモリ管理の欠如

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

---

## 9. FAQ

### Q1: librosaとsoundfileの使い分けは？

**A**: `librosa` は分析・特徴抽出が得意で、読み込み時に自動リサンプリングやモノラル変換が可能。`soundfile` は高速なI/Oに特化し、大容量ファイルの読み書きに向く。前処理・分析には `librosa`、最終出力の保存には `soundfile` という使い分けが一般的。

### Q2: リアルタイム処理でのバッファサイズはどう決めるか？

**A**: バッファサイズはレイテンシとスループットのトレードオフ。音声認識の場合、`chunk_size = sample_rate * 0.1`（100ms）が一般的。WebRTCでは20msが標準。バッファが小さすぎると処理オーバーヘッドが増加し、大きすぎると応答遅延が増える。

### Q3: GPU vs CPU、音声処理ではどちらを使うべきか？

**A**: 前処理（リサンプリング、FFT、ノイズ除去）はCPUで十分高速。GPUが有利なのはディープラーニングベースのノイズ除去（RNNoise、DeepFilterNet）や、大規模バッチの特徴抽出。リアルタイム処理ではCPU処理の方がレイテンシが安定する場合が多い。

---

## 10. まとめ

| カテゴリ | ポイント |
|---------|---------|
| 基本設定 | 音声認識用は16kHz/16bit/モノラルが標準 |
| 前処理 | リサンプリング→正規化→ノイズ除去の順序を守る |
| 特徴抽出 | MFCC(13-40次元)+デルタが汎用的、Whisper系はメルスペクトログラム |
| ノイズ除去 | スペクトラルゲートが基本、高品質要求にはDNN系 |
| VAD | エネルギーベースが高速、WebRTC VADが実用的 |
| フォーマット | ffmpegで堅牢に変換、APIプリセットで統一 |
| パイプライン | ステップを分離・合成可能に設計し、テスタビリティを確保 |

---

## 次に読むべきガイド

- [00-audio-apis.md](./00-audio-apis.md) — 音声AI APIの比較・統合
- 音声AIモデルの学習 — カスタムモデルのファインチューニング
- リアルタイム音声アプリケーション — WebRTC/WebSocketの実装

---

## 参考文献

1. librosa 公式ドキュメント — https://librosa.org/doc/
2. Jurafsky & Martin, "Speech and Language Processing" — https://web.stanford.edu/~jurafsky/slp3/
3. soundfile ドキュメント — https://python-soundfile.readthedocs.io/
4. ffmpeg 公式ドキュメント — https://ffmpeg.org/documentation.html
5. RNNoise: Learning Noise Suppression — https://jmvalin.ca/demo/rnnoise/
