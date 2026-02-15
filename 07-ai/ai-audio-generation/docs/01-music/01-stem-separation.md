# ステム分離 — Demucs、LALAL.AI

> 音楽トラックをボーカル・ドラム・ベース・その他に分離するステム分離技術の仕組みと実践を解説する

## この章で学ぶこと

1. ステム分離の技術的原理（スペクトログラムマスキング、ニューラルネットワーク分離）
2. 主要ツール（Demucs、LALAL.AI、Spleeter）の特徴と使い分け
3. ステム分離の実装パターンと品質改善テクニック

---

## 1. ステム分離の技術基盤

### 1.1 ステム分離の概念

```
ステム分離の基本概念
==================================================

入力: ミックスされた楽曲（2ch ステレオ）
  ┌─────────────────────────────┐
  │  ボーカル + ドラム + ベース  │
  │  + ギター + ピアノ + ...    │
  │  = 1つの波形                │
  └──────────────┬──────────────┘
                 │
          ステム分離モデル
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌────────┐ ┌────────┐ ┌────────┐
│ボーカル │ │ドラム  │ │ベース  │
│ (Vocal)│ │(Drums) │ │(Bass)  │
└────────┘ └────────┘ └────────┘
                 │
                 ▼
           ┌────────┐
           │その他  │
           │(Other) │
           │ギター等│
           └────────┘

理想: mix = vocal + drums + bass + other
現実: mix ≈ vocal + drums + bass + other + artifacts
==================================================
```

### 1.2 技術的アプローチの進化

```
ステム分離技術の進化
==================================================

第1世代: スペクトログラムマスキング
  ┌──────┐    ┌─────┐    ┌──────┐    ┌──────┐
  │ STFT │───→│マスク│───→│適用  │───→│ ISTFT│
  └──────┘    │推定  │    │      │    └──────┘
              └─────┘    └──────┘
  * 単純だが品質に限界

第2世代: U-Net ベース（Spleeter, Open-Unmix）
  ┌──────┐    ┌──────────┐    ┌──────┐
  │スペク│───→│U-Net     │───→│マスク│
  │トログ│    │(Encoder- │    │適用  │
  │ラム  │    │ Decoder) │    │      │
  └──────┘    └──────────┘    └──────┘
  * スペクトログラム領域で処理

第3世代: ハイブリッド（Demucs v4 / HTDemucs）
  ┌──────┐    ┌────────────────┐    ┌──────┐
  │波形  │───→│Temporal Branch │───→│      │
  │      │    │(時間領域CNN)   │    │ 統合 │
  └──────┘    └────────────────┘    │      │
  ┌──────┐    ┌────────────────┐    │      │
  │スペク│───→│Spectral Branch │───→│      │
  │トログ│    │(Transformer)   │    │      │
  │ラム  │    └────────────────┘    └──────┘
  * 時間領域+周波数領域のハイブリッド
  * Transformerによるグローバルな文脈理解
==================================================
```

### 1.3 マスキング手法の詳細

ステム分離の核心技術はマスキングである。入力のスペクトログラムに対して、各音源に対応するマスクを推定し、それを適用することで各音源を分離する。

```python
import numpy as np
import torch

class MaskingMethods:
    """ステム分離におけるマスキング手法"""

    @staticmethod
    def ideal_binary_mask(source_stft, mix_stft):
        """
        理想バイナリマスク（IBM）
        - 各時間-周波数ビンで支配的な音源を判定
        - マスク値は0または1
        - 最も単純だがアーティファクトが発生しやすい
        """
        source_mag = np.abs(source_stft)
        mix_mag = np.abs(mix_stft)
        # 元の音源が混合の50%以上を占めるビンを1に
        mask = (source_mag > 0.5 * mix_mag).astype(float)
        return mask

    @staticmethod
    def ideal_ratio_mask(source_stft, mix_stft):
        """
        理想比率マスク（IRM）
        - 各ビンでの音源の比率をマスク値とする
        - ソフトマスクでIBMより自然な音質
        - 値は0〜1の連続値
        """
        source_mag = np.abs(source_stft)
        mix_mag = np.abs(mix_stft) + 1e-10
        mask = source_mag / mix_mag
        return np.clip(mask, 0, 1)

    @staticmethod
    def wiener_filter_mask(sources_stfts, mix_stft):
        """
        ウィーナーフィルタマスク
        - 全音源のパワースペクトルの比率でマスクを計算
        - 最も理論的に正当なマスキング手法
        - Demucsの後処理でも使用される
        """
        powers = [np.abs(s) ** 2 for s in sources_stfts]
        total_power = sum(powers) + 1e-10
        masks = [p / total_power for p in powers]
        return masks

    @staticmethod
    def complex_ideal_ratio_mask(source_stft, mix_stft):
        """
        複素理想比率マスク（cIRM）
        - 位相情報も含むマスク
        - 実部と虚部を個別に推定
        - 位相再構成の品質が向上
        """
        mix_mag = np.abs(mix_stft) + 1e-10
        # 複素マスク = source / mix
        mask_real = np.real(source_stft) / np.real(mix_stft + 1e-10)
        mask_imag = np.imag(source_stft) / np.imag(mix_stft + 1e-10)
        return mask_real, mask_imag


class SpectrogramProcessor:
    """スペクトログラム処理の基盤クラス"""

    def __init__(self, n_fft: int = 4096, hop_length: int = 1024,
                 sr: int = 44100):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr

    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """STFT計算"""
        window = np.hanning(self.n_fft)
        n_frames = (len(audio) - self.n_fft) // self.hop_length + 1
        stft = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=complex)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft] * window
            stft[:, i] = np.fft.rfft(frame)

        return stft

    def inverse_stft(self, stft: np.ndarray) -> np.ndarray:
        """逆STFT計算（Griffin-Lim法ベース）"""
        n_frames = stft.shape[1]
        output_length = self.n_fft + (n_frames - 1) * self.hop_length
        output = np.zeros(output_length)
        window_sum = np.zeros(output_length)
        window = np.hanning(self.n_fft)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = np.fft.irfft(stft[:, i])
            output[start:start + self.n_fft] += frame * window
            window_sum[start:start + self.n_fft] += window ** 2

        # ウィンドウ正規化
        mask = window_sum > 1e-8
        output[mask] /= window_sum[mask]
        return output

    def apply_mask(self, mix_stft: np.ndarray,
                   mask: np.ndarray) -> np.ndarray:
        """マスク適用"""
        return mix_stft * mask

    def separate_with_masks(self, mix_audio: np.ndarray,
                            masks: list) -> list:
        """マスクを使って音源分離を実行"""
        mix_stft = self.compute_stft(mix_audio)
        separated = []
        for mask in masks:
            source_stft = self.apply_mask(mix_stft, mask)
            source_audio = self.inverse_stft(source_stft)
            separated.append(source_audio)
        return separated
```

### 1.4 Demucs v4（HTDemucs）のアーキテクチャ詳細

```
HTDemucs（Hybrid Transformer Demucs）の内部構造
==================================================

入力: ステレオ波形 (2, T)
      │
      ├─────────────────────────────────────┐
      │                                     │
      ▼                                     ▼
┌─────────────┐                    ┌─────────────┐
│ Temporal     │                    │ Spectral    │
│ Encoder      │                    │ Encoder     │
│ (1D Conv)    │                    │ (2D Conv)   │
│              │                    │             │
│ x5 layers    │                    │ x5 layers   │
│ ch: 48→384   │                    │ STFT →      │
│ stride: 4    │                    │ スペクトロ  │
│              │                    │ グラム処理  │
└──────┬──────┘                    └──────┬──────┘
       │                                   │
       ▼                                   ▼
┌─────────────────────────────────────────────┐
│              Cross-Domain                    │
│              Transformer                     │
│                                             │
│  ┌──────────────┐    ┌──────────────┐       │
│  │ Self-Attention│    │ Cross-Attention│     │
│  │ (Temporal)   │←──→│ (Spectral)   │       │
│  └──────────────┘    └──────────────┘       │
│                                             │
│  * 時間領域と周波数領域の相互参照           │
│  * グローバルな文脈の理解                   │
│  * 5層のTransformerブロック                 │
└───────────┬─────────────────┬───────────────┘
            │                 │
            ▼                 ▼
┌──────────────┐    ┌──────────────┐
│ Temporal     │    │ Spectral    │
│ Decoder      │    │ Decoder     │
│ (1D DeConv)  │    │ (2D DeConv) │
│              │    │             │
│ x5 layers    │    │ x5 layers   │
│ Skip Connect │    │ Skip Connect│
└──────┬───────┘    └──────┬──────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────┐
│ 出力統合                      │
│ temporal_out + spectral_out  │
│ → 4ソース x (2, T)           │
│ [drums, bass, other, vocals] │
└──────────────────────────────┘

モデルパラメータ:
- パラメータ数: 約83M（htdemucs）/ 約83M（htdemucs_ft）
- 入力: 44.1kHz ステレオ
- 処理: 7.8秒のセグメント（デフォルト）
- オーバーラップ: 25%
==================================================
```

---

## 2. Demucs の実装

### 2.1 基本的な使い方

```python
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Demucs v4 (HTDemucs) モデルのロード
model = get_model("htdemucs_ft")  # ファインチューニング版
model.eval()

# GPU使用（利用可能な場合）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 音声ファイルの読み込み
waveform, sr = torchaudio.load("song.wav")

# リサンプリング（Demucsは44.1kHzを期待）
if sr != model.samplerate:
    resampler = torchaudio.transforms.Resample(sr, model.samplerate)
    waveform = resampler(waveform)

# ステム分離の実行
with torch.no_grad():
    waveform = waveform.unsqueeze(0).to(device)  # バッチ次元追加
    sources = apply_model(
        model,
        waveform,
        shifts=1,       # ランダムシフトの回数（品質向上、速度低下）
        overlap=0.25,   # オーバーラップ率
    )

# 結果の取得
# sources shape: (batch, n_sources, channels, samples)
source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']

for i, name in enumerate(source_names):
    source_audio = sources[0, i]  # (channels, samples)
    torchaudio.save(f"{name}.wav", source_audio.cpu(), model.samplerate)
    print(f"保存: {name}.wav")
```

### 2.2 CLI での使用

```python
# コマンドラインでの使用方法（概念コード）

"""
# 基本的な分離
demucs song.wav

# モデル指定
demucs --model htdemucs_ft song.wav

# 2ステム分離（ボーカル/伴奏のみ）
demucs --two-stems vocals song.wav

# 出力先指定
demucs -o ./output song.wav

# 品質向上オプション
demucs --shifts 5 --overlap 0.5 song.wav

# MP3出力
demucs --mp3 --mp3-bitrate 320 song.wav

# バッチ処理
demucs song1.wav song2.wav song3.wav
"""

# Python スクリプトでのバッチ処理
import subprocess
from pathlib import Path

def batch_separate(input_dir: str, output_dir: str, model: str = "htdemucs_ft"):
    """ディレクトリ内の全音声ファイルをステム分離"""
    input_path = Path(input_dir)
    audio_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.mp3"))

    for audio_file in audio_files:
        print(f"処理中: {audio_file.name}")
        cmd = [
            "demucs",
            "--model", model,
            "--out", output_dir,
            "--shifts", "3",
            str(audio_file),
        ]
        subprocess.run(cmd, check=True)
        print(f"完了: {audio_file.name}")

batch_separate("./songs", "./separated")
```

### 2.3 カスタムパイプライン

```python
import torch
import torchaudio
import numpy as np

class StemSeparationPipeline:
    """ステム分離 + 後処理パイプライン"""

    def __init__(self, model_name="htdemucs_ft"):
        from demucs.pretrained import get_model
        self.model = get_model(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def separate(self, audio_path: str) -> dict:
        """分離 + 品質改善"""
        from demucs.apply import apply_model

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.model.samplerate:
            resampler = torchaudio.transforms.Resample(sr, self.model.samplerate)
            waveform = resampler(waveform)

        with torch.no_grad():
            sources = apply_model(
                self.model,
                waveform.unsqueeze(0).to(self.device),
                shifts=3,
                overlap=0.25,
            )

        results = {}
        for i, name in enumerate(self.model.sources):
            stem = sources[0, i].cpu()
            # 後処理: ノイズゲート + 正規化
            stem = self._noise_gate(stem, threshold_db=-60)
            stem = self._normalize(stem, target_db=-3)
            results[name] = stem

        return results

    def _noise_gate(self, audio, threshold_db=-60):
        """ノイズゲート: 閾値以下の音を無音化"""
        threshold = 10 ** (threshold_db / 20)
        mask = torch.abs(audio) > threshold
        return audio * mask.float()

    def _normalize(self, audio, target_db=-3):
        """ピーク正規化"""
        peak = torch.abs(audio).max()
        if peak > 0:
            target_level = 10 ** (target_db / 20)
            audio = audio * (target_level / peak)
        return audio

    def karaoke_mix(self, audio_path: str) -> torch.Tensor:
        """カラオケ版を生成（ボーカル除去）"""
        stems = self.separate(audio_path)
        # ボーカル以外を合成
        karaoke = stems["drums"] + stems["bass"] + stems["other"]
        return karaoke
```

### 2.4 高度な分離パイプライン

```python
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

class AdvancedStemSeparation:
    """高度なステム分離パイプライン（品質最大化）"""

    def __init__(self, model_name: str = "htdemucs_ft",
                 device: str = "auto"):
        from demucs.pretrained import get_model
        self.model = get_model(model_name)
        self.model.eval()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

    def separate_high_quality(self, audio_path: str,
                               shifts: int = 5,
                               overlap: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        高品質分離モード

        Parameters:
            shifts: ランダムシフトの回数（高いほど高品質だが遅い）
                    - 1: 高速（通常品質）
                    - 3: バランス（推奨）
                    - 5-10: 最高品質（低速）
            overlap: セグメント間のオーバーラップ率
                    - 0.25: 通常
                    - 0.5: 高品質
                    - 0.75: 最高品質（非常に遅い）
        """
        from demucs.apply import apply_model

        waveform, sr = torchaudio.load(audio_path)

        # モノラル→ステレオ変換
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        if sr != self.model.samplerate:
            resampler = torchaudio.transforms.Resample(sr, self.model.samplerate)
            waveform = resampler(waveform)

        with torch.no_grad():
            sources = apply_model(
                self.model,
                waveform.unsqueeze(0).to(self.device),
                shifts=shifts,
                overlap=overlap,
                progress=True,
            )

        results = {}
        for i, name in enumerate(self.model.sources):
            results[name] = sources[0, i].cpu()

        return results

    def separate_with_wiener(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """
        Wienerフィルタ後処理付き分離

        Demucsの出力に対してWienerフィルタを適用し、
        ブリード（漏れ込み）を低減する
        """
        # 通常の分離
        stems = self.separate_high_quality(audio_path, shifts=3)

        # Wienerフィルタ後処理
        mix, sr = torchaudio.load(audio_path)
        if sr != self.model.samplerate:
            resampler = torchaudio.transforms.Resample(sr, self.model.samplerate)
            mix = resampler(mix)

        refined = self._apply_wiener_filter(mix, stems)
        return refined

    def _apply_wiener_filter(self, mix: torch.Tensor,
                              stems: Dict[str, torch.Tensor],
                              n_fft: int = 4096) -> Dict[str, torch.Tensor]:
        """Wienerフィルタによる分離品質の改善"""
        # ミックスのSTFT
        mix_stft = torch.stft(
            mix, n_fft=n_fft, return_complex=True
        )

        # 各ステムのSTFT
        stem_stfts = {}
        for name, stem in stems.items():
            stem_stfts[name] = torch.stft(
                stem, n_fft=n_fft, return_complex=True
            )

        # Wienerフィルタマスクの計算
        powers = {name: torch.abs(stft) ** 2
                  for name, stft in stem_stfts.items()}
        total_power = sum(powers.values()) + 1e-10

        refined = {}
        for name in stems:
            mask = powers[name] / total_power
            refined_stft = mix_stft * mask

            # 逆STFT
            refined_audio = torch.istft(
                refined_stft, n_fft=n_fft
            )
            refined[name] = refined_audio

        return refined

    def ensemble_separate(self, audio_path: str,
                           models: list = None) -> Dict[str, torch.Tensor]:
        """
        アンサンブル分離（複数モデルの結果を統合）

        複数のモデルで分離し、結果を平均化することで
        個々のモデルのアーティファクトを低減する
        """
        if models is None:
            models = ["htdemucs", "htdemucs_ft"]

        all_stems = []
        for model_name in models:
            from demucs.pretrained import get_model
            model = get_model(model_name)
            model.eval().to(self.device)

            waveform, sr = torchaudio.load(audio_path)
            if sr != model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                waveform = resampler(waveform)

            from demucs.apply import apply_model
            with torch.no_grad():
                sources = apply_model(
                    model,
                    waveform.unsqueeze(0).to(self.device),
                    shifts=3,
                    overlap=0.25,
                )
            all_stems.append(sources)

        # アンサンブル（平均化）
        ensemble = sum(all_stems) / len(all_stems)

        results = {}
        source_names = self.model.sources
        for i, name in enumerate(source_names):
            results[name] = ensemble[0, i].cpu()

        return results


class StemQualityAnalyzer:
    """分離品質の分析ツール"""

    @staticmethod
    def compute_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
        """SDR（Signal-to-Distortion Ratio）の計算"""
        # 長さを合わせる
        min_len = min(len(reference), len(estimated))
        reference = reference[:min_len]
        estimated = estimated[:min_len]

        noise = estimated - reference
        sdr = 10 * np.log10(
            np.sum(reference ** 2) / (np.sum(noise ** 2) + 1e-10)
        )
        return sdr

    @staticmethod
    def compute_sir(reference: np.ndarray, estimated: np.ndarray,
                    interference: np.ndarray) -> float:
        """SIR（Source-to-Interference Ratio）の計算"""
        min_len = min(len(reference), len(estimated), len(interference))
        reference = reference[:min_len]
        estimated = estimated[:min_len]
        interference = interference[:min_len]

        sir = 10 * np.log10(
            np.sum(reference ** 2) / (np.sum(interference ** 2) + 1e-10)
        )
        return sir

    @staticmethod
    def compute_sar(reference: np.ndarray, estimated: np.ndarray) -> float:
        """SAR（Source-to-Artifact Ratio）の計算"""
        min_len = min(len(reference), len(estimated))
        reference = reference[:min_len]
        estimated = estimated[:min_len]

        # アーティファクト = 推定 - 参照 - 干渉
        artifact = estimated - reference
        sar = 10 * np.log10(
            np.sum(reference ** 2) / (np.sum(artifact ** 2) + 1e-10)
        )
        return sar

    def full_evaluation(self, reference_stems: dict,
                        estimated_stems: dict) -> dict:
        """全ステムの品質評価"""
        results = {}
        for name in reference_stems:
            if name in estimated_stems:
                ref = reference_stems[name]
                est = estimated_stems[name]
                if isinstance(ref, torch.Tensor):
                    ref = ref.numpy()
                if isinstance(est, torch.Tensor):
                    est = est.numpy()
                # モノラル化
                if ref.ndim == 2:
                    ref = ref.mean(axis=0)
                if est.ndim == 2:
                    est = est.mean(axis=0)

                results[name] = {
                    "SDR (dB)": round(self.compute_sdr(ref, est), 2),
                    "SAR (dB)": round(self.compute_sar(ref, est), 2),
                }
        return results
```

### 2.5 メモリ効率の良い長時間楽曲の処理

```python
import torch
import torchaudio
import numpy as np
from typing import Optional

class LongTrackSeparator:
    """長時間楽曲のメモリ効率的なステム分離"""

    def __init__(self, model_name: str = "htdemucs_ft",
                 max_memory_gb: float = 4.0):
        from demucs.pretrained import get_model
        self.model = get_model(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_memory_gb = max_memory_gb

    def _estimate_chunk_size(self, channels: int, sr: int) -> int:
        """利用可能なメモリから最適なチャンクサイズを推定"""
        # GPU VRAMに基づく推定（概算）
        if self.device.type == "cuda":
            total_mem = torch.cuda.get_device_properties(0).total_memory
            available = min(total_mem * 0.7, self.max_memory_gb * 1e9)
        else:
            available = self.max_memory_gb * 1e9

        # 1サンプルあたりのメモリ使用量（概算: モデルの4倍程度）
        bytes_per_sample = channels * 4 * 4  # float32 * 4x overhead
        max_samples = int(available / bytes_per_sample)

        # 最大30秒、最小5秒に制限
        max_seconds = 30
        min_seconds = 5
        chunk_seconds = np.clip(max_samples / sr, min_seconds, max_seconds)

        return int(chunk_seconds * sr)

    def separate_long_track(self, audio_path: str,
                             output_dir: str,
                             overlap_seconds: float = 2.0) -> dict:
        """
        長時間楽曲をチャンク分割して分離

        Parameters:
            audio_path: 入力音声パス
            output_dir: 出力ディレクトリ
            overlap_seconds: チャンク間のオーバーラップ（秒）
        """
        from demucs.apply import apply_model
        from pathlib import Path

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.model.samplerate:
            resampler = torchaudio.transforms.Resample(sr, self.model.samplerate)
            waveform = resampler(waveform)
            sr = self.model.samplerate

        total_samples = waveform.shape[-1]
        total_seconds = total_samples / sr
        print(f"楽曲長: {total_seconds:.1f}秒")

        chunk_size = self._estimate_chunk_size(waveform.shape[0], sr)
        overlap = int(overlap_seconds * sr)
        print(f"チャンクサイズ: {chunk_size / sr:.1f}秒, "
              f"オーバーラップ: {overlap / sr:.1f}秒")

        # 出力バッファ
        n_sources = len(self.model.sources)
        output = torch.zeros(n_sources, waveform.shape[0], total_samples)
        weight = torch.zeros(total_samples)

        # チャンク処理
        pos = 0
        chunk_idx = 0
        while pos < total_samples:
            end = min(pos + chunk_size, total_samples)
            chunk = waveform[:, pos:end]

            print(f"チャンク {chunk_idx}: {pos/sr:.1f}s - {end/sr:.1f}s")

            with torch.no_grad():
                sources = apply_model(
                    self.model,
                    chunk.unsqueeze(0).to(self.device),
                    shifts=1,
                    overlap=0.25,
                )

            # クロスフェードウィンドウ
            chunk_len = end - pos
            fade = self._make_crossfade_window(chunk_len, overlap)

            for i in range(n_sources):
                output[i, :, pos:end] += sources[0, i].cpu() * fade
            weight[pos:end] += fade

            # メモリ解放
            del sources
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            pos += chunk_size - overlap
            chunk_idx += 1

        # 重み正規化
        weight = torch.clamp(weight, min=1e-8)
        for i in range(n_sources):
            output[i] /= weight

        # 保存
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results = {}
        for i, name in enumerate(self.model.sources):
            path = output_path / f"{name}.wav"
            torchaudio.save(str(path), output[i], sr)
            results[name] = str(path)
            print(f"保存: {path}")

        return results

    def _make_crossfade_window(self, length: int, overlap: int) -> torch.Tensor:
        """クロスフェードウィンドウの生成"""
        window = torch.ones(length)
        if overlap > 0 and overlap < length:
            # フェードイン
            fade_in = torch.linspace(0, 1, overlap)
            window[:overlap] = fade_in
            # フェードアウト
            fade_out = torch.linspace(1, 0, overlap)
            window[-overlap:] = fade_out
        return window
```

---

## 3. クラウドサービスの利用

### 3.1 LALAL.AI API

```python
import requests
import time

class LALALClient:
    """LALAL.AI ステム分離クライアント（概念）"""

    BASE_URL = "https://www.lalal.ai/api/v1"

    def __init__(self, api_key: str):
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def separate(self, audio_path: str, stem_type: str = "vocals") -> dict:
        """
        ステム分離を実行

        stem_type options:
        - vocals: ボーカル/伴奏
        - drums: ドラム分離
        - bass: ベース分離
        - electric_guitar: エレキギター分離
        - piano: ピアノ分離
        - synthesizer: シンセサイザー分離
        """
        # Step 1: ファイルアップロード
        with open(audio_path, "rb") as f:
            upload_resp = requests.post(
                f"{self.BASE_URL}/upload",
                files={"file": f},
                data={"stem": stem_type},
                headers=self.headers,
            )
        task_id = upload_resp.json()["task_id"]

        # Step 2: 処理完了を待機
        while True:
            status = requests.get(
                f"{self.BASE_URL}/status/{task_id}",
                headers=self.headers,
            ).json()
            if status["state"] == "done":
                return status["result"]
            elif status["state"] == "error":
                raise Exception(f"分離失敗: {status['error']}")
            time.sleep(2)
```

### 3.2 Spleeter の使い方

```python
class SpleeterPipeline:
    """
    Spleeter（Deezer）によるステム分離

    特徴:
    - Tensorflow ベース
    - 非常に高速（GPUなしでもリアルタイム以下）
    - 2stems/4stems/5stems モデル
    - 品質はDemucsに劣るが速度で優位
    """

    def __init__(self, n_stems: int = 2):
        """
        n_stems: 分離ステム数
        - 2: vocals/accompaniment
        - 4: vocals/drums/bass/other
        - 5: vocals/drums/bass/piano/other
        """
        from spleeter.separator import Separator
        self.separator = Separator(f'spleeter:{n_stems}stems')
        self.n_stems = n_stems

    def separate(self, audio_path: str, output_dir: str = "./output"):
        """分離を実行"""
        self.separator.separate_to_file(
            audio_path,
            output_dir,
            codec="wav",
            bitrate="320k",
        )

    def separate_to_dict(self, audio_path: str) -> dict:
        """分離結果をNumPy配列として取得"""
        import numpy as np
        from spleeter.audio.adapter import AudioAdapter

        adapter = AudioAdapter.default()
        waveform, rate = adapter.load(audio_path, sample_rate=44100)

        prediction = self.separator.separate(waveform)
        return prediction  # {"vocals": np.ndarray, "accompaniment": np.ndarray, ...}

    def quick_vocal_extract(self, audio_path: str, output_path: str):
        """簡易ボーカル抽出"""
        import soundfile as sf
        results = self.separate_to_dict(audio_path)
        vocals = results["vocals"]
        sf.write(output_path, vocals, 44100)
```

### 3.3 分離結果の後処理パイプライン

```python
import numpy as np
import torch
import torchaudio

class StemPostProcessor:
    """分離ステムの後処理"""

    def __init__(self, sr: int = 44100):
        self.sr = sr

    def remove_bleed(self, stem: np.ndarray, mix: np.ndarray,
                     other_stems: list, threshold_db: float = -40) -> np.ndarray:
        """
        ブリード（他ステムの漏れ込み）除去

        スペクトラルゲーティングにより、他のステムに支配されている
        時間-周波数ビンのエネルギーを抑制する
        """
        n_fft = 4096
        hop = 1024

        # STFT計算
        from scipy.signal import stft as scipy_stft, istft as scipy_istft

        _, _, stem_stft = scipy_stft(stem, fs=self.sr, nperseg=n_fft,
                                      noverlap=n_fft - hop)
        stem_power = np.abs(stem_stft) ** 2

        # 他ステムの合計パワー
        other_power = np.zeros_like(stem_power)
        for other in other_stems:
            _, _, other_stft = scipy_stft(other, fs=self.sr, nperseg=n_fft,
                                           noverlap=n_fft - hop)
            other_power += np.abs(other_stft) ** 2

        # マスク: ステムが支配的な部分のみ通過
        ratio = stem_power / (stem_power + other_power + 1e-10)
        threshold = 10 ** (threshold_db / 10)
        mask = np.where(ratio > 0.3, 1.0, ratio / 0.3)

        # マスク適用
        cleaned_stft = stem_stft * mask

        # 逆STFT
        _, cleaned = scipy_istft(cleaned_stft, fs=self.sr, nperseg=n_fft,
                                  noverlap=n_fft - hop)

        return cleaned[:len(stem)]

    def smooth_transitions(self, stem: np.ndarray,
                           fade_ms: float = 5.0) -> np.ndarray:
        """
        エッジスムージング

        分離時のアーティファクト（急激な立ち上がり/立ち下がり）を
        軽いフェードで緩和する
        """
        fade_samples = int(fade_ms * self.sr / 1000)
        if fade_samples < 2:
            return stem

        # 音声区間を検出
        threshold = np.max(np.abs(stem)) * 0.01
        is_active = np.abs(stem) > threshold

        # 立ち上がり/立ち下がりエッジの検出
        edges = np.diff(is_active.astype(int))
        onsets = np.where(edges == 1)[0]
        offsets = np.where(edges == -1)[0]

        result = stem.copy()

        # フェードイン適用
        for onset in onsets:
            start = max(0, onset - fade_samples // 2)
            end = min(len(stem), onset + fade_samples // 2)
            fade = np.linspace(0, 1, end - start)
            result[start:end] *= fade

        # フェードアウト適用
        for offset in offsets:
            start = max(0, offset - fade_samples // 2)
            end = min(len(stem), offset + fade_samples // 2)
            fade = np.linspace(1, 0, end - start)
            result[start:end] *= fade

        return result

    def phase_align(self, stem: np.ndarray,
                    reference: np.ndarray) -> np.ndarray:
        """
        位相整合

        分離ステムと元のミックスの位相を比較し、
        ステムの位相ずれを補正する
        """
        min_len = min(len(stem), len(reference))
        stem = stem[:min_len]
        reference = reference[:min_len]

        # 相互相関で最適な遅延を推定
        correlation = np.correlate(stem, reference, mode='full')
        delay = np.argmax(np.abs(correlation)) - len(stem) + 1

        # 遅延補正
        if delay > 0:
            aligned = np.concatenate([np.zeros(delay), stem[:-delay]])
        elif delay < 0:
            aligned = np.concatenate([stem[-delay:], np.zeros(-delay)])
        else:
            aligned = stem

        # 極性チェック（反転している場合の補正）
        corr_normal = np.sum(aligned * reference)
        corr_inverted = np.sum(-aligned * reference)
        if corr_inverted > corr_normal:
            aligned = -aligned

        return aligned
```

---

## 4. 比較表

### 4.1 主要ステム分離ツール比較

| 項目 | Demucs v4 | Spleeter | LALAL.AI | iZotope RX |
|------|----------|----------|---------|------------|
| 種別 | OSS | OSS | SaaS | 商用ソフト |
| 品質(SDR) | 9.0+ dB | 5.9 dB | 8.5+ dB | 8.0+ dB |
| ステム数 | 4/6 | 2/4/5 | 最大8 | 柔軟 |
| 速度 | 中程度 | 高速 | 中程度 | 中程度 |
| GPU必須 | 推奨 | 不要 | 不要 | 不要 |
| オフライン | 可能 | 可能 | 不可 | 可能 |
| コスト | 無料 | 無料 | 従量課金 | $399+ |
| APIアクセス | Python | Python | REST | プラグイン |

### 4.2 用途別推奨ツール

| ユースケース | 推奨 | 理由 |
|-------------|------|------|
| DJ向けアカペラ抽出 | Demucs v4 | 最高品質、無料 |
| カラオケ作成 | Demucs / LALAL.AI | ボーカル除去品質 |
| リミックス素材 | Demucs v4 (6stems) | 細かい楽器分離 |
| ポッドキスト音声分離 | Spleeter | 高速、2stems十分 |
| プロ品質マスタリング | iZotope RX | 最高の柔軟性 |
| バッチ処理 | Demucs CLI | スクリプト化容易 |
| 非エンジニア | LALAL.AI | Web UIで簡単 |

### 4.3 モデル別Demucsバリエーション

| モデル名 | パラメータ数 | ステム数 | SDR(ボーカル) | 特徴 |
|---------|-----------|---------|-------------|------|
| htdemucs | 83M | 4 | 8.5 dB | 標準モデル |
| htdemucs_ft | 83M | 4 | 9.0 dB | ファインチューニング版 |
| htdemucs_6s | 83M | 6 | 7.8 dB | ギター・ピアノ追加 |
| mdx_extra | - | 4 | 8.3 dB | MDXモデル（軽量） |
| mdx_extra_q | - | 4 | 8.0 dB | MDX量子化版（最軽量） |

### 4.4 処理速度の比較（5分のステレオ楽曲）

| モデル | GPU (RTX 3060) | GPU (RTX 4090) | CPU (i7-13700) | Apple M2 |
|-------|----------------|----------------|---------------|----------|
| htdemucs_ft | 25秒 | 12秒 | 3分 | 1.5分 |
| htdemucs_6s | 30秒 | 15秒 | 4分 | 2分 |
| Spleeter 2stems | 5秒 | 3秒 | 15秒 | 10秒 |
| Spleeter 5stems | 12秒 | 6秒 | 40秒 | 25秒 |

---

## 5. アンチパターン

### 5.1 アンチパターン: 分離品質の過信

```python
# BAD: 分離結果をそのまま使用
def bad_remix(song_path):
    stems = separate(song_path)
    # 問題: ボーカルステムに伴奏の漏れ込み(bleed)がある
    # 問題: ドラムステムにゴースト音が含まれる
    return mix(stems["vocals"] * 1.5, stems["drums"], stems["bass"])

# GOOD: アーティファクト対策を含む
def good_remix(song_path):
    stems = separate(song_path)

    # 1. ボーカルのブリード除去
    vocals = apply_spectral_gate(stems["vocals"], threshold=-40)

    # 2. クロスフェード処理でアーティファクト軽減
    vocals = smooth_edges(vocals, fade_ms=10)

    # 3. 位相整合（分離時の位相ずれ補正）
    drums = phase_align(stems["drums"], reference=original_mix)

    return mix(vocals * 1.5, drums, stems["bass"], stems["other"])
```

### 5.2 アンチパターン: メモリ管理の不備

```python
# BAD: 長い楽曲をメモリに全ロード
def bad_separate_long(audio_path):
    audio, sr = torchaudio.load(audio_path)  # 10分 = ~100MB
    # GPU VRAM不足でクラッシュ
    sources = model(audio.unsqueeze(0).cuda())

# GOOD: チャンク処理でメモリ制御
def good_separate_long(audio_path, chunk_seconds=30, overlap_seconds=5):
    """長い楽曲をチャンク分割して処理"""
    audio, sr = torchaudio.load(audio_path)
    chunk_size = chunk_seconds * sr
    overlap = overlap_seconds * sr

    all_sources = []
    pos = 0
    while pos < audio.shape[-1]:
        end = min(pos + chunk_size, audio.shape[-1])
        chunk = audio[:, pos:end]

        with torch.no_grad():
            sources = apply_model(model, chunk.unsqueeze(0).cuda())
            all_sources.append(sources.cpu())

        # メモリ解放
        torch.cuda.empty_cache()
        pos += chunk_size - overlap

    # オーバーラップ部分をクロスフェードして結合
    return crossfade_merge(all_sources, overlap)
```

### 5.3 アンチパターン: モデル選択のミス

```python
# BAD: 用途に合わないモデルを使用
def bad_model_selection():
    # 歌声変換の前処理にSpleeterを使用 → 品質不足
    spleeter_result = spleeter_2stems("song.wav")
    rvc_convert(spleeter_result["vocals"])  # 入力の品質が低い

    # 大量バッチ処理にhtdemucs_ft + shifts=10 → 時間がかかりすぎ
    for song in glob("*.wav"):
        demucs_separate(song, model="htdemucs_ft", shifts=10)

# GOOD: 用途に応じたモデルと設定の使い分け
def good_model_selection(use_case: str, audio_path: str):
    """用途に最適なモデルと設定を自動選択"""
    configs = {
        "voice_conversion": {
            "model": "htdemucs_ft",
            "shifts": 3,
            "two_stems": "vocals",
            "reason": "ボーカル品質最優先、2ステムで効率化",
        },
        "batch_karaoke": {
            "model": "htdemucs",
            "shifts": 1,
            "two_stems": "vocals",
            "reason": "速度重視、品質は十分",
        },
        "remix_production": {
            "model": "htdemucs_6s",
            "shifts": 5,
            "two_stems": None,
            "reason": "楽器別分離、最高品質",
        },
        "quick_preview": {
            "model": "mdx_extra_q",
            "shifts": 1,
            "two_stems": "vocals",
            "reason": "最速プレビュー",
        },
    }

    config = configs.get(use_case, configs["batch_karaoke"])
    print(f"選択: {config['model']} ({config['reason']})")
    return config
```

### 5.4 アンチパターン: 分離ステムの再構成不整合

```python
# BAD: 分離ステムを合算してもオリジナルに戻らない
def bad_reconstruct(stems):
    # 各ステムを個別に加工してから合成
    vocals = normalize(stems["vocals"])      # 音量変更
    drums = eq(stems["drums"])               # EQ変更
    bass = compress(stems["bass"])           # ダイナミクス変更
    other = stems["other"]

    remix = vocals + drums + bass + other
    # 問題: 合計がオリジナルと大幅に異なる
    return remix

# GOOD: 分離→加工→再構成の一貫性を保つ
def good_reconstruct(stems, mix_original):
    """分離・加工後もオリジナルとの一貫性を維持"""
    vocals = normalize(stems["vocals"])
    drums = eq(stems["drums"])
    bass = compress(stems["bass"])
    other = stems["other"]

    remix = vocals + drums + bass + other

    # 残差（分離誤差）の補正
    residual = mix_original - sum(stems.values())
    remix += residual  # 分離時の誤差を戻す

    # ラウドネスをオリジナルに合わせる
    original_rms = np.sqrt(np.mean(mix_original ** 2))
    remix_rms = np.sqrt(np.mean(remix ** 2))
    if remix_rms > 0:
        remix = remix * (original_rms / remix_rms)

    return remix
```

---

## 6. 実践的なユースケース

### 6.1 カラオケ動画自動生成

```python
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path

class KaraokeGenerator:
    """カラオケ版の自動生成パイプライン"""

    def __init__(self):
        self.pipeline = StemSeparationPipeline(model_name="htdemucs_ft")

    def create_karaoke(self, input_path: str, output_dir: str,
                       keep_backing_vocals: bool = False) -> dict:
        """
        カラオケ版を生成

        Parameters:
            keep_backing_vocals: Trueの場合、コーラス/ハモリは維持
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ステム分離
        stems = self.pipeline.separate(input_path)

        # カラオケミックス（ボーカルなし）
        karaoke = stems["drums"] + stems["bass"] + stems["other"]

        if keep_backing_vocals:
            # メインボーカルのみ除去（コーラスは維持）
            # 中央定位のボーカルのみを除去する
            vocals = stems["vocals"]
            mid = (vocals[0] + vocals[1]) / 2  # モノラル成分=メインボーカル
            side = (vocals[0] - vocals[1]) / 2  # ステレオ成分=コーラス/ハモリ
            # コーラス成分を戻す
            chorus = torch.stack([side, -side])
            karaoke += chorus * 0.7

        # ボーカルガイド版（ボーカル音量を下げた版）
        guide = karaoke + stems["vocals"] * 0.15

        # 保存
        sr = self.pipeline.model.samplerate
        results = {
            "karaoke": str(output_path / "karaoke.wav"),
            "guide": str(output_path / "guide.wav"),
            "vocals": str(output_path / "vocals.wav"),
            "instrumental": str(output_path / "instrumental.wav"),
        }

        torchaudio.save(results["karaoke"], karaoke, sr)
        torchaudio.save(results["guide"], guide, sr)
        torchaudio.save(results["vocals"], stems["vocals"], sr)
        torchaudio.save(results["instrumental"], karaoke, sr)

        return results


class DJToolkit:
    """DJ向けステム分離ツールキット"""

    def __init__(self):
        self.pipeline = StemSeparationPipeline(model_name="htdemucs_ft")

    def extract_acapella(self, song_path: str, output_path: str,
                          quality: str = "high"):
        """アカペラ抽出"""
        stems = self.pipeline.separate(song_path)
        vocals = stems["vocals"]

        # 品質に応じた後処理
        if quality == "high":
            # ノイズゲート + スペクトラルクリーニング
            vocals = self._spectral_clean(vocals)

        torchaudio.save(output_path, vocals, self.pipeline.model.samplerate)

    def create_drum_break(self, song_path: str, output_path: str):
        """ドラムブレイク抽出"""
        stems = self.pipeline.separate(song_path)
        drums = stems["drums"]

        # ドラムステムの品質向上
        drums = self._enhance_drums(drums)

        torchaudio.save(output_path, drums, self.pipeline.model.samplerate)

    def create_loop_pack(self, song_path: str, output_dir: str,
                          bpm: float = None):
        """ループパック生成（各ステムをBPM同期で分割）"""
        stems = self.pipeline.separate(song_path)
        sr = self.pipeline.model.samplerate

        # BPM検出（指定されていない場合）
        if bpm is None:
            import librosa
            mix_np = stems["drums"].numpy().mean(axis=0)
            tempo, _ = librosa.beat.beat_track(y=mix_np, sr=sr)
            bpm = float(tempo)

        # 1小節のサンプル数
        bar_samples = int(4 * 60 / bpm * sr)  # 4拍=1小節

        output_path = Path(output_dir)
        for name, stem in stems.items():
            stem_dir = output_path / name
            stem_dir.mkdir(parents=True, exist_ok=True)

            n_bars = stem.shape[-1] // bar_samples
            for i in range(min(n_bars, 16)):  # 最大16小節
                start = i * bar_samples
                end = start + bar_samples
                loop = stem[:, start:end]

                loop_path = stem_dir / f"loop_{i+1:02d}.wav"
                torchaudio.save(str(loop_path), loop, sr)

        print(f"BPM: {bpm:.1f}, {n_bars}小節をエクスポート")

    def _spectral_clean(self, audio):
        """スペクトラルクリーニング"""
        return audio  # プレースホルダー

    def _enhance_drums(self, drums):
        """ドラムステムの品質向上"""
        return drums  # プレースホルダー
```

### 6.2 音楽教育・練習支援

```python
class MusicPracticeHelper:
    """音楽教育・練習支援ツール"""

    def __init__(self):
        self.pipeline = StemSeparationPipeline(model_name="htdemucs_ft")

    def create_practice_tracks(self, song_path: str,
                                instrument: str,
                                output_dir: str) -> dict:
        """
        練習用トラックを生成

        - 指定楽器のみのソロトラック
        - 指定楽器を除いたマイナスワントラック
        - 指定楽器の音量を調整可能なミックス
        """
        stems = self.pipeline.separate(song_path)
        sr = self.pipeline.model.samplerate

        stem_map = {
            "vocal": "vocals",
            "guitar": "other",  # 4ステムモデルではotherに含まれる
            "bass": "bass",
            "drums": "drums",
        }

        target_stem = stem_map.get(instrument, "vocals")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # ソロトラック（指定楽器のみ）
        solo = stems[target_stem]
        solo_path = output_path / f"{instrument}_solo.wav"
        torchaudio.save(str(solo_path), solo, sr)
        results["solo"] = str(solo_path)

        # マイナスワン（指定楽器を除去）
        minus_one = sum(s for n, s in stems.items() if n != target_stem)
        minus_path = output_path / f"minus_{instrument}.wav"
        torchaudio.save(str(minus_path), minus_one, sr)
        results["minus_one"] = str(minus_path)

        # 音量調整版（楽器25%, 50%, 75%）
        for level in [0.25, 0.50, 0.75]:
            mixed = minus_one + solo * level
            level_path = output_path / f"{instrument}_{int(level*100)}pct.wav"
            torchaudio.save(str(level_path), mixed, sr)
            results[f"level_{int(level*100)}"] = str(level_path)

        return results

    def create_slow_practice(self, song_path: str,
                              tempo_factor: float = 0.75,
                              output_path: str = "slow_practice.wav"):
        """テンポを落とした練習版を生成（ピッチ維持）"""
        import librosa

        stems = self.pipeline.separate(song_path)
        sr = self.pipeline.model.samplerate

        # 各ステムのテンポ変更（ピッチ維持）
        slowed_stems = {}
        for name, stem in stems.items():
            stem_np = stem.numpy()
            slowed_channels = []
            for ch in range(stem_np.shape[0]):
                slowed = librosa.effects.time_stretch(
                    stem_np[ch], rate=tempo_factor
                )
                slowed_channels.append(slowed)
            slowed_stems[name] = torch.tensor(np.array(slowed_channels))

        # リミックス
        slow_mix = sum(slowed_stems.values())
        torchaudio.save(output_path, slow_mix, sr)

        return output_path
```

---

## 6. FAQ

### Q1: ステム分離の品質を測る指標は何ですか？

SDR（Signal-to-Distortion Ratio）が最も広く使われる指標で、dB単位で表されます。高いほど分離品質が高く、Demucs v4は約9dB以上を達成しています。他にSIR（Source-to-Interference Ratio、他ソースの漏れ込み量）、SAR（Source-to-Artifact Ratio、アーティファクトの量）もあります。ただし、数値上の品質と聴感上の品質は必ずしも一致しないため、最終的には聴感評価も重要です。

### Q2: ボーカル以外の楽器（ギター、ピアノなど）を個別に分離できますか？

Demucs v4の6ステムモデル（htdemucs_6s）ではドラム、ベース、ボーカル、ギター、ピアノ、その他の6トラックに分離可能です。LALAL.AIはさらに細かく、エレキギター、アコースティックギター、ピアノ、シンセサイザーなど最大8種類の分離に対応しています。ただし、分離するステム数が増えるほど個々の品質は低下する傾向があります。

### Q3: リアルタイムでのステム分離は可能ですか？

2025年時点では、高品質なリアルタイムステム分離は困難です。Demucs v4はGPU使用でもリアルタイムの3-5倍の処理時間が必要です。ただし、軽量モデル（Spleeter、Open-Unmix）やDemucsのストリーミングモードを使えば準リアルタイム（1-2秒遅延）での処理は可能です。DJソフトウェア（djay、Traktor）には組み込みのリアルタイムステム分離が搭載されており、品質と速度のトレードオフの上で実用化されています。

### Q4: 分離結果の品質が悪い場合に改善する方法は？

いくつかのテクニックがあります。(1) shiftsパラメータを増やす（3-10）: ランダムシフト平均化で品質向上。(2) overlapを増やす（0.25→0.5）: セグメント境界のアーティファクト低減。(3) モデルアンサンブル: htdemucsとhtdemucs_ftの結果を平均化。(4) Wienerフィルタ後処理: 分離結果にWienerフィルタを適用してブリード除去。(5) 入力品質の改善: 高ビットレートの音源を使用し、事前にノイズ除去を行う。

### Q5: ステム分離を音楽制作のワークフローに組み込むには？

DAW（Digital Audio Workstation）との統合方法として、(1) VST/AUプラグイン: iZotope RX、Spectralayers等のプラグインを使用。(2) バッチ前処理: Demucs CLIでフォルダ単位で分離し、結果をDAWにインポート。(3) Python統合: Pythonスクリプトで分離→後処理→エクスポートを自動化し、DAWのセッションフォルダに直接出力。(4) リモートサーバー: GPU搭載サーバーでAPIを構築し、DAWからHTTPリクエストで分離を実行。

### Q6: 著作権上、ステム分離した音源をどこまで利用できますか？

著作権法上、既存楽曲のステム分離は「複製」に該当する可能性があります。個人的な練習・学習目的であれば私的使用として許容される場合が多いですが、分離結果を公開・配布・商用利用する場合は著作権者の許諾が必要です。特にサンプリングやリミックスとして配信する場合はライセンス取得が必須です。DJプレイでの使用は国・地域の著作権法と会場のライセンス契約に依存します。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 技術進化 | マスキング→U-Net→ハイブリッド（時間+周波数）へ |
| Demucs v4 | 最高品質のOSS。4ステム/6ステム対応 |
| 品質指標 | SDR 9dB+が現在の最高水準 |
| 主な制約 | ブリード（漏れ込み）、アーティファクト、位相ずれ |
| 後処理 | ノイズゲート、スペクトラルゲート、位相整合が重要 |
| メモリ管理 | 長い楽曲はチャンク分割 + クロスフェード結合 |
| モデル選択 | 用途に応じてhtdemucs_ft/htdemucs_6s/Spleeterを使い分け |
| Wienerフィルタ | 後処理で分離品質を追加改善可能 |

## 次に読むべきガイド

- [02-audio-effects.md](./02-audio-effects.md) — AI音声エフェクト（EQ、ノイズ除去）
- [00-music-generation.md](./00-music-generation.md) — 音楽生成との組み合わせ
- [../03-development/01-audio-processing.md](../03-development/01-audio-processing.md) — librosa/torchaudioによる実装

## 参考文献

1. Rouard, S., et al. (2023). "Hybrid Transformers for Music Source Separation" — HTDemucs論文。ハイブリッドTransformerによる音楽ソース分離
2. Défossez, A. (2021). "Hybrid Spectrogram and Waveform Source Separation" — Demucs v3論文。スペクトログラム+波形のハイブリッドアプローチ
3. Hennequin, R., et al. (2020). "Spleeter: a fast and efficient music source separation tool" — Spleeter論文。Deezerによる軽量分離ツール
4. Stöter, F.R., et al. (2019). "Open-Unmix - A Reference Implementation for Music Source Separation" — Open-Unmix。オープンソースのリファレンス実装
5. Vincent, E., et al. (2006). "Performance measurement in blind audio source separation" — SDR/SIR/SAR評価指標の定義論文
6. Uhlich, S., et al. (2017). "Improving music source separation based on deep neural networks through data augmentation and network blending" — データ拡張とネットワーク統合による品質改善
