# 音声エフェクト — AI EQ、ノイズ除去、マスタリング

> AIを活用した音声エフェクト処理（EQ、ノイズ除去、マスタリング）の技術と実装を解説する

## この章で学ぶこと

1. 従来の音声エフェクトとAIエフェクトの違い、主要な処理カテゴリ
2. AIノイズ除去、自動EQ、自動マスタリングの技術的仕組みと実装
3. 実践的なエフェクトチェーン構築と品質改善パターン

---

## 1. 音声エフェクトの基礎

### 1.1 エフェクトチェーンの構成

```
標準的な音声エフェクトチェーン
==================================================

入力音声
  │
  ▼
┌──────────────┐
│ 1. ノイズ除去 │  背景ノイズ、ハム音除去
│   (Denoise)  │  ← AI が最も効果的
└──────┬───────┘
       ▼
┌──────────────┐
│ 2. EQ        │  周波数バランス調整
│  (Equalizer) │  ← AI で自動最適化
└──────┬───────┘
       ▼
┌──────────────┐
│ 3. コンプ    │  ダイナミックレンジ圧縮
│ (Compressor) │  音量差を均一化
└──────┬───────┘
       ▼
┌──────────────┐
│ 4. リバーブ   │  空間的広がり付与
│  (Reverb)    │
└──────┬───────┘
       ▼
┌──────────────┐
│ 5. リミッター │  ピーク制御
│  (Limiter)   │  クリッピング防止
└──────┬───────┘
       ▼
┌──────────────┐
│ 6. マスタリング│  最終調整
│  (Mastering) │  ← AI で自動化可能
└──────┬───────┘
       ▼
  出力音声
==================================================
```

### 1.2 従来 vs AI エフェクトの比較

```python
# 従来のノイズ除去 vs AIノイズ除去の比較

import numpy as np

# 従来手法: スペクトルサブトラクション
def spectral_subtraction(noisy_signal, noise_profile, sr=16000):
    """
    スペクトルサブトラクション法
    - ノイズプロファイル（無音区間）を事前に推定
    - ノイズスペクトルをクリーン音声から減算
    - 制限: 定常ノイズにのみ有効、ミュージカルノイズ発生
    """
    n_fft = 2048
    hop = 512

    # STFT
    noisy_stft = np.fft.rfft(noisy_signal)
    noise_stft = np.fft.rfft(noise_profile)

    # スペクトル減算
    noise_power = np.abs(noise_stft) ** 2
    noisy_power = np.abs(noisy_stft) ** 2
    clean_power = np.maximum(noisy_power - noise_power, 0)

    # 位相は元のまま保持
    phase = np.angle(noisy_stft)
    clean_stft = np.sqrt(clean_power) * np.exp(1j * phase)

    return np.fft.irfft(clean_stft)

# AI手法: ニューラルネットワーク
def ai_denoise(noisy_signal, model):
    """
    AIノイズ除去
    - 非定常ノイズにも対応
    - ノイズプロファイル不要
    - ミュージカルノイズが発生しにくい
    """
    # 前処理
    mel_spec = compute_mel_spectrogram(noisy_signal)
    # マスク推定（U-Net型モデル）
    clean_mask = model.predict(mel_spec)
    # マスク適用
    clean_spec = mel_spec * clean_mask
    # 逆変換
    return inverse_mel_spectrogram(clean_spec)
```

### 1.3 エフェクトの信号処理的分類

音声エフェクトは信号処理の観点から、大きく以下のカテゴリに分類できる。それぞれのカテゴリでAIがどのように従来手法を改善しているかを理解することが重要である。

```
音声エフェクトの信号処理的分類
==================================================

1. 時間領域エフェクト
   ├── ディレイ（遅延）
   ├── リバーブ（残響）
   ├── コーラス（揺れ）
   └── フランジャー / フェイザー

2. 周波数領域エフェクト
   ├── EQ（周波数バランス調整）
   ├── ローパス / ハイパスフィルタ
   ├── ノッチフィルタ（特定周波数除去）
   └── ワウ（周波数スイープ）

3. ダイナミクス系エフェクト
   ├── コンプレッサー（音量差圧縮）
   ├── リミッター（ピーク制御）
   ├── ノイズゲート（無音時のノイズ除去）
   └── エキスパンダー（静音部をさらに静かに）

4. ノイズ処理系
   ├── ノイズ除去（定常/非定常）
   ├── デリバーブ（残響除去）
   ├── ディエッサー（歯擦音除去）
   └── ハム除去（電源ノイズ除去）

5. 空間系エフェクト
   ├── パンニング（左右配置）
   ├── ステレオイメージング
   ├── バイノーラル処理（3D音響）
   └── HRTF（頭部伝達関数）

AIの得意領域:
  ★★★ ノイズ処理系 — 非定常ノイズ対応で圧倒的優位
  ★★☆ 周波数領域 — ターゲットプロファイル自動調整
  ★★☆ ダイナミクス系 — 文脈に応じた自動パラメータ
  ★☆☆ 空間系 — ルーム推定と自動リバーブ
  ★☆☆ 時間領域 — 創造的エフェクトは人間が優位
==================================================
```

### 1.4 音声処理における基本概念の詳細

```python
import numpy as np
from scipy import signal

class AudioEffectsFoundation:
    """音声エフェクトの基礎概念を示すクラス"""

    @staticmethod
    def compute_stft(audio: np.ndarray, n_fft: int = 2048,
                     hop_length: int = 512) -> np.ndarray:
        """
        短時間フーリエ変換（STFT）
        - 音声を時間-周波数表現に変換
        - エフェクト処理の基盤となる変換
        - n_fft: FFTウィンドウサイズ（周波数解像度に影響）
        - hop_length: ウィンドウ間隔（時間解像度に影響）
        """
        window = np.hanning(n_fft)
        n_frames = (len(audio) - n_fft) // hop_length + 1
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft] * window
            stft[:, i] = np.fft.rfft(frame)

        return stft

    @staticmethod
    def compute_magnitude_phase(stft: np.ndarray):
        """スペクトログラムを振幅と位相に分解"""
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        return magnitude, phase

    @staticmethod
    def reconstruct_from_stft(magnitude: np.ndarray, phase: np.ndarray,
                               hop_length: int = 512) -> np.ndarray:
        """振幅と位相からSTFTを再構成し、逆変換で音声に戻す"""
        stft = magnitude * np.exp(1j * phase)
        n_fft = (stft.shape[0] - 1) * 2
        n_frames = stft.shape[1]
        output_length = n_fft + (n_frames - 1) * hop_length
        output = np.zeros(output_length)
        window = np.hanning(n_fft)

        for i in range(n_frames):
            start = i * hop_length
            frame = np.fft.irfft(stft[:, i])
            output[start:start + n_fft] += frame * window

        return output

    @staticmethod
    def db_to_linear(db: float) -> float:
        """デシベルをリニア値に変換"""
        return 10 ** (db / 20)

    @staticmethod
    def linear_to_db(linear: float) -> float:
        """リニア値をデシベルに変換"""
        return 20 * np.log10(max(linear, 1e-10))

    @staticmethod
    def compute_rms(audio: np.ndarray) -> float:
        """RMS（二乗平均平方根）レベルを計算"""
        return np.sqrt(np.mean(audio ** 2))

    @staticmethod
    def compute_peak(audio: np.ndarray) -> float:
        """ピークレベルを計算"""
        return np.max(np.abs(audio))

    @staticmethod
    def compute_crest_factor(audio: np.ndarray) -> float:
        """クレストファクター（ピーク/RMS比）を計算"""
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        if rms > 0:
            return 20 * np.log10(peak / rms)
        return float('inf')
```

---

## 2. AIノイズ除去

### 2.1 主要なAIノイズ除去モデル

```python
# 1. Meta Denoiser (Demucs ベース)
import torchaudio

def demucs_denoise(audio_path: str) -> torch.Tensor:
    """Demucsベースのノイズ除去"""
    from denoiser import pretrained
    from denoiser.dsp import convert_audio

    model = pretrained.dns64()
    model.eval()

    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)

    with torch.no_grad():
        denoised = model(wav.unsqueeze(0))[0]

    return denoised

# 2. noisereduce ライブラリ
import noisereduce as nr
import soundfile as sf

def noisereduce_simple(audio_path: str) -> np.ndarray:
    """noisereduce による簡易ノイズ除去"""
    audio, sr = sf.read(audio_path)

    # 自動ノイズプロファイル推定
    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,     # 非定常ノイズ対応
        prop_decrease=0.8,    # ノイズ削減量（0-1）
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
    )

    return reduced

# 3. RNNoise (Mozilla)
def rnnoise_denoise(audio_path: str):
    """
    RNNoise: 超軽量リアルタイムノイズ除去
    - GRU ベースのリカレントモデル
    - CPU上でリアルタイム動作
    - 48kHz / 16bit モノラル入力
    """
    import rnnoise
    denoiser = rnnoise.RNNoise()

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # フレーム単位（10ms = 480サンプル @ 48kHz）で処理
    frame_size = 480
    denoised_frames = []
    for i in range(0, len(audio_data), frame_size * 2):
        frame = audio_data[i:i + frame_size * 2]
        denoised = denoiser.process_frame(frame)
        denoised_frames.append(denoised)

    return b"".join(denoised_frames)
```

### 2.2 ノイズ除去パイプライン

```
AIノイズ除去の処理フロー
==================================================

   入力音声 (ノイズ混入)
       │
       ▼
┌─────────────────┐
│ VAD (音声区間検出)│  音声/非音声区間の識別
└────────┬────────┘
         ▼
┌─────────────────┐
│ ノイズ分類       │  ノイズ種別の自動判定
│ - 定常ノイズ     │  (エアコン、ファン等)
│ - 非定常ノイズ   │  (キーボード、ドア等)
│ - 反響          │  (エコー、残響)
└────────┬────────┘
         ▼
┌─────────────────┐
│ モデル選択       │  ノイズ種別に最適な
│                 │  除去アルゴリズムを選択
└────────┬────────┘
         ▼
┌─────────────────┐
│ ノイズ除去実行   │  DNNベースの除去処理
└────────┬────────┘
         ▼
┌─────────────────┐
│ 後処理          │  アーティファクト抑制
│ - スムージング   │  急激な変化を緩和
│ - ゲーティング   │  残留ノイズの除去
└────────┬────────┘
         ▼
   出力音声 (クリーン)
==================================================
```

### 2.3 高度なノイズ除去: マルチステージパイプライン

実務でのノイズ除去は単一のモデルではなく、複数のステージを組み合わせることで最高品質を達成する。各ステージは特定のノイズタイプに対応し、順次処理される。

```python
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class NoiseAnalysisResult:
    """ノイズ分析結果"""
    has_hum: bool           # 電源ハム（50Hz/60Hz）
    has_broadband: bool     # ブロードバンドノイズ（ホワイトノイズ系）
    has_impulse: bool       # インパルスノイズ（クリック、ポップ）
    has_reverb: bool        # 過剰な残響
    has_sibilance: bool     # 過剰な歯擦音
    snr_estimate: float     # 推定SNR（dB）
    dominant_noise_freq: Optional[float]  # 主要ノイズ周波数

class MultiStageDenoiser:
    """マルチステージノイズ除去パイプライン"""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze_noise(self, audio: np.ndarray) -> NoiseAnalysisResult:
        """ノイズの種類と特性を分析"""
        # スペクトル分析
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sr)
        magnitude = np.abs(fft)

        # 電源ハム検出（50Hz/60Hz付近のピーク）
        hum_50 = self._check_peak(freqs, magnitude, 50, bandwidth=5)
        hum_60 = self._check_peak(freqs, magnitude, 60, bandwidth=5)
        has_hum = hum_50 or hum_60

        # ブロードバンドノイズ検出（高周波帯域のフロアレベル）
        high_freq_mask = freqs > 4000
        noise_floor = np.median(magnitude[high_freq_mask])
        signal_level = np.max(magnitude)
        has_broadband = noise_floor > signal_level * 0.01

        # SNR推定
        snr_estimate = 20 * np.log10(signal_level / (noise_floor + 1e-10))

        # インパルスノイズ検出（急激な振幅変化）
        diff = np.abs(np.diff(audio))
        impulse_threshold = np.mean(diff) + 5 * np.std(diff)
        has_impulse = np.any(diff > impulse_threshold)

        # 残響推定（エネルギー減衰特性）
        has_reverb = self._estimate_reverb(audio)

        # 歯擦音検出（4-10kHz帯域のエネルギー集中）
        sibilance_mask = (freqs > 4000) & (freqs < 10000)
        mid_mask = (freqs > 500) & (freqs < 4000)
        sibilance_ratio = np.mean(magnitude[sibilance_mask]) / (
            np.mean(magnitude[mid_mask]) + 1e-10
        )
        has_sibilance = sibilance_ratio > 1.5

        return NoiseAnalysisResult(
            has_hum=has_hum,
            has_broadband=has_broadband,
            has_impulse=has_impulse,
            has_reverb=has_reverb,
            has_sibilance=has_sibilance,
            snr_estimate=snr_estimate,
            dominant_noise_freq=50.0 if hum_50 else (60.0 if hum_60 else None),
        )

    def process(self, audio: np.ndarray) -> np.ndarray:
        """マルチステージノイズ除去を実行"""
        analysis = self.analyze_noise(audio)
        processed = audio.copy()

        # Stage 1: ハム除去（検出された場合のみ）
        if analysis.has_hum:
            processed = self._remove_hum(
                processed, analysis.dominant_noise_freq
            )
            print(f"Stage 1: ハム除去 ({analysis.dominant_noise_freq}Hz)")

        # Stage 2: インパルスノイズ除去
        if analysis.has_impulse:
            processed = self._remove_impulse(processed)
            print("Stage 2: インパルスノイズ除去")

        # Stage 3: ブロードバンドノイズ除去（AI）
        if analysis.has_broadband:
            strength = self._determine_strength(analysis.snr_estimate)
            processed = self._ai_denoise(processed, strength)
            print(f"Stage 3: ブロードバンドノイズ除去 (強度: {strength})")

        # Stage 4: デリバーブ（残響除去）
        if analysis.has_reverb:
            processed = self._dereverberate(processed)
            print("Stage 4: デリバーブ")

        # Stage 5: ディエッサー（歯擦音抑制）
        if analysis.has_sibilance:
            processed = self._deess(processed)
            print("Stage 5: ディエッサー")

        return processed

    def _check_peak(self, freqs, magnitude, target_freq, bandwidth=5):
        """特定周波数にピークがあるか検出"""
        mask = (freqs > target_freq - bandwidth) & (
            freqs < target_freq + bandwidth
        )
        if not mask.any():
            return False
        peak_level = np.max(magnitude[mask])
        surrounding_mask = (
            (freqs > target_freq - 50) & (freqs < target_freq + 50)
        )
        surrounding_mean = np.mean(magnitude[surrounding_mask])
        return peak_level > surrounding_mean * 3

    def _estimate_reverb(self, audio: np.ndarray) -> bool:
        """残響の有無を推定"""
        # 自己相関を計算して残響の特性を推定
        autocorr = np.correlate(audio[:4096], audio[:4096], mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]
        # 残響がある場合、自己相関の減衰が遅い
        decay_point = np.argmax(autocorr < 0.1)
        decay_time_ms = decay_point / self.sr * 1000
        return decay_time_ms > 100  # 100ms以上の残響

    def _determine_strength(self, snr: float) -> str:
        """SNRに基づいてノイズ除去強度を決定"""
        if snr > 30:
            return "light"
        elif snr > 15:
            return "medium"
        else:
            return "heavy"

    def _remove_hum(self, audio, freq):
        """ノッチフィルタによるハム除去"""
        from scipy.signal import iirnotch, filtfilt
        # 基本周波数と倍音（2次, 3次）を除去
        for harmonic in [1, 2, 3]:
            b, a = iirnotch(freq * harmonic, Q=30, fs=self.sr)
            audio = filtfilt(b, a, audio)
        return audio

    def _remove_impulse(self, audio):
        """中央値フィルタによるインパルスノイズ除去"""
        from scipy.signal import medfilt
        diff = np.abs(np.diff(audio))
        threshold = np.mean(diff) + 4 * np.std(diff)
        impulse_mask = np.concatenate([[False], diff > threshold])
        filtered = medfilt(audio, kernel_size=5)
        result = audio.copy()
        result[impulse_mask] = filtered[impulse_mask]
        return result

    def _ai_denoise(self, audio, strength="medium"):
        """AIベースのブロードバンドノイズ除去"""
        import noisereduce as nr
        prop_map = {"light": 0.5, "medium": 0.7, "heavy": 0.85}
        return nr.reduce_noise(
            y=audio, sr=self.sr,
            stationary=False,
            prop_decrease=prop_map[strength],
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
        )

    def _dereverberate(self, audio):
        """WPE（Weighted Prediction Error）ベースの残響除去"""
        # 簡易的な残響除去（スペクトラルサブトラクション応用）
        stft = np.fft.rfft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        # 残響テールの推定と除去
        reverb_estimate = np.convolve(magnitude, np.ones(10) / 10, mode='same')
        clean_magnitude = np.maximum(magnitude - 0.3 * reverb_estimate, 0)
        clean_stft = clean_magnitude * np.exp(1j * phase)
        return np.fft.irfft(clean_stft, n=len(audio))

    def _deess(self, audio):
        """周波数帯域ベースのディエッサー"""
        from scipy.signal import butter, filtfilt
        # 4-10kHz帯域のエネルギーを検出して抑制
        b, a = butter(4, [4000, 10000], btype='band', fs=self.sr)
        sibilance_band = filtfilt(b, a, audio)
        envelope = np.abs(sibilance_band)
        # スムージング
        from scipy.ndimage import uniform_filter1d
        envelope = uniform_filter1d(envelope, size=int(self.sr * 0.01))
        # 閾値以上の歯擦音を抑制
        threshold = np.percentile(envelope, 90)
        gain = np.where(envelope > threshold, threshold / (envelope + 1e-10), 1.0)
        gain = np.clip(gain, 0.3, 1.0)
        audio_deessed = audio - sibilance_band + sibilance_band * gain
        return audio_deessed
```

### 2.4 ノイズ除去品質の客観評価

```python
import numpy as np
from typing import Dict

class DenoiseQualityMetrics:
    """ノイズ除去の品質を客観的に評価するメトリクス集"""

    @staticmethod
    def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
        """
        SNR（Signal-to-Noise Ratio）
        - クリーン信号とノイズの比率
        - 高いほどノイズが少ない
        """
        noise = noisy - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        return 10 * np.log10(signal_power / (noise_power + 1e-10))

    @staticmethod
    def compute_pesq(clean: np.ndarray, denoised: np.ndarray,
                     sr: int = 16000) -> float:
        """
        PESQ（Perceptual Evaluation of Speech Quality）
        - ITU-T P.862 規格
        - 音声品質の知覚的評価（-0.5 ~ 4.5）
        - 高いほど良い
        """
        from pesq import pesq
        return pesq(sr, clean, denoised, 'wb')  # wb=ワイドバンド

    @staticmethod
    def compute_stoi(clean: np.ndarray, denoised: np.ndarray,
                     sr: int = 16000) -> float:
        """
        STOI（Short-Time Objective Intelligibility）
        - 音声の了解度（明瞭度）を評価（0 ~ 1）
        - 高いほど明瞭
        """
        from pystoi import stoi
        return stoi(clean, denoised, sr, extended=True)

    @staticmethod
    def compute_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
        """
        SDR（Signal-to-Distortion Ratio）
        - 音源分離の標準的な評価指標
        - 高いほど歪みが少ない
        """
        noise = estimated - reference
        sdr = 10 * np.log10(
            np.sum(reference ** 2) / (np.sum(noise ** 2) + 1e-10)
        )
        return sdr

    @staticmethod
    def compute_sisdr(reference: np.ndarray, estimated: np.ndarray) -> float:
        """
        SI-SDR（Scale-Invariant SDR）
        - スケール不変のSDR（音量差に影響されない）
        """
        alpha = np.dot(estimated, reference) / (np.dot(reference, reference) + 1e-10)
        target = alpha * reference
        noise = estimated - target
        return 10 * np.log10(
            np.sum(target ** 2) / (np.sum(noise ** 2) + 1e-10)
        )

    def full_evaluation(self, clean: np.ndarray, noisy: np.ndarray,
                        denoised: np.ndarray, sr: int = 16000) -> Dict:
        """総合評価レポート"""
        return {
            "入力SNR (dB)": round(self.compute_snr(clean, noisy), 2),
            "出力SNR (dB)": round(self.compute_snr(clean, denoised), 2),
            "SNR改善 (dB)": round(
                self.compute_snr(clean, denoised) - self.compute_snr(clean, noisy), 2
            ),
            "PESQ": round(self.compute_pesq(clean, denoised, sr), 3),
            "STOI": round(self.compute_stoi(clean, denoised, sr), 4),
            "SDR (dB)": round(self.compute_sdr(clean, denoised), 2),
            "SI-SDR (dB)": round(self.compute_sisdr(clean, denoised), 2),
        }

# 使用例
"""
metrics = DenoiseQualityMetrics()
report = metrics.full_evaluation(clean_audio, noisy_audio, denoised_audio, sr=16000)
for metric, value in report.items():
    print(f"{metric}: {value}")

# 出力例:
# 入力SNR (dB): 5.23
# 出力SNR (dB): 22.17
# SNR改善 (dB): 16.94
# PESQ: 3.456
# STOI: 0.9234
# SDR (dB): 18.92
# SI-SDR (dB): 17.85
"""
```

### 2.5 リアルタイムノイズ除去の実装

```python
import numpy as np
import threading
import queue
from typing import Callable

class RealtimeDenoiser:
    """リアルタイムノイズ除去エンジン"""

    def __init__(self, model_type: str = "rnnoise", sr: int = 48000,
                 frame_ms: int = 10):
        self.sr = sr
        self.frame_ms = frame_ms
        self.frame_size = int(sr * frame_ms / 1000)
        self.model_type = model_type
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False

        # 遅延計測用
        self.total_frames = 0
        self.processing_time_sum = 0

    def start(self):
        """処理スレッドを開始"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """処理スレッドを停止"""
        self.running = False
        self.thread.join()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """1フレームを処理（同期版）"""
        import time
        start = time.perf_counter()

        if self.model_type == "rnnoise":
            denoised = self._rnnoise_frame(frame)
        elif self.model_type == "demucs":
            denoised = self._demucs_frame(frame)
        else:
            denoised = frame

        elapsed = time.perf_counter() - start
        self.total_frames += 1
        self.processing_time_sum += elapsed

        return denoised

    def _process_loop(self):
        """バックグラウンド処理ループ"""
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                denoised = self.process_frame(frame)
                self.output_queue.put(denoised)
            except queue.Empty:
                continue

    def _rnnoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """RNNoiseによるフレーム処理"""
        # RNNoiseは480サンプル（10ms @ 48kHz）単位で処理
        # 実際にはrnnoise Cライブラリのバインディングを使用
        return frame  # プレースホルダー

    def _demucs_frame(self, frame: np.ndarray) -> np.ndarray:
        """Demucs Denoiserによるフレーム処理"""
        # GPUを使用したリアルタイム推論
        return frame  # プレースホルダー

    def get_latency_stats(self) -> dict:
        """遅延統計を取得"""
        if self.total_frames == 0:
            return {"avg_ms": 0, "frame_ms": self.frame_ms}
        avg_ms = (self.processing_time_sum / self.total_frames) * 1000
        return {
            "avg_processing_ms": round(avg_ms, 2),
            "frame_duration_ms": self.frame_ms,
            "realtime_ratio": round(avg_ms / self.frame_ms, 3),
            "total_frames": self.total_frames,
            "is_realtime": avg_ms < self.frame_ms,
        }


class StreamingDenoisePipeline:
    """PyAudioを使ったストリーミングノイズ除去"""

    def __init__(self, model_type="rnnoise"):
        import pyaudio
        self.pa = pyaudio.PyAudio()
        self.denoiser = RealtimeDenoiser(model_type=model_type)

    def run(self, input_device=None, output_device=None):
        """リアルタイムノイズ除去を実行"""
        import pyaudio

        sr = self.denoiser.sr
        frame_size = self.denoiser.frame_size

        def callback(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # 正規化

            denoised = self.denoiser.process_frame(audio)

            out_data = (denoised * 32768.0).astype(np.int16).tobytes()
            return (out_data, pyaudio.paContinue)

        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sr,
            input=True,
            output=True,
            input_device_index=input_device,
            output_device_index=output_device,
            frames_per_buffer=frame_size,
            stream_callback=callback,
        )

        stream.start_stream()
        print("リアルタイムノイズ除去を開始しました")
        print("Ctrl+C で停止")

        try:
            while stream.is_active():
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            stats = self.denoiser.get_latency_stats()
            print(f"遅延統計: {stats}")
```

---

## 3. AI EQ と自動マスタリング

### 3.1 AI EQ の実装概念

```python
import numpy as np

class AIEqualizer:
    """AIベースの自動EQ"""

    # 標準的なEQバンド
    BANDS = {
        "Sub Bass":    (20, 60),
        "Bass":        (60, 250),
        "Low Mid":     (250, 500),
        "Mid":         (500, 2000),
        "Upper Mid":   (2000, 4000),
        "Presence":    (4000, 6000),
        "Brilliance":  (6000, 20000),
    }

    # ターゲットプロファイル（ジャンル別）
    TARGET_PROFILES = {
        "podcast": {
            "Sub Bass": -6, "Bass": -2, "Low Mid": 0,
            "Mid": +2, "Upper Mid": +3, "Presence": +1, "Brilliance": -2,
        },
        "pop_vocal": {
            "Sub Bass": -3, "Bass": 0, "Low Mid": -1,
            "Mid": +1, "Upper Mid": +2, "Presence": +3, "Brilliance": +1,
        },
        "broadcast": {
            "Sub Bass": -12, "Bass": -3, "Low Mid": 0,
            "Mid": +2, "Upper Mid": +1, "Presence": +2, "Brilliance": 0,
        },
    }

    def analyze_spectrum(self, audio: np.ndarray, sr: int) -> dict:
        """スペクトル解析"""
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        magnitude = np.abs(fft)

        band_levels = {}
        for name, (low, high) in self.BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                level = 20 * np.log10(np.mean(magnitude[mask]) + 1e-10)
                band_levels[name] = level
        return band_levels

    def compute_eq_curve(self, audio: np.ndarray, sr: int,
                         target: str = "podcast") -> dict:
        """AI EQ: 現在のスペクトルとターゲットの差分からEQカーブを計算"""
        current = self.analyze_spectrum(audio, sr)
        target_profile = self.TARGET_PROFILES[target]

        eq_adjustments = {}
        for band in self.BANDS:
            diff = target_profile[band] - (current[band] - np.mean(list(current.values())))
            # 過度な補正を制限（最大 +-12dB）
            eq_adjustments[band] = np.clip(diff, -12, 12)

        return eq_adjustments
```

### 3.2 パラメトリックEQの実装

```python
import numpy as np
from scipy.signal import sosfilt, sosfiltfilt
from scipy.signal import iirpeak, iirnotch

class ParametricEQ:
    """パラメトリックEQの実装"""

    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.bands = []

    def add_band(self, freq: float, gain_db: float, q: float = 1.0,
                 band_type: str = "peak"):
        """
        EQバンドを追加

        Parameters:
            freq: 中心周波数 (Hz)
            gain_db: ゲイン (dB)
            q: Q値（帯域幅の逆数。高いほど狭い）
            band_type: "peak", "low_shelf", "high_shelf", "notch"
        """
        self.bands.append({
            "freq": freq,
            "gain_db": gain_db,
            "q": q,
            "type": band_type,
        })

    def _design_peak_filter(self, freq, gain_db, q):
        """ピーキングEQフィルタの設計（RBJクックブック）"""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / self.sr
        alpha = np.sin(w0) / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        return np.array([b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0])

    def _design_low_shelf(self, freq, gain_db, q=0.707):
        """ローシェルフフィルタの設計"""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / self.sr
        alpha = np.sin(w0) / (2 * q)

        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        return np.array([b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0])

    def _design_high_shelf(self, freq, gain_db, q=0.707):
        """ハイシェルフフィルタの設計"""
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / self.sr
        alpha = np.sin(w0) / (2 * q)

        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        return np.array([b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0])

    def apply(self, audio: np.ndarray) -> np.ndarray:
        """全バンドのEQを適用"""
        result = audio.copy()
        for band in self.bands:
            if band["type"] == "peak":
                sos = self._design_peak_filter(
                    band["freq"], band["gain_db"], band["q"]
                )
            elif band["type"] == "low_shelf":
                sos = self._design_low_shelf(band["freq"], band["gain_db"])
            elif band["type"] == "high_shelf":
                sos = self._design_high_shelf(band["freq"], band["gain_db"])
            else:
                continue
            # SOS形式に変換して適用
            sos_2d = sos.reshape(1, 6)
            result = sosfiltfilt(sos_2d, result)
        return result


# 使用例: ポッドキャスト向けAI EQプリセット
def apply_podcast_eq(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """ポッドキャスト向けの自動EQ"""
    eq = ParametricEQ(sr=sr)

    # ローカット（マイク振動やエアコンノイズ除去）
    eq.add_band(80, -18, q=0.707, band_type="high_shelf")  # 実際はHPF

    # 近接効果の補正（ボーカルの低域ブースト抑制）
    eq.add_band(200, -3, q=1.0, band_type="peak")

    # 明瞭度向上（プレゼンス帯域）
    eq.add_band(3000, +3, q=1.5, band_type="peak")

    # 息づかい・歯擦音抑制
    eq.add_band(6000, -2, q=2.0, band_type="peak")

    # エアー（高域の空気感）
    eq.add_band(12000, +1, q=0.707, band_type="high_shelf")

    return eq.apply(audio)
```

### 3.3 自動マスタリング

```python
class AutoMastering:
    """AI自動マスタリング処理"""

    def __init__(self, target_lufs=-14.0, target_true_peak=-1.0):
        self.target_lufs = target_lufs        # 配信標準: -14 LUFS
        self.target_true_peak = target_true_peak

    def master(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """自動マスタリングチェーン"""
        # Step 1: DC オフセット除去
        audio = audio - np.mean(audio)

        # Step 2: AI EQ（スペクトルバランス補正）
        audio = self.apply_ai_eq(audio, sr)

        # Step 3: マルチバンドコンプレッション
        audio = self.multiband_compress(audio, sr)

        # Step 4: ステレオイメージ調整
        if audio.ndim == 2 and audio.shape[0] == 2:
            audio = self.stereo_enhance(audio)

        # Step 5: ラウドネス正規化（LUFS準拠）
        audio = self.loudness_normalize(audio, sr)

        # Step 6: True Peak リミッティング
        audio = self.true_peak_limit(audio, sr)

        return audio

    def loudness_normalize(self, audio, sr):
        """ITU-R BS.1770 準拠のラウドネス正規化"""
        current_lufs = self.measure_lufs(audio, sr)
        gain_db = self.target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear

    def multiband_compress(self, audio, sr):
        """マルチバンドコンプレッション"""
        bands = [
            (20, 200, {"threshold": -20, "ratio": 2.0}),   # 低域
            (200, 2000, {"threshold": -18, "ratio": 2.5}),  # 中域
            (2000, 20000, {"threshold": -22, "ratio": 3.0}), # 高域
        ]
        result = np.zeros_like(audio)
        for low, high, params in bands:
            band = bandpass_filter(audio, low, high, sr)
            compressed = compress(band, **params)
            result += compressed
        return result

    def measure_lufs(self, audio, sr):
        """LUFS測定（簡略版）"""
        # K-weight フィルタ適用
        # ゲート処理
        # RMS計算
        rms = np.sqrt(np.mean(audio ** 2))
        return 20 * np.log10(rms + 1e-10)
```

### 3.4 LUFS準拠の高精度ラウドネス正規化

```python
import numpy as np
from scipy.signal import sosfilt

class LUFSMeter:
    """
    ITU-R BS.1770-5 準拠のLUFS測定器

    LUFS (Loudness Units Full Scale) は人間の聴覚特性を反映した
    ラウドネス測定単位で、各配信プラットフォームの基準値として使用される。
    """

    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.block_size = int(0.4 * sr)  # 400ms ゲーティングブロック
        self.overlap = int(0.1 * sr)     # 75% オーバーラップ（100ms ステップ）

    def _k_weight_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        K-weightingフィルタ
        - Stage 1: シェルビングフィルタ（高域ブースト）
        - Stage 2: ハイパスフィルタ（低域カット）
        人間の聴覚感度を反映
        """
        # Stage 1: ハイシェルフフィルタ（+4dB @ 高域）
        f0 = 1681.974450955533
        G = 3.999843853973347
        Q = 0.7071752369554196

        K = np.tan(np.pi * f0 / self.sr)
        Vh = 10 ** (G / 20)
        Vb = Vh ** 0.4996667741545416

        a0 = 1.0 + K / Q + K * K
        b0 = (Vh + Vb * K / Q + K * K) / a0
        b1 = 2.0 * (K * K - Vh) / a0
        b2 = (Vh - Vb * K / Q + K * K) / a0
        a1 = 2.0 * (K * K - 1.0) / a0
        a2 = (1.0 - K / Q + K * K) / a0

        sos1 = np.array([[b0, b1, b2, 1.0, a1, a2]])

        # Stage 2: ハイパスフィルタ
        f0 = 38.13547087602444
        Q = 0.5003270373238773

        K = np.tan(np.pi * f0 / self.sr)
        a0 = 1.0 + K / Q + K * K
        b0 = 1.0 / a0
        b1 = -2.0 / a0
        b2 = 1.0 / a0
        a1 = 2.0 * (K * K - 1.0) / a0
        a2 = (1.0 - K / Q + K * K) / a0

        sos2 = np.array([[b0, b1, b2, 1.0, a1, a2]])

        filtered = sosfilt(sos1, audio)
        filtered = sosfilt(sos2, filtered)
        return filtered

    def measure_integrated(self, audio: np.ndarray) -> float:
        """
        統合ラウドネス（Integrated Loudness）の計測

        BS.1770のアルゴリズム:
        1. K-weightingフィルタ適用
        2. 400msブロックごとのラウドネス計算
        3. 絶対ゲート（-70 LUFS以上のブロックのみ）
        4. 相対ゲート（平均-10 LUFS以上のブロックのみ）
        5. ゲート通過ブロックの平均 = 統合ラウドネス
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)  # (channels, samples)

        n_channels = audio.shape[0]

        # チャンネル重み（サラウンド対応）
        channel_weights = {
            1: [1.0],
            2: [1.0, 1.0],
            6: [1.0, 1.0, 1.0, 0.0, 1.41, 1.41],  # 5.1ch
        }
        weights = channel_weights.get(n_channels, [1.0] * n_channels)

        # K-weightingフィルタ適用
        filtered = np.array([self._k_weight_filter(ch) for ch in audio])

        # ブロックごとのラウドネス計算
        step = self.block_size - self.overlap
        n_blocks = max(1, (filtered.shape[1] - self.block_size) // step + 1)

        block_loudness = []
        for i in range(n_blocks):
            start = i * step
            end = start + self.block_size
            if end > filtered.shape[1]:
                break

            # チャンネルごとの平均パワー
            power_sum = 0
            for ch in range(n_channels):
                block = filtered[ch, start:end]
                power_sum += weights[ch] * np.mean(block ** 2)

            loudness = -0.691 + 10 * np.log10(power_sum + 1e-10)
            block_loudness.append(loudness)

        block_loudness = np.array(block_loudness)

        # 絶対ゲート（-70 LUFS）
        abs_gate_mask = block_loudness > -70
        if not abs_gate_mask.any():
            return -70.0

        # 相対ゲート
        abs_gated_mean = np.mean(
            10 ** (block_loudness[abs_gate_mask] / 10)
        )
        relative_threshold = 10 * np.log10(abs_gated_mean + 1e-10) - 10

        rel_gate_mask = block_loudness > relative_threshold
        if not rel_gate_mask.any():
            return -70.0

        # 最終計算
        final_mean = np.mean(
            10 ** (block_loudness[rel_gate_mask] / 10)
        )
        integrated_loudness = -0.691 + 10 * np.log10(final_mean + 1e-10)

        return round(integrated_loudness, 1)

    def measure_momentary(self, audio: np.ndarray) -> np.ndarray:
        """
        モーメンタリーラウドネス（400msウィンドウ）
        リアルタイムメーター表示用
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        filtered = np.array([self._k_weight_filter(ch) for ch in audio])
        step = int(0.1 * self.sr)  # 100msステップ
        n_steps = (filtered.shape[1] - self.block_size) // step + 1

        momentary = []
        for i in range(n_steps):
            start = i * step
            end = start + self.block_size
            power = np.mean(np.sum(filtered[:, start:end] ** 2, axis=0))
            lufs = -0.691 + 10 * np.log10(power + 1e-10)
            momentary.append(lufs)

        return np.array(momentary)


# プラットフォーム別LUFS要件
PLATFORM_LOUDNESS_SPECS = {
    "Spotify": {
        "target_lufs": -14.0,
        "true_peak_limit": -1.0,
        "normalization": "トラック別またはアルバム別",
        "notes": "ラウドネス超過時は自動で下げられる",
    },
    "Apple Music": {
        "target_lufs": -16.0,
        "true_peak_limit": -1.0,
        "normalization": "Sound Check有効時のみ",
        "notes": "Sound Check OFFの場合は正規化されない",
    },
    "YouTube": {
        "target_lufs": -14.0,
        "true_peak_limit": -1.0,
        "normalization": "常に適用",
        "notes": "-14 LUFS超過は下げられるが、以下は上げられない",
    },
    "Amazon Music": {
        "target_lufs": -14.0,
        "true_peak_limit": -2.0,
        "normalization": "自動",
        "notes": "True Peak制限がやや厳しい",
    },
    "Podcast (一般)": {
        "target_lufs": -16.0,
        "true_peak_limit": -1.0,
        "normalization": "推奨値（強制ではない）",
        "notes": "モノラル -19 LUFS推奨の場合も",
    },
    "放送 (EBU R128)": {
        "target_lufs": -23.0,
        "true_peak_limit": -1.0,
        "normalization": "厳格に準拠",
        "notes": "欧州放送基準。許容誤差 ±1 LU",
    },
    "放送 (ATSC A/85)": {
        "target_lufs": -24.0,
        "true_peak_limit": -2.0,
        "normalization": "厳格に準拠",
        "notes": "米国放送基準",
    },
}
```

### 3.5 コンプレッサーの詳細実装

```python
import numpy as np

class DynamicCompressor:
    """
    ダイナミックレンジコンプレッサーの実装

    コンプレッサーはオーディオ信号のダイナミックレンジ（最大音量と最小音量の差）
    を圧縮し、音量差を均一化するエフェクトである。
    """

    def __init__(self, sr: int = 44100, threshold_db: float = -20,
                 ratio: float = 4.0, attack_ms: float = 5.0,
                 release_ms: float = 50.0, knee_db: float = 6.0,
                 makeup_gain_db: float = 0.0):
        """
        Parameters:
            threshold_db: 圧縮開始レベル（dB）
            ratio: 圧縮比（4:1 = 閾値を超えた4dBに対し1dB出力）
            attack_ms: アタックタイム（圧縮開始までの時間）
            release_ms: リリースタイム（圧縮解除までの時間）
            knee_db: ニー幅（ソフトニーの範囲）
            makeup_gain_db: メイクアップゲイン（圧縮後の音量補正）
        """
        self.sr = sr
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_coeff = np.exp(-1 / (attack_ms * sr / 1000))
        self.release_coeff = np.exp(-1 / (release_ms * sr / 1000))
        self.knee_db = knee_db
        self.makeup_gain = 10 ** (makeup_gain_db / 20)

    def _compute_gain(self, level_db: float) -> float:
        """ゲインリダクションの計算（ソフトニー対応）"""
        if self.knee_db <= 0:
            # ハードニー
            if level_db <= self.threshold_db:
                return 0.0
            else:
                return -(level_db - self.threshold_db) * (1 - 1 / self.ratio)
        else:
            # ソフトニー
            half_knee = self.knee_db / 2
            if level_db < self.threshold_db - half_knee:
                return 0.0
            elif level_db > self.threshold_db + half_knee:
                return -(level_db - self.threshold_db) * (1 - 1 / self.ratio)
            else:
                # ニー内の滑らかな遷移
                x = level_db - self.threshold_db + half_knee
                return -(x ** 2) / (2 * self.knee_db) * (1 - 1 / self.ratio)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """コンプレッション処理"""
        output = np.zeros_like(audio)
        envelope = 0.0

        for i in range(len(audio)):
            # エンベロープ追従
            level = np.abs(audio[i])
            if level > envelope:
                envelope = self.attack_coeff * envelope + (1 - self.attack_coeff) * level
            else:
                envelope = self.release_coeff * envelope + (1 - self.release_coeff) * level

            # dBに変換
            level_db = 20 * np.log10(envelope + 1e-10)

            # ゲインリダクション計算
            gain_reduction_db = self._compute_gain(level_db)
            gain = 10 ** (gain_reduction_db / 20)

            # ゲイン適用 + メイクアップゲイン
            output[i] = audio[i] * gain * self.makeup_gain

        return output

    def auto_makeup_gain(self, audio: np.ndarray) -> float:
        """自動メイクアップゲインの計算"""
        # 圧縮前後のRMSレベル差から自動計算
        original_rms = np.sqrt(np.mean(audio ** 2))
        compressed = self.process(audio)
        compressed_rms = np.sqrt(np.mean(compressed ** 2))
        if compressed_rms > 0:
            gain_db = 20 * np.log10(original_rms / compressed_rms)
            return gain_db
        return 0.0


class MultibandCompressor:
    """マルチバンドコンプレッサー"""

    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.bands = [
            {"name": "Low", "range": (20, 250),
             "threshold": -20, "ratio": 2.0, "attack": 10, "release": 100},
            {"name": "Low-Mid", "range": (250, 1000),
             "threshold": -18, "ratio": 2.5, "attack": 8, "release": 80},
            {"name": "Mid", "range": (1000, 4000),
             "threshold": -22, "ratio": 3.0, "attack": 5, "release": 50},
            {"name": "High", "range": (4000, 20000),
             "threshold": -24, "ratio": 3.5, "attack": 3, "release": 40},
        ]

    def _bandpass(self, audio, low, high):
        """バンドパスフィルタ"""
        from scipy.signal import butter, sosfiltfilt
        nyq = self.sr / 2
        low_norm = max(low / nyq, 0.001)
        high_norm = min(high / nyq, 0.999)
        sos = butter(4, [low_norm, high_norm], btype='band', output='sos')
        return sosfiltfilt(sos, audio)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """マルチバンドコンプレッション"""
        result = np.zeros_like(audio)

        for band in self.bands:
            low, high = band["range"]
            band_audio = self._bandpass(audio, low, high)

            comp = DynamicCompressor(
                sr=self.sr,
                threshold_db=band["threshold"],
                ratio=band["ratio"],
                attack_ms=band["attack"],
                release_ms=band["release"],
                knee_db=6.0,
            )

            compressed = comp.process(band_audio)
            result += compressed

        return result
```

---

## 4. 比較表

### 4.1 ノイズ除去ツール比較

| 項目 | RNNoise | noisereduce | Demucs Denoiser | Adobe Podcast | Krisp |
|------|---------|-------------|-----------------|--------------|-------|
| 種別 | OSS | OSS | OSS | SaaS | SaaS |
| リアルタイム | 対応 | 非対応 | 非対応 | 非対応 | 対応 |
| 品質 | 良い | 良い | 非常に良い | 最高 | 非常に良い |
| 計算コスト | 極めて低い | 低い | 高い(GPU) | クラウド | 低い |
| 非定常ノイズ | 対応 | 対応 | 対応 | 対応 | 対応 |
| 音声劣化 | 少ない | 中程度 | 少ない | 最小 | 少ない |
| 対応入力 | 48kHz mono | 柔軟 | 柔軟 | 柔軟 | 柔軟 |

### 4.2 マスタリングサービス比較

| 項目 | LANDR | eMastered | iZotope Ozone | CloudBounce |
|------|-------|-----------|--------------|-------------|
| AI自動化 | 完全自動 | 完全自動 | 半自動 | 完全自動 |
| カスタマイズ | 3プリセット | スライダー | フル制御 | 限定的 |
| 品質 | 高い | 高い | 最高 | 中〜高 |
| 価格 | $4.99/曲〜 | $3.99/曲〜 | $249買切 | $2.99/曲〜 |
| LUFS準拠 | 対応 | 対応 | 対応 | 対応 |
| プロ利用 | やや不向き | やや不向き | 業界標準 | やや不向き |

### 4.3 AIエフェクトプラグイン比較

| 項目 | iZotope RX | Waves Clarity | Sonnox | Accusonus ERA |
|------|-----------|--------------|--------|-------------|
| ノイズ除去 | 最高品質 | 非常に良い | 良い | 良い |
| 自動EQ | Neutron連携 | 非対応 | 非対応 | 非対応 |
| デリバーブ | 最高品質 | 良い | 良い | 中程度 |
| ディエッサー | 高品質 | 良い | 最高品質 | 良い |
| バッチ処理 | 対応 | 非対応 | 非対応 | 対応 |
| リアルタイム | 一部対応 | 対応 | 対応 | 対応 |
| DAW統合 | VST/AU/AAX | VST/AU/AAX | VST/AU/AAX | VST/AU/AAX |
| 価格帯 | $399-1199 | $149-249 | $299+ | $99-199 |
| AI深度 | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★☆☆ |

---

## 5. アンチパターン

### 5.1 アンチパターン: エフェクトの順序無視

```python
# BAD: エフェクトの順序が不適切
def bad_effects_chain(audio, sr):
    audio = apply_reverb(audio)      # リバーブ後のノイズ除去は困難
    audio = apply_compression(audio) # 圧縮後のEQは予測困難
    audio = denoise(audio)           # リバーブのテールもノイズと判定
    audio = equalize(audio)          # 既に歪んだスペクトルをEQ
    return audio

# GOOD: 正しいエフェクト順序
def good_effects_chain(audio, sr):
    # 1. まずノイズ除去（クリーンな信号を確保）
    audio = denoise(audio)
    # 2. EQ（周波数バランス調整）
    audio = equalize(audio, target="podcast")
    # 3. コンプレッション（ダイナミックレンジ制御）
    audio = compress(audio, threshold=-20, ratio=3)
    # 4. リバーブ等の空間系（クリーンな信号に適用）
    audio = apply_reverb(audio, room_size=0.3)
    # 5. リミッター（最終的なピーク制御）
    audio = limit(audio, ceiling=-1.0)
    return audio
```

### 5.2 アンチパターン: 過度なノイズ除去

```python
# BAD: ノイズ除去を最大に設定
def bad_denoise(audio, sr):
    return nr.reduce_noise(
        y=audio, sr=sr,
        prop_decrease=1.0,  # 100%除去 → 音声も劣化
        n_fft=4096,
    )

# GOOD: 適切な設定と段階的処理
def good_denoise(audio, sr, strength="medium"):
    """段階的なノイズ除去"""
    settings = {
        "light":  {"prop_decrease": 0.5, "note": "軽微なノイズ向け"},
        "medium": {"prop_decrease": 0.7, "note": "標準的なノイズ向け"},
        "heavy":  {"prop_decrease": 0.85, "note": "強いノイズ向け（音声劣化注意）"},
    }
    s = settings[strength]

    result = nr.reduce_noise(
        y=audio, sr=sr,
        stationary=False,
        prop_decrease=s["prop_decrease"],
        freq_mask_smooth_hz=500,    # スムージングでアーティファクト軽減
        time_mask_smooth_ms=50,
    )

    # 音声品質チェック
    snr = compute_snr(audio, result)
    if snr < 5:
        print("警告: 音声品質が大幅に劣化しています。strengthを下げてください。")

    return result
```

### 5.3 アンチパターン: LUFS準拠なしのラウドネス調整

```python
# BAD: ピーク正規化だけでラウドネスを調整
def bad_loudness(audio):
    peak = np.max(np.abs(audio))
    return audio / peak * 0.9  # ピーク -0.9dBFS
    # 問題: LUFS は音声の内容（ダイナミクス）に依存
    #        同じピークでもポッドキャストと音楽ではLUFSが大幅に異なる

# GOOD: LUFS測定 → 正規化 → True Peak リミッティング
def good_loudness(audio, sr, target_lufs=-14.0, true_peak=-1.0):
    """LUFS準拠のラウドネス正規化"""
    meter = LUFSMeter(sr=sr)

    # 現在のLUFSを測定
    current_lufs = meter.measure_integrated(audio)
    print(f"現在のLUFS: {current_lufs}")

    # LUFS差分から必要なゲインを計算
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    audio = audio * gain_linear

    # True Peakリミッティング
    true_peak_linear = 10 ** (true_peak / 20)
    oversampled = np.interp(
        np.linspace(0, len(audio) - 1, len(audio) * 4),
        np.arange(len(audio)),
        audio,
    )
    peak = np.max(np.abs(oversampled))
    if peak > true_peak_linear:
        audio = audio * (true_peak_linear / peak)

    final_lufs = meter.measure_integrated(audio)
    print(f"正規化後のLUFS: {final_lufs}")

    return audio
```

### 5.4 アンチパターン: サンプルレート不整合

```python
# BAD: サンプルレートを考慮しないエフェクト適用
def bad_sample_rate(audio_path):
    audio, sr = sf.read(audio_path)  # sr=48000かもしれない
    # RNNoiseは48kHz前提
    denoised = rnnoise(audio)
    # noisereduceは内部でsrを使う
    eq_applied = apply_eq(denoised)  # EQの周波数がずれる
    return eq_applied

# GOOD: サンプルレートの統一管理
def good_sample_rate(audio_path, target_sr=44100):
    """サンプルレートを統一してからエフェクト適用"""
    import librosa

    audio, sr = sf.read(audio_path)
    print(f"元のサンプルレート: {sr}Hz")

    # エフェクトチェーンの要求サンプルレートに統一
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        print(f"リサンプリング: {sr}Hz → {target_sr}Hz")

    # 全エフェクトで同じsrを使用
    audio = denoise(audio, sr=target_sr)
    audio = equalize(audio, sr=target_sr)
    audio = compress(audio, sr=target_sr)
    audio = normalize_lufs(audio, sr=target_sr)

    return audio, target_sr
```

---

## 6. 実践的なユースケース

### 6.1 ポッドキャスト完全処理パイプライン

```python
import numpy as np
import soundfile as sf
from pathlib import Path

class PodcastProcessor:
    """ポッドキャスト音声の完全処理パイプライン"""

    def __init__(self, sr: int = 44100):
        self.sr = sr

    def process_episode(self, input_path: str, output_path: str,
                        speakers: int = 2):
        """ポッドキャストエピソードの完全処理"""
        audio, sr = sf.read(input_path)

        # サンプルレート統一
        if sr != self.sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        # Step 1: ノイズ分析と除去
        print("Step 1: ノイズ除去")
        audio = self._denoise(audio)

        # Step 2: ハイパスフィルタ（80Hz以下カット）
        print("Step 2: ハイパスフィルタ")
        audio = self._highpass(audio, cutoff=80)

        # Step 3: EQ
        print("Step 3: EQ補正")
        audio = self._podcast_eq(audio)

        # Step 4: コンプレッション
        print("Step 4: コンプレッション")
        comp = DynamicCompressor(
            sr=self.sr, threshold_db=-18, ratio=3.0,
            attack_ms=5, release_ms=50, knee_db=6.0,
        )
        audio = comp.process(audio)

        # Step 5: ラウドネス正規化（-16 LUFS）
        print("Step 5: ラウドネス正規化")
        meter = LUFSMeter(sr=self.sr)
        current_lufs = meter.measure_integrated(audio)
        gain_db = -16.0 - current_lufs
        audio = audio * (10 ** (gain_db / 20))

        # Step 6: True Peak リミッティング
        print("Step 6: リミッティング")
        audio = self._true_peak_limit(audio, ceiling_db=-1.0)

        # 保存
        sf.write(output_path, audio, self.sr, subtype='PCM_24')
        print(f"完了: {output_path}")

        # 品質レポート
        final_lufs = meter.measure_integrated(audio)
        peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        print(f"最終LUFS: {final_lufs:.1f}")
        print(f"ピークレベル: {peak_db:.1f} dBFS")

    def _denoise(self, audio):
        """ポッドキャスト向けノイズ除去"""
        import noisereduce as nr
        return nr.reduce_noise(
            y=audio, sr=self.sr,
            stationary=False,
            prop_decrease=0.65,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
        )

    def _highpass(self, audio, cutoff=80):
        """ハイパスフィルタ"""
        from scipy.signal import butter, sosfiltfilt
        nyq = self.sr / 2
        sos = butter(4, cutoff / nyq, btype='high', output='sos')
        return sosfiltfilt(sos, audio)

    def _podcast_eq(self, audio):
        """ポッドキャスト向けEQ"""
        eq = ParametricEQ(sr=self.sr)
        eq.add_band(200, -2, q=1.0, band_type="peak")    # 近接効果補正
        eq.add_band(3000, +3, q=1.5, band_type="peak")   # プレゼンスブースト
        eq.add_band(6000, -1.5, q=2.0, band_type="peak") # 歯擦音抑制
        return eq.apply(audio)

    def _true_peak_limit(self, audio, ceiling_db=-1.0):
        """True Peakリミッティング"""
        ceiling = 10 ** (ceiling_db / 20)
        peak = np.max(np.abs(audio))
        if peak > ceiling:
            audio = audio * (ceiling / peak)
        return audio
```

### 6.2 音楽マスタリング自動化

```python
class MusicMasteringPipeline:
    """音楽マスタリングの自動化パイプライン"""

    def __init__(self, sr: int = 44100, target_platform: str = "Spotify"):
        self.sr = sr
        self.target_platform = target_platform
        self.specs = PLATFORM_LOUDNESS_SPECS.get(target_platform, {})

    def master(self, input_path: str, output_path: str,
               genre: str = "pop") -> dict:
        """マスタリング処理"""
        audio, sr = sf.read(input_path, always_2d=True)
        audio = audio.T  # (channels, samples)

        if sr != self.sr:
            import librosa
            audio = np.array([
                librosa.resample(ch, orig_sr=sr, target_sr=self.sr)
                for ch in audio
            ])

        # 分析レポート
        pre_analysis = self._analyze(audio)
        print(f"分析結果: {pre_analysis}")

        # DC除去
        audio = audio - np.mean(audio, axis=1, keepdims=True)

        # ジャンル別EQ
        genre_eq = self._get_genre_eq(genre)
        for ch in range(audio.shape[0]):
            audio[ch] = genre_eq.apply(audio[ch])

        # マルチバンドコンプレッション
        mb_comp = MultibandCompressor(sr=self.sr)
        for ch in range(audio.shape[0]):
            audio[ch] = mb_comp.process(audio[ch])

        # ステレオイメージング（ステレオの場合）
        if audio.shape[0] == 2:
            audio = self._stereo_enhance(audio)

        # ラウドネス正規化
        target_lufs = self.specs.get("target_lufs", -14.0)
        meter = LUFSMeter(sr=self.sr)
        current = meter.measure_integrated(audio)
        gain = 10 ** ((target_lufs - current) / 20)
        audio = audio * gain

        # True Peak リミッティング
        tp_limit = self.specs.get("true_peak_limit", -1.0)
        audio = self._true_peak_limit_stereo(audio, tp_limit)

        # 保存
        sf.write(output_path, audio.T, self.sr, subtype='PCM_24')

        # 最終分析
        post_analysis = self._analyze(audio)
        return {
            "before": pre_analysis,
            "after": post_analysis,
            "target_platform": self.target_platform,
            "output": output_path,
        }

    def _analyze(self, audio):
        """音声の分析"""
        meter = LUFSMeter(sr=self.sr)
        return {
            "lufs": meter.measure_integrated(audio),
            "peak_db": round(20 * np.log10(np.max(np.abs(audio)) + 1e-10), 1),
            "dynamic_range_db": round(
                20 * np.log10(np.max(np.abs(audio)) + 1e-10) -
                20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10), 1
            ),
        }

    def _get_genre_eq(self, genre):
        """ジャンル別EQプリセット"""
        eq = ParametricEQ(sr=self.sr)
        if genre == "pop":
            eq.add_band(60, +2, q=1.0, band_type="peak")
            eq.add_band(3000, +2, q=1.5, band_type="peak")
            eq.add_band(10000, +1, q=0.707, band_type="high_shelf")
        elif genre == "rock":
            eq.add_band(100, +3, q=0.8, band_type="peak")
            eq.add_band(2500, +2, q=1.2, band_type="peak")
            eq.add_band(8000, +1.5, q=0.707, band_type="high_shelf")
        elif genre == "classical":
            eq.add_band(250, -1, q=1.0, band_type="peak")
            eq.add_band(5000, +1, q=0.707, band_type="high_shelf")
        elif genre == "hiphop":
            eq.add_band(50, +4, q=1.0, band_type="peak")
            eq.add_band(150, +2, q=1.0, band_type="peak")
            eq.add_band(4000, +2, q=1.5, band_type="peak")
        return eq

    def _stereo_enhance(self, audio):
        """ステレオイメージの強調"""
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2
        # サイド成分を軽くブースト（ステレオ感の強調）
        side = side * 1.2
        audio[0] = mid + side
        audio[1] = mid - side
        return audio

    def _true_peak_limit_stereo(self, audio, ceiling_db):
        """ステレオTrue Peakリミッティング"""
        ceiling = 10 ** (ceiling_db / 20)
        for ch in range(audio.shape[0]):
            peak = np.max(np.abs(audio[ch]))
            if peak > ceiling:
                audio[ch] = audio[ch] * (ceiling / peak)
        return audio
```

---

## 7. FAQ

### Q1: ポッドキャスト録音のノイズ除去で推奨される設定は？

ポッドキャストの場合、(1) まず環境ノイズをnoisereduceで軽〜中程度（prop_decrease=0.6-0.7）除去、(2) ハイパスフィルタで80Hz以下をカット（エアコン音、振動）、(3) ディエッサーで歯擦音を制御。重要なのは過度に除去しないこと。少量のノイズが残る方が、アーティファクトだらけのクリーン音声より聞きやすいです。Adobe Podcastの「Enhance Speech」機能は最も簡単かつ高品質な選択肢です。

### Q2: LUFS（ラウドネスユニット）とは何ですか？なぜ重要ですか？

LUFSはITU-R BS.1770で規定されたラウドネス測定単位で、人間の聴覚特性を考慮した音量の指標です。各配信プラットフォームが要求するLUFS値が異なります。Spotify: -14 LUFS、YouTube: -14 LUFS、Apple Music: -16 LUFS、ポッドキャスト: -16 to -14 LUFS。この値に合わせないと、プラットフォーム側で自動的にラウドネス調整が行われ、意図しない音質変化が起きます。

### Q3: AIマスタリングはプロのマスタリングエンジニアを置き換えられますか？

2025年時点では「完全な置き換え」には至っていません。AIマスタリングは、(1) 技術的な正確さ（LUFS準拠、True Peakリミッティング）、(2) 一貫した品質、(3) 低コスト・高速という利点があります。一方で、(1) 楽曲の芸術的意図の理解、(2) マスタリング連作での統一感、(3) 問題の発見と創造的解決、はまだ人間のエンジニアが優れています。Demo/ポッドキャスト/YouTubeコンテンツにはAIマスタリングが十分で、商業リリースにはプロのエンジニアを推奨します。

### Q4: リアルタイムでのAIエフェクト処理に必要なハードウェア要件は？

リアルタイム処理の要件は使用するモデルにより大きく異なります。RNNoiseはCPUのみで動作し、Raspberry Piでも処理可能（遅延約5ms）です。Demucs Denoiserの場合はGPU（NVIDIA RTX 3060以上）が推奨され、遅延は20-50ms程度です。重要なのは「バッファサイズとレイテンシのトレードオフ」で、バッファを小さくすると遅延は減るが処理負荷が増えます。一般的に、音声通話では40ms以下、音楽演奏では10ms以下が快適なレイテンシの目安です。

### Q5: EQとコンプレッサーのパラメータ設定の経験則は？

ボーカル収録の場合、EQではまず80Hz以下のハイパスフィルタでランブルノイズを除去し、200-300Hzの「もこもこ感」を2-3dBカット、2-4kHzの「プレゼンス」を2-3dBブーストします。コンプレッサーではスレッショルド-18〜-24dB、レシオ3:1〜4:1、アタック5-10ms、リリース40-80msが出発点です。ただしこれらはあくまで経験則であり、実際の素材や環境に応じて耳で調整することが最も重要です。

### Q6: ディザリングとは何ですか？いつ必要ですか？

ディザリングはビット深度を下げる際（例: 24bit→16bit）に発生する量子化歪みを軽減するために、意図的に微小なノイズを付加する処理です。マスタリングの最終段階でCD用16bit WAVやMP3に変換する際に必要になります。ディザリングなしにビット深度を下げると、低レベル信号で量子化ノイズが知覚可能な歪みとなります。TPDF（Triangular Probability Density Function）ディザが最も一般的で、iZotope OzoneやFabFilter Pro-Lなどのリミッタープラグインに組み込まれています。

---

## まとめ

| 項目 | 要点 |
|------|------|
| エフェクト順序 | ノイズ除去 → EQ → コンプ → 空間系 → リミッター |
| AIノイズ除去 | 非定常ノイズ対応、プロファイル不要が利点 |
| AI EQ | ターゲットプロファイルとの差分で自動調整 |
| 自動マスタリング | LUFS正規化が核。配信先の要求値に準拠 |
| 過度な処理の回避 | ノイズ除去は70-80%が目安。100%は音声劣化 |
| プロ vs AI | 配信コンテンツはAI十分。商業リリースはプロ推奨 |
| 品質評価 | SNR/PESQ/STOI/SDRで客観評価。聴感評価も重要 |
| リアルタイム | RNNoiseはCPUで可能。GPU系は20-50ms遅延 |

## 次に読むべきガイド

- [03-midi-ai.md](./03-midi-ai.md) — MIDI×AI（自動作曲、コード進行生成）
- [01-stem-separation.md](./01-stem-separation.md) — ステム分離との組み合わせ
- [../03-development/01-audio-processing.md](../03-development/01-audio-processing.md) — librosa/torchaudio実装

## 参考文献

1. Défossez, A., et al. (2020). "Real Time Speech Enhancement in the Waveform Domain" — Meta Denoiser論文。波形領域でのリアルタイムノイズ除去
2. Valin, J.M., et al. (2018). "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement" — RNNoise論文。超軽量リアルタイムノイズ除去
3. ITU-R BS.1770-5 (2023). "Algorithms to measure audio programme loudness and true-peak audio level" — ラウドネス測定の国際規格
4. EBU R128 (2020). "Loudness normalisation and permitted maximum level of audio signals" — 欧州放送連合のラウドネス規格
5. Rix, A.W., et al. (2001). "Perceptual Evaluation of Speech Quality (PESQ)" — ITU-T P.862 音声品質評価規格
6. Taal, C.H., et al. (2011). "An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech" — STOI音声明瞭度評価
7. Smith, J.O. (2007). "Introduction to Digital Filters with Audio Applications" — デジタルフィルタ設計の教科書
8. Zölzer, U. (2011). "DAFX: Digital Audio Effects" — デジタルオーディオエフェクトの包括的教科書
