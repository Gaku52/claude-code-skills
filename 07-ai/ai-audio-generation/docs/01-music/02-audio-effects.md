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

### 3.2 自動マスタリング

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

---

## 6. FAQ

### Q1: ポッドキャスト録音のノイズ除去で推奨される設定は？

ポッドキャストの場合、(1) まず環境ノイズをnoisereduceで軽〜中程度（prop_decrease=0.6-0.7）除去、(2) ハイパスフィルタで80Hz以下をカット（エアコン音、振動）、(3) ディエッサーで歯擦音を制御。重要なのは過度に除去しないこと。少量のノイズが残る方が、アーティファクトだらけのクリーン音声より聞きやすいです。Adobe Podcastの「Enhance Speech」機能は最も簡単かつ高品質な選択肢です。

### Q2: LUFS（ラウドネスユニット）とは何ですか？なぜ重要ですか？

LUFSはITU-R BS.1770で規定されたラウドネス測定単位で、人間の聴覚特性を考慮した音量の指標です。各配信プラットフォームが要求するLUFS値が異なります。Spotify: -14 LUFS、YouTube: -14 LUFS、Apple Music: -16 LUFS、ポッドキャスト: -16 to -14 LUFS。この値に合わせないと、プラットフォーム側で自動的にラウドネス調整が行われ、意図しない音質変化が起きます。

### Q3: AIマスタリングはプロのマスタリングエンジニアを置き換えられますか？

2025年時点では「完全な置き換え」には至っていません。AIマスタリングは、(1) 技術的な正確さ（LUFS準拠、True Peakリミッティング）、(2) 一貫した品質、(3) 低コスト・高速という利点があります。一方で、(1) 楽曲の芸術的意図の理解、(2) マスタリング連作での統一感、(3) 問題の発見と創造的解決、はまだ人間のエンジニアが優れています。Demo/ポッドキャスト/YouTubeコンテンツにはAIマスタリングが十分で、商業リリースにはプロのエンジニアを推奨します。

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

## 次に読むべきガイド

- [03-midi-ai.md](./03-midi-ai.md) — MIDI×AI（自動作曲、コード進行生成）
- [01-stem-separation.md](./01-stem-separation.md) — ステム分離との組み合わせ
- [../03-development/01-audio-processing.md](../03-development/01-audio-processing.md) — librosa/torchaudio実装

## 参考文献

1. Défossez, A., et al. (2020). "Real Time Speech Enhancement in the Waveform Domain" — Meta Denoiser論文。波形領域でのリアルタイムノイズ除去
2. Valin, J.M., et al. (2018). "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement" — RNNoise論文。超軽量リアルタイムノイズ除去
3. ITU-R BS.1770-5 (2023). "Algorithms to measure audio programme loudness and true-peak audio level" — ラウドネス測定の国際規格
