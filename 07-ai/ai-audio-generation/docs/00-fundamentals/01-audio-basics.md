# 音声基礎 — サンプリング、周波数、フーリエ変換

> デジタル音声の物理的・数学的基礎を理解し、音声AIに必要な信号処理の土台を固める

## この章で学ぶこと

1. 音の物理的性質とデジタル化の原理（サンプリング定理、量子化）
2. 周波数解析の基礎（フーリエ変換、スペクトログラム、メル尺度）
3. 音声特徴量の抽出手法（MFCC、メルスペクトログラム）と実装

---

## 1. 音の物理的性質

### 1.1 音波の基本パラメータ

```
音波の基本要素
==================================================

  振幅(Amplitude)
  ↑
  │    ╭──╮        ╭──╮
  │   ╱    ╲      ╱    ╲
  │  ╱      ╲    ╱      ╲       → 時間(t)
──┼─╱────────╲──╱────────╲──────
  │          ╲╱            ╲╱
  │
  │  |←── 1周期(T) ──→|
  │
  │  周波数 f = 1/T [Hz]
  │  振幅 A: 音の大きさ（音量）
  │  位相 φ: 波形の開始位置
==================================================
```

### 1.2 音の三要素

```python
import numpy as np

# 音の三要素を信号として表現

def generate_tone(frequency, amplitude, duration, sample_rate=44100):
    """
    音の三要素:
    - 周波数 (frequency): 音の高さ [Hz]
    - 振幅 (amplitude): 音の大きさ [0.0 - 1.0]
    - 波形 (waveform): 音色を決定
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 純音（サイン波）
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)

    # 倍音を含む音（音色が変わる）
    harmonics = (
        amplitude * np.sin(2 * np.pi * frequency * t) +          # 基本周波数
        amplitude * 0.5 * np.sin(2 * np.pi * 2 * frequency * t) + # 第2倍音
        amplitude * 0.25 * np.sin(2 * np.pi * 3 * frequency * t)  # 第3倍音
    )

    return sine_wave, harmonics

# A4 = 440Hz の音を生成
pure_tone, rich_tone = generate_tone(440, 0.8, 1.0)
print(f"純音サンプル数: {len(pure_tone)}")
print(f"倍音入りサンプル数: {len(rich_tone)}")
```

---

## 2. デジタル音声の基礎

### 2.1 サンプリング（標本化）

```
アナログ → デジタル変換（ADC）
==================================================

アナログ波形:
  ↑
  │   ╭─╮    ╭─╮
  │  ╱   ╲  ╱   ╲
  │ ╱     ╲╱     ╲
──┼──────────────────→ t

サンプリング（離散化）:
  ↑
  │   ●        ●
  │  ●  ●    ●  ●
  │ ●    ●  ●    ●
──┼──●────●──●────●──→ t
  │ ↑    ↑
  │ サンプリング間隔 = 1/fs

量子化（ビット深度で精度決定）:
  ↑ 16bit = 65,536段階
  │ ■        ■
  │ ■  ■    ■  ■
  │ ■    ■  ■    ■
──┼──■────■──■────■──→ t
==================================================

ナイキスト定理: fs ≥ 2 × fmax
  人間の可聴域 ~20kHz → fs ≥ 40kHz
  CD品質: 44.1kHz / 16bit
```

### 2.2 主要なサンプルレートと用途

```python
# 主要サンプルレートとその用途

sample_rates = {
    8000:  "電話音声（G.711）/ 音声認識の最低要件",
    16000: "音声認識標準（Whisper推奨）/ VoIP",
    22050: "低品質音声合成 / AM放送相当",
    44100: "CD品質 / 音楽配信標準",
    48000: "DVD / 動画音声 / プロオーディオ標準",
    96000: "ハイレゾ音源 / スタジオ録音",
}

# ビット深度とダイナミックレンジ
bit_depths = {
    8:  {"レベル数": 256,    "ダイナミックレンジ_dB": 48,  "用途": "低品質音声"},
    16: {"レベル数": 65536,  "ダイナミックレンジ_dB": 96,  "用途": "CD / 標準音声"},
    24: {"レベル数": 16777216, "ダイナミックレンジ_dB": 144, "用途": "プロオーディオ"},
    32: {"レベル数": "float32", "ダイナミックレンジ_dB": 192, "用途": "内部処理"},
}

# データ量計算
def calc_audio_size(sample_rate, bit_depth, channels, duration_sec):
    """非圧縮音声のデータ量を計算"""
    bytes_per_sample = bit_depth // 8
    total_bytes = sample_rate * bytes_per_sample * channels * duration_sec
    return total_bytes / (1024 * 1024)  # MB

# 1分間のステレオ音声のサイズ
cd_quality = calc_audio_size(44100, 16, 2, 60)
print(f"CD品質 1分間: {cd_quality:.1f} MB")  # 約10.1 MB
```

---

## 3. フーリエ変換

### 3.1 時間領域と周波数領域

```
フーリエ変換の概念
==================================================

時間領域                  周波数領域
(波形)                   (スペクトル)
                  FFT
  ↑ ╭╮  ╭╮   ────────→    ↑
  │╱  ╲╱  ╲               │ ▌
  │        ╱╲              │ ▌  ▌
──┼──────────→ t   ────────┼─▌──▌──▌──→ f
                           │440 880 1320
                    IFFT      Hz  Hz  Hz
                ←────────
                           基本波 + 倍音成分

重要な関係:
- 時間領域の複雑な波形 = 周波数領域の単純な成分の合成
- 短い音 → 広い周波数帯域（不確定性原理）
- 周期的な音 → 離散的なスペクトル線
==================================================
```

### 3.2 FFTの実装

```python
import numpy as np

def compute_fft(signal, sample_rate):
    """
    高速フーリエ変換（FFT）による周波数解析

    Parameters:
        signal: 入力信号（1D配列）
        sample_rate: サンプルレート [Hz]

    Returns:
        freqs: 周波数軸 [Hz]
        magnitude: 各周波数の振幅
    """
    n = len(signal)
    # FFT計算
    fft_result = np.fft.rfft(signal)
    # 振幅スペクトル（正規化）
    magnitude = np.abs(fft_result) / n * 2
    # 周波数軸
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    return freqs, magnitude

# 440Hz + 880Hz の合成波
sr = 44100
t = np.linspace(0, 1.0, sr, endpoint=False)
signal = 0.7 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)

freqs, magnitude = compute_fft(signal, sr)

# ピーク検出
peak_indices = np.where(magnitude > 0.1)[0]
for idx in peak_indices:
    print(f"周波数: {freqs[idx]:.0f} Hz, 振幅: {magnitude[idx]:.2f}")
# 出力: 周波数: 440 Hz, 振幅: 0.70
#        周波数: 880 Hz, 振幅: 0.30
```

### 3.3 STFT（短時間フーリエ変換）とスペクトログラム

```python
import numpy as np

def compute_stft(signal, sample_rate, window_size=2048, hop_size=512):
    """
    短時間フーリエ変換（STFT）
    - 信号を小さな窓（フレーム）に分割してFFTを適用
    - 時間×周波数の2次元表現（スペクトログラム）を生成

    Parameters:
        signal: 入力信号
        sample_rate: サンプルレート
        window_size: 窓サイズ（FFTポイント数）
        hop_size: 窓のシフト量
    """
    # ハニング窓
    window = np.hanning(window_size)

    # フレーム数
    n_frames = (len(signal) - window_size) // hop_size + 1

    # STFT行列を初期化
    stft_matrix = np.zeros((window_size // 2 + 1, n_frames), dtype=complex)

    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start:start + window_size] * window
        stft_matrix[:, i] = np.fft.rfft(frame)

    # パワースペクトログラム（dBスケール）
    power_spec = np.abs(stft_matrix) ** 2
    log_spec = 10 * np.log10(power_spec + 1e-10)

    return log_spec

# パラメータの意味
stft_params = {
    "window_size": "周波数分解能を決定（大きい→高周波数分解能、低時間分解能）",
    "hop_size": "時間分解能を決定（小さい→高時間分解能、計算コスト増）",
    "window_type": "スペクトル漏れの制御（ハニング、ハミング、ブラックマン等）",
}
```

---

## 4. メル尺度とMFCC

### 4.1 メル尺度の変換

```python
import numpy as np

def hz_to_mel(hz):
    """Hz → メル尺度変換"""
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    """メル尺度 → Hz変換"""
    return 700 * (10 ** (mel / 2595) - 1)

# メル尺度は人間の聴覚特性を反映
# 低周波数域は線形に近く、高周波数域は対数的に圧縮される
frequencies = [100, 200, 500, 1000, 2000, 4000, 8000, 16000]
for f in frequencies:
    m = hz_to_mel(f)
    print(f"{f:6d} Hz → {m:7.1f} mel")

# 出力例:
#    100 Hz →   150.5 mel
#    200 Hz →   283.2 mel
#    500 Hz →   607.5 mel
#   1000 Hz →  1000.0 mel  ← 1000Hzが基準
#   2000 Hz →  1500.0 mel
#   4000 Hz →  2146.1 mel
#   8000 Hz →  2840.0 mel
#  16000 Hz →  3564.5 mel

def compute_mel_filterbank(n_filters, n_fft, sample_rate, fmin=0, fmax=None):
    """メルフィルタバンクを生成"""
    if fmax is None:
        fmax = sample_rate / 2

    # メル尺度で等間隔にフィルタ中心周波数を配置
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points)

    # FFTビンに変換
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # 三角フィルタバンク
    filterbank = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        for j in range(bins[i], bins[i + 1]):
            filterbank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
        for j in range(bins[i + 1], bins[i + 2]):
            filterbank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])

    return filterbank
```

### 4.2 MFCC（メル周波数ケプストラム係数）

```python
import numpy as np

def compute_mfcc(signal, sample_rate, n_mfcc=13, n_filters=40, n_fft=2048):
    """
    MFCC計算の完全なパイプライン

    Step 1: プリエンファシス（高周波成分の強調）
    Step 2: フレーム分割 + 窓関数
    Step 3: FFT → パワースペクトル
    Step 4: メルフィルタバンク適用
    Step 5: 対数圧縮
    Step 6: DCT（離散コサイン変換）
    """
    # Step 1: プリエンファシス
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # Step 2: フレーム分割
    frame_size = n_fft
    hop_size = frame_size // 4
    n_frames = (len(emphasized) - frame_size) // hop_size + 1

    frames = np.zeros((n_frames, frame_size))
    for i in range(n_frames):
        start = i * hop_size
        frames[i] = emphasized[start:start + frame_size] * np.hanning(frame_size)

    # Step 3: パワースペクトル
    power_spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2 / n_fft

    # Step 4: メルフィルタバンク適用
    mel_filters = compute_mel_filterbank(n_filters, n_fft, sample_rate)
    mel_spectrum = np.dot(power_spectrum, mel_filters.T)

    # Step 5: 対数圧縮
    log_mel = np.log(mel_spectrum + 1e-10)

    # Step 6: DCT → MFCC
    from scipy.fft import dct
    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    return mfcc

# 使用例（概念）
# mfcc = compute_mfcc(audio_signal, 16000)
# print(f"MFCC shape: {mfcc.shape}")  # (フレーム数, 13)
```

---

## 5. 比較表

### 5.1 音声特徴量の比較

| 特徴量 | 次元数 | 用途 | 人間の聴覚反映 | 計算コスト |
|--------|-------|------|---------------|-----------|
| 生波形 | サンプル数 | WaveNet入力 | なし | 最小 |
| FFTスペクトル | N/2+1 | 周波数解析 | なし | 低 |
| メルスペクトログラム | 80-128 | TTS入力 | 高い | 中 |
| MFCC | 13-40 | STT入力 | 高い | 中 |
| クロマグラム | 12 | 音楽解析 | 中程度 | 中 |
| ピッチ | 1 | 韻律解析 | 高い | 低 |

### 5.2 音声フォーマットの比較

| フォーマット | 圧縮 | ビットレート(参考) | 品質 | 主な用途 |
|-------------|------|-------------------|------|---------|
| WAV | 非圧縮 | ~1411 kbps (CD) | 最高 | 編集/処理 |
| FLAC | ロスレス | ~800-1000 kbps | 最高 | アーカイブ |
| MP3 | ロッシー | 128-320 kbps | 高 | 音楽配信 |
| AAC | ロッシー | 128-256 kbps | 高 | ストリーミング |
| OGG Vorbis | ロッシー | 128-320 kbps | 高 | ゲーム/Web |
| Opus | ロッシー | 6-510 kbps | 最高(低帯域) | WebRTC/VoIP |
| PCM (raw) | 非圧縮 | 可変 | 最高 | 内部処理 |

---

## 6. アンチパターン

### 6.1 アンチパターン: サンプルレート不一致の無視

```python
# BAD: サンプルレートを確認せずにモデルに入力
def bad_process(audio_file):
    import soundfile as sf
    audio, sr = sf.read(audio_file)
    # Whisperは16kHzを期待しているが、44.1kHzのまま入力
    result = whisper_model.transcribe(audio)  # 精度低下 or エラー
    return result

# GOOD: 明示的にリサンプリング
def good_process(audio_file, target_sr=16000):
    import soundfile as sf
    import librosa

    audio, sr = sf.read(audio_file)
    print(f"元のサンプルレート: {sr} Hz")

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        print(f"リサンプリング: {sr} → {target_sr} Hz")

    result = whisper_model.transcribe(audio)
    return result
```

### 6.2 アンチパターン: 窓サイズの不適切な選択

```python
# BAD: 音声の特性を考慮せずに固定の窓サイズを使用
def bad_stft(signal, sr):
    # 窓サイズが大きすぎる → 時間分解能が低下
    # 過渡的な音（破裂音など）が検出できない
    return np.fft.rfft(signal[:16384])  # 約370ms @ 44.1kHz

# GOOD: 用途に応じた窓サイズを選択
def good_stft(signal, sr, analysis_type="speech"):
    """
    推奨窓サイズ:
    - 音声認識: 25ms窓 / 10msシフト（400/160 @ 16kHz）
    - 音楽解析: 46ms窓 / 12msシフト（2048/512 @ 44.1kHz）
    - ピッチ検出: 長め（50-100ms）で周波数分解能を確保
    """
    params = {
        "speech": {"window_ms": 25, "hop_ms": 10},
        "music":  {"window_ms": 46, "hop_ms": 12},
        "pitch":  {"window_ms": 64, "hop_ms": 16},
    }

    p = params[analysis_type]
    window_size = int(sr * p["window_ms"] / 1000)
    hop_size = int(sr * p["hop_ms"] / 1000)

    # 2のべき乗に丸める（FFT効率化）
    n_fft = 2 ** int(np.ceil(np.log2(window_size)))

    print(f"窓サイズ: {window_size} ({p['window_ms']}ms), n_fft: {n_fft}")
    # STFT計算...
```

---

## 7. FAQ

### Q1: サンプルレートは高いほど良いのですか？

必ずしもそうではありません。ナイキスト定理により、記録可能な最高周波数はサンプルレートの半分です。人間の可聴域は約20kHzなので、44.1kHz（CD品質）で十分です。音声認識では16kHzが標準であり、それ以上にしても精度は向上しません。サンプルレートを上げるとデータ量と計算コストが増えるため、用途に応じた適切な値を選ぶことが重要です。

### Q2: メルスペクトログラムとMFCC、どちらを使うべきですか？

近年のディープラーニングベースのモデル（Whisper、VITS等）ではメルスペクトログラムを直接入力とするのが主流です。MFCCは次元数が少なく計算効率が良いため、従来のSTTシステムやリソース制約のある環境では有用です。一般的に、ニューラルネットワークはメルスペクトログラムからより豊富な情報を抽出できるため、十分な計算リソースがある場合はメルスペクトログラムを推奨します。

### Q3: 窓関数の種類はどう選べばよいですか？

汎用的にはハニング窓（Hann window）が推奨されます。ハニング窓はサイドローブが小さく、スペクトル漏れが少ないため、ほとんどの音声処理タスクに適しています。ハミング窓はハニング窓と似ていますが、端点がゼロにならないため音声認識でよく使われます。ブラックマン窓はさらにサイドローブが小さいですが、メインローブが広くなり周波数分解能が低下します。特別な理由がなければハニング窓を選んでください。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 音波 | 周波数（高さ）、振幅（大きさ）、波形（音色）の3要素 |
| サンプリング | ナイキスト定理: fs >= 2 * fmax。音声認識は16kHzが標準 |
| 量子化 | 16bit（CD）で十分。内部処理はfloat32推奨 |
| フーリエ変換 | 時間領域→周波数領域の変換。FFTでO(NlogN)で計算 |
| STFT | 短時間窓でFFT。時間-周波数の2D表現を生成 |
| メル尺度 | 人間の聴覚特性を反映した周波数尺度 |
| MFCC | メルスペクトログラム+DCTで得られるコンパクトな特徴量 |

## 次に読むべきガイド

- [02-tts-technologies.md](./02-tts-technologies.md) — TTS技術の詳細
- [03-stt-technologies.md](./03-stt-technologies.md) — STT技術の詳細
- [../03-development/01-audio-processing.md](../03-development/01-audio-processing.md) — librosa/torchaudioによる実装

## 参考文献

1. Smith, S.W. "The Scientist and Engineer's Guide to Digital Signal Processing" — デジタル信号処理の定番テキスト。FFT、フィルタリングの基礎を網羅
2. Müller, M. (2015). "Fundamentals of Music Processing" — 音楽情報処理の基礎。STFT、クロマグラム、MFCCを詳細に解説
3. Rabiner, L.R. & Schafer, R.W. (2010). "Theory and Applications of Digital Speech Processing" — 音声信号処理の古典的名著。サンプリングからLPC解析まで
4. Stevens, S.S. & Volkmann, J. (1940). "The Relation of Pitch to Frequency" — メル尺度の原論文。人間の聴覚特性に基づく周波数知覚の研究

---

## 8. 高度な音声解析技術

### 8.1 ピッチ検出アルゴリズム

```python
import numpy as np

def autocorrelation_pitch(signal, sample_rate, fmin=80, fmax=400):
    """
    自己相関法によるピッチ（基本周波数 F0）検出
    
    Parameters:
        signal: 入力信号（1フレーム分）
        sample_rate: サンプルレート
        fmin: 最小周波数 [Hz]（デフォルト: 80Hz = 男性の低い声）
        fmax: 最大周波数 [Hz]（デフォルト: 400Hz = 女性の高い声）
    
    Returns:
        f0: 推定基本周波数 [Hz]。有声音でない場合は0
    """
    # ラグの範囲をサンプル数に変換
    lag_min = int(sample_rate / fmax)
    lag_max = int(sample_rate / fmin)
    
    # 自己相関を計算
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:]  # 正のラグのみ
    
    # 正規化
    autocorr = autocorr / autocorr[0]
    
    # 指定範囲内でピークを探索
    search_range = autocorr[lag_min:lag_max]
    if len(search_range) == 0:
        return 0.0
    
    peak_idx = np.argmax(search_range) + lag_min
    peak_value = autocorr[peak_idx]
    
    # 有声/無声判定（閾値）
    if peak_value < 0.3:
        return 0.0  # 無声音
    
    f0 = sample_rate / peak_idx
    return f0


def yin_pitch_detection(signal, sample_rate, fmin=80, fmax=500, threshold=0.1):
    """
    YINアルゴリズムによるピッチ検出
    - 自己相関法より精度が高い
    - 2002年にCheveigne & Kawahara が提案
    
    特徴:
    - 差分関数の累積平均正規化
    - オクターブエラーが少ない
    """
    # Step 1: 差分関数
    tau_min = int(sample_rate / fmax)
    tau_max = int(sample_rate / fmin)
    
    n = len(signal)
    diff = np.zeros(tau_max)
    
    for tau in range(1, tau_max):
        diff[tau] = np.sum((signal[:n-tau] - signal[tau:n]) ** 2)
    
    # Step 2: 累積平均正規化差分関数（CMNDF）
    cmndf = np.ones(tau_max)
    running_sum = 0.0
    for tau in range(1, tau_max):
        running_sum += diff[tau]
        cmndf[tau] = diff[tau] / (running_sum / tau) if running_sum > 0 else 1.0
    
    # Step 3: 閾値以下の最初のディップを探索
    for tau in range(tau_min, tau_max):
        if cmndf[tau] < threshold:
            # パラボラ補間で精度向上
            if tau > 0 and tau < tau_max - 1:
                alpha = cmndf[tau - 1]
                beta = cmndf[tau]
                gamma = cmndf[tau + 1]
                peak = tau + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            else:
                peak = tau
            return sample_rate / peak
    
    return 0.0  # ピッチ検出失敗


# ピッチ検出の応用例
def analyze_voice_characteristics(audio, sr):
    """音声の特性を分析（ピッチ、フォルマント、エネルギー）"""
    frame_size = int(0.025 * sr)  # 25ms
    hop_size = int(0.010 * sr)    # 10ms
    
    f0_values = []
    energy_values = []
    
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        
        # ピッチ検出
        f0 = yin_pitch_detection(frame, sr)
        f0_values.append(f0)
        
        # エネルギー
        energy = np.sqrt(np.mean(frame ** 2))
        energy_values.append(energy)
    
    # 有声音フレームのみでF0統計を計算
    voiced_f0 = [f for f in f0_values if f > 0]
    
    return {
        "mean_f0": np.mean(voiced_f0) if voiced_f0 else 0,
        "std_f0": np.std(voiced_f0) if voiced_f0 else 0,
        "min_f0": np.min(voiced_f0) if voiced_f0 else 0,
        "max_f0": np.max(voiced_f0) if voiced_f0 else 0,
        "voicing_ratio": len(voiced_f0) / len(f0_values),
        "mean_energy": np.mean(energy_values),
    }
```

### 8.2 フォルマント分析

```python
import numpy as np
from scipy.signal import lfilter, lpc

def extract_formants(signal, sample_rate, n_formants=4, lpc_order=12):
    """
    LPC（線形予測符号化）によるフォルマント抽出
    
    フォルマント: 声道の共鳴周波数
    - F1: 顎の開き（開口度）に関連 (~300-800Hz)
    - F2: 舌の前後位置に関連 (~800-2500Hz)
    - F3: 唇の丸めに関連 (~2500-3500Hz)
    
    Parameters:
        signal: 音声信号（1フレーム分）
        sample_rate: サンプルレート
        n_formants: 抽出するフォルマント数
        lpc_order: LPC次数（通常 2 + サンプルレート/1000）
    """
    # プリエンファシス（高域強調）
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    
    # ハミング窓適用
    windowed = emphasized * np.hamming(len(emphasized))
    
    # LPC係数を計算
    a = lpc(windowed, lpc_order)
    
    # LPC多項式の根を求める
    roots = np.roots(a)
    
    # 正の虚部を持つ根のみを選択（共役のうち片方）
    roots = roots[np.imag(roots) >= 0]
    
    # 角度から周波数に変換
    angles = np.arctan2(np.imag(roots), np.real(roots))
    frequencies = angles * (sample_rate / (2 * np.pi))
    
    # 帯域幅を計算
    bandwidths = -0.5 * sample_rate * np.log(np.abs(roots)) / np.pi
    
    # 有効なフォルマントのフィルタリング
    valid = (frequencies > 90) & (frequencies < sample_rate / 2 - 50) & (bandwidths < 400)
    frequencies = frequencies[valid]
    bandwidths = bandwidths[valid]
    
    # 周波数でソート
    sorted_idx = np.argsort(frequencies)
    frequencies = frequencies[sorted_idx][:n_formants]
    bandwidths = bandwidths[sorted_idx][:n_formants]
    
    return frequencies, bandwidths

# 日本語母音のフォルマント参考値
japanese_vowel_formants = {
    "あ (a)": {"F1": 800, "F2": 1200, "特徴": "最も開口度が大きい"},
    "い (i)": {"F1": 300, "F2": 2300, "特徴": "F2が高い（舌が前方）"},
    "う (u)": {"F1": 350, "F2": 1100, "特徴": "唇が丸まる"},
    "え (e)": {"F1": 500, "F2": 1900, "特徴": "中程度の開口"},
    "お (o)": {"F1": 500, "F2": 800,  "特徴": "F2が低い（舌が後方）"},
}
```

### 8.3 音声品質指標

```python
# 音声品質を測定するための各種指標

def compute_snr(clean_signal, noisy_signal):
    """
    SNR（信号対雑音比）を計算
    
    SNR = 10 * log10(signal_power / noise_power)
    
    高いほど良い。一般的な目安:
    - > 40dB: 非常にクリーン
    - 20-40dB: 良好
    - 10-20dB: ノイズが気になる
    - < 10dB: 品質が低い
    """
    noise = noisy_signal - clean_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)


def compute_pesq_wrapper(reference_path, degraded_path, sample_rate=16000):
    """
    PESQ（Perceptual Evaluation of Speech Quality）の計算
    ITU-T P.862 に基づく客観的音声品質指標
    
    スコア範囲: -0.5 ~ 4.5
    - 4.5: 劣化なし
    - 3.8+: 非常に良い
    - 3.0-3.8: 良い
    - 2.0-3.0: 普通
    - < 2.0: 悪い
    """
    from pesq import pesq
    import soundfile as sf
    
    ref, sr_ref = sf.read(reference_path)
    deg, sr_deg = sf.read(degraded_path)
    
    # リサンプリングが必要な場合
    if sr_ref != sample_rate:
        import librosa
        ref = librosa.resample(ref, orig_sr=sr_ref, target_sr=sample_rate)
    if sr_deg != sample_rate:
        deg = librosa.resample(deg, orig_sr=sr_deg, target_sr=sample_rate)
    
    # 長さを合わせる
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]
    
    score = pesq(sample_rate, ref, deg, 'wb')  # 'wb'=広帯域, 'nb'=狭帯域
    return score


def compute_stoi(clean, degraded, sr=16000):
    """
    STOI（Short-Time Objective Intelligibility）
    音声の明瞭度を評価する指標
    
    スコア範囲: 0 ~ 1
    - > 0.9: 非常に明瞭
    - 0.7-0.9: 明瞭
    - 0.5-0.7: やや不明瞭
    - < 0.5: 不明瞭
    """
    from pystoi import stoi
    
    min_len = min(len(clean), len(degraded))
    return stoi(clean[:min_len], degraded[:min_len], sr, extended=True)


# 包括的な音声品質評価
def comprehensive_quality_assessment(reference_path, test_path, sr=16000):
    """音声品質の包括的評価"""
    import soundfile as sf
    
    ref, _ = sf.read(reference_path)
    test, _ = sf.read(test_path)
    
    min_len = min(len(ref), len(test))
    ref, test = ref[:min_len], test[:min_len]
    
    results = {
        "SNR (dB)": compute_snr(ref, test),
        "PESQ": "要pesqライブラリ",
        "STOI": "要pystoiライブラリ",
        "RMS差": float(np.sqrt(np.mean((ref - test) ** 2))),
        "ピーク差": float(np.max(np.abs(ref)) - np.max(np.abs(test))),
        "スペクトル歪み": "要計算",
    }
    
    return results
```

---

## 9. 実践的な音声処理パターン

### 9.1 リアルタイム音声入力と処理

```python
import numpy as np
import queue
import threading

class RealtimeAudioProcessor:
    """リアルタイム音声入力処理の基本パターン"""
    
    def __init__(self, sample_rate=16000, chunk_duration_ms=100):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.audio_queue = queue.Queue()
        self.is_running = False
    
    def start_recording(self):
        """マイクからの音声入力を開始"""
        import sounddevice as sd
        
        self.is_running = True
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            self.audio_queue.put(indata.copy().flatten())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=callback,
        )
        self.stream.start()
    
    def stop_recording(self):
        """録音停止"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def process_chunks(self, processor_func):
        """チャンクごとにプロセッサ関数を適用"""
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                result = processor_func(chunk)
                if result is not None:
                    yield result
            except queue.Empty:
                continue


class CircularAudioBuffer:
    """リングバッファによる音声データ管理"""
    
    def __init__(self, duration_sec, sample_rate=16000):
        self.buffer_size = int(duration_sec * sample_rate)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.sample_rate = sample_rate
    
    def write(self, data):
        """データをバッファに書き込み"""
        n = len(data)
        if n >= self.buffer_size:
            self.buffer[:] = data[-self.buffer_size:]
            self.write_pos = 0
        else:
            end_pos = self.write_pos + n
            if end_pos <= self.buffer_size:
                self.buffer[self.write_pos:end_pos] = data
            else:
                first_part = self.buffer_size - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:n - first_part] = data[first_part:]
            self.write_pos = end_pos % self.buffer_size
    
    def read_last(self, duration_sec):
        """直近N秒のデータを取得"""
        n_samples = int(duration_sec * self.sample_rate)
        n_samples = min(n_samples, self.buffer_size)
        
        if self.write_pos >= n_samples:
            return self.buffer[self.write_pos - n_samples:self.write_pos].copy()
        else:
            first_part = self.buffer[-(n_samples - self.write_pos):]
            second_part = self.buffer[:self.write_pos]
            return np.concatenate([first_part, second_part])
```

### 9.2 音声ファイルの効率的なバッチ処理

```python
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class BatchProcessResult:
    """バッチ処理結果"""
    file_path: str
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None
    processing_time: float = 0.0

class AudioBatchProcessor:
    """音声ファイルのバッチ処理エンジン"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def process_directory(
        self,
        input_dir: str,
        processor: Callable,
        file_patterns: list = ["*.wav", "*.mp3", "*.flac"],
        output_dir: Optional[str] = None,
    ) -> list[BatchProcessResult]:
        """ディレクトリ内の音声ファイルを並列処理"""
        input_path = Path(input_dir)
        files = []
        for pattern in file_patterns:
            files.extend(input_path.glob(pattern))
        
        if not files:
            print(f"警告: {input_dir} に音声ファイルが見つかりません")
            return []
        
        print(f"処理対象: {len(files)} ファイル")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(self._process_single, f, processor, output_dir): f
                for f in files
            }
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                status = "OK" if result.success else "NG"
                print(f"  [{status}] {Path(result.file_path).name} "
                      f"({result.processing_time:.2f}s)")
        
        # サマリー
        success = sum(1 for r in results if r.success)
        print(f"\n完了: {success}/{len(results)} 成功")
        return results
    
    def _process_single(self, file_path, processor, output_dir):
        """単一ファイルの処理"""
        import time
        start = time.time()
        
        try:
            result = processor(str(file_path), output_dir)
            return BatchProcessResult(
                file_path=str(file_path),
                success=True,
                result=result,
                processing_time=time.time() - start,
            )
        except Exception as e:
            return BatchProcessResult(
                file_path=str(file_path),
                success=False,
                error=str(e),
                processing_time=time.time() - start,
            )
```

---

## 10. 音声データの可視化

### 10.1 波形とスペクトログラムの可視化

```python
import numpy as np

def create_visualization_data(audio, sr):
    """
    音声データの可視化用データを生成
    （matplotlib不要の数値データとして出力）
    """
    # 波形データ（ダウンサンプリングして表示用に）
    display_sr = 1000  # 1kHz に間引き
    factor = sr // display_sr
    waveform_display = audio[::factor]
    
    # スペクトログラム
    n_fft = 2048
    hop_length = 512
    n_frames = (len(audio) - n_fft) // hop_length + 1
    
    spectrogram = np.zeros((n_fft // 2 + 1, n_frames))
    window = np.hanning(n_fft)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        spectrogram[:, i] = np.abs(np.fft.rfft(frame))
    
    log_spec = 20 * np.log10(spectrogram + 1e-10)
    
    # 時間軸
    time_axis = np.arange(n_frames) * hop_length / sr
    # 周波数軸
    freq_axis = np.fft.rfftfreq(n_fft, 1.0 / sr)
    
    return {
        "waveform": waveform_display,
        "waveform_time": np.arange(len(waveform_display)) / display_sr,
        "spectrogram": log_spec,
        "spec_time": time_axis,
        "spec_freq": freq_axis,
        "duration": len(audio) / sr,
        "sample_rate": sr,
    }

# ASCII アートによる簡易スペクトログラム表示
def ascii_spectrogram(audio, sr, n_rows=20, n_cols=80):
    """ターミナルで表示可能なASCIIスペクトログラム"""
    n_fft = 2048
    hop_length = len(audio) // n_cols
    
    chars = " ░▒▓█"
    
    spec_data = np.zeros((n_fft // 2 + 1, n_cols))
    window = np.hanning(n_fft)
    
    for i in range(n_cols):
        start = i * hop_length
        if start + n_fft > len(audio):
            break
        frame = audio[start:start + n_fft] * window
        spec_data[:, i] = np.abs(np.fft.rfft(frame))
    
    log_spec = 20 * np.log10(spec_data + 1e-10)
    
    # n_rows にリサイズ（周波数軸を間引き）
    freq_indices = np.linspace(0, spec_data.shape[0] - 1, n_rows, dtype=int)
    display = log_spec[freq_indices]
    
    # 正規化
    vmin, vmax = np.percentile(display, [5, 95])
    display = np.clip((display - vmin) / (vmax - vmin + 1e-10), 0, 1)
    
    # ASCII文字に変換
    lines = []
    for row in reversed(range(n_rows)):
        line = ""
        for col in range(min(n_cols, display.shape[1])):
            idx = int(display[row, col] * (len(chars) - 1))
            line += chars[idx]
        lines.append(line)
    
    return "\n".join(lines)
```

---

## 11. デジタルフィルタの基礎

### 11.1 FIRフィルタとIIRフィルタ

```python
import numpy as np
from scipy.signal import firwin, butter, sosfilt, lfilter

def apply_lowpass_fir(audio, sr, cutoff_hz, n_taps=101):
    """
    FIRローパスフィルタ
    - 線形位相（位相歪みなし）
    - 安定（常に安定）
    - タップ数が多いと計算コストが高い
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    coeffs = firwin(n_taps, normalized_cutoff)
    return lfilter(coeffs, 1.0, audio)

def apply_highpass_iir(audio, sr, cutoff_hz, order=4):
    """
    IIRハイパスフィルタ（Butterworth）
    - 少ない次数で急峻なカットオフ
    - 非線形位相
    - 不安定になる可能性あり（高次数時）
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    sos = butter(order, normalized_cutoff, btype='high', output='sos')
    return sosfilt(sos, audio)

def apply_bandpass(audio, sr, low_hz, high_hz, order=4):
    """バンドパスフィルタ"""
    nyquist = sr / 2
    low = low_hz / nyquist
    high = high_hz / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, audio)

# 用途別のフィルタ設定例
filter_presets = {
    "音声認識前処理": {
        "type": "bandpass",
        "low": 80,     # 80Hz以下をカット（ハム音、振動）
        "high": 8000,  # 8kHz以上をカット（高周波ノイズ）
        "説明": "音声に不要な帯域を除去してSTT精度を向上",
    },
    "ポッドキャスト": {
        "type": "highpass",
        "cutoff": 80,
        "説明": "低域のランブルノイズを除去",
    },
    "電話音声（狭帯域）": {
        "type": "bandpass",
        "low": 300,
        "high": 3400,
        "説明": "電話帯域（G.711）に制限",
    },
    "サブベース除去": {
        "type": "highpass",
        "cutoff": 30,
        "説明": "人間に聞こえない超低域を除去（DC成分含む）",
    },
}
```

### 11.2 ディジタルフィルタの周波数応答

```python
def analyze_filter_response(b, a, sr, n_points=1024):
    """
    フィルタの周波数応答を計算
    
    Parameters:
        b, a: フィルタ係数
        sr: サンプルレート
        n_points: 計算ポイント数
    
    Returns:
        freqs: 周波数軸 [Hz]
        magnitude_db: 振幅応答 [dB]
        phase_deg: 位相応答 [度]
    """
    w = np.linspace(0, np.pi, n_points)
    
    # 周波数応答 H(e^jw) を計算
    h = np.zeros(n_points, dtype=complex)
    for i, wi in enumerate(w):
        # 分子
        num = sum(b[k] * np.exp(-1j * k * wi) for k in range(len(b)))
        # 分母
        den = sum(a[k] * np.exp(-1j * k * wi) for k in range(len(a)))
        h[i] = num / den
    
    freqs = w * sr / (2 * np.pi)
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
    phase_deg = np.degrees(np.angle(h))
    
    return freqs, magnitude_db, phase_deg
```

---

## 12. 追加のFAQ

### Q4: WAVファイルとFLACファイル、どちらを使うべきですか？

WAVは非圧縮で読み書きが最も高速ですが、ファイルサイズが大きくなります。FLACはロスレス圧縮で元のデータと完全に同一の信号を復元でき、サイズは約50-60%に縮小されます。音声処理のパイプライン内部ではWAV（またはメモリ上の生配列）が効率的ですが、保存・転送にはFLACが適しています。AI APIへの入力としてはWAVが最も互換性が高いですが、多くのAPIはFLACやMP3も受け付けます。

### Q5: 音声データの前処理で最も重要なステップは何ですか？

最も重要なのはサンプルレートの統一（リサンプリング）です。多くのSTTモデルは16kHzを前提としており、不一致があると精度が大幅に低下します。次に重要なのは正規化（音量の統一）で、これにより入力レベルの違いによるモデル性能のバラつきを防ぎます。3番目にノイズ除去ですが、これは音声の品質に応じて適用の有無を判断してください。クリーンな環境で録音された音声にノイズ除去を適用すると、逆に品質が低下することがあります。

### Q6: dBFS、dBSPL、LUFS の違いは何ですか？

これらは異なる「音量」の尺度です。(1) dBFS (decibels Full Scale): デジタル音声での絶対的な音量。0 dBFS が最大値で、実際の値は常に負（例: -20 dBFS）。(2) dBSPL (decibels Sound Pressure Level): 物理的な音圧レベル。20μPaを基準とした人間の耳に届く音量。(3) LUFS (Loudness Units Full Scale): ITU-R BS.1770に基づく知覚的ラウドネス。人間の聴覚特性（K-weightフィルタ）を考慮した値で、配信プラットフォームの音量基準（Spotify: -14 LUFS等）に使われます。音声AIの開発では主にdBFSとLUFSを使います。

---

## まとめ（拡張版）

| 項目 | 要点 |
|------|------|
| 音波 | 周波数（高さ）、振幅（大きさ）、波形（音色）の3要素 |
| サンプリング | ナイキスト定理: fs >= 2 * fmax。音声認識は16kHzが標準 |
| 量子化 | 16bit（CD）で十分。内部処理はfloat32推奨 |
| フーリエ変換 | 時間領域→周波数領域の変換。FFTでO(NlogN)で計算 |
| STFT | 短時間窓でFFT。時間-周波数の2D表現を生成 |
| メル尺度 | 人間の聴覚特性を反映した周波数尺度 |
| MFCC | メルスペクトログラム+DCTで得られるコンパクトな特徴量 |
| ピッチ検出 | YINアルゴリズムが高精度。自己相関法は高速 |
| フォルマント | LPC分析で声道共鳴を推定。母音識別に重要 |
| フィルタ | FIR（安定・線形位相）vs IIR（低コスト・非線形位相） |
| 品質指標 | SNR, PESQ, STOIが主要。用途に応じて選択 |

## 次に読むべきガイド

- [02-tts-technologies.md](./02-tts-technologies.md) — TTS技術の詳細
- [03-stt-technologies.md](./03-stt-technologies.md) — STT技術の詳細
- [../03-development/01-audio-processing.md](../03-development/01-audio-processing.md) — librosa/torchaudioによる実装

## 参考文献

1. Smith, S.W. "The Scientist and Engineer's Guide to Digital Signal Processing" — デジタル信号処理の定番テキスト。FFT、フィルタリングの基礎を網羅
2. Müller, M. (2015). "Fundamentals of Music Processing" — 音楽情報処理の基礎。STFT、クロマグラム、MFCCを詳細に解説
3. Rabiner, L.R. & Schafer, R.W. (2010). "Theory and Applications of Digital Speech Processing" — 音声信号処理の古典的名著。サンプリングからLPC解析まで
4. Stevens, S.S. & Volkmann, J. (1940). "The Relation of Pitch to Frequency" — メル尺度の原論文。人間の聴覚特性に基づく周波数知覚の研究
5. de Cheveigne, A. & Kawahara, H. (2002). "YIN, a fundamental frequency estimator for speech and music" — YINアルゴリズムの原論文。高精度ピッチ検出
6. Rix, A.W., et al. (2001). "Perceptual evaluation of speech quality (PESQ)" — PESQ音声品質指標の原論文
