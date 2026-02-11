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
