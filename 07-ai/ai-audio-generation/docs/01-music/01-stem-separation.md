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

---

## 6. FAQ

### Q1: ステム分離の品質を測る指標は何ですか？

SDR（Signal-to-Distortion Ratio）が最も広く使われる指標で、dB単位で表されます。高いほど分離品質が高く、Demucs v4は約9dB以上を達成しています。他にSIR（Source-to-Interference Ratio、他ソースの漏れ込み量）、SAR（Source-to-Artifact Ratio、アーティファクトの量）もあります。ただし、数値上の品質と聴感上の品質は必ずしも一致しないため、最終的には聴感評価も重要です。

### Q2: ボーカル以外の楽器（ギター、ピアノなど）を個別に分離できますか？

Demucs v4の6ステムモデル（htdemucs_6s）ではドラム、ベース、ボーカル、ギター、ピアノ、その他の6トラックに分離可能です。LALAL.AIはさらに細かく、エレキギター、アコースティックギター、ピアノ、シンセサイザーなど最大8種類の分離に対応しています。ただし、分離するステム数が増えるほど個々の品質は低下する傾向があります。

### Q3: リアルタイムでのステム分離は可能ですか？

2025年時点では、高品質なリアルタイムステム分離は困難です。Demucs v4はGPU使用でもリアルタイムの3-5倍の処理時間が必要です。ただし、軽量モデル（Spleeter、Open-Unmix）やDemucsのストリーミングモードを使えば準リアルタイム（1-2秒遅延）での処理は可能です。DJソフトウェア（djay、Traktor）には組み込みのリアルタイムステム分離が搭載されており、品質と速度のトレードオフの上で実用化されています。

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

## 次に読むべきガイド

- [02-audio-effects.md](./02-audio-effects.md) — AI音声エフェクト（EQ、ノイズ除去）
- [00-music-generation.md](./00-music-generation.md) — 音楽生成との組み合わせ
- [../03-development/01-audio-processing.md](../03-development/01-audio-processing.md) — librosa/torchaudioによる実装

## 参考文献

1. Rouard, S., et al. (2023). "Hybrid Transformers for Music Source Separation" — HTDemucs論文。ハイブリッドTransformerによる音楽ソース分離
2. Défossez, A. (2021). "Hybrid Spectrogram and Waveform Source Separation" — Demucs v3論文。スペクトログラム+波形のハイブリッドアプローチ
3. Hennequin, R., et al. (2020). "Spleeter: a fast and efficient music source separation tool" — Spleeter論文。Deezerによる軽量分離ツール
4. Stöter, F.R., et al. (2019). "Open-Unmix - A Reference Implementation for Music Source Separation" — Open-Unmix。オープンソースのリファレンス実装
