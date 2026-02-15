# ボイスクローニング — RVC、So-VITS、倫理的考慮

> 音声クローニング技術の仕組み、主要フレームワーク（RVC、So-VITS-SVC）、倫理的・法的課題を解説する

## この章で学ぶこと

1. ボイスクローニングの技術的原理（話者埋め込み、声質変換、ゼロショット合成）
2. 主要フレームワーク（RVC、So-VITS-SVC、OpenVoice）の実装と使い分け
3. 音声前処理・後処理パイプラインの設計と品質最適化
4. リアルタイム声質変換の実装パターンとレイテンシ最適化
5. 倫理的・法的課題と責任あるAI音声技術の利用
6. 音声透かし・AI音声検出技術

---

## 1. ボイスクローニングの技術基盤

### 1.1 技術分類

```
ボイスクローニングの3つのアプローチ
==================================================

1. TTS型クローニング（テキスト → ターゲット音声）
   ┌──────┐   ┌──────────────┐   ┌──────────┐
   │テキスト│──→│ TTS + 話者   │──→│ターゲット│
   │      │   │ 埋め込み      │   │  音声    │
   └──────┘   └──────────────┘   └──────────┘
   * 例: VALL-E, YourTTS, XTTS
   * テキストから直接ターゲット話者の音声を生成
   * ナレーション、吹き替え、オーディオブックに最適

2. SVC型（歌声変換: ソース音声 → ターゲット音声）
   ┌──────┐   ┌──────────────┐   ┌──────────┐
   │ソース │──→│ 声質変換     │──→│ターゲット│
   │ 音声  │   │ (Voice Conv.)│   │  音声    │
   └──────┘   └──────────────┘   └──────────┘
   * 例: RVC, So-VITS-SVC
   * ソース音声の内容・韻律を保持しつつ声質のみ変換
   * 歌声カバー、ボイスチェンジャーに最適

3. ゼロショット型（少量サンプルでクローン）
   ┌──────┐   ┌──────────────┐   ┌──────────┐
   │参照   │──→│ 話者特徴     │   │          │
   │音声   │   │ 抽出         │──→│ 新しい   │
   │(3-10秒)│  └──────────────┘   │  音声    │
   │      │   ┌──────────────┐   │          │
   │テキスト│──→│ ベースTTS    │──→│          │
   └──────┘   └──────────────┘   └──────────┘
   * 例: OpenVoice, ElevenLabs, XTTS v2
   * 学習不要で即座にクローニング可能
   * プロトタイプ、少量データのケースに最適
==================================================
```

### 1.2 話者埋め込み（Speaker Embedding）

```python
# 話者埋め込みの概念

import torch
import torch.nn as nn

class SpeakerEncoder(nn.Module):
    """
    話者埋め込みエンコーダ
    - 音声から話者固有の特徴ベクトルを抽出
    - 声質、話し方の癖、発声特性をコンパクトに表現
    """

    def __init__(self, input_dim=80, embedding_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 256, num_layers=3, batch_first=True)
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, mel_spectrogram):
        """
        入力: メルスペクトログラム (batch, time, 80)
        出力: 話者埋め込みベクトル (batch, 256)
        """
        # LSTMで時系列を処理
        output, (hidden, _) = self.lstm(mel_spectrogram)
        # 最後の隠れ状態を射影
        embedding = self.projection(hidden[-1])
        # L2正規化
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
        return embedding

# 話者埋め込みの利用
# 同じ話者の音声 → 近いベクトル（コサイン類似度 > 0.85）
# 異なる話者の音声 → 遠いベクトル（コサイン類似度 < 0.5）
```

### 1.3 内容と話者の分離（Disentanglement）

ボイスクローニングの核心は「何を言っているか（内容）」と「誰が言っているか（話者）」を分離することにある。

```python
import torch
import torch.nn as nn
import numpy as np

class ContentSpeakerDisentanglement:
    """
    内容特徴と話者特徴の分離

    音声 → [Content Encoder] → 内容特徴 (言語情報、音素、韻律)
         → [Speaker Encoder] → 話者特徴 (声質、音色、フォルマント)

    変換時:
    ソース音声の内容特徴 + ターゲット話者の特徴 → 変換音声
    """

    def __init__(self):
        self.content_encoder = None   # HuBERT, ContentVec 等
        self.speaker_encoder = None   # ECAPA-TDNN, WavLM 等
        self.decoder = None           # VITS, HiFi-GAN 等

    def extract_content(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        内容特徴の抽出

        HuBERT (Hidden-Unit BERT) を使用:
        - 自己教師あり学習で音声の言語的特徴を学習
        - 話者情報を含まない言語内容の表現を出力
        - 各フレーム（20ms）で768次元のベクトル

        ContentVec (改良版):
        - HuBERTを声質情報を除去するように改良
        - So-VITS-SVCで標準的に使用
        """
        # HuBERT特徴抽出の概念コード
        import torchaudio
        from transformers import HubertModel

        model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        model.eval()

        # 前処理
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_tensor = resampler(audio_tensor)

        with torch.no_grad():
            outputs = model(audio_tensor)
            content_features = outputs.last_hidden_state  # (1, T, 768)

        return content_features.squeeze(0).numpy()

    def extract_speaker(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        話者特徴の抽出

        ECAPA-TDNN を使用:
        - 話者照合タスクで学習された強力なエンコーダ
        - 時間方向の統計プーリングで固定長ベクトルを生成
        - 192次元の話者埋め込みを出力

        使用場面:
        - 話者クラスタリング（複数話者の識別）
        - 話者照合（同一人物かの判定）
        - ボイスクローニングの話者条件付け
        """
        from speechbrain.pretrained import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )

        # 話者埋め込みの取得
        signal = torch.FloatTensor(audio).unsqueeze(0)
        embeddings = classifier.encode_batch(signal)

        return embeddings.squeeze().numpy()

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """話者埋め込みのコサイン類似度"""
        cosine_sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10
        )
        return float(cosine_sim)


class PitchExtractor:
    """
    ピッチ（F0）抽出

    ボイスクローニングでは正確なピッチ推定が品質に直結する。
    特に歌声変換では、ピッチのずれが致命的な品質低下を引き起こす。
    """

    def __init__(self, sr: int = 44100):
        self.sr = sr

    def extract_with_crepe(self, audio: np.ndarray) -> tuple:
        """
        CREPE (Convolutional Representation for Pitch Estimation)
        - ニューラルネットワークベースのピッチ推定
        - 高い推定精度（従来法の10倍以上）
        - RVCのデフォルトピッチ推定器
        """
        import crepe

        # ピッチ推定
        time_axis, frequency, confidence, activation = crepe.predict(
            audio, self.sr,
            model_capacity="full",  # tiny, small, medium, large, full
            viterbi=True,           # ビタビ平滑化（安定性向上）
            step_size=10,           # ms単位のステップサイズ
        )

        # 信頼度の低いフレームは無声区間
        frequency[confidence < 0.5] = 0

        return time_axis, frequency, confidence

    def extract_with_rmvpe(self, audio: np.ndarray) -> np.ndarray:
        """
        RMVPE (Robust Model for Voice Pitch Estimation)
        - CREPEの改良版、ノイズに強い
        - RVC v2で推奨されるピッチ推定器
        - 歌声の装飾音（ビブラート、こぶし）の追従性が高い
        """
        # RMVPEはRVCプロジェクト内で提供されるピッチ推定器
        # ここでは概念的な使い方を示す
        from infer.lib.rmvpe import RMVPE

        rmvpe = RMVPE("rmvpe.pt", is_half=False, device="cuda")
        f0 = rmvpe.infer_from_audio(audio, thred=0.03)

        return f0

    def pitch_shift(self, f0: np.ndarray, semitones: int) -> np.ndarray:
        """
        ピッチシフト（半音単位）

        男声→女声: +12 (1オクターブ上)
        女声→男声: -12 (1オクターブ下)
        同性間の微調整: -3〜+3
        """
        if semitones == 0:
            return f0
        # 半音ごとに 2^(1/12) 倍
        factor = 2 ** (semitones / 12)
        shifted = f0.copy()
        voiced = shifted > 0  # 有声区間のみシフト
        shifted[voiced] *= factor
        return shifted

    def add_vibrato(self, f0: np.ndarray, rate: float = 5.5,
                     depth: float = 0.5, sr: int = 100) -> np.ndarray:
        """
        ビブラートの付加

        Parameters:
            rate: ビブラート周波数 (Hz)、歌声では4-7Hz
            depth: ビブラート深さ (半音)、0.3-1.0が自然
            sr: F0のサンプリングレート
        """
        t = np.arange(len(f0)) / sr
        vibrato = depth * np.sin(2 * np.pi * rate * t)
        # 半音単位のビブラートをF0に適用
        modulated = f0.copy()
        voiced = modulated > 0
        modulated[voiced] *= 2 ** (vibrato[voiced] / 12)
        return modulated
```

### 1.4 ニューラルボコーダの役割

```python
class NeuralVocoder:
    """
    ニューラルボコーダの概要

    ボコーダ = メルスペクトログラム → 波形 の変換器

    ボイスクローニングのパイプライン:
    内容特徴 + 話者特徴 + F0 → メルスペクトログラム → ボコーダ → 波形

    主要なニューラルボコーダ:
    """

    VOCODER_COMPARISON = {
        "HiFi-GAN": {
            "方式": "GAN (Generative Adversarial Network)",
            "品質": "非常に高い",
            "速度": "リアルタイムの100倍以上",
            "特徴": "RVC, So-VITS-SVCで使用。品質と速度のバランスが最良",
            "パラメータ数": "~14M",
        },
        "WaveNet": {
            "方式": "自己回帰モデル",
            "品質": "最高（基準）",
            "速度": "非常に遅い（リアルタイムの1/10以下）",
            "特徴": "Google DeepMindが開発。品質は最高だが実用的でない",
            "パラメータ数": "~2M",
        },
        "WaveGlow": {
            "方式": "Flow-based",
            "品質": "高い",
            "速度": "リアルタイムの10倍",
            "特徴": "NVIDIAが開発。並列生成可能",
            "パラメータ数": "~87M",
        },
        "UnivNet": {
            "方式": "GAN",
            "品質": "高い",
            "速度": "HiFi-GAN同等",
            "特徴": "位相情報の再構成が優秀",
            "パラメータ数": "~15M",
        },
        "BigVGAN": {
            "方式": "GAN",
            "品質": "非常に高い",
            "速度": "HiFi-GAN同等",
            "特徴": "大規模学習で汎化性能が高い。未知の話者にも強い",
            "パラメータ数": "~112M",
        },
    }

    @staticmethod
    def hifigan_inference_example():
        """HiFi-GAN によるボコーディングの例"""
        import torch
        from models import Generator  # HiFi-GAN のジェネレータ

        # モデルロード
        generator = Generator()
        checkpoint = torch.load("g_02500000")
        generator.load_state_dict(checkpoint["generator"])
        generator.eval()

        # メルスペクトログラムから波形生成
        mel = torch.FloatTensor(mel_spectrogram).unsqueeze(0)

        with torch.no_grad():
            audio = generator(mel)  # 波形出力

        # audio shape: (1, 1, T) where T = mel_length * hop_size
        return audio.squeeze().numpy()
```

---

## 2. 主要フレームワーク

### 2.1 RVC（Retrieval-based Voice Conversion）

```python
# RVC の使い方（概念）

class RVCPipeline:
    """
    RVC: 検索ベース声質変換
    - HuBERT で音声の内容（Content）特徴を抽出
    - ピッチ推定で声の高さを保持
    - 学習済みモデルでターゲット話者の声質に変換
    """

    def __init__(self, model_path: str, index_path: str = None):
        self.model = self.load_model(model_path)
        self.index = self.load_index(index_path)  # FAISS検索インデックス
        self.hubert = self.load_hubert()

    def convert(
        self,
        source_audio: str,
        pitch_shift: int = 0,    # 半音単位（男→女: +12）
        feature_ratio: float = 0.75,  # 検索特徴の混合比
        protect: float = 0.33,   # 子音保護（0=保護なし, 0.5=最大）
    ):
        """声質変換を実行"""
        import soundfile as sf

        audio, sr = sf.read(source_audio)

        # Step 1: HuBERT で内容特徴を抽出
        content_features = self.hubert.extract(audio, sr)

        # Step 2: FAISS インデックスで近傍検索（検索ベースの声質マッチング）
        if self.index is not None:
            retrieved = self.index.search(content_features, k=8)
            content_features = (
                feature_ratio * retrieved +
                (1 - feature_ratio) * content_features
            )

        # Step 3: ピッチ推定と変換
        f0 = self.estimate_pitch(audio, sr)
        if pitch_shift != 0:
            f0 = f0 * (2 ** (pitch_shift / 12))

        # Step 4: 声質変換モデルで音声生成
        converted = self.model.generate(content_features, f0, protect=protect)

        return converted

    def train(self, dataset_path: str, epochs: int = 200):
        """RVCモデルの学習"""
        # 学習データ: ターゲット話者の音声 10分〜1時間
        # Step 1: HuBERT 特徴抽出
        # Step 2: ピッチ抽出
        # Step 3: モデル学習（VITS ベースのジェネレータ）
        # Step 4: FAISS インデックス作成
        pass

# 使用例
rvc = RVCPipeline("model.pth", "model.index")
converted_audio = rvc.convert(
    "input.wav",
    pitch_shift=0,
    feature_ratio=0.75,
)
```

### 2.1b RVC のアーキテクチャ詳細

```
RVC v2 の内部アーキテクチャ
==================================================

入力音声 (wav, 44.1kHz/48kHz)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
┌─────────────┐                  ┌─────────────────┐
│ HuBERT      │                  │ ピッチ推定       │
│ (Content    │                  │ (RMVPE/CREPE)   │
│  Encoder)   │                  │                 │
│             │                  │ → F0 contour    │
│ 768-dim     │                  │ → 有声/無声     │
│ features    │                  │                 │
└──────┬──────┘                  └────────┬────────┘
       │                                  │
       ▼                                  │
┌─────────────────┐                       │
│ FAISS Index     │                       │
│ (k-NN検索)      │                       │
│                 │                       │
│ 学習データの     │                       │
│ 近傍特徴を検索   │                       │
│                 │                       │
│ feature_ratio   │                       │
│ で混合          │                       │
└────────┬────────┘                       │
         │                                │
         ▼                                ▼
┌─────────────────────────────────────────────┐
│  VITS-based Generator                        │
│                                             │
│  ┌──────────────┐   ┌────────────────┐      │
│  │ Posterior     │   │ Flow           │      │
│  │ Encoder      │──→│ (Normalizing   │      │
│  │              │   │  Flow)          │      │
│  └──────────────┘   └───────┬────────┘      │
│                              │               │
│                     ┌────────▼────────┐      │
│  F0 ─────────────→ │ Decoder         │      │
│                     │ (HiFi-GAN v2)   │      │
│  Speaker ────────→ │                 │      │
│  Embedding         │ → 波形生成      │      │
│                     └────────┬────────┘      │
└──────────────────────────────┼───────────────┘
                               │
                               ▼
                    変換された音声 (wav)

モデルバリエーション:
- v1: 256次元HuBERT特徴
- v2: 768次元HuBERT特徴 + RMVPE
- v2 48kHz: 高品質48kHz出力対応

protect パラメータ:
- 0.0: 保護なし（完全変換）
- 0.33: 推奨値（子音を保護）
- 0.5: 最大保護（原音に近い子音）
- 子音（破裂音、摩擦音など）は声質変換で
  劣化しやすいため、部分的に原音を保持する
==================================================
```

### 2.1c RVC 学習パイプラインの詳細

```python
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class RVCTrainingPipeline:
    """
    RVC モデルの学習パイプライン

    学習フロー:
    1. 音声データの前処理
    2. HuBERT 特徴抽出
    3. ピッチ（F0）抽出
    4. モデル学習
    5. FAISS インデックス構築
    """

    def __init__(self, experiment_name: str, sr: int = 40000,
                 version: str = "v2"):
        self.experiment_name = experiment_name
        self.sr = sr
        self.version = version
        self.exp_dir = Path(f"logs/{experiment_name}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_audio(self, input_dir: str,
                          target_sr: int = None) -> Dict:
        """
        Step 1: 音声データの前処理

        処理内容:
        - リサンプリング（40kHz or 48kHz）
        - 無音区間の除去
        - セグメント分割（3-10秒）
        - ラウドネス正規化
        """
        import librosa
        import soundfile as sf

        if target_sr is None:
            target_sr = self.sr

        input_path = Path(input_dir)
        audio_files = (
            list(input_path.glob("*.wav")) +
            list(input_path.glob("*.mp3")) +
            list(input_path.glob("*.flac"))
        )

        stats = {"total_files": len(audio_files), "total_duration": 0,
                 "segments": 0, "skipped": 0}

        output_dir = self.exp_dir / "preprocessed"
        output_dir.mkdir(exist_ok=True)

        for audio_file in audio_files:
            try:
                audio, sr = librosa.load(str(audio_file), sr=target_sr)

                # ノイズ評価
                snr = self._estimate_snr(audio)
                if snr < 15:
                    print(f"警告: SNR低 ({snr:.1f}dB): {audio_file.name}")
                    stats["skipped"] += 1
                    continue

                # 無音除去
                intervals = librosa.effects.split(
                    audio, top_db=40, frame_length=2048, hop_length=512
                )

                # セグメント分割
                segment_idx = 0
                for start, end in intervals:
                    segment = audio[start:end]
                    duration = len(segment) / target_sr

                    if duration < 1.0:
                        continue  # 短すぎるセグメントはスキップ

                    # 長いセグメントは分割
                    max_duration = 10.0
                    if duration > max_duration:
                        n_splits = int(np.ceil(duration / max_duration))
                        split_size = len(segment) // n_splits
                        for i in range(n_splits):
                            sub = segment[i * split_size:(i + 1) * split_size]
                            self._save_segment(
                                sub, target_sr, output_dir,
                                audio_file.stem, segment_idx
                            )
                            segment_idx += 1
                            stats["segments"] += 1
                    else:
                        self._save_segment(
                            segment, target_sr, output_dir,
                            audio_file.stem, segment_idx
                        )
                        segment_idx += 1
                        stats["segments"] += 1

                    stats["total_duration"] += duration

            except Exception as e:
                print(f"エラー: {audio_file.name}: {e}")
                stats["skipped"] += 1

        print(f"前処理完了: {stats['segments']}セグメント, "
              f"{stats['total_duration']:.1f}秒, "
              f"{stats['skipped']}ファイルスキップ")
        return stats

    def extract_features(self) -> None:
        """
        Step 2-3: HuBERT 特徴とF0の抽出

        各セグメントに対して:
        - HuBERT 特徴ベクトル（768次元/フレーム）
        - F0 輪郭（ピッチ）
        を抽出して保存
        """
        preprocessed_dir = self.exp_dir / "preprocessed"
        feature_dir = self.exp_dir / "features"
        feature_dir.mkdir(exist_ok=True)

        wav_files = list(preprocessed_dir.glob("*.wav"))
        print(f"特徴抽出: {len(wav_files)} ファイル")

        for wav_file in wav_files:
            # HuBERT 特徴抽出
            hubert_features = self._extract_hubert(wav_file)
            np.save(
                feature_dir / f"{wav_file.stem}_hubert.npy",
                hubert_features
            )

            # F0 抽出（RMVPE）
            f0 = self._extract_f0(wav_file)
            np.save(
                feature_dir / f"{wav_file.stem}_f0.npy",
                f0
            )

    def train_model(self, epochs: int = 200, batch_size: int = 8,
                     save_every: int = 50, lr: float = 1e-4) -> None:
        """
        Step 4: モデル学習

        学習パラメータの目安:
        - 10分のデータ: 200-300エポック
        - 30分のデータ: 100-200エポック
        - 1時間以上: 50-100エポック
        - バッチサイズ: VRAM 8GB → 4-8, 12GB → 8-16
        """
        training_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "save_every_epoch": save_every,
            "pretrain_g": "pretrained_v2/f0G40k.pth",
            "pretrain_d": "pretrained_v2/f0D40k.pth",
            "if_f0": True,
            "version": self.version,
        }

        print(f"学習開始: {epochs}エポック, バッチサイズ{batch_size}")
        print(f"設定: {training_config}")

        # 学習ループの概念コード
        # 実際にはRVCのtrain.pyを使用
        # python train_nsf_sim_cache_sid_load_pretrain.py
        return training_config

    def build_index(self, n_trees: int = 384) -> str:
        """
        Step 5: FAISS インデックス構築

        FAISSインデックスにより、推論時に学習データの
        近傍特徴を高速検索し、変換品質を向上させる
        """
        import faiss

        feature_dir = self.exp_dir / "features"
        hubert_files = list(feature_dir.glob("*_hubert.npy"))

        # 全特徴ベクトルを結合
        all_features = []
        for f in hubert_files:
            features = np.load(f)
            all_features.append(features)

        features_matrix = np.vstack(all_features).astype(np.float32)
        print(f"インデックス構築: {features_matrix.shape[0]} ベクトル, "
              f"{features_matrix.shape[1]} 次元")

        # IVFインデックスの構築
        n_ivf = min(int(4 * np.sqrt(features_matrix.shape[0])),
                    features_matrix.shape[0])

        index = faiss.index_factory(
            features_matrix.shape[1],
            f"IVF{n_ivf},Flat"
        )
        index.train(features_matrix)
        index.add(features_matrix)

        index_path = str(self.exp_dir / f"{self.experiment_name}.index")
        faiss.write_index(index, index_path)
        print(f"インデックス保存: {index_path}")

        return index_path

    def _save_segment(self, audio, sr, output_dir, stem, idx):
        """セグメントの保存"""
        import soundfile as sf
        output_path = output_dir / f"{stem}_{idx:04d}.wav"
        sf.write(str(output_path), audio, sr)

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """簡易SNR推定"""
        signal_power = np.mean(audio ** 2)
        # 最も静かな10%をノイズフロアと推定
        sorted_power = np.sort(np.abs(audio))
        noise_floor = sorted_power[:len(sorted_power) // 10]
        noise_power = np.mean(noise_floor ** 2) + 1e-10
        return 10 * np.log10(signal_power / noise_power)

    def _extract_hubert(self, wav_path):
        """HuBERT特徴抽出のプレースホルダー"""
        return np.random.randn(100, 768)  # 実際はHuBERTモデルを使用

    def _extract_f0(self, wav_path):
        """F0抽出のプレースホルダー"""
        return np.random.randn(100)  # 実際はRMVPEを使用


# 学習品質を左右する要因
training_quality_factors = {
    "データ品質（最重要）": {
        "クリーンな録音": "ノイズ、残響、BGMなし",
        "一定の音量": "ラウドネス正規化済み",
        "一人の話者のみ": "複数話者の混在は不可",
        "自然な発話": "朗読調よりも自然な会話",
        "録音環境": "同一環境での録音が望ましい",
    },
    "データ量": {
        "最低": "10分（品質は限定的）",
        "推奨": "30分-1時間（良好な品質）",
        "理想": "2時間以上（最高品質）",
        "注意": "量より質。ノイズ入り10時間 < クリーン30分",
    },
    "ハイパーパラメータ": {
        "エポック数": "過学習に注意。200-300が目安",
        "学習率": "1e-4 が安定。過学習時は下げる",
        "バッチサイズ": "VRAM容量に依存。大きいほど安定",
        "feature_ratio": "0.6-0.8 が推奨。高すぎると不自然",
    },
}
```

### 2.2 So-VITS-SVC（Singing Voice Conversion）

```python
# So-VITS-SVC の概念

class SoVITSSVC:
    """
    So-VITS-SVC: 歌声変換特化
    - VITS アーキテクチャベース
    - ContentVec / HuBERT で内容抽出
    - 歌声のピッチ、ビブラート、表現を保持しつつ声質変換
    """

    def __init__(self, model_path: str, config_path: str):
        self.model = self.load_model(model_path, config_path)

    def infer(
        self,
        source_audio: str,
        speaker_id: int = 0,
        transpose: int = 0,      # キー変更（半音単位）
        auto_predict_f0: bool = True,
        cluster_ratio: float = 0.0,  # クラスタリング特徴の混合比
        noise_scale: float = 0.4,    # ノイズスケール（表現力制御）
    ):
        """歌声変換の推論"""
        # 内部処理:
        # 1. 音声をContentVecで内容エンコード
        # 2. F0（ピッチ）推定
        # 3. VITSデコーダでターゲット話者の音声を生成
        pass

    def preprocess_dataset(self, audio_dir: str):
        """学習データの前処理"""
        # 1. 音声を自動的にセグメント分割
        # 2. リサンプリング（44.1kHz）
        # 3. 無音除去
        # 4. ラウドネス正規化
        preprocessing_steps = {
            "resample": 44100,
            "silence_threshold": -40,  # dB
            "segment_duration": (5, 15),  # 秒
            "normalize_loudness": -23,  # LUFS
        }
        return preprocessing_steps

# 学習に必要なデータ量の目安
training_guidelines = {
    "最小": "30分（品質低）",
    "推奨": "2-4時間（良好な品質）",
    "理想": "5時間以上（最高品質）",
    "データ品質": "ドライ音声（リバーブなし）、ノイズなし、一定音量",
}
```

### 2.2b RVC と So-VITS-SVC の差異

```
RVC と So-VITS-SVC の詳細比較
==================================================

                  RVC                   So-VITS-SVC
用途           話し声・歌声          歌声に特化
内容抽出       HuBERT               ContentVec
F0推定         RMVPE/CREPE           DIO/Harvest/CREPE
検索機構       FAISS k-NN            K-Meansクラスタ
デコーダ       VITS + HiFi-GAN       VITS + NSF-HiFi-GAN
学習データ量   10分〜                2時間〜
学習時間       30分〜1時間           数時間〜半日
推論速度       高速（リアルタイム可）  中速
WebUI          付属                   別途
モデルサイズ   ~50MB                 ~150MB

RVCの利点:
- 少量データでも良好な結果
- リアルタイム変換に対応
- WebUI完備で初心者にも使いやすい
- FAISSによる高品質な検索ベース変換

So-VITS-SVCの利点:
- 歌声の表現力（ビブラート等）の再現性が高い
- ContentVecにより話者情報がより除去される
- ノイズスケールで表現力を制御可能
- 大量データでの最高品質
==================================================
```

### 2.3 OpenVoice（ゼロショット）

```python
# OpenVoice: ゼロショットボイスクローニング

class OpenVoiceExample:
    """
    OpenVoice（MyShell AI）の使い方
    - 数秒の参照音声だけでクローニング
    - ベースTTSで音声を生成 → トーン変換でターゲット話者に
    - 多言語対応
    """

    def clone_and_speak(
        self,
        reference_audio: str,    # ターゲット話者の参照音声（3-10秒）
        text: str,               # 読み上げるテキスト
        language: str = "ja",
    ):
        """ゼロショットボイスクローニング"""
        from openvoice import se_extractor
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter

        # Step 1: 参照音声から話者特徴を抽出
        target_se = se_extractor.get_se(
            reference_audio,
            tone_color_converter,
            vad=True,
        )

        # Step 2: ベースTTSで音声を生成
        base_audio = base_speaker_tts.tts(
            text,
            language=language,
            speaker="default",
        )

        # Step 3: トーンカラー変換（ベース音声 → ターゲット話者の声質）
        converted = tone_color_converter.convert(
            audio_src=base_audio,
            src_se=base_speaker_se,
            tgt_se=target_se,
        )

        return converted
```

### 2.4 XTTS v2（Coqui TTS）

```python
class XTTSv2Example:
    """
    XTTS v2: 多言語ゼロショットTTS
    - 24kHz / 16言語対応
    - 6秒の参照音声でクローニング
    - Apache 2.0ライセンス（商用利用可）
    """

    def setup(self):
        """XTTS v2 のセットアップ"""
        from TTS.api import TTS

        # モデルのダウンロードとロード
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.tts.to("cuda" if torch.cuda.is_available() else "cpu")

    def clone_voice(self, reference_audio: str, text: str,
                     language: str = "ja",
                     output_path: str = "output.wav"):
        """
        ゼロショットボイスクローニング

        Parameters:
            reference_audio: 参照音声（6秒以上推奨）
            text: 読み上げテキスト
            language: 言語コード（ja, en, zh, ko, ...）
        """
        self.tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            language=language,
            file_path=output_path,
        )
        return output_path

    def streaming_clone(self, reference_audio: str, text: str,
                         language: str = "ja"):
        """
        ストリーミングTTS（低遅延版）

        最初のチャンクが ~200ms で出力開始
        全体のレイテンシ: ~500ms
        """
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config = XttsConfig()
        model = Xtts.init_from_config(config)

        # 参照音声のエンコード
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=reference_audio
        )

        # ストリーミング生成
        chunks = model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=20,
        )

        for i, chunk in enumerate(chunks):
            # 各チャンクをリアルタイムで再生
            yield chunk.cpu().numpy()

    def finetune(self, dataset_dir: str, epochs: int = 10):
        """
        ファインチューニング

        ゼロショットでは不十分な場合、少量データで追加学習
        15分程度のデータで大幅に品質向上
        """
        finetuning_config = {
            "output_path": f"models/xtts_finetuned/",
            "epochs": epochs,
            "batch_size": 2,
            "learning_rate": 5e-6,  # ファインチューニング用の低い学習率
            "dataset": {
                "path": dataset_dir,
                "language": "ja",
                "min_duration": 2.0,
                "max_duration": 12.0,
            },
        }
        return finetuning_config


# ゼロショット品質改善のテクニック
zero_shot_tips = {
    "参照音声の選び方": [
        "6-15秒の長さが最適（短すぎるとNG）",
        "クリーンな録音（ノイズ、BGM、残響なし）",
        "自然な話し方（演技調は避ける）",
        "感情表現が中立的なサンプル",
        "複数の参照音声を試して最良を選ぶ",
    ],
    "テキストの工夫": [
        "長い文は分割して生成し、結合する",
        "句読点を適切に配置して間(ま)を制御",
        "感嘆符、疑問符で抑揚を調整",
        "漢字の読みが正しいか確認（固有名詞）",
    ],
    "後処理": [
        "リバーブを軽く付加して自然さ向上",
        "EQで不自然な周波数帯を補正",
        "ノイズゲートで微小ノイズを除去",
        "ラウドネス正規化（-16 LUFS for配信）",
    ],
}
```

---

## 3. リアルタイム声質変換

### 3.1 リアルタイムパイプラインの設計

```python
import numpy as np
import torch
import threading
import queue
from typing import Optional

class RealTimeVoiceConverter:
    """
    リアルタイム声質変換パイプライン

    レイテンシ目標:
    - 通話品質: < 100ms
    - 配信品質: < 200ms
    - 許容最大: < 300ms（人間が遅延を感じ始める）

    バッファ設計:
    - 入力バッファ: block_size サンプル（通常512-2048）
    - 処理バッファ: extra_size サンプル（前後文脈）
    - 出力バッファ: block_size サンプル
    """

    def __init__(self, model_path: str, index_path: str = None,
                 block_size: int = 512, extra_size: int = 48000,
                 sr: int = 44100, device: str = "cuda"):
        self.block_size = block_size
        self.extra_size = extra_size
        self.sr = sr
        self.device = device

        # RVCモデルのロード
        self.model = self._load_model(model_path)
        self.index = self._load_index(index_path)

        # リングバッファ
        self.input_buffer = np.zeros(extra_size + block_size)
        self.output_queue = queue.Queue(maxsize=10)

        # 設定
        self.pitch_shift = 0
        self.feature_ratio = 0.75
        self.protect = 0.33

        # 統計
        self.processing_times = []

    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        """
        1ブロックの音声を変換

        Parameters:
            audio_block: (block_size,) の入力音声

        Returns:
            変換された音声ブロック
        """
        import time
        start = time.perf_counter()

        # 入力バッファを更新（リングバッファ）
        self.input_buffer = np.roll(self.input_buffer, -self.block_size)
        self.input_buffer[-self.block_size:] = audio_block

        # 文脈を含む全体を処理
        with torch.no_grad():
            # HuBERT特徴抽出
            input_tensor = torch.FloatTensor(self.input_buffer).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # 変換処理（概念）
            converted = self._convert(input_tensor)

        # 出力の最後のblock_sizeサンプルを返す
        output = converted[-self.block_size:]

        elapsed = time.perf_counter() - start
        self.processing_times.append(elapsed)

        return output

    def get_latency_stats(self) -> dict:
        """レイテンシ統計"""
        if not self.processing_times:
            return {}

        times = np.array(self.processing_times[-100:])  # 最新100ブロック
        buffer_latency = self.block_size / self.sr * 1000  # ms

        return {
            "バッファレイテンシ": f"{buffer_latency:.1f} ms",
            "処理時間（平均）": f"{np.mean(times)*1000:.1f} ms",
            "処理時間（最大）": f"{np.max(times)*1000:.1f} ms",
            "処理時間（P95）": f"{np.percentile(times, 95)*1000:.1f} ms",
            "総レイテンシ（推定）": f"{buffer_latency + np.mean(times)*1000:.1f} ms",
            "リアルタイム比": f"{np.mean(times) * self.sr / self.block_size:.2f}x",
        }

    def _load_model(self, path):
        """モデルロードのプレースホルダー"""
        return None

    def _load_index(self, path):
        """インデックスロードのプレースホルダー"""
        return None

    def _convert(self, input_tensor):
        """変換処理のプレースホルダー"""
        return input_tensor.cpu().numpy().squeeze()


class AudioStreamHandler:
    """
    オーディオストリーム処理

    PyAudioやsounddeviceを使ったリアルタイム入出力
    """

    def __init__(self, converter: RealTimeVoiceConverter,
                 input_device: int = None, output_device: int = None):
        self.converter = converter
        self.input_device = input_device
        self.output_device = output_device
        self.is_running = False

    def start(self):
        """ストリーム処理を開始"""
        import sounddevice as sd

        self.is_running = True

        def callback(indata, outdata, frames, time, status):
            if status:
                print(f"ステータス: {status}")

            # 入力をモノラルに変換
            mono_input = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()

            # 声質変換
            converted = self.converter.process_block(mono_input)

            # 出力（ステレオ）
            outdata[:, 0] = converted
            if outdata.shape[1] > 1:
                outdata[:, 1] = converted

        block_size = self.converter.block_size
        sr = self.converter.sr

        print(f"ストリーム開始: {sr}Hz, ブロックサイズ {block_size}")
        print(f"理論レイテンシ: {block_size / sr * 1000:.1f} ms")

        with sd.Stream(
            samplerate=sr,
            blocksize=block_size,
            channels=1,
            dtype=np.float32,
            callback=callback,
            device=(self.input_device, self.output_device),
        ):
            print("リアルタイム変換中... Ctrl+C で停止")
            while self.is_running:
                import time
                time.sleep(0.1)

    def stop(self):
        """ストリーム処理を停止"""
        self.is_running = False

    def list_devices(self):
        """利用可能なオーディオデバイスを一覧"""
        import sounddevice as sd
        print(sd.query_devices())


# レイテンシ最適化のガイドライン
latency_optimization = {
    "block_size の調整": {
        "512 (11.6ms @ 44.1kHz)": "最低レイテンシ、GPU負荷高",
        "1024 (23.2ms)": "バランス型、推奨",
        "2048 (46.4ms)": "安定性重視、古いGPUで推奨",
    },
    "GPU最適化": {
        "ONNX Runtime": "PyTorchから20-30%高速化",
        "TensorRT": "最大50%高速化（NVIDIA専用）",
        "Half Precision (FP16)": "メモリ削減+高速化",
        "CUDA Stream": "非同期処理でGPU効率化",
    },
    "CPU最適化（GPU非使用時）": {
        "ONNX Runtime CPU": "PyTorchの2-3倍高速",
        "OpenVINO": "Intel CPU で最適化",
        "CoreML": "Apple Silicon で最適化",
    },
}
```

---

## 4. 倫理的・法的課題

### 4.1 リスクマトリクス

```
ボイスクローニングのリスクマトリクス
==================================================

影響度
  高 │  ┌──────────┐  ┌──────────────┐
     │  │詐欺・なり│  │政治的       │
     │  │すまし    │  │ディープフェイク│
     │  └──────────┘  └──────────────┘
  中 │  ┌──────────┐  ┌──────────────┐
     │  │無断での  │  │死者の声の   │
     │  │声の商用利用│ │再現        │
     │  └──────────┘  └──────────────┘
  低 │  ┌──────────┐  ┌──────────────┐
     │  │パロディ  │  │個人的な     │
     │  │コンテンツ│  │楽しみ       │
     │  └──────────┘  └──────────────┘
     └──────────────────────────────────
        低        発生確率        高

対策レイヤー:
  技術: 音声透かし / AI検出 / 認証
  法律: 個人の声の権利保護法 / 不正利用罰則
  倫理: 同意取得 / 使用ガイドライン / 透明性
==================================================
```

### 4.2 各国の法規制の動向

```python
# 各国のボイスクローニング規制の状況（2025年時点）

regulatory_landscape = {
    "米国": {
        "連邦法": "包括的な規制なし（2025年時点）",
        "州法": {
            "カリフォルニア": "AB 2655 — AI生成音声にラベル付け義務",
            "テネシー": "ELVIS Act — 音声のAI複製を規制",
            "ニューヨーク": "声の肖像権を広範に保護",
            "イリノイ": "BIPA — 生体情報（声紋含む）の保護",
        },
        "FTC": "AI音声によるなりすまし詐欺を重点取締り",
    },
    "EU": {
        "AI Act": "ディープフェイク音声に透明性要件（2025年施行開始）",
        "GDPR": "声は個人データ。処理には法的根拠が必要",
        "要件": [
            "AI生成音声であることの明示",
            "本人の同意（正当な利益でない場合）",
            "データ処理の記録",
            "異議申立て権の保障",
        ],
    },
    "日本": {
        "直接規制": "ボイスクローニング専用の法律は未整備（2025年）",
        "適用可能な既存法": {
            "不正競争防止法": "著名人の声の不正使用",
            "著作権法": "実演家の権利（歌声等）",
            "肖像権": "判例に基づく保護（声も含む可能性）",
            "刑法": "詐欺罪（なりすまし目的）",
            "個人情報保護法": "声紋データの保護",
        },
        "動向": "文化庁がAI生成コンテンツに関するガイドラインを検討中",
    },
    "中国": {
        "AI合成音声規制": "2023年施行、AI音声にラベル必須",
        "ディープフェイク規制": "同意なしの合成は違法",
        "プラットフォーム責任": "配信プラットフォームに検出義務",
    },
}

# 商用利用時のチェックリスト
commercial_use_checklist = [
    "対象者からの明示的な書面同意を取得",
    "使用目的・範囲を契約書に明記",
    "同意の撤回プロセスを確立",
    "AI生成音声であることのラベル付け",
    "音声透かしの埋め込み",
    "悪用防止策の実装（利用制限、監視）",
    "データ保護影響評価（DPIA）の実施",
    "該当国・地域の法規制の確認",
    "保険の検討（賠償責任）",
    "定期的な法規制アップデートの確認",
]
```

### 4.3 責任ある利用のためのガイドライン

```python
# ボイスクローニングの責任ある利用チェックリスト

responsible_use_checklist = {
    "同意": {
        "対象者の明示的同意を取得": True,
        "使用目的を明確に説明": True,
        "撤回権を保障": True,
    },
    "透明性": {
        "AI生成音声であることを明示": True,
        "使用技術の開示": True,
        "生成物に透かしを埋め込み": True,
    },
    "安全性": {
        "なりすまし防止策": True,
        "悪用検出メカニズム": True,
        "アクセス制御": True,
    },
    "法令遵守": {
        "個人情報保護法": True,
        "各国の声の権利法": True,
        "プラットフォーム利用規約": True,
    },
}

# 音声透かしの埋め込み例
def embed_watermark(audio, sr, identifier="ai-generated"):
    """不可聴域に透かしを埋め込み"""
    import numpy as np

    # 超音波帯域（18-20kHz）にID情報をエンコード
    # 人間には聞こえないが、検出器で識別可能
    watermark_freq = 19000  # Hz
    t = np.arange(len(audio)) / sr
    watermark = 0.001 * np.sin(2 * np.pi * watermark_freq * t)

    return audio + watermark
```

### 4.4 AI音声検出技術

```python
import numpy as np
import torch
from typing import Dict, Tuple

class AIVoiceDetector:
    """
    AI生成音声の検出器

    ディープフェイク音声を検出するための技術:
    1. スペクトル分析（AI生成特有のアーティファクト検出）
    2. 話者照合（参照音声との一致度）
    3. 音声透かしの検出
    4. ニューラルネットワークベースの分類
    """

    def detect_spectral_artifacts(self, audio: np.ndarray,
                                    sr: int = 44100) -> Dict:
        """
        スペクトルアーティファクトの検出

        AI生成音声に特有のパターン:
        - ナイキスト周波数付近のエネルギー異常
        - 高周波帯域の不自然な減衰
        - スペクトログラムの周期的パターン
        - フォルマント遷移の不自然さ
        """
        from scipy.signal import stft as scipy_stft

        # STFTの計算
        f, t, Zxx = scipy_stft(audio, fs=sr, nperseg=2048)
        magnitude = np.abs(Zxx)

        # 1. 高周波帯域の分析
        high_freq_idx = f > sr * 0.4  # ナイキスト近傍
        high_freq_energy = np.mean(magnitude[high_freq_idx])
        total_energy = np.mean(magnitude)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-10)

        # 2. スペクトラルフラットネス（自然音声は不均一）
        spectral_flatness = np.exp(np.mean(np.log(magnitude + 1e-10))) / (
            np.mean(magnitude) + 1e-10
        )

        # 3. サブバンドごとのエネルギー分布
        subbands = np.array_split(magnitude, 8, axis=0)
        subband_energies = [np.mean(sb) for sb in subbands]
        subband_variance = np.var(subband_energies)

        # 判定
        indicators = {
            "high_freq_ratio": round(float(high_freq_ratio), 4),
            "spectral_flatness": round(float(spectral_flatness), 4),
            "subband_variance": round(float(subband_variance), 6),
        }

        # 簡易スコアリング（0=自然, 1=AI生成の可能性高）
        score = 0.0
        if high_freq_ratio < 0.01:  # 高周波が不自然に少ない
            score += 0.3
        if spectral_flatness > 0.5:  # スペクトルが均一すぎる
            score += 0.3
        if subband_variance < 0.001:  # エネルギー分布が均一すぎる
            score += 0.4

        indicators["ai_probability"] = round(min(score, 1.0), 2)
        return indicators

    def detect_watermark(self, audio: np.ndarray,
                          sr: int = 44100) -> Dict:
        """
        音声透かしの検出

        超音波帯域（18-20kHz）の透かし信号を検出
        """
        from scipy.fft import rfft, rfftfreq

        # FFT
        N = len(audio)
        freqs = rfftfreq(N, 1/sr)
        fft_vals = np.abs(rfft(audio))

        # 透かし帯域のエネルギー
        watermark_band = (freqs > 18000) & (freqs < 20000)
        reference_band = (freqs > 15000) & (freqs < 17000)

        watermark_energy = np.mean(fft_vals[watermark_band])
        reference_energy = np.mean(fft_vals[reference_band]) + 1e-10

        ratio = watermark_energy / reference_energy

        return {
            "watermark_detected": ratio > 2.0,
            "watermark_energy_ratio": round(float(ratio), 2),
            "watermark_band_energy": round(float(watermark_energy), 6),
        }

    def speaker_verification(self, audio: np.ndarray,
                               reference_audio: np.ndarray,
                               sr: int = 16000) -> Dict:
        """
        話者照合

        参照音声と比較して同一話者かどうかを判定
        AI生成音声は本物の話者との微妙な差異がある
        """
        from speechbrain.pretrained import SpeakerRecognition

        verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )

        # 類似度スコア
        score, prediction = verification.verify_batch(
            torch.FloatTensor(audio).unsqueeze(0),
            torch.FloatTensor(reference_audio).unsqueeze(0),
        )

        return {
            "similarity_score": round(float(score), 4),
            "same_speaker": bool(prediction),
            "threshold": 0.25,
        }


# AI音声検出の主要サービス・ツール
detection_tools = {
    "Resemble AI Detect": {
        "種別": "API サービス",
        "精度": "~98%",
        "特徴": "リアルタイム検出対応",
    },
    "Pindrop": {
        "種別": "企業向けソリューション",
        "精度": "~99%",
        "特徴": "コールセンター向け、大規模対応",
    },
    "SpeechBrain": {
        "種別": "OSS フレームワーク",
        "精度": "モデル依存",
        "特徴": "研究用、カスタマイズ可能",
    },
    "ASVspoof": {
        "種別": "研究ベンチマーク",
        "精度": "ベースライン",
        "特徴": "音声なりすまし検出の国際評価",
    },
}
```

---

## 5. 比較表

### 5.1 主要ボイスクローニングツール比較

| 項目 | RVC | So-VITS-SVC | OpenVoice | ElevenLabs | XTTS v2 | VALL-E |
|------|-----|------------|-----------|-----------|---------|--------|
| 種別 | OSS | OSS | OSS | SaaS | OSS | 研究 |
| タイプ | SVC(話声) | SVC(歌声) | ゼロショット | ゼロショット | ゼロショット | ゼロショット |
| 必要データ | 10分〜 | 2時間〜 | 3秒 | 1分 | 6秒 | 3秒 |
| 学習時間 | 30分〜 | 数時間 | 不要 | 不要 | 不要 | - |
| 品質 | 高い | 高い(歌声) | 中〜高 | 非常に高い | 高い | 高い |
| リアルタイム | 対応 | 一部 | 非対応 | 対応 | 一部対応 | - |
| 日本語 | 対応 | 対応 | 対応 | 対応 | 対応 | 限定 |
| GPU要件 | 4GB+ | 8GB+ | 4GB+ | 不要 | 4GB+ | - |
| ライセンス | MIT | AGPL | MIT | 商用 | Apache 2.0 | - |

### 5.2 用途別推奨ツール

| ユースケース | 推奨 | 理由 |
|-------------|------|------|
| 歌声カバー制作 | RVC / So-VITS-SVC | 歌声変換に特化 |
| ナレーション | ElevenLabs / XTTS v2 | テキスト入力で簡単 |
| プロトタイプ | OpenVoice | ゼロショット、セットアップ簡単 |
| 高品質商用 | ElevenLabs | 最高品質、API完備 |
| リアルタイム通話 | RVC | 低遅延対応 |
| 研究開発 | RVC / OpenVoice | OSS、カスタマイズ可 |
| 多言語対応 | XTTS v2 | 16言語対応、OSS |
| オーディオブック | ElevenLabs / XTTS v2 | 長文対応、安定品質 |
| ゲーム開発 | XTTS v2 / RVC | オフライン実行可能 |
| アクセシビリティ | XTTS v2 | OSS、カスタマイズ可能 |

### 5.3 GPU/メモリ要件の詳細

| ツール | 最小VRAM | 推奨VRAM | CPU推論 | モデルサイズ |
|--------|---------|---------|---------|------------|
| RVC v2 | 4GB | 8GB | 可（遅い） | ~50MB |
| So-VITS-SVC | 8GB | 12GB | 困難 | ~150MB |
| OpenVoice | 4GB | 6GB | 可 | ~200MB |
| XTTS v2 | 4GB | 8GB | 可（遅い） | ~1.8GB |
| Demucs + RVC | 8GB | 12GB | 可（非常に遅い） | ~250MB |

---

## 6. アンチパターン

### 6.1 アンチパターン: 同意なきクローニング

```python
# BAD: 同意なしにボイスクローニングを実施
def bad_clone(public_video_url):
    # 公開動画から音声を抽出してクローニング
    audio = download_audio(public_video_url)
    model = train_voice_model(audio)  # 倫理的・法的に問題
    return model

# GOOD: 明確な同意プロセスを経る
def good_clone(consent_form, audio_files):
    """同意に基づくボイスクローニング"""
    # Step 1: 同意の確認
    if not consent_form.is_valid():
        raise ConsentError("有効な同意が取得されていません")

    if not consent_form.allows_purpose("voice_cloning"):
        raise ConsentError("ボイスクローニングの同意が含まれていません")

    # Step 2: 同意記録の保存
    consent_record = {
        "person": consent_form.person_name,
        "date": datetime.now().isoformat(),
        "purpose": consent_form.allowed_purposes,
        "expiry": consent_form.expiry_date,
        "revocable": True,
    }
    save_consent_record(consent_record)

    # Step 3: クローニング実行
    model = train_voice_model(audio_files)

    # Step 4: 透かし埋め込み
    model.set_watermark(consent_record["person"])

    return model
```

### 6.2 アンチパターン: 学習データの品質軽視

```python
# BAD: 雑多な音声をそのまま学習
def bad_training(audio_folder):
    all_files = glob("*.wav")  # ノイズ入り、異なる環境、BGM混入
    train(all_files)

# GOOD: 厳選されたクリーンデータで学習
def good_training(audio_folder):
    all_files = glob("*.wav")
    clean_files = []

    for f in all_files:
        audio, sr = load(f)
        # 品質チェック
        snr = compute_snr(audio)
        if snr < 20:
            print(f"スキップ（低SNR）: {f}")
            continue
        if has_background_music(audio):
            print(f"スキップ（BGM検出）: {f}")
            continue
        if detect_reverb_level(audio) > 0.3:
            print(f"スキップ（残響過多）: {f}")
            continue

        # 正規化
        audio = normalize(audio, target_db=-20)
        # 無音除去
        audio = trim_silence(audio, threshold_db=-40)

        clean_files.append((f, audio))

    print(f"学習データ: {len(clean_files)}/{len(all_files)} ファイル使用")
    train([f for f, _ in clean_files])
```

### 6.3 アンチパターン: ピッチ設定の不適切

```python
# BAD: 大きすぎるピッチシフト
def bad_pitch_conversion(source_audio):
    # 男声→女声で+24（2オクターブ）は不自然
    converted = rvc.convert(source_audio, pitch_shift=24)
    return converted  # ケロケロボイスになる

# GOOD: 適切なピッチシフト範囲
def good_pitch_conversion(source_audio, source_gender, target_gender):
    """性別に応じた適切なピッチシフト"""
    pitch_guide = {
        ("male", "female"): {"range": (8, 14), "recommended": 12},
        ("female", "male"): {"range": (-14, -8), "recommended": -12},
        ("male", "male"): {"range": (-5, 5), "recommended": 0},
        ("female", "female"): {"range": (-5, 5), "recommended": 0},
    }

    guide = pitch_guide.get(
        (source_gender, target_gender),
        {"range": (-5, 5), "recommended": 0}
    )

    pitch_shift = guide["recommended"]
    print(f"ピッチシフト: {pitch_shift} 半音 "
          f"(推奨範囲: {guide['range']})")

    # 段階的に試してベストを選択
    best_result = None
    best_quality = 0

    for ps in range(guide["range"][0], guide["range"][1] + 1, 2):
        result = rvc.convert(source_audio, pitch_shift=ps)
        quality = evaluate_quality(result)
        if quality > best_quality:
            best_quality = quality
            best_result = result

    return best_result
```

### 6.4 アンチパターン: 後処理の欠如

```python
# BAD: 変換結果をそのまま使用
def bad_postprocess(converted_audio):
    save(converted_audio, "output.wav")

# GOOD: 適切な後処理を行う
def good_postprocess(converted_audio, sr=44100):
    """変換後の品質向上パイプライン"""
    import numpy as np
    from scipy.signal import butter, filtfilt

    # 1. DC成分の除去
    converted_audio = converted_audio - np.mean(converted_audio)

    # 2. ハイパスフィルター（低周波ノイズ除去）
    b, a = butter(4, 80 / (sr / 2), btype='high')
    converted_audio = filtfilt(b, a, converted_audio)

    # 3. ローパスフィルター（超高周波アーティファクト除去）
    b, a = butter(4, 18000 / (sr / 2), btype='low')
    converted_audio = filtfilt(b, a, converted_audio)

    # 4. ノイズゲート
    threshold = np.max(np.abs(converted_audio)) * 0.01
    gate_mask = np.abs(converted_audio) > threshold
    converted_audio *= gate_mask.astype(float)

    # 5. ピーク正規化
    peak = np.max(np.abs(converted_audio))
    if peak > 0:
        target_db = -3
        target_level = 10 ** (target_db / 20)
        converted_audio = converted_audio * (target_level / peak)

    # 6. ディザリング（量子化ノイズの均一化）
    dither = np.random.triangular(-1, 0, 1, len(converted_audio))
    dither *= 2 ** -16  # 16bit相当のディザ
    converted_audio += dither

    # 7. 透かし埋め込み（AI生成の識別用）
    converted_audio = embed_watermark(converted_audio, sr)

    return converted_audio
```

---

## 7. 実践的なユースケース

### 7.1 歌声カバー制作パイプライン

```python
class SongCoverPipeline:
    """
    歌声カバー制作の完全パイプライン

    1. ステム分離（ボーカル抽出）
    2. 声質変換（RVC）
    3. 後処理（EQ、リバーブ）
    4. リミックス
    """

    def __init__(self, rvc_model_path: str, rvc_index_path: str = None):
        # ステム分離モデル
        from demucs.pretrained import get_model
        self.demucs = get_model("htdemucs_ft")
        self.demucs.eval()

        # RVCモデル
        self.rvc = RVCPipeline(rvc_model_path, rvc_index_path)

    def create_cover(self, input_song: str, output_path: str,
                      pitch_shift: int = 0,
                      reverb_amount: float = 0.3) -> str:
        """
        カバー制作の全工程

        Parameters:
            input_song: 原曲のパス
            output_path: 出力パス
            pitch_shift: ピッチシフト（半音単位）
            reverb_amount: リバーブ量（0-1）
        """
        import torchaudio
        from demucs.apply import apply_model

        # Step 1: ステム分離
        print("Step 1: ステム分離...")
        waveform, sr = torchaudio.load(input_song)
        if sr != self.demucs.samplerate:
            resampler = torchaudio.transforms.Resample(sr, self.demucs.samplerate)
            waveform = resampler(waveform)

        with torch.no_grad():
            sources = apply_model(
                self.demucs,
                waveform.unsqueeze(0),
                shifts=3,
                overlap=0.25,
            )

        vocals = sources[0, 3]  # vocals
        instrumental = sources[0, 0] + sources[0, 1] + sources[0, 2]

        # ボーカルを一時ファイルに保存
        vocal_path = "/tmp/vocals_temp.wav"
        torchaudio.save(vocal_path, vocals, self.demucs.samplerate)

        # Step 2: 声質変換
        print("Step 2: 声質変換...")
        converted_vocals = self.rvc.convert(
            vocal_path,
            pitch_shift=pitch_shift,
            feature_ratio=0.75,
            protect=0.33,
        )

        # Step 3: 後処理
        print("Step 3: 後処理...")
        converted_vocals = self._apply_eq(converted_vocals, sr)
        if reverb_amount > 0:
            converted_vocals = self._apply_reverb(
                converted_vocals, sr, amount=reverb_amount
            )

        # Step 4: リミックス
        print("Step 4: リミックス...")
        converted_tensor = torch.FloatTensor(converted_vocals)
        if converted_tensor.dim() == 1:
            converted_tensor = converted_tensor.unsqueeze(0).repeat(2, 1)

        # 長さを合わせる
        min_len = min(converted_tensor.shape[1], instrumental.shape[1])
        final_mix = converted_tensor[:, :min_len] + instrumental[:, :min_len]

        # ピーク正規化
        peak = torch.abs(final_mix).max()
        if peak > 0.95:
            final_mix = final_mix * (0.95 / peak)

        torchaudio.save(output_path, final_mix, self.demucs.samplerate)
        print(f"カバー完成: {output_path}")
        return output_path

    def _apply_eq(self, audio, sr):
        """ボーカル用EQ"""
        return audio  # プレースホルダー

    def _apply_reverb(self, audio, sr, amount=0.3):
        """リバーブ付加"""
        return audio  # プレースホルダー
```

### 7.2 多言語ナレーション生成

```python
class MultilingualNarrator:
    """
    多言語ナレーション生成システム

    用途:
    - eラーニング教材の多言語化
    - 企業プレゼンテーションの翻訳
    - ドキュメンタリーの吹き替え
    """

    def __init__(self):
        self.tts = None  # XTTS v2 等

    def generate_narration(self, text: str, reference_audio: str,
                            language: str, output_path: str,
                            speed: float = 1.0) -> dict:
        """
        ナレーション生成

        Parameters:
            text: ナレーションテキスト
            reference_audio: 話者の参照音声
            language: 言語コード
            speed: 話速（0.5-2.0）
        """
        # 長文の場合は文単位で分割して生成
        sentences = self._split_sentences(text, language)

        audio_segments = []
        for i, sentence in enumerate(sentences):
            print(f"生成中 [{i+1}/{len(sentences)}]: {sentence[:30]}...")

            # TTS生成
            audio = self.tts.tts(
                text=sentence,
                speaker_wav=reference_audio,
                language=language,
            )

            # 話速調整
            if speed != 1.0:
                audio = self._adjust_speed(audio, speed)

            audio_segments.append(audio)

        # セグメントの結合（自然な間を挿入）
        final_audio = self._concatenate_with_pauses(
            audio_segments, pause_ms=300
        )

        # 保存
        import soundfile as sf
        sf.write(output_path, final_audio, 24000)

        return {
            "output_path": output_path,
            "duration": len(final_audio) / 24000,
            "sentences": len(sentences),
            "language": language,
        }

    def _split_sentences(self, text, language):
        """言語に応じた文分割"""
        if language == "ja":
            # 日本語: 句点で分割
            return [s.strip() for s in text.split("。") if s.strip()]
        else:
            # その他: ピリオドで分割
            return [s.strip() for s in text.split(".") if s.strip()]

    def _adjust_speed(self, audio, speed):
        """話速調整（ピッチ維持）"""
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed)

    def _concatenate_with_pauses(self, segments, pause_ms=300):
        """セグメント結合（ポーズ付き）"""
        pause_samples = int(pause_ms / 1000 * 24000)
        pause = np.zeros(pause_samples)

        result = []
        for i, seg in enumerate(segments):
            result.append(seg)
            if i < len(segments) - 1:
                result.append(pause)

        return np.concatenate(result)
```

---

## 8. FAQ

### Q1: ボイスクローニングは合法ですか？

法的位置づけは国・地域によって大きく異なります。米国ではいくつかの州（カリフォルニア、ニューヨーク等）で「声の肖像権」を保護する法律が制定されています。EUのAI規制法では、ディープフェイク音声の生成に透明性要件を課しています。日本では2025年時点で直接的な規制法はありませんが、不正競争防止法、名誉毀損、著作権法（実演家の権利）で一定の保護があります。商用利用時は必ず法的助言を受けてください。

### Q2: RVCのリアルタイム変換のレイテンシはどの程度ですか？

RVCのリアルタイム変換では、RTX 3060以上のGPUで約40-80msのレイテンシが報告されています。設定としては、block_size=512（約11ms @ 44.1kHz）、extra_size=48000で安定動作します。レイテンシを下げるには、(1) 小さいblock_sizeを使用（ただし安定性低下）、(2) ONNX Runtime やTensorRTで推論を最適化、(3) 高性能GPU（RTX 4090等）を使用。通話用途には100ms以下が必要で、RVCは条件次第でこれを達成可能です。

### Q3: 少量データ（1分以下）でのクローニング品質を上げるには？

ゼロショット型（OpenVoice、ElevenLabs、XTTS v2）が最適です。学習型（RVC、So-VITS-SVC）は最低10分以上を推奨しますが、少量データでの改善策として、(1) データ拡張: ピッチシフト、タイムストレッチ、ノイズ付加で疑似データを増やす、(2) 転移学習: 類似した声質の事前学習モデルからファインチューニング、(3) 高品質データ: 少量でもクリーンなスタジオ録音品質を確保。ElevenLabsのInstant Voice Cloningは約1分のサンプルで実用的な品質を達成しています。

### Q4: RVCの feature_ratio と protect パラメータの最適値は？

feature_ratio は FAISS インデックスからの検索特徴とHuBERT特徴の混合比率です。0.0で検索特徴を使わず（HuBERTのみ）、1.0で検索特徴のみとなります。推奨は0.6-0.8で、高すぎるとターゲット話者に寄りすぎて不自然になり、低すぎると声質変換が不十分になります。protect は子音保護パラメータで、0.0で保護なし、0.5で最大保護です。推奨は0.33で、子音（特に破裂音、摩擦音）の明瞭度を維持しつつ自然な変換を実現します。

### Q5: 歌声カバーを高品質に仕上げるコツは？

(1) ステム分離の品質を最大化: Demucs v4 (htdemucs_ft) でshifts=5、overlap=0.5に設定。(2) 適切なピッチシフト: 原曲のキーとターゲット話者のキーの差を計算し、最適な半音数を設定。(3) RVCの設定: feature_ratio=0.75、protect=0.33が安定。(4) 後処理が極めて重要: 変換後のボーカルにEQ、コンプレッション、リバーブを適用。(5) ミックスバランス: 変換ボーカルと伴奏のバランスを原曲と同じレベルに調整。(6) 位相整合: 変換ボーカルと伴奏の位相ずれを確認・補正。

### Q6: AI音声をリアルタイムで検出する方法はありますか？

2025年時点で商用利用可能なリアルタイム検出ソリューションがいくつかあります。Pindropはコールセンター向けのリアルタイム検出を提供しており、通話中にAI音声を検知できます。Resemble AI Detectはスペクトル分析ベースのAPIを提供しています。技術的には、スペクトルアーティファクト検出、話者照合（参照音声との一致度チェック）、音声透かし検出の3つのアプローチがあります。ただし、検出技術も生成技術の進歩に追いつくのが困難な「いたちごっこ」の側面があり、完全な検出は現時点では不可能です。

### Q7: ボイスクローニングモデルのデプロイ方法は？

主なデプロイパターンは (1) ローカル実行: GPU搭載のマシンでPythonスクリプトとして実行。個人利用や小規模利用に適する。(2) APIサーバー: FastAPI/Flask + GPU サーバーでREST APIを構築。複数ユーザーからのリクエストを処理。(3) サーバーレス: AWS Lambda + SageMaker EndpointやGCP Cloud Functions + Vertex AI。スケーラビリティが高いが、コールドスタートの問題あり。(4) エッジデプロイ: ONNX Runtime + モバイルデバイス。オフライン処理が可能だが、品質はサーバー実行に劣る。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 3つのアプローチ | TTS型、SVC型（声質変換）、ゼロショット型 |
| RVC | 検索ベース声質変換。10分〜のデータで学習。リアルタイム対応 |
| So-VITS-SVC | 歌声変換特化。2時間以上のデータ推奨 |
| OpenVoice | ゼロショット。3秒の参照音声でクローニング |
| XTTS v2 | 多言語ゼロショット。16言語対応。OSS |
| 内容/話者分離 | HuBERT/ContentVecで内容抽出、ECAPA-TDNNで話者抽出 |
| ピッチ推定 | RMVPE（RVC推奨）、CREPE、DIO/Harvest |
| ニューラルボコーダ | HiFi-GAN が品質・速度のバランスで最良 |
| データ品質 | クリーン、ドライ、一定音量が最重要 |
| リアルタイム変換 | RVC で 40-80ms レイテンシ（GPU使用時） |
| 倫理 | 同意取得、透かし埋め込み、使用目的の明示が必須 |
| AI音声検出 | スペクトル分析、話者照合、透かし検出の3手法 |

## 次に読むべきガイド

- [01-voice-assistants.md](./01-voice-assistants.md) — 音声アシスタント実装
- [../00-fundamentals/02-tts-technologies.md](../00-fundamentals/02-tts-technologies.md) — TTS技術基盤
- [../03-development/02-real-time-audio.md](../03-development/02-real-time-audio.md) — リアルタイム音声
- [../01-music/01-stem-separation.md](../01-music/01-stem-separation.md) — ステム分離（カバー制作の前工程）

## 参考文献

1. Wang, Z., et al. (2023). "VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" — VALL-E論文。3秒の参照音声でのゼロショットTTS
2. Qin, Z., et al. (2024). "OpenVoice: Versatile Instant Voice Cloning" — OpenVoice論文。マルチスタイル・多言語のゼロショットクローニング
3. so-vits-svc contributors (2023). "SoftVC VITS Singing Voice Conversion" — So-VITS-SVC。歌声変換のオープンソースフレームワーク
4. RVC-Project contributors (2023). "Retrieval-based Voice Conversion WebUI" — RVC。検索ベース声質変換の主要実装
5. Coqui TTS (2024). "XTTS v2: Cross-lingual Text-to-Speech" — 多言語ゼロショットTTS
6. Kong, J., et al. (2020). "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" — HiFi-GANボコーダ
7. Hsu, W., et al. (2021). "HuBERT: Self-Supervised Speech Representation Learning" — HuBERT。自己教師あり音声表現学習
8. Kim, J., et al. (2021). "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" — VITS。RVC/So-VITSの基盤アーキテクチャ
9. EU AI Act (2024). "Regulation laying down harmonised rules on artificial intelligence" — EU AI規制法
10. ASVspoof Challenge (2024). "Automatic Speaker Verification Spoofing and Countermeasures Challenge" — 音声なりすまし検出の国際評価
