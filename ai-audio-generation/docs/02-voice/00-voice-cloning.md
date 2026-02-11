# ボイスクローニング — RVC、So-VITS、倫理的考慮

> 音声クローニング技術の仕組み、主要フレームワーク（RVC、So-VITS-SVC）、倫理的・法的課題を解説する

## この章で学ぶこと

1. ボイスクローニングの技術的原理（話者埋め込み、声質変換、ゼロショット合成）
2. 主要フレームワーク（RVC、So-VITS-SVC、OpenVoice）の実装と使い分け
3. 倫理的・法的課題と責任あるAI音声技術の利用

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

2. SVC型（歌声変換: ソース音声 → ターゲット音声）
   ┌──────┐   ┌──────────────┐   ┌──────────┐
   │ソース │──→│ 声質変換     │──→│ターゲット│
   │ 音声  │   │ (Voice Conv.)│   │  音声    │
   └──────┘   └──────────────┘   └──────────┘
   * 例: RVC, So-VITS-SVC

3. ゼロショット型（少量サンプルでクローン）
   ┌──────┐   ┌──────────────┐   ┌──────────┐
   │参照   │──→│ 話者特徴     │   │          │
   │音声   │   │ 抽出         │──→│ 新しい   │
   │(3-10秒)│  └──────────────┘   │  音声    │
   │      │   ┌──────────────┐   │          │
   │テキスト│──→│ ベースTTS    │──→│          │
   └──────┘   └──────────────┘   └──────────┘
   * 例: OpenVoice, ElevenLabs
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

---

## 3. 倫理的・法的課題

### 3.1 リスクマトリクス

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

### 3.2 責任ある利用のためのガイドライン

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

---

## 4. 比較表

### 4.1 主要ボイスクローニングツール比較

| 項目 | RVC | So-VITS-SVC | OpenVoice | ElevenLabs | VALL-E |
|------|-----|------------|-----------|-----------|--------|
| 種別 | OSS | OSS | OSS | SaaS | 研究 |
| タイプ | SVC(話声) | SVC(歌声) | ゼロショット | ゼロショット | ゼロショット |
| 必要データ | 10分〜 | 2時間〜 | 3秒 | 1分 | 3秒 |
| 学習時間 | 30分〜 | 数時間 | 不要 | 不要 | - |
| 品質 | 高い | 高い(歌声) | 中〜高 | 非常に高い | 高い |
| リアルタイム | 対応 | 一部 | 非対応 | 対応 | - |
| 日本語 | 対応 | 対応 | 対応 | 対応 | 限定 |
| GPU要件 | 4GB+ | 8GB+ | 4GB+ | 不要 | - |

### 4.2 用途別推奨ツール

| ユースケース | 推奨 | 理由 |
|-------------|------|------|
| 歌声カバー制作 | RVC / So-VITS-SVC | 歌声変換に特化 |
| ナレーション | ElevenLabs / OpenVoice | テキスト入力で簡単 |
| プロトタイプ | OpenVoice | ゼロショット、セットアップ簡単 |
| 高品質商用 | ElevenLabs | 最高品質、API完備 |
| リアルタイム通話 | RVC | 低遅延対応 |
| 研究開発 | RVC / OpenVoice | OSS、カスタマイズ可 |

---

## 5. アンチパターン

### 5.1 アンチパターン: 同意なきクローニング

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

### 5.2 アンチパターン: 学習データの品質軽視

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

---

## 6. FAQ

### Q1: ボイスクローニングは合法ですか？

法的位置づけは国・地域によって大きく異なります。米国ではいくつかの州（カリフォルニア、ニューヨーク等）で「声の肖像権」を保護する法律が制定されています。EUのAI規制法では、ディープフェイク音声の生成に透明性要件を課しています。日本では2025年時点で直接的な規制法はありませんが、不正競争防止法、名誉毀損、著作権法（実演家の権利）で一定の保護があります。商用利用時は必ず法的助言を受けてください。

### Q2: RVCのリアルタイム変換のレイテンシはどの程度ですか？

RVCのリアルタイム変換では、RTX 3060以上のGPUで約40-80msのレイテンシが報告されています。設定としては、block_size=512（約11ms @ 44.1kHz）、extra_size=48000で安定動作します。レイテンシを下げるには、(1) 小さいblock_sizeを使用（ただし安定性低下）、(2) ONNX Runtime やTensorRTで推論を最適化、(3) 高性能GPU（RTX 4090等）を使用。通話用途には100ms以下が必要で、RVCは条件次第でこれを達成可能です。

### Q3: 少量データ（1分以下）でのクローニング品質を上げるには？

ゼロショット型（OpenVoice、ElevenLabs）が最適です。学習型（RVC、So-VITS-SVC）は最低10分以上を推奨しますが、少量データでの改善策として、(1) データ拡張: ピッチシフト、タイムストレッチ、ノイズ付加で疑似データを増やす、(2) 転移学習: 類似した声質の事前学習モデルからファインチューニング、(3) 高品質データ: 少量でもクリーンなスタジオ録音品質を確保。ElevenLabsのInstant Voice Cloningは約1分のサンプルで実用的な品質を達成しています。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 3つのアプローチ | TTS型、SVC型（声質変換）、ゼロショット型 |
| RVC | 検索ベース声質変換。10分〜のデータで学習。リアルタイム対応 |
| So-VITS-SVC | 歌声変換特化。2時間以上のデータ推奨 |
| OpenVoice | ゼロショット。3秒の参照音声でクローニング |
| データ品質 | クリーン、ドライ、一定音量が最重要 |
| 倫理 | 同意取得、透かし埋め込み、使用目的の明示が必須 |

## 次に読むべきガイド

- [01-voice-assistants.md](./01-voice-assistants.md) — 音声アシスタント実装
- [../00-fundamentals/02-tts-technologies.md](../00-fundamentals/02-tts-technologies.md) — TTS技術基盤
- [../03-development/02-real-time-audio.md](../03-development/02-real-time-audio.md) — リアルタイム音声

## 参考文献

1. Wang, Z., et al. (2023). "VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" — VALL-E論文。3秒の参照音声でのゼロショットTTS
2. Qin, Z., et al. (2024). "OpenVoice: Versatile Instant Voice Cloning" — OpenVoice論文。マルチスタイル・多言語のゼロショットクローニング
3. so-vits-svc contributors (2023). "SoftVC VITS Singing Voice Conversion" — So-VITS-SVC。歌声変換のオープンソースフレームワーク
4. RVC-Project contributors (2023). "Retrieval-based Voice Conversion WebUI" — RVC。検索ベース声質変換の主要実装
