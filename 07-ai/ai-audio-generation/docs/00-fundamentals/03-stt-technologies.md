# STT技術 — Whisper、Google Speech、Azure Speech

> 音声をテキストに変換するSTT（Speech-to-Text）技術の仕組み、主要サービスの比較、実装パターンを解説する

## この章で学ぶこと

1. 現代STTのアーキテクチャ（CTC、Attention、Transducer）と動作原理
2. OpenAI Whisperの仕組み、使い方、ファインチューニング
3. クラウドSTT API（Google、Azure、AWS）の実装と使い分け

---

## 1. STT技術のアーキテクチャ

### 1.1 主要なSTTアーキテクチャ

```
STTの3つのアーキテクチャ
==================================================

1. CTC（Connectionist Temporal Classification）
┌────────┐    ┌──────────┐    ┌─────┐
│メル     │───→│Encoder   │───→│CTC  │───→ テキスト
│スペクト │    │(Conformer)│   │Loss │
│グラム   │    └──────────┘    └─────┘
  * デコーダ不要で高速
  * 条件付き独立仮定（精度に限界）

2. Attention-based Encoder-Decoder
┌────────┐    ┌──────────┐    ┌──────────┐
│メル     │───→│Encoder   │───→│Decoder   │───→ テキスト
│スペクト │    │(Transformer)│  │(自己回帰)│
│グラム   │    └──────────┘    └──────────┘
                    ↕ Attention
  * 高精度（文脈を考慮）
  * 自己回帰のため低速

3. Transducer（RNN-T / Conformer-T）
┌────────┐    ┌──────────┐
│メル     │───→│Encoder   │──┐
│スペクト │    └──────────┘  │ Joint
│グラム   │                  ├──────→ テキスト
              ┌──────────┐  │
              │Prediction│──┘
              │Network   │
              └──────────┘
  * ストリーミング対応
  * CTC + Attentionの良いとこ取り
==================================================
```

### 1.2 Whisperのアーキテクチャ

```
Whisper アーキテクチャ詳細
==================================================

音声入力 (30秒パディング)
    │
    ▼
┌─────────────────────┐
│ メルスペクトログラム   │  80チャネル, 30秒固定
│ (80 x 3000 frames)  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Audio Encoder       │
│  ├─ Conv1D (2層)     │  位置エンコーディング
│  └─ Transformer      │  tiny:4層, base:6層,
│     (Self-Attention)  │  small:12層, medium:24層,
└──────────┬──────────┘  large:32層
           │
           │ Cross-Attention
           ▼
┌─────────────────────┐
│  Text Decoder        │  自己回帰的にトークン生成
│  ├─ Self-Attention   │
│  ├─ Cross-Attention  │  ← Encoder出力を参照
│  └─ FFN              │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Special Tokens      │
│  <|startoftranscript|>│
│  <|ja|>              │  言語タグ
│  <|transcribe|>      │  タスク指定
│  <|notimestamps|>    │  タイムスタンプ制御
└──────────┬──────────┘
           ▼
      テキスト出力
==================================================
```

---

## 2. Whisperの実装

### 2.1 基本的な使い方

```python
import whisper

# モデルのロード
model = whisper.load_model("large-v3")  # tiny, base, small, medium, large-v3

# 基本的な文字起こし
result = model.transcribe(
    "audio.wav",
    language="ja",           # 言語指定（自動検出も可能）
    task="transcribe",       # transcribe or translate
    fp16=True,               # GPU使用時はFP16で高速化
)

print(result["text"])

# セグメント単位の詳細結果
for segment in result["segments"]:
    print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")
    print(f"  信頼度: {segment['avg_logprob']:.3f}")
```

### 2.2 faster-whisper（高速版）

```python
from faster_whisper import WhisperModel

# CTranslate2 による最適化版（2-4倍高速）
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",  # float16, int8_float16, int8
)

# バッチ処理
segments, info = model.transcribe(
    "audio.wav",
    language="ja",
    beam_size=5,
    best_of=5,
    vad_filter=True,         # VADで無音区間をスキップ
    vad_parameters=dict(
        min_silence_duration_ms=500,  # 500ms以上の無音で分割
    ),
)

print(f"検出言語: {info.language} (確率: {info.language_probability:.2f})")

for segment in segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
```

### 2.3 Whisper のファインチューニング

```python
# Hugging Face Transformers によるファインチューニング

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Audio

# モデルとプロセッサのロード
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 日本語用のタスク設定
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="ja", task="transcribe"
)

# データセットの準備
dataset = load_dataset("mozilla-foundation/common_voice_16_1", "ja")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    """音声データの前処理"""
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# トレーニング設定
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-ja-finetuned",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    predict_with_generate=True,
    generation_max_length=225,
)
```

---

## 3. クラウドSTT API

### 3.1 Google Speech-to-Text

```python
from google.cloud import speech_v2 as speech

def google_stt(audio_file: str, language: str = "ja-JP") -> str:
    """Google Cloud Speech-to-Text V2"""
    client = speech.SpeechClient()

    with open(audio_file, "rb") as f:
        audio_content = f.read()

    config = speech.RecognitionConfig(
        auto_decoding_config=speech.AutoDetectDecodingConfig(),
        language_codes=[language],
        model="long",  # long, short, telephony, medical_dictation
        features=speech.RecognitionFeatures(
            enable_automatic_punctuation=True,  # 自動句読点
            enable_word_time_offsets=True,       # 単語タイムスタンプ
            enable_word_confidence=True,         # 単語信頼度
        ),
    )

    request = speech.RecognizeRequest(
        recognizer="projects/my-project/locations/global/recognizers/_",
        config=config,
        content=audio_content,
    )

    response = client.recognize(request=request)

    for result in response.results:
        alt = result.alternatives[0]
        print(f"テキスト: {alt.transcript}")
        print(f"信頼度: {alt.confidence:.3f}")
        for word in alt.words:
            print(f"  {word.word} ({word.start_offset} - {word.end_offset})")

    return response.results[0].alternatives[0].transcript
```

### 3.2 Azure Speech Services

```python
import azure.cognitiveservices.speech as speechsdk

def azure_stt(audio_file: str) -> str:
    """Azure Speech-to-Text"""
    speech_config = speechsdk.SpeechConfig(
        subscription="your-key",
        region="japaneast"
    )
    speech_config.speech_recognition_language = "ja-JP"

    # 詳細設定
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
        "5000"
    )
    speech_config.set_property(
        speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
        "1000"
    )

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # 連続認識（長時間音声向け）
    results = []

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            results.append(evt.result.text)
            print(f"認識: {evt.result.text}")

    def on_canceled(evt):
        print(f"キャンセル: {evt.cancellation_details.reason}")

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)

    recognizer.start_continuous_recognition()

    import time
    time.sleep(30)  # 認識完了を待機（実際はイベントベースで制御）
    recognizer.stop_continuous_recognition()

    return " ".join(results)
```

### 3.3 プロバイダー統合ラッパー

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    segments: list
    provider: str

class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str]) -> TranscriptionResult:
        pass

class UnifiedSTT:
    """複数STTプロバイダーの統合インターフェース"""

    def __init__(self):
        self.providers: dict[str, STTProvider] = {}
        self.fallback_order = ["whisper", "google", "azure"]

    def register(self, name: str, provider: STTProvider):
        self.providers[name] = provider

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "ja",
        provider: Optional[str] = None,
    ) -> TranscriptionResult:
        """文字起こし（フォールバック付き）"""
        if provider:
            return self.providers[provider].transcribe(audio_path, language)

        last_error = None
        for name in self.fallback_order:
            if name not in self.providers:
                continue
            try:
                result = self.providers[name].transcribe(audio_path, language)
                if result.confidence > 0.5:  # 信頼度閾値
                    return result
            except Exception as e:
                last_error = e
                print(f"{name} failed: {e}")
                continue

        raise RuntimeError(f"全プロバイダー失敗: {last_error}")
```

---

## 4. 比較表

### 4.1 主要STTサービス比較

| 項目 | Whisper (local) | Google Speech | Azure Speech | AWS Transcribe | Deepgram |
|------|----------------|---------------|-------------|---------------|----------|
| 日本語WER | 5-8% | 6-10% | 6-9% | 8-12% | 7-10% |
| リアルタイム | 非対応(※) | 対応 | 対応 | 対応 | 対応 |
| ストリーミング | 非対応(※) | 対応 | 対応 | 対応 | 対応 |
| 話者分離 | 非対応 | 対応 | 対応 | 対応 | 対応 |
| コスト | GPU費用のみ | $0.016/分 | $0.016/分 | $0.024/分 | $0.0043/分 |
| オフライン | 可能 | 不可 | 不可 | 不可 | 不可 |
| カスタム語彙 | ファインチューニング | ブースト対応 | カスタム辞書 | カスタム語彙 | キーワード |
| 句読点自動挿入 | 限定的 | 対応 | 対応 | 対応 | 対応 |

※ faster-whisper + VAD で擬似リアルタイムは可能

### 4.2 Whisperモデルサイズ比較

| モデル | パラメータ数 | VRAM | 速度(相対) | 日本語精度 | 推奨用途 |
|--------|------------|------|-----------|-----------|---------|
| tiny | 39M | ~1GB | 32x | 低い | テスト/プロトタイプ |
| base | 74M | ~1GB | 16x | やや低い | 軽量アプリ |
| small | 244M | ~2GB | 6x | 中程度 | バランス型 |
| medium | 769M | ~5GB | 2x | 高い | 品質重視 |
| large-v3 | 1550M | ~10GB | 1x | 最高 | 最高精度 |
| large-v3-turbo | 809M | ~6GB | 4x | 高い | 速度と精度の両立 |

---

## 5. アンチパターン

### 5.1 アンチパターン: VADなしの長時間音声処理

```python
# BAD: 長時間音声をそのまま処理
def bad_transcribe_long(audio_path):
    # 2時間の音声 → メモリ不足 / タイムアウト
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# GOOD: VAD + チャンク分割で処理
from faster_whisper import WhisperModel
import numpy as np

def good_transcribe_long(audio_path, chunk_duration=30):
    """VAD付き長時間音声の文字起こし"""
    model = WhisperModel("large-v3", device="cuda")

    # VADフィルタ付きで処理（自動的に音声区間を検出）
    segments, info = model.transcribe(
        audio_path,
        language="ja",
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
    )

    full_text = []
    for segment in segments:
        full_text.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        })

    return full_text
```

### 5.2 アンチパターン: 信頼度スコアの無視

```python
# BAD: 認識結果をそのまま信用
def bad_process(result):
    return result["text"]  # ハルシネーションが含まれる可能性

# GOOD: 信頼度ベースのフィルタリング
def good_process(segments, confidence_threshold=-0.5):
    """信頼度に基づく品質フィルタリング"""
    filtered = []
    low_confidence = []

    for seg in segments:
        if seg["avg_logprob"] > confidence_threshold:
            filtered.append(seg["text"])
        else:
            # 低信頼度セグメントは要確認としてマーク
            low_confidence.append({
                "time": f"{seg['start']:.1f}-{seg['end']:.1f}s",
                "text": seg["text"],
                "confidence": seg["avg_logprob"],
            })

    if low_confidence:
        print(f"警告: {len(low_confidence)}個の低信頼度セグメントあり")
        for lc in low_confidence:
            print(f"  [{lc['time']}] {lc['text']} (logprob: {lc['confidence']:.3f})")

    return " ".join(filtered), low_confidence
```

---

## 6. FAQ

### Q1: Whisperはリアルタイム音声認識に使えますか？

標準のWhisperは30秒固定の入力を前提としたバッチ処理モデルのため、そのままではリアルタイム認識に不向きです。ただし、faster-whisperとVADを組み合わせた擬似リアルタイム処理や、whisper-streamingプロジェクトによるストリーミング対応は可能です。真のリアルタイム処理が必要な場合は、Google Speech-to-TextやAzure Speechのストリーミング認識APIを使うか、Whisperをストリーミング用にチューニングしたモデル（例: Distil-Whisper）を検討してください。

### Q2: 日本語STTの精度を上げるにはどうすればよいですか？

主な改善策は5つあります。(1) モデルサイズの拡大（large-v3が最高精度）。(2) 音声前処理の改善（ノイズ除去、正規化、リサンプリング）。(3) VADによる無音・非音声区間の除去。(4) 日本語特化データでのファインチューニング（ReazonSpeechデータセット等）。(5) 後処理の追加（句読点挿入、固有名詞補正、LLMによる校正）。特に、ドメイン特化のファインチューニングは専門用語の認識精度を大幅に改善します。

### Q3: 複数話者の音声を区別して文字起こしするには？

話者分離（Speaker Diarization）が必要です。Whisperは単体では話者分離機能を持ちませんが、pyannote-audioと組み合わせることで実現できます。手順は、(1) pyannote-audioで話者分離を実行、(2) 各話者区間ごとにWhisperで文字起こし、(3) タイムスタンプを照合して統合。クラウドAPIを使う場合は、Google Speech-to-TextやAzure Speechに組み込みの話者分離機能があり、設定を有効にするだけで利用できます。

---

## まとめ

| 項目 | 要点 |
|------|------|
| アーキテクチャ | CTC（高速）、Attention（高精度）、Transducer（ストリーミング） |
| Whisper | 汎用性最高のOSSモデル。large-v3が最高精度 |
| faster-whisper | CTranslate2最適化で2-4倍高速。VADフィルタ付き |
| クラウドAPI | リアルタイム・ストリーミングにはGoogle/Azureが優位 |
| 精度改善 | 前処理 + VAD + ファインチューニング + 後処理の4段階 |
| 話者分離 | pyannote-audio + Whisper、またはクラウドAPIの組込機能 |

## 次に読むべきガイド

- [../02-voice/01-voice-assistants.md](../02-voice/01-voice-assistants.md) — 音声アシスタント実装
- [../02-voice/02-podcast-tools.md](../02-voice/02-podcast-tools.md) — ポッドキャスト文字起こし
- [../03-development/02-real-time-audio.md](../03-development/02-real-time-audio.md) — リアルタイム音声処理

## 参考文献

1. Radford, A., et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision" — Whisper論文。680K時間のデータで学習した大規模音声認識モデル
2. Gulati, A., et al. (2020). "Conformer: Convolution-augmented Transformer for Speech Recognition" — Conformer論文。CNN + Transformer の融合アーキテクチャ
3. Graves, A., et al. (2012). "Sequence Transduction with Recurrent Neural Networks" — RNN-T原論文。ストリーミング音声認識の基盤技術
4. Bredin, H., et al. (2023). "pyannote.audio 2.1 speaker diarization pipeline" — pyannote-audio論文。話者分離の代表的フレームワーク
