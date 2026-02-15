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

### 1.2 CTCアーキテクチャの詳細

CTC（Connectionist Temporal Classification）は、入力と出力のアライメントを明示的に求めず、全ての可能なアライメントの合計確率を最大化するアプローチである。

```python
# CTCベースのSTTモデル概念実装
import torch
import torch.nn as nn
import torchaudio

class CTCModel(nn.Module):
    """CTC損失を使用した音声認識モデル"""

    def __init__(
        self,
        input_dim: int = 80,      # メルスペクトログラムのビン数
        hidden_dim: int = 512,
        num_layers: int = 6,
        vocab_size: int = 5000,    # サブワード語彙サイズ
        dropout: float = 0.1,
    ):
        super().__init__()

        # 特徴量前処理: Conv2Dサブサンプリング
        self.conv_subsample = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Conformerエンコーダ
        conv_out_dim = 32 * (input_dim // 4)
        self.linear_in = nn.Linear(conv_out_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 出力層（blank token含む）
        self.output_proj = nn.Linear(hidden_dim, vocab_size + 1)  # +1 for blank

    def forward(self, x, x_lengths):
        """
        x: (batch, time, mel_bins) メルスペクトログラム
        x_lengths: (batch,) 各入力の長さ
        """
        # (B, T, F) -> (B, 1, T, F) -> Conv2D
        x = x.unsqueeze(1)
        x = self.conv_subsample(x)

        # (B, C, T/4, F/4) -> (B, T/4, C*F/4)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)

        x = self.linear_in(x)
        x = self.encoder(x)
        logits = self.output_proj(x)

        # CTC用にlog_softmax
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs

    def decode_greedy(self, log_probs):
        """グリーディデコード: 最も確率の高いトークンを選択"""
        predictions = torch.argmax(log_probs, dim=-1)

        # blank除去と連続重複除去
        decoded = []
        prev = -1
        for token in predictions[0]:
            token = token.item()
            if token != 0 and token != prev:  # 0 = blank
                decoded.append(token)
            prev = token

        return decoded
```

CTCの主な特徴は以下の通り:

- **利点**: デコーダが不要なため推論が高速、ストリーミング処理との相性が良い
- **限界**: 条件付き独立仮定により、出力トークン間の依存関係をモデル化できない
- **改善策**: 外部言語モデルとの組み合わせ、CTC+Attentionハイブリッド

### 1.3 Transducerアーキテクチャの詳細

Transducer（特にRNN-T / Conformer-T）は、ストリーミング音声認識の主流アーキテクチャである。

```python
# RNN-Transducerの概念構造
class RNNTransducer(nn.Module):
    """RNN-T モデルの簡略実装"""

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        decoder_dim: int = 256,
        joint_dim: int = 512,
        vocab_size: int = 5000,
    ):
        super().__init__()

        # Encoder: 音響特徴量を処理（Conformerベース）
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=encoder_dim,
            num_layers=6,
            batch_first=True,
            bidirectional=False,  # ストリーミングのため単方向
        )

        # Prediction Network: 過去の出力トークンを処理
        self.prediction = nn.LSTM(
            input_size=vocab_size,
            hidden_size=decoder_dim,
            num_layers=2,
            batch_first=True,
        )

        # Joint Network: EncoderとPredictionの出力を結合
        self.joint_enc = nn.Linear(encoder_dim, joint_dim)
        self.joint_pred = nn.Linear(decoder_dim, joint_dim)
        self.joint_out = nn.Linear(joint_dim, vocab_size + 1)  # +blank

    def forward(self, audio_features, prev_tokens):
        """
        audio_features: (B, T, input_dim)
        prev_tokens: (B, U, vocab_size) one-hot
        """
        enc_out, _ = self.encoder(audio_features)         # (B, T, enc_dim)
        pred_out, _ = self.prediction(prev_tokens)        # (B, U, dec_dim)

        # Joint: (B, T, 1, joint_dim) + (B, 1, U, joint_dim)
        enc_proj = self.joint_enc(enc_out).unsqueeze(2)   # (B, T, 1, J)
        pred_proj = self.joint_pred(pred_out).unsqueeze(1) # (B, 1, U, J)

        joint = torch.tanh(enc_proj + pred_proj)          # (B, T, U, J)
        logits = self.joint_out(joint)                    # (B, T, U, V+1)

        return logits
```

Transducerの重要な設計ポイント:

| 要素 | 説明 | 設計上の考慮点 |
|------|------|--------------|
| Encoder | 音響特徴量の処理 | 単方向LSTM/Conformerでストリーミング対応 |
| Prediction Network | 言語モデルの役割 | 小さめのLSTMで十分（軽量化） |
| Joint Network | 結合判定 | ブランク vs 出力の判定がボトルネック |
| ビームサーチ | デコード | ビーム幅4-10が精度と速度のバランス |

### 1.4 Whisperのアーキテクチャ

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

### 1.5 Whisper の学習データと多言語対応

Whisperは680,000時間のインターネット音声データで学習されている。この大規模弱教師あり学習が高い汎化性能の源泉となっている。

```
Whisper 学習データ構成
==================================================

総量: 680,000 時間の音声データ

言語分布:
  英語           : ████████████████████ 65%
  ヨーロッパ言語 : ██████ 15%
  アジア言語     : ████ 10%
    - 日本語     : 約7,000時間（推定）
    - 中国語     : 約12,000時間（推定）
    - 韓国語     : 約3,000時間（推定）
  その他         : ███ 10%

タスク分布:
  文字起こし     : 75%（音声→同一言語テキスト）
  翻訳           : 25%（音声→英語テキスト）

データソース:
  - インターネット上の字幕付き動画
  - ポッドキャスト + 書き起こし
  - オーディオブック + テキスト
  ※ 弱教師あり = 完全なアライメントなしで学習
==================================================
```

### 1.6 アーキテクチャの選択指針

```
STTアーキテクチャ 選択フローチャート
==================================================

               ストリーミングが必要？
              /                    \
          Yes                       No
           |                        |
     リアルタイム性            最高精度が必要？
     最優先？               /            \
    /        \           Yes              No
 Yes          No          |               |
  |            |     Attention-based    CTC + 言語モデル
Transducer  Transducer  Encoder-Decoder  (コスト重視)
(Conformer-T) (RNN-T)   (Whisper等)
  |            |          |               |
遅延<100ms  遅延<300ms  バッチ処理     低コスト処理
on-device   サーバー    最高精度      組込みデバイス

代表的な実装:
  Transducer : Google USM, Conformer-Transducer
  Attention  : Whisper, Canary (NVIDIA NeMo)
  CTC        : wav2vec 2.0, HuBERT
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

### 2.2 Whisperの詳細オプション

```python
import whisper
import numpy as np

# 詳細なパラメータ設定
model = whisper.load_model("large-v3")

result = model.transcribe(
    "audio.wav",
    language="ja",
    task="transcribe",

    # デコーディングオプション
    temperature=0.0,            # 0.0 = グリーディ、>0 = サンプリング
    best_of=5,                  # temperature > 0 のとき複数候補から最善を選択
    beam_size=5,                # ビームサーチのビーム数
    patience=1.0,               # ビームサーチの忍耐度
    length_penalty=None,        # 長さペナルティ（Noneで無効）

    # 前処理オプション
    fp16=True,                  # FP16で推論（GPU必須）
    no_speech_threshold=0.6,    # 無音セグメントの閾値
    logprob_threshold=-1.0,     # 低信頼度セグメントの閾値
    compression_ratio_threshold=2.4,  # 繰り返し検出の閾値

    # 出力制御
    word_timestamps=True,       # 単語レベルのタイムスタンプ
    prepend_punctuations="\"'"¿([{-",  # 前置句読点
    append_punctuations="\"'.。,，!！?？:：\"')]}、",  # 後置句読点

    # 初期プロンプト（コンテキスト提供）
    initial_prompt="以下は会議の議事録です。参加者は田中、佐藤、鈴木の3名です。",

    # 条件付きテキスト（特定フォーマットの強制）
    condition_on_previous_text=True,  # 前セグメントのテキストを条件に
)

# 単語レベルのタイムスタンプ
for segment in result["segments"]:
    if "words" in segment:
        for word in segment["words"]:
            print(f"  [{word['start']:.2f}s - {word['end']:.2f}s] {word['word']}")
```

### 2.3 faster-whisper（高速版）

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

### 2.4 faster-whisper の高度な設定

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline
import numpy as np

def advanced_faster_whisper_transcription(audio_path: str) -> dict:
    """
    faster-whisperの高度な設定を使った文字起こし
    - VADフィルタリング
    - バッチ推論
    - 詳細パラメータ調整
    """
    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
        cpu_threads=4,           # CPUスレッド数
        num_workers=2,           # データローダワーカー数
    )

    # 高度なVAD設定
    vad_params = {
        "threshold": 0.5,                  # VAD確率閾値
        "min_speech_duration_ms": 250,     # 最小発話持続時間
        "max_speech_duration_s": 30,       # 最大発話持続時間（Whisperの上限に合わせる）
        "min_silence_duration_ms": 500,    # 無音区間の最小持続時間
        "speech_pad_ms": 200,              # 発話前後のパディング
    }

    segments, info = model.transcribe(
        audio_path,
        language="ja",
        beam_size=5,
        best_of=5,
        patience=1.5,
        length_penalty=1.0,
        repetition_penalty=1.2,    # 繰り返し抑制
        no_repeat_ngram_size=3,    # 3-gram繰り返し禁止
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 段階的温度
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=vad_params,
        initial_prompt="以下は日本語の音声です。",
    )

    results = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": [],
    }

    for segment in segments:
        seg_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob,
            "compression_ratio": segment.compression_ratio,
        }

        if segment.words:
            seg_data["words"] = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                }
                for w in segment.words
            ]

        results["segments"].append(seg_data)

    return results
```

### 2.5 Whisper のファインチューニング

```python
# Hugging Face Transformers によるファインチューニング

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Audio
import evaluate
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

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

# データセットの前処理を適用
dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names["train"],
    num_proc=4,
)

# カスタムデータコレーター
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# 評価メトリクス
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

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
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
)

# トレーナーの初期化と学習
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
```

### 2.6 日本語特化のファインチューニングデータセット

```python
# ReazonSpeechを使った日本語ファインチューニング

from datasets import load_dataset, Audio
from transformers import WhisperProcessor

def prepare_reazon_speech_dataset():
    """
    ReazonSpeech: 日本語特化の大規模音声データセット
    - 約19,000時間の日本語音声
    - NHKニュースの読み上げ音声
    - 高品質なアライメント
    """
    # ReazonSpeechデータセットのロード
    dataset = load_dataset(
        "reazon-research/reazonspeech",
        "all",
        trust_remote_code=True,
    )

    # サンプリングレートの統一
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    def preprocess(batch):
        audio = batch["audio"]
        # Whisper用の特徴量抽出
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=16000,
        ).input_features[0]
        # テキストのトークン化
        batch["labels"] = processor.tokenizer(
            batch["transcription"]
        ).input_ids
        return batch

    dataset = dataset.map(preprocess, num_proc=8)
    return dataset

def prepare_jsut_dataset():
    """
    JSUT (Japanese Speech corpus of Saruwatari-lab, University of Tokyo)
    - 約10時間の高品質日本語音声
    - 1名の女性話者
    - ファインチューニングの小規模実験に最適
    """
    dataset = load_dataset("esb/jsut", trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

def prepare_common_voice_ja():
    """
    Common Voice 日本語
    - Mozilla による多話者日本語データセット
    - 多様な話者によるクラウドソーシングデータ
    - バリデーション済みの高品質サブセットあり
    """
    dataset = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        "ja",
        trust_remote_code=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # バリデーション済みのデータのみフィルタリング
    dataset = dataset.filter(lambda x: x["up_votes"] >= 2 and x["down_votes"] == 0)

    return dataset
```

### 2.7 Whisperの後処理パイプライン

```python
import re
from typing import Optional

class WhisperPostProcessor:
    """Whisper出力の後処理パイプライン"""

    def __init__(self):
        self.custom_dict = {}  # カスタム辞書（固有名詞等）

    def add_custom_words(self, word_map: dict):
        """カスタム辞書の追加（誤認識の修正用）"""
        self.custom_dict.update(word_map)

    def process(self, text: str) -> str:
        """全後処理ステップを順次実行"""
        text = self._remove_hallucinations(text)
        text = self._fix_punctuation(text)
        text = self._apply_custom_dict(text)
        text = self._normalize_numbers(text)
        text = self._remove_filler_words(text)
        return text.strip()

    def _remove_hallucinations(self, text: str) -> str:
        """Whisperのハルシネーション（幻覚）パターンを除去"""
        # 繰り返しパターンの検出と除去
        # 例: "ありがとうございますありがとうございますありがとうございます"
        for length in range(5, 50):
            pattern = r'(.{' + str(length) + r',})\1{2,}'
            text = re.sub(pattern, r'\1', text)

        # 典型的なハルシネーションフレーズ
        hallucination_patterns = [
            r'ご視聴ありがとうございました。?$',
            r'チャンネル登録.*お願いします。?$',
            r'お疲れ様でした。?$',
            r'(?:\.{3,})',  # 連続するピリオド
            r'(?:。{2,})',  # 連続する句点
        ]
        for pattern in hallucination_patterns:
            text = re.sub(pattern, '', text)

        return text

    def _fix_punctuation(self, text: str) -> str:
        """句読点の修正"""
        # 半角句読点を全角に統一
        text = text.replace(',', '、')
        text = text.replace('.', '。')
        text = text.replace('!', '！')
        text = text.replace('?', '？')

        # 連続する句読点を1つに
        text = re.sub(r'[、。]{2,}', '。', text)

        # 文末に句点がない場合に追加
        if text and text[-1] not in '。！？':
            text += '。'

        return text

    def _apply_custom_dict(self, text: str) -> str:
        """カスタム辞書による置換"""
        for wrong, correct in self.custom_dict.items():
            text = text.replace(wrong, correct)
        return text

    def _normalize_numbers(self, text: str) -> str:
        """数値表現の正規化"""
        # 全角数字を半角に変換
        zen_to_han = str.maketrans('０１２３４５６７８９', '0123456789')
        text = text.translate(zen_to_han)
        return text

    def _remove_filler_words(self, text: str) -> str:
        """フィラーワード（えー、あの）の除去"""
        fillers = [
            r'えー[、と]?\s*',
            r'あのー?\s*',
            r'まあ[、]?\s*',
            r'その[、]?\s*(?=\S)',
            r'ええと[、]?\s*',
        ]
        for filler in fillers:
            text = re.sub(filler, '', text)
        return text


# 使用例
post_processor = WhisperPostProcessor()
post_processor.add_custom_words({
    "ファスターウィスパー": "faster-whisper",
    "パイトーチ": "PyTorch",
    "テンサーフロー": "TensorFlow",
    "ギットハブ": "GitHub",
})

raw_text = "えー、本日はファスターウィスパーについて、あの、説明します。。。"
cleaned = post_processor.process(raw_text)
print(cleaned)
# → "本日はfaster-whisperについて説明します。"
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

### 3.2 Google Speech-to-Text ストリーミング認識

```python
from google.cloud import speech_v1
import pyaudio
import queue
import threading

class GoogleStreamingSTT:
    """Google Cloud STTのストリーミング認識実装"""

    def __init__(
        self,
        language: str = "ja-JP",
        sample_rate: int = 16000,
        model: str = "latest_long",
    ):
        self.client = speech_v1.SpeechClient()
        self.config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language,
            model=model,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            # 話者分離を有効化
            diarization_config=speech_v1.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=2,
                max_speaker_count=4,
            ),
            # カスタム語彙のブースト
            speech_contexts=[
                speech_v1.SpeechContext(
                    phrases=["Whisper", "STT", "TTS", "API"],
                    boost=15.0,
                ),
            ],
        )
        self.streaming_config = speech_v1.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True,
            single_utterance=False,  # 複数発話を継続認識
        )
        self.audio_queue = queue.Queue()
        self.sample_rate = sample_rate

    def start_microphone_stream(self, on_result, duration_seconds=60):
        """マイク入力からストリーミング認識"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1600,  # 100msチャンク
            stream_callback=lambda in_data, *args: (
                self.audio_queue.put(in_data),
                (None, pyaudio.paContinue),
            )[-1],
        )

        def request_generator():
            yield speech_v1.StreamingRecognizeRequest(
                streaming_config=self.streaming_config
            )
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    return
                yield speech_v1.StreamingRecognizeRequest(audio_content=chunk)

        responses = self.client.streaming_recognize(requests=request_generator())

        for response in responses:
            for result in response.results:
                alt = result.alternatives[0]
                on_result({
                    "is_final": result.is_final,
                    "transcript": alt.transcript,
                    "confidence": alt.confidence if result.is_final else None,
                    "stability": result.stability,
                })

        stream.stop_stream()
        stream.close()
        p.terminate()
```

### 3.3 Azure Speech Services

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

### 3.4 Azure Speech Services 高度な機能

```python
import azure.cognitiveservices.speech as speechsdk
import json

class AzureAdvancedSTT:
    """Azure Speech Servicesの高度な機能を活用したSTT"""

    def __init__(self, subscription_key: str, region: str = "japaneast"):
        self.speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key,
            region=region,
        )
        self.speech_config.speech_recognition_language = "ja-JP"

    def transcribe_with_diarization(self, audio_file: str) -> list[dict]:
        """話者分離付き文字起こし"""
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

        conversation_transcriber = speechsdk.ConversationTranscriber(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )

        results = []
        done = False

        def on_transcribed(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                results.append({
                    "speaker_id": evt.result.speaker_id,
                    "text": evt.result.text,
                    "offset_ms": evt.result.offset / 10000,  # 100nsから変換
                    "duration_ms": evt.result.duration / 10000,
                })

        def on_stopped(evt):
            nonlocal done
            done = True

        conversation_transcriber.transcribed.connect(on_transcribed)
        conversation_transcriber.session_stopped.connect(on_stopped)
        conversation_transcriber.canceled.connect(on_stopped)

        conversation_transcriber.start_transcribing_async().get()

        import time
        while not done:
            time.sleep(0.5)

        conversation_transcriber.stop_transcribing_async().get()
        return results

    def transcribe_with_pronunciation_assessment(
        self, audio_file: str, reference_text: str
    ) -> dict:
        """発音評価付き文字起こし（語学学習向け）"""
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True,
        )

        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )

        pronunciation_config.apply_to(recognizer)

        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            assessment = speechsdk.PronunciationAssessmentResult(result)
            return {
                "text": result.text,
                "accuracy_score": assessment.accuracy_score,
                "pronunciation_score": assessment.pronunciation_score,
                "completeness_score": assessment.completeness_score,
                "fluency_score": assessment.fluency_score,
                "words": [
                    {
                        "word": w.word,
                        "accuracy_score": w.accuracy_score,
                        "error_type": w.error_type,
                    }
                    for w in assessment.words
                ],
            }
        return {"error": str(result.reason)}

    def transcribe_with_keyword_spotting(
        self, audio_file: str, keywords: list[str]
    ) -> dict:
        """キーワードスポッティング付き文字起こし"""
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "10000"
        )

        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

        # カスタムキーワードモデルの作成（簡略版）
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )

        # 語句リストを追加
        phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
        for keyword in keywords:
            phrase_list.addPhrase(keyword)

        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            # キーワードの出現をチェック
            found_keywords = [
                kw for kw in keywords if kw in result.text
            ]
            return {
                "text": result.text,
                "found_keywords": found_keywords,
                "confidence": result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                ),
            }
        return {"error": str(result.reason)}
```

### 3.5 AWS Transcribe の実装

```python
import boto3
import json
import time
from urllib.request import urlopen

class AWSTranscribeSTT:
    """AWS Transcribe による文字起こし"""

    def __init__(self, region: str = "ap-northeast-1"):
        self.client = boto3.client("transcribe", region_name=region)
        self.s3_client = boto3.client("s3", region_name=region)

    def transcribe_file(
        self,
        s3_uri: str,
        job_name: str,
        language: str = "ja-JP",
        speaker_count: int = 2,
    ) -> dict:
        """S3上の音声ファイルを文字起こし"""
        self.client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat="wav",
            LanguageCode=language,
            Settings={
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": speaker_count,
                "ShowAlternatives": True,
                "MaxAlternatives": 3,
                "VocabularyName": "my-custom-vocab",  # カスタム語彙
            },
        )

        # ジョブ完了を待機
        while True:
            status = self.client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]

            if job_status in ["COMPLETED", "FAILED"]:
                break
            time.sleep(5)

        if job_status == "COMPLETED":
            result_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            result = json.loads(urlopen(result_url).read())
            return self._parse_result(result)
        else:
            raise RuntimeError(f"文字起こしジョブ失敗: {job_status}")

    def _parse_result(self, result: dict) -> dict:
        """AWS Transcribe結果のパース"""
        items = result["results"]["items"]
        segments = []
        current_speaker = None

        for item in items:
            speaker = item.get("speaker_label", current_speaker)
            if speaker != current_speaker:
                segments.append({
                    "speaker": speaker,
                    "text": "",
                    "start": float(item.get("start_time", 0)),
                })
                current_speaker = speaker

            if item["type"] == "pronunciation":
                segments[-1]["text"] += item["alternatives"][0]["content"]
                segments[-1]["end"] = float(item.get("end_time", 0))
            elif item["type"] == "punctuation":
                segments[-1]["text"] += item["alternatives"][0]["content"]

        return {
            "full_text": result["results"]["transcripts"][0]["transcript"],
            "segments": segments,
        }

    def create_custom_vocabulary(
        self,
        vocab_name: str,
        phrases: list[dict],
        language: str = "ja-JP",
    ):
        """カスタム語彙の作成"""
        self.client.create_vocabulary(
            VocabularyName=vocab_name,
            LanguageCode=language,
            Phrases=[
                {
                    "Phrase": p["phrase"],
                    "IPA": p.get("ipa"),
                    "DisplayAs": p.get("display_as", p["phrase"]),
                }
                for p in phrases
            ],
        )
```

### 3.6 Deepgram の実装

```python
from deepgram import DeepgramClient, PrerecordedOptions, LiveOptions
import asyncio

class DeepgramSTT:
    """Deepgramによる高速文字起こし"""

    def __init__(self, api_key: str):
        self.client = DeepgramClient(api_key)

    def transcribe_file(self, audio_path: str) -> dict:
        """ファイルの文字起こし"""
        with open(audio_path, "rb") as f:
            buffer_data = f.read()

        payload = {"buffer": buffer_data}

        options = PrerecordedOptions(
            model="nova-2",           # 最新モデル
            language="ja",
            smart_format=True,        # 自動フォーマット
            punctuate=True,           # 句読点挿入
            diarize=True,             # 話者分離
            utterances=True,          # 発話単位分割
            detect_language=True,     # 言語自動検出
            paragraphs=True,          # 段落分割
            summarize="v2",           # 要約生成
            topics=True,              # トピック抽出
            intents=True,             # 意図分析
            sentiment=True,           # 感情分析
        )

        response = self.client.listen.prerecorded.v("1").transcribe_file(
            payload, options
        )

        result = response.to_dict()

        return {
            "transcript": result["results"]["channels"][0]["alternatives"][0]["transcript"],
            "confidence": result["results"]["channels"][0]["alternatives"][0]["confidence"],
            "words": result["results"]["channels"][0]["alternatives"][0]["words"],
            "paragraphs": result["results"]["channels"][0]["alternatives"][0].get("paragraphs"),
            "summaries": result["results"].get("summary"),
            "topics": result["results"].get("topics"),
            "sentiments": result["results"].get("sentiments"),
        }

    async def transcribe_stream(self, audio_stream, on_result):
        """ストリーミング文字起こし"""
        options = LiveOptions(
            model="nova-2",
            language="ja",
            punctuate=True,
            interim_results=True,
            utterance_end_ms=1000,
            vad_events=True,
            smart_format=True,
        )

        connection = self.client.listen.live.v("1")

        async def on_message(self_conn, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if transcript:
                on_result({
                    "text": transcript,
                    "is_final": result.is_final,
                    "speech_final": result.speech_final,
                })

        connection.on("Results", on_message)

        await connection.start(options)

        async for chunk in audio_stream:
            connection.send(chunk)

        await connection.finish()
```

### 3.7 プロバイダー統合ラッパー

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time
import hashlib

@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    segments: list
    provider: str
    latency_ms: float = 0.0
    word_count: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.word_count = len(self.text.split())

class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str]) -> TranscriptionResult:
        pass

class UnifiedSTT:
    """複数STTプロバイダーの統合インターフェース"""

    def __init__(self):
        self.providers: dict[str, STTProvider] = {}
        self.fallback_order = ["whisper", "google", "azure"]
        self._cache: dict[str, TranscriptionResult] = {}
        self._metrics: dict[str, dict] = {}

    def register(self, name: str, provider: STTProvider):
        self.providers[name] = provider
        self._metrics[name] = {
            "success": 0,
            "failure": 0,
            "total_latency_ms": 0.0,
        }

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "ja",
        provider: Optional[str] = None,
        use_cache: bool = True,
    ) -> TranscriptionResult:
        """文字起こし（フォールバック付き）"""
        # キャッシュチェック
        cache_key = self._make_cache_key(audio_path, language)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if provider:
            result = self._transcribe_single(provider, audio_path, language)
            if use_cache:
                self._cache[cache_key] = result
            return result

        last_error = None
        for name in self.fallback_order:
            if name not in self.providers:
                continue
            try:
                result = self._transcribe_single(name, audio_path, language)
                if result.confidence > 0.5:  # 信頼度閾値
                    if use_cache:
                        self._cache[cache_key] = result
                    return result
            except Exception as e:
                last_error = e
                print(f"{name} failed: {e}")
                continue

        raise RuntimeError(f"全プロバイダー失敗: {last_error}")

    def _transcribe_single(
        self, name: str, audio_path: str, language: Optional[str]
    ) -> TranscriptionResult:
        """単一プロバイダーで文字起こし（メトリクス計測付き）"""
        start = time.time()
        try:
            result = self.providers[name].transcribe(audio_path, language)
            latency = (time.time() - start) * 1000
            result.latency_ms = latency
            self._metrics[name]["success"] += 1
            self._metrics[name]["total_latency_ms"] += latency
            return result
        except Exception as e:
            self._metrics[name]["failure"] += 1
            raise

    def _make_cache_key(self, audio_path: str, language: Optional[str]) -> str:
        """キャッシュキーの生成"""
        with open(audio_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        return f"{file_hash}:{language}"

    def get_metrics(self) -> dict:
        """各プロバイダーのメトリクスを取得"""
        metrics = {}
        for name, m in self._metrics.items():
            total = m["success"] + m["failure"]
            metrics[name] = {
                "total_requests": total,
                "success_rate": m["success"] / total if total > 0 else 0,
                "avg_latency_ms": (
                    m["total_latency_ms"] / m["success"]
                    if m["success"] > 0 else 0
                ),
            }
        return metrics
```

---

## 4. 話者分離（Speaker Diarization）

### 4.1 pyannote-audio を使った話者分離

```python
from pyannote.audio import Pipeline
import torch

class SpeakerDiarizer:
    """pyannote-audioを使った話者分離"""

    def __init__(self, auth_token: str):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token,
        )
        # GPU使用
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    def diarize(
        self,
        audio_path: str,
        min_speakers: int = 2,
        max_speakers: int = 5,
    ) -> list[dict]:
        """話者分離を実行"""
        diarization = self.pipeline(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start,
            })

        return segments

    def diarize_and_transcribe(
        self,
        audio_path: str,
        whisper_model,
        min_speakers: int = 2,
        max_speakers: int = 5,
    ) -> list[dict]:
        """話者分離 + Whisper文字起こしの統合"""
        import librosa
        import numpy as np

        # Step 1: 話者分離
        diarization_result = self.diarize(
            audio_path, min_speakers, max_speakers
        )

        # Step 2: 音声の読み込み
        audio, sr = librosa.load(audio_path, sr=16000)

        # Step 3: 各話者区間ごとにWhisperで文字起こし
        results = []
        for segment in diarization_result:
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            segment_audio = audio[start_sample:end_sample]

            # 短すぎるセグメントはスキップ
            if len(segment_audio) / sr < 0.5:
                continue

            # 一時ファイルに書き出して文字起こし
            import soundfile as sf
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, segment_audio, sr)
                result = whisper_model.transcribe(
                    f.name,
                    language="ja",
                    fp16=torch.cuda.is_available(),
                )

            results.append({
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"],
                "text": result["text"].strip(),
            })

        return results
```

### 4.2 話者分離結果のフォーマット出力

```python
def format_diarized_transcript(
    segments: list[dict],
    format_type: str = "text",
) -> str:
    """話者分離結果をフォーマットして出力"""

    if format_type == "text":
        lines = []
        current_speaker = None
        for seg in segments:
            if seg["speaker"] != current_speaker:
                current_speaker = seg["speaker"]
                lines.append(f"\n[{current_speaker}]")
            timestamp = f"({seg['start']:.1f}s - {seg['end']:.1f}s)"
            lines.append(f"  {timestamp} {seg['text']}")
        return "\n".join(lines)

    elif format_type == "srt":
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start = _format_srt_time(seg["start"])
            end = _format_srt_time(seg["end"])
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(f"[{seg['speaker']}] {seg['text']}")
            srt_lines.append("")
        return "\n".join(srt_lines)

    elif format_type == "json":
        import json
        return json.dumps(segments, ensure_ascii=False, indent=2)

    elif format_type == "vtt":
        vtt_lines = ["WEBVTT", ""]
        for seg in segments:
            start = _format_vtt_time(seg["start"])
            end = _format_vtt_time(seg["end"])
            vtt_lines.append(f"{start} --> {end}")
            vtt_lines.append(f"<v {seg['speaker']}>{seg['text']}</v>")
            vtt_lines.append("")
        return "\n".join(vtt_lines)

    raise ValueError(f"未対応のフォーマット: {format_type}")


def _format_srt_time(seconds: float) -> str:
    """SRT形式のタイムスタンプに変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_time(seconds: float) -> str:
    """VTT形式のタイムスタンプに変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
```

---

## 5. 精度評価と改善

### 5.1 WER/CER の計算

```python
import evaluate

class STTEvaluator:
    """STTの精度評価ツール"""

    def __init__(self):
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict:
        """WERとCERを計算"""
        wer = self.wer_metric.compute(
            predictions=predictions,
            references=references,
        )
        cer = self.cer_metric.compute(
            predictions=predictions,
            references=references,
        )

        # 詳細分析
        analysis = self._analyze_errors(predictions, references)

        return {
            "wer": wer,
            "cer": cer,
            "num_samples": len(predictions),
            "error_analysis": analysis,
        }

    def _analyze_errors(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict:
        """エラーパターンの分析"""
        substitutions = 0
        insertions = 0
        deletions = 0

        for pred, ref in zip(predictions, references):
            pred_chars = list(pred)
            ref_chars = list(ref)

            # 編集距離のバックトレースからエラータイプを分類
            n, m = len(ref_chars), len(pred_chars)
            dp = [[0] * (m + 1) for _ in range(n + 1)]

            for i in range(n + 1):
                dp[i][0] = i
            for j in range(m + 1):
                dp[0][j] = j

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    if ref_chars[i-1] == pred_chars[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(
                            dp[i-1][j],    # 削除
                            dp[i][j-1],    # 挿入
                            dp[i-1][j-1],  # 置換
                        )

            # バックトレース
            i, j = n, m
            while i > 0 or j > 0:
                if i > 0 and j > 0 and ref_chars[i-1] == pred_chars[j-1]:
                    i -= 1
                    j -= 1
                elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                    substitutions += 1
                    i -= 1
                    j -= 1
                elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                    insertions += 1
                    j -= 1
                elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                    deletions += 1
                    i -= 1

        total_errors = substitutions + insertions + deletions
        return {
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "total_errors": total_errors,
            "error_distribution": {
                "substitution_rate": substitutions / total_errors if total_errors > 0 else 0,
                "insertion_rate": insertions / total_errors if total_errors > 0 else 0,
                "deletion_rate": deletions / total_errors if total_errors > 0 else 0,
            },
        }

# 使用例
evaluator = STTEvaluator()
results = evaluator.evaluate(
    predictions=["こんにちわ世界"],
    references=["こんにちは世界"],
)
print(f"WER: {results['wer']:.4f}")
print(f"CER: {results['cer']:.4f}")
```

### 5.2 精度改善テクニック

```python
# 音声前処理による精度改善
import librosa
import numpy as np
import noisereduce as nr

class STTPreprocessor:
    """STT精度改善のための前処理パイプライン"""

    def preprocess(self, audio_path: str) -> np.ndarray:
        """包括的な前処理を実行"""
        # 1. 音声読み込み（16kHzにリサンプリング）
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 2. 無音トリミング
        audio = self._trim_silence(audio, sr)

        # 3. ノイズ除去
        audio = self._denoise(audio, sr)

        # 4. 正規化
        audio = self._normalize(audio)

        # 5. 音声区間のみ抽出
        audio = self._extract_speech(audio, sr)

        return audio

    def _trim_silence(
        self, audio: np.ndarray, sr: int, top_db: float = 30
    ) -> np.ndarray:
        """前後の無音をトリミング"""
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

    def _denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """スペクトル減算によるノイズ除去"""
        # noisereduceライブラリを使用
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,   # 定常ノイズ仮定
            prop_decrease=0.75, # ノイズ削減率（0.5-1.0）
        )
        return reduced

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """ピーク正規化"""
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        return audio

    def _extract_speech(
        self, audio: np.ndarray, sr: int
    ) -> np.ndarray:
        """Silero VADで音声区間のみ抽出"""
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        get_speech_timestamps = utils[0]

        # 音声をtensorに変換
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # 音声区間を検出
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sr,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )

        # 音声区間のみ結合
        if speech_timestamps:
            speech_segments = []
            for ts in speech_timestamps:
                speech_segments.append(audio[ts["start"]:ts["end"]])
            return np.concatenate(speech_segments)
        return audio
```

---

## 6. 比較表

### 6.1 主要STTサービス比較

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
| 感情分析 | 非対応 | 非対応 | 非対応 | 非対応 | 対応 |
| 要約生成 | 非対応 | 非対応 | 非対応 | 非対応 | 対応 |
| 発音評価 | 非対応 | 非対応 | 対応 | 非対応 | 非対応 |

※ faster-whisper + VAD で擬似リアルタイムは可能

### 6.2 Whisperモデルサイズ比較

| モデル | パラメータ数 | VRAM | 速度(相対) | 日本語精度 | 推奨用途 |
|--------|------------|------|-----------|-----------|---------|
| tiny | 39M | ~1GB | 32x | 低い | テスト/プロトタイプ |
| base | 74M | ~1GB | 16x | やや低い | 軽量アプリ |
| small | 244M | ~2GB | 6x | 中程度 | バランス型 |
| medium | 769M | ~5GB | 2x | 高い | 品質重視 |
| large-v3 | 1550M | ~10GB | 1x | 最高 | 最高精度 |
| large-v3-turbo | 809M | ~6GB | 4x | 高い | 速度と精度の両立 |

### 6.3 STTユースケース別推奨構成

| ユースケース | 推奨プロバイダー | 構成 | 理由 |
|------------|----------------|------|------|
| 会議議事録 | Azure Speech | ストリーミング + 話者分離 | 話者分離精度が高い |
| ポッドキャスト文字起こし | Whisper large-v3 | バッチ処理 + 後処理 | 最高精度、コスト効率 |
| コールセンター | Google STT | ストリーミング + カスタム語彙 | 低遅延、用語ブースト |
| 医療音声記録 | Azure Speech | カスタムモデル + 話者分離 | 医療用語対応 |
| リアルタイム字幕 | Deepgram | WebSocket + nova-2 | 最低遅延 |
| 多言語対応 | Whisper API | バッチ処理 | 97言語対応 |
| オフライン処理 | faster-whisper | ローカルGPU | ネットワーク不要 |
| 語学学習 | Azure Speech | 発音評価機能 | 発音スコアリング |

---

## 7. アンチパターン

### 7.1 アンチパターン: VADなしの長時間音声処理

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

### 7.2 アンチパターン: 信頼度スコアの無視

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

### 7.3 アンチパターン: 前処理なしでの直接認識

```python
# BAD: 生の音声をそのままSTTに入力
def bad_raw_transcribe(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]
    # → ノイズ混入、不適切なサンプリングレートで精度低下

# GOOD: 適切な前処理パイプライン
def good_preprocessed_transcribe(audio_path):
    """前処理パイプラインを通してから文字起こし"""
    preprocessor = STTPreprocessor()

    # 1. 前処理
    processed_audio = preprocessor.preprocess(audio_path)

    # 2. 一時ファイルに保存
    import soundfile as sf
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, processed_audio, 16000)
        temp_path = f.name

    # 3. 文字起こし
    result = model.transcribe(temp_path, language="ja")

    # 4. 後処理
    post_processor = WhisperPostProcessor()
    cleaned_text = post_processor.process(result["text"])

    return cleaned_text
```

### 7.4 アンチパターン: 単一プロバイダーへの依存

```python
# BAD: 1つのAPIに完全依存
def bad_single_provider(audio_path):
    try:
        return google_stt(audio_path)
    except Exception:
        raise  # サービス停止時に全機能停止

# GOOD: フォールバック付きマルチプロバイダー
def good_multi_provider(audio_path):
    """複数プロバイダーによる冗長化"""
    stt = UnifiedSTT()
    stt.register("whisper", WhisperProvider())
    stt.register("google", GoogleProvider())
    stt.register("azure", AzureProvider())

    result = stt.transcribe(
        audio_path,
        language="ja",
        use_cache=True,
    )
    return result
```

---

## 8. FAQ

### Q1: Whisperはリアルタイム音声認識に使えますか？

標準のWhisperは30秒固定の入力を前提としたバッチ処理モデルのため、そのままではリアルタイム認識に不向きです。ただし、faster-whisperとVADを組み合わせた擬似リアルタイム処理や、whisper-streamingプロジェクトによるストリーミング対応は可能です。真のリアルタイム処理が必要な場合は、Google Speech-to-TextやAzure Speechのストリーミング認識APIを使うか、Whisperをストリーミング用にチューニングしたモデル（例: Distil-Whisper）を検討してください。

### Q2: 日本語STTの精度を上げるにはどうすればよいですか？

主な改善策は5つあります。(1) モデルサイズの拡大（large-v3が最高精度）。(2) 音声前処理の改善（ノイズ除去、正規化、リサンプリング）。(3) VADによる無音・非音声区間の除去。(4) 日本語特化データでのファインチューニング（ReazonSpeechデータセット等）。(5) 後処理の追加（句読点挿入、固有名詞補正、LLMによる校正）。特に、ドメイン特化のファインチューニングは専門用語の認識精度を大幅に改善します。

### Q3: 複数話者の音声を区別して文字起こしするには？

話者分離（Speaker Diarization）が必要です。Whisperは単体では話者分離機能を持ちませんが、pyannote-audioと組み合わせることで実現できます。手順は、(1) pyannote-audioで話者分離を実行、(2) 各話者区間ごとにWhisperで文字起こし、(3) タイムスタンプを照合して統合。クラウドAPIを使う場合は、Google Speech-to-TextやAzure Speechに組み込みの話者分離機能があり、設定を有効にするだけで利用できます。

### Q4: Whisperのハルシネーション（幻覚）を防ぐには？

Whisperのハルシネーションは、無音区間や非音声区間で発生しやすい問題です。対策として以下が有効です。(1) VADフィルタの使用で無音区間を事前に除去する。(2) `no_speech_threshold` パラメータを調整する（デフォルト0.6、上げると厳しく判定）。(3) `compression_ratio_threshold` でリピート検出する（デフォルト2.4）。(4) 後処理で典型的なハルシネーションパターン（「ご視聴ありがとうございました」等）を除去する。(5) `logprob_threshold` で低信頼度セグメントをフィルタリングする。

### Q5: STTの処理コストを最小化するには？

コスト最適化の主な戦略は以下の通りです。(1) VADで音声区間のみを処理し、無音部分の課金を避ける。(2) 頻繁に同じ音声を処理する場合はキャッシュを活用する。(3) リアルタイム性が不要な場合はバッチAPIを使用する（一般にバッチの方が安価）。(4) Whisperをローカルで実行すれば、GPU費用のみでAPI課金なし。(5) 短い音声にはWhisper APIの従量課金、長時間音声にはfaster-whisperローカル処理が経済的。(6) Deepgramは1分あたり$0.0043と最安値であり、コスト重視の場合は有力な選択肢。

### Q6: STT結果をLLMで後処理する方法は？

```python
from openai import OpenAI

def llm_post_process(raw_transcript: str, context: str = "") -> str:
    """LLMによるSTT結果の後処理"""
    client = OpenAI()

    prompt = f"""以下は音声認識の結果です。以下のルールに従って修正してください:
1. 誤認識と思われる部分を文脈から推測して修正
2. 句読点を適切に挿入
3. フィラーワード（えー、あの、まあ）を除去
4. 固有名詞の表記を統一
5. 口語表現を適切な書き言葉に変換

{f"コンテキスト: {context}" if context else ""}

音声認識結果:
{raw_transcript}

修正後のテキスト:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    return response.choices[0].message.content
```

---

## まとめ

| 項目 | 要点 |
|------|------|
| アーキテクチャ | CTC（高速）、Attention（高精度）、Transducer（ストリーミング） |
| Whisper | 汎用性最高のOSSモデル。large-v3が最高精度 |
| faster-whisper | CTranslate2最適化で2-4倍高速。VADフィルタ付き |
| クラウドAPI | リアルタイム・ストリーミングにはGoogle/Azureが優位 |
| Deepgram | 最低遅延・最安値。感情分析・要約機能も内蔵 |
| 精度改善 | 前処理 + VAD + ファインチューニング + 後処理の4段階 |
| 話者分離 | pyannote-audio + Whisper、またはクラウドAPIの組込機能 |
| ハルシネーション対策 | VADフィルタ + 閾値調整 + 後処理パターンマッチ |
| コスト最適化 | VADフィルタ + キャッシュ + ローカル処理 + Deepgram |

## 次に読むべきガイド

- [../02-voice/01-voice-assistants.md](../02-voice/01-voice-assistants.md) — 音声アシスタント実装
- [../02-voice/02-podcast-tools.md](../02-voice/02-podcast-tools.md) — ポッドキャスト文字起こし
- [../03-development/02-real-time-audio.md](../03-development/02-real-time-audio.md) — リアルタイム音声処理

## 参考文献

1. Radford, A., et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision" — Whisper論文。680K時間のデータで学習した大規模音声認識モデル
2. Gulati, A., et al. (2020). "Conformer: Convolution-augmented Transformer for Speech Recognition" — Conformer論文。CNN + Transformer の融合アーキテクチャ
3. Graves, A., et al. (2012). "Sequence Transduction with Recurrent Neural Networks" — RNN-T原論文。ストリーミング音声認識の基盤技術
4. Bredin, H., et al. (2023). "pyannote.audio 2.1 speaker diarization pipeline" — pyannote-audio論文。話者分離の代表的フレームワーク
5. Peng, Y., et al. (2023). "Reproducing Whisper-Style Training Using an Open-Source Toolkit and Publicly Available Data" — Whisper再現研究。OSSでのWhisperスタイル学習
6. Park, D.S., et al. (2019). "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" — データ拡張手法。音声認識精度を大幅に改善
