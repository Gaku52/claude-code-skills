# TTS技術 — VITS、Bark、ElevenLabs

> テキストから自然な音声を生成するTTS（Text-to-Speech）技術の仕組み、主要モデル、実装方法を解説する

## この章で学ぶこと

1. 現代TTS技術のアーキテクチャと進化（連結合成→ニューラルTTS→End-to-End）
2. 主要OSSモデル（VITS、Bark、Coqui TTS）の仕組みと使い分け
3. クラウドTTS API（ElevenLabs、OpenAI TTS、Google TTS）の実装パターン

---

## 1. TTS技術の進化

### 1.1 TTSパイプラインの変遷

```
TTS技術の進化
==================================================

第1世代: 連結合成（1990s-2010s）
  テキスト → 言語処理 → 音素列 → 波形DB検索 → 連結 → 音声
  例: フォルマント合成、単位選択合成

第2世代: 統計パラメトリック（2010s）
  テキスト → 言語特徴 → HMM/DNN → パラメータ → ボコーダ → 音声
  例: HTS、Merlin

第3世代: ニューラルTTS（2016-2020）
  テキスト → Encoder → Attention → Decoder → ボコーダ → 音声
  例: Tacotron 2 + WaveGlow/HiFi-GAN

第4世代: End-to-End（2021-現在）
  テキスト ──────→ 単一モデル ──────→ 音声
  例: VITS、VALL-E、Bark
==================================================
```

### 1.2 第3世代パイプライン（Tacotron 2型）

```
Tacotron 2 + HiFi-GAN パイプライン
==================================================

入力テキスト
    │
    ▼
┌─────────────┐
│ テキスト前処理 │  数字→読み、略語展開、G2P
└──────┬──────┘
       ▼
┌─────────────┐
│  Encoder     │  文字/音素の埋め込み + BiLSTM/Transformer
│ (テキスト)   │  → 言語的特徴の抽出
└──────┬──────┘
       │  Attention（どの文字をいつ読むか）
       ▼
┌─────────────┐
│  Decoder     │  自己回帰的にメルスペクトログラム生成
│ (音響)      │  フレームごとに予測
└──────┬──────┘
       ▼
┌─────────────┐
│  Vocoder     │  メルスペクトログラム → 波形
│ (HiFi-GAN)  │  GAN ベースの高速・高品質変換
└──────┬──────┘
       ▼
   音声波形
==================================================
```

---

## 2. 主要モデルの詳細

### 2.1 VITS（Conditional Variational Autoencoder with Adversarial Learning）

```python
# VITS の概念的なアーキテクチャ

class VITSArchitecture:
    """
    VITS: End-to-End TTS
    - テキスト → 音声を1つのモデルで完結
    - VAE + Flow + GAN の組み合わせ
    - 高速推論（リアルタイム以上）
    """

    def __init__(self):
        # テキストエンコーダ: テキスト → 言語的特徴
        self.text_encoder = TransformerEncoder(
            hidden_dim=192,
            n_layers=6,
            n_heads=2,
        )

        # ポステリアエンコーダ: 音声 → 潜在表現（学習時のみ）
        self.posterior_encoder = WaveNetEncoder(
            in_channels=513,  # リニアスペクトログラム
            hidden_dim=192,
        )

        # フローモデル: テキスト分布 ↔ 音声分布の橋渡し
        self.flow = ResidualCouplingBlock(
            channels=192,
            n_flows=4,
        )

        # デコーダ（HiFi-GAN ベース）: 潜在表現 → 波形
        self.decoder = HiFiGANGenerator(
            initial_channels=192,
            upsample_rates=[8, 8, 2, 2],
        )

    def inference(self, text):
        """推論（テキスト→音声）"""
        # 1. テキストエンコード
        text_hidden, text_mask = self.text_encoder(text)
        # 2. 単調アライメント探索（MAS）
        duration = monotonic_alignment_search(text_hidden)
        # 3. フロー逆変換で潜在表現生成
        z = self.flow.reverse(text_hidden, duration)
        # 4. 波形デコード
        audio = self.decoder(z)
        return audio
```

### 2.2 Bark（Suno AI）

```python
# Bark の使用例

from bark import SAMPLE_RATE, generate_audio, preload_models

# モデルのプリロード
preload_models()

# 基本的なテキスト音声合成
text_prompt = "こんにちは、私はAIアシスタントです。今日はいい天気ですね。"
audio_array = generate_audio(text_prompt)

# Bark の特殊タグ（非言語音声も生成可能）
special_prompts = {
    "笑い":    "今日は楽しかったです [laughs] 本当に最高でした",
    "歌":      "♪ ラララ、素敵な一日 ♪",
    "ため息":  "はぁ... [sighs] 疲れました",
    "音楽":    "[music] 美しいメロディーが流れています",
}

# 話者プリセットの使用
audio = generate_audio(
    text_prompt,
    history_prompt="v2/ja_speaker_0",  # 日本語話者プリセット
)

# 長文の分割処理
def generate_long_audio(long_text, max_length=200):
    """長文テキストを分割して音声合成"""
    import numpy as np

    sentences = split_into_sentences(long_text)
    audio_segments = []

    for sentence in sentences:
        if len(sentence) > max_length:
            # さらに分割
            chunks = split_by_punctuation(sentence, max_length)
            for chunk in chunks:
                audio_segments.append(generate_audio(chunk))
        else:
            audio_segments.append(generate_audio(sentence))

    # 音声を結合（短い無音を挿入）
    silence = np.zeros(int(SAMPLE_RATE * 0.3))  # 300ms
    result = np.concatenate(
        [np.concatenate([seg, silence]) for seg in audio_segments]
    )
    return result
```

### 2.3 ElevenLabs API

```python
import requests

class ElevenLabsTTS:
    """ElevenLabs TTS API ラッパー"""

    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }

    def text_to_speech(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
    ) -> bytes:
        """テキストを音声に変換"""
        url = f"{self.BASE_URL}/text-to-speech/{voice_id}"
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,         # 安定性（低→表現豊か）
                "similarity_boost": similarity_boost,  # 声質の一貫性
            },
        }
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.content  # MP3バイナリ

    def list_voices(self) -> list:
        """利用可能な音声一覧を取得"""
        url = f"{self.BASE_URL}/voices"
        response = requests.get(url, headers=self.headers)
        return response.json()["voices"]

    def clone_voice(self, name: str, audio_files: list) -> str:
        """音声クローニング（Instant Voice Cloning）"""
        url = f"{self.BASE_URL}/voices/add"
        files = [("files", open(f, "rb")) for f in audio_files]
        data = {"name": name}
        headers = {"xi-api-key": self.api_key}
        response = requests.post(url, data=data, files=files, headers=headers)
        return response.json()["voice_id"]

# 使用例
tts = ElevenLabsTTS("your-api-key")
audio = tts.text_to_speech("こんにちは、世界！")
with open("output.mp3", "wb") as f:
    f.write(audio)
```

### 2.4 OpenAI TTS API

```python
from openai import OpenAI

client = OpenAI()

# 基本的なTTS
response = client.audio.speech.create(
    model="tts-1",       # tts-1（高速） or tts-1-hd（高品質）
    voice="alloy",       # alloy, echo, fable, onyx, nova, shimmer
    input="音声合成のテストです。OpenAIのTTSは非常に自然な音声を生成します。",
    response_format="mp3",  # mp3, opus, aac, flac, wav, pcm
    speed=1.0,           # 0.25 〜 4.0
)

# ファイルに保存
response.stream_to_file("output.mp3")

# ストリーミングTTS
response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="ストリーミングで音声を生成します。",
    response_format="pcm",  # ストリーミングにはPCMかOpusが最適
)

# チャンクごとに処理
for chunk in response.iter_bytes(chunk_size=4096):
    audio_player.play(chunk)
```

---

## 3. 音声の品質制御

### 3.1 パラメータチューニング

```
TTS品質制御パラメータ
==================================================

┌──────────────────────────────────────────┐
│            品質制御の軸                   │
│                                          │
│   自然さ ←──────────→ 安定性              │
│   (expressiveness)    (stability)        │
│                                          │
│   低安定性:                               │
│   ・感情表現が豊か                         │
│   ・声質が変動しやすい                     │
│   ・長文で品質が揺れる                     │
│                                          │
│   高安定性:                               │
│   ・一貫した声質                           │
│   ・単調になりがち                         │
│   ・ナレーション向き                       │
│                                          │
│   ────────────────────────               │
│                                          │
│   速度 0.5x ←────────→ 2.0x              │
│   ゆっくり      標準1.0x     速い         │
│                                          │
│   温度（sampling_temperature）            │
│   低(0.1): 安全で予測可能                  │
│   高(1.0): 多様で表現豊か                  │
└──────────────────────────────────────────┘
==================================================
```

---

## 4. 比較表

### 4.1 主要TTSモデル/サービス比較

| 項目 | VITS | Bark | ElevenLabs | OpenAI TTS | Google TTS |
|------|------|------|-----------|-----------|-----------|
| 種別 | OSS | OSS | API | API | API |
| 品質(MOS) | 4.0-4.3 | 3.8-4.2 | 4.5-4.8 | 4.3-4.6 | 4.0-4.4 |
| 日本語対応 | 要学習 | 対応 | 対応 | 対応 | 対応 |
| リアルタイム性 | 高速 | 遅い | 中程度 | 高速 | 高速 |
| 声クローン | 要学習 | プリセット | 対応 | 非対応 | 非対応 |
| 感情表現 | 限定的 | 非言語音OK | 高い | 中程度 | 中程度 |
| コスト | 無料 | 無料 | 従量課金 | 従量課金 | 従量課金 |
| GPU要件 | 必要 | 必要(大) | 不要 | 不要 | 不要 |

### 4.2 用途別TTS推奨選定

| ユースケース | 推奨 | 理由 |
|-------------|------|------|
| プロトタイプ | OpenAI TTS | セットアップ簡単、品質十分 |
| 商用ナレーション | ElevenLabs | 最高品質、声クローン対応 |
| ゲーム/アプリ組込 | VITS | ローカル実行、カスタマイズ可 |
| 多言語対応 | Google TTS / Bark | 広い言語カバレッジ |
| 低コスト大量生成 | Coqui TTS / Piper | OSS、GPUあれば無料 |
| 感情豊かな音声 | Bark / ElevenLabs | 非言語音、感情制御が可能 |
| リアルタイム対話 | OpenAI TTS + PCM | ストリーミング対応、低遅延 |
| オフライン動作 | Piper / VITS | ローカル実行、ネット不要 |

---

## 5. アンチパターン

### 5.1 アンチパターン: テキスト前処理の省略

```python
# BAD: テキストをそのままTTSに渡す
def bad_tts(text):
    return tts_model.synthesize(text)
    # 問題: "100km" → "ひゃっけーえむ"（意味不明な読み）
    # 問題: "2026/2/11" → 予測不能な読み
    # 問題: "API" → "あぴ"

# GOOD: テキスト正規化パイプラインを通す
import re

def normalize_text_for_tts(text: str) -> str:
    """TTS向けテキスト正規化"""
    # 数字の読み変換
    text = re.sub(r'(\d+)km', lambda m: f'{m.group(1)}キロメートル', text)
    text = re.sub(r'(\d+)kg', lambda m: f'{m.group(1)}キログラム', text)

    # 日付の変換
    text = re.sub(
        r'(\d{4})/(\d{1,2})/(\d{1,2})',
        lambda m: f'{m.group(1)}年{m.group(2)}月{m.group(3)}日',
        text
    )

    # 英字略語の処理
    abbreviations = {"API": "エーピーアイ", "AI": "エーアイ", "URL": "ユーアールエル"}
    for abbr, reading in abbreviations.items():
        text = text.replace(abbr, reading)

    return text

def good_tts(text):
    normalized = normalize_text_for_tts(text)
    return tts_model.synthesize(normalized)
```

### 5.2 アンチパターン: 長文の一括合成

```python
# BAD: 長文をそのまま一度に合成
def bad_long_tts(long_text):
    # 問題1: メモリ不足
    # 問題2: 品質劣化（後半ほど品質低下）
    # 問題3: 途中でエラーが起きると全てやり直し
    return tts_model.synthesize(long_text)

# GOOD: 文単位で分割して合成・結合
def good_long_tts(long_text, max_chars=150):
    import numpy as np

    # 文分割（句読点で区切る）
    sentences = re.split(r'(?<=[。！？\n])', long_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    audio_parts = []
    for i, sentence in enumerate(sentences):
        try:
            audio = tts_model.synthesize(sentence)
            audio_parts.append(audio)

            # 文間にポーズを挿入（300ms）
            pause = np.zeros(int(sample_rate * 0.3))
            audio_parts.append(pause)

        except Exception as e:
            print(f"文 {i} の合成に失敗: {e}")
            # 失敗した文はスキップして続行
            continue

    return np.concatenate(audio_parts)
```

---

## 6. FAQ

### Q1: TTSの音声品質を客観的に評価する方法はありますか？

主要な評価指標として、MOS（Mean Opinion Score、5段階主観評価）が最も広く使われています。自動評価としては、PESQ（知覚音声品質評価）、UTMOS（ニューラルMOS予測器）、話者類似度（Speaker Similarity）などがあります。実用的には、ターゲット用途に近い文章セットを用意し、複数の評価者にブラインドテストで聴き比べてもらうABテストが効果的です。

### Q2: 日本語TTSでイントネーションが不自然になる原因と対策は？

主な原因は、(1) アクセント辞書の不足（固有名詞、新語）、(2) 文脈依存のアクセント変化への対応不足、(3) 学習データの偏りです。対策として、OpenJTalk のユーザー辞書にアクセント情報を追加する、SSML（Speech Synthesis Markup Language）でプロソディを明示的に制御する、ドメイン特化データでファインチューニングする、といった方法があります。

### Q3: ストリーミングTTSを実装する際の注意点は？

ストリーミングTTSでは以下が重要です。(1) バッファリング戦略: 最初のチャンクをなるべく早く送出し（First Token Latency の最小化）、以降はスムーズに再生できるバッファサイズを確保する。(2) 音声フォーマット: PCMやOpusなど、チャンク境界で分割可能なフォーマットを使用する（MP3はフレーム境界に注意）。(3) エラーハンドリング: ネットワーク切断時の再接続、部分的な音声バッファの処理を設計する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 進化の方向 | パイプライン分離 → End-to-End 統合へ |
| 主要OSS | VITS（高速E2E）、Bark（多機能）、Piper（軽量） |
| 主要API | ElevenLabs（最高品質）、OpenAI（簡単）、Google（多言語） |
| 前処理 | テキスト正規化（数字、略語、日付）が品質に直結 |
| 長文対応 | 文単位分割→合成→結合が基本パターン |
| 品質制御 | stability/similarity/speed/temperatureで調整 |

## 次に読むべきガイド

- [03-stt-technologies.md](./03-stt-technologies.md) — STT技術（Whisper、Google、Azure）
- [../02-voice/00-voice-cloning.md](../02-voice/00-voice-cloning.md) — ボイスクローニング技術
- [../03-development/00-audio-apis.md](../03-development/00-audio-apis.md) — 音声API実装ガイド

## 参考文献

1. Kim, J., et al. (2021). "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" — VITS原論文。VAE+Flow+GANによるEnd-to-End TTS
2. Kong, J., et al. (2020). "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" — HiFi-GANボコーダ論文
3. Wang, C., et al. (2023). "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" — VALL-E論文。音声のコーデックトークン化によるゼロショットTTS
4. Shen, J., et al. (2018). "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" — Tacotron 2 論文
