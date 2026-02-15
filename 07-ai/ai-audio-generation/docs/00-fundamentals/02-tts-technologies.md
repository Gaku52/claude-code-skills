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

---

## 7. 高度なTTS実装パターン

### 7.1 SSMLによるプロソディ制御

```python
# SSML（Speech Synthesis Markup Language）による細かな音声制御

class SSMLBuilder:
    """SSML構築ヘルパー"""
    
    def __init__(self):
        self.elements = []
    
    def add_text(self, text: str) -> 'SSMLBuilder':
        """通常テキスト"""
        self.elements.append(text)
        return self
    
    def add_break(self, time_ms: int = 500) -> 'SSMLBuilder':
        """ポーズ挿入"""
        self.elements.append(f'<break time="{time_ms}ms"/>')
        return self
    
    def add_emphasis(self, text: str, level: str = "moderate") -> 'SSMLBuilder':
        """強調（strong, moderate, reduced）"""
        self.elements.append(f'<emphasis level="{level}">{text}</emphasis>')
        return self
    
    def add_prosody(
        self, text: str, 
        rate: str = "medium", 
        pitch: str = "medium",
        volume: str = "medium"
    ) -> 'SSMLBuilder':
        """速度・ピッチ・音量制御"""
        self.elements.append(
            f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">{text}</prosody>'
        )
        return self
    
    def add_say_as(self, text: str, interpret_as: str) -> 'SSMLBuilder':
        """読み方の指定（date, time, telephone, cardinal, ordinal）"""
        self.elements.append(
            f'<say-as interpret-as="{interpret_as}">{text}</say-as>'
        )
        return self
    
    def add_phoneme(self, text: str, ph: str, alphabet: str = "ipa") -> 'SSMLBuilder':
        """発音記号の明示指定"""
        self.elements.append(
            f'<phoneme alphabet="{alphabet}" ph="{ph}">{text}</phoneme>'
        )
        return self
    
    def build(self) -> str:
        """SSML文字列を生成"""
        content = "".join(self.elements)
        return f'<speak>{content}</speak>'


# 使用例: ニュース読み上げ
builder = SSMLBuilder()
ssml = (
    builder
    .add_prosody("本日のニュースをお伝えします。", rate="slow", pitch="low")
    .add_break(800)
    .add_emphasis("速報です。", level="strong")
    .add_break(500)
    .add_text("本日")
    .add_say_as("2026年2月14日", interpret_as="date")
    .add_text("、東京都内で大規模な技術カンファレンスが開催されました。")
    .add_break(600)
    .add_prosody("参加者は", rate="medium")
    .add_say_as("5000", interpret_as="cardinal")
    .add_text("人を超え、過去最大規模となりました。")
    .build()
)

print(ssml)
# Google TTS や Azure TTS で使用可能
```

### 7.2 マルチスピーカーTTSシステム

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class SpeakerGender(Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"

class EmotionType(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISE = "surprise"
    WHISPER = "whisper"

@dataclass
class SpeakerProfile:
    """話者プロファイル"""
    name: str
    gender: SpeakerGender
    voice_id: str
    language: str = "ja-JP"
    default_speed: float = 1.0
    default_pitch: float = 0.0  # semitones
    emotion: EmotionType = EmotionType.NEUTRAL
    style_weights: dict = field(default_factory=dict)

@dataclass
class DialogueLine:
    """対話の1行"""
    speaker: str
    text: str
    emotion: Optional[EmotionType] = None
    pause_after_ms: int = 500

class MultiSpeakerTTS:
    """マルチスピーカーTTSエンジン"""
    
    def __init__(self, tts_engine):
        self.tts_engine = tts_engine
        self.speakers: dict[str, SpeakerProfile] = {}
        self.sample_rate = 24000
    
    def register_speaker(self, profile: SpeakerProfile):
        """話者を登録"""
        self.speakers[profile.name] = profile
    
    def synthesize_dialogue(
        self, 
        dialogue: list[DialogueLine],
        output_path: str,
        crossfade_ms: int = 50,
    ) -> np.ndarray:
        """対話全体を合成"""
        audio_segments = []
        
        for line in dialogue:
            speaker = self.speakers[line.speaker]
            emotion = line.emotion or speaker.emotion
            
            # 話者固有の設定で合成
            audio = self.tts_engine.synthesize(
                text=line.text,
                voice_id=speaker.voice_id,
                speed=speaker.default_speed,
                pitch=speaker.default_pitch,
                emotion=emotion.value,
            )
            
            audio_segments.append(audio)
            
            # ポーズを挿入
            pause = np.zeros(int(self.sample_rate * line.pause_after_ms / 1000))
            audio_segments.append(pause)
        
        # クロスフェードで結合
        result = self._crossfade_concat(audio_segments, crossfade_ms)
        
        # ファイルに保存
        self._save_audio(result, output_path)
        return result
    
    def _crossfade_concat(self, segments: list[np.ndarray], fade_ms: int) -> np.ndarray:
        """クロスフェード付き結合"""
        fade_samples = int(self.sample_rate * fade_ms / 1000)
        
        if len(segments) == 0:
            return np.array([])
        
        result = segments[0]
        for seg in segments[1:]:
            if len(seg) == 0:
                continue
            if len(result) >= fade_samples and len(seg) >= fade_samples:
                # フェードアウト・フェードイン
                fade_out = np.linspace(1.0, 0.0, fade_samples)
                fade_in = np.linspace(0.0, 1.0, fade_samples)
                
                result[-fade_samples:] *= fade_out
                seg_copy = seg.copy()
                seg_copy[:fade_samples] *= fade_in
                
                # オーバーラップ部分を加算
                result[-fade_samples:] += seg_copy[:fade_samples]
                result = np.concatenate([result, seg_copy[fade_samples:]])
            else:
                result = np.concatenate([result, seg])
        
        return result
    
    def _save_audio(self, audio: np.ndarray, path: str):
        """音声ファイルに保存"""
        import soundfile as sf
        sf.write(path, audio, self.sample_rate)


# 使用例: オーディオドラマの対話合成
multi_tts = MultiSpeakerTTS(tts_engine=None)  # 実際のエンジンを渡す

multi_tts.register_speaker(SpeakerProfile(
    name="太郎",
    gender=SpeakerGender.MALE,
    voice_id="ja_male_01",
    default_speed=1.0,
    default_pitch=-2.0,
))

multi_tts.register_speaker(SpeakerProfile(
    name="花子",
    gender=SpeakerGender.FEMALE,
    voice_id="ja_female_01",
    default_speed=1.05,
    default_pitch=2.0,
))

dialogue = [
    DialogueLine("太郎", "おはようございます。今日の会議の準備はできましたか？"),
    DialogueLine("花子", "はい、資料は全部揃っています。", emotion=EmotionType.HAPPY),
    DialogueLine("太郎", "素晴らしい。では10時に会議室で。", pause_after_ms=800),
    DialogueLine("花子", "承知しました。お先に失礼します。"),
]

audio = multi_tts.synthesize_dialogue(dialogue, "dialogue_output.wav")
```

### 7.3 感情制御付きTTSパイプライン

```python
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class EmotionParameters:
    """感情パラメータ"""
    name: str
    stability: float        # 0.0-1.0: 低=表現豊か、高=安定
    similarity_boost: float # 0.0-1.0: 声質の一貫性
    style: float           # 0.0-1.0: スタイルの強さ
    speed: float           # 速度倍率
    pitch_shift: float     # ピッチシフト（semitones）

# 感情プリセット
EMOTION_PRESETS = {
    "neutral": EmotionParameters("neutral", 0.5, 0.75, 0.0, 1.0, 0.0),
    "happy": EmotionParameters("happy", 0.3, 0.7, 0.6, 1.1, 1.5),
    "sad": EmotionParameters("sad", 0.6, 0.8, 0.5, 0.85, -1.5),
    "angry": EmotionParameters("angry", 0.2, 0.6, 0.8, 1.15, 0.5),
    "excited": EmotionParameters("excited", 0.2, 0.65, 0.7, 1.2, 2.0),
    "calm": EmotionParameters("calm", 0.8, 0.85, 0.3, 0.9, -0.5),
    "whisper": EmotionParameters("whisper", 0.7, 0.9, 0.4, 0.8, -1.0),
    "narration": EmotionParameters("narration", 0.6, 0.8, 0.2, 0.95, 0.0),
}


class EmotionAwareTTS:
    """感情認識付きTTSパイプライン"""
    
    def __init__(self, tts_client, sentiment_analyzer=None):
        self.tts_client = tts_client
        self.sentiment_analyzer = sentiment_analyzer
    
    def auto_detect_emotion(self, text: str) -> str:
        """テキストから感情を自動推定"""
        if self.sentiment_analyzer:
            return self.sentiment_analyzer.predict(text)
        
        # 簡易ルールベース推定
        emotion_keywords = {
            "happy": ["嬉しい", "楽しい", "素晴らしい", "最高", "やった", "ありがとう"],
            "sad": ["悲しい", "残念", "辛い", "寂しい", "申し訳"],
            "angry": ["怒", "許せない", "ふざけるな", "いい加減にしろ"],
            "excited": ["すごい", "驚き", "信じられない", "！！"],
            "calm": ["穏やか", "静か", "ゆっくり"],
        }
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return emotion
        
        return "neutral"
    
    def synthesize_with_emotion(
        self,
        text: str,
        voice_id: str,
        emotion: Optional[str] = None,
    ) -> bytes:
        """感情付き音声合成"""
        if emotion is None:
            emotion = self.auto_detect_emotion(text)
        
        params = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["neutral"])
        
        audio = self.tts_client.text_to_speech(
            text=text,
            voice_id=voice_id,
            stability=params.stability,
            similarity_boost=params.similarity_boost,
            style=params.style,
            speed=params.speed,
        )
        
        return audio
    
    def synthesize_narrative(
        self,
        text: str,
        voice_id: str,
    ) -> list[bytes]:
        """長文ナレーションを感情変化付きで合成"""
        # 文ごとに分割
        sentences = re.split(r'(?<=[。！？\n])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        audio_parts = []
        for sentence in sentences:
            emotion = self.auto_detect_emotion(sentence)
            audio = self.synthesize_with_emotion(sentence, voice_id, emotion)
            audio_parts.append(audio)
        
        return audio_parts
```

---

## 8. テキスト前処理の詳細実装

### 8.1 日本語テキスト正規化パイプライン

```python
import re
from typing import Callable

class JapaneseTextNormalizer:
    """日本語TTS向けテキスト正規化パイプライン"""
    
    def __init__(self):
        self.rules: list[tuple[str, Callable]] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """デフォルト正規化ルールの設定"""
        self.rules = [
            ("全角英数→半角", self._zenkaku_to_hankaku),
            ("URL除去", self._remove_urls),
            ("メールアドレス正規化", self._normalize_email),
            ("数値の読み変換", self._normalize_numbers),
            ("日付の読み変換", self._normalize_dates),
            ("時刻の読み変換", self._normalize_times),
            ("単位の読み変換", self._normalize_units),
            ("英字略語の展開", self._expand_abbreviations),
            ("記号の読み変換", self._normalize_symbols),
            ("繰り返し記号の処理", self._handle_repetition),
            ("不要な空白の正規化", self._normalize_whitespace),
        ]
    
    def normalize(self, text: str) -> str:
        """テキスト正規化を実行"""
        for rule_name, rule_func in self.rules:
            text = rule_func(text)
        return text
    
    def _zenkaku_to_hankaku(self, text: str) -> str:
        """全角英数字を半角に変換"""
        result = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            else:
                result.append(char)
        return "".join(result)
    
    def _remove_urls(self, text: str) -> str:
        """URLを除去またはドメイン名のみに置換"""
        return re.sub(
            r'https?://[^\s]+',
            lambda m: self._url_to_readable(m.group(0)),
            text
        )
    
    def _url_to_readable(self, url: str) -> str:
        """URLを読み上げ可能な形式に変換"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        return f"{domain}のリンク"
    
    def _normalize_email(self, text: str) -> str:
        """メールアドレスを読み上げ形式に"""
        return re.sub(
            r'[\w.-]+@[\w.-]+\.\w+',
            lambda m: m.group(0).replace("@", "アットマーク").replace(".", "ドット"),
            text
        )
    
    def _normalize_numbers(self, text: str) -> str:
        """数値の読み変換"""
        # 小数点
        text = re.sub(r'(\d+)\.(\d+)', r'\1点\2', text)
        # パーセンテージ
        text = re.sub(r'(\d+(?:\.\d+)?)%', r'\1パーセント', text)
        # カンマ区切り数値
        text = re.sub(r'(\d{1,3}(?:,\d{3})+)', lambda m: m.group(0).replace(",", ""), text)
        return text
    
    def _normalize_dates(self, text: str) -> str:
        """日付の読み変換"""
        # YYYY/MM/DD or YYYY-MM-DD
        text = re.sub(
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
            lambda m: f'{m.group(1)}年{m.group(2)}月{m.group(3)}日',
            text
        )
        # MM/DD
        text = re.sub(
            r'(\d{1,2})[/-](\d{1,2})(?!\d)',
            lambda m: f'{m.group(1)}月{m.group(2)}日',
            text
        )
        return text
    
    def _normalize_times(self, text: str) -> str:
        """時刻の読み変換"""
        text = re.sub(
            r'(\d{1,2}):(\d{2})(?::(\d{2}))?',
            lambda m: f'{m.group(1)}時{m.group(2)}分' + (f'{m.group(3)}秒' if m.group(3) else ''),
            text
        )
        return text
    
    def _normalize_units(self, text: str) -> str:
        """単位の読み変換"""
        units = {
            "km": "キロメートル", "cm": "センチメートル", "mm": "ミリメートル",
            "m": "メートル", "kg": "キログラム", "g": "グラム", "mg": "ミリグラム",
            "km/h": "キロメートル毎時", "m/s": "メートル毎秒",
            "GB": "ギガバイト", "MB": "メガバイト", "KB": "キロバイト",
            "TB": "テラバイト", "Hz": "ヘルツ", "kHz": "キロヘルツ",
            "MHz": "メガヘルツ", "GHz": "ギガヘルツ",
            "℃": "度", "°C": "度",
        }
        for unit, reading in sorted(units.items(), key=lambda x: -len(x[0])):
            text = re.sub(
                rf'(\d+(?:\.\d+)?)\s*{re.escape(unit)}(?!\w)',
                rf'\1{reading}',
                text
            )
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """英字略語を展開"""
        abbrevs = {
            "API": "エーピーアイ", "AI": "エーアイ", "URL": "ユーアールエル",
            "HTML": "エイチティーエムエル", "CSS": "シーエスエス",
            "GPU": "ジーピーユー", "CPU": "シーピーユー",
            "RAM": "ラム", "SSD": "エスエスディー",
            "TTS": "ティーティーエス", "STT": "エスティーティー",
            "IoT": "アイオーティー", "SDK": "エスディーケー",
            "PDF": "ピーディーエフ", "FAQ": "エフエーキュー",
            "OS": "オーエス", "DB": "ディービー",
        }
        for abbr, reading in sorted(abbrevs.items(), key=lambda x: -len(x[0])):
            text = re.sub(rf'\b{abbr}\b', reading, text)
        return text
    
    def _normalize_symbols(self, text: str) -> str:
        """記号の読み変換"""
        symbols = {
            "→": "から", "←": "から", "↑": "上", "↓": "下",
            "＋": "プラス", "－": "マイナス", "×": "かける", "÷": "割る",
            "※": "注", "★": "", "☆": "", "♪": "",
            "&": "アンド", "#": "シャープ",
        }
        for sym, reading in symbols.items():
            text = text.replace(sym, reading)
        return text
    
    def _handle_repetition(self, text: str) -> str:
        """繰り返し記号（々）の処理は不要（日本語として自然）"""
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """空白の正規化"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# 使用例
normalizer = JapaneseTextNormalizer()

test_texts = [
    "2026/2/14のAI技術カンファレンスで、GPUを100kg搭載したサーバーの話がありました。",
    "URLはhttps://example.comです。メールはinfo@example.comまで。",
    "温度は36.5℃、速度は120km/hでした。",
    "FAQ: APIのレスポンスは99.9%の確率で200ms以内です。",
]

for text in test_texts:
    normalized = normalizer.normalize(text)
    print(f"原文: {text}")
    print(f"正規: {normalized}")
    print()
```

### 8.2 文分割アルゴリズム

```python
import re
from dataclasses import dataclass

@dataclass
class SentenceChunk:
    """分割された文のチャンク"""
    text: str
    char_count: int
    is_continuation: bool = False

class TTSSentenceSplitter:
    """TTS向け文分割器"""
    
    def __init__(self, max_chars: int = 150, min_chars: int = 10):
        self.max_chars = max_chars
        self.min_chars = min_chars
    
    def split(self, text: str) -> list[SentenceChunk]:
        """テキストをTTS向けに分割"""
        # Step 1: 文単位で分割
        sentences = self._split_sentences(text)
        
        # Step 2: 長すぎる文をさらに分割
        chunks = []
        for sentence in sentences:
            if len(sentence) <= self.max_chars:
                chunks.append(SentenceChunk(
                    text=sentence, 
                    char_count=len(sentence)
                ))
            else:
                sub_chunks = self._split_long_sentence(sentence)
                for i, sub in enumerate(sub_chunks):
                    chunks.append(SentenceChunk(
                        text=sub,
                        char_count=len(sub),
                        is_continuation=i > 0,
                    ))
        
        # Step 3: 短すぎる文を結合
        chunks = self._merge_short_chunks(chunks)
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """文単位で分割"""
        # 句読点・感嘆符・疑問符・改行で分割
        parts = re.split(r'(?<=[。！？\n])\s*', text)
        return [p.strip() for p in parts if p.strip()]
    
    def _split_long_sentence(self, sentence: str) -> list[str]:
        """長い文を読点やスペースで分割"""
        # 読点で分割を試みる
        parts = re.split(r'(?<=[、，,])\s*', sentence)
        
        result = []
        current = ""
        for part in parts:
            if len(current) + len(part) <= self.max_chars:
                current += part
            else:
                if current:
                    result.append(current.strip())
                current = part
        if current:
            result.append(current.strip())
        
        return result
    
    def _merge_short_chunks(self, chunks: list[SentenceChunk]) -> list[SentenceChunk]:
        """短すぎるチャンクを結合"""
        if not chunks:
            return chunks
        
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if merged[-1].char_count < self.min_chars:
                # 前のチャンクと結合
                merged[-1] = SentenceChunk(
                    text=merged[-1].text + chunk.text,
                    char_count=merged[-1].char_count + chunk.char_count,
                )
            else:
                merged.append(chunk)
        
        return merged


# 使用例
splitter = TTSSentenceSplitter(max_chars=100)
text = """
人工知能の進歩は目覚ましく、特にテキスト音声合成の分野では、
2020年代に入ってから人間の発話と区別がつかないレベルの品質に達しました。
VITSやBarkなどのEnd-to-Endモデルの登場により、
テキストから直接高品質な音声を生成できるようになっています。
今後はさらなる多言語対応と感情表現の向上が期待されています！
"""

chunks = splitter.split(text)
for i, chunk in enumerate(chunks):
    cont = " (続き)" if chunk.is_continuation else ""
    print(f"チャンク{i+1}{cont}: [{chunk.char_count}文字] {chunk.text}")
```

---

## 9. 音声品質評価システム

### 9.1 自動MOS予測（UTMOS）

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class QualityScore:
    """音声品質スコア"""
    mos: float              # Mean Opinion Score (1.0-5.0)
    naturalness: float      # 自然さ (0.0-1.0)
    intelligibility: float  # 明瞭さ (0.0-1.0)
    speaker_similarity: float  # 話者類似度 (0.0-1.0)
    prosody_score: float    # プロソディ (0.0-1.0)

class TTSQualityEvaluator:
    """TTS品質自動評価器"""
    
    def __init__(self):
        self.utmos_model = None  # UTMOS モデル
        self.pesq_available = False
    
    def evaluate(self, generated_audio: np.ndarray, 
                 reference_audio: np.ndarray = None,
                 sample_rate: int = 24000) -> QualityScore:
        """音声品質を総合評価"""
        
        # 1. MOS予測（UTMOS）
        mos = self._predict_mos(generated_audio, sample_rate)
        
        # 2. 自然さスコア
        naturalness = self._evaluate_naturalness(generated_audio, sample_rate)
        
        # 3. 明瞭さ（STTで逆変換して一致率を計算）
        intelligibility = self._evaluate_intelligibility(generated_audio, sample_rate)
        
        # 4. 話者類似度（参照音声がある場合）
        similarity = 0.0
        if reference_audio is not None:
            similarity = self._evaluate_speaker_similarity(
                generated_audio, reference_audio, sample_rate
            )
        
        # 5. プロソディスコア
        prosody = self._evaluate_prosody(generated_audio, sample_rate)
        
        return QualityScore(
            mos=mos,
            naturalness=naturalness,
            intelligibility=intelligibility,
            speaker_similarity=similarity,
            prosody_score=prosody,
        )
    
    def _predict_mos(self, audio: np.ndarray, sr: int) -> float:
        """UTMOS: 自動MOS予測"""
        # 実際にはUTMOSモデルを使用
        # ここでは簡易的な特徴ベースの推定
        
        # SNR推定
        signal_power = np.mean(audio ** 2)
        if signal_power == 0:
            return 1.0
        
        # スペクトル平坦度（自然音声は低い）
        spectrum = np.abs(np.fft.rfft(audio))
        spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-10))) / (np.mean(spectrum) + 1e-10)
        
        # 簡易MOS推定（実際にはニューラルネットワーク使用）
        mos = 3.5 - 2.0 * spectral_flatness + 0.5 * np.log10(signal_power + 1e-10)
        return np.clip(mos, 1.0, 5.0)
    
    def _evaluate_naturalness(self, audio: np.ndarray, sr: int) -> float:
        """自然さの評価"""
        # ピッチの変動パターンを分析
        # 自然な音声は適度なピッチ変動がある
        frame_size = int(sr * 0.025)
        hop_size = int(sr * 0.010)
        
        pitches = []
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            pitch = self._estimate_pitch(frame, sr)
            if pitch > 0:
                pitches.append(pitch)
        
        if len(pitches) < 2:
            return 0.5
        
        # ピッチの変動係数（CV）を計算
        pitch_cv = np.std(pitches) / (np.mean(pitches) + 1e-10)
        
        # 自然な音声のCV範囲 (0.1-0.3)
        if 0.1 <= pitch_cv <= 0.3:
            naturalness = 1.0
        elif pitch_cv < 0.1:
            naturalness = pitch_cv / 0.1
        else:
            naturalness = max(0.0, 1.0 - (pitch_cv - 0.3) / 0.5)
        
        return naturalness
    
    def _estimate_pitch(self, frame: np.ndarray, sr: int) -> float:
        """自己相関法によるピッチ推定"""
        if np.max(np.abs(frame)) < 0.01:
            return 0.0
        
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        
        # 最初のピークを探索（50Hz-500Hzの範囲）
        min_lag = sr // 500
        max_lag = sr // 50
        
        if max_lag >= len(corr):
            return 0.0
        
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            return 0.0
        
        peak_idx = np.argmax(search) + min_lag
        return sr / peak_idx if peak_idx > 0 else 0.0
    
    def _evaluate_intelligibility(self, audio: np.ndarray, sr: int) -> float:
        """明瞭さの評価（STOI近似）"""
        # 短時間エネルギーの変動を分析
        frame_size = int(sr * 0.025)
        hop_size = int(sr * 0.010)
        
        energies = []
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            energies.append(np.sum(frame ** 2))
        
        if len(energies) < 2:
            return 0.5
        
        energies = np.array(energies)
        # エネルギーの動的範囲
        dynamic_range = 10 * np.log10(
            (np.max(energies) + 1e-10) / (np.mean(energies) + 1e-10)
        )
        
        # 適度な動的範囲が明瞭さの指標
        return np.clip(dynamic_range / 30.0, 0.0, 1.0)
    
    def _evaluate_speaker_similarity(
        self, gen_audio: np.ndarray, ref_audio: np.ndarray, sr: int
    ) -> float:
        """話者類似度の評価"""
        # MFCCベースの簡易比較
        gen_mfcc = self._extract_mfcc(gen_audio, sr)
        ref_mfcc = self._extract_mfcc(ref_audio, sr)
        
        # コサイン類似度
        gen_mean = np.mean(gen_mfcc, axis=0)
        ref_mean = np.mean(ref_mfcc, axis=0)
        
        similarity = np.dot(gen_mean, ref_mean) / (
            np.linalg.norm(gen_mean) * np.linalg.norm(ref_mean) + 1e-10
        )
        
        return max(0.0, similarity)
    
    def _extract_mfcc(self, audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """MFCC抽出（簡易版）"""
        # 実際にはlibrosaを使用
        frame_size = int(sr * 0.025)
        hop_size = int(sr * 0.010)
        
        mfccs = []
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            spectrum = np.abs(np.fft.rfft(frame))
            log_spectrum = np.log(spectrum + 1e-10)
            cepstrum = np.fft.irfft(log_spectrum)
            mfccs.append(cepstrum[:n_mfcc])
        
        return np.array(mfccs) if mfccs else np.zeros((1, n_mfcc))
    
    def _evaluate_prosody(self, audio: np.ndarray, sr: int) -> float:
        """プロソディの評価"""
        # エネルギーとピッチの時間変化パターンを評価
        frame_size = int(sr * 0.025)
        hop_size = int(sr * 0.010)
        
        energies = []
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            energies.append(np.sqrt(np.mean(frame ** 2)))
        
        if len(energies) < 10:
            return 0.5
        
        energies = np.array(energies)
        
        # エネルギー変動のスムーズさ（急激な変化が少ないほど良い）
        diff = np.abs(np.diff(energies))
        smoothness = 1.0 - np.clip(np.mean(diff) / (np.mean(energies) + 1e-10), 0, 1)
        
        return smoothness


# 使用例
evaluator = TTSQualityEvaluator()

# ダミー音声で評価
sample_rate = 24000
duration = 3.0
t = np.linspace(0, duration, int(sample_rate * duration))
test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz テストトーン

score = evaluator.evaluate(test_audio, sample_rate=sample_rate)
print(f"MOS: {score.mos:.2f}")
print(f"自然さ: {score.naturalness:.2f}")
print(f"明瞭さ: {score.intelligibility:.2f}")
print(f"プロソディ: {score.prosody_score:.2f}")
```

---

## 10. TTS本番デプロイメント

### 10.1 キャッシュ戦略とパフォーマンス最適化

```python
import hashlib
import json
import os
import time
from typing import Optional
from dataclasses import dataclass

@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    audio_path: str
    text_hash: str
    voice_id: str
    params_hash: str
    created_at: float
    file_size: int
    access_count: int = 0
    last_accessed: float = 0.0

class TTSCacheManager:
    """TTS音声キャッシュマネージャー"""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 1000):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_path = os.path.join(cache_dir, "cache_index.json")
        self.index: dict[str, CacheEntry] = {}
        os.makedirs(cache_dir, exist_ok=True)
        self._load_index()
    
    def _make_key(self, text: str, voice_id: str, params: dict) -> str:
        """キャッシュキーの生成"""
        content = json.dumps({
            "text": text,
            "voice_id": voice_id,
            "params": params,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, text: str, voice_id: str, params: dict) -> Optional[bytes]:
        """キャッシュからの取得"""
        key = self._make_key(text, voice_id, params)
        
        if key not in self.index:
            return None
        
        entry = self.index[key]
        if not os.path.exists(entry.audio_path):
            del self.index[key]
            return None
        
        # アクセス記録の更新
        entry.access_count += 1
        entry.last_accessed = time.time()
        
        with open(entry.audio_path, "rb") as f:
            return f.read()
    
    def put(self, text: str, voice_id: str, params: dict, audio: bytes):
        """キャッシュへの保存"""
        key = self._make_key(text, voice_id, params)
        
        # 容量チェック
        self._ensure_capacity(len(audio))
        
        audio_path = os.path.join(self.cache_dir, f"{key}.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio)
        
        self.index[key] = CacheEntry(
            audio_path=audio_path,
            text_hash=hashlib.md5(text.encode()).hexdigest(),
            voice_id=voice_id,
            params_hash=hashlib.md5(json.dumps(params).encode()).hexdigest(),
            created_at=time.time(),
            file_size=len(audio),
            last_accessed=time.time(),
        )
        
        self._save_index()
    
    def _ensure_capacity(self, needed_bytes: int):
        """容量確保（LRUで古いエントリを削除）"""
        current_size = sum(e.file_size for e in self.index.values())
        
        while current_size + needed_bytes > self.max_size_bytes and self.index:
            # 最もアクセスが古いエントリを削除
            oldest_key = min(
                self.index.keys(),
                key=lambda k: self.index[k].last_accessed
            )
            entry = self.index.pop(oldest_key)
            if os.path.exists(entry.audio_path):
                os.remove(entry.audio_path)
            current_size -= entry.file_size
    
    def _load_index(self):
        """インデックスの読み込み"""
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                data = json.load(f)
            for key, entry_data in data.items():
                self.index[key] = CacheEntry(**entry_data)
    
    def _save_index(self):
        """インデックスの保存"""
        data = {}
        for key, entry in self.index.items():
            data[key] = {
                "audio_path": entry.audio_path,
                "text_hash": entry.text_hash,
                "voice_id": entry.voice_id,
                "params_hash": entry.params_hash,
                "created_at": entry.created_at,
                "file_size": entry.file_size,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
            }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def stats(self) -> dict:
        """キャッシュ統計"""
        total_size = sum(e.file_size for e in self.index.values())
        total_accesses = sum(e.access_count for e in self.index.values())
        return {
            "entries": len(self.index),
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "usage_percent": total_size / self.max_size_bytes * 100,
            "total_accesses": total_accesses,
            "hit_rate_estimate": "N/A",
        }
```

### 10.2 バッチTTSプロセッサ

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor

@dataclass
class TTSJob:
    """TTS処理ジョブ"""
    job_id: str
    text: str
    voice_id: str
    params: dict
    priority: int = 0
    callback: Optional[Callable] = None
    status: str = "pending"
    result: Optional[bytes] = None
    error: Optional[str] = None
    created_at: float = 0.0
    completed_at: float = 0.0

class BatchTTSProcessor:
    """バッチTTS処理エンジン"""
    
    def __init__(
        self, 
        tts_client,
        cache_manager: TTSCacheManager,
        max_concurrent: int = 5,
        rate_limit_per_second: float = 10.0,
    ):
        self.tts_client = tts_client
        self.cache = cache_manager
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit_per_second
        self.jobs: dict[str, TTSJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._last_request_time = 0.0
    
    async def submit_batch(self, jobs: list[TTSJob]) -> list[TTSJob]:
        """バッチジョブの投入"""
        for job in jobs:
            job.created_at = time.time()
            self.jobs[job.job_id] = job
        
        # 優先度順にソート
        sorted_jobs = sorted(jobs, key=lambda j: -j.priority)
        
        # キャッシュチェック
        uncached_jobs = []
        for job in sorted_jobs:
            cached = self.cache.get(job.text, job.voice_id, job.params)
            if cached:
                job.result = cached
                job.status = "completed"
                job.completed_at = time.time()
            else:
                uncached_jobs.append(job)
        
        # 未キャッシュのジョブを並行処理
        if uncached_jobs:
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [
                self._process_with_semaphore(semaphore, job)
                for job in uncached_jobs
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return jobs
    
    async def _process_with_semaphore(self, semaphore, job: TTSJob):
        """セマフォ付きジョブ処理"""
        async with semaphore:
            await self._process_job(job)
    
    async def _process_job(self, job: TTSJob):
        """個別ジョブの処理"""
        try:
            # レートリミット
            await self._rate_limit_wait()
            
            job.status = "processing"
            
            # TTS実行（同期関数をスレッドプールで実行）
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.tts_client.text_to_speech,
                job.text,
                job.voice_id,
                job.params,
            )
            
            job.result = result
            job.status = "completed"
            job.completed_at = time.time()
            
            # キャッシュに保存
            self.cache.put(job.text, job.voice_id, job.params, result)
            
            # コールバック
            if job.callback:
                job.callback(job)
                
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = time.time()
    
    async def _rate_limit_wait(self):
        """レートリミット制御"""
        now = time.time()
        min_interval = 1.0 / self.rate_limit
        elapsed = now - self._last_request_time
        
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    def get_progress(self) -> dict:
        """進捗状況"""
        total = len(self.jobs)
        completed = sum(1 for j in self.jobs.values() if j.status == "completed")
        failed = sum(1 for j in self.jobs.values() if j.status == "failed")
        processing = sum(1 for j in self.jobs.values() if j.status == "processing")
        pending = sum(1 for j in self.jobs.values() if j.status == "pending")
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "pending": pending,
            "progress_percent": completed / total * 100 if total > 0 else 0,
        }
```

---

## 11. トラブルシューティングガイド

### 11.1 よくある問題と解決策

```
TTS トラブルシューティング
==================================================

問題1: 生成音声にノイズが入る
─────────────────────────────
原因:
  - 入力テキストに特殊文字や制御文字が含まれている
  - モデルの温度設定が高すぎる
  - GPU メモリ不足による計算精度の低下

解決策:
  1. テキスト前処理パイプラインで特殊文字を除去
  2. temperature を 0.5-0.7 に下げる
  3. バッチサイズを小さくする / FP32 に切り替え

問題2: 日本語のイントネーションが不自然
─────────────────────────────
原因:
  - アクセント辞書に登録されていない固有名詞
  - 文脈依存のアクセント変化に対応できていない
  - 学習データの地域的偏り

解決策:
  1. OpenJTalk のユーザー辞書にアクセント情報を追加
  2. SSMLで明示的にプロソディを制御
  3. ドメイン特化データでファインチューニング

問題3: 長文でモデルが途中で止まる
─────────────────────────────
原因:
  - コンテキスト長の制限超過
  - アテンション機構の破綻（長文で注意が散漫になる）
  - メモリリーク

解決策:
  1. 文単位で分割して合成→結合
  2. 各文の合成後にGPUキャッシュをクリア
  3. 分割ポイントで自然なポーズを挿入

問題4: 話者の声質が一貫しない
─────────────────────────────
原因:
  - stability 設定が低すぎる
  - 文ごとに独立して合成している
  - 話者埋め込みの不一致

解決策:
  1. stability を 0.6-0.8 に上げる
  2. 前の文の音声を参照プロンプトとして使用
  3. 同一の話者埋め込みベクトルを全文で共有

問題5: APIレートリミットに到達
─────────────────────────────
原因:
  - 短時間に大量のリクエストを送信
  - キャッシュが効いていない
  - リトライロジックが無限ループ

解決策:
  1. TTSCacheManager でキャッシュを実装
  2. Exponential Backoff でリトライ
  3. バッチ処理でリクエスト数を削減
==================================================
```

### 11.2 パフォーマンス最適化チェックリスト

```
TTS パフォーマンス最適化チェックリスト
==================================================

□ テキスト前処理
  □ 正規化パイプラインを適用しているか
  □ 文分割の最大長は適切か（100-200文字推奨）
  □ 不要な記号・空白は除去されているか

□ モデル選択
  □ 用途に適したモデルサイズか
  □ FP16/INT8 量子化を検討したか
  □ ONNX Runtime / TensorRT での最適化は可能か

□ キャッシュ戦略
  □ 同一テキスト+声のキャッシュを実装しているか
  □ キャッシュのTTLは適切か
  □ キャッシュのメモリ/ディスク上限を設定しているか

□ バッチ処理
  □ 複数リクエストをバッチ化しているか
  □ 非同期処理を活用しているか
  □ レートリミットを考慮しているか

□ ストリーミング
  □ First Token Latency を最小化しているか
  □ バッファサイズは適切か
  □ PCM/Opus など分割可能なフォーマットを使用しているか

□ GPU最適化（ローカルモデル）
  □ GPU メモリを適切に管理しているか
  □ torch.cuda.empty_cache() を適宜呼んでいるか
  □ Mixed Precision（AMP）を使用しているか

□ モニタリング
  □ 音声品質の定期評価を行っているか
  □ レイテンシーを監視しているか
  □ エラー率を追跡しているか
==================================================
```

---

## 12. 追加FAQ

### Q4: VITSのファインチューニングに必要なデータ量は？

最低でも1時間程度の高品質な音声データとその書き起こしが必要です。理想的には5-10時間のデータがあると品質が安定します。データ要件として、(1) サンプリングレート: 22050Hz以上、(2) フォーマット: WAV（無圧縮）推奨、(3) 環境: 反響や雑音が少ない録音環境、(4) テキスト: 正確な書き起こし（アクセント情報があるとなお良い）。学習時間はGPU（RTX 3090等）で10-24時間程度です。overfittingを避けるため、バリデーションセットを必ず用意し、定期的にMOS評価を行ってください。

### Q5: OpenAI TTSとElevenLabsのどちらを選ぶべきですか？

用途によって異なります。**OpenAI TTS**は (1) セットアップが簡単（既存のOpenAI APIキーで利用可能）、(2) ストリーミング対応が優秀、(3) コストが比較的安い、(4) 6種類の固定ボイスで安定した品質。**ElevenLabs**は (1) 音声クローニングに対応（任意の声を再現可能）、(2) 感情制御が詳細（stability, similarity, style パラメータ）、(3) ボイスライブラリが豊富、(4) 日本語対応がやや強い。プロトタイプや対話型アプリケーションにはOpenAI TTS、ナレーション品質が重要なコンテンツ制作やカスタムボイスが必要な場合はElevenLabsを推奨します。

### Q6: ブラウザ上でTTSを実行する方法は？

主に3つのアプローチがあります。(1) **Web Speech API**: ブラウザ内蔵のTTSで、追加ライブラリ不要。品質は端末依存で、`speechSynthesis.speak(new SpeechSynthesisUtterance("テスト"))` で簡単に利用可能。(2) **クラウドTTS API呼び出し**: バックエンドサーバー経由でOpenAI TTSやElevenLabsを呼び出し、生成された音声をWeb Audio APIで再生。(3) **ONNX Runtime Web**: VITSなどのモデルをONNX形式に変換し、ブラウザ内のWebGLやWebAssemblyで推論。オフライン動作が可能ですが、モデルサイズとレイテンシーに制限あり。多くのプロダクションアプリでは(2)のクラウドAPI呼び出しが品質・開発効率の点で最適です。

### Q7: TTSの音声にエフェクト（リバーブ、EQ等）を適用するには？

TTSで生成された音声は通常「ドライ」な音声のため、用途に応じた後処理が効果的です。Pythonでは`pedalboard`ライブラリ（Spotify製）が使いやすく、`Reverb`, `Compressor`, `LowpassFilter`, `HighpassFilter`などのエフェクトをチェーン的に適用できます。Webアプリケーションでは Web Audio API の`ConvolverNode`（リバーブ）、`BiquadFilterNode`（EQ）、`DynamicsCompressorNode`（コンプレッサー）が利用可能です。ポイントとして、EQでの低域カット（80Hz以下を除去）とハイシェルフブースト（8kHz+）は多くの場合TTSの明瞭さを改善します。

---

## 13. 参考文献（追加）

5. Ren, Y., et al. (2021). "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" — 非自己回帰型TTSの代表的モデル
6. Tan, X., et al. (2024). "NaturalSpeech 3: Zero-Shot Speech Synthesis with a Factorized Codec and Diffusion Models" — NaturalSpeech 3論文。ファクタ化コーデックによるゼロショットTTS
7. Le, M., et al. (2024). "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale" — Meta Voicebox。テキストガイド型多言語音声生成
8. Łajszczak, M., et al. (2024). "BASE TTS: Lessons from building a billion-parameter text-to-speech model on 100K hours of data" — Amazon BASE TTS。大規模TTS学習の知見
