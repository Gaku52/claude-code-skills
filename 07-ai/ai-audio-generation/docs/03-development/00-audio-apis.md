# 音声AI API 比較・統合・活用ガイド

> Google Cloud Speech、Amazon Polly、Azure Speech Services、OpenAI Whisper など主要音声AI APIの特徴・料金・統合方法を体系的に解説し、最適な選定と実装を支援する。

---

## この章で学ぶこと

1. **主要音声AI APIの機能・料金・精度を比較**し、ユースケース別に最適なサービスを選定できる
2. **REST/gRPC/WebSocket各プロトコルでの統合パターン**を理解し、音声認識・合成を実装できる
3. **フォールバック・キャッシュ・レート制限**など本番運用で必要な設計手法を習得する

---

## 1. 音声AI APIの全体像

### 1.1 主要サービスのカテゴリ

```
+----------------------------------------------------------+
|                   音声AI APIエコシステム                    |
+----------------------------------------------------------+
|                                                          |
|  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  |
|  │  音声認識     │  │  音声合成     │  │  音声分析      │  |
|  │  (STT)       │  │  (TTS)       │  │  (Analysis)   │  |
|  ├──────────────┤  ├──────────────┤  ├───────────────┤  |
|  │ Google STT   │  │ Amazon Polly │  │ 話者識別       │  |
|  │ Azure Speech │  │ Azure TTS    │  │ 感情分析       │  |
|  │ AWS Transcr. │  │ Google TTS   │  │ キーワード検出  │  |
|  │ Whisper API  │  │ ElevenLabs   │  │ 言語検出       │  |
|  │ Deepgram     │  │ OpenAI TTS   │  │ トピック分類    │  |
|  └──────────────┘  └──────────────┘  └───────────────┘  |
+----------------------------------------------------------+
```

### 1.2 APIの通信パターン

```
+-------------------+     +-------------------+
|  クライアント      |     |  音声AI API       |
+-------------------+     +-------------------+
|                   |     |                   |
| [REST/HTTP]       |────>| バッチ処理         |
|  音声ファイル送信  |<────| 結果JSON返却       |
|                   |     |                   |
| [WebSocket]       |<===>| リアルタイム処理    |
|  ストリーミング    |<===>| 逐次結果返却       |
|                   |     |                   |
| [gRPC]            |<===>| 高速双方向通信     |
|  バイナリ最適化    |<===>| Protocol Buffers  |
+-------------------+     +-------------------+
```

### 1.3 API選定のフローチャート

```
音声AI API 選定ガイド
==================================================

Q1: リアルタイム処理が必要か？
    │
    ├── Yes → Q2: 遅延要件は？
    │         ├── <100ms → Deepgram (WebSocket)
    │         ├── <300ms → Azure Speech / Google STT
    │         └── <500ms → AWS Transcribe Streaming
    │
    └── No → Q3: 何が重要か？
              ├── 精度最優先 → Whisper API / Google STT
              ├── コスト最優先 → Deepgram / Whisper OSS
              ├── カスタマイズ → Azure Custom Speech
              └── オフライン → Whisper / faster-whisper

Q4: TTS（音声合成）も必要か？
    ├── 日本語品質重視 → Azure Speech TTS
    ├── 音声クローン → ElevenLabs
    ├── SSML制御 → Amazon Polly / Azure
    └── シンプルAPI → OpenAI TTS
==================================================
```

---

## 2. 主要STT（音声認識）API比較

### 2.1 比較表：音声認識API

| 項目 | Google Cloud STT | Azure Speech | AWS Transcribe | OpenAI Whisper | Deepgram |
|------|-----------------|--------------|----------------|---------------|----------|
| 対応言語数 | 125+ | 100+ | 100+ | 97 | 36 |
| リアルタイム | 対応 | 対応 | 対応 | 非対応(API版) | 対応 |
| 話者分離 | 対応 | 対応 | 対応 | 非対応 | 対応 |
| カスタム語彙 | 対応 | 対応 | 対応 | 非対応 | 対応 |
| 日本語精度 | 高 | 高 | 中~高 | 高 | 中 |
| 料金/分 | $0.006~ | $0.0053~ | $0.024 | $0.006 | $0.0043~ |
| セルフホスト | 不可 | コンテナ可 | 不可 | OSS利用可 | 不可 |
| 最大音声長 | 480分 | 無制限(ストリーム) | 14,400分 | 25MB | 無制限 |
| 感情分析 | 非対応 | 非対応 | 非対応 | 非対応 | 対応 |
| 要約生成 | 非対応 | 非対応 | 非対応 | 非対応 | 対応 |

### 2.2 Google Cloud Speech-to-Text の実装

```python
# Google Cloud Speech-to-Text: 同期認識
from google.cloud import speech_v1

def transcribe_audio_sync(audio_path: str, language: str = "ja-JP") -> str:
    """音声ファイルを同期的に文字起こしする"""
    client = speech_v1.SpeechClient()

    with open(audio_path, "rb") as f:
        audio_content = f.read()

    audio = speech_v1.RecognitionAudio(content=audio_content)
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language,
        # 高精度オプション
        enable_automatic_punctuation=True,  # 自動句読点
        enable_word_time_offsets=True,       # 単語タイムスタンプ
        model="latest_long",                 # 長時間音声用モデル
        use_enhanced=True,                   # 強化モデル使用
    )

    response = client.recognize(config=config, audio=audio)

    results = []
    for result in response.results:
        alt = result.alternatives[0]
        results.append({
            "transcript": alt.transcript,
            "confidence": alt.confidence,
            "words": [
                {
                    "word": w.word,
                    "start": w.start_time.total_seconds(),
                    "end": w.end_time.total_seconds(),
                }
                for w in alt.words
            ],
        })
    return results


# Google Cloud Speech-to-Text V2: 最新API
from google.cloud import speech_v2 as speech

def transcribe_v2(
    audio_path: str,
    project_id: str,
    language: str = "ja-JP",
) -> list[dict]:
    """V2 APIで文字起こし（より多機能）"""
    client = speech.SpeechClient()

    with open(audio_path, "rb") as f:
        audio_content = f.read()

    config = speech.RecognitionConfig(
        auto_decoding_config=speech.AutoDetectDecodingConfig(),
        language_codes=[language],
        model="long",
        features=speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            enable_word_confidence=True,
            multi_channel_mode=speech.RecognitionFeatures.MultiChannelMode.SEPARATE_RECOGNITION_PER_CHANNEL,
        ),
    )

    request = speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/global/recognizers/_",
        config=config,
        content=audio_content,
    )

    response = client.recognize(request=request)

    results = []
    for result in response.results:
        alt = result.alternatives[0]
        results.append({
            "transcript": alt.transcript,
            "confidence": alt.confidence,
        })

    return results


# Google Cloud STT: 非同期処理（長時間音声向け）
def transcribe_async(
    gcs_uri: str,
    language: str = "ja-JP",
) -> list[dict]:
    """GCS上の長時間音声を非同期で文字起こし"""
    client = speech_v1.SpeechClient()

    audio = speech_v1.RecognitionAudio(uri=gcs_uri)
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        model="latest_long",
        # 話者分離
        diarization_config=speech_v1.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=2,
            max_speaker_count=6,
        ),
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    print("処理中... (数分かかる場合があります)")

    response = operation.result(timeout=3600)  # 最大1時間待機

    results = []
    for result in response.results:
        alt = result.alternatives[0]
        results.append({
            "transcript": alt.transcript,
            "confidence": alt.confidence,
        })

    return results
```

### 2.3 Azure Speech Services の実装

```python
# Azure Speech Services: リアルタイムストリーミング認識
import azure.cognitiveservices.speech as speechsdk
import asyncio
from typing import Callable

class AzureRealtimeTranscriber:
    """Azure Speech Servicesを使ったリアルタイム文字起こし"""

    def __init__(self, subscription_key: str, region: str = "japaneast"):
        self.config = speechsdk.SpeechConfig(
            subscription=subscription_key,
            region=region,
        )
        self.config.speech_recognition_language = "ja-JP"
        # 高精度設定
        self.config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "15000"
        )
        self.config.enable_dictation()

    def transcribe_from_microphone(
        self, on_recognized: Callable[[str], None]
    ):
        """マイク入力からリアルタイム文字起こし"""
        audio_config = speechsdk.AudioConfig(
            use_default_microphone=True
        )
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.config,
            audio_config=audio_config,
        )

        # イベントハンドラ登録
        recognizer.recognized.connect(
            lambda evt: on_recognized(evt.result.text)
        )
        recognizer.session_stopped.connect(
            lambda evt: print("セッション終了")
        )

        recognizer.start_continuous_recognition()
        return recognizer  # stop_continuous_recognition()で停止

    def transcribe_from_file(self, file_path: str) -> list[dict]:
        """音声ファイルから文字起こし（話者分離付き）"""
        audio_config = speechsdk.AudioConfig(filename=file_path)
        # 会話文字起こし（話者分離対応）
        conversation_transcriber = speechsdk.ConversationTranscriber(
            speech_config=self.config,
            audio_config=audio_config,
        )

        results = []
        done = asyncio.Event()

        def on_transcribed(evt):
            results.append({
                "speaker": evt.result.speaker_id,
                "text": evt.result.text,
                "offset": evt.result.offset,
            })

        conversation_transcriber.transcribed.connect(on_transcribed)
        conversation_transcriber.session_stopped.connect(
            lambda _: done.set()
        )

        conversation_transcriber.start_transcribing_async()
        done.wait()
        return results

    def transcribe_with_translation(
        self,
        file_path: str,
        source_lang: str = "ja-JP",
        target_langs: list[str] = ["en"],
    ) -> dict:
        """音声翻訳（STT + 翻訳の同時実行）"""
        translation_config = speechsdk.translation.SpeechTranslationConfig(
            subscription=self.config.subscription_key,
            region=self.config.region,
        )
        translation_config.speech_recognition_language = source_lang
        for lang in target_langs:
            translation_config.add_target_language(lang)

        audio_config = speechsdk.AudioConfig(filename=file_path)
        recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=translation_config,
            audio_config=audio_config,
        )

        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.TranslatedSpeech:
            return {
                "source_text": result.text,
                "translations": {
                    lang: result.translations[lang]
                    for lang in target_langs
                },
            }
        return {"error": str(result.reason)}
```

### 2.4 OpenAI Whisper API の実装

```python
# OpenAI Whisper API: シンプルで高精度な文字起こし
from openai import OpenAI
from pathlib import Path

def transcribe_with_whisper(
    audio_path: str,
    language: str = "ja",
    response_format: str = "verbose_json",
) -> dict:
    """Whisper APIで文字起こし"""
    client = OpenAI()

    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format=response_format,
            # タイムスタンプ粒度: segment or word
            timestamp_granularities=["word", "segment"],
        )

    return {
        "text": result.text,
        "language": result.language,
        "duration": result.duration,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for s in result.segments
        ],
        "words": [
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
            }
            for w in result.words
        ],
    }


def translate_with_whisper(audio_path: str) -> dict:
    """Whisper APIで音声を英語に翻訳"""
    client = OpenAI()

    with open(audio_path, "rb") as audio_file:
        result = client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
        )

    return {
        "text": result.text,
        "source_language": result.language,
        "duration": result.duration,
    }


def transcribe_large_file(
    audio_path: str,
    chunk_duration_ms: int = 600000,  # 10分
    language: str = "ja",
) -> list[dict]:
    """
    大容量ファイルの分割文字起こし
    Whisper APIのファイルサイズ制限（25MB）を超える場合に使用
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    chunks = []

    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_path = f"/tmp/whisper_chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3", bitrate="64k")

        result = transcribe_with_whisper(chunk_path, language=language)
        result["chunk_start_ms"] = i
        result["chunk_end_ms"] = min(i + chunk_duration_ms, len(audio))
        chunks.append(result)

        Path(chunk_path).unlink()  # 一時ファイル削除

    return chunks
```

### 2.5 Deepgram の実装

```python
from deepgram import DeepgramClient, PrerecordedOptions, LiveOptions
import asyncio
import json

class DeepgramSTT:
    """Deepgramによる高機能文字起こし"""

    def __init__(self, api_key: str):
        self.client = DeepgramClient(api_key)

    def transcribe_file(
        self,
        audio_path: str,
        model: str = "nova-2",
        language: str = "ja",
    ) -> dict:
        """ファイルの文字起こし（全機能活用）"""
        with open(audio_path, "rb") as f:
            buffer_data = f.read()

        payload = {"buffer": buffer_data}

        options = PrerecordedOptions(
            model=model,
            language=language,
            smart_format=True,
            punctuate=True,
            diarize=True,
            utterances=True,
            detect_language=True,
            paragraphs=True,
            summarize="v2",
            topics=True,
            intents=True,
            sentiment=True,
        )

        response = self.client.listen.prerecorded.v("1").transcribe_file(
            payload, options
        )

        result = response.to_dict()
        channel = result["results"]["channels"][0]["alternatives"][0]

        return {
            "transcript": channel["transcript"],
            "confidence": channel["confidence"],
            "words": channel.get("words", []),
            "paragraphs": channel.get("paragraphs"),
            "summaries": result["results"].get("summary"),
            "topics": result["results"].get("topics"),
            "sentiments": result["results"].get("sentiments"),
        }

    async def transcribe_stream(
        self,
        audio_stream,
        on_result,
        model: str = "nova-2",
        language: str = "ja",
    ):
        """ストリーミング文字起こし"""
        options = LiveOptions(
            model=model,
            language=language,
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

    def transcribe_url(self, audio_url: str) -> dict:
        """URLからの文字起こし（ファイルアップロード不要）"""
        payload = {"url": audio_url}

        options = PrerecordedOptions(
            model="nova-2",
            language="ja",
            smart_format=True,
            diarize=True,
        )

        response = self.client.listen.prerecorded.v("1").transcribe_url(
            payload, options
        )

        return response.to_dict()
```

---

## 3. 主要TTS（音声合成）API比較

### 3.1 比較表：音声合成API

| 項目 | Amazon Polly | Azure TTS | Google TTS | OpenAI TTS | ElevenLabs |
|------|-------------|-----------|------------|------------|------------|
| 音声数 | 60+ | 400+ | 220+ | 6 | カスタム無制限 |
| SSML対応 | 対応 | 対応 | 対応 | 非対応 | 部分対応 |
| Neural音声 | 対応 | 対応 | 対応 | 標準 | 標準 |
| 音声クローン | 非対応 | カスタム可 | カスタム可 | 非対応 | 対応 |
| 日本語音声数 | 4 | 20+ | 10+ | 6(多言語) | カスタム |
| 料金/100万文字 | $4(標準) | $4~$16 | $4~$16 | $15 | $3~$99 |
| リアルタイム | 対応 | 対応 | 対応 | 対応 | 対応 |
| 感情表現 | 限定的 | 豊富 | 限定的 | 自動 | 豊富 |

### 3.2 Amazon Polly の実装

```python
# Amazon Polly: SSML対応の音声合成
import boto3
from contextlib import closing

class PollyTTSEngine:
    """Amazon Polly音声合成エンジン"""

    def __init__(self, region: str = "ap-northeast-1"):
        self.client = boto3.client("polly", region_name=region)

    def synthesize(
        self,
        text: str,
        voice_id: str = "Mizuki",  # 日本語女性
        engine: str = "neural",
        output_format: str = "mp3",
    ) -> bytes:
        """テキストから音声を合成"""
        response = self.client.synthesize_speech(
            Text=text,
            VoiceId=voice_id,
            Engine=engine,
            OutputFormat=output_format,
            LanguageCode="ja-JP",
        )

        with closing(response["AudioStream"]) as stream:
            return stream.read()

    def synthesize_ssml(self, ssml: str, voice_id: str = "Mizuki") -> bytes:
        """SSML記法で細かい制御を行った音声合成"""
        ssml_text = f"""
        <speak>
            <prosody rate="90%" pitch="+5%">
                {ssml}
            </prosody>
            <break time="500ms"/>
            <emphasis level="strong">重要なポイント</emphasis>
        </speak>
        """
        response = self.client.synthesize_speech(
            Text=ssml_text,
            TextType="ssml",
            VoiceId=voice_id,
            Engine="neural",
            OutputFormat="mp3",
        )
        with closing(response["AudioStream"]) as stream:
            return stream.read()

    def synthesize_long_text(
        self,
        text: str,
        voice_id: str = "Mizuki",
        s3_bucket: str = "my-audio-bucket",
        s3_key_prefix: str = "tts-output/",
    ) -> str:
        """長文テキストの非同期合成（S3出力）"""
        response = self.client.start_speech_synthesis_task(
            Text=text,
            VoiceId=voice_id,
            Engine="neural",
            OutputFormat="mp3",
            OutputS3BucketName=s3_bucket,
            OutputS3KeyPrefix=s3_key_prefix,
            LanguageCode="ja-JP",
        )
        task_id = response["SynthesisTask"]["TaskId"]
        return task_id

    def list_japanese_voices(self) -> list[dict]:
        """利用可能な日本語音声一覧を取得"""
        response = self.client.describe_voices(LanguageCode="ja-JP")
        return [
            {
                "id": v["Id"],
                "name": v["Name"],
                "gender": v["Gender"],
                "engines": v["SupportedEngines"],
            }
            for v in response["Voices"]
        ]
```

### 3.3 OpenAI TTS の実装

```python
from openai import OpenAI
from pathlib import Path

class OpenAITTSEngine:
    """OpenAI TTS音声合成エンジン"""

    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def __init__(self):
        self.client = OpenAI()

    def synthesize(
        self,
        text: str,
        voice: str = "nova",
        model: str = "tts-1",  # tts-1 or tts-1-hd
        speed: float = 1.0,
        output_path: str = "output.mp3",
    ) -> str:
        """テキストから音声を合成"""
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            speed=speed,
            response_format="mp3",  # mp3, opus, aac, flac, wav, pcm
        )

        response.stream_to_file(output_path)
        return output_path

    def synthesize_streaming(
        self,
        text: str,
        voice: str = "nova",
        model: str = "tts-1",
    ):
        """ストリーミング音声合成（低遅延）"""
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="opus",
        )

        # チャンク単位でストリーミング
        for chunk in response.iter_bytes(chunk_size=4096):
            yield chunk

    def synthesize_batch(
        self,
        texts: list[str],
        voice: str = "nova",
        output_dir: str = "./tts_output",
    ) -> list[str]:
        """複数テキストの一括合成"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        for i, text in enumerate(texts):
            file_path = str(output_path / f"speech_{i:04d}.mp3")
            self.synthesize(text, voice=voice, output_path=file_path)
            results.append(file_path)

        return results
```

### 3.4 ElevenLabs の実装

```python
from elevenlabs import ElevenLabs, VoiceSettings

class ElevenLabsTTS:
    """ElevenLabs音声合成（音声クローン対応）"""

    def __init__(self, api_key: str):
        self.client = ElevenLabs(api_key=api_key)

    def synthesize(
        self,
        text: str,
        voice_id: str = "pNInz6obpgDQGcFmaJgB",  # Adam
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.5,
    ) -> bytes:
        """テキストから音声を合成"""
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=True,
            ),
        )
        return b"".join(audio)

    def clone_voice(
        self,
        name: str,
        description: str,
        audio_files: list[str],
    ) -> str:
        """音声クローンの作成"""
        files = []
        for path in audio_files:
            with open(path, "rb") as f:
                files.append(f.read())

        voice = self.client.voices.add(
            name=name,
            description=description,
            files=files,
        )
        return voice.voice_id

    def list_voices(self) -> list[dict]:
        """利用可能な音声一覧"""
        voices = self.client.voices.get_all()
        return [
            {
                "voice_id": v.voice_id,
                "name": v.name,
                "category": v.category,
                "labels": v.labels,
            }
            for v in voices.voices
        ]
```

---

## 4. マルチプロバイダ統合アーキテクチャ

### 4.1 フォールバック付き統合クライアント

```
┌────────────────────────────────────────────────┐
│           統合音声AIクライアント                  │
├────────────────────────────────────────────────┤
│                                                │
│  Request ──> [ルーター] ──> Primary Provider    │
│                  │              │               │
│                  │         (失敗時)              │
│                  │              v               │
│                  └───> Fallback Provider        │
│                              │                 │
│                         (失敗時)                │
│                              v                 │
│                       Local Fallback           │
│                       (Whisper OSS等)           │
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ キャッシュ │  │レート制限 │  │ メトリクス│    │
│  └──────────┘  └──────────┘  └──────────┘    │
└────────────────────────────────────────────────┘
```

```python
# マルチプロバイダ統合クライアント
from abc import ABC, abstractmethod
from typing import Optional
import time
import hashlib
import json

class STTProvider(ABC):
    """音声認識プロバイダの抽象基底クラス"""

    @abstractmethod
    def transcribe(self, audio: bytes, language: str) -> dict:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

class TTSProvider(ABC):
    """音声合成プロバイダの抽象基底クラス"""

    @abstractmethod
    def synthesize(self, text: str, voice: str) -> bytes:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

class MultiProviderSTT:
    """フォールバック付きマルチプロバイダSTTクライアント"""

    def __init__(self):
        self.providers: list[tuple[str, STTProvider]] = []
        self.cache: dict[str, dict] = {}
        self.metrics: dict[str, dict] = {}
        self.rate_limits: dict[str, dict] = {}

    def add_provider(
        self,
        name: str,
        provider: STTProvider,
        max_requests_per_min: int = 60,
        priority: int = 0,
    ):
        """プロバイダを優先順位順に追加"""
        self.providers.append((name, provider))
        self.providers.sort(key=lambda x: priority)
        self.metrics[name] = {
            "success": 0, "failure": 0, "total_latency": 0.0,
        }
        self.rate_limits[name] = {
            "max": max_requests_per_min,
            "requests": [],
        }

    def _check_rate_limit(self, name: str) -> bool:
        """レート制限チェック"""
        limit = self.rate_limits[name]
        now = time.time()
        # 1分以内のリクエストのみ保持
        limit["requests"] = [
            t for t in limit["requests"] if now - t < 60
        ]
        return len(limit["requests"]) < limit["max"]

    def _get_cache_key(self, audio: bytes, language: str) -> str:
        """キャッシュキーを生成"""
        audio_hash = hashlib.sha256(audio).hexdigest()
        return f"{audio_hash}:{language}"

    def transcribe(
        self,
        audio: bytes,
        language: str = "ja-JP",
        use_cache: bool = True,
    ) -> dict:
        """フォールバック付き文字起こし"""
        # キャッシュ確認
        cache_key = self._get_cache_key(audio, language)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        last_error = None
        for name, provider in self.providers:
            if not provider.is_available():
                continue
            if not self._check_rate_limit(name):
                continue

            try:
                start = time.time()
                result = provider.transcribe(audio, language)
                latency = time.time() - start

                # メトリクス更新
                self.metrics[name]["success"] += 1
                self.metrics[name]["total_latency"] += latency
                self.rate_limits[name]["requests"].append(time.time())

                # キャッシュ保存
                result["provider"] = name
                result["latency"] = latency
                if use_cache:
                    self.cache[cache_key] = result

                return result

            except Exception as e:
                self.metrics[name]["failure"] += 1
                last_error = e
                continue

        raise RuntimeError(
            f"全プロバイダで文字起こし失敗: {last_error}"
        )

    def get_metrics(self) -> dict:
        """メトリクスの取得"""
        result = {}
        for name, m in self.metrics.items():
            total = m["success"] + m["failure"]
            result[name] = {
                "total": total,
                "success_rate": m["success"] / total if total > 0 else 0,
                "avg_latency": (
                    m["total_latency"] / m["success"]
                    if m["success"] > 0 else 0
                ),
            }
        return result
```

### 4.2 統合TTS クライアント

```python
class MultiProviderTTS:
    """フォールバック付きマルチプロバイダTTSクライアント"""

    def __init__(self):
        self.providers: dict[str, TTSProvider] = {}
        self.fallback_order: list[str] = []
        self._cache: dict[str, bytes] = {}

    def register(self, name: str, provider: TTSProvider):
        self.providers[name] = provider
        self.fallback_order.append(name)

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        provider: Optional[str] = None,
        use_cache: bool = True,
    ) -> bytes:
        """音声合成（フォールバック付き）"""
        cache_key = hashlib.sha256(
            f"{text}:{voice}:{provider}".encode()
        ).hexdigest()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        providers_to_try = (
            [provider] if provider
            else self.fallback_order
        )

        last_error = None
        for name in providers_to_try:
            if name not in self.providers:
                continue
            try:
                p = self.providers[name]
                if not p.is_available():
                    continue
                audio = p.synthesize(text, voice or "default")
                if use_cache:
                    self._cache[cache_key] = audio
                return audio
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"全TTSプロバイダ失敗: {last_error}")
```

---

## 5. ストリーミング処理パターン

### 5.1 WebSocketによるリアルタイム処理

```python
# WebSocketベースのリアルタイム音声認識サーバー
import asyncio
import websockets
import json
from google.cloud import speech_v1

async def audio_stream_handler(websocket, path):
    """WebSocketで音声ストリームを受信し、逐次文字起こし結果を返す"""
    client = speech_v1.SpeechClient()

    config = speech_v1.StreamingRecognitionConfig(
        config=speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ja-JP",
            enable_automatic_punctuation=True,
        ),
        interim_results=True,  # 中間結果も返す
    )

    async def request_generator():
        """音声チャンクをgRPCリクエストに変換"""
        yield speech_v1.StreamingRecognizeRequest(
            streaming_config=config
        )
        async for message in websocket:
            if isinstance(message, bytes):
                yield speech_v1.StreamingRecognizeRequest(
                    audio_content=message
                )

    # ストリーミング認識実行
    responses = client.streaming_recognize(
        requests=request_generator()
    )

    for response in responses:
        for result in response.results:
            msg = {
                "is_final": result.is_final,
                "transcript": result.alternatives[0].transcript,
                "confidence": (
                    result.alternatives[0].confidence
                    if result.is_final else None
                ),
            }
            await websocket.send(json.dumps(msg, ensure_ascii=False))

# サーバー起動
async def main():
    async with websockets.serve(audio_stream_handler, "0.0.0.0", 8765):
        await asyncio.Future()  # 永続実行
```

### 5.2 FastAPI によるストリーミングAPIサーバー

```python
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import io

app = FastAPI(title="音声AI API Gateway")

@app.post("/api/v1/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    language: str = "ja",
    provider: str = "whisper",
):
    """音声ファイルを文字起こし"""
    audio_bytes = await file.read()

    stt_client = MultiProviderSTT()
    # プロバイダー登録は省略

    result = stt_client.transcribe(audio_bytes, language)
    return result

@app.post("/api/v1/tts")
async def text_to_speech(
    text: str,
    voice: str = "nova",
    provider: str = "openai",
):
    """テキストを音声合成"""
    tts_client = MultiProviderTTS()
    audio_bytes = tts_client.synthesize(text, voice, provider)

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=speech.mp3"},
    )

@app.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket):
    """WebSocketによるストリーミング文字起こし"""
    await websocket.accept()

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            # STT処理（省略）
            result = {"text": "...", "is_final": True}
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass

@app.get("/api/v1/metrics")
async def get_metrics():
    """APIメトリクスを取得"""
    stt_client = MultiProviderSTT()
    return stt_client.get_metrics()
```

---

## 6. コスト最適化

### 6.1 コスト比較シミュレーション

```python
class CostCalculator:
    """音声API利用コストの計算"""

    # 料金テーブル（2024年時点の参考価格）
    PRICING = {
        "google_stt": {
            "standard": 0.006,      # $/分
            "enhanced": 0.009,
            "data_logging_opt_in": 0.004,
        },
        "azure_stt": {
            "standard": 0.0053,     # $/分（東日本リージョン）
            "custom": 0.0106,
        },
        "aws_transcribe": {
            "standard": 0.024,      # $/分
            "medical": 0.075,
        },
        "whisper_api": {
            "standard": 0.006,      # $/分
        },
        "deepgram": {
            "nova_2": 0.0043,       # $/分
            "enhanced": 0.0145,
        },
        "openai_tts": {
            "tts_1": 15.0,          # $/100万文字
            "tts_1_hd": 30.0,
        },
        "amazon_polly": {
            "standard": 4.0,        # $/100万文字
            "neural": 16.0,
        },
    }

    def estimate_stt_cost(
        self,
        provider: str,
        tier: str,
        audio_minutes: float,
    ) -> float:
        """STTコストの見積もり"""
        rate = self.PRICING.get(provider, {}).get(tier, 0)
        return rate * audio_minutes

    def estimate_tts_cost(
        self,
        provider: str,
        tier: str,
        character_count: int,
    ) -> float:
        """TTSコストの見積もり"""
        rate = self.PRICING.get(provider, {}).get(tier, 0)
        return rate * (character_count / 1_000_000)

    def compare_providers(
        self,
        audio_minutes: float,
        monthly: bool = True,
    ) -> dict:
        """プロバイダ間のコスト比較"""
        multiplier = 30 if monthly else 1
        total_minutes = audio_minutes * multiplier

        comparison = {}
        for provider, tiers in self.PRICING.items():
            if any(k in provider for k in ["stt", "transcribe", "whisper", "deepgram"]):
                for tier, rate in tiers.items():
                    key = f"{provider}_{tier}"
                    comparison[key] = {
                        "rate_per_min": rate,
                        "total_cost": rate * total_minutes,
                        "total_minutes": total_minutes,
                    }

        # コスト順にソート
        return dict(sorted(
            comparison.items(),
            key=lambda x: x[1]["total_cost"]
        ))

# 使用例
calc = CostCalculator()
comparison = calc.compare_providers(
    audio_minutes=60,  # 1日60分
    monthly=True,       # 月間コスト
)
for provider, cost in comparison.items():
    print(f"{provider}: ${cost['total_cost']:.2f}/月")
```

---

## 7. アンチパターン

### 7.1 アンチパターン：同期バッチ処理のみに依存

```python
# NG: 長時間音声をすべてメモリに読み込んで同期処理
def bad_transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        huge_audio = f.read()  # 数GBのファイルでもメモリに全読込
    # タイムアウトのリスク、メモリ不足のリスク
    result = client.recognize(audio=huge_audio)
    return result

# OK: チャンク分割 + 非同期処理
async def good_transcribe(audio_path: str) -> list[str]:
    chunks = split_audio(audio_path, chunk_seconds=30)
    tasks = [transcribe_chunk(c) for c in chunks]
    return await asyncio.gather(*tasks)
```

**問題点**: 大容量音声ファイルをメモリに全読込するとOOM（メモリ不足）やタイムアウトが発生する。ストリーミングまたはチャンク分割で処理すること。

### 7.2 アンチパターン：APIキーのハードコード

```python
# NG: ソースコードにAPIキーを直接記述
client = SpeechClient(api_key="sk-1234567890abcdef")

# OK: 環境変数またはシークレットマネージャーを使用
import os
from google.cloud import secretmanager

def get_api_key(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/my-project/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")
```

**問題点**: APIキーがバージョン管理に含まれるとセキュリティリスク。環境変数、Secret Manager、Vaultなどを使って安全に管理する。

### 7.3 アンチパターン：レート制限の無視

```python
# NG: レート制限を考慮せず高速ループ
def bad_batch_transcribe(files):
    results = []
    for f in files:
        results.append(api.transcribe(f))  # レート制限でエラー
    return results

# OK: レート制限を考慮した処理
import time
from functools import wraps

def rate_limited(max_per_second=1):
    min_interval = 1.0 / max_per_second
    last_time = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_time[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_time[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limited(max_per_second=5)
def good_transcribe(audio_path):
    return api.transcribe(audio_path)
```

---

## 8. FAQ

### Q1: 日本語音声認識で最も精度が高いAPIは？

**A**: 一般的な会話音声であればOpenAI Whisperが高い精度を示す。ただし、専門用語が多い場合はGoogle Cloud STTやAzure Speechのカスタム語彙機能を使うことで精度が向上する。医療・法律・金融など特定ドメインでは、カスタムモデル訓練が可能なAzureが有利。

### Q2: リアルタイム音声認識の遅延はどの程度か？

**A**: 主要APIのリアルタイム認識レイテンシは以下のとおり。

| API | 平均遅延 | 最小遅延 | 備考 |
|-----|---------|---------|------|
| Google STT | 200-400ms | 100ms | gRPC使用時 |
| Azure Speech | 150-300ms | 80ms | 東日本リージョン |
| Deepgram | 100-250ms | 50ms | WebSocket使用時 |
| AWS Transcribe | 300-500ms | 200ms | WebSocket使用時 |

### Q3: 音声合成で最も自然な日本語音声はどれか？

**A**: Azure Speech Servicesが日本語Neural音声の種類が最も豊富（20+）で、感情表現やスタイル切替も可能。ElevenLabsは音声クローンの品質が高く、特定話者の再現に優れる。Amazon Pollyは安定性と低コストが強み。

### Q4: APIの料金を最小化するには？

**A**: (1) キャッシュを活用し同じ音声の再処理を避ける、(2) 音声を適切にトリミングして無音部分を送信しない、(3) バッチ処理可能なものはリアルタイムAPIを使わない、(4) 短い音声にはWhisper APIの従量課金が有利。

### Q5: オフラインで使える音声認識は？

**A**: OpenAI Whisperのオープンソース版をローカルで実行する方法が最も実用的。faster-whisperを使えばCTranslate2最適化により2-4倍高速に動作する。NVIDIA GPUがあれば `compute_type="float16"` でさらに高速化可能。CPUのみの環境では `compute_type="int8"` で量子化すると実用的な速度になる。

### Q6: 音声合成のSSML記法はどう使うか？

**A**: SSML（Speech Synthesis Markup Language）はXMLベースで音声合成を細かく制御する規格。主なタグ: `<prosody>` で速度・ピッチ・音量を制御、`<break>` でポーズ挿入、`<emphasis>` で強調、`<say-as>` で読み方指定（日付・数値等）。Amazon PollyとAzure Speechが最も豊富なSSMLサポートを提供している。

---

## 9. まとめ

| カテゴリ | ポイント |
|---------|---------|
| STT選定 | 精度重視ならWhisper、リアルタイムならAzure/Deepgram、カスタマイズならGoogle |
| TTS選定 | 日本語品質ならAzure、低コストならPolly、クローンならElevenLabs |
| 統合設計 | マルチプロバイダ+フォールバックで可用性確保 |
| リアルタイム | WebSocket/gRPCストリーミングで低レイテンシ実現 |
| コスト最適化 | キャッシュ、チャンク分割、適切なAPI選択で削減 |
| セキュリティ | APIキーはSecret Manager管理、音声データは暗号化転送 |
| 運用監視 | レイテンシ・エラー率・コストのメトリクスを常時監視 |

---

## 次に読むべきガイド

- [01-audio-processing.md](./01-audio-processing.md) — 音声処理パイプラインの実装
- [02-real-time-audio.md](./02-real-time-audio.md) — リアルタイム音声処理
- [../00-fundamentals/03-stt-technologies.md](../00-fundamentals/03-stt-technologies.md) — STT技術の詳細

---

## 参考文献

1. Google Cloud Speech-to-Text ドキュメント — https://cloud.google.com/speech-to-text/docs
2. Azure AI Speech Services ドキュメント — https://learn.microsoft.com/azure/ai-services/speech-service/
3. OpenAI Whisper API リファレンス — https://platform.openai.com/docs/guides/speech-to-text
4. Amazon Polly 開発者ガイド — https://docs.aws.amazon.com/polly/latest/dg/
5. Deepgram API ドキュメント — https://developers.deepgram.com/docs
6. ElevenLabs API ドキュメント — https://docs.elevenlabs.io/
