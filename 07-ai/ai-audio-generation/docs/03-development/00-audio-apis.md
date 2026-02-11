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

---

## 2. 主要STT（音声認識）API比較

### 2.1 比較表：音声認識API

| 項目 | Google Cloud STT | Azure Speech | AWS Transcribe | OpenAI Whisper | Deepgram |
|------|-----------------|--------------|----------------|---------------|----------|
| 対応言語数 | 125+ | 100+ | 100+ | 97 | 36 |
| リアルタイム | 対応 | 対応 | 対応 | 非対応(API版) | 対応 |
| 話者分離 | 対応 | 対応 | 対応 | 非対応 | 対応 |
| カスタム語彙 | 対応 | 対応 | 対応 | 非対応 | 対応 |
| 日本語精度 | 高 | 高 | 中〜高 | 高 | 中 |
| 料金/分 | $0.006〜 | $0.0053〜 | $0.024 | $0.006 | $0.0043〜 |
| セルフホスト | 不可 | コンテナ可 | 不可 | OSS利用可 | 不可 |
| 最大音声長 | 480分 | 無制限(ストリーム) | 14,400分 | 25MB | 無制限 |

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
```

### 2.4 OpenAI Whisper API の実装

```python
# OpenAI Whisper API: シンプルで高精度な文字起こし
from openai import OpenAI

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
| 料金/100万文字 | $4(標準) | $4〜$16 | $4〜$16 | $15 | $3〜$99 |
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

class MultiProviderSTT:
    """フォールバック付きマルチプロバイダSTTクライアント"""

    def __init__(self):
        self.providers: list[STTProvider] = []
        self.cache: dict[str, dict] = {}
        self.metrics: dict[str, dict] = {}
        self.rate_limits: dict[str, dict] = {}

    def add_provider(
        self,
        provider: STTProvider,
        name: str,
        max_requests_per_min: int = 60,
    ):
        """プロバイダを優先順位順に追加"""
        self.providers.append(provider)
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
        for i, provider in enumerate(self.providers):
            name = type(provider).__name__
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

---

## 6. アンチパターン

### 6.1 アンチパターン：同期バッチ処理のみに依存

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

### 6.2 アンチパターン：APIキーのハードコード

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

---

## 7. FAQ

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

---

## 8. まとめ

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
- 音声AI基礎理論 — 音声信号処理の理論的背景
- AI音声アプリケーション設計 — エンドツーエンドの音声アプリ構築

---

## 参考文献

1. Google Cloud Speech-to-Text ドキュメント — https://cloud.google.com/speech-to-text/docs
2. Azure AI Speech Services ドキュメント — https://learn.microsoft.com/azure/ai-services/speech-service/
3. OpenAI Whisper API リファレンス — https://platform.openai.com/docs/guides/speech-to-text
4. Amazon Polly 開発者ガイド — https://docs.aws.amazon.com/polly/latest/dg/
5. Deepgram API ドキュメント — https://developers.deepgram.com/docs
