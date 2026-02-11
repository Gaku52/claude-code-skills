# 音声アシスタント — カスタムウェイクワード、対話AI

> カスタム音声アシスタントの構築技術（ウェイクワード検出、対話管理、音声インターフェース設計）を解説する

## この章で学ぶこと

1. 音声アシスタントの全体アーキテクチャとパイプライン設計
2. ウェイクワード検出、対話管理、マルチターン会話の実装
3. LLM統合によるインテリジェント音声対話システムの構築

---

## 1. 音声アシスタントのアーキテクチャ

### 1.1 全体パイプライン

```
音声アシスタント パイプライン
==================================================

          常時リスニング
               │
               ▼
┌──────────────────────┐
│  Wake Word Detection  │  「Hey, アシスタント」
│  (ウェイクワード検出)  │  Porcupine / OpenWakeWord
└───────────┬──────────┘
            │ 検出!
            ▼
┌──────────────────────┐
│  VAD + 録音           │  音声区間を検出して録音
│  (Voice Activity Det.)│  webrtcvad / Silero VAD
└───────────┬──────────┘
            │ 発話終了
            ▼
┌──────────────────────┐
│  STT (音声認識)       │  Whisper / Google STT
│                      │  「明日の天気を教えて」
└───────────┬──────────┘
            │ テキスト
            ▼
┌──────────────────────┐
│  NLU / LLM           │  意図理解 + 応答生成
│  (自然言語理解)       │  GPT-4o / Claude
└───────────┬──────────┘
            │ 応答テキスト
            ▼
┌──────────────────────┐
│  TTS (音声合成)       │  OpenAI TTS / VITS
│                      │  「明日は晴れです」
└───────────┬──────────┘
            │ 音声
            ▼
       スピーカー出力
==================================================
```

### 1.2 ウェイクワード検出

```python
# Porcupine（Picovoice）によるウェイクワード検出

import pvporcupine
import pyaudio
import struct

class WakeWordDetector:
    """ウェイクワード検出器"""

    def __init__(self, access_key: str, keyword: str = "computer"):
        """
        keyword options:
        - 組み込み: "alexa", "computer", "jarvis", "hey google" 等
        - カスタム: Picovoice Console で作成した .ppn ファイル
        """
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=[keyword],
            sensitivities=[0.7],  # 感度（0-1、高いほど誤検知増）
        )

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
        )

    def listen(self, callback):
        """ウェイクワードを待機"""
        print("ウェイクワード待機中...")
        try:
            while True:
                pcm = self.stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from(
                    "h" * self.porcupine.frame_length, pcm
                )

                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    print("ウェイクワード検出!")
                    callback()

        except KeyboardInterrupt:
            self.cleanup()

    def cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.porcupine.delete()
```

### 1.3 OpenWakeWord（OSS版）

```python
# OpenWakeWord: オープンソースのウェイクワード検出

from openwakeword import Model
import pyaudio
import numpy as np

class OpenWakeWordDetector:
    """OSSウェイクワード検出（OpenWakeWord）"""

    def __init__(self, model_path: str = None):
        self.model = Model(
            wakeword_models=[model_path] if model_path else None,
            inference_framework="onnx",  # onnx or tflite
        )

        self.audio = pyaudio.PyAudio()
        self.chunk_size = 1280  # 80ms @ 16kHz

    def listen_continuous(self, on_wake):
        """連続リスニング"""
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        print("リスニング開始...")
        while True:
            audio_data = stream.read(self.chunk_size)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # 推論
            prediction = self.model.predict(audio_array)

            for mdl_name, score in prediction.items():
                if score > 0.5:  # 閾値
                    print(f"検出: {mdl_name} (score: {score:.3f})")
                    on_wake()
```

---

## 2. 対話管理

### 2.1 LLM統合の対話エンジン

```python
from openai import OpenAI
import json

class VoiceAssistantEngine:
    """LLM統合の音声アシスタントエンジン"""

    def __init__(self):
        self.client = OpenAI()
        self.conversation_history = []
        self.system_prompt = """あなたは日本語の音声アシスタントです。
以下のルールに従ってください:
- 簡潔に回答する（音声で聞くので長すぎない）
- 1-3文で回答する
- 専門用語は避ける
- 友好的な口調"""

        # ツール定義（Function Calling）
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "指定された場所の天気を取得する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "場所名"},
                            "date": {"type": "string", "description": "日付 (YYYY-MM-DD)"},
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "set_timer",
                    "description": "タイマーを設定する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "minutes": {"type": "integer", "description": "分数"},
                            "label": {"type": "string", "description": "タイマーのラベル"},
                        },
                        "required": ["minutes"],
                    },
                },
            },
        ]

    def process_input(self, user_text: str) -> str:
        """ユーザー入力を処理して応答を生成"""
        self.conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history[-10:],  # 直近10ターン
            ],
            tools=self.tools,
            max_tokens=200,  # 音声出力なので短め
        )

        message = response.choices[0].message

        # Function Call の処理
        if message.tool_calls:
            return self._handle_tool_calls(message)

        assistant_text = message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_text,
        })

        return assistant_text

    def _handle_tool_calls(self, message):
        """ツール呼び出しの処理"""
        self.conversation_history.append(message)

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            result = self._execute_function(func_name, args)

            self.conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # ツール結果を踏まえた応答生成
        followup = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history[-10:],
            ],
        )

        return followup.choices[0].message.content

    def _execute_function(self, name, args):
        """関数の実行"""
        if name == "get_weather":
            return {"weather": "晴れ", "temperature": 22, "location": args["location"]}
        elif name == "set_timer":
            return {"status": "set", "minutes": args["minutes"]}
        return {"error": "unknown function"}
```

### 2.2 完全統合パイプライン

```python
import threading
import queue

class VoiceAssistant:
    """完全統合型音声アシスタント"""

    def __init__(self):
        self.wake_detector = WakeWordDetector(access_key="...", keyword="computer")
        self.stt = WhisperSTT(model="base")
        self.engine = VoiceAssistantEngine()
        self.tts = OpenAITTS(voice="nova")
        self.is_listening = False
        self.audio_queue = queue.Queue()

    def run(self):
        """メインループ"""
        print("音声アシスタント起動")
        self.wake_detector.listen(callback=self._on_wake)

    def _on_wake(self):
        """ウェイクワード検出時の処理"""
        # 応答音を再生（「ポーン」）
        play_acknowledgment_sound()

        # 音声録音（VAD で自動終了）
        audio = self._record_with_vad(max_duration=10)

        # STT
        user_text = self.stt.transcribe(audio)
        print(f"ユーザー: {user_text}")

        if not user_text.strip():
            return

        # LLM処理
        response_text = self.engine.process_input(user_text)
        print(f"アシスタント: {response_text}")

        # TTS + 再生
        audio_response = self.tts.synthesize(response_text)
        play_audio(audio_response)

    def _record_with_vad(self, max_duration=10, silence_threshold=1.5):
        """VAD付き録音（無音が続いたら自動停止）"""
        import webrtcvad

        vad = webrtcvad.Vad(2)  # 0-3（高いほど厳しい）
        frames = []
        silent_frames = 0
        max_silent = int(silence_threshold / 0.03)  # 30msフレーム

        # 録音ストリーム
        stream = open_audio_stream(sample_rate=16000, frame_duration_ms=30)

        for _ in range(int(max_duration / 0.03)):
            frame = stream.read()
            frames.append(frame)

            is_speech = vad.is_speech(frame, 16000)
            if not is_speech:
                silent_frames += 1
            else:
                silent_frames = 0

            if silent_frames > max_silent and len(frames) > 30:
                break

        return b"".join(frames)
```

---

## 3. マルチモーダル対話

### 3.1 OpenAI Realtime API

```
OpenAI Realtime API アーキテクチャ
==================================================

クライアント                  サーバー
    │                           │
    │  WebSocket接続             │
    │ ─────────────────────────→│
    │                           │
    │  音声ストリーム送信        │
    │  (PCM 24kHz 16bit)        │
    │ ─────────────────────────→│
    │                           │
    │         GPT-4o            │
    │     音声→理解→生成        │
    │                           │
    │  音声ストリーム受信        │
    │←───────────────────────── │
    │  (PCM 24kHz 16bit)        │
    │                           │
    │  Function Call             │
    │←───────────────────────── │
    │  結果送信                  │
    │ ─────────────────────────→│
    │                           │
    │  続き音声ストリーム        │
    │←───────────────────────── │

特徴:
- STT/LLM/TTS が単一モデルで統合
- 300ms以下のレイテンシ
- 割り込み（Interruption）対応
- 感情・トーンの理解
==================================================
```

---

## 4. 比較表

### 4.1 音声アシスタント構築アプローチ比較

| 項目 | パイプライン型 | Realtime API型 | エッジ型 |
|------|-------------|--------------|---------|
| アーキテクチャ | STT+LLM+TTS | 統合モデル | オンデバイス |
| レイテンシ | 1-3秒 | 0.3-1秒 | 0.5-2秒 |
| カスタマイズ | 各コンポーネント独立 | 限定的 | フル制御 |
| コスト | 各API合算 | API従量課金 | GPU初期投資 |
| プライバシー | クラウド送信 | クラウド送信 | ローカル完結 |
| 品質 | 組み合わせ次第 | 最高 | 中程度 |
| オフライン | 不可(※) | 不可 | 可能 |

### 4.2 ウェイクワード検出エンジン比較

| 項目 | Porcupine | OpenWakeWord | Snowboy | Mycroft Precise |
|------|-----------|-------------|---------|-----------------|
| ライセンス | 商用(無料枠あり) | Apache 2.0 | 終了 | Apache 2.0 |
| カスタムワード | 対応 | 対応 | - | 対応 |
| 精度 | 非常に高い | 高い | - | 中程度 |
| 誤検知率 | 非常に低い | 低い | - | 中程度 |
| CPU使用率 | 極めて低い | 低い | - | 中程度 |
| プラットフォーム | 多数 | Python | - | Python/Linux |
| エッジ対応 | RPi対応 | RPi対応 | - | RPi対応 |

---

## 5. アンチパターン

### 5.1 アンチパターン: 同期処理のブロッキング

```python
# BAD: 全処理を同期的に実行（UI/UXが最悪）
def bad_assistant_loop():
    while True:
        wake_word_detected = listen_for_wake_word()  # ブロック
        if wake_word_detected:
            audio = record_audio()          # ブロック
            text = transcribe(audio)        # ブロック（1-3秒）
            response = generate(text)       # ブロック（1-5秒）
            speech = synthesize(response)   # ブロック（1-2秒）
            play(speech)                    # ブロック
            # 合計3-10秒の無反応時間

# GOOD: 非同期 + ストリーミング処理
import asyncio

async def good_assistant_loop():
    while True:
        await listen_for_wake_word_async()

        # 応答音を即座に再生（フィードバック）
        asyncio.create_task(play_acknowledgment())

        # 録音とSTTを並行（ストリーミング）
        audio_stream = record_audio_stream()

        # STTストリーミング（部分結果をリアルタイム表示）
        partial_text = ""
        async for chunk in audio_stream:
            partial = await stt_streaming(chunk)
            if partial:
                partial_text = partial
                display_partial(partial_text)

        # LLMストリーミング
        response_stream = generate_streaming(partial_text)

        # TTSストリーミング（LLM出力を逐次音声化）
        async for text_chunk in response_stream:
            audio_chunk = await tts_streaming(text_chunk)
            await play_async(audio_chunk)
```

### 5.2 アンチパターン: エラーハンドリングの欠如

```python
# BAD: エラーで完全停止
def bad_process(audio):
    text = stt(audio)
    response = llm(text)
    speech = tts(response)
    return speech  # どこかで例外 → 全停止

# GOOD: グレースフルデグラデーション
async def good_process(audio):
    """段階的なフォールバック"""
    # STT with fallback
    try:
        text = await stt_primary(audio)
    except Exception:
        try:
            text = await stt_fallback(audio)
        except Exception:
            await speak("すみません、聞き取れませんでした。もう一度お願いします。")
            return

    # LLM with timeout
    try:
        response = await asyncio.wait_for(llm(text), timeout=5.0)
    except asyncio.TimeoutError:
        response = "申し訳ありません、処理に時間がかかっています。"
    except Exception:
        response = "エラーが発生しました。もう一度お試しください。"

    # TTS with fallback
    try:
        await speak_with_tts(response)
    except Exception:
        # TTS失敗時はテキスト表示
        display_text(response)
```

---

## 6. FAQ

### Q1: カスタムウェイクワードを作るにはどうすればよいですか？

主に3つの方法があります。(1) Picovoice Console: Web上でウェイクワードを入力し、.ppnファイルを生成（商用利用は有料）。(2) OpenWakeWord: 自分のデータ（100-500サンプル）でモデルを学習。TTS生成の合成音声でもデータ作成可能。(3) 自作: 小型のCNNまたはRNNモデルをMFCC特徴量で学習。いずれの方法でも、誤検知率テスト（False Accept Rate < 1回/24時間が目安）を必ず実施してください。

### Q2: 音声アシスタントのレイテンシを改善するには？

レイテンシ改善の主要策は、(1) ストリーミングSTT: バッチ処理ではなくストリーミング認識を使用（Google/Azure STT）。(2) LLMストリーミング: 最初のトークンが生成された時点でTTSを開始。(3) TTSストリーミング: PCM/Opusフォーマットで逐次再生。(4) 事前キャッシュ: 頻出応答（挨拶、確認等）を事前に音声化しておく。(5) エッジ推論: STTをローカル（faster-whisper tiny/base）で実行。これらを組み合わせると、体感1秒以下の応答が実現可能です。

### Q3: プライバシーに配慮した音声アシスタントを作るには？

プライバシー重視の設計として、(1) オンデバイス処理: ウェイクワード検出とVADは必ずローカルで実行し、ウェイクワード検出前の音声はクラウドに送信しない。(2) ローカルSTT: faster-whisperをローカルGPUで実行。(3) 録音の最小化: VAD終了後すぐに録音停止、処理後は即座に音声データを削除。(4) 暗号化: 通信は必ずTLS、保存データは暗号化。(5) ユーザー制御: ミュートボタン、履歴削除機能、データ収集のオプトイン/アウトを提供。

---

## まとめ

| 項目 | 要点 |
|------|------|
| パイプライン | Wake Word → VAD → STT → LLM → TTS の5段階 |
| ウェイクワード | Porcupine（商用品質）、OpenWakeWord（OSS） |
| 対話管理 | LLM + Function Calling で柔軟な対話実現 |
| レイテンシ | ストリーミング処理で体感1秒以下が可能 |
| マルチモーダル | OpenAI Realtime APIで統合音声対話 |
| プライバシー | ウェイクワード検出はローカル必須 |

## 次に読むべきガイド

- [02-podcast-tools.md](./02-podcast-tools.md) — ポッドキャストツール
- [../03-development/02-real-time-audio.md](../03-development/02-real-time-audio.md) — リアルタイム音声処理
- [../00-fundamentals/03-stt-technologies.md](../00-fundamentals/03-stt-technologies.md) — STT技術詳細

## 参考文献

1. Picovoice Documentation (2025). "Porcupine Wake Word Engine" — 商用品質のウェイクワードエンジンのドキュメント
2. OpenAI (2024). "Realtime API Documentation" — GPT-4oベースのリアルタイム音声対話APIのガイド
3. Rasa Open Source (2024). "Building Conversational AI" — オープンソースの対話管理フレームワークのドキュメント
