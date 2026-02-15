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

### 1.4 VAD（Voice Activity Detection）の詳細実装

```python
import numpy as np
import pyaudio
from typing import Optional, Callable

class SileroVADDetector:
    """
    Silero VADによる高精度音声区間検出

    Silero VADはPyTorchベースの軽量VADモデルで、
    webrtcvadより高精度で言語非依存の音声検出が可能。
    """

    def __init__(self, threshold: float = 0.5, sr: int = 16000):
        import torch
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
        )
        self.get_speech_timestamps = utils[0]
        self.threshold = threshold
        self.sr = sr
        self.window_size = 512  # 32ms @ 16kHz

    def detect_speech_regions(self, audio: np.ndarray) -> list:
        """音声区間を検出"""
        import torch
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sr,
        )
        return speech_timestamps

    def is_speech(self, frame: np.ndarray) -> bool:
        """フレーム単位での音声判定"""
        import torch
        frame_tensor = torch.tensor(frame, dtype=torch.float32)
        confidence = self.model(frame_tensor, self.sr).item()
        return confidence > self.threshold


class AdaptiveVAD:
    """
    環境適応型VAD

    背景ノイズレベルに動的に適応し、
    さまざまな環境で安定した音声検出を実現する。
    """

    def __init__(self, sr: int = 16000, frame_ms: int = 30):
        self.sr = sr
        self.frame_size = int(sr * frame_ms / 1000)
        self.noise_floor = 0.0
        self.noise_alpha = 0.95  # ノイズフロア追従係数
        self.speech_threshold_db = 15  # ノイズフロアからの閾値
        self.hangover_frames = 10  # 発話終了後の保持フレーム数
        self.hangover_counter = 0
        self.is_speaking = False

    def update_noise_floor(self, frame: np.ndarray):
        """ノイズフロアの動的更新"""
        rms = np.sqrt(np.mean(frame ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        if not self.is_speaking:
            # 非音声区間でのみノイズフロアを更新
            self.noise_floor = (
                self.noise_alpha * self.noise_floor +
                (1 - self.noise_alpha) * rms_db
            )

    def process_frame(self, frame: np.ndarray) -> bool:
        """フレームを処理して音声有無を判定"""
        rms = np.sqrt(np.mean(frame ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        self.update_noise_floor(frame)

        # 動的閾値
        threshold = self.noise_floor + self.speech_threshold_db

        if rms_db > threshold:
            self.is_speaking = True
            self.hangover_counter = self.hangover_frames
        elif self.hangover_counter > 0:
            self.hangover_counter -= 1
        else:
            self.is_speaking = False

        return self.is_speaking


class SmartRecorder:
    """
    VAD統合スマート録音器

    ウェイクワード検出後、ユーザーの発話を
    VADで自動的に区間検出して録音する。
    """

    def __init__(self, sr: int = 16000, max_duration: float = 15.0,
                 silence_timeout: float = 1.5, min_duration: float = 0.5):
        """
        Parameters:
            sr: サンプルレート
            max_duration: 最大録音時間（秒）
            silence_timeout: 無音がこの時間続いたら録音終了（秒）
            min_duration: 最小録音時間（秒）
        """
        self.sr = sr
        self.max_duration = max_duration
        self.silence_timeout = silence_timeout
        self.min_duration = min_duration
        self.vad = AdaptiveVAD(sr=sr)

    def record(self) -> Optional[np.ndarray]:
        """VAD付き録音を実行"""
        pa = pyaudio.PyAudio()
        frame_size = self.vad.frame_size

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sr,
            input=True,
            frames_per_buffer=frame_size,
        )

        frames = []
        silent_frames = 0
        max_silent = int(self.silence_timeout * self.sr / frame_size)
        max_frames = int(self.max_duration * self.sr / frame_size)
        min_frames = int(self.min_duration * self.sr / frame_size)
        speech_started = False

        print("録音中... (話してください)")

        for _ in range(max_frames):
            data = stream.read(frame_size, exception_on_overflow=False)
            frame = np.frombuffer(data, dtype=np.float32)
            frames.append(frame)

            is_speech = self.vad.process_frame(frame)

            if is_speech:
                speech_started = True
                silent_frames = 0
            elif speech_started:
                silent_frames += 1

            # 十分な発話後、無音が続いたら終了
            if speech_started and silent_frames > max_silent and len(frames) > min_frames:
                break

        stream.stop_stream()
        stream.close()
        pa.terminate()

        if not speech_started or len(frames) < min_frames:
            print("音声が検出されませんでした")
            return None

        audio = np.concatenate(frames)
        print(f"録音完了: {len(audio) / self.sr:.1f}秒")
        return audio
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

### 2.3 高度な対話管理: コンテキスト管理とスロットフィリング

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime

class DialogState(Enum):
    """対話状態"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    CONFIRMING = "confirming"
    ERROR = "error"

@dataclass
class ConversationContext:
    """会話コンテキスト管理"""
    session_id: str
    started_at: datetime = field(default_factory=datetime.now)
    history: List[Dict] = field(default_factory=list)
    user_profile: Dict = field(default_factory=dict)
    current_intent: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    state: DialogState = DialogState.IDLE
    turn_count: int = 0

    def add_turn(self, role: str, content: str):
        """ターンを追加"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        if role == "user":
            self.turn_count += 1

    def get_recent_history(self, n: int = 10) -> List[Dict]:
        """直近N件の履歴を取得"""
        return self.history[-n:]

    def summarize(self) -> str:
        """長くなった会話を要約"""
        if len(self.history) <= 10:
            return ""
        # 古い履歴を要約テキストに圧縮
        old_turns = self.history[:-10]
        topics = set()
        for turn in old_turns:
            if turn["role"] == "user":
                topics.add(turn["content"][:50])
        return f"これまでの話題: {', '.join(list(topics)[:5])}"


class IntentSlotManager:
    """インテント・スロット管理"""

    INTENT_SCHEMAS = {
        "set_alarm": {
            "required_slots": ["time"],
            "optional_slots": ["label", "repeat"],
            "confirm_before_execute": True,
            "prompts": {
                "time": "何時にアラームを設定しますか？",
                "label": "アラームにラベルを付けますか？",
            },
        },
        "play_music": {
            "required_slots": ["query"],
            "optional_slots": ["source", "shuffle"],
            "confirm_before_execute": False,
            "prompts": {
                "query": "何を再生しますか？",
            },
        },
        "get_weather": {
            "required_slots": ["location"],
            "optional_slots": ["date", "detail_level"],
            "confirm_before_execute": False,
            "prompts": {
                "location": "どの場所の天気を知りたいですか？",
            },
        },
        "send_message": {
            "required_slots": ["recipient", "message"],
            "optional_slots": ["app"],
            "confirm_before_execute": True,
            "prompts": {
                "recipient": "誰にメッセージを送りますか？",
                "message": "メッセージの内容は何ですか？",
            },
        },
        "control_device": {
            "required_slots": ["device", "action"],
            "optional_slots": ["value"],
            "confirm_before_execute": False,
            "prompts": {
                "device": "どのデバイスを操作しますか？",
                "action": "何をしますか？（オン/オフ/調整）",
            },
        },
    }

    def check_slots(self, intent: str, filled_slots: dict) -> Optional[str]:
        """
        未入力のスロットを確認し、次の質問を返す

        Returns:
            None: 全スロット入力済み
            str: 次に聞くべき質問
        """
        schema = self.INTENT_SCHEMAS.get(intent)
        if not schema:
            return None

        for slot in schema["required_slots"]:
            if slot not in filled_slots or not filled_slots[slot]:
                return schema["prompts"].get(slot, f"{slot}を教えてください")

        return None

    def needs_confirmation(self, intent: str) -> bool:
        """実行前に確認が必要か"""
        schema = self.INTENT_SCHEMAS.get(intent, {})
        return schema.get("confirm_before_execute", False)

    def generate_confirmation(self, intent: str, slots: dict) -> str:
        """確認メッセージを生成"""
        if intent == "set_alarm":
            time = slots.get("time", "不明")
            label = slots.get("label", "")
            msg = f"{time}にアラームを設定します"
            if label:
                msg += f"（ラベル: {label}）"
            return msg + "。よろしいですか？"
        elif intent == "send_message":
            recipient = slots.get("recipient", "不明")
            message = slots.get("message", "不明")
            return f"{recipient}に「{message}」と送信します。よろしいですか？"
        return "実行してよろしいですか？"


class AdvancedDialogManager:
    """高度な対話管理システム"""

    def __init__(self):
        self.client = OpenAI()
        self.slot_manager = IntentSlotManager()
        self.contexts: Dict[str, ConversationContext] = {}

    def get_or_create_context(self, session_id: str) -> ConversationContext:
        """セッションコンテキストを取得または作成"""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(
                session_id=session_id
            )
        return self.contexts[session_id]

    def process(self, session_id: str, user_input: str) -> str:
        """対話処理のメインループ"""
        ctx = self.get_or_create_context(session_id)
        ctx.add_turn("user", user_input)
        ctx.state = DialogState.PROCESSING

        # 確認待ち状態の処理
        if ctx.state == DialogState.CONFIRMING:
            return self._handle_confirmation(ctx, user_input)

        # LLMでインテントとスロットを抽出
        intent_result = self._extract_intent(ctx, user_input)

        if intent_result.get("intent"):
            ctx.current_intent = intent_result["intent"]
            ctx.slots.update(intent_result.get("slots", {}))

            # スロットの充足チェック
            next_question = self.slot_manager.check_slots(
                ctx.current_intent, ctx.slots
            )

            if next_question:
                ctx.add_turn("assistant", next_question)
                return next_question

            # 確認が必要な場合
            if self.slot_manager.needs_confirmation(ctx.current_intent):
                confirmation = self.slot_manager.generate_confirmation(
                    ctx.current_intent, ctx.slots
                )
                ctx.state = DialogState.CONFIRMING
                ctx.add_turn("assistant", confirmation)
                return confirmation

            # 実行
            result = self._execute_intent(ctx)
            ctx.add_turn("assistant", result)
            ctx.current_intent = None
            ctx.slots.clear()
            return result

        # 通常の対話応答
        response = self._generate_response(ctx, user_input)
        ctx.add_turn("assistant", response)
        return response

    def _extract_intent(self, ctx: ConversationContext,
                        user_input: str) -> dict:
        """LLMでインテントとスロットを抽出"""
        extraction_prompt = f"""
ユーザーの入力からインテントとスロットを抽出してください。
利用可能なインテント: {list(IntentSlotManager.INTENT_SCHEMAS.keys())}

ユーザー入力: {user_input}

JSON形式で回答:
{{"intent": "インテント名またはnull", "slots": {{"slot名": "値"}}}}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": extraction_prompt}],
            response_format={"type": "json_object"},
            max_tokens=200,
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"intent": None, "slots": {}}

    def _handle_confirmation(self, ctx: ConversationContext,
                             user_input: str) -> str:
        """確認応答の処理"""
        affirmative = any(w in user_input for w in ["はい", "うん", "OK", "お願い", "yes"])
        negative = any(w in user_input for w in ["いいえ", "いや", "やめ", "キャンセル", "no"])

        if affirmative:
            result = self._execute_intent(ctx)
            ctx.current_intent = None
            ctx.slots.clear()
            ctx.state = DialogState.IDLE
            return result
        elif negative:
            ctx.current_intent = None
            ctx.slots.clear()
            ctx.state = DialogState.IDLE
            return "キャンセルしました。"
        else:
            return "はい、またはいいえでお答えください。"

    def _execute_intent(self, ctx: ConversationContext) -> str:
        """インテントを実行"""
        intent = ctx.current_intent
        slots = ctx.slots

        if intent == "set_alarm":
            return f"アラームを{slots.get('time', '')}に設定しました。"
        elif intent == "get_weather":
            return f"{slots.get('location', '')}の天気は晴れ、気温22度です。"
        elif intent == "play_music":
            return f"「{slots.get('query', '')}」を再生します。"
        elif intent == "send_message":
            return f"{slots.get('recipient', '')}にメッセージを送信しました。"
        elif intent == "control_device":
            return f"{slots.get('device', '')}を{slots.get('action', '')}しました。"

        return "処理が完了しました。"

    def _generate_response(self, ctx: ConversationContext,
                           user_input: str) -> str:
        """通常の対話応答を生成"""
        messages = [
            {"role": "system", "content": "簡潔に、1-3文で回答してください。"},
        ]

        # 要約があれば追加
        summary = ctx.summarize()
        if summary:
            messages.append({"role": "system", "content": summary})

        # 直近の履歴
        for turn in ctx.get_recent_history(8):
            messages.append({
                "role": turn["role"],
                "content": turn["content"],
            })

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
        )

        return response.choices[0].message.content
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

### 3.2 Realtime API の実装

```python
import asyncio
import websockets
import json
import base64
import numpy as np

class RealtimeVoiceAssistant:
    """
    OpenAI Realtime API を使ったリアルタイム音声アシスタント

    従来のパイプライン（STT→LLM→TTS）と異なり、
    音声入力を直接理解して音声で応答する統合モデルを使用。
    レイテンシが大幅に削減される。
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-realtime-preview"):
        self.api_key = api_key
        self.model = model
        self.ws = None
        self.tools = []
        self.on_audio_callback = None

    async def connect(self):
        """WebSocket接続を確立"""
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        self.ws = await websockets.connect(url, extra_headers=headers)

        # セッション設定
        await self._send({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "あなたは日本語の音声アシスタントです。簡潔に回答してください。",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "tools": self.tools,
            },
        })

    async def send_audio(self, audio_chunk: bytes):
        """音声チャンクを送信"""
        encoded = base64.b64encode(audio_chunk).decode()
        await self._send({
            "type": "input_audio_buffer.append",
            "audio": encoded,
        })

    async def listen_responses(self):
        """応答を受信するループ"""
        async for message in self.ws:
            event = json.loads(message)
            event_type = event.get("type", "")

            if event_type == "response.audio.delta":
                # 音声応答チャンクを受信
                audio_data = base64.b64decode(event["delta"])
                if self.on_audio_callback:
                    self.on_audio_callback(audio_data)

            elif event_type == "response.audio_transcript.delta":
                # テキストトランスクリプト
                print(event.get("delta", ""), end="", flush=True)

            elif event_type == "input_audio_buffer.speech_started":
                # ユーザーが話し始めた（割り込み検知）
                print("\n[ユーザー発話開始]")

            elif event_type == "input_audio_buffer.speech_stopped":
                print("\n[ユーザー発話終了]")

            elif event_type == "response.function_call_arguments.done":
                # Function Call を処理
                await self._handle_function_call(event)

            elif event_type == "error":
                print(f"エラー: {event.get('error', {}).get('message', '')}")

    async def _send(self, data: dict):
        """メッセージ送信"""
        await self.ws.send(json.dumps(data))

    async def _handle_function_call(self, event):
        """Function Callの処理"""
        call_id = event.get("call_id", "")
        name = event.get("name", "")
        args = json.loads(event.get("arguments", "{}"))

        # 関数実行
        result = self._execute_function(name, args)

        # 結果を返送
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result, ensure_ascii=False),
            },
        })

        # 応答生成をトリガー
        await self._send({"type": "response.create"})

    def _execute_function(self, name, args):
        """関数実行（プレースホルダー）"""
        return {"status": "ok", "result": f"{name}を実行しました"}

    def add_tool(self, name: str, description: str, parameters: dict):
        """ツール（関数）を追加"""
        self.tools.append({
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters,
        })


async def main():
    """Realtime API アシスタントのメイン処理"""
    assistant = RealtimeVoiceAssistant(api_key="sk-...")

    # ツール登録
    assistant.add_tool(
        name="get_weather",
        description="天気を取得する",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    )

    # 音声出力コールバック
    def play_audio(data):
        # PyAudioで再生
        pass
    assistant.on_audio_callback = play_audio

    # 接続
    await assistant.connect()

    # マイク入力タスクと応答受信タスクを並行実行
    await asyncio.gather(
        capture_microphone(assistant),
        assistant.listen_responses(),
    )
```

### 3.3 エッジデバイス向け音声アシスタント

```python
class EdgeVoiceAssistant:
    """
    エッジデバイス（Raspberry Pi等）向け音声アシスタント

    ウェイクワード検出とVADをローカルで実行し、
    STT/LLM/TTSはクラウドまたはローカルモデルを選択可能。
    プライバシー重視の設計。
    """

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()

        # ウェイクワード（常にローカル）
        self.wake_detector = OpenWakeWordDetector()

        # STT（ローカルまたはクラウド）
        if self.config["stt_local"]:
            self.stt = self._init_local_stt()
        else:
            self.stt = self._init_cloud_stt()

        # LLM
        if self.config["llm_local"]:
            self.llm = self._init_local_llm()
        else:
            self.llm = self._init_cloud_llm()

        # TTS
        if self.config["tts_local"]:
            self.tts = self._init_local_tts()
        else:
            self.tts = self._init_cloud_tts()

        self.recorder = SmartRecorder(sr=16000)

    def _default_config(self):
        return {
            "stt_local": True,      # faster-whisperをローカル実行
            "llm_local": False,     # LLMはクラウド推奨
            "tts_local": True,      # Piper TTSをローカル実行
            "stt_model": "base",    # Whisperモデルサイズ
            "wake_word": "hey_jarvis",
            "language": "ja",
            "max_recording_sec": 10,
        }

    def _init_local_stt(self):
        """ローカルSTT（faster-whisper）"""
        from faster_whisper import WhisperModel
        return WhisperModel(
            self.config["stt_model"],
            device="cpu",
            compute_type="int8",  # 軽量推論
        )

    def _init_cloud_stt(self):
        """クラウドSTT"""
        return None  # OpenAI Whisper API等

    def _init_local_llm(self):
        """ローカルLLM（llama.cpp等）"""
        return None  # Placeholder

    def _init_cloud_llm(self):
        """クラウドLLM"""
        from openai import OpenAI
        return OpenAI()

    def _init_local_tts(self):
        """ローカルTTS（Piper）"""
        return None  # Piper TTS

    def _init_cloud_tts(self):
        """クラウドTTS"""
        return None  # OpenAI TTS

    def run(self):
        """メインループ"""
        print(f"エッジ音声アシスタント起動")
        print(f"STT: {'ローカル' if self.config['stt_local'] else 'クラウド'}")
        print(f"LLM: {'ローカル' if self.config['llm_local'] else 'クラウド'}")
        print(f"TTS: {'ローカル' if self.config['tts_local'] else 'クラウド'}")

        self.wake_detector.listen_continuous(on_wake=self._on_wake)

    def _on_wake(self):
        """ウェイクワード検出時の処理"""
        import time

        # LED点灯（GPIOがある場合）
        self._set_status_led("listening")

        # 録音
        audio = self.recorder.record()
        if audio is None:
            self._set_status_led("idle")
            return

        # STT
        self._set_status_led("processing")
        start = time.time()

        if self.config["stt_local"]:
            segments, info = self.stt.transcribe(
                audio, language="ja", beam_size=3
            )
            user_text = " ".join(s.text for s in segments)
        else:
            user_text = self._cloud_transcribe(audio)

        stt_time = time.time() - start
        print(f"STT ({stt_time:.2f}秒): {user_text}")

        if not user_text.strip():
            self._set_status_led("idle")
            return

        # LLM
        start = time.time()
        response = self._generate_response(user_text)
        llm_time = time.time() - start
        print(f"LLM ({llm_time:.2f}秒): {response}")

        # TTS
        self._set_status_led("speaking")
        start = time.time()
        self._speak(response)
        tts_time = time.time() - start
        print(f"TTS ({tts_time:.2f}秒)")

        total = stt_time + llm_time + tts_time
        print(f"合計レイテンシ: {total:.2f}秒")

        self._set_status_led("idle")

    def _generate_response(self, text: str) -> str:
        """応答生成"""
        if self.config["llm_local"]:
            return "ローカルLLMの応答"  # Placeholder
        else:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "簡潔に1-2文で回答してください。"},
                    {"role": "user", "content": text},
                ],
                max_tokens=100,
            )
            return response.choices[0].message.content

    def _speak(self, text: str):
        """テキストを音声で再生"""
        pass  # TTS処理

    def _set_status_led(self, status: str):
        """ステータスLEDの制御（Raspberry Pi GPIO）"""
        led_colors = {
            "idle": (0, 0, 0),       # 消灯
            "listening": (0, 0, 255),  # 青
            "processing": (255, 255, 0), # 黄
            "speaking": (0, 255, 0),   # 緑
            "error": (255, 0, 0),      # 赤
        }
        # GPIO制御のプレースホルダー
        color = led_colors.get(status, (0, 0, 0))

    def _cloud_transcribe(self, audio: np.ndarray) -> str:
        """クラウドSTT"""
        return ""  # Placeholder
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

### 4.3 STTモデル比較（音声アシスタント向け）

| モデル | レイテンシ | 日本語精度 | オフライン | コスト | 推奨用途 |
|-------|----------|----------|---------|-------|---------|
| Whisper large-v3 | 3-10秒 | 最高 | 可(GPU) | 無料 | バッチ処理 |
| Whisper base | 1-3秒 | 良い | 可(CPU) | 無料 | エッジ |
| faster-whisper | 0.5-2秒 | 高い | 可 | 無料 | エッジ推奨 |
| Google STT | 0.3-1秒 | 非常に高い | 不可 | 従量課金 | クラウド推奨 |
| Azure STT | 0.3-1秒 | 非常に高い | 不可 | 従量課金 | エンタープライズ |
| Deepgram | 0.2-0.5秒 | 高い | 不可 | 従量課金 | 低遅延 |

### 4.4 TTS選択肢比較

| TTS | 自然さ | 日本語 | レイテンシ | ストリーミング | コスト |
|-----|-------|-------|----------|-------------|-------|
| OpenAI TTS | 最高 | 対応 | 0.5-1秒 | 対応 | $15/1M文字 |
| ElevenLabs | 最高 | 対応 | 0.3-1秒 | 対応 | $5/月〜 |
| Google Cloud TTS | 高い | 対応 | 0.3-0.5秒 | 対応 | 従量課金 |
| Azure TTS | 高い | 対応 | 0.3-0.5秒 | 対応 | 従量課金 |
| VOICEVOX | 高い(アニメ系) | 日本語のみ | 0.5-2秒 | 非対応 | 無料 |
| Piper | 中程度 | 限定的 | 0.1-0.3秒 | 非対応 | 無料 |

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

### 5.3 アンチパターン: ウェイクワードの感度設定ミス

```python
# BAD: 感度を高く設定しすぎ → 誤検知頻発
def bad_wake_word():
    detector = WakeWordDetector(
        sensitivity=0.99,  # テレビの音にも反応してしまう
    )

# BAD: 感度を低く設定しすぎ → 反応しない
def bad_wake_word_low():
    detector = WakeWordDetector(
        sensitivity=0.1,  # 大声で叫んでも反応しない
    )

# GOOD: 環境に応じた感度調整 + 二段階検証
def good_wake_word():
    """二段階検証による誤検知低減"""
    detector = WakeWordDetector(
        sensitivity=0.6,  # やや高めに設定
    )

    def on_first_detection():
        """一次検出後の確認処理"""
        # 直後の音声を分析して話者を確認
        audio = record_short(duration_ms=500)

        # 話者照合（登録済み話者かどうか）
        if verify_speaker(audio):
            # 本物の呼びかけ → アシスタント起動
            start_assistant()
        else:
            # テレビ等の誤検知 → 無視
            pass

    detector.listen(callback=on_first_detection)
```

### 5.4 アンチパターン: 会話コンテキストの未管理

```python
# BAD: 毎ターン独立した処理
def bad_conversation(text):
    # 前のターンの内容を覚えていない
    response = llm(text)  # 「それ」が何か分からない
    return response

# GOOD: コンテキスト管理付き
class GoodConversation:
    def __init__(self):
        self.history = []
        self.max_history = 20
        self.entity_memory = {}  # 言及されたエンティティを記憶

    def process(self, text):
        self.history.append({"role": "user", "content": text})

        # エンティティ抽出と記憶
        entities = extract_entities(text)
        self.entity_memory.update(entities)

        # コンテキスト付きでLLM呼び出し
        response = llm(
            messages=self.history[-self.max_history:],
            context=self.entity_memory,
        )

        self.history.append({"role": "assistant", "content": response})

        # 履歴が長くなったら要約
        if len(self.history) > self.max_history:
            summary = summarize(self.history[:self.max_history // 2])
            self.history = [
                {"role": "system", "content": f"これまでの要約: {summary}"},
                *self.history[self.max_history // 2:],
            ]

        return response
```

---

## 6. 実践的なユースケース

### 6.1 スマートホーム音声コントローラ

```python
class SmartHomeVoiceController:
    """スマートホーム音声コントローラ"""

    def __init__(self):
        self.devices = {
            "リビングの照明": {"type": "light", "id": "living_light"},
            "寝室の照明": {"type": "light", "id": "bedroom_light"},
            "エアコン": {"type": "ac", "id": "main_ac"},
            "テレビ": {"type": "tv", "id": "living_tv"},
            "加湿器": {"type": "humidifier", "id": "bedroom_hum"},
        }

        self.engine = VoiceAssistantEngine()
        self.engine.tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "control_light",
                    "description": "照明の制御（オン/オフ/調光/色変更）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device_id": {"type": "string"},
                            "action": {
                                "type": "string",
                                "enum": ["on", "off", "dim", "brighten"],
                            },
                            "brightness": {
                                "type": "integer",
                                "minimum": 0, "maximum": 100,
                            },
                            "color": {"type": "string"},
                        },
                        "required": ["device_id", "action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "control_ac",
                    "description": "エアコンの制御",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["on", "off", "set_temp"],
                            },
                            "temperature": {"type": "integer"},
                            "mode": {
                                "type": "string",
                                "enum": ["cool", "heat", "auto", "dry"],
                            },
                        },
                        "required": ["action"],
                    },
                },
            },
        ])

    def process_command(self, voice_text: str) -> str:
        """音声コマンドを処理"""
        return self.engine.process_input(voice_text)
```

### 6.2 会議アシスタント

```python
class MeetingAssistant:
    """会議中に動作する音声アシスタント"""

    def __init__(self):
        self.client = OpenAI()
        self.transcript = []
        self.action_items = []
        self.participants = set()

    def process_segment(self, speaker: str, text: str):
        """発話セグメントを処理"""
        self.transcript.append({
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        })
        self.participants.add(speaker)

        # アクションアイテムの自動検出
        if self._is_action_item(text):
            self.action_items.append({
                "speaker": speaker,
                "text": text,
                "detected_at": datetime.now().isoformat(),
            })

    def _is_action_item(self, text: str) -> bool:
        """アクションアイテムかどうかを判定"""
        indicators = [
            "やっておきます", "確認します", "対応します",
            "担当します", "調べておきます", "報告します",
            "次回までに", "来週までに", "明日までに",
        ]
        return any(ind in text for ind in indicators)

    def generate_summary(self) -> str:
        """会議の要約を生成"""
        transcript_text = "\n".join(
            f"{t['speaker']}: {t['text']}" for t in self.transcript
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """
会議の議事録から以下を抽出してください:
1. 議題の要約（3-5行）
2. 決定事項
3. アクションアイテム（担当者・期限付き）
4. 次回の議題候補
"""},
                {"role": "user", "content": transcript_text},
            ],
        )

        return response.choices[0].message.content

    def answer_question(self, question: str) -> str:
        """会議内容に関する質問に回答"""
        transcript_text = "\n".join(
            f"{t['speaker']}: {t['text']}" for t in self.transcript[-50:]
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"以下の会議内容に基づいて質問に回答してください:\n{transcript_text}"},
                {"role": "user", "content": question},
            ],
            max_tokens=200,
        )

        return response.choices[0].message.content
```

---

## 7. FAQ

### Q1: カスタムウェイクワードを作るにはどうすればよいですか？

主に3つの方法があります。(1) Picovoice Console: Web上でウェイクワードを入力し、.ppnファイルを生成（商用利用は有料）。(2) OpenWakeWord: 自分のデータ（100-500サンプル）でモデルを学習。TTS生成の合成音声でもデータ作成可能。(3) 自作: 小型のCNNまたはRNNモデルをMFCC特徴量で学習。いずれの方法でも、誤検知率テスト（False Accept Rate < 1回/24時間が目安）を必ず実施してください。

### Q2: 音声アシスタントのレイテンシを改善するには？

レイテンシ改善の主要策は、(1) ストリーミングSTT: バッチ処理ではなくストリーミング認識を使用（Google/Azure STT）。(2) LLMストリーミング: 最初のトークンが生成された時点でTTSを開始。(3) TTSストリーミング: PCM/Opusフォーマットで逐次再生。(4) 事前キャッシュ: 頻出応答（挨拶、確認等）を事前に音声化しておく。(5) エッジ推論: STTをローカル（faster-whisper tiny/base）で実行。これらを組み合わせると、体感1秒以下の応答が実現可能です。

### Q3: プライバシーに配慮した音声アシスタントを作るには？

プライバシー重視の設計として、(1) オンデバイス処理: ウェイクワード検出とVADは必ずローカルで実行し、ウェイクワード検出前の音声はクラウドに送信しない。(2) ローカルSTT: faster-whisperをローカルGPUで実行。(3) 録音の最小化: VAD終了後すぐに録音停止、処理後は即座に音声データを削除。(4) 暗号化: 通信は必ずTLS、保存データは暗号化。(5) ユーザー制御: ミュートボタン、履歴削除機能、データ収集のオプトイン/アウトを提供。

### Q4: 音声アシスタントをマルチ言語対応にするには？

マルチ言語対応のアプローチとして、(1) 言語検出: ウェイクワード検出後の音声を言語識別し、適切なSTTモデルを選択。Whisperは自動言語検出に対応しています。(2) LLM: GPT-4oやClaude等の多言語モデルはプロンプトなしで多言語に対応可能。(3) TTS: 各言語に対応したTTSモデルまたはAPIを用意。(4) ウェイクワード: 言語別のウェイクワードを用意するか、言語非依存のウェイクワード（固有名詞等）を使用。

### Q5: 音声アシスタントの割り込み（Interruption）対応はどう実装しますか？

割り込み対応の実装方法として、(1) OpenAI Realtime API: server_vadによる自動割り込み検知が組み込み。(2) パイプライン型: TTS再生中もマイクを監視し、VADが音声を検出したらTTS再生を中断。(3) 技術的課題: スピーカーの出力をマイクが拾う「エコー」を除去するAEC（Acoustic Echo Cancellation）が必要。WebRTCのAECモジュールやspeexdsp-pyが利用可能。(4) UX: 中断時は「はい？」等の短い応答を返し、新しい発話を待つのが自然な対話フローです。

### Q6: Raspberry Piで音声アシスタントを動かす際の推奨構成は？

Raspberry Pi 4（4GB以上）での推奨構成として、(1) マイク: ReSpeaker 2-Mic Hat またはUSBマイク（指向性推奨）。(2) スピーカー: 3.5mmジャックまたはBluetooth。(3) ウェイクワード: Porcupine（CPU使用率2%以下）またはOpenWakeWord。(4) STT: faster-whisper tinyモデル（CPU推論で3秒以下）。(5) LLM: クラウドAPI推奨（GPT-4o-mini等）。ローカルの場合はGemma 2B等の小型モデル。(6) TTS: Piper（CPU推論でリアルタイム以下）。(7) OS: Raspberry Pi OS Lite（GUIなし）で軽量化。全体でのレイテンシは2-4秒程度です。

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
| エッジ展開 | RPi4でfaster-whisper+Piper構成が実用的 |
| 対話状態管理 | インテント/スロット管理で構造化対話を実現 |

## 次に読むべきガイド

- [02-podcast-tools.md](./02-podcast-tools.md) — ポッドキャストツール
- [../03-development/02-real-time-audio.md](../03-development/02-real-time-audio.md) — リアルタイム音声処理
- [../00-fundamentals/03-stt-technologies.md](../00-fundamentals/03-stt-technologies.md) — STT技術詳細

## 参考文献

1. Picovoice Documentation (2025). "Porcupine Wake Word Engine" — 商用品質のウェイクワードエンジンのドキュメント
2. OpenAI (2024). "Realtime API Documentation" — GPT-4oベースのリアルタイム音声対話APIのガイド
3. Rasa Open Source (2024). "Building Conversational AI" — オープンソースの対話管理フレームワークのドキュメント
4. Silero Team (2024). "Silero VAD" — 高精度軽量VADモデルの実装と評価
5. Radford, A., et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision" — Whisper論文。大規模弱教師あり学習によるSTT
6. Hughes, T., et al. (2023). "OpenWakeWord: An Open-Source Wakeword Detection Library" — OSSウェイクワード検出ライブラリ
