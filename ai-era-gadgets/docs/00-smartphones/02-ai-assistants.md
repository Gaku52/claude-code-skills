# AIアシスタント — Siri / Google Assistant / Alexa と LLM統合

> 音声AIアシスタントの仕組みから、音声認識パイプライン、LLM（大規模言語モデル）との統合、そしてカスタム音声アプリの開発手法まで体系的に解説する。

---

## この章で学ぶこと

1. **音声認識パイプラインの構造** — 音声入力から意図理解・応答生成までの処理フロー
2. **主要アシスタントの技術比較** — Siri / Google Assistant / Alexa の設計思想と強み
3. **LLM統合の最前線** — ChatGPT / Gemini によるアシスタント進化と開発手法

---

## 1. 音声アシスタントのアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│              音声アシスタント処理パイプライン                    │
│                                                               │
│  ユーザー発話                                                  │
│      │                                                        │
│      ▼                                                        │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ Wake Word│──▶│ ASR      │──▶│ NLU      │──▶│ Dialog   │   │
│  │ Detection│   │ (音声→  │   │ (意図    │   │ Manager  │   │
│  │ "Hey     │   │  テキスト)│   │  理解)   │   │ (対話    │   │
│  │  Siri"   │   │ Whisper等│   │ BERT等   │   │  管理)   │   │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                    │          │
│                                                    ▼          │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ スピーカー│◀──│ TTS      │◀──│ Response │◀──│ Action   │   │
│  │ 出力     │   │ (テキスト│   │ Generator│   │ Executor │   │
│  │          │   │  →音声)  │   │ (LLM)    │   │ (API呼出)│   │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 従来型 vs LLM統合型

```
┌─────────────────────────────────────────────┐
│  【従来型】インテントベース                    │
│                                               │
│  "明日の天気は？"                             │
│     ↓ NLU                                    │
│  Intent: weather_query                        │
│  Slot: date=tomorrow, location=current        │
│     ↓ ルールベース                            │
│  Weather API → 定型応答テンプレート            │
│                                               │
│  【LLM統合型】自由対話                        │
│                                               │
│  "明日の天気は？ ピクニックに行けそう？"       │
│     ↓ LLM (Gemini / GPT-4)                  │
│  ・天気API呼び出し                            │
│  ・気温・降水確率を考慮                       │
│  ・「晴れで25℃なのでピクニック日和です！       │
│    ただし午後から風が強まるので午前中が        │
│    おすすめです」                              │
└─────────────────────────────────────────────┘
```

---

## 2. コード例

### コード例 1: Google Actions SDK によるカスタムアクション

```javascript
const { conversation } = require('@assistant/conversation');
const functions = require('firebase-functions');

const app = conversation();

// メインインテントの処理
app.handle('greeting', (conv) => {
  conv.add('こんにちは！AIアシスタントガイドへようこそ。');
  conv.add('何についてお手伝いしましょうか？');
});

// 天気照会のインテント処理
app.handle('weather_query', async (conv) => {
  const location = conv.intent.params.location?.resolved;
  const date = conv.intent.params.date?.resolved;

  // 外部API呼び出し
  const weather = await fetchWeather(location, date);

  conv.add(`${location}の${date}の天気は${weather.condition}、`
    + `気温は${weather.temp}℃の予報です。`);

  if (weather.rain_probability > 50) {
    conv.add('傘をお持ちになることをおすすめします。');
  }
});

exports.ActionsOnGoogleFulfillment = functions.https.onRequest(app);
```

### コード例 2: Alexa Skill（Python Lambda）

```python
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.utils import is_intent_name

sb = SkillBuilder()

class RecipeIntentHandler(AbstractRequestHandler):
    """料理レシピを提案するスキル"""

    def can_handle(self, handler_input):
        return is_intent_name("RecipeIntent")(handler_input)

    def handle(self, handler_input):
        slots = handler_input.request_envelope.request.intent.slots
        ingredient = slots["ingredient"].value

        # LLMでレシピ生成（Bedrock経由）
        import boto3
        bedrock = boto3.client('bedrock-runtime')
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-sonnet',
            body=json.dumps({
                "prompt": f"{ingredient}を使った簡単なレシピを1つ提案してください。",
                "max_tokens": 200
            })
        )
        recipe = json.loads(response['body'].read())['completion']

        speech = f"{ingredient}を使ったレシピをご紹介します。{recipe}"
        return handler_input.response_builder.speak(speech).response

sb.add_request_handler(RecipeIntentHandler())
lambda_handler = sb.lambda_handler()
```

### コード例 3: Siri Shortcuts + Intents（Swift）

```swift
import Intents
import IntentsUI

// カスタムIntentの定義（Xcode Intent Definition File）
class OrderCoffeeIntentHandler: NSObject, OrderCoffeeIntentHandling {

    func handle(intent: OrderCoffeeIntent,
                completion: @escaping (OrderCoffeeIntentResponse) -> Void) {

        let coffeeType = intent.coffeeType ?? "ラテ"
        let size = intent.size ?? .medium

        // 注文処理
        CoffeeAPI.placeOrder(type: coffeeType, size: size) { result in
            switch result {
            case .success(let order):
                let response = OrderCoffeeIntentResponse(code: .success,
                    userActivity: nil)
                response.orderNumber = order.id
                response.estimatedTime = "\(order.waitMinutes)分"
                completion(response)

            case .failure:
                completion(OrderCoffeeIntentResponse(code: .failure,
                    userActivity: nil))
            }
        }
    }

    // Siriへの提案
    func resolveCoffeeType(for intent: OrderCoffeeIntent,
        with completion: @escaping (INStringResolutionResult) -> Void) {
        if let type = intent.coffeeType {
            completion(.success(with: type))
        } else {
            completion(.needsValue())
        }
    }
}
```

### コード例 4: OpenAI Realtime API によるリアルタイム音声対話

```python
import asyncio
import websockets
import json
import base64
import pyaudio

async def realtime_voice_assistant():
    """OpenAI Realtime API でリアルタイム音声アシスタント"""

    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    async with websockets.connect(url, extra_headers=headers) as ws:
        # セッション設定
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "あなたは親切な日本語アシスタントです。",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",  # サーバー側VAD
                    "threshold": 0.5
                }
            }
        }))

        # マイク入力を送信
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16,
                          channels=1, rate=24000,
                          input=True, frames_per_buffer=1024)

        async def send_audio():
            while True:
                data = stream.read(1024, exception_on_overflow=False)
                encoded = base64.b64encode(data).decode()
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": encoded
                }))
                await asyncio.sleep(0.04)

        async def receive_response():
            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] == "response.audio.delta":
                    audio_data = base64.b64decode(msg["delta"])
                    # スピーカーに出力
                    play_audio(audio_data)

        await asyncio.gather(send_audio(), receive_response())

asyncio.run(realtime_voice_assistant())
```

### コード例 5: ローカルLLM音声アシスタント（Whisper + Ollama）

```python
import whisper
import ollama
import pyttsx3
import sounddevice as sd
import numpy as np

class LocalVoiceAssistant:
    """完全ローカルで動作する音声アシスタント"""

    def __init__(self):
        # Whisper（音声認識）
        self.asr_model = whisper.load_model("base")
        # TTS エンジン
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 180)

    def listen(self, duration=5, sample_rate=16000):
        """マイクから音声を録音"""
        print("聞いています...")
        audio = sd.rec(int(duration * sample_rate),
                      samplerate=sample_rate, channels=1,
                      dtype='float32')
        sd.wait()
        return audio.flatten()

    def transcribe(self, audio):
        """音声をテキストに変換（Whisper）"""
        result = self.asr_model.transcribe(
            audio, language="ja", fp16=False
        )
        return result["text"]

    def think(self, user_text, context=None):
        """ローカルLLMで応答生成（Ollama）"""
        messages = [
            {"role": "system",
             "content": "簡潔に日本語で回答してください。"},
        ]
        if context:
            messages.append({"role": "assistant", "content": context})
        messages.append({"role": "user", "content": user_text})

        response = ollama.chat(model="gemma2:9b", messages=messages)
        return response['message']['content']

    def speak(self, text):
        """テキストを音声で読み上げ"""
        print(f"アシスタント: {text}")
        self.tts.say(text)
        self.tts.runAndWait()

    def run(self):
        """メインループ"""
        print("ローカル音声アシスタント起動（Ctrl+Cで終了）")
        context = None
        while True:
            audio = self.listen()
            text = self.transcribe(audio)
            print(f"ユーザー: {text}")

            if "終了" in text or "さようなら" in text:
                self.speak("さようなら。")
                break

            response = self.think(text, context)
            context = response
            self.speak(response)

assistant = LocalVoiceAssistant()
assistant.run()
```

---

## 3. 比較表

### 比較表 1: 主要AIアシスタント比較

| 項目 | Siri (Apple) | Google Assistant | Alexa (Amazon) |
|------|-------------|-----------------|---------------|
| LLM統合 | Apple Intelligence + ChatGPT | Gemini | Alexa LLM + Bedrock |
| オンデバイス処理 | Neural Engine | Tensor TPU | 限定的 |
| 対応言語数 | 21言語 | 40+言語 | 8言語 |
| スマートホーム | HomeKit | Google Home | Alexa Smart Home |
| サードパーティ拡張 | Shortcuts / App Intents | Actions on Google | Alexa Skills |
| プライバシー | データは端末優先 | Googleアカウント連携 | クラウド処理中心 |
| 音声認識精度 | 高い（英語） | 最高水準 | 高い |
| マルチモーダル | テキスト+音声+画像 | テキスト+音声+画像+動画 | テキスト+音声 |

### 比較表 2: 音声認識技術の比較

| モデル | 開発元 | パラメータ | 対応言語 | WER (英語) | ローカル実行 |
|--------|-------|----------|---------|-----------|------------|
| Whisper large-v3 | OpenAI | 1.5B | 100+ | 3.0% | 可（GPU推奨） |
| Gemini ASR | Google | 非公開 | 100+ | 2.8% | 一部 |
| Azure Speech | Microsoft | 非公開 | 100+ | 3.5% | 不可 |
| Vosk | Alpha Cephei | ~50M | 20+ | 8.0% | 可（CPU可） |
| Whisper tiny | OpenAI | 39M | 100+ | 8.5% | 可（CPU可） |

---

## 4. アンチパターン

### アンチパターン 1: すべてをLLMに委ねる

```
❌ 悪い例:
「タイマーを3分にセット」→ LLM（GPT-4）に送信して応答を待つ
→ 2秒の遅延、不必要なAPI費用

✅ 正しいアプローチ:
- 単純なコマンド（タイマー、アラーム、電話）→ ルールベースで即座に実行
- 複雑な質問（調べもの、要約、創作）→ LLMにルーティング
- ハイブリッド設計: インテント分類器が振り分ける
```

### アンチパターン 2: コンテキストを無視した単発応答

```
❌ 悪い例:
ユーザー: 「東京の天気は？」→ 「晴れです」
ユーザー: 「じゃあ明日は？」→ 「何についてですか？」（文脈喪失）

✅ 正しいアプローチ:
- 対話履歴（直近5〜10ターン）をLLMコンテキストに含める
- エンティティ解決: 「じゃあ明日は」→ 「東京の明日の天気」と推論
- セッション管理で文脈を維持する
```

---

## 5. FAQ

### Q1: Siri は ChatGPT で何が変わりましたか？

**A:** Apple Intelligence により、Siri は画面上のコンテキストを理解し、アプリ間で横断的な操作が可能になりました。例えば「昨日友人から送られた写真をメールに添付して」のような複合タスクを処理できます。複雑な質問は ChatGPT にオフロードされますが、ユーザーの許可が毎回求められ、プライバシーが保護されます。

### Q2: 音声アシスタントの応答速度を改善するには？

**A:** 主な改善ポイントは以下の3つです:
1. **Wake Word検出をオンデバイスに** — ネットワーク遅延ゼロで起動
2. **ストリーミングASR** — 発話完了前から認識を開始
3. **投機的実行** — NLUが高確信度のインテントを検出したら、ASR完了前にAPI呼び出しを開始

### Q3: 自作の音声アシスタントを作るにはどうすればよいですか？

**A:** 最小構成は以下の通りです:
1. **音声認識**: Whisper（ローカル）または Google Speech-to-Text（クラウド）
2. **対話管理**: Ollama + ローカルLLM または OpenAI API
3. **音声合成**: pyttsx3（ローカル）、VOICEVOX（高品質日本語）、または OpenAI TTS
4. **統合**: Python でパイプラインを構築（コード例5参照）

---

## まとめ

| 項目 | ポイント |
|------|---------|
| パイプライン | Wake Word → ASR → NLU → Dialog → Action → TTS |
| LLM統合 | 複雑な質問・マルチステップタスクをLLMが処理 |
| オンデバイス | プライバシー・低遅延のためにWake Word/ASRをローカル実行 |
| 主要プラットフォーム | Siri（Apple Intelligence）、Google（Gemini）、Alexa（Bedrock） |
| 開発手法 | App Intents / Actions SDK / Alexa Skills Kit |
| 今後の展望 | マルチモーダル対話、プロアクティブアシスタント |

---

## 次に読むべきガイド

- [ウェアラブル — Apple Watch / Galaxy Watch](./03-wearables.md)
- [音声AI概要 — TTS/STT/音楽生成](../../ai-audio-generation/docs/00-fundamentals/00-audio-ai-overview.md)
- [ボイスクローン — ElevenLabs、RVC](../../ai-audio-generation/docs/02-voice/00-voice-cloning.md)

---

## 参考文献

1. **Apple** — "Introducing Apple Intelligence," apple.com, 2024
2. **Google** — "Gemini in Google Assistant," blog.google, 2024
3. **Amazon** — "Alexa LLM and Conversational AI," developer.amazon.com, 2024
4. **Radford, A. et al.** — "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)," arXiv:2212.04356, 2022
