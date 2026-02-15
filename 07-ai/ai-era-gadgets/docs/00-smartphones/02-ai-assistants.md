# AIアシスタント — Siri / Google Assistant / Alexa と LLM統合

> 音声AIアシスタントの仕組みから、音声認識パイプライン、LLM（大規模言語モデル）との統合、そしてカスタム音声アプリの開発手法まで体系的に解説する。

---

## この章で学ぶこと

1. **音声認識パイプラインの構造** — 音声入力から意図理解・応答生成までの処理フロー
2. **主要アシスタントの技術比較** — Siri / Google Assistant / Alexa の設計思想と強み
3. **LLM統合の最前線** — ChatGPT / Gemini によるアシスタント進化と開発手法
4. **音声アプリの実装** — カスタムスキル・アクション開発の実践的手法
5. **ローカルLLM音声アシスタント** — プライバシー重視のオンデバイス音声AI構築

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

### 1.2 Wake Word 検出の技術詳細

Wake Word（ウェイクワード）検出は、常にマイクを監視しつつ最小限の消費電力で動作する必要がある、音声アシスタントの最も重要なコンポーネントです。

```
┌─────────────────────────────────────────────────────┐
│       Wake Word 検出のアーキテクチャ                   │
│                                                       │
│  マイク入力（常時）                                    │
│      │                                                │
│      ▼                                                │
│  ┌──────────────────────────┐                         │
│  │ DSP (Digital Signal      │  ← 超低消費電力（~1mW） │
│  │  Processor)              │     常時リスニング       │
│  │ - VAD（音声区間検出）    │                         │
│  │ - 前処理（ノイズ除去）    │                         │
│  └────────────┬─────────────┘                         │
│               │ 音声検出時のみ                         │
│               ▼                                       │
│  ┌──────────────────────────┐                         │
│  │ NPU / 軽量CNN            │  ← 低消費電力（~10mW）  │
│  │ - "Hey Siri" 検出        │     小さなキーワードモデル│
│  │ - 話者識別（誰の声か）    │     ~200KB モデル        │
│  └────────────┬─────────────┘                         │
│               │ ウェイクワード検出時                    │
│               ▼                                       │
│  ┌──────────────────────────┐                         │
│  │ メインプロセッサ起動      │  ← 通常消費電力          │
│  │ - フルASR開始            │     Whisper等の大型モデル │
│  │ - クラウド接続            │                         │
│  └──────────────────────────┘                         │
│                                                       │
│  バッテリー影響:                                       │
│  DSP常時リスニング: 1日あたり ~1-2% のバッテリー消費   │
│  (NPUに移行することでさらに効率化が進む)               │
└─────────────────────────────────────────────────────┘
```

### 1.3 音声認識（ASR）のストリーミング処理

```
┌─────────────────────────────────────────────────────┐
│     ストリーミングASRの処理フロー                      │
│                                                       │
│  ユーザー発話:                                        │
│  "明日  の  天気  を  教えて  ください"                │
│   │     │    │    │    │       │                      │
│   ▼     ▼    ▼    ▼    ▼       ▼                      │
│  [チャンク1][チャンク2][チャンク3]                      │
│   │                                                   │
│   ▼                                                   │
│  ストリーミングデコーダ                                │
│   │                                                   │
│   ├── 部分結果: "あした"                              │
│   ├── 部分結果: "あしたの てんき"                     │
│   ├── 部分結果: "あしたの てんきを おしえて"          │
│   └── 最終結果: "明日の天気を教えてください"           │
│                                                       │
│  メリット:                                            │
│  - 発話完了前から処理開始 → 体感遅延の大幅削減        │
│  - 部分結果でインテント推定を先行実行（投機的実行）    │
│  - 確信度が高ければASR完了前にAPI呼び出し開始          │
│                                                       │
│  レイテンシ比較:                                      │
│  バッチASR:     発話完了 → 全体認識 → 結果 (~1.5秒)  │
│  ストリーミング: 発話中 → 逐次認識 → 結果 (~0.3秒)    │
└─────────────────────────────────────────────────────┘
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

### コード例 6: Apple App Intents（iOS 16+）による Siri 統合

```swift
import AppIntents

/// App Intents フレームワークによるSiri統合（iOS 16+）
/// 従来の SiriKit Intent Definition よりも簡潔で型安全
struct SearchRecipeIntent: AppIntent {
    static var title: LocalizedStringResource = "レシピ検索"
    static var description = IntentDescription("食材からレシピを検索します")

    // Siriに自然言語で問いかけるとこのパラメータが自動抽出される
    @Parameter(title: "食材")
    var ingredient: String

    @Parameter(title: "調理時間（分）", default: 30)
    var maxCookingTime: Int

    // Siri Shortcuts アプリにも自動表示される
    static var parameterSummary: some ParameterSummary {
        Summary("「\(\.$ingredient)」を使った\(\.$maxCookingTime)分以内のレシピ")
    }

    func perform() async throws -> some IntentResult & ProvidesDialog & ShowsSnippetView {
        // レシピ検索ロジック
        let recipes = try await RecipeService.search(
            ingredient: ingredient,
            maxTime: maxCookingTime
        )

        guard let topRecipe = recipes.first else {
            return .result(
                dialog: "\(ingredient)を使ったレシピが見つかりませんでした。"
            )
        }

        // Siri応答 + リッチUIスニペット
        return .result(
            dialog: "\(topRecipe.name)はいかがですか？調理時間は\(topRecipe.cookingTime)分です。",
            view: RecipeSnippetView(recipe: topRecipe)
        )
    }
}

/// Shortcuts アプリで表示されるショートカットプロバイダ
struct RecipeShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: SearchRecipeIntent(),
            phrases: [
                "「\(.applicationName)」で\(\.$ingredient)のレシピを探して",
                "「\(.applicationName)」で簡単な料理を提案して",
            ],
            shortTitle: "レシピ検索",
            systemImageName: "fork.knife"
        )
    }
}
```

### コード例 7: Function Calling による外部API連携アシスタント

```python
import openai
import json
import requests

class FunctionCallingAssistant:
    """
    Function Calling を使った高度なAIアシスタント
    LLMが適切なAPIを自動選択して実行する

    なぜ Function Calling か:
    - 従来のインテント分類では対応できない複雑なクエリに対応
    - LLMが文脈に応じて適切な関数を自動選択
    - 複数の関数を順次呼び出すチェーン実行が可能
    """

    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "指定した場所の天気予報を取得する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "都市名"},
                            "date": {"type": "string", "description": "日付（YYYY-MM-DD）"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_restaurant",
                    "description": "レストランを検索する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "cuisine": {"type": "string", "description": "料理のジャンル"},
                            "budget": {"type": "integer", "description": "予算（円）"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_reminder",
                    "description": "リマインダーを設定する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "datetime": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                        },
                        "required": ["title", "datetime"]
                    }
                }
            }
        ]

    def chat(self, user_message, conversation_history=None):
        """ユーザーメッセージを処理し、必要に応じてAPIを呼び出す"""
        if conversation_history is None:
            conversation_history = []

        messages = [
            {"role": "system", "content": "あなたは親切な日本語アシスタントです。"
             "ユーザーの要望に応じて適切なツールを使用してください。"},
            *conversation_history,
            {"role": "user", "content": user_message}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # Function Call が要求された場合
        if message.tool_calls:
            results = []
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                # 関数を実行
                result = self._execute_function(func_name, args)
                results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False)
                })

            # 関数の結果をLLMに渡して最終応答を生成
            messages.append(message)
            messages.extend(results)

            final_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            return final_response.choices[0].message.content

        return message.content

    def _execute_function(self, name, args):
        """関数を実行して結果を返す"""
        if name == "get_weather":
            return {"condition": "晴れ", "temp": 22, "rain_prob": 10}
        elif name == "search_restaurant":
            return {"name": "鮨かねさか", "rating": 4.8, "price": "¥15,000"}
        elif name == "set_reminder":
            return {"status": "success", "id": "rem_123"}
        return {"error": "Unknown function"}

# 使用例
assistant = FunctionCallingAssistant(api_key="sk-...")
# 複合クエリ: 天気確認 + レストラン検索を自動で実行
response = assistant.chat(
    "明日の東京の天気を調べて、天気が良ければ表参道でランチのお店を探して"
)
print(response)
```

### コード例 8: Home Assistant ローカル音声パイプライン（Wyoming Protocol）

```python
"""
Home Assistant Wyoming Protocol を使ったローカル音声パイプライン

Wyoming Protocol は音声処理コンポーネント間の通信プロトコル:
  マイク → Wake Word (openWakeWord) → ASR (Whisper) →
  Intent → TTS (Piper) → スピーカー

全てローカルで動作し、クラウド不要
"""
import asyncio
import json
from wyoming.server import AsyncServer
from wyoming.asr import Transcribe, Transcript
from wyoming.wake import Detection
import whisper
import numpy as np

class LocalASRServer:
    """Whisper ベースのローカルASRサーバー"""

    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)
        print(f"Whisper {model_name} モデルロード完了")

    async def handle_client(self, reader, writer):
        """Wyoming プロトコルでASRリクエストを処理"""
        # 音声データを受信
        audio_data = await self._receive_audio(reader)

        # Whisper で認識
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = self.model.transcribe(audio_np, language="ja", fp16=False)

        transcript = result["text"].strip()
        print(f"認識結果: {transcript}")

        # Wyoming プロトコルで結果を返す
        response = Transcript(text=transcript)
        await self._send_response(writer, response)

    async def _receive_audio(self, reader):
        """音声データの受信"""
        chunks = []
        while True:
            data = await reader.read(4096)
            if not data:
                break
            chunks.append(data)
        return b"".join(chunks)

    async def _send_response(self, writer, response):
        """応答の送信"""
        writer.write(json.dumps({"text": response.text}).encode())
        await writer.drain()
        writer.close()

async def main():
    server = LocalASRServer(model_name="small")
    srv = await asyncio.start_server(
        server.handle_client, "0.0.0.0", 10300  # Wyoming ASR ポート
    )
    print("Wyoming ASR サーバー起動: port 10300")
    async with srv:
        await srv.serve_forever()

# 起動: python wyoming_asr.py
# Home Assistant で Wyoming 統合を追加して接続
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

### 比較表 3: TTS（音声合成）技術の比較

| エンジン | 開発元 | 日本語品質 | レイテンシ | ローカル実行 | コスト |
|---------|-------|----------|----------|------------|--------|
| OpenAI TTS | OpenAI | 非常に高い | ~500ms | 不可 | API課金 |
| Google Cloud TTS | Google | 非常に高い | ~300ms | 不可 | API課金 |
| Amazon Polly | AWS | 高い | ~200ms | 不可 | API課金 |
| Piper | Rhasspy | 中〜高 | ~50ms | 可（CPU可） | 無料 |
| VOICEVOX | VOICEVOX | 高い（キャラ音声） | ~100ms | 可 | 無料 |
| Style-TTS 2 | 研究 | 高い | ~200ms | 可（GPU推奨） | 無料 |

### 比較表 4: 音声アシスタント開発プラットフォーム比較

| 項目 | Alexa Skills Kit | Google Actions | Apple App Intents | Rasa + Wyoming |
|------|-----------------|---------------|-------------------|---------------|
| 開発言語 | Python/Node.js | Node.js | Swift | Python |
| ホスティング | AWS Lambda | Firebase | App内蔵 | セルフホスト |
| NLU | Alexa NLU | Dialogflow | 自動 | Rasa NLU |
| LLM統合 | Bedrock | Vertex AI | ChatGPT連携 | Ollama等 |
| 収益化 | In-Skill Purchases | - | App Store | - |
| プライバシー | クラウド必須 | クラウド必須 | オンデバイス可 | 完全ローカル可 |
| 日本語対応 | 完全対応 | 完全対応 | 完全対応 | コミュニティ |

---

## 4. 実践的ユースケース

### ユースケース 1: マルチモーダル音声アシスタント

```python
class MultimodalAssistant:
    """
    テキスト + 音声 + 画像を統合するマルチモーダルアシスタント
    例: 「この写真の料理のカロリーを教えて」と音声で質問 + 写真撮影
    """
    def __init__(self):
        self.asr = whisper.load_model("base")
        self.vlm_client = openai.OpenAI()  # GPT-4V

    async def process_multimodal_query(self, audio_data, image_data=None):
        """音声 + 画像のマルチモーダルクエリを処理"""

        # 1. 音声認識
        text = self.asr.transcribe(audio_data, language="ja")["text"]
        print(f"認識テキスト: {text}")

        # 2. 画像がある場合はマルチモーダルLLMで処理
        if image_data:
            import base64
            image_b64 = base64.b64encode(image_data).decode()

            response = self.vlm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content

        # 3. テキストのみの場合
        response = self.vlm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": text}]
        )
        return response.choices[0].message.content
```

### ユースケース 2: プロアクティブアシスタント

```python
class ProactiveAssistant:
    """
    ユーザーが明示的に指示しなくても、コンテキストに基づいて
    先回りして情報を提供するプロアクティブアシスタント

    なぜプロアクティブか:
    - 従来のアシスタントは「質問→回答」の受動型
    - プロアクティブ型は状況を監視し、適切なタイミングで介入
    - Apple Intelligence の「Personal Context」がこの方向性
    """
    def __init__(self):
        self.context_store = {}
        self.rules = []

    def update_context(self, context_type, data):
        """コンテキスト情報を更新"""
        self.context_store[context_type] = {
            "data": data,
            "timestamp": time.time()
        }
        self._evaluate_rules()

    def _evaluate_rules(self):
        """ルールを評価してプロアクティブな提案を生成"""
        suggestions = []

        # ルール1: 出発時刻が近い場合、交通情報を提供
        if "calendar" in self.context_store and "location" in self.context_store:
            next_event = self.context_store["calendar"]["data"]
            if next_event.get("departure_in_minutes", float('inf')) < 30:
                weather = self._get_weather()
                traffic = self._get_traffic(next_event["location"])
                suggestions.append(
                    f"{next_event['title']}の{next_event['departure_in_minutes']}分前です。"
                    f"現在の交通状況: {traffic['duration']}分。"
                    f"{'傘を忘れずに。' if weather['rain_prob'] > 50 else ''}"
                )

        # ルール2: 異常な健康データを検出
        if "health" in self.context_store:
            hr = self.context_store["health"]["data"].get("heart_rate", 0)
            if hr > 100 and self.context_store.get("activity", {}).get("data", {}).get("type") == "resting":
                suggestions.append(
                    f"安静時心拍数が{hr}BPMと高めです。"
                    "ストレスや脱水の可能性があります。水分を取りましょう。"
                )

        # ルール3: 定期的なタスクのリマインド
        if "habits" in self.context_store:
            habits = self.context_store["habits"]["data"]
            for habit in habits:
                if habit["due"] and not habit["completed"]:
                    suggestions.append(f"まだ「{habit['name']}」が完了していません。")

        for suggestion in suggestions:
            self._notify_user(suggestion)

    def _notify_user(self, message):
        """ユーザーに通知"""
        print(f"[プロアクティブ提案] {message}")
```

---

## 5. トラブルシューティング

### 問題 1: 音声認識の精度が低い

```
症状: アシスタントが発話内容を正しく認識しない

対処法:
1. 環境ノイズの問題
   → 静かな環境で使用する
   → ビームフォーミングマイク搭載デバイスを使用
   → Whisper の場合、--condition_on_previous_text=False で誤認識連鎖を防止

2. 言語・方言の問題
   → 音声認識の言語設定を確認
   → Google Assistant: 日本語(日本)を明示設定
   → Whisper: language="ja" を明示指定

3. マイクの問題
   → マイクの権限が許可されているか確認
   → マイクにカバーやケースが被っていないか確認
   → Bluetooth接続のマイクは遅延が大きい場合がある

4. ネットワーク遅延
   → Wi-Fi 接続を確認（クラウドASRの場合）
   → オフライン認識を有効化（対応デバイスの場合）
```

### 問題 2: レスポンスが遅い

```
症状: 音声コマンドから応答まで数秒かかる

チェックポイント:
1. ASR処理時間
   → ストリーミングASR を有効化（Google: ストリーミング認識を使用）
   → ローカルASRの場合、Whisper tiny/base にダウングレード

2. LLM応答時間
   → GPT-4 → GPT-3.5-turbo に切り替え（速度重視の場合）
   → ローカルLLM: Gemma 2B / Phi-3 Mini を使用
   → ストリーミング応答を有効化

3. TTS処理時間
   → Piper（ローカル）: ~50ms で高速
   → OpenAI TTS: ストリーミング対応で体感速度改善
   → 最初の文だけ先にTTS → 残りを並列処理

4. ネットワーク遅延
   → CDNに近いリージョンのAPIを使用
   → WebSocket で持続接続（HTTP毎回接続より高速）
```

### 問題 3: Wake Word の誤検出

```
症状: アシスタントが呼ばれていないのに起動する

対処法:
1. Wake Word モデルの感度調整
   → Alexa: 設定 > ウェイクワード感度を「低」に変更
   → Google: 「OK Google」の再トレーニング
   → Apple: 「Hey Siri」の再学習

2. 環境音対策
   → テレビやラジオの音声に反応する場合
   → デバイスをスピーカーから離す
   → マルチマイクデバイスはビームフォーミングで話者方向を限定

3. 話者認識の活用
   → 登録した声のみに反応する設定を有効化
   → Apple: 「Hey Siri」に個人の声を認識させる
   → Google: Voice Match で家族の声を登録

4. openWakeWord（ローカル）の場合
   → 検出閾値を 0.5 → 0.7 に引き上げ
   → カスタムウェイクワードの学習データを増やす
```

### 問題 4: スマートホーム連携が動作しない

```
症状: 「リビングの照明をつけて」が動作しない

対処法:
1. デバイス名の問題
   → アシスタントに登録されたデバイス名を確認
   → 「リビングの照明」ではなく登録名「リビングライト」を使用
   → デバイス名を短く明確にリネーム

2. アカウント連携の問題
   → スマートホームプロバイダとの連携を再設定
   → OAuth トークンの有効期限切れを確認
   → Home Assistant: Nabu Casa クラウド接続を確認

3. ネットワークの問題
   → IoTデバイスがWi-Fiに接続されているか確認
   → VLANを使用している場合、mDNS/UPnPの転送設定を確認
   → Thread/Zigbee: Border Router が動作しているか確認
```

---

## 6. パフォーマンス最適化Tips

### Tip 1: End-to-End レイテンシの削減

```
┌────────────────────────────────────────────────────┐
│     音声アシスタント レイテンシ最適化                │
├────────────────────────────────────────────────────┤
│                                                      │
│  従来型パイプライン (合計: 2-5秒)                    │
│  ASR(1s) → NLU(0.3s) → API(0.5s) → TTS(0.5s)     │
│                                                      │
│  最適化後パイプライン (合計: 0.5-1.5秒)             │
│                                                      │
│  1. ストリーミングASR: 発話中から認識開始            │
│     → 発話完了時にはほぼ認識完了                     │
│                                                      │
│  2. 投機的実行: ASR部分結果でIntent推定開始          │
│     → "明日の天気" の時点で Weather API を先行呼出   │
│                                                      │
│  3. ストリーミングTTS: LLMの最初のトークンで         │
│     TTS開始、音声生成と出力を並列実行                │
│                                                      │
│  4. キャッシュ: 頻出クエリの結果をキャッシュ         │
│     → 「今何時？」等はローカルで即時応答             │
│                                                      │
│  5. プリウォーム: Wake Word検出時に                  │
│     LLMコネクション確立とモデルロードを先行実行      │
└────────────────────────────────────────────────────┘
```

### Tip 2: Whisper モデルの選択ガイド

```
Whisper モデル選択フローチャート:

デバイスのスペックは？
    │
    ├── スマートフォン / Raspberry Pi
    │   → Whisper tiny (39M, ~1GB RAM)
    │   → 精度: WER 8.5% (英語)
    │   → 速度: リアルタイムの ~6倍速
    │
    ├── ノートPC (CPU only)
    │   → Whisper base (74M, ~2GB RAM)
    │   → 精度: WER 5.0% (英語)
    │   → 速度: リアルタイムの ~4倍速
    │
    ├── デスクトップ (GPU あり)
    │   → Whisper small (244M, ~4GB RAM)
    │   → 精度: WER 3.4% (英語)
    │   → 速度: リアルタイムの ~15倍速
    │
    └── サーバー (H100等)
        → Whisper large-v3 (1.5B, ~10GB RAM)
        → 精度: WER 3.0% (英語)
        → 速度: リアルタイムの ~50倍速

日本語の場合:
  - large-v3 が最も高精度
  - base でも実用的な精度（CER ~10%）
  - faster-whisper (CTranslate2) で 2-4倍高速化
  - distil-whisper で精度を維持しつつ 6倍高速化
```

### Tip 3: コスト最適化

```
音声アシスタントの運用コスト比較（1000リクエスト/日の場合）:

┌───────────────────────────────────────────────────┐
│  構成パターンA: フルクラウド                       │
│  ASR: Google Cloud Speech ($0.006/15s)             │
│  LLM: GPT-4o ($0.01/1K tokens * 500 tokens avg)   │
│  TTS: Google Cloud TTS ($4/1M chars)               │
│  月額: ~$200-400                                   │
├───────────────────────────────────────────────────┤
│  構成パターンB: ハイブリッド                       │
│  ASR: Whisper (ローカル、無料)                     │
│  LLM: GPT-3.5-turbo ($0.002/1K tokens)            │
│  TTS: Piper (ローカル、無料)                       │
│  月額: ~$30-60                                     │
├───────────────────────────────────────────────────┤
│  構成パターンC: 完全ローカル                       │
│  ASR: Whisper (ローカル)                           │
│  LLM: Ollama + Gemma 9B (ローカル)                │
│  TTS: Piper / VOICEVOX (ローカル)                  │
│  月額: $0 (電気代のみ)                             │
│  ※ 初期投資: PC ($1,000-2,000)                     │
└───────────────────────────────────────────────────┘
```

---

## 7. 設計パターン

### パターン 1: ハイブリッドルーティング

```python
class HybridRouter:
    """
    単純なコマンドはルールベースで即時処理、
    複雑な質問はLLMにルーティングするハイブリッド設計

    なぜハイブリッドか:
    - 「タイマー3分」にLLMは不要（遅延・コストの無駄）
    - 「明日の天気を考慮してコーデを提案」はLLMが適切
    """
    def __init__(self):
        self.simple_commands = {
            "タイマー": self._handle_timer,
            "アラーム": self._handle_alarm,
            "音量": self._handle_volume,
            "電話": self._handle_call,
        }
        self.llm = OllamaClient()

    def route(self, text):
        # 1. 単純コマンドのマッチング（正規表現ベース）
        for keyword, handler in self.simple_commands.items():
            if keyword in text:
                return handler(text), "rule"

        # 2. LLMにルーティング
        return self.llm.chat(text), "llm"

    def _handle_timer(self, text):
        import re
        match = re.search(r'(\d+)\s*分', text)
        if match:
            minutes = int(match.group(1))
            return f"タイマーを{minutes}分にセットしました。"
        return "何分のタイマーですか？"
```

### パターン 2: コンテキスト維持型対話管理

```python
class ContextualDialogManager:
    """
    対話コンテキストを維持し、自然な連続会話を実現する

    問題: 「東京の天気は？」→「じゃあ明日は？」
    → 文脈がないと「何のことですか？」になる

    解決: 直近の対話履歴を保持し、代名詞・省略を解決
    """
    def __init__(self, max_history=10):
        self.history = []
        self.entities = {}  # 抽出されたエンティティのキャッシュ
        self.max_history = max_history

    def process(self, user_input):
        # エンティティ追跡
        new_entities = self._extract_entities(user_input)
        self.entities.update(new_entities)

        # 省略された情報を補完
        enriched_input = self._resolve_references(user_input)

        # 対話履歴に追加
        self.history.append({"role": "user", "content": enriched_input})

        # LLMで応答生成（履歴を含む）
        response = self._generate_response()

        self.history.append({"role": "assistant", "content": response})

        # 履歴の上限管理
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        return response

    def _extract_entities(self, text):
        """テキストからエンティティを抽出"""
        entities = {}
        # 場所の抽出
        locations = ["東京", "大阪", "名古屋", "福岡", "札幌"]
        for loc in locations:
            if loc in text:
                entities["location"] = loc
        # 日付の抽出
        if "明日" in text:
            entities["date"] = "tomorrow"
        elif "今日" in text:
            entities["date"] = "today"
        return entities

    def _resolve_references(self, text):
        """代名詞や省略を解決"""
        # 「じゃあ明日は？」→ 「じゃあ東京の明日の天気は？」
        if len(text) < 10 and "location" in self.entities:
            if "明日" in text or "今日" in text:
                text = f"{self.entities['location']}の{text}"
        return text
```

---

## 8. アンチパターン

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

### アンチパターン 3: エラーハンドリングなしの音声パイプライン

```
❌ 悪い例:
ASR失敗 → クラッシュ
LLM タイムアウト → 無応答のまま固まる

✅ 正しいアプローチ:
- ASR失敗時: 「すみません、聞き取れませんでした。もう一度お願いします。」
- LLMタイムアウト: 「少々お待ちください...」→ 5秒後に「処理に時間がかかっています」
- ネットワークエラー: ローカルフォールバック（オフライン応答）
- TTS失敗: テキスト表示にフォールバック
```

### アンチパターン 4: プライバシーを考慮しない設計

```
❌ 悪い例:
- 全音声データをクラウドに送信して永続保存
- ユーザーの同意なく会話ログを学習データに使用
- 子供の音声を区別せず処理

✅ 正しいアプローチ:
- Wake Word 検出前の音声はデバイスから出さない
- 音声データの自動削除ポリシーを設定（30日等）
- ユーザーに録音データの閲覧・削除機能を提供
- 子供アカウントには COPPA 準拠の制限を適用
- ローカル処理オプション（Whisper + Ollama）を提供
```

---

## 9. エッジケース分析

### エッジケース 1: 多言語混在の発話

日本語と英語が混在する発話（コードスイッチング）は、ASRモデルにとって困難なケースです。

```
例: 「ChatGPTのAPIキーをSlackにセットアップして」

課題:
- "ChatGPT", "API", "Slack" は英語の固有名詞
- "キー", "セットアップ" は外来語（カタカナ）
- 日本語モードの ASR は英語部分を誤認識しやすい

対策:
1. Whisper large-v3 は多言語対応で混在に強い
2. 後処理で固有名詞を辞書マッチングで補正
3. ホットワード機能（Whisper の initial_prompt パラメータ）
   → initial_prompt="ChatGPT, API, Slack, セットアップ"
4. ドメイン特化の語彙リストを ASR に提供
```

### エッジケース 2: 騒音環境での音声認識

```
環境ノイズ別の認識精度低下:

| 環境 | SNR (dB) | WER増加率 | 対策 |
|------|----------|----------|------|
| 静かなオフィス | 30+ | +0% | 対策不要 |
| カフェ | 15-20 | +5-10% | ビームフォーミング |
| 車内（走行中） | 10-15 | +10-20% | ノイズキャンセリング |
| 工事現場 | 0-5 | +30-50% | 近接マイク必須 |
| 音楽再生中 | 5-10 | +20-30% | AEC（音響エコー除去） |

技術的対策:
1. AEC (Acoustic Echo Cancellation)
   → スピーカーから出ている音楽/応答音を除去
2. Beamforming
   → 複数マイクで話者方向の音を強調
3. RNNoise / DeepFilterNet
   → AIベースのリアルタイムノイズ除去
4. VAD (Voice Activity Detection)
   → 人の声がある区間のみASRに送信
```

---

## 10. 開発者チェックリスト

```
音声アシスタント開発チェックリスト:

□ 音声認識（ASR）
  □ Whisper / Google STT / Azure Speech の選定
  □ ストリーミング認識の実装
  □ 言語・方言の設定
  □ ノイズ耐性のテスト

□ 自然言語理解（NLU）
  □ インテント分類の設計
  □ スロット/エンティティの定義
  □ LLM vs ルールベースのルーティング
  □ コンテキスト管理の実装

□ 応答生成
  □ LLMの選定（クラウド vs ローカル）
  □ Function Calling の設計
  □ 安全フィルターの実装
  □ レスポンス長の制限

□ 音声合成（TTS）
  □ TTS エンジンの選定
  □ 日本語の自然さ確認
  □ ストリーミングTTSの実装

□ パフォーマンス
  □ End-to-End レイテンシ < 2秒
  □ Wake Word 誤検出率 < 1%
  □ ASR 精度テスト
  □ バッテリー影響の計測

□ プライバシー
  □ 音声データの保存ポリシー
  □ ユーザー同意フローの実装
  □ データ削除機能の提供
```

---

## FAQ

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

### Q4: Alexa Skills と Google Actions の開発、どちらが始めやすいですか？

**A:** Alexa Skills Kit の方が入門には適しています。理由は: 1) ドキュメントが充実している、2) AWS Lambda との統合が簡単、3) 無料枠が広い。一方、Google Actions は Dialogflow との統合で複雑な対話設計がしやすく、多言語対応が強力です。日本市場ではどちらも利用可能ですが、Alexaの方がスキルストアのエコシステムが大きいです。

### Q5: ローカル音声アシスタントのプライバシーはどの程度保護されますか？

**A:** 完全ローカル構成（Whisper + Ollama + Piper）では、音声データが一切外部に送信されません。インターネット接続なしで動作するため、盗聴やデータ漏洩のリスクがゼロです。ただし、ローカルLLMの品質はクラウドLLM（GPT-4, Gemini）より劣るため、精度とプライバシーのトレードオフを考慮する必要があります。Gemma 9B や Llama 3.1 8B の Q4量子化版は日常会話なら十分実用的な品質です。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| パイプライン | Wake Word → ASR → NLU → Dialog → Action → TTS |
| LLM統合 | 複雑な質問・マルチステップタスクをLLMが処理 |
| オンデバイス | プライバシー・低遅延のためにWake Word/ASRをローカル実行 |
| 主要プラットフォーム | Siri（Apple Intelligence）、Google（Gemini）、Alexa（Bedrock） |
| 開発手法 | App Intents / Actions SDK / Alexa Skills Kit |
| Function Calling | LLMが適切なAPIを自動選択して実行する設計パターン |
| ハイブリッドルーティング | 単純コマンドはルールベース、複雑な質問はLLM |
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
5. **Home Assistant** — "Wyoming Protocol for Voice," home-assistant.io, 2024
6. **OpenAI** — "Function Calling Guide," platform.openai.com, 2024
