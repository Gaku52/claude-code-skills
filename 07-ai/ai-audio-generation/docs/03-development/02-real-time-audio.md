# リアルタイム音声 — WebRTC・ストリーミング STT/TTS

> 低遅延の音声通信とリアルタイム音声認識・合成を組み合わせ、対話的音声アプリケーションを構築する技術を体系的に学ぶ。

---

## この章で学ぶこと

1. **WebRTC 基盤** — ブラウザ間・サーバー間のリアルタイム音声通信の仕組みとシグナリング設計
2. **ストリーミング STT** — 音声をリアルタイムでテキスト変換するアーキテクチャと実装
3. **ストリーミング TTS** — テキストをリアルタイムで音声合成し低遅延で配信する手法

---

## 1. リアルタイム音声アーキテクチャ

### 1.1 全体構成

```
+-------------------+                    +-------------------+
|  クライアント A    |                    |  クライアント B    |
|  +------+         |    WebRTC          |         +------+  |
|  | マイク | -------+----- P2P ---------+-------- | スピーカ| |
|  +------+         |                    |         +------+  |
|  | スピーカ| <------+----- P2P ---------+-------- | マイク | |
|  +------+         |                    |         +------+  |
+-------------------+                    +-------------------+
         |                                        |
         |  音声ストリーム                         |
         v                                        v
+-----------------------------------------------------------+
|                    メディアサーバー                          |
|  +------------+  +------------+  +------------+            |
|  | STT Engine |  | AI 処理    |  | TTS Engine |            |
|  | (リアルタイム)|  | (翻訳/要約)|  | (リアルタイム)|          |
|  +------------+  +------------+  +------------+            |
+-----------------------------------------------------------+
```

### 1.2 レイテンシ要件

```
+-------------------------------------------------------+
| リアルタイム音声の遅延バジェット                         |
+-------------------------------------------------------+
| 全体目標: < 300ms (会話として自然)                      |
|                                                         |
| 内訳:                                                   |
|  音声キャプチャ   : 10-20ms  [===]                      |
|  エンコード       : 5-10ms   [==]                       |
|  ネットワーク転送 : 20-100ms [========]                  |
|  デコード         : 5-10ms   [==]                       |
|  STT 処理        : 50-200ms [==============]            |
|  AI 処理         : 50-500ms [====================]      |
|  TTS 処理        : 50-200ms [==============]            |
|  再生バッファ     : 10-20ms  [===]                      |
|                                                         |
|  合計: 200-1060ms (AI処理込みだと厳しい)                |
|  → ストリーミングで並列化が必須                          |
+-------------------------------------------------------+
```

---

## 2. WebRTC 基盤

### 2.1 シグナリングとピア接続

```python
# コード例 1: Python (aiortc) による WebRTC サーバー
import asyncio
import json
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

relay = MediaRelay()
pcs = set()

async def offer(request):
    """WebRTC Offer を受け取り Answer を返す"""
    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params["sdp"],
        type=params["type"]
    )

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print(f"Track received: {track.kind}")
        if track.kind == "audio":
            # 受信した音声をSTT処理パイプラインに送る
            processor = AudioProcessor(track)
            pc.addTrack(processor.output_track)

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })

app = web.Application()
app.router.add_post("/offer", offer)
web.run_app(app, port=8080)
```

### 2.2 クライアント側の実装

```javascript
// コード例 2: ブラウザ側 WebRTC + リアルタイム文字起こし表示
class RealtimeAudioClient {
  constructor(serverUrl) {
    this.serverUrl = serverUrl;
    this.pc = null;
    this.transcriptCallback = null;
  }

  async start() {
    // マイク入力を取得
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000,
      },
      video: false,
    });

    // PeerConnection を作成
    this.pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    // 音声トラックを追加
    stream.getAudioTracks().forEach((track) => {
      this.pc.addTrack(track, stream);
    });

    // サーバーからの音声トラックを受信
    this.pc.ontrack = (event) => {
      const audio = new Audio();
      audio.srcObject = event.streams[0];
      audio.play();
    };

    // DataChannel でリアルタイム文字起こしを受信
    this.dc = this.pc.createDataChannel("transcription");
    this.dc.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (this.transcriptCallback) {
        this.transcriptCallback(data);
      }
    };

    // Offer/Answer 交換
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    const response = await fetch(`${this.serverUrl}/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: offer.sdp,
        type: offer.type,
      }),
    });

    const answer = await response.json();
    await this.pc.setRemoteDescription(answer);
  }

  onTranscript(callback) {
    this.transcriptCallback = callback;
  }

  async stop() {
    if (this.pc) {
      this.pc.close();
    }
  }
}

// 使用例
const client = new RealtimeAudioClient("https://api.example.com");
client.onTranscript((data) => {
  document.getElementById("transcript").textContent += data.text;
});
await client.start();
```

---

## 3. ストリーミング STT

### 3.1 ストリーミング音声認識のアーキテクチャ

```
音声入力 (連続)
  |
  v
+--------+    +--------+    +--------+    +--------+
| 音声   | -> | VAD    | -> | チャンク | -> | STT   |
| バッファ|    | 判定   |    | 分割    |    | エンジン|
+--------+    +--------+    +--------+    +--------+
                                               |
                                    +----------+----------+
                                    |                     |
                              +---------+           +---------+
                              | 中間結果 |           | 確定結果 |
                              | (partial)|           | (final) |
                              +---------+           +---------+
                                    |                     |
                                    v                     v
                              リアルタイム表示        後続処理
                              (タイピング風)         (翻訳/要約)
```

### 3.2 Google Cloud Speech-to-Text ストリーミング

```python
# コード例 3: Google Cloud STT ストリーミング認識
from google.cloud import speech
import pyaudio
import queue
import threading

class StreamingSTT:
    def __init__(self, language="ja-JP", sample_rate=16000):
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language,
            enable_automatic_punctuation=True,
            model="latest_long",  # 長時間向けモデル
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True,  # 中間結果を有効化
        )
        self.audio_queue = queue.Queue()
        self.sample_rate = sample_rate

    def audio_generator(self):
        """マイクからの音声を yield する"""
        while True:
            chunk = self.audio_queue.get()
            if chunk is None:
                return
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def start_microphone(self):
        """マイク入力を開始する"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1600,  # 100ms チャンク
            stream_callback=self._audio_callback,
        )
        return stream

    def _audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return None, pyaudio.paContinue

    def recognize_stream(self, on_result):
        """ストリーミング認識を実行する"""
        requests = self.audio_generator()
        responses = self.client.streaming_recognize(
            self.streaming_config, requests
        )

        for response in responses:
            for result in response.results:
                transcript = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence

                if result.is_final:
                    on_result({
                        "type": "final",
                        "text": transcript,
                        "confidence": confidence,
                    })
                else:
                    on_result({
                        "type": "partial",
                        "text": transcript,
                    })

# 使用例
stt = StreamingSTT(language="ja-JP")
mic_stream = stt.start_microphone()

def handle_result(result):
    prefix = "[確定]" if result["type"] == "final" else "[中間]"
    print(f'{prefix} {result["text"]}')

stt.recognize_stream(handle_result)
```

---

## 4. ストリーミング TTS

### 4.1 低遅延 TTS パイプライン

```python
# コード例 4: チャンク単位のストリーミング TTS
import asyncio
import edge_tts

async def streaming_tts(text: str, voice: str = "ja-JP-NanamiNeural"):
    """
    テキストを受け取り、音声チャンクをストリーミングで返す。
    最初のチャンクまでの遅延 (TTFB) を最小化する。
    """
    communicate = edge_tts.Communicate(text, voice)
    audio_chunks = []

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]
        elif chunk["type"] == "WordBoundary":
            # 単語の境界情報（リップシンク等に使用可能）
            print(f"  Word: {chunk['text']} at {chunk['offset']}ms")

async def realtime_conversation_tts(text_stream):
    """
    LLM からのテキストストリームをリアルタイムで音声に変換する。
    文単位でバッファリングし、句読点で区切って TTS に送る。
    """
    buffer = ""
    sentence_delimiters = {"。", "！", "？", ".", "!", "?", "\n"}

    async for text_chunk in text_stream:
        buffer += text_chunk

        # 文の区切りを検出
        for delimiter in sentence_delimiters:
            if delimiter in buffer:
                sentences = buffer.split(delimiter)
                for sentence in sentences[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        # 文単位で TTS に送信（並列で次の文も処理開始）
                        async for audio_chunk in streaming_tts(
                            sentence + delimiter
                        ):
                            yield audio_chunk
                buffer = sentences[-1]

    # 残りのバッファを処理
    if buffer.strip():
        async for audio_chunk in streaming_tts(buffer):
            yield audio_chunk
```

### 4.2 WebSocket による双方向ストリーミング

```python
# コード例 5: FastAPI WebSocket でリアルタイム STT + TTS
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import numpy as np

app = FastAPI()

@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()

    stt_engine = StreamingSTTEngine()
    tts_engine = StreamingTTSEngine()

    async def process_audio():
        """クライアントからの音声を STT 処理する"""
        try:
            while True:
                audio_bytes = await websocket.receive_bytes()
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                # STT でテキストに変換
                result = await stt_engine.process_chunk(audio_array)

                if result and result["type"] == "final":
                    # 確定テキストを送信
                    await websocket.send_json({
                        "type": "transcript",
                        "text": result["text"],
                    })

                    # AI 応答を生成 + TTS
                    ai_response = await generate_ai_response(result["text"])
                    async for audio_chunk in tts_engine.synthesize_stream(
                        ai_response
                    ):
                        await websocket.send_bytes(audio_chunk)

                elif result and result["type"] == "partial":
                    await websocket.send_json({
                        "type": "partial_transcript",
                        "text": result["text"],
                    })

        except WebSocketDisconnect:
            print("Client disconnected")

    await process_audio()
```

---

## 5. プロトコル・コーデック比較

### 5.1 音声コーデック比較

| コーデック | ビットレート | 遅延 | 品質 | WebRTC対応 | 用途 |
|-----------|-------------|------|------|-----------|------|
| Opus | 6-510 kbps | 5ms | 優 | ○ | 音声通話 (推奨) |
| G.711 | 64 kbps | 0.125ms | 可 | ○ | 電話 (レガシー) |
| AAC | 8-320 kbps | 20ms | 優 | △ | 配信 |
| Lyra | 3-9 kbps | 10ms | 良 | x | 低帯域環境 |
| Codec2 | 0.7-3.2 kbps | 40ms | 可 | x | IoT/組込み |

### 5.2 通信プロトコル比較

| プロトコル | 遅延 | 信頼性 | 双方向 | 用途 |
|-----------|------|--------|--------|------|
| WebRTC | 極低 | UDP (ベストエフォート) | ○ | P2P音声通話 |
| WebSocket | 低 | TCP (保証) | ○ | テキスト/制御チャネル |
| gRPC Streaming | 低 | HTTP/2 | ○ | STT/TTS API |
| HTTP SSE | 中 | HTTP/1.1 | 片方向 | TTS 出力配信 |

---

## 6. アンチパターン

### アンチパターン 1: 「全文を待ってから TTS」

```
[誤り] LLM の出力が全て完了してから TTS を開始する

  LLM生成 (3秒) ────────────> TTS処理 (2秒) ────> 再生
  ユーザー待ち時間: 5秒

[正解] 文単位でストリーミング TTS を開始する

  LLM "こんにちは。" -> TTS -> 再生開始 (0.3秒)
  LLM "今日は天気が..." -> TTS -> 再生 (並列)
  ユーザー体感遅延: 0.3秒

  ポイント:
  - 句読点で文を区切ってTTSに送る
  - 前の文の再生中に次の文のTTSを並列処理
  - オーディオバッファで途切れを防止
```

### アンチパターン 2: 「VAD なしで常時ストリーミング」

```
[誤り] 無音を含む全ての音声を常時 STT エンジンに送信する

問題点:
- 無駄な API コスト（無音区間も課金対象）
- STT エンジンが無音からノイズを誤認識
- ネットワーク帯域の無駄遣い

[正解] VAD (Voice Activity Detection) で音声区間のみ処理する
  1. クライアント側で VAD を実行（WebRTC の内蔵 VAD or Silero VAD）
  2. 音声が検出された区間のみサーバーに送信
  3. 無音が一定時間続いたら「発話終了」として確定
```

---

## 7. FAQ

### Q1: WebRTC と WebSocket のどちらを使うべきですか？

**A:** 用途によって使い分けます。

- **WebRTC**: 低遅延が最優先の音声通話。P2P で 100ms 以下の遅延を実現可能。ただし、サーバー側での音声処理（STT 等）には SFU/MCU パターンが必要
- **WebSocket**: テキストデータの送受信、制御チャネルとして使用。音声バイナリも送れるが、UDP ではないため遅延は WebRTC より大きい
- **ハイブリッド**: 音声データは WebRTC、文字起こし結果やメタデータは WebSocket で送る構成が実用的

### Q2: リアルタイム STT の精度を上げるには？

**A:** 以下のアプローチが有効です。

1. **ドメイン特化語彙**: 専門用語のブースト（Google Cloud STT の `speech_contexts` 等）
2. **ノイズ除去前処理**: RNNoise や WebRTC 内蔵のエコーキャンセラー・ノイズ抑制を活用
3. **チャンクサイズの最適化**: 短すぎると文脈不足、長すぎると遅延増大。100-300ms が一般的
4. **エンドポイント検出**: 発話終了の判定閾値を調整（早すぎる確定を防ぐ）

### Q3: 同時接続数が増えた場合のスケーリング戦略は？

**A:** 以下の戦略を段階的に適用します。

1. **SFU (Selective Forwarding Unit)**: メディアサーバーがストリームを中継。Janus, mediasoup が代表的
2. **STT/TTS の水平スケーリング**: Kubernetes 上で STT/TTS ワーカーをオートスケール
3. **リージョン分散**: ユーザーに近いリージョンにメディアサーバーを配置
4. **GPU リソース管理**: TTS/STT の GPU ワーカーを共有プール化し効率化

---

## 8. まとめ

| コンポーネント | 技術 | 推奨ツール | 遅延目標 |
|--------------|------|-----------|---------|
| 音声通信 | WebRTC | aiortc, mediasoup | < 100ms |
| シグナリング | WebSocket | FastAPI, Socket.IO | < 50ms |
| ストリーミング STT | gRPC Streaming | Google STT, Whisper Streaming | < 200ms |
| ストリーミング TTS | チャンク合成 | Edge TTS, ElevenLabs | < 300ms |
| VAD | RNN/ルールベース | Silero VAD, WebRTC VAD | < 10ms |
| コーデック | Opus | WebRTC 内蔵 | 5ms |

---

## 次に読むべきガイド

- [ポッドキャストツール](../02-voice/02-podcast-tools.md) — 録音済み音声の文字起こし・要約・編集
- [音声合成の基礎](../01-basics/01-tts-fundamentals.md) — TTS エンジンの選択と活用
- [WebRTC の基礎](../../../network-fundamentals/docs/03-protocols/05-webrtc.md) — WebRTC プロトコルの詳細

---

## 参考文献

1. Loreto, S. & Romano, S.P. (2014). "Real-Time Communication with WebRTC." *O'Reilly Media*. https://www.oreilly.com/library/view/real-time-communication-with/9781449371location/
2. Google Cloud. (2024). "Streaming Speech-to-Text." *Google Cloud Documentation*. https://cloud.google.com/speech-to-text/docs/streaming-recognize
3. Valin, J.-M. et al. (2012). "Definition of the Opus Audio Codec." *RFC 6716, IETF*. https://www.rfc-editor.org/rfc/rfc6716
