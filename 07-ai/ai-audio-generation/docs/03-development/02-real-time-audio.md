# リアルタイム音声 — WebRTC・ストリーミング STT/TTS

> 低遅延の音声通信とリアルタイム音声認識・合成を組み合わせ、対話的音声アプリケーションを構築する技術を体系的に学ぶ。

---

## この章で学ぶこと

1. **WebRTC 基盤** — ブラウザ間・サーバー間のリアルタイム音声通信の仕組みとシグナリング設計
2. **ストリーミング STT** — 音声をリアルタイムでテキスト変換するアーキテクチャと実装
3. **ストリーミング TTS** — テキストをリアルタイムで音声合成し低遅延で配信する手法
4. **メディアサーバー設計** — SFU/MCU パターン、スケーラビリティ、GPU リソース管理
5. **障害対応と品質監視** — 接続断復旧、品質メトリクス、本番運用のノウハウ

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

### 1.3 遅延最適化戦略

```
パイプライン並列化によるレイテンシ削減:
==================================================

[従来型 — 直列処理]
  音声入力 → STT完了待ち → AI処理完了待ち → TTS完了待ち → 再生
  総遅延: 2000-3000ms

[最適化型 — パイプライン並列化]
  音声入力 ──> STT(chunk1) → AI(chunk1) → TTS(chunk1) → 再生
               STT(chunk2) → AI(chunk2) → TTS(chunk2) → 再生
               STT(chunk3) → ...

  各段階がストリーミングで次段階に即座にデータを渡す
  初回応答遅延: 300-500ms
  体感遅延: 文単位で応答が聞こえるため自然

[さらなる最適化 — 投機的処理]
  STT中間結果で AI 推論を先行開始
  → 確定結果で差分のみ再計算
  → 初回応答をさらに 100-200ms 短縮可能
==================================================
```

```python
# コード例: パイプライン並列化の実装
import asyncio
from typing import AsyncIterator

class StreamingPipeline:
    """STT → AI → TTS のストリーミングパイプライン"""

    def __init__(self, stt_engine, ai_engine, tts_engine):
        self.stt = stt_engine
        self.ai = ai_engine
        self.tts = tts_engine
        self.audio_queue = asyncio.Queue()
        self.text_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

    async def run(self):
        """3つのパイプラインステージを並列実行"""
        await asyncio.gather(
            self._stt_stage(),
            self._ai_stage(),
            self._tts_stage(),
        )

    async def _stt_stage(self):
        """音声チャンクを受け取り、テキストに変換"""
        while True:
            audio_chunk = await self.audio_queue.get()
            if audio_chunk is None:
                await self.text_queue.put(None)
                break

            result = await self.stt.process_chunk(audio_chunk)
            if result and result["type"] == "final":
                await self.text_queue.put(result["text"])

    async def _ai_stage(self):
        """テキストを受け取り、AI応答を生成"""
        while True:
            text = await self.text_queue.get()
            if text is None:
                await self.response_queue.put(None)
                break

            # ストリーミングで応答生成
            buffer = ""
            async for token in self.ai.generate_stream(text):
                buffer += token
                # 句読点で区切ってTTSに送信
                if any(d in buffer for d in "。！？.!?\n"):
                    await self.response_queue.put(buffer)
                    buffer = ""
            if buffer:
                await self.response_queue.put(buffer)

    async def _tts_stage(self):
        """テキストを受け取り、音声に変換"""
        while True:
            text = await self.response_queue.get()
            if text is None:
                break

            async for audio_chunk in self.tts.synthesize_stream(text):
                yield audio_chunk
```

### 1.4 アーキテクチャパターン比較

```
+------------------------------------------------------------+
| リアルタイム音声アプリのアーキテクチャパターン               |
+------------------------------------------------------------+
|                                                              |
| 1. P2P 直接通信                                              |
|    Client A <--WebRTC--> Client B                            |
|    利点: 最低遅延、サーバー不要                               |
|    欠点: STT/TTS 処理不可、NAT越え問題                       |
|    適用: 1対1通話、スケール不要                               |
|                                                              |
| 2. SFU (Selective Forwarding Unit)                           |
|    Client A --> SFU --> Client B                             |
|                   +--> Client C                              |
|                   +--> STT/TTS処理                           |
|    利点: スケーラブル、サーバー処理可能                       |
|    欠点: サーバーコスト、やや遅延増加                         |
|    適用: グループ通話、AI音声アシスタント                     |
|                                                              |
| 3. MCU (Multipoint Control Unit)                             |
|    Client A --> MCU(ミキシング) --> Client B                  |
|    Client C -->                  --> Client D                 |
|    利点: クライアント負荷最小、一律配信                       |
|    欠点: サーバー負荷大、遅延大                               |
|    適用: 大規模会議、録画配信                                 |
|                                                              |
| 4. ハイブリッド (SFU + AI Processing)                        |
|    Client --> SFU --> AI Worker Pool --> SFU --> Client       |
|    利点: AI処理とリアルタイム性の両立                         |
|    欠点: 設計複雑                                             |
|    適用: リアルタイム翻訳、AI音声対話                         |
+------------------------------------------------------------+
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

### 2.3 TURN/STUN サーバー設計

```python
# コード例: TURN/STUN 設定とフォールバック戦略
class ICEConfig:
    """ICE (Interactive Connectivity Establishment) 設定管理"""

    def __init__(self):
        self.ice_servers = [
            # STUN サーバー（NAT 越え、無料）
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},

            # TURN サーバー（リレー、有料だが確実）
            {
                "urls": [
                    "turn:turn.example.com:3478?transport=udp",
                    "turn:turn.example.com:3478?transport=tcp",
                    "turns:turn.example.com:5349?transport=tcp",  # TLS
                ],
                "username": "user",
                "credential": "password",
            },
        ]

    def get_config(self, force_relay=False):
        """RTCConfiguration を返す"""
        config = {
            "iceServers": self.ice_servers,
            "iceTransportPolicy": "relay" if force_relay else "all",
            "bundlePolicy": "max-bundle",
            "rtcpMuxPolicy": "require",
        }
        return config

    @staticmethod
    async def test_connectivity(pc):
        """接続性テスト — STUN/TURN の到達性を確認"""
        stats = await pc.getStats()
        candidate_pairs = []

        for report in stats.values():
            if report.type == "candidate-pair" and report.nominated:
                candidate_pairs.append({
                    "local_type": report.local_candidate_type,
                    "remote_type": report.remote_candidate_type,
                    "protocol": report.protocol,
                    "rtt": report.current_round_trip_time,
                })

        return candidate_pairs
```

```javascript
// コード例: ブラウザ側 ICE 接続状態の監視
class ICEMonitor {
  constructor(pc) {
    this.pc = pc;
    this.connectionLog = [];

    // ICE 接続状態の変化を監視
    pc.oniceconnectionstatechange = () => {
      const state = pc.iceConnectionState;
      console.log(`ICE state: ${state}`);
      this.connectionLog.push({
        state,
        timestamp: Date.now(),
      });

      switch (state) {
        case "checking":
          this._showStatus("接続中...");
          break;
        case "connected":
          this._showStatus("接続完了");
          this._logConnectionType();
          break;
        case "disconnected":
          this._showStatus("切断 — 再接続中...");
          this._attemptReconnect();
          break;
        case "failed":
          this._showStatus("接続失敗");
          this._fallbackToTURN();
          break;
      }
    };

    // ICE 候補の収集を監視
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        console.log(`ICE candidate: ${event.candidate.type} ${event.candidate.protocol}`);
      }
    };
  }

  async _logConnectionType() {
    const stats = await this.pc.getStats();
    stats.forEach((report) => {
      if (report.type === "candidate-pair" && report.nominated) {
        console.log(`接続種別: ${report.localCandidateType} -> ${report.remoteCandidateType}`);
        console.log(`プロトコル: ${report.protocol}`);
        console.log(`RTT: ${report.currentRoundTripTime}ms`);
      }
    });
  }

  async _attemptReconnect() {
    // ICE restart を試行
    const offer = await this.pc.createOffer({ iceRestart: true });
    await this.pc.setLocalDescription(offer);
    // シグナリングサーバー経由で再ネゴシエーション
  }

  _fallbackToTURN() {
    // P2P 失敗時に TURN リレーにフォールバック
    console.log("TURN リレーへフォールバック");
  }

  _showStatus(message) {
    console.log(`[ICE] ${message}`);
  }
}
```

### 2.4 AudioWorklet による低遅延音声処理

```javascript
// コード例: AudioWorklet でブラウザ内リアルタイム音声処理
// audio-processor.js (AudioWorklet Processor)
class RealtimeAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(0);
    this.bufferSize = 4096; // 256ms @ 16kHz
    this.isRecording = true;

    this.port.onmessage = (event) => {
      if (event.data.type === "stop") {
        this.isRecording = false;
      }
    };
  }

  process(inputs, outputs, parameters) {
    if (!this.isRecording) return false;

    const input = inputs[0];
    if (input.length === 0) return true;

    const channelData = input[0]; // モノラル

    // バッファに追加
    const newBuffer = new Float32Array(this.buffer.length + channelData.length);
    newBuffer.set(this.buffer);
    newBuffer.set(channelData, this.buffer.length);
    this.buffer = newBuffer;

    // バッファが十分溜まったら送信
    if (this.buffer.length >= this.bufferSize) {
      // Float32 → Int16 変換
      const int16Data = new Int16Array(this.buffer.length);
      for (let i = 0; i < this.buffer.length; i++) {
        const s = Math.max(-1, Math.min(1, this.buffer[i]));
        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }

      this.port.postMessage(
        { type: "audio", data: int16Data.buffer },
        [int16Data.buffer]
      );
      this.buffer = new Float32Array(0);
    }

    return true;
  }
}

registerProcessor("realtime-audio-processor", RealtimeAudioProcessor);
```

```javascript
// コード例: AudioWorklet を使ったクライアント統合
class AudioWorkletClient {
  constructor() {
    this.audioContext = null;
    this.workletNode = null;
    this.websocket = null;
  }

  async start(wsUrl) {
    // AudioContext 初期化
    this.audioContext = new AudioContext({ sampleRate: 16000 });

    // AudioWorklet モジュールをロード
    await this.audioContext.audioWorklet.addModule("audio-processor.js");

    // マイク入力を取得
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    // AudioWorkletNode を作成
    const source = this.audioContext.createMediaStreamSource(stream);
    this.workletNode = new AudioWorkletNode(
      this.audioContext,
      "realtime-audio-processor"
    );

    // WebSocket で音声データをサーバーに送信
    this.websocket = new WebSocket(wsUrl);
    this.websocket.binaryType = "arraybuffer";

    this.workletNode.port.onmessage = (event) => {
      if (
        event.data.type === "audio" &&
        this.websocket.readyState === WebSocket.OPEN
      ) {
        this.websocket.send(event.data.data);
      }
    };

    // サーバーからの応答（文字起こし結果）を受信
    this.websocket.onmessage = (event) => {
      if (typeof event.data === "string") {
        const result = JSON.parse(event.data);
        this.onTranscript(result);
      } else {
        // バイナリデータ = TTS 音声
        this.playAudio(event.data);
      }
    };

    // 接続
    source.connect(this.workletNode);
    this.workletNode.connect(this.audioContext.destination);
  }

  onTranscript(result) {
    console.log(`[${result.type}] ${result.text}`);
  }

  async playAudio(arrayBuffer) {
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);
    source.start();
  }

  stop() {
    this.workletNode?.port.postMessage({ type: "stop" });
    this.websocket?.close();
    this.audioContext?.close();
  }
}
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

### 3.3 Whisper ストリーミング認識

```python
# コード例: faster-whisper によるリアルタイムストリーミング STT
import numpy as np
import asyncio
from faster_whisper import WhisperModel
from collections import deque

class WhisperStreamingSTT:
    """
    Whisper をストリーミング風に使用するラッパー。
    Whisper 自体はバッチ処理モデルだが、
    スライディングウィンドウで擬似ストリーミングを実現する。
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ja",
        chunk_duration: float = 2.0,       # 処理チャンク長（秒）
        overlap_duration: float = 0.5,      # オーバーラップ長（秒）
        vad_threshold: float = 0.5,
    ):
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        self.language = language
        self.sample_rate = 16000
        self.chunk_samples = int(chunk_duration * self.sample_rate)
        self.overlap_samples = int(overlap_duration * self.sample_rate)
        self.vad_threshold = vad_threshold

        # 音声バッファ
        self.audio_buffer = np.array([], dtype=np.float32)
        self.previous_text = ""
        self.confirmed_text = ""

    async def process_chunk(self, audio_chunk: np.ndarray) -> dict | None:
        """
        音声チャンクを受け取り、認識結果を返す。

        Returns:
            dict with "type" ("partial" or "final") and "text"
        """
        # バッファに追加
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # チャンクサイズに満たない場合は待機
        if len(self.audio_buffer) < self.chunk_samples:
            return None

        # VAD チェック
        energy = np.sqrt(np.mean(self.audio_buffer[-self.chunk_samples:] ** 2))
        if energy < 0.01:  # 無音判定
            return None

        # Whisper で認識
        segments, info = self.model.transcribe(
            self.audio_buffer[-self.chunk_samples:],
            language=self.language,
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters={"threshold": self.vad_threshold},
        )

        current_text = ""
        for segment in segments:
            current_text += segment.text

        if not current_text:
            return None

        # 前回との差分で中間/確定を判定
        if current_text == self.previous_text:
            # テキストが変化しない = 発話終了の可能性
            self.confirmed_text += current_text
            self.audio_buffer = self.audio_buffer[-self.overlap_samples:]
            self.previous_text = ""
            return {"type": "final", "text": current_text}
        else:
            self.previous_text = current_text
            return {"type": "partial", "text": current_text}

    def reset(self):
        """バッファをリセット"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.previous_text = ""
        self.confirmed_text = ""
```

### 3.4 Azure Speech ストリーミング

```python
# コード例: Azure Speech SDK によるストリーミング STT（詳細版）
import azure.cognitiveservices.speech as speechsdk
import json
import time

class AzureStreamingSTT:
    """Azure Speech SDK を使ったストリーミング音声認識"""

    def __init__(
        self,
        subscription_key: str,
        region: str = "japaneast",
        language: str = "ja-JP",
    ):
        self.speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key,
            region=region,
        )
        self.speech_config.speech_recognition_language = language

        # 詳細な認識設定
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "5000"  # 初期無音タイムアウト: 5秒
        )
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
            "1000"  # 発話終了判定: 1秒
        )
        self.speech_config.enable_dictation()  # ディクテーションモード

        # フレーズリスト（認識精度向上用）
        self.phrase_list = None

        # コールバック
        self.on_partial = None
        self.on_final = None
        self.on_error = None

        # 統計
        self.stats = {
            "total_recognized": 0,
            "total_duration_ms": 0,
            "errors": 0,
        }

    def add_phrases(self, phrases: list[str]):
        """認識精度を向上させるフレーズを追加"""
        self.phrase_list = phrases

    def start_continuous(self):
        """マイクからの連続音声認識を開始"""
        audio_config = speechsdk.audio.AudioConfig(
            use_default_microphone=True
        )
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )

        # フレーズリストを設定
        if self.phrase_list:
            phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(
                recognizer
            )
            for phrase in self.phrase_list:
                phrase_list_grammar.addPhrase(phrase)

        # イベントハンドラ登録
        recognizer.recognizing.connect(self._on_recognizing)
        recognizer.recognized.connect(self._on_recognized)
        recognizer.canceled.connect(self._on_canceled)
        recognizer.session_started.connect(
            lambda evt: print(f"Session started: {evt.session_id}")
        )
        recognizer.session_stopped.connect(
            lambda evt: print(f"Session stopped: {evt.session_id}")
        )

        # 連続認識開始
        recognizer.start_continuous_recognition()
        return recognizer

    def start_from_stream(self, format_info=None):
        """プッシュストリームからの音声認識"""
        if format_info is None:
            format_info = speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000,
                bits_per_sample=16,
                channels=1,
            )

        push_stream = speechsdk.audio.PushAudioInputStream(format_info)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )

        # イベントハンドラ登録（同上）
        recognizer.recognizing.connect(self._on_recognizing)
        recognizer.recognized.connect(self._on_recognized)
        recognizer.canceled.connect(self._on_canceled)

        recognizer.start_continuous_recognition()
        return recognizer, push_stream

    def _on_recognizing(self, evt):
        """中間結果のコールバック"""
        if self.on_partial:
            self.on_partial({
                "type": "partial",
                "text": evt.result.text,
                "offset": evt.result.offset,
                "duration": evt.result.duration,
            })

    def _on_recognized(self, evt):
        """確定結果のコールバック"""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            self.stats["total_recognized"] += 1
            self.stats["total_duration_ms"] += evt.result.duration / 10000

            if self.on_final:
                # 詳細な認識結果を JSON で取得
                detail_json = evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult, ""
                )
                detail = json.loads(detail_json) if detail_json else {}

                self.on_final({
                    "type": "final",
                    "text": evt.result.text,
                    "confidence": detail.get("NBest", [{}])[0].get(
                        "Confidence", 0.0
                    ),
                    "offset_ms": evt.result.offset / 10000,
                    "duration_ms": evt.result.duration / 10000,
                    "words": detail.get("NBest", [{}])[0].get(
                        "Words", []
                    ),
                })

    def _on_canceled(self, evt):
        """キャンセル/エラーのコールバック"""
        self.stats["errors"] += 1
        if self.on_error:
            self.on_error({
                "reason": str(evt.cancellation_details.reason),
                "error_details": evt.cancellation_details.error_details,
            })
```

### 3.5 gRPC ストリーミング STT サーバー

```python
# コード例: gRPC ベースのストリーミング STT サーバー
import grpc
from concurrent import futures
import numpy as np
from faster_whisper import WhisperModel

# Proto定義（概念）:
# service StreamingSTT {
#   rpc StreamRecognize(stream AudioChunk) returns (stream RecognitionResult);
# }
# message AudioChunk { bytes audio_data = 1; int32 sample_rate = 2; }
# message RecognitionResult { string text = 1; bool is_final = 2; float confidence = 3; }

class StreamingSTTServicer:
    """gRPC ストリーミング STT サービス"""

    def __init__(self):
        self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        self.active_sessions = {}

    async def StreamRecognize(self, request_iterator, context):
        """双方向ストリーミング RPC"""
        session_id = context.peer()
        audio_buffer = np.array([], dtype=np.float32)
        chunk_size = 16000 * 2  # 2秒分

        async for request in request_iterator:
            # bytes → numpy 変換
            audio_chunk = np.frombuffer(request.audio_data, dtype=np.int16)
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            audio_buffer = np.concatenate([audio_buffer, audio_float])

            # チャンクサイズに達したら認識実行
            if len(audio_buffer) >= chunk_size:
                segments, info = self.model.transcribe(
                    audio_buffer,
                    language="ja",
                    beam_size=5,
                    vad_filter=True,
                )

                text = "".join(seg.text for seg in segments)
                if text:
                    # 中間結果を返す
                    yield RecognitionResult(
                        text=text,
                        is_final=False,
                        confidence=0.0,
                    )

                # オーバーラップを残してバッファクリア
                overlap = 16000  # 1秒
                audio_buffer = audio_buffer[-overlap:]

        # 最終結果
        if len(audio_buffer) > 0:
            segments, info = self.model.transcribe(audio_buffer, language="ja")
            text = "".join(seg.text for seg in segments)
            if text:
                yield RecognitionResult(
                    text=text,
                    is_final=True,
                    confidence=info.language_probability,
                )
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

### 4.3 ElevenLabs ストリーミング TTS

```python
# コード例: ElevenLabs WebSocket ストリーミング TTS
import websockets
import json
import asyncio

class ElevenLabsStreamingTTS:
    """ElevenLabs の WebSocket API を使ったストリーミング TTS"""

    def __init__(self, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = "eleven_multilingual_v2"
        self.ws_url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/"
            f"{voice_id}/stream-input?model_id={self.model_id}"
        )

    async def stream_text_to_speech(
        self,
        text_iterator,
        output_format: str = "pcm_16000",
    ):
        """
        テキストイテレータから音声チャンクをストリーミング生成。
        LLM のストリーミング出力をそのまま渡せる。
        """
        async with websockets.connect(self.ws_url) as ws:
            # 初期設定メッセージ
            await ws.send(json.dumps({
                "text": " ",  # 初期バッファ
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
                "xi_api_key": self.api_key,
                "output_format": output_format,
                "flush": False,
            }))

            # テキストを送信するタスク
            async def send_text():
                async for text_chunk in text_iterator:
                    await ws.send(json.dumps({
                        "text": text_chunk,
                        "flush": False,
                    }))

                # 終了シグナル
                await ws.send(json.dumps({
                    "text": "",
                    "flush": True,
                }))

            # 音声を受信するタスク
            async def receive_audio():
                while True:
                    try:
                        response = await ws.recv()
                        data = json.loads(response)

                        if "audio" in data and data["audio"]:
                            import base64
                            audio_bytes = base64.b64decode(data["audio"])
                            yield audio_bytes

                        if data.get("isFinal"):
                            break

                    except websockets.exceptions.ConnectionClosed:
                        break

            # 送信と受信を並列実行
            send_task = asyncio.create_task(send_text())
            async for audio_chunk in receive_audio():
                yield audio_chunk

            await send_task


# 使用例: LLM + ElevenLabs ストリーミング
async def llm_to_speech():
    """LLM の出力をリアルタイムで音声に変換"""
    import openai

    client = openai.AsyncOpenAI()
    tts = ElevenLabsStreamingTTS(api_key="your-api-key")

    # LLM ストリーミング
    async def llm_stream():
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "日本の四季について教えて"}],
            stream=True,
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # LLM → TTS ストリーミングパイプライン
    async for audio_chunk in tts.stream_text_to_speech(llm_stream()):
        # 音声チャンクを再生またはクライアントに送信
        await play_audio(audio_chunk)
```

### 4.4 TTS プリフェッチとバッファリング

```python
# コード例: TTS 音声のプリフェッチ・バッファリング戦略
import asyncio
from collections import deque
import time

class TTSAudioBuffer:
    """
    TTS 音声チャンクのバッファリングと再生制御。
    途切れのないスムーズな再生を実現する。
    """

    def __init__(
        self,
        min_buffer_ms: int = 200,      # 最小バッファ（再生開始閾値）
        max_buffer_ms: int = 2000,     # 最大バッファ
        sample_rate: int = 16000,
        bytes_per_sample: int = 2,     # 16bit PCM
    ):
        self.min_buffer_ms = min_buffer_ms
        self.max_buffer_ms = max_buffer_ms
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.bytes_per_ms = sample_rate * bytes_per_sample / 1000

        self.buffer = deque()
        self.total_buffered_bytes = 0
        self.is_playing = False
        self.playback_started = False

        # メトリクス
        self.metrics = {
            "underruns": 0,          # バッファ枯渇回数
            "ttfb_ms": 0,            # 最初の音声チャンクまでの遅延
            "total_chunks": 0,
            "start_time": None,
        }

    async def add_chunk(self, audio_bytes: bytes):
        """音声チャンクをバッファに追加"""
        if self.metrics["start_time"] is None:
            self.metrics["start_time"] = time.monotonic()

        self.buffer.append(audio_bytes)
        self.total_buffered_bytes += len(audio_bytes)
        self.metrics["total_chunks"] += 1

        # TTFB 記録
        if not self.playback_started and self.metrics["ttfb_ms"] == 0:
            self.metrics["ttfb_ms"] = (
                time.monotonic() - self.metrics["start_time"]
            ) * 1000

        # 最小バッファに達したら再生開始
        buffered_ms = self.total_buffered_bytes / self.bytes_per_ms
        if not self.playback_started and buffered_ms >= self.min_buffer_ms:
            self.playback_started = True

    async def get_chunk(self) -> bytes | None:
        """再生用の音声チャンクを取得"""
        if not self.playback_started:
            return None

        if len(self.buffer) == 0:
            self.metrics["underruns"] += 1
            return None

        chunk = self.buffer.popleft()
        self.total_buffered_bytes -= len(chunk)
        return chunk

    @property
    def buffered_ms(self) -> float:
        """現在のバッファ量（ミリ秒）"""
        return self.total_buffered_bytes / self.bytes_per_ms

    def get_metrics(self) -> dict:
        """バッファメトリクスを返す"""
        return {
            **self.metrics,
            "buffered_ms": self.buffered_ms,
            "buffer_chunks": len(self.buffer),
        }
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

### 5.3 Opus コーデック詳細設定

```python
# コード例: Opus コーデック設定の最適化
import opuslib

class OpusConfig:
    """Opus コーデックの用途別最適設定"""

    PRESETS = {
        "voip": {
            "application": opuslib.APPLICATION_VOIP,
            "bitrate": 24000,         # 24 kbps
            "frame_size_ms": 20,      # 20ms フレーム
            "bandwidth": "narrowband", # 8kHz
            "fec": True,              # 前方誤り訂正
            "dtx": True,              # 無音時送信停止
            "packet_loss": 10,        # 10% パケットロス想定
            "description": "音声通話向け。低ビットレートで高品質",
        },
        "realtime_stt": {
            "application": opuslib.APPLICATION_VOIP,
            "bitrate": 32000,         # 32 kbps
            "frame_size_ms": 20,
            "bandwidth": "wideband",   # 16kHz（STT推奨）
            "fec": False,
            "dtx": False,             # STT用途では無音も重要
            "packet_loss": 0,
            "description": "リアルタイムSTT向け。音声品質優先",
        },
        "music_streaming": {
            "application": opuslib.APPLICATION_AUDIO,
            "bitrate": 128000,        # 128 kbps
            "frame_size_ms": 20,
            "bandwidth": "fullband",   # 48kHz
            "fec": True,
            "dtx": False,
            "packet_loss": 5,
            "description": "音楽配信向け。フルバンド高品質",
        },
    }

    @classmethod
    def create_encoder(cls, preset: str = "voip", sample_rate: int = 16000):
        """プリセットからエンコーダを作成"""
        config = cls.PRESETS[preset]
        channels = 1

        encoder = opuslib.Encoder(
            sample_rate,
            channels,
            config["application"],
        )
        encoder.bitrate = config["bitrate"]

        if config["fec"]:
            encoder.inband_fec = True
            encoder.packet_loss_perc = config["packet_loss"]

        if config["dtx"]:
            encoder.dtx = True

        return encoder, config

    @classmethod
    def create_decoder(cls, sample_rate: int = 16000):
        """デコーダを作成"""
        return opuslib.Decoder(sample_rate, 1)
```

---

## 6. VAD (Voice Activity Detection) 統合

### 6.1 Silero VAD のリアルタイム統合

```python
# コード例: Silero VAD をリアルタイムパイプラインに統合
import torch
import numpy as np
from collections import deque

class RealtimeVAD:
    """
    リアルタイム VAD (Voice Activity Detection)。
    無音区間の検出、発話開始/終了イベントの生成を行う。
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,       # 最小発話長
        min_silence_ms: int = 500,      # 発話終了判定の無音長
        speech_pad_ms: int = 100,       # 発話前後のパディング
        sample_rate: int = 16000,
        window_size_ms: int = 32,       # VAD ウィンドウサイズ
    ):
        # Silero VAD モデルをロード
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()

        self.threshold = threshold
        self.sample_rate = sample_rate
        self.window_size = int(window_size_ms * sample_rate / 1000)
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_ms * sample_rate / 1000)
        self.speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)

        # 状態管理
        self.is_speaking = False
        self.speech_start = 0
        self.speech_end = 0
        self.silence_count = 0
        self.speech_count = 0
        self.current_sample = 0

        # 音声バッファ（発話前パディング用）
        self.pre_buffer = deque(
            maxlen=self.speech_pad_samples // self.window_size + 1
        )

    def process(self, audio_chunk: np.ndarray) -> list[dict]:
        """
        音声チャンクを処理し、VAD イベントを返す。

        Returns:
            list of events: [{"type": "speech_start"/"speech_end", "sample": int}]
        """
        events = []
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # ウィンドウ単位で処理
        for i in range(0, len(audio_tensor), self.window_size):
            window = audio_tensor[i:i + self.window_size]
            if len(window) < self.window_size:
                # パディング
                window = torch.nn.functional.pad(
                    window, (0, self.window_size - len(window))
                )

            # VAD 推論
            speech_prob = self.model(window, self.sample_rate).item()

            if speech_prob >= self.threshold:
                self.speech_count += self.window_size
                self.silence_count = 0

                if not self.is_speaking:
                    if self.speech_count >= self.min_speech_samples:
                        self.is_speaking = True
                        self.speech_start = self.current_sample - self.speech_count
                        events.append({
                            "type": "speech_start",
                            "sample": self.speech_start,
                            "time_ms": self.speech_start / self.sample_rate * 1000,
                        })
            else:
                self.silence_count += self.window_size

                if self.is_speaking:
                    if self.silence_count >= self.min_silence_samples:
                        self.is_speaking = False
                        self.speech_end = self.current_sample
                        duration_ms = (
                            (self.speech_end - self.speech_start)
                            / self.sample_rate * 1000
                        )
                        events.append({
                            "type": "speech_end",
                            "sample": self.speech_end,
                            "time_ms": self.speech_end / self.sample_rate * 1000,
                            "duration_ms": duration_ms,
                        })
                        self.speech_count = 0

                if not self.is_speaking:
                    self.speech_count = 0

            self.current_sample += self.window_size

            # プリバッファに保存
            self.pre_buffer.append(window.numpy())

        return events

    def reset(self):
        """状態をリセット"""
        self.model.reset_states()
        self.is_speaking = False
        self.speech_count = 0
        self.silence_count = 0
        self.current_sample = 0
        self.pre_buffer.clear()
```

### 6.2 VAD 統合型ストリーミングパイプライン

```python
# コード例: VAD + STT + TTS 統合パイプライン
import asyncio
import numpy as np

class VADIntegratedPipeline:
    """VAD を使ったインテリジェントなストリーミングパイプライン"""

    def __init__(self, vad, stt_engine, tts_engine, ai_engine):
        self.vad = vad
        self.stt = stt_engine
        self.tts = tts_engine
        self.ai = ai_engine

        # 発話セグメントバッファ
        self.speech_buffer = np.array([], dtype=np.float32)
        self.is_collecting = False

    async def process_audio_stream(self, audio_stream):
        """
        音声ストリームを処理し、AI応答を音声で返す。
        VAD により無音区間をスキップし、効率的に処理する。
        """
        async for audio_chunk in audio_stream:
            # VAD で音声区間を検出
            events = self.vad.process(audio_chunk)

            for event in events:
                if event["type"] == "speech_start":
                    self.is_collecting = True
                    self.speech_buffer = np.array([], dtype=np.float32)

                elif event["type"] == "speech_end":
                    self.is_collecting = False

                    if len(self.speech_buffer) > 0:
                        # 発話セグメントを STT で認識
                        result = await self.stt.transcribe(self.speech_buffer)

                        if result and result["text"]:
                            # AI 応答生成
                            response_text = await self.ai.generate(result["text"])

                            # TTS でストリーミング音声合成
                            async for tts_chunk in self.tts.synthesize_stream(
                                response_text
                            ):
                                yield {
                                    "type": "audio",
                                    "data": tts_chunk,
                                }

                            # メタデータも送信
                            yield {
                                "type": "metadata",
                                "user_text": result["text"],
                                "ai_text": response_text,
                                "speech_duration_ms": event["duration_ms"],
                            }

            # 発話中はバッファに蓄積
            if self.is_collecting:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                self.speech_buffer = np.concatenate(
                    [self.speech_buffer, audio_float]
                )
```

---

## 7. メディアサーバー設計

### 7.1 mediasoup ベースの SFU サーバー

```javascript
// コード例: mediasoup (Node.js) による SFU サーバー
const mediasoup = require("mediasoup");

class SFUServer {
  constructor() {
    this.workers = [];
    this.routers = new Map(); // roomId -> Router
    this.transports = new Map(); // peerId -> Transport
    this.producers = new Map(); // peerId -> Producer
    this.consumers = new Map(); // peerId -> [Consumer]
  }

  async init(numWorkers = 4) {
    // Worker プロセスを起動
    for (let i = 0; i < numWorkers; i++) {
      const worker = await mediasoup.createWorker({
        logLevel: "warn",
        rtcMinPort: 10000 + i * 1000,
        rtcMaxPort: 10999 + i * 1000,
      });

      worker.on("died", () => {
        console.error(`Worker ${i} died, restarting...`);
        this._restartWorker(i);
      });

      this.workers.push(worker);
    }
    console.log(`${numWorkers} mediasoup workers started`);
  }

  async createRoom(roomId) {
    // ラウンドロビンで Worker を選択
    const worker = this.workers[this.routers.size % this.workers.length];

    const router = await worker.createRouter({
      mediaCodecs: [
        {
          kind: "audio",
          mimeType: "audio/opus",
          clockRate: 48000,
          channels: 2,
          parameters: {
            "sprop-stereo": 1,
            usedtx: 1,
          },
        },
      ],
    });

    this.routers.set(roomId, router);
    return router;
  }

  async createWebRtcTransport(roomId, peerId, direction) {
    const router = this.routers.get(roomId);

    const transport = await router.createWebRtcTransport({
      listenIps: [
        { ip: "0.0.0.0", announcedIp: process.env.PUBLIC_IP },
      ],
      enableUdp: true,
      enableTcp: true,
      preferUdp: true,
      initialAvailableOutgoingBitrate: 128000,
    });

    transport.on("dtlsstatechange", (state) => {
      if (state === "closed") {
        transport.close();
      }
    });

    this.transports.set(`${peerId}-${direction}`, transport);

    return {
      id: transport.id,
      iceParameters: transport.iceParameters,
      iceCandidates: transport.iceCandidates,
      dtlsParameters: transport.dtlsParameters,
    };
  }

  async produce(peerId, transportId, kind, rtpParameters) {
    const transport = this.transports.get(`${peerId}-send`);
    const producer = await transport.produce({ kind, rtpParameters });

    this.producers.set(peerId, producer);

    // 他の参加者に通知
    producer.on("transportclose", () => {
      producer.close();
      this.producers.delete(peerId);
    });

    return producer.id;
  }

  async consume(roomId, consumerPeerId, producerPeerId) {
    const router = this.routers.get(roomId);
    const producer = this.producers.get(producerPeerId);
    const transport = this.transports.get(`${consumerPeerId}-recv`);

    if (!router.canConsume({ producerId: producer.id, rtpCapabilities: {} })) {
      throw new Error("Cannot consume");
    }

    const consumer = await transport.consume({
      producerId: producer.id,
      rtpCapabilities: router.rtpCapabilities,
      paused: false,
    });

    return {
      id: consumer.id,
      producerId: producer.id,
      kind: consumer.kind,
      rtpParameters: consumer.rtpParameters,
    };
  }
}
```

### 7.2 Kubernetes でのスケーリング設計

```yaml
# コード例: Kubernetes マニフェスト — STT/TTS ワーカーのオートスケーリング

# STT Worker Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stt-worker
  labels:
    app: realtime-audio
    component: stt
spec:
  replicas: 2
  selector:
    matchLabels:
      app: stt-worker
  template:
    metadata:
      labels:
        app: stt-worker
    spec:
      containers:
        - name: stt-worker
          image: your-registry/stt-worker:latest
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
              nvidia.com/gpu: "1"    # GPU 必須
            limits:
              cpu: "4"
              memory: "8Gi"
              nvidia.com/gpu: "1"
          env:
            - name: WHISPER_MODEL
              value: "large-v3"
            - name: COMPUTE_TYPE
              value: "float16"
            - name: MAX_CONCURRENT_STREAMS
              value: "8"
          ports:
            - containerPort: 50051   # gRPC
          livenessProbe:
            grpc:
              port: 50051
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            grpc:
              port: 50051
            initialDelaySeconds: 15

---
# HPA (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: stt-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: stt-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: active_streams
        target:
          type: AverageValue
          averageValue: "6"    # 1Pod あたり 6 ストリーム目標

---
# TTS Worker Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tts-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: tts-worker
  template:
    spec:
      containers:
        - name: tts-worker
          image: your-registry/tts-worker:latest
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          env:
            - name: TTS_ENGINE
              value: "edge-tts"
            - name: CACHE_SIZE_MB
              value: "512"
```

---

## 8. 品質監視とデバッグ

### 8.1 音声品質メトリクス

```python
# コード例: リアルタイム音声品質モニタリング
import time
import numpy as np
from dataclasses import dataclass, field

@dataclass
class AudioQualityMetrics:
    """リアルタイム音声品質メトリクス"""

    # 遅延関連
    e2e_latency_ms: list = field(default_factory=list)
    stt_latency_ms: list = field(default_factory=list)
    tts_ttfb_ms: list = field(default_factory=list)

    # 音声品質
    snr_db: list = field(default_factory=list)
    packet_loss_rate: list = field(default_factory=list)
    jitter_ms: list = field(default_factory=list)

    # STT 品質
    stt_confidence: list = field(default_factory=list)
    partial_to_final_changes: int = 0

    # バッファ状態
    buffer_underruns: int = 0
    buffer_overruns: int = 0

class QualityMonitor:
    """リアルタイム品質監視"""

    def __init__(self, report_interval_sec: int = 10):
        self.metrics = AudioQualityMetrics()
        self.report_interval = report_interval_sec
        self.last_report_time = time.monotonic()
        self.alerts = []

    def record_latency(self, stage: str, latency_ms: float):
        """遅延を記録"""
        if stage == "e2e":
            self.metrics.e2e_latency_ms.append(latency_ms)
        elif stage == "stt":
            self.metrics.stt_latency_ms.append(latency_ms)
        elif stage == "tts_ttfb":
            self.metrics.tts_ttfb_ms.append(latency_ms)

        # アラート判定
        if stage == "e2e" and latency_ms > 1000:
            self.alerts.append({
                "level": "warning",
                "message": f"E2E latency {latency_ms:.0f}ms exceeds 1000ms",
                "timestamp": time.time(),
            })

    def record_audio_quality(self, audio_chunk: np.ndarray, noise_estimate: float = 0):
        """音声品質を記録"""
        # SNR 計算
        signal_power = np.mean(audio_chunk ** 2)
        if noise_estimate > 0:
            snr = 10 * np.log10(signal_power / noise_estimate)
            self.metrics.snr_db.append(snr)

    def record_packet_loss(self, expected: int, received: int):
        """パケットロス率を記録"""
        loss_rate = 1 - (received / expected) if expected > 0 else 0
        self.metrics.packet_loss_rate.append(loss_rate)

        if loss_rate > 0.05:
            self.alerts.append({
                "level": "warning",
                "message": f"Packet loss {loss_rate:.1%} exceeds 5%",
                "timestamp": time.time(),
            })

    def get_report(self) -> dict:
        """品質レポートを生成"""
        def safe_stats(data):
            if not data:
                return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
            arr = np.array(data)
            return {
                "avg": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(np.max(arr)),
            }

        return {
            "latency": {
                "e2e": safe_stats(self.metrics.e2e_latency_ms),
                "stt": safe_stats(self.metrics.stt_latency_ms),
                "tts_ttfb": safe_stats(self.metrics.tts_ttfb_ms),
            },
            "audio_quality": {
                "snr_db": safe_stats(self.metrics.snr_db),
                "packet_loss": safe_stats(self.metrics.packet_loss_rate),
            },
            "buffer": {
                "underruns": self.metrics.buffer_underruns,
                "overruns": self.metrics.buffer_overruns,
            },
            "alerts": self.alerts[-10:],  # 最新10件
        }
```

### 8.2 WebRTC 統計情報の収集

```javascript
// コード例: WebRTC getStats() による詳細統計収集
class WebRTCStatsCollector {
  constructor(pc, intervalMs = 5000) {
    this.pc = pc;
    this.intervalMs = intervalMs;
    this.history = [];
    this.previousStats = null;
    this.timer = null;
  }

  start() {
    this.timer = setInterval(() => this.collect(), this.intervalMs);
  }

  stop() {
    clearInterval(this.timer);
  }

  async collect() {
    const stats = await this.pc.getStats();
    const report = {
      timestamp: Date.now(),
      inbound: {},
      outbound: {},
      connection: {},
    };

    stats.forEach((stat) => {
      // 受信音声トラック
      if (stat.type === "inbound-rtp" && stat.kind === "audio") {
        report.inbound = {
          packetsReceived: stat.packetsReceived,
          packetsLost: stat.packetsLost,
          jitter: stat.jitter,
          bytesReceived: stat.bytesReceived,
          // 前回との差分でビットレート計算
          bitrate: this._calcBitrate(stat, "inbound"),
          // パケットロス率
          lossRate:
            stat.packetsLost /
            (stat.packetsReceived + stat.packetsLost || 1),
        };
      }

      // 送信音声トラック
      if (stat.type === "outbound-rtp" && stat.kind === "audio") {
        report.outbound = {
          packetsSent: stat.packetsSent,
          bytesSent: stat.bytesSent,
          bitrate: this._calcBitrate(stat, "outbound"),
          retransmittedPacketsSent: stat.retransmittedPacketsSent || 0,
        };
      }

      // 接続情報
      if (stat.type === "candidate-pair" && stat.nominated) {
        report.connection = {
          rtt: stat.currentRoundTripTime * 1000, // ms
          availableOutgoingBitrate: stat.availableOutgoingBitrate,
          localCandidateType: stat.localCandidateType,
          remoteCandidateType: stat.remoteCandidateType,
          protocol: stat.protocol,
        };
      }
    });

    this.history.push(report);

    // アラート判定
    if (report.inbound.lossRate > 0.05) {
      console.warn(
        `High packet loss: ${(report.inbound.lossRate * 100).toFixed(1)}%`
      );
    }
    if (report.connection.rtt > 200) {
      console.warn(`High RTT: ${report.connection.rtt.toFixed(0)}ms`);
    }

    return report;
  }

  _calcBitrate(stat, direction) {
    if (!this.previousStats) {
      this.previousStats = {};
      return 0;
    }
    const key = `${direction}_bytes`;
    const prevBytes = this.previousStats[key] || 0;
    const currentBytes =
      direction === "inbound" ? stat.bytesReceived : stat.bytesSent;

    const bitrate =
      ((currentBytes - prevBytes) * 8) / (this.intervalMs / 1000);
    this.previousStats[key] = currentBytes;
    return bitrate;
  }

  getSummary() {
    if (this.history.length === 0) return null;

    const rtts = this.history
      .filter((r) => r.connection.rtt)
      .map((r) => r.connection.rtt);
    const lossRates = this.history
      .filter((r) => r.inbound.lossRate !== undefined)
      .map((r) => r.inbound.lossRate);

    return {
      avgRtt: rtts.reduce((a, b) => a + b, 0) / rtts.length || 0,
      maxRtt: Math.max(...rtts, 0),
      avgLossRate:
        lossRates.reduce((a, b) => a + b, 0) / lossRates.length || 0,
      totalSamples: this.history.length,
    };
  }
}
```

---

## 9. 接続復旧とフォールバック

### 9.1 自動再接続ロジック

```python
# コード例: 指数バックオフ付き自動再接続
import asyncio
import random
import logging

logger = logging.getLogger(__name__)

class ReconnectManager:
    """WebSocket / WebRTC の自動再接続管理"""

    def __init__(
        self,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: float = 0.5,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_count = 0
        self.is_connected = False

    async def connect_with_retry(self, connect_fn, on_connected=None):
        """
        指数バックオフ + ジッタで再接続を試行。

        Args:
            connect_fn: 接続関数（async callable）
            on_connected: 接続成功時のコールバック
        """
        while self.retry_count < self.max_retries:
            try:
                connection = await connect_fn()
                self.is_connected = True
                self.retry_count = 0
                logger.info("Connected successfully")

                if on_connected:
                    await on_connected(connection)

                return connection

            except Exception as e:
                self.retry_count += 1
                delay = self._calc_delay()

                logger.warning(
                    f"Connection failed (attempt {self.retry_count}/"
                    f"{self.max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )

                if self.retry_count >= self.max_retries:
                    logger.error("Max retries exceeded, giving up")
                    raise

                await asyncio.sleep(delay)

    def _calc_delay(self) -> float:
        """指数バックオフ + ジッタ"""
        delay = min(
            self.base_delay * (2 ** (self.retry_count - 1)),
            self.max_delay
        )
        # ジッタを追加（±jitter の範囲）
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return max(0.1, delay)

    def reset(self):
        """リトライカウンタをリセット"""
        self.retry_count = 0
        self.is_connected = False
```

### 9.2 プロトコルフォールバック

```python
# コード例: WebRTC → WebSocket → HTTP ポーリングのフォールバック
class ProtocolFallback:
    """
    通信プロトコルのフォールバック戦略。
    WebRTC (最低遅延) → WebSocket (低遅延) → HTTP SSE (中遅延)
    """

    PROTOCOLS = [
        {
            "name": "webrtc",
            "latency": "lowest",
            "description": "P2P/SFU 経由のリアルタイム音声",
        },
        {
            "name": "websocket",
            "latency": "low",
            "description": "WebSocket バイナリフレームで音声送受信",
        },
        {
            "name": "http_sse",
            "latency": "medium",
            "description": "HTTP POST で音声送信、SSE で結果受信",
        },
    ]

    def __init__(self):
        self.current_protocol = None
        self.available_protocols = list(self.PROTOCOLS)
        self.fallback_history = []

    async def connect(self, server_url: str):
        """最適なプロトコルで接続を試行"""
        for protocol in self.available_protocols:
            try:
                connection = await self._try_connect(
                    protocol["name"], server_url
                )
                self.current_protocol = protocol
                return connection
            except Exception as e:
                self.fallback_history.append({
                    "protocol": protocol["name"],
                    "error": str(e),
                    "timestamp": time.time(),
                })
                continue

        raise ConnectionError("All protocols failed")

    async def _try_connect(self, protocol: str, url: str):
        """プロトコル別の接続処理"""
        if protocol == "webrtc":
            return await self._connect_webrtc(url)
        elif protocol == "websocket":
            return await self._connect_websocket(url)
        elif protocol == "http_sse":
            return await self._connect_http_sse(url)

    async def _connect_webrtc(self, url):
        """WebRTC 接続"""
        from aiortc import RTCPeerConnection, RTCSessionDescription
        import aiohttp

        pc = RTCPeerConnection()
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}/webrtc/offer",
                json={"sdp": offer.sdp, "type": offer.type},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                answer = await resp.json()

        await pc.setRemoteDescription(
            RTCSessionDescription(**answer)
        )
        return pc

    async def _connect_websocket(self, url):
        """WebSocket 接続"""
        import websockets
        ws_url = url.replace("http", "ws") + "/ws/audio"
        ws = await websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        )
        return ws

    async def _connect_http_sse(self, url):
        """HTTP SSE 接続"""
        import aiohttp
        session = aiohttp.ClientSession()
        # SSE エンドポイントに接続
        sse_response = await session.get(
            f"{url}/sse/audio",
            timeout=aiohttp.ClientTimeout(total=None),
        )
        return {"session": session, "response": sse_response}
```

---

## 10. アンチパターン

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

### アンチパターン 3: 「再接続ロジックの欠如」

```
[誤り] WebSocket/WebRTC の切断を想定しない実装

  ws = new WebSocket(url);
  ws.onopen = () => { /* 接続完了 */ };
  // onclose/onerror で何もしない → 一度切れたら永久に切断

問題点:
- モバイル回線では頻繁に切断が発生する
- ネットワーク切替（WiFi→4G）で確実に切断される
- ユーザーが手動リロードする必要がある

[正解] 指数バックオフ付きの自動再接続を実装する
  1. onclose/onerror でリトライスケジューラを起動
  2. 指数バックオフ（1s, 2s, 4s, 8s...）で再接続
  3. ジッタを追加してサーバー負荷を分散
  4. 最大リトライ回数の設定
  5. 再接続後のセッション復旧（STT コンテキスト等）
```

### アンチパターン 4: 「AudioContext の不適切な管理」

```javascript
// BAD: ユーザー操作なしで AudioContext を作成
const ctx = new AudioContext(); // autoplay policy で suspend される

// BAD: 毎回新しい AudioContext を作成
function playAudio(data) {
  const ctx = new AudioContext(); // リソースリーク
  // ...
}

// GOOD: ユーザー操作で resume し、シングルトンで管理
class AudioManager {
  constructor() {
    this.ctx = null;
  }

  async init() {
    this.ctx = new AudioContext({ sampleRate: 16000 });
    if (this.ctx.state === "suspended") {
      // ユーザー操作（ボタンクリック等）で resume
      await this.ctx.resume();
    }
  }

  getContext() {
    return this.ctx;
  }
}
```

---

## 11. FAQ

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

### Q4: Whisper をリアルタイムで使えますか？

**A:** Whisper はバッチ処理モデルのため、そのままではリアルタイムには不向きですが、以下の工夫で擬似リアルタイム化が可能です。

1. **スライディングウィンドウ方式**: 2-3秒のチャンクをオーバーラップ付きで連続処理。faster-whisper を使えば GPU 上で 0.5 秒以下で処理可能
2. **VAD 前処理**: 無音区間をスキップし、発話区間のみ Whisper に送信することで処理量を削減
3. **投機的中間結果**: バッファが溜まるたびに認識を実行し、前回結果との差分で中間/確定を判定
4. **whisper-streaming**: OSS ライブラリ `whisper_streaming` が上記戦略を実装済み

### Q5: モバイルアプリでリアルタイム音声を実装するコツは？

**A:** モバイル特有の課題と対策を整理します。

1. **バッテリー消費**: VAD で無音時の処理を最小化。バックグラウンド時は WebSocket を維持しつつ音声処理は停止
2. **ネットワーク不安定**: 指数バックオフ再接続、Opus FEC（前方誤り訂正）の有効化、バッファリング戦略の調整
3. **音声入出力**: iOS では AVAudioSession の適切なカテゴリ設定（`.playAndRecord`）が必須。Android では AudioRecord + AudioTrack を使用
4. **エコーキャンセレーション**: スピーカーとマイクが近いため、AEC（Acoustic Echo Cancellation）が重要。WebRTC のエコーキャンセラーを活用
5. **省電力モード**: バックグラウンド遷移時の処理、画面ロック時の音声継続について OS の制約を理解する

### Q6: 音声チャットボットの応答遅延を 1 秒以内にするには？

**A:** エンドツーエンド遅延 1 秒以内を達成するための具体的な構成です。

1. **STT**: faster-whisper (GPU) で 200ms 以内。VAD で発話終了を素早く検出
2. **LLM**: GPT-4o-mini や Claude 3.5 Haiku などの高速モデルをストリーミングで使用。最初のトークン到着まで 100-200ms
3. **TTS**: Edge TTS や ElevenLabs Turbo v2 で TTFB 100-200ms。最初の文の音声を即座に再生開始
4. **パイプライン**: 各段階をストリーミングで連結。STT 確定 → LLM 開始 → 最初の句読点で TTS 開始 → 即再生
5. **プリウォーム**: モデルのロード、WebSocket 接続、TTS エンジンの初期化を事前に完了させておく

---

## 12. まとめ

| コンポーネント | 技術 | 推奨ツール | 遅延目標 |
|--------------|------|-----------|---------|
| 音声通信 | WebRTC | aiortc, mediasoup | < 100ms |
| シグナリング | WebSocket | FastAPI, Socket.IO | < 50ms |
| ストリーミング STT | gRPC Streaming | Google STT, Whisper Streaming | < 200ms |
| ストリーミング TTS | チャンク合成 | Edge TTS, ElevenLabs | < 300ms |
| VAD | RNN/ルールベース | Silero VAD, WebRTC VAD | < 10ms |
| コーデック | Opus | WebRTC 内蔵 | 5ms |
| メディアサーバー | SFU | mediasoup, Janus | < 50ms |
| 品質監視 | getStats / カスタム | Prometheus + Grafana | リアルタイム |

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
4. mediasoup Documentation. (2024). "mediasoup — Cutting Edge WebRTC Video Conferencing." https://mediasoup.org/documentation/
5. Silero Team. (2021). "Silero VAD: pre-trained enterprise-grade Voice Activity Detector." https://github.com/snakers4/silero-vad
6. ElevenLabs. (2024). "WebSocket Streaming API Documentation." https://docs.elevenlabs.io/api-reference/websockets
7. Microsoft. (2024). "Azure Speech SDK Streaming Recognition." https://learn.microsoft.com/azure/cognitive-services/speech-service/how-to-recognize-speech
