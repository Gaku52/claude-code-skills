# WebSocket

> WebSocketはHTTP上で確立される双方向リアルタイム通信プロトコル。チャット、リアルタイム通知、ゲーム、金融データ配信など、サーバーからのプッシュが必要なアプリケーションの基盤。

## この章で学ぶこと

- [ ] WebSocketのハンドシェイクと通信の仕組みを理解する
- [ ] HTTPとの違いとWebSocketが解決する課題を把握する
- [ ] 実装パターンとベストプラクティスを学ぶ

---

## 1. なぜWebSocketが必要か

```
HTTPの限界:
  HTTP = リクエスト/レスポンス型（クライアント起点）
  → サーバーから自発的にデータを送れない

従来の回避策:
  ① ポーリング（Polling）:
     → クライアントが定期的にリクエスト
     → 無駄なリクエストが多い
     → リアルタイム性が低い（インターバル依存）

  ② ロングポーリング（Long Polling）:
     → リクエストを保持し、データがあれば即応答
     → サーバーリソースを消費
     → 接続の再確立コスト

  ③ Server-Sent Events（SSE）:
     → サーバー→クライアントの一方向ストリーム
     → テキストデータのみ
     → 双方向通信は不可

  WebSocket:
     → 真の双方向通信
     → 低レイテンシ（常時接続）
     → バイナリデータ対応
     → ヘッダーオーバーヘッドが小さい（2-14バイト）

  比較:
  ┌───────────┬──────────┬──────────┬──────────┬──────────┐
  │           │ Polling  │ Long Poll│ SSE      │WebSocket │
  ├───────────┼──────────┼──────────┼──────────┼──────────┤
  │ 方向      │ 単方向   │ 単方向   │ 単方向   │ 双方向   │
  │ レイテンシ│ 高       │ 中       │ 低       │ 最低     │
  │ サーバー  │ 低       │ 中       │ 低       │ 中       │
  │ 負荷      │          │          │          │          │
  │ 複雑さ    │ 低       │ 中       │ 低       │ 高       │
  └───────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 2. WebSocketハンドシェイク

```
WebSocket接続はHTTPアップグレードで確立:

クライアント → サーバー（HTTPリクエスト）:
  GET /chat HTTP/1.1
  Host: example.com
  Upgrade: websocket
  Connection: Upgrade
  Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
  Sec-WebSocket-Version: 13

サーバー → クライアント（HTTPレスポンス）:
  HTTP/1.1 101 Switching Protocols
  Upgrade: websocket
  Connection: Upgrade
  Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

  101 Switching Protocols 以降:
  → HTTPからWebSocketプロトコルに切り替わる
  → 同じTCP接続を再利用
  → 以降はWebSocketフレームで通信

Sec-WebSocket-Accept の計算:
  1. Sec-WebSocket-Key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
  2. SHA-1ハッシュ
  3. Base64エンコード
  → クロスプロトコル攻撃の防止
```

---

## 3. WebSocketフレーム

```
WebSocketフレーム構造:

  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  ┌─┬───┬─┬───────┬─────────────────────────────────────────────┐
  │F│RSV│O│ペイロ│  拡張ペイロード長（0, 16, 64ビット）         │
  │I│1-3│P│ード長 │                                              │
  │N│   │C│(7bit)│                                              │
  ├─┴───┴─┴───────┼─────────────────────────────────────────────┤
  │ マスクキー（32ビット、クライアント→サーバーのみ）            │
  ├───────────────────────────────────────────────────────────────┤
  │ ペイロードデータ                                              │
  └───────────────────────────────────────────────────────────────┘

  OPコード:
  0x0: 継続フレーム
  0x1: テキストフレーム（UTF-8）
  0x2: バイナリフレーム
  0x8: 接続クローズ
  0x9: Ping
  0xA: Pong

  マスキング:
  → クライアント→サーバーは必ずマスク
  → サーバー→クライアントはマスクなし
  → プロキシキャッシュ汚染攻撃の防止
```

---

## 4. サーバー実装（Node.js）

```typescript
// ws ライブラリを使用
import { WebSocketServer, WebSocket } from 'ws';

const wss = new WebSocketServer({ port: 8080 });

// 接続管理
const clients = new Map<string, WebSocket>();

wss.on('connection', (ws, req) => {
  const clientId = crypto.randomUUID();
  clients.set(clientId, ws);
  console.log(`Client connected: ${clientId}`);

  // メッセージ受信
  ws.on('message', (data) => {
    const message = JSON.parse(data.toString());

    switch (message.type) {
      case 'chat':
        // 全クライアントにブロードキャスト
        broadcast({ type: 'chat', from: clientId, text: message.text });
        break;
      case 'dm':
        // 特定クライアントに送信
        const target = clients.get(message.to);
        target?.send(JSON.stringify({ type: 'dm', from: clientId, text: message.text }));
        break;
    }
  });

  // Ping/Pong でヘルスチェック
  const interval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.ping();
    }
  }, 30000);

  ws.on('pong', () => {
    // クライアントは生きている
  });

  ws.on('close', () => {
    clients.delete(clientId);
    clearInterval(interval);
    console.log(`Client disconnected: ${clientId}`);
  });

  ws.on('error', (error) => {
    console.error(`WebSocket error: ${error.message}`);
  });
});

function broadcast(message: object): void {
  const data = JSON.stringify(message);
  clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(data);
    }
  });
}
```

---

## 5. クライアント実装

```typescript
// ブラウザ WebSocket API
class ChatClient {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  connect(url: string): void {
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('Connected');
    };

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };

    this.ws.onclose = (event) => {
      if (!event.wasClean) {
        // 意図しない切断 → 再接続
        this.scheduleReconnect(url);
      }
    };

    this.ws.onerror = () => {
      // onerror の後に onclose が呼ばれる
    };
  }

  send(type: string, data: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, ...data }));
    }
  }

  private scheduleReconnect(url: string): void {
    this.reconnectTimer = setTimeout(() => this.connect(url), 3000);
  }

  disconnect(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close(1000, 'Normal closure');
  }

  private handleMessage(message: any): void {
    // アプリケーション固有のメッセージ処理
  }
}
```

---

## 6. スケーリングとベストプラクティス

```
スケーリングの課題:
  WebSocket = ステートフル（接続を保持）
  → 通常のHTTPロードバランシングが使えない

解決策:
  ① Sticky Sessions:
     → 同じクライアントを同じサーバーに送る
     → ロードバランサーでCookieまたはIPハッシュ

  ② Pub/Sub バックエンド:
     → Redis Pub/Sub でサーバー間メッセージ連携
     → サーバーA のクライアント → Redis → サーバーB のクライアント

  ③ 専用WebSocketサービス:
     → Socket.IO, Pusher, Ably, AWS API Gateway WebSocket

ベストプラクティス:
  1. ハートビート: Ping/Pongで接続の生存確認（30秒間隔）
  2. 再接続: 指数バックオフで自動再接続
  3. メッセージ形式: JSON with type フィールド
  4. 認証: ハンドシェイク時にトークン検証
  5. レート制限: クライアントごとのメッセージ数制限
  6. 圧縮: permessage-deflate 拡張
  7. バイナリ: 大量データにはバイナリフレームを使用
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| WebSocket | HTTP上の双方向リアルタイム通信 |
| ハンドシェイク | HTTP 101 Switching Protocols |
| フレーム | 2-14バイトのヘッダー、テキスト/バイナリ |
| スケーリング | Sticky Session + Pub/Sub |
| 代替手段 | SSE（一方向）、Polling（互換性） |

---

## 次に読むべきガイド
→ [[03-grpc.md]] — gRPC

---

## 参考文献
1. RFC 6455. "The WebSocket Protocol." IETF, 2011.
2. RFC 7692. "Compression Extensions for WebSocket." IETF, 2015.
