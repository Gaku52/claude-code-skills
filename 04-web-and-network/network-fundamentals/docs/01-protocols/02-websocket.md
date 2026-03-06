# WebSocket

> WebSocketはHTTP上で確立される双方向リアルタイム通信プロトコル。チャット、リアルタイム通知、ゲーム、金融データ配信など、サーバーからのプッシュが必要なアプリケーションの基盤。RFC 6455で標準化されたこのプロトコルは、従来のHTTPポーリングの限界を克服し、クライアントとサーバー間の真の全二重通信を実現する。

## この章で学ぶこと

- [ ] WebSocketのハンドシェイクと通信の仕組みを理解する
- [ ] HTTPとの違いとWebSocketが解決する課題を把握する
- [ ] フレーム構造とプロトコルの内部動作を理解する
- [ ] サーバーサイド・クライアントサイド両方の実装パターンを習得する
- [ ] Socket.IOなどのライブラリを用いた実践的な開発手法を学ぶ
- [ ] スケーリング、セキュリティ、パフォーマンス最適化の知識を身につける
- [ ] アンチパターンを把握し、本番環境での問題を未然に防ぐ

---

## 1. なぜWebSocketが必要か

### 1.1 HTTPの根本的な制約

HTTPはリクエスト/レスポンスモデルに基づいている。通信は常にクライアント起点であり、サーバーがクライアントに対して能動的にデータを送信する手段を持たない。この制約はWebの初期においては問題にならなかった。静的ページの配信や、フォーム送信のような単発のやりとりにはリクエスト/レスポンスモデルで十分だったためである。

しかし、Webアプリケーションが高度化するにつれ、リアルタイム性の要求が急速に高まった。チャットアプリケーション、株価ティッカー、オンラインゲーム、共同編集ツールなど、サーバーからクライアントへの即時データ配信が不可欠なユースケースが増加した。

### 1.2 従来の回避策とその限界

```
HTTPの限界を回避するための技術の変遷:

  ① ポーリング（Polling）:
     ┌─────────┐         ┌─────────┐
     │ Client  │ ──GET──→│ Server  │    クライアントが一定間隔で
     │         │ ←─200── │         │    サーバーに問い合わせる
     │         │         │         │
     │         │ ──GET──→│         │    データがなくても毎回リクエスト
     │         │ ←─204── │         │    が発生し、帯域幅を浪費する
     │         │         │         │
     │         │ ──GET──→│         │    インターバルが長いと遅延が大きく
     │         │ ←─200── │         │    短いとサーバー負荷が増大する
     └─────────┘         └─────────┘

     問題点:
     - 無駄なリクエストが大量に発生（データがない場合も含む）
     - リアルタイム性がインターバル間隔に依存
     - HTTPヘッダーのオーバーヘッドが毎回発生（約800バイト/リクエスト）
     - サーバーのCPU・メモリリソースを不必要に消費

  ② ロングポーリング（Long Polling）:
     ┌─────────┐         ┌─────────┐
     │ Client  │ ──GET──→│ Server  │    サーバーはデータが利用可能に
     │         │         │ (待機)  │    なるまでレスポンスを保留する
     │         │         │  ...    │
     │         │         │  ...    │    タイムアウトまでデータがなければ
     │         │ ←─200── │ (送信)  │    空レスポンスを返す
     │         │ ──GET──→│         │
     │         │         │ (待機)  │    即座に次のリクエストを送信し
     └─────────┘         └─────────┘    擬似的なプッシュを実現

     問題点:
     - サーバー側で接続を長時間保持するためリソース消費が大きい
     - 接続の再確立コスト（TCP/TLSハンドシェイク）が毎回発生
     - 高頻度のメッセージでは結局ポーリングと同等の負荷になる
     - HTTP/1.1ではブラウザの同時接続数制限（6接続/ドメイン）の影響を受ける

  ③ Server-Sent Events（SSE）:
     ┌─────────┐         ┌─────────┐
     │ Client  │ ──GET──→│ Server  │    HTTPコネクション上で
     │         │ ←─data──│         │    サーバーからクライアントへの
     │         │ ←─data──│         │    一方向ストリーミング
     │         │ ←─data──│         │
     └─────────┘         └─────────┘

     利点: 自動再接続、イベントID管理が組み込み
     問題点:
     - サーバー→クライアントの一方向のみ
     - テキストデータ（UTF-8）のみ対応
     - HTTP/1.1では同時接続数制限の影響を受ける
     - バイナリデータの送信には別途HTTPリクエストが必要
```

### 1.3 WebSocketが提供するソリューション

WebSocketは上記すべての問題を根本的に解決する。初回のHTTPハンドシェイク後、TCPコネクション上でプロトコルを切り替え、双方向の全二重通信チャネルを確立する。これにより以下の利点が得られる。

1. **真の双方向通信**: クライアントとサーバーが対等にメッセージを送受信できる
2. **低レイテンシ**: 常時接続のため、接続確立のオーバーヘッドがない
3. **低オーバーヘッド**: フレームヘッダーはわずか2〜14バイト（HTTPヘッダーの数百分の1）
4. **バイナリデータ対応**: テキストとバイナリの両方を効率的に転送可能
5. **プロトコルレベルのKeep-Alive**: Ping/Pongフレームによる接続状態の監視

### 1.4 リアルタイム通信技術の比較表

```
  ┌────────────────┬───────────┬───────────┬───────────┬───────────┐
  │ 特性           │ Polling   │ Long Poll │ SSE       │ WebSocket │
  ├────────────────┼───────────┼───────────┼───────────┼───────────┤
  │ 通信方向       │ 単方向    │ 単方向    │ 単方向    │ 双方向    │
  │                │ C→S       │ C→S       │ S→C       │ C↔S       │
  ├────────────────┼───────────┼───────────┼───────────┼───────────┤
  │ レイテンシ     │ 高        │ 中        │ 低        │ 最低      │
  │ (平均)         │ interval/2│ ~100ms    │ ~50ms     │ ~10ms     │
  ├────────────────┼───────────┼───────────┼───────────┼───────────┤
  │ サーバー負荷   │ 高        │ 中〜高    │ 低        │ 低〜中    │
  │ (1万接続時)    │           │           │           │           │
  ├────────────────┼───────────┼───────────┼───────────┼───────────┤
  │ 帯域効率       │ 低        │ 低〜中    │ 高        │ 最高      │
  │ (ヘッダー)     │ ~800B/req │ ~800B/req │ ~50B/msg  │ ~6B/msg   │
  ├────────────────┼───────────┼───────────┼───────────┼───────────┤
  │ バイナリ対応   │ 可能      │ 可能      │ 不可      │ 可能      │
  ├────────────────┼───────────┼───────────┼───────────┼───────────┤
  │ 自動再接続     │ 手動実装  │ 手動実装  │ 組み込み  │ 手動実装  │
  ├────────────────┼───────────┼───────────┼───────────┼───────────┤
  │ HTTP/2互換性   │ 完全      │ 完全      │ 改善      │ 限定的    │
  ├────────────────┼───────────┼───────────┼───────────┼───────────┤
  │ ファイアウォール│ 問題なし  │ 問題なし  │ 問題なし  │ 要注意    │
  │ 透過性         │           │           │           │           │
  ├────────────────┼───────────┼───────────┼───────────┼───────────┤
  │ 実装複雑度     │ 低        │ 中        │ 低        │ 高        │
  ├────────────────┼───────────┼───────────┼───────────┼───────────┤
  │ 推奨ユースケース│ 低頻度    │ 中頻度    │ 通知系    │ リアル    │
  │                │ 更新      │ 更新      │ フィード  │ タイム    │
  └────────────────┴───────────┴───────────┴───────────┴───────────┘
```

### 1.5 WebSocketが適しているユースケース

WebSocketは万能ではない。以下に適性の高いケースと低いケースを整理する。

**適性が高いケース:**
- チャットアプリケーション（1対1、グループ）
- リアルタイム共同編集（Google Docs型）
- 金融データのストリーミング（株価、為替レート）
- オンラインゲーム（マルチプレイヤー）
- IoTデバイスのリアルタイム監視ダッシュボード
- ライブスポーツのスコア更新
- リアルタイム通知システム

**適性が低いケース:**
- 単純なCRUD操作（REST APIで十分）
- 低頻度の更新（5分以上の間隔ならポーリングで十分）
- 一方向のイベント通知のみ（SSEで十分）
- SEOが重要なコンテンツ配信（HTTPが適切）
- ファイルアップロード/ダウンロード（HTTPの方が効率的）

---

## 2. WebSocketハンドシェイク

### 2.1 ハンドシェイクの全体フロー

WebSocket接続はHTTPアップグレード機構を利用して確立される。このプロセスは「オープニングハンドシェイク」と呼ばれ、クライアントがHTTP GETリクエストにWebSocketアップグレードヘッダーを含めて送信し、サーバーが101 Switching Protocolsで応答することで完了する。

```
WebSocketハンドシェイクの詳細フロー:

  クライアント                    サーバー
      │                              │
      │  ① TCP 3-way handshake       │
      │  ─────── SYN ──────────────→ │
      │  ←────── SYN+ACK ─────────── │
      │  ─────── ACK ──────────────→ │
      │                              │
      │  ② TLS handshake (wss://の場合) │
      │  ─────── ClientHello ──────→ │
      │  ←────── ServerHello ─────── │
      │  ←────── Certificate ─────── │
      │  ─────── Key Exchange ─────→ │
      │  ←────── Finished ─────────  │
      │                              │
      │  ③ HTTP Upgrade Request      │
      │  ─── GET /chat HTTP/1.1 ───→ │
      │      Upgrade: websocket      │
      │      Connection: Upgrade     │
      │      Sec-WebSocket-Key: xxx  │
      │      Sec-WebSocket-Version: 13│
      │                              │
      │  ④ HTTP 101 Response         │
      │  ←── 101 Switching ───────── │
      │      Protocols               │
      │      Upgrade: websocket      │
      │      Connection: Upgrade     │
      │      Sec-WebSocket-Accept: yyy│
      │                              │
      │  ⑤ WebSocket通信開始         │
      │  ←════ WebSocketフレーム ════→│
      │  ←════ WebSocketフレーム ════→│
      │                              │
      │  ⑥ クローズハンドシェイク    │
      │  ─── Close Frame ──────────→ │
      │  ←── Close Frame ─────────── │
      │  ─── TCP FIN ──────────────→ │
      │                              │
```

### 2.2 ハンドシェイクリクエストの詳細

```
クライアント → サーバー（HTTPリクエスト）:

  GET /chat HTTP/1.1
  Host: example.com
  Upgrade: websocket
  Connection: Upgrade
  Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
  Sec-WebSocket-Version: 13
  Sec-WebSocket-Protocol: chat, superchat
  Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits
  Origin: https://example.com
  Cookie: session=abc123

各ヘッダーの役割:
  ┌───────────────────────────┬─────────────────────────────────────────┐
  │ ヘッダー                  │ 説明                                    │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ Upgrade: websocket        │ WebSocketへのプロトコル切替を要求       │
  │                           │ （必須）                                │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ Connection: Upgrade       │ Upgradeヘッダーがホップバイホップで     │
  │                           │ あることを示す（必須）                  │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ Sec-WebSocket-Key         │ 16バイトのランダム値をBase64エンコード  │
  │                           │ したもの。サーバー検証用（必須）        │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ Sec-WebSocket-Version     │ プロトコルバージョン。現行は13（必須）  │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ Sec-WebSocket-Protocol    │ サブプロトコルの候補リスト（任意）      │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ Sec-WebSocket-Extensions  │ 使用したい拡張機能（任意）              │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ Origin                    │ ブラウザクライアントの起源（CORS的検証）│
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ Cookie                    │ 認証情報（既存セッション利用時）        │
  └───────────────────────────┴─────────────────────────────────────────┘
```

### 2.3 ハンドシェイクレスポンスの詳細

```
サーバー → クライアント（HTTPレスポンス）:

  HTTP/1.1 101 Switching Protocols
  Upgrade: websocket
  Connection: Upgrade
  Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
  Sec-WebSocket-Protocol: chat
  Sec-WebSocket-Extensions: permessage-deflate

  101 Switching Protocols:
  → このレスポンス以降、同じTCP接続がWebSocketプロトコルに切り替わる
  → HTTPのセマンティクスは適用されなくなる
  → 以降はWebSocketフレーム単位で通信が行われる
```

### 2.4 Sec-WebSocket-Accept の計算過程

Sec-WebSocket-Accept値は、クライアントが送信したSec-WebSocket-KeyとRFC 6455で定義されたGUID（マジックストリング）を結合し、SHA-1ハッシュを計算してBase64エンコードすることで生成される。この仕組みはクロスプロトコル攻撃を防ぐために設計されている。

```typescript
// Sec-WebSocket-Accept の計算実装
import { createHash } from 'crypto';

function computeAcceptKey(clientKey: string): string {
  // RFC 6455で規定されたマジックストリング（GUID）
  const MAGIC_STRING = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

  // Step 1: クライアントキーとGUIDを結合
  const combined = clientKey + MAGIC_STRING;
  // 例: "dGhlIHNhbXBsZSBub25jZQ==" + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

  // Step 2: SHA-1ハッシュを計算
  const hash = createHash('sha1').update(combined).digest();

  // Step 3: Base64エンコード
  const acceptKey = hash.toString('base64');
  // 結果: "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

  return acceptKey;
}

// 使用例
const clientKey = 'dGhlIHNhbXBsZSBub25jZQ==';
console.log(computeAcceptKey(clientKey));
// → "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
```

このメカニズムの目的は、サーバーがWebSocketプロトコルを理解していることの証明である。HTTPサーバーが誤ってWebSocket接続を受け入れることを防ぎ、プロキシがキャッシュを汚染する攻撃（Cache Poisoning）も防止する。ただし、これは暗号学的な認証ではなく、あくまでプロトコル互換性の確認であることに注意が必要である。

---

## 3. WebSocketフレーム構造

### 3.1 フレームフォーマットの詳細

WebSocketプロトコルはフレーム単位でデータを送受信する。各フレームは2バイト以上のヘッダーとペイロードで構成される。

```
WebSocketフレームの詳細構造（RFC 6455 Section 5.2）:

   0                   1                   2                   3
   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  ┌─┬─┬─┬─┬───────┬─┬─────────────────────────────────────────────┐
  │F│R│R│R│ opcode│M│    Payload length (7 bits)                   │
  │I│S│S│S│ (4bit)│A│                                              │
  │N│V│V│V│       │S│                                              │
  │ │1│2│3│       │K│                                              │
  ├─┴─┴─┴─┴───────┴─┼─────────────────────────────────────────────┤
  │ Extended payload length (16 or 64 bits, if payload len == 126  │
  │ or 127)                                                        │
  ├─────────────────────────────────────────────────────────────────┤
  │ Masking-key (32 bits, only if MASK bit is set)                 │
  ├─────────────────────────────────────────────────────────────────┤
  │ Payload Data (extension data + application data)               │
  │ ...                                                            │
  └─────────────────────────────────────────────────────────────────┘

  各フィールドの説明:
  ┌──────────────┬──────┬──────────────────────────────────────────┐
  │ フィールド   │ ビット│ 説明                                     │
  ├──────────────┼──────┼──────────────────────────────────────────┤
  │ FIN          │ 1    │ 1=最終フレーム、0=継続フレームが後続     │
  ├──────────────┼──────┼──────────────────────────────────────────┤
  │ RSV1-3       │ 各1  │ 拡張用。通常は0。permessage-deflateで   │
  │              │      │ RSV1=1を使用                             │
  ├──────────────┼──────┼──────────────────────────────────────────┤
  │ Opcode       │ 4    │ フレーム種別（下記参照）                 │
  ├──────────────┼──────┼──────────────────────────────────────────┤
  │ MASK         │ 1    │ 1=マスクキーあり（C→S必須）             │
  ├──────────────┼──────┼──────────────────────────────────────────┤
  │ Payload len  │ 7    │ 0-125: そのままの長さ                    │
  │              │      │ 126: 次の2バイトが実際の長さ             │
  │              │      │ 127: 次の8バイトが実際の長さ             │
  ├──────────────┼──────┼──────────────────────────────────────────┤
  │ Masking-key  │ 32   │ ペイロードのXORマスクキー                │
  ├──────────────┼──────┼──────────────────────────────────────────┤
  │ Payload      │ 可変 │ 実際のデータ                             │
  └──────────────┴──────┴──────────────────────────────────────────┘

  Opcode一覧:
  ┌──────┬────────────┬──────────────────────────────────────────┐
  │ 値   │ 名称       │ 説明                                     │
  ├──────┼────────────┼──────────────────────────────────────────┤
  │ 0x0  │ Continuation│ 分割メッセージの継続フレーム             │
  │ 0x1  │ Text       │ テキストデータ（UTF-8エンコード）        │
  │ 0x2  │ Binary     │ バイナリデータ                           │
  │ 0x3-7│ Reserved   │ 将来の非制御フレーム用に予約             │
  │ 0x8  │ Close      │ 接続クローズ要求                         │
  │ 0x9  │ Ping       │ ヘルスチェック要求                       │
  │ 0xA  │ Pong       │ Pingへの応答                             │
  │ 0xB-F│ Reserved   │ 将来の制御フレーム用に予約               │
  └──────┴────────────┴──────────────────────────────────────────┘
```

### 3.2 フレームのマスキング

マスキングは、クライアントからサーバーへ送信されるすべてのフレームに対して適用される。これはセキュリティ上の理由から必須であり、プロキシキャッシュ汚染攻撃を防ぐために導入された。

```typescript
// マスキングアルゴリズムの実装
function maskPayload(payload: Buffer, maskKey: Buffer): Buffer {
  const masked = Buffer.alloc(payload.length);
  for (let i = 0; i < payload.length; i++) {
    // 各バイトをマスクキーの対応するバイトとXOR
    masked[i] = payload[i] ^ maskKey[i % 4];
  }
  return masked;
}

// アンマスキングも同じアルゴリズム（XORの性質: A ^ B ^ B = A）
function unmaskPayload(masked: Buffer, maskKey: Buffer): Buffer {
  return maskPayload(masked, maskKey); // 同一処理
}

// 使用例
const maskKey = Buffer.from([0x37, 0xfa, 0x21, 0x3d]);
const original = Buffer.from('Hello');
const masked = maskPayload(original, maskKey);
const restored = unmaskPayload(masked, maskKey);
console.log(restored.toString()); // → "Hello"
```

### 3.3 メッセージの分割（フラグメンテーション）

大きなメッセージは複数のフレームに分割して送信できる。これによりメモリ使用量を抑えつつ、ストリーミング的にデータを送信することが可能になる。

```
メッセージフラグメンテーションの例:

  "Hello, World! This is a long message." を3フレームに分割:

  フレーム1: FIN=0, Opcode=0x1 (Text), Payload="Hello, "
    → 最初のフレーム（FIN=0は「まだ続きがある」の意味）
    → Opcodeはメッセージ全体の型を示す

  フレーム2: FIN=0, Opcode=0x0 (Continuation), Payload="World! This "
    → 中間フレーム（FIN=0, Opcode=0x0で継続を示す）

  フレーム3: FIN=1, Opcode=0x0 (Continuation), Payload="is a long message."
    → 最終フレーム（FIN=1で「これが最後」を示す）

  重要な制約:
  - 制御フレーム（Ping/Pong/Close）はフラグメンテーション不可
  - 制御フレームは分割されたメッセージの途中に挿入可能
  - 制御フレームのペイロードは125バイト以下でなければならない
```

### 3.4 クローズハンドシェイク

WebSocket接続の終了は双方合意のクローズハンドシェイクによって行われる。

```
クローズハンドシェイクの流れ:

  ┌─────────┐                      ┌─────────┐
  │ Client  │                      │ Server  │
  │         │ ── Close(1000) ────→ │         │  1. 一方がCloseフレームを送信
  │         │                      │         │     ステータスコード + 理由
  │         │ ←── Close(1000) ──── │         │  2. 相手もCloseフレームで応答
  │         │                      │         │
  │         │ ── TCP FIN ────────→ │         │  3. TCP接続を切断
  └─────────┘                      └─────────┘

  ステータスコード一覧:
  ┌──────┬──────────────────┬───────────────────────────────────────┐
  │ コード│ 名称              │ 説明                                 │
  ├──────┼──────────────────┼───────────────────────────────────────┤
  │ 1000 │ Normal Closure   │ 正常終了                             │
  │ 1001 │ Going Away       │ サーバーシャットダウン/ページ遷移    │
  │ 1002 │ Protocol Error   │ プロトコル違反                       │
  │ 1003 │ Unsupported Data │ 未対応のデータ型を受信               │
  │ 1005 │ No Status Rcvd   │ ステータスコードなし（内部用）       │
  │ 1006 │ Abnormal Closure │ 異常切断（内部用、送信不可）         │
  │ 1007 │ Invalid Payload  │ 不正なペイロード（例: 不正UTF-8）    │
  │ 1008 │ Policy Violation │ ポリシー違反                         │
  │ 1009 │ Message Too Big  │ メッセージサイズ超過                 │
  │ 1010 │ Mandatory Ext.   │ 必要な拡張機能が未対応               │
  │ 1011 │ Internal Error   │ サーバー内部エラー                   │
  │ 1015 │ TLS Handshake    │ TLSハンドシェイク失敗（内部用）      │
  └──────┴──────────────────┴───────────────────────────────────────┘
```

---

## 4. サーバー実装パターン

### 4.1 Node.js + ws ライブラリによる本格実装

```typescript
// server.ts - 本格的なWebSocketサーバー実装
import { WebSocketServer, WebSocket, RawData } from 'ws';
import { createServer, IncomingMessage } from 'http';
import { parse as parseUrl } from 'url';

// =============================================================
// 型定義
// =============================================================
interface ClientInfo {
  id: string;
  ws: WebSocket;
  rooms: Set<string>;
  isAlive: boolean;
  lastActivity: number;
  metadata: Record<string, unknown>;
}

interface Message {
  type: string;
  room?: string;
  to?: string;
  data?: unknown;
  timestamp: number;
}

// =============================================================
// WebSocketサーバークラス
// =============================================================
class RealtimeServer {
  private wss: WebSocketServer;
  private clients: Map<string, ClientInfo> = new Map();
  private rooms: Map<string, Set<string>> = new Map();
  private heartbeatInterval: ReturnType<typeof setInterval>;
  private messageHandlers: Map<string, (client: ClientInfo, msg: Message) => void>;

  constructor(port: number) {
    const server = createServer((req, res) => {
      // HTTPエンドポイント（ヘルスチェック等）
      if (req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          status: 'ok',
          connections: this.clients.size,
          rooms: this.rooms.size,
          uptime: process.uptime(),
        }));
        return;
      }
      res.writeHead(404).end();
    });

    this.wss = new WebSocketServer({
      server,
      // ハンドシェイク時の認証
      verifyClient: (info, callback) => {
        const token = this.extractToken(info.req);
        if (!token || !this.validateToken(token)) {
          callback(false, 401, 'Unauthorized');
          return;
        }
        callback(true);
      },
      // 最大ペイロードサイズ（1MB）
      maxPayload: 1024 * 1024,
      // permessage-deflate 圧縮
      perMessageDeflate: {
        zlibDeflateOptions: { chunkSize: 1024, memLevel: 7, level: 3 },
        zlibInflateOptions: { chunkSize: 10 * 1024 },
        clientNoContextTakeover: true,
        serverNoContextTakeover: true,
        serverMaxWindowBits: 10,
        concurrencyLimit: 10,
        threshold: 1024, // 1KB以上のメッセージのみ圧縮
      },
    });

    // メッセージハンドラーの登録
    this.messageHandlers = new Map([
      ['join', this.handleJoin.bind(this)],
      ['leave', this.handleLeave.bind(this)],
      ['broadcast', this.handleBroadcast.bind(this)],
      ['direct', this.handleDirectMessage.bind(this)],
      ['room_message', this.handleRoomMessage.bind(this)],
    ]);

    this.setupConnectionHandler();
    this.heartbeatInterval = this.startHeartbeat();

    server.listen(port, () => {
      console.log(`WebSocket server listening on port ${port}`);
    });
  }

  // ---------------------------------------------------------
  // 接続ハンドラー
  // ---------------------------------------------------------
  private setupConnectionHandler(): void {
    this.wss.on('connection', (ws: WebSocket, req: IncomingMessage) => {
      const clientId = crypto.randomUUID();
      const clientInfo: ClientInfo = {
        id: clientId,
        ws,
        rooms: new Set(),
        isAlive: true,
        lastActivity: Date.now(),
        metadata: {
          ip: req.socket.remoteAddress,
          userAgent: req.headers['user-agent'],
          connectedAt: new Date().toISOString(),
        },
      };

      this.clients.set(clientId, clientInfo);
      console.log(`Client connected: ${clientId} (total: ${this.clients.size})`);

      // ウェルカムメッセージ
      this.sendTo(ws, {
        type: 'welcome',
        data: { clientId, serverTime: Date.now() },
        timestamp: Date.now(),
      });

      // メッセージ受信
      ws.on('message', (raw: RawData) => {
        try {
          clientInfo.lastActivity = Date.now();
          const message: Message = JSON.parse(raw.toString());
          this.routeMessage(clientInfo, message);
        } catch (error) {
          this.sendTo(ws, {
            type: 'error',
            data: { message: 'Invalid message format' },
            timestamp: Date.now(),
          });
        }
      });

      // Pong応答
      ws.on('pong', () => {
        clientInfo.isAlive = true;
      });

      // 切断処理
      ws.on('close', (code: number, reason: Buffer) => {
        console.log(`Client disconnected: ${clientId} (code: ${code})`);
        // 所属ルームから退出
        for (const room of clientInfo.rooms) {
          this.leaveRoom(clientId, room);
        }
        this.clients.delete(clientId);
      });

      // エラー処理
      ws.on('error', (error: Error) => {
        console.error(`WebSocket error for ${clientId}: ${error.message}`);
      });
    });
  }

  // ---------------------------------------------------------
  // メッセージルーティング
  // ---------------------------------------------------------
  private routeMessage(client: ClientInfo, message: Message): void {
    const handler = this.messageHandlers.get(message.type);
    if (handler) {
      handler(client, message);
    } else {
      this.sendTo(client.ws, {
        type: 'error',
        data: { message: `Unknown message type: ${message.type}` },
        timestamp: Date.now(),
      });
    }
  }

  // ---------------------------------------------------------
  // メッセージハンドラー
  // ---------------------------------------------------------
  private handleJoin(client: ClientInfo, msg: Message): void {
    const room = msg.room;
    if (!room) return;
    this.joinRoom(client.id, room);
    this.sendTo(client.ws, {
      type: 'joined',
      room,
      data: { members: this.getRoomMembers(room).length },
      timestamp: Date.now(),
    });
  }

  private handleLeave(client: ClientInfo, msg: Message): void {
    const room = msg.room;
    if (!room) return;
    this.leaveRoom(client.id, room);
  }

  private handleBroadcast(client: ClientInfo, msg: Message): void {
    this.broadcast({
      type: 'broadcast',
      data: { from: client.id, content: msg.data },
      timestamp: Date.now(),
    }, client.id);
  }

  private handleDirectMessage(client: ClientInfo, msg: Message): void {
    if (!msg.to) return;
    const target = this.clients.get(msg.to);
    if (target) {
      this.sendTo(target.ws, {
        type: 'direct',
        data: { from: client.id, content: msg.data },
        timestamp: Date.now(),
      });
    }
  }

  private handleRoomMessage(client: ClientInfo, msg: Message): void {
    const room = msg.room;
    if (!room || !client.rooms.has(room)) return;
    this.broadcastToRoom(room, {
      type: 'room_message',
      room,
      data: { from: client.id, content: msg.data },
      timestamp: Date.now(),
    }, client.id);
  }

  // ---------------------------------------------------------
  // ルーム管理
  // ---------------------------------------------------------
  private joinRoom(clientId: string, room: string): void {
    if (!this.rooms.has(room)) {
      this.rooms.set(room, new Set());
    }
    this.rooms.get(room)!.add(clientId);
    this.clients.get(clientId)?.rooms.add(room);
  }

  private leaveRoom(clientId: string, room: string): void {
    this.rooms.get(room)?.delete(clientId);
    if (this.rooms.get(room)?.size === 0) {
      this.rooms.delete(room);
    }
    this.clients.get(clientId)?.rooms.delete(room);
  }

  private getRoomMembers(room: string): string[] {
    return Array.from(this.rooms.get(room) || []);
  }

  // ---------------------------------------------------------
  // 送信ユーティリティ
  // ---------------------------------------------------------
  private sendTo(ws: WebSocket, message: Message): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  private broadcast(message: Message, excludeId?: string): void {
    const data = JSON.stringify(message);
    this.clients.forEach((client) => {
      if (client.id !== excludeId && client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(data);
      }
    });
  }

  private broadcastToRoom(room: string, message: Message, excludeId?: string): void {
    const data = JSON.stringify(message);
    const members = this.rooms.get(room);
    if (!members) return;
    for (const memberId of members) {
      if (memberId === excludeId) continue;
      const client = this.clients.get(memberId);
      if (client && client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(data);
      }
    }
  }

  // ---------------------------------------------------------
  // ハートビート
  // ---------------------------------------------------------
  private startHeartbeat(): ReturnType<typeof setInterval> {
    return setInterval(() => {
      this.clients.forEach((client, id) => {
        if (!client.isAlive) {
          console.log(`Client ${id} failed heartbeat, terminating`);
          client.ws.terminate();
          this.clients.delete(id);
          return;
        }
        client.isAlive = false;
        client.ws.ping();
      });
    }, 30000);
  }

  // ---------------------------------------------------------
  // 認証ユーティリティ
  // ---------------------------------------------------------
  private extractToken(req: IncomingMessage): string | null {
    const url = parseUrl(req.url || '', true);
    return (url.query.token as string) || null;
  }

  private validateToken(token: string): boolean {
    // 実際のアプリケーションではJWT検証等を行う
    return token.length > 0;
  }

  // ---------------------------------------------------------
  // シャットダウン
  // ---------------------------------------------------------
  shutdown(): void {
    clearInterval(this.heartbeatInterval);
    this.clients.forEach((client) => {
      client.ws.close(1001, 'Server shutting down');
    });
    this.wss.close();
  }
}

// サーバー起動
const server = new RealtimeServer(8080);

// グレースフルシャットダウン
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down...');
  server.shutdown();
  process.exit(0);
});
```

### 4.2 Go言語によるWebSocketサーバー

```go
// main.go - Go + gorilla/websocket による実装
package main

import (
    "encoding/json"
    "log"
    "net/http"
    "sync"
    "time"

    "github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
    ReadBufferSize:  1024,
    WriteBufferSize: 1024,
    CheckOrigin: func(r *http.Request) bool {
        // 本番環境では適切なオリジン検証を行うこと
        origin := r.Header.Get("Origin")
        return origin == "https://example.com"
    },
}

type Hub struct {
    clients    map[*Client]bool
    broadcast  chan []byte
    register   chan *Client
    unregister chan *Client
    mu         sync.RWMutex
}

type Client struct {
    hub  *Hub
    conn *websocket.Conn
    send chan []byte
}

type Message struct {
    Type string          `json:"type"`
    Data json.RawMessage `json:"data"`
}

func newHub() *Hub {
    return &Hub{
        clients:    make(map[*Client]bool),
        broadcast:  make(chan []byte, 256),
        register:   make(chan *Client),
        unregister: make(chan *Client),
    }
}

func (h *Hub) run() {
    for {
        select {
        case client := <-h.register:
            h.mu.Lock()
            h.clients[client] = true
            h.mu.Unlock()
        case client := <-h.unregister:
            h.mu.Lock()
            if _, ok := h.clients[client]; ok {
                delete(h.clients, client)
                close(client.send)
            }
            h.mu.Unlock()
        case message := <-h.broadcast:
            h.mu.RLock()
            for client := range h.clients {
                select {
                case client.send <- message:
                default:
                    close(client.send)
                    delete(h.clients, client)
                }
            }
            h.mu.RUnlock()
        }
    }
}

func (c *Client) readPump() {
    defer func() {
        c.hub.unregister <- c
        c.conn.Close()
    }()
    c.conn.SetReadLimit(512 * 1024) // 512KB
    c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
    c.conn.SetPongHandler(func(string) error {
        c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
        return nil
    })
    for {
        _, message, err := c.conn.ReadMessage()
        if err != nil {
            break
        }
        c.hub.broadcast <- message
    }
}

func (c *Client) writePump() {
    ticker := time.NewTicker(30 * time.Second)
    defer func() {
        ticker.Stop()
        c.conn.Close()
    }()
    for {
        select {
        case message, ok := <-c.send:
            if !ok {
                c.conn.WriteMessage(websocket.CloseMessage, []byte{})
                return
            }
            c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
            if err := c.conn.WriteMessage(websocket.TextMessage, message); err != nil {
                return
            }
        case <-ticker.C:
            c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
            if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
                return
            }
        }
    }
}

func serveWs(hub *Hub, w http.ResponseWriter, r *http.Request) {
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        log.Println("Upgrade error:", err)
        return
    }
    client := &Client{hub: hub, conn: conn, send: make(chan []byte, 256)}
    client.hub.register <- client
    go client.writePump()
    go client.readPump()
}

func main() {
    hub := newHub()
    go hub.run()
    http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
        serveWs(hub, w, r)
    })
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

---

## 5. クライアント実装パターン

### 5.1 堅牢なブラウザクライアント

実用的なWebSocketクライアントには、再接続ロジック、メッセージキューイング、イベントエミッターパターンが不可欠である。以下は本番環境を想定した実装例である。

```typescript
// websocket-client.ts - 本番向けWebSocketクライアント
type MessageHandler = (data: unknown) => void;
type ConnectionState = 'connecting' | 'connected' | 'disconnecting' | 'disconnected';

interface WebSocketClientOptions {
  url: string;
  protocols?: string | string[];
  reconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectBaseDelay?: number;
  reconnectMaxDelay?: number;
  heartbeatInterval?: number;
  messageQueueSize?: number;
}

class RobustWebSocketClient {
  private ws: WebSocket | null = null;
  private state: ConnectionState = 'disconnected';
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private messageQueue: string[] = [];
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private options: Required<WebSocketClientOptions>;

  constructor(options: WebSocketClientOptions) {
    this.options = {
      protocols: [],
      reconnect: true,
      maxReconnectAttempts: 10,
      reconnectBaseDelay: 1000,
      reconnectMaxDelay: 30000,
      heartbeatInterval: 30000,
      messageQueueSize: 100,
      ...options,
    };
  }

  // ---------------------------------------------------
  // 接続管理
  // ---------------------------------------------------
  connect(): void {
    if (this.state === 'connecting' || this.state === 'connected') {
      console.warn('WebSocket is already connected or connecting');
      return;
    }

    this.state = 'connecting';
    this.emit('stateChange', { state: this.state });

    try {
      this.ws = new WebSocket(this.options.url, this.options.protocols);
    } catch (error) {
      this.handleConnectionFailure();
      return;
    }

    this.ws.onopen = () => {
      this.state = 'connected';
      this.reconnectAttempts = 0;
      this.emit('stateChange', { state: this.state });
      this.emit('connected', {});
      this.startHeartbeat();
      this.flushMessageQueue();
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'pong') {
          // ハートビート応答処理
          return;
        }
        this.emit(message.type, message.data);
        this.emit('message', message);
      } catch {
        // JSONでないメッセージ
        this.emit('rawMessage', event.data);
      }
    };

    this.ws.onclose = (event: CloseEvent) => {
      this.stopHeartbeat();
      const wasConnected = this.state === 'connected';
      this.state = 'disconnected';
      this.emit('stateChange', { state: this.state });
      this.emit('disconnected', {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean,
      });

      // 意図しない切断で再接続が有効な場合
      if (wasConnected && !event.wasClean && this.options.reconnect) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      this.emit('error', { message: 'WebSocket connection error' });
    };
  }

  // ---------------------------------------------------
  // 再接続（指数バックオフ + ジッター）
  // ---------------------------------------------------
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      this.emit('reconnectFailed', {
        attempts: this.reconnectAttempts,
      });
      return;
    }

    // 指数バックオフ: baseDelay * 2^attempts
    const exponentialDelay =
      this.options.reconnectBaseDelay * Math.pow(2, this.reconnectAttempts);

    // 最大遅延でキャップ
    const cappedDelay = Math.min(exponentialDelay, this.options.reconnectMaxDelay);

    // ジッター: 0.5〜1.5倍のランダム係数
    const jitter = 0.5 + Math.random();
    const delay = Math.floor(cappedDelay * jitter);

    this.reconnectAttempts++;
    this.emit('reconnecting', {
      attempt: this.reconnectAttempts,
      delay,
    });

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  // ---------------------------------------------------
  // メッセージ送信（キュー付き）
  // ---------------------------------------------------
  send(type: string, data: unknown = {}): boolean {
    const message = JSON.stringify({ type, data, timestamp: Date.now() });

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(message);
      return true;
    }

    // 接続中はキューに追加
    if (this.messageQueue.length < this.options.messageQueueSize) {
      this.messageQueue.push(message);
      return false;
    }

    console.warn('Message queue is full, dropping message');
    return false;
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      const message = this.messageQueue.shift()!;
      this.ws.send(message);
    }
  }

  // ---------------------------------------------------
  // ハートビート
  // ---------------------------------------------------
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      }
    }, this.options.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // ---------------------------------------------------
  // イベントエミッター
  // ---------------------------------------------------
  on(event: string, handler: MessageHandler): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);

    // アンサブスクライブ関数を返す
    return () => {
      this.handlers.get(event)?.delete(handler);
    };
  }

  private emit(event: string, data: unknown): void {
    this.handlers.get(event)?.forEach((handler) => {
      try {
        handler(data);
      } catch (error) {
        console.error(`Error in handler for event "${event}":`, error);
      }
    });
  }

  // ---------------------------------------------------
  // 切断
  // ---------------------------------------------------
  disconnect(code = 1000, reason = 'Normal closure'): void {
    this.state = 'disconnecting';
    this.options.reconnect = false; // 再接続を無効化
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    this.stopHeartbeat();
    this.ws?.close(code, reason);
  }

  // ---------------------------------------------------
  // ステート取得
  // ---------------------------------------------------
  getState(): ConnectionState {
    return this.state;
  }

  getQueueSize(): number {
    return this.messageQueue.length;
  }
}

// 使用例
const client = new RobustWebSocketClient({
  url: 'wss://api.example.com/ws',
  reconnect: true,
  maxReconnectAttempts: 15,
  reconnectBaseDelay: 1000,
  heartbeatInterval: 25000,
});

// イベントリスナー登録
client.on('connected', () => {
  console.log('WebSocket connected');
  client.send('join', { room: 'general' });
});

client.on('chat', (data) => {
  console.log('Chat message:', data);
});

client.on('reconnecting', (info) => {
  console.log(`Reconnecting (attempt ${(info as any).attempt})...`);
});

client.connect();
```

### 5.2 React Hooks による WebSocket統合

```typescript
// useWebSocket.ts - React用カスタムフック
import { useRef, useState, useEffect, useCallback } from 'react';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: unknown) => void;
  onConnect?: () => void;
  onDisconnect?: (event: CloseEvent) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
}

interface UseWebSocketReturn {
  send: (data: unknown) => void;
  isConnected: boolean;
  lastMessage: unknown | null;
  disconnect: () => void;
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    url,
    onMessage,
    onConnect,
    onDisconnect,
    reconnect = true,
    reconnectInterval = 3000,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<unknown | null>(null);

  const connect = useCallback(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setIsConnected(true);
      onConnect?.();
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLastMessage(data);
      onMessage?.(data);
    };

    ws.onclose = (event) => {
      setIsConnected(false);
      onDisconnect?.(event);
      if (reconnect && !event.wasClean) {
        reconnectTimerRef.current = setTimeout(connect, reconnectInterval);
      }
    };

    ws.onerror = () => {
      // エラー処理（oncloseが後続する）
    };

    wsRef.current = ws;
  }, [url, onMessage, onConnect, onDisconnect, reconnect, reconnectInterval]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      wsRef.current?.close(1000);
    };
  }, [connect]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }
    wsRef.current?.close(1000, 'User initiated disconnect');
  }, []);

  return { send, isConnected, lastMessage, disconnect };
}

// コンポーネントでの使用例
// function ChatRoom() {
//   const { send, isConnected, lastMessage } = useWebSocket({
//     url: 'wss://api.example.com/ws',
//     onMessage: (data) => console.log('Received:', data),
//   });
//
//   return (
//     <div>
//       <p>Status: {isConnected ? 'Connected' : 'Disconnected'}</p>
//       <button onClick={() => send({ type: 'chat', text: 'Hello!' })}>
//         Send
//       </button>
//     </div>
//   );
// }
```

---

## 6. Socket.IO による高レベル抽象化

### 6.1 Socket.IO の概要

Socket.IOはWebSocket上に構築されたリアルタイム通信ライブラリであり、WebSocket生APIに対して多くの付加価値を提供する。

```
Socket.IO と 生WebSocket の比較:

  ┌──────────────────────┬──────────────┬──────────────────────────┐
  │ 機能                 │ 生WebSocket  │ Socket.IO                │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ 自動再接続           │ 手動実装     │ 組み込み                 │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ フォールバック       │ なし         │ Long Polling → WebSocket │
  │ (WebSocket非対応時)  │              │                          │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ ルーム機能           │ 手動実装     │ 組み込み                 │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ 名前空間             │ なし         │ 組み込み                 │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ ACK(送達確認)        │ 手動実装     │ 組み込み                 │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ バイナリサポート     │ 手動管理     │ 自動検出・分離           │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ ブロードキャスト     │ 手動実装     │ 組み込み（ルーム対応）   │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ ミドルウェア         │ なし         │ 組み込み                 │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ マルチサーバー対応   │ 手動実装     │ Adapterで対応            │
  │ (Redis等)            │              │                          │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ プロトコル           │ 標準準拠     │ 独自プロトコル           │
  │ 互換性               │              │ （生WSクライアント不可） │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ オーバーヘッド       │ 最小         │ やや大きい               │
  ├──────────────────────┼──────────────┼──────────────────────────┤
  │ 学習コスト           │ 中〜高       │ 低〜中                   │
  └──────────────────────┴──────────────┴──────────────────────────┘

  重要な注意: Socket.IOクライアントは生WebSocketサーバーに接続できず、
  逆もまた然り。Socket.IOは独自のプロトコルレイヤーを使用している。
```

### 6.2 Socket.IO サーバー実装

```typescript
// socket-io-server.ts - Socket.IO による実装
import { Server, Socket } from 'socket.io';
import { createServer } from 'http';
import { createAdapter } from '@socket.io/redis-adapter';
import { createClient } from 'redis';

const httpServer = createServer();

const io = new Server(httpServer, {
  cors: {
    origin: ['https://example.com'],
    methods: ['GET', 'POST'],
    credentials: true,
  },
  pingInterval: 25000,    // Pingを送信する間隔
  pingTimeout: 20000,     // Pong応答を待つタイムアウト
  maxHttpBufferSize: 1e6, // 最大1MB
  transports: ['websocket', 'polling'], // トランスポート優先順位
});

// ---------------------------------------------------
// Redis Adapter（マルチサーバー対応）
// ---------------------------------------------------
async function setupRedisAdapter(): Promise<void> {
  const pubClient = createClient({ url: 'redis://localhost:6379' });
  const subClient = pubClient.duplicate();
  await Promise.all([pubClient.connect(), subClient.connect()]);
  io.adapter(createAdapter(pubClient, subClient));
  console.log('Redis adapter connected');
}

// ---------------------------------------------------
// ミドルウェア（認証）
// ---------------------------------------------------
io.use((socket: Socket, next) => {
  const token = socket.handshake.auth.token;
  if (!token) {
    return next(new Error('Authentication required'));
  }
  try {
    // JWT検証（例示のため簡略化）
    const decoded = verifyJWT(token);
    (socket as any).userId = decoded.userId;
    (socket as any).username = decoded.username;
    next();
  } catch {
    next(new Error('Invalid token'));
  }
});

// ---------------------------------------------------
// 名前空間: チャット
// ---------------------------------------------------
const chatNamespace = io.of('/chat');

chatNamespace.on('connection', (socket: Socket) => {
  const userId = (socket as any).userId;
  const username = (socket as any).username;
  console.log(`User connected: ${username} (${userId})`);

  // ルームに参加
  socket.on('joinRoom', async (roomName: string) => {
    await socket.join(roomName);
    socket.to(roomName).emit('userJoined', { userId, username, roomName });
    // ルームの参加者数を取得
    const members = await chatNamespace.in(roomName).fetchSockets();
    socket.emit('roomInfo', {
      roomName,
      memberCount: members.length,
    });
  });

  // ルームから退出
  socket.on('leaveRoom', async (roomName: string) => {
    await socket.leave(roomName);
    socket.to(roomName).emit('userLeft', { userId, username, roomName });
  });

  // メッセージ送信（ACK付き）
  socket.on('sendMessage', (data: { room: string; text: string }, ack) => {
    const message = {
      id: crypto.randomUUID(),
      from: { userId, username },
      text: data.text,
      timestamp: Date.now(),
    };
    socket.to(data.room).emit('newMessage', message);
    // 送達確認を返す
    ack?.({ status: 'ok', messageId: message.id });
  });

  // タイピングインジケーター
  socket.on('typing', (roomName: string) => {
    socket.to(roomName).volatile.emit('userTyping', { userId, username });
  });

  // 切断処理
  socket.on('disconnect', (reason: string) => {
    console.log(`User disconnected: ${username} (reason: ${reason})`);
  });
});

// ---------------------------------------------------
// 名前空間: 通知
// ---------------------------------------------------
const notificationNamespace = io.of('/notifications');

notificationNamespace.on('connection', (socket: Socket) => {
  const userId = (socket as any).userId;
  // ユーザー固有のルームに参加（個別通知用）
  socket.join(`user:${userId}`);
});

// 外部から通知を送信する関数
function sendNotification(userId: string, notification: object): void {
  notificationNamespace.to(`user:${userId}`).emit('notification', notification);
}

// ---------------------------------------------------
// JWT検証（簡略化）
// ---------------------------------------------------
function verifyJWT(token: string): { userId: string; username: string } {
  // 実際のアプリケーションではjsonwebtokenライブラリ等を使用
  return { userId: 'user-1', username: 'demo' };
}

// ---------------------------------------------------
// サーバー起動
// ---------------------------------------------------
async function main(): Promise<void> {
  await setupRedisAdapter();
  httpServer.listen(3000, () => {
    console.log('Socket.IO server listening on port 3000');
  });
}

main().catch(console.error);
```

### 6.3 Socket.IO クライアント実装

```typescript
// socket-io-client.ts - Socket.IO クライアント
import { io, Socket } from 'socket.io-client';

class ChatService {
  private socket: Socket;

  constructor(serverUrl: string, authToken: string) {
    this.socket = io(`${serverUrl}/chat`, {
      auth: { token: authToken },
      transports: ['websocket'],          // WebSocketを優先
      reconnection: true,                 // 自動再接続を有効化
      reconnectionAttempts: 10,           // 最大再接続試行回数
      reconnectionDelay: 1000,            // 初回再接続遅延
      reconnectionDelayMax: 10000,        // 最大再接続遅延
      timeout: 5000,                      // 接続タイムアウト
    });

    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    this.socket.on('connect', () => {
      console.log('Connected to chat server');
    });

    this.socket.on('connect_error', (error: Error) => {
      console.error('Connection error:', error.message);
    });

    this.socket.on('disconnect', (reason: string) => {
      console.log('Disconnected:', reason);
      if (reason === 'io server disconnect') {
        // サーバーが明示的に切断した場合、手動で再接続
        this.socket.connect();
      }
    });

    // 受信イベント
    this.socket.on('newMessage', (message) => {
      console.log('New message:', message);
    });

    this.socket.on('userJoined', (data) => {
      console.log(`${data.username} joined ${data.roomName}`);
    });

    this.socket.on('userTyping', (data) => {
      console.log(`${data.username} is typing...`);
    });
  }

  joinRoom(roomName: string): void {
    this.socket.emit('joinRoom', roomName);
  }

  sendMessage(room: string, text: string): Promise<{ status: string; messageId: string }> {
    return new Promise((resolve) => {
      // ACK付きのemit
      this.socket.emit('sendMessage', { room, text }, (response: any) => {
        resolve(response);
      });
    });
  }

  notifyTyping(room: string): void {
    this.socket.volatile.emit('typing', room);
  }

  disconnect(): void {
    this.socket.disconnect();
  }
}
```

---

## 7. スケーリングとアーキテクチャ

### 7.1 WebSocketスケーリングの課題

WebSocket接続はステートフルである。HTTPのようにリクエスト単位で任意のサーバーに振り分けることができないため、スケーリングには特別な考慮が必要になる。

```
WebSocketスケーリングアーキテクチャ:

  ┌──────────────────────────────────────────────────────────────┐
  │                       ロードバランサー                       │
  │                    (Sticky Sessions/IP Hash)                 │
  │  ┌──────────┐     ┌──────────┐     ┌──────────┐            │
  │  │ WS要求 A │     │ WS要求 B │     │ WS要求 C │            │
  └──┼──────────┼─────┼──────────┼─────┼──────────┼────────────┘
     │          │     │          │     │          │
     ▼          │     ▼          │     ▼          │
  ┌─────────┐  │  ┌─────────┐  │  ┌─────────┐  │
  │ WS      │  │  │ WS      │  │  │ WS      │  │
  │ Server 1│  │  │ Server 2│  │  │ Server 3│  │
  │ (100接続)│  │  │ (100接続)│  │  │ (100接続)│  │
  └────┬────┘  │  └────┬────┘  │  └────┬────┘  │
       │       │       │       │       │       │
       ▼       │       ▼       │       ▼       │
  ┌────────────┴───────────────┴───────────────┴───────────┐
  │                    Redis Pub/Sub                        │
  │              (サーバー間メッセージ連携)                  │
  │                                                        │
  │  Server1のクライアントAがServer2のクライアントBに        │
  │  メッセージを送る場合:                                  │
  │    A → Server1 → Redis(publish) → Server2 → B          │
  └────────────────────────────────────────────────────────┘

  単一サーバーの接続数目安（一般的なハードウェアの場合）:
  ┌───────────────────┬──────────────────┬───────────────────┐
  │ メモリ            │ アイドル接続     │ アクティブ接続    │
  ├───────────────────┼──────────────────┼───────────────────┤
  │ 1 GB              │ ~50,000          │ ~10,000           │
  │ 4 GB              │ ~200,000         │ ~50,000           │
  │ 16 GB             │ ~500,000+        │ ~150,000          │
  └───────────────────┴──────────────────┴───────────────────┘
  ※ アクティブ接続はメッセージ処理のCPUコストを含む
```

### 7.2 ロードバランサーの設定

WebSocket接続に対応するロードバランサーの設定例を示す。

```nginx
# nginx.conf - WebSocket対応のリバースプロキシ設定
upstream websocket_backend {
    # IPハッシュによるSticky Session
    ip_hash;

    server ws-server-1:8080;
    server ws-server-2:8080;
    server ws-server-3:8080;
}

server {
    listen 443 ssl;
    server_name ws.example.com;

    ssl_certificate     /etc/ssl/certs/example.com.crt;
    ssl_certificate_key /etc/ssl/private/example.com.key;

    location /ws {
        proxy_pass http://websocket_backend;

        # WebSocketアップグレードに必要な設定
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # クライアント情報の転送
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # タイムアウト設定（WebSocketの長時間接続に対応）
        proxy_read_timeout 86400s;  # 24時間
        proxy_send_timeout 86400s;
    }
}
```

### 7.3 セキュリティの考慮事項

WebSocket通信におけるセキュリティは、初回のHTTPハンドシェイクでの認証と、通信中のメッセージ検証の両面で対策が必要である。

**認証戦略:**

1. **ハンドシェイク時のトークン認証**: クエリパラメータまたはCookieでJWTを送信し、サーバーのverifyClientフックで検証する
2. **最初のメッセージでの認証**: 接続確立後、最初のメッセージとして認証情報を送信する（WebSocket APIではカスタムヘッダーが送れないため）
3. **定期的なトークンリフレッシュ**: 長時間接続ではトークンの有効期限が切れるため、WebSocket上でリフレッシュメカニズムを実装する

**入力検証:**

```typescript
// メッセージバリデーションの実装例
import { z } from 'zod';

// メッセージスキーマの定義
const ChatMessageSchema = z.object({
  type: z.literal('chat'),
  room: z.string().min(1).max(100).regex(/^[a-zA-Z0-9_-]+$/),
  text: z.string().min(1).max(5000).trim(),
});

const JoinRoomSchema = z.object({
  type: z.literal('join'),
  room: z.string().min(1).max(100).regex(/^[a-zA-Z0-9_-]+$/),
});

const MessageSchema = z.discriminatedUnion('type', [
  ChatMessageSchema,
  JoinRoomSchema,
]);

// メッセージ受信時のバリデーション
function handleIncomingMessage(rawData: string): void {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawData);
  } catch {
    // 不正なJSONは即座に拒否
    return;
  }

  const result = MessageSchema.safeParse(parsed);
  if (!result.success) {
    console.warn('Invalid message:', result.error.issues);
    return;
  }

  // バリデーション済みのメッセージを処理
  const message = result.data;
  switch (message.type) {
    case 'chat':
      processChatMessage(message);
      break;
    case 'join':
      processJoinRoom(message);
      break;
  }
}

// レート制限の実装
class RateLimiter {
  private counters: Map<string, { count: number; resetAt: number }> = new Map();

  constructor(
    private maxRequests: number,
    private windowMs: number,
  ) {}

  isAllowed(clientId: string): boolean {
    const now = Date.now();
    const entry = this.counters.get(clientId);

    if (!entry || now > entry.resetAt) {
      this.counters.set(clientId, { count: 1, resetAt: now + this.windowMs });
      return true;
    }

    if (entry.count >= this.maxRequests) {
      return false;
    }

    entry.count++;
    return true;
  }
}

// 1秒間に最大10メッセージ
const rateLimiter = new RateLimiter(10, 1000);

function processChatMessage(message: z.infer<typeof ChatMessageSchema>): void {
  // メッセージ処理ロジック
}

function processJoinRoom(message: z.infer<typeof JoinRoomSchema>): void {
  // ルーム参加ロジック
}
```

---

## 8. アンチパターン

### 8.1 アンチパターン1: 無制限のブロードキャスト

全クライアントに対して無差別にブロードキャストを行うと、接続数の増加に比例してサーバーの送信負荷が爆発的に増大する。これは「ブロードキャストストーム」と呼ばれ、本番環境で最も頻繁に見られる障害原因の一つである。

```typescript
// ダメな例: 全クライアントへの無制限ブロードキャスト
// 1,000接続 × 1,000メッセージ/秒 = 1,000,000メッセージ/秒の送信負荷

wss.on('connection', (ws) => {
  ws.on('message', (data) => {
    // 全クライアントに転送 → N^2問題が発生
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(data);  // 送信バッファが溢れる危険
      }
    });
  });
});

// 改善例: ルームベースの配信 + レート制限
wss.on('connection', (ws) => {
  const clientRooms = new Set<string>();
  const messageRateLimit = new RateLimiter(10, 1000);
  const clientId = crypto.randomUUID();

  ws.on('message', (raw) => {
    // レート制限チェック
    if (!messageRateLimit.isAllowed(clientId)) {
      ws.send(JSON.stringify({ type: 'error', data: 'Rate limit exceeded' }));
      return;
    }

    const message = JSON.parse(raw.toString());

    // ルーム内のメンバーのみに送信
    if (message.room && clientRooms.has(message.room)) {
      const roomMembers = rooms.get(message.room);
      if (roomMembers) {
        const payload = JSON.stringify(message);
        for (const memberId of roomMembers) {
          if (memberId === clientId) continue;
          const member = clients.get(memberId);
          if (member && member.readyState === WebSocket.OPEN) {
            // バッファリングされたメッセージ量をチェック
            if (member.bufferedAmount < 1024 * 1024) {
              member.send(payload);
            }
          }
        }
      }
    }
  });
});
```

**問題点の整理:**
- 接続数Nに対して、1メッセージの送信コストがO(N)になる
- 全員がメッセージを送信すると、トータルのコストはO(N^2)
- `bufferedAmount`の確認なしに送信すると、送信バッファのメモリが際限なく増大
- サーバーのCPU使用率が100%に張り付き、新規接続を受け付けられなくなる

**対策:**
1. ルームベースの配信スコープ制限
2. メッセージのレート制限（クライアント単位、ルーム単位）
3. `bufferedAmount`の監視と閾値超過時のスキップ
4. メッセージの集約（バッチ送信）

### 8.2 アンチパターン2: 再接続戦略の欠如

WebSocket接続は様々な理由で予期せず切断される。ネットワーク障害、サーバーの再起動、ロードバランサーのタイムアウトなどが代表的な原因である。再接続戦略を持たないクライアントは、ユーザー体験を著しく損なう。

```typescript
// ダメな例: 再接続ロジックがない
const ws = new WebSocket('wss://api.example.com/ws');
ws.onclose = () => {
  console.log('Connection lost');  // ここで終わり。ユーザーは手動リロードが必要
};

// さらにダメな例: 固定間隔での即座再接続
ws.onclose = () => {
  setTimeout(() => {
    new WebSocket('wss://api.example.com/ws');
    // 問題1: サーバーダウン中に全クライアントが同時に再接続を試みる
    // 問題2: 「サンダリングハード」問題 → サーバー復旧直後に接続殺到
    // 問題3: 固定間隔のため負荷が分散されない
  }, 1000);
};

// 改善例: 指数バックオフ + ジッター + 最大試行回数
class ReconnectionStrategy {
  private attempt = 0;
  private maxAttempts = 10;
  private baseDelay = 1000;    // 1秒
  private maxDelay = 60000;    // 60秒
  private timer: ReturnType<typeof setTimeout> | null = null;

  reset(): void {
    this.attempt = 0;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  shouldRetry(): boolean {
    return this.attempt < this.maxAttempts;
  }

  getNextDelay(): number {
    // 指数バックオフ
    const exponential = this.baseDelay * Math.pow(2, this.attempt);
    // 最大値でキャップ
    const capped = Math.min(exponential, this.maxDelay);
    // フルジッター（0〜cappedの範囲でランダム）
    // これにより、複数クライアントの再接続タイミングが分散される
    const jittered = Math.random() * capped;

    this.attempt++;
    return Math.floor(jittered);
  }

  scheduleReconnect(callback: () => void): void {
    if (!this.shouldRetry()) {
      console.error('Max reconnection attempts reached');
      return;
    }
    const delay = this.getNextDelay();
    console.log(`Reconnecting in ${delay}ms (attempt ${this.attempt}/${this.maxAttempts})`);
    this.timer = setTimeout(callback, delay);
  }
}
```

再接続戦略の比較:

```
  ┌─────────────────┬──────────────┬──────────────┬──────────────────┐
  │ 戦略            │ 遅延パターン │ サーバー負荷 │ 回復速度         │
  ├─────────────────┼──────────────┼──────────────┼──────────────────┤
  │ 固定間隔        │ 1s,1s,1s,... │ 非常に高い   │ 速い（過負荷）   │
  │ 指数バックオフ  │ 1s,2s,4s,8s  │ 中           │ 初回は速い       │
  │ +ジッター       │ ランダム     │ 低           │ 平均的           │
  │ +ジッター+上限  │ ランダム     │ 最低         │ 最適なバランス   │
  └─────────────────┴──────────────┴──────────────┴──────────────────┘
```

### 8.3 アンチパターン3: メモリリークを伴うイベントリスナー管理

WebSocket接続のライフサイクル管理を怠ると、イベントリスナーやタイマーのメモリリークが蓄積し、長期運用でサーバーが不安定になる。

```typescript
// ダメな例: クリーンアップが不十分
wss.on('connection', (ws) => {
  // タイマーを設定するが、切断時にクリアしない
  setInterval(() => {
    ws.ping();  // 切断後もタイマーが残り続ける
  }, 30000);

  // 外部イベントリスナーを追加するが、削除しない
  eventEmitter.on('globalUpdate', (data) => {
    ws.send(JSON.stringify(data));  // 切断後にエラーが発生
  });
});

// 改善例: 完全なクリーンアップ
wss.on('connection', (ws) => {
  // タイマーの参照を保持
  const heartbeat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.ping();
    }
  }, 30000);

  // イベントリスナーの参照を保持
  const globalUpdateHandler = (data: unknown) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
    }
  };
  eventEmitter.on('globalUpdate', globalUpdateHandler);

  // 切断時に全リソースを解放
  ws.on('close', () => {
    clearInterval(heartbeat);
    eventEmitter.off('globalUpdate', globalUpdateHandler);
    // その他のクリーンアップ処理
  });
});
```

---

## 9. エッジケース分析

### 9.1 エッジケース1: 中間プロキシによる接続断

企業ネットワークやモバイル通信では、透過プロキシやロードバランサーがWebSocket接続を予期せず切断する場合がある。特に問題となるのは以下のケースである。

```
中間プロキシによる接続断のパターン:

  ケース1: アイドルタイムアウト
  ┌─────────┐    ┌───────────┐    ┌─────────┐
  │ Client  │────│ Proxy/LB  │────│ Server  │
  │         │    │ (60秒で   │    │         │
  │         │    │  切断)    │    │         │
  └─────────┘    └───────────┘    └─────────┘

  → プロキシが一定時間データ転送がない接続をクローズする
  → 対策: 30秒間隔でPing/Pongを送信し、アイドル状態を防ぐ

  ケース2: TLSインスペクション
  → 企業のファイアウォールがWSS接続を解析しようとして失敗
  → 対策: WSSを使用しつつ、フォールバックとしてHTTPSロングポーリングを用意

  ケース3: プロキシのバッファリング
  → 一部のプロキシがWebSocketフレームをバッファリングし、
     リアルタイム性が失われる
  → 対策: X-Accel-Buffering: no ヘッダーの設定、
     またはプロキシのバッファリング無効化
```

```typescript
// 中間プロキシ対策を組み込んだ堅牢な接続管理
class ProxyAwareWebSocket {
  private ws: WebSocket | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private pongReceived = true;
  private missedPongs = 0;
  private readonly MAX_MISSED_PONGS = 2;

  connect(url: string): void {
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.startPingPong();
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'pong') {
        this.pongReceived = true;
        this.missedPongs = 0;
        return;
      }
      // 通常のメッセージ処理
    };

    this.ws.onclose = () => {
      this.stopPingPong();
    };
  }

  private startPingPong(): void {
    // 25秒間隔（多くのプロキシの60秒タイムアウトの半分以下）
    this.pingTimer = setInterval(() => {
      if (!this.pongReceived) {
        this.missedPongs++;
        if (this.missedPongs >= this.MAX_MISSED_PONGS) {
          // プロキシが接続を静かに切断した可能性
          console.warn('Connection appears dead, reconnecting...');
          this.ws?.close(4000, 'Pong timeout');
          return;
        }
      }
      this.pongReceived = false;
      this.ws?.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
    }, 25000);
  }

  private stopPingPong(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }
}
```

### 9.2 エッジケース2: メッセージ順序の保証と欠落

WebSocket over TCPは順序保証を持つが、アプリケーションレベルでは以下の場合にメッセージの順序問題や欠落が発生する。

**発生パターン:**

1. **再接続中のメッセージ欠落**: 切断から再接続完了までの間にサーバーが送信したメッセージは失われる
2. **マルチサーバー環境での順序逆転**: Redis Pub/Subを経由するメッセージと直接送信のメッセージで到着順が変わる可能性がある
3. **クライアント側のバッファオーバーフロー**: 処理速度を超えるメッセージが到着した場合、ブラウザのメモリが枯渇する

```typescript
// メッセージ順序保証と欠落検出の実装
interface SequencedMessage {
  seq: number;         // シーケンス番号
  type: string;
  data: unknown;
  timestamp: number;
}

class OrderedMessageHandler {
  private expectedSeq = 0;
  private buffer: Map<number, SequencedMessage> = new Map();
  private maxBufferSize = 1000;
  private lastProcessedSeq = -1;

  // メッセージ受信時の処理
  receive(message: SequencedMessage): SequencedMessage[] {
    const processed: SequencedMessage[] = [];

    // 重複チェック
    if (message.seq <= this.lastProcessedSeq) {
      console.warn(`Duplicate message detected: seq=${message.seq}`);
      return processed;
    }

    // 期待通りの順序であれば即座に処理
    if (message.seq === this.expectedSeq) {
      processed.push(message);
      this.lastProcessedSeq = message.seq;
      this.expectedSeq++;

      // バッファ内の連続するメッセージも処理
      while (this.buffer.has(this.expectedSeq)) {
        const buffered = this.buffer.get(this.expectedSeq)!;
        this.buffer.delete(this.expectedSeq);
        processed.push(buffered);
        this.lastProcessedSeq = this.expectedSeq;
        this.expectedSeq++;
      }
    } else if (message.seq > this.expectedSeq) {
      // 先行するメッセージが欠落 → バッファに保存
      if (this.buffer.size < this.maxBufferSize) {
        this.buffer.set(message.seq, message);
      }
      console.warn(
        `Gap detected: expected=${this.expectedSeq}, received=${message.seq}, ` +
        `missing ${message.seq - this.expectedSeq} message(s)`
      );
    }

    return processed;
  }

  // 再接続時にサーバーに欠落範囲を通知
  getMissingRange(): { from: number; to: number } | null {
    if (this.buffer.size === 0) return null;
    const minBuffered = Math.min(...this.buffer.keys());
    return { from: this.expectedSeq, to: minBuffered - 1 };
  }

  // 再接続時のリセット（最後に処理したシーケンス番号は保持）
  resetForReconnect(): number {
    this.buffer.clear();
    return this.lastProcessedSeq;
  }
}

// サーバー側: 再接続時の欠落メッセージ再送
class MessageHistory {
  private history: SequencedMessage[] = [];
  private maxHistory = 10000;

  store(message: SequencedMessage): void {
    this.history.push(message);
    // 古いメッセージを削除
    if (this.history.length > this.maxHistory) {
      this.history = this.history.slice(-this.maxHistory);
    }
  }

  getMessagesSince(seq: number): SequencedMessage[] {
    return this.history.filter((msg) => msg.seq > seq);
  }
}
```

### 9.3 エッジケース3: ブラウザのバックグラウンドタブ制限

モダンブラウザは、バックグラウンドタブのリソース消費を制限するために、タイマーのスロットリングや接続のサスペンドを行う場合がある。

```typescript
// バックグラウンドタブ対策
class VisibilityAwareConnection {
  private ws: WebSocket | null = null;
  private isBackgrounded = false;
  private lastServerMessage = Date.now();

  constructor() {
    // Page Visibility APIで状態を監視
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.onBackground();
      } else {
        this.onForeground();
      }
    });
  }

  private onBackground(): void {
    this.isBackgrounded = true;
    // バックグラウンドではPing間隔を延長し、不要なメッセージ受信を減らす
    this.ws?.send(JSON.stringify({
      type: 'presence',
      status: 'background',
    }));
  }

  private onForeground(): void {
    this.isBackgrounded = false;
    // フォアグラウンド復帰時に接続状態を確認
    const timeSinceLastMessage = Date.now() - this.lastServerMessage;

    if (timeSinceLastMessage > 60000) {
      // 60秒以上メッセージがない場合、接続が死んでいる可能性
      this.ws?.close(4001, 'Stale connection');
      // 再接続ロジックが発動する
    } else {
      // 接続は生きている → 最新データを要求
      this.ws?.send(JSON.stringify({
        type: 'presence',
        status: 'foreground',
        lastSeq: this.getLastProcessedSeq(),
      }));
    }
  }

  private getLastProcessedSeq(): number {
    // 最後に処理したシーケンス番号を返す
    return 0; // 実装省略
  }
}
```

---

## 10. パフォーマンス最適化

### 10.1 メッセージ圧縮

WebSocketにはpermessage-deflate拡張（RFC 7692）が定義されており、メッセージ単位でのzlib圧縮が可能である。

```
permessage-deflate の動作:

  圧縮なし:
    クライアント → [JSONテキスト 2KB] → サーバー

  圧縮あり:
    クライアント → [deflate圧縮 ~400B] → サーバー

  圧縮率の目安（JSONデータの場合）:
  ┌──────────────────┬───────────┬──────────┬───────────┐
  │ データサイズ     │ 圧縮前    │ 圧縮後   │ 圧縮率    │
  ├──────────────────┼───────────┼──────────┼───────────┤
  │ 小さいJSON       │ 100 B     │ ~90 B    │ 10%       │
  │ (圧縮非推奨)     │           │          │           │
  ├──────────────────┼───────────┼──────────┼───────────┤
  │ 中規模JSON       │ 1 KB      │ ~300 B   │ 70%       │
  ├──────────────────┼───────────┼──────────┼───────────┤
  │ 大規模JSON       │ 10 KB     │ ~2 KB    │ 80%       │
  ├──────────────────┼───────────┼──────────┼───────────┤
  │ 繰り返し構造     │ 50 KB     │ ~5 KB    │ 90%       │
  └──────────────────┴───────────┴──────────┴───────────┘

  注意: 小さなメッセージの圧縮はCPUオーバーヘッドが利点を上回る
  → 一般的にはthreshold（1024バイト等）を設定し、それ以上のみ圧縮する
```

### 10.2 バイナリプロトコルの活用

JSON over WebSocketは可読性に優れるが、高頻度通信ではオーバーヘッドが問題になる。Protocol BuffersやMessagePackなどのバイナリシリアライゼーションを活用することで、帯域幅とパース速度を大幅に改善できる。

```typescript
// MessagePack を使ったバイナリ通信の例
import { encode, decode } from '@msgpack/msgpack';

// JSON vs MessagePack のサイズ比較
const chatMessage = {
  type: 'chat',
  room: 'general',
  from: { id: 'user-123', name: 'Alice' },
  text: 'Hello, World!',
  timestamp: 1709712000000,
};

// JSON: 約120バイト
const jsonSize = JSON.stringify(chatMessage).length;

// MessagePack: 約80バイト（約33%削減）
const msgpackData = encode(chatMessage);
const msgpackSize = msgpackData.byteLength;

// WebSocketでの使用
function sendBinary(ws: WebSocket, data: object): void {
  const encoded = encode(data);
  ws.send(encoded); // バイナリフレーム（opcode 0x2）として送信
}

function receiveBinary(event: MessageEvent): object {
  if (event.data instanceof ArrayBuffer) {
    return decode(new Uint8Array(event.data)) as object;
  }
  return JSON.parse(event.data);
}
```

---

## 11. 演習問題

### 11.1 基礎演習: エコーサーバーの実装

**目標:** WebSocketの基本的な送受信を理解する

```
課題:
  1. Node.js + ws ライブラリでWebSocketサーバーを作成する
  2. クライアントから受信したメッセージをそのまま返す（エコー）
  3. 接続時にウェルカムメッセージを送信する
  4. 切断時にログを出力する
  5. ブラウザのコンソールからWebSocket APIで接続テストを行う

期待される動作:
  クライアント → "Hello" → サーバー
  クライアント ← "Echo: Hello" ← サーバー

ヒント:
  - WebSocketServer のインスタンスを作成する
  - 'connection' イベントでクライアントを受け付ける
  - 'message' イベントでメッセージを受信し、加工して返す
  - ws.send() でメッセージを送信する

拡張課題:
  - メッセージに受信時刻のタイムスタンプを付与する
  - 累計メッセージ数をカウントし、レスポンスに含める
  - 接続中のクライアント数をウェルカムメッセージに含める
```

### 11.2 応用演習: チャットルームの実装

**目標:** ルーム管理、ブロードキャスト、メッセージ形式の設計を実践する

```
課題:
  1. 複数のチャットルームをサポートするWebSocketサーバーを実装する
  2. 以下のメッセージタイプを実装する:
     - join: ルームに参加
     - leave: ルームから退出
     - message: ルーム内にメッセージ送信
     - list_rooms: 存在するルーム一覧を取得
     - list_members: ルームのメンバー一覧を取得
  3. メッセージはJSON形式とし、typeフィールドで種別を識別する
  4. ルーム参加/退出時に、同じルームの他のメンバーに通知する
  5. 30秒間隔のPing/Pongヘルスチェックを実装する

メッセージプロトコル例:
  送信: { "type": "join", "room": "general" }
  受信: { "type": "joined", "room": "general", "members": 3 }

  送信: { "type": "message", "room": "general", "text": "Hi!" }
  受信: { "type": "message", "room": "general",
          "from": "user-abc", "text": "Hi!",
          "timestamp": 1709712000000 }

拡張課題:
  - ニックネーム機能の追加
  - メッセージ履歴の保持（最新50件）
  - タイピングインジケーターの実装
  - ダイレクトメッセージ機能の追加
```

### 11.3 発展演習: リアルタイムコラボレーション

**目標:** OT（Operational Transformation）やCRDTの基本概念を理解し、同時編集の課題に取り組む

```
課題:
  1. 複数ユーザーが同時にテキストを編集できるリアルタイムエディタを実装する
  2. 以下の要素を含むアーキテクチャを設計する:
     - WebSocketサーバー（操作の中継と競合解決）
     - クライアント（テキストエリアとWebSocket通信）
     - 操作ログ（編集履歴の記録）
  3. 操作は以下の形式で送受信する:
     - insert: { type: "insert", pos: 5, text: "hello" }
     - delete: { type: "delete", pos: 5, len: 3 }
  4. 基本的な競合解決を実装する:
     - 同じ位置への同時挿入 → クライアントIDで順序決定
     - 削除範囲と挿入位置の重複 → 位置の調整
  5. undo/redo機能を実装する

アーキテクチャ:
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Editor A │ ←──→ │  Server  │ ←──→ │ Editor B │
  │ (Browser)│      │ (Node.js)│      │ (Browser)│
  └──────────┘      └────┬─────┘      └──────────┘
                         │
                    ┌────┴─────┐
                    │ Document │
                    │  State   │
                    │ (In-mem) │
                    └──────────┘

  ヒント:
  - 最初はシンプルな「最後の書き込みが勝つ」方式で実装する
  - 次にシーケンス番号ベースの競合検出を追加する
  - 最終的にOTアルゴリズムの基本形を実装する

  参考アルゴリズム:
  - OT (Operational Transformation): Google Docsで使用
  - CRDT (Conflict-free Replicated Data Type): Figma、Notionで使用

拡張課題:
  - カーソル位置のリアルタイム共有
  - ユーザーごとのカーソル色の割り当て
  - オフライン編集とオンライン復帰時の同期
  - 操作履歴の永続化（データベースへの保存）
```

---

## 12. WebSocket と HTTP/2、HTTP/3 の関係

### 12.1 HTTP/2 における WebSocket

HTTP/2にはServer PushやストリームMultiplexingが組み込まれているが、これらはWebSocketの代替にはならない。HTTP/2のServer Pushはリソースの先読みを目的としたものであり、任意のタイミングでのデータ送信はできない。

RFC 8441（Bootstrapping WebSockets with HTTP/2）により、HTTP/2接続上でWebSocketを確立する仕組みが標準化された。これにより、HTTP/2のマルチプレキシングの恩恵を受けつつWebSocket通信が可能になる。

### 12.2 HTTP/3 (QUIC) と WebSocket

HTTP/3はUDP上のQUICプロトコルをベースとしている。RFC 9220（Bootstrapping WebSockets with HTTP/3）によりHTTP/3上でのWebSocket接続も標準化されている。QUICのHead-of-line Blocking回避やコネクションマイグレーション（WiFi→モバイル回線の切り替え時に接続維持）は、WebSocket通信にも利点をもたらす。

### 12.3 WebTransport

WebTransportはHTTP/3上に構築された新しいAPIであり、WebSocketの代替候補として注目されている。

```
WebTransport と WebSocket の比較:
  ┌──────────────────┬───────────────────┬────────────────────────┐
  │ 特性             │ WebSocket         │ WebTransport           │
  ├──────────────────┼───────────────────┼────────────────────────┤
  │ トランスポート   │ TCP               │ QUIC (UDP)             │
  ├──────────────────┼───────────────────┼────────────────────────┤
  │ HOL Blocking     │ あり              │ なし                   │
  ├──────────────────┼───────────────────┼────────────────────────┤
  │ 信頼性           │ 完全保証          │ 信頼性あり/なし選択可  │
  ├──────────────────┼───────────────────┼────────────────────────┤
  │ 複数ストリーム   │ 1接続1ストリーム  │ 複数ストリーム対応     │
  ├──────────────────┼───────────────────┼────────────────────────┤
  │ コネクション     │ 不可              │ 対応                   │
  │ マイグレーション │                   │ (QUIC機能)             │
  ├──────────────────┼───────────────────┼────────────────────────┤
  │ 0-RTT接続確立    │ 不可              │ 対応                   │
  ├──────────────────┼───────────────────┼────────────────────────┤
  │ ブラウザ対応     │ ほぼ全て          │ Chrome系のみ           │
  │ (2025年時点)     │                   │ (拡大中)               │
  ├──────────────────┼───────────────────┼────────────────────────┤
  │ 成熟度           │ 高い              │ 発展中                 │
  └──────────────────┴───────────────────┴────────────────────────┘
```

---

## 13. テストとデバッグ

### 13.1 WebSocketのテスト手法

```typescript
// Jest + ws を使ったWebSocketサーバーのテスト例
import { WebSocketServer, WebSocket } from 'ws';

describe('WebSocket Server', () => {
  let wss: WebSocketServer;
  let serverPort: number;

  beforeAll((done) => {
    wss = new WebSocketServer({ port: 0 }, () => {
      serverPort = (wss.address() as any).port;
      done();
    });

    wss.on('connection', (ws) => {
      ws.on('message', (data) => {
        const msg = JSON.parse(data.toString());
        if (msg.type === 'echo') {
          ws.send(JSON.stringify({
            type: 'echo',
            data: msg.data,
            timestamp: Date.now(),
          }));
        }
      });
    });
  });

  afterAll((done) => {
    wss.close(done);
  });

  test('should echo messages back', (done) => {
    const client = new WebSocket(`ws://localhost:${serverPort}`);

    client.on('open', () => {
      client.send(JSON.stringify({ type: 'echo', data: 'hello' }));
    });

    client.on('message', (data) => {
      const response = JSON.parse(data.toString());
      expect(response.type).toBe('echo');
      expect(response.data).toBe('hello');
      expect(response.timestamp).toBeDefined();
      client.close();
      done();
    });
  });

  test('should handle multiple concurrent connections', (done) => {
    const clientCount = 10;
    let completedCount = 0;

    for (let i = 0; i < clientCount; i++) {
      const client = new WebSocket(`ws://localhost:${serverPort}`);
      client.on('open', () => {
        client.send(JSON.stringify({ type: 'echo', data: `msg-${i}` }));
      });
      client.on('message', (data) => {
        const response = JSON.parse(data.toString());
        expect(response.data).toBe(`msg-${i}`);
        client.close();
        completedCount++;
        if (completedCount === clientCount) {
          done();
        }
      });
    }
  });
});
```

### 13.2 デバッグツール

WebSocket通信のデバッグには以下のツールが有用である。

1. **Chrome DevTools**: Network タブ → WS フィルター → メッセージの送受信をリアルタイム確認
2. **wscat**: コマンドラインからWebSocket接続をテストするツール
3. **Postman**: WebSocketリクエストの送受信に対応
4. **Wireshark**: WebSocketフレームをパケットレベルで解析

```bash
# wscat を使ったテスト
# インストール
npm install -g wscat

# サーバーに接続
wscat -c ws://localhost:8080

# サブプロトコル指定で接続
wscat -c ws://localhost:8080 -s chat

# ヘッダー付きで接続
wscat -c ws://localhost:8080 -H "Authorization: Bearer token123"
```

---

## 14. FAQ

### Q1: WebSocket接続は何本まで維持できるか？

**A:** 単一サーバーにおける最大接続数は主にメモリとファイルディスクリプタの制限に依存する。Linuxの場合、デフォルトのファイルディスクリプタ上限は1024だが、`ulimit -n`で引き上げることができる。1接続あたりのメモリ消費はアイドル状態で約20〜50KBであり、4GBのメモリを持つサーバーであれば理論上10万接続以上を維持できる。ただし、メッセージ処理のCPU負荷やアプリケーション固有のメモリ使用量を加味すると、実用的な上限はそれより低くなる。C10K問題（1万同時接続）は現代のサーバーでは容易に解決可能であり、C100K（10万接続）やそれ以上も適切なチューニングとアーキテクチャ設計で達成できる。

### Q2: WebSocketとSSE（Server-Sent Events）はどちらを選ぶべきか？

**A:** 選択基準は通信の方向性と要件に依存する。サーバーからクライアントへの一方向通知（ニュースフィード、株価更新、進捗通知）であればSSEが適している。SSEはHTTP上で動作するため、既存のインフラとの互換性が高く、自動再接続やイベントID管理が組み込みで提供される。HTTP/2環境ではSSEのパフォーマンスも優れている。一方、クライアントからサーバーへのリアルタイム送信も必要な場合（チャット、ゲーム、共同編集）はWebSocketが適している。「サーバーからのプッシュ」だけが目的であれば、WebSocketの複雑さを引き受ける必要はなく、SSEを第一候補とすべきである。

### Q3: WebSocket接続にCORS制限は適用されるか？

**A:** WebSocket接続自体にはCORSポリシーは適用されない。ブラウザはWebSocket接続のプリフライトリクエスト（OPTIONS）を送信しない。ただし、ブラウザはハンドシェイクリクエストに`Origin`ヘッダーを自動的に付与するため、サーバー側で`Origin`ヘッダーを検証することでオリジンベースのアクセス制御を実装できる。wsライブラリでは`verifyClient`オプション、Socket.IOでは`cors`オプションで設定する。Originヘッダーはブラウザが自動設定するものであり、ブラウザ以外のクライアント（curlやNode.js）では任意の値を設定できるため、Origin検証だけでは完全なセキュリティは担保できない。トークンベースの認証と組み合わせることが推奨される。

### Q4: WebSocketの通信をTLS（WSS）で保護すべきか？

**A:** 本番環境では必ずWSS（WebSocket over TLS）を使用すべきである。理由は三つある。第一に、平文のWebSocket通信は中間者攻撃やパケットスニッフィングに脆弱である。第二に、多くの企業ネットワークやISPの透過プロキシは、暗号化されていないWebSocket接続を正しく処理できず、接続が失敗することがある。WSSを使用することで、プロキシを透過できる可能性が大幅に向上する。第三に、HTTP/2環境ではTLSが実質的に必須であり、WSS接続もHTTP/2のマルチプレキシングの恩恵を受けられる。パフォーマンスへの影響は、TLS 1.3のハンドシェイクが1-RTTで完了するため、初回接続時のわずかなオーバーヘッドを除いて無視できる水準である。

### Q5: WebSocket接続が頻繁に切断される場合、どう対処すべきか？

**A:** 頻繁な切断の原因は複数考えられる。(1) ロードバランサーやプロキシのアイドルタイムアウト: Ping/Pongフレームを定期的に送信してアイドル状態を防ぐ（推奨間隔は25〜30秒）。(2) ネットワークの不安定さ: 指数バックオフ付きの自動再接続を実装し、ジッターを加えてサーバーへの負荷集中を避ける。(3) サーバー側のリソース不足: メモリ使用量とファイルディスクリプタ数を監視し、適切なリソース制限を設定する。(4) クライアントのバックグラウンド化: Page Visibility APIを活用し、バックグラウンドタブでの通信頻度を下げる。また、切断イベントのcloseコードとreasonを分析することで、切断原因の特定に役立つ。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| WebSocket | HTTP上の双方向リアルタイム通信プロトコル（RFC 6455） |
| ハンドシェイク | HTTP 101 Switching Protocols によるプロトコル切替 |
| フレーム | 2〜14バイトのヘッダー、テキスト/バイナリ対応 |
| マスキング | クライアント→サーバーは必須、XORベースの難読化 |
| 接続管理 | Ping/Pong（30秒間隔）、指数バックオフ再接続 |
| スケーリング | Sticky Session + Redis Pub/Sub による水平展開 |
| Socket.IO | 自動再接続、ルーム、名前空間等の高レベル抽象化 |
| セキュリティ | WSS必須、Origin検証、トークン認証、入力検証 |
| 代替手段 | SSE（一方向通知）、WebTransport（次世代） |
| アンチパターン | 無制限ブロードキャスト、再接続戦略欠如、メモリリーク |

---

## 次に読むべきガイド
- [[03-grpc.md]] - gRPC
- [[01-http.md]] - HTTP/HTTPS（WebSocketの基盤プロトコル）

---

## 参考文献

1. Fette, I. and Melnikov, A. "The WebSocket Protocol." RFC 6455, IETF, December 2011. https://datatracker.ietf.org/doc/html/rfc6455
2. Yoshino, T. "Compression Extensions for WebSocket." RFC 7692, IETF, December 2015. https://datatracker.ietf.org/doc/html/rfc7692
3. McManus, P. "Bootstrapping WebSockets with HTTP/2." RFC 8441, IETF, September 2018. https://datatracker.ietf.org/doc/html/rfc8441
4. Hamilton, R. "Bootstrapping WebSockets with HTTP/3." RFC 9220, IETF, June 2022. https://datatracker.ietf.org/doc/html/rfc9220
5. Grigorik, I. "High Performance Browser Networking." O'Reilly Media, 2013. Chapter 17: WebSocket.
6. Socket.IO Documentation. "Socket.IO Server API." https://socket.io/docs/v4/server-api/
7. MDN Web Docs. "WebSocket API." Mozilla Developer Network. https://developer.mozilla.org/en-US/docs/Web/API/WebSocket
