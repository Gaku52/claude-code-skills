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
