# gRPC

> gRPCはGoogleが開発した高性能RPCフレームワーク。Protocol Buffersによる型安全な通信、HTTP/2ベースの多重化、4種類のストリーミングパターンで、マイクロサービス間通信の標準的選択肢。

## この章で学ぶこと

- [ ] gRPCの基本概念とRESTとの違いを理解する
- [ ] Protocol Buffersのスキーマ定義を把握する
- [ ] 4種類のストリーミングパターンを学ぶ

---

## 1. gRPCの基本

```
gRPC = Google Remote Procedure Call
     = HTTP/2 + Protocol Buffers のRPCフレームワーク

RPC（Remote Procedure Call）:
  → リモートの関数を、ローカル関数のように呼び出す

  クライアント                     サーバー
  ┌──────────────┐              ┌──────────────┐
  │ const user = │   HTTP/2     │ GetUser(req)  │
  │  await       │ ──────────→  │ {             │
  │  client      │              │   return user │
  │  .getUser()  │ ←──────────  │ }             │
  └──────────────┘   protobuf   └──────────────┘

  gRPC vs REST:
  ┌──────────────┬──────────────┬──────────────┐
  │              │ REST         │ gRPC         │
  ├──────────────┼──────────────┼──────────────┤
  │ プロトコル    │ HTTP/1.1     │ HTTP/2       │
  │ データ形式    │ JSON(テキスト)│ Protobuf(バイナリ)│
  │ スキーマ     │ OpenAPI(任意)│ .proto(必須) │
  │ ストリーミング│ 制限あり     │ 4パターン    │
  │ コード生成    │ 任意         │ 自動         │
  │ ブラウザ     │ ネイティブ   │ grpc-web必要 │
  │ 速度         │ 普通         │ 高速(5-10倍) │
  │ 可読性       │ 高い         │ 低い         │
  └──────────────┴──────────────┴──────────────┘
```

---

## 2. Protocol Buffers

```protobuf
// user.proto — スキーマ定義
syntax = "proto3";

package user.v1;

// メッセージ定義（データ構造）
message User {
  string id = 1;         // フィールド番号（バイナリ識別子）
  string name = 2;
  string email = 3;
  int32 age = 4;
  repeated string roles = 5;  // 配列
  Address address = 6;        // ネストされたメッセージ
  optional string phone = 7;  // オプショナル
}

message Address {
  string street = 1;
  string city = 2;
  string country = 3;
}

// サービス定義（RPC）
service UserService {
  // Unary: 1リクエスト → 1レスポンス
  rpc GetUser(GetUserRequest) returns (GetUserResponse);

  // Server Streaming: 1リクエスト → 複数レスポンス
  rpc ListUsers(ListUsersRequest) returns (stream User);

  // Client Streaming: 複数リクエスト → 1レスポンス
  rpc UploadUsers(stream User) returns (UploadUsersResponse);

  // Bidirectional Streaming: 双方向ストリーム
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

message GetUserRequest {
  string id = 1;
}

message GetUserResponse {
  User user = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message UploadUsersResponse {
  int32 count = 1;
}

message ChatMessage {
  string from = 1;
  string text = 2;
  int64 timestamp = 3;
}
```

```
Protocol Buffers のエンコーディング:

  JSON:       {"id": "123", "name": "Taro", "age": 25}
  サイズ:     約42バイト

  Protobuf:   0a 03 31 32 33 12 04 54 61 72 6f 20 19
  サイズ:     約13バイト（JSON の約1/3）

  フィールド番号によるエンコード:
  → フィールド名をバイナリに含めない
  → フィールド番号 + ワイヤータイプで識別
  → 後方互換性: 番号を変えなければOK（名前変更は安全）
```

---

## 3. 4種類のストリーミング

```
① Unary RPC（通常のRPC）:
  クライアント ── リクエスト ──→ サーバー
  クライアント ←── レスポンス ── サーバー
  例: ユーザー情報取得

② Server Streaming RPC:
  クライアント ── リクエスト ──→ サーバー
  クライアント ←── データ1 ──── サーバー
  クライアント ←── データ2 ──── サーバー
  クライアント ←── データ3 ──── サーバー
  例: 検索結果の段階的配信、ログストリーミング

③ Client Streaming RPC:
  クライアント ── データ1 ──→ サーバー
  クライアント ── データ2 ──→
  クライアント ── データ3 ──→
  クライアント ←── レスポンス ── サーバー
  例: ファイルアップロード、バッチデータ送信

④ Bidirectional Streaming RPC:
  クライアント ── データ ──→ サーバー
  クライアント ←── データ ── サーバー
  クライアント ── データ ──→ サーバー
  クライアント ←── データ ── サーバー
  例: チャット、リアルタイム協調編集
```

---

## 4. サーバー実装（Node.js / TypeScript）

```typescript
// server.ts
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';

const packageDef = protoLoader.loadSync('user.proto');
const proto = grpc.loadPackageDefinition(packageDef) as any;

// Unary RPC
function getUser(
  call: grpc.ServerUnaryCall<any, any>,
  callback: grpc.sendUnaryData<any>,
): void {
  const userId = call.request.id;
  const user = findUserById(userId);

  if (!user) {
    callback({
      code: grpc.status.NOT_FOUND,
      message: `User ${userId} not found`,
    });
    return;
  }

  callback(null, { user });
}

// Server Streaming
function listUsers(call: grpc.ServerWritableStream<any, any>): void {
  const users = getAllUsers();

  for (const user of users) {
    call.write(user);
  }

  call.end();
}

// サーバー起動
const server = new grpc.Server();
server.addService(proto.user.v1.UserService.service, {
  getUser,
  listUsers,
});

server.bindAsync(
  '0.0.0.0:50051',
  grpc.ServerCredentials.createInsecure(),
  () => server.start(),
);
```

---

## 5. gRPCのエラーハンドリング

```
gRPC ステータスコード（HTTP ステータスコードとは別）:

  ┌────────────────────┬──────┬─────────────────────────────┐
  │ コード             │ 番号 │ 説明                         │
  ├────────────────────┼──────┼─────────────────────────────┤
  │ OK                 │ 0    │ 成功                         │
  │ CANCELLED          │ 1    │ クライアントがキャンセル     │
  │ INVALID_ARGUMENT   │ 3    │ 不正な引数                   │
  │ NOT_FOUND          │ 5    │ リソースが見つからない       │
  │ ALREADY_EXISTS     │ 6    │ 既に存在                     │
  │ PERMISSION_DENIED  │ 7    │ 権限なし                     │
  │ UNAUTHENTICATED    │ 16   │ 未認証                       │
  │ RESOURCE_EXHAUSTED │ 8    │ レート制限等                 │
  │ INTERNAL           │ 13   │ サーバー内部エラー           │
  │ UNAVAILABLE        │ 14   │ サービス利用不可             │
  │ DEADLINE_EXCEEDED  │ 4    │ タイムアウト                 │
  └────────────────────┴──────┴─────────────────────────────┘

  HTTP ステータスコードとのマッピング:
  NOT_FOUND → 404
  INVALID_ARGUMENT → 400
  UNAUTHENTICATED → 401
  PERMISSION_DENIED → 403
  INTERNAL → 500
  UNAVAILABLE → 503
```

---

## 6. gRPCの採用基準

```
gRPCが適している場面:
  ✓ マイクロサービス間通信（内部API）
  ✓ 低レイテンシが重要
  ✓ 型安全性が必須
  ✓ ストリーミングが必要
  ✓ 多言語環境（コード生成で統一）

RESTが適している場面:
  ✓ パブリックAPI
  ✓ ブラウザから直接アクセス
  ✓ シンプルなCRUD
  ✓ 可読性・デバッグのしやすさ重視
  ✓ キャッシュの活用（CDN等）

ハイブリッドアーキテクチャ:
  外部 → REST API Gateway → 内部 gRPC マイクロサービス

  ブラウザ/モバイル
       ↓ REST/GraphQL
  API Gateway（REST ↔ gRPC変換）
       ↓ gRPC
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ User    │←→│ Order   │←→│ Payment │
  │ Service │  │ Service │  │ Service │
  └─────────┘  └─────────┘  └─────────┘
                  ↕ gRPC
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| gRPC | HTTP/2 + Protobufの高性能RPCフレームワーク |
| Protocol Buffers | バイナリシリアライズ、JSONの1/3サイズ |
| ストリーミング | Unary, Server, Client, Bidirectional の4種 |
| 用途 | マイクロサービス間通信に最適 |

---

## 次に読むべきガイド
→ [[../02-http/00-http-basics.md]] — HTTP基礎

---

## 参考文献
1. gRPC Documentation. "Introduction to gRPC." grpc.io, 2024.
2. Google. "Protocol Buffers Language Guide." protobuf.dev, 2024.
