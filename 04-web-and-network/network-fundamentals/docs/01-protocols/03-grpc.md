# gRPC

> gRPCはGoogleが開発した高性能RPCフレームワーク。Protocol Buffersによる型安全な通信、HTTP/2ベースの多重化、4種類のストリーミングパターンで、マイクロサービス間通信の標準的選択肢。

## この章で学ぶこと

- [ ] gRPCの基本概念とRESTとの違いを理解する
- [ ] Protocol Buffersのスキーマ定義を把握する
- [ ] 4種類のストリーミングパターンを学ぶ
- [ ] サーバー・クライアント実装を習得する
- [ ] エラーハンドリングとインターセプター設計を理解する
- [ ] gRPC-Webとモバイル対応を学ぶ
- [ ] パフォーマンス最適化と運用ベストプラクティスを把握する

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
  │ デバッグ     │ curl等で簡単 │ 専用ツール必要│
  │ 負荷分散     │ L7 LB       │ L7 gRPC LB   │
  │ キャッシュ   │ HTTP標準     │ 独自実装必要 │
  └──────────────┴──────────────┴──────────────┘

gRPCの歴史:
  2001: Google社内で「Stubby」として開発開始
        → 全社的にマイクロサービス間通信の標準に
  2015: gRPC 1.0 としてOSS化
  2017: CNCF（Cloud Native Computing Foundation）に寄贈
  2024: 主要クラウドベンダーが全面サポート
        → Kubernetes、Istio、Envoy等と密接に連携

gRPCを採用している主要企業:
  Google:    全マイクロサービス間通信
  Netflix:   リアルタイムストリーミング
  Slack:     リアルタイムメッセージング
  Square:    決済処理
  Cisco:     ネットワーク機器管理
  CoreOS:    etcd のクライアントAPI
  Dropbox:   ファイル同期サービス
```

---

## 2. Protocol Buffers

```protobuf
// user.proto — スキーマ定義
syntax = "proto3";

package user.v1;

// Go のパッケージパス指定
option go_package = "github.com/example/user/v1;userv1";
// Java のパッケージ指定
option java_package = "com.example.user.v1";
option java_multiple_files = true;

// メッセージ定義（データ構造）
message User {
  string id = 1;         // フィールド番号（バイナリ識別子）
  string name = 2;
  string email = 3;
  int32 age = 4;
  repeated string roles = 5;  // 配列
  Address address = 6;        // ネストされたメッセージ
  optional string phone = 7;  // オプショナル

  // enum定義
  UserStatus status = 8;

  // タイムスタンプ型（Well-Known Types）
  google.protobuf.Timestamp created_at = 9;
  google.protobuf.Timestamp updated_at = 10;

  // マップ型
  map<string, string> metadata = 11;

  // oneof（排他的フィールド）
  oneof notification_preference {
    EmailPreference email_pref = 12;
    SmsPreference sms_pref = 13;
    PushPreference push_pref = 14;
  }
}

// Enum定義
enum UserStatus {
  USER_STATUS_UNSPECIFIED = 0;  // デフォルト値（必須）
  USER_STATUS_ACTIVE = 1;
  USER_STATUS_INACTIVE = 2;
  USER_STATUS_SUSPENDED = 3;
  USER_STATUS_DELETED = 4;
}

message EmailPreference {
  bool daily_digest = 1;
  bool weekly_summary = 2;
}

message SmsPreference {
  string phone_number = 1;
  bool urgent_only = 2;
}

message PushPreference {
  string device_token = 1;
  repeated string topics = 2;
}

message Address {
  string street = 1;
  string city = 2;
  string state = 3;
  string country = 4;
  string zip_code = 5;
  double latitude = 6;
  double longitude = 7;
}

// Well-Known Typesのインポート
import "google/protobuf/timestamp.proto";
import "google/protobuf/duration.proto";
import "google/protobuf/empty.proto";
import "google/protobuf/field_mask.proto";
import "google/protobuf/wrappers.proto";
import "google/protobuf/any.proto";
import "google/protobuf/struct.proto";

// サービス定義（RPC）
service UserService {
  // Unary: 1リクエスト → 1レスポンス
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
  rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty);

  // Server Streaming: 1リクエスト → 複数レスポンス
  rpc ListUsers(ListUsersRequest) returns (stream User);
  rpc WatchUserUpdates(WatchUserUpdatesRequest) returns (stream UserEvent);

  // Client Streaming: 複数リクエスト → 1レスポンス
  rpc UploadUsers(stream User) returns (UploadUsersResponse);
  rpc BatchCreateUsers(stream CreateUserRequest) returns (BatchCreateUsersResponse);

  // Bidirectional Streaming: 双方向ストリーム
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
  rpc SyncUsers(stream UserSyncRequest) returns (stream UserSyncResponse);
}

message GetUserRequest {
  string id = 1;
  // FieldMask で返却フィールドを制御
  google.protobuf.FieldMask field_mask = 2;
}

message GetUserResponse {
  User user = 1;
}

message CreateUserRequest {
  string name = 1;
  string email = 2;
  int32 age = 3;
  repeated string roles = 4;
  Address address = 5;
}

message CreateUserResponse {
  User user = 1;
}

message UpdateUserRequest {
  string id = 1;
  User user = 2;
  // FieldMask で更新するフィールドを指定（部分更新）
  google.protobuf.FieldMask update_mask = 3;
}

message UpdateUserResponse {
  User user = 1;
}

message DeleteUserRequest {
  string id = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
  string filter = 3;     // フィルタリング条件
  string order_by = 4;   // ソート条件
}

message UploadUsersResponse {
  int32 count = 1;
  repeated string failed_ids = 2;
}

message BatchCreateUsersResponse {
  int32 success_count = 1;
  int32 failure_count = 2;
  repeated BatchError errors = 3;
}

message BatchError {
  int32 index = 1;
  string message = 2;
  int32 code = 3;
}

message WatchUserUpdatesRequest {
  repeated string user_ids = 1;
  repeated string event_types = 2;
}

message UserEvent {
  string event_type = 1;  // "created", "updated", "deleted"
  User user = 2;
  google.protobuf.Timestamp timestamp = 3;
}

message ChatMessage {
  string from = 1;
  string text = 2;
  int64 timestamp = 3;
  string room_id = 4;
}

message UserSyncRequest {
  string operation = 1;
  User user = 2;
  string client_id = 3;
}

message UserSyncResponse {
  string operation = 1;
  User user = 2;
  bool conflict = 3;
  User resolved_user = 4;
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

  ワイヤータイプ:
  ┌──────┬──────────────┬────────────────────────────┐
  │ Type │ 名前         │ 対象                        │
  ├──────┼──────────────┼────────────────────────────┤
  │ 0    │ Varint       │ int32, int64, bool, enum    │
  │ 1    │ 64-bit       │ fixed64, sfixed64, double   │
  │ 2    │ Length-delim  │ string, bytes, message,     │
  │      │              │ repeated                    │
  │ 5    │ 32-bit       │ fixed32, sfixed32, float    │
  └──────┴──────────────┴────────────────────────────┘

  エンコード例（id = "123", フィールド番号1, ワイヤータイプ2）:
  バイト1: 0a = (1 << 3) | 2 = フィールド1, Length-delimited
  バイト2: 03 = 長さ3バイト
  バイト3-5: 31 32 33 = "123" (ASCII)

Protocol Buffers のスカラー型一覧:
  ┌──────────┬──────────────────┬────────────────┐
  │ Proto型   │ Go型             │ TypeScript型   │
  ├──────────┼──────────────────┼────────────────┤
  │ double   │ float64          │ number         │
  │ float    │ float32          │ number         │
  │ int32    │ int32            │ number         │
  │ int64    │ int64            │ bigint/string  │
  │ uint32   │ uint32           │ number         │
  │ uint64   │ uint64           │ bigint/string  │
  │ sint32   │ int32            │ number         │
  │ sint64   │ int64            │ bigint/string  │
  │ fixed32  │ uint32           │ number         │
  │ fixed64  │ uint64           │ bigint/string  │
  │ sfixed32 │ int32            │ number         │
  │ sfixed64 │ int64            │ bigint/string  │
  │ bool     │ bool             │ boolean        │
  │ string   │ string           │ string         │
  │ bytes    │ []byte           │ Uint8Array     │
  └──────────┴──────────────────┴────────────────┘

  int32 vs sint32:
  → int32: 正の値が多い場合に効率的（Varint）
  → sint32: 負の値が多い場合に効率的（ZigZag + Varint）

  int64 の注意:
  → JavaScriptのnumberは53ビット精度
  → int64はJSではstringまたはbigintに変換される
  → APIでIDを扱う場合はstringが安全
```

---

## 3. Protocol Buffersのベストプラクティス

```protobuf
// === フィールド番号の管理 ===

// フィールド番号のルール:
// 1-15: 1バイトでエンコード → 頻出フィールドに使用
// 16-2047: 2バイトでエンコード
// 2048-: 3バイト以上
// 19000-19999: 予約済み（使用不可）

message OptimizedMessage {
  // 頻出フィールドに1-15を割り当て
  string id = 1;
  string name = 2;
  int32 status = 3;

  // あまり使わないフィールドは16以降
  string description = 16;
  map<string, string> metadata = 17;
  repeated string tags = 18;
}

// === 後方互換性の維持 ===

// 安全な変更:
// ✓ フィールドの追加（新しい番号で）
// ✓ フィールドの削除（番号を予約して）
// ✓ フィールド名の変更（番号が同じなら）
// ✓ optional ↔ repeated（互換性あり）

// 破壊的変更（絶対に避ける）:
// ✗ フィールド番号の変更
// ✗ フィールドの型変更（int32 → string等）
// ✗ 削除したフィールド番号の再利用

message UserV2 {
  string id = 1;
  string full_name = 2;  // 名前変更OK（番号同じ）
  string email = 3;

  // フィールド4,5は過去に使用済み → 予約
  reserved 4, 5;
  reserved "age", "phone";

  // 新しいフィールドは新しい番号で
  string display_name = 6;
  UserProfile profile = 7;
}

// === パッケージのバージョニング ===

// APIバージョンをパッケージに含める
// package company.service.v1;
// package company.service.v2;

// ファイル構造:
// proto/
//   user/
//     v1/
//       user.proto
//       user_service.proto
//     v2/
//       user.proto
//       user_service.proto
```

```
Protocol Buffers のパフォーマンス比較:

  シリアライズ速度（1M メッセージ/秒）:
  ┌──────────────┬───────┬──────────┐
  │ 形式         │ 速度  │ 相対比   │
  ├──────────────┼───────┼──────────┤
  │ Protobuf     │ 2.8M  │ 1x       │
  │ FlatBuffers  │ 3.2M  │ 0.87x    │
  │ MessagePack  │ 1.5M  │ 1.87x    │
  │ JSON         │ 0.8M  │ 3.5x     │
  │ XML          │ 0.3M  │ 9.3x     │
  └──────────────┴───────┴──────────┘

  シリアライズ後のサイズ（同一データ）:
  ┌──────────────┬────────┬──────────┐
  │ 形式         │ バイト │ 相対比   │
  ├──────────────┼────────┼──────────┤
  │ Protobuf     │ 34     │ 1x       │
  │ FlatBuffers  │ 44     │ 1.29x    │
  │ MessagePack  │ 45     │ 1.32x    │
  │ JSON         │ 82     │ 2.41x    │
  │ XML          │ 137    │ 4.03x    │
  └──────────────┴────────┴──────────┘
```

---

## 4. 4種類のストリーミング

```
① Unary RPC（通常のRPC）:
  クライアント ── リクエスト ──→ サーバー
  クライアント ←── レスポンス ── サーバー
  例: ユーザー情報取得、認証、設定変更

  特徴:
  → 最もシンプルなパターン
  → RESTのリクエスト/レスポンスに相当
  → 大多数のAPIコールはこのパターン

② Server Streaming RPC:
  クライアント ── リクエスト ──→ サーバー
  クライアント ←── データ1 ──── サーバー
  クライアント ←── データ2 ──── サーバー
  クライアント ←── データ3 ──── サーバー
  クライアント ←── 完了通知 ─── サーバー
  例: 検索結果の段階的配信、ログストリーミング、
      株価リアルタイム配信、ニュースフィード

  特徴:
  → サーバーが任意のタイミングでデータを送信
  → クライアントはストリームの完了を待つ
  → 大量データの段階的返却に最適

③ Client Streaming RPC:
  クライアント ── データ1 ──→ サーバー
  クライアント ── データ2 ──→
  クライアント ── データ3 ──→
  クライアント ── 完了通知 ──→
  クライアント ←── レスポンス ── サーバー
  例: ファイルアップロード、バッチデータ送信、
      センサーデータ収集、ログ集約

  特徴:
  → クライアントが任意の数のメッセージを送信
  → サーバーは全メッセージ受信後にレスポンスを返す
  → 集約処理に最適

④ Bidirectional Streaming RPC:
  クライアント ── データ ──→ サーバー
  クライアント ←── データ ── サーバー
  クライアント ── データ ──→ サーバー
  クライアント ←── データ ── サーバー
  例: チャット、リアルタイム協調編集、
      ゲームの状態同期、音声/映像通話

  特徴:
  → 両方が独立してメッセージを送受信
  → 順序は送信側が制御
  → WebSocketに近い双方向通信
  → 最も柔軟だが実装も最も複雑

ストリーミングパターンの選択基準:
  ┌────────────────┬────────────┬────────────────────────┐
  │ パターン        │ 選択理由    │ 典型的なユースケース    │
  ├────────────────┼────────────┼────────────────────────┤
  │ Unary          │ 単純な操作  │ CRUD、認証、設定取得    │
  │ Server Stream  │ 大量データ  │ 検索結果、ログ配信      │
  │ Client Stream  │ データ集約  │ アップロード、バッチ    │
  │ Bidi Stream    │ リアルタイム│ チャット、ゲーム同期    │
  └────────────────┴────────────┴────────────────────────┘
```

---

## 5. サーバー実装（Node.js / TypeScript）

```typescript
// === プロジェクトセットアップ ===
// package.json の依存関係:
// "@grpc/grpc-js": "^1.9.0"
// "@grpc/proto-loader": "^0.7.0"
// "google-protobuf": "^3.21.0"

// === 方法1: 動的ロード（開発向け） ===
// server.ts
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { v4 as uuidv4 } from 'uuid';

// Proto定義のロード
const packageDef = protoLoader.loadSync('proto/user/v1/user_service.proto', {
  keepCase: true,         // フィールド名をsnake_caseのまま
  longs: String,          // int64をstringに変換
  enums: String,          // enumをstringに変換
  defaults: true,         // デフォルト値を含める
  oneofs: true,           // oneofフィールドを含める
  includeDirs: ['proto'], // インポートパスの検索ディレクトリ
});

const proto = grpc.loadPackageDefinition(packageDef) as any;

// インメモリデータストア（デモ用）
const users = new Map<string, any>();

// === Unary RPC 実装 ===
function getUser(
  call: grpc.ServerUnaryCall<any, any>,
  callback: grpc.sendUnaryData<any>,
): void {
  const userId = call.request.id;

  // メタデータからの情報取得
  const metadata = call.metadata;
  const requestId = metadata.get('x-request-id')[0] || uuidv4();
  const authToken = metadata.get('authorization')[0];

  console.log(`[${requestId}] GetUser called for id: ${userId}`);

  // 認証チェック
  if (!authToken) {
    callback({
      code: grpc.status.UNAUTHENTICATED,
      message: 'Authentication token is required',
      metadata: createErrorMetadata(requestId),
    });
    return;
  }

  const user = users.get(userId);

  if (!user) {
    callback({
      code: grpc.status.NOT_FOUND,
      message: `User ${userId} not found`,
      metadata: createErrorMetadata(requestId),
    });
    return;
  }

  // FieldMask対応（指定されたフィールドのみ返却）
  const fieldMask = call.request.field_mask;
  if (fieldMask && fieldMask.paths.length > 0) {
    const filteredUser = filterByFieldMask(user, fieldMask.paths);
    callback(null, { user: filteredUser });
  } else {
    callback(null, { user });
  }
}

// === CreateUser 実装 ===
function createUser(
  call: grpc.ServerUnaryCall<any, any>,
  callback: grpc.sendUnaryData<any>,
): void {
  const { name, email, age, roles, address } = call.request;

  // バリデーション
  const errors: string[] = [];
  if (!name || name.trim().length === 0) {
    errors.push('name is required');
  }
  if (!email || !email.includes('@')) {
    errors.push('valid email is required');
  }
  if (age !== undefined && (age < 0 || age > 150)) {
    errors.push('age must be between 0 and 150');
  }

  if (errors.length > 0) {
    callback({
      code: grpc.status.INVALID_ARGUMENT,
      message: `Validation failed: ${errors.join(', ')}`,
    });
    return;
  }

  // メール重複チェック
  for (const [, existingUser] of users) {
    if (existingUser.email === email) {
      callback({
        code: grpc.status.ALREADY_EXISTS,
        message: `User with email ${email} already exists`,
      });
      return;
    }
  }

  const user = {
    id: uuidv4(),
    name,
    email,
    age: age || 0,
    roles: roles || [],
    address: address || null,
    status: 'USER_STATUS_ACTIVE',
    created_at: { seconds: Math.floor(Date.now() / 1000), nanos: 0 },
    updated_at: { seconds: Math.floor(Date.now() / 1000), nanos: 0 },
    metadata: {},
  };

  users.set(user.id, user);
  console.log(`User created: ${user.id}`);

  callback(null, { user });
}

// === UpdateUser 実装（FieldMask対応の部分更新） ===
function updateUser(
  call: grpc.ServerUnaryCall<any, any>,
  callback: grpc.sendUnaryData<any>,
): void {
  const { id, user: updateData, update_mask } = call.request;

  const existingUser = users.get(id);
  if (!existingUser) {
    callback({
      code: grpc.status.NOT_FOUND,
      message: `User ${id} not found`,
    });
    return;
  }

  // FieldMaskに基づいた部分更新
  if (update_mask && update_mask.paths.length > 0) {
    for (const path of update_mask.paths) {
      if (path in updateData) {
        existingUser[path] = updateData[path];
      }
    }
  } else {
    // FieldMaskがない場合は全フィールド更新
    Object.assign(existingUser, updateData, { id });
  }

  existingUser.updated_at = {
    seconds: Math.floor(Date.now() / 1000),
    nanos: 0,
  };

  users.set(id, existingUser);
  callback(null, { user: existingUser });
}

// === Server Streaming 実装 ===
function listUsers(call: grpc.ServerWritableStream<any, any>): void {
  const { page_size, filter, order_by } = call.request;
  const limit = page_size || 100;

  let userList = Array.from(users.values());

  // フィルタリング
  if (filter) {
    userList = applyFilter(userList, filter);
  }

  // ソート
  if (order_by) {
    userList = applySort(userList, order_by);
  }

  // ページサイズに制限
  userList = userList.slice(0, limit);

  // ストリームで1件ずつ送信
  let index = 0;
  const sendNext = () => {
    if (index < userList.length) {
      const canWrite = call.write(userList[index]);
      index++;

      if (canWrite) {
        // すぐに次を送信
        setImmediate(sendNext);
      } else {
        // バックプレッシャー: drain イベントを待つ
        call.once('drain', sendNext);
      }
    } else {
      call.end();
    }
  };

  sendNext();

  // クライアントのキャンセルを監視
  call.on('cancelled', () => {
    console.log('ListUsers stream cancelled by client');
  });
}

// === Client Streaming 実装 ===
function uploadUsers(
  call: grpc.ServerReadableStream<any, any>,
  callback: grpc.sendUnaryData<any>,
): void {
  let count = 0;
  const failedIds: string[] = [];

  call.on('data', (user: any) => {
    try {
      // バリデーションとストア
      if (!user.name || !user.email) {
        failedIds.push(user.id || 'unknown');
        return;
      }

      const id = user.id || uuidv4();
      users.set(id, { ...user, id });
      count++;
    } catch (error) {
      failedIds.push(user.id || 'unknown');
    }
  });

  call.on('end', () => {
    callback(null, {
      count,
      failed_ids: failedIds,
    });
  });

  call.on('error', (error: Error) => {
    console.error('Upload stream error:', error);
    callback({
      code: grpc.status.INTERNAL,
      message: `Stream error: ${error.message}`,
    });
  });
}

// === Bidirectional Streaming 実装 ===
function chat(call: grpc.ServerDuplexStream<any, any>): void {
  const roomId = call.metadata.get('room-id')[0] as string || 'default';
  console.log(`Chat stream opened for room: ${roomId}`);

  call.on('data', (message: any) => {
    console.log(`[${roomId}] ${message.from}: ${message.text}`);

    // エコーバック（実際にはブロードキャスト）
    call.write({
      from: 'server',
      text: `Received: ${message.text}`,
      timestamp: Date.now(),
      room_id: roomId,
    });

    // ボットレスポンス（デモ用）
    if (message.text.toLowerCase().includes('hello')) {
      call.write({
        from: 'bot',
        text: `Hello, ${message.from}! Welcome to room ${roomId}.`,
        timestamp: Date.now(),
        room_id: roomId,
      });
    }
  });

  call.on('end', () => {
    console.log(`Chat stream closed for room: ${roomId}`);
    call.end();
  });

  call.on('error', (error: Error) => {
    console.error(`Chat stream error in room ${roomId}:`, error);
  });

  call.on('cancelled', () => {
    console.log(`Chat stream cancelled for room: ${roomId}`);
  });
}

// === ヘルパー関数 ===
function createErrorMetadata(requestId: string): grpc.Metadata {
  const metadata = new grpc.Metadata();
  metadata.set('x-request-id', requestId);
  return metadata;
}

function filterByFieldMask(obj: any, paths: string[]): any {
  const result: any = {};
  for (const path of paths) {
    if (path in obj) {
      result[path] = obj[path];
    }
  }
  return result;
}

function applyFilter(users: any[], filter: string): any[] {
  // シンプルなフィルタ実装（例: "status=active"）
  const [key, value] = filter.split('=');
  return users.filter(u => String(u[key]) === value);
}

function applySort(users: any[], orderBy: string): any[] {
  const desc = orderBy.startsWith('-');
  const field = desc ? orderBy.slice(1) : orderBy;
  return users.sort((a, b) => {
    const cmp = a[field] < b[field] ? -1 : a[field] > b[field] ? 1 : 0;
    return desc ? -cmp : cmp;
  });
}

// === サーバー起動 ===
function startServer(): void {
  const server = new grpc.Server({
    // サーバー設定
    'grpc.max_receive_message_length': 10 * 1024 * 1024, // 10MB
    'grpc.max_send_message_length': 10 * 1024 * 1024,    // 10MB
    'grpc.keepalive_time_ms': 60000,                      // 60秒
    'grpc.keepalive_timeout_ms': 20000,                   // 20秒
    'grpc.keepalive_permit_without_calls': 1,             // コールなしでもKeepalive
  });

  server.addService(proto.user.v1.UserService.service, {
    getUser,
    createUser,
    updateUser,
    listUsers,
    uploadUsers,
    chat,
  });

  const address = '0.0.0.0:50051';

  server.bindAsync(
    address,
    grpc.ServerCredentials.createInsecure(),
    (error, port) => {
      if (error) {
        console.error('Server bind error:', error);
        process.exit(1);
      }
      console.log(`gRPC server listening on port ${port}`);
    },
  );

  // グレースフルシャットダウン
  process.on('SIGTERM', () => {
    console.log('Received SIGTERM, shutting down gracefully...');
    server.tryShutdown((error) => {
      if (error) {
        console.error('Error during shutdown:', error);
        server.forceShutdown();
      }
      console.log('Server shut down successfully');
      process.exit(0);
    });
  });
}

startServer();
```

---

## 6. クライアント実装（Node.js / TypeScript）

```typescript
// client.ts
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';

const packageDef = protoLoader.loadSync('proto/user/v1/user_service.proto', {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
  includeDirs: ['proto'],
});

const proto = grpc.loadPackageDefinition(packageDef) as any;

// === クライアント作成 ===
function createClient(address: string = 'localhost:50051') {
  const client = new proto.user.v1.UserService(
    address,
    grpc.credentials.createInsecure(),
    {
      // クライアント設定
      'grpc.keepalive_time_ms': 30000,
      'grpc.keepalive_timeout_ms': 10000,
      'grpc.max_receive_message_length': 10 * 1024 * 1024,
      'grpc.initial_reconnect_backoff_ms': 1000,
      'grpc.max_reconnect_backoff_ms': 30000,
    },
  );

  return client;
}

// === Unary RPCの呼び出し ===
async function getUser(
  client: any,
  userId: string,
): Promise<any> {
  return new Promise((resolve, reject) => {
    // メタデータの設定
    const metadata = new grpc.Metadata();
    metadata.set('authorization', 'Bearer my-token');
    metadata.set('x-request-id', generateRequestId());

    // デッドライン（タイムアウト）の設定
    const deadline = new Date();
    deadline.setSeconds(deadline.getSeconds() + 5); // 5秒タイムアウト

    client.getUser(
      { id: userId, field_mask: { paths: ['name', 'email'] } },
      metadata,
      { deadline },
      (error: grpc.ServiceError | null, response: any) => {
        if (error) {
          handleGrpcError(error);
          reject(error);
          return;
        }
        resolve(response.user);
      },
    );
  });
}

// === Server Streaming の呼び出し ===
async function listAllUsers(client: any): Promise<any[]> {
  return new Promise((resolve, reject) => {
    const users: any[] = [];

    const metadata = new grpc.Metadata();
    metadata.set('authorization', 'Bearer my-token');

    const call = client.listUsers(
      { page_size: 100, filter: 'status=USER_STATUS_ACTIVE' },
      metadata,
    );

    call.on('data', (user: any) => {
      users.push(user);
      console.log(`Received user: ${user.name}`);
    });

    call.on('end', () => {
      console.log(`Total users received: ${users.length}`);
      resolve(users);
    });

    call.on('error', (error: grpc.ServiceError) => {
      handleGrpcError(error);
      reject(error);
    });

    call.on('status', (status: grpc.StatusObject) => {
      console.log(`Stream status: ${status.code} - ${status.details}`);
    });

    // 10秒後にキャンセル（タイムアウト）
    setTimeout(() => {
      call.cancel();
    }, 10000);
  });
}

// === Client Streaming の呼び出し ===
async function batchUploadUsers(
  client: any,
  userList: any[],
): Promise<any> {
  return new Promise((resolve, reject) => {
    const metadata = new grpc.Metadata();
    metadata.set('authorization', 'Bearer my-token');

    const call = client.uploadUsers(
      metadata,
      (error: grpc.ServiceError | null, response: any) => {
        if (error) {
          handleGrpcError(error);
          reject(error);
          return;
        }
        console.log(`Uploaded ${response.count} users`);
        resolve(response);
      },
    );

    // ユーザーを1件ずつストリーム送信
    for (const user of userList) {
      call.write(user);
    }

    // ストリーム終了
    call.end();
  });
}

// === Bidirectional Streaming の呼び出し ===
async function startChat(
  client: any,
  userName: string,
  roomId: string,
): Promise<void> {
  const metadata = new grpc.Metadata();
  metadata.set('authorization', 'Bearer my-token');
  metadata.set('room-id', roomId);

  const call = client.chat(metadata);

  // サーバーからのメッセージ受信
  call.on('data', (message: any) => {
    console.log(`[${message.from}]: ${message.text}`);
  });

  call.on('end', () => {
    console.log('Chat stream ended');
  });

  call.on('error', (error: grpc.ServiceError) => {
    if (error.code !== grpc.status.CANCELLED) {
      handleGrpcError(error);
    }
  });

  // メッセージ送信
  call.write({
    from: userName,
    text: 'Hello everyone!',
    timestamp: Date.now(),
    room_id: roomId,
  });

  // 定期的にメッセージ送信（デモ用）
  const interval = setInterval(() => {
    call.write({
      from: userName,
      text: `Ping at ${new Date().toISOString()}`,
      timestamp: Date.now(),
      room_id: roomId,
    });
  }, 5000);

  // 30秒後にチャット終了
  setTimeout(() => {
    clearInterval(interval);
    call.end();
  }, 30000);
}

// === エラーハンドリング ===
function handleGrpcError(error: grpc.ServiceError): void {
  const statusName = Object.keys(grpc.status).find(
    key => grpc.status[key as keyof typeof grpc.status] === error.code,
  );

  console.error(`gRPC Error [${statusName}] (${error.code}): ${error.message}`);

  // メタデータの詳細を表示
  if (error.metadata) {
    const requestId = error.metadata.get('x-request-id');
    if (requestId.length > 0) {
      console.error(`  Request ID: ${requestId[0]}`);
    }
  }

  // エラーコードに応じた処理
  switch (error.code) {
    case grpc.status.UNAVAILABLE:
      console.error('  → Service is unavailable. Retry with backoff.');
      break;
    case grpc.status.DEADLINE_EXCEEDED:
      console.error('  → Request timed out. Consider increasing deadline.');
      break;
    case grpc.status.UNAUTHENTICATED:
      console.error('  → Authentication failed. Check credentials.');
      break;
    case grpc.status.PERMISSION_DENIED:
      console.error('  → Permission denied. Check authorization.');
      break;
    case grpc.status.RESOURCE_EXHAUSTED:
      console.error('  → Rate limited. Back off and retry.');
      break;
    case grpc.status.NOT_FOUND:
      console.error('  → Resource not found.');
      break;
    default:
      console.error('  → Unexpected error.');
  }
}

function generateRequestId(): string {
  return `req-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// === メイン実行 ===
async function main(): Promise<void> {
  const client = createClient();

  try {
    // ユーザー作成
    const created = await new Promise<any>((resolve, reject) => {
      client.createUser(
        { name: 'Taro', email: 'taro@example.com', age: 25 },
        (err: any, res: any) => err ? reject(err) : resolve(res),
      );
    });
    console.log('Created:', created.user);

    // ユーザー取得
    const user = await getUser(client, created.user.id);
    console.log('Got:', user);

    // ユーザー一覧（ストリーミング）
    const allUsers = await listAllUsers(client);
    console.log('All users:', allUsers);

  } catch (error) {
    console.error('Error:', error);
  } finally {
    client.close();
  }
}

main();
```

---

## 7. Go言語での実装

```go
// === サーバー側（Go） ===
// server/main.go
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"

	pb "github.com/example/user/v1"
)

type userServer struct {
	pb.UnimplementedUserServiceServer
	mu    sync.RWMutex
	users map[string]*pb.User
}

func newUserServer() *userServer {
	return &userServer{
		users: make(map[string]*pb.User),
	}
}

// Unary RPC
func (s *userServer) GetUser(
	ctx context.Context,
	req *pb.GetUserRequest,
) (*pb.GetUserResponse, error) {
	// メタデータの取得
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, status.Error(codes.Internal, "failed to get metadata")
	}

	// 認証チェック
	authTokens := md.Get("authorization")
	if len(authTokens) == 0 {
		return nil, status.Error(codes.Unauthenticated, "missing auth token")
	}

	// コンテキストのキャンセルチェック
	if ctx.Err() == context.Canceled {
		return nil, status.Error(codes.Canceled, "request canceled")
	}
	if ctx.Err() == context.DeadlineExceeded {
		return nil, status.Error(codes.DeadlineExceeded, "deadline exceeded")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	user, exists := s.users[req.Id]
	if !exists {
		return nil, status.Errorf(
			codes.NotFound,
			"user %s not found", req.Id,
		)
	}

	// レスポンスメタデータの設定
	header := metadata.New(map[string]string{
		"x-request-id": fmt.Sprintf("req-%d", time.Now().UnixNano()),
	})
	grpc.SetHeader(ctx, header)

	return &pb.GetUserResponse{User: user}, nil
}

// Server Streaming RPC
func (s *userServer) ListUsers(
	req *pb.ListUsersRequest,
	stream pb.UserService_ListUsersServer,
) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	count := 0
	for _, user := range s.users {
		// コンテキストのキャンセルチェック
		if stream.Context().Err() != nil {
			return status.Error(codes.Canceled, "stream canceled")
		}

		if err := stream.Send(user); err != nil {
			return status.Errorf(codes.Internal, "send error: %v", err)
		}

		count++
		if req.PageSize > 0 && int32(count) >= req.PageSize {
			break
		}
	}

	return nil
}

// Client Streaming RPC
func (s *userServer) UploadUsers(
	stream pb.UserService_UploadUsersServer,
) error {
	var count int32

	for {
		user, err := stream.Recv()
		if err == io.EOF {
			// 全メッセージ受信完了
			return stream.SendAndClose(&pb.UploadUsersResponse{
				Count: count,
			})
		}
		if err != nil {
			return status.Errorf(codes.Internal, "recv error: %v", err)
		}

		s.mu.Lock()
		s.users[user.Id] = user
		s.mu.Unlock()
		count++
	}
}

// Bidirectional Streaming RPC
func (s *userServer) Chat(
	stream pb.UserService_ChatServer,
) error {
	for {
		msg, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return status.Errorf(codes.Internal, "recv error: %v", err)
		}

		log.Printf("[%s] %s: %s", msg.RoomId, msg.From, msg.Text)

		// レスポンスを送信
		reply := &pb.ChatMessage{
			From:      "server",
			Text:      fmt.Sprintf("Echo: %s", msg.Text),
			Timestamp: time.Now().Unix(),
			RoomId:    msg.RoomId,
		}

		if err := stream.Send(reply); err != nil {
			return status.Errorf(codes.Internal, "send error: %v", err)
		}
	}
}

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	// サーバーオプション
	opts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(10 * 1024 * 1024), // 10MB
		grpc.MaxSendMsgSize(10 * 1024 * 1024), // 10MB
		grpc.KeepaliveParams(keepalive.ServerParameters{
			MaxConnectionIdle:     15 * time.Minute,
			MaxConnectionAge:      30 * time.Minute,
			MaxConnectionAgeGrace: 5 * time.Second,
			Time:                  5 * time.Minute,
			Timeout:               1 * time.Second,
		}),
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			MinTime:             5 * time.Second,
			PermitWithoutStream: true,
		}),
	}

	s := grpc.NewServer(opts...)
	pb.RegisterUserServiceServer(s, newUserServer())

	log.Printf("gRPC server listening on %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
```

---

## 8. gRPCのエラーハンドリング

```
gRPC ステータスコード（HTTP ステータスコードとは別）:

  ┌────────────────────┬──────┬─────────────────────────────┐
  │ コード             │ 番号 │ 説明                         │
  ├────────────────────┼──────┼─────────────────────────────┤
  │ OK                 │ 0    │ 成功                         │
  │ CANCELLED          │ 1    │ クライアントがキャンセル     │
  │ UNKNOWN            │ 2    │ 不明なエラー                 │
  │ INVALID_ARGUMENT   │ 3    │ 不正な引数                   │
  │ DEADLINE_EXCEEDED  │ 4    │ タイムアウト                 │
  │ NOT_FOUND          │ 5    │ リソースが見つからない       │
  │ ALREADY_EXISTS     │ 6    │ 既に存在                     │
  │ PERMISSION_DENIED  │ 7    │ 権限なし                     │
  │ RESOURCE_EXHAUSTED │ 8    │ レート制限等                 │
  │ FAILED_PRECONDITION│ 9    │ 前提条件不一致               │
  │ ABORTED            │ 10   │ 操作中断（トランザクション等）│
  │ OUT_OF_RANGE       │ 11   │ 範囲外アクセス               │
  │ UNIMPLEMENTED      │ 12   │ 未実装のRPC                  │
  │ INTERNAL           │ 13   │ サーバー内部エラー           │
  │ UNAVAILABLE        │ 14   │ サービス利用不可             │
  │ DATA_LOSS          │ 15   │ データ損失                   │
  │ UNAUTHENTICATED    │ 16   │ 未認証                       │
  └────────────────────┴──────┴─────────────────────────────┘

  HTTP ステータスコードとのマッピング:
  ┌──────────────────────┬────────────────────┐
  │ gRPC Code            │ HTTP Status        │
  ├──────────────────────┼────────────────────┤
  │ OK                   │ 200 OK             │
  │ CANCELLED            │ 499 Client Closed  │
  │ UNKNOWN              │ 500 Internal       │
  │ INVALID_ARGUMENT     │ 400 Bad Request    │
  │ DEADLINE_EXCEEDED    │ 504 Gateway Timeout│
  │ NOT_FOUND            │ 404 Not Found      │
  │ ALREADY_EXISTS       │ 409 Conflict       │
  │ PERMISSION_DENIED    │ 403 Forbidden      │
  │ RESOURCE_EXHAUSTED   │ 429 Too Many Req   │
  │ FAILED_PRECONDITION  │ 400 Bad Request    │
  │ ABORTED              │ 409 Conflict       │
  │ OUT_OF_RANGE         │ 400 Bad Request    │
  │ UNIMPLEMENTED        │ 501 Not Implemented│
  │ INTERNAL             │ 500 Internal       │
  │ UNAVAILABLE          │ 503 Unavailable    │
  │ DATA_LOSS            │ 500 Internal       │
  │ UNAUTHENTICATED      │ 401 Unauthorized   │
  └──────────────────────┴────────────────────┘

  エラーコード選択ガイド:
  「引数が不正」 → INVALID_ARGUMENT
  「見つからない」 → NOT_FOUND
  「既に存在する」 → ALREADY_EXISTS
  「認証が必要」 → UNAUTHENTICATED
  「権限がない」 → PERMISSION_DENIED
  「レート制限」 → RESOURCE_EXHAUSTED
  「楽観ロック失敗」 → ABORTED
  「未実装API」 → UNIMPLEMENTED
  「一時的障害」 → UNAVAILABLE（リトライ可能）
  「内部エラー」 → INTERNAL（リトライ不可能かも）
```

```go
// === Rich Error Model（google.rpc.Status） ===
// Go での実装例

import (
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func validateAndReturnError(req *pb.CreateUserRequest) error {
	// フィールドバリデーションエラー
	var violations []*errdetails.BadRequest_FieldViolation

	if req.Name == "" {
		violations = append(violations, &errdetails.BadRequest_FieldViolation{
			Field:       "name",
			Description: "Name is required and cannot be empty",
		})
	}

	if !isValidEmail(req.Email) {
		violations = append(violations, &errdetails.BadRequest_FieldViolation{
			Field:       "email",
			Description: "Email must be a valid email address",
		})
	}

	if req.Age < 0 || req.Age > 150 {
		violations = append(violations, &errdetails.BadRequest_FieldViolation{
			Field:       "age",
			Description: "Age must be between 0 and 150",
		})
	}

	if len(violations) > 0 {
		st := status.New(codes.InvalidArgument, "validation failed")

		br := &errdetails.BadRequest{
			FieldViolations: violations,
		}

		st, err := st.WithDetails(br)
		if err != nil {
			return status.Error(codes.Internal, "failed to attach error details")
		}

		return st.Err()
	}

	return nil
}

// リトライ情報付きエラー
func rateLimitError() error {
	st := status.New(codes.ResourceExhausted, "rate limit exceeded")

	retryInfo := &errdetails.RetryInfo{
		RetryDelay: durationpb.New(30 * time.Second),
	}

	st, _ = st.WithDetails(retryInfo)
	return st.Err()
}

// デバッグ情報付きエラー
func internalErrorWithDebug(err error) error {
	st := status.New(codes.Internal, "internal server error")

	debugInfo := &errdetails.DebugInfo{
		StackEntries: []string{
			"github.com/example/service/handler.go:42",
			"github.com/example/service/main.go:15",
		},
		Detail: err.Error(),
	}

	st, _ = st.WithDetails(debugInfo)
	return st.Err()
}
```

---

## 9. インターセプター（ミドルウェア）

```go
// === Go でのインターセプター ===

// Unary サーバーインターセプター（ログ）
func loggingUnaryInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	start := time.Now()

	// メタデータの取得
	md, _ := metadata.FromIncomingContext(ctx)
	requestID := ""
	if ids := md.Get("x-request-id"); len(ids) > 0 {
		requestID = ids[0]
	}

	// ハンドラーの実行
	resp, err := handler(ctx, req)

	// ログ出力
	duration := time.Since(start)
	statusCode := codes.OK
	if err != nil {
		statusCode = status.Code(err)
	}

	log.Printf(
		"[%s] %s | %s | %v | %s",
		requestID,
		info.FullMethod,
		statusCode,
		duration,
		err,
	)

	return resp, err
}

// Unary サーバーインターセプター（認証）
func authUnaryInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	// ヘルスチェック等は認証スキップ
	if info.FullMethod == "/grpc.health.v1.Health/Check" {
		return handler(ctx, req)
	}

	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, status.Error(codes.Unauthenticated, "missing metadata")
	}

	tokens := md.Get("authorization")
	if len(tokens) == 0 {
		return nil, status.Error(codes.Unauthenticated, "missing token")
	}

	token := strings.TrimPrefix(tokens[0], "Bearer ")

	// トークン検証
	claims, err := validateToken(token)
	if err != nil {
		return nil, status.Errorf(codes.Unauthenticated, "invalid token: %v", err)
	}

	// コンテキストにユーザー情報を追加
	ctx = context.WithValue(ctx, "user_id", claims.UserID)
	ctx = context.WithValue(ctx, "user_roles", claims.Roles)

	return handler(ctx, req)
}

// Stream サーバーインターセプター（ログ）
func loggingStreamInterceptor(
	srv interface{},
	ss grpc.ServerStream,
	info *grpc.StreamServerInfo,
	handler grpc.StreamHandler,
) error {
	start := time.Now()

	err := handler(srv, ss)

	duration := time.Since(start)
	statusCode := codes.OK
	if err != nil {
		statusCode = status.Code(err)
	}

	log.Printf(
		"Stream %s | %s | %v",
		info.FullMethod,
		statusCode,
		duration,
	)

	return err
}

// リカバリーインターセプター（パニック回復）
func recoveryUnaryInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (resp interface{}, err error) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Panic recovered in %s: %v\n%s",
				info.FullMethod, r, debug.Stack())
			err = status.Errorf(codes.Internal, "internal error")
		}
	}()
	return handler(ctx, req)
}

// メトリクスインターセプター
func metricsUnaryInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	start := time.Now()

	resp, err := handler(ctx, req)

	duration := time.Since(start)
	statusCode := status.Code(err)

	// Prometheus メトリクス記録
	grpcRequestsTotal.WithLabelValues(
		info.FullMethod,
		statusCode.String(),
	).Inc()

	grpcRequestDuration.WithLabelValues(
		info.FullMethod,
	).Observe(duration.Seconds())

	return resp, err
}

// サーバー起動時にインターセプターを設定
func main() {
	s := grpc.NewServer(
		grpc.ChainUnaryInterceptor(
			recoveryUnaryInterceptor,   // 1番目: パニック回復
			loggingUnaryInterceptor,    // 2番目: ログ
			metricsUnaryInterceptor,    // 3番目: メトリクス
			authUnaryInterceptor,       // 4番目: 認証
		),
		grpc.ChainStreamInterceptor(
			loggingStreamInterceptor,
		),
	)
	// ...
}
```

```typescript
// === TypeScript でのインターセプター ===

// クライアント側インターセプター（リトライ）
import * as grpc from '@grpc/grpc-js';

function retryInterceptor(
  options: grpc.InterceptorOptions,
  nextCall: grpc.NextCall,
): grpc.InterceptingCall {
  const maxRetries = 3;
  const retryableStatuses = [
    grpc.status.UNAVAILABLE,
    grpc.status.DEADLINE_EXCEEDED,
    grpc.status.RESOURCE_EXHAUSTED,
  ];

  let retryCount = 0;
  let savedMetadata: grpc.Metadata;
  let savedMessage: any;
  let savedReceiveMessage: any;

  const requester = new grpc.RequesterBuilder()
    .withStart((metadata, listener, next) => {
      savedMetadata = metadata;
      const newListener = new grpc.ListenerBuilder()
        .withOnReceiveStatus((status, next) => {
          if (
            retryableStatuses.includes(status.code) &&
            retryCount < maxRetries
          ) {
            retryCount++;
            const delay = Math.pow(2, retryCount) * 100; // 指数バックオフ
            console.log(
              `Retrying (${retryCount}/${maxRetries}) after ${delay}ms`,
            );
            setTimeout(() => {
              // リトライ実行
              const newCall = nextCall(options);
              newCall.start(savedMetadata, listener);
              newCall.sendMessage(savedMessage);
              newCall.halfClose();
            }, delay);
          } else {
            next(status);
          }
        })
        .build();
      next(metadata, newListener);
    })
    .withSendMessage((message, next) => {
      savedMessage = message;
      next(message);
    })
    .build();

  return new grpc.InterceptingCall(nextCall(options), requester);
}

// クライアントにインターセプターを設定
const client = new proto.user.v1.UserService(
  'localhost:50051',
  grpc.credentials.createInsecure(),
  {
    interceptors: [retryInterceptor],
  },
);
```

---

## 10. gRPC-WebとConnect

```
gRPC-Web:
  → ブラウザから直接gRPCサーバーにアクセス
  → 制限: Unary と Server Streaming のみ
  → Client Streaming と Bidi Streaming は非対応
  → Envoy や gRPC-Web プロキシが必要

  ブラウザ ─── gRPC-Web ──→ Envoy Proxy ─── gRPC ──→ gRPCサーバー

  Envoy プロキシ設定:
  ┌─────────────────────────────────────────┐
  │ ブラウザ                                 │
  │ (gRPC-Web / HTTP/1.1 or HTTP/2)         │
  └────────────┬────────────────────────────┘
               ↓
  ┌─────────────────────────────────────────┐
  │ Envoy Proxy                              │
  │ - gRPC-Web ↔ gRPC 変換                  │
  │ - CORS ヘッダー付与                      │
  │ - TLS 終端                               │
  └────────────┬────────────────────────────┘
               ↓
  ┌─────────────────────────────────────────┐
  │ gRPC サーバー (HTTP/2)                   │
  └─────────────────────────────────────────┘

Connect Protocol（新しい選択肢）:
  → Buf社が開発したgRPC互換プロトコル
  → HTTP/1.1, HTTP/2, HTTP/3 に対応
  → プロキシ不要でブラウザから直接接続可能
  → gRPC, gRPC-Web, Connect の3プロトコルに互換
  → curl でテスト可能（JSONサポート）

  Connect の利点:
  ① プロキシ不要
  ② curl でデバッグ可能
  ③ ストリーミング対応（Server Streaming含む）
  ④ gRPCサーバーとの互換性
  ⑤ 既存のProtobuf定義をそのまま使用
```

```typescript
// === gRPC-Web クライアント（ブラウザ） ===
// @connectrpc/connect-web を使用

import { createConnectTransport } from '@connectrpc/connect-web';
import { createClient } from '@connectrpc/connect';
import { UserService } from './gen/user/v1/user_service_connect';

// トランスポート作成
const transport = createConnectTransport({
  baseUrl: 'https://api.example.com',
  // gRPC-Web プロトコル使用
  // useBinaryFormat: true,
});

// クライアント作成
const client = createClient(UserService, transport);

// Unary RPC
async function getUser(id: string) {
  try {
    const response = await client.getUser({ id });
    console.log('User:', response.user);
    return response.user;
  } catch (error) {
    if (error instanceof ConnectError) {
      console.error(`Error [${error.code}]: ${error.message}`);
      // エラー詳細の取得
      for (const detail of error.details) {
        console.error('Detail:', detail);
      }
    }
    throw error;
  }
}

// Server Streaming
async function watchUsers() {
  try {
    for await (const event of client.watchUserUpdates({
      eventTypes: ['created', 'updated'],
    })) {
      console.log(`Event: ${event.eventType}`, event.user);
      // UIの更新
      updateUserList(event);
    }
  } catch (error) {
    console.error('Stream error:', error);
  }
}
```

---

## 11. デッドラインとタイムアウト

```
デッドライン（Deadline）:
  → リクエストの絶対的な期限時刻
  → 「この時刻までにレスポンスが返らなければキャンセル」
  → gRPCでは「タイムアウト」ではなく「デッドライン」を使用

  重要: デッドラインはサービス間で伝播する

  クライアント                サービスA              サービスB
  デッドライン: 5秒          残り: 4.5秒            残り: 3秒
  ─────────────→            ─────────────→         ──────→
                             処理: 0.5秒             処理: 1秒
                             残り: 4.5秒             残り: 3秒
  ←─────────────            ←─────────────         ←──────

  デッドラインの伝播:
  → クライアントが5秒のデッドラインを設定
  → サービスAが受信時点で残り4.5秒
  → サービスBへのリクエストにも残り時間が伝播
  → どこかで期限切れ → DEADLINE_EXCEEDED エラー

推奨デッドライン値:
  ┌─────────────────────┬──────────┐
  │ 操作                 │ デッドライン│
  ├─────────────────────┼──────────┤
  │ 高速なルックアップ   │ 100ms    │
  │ 通常のCRUD           │ 1-5秒    │
  │ 検索・集計           │ 10-30秒  │
  │ バッチ処理           │ 60-300秒 │
  │ ファイルアップロード │ 300-600秒│
  └─────────────────────┴──────────┘

  デッドラインが切れた場合の動作:
  → サーバーは処理を中断すべき（リソース節約）
  → ctx.Err() でチェック
  → 既に完了した副作用のロールバックは考慮が必要
```

```go
// === デッドラインの実装（Go） ===

// クライアント側
func callWithDeadline(client pb.UserServiceClient) error {
	// 5秒のデッドラインを設定
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.GetUser(ctx, &pb.GetUserRequest{Id: "123"})
	if err != nil {
		st, ok := status.FromError(err)
		if ok && st.Code() == codes.DeadlineExceeded {
			log.Println("Request timed out!")
			return err
		}
		return err
	}

	log.Printf("User: %v", resp.User)
	return nil
}

// サーバー側（デッドラインチェック付き処理）
func (s *userServer) HeavyComputation(
	ctx context.Context,
	req *pb.HeavyRequest,
) (*pb.HeavyResponse, error) {
	results := make([]string, 0)

	for i := 0; i < 1000; i++ {
		// 定期的にデッドラインをチェック
		select {
		case <-ctx.Done():
			// デッドライン超過 or キャンセル
			return nil, status.Error(
				codes.DeadlineExceeded,
				"operation canceled due to deadline",
			)
		default:
			// 処理を続行
			result := processItem(i)
			results = append(results, result)
		}
	}

	return &pb.HeavyResponse{Results: results}, nil
}
```

---

## 12. ロードバランシングとサービスメッシュ

```
gRPC のロードバランシング:

  HTTP/1.1 の LB: 接続ごとに振り分け → gRPCには不向き
  gRPC の LB:     リクエスト（RPC）ごとに振り分け → L7 LB が必要

  ① クライアントサイドLB:
     → クライアントがサーバー一覧を把握
     → ラウンドロビン、重み付け等を自分で実行
     → サービスディスカバリと連携（DNS, Consul, etcd）

     クライアント（LBロジック内蔵）
          ↓ ↓ ↓
     サーバー1  サーバー2  サーバー3

  ② プロキシLB（L7）:
     → Envoy, Nginx, HAProxy 等
     → HTTP/2 ストリーム単位で振り分け
     → ヘルスチェック、サーキットブレーカー機能

     クライアント → L7 LB → サーバー1/2/3

  ③ サービスメッシュ:
     → Istio, Linkerd 等
     → サイドカープロキシがLBを担当
     → mTLS, トレーシング, レート制限も統合

     ┌──────────────────────┐
     │ Pod                  │
     │ ┌────────┐ ┌───────┐│
     │ │ App    │→│ Envoy ││──→ 他のPod
     │ │(gRPC)  │ │sidecar││
     │ └────────┘ └───────┘│
     └──────────────────────┘

  Kubernetes での gRPC LB:
  → 標準の Service はL4（TCP）LB
  → gRPC には不適切（1接続に全RPC集中）
  → 解決策:
     ① Headless Service + クライアントサイドLB
     ② Istio / Linkerd（サービスメッシュ）
     ③ gRPC-aware Ingress（Envoy, Traefik）

Envoy での gRPC ロードバランシング設定:
  clusters:
  - name: grpc_backend
    type: STRICT_DNS
    lb_policy: ROUND_ROBIN
    http2_protocol_options: {}
    health_checks:
    - grpc_health_check: {}
      timeout: 5s
      interval: 10s
    load_assignment:
      cluster_name: grpc_backend
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: backend1
                port_value: 50051
        - endpoint:
            address:
              socket_address:
                address: backend2
                port_value: 50051
```

---

## 13. ヘルスチェックとリフレクション

```protobuf
// === gRPC Health Checking Protocol ===
// grpc.health.v1.Health サービス（標準仕様）

// 使用するproto:
// grpc/health/v1/health.proto （gRPCに同梱）

service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
  rpc Watch(HealthCheckRequest) returns (stream HealthCheckResponse);
}

message HealthCheckRequest {
  string service = 1;  // 空文字 = サーバー全体
}

message HealthCheckResponse {
  enum ServingStatus {
    UNKNOWN = 0;
    SERVING = 1;
    NOT_SERVING = 2;
    SERVICE_UNKNOWN = 3;
  }
  ServingStatus status = 1;
}
```

```go
// Go でのヘルスチェック実装
import (
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
)

func main() {
	s := grpc.NewServer()

	// ヘルスチェックサーバーの登録
	healthServer := health.NewServer()
	healthpb.RegisterHealthServer(s, healthServer)

	// サービスの状態を設定
	healthServer.SetServingStatus(
		"user.v1.UserService",
		healthpb.HealthCheckResponse_SERVING,
	)

	// メンテナンス時
	// healthServer.SetServingStatus(
	//   "user.v1.UserService",
	//   healthpb.HealthCheckResponse_NOT_SERVING,
	// )

	// gRPC Reflection（デバッグ用）
	// grpcurl 等のツールからサービス情報を取得可能
	reflection.Register(s)

	// サーバー起動...
}
```

```bash
# === grpcurl でのテスト ===

# サービス一覧の取得（Reflection が有効な場合）
grpcurl -plaintext localhost:50051 list

# サービスのメソッド一覧
grpcurl -plaintext localhost:50051 list user.v1.UserService

# メソッドの詳細
grpcurl -plaintext localhost:50051 describe user.v1.UserService.GetUser

# Unary RPC の呼び出し
grpcurl -plaintext \
  -d '{"id": "123"}' \
  localhost:50051 user.v1.UserService/GetUser

# メタデータ付きの呼び出し
grpcurl -plaintext \
  -H 'authorization: Bearer my-token' \
  -H 'x-request-id: test-001' \
  -d '{"name": "Taro", "email": "taro@example.com"}' \
  localhost:50051 user.v1.UserService/CreateUser

# ヘルスチェック
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check

# 特定サービスのヘルスチェック
grpcurl -plaintext \
  -d '{"service": "user.v1.UserService"}' \
  localhost:50051 grpc.health.v1.Health/Check

# Server Streaming の呼び出し
grpcurl -plaintext \
  -d '{"page_size": 10}' \
  localhost:50051 user.v1.UserService/ListUsers

# Proto ファイルを指定して呼び出し（Reflection なし）
grpcurl -plaintext \
  -import-path ./proto \
  -proto user/v1/user_service.proto \
  -d '{"id": "123"}' \
  localhost:50051 user.v1.UserService/GetUser
```

---

## 14. セキュリティ

```
gRPC のセキュリティ:

  ① TLS / mTLS:
     → 通信の暗号化（TLS）
     → 相互認証（mTLS）: クライアントもサーバーも証明書を持つ
     → マイクロサービス間ではmTLSが推奨

  ② トークン認証:
     → Authorization ヘッダーに JWT を設定
     → メタデータとして送信
     → インターセプターで検証

  ③ API キー:
     → メタデータにAPI Keyを設定
     → 主にサービス間認証に使用

  mTLS の構成:
  ┌──────────┐    TLS    ┌──────────┐
  │ Client   │ ←──────→  │ Server   │
  │ cert.pem │    双方    │ cert.pem │
  │ key.pem  │   検証    │ key.pem  │
  └──────────┘           └──────────┘
       ↑                      ↑
       └──── CA証明書で検証 ────┘
```

```go
// === TLS設定（Go） ===

// サーバー側（TLS）
func startTLSServer() {
	creds, err := credentials.NewServerTLSFromFile(
		"server-cert.pem",
		"server-key.pem",
	)
	if err != nil {
		log.Fatalf("failed to load TLS: %v", err)
	}

	s := grpc.NewServer(grpc.Creds(creds))
	// サービス登録...
}

// サーバー側（mTLS）
func startMTLSServer() {
	cert, err := tls.LoadX509KeyPair("server-cert.pem", "server-key.pem")
	if err != nil {
		log.Fatalf("failed to load server cert: %v", err)
	}

	certPool := x509.NewCertPool()
	ca, err := os.ReadFile("ca-cert.pem")
	if err != nil {
		log.Fatalf("failed to load CA cert: %v", err)
	}
	certPool.AppendCertsFromPEM(ca)

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    certPool,
	}

	creds := credentials.NewTLS(tlsConfig)
	s := grpc.NewServer(grpc.Creds(creds))
	// サービス登録...
}

// クライアント側（mTLS）
func createMTLSClient() pb.UserServiceClient {
	cert, _ := tls.LoadX509KeyPair("client-cert.pem", "client-key.pem")

	certPool := x509.NewCertPool()
	ca, _ := os.ReadFile("ca-cert.pem")
	certPool.AppendCertsFromPEM(ca)

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		RootCAs:      certPool,
	}

	creds := credentials.NewTLS(tlsConfig)
	conn, _ := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(creds))
	return pb.NewUserServiceClient(conn)
}
```

---

## 15. パフォーマンス最適化

```
gRPC パフォーマンスのチューニング:

  ① メッセージサイズ:
     → デフォルト最大: 4MB（送受信とも）
     → 大きなメッセージ: サイズ上限を引き上げ
     → 超大きなデータ: ストリーミングに分割

  ② Keepalive:
     → 接続を維持してハンドシェイクコストを削減
     → クライアント/サーバー双方で設定
     → ロードバランサーのアイドルタイムアウトと整合

  ③ 接続プーリング:
     → 1接続で多重化されるが、CPU負荷が高い場合は複数接続
     → チャネル（接続）あたりの同時ストリーム数に注意
     → HTTP/2のデフォルト同時ストリーム: 100

  ④ コンプレッション:
     → gzip 圧縮でメッセージサイズを削減
     → CPU とネットワーク帯域のトレードオフ
     → テキスト多めのメッセージに効果的

  ⑤ バッチ処理:
     → 細かいRPCを大量に送るより、バッチでまとめて送る
     → Client Streaming で連続送信
     → repeated フィールドでバッチリクエスト

パフォーマンス比較（実測値の目安）:
  ┌──────────────────────┬───────────┬───────────┐
  │ メトリクス           │ REST/JSON │ gRPC      │
  ├──────────────────────┼───────────┼───────────┤
  │ シリアライズ速度     │ 1x        │ 5-10x     │
  │ メッセージサイズ     │ 1x        │ 0.3-0.5x  │
  │ レイテンシ           │ 1x        │ 0.5-0.7x  │
  │ スループット         │ 1x        │ 2-5x      │
  │ CPU使用率            │ 1x        │ 0.5-0.8x  │
  └──────────────────────┴───────────┴───────────┘
```

```go
// === 接続プーリングとコンプレッション（Go） ===

import "google.golang.org/grpc/encoding/gzip"

// gzip 圧縮を有効にしたクライアント
func createCompressedClient() pb.UserServiceClient {
	conn, _ := grpc.Dial(
		"localhost:50051",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.UseCompressor(gzip.Name),
		),
	)
	return pb.NewUserServiceClient(conn)
}

// 接続プール（複数接続の管理）
type ClientPool struct {
	clients []pb.UserServiceClient
	conns   []*grpc.ClientConn
	mu      sync.Mutex
	next    int
}

func NewClientPool(address string, size int) (*ClientPool, error) {
	pool := &ClientPool{
		clients: make([]pb.UserServiceClient, size),
		conns:   make([]*grpc.ClientConn, size),
	}

	for i := 0; i < size; i++ {
		conn, err := grpc.Dial(address,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		)
		if err != nil {
			pool.Close()
			return nil, err
		}
		pool.conns[i] = conn
		pool.clients[i] = pb.NewUserServiceClient(conn)
	}

	return pool, nil
}

func (p *ClientPool) GetClient() pb.UserServiceClient {
	p.mu.Lock()
	defer p.mu.Unlock()

	client := p.clients[p.next]
	p.next = (p.next + 1) % len(p.clients)
	return client
}

func (p *ClientPool) Close() {
	for _, conn := range p.conns {
		if conn != nil {
			conn.Close()
		}
	}
}
```

---

## 16. テスト

```go
// === gRPC サーバーのテスト（Go） ===

import (
	"testing"
	"google.golang.org/grpc/test/bufconn"
)

const bufSize = 1024 * 1024

func setupTestServer(t *testing.T) (pb.UserServiceClient, func()) {
	lis := bufconn.Listen(bufSize)

	s := grpc.NewServer()
	pb.RegisterUserServiceServer(s, newUserServer())

	go func() {
		if err := s.Serve(lis); err != nil {
			t.Fatalf("server exited with error: %v", err)
		}
	}()

	// bufconn を使ってインメモリ接続
	conn, err := grpc.DialContext(
		context.Background(),
		"bufnet",
		grpc.WithContextDialer(func(ctx context.Context, s string) (net.Conn, error) {
			return lis.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}

	client := pb.NewUserServiceClient(conn)

	cleanup := func() {
		conn.Close()
		s.Stop()
	}

	return client, cleanup
}

func TestGetUser(t *testing.T) {
	client, cleanup := setupTestServer(t)
	defer cleanup()

	ctx := context.Background()

	// まずユーザーを作成
	createResp, err := client.CreateUser(ctx, &pb.CreateUserRequest{
		Name:  "Test User",
		Email: "test@example.com",
		Age:   25,
	})
	if err != nil {
		t.Fatalf("CreateUser failed: %v", err)
	}

	// 作成したユーザーを取得
	getResp, err := client.GetUser(ctx, &pb.GetUserRequest{
		Id: createResp.User.Id,
	})
	if err != nil {
		t.Fatalf("GetUser failed: %v", err)
	}

	if getResp.User.Name != "Test User" {
		t.Errorf("expected name 'Test User', got '%s'", getResp.User.Name)
	}
}

func TestGetUser_NotFound(t *testing.T) {
	client, cleanup := setupTestServer(t)
	defer cleanup()

	_, err := client.GetUser(context.Background(), &pb.GetUserRequest{
		Id: "nonexistent-id",
	})

	if err == nil {
		t.Fatal("expected error, got nil")
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatal("expected gRPC status error")
	}

	if st.Code() != codes.NotFound {
		t.Errorf("expected NOT_FOUND, got %v", st.Code())
	}
}

func TestListUsers_Streaming(t *testing.T) {
	client, cleanup := setupTestServer(t)
	defer cleanup()

	ctx := context.Background()

	// テストデータ作成
	for i := 0; i < 5; i++ {
		client.CreateUser(ctx, &pb.CreateUserRequest{
			Name:  fmt.Sprintf("User %d", i),
			Email: fmt.Sprintf("user%d@example.com", i),
		})
	}

	// ストリーミングで取得
	stream, err := client.ListUsers(ctx, &pb.ListUsersRequest{
		PageSize: 10,
	})
	if err != nil {
		t.Fatalf("ListUsers failed: %v", err)
	}

	var users []*pb.User
	for {
		user, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream recv error: %v", err)
		}
		users = append(users, user)
	}

	if len(users) != 5 {
		t.Errorf("expected 5 users, got %d", len(users))
	}
}
```

---

## 17. gRPCの採用基準

```
gRPCが適している場面:
  ✓ マイクロサービス間通信（内部API）
  ✓ 低レイテンシが重要
  ✓ 型安全性が必須
  ✓ ストリーミングが必要
  ✓ 多言語環境（コード生成で統一）
  ✓ 高スループットが要求される
  ✓ Protocol定義をIDLとして管理したい
  ✓ バイナリデータの効率的な転送

RESTが適している場面:
  ✓ パブリックAPI
  ✓ ブラウザから直接アクセス
  ✓ シンプルなCRUD
  ✓ 可読性・デバッグのしやすさ重視
  ✓ キャッシュの活用（CDN等）
  ✓ サードパーティとの連携
  ✓ ドキュメントの公開

GraphQLが適している場面:
  ✓ フロントエンドが柔軟にデータを取得したい
  ✓ 複数リソースを1リクエストで取得
  ✓ クライアント主導のデータ取得パターン

ハイブリッドアーキテクチャ（推奨パターン）:
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
              ┌─────────┐
              │Inventory│
              │ Service │
              └─────────┘

  実際の企業での採用パターン:
  ┌──────────┬──────────────┬──────────────────────┐
  │ レイヤー  │ プロトコル    │ 理由                  │
  ├──────────┼──────────────┼──────────────────────┤
  │ 外部API   │ REST         │ 汎用性、ドキュメント  │
  │ BFF      │ GraphQL      │ フロント最適化         │
  │ 内部通信  │ gRPC         │ 高速、型安全          │
  │ イベント  │ Kafka/NATS   │ 非同期、デカップリング│
  │ リアルタイム│ WebSocket   │ ブラウザ双方向通信    │
  └──────────┴──────────────┴──────────────────────┘

移行戦略（REST → gRPC）:
  Phase 1: 新サービスをgRPCで作成
  Phase 2: API GatewayでREST ↔ gRPC変換
  Phase 3: 内部通信を順次gRPCに移行
  Phase 4: パブリックAPIはRESTを維持

  注意点:
  → 一気に移行しない（段階的に）
  → RESTとgRPCの共存を前提に設計
  → Proto定義のバージョン管理を整備
  → CI/CDにProtoのlint/breakingチェックを組み込む
```

---

## 18. Buf（Proto管理ツール）

```yaml
# buf.yaml — Buf設定ファイル
version: v2
modules:
  - path: proto
    name: buf.build/example/user
lint:
  use:
    - DEFAULT
  except:
    - PACKAGE_VERSION_SUFFIX
breaking:
  use:
    - FILE
  except:
    - EXTENSION_NO_DELETE

# buf.gen.yaml — コード生成設定
version: v2
managed:
  enabled: true
  override:
    - file_option: go_package_prefix
      value: github.com/example/gen
plugins:
  - remote: buf.build/protocolbuffers/go
    out: gen/go
    opt:
      - paths=source_relative
  - remote: buf.build/grpc/go
    out: gen/go
    opt:
      - paths=source_relative
  - remote: buf.build/connectrpc/go
    out: gen/go
    opt:
      - paths=source_relative
  - remote: buf.build/connectrpc/es
    out: gen/ts
```

```bash
# Buf の主要コマンド

# Proto ファイルの lint チェック
buf lint

# 破壊的変更の検出
buf breaking --against '.git#branch=main'

# コード生成
buf generate

# Proto ファイルのフォーマット
buf format -w

# 依存関係の更新
buf dep update

# BSR（Buf Schema Registry）へのプッシュ
buf push
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| gRPC | HTTP/2 + Protobufの高性能RPCフレームワーク |
| Protocol Buffers | バイナリシリアライズ、JSONの1/3サイズ |
| ストリーミング | Unary, Server, Client, Bidirectional の4種 |
| インターセプター | ログ、認証、メトリクス等のミドルウェア |
| エラーハンドリング | 16種類のステータスコード + Rich Error Model |
| デッドライン | タイムアウトの伝播によるカスケード防止 |
| セキュリティ | TLS/mTLS + トークン認証 |
| gRPC-Web/Connect | ブラウザからのgRPCアクセス |
| 用途 | マイクロサービス間通信に最適、外部はREST併用 |
| Buf | Proto管理、lint、破壊的変更検出 |

---

## 次に読むべきガイド
→ [[../02-http/00-http-basics.md]] — HTTP基礎

---

## 参考文献
1. gRPC Documentation. "Introduction to gRPC." grpc.io, 2024.
2. Google. "Protocol Buffers Language Guide." protobuf.dev, 2024.
3. gRPC. "gRPC Health Checking Protocol." github.com/grpc, 2024.
4. Buf. "Buf CLI Documentation." buf.build, 2024.
5. Connect. "Connect Protocol Specification." connectrpc.com, 2024.
6. CNCF. "gRPC in Cloud Native Architecture." cncf.io, 2024.
7. Google. "gRPC Error Handling." grpc.io/docs/guides/error, 2024.
8. Envoy Proxy. "gRPC Bridging." envoyproxy.io, 2024.
