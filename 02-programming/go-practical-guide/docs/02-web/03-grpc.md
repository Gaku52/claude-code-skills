# gRPC -- Protocol Buffers, サービス定義, ストリーミング

> gRPCはProtocol Buffersベースの高性能RPCフレームワークであり、型安全なサービス定義・双方向ストリーミング・gRPC-Gatewayで柔軟なAPI設計を実現する。

---

## この章で学ぶこと

1. **Protocol Buffers** -- サービス定義とコード生成
2. **4種類のRPCパターン** -- Unary/Server/Client/Bi-directional Streaming
3. **gRPC-Gateway** -- REST APIとの統合

---

### コード例 1: Proto定義

```protobuf
syntax = "proto3";
package user.v1;
option go_package = "gen/user/v1;userv1";

service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc ListUsers(ListUsersRequest) returns (stream User);
  rpc CreateUsers(stream CreateUserRequest) returns (CreateUsersResponse);
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

message User {
  int64 id = 1;
  string name = 2;
  string email = 3;
}

message GetUserRequest {
  int64 id = 1;
}

message GetUserResponse {
  User user = 1;
}
```

### コード例 2: gRPCサーバー実装

```go
type userServer struct {
    userv1.UnimplementedUserServiceServer
    users map[int64]*userv1.User
}

func (s *userServer) GetUser(ctx context.Context, req *userv1.GetUserRequest) (*userv1.GetUserResponse, error) {
    user, ok := s.users[req.Id]
    if !ok {
        return nil, status.Errorf(codes.NotFound, "user %d not found", req.Id)
    }
    return &userv1.GetUserResponse{User: user}, nil
}

func main() {
    lis, _ := net.Listen("tcp", ":50051")
    s := grpc.NewServer(
        grpc.UnaryInterceptor(loggingInterceptor),
    )
    userv1.RegisterUserServiceServer(s, &userServer{
        users: make(map[int64]*userv1.User),
    })
    s.Serve(lis)
}
```

### コード例 3: サーバーストリーミング

```go
func (s *userServer) ListUsers(req *userv1.ListUsersRequest, stream userv1.UserService_ListUsersServer) error {
    for _, user := range s.users {
        if err := stream.Send(user); err != nil {
            return err
        }
    }
    return nil
}

// クライアント側
func listUsers(client userv1.UserServiceClient) error {
    stream, err := client.ListUsers(context.Background(), &userv1.ListUsersRequest{})
    if err != nil {
        return err
    }
    for {
        user, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
        fmt.Printf("User: %v\n", user)
    }
    return nil
}
```

### コード例 4: インターセプタ (ミドルウェア)

```go
func loggingInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    start := time.Now()
    resp, err := handler(ctx, req)
    log.Printf("method=%s duration=%v err=%v",
        info.FullMethod, time.Since(start), err)
    return resp, err
}

// チェーンインターセプタ
s := grpc.NewServer(
    grpc.ChainUnaryInterceptor(
        loggingInterceptor,
        authInterceptor,
        recoveryInterceptor,
    ),
)
```

### コード例 5: gRPC-Gateway (REST変換)

```protobuf
import "google/api/annotations.proto";

service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse) {
    option (google.api.http) = {
      get: "/api/v1/users/{id}"
    };
  }
}
```

```go
func runGateway() error {
    ctx := context.Background()
    mux := runtime.NewServeMux()
    opts := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}

    err := userv1.RegisterUserServiceHandlerFromEndpoint(ctx, mux, "localhost:50051", opts)
    if err != nil {
        return err
    }
    return http.ListenAndServe(":8080", mux)
}
```

---

## 2. ASCII図解

### 図1: gRPC通信フロー

```
Client (Go)                    Server (Go)
┌──────────┐    HTTP/2 +      ┌──────────┐
│ Generated│    Protocol      │ Generated│
│ Stub     │    Buffers       │ Service  │
│          │ ────────────────>│          │
│ .proto   │    (バイナリ)     │ .proto   │
│ → Go code│ <────────────────│ → Go code│
└──────────┘                  └──────────┘

  protoc --go_out=. --go-grpc_out=. user.proto
```

### 図2: 4種類のRPCパターン

```
1. Unary RPC (1:1)
   Client ──[Request]──> Server
   Client <──[Response]── Server

2. Server Streaming (1:N)
   Client ──[Request]──> Server
   Client <──[Response1]── Server
   Client <──[Response2]── Server
   Client <──[Response3]── Server

3. Client Streaming (N:1)
   Client ──[Request1]──> Server
   Client ──[Request2]──> Server
   Client ──[Request3]──> Server
   Client <──[Response]── Server

4. Bidirectional Streaming (N:M)
   Client ──[Request1]──> Server
   Client <──[Response1]── Server
   Client ──[Request2]──> Server
   Client <──[Response2]── Server
```

### 図3: gRPC-Gateway アーキテクチャ

```
REST Client          gRPC-Gateway         gRPC Server
┌──────────┐    ┌──────────────┐    ┌──────────┐
│ Browser  │    │ HTTP/JSON    │    │ protobuf │
│ curl     │───>│  ↓ 変換 ↓    │───>│ Service  │
│ Postman  │    │ gRPC/protobuf│    │          │
│          │<───│  ↑ 変換 ↑    │<───│          │
└──────────┘    │ gRPC/protobuf│    └──────────┘
  :8080         │  ↓ 変換 ↓    │      :50051
                │ HTTP/JSON    │
                └──────────────┘

gRPC Client ─────────────────────> gRPC Server
  (直接接続も可能)                   :50051
```

---

## 3. 比較表

### 表1: gRPC vs REST

| 項目 | gRPC | REST (JSON) |
|------|------|-------------|
| プロトコル | HTTP/2 | HTTP/1.1 or HTTP/2 |
| シリアライズ | Protocol Buffers (バイナリ) | JSON (テキスト) |
| パフォーマンス | 非常に高速 | 中程度 |
| 型安全性 | 強い (.protoから生成) | 弱い (OpenAPIで補完) |
| ストリーミング | 双方向対応 | SSE/WebSocket |
| ブラウザ対応 | grpc-web必要 | ネイティブ |
| デバッグ | 専用ツール(grpcurl) | curl, Postman |
| エコシステム | 中 | 非常に大きい |

### 表2: gRPCステータスコード

| gRPC Code | HTTP相当 | 用途 |
|-----------|---------|------|
| OK | 200 | 成功 |
| NotFound | 404 | リソース不在 |
| InvalidArgument | 400 | バリデーションエラー |
| Unauthenticated | 401 | 認証エラー |
| PermissionDenied | 403 | 権限エラー |
| Internal | 500 | サーバーエラー |
| Unavailable | 503 | サービス利用不可 |
| DeadlineExceeded | 504 | タイムアウト |

---

## 4. アンチパターン

### アンチパターン 1: 巨大なメッセージ

```protobuf
// BAD: 1つのレスポンスに大量データ
message ListUsersResponse {
  repeated User users = 1;  // 100万件返す可能性
}

// GOOD: ページネーションまたはストリーミング
message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
}
```

### アンチパターン 2: エラー詳細を返さない

```go
// BAD: 汎用的なエラーメッセージ
return nil, status.Error(codes.Internal, "error")

// GOOD: 詳細なエラー情報を付与
st := status.New(codes.InvalidArgument, "validation failed")
st, _ = st.WithDetails(&errdetails.BadRequest{
    FieldViolations: []*errdetails.BadRequest_FieldViolation{
        {Field: "email", Description: "invalid email format"},
    },
})
return nil, st.Err()
```

---

## 5. FAQ

### Q1: gRPCはいつ選ぶべきか？

マイクロサービス間通信、低レイテンシが必要な内部API、ストリーミングが必要な場面で選ぶ。外部公開APIにはREST (またはgRPC-Gateway併用)が適切。

### Q2: Protocol Buffersのバージョン互換性は？

フィールド番号を変更しない限り後方互換。新フィールド追加は安全。フィールド削除は`reserved`で番号を予約する。これによりローリングアップデートが可能。

### Q3: gRPCのテストはどう書くか？

`bufconn`パッケージでインメモリ接続を作成し、実際のgRPCサーバーをテスト内で起動する。ネットワーク不要で高速にテスト可能。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Protocol Buffers | .protoからGo/他言語のコードを自動生成 |
| Unary RPC | 1リクエスト→1レスポンス |
| ストリーミング | Server/Client/Bidirectionalの3パターン |
| Interceptor | gRPC版ミドルウェア |
| Status Code | 独自コード体系 (codes.NotFound等) |
| gRPC-Gateway | REST API自動変換 |

---

## 次に読むべきガイド

- [04-testing.md](./04-testing.md) -- テスト
- [../03-tools/03-deployment.md](../03-tools/03-deployment.md) -- デプロイ
- [../01-concurrency/03-context.md](../01-concurrency/03-context.md) -- Context

---

## 参考文献

1. **gRPC Go** -- https://grpc.io/docs/languages/go/
2. **Protocol Buffers Language Guide** -- https://protobuf.dev/programming-guides/proto3/
3. **gRPC-Gateway** -- https://grpc-ecosystem.github.io/grpc-gateway/
