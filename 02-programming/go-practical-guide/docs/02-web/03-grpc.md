# gRPC -- Protocol Buffers, サービス定義, ストリーミング

> gRPCはProtocol Buffersベースの高性能RPCフレームワークであり、型安全なサービス定義・双方向ストリーミング・gRPC-Gatewayで柔軟なAPI設計を実現する。

---

## この章で学ぶこと

1. **Protocol Buffers** -- サービス定義とコード生成
2. **4種類のRPCパターン** -- Unary/Server/Client/Bi-directional Streaming
3. **gRPC-Gateway** -- REST APIとの統合
4. **インターセプタ** -- 認証、ロギング、リカバリ等のミドルウェア
5. **エラーハンドリング** -- ステータスコードとエラー詳細
6. **テスト** -- bufconn、モック、インテグレーションテスト
7. **パフォーマンス最適化** -- コネクション管理、ロードバランシング

---

## 1. Protocol Buffers 基礎

### コード例 1: Proto定義の設計

```protobuf
syntax = "proto3";
package user.v1;
option go_package = "gen/user/v1;userv1";

import "google/protobuf/timestamp.proto";
import "google/protobuf/field_mask.proto";
import "google/protobuf/empty.proto";

// UserService はユーザー管理サービス
service UserService {
  // Unary RPC: 単一ユーザー取得
  rpc GetUser(GetUserRequest) returns (GetUserResponse);

  // Unary RPC: ユーザー一覧取得（ページネーション付き）
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);

  // Unary RPC: ユーザー作成
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);

  // Unary RPC: ユーザー更新（部分更新対応）
  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);

  // Unary RPC: ユーザー削除
  rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty);

  // Server Streaming: ユーザーの変更をリアルタイム配信
  rpc WatchUsers(WatchUsersRequest) returns (stream UserEvent);

  // Client Streaming: バッチユーザー作成
  rpc BatchCreateUsers(stream CreateUserRequest) returns (BatchCreateUsersResponse);

  // Bidirectional Streaming: チャット
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

// User はユーザーメッセージ
message User {
  int64 id = 1;
  string name = 2;
  string email = 3;
  UserRole role = 4;
  UserProfile profile = 5;
  google.protobuf.Timestamp created_at = 6;
  google.protobuf.Timestamp updated_at = 7;
}

// UserRole はユーザーの役割
enum UserRole {
  USER_ROLE_UNSPECIFIED = 0;
  USER_ROLE_ADMIN = 1;
  USER_ROLE_MEMBER = 2;
  USER_ROLE_VIEWER = 3;
}

// UserProfile はユーザーのプロフィール情報
message UserProfile {
  string bio = 1;
  string avatar_url = 2;
  string location = 3;
  string website = 4;
}

// GetUserRequest はGetUserのリクエスト
message GetUserRequest {
  int64 id = 1;
}

// GetUserResponse はGetUserのレスポンス
message GetUserResponse {
  User user = 1;
}

// ListUsersRequest はListUsersのリクエスト（ページネーション付き）
message ListUsersRequest {
  int32 page_size = 1;    // 最大100
  string page_token = 2;  // 次ページのトークン
  string filter = 3;      // フィルター条件（例: "role=admin"）
  string order_by = 4;    // ソート順（例: "name asc"）
}

// ListUsersResponse はListUsersのレスポンス
message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
  int32 total_count = 3;
}

// CreateUserRequest はCreateUserのリクエスト
message CreateUserRequest {
  string name = 1;
  string email = 2;
  UserRole role = 3;
  UserProfile profile = 4;
}

// CreateUserResponse はCreateUserのレスポンス
message CreateUserResponse {
  User user = 1;
}

// UpdateUserRequest はUpdateUserのリクエスト（部分更新対応）
message UpdateUserRequest {
  User user = 1;
  // 更新するフィールドを指定（部分更新）
  google.protobuf.FieldMask update_mask = 2;
}

// UpdateUserResponse はUpdateUserのレスポンス
message UpdateUserResponse {
  User user = 1;
}

// DeleteUserRequest はDeleteUserのリクエスト
message DeleteUserRequest {
  int64 id = 1;
}

// WatchUsersRequest はWatchUsersのリクエスト
message WatchUsersRequest {
  repeated int64 user_ids = 1; // 監視対象のユーザーID（空なら全員）
}

// UserEvent はユーザーの変更イベント
message UserEvent {
  EventType type = 1;
  User user = 2;
  google.protobuf.Timestamp occurred_at = 3;

  enum EventType {
    EVENT_TYPE_UNSPECIFIED = 0;
    EVENT_TYPE_CREATED = 1;
    EVENT_TYPE_UPDATED = 2;
    EVENT_TYPE_DELETED = 3;
  }
}

// BatchCreateUsersResponse はバッチ作成のレスポンス
message BatchCreateUsersResponse {
  int32 created_count = 1;
  repeated User users = 2;
}

// ChatMessage はチャットメッセージ
message ChatMessage {
  string sender_id = 1;
  string content = 2;
  google.protobuf.Timestamp sent_at = 3;
}
```

### コード例 2: Proto ファイルの構成とBuf設定

```yaml
# buf.yaml -- Bufの設定ファイル
version: v1
name: buf.build/myorg/myapi
breaking:
  use:
    - FILE
lint:
  use:
    - DEFAULT
  except:
    - PACKAGE_VERSION_SUFFIX
```

```yaml
# buf.gen.yaml -- コード生成設定
version: v1
managed:
  enabled: true
  go_package_prefix:
    default: github.com/myorg/myapp/gen
plugins:
  - plugin: buf.build/protocolbuffers/go
    out: gen
    opt: paths=source_relative
  - plugin: buf.build/grpc/go
    out: gen
    opt: paths=source_relative
  - plugin: buf.build/grpc-ecosystem/gateway
    out: gen
    opt:
      - paths=source_relative
      - generate_unbound_methods=true
  - plugin: buf.build/grpc-ecosystem/openapiv2
    out: gen
    opt:
      - allow_merge=true
      - merge_file_name=api
```

```bash
# プロジェクト構成
proto/
├── buf.yaml
├── buf.gen.yaml
├── buf.lock
└── user/
    └── v1/
        ├── user.proto
        └── user_service.proto

# コード生成コマンド
buf generate

# Lint チェック
buf lint

# 破壊的変更の検出
buf breaking --against '.git#branch=main'
```

---

## 2. gRPCサーバー実装

### コード例 3: サーバー実装（Unary RPC）

```go
package server

import (
    "context"
    "fmt"
    "log"
    "net"
    "sync"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
    "google.golang.org/protobuf/types/known/emptypb"
    "google.golang.org/protobuf/types/known/timestamppb"

    userv1 "github.com/myorg/myapp/gen/user/v1"
)

// userServer はUserServiceの実装
type userServer struct {
    userv1.UnimplementedUserServiceServer
    mu    sync.RWMutex
    users map[int64]*userv1.User
    nextID int64
}

// NewUserServer は新しいUserServerを作成する
func NewUserServer() *userServer {
    return &userServer{
        users:  make(map[int64]*userv1.User),
        nextID: 1,
    }
}

// GetUser はユーザーを取得する（Unary RPC）
func (s *userServer) GetUser(ctx context.Context, req *userv1.GetUserRequest) (*userv1.GetUserResponse, error) {
    // バリデーション
    if req.Id <= 0 {
        return nil, status.Errorf(codes.InvalidArgument, "invalid user id: %d", req.Id)
    }

    s.mu.RLock()
    defer s.mu.RUnlock()

    user, ok := s.users[req.Id]
    if !ok {
        return nil, status.Errorf(codes.NotFound, "user %d not found", req.Id)
    }

    return &userv1.GetUserResponse{User: user}, nil
}

// ListUsers はユーザー一覧を取得する（ページネーション付き）
func (s *userServer) ListUsers(ctx context.Context, req *userv1.ListUsersRequest) (*userv1.ListUsersResponse, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()

    pageSize := int(req.PageSize)
    if pageSize <= 0 || pageSize > 100 {
        pageSize = 20 // デフォルト
    }

    // 全ユーザーをIDでソート
    var allUsers []*userv1.User
    for _, u := range s.users {
        allUsers = append(allUsers, u)
    }

    // ページネーション処理（簡易版）
    start := 0
    if req.PageToken != "" {
        // トークンからオフセットを復元
        fmt.Sscanf(req.PageToken, "%d", &start)
    }

    end := start + pageSize
    if end > len(allUsers) {
        end = len(allUsers)
    }

    var nextPageToken string
    if end < len(allUsers) {
        nextPageToken = fmt.Sprintf("%d", end)
    }

    return &userv1.ListUsersResponse{
        Users:         allUsers[start:end],
        NextPageToken: nextPageToken,
        TotalCount:    int32(len(allUsers)),
    }, nil
}

// CreateUser はユーザーを作成する
func (s *userServer) CreateUser(ctx context.Context, req *userv1.CreateUserRequest) (*userv1.CreateUserResponse, error) {
    // バリデーション
    if req.Name == "" {
        return nil, status.Errorf(codes.InvalidArgument, "name is required")
    }
    if req.Email == "" {
        return nil, status.Errorf(codes.InvalidArgument, "email is required")
    }

    // メール重複チェック
    s.mu.Lock()
    defer s.mu.Unlock()

    for _, u := range s.users {
        if u.Email == req.Email {
            return nil, status.Errorf(codes.AlreadyExists, "email %s already exists", req.Email)
        }
    }

    now := timestamppb.Now()
    user := &userv1.User{
        Id:        s.nextID,
        Name:      req.Name,
        Email:     req.Email,
        Role:      req.Role,
        Profile:   req.Profile,
        CreatedAt: now,
        UpdatedAt: now,
    }
    s.users[s.nextID] = user
    s.nextID++

    return &userv1.CreateUserResponse{User: user}, nil
}

// UpdateUser はユーザーを更新する（FieldMask対応）
func (s *userServer) UpdateUser(ctx context.Context, req *userv1.UpdateUserRequest) (*userv1.UpdateUserResponse, error) {
    if req.User == nil || req.User.Id <= 0 {
        return nil, status.Error(codes.InvalidArgument, "user with valid id is required")
    }

    s.mu.Lock()
    defer s.mu.Unlock()

    existing, ok := s.users[req.User.Id]
    if !ok {
        return nil, status.Errorf(codes.NotFound, "user %d not found", req.User.Id)
    }

    // FieldMask による部分更新
    if req.UpdateMask != nil && len(req.UpdateMask.Paths) > 0 {
        for _, path := range req.UpdateMask.Paths {
            switch path {
            case "name":
                existing.Name = req.User.Name
            case "email":
                existing.Email = req.User.Email
            case "role":
                existing.Role = req.User.Role
            case "profile.bio":
                if existing.Profile == nil {
                    existing.Profile = &userv1.UserProfile{}
                }
                existing.Profile.Bio = req.User.Profile.GetBio()
            case "profile.avatar_url":
                if existing.Profile == nil {
                    existing.Profile = &userv1.UserProfile{}
                }
                existing.Profile.AvatarUrl = req.User.Profile.GetAvatarUrl()
            default:
                return nil, status.Errorf(codes.InvalidArgument, "unknown field: %s", path)
            }
        }
    } else {
        // FieldMask なしの場合は全フィールド更新
        existing.Name = req.User.Name
        existing.Email = req.User.Email
        existing.Role = req.User.Role
        existing.Profile = req.User.Profile
    }

    existing.UpdatedAt = timestamppb.Now()

    return &userv1.UpdateUserResponse{User: existing}, nil
}

// DeleteUser はユーザーを削除する
func (s *userServer) DeleteUser(ctx context.Context, req *userv1.DeleteUserRequest) (*emptypb.Empty, error) {
    if req.Id <= 0 {
        return nil, status.Errorf(codes.InvalidArgument, "invalid user id: %d", req.Id)
    }

    s.mu.Lock()
    defer s.mu.Unlock()

    if _, ok := s.users[req.Id]; !ok {
        return nil, status.Errorf(codes.NotFound, "user %d not found", req.Id)
    }

    delete(s.users, req.Id)
    return &emptypb.Empty{}, nil
}
```

### コード例 4: サーバーストリーミングRPC

```go
// WatchUsers はユーザーの変更をリアルタイム配信する（Server Streaming）
func (s *userServer) WatchUsers(req *userv1.WatchUsersRequest, stream userv1.UserService_WatchUsersServer) error {
    log.Printf("WatchUsers started for user IDs: %v", req.UserIds)

    // イベントチャネルを監視
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-stream.Context().Done():
            // クライアントが切断した場合
            log.Printf("WatchUsers: client disconnected")
            return nil
        case <-ticker.C:
            // 変更があればイベントを送信（実際にはイベントバスから取得）
            event := s.checkForChanges(req.UserIds)
            if event != nil {
                if err := stream.Send(event); err != nil {
                    return status.Errorf(codes.Internal, "failed to send event: %v", err)
                }
            }
        }
    }
}

func (s *userServer) checkForChanges(watchIDs []int64) *userv1.UserEvent {
    // 実際の実装ではイベントバスやCDCを使う
    return nil
}
```

### コード例 5: クライアントストリーミングRPC

```go
// BatchCreateUsers はバッチユーザー作成（Client Streaming）
func (s *userServer) BatchCreateUsers(stream userv1.UserService_BatchCreateUsersServer) error {
    var createdUsers []*userv1.User
    var count int32

    for {
        req, err := stream.Recv()
        if err != nil {
            if err.Error() == "EOF" {
                // クライアントが送信完了
                return stream.SendAndClose(&userv1.BatchCreateUsersResponse{
                    CreatedCount: count,
                    Users:        createdUsers,
                })
            }
            return status.Errorf(codes.Internal, "failed to receive: %v", err)
        }

        // 個別にユーザーを作成
        resp, err := s.CreateUser(stream.Context(), req)
        if err != nil {
            log.Printf("BatchCreateUsers: skip user %s: %v", req.Name, err)
            continue // エラーがあってもスキップして続行
        }

        createdUsers = append(createdUsers, resp.User)
        count++
    }
}
```

### コード例 6: 双方向ストリーミングRPC

```go
// Chat は双方向ストリーミングチャット
func (s *userServer) Chat(stream userv1.UserService_ChatServer) error {
    log.Println("Chat: new connection")

    for {
        msg, err := stream.Recv()
        if err != nil {
            if err.Error() == "EOF" {
                log.Println("Chat: client closed")
                return nil
            }
            return status.Errorf(codes.Internal, "failed to receive: %v", err)
        }

        log.Printf("Chat: received from %s: %s", msg.SenderId, msg.Content)

        // エコー応答（実際にはブロードキャストなど）
        reply := &userv1.ChatMessage{
            SenderId: "server",
            Content:  fmt.Sprintf("Echo: %s", msg.Content),
            SentAt:   timestamppb.Now(),
        }

        if err := stream.Send(reply); err != nil {
            return status.Errorf(codes.Internal, "failed to send: %v", err)
        }
    }
}
```

---

## 3. gRPCクライアント実装

### コード例 7: クライアントの接続と呼び出し

```go
package client

import (
    "context"
    "crypto/tls"
    "fmt"
    "io"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials"
    "google.golang.org/grpc/credentials/insecure"
    "google.golang.org/grpc/keepalive"
    "google.golang.org/grpc/metadata"

    userv1 "github.com/myorg/myapp/gen/user/v1"
)

// UserClient はgRPCクライアントのラッパー
type UserClient struct {
    conn   *grpc.ClientConn
    client userv1.UserServiceClient
}

// NewUserClient は新しいgRPCクライアントを作成する
func NewUserClient(addr string, opts ...grpc.DialOption) (*UserClient, error) {
    // デフォルトオプション
    defaultOpts := []grpc.DialOption{
        grpc.WithKeepaliveParams(keepalive.ClientParameters{
            Time:                10 * time.Second, // KeepAlive ping間隔
            Timeout:             3 * time.Second,  // Ping応答のタイムアウト
            PermitWithoutStream: false,             // ストリームがない場合pingしない
        }),
        grpc.WithDefaultCallOptions(
            grpc.MaxCallRecvMsgSize(10 * 1024 * 1024), // 10MB
            grpc.MaxCallSendMsgSize(10 * 1024 * 1024), // 10MB
        ),
    }

    allOpts := append(defaultOpts, opts...)

    conn, err := grpc.NewClient(addr, allOpts...)
    if err != nil {
        return nil, fmt.Errorf("grpc dial: %w", err)
    }

    return &UserClient{
        conn:   conn,
        client: userv1.NewUserServiceClient(conn),
    }, nil
}

// NewInsecureUserClient はTLSなしのクライアントを作成する（開発用）
func NewInsecureUserClient(addr string) (*UserClient, error) {
    return NewUserClient(addr,
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    )
}

// NewSecureUserClient はTLS付きのクライアントを作成する（本番用）
func NewSecureUserClient(addr string) (*UserClient, error) {
    tlsConfig := &tls.Config{
        MinVersion: tls.VersionTLS13,
    }
    return NewUserClient(addr,
        grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)),
    )
}

// Close は接続を閉じる
func (c *UserClient) Close() error {
    return c.conn.Close()
}

// GetUser はUnary RPCでユーザーを取得する
func (c *UserClient) GetUser(ctx context.Context, id int64) (*userv1.User, error) {
    // タイムアウト設定
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    resp, err := c.client.GetUser(ctx, &userv1.GetUserRequest{Id: id})
    if err != nil {
        return nil, fmt.Errorf("GetUser: %w", err)
    }
    return resp.User, nil
}

// ListAllUsers はページネーションで全ユーザーを取得する
func (c *UserClient) ListAllUsers(ctx context.Context) ([]*userv1.User, error) {
    var allUsers []*userv1.User
    pageToken := ""

    for {
        resp, err := c.client.ListUsers(ctx, &userv1.ListUsersRequest{
            PageSize:  100,
            PageToken: pageToken,
        })
        if err != nil {
            return nil, fmt.Errorf("ListUsers: %w", err)
        }

        allUsers = append(allUsers, resp.Users...)

        if resp.NextPageToken == "" {
            break
        }
        pageToken = resp.NextPageToken
    }

    return allUsers, nil
}

// WatchUsers はServer Streamingでユーザーイベントを受信する
func (c *UserClient) WatchUsers(ctx context.Context, userIDs []int64, handler func(*userv1.UserEvent)) error {
    stream, err := c.client.WatchUsers(ctx, &userv1.WatchUsersRequest{
        UserIds: userIDs,
    })
    if err != nil {
        return fmt.Errorf("WatchUsers: %w", err)
    }

    for {
        event, err := stream.Recv()
        if err == io.EOF {
            return nil
        }
        if err != nil {
            return fmt.Errorf("WatchUsers recv: %w", err)
        }
        handler(event)
    }
}

// CreateWithMetadata はメタデータ付きでユーザーを作成する
func (c *UserClient) CreateWithMetadata(ctx context.Context, name, email, token string) (*userv1.User, error) {
    // メタデータ（HTTPヘッダーに相当）を付与
    md := metadata.Pairs(
        "authorization", "Bearer "+token,
        "x-request-id", generateRequestID(),
    )
    ctx = metadata.NewOutgoingContext(ctx, md)

    // レスポンスヘッダーとトレーラーを受信
    var header, trailer metadata.MD

    resp, err := c.client.CreateUser(ctx,
        &userv1.CreateUserRequest{
            Name:  name,
            Email: email,
            Role:  userv1.UserRole_USER_ROLE_MEMBER,
        },
        grpc.Header(&header),
        grpc.Trailer(&trailer),
    )
    if err != nil {
        return nil, err
    }

    // レスポンスヘッダーからレート制限情報を取得
    if remaining := header.Get("x-ratelimit-remaining"); len(remaining) > 0 {
        log.Printf("Rate limit remaining: %s", remaining[0])
    }

    return resp.User, nil
}

func generateRequestID() string {
    return fmt.Sprintf("req-%d", time.Now().UnixNano())
}
```

---

## 4. インターセプタ（ミドルウェア）

### コード例 8: Unaryインターセプタ

```go
package interceptor

import (
    "context"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/metadata"
    "google.golang.org/grpc/status"
)

// LoggingUnaryInterceptor はリクエストのロギングを行う
func LoggingUnaryInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    start := time.Now()

    // メタデータからリクエストIDを取得
    requestID := "unknown"
    if md, ok := metadata.FromIncomingContext(ctx); ok {
        if ids := md.Get("x-request-id"); len(ids) > 0 {
            requestID = ids[0]
        }
    }

    // ハンドラ実行
    resp, err := handler(ctx, req)

    // ログ出力
    duration := time.Since(start)
    code := codes.OK
    if err != nil {
        code = status.Code(err)
    }

    log.Printf("[gRPC] method=%s request_id=%s code=%s duration=%v",
        info.FullMethod, requestID, code, duration)

    return resp, err
}

// AuthUnaryInterceptor は認証を行う
func AuthUnaryInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    // ヘルスチェックなど認証不要なメソッドをスキップ
    skipMethods := map[string]bool{
        "/grpc.health.v1.Health/Check": true,
        "/grpc.reflection.v1.ServerReflection/ServerReflectionInfo": true,
    }
    if skipMethods[info.FullMethod] {
        return handler(ctx, req)
    }

    // メタデータからトークンを取得
    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return nil, status.Error(codes.Unauthenticated, "missing metadata")
    }

    tokens := md.Get("authorization")
    if len(tokens) == 0 {
        return nil, status.Error(codes.Unauthenticated, "missing authorization token")
    }

    // トークンの検証
    userID, err := validateToken(tokens[0])
    if err != nil {
        return nil, status.Errorf(codes.Unauthenticated, "invalid token: %v", err)
    }

    // ユーザーIDをコンテキストに格納
    ctx = context.WithValue(ctx, userIDKey{}, userID)

    return handler(ctx, req)
}

type userIDKey struct{}

func UserIDFromContext(ctx context.Context) (string, bool) {
    id, ok := ctx.Value(userIDKey{}).(string)
    return id, ok
}

func validateToken(token string) (string, error) {
    // 実際にはJWT検証などを行う
    if token == "" {
        return "", fmt.Errorf("empty token")
    }
    return "user-123", nil
}

// RecoveryUnaryInterceptor はパニックから回復する
func RecoveryUnaryInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (resp interface{}, err error) {
    defer func() {
        if r := recover(); r != nil {
            log.Printf("[PANIC] method=%s panic=%v", info.FullMethod, r)
            err = status.Errorf(codes.Internal, "internal server error")
        }
    }()
    return handler(ctx, req)
}

// RateLimitUnaryInterceptor はレート制限を行う
func RateLimitUnaryInterceptor(limiter *RateLimiter) grpc.UnaryServerInterceptor {
    return func(
        ctx context.Context,
        req interface{},
        info *grpc.UnaryServerInfo,
        handler grpc.UnaryHandler,
    ) (interface{}, error) {
        if !limiter.Allow() {
            return nil, status.Error(codes.ResourceExhausted, "rate limit exceeded")
        }
        return handler(ctx, req)
    }
}

// ValidationUnaryInterceptor はリクエストのバリデーションを行う
func ValidationUnaryInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    if v, ok := req.(interface{ Validate() error }); ok {
        if err := v.Validate(); err != nil {
            return nil, status.Errorf(codes.InvalidArgument, "validation failed: %v", err)
        }
    }
    return handler(ctx, req)
}

// TimeoutUnaryInterceptor はデフォルトタイムアウトを設定する
func TimeoutUnaryInterceptor(defaultTimeout time.Duration) grpc.UnaryServerInterceptor {
    return func(
        ctx context.Context,
        req interface{},
        info *grpc.UnaryServerInfo,
        handler grpc.UnaryHandler,
    ) (interface{}, error) {
        if _, ok := ctx.Deadline(); !ok {
            var cancel context.CancelFunc
            ctx, cancel = context.WithTimeout(ctx, defaultTimeout)
            defer cancel()
        }
        return handler(ctx, req)
    }
}
```

### コード例 9: Stream インターセプタ

```go
package interceptor

import (
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// LoggingStreamInterceptor はストリームのロギングを行う
func LoggingStreamInterceptor(
    srv interface{},
    ss grpc.ServerStream,
    info *grpc.StreamServerInfo,
    handler grpc.StreamHandler,
) error {
    start := time.Now()
    log.Printf("[gRPC Stream] method=%s started", info.FullMethod)

    err := handler(srv, ss)

    duration := time.Since(start)
    code := codes.OK
    if err != nil {
        code = status.Code(err)
    }

    log.Printf("[gRPC Stream] method=%s code=%s duration=%v",
        info.FullMethod, code, duration)

    return err
}

// RecoveryStreamInterceptor はストリームのパニックから回復する
func RecoveryStreamInterceptor(
    srv interface{},
    ss grpc.ServerStream,
    info *grpc.StreamServerInfo,
    handler grpc.StreamHandler,
) (err error) {
    defer func() {
        if r := recover(); r != nil {
            log.Printf("[PANIC] stream method=%s panic=%v", info.FullMethod, r)
            err = status.Errorf(codes.Internal, "internal server error")
        }
    }()
    return handler(srv, ss)
}
```

### コード例 10: サーバー起動（インターセプタ統合）

```go
package main

import (
    "context"
    "log"
    "net"
    "os"
    "os/signal"
    "syscall"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/health"
    "google.golang.org/grpc/health/grpc_health_v1"
    "google.golang.org/grpc/reflection"
)

func main() {
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("failed to listen: %v", err)
    }

    // サーバーオプション
    s := grpc.NewServer(
        // Unary インターセプタチェーン（順番に実行）
        grpc.ChainUnaryInterceptor(
            RecoveryUnaryInterceptor,
            LoggingUnaryInterceptor,
            TimeoutUnaryInterceptor(30*time.Second),
            AuthUnaryInterceptor,
            ValidationUnaryInterceptor,
        ),
        // Stream インターセプタチェーン
        grpc.ChainStreamInterceptor(
            RecoveryStreamInterceptor,
            LoggingStreamInterceptor,
        ),
        // メッセージサイズ制限
        grpc.MaxRecvMsgSize(10 * 1024 * 1024), // 10MB
        grpc.MaxSendMsgSize(10 * 1024 * 1024), // 10MB
        // KeepAlive設定
        grpc.KeepaliveParams(keepalive.ServerParameters{
            MaxConnectionIdle:     15 * time.Minute,
            MaxConnectionAge:      30 * time.Minute,
            MaxConnectionAgeGrace: 5 * time.Second,
            Time:                  5 * time.Minute,
            Timeout:               1 * time.Second,
        }),
        grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
            MinTime:             5 * time.Second,
            PermitWithoutStream: false,
        }),
    )

    // サービス登録
    userv1.RegisterUserServiceServer(s, NewUserServer())

    // ヘルスチェック
    healthServer := health.NewServer()
    grpc_health_v1.RegisterHealthServer(s, healthServer)
    healthServer.SetServingStatus("user.v1.UserService", grpc_health_v1.HealthCheckResponse_SERVING)

    // リフレクション（開発環境用、grpcurl等で使用）
    reflection.Register(s)

    // Graceful Shutdown
    go func() {
        log.Printf("gRPC server listening on :50051")
        if err := s.Serve(lis); err != nil {
            log.Fatalf("failed to serve: %v", err)
        }
    }()

    // シグナル待ち
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit

    log.Println("Shutting down gRPC server...")
    healthServer.SetServingStatus("user.v1.UserService", grpc_health_v1.HealthCheckResponse_NOT_SERVING)

    // Graceful stop（進行中のRPCが完了するのを待つ）
    stopped := make(chan struct{})
    go func() {
        s.GracefulStop()
        close(stopped)
    }()

    // タイムアウト付き待機
    select {
    case <-stopped:
        log.Println("Server stopped gracefully")
    case <-time.After(10 * time.Second):
        log.Println("Force stopping server")
        s.Stop()
    }
}
```

---

## 5. gRPC-Gateway (REST変換)

### コード例 11: gRPC-Gateway定義

```protobuf
syntax = "proto3";
package user.v1;

import "google/api/annotations.proto";
import "google/api/field_behavior.proto";
import "protoc-gen-openapiv2/options/annotations.proto";

option (grpc.gateway.protoc_gen_openapiv2.options.openapiv2_swagger) = {
  info: {
    title: "User API";
    version: "1.0";
    description: "User management API";
  };
  schemes: HTTPS;
  consumes: "application/json";
  produces: "application/json";
  security_definitions: {
    security: {
      key: "BearerAuth";
      value: {
        type: TYPE_API_KEY;
        in: IN_HEADER;
        name: "Authorization";
      };
    };
  };
};

service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse) {
    option (google.api.http) = {
      get: "/api/v1/users/{id}"
    };
  }

  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse) {
    option (google.api.http) = {
      get: "/api/v1/users"
    };
  }

  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse) {
    option (google.api.http) = {
      post: "/api/v1/users"
      body: "*"
    };
  }

  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse) {
    option (google.api.http) = {
      patch: "/api/v1/users/{user.id}"
      body: "*"
    };
  }

  rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty) {
    option (google.api.http) = {
      delete: "/api/v1/users/{id}"
    };
  }
}
```

### コード例 12: gRPC-Gatewayサーバー

```go
package main

import (
    "context"
    "fmt"
    "log"
    "net"
    "net/http"
    "time"

    "github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    "google.golang.org/protobuf/encoding/protojson"

    userv1 "github.com/myorg/myapp/gen/user/v1"
)

func main() {
    ctx := context.Background()

    // gRPCサーバー起動
    go runGRPCServer()

    // gRPC-Gatewayの設定
    mux := runtime.NewServeMux(
        // JSON出力オプション
        runtime.WithMarshalerOption(runtime.MIMEWildcard, &runtime.JSONPb{
            MarshalOptions: protojson.MarshalOptions{
                UseProtoNames:   true,  // snake_case フィールド名
                EmitUnpopulated: false, // ゼロ値のフィールドを省略
            },
            UnmarshalOptions: protojson.UnmarshalOptions{
                DiscardUnknown: true, // 未知のフィールドを無視
            },
        }),
        // エラーハンドリングのカスタマイズ
        runtime.WithErrorHandler(customErrorHandler),
        // メタデータの転送
        runtime.WithMetadata(func(ctx context.Context, r *http.Request) metadata.MD {
            md := metadata.MD{}
            if auth := r.Header.Get("Authorization"); auth != "" {
                md.Set("authorization", auth)
            }
            if reqID := r.Header.Get("X-Request-ID"); reqID != "" {
                md.Set("x-request-id", reqID)
            }
            return md
        }),
    )

    // gRPCバックエンドへの接続
    opts := []grpc.DialOption{
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    }
    err := userv1.RegisterUserServiceHandlerFromEndpoint(ctx, mux, "localhost:50051", opts)
    if err != nil {
        log.Fatalf("failed to register gateway: %v", err)
    }

    // HTTPサーバー（CORS、ロギングミドルウェア付き）
    handler := corsMiddleware(loggingMiddleware(mux))

    log.Printf("gRPC-Gateway listening on :8080")
    if err := http.ListenAndServe(":8080", handler); err != nil {
        log.Fatalf("failed to serve: %v", err)
    }
}

// customErrorHandler はgRPCエラーをHTTPレスポンスに変換する
func customErrorHandler(
    ctx context.Context,
    mux *runtime.ServeMux,
    marshaler runtime.Marshaler,
    w http.ResponseWriter,
    r *http.Request,
    err error,
) {
    st := status.Convert(err)
    httpStatus := runtime.HTTPStatusFromCode(st.Code())

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(httpStatus)

    body := map[string]interface{}{
        "error": map[string]interface{}{
            "code":    int(st.Code()),
            "message": st.Message(),
            "status":  st.Code().String(),
        },
    }

    // エラー詳細がある場合
    for _, detail := range st.Details() {
        body["error"].(map[string]interface{})["details"] = detail
    }

    data, _ := json.Marshal(body)
    w.Write(data)
}

// corsMiddleware はCORSヘッダーを設定する
func corsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")

        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusNoContent)
            return
        }

        next.ServeHTTP(w, r)
    })
}

// loggingMiddleware はHTTPリクエストをログに記録する
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        log.Printf("[HTTP] %s %s %v", r.Method, r.URL.Path, time.Since(start))
    })
}
```

---

## 6. テスト

### コード例 13: bufconn を使ったテスト

```go
package server_test

import (
    "context"
    "log"
    "net"
    "testing"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/credentials/insecure"
    "google.golang.org/grpc/status"
    "google.golang.org/grpc/test/bufconn"

    userv1 "github.com/myorg/myapp/gen/user/v1"
)

const bufSize = 1024 * 1024

// setupTestServer はインメモリgRPCサーバーを起動する
func setupTestServer(t *testing.T) (userv1.UserServiceClient, func()) {
    t.Helper()

    lis := bufconn.Listen(bufSize)
    s := grpc.NewServer()
    userv1.RegisterUserServiceServer(s, NewUserServer())

    go func() {
        if err := s.Serve(lis); err != nil {
            log.Printf("server error: %v", err)
        }
    }()

    // bufconn用のダイアラー
    dialer := func(context.Context, string) (net.Conn, error) {
        return lis.Dial()
    }

    conn, err := grpc.NewClient("passthrough:///bufnet",
        grpc.WithContextDialer(dialer),
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    )
    if err != nil {
        t.Fatalf("failed to dial bufnet: %v", err)
    }

    client := userv1.NewUserServiceClient(conn)
    cleanup := func() {
        conn.Close()
        s.Stop()
    }

    return client, cleanup
}

func TestGetUser_NotFound(t *testing.T) {
    client, cleanup := setupTestServer(t)
    defer cleanup()

    ctx := context.Background()
    _, err := client.GetUser(ctx, &userv1.GetUserRequest{Id: 999})
    if err == nil {
        t.Fatal("expected error, got nil")
    }

    st, ok := status.FromError(err)
    if !ok {
        t.Fatalf("expected gRPC status error, got: %v", err)
    }

    if st.Code() != codes.NotFound {
        t.Errorf("expected NotFound, got %v", st.Code())
    }
}

func TestCreateUser(t *testing.T) {
    client, cleanup := setupTestServer(t)
    defer cleanup()

    ctx := context.Background()
    resp, err := client.CreateUser(ctx, &userv1.CreateUserRequest{
        Name:  "Test User",
        Email: "test@example.com",
        Role:  userv1.UserRole_USER_ROLE_MEMBER,
    })
    if err != nil {
        t.Fatalf("CreateUser failed: %v", err)
    }

    if resp.User.Name != "Test User" {
        t.Errorf("Name = %q, want %q", resp.User.Name, "Test User")
    }
    if resp.User.Email != "test@example.com" {
        t.Errorf("Email = %q, want %q", resp.User.Email, "test@example.com")
    }
    if resp.User.Id <= 0 {
        t.Error("expected positive ID")
    }
}

func TestCreateUser_DuplicateEmail(t *testing.T) {
    client, cleanup := setupTestServer(t)
    defer cleanup()

    ctx := context.Background()

    // 1回目: 成功
    _, err := client.CreateUser(ctx, &userv1.CreateUserRequest{
        Name:  "User 1",
        Email: "dup@example.com",
    })
    if err != nil {
        t.Fatalf("first CreateUser failed: %v", err)
    }

    // 2回目: 重複エラー
    _, err = client.CreateUser(ctx, &userv1.CreateUserRequest{
        Name:  "User 2",
        Email: "dup@example.com",
    })
    if err == nil {
        t.Fatal("expected error for duplicate email")
    }

    st, _ := status.FromError(err)
    if st.Code() != codes.AlreadyExists {
        t.Errorf("expected AlreadyExists, got %v", st.Code())
    }
}

func TestListUsers_Pagination(t *testing.T) {
    client, cleanup := setupTestServer(t)
    defer cleanup()

    ctx := context.Background()

    // 5人のユーザーを作成
    for i := 0; i < 5; i++ {
        _, err := client.CreateUser(ctx, &userv1.CreateUserRequest{
            Name:  fmt.Sprintf("User %d", i),
            Email: fmt.Sprintf("user%d@example.com", i),
        })
        if err != nil {
            t.Fatalf("CreateUser %d failed: %v", i, err)
        }
    }

    // ページサイズ2で取得
    resp, err := client.ListUsers(ctx, &userv1.ListUsersRequest{PageSize: 2})
    if err != nil {
        t.Fatalf("ListUsers failed: %v", err)
    }

    if len(resp.Users) != 2 {
        t.Errorf("got %d users, want 2", len(resp.Users))
    }
    if resp.NextPageToken == "" {
        t.Error("expected next_page_token")
    }
    if resp.TotalCount != 5 {
        t.Errorf("total_count = %d, want 5", resp.TotalCount)
    }
}
```

---

## 7. エラーハンドリング

### コード例 14: 詳細なエラーレスポンス

```go
package server

import (
    "google.golang.org/genproto/googleapis/rpc/errdetails"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// バリデーションエラーを構築する
func validationError(violations map[string]string) error {
    st := status.New(codes.InvalidArgument, "validation failed")

    var fieldViolations []*errdetails.BadRequest_FieldViolation
    for field, desc := range violations {
        fieldViolations = append(fieldViolations, &errdetails.BadRequest_FieldViolation{
            Field:       field,
            Description: desc,
        })
    }

    detailed, err := st.WithDetails(&errdetails.BadRequest{
        FieldViolations: fieldViolations,
    })
    if err != nil {
        return st.Err()
    }
    return detailed.Err()
}

// リソース不存在エラー
func notFoundError(resourceType, resourceID string) error {
    st := status.New(codes.NotFound, fmt.Sprintf("%s not found", resourceType))
    detailed, err := st.WithDetails(&errdetails.ResourceInfo{
        ResourceType: resourceType,
        ResourceName: resourceID,
        Description:  fmt.Sprintf("%s with id %s was not found", resourceType, resourceID),
    })
    if err != nil {
        return st.Err()
    }
    return detailed.Err()
}

// レート制限エラー
func rateLimitError(retryAfter time.Duration) error {
    st := status.New(codes.ResourceExhausted, "rate limit exceeded")
    detailed, err := st.WithDetails(&errdetails.RetryInfo{
        RetryDelay: durationpb.New(retryAfter),
    })
    if err != nil {
        return st.Err()
    }
    return detailed.Err()
}

// クライアント側でのエラー詳細の取得
func handleGRPCError(err error) {
    st := status.Convert(err)

    log.Printf("Code: %s, Message: %s", st.Code(), st.Message())

    for _, detail := range st.Details() {
        switch d := detail.(type) {
        case *errdetails.BadRequest:
            for _, v := range d.FieldViolations {
                log.Printf("Field: %s, Description: %s", v.Field, v.Description)
            }
        case *errdetails.ResourceInfo:
            log.Printf("Resource: %s/%s", d.ResourceType, d.ResourceName)
        case *errdetails.RetryInfo:
            log.Printf("Retry after: %v", d.RetryDelay.AsDuration())
        }
    }
}
```

---

## 8. ASCII図解

### 図1: gRPC通信フロー

```
Client (Go)                    Server (Go)
┌──────────────────┐    HTTP/2 +      ┌──────────────────┐
│                  │    Protocol      │                  │
│ Generated Stub   │    Buffers       │ Generated Service│
│ (UserServiceClient)│ ──────────────>│(UserServiceServer)│
│                  │    (バイナリ)     │                  │
│ .proto → Go code │ <──────────────│ .proto → Go code │
│                  │                  │                  │
│ grpc.ClientConn  │    TLS +        │ grpc.Server      │
│                  │    HTTP/2        │                  │
└──────────────────┘    Multiplexed  └──────────────────┘
                        Streams

コード生成パイプライン:
  .proto ──> protoc / buf generate
                │
                ├── *.pb.go        (メッセージ型)
                ├── *_grpc.pb.go   (サービスインターフェース)
                ├── *.pb.gw.go     (gRPC-Gateway)
                └── *.swagger.json (OpenAPI仕様)
```

### 図2: 4種類のRPCパターン

```
1. Unary RPC (1:1) -- GetUser, CreateUser
   Client ──[Request]──> Server
   Client <──[Response]── Server
   最も基本的。REST APIの代替。

2. Server Streaming (1:N) -- WatchUsers
   Client ──[Request]──────> Server
   Client <──[Response 1]── Server
   Client <──[Response 2]── Server
   Client <──[Response 3]── Server
   Client <──[EOF]────────── Server
   リアルタイム通知、大量データの分割送信。

3. Client Streaming (N:1) -- BatchCreateUsers
   Client ──[Request 1]──> Server
   Client ──[Request 2]──> Server
   Client ──[Request 3]──> Server
   Client ──[EOF]────────> Server
   Client <──[Response]──── Server
   ファイルアップロード、バッチ処理。

4. Bidirectional Streaming (N:M) -- Chat
   Client ──[Request 1]──> Server
   Client <──[Response 1]── Server
   Client ──[Request 2]──> Server
   Client ──[Request 3]──> Server
   Client <──[Response 2]── Server
   Client <──[Response 3]── Server
   チャット、ゲーム、リアルタイムコラボ。
```

### 図3: gRPC-Gateway アーキテクチャ

```
                    ┌─────────────────────────────────────────┐
REST Client         │            gRPC-Gateway                 │     gRPC Server
┌──────────┐   HTTP │  ┌──────────────────────────────────┐  │ gRPC ┌──────────┐
│ Browser  │──────>│  │  HTTP/JSON → gRPC/Protobuf       │  │────>│ Service  │
│ curl     │       │  │  ┌──────┐    ┌───────┐           │  │     │ Handler  │
│ Postman  │       │  │  │Router│───>│Marshal│           │  │     │          │
│ Mobile   │       │  │  │Match │    │Convert│           │  │     │          │
│          │<──────│  │  └──────┘    └───────┘           │  │<────│          │
└──────────┘   HTTP │  │  gRPC/Protobuf → HTTP/JSON       │  │ gRPC └──────────┘
  :8080        JSON │  └──────────────────────────────────┘  │       :50051
                    └─────────────────────────────────────────┘

gRPC Client ────────────────────────────────────────────> gRPC Server
  (直接接続、最高性能)                                       :50051

RESTとgRPCの両方をサポート:
  /api/v1/users/{id}  →  UserService.GetUser()
  /api/v1/users       →  UserService.ListUsers()
  POST /api/v1/users  →  UserService.CreateUser()
```

### 図4: インターセプタチェーン

```
リクエスト
  │
  ▼
┌────────────────────────────────────────────────┐
│              Interceptor Chain                  │
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Recovery │─>│ Logging  │─>│ Timeout  │    │
│  │ panic回復│  │ ログ記録  │  │タイムアウト│    │
│  └──────────┘  └──────────┘  └──────────┘    │
│       │                                  │     │
│       ▼                                  ▼     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   Auth   │─>│Validation│─>│ RateLimit│    │
│  │ 認証     │  │入力検証   │  │レート制限 │    │
│  └──────────┘  └──────────┘  └──────────┘    │
│                                    │           │
└────────────────────────────────────┼───────────┘
                                     │
                                     ▼
                              ┌──────────┐
                              │  Handler │
                              │ 本体処理  │
                              └──────────┘
```

### 図5: gRPCヘルスチェックとロードバランシング

```
┌────────────────────────────────────────────────────────┐
│                  Load Balancer                          │
│  (Envoy / Nginx / Kubernetes Service)                  │
│                                                        │
│  ヘルスチェック:                                         │
│  grpc_health_v1.Health/Check → SERVING / NOT_SERVING   │
│                                                        │
│  ロードバランシング戦略:                                 │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐   │
│  │ Round Robin│  │ Least Conn   │  │ Weighted     │   │
│  │ 均等分散   │  │ 最小接続数    │  │ 重み付け      │   │
│  └────────────┘  └──────────────┘  └─────────────┘   │
└─────────┬──────────────┬──────────────┬───────────────┘
          │              │              │
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Server 1 │  │ Server 2 │  │ Server 3 │
    │ :50051   │  │ :50052   │  │ :50053   │
    │ SERVING  │  │ SERVING  │  │ SERVING  │
    └──────────┘  └──────────┘  └──────────┘
```

---

## 9. 比較表

### 表1: gRPC vs REST 詳細比較

| 項目 | gRPC | REST (JSON) |
|------|------|-------------|
| プロトコル | HTTP/2 | HTTP/1.1 or HTTP/2 |
| シリアライズ | Protocol Buffers (バイナリ) | JSON (テキスト) |
| パフォーマンス | 非常に高速（10-100倍） | 中程度 |
| 型安全性 | 強い (.protoから生成) | 弱い (OpenAPIで補完) |
| ストリーミング | 双方向対応 | SSE/WebSocket |
| ブラウザ対応 | grpc-web/Connect必要 | ネイティブ |
| デバッグ | grpcurl, grpcui | curl, Postman |
| エコシステム | 中 | 非常に大きい |
| コード生成 | 自動（proto） | 手動/OpenAPI |
| バージョニング | パッケージバージョン | URL/ヘッダー |
| エラー体系 | gRPC Status Code | HTTP Status Code |
| 推奨場面 | マイクロサービス間通信 | 外部公開API |

### 表2: gRPCステータスコード詳細

| gRPC Code | HTTP相当 | 用途 | 例 |
|-----------|---------|------|-----|
| OK (0) | 200 | 成功 | 正常完了 |
| Cancelled (1) | 499 | クライアントキャンセル | リクエスト中断 |
| Unknown (2) | 500 | 不明なエラー | 予期しない例外 |
| InvalidArgument (3) | 400 | バリデーションエラー | 不正なメール形式 |
| DeadlineExceeded (4) | 504 | タイムアウト | 処理時間超過 |
| NotFound (5) | 404 | リソース不在 | ユーザーが存在しない |
| AlreadyExists (6) | 409 | リソース重複 | メールアドレス重複 |
| PermissionDenied (7) | 403 | 権限エラー | 管理者権限が必要 |
| ResourceExhausted (8) | 429 | リソース枯渇 | レート制限超過 |
| FailedPrecondition (9) | 400 | 前提条件不一致 | ETAGミスマッチ |
| Aborted (10) | 409 | 操作の中止 | トランザクション競合 |
| OutOfRange (11) | 400 | 範囲外 | ページトークン無効 |
| Unimplemented (12) | 501 | 未実装 | メソッド未サポート |
| Internal (13) | 500 | サーバーエラー | 内部処理失敗 |
| Unavailable (14) | 503 | サービス利用不可 | メンテナンス中 |
| DataLoss (15) | 500 | データ損失 | データ破損検出 |
| Unauthenticated (16) | 401 | 認証エラー | トークン無効 |

### 表3: RPCパターン選択ガイド

| パターン | 用途 | メッセージ数 | 例 |
|---------|------|------------|-----|
| Unary | 基本的なリクエスト/レスポンス | 1:1 | CRUD操作、認証 |
| Server Streaming | サーバーから連続データ | 1:N | リアルタイム通知、大量データ取得 |
| Client Streaming | クライアントから連続データ | N:1 | ファイルアップロード、バッチ処理 |
| Bidirectional | 双方向のリアルタイム通信 | N:M | チャット、ゲーム、共同編集 |

### 表4: Protocol Buffers ベストプラクティス

| ルール | 説明 | 例 |
|--------|------|-----|
| フィールド番号の予約 | 削除したフィールドの番号をreserved | `reserved 3, 15;` |
| enum のデフォルト値 | 0番目はUNSPECIFIED | `ROLE_UNSPECIFIED = 0;` |
| パッケージバージョニング | パッケージ名にバージョン | `package user.v1;` |
| FieldMask | 部分更新に使用 | `update_mask` フィールド |
| ページネーション | page_size + page_token | `string next_page_token;` |
| Timestamp | 日時はgoogle.protobuf.Timestamp | `import "google/protobuf/timestamp.proto";` |

---

## 10. アンチパターン

### アンチパターン 1: 巨大なメッセージ

```protobuf
// BAD: 1つのレスポンスに大量データ
message ListUsersResponse {
  repeated User users = 1;  // 100万件返す可能性 → メモリ不足
}

// GOOD: ページネーションまたはストリーミング
// 方法1: ページネーション
message ListUsersRequest {
  int32 page_size = 1;     // 最大100
  string page_token = 2;   // カーソル
}
message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
}

// 方法2: Server Streaming（大量データ向け）
rpc StreamUsers(StreamUsersRequest) returns (stream User);
```

### アンチパターン 2: エラー詳細を返さない

```go
// BAD: 汎用的なエラーメッセージ
return nil, status.Error(codes.Internal, "error")
// → クライアントは何が問題かわからない

// GOOD: 詳細なエラー情報を付与
st := status.New(codes.InvalidArgument, "validation failed")
st, _ = st.WithDetails(&errdetails.BadRequest{
    FieldViolations: []*errdetails.BadRequest_FieldViolation{
        {Field: "email", Description: "invalid email format"},
        {Field: "name", Description: "name must be 1-100 characters"},
    },
})
return nil, st.Err()
```

### アンチパターン 3: Context を無視する

```go
// BAD: context を無視して長時間処理
func (s *server) SlowRPC(ctx context.Context, req *pb.Request) (*pb.Response, error) {
    result := heavyComputation() // ctx.Done() をチェックしない
    return &pb.Response{Data: result}, nil
}

// GOOD: context のキャンセルを定期的にチェック
func (s *server) SlowRPC(ctx context.Context, req *pb.Request) (*pb.Response, error) {
    resultCh := make(chan string, 1)
    go func() {
        resultCh <- heavyComputation()
    }()

    select {
    case result := <-resultCh:
        return &pb.Response{Data: result}, nil
    case <-ctx.Done():
        return nil, status.FromContextError(ctx.Err()).Err()
    }
}
```

### アンチパターン 4: フィールド番号の変更

```protobuf
// BAD: 既存フィールドの番号を変更（後方互換性が壊れる）
// Before:
message User {
  string name = 1;
  string email = 2;
}
// After (壊れる):
message User {
  string email = 1;  // 番号変更 → 既存クライアントが壊れる
  string name = 2;
}

// GOOD: 新フィールドは新番号で追加
message User {
  string name = 1;
  string email = 2;
  string phone = 3;  // 新規フィールドは次の番号
  reserved 4;        // 削除したフィールドの番号は予約
}
```

### アンチパターン 5: Unimplemented メソッドの放置

```go
// BAD: UnimplementedServerをそのまま埋め込むだけ
type myServer struct {
    pb.UnimplementedMyServiceServer
}
// 未実装のメソッドが呼ばれるとUnimplementedエラーが返る
// → 本番で気づかない

// GOOD: mustEmbedUnimplemented で未実装を検出
// もしくは全メソッドを明示的に実装し、未対応のものはエラーを返す
func (s *myServer) NotYetImplemented(ctx context.Context, req *pb.Request) (*pb.Response, error) {
    return nil, status.Error(codes.Unimplemented,
        "NotYetImplemented is not yet available, planned for v2.0")
}
```

---

## 11. FAQ

### Q1: gRPCはいつ選ぶべきか？

マイクロサービス間通信、低レイテンシが必要な内部API、ストリーミングが必要な場面で選ぶ。外部公開APIにはREST（またはgRPC-Gateway併用）が適切。

判断基準:
- **gRPC適**: マイクロサービス間、高スループット、型安全性重視、ストリーミング必要
- **REST適**: 外部公開API、ブラウザ直接通信、シンプルなCRUD、既存システム統合
- **両方**: gRPC-Gatewayで内部はgRPC、外部はRESTを提供

### Q2: Protocol Buffersのバージョン互換性は？

フィールド番号を変更しない限り後方互換。新フィールド追加は安全。フィールド削除は`reserved`で番号を予約する。これによりローリングアップデートが可能。

互換性ルール:
- フィールド追加: 安全（旧クライアントはデフォルト値で受信）
- フィールド削除: `reserved`で番号予約すれば安全
- フィールド型変更: 非互換（新しい番号でフィールドを追加）
- enum値追加: 安全（旧クライアントは未知値として扱う）
- サービスメソッド追加: 安全

### Q3: gRPCのテストはどう書くか？

`bufconn`パッケージでインメモリ接続を作成し、実際のgRPCサーバーをテスト内で起動する。ネットワーク不要で高速にテスト可能。インターセプタのテストも含められる。

テスト戦略:
1. **単体テスト**: bufconnでサーバー+クライアントをテスト
2. **インテグレーションテスト**: 実際のサーバーを起動してテスト
3. **モック**: mockgenでクライアントインターフェースのモックを生成
4. **E2Eテスト**: grpcurl等で手動テスト

### Q4: gRPC-Gatewayとgrpc-webとConnectの違いは？

- **gRPC-Gateway**: gRPCをREST/JSONに変換するリバースプロキシ。別プロセスとして動作
- **grpc-web**: ブラウザからgRPCを呼び出すためのプロトコル。Envoyプロキシが必要
- **Connect**: Buf社のRPCフレームワーク。gRPC/gRPC-Web/Connectプロトコルを1つのハンドラで対応。プロキシ不要

### Q5: gRPCのパフォーマンスチューニングは？

1. **KeepAlive設定**: 接続の再利用でレイテンシ削減
2. **メッセージサイズ制限**: 適切なサイズ制限で安全性確保
3. **コネクションプール**: 複数の接続を使って並列処理
4. **圧縮**: gzip圧縮でネットワーク帯域削減
5. **ストリーミング**: 大量データはストリーミングで分割送信
6. **サーバーリフレクション**: 本番では無効にしてセキュリティ向上

### Q6: grpcurl の使い方は？

```bash
# サービス一覧
grpcurl -plaintext localhost:50051 list

# メソッド一覧
grpcurl -plaintext localhost:50051 list user.v1.UserService

# Unary RPC呼び出し
grpcurl -plaintext -d '{"id": 1}' localhost:50051 user.v1.UserService/GetUser

# メタデータ付き
grpcurl -plaintext \
  -H 'Authorization: Bearer token123' \
  -d '{"name": "Test", "email": "test@example.com"}' \
  localhost:50051 user.v1.UserService/CreateUser

# Server Streaming
grpcurl -plaintext -d '{"user_ids": [1, 2, 3]}' \
  localhost:50051 user.v1.UserService/WatchUsers
```

---

## まとめ

| 概念 | 要点 |
|------|------|
| Protocol Buffers | .protoからGo/他言語のコードを自動生成 |
| Unary RPC | 1リクエスト→1レスポンス。基本パターン |
| Server Streaming | サーバーから連続レスポンス。リアルタイム通知 |
| Client Streaming | クライアントから連続リクエスト。バッチ処理 |
| Bidirectional | 双方向リアルタイム通信。チャット等 |
| Interceptor | gRPC版ミドルウェア。認証、ロギング、リカバリ |
| Status Code | 独自コード体系。エラー詳細はWithDetailsで付与 |
| gRPC-Gateway | REST API自動変換。外部公開との共存 |
| FieldMask | 部分更新パターン。帯域削減 |
| bufconn | インメモリテスト。高速・ネットワーク不要 |
| ヘルスチェック | grpc_health_v1。ロードバランサ連携 |
| Graceful Shutdown | GracefulStop()で安全な停止 |

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
4. **Buf** -- https://buf.build/docs/
5. **Connect** -- https://connectrpc.com/
6. **Google API Design Guide** -- https://cloud.google.com/apis/design
7. **gRPC Status Codes** -- https://grpc.github.io/grpc/core/md_doc_statuscodes.html
8. **grpcurl** -- https://github.com/fullstorydev/grpcurl
9. **go-grpc-middleware** -- https://github.com/grpc-ecosystem/go-grpc-middleware
