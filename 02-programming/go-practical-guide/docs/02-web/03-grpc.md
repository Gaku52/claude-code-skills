# gRPC -- Protocol Buffers, Service Definitions, Streaming

> gRPC is a high-performance RPC framework based on Protocol Buffers, enabling flexible API design through type-safe service definitions, bidirectional streaming, and gRPC-Gateway.

---

## What You Will Learn in This Chapter

1. **Protocol Buffers** -- Service definitions and code generation
2. **Four RPC patterns** -- Unary / Server / Client / Bi-directional Streaming
3. **gRPC-Gateway** -- Integration with REST APIs
4. **Interceptors** -- Middleware for authentication, logging, recovery, and more
5. **Error Handling** -- Status codes and error details
6. **Testing** -- bufconn, mocks, and integration tests
7. **Performance Optimization** -- Connection management, load balancing


## Prerequisites

Reading this guide will be easier if you have the following background knowledge:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Databases -- database/sql, sqlx, GORM](./02-database.md)

---

## 1. Protocol Buffers Basics

### Code Example 1: Designing a Proto Definition

```protobuf
syntax = "proto3";
package user.v1;
option go_package = "gen/user/v1;userv1";

import "google/protobuf/timestamp.proto";
import "google/protobuf/field_mask.proto";
import "google/protobuf/empty.proto";

// UserService is the user management service
service UserService {
  // Unary RPC: retrieve a single user
  rpc GetUser(GetUserRequest) returns (GetUserResponse);

  // Unary RPC: list users (with pagination)
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);

  // Unary RPC: create a user
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);

  // Unary RPC: update a user (supports partial updates)
  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);

  // Unary RPC: delete a user
  rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty);

  // Server Streaming: deliver user changes in real time
  rpc WatchUsers(WatchUsersRequest) returns (stream UserEvent);

  // Client Streaming: batch user creation
  rpc BatchCreateUsers(stream CreateUserRequest) returns (BatchCreateUsersResponse);

  // Bidirectional Streaming: chat
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

// User is the user message
message User {
  int64 id = 1;
  string name = 2;
  string email = 3;
  UserRole role = 4;
  UserProfile profile = 5;
  google.protobuf.Timestamp created_at = 6;
  google.protobuf.Timestamp updated_at = 7;
}

// UserRole is the role of the user
enum UserRole {
  USER_ROLE_UNSPECIFIED = 0;
  USER_ROLE_ADMIN = 1;
  USER_ROLE_MEMBER = 2;
  USER_ROLE_VIEWER = 3;
}

// UserProfile holds the user's profile information
message UserProfile {
  string bio = 1;
  string avatar_url = 2;
  string location = 3;
  string website = 4;
}

// GetUserRequest is the request for GetUser
message GetUserRequest {
  int64 id = 1;
}

// GetUserResponse is the response for GetUser
message GetUserResponse {
  User user = 1;
}

// ListUsersRequest is the request for ListUsers (with pagination)
message ListUsersRequest {
  int32 page_size = 1;    // max 100
  string page_token = 2;  // token for the next page
  string filter = 3;      // filter condition (e.g., "role=admin")
  string order_by = 4;    // sort order (e.g., "name asc")
}

// ListUsersResponse is the response for ListUsers
message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
  int32 total_count = 3;
}

// CreateUserRequest is the request for CreateUser
message CreateUserRequest {
  string name = 1;
  string email = 2;
  UserRole role = 3;
  UserProfile profile = 4;
}

// CreateUserResponse is the response for CreateUser
message CreateUserResponse {
  User user = 1;
}

// UpdateUserRequest is the request for UpdateUser (supports partial updates)
message UpdateUserRequest {
  User user = 1;
  // Specify fields to update (partial update)
  google.protobuf.FieldMask update_mask = 2;
}

// UpdateUserResponse is the response for UpdateUser
message UpdateUserResponse {
  User user = 1;
}

// DeleteUserRequest is the request for DeleteUser
message DeleteUserRequest {
  int64 id = 1;
}

// WatchUsersRequest is the request for WatchUsers
message WatchUsersRequest {
  repeated int64 user_ids = 1; // user IDs to watch (empty means all)
}

// UserEvent is a user change event
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

// BatchCreateUsersResponse is the response for batch creation
message BatchCreateUsersResponse {
  int32 created_count = 1;
  repeated User users = 2;
}

// ChatMessage is a chat message
message ChatMessage {
  string sender_id = 1;
  string content = 2;
  google.protobuf.Timestamp sent_at = 3;
}
```

### Code Example 2: Proto File Layout and Buf Configuration

```yaml
# buf.yaml -- Buf configuration file
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
# buf.gen.yaml -- code generation configuration
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
# Project layout
proto/
├── buf.yaml
├── buf.gen.yaml
├── buf.lock
└── user/
    └── v1/
        ├── user.proto
        └── user_service.proto

# Code generation command
buf generate

# Lint check
buf lint

# Detect breaking changes
buf breaking --against '.git#branch=main'
```

---

## 2. gRPC Server Implementation

### Code Example 3: Server Implementation (Unary RPC)

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

// userServer is the implementation of UserService
type userServer struct {
    userv1.UnimplementedUserServiceServer
    mu    sync.RWMutex
    users map[int64]*userv1.User
    nextID int64
}

// NewUserServer creates a new UserServer
func NewUserServer() *userServer {
    return &userServer{
        users:  make(map[int64]*userv1.User),
        nextID: 1,
    }
}

// GetUser retrieves a user (Unary RPC)
func (s *userServer) GetUser(ctx context.Context, req *userv1.GetUserRequest) (*userv1.GetUserResponse, error) {
    // Validation
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

// ListUsers retrieves a list of users (with pagination)
func (s *userServer) ListUsers(ctx context.Context, req *userv1.ListUsersRequest) (*userv1.ListUsersResponse, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()

    pageSize := int(req.PageSize)
    if pageSize <= 0 || pageSize > 100 {
        pageSize = 20 // default
    }

    // Sort all users by ID
    var allUsers []*userv1.User
    for _, u := range s.users {
        allUsers = append(allUsers, u)
    }

    // Pagination (simplified)
    start := 0
    if req.PageToken != "" {
        // Restore offset from token
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

// CreateUser creates a user
func (s *userServer) CreateUser(ctx context.Context, req *userv1.CreateUserRequest) (*userv1.CreateUserResponse, error) {
    // Validation
    if req.Name == "" {
        return nil, status.Errorf(codes.InvalidArgument, "name is required")
    }
    if req.Email == "" {
        return nil, status.Errorf(codes.InvalidArgument, "email is required")
    }

    // Check for duplicate email
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

// UpdateUser updates a user (supports FieldMask)
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

    // Partial update with FieldMask
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
        // Without FieldMask, update all fields
        existing.Name = req.User.Name
        existing.Email = req.User.Email
        existing.Role = req.User.Role
        existing.Profile = req.User.Profile
    }

    existing.UpdatedAt = timestamppb.Now()

    return &userv1.UpdateUserResponse{User: existing}, nil
}

// DeleteUser deletes a user
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

### Code Example 4: Server Streaming RPC

```go
// WatchUsers streams user changes in real time (Server Streaming)
func (s *userServer) WatchUsers(req *userv1.WatchUsersRequest, stream userv1.UserService_WatchUsersServer) error {
    log.Printf("WatchUsers started for user IDs: %v", req.UserIds)

    // Watch event channel
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-stream.Context().Done():
            // Client disconnected
            log.Printf("WatchUsers: client disconnected")
            return nil
        case <-ticker.C:
            // Send event if a change occurred (in practice, fetched from an event bus)
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
    // A real implementation would use an event bus or CDC
    return nil
}
```

### Code Example 5: Client Streaming RPC

```go
// BatchCreateUsers performs batch user creation (Client Streaming)
func (s *userServer) BatchCreateUsers(stream userv1.UserService_BatchCreateUsersServer) error {
    var createdUsers []*userv1.User
    var count int32

    for {
        req, err := stream.Recv()
        if err != nil {
            if err.Error() == "EOF" {
                // Client finished sending
                return stream.SendAndClose(&userv1.BatchCreateUsersResponse{
                    CreatedCount: count,
                    Users:        createdUsers,
                })
            }
            return status.Errorf(codes.Internal, "failed to receive: %v", err)
        }

        // Create each user individually
        resp, err := s.CreateUser(stream.Context(), req)
        if err != nil {
            log.Printf("BatchCreateUsers: skip user %s: %v", req.Name, err)
            continue // Skip on error and continue
        }

        createdUsers = append(createdUsers, resp.User)
        count++
    }
}
```

### Code Example 6: Bidirectional Streaming RPC

```go
// Chat is a bidirectional streaming chat
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

        // Echo reply (in practice, broadcast or similar)
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

## 3. gRPC Client Implementation

### Code Example 7: Client Connection and Invocation

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

// UserClient is a wrapper around the gRPC client
type UserClient struct {
    conn   *grpc.ClientConn
    client userv1.UserServiceClient
}

// NewUserClient creates a new gRPC client
func NewUserClient(addr string, opts ...grpc.DialOption) (*UserClient, error) {
    // Default options
    defaultOpts := []grpc.DialOption{
        grpc.WithKeepaliveParams(keepalive.ClientParameters{
            Time:                10 * time.Second, // KeepAlive ping interval
            Timeout:             3 * time.Second,  // Ping response timeout
            PermitWithoutStream: false,             // Do not ping when no stream is active
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

// NewInsecureUserClient creates a client without TLS (for development)
func NewInsecureUserClient(addr string) (*UserClient, error) {
    return NewUserClient(addr,
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    )
}

// NewSecureUserClient creates a client with TLS (for production)
func NewSecureUserClient(addr string) (*UserClient, error) {
    tlsConfig := &tls.Config{
        MinVersion: tls.VersionTLS13,
    }
    return NewUserClient(addr,
        grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)),
    )
}

// Close closes the connection
func (c *UserClient) Close() error {
    return c.conn.Close()
}

// GetUser retrieves a user via Unary RPC
func (c *UserClient) GetUser(ctx context.Context, id int64) (*userv1.User, error) {
    // Set timeout
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    resp, err := c.client.GetUser(ctx, &userv1.GetUserRequest{Id: id})
    if err != nil {
        return nil, fmt.Errorf("GetUser: %w", err)
    }
    return resp.User, nil
}

// ListAllUsers retrieves all users using pagination
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

// WatchUsers receives user events via Server Streaming
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

// CreateWithMetadata creates a user with attached metadata
func (c *UserClient) CreateWithMetadata(ctx context.Context, name, email, token string) (*userv1.User, error) {
    // Attach metadata (equivalent to HTTP headers)
    md := metadata.Pairs(
        "authorization", "Bearer "+token,
        "x-request-id", generateRequestID(),
    )
    ctx = metadata.NewOutgoingContext(ctx, md)

    // Receive response header and trailer
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

    // Extract rate limit info from the response header
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

## 4. Interceptors (Middleware)

### Code Example 8: Unary Interceptors

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

// LoggingUnaryInterceptor logs each request
func LoggingUnaryInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    start := time.Now()

    // Retrieve request ID from metadata
    requestID := "unknown"
    if md, ok := metadata.FromIncomingContext(ctx); ok {
        if ids := md.Get("x-request-id"); len(ids) > 0 {
            requestID = ids[0]
        }
    }

    // Invoke the handler
    resp, err := handler(ctx, req)

    // Log output
    duration := time.Since(start)
    code := codes.OK
    if err != nil {
        code = status.Code(err)
    }

    log.Printf("[gRPC] method=%s request_id=%s code=%s duration=%v",
        info.FullMethod, requestID, code, duration)

    return resp, err
}

// AuthUnaryInterceptor performs authentication
func AuthUnaryInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    // Skip methods that don't require authentication, such as health checks
    skipMethods := map[string]bool{
        "/grpc.health.v1.Health/Check": true,
        "/grpc.reflection.v1.ServerReflection/ServerReflectionInfo": true,
    }
    if skipMethods[info.FullMethod] {
        return handler(ctx, req)
    }

    // Retrieve token from metadata
    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return nil, status.Error(codes.Unauthenticated, "missing metadata")
    }

    tokens := md.Get("authorization")
    if len(tokens) == 0 {
        return nil, status.Error(codes.Unauthenticated, "missing authorization token")
    }

    // Validate token
    userID, err := validateToken(tokens[0])
    if err != nil {
        return nil, status.Errorf(codes.Unauthenticated, "invalid token: %v", err)
    }

    // Store user ID in the context
    ctx = context.WithValue(ctx, userIDKey{}, userID)

    return handler(ctx, req)
}

type userIDKey struct{}

func UserIDFromContext(ctx context.Context) (string, bool) {
    id, ok := ctx.Value(userIDKey{}).(string)
    return id, ok
}

func validateToken(token string) (string, error) {
    // In practice, perform JWT validation, etc.
    if token == "" {
        return "", fmt.Errorf("empty token")
    }
    return "user-123", nil
}

// RecoveryUnaryInterceptor recovers from panics
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

// RateLimitUnaryInterceptor enforces rate limits
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

// ValidationUnaryInterceptor validates requests
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

// TimeoutUnaryInterceptor sets a default timeout
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

### Code Example 9: Stream Interceptors

```go
package interceptor

import (
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// LoggingStreamInterceptor logs streaming calls
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

// RecoveryStreamInterceptor recovers from panics in streams
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

### Code Example 10: Starting the Server (Integrating Interceptors)

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

    // Server options
    s := grpc.NewServer(
        // Unary interceptor chain (executed in order)
        grpc.ChainUnaryInterceptor(
            RecoveryUnaryInterceptor,
            LoggingUnaryInterceptor,
            TimeoutUnaryInterceptor(30*time.Second),
            AuthUnaryInterceptor,
            ValidationUnaryInterceptor,
        ),
        // Stream interceptor chain
        grpc.ChainStreamInterceptor(
            RecoveryStreamInterceptor,
            LoggingStreamInterceptor,
        ),
        // Message size limits
        grpc.MaxRecvMsgSize(10 * 1024 * 1024), // 10MB
        grpc.MaxSendMsgSize(10 * 1024 * 1024), // 10MB
        // KeepAlive configuration
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

    // Register service
    userv1.RegisterUserServiceServer(s, NewUserServer())

    // Health check
    healthServer := health.NewServer()
    grpc_health_v1.RegisterHealthServer(s, healthServer)
    healthServer.SetServingStatus("user.v1.UserService", grpc_health_v1.HealthCheckResponse_SERVING)

    // Reflection (for development, used by grpcurl, etc.)
    reflection.Register(s)

    // Graceful shutdown
    go func() {
        log.Printf("gRPC server listening on :50051")
        if err := s.Serve(lis); err != nil {
            log.Fatalf("failed to serve: %v", err)
        }
    }()

    // Wait for signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit

    log.Println("Shutting down gRPC server...")
    healthServer.SetServingStatus("user.v1.UserService", grpc_health_v1.HealthCheckResponse_NOT_SERVING)

    // Graceful stop (wait for in-flight RPCs to complete)
    stopped := make(chan struct{})
    go func() {
        s.GracefulStop()
        close(stopped)
    }()

    // Wait with timeout
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

## 5. gRPC-Gateway (REST Translation)

### Code Example 11: gRPC-Gateway Definition

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

### Code Example 12: gRPC-Gateway Server

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

    // Start the gRPC server
    go runGRPCServer()

    // Configure gRPC-Gateway
    mux := runtime.NewServeMux(
        // JSON output options
        runtime.WithMarshalerOption(runtime.MIMEWildcard, &runtime.JSONPb{
            MarshalOptions: protojson.MarshalOptions{
                UseProtoNames:   true,  // snake_case field names
                EmitUnpopulated: false, // omit zero-value fields
            },
            UnmarshalOptions: protojson.UnmarshalOptions{
                DiscardUnknown: true, // ignore unknown fields
            },
        }),
        // Customize error handling
        runtime.WithErrorHandler(customErrorHandler),
        // Forward metadata
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

    // Connect to the gRPC backend
    opts := []grpc.DialOption{
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    }
    err := userv1.RegisterUserServiceHandlerFromEndpoint(ctx, mux, "localhost:50051", opts)
    if err != nil {
        log.Fatalf("failed to register gateway: %v", err)
    }

    // HTTP server (with CORS and logging middleware)
    handler := corsMiddleware(loggingMiddleware(mux))

    log.Printf("gRPC-Gateway listening on :8080")
    if err := http.ListenAndServe(":8080", handler); err != nil {
        log.Fatalf("failed to serve: %v", err)
    }
}

// customErrorHandler converts gRPC errors to HTTP responses
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

    // If error details are present
    for _, detail := range st.Details() {
        body["error"].(map[string]interface{})["details"] = detail
    }

    data, _ := json.Marshal(body)
    w.Write(data)
}

// corsMiddleware sets CORS headers
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

// loggingMiddleware logs HTTP requests
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        log.Printf("[HTTP] %s %s %v", r.Method, r.URL.Path, time.Since(start))
    })
}
```

---

## 6. Testing

### Code Example 13: Testing with bufconn

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

// setupTestServer starts an in-memory gRPC server
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

    // Dialer for bufconn
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

    // First call: success
    _, err := client.CreateUser(ctx, &userv1.CreateUserRequest{
        Name:  "User 1",
        Email: "dup@example.com",
    })
    if err != nil {
        t.Fatalf("first CreateUser failed: %v", err)
    }

    // Second call: duplicate error
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

    // Create 5 users
    for i := 0; i < 5; i++ {
        _, err := client.CreateUser(ctx, &userv1.CreateUserRequest{
            Name:  fmt.Sprintf("User %d", i),
            Email: fmt.Sprintf("user%d@example.com", i),
        })
        if err != nil {
            t.Fatalf("CreateUser %d failed: %v", i, err)
        }
    }

    // Fetch with page size 2
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

## 7. Error Handling

### Code Example 14: Detailed Error Responses

```go
package server

import (
    "google.golang.org/genproto/googleapis/rpc/errdetails"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// Build a validation error
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

// Resource not found error
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

// Rate limit error
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

// Extract error details on the client side
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

## 8. ASCII Diagrams

### Figure 1: gRPC Communication Flow

```
Client (Go)                    Server (Go)
┌──────────────────┐    HTTP/2 +      ┌──────────────────┐
│                  │    Protocol      │                  │
│ Generated Stub   │    Buffers       │ Generated Service│
│ (UserServiceClient)│ ──────────────>│(UserServiceServer)│
│                  │    (binary)      │                  │
│ .proto → Go code │ <──────────────│ .proto → Go code │
│                  │                  │                  │
│ grpc.ClientConn  │    TLS +        │ grpc.Server      │
│                  │    HTTP/2        │                  │
└──────────────────┘    Multiplexed  └──────────────────┘
                        Streams

Code generation pipeline:
  .proto ──> protoc / buf generate
                │
                ├── *.pb.go        (message types)
                ├── *_grpc.pb.go   (service interfaces)
                ├── *.pb.gw.go     (gRPC-Gateway)
                └── *.swagger.json (OpenAPI spec)
```

### Figure 2: The Four RPC Patterns

```
1. Unary RPC (1:1) -- GetUser, CreateUser
   Client ──[Request]──> Server
   Client <──[Response]── Server
   The most basic pattern. A REST API alternative.

2. Server Streaming (1:N) -- WatchUsers
   Client ──[Request]──────> Server
   Client <──[Response 1]── Server
   Client <──[Response 2]── Server
   Client <──[Response 3]── Server
   Client <──[EOF]────────── Server
   Real-time notifications, chunked delivery of large data.

3. Client Streaming (N:1) -- BatchCreateUsers
   Client ──[Request 1]──> Server
   Client ──[Request 2]──> Server
   Client ──[Request 3]──> Server
   Client ──[EOF]────────> Server
   Client <──[Response]──── Server
   File uploads, batch processing.

4. Bidirectional Streaming (N:M) -- Chat
   Client ──[Request 1]──> Server
   Client <──[Response 1]── Server
   Client ──[Request 2]──> Server
   Client ──[Request 3]──> Server
   Client <──[Response 2]── Server
   Client <──[Response 3]── Server
   Chat, games, real-time collaboration.
```

### Figure 3: gRPC-Gateway Architecture

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
  (direct connection, highest performance)                 :50051

Supports both REST and gRPC:
  /api/v1/users/{id}  →  UserService.GetUser()
  /api/v1/users       →  UserService.ListUsers()
  POST /api/v1/users  →  UserService.CreateUser()
```

### Figure 4: Interceptor Chain

```
Request
  │
  ▼
┌────────────────────────────────────────────────┐
│              Interceptor Chain                  │
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Recovery │─>│ Logging  │─>│ Timeout  │    │
│  │panic recv│  │   logs   │  │ timeout  │    │
│  └──────────┘  └──────────┘  └──────────┘    │
│       │                                  │     │
│       ▼                                  ▼     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   Auth   │─>│Validation│─>│ RateLimit│    │
│  │  authN   │  │input chk │  │rate limit│    │
│  └──────────┘  └──────────┘  └──────────┘    │
│                                    │           │
└────────────────────────────────────┼───────────┘
                                     │
                                     ▼
                              ┌──────────┐
                              │  Handler │
                              │ business │
                              └──────────┘
```

### Figure 5: gRPC Health Checks and Load Balancing

```
┌────────────────────────────────────────────────────────┐
│                  Load Balancer                          │
│  (Envoy / Nginx / Kubernetes Service)                  │
│                                                        │
│  Health check:                                         │
│  grpc_health_v1.Health/Check → SERVING / NOT_SERVING   │
│                                                        │
│  Load balancing strategies:                            │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐   │
│  │ Round Robin│  │ Least Conn   │  │ Weighted     │   │
│  │ even split │  │ fewest conns │  │  weighted    │   │
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

## 9. Comparison Tables

### Table 1: Detailed Comparison of gRPC vs REST

| Item | gRPC | REST (JSON) |
|------|------|-------------|
| Protocol | HTTP/2 | HTTP/1.1 or HTTP/2 |
| Serialization | Protocol Buffers (binary) | JSON (text) |
| Performance | Very fast (10-100x) | Moderate |
| Type safety | Strong (generated from .proto) | Weak (supplemented by OpenAPI) |
| Streaming | Bidirectional support | SSE/WebSocket |
| Browser support | Requires grpc-web/Connect | Native |
| Debugging | grpcurl, grpcui | curl, Postman |
| Ecosystem | Medium | Very large |
| Code generation | Automatic (proto) | Manual/OpenAPI |
| Versioning | Package version | URL/header |
| Error model | gRPC Status Code | HTTP Status Code |
| Recommended use | Microservice-to-microservice communication | Public external APIs |

### Table 2: gRPC Status Codes in Detail

| gRPC Code | HTTP equivalent | Use | Example |
|-----------|-----------------|-----|---------|
| OK (0) | 200 | Success | Normal completion |
| Cancelled (1) | 499 | Client canceled | Request interrupted |
| Unknown (2) | 500 | Unknown error | Unexpected exception |
| InvalidArgument (3) | 400 | Validation error | Invalid email format |
| DeadlineExceeded (4) | 504 | Timeout | Processing time exceeded |
| NotFound (5) | 404 | Resource missing | User does not exist |
| AlreadyExists (6) | 409 | Resource conflict | Email already exists |
| PermissionDenied (7) | 403 | Permission error | Admin rights required |
| ResourceExhausted (8) | 429 | Resource exhausted | Rate limit exceeded |
| FailedPrecondition (9) | 400 | Precondition mismatch | ETag mismatch |
| Aborted (10) | 409 | Operation aborted | Transaction conflict |
| OutOfRange (11) | 400 | Out of range | Invalid page token |
| Unimplemented (12) | 501 | Not implemented | Method not supported |
| Internal (13) | 500 | Server error | Internal processing failure |
| Unavailable (14) | 503 | Service unavailable | Under maintenance |
| DataLoss (15) | 500 | Data loss | Data corruption detected |
| Unauthenticated (16) | 401 | Authentication error | Invalid token |

### Table 3: RPC Pattern Selection Guide

| Pattern | Use | Message count | Example |
|---------|-----|---------------|---------|
| Unary | Basic request/response | 1:1 | CRUD operations, authentication |
| Server Streaming | Continuous data from server | 1:N | Real-time notifications, large data retrieval |
| Client Streaming | Continuous data from client | N:1 | File uploads, batch processing |
| Bidirectional | Bidirectional real-time communication | N:M | Chat, games, collaborative editing |

### Table 4: Protocol Buffers Best Practices

| Rule | Description | Example |
|------|-------------|---------|
| Reserve field numbers | Mark deleted field numbers as reserved | `reserved 3, 15;` |
| Enum default value | The 0 value is UNSPECIFIED | `ROLE_UNSPECIFIED = 0;` |
| Package versioning | Include version in package name | `package user.v1;` |
| FieldMask | Use for partial updates | `update_mask` field |
| Pagination | page_size + page_token | `string next_page_token;` |
| Timestamp | Use google.protobuf.Timestamp for datetimes | `import "google/protobuf/timestamp.proto";` |

---

## 10. Anti-Patterns

### Anti-Pattern 1: Oversized Messages

```protobuf
// BAD: returning a huge amount of data in one response
message ListUsersResponse {
  repeated User users = 1;  // could return 1M rows → out of memory
}

// GOOD: use pagination or streaming
// Option 1: pagination
message ListUsersRequest {
  int32 page_size = 1;     // max 100
  string page_token = 2;   // cursor
}
message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
}

// Option 2: Server Streaming (for large volumes)
rpc StreamUsers(StreamUsersRequest) returns (stream User);
```

### Anti-Pattern 2: Not Returning Error Details

```go
// BAD: generic error message
return nil, status.Error(codes.Internal, "error")
// → the client has no idea what went wrong

// GOOD: attach detailed error information
st := status.New(codes.InvalidArgument, "validation failed")
st, _ = st.WithDetails(&errdetails.BadRequest{
    FieldViolations: []*errdetails.BadRequest_FieldViolation{
        {Field: "email", Description: "invalid email format"},
        {Field: "name", Description: "name must be 1-100 characters"},
    },
})
return nil, st.Err()
```

### Anti-Pattern 3: Ignoring the Context

```go
// BAD: running a long operation while ignoring the context
func (s *server) SlowRPC(ctx context.Context, req *pb.Request) (*pb.Response, error) {
    result := heavyComputation() // does not check ctx.Done()
    return &pb.Response{Data: result}, nil
}

// GOOD: periodically check the context for cancellation
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

### Anti-Pattern 4: Changing Field Numbers

```protobuf
// BAD: changing the number of an existing field (breaks backward compatibility)
// Before:
message User {
  string name = 1;
  string email = 2;
}
// After (broken):
message User {
  string email = 1;  // number changed → existing clients break
  string name = 2;
}

// GOOD: add new fields with new numbers
message User {
  string name = 1;
  string email = 2;
  string phone = 3;  // new fields get the next available number
  reserved 4;        // reserve the numbers of deleted fields
}
```

### Anti-Pattern 5: Leaving Unimplemented Methods

```go
// BAD: simply embedding UnimplementedServer and nothing more
type myServer struct {
    pb.UnimplementedMyServiceServer
}
// Unimplemented methods return an Unimplemented error when called
// → you may not notice in production

// GOOD: use mustEmbedUnimplemented to detect unimplemented methods,
// or explicitly implement every method and return an error for those not yet supported.
func (s *myServer) NotYetImplemented(ctx context.Context, req *pb.Request) (*pb.Response, error) {
    return nil, status.Error(codes.Unimplemented,
        "NotYetImplemented is not yet available, planned for v2.0")
}
```

---

## 11. FAQ

### Q1: When should I choose gRPC?

Choose gRPC for microservice-to-microservice communication, internal APIs that require low latency, and situations where streaming is needed. REST (or REST combined with gRPC-Gateway) is more appropriate for public, externally exposed APIs.

Decision criteria:
- **gRPC fits**: inter-microservice communication, high throughput, strong emphasis on type safety, streaming required
- **REST fits**: public external APIs, direct browser communication, simple CRUD, integration with existing systems
- **Both**: use gRPC-Gateway to provide gRPC internally and REST externally

### Q2: What about Protocol Buffers version compatibility?

Backward compatibility is preserved as long as you do not change existing field numbers. Adding new fields is safe. When deleting a field, reserve its number with `reserved` so it cannot be reused. This enables rolling updates.

Compatibility rules:
- Adding a field: safe (old clients receive the default value)
- Deleting a field: safe if the number is reserved with `reserved`
- Changing a field's type: incompatible (add a new field with a new number instead)
- Adding an enum value: safe (old clients treat it as unknown)
- Adding a service method: safe

### Q3: How do I write gRPC tests?

Use the `bufconn` package to create an in-memory connection and start the actual gRPC server inside your tests. This allows fast testing without real network I/O and lets you test interceptors as well.

Testing strategy:
1. **Unit tests**: test server and client together using bufconn
2. **Integration tests**: start a real server and test against it
3. **Mocks**: generate client interface mocks with mockgen
4. **E2E tests**: manual testing with tools such as grpcurl

### Q4: What is the difference between gRPC-Gateway, grpc-web, and Connect?

- **gRPC-Gateway**: a reverse proxy that converts gRPC into REST/JSON. Runs as a separate process.
- **grpc-web**: a protocol for calling gRPC from browsers. Requires an Envoy proxy.
- **Connect**: Buf's RPC framework. Supports the gRPC, gRPC-Web, and Connect protocols through a single handler. No proxy required.

### Q5: How do I tune gRPC performance?

1. **KeepAlive settings**: reduce latency by reusing connections
2. **Message size limits**: enforce sensible size limits for safety
3. **Connection pooling**: use multiple connections to process in parallel
4. **Compression**: use gzip compression to save bandwidth
5. **Streaming**: split large data and send it via streaming
6. **Server reflection**: disable it in production to improve security

### Q6: How do I use grpcurl?

```bash
# List services
grpcurl -plaintext localhost:50051 list

# List methods
grpcurl -plaintext localhost:50051 list user.v1.UserService

# Call a Unary RPC
grpcurl -plaintext -d '{"id": 1}' localhost:50051 user.v1.UserService/GetUser

# With metadata
grpcurl -plaintext \
  -H 'Authorization: Bearer token123' \
  -d '{"name": "Test", "email": "test@example.com"}' \
  localhost:50051 user.v1.UserService/CreateUser

# Server Streaming
grpcurl -plaintext -d '{"user_ids": [1, 2, 3]}' \
  localhost:50051 user.v1.UserService/WatchUsers
```

---


## FAQ

### Q1: What is the single most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens when you go beyond theory and actually write code to verify how it behaves.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping into advanced topics. We recommend firmly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in real-world practice?

The knowledge in this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architectural design.

---

## Summary

| Concept | Key point |
|---------|-----------|
| Protocol Buffers | Automatically generate Go (and other languages) code from .proto |
| Unary RPC | One request to one response. The basic pattern. |
| Server Streaming | Continuous responses from the server. Real-time notifications. |
| Client Streaming | Continuous requests from the client. Batch processing. |
| Bidirectional | Bidirectional real-time communication. Chat and similar use cases. |
| Interceptor | The gRPC equivalent of middleware. Authentication, logging, recovery. |
| Status Code | Its own code system. Attach error details with WithDetails. |
| gRPC-Gateway | Automatic REST API conversion. Coexistence with external-facing APIs. |
| FieldMask | The partial update pattern. Reduces bandwidth. |
| bufconn | In-memory testing. Fast and requires no network. |
| Health check | grpc_health_v1. Integrates with load balancers. |
| Graceful Shutdown | Safely stop the server with GracefulStop(). |

---

## Recommended Next Guides

- [04-testing.md](./04-testing.md) -- Testing
- [../03-tools/03-deployment.md](../03-tools/03-deployment.md) -- Deployment
- [../01-concurrency/03-context.md](../01-concurrency/03-context.md) -- Context

---

## References

1. **gRPC Go** -- https://grpc.io/docs/languages/go/
2. **Protocol Buffers Language Guide** -- https://protobuf.dev/programming-guides/proto3/
3. **gRPC-Gateway** -- https://grpc-ecosystem.github.io/grpc-gateway/
4. **Buf** -- https://buf.build/docs/
5. **Connect** -- https://connectrpc.com/
6. **Google API Design Guide** -- https://cloud.google.com/apis/design
7. **gRPC Status Codes** -- https://grpc.github.io/grpc/core/md_doc_statuscodes.html
8. **grpcurl** -- https://github.com/fullstorydev/grpcurl
