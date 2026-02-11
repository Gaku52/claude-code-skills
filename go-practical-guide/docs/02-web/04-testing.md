# Go テスト完全ガイド

> table-driven tests、testify、httptestを駆使してGoコードの品質を保証する実践的テスト手法

## この章で学ぶこと

1. **table-driven tests** のパターンを使い、網羅的かつ保守しやすいテストを書く方法
2. **testify** ライブラリによるアサーション・モック・スイートの活用法
3. **httptest** パッケージでHTTPハンドラとクライアントをテストする技法

---

## 1. Goテストの基本構造

### 1-1. テストファイルの命名規則

```
project/
├── handler.go
├── handler_test.go      ← 同一パッケージ
├── service.go
└── service_test.go
```

Goのテストは `_test.go` サフィックスを持つファイルに記述する。テスト関数は `Test` プレフィックスで始まり、`*testing.T` を引数に取る。

### コード例1: 最小のテスト

```go
package calc

import "testing"

func Add(a, b int) int {
    return a + b
}

func TestAdd(t *testing.T) {
    got := Add(2, 3)
    want := 5
    if got != want {
        t.Errorf("Add(2, 3) = %d, want %d", got, want)
    }
}
```

### テスト実行フロー

```
+------------------+     +------------------+     +------------------+
|  go test ./...   | --> | コンパイル        | --> | テストバイナリ    |
|  コマンド実行     |     | *_test.go を含む  |     | 実行・結果表示    |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
  フラグ解析              テスト関数発見            PASS / FAIL 判定
  -v, -run, -cover       Test*, Benchmark*        exit code 0 or 1
```

---

## 2. Table-Driven Tests

Go で最も推奨されるテストパターン。テストケースをスライスで定義し、ループで実行する。

### コード例2: 基本的なtable-driven test

```go
func TestDivide(t *testing.T) {
    tests := []struct {
        name      string
        a, b      float64
        want      float64
        wantError bool
    }{
        {name: "正常な除算", a: 10, b: 2, want: 5, wantError: false},
        {name: "小数結果", a: 7, b: 3, want: 2.3333, wantError: false},
        {name: "ゼロ除算", a: 5, b: 0, want: 0, wantError: true},
        {name: "負の数", a: -10, b: 2, want: -5, wantError: false},
        {name: "両方負", a: -10, b: -2, want: 5, wantError: false},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := Divide(tt.a, tt.b)
            if (err != nil) != tt.wantError {
                t.Fatalf("Divide(%v, %v) error = %v, wantError %v",
                    tt.a, tt.b, err, tt.wantError)
            }
            if !tt.wantError && math.Abs(got-tt.want) > 0.001 {
                t.Errorf("Divide(%v, %v) = %v, want %v",
                    tt.a, tt.b, got, tt.want)
            }
        })
    }
}
```

### Table-Driven Test の構造

```
+------------------------------------------+
|  tests := []struct{ ... }{               |
|    +------------------------------------+|
|    | ケース1: name, input, expected     ||
|    +------------------------------------+|
|    | ケース2: name, input, expected     ||
|    +------------------------------------+|
|    | ケース3: name, input, expected     ||
|    +------------------------------------+|
|  }                                       |
|                                          |
|  for _, tt := range tests {              |
|    t.Run(tt.name, func(t *testing.T){    |
|      // テストロジック                    |
|    })                                    |
|  }                                       |
+------------------------------------------+
         |
         v
  $ go test -run TestDivide/ゼロ除算
  --- 個別ケースの実行も可能
```

### コード例3: 並列実行のtable-driven test

```go
func TestSlowOperation(t *testing.T) {
    tests := []struct {
        name  string
        input string
        want  string
    }{
        {"ケースA", "hello", "HELLO"},
        {"ケースB", "world", "WORLD"},
        {"ケースC", "go", "GO"},
    }

    for _, tt := range tests {
        tt := tt // ループ変数のキャプチャ（Go 1.22未満で必要）
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel() // 並列実行を有効化
            got := strings.ToUpper(tt.input)
            if got != tt.want {
                t.Errorf("ToUpper(%q) = %q, want %q", tt.input, got, tt.want)
            }
        })
    }
}
```

---

## 3. testify ライブラリ

### インストール

```bash
go get github.com/stretchr/testify
```

### コード例4: testify/assert と testify/require

```go
package user_test

import (
    "testing"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestCreateUser(t *testing.T) {
    // assert: 失敗してもテスト続行
    user, err := CreateUser("Alice", "alice@example.com")
    assert.NoError(t, err, "ユーザー作成でエラーが発生した")
    assert.Equal(t, "Alice", user.Name)
    assert.NotEmpty(t, user.ID)

    // require: 失敗したらテスト即中断
    token, err := user.GenerateToken()
    require.NoError(t, err, "トークン生成は必須")
    require.NotEmpty(t, token)

    // さらにトークンを使ったテスト
    claims, err := ParseToken(token)
    assert.NoError(t, err)
    assert.Equal(t, user.ID, claims.UserID)
}
```

### assert vs require 比較表

| 項目 | `assert` | `require` |
|------|----------|-----------|
| 失敗時の動作 | テスト続行（`t.Errorf`相当） | テスト即中断（`t.Fatalf`相当） |
| 用途 | 複数の検証を一度に実行 | 後続テストの前提条件を検証 |
| 戻り値 | `bool`（成功/失敗） | なし（失敗時にt.FailNow） |
| 推奨場面 | 値の比較、属性チェック | nilチェック、エラーチェック |
| 出力 | 全ての失敗を一括表示 | 最初の失敗のみ表示 |

### コード例5: testify/mock

```go
// インターフェース定義
type UserRepository interface {
    FindByID(id string) (*User, error)
    Save(user *User) error
}

// モック生成
type MockUserRepo struct {
    mock.Mock
}

func (m *MockUserRepo) FindByID(id string) (*User, error) {
    args := m.Called(id)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).(*User), args.Error(1)
}

func (m *MockUserRepo) Save(user *User) error {
    args := m.Called(user)
    return args.Error(0)
}

// テストでの使用
func TestUpdateUserName(t *testing.T) {
    mockRepo := new(MockUserRepo)

    existingUser := &User{ID: "123", Name: "Alice"}
    mockRepo.On("FindByID", "123").Return(existingUser, nil)
    mockRepo.On("Save", mock.AnythingOfType("*User")).Return(nil)

    service := NewUserService(mockRepo)
    err := service.UpdateName("123", "Bob")

    assert.NoError(t, err)
    mockRepo.AssertExpectations(t)
    mockRepo.AssertCalled(t, "Save", mock.MatchedBy(func(u *User) bool {
        return u.Name == "Bob"
    }))
}
```

---

## 4. httptest パッケージ

### HTTPテストの全体像

```
+----------------------------+
|  テスト対象の選択           |
+----------------------------+
        |             |
        v             v
+-------------+ +----------------+
| サーバー側   | | クライアント側  |
| ハンドラを   | | 外部API呼出を  |
| テスト       | | テスト          |
+-------------+ +----------------+
        |             |
        v             v
+-------------+ +----------------+
| httptest.   | | httptest.      |
| NewRecorder | | NewServer      |
| リクエスト→  | | モックサーバー  |
| レスポンス   | | を立てて検証    |
+-------------+ +----------------+
```

### コード例6: httptest.NewRecorder でハンドラテスト

```go
func TestHealthHandler(t *testing.T) {
    // ハンドラ定義
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
    })

    // リクエスト作成
    req := httptest.NewRequest("GET", "/health", nil)
    rec := httptest.NewRecorder()

    // ハンドラ実行
    handler.ServeHTTP(rec, req)

    // 検証
    assert.Equal(t, http.StatusOK, rec.Code)
    assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")

    var body map[string]string
    err := json.Unmarshal(rec.Body.Bytes(), &body)
    require.NoError(t, err)
    assert.Equal(t, "ok", body["status"])
}
```

### コード例7: httptest.NewServer で外部APIモック

```go
func TestFetchUserFromAPI(t *testing.T) {
    // モックサーバー作成
    mockServer := httptest.NewServer(http.HandlerFunc(
        func(w http.ResponseWriter, r *http.Request) {
            assert.Equal(t, "/api/users/42", r.URL.Path)
            assert.Equal(t, "Bearer test-token", r.Header.Get("Authorization"))

            w.Header().Set("Content-Type", "application/json")
            json.NewEncoder(w).Encode(map[string]interface{}{
                "id":   42,
                "name": "Alice",
            })
        },
    ))
    defer mockServer.Close()

    // テスト対象のクライアントにモックURLを注入
    client := NewAPIClient(mockServer.URL, "test-token")
    user, err := client.FetchUser(42)

    require.NoError(t, err)
    assert.Equal(t, 42, user.ID)
    assert.Equal(t, "Alice", user.Name)
}
```

### コード例8: ミドルウェアのテスト

```go
func TestAuthMiddleware(t *testing.T) {
    tests := []struct {
        name       string
        token      string
        wantStatus int
    }{
        {"有効なトークン", "valid-token", http.StatusOK},
        {"無効なトークン", "bad-token", http.StatusUnauthorized},
        {"トークンなし", "", http.StatusUnauthorized},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                w.WriteHeader(http.StatusOK)
            })

            handler := AuthMiddleware(inner)
            req := httptest.NewRequest("GET", "/protected", nil)
            if tt.token != "" {
                req.Header.Set("Authorization", "Bearer "+tt.token)
            }
            rec := httptest.NewRecorder()

            handler.ServeHTTP(rec, req)
            assert.Equal(t, tt.wantStatus, rec.Code)
        })
    }
}
```

---

## 5. テストカバレッジとベンチマーク

### カバレッジ

```bash
# カバレッジ付きテスト実行
go test -cover ./...

# HTMLレポート生成
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html

# 関数ごとのカバレッジ
go tool cover -func=coverage.out
```

### コード例9: ベンチマークテスト

```go
func BenchmarkJSONMarshal(b *testing.B) {
    user := User{ID: "1", Name: "Alice", Email: "alice@example.com"}

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        json.Marshal(user)
    }
}

func BenchmarkJSONMarshalParallel(b *testing.B) {
    user := User{ID: "1", Name: "Alice", Email: "alice@example.com"}

    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            json.Marshal(user)
        }
    })
}
```

---

## 6. テスト手法の比較

| テスト手法 | 速度 | 外部依存 | 信頼性 | 保守コスト | 適用場面 |
|-----------|------|---------|--------|-----------|---------|
| 単体テスト（table-driven） | 非常に速い | なし | 高 | 低 | 関数・メソッド単位 |
| httptest.Recorder | 速い | なし | 高 | 低 | HTTPハンドラ |
| httptest.Server | 速い | なし | 中〜高 | 中 | HTTP クライアント |
| testify/mock | 速い | なし | 中 | 中 | 依存の多いコンポーネント |
| 統合テスト（DB込み） | 遅い | あり | 非常に高 | 高 | エンドツーエンド検証 |
| testcontainers | 遅い | Docker | 非常に高 | 高 | 本番に近い環境で検証 |

---

## 7. アンチパターン

### アンチパターン1: テストケース名の省略

```go
// NG: 失敗時にどのケースか分からない
func TestParse(t *testing.T) {
    tests := []struct {
        input string
        want  int
    }{
        {"42", 42},
        {"abc", 0},
    }
    for _, tt := range tests {
        // t.Run がない → 失敗時にケース特定が困難
        got, _ := Parse(tt.input)
        if got != tt.want {
            t.Errorf("Parse(%q) = %d, want %d", tt.input, got, tt.want)
        }
    }
}

// OK: t.Run で各ケースに名前をつける
func TestParse(t *testing.T) {
    tests := []struct {
        name  string
        input string
        want  int
    }{
        {"数値文字列", "42", 42},
        {"非数値", "abc", 0},
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, _ := Parse(tt.input)
            assert.Equal(t, tt.want, got)
        })
    }
}
```

### アンチパターン2: テスト内でのsleep依存

```go
// NG: 時間依存でフレーキーテストになる
func TestAsyncProcess(t *testing.T) {
    go StartProcess()
    time.Sleep(2 * time.Second) // 環境によって不安定
    assert.True(t, IsProcessDone())
}

// OK: チャネルやコンテキストで同期を取る
func TestAsyncProcess(t *testing.T) {
    done := make(chan struct{})
    go func() {
        StartProcess()
        close(done)
    }()

    select {
    case <-done:
        assert.True(t, IsProcessDone())
    case <-time.After(5 * time.Second):
        t.Fatal("タイムアウト: プロセスが完了しなかった")
    }
}
```

---

## FAQ

### Q1. `testing.T` の `Error` と `Fatal` の違いは？

`t.Error` / `t.Errorf` はテストを失敗としてマークするが処理は続行する。`t.Fatal` / `t.Fatalf` はテストを即座に中断する。後続の検証が前提条件に依存する場合は `Fatal` を使い、独立した検証には `Error` を使う。testify では `assert` が `Error` 相当、`require` が `Fatal` 相当。

### Q2. テストヘルパー関数に `t.Helper()` は必要？

必要。`t.Helper()` を呼ぶと、テスト失敗時のエラー出力でヘルパー関数ではなく呼び出し元の行番号が表示される。デバッグ効率が大幅に上がるため、テストユーティリティ関数には必ず付ける。

```go
func assertJSON(t *testing.T, body []byte, key, want string) {
    t.Helper() // これがないとこの関数の行番号が表示される
    var m map[string]string
    require.NoError(t, json.Unmarshal(body, &m))
    assert.Equal(t, want, m[key])
}
```

### Q3. テストのカバレッジは何%を目指すべき？

一般に70〜80%が現実的な目標。100%を目指すとテストの保守コストが跳ね上がり、些末なコードにまでテストを書くことになる。重要なのはクリティカルパス（ビジネスロジック、エラーハンドリング）のカバレッジを高くすること。

---

## まとめ

| 概念 | 要点 |
|------|------|
| table-driven tests | テストケースをスライスで定義し `t.Run` でループ実行 |
| t.Parallel() | サブテストの並列実行で高速化 |
| testify/assert | 失敗しても続行する柔軟なアサーション |
| testify/require | 前提条件の検証に使う即中断アサーション |
| testify/mock | インターフェースベースのモック生成 |
| httptest.NewRecorder | ハンドラのユニットテスト |
| httptest.NewServer | 外部API呼び出しのモック |
| カバレッジ | `go test -cover` で測定、70-80%が現実的目標 |
| ベンチマーク | `Benchmark` プレフィックスで性能測定 |

---

## 次に読むべきガイド

- **03-tools/00-cli-development.md** — CLI開発：cobra、flag、promptui
- **03-tools/02-profiling.md** — プロファイリング：pprof、trace
- **03-tools/04-best-practices.md** — ベストプラクティス：Effective Go

---

## 参考文献

1. **Go公式 — Testing パッケージ** https://pkg.go.dev/testing
2. **stretchr/testify GitHub** https://github.com/stretchr/testify
3. **Go Blog — Using Subtests and Sub-benchmarks** https://go.dev/blog/subtests
4. **Dave Cheney — Writing Table Driven Tests in Go** https://dave.cheney.net/2019/05/07/prefer-table-driven-tests
