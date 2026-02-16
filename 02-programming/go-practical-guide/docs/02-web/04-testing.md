# Go テスト完全ガイド

> table-driven tests、testify、httptestを駆使してGoコードの品質を保証する実践的テスト手法

## この章で学ぶこと

1. **table-driven tests** のパターンを使い、網羅的かつ保守しやすいテストを書く方法
2. **testify** ライブラリによるアサーション・モック・スイートの活用法
3. **httptest** パッケージでHTTPハンドラとクライアントをテストする技法
4. **統合テスト** とテストヘルパーの設計パターン
5. **テストカバレッジ** とベンチマークの実践的活用

---

## 1. Goテストの基本構造

### 1-1. テストファイルの命名規則

```
project/
├── handler.go
├── handler_test.go      ← 同一パッケージ
├── service.go
├── service_test.go
├── handler_integration_test.go  ← 統合テスト
└── testdata/                    ← テストデータ用ディレクトリ
    ├── golden_response.json
    └── fixtures/
        ├── users.json
        └── config.yaml
```

Goのテストは `_test.go` サフィックスを持つファイルに記述する。テスト関数は `Test` プレフィックスで始まり、`*testing.T` を引数に取る。`testdata` ディレクトリはGoのビルドシステムから無視されるため、テスト用のフィクスチャやゴールデンファイルを格納するのに最適である。

### 1-2. テストパッケージの選択

テストファイルのパッケージ名には2つの方式がある。

```go
// 方式1: 同一パッケージテスト（ホワイトボックステスト）
// handler_test.go
package myapp

// プライベートな関数やフィールドにアクセスできる
func TestInternalLogic(t *testing.T) {
    result := internalHelper("input")  // 非公開関数を直接テスト
    if result != "expected" {
        t.Errorf("internalHelper() = %q, want %q", result, "expected")
    }
}

// 方式2: 外部パッケージテスト（ブラックボックステスト）
// handler_test.go
package myapp_test

import "myproject/myapp"

// 公開APIのみをテスト（ユーザー視点の検証）
func TestPublicAPI(t *testing.T) {
    result := myapp.Process("input")
    if result != "expected" {
        t.Errorf("Process() = %q, want %q", result, "expected")
    }
}
```

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

### 1-3. テスト実行の主要フラグ

```bash
# 基本実行
go test ./...                       # 全パッケージのテスト実行
go test -v ./...                    # 詳細出力
go test -run TestAdd ./...          # 特定テストのみ実行
go test -run TestDivide/ゼロ除算    # サブテスト指定

# 並列・タイムアウト制御
go test -parallel 4 ./...           # 並列実行数を指定
go test -timeout 60s ./...          # テスト全体のタイムアウト
go test -count=5 ./...              # 繰り返し実行（キャッシュ無効化）

# カバレッジ
go test -cover ./...                # カバレッジ付き実行
go test -coverprofile=coverage.out  # プロファイル出力
go test -covermode=atomic ./...     # 並行テスト対応のカバレッジ

# レース検出
go test -race ./...                 # データレース検出
go test -race -count=10 ./...       # 繰り返しでレース検出精度向上

# ビルドタグ
go test -tags=integration ./...     # タグ指定テスト
go test -short ./...                # 短縮テストモード
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
        {name: "非常に小さい数", a: 1, b: 1000000, want: 0.000001, wantError: false},
        {name: "非常に大きい数", a: 1e18, b: 1e9, want: 1e9, wantError: false},
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

### コード例4: セットアップ/ティアダウン付きtable-driven test

```go
func TestDatabaseOperations(t *testing.T) {
    tests := []struct {
        name    string
        setup   func(db *sql.DB)    // テスト前のセットアップ
        action  func(db *sql.DB) error
        verify  func(t *testing.T, db *sql.DB)
        cleanup func(db *sql.DB)    // テスト後のクリーンアップ
    }{
        {
            name: "ユーザー作成",
            setup: func(db *sql.DB) {
                db.Exec("DELETE FROM users")
            },
            action: func(db *sql.DB) error {
                _, err := db.Exec("INSERT INTO users (name, email) VALUES (?, ?)",
                    "Alice", "alice@example.com")
                return err
            },
            verify: func(t *testing.T, db *sql.DB) {
                var count int
                db.QueryRow("SELECT COUNT(*) FROM users").Scan(&count)
                if count != 1 {
                    t.Errorf("ユーザー数 = %d, want 1", count)
                }
            },
            cleanup: func(db *sql.DB) {
                db.Exec("DELETE FROM users")
            },
        },
        {
            name: "重複メール拒否",
            setup: func(db *sql.DB) {
                db.Exec("DELETE FROM users")
                db.Exec("INSERT INTO users (name, email) VALUES (?, ?)",
                    "Alice", "alice@example.com")
            },
            action: func(db *sql.DB) error {
                _, err := db.Exec("INSERT INTO users (name, email) VALUES (?, ?)",
                    "Bob", "alice@example.com")
                return err
            },
            verify: func(t *testing.T, db *sql.DB) {
                // エラーが発生することを検証（action の戻り値で確認）
            },
            cleanup: func(db *sql.DB) {
                db.Exec("DELETE FROM users")
            },
        },
    }

    db := setupTestDB(t) // テスト用DB接続
    defer db.Close()

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            if tt.setup != nil {
                tt.setup(db)
            }
            if tt.cleanup != nil {
                defer tt.cleanup(db)
            }

            err := tt.action(db)
            if tt.verify != nil {
                tt.verify(t, db)
            }
            _ = err // 必要に応じてエラー検証
        })
    }
}
```

### コード例5: カスタムマッチャー付きtable-driven test

```go
func TestParseConfig(t *testing.T) {
    tests := []struct {
        name    string
        input   string
        check   func(t *testing.T, cfg *Config, err error) // カスタム検証関数
    }{
        {
            name:  "完全な設定",
            input: `{"host": "localhost", "port": 8080, "debug": true}`,
            check: func(t *testing.T, cfg *Config, err error) {
                t.Helper()
                require.NoError(t, err)
                assert.Equal(t, "localhost", cfg.Host)
                assert.Equal(t, 8080, cfg.Port)
                assert.True(t, cfg.Debug)
            },
        },
        {
            name:  "デフォルト値適用",
            input: `{}`,
            check: func(t *testing.T, cfg *Config, err error) {
                t.Helper()
                require.NoError(t, err)
                assert.Equal(t, "0.0.0.0", cfg.Host, "デフォルトホスト")
                assert.Equal(t, 3000, cfg.Port, "デフォルトポート")
                assert.False(t, cfg.Debug, "デフォルトはdebug無効")
            },
        },
        {
            name:  "不正なJSON",
            input: `{invalid`,
            check: func(t *testing.T, cfg *Config, err error) {
                t.Helper()
                require.Error(t, err)
                assert.Nil(t, cfg)
                assert.Contains(t, err.Error(), "invalid")
            },
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            cfg, err := ParseConfig([]byte(tt.input))
            tt.check(t, cfg, err)
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

### コード例6: testify/assert と testify/require

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

### testify の主要アサーション一覧

```go
// 等値比較
assert.Equal(t, expected, actual)           // DeepEqual比較
assert.NotEqual(t, unexpected, actual)
assert.EqualValues(t, expected, actual)     // 型変換込みの比較

// nil / empty チェック
assert.Nil(t, obj)
assert.NotNil(t, obj)
assert.Empty(t, collection)                // len == 0
assert.NotEmpty(t, collection)

// 真偽
assert.True(t, condition)
assert.False(t, condition)

// エラー
assert.NoError(t, err)
assert.Error(t, err)
assert.ErrorIs(t, err, ErrNotFound)        // errors.Is 相当
assert.ErrorAs(t, err, &target)            // errors.As 相当
assert.ErrorContains(t, err, "not found")

// コレクション
assert.Contains(t, list, element)
assert.NotContains(t, list, element)
assert.Len(t, list, expectedLen)
assert.ElementsMatch(t, expected, actual)  // 順序無視の比較

// 文字列
assert.Contains(t, str, substring)
assert.Regexp(t, regexp, str)

// 数値
assert.Greater(t, a, b)
assert.GreaterOrEqual(t, a, b)
assert.InDelta(t, expected, actual, delta)  // 浮動小数点の近似比較

// パニック
assert.Panics(t, func() { panicFunc() })
assert.NotPanics(t, func() { safeFunc() })

// JSON
assert.JSONEq(t, expectedJSON, actualJSON)  // JSON文字列の意味的比較

// 時間
assert.WithinDuration(t, expected, actual, delta)
```

### コード例7: testify/mock

```go
// インターフェース定義
type UserRepository interface {
    FindByID(id string) (*User, error)
    FindByEmail(email string) (*User, error)
    Save(user *User) error
    Delete(id string) error
    List(offset, limit int) ([]*User, error)
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

func (m *MockUserRepo) FindByEmail(email string) (*User, error) {
    args := m.Called(email)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).(*User), args.Error(1)
}

func (m *MockUserRepo) Save(user *User) error {
    args := m.Called(user)
    return args.Error(0)
}

func (m *MockUserRepo) Delete(id string) error {
    args := m.Called(id)
    return args.Error(0)
}

func (m *MockUserRepo) List(offset, limit int) ([]*User, error) {
    args := m.Called(offset, limit)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).([]*User), args.Error(1)
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

// より高度なモックパターン
func TestUserServiceEdgeCases(t *testing.T) {
    t.Run("ユーザーが見つからない場合", func(t *testing.T) {
        mockRepo := new(MockUserRepo)
        mockRepo.On("FindByID", "999").Return(nil, ErrNotFound)

        service := NewUserService(mockRepo)
        err := service.UpdateName("999", "Bob")

        assert.ErrorIs(t, err, ErrNotFound)
        mockRepo.AssertNotCalled(t, "Save")
    })

    t.Run("保存失敗時のロールバック", func(t *testing.T) {
        mockRepo := new(MockUserRepo)
        existingUser := &User{ID: "123", Name: "Alice"}
        mockRepo.On("FindByID", "123").Return(existingUser, nil)
        mockRepo.On("Save", mock.Anything).Return(errors.New("db error"))

        service := NewUserService(mockRepo)
        err := service.UpdateName("123", "Bob")

        assert.Error(t, err)
        assert.Contains(t, err.Error(), "db error")
    })

    t.Run("呼び出し回数の検証", func(t *testing.T) {
        mockRepo := new(MockUserRepo)
        users := []*User{
            {ID: "1", Name: "Alice"},
            {ID: "2", Name: "Bob"},
        }
        mockRepo.On("List", 0, 10).Return(users, nil).Once()

        service := NewUserService(mockRepo)
        result, _ := service.ListUsers(0, 10)

        assert.Len(t, result, 2)
        mockRepo.AssertNumberOfCalls(t, "List", 1)
    })
}
```

### コード例8: testify/suite

```go
package user_test

import (
    "database/sql"
    "testing"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/suite"
)

// テストスイートの定義
type UserServiceSuite struct {
    suite.Suite
    db      *sql.DB
    service *UserService
    repo    *UserRepo
}

// スイート開始前に一度だけ実行
func (s *UserServiceSuite) SetupSuite() {
    db, err := sql.Open("sqlite3", ":memory:")
    s.Require().NoError(err)
    s.db = db

    // テーブル作成
    _, err = db.Exec(`
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    `)
    s.Require().NoError(err)

    s.repo = NewUserRepo(db)
    s.service = NewUserService(s.repo)
}

// 各テスト前に実行
func (s *UserServiceSuite) SetupTest() {
    s.db.Exec("DELETE FROM users")
}

// 各テスト後に実行
func (s *UserServiceSuite) TearDownTest() {
    // 必要に応じてクリーンアップ
}

// スイート終了時に一度だけ実行
func (s *UserServiceSuite) TearDownSuite() {
    s.db.Close()
}

// テストケース
func (s *UserServiceSuite) TestCreateUser() {
    user, err := s.service.Create("Alice", "alice@example.com")
    s.NoError(err)
    s.NotEmpty(user.ID)
    s.Equal("Alice", user.Name)
}

func (s *UserServiceSuite) TestCreateDuplicateEmail() {
    _, err := s.service.Create("Alice", "alice@example.com")
    s.NoError(err)

    _, err = s.service.Create("Bob", "alice@example.com")
    s.Error(err)
    s.ErrorIs(err, ErrDuplicate)
}

func (s *UserServiceSuite) TestFindUser() {
    created, _ := s.service.Create("Alice", "alice@example.com")

    found, err := s.service.FindByID(created.ID)
    s.NoError(err)
    s.Equal(created.ID, found.ID)
    s.Equal("Alice", found.Name)
}

func (s *UserServiceSuite) TestDeleteUser() {
    created, _ := s.service.Create("Alice", "alice@example.com")

    err := s.service.Delete(created.ID)
    s.NoError(err)

    _, err = s.service.FindByID(created.ID)
    s.ErrorIs(err, ErrNotFound)
}

// スイート実行のエントリポイント
func TestUserServiceSuite(t *testing.T) {
    suite.Run(t, new(UserServiceSuite))
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

### コード例9: httptest.NewRecorder でハンドラテスト

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

### コード例10: JSONリクエスト・レスポンスの完全テスト

```go
func TestCreateUserHandler(t *testing.T) {
    tests := []struct {
        name       string
        body       interface{}
        wantStatus int
        wantBody   map[string]interface{}
    }{
        {
            name:       "正常作成",
            body:       map[string]string{"name": "Alice", "email": "alice@example.com"},
            wantStatus: http.StatusCreated,
            wantBody:   map[string]interface{}{"name": "Alice", "email": "alice@example.com"},
        },
        {
            name:       "名前が空",
            body:       map[string]string{"name": "", "email": "alice@example.com"},
            wantStatus: http.StatusBadRequest,
            wantBody:   map[string]interface{}{"error": "name is required"},
        },
        {
            name:       "メールが不正",
            body:       map[string]string{"name": "Alice", "email": "invalid"},
            wantStatus: http.StatusBadRequest,
            wantBody:   map[string]interface{}{"error": "invalid email format"},
        },
        {
            name:       "不正なJSON",
            body:       "invalid json",
            wantStatus: http.StatusBadRequest,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            var bodyReader io.Reader
            switch v := tt.body.(type) {
            case string:
                bodyReader = strings.NewReader(v)
            default:
                jsonBytes, _ := json.Marshal(v)
                bodyReader = bytes.NewReader(jsonBytes)
            }

            req := httptest.NewRequest("POST", "/api/users", bodyReader)
            req.Header.Set("Content-Type", "application/json")
            rec := httptest.NewRecorder()

            handler := NewRouter() // テスト対象のルーター
            handler.ServeHTTP(rec, req)

            assert.Equal(t, tt.wantStatus, rec.Code)

            if tt.wantBody != nil {
                var got map[string]interface{}
                err := json.Unmarshal(rec.Body.Bytes(), &got)
                require.NoError(t, err)
                for key, want := range tt.wantBody {
                    assert.Equal(t, want, got[key], "フィールド %s が一致しない", key)
                }
            }
        })
    }
}
```

### コード例11: httptest.NewServer で外部APIモック

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

### コード例12: 複数エンドポイントのモックサーバー

```go
func TestExternalAPIClient(t *testing.T) {
    // 複数エンドポイントに対応するモックサーバー
    mockServer := httptest.NewServer(http.HandlerFunc(
        func(w http.ResponseWriter, r *http.Request) {
            switch {
            case r.Method == "GET" && r.URL.Path == "/api/users":
                // ユーザー一覧
                w.Header().Set("Content-Type", "application/json")
                json.NewEncoder(w).Encode([]map[string]interface{}{
                    {"id": 1, "name": "Alice"},
                    {"id": 2, "name": "Bob"},
                })

            case r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/api/users/"):
                // 個別ユーザー取得
                id := strings.TrimPrefix(r.URL.Path, "/api/users/")
                if id == "999" {
                    w.WriteHeader(http.StatusNotFound)
                    json.NewEncoder(w).Encode(map[string]string{"error": "not found"})
                    return
                }
                w.Header().Set("Content-Type", "application/json")
                json.NewEncoder(w).Encode(map[string]interface{}{
                    "id":   1,
                    "name": "Alice",
                })

            case r.Method == "POST" && r.URL.Path == "/api/users":
                // ユーザー作成
                var body map[string]string
                json.NewDecoder(r.Body).Decode(&body)
                w.WriteHeader(http.StatusCreated)
                json.NewEncoder(w).Encode(map[string]interface{}{
                    "id":   100,
                    "name": body["name"],
                })

            default:
                w.WriteHeader(http.StatusNotFound)
            }
        },
    ))
    defer mockServer.Close()

    client := NewAPIClient(mockServer.URL, "test-token")

    t.Run("ユーザー一覧取得", func(t *testing.T) {
        users, err := client.ListUsers()
        require.NoError(t, err)
        assert.Len(t, users, 2)
    })

    t.Run("存在しないユーザー", func(t *testing.T) {
        _, err := client.FetchUser(999)
        assert.ErrorIs(t, err, ErrNotFound)
    })

    t.Run("ユーザー作成", func(t *testing.T) {
        user, err := client.CreateUser("Charlie")
        require.NoError(t, err)
        assert.Equal(t, "Charlie", user.Name)
    })
}
```

### コード例13: ミドルウェアのテスト

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
        {"期限切れトークン", "expired-token", http.StatusUnauthorized},
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

// レート制限ミドルウェアのテスト
func TestRateLimitMiddleware(t *testing.T) {
    inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
    })

    // 1秒間に3回までのレート制限
    handler := RateLimitMiddleware(3, time.Second)(inner)

    // 3回は成功する
    for i := 0; i < 3; i++ {
        req := httptest.NewRequest("GET", "/api/data", nil)
        req.RemoteAddr = "192.168.1.1:12345"
        rec := httptest.NewRecorder()
        handler.ServeHTTP(rec, req)
        assert.Equal(t, http.StatusOK, rec.Code, "リクエスト %d", i+1)
    }

    // 4回目は制限にかかる
    req := httptest.NewRequest("GET", "/api/data", nil)
    req.RemoteAddr = "192.168.1.1:12345"
    rec := httptest.NewRecorder()
    handler.ServeHTTP(rec, req)
    assert.Equal(t, http.StatusTooManyRequests, rec.Code)
}

// CORSミドルウェアのテスト
func TestCORSMiddleware(t *testing.T) {
    inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
    })

    handler := CORSMiddleware(CORSConfig{
        AllowOrigins: []string{"https://example.com"},
        AllowMethods: []string{"GET", "POST"},
        AllowHeaders: []string{"Content-Type", "Authorization"},
    })(inner)

    t.Run("プリフライトリクエスト", func(t *testing.T) {
        req := httptest.NewRequest("OPTIONS", "/api/data", nil)
        req.Header.Set("Origin", "https://example.com")
        req.Header.Set("Access-Control-Request-Method", "POST")
        rec := httptest.NewRecorder()

        handler.ServeHTTP(rec, req)

        assert.Equal(t, http.StatusNoContent, rec.Code)
        assert.Equal(t, "https://example.com",
            rec.Header().Get("Access-Control-Allow-Origin"))
    })

    t.Run("許可されていないオリジン", func(t *testing.T) {
        req := httptest.NewRequest("GET", "/api/data", nil)
        req.Header.Set("Origin", "https://evil.com")
        rec := httptest.NewRecorder()

        handler.ServeHTTP(rec, req)

        assert.Empty(t, rec.Header().Get("Access-Control-Allow-Origin"))
    })
}
```

### コード例14: TLSサーバーのテスト

```go
func TestHTTPSClient(t *testing.T) {
    // TLS付きのテストサーバー
    tlsServer := httptest.NewTLSServer(http.HandlerFunc(
        func(w http.ResponseWriter, r *http.Request) {
            w.WriteHeader(http.StatusOK)
            w.Write([]byte("secure"))
        },
    ))
    defer tlsServer.Close()

    // TLSサーバーのクライアントを取得
    client := tlsServer.Client()

    resp, err := client.Get(tlsServer.URL + "/secure")
    require.NoError(t, err)
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)
    assert.Equal(t, "secure", string(body))
}
```

---

## 5. テストヘルパーとユーティリティ

### コード例15: テストヘルパー関数

```go
// testhelper.go (テストパッケージ内)

// t.Helper() で呼び出し元の行番号を表示
func assertJSON(t *testing.T, body []byte, key, want string) {
    t.Helper() // これがないとこの関数の行番号が表示される
    var m map[string]string
    require.NoError(t, json.Unmarshal(body, &m))
    assert.Equal(t, want, m[key])
}

// テスト用HTTPリクエストビルダー
func newJSONRequest(t *testing.T, method, url string, body interface{}) *http.Request {
    t.Helper()
    var reader io.Reader
    if body != nil {
        jsonBytes, err := json.Marshal(body)
        require.NoError(t, err)
        reader = bytes.NewReader(jsonBytes)
    }
    req := httptest.NewRequest(method, url, reader)
    req.Header.Set("Content-Type", "application/json")
    return req
}

// テスト用DBセットアップ
func setupTestDB(t *testing.T) *sql.DB {
    t.Helper()
    db, err := sql.Open("sqlite3", ":memory:")
    require.NoError(t, err)

    // マイグレーション実行
    _, err = db.Exec(testSchema)
    require.NoError(t, err)

    // t.Cleanup で自動クリーンアップ（Go 1.14+）
    t.Cleanup(func() {
        db.Close()
    })

    return db
}

// テスト用一時ファイル
func createTempFile(t *testing.T, content string) string {
    t.Helper()
    f, err := os.CreateTemp("", "test-*")
    require.NoError(t, err)

    _, err = f.WriteString(content)
    require.NoError(t, err)
    f.Close()

    t.Cleanup(func() {
        os.Remove(f.Name())
    })

    return f.Name()
}

// テスト用環境変数設定
func setEnv(t *testing.T, key, value string) {
    t.Helper()
    original := os.Getenv(key)
    os.Setenv(key, value)
    t.Cleanup(func() {
        if original == "" {
            os.Unsetenv(key)
        } else {
            os.Setenv(key, original)
        }
    })
}
```

### コード例16: ゴールデンファイルテスト

```go
// ゴールデンファイルパターン: 期待出力をファイルに保存
var update = flag.Bool("update", false, "ゴールデンファイルを更新する")

func TestRenderTemplate(t *testing.T) {
    tests := []struct {
        name string
        data interface{}
    }{
        {"ユーザープロフィール", User{Name: "Alice", Age: 30}},
        {"空データ", User{}},
        {"日本語名", User{Name: "太郎", Age: 25}},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := RenderTemplate(tt.data)

            goldenFile := filepath.Join("testdata", t.Name()+".golden")

            if *update {
                // ゴールデンファイルを更新
                os.MkdirAll(filepath.Dir(goldenFile), 0755)
                os.WriteFile(goldenFile, []byte(got), 0644)
                return
            }

            // ゴールデンファイルと比較
            want, err := os.ReadFile(goldenFile)
            require.NoError(t, err, "ゴールデンファイルが見つかりません。-update フラグで生成してください")
            assert.Equal(t, string(want), got)
        })
    }
}
```

### コード例17: テスト用のタイムコントロール

```go
// インターフェースで時刻を抽象化
type Clock interface {
    Now() time.Time
}

// 本番用
type RealClock struct{}
func (RealClock) Now() time.Time { return time.Now() }

// テスト用
type FakeClock struct {
    current time.Time
}
func (c *FakeClock) Now() time.Time { return c.current }
func (c *FakeClock) Advance(d time.Duration) { c.current = c.current.Add(d) }

// 使用例
type TokenService struct {
    clock     Clock
    ttl       time.Duration
}

func (s *TokenService) IsExpired(token *Token) bool {
    return s.clock.Now().After(token.ExpiresAt)
}

// テスト
func TestTokenExpiry(t *testing.T) {
    fakeClock := &FakeClock{current: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)}
    service := &TokenService{clock: fakeClock, ttl: time.Hour}

    token := &Token{
        ExpiresAt: time.Date(2024, 1, 1, 1, 0, 0, 0, time.UTC),
    }

    // 有効期限内
    assert.False(t, service.IsExpired(token))

    // 時間を進める
    fakeClock.Advance(2 * time.Hour)

    // 有効期限切れ
    assert.True(t, service.IsExpired(token))
}
```

---

## 6. 統合テストとビルドタグ

### コード例18: ビルドタグによるテスト分離

```go
//go:build integration

package store_test

import (
    "database/sql"
    "os"
    "testing"

    _ "github.com/lib/pq"
)

func TestPostgresUserStore(t *testing.T) {
    dsn := os.Getenv("TEST_DATABASE_URL")
    if dsn == "" {
        t.Skip("TEST_DATABASE_URL が設定されていません")
    }

    db, err := sql.Open("postgres", dsn)
    require.NoError(t, err)
    defer db.Close()

    store := NewUserStore(db)

    t.Run("CRUD操作", func(t *testing.T) {
        // Create
        user, err := store.Create(&User{Name: "Alice", Email: "alice@test.com"})
        require.NoError(t, err)
        assert.NotEmpty(t, user.ID)

        // Read
        found, err := store.FindByID(user.ID)
        require.NoError(t, err)
        assert.Equal(t, "Alice", found.Name)

        // Update
        found.Name = "Alice Updated"
        err = store.Update(found)
        require.NoError(t, err)

        // Delete
        err = store.Delete(user.ID)
        require.NoError(t, err)

        _, err = store.FindByID(user.ID)
        assert.ErrorIs(t, err, ErrNotFound)
    })
}
```

```bash
# 通常テスト（統合テスト除外）
go test ./...

# 統合テスト込み
go test -tags=integration ./...

# 統合テストのみ
go test -tags=integration -run TestPostgres ./...
```

### コード例19: testing.Short() による短縮テスト

```go
func TestHeavyComputation(t *testing.T) {
    if testing.Short() {
        t.Skip("短縮モードでは重い計算テストをスキップ")
    }

    // 時間のかかるテスト
    result := HeavyComputation(largeDataset)
    assert.Equal(t, expectedResult, result)
}
```

```bash
# 短縮モード（CIの高速フィードバック用）
go test -short ./...

# 完全テスト
go test ./...
```

### コード例20: testcontainers によるDockerベーステスト

```go
package store_test

import (
    "context"
    "testing"

    "github.com/testcontainers/testcontainers-go"
    "github.com/testcontainers/testcontainers-go/modules/postgres"
    "github.com/testcontainers/testcontainers-go/wait"
)

func TestWithPostgresContainer(t *testing.T) {
    if testing.Short() {
        t.Skip("短縮モードではコンテナテストをスキップ")
    }

    ctx := context.Background()

    // PostgreSQLコンテナ起動
    container, err := postgres.RunContainer(ctx,
        testcontainers.WithImage("postgres:16-alpine"),
        postgres.WithDatabase("testdb"),
        postgres.WithUsername("test"),
        postgres.WithPassword("test"),
        testcontainers.WithWaitStrategy(
            wait.ForLog("database system is ready to accept connections").
                WithOccurrence(2),
        ),
    )
    require.NoError(t, err)
    defer container.Terminate(ctx)

    // 接続文字列取得
    connStr, err := container.ConnectionString(ctx, "sslmode=disable")
    require.NoError(t, err)

    // DBに接続してテスト実行
    db, err := sql.Open("postgres", connStr)
    require.NoError(t, err)
    defer db.Close()

    // マイグレーション実行
    runMigrations(t, db)

    // テスト実行
    store := NewUserStore(db)
    user, err := store.Create(&User{Name: "Alice", Email: "alice@test.com"})
    require.NoError(t, err)
    assert.NotEmpty(t, user.ID)
}
```

---

## 7. テストカバレッジとベンチマーク

### カバレッジ

```bash
# カバレッジ付きテスト実行
go test -cover ./...

# HTMLレポート生成
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html

# 関数ごとのカバレッジ
go tool cover -func=coverage.out

# 特定パッケージのカバレッジ
go test -coverprofile=coverage.out -coverpkg=./internal/... ./...

# カバレッジ閾値チェック（CIで使用）
COVERAGE=$(go test -cover ./... | grep -oP '\d+\.\d+%' | head -1 | tr -d '%')
if (( $(echo "$COVERAGE < 70" | bc -l) )); then
    echo "カバレッジが70%未満: $COVERAGE%"
    exit 1
fi
```

### コード例21: ベンチマークテスト

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

// メモリアロケーション計測
func BenchmarkStringConcat(b *testing.B) {
    b.Run("Plus演算子", func(b *testing.B) {
        b.ReportAllocs()
        for i := 0; i < b.N; i++ {
            s := ""
            for j := 0; j < 100; j++ {
                s += "a"
            }
        }
    })

    b.Run("strings.Builder", func(b *testing.B) {
        b.ReportAllocs()
        for i := 0; i < b.N; i++ {
            var sb strings.Builder
            for j := 0; j < 100; j++ {
                sb.WriteString("a")
            }
            _ = sb.String()
        }
    })

    b.Run("bytes.Buffer", func(b *testing.B) {
        b.ReportAllocs()
        for i := 0; i < b.N; i++ {
            var buf bytes.Buffer
            for j := 0; j < 100; j++ {
                buf.WriteString("a")
            }
            _ = buf.String()
        }
    })
}
```

```bash
# ベンチマーク実行
go test -bench=BenchmarkJSON -benchmem ./...

# 出力例:
# BenchmarkJSONMarshal-8       5000000    320 ns/op    128 B/op    2 allocs/op
# BenchmarkJSONMarshalParallel-8  20000000  85 ns/op   128 B/op    2 allocs/op

# ベンチマーク比較（benchstat）
go test -bench=. -count=10 > old.txt
# コード変更後
go test -bench=. -count=10 > new.txt
benchstat old.txt new.txt
```

### コード例22: サブベンチマークで入力サイズを変化させる

```go
func BenchmarkSort(b *testing.B) {
    sizes := []int{10, 100, 1000, 10000, 100000}

    for _, size := range sizes {
        b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
            // テストデータの準備（計測外）
            data := make([]int, size)
            for i := range data {
                data[i] = rand.Intn(size * 10)
            }

            b.ResetTimer()
            for i := 0; i < b.N; i++ {
                // データをコピーしてソート
                cp := make([]int, len(data))
                copy(cp, data)
                sort.Ints(cp)
            }
        })
    }
}
```

---

## 8. テスト手法の比較

| テスト手法 | 速度 | 外部依存 | 信頼性 | 保守コスト | 適用場面 |
|-----------|------|---------|--------|-----------|---------|
| 単体テスト（table-driven） | 非常に速い | なし | 高 | 低 | 関数・メソッド単位 |
| httptest.Recorder | 速い | なし | 高 | 低 | HTTPハンドラ |
| httptest.Server | 速い | なし | 中〜高 | 中 | HTTP クライアント |
| testify/mock | 速い | なし | 中 | 中 | 依存の多いコンポーネント |
| testify/suite | 速い | 任意 | 高 | 中 | 共通セットアップが必要な一連のテスト |
| ゴールデンファイル | 速い | なし | 高 | 低 | テンプレート出力、シリアライズ |
| 統合テスト（DB込み） | 遅い | あり | 非常に高 | 高 | エンドツーエンド検証 |
| testcontainers | 遅い | Docker | 非常に高 | 高 | 本番に近い環境で検証 |

---

## 9. テストダブルの比較

```
+--------------------------------------------------------------+
|                    テストダブルの種類                           |
+--------------------------------------------------------------+
|                                                              |
|  +----------+   +--------+   +------+   +------+   +------+ |
|  | Dummy    |   | Stub   |   | Spy  |   | Mock |   | Fake | |
|  | 使わない  |   | 固定値  |   | 記録  |   | 検証  |   | 簡易  | |
|  | 引数埋め  |   | を返す  |   | する  |   | する  |   | 実装  | |
|  +----------+   +--------+   +------+   +------+   +------+ |
|                                                              |
|  複雑さ:  低 ←←←←←←←←←←←←←←←←←←←→→→→→→→→→→→→→→→→→ 高     |
+--------------------------------------------------------------+
```

### コード例23: 各テストダブルの実装例

```go
// テスト対象のインターフェース
type EmailSender interface {
    Send(to, subject, body string) error
}

// Dummy: 引数を埋めるためだけに使う
type DummyEmailSender struct{}
func (d DummyEmailSender) Send(to, subject, body string) error { return nil }

// Stub: 固定値を返す
type StubEmailSender struct {
    Err error // 返すエラーを設定
}
func (s StubEmailSender) Send(to, subject, body string) error { return s.Err }

// Spy: 呼び出しを記録する
type SpyEmailSender struct {
    Calls []struct {
        To, Subject, Body string
    }
}
func (s *SpyEmailSender) Send(to, subject, body string) error {
    s.Calls = append(s.Calls, struct{ To, Subject, Body string }{to, subject, body})
    return nil
}

// Fake: 簡易実装（実際にメール送信する代わりにメモリに保存）
type FakeEmailSender struct {
    mu     sync.Mutex
    Inbox  map[string][]Message
}

type Message struct {
    Subject string
    Body    string
}

func NewFakeEmailSender() *FakeEmailSender {
    return &FakeEmailSender{Inbox: make(map[string][]Message)}
}

func (f *FakeEmailSender) Send(to, subject, body string) error {
    f.mu.Lock()
    defer f.mu.Unlock()
    f.Inbox[to] = append(f.Inbox[to], Message{Subject: subject, Body: body})
    return nil
}

// テストでの使い分け
func TestNotificationService(t *testing.T) {
    t.Run("Spy で送信内容を検証", func(t *testing.T) {
        spy := &SpyEmailSender{}
        service := NewNotificationService(spy)

        service.NotifyUser("alice@example.com", "Welcome!")

        require.Len(t, spy.Calls, 1)
        assert.Equal(t, "alice@example.com", spy.Calls[0].To)
        assert.Contains(t, spy.Calls[0].Subject, "Welcome")
    })

    t.Run("Stub でエラーケースを検証", func(t *testing.T) {
        stub := StubEmailSender{Err: errors.New("SMTP error")}
        service := NewNotificationService(stub)

        err := service.NotifyUser("alice@example.com", "Welcome!")
        assert.Error(t, err)
    })

    t.Run("Fake で複数メール送信を検証", func(t *testing.T) {
        fake := NewFakeEmailSender()
        service := NewNotificationService(fake)

        service.NotifyUser("alice@example.com", "Welcome!")
        service.NotifyUser("alice@example.com", "Update!")

        assert.Len(t, fake.Inbox["alice@example.com"], 2)
    })
}
```

---

## 10. テスト実行の最適化

### テスト実行速度の改善

```go
// TestMain でパッケージレベルのセットアップ/ティアダウン
func TestMain(m *testing.M) {
    // 全テスト前のセットアップ
    db := setupDatabase()
    seedTestData(db)

    // テスト実行
    code := m.Run()

    // 全テスト後のクリーンアップ
    db.Close()
    os.Exit(code)
}

// t.Parallel() による並列化
func TestParallelOperations(t *testing.T) {
    // 並列テストではt.Setenv()を使えない（Go 1.17+）
    // 代わりに共有リソースを避ける設計にする

    tests := []struct {
        name string
        fn   func(t *testing.T)
    }{
        {"テスト1", func(t *testing.T) { /* ... */ }},
        {"テスト2", func(t *testing.T) { /* ... */ }},
        {"テスト3", func(t *testing.T) { /* ... */ }},
    }

    for _, tt := range tests {
        tt := tt
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel()
            tt.fn(t)
        })
    }
}
```

### CI環境でのテスト設定

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'

      - name: 単体テスト
        run: go test -race -cover -short ./...

      - name: 統合テスト
        env:
          TEST_DATABASE_URL: postgres://test:test@localhost:5432/testdb?sslmode=disable
        run: go test -race -tags=integration ./...

      - name: カバレッジレポート
        run: |
          go test -coverprofile=coverage.out ./...
          go tool cover -func=coverage.out
```

---

## 11. アンチパターン

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

### アンチパターン3: テスト間の依存関係

```go
// NG: テストの実行順序に依存
var sharedState string

func TestStep1(t *testing.T) {
    sharedState = "initialized"
    // ...
}

func TestStep2(t *testing.T) {
    // sharedState が "initialized" であることを前提にしている
    // TestStep1 が先に実行されないと失敗する
    require.Equal(t, "initialized", sharedState)
}

// OK: 各テストは独立して実行可能
func TestStep1(t *testing.T) {
    state := setup()
    // ...
}

func TestStep2(t *testing.T) {
    state := setup() // 各テストで独立にセットアップ
    require.Equal(t, "initialized", state)
}
```

### アンチパターン4: 実装詳細のテスト

```go
// NG: 内部実装に密結合したテスト
func TestUserService_Create(t *testing.T) {
    service := NewUserService(repo)
    service.Create("Alice", "alice@example.com")

    // 内部キャッシュの状態を直接検証
    assert.Len(t, service.cache, 1)                    // 内部フィールドへのアクセス
    assert.Equal(t, "Alice", service.cache["alice"].Name) // 実装詳細に依存
}

// OK: 公開APIの振る舞いをテスト
func TestUserService_Create(t *testing.T) {
    service := NewUserService(repo)
    user, err := service.Create("Alice", "alice@example.com")
    require.NoError(t, err)

    // 公開APIで検証
    found, err := service.FindByEmail("alice@example.com")
    require.NoError(t, err)
    assert.Equal(t, user.ID, found.ID)
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

### Q4. t.Cleanup() と defer の違いは？

`t.Cleanup()` はGo 1.14で追加され、テスト関数の終了時（サブテストの終了時も含む）にクリーンアップ関数を登録する。`defer` との違いは、テストヘルパー関数内でリソースのクリーンアップを登録できる点。ヘルパー関数内の `defer` はヘルパー関数の終了時に実行されるが、`t.Cleanup()` はテスト全体の終了時に実行される。

```go
// defer はヘルパー関数終了時に実行される（意図と異なる）
func setupDB(t *testing.T) *sql.DB {
    db, _ := sql.Open("sqlite3", ":memory:")
    defer db.Close() // NG: この関数を抜けた時点でDBが閉じる
    return db
}

// t.Cleanup はテスト終了時に実行される（正しい）
func setupDB(t *testing.T) *sql.DB {
    t.Helper()
    db, _ := sql.Open("sqlite3", ":memory:")
    t.Cleanup(func() { db.Close() }) // OK: テスト終了時に閉じる
    return db
}
```

### Q5. フレーキーテスト（不安定なテスト）の対処法は？

フレーキーテストの主な原因と対策:
1. **時間依存** → `FakeClock` パターンで時刻を制御する
2. **並行処理の競合** → `-race` フラグで検出、同期プリミティブを使用
3. **外部サービス依存** → モック/テストコンテナで分離
4. **テスト間の共有状態** → 各テストを独立させる（`t.Parallel()` で発見しやすい）
5. **ランダム入力** → シード値を固定してログ出力（再現可能にする）

### Q6. go test のキャッシュを無効化するには？

`go test` はテスト結果をキャッシュする。キャッシュを無効化するには以下の方法がある。

```bash
# -count=1 でキャッシュ無効化（最も一般的）
go test -count=1 ./...

# キャッシュ全体をクリア
go clean -testcache

# 環境変数でも制御可能
GOFLAGS="-count=1" go test ./...
```

---

## まとめ

| 概念 | 要点 |
|------|------|
| table-driven tests | テストケースをスライスで定義し `t.Run` でループ実行 |
| t.Parallel() | サブテストの並列実行で高速化 |
| testify/assert | 失敗しても続行する柔軟なアサーション |
| testify/require | 前提条件の検証に使う即中断アサーション |
| testify/mock | インターフェースベースのモック生成 |
| testify/suite | セットアップ/ティアダウンの共有 |
| httptest.NewRecorder | ハンドラのユニットテスト |
| httptest.NewServer | 外部API呼び出しのモック |
| httptest.NewTLSServer | TLS通信のテスト |
| ゴールデンファイル | 期待出力をファイルで管理 |
| ビルドタグ | integration タグで統合テストを分離 |
| testcontainers | Dockerコンテナによる本番環境テスト |
| t.Cleanup() | テスト終了時のリソース解放を登録 |
| t.Helper() | ヘルパー関数でのエラー行番号を正しく表示 |
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
5. **testcontainers-go GitHub** https://github.com/testcontainers/testcontainers-go
6. **Go公式 — Code Coverage for Go Integration Tests** https://go.dev/blog/integration-test-coverage
