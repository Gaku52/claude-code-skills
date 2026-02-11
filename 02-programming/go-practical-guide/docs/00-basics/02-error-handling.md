# エラーハンドリング -- Goのエラー設計哲学

> Goはerror interfaceを中心とした明示的なエラーハンドリングを採用し、errors.Is/As・sentinel errors・wrappingで堅牢なエラー伝搬を実現する。

---

## この章で学ぶこと

1. **error interface** -- Go のエラーが単なるインターフェースである理由
2. **errors.Is / errors.As** -- エラーチェーンの検査方法
3. **エラーラッピング** -- `fmt.Errorf("%w", err)` による文脈の追加

---

## 1. error interface の基本

### コード例 1: error interface

```go
// error は組み込みインターフェース
type error interface {
    Error() string
}

// カスタムエラー型
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: %s - %s", e.Field, e.Message)
}
```

### コード例 2: sentinel errors

```go
var (
    ErrNotFound     = errors.New("not found")
    ErrUnauthorized = errors.New("unauthorized")
    ErrConflict     = errors.New("conflict")
)

func FindUser(id int) (*User, error) {
    user, exists := db[id]
    if !exists {
        return nil, ErrNotFound
    }
    return user, nil
}
```

### コード例 3: エラーラッピング

```go
func GetUserProfile(id int) (*Profile, error) {
    user, err := FindUser(id)
    if err != nil {
        return nil, fmt.Errorf("get user profile (id=%d): %w", id, err)
    }
    profile, err := loadProfile(user)
    if err != nil {
        return nil, fmt.Errorf("load profile for %s: %w", user.Name, err)
    }
    return profile, nil
}
```

### コード例 4: errors.Is と errors.As

```go
func handleError(err error) {
    // errors.Is: エラーチェーンに特定のエラーが含まれるか
    if errors.Is(err, ErrNotFound) {
        fmt.Println("リソースが見つかりません")
        return
    }

    // errors.As: エラーチェーンから特定の型を取り出す
    var ve *ValidationError
    if errors.As(err, &ve) {
        fmt.Printf("バリデーションエラー: フィールド=%s\n", ve.Field)
        return
    }

    fmt.Printf("予期しないエラー: %v\n", err)
}
```

### コード例 5: 複数エラーの結合 (Go 1.20+)

```go
func validateUser(u *User) error {
    var errs []error

    if u.Name == "" {
        errs = append(errs, &ValidationError{Field: "name", Message: "required"})
    }
    if u.Email == "" {
        errs = append(errs, &ValidationError{Field: "email", Message: "required"})
    }
    if len(u.Password) < 8 {
        errs = append(errs, &ValidationError{Field: "password", Message: "too short"})
    }

    return errors.Join(errs...) // Go 1.20+: 複数エラーを結合
}
```

### コード例 6: カスタムエラー型にUnwrapを実装

```go
type AppError struct {
    Code    int
    Message string
    Err     error
}

func (e *AppError) Error() string {
    return fmt.Sprintf("[%d] %s: %v", e.Code, e.Message, e.Err)
}

func (e *AppError) Unwrap() error {
    return e.Err
}

// 使用例
err := &AppError{
    Code:    404,
    Message: "user not found",
    Err:     ErrNotFound,
}
fmt.Println(errors.Is(err, ErrNotFound)) // true
```

---

## 2. ASCII図解

### 図1: エラーチェーン

```
fmt.Errorf("handler: %w",
  fmt.Errorf("service: %w",
    fmt.Errorf("repo: %w",
      ErrNotFound)))

エラーチェーン:
┌─────────────────┐
│ "handler: ..."  │
│   Unwrap() ─────┼──> ┌──────────────────┐
└─────────────────┘    │ "service: ..."   │
                       │   Unwrap() ──────┼──> ┌────────────────┐
                       └──────────────────┘    │ "repo: ..."    │
                                               │   Unwrap() ────┼──> ErrNotFound
                                               └────────────────┘
errors.Is(err, ErrNotFound) → チェーンを辿って true
```

### 図2: errors.Is vs errors.As

```
┌─────────────────────────────────────────────┐
│              errors.Is(err, target)          │
│  目的: 特定のエラー値と一致するか検査           │
│  探索: Unwrap()を再帰的に辿る                  │
│  比較: == または Is()メソッド                   │
│  戻値: bool                                   │
├─────────────────────────────────────────────┤
│              errors.As(err, &target)         │
│  目的: 特定のエラー型を取り出す                 │
│  探索: Unwrap()を再帰的に辿る                  │
│  比較: 型アサーション                           │
│  戻値: bool (targetに値がセットされる)          │
└─────────────────────────────────────────────┘
```

### 図3: エラーハンドリングの判断フロー

```
         エラーが発生
              │
              ▼
     ┌────────────────┐
     │ リカバリ可能？   │
     └───┬────────┬───┘
        YES      NO
         │        │
         ▼        ▼
   ┌──────────┐ ┌──────────┐
   │ ログ出力  │ │ %wで     │
   │ デフォルト│ │ ラップして│
   │ 値を返す  │ │ 上位に返す│
   └──────────┘ └──────────┘
         │
         ▼
   ┌──────────────┐
   │ 文脈情報を    │
   │ 追加して返す   │
   └──────────────┘
```

---

## 3. 比較表

### 表1: エラー処理アプローチ比較

| アプローチ | Go | Java | Rust | Python |
|-----------|-----|------|------|--------|
| 仕組み | 戻り値 (error) | 例外 (Exception) | Result<T,E> | 例外 (Exception) |
| 未処理時の動作 | コンパイルは通る | クラッシュ | コンパイルエラー | クラッシュ |
| 型情報 | interface (動的) | クラス階層 | enum (静的) | クラス階層 |
| 網羅性チェック | なし | なし (checked除く) | あり (match) | なし |
| 制御フロー | 明示的 if err != nil | try-catch | ? 演算子 | try-except |

### 表2: Go エラーパターン比較

| パターン | 用途 | 例 |
|---------|------|-----|
| sentinel error | 既知のエラー条件 | `ErrNotFound` |
| カスタムエラー型 | 追加情報が必要 | `*ValidationError` |
| `fmt.Errorf("%w")` | 文脈追加 | `"open config: %w"` |
| `errors.Join` | 複数エラー集約 | バリデーション |
| panic/recover | 本当に回復不能な状態 | プログラミングエラー |

---

## 4. アンチパターン

### アンチパターン 1: エラーを握りつぶす

```go
// BAD: エラーを無視
result, _ := doSomething()

// BAD: エラーをログだけして処理しない
if err != nil {
    log.Println(err) // 呼び出し元は成功したと思う
}

// GOOD: エラーを返す
result, err := doSomething()
if err != nil {
    return fmt.Errorf("do something: %w", err)
}
```

### アンチパターン 2: エラーメッセージの重複

```go
// BAD: "failed to" が連鎖して冗長
// "failed to get user: failed to query db: failed to connect: timeout"
return fmt.Errorf("failed to get user: %w", err)

// GOOD: 簡潔に文脈を追加
// "get user: query db: connect: timeout"
return fmt.Errorf("get user: %w", err)
```

---

## 5. FAQ

### Q1: panicはいつ使うべきか？

panicはプログラミングエラー（nilポインタ参照、範囲外アクセス等）や、回復不能な初期化エラーのみに使う。通常のビジネスロジックでは必ず `error` を返す。ライブラリはpanicを呼び出し元に漏らしてはならない。

### Q2: `%w` と `%v` の違いは？

`%w` はエラーをラップし、`errors.Is`/`errors.As` でチェーン検査可能にする。`%v` は単にエラーメッセージを文字列として埋め込む。原則として `%w` を使うが、内部実装を隠蔽したい場合（ライブラリの公開API等）は `%v` を使う。

### Q3: エラーメッセージの命名規則は？

Go の慣例: (1) 小文字で始める、(2) "failed to" を付けない、(3) パッケージ名をプレフィックスにしない（ラップで自然に付く）、(4) 句読点で終わらない。例: `"open config file: %w"` が良い形。

---

## まとめ

| 概念 | 要点 |
|------|------|
| error interface | `Error() string` を持つインターフェース |
| sentinel error | `var ErrXxx = errors.New(...)` で定義 |
| ラッピング | `fmt.Errorf("context: %w", err)` |
| errors.Is | エラーチェーンに特定のエラーが含まれるか |
| errors.As | エラーチェーンから特定の型を取り出す |
| errors.Join | 複数エラーの結合 (Go 1.20+) |
| panic/recover | 回復不能なエラーのみ |

---

## 次に読むべきガイド

- [03-packages-modules.md](./03-packages-modules.md) -- パッケージとモジュール
- [../02-web/04-testing.md](../02-web/04-testing.md) -- テストにおけるエラー検証
- [../03-tools/04-best-practices.md](../03-tools/04-best-practices.md) -- ベストプラクティス

---

## 参考文献

1. **Go Blog, "Working with Errors in Go 1.13"** -- https://go.dev/blog/go1.13-errors
2. **Go Blog, "Error handling and Go"** -- https://go.dev/blog/error-handling-and-go
3. **Standard library: errors package** -- https://pkg.go.dev/errors
