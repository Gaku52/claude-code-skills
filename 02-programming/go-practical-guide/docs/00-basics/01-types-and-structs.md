# 型とstruct -- Goの型システムを理解する

> Goは静的型付けと構造的部分型を採用し、struct・interface・type assertionで柔軟かつ安全なデータモデリングを実現する。

---

## この章で学ぶこと

1. **基本型と複合型** -- int/string/sliceからstructまで
2. **インターフェースと構造的部分型** -- 暗黙的なインターフェース満足
3. **type assertionとtype switch** -- 動的な型判定の安全な方法

---

## 1. 基本型

### コード例 1: 基本型の宣言

```go
var (
    i    int     = 42
    f    float64 = 3.14
    s    string  = "hello"
    b    bool    = true
    r    rune    = '日'    // int32のエイリアス
    by   byte    = 0xFF   // uint8のエイリアス
)
```

### コード例 2: struct定義とメソッド

```go
type User struct {
    ID        int
    Name      string
    Email     string
    CreatedAt time.Time
}

func (u User) DisplayName() string {
    return fmt.Sprintf("%s (%d)", u.Name, u.ID)
}

func (u *User) UpdateEmail(email string) {
    u.Email = email // ポインタレシーバで元の値を変更
}
```

### コード例 3: struct埋め込み (Embedding)

```go
type Animal struct {
    Name string
}

func (a Animal) Speak() string {
    return fmt.Sprintf("I am %s", a.Name)
}

type Dog struct {
    Animal        // 埋め込み（継承ではない）
    Breed  string
}

func main() {
    d := Dog{
        Animal: Animal{Name: "Pochi"},
        Breed:  "Shiba",
    }
    fmt.Println(d.Speak()) // Animal のメソッドに直接アクセス
}
```

### コード例 4: interface と構造的部分型

```go
type Stringer interface {
    String() string
}

type Point struct {
    X, Y int
}

// Point は Stringer を「明示的に宣言せず」満たす
func (p Point) String() string {
    return fmt.Sprintf("(%d, %d)", p.X, p.Y)
}

func PrintAnything(s Stringer) {
    fmt.Println(s.String())
}
```

### コード例 5: type assertion と type switch

```go
func describe(i interface{}) string {
    switch v := i.(type) {
    case int:
        return fmt.Sprintf("integer: %d", v)
    case string:
        return fmt.Sprintf("string: %q", v)
    case User:
        return fmt.Sprintf("user: %s", v.Name)
    default:
        return fmt.Sprintf("unknown: %T", v)
    }
}

// 安全な type assertion
func toInt(i interface{}) (int, bool) {
    v, ok := i.(int)
    return v, ok
}
```

### コード例 6: カスタム型と型変換

```go
type Celsius float64
type Fahrenheit float64

func (c Celsius) ToFahrenheit() Fahrenheit {
    return Fahrenheit(c*9/5 + 32)
}

type UserID int64
type OrderID int64

// コンパイルエラー: 型安全
// var uid UserID = OrderID(1) // cannot use OrderID(1) as type UserID
```

---

## 2. ASCII図解

### 図1: Goの型ヒエラルキー

```
                     ┌──────────┐
                     │   any    │
                     │(interface│
                     │   {})    │
                     └────┬─────┘
           ┌──────────────┼──────────────┐
     ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼──────┐
     │  基本型    │  │  複合型    │  │ 参照型     │
     │ int,float │  │ struct   │  │ slice,map │
     │ string    │  │ array    │  │ channel   │
     │ bool,rune │  │          │  │ pointer   │
     └───────────┘  └──────────┘  └───────────┘
```

### 図2: struct埋め込みのメモリレイアウト

```
Dog struct:
┌──────────────────────────────────┐
│  Animal (embedded)               │
│  ┌──────────────────────────┐    │
│  │ Name string              │    │
│  └──────────────────────────┘    │
│  Breed string                    │
└──────────────────────────────────┘

アクセス:
  d.Name   →  d.Animal.Name  (省略形)
  d.Speak() → d.Animal.Speak() (メソッド昇格)
```

### 図3: interface満足の仕組み

```
┌────────────────┐          ┌────────────────┐
│  io.Reader     │          │  *os.File      │
│  ┌────────────┐│          │  ┌────────────┐│
│  │ Read([]byte)│├─ 満たす ─┤  │ Read([]byte)││
│  │ (int, error)││  (暗黙)  │  │ (int, error)││
│  └────────────┘│          │  │ Write(...)  ││
└────────────────┘          │  │ Close()     ││
                            │  └────────────┘│
┌────────────────┐          └────────────────┘
│  io.ReadCloser │               ▲
│  ┌────────────┐│    こちらも    │
│  │ Read(...)   ││── 満たす ─────┘
│  │ Close()     ││
│  └────────────┘│
└────────────────┘
```

---

## 3. 比較表

### 表1: 値レシーバ vs ポインタレシーバ

| 項目 | 値レシーバ `(t T)` | ポインタレシーバ `(t *T)` |
|------|-------------------|------------------------|
| コピー | 呼び出し毎にコピー | ポインタのみコピー |
| 元の値の変更 | 不可 | 可能 |
| nil呼び出し | 不可 | 可能（要nilチェック） |
| interface満足 | T, *T 両方 | *T のみ |
| 推奨場面 | 小さいstruct、イミュータブル | 大きいstruct、ミュータブル |

### 表2: Goの型ゼロ値一覧

| 型 | ゼロ値 | 備考 |
|-----|-------|------|
| int, float64 | `0`, `0.0` | 数値型は全て0 |
| string | `""` | 空文字列 |
| bool | `false` | |
| pointer | `nil` | |
| slice | `nil` | `len()=0`, `cap()=0` |
| map | `nil` | 代入前にmake必要 |
| channel | `nil` | 送受信でブロック |
| struct | 各フィールドのゼロ値 | |
| interface | `nil` | type, value 両方 nil |

---

## 4. アンチパターン

### アンチパターン 1: 巨大インターフェース

```go
// BAD: 巨大なインターフェース
type Repository interface {
    FindUser(id int) (*User, error)
    CreateUser(u *User) error
    UpdateUser(u *User) error
    DeleteUser(id int) error
    FindOrder(id int) (*Order, error)
    CreateOrder(o *Order) error
    // ... 20個以上のメソッド
}

// GOOD: 小さなインターフェースに分割
type UserReader interface {
    FindUser(id int) (*User, error)
}

type UserWriter interface {
    CreateUser(u *User) error
    UpdateUser(u *User) error
    DeleteUser(id int) error
}
```

### アンチパターン 2: nilインターフェースの罠

```go
// BAD: nil ポインタを interface に入れると nil 判定が壊れる
func getUser() *User {
    return nil
}

func process() {
    var u interface{} = getUser()
    if u != nil {
        // ここに到達してしまう！ (interface{type:*User, value:nil})
        fmt.Println("not nil!") // 予期せず実行される
    }
}

// GOOD: 明示的にnilを返す
func getUserSafe() interface{} {
    u := findUser()
    if u == nil {
        return nil // interface自体がnilになる
    }
    return u
}
```

---

## 5. FAQ

### Q1: structに「コンストラクタ」はあるか？

Goにはコンストラクタ構文がないが、慣例として `New関数名` を使う。例: `func NewUser(name string) *User`。これはファクトリ関数パターンであり、バリデーションやデフォルト値の設定に適している。

### Q2: いつ値レシーバ、いつポインタレシーバを使うべきか？

原則: (1) structを変更するならポインタレシーバ、(2) structが大きい（フィールドが多い）ならポインタレシーバ、(3) 一貫性のため同一型のレシーバは統一する。迷ったらポインタレシーバを選ぶ。

### Q3: any型(interface{})を多用してよいか？

避けるべき。Go 1.18以降はジェネリクスが使えるため、`any` の利用は本当に任意の型を受け取る必要がある場面（JSONパース等）に限定する。型安全性を損なうため、可能な限り具体的な型やインターフェースを使う。

---

## まとめ

| 概念 | 要点 |
|------|------|
| 基本型 | int, float64, string, bool, rune, byte |
| struct | フィールドの集合。メソッドを持てる |
| 埋め込み | 継承ではなくコンポジション |
| interface | メソッドセット。暗黙的に満たされる |
| type assertion | `v, ok := i.(Type)` で安全に判定 |
| type switch | `switch v := i.(type)` で分岐 |
| ゼロ値 | 全ての型にゼロ値がある |

---

## 次に読むべきガイド

- [02-error-handling.md](./02-error-handling.md) -- エラーハンドリング
- [03-packages-modules.md](./03-packages-modules.md) -- パッケージとモジュール
- [../03-tools/01-generics.md](../03-tools/01-generics.md) -- ジェネリクス

---

## 参考文献

1. **The Go Programming Language Specification -- Types** -- https://go.dev/ref/spec#Types
2. **Effective Go -- Interfaces** -- https://go.dev/doc/effective_go#interfaces
3. **Go Blog, "The Laws of Reflection"** -- https://go.dev/blog/laws-of-reflection
