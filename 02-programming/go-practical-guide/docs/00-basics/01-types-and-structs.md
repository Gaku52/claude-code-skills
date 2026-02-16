# 型とstruct -- Goの型システムを理解する

> Goは静的型付けと構造的部分型を採用し、struct・interface・type assertionで柔軟かつ安全なデータモデリングを実現する。

---

## この章で学ぶこと

1. **基本型と複合型** -- int/string/sliceからstructまで
2. **インターフェースと構造的部分型** -- 暗黙的なインターフェース満足
3. **type assertionとtype switch** -- 動的な型判定の安全な方法
4. **ジェネリクスと型制約** -- Go 1.18+ の型パラメータ
5. **実践的なデータモデリング** -- ドメインモデルの設計手法

---

## 1. 基本型

### 1.1 数値型の詳細

Go は整数型と浮動小数点型に明確なビット幅を持たせている。これにより、プラットフォーム間での動作が予測可能になる。

| 型 | サイズ | 範囲 | 用途 |
|----|--------|------|------|
| `int8` | 1 byte | -128 ~ 127 | 省メモリが必要な場合 |
| `int16` | 2 bytes | -32768 ~ 32767 | 省メモリが必要な場合 |
| `int32` | 4 bytes | -2^31 ~ 2^31-1 | rune のエイリアス |
| `int64` | 8 bytes | -2^63 ~ 2^63-1 | タイムスタンプ、大きな整数 |
| `int` | 4 or 8 bytes | プラットフォーム依存 | 一般的な整数（デフォルト） |
| `uint8` | 1 byte | 0 ~ 255 | byte のエイリアス |
| `uint16` | 2 bytes | 0 ~ 65535 | ポート番号等 |
| `uint32` | 4 bytes | 0 ~ 2^32-1 | IPv4アドレス等 |
| `uint64` | 8 bytes | 0 ~ 2^64-1 | ハッシュ値等 |
| `float32` | 4 bytes | IEEE 754 単精度 | GPU演算、省メモリ |
| `float64` | 8 bytes | IEEE 754 倍精度 | 一般的な浮動小数点（デフォルト） |
| `complex64` | 8 bytes | float32 実部+虚部 | 信号処理等 |
| `complex128` | 16 bytes | float64 実部+虚部 | 科学計算等 |

### コード例 1: 基本型の宣言と操作

```go
package main

import (
    "fmt"
    "math"
    "unicode/utf8"
)

func main() {
    // 基本型の宣言
    var (
        i    int     = 42
        f    float64 = 3.14
        s    string  = "hello"
        b    bool    = true
        r    rune    = '日'    // int32のエイリアス
        by   byte    = 0xFF   // uint8のエイリアス
    )

    // 短縮宣言（型推論）
    x := 100           // int
    pi := 3.14159      // float64
    msg := "Go言語"     // string
    flag := true        // bool

    // 型変換（暗黙の変換は不可、明示的な変換が必要）
    var n int = 42
    var f64 float64 = float64(n)       // int → float64
    var n32 int32 = int32(n)           // int → int32
    var u uint = uint(n)               // int → uint（負の値に注意）

    // 文字列とバイト列の関係
    str := "Hello, 世界"
    bytes := []byte(str)               // 文字列 → バイトスライス
    runes := []rune(str)               // 文字列 → ルーンスライス

    fmt.Printf("len(str)=%d bytes\n", len(str))           // 13 bytes (UTF-8)
    fmt.Printf("rune count=%d\n", utf8.RuneCountInString(str)) // 9 文字
    fmt.Printf("runes=%v\n", runes)                       // Unicode コードポイント

    // 数値の限界値
    fmt.Printf("int8 max: %d\n", math.MaxInt8)
    fmt.Printf("int64 max: %d\n", math.MaxInt64)
    fmt.Printf("float64 max: %e\n", math.MaxFloat64)

    // ビット演算
    a := 0b1010  // 10
    b2 := 0b1100 // 12
    fmt.Printf("AND: %04b\n", a&b2)   // 1000
    fmt.Printf("OR:  %04b\n", a|b2)   // 1110
    fmt.Printf("XOR: %04b\n", a^b2)   // 0110
    fmt.Printf("NOT: %04b\n", ^a)     // ...0101

    _ = i; _ = f; _ = s; _ = b; _ = r; _ = by
    _ = x; _ = pi; _ = msg; _ = flag
    _ = f64; _ = n32; _ = u; _ = bytes
}
```

### 1.2 文字列の内部構造

Go の文字列は不変（immutable）なバイト列である。内部的には `string` は以下の構造を持つ:

```go
// runtime/string.go (概念的な構造)
type stringHeader struct {
    Data unsafe.Pointer  // バイト列へのポインタ
    Len  int             // バイト長
}
```

文字列の重要な特性:
- イミュータブル（一度作成したら変更不可）
- UTF-8 エンコーディング
- `len()` はバイト数を返す（文字数ではない）
- インデックスアクセスはバイト単位
- `range` ループはルーン（Unicodeコードポイント）単位

```go
package main

import (
    "fmt"
    "strings"
    "unicode/utf8"
)

func main() {
    s := "Go言語プログラミング"

    // バイト単位の操作
    fmt.Printf("len=%d bytes\n", len(s))  // バイト数

    // ルーン単位のイテレーション
    for i, r := range s {
        fmt.Printf("byte_offset=%d, rune=%c, unicode=%U\n", i, r, r)
    }

    // 文字列の連結
    // 少数の連結なら + 演算子で十分
    greeting := "Hello" + ", " + "World"

    // 大量の連結は strings.Builder を使う（効率的）
    var builder strings.Builder
    for i := 0; i < 1000; i++ {
        fmt.Fprintf(&builder, "item %d, ", i)
    }
    result := builder.String()
    _ = result

    // 文字列のスライス（バイト単位）
    sub := s[:2]  // "Go" (ASCII文字なのでバイトとルーンが一致)
    fmt.Println(sub)

    // マルチバイト文字を安全に扱う
    runes := []rune(s)
    first3 := string(runes[:3])  // "Go言"
    fmt.Println(first3)

    // 文字数のカウント
    fmt.Printf("rune count=%d\n", utf8.RuneCountInString(s))

    _ = greeting
}
```

### コード例 2: struct定義とメソッド

```go
package main

import (
    "encoding/json"
    "fmt"
    "time"
)

// User はユーザー情報を表す構造体
type User struct {
    ID        int       `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    Age       int       `json:"age,omitempty"`       // 0のとき省略
    IsAdmin   bool      `json:"is_admin"`
    CreatedAt time.Time `json:"created_at"`
    password  string    // 非公開フィールド（小文字始まり）
}

// NewUser はUserのファクトリ関数（コンストラクタ相当）
func NewUser(name, email string) *User {
    return &User{
        Name:      name,
        Email:     email,
        CreatedAt: time.Now(),
    }
}

// DisplayName は表示名を返す（値レシーバ -- 構造体を変更しない）
func (u User) DisplayName() string {
    return fmt.Sprintf("%s (%d)", u.Name, u.ID)
}

// UpdateEmail はメールアドレスを更新する（ポインタレシーバ -- 構造体を変更する）
func (u *User) UpdateEmail(email string) {
    u.Email = email
}

// SetPassword はパスワードを設定する
func (u *User) SetPassword(password string) {
    u.password = password // 非公開フィールドへのアクセス
}

// Validate はバリデーションを実行する
func (u User) Validate() error {
    if u.Name == "" {
        return fmt.Errorf("name is required")
    }
    if u.Email == "" {
        return fmt.Errorf("email is required")
    }
    if u.Age < 0 || u.Age > 200 {
        return fmt.Errorf("invalid age: %d", u.Age)
    }
    return nil
}

// String は fmt.Stringer インターフェースを実装
func (u User) String() string {
    return fmt.Sprintf("User{id=%d, name=%q, email=%q}", u.ID, u.Name, u.Email)
}

// MarshalJSON はカスタム JSON シリアライゼーション
func (u User) MarshalJSON() ([]byte, error) {
    type Alias User // 無限再帰を避ける
    return json.Marshal(&struct {
        Alias
        CreatedAtStr string `json:"created_at_formatted"`
    }{
        Alias:        Alias(u),
        CreatedAtStr: u.CreatedAt.Format("2006-01-02 15:04:05"),
    })
}

func main() {
    user := NewUser("Alice", "alice@example.com")
    user.ID = 1
    user.Age = 30
    user.SetPassword("secret123")

    fmt.Println(user)
    fmt.Println(user.DisplayName())

    if err := user.Validate(); err != nil {
        fmt.Printf("Validation error: %v\n", err)
    }

    // JSON シリアライゼーション
    data, _ := json.MarshalIndent(user, "", "  ")
    fmt.Println(string(data))

    // JSON デシリアライゼーション
    jsonStr := `{"id":2,"name":"Bob","email":"bob@example.com","age":25}`
    var user2 User
    json.Unmarshal([]byte(jsonStr), &user2)
    fmt.Println(user2)
}
```

### コード例 3: struct埋め込み (Embedding) と委譲

```go
package main

import (
    "fmt"
    "time"
)

// BaseModel は共通フィールドを提供する
type BaseModel struct {
    ID        int
    CreatedAt time.Time
    UpdatedAt time.Time
}

// BeforeSave は保存前にタイムスタンプを更新する
func (b *BaseModel) BeforeSave() {
    now := time.Now()
    if b.CreatedAt.IsZero() {
        b.CreatedAt = now
    }
    b.UpdatedAt = now
}

// Animal は動物の基本情報
type Animal struct {
    Name string
    Age  int
}

func (a Animal) Speak() string {
    return fmt.Sprintf("I am %s, %d years old", a.Name, a.Age)
}

func (a Animal) Move() string {
    return "moving..."
}

// Dog はAnimalを埋め込む（継承ではなくコンポジション）
type Dog struct {
    Animal        // 埋め込み
    Breed  string
}

// Speak はAnimalのSpeakをオーバーライド（メソッドの隠蔽）
func (d Dog) Speak() string {
    return fmt.Sprintf("Woof! I am %s the %s", d.Name, d.Breed)
}

// Cat もAnimalを埋め込む
type Cat struct {
    Animal
    Indoor bool
}

func (c Cat) Speak() string {
    return fmt.Sprintf("Meow! I am %s", c.Name)
}

// Product はBaseModelを埋め込む実践的な例
type Product struct {
    BaseModel
    Name     string
    Price    float64
    Category string
    Tags     []string
}

func (p *Product) Save() {
    p.BeforeSave() // BaseModelのメソッドが昇格して使える
    fmt.Printf("Saving product: %s (created=%v, updated=%v)\n",
        p.Name, p.CreatedAt.Format("15:04:05"), p.UpdatedAt.Format("15:04:05"))
}

// 複数の埋め込み
type Logger struct{}
func (l Logger) Log(msg string) { fmt.Printf("[LOG] %s\n", msg) }

type Metrics struct{}
func (m Metrics) Record(name string, value float64) {
    fmt.Printf("[METRIC] %s=%.2f\n", name, value)
}

type Service struct {
    Logger  // ロギング機能
    Metrics // メトリクス機能
    Name string
}

func main() {
    d := Dog{
        Animal: Animal{Name: "Pochi", Age: 3},
        Breed:  "Shiba",
    }
    fmt.Println(d.Speak())  // Dog.Speak() が呼ばれる
    fmt.Println(d.Move())   // Animal.Move() が昇格して呼ばれる
    fmt.Println(d.Name)     // d.Animal.Name の省略形
    fmt.Println(d.Animal.Speak()) // 元のAnimal.Speak()を明示的に呼ぶ

    p := &Product{
        Name:     "Go Book",
        Price:    4980,
        Category: "Books",
        Tags:     []string{"programming", "golang"},
    }
    p.Save()

    svc := Service{Name: "UserService"}
    svc.Log("service started")        // Logger.Log が昇格
    svc.Record("requests", 42.0)      // Metrics.Record が昇格
}
```

### コード例 4: interface と構造的部分型

```go
package main

import (
    "fmt"
    "io"
    "math"
    "strings"
)

// 小さなインターフェース（Go の推奨パターン）
type Stringer interface {
    String() string
}

type Area interface {
    Area() float64
}

type Perimeter interface {
    Perimeter() float64
}

// インターフェースの合成
type Shape interface {
    Area
    Perimeter
    Stringer
}

// 具象型: Circle
type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * math.Pi * c.Radius
}

func (c Circle) String() string {
    return fmt.Sprintf("Circle{radius=%.2f}", c.Radius)
}

// 具象型: Rectangle
type Rectangle struct {
    Width, Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

func (r Rectangle) String() string {
    return fmt.Sprintf("Rectangle{%.2f x %.2f}", r.Width, r.Height)
}

// 具象型: Triangle
type Triangle struct {
    A, B, C float64 // 3辺の長さ
}

func (t Triangle) Area() float64 {
    s := (t.A + t.B + t.C) / 2
    return math.Sqrt(s * (s - t.A) * (s - t.B) * (s - t.C))
}

func (t Triangle) Perimeter() float64 {
    return t.A + t.B + t.C
}

func (t Triangle) String() string {
    return fmt.Sprintf("Triangle{a=%.2f, b=%.2f, c=%.2f}", t.A, t.B, t.C)
}

// Shape を受け取る関数（多態性）
func printShapeInfo(s Shape) {
    fmt.Printf("%s: area=%.2f, perimeter=%.2f\n", s, s.Area(), s.Perimeter())
}

// Area だけを要求する関数（最小限のインターフェース）
func totalArea(shapes []Area) float64 {
    var total float64
    for _, s := range shapes {
        total += s.Area()
    }
    return total
}

// io.Reader を活用した例
func countLines(r io.Reader) (int, error) {
    buf := make([]byte, 32*1024)
    count := 0
    for {
        n, err := r.Read(buf)
        for i := 0; i < n; i++ {
            if buf[i] == '\n' {
                count++
            }
        }
        if err != nil {
            if err == io.EOF {
                return count, nil
            }
            return count, err
        }
    }
}

func main() {
    shapes := []Shape{
        Circle{Radius: 5},
        Rectangle{Width: 10, Height: 3},
        Triangle{A: 3, B: 4, C: 5},
    }

    for _, s := range shapes {
        printShapeInfo(s)
    }

    // Area インターフェースとして使う
    areas := make([]Area, len(shapes))
    for i, s := range shapes {
        areas[i] = s
    }
    fmt.Printf("Total area: %.2f\n", totalArea(areas))

    // io.Reader の活用
    text := "line 1\nline 2\nline 3\n"
    reader := strings.NewReader(text)
    lines, _ := countLines(reader)
    fmt.Printf("Lines: %d\n", lines)
}
```

### コード例 5: type assertion と type switch

```go
package main

import (
    "fmt"
    "io"
    "os"
    "strings"
)

// 実践的な type switch の活用
type Result struct {
    Value interface{}
    Err   error
}

func describe(i interface{}) string {
    switch v := i.(type) {
    case nil:
        return "nil"
    case int:
        return fmt.Sprintf("integer: %d", v)
    case int64:
        return fmt.Sprintf("int64: %d", v)
    case float64:
        return fmt.Sprintf("float64: %.2f", v)
    case string:
        return fmt.Sprintf("string: %q (len=%d)", v, len(v))
    case bool:
        return fmt.Sprintf("bool: %t", v)
    case []byte:
        return fmt.Sprintf("bytes: %x (len=%d)", v, len(v))
    case []int:
        return fmt.Sprintf("[]int: %v (len=%d)", v, len(v))
    case map[string]interface{}:
        return fmt.Sprintf("map: %d entries", len(v))
    case error:
        return fmt.Sprintf("error: %v", v)
    case fmt.Stringer:
        return fmt.Sprintf("Stringer: %s", v.String())
    case io.Reader:
        return "io.Reader"
    default:
        return fmt.Sprintf("unknown: %T = %v", v, v)
    }
}

// 安全な type assertion（カンマOKイディオム）
func toInt(i interface{}) (int, bool) {
    v, ok := i.(int)
    return v, ok
}

// インターフェースの型アサーションを使った機能拡張
type Closer interface {
    Close() error
}

type Flusher interface {
    Flush() error
}

func cleanup(w io.Writer) error {
    // ライターがFlusherインターフェースも持っていたらFlushを呼ぶ
    if f, ok := w.(Flusher); ok {
        if err := f.Flush(); err != nil {
            return fmt.Errorf("flush: %w", err)
        }
    }

    // ライターがCloserインターフェースも持っていたらCloseを呼ぶ
    if c, ok := w.(Closer); ok {
        if err := c.Close(); err != nil {
            return fmt.Errorf("close: %w", err)
        }
    }

    return nil
}

// 複数のインターフェースを要求する型制約
type ReadWriteCloser interface {
    io.Reader
    io.Writer
    io.Closer
}

// JSONのデコード結果を型安全に処理
func processJSON(data map[string]interface{}) {
    for key, val := range data {
        switch v := val.(type) {
        case string:
            fmt.Printf("  %s: string = %q\n", key, v)
        case float64: // JSONの数値はfloat64になる
            if v == float64(int(v)) {
                fmt.Printf("  %s: int = %d\n", key, int(v))
            } else {
                fmt.Printf("  %s: float = %.2f\n", key, v)
            }
        case bool:
            fmt.Printf("  %s: bool = %t\n", key, v)
        case nil:
            fmt.Printf("  %s: null\n", key)
        case []interface{}:
            fmt.Printf("  %s: array (len=%d)\n", key, len(v))
        case map[string]interface{}:
            fmt.Printf("  %s: object (keys=%d)\n", key, len(v))
        }
    }
}

func main() {
    // describe の使用例
    values := []interface{}{
        42, 3.14, "hello", true, nil,
        []byte{0xDE, 0xAD},
        []int{1, 2, 3},
        os.Stdout,
    }

    for _, v := range values {
        fmt.Println(describe(v))
    }

    // 安全な型アサーション
    var i interface{} = 42
    if n, ok := toInt(i); ok {
        fmt.Printf("Got int: %d\n", n)
    }

    var s interface{} = "not an int"
    if _, ok := toInt(s); !ok {
        fmt.Println("Not an int")
    }

    // cleanup の使用例
    reader := strings.NewReader("test")
    cleanup(reader) // Close/Flushを持たない → 何もしない

    // JSON結果の処理
    data := map[string]interface{}{
        "name":   "Alice",
        "age":    30.0,
        "active": true,
        "score":  95.5,
        "tags":   []interface{}{"go", "programming"},
    }
    fmt.Println("JSON data:")
    processJSON(data)
}
```

### コード例 6: カスタム型と型変換

```go
package main

import (
    "fmt"
    "strings"
    "time"
)

// カスタム型で型安全性を高める
type UserID int64
type OrderID int64
type ProductID int64

// 異なるID型の混同を防ぐ
// var uid UserID = OrderID(1) // コンパイルエラー: cannot use OrderID(1) as type UserID

// 温度型の例
type Celsius float64
type Fahrenheit float64
type Kelvin float64

func (c Celsius) ToFahrenheit() Fahrenheit {
    return Fahrenheit(c*9/5 + 32)
}

func (c Celsius) ToKelvin() Kelvin {
    return Kelvin(c + 273.15)
}

func (f Fahrenheit) ToCelsius() Celsius {
    return Celsius((f - 32) * 5 / 9)
}

func (c Celsius) String() string {
    return fmt.Sprintf("%.1f°C", float64(c))
}

func (f Fahrenheit) String() string {
    return fmt.Sprintf("%.1f°F", float64(f))
}

func (k Kelvin) String() string {
    return fmt.Sprintf("%.1fK", float64(k))
}

// カスタム文字列型
type Email string

func (e Email) Validate() error {
    s := string(e)
    if !strings.Contains(s, "@") {
        return fmt.Errorf("invalid email: %q", s)
    }
    parts := strings.Split(s, "@")
    if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
        return fmt.Errorf("invalid email format: %q", s)
    }
    if !strings.Contains(parts[1], ".") {
        return fmt.Errorf("invalid email domain: %q", parts[1])
    }
    return nil
}

func (e Email) Domain() string {
    parts := strings.Split(string(e), "@")
    if len(parts) == 2 {
        return parts[1]
    }
    return ""
}

func (e Email) String() string {
    return string(e)
}

// カスタム Duration ラッパー
type Timeout time.Duration

func (t Timeout) Duration() time.Duration {
    return time.Duration(t)
}

func (t Timeout) String() string {
    return time.Duration(t).String()
}

// 列挙型の模倣（iota を使用）
type Status int

const (
    StatusPending  Status = iota // 0
    StatusActive                  // 1
    StatusInactive                // 2
    StatusDeleted                 // 3
)

func (s Status) String() string {
    names := [...]string{"pending", "active", "inactive", "deleted"}
    if int(s) < len(names) {
        return names[s]
    }
    return fmt.Sprintf("Status(%d)", s)
}

func (s Status) IsValid() bool {
    return s >= StatusPending && s <= StatusDeleted
}

// ビットフラグの列挙型
type Permission uint8

const (
    PermRead    Permission = 1 << iota // 1
    PermWrite                           // 2
    PermExecute                         // 4
    PermAdmin                           // 8
)

func (p Permission) Has(flag Permission) bool {
    return p&flag != 0
}

func (p Permission) String() string {
    var perms []string
    if p.Has(PermRead) {
        perms = append(perms, "read")
    }
    if p.Has(PermWrite) {
        perms = append(perms, "write")
    }
    if p.Has(PermExecute) {
        perms = append(perms, "execute")
    }
    if p.Has(PermAdmin) {
        perms = append(perms, "admin")
    }
    if len(perms) == 0 {
        return "none"
    }
    return strings.Join(perms, "|")
}

func main() {
    // 温度変換
    temp := Celsius(100)
    fmt.Printf("%s = %s = %s\n", temp, temp.ToFahrenheit(), temp.ToKelvin())

    // Email バリデーション
    emails := []Email{
        "alice@example.com",
        "invalid-email",
        "bob@",
    }
    for _, e := range emails {
        if err := e.Validate(); err != nil {
            fmt.Printf("  %s: %v\n", e, err)
        } else {
            fmt.Printf("  %s: valid (domain=%s)\n", e, e.Domain())
        }
    }

    // ステータス
    status := StatusActive
    fmt.Printf("Status: %s (valid=%t)\n", status, status.IsValid())

    // パーミッション（ビットフラグ）
    perm := PermRead | PermWrite
    fmt.Printf("Permissions: %s\n", perm)
    fmt.Printf("Has read: %t\n", perm.Has(PermRead))
    fmt.Printf("Has admin: %t\n", perm.Has(PermAdmin))

    perm = perm | PermAdmin
    fmt.Printf("After adding admin: %s\n", perm)
}
```

### コード例 7: スライスの内部構造と操作

```go
package main

import (
    "fmt"
    "slices" // Go 1.21+
    "sort"
)

// スライスの内部構造（概念的）
// type slice struct {
//     array unsafe.Pointer  // 基底配列へのポインタ
//     len   int             // 長さ
//     cap   int             // 容量
// }

func main() {
    // スライスの作成方法
    s1 := []int{1, 2, 3, 4, 5}       // リテラル
    s2 := make([]int, 5)              // make (length=5, capacity=5)
    s3 := make([]int, 0, 10)          // make (length=0, capacity=10)

    fmt.Printf("s1: len=%d, cap=%d, %v\n", len(s1), cap(s1), s1)
    fmt.Printf("s2: len=%d, cap=%d, %v\n", len(s2), cap(s2), s2)
    fmt.Printf("s3: len=%d, cap=%d, %v\n", len(s3), cap(s3), s3)

    // append と容量の成長
    var growing []int
    prevCap := cap(growing)
    for i := 0; i < 20; i++ {
        growing = append(growing, i)
        if cap(growing) != prevCap {
            fmt.Printf("len=%2d, cap changed: %d → %d\n",
                len(growing), prevCap, cap(growing))
            prevCap = cap(growing)
        }
    }

    // スライスは基底配列を共有する（注意が必要）
    original := []int{1, 2, 3, 4, 5}
    sub := original[1:3]         // [2, 3]
    sub[0] = 99                  // original も変更される！
    fmt.Println("original:", original) // [1, 99, 3, 4, 5]

    // コピーで独立したスライスを作る
    independent := make([]int, len(original))
    copy(independent, original)
    independent[0] = 777
    fmt.Println("original:", original)    // 変わらない
    fmt.Println("independent:", independent) // [777, 99, 3, 4, 5]

    // slicesパッケージ（Go 1.21+）の活用
    nums := []int{3, 1, 4, 1, 5, 9, 2, 6}
    slices.Sort(nums)
    fmt.Println("sorted:", nums)

    idx, found := slices.BinarySearch(nums, 5)
    fmt.Printf("BinarySearch(5): idx=%d, found=%t\n", idx, found)

    // Contains
    fmt.Printf("Contains(9): %t\n", slices.Contains(nums, 9))
    fmt.Printf("Contains(7): %t\n", slices.Contains(nums, 7))

    // フィルタリング（手動）
    evens := make([]int, 0)
    for _, n := range nums {
        if n%2 == 0 {
            evens = append(evens, n)
        }
    }
    fmt.Println("evens:", evens)

    // カスタムソート
    type Person struct {
        Name string
        Age  int
    }
    people := []Person{
        {"Alice", 30},
        {"Bob", 25},
        {"Carol", 35},
    }
    sort.Slice(people, func(i, j int) bool {
        return people[i].Age < people[j].Age
    })
    fmt.Println("sorted by age:", people)
}
```

### コード例 8: マップの詳細

```go
package main

import (
    "fmt"
    "maps" // Go 1.21+
    "sort"
    "sync"
)

func main() {
    // マップの作成
    m1 := map[string]int{
        "alice": 95,
        "bob":   87,
        "carol": 92,
    }

    // make で空マップを作成
    m2 := make(map[string]int)
    m2["dave"] = 88

    // 要素の取得（カンマOKイディオム）
    score, ok := m1["alice"]
    fmt.Printf("alice: score=%d, exists=%t\n", score, ok)

    score, ok = m1["eve"]
    fmt.Printf("eve: score=%d, exists=%t\n", score, ok) // 0, false

    // 削除
    delete(m1, "bob")
    fmt.Printf("After delete: %v\n", m1)

    // マップの走査（順序は非決定的）
    for key, val := range m1 {
        fmt.Printf("  %s: %d\n", key, val)
    }

    // ソートされたキーで走査
    keys := make([]string, 0, len(m1))
    for k := range m1 {
        keys = append(keys, k)
    }
    sort.Strings(keys)
    fmt.Println("Sorted iteration:")
    for _, k := range keys {
        fmt.Printf("  %s: %d\n", k, m1[k])
    }

    // nil map vs empty map
    var nilMap map[string]int
    emptyMap := map[string]int{}
    fmt.Printf("nil map == nil: %t\n", nilMap == nil)     // true
    fmt.Printf("empty map == nil: %t\n", emptyMap == nil)  // false
    // nilMap["key"] = 1  // panic: assignment to entry in nil map
    _ = nilMap["key"]    // 読み取りはOK（ゼロ値が返る）

    // マップのマップ（ネスト）
    graph := map[string]map[string]int{
        "A": {"B": 1, "C": 4},
        "B": {"C": 2, "D": 5},
        "C": {"D": 1},
    }

    // ネストマップの安全なアクセス
    if neighbors, ok := graph["A"]; ok {
        for node, weight := range neighbors {
            fmt.Printf("A -> %s: weight=%d\n", node, weight)
        }
    }

    // マップをセットとして使う
    set := make(map[string]struct{})
    set["apple"] = struct{}{}
    set["banana"] = struct{}{}
    set["cherry"] = struct{}{}

    if _, exists := set["apple"]; exists {
        fmt.Println("apple is in the set")
    }

    // mapsパッケージ（Go 1.21+）
    clone := maps.Clone(m1)
    fmt.Printf("clone: %v\n", clone)

    // 単語カウント（実践的な例）
    text := "the quick brown fox jumps over the lazy dog the fox"
    wordCount := make(map[string]int)
    for _, word := range splitWords(text) {
        wordCount[word]++
    }
    fmt.Println("Word counts:", wordCount)

    // 並行安全なマップ操作（sync.Map は別セクション参照）
    _ = sync.Map{}
}

func splitWords(s string) []string {
    var words []string
    word := ""
    for _, r := range s {
        if r == ' ' || r == '\n' || r == '\t' {
            if word != "" {
                words = append(words, word)
                word = ""
            }
        } else {
            word += string(r)
        }
    }
    if word != "" {
        words = append(words, word)
    }
    return words
}
```

### コード例 9: ジェネリクスと型制約 (Go 1.18+)

```go
package main

import (
    "cmp"
    "fmt"
    "slices"
)

// 型制約: Ordered は比較可能な型を制約する
type Ordered interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
        ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
        ~float32 | ~float64 |
        ~string
}

// ジェネリックな Pair 型
type Pair[T, U any] struct {
    First  T
    Second U
}

func NewPair[T, U any](first T, second U) Pair[T, U] {
    return Pair[T, U]{First: first, Second: second}
}

func (p Pair[T, U]) String() string {
    return fmt.Sprintf("(%v, %v)", p.First, p.Second)
}

// ジェネリックな Optional 型（Rust の Option に相当）
type Optional[T any] struct {
    value   T
    present bool
}

func Some[T any](value T) Optional[T] {
    return Optional[T]{value: value, present: true}
}

func None[T any]() Optional[T] {
    return Optional[T]{}
}

func (o Optional[T]) Get() (T, bool) {
    return o.value, o.present
}

func (o Optional[T]) GetOrElse(defaultValue T) T {
    if o.present {
        return o.value
    }
    return defaultValue
}

func (o Optional[T]) Map(f func(T) T) Optional[T] {
    if o.present {
        return Some(f(o.value))
    }
    return None[T]()
}

// ジェネリックな Set
type Set[T comparable] struct {
    items map[T]struct{}
}

func NewSet[T comparable](items ...T) *Set[T] {
    s := &Set[T]{items: make(map[T]struct{})}
    for _, item := range items {
        s.Add(item)
    }
    return s
}

func (s *Set[T]) Add(item T) {
    s.items[item] = struct{}{}
}

func (s *Set[T]) Remove(item T) {
    delete(s.items, item)
}

func (s *Set[T]) Contains(item T) bool {
    _, ok := s.items[item]
    return ok
}

func (s *Set[T]) Len() int {
    return len(s.items)
}

func (s *Set[T]) Union(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for item := range s.items {
        result.Add(item)
    }
    for item := range other.items {
        result.Add(item)
    }
    return result
}

func (s *Set[T]) Intersection(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for item := range s.items {
        if other.Contains(item) {
            result.Add(item)
        }
    }
    return result
}

// cmp.Ordered を使ったジェネリック関数（Go 1.21+）
func Clamp[T cmp.Ordered](value, min, max T) T {
    if value < min {
        return min
    }
    if value > max {
        return max
    }
    return value
}

// ジェネリックな LinkedList
type Node[T any] struct {
    Value T
    Next  *Node[T]
}

type LinkedList[T any] struct {
    Head *Node[T]
    Len  int
}

func (ll *LinkedList[T]) Push(value T) {
    ll.Head = &Node[T]{Value: value, Next: ll.Head}
    ll.Len++
}

func (ll *LinkedList[T]) Pop() (T, bool) {
    if ll.Head == nil {
        var zero T
        return zero, false
    }
    value := ll.Head.Value
    ll.Head = ll.Head.Next
    ll.Len--
    return value, true
}

func (ll *LinkedList[T]) ForEach(fn func(T)) {
    for node := ll.Head; node != nil; node = node.Next {
        fn(node.Value)
    }
}

func main() {
    // Pair
    p1 := NewPair("name", 42)
    p2 := NewPair(3.14, true)
    fmt.Println(p1, p2)

    // Optional
    opt1 := Some(42)
    opt2 := None[string]()

    if v, ok := opt1.Get(); ok {
        fmt.Printf("opt1: %d\n", v)
    }
    fmt.Printf("opt2: %q\n", opt2.GetOrElse("default"))

    doubled := opt1.Map(func(n int) int { return n * 2 })
    fmt.Printf("doubled: %v\n", doubled.GetOrElse(0))

    // Set
    s1 := NewSet("go", "rust", "python")
    s2 := NewSet("python", "java", "go")

    union := s1.Union(s2)
    intersection := s1.Intersection(s2)

    fmt.Printf("s1 contains 'go': %t\n", s1.Contains("go"))
    fmt.Printf("union size: %d\n", union.Len())
    fmt.Printf("intersection size: %d\n", intersection.Len())

    // Clamp
    fmt.Printf("Clamp(15, 0, 10): %d\n", Clamp(15, 0, 10))
    fmt.Printf("Clamp(-5, 0, 10): %d\n", Clamp(-5, 0, 10))
    fmt.Printf("Clamp(5, 0, 10): %d\n", Clamp(5, 0, 10))

    // LinkedList
    ll := &LinkedList[int]{}
    ll.Push(1)
    ll.Push(2)
    ll.Push(3)

    ll.ForEach(func(v int) {
        fmt.Printf("%d -> ", v)
    })
    fmt.Println("nil")

    // slicesパッケージとジェネリクス
    nums := []int{5, 3, 8, 1, 9}
    slices.Sort(nums)
    fmt.Println("sorted:", nums)
}
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
     │ complex   │  │          │  │ function  │
     └───────────┘  └──────────┘  └───────────┘

整数型の詳細:
  ┌─────────────────────────────────────────┐
  │ 符号あり                                 │
  │  int8 ─ int16 ─ int32 ─ int64 ─ int    │
  │  (rune = int32)                          │
  │                                          │
  │ 符号なし                                 │
  │  uint8 ─ uint16 ─ uint32 ─ uint64 ─ uint│
  │  (byte = uint8)                          │
  │                                          │
  │ 特殊                                     │
  │  uintptr (ポインタサイズの符号なし整数)    │
  └─────────────────────────────────────────┘
```

### 図2: struct埋め込みのメモリレイアウト

```
Dog struct:
┌──────────────────────────────────┐
│  Animal (embedded)               │
│  ┌──────────────────────────┐    │
│  │ Name string              │    │
│  │ Age  int                 │    │
│  └──────────────────────────┘    │
│  Breed string                    │
└──────────────────────────────────┘

アクセス:
  d.Name   →  d.Animal.Name  (省略形)
  d.Age    →  d.Animal.Age   (省略形)
  d.Speak() → d.Animal.Speak() (メソッド昇格)

複数埋め込み時の名前衝突:
┌──────────────────────────────────┐
│  Service                         │
│  ┌─────────────┐ ┌────────────┐ │
│  │ Logger      │ │ Metrics    │ │
│  │  Log()      │ │  Record()  │ │
│  └─────────────┘ └────────────┘ │
│  Name string                     │
│  ※ LoggerとMetricsに同名メソッド │
│    があればコンパイルエラー        │
│    (明示的にアクセスが必要)        │
└──────────────────────────────────┘
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
                            │  │ Stat()      ││
┌────────────────┐          │  └────────────┘│
│  io.ReadCloser │          └────────────────┘
│  ┌────────────┐│               ▲
│  │ Read(...)   ││    こちらも    │
│  │ Close()     ││── 満たす ─────┘
│  └────────────┘│
└────────────────┘

インターフェースの内部表現 (iface):
┌──────────────────────────┐
│  interface value          │
│  ┌──────────┐ ┌────────┐│
│  │  type    │ │ value  ││
│  │  *itab   │ │ *data  ││
│  └──────────┘ └────────┘│
└──────────────────────────┘

itab = interface table (メソッドテーブルのキャッシュ)
data = 実際の値へのポインタ

nil interface vs nil pointer:
  var w io.Writer         // type=nil, data=nil → w == nil (true)
  var f *os.File = nil
  w = f                   // type=*os.File, data=nil → w == nil (false!)
```

### 図4: スライスの内部構造

```
s := []int{1, 2, 3, 4, 5}

スライスヘッダ:
┌─────────┐
│ ptr  ────┼──> ┌───┬───┬───┬───┬───┐
│ len = 5  │    │ 1 │ 2 │ 3 │ 4 │ 5 │  基底配列
│ cap = 5  │    └───┴───┴───┴───┴───┘
└─────────┘

sub := s[1:3]

┌─────────┐          ┌───┬───┬───┬───┬───┐
│ ptr  ────┼──────────┼───┤ 2 │ 3 │ 4 │ 5 │
│ len = 2  │          │ 1 │   │   │   │   │
│ cap = 4  │          └───┴───┴───┴───┴───┘
└─────────┘          ↑       ↑
                     s[0]    sub[0] = s[1]

append による再割当:
s2 := append(s, 6)

cap が不足 → 新しい配列を確保（約2倍の容量）
┌─────────┐
│ ptr  ────┼──> ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ len = 6  │    │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │   │   │   │   │
│ cap = 10 │    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
└─────────┘
※ 元のスライスとは基底配列が異なる（独立）
```

### 図5: マップの概念的な内部構造

```
map[string]int:
┌──────────────────────────────────────┐
│ ハッシュテーブル                       │
│                                      │
│  バケット配列:                         │
│  ┌─────────┐                          │
│  │bucket[0]│──> [key1:val1, key2:val2]│
│  ├─────────┤                          │
│  │bucket[1]│──> [key3:val3]           │
│  ├─────────┤                          │
│  │bucket[2]│──> (empty)               │
│  ├─────────┤                          │
│  │bucket[3]│──> [key4:val4, ...]      │
│  └─────────┘                          │
│                                      │
│  各バケットは最大8エントリを保持        │
│  オーバーフロー時は overflow bucket    │
│  にチェーンする                        │
│                                      │
│  成長: 要素数/バケット数 > 6.5 で      │
│  バケット配列を2倍に拡張               │
└──────────────────────────────────────┘
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
| sync.Mutex含む | 不可（コピーされる） | 必須（コピーは禁止） |
| スライスへの格納 | 値がコピーされる | ポインタが格納される |

### 表2: Goの型ゼロ値一覧

| 型 | ゼロ値 | 備考 |
|-----|-------|------|
| int, float64 | `0`, `0.0` | 数値型は全て0 |
| string | `""` | 空文字列 |
| bool | `false` | |
| pointer | `nil` | |
| slice | `nil` | `len()=0`, `cap()=0`。appendは可能 |
| map | `nil` | 読み取り可、代入前にmake必要 |
| channel | `nil` | 送受信で永久ブロック |
| struct | 各フィールドのゼロ値 | |
| interface | `nil` | type, value 両方 nil |
| function | `nil` | |
| array | 各要素のゼロ値 | 固定長 |

### 表3: コレクション型の比較

| 特性 | Array | Slice | Map | sync.Map |
|------|-------|-------|-----|----------|
| サイズ | 固定 | 動的 | 動的 | 動的 |
| メモリ | スタック可 | ヒープ | ヒープ | ヒープ |
| キーの型 | int (インデックス) | int (インデックス) | comparable | any |
| 順序保証 | あり | あり | なし | なし |
| 並行安全 | なし | なし | なし | あり |
| ゼロ値 | 使用可 | nil (appendは可) | nil (代入不可) | 使用可 |
| 等価比較 | `==` 可 | 不可 (reflect) | 不可 | 不可 |

### 表4: interface の設計指針

| 原則 | 説明 | 例 |
|------|------|-----|
| 小さく保つ | 1-3メソッドが理想 | io.Reader (1), io.ReadWriter (2) |
| consumer側で定義 | 使う側がインターフェースを宣言 | handler.Getter |
| 動詞 + "er" で命名 | アクション名から命名 | Reader, Writer, Closer |
| 必要になってから抽象化 | 事前定義を避ける | テスト時にmockが必要になったら |
| 合成で拡張 | 既存インターフェースを埋め込む | io.ReadWriteCloser |

---

## 4. アンチパターン

### アンチパターン 1: 巨大インターフェース

```go
// BAD: 巨大なインターフェース（God Interface）
type Repository interface {
    FindUser(id int) (*User, error)
    CreateUser(u *User) error
    UpdateUser(u *User) error
    DeleteUser(id int) error
    FindOrder(id int) (*Order, error)
    CreateOrder(o *Order) error
    UpdateOrder(o *Order) error
    DeleteOrder(id int) error
    FindProduct(id int) (*Product, error)
    // ... 20個以上のメソッド
}

// 問題点:
// 1. テスト時のmock作成が大変
// 2. 実装に不必要なメソッドが強制される
// 3. 変更の影響範囲が広い

// GOOD: ロールベースの小さなインターフェースに分割
type UserReader interface {
    FindUser(id int) (*User, error)
}

type UserWriter interface {
    CreateUser(u *User) error
    UpdateUser(u *User) error
    DeleteUser(id int) error
}

// 必要に応じて合成
type UserRepository interface {
    UserReader
    UserWriter
}

// テスト時は必要なインターフェースだけmock
type mockUserReader struct {
    users map[int]*User
}

func (m *mockUserReader) FindUser(id int) (*User, error) {
    u, ok := m.users[id]
    if !ok {
        return nil, ErrNotFound
    }
    return u, nil
}
```

### アンチパターン 2: nilインターフェースの罠

```go
// BAD: nil ポインタを interface に入れると nil 判定が壊れる
type Logger interface {
    Log(msg string)
}

type FileLogger struct {
    Path string
}

func (f *FileLogger) Log(msg string) {
    fmt.Printf("[%s] %s\n", f.Path, msg)
}

func getLogger() *FileLogger {
    // 条件によりnilを返す
    return nil
}

func processWithLogger() {
    var logger Logger = getLogger()  // *FileLogger(nil) を代入
    if logger != nil {
        // ここに到達してしまう！
        // interface{type:*FileLogger, value:nil} は nil ではない
        logger.Log("hello") // nil pointer dereference → PANIC
    }
}

// GOOD: 明示的に nil interface を返す
func getLoggerSafe() Logger {
    f := findLogger()
    if f == nil {
        return nil  // interface 自体が nil になる
    }
    return f
}

// GOOD: reflect を使ったnil チェック（高コスト、非推奨）
import "reflect"

func isNil(v interface{}) bool {
    if v == nil {
        return true
    }
    rv := reflect.ValueOf(v)
    switch rv.Kind() {
    case reflect.Ptr, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func:
        return rv.IsNil()
    }
    return false
}
```

### アンチパターン 3: 構造体のフィールドタグの間違い

```go
// BAD: JSON タグのスペースミス
type Config struct {
    Host string `json: "host"` // スペースが入っている → タグが無視される
    Port int    `json:"port" `  // 末尾にスペース → 正常に動かない
}

// BAD: バリデーションタグの不整合
type Request struct {
    Name  string `json:"name" validate:"required"`
    Email string `json:"email" validate:"required,email"`
    Age   int    `json:"age" validate:"min=0,max=200"`  // OK
    // Age   int    `json:"age" validate:"min=0, max=200"` // BAD: スペースが入る
}

// GOOD: go vet で検出できるが、手動確認も必要
type Config struct {
    Host     string        `json:"host" yaml:"host" env:"APP_HOST"`
    Port     int           `json:"port" yaml:"port" env:"APP_PORT"`
    Timeout  time.Duration `json:"timeout" yaml:"timeout" env:"APP_TIMEOUT"`
    LogLevel string        `json:"log_level" yaml:"log_level" env:"APP_LOG_LEVEL"`
}
```

### アンチパターン 4: スライスの容量漏洩

```go
// BAD: 大きなスライスの一部だけを保持 → 基底配列がGCされない
func getFirstThree(data []byte) []byte {
    return data[:3]  // 元のdata全体がメモリに残る
}

// GOOD: コピーして基底配列への参照を切る
func getFirstThreeSafe(data []byte) []byte {
    result := make([]byte, 3)
    copy(result, data[:3])
    return result
}

// BAD: append が意図せず元のスライスを変更
func appendToSlice(s []int) []int {
    return append(s, 999)  // cap に余裕があると元の基底配列を変更
}

// GOOD: フルスライス式で容量を制限
func safeSubslice(s []int) []int {
    sub := s[1:3:3]  // s[low:high:max] → cap = max - low
    return append(sub, 999)  // 必ず新しい配列が確保される
}
```

### アンチパターン 5: ゼロ値の活用を忘れる

```go
// BAD: 不必要な初期化
var mu = &sync.Mutex{}     // ポインタにする必要なし
var buf = bytes.Buffer{}   // 明示的な初期化は不要
var wg = &sync.WaitGroup{} // ゼロ値で十分

// GOOD: ゼロ値を活用
var mu sync.Mutex           // ゼロ値で使用可能
var buf bytes.Buffer         // ゼロ値で使用可能
var wg sync.WaitGroup        // ゼロ値で使用可能
var once sync.Once           // ゼロ値で使用可能

// ゼロ値が有用な型の例:
// sync.Mutex     → ロックされていない状態
// sync.WaitGroup → カウンタ0
// bytes.Buffer   → 空のバッファ
// strings.Builder → 空のビルダー
```

---

## 5. FAQ

### Q1: structに「コンストラクタ」はあるか？

Goにはコンストラクタ構文がないが、慣例として `New関数名` を使う。例: `func NewUser(name string) *User`。これはファクトリ関数パターンであり、バリデーションやデフォルト値の設定に適している。

```go
// 基本的なファクトリ関数
func NewUser(name, email string) *User {
    return &User{
        Name:      name,
        Email:     email,
        CreatedAt: time.Now(),
    }
}

// Functional Options パターン（複雑な初期化向け）
type ServerOption func(*Server)

func WithPort(port int) ServerOption {
    return func(s *Server) { s.Port = port }
}

func WithTLS(cert, key string) ServerOption {
    return func(s *Server) {
        s.TLS = true
        s.CertFile = cert
        s.KeyFile = key
    }
}

func NewServer(host string, opts ...ServerOption) *Server {
    s := &Server{
        Host: host,
        Port: 8080, // デフォルト値
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// 使用例
srv := NewServer("localhost",
    WithPort(9090),
    WithTLS("cert.pem", "key.pem"),
)
```

### Q2: いつ値レシーバ、いつポインタレシーバを使うべきか？

原則: (1) structを変更するならポインタレシーバ、(2) structが大きい（フィールドが多い）ならポインタレシーバ、(3) 一貫性のため同一型のレシーバは統一する。迷ったらポインタレシーバを選ぶ。

具体的な判断基準:
- **ポインタレシーバが必要**: `sync.Mutex` を含む、フィールドを変更する、大きな構造体（目安: 3フィールド超）
- **値レシーバが適切**: 小さい構造体（`time.Time`など）、イミュータブルなメソッド、基本型のカスタム型
- **混在禁止**: 同一型で値レシーバとポインタレシーバを混在させない（interface 満足の問題）

### Q3: any型(interface{})を多用してよいか？

避けるべき。Go 1.18以降はジェネリクスが使えるため、`any` の利用は本当に任意の型を受け取る必要がある場面（JSONパース等）に限定する。型安全性を損なうため、可能な限り具体的な型やインターフェースを使う。

`any` の許容される用途:
- `encoding/json` のMarshal/Unmarshal
- ロギングライブラリの引数
- リフレクションが必要な場面
- テストヘルパー

### Q4: struct のフィールド順序はメモリに影響するか？

はい。Go のコンパイラは構造体フィールドをメモリアラインメントに合わせてパディングを挿入する。フィールドの順序を工夫することで、パディングを減らしメモリ効率を改善できる。

```go
// BAD: パディングが多い（24 bytes）
type BadLayout struct {
    A bool    // 1 byte + 7 bytes padding
    B int64   // 8 bytes
    C bool    // 1 byte + 7 bytes padding
}

// GOOD: パディングが少ない（16 bytes）
type GoodLayout struct {
    B int64   // 8 bytes
    A bool    // 1 byte
    C bool    // 1 byte + 6 bytes padding
}

// fieldalignment ツールで確認可能
// go install golang.org/x/tools/go/analysis/passes/fieldalignment/cmd/fieldalignment@latest
// fieldalignment -fix ./...
```

### Q5: comparable 制約とは何か？

`comparable` は Go 1.18 で導入された組み込み型制約で、`==` と `!=` 演算子が使える型を制約する。`map` のキー型や、ジェネリックな Set の要素型に使う。

`comparable` に含まれる型: bool, 整数型, 浮動小数点型, complex, string, pointer, channel, array (要素がcomparable), struct (全フィールドがcomparable)

`comparable` に含まれない型: slice, map, function

---

## 6. まとめ

| 概念 | 要点 |
|------|------|
| 基本型 | int, float64, string, bool, rune, byte。ビット幅の明示が推奨される場面もある |
| struct | フィールドの集合。メソッドを持てる。JSONタグでシリアライゼーション制御 |
| 埋め込み | 継承ではなくコンポジション。メソッドとフィールドが昇格される |
| interface | メソッドセット。暗黙的に満たされる。consumer側で定義するのが推奨 |
| type assertion | `v, ok := i.(Type)` で安全に判定 |
| type switch | `switch v := i.(type)` で分岐 |
| ゼロ値 | 全ての型にゼロ値がある。有用なゼロ値を設計する |
| カスタム型 | `type X underlying` で型安全性向上。iota で列挙型を模倣 |
| ジェネリクス | Go 1.18+ の型パラメータ。`comparable`, `any` 等の型制約 |
| スライス | 動的配列。基底配列の共有に注意。append は再割当の可能性あり |
| マップ | ハッシュテーブル。nil map への代入は panic。順序は非決定的 |

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
4. **Go Blog, "Strings, bytes, runes and characters in Go"** -- https://go.dev/blog/strings
5. **Go Wiki, "SliceTricks"** -- https://go.dev/wiki/SliceTricks
6. **Go Blog, "An Introduction to Generics"** -- https://go.dev/blog/intro-generics
7. **Go Blog, "Maps in Go"** -- https://go.dev/blog/maps
