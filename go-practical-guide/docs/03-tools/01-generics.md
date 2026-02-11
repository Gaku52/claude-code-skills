# Go ジェネリクスガイド

> Go 1.18で導入された型パラメータと制約を使い、型安全で再利用可能なコードを書く

## この章で学ぶこと

1. **型パラメータ** の構文と基本的な使い方（ジェネリック関数・型）
2. **制約（constraints）** の定義方法と標準ライブラリの制約パッケージ
3. **実践パターン** — コレクション操作、リポジトリパターン、Result型の実装

---

## 1. ジェネリクスの基本

### ジェネリクス導入前後の比較

```
【導入前】型ごとに関数を複製

func MaxInt(a, b int) int         { if a > b { return a }; return b }
func MaxFloat(a, b float64) float64 { if a > b { return a }; return b }
func MaxString(a, b string) string  { if a > b { return a }; return b }

           ↓ ジェネリクスで統一

【導入後】一つの関数で全ての型に対応

func Max[T cmp.Ordered](a, b T) T { if a > b { return a }; return b }
```

### 型パラメータの構文

```
+-------- 型パラメータリスト --------+
|                                    |
func  FuncName [ T  constraint ]( params ) returns
                 |       |
                 |       +--- 制約: T が満たすべき条件
                 +----------- 型パラメータ名
```

### コード例1: 最初のジェネリック関数

```go
package main

import (
    "cmp"
    "fmt"
)

// T は cmp.Ordered を満たす任意の型
func Max[T cmp.Ordered](a, b T) T {
    if a > b {
        return a
    }
    return b
}

func Min[T cmp.Ordered](a, b T) T {
    if a < b {
        return a
    }
    return b
}

func Clamp[T cmp.Ordered](val, lo, hi T) T {
    return Max(lo, Min(val, hi))
}

func main() {
    fmt.Println(Max(3, 7))           // 7
    fmt.Println(Max(3.14, 2.71))     // 3.14
    fmt.Println(Max("apple", "banana")) // banana
    fmt.Println(Clamp(150, 0, 100))  // 100
}
```

### コード例2: ジェネリックなスライス操作

```go
// Map はスライスの各要素に関数を適用する
func Map[T, U any](s []T, f func(T) U) []U {
    result := make([]U, len(s))
    for i, v := range s {
        result[i] = f(v)
    }
    return result
}

// Filter はスライスから条件を満たす要素を抽出する
func Filter[T any](s []T, pred func(T) bool) []T {
    var result []T
    for _, v := range s {
        if pred(v) {
            result = append(result, v)
        }
    }
    return result
}

// Reduce はスライスを単一の値に集約する
func Reduce[T, U any](s []T, init U, f func(U, T) U) U {
    acc := init
    for _, v := range s {
        acc = f(acc, v)
    }
    return acc
}

// 使用例
nums := []int{1, 2, 3, 4, 5}
doubled := Map(nums, func(n int) int { return n * 2 })
// [2, 4, 6, 8, 10]

evens := Filter(nums, func(n int) bool { return n%2 == 0 })
// [2, 4]

sum := Reduce(nums, 0, func(acc, n int) int { return acc + n })
// 15
```

---

## 2. 制約（Constraints）

### 制約の種類

```
+-------------------+
|   any (interface{})|  ← 最も緩い: 全ての型を許容
+-------------------+
        |
+-------------------+
|   comparable      |  ← == と != が使える型
+-------------------+
        |
+-------------------+
|   cmp.Ordered     |  ← 比較演算子が使える型 (<, >, <=, >=)
+-------------------+
        |
+-------------------+
|  カスタム制約      |  ← 特定のメソッドや型を要求
+-------------------+
```

### コード例3: カスタム制約の定義

```go
// メソッドベースの制約
type Stringer interface {
    String() string
}

// 型集合ベースの制約（union）
type Number interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
    ~float32 | ~float64
}

// チルダ (~) は基底型を指定
// ~int は「基底型が int である全ての型」を含む
type MyInt int      // ~int に含まれる
type Score int      // ~int に含まれる

// 複合制約: メソッド + 型集合
type OrderedStringer interface {
    cmp.Ordered
    String() string
}

// 実用例: Sum 関数
func Sum[T Number](nums []T) T {
    var total T
    for _, n := range nums {
        total += n
    }
    return total
}

fmt.Println(Sum([]int{1, 2, 3}))        // 6
fmt.Println(Sum([]float64{1.1, 2.2}))   // 3.3
```

### 主要な制約の比較表

| 制約 | 許容される型 | 使える演算 | 用途 |
|------|------------|-----------|------|
| `any` | 全ての型 | なし（インターフェース経由のみ） | コンテナ、ラッパー |
| `comparable` | 比較可能な型 | `==`, `!=` | マップのキー、重複排除 |
| `cmp.Ordered` | 順序付き型 | `<`, `>`, `<=`, `>=`, `==` | ソート、最大最小 |
| `~int \| ~float64` | 指定型の基底型を持つ型 | 数値演算 | 計算、集計 |
| カスタム interface | メソッドを持つ型 | 指定メソッド | ドメイン固有のロジック |

### ~ (チルダ) あり/なし 比較表

| 制約定義 | `int` | `type MyInt int` | `type Score int` |
|---------|-------|-----------------|-----------------|
| `int` | 合致 | 不一致 | 不一致 |
| `~int` | 合致 | 合致 | 合致 |

---

## 3. ジェネリック型

### コード例4: ジェネリックなデータ構造

```go
// スタック
type Stack[T any] struct {
    items []T
}

func NewStack[T any]() *Stack[T] {
    return &Stack[T]{}
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    last := len(s.items) - 1
    item := s.items[last]
    s.items = s.items[:last]
    return item, true
}

func (s *Stack[T]) Peek() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    return s.items[len(s.items)-1], true
}

func (s *Stack[T]) Len() int {
    return len(s.items)
}

// 使用例
intStack := NewStack[int]()
intStack.Push(1)
intStack.Push(2)
val, _ := intStack.Pop() // 2

strStack := NewStack[string]()
strStack.Push("hello")
```

### コード例5: Result 型（エラーハンドリング改善）

```go
// Result はエラーまたは値を持つ型
type Result[T any] struct {
    value T
    err   error
}

func Ok[T any](value T) Result[T] {
    return Result[T]{value: value}
}

func Err[T any](err error) Result[T] {
    return Result[T]{err: err}
}

func (r Result[T]) IsOk() bool {
    return r.err == nil
}

func (r Result[T]) Unwrap() (T, error) {
    return r.value, r.err
}

func (r Result[T]) UnwrapOr(defaultVal T) T {
    if r.err != nil {
        return defaultVal
    }
    return r.value
}

// Map: 値がある場合のみ変換を適用
func MapResult[T, U any](r Result[T], f func(T) U) Result[U] {
    if r.err != nil {
        return Err[U](r.err)
    }
    return Ok(f(r.value))
}

// 使用例
result := Ok(42)
doubled := MapResult(result, func(n int) int { return n * 2 })
val, _ := doubled.Unwrap() // 84
```

---

## 4. 実践パターン

### コード例6: ジェネリックなリポジトリパターン

```go
type Entity interface {
    GetID() string
}

type Repository[T Entity] interface {
    FindByID(id string) (T, error)
    FindAll() ([]T, error)
    Save(entity T) error
    Delete(id string) error
}

// インメモリ実装
type InMemoryRepo[T Entity] struct {
    mu    sync.RWMutex
    store map[string]T
}

func NewInMemoryRepo[T Entity]() *InMemoryRepo[T] {
    return &InMemoryRepo[T]{
        store: make(map[string]T),
    }
}

func (r *InMemoryRepo[T]) FindByID(id string) (T, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    entity, ok := r.store[id]
    if !ok {
        var zero T
        return zero, fmt.Errorf("entity %s not found", id)
    }
    return entity, nil
}

func (r *InMemoryRepo[T]) Save(entity T) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.store[entity.GetID()] = entity
    return nil
}

// 使用例
type User struct {
    ID   string
    Name string
}

func (u User) GetID() string { return u.ID }

repo := NewInMemoryRepo[User]()
repo.Save(User{ID: "1", Name: "Alice"})
user, _ := repo.FindByID("1")
```

### ジェネリクスの型推論フロー

```
Max(3, 7)
   |
   +-- コンパイラが引数の型を推論
   |     3 → int,  7 → int
   |
   +-- T = int と決定
   |
   +-- Max[int](3, 7) として展開
   |
   +-- int は cmp.Ordered を満たすか？ → Yes
   |
   +-- コンパイル成功

Max(3, 7.0)
   |
   +-- 3 → int,  7.0 → float64
   |
   +-- 型が一致しない → コンパイルエラー
   |
   +-- 修正: Max(float64(3), 7.0) または Max[float64](3, 7.0)
```

---

## 5. アンチパターン

### アンチパターン1: 不要なジェネリクス化

```go
// NG: ジェネリクスが不要なケース
func PrintValue[T any](v T) {
    fmt.Println(v) // any なら interface{} で十分
}

// OK: interface{} または any を直接使う
func PrintValue(v any) {
    fmt.Println(v)
}

// NG: 型パラメータが1つの具体型にしか使われない
func ParseUserJSON[T User](data []byte) (T, error) {
    var result T
    err := json.Unmarshal(data, &result)
    return result, err
}

// OK: 具体型を直接使う
func ParseUserJSON(data []byte) (User, error) {
    var user User
    err := json.Unmarshal(data, &user)
    return user, err
}
```

### アンチパターン2: 過度に複雑な制約

```go
// NG: 制約が複雑すぎて可読性が低い
type ComplexConstraint[K comparable, V interface {
    ~int | ~string
    fmt.Stringer
    encoding.BinaryMarshaler
}] struct {
    data map[K]V
}

// OK: 制約を分離して名前をつける
type Serializable interface {
    fmt.Stringer
    encoding.BinaryMarshaler
}

type ValueType interface {
    ~int | ~string
    Serializable
}

type Store[K comparable, V ValueType] struct {
    data map[K]V
}
```

---

## FAQ

### Q1. ジェネリクスとインターフェースの使い分けは？

ジェネリクスは「同じアルゴリズムを異なる型に適用する」場合に使う。インターフェースは「異なる実装を同じ振る舞いとして抽象化する」場合に使う。例えば、ソートアルゴリズムはジェネリクス向き。一方、データベース接続のような多態性はインターフェース向き。

### Q2. ジェネリクスはパフォーマンスに影響するか？

Go のジェネリクスはコンパイル時にGCShape stenciling（形状ベースの特殊化）を行う。ポインタ型は共通の実装を共有し、値型は必要に応じて特殊化される。大半のケースでインターフェース経由の呼び出しより高速または同等。

### Q3. Go 1.18以降、標準ライブラリでジェネリクスはどう使われている？

`slices` パッケージ（ソート、検索、比較）、`maps` パッケージ（キー取得、値取得、クローン）、`cmp` パッケージ（比較関数）が追加された。`sync.Map` のジェネリクス版は標準ライブラリにはないが、サードパーティで提供されている。

---

## まとめ

| 概念 | 要点 |
|------|------|
| 型パラメータ `[T ...]` | 関数・型を複数の型に対して一般化 |
| `any` | 全ての型を許容する制約（= `interface{}`） |
| `comparable` | `==` / `!=` が使える型の制約 |
| `cmp.Ordered` | 比較演算子が使える型の制約 |
| `~T` (チルダ) | 基底型が T である全ての型を含む |
| 型推論 | 引数から型パラメータを自動推論 |
| ゼロ値 | `var zero T` でジェネリック型のゼロ値を取得 |
| `slices` / `maps` | 標準ライブラリのジェネリックユーティリティ |

---

## 次に読むべきガイド

- **03-tools/02-profiling.md** — プロファイリング：pprof、trace
- **03-tools/04-best-practices.md** — ベストプラクティス：Effective Go
- **03-tools/00-cli-development.md** — CLI開発：cobra、flag、promptui

---

## 参考文献

1. **Go公式 — Type Parameters Proposal** https://go.googlesource.com/proposal/+/refs/heads/master/design/43651-type-parameters.md
2. **Go公式 — Tutorial: Getting started with generics** https://go.dev/doc/tutorial/generics
3. **Go Blog — An Introduction To Generics** https://go.dev/blog/intro-generics
4. **Go標準ライブラリ — slices パッケージ** https://pkg.go.dev/slices
