# Go ジェネリクスガイド

> Go 1.18で導入された型パラメータと制約を使い、型安全で再利用可能なコードを書く

## この章で学ぶこと

1. **型パラメータ** の構文と基本的な使い方（ジェネリック関数・型）
2. **制約（constraints）** の定義方法と標準ライブラリの制約パッケージ
3. **実践パターン** — コレクション操作、リポジトリパターン、Result型の実装
4. **標準ライブラリ** の `slices`、`maps`、`cmp` パッケージの活用
5. **パフォーマンス特性** とジェネリクスの適用判断基準

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

// Find は条件を満たす最初の要素を返す
func Find[T any](s []T, pred func(T) bool) (T, bool) {
    for _, v := range s {
        if pred(v) {
            return v, true
        }
    }
    var zero T
    return zero, false
}

// GroupBy はキー関数に基づいてグルーピングする
func GroupBy[T any, K comparable](s []T, keyFn func(T) K) map[K][]T {
    result := make(map[K][]T)
    for _, v := range s {
        key := keyFn(v)
        result[key] = append(result[key], v)
    }
    return result
}

// Chunk はスライスを指定サイズのチャンクに分割する
func Chunk[T any](s []T, size int) [][]T {
    if size <= 0 {
        return nil
    }
    var chunks [][]T
    for i := 0; i < len(s); i += size {
        end := i + size
        if end > len(s) {
            end = len(s)
        }
        chunks = append(chunks, s[i:end])
    }
    return chunks
}

// Unique は重複を排除したスライスを返す
func Unique[T comparable](s []T) []T {
    seen := make(map[T]struct{})
    var result []T
    for _, v := range s {
        if _, ok := seen[v]; !ok {
            seen[v] = struct{}{}
            result = append(result, v)
        }
    }
    return result
}

// 使用例
func main() {
    nums := []int{1, 2, 3, 4, 5}
    doubled := Map(nums, func(n int) int { return n * 2 })
    // [2, 4, 6, 8, 10]

    evens := Filter(nums, func(n int) bool { return n%2 == 0 })
    // [2, 4]

    sum := Reduce(nums, 0, func(acc, n int) int { return acc + n })
    // 15

    // 文字列操作
    words := []string{"hello", "world", "go", "generics"}
    lengths := Map(words, func(s string) int { return len(s) })
    // [5, 5, 2, 8]

    longWords := Filter(words, func(s string) bool { return len(s) > 3 })
    // ["hello", "world", "generics"]

    // グルーピング
    type User struct {
        Name string
        Role string
    }
    users := []User{
        {"Alice", "admin"}, {"Bob", "user"}, {"Charlie", "admin"}, {"Dave", "user"},
    }
    byRole := GroupBy(users, func(u User) string { return u.Role })
    // map["admin":[Alice, Charlie] "user":[Bob, Dave]]

    // 重複排除
    ids := []int{1, 2, 3, 2, 1, 4, 3, 5}
    unique := Unique(ids) // [1, 2, 3, 4, 5]
}
```

### コード例3: FlatMap と Zip

```go
// FlatMap はスライスの各要素をスライスに変換してフラット化する
func FlatMap[T, U any](s []T, f func(T) []U) []U {
    var result []U
    for _, v := range s {
        result = append(result, f(v)...)
    }
    return result
}

// Zip は2つのスライスを組にする
func Zip[T, U any](a []T, b []U) []Pair[T, U] {
    minLen := len(a)
    if len(b) < minLen {
        minLen = len(b)
    }
    result := make([]Pair[T, U], minLen)
    for i := 0; i < minLen; i++ {
        result[i] = Pair[T, U]{First: a[i], Second: b[i]}
    }
    return result
}

type Pair[T, U any] struct {
    First  T
    Second U
}

// Partition は条件に基づいてスライスを2つに分割する
func Partition[T any](s []T, pred func(T) bool) (matched, unmatched []T) {
    for _, v := range s {
        if pred(v) {
            matched = append(matched, v)
        } else {
            unmatched = append(unmatched, v)
        }
    }
    return
}

// 使用例
func example() {
    // FlatMap: 文をトークンに分割
    sentences := []string{"hello world", "go generics"}
    tokens := FlatMap(sentences, func(s string) []string {
        return strings.Split(s, " ")
    })
    // ["hello", "world", "go", "generics"]

    // Zip: 名前とスコアを組にする
    names := []string{"Alice", "Bob", "Charlie"}
    scores := []int{90, 85, 95}
    pairs := Zip(names, scores)
    // [{Alice, 90}, {Bob, 85}, {Charlie, 95}]

    // Partition: 合格と不合格に分ける
    pass, fail := Partition(scores, func(s int) bool { return s >= 90 })
    // pass: [90, 95], fail: [85]
}
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

### コード例4: カスタム制約の定義

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

// 整数のみの制約
type Integer interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

// 浮動小数点のみの制約
type Float interface {
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

// 実用例: Average 関数（浮動小数点を返す）
func Average[T Number](nums []T) float64 {
    if len(nums) == 0 {
        return 0
    }
    var sum T
    for _, n := range nums {
        sum += n
    }
    return float64(sum) / float64(len(nums))
}

// 実用例: Abs 関数（符号付き数値）
type Signed interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64
}

func Abs[T Signed](v T) T {
    if v < 0 {
        return -v
    }
    return v
}

fmt.Println(Sum([]int{1, 2, 3}))        // 6
fmt.Println(Sum([]float64{1.1, 2.2}))   // 3.3
fmt.Println(Average([]int{10, 20, 30})) // 20.0
fmt.Println(Abs(-42))                    // 42
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

### コード例5: 複合制約の実践

```go
// Comparable + Stringer を両方満たす制約
type ComparableStringer interface {
    comparable
    String() string
}

// マップのキーとして使え、文字列表現を持つ型
func PrintMap[K ComparableStringer, V any](m map[K]V) {
    for k, v := range m {
        fmt.Printf("%s: %v\n", k.String(), v)
    }
}

// 制約インターフェースの合成
type Numeric interface {
    Integer | Float
}

type Addable interface {
    Numeric
    comparable
}

// JSON シリアライズ可能な制約
type JSONSerializable interface {
    comparable
    MarshalJSON() ([]byte, error)
    UnmarshalJSON([]byte) error
}

// バリデーション可能な制約
type Validatable interface {
    Validate() error
}

// バリデーション付きの保存関数
func SaveAll[T Validatable](items []T) error {
    for i, item := range items {
        if err := item.Validate(); err != nil {
            return fmt.Errorf("item[%d]: %w", i, err)
        }
    }
    // 保存処理...
    return nil
}
```

---

## 3. ジェネリック型

### コード例6: ジェネリックなデータ構造

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

func (s *Stack[T]) IsEmpty() bool {
    return len(s.items) == 0
}

// 使用例
intStack := NewStack[int]()
intStack.Push(1)
intStack.Push(2)
val, _ := intStack.Pop() // 2

strStack := NewStack[string]()
strStack.Push("hello")
```

### コード例7: ジェネリックなキュー

```go
// Queue はジェネリックなFIFOキュー
type Queue[T any] struct {
    items []T
}

func NewQueue[T any]() *Queue[T] {
    return &Queue[T]{}
}

func (q *Queue[T]) Enqueue(item T) {
    q.items = append(q.items, item)
}

func (q *Queue[T]) Dequeue() (T, bool) {
    if len(q.items) == 0 {
        var zero T
        return zero, false
    }
    item := q.items[0]
    q.items = q.items[1:]
    return item, true
}

func (q *Queue[T]) Peek() (T, bool) {
    if len(q.items) == 0 {
        var zero T
        return zero, false
    }
    return q.items[0], true
}

func (q *Queue[T]) Len() int {
    return len(q.items)
}

// PriorityQueue は優先度付きキュー
type PriorityQueue[T any] struct {
    items []T
    less  func(a, b T) bool
}

func NewPriorityQueue[T any](less func(a, b T) bool) *PriorityQueue[T] {
    return &PriorityQueue[T]{less: less}
}

func (pq *PriorityQueue[T]) Push(item T) {
    pq.items = append(pq.items, item)
    pq.up(len(pq.items) - 1)
}

func (pq *PriorityQueue[T]) Pop() (T, bool) {
    if len(pq.items) == 0 {
        var zero T
        return zero, false
    }
    n := len(pq.items) - 1
    pq.items[0], pq.items[n] = pq.items[n], pq.items[0]
    item := pq.items[n]
    pq.items = pq.items[:n]
    if n > 0 {
        pq.down(0)
    }
    return item, true
}

func (pq *PriorityQueue[T]) up(j int) {
    for {
        i := (j - 1) / 2
        if i == j || !pq.less(pq.items[j], pq.items[i]) {
            break
        }
        pq.items[i], pq.items[j] = pq.items[j], pq.items[i]
        j = i
    }
}

func (pq *PriorityQueue[T]) down(i int) {
    n := len(pq.items)
    for {
        left := 2*i + 1
        if left >= n {
            break
        }
        j := left
        if right := left + 1; right < n && pq.less(pq.items[right], pq.items[left]) {
            j = right
        }
        if !pq.less(pq.items[j], pq.items[i]) {
            break
        }
        pq.items[i], pq.items[j] = pq.items[j], pq.items[i]
        i = j
    }
}

func (pq *PriorityQueue[T]) Len() int {
    return len(pq.items)
}

// 使用例
pq := NewPriorityQueue(func(a, b int) bool { return a < b })
pq.Push(3)
pq.Push(1)
pq.Push(2)
val, _ := pq.Pop() // 1（最小値が先に出る）
```

### コード例8: ジェネリックな並行安全マップ

```go
// SyncMap はジェネリックな並行安全マップ
type SyncMap[K comparable, V any] struct {
    mu sync.RWMutex
    m  map[K]V
}

func NewSyncMap[K comparable, V any]() *SyncMap[K, V] {
    return &SyncMap[K, V]{
        m: make(map[K]V),
    }
}

func (sm *SyncMap[K, V]) Get(key K) (V, bool) {
    sm.mu.RLock()
    defer sm.mu.RUnlock()
    val, ok := sm.m[key]
    return val, ok
}

func (sm *SyncMap[K, V]) Set(key K, value V) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    sm.m[key] = value
}

func (sm *SyncMap[K, V]) Delete(key K) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    delete(sm.m, key)
}

func (sm *SyncMap[K, V]) Len() int {
    sm.mu.RLock()
    defer sm.mu.RUnlock()
    return len(sm.m)
}

func (sm *SyncMap[K, V]) Range(fn func(K, V) bool) {
    sm.mu.RLock()
    defer sm.mu.RUnlock()
    for k, v := range sm.m {
        if !fn(k, v) {
            break
        }
    }
}

// GetOrSet は値が存在しなければ設定して返す
func (sm *SyncMap[K, V]) GetOrSet(key K, defaultVal V) V {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    if val, ok := sm.m[key]; ok {
        return val
    }
    sm.m[key] = defaultVal
    return defaultVal
}

// 使用例
cache := NewSyncMap[string, int]()
cache.Set("count", 42)
val, ok := cache.Get("count") // 42, true
```

### コード例9: Result 型（エラーハンドリング改善）

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

func (r Result[T]) IsErr() bool {
    return r.err != nil
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

func (r Result[T]) UnwrapOrElse(fn func(error) T) T {
    if r.err != nil {
        return fn(r.err)
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

// FlatMap: 値がある場合に別のResult生成関数を適用
func FlatMapResult[T, U any](r Result[T], f func(T) Result[U]) Result[U] {
    if r.err != nil {
        return Err[U](r.err)
    }
    return f(r.value)
}

// Collect: Result のスライスから成功値を収集（1つでもエラーなら失敗）
func Collect[T any](results []Result[T]) Result[[]T] {
    values := make([]T, 0, len(results))
    for _, r := range results {
        if r.IsErr() {
            return Err[[]T](r.err)
        }
        values = append(values, r.value)
    }
    return Ok(values)
}

// 使用例
result := Ok(42)
doubled := MapResult(result, func(n int) int { return n * 2 })
val, _ := doubled.Unwrap() // 84

// チェーン
func fetchUser(id string) Result[User] {
    user, err := db.FindUser(id)
    if err != nil {
        return Err[User](err)
    }
    return Ok(*user)
}

func getEmail(u User) Result[string] {
    if u.Email == "" {
        return Err[string](errors.New("email not set"))
    }
    return Ok(u.Email)
}

// Result チェーン
email := FlatMapResult(fetchUser("123"), getEmail)
fmt.Println(email.UnwrapOr("no-email@example.com"))
```

### コード例10: Optional 型

```go
// Optional はnil安全な値コンテナ
type Optional[T any] struct {
    value *T
}

func Some[T any](v T) Optional[T] {
    return Optional[T]{value: &v}
}

func None[T any]() Optional[T] {
    return Optional[T]{}
}

func (o Optional[T]) IsPresent() bool {
    return o.value != nil
}

func (o Optional[T]) Get() (T, bool) {
    if o.value == nil {
        var zero T
        return zero, false
    }
    return *o.value, true
}

func (o Optional[T]) OrElse(defaultVal T) T {
    if o.value == nil {
        return defaultVal
    }
    return *o.value
}

func (o Optional[T]) IfPresent(fn func(T)) {
    if o.value != nil {
        fn(*o.value)
    }
}

func MapOptional[T, U any](o Optional[T], f func(T) U) Optional[U] {
    if o.value == nil {
        return None[U]()
    }
    return Some(f(*o.value))
}

// 使用例
name := Some("Alice")
name.IfPresent(func(n string) {
    fmt.Printf("Hello, %s!\n", n)
})

empty := None[string]()
fmt.Println(empty.OrElse("anonymous")) // "anonymous"
```

---

## 4. 実践パターン

### コード例11: ジェネリックなリポジトリパターン

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

func (r *InMemoryRepo[T]) FindAll() ([]T, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    result := make([]T, 0, len(r.store))
    for _, entity := range r.store {
        result = append(result, entity)
    }
    return result, nil
}

func (r *InMemoryRepo[T]) Save(entity T) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.store[entity.GetID()] = entity
    return nil
}

func (r *InMemoryRepo[T]) Delete(id string) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    if _, ok := r.store[id]; !ok {
        return fmt.Errorf("entity %s not found", id)
    }
    delete(r.store, id)
    return nil
}

// FindBy は条件に一致するエンティティを検索する
func (r *InMemoryRepo[T]) FindBy(pred func(T) bool) []T {
    r.mu.RLock()
    defer r.mu.RUnlock()
    var result []T
    for _, entity := range r.store {
        if pred(entity) {
            result = append(result, entity)
        }
    }
    return result
}

// 使用例
type User struct {
    ID   string
    Name string
    Age  int
}

func (u User) GetID() string { return u.ID }

repo := NewInMemoryRepo[User]()
repo.Save(User{ID: "1", Name: "Alice", Age: 30})
repo.Save(User{ID: "2", Name: "Bob", Age: 25})
user, _ := repo.FindByID("1")

// 条件検索
adults := repo.FindBy(func(u User) bool { return u.Age >= 18 })
```

### コード例12: ジェネリックなページネーション

```go
// Page はページネーション結果を表す
type Page[T any] struct {
    Items      []T `json:"items"`
    Total      int `json:"total"`
    Page       int `json:"page"`
    PageSize   int `json:"page_size"`
    TotalPages int `json:"total_pages"`
    HasNext    bool `json:"has_next"`
    HasPrev    bool `json:"has_prev"`
}

// Paginate はスライスをページネーションする
func Paginate[T any](items []T, page, pageSize int) Page[T] {
    total := len(items)
    totalPages := (total + pageSize - 1) / pageSize

    if page < 1 {
        page = 1
    }
    if page > totalPages && totalPages > 0 {
        page = totalPages
    }

    start := (page - 1) * pageSize
    end := start + pageSize
    if start > total {
        start = total
    }
    if end > total {
        end = total
    }

    return Page[T]{
        Items:      items[start:end],
        Total:      total,
        Page:       page,
        PageSize:   pageSize,
        TotalPages: totalPages,
        HasNext:    page < totalPages,
        HasPrev:    page > 1,
    }
}

// 使用例
users := getAllUsers() // []User
page := Paginate(users, 2, 10) // 2ページ目、1ページ10件
fmt.Printf("Page %d/%d, %d items\n", page.Page, page.TotalPages, len(page.Items))
```

### コード例13: ジェネリックなキャッシュ

```go
// Cache はTTL付きのジェネリックキャッシュ
type Cache[K comparable, V any] struct {
    mu      sync.RWMutex
    items   map[K]cacheItem[V]
    ttl     time.Duration
    maxSize int
}

type cacheItem[V any] struct {
    value     V
    expiresAt time.Time
}

func NewCache[K comparable, V any](ttl time.Duration, maxSize int) *Cache[K, V] {
    return &Cache[K, V]{
        items:   make(map[K]cacheItem[V]),
        ttl:     ttl,
        maxSize: maxSize,
    }
}

func (c *Cache[K, V]) Get(key K) (V, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()

    item, ok := c.items[key]
    if !ok || time.Now().After(item.expiresAt) {
        var zero V
        return zero, false
    }
    return item.value, true
}

func (c *Cache[K, V]) Set(key K, value V) {
    c.mu.Lock()
    defer c.mu.Unlock()

    // maxSize を超えたら期限切れのアイテムを削除
    if len(c.items) >= c.maxSize {
        c.evictExpired()
    }

    c.items[key] = cacheItem[V]{
        value:     value,
        expiresAt: time.Now().Add(c.ttl),
    }
}

func (c *Cache[K, V]) Delete(key K) {
    c.mu.Lock()
    defer c.mu.Unlock()
    delete(c.items, key)
}

func (c *Cache[K, V]) evictExpired() {
    now := time.Now()
    for k, item := range c.items {
        if now.After(item.expiresAt) {
            delete(c.items, k)
        }
    }
}

// GetOrLoad はキャッシュに値がなければloader関数で取得してキャッシュする
func (c *Cache[K, V]) GetOrLoad(key K, loader func(K) (V, error)) (V, error) {
    if val, ok := c.Get(key); ok {
        return val, nil
    }

    val, err := loader(key)
    if err != nil {
        var zero V
        return zero, err
    }

    c.Set(key, val)
    return val, nil
}

// 使用例
userCache := NewCache[string, *User](5*time.Minute, 1000)
user, err := userCache.GetOrLoad("user-123", func(id string) (*User, error) {
    return db.FindUser(id)
})
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

## 5. 標準ライブラリのジェネリック関数

### コード例14: slices パッケージ

```go
import "slices"

// ソート
nums := []int{3, 1, 4, 1, 5, 9, 2, 6}
slices.Sort(nums) // [1, 1, 2, 3, 4, 5, 6, 9]

// カスタムソート
type User struct {
    Name string
    Age  int
}
users := []User{{"Charlie", 30}, {"Alice", 25}, {"Bob", 35}}
slices.SortFunc(users, func(a, b User) int {
    return cmp.Compare(a.Age, b.Age)
})
// [{Alice 25}, {Charlie 30}, {Bob 35}]

// 安定ソート（同じキーの要素の順序を保持）
slices.SortStableFunc(users, func(a, b User) int {
    return cmp.Compare(a.Name, b.Name)
})

// 二分探索
sorted := []int{1, 2, 3, 4, 5, 6, 7, 8, 9}
idx, found := slices.BinarySearch(sorted, 5) // 4, true

// 含有チェック
slices.Contains([]string{"a", "b", "c"}, "b") // true

// 最大・最小
slices.Max([]int{3, 1, 4, 1, 5}) // 5
slices.Min([]int{3, 1, 4, 1, 5}) // 1

// コンパクト（連続する重複を除去）
nums = []int{1, 1, 2, 3, 3, 3, 4}
slices.Compact(nums) // [1, 2, 3, 4]

// リバース
slices.Reverse([]int{1, 2, 3}) // [3, 2, 1]

// インデックス検索
slices.Index([]string{"a", "b", "c"}, "b") // 1

// 等値比較
slices.Equal([]int{1, 2, 3}, []int{1, 2, 3}) // true

// クローン
original := []int{1, 2, 3}
cloned := slices.Clone(original) // ディープコピー
```

### コード例15: maps パッケージ

```go
import "maps"

m := map[string]int{"a": 1, "b": 2, "c": 3}

// キー一覧
keys := maps.Keys(m) // イテレータを返す（Go 1.23+）

// 値一覧
values := maps.Values(m) // イテレータを返す

// クローン
cloned := maps.Clone(m) // 浅いコピー

// 等値比較
maps.Equal(m, cloned) // true

// コピー（dstにsrcをマージ）
dst := map[string]int{"a": 10, "d": 4}
maps.Copy(dst, m) // dst = {"a": 1, "b": 2, "c": 3, "d": 4}

// 条件による削除
maps.DeleteFunc(m, func(k string, v int) bool {
    return v < 2
})
// m = {"b": 2, "c": 3}
```

### コード例16: cmp パッケージ

```go
import "cmp"

// 比較
cmp.Compare(1, 2)     // -1
cmp.Compare(2, 2)     //  0
cmp.Compare(3, 2)     //  1

// ゼロ値チェック
cmp.Or(0, 42)         // 42（最初の非ゼロ値）
cmp.Or("", "default") // "default"
cmp.Or("hello", "default") // "hello"

// 複数フォールバック
cmp.Or("", "", "fallback") // "fallback"

// ソートキーの合成
type Employee struct {
    Department string
    Name       string
    Salary     int
}

employees := []Employee{...}
slices.SortFunc(employees, func(a, b Employee) int {
    // まず部門でソート、同じなら名前でソート
    if c := cmp.Compare(a.Department, b.Department); c != 0 {
        return c
    }
    return cmp.Compare(a.Name, b.Name)
})
```

---

## 6. パフォーマンス特性

### GCShape Stenciling

```
+----------------------------------------------------------+
|  Go ジェネリクスのコンパイル戦略                            |
+----------------------------------------------------------+
|                                                          |
|  func Max[T cmp.Ordered](a, b T) T                      |
|                                                          |
|  コンパイル時:                                            |
|  +-------------------+  +-------------------+            |
|  | ポインタ型         |  | 値型              |            |
|  | (*User, *string等)|  | (int, float64等)  |            |
|  | → 共通の実装を共有 |  | → 型ごとに特殊化  |            |
|  +-------------------+  +-------------------+            |
|                                                          |
|  GCShape = 同じメモリレイアウトの型は同じ実装を共有        |
|  → コードサイズの爆発を防ぐ                               |
|  → ポインタ型はすべて同じ shape                           |
+----------------------------------------------------------+
```

### コード例17: ベンチマークによるパフォーマンス比較

```go
// インターフェース版
func SumInterface(nums []interface{}) int {
    sum := 0
    for _, n := range nums {
        sum += n.(int)
    }
    return sum
}

// ジェネリック版
func SumGeneric[T Number](nums []T) T {
    var sum T
    for _, n := range nums {
        sum += n
    }
    return sum
}

// 具体型版
func SumInt(nums []int) int {
    sum := 0
    for _, n := range nums {
        sum += n
    }
    return sum
}

// ベンチマーク
func BenchmarkSumInterface(b *testing.B) {
    nums := make([]interface{}, 1000)
    for i := range nums { nums[i] = i }
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        SumInterface(nums)
    }
}

func BenchmarkSumGeneric(b *testing.B) {
    nums := make([]int, 1000)
    for i := range nums { nums[i] = i }
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        SumGeneric(nums)
    }
}

func BenchmarkSumConcrete(b *testing.B) {
    nums := make([]int, 1000)
    for i := range nums { nums[i] = i }
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        SumInt(nums)
    }
}

// 典型的な結果:
// BenchmarkSumInterface-8   500000  2800 ns/op  0 B/op  0 allocs/op
// BenchmarkSumGeneric-8    2000000   600 ns/op  0 B/op  0 allocs/op
// BenchmarkSumConcrete-8   2000000   580 ns/op  0 B/op  0 allocs/op
// → ジェネリクスは具体型とほぼ同等、interfaceより大幅に高速
```

---

## 7. ジェネリクスの適用判断

### ジェネリクスを使うべき場面

```
+----------------------------------------------------------+
|  ジェネリクスの適用判断フロー                               |
+----------------------------------------------------------+
|                                                          |
|  同じロジックを異なる型に適用したい？                      |
|    |                                                     |
|    +-- YES → 型パラメータが2つ以上の具体型で使われる？     |
|    |           |                                         |
|    |           +-- YES → ジェネリクスが適切              |
|    |           +-- NO  → 具体型を直接使う                |
|    |                                                     |
|    +-- NO  → 異なる実装を同じ振る舞いに抽象化したい？     |
|              |                                           |
|              +-- YES → インターフェースが適切            |
|              +-- NO  → ジェネリクスは不要                |
+----------------------------------------------------------+
```

| 場面 | 推奨 | 理由 |
|------|------|------|
| コレクション操作（Map, Filter, Reduce） | ジェネリクス | 同じアルゴリズムを全ての型に適用 |
| データ構造（Stack, Queue, Tree） | ジェネリクス | 型安全なコンテナ |
| DB接続の抽象化 | インターフェース | 実装が異なる（MySQL vs PostgreSQL） |
| HTTPハンドラ | インターフェース | http.Handler パターン |
| ソートアルゴリズム | ジェネリクス | 比較可能な全ての型に対応 |
| ロガー | インターフェース | 出力先が異なる |
| `fmt.Println(v any)` のような関数 | `any` 引数 | ジェネリクスは不要 |

---

## 8. アンチパターン

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

### アンチパターン3: ジェネリクスで多態性を実現しようとする

```go
// NG: ジェネリクスで振る舞いの切り替え
func Process[T Animal](a T) string {
    // T の具体型によって処理を変えたい
    // → ジェネリクスでは型に基づくディスパッチはできない
}

// OK: インターフェースを使う
type Animal interface {
    Speak() string
}

func Process(a Animal) string {
    return a.Speak()
}
```

### アンチパターン4: ゼロ値の誤った扱い

```go
// NG: ジェネリックなゼロ値チェック
func IsZero[T any](v T) bool {
    // any にはゼロ値比較の演算がない → コンパイルエラー
    return v == T{} // 不可
}

// OK: comparable 制約を使う
func IsZero[T comparable](v T) bool {
    var zero T
    return v == zero
}

// OK: reflect を使う（any の場合）
func IsZeroAny(v any) bool {
    return reflect.ValueOf(v).IsZero()
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

### Q4. 型パラメータにメソッドを定義できるか？

型パラメータ自体にはメソッドを定義できない。ただし、ジェネリック型（例: `Stack[T any]`）にはメソッドを定義可能。メソッドの型パラメータは型定義で宣言されたものを使い、メソッド宣言で新しい型パラメータを追加することはできない。

```go
type Stack[T any] struct { items []T }

// OK: 型定義の T を使う
func (s *Stack[T]) Push(item T) { ... }

// NG: メソッドに新しい型パラメータを追加
func (s *Stack[T]) Map[U any](f func(T) U) *Stack[U] { ... } // コンパイルエラー

// OK: 関数として定義する
func MapStack[T, U any](s *Stack[T], f func(T) U) *Stack[U] { ... }
```

### Q5. ジェネリクスと reflect の使い分けは？

ジェネリクスはコンパイル時の型安全性を保証し、パフォーマンスも良好。reflect はランタイムの型情報にアクセスでき柔軟性が高いが、型安全性がなくパフォーマンスも劣る。原則としてジェネリクスで解決できる場合はジェネリクスを使い、JSON マーシャリングやORM のようにランタイムで型を動的に扱う必要がある場合のみ reflect を使う。

```go
// ジェネリクスで解決できるケース → ジェネリクスを使う
func Contains[T comparable](slice []T, target T) bool {
    for _, v := range slice {
        if v == target {
            return true
        }
    }
    return false
}

// reflect が必要なケース → 構造体のフィールドを動的に走査
func StructToMap(v any) map[string]any {
    result := make(map[string]any)
    val := reflect.ValueOf(v)
    typ := val.Type()
    for i := 0; i < val.NumField(); i++ {
        field := typ.Field(i)
        if field.IsExported() {
            result[field.Name] = val.Field(i).Interface()
        }
    }
    return result
}
```

### Q6. ジェネリクスで再帰的な型制約は可能か？

Go 1.18時点では直接的な再帰制約はサポートされていないが、間接的に実現可能。

```go
// 自己参照型のパターン
type Comparable[T any] interface {
    CompareTo(other T) int
}

// 使用例
type MyString string

func (s MyString) CompareTo(other MyString) int {
    return strings.Compare(string(s), string(other))
}

func Sort[T Comparable[T]](items []T) {
    slices.SortFunc(items, func(a, b T) int {
        return a.CompareTo(b)
    })
}
```

---

### Q7. union 型制約内のメソッドは呼び出せるか？

union 型（`int | string` など）はメソッドを持たないため、union 型制約のみではメソッド呼び出しはできない。メソッドを呼び出したい場合は、インターフェースメソッドを制約に追加する必要がある。

```go
// NG: union 型にはメソッドがない
type Numeric interface {
    ~int | ~float64
}

func Double[T Numeric](v T) string {
    return v.String() // コンパイルエラー: String() は定義されていない
}

// OK: メソッドを制約に含める
type StringableNumeric interface {
    ~int | ~float64
    String() string
}
```

### Q8. ジェネリクスでイテレータパターンは実現できるか？

Go 1.23 以降の range over function（レンジ関数）と組み合わせることで、型安全なイテレータを実装できる。

```go
// iter.Seq を使ったジェネリックイテレータ（Go 1.23+）
func Filter[T any](seq iter.Seq[T], predicate func(T) bool) iter.Seq[T] {
    return func(yield func(T) bool) {
        for v := range seq {
            if predicate(v) {
                if !yield(v) {
                    return
                }
            }
        }
    }
}

func Map[T, U any](seq iter.Seq[T], transform func(T) U) iter.Seq[U] {
    return func(yield func(U) bool) {
        for v := range seq {
            if !yield(transform(v)) {
                return
            }
        }
    }
}

// 使用例
numbers := slices.Values([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
evens := Filter(numbers, func(n int) bool { return n%2 == 0 })
doubled := Map(evens, func(n int) int { return n * 2 })
for v := range doubled {
    fmt.Println(v) // 4, 8, 12, 16, 20
}
```

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
| Result / Optional | エラーハンドリング・nil安全のジェネリック型 |
| GCShape stenciling | コンパイル時の型特殊化戦略 |

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
5. **Go標準ライブラリ — maps パッケージ** https://pkg.go.dev/maps
6. **Go標準ライブラリ — cmp パッケージ** https://pkg.go.dev/cmp
