# Go Generics Guide

> Using type parameters and constraints introduced in Go 1.18 to write type-safe, reusable code

## What You'll Learn in This Chapter

1. **Type parameters** — syntax and basic usage (generic functions and types)
2. **Constraints** — how to define them and the standard library constraints package
3. **Practical patterns** — collection operations, repository pattern, Result type implementation
4. **Standard library** — leveraging the `slices`, `maps`, and `cmp` packages
5. **Performance characteristics** and criteria for deciding when to apply generics


## Prerequisites

Your understanding will deepen if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content of [Go CLI Development Guide](./00-cli-development.md)

---

## 1. Generics Basics

### Before and After Generics

```
[Before] Duplicating functions for each type

func MaxInt(a, b int) int         { if a > b { return a }; return b }
func MaxFloat(a, b float64) float64 { if a > b { return a }; return b }
func MaxString(a, b string) string  { if a > b { return a }; return b }

           ↓ Unified with generics

[After] A single function handles all types

func MaxT cmp.Ordered T { if a > b { return a }; return b }
```

### Type Parameter Syntax

```
+-------- Type parameter list --------+
|                                     |
func  FuncName  T  constraint  returns
                 |       |
                 |       +--- Constraint: conditions T must satisfy
                 +----------- Type parameter name
```

### Code Example 1: Your First Generic Function

```go
package main

import (
    "cmp"
    "fmt"
)

// T is any type satisfying cmp.Ordered
func MaxT cmp.Ordered T {
    if a > b {
        return a
    }
    return b
}

func MinT cmp.Ordered T {
    if a < b {
        return a
    }
    return b
}

func ClampT cmp.Ordered T {
    return Max(lo, Min(val, hi))
}

func main() {
    fmt.Println(Max(3, 7))           // 7
    fmt.Println(Max(3.14, 2.71))     // 3.14
    fmt.Println(Max("apple", "banana")) // banana
    fmt.Println(Clamp(150, 0, 100))  // 100
}
```

### Code Example 2: Generic Slice Operations

```go
// Map applies a function to each element of a slice
func MapT, U any U) []U {
    result := make([]U, len(s))
    for i, v := range s {
        result[i] = f(v)
    }
    return result
}

// Filter extracts elements from a slice that satisfy a condition
func FilterT any bool) []T {
    var result []T
    for _, v := range s {
        if pred(v) {
            result = append(result, v)
        }
    }
    return result
}

// Reduce aggregates a slice into a single value
func ReduceT, U any U) U {
    acc := init
    for _, v := range s {
        acc = f(acc, v)
    }
    return acc
}

// Find returns the first element that satisfies the condition
func FindT any bool) (T, bool) {
    for _, v := range s {
        if pred(v) {
            return v, true
        }
    }
    var zero T
    return zero, false
}

// GroupBy groups elements based on a key function
func GroupByT any, K comparable K) map[K][]T {
    result := make(map[K][]T)
    for _, v := range s {
        key := keyFn(v)
        result[key] = append(result[key], v)
    }
    return result
}

// Chunk splits a slice into chunks of the specified size
func ChunkT any [][]T {
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

// Unique returns a slice with duplicates removed
func UniqueT comparable []T {
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

// Usage examples
func main() {
    nums := []int{1, 2, 3, 4, 5}
    doubled := Map(nums, func(n int) int { return n * 2 })
    // [2, 4, 6, 8, 10]

    evens := Filter(nums, func(n int) bool { return n%2 == 0 })
    // [2, 4]

    sum := Reduce(nums, 0, func(acc, n int) int { return acc + n })
    // 15

    // String operations
    words := []string{"hello", "world", "go", "generics"}
    lengths := Map(words, func(s string) int { return len(s) })
    // [5, 5, 2, 8]

    longWords := Filter(words, func(s string) bool { return len(s) > 3 })
    // ["hello", "world", "generics"]

    // Grouping
    type User struct {
        Name string
        Role string
    }
    users := []User{
        {"Alice", "admin"}, {"Bob", "user"}, {"Charlie", "admin"}, {"Dave", "user"},
    }
    byRole := GroupBy(users, func(u User) string { return u.Role })
    // map["admin":[Alice, Charlie] "user":[Bob, Dave]]

    // Deduplication
    ids := []int{1, 2, 3, 2, 1, 4, 3, 5}
    unique := Unique(ids) // [1, 2, 3, 4, 5]
}
```

### Code Example 3: FlatMap and Zip

```go
// FlatMap converts each element of a slice into a slice and flattens the result
func FlatMapT, U any []U) []U {
    var result []U
    for _, v := range s {
        result = append(result, f(v)...)
    }
    return result
}

// Zip pairs up elements from two slices
func ZipT, U any []Pair[T, U] {
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

// Partition splits a slice into two based on a condition
func PartitionT any bool) (matched, unmatched []T) {
    for _, v := range s {
        if pred(v) {
            matched = append(matched, v)
        } else {
            unmatched = append(unmatched, v)
        }
    }
    return
}

// Usage examples
func example() {
    // FlatMap: split sentences into tokens
    sentences := []string{"hello world", "go generics"}
    tokens := FlatMap(sentences, func(s string) []string {
        return strings.Split(s, " ")
    })
    // ["hello", "world", "go", "generics"]

    // Zip: pair names with scores
    names := []string{"Alice", "Bob", "Charlie"}
    scores := []int{90, 85, 95}
    pairs := Zip(names, scores)
    // [{Alice, 90}, {Bob, 85}, {Charlie, 95}]

    // Partition: separate pass and fail
    pass, fail := Partition(scores, func(s int) bool { return s >= 90 })
    // pass: [90, 95], fail: [85]
}
```

---

## 2. Constraints

### Types of Constraints

```
+-------------------+
|   any (interface{})|  ← Loosest: accepts every type
+-------------------+
        |
+-------------------+
|   comparable      |  ← Types that support == and !=
+-------------------+
        |
+-------------------+
|   cmp.Ordered     |  ← Types that support comparison operators (<, >, <=, >=)
+-------------------+
        |
+-------------------+
|  Custom constraint |  ← Requires specific methods or types
+-------------------+
```

### Code Example 4: Defining Custom Constraints

```go
// Method-based constraint
type Stringer interface {
    String() string
}

// Type set-based constraint (union)
type Number interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
    ~float32 | ~float64
}

// Constraint for integers only
type Integer interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

// Constraint for floating-point only
type Float interface {
    ~float32 | ~float64
}

// The tilde (~) designates an underlying type
// ~int includes "all types whose underlying type is int"
type MyInt int      // included in ~int
type Score int      // included in ~int

// Composite constraint: methods + type set
type OrderedStringer interface {
    cmp.Ordered
    String() string
}

// Practical example: Sum function
func SumT Number T {
    var total T
    for _, n := range nums {
        total += n
    }
    return total
}

// Practical example: Average function (returns floating-point)
func AverageT Number float64 {
    if len(nums) == 0 {
        return 0
    }
    var sum T
    for _, n := range nums {
        sum += n
    }
    return float64(sum) / float64(len(nums))
}

// Practical example: Abs function (signed numbers)
type Signed interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64
}

func AbsT Signed T {
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

### Comparison Table of Major Constraints

| Constraint | Allowed types | Available operations | Use cases |
|------|------------|-----------|------|
| `any` | All types | None (only through interface) | Containers, wrappers |
| `comparable` | Comparable types | `==`, `!=` | Map keys, deduplication |
| `cmp.Ordered` | Ordered types | `<`, `>`, `<=`, `>=`, `==` | Sorting, min/max |
| `~int \| ~float64` | Types with the specified underlying type | Numeric operations | Calculation, aggregation |
| Custom interface | Types with methods | Specified methods | Domain-specific logic |

### With/Without ~ (Tilde) Comparison Table

| Constraint definition | `int` | `type MyInt int` | `type Score int` |
|---------|-------|-----------------|-----------------|
| `int` | Match | Mismatch | Mismatch |
| `~int` | Match | Match | Match |

### Code Example 5: Composite Constraints in Practice

```go
// Constraint that satisfies both Comparable and Stringer
type ComparableStringer interface {
    comparable
    String() string
}

// Types that can be used as map keys and have string representations
func PrintMapK ComparableStringer, V any {
    for k, v := range m {
        fmt.Printf("%s: %v\n", k.String(), v)
    }
}

// Composition of constraint interfaces
type Numeric interface {
    Integer | Float
}

type Addable interface {
    Numeric
    comparable
}

// Constraint for JSON-serializable types
type JSONSerializable interface {
    comparable
    MarshalJSON() ([]byte, error)
    UnmarshalJSON([]byte) error
}

// Constraint for validatable types
type Validatable interface {
    Validate() error
}

// Save function with validation
func SaveAllT Validatable error {
    for i, item := range items {
        if err := item.Validate(); err != nil {
            return fmt.Errorf("item[%d]: %w", i, err)
        }
    }
    // Save processing...
    return nil
}
```

---

## 3. Generic Types

### Code Example 6: Generic Data Structures

```go
// Stack
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

// Usage examples
intStack := NewStack[int]()
intStack.Push(1)
intStack.Push(2)
val, _ := intStack.Pop() // 2

strStack := NewStack[string]()
strStack.Push("hello")
```

### Code Example 7: Generic Queue

```go
// Queue is a generic FIFO queue
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

// PriorityQueue is a priority queue
type PriorityQueue[T any] struct {
    items []T
    less  func(a, b T) bool
}

func NewPriorityQueueT any bool) *PriorityQueue[T] {
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

// Usage example
pq := NewPriorityQueue(func(a, b int) bool { return a < b })
pq.Push(3)
pq.Push(1)
pq.Push(2)
val, _ := pq.Pop() // 1 (the smallest value comes out first)
```

### Code Example 8: Generic Concurrency-Safe Map

```go
// SyncMap is a generic concurrency-safe map
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

// GetOrSet sets and returns the value if it doesn't exist
func (sm *SyncMap[K, V]) GetOrSet(key K, defaultVal V) V {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    if val, ok := sm.m[key]; ok {
        return val
    }
    sm.m[key] = defaultVal
    return defaultVal
}

// Usage example
cache := NewSyncMap[string, int]()
cache.Set("count", 42)
val, ok := cache.Get("count") // 42, true
```

### Code Example 9: Result Type (Improved Error Handling)

```go
// Result is a type that holds either an error or a value
type Result[T any] struct {
    value T
    err   error
}

func OkT any Result[T] {
    return Result[T]{value: value}
}

func ErrT any Result[T] {
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

// Map: applies the transformation only when there is a value
func MapResultT, U any U) Result[U] {
    if r.err != nil {
        return ErrU
    }
    return Ok(f(r.value))
}

// FlatMap: applies another Result-producing function when there is a value
func FlatMapResultT, U any Result[U]) Result[U] {
    if r.err != nil {
        return ErrU
    }
    return f(r.value)
}

// Collect: gathers success values from a slice of Results (fails if any one errors)
func CollectT any Result[[]T] {
    values := make([]T, 0, len(results))
    for _, r := range results {
        if r.IsErr() {
            return Err[[]T](r.err)
        }
        values = append(values, r.value)
    }
    return Ok(values)
}

// Usage examples
result := Ok(42)
doubled := MapResult(result, func(n int) int { return n * 2 })
val, _ := doubled.Unwrap() // 84

// Chaining
func fetchUser(id string) Result[User] {
    user, err := db.FindUser(id)
    if err != nil {
        return ErrUser
    }
    return Ok(*user)
}

func getEmail(u User) Result[string] {
    if u.Email == "" {
        return Errstring)
    }
    return Ok(u.Email)
}

// Result chain
email := FlatMapResult(fetchUser("123"), getEmail)
fmt.Println(email.UnwrapOr("no-email@example.com"))
```

### Code Example 10: Optional Type

```go
// Optional is a nil-safe value container
type Optional[T any] struct {
    value *T
}

func SomeT any Optional[T] {
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

func MapOptionalT, U any U) Optional[U] {
    if o.value == nil {
        return None[U]()
    }
    return Some(f(*o.value))
}

// Usage examples
name := Some("Alice")
name.IfPresent(func(n string) {
    fmt.Printf("Hello, %s!\n", n)
})

empty := None[string]()
fmt.Println(empty.OrElse("anonymous")) // "anonymous"
```

---

## 4. Practical Patterns

### Code Example 11: Generic Repository Pattern

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

// In-memory implementation
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

// FindBy searches for entities matching a condition
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

// Usage examples
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

// Conditional search
adults := repo.FindBy(func(u User) bool { return u.Age >= 18 })
```

### Code Example 12: Generic Pagination

```go
// Page represents a pagination result
type Page[T any] struct {
    Items      []T `json:"items"`
    Total      int `json:"total"`
    Page       int `json:"page"`
    PageSize   int `json:"page_size"`
    TotalPages int `json:"total_pages"`
    HasNext    bool `json:"has_next"`
    HasPrev    bool `json:"has_prev"`
}

// Paginate paginates a slice
func PaginateT any Page[T] {
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

// Usage example
users := getAllUsers() // []User
page := Paginate(users, 2, 10) // Page 2, 10 items per page
fmt.Printf("Page %d/%d, %d items\n", page.Page, page.TotalPages, len(page.Items))
```

### Code Example 13: Generic Cache

```go
// Cache is a generic cache with TTL
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

func NewCacheK comparable, V any *Cache[K, V] {
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

    // Evict expired items when maxSize is exceeded
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

// GetOrLoad retrieves via loader and caches the result if the value is absent
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

// Usage example
userCache := NewCachestring, *User
user, err := userCache.GetOrLoad("user-123", func(id string) (*User, error) {
    return db.FindUser(id)
})
```

### Type Inference Flow for Generics

```
Max(3, 7)
   |
   +-- Compiler infers the types of the arguments
   |     3 → int,  7 → int
   |
   +-- Determined: T = int
   |
   +-- Expanded as Maxint
   |
   +-- Does int satisfy cmp.Ordered? → Yes
   |
   +-- Compilation succeeds

Max(3, 7.0)
   |
   +-- 3 → int,  7.0 → float64
   |
   +-- Types don't match → compile error
   |
   +-- Fix: Max(float64(3), 7.0) or Maxfloat64
```

---

## 5. Generic Functions in the Standard Library

### Code Example 14: slices Package

```go
import "slices"

// Sorting
nums := []int{3, 1, 4, 1, 5, 9, 2, 6}
slices.Sort(nums) // [1, 1, 2, 3, 4, 5, 6, 9]

// Custom sort
type User struct {
    Name string
    Age  int
}
users := []User{{"Charlie", 30}, {"Alice", 25}, {"Bob", 35}}
slices.SortFunc(users, func(a, b User) int {
    return cmp.Compare(a.Age, b.Age)
})
// [{Alice 25}, {Charlie 30}, {Bob 35}]

// Stable sort (preserves order of elements with the same key)
slices.SortStableFunc(users, func(a, b User) int {
    return cmp.Compare(a.Name, b.Name)
})

// Binary search
sorted := []int{1, 2, 3, 4, 5, 6, 7, 8, 9}
idx, found := slices.BinarySearch(sorted, 5) // 4, true

// Containment check
slices.Contains([]string{"a", "b", "c"}, "b") // true

// Max / min
slices.Max([]int{3, 1, 4, 1, 5}) // 5
slices.Min([]int{3, 1, 4, 1, 5}) // 1

// Compact (removes consecutive duplicates)
nums = []int{1, 1, 2, 3, 3, 3, 4}
slices.Compact(nums) // [1, 2, 3, 4]

// Reverse
slices.Reverse([]int{1, 2, 3}) // [3, 2, 1]

// Index lookup
slices.Index([]string{"a", "b", "c"}, "b") // 1

// Equality comparison
slices.Equal([]int{1, 2, 3}, []int{1, 2, 3}) // true

// Clone
original := []int{1, 2, 3}
cloned := slices.Clone(original) // deep copy
```

### Code Example 15: maps Package

```go
import "maps"

m := map[string]int{"a": 1, "b": 2, "c": 3}

// List of keys
keys := maps.Keys(m) // returns an iterator (Go 1.23+)

// List of values
values := maps.Values(m) // returns an iterator

// Clone
cloned := maps.Clone(m) // shallow copy

// Equality comparison
maps.Equal(m, cloned) // true

// Copy (merges src into dst)
dst := map[string]int{"a": 10, "d": 4}
maps.Copy(dst, m) // dst = {"a": 1, "b": 2, "c": 3, "d": 4}

// Conditional deletion
maps.DeleteFunc(m, func(k string, v int) bool {
    return v < 2
})
// m = {"b": 2, "c": 3}
```

### Code Example 16: cmp Package

```go
import "cmp"

// Comparison
cmp.Compare(1, 2)     // -1
cmp.Compare(2, 2)     //  0
cmp.Compare(3, 2)     //  1

// Zero value check
cmp.Or(0, 42)         // 42 (first non-zero value)
cmp.Or("", "default") // "default"
cmp.Or("hello", "default") // "hello"

// Multiple fallbacks
cmp.Or("", "", "fallback") // "fallback"

// Compositing sort keys
type Employee struct {
    Department string
    Name       string
    Salary     int
}

employees := []Employee{...}
slices.SortFunc(employees, func(a, b Employee) int {
    // Sort by department first, then by name if equal
    if c := cmp.Compare(a.Department, b.Department); c != 0 {
        return c
    }
    return cmp.Compare(a.Name, b.Name)
})
```

---

## 6. Performance Characteristics

### GCShape Stenciling

```
+----------------------------------------------------------+
|  Go generics compilation strategy                        |
+----------------------------------------------------------+
|                                                          |
|  func MaxT cmp.Ordered T                                 |
|                                                          |
|  At compile time:                                        |
|  +-------------------+  +-------------------+            |
|  | Pointer types     |  | Value types       |            |
|  | (*User, *string, etc.) | (int, float64, etc.) |       |
|  | → share a common impl. | → specialized per type |     |
|  +-------------------+  +-------------------+            |
|                                                          |
|  GCShape = types with the same memory layout share one impl. |
|  → Prevents code size explosion                          |
|  → All pointer types share the same shape                |
+----------------------------------------------------------+
```

### Code Example 17: Performance Comparison via Benchmarks

```go
// Interface version
func SumInterface(nums []interface{}) int {
    sum := 0
    for _, n := range nums {
        sum += n.(int)
    }
    return sum
}

// Generic version
func SumGenericT Number T {
    var sum T
    for _, n := range nums {
        sum += n
    }
    return sum
}

// Concrete-type version
func SumInt(nums []int) int {
    sum := 0
    for _, n := range nums {
        sum += n
    }
    return sum
}

// Benchmarks
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

// Typical results:
// BenchmarkSumInterface-8   500000  2800 ns/op  0 B/op  0 allocs/op
// BenchmarkSumGeneric-8    2000000   600 ns/op  0 B/op  0 allocs/op
// BenchmarkSumConcrete-8   2000000   580 ns/op  0 B/op  0 allocs/op
// → Generics perform roughly on par with concrete types, far faster than interface
```

---

## 7. Deciding When to Apply Generics

### When to Use Generics

```
+----------------------------------------------------------+
|  Decision flow for applying generics                     |
+----------------------------------------------------------+
|                                                          |
|  Want to apply the same logic to different types?        |
|    |                                                     |
|    +-- YES → Will the type parameter be used with 2+ concrete types? |
|    |           |                                         |
|    |           +-- YES → Generics are appropriate        |
|    |           +-- NO  → Use concrete types directly     |
|    |                                                     |
|    +-- NO  → Want to abstract different implementations into the same behavior? |
|              |                                           |
|              +-- YES → Use an interface                  |
|              +-- NO  → Generics aren't needed            |
+----------------------------------------------------------+
```

| Scenario | Recommendation | Rationale |
|------|------|------|
| Collection operations (Map, Filter, Reduce) | Generics | Same algorithm applied to all types |
| Data structures (Stack, Queue, Tree) | Generics | Type-safe containers |
| Abstracting DB connections | Interface | Implementations differ (MySQL vs PostgreSQL) |
| HTTP handlers | Interface | http.Handler pattern |
| Sorting algorithms | Generics | Supports all comparable types |
| Loggers | Interface | Different output destinations |
| Functions like `fmt.Println(v any)` | `any` parameter | Generics aren't needed |

---

## 8. Anti-Patterns

### Anti-Pattern 1: Unnecessary Use of Generics

```go
// BAD: a case where generics aren't needed
func PrintValueT any {
    fmt.Println(v) // For any, interface{} would suffice
}

// GOOD: use interface{} or any directly
func PrintValue(v any) {
    fmt.Println(v)
}

// BAD: type parameter used with only one concrete type
func ParseUserJSONT User (T, error) {
    var result T
    err := json.Unmarshal(data, &result)
    return result, err
}

// GOOD: use the concrete type directly
func ParseUserJSON(data []byte) (User, error) {
    var user User
    err := json.Unmarshal(data, &user)
    return user, err
}
```

### Anti-Pattern 2: Overly Complex Constraints

```go
// BAD: constraint is too complex and hard to read
type ComplexConstraint[K comparable, V interface {
    ~int | ~string
    fmt.Stringer
    encoding.BinaryMarshaler
}] struct {
    data map[K]V
}

// GOOD: split constraints and name them
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

### Anti-Pattern 3: Trying to Achieve Polymorphism with Generics

```go
// BAD: switching behavior with generics
func ProcessT Animal string {
    // Wanting to change behavior based on the concrete type of T
    // → Generics don't support type-based dispatch
}

// GOOD: use an interface
type Animal interface {
    Speak() string
}

func Process(a Animal) string {
    return a.Speak()
}
```

### Anti-Pattern 4: Mishandling Zero Values

```go
// BAD: generic zero-value check
func IsZeroT any bool {
    // any doesn't support zero-value comparison → compile error
    return v == T{} // not allowed
}

// GOOD: use the comparable constraint
func IsZeroT comparable bool {
    var zero T
    return v == zero
}

// GOOD: use reflect (for the any case)
func IsZeroAny(v any) bool {
    return reflect.ValueOf(v).IsZero()
}
```


---

## Practice Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate the input data
- Implement error handling appropriately
- Write test code as well

```python
# Exercise 1: basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Look up by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistical information"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient: {slow_time:.4f} sec")
    print(f"Efficient:   {fast_time:.6f} sec")
    print(f"Speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure improvements with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Incorrect configuration file | Verify the configuration file path and format |
| Timeout | Network latency/resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Growing data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check the executing user's permissions, review settings |
| Data inconsistency | Concurrency conflicts | Introduce locking, manage transactions |

### Debugging Procedure

1. **Check the error message**: Read the stack trace and identify where the error occurred
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: Enumerate possible causes
4. **Verify incrementally**: Use log output and a debugger to test the hypotheses
5. **Fix and regression test**: After fixing, run tests for related areas as well

```python
# Debugging utilities
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function inputs and outputs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return value: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception raised: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Problems

Diagnostic procedure when performance problems occur:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Look for memory leaks
3. **Check for I/O waits**: Review disk and network I/O status
4. **Check concurrent connections**: Inspect the state of the connection pool

| Problem type | Diagnostic tools | Countermeasures |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithmic improvements, parallelization |
| Memory leaks | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Asynchronous I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexes, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes criteria to apply when making technology choices.

| Criterion | When to prioritize | When compromises are acceptable |
|---------|------------|-------------|
| Performance | Real-time processing, large-scale data | Admin dashboards, batch processing |
| Maintainability | Long-term operations, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal information, financial data | Public data, internal use |
| Development speed | MVPs, time-to-market | Quality-first, mission-critical systems |

### Choosing an Architecture Pattern

```
┌─────────────────────────────────────────────────┐
│          Architecture selection flow             │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① What is the team size?                        │
│    ├─ Small (1-5) → Monolith                     │
│    └─ Large (10+) → go to ②                      │
│                                                 │
│  ② What is the deployment frequency?             │
│    ├─ Weekly or less → Monolith + modular split  │
│    └─ Daily/multiple → go to ③                   │
│                                                 │
│  ③ How independent are the teams?                │
│    ├─ High → Microservices                       │
│    └─ Moderate → Modular monolith                │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Analyzing Trade-offs

Technical decisions always come with trade-offs. Analyze them from the following perspectives:

**1. Short-term vs. long-term cost**
- A method that is fast in the short term may become technical debt in the long run
- Conversely, over-engineering carries high short-term cost and can delay the project

**2. Consistency vs. flexibility**
- A unified tech stack has low learning cost
- Adopting diverse technologies enables the right tool for the job but increases operational cost

**3. Level of abstraction**
- Higher abstraction increases reusability but can make debugging harder
- Lower abstraction is intuitive but tends to produce code duplication

```python
# Template for recording design decisions
class ArchitectureDecisionRecord:
    """Create an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe the background and problem"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Background\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## FAQ

### Q1. How do I choose between generics and interfaces?

Use generics when "applying the same algorithm to different types." Use interfaces when "abstracting different implementations as the same behavior." For example, sorting algorithms are a good fit for generics. On the other hand, polymorphism such as a database connection is a good fit for interfaces.

### Q2. Do generics affect performance?

Go's generics perform GCShape stenciling (shape-based specialization) at compile time. Pointer types share a common implementation, while value types are specialized as needed. In most cases they are as fast as, or faster than, invocations through an interface.

### Q3. How are generics used in the standard library since Go 1.18?

The `slices` package (sort, search, compare), the `maps` package (get keys, get values, clone), and the `cmp` package (comparison functions) have been added. There is no generic version of `sync.Map` in the standard library, but third-party packages provide one.

### Q4. Can you define methods on a type parameter?

You cannot define methods on a type parameter itself. However, you can define methods on a generic type (e.g., `Stack[T any]`). Methods use the type parameters declared in the type definition; you cannot add new type parameters in a method declaration.

```go
type Stack[T any] struct { items []T }

// OK: use T from the type definition
func (s *Stack[T]) Push(item T) { ... }

// BAD: adding a new type parameter to a method
func (s *Stack[T]) MapU any U) *Stack[U] { ... } // compile error

// OK: define it as a function
func MapStackT, U any U) *Stack[U] { ... }
```

### Q5. How do I choose between generics and reflect?

Generics guarantee compile-time type safety and offer good performance. reflect provides access to runtime type information and is very flexible, but it lacks type safety and has worse performance. As a rule, use generics when they can solve the problem; use reflect only when you need to handle types dynamically at runtime, such as in JSON marshaling or ORMs.

```go
// Case solvable with generics → use generics
func ContainsT comparable bool {
    for _, v := range slice {
        if v == target {
            return true
        }
    }
    return false
}

// Case that requires reflect → dynamically traverse struct fields
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

### Q6. Are recursive type constraints possible with generics?

As of Go 1.18, direct recursive constraints are not supported, but you can achieve them indirectly.

```go
// Self-referential type pattern
type Comparable[T any] interface {
    CompareTo(other T) int
}

// Usage example
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

### Q7. Can I call methods inside a union-type constraint?

Union types (such as `int | string`) do not have methods, so you cannot call methods using only a union-type constraint. To call methods, you need to add interface methods to the constraint.

```go
// BAD: union types have no methods
type Numeric interface {
    ~int | ~float64
}

func DoubleT Numeric string {
    return v.String() // compile error: String() is not defined
}

// OK: include methods in the constraint
type StringableNumeric interface {
    ~int | ~float64
    String() string
}
```

### Q8. Can the iterator pattern be implemented with generics?

By combining generics with range-over-function (range functions) in Go 1.23 and later, you can implement type-safe iterators.

```go
// Generic iterator using iter.Seq (Go 1.23+)
func FilterT any bool) iter.Seq[T] {
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

func MapT, U any U) iter.Seq[U] {
    return func(yield func(U) bool) {
        for v := range seq {
            if !yield(transform(v)) {
                return
            }
        }
    }
}

// Usage example
numbers := slices.Values([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
evens := Filter(numbers, func(n int) bool { return n%2 == 0 })
doubled := Map(evens, func(n int) int { return n * 2 })
for v := range doubled {
    fmt.Println(v) // 4, 8, 12, 16, 20
}
```

---

## Summary

| Concept | Key points |
|------|------|
| Type parameters `[T ...]` | Generalize functions and types over multiple types |
| `any` | Constraint that allows any type (= `interface{}`) |
| `comparable` | Constraint for types that support `==` / `!=` |
| `cmp.Ordered` | Constraint for types that support comparison operators |
| `~T` (tilde) | Includes all types whose underlying type is T |
| Type inference | Automatically infer type parameters from arguments |
| Zero value | Obtain the zero value of a generic type with `var zero T` |
| `slices` / `maps` | Generic utilities in the standard library |
| Result / Optional | Generic types for error handling and nil safety |
| GCShape stenciling | Type-specialization strategy at compile time |

---

## Recommended Next Guides

- **03-tools/02-profiling.md** — Profiling: pprof, trace
- **03-tools/04-best-practices.md** — Best practices: Effective Go
- **03-tools/00-cli-development.md** — CLI development: cobra, flag, promptui

---

## References

1. **Go Official — Type Parameters Proposal** https://go.googlesource.com/proposal/+/refs/heads/master/design/43651-type-parameters.md
2. **Go Official — Tutorial: Getting started with generics** https://go.dev/doc/tutorial/generics
3. **Go Blog — An Introduction To Generics** https://go.dev/blog/intro-generics
4. **Go Standard Library — slices package** https://pkg.go.dev/slices
5. **Go Standard Library — maps package** https://pkg.go.dev/maps
6. **Go Standard Library — cmp package** https://pkg.go.dev/cmp
