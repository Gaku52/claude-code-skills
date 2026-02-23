# システム言語比較（C, C++, Rust, Go, Zig）

> システム言語は「ハードウェアに近い制御」と「高いパフォーマンス」を提供する。OS、ドライバ、ゲームエンジン、インフラツールの基盤。

## この章で学ぶこと

- [ ] 主要システム言語の特徴と適用領域を把握する
- [ ] メモリ管理戦略の違いを理解する
- [ ] 安全性とパフォーマンスのトレードオフを判断できる
- [ ] 各言語のビルドシステムとツールチェーンを理解する
- [ ] プロジェクト要件に応じた言語選択ができる
- [ ] 各言語のエラーハンドリング戦略を比較できる

---

## 1. 比較表

```
┌──────────────┬────────┬────────┬────────┬────────┬────────┐
│              │ C      │ C++    │ Rust   │ Go     │ Zig    │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ 登場年        │ 1972   │ 1985   │ 2015   │ 2012   │ 2016   │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ 設計者        │ D.Ritchie│Stroustrup│ Hoare+│ Pike+  │ A.Kelley│
│              │        │        │ Mozilla│ Google │        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ メモリ管理    │ 手動   │ 手動   │ 所有権 │ GC     │ 手動   │
│              │ malloc │ RAII   │ Borrow │ 並行GC │ alloc  │
│              │ free   │ smart  │ checker│ 低遅延 │ comptime│
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ 安全性        │ 低い   │ 中程度 │ 高い   │ 高い   │ 中程度 │
│              │ UB多   │ UB有   │ UB無(safe)│メモリ安全│ UB有  │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ パフォーマンス│ 最速   │ 最速   │ 最速   │ 高速   │ 最速   │
│              │        │        │        │ GC pause│        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ コンパイル速度│ 速い   │ 遅い   │ 遅い   │ 非常速 │ 速い   │
│              │        │ ヘッダ │ 借用検査│        │ 増分   │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ 学習コスト    │ 中程度 │ 高い   │ 高い   │ 低い   │ 中程度 │
│              │        │ 膨大仕様│ 借用   │ 25KW  │        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ 抽象化       │ 最低限 │ 豊富   │ 豊富   │ 最低限 │ 最低限 │
│ レベル       │ 関数   │ テンプレ│ トレイト│ インタフェ│ comptime│
│              │        │ OOP    │ ジェネリ│ ース   │        │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ エラー処理   │ 戻り値 │ 例外   │ Result │ error  │ error  │
│              │ errno  │ RAII   │ Option │ multi  │ union  │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ 並行処理     │ pthread│ thread │ Send/  │ goroutine│ async  │
│              │ fork   │ async  │ Sync   │ channel│ evented│
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ 主な用途      │ OS     │ ゲーム │ インフラ│ クラウド│ 組み込み│
│              │ 組み込み│ ブラウザ│ CLI    │ ツール │ システム│
│              │ カーネル│ DB     │ Wasm   │ マイクロ│ ゲーム │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ 標準ライブラリ│ 最小限 │ 大きい │ 中程度 │ 充実   │ 最小限 │
│              │ libc   │ STL    │ std    │ net等  │ std    │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ ビルドシステム│ Make   │ CMake  │ Cargo  │ go build│ zig build│
│              │ Meson  │ Bazel  │        │        │        │
└──────────────┴────────┴────────┴────────┴────────┴────────┘
```

---

## 2. メモリ管理モデルの詳細比較

### 2.1 C — 手動メモリ管理

```c
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// C: 手動メモリ管理 — malloc/free の対で管理
typedef struct {
    char* name;
    int age;
    char** tags;
    int tag_count;
} User;

User* user_create(const char* name, int age) {
    User* user = (User*)malloc(sizeof(User));
    if (!user) return NULL;  // メモリ確保失敗

    user->name = strdup(name);  // 文字列のコピーを確保
    if (!user->name) {
        free(user);
        return NULL;
    }

    user->age = age;
    user->tags = NULL;
    user->tag_count = 0;
    return user;
}

int user_add_tag(User* user, const char* tag) {
    // realloc で配列を拡張
    char** new_tags = (char**)realloc(
        user->tags,
        sizeof(char*) * (user->tag_count + 1)
    );
    if (!new_tags) return -1;  // メモリ確保失敗

    user->tags = new_tags;
    user->tags[user->tag_count] = strdup(tag);
    if (!user->tags[user->tag_count]) return -1;

    user->tag_count++;
    return 0;
}

void user_destroy(User* user) {
    if (!user) return;

    free(user->name);
    for (int i = 0; i < user->tag_count; i++) {
        free(user->tags[i]);
    }
    free(user->tags);
    free(user);
}

// 使用例
void example(void) {
    User* alice = user_create("Alice", 30);
    if (!alice) {
        fprintf(stderr, "メモリ確保失敗\n");
        return;
    }

    user_add_tag(alice, "admin");
    user_add_tag(alice, "developer");

    printf("Name: %s, Age: %d\n", alice->name, alice->age);

    user_destroy(alice);  // 必ず解放する（忘れるとメモリリーク）
    // alice = NULL;      // ダングリングポインタ防止（推奨）
}

// 典型的なCのバグ: Use After Free
void dangerous_example(void) {
    char* name = strdup("Alice");
    free(name);
    // printf("%s\n", name);  // 未定義動作！ 解放後のメモリにアクセス
}

// 典型的なCのバグ: バッファオーバーフロー
void buffer_overflow(void) {
    char buf[10];
    // strcpy(buf, "This is a very long string");  // 危険！
    strncpy(buf, "This is a very long string", sizeof(buf) - 1);  // 安全
    buf[sizeof(buf) - 1] = '\0';
}

// 典型的なCのバグ: ダブルフリー
void double_free_example(void) {
    char* ptr = malloc(100);
    free(ptr);
    // free(ptr);  // 未定義動作！ 二重解放
}
```

### 2.2 C++ — RAII とスマートポインタ

```cpp
#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <optional>

// C++: RAII (Resource Acquisition Is Initialization)
// コンストラクタで獲得、デストラクタで解放

class User {
public:
    User(std::string name, int age)
        : name_(std::move(name)), age_(age) {}

    // デストラクタ — スコープを抜けると自動呼び出し
    ~User() {
        std::cout << "User " << name_ << " destroyed" << std::endl;
    }

    void add_tag(std::string tag) {
        tags_.push_back(std::move(tag));
    }

    const std::string& name() const { return name_; }
    int age() const { return age_; }
    const std::vector<std::string>& tags() const { return tags_; }

private:
    std::string name_;      // std::string が内部メモリを管理
    int age_;
    std::vector<std::string> tags_;  // vector が配列メモリを管理
};

// スマートポインタの使い分け
void smart_pointer_example() {
    // unique_ptr: 唯一の所有権（最も一般的）
    auto alice = std::make_unique<User>("Alice", 30);
    alice->add_tag("admin");

    // 所有権の移動（ムーブ）
    auto owner = std::move(alice);
    // alice はもう使えない（nullptr）

    // shared_ptr: 共有所有権（参照カウント）
    auto bob = std::make_shared<User>("Bob", 25);
    {
        auto bob_ref = bob;  // 参照カウント +1
        std::cout << "ref count: " << bob.use_count() << std::endl;  // 2
    }
    // bob_ref がスコープを抜けて参照カウント -1
    std::cout << "ref count: " << bob.use_count() << std::endl;  // 1

    // weak_ptr: 循環参照の防止
    std::weak_ptr<User> weak = bob;
    if (auto locked = weak.lock()) {
        std::cout << "User still alive: " << locked->name() << std::endl;
    }
}

// ムーブセマンティクス（C++11以降）
class LargeBuffer {
    std::vector<uint8_t> data_;

public:
    explicit LargeBuffer(size_t size) : data_(size, 0) {}

    // ムーブコンストラクタ — データの所有権を移動（コピーなし）
    LargeBuffer(LargeBuffer&& other) noexcept
        : data_(std::move(other.data_)) {}

    // ムーブ代入演算子
    LargeBuffer& operator=(LargeBuffer&& other) noexcept {
        data_ = std::move(other.data_);
        return *this;
    }

    // コピーを禁止（大きいデータの意図しないコピーを防ぐ）
    LargeBuffer(const LargeBuffer&) = delete;
    LargeBuffer& operator=(const LargeBuffer&) = delete;

    size_t size() const { return data_.size(); }
};

// std::optional で null を型安全に扱う
std::optional<User> find_user(const std::string& name) {
    if (name == "Alice") {
        return User("Alice", 30);
    }
    return std::nullopt;
}

void optional_example() {
    auto user = find_user("Alice");
    if (user.has_value()) {
        std::cout << user->name() << std::endl;
    }

    // value_or でデフォルト値
    auto name = find_user("Bob")
        .transform([](const User& u) { return u.name(); })
        .value_or("Unknown");
}

// コンセプト（C++20）— テンプレートの型制約
template<typename T>
concept Printable = requires(T t) {
    { std::cout << t } -> std::same_as<std::ostream&>;
};

template<Printable T>
void print(const T& value) {
    std::cout << value << std::endl;
}

// Ranges（C++20）— 関数型スタイルのデータ処理
#include <ranges>
#include <algorithm>

void ranges_example() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto result = numbers
        | std::views::filter([](int n) { return n % 2 == 0; })
        | std::views::transform([](int n) { return n * n; })
        | std::views::take(3);

    for (int n : result) {
        std::cout << n << " ";  // 4 16 36
    }
}
```

### 2.3 Rust — 所有権と借用チェッカー

```rust
use std::collections::HashMap;

// Rust: 所有権システム — コンパイル時にメモリ安全性を保証
// 3つのルール:
// 1. 各値は1つの所有者を持つ
// 2. 所有者がスコープを離れると値は破棄される
// 3. &T（不変借用）は複数可、&mut T（可変借用）は1つだけ

struct User {
    name: String,
    age: u32,
    tags: Vec<String>,
}

impl User {
    fn new(name: impl Into<String>, age: u32) -> Self {
        User {
            name: name.into(),
            age,
            tags: Vec::new(),
        }
    }

    fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
    }

    // &self: 不変借用（読み取りのみ）
    fn display(&self) -> String {
        format!("{} (age: {}, tags: {:?})", self.name, self.age, self.tags)
    }

    // self: 所有権を消費（呼び出し後は使えない）
    fn into_name(self) -> String {
        self.name  // 所有権が移動
    }
}

fn ownership_example() {
    let mut alice = User::new("Alice", 30);
    alice.add_tag("admin");
    alice.add_tag("developer");

    // 不変借用（同時に複数可能）
    let display1 = alice.display();
    let display2 = alice.display();
    println!("{}", display1);
    println!("{}", display2);

    // 所有権の移動
    let name = alice.into_name();
    println!("Name: {}", name);
    // println!("{}", alice.display());  // コンパイルエラー！ aliceはもう使えない
}

// ライフタイム — 参照の有効期間をコンパイラに伝える
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

struct Config<'a> {
    name: &'a str,
    values: Vec<&'a str>,
}

impl<'a> Config<'a> {
    fn new(name: &'a str) -> Self {
        Config { name, values: Vec::new() }
    }

    fn add_value(&mut self, value: &'a str) {
        self.values.push(value);
    }
}

// Result/Option によるエラーハンドリング
use std::fs;
use std::io;

fn read_config(path: &str) -> Result<HashMap<String, String>, io::Error> {
    let content = fs::read_to_string(path)?;  // ? 演算子でエラー伝播
    let mut config = HashMap::new();

    for line in content.lines() {
        if let Some((key, value)) = line.split_once('=') {
            config.insert(key.trim().to_string(), value.trim().to_string());
        }
    }

    Ok(config)
}

// パターンマッチとenum（代数的データ型）
#[derive(Debug)]
enum Shape {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Triangle { base: f64, height: f64 },
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle { radius } => std::f64::consts::PI * radius * radius,
            Shape::Rectangle { width, height } => width * height,
            Shape::Triangle { base, height } => base * height / 2.0,
        }
    }
}

// トレイト — インターフェースとジェネリクスの基盤
trait Summary {
    fn summarize(&self) -> String;

    // デフォルト実装
    fn summarize_short(&self) -> String {
        format!("{}...", &self.summarize()[..20])
    }
}

impl Summary for User {
    fn summarize(&self) -> String {
        format!("{} ({}歳)", self.name, self.age)
    }
}

// トレイト境界によるジェネリクス
fn print_summary(item: &impl Summary) {
    println!("{}", item.summarize());
}

// where句を使った複雑な制約
fn process_items<T>(items: &[T]) -> Vec<String>
where
    T: Summary + std::fmt::Debug,
{
    items.iter()
        .map(|item| item.summarize())
        .collect()
}

// 並行処理 — Send/Syncトレイトでコンパイル時安全性保証
use std::sync::{Arc, Mutex};
use std::thread;

fn concurrent_counter() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", *counter.lock().unwrap());  // 10
}

// チャネルによるメッセージパッシング
use std::sync::mpsc;

fn channel_example() {
    let (tx, rx) = mpsc::channel();

    // 送信側
    for i in 0..5 {
        let tx = tx.clone();
        thread::spawn(move || {
            tx.send(format!("Message {}", i)).unwrap();
        });
    }
    drop(tx);  // 元の送信者を閉じる

    // 受信側
    for received in rx {
        println!("Got: {}", received);
    }
}

// async/await（非同期処理）
use tokio;

#[tokio::main]
async fn main() {
    let urls = vec![
        "https://example.com",
        "https://example.org",
    ];

    let mut handles = vec![];
    for url in urls {
        handles.push(tokio::spawn(async move {
            let resp = reqwest::get(url).await.unwrap();
            (url.to_string(), resp.status().as_u16())
        }));
    }

    for handle in handles {
        let (url, status) = handle.await.unwrap();
        println!("{}: {}", url, status);
    }
}
```

### 2.4 Go — ガベージコレクタ + 軽量並行処理

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

// Go: GCによるメモリ管理 + goroutine による軽量並行処理
// 設計哲学: シンプルさ、高速コンパイル、並行処理

// 構造体（Go にはクラスがない）
type User struct {
    Name string
    Age  int
    Tags []string
}

// メソッド（レシーバ付き関数）
func (u *User) AddTag(tag string) {
    u.Tags = append(u.Tags, tag)
}

func (u User) Display() string {
    return fmt.Sprintf("%s (age: %d, tags: %v)", u.Name, u.Age, u.Tags)
}

// インターフェース — 暗黙的実装（implements 宣言不要）
type Summarizer interface {
    Summarize() string
}

func (u User) Summarize() string {
    return fmt.Sprintf("%s (%d歳)", u.Name, u.Age)
}

// User は Summarizer を暗黙的に実装
func PrintSummary(s Summarizer) {
    fmt.Println(s.Summarize())
}

// エラーハンドリング — 明示的な error 値の返却
type AppError struct {
    Code    int
    Message string
}

func (e *AppError) Error() string {
    return fmt.Sprintf("[%d] %s", e.Code, e.Message)
}

func FindUser(name string) (*User, error) {
    if name == "" {
        return nil, &AppError{Code: 400, Message: "name is required"}
    }
    if name == "Alice" {
        return &User{Name: "Alice", Age: 30}, nil
    }
    return nil, &AppError{Code: 404, Message: "user not found"}
}

// errors.Is / errors.As によるエラー判定（Go 1.13+）
func example() {
    user, err := FindUser("Bob")
    if err != nil {
        var appErr *AppError
        if errors.As(err, &appErr) {
            log.Printf("App error: code=%d, msg=%s", appErr.Code, appErr.Message)
        } else {
            log.Printf("Unknown error: %v", err)
        }
        return
    }
    fmt.Println(user.Display())
}

// Goroutine + Channel — Go の並行処理の核
func goroutineExample() {
    ch := make(chan string, 10)  // バッファ付きチャネル

    urls := []string{
        "https://example.com",
        "https://example.org",
        "https://example.net",
    }

    for _, url := range urls {
        go func(u string) {
            // HTTPリクエストを並行実行
            result := fmt.Sprintf("Fetched: %s", u)
            ch <- result
        }(url)
    }

    for range urls {
        fmt.Println(<-ch)
    }
}

// select文 — 複数チャネルの待機
func selectExample() {
    ch1 := make(chan string)
    ch2 := make(chan string)

    go func() {
        time.Sleep(100 * time.Millisecond)
        ch1 <- "from ch1"
    }()

    go func() {
        time.Sleep(200 * time.Millisecond)
        ch2 <- "from ch2"
    }()

    for i := 0; i < 2; i++ {
        select {
        case msg := <-ch1:
            fmt.Println(msg)
        case msg := <-ch2:
            fmt.Println(msg)
        case <-time.After(1 * time.Second):
            fmt.Println("timeout")
        }
    }
}

// Context によるキャンセレーション
func contextExample(ctx context.Context) error {
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    ch := make(chan string, 1)
    go func() {
        // 長時間処理
        time.Sleep(3 * time.Second)
        ch <- "done"
    }()

    select {
    case result := <-ch:
        fmt.Println(result)
        return nil
    case <-ctx.Done():
        return ctx.Err()  // context.DeadlineExceeded or context.Canceled
    }
}

// WaitGroup で複数 goroutine の完了を待機
func waitGroupExample() {
    var wg sync.WaitGroup
    results := make([]string, 5)

    for i := 0; i < 5; i++ {
        wg.Add(1)
        go func(idx int) {
            defer wg.Done()
            results[idx] = fmt.Sprintf("result-%d", idx)
        }(i)
    }

    wg.Wait()
    fmt.Println(results)
}

// ジェネリクス（Go 1.18+）
func Map[T any, U any](slice []T, f func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = f(v)
    }
    return result
}

func Filter[T any](slice []T, predicate func(T) bool) []T {
    var result []T
    for _, v := range slice {
        if predicate(v) {
            result = append(result, v)
        }
    }
    return result
}

// 型制約
type Number interface {
    ~int | ~int64 | ~float64
}

func Sum[T Number](numbers []T) T {
    var total T
    for _, n := range numbers {
        total += n
    }
    return total
}

// 使用例
func genericsExample() {
    names := []string{"Alice", "Bob", "Carol"}
    upper := Map(names, strings.ToUpper)
    // → ["ALICE", "BOB", "CAROL"]

    numbers := []int{1, 2, 3, 4, 5, 6}
    evens := Filter(numbers, func(n int) bool { return n%2 == 0 })
    // → [2, 4, 6]

    total := Sum([]int{1, 2, 3, 4, 5})
    // → 15
}
```

### 2.5 Zig — コンパイル時計算と明示性

```zig
const std = @import("std");
const Allocator = std.mem.Allocator;

// Zig: 手動メモリ管理 + comptime（コンパイル時計算）
// 設計哲学: 隠れた制御フローなし、隠れたメモリ割り当てなし

// アロケータを明示的に渡す（隠れた割り当てなし）
const User = struct {
    name: []const u8,
    age: u32,
    tags: std.ArrayList([]const u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8, age: u32) User {
        return User{
            .name = name,
            .age = age,
            .tags = std.ArrayList([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *User) void {
        self.tags.deinit();
    }

    pub fn addTag(self: *User, tag: []const u8) !void {
        try self.tags.append(tag);
    }

    pub fn display(self: User) void {
        std.debug.print("User: {s} (age: {})\n", .{ self.name, self.age });
    }
};

// エラーハンドリング — error union 型
const FileError = error{
    FileNotFound,
    PermissionDenied,
    OutOfMemory,
};

fn readConfig(path: []const u8) FileError![]const u8 {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return FileError.FileNotFound,
        error.AccessDenied => return FileError.PermissionDenied,
        else => return FileError.FileNotFound,
    };
    defer file.close();

    // ファイル読み込み
    return file.readToEndAlloc(std.heap.page_allocator, 1024 * 1024) catch {
        return FileError.OutOfMemory;
    };
}

// comptime — コンパイル時計算（Zigの最大の特徴）
fn fibonacci(comptime n: u32) u32 {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// コンパイル時に計算されるので実行時コストゼロ
const fib_10 = fibonacci(10);  // コンパイル時に 55 と計算される

// comptime による型生成
fn Matrix(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    return struct {
        data: [rows][cols]T,

        const Self = @This();

        pub fn init() Self {
            return Self{
                .data = [_][cols]T{[_]T{0} ** cols} ** rows,
            };
        }

        pub fn get(self: Self, row: usize, col: usize) T {
            return self.data[row][col];
        }

        pub fn set(self: *Self, row: usize, col: usize, value: T) void {
            self.data[row][col] = value;
        }
    };
}

// 使用例
const Mat3x3 = Matrix(f64, 3, 3);

pub fn main() void {
    var mat = Mat3x3.init();
    mat.set(0, 0, 1.0);
    mat.set(1, 1, 1.0);
    mat.set(2, 2, 1.0);
    // 3x3 単位行列
}

// defer / errdefer — リソース管理
fn processFile(path: []const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();  // 関数終了時に必ず実行

    const buffer = try std.heap.page_allocator.alloc(u8, 4096);
    errdefer std.heap.page_allocator.free(buffer);  // エラー時のみ実行

    // ファイル処理...
}

// テスト（言語組み込み）
test "fibonacci" {
    try std.testing.expectEqual(fibonacci(0), 0);
    try std.testing.expectEqual(fibonacci(1), 1);
    try std.testing.expectEqual(fibonacci(10), 55);
}

test "user creation" {
    var user = User.init(std.testing.allocator, "Alice", 30);
    defer user.deinit();

    try user.addTag("admin");
    try std.testing.expectEqual(user.tags.items.len, 1);
}
```

---

## 3. エラーハンドリング戦略の比較

```
┌──────────┬────────────────────┬──────────────────────────────────┐
│ 言語      │ 主なメカニズム      │ 特徴                              │
├──────────┼────────────────────┼──────────────────────────────────┤
│ C        │ 戻り値 + errno     │ エラーチェック漏れが起きやすい      │
│          │                    │ 情報量が少ない                     │
├──────────┼────────────────────┼──────────────────────────────────┤
│ C++      │ 例外 + RAII        │ スタックアンワインドで自動クリーンアップ│
│          │ std::expected(C++23)│ noexcept で例外なし関数を明示      │
├──────────┼────────────────────┼──────────────────────────────────┤
│ Rust     │ Result<T,E> +      │ ? 演算子で簡潔なエラー伝播         │
│          │ Option<T>          │ panic! は回復不能エラーのみ         │
├──────────┼────────────────────┼──────────────────────────────────┤
│ Go       │ error インターフェース│ (value, error) タプルで返却       │
│          │ errors.Is/As       │ 明示的だが冗長になりがち            │
├──────────┼────────────────────┼──────────────────────────────────┤
│ Zig      │ error union        │ try / catch で簡潔に記述           │
│          │ errdefer           │ エラー時のクリーンアップが容易       │
└──────────┴────────────────────┴──────────────────────────────────┘
```

### 同じエラーハンドリングパターンの比較

```c
// C: 戻り値でエラーを伝える
#include <stdio.h>
#include <errno.h>

int read_int_from_file(const char* path, int* result) {
    FILE* f = fopen(path, "r");
    if (!f) {
        return -1;  // エラー（errno にエラーコード）
    }

    if (fscanf(f, "%d", result) != 1) {
        fclose(f);
        return -2;  // パースエラー
    }

    fclose(f);
    return 0;  // 成功
}

// 呼び出し側
void caller(void) {
    int value;
    int ret = read_int_from_file("config.txt", &value);
    if (ret == -1) {
        fprintf(stderr, "ファイルが開けません: %s\n", strerror(errno));
    } else if (ret == -2) {
        fprintf(stderr, "パースエラー\n");
    } else {
        printf("値: %d\n", value);
    }
}
```

```cpp
// C++: 例外 + std::expected (C++23)
#include <expected>
#include <fstream>
#include <string>

enum class ReadError {
    FileNotFound,
    ParseError,
};

// C++23: std::expected
std::expected<int, ReadError> read_int_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return std::unexpected(ReadError::FileNotFound);
    }

    int value;
    if (!(file >> value)) {
        return std::unexpected(ReadError::ParseError);
    }

    return value;
}

// 呼び出し側
void caller() {
    auto result = read_int_from_file("config.txt");
    if (result.has_value()) {
        std::cout << "値: " << result.value() << std::endl;
    } else {
        switch (result.error()) {
            case ReadError::FileNotFound:
                std::cerr << "ファイルが見つかりません" << std::endl;
                break;
            case ReadError::ParseError:
                std::cerr << "パースエラー" << std::endl;
                break;
        }
    }

    // transform でチェーン
    auto doubled = read_int_from_file("config.txt")
        .transform([](int v) { return v * 2; });
}
```

```rust
// Rust: Result + ? 演算子
use std::fs;
use std::num::ParseIntError;
use thiserror::Error;

#[derive(Error, Debug)]
enum ReadError {
    #[error("ファイルが開けません: {0}")]
    FileNotFound(#[from] std::io::Error),
    #[error("パースエラー: {0}")]
    ParseError(#[from] ParseIntError),
}

fn read_int_from_file(path: &str) -> Result<i32, ReadError> {
    let content = fs::read_to_string(path)?;  // io::Error → ReadError
    let value: i32 = content.trim().parse()?;  // ParseIntError → ReadError
    Ok(value)
}

// 呼び出し側
fn caller() {
    match read_int_from_file("config.txt") {
        Ok(value) => println!("値: {}", value),
        Err(ReadError::FileNotFound(e)) => eprintln!("ファイルエラー: {}", e),
        Err(ReadError::ParseError(e)) => eprintln!("パースエラー: {}", e),
    }

    // map / and_then でチェーン
    let doubled = read_int_from_file("config.txt")
        .map(|v| v * 2);
}
```

```go
// Go: error インターフェース
package main

import (
    "errors"
    "fmt"
    "os"
    "strconv"
    "strings"
)

type ReadError struct {
    Kind    string
    Message string
    Err     error
}

func (e *ReadError) Error() string {
    return fmt.Sprintf("%s: %s", e.Kind, e.Message)
}

func (e *ReadError) Unwrap() error {
    return e.Err
}

func readIntFromFile(path string) (int, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return 0, &ReadError{
            Kind:    "file_not_found",
            Message: fmt.Sprintf("ファイルが開けません: %s", path),
            Err:     err,
        }
    }

    value, err := strconv.Atoi(strings.TrimSpace(string(data)))
    if err != nil {
        return 0, &ReadError{
            Kind:    "parse_error",
            Message: "パースエラー",
            Err:     err,
        }
    }

    return value, nil
}

// 呼び出し側
func caller() {
    value, err := readIntFromFile("config.txt")
    if err != nil {
        var readErr *ReadError
        if errors.As(err, &readErr) {
            fmt.Printf("エラー種別: %s, メッセージ: %s\n", readErr.Kind, readErr.Message)
        } else {
            fmt.Printf("不明なエラー: %v\n", err)
        }
        return
    }
    fmt.Printf("値: %d\n", value)
}
```

---

## 4. ビルドシステムとツールチェーン

```
┌──────────┬──────────────┬─────────────────────────────────────┐
│ 言語      │ ビルドツール  │ 特徴                                 │
├──────────┼──────────────┼─────────────────────────────────────┤
│ C        │ Make, CMake  │ 歴史的に最も使われている               │
│          │ Meson, Ninja │ ビルドスクリプトの記述が複雑             │
│          │              │ プラットフォーム依存が大きい             │
├──────────┼──────────────┼─────────────────────────────────────┤
│ C++      │ CMake        │ 事実上の標準だが設定が難解              │
│          │ Bazel        │ 大規模プロジェクト向け（Google製）      │
│          │ Conan,vcpkg  │ パッケージマネージャ                    │
├──────────┼──────────────┼─────────────────────────────────────┤
│ Rust     │ Cargo        │ ビルド+パッケージ+テスト+ベンチ統合     │
│          │              │ toml設定、cargo.lock で再現性確保       │
│          │              │ 最も優れたツールチェーン体験             │
├──────────┼──────────────┼─────────────────────────────────────┤
│ Go       │ go build     │ go mod でモジュール管理                │
│          │              │ 外部ツール不要、go コマンドで完結       │
│          │              │ クロスコンパイルが非常に簡単             │
├──────────┼──────────────┼─────────────────────────────────────┤
│ Zig      │ zig build    │ build.zig で宣言的ビルド               │
│          │              │ C/C++ のクロスコンパイラとしても使用可能 │
│          │              │ libc を同梱                           │
└──────────┴──────────────┴─────────────────────────────────────┘
```

```toml
# Rust: Cargo.toml — 依存管理の模範例
[package]
name = "my-cli-tool"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
anyhow = "1"
tracing = "0.1"

[dev-dependencies]
assert_cmd = "2"
predicates = "3"
tempfile = "3"

[profile.release]
lto = true        # リンク時最適化
strip = true      # デバッグ情報除去
codegen-units = 1 # 最大最適化
```

```go
// Go: go.mod — シンプルなモジュール管理
// go.mod
module github.com/user/myapp

go 1.22

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/jackc/pgx/v5 v5.5.3
    go.uber.org/zap v1.27.0
)

// クロスコンパイル（コマンド1つ）
// GOOS=linux GOARCH=amd64 go build -o myapp-linux
// GOOS=darwin GOARCH=arm64 go build -o myapp-mac
// GOOS=windows GOARCH=amd64 go build -o myapp.exe
```

---

## 5. 適用領域の詳細

### 5.1 領域別最適言語マッピング

```
OS カーネル:
  C:    Linux kernel（3,000万行超）
  Rust: Linux kernel 新モジュール（6.1から公式サポート）
  C++:  Windows kernel の一部

ブラウザエンジン:
  C++:  Chromium (Blink), WebKit
  Rust: Firefox (Servo コンポーネント → Stylo CSS エンジン)

データベース:
  C:    SQLite, PostgreSQL
  C++:  MySQL, MongoDB, RocksDB, ClickHouse
  Rust: SurrealDB, TiKV, Neon (PostgreSQL互換)
  Go:   CockroachDB, TiDB, InfluxDB

ゲームエンジン:
  C++:  Unreal Engine, Unity (C# + C++内部)
  Rust: Bevy (新興だが成長中)
  Zig:  一部のインディーゲームエンジン

クラウドインフラ:
  Go:   Docker, Kubernetes, Terraform, Prometheus, Grafana, etcd
  Rust: Firecracker (AWS Lambda基盤), Bottlerocket, Linkerd2-proxy

CLI ツール:
  Rust: ripgrep, bat, fd, exa/eza, starship, zoxide, delta
  Go:   gh (GitHub CLI), lazygit, fzf, Hugo, k9s

暗号・セキュリティ:
  Rust: BoringSSL一部, rustls
  C:    OpenSSL, libsodium
  Go:   crypto/tls (標準ライブラリ)

組み込み/IoT:
  C:    圧倒的シェア（FreeRTOS, Zephyr の一部）
  Rust: Embassy (async embedded), RTIC
  Zig:  組み込みLinux, マイクロコントローラ

WebAssembly:
  Rust: Yew, Leptos, wasm-bindgen
  C/C++: Emscripten
  Go:   TinyGo (最適化版)
  Zig:  ネイティブWasm出力
```

### 5.2 パフォーマンスベンチマーク

```
Benchmark: HTTP サーバー（requests/sec, 高いほど良い）
  C (epoll直接):  500,000+
  Rust (actix):   400,000+
  Go (net/http):  200,000+
  C++ (drogon):   350,000+
  Zig (zap):      450,000+

Benchmark: JSON パース（1GB ファイル）
  C (simdjson):   2.5 GB/s
  Rust (simd-json): 2.3 GB/s
  C++ (simdjson): 2.5 GB/s
  Go (encoding):  0.3 GB/s
  Go (sonic):     1.5 GB/s

Benchmark: コンパイル時間（中規模プロジェクト）
  Go:    2-5 秒
  C:     5-15 秒
  Zig:   5-15 秒
  Rust:  30-120 秒（増分は10-30秒）
  C++:   60-300 秒

Benchmark: バイナリサイズ（Hello World）
  C:     16 KB (static: 800 KB)
  Go:    1.8 MB (static by default)
  Rust:  300 KB (stripped)
  Zig:   5 KB (stripped)
  C++:   20 KB (dynamic)

※ 実際のパフォーマンスはワークロードに大きく依存する
※ ベンチマークは参考値として捉えること
```

---

## 6. メモリ安全性の議論

### 6.1 米国政府の勧告（2024年）

```
2024年2月、米国ホワイトハウスは「メモリ安全な言語への移行」を勧告:
- C/C++ のメモリ関連脆弱性がサイバー攻撃の主因
- CVE の約70%がメモリ安全性に起因
- Rust, Go, Java, C# 等のメモリ安全な言語を推奨

影響:
- Linux kernel への Rust 採用が加速
- Android の新規コードにおける Rust 比率が増加
- DARPA の TRACTOR プログラム（C → Rust 自動変換研究）
- NSA のサイバーセキュリティガイダンスで Rust を推奨
```

### 6.2 各言語のメモリ安全性メカニズム

```
C:
  ✗ バッファオーバーフロー
  ✗ Use After Free
  ✗ ダブルフリー
  ✗ Null ポインタ参照
  △ AddressSanitizer, Valgrind で動的検出

C++:
  △ スマートポインタで一部解決
  △ RAII でリソース漏れを防止
  ✗ 生ポインタの unsafe な操作は依然可能
  △ 静的解析ツール（Clang-Tidy, PVS-Studio）

Rust:
  ✓ 所有権 + 借用チェッカーでコンパイル時に保証
  ✓ Null なし（Option<T> で表現）
  ✓ データ競合なし（Send/Sync トレイト）
  △ unsafe ブロック内は保証なし（最小限に抑える）

Go:
  ✓ GC でメモリリーク/ダングリングポインタなし
  ✓ 境界チェックあり（ランタイム）
  △ Race detector（動的検出）
  ✗ Null ポインタ（nil）によるパニックは可能

Zig:
  △ 手動管理だがアロケータの明示で追跡が容易
  △ テストアロケータでリーク検出
  △ undefined behavior はあるが最小限
  ✓ 安全性チェックをビルドモードで制御可能
```

---

## 7. 並行処理モデルの詳細比較

```
┌──────────────┬──────────────────────────────────────────────┐
│ C            │ POSIX threads (pthread)                      │
│              │ - 低レベル、OS スレッドを直接操作             │
│              │ - mutex, condition variable, semaphore        │
│              │ - エラーが起きやすい（デッドロック、レース）    │
├──────────────┼──────────────────────────────────────────────┤
│ C++          │ std::thread + std::async (C++11)             │
│              │ std::jthread (C++20, 自動join)               │
│              │ - std::mutex, std::shared_mutex              │
│              │ - std::atomic で lock-free プログラミング     │
│              │ - coroutines (C++20)                         │
├──────────────┼──────────────────────────────────────────────┤
│ Rust         │ std::thread + crossbeam                      │
│              │ - Send/Sync トレイトでコンパイル時安全性      │
│              │ - Arc<Mutex<T>> で共有状態                    │
│              │ - mpsc チャネル、crossbeam チャネル           │
│              │ - async/await (tokio, async-std)              │
│              │ - Rayon（データ並列処理）                     │
├──────────────┼──────────────────────────────────────────────┤
│ Go           │ goroutine + channel                          │
│              │ - 軽量（初期 2KB スタック、動的拡張）          │
│              │ - 数百万の goroutine を実行可能               │
│              │ - select 文で複数チャネルを待機               │
│              │ - "Don't communicate by sharing memory;       │
│              │    share memory by communicating"             │
├──────────────┼──────────────────────────────────────────────┤
│ Zig          │ async/await（言語組み込み）                    │
│              │ - イベント駆動 I/O                            │
│              │ - std.Thread で OS スレッド                   │
│              │ - アロケータを通じた制御                       │
└──────────────┴──────────────────────────────────────────────┘
```

---

## 8. 実践的なプロジェクト構成例

### 8.1 Rust CLI プロジェクト

```
my-cli/
├── Cargo.toml
├── Cargo.lock
├── src/
│   ├── main.rs           # エントリポイント
│   ├── lib.rs            # ライブラリルート
│   ├── cli.rs            # clap によるCLI定義
│   ├── config.rs         # 設定管理
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── init.rs       # init サブコマンド
│   │   └── run.rs        # run サブコマンド
│   ├── core/
│   │   ├── mod.rs
│   │   ├── engine.rs     # コアロジック
│   │   └── types.rs      # 型定義
│   └── utils/
│       ├── mod.rs
│       └── fs.rs         # ファイルシステムユーティリティ
├── tests/
│   ├── integration_test.rs
│   └── fixtures/
├── benches/
│   └── benchmark.rs      # criterion によるベンチマーク
└── .github/
    └── workflows/
        └── ci.yml
```

### 8.2 Go Web API プロジェクト

```
myapp/
├── go.mod
├── go.sum
├── cmd/
│   └── server/
│       └── main.go        # エントリポイント
├── internal/              # 外部から import 不可
│   ├── handler/
│   │   ├── user.go        # ユーザーハンドラ
│   │   └── middleware.go  # ミドルウェア
│   ├── service/
│   │   └── user.go        # ビジネスロジック
│   ├── repository/
│   │   └── user.go        # データアクセス
│   ├── model/
│   │   └── user.go        # ドメインモデル
│   └── config/
│       └── config.go      # 設定
├── pkg/                   # 外部から import 可
│   └── response/
│       └── json.go
├── migrations/
│   └── 001_create_users.sql
├── Dockerfile
├── Makefile
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 9. 選択指針の詳細フローチャート

```
Q1: GC のポーズが許容できるか？
├── はい → Q2
└── いいえ → Q3

Q2: シンプルさと高速コンパイルが重要か？
├── はい → Go
│   適用: マイクロサービス, CLI, DevOpsツール
│   利点: 学習が容易、チーム全体で統一しやすい
└── いいえ → Q4

Q3: メモリ安全性が必要か？
├── はい → Rust
│   適用: インフラ, セキュリティ, Wasm, CLI
│   利点: コンパイル時安全性保証、ゼロコスト抽象化
└── いいえ → Q5

Q4: 関数型プログラミングやジェネリクスを多用するか？
├── はい → Rust
│   適用: ライブラリ, フレームワーク, 言語ツール
└── いいえ → Go
    適用: CRUD API, ネットワークサービス

Q5: 既存の C/C++ コードベースとの統合が必要か？
├── C++ コードベース → C++
│   適用: ゲーム, ブラウザ, 既存システムの拡張
├── C コードベース → C or Zig
│   Zig は C ヘッダを直接 import 可能
└── 新規プロジェクト → Rust or Zig
    Zig: C の代替、組み込み特化

Q6: ゲーム開発か？
├── はい → C++（Unreal）or Rust（Bevy）
└── いいえ → 上記フローに従う
```

### よくある誤解と補正

```
誤解: 「Go は遅い」
現実: GC あるが HTTP サーバーでは十分高速。多くの用途で C++ を選ぶ必要はない。

誤解: 「Rust は難しすぎて実用的でない」
現実: 学習曲線は急だが、慣れれば生産性は高い。
      借用チェッカーに慣れるまでの2-4週間が山場。

誤解: 「C は古くて使うべきでない」
現実: 組み込み、カーネル、特殊なシステムでは今でも最適な選択肢。
      ABI の安定性はCが最も優れている。

誤解: 「C++ は複雑すぎる」
現実: Modern C++ (C++17/20/23) は大幅に改善。
      ただし全機能を使う必要はない。プロジェクトで使用する機能を限定すべき。

誤解: 「Zig はまだ実験的」
現実: 本番利用が増加中。Uber の bazel-zig-cc、
      Bun (JavaScript ランタイム) が Zig で書かれている。
```

---

## 10. 学習リソースとロードマップ

```
C:
  入門: K&R "The C Programming Language"
  実践: "Expert C Programming"
  期間: 基本文法 2週間、ポインタ習得 1-2ヶ月

C++:
  入門: "A Tour of C++" (Stroustrup)
  実践: "Effective Modern C++" (Meyers)
  期間: 基本文法 1ヶ月、Modern C++ 習得 3-6ヶ月

Rust:
  入門: "The Rust Programming Language" (公式Book)
  実践: "Rust in Action", "Zero To Production"
  期間: 基本文法 2-3週間、借用チェッカー克服 1-2ヶ月

Go:
  入門: "The Go Programming Language" (Donovan & Kernighan)
  実践: "Let's Go" (Alex Edwards)
  期間: 基本文法 1-2週間、実務レベル 1-2ヶ月

Zig:
  入門: ziglearn.org, "Zig Guide"
  実践: zig.guide, std ライブラリのソースコード
  期間: 基本文法 2-3週間、comptime 習得 1-2ヶ月
```

---

## まとめ

| 言語 | 哲学 | 最適なユースケース | 2025年の状況 |
|------|------|-----------------|-------------|
| C | 最小限の抽象化 | OS, カーネル, 組み込み | 不動の地位。ABI の lingua franca |
| C++ | ゼロコスト抽象化 | ゲーム, ブラウザ, DB | C++23 で大幅改善。依然として巨大 |
| Rust | 安全性+パフォーマンス | インフラ, CLI, Wasm | 急成長。Linux kernel 採用で正統性確立 |
| Go | シンプルさ+並行処理 | クラウド, マイクロサービス | クラウドインフラの事実上の標準言語 |
| Zig | C のモダン代替 | 組み込み, システム | Bun で知名度上昇。C の後継候補 |

---

## 次に読むべきガイド
→ [[02-jvm-languages.md]] -- JVM言語比較

---

## 参考文献
1. Blandy, J., Orendorff, J. & Tindall, L. "Programming Rust." 2nd Ed, O'Reilly, 2021.
2. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, 2015.
3. Stroustrup, B. "A Tour of C++." 3rd Ed, Addison-Wesley, 2022.
4. Kernighan, B. & Ritchie, D. "The C Programming Language." 2nd Ed, Prentice Hall, 1988.
5. Klabnik, S. & Nichols, C. "The Rust Programming Language." No Starch Press, 2023.
6. "The White House: Back to the Building Blocks." Technical Report, 2024.
7. "Rust for Linux." rust-for-linux.com.
8. "State of Developer Ecosystem 2024." JetBrains.
9. Kelley, A. "The Zig Programming Language." ziglang.org.
10. "Benchmarks Game." benchmarksgame-team.pages.debian.net.
