# スマートポインタ -- 所有権と参照カウントによる柔軟なメモリ管理

> Box、Rc、Arc、RefCell、Mutex等のスマートポインタは、所有権システムの制約を安全に緩和し、共有所有権や内部可変性といった高度なパターンを実現する。

---

## この章で学ぶこと

1. **Box / Rc / Arc** -- ヒープ確保、参照カウント、スレッド安全な共有所有権を理解する
2. **RefCell / Cell** -- 内部可変性パターンで不変参照越しにデータを変更する方法を習得する
3. **Mutex / RwLock** -- スレッド間でデータを安全に共有する仕組みを学ぶ
4. **Cow / Pin** -- 遅延クローンとメモリ固定の用途と実装を理解する
5. **カスタムスマートポインタ** -- Deref / Drop トレイトを実装して独自のスマートポインタを設計する

---

## 1. スマートポインタの全体像

```
┌─────────────────────────────────────────────────────────────┐
│                  スマートポインタ一覧                         │
├─────────────┬────────────────┬──────────────────────────────┤
│ 型           │ 所有権         │ 主な用途                     │
├─────────────┼────────────────┼──────────────────────────────┤
│ Box<T>      │ 単一所有       │ ヒープ確保、再帰型           │
│ Rc<T>       │ 共有(単スレ)  │ グラフ、複数所有者            │
│ Arc<T>      │ 共有(マルチ)  │ スレッド間共有               │
│ Cell<T>     │ 内部可変(Copy)│ 不変参照越しの書き換え       │
│ RefCell<T>  │ 内部可変       │ 実行時借用チェック           │
│ Mutex<T>    │ 排他ロック     │ スレッド間の可変アクセス     │
│ RwLock<T>   │ 読み書きロック │ 多読み少書きのスレッド共有   │
│ Cow<'a, T>  │ 遅延クローン   │ 変更時のみクローン           │
│ Pin<P>      │ 固定           │ 自己参照型、async            │
│ Weak<T>     │ 弱参照         │ 循環参照の防止               │
└─────────────┴────────────────┴──────────────────────────────┘
```

### 1.1 スマートポインタとは何か

スマートポインタとは、ポインタのように振る舞いつつ、追加のメタデータや機能を持つデータ構造である。Rustでは `Deref` トレイトと `Drop` トレイトを実装することで、通常の参照のように使いながら、自動的なリソース管理を行える。

```rust
use std::ops::Deref;

// Deref トレイトの仕組み
// Box<T> は Deref<Target = T> を実装している
fn takes_str(s: &str) {
    println!("{}", s);
}

fn main() {
    let boxed_string = Box::new(String::from("hello"));

    // Box<String> → &String → &str (Deref coercion)
    takes_str(&boxed_string);

    // 明示的な Deref
    let s: &String = &*boxed_string;
    println!("{}", s);
}
```

### 1.2 Deref と Drop の重要性

```
┌──────────────────────────────────────────────────────┐
│  Deref トレイト                                       │
│  - スマートポインタを透過的に参照として使えるようにする│
│  - Deref coercion: &Box<T> → &T の自動変換            │
│  - DerefMut: 可変参照への自動変換                     │
│                                                      │
│  Drop トレイト                                       │
│  - スコープ終了時にリソースを自動解放する              │
│  - ファイルハンドル、ネットワーク接続、メモリの解放    │
│  - RAII パターンの実現                                │
└──────────────────────────────────────────────────────┘
```

```rust
use std::ops::{Deref, DerefMut};

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x: T) -> Self {
        MyBox(x)
    }
}

impl<T> Deref for MyBox<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for MyBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> Drop for MyBox<T> {
    fn drop(&mut self) {
        println!("MyBox がドロップされました");
    }
}

fn main() {
    let x = MyBox::new(42);
    println!("値: {}", *x); // Deref で i32 として参照

    let mut s = MyBox::new(String::from("hello"));
    s.push_str(" world"); // DerefMut で String として可変参照
    println!("{}", *s);
} // MyBox がドロップされる
```

---

## 2. Box<T>

### 例1: Box の基本と再帰型

```rust
// 再帰的なデータ構造(コンパイル時にサイズ不明)
#[derive(Debug)]
enum List {
    Cons(i32, Box<List>),
    Nil,
}

fn main() {
    // Box: ヒープにデータを確保
    let b = Box::new(5);
    println!("Box の中身: {}", b);

    // 再帰型の使用
    let list = List::Cons(1,
        Box::new(List::Cons(2,
            Box::new(List::Cons(3,
                Box::new(List::Nil))))));
    println!("{:?}", list);
}
```

### Box のメモリレイアウト

```
  スタック          ヒープ
  ┌──────────┐     ┌───────────────┐
  │ Box<i32> │     │               │
  │ ptr ─────────>│     42        │
  │          │     │               │
  └──────────┘     └───────────────┘
  8 bytes           4 bytes

  Box<T> はスタック上のポインタ1つ分のサイズ
  Drop 時にヒープメモリを自動解放
```

### 例2: Box によるトレイトオブジェクト

```rust
trait Animal {
    fn name(&self) -> &str;
    fn sound(&self) -> &str;
    fn info(&self) -> String {
        format!("{}は{}と鳴く", self.name(), self.sound())
    }
}

struct Dog {
    name: String,
}

impl Animal for Dog {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "ワン" }
}

struct Cat {
    name: String,
}

impl Animal for Cat {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "ニャー" }
}

// Box<dyn Trait> で動的ディスパッチ
fn create_animal(kind: &str, name: &str) -> Box<dyn Animal> {
    match kind {
        "dog" => Box::new(Dog { name: name.to_string() }),
        "cat" => Box::new(Cat { name: name.to_string() }),
        _ => panic!("不明な動物種"),
    }
}

fn main() {
    let animals: Vec<Box<dyn Animal>> = vec![
        create_animal("dog", "ポチ"),
        create_animal("cat", "タマ"),
        create_animal("dog", "ハチ"),
    ];

    for animal in &animals {
        println!("{}", animal.info());
    }
}
```

### 例3: Box による大きなデータのムーブ最適化

```rust
// 大きな構造体
struct LargeStruct {
    data: [u8; 1_000_000], // 1MB
}

// スタック上にあると、ムーブ時に1MBのコピーが発生
// Box で包むと、ポインタ(8bytes)のコピーだけで済む
fn create_large() -> Box<LargeStruct> {
    Box::new(LargeStruct {
        data: [0u8; 1_000_000],
    })
}

fn process_large(data: Box<LargeStruct>) {
    println!("データサイズ: {} bytes", data.data.len());
    // Box はスコープ終了時にヒープメモリを解放
}

fn main() {
    let large = create_large(); // ポインタのムーブのみ
    process_large(large);       // ポインタのムーブのみ

    // Box のサイズはポインタ1つ分
    println!("Box<LargeStruct> のサイズ: {} bytes",
        std::mem::size_of::<Box<LargeStruct>>());
    // 8 bytes (64bit環境)
}
```

---

## 3. Rc<T> と Arc<T>

### 例4: Rc による共有所有権

```rust
use std::rc::Rc;

#[derive(Debug)]
struct SharedData {
    value: String,
}

fn main() {
    let data = Rc::new(SharedData {
        value: "共有データ".to_string(),
    });
    println!("参照カウント: {}", Rc::strong_count(&data)); // 1

    let data2 = Rc::clone(&data); // 参照カウント増加(データはコピーしない)
    println!("参照カウント: {}", Rc::strong_count(&data)); // 2

    {
        let data3 = Rc::clone(&data);
        println!("参照カウント: {}", Rc::strong_count(&data)); // 3
    } // data3 が drop → 参照カウント減少

    println!("参照カウント: {}", Rc::strong_count(&data)); // 2
    println!("値: {}", data.value);
}
```

### 例5: Arc によるスレッド間共有

```rust
use std::sync::Arc;
use std::thread;

fn main() {
    let data = Arc::new(vec![1, 2, 3, 4, 5]);
    let mut handles = vec![];

    for i in 0..3 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let sum: i32 = data.iter().sum();
            println!("スレッド{}: 合計={}", i, sum);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
```

### Rc / Arc の仕組み

```
  Rc::clone は参照カウントを増やすだけ
  データ自体はコピーしない

  data ──┐
         │     ┌────────────────────────┐
  data2 ─┼────>│ strong_count: 3        │
         │     │ weak_count: 0          │
  data3 ─┘     │ ┌────────────────────┐ │
               │ │ value: SharedData  │ │
               │ └────────────────────┘ │
               └────────────────────────┘

  全ての Rc が drop されたとき(count == 0)にデータを解放
  Rc: 単一スレッド用(Send を実装しない)
  Arc: マルチスレッド用(アトミック操作で参照カウント管理)
```

### 例6: Weak<T> による循環参照の防止

```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;

#[derive(Debug)]
struct Node {
    value: i32,
    parent: RefCell<Weak<Node>>,
    children: RefCell<Vec<Rc<Node>>>,
}

impl Node {
    fn new(value: i32) -> Rc<Node> {
        Rc::new(Node {
            value,
            parent: RefCell::new(Weak::new()),
            children: RefCell::new(vec![]),
        })
    }

    fn add_child(parent: &Rc<Node>, child: &Rc<Node>) {
        // 子ノードに親への弱い参照を設定
        *child.parent.borrow_mut() = Rc::downgrade(parent);
        // 親ノードに子への強い参照を追加
        parent.children.borrow_mut().push(Rc::clone(child));
    }
}

fn main() {
    let root = Node::new(1);
    let child1 = Node::new(2);
    let child2 = Node::new(3);

    Node::add_child(&root, &child1);
    Node::add_child(&root, &child2);

    // 参照カウントの確認
    println!("root の強い参照: {}", Rc::strong_count(&root));
    println!("root の弱い参照: {}", Rc::weak_count(&root));
    println!("child1 の強い参照: {}", Rc::strong_count(&child1));

    // 弱い参照から強い参照への昇格
    if let Some(parent) = child1.parent.borrow().upgrade() {
        println!("child1 の親の値: {}", parent.value);
    }

    // root を drop した場合、弱い参照は無効になる
    drop(root);
    println!("root drop 後の child1 の親: {:?}",
        child1.parent.borrow().upgrade()); // None
}
```

### 例7: Rc を使ったグラフ構造

```rust
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

type NodeRef = Rc<RefCell<GraphNode>>;

#[derive(Debug)]
struct GraphNode {
    id: String,
    edges: Vec<NodeRef>,
}

struct Graph {
    nodes: HashMap<String, NodeRef>,
}

impl Graph {
    fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
        }
    }

    fn add_node(&mut self, id: &str) -> NodeRef {
        let node = Rc::new(RefCell::new(GraphNode {
            id: id.to_string(),
            edges: Vec::new(),
        }));
        self.nodes.insert(id.to_string(), Rc::clone(&node));
        node
    }

    fn add_edge(&self, from: &str, to: &str) {
        if let (Some(from_node), Some(to_node)) = (self.nodes.get(from), self.nodes.get(to)) {
            from_node.borrow_mut().edges.push(Rc::clone(to_node));
        }
    }

    fn neighbors(&self, id: &str) -> Vec<String> {
        self.nodes
            .get(id)
            .map(|node| {
                node.borrow()
                    .edges
                    .iter()
                    .map(|n| n.borrow().id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn main() {
    let mut graph = Graph::new();
    graph.add_node("A");
    graph.add_node("B");
    graph.add_node("C");

    graph.add_edge("A", "B");
    graph.add_edge("A", "C");
    graph.add_edge("B", "C");

    println!("A の隣接ノード: {:?}", graph.neighbors("A")); // ["B", "C"]
    println!("B の隣接ノード: {:?}", graph.neighbors("B")); // ["C"]
    println!("C の隣接ノード: {:?}", graph.neighbors("C")); // []
}
```

---

## 4. RefCell<T> と内部可変性

### 例8: RefCell による実行時借用チェック

```rust
use std::cell::RefCell;

#[derive(Debug)]
struct Logger {
    messages: RefCell<Vec<String>>,
}

impl Logger {
    fn new() -> Self {
        Logger {
            messages: RefCell::new(Vec::new()),
        }
    }

    // &self (不変参照) なのに中身を変更できる
    fn log(&self, msg: &str) {
        self.messages.borrow_mut().push(msg.to_string());
    }

    fn dump(&self) {
        let msgs = self.messages.borrow(); // 不変借用
        for msg in msgs.iter() {
            println!("[LOG] {}", msg);
        }
    }

    fn count(&self) -> usize {
        self.messages.borrow().len()
    }

    fn clear(&self) {
        self.messages.borrow_mut().clear();
    }
}

fn main() {
    let logger = Logger::new();
    logger.log("初期化完了");
    logger.log("処理開始");
    logger.log("処理完了");
    println!("ログ件数: {}", logger.count());
    logger.dump();
    logger.clear();
    println!("クリア後のログ件数: {}", logger.count());
}
```

### 例9: Cell<T> の使い方

```rust
use std::cell::Cell;

struct Counter {
    count: Cell<u32>,
    name: String,
}

impl Counter {
    fn new(name: &str) -> Self {
        Counter {
            count: Cell::new(0),
            name: name.to_string(),
        }
    }

    fn increment(&self) {
        // Cell は get/set でアトミックに値を入れ替え
        self.count.set(self.count.get() + 1);
    }

    fn get_count(&self) -> u32 {
        self.count.get()
    }
}

// Cell と RefCell の違い
// Cell<T>: T が Copy の場合に使用。get/set のみ。参照を取れない
// RefCell<T>: 任意の T に使用。borrow/borrow_mut で参照を取得

fn main() {
    let counter = Counter::new("テスト");
    counter.increment();
    counter.increment();
    counter.increment();
    println!("{}: {}", counter.name, counter.get_count()); // 3
}
```

### 例10: Rc<RefCell<T>> パターン

```rust
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
struct Node {
    value: i32,
    children: Vec<Rc<RefCell<Node>>>,
}

impl Node {
    fn new(value: i32) -> Rc<RefCell<Node>> {
        Rc::new(RefCell::new(Node {
            value,
            children: vec![],
        }))
    }

    fn add_child(parent: &Rc<RefCell<Node>>, child: Rc<RefCell<Node>>) {
        parent.borrow_mut().children.push(child);
    }
}

fn sum_tree(node: &Rc<RefCell<Node>>) -> i32 {
    let borrowed = node.borrow();
    let mut sum = borrowed.value;
    for child in &borrowed.children {
        sum += sum_tree(child);
    }
    sum
}

fn main() {
    let root = Node::new(1);
    let child1 = Node::new(2);
    let child2 = Node::new(3);
    let grandchild = Node::new(4);

    Node::add_child(&child1, Rc::clone(&grandchild));
    Node::add_child(&root, Rc::clone(&child1));
    Node::add_child(&root, Rc::clone(&child2));

    // 共有所有の状態で値を変更
    grandchild.borrow_mut().value = 10;

    println!("ツリーの合計: {}", sum_tree(&root)); // 1 + 2 + 3 + 10 = 16
    println!("root: {:?}", root.borrow());
}
```

### 例11: RefCell のパニック回避テクニック

```rust
use std::cell::RefCell;

struct SafeContainer {
    data: RefCell<Vec<String>>,
}

impl SafeContainer {
    fn new() -> Self {
        SafeContainer {
            data: RefCell::new(Vec::new()),
        }
    }

    // try_borrow / try_borrow_mut でパニックを回避
    fn safe_push(&self, item: String) -> Result<(), String> {
        match self.data.try_borrow_mut() {
            Ok(mut data) => {
                data.push(item);
                Ok(())
            }
            Err(_) => Err("借用中のため変更できません".to_string()),
        }
    }

    fn safe_read(&self) -> Result<Vec<String>, String> {
        match self.data.try_borrow() {
            Ok(data) => Ok(data.clone()),
            Err(_) => Err("可変借用中のため読み取れません".to_string()),
        }
    }

    // 借用のスコープを明確に制限する
    fn process(&self) {
        // NG: この書き方はパニックする可能性がある
        // let r = self.data.borrow();
        // self.data.borrow_mut().push("new".to_string()); // パニック!

        // OK: 借用を早期にドロップ
        let items: Vec<String> = {
            let r = self.data.borrow();
            r.clone()
        };
        // ここで r はドロップ済み
        for item in items {
            println!("処理: {}", item);
        }
        self.data.borrow_mut().push("処理完了".to_string());
    }
}

fn main() {
    let container = SafeContainer::new();
    container.safe_push("hello".to_string()).unwrap();
    container.safe_push("world".to_string()).unwrap();
    container.process();
    println!("{:?}", container.safe_read().unwrap());
}
```

---

## 5. Mutex<T> と RwLock<T>

### 例12: Arc<Mutex<T>> によるスレッド安全な共有

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            // MutexGuard が drop される → 自動的にロック解放
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("結果: {}", *counter.lock().unwrap()); // 10
}
```

### 例13: RwLock によるリーダー・ライター制御

```rust
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

struct SharedConfig {
    data: Arc<RwLock<HashMap<String, String>>>,
}

use std::collections::HashMap;

impl SharedConfig {
    fn new() -> Self {
        SharedConfig {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn get(&self, key: &str) -> Option<String> {
        let read_lock = self.data.read().unwrap();
        read_lock.get(key).cloned()
    }

    fn set(&self, key: String, value: String) {
        let mut write_lock = self.data.write().unwrap();
        write_lock.insert(key, value);
    }

    fn get_all(&self) -> HashMap<String, String> {
        let read_lock = self.data.read().unwrap();
        read_lock.clone()
    }
}

impl Clone for SharedConfig {
    fn clone(&self) -> Self {
        SharedConfig {
            data: Arc::clone(&self.data),
        }
    }
}

fn main() {
    let config = SharedConfig::new();
    config.set("host".to_string(), "localhost".to_string());
    config.set("port".to_string(), "8080".to_string());

    let mut handles = vec![];

    // 複数リーダー
    for i in 0..5 {
        let config = config.clone();
        handles.push(thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            let all = config.get_all();
            println!("リーダー{}: {:?}", i, all);
        }));
    }

    // 1ライター
    {
        let config = config.clone();
        handles.push(thread::spawn(move || {
            config.set("debug".to_string(), "true".to_string());
            println!("ライター: debug=true を設定");
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("最終設定: {:?}", config.get_all());
}
```

### 例14: Mutex のデッドロック回避

```rust
use std::sync::{Arc, Mutex};
use std::thread;

// デッドロックの典型例
// スレッド1: lock(A) → lock(B)
// スレッド2: lock(B) → lock(A)
// → 互いにロックを待ち合って永久にブロック

// 回避策1: ロックの順序を統一する
fn safe_transfer(
    from: &Mutex<i64>,
    to: &Mutex<i64>,
    amount: i64,
) {
    // 常にアドレスが小さい方を先にロック
    let (first, second, is_reversed) = {
        let from_ptr = from as *const Mutex<i64> as usize;
        let to_ptr = to as *const Mutex<i64> as usize;
        if from_ptr < to_ptr {
            (from, to, false)
        } else {
            (to, from, true)
        }
    };

    let mut first_guard = first.lock().unwrap();
    let mut second_guard = second.lock().unwrap();

    if is_reversed {
        *first_guard += amount;
        *second_guard -= amount;
    } else {
        *first_guard -= amount;
        *second_guard += amount;
    }
}

// 回避策2: try_lock でタイムアウト
fn try_transfer(
    from: &Mutex<i64>,
    to: &Mutex<i64>,
    amount: i64,
) -> Result<(), &'static str> {
    // ロック取得を試み、失敗したらリトライ
    for _ in 0..100 {
        if let Ok(mut from_guard) = from.try_lock() {
            if let Ok(mut to_guard) = to.try_lock() {
                *from_guard -= amount;
                *to_guard += amount;
                return Ok(());
            }
        }
        std::thread::yield_now();
    }
    Err("ロック取得に失敗")
}

fn main() {
    let account_a = Arc::new(Mutex::new(1000i64));
    let account_b = Arc::new(Mutex::new(1000i64));

    let mut handles = vec![];

    // 複数スレッドで同時に送金
    for _ in 0..10 {
        let a = Arc::clone(&account_a);
        let b = Arc::clone(&account_b);
        handles.push(thread::spawn(move || {
            safe_transfer(&a, &b, 100);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("口座A: {}", account_a.lock().unwrap());
    println!("口座B: {}", account_b.lock().unwrap());
    // 合計は常に 2000
}
```

---

## 6. Cow (Clone on Write)

### 例15: Cow の基本

```rust
use std::borrow::Cow;

fn normalize(input: &str) -> Cow<'_, str> {
    if input.contains(' ') {
        // 変更が必要な場合のみ新しい String を作成
        Cow::Owned(input.replace(' ', "_"))
    } else {
        // 変更不要ならそのまま借用を返す
        Cow::Borrowed(input)
    }
}

fn main() {
    let s1 = normalize("hello_world"); // Borrowed → コピーなし
    let s2 = normalize("hello world"); // Owned → 新しい String
    println!("{}, {}", s1, s2);

    // Cow は Deref で &str として使える
    fn takes_str(s: &str) {
        println!("受信: {}", s);
    }
    takes_str(&s1);
    takes_str(&s2);
}
```

### 例16: Cow の実践的な活用

```rust
use std::borrow::Cow;

// エスケープ処理: 変更が必要な場合のみアロケーション
fn html_escape(input: &str) -> Cow<'_, str> {
    if input.contains(|c: char| matches!(c, '<' | '>' | '&' | '"' | '\'')) {
        let mut result = String::with_capacity(input.len());
        for ch in input.chars() {
            match ch {
                '<' => result.push_str("&lt;"),
                '>' => result.push_str("&gt;"),
                '&' => result.push_str("&amp;"),
                '"' => result.push_str("&quot;"),
                '\'' => result.push_str("&#39;"),
                _ => result.push(ch),
            }
        }
        Cow::Owned(result)
    } else {
        Cow::Borrowed(input)
    }
}

// パス正規化: 必要な場合のみクローン
fn normalize_path(path: &str) -> Cow<'_, str> {
    if path.starts_with("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        Cow::Owned(format!("{}{}", home, &path[1..]))
    } else if path.contains("//") {
        Cow::Owned(path.replace("//", "/"))
    } else {
        Cow::Borrowed(path)
    }
}

// Cow をジェネリックに使う
fn process_items<'a>(items: &'a [String], prefix: &str) -> Vec<Cow<'a, str>> {
    items
        .iter()
        .map(|item| {
            if item.starts_with(prefix) {
                Cow::Borrowed(item.as_str())
            } else {
                Cow::Owned(format!("{}{}", prefix, item))
            }
        })
        .collect()
}

fn main() {
    // HTML エスケープ
    let safe = html_escape("Hello World");       // Borrowed
    let escaped = html_escape("<script>alert('xss')</script>"); // Owned
    println!("安全: {}", safe);
    println!("エスケープ: {}", escaped);

    // パス正規化
    let path1 = normalize_path("/usr/local/bin");  // Borrowed
    let path2 = normalize_path("~/Documents");     // Owned
    println!("パス1: {}", path1);
    println!("パス2: {}", path2);

    // ジェネリック処理
    let items = vec!["hello".to_string(), "prefix_world".to_string()];
    let processed = process_items(&items, "prefix_");
    for item in &processed {
        println!("  {}", item);
    }
}
```

---

## 7. Pin<P>

### 7.1 Pin の概要

`Pin<P>` はポインタ `P` が指すデータのメモリ上の位置を固定する。主に自己参照型と async/await で使用される。

```
┌──────────────────────────────────────────────────────┐
│  Pin<P> の目的                                        │
├──────────────────────────────────────────────────────┤
│                                                      │
│  問題: 自己参照型はムーブすると内部参照が壊れる       │
│                                                      │
│  ムーブ前:                                           │
│  ┌──────────────────┐                                │
│  │ data: "hello"    │ ← ptr が data を指す           │
│  │ ptr: &data ──────┘                                │
│  └──────────────────┘                                │
│                                                      │
│  ムーブ後:                                           │
│  ┌──────────────────┐  ← 新しいアドレス              │
│  │ data: "hello"    │                                │
│  │ ptr: &旧data ────── → ダングリング！              │
│  └──────────────────┘                                │
│                                                      │
│  Pin を使うと: ムーブが防止される → 安全              │
└──────────────────────────────────────────────────────┘
```

### 例17: Pin の基本的な使い方

```rust
use std::pin::Pin;
use std::marker::PhantomPinned;

struct SelfReferential {
    data: String,
    // 自己参照ポインタ(初期化後にセット)
    ptr: *const String,
    // Unpin を実装しないようにする
    _pin: PhantomPinned,
}

impl SelfReferential {
    fn new(data: String) -> Pin<Box<Self>> {
        let s = SelfReferential {
            data,
            ptr: std::ptr::null(),
            _pin: PhantomPinned,
        };
        let mut boxed = Box::pin(s);

        // 自己参照ポインタをセット
        let self_ptr: *const String = &boxed.data;
        unsafe {
            let mut_ref = Pin::as_mut(&mut boxed);
            Pin::get_unchecked_mut(mut_ref).ptr = self_ptr;
        }

        boxed
    }

    fn get_data(&self) -> &str {
        &self.data
    }

    fn get_ptr_data(&self) -> &str {
        unsafe { &*self.ptr }
    }
}

fn main() {
    let pinned = SelfReferential::new("Hello, Pin!".to_string());
    println!("data: {}", pinned.get_data());
    println!("ptr→data: {}", pinned.get_ptr_data());

    // pinned はムーブできない(Pin で固定されている)
    // let moved = pinned; // コンパイルエラー (PhantomPinned のため)
}
```

### 例18: async/await と Pin

```rust
use std::pin::Pin;
use std::future::Future;

// async fn は内部的に自己参照構造を生成する
// そのため Future を直接扱う場合は Pin が必要

// 手動で Future トレイトを実装する例
struct CountdownFuture {
    count: u32,
}

impl Future for CountdownFuture {
    type Output = String;

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.count == 0 {
            std::task::Poll::Ready("カウントダウン完了!".to_string())
        } else {
            self.count -= 1;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}

// Pin を要求する関数
fn execute_pinned<F: Future<Output = String>>(future: Pin<Box<F>>) {
    // 実際にはランタイムが poll を呼ぶ
    println!("Future を受け取りました");
}

fn main() {
    let future = CountdownFuture { count: 3 };
    let pinned = Box::pin(future);
    execute_pinned(pinned);
}
```

---

## 8. カスタムスマートポインタの設計

### 例19: 監査ログ付きスマートポインタ

```rust
use std::ops::{Deref, DerefMut};
use std::cell::Cell;

struct Audited<T> {
    value: T,
    read_count: Cell<u64>,
    write_count: Cell<u64>,
    name: String,
}

impl<T> Audited<T> {
    fn new(name: &str, value: T) -> Self {
        Audited {
            value,
            read_count: Cell::new(0),
            write_count: Cell::new(0),
            name: name.to_string(),
        }
    }

    fn stats(&self) -> (u64, u64) {
        (self.read_count.get(), self.write_count.get())
    }
}

impl<T> Deref for Audited<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.read_count.set(self.read_count.get() + 1);
        &self.value
    }
}

impl<T> DerefMut for Audited<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.write_count.set(self.write_count.get() + 1);
        &mut self.value
    }
}

impl<T> Drop for Audited<T> {
    fn drop(&mut self) {
        let (reads, writes) = self.stats();
        println!("[監査] '{}': 読取={}, 書込={}", self.name, reads, writes);
    }
}

fn main() {
    {
        let mut data = Audited::new("重要データ", vec![1, 2, 3]);

        // 読み取りアクセス
        println!("長さ: {}", data.len());       // Deref → read_count++
        println!("先頭: {}", data[0]);           // Deref → read_count++

        // 書き込みアクセス
        data.push(4);                            // DerefMut → write_count++
        data.push(5);                            // DerefMut → write_count++

        let (reads, writes) = data.stats();
        println!("中間統計: 読取={}, 書込={}", reads, writes);
    }
    // ドロップ時に最終統計が表示される
}
```

### 例20: プール管理スマートポインタ

```rust
use std::ops::Deref;
use std::sync::{Arc, Mutex};

struct PoolItem<T> {
    value: T,
    pool: Arc<Mutex<Vec<T>>>,
}

impl<T> Deref for PoolItem<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> Drop for PoolItem<T> {
    fn drop(&mut self) {
        // ドロップ時にプールに返却
        // 注意: std::mem::replace で値を取り出す
        let value = unsafe {
            std::ptr::read(&self.value)
        };
        if let Ok(mut pool) = self.pool.lock() {
            pool.push(value);
            println!("プールに返却 (プールサイズ: {})", pool.len());
        }
        // value のドロップを防ぐ
        std::mem::forget(std::mem::ManuallyDrop::new(()));
    }
}

struct Pool<T> {
    items: Arc<Mutex<Vec<T>>>,
}

impl<T> Pool<T> {
    fn new(items: Vec<T>) -> Self {
        Pool {
            items: Arc::new(Mutex::new(items)),
        }
    }

    fn acquire(&self) -> Option<PoolItem<T>> {
        let mut items = self.items.lock().unwrap();
        items.pop().map(|value| {
            println!("プールから取得 (残り: {})", items.len());
            PoolItem {
                value,
                pool: Arc::clone(&self.items),
            }
        })
    }
}

fn main() {
    let pool = Pool::new(vec![
        String::from("接続1"),
        String::from("接続2"),
        String::from("接続3"),
    ]);

    {
        let conn1 = pool.acquire().unwrap();
        let conn2 = pool.acquire().unwrap();
        println!("使用中: {}, {}", *conn1, *conn2);
        // conn2 がドロップ → プールに返却
    }
    // conn1 もドロップ → プールに返却

    // 返却された接続を再利用
    let conn3 = pool.acquire().unwrap();
    println!("再利用: {}", *conn3);
}
```

---

## 9. 比較表

### 9.1 スマートポインタ選択ガイド

| 要件 | 選択する型 | 理由 |
|------|-----------|------|
| ヒープに確保したい | `Box<T>` | 最も単純。単一所有 |
| 同一スレッドで共有 | `Rc<T>` | 参照カウントで共有所有 |
| スレッド間で共有(読取のみ) | `Arc<T>` | アトミック参照カウント |
| スレッド間で共有(変更あり) | `Arc<Mutex<T>>` | ロックで排他アクセス |
| 不変参照越しに変更 | `RefCell<T>` | 実行時借用チェック |
| Copy型の内部可変性 | `Cell<T>` | get/set で値を差し替え |
| 多読み少書き(スレッド) | `Arc<RwLock<T>>` | 読み取りは並行可能 |
| 変更時のみクローン | `Cow<'a, T>` | 不要なアロケーション回避 |
| 循環参照の防止 | `Weak<T>` | 参照カウントに含まれない |
| 自己参照・async | `Pin<P>` | メモリ位置を固定 |

### 9.2 Rc vs Arc

| 特性 | Rc<T> | Arc<T> |
|------|-------|--------|
| スレッド安全 | No (Send 未実装) | Yes |
| 参照カウント操作 | 通常の加減算 | アトミック操作 |
| オーバーヘッド | 小さい | やや大きい |
| 用途 | 単一スレッド内の共有 | マルチスレッドの共有 |
| 内部可変性 | + RefCell | + Mutex / RwLock |

### 9.3 RefCell vs Mutex vs RwLock

| 特性 | RefCell<T> | Mutex<T> | RwLock<T> |
|------|-----------|----------|-----------|
| スレッド安全 | No | Yes | Yes |
| 借用チェック | 実行時 | ロック | ロック |
| 複数リーダー | borrow() で可能 | 不可 | read() で可能 |
| ライターの排他性 | borrow_mut() | lock() | write() |
| 違反時の動作 | パニック | ブロック | ブロック |
| try操作 | try_borrow() | try_lock() | try_read/try_write() |
| Poison | なし | あり | あり |
| オーバーヘッド | 小さい | 中程度 | 中程度 |

### 9.4 メモリレイアウト比較

| 型 | スタックサイズ | ヒープ使用 |
|----|---------------|-----------|
| `T` | `size_of::<T>()` | なし |
| `Box<T>` | ポインタ1つ (8B) | `size_of::<T>()` |
| `Rc<T>` | ポインタ1つ (8B) | `size_of::<T>()` + カウンタ(16B) |
| `Arc<T>` | ポインタ1つ (8B) | `size_of::<T>()` + アトミックカウンタ(16B) |
| `RefCell<T>` | `size_of::<T>()` + フラグ | なし |
| `Cell<T>` | `size_of::<T>()` | なし |

---

## 10. アンチパターン

### アンチパターン1: RefCell の実行時パニック

```rust
use std::cell::RefCell;

// BAD: 同時に borrow と borrow_mut → 実行時パニック
fn bad_example() {
    let data = RefCell::new(vec![1, 2, 3]);
    let r1 = data.borrow();     // 不変借用
    // let r2 = data.borrow_mut(); // パニック！不変借用中に可変借用
    println!("{:?}", r1);
}

// GOOD: 借用のスコープを制限する
fn good_example() {
    let data = RefCell::new(vec![1, 2, 3]);
    {
        let r1 = data.borrow();
        println!("{:?}", r1);
    } // r1 が drop される
    let mut r2 = data.borrow_mut(); // OK
    r2.push(4);
}

fn main() {
    bad_example();
    good_example();
}
```

### アンチパターン2: 不要な Arc<Mutex<T>>

```rust
use std::sync::{Arc, Mutex};
use std::cell::RefCell;

// BAD: 単一スレッドなのに Arc<Mutex> を使用
fn single_thread_bad() {
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));
    let mut guard = data.lock().unwrap();
    guard.push(4);
}

// GOOD: 単一スレッドなら RefCell で十分
fn single_thread_good() {
    let data = RefCell::new(vec![1, 2, 3]);
    data.borrow_mut().push(4);
}

fn main() {
    single_thread_bad();
    single_thread_good();
}
```

### アンチパターン3: Rc の循環参照によるメモリリーク

```rust
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug)]
struct BadNode {
    value: i32,
    next: RefCell<Option<Rc<BadNode>>>,
}

fn demonstrate_leak() {
    let a = Rc::new(BadNode {
        value: 1,
        next: RefCell::new(None),
    });
    let b = Rc::new(BadNode {
        value: 2,
        next: RefCell::new(Some(Rc::clone(&a))),
    });
    // 循環参照を作成！
    *a.next.borrow_mut() = Some(Rc::clone(&b));

    // a → b → a → b → ... の循環
    // 参照カウントが 0 にならないためメモリリーク
    println!("a の参照カウント: {}", Rc::strong_count(&a)); // 2
    println!("b の参照カウント: {}", Rc::strong_count(&b)); // 2
}
// a と b がドロップされても、互いに参照し合っているため
// 参照カウントは 1 のまま → メモリリーク

// GOOD: Weak を使って循環を防ぐ
use std::rc::Weak;

#[derive(Debug)]
struct GoodNode {
    value: i32,
    next: RefCell<Option<Rc<GoodNode>>>,
    prev: RefCell<Option<Weak<GoodNode>>>,  // 弱い参照
}

fn main() {
    demonstrate_leak();
    println!("循環参照のデモ完了 (メモリリークが発生)");
}
```

### アンチパターン4: Box の不要な使用

```rust
// BAD: 小さい値を Box に包む必要はない
fn bad_box() {
    let x = Box::new(42);  // ヒープ確保のオーバーヘッド
    println!("{}", x);
}

// GOOD: スタックに直接置く
fn good_stack() {
    let x = 42;
    println!("{}", x);
}

// Box が必要なケース
fn good_box_usage() {
    // 1. 再帰型
    enum List<T> {
        Cons(T, Box<List<T>>),
        Nil,
    }

    // 2. トレイトオブジェクト
    let _: Box<dyn std::fmt::Display> = Box::new(42);

    // 3. 大きなデータ
    let _: Box<[u8; 1_000_000]> = Box::new([0u8; 1_000_000]);
}

fn main() {
    bad_box();
    good_stack();
    good_box_usage();
}
```

---

## 11. FAQ

### Q1: Box<T> はいつ使いますか？

**A:** 主に以下の場面です:
- **再帰的なデータ構造** (リスト、ツリーなど): コンパイル時にサイズが決まらない
- **大きなデータのムーブ回避**: スタックではなくヒープに置く
- **トレイトオブジェクト**: `Box<dyn Trait>` で動的ディスパッチ
- **所有権の転送**: ヒープに置いてポインタだけムーブ

### Q2: Rc の循環参照はどう防ぎますか？

**A:** `Weak<T>` を使います。Rc::downgrade() で弱い参照を作成し、循環参照を防ぎます:
```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;
struct Node {
    parent: RefCell<Weak<Node>>,   // 弱い参照 → 循環を防ぐ
    children: RefCell<Vec<Rc<Node>>>, // 強い参照
}
```
Weak は参照カウントに含まれないため、循環があっても正しくメモリが解放されます。

### Q3: Pin<T> は何のためにありますか？

**A:** Pin は値のメモリ上の位置を固定し、ムーブを防止します。主に以下で必要です:
- **async/await**: Future は自己参照構造を持つため、ムーブすると参照が壊れる
- **自己参照型**: 構造体内の参照が自身のフィールドを指す場合
- 多くの場合、`Box::pin()` や `pin!()` マクロ経由で使います。

### Q4: Mutex の Poison (毒) とは何ですか？

**A:** ロックを保持したスレッドがパニックした場合、Mutex は "poisoned" 状態になります。他のスレッドが `lock()` すると `PoisonError` が返されます。これは、パニックによりデータが不整合な状態にある可能性を通知するための安全機構です:
```rust
use std::sync::Mutex;
let m = Mutex::new(42);

// Poison を無視して使う場合
let value = m.lock().unwrap_or_else(|e| e.into_inner());
```

### Q5: Cell<T> と RefCell<T> はどう使い分けますか？

**A:**
- `Cell<T>` は `T: Copy` の型に使用。値の get/set のみで、参照を取ることはできない。オーバーヘッドが最小
- `RefCell<T>` は任意の型に使用。`borrow()` / `borrow_mut()` で参照を取得する。実行時に借用ルールを検証する
- 可能なら `Cell<T>` を優先。参照が必要な場合は `RefCell<T>` を使う

### Q6: Arc<Mutex<T>> vs Arc<RwLock<T>> の選択基準は？

**A:**
- **Arc<Mutex<T>>**: 読み取りと書き込みが同程度の頻度。実装が単純。デッドロックのリスクが低い
- **Arc<RwLock<T>>**: 読み取りが圧倒的に多く、書き込みは稀。読み取り同士は並行実行可能なので高スループット
- 迷ったらまず `Mutex` を使い、プロファイリングで読み取りがボトルネックと分かったら `RwLock` に移行する

---

## 12. まとめ

| 概念 | 要点 |
|------|------|
| Box<T> | ヒープ確保。単一所有。再帰型やdyn Traitに必須 |
| Rc<T> | 参照カウントで共有所有。単一スレッド専用 |
| Arc<T> | アトミック参照カウント。マルチスレッド対応 |
| Weak<T> | 弱い参照。循環参照を防止。カウントに含まれない |
| RefCell<T> | 実行時借用チェック。内部可変性パターン |
| Cell<T> | Copy型の内部可変性。get/setのみ |
| Mutex<T> | 排他ロック。Arc と組み合わせてスレッド間共有 |
| RwLock<T> | 読み書きロック。多読み少書きに最適 |
| Cow<T> | 変更時のみクローン。不要なアロケーション回避 |
| Pin<P> | メモリ位置を固定。async/await に必須 |
| Deref/Drop | スマートポインタの基盤トレイト |

---

## 次に読むべきガイド

- [02-closures-fn-traits.md](02-closures-fn-traits.md) -- クロージャとFnトレイト
- [../03-systems/01-concurrency.md](../03-systems/01-concurrency.md) -- 並行プログラミング詳解
- [03-unsafe-rust.md](03-unsafe-rust.md) -- unsafe と生ポインタ

---

## 参考文献

1. **The Rust Programming Language - Ch.15 Smart Pointers** -- https://doc.rust-lang.org/book/ch15-00-smart-pointers.html
2. **The Rustonomicon - Concurrency** -- https://doc.rust-lang.org/nomicon/concurrency.html
3. **Rust std::cell Module** -- https://doc.rust-lang.org/std/cell/
4. **Rust std::sync Module** -- https://doc.rust-lang.org/std/sync/
5. **Rust std::pin Module** -- https://doc.rust-lang.org/std/pin/
6. **Rust std::borrow::Cow** -- https://doc.rust-lang.org/std/borrow/enum.Cow.html
