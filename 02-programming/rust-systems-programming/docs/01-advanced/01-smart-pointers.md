# スマートポインタ -- 所有権と参照カウントによる柔軟なメモリ管理

> Box、Rc、Arc、RefCell、Mutex等のスマートポインタは、所有権システムの制約を安全に緩和し、共有所有権や内部可変性といった高度なパターンを実現する。

---

## この章で学ぶこと

1. **Box / Rc / Arc** -- ヒープ確保、参照カウント、スレッド安全な共有所有権を理解する
2. **RefCell / Cell** -- 内部可変性パターンで不変参照越しにデータを変更する方法を習得する
3. **Mutex / RwLock** -- スレッド間でデータを安全に共有する仕組みを学ぶ

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
└─────────────┴────────────────┴──────────────────────────────┘
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

---

## 3. Rc<T> と Arc<T>

### 例2: Rc による共有所有権

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

### 例3: Arc によるスレッド間共有

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

---

## 4. RefCell<T> と内部可変性

### 例4: RefCell による実行時借用チェック

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
}

fn main() {
    let logger = Logger::new();
    logger.log("初期化完了");
    logger.log("処理開始");
    logger.log("処理完了");
    logger.dump();
}
```

### 例5: Rc<RefCell<T>> パターン

```rust
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
struct Node {
    value: i32,
    children: Vec<Rc<RefCell<Node>>>,
}

fn main() {
    let leaf = Rc::new(RefCell::new(Node {
        value: 3,
        children: vec![],
    }));

    let branch = Rc::new(RefCell::new(Node {
        value: 5,
        children: vec![Rc::clone(&leaf)],
    }));

    // leaf の値を変更(複数の所有者がいても変更可能)
    leaf.borrow_mut().value = 10;

    println!("branch: {:?}", branch.borrow());
}
```

---

## 5. Mutex<T> と RwLock<T>

### 例6: Arc<Mutex<T>> によるスレッド安全な共有

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

### 例7: Cow (Clone on Write)

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
}
```

---

## 6. 比較表

### 6.1 スマートポインタ選択ガイド

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

### 6.2 Rc vs Arc

| 特性 | Rc<T> | Arc<T> |
|------|-------|--------|
| スレッド安全 | No (Send 未実装) | Yes |
| 参照カウント操作 | 通常の加減算 | アトミック操作 |
| オーバーヘッド | 小さい | やや大きい |
| 用途 | 単一スレッド内の共有 | マルチスレッドの共有 |
| 内部可変性 | + RefCell | + Mutex / RwLock |

---

## 7. アンチパターン

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
```

### アンチパターン2: 不要な Arc<Mutex<T>>

```rust
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
```

---

## 8. FAQ

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

---

## 9. まとめ

| 概念 | 要点 |
|------|------|
| Box<T> | ヒープ確保。単一所有。再帰型やdyn Traitに必須 |
| Rc<T> | 参照カウントで共有所有。単一スレッド専用 |
| Arc<T> | アトミック参照カウント。マルチスレッド対応 |
| RefCell<T> | 実行時借用チェック。内部可変性パターン |
| Cell<T> | Copy型の内部可変性。get/setのみ |
| Mutex<T> | 排他ロック。Arc と組み合わせてスレッド間共有 |
| RwLock<T> | 読み書きロック。多読み少書きに最適 |
| Cow<T> | 変更時のみクローン。不要なアロケーション回避 |

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
