# Smart Pointers -- Flexible Memory Management via Ownership and Reference Counting

> Smart pointers such as Box, Rc, Arc, RefCell, and Mutex safely relax the constraints of the ownership system, enabling advanced patterns like shared ownership and interior mutability.

---

## What You Will Learn in This Chapter

1. **Box / Rc / Arc** -- Understand heap allocation, reference counting, and thread-safe shared ownership
2. **RefCell / Cell** -- Master the interior mutability pattern for modifying data through immutable references
3. **Mutex / RwLock** -- Learn the mechanisms for safely sharing data between threads
4. **Cow / Pin** -- Understand the use cases and implementations of lazy cloning and memory pinning
5. **Custom smart pointers** -- Implement the Deref / Drop traits to design your own smart pointers


## Prerequisites

Reading the following before this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- The content of [Lifetimes Explained -- The Mechanism for Proving Reference Validity at Compile Time](./00-lifetimes.md)

---

## 1. Overview of Smart Pointers

```
┌─────────────────────────────────────────────────────────────┐
│                  Smart Pointer Catalog                      │
├─────────────┬────────────────┬──────────────────────────────┤
│ Type        │ Ownership      │ Primary Use                  │
├─────────────┼────────────────┼──────────────────────────────┤
│ Box<T>      │ Single owner   │ Heap allocation, recursive   │
│ Rc<T>       │ Shared (1-thr) │ Graphs, multiple owners      │
│ Arc<T>      │ Shared (multi) │ Sharing across threads       │
│ Cell<T>     │ Interior (Copy)│ Modify through &T            │
│ RefCell<T>  │ Interior mut.  │ Runtime borrow checking      │
│ Mutex<T>    │ Exclusive lock │ Mutable cross-thread access  │
│ RwLock<T>   │ Read/write lock│ Many readers, few writers    │
│ Cow<'a, T>  │ Lazy clone     │ Clone only on modification   │
│ Pin<P>      │ Pinned         │ Self-referential, async      │
│ Weak<T>     │ Weak reference │ Prevent reference cycles     │
└─────────────┴────────────────┴──────────────────────────────┘
```

### 1.1 What Is a Smart Pointer?

A smart pointer is a data structure that behaves like a pointer but carries additional metadata or functionality. In Rust, by implementing the `Deref` and `Drop` traits, you can use a type like an ordinary reference while it manages resources automatically.

```rust
use std::ops::Deref;

// How the Deref trait works
// Box<T> implements Deref<Target = T>
fn takes_str(s: &str) {
    println!("{}", s);
}

fn main() {
    let boxed_string = Box::new(String::from("hello"));

    // Box<String> -> &String -> &str (Deref coercion)
    takes_str(&boxed_string);

    // Explicit Deref
    let s: &String = &*boxed_string;
    println!("{}", s);
}
```

### 1.2 The Importance of Deref and Drop

```
┌──────────────────────────────────────────────────────┐
│  Deref trait                                         │
│  - Lets you use a smart pointer transparently        │
│    as a reference                                    │
│  - Deref coercion: &Box<T> -> &T automatic conversion│
│  - DerefMut: automatic conversion to mutable refs    │
│                                                      │
│  Drop trait                                          │
│  - Automatically releases resources at scope exit    │
│  - File handles, network connections, memory release │
│  - Realizes the RAII pattern                         │
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
        println!("MyBox has been dropped");
    }
}

fn main() {
    let x = MyBox::new(42);
    println!("Value: {}", *x); // Deref to reference an i32

    let mut s = MyBox::new(String::from("hello"));
    s.push_str(" world"); // DerefMut for mutable reference as String
    println!("{}", *s);
} // MyBox is dropped
```

---

## 2. Box<T>

### Example 1: Box Basics and Recursive Types

```rust
// Recursive data structure (size unknown at compile time)
#[derive(Debug)]
enum List {
    Cons(i32, Box<List>),
    Nil,
}

fn main() {
    // Box: allocate data on the heap
    let b = Box::new(5);
    println!("Box contents: {}", b);

    // Using a recursive type
    let list = List::Cons(1,
        Box::new(List::Cons(2,
            Box::new(List::Cons(3,
                Box::new(List::Nil))))));
    println!("{:?}", list);
}
```

### Memory Layout of Box

```
  Stack             Heap
  ┌──────────┐     ┌───────────────┐
  │ Box<i32> │     │               │
  │ ptr ─────────>│     42        │
  │          │     │               │
  └──────────┘     └───────────────┘
  8 bytes           4 bytes

  Box<T> is the size of a single pointer on the stack
  Heap memory is automatically released on Drop
```

### Example 2: Trait Objects via Box

```rust
trait Animal {
    fn name(&self) -> &str;
    fn sound(&self) -> &str;
    fn info(&self) -> String {
        format!("{} says {}", self.name(), self.sound())
    }
}

struct Dog {
    name: String,
}

impl Animal for Dog {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "Woof" }
}

struct Cat {
    name: String,
}

impl Animal for Cat {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "Meow" }
}

// Dynamic dispatch via Box<dyn Trait>
fn create_animal(kind: &str, name: &str) -> Box<dyn Animal> {
    match kind {
        "dog" => Box::new(Dog { name: name.to_string() }),
        "cat" => Box::new(Cat { name: name.to_string() }),
        _ => panic!("Unknown animal kind"),
    }
}

fn main() {
    let animals: Vec<Box<dyn Animal>> = vec![
        create_animal("dog", "Pochi"),
        create_animal("cat", "Tama"),
        create_animal("dog", "Hachi"),
    ];

    for animal in &animals {
        println!("{}", animal.info());
    }
}
```

### Example 3: Move Optimization for Large Data with Box

```rust
// A large struct
struct LargeStruct {
    data: [u8; 1_000_000], // 1MB
}

// On the stack, moving incurs a 1MB copy
// Wrapped in Box, only the pointer (8 bytes) is copied
fn create_large() -> Box<LargeStruct> {
    Box::new(LargeStruct {
        data: [0u8; 1_000_000],
    })
}

fn process_large(data: Box<LargeStruct>) {
    println!("Data size: {} bytes", data.data.len());
    // Box releases heap memory at the end of its scope
}

fn main() {
    let large = create_large(); // Only the pointer is moved
    process_large(large);       // Only the pointer is moved

    // Box is the size of a single pointer
    println!("Size of Box<LargeStruct>: {} bytes",
        std::mem::size_of::<Box<LargeStruct>>());
    // 8 bytes (on 64-bit platforms)
}
```

---

## 3. Rc<T> and Arc<T>

### Example 4: Shared Ownership via Rc

```rust
use std::rc::Rc;

#[derive(Debug)]
struct SharedData {
    value: String,
}

fn main() {
    let data = Rc::new(SharedData {
        value: "shared data".to_string(),
    });
    println!("Reference count: {}", Rc::strong_count(&data)); // 1

    let data2 = Rc::clone(&data); // Increases the reference count (does not copy data)
    println!("Reference count: {}", Rc::strong_count(&data)); // 2

    {
        let data3 = Rc::clone(&data);
        println!("Reference count: {}", Rc::strong_count(&data)); // 3
    } // data3 is dropped -> reference count decreases

    println!("Reference count: {}", Rc::strong_count(&data)); // 2
    println!("Value: {}", data.value);
}
```

### Example 5: Cross-Thread Sharing with Arc

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
            println!("Thread {}: sum={}", i, sum);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
```

### How Rc / Arc Work

```
  Rc::clone only increments the reference count
  The data itself is not copied

  data ──┐
         │     ┌────────────────────────┐
  data2 ─┼────>│ strong_count: 3        │
         │     │ weak_count: 0          │
  data3 ─┘     │ ┌────────────────────┐ │
               │ │ value: SharedData  │ │
               │ └────────────────────┘ │
               └────────────────────────┘

  Data is freed when all Rcs are dropped (count == 0)
  Rc: single-threaded use (does not implement Send)
  Arc: multi-threaded use (manages reference count via atomic operations)
```

### Example 6: Preventing Reference Cycles with Weak<T>

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
        // Set a weak reference to the parent on the child node
        *child.parent.borrow_mut() = Rc::downgrade(parent);
        // Add a strong reference to the child on the parent node
        parent.children.borrow_mut().push(Rc::clone(child));
    }
}

fn main() {
    let root = Node::new(1);
    let child1 = Node::new(2);
    let child2 = Node::new(3);

    Node::add_child(&root, &child1);
    Node::add_child(&root, &child2);

    // Check the reference counts
    println!("Strong refs to root: {}", Rc::strong_count(&root));
    println!("Weak refs to root: {}", Rc::weak_count(&root));
    println!("Strong refs to child1: {}", Rc::strong_count(&child1));

    // Promote a weak reference to a strong reference
    if let Some(parent) = child1.parent.borrow().upgrade() {
        println!("child1's parent value: {}", parent.value);
    }

    // After dropping root, the weak reference becomes invalid
    drop(root);
    println!("child1's parent after dropping root: {:?}",
        child1.parent.borrow().upgrade()); // None
}
```

### Example 7: Graph Structure Using Rc

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

    println!("Neighbors of A: {:?}", graph.neighbors("A")); // ["B", "C"]
    println!("Neighbors of B: {:?}", graph.neighbors("B")); // ["C"]
    println!("Neighbors of C: {:?}", graph.neighbors("C")); // []
}
```

---

## 4. RefCell<T> and Interior Mutability

### Example 8: Runtime Borrow Checking with RefCell

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

    // Modifies contents even though &self (immutable reference) is taken
    fn log(&self, msg: &str) {
        self.messages.borrow_mut().push(msg.to_string());
    }

    fn dump(&self) {
        let msgs = self.messages.borrow(); // immutable borrow
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
    logger.log("initialization complete");
    logger.log("processing started");
    logger.log("processing complete");
    println!("Number of logs: {}", logger.count());
    logger.dump();
    logger.clear();
    println!("Number of logs after clearing: {}", logger.count());
}
```

### Example 9: How to Use Cell<T>

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
        // Cell atomically swaps the value via get/set
        self.count.set(self.count.get() + 1);
    }

    fn get_count(&self) -> u32 {
        self.count.get()
    }
}

// Difference between Cell and RefCell
// Cell<T>: used when T is Copy. Only get/set; cannot obtain a reference
// RefCell<T>: usable for any T. Acquire references via borrow/borrow_mut

fn main() {
    let counter = Counter::new("test");
    counter.increment();
    counter.increment();
    counter.increment();
    println!("{}: {}", counter.name, counter.get_count()); // 3
}
```

### Example 10: The Rc<RefCell<T>> Pattern

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

    // Modify the value while in shared ownership
    grandchild.borrow_mut().value = 10;

    println!("Tree sum: {}", sum_tree(&root)); // 1 + 2 + 3 + 10 = 16
    println!("root: {:?}", root.borrow());
}
```

### Example 11: Techniques to Avoid Panics with RefCell

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

    // Avoid panics with try_borrow / try_borrow_mut
    fn safe_push(&self, item: String) -> Result<(), String> {
        match self.data.try_borrow_mut() {
            Ok(mut data) => {
                data.push(item);
                Ok(())
            }
            Err(_) => Err("Cannot modify because it is currently borrowed".to_string()),
        }
    }

    fn safe_read(&self) -> Result<Vec<String>, String> {
        match self.data.try_borrow() {
            Ok(data) => Ok(data.clone()),
            Err(_) => Err("Cannot read because it is currently mutably borrowed".to_string()),
        }
    }

    // Clearly limit the scope of borrows
    fn process(&self) {
        // BAD: this style can panic
        // let r = self.data.borrow();
        // self.data.borrow_mut().push("new".to_string()); // panic!

        // OK: drop the borrow early
        let items: Vec<String> = {
            let r = self.data.borrow();
            r.clone()
        };
        // r has been dropped here
        for item in items {
            println!("processing: {}", item);
        }
        self.data.borrow_mut().push("processing complete".to_string());
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

## 5. Mutex<T> and RwLock<T>

### Example 12: Thread-Safe Sharing via Arc<Mutex<T>>

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
            // MutexGuard is dropped -> lock is automatically released
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap()); // 10
}
```

### Example 13: Reader/Writer Control with RwLock

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

    // Multiple readers
    for i in 0..5 {
        let config = config.clone();
        handles.push(thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            let all = config.get_all();
            println!("Reader {}: {:?}", i, all);
        }));
    }

    // One writer
    {
        let config = config.clone();
        handles.push(thread::spawn(move || {
            config.set("debug".to_string(), "true".to_string());
            println!("Writer: set debug=true");
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final config: {:?}", config.get_all());
}
```

### Example 14: Avoiding Deadlocks with Mutex

```rust
use std::sync::{Arc, Mutex};
use std::thread;

// Classic deadlock example
// Thread 1: lock(A) -> lock(B)
// Thread 2: lock(B) -> lock(A)
// -> They wait for each other's locks and block forever

// Mitigation 1: enforce a consistent lock order
fn safe_transfer(
    from: &Mutex<i64>,
    to: &Mutex<i64>,
    amount: i64,
) {
    // Always lock the one with the smaller address first
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

// Mitigation 2: timeout via try_lock
fn try_transfer(
    from: &Mutex<i64>,
    to: &Mutex<i64>,
    amount: i64,
) -> Result<(), &'static str> {
    // Attempt to acquire the lock; retry on failure
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
    Err("failed to acquire lock")
}

fn main() {
    let account_a = Arc::new(Mutex::new(1000i64));
    let account_b = Arc::new(Mutex::new(1000i64));

    let mut handles = vec![];

    // Concurrent transfers from multiple threads
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

    println!("Account A: {}", account_a.lock().unwrap());
    println!("Account B: {}", account_b.lock().unwrap());
    // The total is always 2000
}
```

---

## 6. Cow (Clone on Write)

### Example 15: Cow Basics

```rust
use std::borrow::Cow;

fn normalize(input: &str) -> Cow<'_, str> {
    if input.contains(' ') {
        // Create a new String only when modification is required
        Cow::Owned(input.replace(' ', "_"))
    } else {
        // If no modification is needed, return the borrow as-is
        Cow::Borrowed(input)
    }
}

fn main() {
    let s1 = normalize("hello_world"); // Borrowed -> no copy
    let s2 = normalize("hello world"); // Owned -> new String
    println!("{}, {}", s1, s2);

    // Cow can be used as &str via Deref
    fn takes_str(s: &str) {
        println!("received: {}", s);
    }
    takes_str(&s1);
    takes_str(&s2);
}
```

### Example 16: Practical Use of Cow

```rust
use std::borrow::Cow;

// Escape processing: allocate only when modification is needed
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

// Path normalization: clone only when needed
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

// Using Cow generically
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
    // HTML escape
    let safe = html_escape("Hello World");       // Borrowed
    let escaped = html_escape("<script>alert('xss')</script>"); // Owned
    println!("Safe: {}", safe);
    println!("Escaped: {}", escaped);

    // Path normalization
    let path1 = normalize_path("/usr/local/bin");  // Borrowed
    let path2 = normalize_path("~/Documents");     // Owned
    println!("Path 1: {}", path1);
    println!("Path 2: {}", path2);

    // Generic processing
    let items = vec!["hello".to_string(), "prefix_world".to_string()];
    let processed = process_items(&items, "prefix_");
    for item in &processed {
        println!("  {}", item);
    }
}
```

---

## 7. Pin<P>

### 7.1 Overview of Pin

`Pin<P>` fixes the memory location of the data that pointer `P` points to. It is mainly used with self-referential types and async/await.

```
┌──────────────────────────────────────────────────────┐
│  Purpose of Pin<P>                                   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Problem: moving a self-referential type breaks      │
│  the internal references                             │
│                                                      │
│  Before move:                                        │
│  ┌──────────────────┐                                │
│  │ data: "hello"    │ <- ptr points to data          │
│  │ ptr: &data ──────┘                                │
│  └──────────────────┘                                │
│                                                      │
│  After move:                                         │
│  ┌──────────────────┐  <- new address                │
│  │ data: "hello"    │                                │
│  │ ptr: &old_data ──── -> dangling!                  │
│  └──────────────────┘                                │
│                                                      │
│  Using Pin: moves are prevented -> safe              │
└──────────────────────────────────────────────────────┘
```

### Example 17: Basic Use of Pin

```rust
use std::pin::Pin;
use std::marker::PhantomPinned;

struct SelfReferential {
    data: String,
    // Self-referential pointer (set after initialization)
    ptr: *const String,
    // Prevent the type from implementing Unpin
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

        // Set the self-referential pointer
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
    println!("ptr->data: {}", pinned.get_ptr_data());

    // pinned cannot be moved (it is fixed by Pin)
    // let moved = pinned; // compile error (because of PhantomPinned)
}
```

### Example 18: async/await and Pin

```rust
use std::pin::Pin;
use std::future::Future;

// async fn internally generates a self-referential structure
// Therefore Pin is required when handling Future directly

// Example of manually implementing the Future trait
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
            std::task::Poll::Ready("countdown complete!".to_string())
        } else {
            self.count -= 1;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}

// A function that requires Pin
fn execute_pinned<F: Future<Output = String>>(future: Pin<Box<F>>) {
    // In practice, the runtime calls poll
    println!("received the Future");
}

fn main() {
    let future = CountdownFuture { count: 3 };
    let pinned = Box::pin(future);
    execute_pinned(pinned);
}
```

---

## 8. Designing Custom Smart Pointers

### Example 19: Smart Pointer with Audit Logging

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
        println!("[audit] '{}': reads={}, writes={}", self.name, reads, writes);
    }
}

fn main() {
    {
        let mut data = Audited::new("important data", vec![1, 2, 3]);

        // Read accesses
        println!("len: {}", data.len());        // Deref -> read_count++
        println!("first: {}", data[0]);          // Deref -> read_count++

        // Write accesses
        data.push(4);                            // DerefMut -> write_count++
        data.push(5);                            // DerefMut -> write_count++

        let (reads, writes) = data.stats();
        println!("intermediate stats: reads={}, writes={}", reads, writes);
    }
    // Final stats are printed on drop
}
```

### Example 20: Pool-Managed Smart Pointer

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
        // Return to the pool on drop
        // Note: extract the value with std::mem::replace
        let value = unsafe {
            std::ptr::read(&self.value)
        };
        if let Ok(mut pool) = self.pool.lock() {
            pool.push(value);
            println!("returned to pool (pool size: {})", pool.len());
        }
        // Prevent value from being dropped
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
            println!("acquired from pool (remaining: {})", items.len());
            PoolItem {
                value,
                pool: Arc::clone(&self.items),
            }
        })
    }
}

fn main() {
    let pool = Pool::new(vec![
        String::from("connection 1"),
        String::from("connection 2"),
        String::from("connection 3"),
    ]);

    {
        let conn1 = pool.acquire().unwrap();
        let conn2 = pool.acquire().unwrap();
        println!("in use: {}, {}", *conn1, *conn2);
        // conn2 is dropped -> returned to the pool
    }
    // conn1 is also dropped -> returned to the pool

    // Reuse a returned connection
    let conn3 = pool.acquire().unwrap();
    println!("reused: {}", *conn3);
}
```

---

## 9. Comparison Tables

### 9.1 Smart Pointer Selection Guide

| Requirement | Type to Choose | Reason |
|------|-----------|------|
| Allocate on the heap | `Box<T>` | Simplest. Single ownership |
| Share within the same thread | `Rc<T>` | Shared ownership via reference counting |
| Share across threads (read only) | `Arc<T>` | Atomic reference counting |
| Share across threads (with mutation) | `Arc<Mutex<T>>` | Exclusive access via locking |
| Mutate through immutable references | `RefCell<T>` | Runtime borrow checking |
| Interior mutability for Copy types | `Cell<T>` | Swap values via get/set |
| Many readers, few writers (threads) | `Arc<RwLock<T>>` | Reads can run in parallel |
| Clone only on modification | `Cow<'a, T>` | Avoids unnecessary allocations |
| Prevent reference cycles | `Weak<T>` | Not counted in the reference count |
| Self-referential / async | `Pin<P>` | Pin the memory location |

### 9.2 Rc vs Arc

| Property | Rc<T> | Arc<T> |
|------|-------|--------|
| Thread-safe | No (does not implement Send) | Yes |
| Reference count operations | Plain inc/dec | Atomic operations |
| Overhead | Small | Slightly larger |
| Use case | Sharing within a single thread | Sharing across threads |
| Interior mutability | + RefCell | + Mutex / RwLock |

### 9.3 RefCell vs Mutex vs RwLock

| Property | RefCell<T> | Mutex<T> | RwLock<T> |
|------|-----------|----------|-----------|
| Thread-safe | No | Yes | Yes |
| Borrow checking | Runtime | Lock | Lock |
| Multiple readers | Possible via borrow() | Not possible | Possible via read() |
| Writer exclusivity | borrow_mut() | lock() | write() |
| Behavior on violation | Panic | Block | Block |
| Try operations | try_borrow() | try_lock() | try_read/try_write() |
| Poison | None | Yes | Yes |
| Overhead | Small | Moderate | Moderate |

### 9.4 Memory Layout Comparison

| Type | Stack Size | Heap Usage |
|----|---------------|-----------|
| `T` | `size_of::<T>()` | None |
| `Box<T>` | One pointer (8B) | `size_of::<T>()` |
| `Rc<T>` | One pointer (8B) | `size_of::<T>()` + counters (16B) |
| `Arc<T>` | One pointer (8B) | `size_of::<T>()` + atomic counters (16B) |
| `RefCell<T>` | `size_of::<T>()` + flag | None |
| `Cell<T>` | `size_of::<T>()` | None |

---

## 10. Anti-Patterns

### Anti-Pattern 1: Runtime Panic with RefCell

```rust
use std::cell::RefCell;

// BAD: simultaneous borrow and borrow_mut -> runtime panic
fn bad_example() {
    let data = RefCell::new(vec![1, 2, 3]);
    let r1 = data.borrow();     // immutable borrow
    // let r2 = data.borrow_mut(); // panic! mutable borrow during immutable borrow
    println!("{:?}", r1);
}

// GOOD: limit the scope of borrows
fn good_example() {
    let data = RefCell::new(vec![1, 2, 3]);
    {
        let r1 = data.borrow();
        println!("{:?}", r1);
    } // r1 is dropped
    let mut r2 = data.borrow_mut(); // OK
    r2.push(4);
}

fn main() {
    bad_example();
    good_example();
}
```

### Anti-Pattern 2: Unnecessary Arc<Mutex<T>>

```rust
use std::sync::{Arc, Mutex};
use std::cell::RefCell;

// BAD: using Arc<Mutex> for single-threaded code
fn single_thread_bad() {
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));
    let mut guard = data.lock().unwrap();
    guard.push(4);
}

// GOOD: RefCell suffices for single-threaded code
fn single_thread_good() {
    let data = RefCell::new(vec![1, 2, 3]);
    data.borrow_mut().push(4);
}

fn main() {
    single_thread_bad();
    single_thread_good();
}
```

### Anti-Pattern 3: Memory Leaks from Rc Reference Cycles

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
    // Create a reference cycle!
    *a.next.borrow_mut() = Some(Rc::clone(&b));

    // Cycle: a -> b -> a -> b -> ...
    // Reference count never reaches 0, causing a memory leak
    println!("Reference count of a: {}", Rc::strong_count(&a)); // 2
    println!("Reference count of b: {}", Rc::strong_count(&b)); // 2
}
// Even after a and b are dropped, they reference each other
// so the reference count stays at 1 -> memory leak

// GOOD: prevent the cycle with Weak
use std::rc::Weak;

#[derive(Debug)]
struct GoodNode {
    value: i32,
    next: RefCell<Option<Rc<GoodNode>>>,
    prev: RefCell<Option<Weak<GoodNode>>>,  // weak reference
}

fn main() {
    demonstrate_leak();
    println!("Reference cycle demo complete (memory leak occurred)");
}
```

### Anti-Pattern 4: Unnecessary Use of Box

```rust
// BAD: no need to wrap a small value in a Box
fn bad_box() {
    let x = Box::new(42);  // overhead of heap allocation
    println!("{}", x);
}

// GOOD: place it directly on the stack
fn good_stack() {
    let x = 42;
    println!("{}", x);
}

// Cases where Box is needed
fn good_box_usage() {
    // 1. Recursive types
    enum List<T> {
        Cons(T, Box<List<T>>),
        Nil,
    }

    // 2. Trait objects
    let _: Box<dyn std::fmt::Display> = Box::new(42);

    // 3. Large data
    let _: Box<[u8; 1_000_000]> = Box::new([0u8; 1_000_000]);
}

fn main() {
    bad_box();
    good_stack();
    good_box_usage();
}
```


---

## Hands-On Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate the input data
- Implement appropriate error handling
- Also write test code

```python
# Exercise 1: Template for the basic implementation
class Exercise1:
    """Practice for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
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
        assert False, "an exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Practice for advanced patterns"""

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
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Delete by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
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
    assert ex.add("d", 4) == False  # size limit
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
# Exercise 3: Performance optimization
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

    print(f"Inefficient version: {slow_time:.4f} sec")
    print(f"Efficient version:   {fast_time:.6f} sec")
    print(f"Speedup factor: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Misconfiguration | Check the path and format of the configuration file |
| Timeout | Network latency / lack of resources | Adjust timeout values, add retry logic |
| Out of memory | Growing data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check the executing user's privileges, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check the error message**: read the stack trace and identify where it occurs
2. **Establish a reproduction procedure**: reproduce the error with minimal code
3. **Formulate hypotheses**: list possible causes
4. **Verify step by step**: validate the hypotheses with logging or a debugger
5. **Fix and regression test**: after fixing, also run tests for related areas

```python
# Debugging utilities
import logging
import traceback
from functools import wraps

# Logger setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs the inputs and outputs of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Problems

Procedure for diagnosing performance issues:

1. **Identify the bottleneck**: measure with profiling tools
2. **Check memory usage**: detect any memory leaks
3. **Check for I/O waits**: examine disk and network I/O conditions
4. **Check the number of concurrent connections**: examine the state of connection pools

| Type of issue | Diagnostic tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithmic improvements, parallelization |
| Memory leak | tracemalloc, objgraph | Properly release references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexes, query optimization |
---

## 11. FAQ

### Q1: When should I use Box<T>?

**A:** Mainly in the following situations:
- **Recursive data structures** (lists, trees, etc.): the size cannot be determined at compile time
- **Avoiding moves of large data**: place on the heap rather than the stack
- **Trait objects**: dynamic dispatch via `Box<dyn Trait>`
- **Transferring ownership**: place on the heap and only move the pointer

### Q2: How do I prevent reference cycles with Rc?

**A:** Use `Weak<T>`. Create a weak reference with Rc::downgrade() to prevent cycles:
```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;
struct Node {
    parent: RefCell<Weak<Node>>,   // weak reference -> prevents cycles
    children: RefCell<Vec<Rc<Node>>>, // strong references
}
```
Because Weak does not contribute to the reference count, memory is properly freed even if cycles exist.

### Q3: What is Pin<T> for?

**A:** Pin fixes a value's memory location and prevents it from being moved. It is mainly required for:
- **async/await**: Future values contain self-referential structures, so moving them would break references
- **Self-referential types**: when a reference inside a struct points to one of its own fields
- In most cases you use it via `Box::pin()` or the `pin!()` macro.

### Q4: What is Mutex "Poison"?

**A:** If a thread holding a lock panics, the Mutex enters a "poisoned" state. When other threads call `lock()`, they receive a `PoisonError`. This is a safety mechanism to signal that the data may be in an inconsistent state due to the panic:
```rust
use std::sync::Mutex;
let m = Mutex::new(42);

// Using the value while ignoring poisoning
let value = m.lock().unwrap_or_else(|e| e.into_inner());
```

### Q5: How do I choose between Cell<T> and RefCell<T>?

**A:**
- `Cell<T>` is used for types where `T: Copy`. Only get/set; you cannot obtain a reference. Has the smallest overhead
- `RefCell<T>` is used for any type. Acquire references with `borrow()` / `borrow_mut()`. It enforces borrowing rules at runtime
- Prefer `Cell<T>` when possible. Use `RefCell<T>` when references are required

### Q6: What are the criteria for choosing between Arc<Mutex<T>> and Arc<RwLock<T>>?

**A:**
- **Arc<Mutex<T>>**: when reads and writes occur at similar frequencies. Implementation is simple. Lower risk of deadlocks
- **Arc<RwLock<T>>**: when reads vastly outnumber writes. Reads can run in parallel for higher throughput
- When in doubt, start with `Mutex`, and if profiling shows reads are the bottleneck, migrate to `RwLock`

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining hands-on experience is the most important. Beyond theory, writing actual code and observing its behavior deepens your understanding.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

The knowledge of this topic is frequently used in everyday development work. It is especially important during code reviews and architectural design.

---

## 12. Summary

| Concept | Key Points |
|------|------|
| Box<T> | Heap allocation. Single ownership. Essential for recursive types and dyn Trait |
| Rc<T> | Shared ownership via reference counting. Single-threaded only |
| Arc<T> | Atomic reference counting. Multi-threaded support |
| Weak<T> | Weak reference. Prevents reference cycles. Not counted toward the count |
| RefCell<T> | Runtime borrow checking. Interior mutability pattern |
| Cell<T> | Interior mutability for Copy types. Only get/set |
| Mutex<T> | Exclusive lock. Combine with Arc for cross-thread sharing |
| RwLock<T> | Read/write lock. Ideal for many readers, few writers |
| Cow<T> | Clones only on modification. Avoids unnecessary allocations |
| Pin<P> | Pins the memory location. Essential for async/await |
| Deref/Drop | The foundational traits of smart pointers |

---

## Recommended Next Reads

- [02-closures-fn-traits.md](02-closures-fn-traits.md) -- Closures and Fn traits
- [../03-systems/01-concurrency.md](../03-systems/01-concurrency.md) -- Concurrent programming explained
- [03-unsafe-rust.md](03-unsafe-rust.md) -- unsafe and raw pointers

---

## References

1. **The Rust Programming Language - Ch.15 Smart Pointers** -- https://doc.rust-lang.org/book/ch15-00-smart-pointers.html
2. **The Rustonomicon - Concurrency** -- https://doc.rust-lang.org/nomicon/concurrency.html
3. **Rust std::cell Module** -- https://doc.rust-lang.org/std/cell/
4. **Rust std::sync Module** -- https://doc.rust-lang.org/std/sync/
5. **Rust std::pin Module** -- https://doc.rust-lang.org/std/pin/
6. **Rust std::borrow::Cow** -- https://doc.rust-lang.org/std/borrow/enum.Cow.html
