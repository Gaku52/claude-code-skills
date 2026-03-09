# 第一級関数（First-Class Functions）

> 関数が「値」として扱える言語では、関数を変数に代入し、引数として渡し、戻り値として返すことができる。これはモダンプログラミングの基盤であり、関数型プログラミングの出発点でもある。Christopher Strachey が1967年に提唱した「第一級（first-class）」の概念は、半世紀以上を経て、ほぼ全ての主要言語に取り入れられるに至った。

---

## この章で学ぶこと

- [ ] 第一級関数の概念と歴史的意義を理解する
- [ ] 関数を値として操作する4つの方法を習得する
- [ ] コールバックパターンと高階関数の設計を理解する
- [ ] 関数合成・部分適用・カリー化の技法を身につける
- [ ] 言語間の第一級関数サポートの差異を比較できる
- [ ] ディスパッチテーブルやストラテジーパターンを実装できる
- [ ] アンチパターンを識別し、適切に回避できる


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## 1. 第一級関数とは何か

### 1.1 定義と歴史的背景

「第一級（First-Class）」という用語は、英国の計算機科学者 Christopher Strachey が1967年の講義ノート "Fundamental Concepts in Programming Languages" で初めて体系的に用いた。Strachey は、プログラミング言語における「値」の地位を以下のように分類した。

```
┌──────────────────────────────────────────────────────────────────┐
│               Strachey の値の分類体系（1967）                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  第一級（First-Class）                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ・変数に束縛（代入）できる                                 │   │
│  │ ・関数の引数として渡せる                                   │   │
│  │ ・関数の戻り値として返せる                                 │   │
│  │ ・データ構造（配列・リスト等）に格納できる                 │   │
│  │ ・実行時に動的に生成できる                                 │   │
│  │ ・固有のアイデンティティを持つ                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  第二級（Second-Class）                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ・関数の引数としては渡せる                                 │   │
│  │ ・変数に代入できない、戻り値にできない場合がある           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  第三級（Third-Class）                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ・言語の構文要素としてのみ存在                             │   │
│  │ ・引数に渡すことすらできない                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

この分類において、多くの言語では整数・文字列・配列などは第一級だが、「関数」を第一級として扱える言語は限られていた。LISP（1958年）は関数を第一級として扱う最初の実用言語であり、その設計思想は Scheme、ML、Haskell を経て、JavaScript、Python、Ruby、そして近年では Rust、Go、Kotlin、Swift に至る現代言語に継承されている。

### 1.2 第一級関数の4つの性質

第一級関数とは「他の値と全く同等に扱える関数」を意味する。具体的には以下の4つの操作が可能であることが要件となる。

```
第一級関数の4つの性質
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [性質 1] 変数への代入
  ┌─────────────────────────────────────┐
  │  const f = function(x) { return x; } │
  │  val    ← ──── 関数値 ──────────→   │
  └─────────────────────────────────────┘

  [性質 2] 引数としての受け渡し
  ┌─────────────────────────────────────┐
  │  apply(f, 42)                        │
  │         ↑                            │
  │       関数を引数として渡す            │
  └─────────────────────────────────────┘

  [性質 3] 戻り値としての返却
  ┌─────────────────────────────────────┐
  │  function make() { return f; }       │
  │                          ↑           │
  │                   関数を返す          │
  └─────────────────────────────────────┘

  [性質 4] データ構造への格納
  ┌─────────────────────────────────────┐
  │  const arr = [f1, f2, f3];           │
  │  const obj = { op: f1 };             │
  │     配列・辞書に関数を入れる          │
  └─────────────────────────────────────┘
```

### 1.3 なぜ第一級関数が重要なのか

第一級関数は以下の理由からモダンプログラミングの基盤とされる。

1. **抽象化の強化**: 処理のパターン（反復・変換・選択）を関数として抽象化し、再利用可能にする
2. **コードの簡潔化**: ボイラープレートコードを排除し、意図を直接表現できる
3. **柔軟な設計**: 振る舞いを実行時に差し替え可能にする（ストラテジーパターン等）
4. **並行処理との親和性**: 副作用のない関数はスレッドセーフであり、並行処理に適する
5. **テスト容易性**: 関数単位でのテストが容易になり、モックの差し替えが自然に行える

---

## 2. 基本操作：コード例で学ぶ4つの性質

### 2.1 性質1 — 変数への代入

関数を変数に代入することで、関数に別名を付けたり、条件に応じて異なる関数を選択できる。

```javascript
// ===== JavaScript =====

// 関数宣言を変数に代入（関数式）
const greet = function(name) {
    return `Hello, ${name}!`;
};

// アロー関数（ES6+）
const greetArrow = (name) => `Hello, ${name}!`;

// 既存の関数を別の変数に代入
const sayHello = greet;
console.log(sayHello("Alice"));  // => "Hello, Alice!"

// 条件に応じた関数の選択
const formatter = process.env.NODE_ENV === "production"
    ? (msg) => `[PROD] ${msg}`
    : (msg) => `[DEV] ${msg}`;

console.log(formatter("Server started"));
// 開発環境: => "[DEV] Server started"
```

```python
# ===== Python =====

def square(x):
    """xの二乗を返す"""
    return x ** 2

# 関数を変数に代入
f = square
print(f(5))      # => 25
print(f.__name__) # => "square"（元の名前を保持）

# lambda式による無名関数
double = lambda x: x * 2
print(double(7))  # => 14

# 条件に応じた関数の選択
import os
log_fn = print if os.getenv("DEBUG") else lambda *args: None
log_fn("デバッグメッセージ")  # DEBUG環境変数がないと何も表示しない
```

```rust
// ===== Rust =====

fn square(x: i32) -> i32 {
    x * x
}

fn main() {
    // 関数ポインタとして代入
    let f: fn(i32) -> i32 = square;
    println!("{}", f(5));  // => 25

    // クロージャを変数に代入
    let double = |x: i32| -> i32 { x * 2 };
    println!("{}", double(7));  // => 14

    // 条件に応じた関数の選択（関数ポインタの場合）
    let debug = true;
    let log_fn: fn(&str) = if debug {
        |msg| println!("[DEBUG] {}", msg)
    } else {
        |_msg| {}  // 何もしない
    };
    log_fn("起動しました");
}
```

### 2.2 性質2 — 引数としての受け渡し（高階関数）

関数を引数として受け取る関数を **高階関数（Higher-Order Function）** と呼ぶ。これは第一級関数の最も強力な応用の一つである。

```javascript
// ===== JavaScript =====

// map: 各要素に変換関数を適用
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(x => x * 2);
// => [2, 4, 6, 8, 10]

// filter: 条件関数を満たす要素のみ抽出
const evens = numbers.filter(x => x % 2 === 0);
// => [2, 4]

// reduce: 累積関数で畳み込み
const sum = numbers.reduce((acc, x) => acc + x, 0);
// => 15

// カスタム高階関数の作成
function applyTwice(fn, value) {
    return fn(fn(value));
}

applyTwice(x => x * 2, 3);   // => 12  (3 → 6 → 12)
applyTwice(x => x + 10, 5);  // => 25  (5 → 15 → 25)

// 汎用的な retry 関数
async function retry(fn, maxAttempts = 3, delay = 1000) {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn();
        } catch (err) {
            if (attempt === maxAttempts) throw err;
            console.log(`Attempt ${attempt} failed, retrying...`);
            await new Promise(r => setTimeout(r, delay));
        }
    }
}

// 使用例
await retry(() => fetch("https://api.example.com/data"), 3, 2000);
```

```python
# ===== Python =====

# 組み込み高階関数
numbers = [1, 2, 3, 4, 5]

# map: 各要素に関数を適用
squared = list(map(lambda x: x ** 2, numbers))
# => [1, 4, 9, 16, 25]

# filter: 条件を満たす要素を抽出
evens = list(filter(lambda x: x % 2 == 0, numbers))
# => [2, 4]

# sorted: キー関数でソート
words = ["banana", "apple", "cherry", "date"]
sorted_by_length = sorted(words, key=len)
# => ["date", "apple", "banana", "cherry"]

# カスタム高階関数
def apply_to_all(fn, items):
    """リスト全要素にfnを適用して新リストを返す"""
    return [fn(item) for item in items]

apply_to_all(str.upper, ["hello", "world"])
# => ["HELLO", "WORLD"]

# デコレータ: 高階関数の典型的応用
import time
import functools

def timer(func):
    """関数の実行時間を計測するデコレータ"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "done"

slow_function()  # => "slow_function: 1.00xx秒" と表示後 "done" を返す
```

### 2.3 性質3 — 戻り値としての返却（関数ファクトリ）

関数を返す関数は **関数ファクトリ** または **関数ジェネレータ** と呼ばれる。これにより、設定済みの関数を動的に生成できる。

```javascript
// ===== JavaScript =====

// 乗数関数ファクトリ
function multiplier(factor) {
    return (x) => x * factor;
}

const double = multiplier(2);
const triple = multiplier(3);
const tenTimes = multiplier(10);

console.log(double(5));    // => 10
console.log(triple(5));    // => 15
console.log(tenTimes(5));  // => 50

// バリデータファクトリ
function createValidator(rules) {
    return (value) => {
        const errors = [];
        for (const rule of rules) {
            const error = rule(value);
            if (error) errors.push(error);
        }
        return { valid: errors.length === 0, errors };
    };
}

// ルール定義
const required = (v) => v ? null : "必須項目です";
const minLength = (n) => (v) => v && v.length >= n
    ? null : `${n}文字以上必要です`;
const pattern = (re, msg) => (v) => re.test(v) ? null : msg;

// バリデータ生成
const validateEmail = createValidator([
    required,
    minLength(5),
    pattern(/^[^\s@]+@[^\s@]+\.[^\s@]+$/, "有効なメールアドレスを入力してください"),
]);

console.log(validateEmail(""));
// => { valid: false, errors: ["必須項目です", "5文字以上必要です", ...] }

console.log(validateEmail("user@example.com"));
// => { valid: true, errors: [] }
```

```python
# ===== Python =====

# ロガーファクトリ
def create_logger(prefix, level="INFO"):
    """指定プレフィックスとレベルのログ関数を生成する"""
    def logger(message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] [{prefix}] {message}")
    return logger

app_log = create_logger("APP")
db_log = create_logger("DB", level="DEBUG")

app_log("サーバー起動")
# => [2026-03-06 10:30:00] [INFO] [APP] サーバー起動

db_log("クエリ実行: SELECT * FROM users")
# => [2026-03-06 10:30:00] [DEBUG] [DB] クエリ実行: SELECT * FROM users
```

### 2.4 性質4 — データ構造への格納（ディスパッチテーブル）

関数を辞書や配列に格納する「ディスパッチテーブル」パターンは、長い if-else チェーンや switch 文を置き換える強力な手法である。

```python
# ===== Python =====

# ディスパッチテーブルによるコマンドパターン
class Calculator:
    def __init__(self):
        self.operations = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b if b != 0 else float("inf"),
            "**": lambda a, b: a ** b,
            "%": lambda a, b: a % b,
        }
        self.history = []

    def calculate(self, a, op, b):
        if op not in self.operations:
            raise ValueError(f"未対応の演算子: {op}")
        result = self.operationsop
        self.history.append(f"{a} {op} {b} = {result}")
        return result

    def add_operation(self, symbol, fn):
        """演算子を動的に追加"""
        self.operations[symbol] = fn

calc = Calculator()
print(calc.calculate(10, "+", 5))   # => 15
print(calc.calculate(2, "**", 10))  # => 1024

# 演算子の動的追加
calc.add_operation("avg", lambda a, b: (a + b) / 2)
print(calc.calculate(10, "avg", 20))  # => 15.0
```

```javascript
// ===== JavaScript =====

// HTTP メソッドのディスパッチテーブル
const handlers = {
    GET:    (req) => ({ status: 200, body: fetchResource(req.path) }),
    POST:   (req) => ({ status: 201, body: createResource(req.body) }),
    PUT:    (req) => ({ status: 200, body: updateResource(req.path, req.body) }),
    DELETE: (req) => ({ status: 204, body: null }),
};

function handleRequest(req) {
    const handler = handlers[req.method];
    if (!handler) {
        return { status: 405, body: "Method Not Allowed" };
    }
    return handler(req);
}

// パイプライン: 関数の配列を順に適用
const pipeline = [
    (data) => ({ ...data, timestamp: Date.now() }),
    (data) => ({ ...data, id: crypto.randomUUID() }),
    (data) => ({ ...data, status: "processed" }),
];

function processThroughPipeline(data, steps) {
    return steps.reduce((acc, step) => step(acc), data);
}

const result = processThroughPipeline(
    { name: "Alice" },
    pipeline
);
// => { name: "Alice", timestamp: 1709..., id: "abc-...", status: "processed" }
```

---

## 3. 高階関数の体系的理解

### 3.1 高階関数の分類

高階関数は「関数を受け取る」ものと「関数を返す」ものに大別される。多くの関数は両方の性質を持つ。

```
高階関数の分類体系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                        高階関数 (Higher-Order Function)
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
        関数を受け取る                      関数を返す
     (Consumer of Functions)          (Producer of Functions)
                 │                               │
     ┌───────────┼──────────┐       ┌────────────┼───────────┐
     │           │          │       │            │           │
   map/       filter/    reduce/  ファクトリ    カリー化    デコレータ
   forEach   find/some  fold                  /部分適用
     │           │          │       │            │           │
  変換系     選択系     集約系    生成系       変換系      修飾系

  例: .map()  例: .filter() 例: .reduce() 例: multiplier() 例: curry() 例: @timer
  各要素を    条件で       蓄積して  設定済み関数   引数を1つ  関数の前後
  変換する    絞り込む     1値に集約 を動的生成    ずつ受取る  に処理追加
```

### 3.2 代表的な高階関数のデータフロー

```
map のデータフロー
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
入力: [1, 2, 3, 4, 5]
関数: x => x * x

  1 ──→ [x => x*x] ──→  1
  2 ──→ [x => x*x] ──→  4
  3 ──→ [x => x*x] ──→  9
  4 ──→ [x => x*x] ──→ 16
  5 ──→ [x => x*x] ──→ 25

出力: [1, 4, 9, 16, 25]


filter のデータフロー
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
入力: [1, 2, 3, 4, 5]
述語: x => x % 2 === 0

  1 ──→ [x%2===0] ──→ false ──→ (除外)
  2 ──→ [x%2===0] ──→ true  ──→ 2
  3 ──→ [x%2===0] ──→ false ──→ (除外)
  4 ──→ [x%2===0] ──→ true  ──→ 4
  5 ──→ [x%2===0] ──→ false ──→ (除外)

出力: [2, 4]


reduce のデータフロー
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
入力: [1, 2, 3, 4, 5]
関数: (acc, x) => acc + x
初期値: 0

  acc=0, x=1 ──→ [acc+x] ──→ acc=1
  acc=1, x=2 ──→ [acc+x] ──→ acc=3
  acc=3, x=3 ──→ [acc+x] ──→ acc=6
  acc=6, x=4 ──→ [acc+x] ──→ acc=10
  acc=10,x=5 ──→ [acc+x] ──→ acc=15

出力: 15
```

### 3.3 map / filter / reduce の実装から学ぶ

標準ライブラリが提供する高階関数を自ら実装することで、その仕組みを深く理解できる。

```javascript
// ===== JavaScript: map / filter / reduce の自作実装 =====

// map の実装
function myMap(arr, fn) {
    const result = [];
    for (let i = 0; i < arr.length; i++) {
        result.push(fn(arr[i], i, arr));
    }
    return result;
}

// filter の実装
function myFilter(arr, predicate) {
    const result = [];
    for (let i = 0; i < arr.length; i++) {
        if (predicate(arr[i], i, arr)) {
            result.push(arr[i]);
        }
    }
    return result;
}

// reduce の実装
function myReduce(arr, fn, initial) {
    let acc = initial;
    let startIndex = 0;
    if (acc === undefined) {
        acc = arr[0];
        startIndex = 1;
    }
    for (let i = startIndex; i < arr.length; i++) {
        acc = fn(acc, arr[i], i, arr);
    }
    return acc;
}

// 動作確認
const nums = [1, 2, 3, 4, 5];
console.log(myMap(nums, x => x * 2));           // => [2, 4, 6, 8, 10]
console.log(myFilter(nums, x => x > 3));        // => [4, 5]
console.log(myReduce(nums, (a, b) => a + b, 0)); // => 15

// reduce で map と filter を再実装
function mapWithReduce(arr, fn) {
    return arr.reduce((acc, x, i) => {
        acc.push(fn(x, i, arr));
        return acc;
    }, []);
}

function filterWithReduce(arr, pred) {
    return arr.reduce((acc, x, i) => {
        if (pred(x, i, arr)) acc.push(x);
        return acc;
    }, []);
}
```

---

## 4. コールバックパターンの深掘り

### 4.1 同期コールバック

```javascript
// ===== JavaScript: 同期コールバックの多様な用例 =====

// イベントリスナー（ブラウザ環境）
document.addEventListener("click", (event) => {
    console.log(`クリック位置: (${event.clientX}, ${event.clientY})`);
});

// Array メソッドチェーン
const users = [
    { name: "Alice", age: 30, role: "admin" },
    { name: "Bob", age: 25, role: "user" },
    { name: "Charlie", age: 35, role: "admin" },
    { name: "Diana", age: 28, role: "user" },
];

// 管理者の名前を年齢順に取得
const adminNames = users
    .filter(u => u.role === "admin")
    .sort((a, b) => a.age - b.age)
    .map(u => u.name);
// => ["Alice", "Charlie"]

// forEach: 副作用のためのコールバック
users.forEach(u => {
    console.log(`${u.name} (${u.age}歳) - ${u.role}`);
});

// find / findIndex: 条件に合う最初の要素
const firstAdmin = users.find(u => u.role === "admin");
// => { name: "Alice", age: 30, role: "admin" }

// every / some: 全要素/一部要素が条件を満たすか
const allAdults = users.every(u => u.age >= 18);  // => true
const hasAdmin = users.some(u => u.role === "admin"); // => true
```

### 4.2 非同期コールバック

```javascript
// ===== JavaScript: 非同期コールバックの進化 =====

// 1. 古典的コールバックスタイル（Node.js）
const fs = require("fs");

fs.readFile("/path/to/file.txt", "utf8", (err, data) => {
    if (err) {
        console.error("読み取りエラー:", err);
        return;
    }
    console.log("ファイル内容:", data);
});

// 2. Promise（ES6+）
function readFileAsync(path) {
    return new Promise((resolve, reject) => {
        fs.readFile(path, "utf8", (err, data) => {
            if (err) reject(err);
            else resolve(data);
        });
    });
}

readFileAsync("/path/to/file.txt")
    .then(data => console.log(data))
    .catch(err => console.error(err));

// 3. async/await（ES2017+）
async function processFile() {
    try {
        const data = await readFileAsync("/path/to/file.txt");
        const processed = data.toUpperCase();
        console.log(processed);
    } catch (err) {
        console.error("処理エラー:", err);
    }
}
```

### 4.3 コールバック地獄と解決策

```
コールバック地獄（Callback Hell）の視覚化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  step1(input, (err, result1) => {
      if (err) handleError(err);
      step2(result1, (err, result2) => {
          if (err) handleError(err);
          step3(result2, (err, result3) => {
              if (err) handleError(err);
              step4(result3, (err, result4) => {    ← 深くネスト！
                  if (err) handleError(err);
                  // ...さらにネストが続く
              });
          });
      });
  });

  解決策の進化:
  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
  │  コールバック    │    │   Promise      │    │  async/await   │
  │  (ES5以前)      │ →  │  .then()チェーン│ →  │  同期風記述    │
  │                │    │  (ES6+)        │    │  (ES2017+)     │
  │  ネスト地獄     │    │  フラットな     │    │  最も読みやすい │
  │  エラー処理散在  │    │  チェーン       │    │  try/catch     │
  └────────────────┘    └────────────────┘    └────────────────┘
```

---

## 5. ストラテジーパターンと関数

### 5.1 従来のOOP実装 vs 第一級関数

ストラテジーパターンは GoF デザインパターンの一つで、アルゴリズムを動的に切り替える設計パターンである。OOP では専用のクラス階層が必要だが、第一級関数を持つ言語では関数一つで実現できる。

```typescript
// ===== TypeScript: OOP的ストラテジー vs 関数的ストラテジー =====

// --- OOP的アプローチ（Javaスタイル）---
interface SortStrategy<T> {
    compare(a: T, b: T): number;
}

class NameSortStrategy implements SortStrategy<User> {
    compare(a: User, b: User): number {
        return a.name.localeCompare(b.name);
    }
}

class AgeSortStrategy implements SortStrategy<User> {
    compare(a: User, b: User): number {
        return a.age - b.age;
    }
}

class UserSorter {
    private strategy: SortStrategy<User>;

    constructor(strategy: SortStrategy<User>) {
        this.strategy = strategy;
    }

    setStrategy(strategy: SortStrategy<User>) {
        this.strategy = strategy;
    }

    sort(users: User[]): User[] {
        return [...users].sort((a, b) => this.strategy.compare(a, b));
    }
}

// --- 関数的アプローチ（第一級関数を活用）---
type Comparator<T> = (a: T, b: T) => number;

const byName: Comparator<User> = (a, b) => a.name.localeCompare(b.name);
const byAge: Comparator<User> = (a, b) => a.age - b.age;
const byNameDesc: Comparator<User> = (a, b) => b.name.localeCompare(a.name);
const byAgeDesc: Comparator<User> = (a, b) => b.age - a.age;

// 合成: 複数のソート条件を組み合わせ
function composeComparators<T>(...comparators: Comparator<T>[]): Comparator<T> {
    return (a, b) => {
        for (const cmp of comparators) {
            const result = cmp(a, b);
            if (result !== 0) return result;
        }
        return 0;
    };
}

// ロール優先 → 年齢順でソート
const byRoleThenAge = composeComparators(
    (a, b) => a.role.localeCompare(b.role),
    byAge
);

const sorted = [...users].sort(byRoleThenAge);
```

### 5.2 比較表: OOP vs 関数的ストラテジー

| 観点 | OOP（クラスベース） | 関数的アプローチ |
|------|---------------------|------------------|
| コード量 | インターフェース + 実装クラスが必要 | 関数1つで完結 |
| 新戦略の追加 | 新クラスの作成が必要 | 新関数の定義のみ |
| 状態の保持 | インスタンス変数で保持可能 | クロージャで保持可能 |
| 型安全性 | インターフェースで強制 | 型エイリアスで表現 |
| テスト | モックオブジェクトが必要な場合あり | 関数単体でテスト可能 |
| 合成 | Compositeパターンが必要 | 関数合成で自然に実現 |
| 直列化 | クラスのシリアライズが必要 | 関数は直列化不可 |
| デバッグ | クラス名で特定しやすい | 無名関数は追跡しにくい |
| 適用場面 | 複雑な状態・ライフサイクル管理 | 単純な振る舞いの差し替え |

---

## 6. 関数合成と部分適用

### 6.1 関数合成（Function Composition）

二つの関数 f と g を組み合わせ、g の出力を f の入力とする新しい関数 f . g を作る操作を関数合成という。数学的には (f . g)(x) = f(g(x)) と表記される。

```
関数合成の概念図
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  compose(f, g) = x => f(g(x))

                 g            f
  入力 x ──→ [double] ──→ [addOne] ──→ 出力
          x=5    10          11

  pipe(g, f) = x => f(g(x))   ※左から右へ読める

                 g            f
  入力 x ──→ [double] ──→ [addOne] ──→ 出力
          x=5    10          11

  compose: 数学的記法（右から左）
  pipe:    プログラミング的記法（左から右）
```

```typescript
// ===== TypeScript: 関数合成の実装 =====

// compose: 右から左へ適用（数学的記法）
function compose<A, B, C>(
    f: (b: B) => C,
    g: (a: A) => B
): (a: A) => C {
    return (a: A) => f(g(a));
}

// pipe: 左から右へ適用（プログラミング的記法）
function pipe<T>(...fns: Array<(arg: T) => T>): (arg: T) => T {
    return (arg: T) => fns.reduce((acc, fn) => fn(acc), arg);
}

// 使用例
const double = (x: number) => x * 2;
const addOne = (x: number) => x + 1;
const square = (x: number) => x * x;

const doubleAndAddOne = compose(addOne, double);
console.log(doubleAndAddOne(5));  // => 11

const transform = pipe(double, addOne, square);
console.log(transform(3));  // => 3 → 6 → 7 → 49
```

```python
# ===== Python: 関数合成 =====

from functools import reduce

def compose(*fns):
    """右から左への関数合成"""
    def composed(x):
        result = x
        for fn in reversed(fns):
            result = fn(result)
        return result
    return composed

def pipe(*fns):
    """左から右への関数合成"""
    def piped(x):
        result = x
        for fn in fns:
            result = fn(result)
        return result
    return piped

# 使用例
double = lambda x: x * 2
add_one = lambda x: x + 1
square = lambda x: x ** 2

transform = pipe(double, add_one, square)
print(transform(3))  # => 49  (3 → 6 → 7 → 49)

# 文字列処理パイプライン
normalize = pipe(
    str.strip,
    str.lower,
    lambda s: s.replace("  ", " "),
)
print(normalize("  Hello   World  "))  # => "hello world"
```

### 6.2 部分適用（Partial Application）

```python
# ===== Python: functools.partial =====

from functools import partial

def power(base, exponent):
    """baseのexponent乗を計算"""
    return base ** exponent

# 部分適用で新しい関数を生成
square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # => 25
print(cube(3))    # => 27

# HTTPクライアントの部分適用
import urllib.request

def fetch(method, url, headers=None, body=None):
    """汎用HTTPリクエスト関数"""
    req = urllib.request.Request(url, method=method, headers=headers or {})
    if body:
        req.data = body.encode()
    return urllib.request.urlopen(req)

# メソッド固定の関数を生成
get = partial(fetch, "GET")
post = partial(fetch, "POST")
put = partial(fetch, "PUT")

# 使用: get("https://api.example.com/users")
```

### 6.3 カリー化（Currying）

カリー化は、n個の引数を取る関数を、1個の引数を取る関数のチェーンに変換する技法である。部分適用とは異なり、常に1引数ずつ適用する点が特徴。

```javascript
// ===== JavaScript: カリー化 =====

// 手動カリー化
function add(a) {
    return function(b) {
        return a + b;
    };
}

const add5 = add(5);
console.log(add5(3));  // => 8

// 汎用カリー化関数
function curry(fn) {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        }
        return function(...moreArgs) {
            return curried.apply(this, args.concat(moreArgs));
        };
    };
}

// 使用例
const curriedAdd = curry((a, b, c) => a + b + c);
console.log(curriedAdd(1)(2)(3));    // => 6
console.log(curriedAdd(1, 2)(3));    // => 6
console.log(curriedAdd(1)(2, 3));    // => 6
console.log(curriedAdd(1, 2, 3));    // => 6

// 実用例: ログ関数のカリー化
const log = curry((level, module, message) => {
    console.log(`[${level}] [${module}] ${message}`);
});

const errorLog = log("ERROR");
const appError = errorLog("APP");
const dbError = errorLog("DB");

appError("接続タイムアウト");
// => [ERROR] [APP] 接続タイムアウト

dbError("クエリ失敗");
// => [ERROR] [DB] クエリ失敗
```

---

## 7. 言語間の第一級関数サポート比較

### 7.1 総合比較表

各言語がどの程度「第一級関数」をサポートしているかを、主要な観点から比較する。

| 機能 / 言語 | JavaScript | Python | Rust | Go | Java | C | Haskell |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 変数への代入 | Yes | Yes | Yes | Yes | 制限あり | ポインタのみ | Yes |
| 引数として渡す | Yes | Yes | Yes | Yes | SAM型 | ポインタのみ | Yes |
| 戻り値として返す | Yes | Yes | Yes | Yes | SAM型 | ポインタのみ | Yes |
| データ構造に格納 | Yes | Yes | Yes | Yes | 制限あり | ポインタのみ | Yes |
| 無名関数 (ラムダ) | Yes | 式のみ | Yes | Yes | 式のみ | No | Yes |
| クロージャ | Yes | Yes | 3種類 | Yes | 制限あり | No | Yes |
| 部分適用 | 手動/ライブラリ | functools | 手動 | 手動 | 手動 | No | 自然 |
| カリー化 | 手動/ライブラリ | 手動 | 手動 | 手動 | No | No | 自動 |
| 関数合成演算子 | No | No | No | No | No | No | Yes (.) |
| 型推論 | 動的型付 | 動的型付 | Yes | Yes | 制限あり | No | Yes |
| ジェネリック高階関数 | Yes (動的) | Yes (動的) | Yes | Yes | Yes | No | Yes |

### 7.2 言語別の詳細特性

```
言語別の第一級関数サポート特性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  JavaScript (ES6+)
  ┌────────────────────────────────────────────────────────────────┐
  │ ・関数は Object のサブタイプ（typeof fn === "function"）       │
  │ ・function 宣言、function 式、アロー関数の3形式               │
  │ ・アロー関数は this を束縛しない（レキシカル this）            │
  │ ・全てのオブジェクトメソッドも関数値                          │
  │ ・非同期: async/await も第一級関数として扱える                │
  └────────────────────────────────────────────────────────────────┘

  Python
  ┌────────────────────────────────────────────────────────────────┐
  │ ・関数は object（type(fn) → <class 'function'>）             │
  │ ・def 文（複数行）と lambda 式（単一式のみ）の2形式           │
  │ ・デコレータ（@）が高階関数の糖衣構文                        │
  │ ・functools モジュールで partial, reduce, lru_cache 等を提供   │
  │ ・内省: fn.__name__, fn.__doc__, fn.__code__ 等にアクセス可能 │
  └────────────────────────────────────────────────────────────────┘

  Rust
  ┌────────────────────────────────────────────────────────────────┐
  │ ・関数ポインタ（fn）とクロージャ（|args| body）の2系統       │
  │ ・クロージャは3つのトレイトで分類:                            │
  │   - Fn:     環境を不変参照でキャプチャ                       │
  │   - FnMut:  環境を可変参照でキャプチャ                       │
  │   - FnOnce: 環境を所有権移動でキャプチャ                     │
  │ ・型推論が強力だが、戻り値型にはdyn/implが必要な場合あり      │
  │ ・ゼロコスト抽象: クロージャはコンパイル時に最適化される      │
  └────────────────────────────────────────────────────────────────┘

  Go
  ┌────────────────────────────────────────────────────────────────┐
  │ ・関数は第一級値（func リテラルで無名関数を生成）             │
  │ ・クロージャは参照でキャプチャ（注意: ループ変数の罠）        │
  │ ・ジェネリクス（Go 1.18+）で型安全な高階関数が記述可能       │
  │ ・メソッド値: obj.Method を関数値として取り出せる             │
  └────────────────────────────────────────────────────────────────┘

  Java
  ┌────────────────────────────────────────────────────────────────┐
  │ ・ラムダ式は SAM（Single Abstract Method）インターフェースの    │
  │   インスタンスとして扱われる                                   │
  │ ・Function<T,R>, Predicate<T>, Consumer<T> 等の関数型IF提供    │
  │ ・メソッド参照: ClassName::method で関数値として取得           │
  │ ・クロージャは「実質final」変数のみキャプチャ可能              │
  └────────────────────────────────────────────────────────────────┘

  Haskell
  ┌────────────────────────────────────────────────────────────────┐
  │ ・全ての関数は自動的にカリー化される                          │
  │ ・関数合成演算子 (.) が組み込み: (f . g) x = f (g x)          │
  │ ・部分適用が最も自然（add 3 で「3を足す関数」が生成）         │
  │ ・型クラスにより多相的な高階関数が自然に記述できる            │
  │ ・遅延評価により無限リストを扱う高階関数も可能                │
  └────────────────────────────────────────────────────────────────┘
```

### 7.3 Rust のクロージャ3種類の詳解

Rust のクロージャは所有権システムと統合されており、3つのトレイトで分類される。これは他の言語には見られない独自の特徴である。

```rust
// ===== Rust: 3種類のクロージャトレイト =====

fn main() {
    // --- Fn: 環境を不変参照でキャプチャ ---
    let name = String::from("Alice");
    let greet = || println!("Hello, {}!", name);  // name を &name でキャプチャ
    greet();  // 何度でも呼べる
    greet();  // 再利用可能
    println!("{}", name);  // name はまだ使える

    // --- FnMut: 環境を可変参照でキャプチャ ---
    let mut count = 0;
    let mut increment = || {
        count += 1;  // count を &mut count でキャプチャ
        println!("count = {}", count);
    };
    increment();  // count = 1
    increment();  // count = 2

    // --- FnOnce: 環境を所有権移動でキャプチャ ---
    let data = vec![1, 2, 3];
    let consume = move || {
        println!("data: {:?}", data);  // data の所有権を移動
        drop(data);  // data を消費
    };
    consume();  // 1回しか呼べない
    // consume();  // コンパイルエラー: 既に消費済み

    // --- 高階関数での使い分け ---
    fn apply_fn<F: Fn()>(f: F) {
        f(); f();  // 複数回呼べる
    }

    fn apply_fn_mut<F: FnMut()>(mut f: F) {
        f(); f();  // 状態を変更しつつ複数回呼べる
    }

    fn apply_fn_once<F: FnOnce()>(f: F) {
        f();  // 1回だけ呼べる
    }
}
```

```
Rust クロージャトレイトの包含関係
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────┐
  │           FnOnce                    │  最も制約が緩い（全クロージャが実装）
  │  ┌──────────────────────────────┐   │
  │  │         FnMut                │   │  FnOnce のサブトレイト
  │  │  ┌───────────────────────┐   │   │
  │  │  │        Fn             │   │   │  最も制約が厳しい
  │  │  │                       │   │   │
  │  │  │  不変参照キャプチャ    │   │   │
  │  │  └───────────────────────┘   │   │
  │  │  可変参照キャプチャ          │   │
  │  └──────────────────────────────┘   │
  │  所有権移動キャプチャ               │
  └─────────────────────────────────────┘

  Fn ⊂ FnMut ⊂ FnOnce

  ・Fn を実装するクロージャは FnMut も FnOnce も実装する
  ・FnMut を実装するクロージャは FnOnce も実装する
  ・関数ポインタ fn は Fn, FnMut, FnOnce 全てを実装する
```

---

## 8. 実践的なデザインパターン

### 8.1 ミドルウェアパターン

Webフレームワーク（Express.js、Koa、axum等）で広く使われるミドルウェアパターンは、第一級関数の典型的な活用例である。

```javascript
// ===== JavaScript: ミドルウェアパターン =====

// シンプルなミドルウェアシステム
class Pipeline {
    constructor() {
        this.middlewares = [];
    }

    use(middleware) {
        this.middlewares.push(middleware);
        return this;  // メソッドチェーン
    }

    async execute(context) {
        let index = 0;
        const next = async () => {
            if (index < this.middlewares.length) {
                const middleware = this.middlewares[index++];
                await middleware(context, next);
            }
        };
        await next();
        return context;
    }
}

// ミドルウェアの定義（各ミドルウェアは関数）
const logger = async (ctx, next) => {
    const start = Date.now();
    console.log(`→ ${ctx.method} ${ctx.path}`);
    await next();
    console.log(`← ${ctx.method} ${ctx.path} (${Date.now() - start}ms)`);
};

const auth = async (ctx, next) => {
    if (!ctx.headers.authorization) {
        ctx.status = 401;
        ctx.body = "Unauthorized";
        return;  // next() を呼ばない → 以降のミドルウェアをスキップ
    }
    ctx.user = verifyToken(ctx.headers.authorization);
    await next();
};

const handler = async (ctx, next) => {
    ctx.status = 200;
    ctx.body = { message: "Hello", user: ctx.user };
    await next();
};

// パイプラインの組み立て
const app = new Pipeline();
app.use(logger).use(auth).use(handler);

// 実行
await app.execute({
    method: "GET",
    path: "/api/users",
    headers: { authorization: "Bearer token123" },
});
```

### 8.2 イベントエミッタパターン

```typescript
// ===== TypeScript: 型安全なイベントエミッタ =====

type EventMap = {
    "user:login":  { userId: string; timestamp: number };
    "user:logout": { userId: string; reason: string };
    "error":       { code: number; message: string };
};

class TypedEventEmitter<T extends Record<string, unknown>> {
    private listeners = new Map<keyof T, Set<(data: any) => void>>();

    on<K extends keyof T>(event: K, handler: (data: T[K]) => void): () => void {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)!.add(handler);

        // 解除関数を返す（これも第一級関数の活用）
        return () => {
            this.listeners.get(event)?.delete(handler);
        };
    }

    emit<K extends keyof T>(event: K, data: T[K]): void {
        this.listeners.get(event)?.forEach(handler => handler(data));
    }
}

// 使用例
const emitter = new TypedEventEmitter<EventMap>();

const unsubscribe = emitter.on("user:login", (data) => {
    console.log(`ユーザー ${data.userId} がログイン (${data.timestamp})`);
});

emitter.emit("user:login", {
    userId: "user-123",
    timestamp: Date.now(),
});

// 不要になったら解除
unsubscribe();
```

### 8.3 メモ化パターン

```javascript
// ===== JavaScript: 汎用メモ化関数 =====

function memoize(fn, options = {}) {
    const cache = new Map();
    const { maxSize = 1000, keyFn = JSON.stringify } = options;

    function memoized(...args) {
        const key = keyFn(args);

        if (cache.has(key)) {
            return cache.get(key);
        }

        const result = fn.apply(this, args);

        // キャッシュサイズ制限
        if (cache.size >= maxSize) {
            const firstKey = cache.keys().next().value;
            cache.delete(firstKey);
        }

        cache.set(key, result);
        return result;
    }

    // キャッシュ操作メソッドを付加
    memoized.cache = cache;
    memoized.clear = () => cache.clear();

    return memoized;
}

// 使用例: フィボナッチ数列
const fib = memoize((n) => {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
});

console.log(fib(50));  // => 12586269025（高速に計算される）

// 使用例: API レスポンスキャッシュ
const fetchUser = memoize(
    async (id) => {
        const res = await fetch(`/api/users/${id}`);
        return res.json();
    },
    { maxSize: 100 }
);
```

```python
# ===== Python: lru_cache デコレータ =====

from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    """メモ化されたフィボナッチ数列"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))
# => 354224848179261915075（高速に計算される）

# キャッシュ統計
print(fibonacci.cache_info())
# => CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)

# キャッシュクリア
fibonacci.cache_clear()
```

---

## 9. アンチパターンと注意点

### 9.1 アンチパターン1: 過剰な抽象化（Abstraction Astronaut）

第一級関数の強力さに魅せられて、過剰に抽象化してしまうケースがある。

```javascript
// ===== アンチパターン: 過剰な抽象化 =====

// BAD: 不要な関数ラッピング
const add = (a) => (b) => a + b;
const multiply = (a) => (b) => a * b;
const compose = (f) => (g) => (x) => f(g(x));
const pipe = (...fns) => (x) => fns.reduce((a, f) => f(a), x);

// 単純な計算なのに読解コストが高い
const result = pipe(
    add(1),
    multiply(2),
    compose(add(3))(multiply(4)),
)(5);
// 何をしているのか一見して分からない...

// GOOD: 適切な抽象度
function calculatePrice(basePrice, taxRate, discount) {
    const afterDiscount = basePrice * (1 - discount);
    const withTax = afterDiscount * (1 + taxRate);
    return Math.round(withTax);
}

calculatePrice(1000, 0.1, 0.2);  // => 880
// 意図が明確で読みやすい
```

**教訓**: 抽象化は「繰り返しパターンが3回以上出現した場合」に検討する（Rule of Three）。1-2回しか使わないパターンを関数化すると、かえって可読性を損なう。

### 9.2 アンチパターン2: this 参照の喪失

JavaScript において、関数を値として取り回す際に最も頻出するバグが `this` 参照の喪失である。

```javascript
// ===== アンチパターン: this の喪失 =====

class UserService {
    constructor() {
        this.users = ["Alice", "Bob"];
    }

    getUsers() {
        return this.users;
    }
}

const service = new UserService();

// BAD: メソッドを変数に代入すると this が失われる
const getUsers = service.getUsers;
// getUsers();  // TypeError: Cannot read property 'users' of undefined

// BAD: コールバックとして渡しても this が失われる
// setTimeout(service.getUsers, 1000);  // 同様のエラー

// --- 解決策 ---

// 解決策1: bind で this を束縛
const getUsersBound = service.getUsers.bind(service);
getUsersBound();  // => ["Alice", "Bob"]

// 解決策2: アロー関数でラップ
setTimeout(() => service.getUsers(), 1000);

// 解決策3: クラスフィールドでアロー関数を使う（推奨）
class UserServiceFixed {
    users = ["Alice", "Bob"];

    // アロー関数のクラスフィールド → this が固定される
    getUsers = () => {
        return this.users;
    };
}

const fixed = new UserServiceFixed();
const fn = fixed.getUsers;
fn();  // => ["Alice", "Bob"]（正常動作）
```

### 9.3 アンチパターン3: ループ変数のクロージャキャプチャ

```javascript
// ===== アンチパターン: ループ変数の罠 =====

// BAD: var を使ったループでのクロージャ
const functions = [];
for (var i = 0; i < 5; i++) {
    functions.push(() => i);
}
console.log(functions.map(f => f()));
// => [5, 5, 5, 5, 5]  全て5になる！（var はブロックスコープを持たない）

// GOOD: let を使う（ブロックスコープ）
const functions2 = [];
for (let i = 0; i < 5; i++) {
    functions2.push(() => i);
}
console.log(functions2.map(f => f()));
// => [0, 1, 2, 3, 4]  期待通り

// GOOD: IIFE（即時実行関数式）で値をキャプチャ
const functions3 = [];
for (var i = 0; i < 5; i++) {
    functions3.push(((captured) => () => captured)(i));
}

// BEST: 関数型アプローチ
const functions4 = Array.from({ length: 5 }, (_, i) => () => i);
console.log(functions4.map(f => f()));
// => [0, 1, 2, 3, 4]
```

```go
// ===== Go: ループ変数の罠（Go 1.21 以前） =====

package main

import "fmt"

func main() {
    // BAD（Go 1.21 以前）: ループ変数はイテレーション間で共有される
    fns := make([]func(), 5)
    for i := 0; i < 5; i++ {
        fns[i] = func() { fmt.Println(i) }
    }
    for _, f := range fns {
        f()  // Go 1.21以前: 全て5。Go 1.22+: 0,1,2,3,4
    }

    // GOOD: ローカル変数にコピー（Go 1.21 以前の対策）
    fns2 := make([]func(), 5)
    for i := 0; i < 5; i++ {
        i := i  // シャドーイングで値をキャプチャ
        fns2[i] = func() { fmt.Println(i) }
    }
}
```

### 9.4 アンチパターン4: パフォーマンスの罠

```javascript
// ===== アンチパターン: 不要なクロージャの生成 =====

// BAD: レンダリングのたびに新しい関数オブジェクトが生成される
function TodoList({ todos, onDelete }) {
    return todos.map(todo => (
        // 毎回新しいアロー関数が生成され、子コンポーネントの
        // メモ化が無効化される
        <TodoItem
            key={todo.id}
            todo={todo}
            onDelete={() => onDelete(todo.id)}  // 毎回新しい関数
        />
    ));
}

// GOOD: useCallback やメソッド参照を使用
function TodoListOptimized({ todos, onDelete }) {
    const handleDelete = useCallback((id) => {
        onDelete(id);
    }, [onDelete]);

    return todos.map(todo => (
        <TodoItem
            key={todo.id}
            todo={todo}
            onDelete={handleDelete}
            todoId={todo.id}
        />
    ));
}
```

---

## 10. 演習問題（3段階）

### 10.1 基礎演習（Beginner）

**演習 B-1: 関数を値として操作する**

以下の要件を満たすコードを JavaScript で実装せよ。

```
要件:
1. 関数 applyOperation(a, b, operation) を定義する
   - a, b は数値、operation は2引数の関数
   - operation(a, b) の結果を返す
2. 加算・減算・乗算・除算の4つの関数を定義する
3. applyOperation を使って以下を計算する:
   - 10 + 5 = 15
   - 10 - 5 = 5
   - 10 * 5 = 50
   - 10 / 5 = 2
```

```javascript
// 解答例

function applyOperation(a, b, operation) {
    return operation(a, b);
}

const add = (a, b) => a + b;
const subtract = (a, b) => a - b;
const multiply = (a, b) => a * b;
const divide = (a, b) => {
    if (b === 0) throw new Error("0で除算はできません");
    return a / b;
};

console.log(applyOperation(10, 5, add));       // => 15
console.log(applyOperation(10, 5, subtract));  // => 5
console.log(applyOperation(10, 5, multiply));  // => 50
console.log(applyOperation(10, 5, divide));    // => 2
```

**演習 B-2: フィルタ関数の実装**

```
要件:
Python で以下を実装せよ。
1. リスト [1, -2, 3, -4, 5, -6, 7, -8, 9, -10] から:
   a. 正の数だけを抽出
   b. 偶数だけを抽出
   c. 絶対値が5以上のものだけを抽出
2. それぞれ filter() と lambda を使って1行で実現する
```

```python
# 解答例

numbers = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]

positives = list(filter(lambda x: x > 0, numbers))
# => [1, 3, 5, 7, 9]

evens = list(filter(lambda x: x % 2 == 0, numbers))
# => [-2, -4, -6, -8, -10]

abs_gte_5 = list(filter(lambda x: abs(x) >= 5, numbers))
# => [5, -6, 7, -8, 9, -10]
```

**演習 B-3: ディスパッチテーブル**

```
要件:
JavaScript で簡易的な文字列変換ディスパッチテーブルを作成せよ。
- "upper": 大文字変換
- "lower": 小文字変換
- "reverse": 文字列反転
- "length": 文字数を文字列で返す
- 未知のコマンドにはエラーメッセージを返す
```

```javascript
// 解答例

const transforms = {
    upper:   (s) => s.toUpperCase(),
    lower:   (s) => s.toLowerCase(),
    reverse: (s) => s.split("").reverse().join(""),
    length:  (s) => `${s.length}文字`,
};

function applyTransform(command, text) {
    const fn = transforms[command];
    if (!fn) {
        return `エラー: 未知のコマンド "${command}"`;
    }
    return fn(text);
}

console.log(applyTransform("upper", "hello"));     // => "HELLO"
console.log(applyTransform("reverse", "hello"));   // => "olleh"
console.log(applyTransform("length", "hello"));    // => "5文字"
console.log(applyTransform("unknown", "hello"));   // => "エラー: 未知のコマンド..."
```

### 10.2 中級演習（Intermediate）

**演習 I-1: 関数合成パイプラインの実装**

```
要件:
TypeScript で以下を実装せよ。
1. pipe 関数を実装する（任意個の関数を左から右へ合成）
2. 以下のデータ処理パイプラインを構築する:
   - 入力: 文字列の配列
   - ステップ1: 全て小文字に変換
   - ステップ2: 3文字以下の単語を除外
   - ステップ3: 重複を除去
   - ステップ4: アルファベット順にソート
   - ステップ5: カンマ区切りの文字列に結合
```

```typescript
// 解答例

// pipe の実装（型安全版は複雑になるため、ここではシンプル版）
function pipe<T>(...fns: Array<(arg: T) => T>): (arg: T) => T {
    return (arg: T) => fns.reduce((acc, fn) => fn(acc), arg);
}

// 各ステップを関数として定義
const toLowerAll = (words: string[]) => words.map(w => w.toLowerCase());
const filterShort = (words: string[]) => words.filter(w => w.length > 3);
const unique = (words: string[]) => [...new Set(words)];
const sortAlpha = (words: string[]) => [...words].sort();
const joinComma = (words: string[]) => words.join(", ");

// パイプラインの構築（型の都合上、段階的に適用）
function processWords(input: string[]): string {
    const step1 = toLowerAll(input);
    const step2 = filterShort(step1);
    const step3 = unique(step2);
    const step4 = sortAlpha(step3);
    return joinComma(step4);
}

// テスト
const input = ["Hello", "WORLD", "the", "HELLO", "JavaScript", "is", "Great", "hello"];
console.log(processWords(input));
// => "great, hello, javascript, world"
```

**演習 I-2: 汎用 retry 関数の実装**

```
要件:
JavaScript で以下を満たす retry 関数を実装せよ。
1. 失敗した関数を指定回数リトライする
2. リトライ間隔は指数バックオフ（1秒、2秒、4秒...）
3. 最大リトライ回数を超えたら最後のエラーをスローする
4. 各リトライ時にログを出力する
5. 成功した場合は結果を返す
```

```javascript
// 解答例

async function retry(fn, options = {}) {
    const {
        maxAttempts = 3,
        baseDelay = 1000,
        backoffFactor = 2,
        onRetry = (attempt, err) => {
            console.log(`リトライ ${attempt}回目: ${err.message}`);
        },
    } = options;

    let lastError;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn(attempt);
        } catch (err) {
            lastError = err;
            if (attempt < maxAttempts) {
                const delay = baseDelay * Math.pow(backoffFactor, attempt - 1);
                onRetry(attempt, err);
                await new Promise(r => setTimeout(r, delay));
            }
        }
    }

    throw new Error(
        `${maxAttempts}回リトライしましたが失敗しました: ${lastError.message}`
    );
}

// テスト: 3回目で成功するシミュレーション
let callCount = 0;
const result = await retry(
    async (attempt) => {
        callCount++;
        if (callCount < 3) {
            throw new Error(`接続エラー（試行${callCount}）`);
        }
        return { data: "成功！" };
    },
    { maxAttempts: 5, baseDelay: 100 }
);
console.log(result);  // => { data: "成功！" }
```

**演習 I-3: デコレータパターンの実装**

```
要件:
Python で以下のデコレータを実装せよ。
1. @validate_args: 全引数が None でないことを検証
2. @log_calls: 関数呼び出し時に引数と戻り値をログ出力
3. 両方のデコレータを重ねて使用する
```

```python
# 解答例

import functools

def validate_args(func):
    """全引数が None でないことを検証するデコレータ"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i, arg in enumerate(args):
            if arg is None:
                raise ValueError(
                    f"{func.__name__}: 引数{i}が None です"
                )
        for key, val in kwargs.items():
            if val is None:
                raise ValueError(
                    f"{func.__name__}: キーワード引数'{key}'が None です"
                )
        return func(*args, **kwargs)
    return wrapper


def log_calls(func):
    """関数呼び出しをログ出力するデコレータ"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"呼び出し: {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"戻り値: {func.__name__} → {result!r}")
        return result
    return wrapper


@log_calls
@validate_args
def divide(a, b):
    """a / b を計算する"""
    if b == 0:
        raise ZeroDivisionError("0で除算はできません")
    return a / b


# テスト
divide(10, 3)
# 呼び出し: divide(10, 3)
# 戻り値: divide → 3.3333333333333335

# divide(10, None)
# ValueError: divide: 引数1が None です
```

### 10.3 上級演習（Advanced）

**演習 A-1: モナド風パイプラインの実装**

```
要件:
TypeScript で Optional（Maybe）モナド風のパイプラインを実装せよ。
1. Optional<T> クラスを実装:
   - of(value): 値をラップ
   - map(fn): 値があれば関数を適用
   - flatMap(fn): Optional を返す関数を適用（ネスト解消）
   - getOrElse(defaultValue): 値がなければデフォルト値を返す
2. ユーザーの住所の郵便番号を安全に取得するパイプラインを構築
```

```typescript
// 解答例

class Optional<T> {
    private constructor(private readonly value: T | null | undefined) {}

    static of<T>(value: T | null | undefined): Optional<T> {
        return new Optional(value);
    }

    static empty<T>(): Optional<T> {
        return new Optional<T>(null);
    }

    isPresent(): boolean {
        return this.value !== null && this.value !== undefined;
    }

    map<U>(fn: (value: T) => U): Optional<U> {
        if (!this.isPresent()) return Optional.empty<U>();
        return Optional.of(fn(this.value!));
    }

    flatMap<U>(fn: (value: T) => Optional<U>): Optional<U> {
        if (!this.isPresent()) return Optional.empty<U>();
        return fn(this.value!);
    }

    getOrElse(defaultValue: T): T {
        return this.isPresent() ? this.value! : defaultValue;
    }

    filter(predicate: (value: T) => boolean): Optional<T> {
        if (!this.isPresent()) return this;
        return predicate(this.value!) ? this : Optional.empty();
    }
}

// ユーザーデータの型定義
interface User {
    name: string;
    address?: {
        street?: string;
        city?: string;
        zip?: string;
    };
}

// 安全な郵便番号取得パイプライン
function getFormattedZip(user: User | null): string {
    return Optional.of(user)
        .flatMap(u => Optional.of(u.address))
        .flatMap(a => Optional.of(a.zip))
        .filter(zip => /^\d{3}-?\d{4}$/.test(zip))
        .map(zip => zip.includes("-") ? zip : `${zip.slice(0,3)}-${zip.slice(3)}`)
        .getOrElse("郵便番号なし");
}

// テスト
const user1: User = { name: "太郎", address: { zip: "1000001" } };
const user2: User = { name: "花子" };
const user3: User | null = null;

console.log(getFormattedZip(user1));  // => "100-0001"
console.log(getFormattedZip(user2));  // => "郵便番号なし"
console.log(getFormattedZip(user3));  // => "郵便番号なし"
```

**演習 A-2: 型安全なイベントバスの設計**

```
要件:
TypeScript で以下を満たすイベントバスを設計・実装せよ。
1. イベント名と型の対応を型レベルで保証する
2. on/off/emit/once の4メソッドを提供する
3. ワイルドカード（*）リスナーをサポートする
4. リスナーの優先度をサポートする
```

```typescript
// 解答例

type EventHandler<T = unknown> = (data: T) => void;

interface ListenerEntry<T = unknown> {
    handler: EventHandler<T>;
    priority: number;
    once: boolean;
}

class EventBus<TEvents extends Record<string, unknown>> {
    private listeners = new Map<string, ListenerEntry[]>();
    private wildcardListeners: ListenerEntry<{ event: string; data: unknown }>[] = [];

    on<K extends keyof TEvents & string>(
        event: K,
        handler: EventHandler<TEvents[K]>,
        priority = 0,
    ): () => void {
        return this.addListener(event, handler, priority, false);
    }

    once<K extends keyof TEvents & string>(
        event: K,
        handler: EventHandler<TEvents[K]>,
        priority = 0,
    ): () => void {
        return this.addListener(event, handler, priority, true);
    }

    onAny(
        handler: EventHandler<{ event: string; data: unknown }>,
        priority = 0,
    ): () => void {
        const entry: ListenerEntry<{ event: string; data: unknown }> = {
            handler,
            priority,
            once: false,
        };
        this.wildcardListeners.push(entry);
        this.wildcardListeners.sort((a, b) => b.priority - a.priority);
        return () => {
            const idx = this.wildcardListeners.indexOf(entry);
            if (idx >= 0) this.wildcardListeners.splice(idx, 1);
        };
    }

    emit<K extends keyof TEvents & string>(event: K, data: TEvents[K]): void {
        // 通常リスナー
        const entries = this.listeners.get(event) || [];
        const remaining: ListenerEntry[] = [];
        for (const entry of entries) {
            entry.handler(data);
            if (!entry.once) remaining.push(entry);
        }
        this.listeners.set(event, remaining);

        // ワイルドカードリスナー
        for (const entry of this.wildcardListeners) {
            entry.handler({ event, data });
        }
    }

    private addListener(
        event: string,
        handler: EventHandler<any>,
        priority: number,
        once: boolean,
    ): () => void {
        const entry: ListenerEntry = { handler, priority, once };
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        const entries = this.listeners.get(event)!;
        entries.push(entry);
        entries.sort((a, b) => b.priority - a.priority);

        return () => {
            const idx = entries.indexOf(entry);
            if (idx >= 0) entries.splice(idx, 1);
        };
    }
}

// 使用例
interface AppEvents {
    "user:login":  { userId: string; time: number };
    "user:logout": { userId: string };
    "error":       { code: number; message: string };
}

const bus = new EventBus<AppEvents>();

// 通常リスナー
const unsub = bus.on("user:login", (data) => {
    console.log(`ログイン: ${data.userId}`);
});

// 1回限りのリスナー
bus.once("error", (data) => {
    console.error(`エラー ${data.code}: ${data.message}`);
});

// ワイルドカードリスナー（全イベント監視）
bus.onAny(({ event, data }) => {
    console.log(`[監査ログ] ${event}:`, data);
});

bus.emit("user:login", { userId: "user-1", time: Date.now() });
unsub();  // リスナー解除
```

**演習 A-3: 遅延評価チェーンの実装**

```
要件:
JavaScript で遅延評価（Lazy Evaluation）コレクション処理を実装せよ。
1. LazySeq クラスを実装:
   - map, filter, take, forEach メソッド
   - forEach が呼ばれるまで実際の計算を行わない
2. 無限数列に対して map → filter → take が動作することを確認
```

```javascript
// 解答例

class LazySeq {
    constructor(iterable) {
        this.source = iterable;
        this.transforms = [];
    }

    static from(iterable) {
        return new LazySeq(iterable);
    }

    static range(start = 0, end = Infinity, step = 1) {
        return new LazySeq({
            *[Symbol.iterator]() {
                for (let i = start; i < end; i += step) {
                    yield i;
                }
            }
        });
    }

    map(fn) {
        const clone = new LazySeq(this.source);
        clone.transforms = [...this.transforms, { type: "map", fn }];
        return clone;
    }

    filter(fn) {
        const clone = new LazySeq(this.source);
        clone.transforms = [...this.transforms, { type: "filter", fn }];
        return clone;
    }

    take(n) {
        const clone = new LazySeq(this.source);
        clone.transforms = [...this.transforms, { type: "take", n }];
        return clone;
    }

    *[Symbol.iterator]() {
        let count = 0;
        let takeLimit = Infinity;

        // take の上限を事前計算
        for (const t of this.transforms) {
            if (t.type === "take") takeLimit = Math.min(takeLimit, t.n);
        }

        outer:
        for (const item of this.source) {
            let value = item;
            let skip = false;

            for (const transform of this.transforms) {
                if (transform.type === "map") {
                    value = transform.fn(value);
                } else if (transform.type === "filter") {
                    if (!transform.fn(value)) {
                        skip = true;
                        break;
                    }
                } else if (transform.type === "take") {
                    // 後で処理
                }
            }

            if (skip) continue;
            if (count >= takeLimit) break;
            count++;
            yield value;
        }
    }

    toArray() {
        return [...this];
    }

    forEach(fn) {
        for (const item of this) {
            fn(item);
        }
    }
}

// テスト: 無限数列から素数を10個取得
function isPrime(n) {
    if (n < 2) return false;
    for (let i = 2; i <= Math.sqrt(n); i++) {
        if (n % i === 0) return false;
    }
    return true;
}

const first10Primes = LazySeq.range(2)
    .filter(isPrime)
    .take(10)
    .toArray();

console.log(first10Primes);
// => [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

// テスト: 遅延評価の確認（無限数列でも動作する）
const result = LazySeq.range(1)
    .map(x => x * x)
    .filter(x => x % 2 === 1)
    .take(5)
    .toArray();

console.log(result);
// => [1, 9, 25, 49, 81]
```

---

## 11. FAQ（よくある質問）

### Q1: 第一級関数とクロージャは同じものですか？

**A**: いいえ、異なる概念です。第一級関数とは「関数を値として扱える」という言語の性質を指し、クロージャは「定義時の環境（変数束縛）を保持した関数」を指します。クロージャは第一級関数の仕組みを前提として成立しますが、第一級関数であってもクロージャでないもの（環境をキャプチャしない純粋な関数ポインタ等）は存在します。

```
関係図:
  ┌──────────────────────────────────────┐
  │         第一級関数                    │
  │  ┌───────────────────────────────┐   │
  │  │      クロージャ                │   │
  │  │  （環境をキャプチャした関数）   │   │
  │  └───────────────────────────────┘   │
  │                                      │
  │  関数ポインタ（環境なし）            │
  │  純粋関数リテラル（環境なし）        │
  └──────────────────────────────────────┘
```

### Q2: ラムダ式と無名関数は同じものですか？

**A**: 実用上はほぼ同義で使われます。厳密には、「ラムダ式」はラムダ計算に由来する数学的な概念で、「無名関数（anonymous function）」は名前を持たない関数リテラルを指すプログラミング用語です。Python では `lambda` キーワードで無名関数を作りますが、式1つしか書けない制限があります。JavaScript のアロー関数 `(x) => x + 1` も無名関数（ラムダ式）の一種です。

### Q3: 全ての関数をアロー関数で書くべきですか？（JavaScript）

**A**: いいえ。アロー関数は便利ですが、以下のケースでは通常の function 宣言が適しています。

1. **`this` バインディングが必要な場合**: アロー関数は独自の `this` を持たないため、オブジェクトのメソッドやプロトタイプメソッドには不向きです
2. **`arguments` オブジェクトが必要な場合**: アロー関数は `arguments` を持ちません
3. **ジェネレータ関数**: `function*` 構文が必要で、アロー関数では書けません
4. **ホイスティングが必要な場合**: function 宣言はホイスティングされますが、アロー関数（const への代入）はされません
5. **コンストラクタ**: アロー関数は `new` で呼び出せません

### Q4: 高階関数を使うとパフォーマンスは低下しますか？

**A**: 一般的に、高階関数の呼び出しオーバーヘッドは無視できるほど小さいです。現代の JavaScript エンジン（V8 等）は、インライン展開や JIT コンパイルにより、高階関数のオーバーヘッドをほぼゼロに最適化します。ただし、以下の場合は注意が必要です。

- **ホットループ内での関数オブジェクト生成**: `arr.map(x => x * 2)` のようにリテラルを直接渡す場合、毎回新しい関数オブジェクトが生成される可能性がある（多くのエンジンは最適化する）
- **React での再レンダリング**: コンポーネント内で毎回新しい関数を生成すると、子コンポーネントの不要な再レンダリングを引き起こす可能性がある
- **Rust**: クロージャはコンパイル時にモノモーフィゼーションされるため、ランタイムオーバーヘッドはゼロ

### Q5: 関数型プログラミングとオブジェクト指向プログラミングはどちらが優れていますか？

**A**: これは二者択一ではありません。現代の多くの言語（JavaScript、Python、Kotlin、Swift、Rust 等）は両方のパラダイムをサポートしており、場面に応じて使い分けるのが最善です。

| 場面 | 関数型が適する | OOPが適する |
|------|--------------|-------------|
| データ変換 | map/filter/reduce パイプライン | - |
| 状態管理 | イミュータブルなデータフロー | ステートフルなオブジェクト |
| 振る舞い切替 | 高階関数・コールバック | ストラテジーパターン（クラス） |
| 大規模設計 | 関数合成・モジュール | クラス階層・DI |
| ドメインモデル | 代数的データ型 | エンティティ・値オブジェクト |
| 並行処理 | 純粋関数・アクター | synchronized・ロック |

### Q6: カリー化と部分適用の違いは何ですか？

**A**: 似ていますが異なる操作です。

- **カリー化（Currying）**: n引数の関数を、1引数関数のチェーンに変換する。`f(a, b, c)` → `f(a)(b)(c)`。Haskell では全ての関数が自動的にカリー化されている
- **部分適用（Partial Application）**: n引数の関数の一部の引数を固定し、残りの引数を受け取る新しい関数を生成する。`f(a, b, c)` で a を固定 → `g(b, c) = f(fixed_a, b, c)`

```
カリー化 vs 部分適用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  元の関数: f(a, b, c)

  カリー化:
  f(a, b, c) → curry(f) → g(a)(b)(c)
  全引数を1つずつ受け取る形に変換

  部分適用:
  f(a, b, c) → partial(f, fixedA) → g(b, c)
  一部の引数を固定して新関数を生成

  違い:
  ・カリー化は常に1引数ずつ
  ・部分適用は任意の数の引数を固定可能
  ・カリー化は関数の形を変える変換
  ・部分適用は引数を埋める操作
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 12. まとめ

### 12.1 概念の整理

| 概念 | 定義 | 代表的な使用場面 |
|------|------|------------------|
| 第一級関数 | 関数を値として扱える言語機能 | 変数代入、引数渡し、戻り値 |
| 高階関数 | 関数を受け取る/返す関数 | map, filter, reduce, デコレータ |
| コールバック | 引数として渡される関数 | イベントハンドラ、非同期処理 |
| ストラテジー | 振る舞いを関数で差し替える設計 | ソート順の切替、バリデーション |
| 部分適用 | 引数の一部を固定して新関数を作る | 設定の固定、ロガー生成 |
| カリー化 | n引数関数を1引数関数のチェーンに変換 | 段階的な引数適用 |
| 関数合成 | 複数の関数を組み合わせて新関数を作る | データ処理パイプライン |
| クロージャ | 定義時の環境を保持した関数 | 状態の隠蔽、ファクトリ |
| ディスパッチテーブル | 関数を辞書に格納して動的に選択 | コマンド処理、ルーティング |
| メモ化 | 結果をキャッシュする高階関数 | 計算コストの高い関数の最適化 |
| ミドルウェア | 処理を挟み込む関数チェーン | HTTPリクエスト処理、ログ |

### 12.2 設計判断のガイドライン

```
第一級関数の適用判断フローチャート
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  処理パターンを抽象化したい？
  │
  ├─ YES → 同じパターンが3回以上出現する？
  │         │
  │         ├─ YES → 高階関数として抽出
  │         │         │
  │         │         ├─ 変換系 → map 的な関数を作成
  │         │         ├─ 選択系 → filter 的な関数を作成
  │         │         └─ 集約系 → reduce 的な関数を作成
  │         │
  │         └─ NO  → インラインのままにする
  │                   （過剰な抽象化を避ける）
  │
  ├─ 振る舞いを実行時に切り替えたい？
  │  │
  │  ├─ YES → 状態が複雑？
  │  │         │
  │  │         ├─ YES → OOPストラテジーパターン
  │  │         └─ NO  → 関数を引数/変数で切り替え
  │  │
  │  └─ NO
  │
  └─ 設定を事前に固定した関数が欲しい？
     │
     ├─ YES → 部分適用 or ファクトリ関数
     └─ NO  → 通常の関数で十分
```

---

## 13. 次に読むべきガイド


---

## 14. 参考文献

1. Abelson, H. & Sussman, G. J. *Structure and Interpretation of Computer Programs (SICP)*, 2nd ed., MIT Press, 1996. -- 第1章「Building Abstractions with Procedures」で第一級関数の概念を徹底的に解説。LISP/Scheme を通じて高階関数・関数合成の本質を学べる古典的名著。

2. Strachey, C. "Fundamental Concepts in Programming Languages," *Higher-Order and Symbolic Computation*, Vol.13, pp.11--49, 2000 (originally 1967 lecture notes). -- 「第一級（first-class）」の概念を最初に体系化した歴史的文献。プログラミング言語の意味論における基礎概念を定義した。

3. Bird, R. *Thinking Functionally with Haskell*, Cambridge University Press, 2015. -- Haskell を題材に、第一級関数・高階関数・関数合成・カリー化を含む関数型プログラミングの考え方を体系的に学べる。

4. Crockford, D. *JavaScript: The Good Parts*, O'Reilly Media, 2008. -- JavaScript における関数の第一級性、クロージャ、コールバックパターンを実践的に解説。JavaScript 特有の落とし穴（this の喪失等）も詳述。

5. Kleppmann, M. *Designing Data-Intensive Applications*, O'Reilly Media, 2017. -- 分散システムにおける関数型パターン（イミュータブルデータ、純粋関数）の重要性を実システムの文脈で論じている。

6. Gamma, E., Helm, R., Johnson, R. & Vlissides, J. *Design Patterns: Elements of Reusable Object-Oriented Software*, Addison-Wesley, 1994. -- ストラテジーパターン、コマンドパターン等、第一級関数で簡潔に表現できるGoFパターンの原典。OOPとFPの対比を理解する上で有用。

---

## 次に読むべきガイド

- [クロージャ（Closures）](./01-closures.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
