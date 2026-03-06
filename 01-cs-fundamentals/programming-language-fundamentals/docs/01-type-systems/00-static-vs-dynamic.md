# 静的型付け vs 動的型付け

> 型システムは「プログラムの正しさを保証する」最も基本的な仕組みであり、
> あらゆるプログラミング言語の設計思想の根幹を成す概念である。

## この章で学ぶこと

- [ ] 型の数学的定義と、型システムが果たす3つの役割を説明できる
- [ ] 静的型付けと動的型付けの本質的な違いを、コンパイルパイプラインの観点から理解する
- [ ] 強い型付けと弱い型付けの区別を、暗黙的型変換の観点から理解する
- [ ] 段階的型付け（Gradual Typing）の理論的背景と産業的意義を理解する
- [ ] 型システムの健全性（Soundness）と完全性（Completeness）のトレードオフを理解する
- [ ] 各言語の型システム設計を比較し、プロジェクトに適した言語選択の判断基準を持つ

---

## 1. 型とは何か --- 数学的基礎から実務的意義まで

### 1.1 型の定義

型とは、数学的には「値の集合」と「その値に対して許される操作の集合」の組である。
この定義は集合論に基礎を置き、型理論（Type Theory）として形式化されている。

```
型の数学的定義:

  型 T = (V, O)
    V: 値の集合（Value set）
    O: 許容される操作の集合（Operation set）

  具体例:

  Int型:
    V = {..., -2, -1, 0, 1, 2, ...}  (整数の集合)
    O = {+, -, *, /, %, ==, <, >, ...}

  String型:
    V = {"", "a", "hello", "世界", ...}  (文字列の集合)
    O = {concat, length, substring, indexOf, ...}

  Bool型:
    V = {true, false}  (真偽値の集合)
    O = {AND, OR, NOT, XOR, ...}

  Unit / Void型:
    V = {()}  (唯一の値を持つ集合)
    O = {}    (操作なし)
```

### 1.2 型が果たす3つの役割

型システムは、プログラミングにおいて以下の3つの役割を同時に果たす。

```
+------------------------------------------------------------------+
|                    型システムの3つの役割                            |
+------------------------------------------------------------------+
|                                                                    |
|  1. 安全性の保証 (Safety Guarantee)                                |
|     - メモリ上のデータの正しい解釈を保証                            |
|     - 不正な操作の防止（文字列に算術演算を適用しない等）             |
|     - バッファオーバーフローや型混同バグの防止                       |
|                                                                    |
|  2. 仕様の文書化 (Specification as Documentation)                  |
|     - 関数の入出力契約を型で表現                                    |
|     - コードを読む人への情報提供                                    |
|     - IDE による補完・ナビゲーションの基盤                          |
|                                                                    |
|  3. 最適化の基盤 (Optimization Foundation)                         |
|     - コンパイラがメモリレイアウトを決定する根拠                     |
|     - 不要な実行時チェックの省略                                    |
|     - 特殊化・インライン化などの最適化判断材料                       |
|                                                                    |
+------------------------------------------------------------------+
```

### 1.3 型と値の関係 --- 型は「契約」である

型を「契約（Contract）」として理解することは、実務上極めて重要である。

```typescript
// TypeScript: 型は関数の「契約」を表現する

// この型シグネチャは以下の契約を表す:
// - 呼び出し側: number型の引数を2つ渡す義務がある
// - 関数側: number型の値を返す義務がある
function divide(numerator: number, denominator: number): number {
    if (denominator === 0) {
        throw new Error("Division by zero");
    }
    return numerator / denominator;
}

// 契約に従った呼び出し --- コンパイル成功
const result = divide(10, 3);  // number型が返る

// 契約違反 --- コンパイルエラー
// const bad = divide("10", 3);
// Error: Argument of type 'string' is not assignable to parameter of type 'number'
```

型が契約として機能する場合、以下の恩恵が得られる。

```
型の契約としての機能:

  呼び出し側の義務        関数側の義務
  ┌──────────────┐      ┌──────────────┐
  │ 正しい型の    │      │ 宣言した型の  │
  │ 引数を渡す    │ ──→  │ 値を返す      │
  └──────────────┘      └──────────────┘
         │                      │
         ▼                      ▼
  コンパイラが検証          コンパイラが検証

  契約違反があれば → コンパイルエラー（静的型付け）
                  → 実行時エラー（動的型付け）
```

### 1.4 型の分類体系

プログラミング言語における型は、以下のように分類できる。

```
型の分類体系:

  プリミティブ型 (Primitive Types)
  ├── 数値型: int, float, double, decimal
  ├── 文字型: char, string
  ├── 論理型: bool
  └── 特殊型: void, unit, never

  複合型 (Composite Types)
  ├── 直積型 (Product Types): struct, tuple, record
  ├── 直和型 (Sum Types): enum, union, variant
  ├── 関数型 (Function Types): (A) -> B
  └── 参照型 (Reference Types): &T, *T, Box<T>

  パラメトリック型 (Parametric Types)
  ├── ジェネリクス: List<T>, Map<K, V>
  ├── 型制約: T extends Comparable<T>
  └── 高カインド型: F[_], Monad[F[_]]

  特殊型 (Special Types)
  ├── トップ型: any (TS), Object (Java), Any (Kotlin)
  ├── ボトム型: never (TS), Nothing (Kotlin/Scala)
  ├── ユニット型: void (C/Java), () (Rust/Haskell)
  └── null許容型: T? (Kotlin), Option<T> (Rust)
```

---

## 2. 静的型付け（Static Typing）--- コンパイル時の安全保証

### 2.1 静的型付けの定義と原理

静的型付けとは、プログラムの実行前（コンパイル時）に全ての式・変数・関数の型を決定し、
型の整合性を検証する方式である。

```
静的型付けのパイプライン:

  ソースコード
      │
      ▼
  ┌──────────────┐
  │  字句解析     │  ソースコードをトークンに分割
  │  (Lexing)     │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  構文解析     │  トークンを抽象構文木(AST)に変換
  │  (Parsing)    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │  型チェック (Type Checking)                    │  ← ★ ここで型エラーを検出
  │                                                │
  │  - 変数の型と代入値の型の一致を検証             │
  │  - 関数の引数・戻り値の型の一致を検証           │
  │  - 式の型の整合性を検証                         │
  │  - ジェネリクスの型パラメータを解決             │
  │  - 型推論による型の自動決定                     │
  └──────┬───────────────────────────────────────┘
         │ 型エラーがあれば → コンパイル停止、エラー報告
         │ 型エラーがなければ ↓
         ▼
  ┌──────────────┐
  │  コード生成   │  機械語・バイトコードへ変換
  │  (Codegen)    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  実行         │  型安全が保証された状態で実行
  │  (Execution)  │
  └──────────────┘
```

### 2.2 代表的な静的型付け言語の比較

#### Java --- 名目的型付け（Nominal Typing）の代表

```java
// Java: 名目的型付け --- 型の名前（宣言）で互換性を判定

interface Printable {
    String format();
}

// Printable を明示的に implements しなければ、
// format() メソッドがあっても Printable として扱えない
class Invoice implements Printable {
    private double amount;
    private String customer;

    public Invoice(double amount, String customer) {
        this.amount = amount;
        this.customer = customer;
    }

    @Override
    public String format() {
        return String.format("Invoice: %s - $%.2f", customer, amount);
    }
}

class Report {
    // format() メソッドがあるが、Printable を implements していない
    public String format() {
        return "Monthly Report";
    }
}

public class Main {
    static void print(Printable p) {
        System.out.println(p.format());
    }

    public static void main(String[] args) {
        print(new Invoice(100.0, "Alice"));  // OK: Invoice は Printable
        // print(new Report());               // コンパイルエラー!
        // Report は format() を持つが、Printable を implements していない
    }
}
```

#### Rust --- 所有権と型システムの融合

```rust
// Rust: 型システムと所有権システムが連携してメモリ安全性を保証

use std::fmt;

// トレイトによる型の振る舞い定義
trait Summary {
    fn summarize(&self) -> String;

    // デフォルト実装も可能
    fn preview(&self) -> String {
        format!("{}...", &self.summarize()[..20.min(self.summarize().len())])
    }
}

struct Article {
    title: String,
    author: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{} by {}", self.title, self.author)
    }
}

// ジェネリクスとトレイト境界
fn notify<T: Summary + fmt::Display>(item: &T) {
    println!("Breaking news: {}", item.summarize());
}

// Result型によるエラー処理 --- エラーの可能性が型で表現される
fn parse_config(path: &str) -> Result<Config, ConfigError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| ConfigError::IoError(e))?;  // ? 演算子でエラー伝播

    let config: Config = toml::from_str(&content)
        .map_err(|e| ConfigError::ParseError(e))?;

    Ok(config)
}

// 呼び出し側は Result を処理する義務がある
fn main() {
    match parse_config("config.toml") {
        Ok(config) => println!("Config loaded: {:?}", config),
        Err(e) => eprintln!("Failed to load config: {}", e),
    }
    // Result を無視すると #[must_use] 警告が出る
}
```

#### Go --- 構造的型付け（Structural Typing）の実践

```go
// Go: 構造的型付け --- 型の構造（メソッドセット）で互換性を判定

package main

import "fmt"

// インターフェースの定義
type Writer interface {
    Write(data []byte) (int, error)
}

// FileWriter は Writer インターフェースを「暗黙的に」実装
// implements キーワードは不要
type FileWriter struct {
    Path string
}

func (fw FileWriter) Write(data []byte) (int, error) {
    fmt.Printf("Writing %d bytes to %s\n", len(data), fw.Path)
    return len(data), nil
}

// ConsoleWriter も Writer インターフェースを暗黙的に実装
type ConsoleWriter struct{}

func (cw ConsoleWriter) Write(data []byte) (int, error) {
    fmt.Print(string(data))
    return len(data), nil
}

// Writer インターフェースを満たす任意の型を受け取れる
func process(w Writer, message string) {
    w.Write([]byte(message))
}

func main() {
    file := FileWriter{Path: "/tmp/output.txt"}
    console := ConsoleWriter{}

    process(file, "Hello, File!")       // OK
    process(console, "Hello, Console!") // OK
    // どちらも Write メソッドを持つため Writer として扱える
}
```

#### TypeScript --- 構造的部分型付け（Structural Subtyping）

```typescript
// TypeScript: 構造的部分型 --- プロパティの構造で互換性を判定

interface Point2D {
    x: number;
    y: number;
}

interface Point3D {
    x: number;
    y: number;
    z: number;
}

function distanceFromOrigin(point: Point2D): number {
    return Math.sqrt(point.x ** 2 + point.y ** 2);
}

const p3d: Point3D = { x: 3, y: 4, z: 5 };

// Point3D は Point2D の全プロパティを含むため、互換性がある
// （構造的部分型: Point3D <: Point2D）
const dist = distanceFromOrigin(p3d);  // OK: 5

// 型エイリアスとインターフェースの使い分け
type Result<T> =
    | { success: true; data: T }
    | { success: false; error: string };

function fetchUser(id: number): Result<{ name: string; email: string }> {
    if (id <= 0) {
        return { success: false, error: "Invalid ID" };
    }
    return {
        success: true,
        data: { name: "Alice", email: "alice@example.com" }
    };
}

// 判別共用体（Discriminated Union）による安全なパターンマッチ
const result = fetchUser(1);
if (result.success) {
    // TypeScript はここで result.data が存在することを推論
    console.log(result.data.name);
} else {
    // ここでは result.error が存在することを推論
    console.log(result.error);
}
```

### 2.3 静的型付けの利点と限界

```
+------------------------------------------------------------------+
|               静的型付けの利点（詳細分析）                          |
+------------------------------------------------------------------+
|                                                                    |
| 1. コンパイル時バグ検出                                            |
|    - 型不一致エラーの早期発見                                      |
|    - null参照の防止（Kotlin, Rust, Swift）                         |
|    - 網羅性チェック（exhaustiveness check）                        |
|    - 到達不能コードの検出                                          |
|                                                                    |
| 2. 開発体験の向上                                                  |
|    - IDE の正確な自動補完                                          |
|    - 安全なリファクタリング（リネーム、メソッド抽出等）             |
|    - Go to Definition / Find All References                        |
|    - インラインエラー表示とクイックフィックス                       |
|                                                                    |
| 3. ドキュメント効果                                                |
|    - 関数シグネチャが仕様書として機能                              |
|    - 型定義がドメインモデルを表現                                  |
|    - 新メンバーのオンボーディングを加速                            |
|                                                                    |
| 4. パフォーマンス最適化                                            |
|    - メモリレイアウトのコンパイル時決定                            |
|    - 仮想関数呼び出しの脱仮想化（devirtualization）               |
|    - 型特殊化（monomorphization in Rust）                          |
|    - 不要なボクシング/アンボクシングの排除                         |
|                                                                    |
| 5. 大規模開発への適性                                              |
|    - モジュール間の型契約による安全な分業                          |
|    - コンパイラが「型の門番」として機能                            |
|    - CI/CD パイプラインでの自動型チェック                          |
|                                                                    |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|               静的型付けの限界（詳細分析）                          |
+------------------------------------------------------------------+
|                                                                    |
| 1. 型注釈のオーバーヘッド                                          |
|    - 特にジェネリクス・高階型の記述は煩雑になりうる               |
|    - 型推論が緩和するが、完全には排除できない                      |
|                                                                    |
| 2. 表現力の限界                                                    |
|    - 動的な構造（JSON, 辞書型データ）の扱いが煩雑                  |
|    - メタプログラミングの制約                                      |
|    - ダックタイピング的パターンの表現が困難な場合がある            |
|                                                                    |
| 3. コンパイル時間                                                  |
|    - 大規模プロジェクトではコンパイル待ちが開発体験を損なう        |
|    - Rust のコンパイル時間は特に問題視される                       |
|    - インクリメンタルコンパイルで緩和するが限界あり               |
|                                                                    |
| 4. 学習曲線                                                        |
|    - ジェネリクス、分散、型クラス等の高度な概念                    |
|    - 型エラーメッセージの理解が困難な場合がある                    |
|    - 特に Haskell, Rust の型システムは習熟に時間を要する          |
|                                                                    |
+------------------------------------------------------------------+
```

---

## 3. 動的型付け（Dynamic Typing）--- 実行時の柔軟性

### 3.1 動的型付けの定義と原理

動的型付けとは、変数や式の型を実行時に決定し、操作の適用時に型の整合性を検証する方式である。
変数は型を持たず、値が型を持つ。

```
動的型付けのパイプライン:

  ソースコード
      │
      ▼
  ┌──────────────────────────────────────────────┐
  │  インタプリタ / JIT コンパイラ                  │
  │                                                │
  │  各ステートメントを順次実行:                    │
  │                                                │
  │  x = 42          → x に整数オブジェクト42を束縛│
  │  x = "hello"     → x に文字列オブジェクトを再束│
  │  y = x + " world"                              │
  │    ↓                                           │
  │  1. x の現在の型を調べる → str                  │
  │  2. " world" の型を調べる → str                │
  │  3. str + str の演算が定義されているか確認       │
  │  4. 定義されている → 実行                       │
  │                                                │
  │  z = x + 42                                    │
  │    ↓                                           │
  │  1. x の現在の型を調べる → str                  │
  │  2. 42 の型を調べる → int                      │
  │  3. str + int の演算が定義されているか確認       │
  │  4. 定義されていない → TypeError ★             │
  │                                                │
  └──────────────────────────────────────────────┘
```

### 3.2 「変数に型がない」とはどういうことか

動的型付けの本質を理解するには、「変数」と「値」の関係を正しく把握する必要がある。

```
静的型付け:                     動的型付け:
  変数は「型付きの箱」            変数は「ラベル（名札）」

  ┌─────────┐                   x ──→ 42 (int)
  │ int: 42  │ ← x                    ↓ 再代入
  └─────────┘                   x ──→ "hello" (str)
  x = "hello" → コンパイルエラー       ↓ 再代入
                                x ──→ [1,2,3] (list)

  変数 x は int 型の値のみ格納可能    変数 x は任意の型の値を
                                     指し示すことができる
```

### 3.3 代表的な動的型付け言語の比較

#### Python --- 強い動的型付け

```python
# Python: 強い動的型付け + ダックタイピング

# 同一変数に異なる型の値を代入可能
x = 42         # int
x = "hello"    # str（再代入OK、型チェックなし）
x = [1, 2, 3]  # list

# ダックタイピングの例
class Duck:
    def quack(self):
        return "Quack!"

    def swim(self):
        return "Swimming..."

class Person:
    def quack(self):
        return "I'm quacking like a duck!"

    def swim(self):
        return "I'm swimming like a duck!"

class RubberDuck:
    def quack(self):
        return "Squeak!"
    # swim() メソッドがない

def perform_duck_actions(duck):
    """duck 引数が quack() と swim() を持てば動作する"""
    print(duck.quack())
    print(duck.swim())

perform_duck_actions(Duck())    # OK
perform_duck_actions(Person())  # OK（ダックタイピング）
# perform_duck_actions(RubberDuck())  # AttributeError: swim() がない
# ↑ この問題は実行時にしか検出できない


# 動的型付けの威力: デコレータ
import functools
import time

def timer(func):
    """任意の関数の実行時間を計測するデコレータ"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function(n):
    return sum(range(n))

@timer
def slow_string(s, repeat):
    return s * repeat

# 型を問わず、あらゆる関数に適用可能
slow_function(1_000_000)
slow_string("hello", 100)
```

#### Ruby --- オブジェクト指向と動的型付けの融合

```ruby
# Ruby: 全てがオブジェクト + 強い動的型付け

# オープンクラス: 既存クラスにメソッドを追加可能
class Integer
  def factorial
    return 1 if self <= 1
    self * (self - 1).factorial
  end

  def prime?
    return false if self < 2
    (2..Math.sqrt(self).to_i).none? { |i| self % i == 0 }
  end
end

puts 5.factorial   # => 120
puts 7.prime?      # => true

# method_missing によるメタプログラミング
class DynamicConfig
  def initialize(data = {})
    @data = data
  end

  def method_missing(name, *args)
    key = name.to_s
    if key.end_with?('=')
      @data[key.chomp('=')] = args.first
    elsif @data.key?(key)
      @data[key]
    else
      super
    end
  end

  def respond_to_missing?(name, include_private = false)
    @data.key?(name.to_s.chomp('=')) || super
  end
end

config = DynamicConfig.new
config.database = "postgresql"  # method_missing で動的にセッター呼び出し
config.port = 5432
puts config.database            # => "postgresql"
puts config.port                # => 5432
```

#### JavaScript --- プロトタイプベースと動的型付け

```javascript
// JavaScript: プロトタイプベース + 弱い動的型付け

// プロトタイプチェーン
const animal = {
    type: "Animal",
    describe() {
        return `I am a ${this.type}`;
    }
};

const dog = Object.create(animal);
dog.type = "Dog";
dog.bark = function() { return "Woof!"; };

console.log(dog.describe()); // "I am a Dog" (プロトタイプのメソッドを継承)
console.log(dog.bark());     // "Woof!"

// 動的プロパティの追加と削除
const user = { name: "Alice" };
user.age = 30;           // プロパティの動的追加
user.greet = function() { return `Hi, I'm ${this.name}`; };
delete user.age;          // プロパティの動的削除

// typeof の落とし穴
console.log(typeof null);       // "object" (歴史的バグ)
console.log(typeof undefined);  // "undefined"
console.log(typeof NaN);        // "number" (NaN は number 型)
console.log(typeof []);         // "object" (配列もオブジェクト)
```

### 3.4 動的型付けの利点と限界

```
+------------------------------------------------------------------+
|               動的型付けの利点（詳細分析）                          |
+------------------------------------------------------------------+
|                                                                    |
| 1. 開発速度                                                        |
|    - 型注釈不要で記述量が少ない                                    |
|    - プロトタイピングが高速                                        |
|    - REPL で対話的にコードを試行錯誤可能                          |
|    - スクリプティング・自動化に最適                                |
|                                                                    |
| 2. 柔軟性                                                          |
|    - ダックタイピングによる多態性                                  |
|    - メタプログラミング（eval, method_missing, __getattr__）       |
|    - 動的なオブジェクト構築（JSON, API応答の処理）                 |
|    - ホットリロード・ライブコーディング                            |
|                                                                    |
| 3. 学習の容易さ                                                    |
|    - 型システムの概念を学ばずにプログラミングを開始可能            |
|    - エラーメッセージが直感的                                      |
|    - 初学者のハードルが低い                                        |
|                                                                    |
| 4. 表現力                                                          |
|    - DSL（ドメイン固有言語）の構築が容易                           |
|    - デコレータ/ミックスインパターンの自然な表現                   |
|    - マクロ的なコード生成が実行時に可能                            |
|                                                                    |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|               動的型付けの限界（詳細分析）                          |
+------------------------------------------------------------------+
|                                                                    |
| 1. 実行時エラーのリスク                                            |
|    - TypeError, AttributeError が本番環境で発生する可能性          |
|    - テストで全てのコードパスをカバーする必要がある                |
|    - 稀なコードパスのバグが長期間潜伏する危険                     |
|                                                                    |
| 2. リファクタリングの困難さ                                        |
|    - 関数名・変数名の変更で全ての呼び出し箇所を特定困難           |
|    - IDE のリネーム機能が不完全（動的呼び出しを追跡不能）         |
|    - 大規模な構造変更のリスクが高い                                |
|                                                                    |
| 3. パフォーマンスオーバーヘッド                                    |
|    - 実行時の型チェック・型情報の保持コスト                        |
|    - JIT コンパイラによる最適化は改善するが限界あり               |
|    - メモリ効率が静的型付け言語に劣る場合がある                   |
|                                                                    |
| 4. 大規模開発での課題                                              |
|    - モジュール間の型契約が暗黙的 → 誤解・不整合のリスク          |
|    - コードベースが大きくなると「読むコストの増加」               |
|    - チーム開発でのコミュニケーションコスト増大                    |
|                                                                    |
+------------------------------------------------------------------+
```

---

## 4. 強い型付け vs 弱い型付け --- 暗黙的型変換の境界線

### 4.1 定義と連続的スペクトラム

「強い型付け」と「弱い型付け」は、静的/動的とは独立した軸であり、
暗黙的な型変換（implicit type coercion）をどの程度許容するかの連続的スペクトラムである。

```
暗黙的型変換の許容度スペクトラム:

  厳格（強い型付け）                           寛容（弱い型付け）
  ←───────────────────────────────────────────────→

  Haskell  Rust  Python  Ruby  Java  C#  Go   C    JavaScript  PHP  Perl
  │        │     │       │     │     │   │    │    │           │    │
  │        │     │       │     │     │   │    │    │           │    │
  暗黙変換を     型が合わない   一部の      暗黙の   広範な暗黙的
  一切許さない   場合はエラー   暗黙変換    キャスト  型変換を許容
                               を許容       を許容

  ※ これは連続的なスペクトラムであり、二値分類ではない
  ※ 同じ言語でも文脈によって暗黙変換の度合いは異なる
```

### 4.2 暗黙的型変換の具体例

#### 強い型付けの例

```python
# Python: 強い型付け --- 暗黙的型変換をほぼ許さない

# 文字列 + 整数 → TypeError
try:
    result = "Age: " + 25
except TypeError as e:
    print(f"Error: {e}")
    # Error: can only concatenate str (not "int") to str

# 明示的な変換が必要
result = "Age: " + str(25)     # OK: "Age: 25"
result = f"Age: {25}"          # OK: f-string は内部で str() を呼ぶ

# 整数 + 浮動小数点 → 例外的に暗黙変換される（数値の拡大変換）
x = 1 + 2.5    # int + float → float (3.5)
# これは Python でも許容される暗黙変換（情報損失がないため）

# bool と int の関係（歴史的経緯）
print(True + 1)   # → 2（bool は int のサブクラス）
print(False + 1)  # → 1
# これは暗黙変換ではなく、継承関係による
```

```rust
// Rust: 最も厳格な型付けの一つ

fn main() {
    let x: i32 = 42;
    let y: i64 = 100;

    // let z = x + y;  // コンパイルエラー!
    // error: cannot add `i64` to `i32`
    // 整数型同士でもサイズが異なれば暗黙変換しない

    let z = x as i64 + y;  // 明示的キャスト必須
    let z = i64::from(x) + y;  // from トレイトによる安全な変換

    // 浮動小数点への変換も明示的
    let a: f64 = x as f64;  // 明示的キャスト必須
    // let b: f64 = x;      // コンパイルエラー!
}
```

#### 弱い型付けの例

```javascript
// JavaScript: 弱い型付け --- 広範な暗黙的型変換

// 加算演算子 + のオーバーロード
console.log("5" + 3);       // "53"  (数値→文字列に変換して連結)
console.log("5" - 3);       // 2     (文字列→数値に変換して減算)
console.log("5" * "3");     // 15    (両方を数値に変換して乗算)
console.log("5" + + "3");   // "53"  (単項+で数値化後、文字列連結)

// 比較演算子の暗黙変換
console.log(0 == "");        // true  (両方を数値に変換: 0 == 0)
console.log(0 == "0");       // true  (文字列→数値: 0 == 0)
console.log("" == "0");      // false (両方文字列: "" !== "0")
console.log(false == "0");   // true  (false→0, "0"→0: 0 == 0)
console.log(null == undefined);  // true  (特別ルール)
console.log(NaN == NaN);    // false (NaN は何とも等しくない)

// ===（厳密等価演算子）で暗黙変換を防ぐ
console.log(0 === "");       // false (型が違うので false)
console.log(0 === "0");      // false
console.log(false === "0");  // false

// 論理演算子の暗黙変換
console.log([] + {});        // "[object Object]"
console.log({} + []);        // 0 (ブラウザ依存)
console.log([] + []);        // ""
console.log(!![]);           // true (空配列は truthy)
console.log(!!0);            // false (0 は falsy)
```

```c
// C: 弱い型付け + 静的型付け

#include <stdio.h>

int main() {
    // 暗黙の型変換（整数の拡大変換）
    int x = 42;
    long y = x;          // int → long (暗黙拡大)
    double z = x;        // int → double (暗黙拡大)

    // 暗黙の型変換（データ損失の可能性あり）
    double pi = 3.14159;
    int truncated = pi;   // double → int: 3（切り捨て、警告のみ）

    // 暗黙のポインタ変換
    int *ip = &x;
    void *vp = ip;       // int* → void*（暗黙変換）
    char *cp = (char*)vp; // void* → char*（明示キャスト必要）

    // 整数とポインタの混同（危険）
    // int *bad = 0x12345678;  // 一部コンパイラで警告なしに通る

    // char と int の暗黙変換
    char c = 65;          // int → char: 'A'
    int ascii = 'A';      // char → int: 65
    printf("%c %d\n", c, ascii);  // A 65

    return 0;
}
```

### 4.3 型変換の分類と安全性

```
型変換の安全性分類:

  ┌─────────────────┬───────────────────┬──────────────┐
  │    変換の種類     │      例            │   安全性      │
  ├─────────────────┼───────────────────┼──────────────┤
  │ 拡大変換         │ int → long        │ 安全          │
  │ (Widening)       │ float → double    │ 情報損失なし  │
  ├─────────────────┼───────────────────┼──────────────┤
  │ 縮小変換         │ long → int        │ 危険          │
  │ (Narrowing)      │ double → float    │ 情報損失あり  │
  ├─────────────────┼───────────────────┼──────────────┤
  │ 意味的変換       │ "42" → 42        │ 条件付き安全  │
  │ (Semantic)       │ "hello" → ???     │ 失敗する場合  │
  ├─────────────────┼───────────────────┼──────────────┤
  │ 再解釈           │ int* → char*     │ 非常に危険    │
  │ (Reinterpret)    │ float → int (bit) │ バグの温床    │
  └─────────────────┴───────────────────┴──────────────┘
```

### 4.4 分類マトリクス --- 2軸4象限

```
                 静的型付け                    動的型付け
           ┌─────────────────────────┬─────────────────────────┐
           │                         │                         │
  強い     │  Java, Rust, Go,        │  Python, Ruby,          │
  型付け   │  Haskell, Kotlin,       │  Elixir, Erlang,        │
           │  Swift, TypeScript,     │  Clojure                │
           │  Scala, F#, OCaml       │                         │
           │                         │                         │
           │  特徴:                  │  特徴:                  │
           │  - 最も安全             │  - 柔軟だが型安全       │
           │  - コンパイル時検出     │  - 暗黙変換しない       │
           │  - IDE サポート充実     │  - テスト重要           │
           │                         │                         │
           ├─────────────────────────┼─────────────────────────┤
           │                         │                         │
  弱い     │  C, C++                 │  JavaScript, PHP,       │
  型付け   │                         │  Perl, Lua              │
           │                         │                         │
           │  特徴:                  │  特徴:                  │
           │  - 高速だが危険         │  - 最も柔軟             │
           │  - 暗黙キャストが罠     │  - 予測困難な挙動       │
           │  - メモリ安全性の課題   │  - 暗黙変換が多い       │
           │                         │                         │
           └─────────────────────────┴─────────────────────────┘
```

### 4.5 比較表: 各象限の言語特性

| 特性 | 静的+強い (Rust) | 静的+弱い (C) | 動的+強い (Python) | 動的+弱い (JS) |
|------|:---:|:---:|:---:|:---:|
| コンパイル時型チェック | あり | あり | なし | なし |
| 暗黙的型変換 | ほぼなし | 多い | ほぼなし | 非常に多い |
| null安全性 | Option型で保証 | null参照あり | None あり | null/undefined |
| メモリ安全性 | 所有権で保証 | 手動管理 | GC | GC |
| 実行速度 | 非常に高速 | 非常に高速 | 中程度 | 中程度(JIT) |
| 開発速度 | 中程度 | 低い | 高い | 高い |
| 大規模開発適性 | 非常に高い | 中程度 | 中程度 | 低い(TSなしの場合) |
| 学習曲線 | 急峻 | 急峻 | 緩やか | 緩やか(罠が多い) |
| エラー検出時点 | コンパイル時 | コンパイル時(一部) | 実行時 | 実行時 |
| 型推論 | 強力 | 限定的 | なし(型ヒント) | なし(TSあり) |

---

## 5. 段階的型付け（Gradual Typing）--- 動的から静的への架け橋

### 5.1 段階的型付けの理論的背景

段階的型付け（Gradual Typing）は、2006年にJeremy SiekとWalid Tahaによって提唱された概念で、
動的型付け言語と静的型付け言語の利点を一つの言語内で共存させるアプローチである。

核心的アイデアは「型注釈のある部分は静的にチェックし、ない部分は動的に扱う」という点にある。

```
段階的型付けの動作原理:

  ソースコード（型注釈あり/なし混在）
      │
      ▼
  ┌────────────────────────────────────────────────┐
  │  型チェッカー（mypy, Pyright, tsc など）         │
  │                                                  │
  │  def greet(name: str) -> str:  ← 型注釈あり      │
  │      return f"Hello, {name}!"   → 静的チェック    │
  │                                                  │
  │  def process(data):            ← 型注釈なし      │
  │      return data.value          → チェック省略    │
  │                                 （any / unknown） │
  │                                                  │
  │  結果:                                           │
  │    型注釈あり部分 → コンパイル時にエラー検出      │
  │    型注釈なし部分 → 実行時に通常通り動的チェック  │
  └────────────────────────────────────────────────┘
      │
      ▼
  段階的に型注釈を追加 → 型カバレッジが向上 → 安全性が向上
```

### 5.2 Python の型ヒント（PEP 484 以降）

Python の型ヒントは、段階的型付けの最も成功した実装例の一つである。

```python
# Python: 段階的型付けの実践

# === Phase 1: 型注釈なし（従来の動的型付け）===
def calculate_total(items, tax_rate):
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    return subtotal * (1 + tax_rate)


# === Phase 2: 基本的な型注釈を追加 ===
from typing import TypedDict

class Item(TypedDict):
    name: str
    price: float
    quantity: int

def calculate_total(items: list[Item], tax_rate: float) -> float:
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    return subtotal * (1 + tax_rate)


# === Phase 3: より厳密な型定義 ===
from typing import TypedDict, NewType
from decimal import Decimal

Price = NewType('Price', Decimal)   # 価格専用の型
TaxRate = NewType('TaxRate', Decimal)  # 税率専用の型

class StrictItem(TypedDict):
    name: str
    price: Price
    quantity: int

def calculate_total_strict(
    items: list[StrictItem],
    tax_rate: TaxRate
) -> Decimal:
    subtotal = sum(
        item['price'] * item['quantity']
        for item in items
    )
    return subtotal * (1 + tax_rate)

# NewType により Price と TaxRate を混同するとmypyがエラーを報告
# price = Price(Decimal("100.00"))
# rate = TaxRate(Decimal("0.10"))
# bad = price + rate  # mypy error: Price と TaxRate は直接加算不可


# === 高度な型ヒント機能 ===
from typing import (
    Generic, TypeVar, Protocol, overload,
    Literal, Union, Optional
)

# ジェネリクス
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class Cache(Generic[K, V]):
    """型安全なキャッシュ"""
    def __init__(self, max_size: int = 100) -> None:
        self._data: dict[K, V] = {}
        self._max_size = max_size

    def get(self, key: K) -> Optional[V]:
        return self._data.get(key)

    def set(self, key: K, value: V) -> None:
        if len(self._data) >= self._max_size:
            oldest = next(iter(self._data))
            del self._data[oldest]
        self._data[key] = value

# 使用時に型パラメータが推論される
cache: Cache[str, int] = Cache(max_size=50)
cache.set("count", 42)    # OK
# cache.set("count", "42")  # mypy error: str は int に代入不可

# Protocol（構造的部分型）
class Renderable(Protocol):
    def render(self) -> str: ...

class HtmlElement:
    def render(self) -> str:
        return "<div>Hello</div>"

class MarkdownText:
    def render(self) -> str:
        return "**Hello**"

def display(item: Renderable) -> None:
    print(item.render())

# HtmlElement も MarkdownText も render() を持つため Renderable として扱える
# Protocol は「ダックタイピングを型で表現する」仕組み
display(HtmlElement())    # OK
display(MarkdownText())   # OK
```

### 5.3 TypeScript の段階的型付け戦略

```typescript
// TypeScript: JavaScript からの段階的移行戦略

// === Step 1: .js → .ts にリネーム + allowJs ===
// tsconfig.json:
// {
//   "compilerOptions": {
//     "allowJs": true,
//     "checkJs": false,
//     "strict": false,
//     "outDir": "./dist"
//   }
// }

// === Step 2: any を使って型エラーを一時的に回避 ===
// 移行初期: 型が不明な箇所に any を使う
function processLegacyData(data: any): any {
    return data.items.map((item: any) => item.value);
}

// === Step 3: any を unknown に置き換え ===
// unknown は any より安全（型ガードが必須）
function processData(data: unknown): string[] {
    if (
        typeof data === "object" &&
        data !== null &&
        "items" in data &&
        Array.isArray((data as { items: unknown[] }).items)
    ) {
        const items = (data as { items: Array<{ value: string }> }).items;
        return items.map(item => item.value);
    }
    throw new Error("Invalid data format");
}

// === Step 4: 適切な型定義を作成 ===
interface DataItem {
    id: number;
    value: string;
    metadata?: Record<string, unknown>;
}

interface DataPayload {
    items: DataItem[];
    total: number;
    page: number;
}

function processTypedData(data: DataPayload): string[] {
    return data.items.map(item => item.value);
}

// === Step 5: strict モードを有効化 ===
// tsconfig.json:
// {
//   "compilerOptions": {
//     "strict": true,
//     "noImplicitAny": true,
//     "strictNullChecks": true,
//     "strictFunctionTypes": true,
//     "strictBindCallApply": true,
//     "noImplicitReturns": true,
//     "noFallthroughCasesInSwitch": true,
//     "exactOptionalPropertyTypes": true
//   }
// }

// strict モード下での安全なコード例
function findUser(
    users: readonly User[],
    id: number
): User | undefined {
    return users.find(user => user.id === id);
}

// strictNullChecks 下では undefined の可能性を処理しないとエラー
const user = findUser(users, 42);
// console.log(user.name);  // Error: Object is possibly 'undefined'

// 正しい処理
if (user !== undefined) {
    console.log(user.name);  // OK: ここでは user は User 型
}

// Optional chaining + Nullish coalescing
const name = user?.name ?? "Unknown";
```

### 5.4 段階的型付けの移行戦略比較

| 移行戦略 | Python (mypy) | TypeScript | PHP (8.0+) |
|----------|:---:|:---:|:---:|
| 型注釈の強制 | 任意 | 任意(strictで強制) | 任意(一部強制可) |
| 実行時の型チェック | なし(ヒントのみ) | なし(コンパイル時のみ) | あり(declare strict) |
| any相当の型 | Any | any | mixed |
| 段階的厳格化 | mypy --strict | strict: true | declare(strict_types=1) |
| 型カバレッジ計測 | mypy --html-report | tsc --noEmit | PHPStan level |
| 構造的部分型 | Protocol | interface | なし(名目的) |
| ジェネリクス | Generic[T] | <T> | テンプレート(限定) |
| null安全 | Optional[T] | strictNullChecks | ?Type |
| 移行の漸進性 | ファイル単位 | ファイル単位 | ファイル単位 |
| エコシステム対応 | typeshed, stubs | DefinitelyTyped | PHPStan, Psalm |

---

## 6. 型システムの健全性と完全性

### 6.1 健全性（Soundness）の定義

型システムの健全性とは、「型チェックを通過したプログラムは、実行時に型エラーを起こさない」
という性質である。形式的には以下のように表現される。

```
健全性の形式的定義:

  もし Gamma |- e : T（環境 Gamma の下で式 e が型 T を持つ）ならば、
  e を評価した結果は型 T の値であるか、停止しないか、
  明示的に許可された例外を投げる。

  直感的に:
    「型チェッカーが "安全" と言ったプログラムは、本当に安全」

  対偶:
    「実行時に型エラーが起きるプログラムは、型チェックを通らない」
```

### 6.2 完全性（Completeness）の定義

型システムの完全性とは、「型的に安全な全てのプログラムが型チェックを通過する」という性質である。

```
完全性の形式的定義:

  もし式 e が実行時に型エラーを起こさないならば、
  Gamma |- e : T となる型 T が存在する。

  直感的に:
    「型チェッカーが "危険" と言ったプログラムの中に、
     実際には安全なものが含まれていない」

  注意: 完全な型システムは、原理的に実現不可能な場合が多い
    （停止問題の帰結として）
```

### 6.3 健全性と完全性のトレードオフ

```
健全性と完全性のトレードオフ:

  ┌──────────────────────────────────────────────┐
  │        全てのプログラムの集合                  │
  │                                                │
  │   ┌──────────────────────────────┐             │
  │   │  型的に安全なプログラム       │             │
  │   │                              │             │
  │   │   ┌──────────────────┐       │             │
  │   │   │ 型チェック通過    │       │             │
  │   │   │ (健全な型システム) │       │             │
  │   │   └──────────────────┘       │             │
  │   │                              │             │
  │   │   ※ この隙間が「偽陽性」     │             │
  │   │   (安全だが型チェック不通過)  │             │
  │   └──────────────────────────────┘             │
  │                                                │
  │   ※ 外側は「型的に安全でないプログラム」        │
  │   健全であれば、これらは型チェックを通らない    │
  └──────────────────────────────────────────────┘

  健全 + 完全:  理想的だが多くの場合実現不可能
  健全 + 不完全: 安全だが一部の正しいプログラムを拒否（Rust, Haskell）
  不健全 + 完全: 全て通すが安全性なし（型なし言語と同等）
  不健全 + 不完全: 一部の安全でないプログラムも通す（TypeScript）
```

### 6.4 各言語の健全性の程度

```
健全性のスペクトラム（言語別）:

  完全に健全                             意図的に不健全
  ←────────────────────────────────────────→

  Haskell  Rust    OCaml   Java    C#    TypeScript  C/C++
  │        │       │       │       │     │           │
  │        │       │       │       │     │           │
  unsafe   unsafe  Obj.    型消去  動的   any,        void*,
  なしで   を除き  magic   (raw   キャス as,         reinterpret
  健全     健全    除き    type)  ト      !           _cast
                   健全    で不          (non-null
                          健全          assertion)
```

```rust
// Rust: 安全なコードは健全、unsafe ブロックで健全性の「穴」を作れる

// 安全なコード --- 型システムが健全性を保証
fn safe_example() {
    let v: Vec<i32> = vec![1, 2, 3];
    // v[10];  // パニック（境界チェックあり）だが型エラーではない

    let x: Option<i32> = Some(42);
    // x + 1;  // コンパイルエラー: Option<i32> に + は定義されていない
    let y = x.unwrap() + 1;  // OK: unwrap で i32 を取り出してから加算
}

// unsafe ブロック --- 開発者が安全性を保証する責任を負う
fn unsafe_example() {
    let x: i32 = 42;
    let ptr = &x as *const i32;

    unsafe {
        // 生ポインタのデリファレンス --- 開発者の責任
        let value = *ptr;
        println!("Value: {}", value);

        // unsafe の典型的な用途:
        // 1. 生ポインタのデリファレンス
        // 2. unsafe 関数の呼び出し（FFI等）
        // 3. ミュータブルな静的変数へのアクセス
        // 4. unsafe トレイトの実装
    }
}
```

```typescript
// TypeScript: 意図的に不健全な箇所がある

// 1. any 型 --- 型チェックを完全に無効化
let data: any = "hello";
data.nonExistentMethod();  // コンパイルOK、実行時エラー

// 2. 型アサーション --- 開発者が型を強制
interface User {
    name: string;
    email: string;
}
const raw: unknown = {};
const user = raw as User;      // コンパイルOK
console.log(user.name);        // undefined（実行時）

// 3. non-null assertion --- null/undefined でないと断言
function getLength(s: string | undefined): number {
    return s!.length;  // s が undefined なら実行時エラー
}

// 4. 共変配列 --- TypeScript の設計上の妥協
const dogs: Dog[] = [new Dog()];
const animals: Animal[] = dogs;  // OK（共変）
animals.push(new Cat());         // コンパイルOK!
// dogs[1] は Cat だが Dog[] の要素として扱われる（不健全）
```

---

## 7. 名目的型付け vs 構造的型付け

### 7.1 名目的型付け（Nominal Typing）

名目的型付けでは、型の互換性は型の名前（宣言）によって決定される。
同じ構造を持っていても、別の名前で宣言された型は互換性がない。

```java
// Java: 名目的型付けの典型例

class Meter {
    private final double value;
    Meter(double value) { this.value = value; }
    double getValue() { return value; }
}

class Kilogram {
    private final double value;
    Kilogram(double value) { this.value = value; }
    double getValue() { return value; }
}

// Meter と Kilogram は同じ構造だが、互換性なし
void processDistance(Meter m) {
    System.out.println("Distance: " + m.getValue() + "m");
}

// processDistance(new Kilogram(5.0));  // コンパイルエラー!
// Kilogram は Meter ではない（名前が異なる）
// → 単位の混同を型レベルで防止できる
// （NASAの火星探査機が単位混同で墜落した事例を防げる設計）
```

### 7.2 構造的型付け（Structural Typing）

構造的型付けでは、型の互換性は型の構造（持っているプロパティやメソッド）によって決定される。

```typescript
// TypeScript: 構造的型付けの典型例

interface HasName {
    name: string;
}

interface HasAge {
    age: number;
}

interface Person {
    name: string;
    age: number;
    email: string;
}

function greetByName(entity: HasName): string {
    return `Hello, ${entity.name}!`;
}

const person: Person = { name: "Alice", age: 30, email: "a@b.com" };
const company = { name: "Acme Corp", founded: 1990 };

// どちらも name プロパティを持つため HasName として扱える
greetByName(person);   // OK
greetByName(company);  // OK（構造的に互換）

// Go も構造的型付けを採用
// Go では「暗黙的にインターフェースを満たす」
```

### 7.3 名目的 vs 構造的の比較

```
名目的型付け vs 構造的型付けの比較:

  ┌────────────────┬──────────────────────┬──────────────────────┐
  │     観点        │   名目的型付け        │   構造的型付け        │
  ├────────────────┼──────────────────────┼──────────────────────┤
  │ 互換性の判定    │ 型の名前・宣言        │ 型の構造・形状        │
  │ 明示性          │ implements/extends   │ 暗黙的に満たす        │
  │                │ の明示宣言が必要      │                      │
  │ 安全性          │ 偶然の一致を防止      │ 偶然の一致がありうる  │
  │ 柔軟性          │ 低い（事前宣言必須）  │ 高い（後付け可能）    │
  │ リファクタリング │ 明示的で追跡しやすい  │ 暗黙的で影響範囲不明  │
  │ 代表言語        │ Java, C#, Kotlin     │ TypeScript, Go       │
  │ ユースケース    │ ドメインモデリング    │ アダプタ/連携         │
  └────────────────┴──────────────────────┴──────────────────────┘
```

---

## 8. アンチパターンと落とし穴

### 8.1 アンチパターン1: any型の乱用（TypeScript）

TypeScript において `any` 型を安易に使うことは、型システムの恩恵を完全に放棄することに等しい。

```typescript
// --- アンチパターン: any の乱用 ---

// BAD: API レスポンスを any で受ける
async function fetchUsers(): Promise<any> {
    const response = await fetch("/api/users");
    return response.json();  // any が返る
}

async function displayUsers() {
    const users = await fetchUsers();
    // any なので何でもアクセスできてしまう
    console.log(users.nonExistent.deeply.nested);  // コンパイルOK!
    // → 実行時に TypeError: Cannot read properties of undefined
}

// BAD: イベントハンドラで any
function handleEvent(event: any) {
    event.target.value.toUpperCase();  // コンパイルOK、実行時に壊れる可能性
}

// BAD: 型が面倒だから any
function processData(data: any): any {
    return data.map((item: any) => ({
        ...item,
        processed: true,
    }));
}

// --- 改善パターン ---

// GOOD: 適切な型定義
interface User {
    id: number;
    name: string;
    email: string;
    role: "admin" | "user" | "guest";
}

interface ApiResponse<T> {
    data: T;
    meta: { total: number; page: number };
}

async function fetchUsers(): Promise<ApiResponse<User[]>> {
    const response = await fetch("/api/users");
    const json: unknown = await response.json();
    return validateApiResponse(json);  // バリデーション付き
}

// GOOD: unknown + 型ガード
function isUser(value: unknown): value is User {
    return (
        typeof value === "object" &&
        value !== null &&
        "id" in value &&
        "name" in value &&
        "email" in value &&
        typeof (value as User).id === "number" &&
        typeof (value as User).name === "string"
    );
}

// GOOD: イベントハンドラの適切な型付け
function handleInputChange(event: React.ChangeEvent<HTMLInputElement>) {
    const value = event.target.value;  // string 型が保証される
    console.log(value.toUpperCase());
}

// GOOD: ジェネリクスで柔軟かつ型安全に
function processData<T>(data: T[]): (T & { processed: boolean })[] {
    return data.map(item => ({ ...item, processed: true }));
}
```

### 8.2 アンチパターン2: 暗黙的型変換への依存（JavaScript）

JavaScript の暗黙的型変換に依存したコードは、予測困難なバグの温床となる。

```javascript
// --- アンチパターン: 暗黙的型変換への依存 ---

// BAD: == による暗黙変換を利用した比較
function isEmptyValue(value) {
    return value == null;  // null と undefined の両方に一致
    // 意図は理解できるが、== の挙動は他の場面で罠になる
}

// BAD: 真偽値としての暗黙評価に依存
function getDisplayName(user) {
    return user.nickname || user.name || "Anonymous";
    // 問題: nickname が "" (空文字列) の場合、falsy なのでスキップされる
    // ユーザーが意図的に空文字を設定した場合の挙動が不正
}

// BAD: + 演算子の暗黙変換に依存
function calculateTotal(items) {
    let total = 0;
    for (const item of items) {
        total += item.price;  // item.price が文字列 "100" だと文字列結合になる
    }
    return total;
    // items = [{price: 10}, {price: "20"}]
    // → total = "1020" (文字列!) ではなく期待値は 30
}

// BAD: 配列のソートにおける暗黙変換
const numbers = [10, 1, 21, 2];
numbers.sort();           // [1, 10, 2, 21] (文字列として比較!)
// デフォルトの sort() は要素を文字列に変換して辞書順でソートする

// --- 改善パターン ---

// GOOD: 厳密等価演算子 === を使う
function isNullOrUndefined(value) {
    return value === null || value === undefined;
}

// GOOD: Nullish coalescing ?? を使う（ES2020+）
function getDisplayName(user) {
    return user.nickname ?? user.name ?? "Anonymous";
    // ?? は null と undefined のみをスキップ（"" や 0 はスキップしない）
}

// GOOD: 明示的な型変換
function calculateTotal(items) {
    let total = 0;
    for (const item of items) {
        const price = Number(item.price);
        if (Number.isNaN(price)) {
            throw new Error(`Invalid price: ${item.price}`);
        }
        total += price;
    }
    return total;
}

// GOOD: 比較関数を明示的に渡す
numbers.sort((a, b) => a - b);  // [1, 2, 10, 21] (数値として比較)
```

### 8.3 アンチパターン3: 型アサーションの過信（TypeScript / Java）

```typescript
// --- アンチパターン: 型アサーション (as) の過信 ---

// BAD: 外部データを型アサーションで無検証に変換
interface Config {
    host: string;
    port: number;
    ssl: boolean;
}

// 外部から来るデータを検証せずにアサーション
const config = JSON.parse(rawJson) as Config;
// rawJson が不正でも Config として扱われてしまう

// BAD: as unknown as T で型システムを完全にバイパス
const num = "hello" as unknown as number;
// コンパイルOK。num.toFixed(2) → 実行時エラー

// --- 改善パターン ---

// GOOD: Zod などのバリデーションライブラリを使用
import { z } from "zod";

const ConfigSchema = z.object({
    host: z.string().min(1),
    port: z.number().int().min(1).max(65535),
    ssl: z.boolean(),
});

type Config = z.infer<typeof ConfigSchema>;

function loadConfig(rawJson: string): Config {
    const parsed = JSON.parse(rawJson);
    return ConfigSchema.parse(parsed);
    // バリデーションに失敗すると ZodError が投げられる
}
```

---

## 9. 型システムの選択基準 --- プロジェクト特性に応じた判断フレームワーク

### 9.1 プロジェクト特性と型システムの適合性

```
プロジェクト特性に基づく型システム選択フレームワーク:

  ┌─────────────────────────────────────────────────────────┐
  │                 判断フロー                                │
  │                                                          │
  │  Q1: プロジェクトの規模は？                               │
  │    ├─ 小規模（〜5000行）→ 動的型付けでも十分管理可能     │
  │    ├─ 中規模（5000〜50000行）→ 段階的型付けが有効        │
  │    └─ 大規模（50000行〜）→ 静的型付けを強く推奨          │
  │                                                          │
  │  Q2: チームの規模は？                                     │
  │    ├─ 1-3人 → 動的型付けでも意思疎通可能                  │
  │    ├─ 4-10人 → 型による契約がコミュニケーションを補助     │
  │    └─ 10人以上 → 静的型付けがほぼ必須                     │
  │                                                          │
  │  Q3: ソフトウェアの寿命は？                               │
  │    ├─ 短期（プロトタイプ、PoC）→ 動的型付けが効率的      │
  │    ├─ 中期（1-3年）→ 段階的型付けで将来に備える          │
  │    └─ 長期（3年以上）→ 静的型付けが保守コストを低減      │
  │                                                          │
  │  Q4: 安全性の要件は？                                     │
  │    ├─ 通常（Webアプリ）→ TypeScript, Kotlin 等が適切     │
  │    ├─ 高い（金融、医療）→ Rust, Haskell, Java 等を推奨   │
  │    └─ 最高（航空宇宙、原子力）→ Ada/SPARK, 形式検証言語  │
  │                                                          │
  └─────────────────────────────────────────────────────────┘
```

### 9.2 ドメイン別推奨マトリクス

| ドメイン | 推奨型システム | 推奨言語 | 根拠 |
|----------|:---:|:---:|------|
| Web フロントエンド | 静的(段階的) | TypeScript | IDEサポート、大規模SPA管理 |
| Web バックエンド | 静的+強い | Go, Kotlin, Rust | 安全性、パフォーマンス |
| データサイエンス | 動的+強い | Python(+型ヒント) | ライブラリエコシステム、探索的開発 |
| システムプログラミング | 静的+強い | Rust, C++ | メモリ安全性、パフォーマンス |
| モバイルアプリ | 静的+強い | Kotlin, Swift | プラットフォームSDK、安全性 |
| スクリプティング/自動化 | 動的+強い | Python, Ruby | 開発速度、簡潔さ |
| 分散システム | 静的+強い | Go, Rust, Erlang/Elixir | 信頼性、並行処理安全性 |
| ゲーム開発 | 静的+強い | C#(Unity), C++(UE) | パフォーマンス、ツール統合 |
| CLI ツール | 静的+強い | Go, Rust | シングルバイナリ、クロスコンパイル |
| 教育/学習 | 動的+強い | Python | 学習曲線の緩やかさ |

### 9.3 型システムの進化トレンド

```
型システムの進化トレンド（2000年代〜現在）:

  2000年代前半
  │  Java 5: ジェネリクス導入
  │  C#: ジェネリクス（reified）
  │
  2006年
  │  Siek & Taha: 段階的型付けの理論提唱
  │
  2010年代前半
  │  TypeScript 0.8 (2012): JavaScript に静的型付けを追加
  │  Dart (2011): 段階的型付けをネイティブサポート
  │  Kotlin (2011): null安全を型システムに組み込み
  │  Rust 1.0 (2015): 所有権 + 借用 + ライフタイム
  │
  2014-2015年
  │  Python PEP 484: 型ヒントの標準化
  │  PHP 7.0: スカラー型宣言
  │  Swift (2014): Optional型 + パターンマッチ
  │
  2015年〜現在
  │  TypeScript: 条件型、テンプレートリテラル型、satisfies 演算子
  │  Python: TypedDict, Protocol, ParamSpec, TypeVarTuple
  │  Rust: GAT (Generic Associated Types), async trait
  │  Kotlin: コンテキストレシーバ、値クラス
  │
  現在のトレンド:
  │  - 段階的型付けの普及（動的→静的への移行パス）
  │  - 代数的データ型の主流化（パターンマッチ + Union型）
  │  - null安全の標準化（Option/Optional型）
  │  - 型レベルプログラミングの高度化
  │  - 依存型の実用化（Idris, Lean）
  │  - エフェクトシステムの研究と実用化
  ▼
```

---

## 10. 実践演習 --- 3段階の学習ステップ

### 演習1: [基礎] 型エラーの比較体験

**目的**: 静的型付けと動的型付けのエラー検出タイミングの違いを体感する。

**課題**: 以下の3つの言語で同一の型エラーを意図的に引き起こし、
エラーメッセージとその検出タイミングを記録・比較せよ。

```python
# Python (動的型付け) で実行する:

# テスト1: 型不一致の算術演算
def test_arithmetic():
    result = "100" + 50
    return result

# テスト2: 存在しないメソッド呼び出し
def test_method():
    x = 42
    return x.upper()

# テスト3: 引数の型不一致
def test_argument():
    items = [1, 2, 3]
    return items.append("not a number")  # 成功する（listは型制約なし）

# それぞれを実行し、エラーメッセージを記録すること
# 追加課題: mypy で同じコードを解析し、結果を比較すること
```

```typescript
// TypeScript (静的型付け) で実行する:

// テスト1: 型不一致の算術演算
function testArithmetic(): number {
    const result: number = "100" + 50;  // コンパイルエラー
    return result;
}

// テスト2: 存在しないメソッド呼び出し
function testMethod(): string {
    const x: number = 42;
    return x.upper();  // コンパイルエラー
}

// テスト3: 引数の型不一致
function testArgument(): void {
    const items: number[] = [1, 2, 3];
    items.push("not a number");  // コンパイルエラー
}

// tsc でコンパイルし、エラーメッセージを記録すること
```

```rust
// Rust (静的 + 強い型付け) で実行する:

// テスト1: 型不一致の算術演算
fn test_arithmetic() -> i32 {
    let result: i32 = "100" + 50;  // コンパイルエラー
    result
}

// テスト2: 存在しないメソッド呼び出し
fn test_method() -> String {
    let x: i32 = 42;
    x.upper()  // コンパイルエラー
}

// テスト3: 引数の型不一致
fn test_argument() {
    let mut items: Vec<i32> = vec![1, 2, 3];
    items.push("not a number");  // コンパイルエラー
}

// cargo build でコンパイルし、エラーメッセージを記録すること
```

**レポート項目**:
1. 各言語のエラーメッセージの内容と分かりやすさ
2. エラーが検出されるタイミング（コンパイル時 vs 実行時）
3. エラーメッセージから問題の原因を特定するまでの時間

### 演習2: [応用] 段階的型付けの実践 --- Python プロジェクトへの型ヒント追加

**目的**: 既存の動的型付けコードに段階的に型注釈を追加し、型カバレッジを向上させる。

**課題**: 以下の型注釈なしコードに型ヒントを追加し、mypy --strict で全チェックを通過させよ。

```python
# 型注釈なしの元コード（これに型ヒントを追加する）

from datetime import datetime, timedelta

class TaskStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Task:
    def __init__(self, title, description, due_date=None):
        self.title = title
        self.description = description
        self.due_date = due_date
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.tags = []

    def is_overdue(self):
        if self.due_date is None:
            return False
        return datetime.now() > self.due_date

    def add_tag(self, tag):
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag):
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False

class TaskManager:
    def __init__(self):
        self.tasks = []
        self.next_id = 1

    def add_task(self, title, description, due_date=None):
        task = Task(title, description, due_date)
        self.tasks.append(task)
        self.next_id += 1
        return task

    def get_overdue_tasks(self):
        return [t for t in self.tasks if t.is_overdue()]

    def get_tasks_by_tag(self, tag):
        return [t for t in self.tasks if tag in t.tags]

    def get_tasks_by_status(self, status):
        return [t for t in self.tasks if t.status == status]

    def summary(self):
        total = len(self.tasks)
        by_status = {}
        for task in self.tasks:
            by_status[task.status] = by_status.get(task.status, 0) + 1
        return {"total": total, "by_status": by_status}
```

**達成基準**:
- `mypy --strict` で 0 errors
- 全ての関数に引数型と戻り値型の注釈がある
- `Optional`, `list`, `dict` などの適切な型を使用
- `TaskStatus` を `Enum` または `Literal` 型で表現する（ボーナス）

### 演習3: [発展] TypeScript の型レベルプログラミング

**目的**: TypeScript の高度な型機能を活用して、型レベルでのバリデーションを実装する。

**課題**: 以下の要件を満たす型安全な API クライアントを TypeScript で実装せよ。

```typescript
// 要件:
// 1. API エンドポイントの定義を型で表現する
// 2. エンドポイントに応じたリクエスト/レスポンスの型を自動推論する
// 3. 存在しないエンドポイントへのアクセスをコンパイル時に防止する

// ヒント: 以下のような型定義から始める

// API スキーマの型定義
interface ApiSchema {
    "/users": {
        GET: { response: User[]; query: { page?: number; limit?: number } };
        POST: { response: User; body: { name: string; email: string } };
    };
    "/users/:id": {
        GET: { response: User; params: { id: string } };
        PUT: { response: User; params: { id: string }; body: Partial<User> };
        DELETE: { response: void; params: { id: string } };
    };
    "/posts": {
        GET: { response: Post[]; query: { authorId?: string } };
        POST: { response: Post; body: { title: string; content: string } };
    };
}

// 型安全な API クライアント（実装せよ）
// const api = createApiClient<ApiSchema>(baseUrl);
//
// 以下が型安全に動作すること:
// const users = await api.get("/users", { query: { page: 1 } });
//   → users の型は User[]
//
// const user = await api.post("/users", { body: { name: "Alice", email: "a@b.com" } });
//   → user の型は User
//
// 以下がコンパイルエラーになること:
// await api.get("/nonexistent");  // 存在しないエンドポイント
// await api.post("/users", { body: { name: 123 } });  // 型不一致
// await api.delete("/users");  // DELETE は /users/:id のみ
```

**達成基準**:
- `tsc --strict` で 0 errors
- Mapped Types, Conditional Types, Template Literal Types を活用
- 実行時のHTTPリクエストも正しく動作する

---

## 11. FAQ --- よくある質問と詳細回答

### Q1: 動的型付け言語でも大規模開発は可能か？

**A**: 可能だが、追加の規律とツールが必要になる。

動的型付け言語で大規模開発を成功させている例は多数存在する（Instagram/Python、Shopify/Ruby、
Netflix/Node.js 等）。ただし、以下の追加コストが発生する。

1. **テストカバレッジの要求が高くなる**: 静的型付けがコンパイル時に検出するエラーを、
   テストで代替する必要がある。一般に、動的型付けのプロジェクトでは 80%以上のテストカバレッジが
   推奨される。

2. **型ヒント/静的解析ツールの活用**: Python の mypy/Pyright、Ruby の Sorbet、
   JavaScript の Flow/TypeScript など、段階的に型を導入する仕組みが事実上必須となる。

3. **コーディング規約とレビュープロセスの厳格化**: 型情報がない分、命名規約、
   ドキュメント、コードレビューでの品質保証が重要になる。

4. **アーキテクチャの工夫**: マイクロサービス化によりサービス単位のコードベースを
   小さく保つ戦略が有効である。

結論として、「不可能ではないが、静的型付けが提供する安全性を別の手段で補完する必要がある」。
プロジェクトの規模が拡大するにつれ、段階的型付けの導入を強く推奨する。

### Q2: TypeScript の `any` と `unknown` の使い分けは？

**A**: 原則として `unknown` を使い、`any` は移行期間中の一時措置としてのみ許容する。

```
any vs unknown の比較:

  any:
  - 型チェックを完全に無効化する
  - あらゆる操作が許可される（型安全性ゼロ）
  - 他の型に代入可能、他の型から代入可能
  - TypeScript を使う意味がなくなる

  unknown:
  - 「何の型か分からない」ことを型レベルで表現
  - 操作する前に型ガード（型の絞り込み）が必須
  - 他の型への代入には明示的なチェックが必要
  - 型安全性を維持しながら柔軟性を確保

  使い分けの指針:
  ┌─────────────────────────────┬──────────────────────┐
  │ 状況                        │ 推奨                  │
  ├─────────────────────────────┼──────────────────────┤
  │ 外部APIのレスポンス          │ unknown + バリデーション│
  │ JSON.parse の結果            │ unknown + 型ガード    │
  │ JSライブラリの移行中          │ any（一時的措置）     │
  │ イベントハンドラの引数        │ 具体的なイベント型    │
  │ catch 節のエラー             │ unknown              │
  │ テスト用のモックデータ        │ 具体的な型            │
  └─────────────────────────────┴──────────────────────┘
```

### Q3: Rust の型システムが「最も安全」と言われる理由は？

**A**: Rust は型システムに所有権（Ownership）と借用（Borrowing）の概念を統合し、
メモリ安全性とスレッド安全性をコンパイル時に保証する唯一の主流言語だからである。

具体的には以下の保証をコンパイル時に提供する。

1. **ダングリングポインタの防止**: 参照のライフタイムを型システムが追跡
2. **二重解放の防止**: 所有権の移動（move）により、値の所有者は常に一つ
3. **データ競合の防止**: `&mut T`（ミュータブル参照）は同時に一つだけ存在可能
4. **null参照の防止**: `Option<T>` 型により、値の不在を型で表現
5. **エラー処理の強制**: `Result<T, E>` 型により、エラーの可能性が型で表現

これらの保証は全てコンパイル時に検証され、実行時のオーバーヘッドはゼロである。

### Q4: 型推論が強力なら、型注釈は不要ではないか？

**A**: 型推論は型注釈の記述量を減らすが、完全に不要にするものではない。
以下の場面では明示的な型注釈が推奨される。

1. **公開API（関数の引数と戻り値）**: ライブラリやモジュールの境界では、
   型注釈がドキュメントとして機能する。推論に頼ると、実装変更が意図せずAPIの型を変える危険がある。

2. **複雑な型が推論される場合**: 推論結果が長大・複雑な型になる場合、
   明示的な型注釈で意図を明確にする方が可読性が高い。

3. **エラーメッセージの改善**: 型注釈があると、型エラーの発生箇所と原因が明確になる。
   推論に頼ると、エラーが離れた場所で報告される場合がある。

ベストプラクティスは「公開APIには明示的に、ローカル変数には推論に任せる」である。

### Q5: 動的型付けから静的型付けへの移行で最も重要な注意点は？

**A**: 段階的に移行し、一度に全てを変換しようとしないことが最も重要である。

推奨される移行戦略は以下の通り。

1. **まず型チェッカーを導入する**（mypy, tsc, PHPStan等）。設定を最も緩い状態で開始する。
2. **新規コードから型注釈を必須にする**。既存コードは後回しにする。
3. **クリティカルなモジュールから順に型注釈を追加する**。共通ライブラリ、API層、データモデル等。
4. **段階的に strictness を上げる**。`noImplicitAny` → `strictNullChecks` → `strict` の順で有効化。
5. **CI/CD に型チェックを組み込む**。型エラーがあればビルドを失敗させる。

移行中の混在状態は避けられないが、これは段階的型付けが本来想定している状態であり、
問題ではない。重要なのは「常に前進し続けること」と「型カバレッジを計測し可視化すること」である。

### Q6: 関数型言語の型システムはオブジェクト指向言語と何が違うのか？

**A**: 関数型言語の型システムは、代数的データ型（ADT）とパターンマッチングを中心に設計されている点が
最大の違いである。

```
オブジェクト指向の型 vs 関数型の型:

  オブジェクト指向（Java, C#）:
    - クラス階層（継承）で型を組織化
    - サブタイプ多態性（ポリモーフィズム）
    - 型の拡張 = 新しいサブクラスの追加が容易
    - 操作の拡張 = 既存クラスへのメソッド追加が困難
    → 「Expression Problem」の片側を解決

  関数型（Haskell, OCaml, Rust）:
    - 代数的データ型（直和型 + 直積型）で型を構成
    - パラメトリック多態性 + 型クラス/トレイト
    - 操作の拡張 = 新しい関数の追加が容易
    - 型の拡張 = 既存のADTへのバリアントの追加が困難
    → 「Expression Problem」のもう片側を解決

  現代の言語はこの2つのアプローチを融合する傾向:
    - Rust: トレイト（型クラス的）+ enum（ADT）
    - Kotlin: sealed class（ADT的）+ interface
    - TypeScript: union type（ADT的）+ interface
    - Scala: case class + trait + パターンマッチ
```

---

## 12. まとめ --- 型システムの全体像

### 12.1 総括比較表

| 分類 | 型チェック時点 | 暗黙変換 | 安全性 | 柔軟性 | 代表言語 |
|------|:---:|:---:|:---:|:---:|:---:|
| 静的 + 強い | コンパイル時 | ほぼなし | 最高 | 低〜中 | Rust, Haskell, Go, Kotlin |
| 静的 + 弱い | コンパイル時 | あり | 中 | 中 | C, C++ |
| 動的 + 強い | 実行時 | ほぼなし | 中 | 高 | Python, Ruby, Elixir |
| 動的 + 弱い | 実行時 | 多い | 低 | 最高 | JavaScript, PHP, Perl |
| 段階的 | 混在 | 設定依存 | 中〜高 | 中〜高 | TypeScript, Python+mypy |

### 12.2 型システム設計の5つの原則

```
型システム設計の5つの原則:

  1. 安全性第一（Safety First）
     型システムの主目的はバグの防止である。
     利便性のために安全性を犠牲にするのは最後の手段。

  2. 段階的厳格化（Gradual Strictness）
     最初から最も厳格な設定を強制せず、
     段階的に型カバレッジと厳格さを向上させる。

  3. 型は仕様である（Types as Specification）
     型注釈は単なる「コンパイラへの指示」ではなく、
     プログラムの仕様・設計意図の表現である。

  4. 推論に頼り、境界で明示する（Infer Locally, Annotate at Boundaries）
     ローカル変数は型推論に任せ、
     公開API・モジュール境界では明示的に型を記述する。

  5. 不可能な状態を表現不可能にする（Make Illegal States Unrepresentable）
     型システムを活用して、論理的に不正な状態を
     そもそも型レベルで構築できないように設計する。
```

### 12.3 学習ロードマップ

```
型システム学習ロードマップ:

  Level 1: 基礎（本章の内容）
  ├── 静的 vs 動的の理解
  ├── 強い vs 弱いの理解
  ├── 段階的型付けの概念
  └── 基本的な型注釈の記述
      │
      ▼
  Level 2: 実践
  ├── ジェネリクス/パラメトリック多態性
  ├── 型推論の仕組みと限界
  ├── 代数的データ型（直和型・直積型）
  └── パターンマッチング
      │
      ▼
  Level 3: 応用
  ├── 型クラス / トレイト / Protocol
  ├── 高カインド型（Higher-Kinded Types）
  ├── 存在型（Existential Types）
  └── 型レベルプログラミング
      │
      ▼
  Level 4: 理論
  ├── ラムダ計算と型理論
  ├── System F / System Fω
  ├── 依存型（Dependent Types）
  └── 線形型（Linear Types）
```

---

## 次に読むべきガイド

- [[01-type-inference.md]] --- 型推論: Hindley-Milner型推論からローカル型推論まで
- [[02-generics.md]] --- ジェネリクスとパラメトリック多態性
- [[03-algebraic-data-types.md]] --- 代数的データ型とパターンマッチング

---

## 参考文献

1. Pierce, B. C. "Types and Programming Languages." MIT Press, 2002.
   --- 型理論の包括的教科書。型システムの数学的基礎から実装まで。通称 TAPL。

2. Siek, J. G. & Taha, W. "Gradual Typing for Functional Languages."
   Scheme and Functional Programming Workshop, 2006.
   --- 段階的型付けの理論的基礎を確立した論文。

3. Cardelli, L. & Wegner, P. "On Understanding Types, Data Abstraction, and Polymorphism."
   Computing Surveys, Vol. 17, No. 4, pp. 471-523, 1985.
   --- 型システムの分類と多態性の理論的枠組みを確立した古典的論文。

4. Harper, R. "Practical Foundations for Programming Languages." 2nd Edition,
   Cambridge University Press, 2016.
   --- プログラミング言語の基礎理論を型理論の観点から体系的に解説。

5. Klabnik, S. & Nichols, C. "The Rust Programming Language." No Starch Press, 2019.
   --- Rust の所有権システムと型システムの実践的解説。公式ドキュメントの書籍版。

6. Vanderkam, D. "Effective TypeScript: 83 Specific Ways to Improve Your TypeScript."
   O'Reilly Media, 2024.
   --- TypeScript の型システムを実務的に活用するためのベストプラクティス集。

7. Mypy Documentation. "Type checking Python programs." https://mypy.readthedocs.io/
   --- Python の段階的型付けにおける標準的な型チェッカーの公式ドキュメント。

