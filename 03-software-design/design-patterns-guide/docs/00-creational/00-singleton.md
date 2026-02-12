# Singleton パターン

> インスタンスがアプリケーション全体で **ただ1つ** であることを保証し、そのグローバルアクセスポイントを提供する生成パターン。

---

## この章で学ぶこと

1. Singleton パターンの目的・構造と、なぜ「インスタンスを1つに制限する」必要があるのかという設計意図（WHY）
2. スレッドセーフな実装方法（Eager Init / DCL / Holder / Enum / モジュールスコープ）の内部動作と使い分け
3. DI（依存性注入）による Singleton の代替手法とテスト容易性の確保
4. Singleton の濫用が招くグローバル状態問題と、適切な利用場面の見極め方
5. 各言語（TypeScript / Java / Python / Go / Kotlin）での実装パターンとエッジケース

---

## 前提知識

このガイドを理解するために、以下の知識を事前に習得しておくことを推奨します。

| トピック | 必要レベル | 参照リンク |
|---------|-----------|-----------|
| オブジェクト指向の基礎（クラス、インスタンス、コンストラクタ） | 必須 | [OOP基礎](../../../02-programming/oop-guide/docs/) |
| SOLID原則（特にSRP、DIP） | 推奨 | [SOLID原則](../../../clean-code-principles/docs/00-principles/01-solid.md) |
| マルチスレッド／並行処理の基礎 | 推奨 | [並行処理](../../../01-cs-fundamentals/os-guide/docs/) |
| ES Module の仕組み（JavaScript/TypeScript） | 推奨 | [MDN Modules](https://developer.mozilla.org/ja/docs/Web/JavaScript/Guide/Modules) |
| DI（依存性注入）の概念 | あると望ましい | [DIP](../../../clean-code-principles/docs/00-principles/01-solid.md) |

---

## 1. Singleton パターンとは何か — 本質的な理解

### 1.1 パターンの定義

Singleton パターンは GoF（Gang of Four）が定義した23のデザインパターンのうち、「生成パターン（Creational Patterns）」に分類される。その意図は以下の通り:

> **あるクラスのインスタンスがただ1つしか存在しないことを保証し、それに対するグローバルなアクセスポイントを提供する。**

この定義には2つの独立した責任が含まれている:

1. **インスタンス数の制限**: クラス自身が自分のインスタンス生成を制御する
2. **グローバルアクセス**: アプリケーションのどこからでもそのインスタンスにアクセスできる

### 1.2 なぜ Singleton が必要なのか（WHY）

Singleton パターンが解決する根本的な問題は「**共有リソースの一貫性保証**」である。

```
【問題のシナリオ】

スレッド A                      スレッド B
    |                               |
    v                               v
new DatabasePool()              new DatabasePool()
    |                               |
    v                               v
コネクション 10本確保           コネクション 10本確保
    |                               |
    v                               v
合計 20本 → データベースの上限超過でエラー！

【Singleton による解決】

スレッド A                      スレッド B
    |                               |
    v                               v
DatabasePool.getInstance()      DatabasePool.getInstance()
    |                               |
    +--------> 同じインスタンス <----+
               |
               v
           コネクション 10本（1回だけ確保）
```

具体的に Singleton が必要になるケース:

1. **データベースコネクションプール**: 複数のプールが生成されるとコネクション数が制御不能になる
2. **ロガー**: 複数のロガーインスタンスが同じファイルに書き込むとデータ競合が発生する
3. **設定オブジェクト**: 異なるインスタンスが異なる設定値を持つと、アプリケーション動作が不整合になる
4. **キャッシュ**: 複数のキャッシュインスタンスは同じデータの重複保存となり、一貫性も失われる
5. **ハードウェアアクセス**: プリンタスプーラやGPUコンテキストなど、物理的に1つしか存在しないリソース

### 1.3 Singleton を使うべきでないケース

一方で、以下の場合は Singleton を避けるべきである:

- **ステートレスなユーティリティ**: 状態を持たないなら、静的メソッドで十分
- **ビジネスロジックの共有**: ドメインサービスを Singleton にすると、テストが困難になる
- **「便利だから」という理由**: グローバル変数の代わりとして使うのは設計の放棄

---

## 2. Singleton の構造

### 2.1 UML クラス図

```
+---------------------------+
|       Singleton           |
+---------------------------+
| - instance: Singleton     |  ← クラス変数（static）
| - data: any               |  ← インスタンス変数
+---------------------------+
| - constructor()           |  ← private: 外部から new 不可
| + getInstance(): Singleton|  ← 唯一のアクセスポイント
| + getData(): any          |
| + setData(d: any): void   |
+---------------------------+
        |
        | instance は 1つだけ生成される
        v
  +------------------+
  | <<instance>>     |
  | Singleton        |
  | data: "some val" |
  +------------------+
```

### 2.2 シーケンス図

```
Client A          Singleton Class          Client B
   |                    |                      |
   |  getInstance()     |                      |
   |------------------->|                      |
   |                    |                      |
   |  instance == null  |                      |
   |  → new Singleton() |                      |
   |  → instance に格納  |                      |
   |                    |                      |
   |  <--- instance ----|                      |
   |                    |                      |
   |                    |   getInstance()      |
   |                    |<---------------------|
   |                    |                      |
   |                    |   instance != null   |
   |                    |   → 既存を返す        |
   |                    |                      |
   |                    |--- instance -------->|
   |                    |                      |

   ※ Client A と Client B は同じインスタンスを受け取る
```

### 2.3 内部動作の詳細

Singleton パターンの内部動作を段階的に理解する:

```
Step 1: 初回アクセス
┌─────────────────────────────────────┐
│ Singleton クラス                     │
│                                     │
│  static instance: null  ←── 未生成  │
│                                     │
│  getInstance() {                    │
│    if (instance == null) {  ← true  │
│      instance = new Singleton()     │
│      ↑ private constructor 呼出     │
│    }                                │
│    return instance  ← 新規生成      │
│  }                                  │
└─────────────────────────────────────┘

Step 2: 2回目以降のアクセス
┌─────────────────────────────────────┐
│ Singleton クラス                     │
│                                     │
│  static instance: [Object] ← 既存  │
│                                     │
│  getInstance() {                    │
│    if (instance == null) { ← false  │
│      // スキップ                     │
│    }                                │
│    return instance  ← 既存を返す     │
│  }                                  │
└─────────────────────────────────────┘
```

---

## 3. 各実装手法の詳細解説

### コード例 1: クラシック Singleton（TypeScript）

最も基本的な実装。シングルスレッド環境（JavaScript/TypeScript）では問題なく動作する。

```typescript
class Singleton {
  private static instance: Singleton | null = null;
  private value: number;

  // private constructor で外部からの new を禁止
  private constructor(value: number) {
    this.value = value;
    console.log("Singleton: コンストラクタ実行（1度だけ）");
  }

  static getInstance(): Singleton {
    if (!Singleton.instance) {
      Singleton.instance = new Singleton(42);
    }
    return Singleton.instance;
  }

  getValue(): number {
    return this.value;
  }

  setValue(value: number): void {
    this.value = value;
  }
}

// 使用例
const a = Singleton.getInstance();
const b = Singleton.getInstance();

console.log(a === b);         // true — 同一インスタンス
console.log(a.getValue());    // 42

a.setValue(100);
console.log(b.getValue());    // 100 — 同じインスタンスなので反映される
```

**WHY — なぜ private constructor なのか?**

コンストラクタを private にすることで、外部コードが `new Singleton()` を呼ぶことを**コンパイル時に防止**する。これがなければ、開発者が誤って複数インスタンスを生成する可能性がある。

```typescript
// private constructor がない場合の危険
const s1 = new Singleton(1);  // 直接生成 → 制御不能
const s2 = new Singleton(2);  // 2つ目が生成されてしまう
```

### コード例 2: モジュールスコープ Singleton（TypeScript — 推奨）

JavaScript/TypeScript では、ES Module 自体がキャッシュされるため、モジュールレベルで export したインスタンスは事実上 Singleton になる。

```typescript
// config.ts — ES Module 自体が Singleton として振る舞う
class AppConfig {
  readonly dbHost: string;
  readonly dbPort: number;
  readonly logLevel: string;
  readonly maxRetries: number;

  constructor() {
    // 環境変数からの読み込み（1度だけ実行される）
    this.dbHost = process.env.DB_HOST ?? "localhost";
    this.dbPort = Number(process.env.DB_PORT ?? 5432);
    this.logLevel = process.env.LOG_LEVEL ?? "info";
    this.maxRetries = Number(process.env.MAX_RETRIES ?? 3);

    console.log("AppConfig: 初期化完了");
  }

  get connectionString(): string {
    return `postgresql://${this.dbHost}:${this.dbPort}/mydb`;
  }
}

// モジュールレベルで1度だけインスタンス化
// import した全てのモジュールが同じインスタンスを受け取る
export const appConfig = new AppConfig();

// --- 利用側 ---
// import { appConfig } from './config';
// console.log(appConfig.connectionString);
```

**WHY — モジュールスコープが推奨される理由:**

1. **言語仕様で保証**: ECMAScript 仕様によりモジュールは1度だけ評価される
2. **コード量が最小**: getInstance() のようなボイラープレートが不要
3. **直感的**: 通常のオブジェクトと同じように使える
4. **Tree-shaking 対応**: 使われなければバンドラが除外する

```
モジュール評価のメカニズム:

1回目の import { appConfig } from './config'
  → config.ts を評価 → new AppConfig() 実行 → キャッシュに保存

2回目の import { appConfig } from './config'
  → キャッシュから取得 → 同じインスタンスが返る（再評価なし）

3回目の import { appConfig } from './config'
  → キャッシュから取得 → 同じインスタンスが返る（再評価なし）
```

### コード例 3: スレッドセーフ — Double-Checked Locking（Java）

マルチスレッド環境では、2つのスレッドが同時に `getInstance()` を呼ぶと、インスタンスが2つ生成される可能性がある。DCL（Double-Checked Locking）はこれを防ぐ。

```java
public class Singleton {
    // volatile: メモリの可視性を保証（CPUキャッシュの問題を防ぐ）
    private static volatile Singleton instance;

    private Singleton() {
        // 重い初期化処理（例: DB接続、設定ファイル読み込み）
        System.out.println("Singleton: 初期化実行");
    }

    public static Singleton getInstance() {
        if (instance == null) {                  // 1st check (ロックなし — 高速パス)
            synchronized (Singleton.class) {     // ロック取得
                if (instance == null) {          // 2nd check (ロック内で再確認)
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

**WHY — なぜ「二重」チェックなのか?**

```
【DCLなし（synchronized のみ）の場合】

Thread A: getInstance() → synchronized ブロックに入る → 待ち
Thread B: getInstance() → synchronized ブロックに入る → 待ち
  ↑ 毎回ロック取得のオーバーヘッド（インスタンス生成済みでも）

【DCLの場合】

Step 1: 1st check（ロックなし）
  instance != null → そのまま return（高速パス、99.99%のケース）
  instance == null → Step 2 へ

Step 2: synchronized ブロック（ロック取得）
  複数スレッドがここに到達しても、1つだけが入れる

Step 3: 2nd check（ロック内）
  別のスレッドが先に生成していた場合、ここで検出して重複生成を防ぐ
```

**WHY — なぜ volatile が必要なのか?**

`new Singleton()` は内部的に以下の3ステップで実行される:

```
① メモリを確保する
② コンストラクタを実行する（初期化）
③ instance 変数にメモリアドレスを代入する

JVM の命令リオーダリングにより、実行順序が ①→③→② になる可能性がある。

Thread A: ①→③ まで完了（②はまだ）
Thread B: 1st check で instance != null → 未初期化のオブジェクトを使用 → バグ！

volatile はリオーダリングを禁止し、①→②→③ の順序を保証する。
```

### コード例 4: Bill Pugh Singleton（Java — 推奨）

Holder パターンとも呼ばれる。JVM のクラスローディングメカニズムを利用した、最もエレガントなスレッドセーフ実装。

```java
public class Singleton {
    private Singleton() {
        System.out.println("Singleton: 初期化実行");
    }

    // 内部クラスは、最初にアクセスされるまでロードされない
    private static class Holder {
        // static final により JVM が初期化の排他制御を保証
        private static final Singleton INSTANCE = new Singleton();
    }

    public static Singleton getInstance() {
        return Holder.INSTANCE;  // 初回呼出時に Holder クラスがロードされる
    }
}
```

**WHY — なぜ Holder パターンが推奨なのか?**

1. **遅延初期化**: `Holder` クラスは `getInstance()` が初めて呼ばれるまでロードされない
2. **スレッドセーフ**: JVM 仕様により、クラスの初期化は排他的に行われる（JLS 12.4.2）
3. **ロックフリー**: synchronized を使わないため、パフォーマンスへの影響がない
4. **コードがシンプル**: volatile も synchronized も不要

### コード例 5: Enum Singleton（Java）

Joshua Bloch（Effective Java著者）が推奨する、Java における最善の Singleton 実装。

```java
public enum Singleton {
    INSTANCE;

    private int value;

    public int getValue() {
        return value;
    }

    public void setValue(int value) {
        this.value = value;
    }

    public void doSomething() {
        System.out.println("Singleton operation with value: " + value);
    }
}

// 使用
Singleton.INSTANCE.setValue(42);
Singleton.INSTANCE.doSomething();
```

**WHY — なぜ Enum が最善なのか?**

1. **スレッドセーフ**: JVM が enum の初期化を保証
2. **シリアライゼーション対応**: デシリアライズで新インスタンスが生成されない（自動対応）
3. **リフレクション攻撃防止**: `Constructor.newInstance()` で enum は生成できない
4. **コードが最小**: ボイラープレートがほぼゼロ

```
通常の Singleton が脆弱な攻撃:

1. リフレクション攻撃
   Constructor<Singleton> c = Singleton.class.getDeclaredConstructor();
   c.setAccessible(true);
   Singleton s2 = c.newInstance();  // 2つ目が生成される！
   → Enum では IllegalArgumentException

2. デシリアライゼーション攻撃
   ObjectInputStream ois = new ObjectInputStream(...);
   Singleton s2 = (Singleton) ois.readObject();  // 別インスタンス！
   → Enum では同じ INSTANCE が返る
```

### コード例 6: Python — メタクラス Singleton

Python では、メタクラスを使うことで、クラスのインスタンス生成プロセスをカスタマイズできる。

```python
import threading

class SingletonMeta(type):
    """スレッドセーフなSingletonメタクラス"""
    _instances: dict = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Double-Checked Locking パターン
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "connected"
        print(f"Database: 初期化実行 (id={id(self)})")

    def query(self, sql: str) -> str:
        return f"Result of: {sql}"

# 使用
db1 = Database()  # "Database: 初期化実行 (id=...)"
db2 = Database()  # 出力なし（既存インスタンスを返す）

print(db1 is db2)            # True
print(db1.query("SELECT 1")) # "Result of: SELECT 1"
```

**WHY — なぜメタクラスなのか?**

```
Python のインスタンス生成フロー:

obj = MyClass(args)
  ↓
MyClass.__call__(args)     ← メタクラスの __call__ が呼ばれる
  ↓
MyClass.__new__(cls)       ← オブジェクトのメモリ確保
  ↓
MyClass.__init__(self)     ← 初期化
  ↓
return obj

SingletonMeta.__call__ をオーバーライドすることで、
__new__ と __init__ の「前」にインスタンス存在チェックを挿入できる。
```

### コード例 7: Python — デコレータ Singleton

メタクラスより軽量な方法として、デコレータを使うアプローチもある。

```python
from functools import wraps

def singleton(cls):
    """Singletonデコレータ"""
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Logger:
    def __init__(self):
        self.logs: list[str] = []
        print("Logger: 初期化実行")

    def log(self, message: str):
        self.logs.append(message)
        print(f"[LOG] {message}")

# 使用
logger1 = Logger()  # "Logger: 初期化実行"
logger2 = Logger()  # 出力なし

logger1.log("Start")
print(logger2.logs)  # ["Start"] — 同じインスタンス
```

### コード例 8: Go — sync.Once

Go 言語では `sync.Once` を使うことで、ゴルーチンセーフな遅延初期化を実現する。

```go
package main

import (
    "fmt"
    "sync"
)

type singleton struct {
    value int
}

var (
    instance *singleton
    once     sync.Once
)

func GetInstance() *singleton {
    once.Do(func() {
        fmt.Println("Singleton: 初期化実行")
        instance = &singleton{value: 42}
    })
    return instance
}

func main() {
    s1 := GetInstance() // "Singleton: 初期化実行"
    s2 := GetInstance() // 出力なし

    fmt.Println(s1 == s2)    // true
    fmt.Println(s1.value)    // 42
}
```

**WHY — sync.Once の内部実装:**

```go
// sync.Once の内部（簡略版）
type Once struct {
    done uint32     // atomic でチェック（高速パス）
    m    Mutex      // 初回のみ使用
}

func (o *Once) Do(f func()) {
    if atomic.LoadUint32(&o.done) == 0 {  // 高速パス
        o.doSlow(f)
    }
}

func (o *Once) doSlow(f func()) {
    o.m.Lock()
    defer o.m.Unlock()
    if o.done == 0 {  // DCL
        defer atomic.StoreUint32(&o.done, 1)
        f()
    }
}
```

### コード例 9: Kotlin — object 宣言

Kotlin では `object` キーワードにより、言語レベルで Singleton がサポートされている。

```kotlin
object AppConfig {
    val dbHost: String = System.getenv("DB_HOST") ?: "localhost"
    val dbPort: Int = System.getenv("DB_PORT")?.toInt() ?: 5432

    fun connectionString(): String =
        "postgresql://$dbHost:$dbPort/mydb"

    init {
        println("AppConfig: 初期化実行")
    }
}

// 使用
fun main() {
    println(AppConfig.connectionString())
    // AppConfig.connectionString() == AppConfig.connectionString() は常に同じ結果
}
```

**WHY — Kotlin の object は内部的に何をしているのか?**

```
Kotlin の object 宣言は、コンパイル時に以下の Java コードに変換される:

public final class AppConfig {
    public static final AppConfig INSTANCE;

    static {
        AppConfig var0 = new AppConfig();
        INSTANCE = var0;
    }

    private AppConfig() { ... }
}

→ Eager Init パターン + private constructor が自動生成される
```

### コード例 10: DI コンテナによる Singleton ライフタイム

実務では、クラス自身が Singleton 制約を管理するより、DI コンテナに委譲する方が圧倒的に優れている。

```typescript
// InversifyJS の例
import { Container, injectable, inject } from "inversify";

// インタフェース
interface Logger {
  log(message: string): void;
}

interface Database {
  query(sql: string): Promise<any>;
}

const TYPES = {
  Logger: Symbol.for("Logger"),
  Database: Symbol.for("Database"),
};

// 実装クラス（Singleton であることを知らない）
@injectable()
class ConsoleLogger implements Logger {
  constructor() {
    console.log("ConsoleLogger: 初期化");
  }

  log(message: string): void {
    console.log(`[LOG] ${message}`);
  }
}

@injectable()
class PostgresDatabase implements Database {
  constructor(@inject(TYPES.Logger) private logger: Logger) {
    this.logger.log("PostgresDatabase: 初期化");
  }

  async query(sql: string): Promise<any> {
    this.logger.log(`Executing: ${sql}`);
    return [];
  }
}

// DI コンテナでライフタイムを制御
const container = new Container();

container
  .bind<Logger>(TYPES.Logger)
  .to(ConsoleLogger)
  .inSingletonScope();   // ← コンテナが1インスタンスを保証

container
  .bind<Database>(TYPES.Database)
  .to(PostgresDatabase)
  .inSingletonScope();

// 取得
const logger1 = container.get<Logger>(TYPES.Logger);
const logger2 = container.get<Logger>(TYPES.Logger);
console.log(logger1 === logger2); // true

// テスト時はモックに差し替え可能
const testContainer = new Container();
testContainer.bind<Logger>(TYPES.Logger).toConstantValue({
  log: jest.fn(),
});
```

**WHY — DI コンテナが優れている理由:**

```
クラス内 Singleton:
┌──────────────────────┐
│ class Database {     │
│   static instance    │  ← クラスがライフタイムを管理
│   private constructor│  ← テストでモック不可
│   static getInstance │  ← 直接参照 = 密結合
│ }                    │
└──────────────────────┘

DI コンテナ Singleton:
┌──────────────────────┐
│ interface Database { │  ← インタフェースに依存
│   query(): ...       │
│ }                    │
│                      │
│ container.bind(...)  │  ← コンテナがライフタイム管理
│   .inSingletonScope()│  ← スコープの変更が1行
│                      │
│ テスト時:             │
│   bind(...).to(Mock) │  ← モック差し替え容易
└──────────────────────┘
```

---

## 4. スレッドセーフ実装の比較

### 4.1 実装戦略の全体像

```
┌──────────────────────────────────────────────────────────────┐
│              スレッドセーフ Singleton 実装戦略                  │
├──────────────┬───────────────────────────────────────────────┤
│  Eager Init  │  クラスロード時に生成（最も単純）                │
│              │  static instance = new Singleton()            │
│              │  利点: 実装簡単、スレッドセーフ保証              │
│              │  欠点: 使われなくてもメモリを消費              │
├──────────────┼───────────────────────────────────────────────┤
│  DCL         │  Double-Checked Locking                       │
│              │  volatile + synchronized                      │
│              │  利点: 遅延初期化 + 高速パス                    │
│              │  欠点: 実装複雑、volatile の理解が必要          │
├──────────────┼───────────────────────────────────────────────┤
│  Holder      │  内部クラスの遅延ロード（Bill Pugh）            │
│              │  利点: 遅延 + ロックフリー + シンプル            │
│              │  欠点: Java 専用のイディオム                    │
├──────────────┼───────────────────────────────────────────────┤
│  Enum        │  JVM が保証。直列化・リフレクションにも対応       │
│              │  利点: 完全防御、コード最小                      │
│              │  欠点: 継承不可、Java 専用                      │
├──────────────┼───────────────────────────────────────────────┤
│  sync.Once   │  Go 標準ライブラリ                              │
│              │  利点: ゴルーチンセーフ、イディオマティック         │
│              │  欠点: Go 専用                                  │
├──────────────┼───────────────────────────────────────────────┤
│  Module      │  ES Module のキャッシュ機構を利用               │
│  Scope       │  利点: 言語仕様で保証、コード最小               │
│              │  欠点: JS/TS 専用、テスト時のリセットに注意      │
└──────────────┴───────────────────────────────────────────────┘
```

### 4.2 パフォーマンス特性

```
アクセス回数 vs レイテンシ（概念図）

レイテンシ
  ^
  |
  |  * Lazy (synchronized毎回)
  |  |
  |  |   * DCL (初回のみロック)
  |  |   |
  |  |   |  * Eager / Holder / Enum / Module
  |  |   |  |
  |--+---+--+-------------------------> アクセス回数
     1   2  3   4   5   ...  1000

  ※ DCL は2回目以降、Eager/Holder と同等の速度
  ※ synchronized 毎回は常にロック取得コストがかかる
```

---

## 5. 比較表

### 比較表 1: Singleton 実装手法の比較

| 手法 | 遅延初期化 | スレッドセーフ | 実装難易度 | 直列化対応 | リフレクション防御 | 推奨度 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Eager Init | No | Yes | 低 | 要対応 | No | B |
| Lazy (同期なし) | Yes | No | 低 | 要対応 | No | D |
| DCL | Yes | Yes | 中 | 要対応 | No | B |
| Holder パターン | Yes | Yes | 中 | 要対応 | No | A |
| Enum (Java) | No | Yes | 低 | 自動 | Yes | S |
| sync.Once (Go) | Yes | Yes | 低 | N/A | N/A | A |
| object (Kotlin) | No | Yes | 低 | 要対応 | No | A |
| モジュールスコープ (JS/TS) | Yes* | N/A | 低 | N/A | N/A | S |

*モジュール初回インポート時に評価される。

### 比較表 2: Singleton vs DI コンテナ vs グローバル変数

| 観点 | クラス内 Singleton | DI コンテナ Singleton | グローバル変数 |
|------|:---:|:---:|:---:|
| テスト容易性 | 低い | 高い | 最低 |
| 結合度 | 高い（直接参照） | 低い（IF経由） | 最高 |
| ライフタイム管理 | クラス自身 | コンテナ | なし |
| グローバル状態 | 露出する | 隠蔽可能 | 完全露出 |
| 柔軟性 | 低い | 高い | 低い |
| 型安全性 | あり | あり | 言語依存 |
| IDEサポート | 良好 | 良好 | 限定的 |

### 比較表 3: 言語別 Singleton 推奨実装

| 言語 | 推奨実装 | 理由 |
|------|---------|------|
| Java | Enum Singleton | 完全防御、Effective Java 推奨 |
| Kotlin | object 宣言 | 言語ネイティブサポート |
| TypeScript | モジュールスコープ export | 言語仕様で保証、最小コード |
| Python | メタクラス or デコレータ | 柔軟、Pythonic |
| Go | sync.Once | 標準ライブラリ、イディオマティック |
| C# | Lazy<T> | .NET 標準、スレッドセーフ |
| Rust | once_cell / std::sync::OnceLock | 所有権システムと統合 |

---

## 6. Singleton と関連パターンの関係

### 6.1 Singleton が使われる場面マップ

```
                    Singleton
                       |
        +--------------+--------------+
        |              |              |
   Logger         Config         Registry
   (ロガー)       (設定)         (レジストリ)
        |              |              |
        |              |              +--- Factory の
        |              |                   プロダクト登録先
        |              +--- Builder で
        |                   複雑な設定を構築
        +--- Observer の
             イベント集約先
```

### 6.2 Singleton と他パターンの組み合わせ

```typescript
// Singleton + Factory: レジストリパターン
class PluginRegistry {
  private static instance: PluginRegistry;
  private plugins = new Map<string, () => Plugin>();

  private constructor() {}

  static getInstance(): PluginRegistry {
    if (!this.instance) {
      this.instance = new PluginRegistry();
    }
    return this.instance;
  }

  register(name: string, factory: () => Plugin): void {
    this.plugins.set(name, factory);
  }

  create(name: string): Plugin {
    const factory = this.plugins.get(name);
    if (!factory) throw new Error(`Unknown plugin: ${name}`);
    return factory();
  }
}

// Singleton + Observer: イベントバス
class EventBus {
  private static instance: EventBus;
  private listeners = new Map<string, Set<Function>>();

  private constructor() {}

  static getInstance(): EventBus {
    if (!this.instance) {
      this.instance = new EventBus();
    }
    return this.instance;
  }

  on(event: string, handler: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
  }

  emit(event: string, data?: any): void {
    this.listeners.get(event)?.forEach(handler => handler(data));
  }
}
```

---

## 7. 実世界での適用例

### 7.1 Node.js のモジュールキャッシュ（Singleton の実例）

Node.js の `require()` / `import` は内部的にモジュールをキャッシュする。これは Singleton パターンの実装そのものである。

```
require('express') の内部動作:

Module._cache = {};  // グローバルキャッシュ

Module._load(filename) {
  if (Module._cache[filename]) {
    return Module._cache[filename].exports;  // キャッシュヒット
  }

  const module = new Module(filename);
  Module._cache[filename] = module;  // キャッシュに保存
  module.load(filename);             // ファイルを読み込み・実行
  return module.exports;
}
```

### 7.2 データベースコネクションプール

```typescript
// 実務的な Singleton コネクションプール
import { Pool, PoolConfig } from 'pg';

class DatabasePool {
  private static pool: Pool | null = null;

  static getPool(): Pool {
    if (!DatabasePool.pool) {
      const config: PoolConfig = {
        host: process.env.DB_HOST ?? 'localhost',
        port: Number(process.env.DB_PORT ?? 5432),
        database: process.env.DB_NAME ?? 'myapp',
        user: process.env.DB_USER ?? 'postgres',
        password: process.env.DB_PASSWORD,
        max: 20,                    // 最大コネクション数
        idleTimeoutMillis: 30000,   // アイドルタイムアウト
        connectionTimeoutMillis: 2000, // 接続タイムアウト
      };

      DatabasePool.pool = new Pool(config);

      // エラーハンドリング
      DatabasePool.pool.on('error', (err) => {
        console.error('Unexpected pool error:', err);
      });

      console.log('DatabasePool: 初期化完了 (max=20)');
    }
    return DatabasePool.pool;
  }

  // テスト用: プールをリセット
  static async reset(): Promise<void> {
    if (DatabasePool.pool) {
      await DatabasePool.pool.end();
      DatabasePool.pool = null;
    }
  }
}

// 使用
async function getUsers() {
  const pool = DatabasePool.getPool();
  const result = await pool.query('SELECT * FROM users');
  return result.rows;
}
```

### 7.3 設定管理（環境別切り替え）

```typescript
// 環境別の設定を Singleton で管理
type Environment = 'development' | 'staging' | 'production';

interface AppSettings {
  readonly env: Environment;
  readonly apiBaseUrl: string;
  readonly logLevel: 'debug' | 'info' | 'warn' | 'error';
  readonly enableMetrics: boolean;
  readonly maxRetries: number;
}

const settings: Record<Environment, AppSettings> = {
  development: {
    env: 'development',
    apiBaseUrl: 'http://localhost:3000',
    logLevel: 'debug',
    enableMetrics: false,
    maxRetries: 1,
  },
  staging: {
    env: 'staging',
    apiBaseUrl: 'https://staging-api.example.com',
    logLevel: 'info',
    enableMetrics: true,
    maxRetries: 3,
  },
  production: {
    env: 'production',
    apiBaseUrl: 'https://api.example.com',
    logLevel: 'warn',
    enableMetrics: true,
    maxRetries: 5,
  },
};

const env = (process.env.NODE_ENV as Environment) ?? 'development';
export const appSettings: Readonly<AppSettings> = Object.freeze(settings[env]);
```

---

## 8. エッジケースと注意点

### 8.1 シリアライゼーションによる Singleton 破壊

```java
// Java: デシリアライゼーションで Singleton が破壊される例
public class Singleton implements Serializable {
    private static final Singleton INSTANCE = new Singleton();

    private Singleton() {}

    public static Singleton getInstance() {
        return INSTANCE;
    }

    // これがないと、デシリアライズ時に新インスタンスが生成される
    // readResolve() で既存インスタンスを返すことで防御
    private Object readResolve() {
        return INSTANCE;
    }
}
```

### 8.2 リフレクションによる Singleton 破壊

```java
// 攻撃コード
Constructor<Singleton> constructor = Singleton.class.getDeclaredConstructor();
constructor.setAccessible(true);
Singleton s2 = constructor.newInstance(); // private constructor を迂回！

// 防御策: コンストラクタ内でチェック
private Singleton() {
    if (INSTANCE != null) {
        throw new IllegalStateException("Singleton already initialized");
    }
}
```

### 8.3 クラスローダーによる複数インスタンス

```
Java EE / OSGi 環境では、異なるクラスローダーが同じクラスを
別々にロードする可能性がある:

ClassLoader A → Singleton.class → instance A
ClassLoader B → Singleton.class → instance B

→ 2つの「Singleton」が存在してしまう！

対策:
- アプリケーションの共有クラスローダーに Singleton を配置
- JNDI を使ってインスタンスを共有
- DI コンテナのスコープ管理に委譲（推奨）
```

### 8.4 JavaScript テスト環境での問題

```typescript
// Jest でモジュールキャッシュがリセットされる問題

// config.ts
export const config = { value: "original" };

// test1.spec.ts
import { config } from './config';
config.value = "modified";  // テスト中に変更

// test2.spec.ts
import { config } from './config';
console.log(config.value);  // "original" or "modified"?
// Jest の --isolateModules 設定によって異なる

// 対策: テストごとにリセットメカニズムを提供
export function resetConfig(): void {
  Object.assign(config, defaultConfig);
}
```

### 8.5 マイクロサービスにおける Singleton の罠

```
モノリス:
┌─────────────────────────────────┐
│ Application                     │
│   Singleton.getInstance() ──→ 1つのインスタンス
│                                 │
│   全てのリクエストが共有          │
└─────────────────────────────────┘

マイクロサービス:
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Instance1│  │ Instance2│  │ Instance3│
│ Singleton│  │ Singleton│  │ Singleton│
│ (state A)│  │ (state B)│  │ (state C)│
└──────────┘  └──────────┘  └──────────┘
  ↑ 各プロセスに独立した Singleton が存在
  ↑ 状態の同期が必要 → Redis 等の外部ストアが必要

教訓: Singleton = プロセス内Singleton であり、分散環境では不十分
```

---

## 9. アンチパターン

### アンチパターン 1: God Singleton（神シングルトン）

```typescript
// NG: 何でも詰め込む「神」シングルトン
class AppState {
  private static instance: AppState;

  user: User | null = null;
  theme: string = "light";
  cart: CartItem[] = [];
  notifications: Notification[] = [];
  searchHistory: string[] = [];
  recentProducts: Product[] = [];
  // ... 50以上のプロパティ

  static getInstance(): AppState {
    if (!this.instance) this.instance = new AppState();
    return this.instance;
  }
}

// 使用側: あらゆるモジュールが AppState に依存
function processOrder() {
  const state = AppState.getInstance();
  state.cart;           // カート機能が依存
  state.user;           // ユーザー機能が依存
  state.notifications;  // 通知機能が依存
}
```

**問題**:
- 単一責任原則（SRP）に違反
- あらゆるモジュールが AppState に依存し、変更の影響範囲が膨大
- テストで不要なプロパティまで初期化が必要

```typescript
// OK: ドメインごとに分割し、DI コンテナで管理
interface UserState {
  currentUser: User | null;
  login(credentials: Credentials): Promise<void>;
  logout(): void;
}

interface CartState {
  items: CartItem[];
  addItem(item: CartItem): void;
  removeItem(id: string): void;
}

interface NotificationState {
  notifications: Notification[];
  add(n: Notification): void;
  markRead(id: string): void;
}

// 各状態を独立して管理
container.bind<UserState>(TYPES.UserState).to(UserStateImpl).inSingletonScope();
container.bind<CartState>(TYPES.CartState).to(CartStateImpl).inSingletonScope();
container.bind<NotificationState>(TYPES.NotificationState).to(NotificationStateImpl).inSingletonScope();
```

### アンチパターン 2: テスト間での状態リーク

```typescript
// NG: テスト間で Singleton の状態がリークする
class Counter {
  private static instance: Counter;
  private count = 0;

  private constructor() {}

  static getInstance(): Counter {
    if (!this.instance) this.instance = new Counter();
    return this.instance;
  }

  increment(): void { this.count++; }
  getCount(): number { return this.count; }
}

// テスト（NG）
describe("feature A", () => {
  it("increments counter", () => {
    Counter.getInstance().increment();
    Counter.getInstance().increment();
    expect(Counter.getInstance().getCount()).toBe(2); // Pass
  });
});

describe("feature B", () => {
  it("starts at zero", () => {
    // 前のテストの状態が残っている！
    expect(Counter.getInstance().getCount()).toBe(0); // FAIL: 2
  });
});
```

```typescript
// OK: リセットメカニズムを提供する
class Counter {
  private static instance: Counter;
  private count = 0;

  private constructor() {}

  static getInstance(): Counter {
    if (!this.instance) this.instance = new Counter();
    return this.instance;
  }

  increment(): void { this.count++; }
  getCount(): number { return this.count; }

  // テスト用リセット（本番コードでは呼ばない）
  static resetForTesting(): void {
    this.instance = undefined as any;
  }
}

// テスト（OK）
beforeEach(() => {
  Counter.resetForTesting();
});
```

### アンチパターン 3: Singleton 内での副作用的な初期化

```typescript
// NG: コンストラクタで外部リソースに接続
class ApiClient {
  private static instance: ApiClient;
  private connection: WebSocket;

  private constructor() {
    // コンストラクタで WebSocket 接続を開始
    // → いつ getInstance() が呼ばれるか制御できない
    // → テストで実際の接続が発生する
    this.connection = new WebSocket("wss://api.example.com");
  }

  static getInstance(): ApiClient {
    if (!this.instance) this.instance = new ApiClient();
    return this.instance;
  }
}
```

```typescript
// OK: 明示的な初期化メソッドを分離
class ApiClient {
  private static instance: ApiClient;
  private connection: WebSocket | null = null;

  private constructor() {}

  static getInstance(): ApiClient {
    if (!this.instance) this.instance = new ApiClient();
    return this.instance;
  }

  async connect(url: string): Promise<void> {
    this.connection = new WebSocket(url);
    await this.waitForOpen();
  }

  private waitForOpen(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.connection!.onopen = () => resolve();
      this.connection!.onerror = (e) => reject(e);
    });
  }
}

// 使用: 初期化タイミングを制御できる
const client = ApiClient.getInstance();
await client.connect("wss://api.example.com");
```

---

## 10. トレードオフ分析

### 10.1 Singleton を使う場合の利点と欠点

```
利点                              欠点
+------------------------------+  +------------------------------+
| インスタンス数を制御できる    |  | グローバル状態になりがち      |
| メモリ効率が良い              |  | テストが困難（モック困難）    |
| 共有リソースの一貫性保証      |  | 結合度が上がる                |
| アクセスポイントが明確        |  | 並行処理の考慮が必要          |
| 遅延初期化が可能              |  | 状態のリセットが困難          |
+------------------------------+  +------------------------------+
```

### 10.2 判断フローチャート

```
インスタンスを1つに制限する必要がある？
│
├── No → Singleton 不要。通常のクラスまたは DI を使う
│
└── Yes
    │
    ├── DI コンテナを使えるか？
    │   ├── Yes → DI の Singleton スコープを使う（推奨）
    │   └── No → 次の質問へ
    │
    ├── 言語は何か？
    │   ├── Java → Enum Singleton
    │   ├── Kotlin → object 宣言
    │   ├── JS/TS → モジュールスコープ export
    │   ├── Python → メタクラス or デコレータ
    │   ├── Go → sync.Once
    │   └── その他 → Holder or DCL
    │
    └── テスト時にモック可能か確認
        ├── Yes → 実装を進める
        └── No → リセットメカニズムを追加する
```

---

## 11. 実践演習

### 演習 1: 基礎 — ロガー Singleton の実装

**課題**: アプリケーション全体で共有されるロガーを Singleton パターンで実装してください。

**要件**:
- ログレベル（debug / info / warn / error）をサポート
- ログメッセージにタイムスタンプを付与
- `getInstance()` によるアクセス
- ログ履歴を保持し、`getHistory()` で取得可能

```typescript
// === あなたの実装をここに書いてください ===

// ヒント: 以下のインタフェースを満たすこと
interface ILogger {
  debug(message: string): void;
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
  getHistory(): string[];
}
```

**期待される出力**:

```
const logger1 = Logger.getInstance();
const logger2 = Logger.getInstance();

logger1.info("Application started");
logger2.warn("Memory usage high");
logger1.error("Connection failed");

console.log(logger1 === logger2);
// true

console.log(logger1.getHistory());
// [
//   "[2026-01-15T10:30:00.000Z] [INFO] Application started",
//   "[2026-01-15T10:30:00.100Z] [WARN] Memory usage high",
//   "[2026-01-15T10:30:00.200Z] [ERROR] Connection failed"
// ]
```

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
type LogLevel = 'debug' | 'info' | 'warn' | 'error';

class Logger implements ILogger {
  private static instance: Logger | null = null;
  private history: string[] = [];
  private minLevel: LogLevel;

  private static readonly LEVELS: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
  };

  private constructor(minLevel: LogLevel = 'debug') {
    this.minLevel = minLevel;
  }

  static getInstance(minLevel?: LogLevel): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger(minLevel);
    }
    return Logger.instance;
  }

  private log(level: LogLevel, message: string): void {
    if (Logger.LEVELS[level] < Logger.LEVELS[this.minLevel]) return;

    const timestamp = new Date().toISOString();
    const entry = `[${timestamp}] [${level.toUpperCase()}] ${message}`;
    this.history.push(entry);
    console.log(entry);
  }

  debug(message: string): void { this.log('debug', message); }
  info(message: string): void { this.log('info', message); }
  warn(message: string): void { this.log('warn', message); }
  error(message: string): void { this.log('error', message); }

  getHistory(): string[] {
    return [...this.history]; // コピーを返す（防御的コピー）
  }

  // テスト用
  static resetForTesting(): void {
    Logger.instance = null;
  }
}
```

</details>

### 演習 2: 応用 — 設定マネージャ with 環境切替

**課題**: 環境（development / staging / production）に応じた設定を管理する Singleton を実装してください。

**要件**:
- モジュールスコープ Singleton として実装
- 環境変数 `NODE_ENV` から環境を判定
- 設定値の取得は型安全に行う
- 設定値の動的な上書き（override）をサポート
- 設定値のバリデーション

```typescript
// === あなたの実装をここに書いてください ===

// ヒント: 以下のインタフェースを満たすこと
interface ConfigManager {
  get<T>(key: string): T;
  get<T>(key: string, defaultValue: T): T;
  set(key: string, value: unknown): void;
  getEnvironment(): string;
  toJSON(): Record<string, unknown>;
}
```

**期待される出力**:

```
// NODE_ENV=development

console.log(configManager.getEnvironment());
// "development"

console.log(configManager.get<string>("apiBaseUrl"));
// "http://localhost:3000"

configManager.set("apiBaseUrl", "http://localhost:4000");
console.log(configManager.get<string>("apiBaseUrl"));
// "http://localhost:4000"

console.log(configManager.get<number>("maxRetries", 5));
// 5 (デフォルト値が返る)
```

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
type Environment = 'development' | 'staging' | 'production';

interface ConfigSchema {
  apiBaseUrl: string;
  logLevel: string;
  enableMetrics: boolean;
  maxRetries: number;
  dbHost: string;
  dbPort: number;
  [key: string]: unknown;
}

const defaults: Record<Environment, ConfigSchema> = {
  development: {
    apiBaseUrl: 'http://localhost:3000',
    logLevel: 'debug',
    enableMetrics: false,
    maxRetries: 1,
    dbHost: 'localhost',
    dbPort: 5432,
  },
  staging: {
    apiBaseUrl: 'https://staging-api.example.com',
    logLevel: 'info',
    enableMetrics: true,
    maxRetries: 3,
    dbHost: 'staging-db.example.com',
    dbPort: 5432,
  },
  production: {
    apiBaseUrl: 'https://api.example.com',
    logLevel: 'warn',
    enableMetrics: true,
    maxRetries: 5,
    dbHost: 'prod-db.example.com',
    dbPort: 5432,
  },
};

class ConfigManagerImpl implements ConfigManager {
  private env: Environment;
  private config: Record<string, unknown>;

  constructor() {
    this.env = (process.env.NODE_ENV as Environment) ?? 'development';
    if (!defaults[this.env]) {
      throw new Error(`Unknown environment: ${this.env}`);
    }
    this.config = { ...defaults[this.env] };
  }

  get<T>(key: string, defaultValue?: T): T {
    const value = this.config[key];
    if (value === undefined) {
      if (defaultValue !== undefined) return defaultValue;
      throw new Error(`Config key not found: ${key}`);
    }
    return value as T;
  }

  set(key: string, value: unknown): void {
    this.config[key] = value;
  }

  getEnvironment(): string {
    return this.env;
  }

  toJSON(): Record<string, unknown> {
    return { ...this.config };
  }
}

// モジュールスコープ Singleton
export const configManager: ConfigManager = new ConfigManagerImpl();
```

</details>

### 演習 3: 発展 — DI コンテナ対応の Singleton

**課題**: DI コンテナ（InversifyJS 風）を使って、テスト可能な Singleton サービスを設計してください。

**要件**:
- `ICacheService` インタフェースを定義
- `RedisCacheService` として本番実装を作成
- `InMemoryCacheService` としてテスト用実装を作成
- DI コンテナで Singleton スコープを設定
- テストコードでモック差し替えを実演

```typescript
// === あなたの実装をここに書いてください ===

// ヒント: 以下のインタフェースを満たすこと
interface ICacheService {
  get<T>(key: string): Promise<T | null>;
  set<T>(key: string, value: T, ttlSeconds?: number): Promise<void>;
  delete(key: string): Promise<boolean>;
  clear(): Promise<void>;
}
```

**期待される出力**:

```
// 本番
const cache = container.get<ICacheService>(TYPES.Cache);
await cache.set("user:1", { name: "Taro" }, 3600);
const user = await cache.get("user:1");
console.log(user); // { name: "Taro" }

// テスト
const testCache = testContainer.get<ICacheService>(TYPES.Cache);
await testCache.set("key", "value");
console.log(await testCache.get("key")); // "value"
await testCache.clear();
console.log(await testCache.get("key")); // null
```

<details>
<summary>模範解答（クリックで展開）</summary>

```typescript
import { Container, injectable, inject } from "inversify";

const TYPES = {
  Cache: Symbol.for("ICacheService"),
  Logger: Symbol.for("ILogger"),
};

// インタフェース定義（上述のICacheService）

// 本番実装
@injectable()
class RedisCacheService implements ICacheService {
  private client: any; // Redis client

  constructor(@inject(TYPES.Logger) private logger: ILogger) {
    this.logger.info("RedisCacheService: 初期化");
    // this.client = createClient({ url: process.env.REDIS_URL });
  }

  async get<T>(key: string): Promise<T | null> {
    this.logger.debug(`Cache GET: ${key}`);
    const value = await this.client.get(key);
    return value ? JSON.parse(value) : null;
  }

  async set<T>(key: string, value: T, ttlSeconds?: number): Promise<void> {
    this.logger.debug(`Cache SET: ${key}`);
    const serialized = JSON.stringify(value);
    if (ttlSeconds) {
      await this.client.setex(key, ttlSeconds, serialized);
    } else {
      await this.client.set(key, serialized);
    }
  }

  async delete(key: string): Promise<boolean> {
    const result = await this.client.del(key);
    return result > 0;
  }

  async clear(): Promise<void> {
    await this.client.flushdb();
  }
}

// テスト用実装
@injectable()
class InMemoryCacheService implements ICacheService {
  private store = new Map<string, { value: string; expiresAt?: number }>();

  async get<T>(key: string): Promise<T | null> {
    const entry = this.store.get(key);
    if (!entry) return null;
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      this.store.delete(key);
      return null;
    }
    return JSON.parse(entry.value);
  }

  async set<T>(key: string, value: T, ttlSeconds?: number): Promise<void> {
    this.store.set(key, {
      value: JSON.stringify(value),
      expiresAt: ttlSeconds ? Date.now() + ttlSeconds * 1000 : undefined,
    });
  }

  async delete(key: string): Promise<boolean> {
    return this.store.delete(key);
  }

  async clear(): Promise<void> {
    this.store.clear();
  }
}

// 本番コンテナ
const container = new Container();
container.bind<ICacheService>(TYPES.Cache)
  .to(RedisCacheService)
  .inSingletonScope();

// テストコンテナ
const testContainer = new Container();
testContainer.bind<ICacheService>(TYPES.Cache)
  .to(InMemoryCacheService)
  .inSingletonScope();

// テスト例
describe("UserService", () => {
  let cache: ICacheService;

  beforeEach(() => {
    cache = testContainer.get<ICacheService>(TYPES.Cache);
  });

  afterEach(async () => {
    await cache.clear(); // 状態リーク防止
  });

  it("should cache user data", async () => {
    await cache.set("user:1", { name: "Taro" }, 3600);
    const user = await cache.get("user:1");
    expect(user).toEqual({ name: "Taro" });
  });
});
```

</details>

---

## 12. FAQ

### Q1: Singleton はいつ使うべきですか？

ロガー、設定オブジェクト、コネクションプール、キャッシュマネージャなど、**アプリケーション全体で共有され、複数インスタンスが存在すると矛盾を起こすリソース**に対して使います。ただし、DI コンテナが利用可能なら、そちらで Singleton スコープを設定する方が望ましいです。

**判断基準チェックリスト:**

| チェック項目 | Yes なら Singleton 検討 |
|-------------|----------------------|
| 複数インスタンスが存在するとバグになるか？ | Yes → 強い根拠 |
| 共有状態の一貫性が必要か？ | Yes → 根拠あり |
| リソースの初期化コストが高いか？ | Yes → 根拠あり |
| DI コンテナが使えないか？ | Yes → Singleton を自前実装 |

### Q2: Singleton はなぜ「アンチパターン」と呼ばれることがあるのですか？

Singleton 自体がアンチパターンなのではなく、**濫用**がアンチパターンです。以下の問題を引き起こしやすい:

1. **グローバル状態**: どこからでもアクセスでき、状態変更の追跡が困難
2. **テスト困難**: モックへの差し替えが難しい（特にクラス内 Singleton）
3. **結合度の増大**: 具象クラスへの直接依存が生まれる
4. **並行処理の複雑化**: 共有状態へのアクセスに排他制御が必要
5. **隠れた依存**: メソッドシグネチャに現れない依存関係が生じる

```typescript
// 隠れた依存の例
function processOrder(order: Order): void {
  // 関数のシグネチャからは Database と Logger への依存が見えない
  const db = Database.getInstance();      // 隠れた依存
  const logger = Logger.getInstance();    // 隠れた依存

  db.save(order);
  logger.info(`Order ${order.id} processed`);
}

// DI で明示的にする
function processOrder(
  order: Order,
  db: IDatabase,       // 依存が明示的
  logger: ILogger      // 依存が明示的
): void {
  db.save(order);
  logger.info(`Order ${order.id} processed`);
}
```

### Q3: ES Module で export したオブジェクトは Singleton ですか？

はい。Node.js や主要バンドラ（webpack, Vite, esbuild）はモジュールを一度だけ評価しキャッシュします。そのため `export const x = new X()` は事実上 Singleton です。

ただし、以下の注意点があります:

| 環境 | 動作 |
|------|------|
| Node.js (CJS) | `require()` はキャッシュする。ただしパスが異なると別モジュール扱い |
| Node.js (ESM) | `import` はキャッシュする |
| Jest | `--isolateModules` でモジュールキャッシュをリセット可能 |
| Webpack | バンドル内で1度だけ評価される |
| SSR (Next.js) | サーバーリクエスト間で共有される（注意が必要） |

### Q4: Singleton と静的クラス（static class）の違いは何ですか？

```
Singleton:
- インスタンスが1つ存在する
- インタフェースを実装できる（ポリモーフィズム）
- DI コンテナで管理可能
- 遅延初期化が可能
- 状態を持てる

静的クラス:
- インスタンスが存在しない
- インタフェースを実装できない
- DI コンテナで管理不可
- クラスロード時に確定
- ステートレス（推奨）
```

Singleton を選ぶべき場合: インタフェースへの準拠、DI 対応、遅延初期化が必要な場合
静的クラスを選ぶべき場合: ステートレスなユーティリティ（Math.max(), String.format() 等）

### Q5: マイクロサービス環境で Singleton は使えますか？

使えますが、Singleton の「唯一性」はプロセス内に限定されることを理解してください。複数のサービスインスタンスが存在する場合、各プロセスに独立した Singleton が存在します。

```
解決策:
1. ステートレス Singleton: 状態を持たなければ問題なし（Logger, Config等）
2. 外部ストア: 状態を Redis や DB に保存し、Singleton はアクセス層として機能
3. リーダー選出: 分散ロック（Redis の SETNX等）で1つのインスタンスだけが処理
```

---

## 13. まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | インスタンスを1つに制限し、グローバルアクセスを提供 |
| 本質的な問題 | 共有リソースの一貫性保証 |
| 利点 | 共有リソースの一元管理、メモリ効率、遅延初期化 |
| 欠点 | グローバル状態、テスト困難、結合度増大、並行処理の複雑化 |
| スレッドセーフ | volatile/synchronized、Holder、Enum、sync.Once 等で対応 |
| Java 推奨 | Enum Singleton（Effective Java 推奨） |
| Kotlin 推奨 | object 宣言 |
| JS/TS 推奨 | モジュールスコープ export |
| Python 推奨 | メタクラス or デコレータ |
| Go 推奨 | sync.Once |
| 最善の代替 | DI コンテナの Singleton スコープ |
| 判断基準 | 複数インスタンスが矛盾を起こす場合のみ使用 |

---

## 次に読むべきガイド

- [Factory Method / Abstract Factory](./01-factory.md) -- 生成の委譲と抽象化。Singleton Registry と Factory の組み合わせ
- [Builder パターン](./02-builder.md) -- 複雑なオブジェクト構築。Singleton の設定構築に活用
- [Prototype パターン](./03-prototype.md) -- クローンによる生成。Singleton との対比
- [SOLID 原則](../../../clean-code-principles/docs/00-principles/01-solid.md) -- 単一責任原則と依存性逆転原則
- [Observer パターン](../02-behavioral/00-observer.md) -- イベントバス Singleton との組み合わせ
- [Facade パターン](../01-structural/02-facade.md) -- Singleton が提供する簡略化インタフェース

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley. -- Singleton パターンの原典
2. Bloch, J. (2018). *Effective Java* (3rd ed.). Addison-Wesley. Item 3: Enforce the singleton property with a private constructor or an enum type.
3. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media. Chapter 5: The Singleton Pattern.
4. Fowler, M. (2004). *Inversion of Control Containers and the Dependency Injection pattern*. martinfowler.com. https://martinfowler.com/articles/injection.html
5. Refactoring.Guru -- Singleton. https://refactoring.guru/design-patterns/singleton
6. Microsoft .NET Documentation -- Dependency injection guidelines. https://learn.microsoft.com/en-us/dotnet/core/extensions/dependency-injection-guidelines
