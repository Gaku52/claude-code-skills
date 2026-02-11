# Singleton パターン

> インスタンスがアプリケーション全体で **ただ1つ** であることを保証し、そのグローバルアクセスポイントを提供する生成パターン。

---

## この章で学ぶこと

1. Singleton パターンの目的・構造と、スレッドセーフな実装方法
2. DI（依存性注入）による Singleton の代替手法とテスト容易性の確保
3. Singleton の濫用が招く問題と、適切な利用場面の見極め方

---

## 1. Singleton の構造

```
+---------------------------+
|       Singleton           |
+---------------------------+
| - instance: Singleton     |
| - data: any               |
+---------------------------+
| - constructor()           |
| + getInstance(): Singleton|
| + getData(): any          |
+---------------------------+
        |
        | 1つだけ生成
        v
  +-----------+
  | instance  |
  +-----------+
```

---

## 2. 基本実装（TypeScript）

### コード例 1: クラシック Singleton

```typescript
class Singleton {
  private static instance: Singleton | null = null;
  private value: number;

  private constructor(value: number) {
    this.value = value;
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
}

// 使用例
const a = Singleton.getInstance();
const b = Singleton.getInstance();
console.log(a === b); // true
```

### コード例 2: モジュールスコープ Singleton（推奨）

```typescript
// config.ts — ES Module 自体が Singleton として振る舞う
class AppConfig {
  readonly dbHost: string;
  readonly dbPort: number;

  constructor() {
    this.dbHost = process.env.DB_HOST ?? "localhost";
    this.dbPort = Number(process.env.DB_PORT ?? 5432);
  }
}

// モジュールレベルで1度だけインスタンス化
export const appConfig = new AppConfig();
```

### コード例 3: スレッドセーフ（Java — Double-Checked Locking）

```java
public class Singleton {
    private static volatile Singleton instance;

    private Singleton() {}

    public static Singleton getInstance() {
        if (instance == null) {                  // 1st check (no lock)
            synchronized (Singleton.class) {
                if (instance == null) {          // 2nd check (lock)
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

### コード例 4: Python — メタクラス Singleton

```python
class SingletonMeta(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "connected"

db1 = Database()
db2 = Database()
assert db1 is db2  # True
```

### コード例 5: DI コンテナによる Singleton ライフタイム

```typescript
// InversifyJS の例
import { Container } from "inversify";

const container = new Container();

container
  .bind<Logger>(TYPES.Logger)
  .to(ConsoleLogger)
  .inSingletonScope();   // コンテナが1インスタンスを保証

const logger1 = container.get<Logger>(TYPES.Logger);
const logger2 = container.get<Logger>(TYPES.Logger);
console.log(logger1 === logger2); // true
```

---

## 3. スレッドセーフ実装の比較図

```
┌──────────────────────────────────────────────────────────┐
│            スレッドセーフ Singleton 実装戦略              │
├──────────────┬───────────────────────────────────────────┤
│  Eager Init  │  クラスロード時に生成（最も単純）          │
│              │  static instance = new Singleton()        │
├──────────────┼───────────────────────────────────────────┤
│  DCL         │  Double-Checked Locking                   │
│              │  volatile + synchronized                  │
├──────────────┼───────────────────────────────────────────┤
│  Holder      │  内部クラスの遅延ロード                   │
│              │  Bill Pugh Singleton                      │
├──────────────┼───────────────────────────────────────────┤
│  Enum (Java) │  JVM が保証。直列化にも対応               │
└──────────────┴───────────────────────────────────────────┘
```

---

## 4. 比較表

### 比較表 1: Singleton 実装手法の比較

| 手法 | 遅延初期化 | スレッドセーフ | 実装難易度 | 直列化対応 |
|------|:---:|:---:|:---:|:---:|
| Eager Init | No | Yes | 低 | 要対応 |
| Lazy (同期なし) | Yes | No | 低 | 要対応 |
| DCL | Yes | Yes | 中 | 要対応 |
| Holder パターン | Yes | Yes | 中 | 要対応 |
| Enum (Java) | No | Yes | 低 | 自動 |
| モジュールスコープ (JS/TS) | Yes* | N/A | 低 | N/A |

*モジュール初回インポート時に評価される。

### 比較表 2: Singleton vs DI コンテナ

| 観点 | クラス内 Singleton | DI コンテナ Singleton |
|------|---|---|
| テスト容易性 | 低い（モック困難） | 高い（差し替え容易） |
| 結合度 | 高い（直接参照） | 低い（インタフェース経由） |
| ライフタイム管理 | クラス自身 | コンテナ |
| グローバル状態 | 露出する | 隠蔽可能 |
| 柔軟性 | 低い | 高い |

---

## 5. DI による代替フロー

```
  クライアント
      |
      | 依存を注入してもらう
      v
 +----------+       +-----------+
 | DI Container | -->| Interface |
 +----------+       +-----------+
      |                   ^
      | .inSingletonScope |
      v                   |
 +----------+        +-----------+
 | 1 instance|  impl | ConcreteA |
 +----------+        +-----------+
```

---

## 6. アンチパターン

### アンチパターン 1: God Singleton

```typescript
// BAD: 何でも詰め込む「神」シングルトン
class AppState {
  private static instance: AppState;
  user: User | null = null;
  theme: string = "light";
  cart: CartItem[] = [];
  notifications: Notification[] = [];
  // ... 50以上のプロパティ
}
```

**問題**: 単一責任原則に違反。あらゆるモジュールが依存し、変更の影響範囲が膨大になる。

**改善**: ドメインごとに分割し、DI コンテナで管理する。

### アンチパターン 2: Singleton をテスト間で共有

```typescript
// BAD: テスト間で状態がリークする
describe("feature A", () => {
  it("sets value", () => {
    Singleton.getInstance().setValue(100);
  });
});

describe("feature B", () => {
  it("reads stale value", () => {
    // 前のテストの 100 が残っている！
    expect(Singleton.getInstance().getValue()).toBe(0); // FAIL
  });
});
```

**改善**: テストごとにインスタンスをリセットするか、DI でモックを注入する。

---

## 7. FAQ

### Q1: Singleton はいつ使うべきですか？

ロガー、設定オブジェクト、コネクションプールなど、アプリケーション全体で共有され、複数インスタンスが存在すると矛盾を起こすリソースに対して使います。ただし、DI コンテナが利用可能なら、そちらで Singleton スコープを設定する方が望ましいです。

### Q2: Singleton はなぜ「アンチパターン」と呼ばれることがあるのですか？

グローバル状態を生み出し、テスト困難性・結合度の増大・並行処理の問題を招きやすいためです。パターン自体が悪いのではなく、**濫用** が問題です。

### Q3: ES Module で export したオブジェクトは Singleton ですか？

はい。Node.js や主要バンドラはモジュールを一度だけ評価しキャッシュします。そのため `export const x = new X()` は事実上 Singleton です。ただし、テスト環境ではモジュールキャッシュがリセットされる場合があるため注意が必要です。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | インスタンスを1つに制限し、グローバルアクセスを提供 |
| 利点 | 共有リソースの一元管理、メモリ効率 |
| 欠点 | グローバル状態、テスト困難、結合度増大 |
| スレッドセーフ | volatile/synchronized、Holder、Enum 等で対応 |
| 推奨代替 | DI コンテナの Singleton スコープ |
| JS/TS 推奨 | モジュールスコープ export |

---

## 次に読むべきガイド

- [Factory Method / Abstract Factory](./01-factory.md) — 生成の委譲と抽象化
- [Builder パターン](./02-builder.md) — 複雑なオブジェクト構築
- [SOLID 原則](../../../clean-code-principles/docs/00-principles/01-solid.md) — 単一責任原則と依存性逆転

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
3. Fowler, M. (2004). *Inversion of Control Containers and the Dependency Injection pattern*. martinfowler.com.
