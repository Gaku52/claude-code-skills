# Adapter パターン

> 互換性のないインタフェースを持つクラスを **ラッパー** で包み、クライアントが期待するインタフェースに変換する構造パターン。

---

## この章で学ぶこと

1. Adapter パターンの2つの形態（オブジェクトアダプタ・クラスアダプタ）と選択基準
2. 既存ライブラリやレガシーコードとの統合におけるアダプタの実践的活用
3. Adapter と Facade・Decorator の違い、過剰適用の回避

---

## 1. Adapter の構造

```
クライアント                   アダプタ                     既存クラス
+--------+     uses     +-------------+    delegates   +-----------+
| Client |------------->|   Adapter   |--------------->| Adaptee   |
+--------+              +-------------+                +-----------+
    |                   | + request() |                | + legacy() |
    | expects           |   ↓         |                +-----------+
    v                   | adaptee     |
+-----------+           | .legacy()   |
| Target    |           +-------------+
| (interface)|
| + request()|
+-----------+
```

---

## 2. オブジェクトアダプタ vs クラスアダプタ

```
オブジェクトアダプタ（委譲）          クラスアダプタ（継承）
┌───────────┐                     ┌───────────────────┐
│  Adapter  │                     │     Adapter       │
│           │ has-a ┌────────┐    │ extends Adaptee   │
│ -adaptee ─┼──────>│Adaptee │    │ implements Target │
│           │       └────────┘    └───────────────────┘
└───────────┘                         ↑           ↑
                                  Adaptee       Target
                                  (継承)      (実装)

推奨: オブジェクトアダプタ（柔軟性が高い）
```

---

## 3. コード例

### コード例 1: 外部ライブラリの Adapter

```typescript
// 既存の外部ライブラリ（変更不可）
class LegacyXmlParser {
  parseXml(xmlString: string): XmlDocument {
    // XML をパースして独自形式で返す
    return { root: xmlString, format: "xml" };
  }
}

// クライアントが期待するインタフェース
interface DataParser {
  parse(input: string): Record<string, unknown>;
}

// Adapter
class XmlParserAdapter implements DataParser {
  private legacyParser = new LegacyXmlParser();

  parse(input: string): Record<string, unknown> {
    const xmlDoc = this.legacyParser.parseXml(input);
    return this.convertToRecord(xmlDoc);
  }

  private convertToRecord(doc: XmlDocument): Record<string, unknown> {
    // XML Document を汎用 Record に変換
    return { data: doc.root };
  }
}

// 使用: クライアントは DataParser だけを知っている
function processData(parser: DataParser, input: string) {
  const result = parser.parse(input);
  console.log(result);
}

processData(new XmlParserAdapter(), "<user>Taro</user>");
```

### コード例 2: ログライブラリの統一

```typescript
// アプリ内の統一ログインタフェース
interface AppLogger {
  info(message: string): void;
  error(message: string, error?: Error): void;
}

// Winston (外部ライブラリ) の Adapter
class WinstonAdapter implements AppLogger {
  constructor(private winston: WinstonLogger) {}

  info(message: string): void {
    this.winston.log("info", message);
  }

  error(message: string, error?: Error): void {
    this.winston.log("error", message, { error });
  }
}

// Pino (外部ライブラリ) の Adapter
class PinoAdapter implements AppLogger {
  constructor(private pino: PinoLogger) {}

  info(message: string): void {
    this.pino.info(message);
  }

  error(message: string, error?: Error): void {
    this.pino.error({ err: error }, message);
  }
}
```

### コード例 3: Python — Adapter

```python
from abc import ABC, abstractmethod

class PaymentGateway(ABC):
    @abstractmethod
    def pay(self, amount: float, currency: str) -> bool: ...

# 外部SDK（変更不可）
class StripeSDK:
    def create_charge(self, amount_cents: int, cur: str) -> dict:
        return {"status": "succeeded"}

class StripeAdapter(PaymentGateway):
    def __init__(self, sdk: StripeSDK):
        self._sdk = sdk

    def pay(self, amount: float, currency: str) -> bool:
        cents = int(amount * 100)
        result = self._sdk.create_charge(cents, currency)
        return result["status"] == "succeeded"

gateway: PaymentGateway = StripeAdapter(StripeSDK())
gateway.pay(29.99, "USD")
```

### コード例 4: イベントシステムの Adapter

```typescript
// DOM イベントと独自イベントシステムの橋渡し
interface AppEventEmitter {
  on(event: string, handler: (data: unknown) => void): void;
  emit(event: string, data: unknown): void;
}

class DOMEventAdapter implements AppEventEmitter {
  constructor(private element: HTMLElement) {}

  on(event: string, handler: (data: unknown) => void): void {
    this.element.addEventListener(event, (e: Event) => {
      handler((e as CustomEvent).detail);
    });
  }

  emit(event: string, data: unknown): void {
    this.element.dispatchEvent(
      new CustomEvent(event, { detail: data })
    );
  }
}
```

### コード例 5: 関数アダプタ（高階関数）

```typescript
// コールバック形式 → Promise 形式のアダプタ
function promisify<T>(
  fn: (callback: (err: Error | null, result: T) => void) => void
): () => Promise<T> {
  return () =>
    new Promise((resolve, reject) => {
      fn((err, result) => {
        if (err) reject(err);
        else resolve(result);
      });
    });
}

// 使用
const readFileAsync = promisify<string>(fs.readFile.bind(fs, "file.txt"));
const content = await readFileAsync();
```

---

## 4. 比較表

### 比較表 1: Adapter vs Facade vs Decorator

| 観点 | Adapter | Facade | Decorator |
|------|---------|--------|-----------|
| 目的 | インタフェース変換 | 複雑さの隠蔽 | 機能追加 |
| 対象 | 1つのクラス | 複数のクラス群 | 1つのオブジェクト |
| インタフェース | 変換する | 単純化する | 同じまま |
| 既存コード | 変更不可 | 変更不要 | 変更不要 |

### 比較表 2: オブジェクトアダプタ vs クラスアダプタ

| 観点 | オブジェクトアダプタ | クラスアダプタ |
|------|:---:|:---:|
| 実現方法 | 委譲（has-a） | 継承（is-a） |
| 複数 Adaptee 対応 | Yes | No |
| Adaptee のメソッド上書き | No | Yes |
| 言語制約 | なし | 多重継承が必要 |
| 推奨度 | 高い | 低い |

---

## 5. アンチパターン

### アンチパターン 1: 薄すぎるアダプタ

```typescript
// BAD: 単にメソッド名を変えただけ
class UselessAdapter implements Target {
  constructor(private adaptee: Adaptee) {}

  doSomething(): void {
    this.adaptee.doSomething(); // まったく同じシグネチャ
  }
}
```

**問題**: インタフェースが同じならアダプタは不要。不要な間接層はコードの可読性を下げる。

### アンチパターン 2: アダプタにビジネスロジックを追加

```typescript
// BAD: Adapter が変換以上の責任を持つ
class OrderAdapter implements NewOrderService {
  convert(legacy: LegacyOrder): Order {
    const order = this.mapFields(legacy);
    order.applyTax();           // ビジネスロジック
    order.validateInventory();  // ビジネスロジック
    return order;
  }
}
```

**改善**: Adapter は変換のみ。ビジネスロジックはサービス層に配置する。

---

## 6. FAQ

### Q1: Adapter はレガシーコード以外でも使いますか？

はい。外部 API、サードパーティライブラリ、異なるチームが開発したモジュールとの統合にも頻繁に使います。テストでモックに差し替える際のインタフェース統一にも有効です。

### Q2: TypeScript でアダプタを書くとき、クラスと関数のどちらが良いですか？

変換が単純なら関数（高階関数、ラッパー関数）で十分です。状態の保持やライフサイクル管理が必要ならクラスを使います。

### Q3: Adapter が多数になった場合の管理方法は？

`adapters/` ディレクトリに集約し、Factory パターンと組み合わせて適切なアダプタを選択する方法が一般的です。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | 互換性のないインタフェースを変換 |
| オブジェクトアダプタ | 委譲ベース、推奨 |
| クラスアダプタ | 継承ベース、多重継承が必要 |
| 適用場面 | 外部ライブラリ、レガシー統合 |
| 注意 | 変換のみに責務を限定する |

---

## 次に読むべきガイド

- [Decorator パターン](./01-decorator.md) — 動的な機能追加
- [Facade パターン](./02-facade.md) — 複雑さの隠蔽
- [Strategy パターン](../02-behavioral/01-strategy.md) — アルゴリズムの交換

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
3. Refactoring.Guru — Adapter. https://refactoring.guru/design-patterns/adapter
