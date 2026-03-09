# ポリモーフィズム

> ポリモーフィズム（多態性）は「同じインターフェースで異なる実装を呼び出せる」仕組み。OOPの最も強力な概念であり、柔軟で拡張性の高い設計の基盤。

## この章で学ぶこと

- [ ] 3種類のポリモーフィズムを理解する
- [ ] 動的ディスパッチの仕組み（vtable）を把握する
- [ ] ポリモーフィズムの実践的な活用パターンを学ぶ
- [ ] 静的ディスパッチと動的ディスパッチの使い分けを理解する
- [ ] パラメトリックポリモーフィズム（ジェネリクス）を活用する
- [ ] 実務での設計パターンとの関連を把握する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [継承](./01-inheritance.md) の内容を理解していること

---

## 1. 3種類のポリモーフィズム

```
1. サブタイプポリモーフィズム（Subtype / Inclusion）
   → 親型の変数にサブクラスのオブジェクトを代入
   → OOPの「ポリモーフィズム」はこれを指すことが多い
   → 実行時に実際の型のメソッドが呼ばれる

2. パラメトリックポリモーフィズム（Parametric）
   → ジェネリクス。型パラメータで汎用的なコードを書く
   → List<T>, Map<K,V> など

3. アドホックポリモーフィズム（Ad-hoc）
   → メソッドオーバーロード。同名で引数の型が異なるメソッド
   → 演算子オーバーロードも含む

比較:
  ┌────────────────┬──────────────┬───────────────┬──────────────┐
  │                │ サブタイプ    │ パラメトリック │ アドホック    │
  ├────────────────┼──────────────┼───────────────┼──────────────┤
  │ 決定タイミング │ 実行時       │ コンパイル時   │ コンパイル時  │
  │ 実現手段       │ 継承/IF実装  │ ジェネリクス   │ オーバーロード│
  │ 型の統一性     │ 共通の親型   │ 型パラメータ   │ 同名異引数    │
  │ 代表例         │ Shape.area() │ List<T>       │ add(int,int)  │
  └────────────────┴──────────────┴───────────────┴──────────────┘
```

---

## 2. サブタイプポリモーフィズム

```
  Shape（インターフェース/抽象クラス）
    area(): number
    draw(): void
       ↑
  ┌────┼────┬───────────┐
  ▼    ▼    ▼           ▼
Circle  Rect  Triangle  Polygon
 area() area()  area()   area()
 各々が独自の実装を持つ

  shapes: Shape[] = [Circle, Rect, Triangle, ...]
  for (shape of shapes) {
    shape.area()  ← 実行時に正しい実装が呼ばれる
  }
```

### 2.1 基本的なサブタイプポリモーフィズム

```typescript
// TypeScript: サブタイプポリモーフィズム
interface Shape {
  area(): number;
  perimeter(): number;
  describe(): string;
}

class Circle implements Shape {
  constructor(private radius: number) {}

  area(): number {
    return Math.PI * this.radius ** 2;
  }

  perimeter(): number {
    return 2 * Math.PI * this.radius;
  }

  describe(): string {
    return `円（半径: ${this.radius}）`;
  }
}

class Rectangle implements Shape {
  constructor(private width: number, private height: number) {}

  area(): number {
    return this.width * this.height;
  }

  perimeter(): number {
    return 2 * (this.width + this.height);
  }

  describe(): string {
    return `長方形（${this.width} x ${this.height}）`;
  }
}

class Triangle implements Shape {
  constructor(
    private a: number,
    private b: number,
    private c: number,
    private height: number,
  ) {}

  area(): number {
    // ヘロンの公式
    const s = (this.a + this.b + this.c) / 2;
    return Math.sqrt(s * (s - this.a) * (s - this.b) * (s - this.c));
  }

  perimeter(): number {
    return this.a + this.b + this.c;
  }

  describe(): string {
    return `三角形（辺: ${this.a}, ${this.b}, ${this.c}）`;
  }
}

// ポリモーフィズム: Shape型で統一的に扱う
function printShapeInfo(shape: Shape): void {
  console.log(`${shape.describe()}: 面積=${shape.area().toFixed(2)}`);
}

// 形状の面積合計を計算（Shape型のみに依存）
function totalArea(shapes: Shape[]): number {
  return shapes.reduce((sum, shape) => sum + shape.area(), 0);
}

// 面積でソート（Shape型のみに依存）
function sortByArea(shapes: Shape[]): Shape[] {
  return [...shapes].sort((a, b) => a.area() - b.area());
}

const shapes: Shape[] = [
  new Circle(5),
  new Rectangle(3, 4),
  new Circle(10),
  new Triangle(3, 4, 5, 3.5),
];

shapes.forEach(printShapeInfo);
// 円（半径: 5）: 面積=78.54
// 長方形（3 x 4）: 面積=12.00
// 円（半径: 10）: 面積=314.16
// 三角形（辺: 3, 4, 5）: 面積=6.00

console.log(`合計面積: ${totalArea(shapes).toFixed(2)}`);
// 合計面積: 410.70
```

### 2.2 インターフェースベースのポリモーフィズム

```python
# Python: プロトコルベースのポリモーフィズム（ダックタイピング + 型ヒント）
from typing import Protocol, runtime_checkable


@runtime_checkable
class Renderable(Protocol):
    """描画可能なオブジェクトのプロトコル"""
    def render(self) -> str: ...
    def width(self) -> int: ...
    def height(self) -> int: ...


@runtime_checkable
class Clickable(Protocol):
    """クリック可能なオブジェクトのプロトコル"""
    def on_click(self, x: int, y: int) -> None: ...
    def is_point_inside(self, x: int, y: int) -> bool: ...


class TextLabel:
    """テキストラベル（Renderable のみ実装）"""
    def __init__(self, text: str, x: int = 0, y: int = 0):
        self.text = text
        self.x = x
        self.y = y

    def render(self) -> str:
        return f'<label x="{self.x}" y="{self.y}">{self.text}</label>'

    def width(self) -> int:
        return len(self.text) * 8  # 1文字8px想定

    def height(self) -> int:
        return 16


class Button:
    """ボタン（Renderable + Clickable 両方を実装）"""
    def __init__(self, label: str, x: int = 0, y: int = 0,
                 w: int = 100, h: int = 30):
        self.label = label
        self.x = x
        self.y = y
        self._width = w
        self._height = h
        self._click_handler: list[callable] = []

    def render(self) -> str:
        return f'<button x="{self.x}" y="{self.y}" w="{self._width}" h="{self._height}">{self.label}</button>'

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def on_click(self, x: int, y: int) -> None:
        print(f"Button '{self.label}' clicked at ({x}, {y})")
        for handler in self._click_handler:
            handler(x, y)

    def is_point_inside(self, x: int, y: int) -> bool:
        return (self.x <= x <= self.x + self._width and
                self.y <= y <= self.y + self._height)

    def add_click_handler(self, handler: callable) -> None:
        self._click_handler.append(handler)


class Image:
    """画像（Renderable + Clickable 両方を実装）"""
    def __init__(self, src: str, x: int = 0, y: int = 0,
                 w: int = 200, h: int = 150):
        self.src = src
        self.x = x
        self.y = y
        self._width = w
        self._height = h

    def render(self) -> str:
        return f'<img src="{self.src}" x="{self.x}" y="{self.y}" w="{self._width}" h="{self._height}" />'

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def on_click(self, x: int, y: int) -> None:
        print(f"Image '{self.src}' clicked at ({x}, {y})")

    def is_point_inside(self, x: int, y: int) -> bool:
        return (self.x <= x <= self.x + self._width and
                self.y <= y <= self.y + self._height)


# ポリモーフィズムの活用: 型に依存しない汎用関数
def render_all(elements: list[Renderable]) -> str:
    """Renderable を実装する全ての要素を描画"""
    return "\n".join(el.render() for el in elements)

def calculate_total_area(elements: list[Renderable]) -> int:
    """全要素の面積合計を計算"""
    return sum(el.width() * el.height() for el in elements)

def handle_click(clickables: list[Clickable], x: int, y: int) -> None:
    """クリック位置に該当する要素のクリックハンドラを呼ぶ"""
    for element in clickables:
        if element.is_point_inside(x, y):
            element.on_click(x, y)


# 使用例
elements: list[Renderable] = [
    TextLabel("こんにちは", x=10, y=10),
    Button("送信", x=10, y=40),
    Image("logo.png", x=10, y=80),
]

print(render_all(elements))
print(f"合計面積: {calculate_total_area(elements)} px^2")

# プロトコルの型チェック
print(isinstance(Button("test"), Renderable))  # True
print(isinstance(Button("test"), Clickable))   # True
print(isinstance(TextLabel("test"), Clickable)) # False
```

### 2.3 Java でのサブタイプポリモーフィズム

```java
// Java: インターフェースによるポリモーフィズム

// 支払い処理のインターフェース
public interface PaymentProcessor {
    PaymentResult process(PaymentRequest request);
    boolean supports(String paymentMethod);
    String getProviderName();
}

// 各プロバイダの実装
public class StripeProcessor implements PaymentProcessor {
    private final StripeClient client;

    public StripeProcessor(String apiKey) {
        this.client = new StripeClient(apiKey);
    }

    @Override
    public PaymentResult process(PaymentRequest request) {
        // Stripe API を使った決済処理
        try {
            var charge = client.charges().create(
                request.getAmount(),
                request.getCurrency(),
                request.getToken()
            );
            return PaymentResult.success(charge.getId(), "stripe");
        } catch (StripeException e) {
            return PaymentResult.failure(e.getMessage(), "stripe");
        }
    }

    @Override
    public boolean supports(String paymentMethod) {
        return List.of("credit_card", "debit_card", "apple_pay").contains(paymentMethod);
    }

    @Override
    public String getProviderName() {
        return "Stripe";
    }
}

public class PayPayProcessor implements PaymentProcessor {
    private final PayPayClient client;

    public PayPayProcessor(String merchantId, String apiSecret) {
        this.client = new PayPayClient(merchantId, apiSecret);
    }

    @Override
    public PaymentResult process(PaymentRequest request) {
        // PayPay API を使った決済処理
        try {
            var result = client.createPayment(
                request.getAmount(),
                request.getOrderId()
            );
            return PaymentResult.success(result.getPaymentId(), "paypay");
        } catch (PayPayException e) {
            return PaymentResult.failure(e.getMessage(), "paypay");
        }
    }

    @Override
    public boolean supports(String paymentMethod) {
        return "paypay".equals(paymentMethod);
    }

    @Override
    public String getProviderName() {
        return "PayPay";
    }
}

public class BankTransferProcessor implements PaymentProcessor {
    @Override
    public PaymentResult process(PaymentRequest request) {
        // 銀行振込処理
        String transferId = generateTransferId();
        return PaymentResult.pending(transferId, "bank_transfer");
    }

    @Override
    public boolean supports(String paymentMethod) {
        return "bank_transfer".equals(paymentMethod);
    }

    @Override
    public String getProviderName() {
        return "Bank Transfer";
    }

    private String generateTransferId() {
        return "BT-" + System.currentTimeMillis();
    }
}

// 利用側: PaymentProcessor のみに依存（具象クラスを知らない）
public class CheckoutService {
    private final List<PaymentProcessor> processors;

    public CheckoutService(List<PaymentProcessor> processors) {
        this.processors = processors;
    }

    public PaymentResult checkout(Order order, String paymentMethod) {
        // ポリモーフィズム: 適切なプロセッサを動的に選択
        PaymentProcessor processor = processors.stream()
            .filter(p -> p.supports(paymentMethod))
            .findFirst()
            .orElseThrow(() -> new UnsupportedPaymentException(
                "サポートされていない決済方法: " + paymentMethod));

        PaymentRequest request = PaymentRequest.from(order);
        System.out.println("決済プロバイダ: " + processor.getProviderName());
        return processor.process(request);
    }
}

// 組み立て（DI）
CheckoutService service = new CheckoutService(List.of(
    new StripeProcessor("sk_test_xxx"),
    new PayPayProcessor("merchant_123", "secret_xxx"),
    new BankTransferProcessor()
));

// 新しい決済方法を追加 → 新しいクラスを追加するだけ
// CheckoutService は一切変更不要（OCP遵守）
```

---

## 3. 動的ディスパッチの仕組み（vtable）

```
仮想関数テーブル（vtable / Virtual Method Table）:
  → C++, Java, C# 等で使われる実装メカニズム
  → 各クラスが持つメソッドポインタの配列
  → 実行時にオブジェクトの実際の型に基づいてメソッドを選択

  メモリレイアウト:

  Circle オブジェクト            Circle の vtable
  ┌──────────────┐            ┌──────────────────┐
  │ vptr ────────┼───────────→│ area() → Circle実装│
  │ radius: 5.0  │            │ perimeter() → ...  │
  └──────────────┘            │ describe() → ...   │
                              └──────────────────┘

  Rectangle オブジェクト         Rectangle の vtable
  ┌──────────────┐            ┌──────────────────┐
  │ vptr ────────┼───────────→│ area() → Rect実装  │
  │ width: 3.0   │            │ perimeter() → ...  │
  │ height: 4.0  │            │ describe() → ...   │
  └──────────────┘            └──────────────────┘

  shape.area() の呼び出し:
    1. shape の vptr を取得
    2. vtable から area() のアドレスを取得
    3. そのアドレスのメソッドを呼ぶ

  コスト: ポインタ間接参照1回分（ほぼゼロコスト）
```

### 3.1 vtable の詳細な動作

```
継承時の vtable の構築:

  Shape vtable:
  ┌──────────────────────┐
  │ [0] area() → ???     │ ← 純粋仮想（抽象メソッド）
  │ [1] perimeter() → ???│
  │ [2] describe() → ??? │
  └──────────────────────┘

  Circle vtable（Shape を継承）:
  ┌───────────────────────────────┐
  │ [0] area() → Circle::area    │ ← オーバーライド
  │ [1] perimeter() → Circle::per│ ← オーバーライド
  │ [2] describe() → Circle::desc│ ← オーバーライド
  └───────────────────────────────┘

  FilledCircle vtable（Circle を継承）:
  ┌───────────────────────────────────┐
  │ [0] area() → Circle::area        │ ← 継承（変更なし）
  │ [1] perimeter() → Circle::per    │ ← 継承（変更なし）
  │ [2] describe() → Filled::describe│ ← オーバーライド
  │ [3] fill() → FilledCircle::fill  │ ← 新規追加
  └───────────────────────────────────┘

  ポイント:
  - vtable はクラスごとに1つ（オブジェクトごとではない）
  - オブジェクトには vptr（vtable へのポインタ）のみ保持
  - メモリオーバーヘッド = 1ポインタ/オブジェクト（通常8バイト）
  - 呼び出しオーバーヘッド = 1間接参照（数ナノ秒）
```

```cpp
// C++: vtable の動作を理解するための例
#include <iostream>
#include <vector>
#include <memory>

class Shape {
public:
    virtual ~Shape() = default;

    // 純粋仮想関数（= 0）: vtable のスロットは存在するが、
    // アドレスは nullptr（サブクラスで実装必須）
    virtual double area() const = 0;
    virtual double perimeter() const = 0;
    virtual std::string describe() const = 0;

    // 非仮想関数: vtable に含まれない。静的に解決される
    void printInfo() const {
        std::cout << describe() << ": area=" << area() << std::endl;
    }
};

class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}

    double area() const override {
        return 3.14159 * radius * radius;
    }

    double perimeter() const override {
        return 2 * 3.14159 * radius;
    }

    std::string describe() const override {
        return "Circle(r=" + std::to_string(radius) + ")";
    }
};

class Rectangle : public Shape {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}

    double area() const override {
        return width * height;
    }

    double perimeter() const override {
        return 2 * (width + height);
    }

    std::string describe() const override {
        return "Rect(" + std::to_string(width) + "x" + std::to_string(height) + ")";
    }
};

int main() {
    // ポリモーフィズム: Shape ポインタの配列
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(5.0));
    shapes.push_back(std::make_unique<Rectangle>(3.0, 4.0));
    shapes.push_back(std::make_unique<Circle>(10.0));

    // 各 shape の vptr を通じて正しい実装が呼ばれる
    for (const auto& shape : shapes) {
        shape->printInfo();
    }
    // Circle(r=5.000000): area=78.5398
    // Rect(3.000000x4.000000): area=12
    // Circle(r=10.000000): area=314.159

    return 0;
}
```

### 3.2 vtable のパフォーマンス考慮

```
vtable 呼び出しのコスト分析:

  直接呼び出し（非仮想）:
    call 0x400520          ; 1命令、アドレスはコンパイル時に確定
    → CPU の分岐予測が100%的中
    → インライン展開も可能

  仮想関数呼び出し（vtable経由）:
    mov rax, [rdi]         ; 1. vptr をロード
    call [rax + offset]    ; 2. vtable からメソッドアドレスを取得して呼ぶ
    → 間接参照1回 + 分岐予測ミスの可能性
    → インライン展開が困難

  パフォーマンスへの影響:
    - 通常のアプリケーション: 影響はほぼゼロ（無視できる）
    - ゲームの内部ループ: 数百万回/フレーム → 1-2%の影響あり得る
    - 数値計算の内部ループ: 影響が顕著になる場合がある

  最適化テクニック:
    1. ホットパスでは仮想関数を避ける
    2. final キーワードで仮想呼び出しを排除（C++/Java）
    3. コンパイラの devirtualization 最適化を活用
    4. 型ごとにバッチ処理（data-oriented design）
```

---

## 4. 実践的な活用パターン

### 4.1 Strategy パターン

```typescript
// 支払い方法のポリモーフィズム
interface PaymentStrategy {
  pay(amount: number): Promise<PaymentResult>;
  validate(): boolean;
  getDisplayName(): string;
  getFee(amount: number): number;
}

interface PaymentResult {
  success: boolean;
  transactionId: string;
  message?: string;
}

class CreditCardPayment implements PaymentStrategy {
  constructor(
    private cardNumber: string,
    private cvv: string,
    private expiry: string,
  ) {}

  async pay(amount: number): Promise<PaymentResult> {
    const fee = this.getFee(amount);
    const totalAmount = amount + fee;
    // クレジットカード決済処理
    return {
      success: true,
      transactionId: `cc-${Date.now()}`,
      message: `¥${totalAmount}を決済しました（手数料: ¥${fee}）`,
    };
  }

  validate(): boolean {
    return (
      this.cardNumber.replace(/\s/g, "").length === 16 &&
      this.cvv.length === 3 &&
      /^\d{2}\/\d{2}$/.test(this.expiry)
    );
  }

  getDisplayName(): string {
    const masked = this.cardNumber.slice(-4).padStart(16, "*");
    return `クレジットカード (****${masked.slice(-4)})`;
  }

  getFee(amount: number): number {
    return Math.round(amount * 0.036); // 3.6%
  }
}

class PayPayPayment implements PaymentStrategy {
  constructor(private userId: string) {}

  async pay(amount: number): Promise<PaymentResult> {
    return {
      success: true,
      transactionId: `pp-${Date.now()}`,
      message: `PayPayで¥${amount}を決済しました`,
    };
  }

  validate(): boolean {
    return this.userId.length > 0;
  }

  getDisplayName(): string {
    return "PayPay";
  }

  getFee(amount: number): number {
    return 0; // PayPayは手数料無料
  }
}

class BankTransferPayment implements PaymentStrategy {
  constructor(
    private bankCode: string,
    private accountNumber: string,
  ) {}

  async pay(amount: number): Promise<PaymentResult> {
    return {
      success: true,
      transactionId: `bt-${Date.now()}`,
      message: `銀行振込の依頼を受け付けました（¥${amount}）`,
    };
  }

  validate(): boolean {
    return this.bankCode.length === 4 && this.accountNumber.length >= 7;
  }

  getDisplayName(): string {
    return `銀行振込 (${this.bankCode})`;
  }

  getFee(amount: number): number {
    return amount >= 30000 ? 440 : 220; // 3万円以上は440円
  }
}

class ConvenienceStorePayment implements PaymentStrategy {
  constructor(private storeType: "seven" | "lawson" | "family") {}

  async pay(amount: number): Promise<PaymentResult> {
    const paymentCode = this.generatePaymentCode();
    return {
      success: true,
      transactionId: `cs-${Date.now()}`,
      message: `コンビニ支払い番号: ${paymentCode}（期限: 3日以内）`,
    };
  }

  validate(): boolean {
    return ["seven", "lawson", "family"].includes(this.storeType);
  }

  getDisplayName(): string {
    const names = { seven: "セブンイレブン", lawson: "ローソン", family: "ファミリーマート" };
    return `コンビニ払い（${names[this.storeType]}）`;
  }

  getFee(amount: number): number {
    return 110; // 一律110円
  }

  private generatePaymentCode(): string {
    return Math.random().toString(36).substring(2, 14).toUpperCase();
  }
}

// 利用側: PaymentStrategy のみに依存
class Checkout {
  async process(strategy: PaymentStrategy, amount: number): Promise<void> {
    console.log(`決済方法: ${strategy.getDisplayName()}`);

    if (!strategy.validate()) {
      throw new Error("決済情報が無効です");
    }

    const fee = strategy.getFee(amount);
    console.log(`手数料: ¥${fee}`);

    const result = await strategy.pay(amount);
    if (result.success) {
      console.log(`決済成功: ${result.message}`);
      console.log(`取引ID: ${result.transactionId}`);
    } else {
      console.log(`決済失敗: ${result.message}`);
    }
    // 新しい決済方法が追加されても、このコードは変更不要
  }
}

// 使用例
const checkout = new Checkout();
await checkout.process(new CreditCardPayment("4111111111111111", "123", "12/25"), 10000);
await checkout.process(new PayPayPayment("user-123"), 5000);
await checkout.process(new ConvenienceStorePayment("seven"), 3000);
```

### 4.2 プラグインシステム

```python
# Python: プラグインシステム
from abc import ABC, abstractmethod
from typing import Any
import json


class FileExporter(ABC):
    """ファイルエクスポーターの抽象基底クラス"""

    @abstractmethod
    def export(self, data: list[dict]) -> bytes:
        """データをバイト列にエクスポート"""
        ...

    @abstractmethod
    def file_extension(self) -> str:
        """ファイル拡張子を返す"""
        ...

    @abstractmethod
    def mime_type(self) -> str:
        """MIMEタイプを返す"""
        ...

    def get_filename(self, base_name: str) -> str:
        """ファイル名を生成"""
        return f"{base_name}{self.file_extension()}"


class CsvExporter(FileExporter):
    def __init__(self, delimiter: str = ",", encoding: str = "utf-8"):
        self.delimiter = delimiter
        self.encoding = encoding

    def export(self, data: list[dict]) -> bytes:
        if not data:
            return b""
        headers = self.delimiter.join(data[0].keys())
        rows = [
            self.delimiter.join(self._escape(str(v)) for v in row.values())
            for row in data
        ]
        content = headers + "\n" + "\n".join(rows)
        return content.encode(self.encoding)

    def file_extension(self) -> str:
        return ".csv"

    def mime_type(self) -> str:
        return "text/csv"

    def _escape(self, value: str) -> str:
        if self.delimiter in value or '"' in value or '\n' in value:
            return f'"{value.replace(chr(34), chr(34)+chr(34))}"'
        return value


class JsonExporter(FileExporter):
    def __init__(self, indent: int = 2, ensure_ascii: bool = False):
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def export(self, data: list[dict]) -> bytes:
        return json.dumps(
            data,
            ensure_ascii=self.ensure_ascii,
            indent=self.indent,
        ).encode("utf-8")

    def file_extension(self) -> str:
        return ".json"

    def mime_type(self) -> str:
        return "application/json"


class ExcelExporter(FileExporter):
    """Excel形式でのエクスポート"""

    def export(self, data: list[dict]) -> bytes:
        # 簡易的なXML SpreadsheetML形式
        xml_parts = ['<?xml version="1.0"?>\n']
        xml_parts.append('<Workbook>\n<Worksheet ss:Name="Sheet1">\n<Table>\n')

        if data:
            # ヘッダー行
            xml_parts.append('<Row>\n')
            for key in data[0].keys():
                xml_parts.append(f'  <Cell><Data ss:Type="String">{key}</Data></Cell>\n')
            xml_parts.append('</Row>\n')

            # データ行
            for row in data:
                xml_parts.append('<Row>\n')
                for value in row.values():
                    data_type = "Number" if isinstance(value, (int, float)) else "String"
                    xml_parts.append(f'  <Cell><Data ss:Type="{data_type}">{value}</Data></Cell>\n')
                xml_parts.append('</Row>\n')

        xml_parts.append('</Table>\n</Worksheet>\n</Workbook>')
        return "".join(xml_parts).encode("utf-8")

    def file_extension(self) -> str:
        return ".xml"

    def mime_type(self) -> str:
        return "application/vnd.ms-excel"


class MarkdownExporter(FileExporter):
    """Markdown テーブル形式でのエクスポート"""

    def export(self, data: list[dict]) -> bytes:
        if not data:
            return b""

        headers = list(data[0].keys())
        lines = []

        # ヘッダー行
        lines.append("| " + " | ".join(headers) + " |")
        # 区切り線
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        # データ行
        for row in data:
            values = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines).encode("utf-8")

    def file_extension(self) -> str:
        return ".md"

    def mime_type(self) -> str:
        return "text/markdown"


# レジストリパターン: エクスポーターを動的に管理
class ExporterRegistry:
    """エクスポーターのレジストリ"""

    def __init__(self):
        self._exporters: dict[str, FileExporter] = {}

    def register(self, format_name: str, exporter: FileExporter) -> None:
        self._exporters[format_name] = exporter

    def get(self, format_name: str) -> FileExporter:
        if format_name not in self._exporters:
            available = ", ".join(self._exporters.keys())
            raise ValueError(
                f"未対応のフォーマット: {format_name}（利用可能: {available}）"
            )
        return self._exporters[format_name]

    def available_formats(self) -> list[str]:
        return list(self._exporters.keys())


# レジストリの初期化
registry = ExporterRegistry()
registry.register("csv", CsvExporter())
registry.register("json", JsonExporter())
registry.register("excel", ExcelExporter())
registry.register("markdown", MarkdownExporter())
registry.register("tsv", CsvExporter(delimiter="\t"))  # TSVもCSVの派生

# 利用側: FileExporter のみに依存
def save_report(format_name: str, data: list[dict], filename: str) -> str:
    exporter = registry.get(format_name)
    content = exporter.export(data)
    full_path = exporter.get_filename(filename)

    with open(full_path, "wb") as f:
        f.write(content)

    print(f"保存完了: {full_path} ({len(content)} bytes, {exporter.mime_type()})")
    return full_path


# 使用例
sample_data = [
    {"名前": "田中太郎", "年齢": 30, "部署": "開発部"},
    {"名前": "鈴木花子", "年齢": 25, "部署": "企画部"},
    {"名前": "佐藤次郎", "年齢": 35, "部署": "営業部"},
]

save_report("csv", sample_data, "report")       # report.csv
save_report("json", sample_data, "report")      # report.json
save_report("markdown", sample_data, "report")  # report.md
```

### 4.3 Observer パターンでのポリモーフィズム

```typescript
// TypeScript: Observer パターン
interface EventListener<T> {
  onEvent(event: T): void;
  getId(): string;
}

interface OrderEvent {
  type: "created" | "paid" | "shipped" | "delivered" | "cancelled";
  orderId: string;
  timestamp: Date;
  data?: Record<string, any>;
}

class EmailNotifier implements EventListener<OrderEvent> {
  constructor(private recipientEmail: string) {}

  onEvent(event: OrderEvent): void {
    const subjects: Record<string, string> = {
      created: "ご注文を受け付けました",
      paid: "お支払いを確認しました",
      shipped: "商品を発送しました",
      delivered: "商品をお届けしました",
      cancelled: "ご注文がキャンセルされました",
    };
    console.log(
      `📧 ${this.recipientEmail} へメール送信: [${subjects[event.type]}] 注文#${event.orderId}`
    );
  }

  getId(): string {
    return `email:${this.recipientEmail}`;
  }
}

class SlackNotifier implements EventListener<OrderEvent> {
  constructor(private channel: string, private webhookUrl: string) {}

  onEvent(event: OrderEvent): void {
    console.log(
      `💬 Slack #${this.channel}: 注文 ${event.orderId} が ${event.type} になりました`
    );
    // webhookUrl にPOSTリクエストを送信
  }

  getId(): string {
    return `slack:${this.channel}`;
  }
}

class InventoryUpdater implements EventListener<OrderEvent> {
  onEvent(event: OrderEvent): void {
    if (event.type === "paid") {
      console.log(`📦 在庫を確保: 注文 ${event.orderId}`);
    } else if (event.type === "cancelled") {
      console.log(`📦 在庫を戻す: 注文 ${event.orderId}`);
    }
  }

  getId(): string {
    return "inventory-updater";
  }
}

class AnalyticsTracker implements EventListener<OrderEvent> {
  private eventCounts: Map<string, number> = new Map();

  onEvent(event: OrderEvent): void {
    const count = this.eventCounts.get(event.type) || 0;
    this.eventCounts.set(event.type, count + 1);
    console.log(
      `📊 Analytics: ${event.type} イベント記録（累計: ${count + 1}）`
    );
  }

  getId(): string {
    return "analytics-tracker";
  }

  getStats(): Map<string, number> {
    return new Map(this.eventCounts);
  }
}

// イベントバス: リスナーのポリモーフィズムを活用
class EventBus<T> {
  private listeners: EventListener<T>[] = [];

  subscribe(listener: EventListener<T>): void {
    this.listeners.push(listener);
    console.log(`✅ リスナー登録: ${listener.getId()}`);
  }

  unsubscribe(listenerId: string): void {
    this.listeners = this.listeners.filter(l => l.getId() !== listenerId);
  }

  publish(event: T): void {
    // ポリモーフィズム: 全リスナーの onEvent を呼ぶ
    // 各リスナーが異なる処理を実行
    for (const listener of this.listeners) {
      try {
        listener.onEvent(event);
      } catch (error) {
        console.error(`リスナー ${listener.getId()} でエラー:`, error);
      }
    }
  }
}

// 使用例
const orderEvents = new EventBus<OrderEvent>();
orderEvents.subscribe(new EmailNotifier("customer@example.com"));
orderEvents.subscribe(new SlackNotifier("orders", "https://hooks.slack.com/xxx"));
orderEvents.subscribe(new InventoryUpdater());
orderEvents.subscribe(new AnalyticsTracker());

// 注文イベントを発行 → 全リスナーが各自の処理を実行
orderEvents.publish({
  type: "paid",
  orderId: "ORD-2024-001",
  timestamp: new Date(),
});
// 📧 customer@example.com へメール送信: [お支払いを確認しました] 注文#ORD-2024-001
// 💬 Slack #orders: 注文 ORD-2024-001 が paid になりました
// 📦 在庫を確保: 注文 ORD-2024-001
// 📊 Analytics: paid イベント記録（累計: 1）
```

---

## 5. アドホックポリモーフィズム

### 5.1 メソッドオーバーロード

```java
// Java: メソッドオーバーロード
public class Calculator {
    // 同名メソッドで引数の型が異なる
    public int add(int a, int b) { return a + b; }
    public double add(double a, double b) { return a + b; }
    public String add(String a, String b) { return a + b; }

    // 引数の数が異なるオーバーロード
    public int add(int a, int b, int c) { return a + b + c; }

    // 型の組み合わせ
    public double add(int a, double b) { return a + b; }
    public double add(double a, int b) { return a + b; }
}

// コンパイル時に呼ぶメソッドが決定（静的ディスパッチ）

// 実用例: ログメソッドのオーバーロード
public class Logger {
    public void log(String message) {
        log("INFO", message, null);
    }

    public void log(String level, String message) {
        log(level, message, null);
    }

    public void log(String level, String message, Throwable error) {
        System.out.printf("[%s] %s%n", level, message);
        if (error != null) {
            error.printStackTrace();
        }
    }

    public void log(String message, Object... args) {
        log("INFO", String.format(message, args), null);
    }
}
```

### 5.2 演算子オーバーロード

```python
# Python: 演算子オーバーロード
from __future__ import annotations
import math


class Vector:
    """2Dベクトルクラス（演算子オーバーロードの活用）"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    # 加算: v1 + v2
    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y)

    # 減算: v1 - v2
    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y)

    # スカラー乗算: v * 3
    def __mul__(self, scalar: float) -> Vector:
        return Vector(self.x * scalar, self.y * scalar)

    # 右側からのスカラー乗算: 3 * v
    def __rmul__(self, scalar: float) -> Vector:
        return self.__mul__(scalar)

    # スカラー除算: v / 2
    def __truediv__(self, scalar: float) -> Vector:
        if scalar == 0:
            raise ZeroDivisionError("ベクトルを0で割ることはできません")
        return Vector(self.x / scalar, self.y / scalar)

    # 負のベクトル: -v
    def __neg__(self) -> Vector:
        return Vector(-self.x, -self.y)

    # 等価比較: v1 == v2
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    # 絶対値（大きさ）: abs(v)
    def __abs__(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    # 真偽値: bool(v) → ゼロベクトルでなければ True
    def __bool__(self) -> bool:
        return not (self.x == 0 and self.y == 0)

    # 内積: v1 @ v2（行列乗算演算子を転用）
    def __matmul__(self, other: Vector) -> float:
        return self.x * other.x + self.y * other.y

    # 文字列表現
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    # ユーティリティメソッド
    def magnitude(self) -> float:
        return abs(self)

    def normalized(self) -> Vector:
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("ゼロベクトルは正規化できません")
        return self / mag

    def angle_to(self, other: Vector) -> float:
        """他のベクトルとの角度（ラジアン）"""
        dot = self @ other
        return math.acos(dot / (abs(self) * abs(other)))

    def rotate(self, angle: float) -> Vector:
        """ベクトルを回転（ラジアン）"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )


# 使用例
v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(v1 + v2)          # (4, 6)
print(v1 - v2)          # (-2, -2)
print(v1 * 3)           # (3, 6)
print(3 * v1)           # (3, 6) ← __rmul__
print(v1 / 2)           # (0.5, 1.0)
print(-v1)              # (-1, -2)
print(abs(v1))           # 2.236...
print(v1 @ v2)          # 11（内積）
print(v1 == Vector(1, 2))  # True
print(v1.normalized())  # (0.447..., 0.894...)

# 物理シミュレーション的な使い方
position = Vector(0, 0)
velocity = Vector(1, 0.5)
acceleration = Vector(0, -0.1)  # 重力

for step in range(10):
    velocity = velocity + acceleration
    position = position + velocity
    print(f"Step {step}: pos={position}, vel={velocity}")
```

```kotlin
// Kotlin: 演算子オーバーロード
data class Money(val amount: Long, val currency: String) {
    // + 演算子
    operator fun plus(other: Money): Money {
        require(currency == other.currency) { "通貨が異なります: $currency vs ${other.currency}" }
        return Money(amount + other.amount, currency)
    }

    // - 演算子
    operator fun minus(other: Money): Money {
        require(currency == other.currency) { "通貨が異なります" }
        return Money(amount - other.amount, currency)
    }

    // * 演算子（スカラー倍）
    operator fun times(multiplier: Int): Money {
        return Money(amount * multiplier, currency)
    }

    // 比較演算子
    operator fun compareTo(other: Money): Int {
        require(currency == other.currency) { "通貨が異なります" }
        return amount.compareTo(other.amount)
    }

    // 単項マイナス
    operator fun unaryMinus(): Money = Money(-amount, currency)

    override fun toString(): String {
        val formatted = String.format("%,d", amount)
        val symbol = when (currency) {
            "JPY" -> "¥"
            "USD" -> "$"
            "EUR" -> "€"
            else -> currency
        }
        return "$symbol$formatted"
    }
}

// 使用例
val price = Money(1000, "JPY")
val tax = Money(100, "JPY")
val total = price + tax           // ¥1,100
val double = price * 2            // ¥2,000
val refund = -total               // ¥-1,100

println(total)                    // ¥1,100
println(price > tax)              // true
println(price + Money(500, "JPY")) // ¥1,500
```

---

## 6. パラメトリックポリモーフィズム（ジェネリクス）

```typescript
// TypeScript: ジェネリクスによるパラメトリックポリモーフィズム

// 汎用的なリポジトリインターフェース
interface Repository<T, ID> {
  findById(id: ID): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: ID): Promise<void>;
  count(): Promise<number>;
}

// 汎用的な検索条件
interface SearchCriteria<T> {
  field: keyof T;
  operator: "eq" | "gt" | "lt" | "contains" | "in";
  value: any;
}

// 汎用的なページネーション結果
interface PaginatedResult<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
  hasPrev: boolean;
}

// エンティティ型の定義
interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

interface Product {
  id: string;
  name: string;
  price: number;
  stock: number;
}

// インメモリ実装（テスト用）
class InMemoryRepository<T extends { id: string }> implements Repository<T, string> {
  private items: Map<string, T> = new Map();

  async findById(id: string): Promise<T | null> {
    return this.items.get(id) || null;
  }

  async findAll(): Promise<T[]> {
    return Array.from(this.items.values());
  }

  async save(entity: T): Promise<T> {
    this.items.set(entity.id, { ...entity });
    return entity;
  }

  async delete(id: string): Promise<void> {
    this.items.delete(id);
  }

  async count(): Promise<number> {
    return this.items.size;
  }

  // 追加: 検索機能
  async search(criteria: SearchCriteria<T>[]): Promise<T[]> {
    const all = await this.findAll();
    return all.filter(item =>
      criteria.every(c => this.matchCriteria(item, c))
    );
  }

  private matchCriteria(item: T, criteria: SearchCriteria<T>): boolean {
    const value = item[criteria.field];
    switch (criteria.operator) {
      case "eq": return value === criteria.value;
      case "gt": return value > criteria.value;
      case "lt": return value < criteria.value;
      case "contains": return String(value).includes(criteria.value);
      case "in": return Array.isArray(criteria.value) && criteria.value.includes(value);
      default: return false;
    }
  }
}

// 使用例: 同じ Repository 実装で異なる型を扱う
const userRepo = new InMemoryRepository<User>();
const productRepo = new InMemoryRepository<Product>();

// 型安全: User と Product を間違えるとコンパイルエラー
await userRepo.save({ id: "1", name: "田中", email: "tanaka@test.com", createdAt: new Date() });
await productRepo.save({ id: "1", name: "ノートPC", price: 150000, stock: 10 });

// ジェネリクスによる汎用ユーティリティ関数
function paginate<T>(items: T[], page: number, pageSize: number): PaginatedResult<T> {
  const start = (page - 1) * pageSize;
  const paginatedItems = items.slice(start, start + pageSize);
  const total = items.length;

  return {
    items: paginatedItems,
    total,
    page,
    pageSize,
    hasNext: start + pageSize < total,
    hasPrev: page > 1,
  };
}

// 型安全にどの型でも使える
const userPage: PaginatedResult<User> = paginate(await userRepo.findAll(), 1, 10);
const productPage: PaginatedResult<Product> = paginate(await productRepo.findAll(), 1, 20);
```

```python
# Python: ジェネリクス（typing.Generic）
from typing import TypeVar, Generic, Optional, Callable
from dataclasses import dataclass


T = TypeVar("T")
E = TypeVar("E")


@dataclass
class Result(Generic[T, E]):
    """Rust の Result 型を模倣したジェネリック型"""
    _value: Optional[T] = None
    _error: Optional[E] = None
    _is_ok: bool = True

    @staticmethod
    def ok(value: T) -> "Result[T, E]":
        return Result(_value=value, _is_ok=True)

    @staticmethod
    def err(error: E) -> "Result[T, E]":
        return Result(_error=error, _is_ok=False)

    def is_ok(self) -> bool:
        return self._is_ok

    def is_err(self) -> bool:
        return not self._is_ok

    def unwrap(self) -> T:
        if not self._is_ok:
            raise ValueError(f"Result is Err: {self._error}")
        return self._value  # type: ignore

    def unwrap_or(self, default: T) -> T:
        return self._value if self._is_ok else default  # type: ignore

    def map(self, fn: Callable[[T], "U"]) -> "Result[U, E]":
        """成功値を変換"""
        if self._is_ok:
            return Result.ok(fn(self._value))  # type: ignore
        return Result.err(self._error)  # type: ignore

    def flat_map(self, fn: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        """成功値を別の Result に変換"""
        if self._is_ok:
            return fn(self._value)  # type: ignore
        return Result.err(self._error)  # type: ignore

    def __repr__(self) -> str:
        if self._is_ok:
            return f"Ok({self._value})"
        return f"Err({self._error})"


# 使用例
def parse_int(s: str) -> Result[int, str]:
    try:
        return Result.ok(int(s))
    except ValueError:
        return Result.err(f"'{s}' は整数に変換できません")

def divide(a: int, b: int) -> Result[float, str]:
    if b == 0:
        return Result.err("ゼロ除算エラー")
    return Result.ok(a / b)


# メソッドチェーン
result = (
    parse_int("42")
    .map(lambda x: x * 2)
    .flat_map(lambda x: divide(x, 7))
)
print(result)  # Ok(12.0)

error_result = (
    parse_int("abc")
    .map(lambda x: x * 2)
)
print(error_result)  # Err('abc' は整数に変換できません)
```

---

## 7. 静的 vs 動的ディスパッチ

```
静的ディスパッチ（コンパイル時に決定）:
  → メソッドオーバーロード
  → Rust のジェネリクス（単相化）
  → C++ のテンプレート
  → 高速（インライン化可能）

動的ディスパッチ（実行時に決定）:
  → サブタイプポリモーフィズム
  → Java/C# の仮想メソッド
  → Rust の dyn Trait
  → 柔軟（実行時にオブジェクトの型で分岐）
  → vtable のオーバーヘッドあり（ほぼゼロだが）
```

```rust
// Rust: 静的 vs 動的ディスパッチの選択
trait Drawable {
    fn draw(&self);
    fn bounding_box(&self) -> (f64, f64, f64, f64);
}

struct Circle { x: f64, y: f64, radius: f64 }
struct Rect { x: f64, y: f64, width: f64, height: f64 }

impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle at ({}, {}) r={}", self.x, self.y, self.radius);
    }
    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        (self.x - self.radius, self.y - self.radius,
         self.x + self.radius, self.y + self.radius)
    }
}

impl Drawable for Rect {
    fn draw(&self) {
        println!("Drawing rect at ({}, {}) {}x{}", self.x, self.y, self.width, self.height);
    }
    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        (self.x, self.y, self.x + self.width, self.y + self.height)
    }
}

// 静的ディスパッチ（ジェネリクス）: コンパイル時に型が確定
// → コンパイラが型ごとに専用のコードを生成（単相化: monomorphization）
fn draw_static<T: Drawable>(item: &T) {
    item.draw(); // インライン化可能、高速
}
// コンパイル後:
// fn draw_static_Circle(item: &Circle) { item.draw(); }
// fn draw_static_Rect(item: &Rect) { item.draw(); }

// 動的ディスパッチ（トレイトオブジェクト）: 実行時に型が確定
fn draw_dynamic(item: &dyn Drawable) {
    item.draw(); // vtable 経由、柔軟
}

// 動的ディスパッチが必要な場面: 異なる型のコレクション
fn main() {
    // 静的ディスパッチ: 同じ型のコレクション
    let circles = vec![
        Circle { x: 0.0, y: 0.0, radius: 5.0 },
        Circle { x: 10.0, y: 10.0, radius: 3.0 },
    ];
    for c in &circles {
        draw_static(c); // Circle 専用のコードが呼ばれる
    }

    // 動的ディスパッチ: 異なる型の混在コレクション
    let shapes: Vec<Box<dyn Drawable>> = vec![
        Box::new(Circle { x: 0.0, y: 0.0, radius: 5.0 }),
        Box::new(Rect { x: 1.0, y: 1.0, width: 3.0, height: 4.0 }),
    ];
    for shape in &shapes {
        draw_dynamic(shape.as_ref()); // vtable 経由で呼ばれる
    }
}
```

### 7.1 選択基準

```
静的ディスパッチを選ぶべき場面:
  ✓ パフォーマンスが最重要（ゲームの内部ループ、数値計算）
  ✓ コンパイル時に型が確定している
  ✓ 同じ型のコレクションを扱う
  ✓ インライン展開のメリットが大きい

動的ディスパッチを選ぶべき場面:
  ✓ 異なる型を同じコレクションで扱いたい
  ✓ プラグインシステムのように実行時に型が決まる
  ✓ コンパイル時間を短縮したい（ジェネリクスの膨張を避ける）
  ✓ バイナリサイズを小さくしたい

  判断フロー:
  1. 異なる型を混在させる必要がある？
     → Yes → 動的ディスパッチ
  2. パフォーマンスが極めて重要？
     → Yes → 静的ディスパッチ
  3. どちらでもよい場合
     → 静的ディスパッチを優先（型安全性が高い）
```

---

## 8. ポリモーフィズムとデザインパターン

```
ポリモーフィズムを活用する主要デザインパターン:

  ┌──────────────────┬────────────────────────────────────────┐
  │ パターン          │ ポリモーフィズムの活用                  │
  ├──────────────────┼────────────────────────────────────────┤
  │ Strategy         │ アルゴリズムの切り替え                   │
  │ Observer         │ 通知先の動的な追加/削除                  │
  │ Command          │ コマンドの統一的な実行/Undo               │
  │ Template Method  │ 処理フローの共通化 + カスタマイズポイント  │
  │ Factory Method   │ 生成するオブジェクトの動的な切り替え      │
  │ State            │ 状態に応じた振る舞いの切り替え            │
  │ Visitor          │ 操作の追加（ダブルディスパッチ）          │
  │ Chain of Resp.   │ 処理の連鎖と委譲                        │
  │ Decorator        │ 機能の動的な追加                        │
  │ Adapter          │ 互換性のないIFの変換                     │
  └──────────────────┴────────────────────────────────────────┘
```

```typescript
// State パターン: 状態に応じた振る舞いの切り替え
interface OrderState {
  readonly name: string;
  pay(order: Order): void;
  ship(order: Order): void;
  deliver(order: Order): void;
  cancel(order: Order): void;
}

class PendingState implements OrderState {
  readonly name = "保留中";

  pay(order: Order): void {
    console.log("支払い処理を開始します");
    order.setState(new PaidState());
  }

  ship(order: Order): void {
    console.log("エラー: 支払い前に発送できません");
  }

  deliver(order: Order): void {
    console.log("エラー: 支払い前に配達できません");
  }

  cancel(order: Order): void {
    console.log("注文をキャンセルしました");
    order.setState(new CancelledState());
  }
}

class PaidState implements OrderState {
  readonly name = "支払い済み";

  pay(order: Order): void {
    console.log("エラー: 既に支払い済みです");
  }

  ship(order: Order): void {
    console.log("商品を発送しました");
    order.setState(new ShippedState());
  }

  deliver(order: Order): void {
    console.log("エラー: 発送前に配達できません");
  }

  cancel(order: Order): void {
    console.log("返金処理を開始し、注文をキャンセルしました");
    order.setState(new CancelledState());
  }
}

class ShippedState implements OrderState {
  readonly name = "発送済み";

  pay(order: Order): void {
    console.log("エラー: 既に支払い済みです");
  }

  ship(order: Order): void {
    console.log("エラー: 既に発送済みです");
  }

  deliver(order: Order): void {
    console.log("商品が配達されました");
    order.setState(new DeliveredState());
  }

  cancel(order: Order): void {
    console.log("エラー: 発送後のキャンセルは受付窓口にお問い合わせください");
  }
}

class DeliveredState implements OrderState {
  readonly name = "配達済み";

  pay(order: Order): void {
    console.log("エラー: 既に配達済みです");
  }

  ship(order: Order): void {
    console.log("エラー: 既に配達済みです");
  }

  deliver(order: Order): void {
    console.log("エラー: 既に配達済みです");
  }

  cancel(order: Order): void {
    console.log("返品処理を開始してください");
  }
}

class CancelledState implements OrderState {
  readonly name = "キャンセル済み";

  pay(order: Order): void {
    console.log("エラー: キャンセル済みの注文です");
  }

  ship(order: Order): void {
    console.log("エラー: キャンセル済みの注文です");
  }

  deliver(order: Order): void {
    console.log("エラー: キャンセル済みの注文です");
  }

  cancel(order: Order): void {
    console.log("エラー: 既にキャンセル済みです");
  }
}

class Order {
  private state: OrderState = new PendingState();

  constructor(public readonly id: string) {}

  setState(state: OrderState): void {
    console.log(`  状態遷移: ${this.state.name} → ${state.name}`);
    this.state = state;
  }

  getStateName(): string {
    return this.state.name;
  }

  // ポリモーフィズム: 現在の状態に応じた振る舞いが呼ばれる
  pay(): void { this.state.pay(this); }
  ship(): void { this.state.ship(this); }
  deliver(): void { this.state.deliver(this); }
  cancel(): void { this.state.cancel(this); }
}

// 使用例
const order = new Order("ORD-001");
order.pay();      // 支払い処理 → PaidState
order.ship();     // 発送 → ShippedState
order.deliver();  // 配達 → DeliveredState
order.cancel();   // 返品処理を開始してください
```

---

## 9. ポリモーフィズムのアンチパターン

```
アンチパターン1: 型チェックの氾濫
  → instanceof / typeof / type() を多用して分岐
  → ポリモーフィズムで解決すべき

アンチパターン2: ダウンキャスト
  → 親型を子型にキャストして子固有のメソッドを呼ぶ
  → インターフェース設計の見直しが必要

アンチパターン3: 空実装
  → インターフェースのメソッドを空実装で満たす
  → ISP（インターフェース分離の原則）違反

アンチパターン4: 過剰なポリモーフィズム
  → 変更の見込みがない部分まで抽象化
  → YAGNI 違反
```

```typescript
// ❌ アンチパターン: 型チェックの氾濫
function calculateDiscount(customer: Customer): number {
  if (customer instanceof PremiumCustomer) {
    return 0.2; // 20%割引
  } else if (customer instanceof RegularCustomer) {
    return 0.05; // 5%割引
  } else if (customer instanceof NewCustomer) {
    return 0.1; // 10%割引（初回特典）
  }
  return 0;
}
// 新しい顧客タイプ追加のたびにここを修正 → OCP違反

// ✅ ポリモーフィズムで解決
interface Customer {
  getDiscount(): number;
  getName(): string;
}

class PremiumCustomer implements Customer {
  getDiscount(): number { return 0.2; }
  getName(): string { return "プレミアム会員"; }
}

class RegularCustomer implements Customer {
  getDiscount(): number { return 0.05; }
  getName(): string { return "一般会員"; }
}

class NewCustomer implements Customer {
  getDiscount(): number { return 0.1; }
  getName(): string { return "新規会員"; }
}

// 利用側: Customer.getDiscount() を呼ぶだけ
function calculateTotal(customer: Customer, price: number): number {
  const discount = customer.getDiscount();
  return price * (1 - discount);
}
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

## まとめ

| 種類 | 決定時期 | 手段 | 典型例 |
|------|---------|------|--------|
| サブタイプ | 実行時 | 継承/インターフェース | Strategy, Plugin, Observer |
| パラメトリック | コンパイル時 | ジェネリクス | List<T>, Repository<T, ID> |
| アドホック | コンパイル時 | オーバーロード | add(int), add(double), 演算子 |

| 設計原則 | ポイント |
|----------|---------|
| OCP | 新機能はクラス追加で対応（既存コード変更なし） |
| LSP | サブタイプは親型の代替として正しく動く |
| DIP | 具象ではなく抽象（インターフェース）に依存 |
| ISP | 不要なメソッドを含まない小さなインターフェース |

---

## 次に読むべきガイド

---

## 参考文献
1. Cardelli, L. "On Understanding Types, Data Abstraction, and Polymorphism." 1985.
2. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
3. Bloch, J. "Effective Java." 3rd edition, 2018.
4. Martin, R. "Clean Architecture." Prentice Hall, 2017.
5. Wadler, P. "Theorems for free!" 1989.
6. Liskov, B. and Wing, J. "A Behavioral Notion of Subtyping." 1994.
