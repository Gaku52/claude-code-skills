# Polymorphism

> Polymorphism is the mechanism that lets you "invoke different implementations through the same interface." It is the most powerful concept in OOP and the foundation of flexible, extensible designs.

## What You Will Learn in This Chapter

- [ ] Understand the three types of polymorphism
- [ ] Grasp the mechanism of dynamic dispatch (vtable)
- [ ] Learn practical usage patterns of polymorphism
- [ ] Understand when to use static vs. dynamic dispatch
- [ ] Leverage parametric polymorphism (generics)
- [ ] Understand the connections to real-world design patterns


## Prerequisites

Reading the following beforehand will deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content of [Inheritance](./01-inheritance.md)

---

## 1. The Three Types of Polymorphism

```
1. Subtype Polymorphism (Subtype / Inclusion)
   -> Assign a subclass object to a variable of the parent type
   -> In OOP, "polymorphism" usually refers to this
   -> The method of the actual type is called at runtime

2. Parametric Polymorphism
   -> Generics. Write generic code using type parameters
   -> List<T>, Map<K,V>, etc.

3. Ad-hoc Polymorphism
   -> Method overloading. Methods with the same name but different argument types
   -> Also includes operator overloading

Comparison:
  +----------------+--------------+----------------+---------------+
  |                | Subtype      | Parametric     | Ad-hoc        |
  +----------------+--------------+----------------+---------------+
  | When decided   | Runtime      | Compile time   | Compile time  |
  | Means          | Inherit / IF | Generics       | Overloading   |
  | Type unity     | Common parent| Type parameter | Same name diff|
  | Typical        | Shape.area() | List<T>        | add(int,int)  |
  +----------------+--------------+----------------+---------------+
```

---

## 2. Subtype Polymorphism

```
  Shape (interface / abstract class)
    area(): number
    draw(): void
       ^
  +----+----+-----------+
  v    v    v           v
Circle  Rect  Triangle  Polygon
 area() area()  area()   area()
 Each has its own implementation

  shapes: Shape[] = [Circle, Rect, Triangle, ...]
  for (shape of shapes) {
    shape.area()  <- the correct implementation is called at runtime
  }
```

### 2.1 Basic Subtype Polymorphism

```typescript
// TypeScript: subtype polymorphism
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
    return `Circle (radius: ${this.radius})`;
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
    return `Rectangle (${this.width} x ${this.height})`;
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
    // Heron's formula
    const s = (this.a + this.b + this.c) / 2;
    return Math.sqrt(s * (s - this.a) * (s - this.b) * (s - this.c));
  }

  perimeter(): number {
    return this.a + this.b + this.c;
  }

  describe(): string {
    return `Triangle (sides: ${this.a}, ${this.b}, ${this.c})`;
  }
}

// Polymorphism: treat uniformly as the Shape type
function printShapeInfo(shape: Shape): void {
  console.log(`${shape.describe()}: area=${shape.area().toFixed(2)}`);
}

// Compute the total area of shapes (depends only on the Shape type)
function totalArea(shapes: Shape[]): number {
  return shapes.reduce((sum, shape) => sum + shape.area(), 0);
}

// Sort by area (depends only on the Shape type)
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
// Circle (radius: 5): area=78.54
// Rectangle (3 x 4): area=12.00
// Circle (radius: 10): area=314.16
// Triangle (sides: 3, 4, 5): area=6.00

console.log(`Total area: ${totalArea(shapes).toFixed(2)}`);
// Total area: 410.70
```

### 2.2 Interface-Based Polymorphism

```python
# Python: protocol-based polymorphism (duck typing + type hints)
from typing import Protocol, runtime_checkable


@runtime_checkable
class Renderable(Protocol):
    """Protocol for renderable objects"""
    def render(self) -> str: ...
    def width(self) -> int: ...
    def height(self) -> int: ...


@runtime_checkable
class Clickable(Protocol):
    """Protocol for clickable objects"""
    def on_click(self, x: int, y: int) -> None: ...
    def is_point_inside(self, x: int, y: int) -> bool: ...


class TextLabel:
    """Text label (implements Renderable only)"""
    def __init__(self, text: str, x: int = 0, y: int = 0):
        self.text = text
        self.x = x
        self.y = y

    def render(self) -> str:
        return f'<label x="{self.x}" y="{self.y}">{self.text}</label>'

    def width(self) -> int:
        return len(self.text) * 8  # assume 8px per character

    def height(self) -> int:
        return 16


class Button:
    """Button (implements both Renderable and Clickable)"""
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
    """Image (implements both Renderable and Clickable)"""
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


# Leveraging polymorphism: type-agnostic generic functions
def render_all(elements: list[Renderable]) -> str:
    """Render all elements that implement Renderable"""
    return "\n".join(el.render() for el in elements)

def calculate_total_area(elements: list[Renderable]) -> int:
    """Calculate the total area of all elements"""
    return sum(el.width() * el.height() for el in elements)

def handle_click(clickables: list[Clickable], x: int, y: int) -> None:
    """Call the click handler of elements at the click position"""
    for element in clickables:
        if element.is_point_inside(x, y):
            element.on_click(x, y)


# Usage
elements: list[Renderable] = [
    TextLabel("Hello", x=10, y=10),
    Button("Submit", x=10, y=40),
    Image("logo.png", x=10, y=80),
]

print(render_all(elements))
print(f"Total area: {calculate_total_area(elements)} px^2")

# Protocol type check
print(isinstance(Button("test"), Renderable))  # True
print(isinstance(Button("test"), Clickable))   # True
print(isinstance(TextLabel("test"), Clickable)) # False
```

### 2.3 Subtype Polymorphism in Java

```java
// Java: polymorphism via interfaces

// Payment processing interface
public interface PaymentProcessor {
    PaymentResult process(PaymentRequest request);
    boolean supports(String paymentMethod);
    String getProviderName();
}

// Implementations for each provider
public class StripeProcessor implements PaymentProcessor {
    private final StripeClient client;

    public StripeProcessor(String apiKey) {
        this.client = new StripeClient(apiKey);
    }

    @Override
    public PaymentResult process(PaymentRequest request) {
        // Payment processing using the Stripe API
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
        // Payment processing using the PayPay API
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
        // Bank transfer processing
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

// Consumer: depends only on PaymentProcessor (unaware of concrete classes)
public class CheckoutService {
    private final List<PaymentProcessor> processors;

    public CheckoutService(List<PaymentProcessor> processors) {
        this.processors = processors;
    }

    public PaymentResult checkout(Order order, String paymentMethod) {
        // Polymorphism: dynamically select the appropriate processor
        PaymentProcessor processor = processors.stream()
            .filter(p -> p.supports(paymentMethod))
            .findFirst()
            .orElseThrow(() -> new UnsupportedPaymentException(
                "Unsupported payment method: " + paymentMethod));

        PaymentRequest request = PaymentRequest.from(order);
        System.out.println("Payment provider: " + processor.getProviderName());
        return processor.process(request);
    }
}

// Wiring (DI)
CheckoutService service = new CheckoutService(List.of(
    new StripeProcessor("sk_test_xxx"),
    new PayPayProcessor("merchant_123", "secret_xxx"),
    new BankTransferProcessor()
));

// Adding a new payment method -> just add a new class
// CheckoutService needs no changes (OCP compliance)
```

---

## 3. Dynamic Dispatch Mechanism (vtable)

```
Virtual function table (vtable / Virtual Method Table):
  -> The implementation mechanism used by C++, Java, C#, etc.
  -> An array of method pointers owned by each class
  -> At runtime, selects the method based on the object's actual type

  Memory layout:

  Circle object                Circle's vtable
  +--------------+            +----------------------+
  | vptr --------+----------->| area() -> Circle impl |
  | radius: 5.0  |            | perimeter() -> ...    |
  +--------------+            | describe() -> ...     |
                              +----------------------+

  Rectangle object             Rectangle's vtable
  +--------------+            +----------------------+
  | vptr --------+----------->| area() -> Rect impl   |
  | width: 3.0   |            | perimeter() -> ...    |
  | height: 4.0  |            | describe() -> ...     |
  +--------------+            +----------------------+

  Invoking shape.area():
    1. Get shape's vptr
    2. Get the address of area() from the vtable
    3. Call the method at that address

  Cost: one pointer indirection (nearly zero cost)
```

### 3.1 Detailed vtable Behavior

```
How the vtable is built with inheritance:

  Shape vtable:
  +----------------------+
  | [0] area() -> ???    | <- pure virtual (abstract method)
  | [1] perimeter() -> ??|
  | [2] describe() -> ???|
  +----------------------+

  Circle vtable (inherits Shape):
  +----------------------------------+
  | [0] area() -> Circle::area       | <- override
  | [1] perimeter() -> Circle::per   | <- override
  | [2] describe() -> Circle::desc   | <- override
  +----------------------------------+

  FilledCircle vtable (inherits Circle):
  +--------------------------------------+
  | [0] area() -> Circle::area           | <- inherited (unchanged)
  | [1] perimeter() -> Circle::per       | <- inherited (unchanged)
  | [2] describe() -> Filled::describe   | <- override
  | [3] fill() -> FilledCircle::fill     | <- newly added
  +--------------------------------------+

  Key points:
  - One vtable per class (not per object)
  - Each object only holds a vptr (pointer to the vtable)
  - Memory overhead = 1 pointer per object (typically 8 bytes)
  - Call overhead = 1 indirection (a few nanoseconds)
```

```cpp
// C++: example to understand vtable behavior
#include <iostream>
#include <vector>
#include <memory>

class Shape {
public:
    virtual ~Shape() = default;

    // Pure virtual functions (= 0): a slot exists in the vtable,
    // but the address is nullptr (subclass must implement)
    virtual double area() const = 0;
    virtual double perimeter() const = 0;
    virtual std::string describe() const = 0;

    // Non-virtual function: not in the vtable. Resolved statically
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
    // Polymorphism: an array of Shape pointers
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(5.0));
    shapes.push_back(std::make_unique<Rectangle>(3.0, 4.0));
    shapes.push_back(std::make_unique<Circle>(10.0));

    // The correct implementation is called through each shape's vptr
    for (const auto& shape : shapes) {
        shape->printInfo();
    }
    // Circle(r=5.000000): area=78.5398
    // Rect(3.000000x4.000000): area=12
    // Circle(r=10.000000): area=314.159

    return 0;
}
```

### 3.2 vtable Performance Considerations

```
Cost analysis of vtable calls:

  Direct call (non-virtual):
    call 0x400520          ; 1 instruction, address fixed at compile time
    -> CPU branch prediction hits 100%
    -> Inlining is also possible

  Virtual function call (via vtable):
    mov rax, [rdi]         ; 1. load the vptr
    call [rax + offset]    ; 2. fetch method address from vtable and call
    -> One indirection + potential branch misprediction
    -> Inlining is difficult

  Performance impact:
    - Typical applications: effectively zero (negligible)
    - Game inner loops: millions of times per frame -> 1-2% impact possible
    - Inner loops of numerical computation: can be significant

  Optimization techniques:
    1. Avoid virtual functions on hot paths
    2. Use the final keyword to eliminate virtual calls (C++/Java)
    3. Take advantage of the compiler's devirtualization optimizations
    4. Batch processing per type (data-oriented design)
```

---

## 4. Practical Usage Patterns

### 4.1 Strategy Pattern

```typescript
// Polymorphism over payment methods
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
    // Credit card payment processing
    return {
      success: true,
      transactionId: `cc-${Date.now()}`,
      message: `Charged ¥${totalAmount} (fee: ¥${fee})`,
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
    return `Credit Card (****${masked.slice(-4)})`;
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
      message: `Charged ¥${amount} via PayPay`,
    };
  }

  validate(): boolean {
    return this.userId.length > 0;
  }

  getDisplayName(): string {
    return "PayPay";
  }

  getFee(amount: number): number {
    return 0; // PayPay has no fees
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
      message: `Bank transfer request accepted (¥${amount})`,
    };
  }

  validate(): boolean {
    return this.bankCode.length === 4 && this.accountNumber.length >= 7;
  }

  getDisplayName(): string {
    return `Bank Transfer (${this.bankCode})`;
  }

  getFee(amount: number): number {
    return amount >= 30000 ? 440 : 220; // 440 yen for 30,000 or more
  }
}

class ConvenienceStorePayment implements PaymentStrategy {
  constructor(private storeType: "seven" | "lawson" | "family") {}

  async pay(amount: number): Promise<PaymentResult> {
    const paymentCode = this.generatePaymentCode();
    return {
      success: true,
      transactionId: `cs-${Date.now()}`,
      message: `Convenience store payment code: ${paymentCode} (valid: within 3 days)`,
    };
  }

  validate(): boolean {
    return ["seven", "lawson", "family"].includes(this.storeType);
  }

  getDisplayName(): string {
    const names = { seven: "7-Eleven", lawson: "Lawson", family: "FamilyMart" };
    return `Convenience Store Payment (${names[this.storeType]})`;
  }

  getFee(amount: number): number {
    return 110; // Flat 110 yen
  }

  private generatePaymentCode(): string {
    return Math.random().toString(36).substring(2, 14).toUpperCase();
  }
}

// Consumer: depends only on PaymentStrategy
class Checkout {
  async process(strategy: PaymentStrategy, amount: number): Promise<void> {
    console.log(`Payment method: ${strategy.getDisplayName()}`);

    if (!strategy.validate()) {
      throw new Error("Invalid payment information");
    }

    const fee = strategy.getFee(amount);
    console.log(`Fee: ¥${fee}`);

    const result = await strategy.pay(amount);
    if (result.success) {
      console.log(`Payment success: ${result.message}`);
      console.log(`Transaction ID: ${result.transactionId}`);
    } else {
      console.log(`Payment failure: ${result.message}`);
    }
    // Even when a new payment method is added, this code does not change
  }
}

// Usage
const checkout = new Checkout();
await checkout.process(new CreditCardPayment("4111111111111111", "123", "12/25"), 10000);
await checkout.process(new PayPayPayment("user-123"), 5000);
await checkout.process(new ConvenienceStorePayment("seven"), 3000);
```

### 4.2 Plugin System

```python
# Python: plugin system
from abc import ABC, abstractmethod
from typing import Any
import json


class FileExporter(ABC):
    """Abstract base class for file exporters"""

    @abstractmethod
    def export(self, data: list[dict]) -> bytes:
        """Export data to a byte stream"""
        ...

    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension"""
        ...

    @abstractmethod
    def mime_type(self) -> str:
        """Return the MIME type"""
        ...

    def get_filename(self, base_name: str) -> str:
        """Generate a file name"""
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
    """Export in Excel format"""

    def export(self, data: list[dict]) -> bytes:
        # A simplified XML SpreadsheetML format
        xml_parts = ['<?xml version="1.0"?>\n']
        xml_parts.append('<Workbook>\n<Worksheet ss:Name="Sheet1">\n<Table>\n')

        if data:
            # Header row
            xml_parts.append('<Row>\n')
            for key in data[0].keys():
                xml_parts.append(f'  <Cell><Data ss:Type="String">{key}</Data></Cell>\n')
            xml_parts.append('</Row>\n')

            # Data rows
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
    """Export as a Markdown table"""

    def export(self, data: list[dict]) -> bytes:
        if not data:
            return b""

        headers = list(data[0].keys())
        lines = []

        # Header row
        lines.append("| " + " | ".join(headers) + " |")
        # Separator line
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        # Data rows
        for row in data:
            values = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines).encode("utf-8")

    def file_extension(self) -> str:
        return ".md"

    def mime_type(self) -> str:
        return "text/markdown"


# Registry pattern: manage exporters dynamically
class ExporterRegistry:
    """Registry of exporters"""

    def __init__(self):
        self._exporters: dict[str, FileExporter] = {}

    def register(self, format_name: str, exporter: FileExporter) -> None:
        self._exporters[format_name] = exporter

    def get(self, format_name: str) -> FileExporter:
        if format_name not in self._exporters:
            available = ", ".join(self._exporters.keys())
            raise ValueError(
                f"Unsupported format: {format_name} (available: {available})"
            )
        return self._exporters[format_name]

    def available_formats(self) -> list[str]:
        return list(self._exporters.keys())


# Initialize the registry
registry = ExporterRegistry()
registry.register("csv", CsvExporter())
registry.register("json", JsonExporter())
registry.register("excel", ExcelExporter())
registry.register("markdown", MarkdownExporter())
registry.register("tsv", CsvExporter(delimiter="\t"))  # TSV is a CSV variant

# Consumer: depends only on FileExporter
def save_report(format_name: str, data: list[dict], filename: str) -> str:
    exporter = registry.get(format_name)
    content = exporter.export(data)
    full_path = exporter.get_filename(filename)

    with open(full_path, "wb") as f:
        f.write(content)

    print(f"Saved: {full_path} ({len(content)} bytes, {exporter.mime_type()})")
    return full_path


# Usage
sample_data = [
    {"name": "Taro Tanaka", "age": 30, "department": "Engineering"},
    {"name": "Hanako Suzuki", "age": 25, "department": "Planning"},
    {"name": "Jiro Sato", "age": 35, "department": "Sales"},
]

save_report("csv", sample_data, "report")       # report.csv
save_report("json", sample_data, "report")      # report.json
save_report("markdown", sample_data, "report")  # report.md
```

### 4.3 Polymorphism in the Observer Pattern

```typescript
// TypeScript: Observer pattern
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
      created: "Your order has been received",
      paid: "Payment has been confirmed",
      shipped: "Your item has been shipped",
      delivered: "Your item has been delivered",
      cancelled: "Your order has been cancelled",
    };
    console.log(
      `Email sent to ${this.recipientEmail}: [${subjects[event.type]}] Order #${event.orderId}`
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
      `Slack #${this.channel}: order ${event.orderId} became ${event.type}`
    );
    // POST request is sent to webhookUrl
  }

  getId(): string {
    return `slack:${this.channel}`;
  }
}

class InventoryUpdater implements EventListener<OrderEvent> {
  onEvent(event: OrderEvent): void {
    if (event.type === "paid") {
      console.log(`Reserve stock: order ${event.orderId}`);
    } else if (event.type === "cancelled") {
      console.log(`Return stock: order ${event.orderId}`);
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
      `Analytics: recorded ${event.type} event (total: ${count + 1})`
    );
  }

  getId(): string {
    return "analytics-tracker";
  }

  getStats(): Map<string, number> {
    return new Map(this.eventCounts);
  }
}

// Event bus: leverages polymorphism of listeners
class EventBus<T> {
  private listeners: EventListener<T>[] = [];

  subscribe(listener: EventListener<T>): void {
    this.listeners.push(listener);
    console.log(`Listener registered: ${listener.getId()}`);
  }

  unsubscribe(listenerId: string): void {
    this.listeners = this.listeners.filter(l => l.getId() !== listenerId);
  }

  publish(event: T): void {
    // Polymorphism: call onEvent on every listener
    // Each listener performs different processing
    for (const listener of this.listeners) {
      try {
        listener.onEvent(event);
      } catch (error) {
        console.error(`Error in listener ${listener.getId()}:`, error);
      }
    }
  }
}

// Usage
const orderEvents = new EventBus<OrderEvent>();
orderEvents.subscribe(new EmailNotifier("customer@example.com"));
orderEvents.subscribe(new SlackNotifier("orders", "https://hooks.slack.com/xxx"));
orderEvents.subscribe(new InventoryUpdater());
orderEvents.subscribe(new AnalyticsTracker());

// Publish an order event -> every listener performs its own processing
orderEvents.publish({
  type: "paid",
  orderId: "ORD-2024-001",
  timestamp: new Date(),
});
// Email sent to customer@example.com: [Payment has been confirmed] Order #ORD-2024-001
// Slack #orders: order ORD-2024-001 became paid
// Reserve stock: order ORD-2024-001
// Analytics: recorded paid event (total: 1)
```

---

## 5. Ad-hoc Polymorphism

### 5.1 Method Overloading

```java
// Java: method overloading
public class Calculator {
    // Same name but different argument types
    public int add(int a, int b) { return a + b; }
    public double add(double a, double b) { return a + b; }
    public String add(String a, String b) { return a + b; }

    // Overload with different number of arguments
    public int add(int a, int b, int c) { return a + b + c; }

    // Combinations of types
    public double add(int a, double b) { return a + b; }
    public double add(double a, int b) { return a + b; }
}

// The method to call is decided at compile time (static dispatch)

// Practical example: overloaded log methods
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

### 5.2 Operator Overloading

```python
# Python: operator overloading
from __future__ import annotations
import math


class Vector:
    """2D vector class (leveraging operator overloading)"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    # Addition: v1 + v2
    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y)

    # Subtraction: v1 - v2
    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y)

    # Scalar multiplication: v * 3
    def __mul__(self, scalar: float) -> Vector:
        return Vector(self.x * scalar, self.y * scalar)

    # Scalar multiplication from the right: 3 * v
    def __rmul__(self, scalar: float) -> Vector:
        return self.__mul__(scalar)

    # Scalar division: v / 2
    def __truediv__(self, scalar: float) -> Vector:
        if scalar == 0:
            raise ZeroDivisionError("cannot divide a vector by zero")
        return Vector(self.x / scalar, self.y / scalar)

    # Negation: -v
    def __neg__(self) -> Vector:
        return Vector(-self.x, -self.y)

    # Equality comparison: v1 == v2
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    # Absolute value (magnitude): abs(v)
    def __abs__(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    # Truthiness: bool(v) -> True if not a zero vector
    def __bool__(self) -> bool:
        return not (self.x == 0 and self.y == 0)

    # Dot product: v1 @ v2 (repurposing the matrix multiplication operator)
    def __matmul__(self, other: Vector) -> float:
        return self.x * other.x + self.y * other.y

    # String representation
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    # Utility methods
    def magnitude(self) -> float:
        return abs(self)

    def normalized(self) -> Vector:
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("cannot normalize a zero vector")
        return self / mag

    def angle_to(self, other: Vector) -> float:
        """Angle (in radians) to another vector"""
        dot = self @ other
        return math.acos(dot / (abs(self) * abs(other)))

    def rotate(self, angle: float) -> Vector:
        """Rotate the vector (in radians)"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )


# Usage
v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(v1 + v2)          # (4, 6)
print(v1 - v2)          # (-2, -2)
print(v1 * 3)           # (3, 6)
print(3 * v1)           # (3, 6) <- __rmul__
print(v1 / 2)           # (0.5, 1.0)
print(-v1)              # (-1, -2)
print(abs(v1))           # 2.236...
print(v1 @ v2)          # 11 (dot product)
print(v1 == Vector(1, 2))  # True
print(v1.normalized())  # (0.447..., 0.894...)

# Physics simulation style usage
position = Vector(0, 0)
velocity = Vector(1, 0.5)
acceleration = Vector(0, -0.1)  # gravity

for step in range(10):
    velocity = velocity + acceleration
    position = position + velocity
    print(f"Step {step}: pos={position}, vel={velocity}")
```

```kotlin
// Kotlin: operator overloading
data class Money(val amount: Long, val currency: String) {
    // + operator
    operator fun plus(other: Money): Money {
        require(currency == other.currency) { "different currencies: $currency vs ${other.currency}" }
        return Money(amount + other.amount, currency)
    }

    // - operator
    operator fun minus(other: Money): Money {
        require(currency == other.currency) { "different currencies" }
        return Money(amount - other.amount, currency)
    }

    // * operator (scalar multiplication)
    operator fun times(multiplier: Int): Money {
        return Money(amount * multiplier, currency)
    }

    // Comparison operator
    operator fun compareTo(other: Money): Int {
        require(currency == other.currency) { "different currencies" }
        return amount.compareTo(other.amount)
    }

    // Unary minus
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

// Usage
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

## 6. Parametric Polymorphism (Generics)

```typescript
// TypeScript: parametric polymorphism via generics

// Generic repository interface
interface Repository<T, ID> {
  findById(id: ID): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: ID): Promise<void>;
  count(): Promise<number>;
}

// Generic search criteria
interface SearchCriteria<T> {
  field: keyof T;
  operator: "eq" | "gt" | "lt" | "contains" | "in";
  value: any;
}

// Generic pagination result
interface PaginatedResult<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
  hasPrev: boolean;
}

// Entity type definitions
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

// In-memory implementation (for testing)
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

  // Additional: search capability
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

// Usage: the same Repository implementation handles different types
const userRepo = new InMemoryRepository<User>();
const productRepo = new InMemoryRepository<Product>();

// Type-safe: confusing User and Product is a compile error
await userRepo.save({ id: "1", name: "Tanaka", email: "tanaka@test.com", createdAt: new Date() });
await productRepo.save({ id: "1", name: "Laptop", price: 150000, stock: 10 });

// Generic utility function using generics
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

// Works in a type-safe manner for any type
const userPage: PaginatedResult<User> = paginate(await userRepo.findAll(), 1, 10);
const productPage: PaginatedResult<Product> = paginate(await productRepo.findAll(), 1, 20);
```

```python
# Python: generics (typing.Generic)
from typing import TypeVar, Generic, Optional, Callable
from dataclasses import dataclass


T = TypeVar("T")
E = TypeVar("E")


@dataclass
class Result(Generic[T, E]):
    """Generic type that mimics Rust's Result"""
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
        """Transform the success value"""
        if self._is_ok:
            return Result.ok(fn(self._value))  # type: ignore
        return Result.err(self._error)  # type: ignore

    def flat_map(self, fn: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        """Transform the success value into another Result"""
        if self._is_ok:
            return fn(self._value)  # type: ignore
        return Result.err(self._error)  # type: ignore

    def __repr__(self) -> str:
        if self._is_ok:
            return f"Ok({self._value})"
        return f"Err({self._error})"


# Usage
def parse_int(s: str) -> Result[int, str]:
    try:
        return Result.ok(int(s))
    except ValueError:
        return Result.err(f"'{s}' cannot be converted to an integer")

def divide(a: int, b: int) -> Result[float, str]:
    if b == 0:
        return Result.err("division by zero")
    return Result.ok(a / b)


# Method chaining
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
print(error_result)  # Err('abc' cannot be converted to an integer)
```

---

## 7. Static vs. Dynamic Dispatch

```
Static dispatch (decided at compile time):
  -> Method overloading
  -> Rust's generics (monomorphization)
  -> C++ templates
  -> Fast (can be inlined)

Dynamic dispatch (decided at runtime):
  -> Subtype polymorphism
  -> Virtual methods in Java/C#
  -> Rust's dyn Trait
  -> Flexible (branches on object type at runtime)
  -> Has vtable overhead (near zero, but present)
```

```rust
// Rust: choosing between static and dynamic dispatch
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

// Static dispatch (generics): type determined at compile time
// -> Compiler generates specialized code per type (monomorphization)
fn draw_static<T: Drawable>(item: &T) {
    item.draw(); // Can be inlined, fast
}
// After compilation:
// fn draw_static_Circle(item: &Circle) { item.draw(); }
// fn draw_static_Rect(item: &Rect) { item.draw(); }

// Dynamic dispatch (trait object): type determined at runtime
fn draw_dynamic(item: &dyn Drawable) {
    item.draw(); // Goes through a vtable, flexible
}

// Situation requiring dynamic dispatch: heterogeneous collections
fn main() {
    // Static dispatch: homogeneous collection
    let circles = vec![
        Circle { x: 0.0, y: 0.0, radius: 5.0 },
        Circle { x: 10.0, y: 10.0, radius: 3.0 },
    ];
    for c in &circles {
        draw_static(c); // Code specialized for Circle is called
    }

    // Dynamic dispatch: collection mixing different types
    let shapes: Vec<Box<dyn Drawable>> = vec![
        Box::new(Circle { x: 0.0, y: 0.0, radius: 5.0 }),
        Box::new(Rect { x: 1.0, y: 1.0, width: 3.0, height: 4.0 }),
    ];
    for shape in &shapes {
        draw_dynamic(shape.as_ref()); // Called through the vtable
    }
}
```

### 7.1 Selection Criteria

```
When to choose static dispatch:
  - Performance is paramount (game inner loops, numerical computation)
  - The type is known at compile time
  - You handle a homogeneous collection
  - Inlining benefits are significant

When to choose dynamic dispatch:
  - You need to handle different types in the same collection
  - The type is determined at runtime, e.g., plugin systems
  - You want to reduce compile time (avoid generic code bloat)
  - You want to reduce binary size

  Decision flow:
  1. Do you need to mix different types?
     -> Yes -> dynamic dispatch
  2. Is performance extremely important?
     -> Yes -> static dispatch
  3. Either is fine
     -> Prefer static dispatch (higher type safety)
```

---

## 8. Polymorphism and Design Patterns

```
Major design patterns that leverage polymorphism:

  +------------------+------------------------------------------+
  | Pattern          | How polymorphism is used                  |
  +------------------+------------------------------------------+
  | Strategy         | Swapping algorithms                       |
  | Observer         | Dynamic add/remove of notification targets |
  | Command          | Uniform execution / Undo of commands       |
  | Template Method  | Shared flow + customization points         |
  | Factory Method   | Dynamic selection of objects to create     |
  | State            | Behavior switching based on state          |
  | Visitor          | Adding operations (double dispatch)        |
  | Chain of Resp.   | Chained processing and delegation          |
  | Decorator        | Dynamic addition of features               |
  | Adapter          | Converting incompatible IFs                |
  +------------------+------------------------------------------+
```

```typescript
// State pattern: behavior switching based on state
interface OrderState {
  readonly name: string;
  pay(order: Order): void;
  ship(order: Order): void;
  deliver(order: Order): void;
  cancel(order: Order): void;
}

class PendingState implements OrderState {
  readonly name = "Pending";

  pay(order: Order): void {
    console.log("Starting payment processing");
    order.setState(new PaidState());
  }

  ship(order: Order): void {
    console.log("Error: cannot ship before payment");
  }

  deliver(order: Order): void {
    console.log("Error: cannot deliver before payment");
  }

  cancel(order: Order): void {
    console.log("Order has been cancelled");
    order.setState(new CancelledState());
  }
}

class PaidState implements OrderState {
  readonly name = "Paid";

  pay(order: Order): void {
    console.log("Error: payment already completed");
  }

  ship(order: Order): void {
    console.log("Item shipped");
    order.setState(new ShippedState());
  }

  deliver(order: Order): void {
    console.log("Error: cannot deliver before shipping");
  }

  cancel(order: Order): void {
    console.log("Started refund processing and cancelled the order");
    order.setState(new CancelledState());
  }
}

class ShippedState implements OrderState {
  readonly name = "Shipped";

  pay(order: Order): void {
    console.log("Error: payment already completed");
  }

  ship(order: Order): void {
    console.log("Error: already shipped");
  }

  deliver(order: Order): void {
    console.log("Item has been delivered");
    order.setState(new DeliveredState());
  }

  cancel(order: Order): void {
    console.log("Error: please contact support for cancellations after shipping");
  }
}

class DeliveredState implements OrderState {
  readonly name = "Delivered";

  pay(order: Order): void {
    console.log("Error: already delivered");
  }

  ship(order: Order): void {
    console.log("Error: already delivered");
  }

  deliver(order: Order): void {
    console.log("Error: already delivered");
  }

  cancel(order: Order): void {
    console.log("Please start the return process");
  }
}

class CancelledState implements OrderState {
  readonly name = "Cancelled";

  pay(order: Order): void {
    console.log("Error: order is already cancelled");
  }

  ship(order: Order): void {
    console.log("Error: order is already cancelled");
  }

  deliver(order: Order): void {
    console.log("Error: order is already cancelled");
  }

  cancel(order: Order): void {
    console.log("Error: already cancelled");
  }
}

class Order {
  private state: OrderState = new PendingState();

  constructor(public readonly id: string) {}

  setState(state: OrderState): void {
    console.log(`  State transition: ${this.state.name} -> ${state.name}`);
    this.state = state;
  }

  getStateName(): string {
    return this.state.name;
  }

  // Polymorphism: behavior corresponding to the current state is invoked
  pay(): void { this.state.pay(this); }
  ship(): void { this.state.ship(this); }
  deliver(): void { this.state.deliver(this); }
  cancel(): void { this.state.cancel(this); }
}

// Usage
const order = new Order("ORD-001");
order.pay();      // payment processing -> PaidState
order.ship();     // shipping -> ShippedState
order.deliver();  // delivery -> DeliveredState
order.cancel();   // Please start the return process
```

---

## 9. Polymorphism Anti-patterns

```
Anti-pattern 1: Type-check explosion
  -> Overuse of instanceof / typeof / type() branching
  -> Should be solved by polymorphism

Anti-pattern 2: Downcasting
  -> Casting a parent type to a child type to call child-specific methods
  -> Indicates the interface design needs revisiting

Anti-pattern 3: Empty implementations
  -> Implementing interface methods with empty bodies
  -> Violates ISP (Interface Segregation Principle)

Anti-pattern 4: Excessive polymorphism
  -> Abstracting parts that are unlikely to change
  -> Violates YAGNI
```

```typescript
// Anti-pattern: type-check explosion
function calculateDiscount(customer: Customer): number {
  if (customer instanceof PremiumCustomer) {
    return 0.2; // 20% discount
  } else if (customer instanceof RegularCustomer) {
    return 0.05; // 5% discount
  } else if (customer instanceof NewCustomer) {
    return 0.1; // 10% discount (first-time perk)
  }
  return 0;
}
// Every time a new customer type is added, this must change -> OCP violation

// Solved with polymorphism
interface Customer {
  getDiscount(): number;
  getName(): string;
}

class PremiumCustomer implements Customer {
  getDiscount(): number { return 0.2; }
  getName(): string { return "Premium Member"; }
}

class RegularCustomer implements Customer {
  getDiscount(): number { return 0.05; }
  getName(): string { return "Regular Member"; }
}

class NewCustomer implements Customer {
  getDiscount(): number { return 0.1; }
  getName(): string { return "New Member"; }
}

// Consumer: simply calls Customer.getDiscount()
function calculateTotal(customer: Customer, price: number): number {
  const discount = customer.getDiscount();
  return price * (1 - discount);
}
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Beyond theory, writing real code and observing the behavior deepens your understanding.

### Q2: What mistakes do beginners often make?

Skipping the fundamentals and jumping to advanced topics. We recommend firmly understanding the basic concepts explained in this guide before moving on to the next steps.

### Q3: How is this used in practice?

The knowledge of this topic is used frequently in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Kind | When decided | Means | Typical example |
|------|-------------|-------|-----------------|
| Subtype | Runtime | Inheritance / interface | Strategy, Plugin, Observer |
| Parametric | Compile time | Generics | List<T>, Repository<T, ID> |
| Ad-hoc | Compile time | Overloading | add(int), add(double), operators |

| Design principle | Key point |
|------------------|-----------|
| OCP | New features are handled by adding classes (no changes to existing code) |
| LSP | A subtype correctly substitutes for its parent type |
| DIP | Depend on abstractions (interfaces), not concretes |
| ISP | Small interfaces that do not include unnecessary methods |

---

## Guides to Read Next

---

## References
1. Cardelli, L. "On Understanding Types, Data Abstraction, and Polymorphism." 1985.
2. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
3. Bloch, J. "Effective Java." 3rd edition, 2018.
4. Martin, R. "Clean Architecture." Prentice Hall, 2017.
5. Wadler, P. "Theorems for free!" 1989.
6. Liskov, B. and Wing, J. "A Behavioral Notion of Subtyping." 1994.
