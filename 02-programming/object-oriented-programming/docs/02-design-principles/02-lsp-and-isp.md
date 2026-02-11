# LSP（リスコフの置換原則）+ ISP（インターフェース分離の原則）

> LSPは「サブタイプは親タイプの代替として使えるべき」、ISPは「クライアントに不要なメソッドへの依存を強制しない」。型の正しさとインターフェースの適切な粒度を保証する原則。

## この章で学ぶこと

- [ ] LSP 違反のパターンとその回避方法を理解する
- [ ] ISP による適切なインターフェース設計を把握する
- [ ] 実践的な設計判断の基準を学ぶ

---

## 1. LSP: リスコフの置換原則

```
定義（Barbara Liskov, 1987）:
  「S が T のサブタイプならば、T型のオブジェクトを
   S型のオブジェクトで置き換えてもプログラムの正しさは変わらない」

平易に:
  → 親クラスが使えるところにサブクラスを入れても壊れない
  → サブクラスは親クラスの「契約」を守る

契約:
  1. 事前条件を強化しない（受け入れ範囲を狭めない）
  2. 事後条件を弱化しない（保証を減らさない）
  3. 不変条件を維持する
```

### LSP 違反の典型例: 正方形と長方形

```typescript
// ❌ LSP違反の典型例
class Rectangle {
  constructor(protected width: number, protected height: number) {}

  setWidth(w: number): void { this.width = w; }
  setHeight(h: number): void { this.height = h; }
  area(): number { return this.width * this.height; }
}

class Square extends Rectangle {
  setWidth(w: number): void {
    this.width = w;
    this.height = w; // 正方形なので幅と高さを同じにする
  }
  setHeight(h: number): void {
    this.width = h;
    this.height = h;
  }
}

// このテストが壊れる = LSP違反
function testRectangle(rect: Rectangle): void {
  rect.setWidth(5);
  rect.setHeight(4);
  console.assert(rect.area() === 20); // Square だと 16 になる！
}

testRectangle(new Rectangle(0, 0)); // ✅ 20
testRectangle(new Square(0, 0));    // ❌ 16（LSP違反）
```

```typescript
// ✅ LSP準拠: 共通のインターフェースで抽象化
interface Shape {
  area(): number;
}

class Rectangle implements Shape {
  constructor(private width: number, private height: number) {}
  area(): number { return this.width * this.height; }
}

class Square implements Shape {
  constructor(private side: number) {}
  area(): number { return this.side * this.side; }
}
// Square は Rectangle を継承しない → LSP問題が発生しない
```

### LSP 違反のパターン

```
パターン1: メソッドの例外追加
  親: withdraw(amount) — 常に成功
  子: withdraw(amount) — 残高不足で例外 ← 事前条件の強化

パターン2: 空実装
  親: save() — データを保存
  子: save() — 何もしない ← 事後条件の弱化

パターン3: 型チェック
  if (animal instanceof Dog) {
    animal.fetch();
  }
  → ポリモーフィズムが壊れている = LSP違反の兆候
```

```python
# ❌ LSP違反: 空実装
class Bird:
    def fly(self) -> str:
        return "飛んでいます"

class Penguin(Bird):
    def fly(self) -> str:
        raise NotImplementedError("ペンギンは飛べません")  # LSP違反

# ✅ LSP準拠: インターフェースを分離
from abc import ABC, abstractmethod

class Bird(ABC):
    @abstractmethod
    def move(self) -> str: ...

class FlyingBird(Bird):
    def move(self) -> str:
        return "飛んでいます"

class Penguin(Bird):
    def move(self) -> str:
        return "泳いでいます"  # 正当な実装
```

---

## 2. ISP: インターフェース分離の原則

```
定義:
  「クライアントに、使わないメソッドへの依存を強制してはならない」

平易に:
  → インターフェースは小さく、焦点を絞る
  → 「太った」インターフェースを「細い」インターフェースに分割

なぜ重要か:
  → 不要なメソッドを実装する負担を減らす
  → 変更の影響を最小限にする
  → テスト時のモック作成が容易
```

### ISP リファクタリング

```typescript
// ❌ ISP違反: 巨大インターフェース
interface SmartDevice {
  print(doc: Document): void;
  scan(): Image;
  fax(doc: Document, number: string): void;
  copy(doc: Document): Document;
  staple(doc: Document): void;
}

// シンプルなプリンターは fax, scan, staple が不要!
class SimplePrinter implements SmartDevice {
  print(doc: Document): void { /* 実装 */ }
  scan(): Image { throw new Error("Not supported"); } // 空実装...
  fax(): void { throw new Error("Not supported"); }   // 空実装...
  copy(): Document { throw new Error("Not supported"); }
  staple(): void { throw new Error("Not supported"); }
}

// ✅ ISP適用: 細かいインターフェースに分離
interface Printer {
  print(doc: Document): void;
}

interface Scanner {
  scan(): Image;
}

interface Faxer {
  fax(doc: Document, number: string): void;
}

// 必要なインターフェースだけ実装
class SimplePrinter implements Printer {
  print(doc: Document): void { /* 実装 */ }
}

class MultiFunctionDevice implements Printer, Scanner, Faxer {
  print(doc: Document): void { /* 実装 */ }
  scan(): Image { /* 実装 */ return new Image(); }
  fax(doc: Document, number: string): void { /* 実装 */ }
}

// 利用側も必要なインターフェースだけに依存
function printReport(printer: Printer): void {
  // Printer だけに依存。Scanner, Faxer は知らない
  printer.print(report);
}
```

### 実践例: リポジトリのISP

```typescript
// ❌ ISP違反: CRUDが全て必要
interface Repository<T> {
  findAll(): Promise<T[]>;
  findById(id: string): Promise<T | null>;
  create(data: Partial<T>): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}

// 読み取り専用サービスにも書き込みメソッドが見える

// ✅ ISP適用: 読み取りと書き込みを分離
interface ReadRepository<T> {
  findAll(): Promise<T[]>;
  findById(id: string): Promise<T | null>;
}

interface WriteRepository<T> {
  create(data: Partial<T>): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}

interface Repository<T> extends ReadRepository<T>, WriteRepository<T> {}

// 読み取り専用サービス
class ReportService {
  constructor(private repo: ReadRepository<Order>) {}
  // 書き込みメソッドにアクセスできない = 安全
}
```

---

## 3. LSP と ISP の関係

```
LSP: サブタイプの正しさを保証
  → 「このクラスは本当に親の代替として使えるか？」
  → 使えない → インターフェースの設計が間違っている

ISP: インターフェースの粒度を最適化
  → 「このインターフェースは細かすぎ？太すぎ？」
  → 不要なメソッドがある → 分割する

LSP違反 → ISPで解決できることが多い:
  Penguin が Bird.fly() を実装できない
  → Bird インターフェースが太すぎる
  → Movable, Flyable に分割（ISP）
  → Penguin は Movable のみ実装（LSP準拠）
```

---

## 4. 判断基準

```
LSPチェックリスト:
  □ サブクラスは親の全メソッドを正しく実装しているか？
  □ 空実装や例外スロー（UnsupportedOperation）がないか？
  □ instanceof による型チェックが不要か？
  □ 親クラスのテストがサブクラスでも通るか？

ISPチェックリスト:
  □ インターフェースの実装者が全メソッドを使っているか？
  □ インターフェースの利用者が全メソッドを必要としているか？
  □ インターフェースのメソッド数は5個以下か？
  □ インターフェースの凝集度は高いか？
```

---

## まとめ

| 原則 | 核心 | 違反のサイン | 解決策 |
|------|------|------------|--------|
| LSP | 代替可能性 | 空実装、instanceof | インターフェース再設計 |
| ISP | 適切な粒度 | 不要メソッド | インターフェース分割 |

---

## 次に読むべきガイド
→ [[03-dip.md]] — DIP（依存性逆転の原則）

---

## 参考文献
1. Liskov, B. "Data Abstraction and Hierarchy." OOPSLA, 1987.
2. Martin, R. "The Interface Segregation Principle." 1996.
