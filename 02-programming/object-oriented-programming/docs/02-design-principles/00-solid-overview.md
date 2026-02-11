# SOLID原則概要

> SOLID は Robert C. Martin（Uncle Bob）が提唱した、OOP設計の5つの基本原則。保守性・拡張性・テスト容易性の高いソフトウェアを作るための指針。

## この章で学ぶこと

- [ ] SOLID 5原則の全体像を把握する
- [ ] なぜSOLIDが重要かを理解する
- [ ] 各原則の適用判断基準を学ぶ

---

## 1. SOLID の5原則

```
S — Single Responsibility Principle（単一責任の原則）
    「クラスを変更する理由は1つだけであるべき」

O — Open/Closed Principle（開放閉鎖の原則）
    「拡張に対して開き、修正に対して閉じる」

L — Liskov Substitution Principle（リスコフの置換原則）
    「サブクラスは親クラスの代替として使えるべき」

I — Interface Segregation Principle（インターフェース分離の原則）
    「クライアントに不要なメソッドへの依存を強制しない」

D — Dependency Inversion Principle（依存性逆転の原則）
    「具象ではなく抽象に依存せよ」
```

---

## 2. なぜSOLIDが重要か

```
SOLIDなし:
  ┌────────────────────────────────────┐
  │ UserService（全部入り）             │
  │ - ユーザー登録                     │
  │ - バリデーション                    │
  │ - DB保存                          │
  │ - メール送信                       │
  │ - ログ出力                         │
  │ - 権限チェック                     │
  └────────────────────────────────────┘
  問題:
  → 1000行超の巨大クラス
  → メール送信の変更がDB保存に影響する可能性
  → テストが困難（全ての依存を用意する必要）
  → チーム開発でコンフリクト多発

SOLIDあり:
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Validator│ │ UserRepo │ │ Mailer   │
  └──────────┘ └──────────┘ └──────────┘
  ┌──────────┐ ┌──────────┐
  │ Logger   │ │ AuthZ    │
  └──────────┘ └──────────┘
       ↑ すべてインターフェースで接続
  ┌──────────────────────────┐
  │ UserService              │
  │ (オーケストレーションのみ)│
  └──────────────────────────┘
  利点:
  → 各クラスが小さく理解しやすい
  → 変更の影響が局所的
  → テストが容易（モック差し替え）
  → チーム分担が明確
```

---

## 3. 5原則の関係

```
SOLIDの関係性:

  SRP（単一責任）: クラスを小さく保つ
    ↓ 小さいクラスが増える
  ISP（インターフェース分離）: 細かいインターフェースで接続
    ↓ インターフェースに依存
  DIP（依存性逆転）: 具象ではなく抽象に依存
    ↓ 抽象を通じて拡張
  OCP（開放閉鎖）: 既存コードを変更せずに拡張
    ↓ 拡張時に互換性を維持
  LSP（リスコフ置換）: サブタイプが正しく代替可能

  → 5原則は相互に補完しあう
  → 1つだけ適用しても効果は限定的
  → 5つ全てを意識して設計する
```

---

## 4. 各原則の概要と例

```typescript
// === S: 単一責任 ===
// ❌ 複数の責任
class User {
  save() { /* DB保存 */ }
  sendEmail() { /* メール送信 */ }
  generateReport() { /* レポート生成 */ }
}

// ✅ 単一の責任
class User { /* ユーザーデータのみ */ }
class UserRepository { save(user: User) { } }
class EmailService { send(to: string, body: string) { } }
class ReportGenerator { generate(user: User) { } }
```

```typescript
// === O: 開放閉鎖 ===
// ❌ 新しい形状を追加するたびに修正が必要
function calculateArea(shape: any): number {
  if (shape.type === "circle") return Math.PI * shape.radius ** 2;
  if (shape.type === "rectangle") return shape.width * shape.height;
  // 新しい形状を追加するたびにここを修正...
}

// ✅ 新しい形状はクラスを追加するだけ
interface Shape { area(): number; }
class Circle implements Shape { area() { return Math.PI * this.radius ** 2; } }
class Rectangle implements Shape { area() { return this.width * this.height; } }
// Triangle を追加しても既存コードは変更不要
```

```typescript
// === L: リスコフ置換 ===
// ❌ 正方形は長方形の代替として使えない
class Rectangle {
  setWidth(w: number) { this.width = w; }
  setHeight(h: number) { this.height = h; }
}
class Square extends Rectangle {
  setWidth(w: number) { this.width = w; this.height = w; } // 親と異なる振る舞い!
}

// ✅ 共通インターフェースで抽象化
interface Shape { area(): number; }
class Rectangle implements Shape { /* ... */ }
class Square implements Shape { /* ... */ }
```

```typescript
// === I: インターフェース分離 ===
// ❌ 巨大インターフェース
interface Worker {
  work(): void;
  eat(): void;
  sleep(): void;
}
// ロボットは eat() と sleep() が不要!

// ✅ 細かいインターフェースに分離
interface Workable { work(): void; }
interface Eatable { eat(): void; }
interface Sleepable { sleep(): void; }
```

```typescript
// === D: 依存性逆転 ===
// ❌ 具象に依存
class OrderService {
  private db = new MySQLDatabase(); // 具象クラスに直接依存
  save(order: Order) { this.db.insert(order); }
}

// ✅ 抽象に依存
interface Database { insert(data: any): void; }
class OrderService {
  constructor(private db: Database) {} // インターフェースに依存
  save(order: Order) { this.db.insert(order); }
}
```

---

## 5. SOLID適用の判断基準

```
過剰適用の警告:
  → 10行のスクリプトにSOLIDは不要
  → 個人プロジェクトの初期段階で過度な抽象化は有害
  → 「YAGNI（You Ain't Gonna Need It）」とのバランス

適用すべき場面:
  ✓ チーム開発のプロダクションコード
  ✓ 長期間メンテナンスされるシステム
  ✓ テストが重要なシステム
  ✓ 変更が頻繁に発生する領域

段階的な適用:
  1. まずシンプルに書く
  2. 変更が発生したら、その部分にSOLIDを適用
  3. 「3回目の変更」で抽象化を検討（Rule of Three）
```

---

## まとめ

| 原則 | 一言で | 効果 |
|------|--------|------|
| SRP | 1クラス1責任 | 変更の影響を局所化 |
| OCP | 拡張は開、修正は閉 | 既存コードの安定性 |
| LSP | 代替可能性 | ポリモーフィズムの正しさ |
| ISP | インターフェースを小さく | 不要な依存の排除 |
| DIP | 抽象に依存 | 疎結合・テスト容易性 |

---

## 次に読むべきガイド
→ [[01-srp-and-ocp.md]] — SRP + OCP 詳細

---

## 参考文献
1. Martin, R. "Agile Software Development." Prentice Hall, 2003.
2. Martin, R. "Clean Architecture." Prentice Hall, 2017.
