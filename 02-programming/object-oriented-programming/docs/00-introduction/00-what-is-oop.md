# OOPとは何か

> オブジェクト指向プログラミング（OOP）は「データとそれを操作する手続きを一つの単位（オブジェクト）にまとめる」プログラミングパラダイム。現実世界のモデリングから大規模ソフトウェアの構造化まで、最も広く使われている設計手法。

## この章で学ぶこと

- [ ] OOPの本質的な考え方を理解する
- [ ] オブジェクトとメッセージパッシングの関係を把握する
- [ ] OOPが解決する問題と適用領域を理解する

---

## 1. OOPの本質

```
プログラミングパラダイムの比較:

  手続き型:    データ + 関数（別々に管理）
  OOP:        データ + 関数 = オブジェクト（一体化）
  関数型:     関数（データ変換のパイプライン）

OOPの核心:
  「世界をオブジェクトの集まりとして捉え、
   オブジェクト間のメッセージのやり取りで処理を進める」

Alan Kayの定義（Smalltalk の設計者）:
  1. Everything is an object（すべてはオブジェクト）
  2. Objects communicate by sending messages（メッセージで通信）
  3. Objects have their own memory（独自のメモリを持つ）
  4. Every object is an instance of a class（クラスのインスタンス）
  5. The class holds shared behavior（クラスが共通の振る舞いを保持）
```

---

## 2. メンタルモデル

```
手続き型のメンタルモデル:
  「手順書」— 上から順に実行する命令の列

  1. ユーザー情報を取得する
  2. バリデーションする
  3. データベースに保存する
  4. メールを送信する

OOPのメンタルモデル:
  「役割を持つ人々の組織」— 各人が責任を持って仕事する

  ┌─────────┐    ┌──────────┐    ┌──────────┐
  │  User    │───→│ Validator│───→│ Database │
  │ (データ) │    │ (検証)   │    │ (保存)   │
  └─────────┘    └──────────┘    └──────────┘
       │                              │
       │         ┌──────────┐         │
       └────────→│ Mailer   │←────────┘
                 │ (通知)   │
                 └──────────┘

  各オブジェクトは:
    - 自分のデータ（状態）を管理
    - 自分の責任範囲の処理を実行
    - 他のオブジェクトにメッセージ（メソッド呼び出し）を送る
```

---

## 3. オブジェクトの3要素

```
オブジェクト = 状態（State）+ 振る舞い（Behavior）+ アイデンティティ（Identity）

  ┌─────────────────────────────────┐
  │        BankAccount              │
  ├─────────────────────────────────┤
  │ 状態（State）:                   │
  │   - owner: "田中太郎"           │
  │   - balance: 100000             │
  │   - accountNumber: "1234567"    │
  ├─────────────────────────────────┤
  │ 振る舞い（Behavior）:            │
  │   - deposit(amount)             │
  │   - withdraw(amount)            │
  │   - getBalance()                │
  ├─────────────────────────────────┤
  │ アイデンティティ（Identity）:     │
  │   - メモリアドレス: 0x7ff...     │
  │   - 同じ状態でも別のオブジェクト  │
  └─────────────────────────────────┘
```

### コード例

```typescript
// TypeScript: 銀行口座オブジェクト
class BankAccount {
  // 状態（State）
  private owner: string;
  private balance: number;
  private readonly accountNumber: string;

  constructor(owner: string, accountNumber: string, initialBalance: number = 0) {
    this.owner = owner;
    this.accountNumber = accountNumber;
    this.balance = initialBalance;
  }

  // 振る舞い（Behavior）
  deposit(amount: number): void {
    if (amount <= 0) throw new Error("入金額は正の数である必要があります");
    this.balance += amount;
  }

  withdraw(amount: number): void {
    if (amount > this.balance) throw new Error("残高不足");
    this.balance -= amount;
  }

  getBalance(): number {
    return this.balance;
  }
}

// アイデンティティ: 同じ状態でも別のオブジェクト
const account1 = new BankAccount("田中", "001", 10000);
const account2 = new BankAccount("田中", "001", 10000);
console.log(account1 === account2); // false（別のオブジェクト）
```

```python
# Python: 同じ概念
class BankAccount:
    def __init__(self, owner: str, account_number: str, initial_balance: float = 0):
        self._owner = owner
        self._account_number = account_number
        self._balance = initial_balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("入金額は正の数である必要があります")
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self._balance:
            raise ValueError("残高不足")
        self._balance -= amount

    @property
    def balance(self) -> float:
        return self._balance
```

```java
// Java: 同じ概念
public class BankAccount {
    private String owner;
    private double balance;
    private final String accountNumber;

    public BankAccount(String owner, String accountNumber, double initialBalance) {
        this.owner = owner;
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }

    public void deposit(double amount) {
        if (amount <= 0) throw new IllegalArgumentException("入金額は正の数");
        this.balance += amount;
    }

    public void withdraw(double amount) {
        if (amount > this.balance) throw new IllegalStateException("残高不足");
        this.balance -= amount;
    }

    public double getBalance() { return balance; }
}
```

---

## 4. メッセージパッシング

```
OOPの本質はメッセージパッシング:

  手続き型的な考え方:
    result = validate(user_data)    ← 関数にデータを渡す

  OOP的な考え方:
    result = validator.validate(user_data)  ← オブジェクトにメッセージを送る

  違い:
    手続き型: 「誰が」処理するか不明
    OOP:     「validator」が責任を持って処理する

  メッセージパッシングの利点:
    1. 責任の所在が明確
    2. 実装を差し替え可能（ポリモーフィズム）
    3. テスト時にモックに差し替え可能
```

---

## 5. OOPが解決する問題

```
OOPなし（手続き型）:
  問題1: グローバル状態の管理困難
    → 誰がどの変数を変更したか追跡不能
    → OOP: カプセル化で状態をオブジェクト内に閉じ込める

  問題2: コードの重複
    → 似た処理を何度も書く
    → OOP: 継承・コンポジションで共通化

  問題3: 変更の影響範囲が不明
    → 1箇所の変更が全体に波及
    → OOP: インターフェースで依存を制御

  問題4: 大規模コードの構造化困難
    → 1万行超えると手続き型は破綻
    → OOP: クラス・パッケージで構造化
```

---

## 6. OOPの適用領域

```
OOPが得意な領域:
  ✓ GUIアプリケーション（ウィジェット階層）
  ✓ ゲーム開発（エンティティ・コンポーネント）
  ✓ エンタープライズアプリ（ビジネスロジック）
  ✓ フレームワーク設計（拡張ポイントの提供）
  ✓ シミュレーション（現実世界のモデリング）

OOPが不得意な領域:
  ✗ データ変換パイプライン → 関数型が適切
  ✗ 数値計算・科学計算 → 手続き型/配列指向が適切
  ✗ スクリプト・グルーコード → シンプルな手続き型が適切
  ✗ 並行処理 → アクターモデル/関数型が適切

現実のプロジェクト:
  → 複数のパラダイムを組み合わせるのが最適
  → OOP + FP のマルチパラダイムが主流
```

---

## 7. 各言語のOOPスタイル

```
┌──────────────┬───────────────────────────────────────┐
│ 言語         │ OOPスタイル                            │
├──────────────┼───────────────────────────────────────┤
│ Java         │ クラスベース・純粋OOP                   │
│ C++          │ クラスベース・マルチパラダイム           │
│ Python       │ クラスベース・ダックタイピング           │
│ TypeScript   │ クラス + 構造的型付け                   │
│ Ruby         │ 純粋OOP（全てがオブジェクト）           │
│ Kotlin       │ クラスベース・データクラス               │
│ Swift        │ プロトコル指向 + クラスベース            │
│ Rust         │ トレイトベース（クラスなし）             │
│ Go           │ 構造体 + インターフェース（クラスなし）  │
│ JavaScript   │ プロトタイプベース + クラス構文          │
└──────────────┴───────────────────────────────────────┘
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| OOPの本質 | データと振る舞いをオブジェクトに統合 |
| 3要素 | 状態 + 振る舞い + アイデンティティ |
| メッセージ | オブジェクト間のメソッド呼び出し |
| 得意分野 | GUI、ゲーム、エンタープライズ、フレームワーク |
| 現実 | OOP + FP のマルチパラダイムが主流 |

---

## 次に読むべきガイド
→ [[01-history-and-evolution.md]] — OOPの歴史と進化

---

## 参考文献
1. Kay, A. "The Early History of Smalltalk." ACM SIGPLAN Notices, 1993.
2. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
