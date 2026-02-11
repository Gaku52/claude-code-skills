# オブジェクト指向プログラミング

> OOPの本質は「カプセル化」「継承」「ポリモーフィズム」の3本柱であり、大規模ソフトウェアの複雑さを管理する手法である。

## この章で学ぶこと

- [ ] OOPの4大原則を説明できる
- [ ] SOLID原則を理解する
- [ ] 継承よりコンポジションを選ぶべき理由を知る

---

## 1. OOPの4大原則

### 1.1 カプセル化、継承、ポリモーフィズム、抽象化

```python
# 1. カプセル化（Encapsulation）
# → データと操作をまとめ、内部を隠蔽
class BankAccount:
    def __init__(self, balance=0):
        self._balance = balance  # プライベート（慣習）

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount

    def get_balance(self):
        return self._balance

# 2. 継承（Inheritance）
# → 既存クラスを拡張して新しいクラスを作成
class SavingsAccount(BankAccount):
    def __init__(self, balance=0, interest_rate=0.02):
        super().__init__(balance)
        self.interest_rate = interest_rate

    def add_interest(self):
        self.deposit(self._balance * self.interest_rate)

# 3. ポリモーフィズム（Polymorphism）
# → 同じインターフェースで異なる振る舞い
class Shape:
    def area(self):
        raise NotImplementedError

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, w, h):
        self.w, self.h = w, h
    def area(self):
        return self.w * self.h

# 異なる型でも同じメソッドを呼べる
shapes = [Circle(5), Rectangle(3, 4)]
total = sum(s.area() for s in shapes)

# 4. 抽象化（Abstraction）
# → 複雑な実装を隠し、シンプルなインターフェースを提供
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def connect(self): pass

    @abstractmethod
    def query(self, sql): pass
```

---

## 2. SOLID原則

```
SOLID原則（Robert C. Martin）:

  S — Single Responsibility（単一責任）
    → クラスは1つの理由でのみ変更されるべき
    → ❌ UserクラスがDB操作+メール送信+バリデーション
    → ✅ User, UserRepository, EmailService, UserValidator

  O — Open/Closed（開放/閉鎖）
    → 拡張に開き、修正に閉じる
    → 新機能追加時に既存コードを変更しない
    → ポリモーフィズムやプラグインで実現

  L — Liskov Substitution（リスコフの置換）
    → サブクラスはスーパークラスと置換可能であるべき
    → ❌ Square extends Rectangle（正方形と長方形の問題）
    → ✅ サブクラスは親の契約を満たす

  I — Interface Segregation（インターフェース分離）
    → クライアントが使わないメソッドに依存すべきでない
    → 大きなインターフェースを小さく分割

  D — Dependency Inversion（依存性逆転）
    → 高レベルモジュールは低レベルモジュールに依存しない
    → 両方とも抽象に依存する
    → DI（依存性注入）で実現
```

---

## 3. 継承 vs コンポジション

```python
# ❌ 継承の乱用（深い継承ツリー）
class Animal: pass
class Mammal(Animal): pass
class DomesticMammal(Mammal): pass
class Dog(DomesticMammal): pass
class GuideDog(Dog): pass  # 5段階の継承... 複雑！

# ✅ コンポジション（has-a 関係）
class Dog:
    def __init__(self):
        self.walker = Walker()        # 歩く機能
        self.barker = Barker()        # 吠える機能
        self.guide_ability = None     # 盲導犬機能（オプション）

    def walk(self):
        self.walker.walk()

# 格言: 「継承よりコンポジションを好め」（GoF）
# 継承: is-a 関係（犬は動物）→ 慎重に使用
# コンポジション: has-a 関係（犬は歩行能力を持つ）→ 柔軟
```

---

## 4. 現代のOOP

```
OOP の現代的な変化:

  クラシックOOP（Java, C#）:
  - 重厚なクラス設計、デザインパターン
  - AbstractSingletonProxyFactoryBean 的な過剰設計

  現代のOOP:
  - データクラス（Python dataclass, Kotlin data class）
  - 不変オブジェクト優先
  - インターフェースベースの設計
  - 関数型との融合（ラムダ、ストリーム）

  OOPを使わない方が良い場面:
  - 小規模スクリプト → 手続き型で十分
  - データ変換パイプライン → 関数型が適切
  - 状態を持たない処理 → 純粋関数が適切
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 4大原則 | カプセル化、継承、ポリモーフィズム、抽象化 |
| SOLID | 5つの設計原則。保守性と拡張性を向上 |
| 継承 vs コンポ | 「継承よりコンポジション」。柔軟性重視 |
| 現代OOP | 関数型との融合、不変性重視、軽量クラス |

---

## 次に読むべきガイド
→ [[02-functional.md]] — 関数型プログラミング

---

## 参考文献
1. Martin, R. C. "Clean Architecture." Prentice Hall, 2017.
2. Gamma, E. et al. "Design Patterns (GoF)." 1994.
3. Bloch, J. "Effective Java." 3rd Edition, 2018.
