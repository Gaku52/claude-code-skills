# OOPの歴史と進化

> OOPは1960年代のSimulaに始まり、Smalltalk、C++、Javaを経て、現代のマルチパラダイム言語へと進化してきた。歴史を知ることで「なぜこう設計されているのか」が分かる。

## この章で学ぶこと

- [ ] OOPの誕生から現代までの進化を理解する
- [ ] 各時代の革新とその影響を把握する
- [ ] 各言語が解決しようとした問題とトレードオフを理解する
- [ ] 現代のOOPがどこに向かっているかを展望する

---

## 1. OOPの年表

```
1960s: 誕生
  1967  Simula       — OOPの祖。クラス・継承の概念を導入
                       ノルウェーの Dahl と Nygaard が開発
                       シミュレーション用途 → 一般化

1970s: 純粋OOPの確立
  1972  Smalltalk    — Alan Kay（Xerox PARC）
                       「すべてはオブジェクト」
                       メッセージパッシング、GC、IDE、GUI
                       → 現代のOOP概念の大部分を確立

1980s: 実用化
  1983  C++          — Bjarne Stroustrup
                       C + OOP。「ゼロコスト抽象化」
                       静的型付け + 多重継承
  1986  Objective-C  — C + Smalltalk のメッセージング
                       → Apple/NeXT で採用

1990s: 普及
  1995  Java         — Sun Microsystems
                       「Write once, run anywhere」
                       単一継承 + インターフェース
                       GC、JVM → エンタープライズの標準
  1995  JavaScript   — プロトタイプベースOOP
                       クラスなしでオブジェクト生成
  1993  Ruby         — まつもとゆきひろ
                       「全てがオブジェクト」、開発者の幸福

2000s: 反省と改良
  2000  C#           — Microsoft（Java への対抗）
                       プロパティ、デリゲート、LINQ
  2003  Scala        — OOP + FP の融合
                       JVM上で動作

2010s: モダンOOP
  2011  Kotlin       — より良いJava。null安全、データクラス
  2014  Swift        — プロトコル指向プログラミング
                       値型中心、参照カウント
  2012  TypeScript   — JavaScript + 型安全
                       構造的型付け

2020s: ポストOOP
  → マルチパラダイム（OOP + FP）が標準
  → 「純粋OOP」から「必要に応じてOOP」へ
  → コンポジション重視、継承縮小
```

---

## 2. 各時代の革新

### Simula（1967）: クラスと継承の発明

```
Simula の革新:
  1. クラス（class）: データと手続きを一体化した「設計図」
  2. オブジェクト: クラスから生成された「実体」
  3. 継承（inheritance）: 既存クラスの拡張
  4. 仮想手続き（virtual procedure）: ポリモーフィズムの原型

背景:
  → 離散事象シミュレーションのために開発
  → 「現実世界のモノをプログラムで表現する」必要性
  → 顧客、車、工場...を「オブジェクト」として表現
```

Simulaが生まれた背景には、1960年代のノルウェーにおけるシミュレーション研究がある。Ole-Johan Dahl と Kristen Nygaard は、ALGOL 60 をベースに、シミュレーションに必要な概念を言語レベルで表現しようとした。

```
Simula の設計思想:

  ALGOL 60 の問題:
    → データ構造と手続きが分離している
    → シミュレーション対象（顧客、車、工場）を
      自然に表現する手段がない
    → コルーチン的な並行処理が必要

  Simula 67 の解決策:
    → クラスでデータと手続きを統合
    → 継承で共通性と差異を表現
    → 仮想手続きで実行時の振る舞いを切り替え
    → コルーチン機能で擬似並行処理を実現

  例: 銀行のシミュレーション
    class Customer:
      到着時刻、サービス時間、待ち時間
      → 各顧客をオブジェクトとして表現

    class Teller:
      顧客キュー、処理中の顧客
      → 窓口係もオブジェクトとして表現

    class Bank:
      窓口係のリスト、シミュレーション時間
      → 全体を管理するオブジェクト
```

Simulaのコード例（疑似コード）:

```
! Simula 67 風の疑似コード
Class Vehicle;
  Virtual: Real Procedure fuelConsumption;
Begin
  Real speed, weight;

  Procedure accelerate(delta);
    Real delta;
  Begin
    speed := speed + delta;
  End;
End;

Vehicle Class Car;
Begin
  Integer passengers;

  Real Procedure fuelConsumption;
  Begin
    fuelConsumption := weight * speed * 0.01 + passengers * 0.5;
  End;
End;

Vehicle Class Truck;
Begin
  Real cargo_weight;

  Real Procedure fuelConsumption;
  Begin
    fuelConsumption := (weight + cargo_weight) * speed * 0.02;
  End;
End;
```

### Smalltalk（1972）: 純粋OOPの確立

```
Smalltalk の革新:
  1. すべてがオブジェクト（数値、真偽値、nil も）
  2. メッセージパッシング（メソッド呼び出しではない）
  3. ガベージコレクション
  4. 統合開発環境（IDE）の発明
  5. MVC パターンの発明
  6. リフレクション（メタプログラミング）

Alan Kay の思想:
  「OOPとはメッセージングのことだ。
   クラスや継承よりも、オブジェクト間のメッセージ交換が本質」

  → 現代の多くの言語は Kay の意図とは異なる方向に進化
  → Kay: 「C++ や Java は私が意図した OOP ではない」
```

Smalltalkが生まれたXerox PARCは、現代のコンピューティングの多くの概念を生み出した研究所である。

```
Xerox PARC の貢献（1970s）:
  → GUI（グラフィカルユーザーインターフェース）
  → WYSIWYG エディタ
  → イーサネット（LAN）
  → レーザープリンタ
  → Smalltalk（OOP + IDE + GUI）

  Alan Kay のビジョン:
    → 「Dynabook」構想
    → 子供でもプログラミングできるコンピュータ
    → オブジェクトが「小さなコンピュータ」のように
      独立して動作し、メッセージで通信する
    → 生物の細胞からインスピレーション

Smalltalk のメッセージパッシング:
  3 + 4
  → 3 に「+」メッセージと引数 4 を送る
  → 3（Integer オブジェクト）が自分で加算方法を決める

  "hello" size
  → "hello" に「size」メッセージを送る
  → String オブジェクトが文字数を返す

  collection do: [:each | each printNl]
  → collection に「do:」メッセージとブロック引数を送る
  → コレクションが自分で反復方法を決める

  重要な違い:
    C++/Java: コンパイラがメソッド呼び出しを解決
    Smalltalk: オブジェクトが動的にメッセージを処理
    → メッセージに対応するメソッドがない場合も処理可能
    → doesNotUnderstand: を使ったメタプログラミング
```

Smalltalkが発明・確立した概念は、現代のソフトウェア開発に深く浸透している。

```
Smalltalk が現代に残した遺産:

  1. MVC パターン（Model-View-Controller）
     → Web フレームワークの基本構造
     → Rails, Django, Spring MVC, ASP.NET MVC
     → React/Vue の設計思想にも影響

  2. IDE（統合開発環境）
     → コードエディタ + デバッガ + ブラウザ を統合
     → Eclipse, IntelliJ IDEA, VS Code の先祖

  3. リファクタリング
     → コードの構造を改善する体系的手法
     → Martin Fowler の著書は Smalltalk コミュニティから発展

  4. テスト駆動開発（TDD）
     → SUnit（Smalltalk の単体テストフレームワーク）
     → JUnit, pytest, Jest の原型

  5. デザインパターン
     → GoF パターンの多くは Smalltalk コミュニティで発見
     → Iterator, Observer, Strategy などは Smalltalk が起源

  6. アジャイル開発
     → XP（エクストリームプログラミング）は Smalltalk プロジェクトから
     → Kent Beck は Smalltalk コミュニティ出身
```

### C++（1983）: 実用化と静的型付け

```
C++ の革新:
  1. C との後方互換性（既存コードの活用）
  2. 静的型付けによるコンパイル時チェック
  3. 多重継承
  4. テンプレート（ジェネリクスの原型）
  5. 演算子オーバーロード
  6. RAII（Resource Acquisition Is Initialization）

影響:
  → 「OOP = クラス + 継承 + ポリモーフィズム」の定義を普及
  → ゼロコスト抽象化の思想
  → ただし複雑さも増大（C++ は最も複雑な言語の一つ）
```

C++の設計思想は「ゼロオーバーヘッド原則」に基づいている。

```
Bjarne Stroustrup の設計哲学:

  1. ゼロオーバーヘッド原則:
     「使わない機能のコストは払わない」
     「使う機能のコストは手書きのコードと同等」
     → 仮想関数テーブルのコストは、
       関数ポインタのテーブルを自分で書くのと同じ

  2. C との互換性:
     → 既存の C コードをそのまま使える
     → 段階的な OOP 導入が可能
     → システムプログラミングでの採用を促進

  3. 多パラダイム:
     → OOP だけでなく、手続き型、ジェネリック、関数型も
     → 「特定のスタイルを強制しない」

  C++ がOOPに与えた影響:
    良い面:
      → OOP を実用的なシステムプログラミングに持ち込んだ
      → 静的型付けと OOP の組み合わせを確立
      → テンプレートメタプログラミングの発見

    悪い面:
      → 多重継承のダイヤモンド問題
      → 過度に複雑な言語仕様
      → 「C with Classes」止まりの使い方が広まった
```

```cpp
// C++: OOP の実用的な例（RAII パターン）

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>

// RAII: リソースの獲得は初期化時に、解放はデストラクタで
class FileHandle {
private:
    std::fstream file;
    std::string filename;

public:
    // コンストラクタでファイルを開く（リソース獲得）
    explicit FileHandle(const std::string& fname)
        : filename(fname) {
        file.open(fname, std::ios::in | std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("ファイルを開けません: " + fname);
        }
        std::cout << "ファイルを開きました: " << fname << std::endl;
    }

    // デストラクタでファイルを閉じる（リソース解放）
    ~FileHandle() {
        if (file.is_open()) {
            file.close();
            std::cout << "ファイルを閉じました: " << filename << std::endl;
        }
    }

    // コピー禁止（リソースの二重解放を防ぐ）
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // ムーブは許可（所有権の移転）
    FileHandle(FileHandle&& other) noexcept
        : file(std::move(other.file)), filename(std::move(other.filename)) {}

    void write(const std::string& data) {
        file << data;
    }

    std::string readAll() {
        file.seekg(0);
        return std::string(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()
        );
    }
};

// スマートポインタ: RAII のメモリ管理版
class ResourceManager {
public:
    // unique_ptr: 排他的所有権
    std::unique_ptr<FileHandle> openFile(const std::string& filename) {
        return std::make_unique<FileHandle>(filename);
    }

    // shared_ptr: 共有所有権（参照カウント）
    std::shared_ptr<std::vector<int>> createSharedData() {
        return std::make_shared<std::vector<int>>();
    }
};

// C++ のテンプレート: コンパイル時のポリモーフィズム
template<typename Shape>
double calculateArea(const Shape& shape) {
    return shape.area(); // コンパイル時にメソッド解決
}

class Circle {
    double radius;
public:
    explicit Circle(double r) : radius(r) {}
    double area() const { return 3.14159 * radius * radius; }
};

class Rectangle {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const { return width * height; }
};

// テンプレートは仮想関数なしでポリモーフィズムを実現
// → ゼロオーバーヘッド（仮想関数テーブルの間接呼び出しなし）
```

### Java（1995）: エンタープライズ標準化

```
Java の革新:
  1. 単一継承 + インターフェース（多重継承の問題を回避）
  2. ガベージコレクション（C++ のメモリ管理から解放）
  3. JVM による移植性
  4. 豊富な標準ライブラリ
  5. パッケージによる名前空間管理

影響:
  → エンタープライズの標準言語に
  → デザインパターン（GoF）の普及
  → ただし「冗長すぎる」「ボイラープレート多すぎ」批判も
  → AbstractSingletonProxyFactoryBean 問題
```

Javaは「一度書けばどこでも動く」というビジョンを実現し、エンタープライズ開発の標準となった。

```java
// Java: エンタープライズパターンの進化

// === Java 1.0 時代（1995）: 基本的なOOP ===
public class Employee {
    private String name;
    private double salary;

    public Employee(String name, double salary) {
        this.name = name;
        this.salary = salary;
    }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public double getSalary() { return salary; }
    public void setSalary(double salary) { this.salary = salary; }

    @Override
    public String toString() {
        return "Employee{name='" + name + "', salary=" + salary + "}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Employee employee = (Employee) o;
        return Double.compare(employee.salary, salary) == 0
            && Objects.equals(name, employee.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, salary);
    }
}
// → ボイラープレートの山（getter/setter/equals/hashCode/toString）


// === Java 5 時代（2004）: ジェネリクス + アノテーション ===
public interface Repository<T, ID> {
    Optional<T> findById(ID id);
    List<T> findAll();
    T save(T entity);
    void deleteById(ID id);
}

public class EmployeeRepository implements Repository<Employee, Long> {
    private final Map<Long, Employee> store = new HashMap<>();

    @Override
    public Optional<Employee> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }

    @Override
    public List<Employee> findAll() {
        return new ArrayList<>(store.values());
    }

    @Override
    public Employee save(Employee employee) {
        store.put(employee.getId(), employee);
        return employee;
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }
}


// === Java 8 時代（2014）: ラムダ + Stream API ===
public class EmployeeService {
    private final Repository<Employee, Long> repository;

    public EmployeeService(Repository<Employee, Long> repository) {
        this.repository = repository;
    }

    // Stream API: 関数型プログラミングの要素
    public List<Employee> getHighPaidEmployees(double threshold) {
        return repository.findAll().stream()
            .filter(e -> e.getSalary() > threshold)
            .sorted(Comparator.comparingDouble(Employee::getSalary).reversed())
            .collect(Collectors.toList());
    }

    public Map<String, Double> getAverageSalaryByDepartment() {
        return repository.findAll().stream()
            .collect(Collectors.groupingBy(
                Employee::getDepartment,
                Collectors.averagingDouble(Employee::getSalary)
            ));
    }

    public Optional<Employee> findHighestPaid() {
        return repository.findAll().stream()
            .max(Comparator.comparingDouble(Employee::getSalary));
    }
}


// === Java 17+ 時代（2021-）: Record + Sealed + Pattern Matching ===

// Record: ボイラープレート削減（不変データクラス）
public record EmployeeRecord(
    long id,
    String name,
    String department,
    double salary
) {
    // コンパクトコンストラクタでバリデーション
    public EmployeeRecord {
        if (salary < 0) throw new IllegalArgumentException("給与は正の数");
        if (name == null || name.isBlank()) throw new IllegalArgumentException("名前は必須");
    }
}

// Sealed class: 代数的データ型
public sealed interface PaymentMethod
    permits CreditCard, BankTransfer, DigitalWallet {
}

public record CreditCard(String number, String expiry) implements PaymentMethod {}
public record BankTransfer(String accountNumber, String bankCode) implements PaymentMethod {}
public record DigitalWallet(String walletId, String provider) implements PaymentMethod {}

// Pattern Matching: switch式で型安全な分岐
public String processPayment(PaymentMethod method, double amount) {
    return switch (method) {
        case CreditCard cc -> "クレジットカード %s で %.0f円決済".formatted(
            cc.number().substring(cc.number().length() - 4), amount);
        case BankTransfer bt -> "銀行振込 %s へ %.0f円送金".formatted(
            bt.bankCode(), amount);
        case DigitalWallet dw -> "%s ウォレット %s で %.0f円決済".formatted(
            dw.provider(), dw.walletId(), amount);
    };
}
```

### Ruby（1993）: 開発者の幸福

Rubyはまつもとゆきひろ（Matz）によって設計され、「プログラマの幸福」を最大化することを目標とした。

```
Ruby の設計哲学:
  → 人間にとって自然な文法
  → 驚き最小の原則（Principle of Least Surprise）
  → 全てがオブジェクト（Smalltalk の影響）
  → メタプログラミング（言語を拡張する力）

Ruby が OOP に与えた影響:
  1. ブロック構文: クロージャの簡潔な記法
  2. open class: 既存クラスを後から拡張
  3. mixin: モジュールによる多重継承の代替
  4. DSL: ドメイン固有言語の構築が容易
  5. Rails: Web 開発における OOP の革命
```

```ruby
# Ruby: 純粋OOP + メタプログラミング

# 全てがオブジェクト
42.class          # => Integer
42.even?          # => true
"hello".reverse   # => "olleh"
nil.class         # => NilClass
true.class        # => TrueClass

# Mixin（多重継承の代替）
module Serializable
  def to_json
    require 'json'
    hash = {}
    instance_variables.each do |var|
      hash[var.to_s.delete('@')] = instance_variable_get(var)
    end
    JSON.generate(hash)
  end
end

module Auditable
  def self.included(base)
    base.instance_variable_set(:@audit_log, [])
  end

  def log_change(message)
    self.class.instance_variable_get(:@audit_log) << {
      timestamp: Time.now,
      object_id: object_id,
      message: message
    }
  end
end

class User
  include Serializable
  include Auditable

  attr_reader :name, :email

  def initialize(name, email)
    @name = name
    @email = email
    log_change("User created: #{name}")
  end

  def update_email(new_email)
    old = @email
    @email = new_email
    log_change("Email changed: #{old} -> #{new_email}")
  end
end

# Open class: 既存クラスの拡張
class String
  def palindrome?
    self == self.reverse
  end
end

"racecar".palindrome?  # => true
"hello".palindrome?    # => false

# メタプログラミング: 動的なメソッド定義
class ActiveRecordLike
  def self.has_attribute(name, type: :string, default: nil)
    # getter
    define_method(name) do
      instance_variable_get("@#{name}") || default
    end

    # setter
    define_method("#{name}=") do |value|
      instance_variable_set("@#{name}", value)
    end

    # クエリメソッド
    define_method("#{name}?") do
      !send(name).nil? && send(name) != "" && send(name) != false
    end
  end
end

class Product < ActiveRecordLike
  has_attribute :name, type: :string
  has_attribute :price, type: :float, default: 0
  has_attribute :in_stock, type: :boolean, default: true
end

p = Product.new
p.name = "Ruby本"
p.price = 3000
p.name?      # => true
p.in_stock?  # => true
```

### Python（1991）: 実用的OOP

Pythonはオランダの Guido van Rossum によって設計された。OOPをサポートするが、それを強制しない実用的な設計となっている。

```
Python の OOP の特徴:
  1. マルチパラダイム: OOP は選択肢の一つ
  2. ダックタイピング: 「アヒルのように歩き、鳴くなら、それはアヒル」
  3. 規約ベースのアクセス制御: _private は強制ではない
  4. 特殊メソッド: __init__, __str__, __eq__ 等でカスタマイズ
  5. デコレータ: メタプログラミングの簡潔な手段
  6. データクラス: Python 3.7+ でボイラープレート削減
```

```python
# Python: モダンなOOPの実践

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
from functools import total_ordering
from datetime import datetime


# データクラス: ボイラープレートの自動生成
@dataclass(frozen=True)  # frozen=True で不変に
@total_ordering
class Money:
    """金額値オブジェクト"""
    amount: int
    currency: str = "JPY"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("金額は0以上である必要があります")

    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"通貨が異なります: {self.currency} vs {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"通貨が異なります: {self.currency} vs {other.currency}")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: int | float) -> Money:
        return Money(int(self.amount * factor), self.currency)

    def __lt__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError("通貨が異なります")
        return self.amount < other.amount

    def __str__(self) -> str:
        if self.currency == "JPY":
            return f"¥{self.amount:,}"
        return f"{self.amount / 100:.2f} {self.currency}"


# プロトコル: 構造的型付け（ダックタイピングの型安全版）
@runtime_checkable
class Discountable(Protocol):
    """割引適用可能なもの"""
    def apply_discount(self, rate: float) -> Money: ...
    @property
    def price(self) -> Money: ...


# 抽象基底クラス
class Product(ABC):
    """商品の抽象基底クラス"""

    def __init__(self, name: str, base_price: Money):
        self._name = name
        self._base_price = base_price

    @property
    def name(self) -> str:
        return self._name

    @property
    def price(self) -> Money:
        return self._base_price

    @abstractmethod
    def description(self) -> str:
        """商品の説明を返す"""
        pass

    def apply_discount(self, rate: float) -> Money:
        """割引後の価格を計算"""
        if not 0 <= rate <= 1:
            raise ValueError("割引率は0〜1の範囲")
        return self._base_price * (1 - rate)


# 具象クラス
@dataclass
class Book(Product):
    """書籍"""
    _name: str = field(init=False)
    _base_price: Money = field(init=False)
    author: str = ""
    isbn: str = ""
    pages: int = 0

    def __init__(self, name: str, price: Money, author: str, isbn: str = "", pages: int = 0):
        super().__init__(name, price)
        self.author = author
        self.isbn = isbn
        self.pages = pages

    def description(self) -> str:
        return f"『{self.name}』{self.author}著 ({self.pages}ページ)"


@dataclass
class Electronics(Product):
    """電子機器"""
    _name: str = field(init=False)
    _base_price: Money = field(init=False)
    brand: str = ""
    warranty_months: int = 12

    def __init__(self, name: str, price: Money, brand: str, warranty_months: int = 12):
        super().__init__(name, price)
        self.brand = brand
        self.warranty_months = warranty_months

    def description(self) -> str:
        return f"{self.brand} {self.name} (保証: {self.warranty_months}ヶ月)"


# デコレータパターン
class DiscountedProduct:
    """割引適用済み商品（デコレータ）"""

    def __init__(self, product: Product, discount_rate: float, reason: str = ""):
        self._product = product
        self._discount_rate = discount_rate
        self._reason = reason

    @property
    def name(self) -> str:
        return f"{self._product.name} [{self._reason}]" if self._reason else self._product.name

    @property
    def price(self) -> Money:
        return self._product.apply_discount(self._discount_rate)

    @property
    def original_price(self) -> Money:
        return self._product.price

    def description(self) -> str:
        return f"{self._product.description()} - {self._discount_rate*100:.0f}%OFF"


# コンテキストマネージャ: Pythonらしいリソース管理
class DatabaseConnection:
    """データベース接続（コンテキストマネージャ）"""

    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._connected = False

    def __enter__(self):
        print(f"DB接続開始: {self._connection_string}")
        self._connected = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._connected:
            print("DB接続終了")
            self._connected = False
        return False  # 例外を再送出

    def query(self, sql: str) -> list[dict]:
        if not self._connected:
            raise RuntimeError("接続されていません")
        print(f"SQL実行: {sql}")
        return []


# 使用例
with DatabaseConnection("postgresql://localhost/mydb") as db:
    results = db.query("SELECT * FROM users")
# → __exit__ が自動的に呼ばれる（例外発生時も）
```

---

## 3. OOPの進化の方向性

```
第1世代（1967-1980）: クラスベース
  Simula, Smalltalk
  → 「世界をオブジェクトでモデリングする」

第2世代（1983-1995）: 静的型付け + 実用化
  C++, Java, C#
  → 「大規模開発を構造化する」

第3世代（2000-2015）: 軽量OOP + FP融合
  Ruby, Scala, Kotlin, Swift
  → 「ボイラープレートを減らし、関数型の良さを取り入れる」

第4世代（2015-現在）: ポストOOP
  Rust（トレイト）, Go（インターフェース）, TypeScript（構造的型付け）
  → 「継承を排除し、コンポジションとインターフェースで設計する」

進化の傾向:
  多重継承 → 単一継承 → 継承よりコンポジション → 継承なし
  ミュータブル → イミュータブル優先
  クラス中心 → インターフェース/トレイト中心
  暗黙的 → 明示的
```

### 3.1 継承の退潮

OOPの歴史において、最も大きな変化の一つは「継承からコンポジションへ」の流れである。

```
継承の退潮の歴史:

  1990s: 継承は OOP の中心
    → GoF: 「インターフェースに対してプログラムせよ」
    → しかし実際には深い継承階層が乱立

  2000s: 継承の問題が認識される
    → Joshua Bloch (Effective Java): 「継承よりコンポジション」
    → 脆弱な基底クラス問題
    → リスコフの置換原則の違反

  2010s: 継承を排除する言語の登場
    → Go: 継承なし、インターフェースの暗黙的実装
    → Rust: 継承なし、トレイトベース
    → Swift: プロトコル指向（値型 + プロトコル）

  2020s: 継承は「限定的に使う」コンセンサス
    → モダンな Java でも sealed class + record が推奨
    → Kotlin の data class は継承不可
    → 深い継承階層は明確なアンチパターン
```

```typescript
// TypeScript: 継承からコンポジションへの進化

// === 1990s スタイル: 深い継承階層 ===
// 問題だらけのアプローチ

/*
abstract class Animal {
  abstract speak(): string;
}

class Mammal extends Animal {
  breathe(): string { return "肺で呼吸"; }
  abstract speak(): string;
}

class Pet extends Mammal {
  constructor(public owner: string) { super(); }
  abstract speak(): string;
}

class Dog extends Pet {
  speak(): string { return "ワン！"; }
}

class Cat extends Pet {
  speak(): string { return "ニャー！"; }
}

// 問題: ペンギンは鳥だが飛べない → 継承階層が破綻
// 問題: 新しい振る舞いの追加が困難
// 問題: テストのための差し替えが困難
*/


// === 2020s スタイル: コンポジション + インターフェース ===

// 振る舞いをインターフェースで定義
interface CanSpeak {
  speak(): string;
}

interface CanMove {
  move(): string;
}

interface CanSwim {
  swim(): string;
}

interface CanFly {
  fly(): string;
}

// コンポジションで組み合わせ
class DogComposed implements CanSpeak, CanMove, CanSwim {
  constructor(
    public readonly name: string,
    public readonly owner: string,
  ) {}

  speak(): string { return `${this.name}: ワン！`; }
  move(): string { return `${this.name}が走る`; }
  swim(): string { return `${this.name}が泳ぐ`; }
}

class PenguinComposed implements CanSpeak, CanMove, CanSwim {
  constructor(public readonly name: string) {}

  speak(): string { return `${this.name}: ペンペン！`; }
  move(): string { return `${this.name}がよちよち歩く`; }
  swim(): string { return `${this.name}が高速で泳ぐ`; }
  // fly() は実装しない → コンパイル時に飛べないことが保証される
}

class EagleComposed implements CanSpeak, CanMove, CanFly {
  constructor(public readonly name: string) {}

  speak(): string { return `${this.name}: ピーッ！`; }
  move(): string { return `${this.name}が飛び回る`; }
  fly(): string { return `${this.name}が大空を飛ぶ`; }
  // swim() は実装しない → 泳げないことが型で表現される
}

// 必要なインターフェースだけを要求
function makeSwimRace(swimmers: CanSwim[]): void {
  for (const s of swimmers) {
    console.log(s.swim());
  }
}

// Dog と Penguin は泳げるが、Eagle は泳げない
// → コンパイル時にエラーで検出
makeSwimRace([
  new DogComposed("ポチ", "田中"),
  new PenguinComposed("ペンタ"),
  // new EagleComposed("タカ"),  // コンパイルエラー: CanSwim を実装していない
]);
```

### 3.2 型システムの進化

OOP言語の型システムも大きく進化してきた。

```
型システムの進化:

  名前的型付け（Nominal Typing）:
    → Java, C#, C++
    → 型名が一致しないと互換性なし
    → 明示的に implements / extends を宣言する必要がある

  構造的型付け（Structural Typing）:
    → TypeScript, Go
    → 型の構造（メソッドのシグネチャ）が一致すれば互換
    → implements を書かなくても、メソッドが一致すれば OK

  ダックタイピング（Duck Typing）:
    → Python, Ruby
    → 実行時に必要なメソッドがあれば OK
    → 「アヒルのように歩き、鳴くなら、アヒル」

  進化の流れ:
    名前的（厳格すぎる）
      → 構造的（柔軟 + 型安全）
        → ダックタイピング + 型ヒント（柔軟 + 文書化）
```

```typescript
// TypeScript: 構造的型付けの実践例

// インターフェースを明示的に実装しなくても型互換
interface Printable {
  toString(): string;
}

interface HasLength {
  length: number;
}

// クラスでも使える
class Document implements Printable {
  constructor(private content: string) {}
  toString(): string { return this.content; }
}

// プレーンオブジェクトでも OK（構造が一致すれば）
const logEntry = {
  toString(): string { return "2024-01-01 INFO: Application started"; }
};

// 配列は HasLength を満たす
const items = [1, 2, 3]; // { length: number } を持つ

function print(item: Printable): void {
  console.log(item.toString());
}

function getLength(item: HasLength): number {
  return item.length;
}

print(new Document("Hello"));   // OK
print(logEntry);                 // OK: 構造が一致
getLength(items);               // OK: length プロパティがある
getLength("hello");             // OK: string も length を持つ
```

---

## 4. 現代のOOP: マルチパラダイム

```kotlin
// Kotlin: モダンOOPの例
// データクラス（ボイラープレート削減）
data class User(
    val name: String,
    val email: String,
    val age: Int
)

// sealed class（代数的データ型 — FPからの影響）
sealed class Result<out T> {
    data class Success<T>(val value: T) : Result<T>()
    data class Failure(val error: Throwable) : Result<Nothing>()
}

// 拡張関数（オブジェクトを変更せずに機能追加）
fun String.isValidEmail(): Boolean =
    this.matches(Regex("^[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}$"))

// 高階関数（FPの要素）
fun <T> List<T>.filterAndMap(
    predicate: (T) -> Boolean,
    transform: (T) -> String
): List<String> = this.filter(predicate).map(transform)
```

```swift
// Swift: プロトコル指向プログラミング
protocol Drawable {
    func draw()
}

protocol Resizable {
    func resize(by factor: Double)
}

// プロトコル拡張（デフォルト実装）
extension Drawable {
    func draw() {
        print("Default drawing")
    }
}

// 値型（struct）+ プロトコル準拠
struct Circle: Drawable, Resizable {
    var radius: Double

    func draw() {
        print("Drawing circle with radius \(radius)")
    }

    func resize(by factor: Double) -> Circle {
        Circle(radius: radius * factor)
    }
}
```

### 4.1 Kotlin: より良いJava

Kotlinは JetBrains が「より良いJava」を目指して設計した言語であり、モダンOOPの多くの特徴を備えている。

```kotlin
// Kotlin: モダンOOPの包括的な例

// === Null安全 ===
fun processUser(name: String?) {
    // コンパイラが null チェックを強制
    val length = name?.length ?: 0
    val upper = name?.uppercase() ?: "UNKNOWN"

    // スマートキャスト
    if (name != null) {
        // この分岐内では name は String（非null）
        println(name.length)
    }
}

// === Sealed class + When 式（網羅的パターンマッチ） ===
sealed interface Shape {
    data class Circle(val radius: Double) : Shape
    data class Rectangle(val width: Double, val height: Double) : Shape
    data class Triangle(val base: Double, val height: Double) : Shape
}

fun area(shape: Shape): Double = when (shape) {
    is Shape.Circle -> Math.PI * shape.radius * shape.radius
    is Shape.Rectangle -> shape.width * shape.height
    is Shape.Triangle -> shape.base * shape.height / 2
    // when 式は網羅的: 新しい Shape を追加するとコンパイルエラー
}

// === 委譲パターン（by キーワード） ===
interface Logger {
    fun log(message: String)
}

class ConsoleLogger : Logger {
    override fun log(message: String) = println("[LOG] $message")
}

class UserService(logger: Logger) : Logger by logger {
    // Logger の実装を ConsoleLogger に委譲
    // log() メソッドを明示的に実装する必要なし

    fun createUser(name: String) {
        log("Creating user: $name")  // 委譲されたメソッド
        // ... ユーザー作成ロジック
    }
}

// === コルーチン（非同期プログラミング） ===
import kotlinx.coroutines.*

class OrderProcessor {
    suspend fun processOrder(orderId: String): Result<Order> {
        return try {
            val order = fetchOrder(orderId)      // 非同期DB取得
            val validated = validateOrder(order)  // バリデーション
            val charged = chargePayment(validated) // 非同期決済
            Result.success(charged)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private suspend fun fetchOrder(id: String): Order = withContext(Dispatchers.IO) {
        // DB からの取得（非同期）
        delay(100) // シミュレーション
        Order(id, "pending")
    }

    private fun validateOrder(order: Order): Order {
        // ビジネスルールのバリデーション
        require(order.status == "pending") { "注文は pending 状態である必要があります" }
        return order
    }

    private suspend fun chargePayment(order: Order): Order = withContext(Dispatchers.IO) {
        // 決済処理（非同期）
        delay(200) // シミュレーション
        order.copy(status = "confirmed")
    }
}
```

### 4.2 Rust: ポストOOPの最前線

Rustはクラスも継承も持たないが、トレイトと構造体で強力なOOP的パターンを実現する。

```rust
// Rust: トレイトベースのOOP

use std::fmt;

// トレイト: インターフェース + デフォルト実装 + 関連型
trait Animal: fmt::Display {
    fn name(&self) -> &str;
    fn sound(&self) -> &str;

    // デフォルト実装
    fn introduce(&self) -> String {
        format!("{}は「{}」と鳴きます", self.name(), self.sound())
    }
}

// 構造体 + トレイト実装
struct Dog {
    name: String,
    breed: String,
}

impl Animal for Dog {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "ワン" }
}

impl fmt::Display for Dog {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "犬「{}」({})", self.name, self.breed)
    }
}

struct Cat {
    name: String,
    indoor: bool,
}

impl Animal for Cat {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "ニャー" }
}

impl fmt::Display for Cat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let location = if self.indoor { "室内飼い" } else { "外飼い" };
        write!(f, "猫「{}」({})", self.name, location)
    }
}

// トレイトオブジェクト: 実行時ポリモーフィズム
fn introduce_all(animals: &[&dyn Animal]) {
    for animal in animals {
        println!("{}", animal.introduce());
    }
}

// ジェネリクス + トレイト境界: コンパイル時ポリモーフィズム
fn loudest_sound<T: Animal>(animals: &[T]) -> &str {
    // コンパイル時に型が確定 → 仮想関数テーブルなし → ゼロコスト
    animals.first().map(|a| a.sound()).unwrap_or("")
}

// 列挙型: 代数的データ型（Rust の強み）
enum Shape {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Triangle { base: f64, height: f64 },
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle { radius } => std::f64::consts::PI * radius * radius,
            Shape::Rectangle { width, height } => width * height,
            Shape::Triangle { base, height } => base * height / 2.0,
        }
    }

    fn perimeter(&self) -> f64 {
        match self {
            Shape::Circle { radius } => 2.0 * std::f64::consts::PI * radius,
            Shape::Rectangle { width, height } => 2.0 * (width + height),
            Shape::Triangle { base, height } => {
                let hyp = (base * base + height * height).sqrt();
                base + height + hyp
            }
        }
    }
}

// 所有権システム: コンパイル時のメモリ安全性保証
struct FileProcessor {
    path: String,
    content: Option<String>,
}

impl FileProcessor {
    fn new(path: &str) -> Self {
        FileProcessor {
            path: path.to_string(),
            content: None,
        }
    }

    // &self: 読み取り専用借用
    fn path(&self) -> &str {
        &self.path
    }

    // &mut self: 可変借用
    fn load(&mut self) -> Result<(), std::io::Error> {
        self.content = Some(std::fs::read_to_string(&self.path)?);
        Ok(())
    }

    // self: 所有権を消費（呼び出し後は使えない）
    fn into_content(self) -> Option<String> {
        self.content
    }
}
```

### 4.3 Go: シンプルさの追求

Goは意図的にOOPの多くの機能を省略し、シンプルさを追求した。

```go
// Go: 構造体 + インターフェース（暗黙的実装）
package main

import (
    "fmt"
    "math"
    "sort"
)

// インターフェース: 暗黙的に実装される
type Shape interface {
    Area() float64
    Perimeter() float64
    String() string
}

// 構造体（クラスの代わり）
type Circle struct {
    Radius float64
}

// メソッド（レシーバ付き関数）
func (c Circle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * math.Pi * c.Radius
}

func (c Circle) String() string {
    return fmt.Sprintf("Circle(r=%.2f)", c.Radius)
}

type Rectangle struct {
    Width, Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

func (r Rectangle) String() string {
    return fmt.Sprintf("Rect(%.2fx%.2f)", r.Width, r.Height)
}

// 埋め込み（Embedding）: 継承の代替
type NamedShape struct {
    Shape // Shape インターフェースを埋め込み
    Name  string
}

func (ns NamedShape) Describe() string {
    return fmt.Sprintf("%s: area=%.2f", ns.Name, ns.Area())
}

// インターフェースの合成
type ReadWriter interface {
    Reader
    Writer
}

type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

// 関数型: 小さなインターフェースを多用
type SortByArea []Shape

func (s SortByArea) Len() int           { return len(s) }
func (s SortByArea) Less(i, j int) bool { return s[i].Area() < s[j].Area() }
func (s SortByArea) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func main() {
    shapes := []Shape{
        Circle{Radius: 5},
        Rectangle{Width: 3, Height: 4},
        Circle{Radius: 2},
        Rectangle{Width: 10, Height: 1},
    }

    sort.Sort(SortByArea(shapes))

    for _, s := range shapes {
        fmt.Printf("%s -> Area: %.2f\n", s, s.Area())
    }
}
```

---

## 5. OOPの未来

```
2020s-2030s の OOP の方向性:

  1. 代数的データ型の普及
     → sealed class (Kotlin, Java 17+)
     → enum + match (Rust)
     → union types (TypeScript)
     → OOP + FP のハイブリッドパターンが標準化

  2. 不変性の主流化
     → record (Java), data class (Kotlin)
     → frozen dataclass (Python)
     → readonly (TypeScript)
     → 「デフォルト不変、必要な時だけ可変」が原則に

  3. 型推論の進化
     → ローカル変数の型推論は当たり前
     → 構造的型付けの普及
     → コンパイル時の安全性 + 記述の簡潔さ

  4. エフェクトシステム
     → 副作用の型レベルでの追跡
     → async/await の進化
     → 純粋関数と副作用の明確な分離

  5. コンポジション API
     → React Hooks, Vue Composition API
     → Swift Protocol Extensions
     → Rust Trait + ジェネリクス
     → 「クラスを使わないOOP的設計」の普及

  6. AI支援によるコード生成
     → OOP設計の自動提案
     → デザインパターンの自動適用
     → リファクタリングの AI 支援
```

### 5.1 エフェクトシステムと副作用の管理

```typescript
// TypeScript: Result型によるエラー処理（FP的アプローチ）

type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

// 副作用を明示的に型で表現
class UserService {
  constructor(
    private readonly db: Database,
    private readonly mailer: Mailer,
  ) {}

  // 戻り値の型が「成功 or 失敗」を明示
  async createUser(
    name: string,
    email: string,
  ): Promise<Result<User, CreateUserError>> {
    // バリデーション（純粋関数）
    const validationResult = this.validateInput(name, email);
    if (!validationResult.ok) return validationResult;

    // DB操作（副作用）
    const existingUser = await this.db.findUserByEmail(email);
    if (existingUser) {
      return err({ type: "DUPLICATE_EMAIL", email });
    }

    const user = await this.db.createUser({ name, email });

    // メール送信（副作用）
    const mailResult = await this.mailer.sendWelcome(user);
    if (!mailResult.ok) {
      // メール失敗はログに記録するが、ユーザー作成は成功
      console.warn("Welcome email failed:", mailResult.error);
    }

    return ok(user);
  }

  private validateInput(
    name: string,
    email: string,
  ): Result<void, CreateUserError> {
    if (!name.trim()) return err({ type: "INVALID_NAME", name });
    if (!email.includes("@")) return err({ type: "INVALID_EMAIL", email });
    return ok(undefined);
  }
}

type CreateUserError =
  | { type: "INVALID_NAME"; name: string }
  | { type: "INVALID_EMAIL"; email: string }
  | { type: "DUPLICATE_EMAIL"; email: string }
  | { type: "DATABASE_ERROR"; cause: Error };
```

### 5.2 コンポジションAPIパターンの台頭

```typescript
// TypeScript: React Hooks スタイル（クラスなしのOOP的設計）

// 状態管理のカプセル化（クラスを使わず）
function useCounter(initialValue: number = 0) {
  let count = initialValue;

  return {
    get value() { return count; },
    increment() { count++; },
    decrement() { count--; },
    reset() { count = initialValue; },
  };
}

// ビジネスロジックのカプセル化
function useShoppingCart() {
  const items: Map<string, { name: string; price: number; qty: number }> = new Map();

  return {
    addItem(id: string, name: string, price: number) {
      const existing = items.get(id);
      if (existing) {
        existing.qty++;
      } else {
        items.set(id, { name, price, qty: 1 });
      }
    },

    removeItem(id: string) {
      items.delete(id);
    },

    get total() {
      let sum = 0;
      for (const item of items.values()) {
        sum += item.price * item.qty;
      }
      return sum;
    },

    get itemCount() {
      let count = 0;
      for (const item of items.values()) {
        count += item.qty;
      }
      return count;
    },

    get isEmpty() {
      return items.size === 0;
    },
  };
}

// コンポジション: 複数の機能を組み合わせ
function useCheckout() {
  const cart = useShoppingCart();
  const step = useCounter(1);

  return {
    cart,
    step,

    get canProceed() {
      if (step.value === 1) return !cart.isEmpty;
      if (step.value === 2) return true; // 配送先入力済み
      return false;
    },

    nextStep() {
      if (this.canProceed && step.value < 3) {
        step.increment();
      }
    },

    previousStep() {
      if (step.value > 1) {
        step.decrement();
      }
    },
  };
}

// → クラスを使わずに、OOP の利点（カプセル化、コンポジション）を実現
// → 関数がオブジェクト（クロージャ）を返すパターン
// → テスト容易（モック不要、純粋関数的）
```

---

## まとめ

| 時代 | 言語 | 革新 |
|------|------|------|
| 1967 | Simula | クラス・継承の発明 |
| 1972 | Smalltalk | 純粋OOP・メッセージング・IDE・MVC |
| 1983 | C++ | 静的型付けOOP・実用化・RAII |
| 1993 | Ruby | 純粋OOP・メタプログラミング・開発者体験 |
| 1995 | Java | エンタープライズ標準・GC・JVM |
| 2010s | Kotlin/Swift | モダンOOP・FP融合・null安全 |
| 2020s | Rust/Go/TS | ポストOOP・コンポジション・型安全 |

---

## 次に読むべきガイド
→ [[02-oop-vs-other-paradigms.md]] — OOP vs 他パラダイム

---

## 参考文献
1. Kay, A. "The Early History of Smalltalk." ACM SIGPLAN, 1993.
2. Stroustrup, B. "The Design and Evolution of C++." Addison-Wesley, 1994.
3. Bloch, J. "Effective Java." 3rd Ed, Addison-Wesley, 2018.
4. Nygaard, K. and Dahl, O-J. "The Development of the Simula Languages." ACM SIGPLAN, 1978.
5. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
6. Matsakis, N. and Klock, F. "The Rust Language." ACM SIGAda, 2014.
7. Odersky, M. and Zenger, M. "Scalable Component Abstractions." OOPSLA, 2005.
