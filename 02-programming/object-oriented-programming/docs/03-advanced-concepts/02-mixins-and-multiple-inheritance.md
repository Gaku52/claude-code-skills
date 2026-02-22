# ミックスインと多重継承

> 多重継承の問題を回避しつつ、複数の振る舞いを組み合わせるための手法。Python のMRO、Ruby のモジュール、TypeScript のミックスインパターンを比較する。

## この章で学ぶこと

- [ ] 多重継承の問題（ダイヤモンド問題）とその解決策を理解する
- [ ] ミックスインパターンの実装方法を把握する
- [ ] 各言語でのアプローチの違いを学ぶ
- [ ] 協調的多重継承（cooperative multiple inheritance）を理解する
- [ ] ミックスインの設計原則とアンチパターンを把握する
- [ ] テスト戦略を習得する

---

## 1. 多重継承の問題

```
ダイヤモンド問題:

      ┌─────────┐
      │    A    │  greet() → "Hello from A"
      └────┬────┘
      ┌────┴────┐
      ▼         ▼
  ┌─────┐   ┌─────┐
  │  B  │   │  C  │  B.greet() → "Hello from B"
  └──┬──┘   └──┬──┘  C.greet() → "Hello from C"
     └────┬────┘
          ▼
      ┌─────┐
      │  D  │  D.greet() → ???（B? C? A?）
      └─────┘

解決策:
  C++:    仮想継承（virtual inheritance）
  Python: MRO（C3線形化）
  Java:   多重継承禁止（インターフェースのみ）
  Ruby:   モジュール（Module）
  Rust:   トレイト（多重継承なし）
```

### 1.1 ダイヤモンド問題の本質

```
ダイヤモンド問題の3つの側面:

  1. メソッド解決の曖昧性（Method Resolution Ambiguity）
     → 同名メソッドが複数の経路で継承される
     → どのバージョンを呼ぶべきか不明確
     → D.greet() は B.greet() か C.greet() か？

  2. コンストラクタの多重呼び出し（Constructor Duplication）
     → A のコンストラクタが B 経由と C 経由で2回呼ばれる
     → リソースの二重初期化、状態の不整合

  3. 状態の共有問題（Shared State Problem）
     → A のフィールドが B と C で別々に初期化される
     → D から見たとき、どの状態が正しいのか？
     → フィールドのコピーが2つ存在する可能性

  歴史:
    → 1969年: Simula 67 は単一継承のみ
    → 1983年: C++ が多重継承を導入
    → 1987年: ダイヤモンド問題が広く認識される
    → 1995年: Java は多重継承を意図的に排除
    → 2000年代: トレイト / ミックスインが主流に
```

### 1.2 C++: 仮想継承

```cpp
// C++: ダイヤモンド問題と仮想継承

#include <iostream>
#include <string>

// 仮想継承なしの場合
class A {
public:
    int value;
    A(int v) : value(v) {
        std::cout << "A(" << v << ") constructed" << std::endl;
    }
    virtual std::string greet() { return "Hello from A"; }
};

// 通常の継承: A のコピーが2つ作られる
class B_normal : public A {
public:
    B_normal() : A(1) {}
    std::string greet() override { return "Hello from B"; }
};

class C_normal : public A {
public:
    C_normal() : A(2) {}
    std::string greet() override { return "Hello from C"; }
};

// class D_normal : public B_normal, public C_normal {};
// → コンパイルエラー: A のメンバーが曖昧
// → d.value は B_normal::value か C_normal::value か？

// 仮想継承: A のインスタンスは1つだけ
class B_virtual : virtual public A {
public:
    B_virtual() : A(1) {}
    std::string greet() override { return "Hello from B"; }
};

class C_virtual : virtual public A {
public:
    C_virtual() : A(2) {}
    std::string greet() override { return "Hello from C"; }
};

class D : public B_virtual, public C_virtual {
public:
    // 仮想基底クラス A のコンストラクタは最派生クラスが呼ぶ
    D() : A(0), B_virtual(), C_virtual() {}

    // greet() はオーバーライドが必要（B と C のどちらか曖昧なため）
    std::string greet() override {
        return "Hello from D (B says: " + B_virtual::greet() + ")";
    }
};

int main() {
    D d;
    std::cout << d.greet() << std::endl;
    std::cout << d.value << std::endl;  // 0（A のコンストラクタは D が呼ぶ）
    // A(0) constructed が1回だけ出力される
    return 0;
}
```

### 1.3 各言語のアプローチ比較

```
┌──────────┬──────────────────────┬───────────────────┬──────────────┐
│ 言語     │ 多重継承の扱い       │ 振る舞いの合成手段│ 状態の共有   │
├──────────┼──────────────────────┼───────────────────┼──────────────┤
│ C++      │ 仮想継承で対応       │ 仮想基底クラス    │ 可能         │
│ Python   │ MRO で線形化         │ 多重継承 + Mixin  │ 可能         │
│ Java     │ クラス多重継承禁止   │ インターフェース  │ 不可         │
│ Kotlin   │ クラス多重継承禁止   │ インターフェース  │ 不可         │
│ Ruby     │ クラス多重継承禁止   │ Module            │ 不可（注1）  │
│ Scala    │ クラス多重継承禁止   │ trait             │ 可能（val）  │
│ Rust     │ クラス自体がない     │ trait             │ 不可         │
│ Swift    │ クラス多重継承禁止   │ Protocol          │ 不可         │
│ TypeScript│ クラス多重継承禁止  │ ミックスイン関数  │ 可能（注2）  │
│ PHP      │ クラス多重継承禁止   │ trait             │ 可能         │
└──────────┴──────────────────────┴───────────────────┴──────────────┘

注1: Ruby の Module は include 先のインスタンス変数にアクセスはできるが、
     Module 自体が状態を持つわけではない
注2: TypeScript のミックスインはクラスを返す関数なので、
     状態（プロパティ）を追加できる
```

---

## 2. Python: MRO（Method Resolution Order）

### 2.1 C3線形化アルゴリズム

```python
# Python: C3線形化によるMRO
class A:
    def greet(self):
        return "Hello from A"

class B(A):
    def greet(self):
        return "Hello from B"

class C(A):
    def greet(self):
        return "Hello from C"

class D(B, C):
    pass

d = D()
print(d.greet())  # "Hello from B"

# MROの確認
print(D.__mro__)
# (D, B, C, A, object)
# → D → B → C → A → object の順で探索
```

```python
# C3線形化アルゴリズムの詳細

# C3線形化の定式化:
# L(C) = C + merge(L(B1), L(B2), ..., L(Bn), B1 B2 ... Bn)
#
# merge のルール:
#   1. 最初のリストの先頭要素を取る
#   2. その要素が他のリストの「先頭以外」に出現しなければ、結果に追加
#   3. 出現する場合、次のリストの先頭要素を試す
#   4. すべてのリストが空になるまで繰り返す

class O: pass   # object の代わり

class A(O):
    def method(self):
        return "A"

class B(O):
    def method(self):
        return "B"

class C(O):
    def method(self):
        return "C"

class D(O):
    def method(self):
        return "D"

class E(A, B):
    pass

class F(C, D):
    pass

class G(E, F):
    pass

# MRO の計算:
# L(O) = [O]
# L(A) = [A, O]
# L(B) = [B, O]
# L(C) = [C, O]
# L(D) = [D, O]
# L(E) = E + merge(L(A), L(B), [A, B])
#       = E + merge([A, O], [B, O], [A, B])
#       = [E, A] + merge([O], [B, O], [B])
#       = [E, A, B] + merge([O], [O])
#       = [E, A, B, O]
# L(F) = [F, C, D, O]
# L(G) = G + merge(L(E), L(F), [E, F])
#       = G + merge([E, A, B, O], [F, C, D, O], [E, F])
#       = [G, E] + merge([A, B, O], [F, C, D, O], [F])
#       = [G, E, A, B] + merge([O], [F, C, D, O], [F])
#       = [G, E, A, B, F] + merge([O], [C, D, O])
#       = [G, E, A, B, F, C, D] + merge([O], [O])
#       = [G, E, A, B, F, C, D, O]

print(G.__mro__)
# (<class 'G'>, <class 'E'>, <class 'A'>, <class 'B'>,
#  <class 'F'>, <class 'C'>, <class 'D'>, <class 'O'>)


# C3線形化が失敗するケース
# 矛盾する継承順序を検出してエラーを出す
class X(A, B): pass  # A が B より先
class Y(B, A): pass  # B が A より先

# class Z(X, Y): pass
# → TypeError: Cannot create a consistent method resolution order (MRO)
#   for bases A, B
# X は A → B の順序を要求し、Y は B → A の順序を要求するため矛盾
```

### 2.2 協調的多重継承（Cooperative Multiple Inheritance）

```python
# super() を使った協調的多重継承

class Base:
    def __init__(self, **kwargs):
        # 最終的な基底クラスは残ったkwargsを無視
        pass

    def process(self, data: str) -> str:
        return data


class LoggingMixin(Base):
    """ログ記録を追加するミックスイン"""
    def __init__(self, *, log_prefix: str = "LOG", **kwargs):
        super().__init__(**kwargs)  # 次のクラスに委譲
        self.log_prefix = log_prefix
        self._logs: list[str] = []

    def process(self, data: str) -> str:
        self._logs.append(f"[{self.log_prefix}] Processing: {data}")
        # super() で MRO の次のクラスの process を呼ぶ
        result = super().process(data)
        self._logs.append(f"[{self.log_prefix}] Result: {result}")
        return result


class ValidationMixin(Base):
    """バリデーションを追加するミックスイン"""
    def __init__(self, *, max_length: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length

    def process(self, data: str) -> str:
        if len(data) > self.max_length:
            raise ValueError(
                f"Data too long: {len(data)} > {self.max_length}"
            )
        return super().process(data)


class TransformMixin(Base):
    """データ変換を追加するミックスイン"""
    def __init__(self, *, uppercase: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.uppercase = uppercase

    def process(self, data: str) -> str:
        if self.uppercase:
            data = data.upper()
        return super().process(data)


class CacheMixin(Base):
    """キャッシュを追加するミックスイン"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache: dict[str, str] = {}

    def process(self, data: str) -> str:
        if data in self._cache:
            return self._cache[data]
        result = super().process(data)
        self._cache[data] = result
        return result


# ミックスインの合成
class TextProcessor(
    LoggingMixin,
    ValidationMixin,
    TransformMixin,
    CacheMixin,
    Base,
):
    """複数のミックスインを合成したテキストプロセッサ"""
    pass


# MRO の確認
print(TextProcessor.__mro__)
# TextProcessor → LoggingMixin → ValidationMixin →
# TransformMixin → CacheMixin → Base → object

# 使用例
processor = TextProcessor(
    log_prefix="TXT",
    max_length=200,
    uppercase=True,
)

result = processor.process("hello world")
print(result)        # "HELLO WORLD"
print(processor._logs)
# ['[TXT] Processing: hello world', '[TXT] Result: HELLO WORLD']

# process の呼び出しチェーン:
# 1. LoggingMixin.process: ログ記録 → super()
# 2. ValidationMixin.process: バリデーション → super()
# 3. TransformMixin.process: 大文字変換 → super()
# 4. CacheMixin.process: キャッシュ → super()
# 5. Base.process: データをそのまま返す
```

```python
# super() のメカニズムの詳細

class A:
    def method(self):
        print("A.method start")
        # A は MRO の最後なので super() は object
        print("A.method end")

class B(A):
    def method(self):
        print("B.method start")
        super().method()  # MRO の次 → C（注: B の次は A ではない!）
        print("B.method end")

class C(A):
    def method(self):
        print("C.method start")
        super().method()  # MRO の次 → A
        print("C.method end")

class D(B, C):
    def method(self):
        print("D.method start")
        super().method()  # MRO の次 → B
        print("D.method end")

d = D()
d.method()
# 出力:
# D.method start
# B.method start
# C.method start    ← B の super() は C を呼ぶ！（A ではない）
# A.method start
# A.method end
# C.method end
# B.method end
# D.method end

# MRO: D → B → C → A → object
# super() は「親クラス」ではなく「MRO の次のクラス」を呼ぶ
```

### 2.3 Python: 実践的ミックスインパターン

```python
# ミックスイン = 単独では使わない、機能を追加するためのクラス
import json
from datetime import datetime
from typing import Any, TypeVar, Type

T = TypeVar("T")


class JsonMixin:
    """JSON変換機能を追加"""
    def to_json(self) -> str:
        return json.dumps(self._to_dict(), default=str, ensure_ascii=False)

    def _to_dict(self) -> dict[str, Any]:
        """シリアライズ対象のフィールドを返す"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                result[key] = value
        return result

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        data = json.loads(json_str)
        return cls(**data)


class TimestampMixin:
    """タイムスタンプ機能を追加"""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.created_at = datetime.now()
            self.updated_at = datetime.now()

        cls.__init__ = new_init

    def touch(self) -> None:
        """updated_at を更新"""
        self.updated_at = datetime.now()


class LoggableMixin:
    """ログ出力機能を追加"""
    def log(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] [{level}] [{type(self).__name__}] {message}")

    def log_error(self, message: str) -> None:
        self.log(message, level="ERROR")

    def log_warning(self, message: str) -> None:
        self.log(message, level="WARNING")


class ValidatableMixin:
    """バリデーション機能を追加"""

    def validate(self) -> list[str]:
        """バリデーションエラーのリストを返す"""
        errors = []
        for attr_name in dir(self):
            if attr_name.startswith("validate_"):
                field_name = attr_name[len("validate_"):]
                validator = getattr(self, attr_name)
                error = validator()
                if error:
                    errors.append(f"{field_name}: {error}")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


class EventEmitterMixin:
    """イベント発行機能を追加"""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._event_handlers: dict[str, list] = {}

        cls.__init__ = new_init

    def on(self, event: str, handler) -> None:
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def emit(self, event: str, *args, **kwargs) -> None:
        for handler in self._event_handlers.get(event, []):
            handler(*args, **kwargs)

    def off(self, event: str, handler=None) -> None:
        if handler is None:
            self._event_handlers.pop(event, None)
        elif event in self._event_handlers:
            self._event_handlers[event] = [
                h for h in self._event_handlers[event] if h != handler
            ]


# ミックスインを組み合わせ
class User(
    JsonMixin,
    TimestampMixin,
    LoggableMixin,
    ValidatableMixin,
    EventEmitterMixin,
):
    def __init__(self, name: str, email: str, age: int = 0):
        self.name = name
        self.email = email
        self.age = age

    def validate_name(self) -> str | None:
        if not self.name or len(self.name) < 2:
            return "名前は2文字以上必要です"
        return None

    def validate_email(self) -> str | None:
        if "@" not in self.email:
            return "メールアドレスの形式が不正です"
        return None

    def validate_age(self) -> str | None:
        if self.age < 0 or self.age > 150:
            return "年齢は0〜150の範囲で指定してください"
        return None


# 使用例
user = User("田中太郎", "tanaka@example.com", age=30)

# JsonMixin
print(user.to_json())
# {"name": "田中太郎", "email": "tanaka@example.com", "age": 30}

# TimestampMixin
print(user.created_at)

# LoggableMixin
user.log("ログインしました")

# ValidatableMixin
print(user.validate())  # []（エラーなし）
print(user.is_valid())  # True

# EventEmitterMixin
user.on("login", lambda: print("ログインイベント発生"))
user.emit("login")
```

### 2.4 Python: __init_subclass__ を使ったメタプログラミング

```python
# __init_subclass__: サブクラス定義時に自動実行されるフック

class RegisterMixin:
    """サブクラスを自動登録するミックスイン"""
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, register_name: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        name = register_name or cls.__name__
        RegisterMixin._registry[name] = cls

    @classmethod
    def get_registered(cls, name: str) -> type | None:
        return RegisterMixin._registry.get(name)

    @classmethod
    def list_registered(cls) -> list[str]:
        return list(RegisterMixin._registry.keys())


class Plugin(RegisterMixin):
    def execute(self):
        raise NotImplementedError


class CSVExporter(Plugin, register_name="csv"):
    def execute(self):
        return "Exporting as CSV..."


class JSONExporter(Plugin, register_name="json"):
    def execute(self):
        return "Exporting as JSON..."


class XMLExporter(Plugin, register_name="xml"):
    def execute(self):
        return "Exporting as XML..."


# 自動登録されたプラグインを使用
print(RegisterMixin.list_registered())
# ['Plugin', 'csv', 'json', 'xml']

exporter_cls = RegisterMixin.get_registered("csv")
exporter = exporter_cls()
print(exporter.execute())  # "Exporting as CSV..."


# __set_name__ を使ったディスクリプタミックスイン
class TypeCheckedMixin:
    """型チェック付きプロパティを自動定義"""

    class TypeCheckedDescriptor:
        def __init__(self, name: str, expected_type: type):
            self.name = name
            self.expected_type = expected_type
            self.private_name = f"_typechecked_{name}"

        def __set_name__(self, owner, name):
            self.name = name
            self.private_name = f"_typechecked_{name}"

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return getattr(obj, self.private_name, None)

        def __set__(self, obj, value):
            if not isinstance(value, self.expected_type):
                raise TypeError(
                    f"{self.name} must be {self.expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            setattr(obj, self.private_name, value)


class StrictUser:
    name = TypeCheckedMixin.TypeCheckedDescriptor("name", str)
    age = TypeCheckedMixin.TypeCheckedDescriptor("age", int)
    email = TypeCheckedMixin.TypeCheckedDescriptor("email", str)

    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email


user = StrictUser("田中", 30, "tanaka@example.com")  # OK
# StrictUser("田中", "30", "tanaka@example.com")  # TypeError
```

---

## 3. Ruby: モジュール

### 3.1 include, extend, prepend の違い

```ruby
# Ruby: Module によるミックスイン

# Module の3つの取り込み方
#   include:  インスタンスメソッドとして取り込む
#   extend:   クラスメソッドとして取り込む
#   prepend:  インスタンスメソッドとして取り込む（メソッド探索で先に見つかる）

module Greetable
  def greet
    "Hello, I'm #{name}"
  end
end

module ClassInfo
  def info
    "Class: #{self.name}, Methods: #{instance_methods(false).count}"
  end
end

module Logging
  def greet
    puts "[LOG] greet called"
    super  # prepend なら元の greet が呼ばれる
  end
end

class Person
  include Greetable  # インスタンスメソッドとして追加
  extend ClassInfo   # クラスメソッドとして追加
  prepend Logging    # greet を上書き（super で元のメソッドを呼べる）

  attr_reader :name

  def initialize(name)
    @name = name
  end
end

person = Person.new("田中")
puts person.greet      # [LOG] greet called  →  "Hello, I'm 田中"
puts Person.info       # "Class: Person, Methods: 1"

# メソッド探索順序
puts Person.ancestors
# [Logging, Person, Greetable, Object, Kernel, BasicObject]
# prepend は Person の前に来る
# include は Person の後に来る
```

```ruby
# Ruby: Module の高度な使い方

module Serializable
  def to_json
    require 'json'
    JSON.generate(instance_variables.each_with_object({}) { |var, hash|
      hash[var.to_s.delete('@')] = instance_variable_get(var)
    })
  end

  def to_hash
    instance_variables.each_with_object({}) { |var, hash|
      hash[var.to_s.delete('@').to_sym] = instance_variable_get(var)
    }
  end
end

module Loggable
  def log(message, level: :info)
    timestamp = Time.now.strftime('%Y-%m-%d %H:%M:%S')
    puts "[#{timestamp}] [#{level.upcase}] [#{self.class.name}] #{message}"
  end
end

module Cacheable
  def self.included(base)
    # Module が include されたときに呼ばれるフック
    base.instance_variable_set(:@cache, {})
    base.extend(ClassMethods)
  end

  module ClassMethods
    def cache
      @cache
    end

    def cached_method(method_name)
      original = instance_method(method_name)
      define_method(method_name) do |*args|
        cache_key = "#{method_name}:#{args.hash}"
        self.class.cache[cache_key] ||= original.bind(self).call(*args)
      end
    end
  end

  def cache_key
    "#{self.class.name}:#{object_id}"
  end
end

module Comparable
  def <=>(other)
    raise NotImplementedError, "#{self.class} must implement <=>"
  end

  def <(other);  (self <=> other) < 0; end
  def >(other);  (self <=> other) > 0; end
  def <=(other); (self <=> other) <= 0; end
  def >=(other); (self <=> other) >= 0; end
  def ==(other); (self <=> other) == 0; end
end

class Product
  include Serializable
  include Loggable
  include Cacheable
  include Comparable

  attr_accessor :name, :price, :category

  def initialize(name, price, category = "general")
    @name = name
    @price = price
    @category = category
  end

  def <=>(other)
    @price <=> other.price
  end

  def expensive?
    @price > 10000
  end
  cached_method :expensive?  # Cacheable でキャッシュ化
end

product = Product.new("ノートPC", 89800, "electronics")
puts product.to_json        # Serializable
product.log("在庫追加")      # Loggable
puts product.cache_key      # Cacheable
puts product > Product.new("マウス", 2980)  # Comparable → true
```

### 3.2 Ruby: concern パターン（Rails）

```ruby
# ActiveSupport::Concern パターン
# Rails で広く使われるミックスインの構造化手法

module ActiveSupport
  module Concern
    def self.extended(base)
      base.instance_variable_set(:@_dependencies, [])
    end

    def included(base)
      @_dependencies.each { |dep| base.include(dep) }
      base.class_eval(&@_included_block) if @_included_block
      base.extend(const_get(:ClassMethods)) if const_defined?(:ClassMethods)
    end

    def class_methods(&block)
      mod = const_defined?(:ClassMethods, false) ?
        const_get(:ClassMethods) :
        const_set(:ClassMethods, Module.new)
      mod.module_eval(&block)
    end

    def included_block(&block)
      @_included_block = block
    end
  end
end

# Concern の使用例
module Searchable
  extend ActiveSupport::Concern

  class_methods do
    def search(query)
      puts "Searching #{self.name} for: #{query}"
      # 実際のRailsでは: where("name LIKE ?", "%#{query}%")
    end

    def search_by_field(field, value)
      puts "Searching #{self.name} by #{field}: #{value}"
    end
  end

  def highlight(query)
    # インスタンスメソッド
    puts "Highlighting '#{query}' in #{self.class.name}"
  end
end

module Taggable
  extend ActiveSupport::Concern

  class_methods do
    def find_by_tag(tag)
      puts "Finding #{self.name} with tag: #{tag}"
    end
  end

  def add_tag(tag)
    @tags ||= []
    @tags << tag
  end

  def tags
    @tags || []
  end
end

module Auditable
  extend ActiveSupport::Concern

  class_methods do
    def audit_log
      @audit_log ||= []
    end
  end

  def audit(action)
    self.class.audit_log << {
      action: action,
      record: self,
      timestamp: Time.now
    }
  end
end

class Article
  include Searchable
  include Taggable
  include Auditable

  attr_accessor :title, :body

  def initialize(title, body)
    @title = title
    @body = body
  end
end

Article.search("Ruby")             # Searchable
Article.find_by_tag("programming") # Taggable

article = Article.new("Ruby入門", "Rubyの基礎...")
article.add_tag("ruby")           # Taggable
article.highlight("Ruby")         # Searchable
article.audit("created")          # Auditable
```

---

## 4. TypeScript: ミックスインパターン

### 4.1 クラス式ベースのミックスイン

```typescript
// TypeScript: ミックスイン（クラス式を使ったパターン）
type Constructor<T = {}> = new (...args: any[]) => T;

// ミックスイン関数
function Timestamped<TBase extends Constructor>(Base: TBase) {
  return class Timestamped extends Base {
    createdAt = new Date();
    updatedAt = new Date();

    touch() {
      this.updatedAt = new Date();
    }
  };
}

function Activatable<TBase extends Constructor>(Base: TBase) {
  return class Activatable extends Base {
    isActive = false;

    activate() { this.isActive = true; }
    deactivate() { this.isActive = false; }
  };
}

function Taggable<TBase extends Constructor>(Base: TBase) {
  return class Taggable extends Base {
    tags: Set<string> = new Set();

    addTag(tag: string) { this.tags.add(tag); }
    removeTag(tag: string) { this.tags.delete(tag); }
    hasTag(tag: string) { return this.tags.has(tag); }
  };
}

// ベースクラス
class User {
  constructor(public name: string, public email: string) {}
}

// ミックスインを合成
const EnhancedUser = Taggable(Activatable(Timestamped(User)));

const user = new EnhancedUser("田中", "tanaka@example.com");
user.activate();          // Activatable
user.addTag("premium");   // Taggable
user.touch();             // Timestamped
```

### 4.2 型安全なミックスイン（高度なパターン）

```typescript
// TypeScript: 型安全なミックスインの実装

// より厳密な型定義
type GConstructor<T = {}> = new (...args: any[]) => T;

// インターフェースで各ミックスインの型を定義
interface HasId {
  id: string;
}

interface HasName {
  name: string;
}

// 制約付きミックスイン: HasId を持つクラスにのみ適用可能
function Persistable<TBase extends GConstructor<HasId>>(Base: TBase) {
  return class Persistable extends Base {
    isPersisted = false;

    async save(): Promise<void> {
      console.log(`Saving entity with id: ${this.id}`);
      this.isPersisted = true;
    }

    async delete(): Promise<void> {
      console.log(`Deleting entity with id: ${this.id}`);
      this.isPersisted = false;
    }
  };
}

// 制約付きミックスイン: HasName を持つクラスにのみ適用可能
function Searchable<TBase extends GConstructor<HasName>>(Base: TBase) {
  return class Searchable extends Base {
    matches(query: string): boolean {
      return this.name.toLowerCase().includes(query.toLowerCase());
    }
  };
}

// Serializable: 任意のクラスに適用可能
function Serializable<TBase extends GConstructor>(Base: TBase) {
  return class Serializable extends Base {
    toJSON(): Record<string, unknown> {
      const result: Record<string, unknown> = {};
      for (const key of Object.keys(this)) {
        const value = (this as any)[key];
        if (typeof value !== "function") {
          result[key] = value;
        }
      }
      return result;
    }

    toString(): string {
      return JSON.stringify(this.toJSON(), null, 2);
    }
  };
}

// Validatable: バリデーションルールを追加
function Validatable<TBase extends GConstructor>(Base: TBase) {
  return class Validatable extends Base {
    private _validationRules: Map<string, (value: any) => string | null> =
      new Map();

    addRule(field: string, rule: (value: any) => string | null): void {
      this._validationRules.set(field, rule);
    }

    validate(): string[] {
      const errors: string[] = [];
      for (const [field, rule] of this._validationRules) {
        const value = (this as any)[field];
        const error = rule(value);
        if (error) errors.push(`${field}: ${error}`);
      }
      return errors;
    }

    isValid(): boolean {
      return this.validate().length === 0;
    }
  };
}

// ベースクラス
class Entity {
  constructor(public id: string, public name: string) {}
}

// ミックスインを合成
const EnhancedEntity = Validatable(
  Serializable(
    Searchable(
      Persistable(Entity)
    )
  )
);

// 使用例
const entity = new EnhancedEntity("1", "Product A");
entity.addRule("name", (v) =>
  v.length < 2 ? "Name must be at least 2 characters" : null,
);

console.log(entity.matches("product"));  // true (Searchable)
console.log(entity.isValid());            // true (Validatable)
console.log(entity.toJSON());             // { id: "1", name: "Product A" }
await entity.save();                      // "Saving entity with id: 1"
```

### 4.3 デコレータベースのミックスイン

```typescript
// TypeScript 5.0+ デコレータを使ったミックスイン

// メソッドデコレータ: ログを追加
function logged(
  target: any,
  context: ClassMethodDecoratorContext,
) {
  const methodName = String(context.name);
  return function (this: any, ...args: any[]) {
    console.log(`[${methodName}] Called with:`, args);
    const result = target.call(this, ...args);
    console.log(`[${methodName}] Returned:`, result);
    return result;
  };
}

// メソッドデコレータ: キャッシュを追加
function cached(
  target: any,
  context: ClassMethodDecoratorContext,
) {
  const cache = new Map<string, any>();
  return function (this: any, ...args: any[]) {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = target.call(this, ...args);
    cache.set(key, result);
    return result;
  };
}

// メソッドデコレータ: リトライを追加
function retry(maxAttempts: number) {
  return function (
    target: any,
    context: ClassMethodDecoratorContext,
  ) {
    return async function (this: any, ...args: any[]) {
      let lastError: Error | undefined;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          return await target.call(this, ...args);
        } catch (error) {
          lastError = error as Error;
          console.log(
            `Attempt ${attempt}/${maxAttempts} failed: ${lastError.message}`,
          );
        }
      }
      throw lastError;
    };
  };
}

// クラスデコレータ: タイムスタンプを追加
function withTimestamps<T extends Constructor>(Base: T) {
  return class extends Base {
    createdAt = new Date();
    updatedAt = new Date();

    touch() {
      this.updatedAt = new Date();
    }
  };
}

// 使用例
@withTimestamps
class ApiClient {
  constructor(private baseUrl: string) {}

  @logged
  @retry(3)
  async fetchData(endpoint: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}${endpoint}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  @logged
  @cached
  computeExpensiveResult(input: number): number {
    // 重い計算のシミュレーション
    let result = 0;
    for (let i = 0; i < input * 1000; i++) {
      result += Math.sin(i);
    }
    return result;
  }
}
```

---

## 5. Rust: トレイト合成

```rust
// Rust: トレイトによる振る舞いの合成

use std::fmt;

// トレイト定義
trait Displayable {
    fn display_name(&self) -> String;
}

trait Serializable {
    fn serialize(&self) -> String;
    fn content_type(&self) -> &str { "application/json" }
}

trait Validatable {
    fn validate(&self) -> Result<(), Vec<String>>;

    fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
}

trait Auditable {
    fn audit_log(&self) -> String;
}

// 複数トレイトを実装
struct Product {
    id: u64,
    name: String,
    price: f64,
    category: String,
}

impl Displayable for Product {
    fn display_name(&self) -> String {
        format!("{} (¥{:.0})", self.name, self.price)
    }
}

impl Serializable for Product {
    fn serialize(&self) -> String {
        format!(
            r#"{{"id":{},"name":"{}","price":{},"category":"{}"}}"#,
            self.id, self.name, self.price, self.category
        )
    }
}

impl Validatable for Product {
    fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.name.is_empty() {
            errors.push("Name is required".to_string());
        }
        if self.price < 0.0 {
            errors.push("Price must be non-negative".to_string());
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

impl Auditable for Product {
    fn audit_log(&self) -> String {
        format!("Product[{}]: {} - ¥{:.0}", self.id, self.name, self.price)
    }
}

// トレイト境界で複数のトレイトを要求
fn process_entity<T>(entity: &T)
where
    T: Displayable + Serializable + Validatable + Auditable,
{
    // バリデーション
    match entity.validate() {
        Ok(()) => println!("✓ Validation passed"),
        Err(errors) => {
            for err in &errors {
                println!("✗ {}", err);
            }
            return;
        }
    }

    // 表示
    println!("Name: {}", entity.display_name());

    // シリアライズ
    println!("JSON: {}", entity.serialize());

    // 監査ログ
    println!("Audit: {}", entity.audit_log());
}

// トレイトオブジェクトとしてまとめる
// 注: 複数のトレイトをトレイトオブジェクトとして使う場合は
//     スーパートレイトを定義する
trait FullyFeatured: Displayable + Serializable + Validatable + Auditable {}

// ブランケット実装: 4つすべてを実装する型は自動的に FullyFeatured
impl<T> FullyFeatured for T
where
    T: Displayable + Serializable + Validatable + Auditable,
{}

fn process_any(entity: &dyn FullyFeatured) {
    println!("Name: {}", entity.display_name());
    println!("JSON: {}", entity.serialize());
}
```

```rust
// Rust: Derive マクロによる自動ミックスイン

// 標準の Derive マクロ
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Point {
    x: i32,
    y: i32,
}

// カスタム Derive マクロ（procedural macro）の使用例
// 注: 実際にはproc-macroクレートが必要
// #[derive(Serialize, Deserialize)]  // serde
// #[derive(Builder)]                  // derive_builder
// #[derive(Display)]                  // derive_more

// Deref と DerefMut でニュータイプパターン
use std::ops::{Deref, DerefMut};

struct Email(String);

impl Email {
    fn new(value: &str) -> Result<Self, String> {
        if value.contains('@') && value.contains('.') {
            Ok(Email(value.to_string()))
        } else {
            Err(format!("Invalid email: {}", value))
        }
    }
}

impl Deref for Email {
    type Target = String;
    fn deref(&self) -> &String {
        &self.0
    }
}

// Email は String のすべてのメソッドを使える
let email = Email::new("user@example.com").unwrap();
println!("Length: {}", email.len());       // String::len()
println!("Upper: {}", email.to_uppercase()); // String::to_uppercase()

// ただし、暗黙の型変換は起きない
// let s: String = email;  // ❌ コンパイルエラー
let s: &str = &email;     // ✅ Deref coercion
```

---

## 6. Java: デフォルトメソッドとインターフェースの合成

```java
// Java: インターフェースのデフォルトメソッドによるミックスイン的パターン

public interface Identifiable {
    String getId();
}

public interface Timestamped {
    Instant getCreatedAt();
    Instant getUpdatedAt();
    void setUpdatedAt(Instant updatedAt);

    default void touch() {
        setUpdatedAt(Instant.now());
    }

    default Duration age() {
        return Duration.between(getCreatedAt(), Instant.now());
    }
}

public interface Loggable {
    default void logInfo(String message) {
        System.out.printf("[INFO] [%s] %s%n", getClass().getSimpleName(), message);
    }

    default void logError(String message, Throwable error) {
        System.out.printf("[ERROR] [%s] %s: %s%n",
            getClass().getSimpleName(), message, error.getMessage());
    }
}

public interface Validatable {
    List<String> validate();

    default boolean isValid() {
        return validate().isEmpty();
    }

    default void validateOrThrow() {
        List<String> errors = validate();
        if (!errors.isEmpty()) {
            throw new ValidationException(
                String.join("; ", errors)
            );
        }
    }
}

public interface Serializable {
    default String toJson() {
        // リフレクションを使った簡易実装
        var sb = new StringBuilder("{");
        var fields = getClass().getDeclaredFields();
        for (int i = 0; i < fields.length; i++) {
            fields[i].setAccessible(true);
            try {
                sb.append(String.format("\"%s\":\"%s\"",
                    fields[i].getName(),
                    fields[i].get(this)));
            } catch (IllegalAccessException e) {
                // skip
            }
            if (i < fields.length - 1) sb.append(",");
        }
        sb.append("}");
        return sb.toString();
    }
}

// 複数のインターフェースを実装
public class User implements
        Identifiable, Timestamped, Loggable,
        Validatable, Serializable {

    private final String id;
    private final String name;
    private final String email;
    private final Instant createdAt;
    private Instant updatedAt;

    public User(String id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.createdAt = Instant.now();
        this.updatedAt = Instant.now();
    }

    @Override
    public String getId() { return id; }

    @Override
    public Instant getCreatedAt() { return createdAt; }

    @Override
    public Instant getUpdatedAt() { return updatedAt; }

    @Override
    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }

    @Override
    public List<String> validate() {
        var errors = new ArrayList<String>();
        if (name == null || name.length() < 2) {
            errors.add("名前は2文字以上必要です");
        }
        if (email == null || !email.contains("@")) {
            errors.add("メールアドレスが不正です");
        }
        return errors;
    }
}

// 使用例
var user = new User("1", "田中", "tanaka@example.com");
user.logInfo("ログイン");        // Loggable
user.touch();                    // Timestamped
System.out.println(user.isValid());  // Validatable
System.out.println(user.toJson());   // Serializable
```

```java
// Java: デフォルトメソッドの衝突解決

interface Flyable {
    default String move() { return "Flying"; }
}

interface Swimmable {
    default String move() { return "Swimming"; }
}

interface Walkable {
    default String move() { return "Walking"; }
}

// 3つの move() が衝突 → 明示的にオーバーライドが必要
class Duck implements Flyable, Swimmable, Walkable {
    @Override
    public String move() {
        // 特定のインターフェースのデフォルト実装を選択可能
        return Flyable.super.move();
    }

    // 状況に応じて切り替え
    public String move(String context) {
        return switch (context) {
            case "air" -> Flyable.super.move();
            case "water" -> Swimmable.super.move();
            case "land" -> Walkable.super.move();
            default -> move();
        };
    }
}

Duck duck = new Duck();
System.out.println(duck.move());          // "Flying"
System.out.println(duck.move("water"));   // "Swimming"
System.out.println(duck.move("land"));    // "Walking"
```

---

## 7. Kotlin: デリゲーションによるミックスイン

```kotlin
// Kotlin: by キーワードによるインターフェース委譲

interface Logger {
    fun log(message: String)
    fun logError(message: String, error: Throwable)
}

interface Cache<K, V> {
    fun get(key: K): V?
    fun put(key: K, value: V)
    fun invalidate(key: K)
    fun clear()
}

// 具象実装
class ConsoleLogger : Logger {
    override fun log(message: String) {
        println("[INFO] $message")
    }

    override fun logError(message: String, error: Throwable) {
        println("[ERROR] $message: ${error.message}")
    }
}

class InMemoryCache<K, V> : Cache<K, V> {
    private val store = mutableMapOf<K, V>()

    override fun get(key: K): V? = store[key]
    override fun put(key: K, value: V) { store[key] = value }
    override fun invalidate(key: K) { store.remove(key) }
    override fun clear() { store.clear() }
}

// by キーワードで委譲（コンポジションだが、インターフェースを満たす）
class UserService(
    private val logger: Logger = ConsoleLogger(),
    private val cache: Cache<String, User> = InMemoryCache(),
) : Logger by logger, Cache<String, User> by cache {

    fun findUser(id: String): User? {
        // Cache.get を直接呼べる（by で委譲されているため）
        val cached = get(id)
        if (cached != null) {
            log("Cache hit for user: $id")
            return cached
        }

        log("Cache miss for user: $id")
        // DB から取得
        val user = fetchFromDb(id) ?: return null
        put(id, user)  // Cache.put を直接呼べる
        return user
    }

    private fun fetchFromDb(id: String): User? {
        log("Fetching user $id from database")
        return User(id, "田中太郎", "tanaka@example.com")
    }
}

data class User(val id: String, val name: String, val email: String)

// 使用例
val service = UserService()
val user = service.findUser("user-1")

// UserService は Logger と Cache の両方のインターフェースを満たす
val logger: Logger = service
val cache: Cache<String, User> = service
```

```kotlin
// Kotlin: 拡張関数によるミックスイン的パターン

// インターフェース + 拡張関数で横断的な機能を追加
interface HasTimestamp {
    val createdAt: Long
    val updatedAt: Long
}

// 拡張関数: HasTimestamp を実装する全ての型に適用
fun HasTimestamp.formatCreatedAt(): String {
    val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    return sdf.format(java.util.Date(createdAt))
}

fun HasTimestamp.isOlderThan(days: Int): Boolean {
    val threshold = System.currentTimeMillis() - days * 86400000L
    return createdAt < threshold
}

interface HasName {
    val name: String
}

fun HasName.initials(): String {
    return name.split(" ")
        .mapNotNull { it.firstOrNull()?.uppercase() }
        .joinToString("")
}

// データクラスで複数のインターフェースを実装
data class Article(
    val id: String,
    override val name: String,
    val content: String,
    override val createdAt: Long = System.currentTimeMillis(),
    override val updatedAt: Long = System.currentTimeMillis(),
) : HasTimestamp, HasName

// 使用例
val article = Article("1", "Kotlin入門", "Kotlinの基礎を学びます")
println(article.formatCreatedAt())  // HasTimestamp 拡張
println(article.isOlderThan(30))    // HasTimestamp 拡張
println(article.initials())          // HasName 拡張 → "K"
```

---

## 8. PHP: トレイト

```php
<?php
// PHP: trait によるミックスイン

trait Timestampable {
    private DateTime $createdAt;
    private DateTime $updatedAt;

    public function initTimestamps(): void {
        $this->createdAt = new DateTime();
        $this->updatedAt = new DateTime();
    }

    public function getCreatedAt(): DateTime {
        return $this->createdAt;
    }

    public function getUpdatedAt(): DateTime {
        return $this->updatedAt;
    }

    public function touch(): void {
        $this->updatedAt = new DateTime();
    }
}

trait SoftDeletable {
    private ?DateTime $deletedAt = null;

    public function softDelete(): void {
        $this->deletedAt = new DateTime();
    }

    public function restore(): void {
        $this->deletedAt = null;
    }

    public function isDeleted(): bool {
        return $this->deletedAt !== null;
    }
}

trait HasSlug {
    private string $slug;

    public function generateSlug(string $title): void {
        $this->slug = strtolower(
            preg_replace('/[^a-zA-Z0-9]+/', '-', $title)
        );
    }

    public function getSlug(): string {
        return $this->slug;
    }
}

// トレイトの合成
class Article {
    use Timestampable;
    use SoftDeletable;
    use HasSlug;

    public function __construct(
        private string $title,
        private string $body,
    ) {
        $this->initTimestamps();
        $this->generateSlug($title);
    }
}

// トレイトの衝突解決
trait A {
    public function hello(): string {
        return "Hello from A";
    }
}

trait B {
    public function hello(): string {
        return "Hello from B";
    }
}

class C {
    use A, B {
        A::hello insteadof B;  // A の hello を優先
        B::hello as helloB;    // B の hello を別名で使用
    }
}

$c = new C();
echo $c->hello();    // "Hello from A"
echo $c->helloB();   // "Hello from B"

// トレイトの要求（abstract メソッド）
trait Loggable {
    abstract protected function getLogPrefix(): string;

    public function log(string $message): void {
        echo "[{$this->getLogPrefix()}] $message\n";
    }
}

class UserService {
    use Loggable;

    protected function getLogPrefix(): string {
        return 'UserService';
    }
}
```

---

## 9. Scala: トレイトの線形化

```scala
// Scala: トレイトのスタック可能な修正（Stackable Modification）

trait IntQueue {
  def get(): Int
  def put(x: Int): Unit
}

class BasicIntQueue extends IntQueue {
  import scala.collection.mutable.ArrayBuffer
  private val buf = new ArrayBuffer[Int]

  def get(): Int = buf.remove(0)
  def put(x: Int): Unit = buf += x
}

// スタック可能な修正: 各トレイトが振る舞いを追加
trait Doubling extends IntQueue {
  abstract override def put(x: Int): Unit = super.put(2 * x)
}

trait Incrementing extends IntQueue {
  abstract override def put(x: Int): Unit = super.put(x + 1)
}

trait Filtering extends IntQueue {
  abstract override def put(x: Int): Unit = {
    if (x >= 0) super.put(x)
    // 負の数は無視
  }
}

// トレイトの合成: 右から左の順で適用
val queue1 = new BasicIntQueue with Incrementing with Filtering
queue1.put(-1)  // Filtering: x >= 0 → 無視
queue1.put(0)   // Filtering: x >= 0 → Incrementing: put(0 + 1) → put(1)
queue1.put(1)   // Filtering: x >= 0 → Incrementing: put(1 + 1) → put(2)
// queue1 contains: [1, 2]

val queue2 = new BasicIntQueue with Filtering with Incrementing
queue2.put(-1)  // Incrementing: put(-1 + 1) → Filtering: 0 >= 0 → put(0)
queue2.put(0)   // Incrementing: put(0 + 1) → Filtering: 1 >= 0 → put(1)
// queue2 contains: [0, 1]

// 線形化の順序が結果を変える！
// with Incrementing with Filtering: 先にFiltering → Incrementing
// with Filtering with Incrementing: 先にIncrementing → Filtering


// Scala: Self-type による依存関係の宣言
trait UserRepository {
  def findUser(id: String): Option[User]
  def saveUser(user: User): Unit
}

trait EmailService {
  def sendEmail(to: String, subject: String, body: String): Unit
}

// Self-type: UserRepository と EmailService が必要
trait UserRegistration {
  self: UserRepository with EmailService =>

  def register(name: String, email: String): User = {
    val user = User(java.util.UUID.randomUUID().toString, name, email)
    saveUser(user)  // UserRepository のメソッド
    sendEmail(email, "Welcome!", s"Welcome, $name!")  // EmailService のメソッド
    user
  }
}

// 実装時にすべての依存を満たす必要がある
class ProductionApp
    extends UserRegistration
    with UserRepository
    with EmailService {

  private var users = Map[String, User]()

  override def findUser(id: String): Option[User] = users.get(id)
  override def saveUser(user: User): Unit = {
    users += (user.id -> user)
  }
  override def sendEmail(to: String, subject: String, body: String): Unit = {
    println(s"Sending email to $to: $subject")
  }
}

case class User(id: String, name: String, email: String)
```

---

## 10. Swift: プロトコル拡張によるミックスイン

```swift
// Swift: Protocol Extension によるミックスイン的パターン

protocol Identifiable {
    var id: String { get }
}

protocol Timestamped {
    var createdAt: Date { get }
    var updatedAt: Date { get set }
}

// Protocol Extension でデフォルト実装を提供
extension Timestamped {
    mutating func touch() {
        updatedAt = Date()
    }

    var age: TimeInterval {
        return Date().timeIntervalSince(createdAt)
    }

    var isRecent: Bool {
        return age < 86400  // 24時間以内
    }
}

protocol Validatable {
    var validationErrors: [String] { get }
}

extension Validatable {
    var isValid: Bool {
        return validationErrors.isEmpty
    }

    func validateOrThrow() throws {
        guard isValid else {
            throw ValidationError.invalid(validationErrors)
        }
    }
}

enum ValidationError: Error {
    case invalid([String])
}

protocol JSONConvertible: Codable {
    // Codable を要求するだけ
}

extension JSONConvertible {
    func toJSON() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(self)
        return String(data: data, encoding: .utf8)!
    }

    static func fromJSON(_ json: String) throws -> Self {
        let decoder = JSONDecoder()
        let data = json.data(using: .utf8)!
        return try decoder.decode(Self.self, from: data)
    }
}

// プロトコルの合成
struct Article: Identifiable, Timestamped, Validatable, JSONConvertible, Codable {
    let id: String
    var title: String
    var body: String
    let createdAt: Date
    var updatedAt: Date

    var validationErrors: [String] {
        var errors: [String] = []
        if title.isEmpty { errors.append("Title is required") }
        if body.count < 10 { errors.append("Body must be at least 10 characters") }
        return errors
    }
}

// 使用例
var article = Article(
    id: "1",
    title: "Swift入門",
    body: "Swiftの基礎を学びます。プロトコルは強力です。",
    createdAt: Date(),
    updatedAt: Date()
)

article.touch()                    // Timestamped
print(article.isRecent)            // Timestamped extension → true
print(article.isValid)             // Validatable extension → true
let json = try article.toJSON()    // JSONConvertible extension
print(json)
```

---

## 11. ミックスイン設計の原則とアンチパターン

### 11.1 設計原則

```
ミックスインの設計原則:

  1. 単一責任の原則（SRP）
     → 各ミックスインは1つの横断的関心事のみを扱う
     → ❌ LoggingAndCachingMixin → ✅ LoggingMixin + CachingMixin

  2. ステートレス優先
     → 状態を持たないミックスインは副作用が少なく安全
     → 状態を持つ場合は、初期化の順序に注意
     → 命名規則: 状態を持つ場合は明示的に（StatefulXxxMixin）

  3. 明示的な依存関係
     → 暗黙の依存（他のミックスインのメソッドを前提）は避ける
     → 必要なら抽象メソッドで要求を明示する
     → Python: Protocol, Ruby: abstract method, Rust: trait bounds

  4. 浅い継承チェーン
     → ミックスインの継承は1段まで（ミックスインがミックスインを継承しない）
     → 合成で解決できないかを先に検討

  5. 命名規則
     → Python: XxxMixin サフィックス
     → Ruby: 形容詞的な名前（Serializable, Loggable, Cacheable）
     → TypeScript: ミックスイン関数は動詞的（withTimestamp, makeLoggable）
     → PHP: XxxTrait サフィックス（またはXxxable）
```

### 11.2 アンチパターン

```
アンチパターン 1: God Mixin
  問題: 1つのミックスインに大量の機能を詰め込む
  症状:
    → 100行以上のミックスイン
    → 無関係な複数の責務
    → 部分的にしか使わないクラスが多い
  解決: 責務ごとに分割

アンチパターン 2: 暗黙の依存
  問題: ミックスインが他のミックスインや特定のフィールドの存在を前提とする
  症状:
    → self.name のようなフィールドへのアクセスが型チェックされない
    → ミックスインの順序を変えると壊れる
  解決: 抽象メソッドで依存を明示、またはプロトコルで型チェック

アンチパターン 3: ミックスインの乱用
  問題: あらゆる機能をミックスインで追加し、クラスの本質が見えなくなる
  症状:
    → class User(A, B, C, D, E, F, G, H, I, J): ...
    → メソッドの出所が追跡困難
    → IDE がメソッドの型を推論できない
  解決: 5個以下に制限、コンポジションを検討

アンチパターン 4: 状態の衝突
  問題: 複数のミックスインが同名のフィールドを追加する
  症状:
    → self._cache が2つのミックスインで別の意味
    → 初期化の順序で結果が変わる
  解決: プレフィックスで名前空間を分ける（_logging_cache, _http_cache）

アンチパターン 5: ダイヤモンドミックスイン
  問題: 複数のミックスインが共通のベースミックスインを継承
  症状:
    → MRO が予期しない順序になる
    → super() チェーンが複雑化
  解決: ミックスインの継承を避け、フラットな構造にする
```

```python
# アンチパターンの具体例と修正

# ❌ アンチパターン: 暗黙の依存
class BadLoggingMixin:
    def log(self, message: str) -> None:
        # self.name の存在を暗黙的に仮定している
        print(f"[{self.name}] {message}")  # type: ignore


# ✅ 修正: プロトコルで依存を明示
from typing import Protocol


class HasName(Protocol):
    name: str


class GoodLoggingMixin:
    """HasName を実装するクラスで使用"""
    def log(self: "HasName", message: str) -> None:
        print(f"[{self.name}] {message}")

    # または抽象メソッドで要求を明示
    # def get_log_prefix(self) -> str:
    #     raise NotImplementedError


# ❌ アンチパターン: 状態の衝突
class CacheMixin1:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache = {}  # HTTP キャッシュ

class CacheMixin2:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache = {}  # 計算結果キャッシュ（衝突!）


# ✅ 修正: 名前空間で分離
class HttpCacheMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._http_cache: dict[str, bytes] = {}

class ComputeCacheMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compute_cache: dict[str, any] = {}


# ❌ アンチパターン: ミックスインの乱用
class OverMixedUser(
    JsonMixin, XmlMixin, CsvMixin,       # シリアライズ3つも要らない
    LoggingMixin, TracingMixin,           # ログとトレース両方？
    CacheMixin, MemcacheMixin, RedisMixin,  # キャッシュ3つ？
    ValidatableMixin, SanitizableMixin,
):
    pass  # クラスの本質が完全に見えない


# ✅ 修正: 必要最小限のミックスイン + コンポジション
class CleanUser(JsonMixin, LoggableMixin, ValidatableMixin):
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        # キャッシュはコンポジションで
        self._cache = CacheService()
```

---

## 12. テスト戦略

### 12.1 ミックスインの単体テスト

```python
# ミックスインの単体テスト戦略

import pytest
from datetime import datetime


# テスト対象のミックスイン
class SerializableMixin:
    def to_dict(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }


class ValidatableMixin:
    def validate(self) -> list[str]:
        errors = []
        for attr in dir(self):
            if attr.startswith("validate_"):
                error = getattr(self, attr)()
                if error:
                    errors.append(error)
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


# テスト用のホストクラス（ミックスインをテストするための最小クラス）
class SerializableHost(SerializableMixin):
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value
        self._internal = "should not be serialized"


class ValidatableHost(ValidatableMixin):
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def validate_name(self) -> str | None:
        if not self.name:
            return "Name is required"
        return None

    def validate_age(self) -> str | None:
        if self.age < 0:
            return "Age must be non-negative"
        return None


# テスト
class TestSerializableMixin:
    def test_to_dict_includes_public_attrs(self):
        obj = SerializableHost("test", 42)
        result = obj.to_dict()
        assert result == {"name": "test", "value": 42}

    def test_to_dict_excludes_private_attrs(self):
        obj = SerializableHost("test", 42)
        result = obj.to_dict()
        assert "_internal" not in result

    def test_to_dict_with_empty_object(self):
        class EmptyHost(SerializableMixin):
            pass
        obj = EmptyHost()
        assert obj.to_dict() == {}


class TestValidatableMixin:
    def test_valid_object(self):
        obj = ValidatableHost("田中", 30)
        assert obj.is_valid()
        assert obj.validate() == []

    def test_invalid_name(self):
        obj = ValidatableHost("", 30)
        assert not obj.is_valid()
        errors = obj.validate()
        assert "Name is required" in errors

    def test_invalid_age(self):
        obj = ValidatableHost("田中", -1)
        assert not obj.is_valid()
        errors = obj.validate()
        assert "Age must be non-negative" in errors

    def test_multiple_errors(self):
        obj = ValidatableHost("", -1)
        errors = obj.validate()
        assert len(errors) == 2


# 複数ミックスインの結合テスト
class TestMixinComposition:
    def test_serializable_and_validatable(self):
        class ComposedHost(SerializableMixin, ValidatableMixin):
            def __init__(self, name: str):
                self.name = name

            def validate_name(self) -> str | None:
                if not self.name:
                    return "Name is required"
                return None

        obj = ComposedHost("test")
        assert obj.is_valid()
        assert obj.to_dict() == {"name": "test"}
```

### 12.2 MRO のテスト

```python
# MRO の順序が正しいことをテストする

class TestMROOrder:
    def test_diamond_mro(self):
        """ダイヤモンド継承のMRO順序を検証"""
        class A:
            def method(self): return "A"

        class B(A):
            def method(self): return "B"

        class C(A):
            def method(self): return "C"

        class D(B, C):
            pass

        # MROの順序を検証
        assert D.__mro__ == (D, B, C, A, object)
        # 最初に見つかるのは B
        assert D().method() == "B"

    def test_cooperative_super_chain(self):
        """協調的super()チェーンの順序を検証"""
        call_order = []

        class Base:
            def process(self):
                call_order.append("Base")

        class MixinA(Base):
            def process(self):
                call_order.append("MixinA")
                super().process()

        class MixinB(Base):
            def process(self):
                call_order.append("MixinB")
                super().process()

        class Combined(MixinA, MixinB):
            def process(self):
                call_order.append("Combined")
                super().process()

        Combined().process()
        assert call_order == ["Combined", "MixinA", "MixinB", "Base"]

    def test_mixin_initialization_order(self):
        """ミックスインの初期化順序を検証"""
        init_order = []

        class Base:
            def __init__(self, **kwargs):
                init_order.append("Base")

        class MixinA(Base):
            def __init__(self, **kwargs):
                init_order.append("MixinA")
                super().__init__(**kwargs)

        class MixinB(Base):
            def __init__(self, **kwargs):
                init_order.append("MixinB")
                super().__init__(**kwargs)

        class Combined(MixinA, MixinB):
            def __init__(self):
                init_order.append("Combined")
                super().__init__()

        Combined()
        assert init_order == ["Combined", "MixinA", "MixinB", "Base"]
```

```typescript
// TypeScript: ミックスインのテスト

describe("Timestamped mixin", () => {
  type Constructor<T = {}> = new (...args: any[]) => T;

  function Timestamped<TBase extends Constructor>(Base: TBase) {
    return class extends Base {
      createdAt = new Date();
      updatedAt = new Date();
      touch() { this.updatedAt = new Date(); }
    };
  }

  class BaseEntity {
    constructor(public id: string) {}
  }

  const TimestampedEntity = Timestamped(BaseEntity);

  it("should add createdAt and updatedAt", () => {
    const entity = new TimestampedEntity("1");
    expect(entity.createdAt).toBeInstanceOf(Date);
    expect(entity.updatedAt).toBeInstanceOf(Date);
  });

  it("should update updatedAt on touch", () => {
    const entity = new TimestampedEntity("1");
    const original = entity.updatedAt;

    // 少し待ってから touch
    jest.advanceTimersByTime(1000);
    entity.touch();

    expect(entity.updatedAt.getTime()).toBeGreaterThanOrEqual(
      original.getTime()
    );
  });

  it("should preserve base class properties", () => {
    const entity = new TimestampedEntity("test-id");
    expect(entity.id).toBe("test-id");
  });

  it("should work with multiple mixins", () => {
    function Activatable<TBase extends Constructor>(Base: TBase) {
      return class extends Base {
        isActive = false;
        activate() { this.isActive = true; }
      };
    }

    const FullEntity = Activatable(Timestamped(BaseEntity));
    const entity = new FullEntity("1");

    // 両方のミックスインが機能する
    expect(entity.createdAt).toBeInstanceOf(Date);
    expect(entity.isActive).toBe(false);
    entity.activate();
    expect(entity.isActive).toBe(true);
  });
});
```

---

## 13. 言語横断比較と選択指針

```
ミックスイン / 多重継承の比較まとめ:

  Python:
    手法: 多重継承 + MRO + super()
    利点: 柔軟、協調的多重継承が可能
    欠点: MRO の理解が必要、実行時エラーのリスク
    推奨: ミックスインは5個以下、XxxMixin 命名規則

  Ruby:
    手法: Module の include / extend / prepend
    利点: 自然な構文、衝突時の柔軟な解決
    欠点: 型チェックがない、prepend の挙動が直感的でない場合
    推奨: Module は1つの責務に、respond_to? で安全にアクセス

  TypeScript:
    手法: クラス式ミックスイン / デコレータ
    利点: 型安全、合成の柔軟さ
    欠点: 型の推論が複雑になることがある
    推奨: 制約付きミックスイン（GConstructor パターン）

  Rust:
    手法: トレイト + ブランケット実装
    利点: コンパイル時の安全性、ゼロコスト抽象
    欠点: 柔軟性は低い（動的な合成は限定的）
    推奨: トレイト境界で制約を明示、Derive で自動実装

  Java:
    手法: インターフェースのデフォルトメソッド
    利点: 後方互換、型安全
    欠点: 状態を持てない、衝突解決が煩雑
    推奨: デフォルトメソッドはユーティリティ的な使い方に限定

  Kotlin:
    手法: by キーワードによるデリゲーション
    利点: コンポジションの簡潔な構文
    欠点: 委譲先のフィールドが必要
    推奨: 委譲 + 拡張関数の組み合わせ

  Scala:
    手法: トレイトの線形化 + Self-type
    利点: 状態を持てる、スタック可能な修正パターン
    欠点: 線形化の順序が直感的でない場合がある
    推奨: Self-type で依存を明示、ケーキパターンはDIコンテナで代替可能

  PHP:
    手法: trait + insteadof / as
    利点: 明示的な衝突解決、シンプル
    欠点: 型チェックが弱い
    推奨: abstract メソッドで要求を明示

  Swift:
    手法: Protocol Extension
    利点: 型安全、Value Type対応、条件付き適合
    欠点: 状態を持てない（Protocol Extension自体は）
    推奨: Protocol合成 + Extension でデフォルト実装
```

```
選択判断フローチャート:

  Q1: 「コードの再利用が必要か？」
  │
  ├── No → 通常のクラス設計で十分
  │
  └── Yes
      │
      Q2: 「再利用したい機能は横断的関心事か？」
      │    （ログ、キャッシュ、認証、バリデーション等）
      │
      ├── Yes → ミックスイン / トレイト
      │         Q2a: 「状態を持つ必要があるか？」
      │         ├── Yes → Python Mixin, Scala Trait, PHP Trait
      │         └── No → インターフェース + デフォルト実装
      │
      └── No
          │
          Q3: 「is-a 関係が成り立つか？」
          │
          ├── Yes → 継承
          └── No → コンポジション（委譲）
```

---

## まとめ

| 言語 | 手法 | 特徴 |
|------|------|------|
| Python | 多重継承 + MRO | C3線形化で順序解決 |
| Ruby | Module include | 最も自然なミックスイン |
| TypeScript | クラス式合成 | 型安全なミックスイン |
| Rust | トレイト | 多重継承なし。トレイトで合成 |
| Java | デフォルトメソッド | インターフェースに実装を追加 |
| Kotlin | by デリゲーション | コンポジションの簡潔な構文 |
| Scala | トレイト線形化 | スタック可能な修正パターン |
| PHP | trait | 明示的な衝突解決（insteadof/as） |
| Swift | Protocol Extension | 型安全なプロトコル指向 |

```
実践的な指針:

  1. ミックスインは5個以下
     → それ以上必要なら設計を見直す
     → クラスの本質的な責務が見えなくなる

  2. 各ミックスインは単一の責任を持つ
     → LoggingMixin + CachingMixin（個別） ✅
     → LoggingAndCachingMixin（混合） ❌

  3. 状態を持つミックスインは最小限に
     → ステートレスなミックスインは安全で予測可能
     → 状態が必要ならコンポジションを検討

  4. 依存関係を明示する
     → 暗黙の依存は避ける
     → 抽象メソッドやプロトコルで要求を明確に

  5. テスタビリティを確保する
     → 各ミックスインは独立してテスト可能に
     → テスト用のホストクラスを用意
     → MRO や合成順序のテストも忘れずに

  6. 命名規則を統一する
     → Python: XxxMixin
     → Ruby: -able サフィックス
     → TypeScript: with/make プレフィックス
     → PHP: XxxTrait or -able
```

---

## 次に読むべきガイド
→ [[03-generics-in-oop.md]] — OOPにおけるジェネリクス

---

## 参考文献
1. Barrett, S. "C3 Linearization." 1996.
2. Bracha, G. "The Programming Language Jigsaw." 1992.
3. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
4. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018.
5. Odersky, M. "Programming in Scala." Artima, 2021.
6. The Rust Programming Language. "Traits." doc.rust-lang.org.
7. Python Documentation. "Multiple Inheritance." docs.python.org.
8. Ruby Documentation. "Modules." ruby-doc.org.
