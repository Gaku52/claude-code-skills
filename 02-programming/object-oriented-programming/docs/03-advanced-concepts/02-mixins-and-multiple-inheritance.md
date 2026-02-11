# ミックスインと多重継承

> 多重継承の問題を回避しつつ、複数の振る舞いを組み合わせるための手法。Python のMRO、Ruby のモジュール、TypeScript のミックスインパターンを比較する。

## この章で学ぶこと

- [ ] 多重継承の問題（ダイヤモンド問題）とその解決策を理解する
- [ ] ミックスインパターンの実装方法を把握する
- [ ] 各言語でのアプローチの違いを学ぶ

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

---

## 2. Python: MRO（Method Resolution Order）

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

### Python: ミックスインパターン

```python
# ミックスイン = 単独では使わない、機能を追加するためのクラス
class JsonMixin:
    """JSON変換機能を追加"""
    def to_json(self) -> str:
        import json
        return json.dumps(self.__dict__, default=str)

    @classmethod
    def from_json(cls, json_str: str):
        import json
        data = json.loads(json_str)
        return cls(**data)

class TimestampMixin:
    """タイムスタンプ機能を追加"""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            from datetime import datetime
            original_init(self, *args, **kwargs)
            self.created_at = datetime.now()
            self.updated_at = datetime.now()

        cls.__init__ = new_init

class LoggableMixin:
    """ログ出力機能を追加"""
    def log(self, message: str) -> None:
        print(f"[{type(self).__name__}] {message}")

# ミックスインを組み合わせ
class User(JsonMixin, TimestampMixin, LoggableMixin):
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

user = User("田中", "tanaka@example.com")
print(user.to_json())       # JsonMixin
print(user.created_at)      # TimestampMixin
user.log("ログインしました") # LoggableMixin
```

---

## 3. Ruby: モジュール

```ruby
# Ruby: Module によるミックスイン
module Serializable
  def to_json
    require 'json'
    JSON.generate(instance_variables.each_with_object({}) { |var, hash|
      hash[var.to_s.delete('@')] = instance_variable_get(var)
    })
  end
end

module Loggable
  def log(message)
    puts "[#{self.class.name}] #{message}"
  end
end

module Cacheable
  def cache_key
    "#{self.class.name}:#{object_id}"
  end
end

class User
  include Serializable  # ミックスイン
  include Loggable
  include Cacheable

  attr_accessor :name, :email

  def initialize(name, email)
    @name = name
    @email = email
  end
end

user = User.new("田中", "tanaka@example.com")
puts user.to_json       # Serializable
user.log("ログイン")     # Loggable
puts user.cache_key     # Cacheable
```

---

## 4. TypeScript: ミックスインパターン

```typescript
// TypeScript: ミックスイン（クラス式を使ったパターン）
type Constructor<T = {}> = new (...args: any[]) => T;

// ミックスイン関数
function Timestamped<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    createdAt = new Date();
    updatedAt = new Date();

    touch() {
      this.updatedAt = new Date();
    }
  };
}

function Activatable<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    isActive = false;

    activate() { this.isActive = true; }
    deactivate() { this.isActive = false; }
  };
}

function Taggable<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
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

---

## 5. 注意点

```
ミックスインの注意点:
  1. 名前衝突: 複数のミックスインが同名メソッドを持つ場合
     → Python: MRO で最初に見つかったものが使われる
     → 明示的に super() を呼んで協調的に動作させる

  2. 状態の管理: ミックスインがフィールドを追加する場合
     → 初期化の順序に注意
     → コンストラクタの呼び出しチェーン

  3. 可読性: ミックスインが多すぎると
     → メソッドの出所が不明
     → IDE のサポートが重要

  4. テスト: 各ミックスインは独立してテスト可能に設計

推奨:
  → ミックスインは5個以下に
  → 各ミックスインは単一の責任を持つ
  → 状態を持つミックスインは最小限に
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

---

## 次に読むべきガイド
→ [[03-generics-in-oop.md]] — OOPにおけるジェネリクス

---

## 参考文献
1. Barrett, S. "C3 Linearization." 1996.
2. Bracha, G. "The Programming Language Jigsaw." 1992.
