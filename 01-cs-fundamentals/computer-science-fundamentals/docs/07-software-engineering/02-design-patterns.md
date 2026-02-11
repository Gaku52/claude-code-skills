# デザインパターン

> デザインパターンは「先人の知恵の結晶」であり、共通の問題に対する再利用可能な解決策である。

## この章で学ぶこと

- [ ] GoFパターンの主要なものを説明できる
- [ ] パターンの適用場面を判断できる
- [ ] アンチパターンを認識できる

---

## 1. 生成パターン

```python
# Singleton: インスタンスを1つに制限
class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Factory Method: 生成を抽象化
class NotificationFactory:
    @staticmethod
    def create(type):
        if type == "email": return EmailNotification()
        if type == "sms": return SMSNotification()
        raise ValueError(f"Unknown type: {type}")

# Builder: 複雑なオブジェクトを段階的に構築
class QueryBuilder:
    def __init__(self):
        self._select = "*"
        self._where = []

    def select(self, fields):
        self._select = fields
        return self  # メソッドチェーン

    def where(self, condition):
        self._where.append(condition)
        return self

    def build(self):
        sql = f"SELECT {self._select} FROM table"
        if self._where:
            sql += " WHERE " + " AND ".join(self._where)
        return sql

query = QueryBuilder().select("name, age").where("age > 20").build()
```

---

## 2. 構造パターン

```python
# Adapter: インターフェースを変換
class OldAPI:
    def get_data_xml(self):
        return "<data>hello</data>"

class APIAdapter:
    def __init__(self, old_api):
        self.old_api = old_api

    def get_data_json(self):
        xml = self.old_api.get_data_xml()
        return {"data": "hello"}  # XML→JSON変換

# Observer: 状態変化を通知
class EventEmitter:
    def __init__(self):
        self._listeners = {}

    def on(self, event, callback):
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event, data=None):
        for callback in self._listeners.get(event, []):
            callback(data)

# Strategy: アルゴリズムを交換可能に
class Sorter:
    def __init__(self, strategy):
        self.strategy = strategy

    def sort(self, data):
        return self.strategy(data)

sorter = Sorter(strategy=sorted)  # 戦略を注入
```

---

## 3. 現代のパターン

```
GoFを超える現代のパターン:

  1. Repository パターン: データアクセスを抽象化
  2. CQRS: コマンド（書込）とクエリ（読取）を分離
  3. Event Sourcing: 状態変化をイベントとして記録
  4. Circuit Breaker: 障害の連鎖を防止
  5. Saga パターン: 分散トランザクションの管理

  アンチパターン:
  - God Object: 1つのクラスが全てを担当
  - Premature Optimization: 早すぎる最適化
  - Golden Hammer: 使い慣れたツールを全てに適用
  - Cargo Cult: 理由を理解せずにパターンを適用
```

---

## まとめ

| カテゴリ | 代表パターン | 用途 |
|---------|------------|------|
| 生成 | Singleton, Factory, Builder | オブジェクト生成の制御 |
| 構造 | Adapter, Decorator, Proxy | クラス/オブジェクトの構成 |
| 振舞 | Observer, Strategy, Command | オブジェクト間の通信 |

---

## 次に読むべきガイド
→ [[03-clean-code.md]] — クリーンコード

---

## 参考文献
1. Gamma, E. et al. "Design Patterns (GoF)." Addison-Wesley, 1994.
2. Freeman, E. et al. "Head First Design Patterns." O'Reilly, 2020.
