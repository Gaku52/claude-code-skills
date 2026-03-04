# デザインパターン

> デザインパターンは「先人の知恵の結晶」であり、共通の問題に対する再利用可能な解決策である。
> パターンを知ることで、設計の語彙が増え、チームのコミュニケーションが円滑になる。

## この章で学ぶこと

- [ ] GoF パターン 23 種の分類と代表的パターンの意図を説明できる
- [ ] 生成・構造・振る舞いの各カテゴリからパターンを選択し実装できる
- [ ] アーキテクチャパターン（MVC / MVVM / Repository）を比較できる
- [ ] アンチパターンを認識し、回避策を提示できる
- [ ] 実際のコードベースでパターンの適用判断ができる

---

## 1. デザインパターンの意義

### 1.1 デザインパターンとは何か

デザインパターンとは、ソフトウェア設計において繰り返し現れる問題と、
その問題に対する汎用的な解決策を体系化したものである。
1994 年に Erich Gamma、Richard Helm、Ralph Johnson、John Vlissides の
4 名（通称 Gang of Four、略称 GoF）が著書
"Design Patterns: Elements of Reusable Object-Oriented Software" で
23 のパターンを整理したことが出発点となった。

デザインパターンは「車輪の再発明」を防ぎ、検証済みの設計を再利用するための道具である。
ただし、パターンはそのままコピーするものではなく、
問題の文脈に合わせて適用するテンプレートとして理解すべきである。

### 1.2 パターンを学ぶ 3 つの利点

```
┌──────────────────────────────────────────────────────────────┐
│              デザインパターンを学ぶ 3 つの利点                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 共通語彙の獲得                                           │
│     「ここは Observer で実装しよう」と言えば                  │
│     チーム全員が設計意図を即座に理解できる                    │
│                                                              │
│  2. 設計判断の高速化                                         │
│     問題を見た瞬間に適切な構造が浮かぶようになる              │
│     ゼロから考える時間を大幅に短縮できる                      │
│                                                              │
│  3. 保守性・拡張性の向上                                     │
│     パターンに沿った設計は変更に強い                          │
│     SOLID 原則と自然に整合する                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 GoF パターンの分類

GoF の 23 パターンは、目的に応じて 3 つのカテゴリに分類される。

| カテゴリ | 目的 | パターン数 | 代表例 |
|----------|------|-----------|--------|
| 生成（Creational） | オブジェクト生成の仕組みを柔軟にする | 5 | Singleton, Factory Method, Abstract Factory, Builder, Prototype |
| 構造（Structural） | クラスやオブジェクトの構成を整理する | 7 | Adapter, Bridge, Composite, Decorator, Facade, Flyweight, Proxy |
| 振る舞い（Behavioral） | オブジェクト間の責務分担と通信を整理する | 11 | Observer, Strategy, Command, Iterator, State, Template Method, Visitor, Chain of Responsibility, Mediator, Memento, Interpreter |

```
┌─────────────────────────────────────────────────────────────┐
│                    GoF 23 パターン 全体図                    │
├───────────────┬────────────────┬────────────────────────────┤
│   生成 (5)    │   構造 (7)     │      振る舞い (11)         │
├───────────────┼────────────────┼────────────────────────────┤
│ Singleton     │ Adapter        │ Chain of Responsibility    │
│ Factory Method│ Bridge         │ Command                    │
│ Abstract Fctry│ Composite      │ Interpreter                │
│ Builder       │ Decorator      │ Iterator                   │
│ Prototype     │ Facade         │ Mediator                   │
│               │ Flyweight      │ Memento                    │
│               │ Proxy          │ Observer                   │
│               │                │ State                      │
│               │                │ Strategy                   │
│               │                │ Template Method            │
│               │                │ Visitor                    │
└───────────────┴────────────────┴────────────────────────────┘
```

### 1.4 パターンの構成要素

各パターンは以下の 4 つの要素で記述される。

1. **パターン名（Name）**: 設計の語彙となる名前
2. **問題（Problem）**: どのような状況で使うのか
3. **解決策（Solution）**: 要素間の関係、責務、協調の記述
4. **結果（Consequences）**: パターン適用のトレードオフ

この章では、各パターンについてこの 4 要素を明示しながら解説を進める。

---

## 2. 生成パターン（Creational Patterns）

生成パターンは、オブジェクトの生成プロセスを抽象化し、
システムがどのようにオブジェクトを作成・構成・表現するかを柔軟にするパターン群である。
直接 `new`（Python では `ClassName()`）を呼び出す代わりに、
生成ロジックを分離することで、変更に強い設計を実現する。

### 2.1 Singleton パターン

#### 意図

あるクラスのインスタンスがシステム全体でただ 1 つであることを保証し、
そのインスタンスへのグローバルなアクセスポイントを提供する。

#### 問題

データベース接続プール、ログマネージャ、設定オブジェクトなど、
複数のインスタンスが存在すると不整合やリソースの浪費が発生する場合がある。
こうしたオブジェクトは、アプリケーション全体で 1 つだけ存在すべきである。

#### 解決策

```
┌─────────────────────────────────┐
│         Singleton               │
├─────────────────────────────────┤
│ - _instance: Singleton = None   │
├─────────────────────────────────┤
│ + __new__(cls): Singleton       │
│ + get_instance(): Singleton     │
│ + operation(): void             │
└─────────────────────────────────┘
         ▲
         │ 唯一のインスタンス
         │
    client code
```

#### Python 実装

```python
import threading


class DatabaseConnection:
    """スレッドセーフな Singleton パターンの実装例。

    __new__ メソッドをオーバーライドし、Lock を使って
    マルチスレッド環境でも安全にインスタンスを1つに制限する。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # ダブルチェックロッキング
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, host: str = "localhost", port: int = 5432):
        if self._initialized:
            return
        self.host = host
        self.port = port
        self.connection = None
        self._initialized = True

    def connect(self) -> str:
        self.connection = f"Connected to {self.host}:{self.port}"
        return self.connection

    def disconnect(self) -> None:
        self.connection = None


# 使用例
db1 = DatabaseConnection("db.example.com", 5432)
db2 = DatabaseConnection("other.host.com", 3306)

assert db1 is db2               # 同一インスタンス
assert db1.host == "db.example.com"  # 最初の初期化値が保持される

db1.connect()
print(db2.connection)  # "Connected to db.example.com:5432"
```

#### Python でのより実用的な Singleton: モジュールレベル変数

Python では、モジュール自体が Singleton として機能する。
そのため、以下のようにモジュールレベルで設定を管理するのが最も自然な方法である。

```python
# config.py — モジュール自体が Singleton
_settings: dict = {}


def load(path: str) -> None:
    """設定ファイルを読み込む。"""
    import json
    with open(path) as f:
        _settings.update(json.load(f))


def get(key: str, default=None):
    """設定値を取得する。"""
    return _settings.get(key, default)
```

#### 結果（トレードオフ）

| メリット | デメリット |
|---------|----------|
| インスタンスの一意性を保証 | グローバル状態を導入するためテストが困難になりやすい |
| 遅延初期化が可能 | マルチスレッドでの競合に注意が必要 |
| メモリ使用量の削減 | 依存関係が暗黙的になりやすい |

#### Singleton を避けるべき場合

Singleton は便利だが、濫用するとグローバル変数と同じ問題を引き起こす。
テスタビリティを重視する場合は、依存性注入（Dependency Injection）を優先し、
DI コンテナ側でライフタイムを「シングルトン」として管理する方が望ましい。

---

### 2.2 Factory Method パターン

#### 意図

オブジェクトの生成をサブクラスに委譲し、
どのクラスをインスタンス化するかを動的に決定できるようにする。

#### 問題

通知システムを例に考える。メール通知、SMS 通知、プッシュ通知など、
通知の種類が増えるたびに生成ロジックを変更するのは Open-Closed 原則に反する。
新しい通知タイプを追加するときに既存コードを修正せずに済む仕組みが必要である。

#### 解決策

```
┌──────────────────────┐
│   NotificationFactory │  (Creator)
│   <<abstract>>       │
├──────────────────────┤
│ + create(): Notif.   │ ← Factory Method
│ + send(msg): void    │
└──────┬───────────────┘
       │ 継承
  ┌────┴──────┬────────────────┐
  ▼           ▼                ▼
┌──────┐  ┌───────┐  ┌─────────────┐
│Email │  │ SMS   │  │ Push        │
│Fctry │  │ Fctry │  │ Fctry       │
└──┬───┘  └──┬────┘  └──┬──────────┘
   │ 生成    │ 生成     │ 生成
   ▼         ▼          ▼
┌──────┐  ┌───────┐  ┌─────────────┐
│Email │  │ SMS   │  │ Push        │
│Notif │  │ Notif │  │ Notification│
└──────┘  └───────┘  └─────────────┘
```

#### Python 実装

```python
from abc import ABC, abstractmethod


class Notification(ABC):
    """通知の基底クラス。"""

    @abstractmethod
    def send(self, message: str) -> str:
        """メッセージを送信する。"""
        ...


class EmailNotification(Notification):
    def __init__(self, recipient: str):
        self.recipient = recipient

    def send(self, message: str) -> str:
        return f"Email to {self.recipient}: {message}"


class SMSNotification(Notification):
    def __init__(self, phone_number: str):
        self.phone_number = phone_number

    def send(self, message: str) -> str:
        return f"SMS to {self.phone_number}: {message}"


class PushNotification(Notification):
    def __init__(self, device_token: str):
        self.device_token = device_token

    def send(self, message: str) -> str:
        return f"Push to {self.device_token}: {message}"


class NotificationFactory:
    """Factory Method パターンによる通知オブジェクト生成。

    辞書ベースのレジストリで拡張性を確保する。
    新しい通知タイプを追加するには register() を呼ぶだけでよい。
    """

    _creators: dict[str, type[Notification]] = {}

    @classmethod
    def register(cls, notification_type: str, creator: type[Notification]) -> None:
        """通知タイプを登録する。"""
        cls._creators[notification_type] = creator

    @classmethod
    def create(cls, notification_type: str, **kwargs) -> Notification:
        """登録済みの通知タイプからインスタンスを生成する。"""
        creator = cls._creators.get(notification_type)
        if creator is None:
            raise ValueError(
                f"Unknown notification type: {notification_type}. "
                f"Available: {list(cls._creators.keys())}"
            )
        return creator(**kwargs)


# タイプの登録
NotificationFactory.register("email", EmailNotification)
NotificationFactory.register("sms", SMSNotification)
NotificationFactory.register("push", PushNotification)

# 使用例
notif = NotificationFactory.create("email", recipient="user@example.com")
print(notif.send("Hello!"))  # "Email to user@example.com: Hello!"

notif2 = NotificationFactory.create("sms", phone_number="+81-90-1234-5678")
print(notif2.send("確認コード: 1234"))
```

#### 結果（トレードオフ）

| メリット | デメリット |
|---------|----------|
| 生成と利用の分離（疎結合） | クラス数が増加する |
| Open-Closed 原則に適合 | 単純なケースでは過剰設計になる |
| テスト時にモックへの差し替えが容易 | 間接層が増えて追跡が難しくなる |

---

### 2.3 Builder パターン

#### 意図

複雑なオブジェクトの構築過程を分離し、
同じ構築過程で異なる表現を生成できるようにする。

#### 問題

コンストラクタの引数が多いオブジェクト（例: HTTP リクエスト、SQL クエリ、
UI コンポーネント）は、引数の順序を間違えやすく可読性も低い。
また、オプション引数が多い場合、コンストラクタのオーバーロードが爆発的に増える
（Telescoping Constructor 問題）。

#### Python 実装

```python
from dataclasses import dataclass, field


@dataclass
class HttpRequest:
    """構築済みの HTTP リクエストオブジェクト。"""
    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: str | None = None
    timeout: int = 30
    retries: int = 0
    auth_token: str | None = None


class HttpRequestBuilder:
    """Builder パターンによる HTTP リクエスト構築。

    メソッドチェーンで段階的にリクエストを組み立て、
    build() で最終的な HttpRequest オブジェクトを返す。
    """

    def __init__(self):
        self._method = "GET"
        self._url = ""
        self._headers: dict[str, str] = {}
        self._body: str | None = None
        self._timeout = 30
        self._retries = 0
        self._auth_token: str | None = None

    def method(self, method: str) -> "HttpRequestBuilder":
        self._method = method.upper()
        return self

    def url(self, url: str) -> "HttpRequestBuilder":
        self._url = url
        return self

    def header(self, key: str, value: str) -> "HttpRequestBuilder":
        self._headers[key] = value
        return self

    def body(self, body: str) -> "HttpRequestBuilder":
        self._body = body
        return self

    def timeout(self, seconds: int) -> "HttpRequestBuilder":
        self._timeout = seconds
        return self

    def retries(self, count: int) -> "HttpRequestBuilder":
        self._retries = count
        return self

    def auth(self, token: str) -> "HttpRequestBuilder":
        self._auth_token = token
        return self

    def build(self) -> HttpRequest:
        if not self._url:
            raise ValueError("URL is required")
        if self._auth_token:
            self._headers["Authorization"] = f"Bearer {self._auth_token}"
        return HttpRequest(
            method=self._method,
            url=self._url,
            headers=self._headers,
            body=self._body,
            timeout=self._timeout,
            retries=self._retries,
            auth_token=self._auth_token,
        )


# 使用例: メソッドチェーンで読みやすく構築
request = (
    HttpRequestBuilder()
    .method("POST")
    .url("https://api.example.com/users")
    .header("Content-Type", "application/json")
    .body('{"name": "Alice", "age": 30}')
    .auth("my-secret-token")
    .timeout(10)
    .retries(3)
    .build()
)

print(request.method)   # "POST"
print(request.url)      # "https://api.example.com/users"
print(request.headers)  # {"Content-Type": "application/json", "Authorization": "Bearer my-secret-token"}
print(request.retries)  # 3


# 比較: Builder を使わない場合（可読性が低い）
# request = HttpRequest("POST", "https://api.example.com/users",
#     {"Content-Type": "application/json", "Authorization": "Bearer my-secret-token"},
#     '{"name": "Alice", "age": 30}', 10, 3, "my-secret-token")
```

#### 結果（トレードオフ）

| メリット | デメリット |
|---------|----------|
| 複雑なオブジェクトを段階的に構築できる | Builder クラス分のコード量が増える |
| メソッドチェーンで可読性が高い | 単純なオブジェクトには過剰 |
| バリデーションを build() に集約できる | イミュータブル設計との組合せに工夫が要る |

---

### 2.4 Prototype パターン

#### 意図

既存のオブジェクトをコピー（クローン）して新しいオブジェクトを生成する。
生成コストが高いオブジェクトや、設定が複雑なオブジェクトの複製に有効である。

#### 問題

ゲームで大量の敵キャラクターを生成する場合、毎回ゼロから構築するのは非効率である。
テンプレートとなるオブジェクトをコピーし、必要な部分だけ変更する方が効率的である。

#### Python 実装

```python
import copy
from dataclasses import dataclass, field


@dataclass
class GameCharacter:
    """ゲームキャラクターのプロトタイプ。"""
    name: str
    hp: int
    attack: int
    defense: int
    skills: list[str] = field(default_factory=list)
    position: dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})

    def clone(self) -> "GameCharacter":
        """ディープコピーで完全な複製を作成する。"""
        return copy.deepcopy(self)


# テンプレートの作成
goblin_template = GameCharacter(
    name="Goblin",
    hp=50,
    attack=10,
    defense=5,
    skills=["slash", "dodge"],
)

# プロトタイプからクローンを生成
goblin1 = goblin_template.clone()
goblin1.name = "Goblin A"
goblin1.position = {"x": 10, "y": 20}

goblin2 = goblin_template.clone()
goblin2.name = "Goblin B"
goblin2.position = {"x": 30, "y": 40}
goblin2.skills.append("poison")  # テンプレートには影響しない

print(goblin_template.skills)  # ["slash", "dodge"]
print(goblin2.skills)          # ["slash", "dodge", "poison"]
```

#### 結果（トレードオフ）

| メリット | デメリット |
|---------|----------|
| 複雑なオブジェクトの生成を簡略化 | ディープコピーのコストに注意 |
| 実行時に動的にプロトタイプを変更可能 | 循環参照のあるオブジェクトでは注意が必要 |
| サブクラス不要で多様なオブジェクトを生成 | コピー後の独立性保証が必要 |

### 2.5 生成パターンの比較表

| パターン | 主な目的 | 適用場面 | Python での典型実装 |
|---------|---------|---------|-------------------|
| Singleton | インスタンスを 1 つに制限 | DB 接続、設定、ログ | `__new__` / モジュール変数 |
| Factory Method | 生成を委譲・抽象化 | 通知、ドキュメント、UI部品 | レジストリ辞書 + `create()` |
| Abstract Factory | 関連オブジェクト群を一括生成 | GUI テーマ、DB ドライバ群 | 抽象基底クラス群 |
| Builder | 複雑なオブジェクトを段階構築 | HTTP リクエスト、SQL、設定 | メソッドチェーン + `build()` |
| Prototype | 既存オブジェクトをコピーして生成 | ゲームキャラ、設定テンプレート | `copy.deepcopy()` |

---

## 3. 構造パターン（Structural Patterns）

構造パターンは、クラスやオブジェクトを組み合わせてより大きな構造を形成する方法を扱う。
インターフェースの不一致を解消したり、新しい機能を動的に追加したり、
複雑なサブシステムを単純なインターフェースで包んだりする。

### 3.1 Adapter パターン

#### 意図

既存クラスのインターフェースを、クライアントが期待する別のインターフェースに変換する。
互換性のないインターフェース同士を協調させるための「変換器」である。

#### 問題

レガシーシステムの API は XML を返すが、新しいシステムは JSON を期待している。
レガシーシステムを書き換えずに、新旧を接続したい。

#### 解決策

```
┌──────────┐       ┌──────────────┐       ┌──────────┐
│  Client  │──────▶│   Adapter    │──────▶│  Adaptee │
│          │       │ (変換層)     │       │ (既存API) │
│ JSON を  │       │ XML→JSON    │       │ XML を    │
│ 期待     │       │ 変換する     │       │ 返す     │
└──────────┘       └──────────────┘       └──────────┘
```

#### Python 実装

```python
from abc import ABC, abstractmethod
import json
from xml.etree import ElementTree as ET


class ModernAPI(ABC):
    """新しいシステムが期待する JSON インターフェース。"""

    @abstractmethod
    def get_data(self) -> dict:
        ...


class LegacyXMLService:
    """レガシーシステム: XML でデータを返す。"""

    def fetch_xml(self) -> str:
        return """
        <users>
            <user>
                <name>Alice</name>
                <age>30</age>
            </user>
            <user>
                <name>Bob</name>
                <age>25</age>
            </user>
        </users>
        """


class XMLToJSONAdapter(ModernAPI):
    """Adapter: XML を返すレガシーサービスを JSON インターフェースに適合させる。

    Adaptee（LegacyXMLService）を内部に保持し、
    クライアントが期待する ModernAPI インターフェースを実装する。
    """

    def __init__(self, legacy_service: LegacyXMLService):
        self._legacy = legacy_service

    def get_data(self) -> dict:
        xml_str = self._legacy.fetch_xml()
        root = ET.fromstring(xml_str)
        users = []
        for user_elem in root.findall("user"):
            users.append({
                "name": user_elem.findtext("name", ""),
                "age": int(user_elem.findtext("age", "0")),
            })
        return {"users": users, "count": len(users)}


# 使用例
legacy = LegacyXMLService()
adapter = XMLToJSONAdapter(legacy)
data = adapter.get_data()
print(json.dumps(data, indent=2))
# {
#   "users": [
#     {"name": "Alice", "age": 30},
#     {"name": "Bob", "age": 25}
#   ],
#   "count": 2
# }
```

#### 結果（トレードオフ）

| メリット | デメリット |
|---------|----------|
| 既存コードを変更せずに互換性を確保 | Adapter クラスが増える |
| 単一責任原則に沿った変換の分離 | 過度な適用はラッパー地獄を招く |
| テスト時にレガシー部分を差し替え可能 | パフォーマンスオーバーヘッドがわずかに発生 |

---

### 3.2 Decorator パターン

#### 意図

オブジェクトに動的に新しい責務を追加する。
サブクラス化による機能拡張の代替手段であり、
単一責任原則を守りながら柔軟に機能を組み合わせられる。

#### 問題

コーヒーショップのシステムを考える。ベースのコーヒーにミルク、砂糖、
ホイップクリームなどのトッピングを自由に組み合わせたい。
サブクラスで全組み合わせを作ると、クラスの爆発的増加が起きる
（CoffeeWithMilk, CoffeeWithSugar, CoffeeWithMilkAndSugar, ...）。

#### 解決策

```
┌─────────────────────┐
│  Component (ABC)    │
│  + cost(): float    │
│  + description(): str│
└─────┬───────────────┘
      │
  ┌───┴──────────────────┐
  ▼                      ▼
┌───────────┐   ┌──────────────────┐
│ Concrete  │   │ Decorator (ABC)  │
│ Coffee    │   │ wraps Component  │
└───────────┘   └──┬───────────────┘
                   │
          ┌────────┼──────────┐
          ▼        ▼          ▼
       ┌──────┐ ┌──────┐ ┌────────┐
       │ Milk │ │Sugar │ │Whipped │
       │ Dec. │ │ Dec. │ │Cream D.│
       └──────┘ └──────┘ └────────┘
```

#### Python 実装

```python
from abc import ABC, abstractmethod


class Beverage(ABC):
    """飲み物の基底クラス。"""

    @abstractmethod
    def cost(self) -> float:
        ...

    @abstractmethod
    def description(self) -> str:
        ...


class Coffee(Beverage):
    """ベースとなるコーヒー。"""

    def cost(self) -> float:
        return 300.0

    def description(self) -> str:
        return "Coffee"


class Espresso(Beverage):
    """ベースとなるエスプレッソ。"""

    def cost(self) -> float:
        return 350.0

    def description(self) -> str:
        return "Espresso"


class BeverageDecorator(Beverage):
    """Decorator の基底クラス。内部に Beverage を保持する。"""

    def __init__(self, beverage: Beverage):
        self._beverage = beverage


class MilkDecorator(BeverageDecorator):
    def cost(self) -> float:
        return self._beverage.cost() + 50.0

    def description(self) -> str:
        return self._beverage.description() + " + Milk"


class SugarDecorator(BeverageDecorator):
    def cost(self) -> float:
        return self._beverage.cost() + 30.0

    def description(self) -> str:
        return self._beverage.description() + " + Sugar"


class WhippedCreamDecorator(BeverageDecorator):
    def cost(self) -> float:
        return self._beverage.cost() + 80.0

    def description(self) -> str:
        return self._beverage.description() + " + Whipped Cream"


# 使用例: Decorator を自由に組み合わせる
order = Coffee()
order = MilkDecorator(order)
order = SugarDecorator(order)
order = WhippedCreamDecorator(order)

print(order.description())  # "Coffee + Milk + Sugar + Whipped Cream"
print(f"合計: {order.cost()}円")  # "合計: 460.0円"

# Python のデコレータ構文 (@decorator) との関係:
# Python の @decorator は関数/クラスのラッピング機能であり、
# GoF の Decorator パターンとは概念的に類似するが別物である。
# ただし、@decorator を使って GoF の Decorator パターンを実装することもできる。
```

#### Python デコレータ（@）を使った Decorator パターン

```python
def logging_decorator(func):
    """関数呼び出しのログを自動追加する Python デコレータ。"""
    def wrapper(*args, **kwargs):
        print(f"[LOG] Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"[LOG] {func.__name__} returned {result}")
        return result
    return wrapper


def retry_decorator(max_retries: int = 3):
    """リトライ機能を追加する Python デコレータ。"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"[RETRY] Attempt {attempt + 1} failed: {e}")
        return wrapper
    return decorator


@logging_decorator
@retry_decorator(max_retries=3)
def fetch_data(url: str) -> str:
    return f"Data from {url}"
```

---

### 3.3 Facade パターン

#### 意図

複雑なサブシステムに対して、統一された簡潔なインターフェースを提供する。
クライアントはサブシステムの詳細を知る必要がなくなる。

#### 問題

オンラインショップの注文処理には、在庫確認、決済処理、配送手配、
メール通知など複数のサブシステムが関わる。
クライアントコードがこれらすべてを直接操作すると、
密結合になり変更が困難になる。

#### 解決策

```
                  ┌─────────────────────┐
  Client ────────▶│   OrderFacade       │
                  │ place_order()       │
                  └──┬──────┬──────┬────┘
                     │      │      │
           ┌─────────┘      │      └──────────┐
           ▼                ▼                  ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │  Inventory   │ │  Payment     │ │  Shipping    │
   │  Service     │ │  Service     │ │  Service     │
   └──────────────┘ └──────────────┘ └──────────────┘
```

#### Python 実装

```python
class InventoryService:
    """在庫管理サブシステム。"""

    def check_stock(self, product_id: str) -> bool:
        print(f"[Inventory] Checking stock for {product_id}")
        return True  # 在庫あり

    def reserve(self, product_id: str, qty: int) -> str:
        print(f"[Inventory] Reserved {qty} units of {product_id}")
        return f"RESERVE-{product_id}-{qty}"


class PaymentService:
    """決済サブシステム。"""

    def authorize(self, amount: float, card_token: str) -> str:
        print(f"[Payment] Authorized {amount} yen with card {card_token[:4]}****")
        return "AUTH-12345"

    def capture(self, auth_id: str) -> bool:
        print(f"[Payment] Captured payment {auth_id}")
        return True


class ShippingService:
    """配送サブシステム。"""

    def calculate_cost(self, address: str) -> float:
        print(f"[Shipping] Calculating cost to {address}")
        return 500.0

    def create_shipment(self, product_id: str, address: str) -> str:
        print(f"[Shipping] Created shipment for {product_id} to {address}")
        return "SHIP-67890"


class NotificationService:
    """通知サブシステム。"""

    def send_confirmation(self, email: str, order_id: str) -> None:
        print(f"[Notification] Sent confirmation for {order_id} to {email}")


class OrderFacade:
    """Facade: 注文処理の複雑さを隠蔽する統一インターフェース。

    クライアントは place_order() を呼ぶだけでよい。
    内部では 4 つのサブシステムが協調して動作する。
    """

    def __init__(self):
        self._inventory = InventoryService()
        self._payment = PaymentService()
        self._shipping = ShippingService()
        self._notification = NotificationService()

    def place_order(
        self,
        product_id: str,
        qty: int,
        card_token: str,
        address: str,
        email: str,
    ) -> dict:
        # Step 1: 在庫確認
        if not self._inventory.check_stock(product_id):
            raise RuntimeError(f"Product {product_id} is out of stock")

        # Step 2: 在庫予約
        reservation = self._inventory.reserve(product_id, qty)

        # Step 3: 配送料計算
        shipping_cost = self._shipping.calculate_cost(address)

        # Step 4: 決済
        total = qty * 1000 + shipping_cost  # 仮の単価 1000 円
        auth_id = self._payment.authorize(total, card_token)
        self._payment.capture(auth_id)

        # Step 5: 配送手配
        shipment_id = self._shipping.create_shipment(product_id, address)

        # Step 6: 確認メール送信
        order_id = f"ORDER-{reservation}-{shipment_id}"
        self._notification.send_confirmation(email, order_id)

        return {
            "order_id": order_id,
            "total": total,
            "shipment_id": shipment_id,
        }


# 使用例: クライアントは Facade だけを知ればよい
facade = OrderFacade()
result = facade.place_order(
    product_id="ITEM-001",
    qty=2,
    card_token="tok_visa_4242424242424242",
    address="東京都渋谷区...",
    email="customer@example.com",
)
print(f"注文完了: {result['order_id']}")
```

---

### 3.4 Composite パターン

#### 意図

オブジェクトをツリー構造に組み立て、個々のオブジェクトとその集合を
同一のインターフェースで扱えるようにする。

#### 問題

ファイルシステムでは、ファイル（葉）とディレクトリ（枝）を
統一的に扱いたい。ディレクトリの中にはファイルもディレクトリも入る。
サイズの計算や表示を再帰的に行いたいが、
ファイルとディレクトリで異なるインターフェースだと扱いにくい。

#### Python 実装

```python
from abc import ABC, abstractmethod


class FileSystemNode(ABC):
    """ファイルシステムのノード（Component）。"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def size(self) -> int:
        """サイズをバイト単位で返す。"""
        ...

    @abstractmethod
    def display(self, indent: int = 0) -> str:
        """ツリー表示用の文字列を返す。"""
        ...


class File(FileSystemNode):
    """ファイル（Leaf）。"""

    def __init__(self, name: str, size_bytes: int):
        super().__init__(name)
        self._size = size_bytes

    def size(self) -> int:
        return self._size

    def display(self, indent: int = 0) -> str:
        return " " * indent + f"[File] {self.name} ({self._size} bytes)"


class Directory(FileSystemNode):
    """ディレクトリ（Composite）。子要素を持つ。"""

    def __init__(self, name: str):
        super().__init__(name)
        self._children: list[FileSystemNode] = []

    def add(self, node: FileSystemNode) -> None:
        self._children.append(node)

    def remove(self, node: FileSystemNode) -> None:
        self._children.remove(node)

    def size(self) -> int:
        return sum(child.size() for child in self._children)

    def display(self, indent: int = 0) -> str:
        lines = [" " * indent + f"[Dir] {self.name} ({self.size()} bytes)"]
        for child in self._children:
            lines.append(child.display(indent + 2))
        return "\n".join(lines)


# 使用例
root = Directory("project")
src = Directory("src")
src.add(File("main.py", 2048))
src.add(File("utils.py", 1024))

tests = Directory("tests")
tests.add(File("test_main.py", 512))

root.add(src)
root.add(tests)
root.add(File("README.md", 256))

print(root.display())
# [Dir] project (3840 bytes)
#   [Dir] src (3072 bytes)
#     [File] main.py (2048 bytes)
#     [File] utils.py (1024 bytes)
#   [Dir] tests (512 bytes)
#     [File] test_main.py (512 bytes)
#   [File] README.md (256 bytes)

print(f"Total size: {root.size()} bytes")  # 3840
```

### 3.5 構造パターンの比較表

| パターン | 主な目的 | 適用場面 | キーワード |
|---------|---------|---------|-----------|
| Adapter | インターフェース変換 | レガシー統合、外部 API 連携 | ラッパー、変換 |
| Bridge | 抽象と実装の分離 | プラットフォーム独立、ドライバ | 分離、独立変化 |
| Composite | ツリー構造の統一操作 | ファイルシステム、UI ツリー、組織図 | 再帰、部分-全体 |
| Decorator | 動的な機能追加 | ストリーム、ミドルウェア、ログ付加 | ラッピング、重ね掛け |
| Facade | 複雑さの隠蔽 | サブシステム統合、API ゲートウェイ | 簡略化、統一窓口 |
| Flyweight | メモリ共有 | 文字描画、ゲームのタイル | 共有、軽量化 |
| Proxy | アクセス制御・代理 | キャッシュ、遅延読込、認可チェック | 代理、制御 |

---

## 4. 振る舞いパターン（Behavioral Patterns）

振る舞いパターンは、オブジェクト間の通信や責務の分担方法を扱う。
アルゴリズムの切り替え、イベント通知、コマンドの実行と取り消しなど、
オブジェクト同士がどのように協調するかを整理するパターン群である。

### 4.1 Observer パターン

#### 意図

オブジェクト間に一対多の依存関係を定義し、
あるオブジェクトの状態が変化したときに、
依存する全てのオブジェクトに自動的に通知・更新を行う。

#### 問題

株価監視システムで、株価が変化するたびに
チャート表示、アラート通知、ログ記録などの複数のコンポーネントを更新したい。
各コンポーネントを直接呼び出すと密結合になり、
新しいコンポーネントの追加が困難になる。

#### 解決策

```
┌───────────────────┐       ┌────────────────────┐
│   Subject         │       │   Observer (ABC)   │
│ (Observable)      │       │                    │
├───────────────────┤       ├────────────────────┤
│ - observers: list │◆─────▶│ + update(data)     │
│ + attach(obs)     │  1..*  └────────┬───────────┘
│ + detach(obs)     │                 │
│ + notify()        │         ┌───────┼──────────┐
└───────────────────┘         ▼       ▼          ▼
                          ┌──────┐ ┌──────┐ ┌──────┐
                          │Chart │ │Alert │ │Logger│
                          │Obs.  │ │Obs.  │ │Obs.  │
                          └──────┘ └──────┘ └──────┘
```

#### Python 実装

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class Observer(ABC):
    """Observer インターフェース。"""

    @abstractmethod
    def update(self, event: str, data: Any) -> None:
        ...


class EventEmitter:
    """汎用的な Observable（Subject）の実装。

    イベント名ごとに Observer を管理し、
    特定のイベントが発生したときに該当する Observer のみに通知する。
    """

    def __init__(self):
        self._observers: dict[str, list[Observer]] = {}

    def on(self, event: str, observer: Observer) -> None:
        """イベントに Observer を登録する。"""
        self._observers.setdefault(event, []).append(observer)

    def off(self, event: str, observer: Observer) -> None:
        """イベントから Observer を解除する。"""
        if event in self._observers:
            self._observers[event].remove(observer)

    def emit(self, event: str, data: Any = None) -> None:
        """イベントを発火し、登録済みの全 Observer に通知する。"""
        for observer in self._observers.get(event, []):
            observer.update(event, data)


@dataclass
class StockPrice:
    symbol: str
    price: float
    change: float


class ChartObserver(Observer):
    """チャート表示の更新。"""

    def update(self, event: str, data: Any) -> None:
        if isinstance(data, StockPrice):
            direction = "▲" if data.change > 0 else "▼"
            print(f"[Chart] {data.symbol}: {data.price:.2f} {direction}")


class AlertObserver(Observer):
    """価格アラートの通知。"""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def update(self, event: str, data: Any) -> None:
        if isinstance(data, StockPrice) and abs(data.change) > self.threshold:
            print(f"[ALERT] {data.symbol} moved {data.change:+.2f}% !")


class LogObserver(Observer):
    """取引ログの記録。"""

    def __init__(self):
        self.log: list[str] = []

    def update(self, event: str, data: Any) -> None:
        if isinstance(data, StockPrice):
            entry = f"{data.symbol},{data.price},{data.change}"
            self.log.append(entry)
            print(f"[Log] Recorded: {entry}")


# 使用例
stock_feed = EventEmitter()

chart = ChartObserver()
alert = AlertObserver(threshold=2.0)
logger = LogObserver()

stock_feed.on("price_update", chart)
stock_feed.on("price_update", alert)
stock_feed.on("price_update", logger)

# 株価更新をシミュレーション
stock_feed.emit("price_update", StockPrice("AAPL", 178.50, +1.2))
# [Chart] AAPL: 178.50 ▲
# [Log] Recorded: AAPL,178.5,1.2

stock_feed.emit("price_update", StockPrice("GOOG", 141.80, -3.5))
# [Chart] GOOG: 141.80 ▼
# [ALERT] GOOG moved -3.50% !
# [Log] Recorded: GOOG,141.8,-3.5
```

---

### 4.2 Strategy パターン

#### 意図

アルゴリズムのファミリーを定義し、それぞれをカプセル化して交換可能にする。
クライアントコードを変更せずにアルゴリズムを切り替えられる。

#### 問題

E コマースサイトの割引計算で、通常割引、会員割引、季節割引など
複数の割引ルールがある。`if-elif` の連鎖で実装すると、
新しい割引ルールの追加や変更のたびに既存コードを修正する必要がある。

#### Python 実装

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


class DiscountStrategy(ABC):
    """割引戦略の基底クラス。"""

    @abstractmethod
    def calculate(self, price: float) -> float:
        """割引後の価格を返す。"""
        ...

    @abstractmethod
    def description(self) -> str:
        """戦略の説明を返す。"""
        ...


class NoDiscount(DiscountStrategy):
    def calculate(self, price: float) -> float:
        return price

    def description(self) -> str:
        return "割引なし"


class PercentageDiscount(DiscountStrategy):
    def __init__(self, percentage: float):
        self._percentage = percentage

    def calculate(self, price: float) -> float:
        return price * (1 - self._percentage / 100)

    def description(self) -> str:
        return f"{self._percentage}% OFF"


class FixedAmountDiscount(DiscountStrategy):
    def __init__(self, amount: float):
        self._amount = amount

    def calculate(self, price: float) -> float:
        return max(0, price - self._amount)

    def description(self) -> str:
        return f"{self._amount}円引き"


class BuyNGetFreeDiscount(DiscountStrategy):
    """N 個買うと 1 個無料。"""

    def __init__(self, buy_count: int):
        self._buy = buy_count

    def calculate(self, price: float) -> float:
        return price * self._buy / (self._buy + 1)

    def description(self) -> str:
        return f"{self._buy} 個買うと 1 個無料"


@dataclass
class ShoppingCart:
    """Strategy パターンで割引戦略を切り替え可能なカート。"""

    items: list[tuple[str, float]]
    strategy: DiscountStrategy

    def set_strategy(self, strategy: DiscountStrategy) -> None:
        """割引戦略を動的に変更する。"""
        self.strategy = strategy

    def subtotal(self) -> float:
        return sum(price for _, price in self.items)

    def total(self) -> float:
        return self.strategy.calculate(self.subtotal())

    def receipt(self) -> str:
        lines = ["--- レシート ---"]
        for name, price in self.items:
            lines.append(f"  {name}: {price:.0f}円")
        lines.append(f"  小計: {self.subtotal():.0f}円")
        lines.append(f"  {self.strategy.description()}")
        lines.append(f"  合計: {self.total():.0f}円")
        return "\n".join(lines)


# 使用例
cart = ShoppingCart(
    items=[("Python入門書", 3000), ("ノート", 500), ("ペン", 200)],
    strategy=NoDiscount(),
)
print(cart.receipt())
# 合計: 3700円

cart.set_strategy(PercentageDiscount(20))
print(cart.receipt())
# 20% OFF → 合計: 2960円

cart.set_strategy(FixedAmountDiscount(500))
print(cart.receipt())
# 500円引き → 合計: 3200円
```

#### Strategy パターンと関数型アプローチの比較

Python では、Strategy パターンをクラスではなく関数で実現することもできる。

```python
from typing import Callable

# 関数型 Strategy
DiscountFunc = Callable[[float], float]

def no_discount(price: float) -> float:
    return price

def percentage_off(pct: float) -> DiscountFunc:
    return lambda price: price * (1 - pct / 100)

def fixed_off(amount: float) -> DiscountFunc:
    return lambda price: max(0, price - amount)

# 使用例
apply_discount: DiscountFunc = percentage_off(15)
print(apply_discount(1000))  # 850.0
```

クラスベースと関数ベースの使い分けは以下の通りである。

| 観点 | クラスベース | 関数ベース |
|------|------------|-----------|
| 状態の保持 | フィールドで自然に保持 | クロージャで保持可能 |
| 説明文の付与 | description() メソッド | docstring または別途管理 |
| テスト容易性 | モックが容易 | 同程度に容易 |
| 拡張性 | 新メソッドの追加が容易 | 新関数の追加が容易 |
| 推奨場面 | 複雑な戦略、複数メソッド | 単純な変換、ワンライナー |

---

### 4.3 Command パターン

#### 意図

リクエストをオブジェクトとしてカプセル化し、
操作の実行・取り消し・やり直し・キューイングを可能にする。

#### 問題

テキストエディタで Undo/Redo 機能を実装したい。
操作ごとに「何をしたか」を記録し、逆操作を実行できるようにしたい。

#### Python 実装

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Command(ABC):
    """コマンドの基底クラス。"""

    @abstractmethod
    def execute(self) -> None:
        ...

    @abstractmethod
    def undo(self) -> None:
        ...

    @abstractmethod
    def description(self) -> str:
        ...


class TextEditor:
    """テキストエディタの本体（Receiver）。"""

    def __init__(self):
        self.content: str = ""
        self.cursor: int = 0

    def insert(self, text: str, position: int) -> None:
        self.content = self.content[:position] + text + self.content[position:]
        self.cursor = position + len(text)

    def delete(self, position: int, length: int) -> str:
        deleted = self.content[position:position + length]
        self.content = self.content[:position] + self.content[position + length:]
        self.cursor = position
        return deleted

    def __repr__(self) -> str:
        return f'TextEditor("{self.content}")'


class InsertCommand(Command):
    def __init__(self, editor: TextEditor, text: str, position: int):
        self._editor = editor
        self._text = text
        self._position = position

    def execute(self) -> None:
        self._editor.insert(self._text, self._position)

    def undo(self) -> None:
        self._editor.delete(self._position, len(self._text))

    def description(self) -> str:
        return f'Insert "{self._text}" at {self._position}'


class DeleteCommand(Command):
    def __init__(self, editor: TextEditor, position: int, length: int):
        self._editor = editor
        self._position = position
        self._length = length
        self._deleted_text: str = ""

    def execute(self) -> None:
        self._deleted_text = self._editor.delete(self._position, self._length)

    def undo(self) -> None:
        self._editor.insert(self._deleted_text, self._position)

    def description(self) -> str:
        return f'Delete {self._length} chars at {self._position}'


class CommandHistory:
    """Undo/Redo を管理する Invoker。"""

    def __init__(self):
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []

    def execute(self, command: Command) -> None:
        command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()  # 新しい操作後は redo 履歴をクリア

    def undo(self) -> None:
        if not self._undo_stack:
            print("Nothing to undo")
            return
        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)
        print(f"Undo: {command.description()}")

    def redo(self) -> None:
        if not self._redo_stack:
            print("Nothing to redo")
            return
        command = self._redo_stack.pop()
        command.execute()
        self._undo_stack.append(command)
        print(f"Redo: {command.description()}")


# 使用例
editor = TextEditor()
history = CommandHistory()

history.execute(InsertCommand(editor, "Hello", 0))
print(editor)  # TextEditor("Hello")

history.execute(InsertCommand(editor, " World", 5))
print(editor)  # TextEditor("Hello World")

history.execute(DeleteCommand(editor, 5, 6))
print(editor)  # TextEditor("Hello")

history.undo()  # Undo: Delete 6 chars at 5
print(editor)  # TextEditor("Hello World")

history.undo()  # Undo: Insert " World" at 5
print(editor)  # TextEditor("Hello")

history.redo()  # Redo: Insert " World" at 5
print(editor)  # TextEditor("Hello World")
```

---

### 4.4 Iterator パターン

#### 意図

コレクションの内部構造を公開せずに、要素に順番にアクセスする方法を提供する。

#### 問題

二分探索木やグラフなどのデータ構造を走査するとき、
走査のロジック（深さ優先、幅優先、中順など）をデータ構造自体に持たせると、
単一責任原則に反する。走査方法を外部化したい。

#### Python 実装

Python では `__iter__` と `__next__` を実装することで、
Iterator パターンを言語レベルでサポートしている。

```python
from collections import deque
from typing import Iterator, Generic, TypeVar

T = TypeVar("T")


class BinaryTreeNode(Generic[T]):
    """二分木のノード。"""

    def __init__(self, value: T):
        self.value = value
        self.left: "BinaryTreeNode[T] | None" = None
        self.right: "BinaryTreeNode[T] | None" = None


class InOrderIterator:
    """中順走査（左 → 根 → 右）の Iterator。"""

    def __init__(self, root: BinaryTreeNode | None):
        self._stack: list[BinaryTreeNode] = []
        self._push_left(root)

    def _push_left(self, node: BinaryTreeNode | None) -> None:
        while node:
            self._stack.append(node)
            node = node.left

    def __iter__(self):
        return self

    def __next__(self):
        if not self._stack:
            raise StopIteration
        node = self._stack.pop()
        self._push_left(node.right)
        return node.value


class BreadthFirstIterator:
    """幅優先走査の Iterator。"""

    def __init__(self, root: BinaryTreeNode | None):
        self._queue: deque[BinaryTreeNode] = deque()
        if root:
            self._queue.append(root)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._queue:
            raise StopIteration
        node = self._queue.popleft()
        if node.left:
            self._queue.append(node.left)
        if node.right:
            self._queue.append(node.right)
        return node.value


class BinaryTree(Generic[T]):
    """複数の走査方法を提供する二分木。"""

    def __init__(self, root: BinaryTreeNode[T] | None = None):
        self.root = root

    def in_order(self) -> InOrderIterator:
        return InOrderIterator(self.root)

    def breadth_first(self) -> BreadthFirstIterator:
        return BreadthFirstIterator(self.root)

    def __iter__(self):
        return self.in_order()


# 使用例: ツリーの構築
#       4
#      / \
#     2   6
#    / \ / \
#   1  3 5  7
root = BinaryTreeNode(4)
root.left = BinaryTreeNode(2)
root.right = BinaryTreeNode(6)
root.left.left = BinaryTreeNode(1)
root.left.right = BinaryTreeNode(3)
root.right.left = BinaryTreeNode(5)
root.right.right = BinaryTreeNode(7)

tree = BinaryTree(root)

print("In-order:", list(tree.in_order()))
# In-order: [1, 2, 3, 4, 5, 6, 7]

print("BFS:", list(tree.breadth_first()))
# BFS: [4, 2, 6, 1, 3, 5, 7]

# for ループでも使える（__iter__ が in_order を返す）
for value in tree:
    print(value, end=" ")
# 1 2 3 4 5 6 7
```

### 4.5 振る舞いパターンの比較表

| パターン | 主な目的 | 適用場面 | キーワード |
|---------|---------|---------|-----------|
| Observer | 状態変化の通知 | イベント、PubSub、リアクティブ | 通知、購読、一対多 |
| Strategy | アルゴリズムの交換 | 割引計算、ソート、認証方式 | 交換可能、ポリシー |
| Command | 操作のオブジェクト化 | Undo/Redo、キューイング、マクロ | 実行、取消、履歴 |
| Iterator | 順次アクセス | コレクション走査、ストリーム | 走査、カーソル |
| State | 状態に応じた振る舞い変更 | ワークフロー、TCP接続、UI状態 | 状態遷移、有限オートマトン |
| Template Method | 処理の骨格を定義 | フレームワーク、ETL、テスト | 骨格、フック、継承 |
| Chain of Responsibility | 処理の連鎖 | ミドルウェア、認可チェーン | 連鎖、パイプライン |
| Mediator | オブジェクト間の仲介 | チャットルーム、管制塔 | 仲介、ハブ |
| Memento | 状態の保存と復元 | スナップショット、チェックポイント | 保存、復元 |
| Visitor | データ構造と処理の分離 | AST 走査、レポート生成 | ダブルディスパッチ |

---

## 5. アーキテクチャパターン

GoF のデザインパターンがクラスレベルの設計を扱うのに対し、
アーキテクチャパターンはアプリケーション全体の構造を扱う。
ここでは代表的な 3 つのパターンを解説する。

### 5.1 MVC（Model-View-Controller）

#### 概要

アプリケーションを 3 つの役割に分離する。

- **Model**: ビジネスロジックとデータ
- **View**: ユーザーインターフェース（表示）
- **Controller**: ユーザー入力を受け取り、Model と View を仲介する

```
┌──────────┐     ユーザー操作     ┌──────────────┐
│          │ ──────────────────▶ │              │
│   View   │                     │  Controller  │
│ (表示)   │ ◀────────────────── │  (制御)      │
│          │     表示の更新       │              │
└────┬─────┘                     └──────┬───────┘
     │                                  │
     │  データの参照                     │ Model の操作
     │                                  │
     │         ┌──────────────┐         │
     └────────▶│    Model     │◀────────┘
               │ (データ/      │
               │  ビジネス     │
               │  ロジック)    │
               └──────────────┘
```

#### 特徴

- Web フレームワーク（Django、Ruby on Rails、Spring MVC）で広く採用
- Model と View の分離により、同じデータを複数の View で表示可能
- テスト時に View を差し替えて Model のロジックを単体テスト可能

### 5.2 MVVM（Model-View-ViewModel）

#### 概要

MVC の派生形で、データバインディングを活用して View と ViewModel を同期する。

- **Model**: データとビジネスロジック
- **View**: UI の表示
- **ViewModel**: View に表示するデータの変換・管理。View の状態を保持する

```
┌──────────┐     データバインディング     ┌──────────────┐
│          │ ◀═══════════════════════▶ │              │
│   View   │     (双方向同期)           │  ViewModel   │
│          │                            │              │
└──────────┘                            └──────┬───────┘
                                               │
                                               │ データ操作
                                               ▼
                                        ┌──────────────┐
                                        │    Model     │
                                        └──────────────┘
```

#### 特徴

- フロントエンドフレームワーク（Vue.js、SwiftUI、WPF）で広く採用
- データバインディングにより View の手動更新が不要
- ViewModel は View に依存しないためテストが容易

### 5.3 Repository パターン

#### 概要

データアクセスロジックを抽象化し、ドメインモデルとデータベースの間に
仲介層を設ける。ビジネスロジックはデータの取得・保存方法の詳細を知る必要がなくなる。

```
┌───────────────┐     ┌────────────────────┐     ┌──────────────┐
│  Business     │────▶│   Repository       │────▶│  Database    │
│  Logic        │     │   (Interface)      │     │  / ORM /     │
│               │◀────│                    │◀────│  API / File  │
└───────────────┘     └────────────────────┘     └──────────────┘
```

#### Python 実装

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic

T = TypeVar("T")


@dataclass
class User:
    id: int
    name: str
    email: str


class UserRepository(ABC):
    """ユーザーリポジトリのインターフェース。"""

    @abstractmethod
    def find_by_id(self, user_id: int) -> User | None:
        ...

    @abstractmethod
    def find_by_email(self, email: str) -> User | None:
        ...

    @abstractmethod
    def find_all(self) -> list[User]:
        ...

    @abstractmethod
    def save(self, user: User) -> None:
        ...

    @abstractmethod
    def delete(self, user_id: int) -> None:
        ...


class InMemoryUserRepository(UserRepository):
    """テスト用のインメモリ実装。"""

    def __init__(self):
        self._store: dict[int, User] = {}

    def find_by_id(self, user_id: int) -> User | None:
        return self._store.get(user_id)

    def find_by_email(self, email: str) -> User | None:
        for user in self._store.values():
            if user.email == email:
                return user
        return None

    def find_all(self) -> list[User]:
        return list(self._store.values())

    def save(self, user: User) -> None:
        self._store[user.id] = user

    def delete(self, user_id: int) -> None:
        self._store.pop(user_id, None)


class UserService:
    """ビジネスロジック層。Repository に依存する。"""

    def __init__(self, repo: UserRepository):
        self._repo = repo

    def register(self, user_id: int, name: str, email: str) -> User:
        existing = self._repo.find_by_email(email)
        if existing:
            raise ValueError(f"Email {email} is already registered")
        user = User(id=user_id, name=name, email=email)
        self._repo.save(user)
        return user

    def get_user(self, user_id: int) -> User:
        user = self._repo.find_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        return user


# 使用例
repo = InMemoryUserRepository()
service = UserService(repo)

alice = service.register(1, "Alice", "alice@example.com")
bob = service.register(2, "Bob", "bob@example.com")

print(service.get_user(1))   # User(id=1, name='Alice', email='alice@example.com')
print(repo.find_all())       # [User(...), User(...)]

# テスト時は InMemoryUserRepository を使い、
# 本番では SQLAlchemyUserRepository 等に差し替える
```

### 5.4 アーキテクチャパターンの比較表

| パターン | View と Logic の結合度 | データ同期 | 主な適用先 | テスト容易性 |
|---------|----------------------|-----------|-----------|------------|
| MVC | Controller が仲介 | 手動（Controller経由） | Web（Django、Rails） | Model のテスト容易 |
| MVVM | データバインディング | 自動（双方向バインド） | SPA（Vue）、モバイル（SwiftUI） | ViewModel のテスト容易 |
| MVP | Presenter が仲介 | 手動（Presenter経由） | Android（旧来）、デスクトップ | Presenter のテスト容易 |
| Repository | 関心の分離 | Repository 経由 | DDD、Clean Architecture | Repository 差し替えで容易 |

---

## 6. パターンの選び方

### 6.1 判断フローチャート

パターンの選択は、直面している問題の性質から判断する。

```
問題の種類は？
│
├── オブジェクトの「生成」に関する問題
│   ├── インスタンスを 1 つに制限したい → Singleton
│   ├── 生成するクラスを動的に決めたい → Factory Method
│   ├── 関連オブジェクト群を一括生成   → Abstract Factory
│   ├── 複雑なオブジェクトを段階構築   → Builder
│   └── 既存オブジェクトをコピーしたい → Prototype
│
├── クラス/オブジェクトの「構造」に関する問題
│   ├── インターフェースの不一致を解消 → Adapter
│   ├── 機能を動的に追加したい       → Decorator
│   ├── 複雑さを隠蔽したい           → Facade
│   ├── ツリー構造を統一的に扱いたい → Composite
│   └── アクセスを制御・代理したい   → Proxy
│
└── オブジェクト間の「振る舞い」に関する問題
    ├── 状態変化を複数に通知したい   → Observer
    ├── アルゴリズムを交換可能にしたい → Strategy
    ├── 操作の実行/取消を管理したい   → Command
    ├── コレクションを走査したい     → Iterator
    └── 状態に応じて振る舞いを変えたい → State
```

### 6.2 パターン選択の原則

1. **問題先行**: パターンありきで設計しない。問題を明確にしてからパターンを探す
2. **最小適用**: 必要最小限のパターンを適用する。複数パターンの組み合わせは慎重に
3. **YAGNI**: 将来の拡張のためにパターンを適用するのは避ける。今必要なものだけ
4. **チームの理解度**: チームメンバーが理解できないパターンは保守コストを増大させる
5. **言語の特性**: Python のダックタイピング、ファーストクラス関数などを活かし、過度なクラス階層を避ける

### 6.3 Python におけるパターンの簡略化

Python の動的型付けとファーストクラス関数により、
Java/C++ で必要だったクラス階層が不要になるケースがある。

| GoF パターン | Java での実装 | Python での簡略化 |
|-------------|-------------|------------------|
| Strategy | インターフェース + 実装クラス群 | 関数を引数として渡す |
| Command | Command インターフェース + 実装 | 関数/lambda + リスト |
| Observer | Observer インターフェース + Subject | コールバック関数のリスト |
| Singleton | private コンストラクタ + static | モジュールレベル変数 |
| Factory Method | 抽象クラス + サブクラス群 | 辞書 + 関数/クラス |
| Iterator | Iterator インターフェース実装 | `__iter__` / `__next__` / ジェネレータ |
| Template Method | 抽象クラス + 継承 | 高階関数またはミックスイン |

---

## 7. アンチパターン

### 7.1 God Object（神オブジェクト）

#### 説明

1 つのクラスがシステムの大部分の機能を担当し、
あらゆることを知り、あらゆることを行うクラスのことである。
単一責任原則（SRP）に完全に違反している。

#### 症状

- クラスのコードが数百行〜数千行に膨らんでいる
- メソッド数が 20 以上ある
- 異なるドメインの責務が 1 つのクラスに混在している
- 変更のたびにそのクラスを修正する必要がある

#### 悪い例

```python
# アンチパターン: God Object
class Application:
    """何でもやるクラス（やってはいけない例）。"""

    def __init__(self):
        self.users = []
        self.products = []
        self.orders = []
        self.db_connection = None
        self.email_client = None
        self.cache = {}
        self.log_file = None

    def connect_to_database(self): ...
    def close_database(self): ...
    def create_user(self, name, email): ...
    def delete_user(self, user_id): ...
    def authenticate_user(self, email, password): ...
    def add_product(self, name, price): ...
    def update_product(self, product_id, **kwargs): ...
    def search_products(self, query): ...
    def create_order(self, user_id, items): ...
    def calculate_shipping(self, address): ...
    def process_payment(self, order_id, card): ...
    def send_confirmation_email(self, user_id, order_id): ...
    def generate_report(self, report_type): ...
    def export_to_csv(self, data): ...
    def clear_cache(self): ...
    def write_log(self, message): ...
    # ... 延々と続く
```

#### 改善策

責務ごとにクラスを分割し、それぞれを疎結合に連携させる。

```python
# 改善: 責務ごとにクラスを分割
class UserService:
    """ユーザー管理に特化。"""
    def create(self, name: str, email: str) -> "User": ...
    def authenticate(self, email: str, password: str) -> bool: ...

class ProductService:
    """商品管理に特化。"""
    def add(self, name: str, price: float) -> "Product": ...
    def search(self, query: str) -> list["Product"]: ...

class OrderService:
    """注文管理に特化。"""
    def create(self, user_id: int, items: list) -> "Order": ...
    def process_payment(self, order_id: int, card: str) -> bool: ...

class NotificationService:
    """通知に特化。"""
    def send_confirmation(self, user_id: int, order_id: int) -> None: ...
```

---

### 7.2 Golden Hammer（金のハンマー）

#### 説明

「ハンマーしか持っていなければ、すべてが釘に見える」という格言に由来する。
特定のパターンや技術に習熟すると、あらゆる問題にそのパターンを適用しようとする傾向のこと。

#### 症状

- 単純な問題に対して過度に複雑なパターンを適用している
- 「念のため」Strategy パターンを使っているが、戦略が 1 つしかない
- Factory を使っているが、生成するクラスが 1 種類しかない
- Observer を使っているが、Observer が 1 つしかない

#### 対策

1. **問題の複雑さに見合った解決策を選ぶ**: パターンの適用は問題が複雑な場合のみ
2. **YAGNI を守る**: 将来のために今パターンを入れるのは避ける
3. **リファクタリングで導入**: 最初はシンプルに実装し、必要になった時点でパターンを導入する

```python
# Golden Hammer の例: 不要な Factory
# 生成するクラスが 1 種類なら Factory は不要

# 過剰設計（やりすぎ）
class LoggerFactory:
    @staticmethod
    def create(logger_type: str):
        if logger_type == "console":
            return ConsoleLogger()
        raise ValueError(f"Unknown: {logger_type}")

logger = LoggerFactory.create("console")  # 常に console しか使わない

# 適切（シンプル）
logger = ConsoleLogger()  # 直接生成で十分
```

---

### 7.3 Cargo Cult Programming（カーゴカルト プログラミング）

#### 説明

パターンやプラクティスの意図を理解せず、
「有名プロジェクトで使われているから」「先輩が書いたから」という理由だけで
盲目的にコピーすること。表面的には正しく見えるが、本質的な理解が欠けている。

#### 症状

- パターンの「形」だけ真似して、解決すべき問題がない
- 全クラスにインターフェースを定義しているが、実装が常に 1 つ
- デザインパターンの名前だけ知っていて、トレードオフを説明できない

#### 対策

1. パターンを適用する前に「なぜ」を自問する
2. パターンのトレードオフ（メリットとデメリットの両方）を理解する
3. パターンなしで書いた場合と比較し、パターンの価値を確認する

---

## 8. 演習問題

### 8.1 基礎レベル（Beginner）

**問題 1**: 以下の要件に対して、どのデザインパターンが適切か答えよ。理由も述べよ。

(a) アプリケーション全体で設定ファイルの内容を共有したい
(b) ログの出力先をファイル、コンソール、リモートサーバーから選択したい
(c) 外部ライブラリのインターフェースが自社システムと合わないので変換したい

<details>
<summary>解答例</summary>

(a) **Singleton パターン**（またはモジュールレベル変数）。
設定オブジェクトはアプリケーション全体で 1 つであるべきであり、
どこからでも同じ設定にアクセスする必要があるため。
Python ではモジュールレベル変数で十分なことが多い。

(b) **Strategy パターン**。
ログ出力のアルゴリズム（出力先）を交換可能にすることで、
実行時に出力先を切り替えられる。
Python では logging モジュールの Handler がまさにこの構造。

(c) **Adapter パターン**。
既存のライブラリを変更せずに、自社システムが期待するインターフェースに
変換するラッパーを作成する。

</details>

**問題 2**: Singleton パターンの問題点を 3 つ挙げ、それぞれの対策を述べよ。

<details>
<summary>解答例</summary>

1. **テストが困難**: グローバル状態を持つため、テスト間で状態が共有される
   → 対策: 依存性注入（DI）を使い、テスト時にモックを渡す

2. **マルチスレッドの競合**: 複数スレッドが同時にインスタンス生成を試みる可能性
   → 対策: Lock によるダブルチェックロッキング、またはモジュールレベル変数の使用

3. **隠れた依存関係**: Singleton を直接参照するコードは暗黙の依存を持つ
   → 対策: コンストラクタインジェクションで明示的に依存を宣言する

</details>

### 8.2 中級レベル（Intermediate）

**問題 3**: 以下の仕様を満たすログシステムを、Decorator パターンを使って実装せよ。

- ベースのログ出力（コンソールに文字列を表示）
- タイムスタンプを付加する Decorator
- ログレベル（INFO, WARN, ERROR）を付加する Decorator
- JSON 形式に変換する Decorator
- これらを自由に組み合わせ可能

<details>
<summary>解答例</summary>

```python
from abc import ABC, abstractmethod
from datetime import datetime
import json


class Logger(ABC):
    @abstractmethod
    def log(self, message: str) -> str:
        ...


class ConsoleLogger(Logger):
    def log(self, message: str) -> str:
        print(message)
        return message


class LogDecorator(Logger):
    def __init__(self, logger: Logger):
        self._logger = logger


class TimestampDecorator(LogDecorator):
    def log(self, message: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self._logger.log(f"[{timestamp}] {message}")


class LevelDecorator(LogDecorator):
    def __init__(self, logger: Logger, level: str = "INFO"):
        super().__init__(logger)
        self._level = level

    def log(self, message: str) -> str:
        return self._logger.log(f"[{self._level}] {message}")


class JsonDecorator(LogDecorator):
    def log(self, message: str) -> str:
        payload = json.dumps({"message": message}, ensure_ascii=False)
        return self._logger.log(payload)


# 組み合わせ例
logger = ConsoleLogger()
logger = TimestampDecorator(logger)
logger = LevelDecorator(logger, "ERROR")
logger.log("Database connection failed")
# [ERROR] [2026-01-15 10:30:00] Database connection failed
```

</details>

**問題 4**: Command パターンを使って、簡単な計算機（四則演算）に
Undo/Redo 機能を実装せよ。

<details>
<summary>解答例</summary>

```python
from abc import ABC, abstractmethod


class CalculatorCommand(ABC):
    @abstractmethod
    def execute(self) -> float: ...
    @abstractmethod
    def undo(self) -> float: ...


class Calculator:
    def __init__(self):
        self.value = 0.0
        self._history: list[CalculatorCommand] = []
        self._redo_stack: list[CalculatorCommand] = []

    def execute(self, command: "CalculatorCommand") -> float:
        result = command.execute()
        self._history.append(command)
        self._redo_stack.clear()
        return result

    def undo(self) -> float:
        if self._history:
            cmd = self._history.pop()
            cmd.undo()
            self._redo_stack.append(cmd)
        return self.value

    def redo(self) -> float:
        if self._redo_stack:
            cmd = self._redo_stack.pop()
            cmd.execute()
            self._history.append(cmd)
        return self.value


class AddCommand(CalculatorCommand):
    def __init__(self, calc: Calculator, operand: float):
        self._calc = calc
        self._operand = operand

    def execute(self) -> float:
        self._calc.value += self._operand
        return self._calc.value

    def undo(self) -> float:
        self._calc.value -= self._operand
        return self._calc.value


class MultiplyCommand(CalculatorCommand):
    def __init__(self, calc: Calculator, operand: float):
        self._calc = calc
        self._operand = operand
        self._prev_value = 0.0

    def execute(self) -> float:
        self._prev_value = self._calc.value
        self._calc.value *= self._operand
        return self._calc.value

    def undo(self) -> float:
        self._calc.value = self._prev_value
        return self._calc.value


calc = Calculator()
calc.execute(AddCommand(calc, 10))     # 10.0
calc.execute(AddCommand(calc, 5))      # 15.0
calc.execute(MultiplyCommand(calc, 3)) # 45.0
calc.undo()                            # 15.0
calc.undo()                            # 10.0
calc.redo()                            # 15.0
```

</details>

### 8.3 上級レベル（Advanced）

**問題 5**: 以下の要件を満たすプラグインシステムを設計・実装せよ。
複数のデザインパターンを組み合わせること。

要件:
- プラグインの動的な登録・解除ができる（Factory + Registry）
- プラグインの実行順序を制御できる（Chain of Responsibility）
- プラグインの実行結果をログに記録できる（Observer）
- 全プラグインの実行を 1 つのコマンドで行える（Facade）

ヒント: まず各コンポーネントのインターフェースを設計し、
次にパターンを組み合わせて全体を構成する。

<details>
<summary>設計の方向性</summary>

```python
from abc import ABC, abstractmethod
from typing import Any


class Plugin(ABC):
    """プラグインの基底クラス。"""
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def priority(self) -> int: ...
    @abstractmethod
    def execute(self, context: dict) -> dict: ...


class PluginRegistry:
    """Factory + Registry: プラグインの登録・生成管理。"""
    _plugins: dict[str, type[Plugin]] = {}

    @classmethod
    def register(cls, plugin_class: type[Plugin]) -> None:
        instance = plugin_class()
        cls._plugins[instance.name()] = plugin_class

    @classmethod
    def create(cls, name: str) -> Plugin:
        return cls._plugins[name]()

    @classmethod
    def create_all(cls) -> list[Plugin]:
        plugins = [cls.create(name) for name in cls._plugins]
        return sorted(plugins, key=lambda p: p.priority())


class PluginEventBus:
    """Observer: プラグイン実行イベントを通知。"""
    _listeners: list = []

    @classmethod
    def subscribe(cls, listener) -> None:
        cls._listeners.append(listener)

    @classmethod
    def publish(cls, event: str, data: Any) -> None:
        for listener in cls._listeners:
            listener(event, data)


class PluginEngine:
    """Facade: プラグインシステムの統一窓口。"""
    def __init__(self):
        self._plugins = PluginRegistry.create_all()

    def run(self, context: dict) -> dict:
        for plugin in self._plugins:
            PluginEventBus.publish("before", {"plugin": plugin.name()})
            context = plugin.execute(context)
            PluginEventBus.publish("after", {"plugin": plugin.name(), "context": context})
        return context
```

具体的なプラグインの実装とテストコードは、読者の演習として残す。
重要なのは、各パターンが独立した責務を持ち、
組み合わせることで柔軟なシステムが構築できる点を理解することである。

</details>

---

## 9. よくある質問（FAQ）

### Q1: デザインパターンはいつ学ぶべきか？

デザインパターンの学習に最適な時期は、ある程度のコーディング経験を積んだ後である。
目安として、以下の条件を満たしていると効果的に学べる。

- オブジェクト指向プログラミングの基礎（クラス、継承、ポリモーフィズム）を理解している
- 数千行以上のプログラムを書いた経験がある
- 「このコード、もっとうまく構造化できないか」と感じたことがある

初学者がパターンだけを暗記しても実践では活かしにくい。
まずコードを書いて「痛み」を経験し、
その痛みを解決する手段としてパターンを学ぶのが効果的である。

### Q2: 全 23 パターンを覚える必要があるか？

全てを暗記する必要はない。重要なのは以下の点である。

1. **カテゴリの理解**: 生成・構造・振る舞いの 3 分類を理解する
2. **頻出パターンの習熟**: 以下の 8〜10 パターンは実務で頻繁に出現する
   - Singleton, Factory Method, Builder（生成）
   - Adapter, Decorator, Facade（構造）
   - Observer, Strategy, Command, Iterator（振る舞い）
3. **引き出しとしての認識**: 残りのパターンは「こういう問題にはこういう解決策がある」と知っておき、必要になったときに詳細を調べられればよい

### Q3: Python ではパターンは不要と聞いたが本当か？

半分正しく、半分間違いである。

Python の動的型付け、ダックタイピング、ファーストクラス関数、デコレータ構文により、
Java/C++ で必要だった「ボイラープレートとしてのパターン」は確かに不要になる場合がある。
例えば、Strategy パターンはクラスを作らずに関数を渡すだけで実現できる。

しかし、パターンの本質は「実装方法」ではなく「設計の意図」である。
「ここは Strategy の考え方で設計している」と伝えることで、
チームメンバーは設計意図を即座に理解できる。
言語が変わっても、パターンの「概念」は有効であり続ける。

Python で重要なのは、パターンの意図を理解した上で、
Python らしい（Pythonic な）方法で実装することである。
Java のパターン実装をそのまま Python に持ち込むのは避けるべきである。

### Q4: パターンの適用判断で迷ったらどうするか？

迷った場合は、以下のステップで判断するとよい。

1. **まずシンプルに書く**: パターンなしで実装する
2. **痛みを感じたらリファクタリング**: 重複、条件分岐の増加、変更の困難さを感じたら
3. **パターンの意図と照合**: 感じた痛みがどのパターンの「問題」に該当するか確認
4. **トレードオフを評価**: パターン適用による複雑さの増加と、得られる柔軟性を天秤にかける
5. **チームと相談**: パターンの適用はチームの理解度に合わせる

「迷ったらシンプルに」が最も安全な原則である。

### Q5: マイクロサービスではデザインパターンはどう変わるか？

マイクロサービスアーキテクチャでは、GoF パターンに加えて
分散システム固有のパターンが重要になる。

- **Circuit Breaker**: 障害のあるサービスへの呼び出しを遮断し、連鎖障害を防ぐ
- **Saga**: 分散トランザクションを一連のローカルトランザクションとして管理する
- **CQRS**: コマンド（書き込み）とクエリ（読み取り）のモデルを分離する
- **Event Sourcing**: 状態の変更をイベントのシーケンスとして記録する
- **API Gateway**: 複数のマイクロサービスへのアクセスを統合する（Facade の分散版）
- **Sidecar**: サービスに付随するプロセスで横断的関心事を処理する

GoF パターンは「クラス内・プロセス内」の設計であり、
分散パターンは「サービス間・プロセス間」の設計である。
両者は排他的ではなく、階層が異なるだけである。

---

## 10. パターンの組み合わせと実践的ガイドライン

### 10.1 よく使われるパターンの組み合わせ

実際のプロジェクトでは、複数のパターンを組み合わせて使うことが多い。
以下は代表的な組み合わせである。

| 組み合わせ | 典型的な用途 | 説明 |
|-----------|------------|------|
| Factory + Strategy | プラグインシステム | Factory で Strategy を生成し、実行時に切り替え |
| Observer + Command | イベント駆動 UI | イベント（Observer）でコマンド（Command）を発火 |
| Composite + Iterator | ツリー走査 | Composite 構造を Iterator で順次処理 |
| Facade + Adapter | レガシー統合 | Adapter で変換し、Facade で統一窓口を提供 |
| Builder + Factory | 複雑なオブジェクト群 | Factory がどの Builder を使うかを決定 |
| Decorator + Strategy | ミドルウェアパイプライン | Decorator で重ね掛けし、各層が Strategy で処理 |

### 10.2 リファクタリングでパターンを導入するタイミング

パターンは「最初から入れる」のではなく「必要になったら入れる」が原則である。
以下のシグナルが現れたらパターン導入を検討する。

1. **同じ条件分岐が複数箇所に出現** → Strategy / State
2. **新しい型の追加で既存コードの修正が必要** → Factory Method
3. **オブジェクト間の通知が複雑化** → Observer / Mediator
4. **Undo 機能の要求** → Command / Memento
5. **外部 API との接続が増加** → Adapter / Facade
6. **コンストラクタの引数が 5 個以上** → Builder

---

## 11. まとめ

### 11.1 この章で学んだこと

| カテゴリ | 学習したパターン | 核心 |
|---------|---------------|------|
| 生成 | Singleton, Factory Method, Builder, Prototype | オブジェクト生成の柔軟性と制御 |
| 構造 | Adapter, Decorator, Facade, Composite | 構造の組み合わせと複雑さの管理 |
| 振る舞い | Observer, Strategy, Command, Iterator | 責務の分担と通信の整理 |
| アーキテクチャ | MVC, MVVM, Repository | アプリケーション全体の構造化 |

### 11.2 パターン習得のロードマップ

```
Phase 1: 基礎理解（この章の内容）
  ├── GoF の 3 分類を理解する
  ├── 頻出 8〜10 パターンの意図を説明できる
  └── Python での実装例を写経する

Phase 2: 実践適用
  ├── 既存コードからパターンを発見する
  ├── リファクタリングでパターンを導入する
  └── パターンの組み合わせを経験する

Phase 3: 設計判断
  ├── 問題に対して適切なパターンを選択できる
  ├── パターンの適用/不適用をトレードオフで判断できる
  └── チームにパターンの意図を説明できる

Phase 4: 応用・発展
  ├── 分散システムパターンを理解する
  ├── ドメイン駆動設計（DDD）のパターンを学ぶ
  └── 関数型プログラミングのパターンと比較する
```

### 11.3 設計における心構え

1. **パターンは目的ではなく手段である**: パターンを使うこと自体が目的にならないよう注意する
2. **シンプルさを優先する**: 最もシンプルな設計が最良の設計であることが多い
3. **問題を理解してからパターンを探す**: パターンカタログを眺めて適用先を探すのは本末転倒
4. **トレードオフを常に意識する**: どのパターンにもメリットとデメリットがある
5. **チームの文脈に合わせる**: チームの技術力、プロジェクトの規模、保守期間を考慮する

---

## 次に読むべきガイド
→ [[03-clean-code.md]] -- クリーンコード

---

## 参考文献

1. Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
   - GoF デザインパターンの原典。23 パターンの定義と C++/Smalltalk での実装例を収録。

2. Freeman, E., Robson, E., Bates, B., & Sierra, K. (2020). *Head First Design Patterns* (2nd Edition). O'Reilly Media.
   - 図解と対話形式でパターンを学べる入門書。Java ベースだが概念の理解に最適。

3. Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley.
   - エンタープライズアプリケーションにおけるアーキテクチャパターン集。Repository、Unit of Work、Data Mapper などを体系化。

4. Martin, R. C. (2017). *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall.
   - SOLID 原則とアーキテクチャパターンの関係を解説。依存性逆転の原則とパターンの適用判断に有用。

5. Buschmann, F. et al. (1996). *Pattern-Oriented Software Architecture Volume 1: A System of Patterns*. Wiley.
   - アーキテクチャレベルのパターン（MVC、Pipes and Filters、Broker 等）を体系化した書籍。
