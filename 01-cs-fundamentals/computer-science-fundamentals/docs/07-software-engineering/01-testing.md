# ソフトウェアテスト

> テストのないコードはレガシーコードである。——Michael Feathers, *Working Effectively with Legacy Code*

ソフトウェアテストは、プログラムが仕様どおりに動作することを検証し、
バグの早期発見・品質保証・保守性向上を実現するための体系的な活動である。
本章では、単体テストから統合テスト・E2E テスト・受入テストに至る分類体系、
TDD や BDD といった開発手法、同値分割・境界値分析などのテスト技法、
モックとスタブの使い分け、CI/CD パイプラインでの自動化戦略、
コードカバレッジの意義と限界、プロパティベーステスト、
そして現場でよく見られるアンチパターンまでを網羅的に扱う。

---

## この章で学ぶこと

- [ ] テストピラミッドの構造と各レイヤーの役割を理解する
- [ ] ユニットテストを AAA パターンで書けるようになる
- [ ] TDD の Red-Green-Refactor サイクルを実践できる
- [ ] BDD のシナリオ駆動テストを説明できる
- [ ] 同値分割・境界値分析・デシジョンテーブルを適用できる
- [ ] モックとスタブの使い分けを判断できる
- [ ] CI/CD パイプラインにテストを組み込む方法を知る
- [ ] カバレッジ指標の種類と限界を理解する
- [ ] プロパティベーステストの基本を知る
- [ ] テストに関する典型的なアンチパターンを識別し、回避できる

---

## 1. テストの重要性

### 1.1 なぜテストを書くのか

ソフトウェア開発においてテストが重要である理由は、大きく分けて以下の 4 つに整理できる。

**1. バグの早期検出とコスト削減**

ソフトウェアの欠陥は、発見が遅れるほど修正コストが指数関数的に増大する。
要件定義段階で見つかった誤りの修正コストを 1 とすると、
設計段階では 5 倍、実装段階では 10 倍、テスト段階では 20 倍、
リリース後には 100 倍以上に達するとされる（Barry Boehm の研究に基づく経験則）。
自動テストを開発と同時に書くことで、欠陥を実装段階で捕捉し、修正コストを最小化できる。

```
バグ修正コストの増大曲線（概念図）:

コスト
  ^
  |                                              * リリース後
  |                                         *
  |                                    *
  |                              *
  |                        *
  |                  *
  |            *
  |       * テスト段階
  |    * 実装段階
  |  * 設計段階
  | * 要件定義
  +--------------------------------------------> 時間
    要件  設計  実装  テスト  運用  保守
```

**2. 回帰（リグレッション）の防止**

既存機能が新しい変更によって壊れることを「回帰バグ」と呼ぶ。
自動テストスイートが存在すれば、コードを変更するたびにテストを実行し、
意図しない影響がないことを即座に確認できる。
これは特に大規模なコードベースやチーム開発において不可欠である。

**3. 設計の改善**

テストを書きやすいコードは、一般に疎結合で高凝集である。
テストを先に書く（TDD）ことで、自然とクリーンな設計に導かれる。
テストしにくいコードは、多くの場合、設計上の問題（密結合、副作用の多さ、責務の混在）を抱えている。

**4. 生きたドキュメントとしての役割**

テストコードは、対象コードの使い方や期待される振る舞いを具体的に示す。
API ドキュメントが古くなっても、テストが通り続ける限り、
テストコード自体が最新の仕様書として機能する。

### 1.2 テストの基本用語

| 用語 | 定義 |
|------|------|
| テストケース (Test Case) | 特定の条件下で期待される結果を検証する最小単位 |
| テストスイート (Test Suite) | 関連するテストケースをまとめたグループ |
| テストランナー (Test Runner) | テストを実行し結果を報告するツール（pytest, JUnit など） |
| テストフィクスチャ (Test Fixture) | テスト実行前後のセットアップ・ティアダウン処理 |
| アサーション (Assertion) | 期待値と実際の値を比較する検証文 |
| SUT (System Under Test) | テスト対象のシステムまたはコンポーネント |
| テストダブル (Test Double) | 本物のオブジェクトの代わりに使う代替品の総称 |
| テストカバレッジ (Test Coverage) | テストによって実行されるコードの割合 |
| 回帰テスト (Regression Test) | 既存機能が壊れていないことを確認するテスト |
| フレイキーテスト (Flaky Test) | 同じ条件で実行しても結果が安定しないテスト |

---

## 2. テストの分類

ソフトウェアテストは、テストの粒度、目的、実行タイミングなどによって多角的に分類される。
ここでは最も基本的な「テストレベルによる分類」を中心に解説する。

### 2.1 テストピラミッド

Mike Cohn が提唱したテストピラミッドは、
テストの自動化戦略における基本指針を図式化したものである。

```
テストピラミッド:

                    /\
                   /  \
                  / E2E \        ← 少数・高コスト・低速・壊れやすい
                 /  テスト \        ブラウザ操作、API全体の結合
                /──────────\
               /            \
              /   統合テスト   \    ← 中程度の量・コスト・速度
             /  Integration   \    DB接続、API連携、サービス間通信
            /──────────────────\
           /                    \
          /   ユニットテスト      \  ← 大量・低コスト・高速・安定
         /    Unit Tests         \   関数・クラス単位、モックで隔離
        /────────────────────────\

  推奨比率:
  ┌──────────────┬────────┬────────────┬──────────────┐
  │ レイヤー     │ 比率   │ 実行速度   │ 保守コスト   │
  ├──────────────┼────────┼────────────┼──────────────┤
  │ ユニット     │ 70%    │ ms 単位    │ 低           │
  │ 統合         │ 20%    │ 秒〜分     │ 中           │
  │ E2E          │ 10%    │ 分〜十分   │ 高           │
  └──────────────┴────────┴────────────┴──────────────┘
```

このピラミッドの意味するところは明快である。
低コストで高速なユニットテストを土台として大量に書き、
統合テストで主要なコンポーネント間の連携を検証し、
E2E テストはビジネス上重要なシナリオに絞って少数書く。

ピラミッドが逆三角形（アイスクリームコーン型）になると、
テストスイート全体が遅く、不安定で、保守コストが高くなる。
これは多くのプロジェクトで見られるアンチパターンである。

```
アンチパターン: アイスクリームコーン型

        ┌────────────────────────────┐
        │      手動テスト (大量)      │  ← 手動でしか確認していない
        ├────────────────────────────┤
        │    E2E テスト (大量)        │  ← 遅い・壊れやすい
        ├────────────────────────────┤
        │  統合テスト (少量)          │
        ├────────────────────────────┤
        │ ユニットテスト (極少)       │  ← ほとんど書かれていない
        └────────────────────────────┘

  問題点:
  - テストスイート全体の実行に数十分〜数時間
  - フレイキーテストが頻発し、テスト結果への信頼が低下
  - 開発者がテストを実行しなくなる悪循環
  - バグの原因特定が困難（粒度が粗すぎる）
```

### 2.2 ユニットテスト（単体テスト）

ユニットテストは、関数やメソッドなどの最小単位を隔離して検証するテストである。
テストピラミッドの土台であり、テストスイート全体の 70% 以上を占めることが推奨される。

#### 特徴

- **高速**: 1 テストあたりミリ秒単位で完了する
- **隔離**: 外部依存（DB、ネットワーク、ファイルシステム）はモックで置き換える
- **決定的**: 同じ入力に対して常に同じ結果を返す
- **独立**: テスト間に順序依存がない

#### AAA パターン

ユニットテストの構造は AAA（Arrange-Act-Assert）パターンに従うのが標準である。

```python
# ===== コード例 1: pytest によるユニットテストと AAA パターン =====

import pytest
from decimal import Decimal


# --- テスト対象のコード ---

class Money:
    """金額を扱うバリューオブジェクト。"""

    def __init__(self, amount: int, currency: str = "JPY"):
        if amount < 0:
            raise ValueError("金額は 0 以上でなければならない")
        if currency not in ("JPY", "USD", "EUR"):
            raise ValueError(f"未対応の通貨: {currency}")
        self.amount = amount
        self.currency = currency

    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("異なる通貨同士の加算はできない")
        return Money(self.amount + other.amount, self.currency)

    def multiply(self, factor: int) -> "Money":
        if factor < 0:
            raise ValueError("係数は 0 以上でなければならない")
        return Money(self.amount * factor, self.currency)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount and self.currency == other.currency

    def __repr__(self) -> str:
        return f"Money({self.amount}, '{self.currency}')"


# --- テストコード ---

class TestMoney:
    """Money クラスのユニットテスト。"""

    # === 正常系テスト ===

    def test_生成_正常な金額で生成できる(self):
        # Arrange（準備）
        amount = 1000
        currency = "JPY"

        # Act（実行）
        money = Money(amount, currency)

        # Assert（検証）
        assert money.amount == 1000
        assert money.currency == "JPY"

    def test_加算_同一通貨の金額を加算できる(self):
        # Arrange
        money1 = Money(1000, "JPY")
        money2 = Money(500, "JPY")

        # Act
        result = money1.add(money2)

        # Assert
        assert result == Money(1500, "JPY")

    def test_乗算_正の係数で乗算できる(self):
        # Arrange
        money = Money(100, "USD")

        # Act
        result = money.multiply(3)

        # Assert
        assert result == Money(300, "USD")

    def test_等価性_同じ金額と通貨なら等しい(self):
        assert Money(500, "JPY") == Money(500, "JPY")

    def test_等価性_金額が異なれば等しくない(self):
        assert Money(500, "JPY") != Money(600, "JPY")

    def test_等価性_通貨が異なれば等しくない(self):
        assert Money(500, "JPY") != Money(500, "USD")

    # === 異常系テスト ===

    def test_生成_負の金額で例外が発生する(self):
        with pytest.raises(ValueError, match="金額は 0 以上"):
            Money(-100, "JPY")

    def test_生成_未対応通貨で例外が発生する(self):
        with pytest.raises(ValueError, match="未対応の通貨"):
            Money(100, "GBP")

    def test_加算_異なる通貨で例外が発生する(self):
        money_jpy = Money(1000, "JPY")
        money_usd = Money(10, "USD")
        with pytest.raises(ValueError, match="異なる通貨"):
            money_jpy.add(money_usd)

    def test_乗算_負の係数で例外が発生する(self):
        money = Money(100, "JPY")
        with pytest.raises(ValueError, match="係数は 0 以上"):
            money.multiply(-1)

    # === 境界値テスト ===

    def test_生成_金額0は有効(self):
        money = Money(0, "JPY")
        assert money.amount == 0

    def test_加算_金額0同士の加算(self):
        result = Money(0, "JPY").add(Money(0, "JPY"))
        assert result == Money(0, "JPY")

    def test_乗算_係数0で金額0になる(self):
        result = Money(1000, "JPY").multiply(0)
        assert result == Money(0, "JPY")
```

#### テスト命名規則

テスト名は「何をテストしているか」「どの条件で」「何が期待されるか」を明確にする。

| 命名スタイル | 例 | 特徴 |
|---|---|---|
| 日本語メソッド名 | `test_加算_同一通貨で加算できる` | 可読性が高い。pytest で利用可能 |
| Given-When-Then | `test_given_same_currency_when_add_then_returns_sum` | BDD 風。条件が明確 |
| should スタイル | `test_add_should_return_sum_for_same_currency` | 期待動作が明確 |
| メソッド名_条件_期待 | `test_add_sameCurrency_returnsSum` | Java/JUnit で一般的 |

### 2.3 統合テスト（Integration Test）

統合テストは、複数のコンポーネントを組み合わせた状態で、
それらが正しく連携するかを検証するテストである。

#### ユニットテストとの違い

| 観点 | ユニットテスト | 統合テスト |
|------|---------------|-----------|
| テスト対象 | 単一の関数・クラス | 複数コンポーネントの連携 |
| 外部依存 | モックで隔離 | 実際のリソースを使用 |
| 実行速度 | ms 単位 | 秒〜分単位 |
| テスト環境 | 特別な準備不要 | DB、APIサーバー等の準備が必要 |
| 検出できるバグ | ロジックの誤り | 接続設定、データ変換、プロトコルの不一致 |
| 安定性 | 高い（決定的） | やや低い（環境依存） |

#### 統合テストの対象例

- データベースとの CRUD 操作
- 外部 API との通信
- メッセージキューの送受信
- ファイルシステムへの読み書き
- 認証・認可フローの全体

```python
# ===== コード例 2: pytest + SQLAlchemy による統合テスト =====

import pytest
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


class User(Base):
    """ユーザーテーブルのモデル。"""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)


class UserRepository:
    """ユーザーのデータアクセスを担当するリポジトリ。"""

    def __init__(self, session):
        self.session = session

    def add(self, name: str, email: str) -> User:
        user = User(name=name, email=email)
        self.session.add(user)
        self.session.commit()
        return user

    def find_by_email(self, email: str) -> User | None:
        return self.session.query(User).filter_by(email=email).first()

    def find_all(self) -> list[User]:
        return self.session.query(User).all()

    def delete(self, user: User) -> None:
        self.session.delete(user)
        self.session.commit()


# --- テスト用フィクスチャ ---

@pytest.fixture
def db_session():
    """テスト用のインメモリ SQLite セッションを提供する。"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def user_repo(db_session):
    """UserRepository のインスタンスを提供する。"""
    return UserRepository(db_session)


# --- 統合テスト ---

class TestUserRepository:
    """UserRepository の統合テスト。実際の DB（インメモリ SQLite）を使用する。"""

    def test_ユーザーを追加して取得できる(self, user_repo):
        # Arrange & Act
        user_repo.add("Alice", "alice@example.com")

        # Assert
        found = user_repo.find_by_email("alice@example.com")
        assert found is not None
        assert found.name == "Alice"
        assert found.email == "alice@example.com"

    def test_存在しないメールで検索するとNoneが返る(self, user_repo):
        result = user_repo.find_by_email("nobody@example.com")
        assert result is None

    def test_複数ユーザーを登録して全件取得できる(self, user_repo):
        user_repo.add("Alice", "alice@example.com")
        user_repo.add("Bob", "bob@example.com")
        user_repo.add("Charlie", "charlie@example.com")

        users = user_repo.find_all()
        assert len(users) == 3

    def test_ユーザーを削除できる(self, user_repo):
        user = user_repo.add("Alice", "alice@example.com")
        user_repo.delete(user)

        assert user_repo.find_by_email("alice@example.com") is None

    def test_重複メールで追加すると例外が発生する(self, user_repo):
        user_repo.add("Alice", "alice@example.com")
        with pytest.raises(Exception):  # IntegrityError
            user_repo.add("Alice2", "alice@example.com")
```

### 2.4 E2E テスト（End-to-End テスト）

E2E テストは、アプリケーション全体をエンドユーザーの視点から検証するテストである。
ブラウザ自動操作ツール（Playwright, Cypress, Selenium）を使い、
実際のユーザー操作をシミュレートする。

#### E2E テストの位置づけ

```
E2E テストのスコープ:

  ブラウザ/クライアント    サーバー           データベース
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  ユーザー操作  │───>│  API/Web     │───>│  データ永続化  │
  │  画面遷移     │<───│  ビジネスロジック│<───│  クエリ実行   │
  │  表示確認     │    │  認証/認可    │    │              │
  └──────────────┘    └──────────────┘    └──────────────┘
       ^                                         |
       |          E2E テストのスコープ              |
       └─────────────────────────────────────────┘
       全レイヤーを貫通して検証する
```

#### E2E テストを書くべき場面

- ログイン → 商品検索 → カート追加 → 決済 のような重要なユーザーフロー
- 法規制やビジネス要件に直結する機能（決済、個人情報処理）
- 過去に重大な障害が発生した箇所

#### E2E テストを避けるべき場面

- 個々のバリデーションルール（ユニットテストで十分）
- 全ての UI パターンの網羅（コストが見合わない）
- 頻繁に変更される UI 要素（テストが壊れやすくなる）

### 2.5 受入テスト（Acceptance Test）

受入テストは、システムがビジネス要件を満たしていることを確認するテストである。
顧客やプロダクトオーナーの視点で書かれ、
「この機能は完了したか？」の判断基準となる。

受入テストは E2E テストと混同されがちだが、異なる概念である。

| 観点 | E2E テスト | 受入テスト |
|------|-----------|-----------|
| 目的 | システム全体の技術的な動作確認 | ビジネス要件の充足確認 |
| 視点 | 開発者 | 顧客・プロダクトオーナー |
| 記述者 | QA エンジニア・開発者 | PO と開発者の協力 |
| 実装手段 | Playwright, Cypress 等 | Cucumber, Behave 等（BDD ツール） |
| 実行頻度 | CI で自動実行 | スプリントレビュー時など |

---

## 3. テスト駆動開発（TDD）

### 3.1 TDD の基本サイクル

TDD（Test-Driven Development）は、Kent Beck が体系化した開発手法であり、
「テストを先に書く」ことを核心とする。

```
TDD の Red-Green-Refactor サイクル:

        ┌─────────────────────────────────┐
        │                                 │
        v                                 │
  ┌───────────┐    ┌───────────┐    ┌───────────┐
  │   RED     │───>│   GREEN   │───>│ REFACTOR  │
  │           │    │           │    │           │
  │ 失敗する   │    │ テストを   │    │ コードを   │
  │ テストを   │    │ 通す最小限 │    │ 整理する   │
  │ 書く      │    │ のコードを │    │ (テストは  │
  │           │    │ 書く      │    │  通ったまま)│
  └───────────┘    └───────────┘    └───────────┘

  各フェーズの詳細:

  RED（赤）:
    1. まだ存在しない機能のテストを書く
    2. テストを実行し、失敗することを確認する
    3. 失敗の理由が「機能が未実装だから」であることを確認する

  GREEN（緑）:
    1. テストを通す最小限のコードを書く
    2. 美しさや効率は一切考えない
    3. ハードコードでも構わない（最初のステップとして）
    4. テストが通ることを確認する

  REFACTOR（リファクタリング）:
    1. 重複を排除する
    2. 命名を改善する
    3. 設計パターンを適用する
    4. テストが通り続けることを常に確認する
```

### 3.2 TDD の実践例: FizzBuzz

TDD の流れを FizzBuzz を題材に示す。

```python
# ===== コード例 3: TDD で FizzBuzz を実装する =====

# --- ステップ 1: RED — 最初のテストを書く ---
# テストファイル: test_fizzbuzz.py

def test_1を渡すと文字列1を返す():
    assert fizzbuzz(1) == "1"

# この時点で fizzbuzz 関数は存在しないため、テストは失敗する（NameError）


# --- ステップ 2: GREEN — テストを通す最小限のコード ---
# プロダクションコード: fizzbuzz.py

def fizzbuzz(n: int) -> str:
    return str(n)

# テスト実行 → PASSED


# --- ステップ 3: RED — 次のテストを追加 ---

def test_3を渡すとFizzを返す():
    assert fizzbuzz(3) == "Fizz"

# テスト実行 → FAILED（"3" != "Fizz"）


# --- ステップ 4: GREEN — テストを通す ---

def fizzbuzz(n: int) -> str:
    if n % 3 == 0:
        return "Fizz"
    return str(n)

# テスト実行 → 2件とも PASSED


# --- ステップ 5: RED — さらにテストを追加 ---

def test_5を渡すとBuzzを返す():
    assert fizzbuzz(5) == "Buzz"

# テスト実行 → FAILED


# --- ステップ 6: GREEN ---

def fizzbuzz(n: int) -> str:
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)

# テスト実行 → 3件とも PASSED


# --- ステップ 7: RED — 15の倍数のテスト ---

def test_15を渡すとFizzBuzzを返す():
    assert fizzbuzz(15) == "FizzBuzz"

# テスト実行 → FAILED（"Fizz" != "FizzBuzz"）


# --- ステップ 8: GREEN ---

def fizzbuzz(n: int) -> str:
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)

# テスト実行 → 4件とも PASSED


# --- ステップ 9: REFACTOR — コードの整理 ---

def fizzbuzz(n: int) -> str:
    """n に対する FizzBuzz の結果を返す。

    - 3 の倍数なら "Fizz"
    - 5 の倍数なら "Buzz"
    - 15 の倍数なら "FizzBuzz"
    - それ以外は数値の文字列表現
    """
    result = ""
    if n % 3 == 0:
        result += "Fizz"
    if n % 5 == 0:
        result += "Buzz"
    return result or str(n)

# テスト実行 → 4件とも PASSED（リファクタリング後もテストは通る）


# --- 最終的なテストスイート ---

import pytest

class TestFizzBuzz:
    """FizzBuzz のテストスイート。"""

    @pytest.mark.parametrize("input_val, expected", [
        (1, "1"),
        (2, "2"),
        (3, "Fizz"),
        (5, "Buzz"),
        (6, "Fizz"),
        (10, "Buzz"),
        (15, "FizzBuzz"),
        (30, "FizzBuzz"),
        (7, "7"),
    ])
    def test_fizzbuzz(self, input_val, expected):
        assert fizzbuzz(input_val) == expected
```

### 3.3 TDD の利点と注意点

**利点:**

1. **設計が使いやすい API に導かれる** — テストを先に書くことで、利用者の視点でインターフェースを考えることになる
2. **回帰テストが自動的に蓄積される** — 開発と同時にテストスイートが成長する
3. **過剰な実装を防ぐ（YAGNI 原則）** — テストで要求された機能だけを実装する
4. **変更への自信** — リファクタリング時にテストが安全網として機能する
5. **デバッグ時間の短縮** — バグが入り込んだ時点でテストが失敗するため、原因箇所の特定が容易

**注意点:**

1. **全てに TDD を適用する必要はない** — 探索的なプロトタイピングや UI 実装では、設計が固まってからテストを書く方が効率的な場合がある
2. **テストの保守コスト** — テストコードもプロダクションコードと同様にメンテナンスが必要
3. **学習コスト** — TDD を効果的に実践するには、テスト設計とリファクタリングのスキルが必要
4. **過度なモック** — TDD に不慣れだと、テストを通すためにモックを多用しがちで、実装の詳細に結合したテストになりやすい

---

## 4. ビヘイビア駆動開発（BDD）

### 4.1 BDD の概要

BDD（Behavior-Driven Development）は、Dan North が提唱した開発手法であり、
TDD をビジネスの視点から再解釈したものである。
「テスト」ではなく「振る舞いの仕様（Specification）」としてシナリオを記述する。

BDD の核心は、開発者・QA・ビジネスサイドの三者（"Three Amigos"）が
共通言語（ユビキタス言語）でシナリオを議論し、
それをそのまま自動テストとして実行可能にすることにある。

### 4.2 Gherkin 記法

BDD のシナリオは Gherkin と呼ばれる自然言語に近い記法で書かれる。

```gherkin
# ===== Gherkin によるシナリオ記述の例 =====

Feature: ショッピングカート
  オンラインショップの顧客として
  商品をカートに追加し、合計金額を確認したい
  購入前に数量の変更や商品の削除もできるようにしたい

  Background:
    Given 以下の商品がカタログに登録されている
      | 商品名      | 単価  |
      | Python入門  | 3000  |
      | Go実践      | 3500  |
      | Rust入門    | 4000  |

  Scenario: 商品をカートに追加する
    Given カートが空である
    When "Python入門" を 1 個カートに追加する
    Then カート内の商品数は 1 である
    And 合計金額は 3000 円である

  Scenario: 複数商品をカートに追加する
    Given カートが空である
    When "Python入門" を 2 個カートに追加する
    And "Go実践" を 1 個カートに追加する
    Then カート内の商品数は 3 である
    And 合計金額は 9500 円である

  Scenario: カートから商品を削除する
    Given カートに "Python入門" が 1 個入っている
    When "Python入門" をカートから削除する
    Then カートが空である
    And 合計金額は 0 円である

  Scenario Outline: 数量による合計金額の計算
    Given カートが空である
    When "<商品名>" を <数量> 個カートに追加する
    Then 合計金額は <期待金額> 円である

    Examples:
      | 商品名     | 数量 | 期待金額 |
      | Python入門 | 1    | 3000    |
      | Python入門 | 3    | 9000    |
      | Go実践     | 2    | 7000    |
      | Rust入門   | 1    | 4000    |
```

### 4.3 TDD と BDD の比較

| 観点 | TDD | BDD |
|------|-----|-----|
| 起点 | 技術的なテスト | ビジネスの振る舞い |
| 記述者 | 開発者 | 開発者 + PO + QA |
| 記法 | プログラミング言語 | Gherkin（自然言語風） |
| 粒度 | 関数・メソッド単位 | ユーザーストーリー単位 |
| ツール | pytest, JUnit 等 | Cucumber, Behave, SpecFlow 等 |
| 主な用途 | 実装の正しさの検証 | 要件の合意と検証 |
| 共通理解 | 開発チーム内 | ビジネスチーム含む全体 |

---

## 5. テスト技法

テスト技法は、テストケースを効率的かつ効果的に設計するための方法論である。
無限にある入力の組み合わせから、バグを発見しやすいテストケースを選び出す手法を学ぶ。

### 5.1 同値分割（Equivalence Partitioning）

入力ドメインを「同じ振る舞いをするグループ（同値クラス）」に分割し、
各クラスから代表値を 1 つずつ選んでテストする技法である。

```
同値分割の例: 年齢による料金区分

  入力: 年齢（0〜150 の整数と仮定）
  ルール:
    - 0〜5 歳    → 無料
    - 6〜12 歳   → 子供料金
    - 13〜64 歳  → 大人料金
    - 65〜150 歳 → シニア料金
    - 上記以外   → エラー

  同値クラス:
  ┌─────────────────────────────────────────────────────────┐
  │ 無効(負) │  無料  │ 子供  │  大人   │ シニア │ 無効(超過) │
  │  < 0    │ 0〜5  │ 6〜12 │ 13〜64 │ 65〜150│  > 150    │
  └─────────────────────────────────────────────────────────┘

  代表値:  -1     3      9      30      80       200
```

### 5.2 境界値分析（Boundary Value Analysis）

同値クラスの境界付近は、off-by-one エラー（1 つずれるバグ）が発生しやすい。
境界値分析では、境界の値とその前後の値をテストする。

```
境界値分析の例: 年齢による料金区分

  テストすべき境界値:

  無効/無料の境界:  -1,  0,  1
  無料/子供の境界:   4,  5,  6,  7
  子供/大人の境界:  11, 12, 13, 14
  大人/シニアの境界: 63, 64, 65, 66
  シニア/無効の境界: 149, 150, 151

  テストケース表:
  ┌──────┬──────────┬───────────┐
  │ 入力 │ 期待区分  │ テスト目的  │
  ├──────┼──────────┼───────────┤
  │  -1  │ エラー   │ 下限外     │
  │   0  │ 無料     │ 下限境界   │
  │   1  │ 無料     │ 下限+1     │
  │   5  │ 無料     │ 上限境界   │
  │   6  │ 子供     │ 次クラス下限│
  │  12  │ 子供     │ 上限境界   │
  │  13  │ 大人     │ 次クラス下限│
  │  64  │ 大人     │ 上限境界   │
  │  65  │ シニア   │ 次クラス下限│
  │ 150  │ シニア   │ 上限境界   │
  │ 151  │ エラー   │ 上限外     │
  └──────┴──────────┴───────────┘
```

### 5.3 デシジョンテーブル（Decision Table）

複数の条件の組み合わせによって動作が変わる場合、
デシジョンテーブルを使って全ての組み合わせを網羅的に列挙する。

```
デシジョンテーブルの例: ECサイトの送料計算

  条件:
    C1: 会員か？           (Yes/No)
    C2: 合計金額 3000円以上？ (Yes/No)
    C3: 離島か？           (Yes/No)

  ┌──────────────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  │ ルール番号        │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │
  ├──────────────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
  │ C1: 会員         │ Yes │ Yes │ Yes │ Yes │ No  │ No  │ No  │ No  │
  │ C2: 3000円以上   │ Yes │ Yes │ No  │ No  │ Yes │ Yes │ No  │ No  │
  │ C3: 離島         │ No  │ Yes │ No  │ Yes │ No  │ Yes │ No  │ Yes │
  ├──────────────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
  │ 送料             │ 0円 │500円│300円│800円│500円│1000円│800円│1500円│
  └──────────────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

### 5.4 ペアワイズテスト（Pairwise Testing）

条件が多くなると全組み合わせの数が爆発的に増大する。
ペアワイズテストは、「任意の 2 因子の全ての値の組み合わせを少なくとも 1 回カバーする」
という基準で、テストケース数を大幅に削減する技法である。

実際のバグの多くは、2 つの要因の相互作用で発生するという研究結果に基づく。

| 因子数 | 全組み合わせ | ペアワイズ | 削減率 |
|--------|------------|-----------|--------|
| 3因子 x 3値 | 27 | 9〜12 | 56〜67% |
| 4因子 x 3値 | 81 | 9〜15 | 81〜89% |
| 10因子 x 3値 | 59,049 | 15〜20 | 99.97% |
| 13因子 x 3値 | 1,594,323 | 15〜20 | 99.999% |

ペアワイズテストの生成には PICT（Microsoft 製）、AllPairs などのツールを使用する。

---

## 6. モックとスタブ

### 6.1 テストダブルの分類

Martin Fowler の分類に基づくテストダブルの種類を整理する。

```
テストダブルの分類:

  ┌─────────────────────────────────────────────────────────┐
  │                   テストダブル (Test Double)              │
  │                                                         │
  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐ │
  │  │ ダミー   │  │ スタブ   │  │ スパイ   │  │   モック   │ │
  │  │ Dummy   │  │ Stub    │  │ Spy     │  │   Mock    │ │
  │  ├─────────┤  ├─────────┤  ├─────────┤  ├───────────┤ │
  │  │引数を   │  │固定値を │  │呼び出し │  │期待される │ │
  │  │埋める   │  │返す    │  │を記録   │  │呼び出しを │ │
  │  │だけ    │  │        │  │する    │  │検証する   │ │
  │  └─────────┘  └─────────┘  └─────────┘  └───────────┘ │
  │                                                         │
  │  ┌─────────────────┐                                    │
  │  │   フェイク       │                                    │
  │  │   Fake          │                                    │
  │  ├─────────────────┤                                    │
  │  │ 簡易実装を持つ   │                                    │
  │  │ (インメモリDB等) │                                    │
  │  └─────────────────┘                                    │
  └─────────────────────────────────────────────────────────┘
```

| 種類 | 目的 | 振る舞い | 検証 |
|------|------|---------|------|
| ダミー (Dummy) | 引数を埋めるために渡す | 何もしない。呼ばれたら例外 | しない |
| スタブ (Stub) | 間接入力を制御する | 事前定義された固定値を返す | しない |
| スパイ (Spy) | 間接出力を記録する | 呼び出し履歴を保持する | 呼び出し履歴を事後検証 |
| モック (Mock) | 期待される相互作用を検証する | 期待に沿わない呼び出しで失敗 | 相互作用を検証 |
| フェイク (Fake) | 軽量な代替実装を提供する | 動作する簡易実装（インメモリ DB 等） | しない |

### 6.2 unittest.mock の実践

```python
# ===== コード例 4: unittest.mock を用いたモックとスタブの実践 =====

from unittest.mock import Mock, patch, MagicMock
import pytest
from dataclasses import dataclass
from typing import Protocol


# --- プロダクションコード ---

@dataclass
class WeatherData:
    """天気データを表すデータクラス。"""
    city: str
    temperature: float
    humidity: float
    description: str


class WeatherApiClient:
    """外部天気 API との通信を担当する。"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_weather(self, city: str) -> dict:
        """外部 API から天気データを取得する（実際のHTTP通信）。"""
        # 実際には requests.get() などで API を呼ぶ
        raise NotImplementedError("本番コードでは HTTP 通信を行う")


class WeatherService:
    """天気情報のビジネスロジックを担当する。"""

    def __init__(self, api_client: WeatherApiClient):
        self.api_client = api_client

    def get_weather(self, city: str) -> WeatherData:
        """指定都市の天気を取得して WeatherData に変換する。"""
        raw = self.api_client.fetch_weather(city)
        return WeatherData(
            city=city,
            temperature=raw["main"]["temp"],
            humidity=raw["main"]["humidity"],
            description=raw["weather"][0]["description"],
        )

    def is_hot(self, city: str, threshold: float = 30.0) -> bool:
        """指定都市が暑いかどうかを判定する。"""
        weather = self.get_weather(city)
        return weather.temperature >= threshold

    def compare_temperature(self, city1: str, city2: str) -> str:
        """2 都市の気温を比較する。"""
        w1 = self.get_weather(city1)
        w2 = self.get_weather(city2)
        if w1.temperature > w2.temperature:
            return f"{city1} の方が暑い"
        elif w1.temperature < w2.temperature:
            return f"{city2} の方が暑い"
        else:
            return "同じ気温"


# --- テストコード（スタブの例）---

class TestWeatherService:
    """WeatherService のテスト。外部 API はスタブで置き換える。"""

    def _create_service_with_stub(self, stub_response: dict) -> WeatherService:
        """スタブ化された API クライアントでサービスを生成する。"""
        mock_client = Mock(spec=WeatherApiClient)
        mock_client.fetch_weather.return_value = stub_response
        return WeatherService(mock_client)

    def _sample_response(self, temp: float = 25.0, humidity: float = 60.0,
                         desc: str = "clear sky") -> dict:
        """テスト用のダミーレスポンスを生成する。"""
        return {
            "main": {"temp": temp, "humidity": humidity},
            "weather": [{"description": desc}],
        }

    def test_天気データを正しく変換できる(self):
        # Arrange: スタブが固定のレスポンスを返すように設定
        service = self._create_service_with_stub(
            self._sample_response(temp=25.0, humidity=60.0, desc="晴れ")
        )

        # Act
        weather = service.get_weather("東京")

        # Assert
        assert weather.city == "東京"
        assert weather.temperature == 25.0
        assert weather.humidity == 60.0
        assert weather.description == "晴れ"

    def test_閾値以上なら暑いと判定する(self):
        service = self._create_service_with_stub(
            self._sample_response(temp=35.0)
        )
        assert service.is_hot("東京", threshold=30.0) is True

    def test_閾値未満なら暑くないと判定する(self):
        service = self._create_service_with_stub(
            self._sample_response(temp=25.0)
        )
        assert service.is_hot("東京", threshold=30.0) is False


# --- テストコード（モックの例：呼び出しの検証）---

class TestWeatherServiceInteraction:
    """WeatherService の相互作用をモックで検証する。"""

    def test_get_weatherがAPIクライアントを正しく呼び出す(self):
        # Arrange
        mock_client = Mock(spec=WeatherApiClient)
        mock_client.fetch_weather.return_value = {
            "main": {"temp": 25.0, "humidity": 60.0},
            "weather": [{"description": "晴れ"}],
        }
        service = WeatherService(mock_client)

        # Act
        service.get_weather("大阪")

        # Assert: API クライアントが正しい引数で呼ばれたことを検証
        mock_client.fetch_weather.assert_called_once_with("大阪")

    def test_compare_temperatureが2回APIを呼び出す(self):
        # Arrange
        mock_client = Mock(spec=WeatherApiClient)
        mock_client.fetch_weather.side_effect = [
            {"main": {"temp": 30.0, "humidity": 50.0},
             "weather": [{"description": "晴れ"}]},
            {"main": {"temp": 25.0, "humidity": 70.0},
             "weather": [{"description": "曇り"}]},
        ]
        service = WeatherService(mock_client)

        # Act
        result = service.compare_temperature("東京", "札幌")

        # Assert
        assert result == "東京 の方が暑い"
        assert mock_client.fetch_weather.call_count == 2
```

### 6.3 モックの使いすぎに注意

モックの過剰使用は、以下の問題を引き起こす。

1. **テストが実装の詳細に結合する** — 内部構造を変更しただけでテストが壊れる
2. **偽の安心感を与える** — モックが正しく設定されていれば通るが、実際の連携は壊れている
3. **テストの可読性が低下する** — モックの設定コードが複雑になると、何をテストしているか分かりにくい

**経験則**: 「自分が所有していないコードはモックしない」（Don't mock what you don't own）。
外部ライブラリや API クライアントを直接モックするのではなく、
薄いラッパー（アダプター）を挟み、そのアダプターをモックする。

---

## 7. CI/CD でのテスト自動化

### 7.1 テスト自動化の全体像

```
CI/CD パイプラインにおけるテスト自動化:

  コード変更
      │
      v
  ┌─────────────────────────────────────────────────────┐
  │                  CI パイプライン                       │
  │                                                     │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ Lint /   │─>│ ユニット  │─>│ 統合     │          │
  │  │ 静的解析  │  │ テスト   │  │ テスト   │          │
  │  │ (数秒)   │  │ (数十秒) │  │ (数分)   │          │
  │  └──────────┘  └──────────┘  └──────────┘          │
  │       │              │             │                │
  │       v              v             v                │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ セキュリ  │  │ カバレッジ│  │ E2E     │          │
  │  │ ティスキャン│ │ レポート │  │ テスト   │          │
  │  │ (数分)   │  │ 生成     │  │ (数十分) │          │
  │  └──────────┘  └──────────┘  └──────────┘          │
  │                      │                              │
  │                      v                              │
  │               ┌──────────┐                          │
  │               │ ビルド / │                          │
  │               │ パッケージ│                          │
  │               └──────────┘                          │
  └─────────────────────────────────────────────────────┘
      │
      v
  ┌─────────────────────────────────────────────────────┐
  │                  CD パイプライン                       │
  │                                                     │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ステージング│─>│ スモーク  │─>│ 本番     │          │
  │  │デプロイ   │  │ テスト   │  │ デプロイ  │          │
  │  └──────────┘  └──────────┘  └──────────┘          │
  └─────────────────────────────────────────────────────┘
```

### 7.2 GitHub Actions によるテスト自動化の例

```yaml
# .github/workflows/test.yml

name: テスト自動実行

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    name: Lint & 静的解析
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff mypy
      - run: ruff check .
      - run: mypy src/

  unit-test:
    name: ユニットテスト
    runs-on: ubuntu-latest
    needs: lint
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[test]"
      - run: pytest tests/unit/ -v --cov=src/ --cov-report=xml
      - uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  integration-test:
    name: 統合テスト
    runs-on: ubuntu-latest
    needs: unit-test
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: testdb
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[test]"
      - run: pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql://testuser:testpass@localhost:5432/testdb

  e2e-test:
    name: E2E テスト
    runs-on: ubuntu-latest
    needs: integration-test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[test]"
      - run: npx playwright install --with-deps chromium
      - run: pytest tests/e2e/ -v --headed=false
```

### 7.3 テスト自動化のベストプラクティス

1. **高速フィードバック** — ユニットテストは PR ごとに必ず実行する。全テストが 5 分以内に完了するのが理想
2. **テストの並列実行** — pytest-xdist などで並列化し、実行時間を短縮する
3. **失敗時の即時停止** — `pytest -x`（最初の失敗で停止）オプションを活用し、無駄な待ち時間を排除する
4. **テスト結果のキャッシュ** — 変更のないモジュールのテストをスキップする
5. **フレイキーテストの管理** — 不安定なテストを隔離し、定期的に修正する

---

## 8. コードカバレッジ

### 8.1 カバレッジ指標の種類

| 指標 | 定義 | 計測対象 |
|------|------|---------|
| 行カバレッジ (Line) | テストで実行された行の割合 | ソースコードの各行 |
| 分岐カバレッジ (Branch) | テストで通過した分岐の割合 | if/else, switch の各分岐 |
| 関数カバレッジ (Function) | テストで呼び出された関数の割合 | 定義された各関数 |
| 条件カバレッジ (Condition) | 各条件式の真偽両方がテストされた割合 | 複合条件の各部分条件 |
| パスカバレッジ (Path) | テストで通過した実行パスの割合 | 全ての実行パス |
| MC/DC | 各条件が独立して判定結果に影響することの検証 | 航空宇宙・自動車等の安全規格 |

### 8.2 カバレッジの目安

| カバレッジ水準 | 意味 | 適用場面 |
|--------------|------|---------|
| 80% 以上 | 標準的な目標。多くのプロジェクトで推奨 | 一般的な Web アプリケーション |
| 90% 以上 | 高品質。コアライブラリで推奨 | ライブラリ、フレームワーク |
| 95% 以上 | 非常に高品質。維持コストも高い | 決済処理、医療系 |
| 100% | 理想的だが、実務上は費用対効果が低い場合が多い | 安全クリティカルシステム |

### 8.3 カバレッジの限界

**カバレッジは「テストの網羅性」を測るが、「テストの品質」は測れない。**

```python
# カバレッジ 100% だがバグを検出できない例

def divide(a: int, b: int) -> float:
    return a / b  # b=0 の場合の処理がない

def test_divide():
    assert divide(10, 2) == 5.0
    # この 1 つのテストだけで行カバレッジ 100% だが、
    # b=0 の場合のテストがないためバグを見逃す
```

カバレッジが高いことは必要条件だが、十分条件ではない。
以下の点を常に意識する必要がある。

1. **カバレッジはコードが「実行された」ことしか示さない** — 正しい結果が返されたかは別問題
2. **カバレッジ目標の数値だけを追うと、品質の低いテストが増える** — assert のないテストや、意味のないテストケース
3. **カバレッジが測定できない品質要因がある** — 性能、ユーザビリティ、セキュリティなど
4. **エッジケースの不在はカバレッジに表れない** — 正常系だけで 100% に達することがある

### 8.4 pytest-cov によるカバレッジ計測

```bash
# カバレッジレポートの生成
pytest --cov=src/ --cov-report=term-missing --cov-report=html

# 出力例:
# Name                     Stmts   Miss  Cover   Missing
# -------------------------------------------------------
# src/money.py                25      2    92%   18, 22
# src/weather_service.py      40      5    88%   35-39
# -------------------------------------------------------
# TOTAL                       65      7    89%
```

---

## 9. プロパティベーステスト

### 9.1 プロパティベーステストとは

従来のテスト（サンプルベーステスト）では、
テスト作成者が具体的な入力値と期待値のペアを手動で選ぶ。
プロパティベーステストでは、入力をランダムに自動生成し、
「どんな入力に対しても成り立つべき性質（プロパティ）」を検証する。

| 比較項目 | サンプルベーステスト | プロパティベーステスト |
|---------|-------------------|---------------------|
| 入力値の決定 | テスト作成者が手動で選択 | フレームワークが自動生成 |
| テストケース数 | 数件〜数十件 | 数百件〜数千件（自動） |
| バグ発見力 | 思いつかないケースは見逃す | 予想外の入力パターンを発見 |
| テストの記述 | 具体的な入力と期待値 | 抽象的な性質（プロパティ） |
| 再現性 | 常に同じ | シードで再現可能 |
| 縮小（shrinking） | なし | 最小の反例を自動探索 |

### 9.2 Hypothesis による実践

```python
# ===== コード例 5: Hypothesis によるプロパティベーステスト =====

from hypothesis import given, assume, settings, example
from hypothesis import strategies as st
import pytest


# --- テスト対象 ---

def sort_list(lst: list[int]) -> list[int]:
    """リストをソートして返す。"""
    return sorted(lst)


def reverse_string(s: str) -> str:
    """文字列を反転させる。"""
    return s[::-1]


def encode_decode(text: str) -> str:
    """UTF-8 でエンコードしてデコードする。"""
    return text.encode("utf-8").decode("utf-8")


def clamp(value: int, min_val: int, max_val: int) -> int:
    """value を min_val〜max_val の範囲に制限する。"""
    if min_val > max_val:
        raise ValueError("min_val は max_val 以下でなければならない")
    return max(min_val, min(value, max_val))


# --- プロパティベーステスト ---

class TestSortListProperties:
    """sort_list のプロパティベーステスト。"""

    @given(st.lists(st.integers()))
    def test_ソート結果の長さは入力と同じ(self, lst):
        """プロパティ: ソートは要素数を変えない。"""
        result = sort_list(lst)
        assert len(result) == len(lst)

    @given(st.lists(st.integers()))
    def test_ソート結果は昇順に並んでいる(self, lst):
        """プロパティ: ソート結果の各要素は前の要素以上。"""
        result = sort_list(lst)
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]

    @given(st.lists(st.integers()))
    def test_ソート結果は入力と同じ要素を含む(self, lst):
        """プロパティ: ソートは要素を変えない（並び替えるだけ）。"""
        result = sort_list(lst)
        assert sorted(result) == sorted(lst)

    @given(st.lists(st.integers()))
    def test_ソートの冪等性(self, lst):
        """プロパティ: 2回ソートしても結果は同じ。"""
        once = sort_list(lst)
        twice = sort_list(once)
        assert once == twice


class TestReverseStringProperties:
    """reverse_string のプロパティベーステスト。"""

    @given(st.text())
    def test_二重反転で元に戻る(self, s):
        """プロパティ: 反転の反転は恒等変換。"""
        assert reverse_string(reverse_string(s)) == s

    @given(st.text())
    def test_反転結果の長さは同じ(self, s):
        """プロパティ: 反転は長さを変えない。"""
        assert len(reverse_string(s)) == len(s)


class TestClampProperties:
    """clamp のプロパティベーステスト。"""

    @given(
        st.integers(min_value=-1000, max_value=1000),
        st.integers(min_value=-1000, max_value=0),
        st.integers(min_value=0, max_value=1000),
    )
    def test_結果は常に範囲内(self, value, min_val, max_val):
        """プロパティ: clamp の結果は必ず [min_val, max_val] の範囲内。"""
        assume(min_val <= max_val)
        result = clamp(value, min_val, max_val)
        assert min_val <= result <= max_val

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
    )
    def test_範囲内の値はそのまま返る(self, value, bound):
        """プロパティ: 既に範囲内の値は変更されない。"""
        min_val = 0
        max_val = max(value, bound)
        min_val_actual = min(0, value)
        # value が [min_val, max_val] 内なら結果は value 自身
        result = clamp(value, min_val, max_val)
        if min_val <= value <= max_val:
            assert result == value


class TestEncodeDecodeProperties:
    """encode_decode のプロパティベーステスト。"""

    @given(st.text())
    def test_ラウンドトリップ(self, text):
        """プロパティ: エンコード→デコードで元に戻る。"""
        assert encode_decode(text) == text
```

### 9.3 プロパティの見つけ方

プロパティベーステストを書く際に、どのようなプロパティを検証すべきかは
初学者がつまずきやすいポイントである。以下に代表的なパターンを示す。

| パターン | 説明 | 例 |
|---------|------|-----|
| ラウンドトリップ | encode → decode で元に戻る | JSON, Base64, 暗号化 |
| 冪等性 | 2 回実行しても結果が同じ | ソート, 正規化, フォーマット |
| 不変量 | 操作前後で保存される性質 | 要素数、合計値 |
| 単調性 | 入力が増えると出力も増える | ソート済みリストへの挿入 |
| 参照実装 | 単純だが正しい実装と比較 | 最適化版 vs ナイーブ版 |
| 逆関数 | f(g(x)) == x | push/pop, insert/delete |
| 帰納法 | 小さい入力から大きい入力への帰納 | 再帰的データ構造 |

---

## 10. テストのアンチパターン

### 10.1 アンチパターン一覧

テストコードにおいてよく見られるアンチパターンを解説する。
これらはテストの信頼性、保守性、可読性を損なう原因となる。

#### アンチパターン 1: テストの相互依存（The Order-Dependent Test）

```python
# ===== アンチパターン: テスト間の順序依存 =====

# 危険: テスト A の結果がテスト B に影響する

class SharedState:
    """グローバルに共有される状態（アンチパターン）。"""
    items = []


class TestBad_テスト間の順序依存:
    """テストの順序に依存する悪い例。"""

    def test_A_アイテムを追加(self):
        SharedState.items.append("item1")
        assert len(SharedState.items) == 1

    def test_B_アイテム数を確認(self):
        # 危険: test_A が先に実行されることを前提としている
        assert len(SharedState.items) == 1  # test_A に依存

    def test_C_アイテムを削除(self):
        SharedState.items.clear()
        assert len(SharedState.items) == 0
        # test_B の後に実行されなければ失敗するかもしれない


# ===== 改善例: 各テストが独立 =====

class TestGood_テストの独立性:
    """各テストが独立している良い例。"""

    def setup_method(self):
        """各テストの前に状態を初期化する。"""
        self.items = []

    def test_A_アイテムを追加(self):
        self.items.append("item1")
        assert len(self.items) == 1

    def test_B_空のリストのサイズは0(self):
        assert len(self.items) == 0

    def test_C_追加して削除すると空になる(self):
        self.items.append("item1")
        self.items.clear()
        assert len(self.items) == 0
```

**問題点:**
- テストの実行順序を変えると結果が変わる
- 並列実行できない
- 1 つのテストが失敗すると、後続のテストも連鎖的に失敗する

**対策:**
- 各テストの前にフィクスチャで状態を初期化する
- グローバル状態を避け、テストごとに独立したインスタンスを使う
- `pytest --randomly`（pytest-randomly プラグイン）で順序をシャッフルして検証する

#### アンチパターン 2: 氷山テスト（The Ice-Cream Cone / The Giant Test）

```python
# ===== アンチパターン: 1つのテストに多すぎるアサーション =====

def test_bad_ユーザー登録から購入まで全部テスト():
    """1 つのテストケースに多すぎる検証を詰め込む悪い例。"""
    # ユーザー登録
    user = register_user("Alice", "alice@example.com")
    assert user.id is not None
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
    assert user.is_active is True

    # ログイン
    token = login(user.email, "password123")
    assert token is not None
    assert len(token) > 0

    # 商品検索
    products = search_products("Python")
    assert len(products) > 0
    assert products[0].name == "Python入門"

    # カートに追加
    cart = add_to_cart(user.id, products[0].id, quantity=2)
    assert len(cart.items) == 1
    assert cart.total == 6000

    # 決済
    order = checkout(user.id, cart.id)
    assert order.status == "completed"
    assert order.total == 6000


# ===== 改善例: 適切な粒度に分割 =====

class TestUserRegistration:
    def test_正常な情報で登録できる(self):
        user = register_user("Alice", "alice@example.com")
        assert user.id is not None
        assert user.name == "Alice"

class TestAuthentication:
    def test_正しい認証情報でログインできる(self):
        # ...（ユーザーはフィクスチャで事前準備）
        pass

class TestShoppingCart:
    def test_商品をカートに追加できる(self):
        # ...（ユーザーとカートはフィクスチャで事前準備）
        pass

class TestCheckout:
    def test_カートの内容で決済できる(self):
        # ...
        pass
```

**問題点:**
- テストが失敗した場合、どの機能に問題があるか特定しにくい
- テスト名から何をテストしているか分からない
- 前半が失敗すると後半のテストがスキップされる

**対策:**
- 1 テスト 1 関心事を原則とする
- テスト名で何をテストしているか明確にする
- フィクスチャで前提条件を準備し、各テストは 1 つの振る舞いだけを検証する

#### アンチパターン 3: フレイキーテスト（The Flickering Test）

フレイキーテストは、同じコードに対して実行するたびに結果が変わるテストである。

**主な原因:**

| 原因 | 例 | 対策 |
|------|-----|------|
| 時刻依存 | `datetime.now()` に依存 | 時刻をインジェクション可能にする |
| 乱数依存 | ランダム値に依存する処理 | シードを固定する |
| 並行処理 | レースコンディション | 適切な同期処理を入れる |
| 外部サービス | ネットワーク遅延・障害 | モックを使用する |
| 共有リソース | 他のテストが変更した DB | テストごとにリセットする |
| タイミング依存 | `sleep(1)` の後に検証 | ポーリングまたはイベント待ちに変える |
| 環境依存 | OS、ロケール、タイムゾーン | テスト環境を固定する |

#### アンチパターン 4: テストのないリファクタリング

テストなしでコードを変更すると、回帰バグを見逃すリスクが高い。
「テストがあるからこそリファクタリングできる」のであり、
テストなしのリファクタリングは単なるギャンブルである。

**対策:**
- リファクタリング前にまずテストを書く（特にレガシーコード）
- Michael Feathers の「レガシーコード改善ガイド」のテクニックを活用する
- 特性テスト（Characterization Test）で現在の振る舞いを記録してからリファクタリングする

#### アンチパターン 5: 過度に具体的なアサーション

```python
# 悪い例: 出力全体を文字列比較
def test_bad_レポート出力():
    result = generate_report(2024, 1)
    assert result == "2024年1月の売上レポート\n売上合計: ¥1,234,567\n前月比: +5.2%\n..."
    # 改行やスペースが1つ変わるだけで失敗する

# 良い例: 重要な情報だけを検証
def test_good_レポートに売上合計が含まれる():
    result = generate_report(2024, 1)
    assert "売上合計" in result
    assert "¥1,234,567" in result
```

---

## 11. テストの設計原則

### 11.1 FIRST 原則

良いユニットテストは FIRST 原則に従う。

| 文字 | 原則 | 説明 |
|------|------|------|
| F | Fast（高速） | テストは数ミリ秒で完了すべき |
| I | Independent（独立） | テスト間に依存関係がない |
| R | Repeatable（再現可能） | どの環境でも同じ結果を返す |
| S | Self-validating（自己検証的） | テスト自身が成功/失敗を判定する |
| T | Timely（適時的） | プロダクションコードと同時に書く |

### 11.2 テストの構造パターン

#### AAA パターン（再掲・詳細）

```
AAA パターンの構造:

  def test_〇〇の場合に△△が起きる():
      # ─── Arrange（準備）──────────────
      #   テスト対象のオブジェクトを生成
      #   テストに必要なデータを用意
      #   依存オブジェクトをセットアップ
      sut = SystemUnderTest()
      input_data = create_test_data()

      # ─── Act（実行）──────────────────
      #   テスト対象の操作を 1 つだけ実行
      result = sut.do_something(input_data)

      # ─── Assert（検証）──────────────
      #   期待される結果を検証
      #   1 テスト 1 概念の検証（理想）
      assert result == expected_value
```

#### Given-When-Then パターン

BDD 寄りの記法で、AAA と本質的に同じだが、ビジネス寄りの用語を使う。

| AAA | Given-When-Then | 説明 |
|-----|----------------|------|
| Arrange | Given（前提条件） | 初期状態の設定 |
| Act | When（操作） | テスト対象の操作 |
| Assert | Then（期待結果） | 結果の検証 |

### 11.3 テストフィクスチャのベストプラクティス

```python
# pytest のフィクスチャ活用例

import pytest

@pytest.fixture
def sample_user():
    """テスト用のユーザーオブジェクトを提供する。"""
    return User(name="テスト太郎", email="test@example.com")

@pytest.fixture
def authenticated_client(sample_user):
    """認証済みの API クライアントを提供する。"""
    client = TestClient(app)
    token = create_token(sample_user)
    client.headers["Authorization"] = f"Bearer {token}"
    return client

# フィクスチャの粒度:
#   - 小さすぎる: 各テストで同じセットアップコードが重複する
#   - 大きすぎる: 不要なセットアップが含まれ、テストが遅くなる
#   - 適切: テストが何をテストしているか明確で、必要最小限のセットアップ
```

---

## 12. 特殊なテスト手法

### 12.1 ミューテーションテスト

ミューテーションテストは、プロダクションコードに意図的な変異（ミュータント）を
導入し、テストスイートがその変異を検出できるかを評価する手法である。

```
ミューテーションテストの流れ:

  元のコード:
    if age >= 18:
        return "成人"

  ミュータント 1:         ミュータント 2:
    if age > 18:            if age >= 19:
        return "成人"           return "成人"
    （>= を > に変更）       （18 を 19 に変更）

  テストスイートが:
    - ミュータントを検出（kill）→ テストの品質が高い
    - ミュータントを見逃す（survive）→ テストに漏れがある
```

Python では mutmut ツールで実行できる。

```bash
# ミューテーションテストの実行
mutmut run --paths-to-mutate=src/
mutmut results
```

### 12.2 スナップショットテスト

スナップショットテストは、関数の出力を「スナップショット」として保存し、
次回の実行時に以前のスナップショットと比較するテスト手法である。
UI コンポーネントやシリアライズされたデータの回帰テストに有用である。

```python
# pytest-snapshot を使ったスナップショットテスト

def test_レポート出力のスナップショット(snapshot):
    result = generate_report(year=2024, month=1)
    snapshot.assert_match(result, "report_2024_01.txt")
    # 初回実行時: スナップショットファイルが生成される
    # 2回目以降: 保存されたスナップショットと比較される
    # 変更時: pytest --snapshot-update で更新
```

### 12.3 コントラクトテスト

マイクロサービス間の API 連携において、
サービス間の「契約（コントラクト）」が守られているかを検証するテスト。
Pact などのツールが使われる。

```
コントラクトテストの概念:

  Consumer（利用側）         Provider（提供側）
  ┌──────────────┐         ┌──────────────┐
  │ フロントエンド │  Contract │ バックエンド  │
  │              │◄────────►│              │
  │  "GET /users │  (Pact)  │  API サーバー │
  │   を呼ぶと   │         │              │
  │   [{id, name}]│         │              │
  │   が返る"    │         │              │
  └──────────────┘         └──────────────┘

  Consumer 側: 期待するリクエスト/レスポンスをコントラクトとして定義
  Provider 側: コントラクトに従ってレスポンスを返せるかを検証
  → 両者が独立してテスト可能。デプロイ前に互換性を確認できる
```

---

## 13. テストフレームワークとツールの比較

### 13.1 Python テストフレームワーク比較

| フレームワーク | 特徴 | テスト記述スタイル | フィクスチャ | パラメータ化 | プラグイン |
|---|---|---|---|---|---|
| pytest | Python で最も広く使われるテストフレームワーク | 関数ベース + クラスベース | `@pytest.fixture`（強力で柔軟） | `@pytest.mark.parametrize` | 1000 以上のプラグイン |
| unittest | Python 標準ライブラリ同梱 | クラスベース（`TestCase` 継承） | `setUp` / `tearDown` | `subTest` | 限定的 |
| nose2 | unittest の拡張（nose の後継） | 関数ベース + クラスベース | プラグインベース | パラメータプラグイン | 中程度 |
| doctest | ドキュメント文字列内にテストを記述 | docstring 内のインタラクティブ例 | なし | なし | なし |

**推奨**: 新規プロジェクトでは **pytest** を第一選択とする。
豊富なプラグインエコシステム、直感的なフィクスチャ機構、
分かりやすいアサーションの失敗メッセージが強みである。

### 13.2 主要言語のテストフレームワーク

| 言語 | フレームワーク | 特徴 |
|------|--------------|------|
| Python | pytest | 関数ベース、強力なフィクスチャ、豊富なプラグイン |
| JavaScript/TypeScript | Jest | Meta 製。スナップショットテスト、モック内蔵 |
| JavaScript/TypeScript | Vitest | Vite ベース。ESM ネイティブ、Jest 互換 API |
| Java | JUnit 5 | アノテーション駆動。パラメータ化テストが強力 |
| Go | testing (標準) | 標準ライブラリで完結。`go test` コマンド |
| Rust | cargo test (標準) | `#[test]` アトリビュート。ドキュメントテスト対応 |
| C# | xUnit.net | .NET の標準的フレームワーク。`[Fact]`, `[Theory]` |
| Ruby | RSpec | BDD スタイル。`describe`, `it` ブロック |

### 13.3 テスト支援ツール

#### モック/スタブ

| ツール | 言語 | 特徴 |
|--------|------|------|
| unittest.mock | Python | 標準ライブラリ。`Mock`, `patch`, `MagicMock` |
| pytest-mock | Python | unittest.mock の pytest ラッパー。`mocker` フィクスチャ |
| responses | Python | requests ライブラリの HTTP モック |
| Mockito | Java | Java の代表的モックライブラリ |
| testdouble.js | JavaScript | JavaScript のテストダブルライブラリ |

#### E2E / ブラウザ自動化

| ツール | 対応ブラウザ | 特徴 |
|--------|------------|------|
| Playwright | Chromium, Firefox, WebKit | Microsoft 製。複数ブラウザ対応、自動待機 |
| Cypress | Chromium ベース | JavaScript ネイティブ。タイムトラベルデバッグ |
| Selenium | 全主要ブラウザ | 最も歴史が長い。WebDriver プロトコル |

#### カバレッジ

| ツール | 言語 | 出力形式 |
|--------|------|---------|
| pytest-cov (coverage.py) | Python | HTML, XML, JSON, ターミナル |
| Istanbul (nyc) | JavaScript | HTML, lcov, text |
| JaCoCo | Java | HTML, XML, CSV |
| gcov / lcov | C/C++ | HTML, テキスト |

#### プロパティベーステスト

| ツール | 言語 | 特徴 |
|--------|------|------|
| Hypothesis | Python | 強力な shrinking、stateful テスト対応 |
| QuickCheck | Haskell | プロパティベーステストの元祖 |
| fast-check | JavaScript/TypeScript | JS/TS 向け。Hypothesis インスパイア |
| PropTest | Rust | Rust 向けプロパティベーステスト |
| jqwik | Java | JUnit 5 統合型プロパティベーステスト |

#### ミューテーションテスト

| ツール | 言語 | 特徴 |
|--------|------|------|
| mutmut | Python | シンプルで使いやすい |
| cosmic-ray | Python | より多くのミュータントオペレータ |
| Stryker | JS/TS, C# | 複数言語対応。リッチなレポート |
| PIT (pitest) | Java | Java の代表的ミューテーションテストツール |

### 13.4 テスト実行の最適化テクニック

テストスイートが成長すると実行時間が問題になる。
以下に主要な最適化テクニックを示す。

```
テスト実行の最適化戦略:

  ┌─────────────────────────────────────────────────────┐
  │              テスト実行の高速化                        │
  ├─────────────────────────────────────────────────────┤
  │                                                     │
  │  1. 並列実行                                         │
  │     pytest-xdist: pytest -n auto                    │
  │     → CPU コア数に応じて自動並列化                     │
  │                                                     │
  │  2. 変更検知ベースの実行                               │
  │     pytest --lf  (前回失敗したテストだけ再実行)         │
  │     pytest --ff  (前回失敗したテストを優先的に実行)      │
  │     pytest-testmon (変更されたコードに関連するテストのみ) │
  │                                                     │
  │  3. 層別実行                                         │
  │     pytest -m "not slow"  (遅いテストをスキップ)       │
  │     pytest -m "unit"      (ユニットテストだけ)          │
  │     pytest -m "smoke"     (スモークテストだけ)          │
  │                                                     │
  │  4. フィクスチャの最適化                               │
  │     scope="session" → テストセッション全体で1回だけ実行  │
  │     scope="module"  → モジュールごとに1回               │
  │     scope="class"   → クラスごとに1回                   │
  │     scope="function"→ テスト関数ごとに1回（デフォルト）   │
  │                                                     │
  │  5. 不要なI/Oの削減                                   │
  │     インメモリDB（SQLite :memory:）の活用              │
  │     ファイルシステムの代わりに StringIO / BytesIO       │
  │     HTTP 通信のモック化                                │
  └─────────────────────────────────────────────────────┘
```

```python
# pytest マーカーによる層別実行の例

import pytest

# テストにマーカーを付与
@pytest.mark.unit
def test_高速なユニットテスト():
    assert 1 + 1 == 2

@pytest.mark.integration
def test_DB接続を伴うテスト():
    # DB に接続するテスト
    pass

@pytest.mark.slow
def test_時間のかかるテスト():
    # 実行に数十秒かかるテスト
    pass

@pytest.mark.e2e
def test_ブラウザ操作テスト():
    # Playwright でブラウザを操作するテスト
    pass

# pyproject.toml での設定:
# [tool.pytest.ini_options]
# markers = [
#     "unit: ユニットテスト",
#     "integration: 統合テスト",
#     "slow: 実行が遅いテスト",
#     "e2e: E2E テスト",
# ]

# 実行例:
# pytest -m unit           → ユニットテストだけ
# pytest -m "not slow"     → 遅いテスト以外
# pytest -m "unit or integration"  → ユニット + 統合
```

### 13.5 テストデータの管理

テストデータの管理はテストの信頼性と保守性に直結する。

#### ファクトリーパターン

```python
# テストデータのファクトリーパターン

from dataclasses import dataclass, field
import uuid


@dataclass
class UserFactory:
    """テスト用のユーザーデータを生成するファクトリー。"""

    name: str = "テスト太郎"
    email: str = field(default_factory=lambda: f"test-{uuid.uuid4().hex[:8]}@example.com")
    age: int = 30
    is_active: bool = True

    def build(self) -> dict:
        """辞書形式でユーザーデータを返す。"""
        return {
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "is_active": self.is_active,
        }

    @classmethod
    def admin(cls) -> "UserFactory":
        """管理者ユーザーのプリセット。"""
        return cls(name="管理者", age=40)

    @classmethod
    def child(cls) -> "UserFactory":
        """子供ユーザーのプリセット。"""
        return cls(name="テスト子供", age=10)


# 使用例
def test_デフォルトユーザーで登録できる():
    user_data = UserFactory().build()
    result = register_user(**user_data)
    assert result.name == "テスト太郎"

def test_管理者で登録できる():
    user_data = UserFactory.admin().build()
    result = register_user(**user_data)
    assert result.name == "管理者"

def test_カスタムデータで登録できる():
    user_data = UserFactory(name="カスタム", age=25).build()
    result = register_user(**user_data)
    assert result.name == "カスタム"
```

#### ビルダーパターン

```python
# テストデータのビルダーパターン

class OrderBuilder:
    """テスト用の注文データをビルダーパターンで生成する。"""

    def __init__(self):
        self._customer_id = "C001"
        self._items = []
        self._discount = 0
        self._shipping_address = "東京都千代田区"

    def with_customer(self, customer_id: str) -> "OrderBuilder":
        self._customer_id = customer_id
        return self

    def with_item(self, name: str, price: int, quantity: int = 1) -> "OrderBuilder":
        self._items.append({"name": name, "price": price, "quantity": quantity})
        return self

    def with_discount(self, discount: int) -> "OrderBuilder":
        self._discount = discount
        return self

    def with_shipping_address(self, address: str) -> "OrderBuilder":
        self._shipping_address = address
        return self

    def build(self) -> dict:
        return {
            "customer_id": self._customer_id,
            "items": self._items,
            "discount": self._discount,
            "shipping_address": self._shipping_address,
        }


# 使用例
def test_複数商品の注文合計():
    order = (
        OrderBuilder()
        .with_item("Python入門", 3000, quantity=2)
        .with_item("Go実践", 3500)
        .with_discount(500)
        .build()
    )
    total = calculate_order_total(order)
    assert total == 9000  # (3000*2 + 3500) - 500
```

---

## 14. テストと設計の関係

### 14.1 テスタビリティと設計品質

テストのしやすさ（テスタビリティ）は、設計品質の優れた指標である。
テストしにくいコードは、ほぼ確実に設計上の問題を抱えている。

```
テスタビリティと設計の関係:

  テストしにくいコードの特徴        対応する設計上の問題
  ──────────────────────        ──────────────────
  ・new で直接依存を生成          → 密結合（Tight Coupling）
  ・グローバル変数に依存          → 隠れた依存関係
  ・static メソッドが多い         → テストダブルで置換不能
  ・1メソッドが数百行             → 単一責任原則の違反
  ・コンストラクタで副作用        → 生成と利用の混在
  ・環境変数に直接アクセス        → 設定の暗黙的依存

  テストしやすいコードの特徴        対応する設計原則
  ──────────────────────        ──────────────────
  ・依存はコンストラクタで注入     → 依存性逆転原則（DIP）
  ・インターフェースに依存        → 開放閉鎖原則（OCP）
  ・メソッドは短く単一目的        → 単一責任原則（SRP）
  ・副作用が少ない純粋関数       → 関数型プログラミング
  ・設定は引数で受け取る         → 明示的な依存関係
```

### 14.2 依存性注入とテスタビリティ

```python
# ===== コード例 6: 依存性注入によるテスタビリティの向上 =====

# 悪い例: 直接依存を生成
class NotificationService_Bad:
    def notify(self, user_id: str, message: str) -> bool:
        # テスト時にもメールが送信されてしまう
        import smtplib
        server = smtplib.SMTP("smtp.example.com")
        server.sendmail("noreply@example.com", user_id, message)
        return True


# 良い例: 依存性注入
from typing import Protocol

class EmailSender(Protocol):
    def send(self, to: str, subject: str, body: str) -> bool: ...

class NotificationService_Good:
    def __init__(self, email_sender: EmailSender):
        self._email_sender = email_sender

    def notify(self, user_id: str, message: str) -> bool:
        return self._email_sender.send(
            to=user_id,
            subject="通知",
            body=message,
        )


# テストコード
class FakeEmailSender:
    """テスト用のフェイク実装。"""
    def __init__(self):
        self.sent_emails = []

    def send(self, to: str, subject: str, body: str) -> bool:
        self.sent_emails.append({"to": to, "subject": subject, "body": body})
        return True


def test_通知が送信される():
    fake_sender = FakeEmailSender()
    service = NotificationService_Good(fake_sender)

    result = service.notify("user@example.com", "テストメッセージ")

    assert result is True
    assert len(fake_sender.sent_emails) == 1
    assert fake_sender.sent_emails[0]["to"] == "user@example.com"
```

### 14.3 Hexagonal Architecture とテスト

Hexagonal Architecture（ポート&アダプターアーキテクチャ）は、
テスタビリティに優れた設計パターンである。

```
Hexagonal Architecture とテスト戦略:

                    ┌──────────────────────┐
                    │    テスト戦略         │
                    └──────────────────────┘

  ┌─────────┐                               ┌─────────┐
  │ アダプター│    ┌────────────────────┐     │ アダプター│
  │ (入力)   │───>│   ポート（入力）     │     │ (出力)   │
  │ HTTP API │    │                    │     │ DB      │
  │ CLI      │    │  ┌──────────────┐  │     │ メール   │
  │ メッセージ│    │  │ ドメイン      │  │───> │ 外部API  │
  │          │    │  │ ロジック      │  │     │          │
  │          │    │  └──────────────┘  │     │          │
  └─────────┘    │   ポート（出力）     │     └─────────┘
                  └────────────────────┘

  テスト戦略:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ユニットテスト → ドメインロジック
    ・外部依存なし。純粋なビジネスルールを検証
    ・ポートをモック/スタブで置換

  統合テスト → アダプター
    ・実際のDBやHTTPサーバーで接続を検証
    ・ポートの実装が正しく動作するか

  E2Eテスト → 入力アダプター → ドメイン → 出力アダプター
    ・全レイヤーを貫通。主要フローのみ
```

---

## 15. 演習問題

### 演習 1（初級）: ユニットテストの実装

以下の `StringCalculator` クラスに対して、pytest を使ったユニットテストを作成せよ。

```python
class StringCalculator:
    """文字列形式の数値を計算する。"""

    def add(self, numbers: str) -> int:
        """カンマ区切りの数値文字列を受け取り、合計を返す。

        ルール:
        - 空文字列の場合は 0 を返す
        - 数値が 1 つの場合はその数値を返す
        - カンマ区切りの複数数値の合計を返す
        - 改行もデリミタとして扱う（"1\n2,3" → 6）
        - 負の数が含まれる場合は ValueError を発生させる
          （エラーメッセージに負の数を含める）
        - 1000 より大きい数は無視する（"2,1001" → 2）
        """
        if not numbers:
            return 0

        delimiters = [",", "\n"]
        for d in delimiters:
            numbers = numbers.replace(d, ",")

        values = [int(x) for x in numbers.split(",")]

        negatives = [v for v in values if v < 0]
        if negatives:
            raise ValueError(f"負の数は許可されていない: {negatives}")

        return sum(v for v in values if v <= 1000)
```

**要件:**
- 正常系テストを 5 件以上
- 異常系テストを 2 件以上
- 境界値テストを 3 件以上
- `@pytest.mark.parametrize` を少なくとも 1 箇所使用

### 演習 2（中級）: TDD で Stack を実装

以下の仕様を満たす `Stack` クラスを TDD で実装せよ。

**仕様:**
1. `push(item)` — アイテムをスタックの最上部に追加する
2. `pop()` — 最上部のアイテムを取り出して返す。空なら `IndexError`
3. `peek()` — 最上部のアイテムを返すが、取り出さない。空なら `IndexError`
4. `is_empty()` — スタックが空なら `True`
5. `size()` — スタック内のアイテム数を返す
6. `max_size` — コンストラクタで指定。`push` 時にサイズ超過なら `OverflowError`

**手順:**
1. まず `is_empty` のテストを書き、実装する（Red → Green）
2. 次に `push` と `size` のテストを書き、実装する
3. `pop` のテスト（正常系・異常系）を書き、実装する
4. `peek` のテストを書き、実装する
5. `max_size` のテストを書き、実装する
6. 全テストが通った状態でリファクタリングする

### 演習 3（上級）: プロパティベーステストの設計

以下の関数群に対して、Hypothesis を使ったプロパティベーステストを設計・実装せよ。

```python
import json
from typing import Any


def json_round_trip(data: dict) -> dict:
    """JSON シリアライズ → デシリアライズのラウンドトリップ。"""
    return json.loads(json.dumps(data))


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """ネストされた辞書をフラットにする。
    例: {"a": {"b": 1}} → {"a.b": 1}
    """
    result = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, new_key))
        else:
            result[new_key] = value
    return result


def compact(lst: list) -> list:
    """リストから None と空文字列を除去する。"""
    return [x for x in lst if x is not None and x != ""]
```

**要件:**
- 各関数に対して 2 つ以上のプロパティを定義する
- `st.dictionaries`, `st.recursive` などの高度な戦略を使用する
- `@example` デコレータでエッジケースを明示する

---

## 16. FAQ（よくある質問）

### Q1: テストカバレッジは何パーセントを目標にすべきか？

一般的には **80% 以上**が推奨される目安である。ただし、カバレッジの数値だけを
追い求めるのは危険である。重要なのは以下の 3 点である。

1. **ビジネスクリティカルな部分**（決済処理、認証、データ変換など）は高いカバレッジを維持する
2. **カバレッジの低い部分**を定期的にレビューし、テストが本当に不要かを判断する
3. **カバレッジの高さよりも、テストの品質**（適切なアサーション、エッジケースの網羅）を重視する

カバレッジが 100% でもバグはゼロにならない（前述のとおり）。
カバレッジは「テストされていない部分を発見するための指標」として活用し、
テストの品質評価には別の手法（ミューテーションテストなど）を併用すべきである。

### Q2: テストが遅くて CI のフィードバックが遅い。どう改善すべきか？

以下のアプローチを段階的に適用する。

**短期的な対策:**
1. **テストの並列実行**: `pytest-xdist` で `pytest -n auto` を使う
2. **遅いテストの特定**: `pytest --durations=20` で遅いテスト Top 20 を確認する
3. **不要な E2E テストの削減**: ユニットテストで代替できるものを移行する
4. **フィクスチャの最適化**: セットアップの重複を排除する

**中期的な対策:**
1. **テストの層別実行**: PR では高速なユニットテストのみ、マージ後に全テストを実行
2. **変更検知**: 変更されたファイルに関連するテストだけを実行する
3. **テスト用 DB の最適化**: PostgreSQL より SQLite（テスト用）、またはインメモリ DB

**長期的な対策:**
1. **テストピラミッドの再構築**: E2E が多すぎる場合、下位レイヤーに移行する
2. **テストインフラの改善**: キャッシュ、並列化、分散実行
3. **テストアーキテクチャの見直し**: Hexagonal Architecture で外部依存を隔離

### Q3: レガシーコードにテストを追加するにはどうすればよいか？

Michael Feathers の「レガシーコード改善ガイド」に基づくアプローチを推奨する。

**ステップ 1: 変更点を特定する**
- 変更が必要な箇所を特定し、そこに関連するコードの依存関係を把握する

**ステップ 2: テストハーネスを確立する**
- 変更箇所をテスト可能にするための最小限のリファクタリングを行う
- 依存関係を切り離す技法:
  - Extract Interface（インターフェースの抽出）
  - Parameterize Constructor（コンストラクタのパラメータ化）
  - Wrap Method（メソッドのラップ）

**ステップ 3: 特性テストを書く**
- 現在の振る舞い（仕様どおりかどうかは問わない）を記録するテストを書く
- これにより、リファクタリング時に意図しない変更を検出できる

**ステップ 4: 変更を加える**
- テストの安全網のもとで、必要な変更を加える

**ステップ 5: テストを改善する**
- 特性テストを正しい仕様に基づくテストに徐々に置き換えていく

### Q4: モックはどの程度使うべきか？

モックの使用量は「テストの種類」と「テスト対象の依存関係」によって異なる。

**モックすべきもの:**
- 外部 API への HTTP リクエスト
- 時刻、乱数などの非決定的な要素
- 送信系の処理（メール送信、Push 通知など）
- テスト環境に存在しない外部サービス

**モックすべきでないもの:**
- テスト対象自身のメソッド（テストの意味がなくなる）
- 単純なバリューオブジェクト
- 標準ライブラリの基本機能

**判断基準:**
- 「このモックを外したら、テストはまだ意味があるか？」を問う
- モックの設定コードがプロダクションコードより長い場合は、設計を見直す

### Q5: E2E テストと統合テストの境界はどこにあるか？

明確な境界は組織やプロジェクトによって異なるが、一般的な目安は以下のとおりである。

| 基準 | 統合テスト | E2E テスト |
|------|-----------|-----------|
| UI の関与 | なし（API レイヤーまで） | あり（ブラウザ操作を含む） |
| テスト対象 | 2〜3 コンポーネントの連携 | システム全体のフロー |
| データ | テスト専用 DB に直接準備 | UI からの入力で準備 |
| 実行速度 | 秒〜分 | 分〜十分 |
| 壊れやすさ | 中程度 | 高い |

実務では「UI を伴うかどうか」を境界とすることが多い。
API に対する統合テストは十分に安定しており、
ブラウザ操作を伴う E2E テストはコストが高いという認識が広まっている。

---

## 17. テスト戦略の実践的ガイドライン

### 15.1 テストを書く順序

新しい機能を開発する際のテスト作成順序の指針を示す。

```
テスト作成の推奨順序:

  1. ドメインロジックのユニットテスト
     └─ ビジネスルール、バリデーション、計算処理
         最も重要で、最もテストしやすい

  2. サービス層のユニットテスト
     └─ ドメインロジックの組み合わせ、外部依存のモック

  3. リポジトリ/データアクセスの統合テスト
     └─ DB とのやり取りの正確性

  4. API エンドポイントの統合テスト
     └─ リクエスト/レスポンスの形式、ステータスコード

  5. 重要なユーザーフローの E2E テスト
     └─ ビジネス上クリティカルなシナリオのみ
```

### 15.2 テストのメンテナンス

テストコードもプロダクションコードと同様にメンテナンスが必要である。

**定期的に行うべきこと:**
1. **遅いテストの改善** — `pytest --durations` で定期計測
2. **フレイキーテストの修正** — 不安定なテストを放置しない
3. **不要なテストの削除** — 仕様変更で不要になったテスト、重複テスト
4. **テスト用ユーティリティの整理** — ファクトリー、ビルダー、共通フィクスチャの最適化
5. **カバレッジレポートの確認** — テストされていない箇所の把握

### 15.3 テストコードの品質基準

テストコードにもコーディング規約を適用する。

| 基準 | 内容 |
|------|------|
| 可読性 | テストを読むだけで仕様が分かる |
| 命名 | テスト名が何を検証しているか明確 |
| 構造 | AAA パターンに従い、各セクションが明確に分離 |
| DRY | フィクスチャでセットアップを共通化（ただしテストの可読性を損なわない範囲で） |
| 独立性 | テスト間の依存がない |
| 速度 | ユニットテストは ms 単位 |
| 安定性 | フレイキーでない |

---

## 16. まとめ

### テスト分類の全体像

| テストレベル | 対象 | 速度 | コスト | 推奨比率 |
|------------|------|------|--------|---------|
| ユニットテスト | 関数・クラス | ms | 低 | 70% |
| 統合テスト | コンポーネント連携 | 秒〜分 | 中 | 20% |
| E2E テスト | システム全体 | 分〜十分 | 高 | 10% |
| 受入テスト | ビジネス要件 | 分〜十分 | 高 | スプリントごと |

### 開発手法の比較

| 手法 | 核心 | ツール | 適用場面 |
|------|------|--------|---------|
| TDD | テストファーストで設計を駆動 | pytest, JUnit | ロジック実装 |
| BDD | ビジネスシナリオを自動テスト化 | Cucumber, Behave | 要件の合意 |
| プロパティベーステスト | ランダム入力で不変量を検証 | Hypothesis, QuickCheck | 汎用ロジック |

### テスト技法の使い分け

| 技法 | 用途 | 効果 |
|------|------|------|
| 同値分割 | 入力ドメインのグループ化 | テストケース数の削減 |
| 境界値分析 | off-by-one エラーの検出 | 境界バグの早期発見 |
| デシジョンテーブル | 複合条件の網羅 | 条件漏れの防止 |
| ペアワイズテスト | 多因子の組み合わせ削減 | テスト効率の最大化 |

### テスト設計の原則

```
テスト設計チェックリスト:

  [ ] FIRST 原則（Fast, Independent, Repeatable, Self-validating, Timely）
  [ ] AAA パターン（Arrange, Act, Assert）
  [ ] 1 テスト 1 概念
  [ ] テスト名が仕様を表現している
  [ ] テストピラミッドのバランスが取れている
  [ ] カバレッジが 80% 以上
  [ ] フレイキーテストがない
  [ ] CI で自動実行されている
```

---

## 次に読むべきガイド

- [[02-design-patterns.md]] — デザインパターン（テスタブルな設計を学ぶ）
- [[03-refactoring.md]] — リファクタリング（テストの安全網を活かして改善する）

---

## 参考文献

1. Beck, K. *Test Driven Development: By Example*. Addison-Wesley, 2002.
   — TDD の原典。Red-Green-Refactor の基本サイクルと実践例を網羅。
2. Feathers, M. *Working Effectively with Legacy Code*. Prentice Hall, 2004.
   — テストのないコード（レガシーコード）にテストを追加するための実践的テクニック。
3. Freeman, S., Pryce, N. *Growing Object-Oriented Software, Guided by Tests*. Addison-Wesley, 2009.
   — モックを活用したテスト駆動設計の名著。ロンドン学派 TDD の代表作。
4. Meszaros, G. *xUnit Test Patterns: Refactoring Test Code*. Addison-Wesley, 2007.
   — テストダブルの分類（ダミー、スタブ、スパイ、モック、フェイク）を体系化。
5. Cohn, M. *Succeeding with Agile*. Addison-Wesley, 2009.
   — テストピラミッドの原典。アジャイル開発におけるテスト戦略を解説。
6. Claessen, K., Hughes, J. "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs." ICFP, 2000.
   — プロパティベーステストの原論文。Hypothesis はこの思想を Python に移植したもの。
7. MacLeod, D. *Hypothesis documentation*. https://hypothesis.readthedocs.io/
   — Python のプロパティベーステストフレームワーク Hypothesis の公式ドキュメント。

---

> テストは品質を作り込むものではない。テストは品質を可視化するものである。
> 品質はテスト可能な設計から生まれる。
