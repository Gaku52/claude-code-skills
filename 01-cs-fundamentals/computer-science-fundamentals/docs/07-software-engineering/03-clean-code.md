# クリーンコード

> コードは書く時間より読む時間の方が10倍長い。読みやすいコードは正しいコードへの最短距離である。

## この章で学ぶこと

- [ ] 良い命名の原則を身につける
- [ ] 関数設計の原則を理解する
- [ ] コードの臭い（Code Smell）を認識できる
- [ ] SOLID原則を理解し実践できる
- [ ] DRY / KISS / YAGNI を正しく適用できる
- [ ] コメントの書き方と不要なコメントの見極め方を学ぶ
- [ ] エラー処理のベストプラクティスを習得する
- [ ] テスタビリティの高いコードを書けるようになる
- [ ] リファクタリングの手法と判断基準を身につける
- [ ] コードレビューで品質を向上させる方法を学ぶ

---

## 1. 命名

### 1.1 命名の基本原則

```python
# ❌ 悪い命名
d = 30          # 何の日数？
lst = []        # 何のリスト？
def proc(x):    # 何の処理？
    pass

# ✅ 良い命名
trial_period_days = 30
active_users = []
def calculate_monthly_revenue(transactions):
    pass

# 命名の原則:
# 1. 意図を明確に: is_active, has_permission, should_retry
# 2. 発音可能に: ❌ genymdhms → ✅ generation_timestamp
# 3. 検索可能に: ❌ 7 → ✅ MAX_RETRY_COUNT = 7
# 4. スコープに比例: ループ変数は短く(i)、グローバルは長く
# 5. 一貫性: get/fetch/retrieve を混在させない
```

### 1.2 命名パターン集

```python
# --- ブール変数の命名 ---
# is_, has_, can_, should_, will_ のプレフィックスを使う
is_active = True
has_permission = False
can_edit = True
should_retry = False
will_expire = True

# ❌ 曖昧なブール名
flag = True        # 何のフラグ？
status = False     # ステータスは値が必要では？
check = True       # チェックの結果？チェックすべき？

# ✅ 明確なブール名
is_email_verified = True
has_admin_role = False
can_access_dashboard = True
should_send_notification = False

# --- コレクション型の命名 ---
# 複数形を使う
users = [user1, user2, user3]
email_addresses = ["a@example.com", "b@example.com"]
order_items = []

# ❌ 単数形のコレクション
user_list = []     # "list" は型情報であり冗長
email_array = []   # "array" も同様

# ✅ Map/Dictの命名は「key_to_value」または「value_by_key」
user_by_id = {"u001": user1, "u002": user2}
price_by_product = {"apple": 100, "banana": 200}
email_to_user = {"a@example.com": user1}

# --- 関数名のパターン ---
# 動詞 + 名詞
def calculate_total_price(items):
    """合計金額を計算する"""
    pass

def validate_email_address(email):
    """メールアドレスを検証する"""
    pass

def send_welcome_email(user):
    """ウェルカムメールを送信する"""
    pass

def parse_csv_file(file_path):
    """CSVファイルを解析する"""
    pass

# --- ファクトリメソッドの命名 ---
def create_user(name, email):
    """新しいユーザーを作成する"""
    pass

def from_json(json_string):
    """JSONから生成する"""
    pass

def build_query(params):
    """クエリを構築する"""
    pass
```

### 1.3 命名のアンチパターン

```python
# --- 1. ハンガリアン記法（現代では不要） ---
# ❌ 型をプレフィックスに含める
str_name = "Alice"
int_age = 30
lst_users = []
dict_config = {}

# ✅ 型は型ヒントで表現する
name: str = "Alice"
age: int = 30
users: list[User] = []
config: dict[str, Any] = {}

# --- 2. 略語の乱用 ---
# ❌ 意味不明な略語
def calc_avg_rev_per_usr(txns):
    pass

# ✅ 完全な単語を使う
def calculate_average_revenue_per_user(transactions):
    pass

# ただし一般的な略語はOK
# id, url, http, api, db, io, cpu, os, ui, html, css
user_id = "u001"
api_url = "https://api.example.com"
db_connection = get_connection()

# --- 3. 二重否定 ---
# ❌ 理解しにくい
if not is_not_active:
    pass

if not disable_feature:
    pass

# ✅ 肯定形を使う
if is_active:
    pass

if enable_feature:
    pass

# --- 4. 汎用的すぎる名前 ---
# ❌ 具体性がない
data = fetch_data()
result = process(data)
temp = calculate(result)
info = get_info()
manager = create_manager()

# ✅ 具体的な名前
user_profiles = fetch_user_profiles()
monthly_revenue = calculate_revenue(user_profiles)
formatted_report = format_revenue_report(monthly_revenue)
server_health_info = get_server_health()
connection_pool_manager = create_connection_pool()
```

### 1.4 命名の一貫性ルール

```python
# プロジェクト内で命名規則を統一する

# --- 同じ概念には同じ動詞を使う ---
# ❌ 混在
def get_user(user_id): pass
def fetch_order(order_id): pass
def retrieve_product(product_id): pass
def obtain_payment(payment_id): pass

# ✅ 統一
def get_user(user_id): pass
def get_order(order_id): pass
def get_product(product_id): pass
def get_payment(payment_id): pass

# --- 対称的な名前を使う ---
# open / close
# start / stop
# begin / end
# insert / delete
# add / remove
# create / destroy
# lock / unlock
# source / target
# first / last
# min / max
# next / previous
# up / down
# show / hide
# enable / disable
# increment / decrement

# --- 命名規約をチーム共有する例 ---
"""
命名規約ドキュメント:
1. クラス名: PascalCase (UserProfile, OrderItem)
2. 関数・変数名: snake_case (calculate_total, user_name)
3. 定数: UPPER_SNAKE_CASE (MAX_RETRY_COUNT, API_BASE_URL)
4. プライベート: _先頭アンダースコア (_internal_method)
5. ブール: is_, has_, can_, should_ プレフィックス
6. コレクション: 複数形 (users, items, orders)
7. 辞書: value_by_key 形式 (user_by_id)
8. イベントハンドラ: on_ プレフィックス (on_click, on_submit)
9. コールバック: _callback サフィックス (success_callback)
10. テスト: test_ プレフィックス (test_create_user)
"""
```

---

## 2. 関数設計

### 2.1 関数設計の基本原則

```python
# 原則: 1関数1責任、短く、引数は少なく

# ❌ 長い関数（複数の責任）
def process_order(order):
    # バリデーション(20行)...
    # 在庫確認(15行)...
    # 決済処理(25行)...
    # メール送信(10行)...
    # ログ記録(5行)...
    pass  # 75行の巨大関数

# ✅ 分割された関数
def process_order(order):
    validate_order(order)
    check_inventory(order.items)
    charge_payment(order.payment)
    send_confirmation_email(order.customer)
    log_order(order)

# 引数の原則:
# 0個（ニラディック）: 理想的
# 1個（モナディック）: 良い
# 2個（ダイアディック）: 許容
# 3個以上: オブジェクトにまとめることを検討
```

### 2.2 関数の抽象度レベル

```python
# 1つの関数内では同じ抽象度レベルを維持する

# ❌ 抽象度が混在
def generate_report(data):
    # 高レベル: データ検証
    if not data:
        raise ValueError("Data is empty")

    # 低レベル: HTMLの直接操作
    html = "<html><head><title>Report</title></head><body>"
    html += "<table>"
    for row in data:
        html += "<tr>"
        for cell in row:
            html += f"<td>{cell}</td>"
        html += "</tr>"
    html += "</table></body></html>"

    # 高レベル: メール送信
    send_email("admin@example.com", "Report", html)
    return html

# ✅ 同じ抽象度レベル
def generate_report(data):
    validate_report_data(data)
    html = build_report_html(data)
    distribute_report(html)
    return html

def validate_report_data(data):
    if not data:
        raise ValueError("Data is empty")
    if not all(isinstance(row, list) for row in data):
        raise TypeError("Data must be a list of lists")

def build_report_html(data):
    header = create_html_header("Report")
    body = create_html_table(data)
    return wrap_in_html(header, body)

def distribute_report(html):
    send_email("admin@example.com", "Report", html)
```

### 2.3 引数設計のベストプラクティス

```python
from dataclasses import dataclass
from typing import Optional

# --- 引数が多すぎる関数 ---
# ❌ 引数が多い
def create_user(name, email, age, address, phone,
                role, department, manager_id,
                start_date, salary):
    pass

# ✅ パラメータオブジェクトを使う
@dataclass
class UserCreationParams:
    name: str
    email: str
    age: int
    address: str
    phone: str
    role: str = "member"
    department: str = "general"
    manager_id: Optional[str] = None
    start_date: Optional[str] = None
    salary: Optional[float] = None

def create_user(params: UserCreationParams):
    # params.name, params.email 等でアクセス
    pass

# --- フラグ引数を避ける ---
# ❌ ブール引数で分岐
def get_users(include_inactive: bool):
    if include_inactive:
        return get_all_users()
    else:
        return get_active_users()

# ✅ 関数を分割
def get_active_users():
    return [u for u in all_users if u.is_active]

def get_all_users():
    return all_users

# --- 出力引数を避ける ---
# ❌ 引数を変更する
def add_to_list(items, new_item):
    items.append(new_item)  # 副作用

# ✅ 新しい値を返す
def add_to_list(items, new_item):
    return [*items, new_item]

# --- デフォルト引数の活用 ---
def connect_to_database(
    host: str = "localhost",
    port: int = 5432,
    database: str = "app",
    timeout_seconds: int = 30,
    max_retries: int = 3
):
    """必要な引数だけ指定すればよい"""
    pass

# 使用例: デフォルトで十分な場合
connect_to_database()

# 一部だけ変更
connect_to_database(host="production-db.example.com", timeout_seconds=60)
```

### 2.4 純粋関数と副作用

```python
# --- 純粋関数: 同じ入力に対して常に同じ出力、副作用なし ---
# ✅ 純粋関数の例
def calculate_tax(price: float, tax_rate: float) -> float:
    return price * tax_rate

def format_full_name(first_name: str, last_name: str) -> str:
    return f"{last_name} {first_name}"

def filter_active_users(users: list) -> list:
    return [u for u in users if u.is_active]

# --- 副作用を持つ関数: 名前で明示する ---
# ✅ 副作用があることが名前から分かる
def save_user_to_database(user):
    """データベースに保存する（副作用）"""
    db.users.insert(user.to_dict())

def send_notification_email(user, message):
    """メールを送信する（副作用）"""
    email_service.send(user.email, message)

def log_access(user_id, resource):
    """アクセスログを記録する（副作用）"""
    logger.info(f"User {user_id} accessed {resource}")

# --- コマンドとクエリの分離（CQS） ---
# ❌ コマンドとクエリが混在
class UserService:
    def get_and_update_last_login(self, user_id):
        """ユーザーを取得しつつ最終ログイン日時も更新する"""
        user = self.db.find(user_id)
        user.last_login = datetime.now()
        self.db.save(user)
        return user  # 取得（クエリ）と更新（コマンド）が混在

# ✅ コマンドとクエリを分離
class UserService:
    def get_user(self, user_id):
        """ユーザーを取得する（クエリ）"""
        return self.db.find(user_id)

    def update_last_login(self, user_id):
        """最終ログイン日時を更新する（コマンド）"""
        user = self.db.find(user_id)
        user.last_login = datetime.now()
        self.db.save(user)
```

### 2.5 関数の長さと複雑度

```python
# --- サイクロマティック複雑度を低く保つ ---

# ❌ 複雑度が高い（分岐が多すぎる）
def calculate_discount(user, order):
    discount = 0
    if user.is_premium:
        if order.total > 10000:
            if user.years_of_membership > 5:
                discount = 0.20
            elif user.years_of_membership > 2:
                discount = 0.15
            else:
                discount = 0.10
        elif order.total > 5000:
            if user.years_of_membership > 5:
                discount = 0.15
            else:
                discount = 0.10
        else:
            discount = 0.05
    else:
        if order.total > 10000:
            discount = 0.05
        elif order.total > 5000:
            discount = 0.03
    return discount

# ✅ テーブル駆動で複雑度を削減
DISCOUNT_TABLE = {
    # (is_premium, min_total, min_years): discount_rate
    (True,  10000, 5): 0.20,
    (True,  10000, 2): 0.15,
    (True,  10000, 0): 0.10,
    (True,   5000, 5): 0.15,
    (True,   5000, 0): 0.10,
    (True,      0, 0): 0.05,
    (False, 10000, 0): 0.05,
    (False,  5000, 0): 0.03,
}

def calculate_discount(user, order):
    for (premium, min_total, min_years), rate in DISCOUNT_TABLE.items():
        if (user.is_premium == premium and
            order.total >= min_total and
            user.years_of_membership >= min_years):
            return rate
    return 0.0

# --- ポリモーフィズムで条件分岐を排除 ---
# ❌ 型による分岐
def calculate_shipping(order):
    if order.shipping_type == "standard":
        return order.weight * 10
    elif order.shipping_type == "express":
        return order.weight * 20 + 500
    elif order.shipping_type == "overnight":
        return order.weight * 30 + 1000
    elif order.shipping_type == "international":
        return order.weight * 50 + 2000
    else:
        raise ValueError(f"Unknown shipping type: {order.shipping_type}")

# ✅ ストラテジーパターン
from abc import ABC, abstractmethod

class ShippingStrategy(ABC):
    @abstractmethod
    def calculate(self, weight: float) -> float:
        pass

class StandardShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 10

class ExpressShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 20 + 500

class OvernightShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 30 + 1000

class InternationalShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 50 + 2000

SHIPPING_STRATEGIES = {
    "standard": StandardShipping(),
    "express": ExpressShipping(),
    "overnight": OvernightShipping(),
    "international": InternationalShipping(),
}

def calculate_shipping(order):
    strategy = SHIPPING_STRATEGIES.get(order.shipping_type)
    if not strategy:
        raise ValueError(f"Unknown shipping type: {order.shipping_type}")
    return strategy.calculate(order.weight)
```

---

## 3. コードの臭い（Code Smell）

### 3.1 代表的なコードの臭い一覧

```
Code Smell（リファクタリングの兆候）:

  ┌──────────────────┬──────────────────────────────┐
  │ 臭い             │ 対策                         │
  ├──────────────────┼──────────────────────────────┤
  │ 長いメソッド      │ メソッド抽出                  │
  │ 大きなクラス      │ クラス分割                    │
  │ 重複コード        │ 共通関数に抽出                │
  │ 長い引数リスト    │ パラメータオブジェクト         │
  │ フラグ引数        │ 関数を分割                    │
  │ コメントが必要    │ コードを自己説明的に           │
  │ 深いネスト       │ 早期リターン、ガード節         │
  │ マジックナンバー  │ 名前付き定数                  │
  │ データの塊        │ データクラスに抽出             │
  │ 特性の横恋慕      │ メソッドを適切なクラスへ移動   │
  │ 変更の発散        │ 責務の分離                    │
  │ 変更の散弾        │ 関連する変更の集約             │
  │ 中間者            │ 委譲の除去                    │
  │ 不適切な親密さ    │ カプセル化の強化               │
  │ 怠惰なクラス      │ クラスの統合                   │
  │ 推測的一般化      │ 不要な抽象化の除去             │
  └──────────────────┴──────────────────────────────┘
```

### 3.2 早期リターンとガード節

```python
# ❌ 深いネスト
def process(user):
    if user:
        if user.is_active:
            if user.has_permission:
                if user.email_verified:
                    # ようやく本来の処理
                    result = do_complex_calculation(user)
                    save_result(result)
                    notify_user(user, result)
                    return result
                else:
                    raise ValueError("Email not verified")
            else:
                raise PermissionError("No permission")
        else:
            raise ValueError("User is not active")
    else:
        raise ValueError("User is None")

# ✅ ガード節で早期リターン
def process(user):
    if not user:
        raise ValueError("User is None")
    if not user.is_active:
        raise ValueError("User is not active")
    if not user.has_permission:
        raise PermissionError("No permission")
    if not user.email_verified:
        raise ValueError("Email not verified")

    # 本来の処理がフラットに書ける
    result = do_complex_calculation(user)
    save_result(result)
    notify_user(user, result)
    return result
```

### 3.3 重複コードの排除

```python
# ❌ コードの重複
class UserReport:
    def generate_csv(self, users):
        # バリデーション
        if not users:
            raise ValueError("No users provided")
        if len(users) > 10000:
            raise ValueError("Too many users")

        # フィルタリング
        active_users = [u for u in users if u.is_active]

        # CSV生成
        lines = ["name,email,role"]
        for user in active_users:
            lines.append(f"{user.name},{user.email},{user.role}")
        return "\n".join(lines)

    def generate_json(self, users):
        # バリデーション（重複!）
        if not users:
            raise ValueError("No users provided")
        if len(users) > 10000:
            raise ValueError("Too many users")

        # フィルタリング（重複!）
        active_users = [u for u in users if u.is_active]

        # JSON生成
        return json.dumps([
            {"name": u.name, "email": u.email, "role": u.role}
            for u in active_users
        ])

# ✅ 共通部分を抽出（Template Methodパターン）
class UserReport:
    def _validate_and_filter(self, users):
        """共通のバリデーションとフィルタリング"""
        if not users:
            raise ValueError("No users provided")
        if len(users) > 10000:
            raise ValueError("Too many users")
        return [u for u in users if u.is_active]

    def generate_csv(self, users):
        active_users = self._validate_and_filter(users)
        lines = ["name,email,role"]
        for user in active_users:
            lines.append(f"{user.name},{user.email},{user.role}")
        return "\n".join(lines)

    def generate_json(self, users):
        active_users = self._validate_and_filter(users)
        return json.dumps([
            {"name": u.name, "email": u.email, "role": u.role}
            for u in active_users
        ])
```

### 3.4 マジックナンバーの排除

```python
# ❌ マジックナンバーだらけ
def check_password(password):
    if len(password) < 8:
        return False
    if len(password) > 128:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True

def calculate_shipping(weight):
    if weight <= 1.0:
        return 500
    elif weight <= 5.0:
        return 500 + (weight - 1.0) * 200
    else:
        return 500 + 800 + (weight - 5.0) * 150

# ✅ 名前付き定数
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128

def check_password(password):
    if len(password) < MIN_PASSWORD_LENGTH:
        return False
    if len(password) > MAX_PASSWORD_LENGTH:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True

BASE_SHIPPING_FEE = 500
LIGHT_WEIGHT_LIMIT = 1.0    # kg
MEDIUM_WEIGHT_LIMIT = 5.0   # kg
LIGHT_RATE_PER_KG = 200     # 円/kg
MEDIUM_RATE_PER_KG = 200    # 円/kg
HEAVY_RATE_PER_KG = 150     # 円/kg

def calculate_shipping(weight):
    if weight <= LIGHT_WEIGHT_LIMIT:
        return BASE_SHIPPING_FEE
    elif weight <= MEDIUM_WEIGHT_LIMIT:
        extra = (weight - LIGHT_WEIGHT_LIMIT) * LIGHT_RATE_PER_KG
        return BASE_SHIPPING_FEE + extra
    else:
        medium_fee = (MEDIUM_WEIGHT_LIMIT - LIGHT_WEIGHT_LIMIT) * MEDIUM_RATE_PER_KG
        heavy_fee = (weight - MEDIUM_WEIGHT_LIMIT) * HEAVY_RATE_PER_KG
        return BASE_SHIPPING_FEE + medium_fee + heavy_fee
```

### 3.5 データの塊（Data Clumps）

```python
# ❌ 同じデータのグループが繰り返し登場
def create_invoice(
    customer_name, customer_email, customer_phone,
    customer_address, customer_city, customer_zip,
    items, tax_rate
):
    pass

def send_invoice(
    customer_name, customer_email, customer_phone,
    customer_address, customer_city, customer_zip,
    invoice_id
):
    pass

def update_customer(
    customer_name, customer_email, customer_phone,
    customer_address, customer_city, customer_zip
):
    pass

# ✅ データクラスにまとめる
@dataclass
class Address:
    street: str
    city: str
    zip_code: str

@dataclass
class Customer:
    name: str
    email: str
    phone: str
    address: Address

def create_invoice(customer: Customer, items: list, tax_rate: float):
    pass

def send_invoice(customer: Customer, invoice_id: str):
    pass

def update_customer(customer: Customer):
    pass
```

---

## 4. SOLID原則

### 4.1 単一責任の原則（SRP: Single Responsibility Principle）

```python
# ❌ 複数の責任を持つクラス
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def save_to_database(self):
        """データベースへの保存（永続化の責任）"""
        db.execute("INSERT INTO users VALUES (?, ?)", (self.name, self.email))

    def send_welcome_email(self):
        """メール送信（通知の責任）"""
        smtp.send(self.email, "Welcome!", f"Hello {self.name}")

    def generate_report(self):
        """レポート生成（表示の責任）"""
        return f"User Report: {self.name} ({self.email})"

    def validate(self):
        """バリデーション（検証の責任）"""
        if "@" not in self.email:
            raise ValueError("Invalid email")

# ✅ 責任ごとにクラスを分割
@dataclass
class User:
    """ユーザーのデータのみを保持"""
    name: str
    email: str

class UserValidator:
    """ユーザーの検証を担当"""
    @staticmethod
    def validate(user: User):
        if not user.name:
            raise ValueError("Name is required")
        if "@" not in user.email:
            raise ValueError("Invalid email")

class UserRepository:
    """ユーザーの永続化を担当"""
    def __init__(self, db_connection):
        self.db = db_connection

    def save(self, user: User):
        self.db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (user.name, user.email)
        )

    def find_by_email(self, email: str) -> Optional[User]:
        row = self.db.execute(
            "SELECT name, email FROM users WHERE email = ?",
            (email,)
        ).fetchone()
        return User(name=row[0], email=row[1]) if row else None

class UserNotifier:
    """ユーザーへの通知を担当"""
    def __init__(self, email_service):
        self.email_service = email_service

    def send_welcome_email(self, user: User):
        self.email_service.send(
            to=user.email,
            subject="Welcome!",
            body=f"Hello {user.name}, welcome to our platform!"
        )

class UserReportGenerator:
    """ユーザーレポートの生成を担当"""
    @staticmethod
    def generate(user: User) -> str:
        return f"User Report: {user.name} ({user.email})"
```

### 4.2 開放閉鎖の原則（OCP: Open/Closed Principle）

```python
# ❌ 新しい形状を追加するたびに既存コードを修正
class AreaCalculator:
    def calculate(self, shape):
        if shape.type == "circle":
            return 3.14159 * shape.radius ** 2
        elif shape.type == "rectangle":
            return shape.width * shape.height
        elif shape.type == "triangle":
            return 0.5 * shape.base * shape.height
        # 新しい形状を追加するたびにここを修正...

# ✅ 拡張に開き、修正に閉じている
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height

    def area(self) -> float:
        return 0.5 * self.base * self.height

# 新しい形状を追加しても既存コードは変更不要
class Trapezoid(Shape):
    def __init__(self, top: float, bottom: float, height: float):
        self.top = top
        self.bottom = bottom
        self.height = height

    def area(self) -> float:
        return 0.5 * (self.top + self.bottom) * self.height

# 使用側は Shape インターフェースにのみ依存
def total_area(shapes: list[Shape]) -> float:
    return sum(s.area() for s in shapes)
```

### 4.3 リスコフの置換原則（LSP: Liskov Substitution Principle）

```python
# ❌ LSP違反: 正方形は長方形のサブタイプとして適切でない
class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    def area(self):
        return self._width * self._height

class Square(Rectangle):
    """正方形: 幅と高さを常に同じに保つ"""
    def __init__(self, side):
        super().__init__(side, side)

    @Rectangle.width.setter
    def width(self, value):
        self._width = value
        self._height = value  # 高さも変わってしまう!

    @Rectangle.height.setter
    def height(self, value):
        self._width = value
        self._height = value

# これが問題になるケース
def resize_rectangle(rect: Rectangle):
    rect.width = 10
    rect.height = 5
    assert rect.area() == 50  # Squareだと失敗する!

# ✅ LSPを守る設計
class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Square(Shape):
    def __init__(self, side: float):
        self.side = side

    def area(self) -> float:
        return self.side ** 2
```

### 4.4 インターフェース分離の原則（ISP: Interface Segregation Principle）

```python
# ❌ 大きすぎるインターフェース
class Worker(ABC):
    @abstractmethod
    def work(self): pass

    @abstractmethod
    def eat(self): pass

    @abstractmethod
    def sleep(self): pass

    @abstractmethod
    def code(self): pass

    @abstractmethod
    def test(self): pass

    @abstractmethod
    def deploy(self): pass

class Robot(Worker):
    def work(self):
        print("Working...")

    def eat(self):
        raise NotImplementedError("Robots don't eat!")  # 違反!

    def sleep(self):
        raise NotImplementedError("Robots don't sleep!")  # 違反!

    def code(self):
        print("Coding...")

    def test(self):
        print("Testing...")

    def deploy(self):
        print("Deploying...")

# ✅ インターフェースを分離
class Workable(ABC):
    @abstractmethod
    def work(self): pass

class Eatable(ABC):
    @abstractmethod
    def eat(self): pass

class Sleepable(ABC):
    @abstractmethod
    def sleep(self): pass

class Codeable(ABC):
    @abstractmethod
    def code(self): pass

class Testable(ABC):
    @abstractmethod
    def test(self): pass

class Deployable(ABC):
    @abstractmethod
    def deploy(self): pass

class Human(Workable, Eatable, Sleepable, Codeable, Testable, Deployable):
    def work(self): print("Working...")
    def eat(self): print("Eating...")
    def sleep(self): print("Sleeping...")
    def code(self): print("Coding...")
    def test(self): print("Testing...")
    def deploy(self): print("Deploying...")

class Robot(Workable, Codeable, Testable, Deployable):
    def work(self): print("Working...")
    def code(self): print("Coding...")
    def test(self): print("Testing...")
    def deploy(self): print("Deploying...")
```

### 4.5 依存性逆転の原則（DIP: Dependency Inversion Principle）

```python
# ❌ 高レベルモジュールが低レベルモジュールに直接依存
import mysql.connector

class UserService:
    def __init__(self):
        # MySQLに直接依存している
        self.connection = mysql.connector.connect(
            host="localhost",
            database="myapp"
        )

    def get_user(self, user_id):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        return cursor.fetchone()

# ✅ 抽象に依存する
class UserRepository(ABC):
    """抽象リポジトリ（インターフェース）"""
    @abstractmethod
    def find_by_id(self, user_id: str) -> Optional[User]:
        pass

    @abstractmethod
    def save(self, user: User) -> None:
        pass

    @abstractmethod
    def delete(self, user_id: str) -> None:
        pass

class MySQLUserRepository(UserRepository):
    """MySQL実装"""
    def __init__(self, connection):
        self.connection = connection

    def find_by_id(self, user_id: str) -> Optional[User]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        return User(name=row[1], email=row[2]) if row else None

    def save(self, user: User) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s)",
            (user.name, user.email)
        )
        self.connection.commit()

    def delete(self, user_id: str) -> None:
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        self.connection.commit()

class InMemoryUserRepository(UserRepository):
    """テスト用インメモリ実装"""
    def __init__(self):
        self.users: dict[str, User] = {}

    def find_by_id(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    def save(self, user: User) -> None:
        self.users[user.email] = user

    def delete(self, user_id: str) -> None:
        self.users.pop(user_id, None)

class UserService:
    """抽象リポジトリに依存（具体的な実装を知らない）"""
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def get_user(self, user_id: str) -> Optional[User]:
        return self.repository.find_by_id(user_id)

    def register_user(self, name: str, email: str) -> User:
        user = User(name=name, email=email)
        self.repository.save(user)
        return user

# 本番環境
production_service = UserService(MySQLUserRepository(db_connection))

# テスト環境
test_service = UserService(InMemoryUserRepository())
```

---

## 5. DRY / KISS / YAGNI

### 5.1 DRY（Don't Repeat Yourself）

```python
# DRY = 知識の重複を避ける（コードの文字列的な重複だけではない）

# ❌ 知識の重複: 税率計算のロジックが散在
class OrderService:
    def calculate_order_total(self, items):
        subtotal = sum(item.price * item.quantity for item in items)
        tax = subtotal * 0.10  # 税率10%がハードコード
        return subtotal + tax

class InvoiceService:
    def generate_invoice(self, order):
        subtotal = order.subtotal
        tax = subtotal * 0.10  # 同じ税率が別の場所に
        return {"subtotal": subtotal, "tax": tax, "total": subtotal + tax}

class ReportService:
    def calculate_revenue(self, orders):
        total = 0
        for order in orders:
            total += order.subtotal * 1.10  # また同じ税率
        return total

# ✅ 税率計算を一箇所に集約
TAX_RATE = 0.10

class TaxCalculator:
    @staticmethod
    def calculate_tax(amount: float) -> float:
        return amount * TAX_RATE

    @staticmethod
    def calculate_total_with_tax(amount: float) -> float:
        return amount + TaxCalculator.calculate_tax(amount)

class OrderService:
    def calculate_order_total(self, items):
        subtotal = sum(item.price * item.quantity for item in items)
        return TaxCalculator.calculate_total_with_tax(subtotal)

class InvoiceService:
    def generate_invoice(self, order):
        tax = TaxCalculator.calculate_tax(order.subtotal)
        total = TaxCalculator.calculate_total_with_tax(order.subtotal)
        return {"subtotal": order.subtotal, "tax": tax, "total": total}

# --- DRYの過剰適用に注意 ---
# 偶然の重複はDRY化しない
# ❌ 偶然の重複を無理にDRY化
def format_address(address):
    return f"{address.street}, {address.city} {address.zip}"

def format_log_entry(entry):
    # たまたま同じフォーマットだが、別の概念
    return f"{entry.message}, {entry.level} {entry.timestamp}"

# ✅ 別々の概念は別々のまま維持する
# format_address と format_log_entry は偶然似ているだけ
# 将来的に異なる方向に変化する可能性が高い
```

### 5.2 KISS（Keep It Simple, Stupid）

```python
# KISS = 不必要な複雑さを避ける

# ❌ 過度に複雑（メタプログラミングの乱用）
class DynamicValidator:
    def __init__(self):
        self._rules = {}

    def register_rule(self, field, rule_type, **kwargs):
        if field not in self._rules:
            self._rules[field] = []
        self._rules[field].append((rule_type, kwargs))

    def validate(self, data):
        errors = []
        for field, rules in self._rules.items():
            value = data.get(field)
            for rule_type, kwargs in rules:
                validator = getattr(self, f"_validate_{rule_type}")
                result = validator(value, **kwargs)
                if not result:
                    errors.append(f"{field}: {rule_type} validation failed")
        return errors

    def _validate_required(self, value):
        return value is not None and value != ""

    def _validate_min_length(self, value, length=0):
        return len(str(value)) >= length

    def _validate_max_length(self, value, length=float("inf")):
        return len(str(value)) <= length

    def _validate_pattern(self, value, regex=""):
        return bool(re.match(regex, str(value)))

# 使う側も複雑
validator = DynamicValidator()
validator.register_rule("email", "required")
validator.register_rule("email", "pattern", regex=r"^[\w.-]+@[\w.-]+\.\w+$")
validator.register_rule("password", "required")
validator.register_rule("password", "min_length", length=8)

# ✅ シンプルで分かりやすい
def validate_registration(data: dict) -> list[str]:
    """登録データのバリデーション"""
    errors = []

    email = data.get("email", "")
    if not email:
        errors.append("メールアドレスは必須です")
    elif not re.match(r"^[\w.-]+@[\w.-]+\.\w+$", email):
        errors.append("メールアドレスの形式が正しくありません")

    password = data.get("password", "")
    if not password:
        errors.append("パスワードは必須です")
    elif len(password) < 8:
        errors.append("パスワードは8文字以上で入力してください")

    return errors

# 必要になった時点で抽象化を検討する
```

### 5.3 YAGNI（You Aren't Gonna Need It）

```python
# YAGNI = 今必要のない機能を先に実装しない

# ❌ 「いつか必要になるかも」で作った過剰設計
class UserService:
    def __init__(self, repository, cache, event_bus, logger,
                 rate_limiter, circuit_breaker, metrics_collector,
                 feature_flags, plugin_manager):
        self.repository = repository
        self.cache = cache
        self.event_bus = event_bus
        self.logger = logger
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.metrics_collector = metrics_collector
        self.feature_flags = feature_flags
        self.plugin_manager = plugin_manager

    def get_user(self, user_id):
        # 実際には cache, event_bus, circuit_breaker は未使用
        self.logger.info(f"Getting user {user_id}")
        self.metrics_collector.increment("user.get")
        return self.repository.find_by_id(user_id)

# ✅ 今必要なものだけを実装
class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def get_user(self, user_id: str) -> Optional[User]:
        return self.repository.find_by_id(user_id)

# キャッシュが本当に必要になったら追加する
# イベントバスが本当に必要になったら追加する
# → 必要なタイミングで段階的に拡張する
```

---

## 6. コメントの書き方

### 6.1 良いコメントと悪いコメント

```python
# --- 悪いコメント（コードを説明するコメント） ---
# ❌ コードを読めば分かることの繰り返し
i = 0  # iを0に初期化する
users = get_users()  # ユーザーを取得する
total = price * quantity  # 金額 = 単価 x 数量

# ❌ 嘘のコメント（コードと乖離している）
# ユーザーの年齢を返す
def get_user_name(user_id):  # 実際には名前を返す
    pass

# ❌ 閉じ括弧コメント（構造が複雑すぎるサイン）
for user in users:
    for order in user.orders:
        for item in order.items:
            process(item)
        # end for item
    # end for order
# end for user

# ❌ コメントアウトされたコード
# def old_calculate_tax(amount):
#     return amount * 0.08
# ↑ いつのコードか分からず誰も消せない

# --- 良いコメント ---
# ✅ WHY（なぜそうするのか）を説明する
# パフォーマンス上の理由でバッチサイズを1000に制限
# 一度に10000件以上処理するとメモリが枯渇する問題があった（Issue #234）
BATCH_SIZE = 1000

# ✅ 法的コメント
# Copyright (c) 2024 Example Corp. All rights reserved.
# Licensed under the MIT License.

# ✅ TODO/FIXME（技術的負債の明示）
# TODO(gaku): Redis導入後にキャッシュ機能を追加する（#456）
# FIXME: 月末のタイムゾーン処理にバグがある（#789）
# HACK: APIの仕様バグの回避策。v3.0リリースで修正予定

# ✅ 結果の警告
# 注意: この関数は外部APIを呼び出すため、3秒以上かかることがある
def fetch_exchange_rates():
    pass

# ✅ 正規表現の説明
# 日本の電話番号にマッチ: 03-1234-5678, 090-1234-5678 等
PHONE_PATTERN = re.compile(r"^0\d{1,4}-\d{1,4}-\d{4}$")

# ✅ ビジネスルールの説明
# 消費税法第29条に基づき、1円未満は切り捨て
tax = math.floor(subtotal * TAX_RATE)
```

### 6.2 docstringのベストプラクティス

```python
def calculate_compound_interest(
    principal: float,
    annual_rate: float,
    years: int,
    compounds_per_year: int = 12
) -> float:
    """複利計算を行う。

    指定された元本に対して、年利率と複利計算の頻度に基づいて
    将来の価値を計算する。

    Args:
        principal: 元本（円）。正の数であること。
        annual_rate: 年利率（0.05 = 5%）。
        years: 運用年数。
        compounds_per_year: 年間の複利計算回数（デフォルト: 12=毎月）。

    Returns:
        将来の元利合計額（円）。

    Raises:
        ValueError: principal が負の場合。
        ValueError: years が 0 以下の場合。

    Examples:
        >>> calculate_compound_interest(1000000, 0.05, 10)
        1647009.49
        >>> calculate_compound_interest(1000000, 0.03, 5, compounds_per_year=1)
        1159274.07

    Note:
        この計算では税金を考慮していない。
        実際の投資では利子所得に対して約20%の税金がかかる。
    """
    if principal < 0:
        raise ValueError("元本は正の数である必要があります")
    if years <= 0:
        raise ValueError("運用年数は1以上である必要があります")

    return principal * (1 + annual_rate / compounds_per_year) ** (compounds_per_year * years)


class ShoppingCart:
    """ショッピングカートを管理するクラス。

    ユーザーがカートに追加した商品の管理、合計金額の計算、
    クーポン適用などの機能を提供する。

    Attributes:
        items: カート内の商品リスト。
        user_id: カートの所有者のユーザーID。
        created_at: カート作成日時。

    Example:
        >>> cart = ShoppingCart(user_id="u001")
        >>> cart.add_item(Item("apple", 100), quantity=3)
        >>> cart.total
        300
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.items: list[CartItem] = []
        self.created_at = datetime.now()
```

---

## 7. エラー処理

### 7.1 例外処理のベストプラクティス

```python
# --- 1. 具体的な例外をキャッチする ---
# ❌ 全ての例外をキャッチ
try:
    result = process_data(data)
except Exception:
    print("Something went wrong")

# ✅ 具体的な例外をキャッチ
try:
    result = process_data(data)
except ValueError as e:
    logger.warning(f"Invalid data: {e}")
    return default_value
except ConnectionError as e:
    logger.error(f"Database connection failed: {e}")
    raise ServiceUnavailableError("Database is temporarily unavailable") from e
except TimeoutError as e:
    logger.error(f"Operation timed out: {e}")
    raise

# --- 2. カスタム例外を定義する ---
class AppError(Exception):
    """アプリケーション共通の基底例外"""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.error_code = error_code

class ValidationError(AppError):
    """バリデーションエラー"""
    def __init__(self, field: str, message: str):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field

class NotFoundError(AppError):
    """リソースが見つからないエラー"""
    def __init__(self, resource: str, resource_id: str):
        message = f"{resource} with id '{resource_id}' not found"
        super().__init__(message, error_code="NOT_FOUND")
        self.resource = resource
        self.resource_id = resource_id

class AuthenticationError(AppError):
    """認証エラー"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, error_code="AUTHENTICATION_ERROR")

class AuthorizationError(AppError):
    """認可エラー"""
    def __init__(self, required_permission: str):
        message = f"Permission '{required_permission}' is required"
        super().__init__(message, error_code="AUTHORIZATION_ERROR")
        self.required_permission = required_permission

# 使用例
class UserService:
    def get_user(self, user_id: str) -> User:
        user = self.repository.find_by_id(user_id)
        if user is None:
            raise NotFoundError("User", user_id)
        return user

    def create_user(self, name: str, email: str) -> User:
        if not name:
            raise ValidationError("name", "Name is required")
        if not self._is_valid_email(email):
            raise ValidationError("email", "Invalid email format")
        if self.repository.find_by_email(email):
            raise ValidationError("email", "Email already registered")
        return self.repository.save(User(name=name, email=email))

# --- 3. 例外をログに記録する ---
import logging
logger = logging.getLogger(__name__)

def process_payment(order):
    try:
        payment_gateway.charge(order.amount, order.payment_method)
    except PaymentDeclinedError as e:
        # ビジネス上想定されるエラー: WARNINGレベル
        logger.warning(
            "Payment declined for order %s: %s",
            order.id, e,
            extra={"order_id": order.id, "amount": order.amount}
        )
        raise
    except PaymentGatewayError as e:
        # システムエラー: ERRORレベル + スタックトレース
        logger.error(
            "Payment gateway error for order %s: %s",
            order.id, e,
            exc_info=True,
            extra={"order_id": order.id}
        )
        raise ServiceUnavailableError("Payment service is temporarily unavailable") from e

# --- 4. 例外の変換（例外チェーン） ---
def get_user_profile(user_id: str) -> UserProfile:
    try:
        data = external_api.fetch_user(user_id)
    except requests.ConnectionError as e:
        raise ServiceUnavailableError(
            "External API is unavailable"
        ) from e  # from e で元の例外を保持
    except requests.Timeout as e:
        raise ServiceUnavailableError(
            "External API request timed out"
        ) from e

    try:
        return UserProfile.from_dict(data)
    except KeyError as e:
        raise DataIntegrityError(
            f"Missing required field in API response: {e}"
        ) from e
```

### 7.2 Null/Noneの安全な扱い

```python
# --- 1. Noneチェックパターン ---
# ❌ Noneチェックの散在
def get_user_city(user_id):
    user = repository.find_by_id(user_id)
    if user is not None:
        address = user.address
        if address is not None:
            city = address.city
            if city is not None:
                return city
    return "Unknown"

# ✅ 早期リターン
def get_user_city(user_id: str) -> str:
    user = repository.find_by_id(user_id)
    if user is None:
        return "Unknown"
    if user.address is None:
        return "Unknown"
    return user.address.city or "Unknown"

# ✅ Optional型を活用（Python 3.10+）
from typing import Optional

def find_user(user_id: str) -> Optional[User]:
    """ユーザーが見つからない場合はNoneを返す"""
    return repository.find_by_id(user_id)

# --- 2. Null Objectパターン ---
class NullUser:
    """ユーザーが存在しない場合のデフォルト値"""
    name = "Guest"
    email = ""
    is_active = False
    permissions = frozenset()

    def has_permission(self, permission: str) -> bool:
        return False

NULL_USER = NullUser()

def get_user_or_default(user_id: str) -> User:
    user = repository.find_by_id(user_id)
    return user if user is not None else NULL_USER

# --- 3. デフォルト値の活用 ---
# dictのgetメソッド
config = {"debug": True, "log_level": "INFO"}
debug_mode = config.get("debug", False)  # キーがなければFalse
log_level = config.get("log_level", "WARNING")  # デフォルトWARNING
timeout = config.get("timeout", 30)  # キーがなければ30
```

### 7.3 リソース管理

```python
# --- コンテキストマネージャ（with文）を使う ---

# ❌ 手動でクローズ（例外時にリークする）
file = open("data.csv", "r")
data = file.read()
file.close()  # 例外発生時にclose()が呼ばれない

# ✅ with文で自動クローズ
with open("data.csv", "r") as file:
    data = file.read()  # 例外が発生してもclose()が呼ばれる

# --- カスタムコンテキストマネージャ ---
from contextlib import contextmanager

@contextmanager
def database_transaction(connection):
    """データベーストランザクション管理"""
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()

# 使用例
with database_transaction(db_connection) as cursor:
    cursor.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    cursor.execute("INSERT INTO logs (action) VALUES (?)", ("user_created",))
    # 例外が発生したら自動的にrollback

# --- 複数リソースの管理 ---
with open("input.csv") as infile, open("output.csv", "w") as outfile:
    for line in infile:
        processed = process_line(line)
        outfile.write(processed)

# --- タイムアウト管理 ---
import signal

@contextmanager
def timeout(seconds):
    """処理のタイムアウト管理"""
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# 使用例
with timeout(5):
    result = long_running_operation()
```

---

## 8. テスタビリティの高いコード

### 8.1 テストしやすいコードの特徴

```python
# --- テストしにくいコード ---
# ❌ グローバル状態への依存
import datetime

user_count = 0  # グローバル変数

class UserService:
    def create_user(self, name, email):
        global user_count
        user = User(
            name=name,
            email=email,
            created_at=datetime.datetime.now()  # 現在時刻に直接依存
        )
        user_count += 1  # グローバル状態を変更
        db.save(user)  # グローバルDBに直接依存
        return user

# --- テストしやすいコード ---
# ✅ 依存性注入
class UserService:
    def __init__(self, repository: UserRepository, clock: Clock):
        self.repository = repository
        self.clock = clock

    def create_user(self, name: str, email: str) -> User:
        user = User(
            name=name,
            email=email,
            created_at=self.clock.now()
        )
        self.repository.save(user)
        return user

# テスト
class FakeClock:
    def __init__(self, fixed_time):
        self._time = fixed_time
    def now(self):
        return self._time

def test_create_user():
    fake_repo = InMemoryUserRepository()
    fake_clock = FakeClock(datetime.datetime(2024, 1, 1, 12, 0, 0))
    service = UserService(repository=fake_repo, clock=fake_clock)

    user = service.create_user("Alice", "alice@example.com")

    assert user.name == "Alice"
    assert user.email == "alice@example.com"
    assert user.created_at == datetime.datetime(2024, 1, 1, 12, 0, 0)
    assert fake_repo.find_by_email("alice@example.com") is not None
```

### 8.2 テストの構造（AAA パターン）

```python
import pytest

class TestShoppingCart:
    """ショッピングカートのテスト"""

    def test_add_item_increases_total(self):
        # Arrange（準備）
        cart = ShoppingCart()
        item = Item(name="Apple", price=100)

        # Act（実行）
        cart.add_item(item, quantity=3)

        # Assert（検証）
        assert cart.total == 300
        assert cart.item_count == 3

    def test_apply_percentage_coupon(self):
        # Arrange
        cart = ShoppingCart()
        cart.add_item(Item("Apple", 100), quantity=5)
        coupon = PercentageCoupon(rate=0.10)  # 10%オフ

        # Act
        cart.apply_coupon(coupon)

        # Assert
        assert cart.total == 450  # 500 - 50

    def test_remove_item_decreases_total(self):
        # Arrange
        cart = ShoppingCart()
        apple = Item("Apple", 100)
        cart.add_item(apple, quantity=3)

        # Act
        cart.remove_item(apple, quantity=1)

        # Assert
        assert cart.total == 200
        assert cart.item_count == 2

    def test_empty_cart_has_zero_total(self):
        # Arrange & Act
        cart = ShoppingCart()

        # Assert
        assert cart.total == 0
        assert cart.item_count == 0
        assert cart.is_empty

    def test_cannot_add_negative_quantity(self):
        # Arrange
        cart = ShoppingCart()
        item = Item("Apple", 100)

        # Act & Assert
        with pytest.raises(ValueError, match="Quantity must be positive"):
            cart.add_item(item, quantity=-1)
```

### 8.3 テストダブル（Mock/Stub/Fake/Spy）

```python
from unittest.mock import Mock, patch, MagicMock

# --- Stub: 固定値を返す ---
class StubPaymentGateway:
    """常に成功する決済ゲートウェイ"""
    def charge(self, amount, card):
        return PaymentResult(success=True, transaction_id="stub-txn-001")

# --- Fake: 簡易実装 ---
class FakeEmailService:
    """実際には送信せず、送信履歴を保持する"""
    def __init__(self):
        self.sent_emails = []

    def send(self, to, subject, body):
        self.sent_emails.append({
            "to": to, "subject": subject, "body": body
        })

# --- Mock: 呼び出しを検証 ---
def test_order_sends_confirmation_email():
    # Arrange
    mock_email = Mock(spec=EmailService)
    service = OrderService(email_service=mock_email)
    order = Order(customer_email="alice@example.com", items=["item1"])

    # Act
    service.complete_order(order)

    # Assert: 正しい引数で呼び出されたか
    mock_email.send_confirmation.assert_called_once_with(
        to="alice@example.com",
        order_id=order.id
    )

# --- Spy: 実際の処理を実行しつつ呼び出しを記録 ---
class SpyLogger:
    def __init__(self, real_logger):
        self.real_logger = real_logger
        self.logged_messages = []

    def info(self, message):
        self.logged_messages.append(("INFO", message))
        self.real_logger.info(message)

    def error(self, message):
        self.logged_messages.append(("ERROR", message))
        self.real_logger.error(message)

# --- patch を使ったモック ---
@patch("app.services.external_api.fetch_user")
def test_get_user_profile(mock_fetch):
    # Arrange
    mock_fetch.return_value = {"name": "Alice", "email": "alice@example.com"}

    # Act
    service = UserProfileService()
    profile = service.get_profile("user-001")

    # Assert
    assert profile.name == "Alice"
    mock_fetch.assert_called_once_with("user-001")
```

---

## 9. リファクタリング

### 9.1 リファクタリングのタイミング

```
リファクタリングすべきタイミング:

  1. 機能追加前: 新機能が追加しやすい構造にする
  2. バグ修正時: バグの原因を取り除く過程で
  3. コードレビュー時: レビューで指摘された改善点を対応
  4. ボーイスカウトルール: 触ったコードを少し良くして戻す

リファクタリングすべきでないタイミング:

  1. デッドライン直前: リスクが高い
  2. 動作が理解できていない時: まずテストを書く
  3. テストがないコード: まずテストを追加してから
  4. 大規模な書き換え: 段階的に進める
```

### 9.2 代表的なリファクタリング手法

```python
# --- 1. メソッド抽出（Extract Method） ---
# ❌ 長い関数
def print_invoice(invoice):
    print("=" * 50)
    print(f"Invoice #{invoice.id}")
    print(f"Date: {invoice.date}")
    print("-" * 50)
    for item in invoice.items:
        total = item.price * item.quantity
        print(f"  {item.name}: {item.quantity} x {item.price} = {total}")
    print("-" * 50)
    subtotal = sum(i.price * i.quantity for i in invoice.items)
    tax = subtotal * 0.10
    total = subtotal + tax
    print(f"  Subtotal: {subtotal}")
    print(f"  Tax (10%): {tax}")
    print(f"  Total: {total}")
    print("=" * 50)

# ✅ メソッドを抽出
def print_invoice(invoice):
    print_header(invoice)
    print_line_items(invoice.items)
    print_totals(invoice.items)

def print_header(invoice):
    print("=" * 50)
    print(f"Invoice #{invoice.id}")
    print(f"Date: {invoice.date}")
    print("-" * 50)

def print_line_items(items):
    for item in items:
        total = item.price * item.quantity
        print(f"  {item.name}: {item.quantity} x {item.price} = {total}")
    print("-" * 50)

def print_totals(items):
    subtotal = calculate_subtotal(items)
    tax = calculate_tax(subtotal)
    total = subtotal + tax
    print(f"  Subtotal: {subtotal}")
    print(f"  Tax (10%): {tax}")
    print(f"  Total: {total}")
    print("=" * 50)

# --- 2. 変数の抽出（Extract Variable） ---
# ❌ 複雑な式
def is_eligible_for_premium(user):
    return (user.age >= 18 and
            user.years_of_membership >= 2 and
            user.total_purchases >= 100000 and
            user.is_active and
            not user.has_violations and
            user.last_login_days_ago <= 30)

# ✅ 意味のある変数に分割
def is_eligible_for_premium(user):
    is_adult = user.age >= 18
    is_long_term_member = user.years_of_membership >= 2
    is_high_value_customer = user.total_purchases >= 100000
    is_recently_active = user.last_login_days_ago <= 30
    has_good_standing = user.is_active and not user.has_violations

    return (is_adult and
            is_long_term_member and
            is_high_value_customer and
            is_recently_active and
            has_good_standing)

# --- 3. 条件式の分解（Decompose Conditional） ---
# ❌ 複雑な条件式
def calculate_charge(date, quantity):
    if (date.month >= 6 and date.month <= 9):
        charge = quantity * SUMMER_RATE + SUMMER_SERVICE_CHARGE
    else:
        charge = quantity * WINTER_RATE + WINTER_SERVICE_CHARGE
    return charge

# ✅ 条件を関数に抽出
def is_summer(date):
    return 6 <= date.month <= 9

def summer_charge(quantity):
    return quantity * SUMMER_RATE + SUMMER_SERVICE_CHARGE

def winter_charge(quantity):
    return quantity * WINTER_RATE + WINTER_SERVICE_CHARGE

def calculate_charge(date, quantity):
    if is_summer(date):
        return summer_charge(quantity)
    return winter_charge(quantity)

# --- 4. パラメータオブジェクトの導入 ---
# ❌ 引数が多い
def search_products(
    category, min_price, max_price,
    brand, color, size,
    sort_by, sort_order,
    page, page_size
):
    pass

# ✅ パラメータオブジェクトに統合
@dataclass
class ProductSearchCriteria:
    category: str = ""
    min_price: float = 0
    max_price: float = float("inf")
    brand: str = ""
    color: str = ""
    size: str = ""

@dataclass
class PaginationParams:
    page: int = 1
    page_size: int = 20
    sort_by: str = "created_at"
    sort_order: str = "desc"

def search_products(
    criteria: ProductSearchCriteria,
    pagination: PaginationParams
):
    pass

# --- 5. ガード節の導入（Replace Nested Conditional with Guard Clauses） ---
# ❌ ネストした条件分岐
def calculate_pay(employee):
    if employee.is_separated:
        result = calculate_separated_pay(employee)
    else:
        if employee.is_retired:
            result = calculate_retired_pay(employee)
        else:
            result = calculate_normal_pay(employee)
    return result

# ✅ ガード節
def calculate_pay(employee):
    if employee.is_separated:
        return calculate_separated_pay(employee)
    if employee.is_retired:
        return calculate_retired_pay(employee)
    return calculate_normal_pay(employee)
```

### 9.3 リファクタリングの安全な進め方

```
安全なリファクタリングの手順:

  1. テストが通ることを確認
  ↓
  2. 小さなステップでリファクタリング
  ↓
  3. テストが通ることを確認
  ↓
  4. コミット
  ↓
  5. 2-4を繰り返す

  ポイント:
  - 1回のコミットで1種類のリファクタリング
  - 機能変更とリファクタリングを混ぜない
  - テストがない場合はまずテストを追加
  - IDEのリファクタリング機能を活用（手作業を減らす）
  - ペアプログラミングやコードレビューを併用

  推奨ツール:
  - Python: ruff, mypy, black, isort
  - JavaScript/TypeScript: ESLint, Prettier
  - Java: IntelliJ IDEA, SpotBugs, PMD
  - Go: gofmt, golint, go vet
```

---

## 10. コードレビュー

### 10.1 コードレビューの観点

```
コードレビューチェックリスト:

  □ 正確性
    - ロジックに誤りがないか
    - エッジケースが考慮されているか
    - 境界値の処理は正しいか
    - 並行処理の安全性は確保されているか

  □ 可読性
    - 命名は適切か
    - 関数は適切な長さか
    - コメントは必要十分か
    - コードの意図が明確か

  □ 保守性
    - SOLID原則に従っているか
    - 適切な抽象化レベルか
    - テストは十分か
    - エラー処理は適切か

  □ パフォーマンス
    - 不要なN+1クエリがないか
    - 適切なインデックスが使われているか
    - メモリリークの可能性はないか
    - キャッシュの活用は適切か

  □ セキュリティ
    - SQLインジェクション対策は十分か
    - 入力値のバリデーションは行われているか
    - 認証・認可のチェックは適切か
    - 機密情報がログに出力されていないか
```

### 10.2 効果的なレビューコメントの書き方

```
レビューコメントのベストプラクティス:

  1. 具体的に指摘する
    ❌ "ここは良くない"
    ✅ "この関数は30行あり、バリデーションと計算と通知の3つの責任を持っています。
        それぞれ別関数に抽出することで可読性が向上します"

  2. 理由を説明する
    ❌ "定数にしてください"
    ✅ "この 86400 は何の値か分かりにくいです。
        SECONDS_PER_DAY = 86400 のように名前付き定数にすると
        コードの意図が明確になります"

  3. 提案を含める
    ❌ "エラー処理が足りない"
    ✅ "外部APIの呼び出しで ConnectionError が発生する可能性があります。
        try-except で捕捉し、適切なリトライまたはフォールバック処理を
        追加することを提案します"

  4. 重要度を示す
    [must]    修正必須（バグ、セキュリティ問題）
    [should]  強く推奨（可読性、保守性の改善）
    [nit]     些細な指摘（スタイル、命名の微調整）
    [question] 質問（理解のため）
    [praise]  良い点の称賛

  5. 肯定的なフィードバックも入れる
    "このテストケースのカバレッジは素晴らしいです！
     特にエッジケースの網羅が丁寧です。"
```

---

## 11. 言語別のクリーンコード実践

### 11.1 TypeScript/JavaScript

```typescript
// --- 型を活用した安全なコード ---

// ❌ any型の乱用
function processData(data: any): any {
    return data.map((item: any) => item.value * 2);
}

// ✅ 適切な型定義
interface DataItem {
    id: string;
    value: number;
    label: string;
}

interface ProcessedItem {
    id: string;
    doubledValue: number;
}

function processData(data: DataItem[]): ProcessedItem[] {
    return data.map(item => ({
        id: item.id,
        doubledValue: item.value * 2
    }));
}

// --- Union型とType Guardの活用 ---
type Result<T> =
    | { success: true; data: T }
    | { success: false; error: string };

function fetchUser(id: string): Result<User> {
    try {
        const user = database.findUser(id);
        if (!user) {
            return { success: false, error: `User ${id} not found` };
        }
        return { success: true, data: user };
    } catch (e) {
        return { success: false, error: `Failed to fetch user: ${e}` };
    }
}

// 使用側で型安全にハンドリング
const result = fetchUser("u001");
if (result.success) {
    console.log(result.data.name);  // 型安全
} else {
    console.error(result.error);
}

// --- 不変性の確保 ---
// ❌ ミュータブル
const cart = { items: [], total: 0 };
cart.items.push(newItem);  // 直接変更
cart.total = calculateTotal(cart.items);

// ✅ イミュータブル
interface Cart {
    readonly items: readonly CartItem[];
    readonly total: number;
}

function addItem(cart: Cart, item: CartItem): Cart {
    const newItems = [...cart.items, item];
    return {
        items: newItems,
        total: calculateTotal(newItems)
    };
}
```

### 11.2 Go

```go
// --- エラー処理のパターン ---

// ❌ エラーを無視
func getUser(id string) *User {
    user, _ := db.FindUser(id)  // エラーを無視
    return user
}

// ✅ エラーを適切に処理
func getUser(id string) (*User, error) {
    user, err := db.FindUser(id)
    if err != nil {
        return nil, fmt.Errorf("failed to get user %s: %w", id, err)
    }
    if user == nil {
        return nil, ErrUserNotFound
    }
    return user, nil
}

// --- インターフェースの活用 ---

// ✅ 小さなインターフェース（Go流）
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type ReadWriter interface {
    Reader
    Writer
}

// ✅ 受け入れるインターフェースは小さく、返す型は具体的に
type UserRepository interface {
    FindByID(id string) (*User, error)
    Save(user *User) error
}

func NewUserService(repo UserRepository) *UserService {
    return &UserService{repo: repo}
}

// --- 構造体のコンストラクタパターン ---
type Server struct {
    host    string
    port    int
    timeout time.Duration
    logger  *log.Logger
}

// Functional Optionsパターン
type ServerOption func(*Server)

func WithPort(port int) ServerOption {
    return func(s *Server) {
        s.port = port
    }
}

func WithTimeout(timeout time.Duration) ServerOption {
    return func(s *Server) {
        s.timeout = timeout
    }
}

func WithLogger(logger *log.Logger) ServerOption {
    return func(s *Server) {
        s.logger = logger
    }
}

func NewServer(host string, opts ...ServerOption) *Server {
    s := &Server{
        host:    host,
        port:    8080,  // デフォルト値
        timeout: 30 * time.Second,
        logger:  log.Default(),
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// 使用例
server := NewServer("localhost",
    WithPort(3000),
    WithTimeout(60*time.Second),
)
```

### 11.3 Java/Kotlin

```java
// --- Optionalの正しい使い方（Java） ---

// ❌ Optionalの誤用
public Optional<User> getUser(String id) {
    User user = repository.findById(id);
    return Optional.ofNullable(user);  // ここまではOK
}

// 呼び出し側の誤用
Optional<User> optUser = getUser("u001");
if (optUser.isPresent()) {  // ❌ get()の前にisPresent()チェック = nullチェックと同じ
    User user = optUser.get();
}

// ✅ Optionalを活用した関数型スタイル
public String getUserDisplayName(String userId) {
    return repository.findById(userId)
        .map(User::getDisplayName)
        .orElse("Unknown User");
}

public void sendWelcomeEmail(String userId) {
    repository.findById(userId)
        .ifPresent(user -> emailService.sendWelcome(user.getEmail()));
}

public User getActiveUser(String userId) {
    return repository.findById(userId)
        .filter(User::isActive)
        .orElseThrow(() -> new UserNotFoundException(userId));
}
```

```kotlin
// --- Kotlinのクリーンコード ---

// データクラスの活用
data class User(
    val id: String,
    val name: String,
    val email: String,
    val isActive: Boolean = true
)

// 拡張関数で既存クラスを拡張
fun String.isValidEmail(): Boolean =
    matches(Regex("^[\\w.-]+@[\\w.-]+\\.\\w+$"))

fun List<User>.activeUsers(): List<User> =
    filter { it.isActive }

// スコープ関数の使い分け
// let: null安全な変換
val displayName = user?.let { "${it.name} (${it.email})" } ?: "Guest"

// apply: オブジェクトの初期化
val config = ServerConfig().apply {
    host = "localhost"
    port = 8080
    timeout = Duration.ofSeconds(30)
}

// also: 副作用（ログ出力等）
val result = repository.findById(id)
    .also { logger.info("Found user: ${it?.name}") }

// run: オブジェクトに対する計算
val summary = order.run {
    "Order #$id: $itemCount items, total = $total"
}

// sealed classで網羅的なパターンマッチ
sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Failure(val error: String) : Result<Nothing>()
    object Loading : Result<Nothing>()
}

fun handleResult(result: Result<User>) = when (result) {
    is Result.Success -> showUser(result.data)
    is Result.Failure -> showError(result.error)
    Result.Loading -> showLoading()
    // whenが網羅的なのでelseは不要
}
```

---

## 12. クリーンコードのアンチパターン集

### 12.1 過剰設計（Over-Engineering）

```python
# ❌ 1つのことしかしないのに過度に抽象化
class AbstractUserValidatorFactory(ABC):
    @abstractmethod
    def create_validator(self) -> AbstractUserValidator:
        pass

class AbstractUserValidator(ABC):
    @abstractmethod
    def validate(self, user: AbstractUserDTO) -> AbstractValidationResult:
        pass

class AbstractUserDTO(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

class AbstractValidationResult(ABC):
    @abstractmethod
    def is_valid(self) -> bool:
        pass

class ConcreteUserValidatorFactory(AbstractUserValidatorFactory):
    def create_validator(self):
        return ConcreteUserValidator()

class ConcreteUserValidator(AbstractUserValidator):
    def validate(self, user):
        return ConcreteValidationResult(bool(user.get_name()))

# ↑ 数百行のコードで実現していることは...

# ✅ これだけで十分
def validate_user(name: str) -> bool:
    return bool(name)
```

### 12.2 早すぎる最適化

```python
# ❌ 早すぎる最適化
# まだプロファイリングもしていないのにビット演算で最適化
def is_even(n):
    return not (n & 1)  # 読みにくい

# ✅ まず可読性を優先
def is_even(n):
    return n % 2 == 0  # 明快

# ✅ パフォーマンスが本当に問題になった場合のみ最適化する
# そしてコメントで理由を記載する
# パフォーマンスプロファイルの結果、この関数が全体の30%を占めていたため
# ビット演算に最適化（ベンチマーク: 2.1ms → 0.8ms, Issue #567）
def is_even_optimized(n):
    return not (n & 1)
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 命名 | 意図を明確に。検索可能に。一貫性を保つ |
| 関数 | 小さく。1つの責任。引数は少なく |
| コードの臭い | 長い関数、重複、深いネスト → リファクタリング |
| SOLID | SRP, OCP, LSP, ISP, DIP の5原則を守る |
| 原則 | DRY, KISS, YAGNI |
| コメント | WHYを書く。コードが語れることは書かない |
| エラー処理 | 具体的な例外、カスタム例外、リソース管理 |
| テスト | テスタビリティ重視の設計。AAA パターン |
| リファクタリング | 小さなステップで安全に。テストが先 |
| レビュー | 具体的に、理由を添えて、提案を含めて |

---

## 次に読むべきガイド
→ [[04-system-design-basics.md]] — システム設計入門

---

## 参考文献
1. Martin, R. C. "Clean Code." Prentice Hall, 2008.
2. Fowler, M. "Refactoring." 2nd Edition, Addison-Wesley, 2018.
3. Martin, R. C. "Clean Architecture." Prentice Hall, 2017.
4. Hunt, A. and Thomas, D. "The Pragmatic Programmer." 20th Anniversary Edition, Addison-Wesley, 2019.
5. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018.
6. Kernighan, B. W. and Pike, R. "The Practice of Programming." Addison-Wesley, 1999.
7. McConnell, S. "Code Complete." 2nd Edition, Microsoft Press, 2004.
8. Beck, K. "Test Driven Development: By Example." Addison-Wesley, 2002.
