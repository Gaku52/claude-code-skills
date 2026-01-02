# 可読性レビューガイド

## 概要

可読性レビューは、コードが他の開発者にとって理解しやすいかを評価します。命名、コメント、フォーマット、コードの複雑度などを確認し、保守性の高いコードを目指します。

## 目次

1. [命名規則](#命名規則)
2. [コメントとドキュメント](#コメントとドキュメント)
3. [コードの複雑度](#コードの複雑度)
4. [フォーマットとスタイル](#フォーマットとスタイル)
5. [マジックナンバー](#マジックナンバー)
6. [言語別ベストプラクティス](#言語別ベストプラクティス)

---

## 命名規則

### 基本原則

1. **意図を明確に表現する**
2. **誤解を招かない**
3. **検索しやすい**
4. **発音できる**

### 変数名

```typescript
// ❌ Bad: 意味不明な略語
const d = new Date();
const usr = getUsr();
const tmp = calc(a, b);

// ✅ Good: 明確な名前
const currentDate = new Date();
const currentUser = getCurrentUser();
const totalPrice = calculateTotalPrice(basePrice, taxRate);

// ❌ Bad: 型名を含む（ハンガリアン記法）
const strName = 'John';
const arrUsers = [];
const objConfig = {};

// ✅ Good: 意味のある名前
const userName = 'John';
const activeUsers = [];
const appConfig = {};

// ❌ Bad: 一文字変数（ループ以外）
const n = users.length;
const x = calculateValue();

// ✅ Good: 具体的な名前
const userCount = users.length;
const discountAmount = calculateDiscount();

// ✅ ループ変数は慣習的に短い名前でOK
for (let i = 0; i < items.length; i++) {
  // ...
}

// ただし、ネストが深い場合は明確な名前を
for (const user of users) {
  for (const order of user.orders) {
    for (const item of order.items) {
      // userIndex, orderIndex, itemIndex などより明確
    }
  }
}
```

### 関数名

```python
# ❌ Bad: 動作が不明確
def process(data):
    pass

def handle(x, y):
    pass

def do_it():
    pass

# ✅ Good: 動作が明確
def validate_email(email: str) -> bool:
    pass

def calculate_tax(amount: float, rate: float) -> float:
    pass

def send_welcome_email(user: User) -> None:
    pass

# ❌ Bad: 副作用が隠されている
def get_user(user_id: str) -> User:
    user = db.find(user_id)
    user.last_accessed = datetime.now()  # 副作用！
    db.save(user)
    return user

# ✅ Good: 名前から副作用が想像できる
def get_and_update_user_access_time(user_id: str) -> User:
    user = db.find(user_id)
    user.last_accessed = datetime.now()
    db.save(user)
    return user

# または、副作用を分離
def get_user(user_id: str) -> User:
    return db.find(user_id)

def update_user_access_time(user: User) -> None:
    user.last_accessed = datetime.now()
    db.save(user)
```

### クラス名

```swift
// ❌ Bad: 曖昧な名前
class Manager {
    // 何をマネージするの？
}

class Helper {
    // 何を手伝うの？
}

class Data {
    // どんなデータ？
}

// ✅ Good: 明確な責務
class UserSessionManager {
    func startSession(for user: User) { }
    func endSession() { }
}

class ImageDownloadHelper {
    func download(from url: URL) -> UIImage? { }
}

class UserProfileData {
    let name: String
    let email: String
}

// ❌ Bad: 動詞で始まるクラス名
class CalculatePrice {
    // クラスは「もの」を表す名詞にすべき
}

// ✅ Good: 名詞または名詞句
class PriceCalculator {
    func calculate(for items: [Item]) -> Decimal { }
}
```

### Boolean変数

```go
// ❌ Bad: 疑問文でない
var status bool
var flag bool
var check bool

// ✅ Good: is, has, can, shouldなど
var isActive bool
var hasPermission bool
var canEdit bool
var shouldRetry bool

// ❌ Bad: 否定形
var isNotValid bool
var hasNoErrors bool

// ✅ Good: 肯定形
var isValid bool
var hasErrors bool

// 使用時
if !isValid {
    // 二重否定を避ける
}

// ❌ Bad: 曖昧
var enabled bool  // 何が？

// ✅ Good: 明確
var isUserEnabled bool
var isFeatureEnabled bool
var isAutoSaveEnabled bool
```

---

## コメントとドキュメント

### 良いコメントの原則

1. **WHYを説明する（WHATではない）**
2. **複雑なロジックを明確にする**
3. **警告や制約を伝える**
4. **TODO/FIXMEを記録する**

### コメントの例

```typescript
// ❌ Bad: コードを繰り返すだけ
// ユーザーIDを取得する
const userId = user.id;

// カウンターをインクリメント
counter++;

// ✅ Good: WHYを説明
// パフォーマンス改善のため、最初の100件のみ処理
// 全件処理すると5秒以上かかることが判明（Issue #123）
const itemsToProcess = items.slice(0, 100);

// Safari 14以下では crypto.randomUUID() が未サポート
// フォールバックとして Math.random() を使用
const id = crypto.randomUUID?.() ?? generateFallbackId();

// ❌ Bad: 古いコードをコメントアウト
// function oldCalculation(x) {
//   return x * 2;
// }
// function deprecatedMethod() {
//   // ...
// }

// ✅ Good: Gitで管理、不要なら削除

// ✅ Good: 制約や警告
/**
 * ユーザーを削除する
 *
 * WARNING: この操作は取り消せません。
 * 削除前に必ず確認ダイアログを表示すること。
 *
 * @param userId - 削除するユーザーのID
 * @throws {UserNotFoundError} ユーザーが存在しない
 * @throws {PermissionError} 削除権限がない
 */
async function deleteUser(userId: string): Promise<void> {
  // ...
}

// ✅ Good: 複雑なアルゴリズムの説明
/**
 * Luhnアルゴリズムでクレジットカード番号を検証
 * https://en.wikipedia.org/wiki/Luhn_algorithm
 *
 * 1. 右から左に、2桁おきに2倍する
 * 2. 2倍した結果が9より大きい場合、9を引く
 * 3. すべての桁を合計
 * 4. 合計が10で割り切れればOK
 */
function validateCreditCard(cardNumber: string): boolean {
  // ...
}

// ✅ Good: TODO/FIXME
// TODO(john): API v2に移行後、この処理を削除 (2024-03-01)
// FIXME: 大量データで遅い。ページネーション実装が必要
// HACK: 一時的な回避策。Issue #456で根本的に修正予定
```

### ドキュメントコメント

```python
# ✅ Good: 包括的なdocstring
def calculate_discount(
    order_total: float,
    customer_tier: str,
    promotion_code: Optional[str] = None
) -> float:
    """
    注文合計から割引額を計算する

    顧客ランクとプロモーションコードに基づいて割引を適用します。
    プロモーションコードが無効な場合は通常割引のみ適用されます。

    Args:
        order_total: 注文合計金額（税抜）
        customer_tier: 顧客ランク ('bronze', 'silver', 'gold', 'platinum')
        promotion_code: プロモーションコード（オプション）

    Returns:
        割引金額（0以上）

    Raises:
        ValueError: order_totalが負の値
        ValueError: customer_tierが無効

    Examples:
        >>> calculate_discount(1000, 'gold')
        100.0
        >>> calculate_discount(1000, 'gold', 'SUMMER2024')
        200.0

    Note:
        プロモーションコードと顧客ランク割引は併用可能ですが、
        合計割引率は50%を超えません。
    """
    if order_total < 0:
        raise ValueError("Order total cannot be negative")

    # ...
```

---

## コードの複雑度

### Cyclomatic Complexity（循環的複雑度）

```swift
// ❌ Bad: 複雑度が高い（CC = 10）
func processOrder(order: Order) -> Bool {
    if order.items.isEmpty {
        return false
    }

    if order.customer == nil {
        return false
    }

    if !order.customer.isActive {
        return false
    }

    if order.total < 0 {
        return false
    }

    if order.total > 10000 && !order.customer.isVerified {
        return false
    }

    if order.shippingAddress == nil {
        return false
    }

    if !isValidAddress(order.shippingAddress) {
        return false
    }

    if order.paymentMethod == nil {
        return false
    }

    // ... さらに条件が続く

    return true
}

// ✅ Good: 複雑度を下げる（Early Return + 関数分割）
func processOrder(order: Order) -> Bool {
    guard validateOrderBasics(order) else { return false }
    guard validateCustomer(order.customer) else { return false }
    guard validateShipping(order) else { return false }
    guard validatePayment(order) else { return false }

    return true
}

private func validateOrderBasics(_ order: Order) -> Bool {
    return !order.items.isEmpty && order.total >= 0
}

private func validateCustomer(_ customer: Customer?) -> Bool {
    guard let customer = customer else { return false }
    return customer.isActive
}

private func validateShipping(_ order: Order) -> Bool {
    guard let address = order.shippingAddress else { return false }
    return isValidAddress(address)
}

private func validatePayment(_ order: Order) -> Bool {
    guard let payment = order.paymentMethod else { return false }

    if order.total > 10000 {
        return order.customer.isVerified
    }

    return true
}
```

### 関数の長さ

```go
// ❌ Bad: 長すぎる関数（100行以上）
func ProcessUserRegistration(req *http.Request) error {
    // パース
    var input struct {
        Email    string
        Password string
        Name     string
    }
    json.NewDecoder(req.Body).Decode(&input)

    // バリデーション
    if input.Email == "" {
        return errors.New("email required")
    }
    // ... 20行のバリデーション

    // パスワードハッシュ化
    hash, _ := bcrypt.GenerateFromPassword([]byte(input.Password), 10)
    // ...

    // ユーザー作成
    user := &User{
        Email:    input.Email,
        Password: string(hash),
        Name:     input.Name,
    }

    // DB保存
    // ... 10行

    // メール送信
    // ... 20行

    // ログ記録
    // ... 10行

    // レスポンス作成
    // ... 10行

    return nil
}

// ✅ Good: 小さな関数に分割
func ProcessUserRegistration(req *http.Request) error {
    input, err := parseRegistrationInput(req)
    if err != nil {
        return fmt.Errorf("parse input: %w", err)
    }

    if err := validateRegistrationInput(input); err != nil {
        return fmt.Errorf("validation: %w", err)
    }

    user, err := createUser(input)
    if err != nil {
        return fmt.Errorf("create user: %w", err)
    }

    if err := sendWelcomeEmail(user); err != nil {
        log.Printf("Failed to send welcome email: %v", err)
        // メール送信失敗は致命的でないので続行
    }

    return nil
}

func parseRegistrationInput(req *http.Request) (*RegistrationInput, error) {
    // 10行程度
}

func validateRegistrationInput(input *RegistrationInput) error {
    // 20行程度
}

func createUser(input *RegistrationInput) (*User, error) {
    // 15行程度
}

func sendWelcomeEmail(user *User) error {
    // 10行程度
}
```

### ネストの深さ

```typescript
// ❌ Bad: ネストが深い
function processData(data: any) {
  if (data) {
    if (data.user) {
      if (data.user.profile) {
        if (data.user.profile.settings) {
          if (data.user.profile.settings.notifications) {
            if (data.user.profile.settings.notifications.email) {
              // やっと処理
              return data.user.profile.settings.notifications.email;
            }
          }
        }
      }
    }
  }
  return null;
}

// ✅ Good: Early ReturnとOptional Chaining
function processData(data: any): string | null {
  // Optional chainingで簡潔に
  return data?.user?.profile?.settings?.notifications?.email ?? null;
}

// または、Guard句で
function processDataWithGuard(data: any): string | null {
  if (!data) return null;
  if (!data.user) return null;
  if (!data.user.profile) return null;
  if (!data.user.profile.settings) return null;
  if (!data.user.profile.settings.notifications) return null;

  return data.user.profile.settings.notifications.email;
}
```

---

## フォーマットとスタイル

### 一貫性

```python
# ❌ Bad: 不一致なスタイル
def calculate_total(items):
    total=0  # スペースなし
    for item in items:
        total += item.price  # スペースあり
    return total

def CalculateDiscount( amount ):  # 命名規則が違う、無駄なスペース
    return amount*0.1  # スペースなし

# ✅ Good: 一貫したスタイル
def calculate_total(items: List[Item]) -> float:
    total = 0
    for item in items:
        total += item.price
    return total


def calculate_discount(amount: float) -> float:
    return amount * 0.1
```

### 行の長さ

```typescript
// ❌ Bad: 長すぎる行
const result = await fetchUserData(userId).then(user => processUserData(user, { includeOrders: true, includePayments: true, includeShipping: true })).catch(error => handleError(error));

// ✅ Good: 適切に改行
const result = await fetchUserData(userId)
  .then(user =>
    processUserData(user, {
      includeOrders: true,
      includePayments: true,
      includeShipping: true,
    })
  )
  .catch(error => handleError(error));

// または
const options = {
  includeOrders: true,
  includePayments: true,
  includeShipping: true,
};

const user = await fetchUserData(userId);
const result = await processUserData(user, options);
```

### 空行の使い方

```swift
// ❌ Bad: 空行が少なすぎる
class UserService {
    private let repository: UserRepository
    private let validator: UserValidator
    init(repository: UserRepository, validator: UserValidator) {
        self.repository = repository
        self.validator = validator
    }
    func createUser(name: String, email: String) throws -> User {
        try validator.validate(name: name, email: email)
        let user = User(name: name, email: email)
        try repository.save(user)
        return user
    }
    func deleteUser(id: String) throws {
        guard let user = try repository.findById(id) else {
            throw UserError.notFound
        }
        try repository.delete(user)
    }
}

// ✅ Good: 適切な空行
class UserService {
    private let repository: UserRepository
    private let validator: UserValidator

    init(repository: UserRepository, validator: UserValidator) {
        self.repository = repository
        self.validator = validator
    }

    func createUser(name: String, email: String) throws -> User {
        try validator.validate(name: name, email: email)

        let user = User(name: name, email: email)
        try repository.save(user)

        return user
    }

    func deleteUser(id: String) throws {
        guard let user = try repository.findById(id) else {
            throw UserError.notFound
        }

        try repository.delete(user)
    }
}
```

---

## マジックナンバー

```python
# ❌ Bad: マジックナンバー
def calculate_shipping(weight):
    if weight < 5:
        return 500
    elif weight < 10:
        return 1000
    else:
        return 1500

def is_valid_age(age):
    return 18 <= age <= 120

def process_orders(orders):
    batch_size = 100  # ローカル変数だが意図不明
    for i in range(0, len(orders), batch_size):
        batch = orders[i:i + batch_size]
        process_batch(batch)

# ✅ Good: 名前付き定数
# 定数は大文字のスネークケース
SHIPPING_WEIGHT_THRESHOLD_LIGHT = 5  # kg
SHIPPING_WEIGHT_THRESHOLD_MEDIUM = 10  # kg
SHIPPING_COST_LIGHT = 500  # 円
SHIPPING_COST_MEDIUM = 1000  # 円
SHIPPING_COST_HEAVY = 1500  # 円

MIN_ADULT_AGE = 18
MAX_HUMAN_AGE = 120

# バッチサイズの根拠をコメント
# DB接続プールのサイズ（200）の半分に設定
# これにより同時に2つのバッチ処理が可能
ORDER_PROCESSING_BATCH_SIZE = 100

def calculate_shipping(weight: float) -> int:
    if weight < SHIPPING_WEIGHT_THRESHOLD_LIGHT:
        return SHIPPING_COST_LIGHT
    elif weight < SHIPPING_WEIGHT_THRESHOLD_MEDIUM:
        return SHIPPING_COST_MEDIUM
    else:
        return SHIPPING_COST_HEAVY

def is_valid_age(age: int) -> bool:
    return MIN_ADULT_AGE <= age <= MAX_HUMAN_AGE

def process_orders(orders: List[Order]) -> None:
    for i in range(0, len(orders), ORDER_PROCESSING_BATCH_SIZE):
        batch = orders[i:i + ORDER_PROCESSING_BATCH_SIZE]
        process_batch(batch)
```

---

## 言語別ベストプラクティス

### TypeScript

```typescript
// ✅ 型定義を活用
type UserId = string;  // 型エイリアスで意図を明確に
type Email = string;

interface User {
  id: UserId;
  email: Email;
  name: string;
}

// ✅ Enumで選択肢を明確に
enum OrderStatus {
  Pending = 'PENDING',
  Processing = 'PROCESSING',
  Shipped = 'SHIPPED',
  Delivered = 'DELIVERED',
  Cancelled = 'CANCELLED',
}

// ✅ Nullableを明示
function findUser(id: UserId): User | null {
  // null が返る可能性を型で表現
}

// ✅ ジェネリクスで再利用性
function first<T>(items: T[]): T | undefined {
  return items[0];
}
```

### Python

```python
# ✅ Type hintsを使用
from typing import List, Optional, Dict

def get_users(
    status: str,
    limit: Optional[int] = None
) -> List[Dict[str, any]]:
    """
    ユーザーリストを取得

    Args:
        status: ユーザーステータス
        limit: 取得件数（Noneの場合は全件）

    Returns:
        ユーザー情報の辞書のリスト
    """
    pass

# ✅ リスト内包表記で簡潔に
# Bad
active_users = []
for user in users:
    if user.is_active:
        active_users.append(user)

# Good
active_users = [user for user in users if user.is_active]

# ✅ コンテキストマネージャーでリソース管理
with open('file.txt') as f:
    content = f.read()
# 自動的にクローズされる
```

### Swift

```swift
// ✅ Guard文でEarly Return
func processUser(_ user: User?) {
    guard let user = user else {
        print("User is nil")
        return
    }

    guard user.isActive else {
        print("User is not active")
        return
    }

    // メイン処理
}

// ✅ Extensionでコードを整理
extension String {
    var isValidEmail: Bool {
        let emailRegex = "[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"
        let predicate = NSPredicate(format: "SELF MATCHES %@", emailRegex)
        return predicate.evaluate(with: self)
    }
}

// ✅ Propertyの明示的な可視性
class UserService {
    private let repository: UserRepository  // privateを明示
    public var currentUser: User?  // publicを明示

    internal func updateCache() {  // internalを明示（デフォルト）
        // ...
    }
}
```

### Go

```go
// ✅ エラーハンドリングを明示
func GetUser(id string) (*User, error) {
    if id == "" {
        return nil, errors.New("id is required")
    }

    user, err := db.FindUser(id)
    if err != nil {
        return nil, fmt.Errorf("find user: %w", err)
    }

    return user, nil
}

// ✅ deferでクリーンアップ
func ProcessFile(path string) error {
    file, err := os.Open(path)
    if err != nil {
        return err
    }
    defer file.Close()  // 確実にクローズ

    // 処理
    return nil
}

// ✅ インターフェースは小さく
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

// 必要な機能だけを要求
func Copy(dst Writer, src Reader) error {
    // ...
}
```

---

## レビューチェックリスト

### 可読性レビュー完全チェックリスト

#### 命名
- [ ] 変数名が意図を明確に表現している
- [ ] 関数名が動作を明確に表現している
- [ ] Boolean変数がis/has/canなどで始まる
- [ ] マジックナンバーが定数化されている

#### コメント
- [ ] WHYが説明されている（WHATではない）
- [ ] 複雑なロジックに説明がある
- [ ] TODO/FIXMEに期限と担当者がある
- [ ] 古いコードがコメントアウトされていない

#### 複雑度
- [ ] 関数が短い（20-30行以内）
- [ ] ネストが浅い（3レベル以内）
- [ ] 循環的複雑度が低い（10以下）

#### フォーマット
- [ ] コーディング規約に従っている
- [ ] インデントが一貫している
- [ ] 行の長さが適切（80-120文字）
- [ ] 空行が適切に使われている

---

## まとめ

可読性の高いコードは、チームの生産性を大きく向上させます。

### 重要ポイント

1. **意図が明確な命名**
2. **WHYを説明するコメント**
3. **シンプルで短い関数**
4. **一貫したスタイル**
5. **マジックナンバーの排除**

### 次のステップ

- [テストレビュー](04-testing.md)
- [保守性レビュー](07-maintainability.md)
- [自動化ガイド](12-automation.md)
