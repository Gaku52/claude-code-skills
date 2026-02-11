# 保守性レビューガイド

## 概要

保守性レビューは、コードの長期的な変更容易性、拡張性、ドキュメンテーションを評価します。将来の開発者が理解しやすく、変更しやすいコードを目指します。

## 目次

1. [技術的負債](#技術的負債)
2. [リファクタリング](#リファクタリング)
3. [ドキュメント](#ドキュメント)
4. [拡張性](#拡張性)
5. [依存関係管理](#依存関係管理)
6. [後方互換性](#後方互換性)

---

## 技術的負債

### 負債の種類と対策

#### 1. 意図的な負債

```typescript
// ✅ 意図的な負債は文書化する
/**
 * ユーザー検索機能
 *
 * TODO(john): パフォーマンス改善が必要
 * 現在は全ユーザーをメモリに読み込んでいるが、
 * ユーザー数が10,000を超えたらページネーション実装が必要。
 * 期限: 2024-03-31
 * Issue: #1234
 */
function searchUsers(query: string): User[] {
  const allUsers = database.getAllUsers();  // FIXME: スケーラビリティ問題
  return allUsers.filter(user =>
    user.name.toLowerCase().includes(query.toLowerCase())
  );
}
```

#### 2. 無意識の負債

```python
# ❌ Bad: 無意識の負債（重複コード）
def calculate_order_total_for_japan(items):
    subtotal = sum(item.price * item.quantity for item in items)
    tax = subtotal * 0.1
    shipping = 500 if subtotal < 5000 else 0
    return subtotal + tax + shipping

def calculate_order_total_for_usa(items):
    subtotal = sum(item.price * item.quantity for item in items)
    tax = subtotal * 0.08
    shipping = 10 if subtotal < 50 else 0
    return subtotal + tax + shipping

# ✅ Good: 共通化してDRY原則に従う
class TaxConfig:
    def __init__(self, rate: float, free_shipping_threshold: float, shipping_cost: float):
        self.rate = rate
        self.free_shipping_threshold = free_shipping_threshold
        self.shipping_cost = shipping_cost

TAX_CONFIGS = {
    'JP': TaxConfig(rate=0.1, free_shipping_threshold=5000, shipping_cost=500),
    'US': TaxConfig(rate=0.08, free_shipping_threshold=50, shipping_cost=10),
}

def calculate_order_total(items: List[Item], country_code: str) -> float:
    config = TAX_CONFIGS.get(country_code)
    if not config:
        raise ValueError(f"Unknown country code: {country_code}")

    subtotal = sum(item.price * item.quantity for item in items)
    tax = subtotal * config.rate
    shipping = 0 if subtotal >= config.free_shipping_threshold else config.shipping_cost

    return subtotal + tax + shipping
```

### 負債の測定

```bash
# Code Climate
# 技術的負債を自動測定
npm install -g codeclimate

codeclimate analyze

# SonarQube
# 複雑度、重複、カバレッジを測定
sonar-scanner \
  -Dsonar.projectKey=my-project \
  -Dsonar.sources=./src
```

---

## リファクタリング

### リファクタリングの原則

1. **テストを書いてから**
2. **小さなステップで**
3. **動作を変えない**
4. **コミットを細かく**

### Extract Method

```swift
// ❌ Bad: 長いメソッド
func processOrder(_ order: Order) throws {
    // 検証
    guard !order.items.isEmpty else {
        throw OrderError.emptyOrder
    }

    guard let customer = order.customer else {
        throw OrderError.noCustomer
    }

    guard customer.isActive else {
        throw OrderError.inactiveCustomer
    }

    // 計算
    var subtotal: Decimal = 0
    for item in order.items {
        subtotal += item.price * Decimal(item.quantity)
    }

    let tax = subtotal * 0.1
    let shipping = subtotal < 5000 ? 500 : 0
    let total = subtotal + tax + shipping

    // 保存
    order.total = total
    try database.save(order)

    // 通知
    let message = "Order \(order.id) total: ¥\(total)"
    try emailService.send(to: customer.email, message: message)
}

// ✅ Good: メソッド抽出
func processOrder(_ order: Order) throws {
    try validateOrder(order)

    let total = calculateOrderTotal(order)
    order.total = total

    try saveOrder(order)
    try sendOrderConfirmation(order)
}

private func validateOrder(_ order: Order) throws {
    guard !order.items.isEmpty else {
        throw OrderError.emptyOrder
    }

    guard let customer = order.customer, customer.isActive else {
        throw OrderError.invalidCustomer
    }
}

private func calculateOrderTotal(_ order: Order) -> Decimal {
    let subtotal = order.items.reduce(0) { $0 + $1.price * Decimal($1.quantity) }
    let tax = subtotal * 0.1
    let shipping = subtotal < 5000 ? 500 : 0
    return subtotal + tax + shipping
}

private func saveOrder(_ order: Order) throws {
    try database.save(order)
}

private func sendOrderConfirmation(_ order: Order) throws {
    guard let customer = order.customer else { return }

    let message = "Order \(order.id) total: ¥\(order.total)"
    try emailService.send(to: customer.email, message: message)
}
```

### Replace Magic Number with Constant

```go
// ❌ Bad: マジックナンバー
func CalculateShipping(weight float64) float64 {
    if weight < 5 {
        return 500
    } else if weight < 10 {
        return 1000
    } else {
        return 1500
    }
}

// ✅ Good: 名前付き定数
const (
    ShippingWeightLight  = 5.0
    ShippingWeightMedium = 10.0

    ShippingCostLight  = 500
    ShippingCostMedium = 1000
    ShippingCostHeavy  = 1500
)

func CalculateShipping(weight float64) float64 {
    switch {
    case weight < ShippingWeightLight:
        return ShippingCostLight
    case weight < ShippingWeightMedium:
        return ShippingCostMedium
    default:
        return ShippingCostHeavy
    }
}
```

### Introduce Parameter Object

```typescript
// ❌ Bad: 多すぎるパラメータ
function createUser(
  name: string,
  email: string,
  age: number,
  address: string,
  city: string,
  country: string,
  postalCode: string,
  phone: string
): User {
  // ...
}

// ✅ Good: パラメータオブジェクト
interface CreateUserParams {
  name: string;
  email: string;
  age: number;
  address: {
    street: string;
    city: string;
    country: string;
    postalCode: string;
  };
  phone: string;
}

function createUser(params: CreateUserParams): User {
  // ...
}

// 使用例
const user = createUser({
  name: 'John Doe',
  email: 'john@example.com',
  age: 30,
  address: {
    street: '123 Main St',
    city: 'Tokyo',
    country: 'Japan',
    postalCode: '123-4567',
  },
  phone: '090-1234-5678',
});
```

---

## ドキュメント

### コードドキュメント

```python
# ✅ Good: 包括的なドキュメント
from typing import List, Optional
from datetime import datetime

class UserService:
    """
    ユーザー管理サービス

    ユーザーの作成、更新、削除などの操作を提供します。
    すべてのメソッドはトランザクション内で実行されます。

    Attributes:
        repository: ユーザーデータの永続化を担当
        validator: ユーザーデータの検証を担当
        email_service: メール送信を担当

    Example:
        >>> service = UserService(repo, validator, email)
        >>> user = service.create_user('john@example.com', 'SecurePass123')
        >>> print(user.id)
        '550e8400-e29b-41d4-a716-446655440000'
    """

    def __init__(
        self,
        repository: UserRepository,
        validator: UserValidator,
        email_service: EmailService
    ):
        self.repository = repository
        self.validator = validator
        self.email_service = email_service

    def create_user(
        self,
        email: str,
        password: str,
        name: Optional[str] = None
    ) -> User:
        """
        新しいユーザーを作成します

        メールアドレスとパスワードを検証し、パスワードをハッシュ化してから
        データベースに保存します。作成後、ウェルカムメールを送信します。

        Args:
            email: ユーザーのメールアドレス（重複不可）
            password: パスワード（最低12文字、大小英字・数字を含む）
            name: ユーザー名（オプション、未指定時はメールから生成）

        Returns:
            作成されたUserオブジェクト

        Raises:
            ValidationError: メールまたはパスワードが無効
            DuplicateEmailError: メールアドレスが既に登録済み
            DatabaseError: データベース操作が失敗

        Example:
            >>> user = service.create_user(
            ...     email='john@example.com',
            ...     password='SecurePass123',
            ...     name='John Doe'
            ... )
            >>> print(user.email)
            'john@example.com'

        Note:
            この操作はトランザクション内で実行されます。
            メール送信に失敗してもユーザー作成は完了します。
        """
        # 実装
        pass
```

### API ドキュメント

```typescript
/**
 * @api {post} /api/users ユーザー作成
 * @apiName CreateUser
 * @apiGroup User
 * @apiVersion 1.0.0
 *
 * @apiDescription 新しいユーザーアカウントを作成します
 *
 * @apiParam {String} email メールアドレス（必須、一意）
 * @apiParam {String} password パスワード（最低12文字）
 * @apiParam {String} [name] ユーザー名（オプション）
 *
 * @apiSuccess {String} id ユーザーID
 * @apiSuccess {String} email メールアドレス
 * @apiSuccess {String} name ユーザー名
 * @apiSuccess {String} createdAt 作成日時（ISO 8601形式）
 *
 * @apiSuccessExample {json} Success-Response:
 *     HTTP/1.1 201 Created
 *     {
 *       "id": "550e8400-e29b-41d4-a716-446655440000",
 *       "email": "john@example.com",
 *       "name": "John Doe",
 *       "createdAt": "2024-01-15T10:30:00Z"
 *     }
 *
 * @apiError ValidationError 入力値が無効
 * @apiError DuplicateEmail メールアドレスが既に使用されている
 *
 * @apiErrorExample {json} Error-Response:
 *     HTTP/1.1 400 Bad Request
 *     {
 *       "error": "ValidationError",
 *       "message": "Invalid email format"
 *     }
 */
router.post('/api/users', createUserHandler);
```

### README

```markdown
# User Service

ユーザー管理マイクロサービス

## 概要

このサービスは、ユーザーアカウントの作成、認証、プロフィール管理を提供します。

## 機能

- ✅ ユーザー登録
- ✅ メール認証
- ✅ パスワードリセット
- ✅ プロフィール更新
- ⏳ 2要素認証（実装中）

## セットアップ

### 必要な環境

- Node.js 18+
- PostgreSQL 14+
- Redis 7+

### インストール

\`\`\`bash
npm install
cp .env.example .env
npm run migrate
\`\`\`

### 起動

\`\`\`bash
# 開発環境
npm run dev

# 本番環境
npm run build
npm start
\`\`\`

## API

詳細は [API Documentation](docs/api.md) を参照

## アーキテクチャ

\`\`\`
src/
├── controllers/    # HTTPハンドラ
├── services/       # ビジネスロジック
├── repositories/   # データアクセス
├── models/         # データモデル
└── utils/          # ユーティリティ
\`\`\`

## テスト

\`\`\`bash
# すべてのテスト
npm test

# カバレッジ
npm run test:coverage
\`\`\`

## デプロイ

詳細は [Deployment Guide](docs/deployment.md) を参照

## トラブルシューティング

### データベース接続エラー

\`\`\`bash
# 接続設定を確認
echo $DATABASE_URL

# PostgreSQLが起動しているか確認
pg_isready -h localhost
\`\`\`

## ライセンス

MIT
```

---

## 拡張性

### Open/Closed Principle

```swift
// ❌ Bad: 拡張のたびに修正が必要
class PaymentProcessor {
    func process(amount: Decimal, method: String) throws {
        switch method {
        case "credit_card":
            processCreditCard(amount: amount)
        case "paypal":
            processPayPal(amount: amount)
        case "bitcoin":
            processBitcoin(amount: amount)
        // 新しい支払い方法を追加するたびにここを修正
        default:
            throw PaymentError.unsupportedMethod
        }
    }
}

// ✅ Good: 拡張に開いている
protocol PaymentMethod {
    func process(amount: Decimal) throws
}

class CreditCardPayment: PaymentMethod {
    func process(amount: Decimal) throws {
        // クレジットカード処理
    }
}

class PayPalPayment: PaymentMethod {
    func process(amount: Decimal) throws {
        // PayPal処理
    }
}

class PaymentProcessor {
    func process(amount: Decimal, method: PaymentMethod) throws {
        try method.process(amount: amount)
        // 新しい支払い方法を追加してもこのコードは変更不要
    }
}

// 新しい支払い方法を追加
class BitcoinPayment: PaymentMethod {
    func process(amount: Decimal) throws {
        // Bitcoin処理
    }
}
```

### Plugin Architecture

```typescript
// ✅ プラグインアーキテクチャ
interface Plugin {
  name: string;
  version: string;
  initialize(app: Application): void;
  shutdown(): Promise<void>;
}

class Application {
  private plugins: Map<string, Plugin> = new Map();

  registerPlugin(plugin: Plugin): void {
    console.log(`Registering plugin: ${plugin.name} v${plugin.version}`);
    plugin.initialize(this);
    this.plugins.set(plugin.name, plugin);
  }

  async shutdown(): Promise<void> {
    for (const plugin of this.plugins.values()) {
      await plugin.shutdown();
    }
  }
}

// プラグイン実装例
class LoggingPlugin implements Plugin {
  name = 'logging';
  version = '1.0.0';

  initialize(app: Application): void {
    // ロギング機能を追加
    console.log('Logging plugin initialized');
  }

  async shutdown(): Promise<void> {
    // クリーンアップ
  }
}

class CachePlugin implements Plugin {
  name = 'cache';
  version = '1.0.0';

  initialize(app: Application): void {
    // キャッシュ機能を追加
    console.log('Cache plugin initialized');
  }

  async shutdown(): Promise<void> {
    // キャッシュをフラッシュ
  }
}

// 使用例
const app = new Application();
app.registerPlugin(new LoggingPlugin());
app.registerPlugin(new CachePlugin());
```

---

## 依存関係管理

### 依存関係の最小化

```go
// ❌ Bad: 不要な依存関係
package user

import (
    "database/sql"
    "net/http"
    "encoding/json"
    "github.com/gorilla/mux"  // 使っていない
    "github.com/sirupsen/logrus"  // 標準logでも十分
)

// ✅ Good: 必要最小限
package user

import (
    "database/sql"
    "encoding/json"
    "log"
    "net/http"
)
```

### 依存関係の更新戦略

```json
// package.json
{
  "dependencies": {
    // ✅ メジャーバージョンは固定、マイナー/パッチは自動更新
    "express": "^4.18.0",

    // ✅ セキュリティクリティカルなライブラリは厳密に固定
    "jsonwebtoken": "9.0.0",

    // ❌ ワイルドカードは避ける
    // "lodash": "*"
  },
  "devDependencies": {
    // 開発ツールは最新を使用
    "typescript": "^5.0.0",
    "jest": "^29.0.0"
  }
}
```

---

## 後方互換性

### API バージョニング

```python
# ✅ URLベースのバージョニング
@app.route('/api/v1/users/<user_id>', methods=['GET'])
def get_user_v1(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email
    })

@app.route('/api/v2/users/<user_id>', methods=['GET'])
def get_user_v2(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify({
        'id': user.id,
        'full_name': user.name,  # フィールド名変更
        'email': user.email,
        'created_at': user.created_at.isoformat(),  # 新フィールド
        'avatar_url': user.avatar_url  # 新フィールド
    })

# ✅ Deprecation警告
@app.route('/api/v1/users', methods=['POST'])
def create_user_v1():
    # Deprecation警告ヘッダー
    response = jsonify({'message': 'User created'})
    response.headers['Warning'] = '299 - "API v1 is deprecated. Please use v2."'
    response.headers['Sunset'] = 'Sat, 31 Dec 2024 23:59:59 GMT'
    return response
```

### 破壊的変更の管理

```typescript
// ✅ Deprecation パターン
class UserService {
  /**
   * @deprecated Use getUserById instead. Will be removed in v3.0.0
   */
  getUser(id: string): Promise<User> {
    console.warn('getUser is deprecated. Use getUserById instead.');
    return this.getUserById(id);
  }

  getUserById(id: string): Promise<User> {
    // 新しい実装
    return this.repository.findById(id);
  }
}

// ✅ Feature Flag
class OrderService {
  async createOrder(order: Order): Promise<Order> {
    if (config.features.newOrderFlow) {
      // 新しいフロー（一部ユーザーのみ）
      return this.createOrderV2(order);
    } else {
      // 従来のフロー
      return this.createOrderV1(order);
    }
  }

  private async createOrderV1(order: Order): Promise<Order> {
    // 従来の実装
  }

  private async createOrderV2(order: Order): Promise<Order> {
    // 新しい実装
  }
}
```

---

## レビューチェックリスト

### 保守性レビュー完全チェックリスト

#### 技術的負債
- [ ] 意図的な負債が文書化されている
- [ ] 重複コードがない
- [ ] 複雑度が適切

#### リファクタリング
- [ ] 長いメソッドが分割されている
- [ ] マジックナンバーが定数化
- [ ] パラメータ数が適切（3個以下）

#### ドキュメント
- [ ] コードにドキュメントがある
- [ ] APIドキュメントが最新
- [ ] READMEが充実

#### 拡張性
- [ ] Open/Closed原則に従っている
- [ ] 新機能追加が容易
- [ ] プラグイン可能

#### 依存関係
- [ ] 依存関係が最小限
- [ ] バージョンが適切
- [ ] セキュリティアップデート対応

#### 後方互換性
- [ ] 破壊的変更がドキュメント化
- [ ] Deprecation警告がある
- [ ] マイグレーションパスがある

---

## まとめ

保守性の高いコードは、長期的な開発コストを削減します。

### 重要ポイント

1. **技術的負債を管理**
2. **継続的リファクタリング**
3. **充実したドキュメント**
4. **拡張性を考慮**
5. **後方互換性を維持**

### 次のステップ

- [セルフレビュー](08-self-review.md)
- [レビュー実施](09-reviewing.md)
- [自動化](12-automation.md)
