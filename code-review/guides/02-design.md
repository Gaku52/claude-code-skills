# 設計レビューガイド

## 概要

設計レビューは、コードのアーキテクチャ、設計パターン、責務分離などを評価します。長期的な保守性と拡張性を確保するための重要なレビュー観点です。

## 目次

1. [アーキテクチャの準拠](#アーキテクチャの準拠)
2. [SOLID原則](#solid原則)
3. [デザインパターン](#デザインパターン)
4. [責務分離](#責務分離)
5. [依存関係管理](#依存関係管理)
6. [言語別設計パターン](#言語別設計パターン)
7. [アンチパターン](#アンチパターン)

---

## アーキテクチャの準拠

### レイヤードアーキテクチャ

#### 基本構造

```
Presentation Layer (UI)
    ↓
Application Layer (Use Cases)
    ↓
Domain Layer (Business Logic)
    ↓
Infrastructure Layer (DB, API)
```

#### TypeScript/React Example

```typescript
// ❌ Bad: レイヤーが混在
class UserProfile extends React.Component {
  async componentDidMount() {
    // UIコンポーネントにビジネスロジックとDB操作が混在
    const response = await fetch('/api/users/123');
    const user = await response.json();

    if (user.age >= 18) {
      user.discount = 0.1;
    }

    await fetch('/api/users/123', {
      method: 'PUT',
      body: JSON.stringify(user)
    });

    this.setState({ user });
  }
}

// ✅ Good: レイヤーを分離
// Presentation Layer
interface UserProfileProps {
  userId: string;
  userService: UserService;
}

function UserProfile({ userId, userService }: UserProfileProps) {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    loadUser();
  }, [userId]);

  async function loadUser() {
    const userData = await userService.getUserWithDiscount(userId);
    setUser(userData);
  }

  return <div>{user?.name}</div>;
}

// Application Layer
class UserService {
  constructor(
    private userRepository: UserRepository,
    private discountCalculator: DiscountCalculator
  ) {}

  async getUserWithDiscount(userId: string): Promise<User> {
    const user = await this.userRepository.findById(userId);
    const discount = this.discountCalculator.calculate(user);

    return {
      ...user,
      discount
    };
  }
}

// Domain Layer
class DiscountCalculator {
  calculate(user: User): number {
    return user.age >= 18 ? 0.1 : 0;
  }
}

// Infrastructure Layer
class UserRepository {
  constructor(private apiClient: ApiClient) {}

  async findById(id: string): Promise<User> {
    return this.apiClient.get(`/api/users/${id}`);
  }
}
```

### MVVMアーキテクチャ（iOS/Swift）

```swift
// ❌ Bad: ViewControllerにすべてが詰まっている
class UserViewController: UIViewController {
    @IBOutlet weak var nameLabel: UILabel!
    @IBOutlet weak var emailLabel: UILabel!

    private var userId: String?

    override func viewDidLoad() {
        super.viewDidLoad()
        loadUser()
    }

    private func loadUser() {
        // ネットワーク処理がView層に
        guard let userId = userId else { return }

        URLSession.shared.dataTask(
            with: URL(string: "https://api.example.com/users/\(userId)")!
        ) { data, response, error in
            guard let data = data else { return }

            // パース処理がView層に
            let user = try? JSONDecoder().decode(User.self, from: data)

            // UI更新がメインスレッドでない
            self.nameLabel.text = user?.name
            self.emailLabel.text = user?.email
        }.resume()
    }
}

// ✅ Good: MVVM分離
// View
class UserViewController: UIViewController {
    @IBOutlet weak var nameLabel: UILabel!
    @IBOutlet weak var emailLabel: UILabel!

    private let viewModel: UserViewModel

    init(viewModel: UserViewModel) {
        self.viewModel = viewModel
        super.init(nibName: nil, bundle: nil)
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        bindViewModel()
        viewModel.loadUser()
    }

    private func bindViewModel() {
        viewModel.onUserLoaded = { [weak self] user in
            DispatchQueue.main.async {
                self?.nameLabel.text = user.name
                self?.emailLabel.text = user.email
            }
        }

        viewModel.onError = { [weak self] error in
            self?.showError(error)
        }
    }
}

// ViewModel
class UserViewModel {
    var onUserLoaded: ((User) -> Void)?
    var onError: ((Error) -> Void)?

    private let userRepository: UserRepository
    private let userId: String

    init(userId: String, userRepository: UserRepository) {
        self.userId = userId
        self.userRepository = userRepository
    }

    func loadUser() {
        userRepository.fetchUser(id: userId) { [weak self] result in
            switch result {
            case .success(let user):
                self?.onUserLoaded?(user)
            case .failure(let error):
                self?.onError?(error)
            }
        }
    }
}

// Repository
protocol UserRepository {
    func fetchUser(id: String, completion: @escaping (Result<User, Error>) -> Void)
}

class APIUserRepository: UserRepository {
    private let networkService: NetworkService

    init(networkService: NetworkService) {
        self.networkService = networkService
    }

    func fetchUser(id: String, completion: @escaping (Result<User, Error>) -> Void) {
        networkService.request(endpoint: .user(id: id), completion: completion)
    }
}
```

---

## SOLID原則

### 1. Single Responsibility Principle (単一責任の原則)

```python
# ❌ Bad: 複数の責任
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def save(self):
        """データベース保存（永続化の責任）"""
        db.execute("INSERT INTO users ...")

    def send_welcome_email(self):
        """メール送信（通知の責任）"""
        smtp.send(self.email, "Welcome!")

    def validate(self):
        """検証（ビジネスルールの責任）"""
        if "@" not in self.email:
            raise ValueError("Invalid email")

# ✅ Good: 責任を分離
class User:
    """ユーザーエンティティ（ドメインモデル）"""
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

class UserValidator:
    """ユーザー検証（検証の責任）"""
    def validate(self, user: User) -> None:
        if "@" not in user.email:
            raise ValueError("Invalid email")

        if len(user.name) < 2:
            raise ValueError("Name too short")

class UserRepository:
    """ユーザー永続化（永続化の責任）"""
    def __init__(self, db):
        self.db = db

    def save(self, user: User) -> None:
        self.db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (user.name, user.email)
        )

class UserNotificationService:
    """ユーザー通知（通知の責任）"""
    def __init__(self, email_service):
        self.email_service = email_service

    def send_welcome_email(self, user: User) -> None:
        self.email_service.send(
            to=user.email,
            subject="Welcome!",
            body=f"Hello {user.name}, welcome to our service!"
        )
```

### 2. Open/Closed Principle (オープン・クローズドの原則)

```typescript
// ❌ Bad: 修正に開いている
class PaymentProcessor {
  processPayment(amount: number, method: string): void {
    if (method === 'credit_card') {
      this.processCreditCard(amount);
    } else if (method === 'paypal') {
      this.processPayPal(amount);
    } else if (method === 'bitcoin') {
      this.processBitcoin(amount);
    }
    // 新しい支払い方法を追加するたびにこのメソッドを修正
  }
}

// ✅ Good: 拡張に開き、修正に閉じている
interface PaymentMethod {
  process(amount: number): Promise<void>;
}

class CreditCardPayment implements PaymentMethod {
  async process(amount: number): Promise<void> {
    // クレジットカード処理
  }
}

class PayPalPayment implements PaymentMethod {
  async process(amount: number): Promise<void> {
    // PayPal処理
  }
}

class BitcoinPayment implements PaymentMethod {
  async process(amount: number): Promise<void> {
    // Bitcoin処理
  }
}

class PaymentProcessor {
  async processPayment(
    amount: number,
    paymentMethod: PaymentMethod
  ): Promise<void> {
    await paymentMethod.process(amount);
    // 新しい支払い方法を追加してもこのメソッドは修正不要
  }
}
```

### 3. Liskov Substitution Principle (リスコフの置換原則)

```swift
// ❌ Bad: 置換できない
class Bird {
    func fly() {
        print("Flying")
    }
}

class Penguin: Bird {
    override func fly() {
        // ペンギンは飛べない！
        fatalError("Penguins can't fly")
    }
}

// ✅ Good: 適切な抽象化
protocol Bird {
    var name: String { get }
}

protocol FlyingBird: Bird {
    func fly()
}

protocol SwimmingBird: Bird {
    func swim()
}

class Sparrow: FlyingBird {
    let name = "Sparrow"

    func fly() {
        print("Sparrow is flying")
    }
}

class Penguin: SwimmingBird {
    let name = "Penguin"

    func swim() {
        print("Penguin is swimming")
    }
}
```

### 4. Interface Segregation Principle (インターフェース分離の原則)

```go
// ❌ Bad: 巨大なインターフェース
type Worker interface {
    Work()
    Eat()
    Sleep()
    GetPaid()
}

type Robot struct{}

func (r Robot) Work() {
    // ロボットは働く
}

func (r Robot) Eat() {
    // ロボットは食べない！
    panic("robots don't eat")
}

func (r Robot) Sleep() {
    // ロボットは寝ない！
    panic("robots don't sleep")
}

func (r Robot) GetPaid() {
    // ロボットは給料をもらわない！
    panic("robots don't get paid")
}

// ✅ Good: 小さなインターフェースに分離
type Worker interface {
    Work()
}

type Eater interface {
    Eat()
}

type Sleeper interface {
    Sleep()
}

type Payable interface {
    GetPaid()
}

type Human struct{}

func (h Human) Work() {
    // 働く
}

func (h Human) Eat() {
    // 食べる
}

func (h Human) Sleep() {
    // 寝る
}

func (h Human) GetPaid() {
    // 給料をもらう
}

type Robot struct{}

func (r Robot) Work() {
    // ロボットは働く
}

// ロボットはWorkerだけ実装すれば良い
```

### 5. Dependency Inversion Principle (依存性逆転の原則)

```typescript
// ❌ Bad: 上位モジュールが下位モジュールに依存
class MySQLDatabase {
  query(sql: string): any[] {
    // MySQL固有の実装
  }
}

class UserService {
  private db: MySQLDatabase;  // 具象クラスに依存

  constructor() {
    this.db = new MySQLDatabase();  // 密結合
  }

  getUser(id: string): User {
    const results = this.db.query(`SELECT * FROM users WHERE id = ${id}`);
    return results[0];
  }
}

// ✅ Good: 両方が抽象に依存
interface Database {
  query(sql: string): Promise<any[]>;
}

class MySQLDatabase implements Database {
  async query(sql: string): Promise<any[]> {
    // MySQL固有の実装
  }
}

class PostgreSQLDatabase implements Database {
  async query(sql: string): Promise<any[]> {
    // PostgreSQL固有の実装
  }
}

class UserService {
  constructor(private db: Database) {}  // 抽象に依存

  async getUser(id: string): Promise<User> {
    const results = await this.db.query(
      `SELECT * FROM users WHERE id = '${id}'`
    );
    return results[0];
  }
}

// 使用時にDI
const mysqlDb = new MySQLDatabase();
const userService = new UserService(mysqlDb);

// データベースを変更しても UserService は変更不要
const postgresDb = new PostgreSQLDatabase();
const userService2 = new UserService(postgresDb);
```

---

## デザインパターン

### 1. Factory Pattern

```python
# ✅ Good: Factory Pattern
from abc import ABC, abstractmethod
from enum import Enum

class NotificationType(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"

class Notification(ABC):
    @abstractmethod
    def send(self, message: str, recipient: str) -> None:
        pass

class EmailNotification(Notification):
    def send(self, message: str, recipient: str) -> None:
        print(f"Sending email to {recipient}: {message}")

class SMSNotification(Notification):
    def send(self, message: str, recipient: str) -> None:
        print(f"Sending SMS to {recipient}: {message}")

class PushNotification(Notification):
    def send(self, message: str, recipient: str) -> None:
        print(f"Sending push to {recipient}: {message}")

class NotificationFactory:
    """Factory: 通知オブジェクトの生成を集約"""
    @staticmethod
    def create(notification_type: NotificationType) -> Notification:
        if notification_type == NotificationType.EMAIL:
            return EmailNotification()
        elif notification_type == NotificationType.SMS:
            return SMSNotification()
        elif notification_type == NotificationType.PUSH:
            return PushNotification()
        else:
            raise ValueError(f"Unknown notification type: {notification_type}")

# 使用例
def notify_user(user_id: str, message: str, preference: NotificationType):
    notification = NotificationFactory.create(preference)
    notification.send(message, user_id)
```

### 2. Strategy Pattern

```swift
// ✅ Good: Strategy Pattern
protocol SortStrategy {
    func sort<T: Comparable>(_ array: [T]) -> [T]
}

class BubbleSortStrategy: SortStrategy {
    func sort<T: Comparable>(_ array: [T]) -> [T] {
        var arr = array
        for i in 0..<arr.count {
            for j in 0..<(arr.count - i - 1) {
                if arr[j] > arr[j + 1] {
                    arr.swapAt(j, j + 1)
                }
            }
        }
        return arr
    }
}

class QuickSortStrategy: SortStrategy {
    func sort<T: Comparable>(_ array: [T]) -> [T] {
        guard array.count > 1 else { return array }

        let pivot = array[array.count / 2]
        let less = array.filter { $0 < pivot }
        let equal = array.filter { $0 == pivot }
        let greater = array.filter { $0 > pivot }

        return sort(less) + equal + sort(greater)
    }
}

class Sorter<T: Comparable> {
    private var strategy: SortStrategy

    init(strategy: SortStrategy) {
        self.strategy = strategy
    }

    func setStrategy(_ strategy: SortStrategy) {
        self.strategy = strategy
    }

    func sort(_ array: [T]) -> [T] {
        return strategy.sort(array)
    }
}

// 使用例
let numbers = [5, 2, 8, 1, 9]

let sorter = Sorter<Int>(strategy: QuickSortStrategy())
let sorted1 = sorter.sort(numbers)

sorter.setStrategy(BubbleSortStrategy())
let sorted2 = sorter.sort(numbers)
```

### 3. Observer Pattern

```typescript
// ✅ Good: Observer Pattern
interface Observer<T> {
  update(data: T): void;
}

class Subject<T> {
  private observers: Set<Observer<T>> = new Set();

  subscribe(observer: Observer<T>): void {
    this.observers.add(observer);
  }

  unsubscribe(observer: Observer<T>): void {
    this.observers.delete(observer);
  }

  notify(data: T): void {
    this.observers.forEach(observer => observer.update(data));
  }
}

// 具体的な実装
interface StockPrice {
  symbol: string;
  price: number;
}

class StockExchange extends Subject<StockPrice> {
  updatePrice(symbol: string, price: number): void {
    this.notify({ symbol, price });
  }
}

class StockDisplay implements Observer<StockPrice> {
  update(data: StockPrice): void {
    console.log(`Display: ${data.symbol} = $${data.price}`);
  }
}

class StockLogger implements Observer<StockPrice> {
  update(data: StockPrice): void {
    console.log(`[${new Date().toISOString()}] ${data.symbol}: ${data.price}`);
  }
}

// 使用例
const exchange = new StockExchange();
const display = new StockDisplay();
const logger = new StockLogger();

exchange.subscribe(display);
exchange.subscribe(logger);

exchange.updatePrice('AAPL', 150.25);
```

---

## 責務分離

### レイヤー別の責務

```typescript
// ✅ Good: 明確な責務分離

// Domain Layer: ビジネスロジックのみ
class Order {
  constructor(
    public readonly id: string,
    public readonly items: OrderItem[],
    public readonly customerId: string
  ) {}

  calculateTotal(): number {
    return this.items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  }

  canBeCancelled(): boolean {
    return this.status === 'pending';
  }
}

// Application Layer: ユースケースの調整
class OrderService {
  constructor(
    private orderRepository: OrderRepository,
    private paymentService: PaymentService,
    private emailService: EmailService
  ) {}

  async placeOrder(customerId: string, items: OrderItem[]): Promise<Order> {
    // ビジネスルール適用
    const order = new Order(generateId(), items, customerId);

    // 永続化
    await this.orderRepository.save(order);

    // 支払い処理
    await this.paymentService.charge(customerId, order.calculateTotal());

    // 通知
    await this.emailService.sendOrderConfirmation(customerId, order);

    return order;
  }
}

// Infrastructure Layer: 技術的詳細
class OrderRepository {
  async save(order: Order): Promise<void> {
    // データベース操作
  }

  async findById(id: string): Promise<Order | null> {
    // データベース操作
  }
}

// Presentation Layer: UI/API
class OrderController {
  constructor(private orderService: OrderService) {}

  async handlePlaceOrder(req: Request, res: Response): Promise<void> {
    try {
      const { customerId, items } = req.body;
      const order = await this.orderService.placeOrder(customerId, items);
      res.json({ success: true, order });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  }
}
```

---

## 依存関係管理

### 依存性注入（DI）

```go
// ✅ Good: Constructor Injection
type UserService struct {
    repo   UserRepository
    cache  Cache
    logger Logger
}

func NewUserService(
    repo UserRepository,
    cache Cache,
    logger Logger,
) *UserService {
    return &UserService{
        repo:   repo,
        cache:  cache,
        logger: logger,
    }
}

func (s *UserService) GetUser(id string) (*User, error) {
    // キャッシュチェック
    if user, err := s.cache.Get(id); err == nil {
        s.logger.Info("Cache hit for user", "id", id)
        return user, nil
    }

    // データベースから取得
    user, err := s.repo.FindByID(id)
    if err != nil {
        s.logger.Error("Failed to get user", "id", id, "error", err)
        return nil, err
    }

    // キャッシュに保存
    s.cache.Set(id, user)

    return user, nil
}
```

---

## アンチパターン

### 1. God Object（神オブジェクト）

```python
# ❌ Bad: すべてを知っている巨大クラス
class Application:
    def __init__(self):
        self.db = Database()
        self.cache = Cache()
        self.logger = Logger()
        self.config = Config()

    def handle_user_registration(self, data):
        # ユーザー登録
        pass

    def handle_order_processing(self, order):
        # 注文処理
        pass

    def handle_payment(self, payment):
        # 支払い処理
        pass

    def send_email(self, email):
        # メール送信
        pass

    # ... 100個以上のメソッド

# ✅ Good: 責務を分離
class UserService:
    def register(self, data): pass

class OrderService:
    def process(self, order): pass

class PaymentService:
    def charge(self, payment): pass

class EmailService:
    def send(self, email): pass
```

### 2. Circular Dependency（循環依存）

```typescript
// ❌ Bad: 循環依存
// user.service.ts
import { OrderService } from './order.service';

class UserService {
  constructor(private orderService: OrderService) {}

  getUserOrders(userId: string) {
    return this.orderService.getOrdersByUser(userId);
  }
}

// order.service.ts
import { UserService } from './user.service';

class OrderService {
  constructor(private userService: UserService) {}

  getOrdersByUser(userId: string) {
    const user = this.userService.getUser(userId);
    // ...
  }
}

// ✅ Good: 中間レイヤーで解決
// user.service.ts
class UserService {
  constructor(private userRepository: UserRepository) {}

  getUser(userId: string): Promise<User> {
    return this.userRepository.findById(userId);
  }
}

// order.service.ts
class OrderService {
  constructor(private orderRepository: OrderRepository) {}

  getOrdersByUser(userId: string): Promise<Order[]> {
    return this.orderRepository.findByUserId(userId);
  }
}

// user-order.service.ts (調整レイヤー)
class UserOrderService {
  constructor(
    private userService: UserService,
    private orderService: OrderService
  ) {}

  async getUserWithOrders(userId: string): Promise<UserWithOrders> {
    const user = await this.userService.getUser(userId);
    const orders = await this.orderService.getOrdersByUser(userId);

    return { user, orders };
  }
}
```

---

## レビューチェックリスト

### 設計レビュー完全チェックリスト

#### アーキテクチャ
- [ ] 適切なアーキテクチャパターンが使用されている
- [ ] レイヤー間の依存関係が正しい方向
- [ ] ドメインロジックがインフラから分離されている

#### SOLID原則
- [ ] 各クラスが単一の責任を持つ
- [ ] 拡張に開き、修正に閉じている
- [ ] 抽象が適切に使用されている
- [ ] インターフェースが小さく分離されている
- [ ] 依存性が逆転している

#### デザインパターン
- [ ] 適切なパターンが使用されている
- [ ] パターンが過剰に使われていない
- [ ] パターンの意図が明確

#### 依存関係
- [ ] 循環依存がない
- [ ] DIが適切に使用されている
- [ ] テスト可能な設計になっている

---

## まとめ

設計レビューは、長期的なコード品質を保証します。

### 重要ポイント

1. **アーキテクチャに従う**
2. **SOLID原則を守る**
3. **適切なパターンを使う**
4. **責務を分離する**
5. **依存関係を管理する**

### 次のステップ

- [可読性レビュー](03-readability.md)
- [テストレビュー](04-testing.md)
- [保守性レビュー](07-maintainability.md)
