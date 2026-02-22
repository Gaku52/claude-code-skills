# DIP（依存性逆転の原則）

> DIPは「上位モジュールは下位モジュールに依存すべきでない。どちらも抽象に依存すべき」。依存性注入（DI）と IoC コンテナを通じて、テスト容易で疎結合なシステムを実現する。

## この章で学ぶこと

- [ ] 依存性逆転の「逆転」が何を意味するか理解する
- [ ] 依存性注入の3つのパターンを把握する
- [ ] IoC コンテナの仕組みと実践的な使い方を学ぶ
- [ ] DIP を活用したアーキテクチャ設計を習得する
- [ ] 実務でのDIPの適用判断基準を身につける

---

## 1. 依存性逆転とは

```
従来の依存関係（上位が下位に依存）:

  ┌──────────────┐
  │ OrderService │ ← 上位モジュール（ビジネスロジック）
  └──────┬───────┘
         │ 依存（直接参照）
         ▼
  ┌──────────────┐
  │ MySQLDatabase│ ← 下位モジュール（インフラ）
  └──────────────┘

  問題: MySQLをPostgreSQLに変更すると OrderService も変更が必要

依存性逆転後（どちらも抽象に依存）:

  ┌──────────────┐     ┌────────────┐
  │ OrderService │────→│ Database   │ ← 抽象（インターフェース）
  └──────────────┘     └────────────┘
                             ↑
                       ┌─────┴──────┐
                       │            │
                ┌──────────┐  ┌──────────┐
                │ MySQL    │  │ Postgres │
                └──────────┘  └──────────┘

  「逆転」の意味:
  → 従来: 上位 → 下位
  → DIP:  上位 → 抽象 ← 下位
  → 下位モジュールが抽象に依存する方向に「逆転」
```

### 1.1 DIPの歴史的背景

```
1996年:
  Robert C. Martin が "The Dependency Inversion Principle" を発表
  SOLID原則の最後の "D" として整理

なぜ「逆転（Inversion）」か:
  従来の手続き型プログラミング:
    上位モジュール → 下位モジュール（直接呼び出し）
    上位が下位の実装詳細に依存

  DIP適用後:
    上位モジュール → 抽象（インターフェース）
    下位モジュール → 抽象（インターフェース）
    下位モジュールが「抽象に合わせる」方向に依存が逆転

DIPの2つのルール:
  1. 上位モジュールは下位モジュールに依存してはならない。
     どちらも抽象に依存すべきである。
  2. 抽象は詳細に依存してはならない。
     詳細が抽象に依存すべきである。

「抽象」= インターフェース、抽象クラス
「詳細」= 具象クラス、具体的な実装
```

### 1.2 DIPなしとありの比較

```typescript
// ❌ DIP違反: 上位が下位の具象クラスに直接依存
class OrderService {
  // MySQLDatabase の具体的な実装に直接依存
  private db = new MySQLDatabase("localhost", 3306, "orders_db");
  // SendGrid の具体的な実装に直接依存
  private mailer = new SendGridMailer("api-key-xxx");
  // Stripe の具体的な実装に直接依存
  private payment = new StripePayment("sk_live_xxx");

  async createOrder(dto: CreateOrderDto): Promise<Order> {
    const order = new Order(dto);
    // MySQLの固有メソッド
    await this.db.mysqlQuery("INSERT INTO orders ...", order);
    // SendGridの固有メソッド
    await this.mailer.sendViaSendGrid(order.email, "注文確認");
    // Stripeの固有メソッド
    await this.payment.stripeCharge(order.total, dto.stripeToken);
    return order;
  }
}

// 問題点:
// 1. MySQLをPostgreSQLに変更 → OrderService を書き換え
// 2. SendGridをAWS SESに変更 → OrderService を書き換え
// 3. テスト時に外部APIを呼んでしまう
// 4. OrderService のテストにDB接続が必要
```

```typescript
// ✅ DIP準拠: 上位も下位も抽象に依存
// --- 抽象（インターフェース）---
interface OrderRepository {
  save(order: Order): Promise<void>;
  findById(id: string): Promise<Order | null>;
}

interface MailService {
  send(to: string, subject: string, body: string): Promise<void>;
}

interface PaymentService {
  charge(amount: number, token: string): Promise<PaymentResult>;
}

// --- 上位モジュール: 抽象にのみ依存 ---
class OrderService {
  constructor(
    private readonly repo: OrderRepository,
    private readonly mailer: MailService,
    private readonly payment: PaymentService,
  ) {}

  async createOrder(dto: CreateOrderDto): Promise<Order> {
    const order = new Order(dto);
    const paymentResult = await this.payment.charge(order.total, dto.paymentToken);
    if (!paymentResult.success) {
      throw new PaymentFailedError(paymentResult.error);
    }
    await this.repo.save(order);
    await this.mailer.send(order.email, "注文確認", "ご注文ありがとうございます。");
    return order;
  }
}

// --- 下位モジュール: 抽象を実装 ---
class MySQLOrderRepository implements OrderRepository {
  constructor(private db: MySQLConnection) {}
  async save(order: Order): Promise<void> { /* MySQL固有の実装 */ }
  async findById(id: string): Promise<Order | null> { /* ... */ }
}

class PostgresOrderRepository implements OrderRepository {
  constructor(private db: PgPool) {}
  async save(order: Order): Promise<void> { /* Postgres固有の実装 */ }
  async findById(id: string): Promise<Order | null> { /* ... */ }
}

class SendGridMailService implements MailService {
  constructor(private apiKey: string) {}
  async send(to: string, subject: string, body: string): Promise<void> { /* ... */ }
}

class AwsSesMailService implements MailService {
  constructor(private sesClient: SESClient) {}
  async send(to: string, subject: string, body: string): Promise<void> { /* ... */ }
}

class StripePaymentService implements PaymentService {
  constructor(private secretKey: string) {}
  async charge(amount: number, token: string): Promise<PaymentResult> { /* ... */ }
}
```

---

## 2. 依存性注入（Dependency Injection）

```
DI = オブジェクトの依存関係を外部から注入する技術

3つのパターン:
  1. コンストラクタ注入（推奨）
  2. セッター注入
  3. インターフェース注入

DIの利点:
  → テスト時にモックを注入できる
  → 実装の差し替えが容易
  → 依存関係が明示的（コンストラクタを見れば分かる）
  → 循環依存の検出が容易
```

### 2.1 コンストラクタ注入（推奨）

```typescript
// === コンストラクタ注入（推奨）===
interface Logger {
  log(message: string): void;
}

interface UserRepository {
  findById(id: string): Promise<User | null>;
  save(user: User): Promise<void>;
}

interface EmailService {
  send(to: string, subject: string, body: string): Promise<void>;
}

class UserService {
  // 全ての依存をコンストラクタで受け取る
  constructor(
    private readonly repo: UserRepository,
    private readonly email: EmailService,
    private readonly logger: Logger,
  ) {}

  async register(data: CreateUserDto): Promise<User> {
    this.logger.log(`Registering user: ${data.email}`);
    const user = new User(data);
    await this.repo.save(user);
    await this.email.send(user.email, "Welcome!", "Registration complete.");
    return user;
  }
}

// テスト: モックを注入
class MockUserRepository implements UserRepository {
  private users: User[] = [];
  async findById(id: string) { return this.users.find(u => u.id === id) ?? null; }
  async save(user: User) { this.users.push(user); }
}

class MockEmailService implements EmailService {
  sent: { to: string; subject: string }[] = [];
  async send(to: string, subject: string) { this.sent.push({ to, subject }); }
}

// テストではモックを注入
const service = new UserService(
  new MockUserRepository(),
  new MockEmailService(),
  new ConsoleLogger(),
);
```

### 2.2 セッター注入

```typescript
// === セッター注入 ===
class ReportService {
  private formatter?: ReportFormatter;

  // セッターで注入（オプショナルな依存に適する）
  setFormatter(formatter: ReportFormatter): void {
    this.formatter = formatter;
  }

  generate(data: ReportData): string {
    const fmt = this.formatter ?? new DefaultFormatter();
    return fmt.format(data);
  }
}

// セッター注入の使いどころ:
// - オプショナルな依存（デフォルト実装がある場合）
// - 実行時に依存を差し替えたい場合
// - フレームワークが要求する場合

// セッター注入の注意点:
// - 必須依存にはコンストラクタ注入を使うべき
// - 呼び忘れのリスクがある
// - 不完全な状態のオブジェクトが存在しうる
```

### 2.3 インターフェース注入

```typescript
// === インターフェース注入 ===
// 注入用のインターフェースを定義
interface LoggerAware {
  setLogger(logger: Logger): void;
}

interface DatabaseAware {
  setDatabase(db: Database): void;
}

// 注入用インターフェースを実装
class UserService implements LoggerAware, DatabaseAware {
  private logger!: Logger;
  private db!: Database;

  setLogger(logger: Logger): void {
    this.logger = logger;
  }

  setDatabase(db: Database): void {
    this.db = db;
  }

  async getUser(id: string): Promise<User> {
    this.logger.log(`Getting user: ${id}`);
    return this.db.query("SELECT * FROM users WHERE id = ?", [id]);
  }
}

// インジェクター
class Injector {
  private logger = new FileLogger();
  private db = new PostgresDatabase();

  inject(service: any): void {
    if (this.isLoggerAware(service)) {
      service.setLogger(this.logger);
    }
    if (this.isDatabaseAware(service)) {
      service.setDatabase(this.db);
    }
  }

  private isLoggerAware(obj: any): obj is LoggerAware {
    return typeof obj.setLogger === "function";
  }

  private isDatabaseAware(obj: any): obj is DatabaseAware {
    return typeof obj.setDatabase === "function";
  }
}
```

### 2.4 DIパターンの比較

```
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│                  │ コンストラクタ│ セッター     │ IF注入       │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 必須依存         │ ○ 最適      │ △ 危険      │ △ 危険      │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ オプショナル依存 │ △ 可能      │ ○ 最適      │ △ 可能      │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 不変性保証       │ ○ readonly  │ ×           │ ×           │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 依存の明示性     │ ○ 明確      │ △ 分散      │ △ 分散      │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ テスト容易性     │ ○           │ ○           │ ○           │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 循環依存検出     │ ○ 即座      │ ×           │ ×           │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 推奨度           │ ★★★        │ ★★          │ ★            │
└──────────────────┴──────────────┴──────────────┴──────────────┘

結論: コンストラクタ注入を基本とし、
      オプショナル依存にのみセッター注入を使う
```

---

## 3. IoC コンテナ

```
IoC（Inversion of Control）コンテナ:
  → 依存関係の解決を自動化するフレームワーク
  → クラスの登録と依存グラフの自動解決

  手動DI:
    const logger = new FileLogger();
    const db = new PostgresDatabase(config);
    const repo = new UserRepository(db);
    const email = new SmtpEmailService(smtpConfig);
    const service = new UserService(repo, email, logger);
    // 依存が深くなると手動では管理困難

  IoCコンテナ:
    container.register(Logger, FileLogger);
    container.register(Database, PostgresDatabase);
    container.register(UserRepository);
    container.register(EmailService, SmtpEmailService);
    container.register(UserService);
    const service = container.resolve(UserService);
    // コンテナが依存グラフを自動解決

IoCコンテナのライフサイクル管理:
  Singleton:  アプリケーション全体で1つのインスタンス
  Transient:  resolve のたびに新しいインスタンスを作成
  Scoped:     リクエストごとに1つのインスタンス（Webアプリケーション）
```

### 3.1 NestJS での IoC コンテナ

```typescript
// NestJS: IoC コンテナの実践例
import { Injectable, Module, Inject } from "@nestjs/common";

// @Injectable() でコンテナに登録
@Injectable()
class UserRepository {
  constructor(private readonly db: DatabaseService) {}
  async findById(id: string): Promise<User | null> { /* ... */ }
}

@Injectable()
class UserService {
  constructor(
    private readonly repo: UserRepository, // 自動注入
    private readonly email: EmailService,  // 自動注入
  ) {}
}

// Module で依存関係を宣言
@Module({
  providers: [
    UserRepository,
    UserService,
    EmailService,
    { provide: "DATABASE", useClass: PostgresDatabase },
  ],
  controllers: [UserController],
})
class UserModule {}
```

```typescript
// NestJS: 高度な依存注入パターン

// 1. カスタムプロバイダー（値の注入）
@Module({
  providers: [
    {
      provide: "CONFIG",
      useValue: {
        apiUrl: "https://api.example.com",
        timeout: 5000,
      },
    },
  ],
})
class AppModule {}

@Injectable()
class ApiService {
  constructor(@Inject("CONFIG") private config: AppConfig) {}
}

// 2. ファクトリープロバイダー
@Module({
  providers: [
    {
      provide: "LOGGER",
      useFactory: (configService: ConfigService) => {
        const logLevel = configService.get("LOG_LEVEL");
        if (logLevel === "debug") {
          return new DebugLogger();
        }
        return new ProductionLogger();
      },
      inject: [ConfigService],
    },
  ],
})
class LoggingModule {}

// 3. 非同期プロバイダー
@Module({
  providers: [
    {
      provide: "DATABASE",
      useFactory: async (config: ConfigService) => {
        const pool = new Pool({
          host: config.get("DB_HOST"),
          port: config.get("DB_PORT"),
          database: config.get("DB_NAME"),
        });
        await pool.connect();
        return pool;
      },
      inject: [ConfigService],
    },
  ],
})
class DatabaseModule {}

// 4. スコープ付きプロバイダー
@Injectable({ scope: Scope.REQUEST })
class RequestContextService {
  private userId: string;

  setUserId(id: string): void {
    this.userId = id;
  }

  getUserId(): string {
    return this.userId;
  }
}
```

### 3.2 Python での DIコンテナ実装

```python
# Python: シンプルなDIコンテナの実装
from typing import Type, TypeVar, Dict, Any, Callable
import inspect

T = TypeVar("T")


class Container:
    """シンプルなDIコンテナ"""

    def __init__(self):
        self._registry: Dict[type, type] = {}
        self._singletons: Dict[type, Any] = {}
        self._factories: Dict[type, Callable] = {}
        self._scoped: Dict[str, Dict[type, Any]] = {}

    def register(self, interface: type, implementation: type) -> None:
        """インターフェースと実装のマッピングを登録"""
        self._registry[interface] = implementation

    def register_singleton(self, interface: type, implementation: type) -> None:
        """シングルトンとして登録"""
        self._registry[interface] = implementation
        # 最初の resolve 時にインスタンスを作成してキャッシュ

    def register_factory(self, interface: type, factory: Callable) -> None:
        """ファクトリー関数を登録"""
        self._factories[interface] = factory

    def register_instance(self, interface: type, instance: Any) -> None:
        """既存のインスタンスを登録"""
        self._singletons[interface] = instance

    def resolve(self, interface: Type[T]) -> T:
        """依存関係を解決してインスタンスを返す"""
        # シングルトンチェック
        if interface in self._singletons:
            return self._singletons[interface]

        # ファクトリーチェック
        if interface in self._factories:
            return self._factories[interface](self)

        # 登録済み実装の取得
        impl = self._registry.get(interface, interface)

        # コンストラクタの引数を自動解決
        params = inspect.signature(impl.__init__).parameters
        deps = {}
        for name, param in params.items():
            if name == "self":
                continue
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in self._registry or param.annotation in self._singletons:
                    deps[name] = self.resolve(param.annotation)

        instance = impl(**deps)
        return instance


# 使い方
from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def log(self, message: str) -> None: ...


class ConsoleLogger(Logger):
    def log(self, message: str) -> None:
        print(f"[LOG] {message}")


class FileLogger(Logger):
    def __init__(self, filepath: str = "app.log"):
        self.filepath = filepath

    def log(self, message: str) -> None:
        with open(self.filepath, "a") as f:
            f.write(f"{message}\n")


class UserRepository(ABC):
    @abstractmethod
    async def find_by_id(self, user_id: str): ...

    @abstractmethod
    async def save(self, user) -> None: ...


class PostgresUserRepository(UserRepository):
    def __init__(self, logger: Logger):
        self.logger = logger

    async def find_by_id(self, user_id: str):
        self.logger.log(f"Finding user: {user_id}")
        # Postgres固有の実装

    async def save(self, user) -> None:
        self.logger.log(f"Saving user: {user.id}")


class EmailService(ABC):
    @abstractmethod
    async def send(self, to: str, subject: str, body: str) -> None: ...


class SmtpEmailService(EmailService):
    def __init__(self, logger: Logger):
        self.logger = logger

    async def send(self, to: str, subject: str, body: str) -> None:
        self.logger.log(f"Sending email to {to}")


class UserService:
    def __init__(self, repo: UserRepository, email: EmailService, logger: Logger):
        self.repo = repo
        self.email = email
        self.logger = logger

    async def register(self, name: str, email_addr: str):
        self.logger.log(f"Registering user: {email_addr}")
        user = {"id": "new_id", "name": name, "email": email_addr}
        await self.repo.save(user)
        await self.email.send(email_addr, "Welcome!", "ご登録ありがとうございます。")
        return user


# コンテナの構築
container = Container()
container.register(Logger, ConsoleLogger)
container.register(UserRepository, PostgresUserRepository)
container.register(EmailService, SmtpEmailService)

# 依存が自動解決される
user_service = container.resolve(UserService)
# UserService(
#   repo=PostgresUserRepository(logger=ConsoleLogger()),
#   email=SmtpEmailService(logger=ConsoleLogger()),
#   logger=ConsoleLogger()
# )
```

### 3.3 Python: dependency-injector ライブラリ

```python
# dependency-injector: 本番利用に適したDIコンテナ
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject


class AppContainer(containers.DeclarativeContainer):
    """アプリケーションのDIコンテナ"""

    # 設定
    config = providers.Configuration()

    # ロガー（シングルトン）
    logger = providers.Singleton(
        FileLogger,
        filepath=config.log_file,
    )

    # データベース接続（シングルトン）
    database = providers.Singleton(
        PostgresDatabase,
        host=config.db.host,
        port=config.db.port,
        name=config.db.name,
    )

    # リポジトリ（ファクトリー: 毎回新規作成）
    user_repository = providers.Factory(
        PostgresUserRepository,
        db=database,
        logger=logger,
    )

    # メールサービス（シングルトン）
    email_service = providers.Singleton(
        SmtpEmailService,
        host=config.smtp.host,
        port=config.smtp.port,
        logger=logger,
    )

    # ユーザーサービス（ファクトリー）
    user_service = providers.Factory(
        UserService,
        repo=user_repository,
        email=email_service,
        logger=logger,
    )


# 使い方
container = AppContainer()
container.config.from_yaml("config.yml")

# テスト時のオーバーライド
with container.email_service.override(MockEmailService()):
    service = container.user_service()
    # MockEmailService が注入される
```

### 3.4 Java Spring での DI

```java
// Spring Framework: IoC コンテナ
import org.springframework.stereotype.Service;
import org.springframework.stereotype.Repository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;

// インターフェース定義
public interface OrderRepository {
    Order save(Order order);
    Optional<Order> findById(String id);
    List<Order> findByUserId(String userId);
}

public interface NotificationService {
    void notify(String userId, String message);
}

// 実装: @Repository アノテーションでコンテナに登録
@Repository
public class JpaOrderRepository implements OrderRepository {
    @Autowired
    private EntityManager em;

    @Override
    public Order save(Order order) {
        return em.merge(order);
    }

    @Override
    public Optional<Order> findById(String id) {
        return Optional.ofNullable(em.find(Order.class, id));
    }

    @Override
    public List<Order> findByUserId(String userId) {
        return em.createQuery(
            "SELECT o FROM Order o WHERE o.userId = :userId", Order.class)
            .setParameter("userId", userId)
            .getResultList();
    }
}

// 複数実装の切り替え: Profile
@Service
@Profile("production")
public class SlackNotificationService implements NotificationService {
    @Override
    public void notify(String userId, String message) {
        // Slack API 呼び出し
    }
}

@Service
@Profile("development")
public class ConsoleNotificationService implements NotificationService {
    @Override
    public void notify(String userId, String message) {
        System.out.printf("[NOTIFY] User: %s, Message: %s%n", userId, message);
    }
}

// サービス: コンストラクタ注入（推奨）
@Service
public class OrderService {
    private final OrderRepository repository;
    private final NotificationService notification;
    private final Logger logger;

    // Spring はコンストラクタが1つなら @Autowired 不要
    public OrderService(
        OrderRepository repository,
        NotificationService notification,
        Logger logger
    ) {
        this.repository = repository;
        this.notification = notification;
        this.logger = logger;
    }

    public Order createOrder(CreateOrderRequest request) {
        logger.info("Creating order for user: {}", request.getUserId());
        Order order = new Order(request);
        Order saved = repository.save(order);
        notification.notify(request.getUserId(), "注文が作成されました");
        return saved;
    }
}

// テスト
@SpringBootTest
class OrderServiceTest {
    @MockBean
    private OrderRepository mockRepository;

    @MockBean
    private NotificationService mockNotification;

    @Autowired
    private OrderService orderService;

    @Test
    void shouldCreateOrder() {
        // モックの設定
        when(mockRepository.save(any())).thenReturn(new Order("order-1"));

        // テスト実行
        Order result = orderService.createOrder(new CreateOrderRequest("user-1"));

        // 検証
        assertNotNull(result);
        verify(mockRepository).save(any());
        verify(mockNotification).notify(eq("user-1"), anyString());
    }
}
```

### 3.5 Go での DI（手動ワイヤリング）

```go
package main

import "context"

// インターフェース定義（Go では暗黙的に実装）
type UserRepository interface {
    FindByID(ctx context.Context, id string) (*User, error)
    Save(ctx context.Context, user *User) error
}

type EmailSender interface {
    Send(ctx context.Context, to, subject, body string) error
}

type Logger interface {
    Info(msg string, args ...any)
    Error(msg string, args ...any)
}

// サービス: コンストラクタ注入
type UserService struct {
    repo   UserRepository
    email  EmailSender
    logger Logger
}

func NewUserService(repo UserRepository, email EmailSender, logger Logger) *UserService {
    return &UserService{
        repo:   repo,
        email:  email,
        logger: logger,
    }
}

func (s *UserService) Register(ctx context.Context, name, emailAddr string) (*User, error) {
    s.logger.Info("registering user", "email", emailAddr)

    user := &User{Name: name, Email: emailAddr}
    if err := s.repo.Save(ctx, user); err != nil {
        s.logger.Error("failed to save user", "error", err)
        return nil, err
    }

    if err := s.email.Send(ctx, emailAddr, "Welcome!", "ご登録ありがとうございます。"); err != nil {
        s.logger.Error("failed to send welcome email", "error", err)
        // メール送信失敗は致命的ではないのでエラーを返さない
    }

    return user, nil
}

// 実装
type PostgresUserRepository struct {
    db *sql.DB
}

func NewPostgresUserRepository(db *sql.DB) *PostgresUserRepository {
    return &PostgresUserRepository{db: db}
}

func (r *PostgresUserRepository) FindByID(ctx context.Context, id string) (*User, error) {
    var user User
    err := r.db.QueryRowContext(ctx,
        "SELECT id, name, email FROM users WHERE id = $1", id).
        Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        return nil, err
    }
    return &user, nil
}

func (r *PostgresUserRepository) Save(ctx context.Context, user *User) error {
    _, err := r.db.ExecContext(ctx,
        "INSERT INTO users (id, name, email) VALUES ($1, $2, $3)",
        user.ID, user.Name, user.Email)
    return err
}

// 手動ワイヤリング（main関数）
func main() {
    // インフラストラクチャの初期化
    db, _ := sql.Open("postgres", "postgres://localhost/mydb")
    defer db.Close()

    logger := NewZapLogger()

    // 手動で依存グラフを構築
    repo := NewPostgresUserRepository(db)
    emailSender := NewSmtpEmailSender("smtp.example.com", 587)
    userService := NewUserService(repo, emailSender, logger)

    // テスト時
    // mockRepo := &MockUserRepository{}
    // mockEmail := &MockEmailSender{}
    // testService := NewUserService(mockRepo, mockEmail, logger)
}
```

```go
// Go: Wire を使った自動ワイヤリング
// +build wireinject

package main

import "github.com/google/wire"

// プロバイダセット
var UserSet = wire.NewSet(
    NewPostgresUserRepository,
    wire.Bind(new(UserRepository), new(*PostgresUserRepository)),
    NewSmtpEmailSender,
    wire.Bind(new(EmailSender), new(*SmtpEmailSender)),
    NewUserService,
)

// ワイヤリング関数
func InitializeUserService(db *sql.DB, smtpConfig SmtpConfig) *UserService {
    wire.Build(UserSet, NewZapLogger)
    return nil // wire が生成する
}
```

---

## 4. DIP の実践パターン

```
パターン1: リポジトリパターン
  ドメイン層 → Repository（インターフェース）← インフラ層

パターン2: ポート&アダプター（ヘキサゴナル）
  アプリケーション → Port（インターフェース）← Adapter（実装）

パターン3: プラグインアーキテクチャ
  コア → Plugin API ← 各プラグイン

共通点:
  → ビジネスロジック（上位）はインターフェース（抽象）のみに依存
  → インフラ（下位）がインターフェースを実装
  → インフラの差し替えがビジネスロジックに影響しない
```

### 4.1 ヘキサゴナルアーキテクチャ

```
ヘキサゴナルアーキテクチャ:

         ┌──── Adapter ────┐
         │   REST API      │
         └────────┬────────┘
                  ↓
    ┌──────── Port ─────────┐
    │  UserController(IF)   │
    └──────────┬────────────┘
               ↓
  ┌──── Application ────────┐
  │    UserService          │
  │    (ビジネスロジック)    │
  └──────────┬──────────────┘
             ↓
    ┌──────── Port ─────────┐
    │  UserRepository(IF)   │
    └──────────┬────────────┘
               ↓
         ┌──── Adapter ────┐
         │  PostgresRepo   │
         └─────────────────┘
```

```typescript
// ヘキサゴナルアーキテクチャの実装例

// === Port（インターフェース）===
// Driving Port（入力ポート）: ユースケースを定義
interface CreateOrderUseCase {
  execute(command: CreateOrderCommand): Promise<OrderDto>;
}

interface GetOrderUseCase {
  execute(query: GetOrderQuery): Promise<OrderDto>;
}

interface CancelOrderUseCase {
  execute(command: CancelOrderCommand): Promise<void>;
}

// Driven Port（出力ポート）: 外部依存を抽象化
interface OrderPersistencePort {
  save(order: Order): Promise<void>;
  findById(id: string): Promise<Order | null>;
  findByUserId(userId: string): Promise<Order[]>;
}

interface PaymentPort {
  processPayment(orderId: string, amount: number): Promise<PaymentResult>;
  refund(transactionId: string): Promise<RefundResult>;
}

interface NotificationPort {
  sendOrderConfirmation(order: Order): Promise<void>;
  sendCancellationNotice(order: Order): Promise<void>;
}

// === Application（ドメインサービス）===
class CreateOrderService implements CreateOrderUseCase {
  constructor(
    private readonly persistence: OrderPersistencePort,
    private readonly payment: PaymentPort,
    private readonly notification: NotificationPort,
  ) {}

  async execute(command: CreateOrderCommand): Promise<OrderDto> {
    // ドメインロジック
    const order = Order.create(command);
    order.validate();

    // 決済処理（Driven Port 経由）
    const paymentResult = await this.payment.processPayment(
      order.id, order.totalAmount
    );
    if (!paymentResult.success) {
      throw new PaymentFailedException(paymentResult.error);
    }
    order.markAsPaid(paymentResult.transactionId);

    // 永続化（Driven Port 経由）
    await this.persistence.save(order);

    // 通知（Driven Port 経由）
    await this.notification.sendOrderConfirmation(order);

    return OrderDto.fromDomain(order);
  }
}

// === Adapter（具体的な実装）===
// Driving Adapter: REST API
class OrderRestController {
  constructor(
    private readonly createOrder: CreateOrderUseCase,
    private readonly getOrder: GetOrderUseCase,
    private readonly cancelOrder: CancelOrderUseCase,
  ) {}

  async handlePost(req: Request, res: Response): Promise<void> {
    const command = new CreateOrderCommand(req.body);
    const result = await this.createOrder.execute(command);
    res.status(201).json(result);
  }

  async handleGet(req: Request, res: Response): Promise<void> {
    const query = new GetOrderQuery(req.params.id);
    const result = await this.getOrder.execute(query);
    res.status(200).json(result);
  }
}

// Driven Adapter: PostgreSQL
class PostgresOrderAdapter implements OrderPersistencePort {
  constructor(private readonly pool: Pool) {}

  async save(order: Order): Promise<void> {
    await this.pool.query(
      "INSERT INTO orders (id, user_id, total, status) VALUES ($1, $2, $3, $4)",
      [order.id, order.userId, order.totalAmount, order.status],
    );
  }

  async findById(id: string): Promise<Order | null> {
    const result = await this.pool.query(
      "SELECT * FROM orders WHERE id = $1", [id]
    );
    return result.rows[0] ? Order.fromPersistence(result.rows[0]) : null;
  }

  async findByUserId(userId: string): Promise<Order[]> {
    const result = await this.pool.query(
      "SELECT * FROM orders WHERE user_id = $1", [userId]
    );
    return result.rows.map(Order.fromPersistence);
  }
}

// Driven Adapter: Stripe
class StripePaymentAdapter implements PaymentPort {
  constructor(private readonly stripe: Stripe) {}

  async processPayment(orderId: string, amount: number): Promise<PaymentResult> {
    try {
      const charge = await this.stripe.charges.create({
        amount: amount * 100, // Stripeはセント単位
        currency: "jpy",
        metadata: { orderId },
      });
      return { success: true, transactionId: charge.id };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async refund(transactionId: string): Promise<RefundResult> {
    const refund = await this.stripe.refunds.create({ charge: transactionId });
    return { success: true, refundId: refund.id };
  }
}

// Driven Adapter: AWS SES
class SesNotificationAdapter implements NotificationPort {
  constructor(private readonly ses: SESClient) {}

  async sendOrderConfirmation(order: Order): Promise<void> {
    await this.ses.send(new SendEmailCommand({
      Destination: { ToAddresses: [order.email] },
      Message: {
        Subject: { Data: "注文確認" },
        Body: { Text: { Data: `注文 #${order.id} を受け付けました。` } },
      },
      Source: "noreply@example.com",
    }));
  }

  async sendCancellationNotice(order: Order): Promise<void> {
    // キャンセル通知の実装
  }
}
```

### 4.2 クリーンアーキテクチャとDIP

```
クリーンアーキテクチャの層構造:

  ┌─────────────────────────────────────────┐
  │         Frameworks & Drivers            │
  │  (Express, React, PostgreSQL, Redis)    │
  │  ┌─────────────────────────────────┐    │
  │  │    Interface Adapters           │    │
  │  │  (Controllers, Gateways, Repos) │    │
  │  │  ┌─────────────────────────┐    │    │
  │  │  │   Application Business  │    │    │
  │  │  │   Rules (Use Cases)     │    │    │
  │  │  │  ┌─────────────────┐    │    │    │
  │  │  │  │   Enterprise    │    │    │    │
  │  │  │  │   Business      │    │    │    │
  │  │  │  │   Rules         │    │    │    │
  │  │  │  │   (Entities)    │    │    │    │
  │  │  │  └─────────────────┘    │    │    │
  │  │  └─────────────────────────┘    │    │
  │  └─────────────────────────────────┘    │
  └─────────────────────────────────────────┘

依存の方向: 外側 → 内側（常に内向き）

  Frameworks  →  Adapters  →  Use Cases  →  Entities
  (具象)         (具象)        (抽象+具象)    (純粋ドメイン)

DIPの役割:
  内側の層は外側の層を知らない
  外側の層が内側のインターフェースを実装する
  → 依存の方向が常に内向き
```

```python
# クリーンアーキテクチャの実装例（Python）

# === Entities（最内層）===
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """注文エンティティ（ビジネスルール）"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    items: list = field(default_factory=list)
    total_amount: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    transaction_id: Optional[str] = None

    def confirm(self, transaction_id: str) -> None:
        """注文を確認する（ビジネスルール）"""
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"注文 {self.id} は確認できる状態ではありません")
        if self.total_amount <= 0:
            raise ValueError("合計金額は正の値である必要があります")
        self.status = OrderStatus.CONFIRMED
        self.transaction_id = transaction_id

    def cancel(self) -> None:
        """注文をキャンセルする（ビジネスルール）"""
        if self.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED):
            raise ValueError("出荷済み・配達済みの注文はキャンセルできません")
        self.status = OrderStatus.CANCELLED

    def can_be_cancelled(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.CONFIRMED)


# === Use Cases（アプリケーションビジネスルール）===
from abc import ABC, abstractmethod


class OrderRepository(ABC):
    """出力ポート: 永続化"""
    @abstractmethod
    async def save(self, order: Order) -> None: ...

    @abstractmethod
    async def find_by_id(self, order_id: str) -> Optional[Order]: ...


class PaymentGateway(ABC):
    """出力ポート: 決済"""
    @abstractmethod
    async def charge(self, amount: float, user_id: str) -> str: ...

    @abstractmethod
    async def refund(self, transaction_id: str) -> None: ...


class NotificationSender(ABC):
    """出力ポート: 通知"""
    @abstractmethod
    async def send_order_confirmation(self, order: Order) -> None: ...


@dataclass
class CreateOrderInput:
    user_id: str
    items: list
    total_amount: float


@dataclass
class CreateOrderOutput:
    order_id: str
    status: str
    transaction_id: str


class CreateOrderUseCase:
    """ユースケース: 注文作成"""

    def __init__(
        self,
        order_repo: OrderRepository,
        payment: PaymentGateway,
        notification: NotificationSender,
    ):
        self._order_repo = order_repo
        self._payment = payment
        self._notification = notification

    async def execute(self, input_data: CreateOrderInput) -> CreateOrderOutput:
        # エンティティの作成（ビジネスルール適用）
        order = Order(
            user_id=input_data.user_id,
            items=input_data.items,
            total_amount=input_data.total_amount,
        )

        # 決済（出力ポート経由）
        transaction_id = await self._payment.charge(
            order.total_amount, order.user_id
        )

        # ビジネスルール: 注文確認
        order.confirm(transaction_id)

        # 永続化（出力ポート経由）
        await self._order_repo.save(order)

        # 通知（出力ポート経由）
        await self._notification.send_order_confirmation(order)

        return CreateOrderOutput(
            order_id=order.id,
            status=order.status.value,
            transaction_id=transaction_id,
        )


# === Interface Adapters（アダプター層）===
class PostgresOrderRepository(OrderRepository):
    def __init__(self, db_pool):
        self._pool = db_pool

    async def save(self, order: Order) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO orders (id, user_id, total_amount, status, transaction_id)
                   VALUES ($1, $2, $3, $4, $5)
                   ON CONFLICT (id) DO UPDATE SET status = $4, transaction_id = $5""",
                order.id, order.user_id, order.total_amount,
                order.status.value, order.transaction_id,
            )

    async def find_by_id(self, order_id: str) -> Optional[Order]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM orders WHERE id = $1", order_id
            )
            if row is None:
                return None
            return Order(
                id=row["id"],
                user_id=row["user_id"],
                total_amount=row["total_amount"],
                status=OrderStatus(row["status"]),
                transaction_id=row["transaction_id"],
            )


class StripePaymentGateway(PaymentGateway):
    def __init__(self, api_key: str):
        import stripe
        stripe.api_key = api_key

    async def charge(self, amount: float, user_id: str) -> str:
        import stripe
        charge = stripe.Charge.create(
            amount=int(amount * 100),
            currency="jpy",
            source="tok_xxx",
            metadata={"user_id": user_id},
        )
        return charge.id

    async def refund(self, transaction_id: str) -> None:
        import stripe
        stripe.Refund.create(charge=transaction_id)
```

---

## 5. DIP のアンチパターン

```
1. サービスロケータ:
   → グローバルなコンテナから直接取得
   → 依存関係が暗黙的（コンストラクタを見ても分からない）
   → テストが困難

   ❌ const repo = ServiceLocator.get(UserRepository);
   ✅ constructor(private repo: UserRepository) {}

2. 過剰な抽象化:
   → 実装が1つしかないのにインターフェースを作る
   → 「将来変わるかも」で不要な抽象化
   → 実際に変更が必要になってから抽象化する

3. 循環依存:
   → A → B → C → A の依存ループ
   → インターフェースの導入位置を見直す

4. コンテナへの依存:
   → コンテナ自体を注入してしまう
   → ❌ constructor(private container: Container) {}
   → 何でも取得できる = 依存が不明確

5. 抽象の漏れ:
   → インターフェースが特定の実装に依存
   → ❌ interface Database { mysqlQuery(...): ... }
   → ✅ interface Database { query(...): ... }
```

### 5.1 サービスロケータのアンチパターン詳細

```typescript
// ❌ サービスロケータパターン（アンチパターン）
class ServiceLocator {
  private static instances = new Map<string, any>();

  static register(name: string, instance: any): void {
    this.instances.set(name, instance);
  }

  static get<T>(name: string): T {
    return this.instances.get(name) as T;
  }
}

// 使う側: 依存が暗黙的
class OrderService {
  async createOrder(dto: CreateOrderDto): Promise<Order> {
    // どこにも依存が宣言されていない
    const repo = ServiceLocator.get<OrderRepository>("orderRepo");
    const email = ServiceLocator.get<EmailService>("emailService");
    const logger = ServiceLocator.get<Logger>("logger");

    // 問題点:
    // 1. コンストラクタを見ても依存が分からない
    // 2. テスト前に ServiceLocator のセットアップが必要
    // 3. 依存の欠落がコンパイル時ではなく実行時に判明
    // 4. 静的メソッドなのでモック困難
  }
}

// ✅ コンストラクタ注入（推奨）
class OrderService {
  constructor(
    private readonly repo: OrderRepository,     // 依存が明示的
    private readonly email: EmailService,        // 依存が明示的
    private readonly logger: Logger,             // 依存が明示的
  ) {}

  // メリット:
  // 1. コンストラクタを見れば依存が一目瞭然
  // 2. テスト時にモックを直接渡せる
  // 3. 依存の欠落はコンパイル時に検出
  // 4. 循環依存もコンパイル時に検出
}
```

### 5.2 過剰な抽象化の回避

```typescript
// ❌ 過剰な抽象化: 実装が1つしかない
interface IUserValidator {
  validate(user: User): ValidationResult;
}

class UserValidator implements IUserValidator {
  validate(user: User): ValidationResult {
    // 唯一の実装
  }
}

// 「I」プレフィックスのインターフェースが大量に...
// interface IUserService → class UserService
// interface IOrderService → class OrderService
// interface IProductService → class ProductService

// ✅ インターフェースが有益な場合:
// 1. 外部サービスとの境界
interface PaymentGateway {
  charge(amount: number): Promise<Result>;
}
// → StripeGateway, MockGateway の2つの実装がある

// 2. テスト時のモックが必要
interface Clock {
  now(): Date;
}
// → SystemClock, FakeClock の2つの実装

// 3. 異なる永続化戦略
interface CacheStore {
  get(key: string): Promise<string | null>;
  set(key: string, value: string, ttl: number): Promise<void>;
}
// → RedisStore, MemoryStore, DynamoDBStore

// ✅ インターフェースが不要な場合:
// 純粋な計算ロジック、ユーティリティ関数
class TaxCalculator {
  calculate(amount: number, rate: number): number {
    return Math.floor(amount * rate);
  }
}
// → インターフェース不要。直接使って問題ない
```

### 5.3 循環依存の解消

```typescript
// ❌ 循環依存: A → B → C → A
class UserService {
  constructor(private orderService: OrderService) {} // UserService → OrderService
}

class OrderService {
  constructor(private paymentService: PaymentService) {} // OrderService → PaymentService
}

class PaymentService {
  constructor(private userService: UserService) {} // PaymentService → UserService ← 循環!
}

// ✅ 解決策1: インターフェースの導入
interface UserLookup {
  findById(id: string): Promise<User | null>;
}

class UserService implements UserLookup {
  constructor(private orderService: OrderService) {}
  async findById(id: string): Promise<User | null> { /* ... */ }
}

class PaymentService {
  // 具象クラスではなくインターフェースに依存
  constructor(private userLookup: UserLookup) {}
}

// ✅ 解決策2: イベントによる疎結合
interface EventBus {
  publish(event: DomainEvent): void;
  subscribe(eventType: string, handler: EventHandler): void;
}

class UserService {
  constructor(private eventBus: EventBus) {}

  async updateUser(userId: string, data: UpdateUserDto): Promise<void> {
    // 直接 OrderService を呼ばず、イベントを発行
    this.eventBus.publish(new UserUpdatedEvent(userId, data));
  }
}

class OrderService {
  constructor(private eventBus: EventBus) {
    // イベントを購読
    this.eventBus.subscribe("UserUpdated", this.handleUserUpdated.bind(this));
  }

  private async handleUserUpdated(event: UserUpdatedEvent): Promise<void> {
    // ユーザー更新に伴う注文の処理
  }
}

// ✅ 解決策3: メディエーターパターン
interface Mediator {
  send<T>(request: Request<T>): Promise<T>;
}

class GetUserQuery implements Request<User> {
  constructor(public readonly userId: string) {}
}

class PaymentService {
  constructor(private mediator: Mediator) {}

  async processPayment(userId: string, amount: number): Promise<void> {
    // 直接 UserService を呼ばず、メディエーター経由
    const user = await this.mediator.send(new GetUserQuery(userId));
    // 決済処理...
  }
}
```

---

## 6. テストとDIP

```typescript
// DIP が テスト容易性を飛躍的に向上させる例

// === テスト対象 ===
class NotificationService {
  constructor(
    private readonly emailSender: EmailSender,
    private readonly smsSender: SmsSender,
    private readonly pushSender: PushNotificationSender,
    private readonly userPreferences: UserPreferencesRepository,
    private readonly logger: Logger,
  ) {}

  async notifyUser(userId: string, message: string): Promise<NotifyResult> {
    const prefs = await this.userPreferences.getByUserId(userId);
    const results: string[] = [];

    if (prefs.emailEnabled) {
      await this.emailSender.send(prefs.email, "通知", message);
      results.push("email");
    }

    if (prefs.smsEnabled) {
      await this.smsSender.send(prefs.phone, message);
      results.push("sms");
    }

    if (prefs.pushEnabled) {
      await this.pushSender.send(prefs.deviceToken, message);
      results.push("push");
    }

    this.logger.log(`Notified user ${userId} via: ${results.join(", ")}`);
    return { channels: results };
  }
}

// === テスト: モックを注入 ===
describe("NotificationService", () => {
  let service: NotificationService;
  let mockEmail: jest.Mocked<EmailSender>;
  let mockSms: jest.Mocked<SmsSender>;
  let mockPush: jest.Mocked<PushNotificationSender>;
  let mockPrefs: jest.Mocked<UserPreferencesRepository>;
  let mockLogger: jest.Mocked<Logger>;

  beforeEach(() => {
    mockEmail = { send: jest.fn().mockResolvedValue(undefined) };
    mockSms = { send: jest.fn().mockResolvedValue(undefined) };
    mockPush = { send: jest.fn().mockResolvedValue(undefined) };
    mockPrefs = { getByUserId: jest.fn() };
    mockLogger = { log: jest.fn() };

    service = new NotificationService(
      mockEmail, mockSms, mockPush, mockPrefs, mockLogger,
    );
  });

  it("メール通知のみ有効な場合、メールだけ送信する", async () => {
    mockPrefs.getByUserId.mockResolvedValue({
      email: "test@example.com",
      emailEnabled: true,
      smsEnabled: false,
      pushEnabled: false,
    });

    const result = await service.notifyUser("user-1", "テストメッセージ");

    expect(mockEmail.send).toHaveBeenCalledWith(
      "test@example.com", "通知", "テストメッセージ"
    );
    expect(mockSms.send).not.toHaveBeenCalled();
    expect(mockPush.send).not.toHaveBeenCalled();
    expect(result.channels).toEqual(["email"]);
  });

  it("全チャネルが有効な場合、全て送信する", async () => {
    mockPrefs.getByUserId.mockResolvedValue({
      email: "test@example.com",
      phone: "+81901234567",
      deviceToken: "device-token-xxx",
      emailEnabled: true,
      smsEnabled: true,
      pushEnabled: true,
    });

    const result = await service.notifyUser("user-1", "テストメッセージ");

    expect(mockEmail.send).toHaveBeenCalled();
    expect(mockSms.send).toHaveBeenCalled();
    expect(mockPush.send).toHaveBeenCalled();
    expect(result.channels).toEqual(["email", "sms", "push"]);
  });

  it("メール送信が失敗した場合、エラーが伝播する", async () => {
    mockPrefs.getByUserId.mockResolvedValue({
      email: "test@example.com",
      emailEnabled: true,
      smsEnabled: false,
      pushEnabled: false,
    });
    mockEmail.send.mockRejectedValue(new Error("SMTP connection failed"));

    await expect(service.notifyUser("user-1", "テスト"))
      .rejects.toThrow("SMTP connection failed");
  });
});
```

---

## 7. DIPの適用判断基準

```
DIPを適用すべき場面:
  ✓ 外部サービスとの境界（DB, API, メール, ファイルシステム）
  ✓ テスト時にモックが必要な依存
  ✓ 将来的に実装が変わる可能性がある部分
  ✓ 複数の実装が存在する（本番, テスト, 開発）
  ✓ ビジネスロジックとインフラの分離が重要

DIPが不要な場面:
  ✗ 純粋な計算ロジック（Math, 文字列操作）
  ✗ 値オブジェクト（Money, Date, Address）
  ✗ ユーティリティ関数
  ✗ プロジェクトの規模が非常に小さい場合
  ✗ プロトタイプ段階

適用の段階的アプローチ:
  1. 最初は直接依存で書く（YAGNI）
  2. テストで困ったら → インターフェースを抽出
  3. 実装の差し替えが必要になったら → DIを導入
  4. 依存グラフが複雑になったら → IoCコンテナを検討

過剰適用の兆候:
  → 実装が1つしかないインターフェースが大量にある
  → コードを読むのにインターフェースと実装を行き来する
  → 新機能追加のたびに3ファイル以上を修正する
  → 「念のため」が理由のインターフェース
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| DIP | 上位も下位も抽象に依存 |
| DI | 依存を外部から注入 |
| コンストラクタ注入 | 最も推奨される方法 |
| IoC コンテナ | 依存解決の自動化 |
| ヘキサゴナル | DIPの代表的アーキテクチャ |
| クリーンアーキテクチャ | DIPで内向きの依存を実現 |

```
DIPの実践まとめ:

  1. 境界を見つける:
     → ビジネスロジックとインフラの境界
     → テスト可能にしたい境界

  2. 抽象を定義する:
     → インターフェース or 抽象クラス
     → 上位モジュールの視点で設計

  3. 依存を注入する:
     → コンストラクタ注入を基本に
     → IoCコンテナで自動解決

  4. テストで検証:
     → モックを注入して単体テスト
     → 実装の差し替えが容易か確認

  5. 過剰適用を避ける:
     → 必要な場所にのみ適用
     → YAGNI: 今必要でなければ後回し
```

---

## 次に読むべきガイド
→ [[../03-advanced-concepts/00-composition-vs-inheritance.md]] — コンポジション vs 継承

---

## 参考文献
1. Martin, R. "The Dependency Inversion Principle." 1996.
2. Fowler, M. "Inversion of Control Containers and the Dependency Injection pattern." 2004.
3. Martin, R. "Clean Architecture." Prentice Hall, 2017.
4. Vernon, V. "Implementing Domain-Driven Design." Addison-Wesley, 2013.
5. Cockburn, A. "Hexagonal Architecture." alistair.cockburn.us, 2005.
6. Seemann, M. "Dependency Injection: Principles, Practices, and Patterns." Manning, 2019.
