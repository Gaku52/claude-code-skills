# DIP (Dependency Inversion Principle)

> DIP states that "high-level modules should not depend on low-level modules. Both should depend on abstractions." Through dependency injection (DI) and IoC containers, achieve testable and loosely coupled systems.

## What You Will Learn in This Chapter

- [ ] Understand what the "inversion" in dependency inversion means
- [ ] Grasp the three patterns of dependency injection
- [ ] Learn the mechanism and practical usage of IoC containers
- [ ] Master architectural design leveraging DIP
- [ ] Acquire criteria for judging when to apply DIP in practice


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the content of [LSP (Liskov Substitution Principle) + ISP (Interface Segregation Principle)](./02-lsp-and-isp.md)

---

## 1. What Is Dependency Inversion

```
Traditional dependency relationship (high-level depends on low-level):

  ┌──────────────┐
  │ OrderService │ ← High-level module (business logic)
  └──────┬───────┘
         │ Depends on (direct reference)
         ▼
  ┌──────────────┐
  │ MySQLDatabase│ ← Low-level module (infrastructure)
  └──────────────┘

  Problem: Changing MySQL to PostgreSQL requires changes to OrderService

After dependency inversion (both depend on abstractions):

  ┌──────────────┐     ┌────────────┐
  │ OrderService │────→│ Database   │ ← Abstraction (interface)
  └──────────────┘     └────────────┘
                             ↑
                       ┌─────┴──────┐
                       │            │
                ┌──────────┐  ┌──────────┐
                │ MySQL    │  │ Postgres │
                └──────────┘  └──────────┘

  Meaning of "inversion":
  → Traditional: high-level → low-level
  → DIP:         high-level → abstraction ← low-level
  → Dependencies are "inverted" so that low-level modules depend on abstractions
```

### 1.1 Historical Background of DIP

```
1996:
  Robert C. Martin published "The Dependency Inversion Principle"
  Organized as the final "D" of the SOLID principles

Why "Inversion":
  Traditional procedural programming:
    High-level module → Low-level module (direct call)
    High-level depends on low-level implementation details

  After applying DIP:
    High-level module → Abstraction (interface)
    Low-level module → Abstraction (interface)
    Dependency is inverted so that low-level modules "conform to the abstraction"

The Two Rules of DIP:
  1. High-level modules should not depend on low-level modules.
     Both should depend on abstractions.
  2. Abstractions should not depend on details.
     Details should depend on abstractions.

"Abstraction" = interface, abstract class
"Detail"      = concrete class, concrete implementation
```

### 1.2 Comparison: With and Without DIP

```typescript
// ❌ DIP violation: high-level directly depends on low-level concrete classes
class OrderService {
  // Directly depends on the concrete MySQLDatabase implementation
  private db = new MySQLDatabase("localhost", 3306, "orders_db");
  // Directly depends on the concrete SendGrid implementation
  private mailer = new SendGridMailer("api-key-xxx");
  // Directly depends on the concrete Stripe implementation
  private payment = new StripePayment("sk_live_xxx");

  async createOrder(dto: CreateOrderDto): Promise<Order> {
    const order = new Order(dto);
    // MySQL-specific method
    await this.db.mysqlQuery("INSERT INTO orders ...", order);
    // SendGrid-specific method
    await this.mailer.sendViaSendGrid(order.email, "注文確認");
    // Stripe-specific method
    await this.payment.stripeCharge(order.total, dto.stripeToken);
    return order;
  }
}

// Problems:
// 1. Changing MySQL to PostgreSQL → rewrite OrderService
// 2. Changing SendGrid to AWS SES → rewrite OrderService
// 3. External APIs are called during tests
// 4. OrderService tests require a DB connection
```

```typescript
// ✅ DIP compliant: both high-level and low-level depend on abstractions
// --- Abstractions (interfaces) ---
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

// --- High-level module: depends only on abstractions ---
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

// --- Low-level modules: implement the abstractions ---
class MySQLOrderRepository implements OrderRepository {
  constructor(private db: MySQLConnection) {}
  async save(order: Order): Promise<void> { /* MySQL-specific implementation */ }
  async findById(id: string): Promise<Order | null> { /* ... */ }
}

class PostgresOrderRepository implements OrderRepository {
  constructor(private db: PgPool) {}
  async save(order: Order): Promise<void> { /* Postgres-specific implementation */ }
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

## 2. Dependency Injection

```
DI = A technique for injecting object dependencies from outside

Three patterns:
  1. Constructor injection (recommended)
  2. Setter injection
  3. Interface injection

Benefits of DI:
  → Mocks can be injected during testing
  → Implementations can be swapped easily
  → Dependencies are explicit (visible from the constructor)
  → Circular dependencies can be detected easily
```

### 2.1 Constructor Injection (Recommended)

```typescript
// === Constructor injection (recommended) ===
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
  // All dependencies received via the constructor
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

// Test: inject mocks
class MockUserRepository implements UserRepository {
  private users: User[] = [];
  async findById(id: string) { return this.users.find(u => u.id === id) ?? null; }
  async save(user: User) { this.users.push(user); }
}

class MockEmailService implements EmailService {
  sent: { to: string; subject: string }[] = [];
  async send(to: string, subject: string) { this.sent.push({ to, subject }); }
}

// Inject mocks for testing
const service = new UserService(
  new MockUserRepository(),
  new MockEmailService(),
  new ConsoleLogger(),
);
```

### 2.2 Setter Injection

```typescript
// === Setter injection ===
class ReportService {
  private formatter?: ReportFormatter;

  // Inject via setter (suitable for optional dependencies)
  setFormatter(formatter: ReportFormatter): void {
    this.formatter = formatter;
  }

  generate(data: ReportData): string {
    const fmt = this.formatter ?? new DefaultFormatter();
    return fmt.format(data);
  }
}

// When to use setter injection:
// - Optional dependencies (when a default implementation exists)
// - When you want to swap dependencies at runtime
// - When required by a framework

// Caveats for setter injection:
// - Use constructor injection for required dependencies
// - Risk of forgetting to call the setter
// - Objects may exist in an incomplete state
```

### 2.3 Interface Injection

```typescript
// === Interface injection ===
// Define interfaces for injection
interface LoggerAware {
  setLogger(logger: Logger): void;
}

interface DatabaseAware {
  setDatabase(db: Database): void;
}

// Implement the injection interfaces
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

// Injector
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

### 2.4 Comparison of DI Patterns

```
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│                  │ Constructor  │ Setter       │ IF injection │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Required deps    │ ○ Optimal   │ △ Risky     │ △ Risky     │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Optional deps    │ △ Possible  │ ○ Optimal   │ △ Possible  │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Immutability     │ ○ readonly  │ ×           │ ×           │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Explicit deps    │ ○ Clear     │ △ Scattered │ △ Scattered │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Testability      │ ○           │ ○           │ ○           │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Circular detect  │ ○ Immediate │ ×           │ ×           │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Recommendation   │ ★★★        │ ★★          │ ★            │
└──────────────────┴──────────────┴──────────────┴──────────────┘

Conclusion: Use constructor injection as the default
            and use setter injection only for optional dependencies
```

---

## 3. IoC Container

```
IoC (Inversion of Control) Container:
  → A framework that automates dependency resolution
  → Registers classes and automatically resolves the dependency graph

  Manual DI:
    const logger = new FileLogger();
    const db = new PostgresDatabase(config);
    const repo = new UserRepository(db);
    const email = new SmtpEmailService(smtpConfig);
    const service = new UserService(repo, email, logger);
    // Deep dependencies become hard to manage manually

  IoC container:
    container.register(Logger, FileLogger);
    container.register(Database, PostgresDatabase);
    container.register(UserRepository);
    container.register(EmailService, SmtpEmailService);
    container.register(UserService);
    const service = container.resolve(UserService);
    // The container automatically resolves the dependency graph

IoC container lifecycle management:
  Singleton:  A single instance for the entire application
  Transient:  A new instance is created on each resolve
  Scoped:     One instance per request (web applications)
```

### 3.1 IoC Container in NestJS

```typescript
// NestJS: a practical IoC container example
import { Injectable, Module, Inject } from "@nestjs/common";

// Register with the container via @Injectable()
@Injectable()
class UserRepository {
  constructor(private readonly db: DatabaseService) {}
  async findById(id: string): Promise<User | null> { /* ... */ }
}

@Injectable()
class UserService {
  constructor(
    private readonly repo: UserRepository, // Automatically injected
    private readonly email: EmailService,  // Automatically injected
  ) {}
}

// Declare dependencies in the module
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
// NestJS: advanced dependency injection patterns

// 1. Custom provider (value injection)
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

// 2. Factory provider
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

// 3. Async provider
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

// 4. Scoped provider
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

### 3.2 Implementing a DI Container in Python

```python
# Python: implementing a simple DI container
from typing import Type, TypeVar, Dict, Any, Callable
import inspect

T = TypeVar("T")


class Container:
    """A simple DI container"""

    def __init__(self):
        self._registry: Dict[type, type] = {}
        self._singletons: Dict[type, Any] = {}
        self._factories: Dict[type, Callable] = {}
        self._scoped: Dict[str, Dict[type, Any]] = {}

    def register(self, interface: type, implementation: type) -> None:
        """Register the mapping between interface and implementation"""
        self._registry[interface] = implementation

    def register_singleton(self, interface: type, implementation: type) -> None:
        """Register as a singleton"""
        self._registry[interface] = implementation
        # Create and cache the instance on first resolve

    def register_factory(self, interface: type, factory: Callable) -> None:
        """Register a factory function"""
        self._factories[interface] = factory

    def register_instance(self, interface: type, instance: Any) -> None:
        """Register an existing instance"""
        self._singletons[interface] = instance

    def resolve(self, interface: Type[T]) -> T:
        """Resolve dependencies and return an instance"""
        # Check singletons
        if interface in self._singletons:
            return self._singletons[interface]

        # Check factories
        if interface in self._factories:
            return self._factoriesinterface

        # Get registered implementation
        impl = self._registry.get(interface, interface)

        # Automatically resolve constructor arguments
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


# Usage
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
        # Postgres-specific implementation

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


# Build the container
container = Container()
container.register(Logger, ConsoleLogger)
container.register(UserRepository, PostgresUserRepository)
container.register(EmailService, SmtpEmailService)

# Dependencies are resolved automatically
user_service = container.resolve(UserService)
# UserService(
#   repo=PostgresUserRepository(logger=ConsoleLogger()),
#   email=SmtpEmailService(logger=ConsoleLogger()),
#   logger=ConsoleLogger()
# )
```

### 3.3 Python: dependency-injector Library

```python
# dependency-injector: a production-ready DI container
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject


class AppContainer(containers.DeclarativeContainer):
    """Application DI container"""

    # Configuration
    config = providers.Configuration()

    # Logger (singleton)
    logger = providers.Singleton(
        FileLogger,
        filepath=config.log_file,
    )

    # Database connection (singleton)
    database = providers.Singleton(
        PostgresDatabase,
        host=config.db.host,
        port=config.db.port,
        name=config.db.name,
    )

    # Repository (factory: creates new instance every time)
    user_repository = providers.Factory(
        PostgresUserRepository,
        db=database,
        logger=logger,
    )

    # Email service (singleton)
    email_service = providers.Singleton(
        SmtpEmailService,
        host=config.smtp.host,
        port=config.smtp.port,
        logger=logger,
    )

    # User service (factory)
    user_service = providers.Factory(
        UserService,
        repo=user_repository,
        email=email_service,
        logger=logger,
    )


# Usage
container = AppContainer()
container.config.from_yaml("config.yml")

# Override for testing
with container.email_service.override(MockEmailService()):
    service = container.user_service()
    # MockEmailService is injected
```

### 3.4 DI in Java Spring

```java
// Spring Framework: IoC container
import org.springframework.stereotype.Service;
import org.springframework.stereotype.Repository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;

// Interface definitions
public interface OrderRepository {
    Order save(Order order);
    Optional<Order> findById(String id);
    List<Order> findByUserId(String userId);
}

public interface NotificationService {
    void notify(String userId, String message);
}

// Implementation: register with container via @Repository annotation
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

// Switching multiple implementations: Profile
@Service
@Profile("production")
public class SlackNotificationService implements NotificationService {
    @Override
    public void notify(String userId, String message) {
        // Call Slack API
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

// Service: constructor injection (recommended)
@Service
public class OrderService {
    private final OrderRepository repository;
    private final NotificationService notification;
    private final Logger logger;

    // Spring does not require @Autowired when there is only one constructor
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

// Test
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
        // Set up mock
        when(mockRepository.save(any())).thenReturn(new Order("order-1"));

        // Execute test
        Order result = orderService.createOrder(new CreateOrderRequest("user-1"));

        // Verify
        assertNotNull(result);
        verify(mockRepository).save(any());
        verify(mockNotification).notify(eq("user-1"), anyString());
    }
}
```

### 3.5 DI in Go (Manual Wiring)

```go
package main

import "context"

// Interface definitions (implemented implicitly in Go)
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

// Service: constructor injection
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
        // Email failure is non-fatal, so do not return an error
    }

    return user, nil
}

// Implementation
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

// Manual wiring (main function)
func main() {
    // Initialize infrastructure
    db, _ := sql.Open("postgres", "postgres://localhost/mydb")
    defer db.Close()

    logger := NewZapLogger()

    // Build the dependency graph manually
    repo := NewPostgresUserRepository(db)
    emailSender := NewSmtpEmailSender("smtp.example.com", 587)
    userService := NewUserService(repo, emailSender, logger)

    // For testing
    // mockRepo := &MockUserRepository{}
    // mockEmail := &MockEmailSender{}
    // testService := NewUserService(mockRepo, mockEmail, logger)
}
```

```go
// Go: automated wiring using Wire
// +build wireinject

package main

import "github.com/google/wire"

// Provider set
var UserSet = wire.NewSet(
    NewPostgresUserRepository,
    wire.Bind(new(UserRepository), new(*PostgresUserRepository)),
    NewSmtpEmailSender,
    wire.Bind(new(EmailSender), new(*SmtpEmailSender)),
    NewUserService,
)

// Wiring function
func InitializeUserService(db *sql.DB, smtpConfig SmtpConfig) *UserService {
    wire.Build(UserSet, NewZapLogger)
    return nil // generated by wire
}
```

---

## 4. Practical Patterns of DIP

```
Pattern 1: Repository pattern
  Domain layer → Repository (interface) ← Infrastructure layer

Pattern 2: Ports & Adapters (Hexagonal)
  Application → Port (interface) ← Adapter (implementation)

Pattern 3: Plugin architecture
  Core → Plugin API ← Individual plugins

Common points:
  → Business logic (high-level) depends only on interfaces (abstractions)
  → Infrastructure (low-level) implements the interfaces
  → Swapping infrastructure does not affect business logic
```

### 4.1 Hexagonal Architecture

```
Hexagonal Architecture:

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
  │    (business logic)     │
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
// Implementation example of hexagonal architecture

// === Ports (interfaces) ===
// Driving ports (input ports): define use cases
interface CreateOrderUseCase {
  execute(command: CreateOrderCommand): Promise<OrderDto>;
}

interface GetOrderUseCase {
  execute(query: GetOrderQuery): Promise<OrderDto>;
}

interface CancelOrderUseCase {
  execute(command: CancelOrderCommand): Promise<void>;
}

// Driven ports (output ports): abstract external dependencies
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

// === Application (domain service) ===
class CreateOrderService implements CreateOrderUseCase {
  constructor(
    private readonly persistence: OrderPersistencePort,
    private readonly payment: PaymentPort,
    private readonly notification: NotificationPort,
  ) {}

  async execute(command: CreateOrderCommand): Promise<OrderDto> {
    // Domain logic
    const order = Order.create(command);
    order.validate();

    // Payment processing (via driven port)
    const paymentResult = await this.payment.processPayment(
      order.id, order.totalAmount
    );
    if (!paymentResult.success) {
      throw new PaymentFailedException(paymentResult.error);
    }
    order.markAsPaid(paymentResult.transactionId);

    // Persistence (via driven port)
    await this.persistence.save(order);

    // Notification (via driven port)
    await this.notification.sendOrderConfirmation(order);

    return OrderDto.fromDomain(order);
  }
}

// === Adapters (concrete implementations) ===
// Driving adapter: REST API
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

// Driven adapter: PostgreSQL
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

// Driven adapter: Stripe
class StripePaymentAdapter implements PaymentPort {
  constructor(private readonly stripe: Stripe) {}

  async processPayment(orderId: string, amount: number): Promise<PaymentResult> {
    try {
      const charge = await this.stripe.charges.create({
        amount: amount * 100, // Stripe uses cents
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

// Driven adapter: AWS SES
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
    // Implementation of cancellation notice
  }
}
```

### 4.2 Clean Architecture and DIP

```
Layered structure of Clean Architecture:

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

Direction of dependency: outer → inner (always inward)

  Frameworks  →  Adapters  →  Use Cases  →  Entities
  (concrete)     (concrete)    (abstract+concrete)  (pure domain)

Role of DIP:
  Inner layers know nothing about outer layers
  Outer layers implement the inner layer's interfaces
  → Dependency direction is always inward
```

```python
# Implementation example of Clean Architecture (Python)

# === Entities (innermost layer) ===
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
    """Order entity (business rules)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    items: list = field(default_factory=list)
    total_amount: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    transaction_id: Optional[str] = None

    def confirm(self, transaction_id: str) -> None:
        """Confirm the order (business rule)"""
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"注文 {self.id} は確認できる状態ではありません")
        if self.total_amount <= 0:
            raise ValueError("合計金額は正の値である必要があります")
        self.status = OrderStatus.CONFIRMED
        self.transaction_id = transaction_id

    def cancel(self) -> None:
        """Cancel the order (business rule)"""
        if self.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED):
            raise ValueError("出荷済み・配達済みの注文はキャンセルできません")
        self.status = OrderStatus.CANCELLED

    def can_be_cancelled(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.CONFIRMED)


# === Use Cases (application business rules) ===
from abc import ABC, abstractmethod


class OrderRepository(ABC):
    """Output port: persistence"""
    @abstractmethod
    async def save(self, order: Order) -> None: ...

    @abstractmethod
    async def find_by_id(self, order_id: str) -> Optional[Order]: ...


class PaymentGateway(ABC):
    """Output port: payment"""
    @abstractmethod
    async def charge(self, amount: float, user_id: str) -> str: ...

    @abstractmethod
    async def refund(self, transaction_id: str) -> None: ...


class NotificationSender(ABC):
    """Output port: notification"""
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
    """Use case: create order"""

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
        # Create entity (apply business rules)
        order = Order(
            user_id=input_data.user_id,
            items=input_data.items,
            total_amount=input_data.total_amount,
        )

        # Payment (via output port)
        transaction_id = await self._payment.charge(
            order.total_amount, order.user_id
        )

        # Business rule: confirm order
        order.confirm(transaction_id)

        # Persistence (via output port)
        await self._order_repo.save(order)

        # Notification (via output port)
        await self._notification.send_order_confirmation(order)

        return CreateOrderOutput(
            order_id=order.id,
            status=order.status.value,
            transaction_id=transaction_id,
        )


# === Interface Adapters (adapter layer) ===
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

## 5. Anti-patterns of DIP

```
1. Service Locator:
   → Directly retrieve from a global container
   → Dependencies are implicit (cannot be seen from the constructor)
   → Difficult to test

   ❌ const repo = ServiceLocator.get(UserRepository);
   ✅ constructor(private repo: UserRepository) {}

2. Over-abstraction:
   → Creating interfaces when there is only one implementation
   → Unnecessary abstraction "in case it changes in the future"
   → Abstract only when change actually becomes necessary

3. Circular dependency:
   → Dependency loop A → B → C → A
   → Reconsider where the interface is introduced

4. Dependency on the container:
   → Injecting the container itself
   → ❌ constructor(private container: Container) {}
   → "Anything can be obtained" = dependencies are unclear

5. Leaky abstraction:
   → Interface depends on a specific implementation
   → ❌ interface Database { mysqlQuery(...): ... }
   → ✅ interface Database { query(...): ... }
```

### 5.1 Details of the Service Locator Anti-pattern

```typescript
// ❌ Service Locator pattern (anti-pattern)
class ServiceLocator {
  private static instances = new Map<string, any>();

  static register(name: string, instance: any): void {
    this.instances.set(name, instance);
  }

  static get<T>(name: string): T {
    return this.instances.get(name) as T;
  }
}

// Consumer side: dependencies are implicit
class OrderService {
  async createOrder(dto: CreateOrderDto): Promise<Order> {
    // Dependencies are declared nowhere
    const repo = ServiceLocator.get<OrderRepository>("orderRepo");
    const email = ServiceLocator.get<EmailService>("emailService");
    const logger = ServiceLocator.get<Logger>("logger");

    // Problems:
    // 1. Cannot tell dependencies by looking at the constructor
    // 2. ServiceLocator must be set up before testing
    // 3. Missing dependencies are discovered at runtime, not compile time
    // 4. Difficult to mock because they are static methods
  }
}

// ✅ Constructor injection (recommended)
class OrderService {
  constructor(
    private readonly repo: OrderRepository,     // Explicit dependency
    private readonly email: EmailService,        // Explicit dependency
    private readonly logger: Logger,             // Explicit dependency
  ) {}

  // Benefits:
  // 1. Dependencies are immediately obvious from the constructor
  // 2. Mocks can be passed directly during testing
  // 3. Missing dependencies are detected at compile time
  // 4. Circular dependencies are also detected at compile time
}
```

### 5.2 Avoiding Over-abstraction

```typescript
// ❌ Over-abstraction: only one implementation
interface IUserValidator {
  validate(user: User): ValidationResult;
}

class UserValidator implements IUserValidator {
  validate(user: User): ValidationResult {
    // The only implementation
  }
}

// Many interfaces with the "I" prefix...
// interface IUserService → class UserService
// interface IOrderService → class OrderService
// interface IProductService → class ProductService

// ✅ When interfaces are beneficial:
// 1. Boundaries with external services
interface PaymentGateway {
  charge(amount: number): Promise<Result>;
}
// → Two implementations exist: StripeGateway, MockGateway

// 2. Mocking needed for testing
interface Clock {
  now(): Date;
}
// → Two implementations: SystemClock, FakeClock

// 3. Different persistence strategies
interface CacheStore {
  get(key: string): Promise<string | null>;
  set(key: string, value: string, ttl: number): Promise<void>;
}
// → RedisStore, MemoryStore, DynamoDBStore

// ✅ When interfaces are not needed:
// Pure computation logic, utility functions
class TaxCalculator {
  calculate(amount: number, rate: number): number {
    return Math.floor(amount * rate);
  }
}
// → No interface needed. Using it directly is fine
```

### 5.3 Resolving Circular Dependencies

```typescript
// ❌ Circular dependency: A → B → C → A
class UserService {
  constructor(private orderService: OrderService) {} // UserService → OrderService
}

class OrderService {
  constructor(private paymentService: PaymentService) {} // OrderService → PaymentService
}

class PaymentService {
  constructor(private userService: UserService) {} // PaymentService → UserService ← cycle!
}

// ✅ Solution 1: introduce an interface
interface UserLookup {
  findById(id: string): Promise<User | null>;
}

class UserService implements UserLookup {
  constructor(private orderService: OrderService) {}
  async findById(id: string): Promise<User | null> { /* ... */ }
}

class PaymentService {
  // Depend on an interface rather than a concrete class
  constructor(private userLookup: UserLookup) {}
}

// ✅ Solution 2: loose coupling through events
interface EventBus {
  publish(event: DomainEvent): void;
  subscribe(eventType: string, handler: EventHandler): void;
}

class UserService {
  constructor(private eventBus: EventBus) {}

  async updateUser(userId: string, data: UpdateUserDto): Promise<void> {
    // Publish an event instead of calling OrderService directly
    this.eventBus.publish(new UserUpdatedEvent(userId, data));
  }
}

class OrderService {
  constructor(private eventBus: EventBus) {
    // Subscribe to the event
    this.eventBus.subscribe("UserUpdated", this.handleUserUpdated.bind(this));
  }

  private async handleUserUpdated(event: UserUpdatedEvent): Promise<void> {
    // Handle orders affected by the user update
  }
}

// ✅ Solution 3: mediator pattern
interface Mediator {
  send<T>(request: Request<T>): Promise<T>;
}

class GetUserQuery implements Request<User> {
  constructor(public readonly userId: string) {}
}

class PaymentService {
  constructor(private mediator: Mediator) {}

  async processPayment(userId: string, amount: number): Promise<void> {
    // Go through the mediator instead of calling UserService directly
    const user = await this.mediator.send(new GetUserQuery(userId));
    // Payment processing...
  }
}
```

---

## 6. Testing and DIP

```typescript
// Example of how DIP dramatically improves testability

// === Target under test ===
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

// === Test: inject mocks ===
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

  it("sends only email when only email notification is enabled", async () => {
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

  it("sends to all channels when all are enabled", async () => {
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

  it("propagates the error when email sending fails", async () => {
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

## 7. Criteria for Applying DIP

```
Situations where DIP should be applied:
  ✓ Boundaries with external services (DB, API, email, file system)
  ✓ Dependencies that need to be mocked in tests
  ✓ Parts where implementations may change in the future
  ✓ Multiple implementations exist (production, test, development)
  ✓ Separation of business logic and infrastructure is important

Situations where DIP is not necessary:
  ✗ Pure computation logic (Math, string operations)
  ✗ Value objects (Money, Date, Address)
  ✗ Utility functions
  ✗ When the project is very small
  ✗ Prototype stage

Staged approach to adoption:
  1. Start with direct dependencies (YAGNI)
  2. Extract an interface when testing becomes painful
  3. Introduce DI when implementation swaps become necessary
  4. Consider an IoC container when the dependency graph becomes complex

Signs of over-application:
  → Many interfaces with only one implementation
  → Reading code requires jumping between interfaces and implementations
  → More than three files must be modified for each new feature
  → Interfaces created "just in case"
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Accumulating practical experience is most important. Understanding deepens not just from theory but from actually writing code and verifying its behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the basics and jumping to applications. We recommend firmly grasping the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

The knowledge of this topic is frequently leveraged in daily development work. In particular, it becomes important during code reviews and architectural design.

---

## Summary

| Concept | Key Point |
|------|---------|
| DIP | Both high-level and low-level depend on abstractions |
| DI | Inject dependencies from outside |
| Constructor injection | The most recommended method |
| IoC container | Automates dependency resolution |
| Hexagonal | Representative architecture of DIP |
| Clean Architecture | Achieves inward dependencies via DIP |

```
Summary of DIP in practice:

  1. Find the boundary:
     → Boundary between business logic and infrastructure
     → Boundary you want to make testable

  2. Define the abstraction:
     → Interface or abstract class
     → Design from the viewpoint of the high-level module

  3. Inject the dependency:
     → Constructor injection as the default
     → Automate resolution with an IoC container

  4. Verify with tests:
     → Inject mocks for unit tests
     → Confirm implementation swaps are easy

  5. Avoid over-application:
     → Apply only where needed
     → YAGNI: if not needed now, defer it
```

---

## Recommended Next Guides

---

## References
1. Martin, R. "The Dependency Inversion Principle." 1996.
2. Fowler, M. "Inversion of Control Containers and the Dependency Injection pattern." 2004.
3. Martin, R. "Clean Architecture." Prentice Hall, 2017.
4. Vernon, V. "Implementing Domain-Driven Design." Addison-Wesley, 2013.
5. Cockburn, A. "Hexagonal Architecture." alistair.cockburn.us, 2005.
6. Seemann, M. "Dependency Injection: Principles, Practices, and Patterns." Manning, 2019.
