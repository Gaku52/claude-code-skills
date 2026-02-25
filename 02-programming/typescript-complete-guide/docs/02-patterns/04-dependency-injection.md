# TypeScript DI（依存性注入）パターン完全ガイド

> inversify, tsyringe, NestJS を中心に、TypeScript で型安全な DI コンテナとインターフェースベースの疎結合設計を実現する

## この章で学ぶこと

1. **DI の基本原則** -- 依存性逆転の原則（DIP）と制御の逆転（IoC）を TypeScript で実践する方法
2. **DI コンテナ** -- inversify / tsyringe / NestJS を使った自動解決、ライフサイクル管理、スコープ設定
3. **テスタビリティ** -- DI によってテスト時にモックを簡単に差し替え、単体テストを高速化する技法
4. **循環依存の検出と解決** -- 実践的なアプローチとツール
5. **パフォーマンス最適化** -- プロダクション環境での実例とベンチマーク
6. **関数型アプローチ** -- Reader Monad と Effect-ts による DI

---

## 目次

1. [DI の基本原則と SOLID-D](#1-di-の基本原則と-solid-d)
2. [手動 DI（Pure DI）](#2-手動-dipure-di)
3. [InversifyJS による DI](#3-inversifyjs-による-di)
4. [tsyringe による軽量 DI](#4-tsyringe-による軽量-di)
5. [NestJS の DI システム](#5-nestjs-の-di-システム)
6. [DI コンテナなしの DI](#6-di-コンテナなしの-di)
7. [テスト容易性とモック注入](#7-テスト容易性とモック注入)
8. [循環依存の検出と解決](#8-循環依存の検出と解決)
9. [パフォーマンス比較とプロダクション事例](#9-パフォーマンス比較とプロダクション事例)
10. [アンチパターン](#10-アンチパターン)
11. [エッジケース分析](#11-エッジケース分析)
12. [演習問題](#12-演習問題)
13. [FAQ](#13-faq)
14. [参考文献](#14-参考文献)

---

## 1. DI の基本原則と SOLID-D

### 1-1. 依存性逆転の原則（Dependency Inversion Principle）

SOLID 原則の "D" は、高レベルモジュール（ビジネスロジック）は低レベルモジュール（インフラ実装）に依存すべきではなく、両方とも抽象（インターフェース）に依存すべきという原則です。

```
■ DIP 適用前（具象依存）

  +------------+        +------------+
  | UserService|------->| PostgresDB |
  +------------+        +------------+
  高レベル               低レベル
  (ビジネスロジック)      (インフラ)

  UserService が PostgresDB に直接依存
  → DB を変更すると UserService も変更が必要
  → テスト時に実際の DB が必要

■ DIP 適用後（抽象依存）

  +------------+        +-----------+
  | UserService|------->| IDatabase |  ← 抽象（インターフェース）
  +------------+        +-----------+
                             ↑
                     +-------+--------+
                     |                |
               +----------+    +----------+
               |PostgresDB|    | MockDB   |
               +----------+    +----------+
               本番             テスト

  UserService は抽象に依存
  → 実装の差し替えが容易
  → テスト時にモックを注入可能
```

### 1-2. 制御の逆転（Inversion of Control）

従来の設計では、アプリケーションコードが依存オブジェクトを自分で生成（`new`）していました。IoC では、外部のコンテナやフレームワークが依存を生成し、アプリケーションに注入します。

```typescript
// 従来の設計（制御はアプリケーション側）
class UserService {
  private userRepo: IUserRepository;

  constructor() {
    // 自分で依存を生成
    this.userRepo = new PostgresUserRepository();
  }
}

// IoC 適用後（制御はコンテナ側）
class UserService {
  constructor(
    private readonly userRepo: IUserRepository // コンテナが注入
  ) {}
}

// コンテナが UserService の依存を解決して注入
const userService = container.resolve(UserService);
```

### 1-3. なぜ TypeScript で DI が重要か

TypeScript の型システムは、DI パターンと非常に相性が良いです。

```typescript
// TypeScript の型システムが依存を保証
interface IUserRepository {
  findById(id: string): Promise<User | null>;
  save(user: User): Promise<void>;
  delete(id: string): Promise<void>;
}

interface IEmailService {
  send(to: string, subject: string, body: string): Promise<void>;
}

interface ILogger {
  info(message: string, meta?: Record<string, unknown>): void;
  error(message: string, error?: unknown): void;
}

// コンストラクタ注入（最も基本的な DI）
class UserService {
  constructor(
    private readonly userRepo: IUserRepository,
    private readonly emailService: IEmailService,
    private readonly logger: ILogger
  ) {}

  async createUser(data: CreateUserDto): Promise<User> {
    this.logger.info("Creating user", { email: data.email });

    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    await this.userRepo.save(user);
    await this.emailService.send(
      user.email,
      "Welcome!",
      `Hello ${user.name}`
    );

    return user;
  }

  async deleteUser(id: string): Promise<void> {
    const user = await this.userRepo.findById(id);
    if (!user) {
      throw new Error(`User ${id} not found`);
    }

    await this.userRepo.delete(id);
    await this.emailService.send(
      user.email,
      "Account Deleted",
      `Goodbye ${user.name}`
    );

    this.logger.info("User deleted", { id });
  }
}
```

**TypeScript + DI のメリット:**

1. **型安全性**: コンパイル時に依存の型が検証される
2. **リファクタリング支援**: IDE がインターフェース変更を追跡
3. **自動補完**: 依存のメソッドが自動補完される
4. **ドキュメント性**: インターフェースが契約を明確化

---

## 2. 手動 DI（Pure DI）

小規模プロジェクトや、ライブラリ依存を避けたい場合は、手動での依存注入が最もシンプルで型安全です。

### 2-1. コンストラクタ注入

```typescript
// ドメイン層（型定義）
interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

interface CreateUserDto {
  name: string;
  email: string;
}

// 抽象（インターフェース）
interface IUserRepository {
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<User | null>;
  save(user: User): Promise<void>;
  delete(id: string): Promise<void>;
}

interface IEmailService {
  send(to: string, subject: string, body: string): Promise<void>;
}

interface ILogger {
  info(message: string, meta?: Record<string, unknown>): void;
  error(message: string, error?: unknown): void;
  warn(message: string, meta?: Record<string, unknown>): void;
}

// 実装（インフラ層）
class PostgresUserRepository implements IUserRepository {
  constructor(private readonly connectionString: string) {}

  async findById(id: string): Promise<User | null> {
    // 実際の DB クエリ
    console.log(`SELECT * FROM users WHERE id = '${id}'`);
    return null;
  }

  async findByEmail(email: string): Promise<User | null> {
    console.log(`SELECT * FROM users WHERE email = '${email}'`);
    return null;
  }

  async save(user: User): Promise<void> {
    console.log(`INSERT INTO users VALUES (...)`, user);
  }

  async delete(id: string): Promise<void> {
    console.log(`DELETE FROM users WHERE id = '${id}'`);
  }
}

class SmtpEmailService implements IEmailService {
  constructor(private readonly smtpUrl: string) {}

  async send(to: string, subject: string, body: string): Promise<void> {
    console.log(`Sending email to ${to}: ${subject}`);
    // 実際の SMTP 送信
  }
}

class ConsoleLogger implements ILogger {
  info(message: string, meta?: Record<string, unknown>): void {
    console.log(`[INFO] ${message}`, meta);
  }

  error(message: string, error?: unknown): void {
    console.error(`[ERROR] ${message}`, error);
  }

  warn(message: string, meta?: Record<string, unknown>): void {
    console.warn(`[WARN] ${message}`, meta);
  }
}

// サービス層（ビジネスロジック）
class UserService {
  constructor(
    private readonly userRepo: IUserRepository,
    private readonly emailService: IEmailService,
    private readonly logger: ILogger
  ) {}

  async createUser(data: CreateUserDto): Promise<User> {
    this.logger.info("Creating user", { email: data.email });

    // ビジネスルール: 重複チェック
    const existing = await this.userRepo.findByEmail(data.email);
    if (existing) {
      throw new Error(`User with email ${data.email} already exists`);
    }

    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    await this.userRepo.save(user);
    await this.emailService.send(
      user.email,
      "Welcome!",
      `Hello ${user.name}, welcome to our service!`
    );

    this.logger.info("User created successfully", { userId: user.id });
    return user;
  }

  async deleteUser(id: string): Promise<void> {
    const user = await this.userRepo.findById(id);
    if (!user) {
      throw new Error(`User ${id} not found`);
    }

    await this.userRepo.delete(id);
    await this.emailService.send(
      user.email,
      "Account Deleted",
      `Goodbye ${user.name}, your account has been deleted.`
    );

    this.logger.info("User deleted", { id });
  }
}

// 手動でワイヤリング（Composition Root）
function createApp() {
  // 環境変数から設定を読み込み
  const databaseUrl = process.env.DATABASE_URL || "postgres://localhost/mydb";
  const smtpUrl = process.env.SMTP_URL || "smtp://localhost";

  // インフラ層のインスタンス生成
  const logger = new ConsoleLogger();
  const userRepo = new PostgresUserRepository(databaseUrl);
  const emailService = new SmtpEmailService(smtpUrl);

  // サービス層のインスタンス生成（依存を注入）
  const userService = new UserService(userRepo, emailService, logger);

  return { userService };
}

// アプリケーションエントリポイント
const app = createApp();

// 使用例
app.userService.createUser({
  name: "Alice",
  email: "alice@example.com",
});
```

### 2-2. 関数注入（Function Injection）

```typescript
// 関数を依存として注入
type GenerateId = () => string;
type GetCurrentTime = () => Date;

class OrderService {
  constructor(
    private readonly generateId: GenerateId,
    private readonly getCurrentTime: GetCurrentTime,
    private readonly logger: ILogger
  ) {}

  createOrder(userId: string, items: OrderItem[]): Order {
    const order: Order = {
      id: this.generateId(), // 注入された関数を使用
      userId,
      items,
      createdAt: this.getCurrentTime(), // 注入された関数を使用
      status: "pending",
    };

    this.logger.info("Order created", { orderId: order.id });
    return order;
  }
}

// 本番環境
const orderService = new OrderService(
  () => crypto.randomUUID(), // 実際の UUID 生成
  () => new Date(), // 実際の現在時刻
  new ConsoleLogger()
);

// テスト環境
const testOrderService = new OrderService(
  () => "test-id-123", // 固定 ID
  () => new Date("2024-01-01"), // 固定時刻
  new MockLogger()
);
```

### 2-3. ファクトリパターンとの組み合わせ

```typescript
// ファクトリを依存として注入
interface IUserRepositoryFactory {
  create(connectionString: string): IUserRepository;
}

class PostgresUserRepositoryFactory implements IUserRepositoryFactory {
  create(connectionString: string): IUserRepository {
    return new PostgresUserRepository(connectionString);
  }
}

class MultiTenantUserService {
  constructor(
    private readonly repoFactory: IUserRepositoryFactory,
    private readonly emailService: IEmailService,
    private readonly logger: ILogger
  ) {}

  async createUserForTenant(
    tenantId: string,
    data: CreateUserDto
  ): Promise<User> {
    // テナントごとに異なる DB 接続を使用
    const connectionString = `postgres://localhost/${tenantId}`;
    const userRepo = this.repoFactory.create(connectionString);

    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    await userRepo.save(user);
    await this.emailService.send(
      user.email,
      "Welcome!",
      `Hello ${user.name}`
    );

    this.logger.info("User created for tenant", { tenantId, userId: user.id });
    return user;
  }
}
```

---

## 3. InversifyJS による DI

InversifyJS は、TypeScript 向けの強力な DI コンテナで、デコレータベースの宣言的な依存管理を提供します。

### 3-1. 基本セットアップ

```typescript
// inversify は reflect-metadata が必要
import "reflect-metadata";
import { Container, injectable, inject, interfaces } from "inversify";

// シンボルでトークンを定義（文字列の衝突を防止）
const TYPES = {
  UserRepository: Symbol.for("UserRepository"),
  OrderRepository: Symbol.for("OrderRepository"),
  EmailService: Symbol.for("EmailService"),
  Logger: Symbol.for("Logger"),
  Database: Symbol.for("Database"),
  UserService: Symbol.for("UserService"),
  OrderService: Symbol.for("OrderService"),
} as const;

// 実装クラスに @injectable デコレータ
@injectable()
class PostgresDatabase {
  constructor() {
    console.log("PostgresDatabase initialized");
  }

  async query(sql: string): Promise<any[]> {
    console.log(`Executing: ${sql}`);
    return [];
  }

  async close(): Promise<void> {
    console.log("Database connection closed");
  }
}

@injectable()
class PostgresUserRepository implements IUserRepository {
  constructor(
    @inject(TYPES.Database) private readonly db: PostgresDatabase
  ) {}

  async findById(id: string): Promise<User | null> {
    const results = await this.db.query(`SELECT * FROM users WHERE id = '${id}'`);
    return results[0] || null;
  }

  async findByEmail(email: string): Promise<User | null> {
    const results = await this.db.query(`SELECT * FROM users WHERE email = '${email}'`);
    return results[0] || null;
  }

  async save(user: User): Promise<void> {
    await this.db.query(`INSERT INTO users VALUES (...)`);
  }

  async delete(id: string): Promise<void> {
    await this.db.query(`DELETE FROM users WHERE id = '${id}'`);
  }
}

@injectable()
class PostgresOrderRepository {
  constructor(
    @inject(TYPES.Database) private readonly db: PostgresDatabase
  ) {}

  async findById(id: string): Promise<Order | null> {
    const results = await this.db.query(`SELECT * FROM orders WHERE id = '${id}'`);
    return results[0] || null;
  }

  async save(order: Order): Promise<void> {
    await this.db.query(`INSERT INTO orders VALUES (...)`);
  }
}

@injectable()
class SmtpEmailService implements IEmailService {
  async send(to: string, subject: string, body: string): Promise<void> {
    console.log(`Sending email to ${to}: ${subject}`);
  }
}

@injectable()
class ConsoleLogger implements ILogger {
  info(message: string, meta?: Record<string, unknown>): void {
    console.log(`[INFO] ${message}`, meta);
  }

  error(message: string, error?: unknown): void {
    console.error(`[ERROR] ${message}`, error);
  }

  warn(message: string, meta?: Record<string, unknown>): void {
    console.warn(`[WARN] ${message}`, meta);
  }
}

// サービスクラスに @inject で依存を宣言
@injectable()
class UserService {
  constructor(
    @inject(TYPES.UserRepository) private readonly userRepo: IUserRepository,
    @inject(TYPES.EmailService) private readonly emailService: IEmailService,
    @inject(TYPES.Logger) private readonly logger: ILogger
  ) {}

  async createUser(data: CreateUserDto): Promise<User> {
    this.logger.info("Creating user", { email: data.email });

    const existing = await this.userRepo.findByEmail(data.email);
    if (existing) {
      throw new Error(`User with email ${data.email} already exists`);
    }

    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    await this.userRepo.save(user);
    await this.emailService.send(
      user.email,
      "Welcome!",
      `Hello ${user.name}`
    );

    this.logger.info("User created", { userId: user.id });
    return user;
  }
}

@injectable()
class OrderService {
  constructor(
    @inject(TYPES.OrderRepository) private readonly orderRepo: PostgresOrderRepository,
    @inject(TYPES.UserRepository) private readonly userRepo: IUserRepository,
    @inject(TYPES.Logger) private readonly logger: ILogger
  ) {}

  async createOrder(userId: string, items: OrderItem[]): Promise<Order> {
    const user = await this.userRepo.findById(userId);
    if (!user) {
      throw new Error(`User ${userId} not found`);
    }

    const order: Order = {
      id: crypto.randomUUID(),
      userId,
      items,
      createdAt: new Date(),
      status: "pending",
    };

    await this.orderRepo.save(order);
    this.logger.info("Order created", { orderId: order.id });
    return order;
  }
}
```

### 3-2. コンテナ設定とスコープ

```
inversify コンテナの解決フロー:

  container.get(TYPES.UserService)
       |
       v
  +------------------+
  | UserService の    |
  | 依存を解析        |
  +------------------+
       |
  +----+----+--------+
  |         |        |
  v         v        v
 UserRepo EmailSvc Logger
  |         |        |
  v         v        v
Database  (new)   (singleton)
(singleton)
```

```typescript
// コンテナの設定
const container = new Container();

// シングルトンスコープ（アプリ全体で1つのインスタンス）
container
  .bind<PostgresDatabase>(TYPES.Database)
  .to(PostgresDatabase)
  .inSingletonScope();

container
  .bind<ILogger>(TYPES.Logger)
  .to(ConsoleLogger)
  .inSingletonScope();

container
  .bind<IUserRepository>(TYPES.UserRepository)
  .to(PostgresUserRepository)
  .inSingletonScope();

container
  .bind<PostgresOrderRepository>(TYPES.OrderRepository)
  .to(PostgresOrderRepository)
  .inSingletonScope();

// トランジェントスコープ（毎回新しいインスタンス）
container
  .bind<IEmailService>(TYPES.EmailService)
  .to(SmtpEmailService)
  .inTransientScope(); // 毎回新規作成

// リクエストスコープ（リクエスト単位で1つのインスタンス）
container
  .bind<UserService>(TYPES.UserService)
  .to(UserService)
  .inRequestScope(); // リクエストごとに新規作成

container
  .bind<OrderService>(TYPES.OrderService)
  .to(OrderService);

// 解決
const userService = container.get<UserService>(TYPES.UserService);
const orderService = container.get<OrderService>(TYPES.OrderService);

// 全ての依存が自動的に注入される
await userService.createUser({
  name: "Bob",
  email: "bob@example.com",
});
```

### 3-3. モジュール分割

大規模プロジェクトでは、コンテナのバインディングをモジュールに分割します。

```typescript
import { ContainerModule, interfaces } from "inversify";

// インフラ層モジュール
const infrastructureModule = new ContainerModule((bind: interfaces.Bind) => {
  bind<ILogger>(TYPES.Logger)
    .to(ConsoleLogger)
    .inSingletonScope();

  bind<PostgresDatabase>(TYPES.Database)
    .to(PostgresDatabase)
    .inSingletonScope();
});

// リポジトリ層モジュール
const repositoryModule = new ContainerModule((bind: interfaces.Bind) => {
  bind<IUserRepository>(TYPES.UserRepository)
    .to(PostgresUserRepository)
    .inSingletonScope();

  bind<PostgresOrderRepository>(TYPES.OrderRepository)
    .to(PostgresOrderRepository)
    .inSingletonScope();
});

// サービス層モジュール
const serviceModule = new ContainerModule((bind: interfaces.Bind) => {
  bind<IEmailService>(TYPES.EmailService)
    .to(SmtpEmailService)
    .inTransientScope();

  bind<UserService>(TYPES.UserService)
    .to(UserService);

  bind<OrderService>(TYPES.OrderService)
    .to(OrderService);
});

// コンテナにモジュールをロード
const container = new Container();
container.load(
  infrastructureModule,
  repositoryModule,
  serviceModule
);

export { container };
```

### 3-4. 条件付きバインディング

```typescript
// 環境に応じて実装を切り替え
const container = new Container();

if (process.env.NODE_ENV === "production") {
  container
    .bind<ILogger>(TYPES.Logger)
    .to(CloudWatchLogger)
    .inSingletonScope();
} else if (process.env.NODE_ENV === "test") {
  container
    .bind<ILogger>(TYPES.Logger)
    .to(MockLogger)
    .inSingletonScope();
} else {
  container
    .bind<ILogger>(TYPES.Logger)
    .to(ConsoleLogger)
    .inSingletonScope();
}

// 名前付きバインディング
container
  .bind<IUserRepository>(TYPES.UserRepository)
  .to(PostgresUserRepository)
  .whenTargetNamed("postgres");

container
  .bind<IUserRepository>(TYPES.UserRepository)
  .to(MongoUserRepository)
  .whenTargetNamed("mongo");

// 使用時に名前で指定
@injectable()
class MultiDbService {
  constructor(
    @inject(TYPES.UserRepository) @named("postgres")
    private readonly pgRepo: IUserRepository,

    @inject(TYPES.UserRepository) @named("mongo")
    private readonly mongoRepo: IUserRepository
  ) {}
}
```

### 3-5. ファクトリバインディング

```typescript
// ファクトリ関数でインスタンス生成をカスタマイズ
container
  .bind<IUserRepository>(TYPES.UserRepository)
  .toFactory<IUserRepository>((context: interfaces.Context) => {
    return (tenantId: string) => {
      const db = context.container.get<PostgresDatabase>(TYPES.Database);
      return new TenantUserRepository(db, tenantId);
    };
  });

// 使用例
const userRepoFactory = container.get<(tenantId: string) => IUserRepository>(
  TYPES.UserRepository
);
const tenant1Repo = userRepoFactory("tenant-1");
const tenant2Repo = userRepoFactory("tenant-2");
```

---

## 4. tsyringe による軽量 DI

tsyringe は Microsoft が開発した軽量 DI コンテナで、inversify よりもシンプルな API を提供します。

### 4-1. 基本セットアップ

```typescript
import "reflect-metadata";
import { container, injectable, inject, singleton, scoped, Lifecycle } from "tsyringe";

// tsyringe はクラストークンを直接使える
@singleton()
class ConfigService {
  get(key: string): string {
    return process.env[key] ?? "";
  }
}

@singleton()
class DatabaseConnection {
  constructor(private readonly config: ConfigService) {
    const url = this.config.get("DATABASE_URL");
    console.log(`Connecting to database: ${url}`);
  }

  async query(sql: string): Promise<any[]> {
    console.log(`Query: ${sql}`);
    return [];
  }
}

@singleton()
class PostgresUserRepository implements IUserRepository {
  constructor(private readonly db: DatabaseConnection) {}

  async findById(id: string): Promise<User | null> {
    await this.db.query(`SELECT * FROM users WHERE id = '${id}'`);
    return null;
  }

  async findByEmail(email: string): Promise<User | null> {
    await this.db.query(`SELECT * FROM users WHERE email = '${email}'`);
    return null;
  }

  async save(user: User): Promise<void> {
    await this.db.query(`INSERT INTO users VALUES (...)`);
  }

  async delete(id: string): Promise<void> {
    await this.db.query(`DELETE FROM users WHERE id = '${id}'`);
  }
}

// インターフェーストークン（抽象の場合）
const IEmailServiceToken = Symbol("IEmailService");

@injectable()
class SmtpEmailService implements IEmailService {
  constructor(private readonly config: ConfigService) {}

  async send(to: string, subject: string, body: string): Promise<void> {
    const smtpUrl = this.config.get("SMTP_URL");
    console.log(`Sending via ${smtpUrl} to ${to}: ${subject}`);
  }
}

// トークンで登録
container.register<IEmailService>(IEmailServiceToken, {
  useClass: SmtpEmailService,
});

@injectable()
class UserService {
  constructor(
    private readonly userRepo: PostgresUserRepository, // クラス直接指定
    @inject(IEmailServiceToken) private readonly emailService: IEmailService, // トークン指定
    private readonly config: ConfigService
  ) {}

  async createUser(data: CreateUserDto): Promise<User> {
    console.log("Creating user:", data.email);

    const existing = await this.userRepo.findByEmail(data.email);
    if (existing) {
      throw new Error(`User with email ${data.email} already exists`);
    }

    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    await this.userRepo.save(user);
    await this.emailService.send(
      user.email,
      "Welcome!",
      `Hello ${user.name}`
    );

    return user;
  }
}

// 解決
const userService = container.resolve(UserService);
await userService.createUser({
  name: "Charlie",
  email: "charlie@example.com",
});
```

### 4-2. ライフサイクルとスコープ

```typescript
import { Lifecycle, scoped, injectable } from "tsyringe";

// シングルトン（デフォルト）
@singleton()
class AppConfig {
  readonly version = "1.0.0";
}

// トランジェント（毎回新規作成）
@injectable()
class RequestLogger {
  private readonly requestId = crypto.randomUUID();

  log(message: string) {
    console.log(`[${this.requestId}] ${message}`);
  }
}

// スコープ指定でライフサイクル制御
@scoped(Lifecycle.ContainerScoped)
class RequestContext {
  constructor(public readonly requestId: string) {}
}

// 手動登録でライフサイクル指定
container.register("DatabasePool", DatabaseConnection, {
  lifecycle: Lifecycle.Singleton,
});

container.register("RequestHandler", RequestHandler, {
  lifecycle: Lifecycle.Transient,
});
```

### 4-3. ファクトリとカスタムプロバイダ

```typescript
// ファクトリ登録
container.register("DatabaseConnection", {
  useFactory: (c) => {
    const config = c.resolve(ConfigService);
    const url = config.get("DATABASE_URL");
    return new DatabaseConnection(url);
  },
});

// 値の登録
container.register("API_KEY", {
  useValue: process.env.API_KEY,
});

container.register("APP_VERSION", {
  useValue: "1.0.0",
});

// 使用例
@injectable()
class ApiClient {
  constructor(@inject("API_KEY") private readonly apiKey: string) {}

  async fetch(endpoint: string) {
    console.log(`Fetching ${endpoint} with key ${this.apiKey}`);
  }
}
```

### 4-4. 子コンテナ（リクエストスコープ）

```typescript
import { container as rootContainer } from "tsyringe";

// HTTP リクエストハンドラ
async function handleRequest(req: Request, res: Response) {
  // リクエストごとに子コンテナを作成
  const requestContainer = rootContainer.createChildContainer();

  // リクエスト固有の値を登録
  requestContainer.register("RequestId", {
    useValue: crypto.randomUUID(),
  });

  requestContainer.register("UserId", {
    useValue: req.headers["x-user-id"],
  });

  // リクエストコンテナからサービスを解決
  const userService = requestContainer.resolve(UserService);

  // リクエスト処理
  const result = await userService.createUser({
    name: req.body.name,
    email: req.body.email,
  });

  res.json(result);

  // 子コンテナを破棄（リソース解放）
  requestContainer.dispose();
}
```

### 4-5. 遅延注入（Lazy Injection）

```typescript
import { inject, delay, injectable } from "tsyringe";

// 重い初期化処理を持つクラス
@singleton()
class HeavyService {
  constructor() {
    console.log("HeavyService initializing... (expensive)");
    // 重い初期化処理
  }

  process(): string {
    return "processed";
  }
}

@injectable()
class OptimizedService {
  constructor(
    // 遅延注入: 実際に使用するまで HeavyService は初期化されない
    @inject(delay(() => HeavyService))
    private readonly heavyServiceFactory: () => HeavyService
  ) {
    console.log("OptimizedService created (HeavyService not yet initialized)");
  }

  async doSomething() {
    console.log("Doing lightweight work...");

    // 必要になったタイミングで初期化
    const heavyService = this.heavyServiceFactory();
    return heavyService.process();
  }
}

const service = container.resolve(OptimizedService);
// この時点では HeavyService はまだ初期化されていない

await service.doSomething();
// ここで初めて HeavyService が初期化される
```

---

## 5. NestJS の DI システム

NestJS は、Angular ライクな DI システムを提供する TypeScript フレームワークです。

### 5-1. Module / Provider / Inject の基本

```typescript
import { Module, Injectable, Inject } from "@nestjs/common";

// Provider（DI で管理されるクラス）
@Injectable()
class ConfigService {
  get(key: string): string {
    return process.env[key] ?? "";
  }
}

@Injectable()
class DatabaseService {
  constructor(private readonly config: ConfigService) {
    const url = this.config.get("DATABASE_URL");
    console.log(`Database initialized: ${url}`);
  }

  async query(sql: string): Promise<any[]> {
    console.log(`Query: ${sql}`);
    return [];
  }
}

@Injectable()
class UserRepository {
  constructor(private readonly db: DatabaseService) {}

  async findById(id: string): Promise<User | null> {
    await this.db.query(`SELECT * FROM users WHERE id = '${id}'`);
    return null;
  }

  async save(user: User): Promise<void> {
    await this.db.query(`INSERT INTO users VALUES (...)`);
  }
}

@Injectable()
class UserService {
  constructor(private readonly userRepo: UserRepository) {}

  async createUser(data: CreateUserDto): Promise<User> {
    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    await this.userRepo.save(user);
    return user;
  }
}

// Module（Provider をグループ化）
@Module({
  providers: [ConfigService, DatabaseService, UserRepository, UserService],
  exports: [UserService], // 他のモジュールから使えるようにエクスポート
})
class UserModule {}
```

### 5-2. カスタムプロバイダ

```typescript
import { Module, Provider } from "@nestjs/common";

// 値プロバイダ
const configProvider: Provider = {
  provide: "APP_CONFIG",
  useValue: {
    apiKey: process.env.API_KEY,
    appName: "MyApp",
    version: "1.0.0",
  },
};

// ファクトリプロバイダ
const databaseProvider: Provider = {
  provide: "DATABASE_CONNECTION",
  useFactory: (config: ConfigService) => {
    const url = config.get("DATABASE_URL");
    return new DatabaseConnection(url);
  },
  inject: [ConfigService], // ファクトリの依存
};

// クラスプロバイダ（エイリアス）
const loggerProvider: Provider = {
  provide: "ILogger",
  useClass: process.env.NODE_ENV === "production"
    ? CloudWatchLogger
    : ConsoleLogger,
};

// 既存プロバイダ（エイリアス）
const userRepoProvider: Provider = {
  provide: "IUserRepository",
  useExisting: UserRepository,
};

@Module({
  providers: [
    ConfigService,
    configProvider,
    databaseProvider,
    loggerProvider,
    UserRepository,
    userRepoProvider,
  ],
})
class AppModule {}

// 使用例
@Injectable()
class SomeService {
  constructor(
    @Inject("APP_CONFIG") private readonly config: any,
    @Inject("DATABASE_CONNECTION") private readonly db: DatabaseConnection,
    @Inject("ILogger") private readonly logger: ILogger,
    @Inject("IUserRepository") private readonly userRepo: UserRepository
  ) {}
}
```

### 5-3. グローバルモジュールと動的モジュール

```typescript
import { Module, Global, DynamicModule } from "@nestjs/common";

// グローバルモジュール（全モジュールで使える）
@Global()
@Module({
  providers: [ConfigService, LoggerService],
  exports: [ConfigService, LoggerService],
})
class CoreModule {}

// 動的モジュール（設定に応じてプロバイダを変更）
@Module({})
class DatabaseModule {
  static forRoot(options: DatabaseOptions): DynamicModule {
    return {
      module: DatabaseModule,
      providers: [
        {
          provide: "DATABASE_OPTIONS",
          useValue: options,
        },
        {
          provide: DatabaseService,
          useFactory: (opts: DatabaseOptions) => {
            return new DatabaseService(opts);
          },
          inject: ["DATABASE_OPTIONS"],
        },
      ],
      exports: [DatabaseService],
    };
  }

  static forFeature(entities: any[]): DynamicModule {
    const providers = entities.map((entity) => ({
      provide: `${entity.name}Repository`,
      useFactory: (db: DatabaseService) => {
        return new Repository(db, entity);
      },
      inject: [DatabaseService],
    }));

    return {
      module: DatabaseModule,
      providers,
      exports: providers,
    };
  }
}

// 使用例
@Module({
  imports: [
    CoreModule,
    DatabaseModule.forRoot({
      host: "localhost",
      port: 5432,
      database: "mydb",
    }),
    DatabaseModule.forFeature([User, Order]),
  ],
})
class AppModule {}
```

### 5-4. リクエストスコープとインジェクションスコープ

```typescript
import { Injectable, Scope, Inject } from "@nestjs/common";
import { REQUEST } from "@nestjs/core";
import { Request } from "express";

// リクエストスコープ（リクエストごとに新規作成）
@Injectable({ scope: Scope.REQUEST })
class RequestScopedService {
  constructor(
    @Inject(REQUEST) private readonly request: Request
  ) {
    console.log(`RequestScopedService created for ${request.url}`);
  }

  getRequestId(): string {
    return this.request.headers["x-request-id"] as string;
  }
}

// トランジェントスコープ（注入されるたびに新規作成）
@Injectable({ scope: Scope.TRANSIENT })
class TransientService {
  private readonly instanceId = crypto.randomUUID();

  getInstanceId(): string {
    return this.instanceId;
  }
}

// デフォルトスコープ（シングルトン）
@Injectable() // scope: Scope.DEFAULT
class SingletonService {
  private readonly createdAt = new Date();

  getCreatedAt(): Date {
    return this.createdAt;
  }
}
```

---

## 6. DI コンテナなしの DI

DI コンテナを使わずに、関数型プログラミングのパターンで依存注入を実現する方法もあります。

### 6-1. Reader Monad パターン

```typescript
// Reader Monad: 依存を「環境」として伝播
type Reader<Env, A> = (env: Env) => A;

// 依存の定義
interface Dependencies {
  userRepo: IUserRepository;
  emailService: IEmailService;
  logger: ILogger;
}

// Reader Monad を返す関数
function createUser(data: CreateUserDto): Reader<Dependencies, Promise<User>> {
  return async (deps) => {
    deps.logger.info("Creating user", { email: data.email });

    const existing = await deps.userRepo.findByEmail(data.email);
    if (existing) {
      throw new Error(`User with email ${data.email} already exists`);
    }

    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    await deps.userRepo.save(user);
    await deps.emailService.send(
      user.email,
      "Welcome!",
      `Hello ${user.name}`
    );

    deps.logger.info("User created", { userId: user.id });
    return user;
  };
}

function deleteUser(id: string): Reader<Dependencies, Promise<void>> {
  return async (deps) => {
    const user = await deps.userRepo.findById(id);
    if (!user) {
      throw new Error(`User ${id} not found`);
    }

    await deps.userRepo.delete(id);
    await deps.emailService.send(
      user.email,
      "Account Deleted",
      `Goodbye ${user.name}`
    );

    deps.logger.info("User deleted", { id });
  };
}

// 使用例
const deps: Dependencies = {
  userRepo: new PostgresUserRepository("postgres://localhost/mydb"),
  emailService: new SmtpEmailService("smtp://localhost"),
  logger: new ConsoleLogger(),
};

// Reader を実行（依存を注入）
const userReader = createUser({
  name: "David",
  email: "david@example.com",
});
const user = await userReader(deps);

const deleteReader = deleteUser(user.id);
await deleteReader(deps);
```

### 6-2. Effect-ts による DI

Effect-ts は、型安全な副作用管理と DI を提供するライブラリです。

```typescript
import { Effect, Context, Layer } from "effect";

// サービスの定義（Context を使う）
class UserRepository extends Context.Tag("UserRepository")<
  UserRepository,
  {
    findById: (id: string) => Effect.Effect<never, Error, User | null>;
    save: (user: User) => Effect.Effect<never, Error, void>;
  }
>() {}

class EmailService extends Context.Tag("EmailService")<
  EmailService,
  {
    send: (to: string, subject: string, body: string) => Effect.Effect<never, Error, void>;
  }
>() {}

class Logger extends Context.Tag("Logger")<
  Logger,
  {
    info: (message: string, meta?: Record<string, unknown>) => Effect.Effect<never, never, void>;
    error: (message: string, error?: unknown) => Effect.Effect<never, never, void>;
  }
>() {}

// ビジネスロジック（Effect を返す）
function createUser(data: CreateUserDto) {
  return Effect.gen(function* (_) {
    const userRepo = yield* _(UserRepository);
    const emailService = yield* _(EmailService);
    const logger = yield* _(Logger);

    yield* _(logger.info("Creating user", { email: data.email }));

    const existing = yield* _(userRepo.findById(data.email));
    if (existing) {
      return yield* _(Effect.fail(new Error(`User ${data.email} exists`)));
    }

    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    yield* _(userRepo.save(user));
    yield* _(emailService.send(user.email, "Welcome!", `Hello ${user.name}`));
    yield* _(logger.info("User created", { userId: user.id }));

    return user;
  });
}

// 実装の Layer（依存の実装を提供）
const UserRepositoryLive = Layer.succeed(
  UserRepository,
  {
    findById: (id: string) =>
      Effect.sync(() => {
        console.log(`Finding user ${id}`);
        return null;
      }),
    save: (user: User) =>
      Effect.sync(() => {
        console.log(`Saving user ${user.id}`);
      }),
  }
);

const EmailServiceLive = Layer.succeed(
  EmailService,
  {
    send: (to: string, subject: string, body: string) =>
      Effect.sync(() => {
        console.log(`Sending email to ${to}: ${subject}`);
      }),
  }
);

const LoggerLive = Layer.succeed(
  Logger,
  {
    info: (message: string, meta?: Record<string, unknown>) =>
      Effect.sync(() => {
        console.log(`[INFO] ${message}`, meta);
      }),
    error: (message: string, error?: unknown) =>
      Effect.sync(() => {
        console.error(`[ERROR] ${message}`, error);
      }),
  }
);

// Layer を合成
const AppLayer = Layer.mergeAll(
  UserRepositoryLive,
  EmailServiceLive,
  LoggerLive
);

// Effect を実行
const program = createUser({
  name: "Eve",
  email: "eve@example.com",
});

const runnable = Effect.provide(program, AppLayer);
await Effect.runPromise(runnable);
```

### 6-3. 関数合成による DI

```typescript
// 高階関数で依存を注入
type WithDependencies<T> = (deps: Dependencies) => T;

// 依存を持つ関数を合成
function compose<A, B, C>(
  f: (b: B) => C,
  g: (a: A) => B
): (a: A) => C {
  return (a) => f(g(a));
}

// 依存を持つ関数の合成
function composeWithDeps<A, B, C>(
  f: WithDependencies<(b: B) => Promise<C>>,
  g: WithDependencies<(a: A) => Promise<B>>
): WithDependencies<(a: A) => Promise<C>> {
  return (deps) => async (a) => {
    const gWithDeps = g(deps);
    const fWithDeps = f(deps);
    const b = await gWithDeps(a);
    return fWithDeps(b);
  };
}

// 使用例
const validateUser: WithDependencies<(data: CreateUserDto) => Promise<CreateUserDto>> =
  (deps) => async (data) => {
    deps.logger.info("Validating user", { email: data.email });
    const existing = await deps.userRepo.findByEmail(data.email);
    if (existing) {
      throw new Error(`User ${data.email} already exists`);
    }
    return data;
  };

const persistUser: WithDependencies<(data: CreateUserDto) => Promise<User>> =
  (deps) => async (data) => {
    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };
    await deps.userRepo.save(user);
    return user;
  };

const sendWelcomeEmail: WithDependencies<(user: User) => Promise<User>> =
  (deps) => async (user) => {
    await deps.emailService.send(
      user.email,
      "Welcome!",
      `Hello ${user.name}`
    );
    return user;
  };

// 関数を合成
const createUserPipeline = (deps: Dependencies) =>
  compose(
    compose(sendWelcomeEmail(deps), persistUser(deps)),
    validateUser(deps)
  );

// 実行
const deps: Dependencies = {
  userRepo: new PostgresUserRepository("postgres://localhost/mydb"),
  emailService: new SmtpEmailService("smtp://localhost"),
  logger: new ConsoleLogger(),
};

const pipeline = createUserPipeline(deps);
const user = await pipeline({
  name: "Frank",
  email: "frank@example.com",
});
```

---

## 7. テスト容易性とモック注入

DI の最大のメリットは、テスト時に依存を簡単にモックに置き換えられることです。

### 7-1. 手動 DI でのモック注入

```typescript
// モック実装
class MockUserRepository implements IUserRepository {
  private users = new Map<string, User>();

  async findById(id: string): Promise<User | null> {
    return this.users.get(id) || null;
  }

  async findByEmail(email: string): Promise<User | null> {
    for (const user of this.users.values()) {
      if (user.email === email) return user;
    }
    return null;
  }

  async save(user: User): Promise<void> {
    this.users.set(user.id, user);
  }

  async delete(id: string): Promise<void> {
    this.users.delete(id);
  }

  // テスト用ヘルパー
  clear() {
    this.users.clear();
  }

  getAll(): User[] {
    return Array.from(this.users.values());
  }
}

class MockEmailService implements IEmailService {
  sentEmails: Array<{ to: string; subject: string; body: string }> = [];

  async send(to: string, subject: string, body: string): Promise<void> {
    this.sentEmails.push({ to, subject, body });
  }

  // テスト用ヘルパー
  clear() {
    this.sentEmails = [];
  }

  getSentTo(email: string) {
    return this.sentEmails.filter((e) => e.to === email);
  }
}

class MockLogger implements ILogger {
  logs: Array<{ level: string; message: string; meta?: any }> = [];

  info(message: string, meta?: Record<string, unknown>): void {
    this.logs.push({ level: "info", message, meta });
  }

  error(message: string, error?: unknown): void {
    this.logs.push({ level: "error", message, meta: error });
  }

  warn(message: string, meta?: Record<string, unknown>): void {
    this.logs.push({ level: "warn", message, meta });
  }

  clear() {
    this.logs = [];
  }

  hasLog(level: string, message: string): boolean {
    return this.logs.some(
      (log) => log.level === level && log.message.includes(message)
    );
  }
}

// テストコード
import { describe, it, expect, beforeEach } from "vitest";

describe("UserService", () => {
  let userService: UserService;
  let mockUserRepo: MockUserRepository;
  let mockEmailService: MockEmailService;
  let mockLogger: MockLogger;

  beforeEach(() => {
    mockUserRepo = new MockUserRepository();
    mockEmailService = new MockEmailService();
    mockLogger = new MockLogger();

    // モックを注入
    userService = new UserService(
      mockUserRepo,
      mockEmailService,
      mockLogger
    );
  });

  it("should create user and send welcome email", async () => {
    const userData: CreateUserDto = {
      name: "Test User",
      email: "test@example.com",
    };

    const user = await userService.createUser(userData);

    // ユーザーが保存されたか確認
    expect(user.id).toBeDefined();
    expect(user.name).toBe("Test User");
    expect(user.email).toBe("test@example.com");

    const savedUser = await mockUserRepo.findById(user.id);
    expect(savedUser).toEqual(user);

    // メールが送信されたか確認
    expect(mockEmailService.sentEmails).toHaveLength(1);
    expect(mockEmailService.sentEmails[0].to).toBe("test@example.com");
    expect(mockEmailService.sentEmails[0].subject).toBe("Welcome!");

    // ログが記録されたか確認
    expect(mockLogger.hasLog("info", "Creating user")).toBe(true);
    expect(mockLogger.hasLog("info", "User created")).toBe(true);
  });

  it("should throw error if user already exists", async () => {
    // 既存ユーザーをセットアップ
    const existingUser: User = {
      id: "existing-id",
      name: "Existing User",
      email: "existing@example.com",
      createdAt: new Date(),
    };
    await mockUserRepo.save(existingUser);

    const userData: CreateUserDto = {
      name: "New User",
      email: "existing@example.com", // 同じメール
    };

    await expect(userService.createUser(userData)).rejects.toThrow(
      "User with email existing@example.com already exists"
    );

    // メールが送信されていないことを確認
    expect(mockEmailService.sentEmails).toHaveLength(0);
  });

  it("should delete user and send goodbye email", async () => {
    // ユーザーを作成
    const user: User = {
      id: "test-id",
      name: "Test User",
      email: "test@example.com",
      createdAt: new Date(),
    };
    await mockUserRepo.save(user);

    await userService.deleteUser(user.id);

    // ユーザーが削除されたか確認
    const deletedUser = await mockUserRepo.findById(user.id);
    expect(deletedUser).toBeNull();

    // メールが送信されたか確認
    expect(mockEmailService.sentEmails).toHaveLength(1);
    expect(mockEmailService.sentEmails[0].to).toBe("test@example.com");
    expect(mockEmailService.sentEmails[0].subject).toBe("Account Deleted");

    // ログが記録されたか確認
    expect(mockLogger.hasLog("info", "User deleted")).toBe(true);
  });
});
```

### 7-2. inversify でのモック注入

```typescript
import { Container } from "inversify";

// テスト用コンテナ
function createTestContainer(): Container {
  const container = new Container();

  // モックを登録
  container
    .bind<IUserRepository>(TYPES.UserRepository)
    .toConstantValue(new MockUserRepository());

  container
    .bind<IEmailService>(TYPES.EmailService)
    .toConstantValue(new MockEmailService());

  container
    .bind<ILogger>(TYPES.Logger)
    .toConstantValue(new MockLogger());

  container
    .bind<UserService>(TYPES.UserService)
    .to(UserService);

  return container;
}

// テストコード
describe("UserService with inversify", () => {
  let container: Container;
  let userService: UserService;
  let mockUserRepo: MockUserRepository;
  let mockEmailService: MockEmailService;

  beforeEach(() => {
    container = createTestContainer();
    userService = container.get<UserService>(TYPES.UserService);
    mockUserRepo = container.get<IUserRepository>(TYPES.UserRepository) as MockUserRepository;
    mockEmailService = container.get<IEmailService>(TYPES.EmailService) as MockEmailService;
  });

  it("should create user", async () => {
    const user = await userService.createUser({
      name: "Test",
      email: "test@example.com",
    });

    expect(mockUserRepo.getAll()).toHaveLength(1);
    expect(mockEmailService.sentEmails).toHaveLength(1);
  });
});
```

### 7-3. tsyringe でのモック注入

```typescript
import { container } from "tsyringe";

describe("UserService with tsyringe", () => {
  let userService: UserService;
  let mockUserRepo: MockUserRepository;
  let mockEmailService: MockEmailService;

  beforeEach(() => {
    // コンテナをリセット
    container.clearInstances();

    // モックを登録
    mockUserRepo = new MockUserRepository();
    mockEmailService = new MockEmailService();

    container.registerInstance(PostgresUserRepository, mockUserRepo as any);
    container.registerInstance(IEmailServiceToken, mockEmailService);

    userService = container.resolve(UserService);
  });

  afterEach(() => {
    container.reset();
  });

  it("should create user", async () => {
    const user = await userService.createUser({
      name: "Test",
      email: "test@example.com",
    });

    expect(mockUserRepo.getAll()).toHaveLength(1);
    expect(mockEmailService.sentEmails).toHaveLength(1);
  });
});
```

### 7-4. テストダブルの種類

```typescript
// 1. Dummy（ダミー）: 引数を埋めるだけで使われない
class DummyLogger implements ILogger {
  info() {}
  error() {}
  warn() {}
}

// 2. Stub（スタブ）: 固定値を返す
class StubUserRepository implements IUserRepository {
  async findById(id: string): Promise<User | null> {
    return {
      id,
      name: "Stub User",
      email: "stub@example.com",
      createdAt: new Date("2024-01-01"),
    };
  }

  async findByEmail(): Promise<User | null> {
    return null;
  }

  async save(): Promise<void> {}
  async delete(): Promise<void> {}
}

// 3. Spy（スパイ）: 呼び出しを記録
class SpyEmailService implements IEmailService {
  callCount = 0;
  lastCall?: { to: string; subject: string; body: string };

  async send(to: string, subject: string, body: string): Promise<void> {
    this.callCount++;
    this.lastCall = { to, subject, body };
  }
}

// 4. Mock（モック）: 期待値を検証
class MockEmailService implements IEmailService {
  private expectedCalls: Array<{ to: string; subject: string }> = [];
  private actualCalls: Array<{ to: string; subject: string; body: string }> = [];

  expectSend(to: string, subject: string) {
    this.expectedCalls.push({ to, subject });
  }

  async send(to: string, subject: string, body: string): Promise<void> {
    this.actualCalls.push({ to, subject, body });
  }

  verify() {
    expect(this.actualCalls.length).toBe(this.expectedCalls.length);
    for (let i = 0; i < this.expectedCalls.length; i++) {
      expect(this.actualCalls[i].to).toBe(this.expectedCalls[i].to);
      expect(this.actualCalls[i].subject).toBe(this.expectedCalls[i].subject);
    }
  }
}

// 5. Fake（フェイク）: 簡易実装
class FakeUserRepository implements IUserRepository {
  private users = new Map<string, User>();
  private emailIndex = new Map<string, string>();

  async findById(id: string): Promise<User | null> {
    return this.users.get(id) || null;
  }

  async findByEmail(email: string): Promise<User | null> {
    const id = this.emailIndex.get(email);
    return id ? this.users.get(id) || null : null;
  }

  async save(user: User): Promise<void> {
    this.users.set(user.id, user);
    this.emailIndex.set(user.email, user.id);
  }

  async delete(id: string): Promise<void> {
    const user = this.users.get(id);
    if (user) {
      this.emailIndex.delete(user.email);
      this.users.delete(id);
    }
  }
}
```

---

## 8. 循環依存の検出と解決

循環依存は、DI システムで最も厄介な問題の1つです。

### 8-1. 循環依存の例

```typescript
// NG: 循環依存
@injectable()
class UserService {
  constructor(
    @inject(TYPES.OrderService) private orderService: OrderService
  ) {}

  async getUserOrders(userId: string) {
    return this.orderService.getOrdersByUser(userId);
  }
}

@injectable()
class OrderService {
  constructor(
    @inject(TYPES.UserService) private userService: UserService
  ) {}

  async getOrdersByUser(userId: string) {
    const user = await this.userService.getUser(userId);
    // ... 注文を取得
  }
}

// エラー: Circular dependency detected
```

### 8-2. 循環依存の図解

```
循環依存の構造:

  UserService
       |
       | depends on
       v
  OrderService
       |
       | depends on
       v
  UserService  ← 循環！
       |
       v
  (無限ループ)

解決策の種類:

1. インターフェース分離
   UserService → IOrderQuery (read-only)
   OrderService → IUserQuery (read-only)

2. イベント駆動
   UserService → Event Bus ← OrderService

3. 中間サービス導入
   UserService → QueryService ← OrderService
```

### 8-3. 解決策 1: インターフェース分離

```typescript
// 読み取り専用インターフェースを分離
interface IOrderQuery {
  getOrdersByUser(userId: string): Promise<Order[]>;
}

interface IUserQuery {
  getUser(userId: string): Promise<User | null>;
}

// UserService は IOrderQuery のみに依存
@injectable()
class UserService implements IUserQuery {
  constructor(
    @inject(TYPES.UserRepository) private userRepo: IUserRepository,
    @inject(TYPES.OrderQuery) private orderQuery: IOrderQuery
  ) {}

  async getUser(userId: string): Promise<User | null> {
    return this.userRepo.findById(userId);
  }

  async getUserWithOrders(userId: string) {
    const user = await this.getUser(userId);
    if (!user) return null;

    const orders = await this.orderQuery.getOrdersByUser(userId);
    return { ...user, orders };
  }
}

// OrderService は IUserQuery のみに依存
@injectable()
class OrderService implements IOrderQuery {
  constructor(
    @inject(TYPES.OrderRepository) private orderRepo: IOrderRepository,
    @inject(TYPES.UserQuery) private userQuery: IUserQuery
  ) {}

  async getOrdersByUser(userId: string): Promise<Order[]> {
    const user = await this.userQuery.getUser(userId);
    if (!user) throw new Error("User not found");

    return this.orderRepo.findByUserId(userId);
  }
}

// バインディング
container.bind<IUserQuery>(TYPES.UserQuery).to(UserService);
container.bind<IOrderQuery>(TYPES.OrderQuery).to(OrderService);
container.bind<UserService>(TYPES.UserService).to(UserService);
container.bind<OrderService>(TYPES.OrderService).to(OrderService);
```

### 8-4. 解決策 2: イベント駆動アーキテクチャ

```typescript
// イベントバスで疎結合化
interface DomainEvent {
  type: string;
  timestamp: Date;
  data: any;
}

interface IEventBus {
  publish(event: DomainEvent): Promise<void>;
  subscribe(eventType: string, handler: (event: DomainEvent) => Promise<void>): void;
}

@injectable()
class EventBus implements IEventBus {
  private handlers = new Map<string, Array<(event: DomainEvent) => Promise<void>>>();

  async publish(event: DomainEvent): Promise<void> {
    const handlers = this.handlers.get(event.type) || [];
    await Promise.all(handlers.map((handler) => handler(event)));
  }

  subscribe(eventType: string, handler: (event: DomainEvent) => Promise<void>): void {
    const handlers = this.handlers.get(eventType) || [];
    handlers.push(handler);
    this.handlers.set(eventType, handlers);
  }
}

// UserService は EventBus にイベントを発行
@injectable()
class UserService {
  constructor(
    @inject(TYPES.UserRepository) private userRepo: IUserRepository,
    @inject(TYPES.EventBus) private eventBus: IEventBus
  ) {}

  async createUser(data: CreateUserDto): Promise<User> {
    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    await this.userRepo.save(user);

    // イベント発行（OrderService への直接依存なし）
    await this.eventBus.publish({
      type: "UserCreated",
      timestamp: new Date(),
      data: { userId: user.id, email: user.email },
    });

    return user;
  }
}

// OrderService はイベントを購読
@injectable()
class OrderService {
  constructor(
    @inject(TYPES.OrderRepository) private orderRepo: IOrderRepository,
    @inject(TYPES.EventBus) private eventBus: IEventBus
  ) {
    // イベントハンドラを登録
    this.eventBus.subscribe("UserCreated", async (event) => {
      console.log("User created, initializing order history", event.data);
      // 注文履歴を初期化
    });
  }

  async createOrder(userId: string, items: OrderItem[]): Promise<Order> {
    const order: Order = {
      id: crypto.randomUUID(),
      userId,
      items,
      createdAt: new Date(),
      status: "pending",
    };

    await this.orderRepo.save(order);

    // イベント発行
    await this.eventBus.publish({
      type: "OrderCreated",
      timestamp: new Date(),
      data: { orderId: order.id, userId },
    });

    return order;
  }
}
```

### 8-5. 解決策 3: 中間サービス導入

```typescript
// クエリサービスを中間に配置
@injectable()
class QueryService {
  constructor(
    @inject(TYPES.UserRepository) private userRepo: IUserRepository,
    @inject(TYPES.OrderRepository) private orderRepo: IOrderRepository
  ) {}

  async getUserWithOrders(userId: string) {
    const user = await this.userRepo.findById(userId);
    if (!user) return null;

    const orders = await this.orderRepo.findByUserId(userId);
    return { ...user, orders };
  }

  async getOrderWithUser(orderId: string) {
    const order = await this.orderRepo.findById(orderId);
    if (!order) return null;

    const user = await this.userRepo.findById(order.userId);
    return { ...order, user };
  }
}

// UserService は QueryService に依存
@injectable()
class UserService {
  constructor(
    @inject(TYPES.UserRepository) private userRepo: IUserRepository,
    @inject(TYPES.QueryService) private queryService: QueryService
  ) {}

  async getUserWithOrders(userId: string) {
    return this.queryService.getUserWithOrders(userId);
  }
}

// OrderService も QueryService に依存
@injectable()
class OrderService {
  constructor(
    @inject(TYPES.OrderRepository) private orderRepo: IOrderRepository,
    @inject(TYPES.QueryService) private queryService: QueryService
  ) {}

  async getOrderWithUser(orderId: string) {
    return this.queryService.getOrderWithUser(orderId);
  }
}
```

### 8-6. 循環依存の検出ツール

```typescript
// 循環依存検出ユーティリティ
class CircularDependencyDetector {
  private graph = new Map<string, Set<string>>();

  addDependency(from: string, to: string) {
    if (!this.graph.has(from)) {
      this.graph.set(from, new Set());
    }
    this.graph.get(from)!.add(to);
  }

  detectCycles(): string[][] {
    const visited = new Set<string>();
    const recursionStack = new Set<string>();
    const cycles: string[][] = [];

    const dfs = (node: string, path: string[]) => {
      visited.add(node);
      recursionStack.add(node);
      path.push(node);

      const dependencies = this.graph.get(node) || new Set();
      for (const dep of dependencies) {
        if (!visited.has(dep)) {
          dfs(dep, path);
        } else if (recursionStack.has(dep)) {
          // 循環検出
          const cycleStart = path.indexOf(dep);
          cycles.push([...path.slice(cycleStart), dep]);
        }
      }

      recursionStack.delete(node);
      path.pop();
    };

    for (const node of this.graph.keys()) {
      if (!visited.has(node)) {
        dfs(node, []);
      }
    }

    return cycles;
  }
}

// 使用例
const detector = new CircularDependencyDetector();
detector.addDependency("UserService", "OrderService");
detector.addDependency("OrderService", "UserService");

const cycles = detector.detectCycles();
if (cycles.length > 0) {
  console.error("Circular dependencies detected:");
  cycles.forEach((cycle) => {
    console.error("  " + cycle.join(" -> "));
  });
}
```

---

## 9. パフォーマンス比較とプロダクション事例

### 9-1. DI ライブラリのベンチマーク

```typescript
import Benchmark from "benchmark";

// ベンチマーク用のシンプルなサービス
@injectable()
class SimpleService {
  getValue(): string {
    return "value";
  }
}

@injectable()
class DependentService {
  constructor(
    @inject(TYPES.SimpleService) private simpleService: SimpleService
  ) {}

  execute(): string {
    return this.simpleService.getValue();
  }
}

// ベンチマークスイート
const suite = new Benchmark.Suite();

// 1. 手動 DI
suite.add("Manual DI", () => {
  const simple = new SimpleService();
  const dependent = new DependentService(simple);
  dependent.execute();
});

// 2. inversify
const inversifyContainer = new Container();
inversifyContainer.bind(TYPES.SimpleService).to(SimpleService);
inversifyContainer.bind(TYPES.DependentService).to(DependentService);

suite.add("inversify (transient)", () => {
  const service = inversifyContainer.get(TYPES.DependentService);
  service.execute();
});

inversifyContainer
  .bind(TYPES.SimpleService)
  .to(SimpleService)
  .inSingletonScope();

suite.add("inversify (singleton)", () => {
  const service = inversifyContainer.get(TYPES.DependentService);
  service.execute();
});

// 3. tsyringe
container.register(SimpleService, { useClass: SimpleService });
container.register(DependentService, { useClass: DependentService });

suite.add("tsyringe (transient)", () => {
  const service = container.resolve(DependentService);
  service.execute();
});

// 結果を出力
suite
  .on("cycle", (event: any) => {
    console.log(String(event.target));
  })
  .on("complete", function (this: any) {
    console.log("Fastest is " + this.filter("fastest").map("name"));
  })
  .run({ async: true });
```

**典型的なベンチマーク結果:**

```
Manual DI               x 10,000,000 ops/sec ±1.2%
inversify (transient)   x    500,000 ops/sec ±2.1%
inversify (singleton)   x  1,000,000 ops/sec ±1.8%
tsyringe (transient)    x    800,000 ops/sec ±1.5%
tsyringe (singleton)    x  1,500,000 ops/sec ±1.3%

Fastest is Manual DI
```

### 9-2. パフォーマンス最適化のポイント

```typescript
// 1. シングルトンスコープを積極的に使う
container
  .bind<ILogger>(TYPES.Logger)
  .to(ConsoleLogger)
  .inSingletonScope(); // ステートレスなサービスはシングルトン

// 2. 遅延初期化
@injectable()
class HeavyService {
  private data: any;

  // コンストラクタでは重い処理をしない
  constructor() {}

  // 必要になった時に初期化
  async initialize() {
    if (!this.data) {
      this.data = await loadHeavyData();
    }
  }

  async process() {
    await this.initialize();
    return this.data;
  }
}

// 3. ファクトリで条件分岐
container.bind(TYPES.UserRepository).toFactory((context) => {
  return (useCache: boolean) => {
    const db = context.container.get<Database>(TYPES.Database);

    if (useCache) {
      const cache = context.container.get<Cache>(TYPES.Cache);
      return new CachedUserRepository(db, cache);
    }

    return new PostgresUserRepository(db);
  };
});

// 4. プリロード
// アプリ起動時に頻繁に使うサービスをプリロード
async function preloadServices(container: Container) {
  const criticalServices = [
    TYPES.Database,
    TYPES.Logger,
    TYPES.ConfigService,
  ];

  await Promise.all(
    criticalServices.map((token) => container.get(token))
  );
}
```

### 9-3. バンドルサイズの比較

```
DI ライブラリのバンドルサイズ（minified + gzipped）:

inversify         ~15 KB
  + reflect-metadata ~10 KB
  = 合計 ~25 KB

tsyringe          ~5 KB
  + reflect-metadata ~10 KB
  = 合計 ~15 KB

手動 DI           0 KB

typed-inject      ~3 KB
  (reflect-metadata 不要)

Effect-ts         ~50 KB
  (DI + 副作用管理 + その他機能含む)
```

### 9-4. プロダクション事例

#### 事例 1: E コマースプラットフォーム（inversify）

```typescript
// 大規模 E コマースサイトでの inversify 使用例
// サービス数: 100+、依存関係: 300+

// モジュール構成
const modules = [
  // コア
  coreModule,          // Logger, Config, EventBus
  databaseModule,      // DB 接続、トランザクション管理
  cacheModule,         // Redis キャッシュ

  // ドメイン
  userModule,          // ユーザー管理
  productModule,       // 商品管理
  orderModule,         // 注文処理
  paymentModule,       // 決済処理
  inventoryModule,     // 在庫管理
  shippingModule,      // 配送管理

  // インフラ
  emailModule,         // メール送信
  smsModule,           // SMS 送信
  searchModule,        // 全文検索
  analyticsModule,     // 分析
];

const container = new Container();
container.load(...modules);

// パフォーマンス最適化
// - 95% のサービスをシングルトンに
// - リクエストスコープは認証情報のみ
// - ファクトリパターンで動的生成を最小化

// 結果
// - 平均レスポンスタイム: 50ms
// - DI オーバーヘッド: <1ms
// - メモリ使用量: 安定
```

#### 事例 2: SaaS プラットフォーム（NestJS）

```typescript
// マルチテナント SaaS での NestJS 使用例
// テナント数: 1000+、月間リクエスト: 1億+

@Module({
  imports: [
    // グローバルモジュール
    ConfigModule.forRoot({ isGlobal: true }),
    LoggerModule.forRoot({ isGlobal: true }),

    // 機能モジュール
    AuthModule,
    TenantModule,
    UserModule,
    ProjectModule,
    TaskModule,
    NotificationModule,

    // インフラモジュール
    DatabaseModule.forRoot({
      type: "postgres",
      poolSize: 20,
    }),
    CacheModule.forRoot({
      type: "redis",
      ttl: 300,
    }),
  ],
})
class AppModule {}

// テナント分離のためのリクエストスコープ
@Injectable({ scope: Scope.REQUEST })
class TenantContext {
  constructor(@Inject(REQUEST) private request: Request) {}

  getTenantId(): string {
    return this.request.headers["x-tenant-id"] as string;
  }

  getDatabaseConnection(): Connection {
    // テナントごとに DB 接続を切り替え
    const tenantId = this.getTenantId();
    return getConnectionForTenant(tenantId);
  }
}

// パフォーマンス最適化
// - キャッシュレイヤーを積極活用
// - DB 接続プーリング
// - バックグラウンドジョブは別コンテナ

// 結果
// - 99パーセンタイルレスポンス: 100ms
// - テナント分離完全実現
// - スケーラビリティ確保
```

#### 事例 3: マイクロサービス（tsyringe）

```typescript
// マイクロサービスアーキテクチャでの tsyringe 使用例
// サービス数: 20、軽量・高速が要件

// 各マイクロサービスは最小限の依存
@singleton()
class ServiceConfig {
  readonly serviceName = process.env.SERVICE_NAME!;
  readonly port = parseInt(process.env.PORT || "3000");
}

@singleton()
class HealthCheckService {
  constructor(
    private config: ServiceConfig,
    private db: DatabaseService
  ) {}

  async check() {
    return {
      service: this.config.serviceName,
      status: "healthy",
      database: await this.db.ping(),
    };
  }
}

// 軽量・高速な起動
async function bootstrap() {
  const service = container.resolve(HealthCheckService);

  const app = express();
  app.get("/health", async (req, res) => {
    const result = await service.check();
    res.json(result);
  });

  app.listen(service["config"].port);
}

// 結果
// - 起動時間: <100ms
// - メモリフットプリント: <50MB
// - コンテナ化に最適
```

---

## 10. アンチパターン

### AP-1: サービスロケータ（アンチパターン）

```typescript
// NG: グローバルコンテナを直接参照（サービスロケータ）
class UserService {
  getUser(id: string) {
    // コンテナをグローバルに参照 → テスト困難、隠れた依存
    const repo = globalContainer.resolve<IUserRepository>("UserRepo");
    return repo.findById(id);
  }
}

// OK: コンストラクタ注入
class UserService {
  constructor(private readonly userRepo: IUserRepository) {}

  getUser(id: string) {
    return this.userRepo.findById(id);
  }
}

// なぜ NG か:
// 1. テスト時にモックを注入できない
// 2. 依存が隠蔽される（コンストラクタを見ても分からない）
// 3. グローバル状態への依存が生まれる
```

### AP-2: 過剰な抽象化

```typescript
// NG: 実装が1つしかないのにインターフェースを作る
interface IStringUtils {
  capitalize(s: string): string;
  truncate(s: string, len: number): string;
}

@injectable()
class StringUtils implements IStringUtils {
  capitalize(s: string): string {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  truncate(s: string, len: number): string {
    return s.slice(0, len);
  }
}

// ユーティリティ関数に DI は不要（純粋関数）

// OK: 純粋関数として直接使う
export function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function truncate(s: string, len: number): string {
  return s.slice(0, len);
}

// DI すべきもの:
// - 外部 I/O（DB、API、ファイル）
// - 状態を持つもの（キャッシュ、セッション）
// - テストで差し替えたいもの（メール送信、決済）
```

### AP-3: 神クラス（God Class）

```typescript
// NG: 1つのクラスが多すぎる依存を持つ
@injectable()
class GodService {
  constructor(
    private userRepo: IUserRepository,
    private orderRepo: IOrderRepository,
    private productRepo: IProductRepository,
    private paymentService: IPaymentService,
    private emailService: IEmailService,
    private smsService: ISmsService,
    private notificationService: INotificationService,
    private analyticsService: IAnalyticsService,
    private logger: ILogger,
    private cache: ICache,
    private eventBus: IEventBus,
    // ... 依存が多すぎる！
  ) {}

  // あらゆる処理を1つのクラスで実行
}

// OK: 責任を分割
@injectable()
class UserService {
  constructor(
    private userRepo: IUserRepository,
    private emailService: IEmailService,
    private logger: ILogger
  ) {}
}

@injectable()
class OrderService {
  constructor(
    private orderRepo: IOrderRepository,
    private paymentService: IPaymentService,
    private logger: ILogger
  ) {}
}

// 目安: コンストラクタ引数は 3〜5 個まで
// それ以上は責任が大きすぎる可能性
```

### AP-4: new キーワードの乱用

```typescript
// NG: DI を使いながら new で直接生成
@injectable()
class UserService {
  constructor(private userRepo: IUserRepository) {}

  async createUser(data: CreateUserDto) {
    const user = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    await this.userRepo.save(user);

    // new で直接生成 → テスト困難
    const emailService = new SmtpEmailService();
    await emailService.send(user.email, "Welcome!", "Hello");

    return user;
  }
}

// OK: 全ての依存を注入
@injectable()
class UserService {
  constructor(
    private userRepo: IUserRepository,
    private emailService: IEmailService // 注入
  ) {}

  async createUser(data: CreateUserDto) {
    const user = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    await this.userRepo.save(user);
    await this.emailService.send(user.email, "Welcome!", "Hello");

    return user;
  }
}
```

---

## 11. エッジケース分析

### EC-1: オプショナル依存

```typescript
// オプショナル依存の扱い方

// inversify でのオプショナル依存
@injectable()
class UserService {
  constructor(
    @inject(TYPES.UserRepository) private userRepo: IUserRepository,
    @inject(TYPES.Cache) @optional() private cache?: ICache
  ) {}

  async getUser(id: string): Promise<User | null> {
    // キャッシュがあれば使う
    if (this.cache) {
      const cached = await this.cache.get<User>(`user:${id}`);
      if (cached) return cached;
    }

    const user = await this.userRepo.findById(id);

    if (this.cache && user) {
      await this.cache.set(`user:${id}`, user, 300);
    }

    return user;
  }
}

// tsyringe でのオプショナル依存
const CacheToken = Symbol("Cache");

@injectable()
class UserService {
  constructor(
    private userRepo: IUserRepository,
    @inject(CacheToken) @optional() private cache?: ICache
  ) {}
}

// コンテナ設定（キャッシュは環境によって有無が変わる）
if (process.env.REDIS_URL) {
  container.register(CacheToken, { useClass: RedisCache });
}

// 手動 DI でのオプショナル依存
class UserService {
  constructor(
    private userRepo: IUserRepository,
    private cache?: ICache // オプショナルパラメータ
  ) {}
}

const userService = new UserService(
  userRepo,
  process.env.REDIS_URL ? new RedisCache() : undefined
);
```

### EC-2: 動的プロバイダ選択

```typescript
// 実行時に実装を切り替える

// 戦略パターン + DI
interface IStorageStrategy {
  save(key: string, value: any): Promise<void>;
  load(key: string): Promise<any>;
}

@injectable()
class LocalStorageStrategy implements IStorageStrategy {
  async save(key: string, value: any): Promise<void> {
    localStorage.setItem(key, JSON.stringify(value));
  }

  async load(key: string): Promise<any> {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : null;
  }
}

@injectable()
class S3StorageStrategy implements IStorageStrategy {
  async save(key: string, value: any): Promise<void> {
    // S3 にアップロード
    console.log(`Uploading to S3: ${key}`);
  }

  async load(key: string): Promise<any> {
    // S3 からダウンロード
    console.log(`Downloading from S3: ${key}`);
    return null;
  }
}

// ファクトリで動的に選択
@injectable()
class StorageService {
  constructor(
    @inject("StorageFactory")
    private strategyFactory: (type: string) => IStorageStrategy
  ) {}

  async saveWithStrategy(
    type: "local" | "s3",
    key: string,
    value: any
  ): Promise<void> {
    const strategy = this.strategyFactory(type);
    await strategy.save(key, value);
  }
}

// コンテナ設定
container
  .bind<(type: string) => IStorageStrategy>("StorageFactory")
  .toFactory((context) => {
    return (type: string) => {
      switch (type) {
        case "local":
          return context.container.get<IStorageStrategy>(LocalStorageStrategy);
        case "s3":
          return context.container.get<IStorageStrategy>(S3StorageStrategy);
        default:
          throw new Error(`Unknown storage type: ${type}`);
      }
    };
  });
```

### EC-3: 条件付きバインディング

```typescript
// 環境やコンテキストに応じてバインディングを変更

// inversify での条件付きバインディング
container
  .bind<ILogger>(TYPES.Logger)
  .to(ConsoleLogger)
  .when((request: interfaces.Request) => {
    // 開発環境では ConsoleLogger
    return process.env.NODE_ENV === "development";
  });

container
  .bind<ILogger>(TYPES.Logger)
  .to(CloudWatchLogger)
  .when((request: interfaces.Request) => {
    // 本番環境では CloudWatchLogger
    return process.env.NODE_ENV === "production";
  });

// ターゲット名による条件分岐
container
  .bind<IDatabase>(TYPES.Database)
  .to(PostgresDatabase)
  .whenTargetNamed("primary");

container
  .bind<IDatabase>(TYPES.Database)
  .to(MySQLDatabase)
  .whenTargetNamed("secondary");

@injectable()
class ReplicationService {
  constructor(
    @inject(TYPES.Database) @named("primary")
    private primaryDb: IDatabase,

    @inject(TYPES.Database) @named("secondary")
    private secondaryDb: IDatabase
  ) {}

  async write(data: any) {
    await this.primaryDb.query("INSERT ...");
  }

  async read(id: string) {
    // 読み取りはセカンダリから
    return this.secondaryDb.query("SELECT ...");
  }
}

// 親コンテキストによる条件分岐
container
  .bind<IUserRepository>(TYPES.UserRepository)
  .to(PostgresUserRepository)
  .whenInjectedInto(UserService);

container
  .bind<IUserRepository>(TYPES.UserRepository)
  .to(CachedUserRepository)
  .whenInjectedInto(AdminService);
```

---

## 12. 演習問題

### 演習 1: 基礎編 - 手動 DI でブログシステム

**課題:**
ブログシステムの以下のクラスを実装し、手動 DI でワイヤリングしてください。

- `IPostRepository`: 記事の CRUD
- `ICommentRepository`: コメントの CRUD
- `ISearchService`: 全文検索
- `PostService`: 記事の作成・公開・検索
- `CommentService`: コメントの投稿・承認

**要件:**
- インターフェースを定義
- 実装クラスを作成（モック実装で OK）
- `createApp` 関数で依存を解決
- テストでモックを注入

```typescript
// ここに実装

interface IPostRepository {
  // TODO: メソッドを定義
}

interface ICommentRepository {
  // TODO: メソッドを定義
}

interface ISearchService {
  // TODO: メソッドを定義
}

class PostService {
  // TODO: 実装
}

class CommentService {
  // TODO: 実装
}

function createApp() {
  // TODO: ワイヤリング
}
```

### 演習 2: 応用編 - inversify でマルチテナントシステム

**課題:**
マルチテナント SaaS の DI システムを inversify で構築してください。

- テナントごとに異なる DB 接続
- テナント固有の設定
- リクエストスコープでテナントコンテキストを管理
- モジュール分割（コア、テナント、ビジネスロジック）

```typescript
// ここに実装

const TYPES = {
  // TODO: トークンを定義
};

interface TenantContext {
  tenantId: string;
  databaseUrl: string;
}

// TODO: モジュールとバインディングを実装
```

### 演習 3: 発展編 - 循環依存の解決

**課題:**
以下の循環依存を持つシステムを、3 つの異なる手法で解決してください。

1. インターフェース分離
2. イベント駆動
3. 中間サービス導入

```typescript
// 循環依存のあるコード
class ArticleService {
  constructor(private commentService: CommentService) {}

  async getArticleWithComments(articleId: string) {
    const article = await this.getArticle(articleId);
    const comments = await this.commentService.getCommentsByArticle(articleId);
    return { ...article, comments };
  }
}

class CommentService {
  constructor(private articleService: ArticleService) {}

  async getCommentsByArticle(articleId: string) {
    const article = await this.articleService.getArticle(articleId);
    if (!article) throw new Error("Article not found");
    // コメント取得
  }
}

// TODO: 3 つの手法で循環依存を解決
```

---

## 13. FAQ

### Q1: inversify と tsyringe のどちらを選ぶべきですか？

**回答:**
プロジェクトの規模と要件に応じて選択します。

**inversify を選ぶべきケース:**
- 大規模プロジェクト（サービス 50+）
- 複雑な依存関係（循環依存の回避、条件付きバインディング）
- モジュール分割が必要
- 詳細なスコープ制御が必要
- チームが DI に精通している

**tsyringe を選ぶべきケース:**
- 小〜中規模プロジェクト（サービス 10〜30）
- シンプルな依存関係
- バンドルサイズを小さくしたい
- 学習コストを抑えたい
- Microsoft エコシステムを使用（TypeScript, VS Code）

**手動 DI を選ぶべきケース:**
- 小規模プロジェクト（サービス <10）
- 最高の型安全性が必要
- デコレータを避けたい（TC39 Stage 3 対応）
- ゼロ依存が望ましい

### Q2: Next.js や Remix で DI は使えますか?

**回答:**
サーバーサイドでは使えますが、React コンポーネントとは別に管理する必要があります。

**サーバーサイド（API Routes, Server Actions）:**
```typescript
// app/api/users/route.ts
import { container } from "@/lib/di-container";
import { UserService } from "@/services/user-service";

export async function POST(req: Request) {
  const userService = container.resolve(UserService);
  const data = await req.json();
  const user = await userService.createUser(data);
  return Response.json(user);
}
```

**React コンポーネント:**
```typescript
// React コンポーネントでは Context API を使う
"use client";

import { createContext, useContext } from "react";

const ServicesContext = createContext<{
  userService: UserService;
} | null>(null);

export function ServicesProvider({ children }: { children: React.ReactNode }) {
  const services = {
    userService: container.resolve(UserService),
  };

  return (
    <ServicesContext.Provider value={services}>
      {children}
    </ServicesContext.Provider>
  );
}

export function useUserService() {
  const context = useContext(ServicesContext);
  if (!context) throw new Error("ServicesProvider not found");
  return context.userService;
}
```

**推奨アプローチ:**
- サーバーサイド: DI コンテナを使用
- クライアントサイド: Context API + hooks
- ハイブリッド: Server Components でサービスを使い、Client Components には props で渡す

### Q3: DI はどの規模のプロジェクトから導入すべきですか?

**回答:**
以下の基準を目安にしてください。

**DI 不要（手動のコンストラクタ注入で十分）:**
- サービスクラス: 1〜5 個
- 外部依存: 0〜2 個（DB、外部 API など）
- 開発者: 1〜2 人
- 例: 個人プロジェクト、プロトタイプ

**軽量 DI（tsyringe）を検討:**
- サービスクラス: 5〜30 個
- 外部依存: 3〜5 個
- 開発者: 2〜5 人
- 例: スタートアップの MVP、中小規模 SaaS

**フル機能 DI（inversify, NestJS）を検討:**
- サービスクラス: 30+ 個
- 外部依存: 5+ 個
- 開発者: 5+ 人
- 例: エンタープライズアプリ、大規模 SaaS

**判断基準:**
1. **テストの複雑さ**: モックの管理が手動で困難になったら DI 導入
2. **依存関係の複雑さ**: グラフ構造が 3 層以上になったら DI 導入
3. **チームサイズ**: 複数人で開発し、依存管理の統一が必要なら DI 導入

### Q4: DI コンテナなしで DI を実現できますか?

**回答:**
はい、関数型プログラミングのパターンで実現できます。

**Reader Monad パターン:**
```typescript
type Reader<Env, A> = (env: Env) => A;

function createUser(data: CreateUserDto): Reader<Dependencies, Promise<User>> {
  return async (deps) => {
    // deps を使った処理
  };
}

// 依存を注入して実行
const user = await createUser(userData)(dependencies);
```

**メリット:**
- ライブラリ不要
- 完全な型安全性
- 関数合成が容易

**デメリット:**
- 学習コスト（関数型プログラミングの知識が必要）
- ボイラープレート（毎回 `(deps)` を渡す）
- IDE サポートが弱い

**推奨ケース:**
- 関数型プログラミングに精通したチーム
- 最高の型安全性が必要
- デコレータを避けたい

### Q5: 循環依存を完全に防ぐ方法はありますか?

**回答:**
アーキテクチャレベルでの対策が必要です。

**1. レイヤードアーキテクチャ:**
```
上位層は下位層に依存できるが、下位層は上位層に依存できない

Presentation Layer (Controllers)
         ↓
Application Layer (Services)
         ↓
Domain Layer (Entities, Interfaces)
         ↓
Infrastructure Layer (Repositories)
```

**2. 依存関係逆転の原則（DIP）:**
```typescript
// 下位層（Infrastructure）は上位層（Domain）のインターフェースに依存

// Domain Layer
interface IUserRepository {
  findById(id: string): Promise<User | null>;
}

// Infrastructure Layer
class PostgresUserRepository implements IUserRepository {
  // IUserRepository に依存（逆転）
}

// Application Layer
class UserService {
  constructor(private userRepo: IUserRepository) {
    // インターフェースに依存
  }
}
```

**3. イベント駆動アーキテクチャ:**
```typescript
// サービス間の直接依存を避け、イベントで疎結合化

class UserService {
  async createUser(data: CreateUserDto) {
    const user = await this.userRepo.save(data);

    // 他のサービスに直接依存せず、イベントを発行
    await this.eventBus.publish({
      type: "UserCreated",
      data: { userId: user.id },
    });
  }
}

class OrderService {
  constructor(private eventBus: IEventBus) {
    // イベントを購読
    this.eventBus.subscribe("UserCreated", this.onUserCreated);
  }

  private async onUserCreated(event: DomainEvent) {
    // ユーザー作成時の処理
  }
}
```

**4. 静的解析ツール:**
```bash
# dependency-cruiser で循環依存を検出
npx depcruise --validate -- src/

# madge で依存グラフを可視化
npx madge --circular --extensions ts src/
```

### Q6: DI を使うとパフォーマンスは低下しますか?

**回答:**
わずかなオーバーヘッドはありますが、実用上は問題になりません。

**ベンチマーク結果:**
- 手動 DI: 10,000,000 ops/sec（基準）
- tsyringe（シングルトン）: 1,500,000 ops/sec（6.7 倍遅い）
- inversify（シングルトン）: 1,000,000 ops/sec（10 倍遅い）

**実際のアプリケーションでの影響:**
- DI のオーバーヘッド: <1ms
- DB クエリ: 10〜100ms
- 外部 API: 100〜1000ms

→ ボトルネックは I/O であり、DI のオーバーヘッドは誤差範囲

**最適化のポイント:**
1. **シングルトンスコープを使う**: トランジェントより 2〜3 倍高速
2. **頻繁に使うサービスをプリロード**: 起動時に解決してキャッシュ
3. **ファクトリを最小限に**: 動的生成はコスト高
4. **バンドルサイズを意識**: クライアントサイドでは tsyringe や手動 DI

---

## 14. 参考文献

### 公式ドキュメント

1. **InversifyJS**
   https://inversify.io/
   強力で軽量な IoC コンテナ。デコレータベースの DI を提供。

2. **tsyringe**
   https://github.com/microsoft/tsyringe
   Microsoft 製の軽量 DI コンテナ。シンプルな API が特徴。

3. **NestJS - Dependency Injection**
   https://docs.nestjs.com/fundamentals/custom-providers
   Angular ライクな DI システム。エンタープライズ向け。

### 書籍

4. **Clean Architecture** -- Robert C. Martin
   依存性逆転の原則（DIP）と IoC の原典。アーキテクチャレベルでの依存管理を学べる。

5. **Dependency Injection Principles, Practices, and Patterns** -- Steven van Deursen, Mark Seemann
   DI パターンの体系的な解説。.NET 中心だが TypeScript にも応用可能。

6. **Domain-Driven Design** -- Eric Evans
   ドメイン駆動設計における依存管理とレイヤードアーキテクチャ。

### オンライン記事

7. **TypeScript Decorators**
   https://www.typescriptlang.org/docs/handbook/decorators.html
   TypeScript の公式ドキュメント。デコレータの基礎を学べる。

8. **Dependency Injection in TypeScript** -- Alex Jover Morales
   https://www.thisdot.co/blog/dependency-injection-in-typescript
   TypeScript での DI 実装の実践的なガイド。

9. **SOLID Principles in TypeScript**
   https://khalilstemmler.com/articles/solid-principles/solid-typescript/
   SOLID 原則の TypeScript での実装例。

### ツールとライブラリ

10. **typed-inject**
    https://github.com/nicojs/typed-inject
    型安全性を最優先した DI ライブラリ。reflect-metadata 不要。

11. **Effect-ts**
    https://effect.website/
    関数型プログラミングの手法で DI を実現。副作用管理も提供。

12. **dependency-cruiser**
    https://github.com/sverweij/dependency-cruiser
    依存関係の可視化と循環依存の検出ツール。

### 関連ガイド

13. **テスト** -- `../03-tooling/02-testing-typescript.md`
    DI を活用したモックとテスト戦略

14. **ビルダーパターン** -- `./01-builder-pattern.md`
    DI と組み合わせたファクトリ設計

15. **tRPC** -- `../04-ecosystem/02-trpc.md`
    DI で構築したサービス層を tRPC で公開

---

## まとめ

### DI の原則

| 概念 | 要点 |
|------|------|
| DIP（依存性逆転の原則） | 高レベルモジュールは抽象に依存すべき |
| IoC（制御の逆転） | フレームワーク/コンテナが依存を解決する |
| コンストラクタ注入 | 最も推奨される DI の形 |

### DI ライブラリ比較

| 特性 | inversify | tsyringe | 手動DI | NestJS | Effect-ts |
|------|-----------|----------|--------|--------|-----------|
| 学習コスト | 中 | 低 | 最低 | 高 | 高 |
| バンドルサイズ | 25KB | 15KB | 0KB | 50KB+ | 50KB |
| 型安全性 | 中 | 中 | 最高 | 中 | 最高 |
| 自動解決 | あり | あり | なし | あり | あり |
| デコレータ | 必要 | 必要 | 不要 | 必要 | 不要 |
| 適用規模 | 大 | 小〜中 | 小 | 大 | 中〜大 |

### DI スコープ比較

| スコープ | 寿命 | 用途 | メモリ使用量 |
|---------|------|------|-------------|
| Singleton | アプリ全体 | DB接続、Logger、Config | 低 |
| Transient | 毎回新規 | ステートレスService | 中 |
| Request | HTTPリクエスト単位 | リクエスト固有データ、認証情報 | 中 |
| Session | ユーザーセッション | ユーザー固有状態 | 高 |

### ベストプラクティス

1. **コンストラクタ注入を優先**: 依存が明示的で、不変性を保証
2. **インターフェースに依存**: 実装ではなく抽象に依存
3. **シングルトンスコープを活用**: ステートレスなサービスはシングルトンに
4. **循環依存を避ける**: レイヤードアーキテクチャ、イベント駆動で対策
5. **テスト容易性を重視**: モック注入が簡単な設計を心がける
6. **適切な粒度**: 1クラスの依存は 3〜5 個まで
7. **サービスロケータを避ける**: グローバルコンテナへの直接参照は NG

### プロダクション導入チェックリスト

- [ ] プロジェクト規模に応じた DI ライブラリを選択
- [ ] 依存関係グラフを可視化（madge, dependency-cruiser）
- [ ] 循環依存の検出とリファクタリング
- [ ] テストでのモック注入を実装
- [ ] パフォーマンスベンチマーク（本番環境）
- [ ] エラーハンドリング（依存解決失敗時）
- [ ] ドキュメント整備（依存関係図、モジュール構成）
- [ ] CI/CD に依存関係チェックを組み込み

---

DI（依存性注入）は、TypeScript でスケーラブルで保守性の高いアプリケーションを構築するための基盤技術です。プロジェクトの規模と要件に応じて適切なアプローチを選択し、SOLID 原則に基づいた設計を心がけましょう。
