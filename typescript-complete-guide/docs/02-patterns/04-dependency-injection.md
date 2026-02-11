# TypeScript DI（依存性注入）パターン

> inversify, tsyringe を中心に、TypeScript で型安全な DI コンテナとインターフェースベースの疎結合設計を実現する

## この章で学ぶこと

1. **DI の基本原則** -- 依存性逆転の原則（DIP）と制御の逆転（IoC）を TypeScript で実践する方法
2. **DI コンテナ** -- inversify / tsyringe を使った自動解決、ライフサイクル管理、スコープ設定
3. **テスタビリティ** -- DI によってテスト時にモックを簡単に差し替え、単体テストを高速化する技法

---

## 1. DI の基本原則

### 1-1. 依存性逆転の原則

```
■ DIP 適用前（具象依存）

  +------------+        +------------+
  | UserService|------->| PostgresDB |
  +------------+        +------------+
  高レベル               低レベル
  (ビジネスロジック)      (インフラ)

  UserService が PostgresDB に直接依存
  → DB を変更すると UserService も変更が必要

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
```

```typescript
// 抽象（インターフェース）
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
}
```

### 1-2. 手動 DI（Pure DI）

```typescript
// 手動でワイヤリング（小規模プロジェクト向け）
function createApp() {
  // インフラ層
  const logger = new ConsoleLogger();
  const db = new PostgresDatabase(process.env.DATABASE_URL!);
  const mailer = new SmtpEmailService(process.env.SMTP_URL!);

  // リポジトリ層
  const userRepo = new PostgresUserRepository(db);
  const orderRepo = new PostgresOrderRepository(db);

  // サービス層（依存を注入）
  const userService = new UserService(userRepo, mailer, logger);
  const orderService = new OrderService(orderRepo, userRepo, logger);

  // コントローラ層
  const userController = new UserController(userService);
  const orderController = new OrderController(orderService);

  return { userController, orderController };
}
```

---

## 2. inversify

### 2-1. 基本セットアップ

```typescript
// inversify は reflect-metadata が必要
import "reflect-metadata";
import { Container, injectable, inject } from "inversify";

// シンボルでトークンを定義（文字列の衝突を防止）
const TOKENS = {
  UserRepository: Symbol.for("UserRepository"),
  EmailService: Symbol.for("EmailService"),
  Logger: Symbol.for("Logger"),
  UserService: Symbol.for("UserService"),
} as const;

// 実装クラスに @injectable デコレータ
@injectable()
class PostgresUserRepository implements IUserRepository {
  async findById(id: string): Promise<User | null> {
    // DB クエリ
    return null;
  }
  async save(user: User): Promise<void> {
    // DB 保存
  }
  async delete(id: string): Promise<void> {
    // DB 削除
  }
}

@injectable()
class SmtpEmailService implements IEmailService {
  async send(to: string, subject: string, body: string): Promise<void> {
    // SMTP 送信
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
}

// サービスクラスに @inject で依存を宣言
@injectable()
class UserService {
  constructor(
    @inject(TOKENS.UserRepository) private userRepo: IUserRepository,
    @inject(TOKENS.EmailService) private emailService: IEmailService,
    @inject(TOKENS.Logger) private logger: ILogger
  ) {}

  // ... メソッド実装
}
```

### 2-2. コンテナ設定

```
inversify コンテナの解決フロー:

  container.get(TOKENS.UserService)
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
 Token1   Token2   Token3
  |         |        |
  v         v        v
 Postgres  Smtp    Console
 UserRepo  Email   Logger
```

```typescript
// コンテナの設定
const container = new Container();

// バインディング
container
  .bind<IUserRepository>(TOKENS.UserRepository)
  .to(PostgresUserRepository)
  .inSingletonScope(); // シングルトン

container
  .bind<IEmailService>(TOKENS.EmailService)
  .to(SmtpEmailService)
  .inTransientScope(); // 毎回新しいインスタンス

container
  .bind<ILogger>(TOKENS.Logger)
  .to(ConsoleLogger)
  .inSingletonScope();

container
  .bind<UserService>(TOKENS.UserService)
  .to(UserService);

// 解決
const userService = container.get<UserService>(TOKENS.UserService);
// 全ての依存が自動的に注入される
```

### 2-3. モジュール分割

```typescript
// モジュールごとにバインディングを分割
import { ContainerModule } from "inversify";

const infrastructureModule = new ContainerModule((bind) => {
  bind<ILogger>(TOKENS.Logger).to(ConsoleLogger).inSingletonScope();
  bind<IDatabase>(TOKENS.Database).to(PostgresDatabase).inSingletonScope();
});

const repositoryModule = new ContainerModule((bind) => {
  bind<IUserRepository>(TOKENS.UserRepository)
    .to(PostgresUserRepository)
    .inSingletonScope();
  bind<IOrderRepository>(TOKENS.OrderRepository)
    .to(PostgresOrderRepository)
    .inSingletonScope();
});

const serviceModule = new ContainerModule((bind) => {
  bind<UserService>(TOKENS.UserService).to(UserService);
  bind<OrderService>(TOKENS.OrderService).to(OrderService);
});

// コンテナにモジュールをロード
const container = new Container();
container.load(infrastructureModule, repositoryModule, serviceModule);
```

---

## 3. tsyringe

### 3-1. 基本セットアップ

```typescript
import "reflect-metadata";
import { container, injectable, inject, singleton } from "tsyringe";

// tsyringe はクラストークンを直接使える
@singleton()
class ConfigService {
  get(key: string): string {
    return process.env[key] ?? "";
  }
}

@singleton()
class PostgresUserRepository implements IUserRepository {
  constructor(private config: ConfigService) {
    // ConfigService が自動注入される
  }

  async findById(id: string): Promise<User | null> {
    const url = this.config.get("DATABASE_URL");
    // ...
    return null;
  }

  async save(user: User): Promise<void> {}
  async delete(id: string): Promise<void> {}
}

// インターフェーストークン
const IEmailService = Symbol("IEmailService");

@injectable()
class SmtpEmailService implements IEmailService {
  async send(to: string, subject: string, body: string): Promise<void> {}
}

// トークンで登録
container.register<IEmailService>(IEmailService, {
  useClass: SmtpEmailService,
});

@injectable()
class UserService {
  constructor(
    private userRepo: PostgresUserRepository, // クラス直接指定
    @inject(IEmailService) private emailService: IEmailService // トークン指定
  ) {}
}

// 解決
const userService = container.resolve(UserService);
```

### 3-2. ファクトリとスコープ

```typescript
import { container, injectable, registry, Lifecycle } from "tsyringe";

// ライフサイクル指定
@injectable()
@registry([
  {
    token: "IUserRepository",
    useClass: PostgresUserRepository,
    options: { lifecycle: Lifecycle.Singleton },
  },
  {
    token: "IEmailService",
    useClass: SmtpEmailService,
    options: { lifecycle: Lifecycle.Transient },
  },
])
class AppModule {}

// ファクトリ登録
container.register("DatabaseConnection", {
  useFactory: (c) => {
    const config = c.resolve(ConfigService);
    return new DatabaseConnection(config.get("DATABASE_URL"));
  },
});

// 値の登録
container.register("API_KEY", { useValue: process.env.API_KEY });

// 子コンテナ（リクエストスコープ）
function handleRequest(req: Request) {
  const childContainer = container.createChildContainer();
  childContainer.register("RequestId", { useValue: req.id });
  const service = childContainer.resolve(UserService);
  // ...
}
```

---

## 4. デコレータなし DI（TC39 Stage 3 対応）

```
デコレータなし DI のアプローチ:

  +-----------------+
  | Token Map       |  トークン → 型 の対応
  | (型安全)         |
  +-----------------+
          |
          v
  +-----------------+
  | Container       |  bind / resolve
  +-----------------+
          |
          v
  +-----------------+
  | Factory関数     |  依存解決 → インスタンス生成
  +-----------------+
```

```typescript
// reflect-metadata 不要のアプローチ
// トークンの定義
class Token<T> {
  constructor(public readonly name: string) {}
}

// 型安全なコンテナ
class TypedContainer {
  private bindings = new Map<Token<any>, () => any>();

  bind<T>(token: Token<T>, factory: (c: TypedContainer) => T): void {
    this.bindings.set(token, () => factory(this));
  }

  bindSingleton<T>(token: Token<T>, factory: (c: TypedContainer) => T): void {
    let instance: T | undefined;
    this.bindings.set(token, () => {
      if (!instance) instance = factory(this);
      return instance;
    });
  }

  resolve<T>(token: Token<T>): T {
    const factory = this.bindings.get(token);
    if (!factory) {
      throw new Error(`No binding for ${token.name}`);
    }
    return factory() as T;
  }
}

// トークン定義
const Tokens = {
  UserRepo: new Token<IUserRepository>("UserRepo"),
  EmailService: new Token<IEmailService>("EmailService"),
  Logger: new Token<ILogger>("Logger"),
  UserService: new Token<UserService>("UserService"),
};

// コンテナ設定
const container = new TypedContainer();
container.bindSingleton(Tokens.Logger, () => new ConsoleLogger());
container.bindSingleton(
  Tokens.UserRepo,
  () => new PostgresUserRepository()
);
container.bind(
  Tokens.EmailService,
  () => new SmtpEmailService()
);
container.bind(Tokens.UserService, (c) =>
  new UserService(
    c.resolve(Tokens.UserRepo),
    c.resolve(Tokens.EmailService),
    c.resolve(Tokens.Logger)
  )
);

// 解決
const service = container.resolve(Tokens.UserService);
```

---

## 比較表

### DI ライブラリ比較

| 特性 | inversify | tsyringe | 手動DI | typed-inject |
|------|-----------|----------|--------|-------------|
| デコレータ | 必要 | 必要 | 不要 | 不要 |
| reflect-metadata | 必要 | 必要 | 不要 | 不要 |
| バンドルサイズ | ~15KB | ~5KB | 0KB | ~3KB |
| 型安全性 | 中 | 中 | 高 | 最高 |
| 自動解決 | あり | あり | なし | あり |
| スコープ管理 | 充実 | 基本 | 手動 | 基本 |
| 学習コスト | 中 | 低 | 最低 | 中 |

### DI スコープ比較

| スコープ | 寿命 | 用途 | メモリ |
|---------|------|------|--------|
| Singleton | アプリ全体 | DB接続、Logger | 低 |
| Transient | 毎回新規 | ステートレスService | 中 |
| Request | HTTPリクエスト単位 | リクエスト固有データ | 中 |
| Session | ユーザーセッション | ユーザー固有状態 | 高 |

---

## アンチパターン

### AP-1: サービスロケータ（アンチパターン）

```typescript
// NG: グローバルコンテナを直接参照（サービスロケータ）
class UserService {
  getUser(id: string) {
    // コンテナをグローバルに参照 → テスト困難
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
// ユーティリティ関数に DI は不要

// OK: 純粋関数として直接使う
function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// DI すべきもの: 外部 I/O、状態を持つもの、テストで差し替えたいもの
```

---

## FAQ

### Q1: inversify と tsyringe のどちらを選ぶべきですか？

大規模プロジェクトやモジュール分割が必要な場合は inversify が適しています。小〜中規模で手軽に始めたい場合は tsyringe が軽量でシンプルです。どちらもデコレータと reflect-metadata に依存するため、これを避けたい場合は手動 DI か typed-inject を検討してください。

### Q2: Next.js や Remix で DI は使えますか？

サーバーサイド（API Routes, Server Actions）では DI コンテナを使えます。ただし、React コンポーネント自体は DI コンテナと相性が良くないため、React 側は Context API や hooks で依存を提供する方が自然です。サーバー側のサービス層のみ DI を適用するのが一般的です。

### Q3: DI はどの規模のプロジェクトから導入すべきですか？

サービスが 5 つ以上、外部依存（DB、外部 API、メール送信など）が 3 つ以上ある場合に検討を始めるのが良いタイミングです。それ以下の規模ではコンストラクタ注入の手動 DI で十分です。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| DIP | 高レベルモジュールは抽象に依存すべき |
| IoC | フレームワーク/コンテナが依存を解決する |
| コンストラクタ注入 | 最も推奨される DI の形 |
| inversify | フル機能の DI コンテナ、大規模向け |
| tsyringe | 軽量 DI、Microsoft 製 |
| 手動 DI | ライブラリ不要、型安全性最高 |

---

## 次に読むべきガイド

- [テスト](../03-tooling/02-testing-typescript.md) -- DI を活用したモックとテスト戦略
- [ビルダーパターン](./01-builder-pattern.md) -- DI と組み合わせたファクトリ設計
- [tRPC](../04-ecosystem/02-trpc.md) -- DI で構築したサービス層を tRPC で公開

---

## 参考文献

1. **InversifyJS** -- A powerful and lightweight IoC container
   https://inversify.io/

2. **tsyringe** -- Lightweight dependency injection container for TypeScript
   https://github.com/microsoft/tsyringe

3. **Clean Architecture** -- Robert C. Martin
   依存性逆転の原則の原典

4. **Dependency Injection in TypeScript** -- Alex Jover Morales
   https://www.typescriptlang.org/docs/handbook/decorators.html
