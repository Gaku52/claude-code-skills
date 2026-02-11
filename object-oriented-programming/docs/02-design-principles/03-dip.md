# DIP（依存性逆転の原則）

> DIPは「上位モジュールは下位モジュールに依存すべきでない。どちらも抽象に依存すべき」。依存性注入（DI）と IoC コンテナを通じて、テスト容易で疎結合なシステムを実現する。

## この章で学ぶこと

- [ ] 依存性逆転の「逆転」が何を意味するか理解する
- [ ] 依存性注入の3つのパターンを把握する
- [ ] IoC コンテナの仕組みと実践的な使い方を学ぶ

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

---

## 2. 依存性注入（Dependency Injection）

```
DI = オブジェクトの依存関係を外部から注入する技術

3つのパターン:
  1. コンストラクタ注入（推奨）
  2. セッター注入
  3. インターフェース注入
```

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
```

```typescript
// NestJS: IoC コンテナの実践例
import { Injectable, Module } from "@nestjs/common";

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

```python
# Python: シンプルなDIコンテナの実装
from typing import Type, TypeVar, Dict, Any

T = TypeVar("T")

class Container:
    def __init__(self):
        self._registry: Dict[type, type] = {}
        self._singletons: Dict[type, Any] = {}

    def register(self, interface: type, implementation: type) -> None:
        self._registry[interface] = implementation

    def resolve(self, interface: Type[T]) -> T:
        if interface in self._singletons:
            return self._singletons[interface]

        impl = self._registry.get(interface, interface)
        # コンストラクタの引数を自動解決
        import inspect
        params = inspect.signature(impl.__init__).parameters
        deps = {}
        for name, param in params.items():
            if name == "self":
                continue
            if param.annotation in self._registry:
                deps[name] = self.resolve(param.annotation)

        instance = impl(**deps)
        return instance

# 使い方
container = Container()
container.register(UserRepository, PostgresUserRepository)
container.register(EmailService, SmtpEmailService)

service = container.resolve(UserService)  # 依存が自動解決
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

---

## 次に読むべきガイド
→ [[../03-advanced-concepts/00-composition-vs-inheritance.md]] — コンポジション vs 継承

---

## 参考文献
1. Martin, R. "The Dependency Inversion Principle." 1996.
2. Fowler, M. "Inversion of Control Containers and the Dependency Injection pattern." 2004.
