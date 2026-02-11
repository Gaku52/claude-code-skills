# Repository パターン — データアクセス抽象化

> Repository パターンでデータアクセスロジックをビジネスロジックから完全に分離し、テスト容易性・保守性・データソースの切り替え可能性を実現するための実践ガイド。

---

## この章で学ぶこと

1. **Repository パターン** の目的と構造、ドメイン駆動設計（DDD）での位置付け
2. **インターフェースと実装の分離** — テスト可能な設計パターン
3. **実装パターン** — Generic Repository、Unit of Work との組み合わせ

---

## 1. Repository パターンの全体像

### 1.1 なぜ Repository が必要か

```
┌──────────────────────────────────────────────────────┐
│  Repository なし（NG）                                │
│                                                      │
│  ┌────────────────────────────────────┐              │
│  │ UserService                        │              │
│  │                                    │              │
│  │ async createUser(data) {           │              │
│  │   // SQL が直接埋め込まれている     │              │
│  │   await db.query(                  │              │
│  │     "INSERT INTO users ..."        │              │
│  │   );                               │              │
│  │   // ビジネスロジックと DB が混在   │              │
│  │   if (user.role === "admin") {     │              │
│  │     await db.query(                │              │
│  │       "INSERT INTO audit_log ..."  │              │
│  │     );                             │              │
│  │   }                                │              │
│  │ }                                  │              │
│  └────────────────────────────────────┘              │
│  問題: テスト困難、DB 変更時に全箇所修正              │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  Repository あり（OK）                                │
│                                                      │
│  ┌──────────────┐     ┌──────────────────┐           │
│  │ UserService  │ ──→ │ UserRepository   │ (Interface)│
│  │ (ビジネス    │     │  findById()      │           │
│  │  ロジック)   │     │  save()          │           │
│  └──────────────┘     │  findByEmail()   │           │
│                       └────────┬─────────┘           │
│                                │                     │
│                       ┌────────▼─────────┐           │
│                       │ PostgresUser     │           │
│                       │ Repository       │ (実装)    │
│                       │ (SQL はここだけ)  │           │
│                       └──────────────────┘           │
│  利点: テスト容易、DB 変更は実装クラスのみ            │
└──────────────────────────────────────────────────────┘
```

### 1.2 レイヤーアーキテクチャでの位置付け

```
┌─────────────────────────────────────────────────┐
│  Presentation Layer (Controller / Handler)       │
│  → HTTP リクエスト/レスポンスの処理              │
├─────────────────────────────────────────────────┤
│  Application Layer (Service / UseCase)           │
│  → ユースケースの調整、トランザクション管理      │
├─────────────────────────────────────────────────┤
│  Domain Layer (Entity / Value Object)            │
│  → ビジネスルール、ドメインロジック              │
│  ┌─────────────────────────────────┐            │
│  │ Repository Interface            │ ← 定義     │
│  │ (ドメイン層に属する)            │            │
│  └─────────────────────────────────┘            │
├─────────────────────────────────────────────────┤
│  Infrastructure Layer                            │
│  ┌─────────────────────────────────┐            │
│  │ Repository Implementation       │ ← 実装     │
│  │ (PostgresUserRepository 等)     │            │
│  └─────────────────────────────────┘            │
│  → DB アクセス、外部 API 呼び出し               │
└─────────────────────────────────────────────────┘
```

---

## 2. TypeScript での実装

### 2.1 インターフェースと実装の分離

```typescript
// === Domain Layer: Repository Interface ===
// domain/repositories/UserRepository.ts

export interface UserRepository {
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<User | null>;
  findAll(options?: FindOptions): Promise<PaginatedResult<User>>;
  save(user: User): Promise<User>;
  delete(id: string): Promise<void>;
}

export interface FindOptions {
  page?: number;
  perPage?: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

export interface PaginatedResult<T> {
  data: T[];
  total: number;
  page: number;
  perPage: number;
  totalPages: number;
}
```

```typescript
// === Infrastructure Layer: PostgreSQL 実装 ===
// infrastructure/repositories/PostgresUserRepository.ts

import { PrismaClient } from "@prisma/client";
import { UserRepository, FindOptions, PaginatedResult } from "../../domain/repositories/UserRepository";
import { User } from "../../domain/entities/User";

export class PostgresUserRepository implements UserRepository {
  constructor(private prisma: PrismaClient) {}

  async findById(id: string): Promise<User | null> {
    const row = await this.prisma.user.findUnique({ where: { id } });
    return row ? this.toDomain(row) : null;
  }

  async findByEmail(email: string): Promise<User | null> {
    const row = await this.prisma.user.findUnique({ where: { email } });
    return row ? this.toDomain(row) : null;
  }

  async findAll(options?: FindOptions): Promise<PaginatedResult<User>> {
    const page = options?.page ?? 1;
    const perPage = options?.perPage ?? 20;
    const skip = (page - 1) * perPage;

    const [rows, total] = await Promise.all([
      this.prisma.user.findMany({
        skip,
        take: perPage,
        orderBy: { [options?.sortBy ?? "createdAt"]: options?.sortOrder ?? "desc" },
      }),
      this.prisma.user.count(),
    ]);

    return {
      data: rows.map(this.toDomain),
      total,
      page,
      perPage,
      totalPages: Math.ceil(total / perPage),
    };
  }

  async save(user: User): Promise<User> {
    const row = await this.prisma.user.upsert({
      where: { id: user.id },
      create: { id: user.id, name: user.name, email: user.email },
      update: { name: user.name, email: user.email },
    });
    return this.toDomain(row);
  }

  async delete(id: string): Promise<void> {
    await this.prisma.user.delete({ where: { id } });
  }

  private toDomain(row: any): User {
    return new User(row.id, row.name, row.email, row.createdAt);
  }
}
```

### 2.2 テスト用のインメモリ実装

```typescript
// === Test: In-Memory 実装 ===
// tests/repositories/InMemoryUserRepository.ts

export class InMemoryUserRepository implements UserRepository {
  private users: Map<string, User> = new Map();

  async findById(id: string): Promise<User | null> {
    return this.users.get(id) ?? null;
  }

  async findByEmail(email: string): Promise<User | null> {
    return [...this.users.values()].find((u) => u.email === email) ?? null;
  }

  async findAll(options?: FindOptions): Promise<PaginatedResult<User>> {
    const all = [...this.users.values()];
    const page = options?.page ?? 1;
    const perPage = options?.perPage ?? 20;
    const start = (page - 1) * perPage;
    const data = all.slice(start, start + perPage);

    return {
      data,
      total: all.length,
      page,
      perPage,
      totalPages: Math.ceil(all.length / perPage),
    };
  }

  async save(user: User): Promise<User> {
    this.users.set(user.id, user);
    return user;
  }

  async delete(id: string): Promise<void> {
    this.users.delete(id);
  }

  // テストヘルパー
  clear(): void {
    this.users.clear();
  }
}
```

### 2.3 Service 層での使用と DI

```typescript
// === Application Layer: Service ===
export class UserService {
  constructor(
    private userRepo: UserRepository,   // インターフェースに依存
    private emailService: EmailService,
  ) {}

  async registerUser(name: string, email: string): Promise<User> {
    // ビジネスルールの検証
    const existing = await this.userRepo.findByEmail(email);
    if (existing) {
      throw new DuplicateEmailError(email);
    }

    // ユーザー作成
    const user = User.create(name, email);
    const saved = await this.userRepo.save(user);

    // 副作用
    await this.emailService.sendWelcome(saved.email);

    return saved;
  }
}

// === テスト ===
describe("UserService", () => {
  let service: UserService;
  let userRepo: InMemoryUserRepository;
  let emailService: jest.Mocked<EmailService>;

  beforeEach(() => {
    userRepo = new InMemoryUserRepository();
    emailService = { sendWelcome: jest.fn() } as any;
    service = new UserService(userRepo, emailService);
  });

  test("新規ユーザーを登録できる", async () => {
    const user = await service.registerUser("Alice", "alice@example.com");

    expect(user.name).toBe("Alice");
    expect(await userRepo.findByEmail("alice@example.com")).not.toBeNull();
    expect(emailService.sendWelcome).toHaveBeenCalledWith("alice@example.com");
  });

  test("重複メールアドレスはエラー", async () => {
    await service.registerUser("Alice", "alice@example.com");

    await expect(
      service.registerUser("Bob", "alice@example.com")
    ).rejects.toThrow(DuplicateEmailError);
  });
});
```

---

## 3. Python での実装

### 3.1 Abstract Base Class による定義

```python
# domain/repositories/user_repository.py
from abc import ABC, abstractmethod
from typing import Optional
from domain.entities.user import User

class UserRepository(ABC):
    @abstractmethod
    async def find_by_id(self, user_id: str) -> Optional[User]:
        ...

    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[User]:
        ...

    @abstractmethod
    async def save(self, user: User) -> User:
        ...

    @abstractmethod
    async def delete(self, user_id: str) -> None:
        ...
```

```python
# infrastructure/repositories/sqlalchemy_user_repository.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from domain.repositories.user_repository import UserRepository
from domain.entities.user import User
from infrastructure.models import UserModel

class SQLAlchemyUserRepository(UserRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def find_by_id(self, user_id: str) -> Optional[User]:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return self._to_domain(row) if row else None

    async def save(self, user: User) -> User:
        model = UserModel(
            id=user.id,
            name=user.name,
            email=user.email,
        )
        merged = await self._session.merge(model)
        await self._session.flush()
        return self._to_domain(merged)

    async def delete(self, user_id: str) -> None:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one()
        await self._session.delete(model)

    def _to_domain(self, model: UserModel) -> User:
        return User(id=model.id, name=model.name, email=model.email)
```

---

## 4. Unit of Work パターンとの組み合わせ

### 4.1 Unit of Work の構造

```
┌──────────────────────────────────────────────────────┐
│              Unit of Work パターン                     │
│                                                      │
│  ┌──────────────────────────────────────┐            │
│  │ UnitOfWork                           │            │
│  │                                      │            │
│  │  ┌─────────────────┐                │            │
│  │  │ UserRepository  │                │            │
│  │  └─────────────────┘                │            │
│  │  ┌─────────────────┐                │            │
│  │  │ OrderRepository │                │            │
│  │  └─────────────────┘                │            │
│  │  ┌─────────────────┐                │            │
│  │  │ Transaction     │                │            │
│  │  │ (共有)          │                │            │
│  │  └─────────────────┘                │            │
│  │                                      │            │
│  │  commit()   ← 全リポジトリの変更を   │            │
│  │  rollback()   一括コミット/ロールバック│           │
│  └──────────────────────────────────────┘            │
└──────────────────────────────────────────────────────┘
```

### 4.2 実装例

```typescript
// Unit of Work インターフェース
interface UnitOfWork {
  users: UserRepository;
  orders: OrderRepository;
  commit(): Promise<void>;
  rollback(): Promise<void>;
}

// Prisma 実装
class PrismaUnitOfWork implements UnitOfWork {
  users: UserRepository;
  orders: OrderRepository;
  private tx: PrismaClient;

  constructor(prisma: PrismaClient) {
    this.tx = prisma;
    this.users = new PostgresUserRepository(this.tx);
    this.orders = new PostgresOrderRepository(this.tx);
  }

  async commit(): Promise<void> {
    // Prisma の場合は $transaction で囲む
    // 実際にはトランザクション内で実行する設計にする
  }

  async rollback(): Promise<void> {
    // ロールバック処理
  }
}

// Service での使用
class OrderService {
  constructor(private uowFactory: () => UnitOfWork) {}

  async placeOrder(userId: string, items: OrderItem[]): Promise<Order> {
    const uow = this.uowFactory();
    try {
      const user = await uow.users.findById(userId);
      if (!user) throw new UserNotFoundError(userId);

      const order = Order.create(user, items);
      await uow.orders.save(order);

      // ユーザーのポイント更新
      user.addPoints(order.calculatePoints());
      await uow.users.save(user);

      await uow.commit();  // 全ての変更を一括コミット
      return order;
    } catch (error) {
      await uow.rollback();
      throw error;
    }
  }
}
```

---

## 5. 比較表

### 5.1 Repository パターンの実装方式比較

| 方式 | 説明 | メリット | デメリット |
|------|------|---------|----------|
| **Specific Repository** | エンティティごとに固有メソッド | 明確な API、型安全 | ボイラープレートが多い |
| **Generic Repository** | 共通 CRUD を基底クラスに | コード量削減 | 抽象化が過度になりがち |
| **CQRS + Repository** | 読み取り/書き込みを分離 | パフォーマンス最適化 | 複雑度が増す |
| **Specification Pattern** | クエリ条件をオブジェクト化 | 柔軟なクエリ構築 | 学習コスト高 |

### 5.2 データアクセスパターン比較

| パターン | 抽象度 | テスト容易性 | 複雑度 | 適用場面 |
|---------|--------|------------|--------|---------|
| **直接 SQL** | 低 | 低（DB 必要） | 低 | 小規模スクリプト |
| **ORM のみ** | 中 | 中 | 低〜中 | 中小規模アプリ |
| **Repository** | 高 | 高 | 中 | 中〜大規模アプリ |
| **Repository + UoW** | 高 | 高 | 高 | 大規模, DDD |
| **CQRS + ES** | 最高 | 高 | 最高 | 複雑ドメイン |

---

## 6. アンチパターン

### 6.1 Repository に ORM の型を漏洩させる

```typescript
// NG: Repository のインターフェースに Prisma の型が漏洩
interface UserRepository {
  findMany(args: Prisma.UserFindManyArgs): Promise<PrismaUser[]>;
  //       ^^^^^ インフラ層の型がドメイン層に漏洩
}

// OK: ドメイン層の型のみを使用
interface UserRepository {
  findAll(options?: FindOptions): Promise<User[]>;
  //                ^^^^^^^^^^^ ドメイン層で定義した型
}
```

**問題点**: ORM を切り替える際にインターフェースも変更が必要になり、Repository パターンの利点が失われる。

### 6.2 Generic Repository を万能と考える

```typescript
// NG: 全てのエンティティに同じ CRUD を提供
interface GenericRepository<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: string): Promise<void>;
}

// 問題: 「ログは削除禁止」「設定は1行のみ」等のドメインルールを表現できない

// OK: ドメインの要件に合わせたインターフェース
interface AuditLogRepository {
  append(log: AuditLog): Promise<void>;  // 追記のみ、削除なし
  findByDateRange(from: Date, to: Date): Promise<AuditLog[]>;
}

interface AppSettingsRepository {
  get(): Promise<AppSettings>;           // 常に1つだけ
  update(settings: AppSettings): Promise<void>;
}
```

---

## 7. FAQ

### Q1. 小規模プロジェクトでも Repository パターンは必要？

**A.** 小規模（CRUD 中心、テーブル 5 個以下）では不要な場合が多い。ORM を直接使う方がシンプル。以下の条件に当てはまる場合に導入を検討:
- ユニットテストでデータベースを使いたくない
- 将来的に DB の変更可能性がある
- ドメインロジックが複雑でサービス層のテストが重要

### Q2. Repository はテーブルごと？集約（Aggregate）ごと？

**A.** DDD を採用している場合は「集約ルートごと」が正解。例えば `Order` と `OrderItem` は別テーブルでも、`OrderRepository` で一括管理する。DDD でなければテーブルごとで問題ない。

```typescript
// DDD: 集約ルート単位
interface OrderRepository {
  findById(id: string): Promise<Order>;  // Order + OrderItems を含む
  save(order: Order): Promise<void>;     // Order + OrderItems を一括保存
}
```

### Q3. Repository をテストする場合、DB を使うべき？モックすべき？

**A.** 両方必要。
- **Repository 実装のテスト** → 実際の DB（テストコンテナ等）で統合テスト
- **Service 層のテスト** → InMemory Repository や Mock でユニットテスト

```
テストピラミッド:
  ┌─────────────┐
  │  E2E テスト  │  ← 少数、本番相当の DB
  ├─────────────┤
  │  統合テスト  │  ← Repository 実装のテスト（TestContainers）
  ├─────────────┤
  │ ユニットテスト│  ← Service のテスト（InMemory Repository）
  └─────────────┘
```

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| **Repository** | データアクセスの抽象化、ドメイン層に定義しインフラ層で実装 |
| **利点** | テスト容易性、DB 切り替え可能、関心の分離 |
| **Unit of Work** | 複数 Repository の変更を1トランザクションで管理 |
| **テスト** | InMemory 実装でサービス層をユニットテスト |
| **注意点** | 過度な抽象化を避け、ドメインの要件に合ったインターフェースを設計 |

---

## 次に読むべきガイド

- [00-mvc-mvvm.md](./00-mvc-mvvm.md) — UI 層のアーキテクチャパターン
- [02-event-sourcing-cqrs.md](./02-event-sourcing-cqrs.md) — イベント駆動とコマンド/クエリ分離
- DDD 入門ガイド — エンティティ、値オブジェクト、集約の設計

---

## 参考文献

1. **Martin Fowler** — "Patterns of Enterprise Application Architecture" — Repository パターンの原典 — https://martinfowler.com/eaaCatalog/repository.html
2. **Microsoft** — "Implementing the Repository and Unit of Work Patterns" — https://learn.microsoft.com/en-us/aspnet/mvc/overview/older-versions/getting-started-with-ef-5-using-mvc-4/implementing-the-repository-and-unit-of-work-patterns-in-an-asp-net-mvc-application
3. **Eric Evans** — "Domain-Driven Design: Tackling Complexity in the Heart of Software" — Addison-Wesley, 2003
