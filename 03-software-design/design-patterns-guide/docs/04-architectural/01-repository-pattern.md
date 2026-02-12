# Repository パターン — データアクセス抽象化

> Repository パターンでデータアクセスロジックをビジネスロジックから完全に分離し、テスト容易性・保守性・データソースの切り替え可能性を実現するための実践ガイド。DDD における集約ルートとの関係、Unit of Work、Specification パターンとの組み合わせ、ORM 別の実装例まで網羅する。

---

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| オブジェクト指向プログラミング | 中級（インターフェース、抽象クラス、DI） | [02-programming](../../../../02-programming/) |
| TypeScript / Python 基礎 | 中級（ジェネリクス、async/await） | [02-programming](../../../../02-programming/) |
| SQL の基礎 | 基礎（SELECT, INSERT, JOIN） | [06-data-and-security](../../../../06-data-and-security/) |
| SOLID 原則 | 基礎（依存性逆転原則 DIP） | [../../clean-code-principles/](../../clean-code-principles/) |
| MVC / MVVM の基礎 | 基礎 | [00-mvc-mvvm.md](./00-mvc-mvvm.md) |

---

## この章で学ぶこと

1. **Repository パターン** の目的と構造、ドメイン駆動設計（DDD）での位置付け
2. **インターフェースと実装の分離** — 依存性逆転原則（DIP）に基づくテスト可能な設計
3. **実装パターン** — Specific Repository、Generic Repository、Specification パターン
4. **Unit of Work パターン** — 複数 Repository のトランザクション管理
5. **ORM 別実装** — Prisma、SQLAlchemy、TypeORM、Drizzle での具体的実装

---

## 1. Repository パターンの全体像

### WHY: なぜ Repository パターンが必要か

データアクセスコードがビジネスロジックに混在すると、以下の問題が発生する:

1. **テスト困難** — ビジネスロジックのテストにデータベース接続が必要
2. **変更の波及** — DB スキーマ変更や ORM 変更が全てのサービスに影響
3. **重複コード** — 同じクエリが複数箇所に散在
4. **関心の分離違反** — 「何のデータが必要か」と「どうやって取得するか」が混在

Repository パターンはこれらを解決する:

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
│  │ UserService  │ ──→ │ UserRepository   │(Interface)│
│  │ (ビジネス    │     │  findById()      │           │
│  │  ロジック)   │     │  save()          │           │
│  └──────────────┘     │  findByEmail()   │           │
│                       └────────┬─────────┘           │
│         テスト時              │         本番時       │
│    ┌──────────────┐  ┌───────▼─────────┐             │
│    │ InMemory     │  │ PostgresUser    │             │
│    │ Repository   │  │ Repository      │ (実装)      │
│    └──────────────┘  │ (SQL はここだけ)  │             │
│                      └─────────────────┘             │
│  利点: テスト容易、DB 変更は実装クラスのみ            │
└──────────────────────────────────────────────────────┘
```

### 1.1 依存性逆転原則（DIP）との関係

```
┌───────────────────────────────────────────────────────────┐
│  依存性逆転原則 (Dependency Inversion Principle)           │
│                                                           │
│  NG: 上位モジュールが下位モジュールに直接依存               │
│                                                           │
│  UserService ──→ PostgresUserRepository ──→ PostgreSQL     │
│  (上位)          (下位)                    (詳細)          │
│                                                           │
│  問題: PostgreSQL を MongoDB に変えると UserService も変更  │
│                                                           │
│  ─────────────────────────────────────────────────────── │
│                                                           │
│  OK: 上位モジュールが抽象（インターフェース）に依存         │
│                                                           │
│  UserService ──→ UserRepository (Interface)                │
│  (上位)          (抽象)                                    │
│                       ▲                                   │
│                       │ 実装                               │
│                       │                                    │
│              PostgresUserRepository ──→ PostgreSQL          │
│              MongoUserRepository   ──→ MongoDB              │
│              InMemoryUserRepository (テスト用)              │
│                                                           │
│  利点: UserService は具体的な DB を知らない                 │
│        DB 変更時は実装クラスを差し替えるだけ                │
└───────────────────────────────────────────────────────────┘
```

### 1.2 レイヤーアーキテクチャでの位置付け

```
┌──────────────────────────────────────────────────┐
│  Presentation Layer (Controller / Handler)         │
│  → HTTP リクエスト/レスポンスの処理                │
├──────────────────────────────────────────────────┤
│  Application Layer (Service / UseCase)             │
│  → ユースケースの調整、トランザクション管理        │
│  → Repository Interface を利用                    │
├──────────────────────────────────────────────────┤
│  Domain Layer (Entity / Value Object)              │
│  → ビジネスルール、ドメインロジック                │
│  ┌─────────────────────────────────┐              │
│  │ Repository Interface            │ ← ここに定義 │
│  │ (ドメイン層に属する)            │              │
│  └─────────────────────────────────┘              │
├──────────────────────────────────────────────────┤
│  Infrastructure Layer                              │
│  ┌─────────────────────────────────┐              │
│  │ Repository Implementation       │ ← ここに実装 │
│  │ (PostgresUserRepository 等)     │              │
│  └─────────────────────────────────┘              │
│  → DB アクセス、外部 API 呼び出し                 │
└──────────────────────────────────────────────────┘

重要: Interface はドメイン層、Implementation はインフラ層
      → 依存の方向が内側（ドメイン）に向かう
```

### 1.3 DDD における集約（Aggregate）と Repository

```
┌───────────────────────────────────────────────────────────┐
│  DDD の集約ルート (Aggregate Root) と Repository           │
│                                                           │
│  原則: Repository は「集約ルートごと」に1つ                │
│                                                           │
│  ┌─────────── Order 集約 ───────────┐                     │
│  │  Order (集約ルート) ★              │                     │
│  │    ├── OrderItem                  │                     │
│  │    ├── OrderItem                  │                     │
│  │    └── ShippingAddress            │                     │
│  └───────────────────────────────────┘                     │
│           │                                               │
│           ▼                                               │
│  ┌────────────────────────────┐                           │
│  │  OrderRepository           │  ← 集約ルート単位          │
│  │  findById(id): Order       │  ← Order + Items + Address│
│  │  save(order): void         │  ← 一括保存               │
│  └────────────────────────────┘                           │
│                                                           │
│  NG: OrderItemRepository は作らない                        │
│      → OrderItem は Order 集約の一部であり、               │
│        単独で取得・保存すべきでない                         │
│                                                           │
│  OK: Customer 集約は別の Repository                        │
│  ┌─────────── Customer 集約 ──────────┐                    │
│  │  Customer (集約ルート) ★            │                    │
│  │    └── Address                     │                    │
│  └────────────────────────────────────┘                    │
│           │                                               │
│           ▼                                               │
│  ┌────────────────────────────┐                           │
│  │  CustomerRepository        │                           │
│  └────────────────────────────┘                           │
└───────────────────────────────────────────────────────────┘
```

---

## 2. TypeScript での実装

### 2.1 インターフェースと実装の分離（Prisma）

```typescript
// ============================================================
// Domain Layer: Repository Interface
// domain/repositories/UserRepository.ts
// ============================================================

import { User } from "../entities/User";

export interface FindOptions {
  page?: number;
  perPage?: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
  filter?: {
    active?: boolean;
    role?: string;
    createdAfter?: Date;
  };
}

export interface PaginatedResult<T> {
  data: T[];
  total: number;
  page: number;
  perPage: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

// Repository Interface — ドメイン層で定義
export interface UserRepository {
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<User | null>;
  findAll(options?: FindOptions): Promise<PaginatedResult<User>>;
  findByIds(ids: string[]): Promise<User[]>;
  save(user: User): Promise<User>;
  saveMany(users: User[]): Promise<User[]>;
  delete(id: string): Promise<void>;
  exists(email: string): Promise<boolean>;
  count(filter?: FindOptions["filter"]): Promise<number>;
}
```

```typescript
// ============================================================
// Infrastructure Layer: PostgreSQL 実装（Prisma）
// infrastructure/repositories/PrismaUserRepository.ts
// ============================================================

import { PrismaClient, User as PrismaUser } from "@prisma/client";
import {
  UserRepository,
  FindOptions,
  PaginatedResult,
} from "../../domain/repositories/UserRepository";
import { User } from "../../domain/entities/User";

export class PrismaUserRepository implements UserRepository {
  constructor(private readonly prisma: PrismaClient) {}

  async findById(id: string): Promise<User | null> {
    const row = await this.prisma.user.findUnique({
      where: { id },
      include: { profile: true },  // 集約の関連エンティティも取得
    });
    return row ? this.toDomain(row) : null;
  }

  async findByEmail(email: string): Promise<User | null> {
    const row = await this.prisma.user.findUnique({
      where: { email },
    });
    return row ? this.toDomain(row) : null;
  }

  async findAll(options?: FindOptions): Promise<PaginatedResult<User>> {
    const page = options?.page ?? 1;
    const perPage = options?.perPage ?? 20;
    const skip = (page - 1) * perPage;

    // フィルター条件の構築
    const where: Record<string, unknown> = {};
    if (options?.filter?.active !== undefined) {
      where.active = options.filter.active;
    }
    if (options?.filter?.role) {
      where.role = options.filter.role;
    }
    if (options?.filter?.createdAfter) {
      where.createdAt = { gte: options.filter.createdAfter };
    }

    const [rows, total] = await Promise.all([
      this.prisma.user.findMany({
        where,
        skip,
        take: perPage,
        orderBy: {
          [options?.sortBy ?? "createdAt"]: options?.sortOrder ?? "desc",
        },
      }),
      this.prisma.user.count({ where }),
    ]);

    const totalPages = Math.ceil(total / perPage);

    return {
      data: rows.map((row) => this.toDomain(row)),
      total,
      page,
      perPage,
      totalPages,
      hasNext: page < totalPages,
      hasPrev: page > 1,
    };
  }

  async findByIds(ids: string[]): Promise<User[]> {
    const rows = await this.prisma.user.findMany({
      where: { id: { in: ids } },
    });
    return rows.map((row) => this.toDomain(row));
  }

  async save(user: User): Promise<User> {
    const row = await this.prisma.user.upsert({
      where: { id: user.id },
      create: {
        id: user.id,
        name: user.name,
        email: user.email,
        role: user.role,
        active: user.active,
      },
      update: {
        name: user.name,
        email: user.email,
        role: user.role,
        active: user.active,
        updatedAt: new Date(),
      },
    });
    return this.toDomain(row);
  }

  async saveMany(users: User[]): Promise<User[]> {
    const results = await this.prisma.$transaction(
      users.map((user) =>
        this.prisma.user.upsert({
          where: { id: user.id },
          create: {
            id: user.id,
            name: user.name,
            email: user.email,
            role: user.role,
            active: user.active,
          },
          update: {
            name: user.name,
            email: user.email,
            role: user.role,
            active: user.active,
          },
        })
      )
    );
    return results.map((row) => this.toDomain(row));
  }

  async delete(id: string): Promise<void> {
    await this.prisma.user.delete({ where: { id } });
  }

  async exists(email: string): Promise<boolean> {
    const count = await this.prisma.user.count({ where: { email } });
    return count > 0;
  }

  async count(filter?: FindOptions["filter"]): Promise<number> {
    const where: Record<string, unknown> = {};
    if (filter?.active !== undefined) where.active = filter.active;
    if (filter?.role) where.role = filter.role;
    return this.prisma.user.count({ where });
  }

  // ドメインモデルへの変換
  private toDomain(row: PrismaUser): User {
    return new User({
      id: row.id,
      name: row.name,
      email: row.email,
      role: row.role as "admin" | "user",
      active: row.active,
      createdAt: row.createdAt,
      updatedAt: row.updatedAt,
    });
  }
}
```

### 2.2 テスト用のインメモリ実装

```typescript
// ============================================================
// Test: In-Memory 実装
// tests/repositories/InMemoryUserRepository.ts
// ============================================================

export class InMemoryUserRepository implements UserRepository {
  private users: Map<string, User> = new Map();

  async findById(id: string): Promise<User | null> {
    return this.users.get(id) ?? null;
  }

  async findByEmail(email: string): Promise<User | null> {
    return (
      [...this.users.values()].find((u) => u.email === email) ?? null
    );
  }

  async findAll(options?: FindOptions): Promise<PaginatedResult<User>> {
    let all = [...this.users.values()];

    // フィルタリング
    if (options?.filter?.active !== undefined) {
      all = all.filter((u) => u.active === options.filter!.active);
    }
    if (options?.filter?.role) {
      all = all.filter((u) => u.role === options.filter!.role);
    }
    if (options?.filter?.createdAfter) {
      all = all.filter(
        (u) => u.createdAt >= options.filter!.createdAfter!
      );
    }

    // ソート
    const sortBy = options?.sortBy ?? "createdAt";
    const sortOrder = options?.sortOrder ?? "desc";
    all.sort((a, b) => {
      const aVal = (a as any)[sortBy];
      const bVal = (b as any)[sortBy];
      const cmp = aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
      return sortOrder === "desc" ? -cmp : cmp;
    });

    // ページネーション
    const page = options?.page ?? 1;
    const perPage = options?.perPage ?? 20;
    const start = (page - 1) * perPage;
    const data = all.slice(start, start + perPage);
    const totalPages = Math.ceil(all.length / perPage);

    return {
      data,
      total: all.length,
      page,
      perPage,
      totalPages,
      hasNext: page < totalPages,
      hasPrev: page > 1,
    };
  }

  async findByIds(ids: string[]): Promise<User[]> {
    return ids
      .map((id) => this.users.get(id))
      .filter((u): u is User => u !== undefined);
  }

  async save(user: User): Promise<User> {
    this.users.set(user.id, user);
    return user;
  }

  async saveMany(users: User[]): Promise<User[]> {
    users.forEach((u) => this.users.set(u.id, u));
    return users;
  }

  async delete(id: string): Promise<void> {
    this.users.delete(id);
  }

  async exists(email: string): Promise<boolean> {
    return [...this.users.values()].some((u) => u.email === email);
  }

  async count(filter?: FindOptions["filter"]): Promise<number> {
    let all = [...this.users.values()];
    if (filter?.active !== undefined) {
      all = all.filter((u) => u.active === filter.active);
    }
    return all.length;
  }

  // テストヘルパー
  clear(): void {
    this.users.clear();
  }

  seed(users: User[]): void {
    users.forEach((u) => this.users.set(u.id, u));
  }

  getAll(): User[] {
    return [...this.users.values()];
  }
}
```

### 2.3 Service 層での使用と DI

```typescript
// ============================================================
// Application Layer: Service
// application/services/UserService.ts
// ============================================================

export class UserService {
  constructor(
    private readonly userRepo: UserRepository,   // インターフェースに依存
    private readonly emailService: EmailService,
    private readonly eventBus: EventBus,
  ) {}

  async registerUser(name: string, email: string): Promise<User> {
    // ビジネスルールの検証
    const existing = await this.userRepo.exists(email);
    if (existing) {
      throw new DuplicateEmailError(email);
    }

    // ドメインオブジェクト生成
    const user = User.create(name, email);

    // 永続化
    const saved = await this.userRepo.save(user);

    // 副作用（ドメインイベント発行）
    await this.eventBus.publish(new UserRegisteredEvent(saved));
    await this.emailService.sendWelcome(saved.email);

    return saved;
  }

  async deactivateUser(id: string): Promise<void> {
    const user = await this.userRepo.findById(id);
    if (!user) throw new UserNotFoundError(id);

    user.deactivate();
    await this.userRepo.save(user);
    await this.eventBus.publish(new UserDeactivatedEvent(user));
  }

  async getUserList(options?: FindOptions): Promise<PaginatedResult<User>> {
    return this.userRepo.findAll({
      ...options,
      filter: { ...options?.filter, active: true },
    });
  }
}

// ============================================================
// DI Container（tsyringe の例）
// ============================================================
import { container } from "tsyringe";

// 本番環境: PostgreSQL 実装を注入
container.register<UserRepository>("UserRepository", {
  useClass: PrismaUserRepository,
});

// テスト環境: InMemory 実装を注入
container.register<UserRepository>("UserRepository", {
  useClass: InMemoryUserRepository,
});
```

### 2.4 Service のテスト

```typescript
// ============================================================
// テスト — InMemory Repository を使用
// ============================================================

describe("UserService", () => {
  let service: UserService;
  let userRepo: InMemoryUserRepository;
  let emailService: jest.Mocked<EmailService>;
  let eventBus: jest.Mocked<EventBus>;

  beforeEach(() => {
    userRepo = new InMemoryUserRepository();
    emailService = { sendWelcome: jest.fn() } as any;
    eventBus = { publish: jest.fn() } as any;
    service = new UserService(userRepo, emailService, eventBus);
  });

  describe("registerUser", () => {
    test("新規ユーザーを登録できる", async () => {
      const user = await service.registerUser("Alice", "alice@example.com");

      // ドメインの検証
      expect(user.name).toBe("Alice");
      expect(user.email).toBe("alice@example.com");
      expect(user.active).toBe(true);

      // 永続化の検証
      const found = await userRepo.findByEmail("alice@example.com");
      expect(found).not.toBeNull();
      expect(found!.id).toBe(user.id);

      // 副作用の検証
      expect(emailService.sendWelcome).toHaveBeenCalledWith("alice@example.com");
      expect(eventBus.publish).toHaveBeenCalledWith(
        expect.objectContaining({ type: "UserRegistered" })
      );
    });

    test("重複メールアドレスはエラー", async () => {
      await service.registerUser("Alice", "alice@example.com");

      await expect(
        service.registerUser("Bob", "alice@example.com")
      ).rejects.toThrow(DuplicateEmailError);

      // 副作用が呼ばれないことを検証
      expect(emailService.sendWelcome).toHaveBeenCalledTimes(1);
    });
  });

  describe("deactivateUser", () => {
    test("ユーザーを無効化できる", async () => {
      const user = await service.registerUser("Alice", "alice@example.com");
      await service.deactivateUser(user.id);

      const found = await userRepo.findById(user.id);
      expect(found!.active).toBe(false);
    });

    test("存在しないユーザーはエラー", async () => {
      await expect(
        service.deactivateUser("non-existent-id")
      ).rejects.toThrow(UserNotFoundError);
    });
  });

  describe("getUserList", () => {
    test("アクティブユーザーのみ返す", async () => {
      const alice = await service.registerUser("Alice", "alice@example.com");
      await service.registerUser("Bob", "bob@example.com");
      await service.deactivateUser(alice.id);

      const result = await service.getUserList();
      expect(result.data).toHaveLength(1);
      expect(result.data[0].name).toBe("Bob");
    });
  });
});
```

---

## 3. Python での実装

### 3.1 Abstract Base Class による定義

```python
# ============================================================
# domain/repositories/user_repository.py
# ============================================================
from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass
from domain.entities.user import User


@dataclass
class FindOptions:
    page: int = 1
    per_page: int = 20
    sort_by: str = "created_at"
    sort_order: str = "desc"
    active: Optional[bool] = None
    role: Optional[str] = None


@dataclass
class PaginatedResult:
    data: List[User]
    total: int
    page: int
    per_page: int

    @property
    def total_pages(self) -> int:
        return -(-self.total // self.per_page)  # 切り上げ除算

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1


class UserRepository(ABC):
    """ユーザーリポジトリインターフェース（ドメイン層で定義）"""

    @abstractmethod
    async def find_by_id(self, user_id: str) -> Optional[User]:
        ...

    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[User]:
        ...

    @abstractmethod
    async def find_all(self, options: Optional[FindOptions] = None) -> PaginatedResult:
        ...

    @abstractmethod
    async def save(self, user: User) -> User:
        ...

    @abstractmethod
    async def delete(self, user_id: str) -> None:
        ...

    @abstractmethod
    async def exists(self, email: str) -> bool:
        ...
```

```python
# ============================================================
# infrastructure/repositories/sqlalchemy_user_repository.py
# ============================================================
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from domain.repositories.user_repository import (
    UserRepository, FindOptions, PaginatedResult
)
from domain.entities.user import User
from infrastructure.models import UserModel


class SQLAlchemyUserRepository(UserRepository):
    """SQLAlchemy を使った PostgreSQL 実装"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def find_by_id(self, user_id: str) -> Optional[User]:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return self._to_domain(row) if row else None

    async def find_by_email(self, email: str) -> Optional[User]:
        stmt = select(UserModel).where(UserModel.email == email)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return self._to_domain(row) if row else None

    async def find_all(self, options: Optional[FindOptions] = None) -> PaginatedResult:
        opts = options or FindOptions()

        # 基本クエリ
        stmt = select(UserModel)
        count_stmt = select(func.count()).select_from(UserModel)

        # フィルター
        if opts.active is not None:
            stmt = stmt.where(UserModel.active == opts.active)
            count_stmt = count_stmt.where(UserModel.active == opts.active)
        if opts.role:
            stmt = stmt.where(UserModel.role == opts.role)
            count_stmt = count_stmt.where(UserModel.role == opts.role)

        # ソート
        sort_col = getattr(UserModel, opts.sort_by, UserModel.created_at)
        stmt = stmt.order_by(
            sort_col.desc() if opts.sort_order == "desc" else sort_col.asc()
        )

        # ページネーション
        offset = (opts.page - 1) * opts.per_page
        stmt = stmt.offset(offset).limit(opts.per_page)

        # 実行
        result = await self._session.execute(stmt)
        count_result = await self._session.execute(count_stmt)
        rows = result.scalars().all()
        total = count_result.scalar()

        return PaginatedResult(
            data=[self._to_domain(row) for row in rows],
            total=total,
            page=opts.page,
            per_page=opts.per_page,
        )

    async def save(self, user: User) -> User:
        model = UserModel(
            id=user.id,
            name=user.name,
            email=user.email,
            role=user.role,
            active=user.active,
        )
        merged = await self._session.merge(model)
        await self._session.flush()
        return self._to_domain(merged)

    async def delete(self, user_id: str) -> None:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one()
        await self._session.delete(model)

    async def exists(self, email: str) -> bool:
        stmt = select(func.count()).select_from(UserModel).where(
            UserModel.email == email
        )
        result = await self._session.execute(stmt)
        return result.scalar() > 0

    def _to_domain(self, model: UserModel) -> User:
        return User(
            id=model.id,
            name=model.name,
            email=model.email,
            role=model.role,
            active=model.active,
            created_at=model.created_at,
        )
```

```python
# ============================================================
# テスト用 InMemory 実装
# tests/repositories/in_memory_user_repository.py
# ============================================================
class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self._users: dict[str, User] = {}

    async def find_by_id(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    async def find_by_email(self, email: str) -> Optional[User]:
        return next(
            (u for u in self._users.values() if u.email == email), None
        )

    async def find_all(self, options: Optional[FindOptions] = None) -> PaginatedResult:
        opts = options or FindOptions()
        users = list(self._users.values())

        # フィルター
        if opts.active is not None:
            users = [u for u in users if u.active == opts.active]
        if opts.role:
            users = [u for u in users if u.role == opts.role]

        # ページネーション
        total = len(users)
        start = (opts.page - 1) * opts.per_page
        data = users[start:start + opts.per_page]

        return PaginatedResult(
            data=data, total=total, page=opts.page, per_page=opts.per_page
        )

    async def save(self, user: User) -> User:
        self._users[user.id] = user
        return user

    async def delete(self, user_id: str) -> None:
        self._users.pop(user_id, None)

    async def exists(self, email: str) -> bool:
        return any(u.email == email for u in self._users.values())

    def clear(self):
        self._users.clear()
```

---

## 4. Specification パターン

### WHY: なぜ Specification パターンが必要か

複雑な検索条件を Repository のメソッドとして追加し続けると、インターフェースが肥大化する:

```typescript
// NG: 検索条件ごとにメソッドが増殖
interface UserRepository {
  findByName(name: string): Promise<User[]>;
  findByRole(role: string): Promise<User[]>;
  findByNameAndRole(name: string, role: string): Promise<User[]>;
  findActiveByRole(role: string): Promise<User[]>;
  findInactiveCreatedBefore(date: Date): Promise<User[]>;
  // ... 無限に増える
}

// OK: Specification パターンで条件をオブジェクト化
interface UserRepository {
  findAll(spec?: Specification<User>): Promise<User[]>;
  // 1つのメソッドで全ての検索条件に対応
}
```

### 4.1 Specification の実装

```typescript
// ============================================================
// Specification パターン
// ============================================================

// 基底 Specification
interface Specification<T> {
  isSatisfiedBy(entity: T): boolean;
  toSQL(): { where: string; params: unknown[] };
  and(other: Specification<T>): Specification<T>;
  or(other: Specification<T>): Specification<T>;
  not(): Specification<T>;
}

abstract class BaseSpecification<T> implements Specification<T> {
  abstract isSatisfiedBy(entity: T): boolean;
  abstract toSQL(): { where: string; params: unknown[] };

  and(other: Specification<T>): Specification<T> {
    return new AndSpecification(this, other);
  }

  or(other: Specification<T>): Specification<T> {
    return new OrSpecification(this, other);
  }

  not(): Specification<T> {
    return new NotSpecification(this);
  }
}

class AndSpecification<T> extends BaseSpecification<T> {
  constructor(private left: Specification<T>, private right: Specification<T>) {
    super();
  }

  isSatisfiedBy(entity: T): boolean {
    return this.left.isSatisfiedBy(entity) && this.right.isSatisfiedBy(entity);
  }

  toSQL(): { where: string; params: unknown[] } {
    const l = this.left.toSQL();
    const r = this.right.toSQL();
    return {
      where: `(${l.where}) AND (${r.where})`,
      params: [...l.params, ...r.params],
    };
  }
}

class OrSpecification<T> extends BaseSpecification<T> {
  constructor(private left: Specification<T>, private right: Specification<T>) {
    super();
  }

  isSatisfiedBy(entity: T): boolean {
    return this.left.isSatisfiedBy(entity) || this.right.isSatisfiedBy(entity);
  }

  toSQL(): { where: string; params: unknown[] } {
    const l = this.left.toSQL();
    const r = this.right.toSQL();
    return {
      where: `(${l.where}) OR (${r.where})`,
      params: [...l.params, ...r.params],
    };
  }
}

class NotSpecification<T> extends BaseSpecification<T> {
  constructor(private spec: Specification<T>) {
    super();
  }

  isSatisfiedBy(entity: T): boolean {
    return !this.spec.isSatisfiedBy(entity);
  }

  toSQL(): { where: string; params: unknown[] } {
    const s = this.spec.toSQL();
    return { where: `NOT (${s.where})`, params: s.params };
  }
}

// 具体的な Specification
class ActiveUserSpec extends BaseSpecification<User> {
  isSatisfiedBy(user: User): boolean {
    return user.active;
  }

  toSQL() {
    return { where: "active = true", params: [] };
  }
}

class UserWithRoleSpec extends BaseSpecification<User> {
  constructor(private role: string) {
    super();
  }

  isSatisfiedBy(user: User): boolean {
    return user.role === this.role;
  }

  toSQL() {
    return { where: "role = $1", params: [this.role] };
  }
}

class CreatedAfterSpec extends BaseSpecification<User> {
  constructor(private date: Date) {
    super();
  }

  isSatisfiedBy(user: User): boolean {
    return user.createdAt >= this.date;
  }

  toSQL() {
    return { where: "created_at >= $1", params: [this.date] };
  }
}

// 使用例: 条件を組み合わせ
const activeAdmins = new ActiveUserSpec().and(new UserWithRoleSpec("admin"));
const recentOrAdmin = new CreatedAfterSpec(lastMonth).or(new UserWithRoleSpec("admin"));

// Repository での使用
const users = await userRepo.findAll(activeAdmins);
```

---

## 5. Unit of Work パターンとの組み合わせ

### WHY: なぜ Unit of Work が必要か

複数の Repository にまたがる操作をアトミックに実行したい場合、各 Repository が独立にコミットすると不整合が生じる:

```
NG: 各 Repository が独立してコミット
  OrderRepository.save(order)    → 成功（コミット済み）
  UserRepository.updatePoints()  → 失敗（ロールバック）
  → 注文は作成されたがポイントが付与されていない不整合

OK: Unit of Work で一括コミット
  UnitOfWork 開始
    OrderRepository.save(order)     → トランザクション内
    UserRepository.updatePoints()   → トランザクション内
  UnitOfWork.commit()               → 全ての変更を一括コミット
  失敗時 → UnitOfWork.rollback()    → 全ての変更を一括ロールバック
```

### 5.1 Unit of Work の構造

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

### 5.2 実装例

```typescript
// ============================================================
// Unit of Work インターフェース
// ============================================================
interface UnitOfWork {
  readonly users: UserRepository;
  readonly orders: OrderRepository;
  readonly auditLogs: AuditLogRepository;
  commit(): Promise<void>;
  rollback(): Promise<void>;
}

// ============================================================
// Prisma 実装
// ============================================================
class PrismaUnitOfWork implements UnitOfWork {
  readonly users: UserRepository;
  readonly orders: OrderRepository;
  readonly auditLogs: AuditLogRepository;

  private constructor(private tx: PrismaClient) {
    this.users = new PrismaUserRepository(tx);
    this.orders = new PrismaOrderRepository(tx);
    this.auditLogs = new PrismaAuditLogRepository(tx);
  }

  // ファクトリメソッド: トランザクション内で実行
  static async execute<T>(
    prisma: PrismaClient,
    fn: (uow: UnitOfWork) => Promise<T>
  ): Promise<T> {
    return prisma.$transaction(async (tx) => {
      const uow = new PrismaUnitOfWork(tx as PrismaClient);
      return fn(uow);
    });
  }

  async commit(): Promise<void> {
    // Prisma $transaction は自動コミット
  }

  async rollback(): Promise<void> {
    // Prisma $transaction は例外時に自動ロールバック
    throw new Error("Transaction aborted");
  }
}

// ============================================================
// Service での使用
// ============================================================
class OrderService {
  constructor(
    private prisma: PrismaClient,
    private eventBus: EventBus,
  ) {}

  async placeOrder(userId: string, items: OrderItem[]): Promise<Order> {
    const result = await PrismaUnitOfWork.execute(this.prisma, async (uow) => {
      // 1. ユーザー取得
      const user = await uow.users.findById(userId);
      if (!user) throw new UserNotFoundError(userId);

      // 2. 注文作成（ドメインロジック）
      const order = Order.create(user, items);

      // 3. 注文保存
      await uow.orders.save(order);

      // 4. ユーザーのポイント更新
      user.addPoints(order.calculatePoints());
      await uow.users.save(user);

      // 5. 監査ログ記録
      await uow.auditLogs.append(
        AuditLog.create("ORDER_PLACED", userId, { orderId: order.id })
      );

      return order;
    });
    // 全ての変更がトランザクション内で一括コミットされる

    // トランザクション成功後にイベント発行
    await this.eventBus.publish(new OrderPlacedEvent(result));

    return result;
  }
}
```

---

## 6. 比較表

### 6.1 Repository パターンの実装方式比較

| 方式 | 説明 | メリット | デメリット | 適する場面 |
|------|------|---------|----------|----------|
| **Specific Repository** | エンティティごとに固有メソッド | 明確な API、型安全、ドメイン表現力 | ボイラープレートが多い | DDD、中〜大規模 |
| **Generic Repository** | 共通 CRUD を基底クラスに | コード量削減、統一的な API | 抽象化が過度になりがち | CRUD 中心、小〜中規模 |
| **Specification Pattern** | クエリ条件をオブジェクト化 | 柔軟なクエリ構築、条件の再利用 | 学習コスト高、複雑 | 複雑な検索要件 |
| **CQRS + Repository** | 読み取り/書き込みを分離 | パフォーマンス最適化 | 複雑度が増す | 読み書き比率が偏る |

### 6.2 データアクセスパターン比較

| パターン | 抽象度 | テスト容易性 | 複雑度 | 学習コスト | 適用場面 |
|---------|--------|------------|--------|----------|---------|
| **直接 SQL** | 低 | 低（DB 必要） | 低 | 低 | 小規模スクリプト |
| **ORM のみ** | 中 | 中 | 低〜中 | 中 | 中小規模アプリ |
| **Active Record** | 中 | 中 | 低 | 低 | Rails/Django |
| **Repository** | 高 | 高 | 中 | 中〜高 | 中〜大規模アプリ |
| **Repository + UoW** | 高 | 高 | 高 | 高 | 大規模, DDD |
| **CQRS + ES** | 最高 | 高 | 最高 | 最高 | 複雑ドメイン |

### 6.3 ORM/ライブラリ別の Repository 実装の特徴

| ORM / Library | 言語 | Repository 実装のしやすさ | UoW サポート | 特記事項 |
|--------------|------|------------------------|-----------|---------|
| **Prisma** | TypeScript | 高 | $transaction | 型安全、スキーマファースト |
| **Drizzle** | TypeScript | 高 | transaction() | SQL に近い、軽量 |
| **TypeORM** | TypeScript | 中 | QueryRunner | Repository パターン組み込み |
| **SQLAlchemy** | Python | 高 | Session | 最も柔軟、UoW 組み込み |
| **Django ORM** | Python | 中 | atomic() | Active Record 寄り |
| **GORM** | Go | 中 | Transaction | シンプルだが型安全性は低い |
| **Entity Framework** | C# | 高 | DbContext | UoW パターンが標準 |

---

## 7. アンチパターン

### 7.1 Repository に ORM の型を漏洩させる

```typescript
// NG: Repository のインターフェースに Prisma の型が漏洩
interface UserRepository {
  findMany(args: Prisma.UserFindManyArgs): Promise<PrismaUser[]>;
  //       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^
  //       インフラ層の型がドメイン層に漏洩
}

// OK: ドメイン層の型のみを使用
interface UserRepository {
  findAll(options?: FindOptions): Promise<PaginatedResult<User>>;
  //                ^^^^^^^^^^^                            ^^^^
  //                ドメイン層で定義した型のみ
}
```

**なぜ NG か**: ORM を切り替える際（Prisma → Drizzle）にインターフェースも変更が必要になり、Repository パターンの「実装の差し替え可能性」が失われる。ドメイン層はインフラストラクチャの詳細を知るべきではない（DIP 違反）。

### 7.2 Generic Repository を万能と考える

```typescript
// NG: 全てのエンティティに同じ CRUD を提供
interface GenericRepository<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: string): Promise<void>;
}

// 問題: ドメインルールを表現できない
// 「ログは削除禁止」「設定は常に1つ」等

// OK: ドメインの要件に合わせたインターフェース
interface AuditLogRepository {
  append(log: AuditLog): Promise<void>;  // 追記のみ、削除不可
  findByDateRange(from: Date, to: Date): Promise<AuditLog[]>;
  // delete() メソッドは意図的に存在しない
}

interface AppSettingsRepository {
  get(): Promise<AppSettings>;           // 常に1つ
  update(settings: AppSettings): Promise<void>;
  // findAll() や delete() は不要
}

interface OrderRepository {
  findById(id: string): Promise<Order>;  // Order + OrderItems を含む
  save(order: Order): Promise<void>;     // Order + OrderItems を一括保存
  // save は集約全体を保存、個別の OrderItem の CRUD は提供しない
}
```

**なぜ NG か**: Generic Repository は「最小公約数」のインターフェースしか提供できず、ドメインの意図（「ログは不変」「集約は一括保存」等）を表現できない。Repository はドメインの言語（ユビキタス言語）で定義すべき。

### 7.3 Repository 内でビジネスロジックを実行する

```typescript
// NG: Repository がビジネスルールを知っている
class PostgresUserRepository implements UserRepository {
  async save(user: User): Promise<User> {
    // ビジネスルールの検証が Repository に混在
    if (!user.email.includes("@")) {
      throw new Error("Invalid email");
    }
    // VIP判定のロジックが Repository に
    if (user.purchaseTotal > 100000) {
      user.role = "vip";
    }
    return this.prisma.user.upsert(/* ... */);
  }
}

// OK: Repository は永続化のみ、ビジネスロジックはドメイン層
class User {
  // ビジネスルールはエンティティに
  updateEmail(newEmail: string): void {
    if (!newEmail.includes("@")) {
      throw new InvalidEmailError(newEmail);
    }
    this.email = newEmail;
  }

  checkVipStatus(): void {
    if (this.purchaseTotal > 100000) {
      this.role = "vip";
    }
  }
}

class PostgresUserRepository implements UserRepository {
  async save(user: User): Promise<User> {
    // 永続化のみ。バリデーションやビジネスロジックは含まない
    return this.prisma.user.upsert(/* ... */);
  }
}
```

**なぜ NG か**: Repository の責務はデータの永続化と取得のみ。ビジネスロジックは Domain Entity や Domain Service に配置する。Repository にロジックがあると、InMemory 実装でテストする際にそのロジックが実行されず、テストの信頼性が下がる。

### 7.4 過度な抽象化（YAGNI 違反）

```typescript
// NG: まだ1種類のDBしか使わないのに抽象化しすぎ
interface IRepositoryFactory<T> {
  createRepository(type: "postgres" | "mongo" | "dynamodb"): IRepository<T>;
}

interface IRepository<T> extends ICrudRepository<T>, ISearchRepository<T>, IAuditableRepository<T> {
  // ...
}

// OK: 必要になった時点で抽象化
// 現時点では Prisma 直接使用で十分なら:
class UserRepository {
  constructor(private prisma: PrismaClient) {}

  async findById(id: string): Promise<User | null> {
    // ...
  }
}
// テスト時にだけ InMemory に差し替えたい場合に Interface を導入
```

**なぜ NG か**: 将来の「可能性」のために現在の複雑さを増やすのは YAGNI 違反。Repository パターンの導入自体も「テストの必要性」か「データソースの切り替え可能性」が明確になってからで十分。

---

## 8. 実践演習

### 演習 1（基礎）: Repository Interface の設計

ブログサービスの `PostRepository` インターフェースを設計せよ。

**要件**:
- 投稿の CRUD
- タグによる検索
- 著者による検索
- 下書き/公開済みのフィルタリング
- ページネーション
- 人気順（いいね数）ソート

**制約**: DDD の集約を意識し、Post は Comment を含む集約とする（CommentRepository は作らない）

**期待する出力**:
- `PostRepository` インターフェース（TypeScript）
- 各メソッドの JSDoc コメント付き

---

### 演習 2（応用）: InMemory Repository + Service テスト

以下の `BookingService` のテストを、`InMemoryRoomRepository` と `InMemoryBookingRepository` を実装して書け。

```typescript
class BookingService {
  constructor(
    private roomRepo: RoomRepository,
    private bookingRepo: BookingRepository,
  ) {}

  async createBooking(roomId: string, userId: string, date: Date): Promise<Booking> {
    const room = await this.roomRepo.findById(roomId);
    if (!room) throw new RoomNotFoundError(roomId);
    if (!room.isAvailable) throw new RoomUnavailableError(roomId);

    const existingBooking = await this.bookingRepo.findByRoomAndDate(roomId, date);
    if (existingBooking) throw new DoubleBookingError(roomId, date);

    const booking = Booking.create(roomId, userId, date);
    return this.bookingRepo.save(booking);
  }
}
```

**テストケース**:
1. 正常予約
2. 存在しない部屋
3. 利用不可の部屋
4. 二重予約

**期待する出力**: `InMemoryRoomRepository`、`InMemoryBookingRepository` の実装と4つのテストケース

---

### 演習 3（発展）: Unit of Work + Specification パターン

EC サイトの注文処理を Unit of Work と Specification パターンで実装せよ。

**要件**:
1. 注文作成時に在庫を減らす（OrderRepository + ProductRepository）
2. 在庫不足の場合はロールバック
3. 「在庫5個以下の商品」を Specification で検索
4. トランザクション内で監査ログも記録

**期待する出力**:
- `LowStockSpec` Specification
- `PlaceOrderUseCase` (Unit of Work 使用)
- テストコード

---

## 9. FAQ

### Q1. 小規模プロジェクトでも Repository パターンは必要？

**A.** 小規模（CRUD 中心、テーブル 5 個以下）では不要な場合が多い。ORM を直接使う方がシンプル。以下の条件に **2つ以上** 当てはまる場合に導入を検討:

1. ユニットテストでデータベースを使いたくない
2. 将来的に DB の変更可能性がある（PostgreSQL → DynamoDB 等）
3. ドメインロジックが複雑でサービス層のテストが重要
4. 複数のデータソースを組み合わせる（DB + 外部 API + キャッシュ）
5. チームが大きく、データアクセス層の統一的なインターフェースが必要

スタートアップの初期段階では ORM 直接使用で始め、テストの必要性が出てきた時点で Repository を導入するのが現実的。

### Q2. Repository はテーブルごと？集約（Aggregate）ごと？

**A.** DDD を採用している場合は **「集約ルートごと」が正解**。例えば `Order` と `OrderItem` は別テーブルでも、`OrderRepository` で一括管理する。DDD でなければテーブルごとで問題ない。

```typescript
// DDD: 集約ルート単位
interface OrderRepository {
  findById(id: string): Promise<Order>;
  // ↑ Order + OrderItems + ShippingAddress を含む集約全体を返す
  save(order: Order): Promise<void>;
  // ↑ Order + OrderItems + ShippingAddress を一括保存
}

// 非DDD: テーブル単位
interface OrderRepository {
  findById(id: string): Promise<Order>;  // Order のみ
}
interface OrderItemRepository {
  findByOrderId(orderId: string): Promise<OrderItem[]>;
}
```

集約単位の Repository のメリットは、ビジネスルール（「OrderItem の合計金額は Order の total と一致する」等）を集約内で一貫して保証できること。

### Q3. Repository をテストする場合、DB を使うべき？モックすべき？

**A.** 両方必要。テストの種類に応じて使い分ける:

```
テストピラミッド:
  ┌─────────────────┐
  │    E2E テスト     │  ← 少数、本番相当の DB
  ├─────────────────┤
  │   統合テスト      │  ← Repository 実装のテスト（TestContainers）
  ├─────────────────┤
  │ ユニットテスト    │  ← Service のテスト（InMemory Repository）
  └─────────────────┘
```

| テスト種類 | Repository | 目的 |
|-----------|-----------|------|
| **ユニットテスト** | InMemory 実装 | Service のビジネスロジック検証 |
| **統合テスト** | 本物の DB（TestContainers） | SQL / ORM の正しさを検証 |
| **E2E テスト** | 本物の DB | システム全体の動作検証 |

```typescript
// 統合テスト: TestContainers で PostgreSQL を起動
describe("PrismaUserRepository (Integration)", () => {
  let prisma: PrismaClient;
  let repo: PrismaUserRepository;

  beforeAll(async () => {
    // Docker で PostgreSQL コンテナを起動
    const container = await new PostgreSqlContainer().start();
    prisma = new PrismaClient({
      datasources: { db: { url: container.getConnectionUri() } },
    });
    await prisma.$executeRaw`CREATE TABLE ...`;
    repo = new PrismaUserRepository(prisma);
  });

  test("save and findById", async () => {
    const user = User.create("Alice", "alice@example.com");
    await repo.save(user);

    const found = await repo.findById(user.id);
    expect(found).not.toBeNull();
    expect(found!.email).toBe("alice@example.com");
  });
});
```

### Q4. Active Record と Repository、どちらを使うべき？

**A.** プロジェクトの規模とフレームワークによる:

| 基準 | Active Record | Repository |
|------|---------------|-----------|
| **フレームワーク** | Rails, Django, Laravel | Express, Spring, 自前 |
| **プロジェクト規模** | 小〜中 | 中〜大 |
| **テスト要件** | 統合テスト中心 | ユニットテスト重視 |
| **ドメインの複雑さ** | 低〜中 | 高 |
| **チーム規模** | 小（1-5人） | 中〜大（5人以上） |

Active Record（Rails の `User.find_by(email: ...)` 等）はフレームワークの規約に従うなら最もシンプル。Repository は DDD やクリーンアーキテクチャを採用する場合に適する。

### Q5. Repository の返り値はドメインエンティティ？DTO？

**A.** **ドメインエンティティ** を返すのが正解。Repository はドメイン層のインターフェースであり、ドメインの言語（Entity, Value Object）で結果を返す。DTO（Data Transfer Object）への変換は Presentation 層（Controller / Serializer）の責務。

```typescript
// OK: ドメインエンティティを返す
interface UserRepository {
  findById(id: string): Promise<User>;  // ← User はドメインエンティティ
}

// NG: DTO を返す
interface UserRepository {
  findById(id: string): Promise<UserResponseDTO>;  // ← これは Controller の仕事
}
```

### Q6. Repository にキャッシュを組み込むべき？

**A.** Repository をデコレーターパターンでラップするのが推奨。Repository インターフェースを変更せずにキャッシュ層を追加できる:

```typescript
// キャッシュ付き Repository（Decorator パターン）
class CachedUserRepository implements UserRepository {
  constructor(
    private inner: UserRepository,  // 実際の DB Repository
    private cache: CacheClient,     // Redis 等
    private ttl: number = 300,      // 5分
  ) {}

  async findById(id: string): Promise<User | null> {
    // 1. キャッシュ確認
    const cached = await this.cache.get(`user:${id}`);
    if (cached) return JSON.parse(cached);

    // 2. DB から取得
    const user = await this.inner.findById(id);
    if (user) {
      await this.cache.set(`user:${id}`, JSON.stringify(user), this.ttl);
    }
    return user;
  }

  async save(user: User): Promise<User> {
    const saved = await this.inner.save(user);
    // キャッシュ無効化
    await this.cache.del(`user:${saved.id}`);
    return saved;
  }

  // ... 他のメソッドも同様
}

// DI 設定
const dbRepo = new PrismaUserRepository(prisma);
const cachedRepo = new CachedUserRepository(dbRepo, redis);
container.register<UserRepository>("UserRepository", { useValue: cachedRepo });
```

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| **Repository** | データアクセスの抽象化。ドメイン層にインターフェースを定義し、インフラ層で実装する |
| **DIP** | 上位モジュール（Service）はインターフェース（Repository）に依存。具体実装は注入 |
| **DDD の集約** | Repository は集約ルートごとに1つ。集約内のエンティティは Repository を通じてのみアクセス |
| **テスト** | InMemory 実装でサービス層をユニットテスト。DB 実装は統合テスト（TestContainers） |
| **Unit of Work** | 複数 Repository の変更を1トランザクションで管理。データの一貫性を保証 |
| **Specification** | 検索条件をオブジェクト化。条件の組み合わせ（AND, OR, NOT）を型安全に表現 |
| **注意点** | 過度な抽象化を避け、ドメインの要件に合ったインターフェースを設計。YAGNI を意識する |
| **キャッシュ** | Decorator パターンで Repository をラップ。インターフェースを変更せずにキャッシュを追加 |

---

## 次に読むべきガイド

- [00-mvc-mvvm.md](./00-mvc-mvvm.md) — UI 層のアーキテクチャパターン（Repository を使う側の設計）
- [02-event-sourcing-cqrs.md](./02-event-sourcing-cqrs.md) — イベント駆動とコマンド/クエリ分離（CQRS での Repository）
- [../../clean-code-principles/](../../clean-code-principles/) — SOLID 原則、依存性逆転原則の詳細
- [../02-behavioral/](../02-behavioral/) — Strategy パターン（Repository の実装切り替え）
- [../../system-design-guide/](../../system-design-guide/) — データベーススケーリングとキャッシュ戦略

---

## 参考文献

1. **Martin Fowler** — "Patterns of Enterprise Application Architecture" — Repository パターンの原典 — https://martinfowler.com/eaaCatalog/repository.html
2. **Martin Fowler** — "Unit of Work" — https://martinfowler.com/eaaCatalog/unitOfWork.html
3. **Eric Evans** — "Domain-Driven Design: Tackling Complexity in the Heart of Software" — Addison-Wesley, 2003
4. **Microsoft** — "Implementing the Repository and Unit of Work Patterns" — https://learn.microsoft.com/en-us/aspnet/mvc/overview/older-versions/getting-started-with-ef-5-using-mvc-4/implementing-the-repository-and-unit-of-work-patterns-in-an-asp-net-mvc-application
5. **Robert C. Martin** — "Clean Architecture" (2017) — 依存性逆転原則と Repository の位置付け
6. **Prisma Documentation** — "Repository pattern with Prisma" — https://www.prisma.io/docs/guides
7. **Vaughn Vernon** — "Implementing Domain-Driven Design" (2013) — 集約と Repository の関係
