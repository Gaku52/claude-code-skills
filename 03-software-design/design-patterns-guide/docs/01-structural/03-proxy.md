# Proxy パターン

> 別のオブジェクトへの **アクセスを制御する代理オブジェクト** を提供し、遅延初期化・アクセス制御・キャッシュなどの横断的関心事を実装する構造パターン。

---

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| インタフェースと多態性 | 基本 | TypeScript / Java / Python の OOP |
| 依存性注入 (DI) | 基本 | [クリーンアーキテクチャ](../../../system-design-guide/docs/02-architecture/01-clean-architecture.md) |
| 非同期プログラミング | 基本 | Promise / async-await |
| Decorator パターン | 推奨 | [Decorator パターン](./01-decorator.md) |
| ES6 Proxy / Reflect | 推奨 | MDN Web Docs |

---

## この章で学ぶこと

1. Proxy パターンが解決する「アクセス制御」問題と、その 5 つの分類（Virtual / Protection / Cache / Remote / Logging）
2. GoF Proxy パターンと JavaScript ES6 Proxy の関係および本質的な違い
3. Proxy と Decorator の明確な使い分け基準
4. 5 言語（TypeScript, Python, Java, Go, Kotlin）での実装パターン
5. Proxy チェーン、動的 Proxy 生成、Smart Reference の高度なテクニック

---

## 1. なぜ Proxy が必要なのか（WHY）

### 1.1 Proxy なしの世界

あるオブジェクトを使うとき、以下のような横断的関心事が生じる。

```
┌──────────────────────────────────────────────────────────┐
│  Proxy なし: 横断的関心事がクライアントに散らばる         │
│                                                          │
│  Client A:                                               │
│    if (!user.isAdmin) throw "Access denied";  ← 認証     │
│    const cached = cache.get(key);              ← キャッシュ│
│    if (!cached) {                                        │
│      const data = service.getData();           ← 本来の処理│
│      cache.set(key, data);                     ← キャッシュ│
│    }                                                     │
│    logger.log("getData called");               ← ログ    │
│                                                          │
│  Client B:                                               │
│    if (!user.isAdmin) throw "Access denied";  ← 同じ認証 │
│    const cached = cache.get(key);              ← 同じ     │
│    if (!cached) {                              キャッシュ  │
│      const data = service.getData();                     │
│      cache.set(key, data);                               │
│    }                                                     │
│    logger.log("getData called");               ← 同じログ │
│                                                          │
│  問題:                                                   │
│  - 認証・キャッシュ・ログが全クライアントに重複           │
│  - ビジネスロジックと横断的関心事が混在                   │
│  - テストで横断的関心事を分離できない                     │
│  - 新しい横断的関心事の追加に全クライアントの修正が必要   │
└──────────────────────────────────────────────────────────┘
```

### 1.2 現実世界のアナロジー

**銀行のATM** を考えてみよう。ATM は銀行口座（RealSubject）への代理（Proxy）である。

- 本人確認（暗証番号） = **Protection Proxy**
- 残高キャッシュ（毎回DBに問い合わせない） = **Cache Proxy**
- 取引記録（通帳記帳） = **Logging Proxy**
- リモートの銀行システムへの接続 = **Remote Proxy**

ATM を通じて口座にアクセスするが、ATM と口座は同じ「取引」インタフェースを持つ。クライアント（利用者）は ATM を通じても、窓口で直接でも、同じ操作ができる。

### 1.3 Proxy ありの世界

```
┌──────────────────────────────────────────────────────────┐
│  Proxy あり: 横断的関心事が Proxy に集約                  │
│                                                          │
│  Client A ──▶ Proxy.getData()                            │
│  Client B ──▶ Proxy.getData()                            │
│                    │                                     │
│                    ├── 認証チェック（Protection）         │
│                    ├── キャッシュ確認（Cache）            │
│                    ├── service.getData()（委譲）          │
│                    ├── キャッシュ保存（Cache）            │
│                    └── ログ記録（Logging）                │
│                                                          │
│  利点:                                                   │
│  - 横断的関心事の一元管理                                 │
│  - クライアントは本来の処理にだけ集中                     │
│  - Proxy は RealSubject と同じインタフェース              │
│  - クライアントは Proxy の存在を意識しない（透過的）      │
└──────────────────────────────────────────────────────────┘
```

### 1.4 Proxy パターンの本質

Proxy の本質は「**同じインタフェースを持つ代理オブジェクトを通じてアクセスを制御する**」ことである。

1. **同一インタフェース**: Proxy と RealSubject は同じインタフェースを実装
2. **透過性**: クライアントは Proxy と RealSubject を区別できない
3. **アクセス制御**: 遅延初期化、認証、キャッシュ、ログなどの横断的関心事を挿入
4. **ライフサイクル管理**: Proxy が RealSubject の生成・破棄を管理できる

> **Proxy vs Decorator の本質的な違い**: Proxy は「アクセスの制御」、Decorator は「機能の追加」が目的。Proxy は RealSubject のライフサイクルを管理するが、Decorator は外部から渡されたオブジェクトを装飾する。

---

## 2. Proxy の構造

### 2.1 クラス図

```
┌─────────────────────────────────────────────────────────┐
│                      UML クラス図                        │
│                                                         │
│            ┌──────────────────┐                          │
│            │    <<interface>> │                          │
│            │     Subject     │                          │
│            │                 │                          │
│            │ + request()     │                          │
│            └────────┬────────┘                          │
│                     │ implements                        │
│              ┌──────┴──────┐                            │
│              │             │                            │
│  ┌───────────▼──────┐  ┌──▼──────────────┐             │
│  │      Proxy       │  │  RealSubject    │             │
│  │                  │  │                 │             │
│  │ - real: Subject  │──│ + request()     │             │
│  │ + request()      │  │                 │             │
│  │  {               │  └─────────────────┘             │
│  │   // 前処理      │                                  │
│  │   real.request() │                                  │
│  │   // 後処理      │                                  │
│  │  }               │                                  │
│  └──────────────────┘                                  │
│                                                         │
│  ┌──────────┐                                          │
│  │  Client  │──────▶ Subject (Proxy or RealSubject)    │
│  └──────────┘                                          │
│                                                         │
│  ポイント:                                              │
│  - Client は Subject インタフェースにのみ依存           │
│  - Proxy は RealSubject への参照を内部に持つ            │
│  - Proxy が RealSubject のライフサイクルを管理可能      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Proxy の種類と分類

```
┌──────────────────────────────────────────────────────────┐
│                   Proxy の分類体系                        │
│                                                          │
│  ┌──────────────┬───────────────────────────────────┐   │
│  │ Virtual      │ 重いオブジェクトの遅延初期化       │   │
│  │ Proxy        │ 例: 画像、DB接続、大量データ       │   │
│  │              │ 目的: パフォーマンス最適化          │   │
│  ├──────────────┼───────────────────────────────────┤   │
│  │ Protection   │ アクセス権限のチェック              │   │
│  │ Proxy        │ 例: RBAC、認証・認可ガード         │   │
│  │              │ 目的: セキュリティ                  │   │
│  ├──────────────┼───────────────────────────────────┤   │
│  │ Cache        │ 結果のキャッシュ（メモ化）         │   │
│  │ Proxy        │ 例: API応答、DB結果、計算結果      │   │
│  │              │ 目的: パフォーマンス最適化          │   │
│  ├──────────────┼───────────────────────────────────┤   │
│  │ Remote       │ リモートオブジェクトのローカル代理 │   │
│  │ Proxy        │ 例: RPC, gRPC, GraphQL スタブ      │   │
│  │              │ 目的: 分散システムの透過性          │   │
│  ├──────────────┼───────────────────────────────────┤   │
│  │ Logging      │ 操作の記録・監査・メトリクス       │   │
│  │ Proxy        │ 例: メソッド呼び出しトレース       │   │
│  │              │ 目的: 可観測性                      │   │
│  ├──────────────┼───────────────────────────────────┤   │
│  │ Smart        │ 追加のハウスキーピング処理         │   │
│  │ Reference    │ 例: 参照カウント、変更通知         │   │
│  │              │ 目的: リソース管理                  │   │
│  └──────────────┴───────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 2.3 シーケンス図（Virtual Proxy）

```
┌─────────────────────────────────────────────────────────┐
│                シーケンス図: Virtual Proxy                │
│                                                         │
│  Client        Proxy          RealSubject               │
│    │             │                                      │
│    │ new Proxy() │                                      │
│    │────────────▶│  (RealSubject はまだ生成されない)     │
│    │             │                                      │
│    │ request()   │                                      │
│    │────────────▶│                                      │
│    │             │ [real == null?]                       │
│    │             │── Yes ──▶ new RealSubject()           │
│    │             │           (ここで初めて生成)          │
│    │             │                  │                    │
│    │             │ real.request()   │                    │
│    │             │─────────────────▶│                    │
│    │             │  result          │                    │
│    │             │◀─────────────────│                    │
│    │  result     │                  │                    │
│    │◀────────────│                  │                    │
│    │             │                                      │
│    │ request()   │  (2回目: すでに生成済み)             │
│    │────────────▶│                                      │
│    │             │ [real != null]                        │
│    │             │ real.request()   │                    │
│    │             │─────────────────▶│                    │
│    │             │  result          │                    │
│    │             │◀─────────────────│                    │
│    │  result     │                  │                    │
│    │◀────────────│                  │                    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. コード例

### コード例 1: Virtual Proxy -- 遅延初期化（TypeScript）

```typescript
// === Subject インタフェース ===

interface Image {
  display(): void;
  getSize(): number;
  getFilename(): string;
}

// === RealSubject ===

class HighResImage implements Image {
  private data: Uint8Array;

  constructor(private filename: string) {
    // 重い処理: ファイルを読み込む（コンストラクタで実行）
    console.log(`[HighResImage] Loading ${filename} from disk...`);
    console.log(`[HighResImage] Decoding image data...`);
    this.data = new Uint8Array(10_000_000); // 10MB のバッファ
    console.log(`[HighResImage] ${filename} loaded (${this.data.length} bytes)`);
  }

  display(): void {
    console.log(`[HighResImage] Displaying ${this.filename} (${this.data.length} bytes)`);
  }

  getSize(): number {
    return this.data.length;
  }

  getFilename(): string {
    return this.filename;
  }
}

// === Proxy ===

class ImageProxy implements Image {
  private real: HighResImage | null = null;

  constructor(private filename: string) {
    // Proxy の生成は軽量（ファイル読み込みは行わない）
    console.log(`[ImageProxy] Created proxy for ${filename}`);
  }

  private ensureLoaded(): HighResImage {
    if (!this.real) {
      console.log(`[ImageProxy] First access - loading real image...`);
      this.real = new HighResImage(this.filename);
    }
    return this.real;
  }

  display(): void {
    this.ensureLoaded().display();
  }

  getSize(): number {
    return this.ensureLoaded().getSize();
  }

  getFilename(): string {
    // ファイル名はProxyが知っているので、RealSubjectを生成する必要がない
    return this.filename;
  }
}

// === 使用例 ===

console.log("=== Creating image gallery ===");
const gallery: Image[] = [
  new ImageProxy("photo1.jpg"),
  new ImageProxy("photo2.jpg"),
  new ImageProxy("photo3.jpg"),
  new ImageProxy("photo4.jpg"),
  new ImageProxy("photo5.jpg"),
];
console.log(`Gallery has ${gallery.length} images`);
// 出力: 5 つの Proxy が作られるだけ。画像データは未ロード。

console.log("\n=== User scrolls to image 2 ===");
gallery[1].display();
// 出力: photo2.jpg のみロードされる

console.log("\n=== User scrolls to image 2 again ===");
gallery[1].display();
// 出力: すでにロード済みなので即座に表示

console.log("\n=== Getting filename (no loading needed) ===");
console.log(gallery[3].getFilename());
// 出力: photo4.jpg（ロードなしで返せる）
```

**ポイント**: 5 枚の画像のうち、実際にロードされるのはユーザーが表示した画像だけ。`getFilename()` はロードなしで返せるメソッドの例。

---

### コード例 2: Protection Proxy -- RBAC アクセス制御（TypeScript）

```typescript
// === Subject インタフェース ===

interface AdminService {
  listUsers(): User[];
  deleteUser(userId: string): void;
  resetDatabase(): void;
  viewAuditLogs(): AuditEntry[];
}

interface User {
  id: string;
  name: string;
  role: "viewer" | "editor" | "admin" | "superadmin";
}

interface AuditEntry {
  action: string;
  userId: string;
  timestamp: Date;
}

// === RealSubject ===

class RealAdminService implements AdminService {
  listUsers(): User[] {
    console.log("[RealAdmin] Listing all users");
    return [
      { id: "1", name: "Alice", role: "admin" },
      { id: "2", name: "Bob", role: "viewer" },
    ];
  }

  deleteUser(userId: string): void {
    console.log(`[RealAdmin] User ${userId} deleted`);
  }

  resetDatabase(): void {
    console.log("[RealAdmin] Database has been reset");
  }

  viewAuditLogs(): AuditEntry[] {
    console.log("[RealAdmin] Fetching audit logs");
    return [{ action: "login", userId: "1", timestamp: new Date() }];
  }
}

// === Protection Proxy ===

type Role = User["role"];

// メソッドごとの必要ロールを定義
const REQUIRED_ROLES: Record<keyof AdminService, Role[]> = {
  listUsers: ["viewer", "editor", "admin", "superadmin"],
  deleteUser: ["admin", "superadmin"],
  resetDatabase: ["superadmin"],
  viewAuditLogs: ["admin", "superadmin"],
};

class AdminProxy implements AdminService {
  constructor(
    private real: AdminService,
    private currentUser: User,
  ) {
    console.log(`[AdminProxy] Created for user ${currentUser.name} (${currentUser.role})`);
  }

  private checkAccess(method: keyof AdminService): void {
    const allowedRoles = REQUIRED_ROLES[method];
    if (!allowedRoles.includes(this.currentUser.role)) {
      const msg = `Access denied: ${method} requires ${allowedRoles.join(" or ")}, ` +
        `but ${this.currentUser.name} has role "${this.currentUser.role}"`;
      console.log(`[AdminProxy] ${msg}`);
      throw new Error(msg);
    }
    console.log(`[AdminProxy] Access granted: ${this.currentUser.name} -> ${method}`);
  }

  listUsers(): User[] {
    this.checkAccess("listUsers");
    return this.real.listUsers();
  }

  deleteUser(userId: string): void {
    this.checkAccess("deleteUser");
    // 追加チェック: 自分自身は削除できない
    if (userId === this.currentUser.id) {
      throw new Error("Cannot delete yourself");
    }
    this.real.deleteUser(userId);
  }

  resetDatabase(): void {
    this.checkAccess("resetDatabase");
    this.real.resetDatabase();
  }

  viewAuditLogs(): AuditEntry[] {
    this.checkAccess("viewAuditLogs");
    return this.real.viewAuditLogs();
  }
}

// === 使用例 ===

const realService = new RealAdminService();

// Admin ユーザー
const admin: User = { id: "1", name: "Alice", role: "admin" };
const adminProxy = new AdminProxy(realService, admin);
adminProxy.listUsers();       // OK
adminProxy.deleteUser("2");   // OK
// adminProxy.resetDatabase(); // Error: requires superadmin

// Viewer ユーザー
const viewer: User = { id: "2", name: "Bob", role: "viewer" };
const viewerProxy = new AdminProxy(realService, viewer);
viewerProxy.listUsers();      // OK
// viewerProxy.deleteUser("1"); // Error: requires admin or superadmin
```

**ポイント**: `REQUIRED_ROLES` テーブルでメソッドごとの必要ロールを宣言的に定義。新しいメソッド追加時もテーブルに 1 行追加するだけ。

---

### コード例 3: Cache Proxy -- TTL + LRU キャッシュ（TypeScript）

```typescript
// === Subject インタフェース ===

interface ApiClient {
  fetchUser(id: string): Promise<User>;
  fetchPosts(userId: string): Promise<Post[]>;
}

interface User {
  id: string;
  name: string;
  email: string;
}

interface Post {
  id: string;
  title: string;
  content: string;
}

// === RealSubject ===

class RealApiClient implements ApiClient {
  async fetchUser(id: string): Promise<User> {
    console.log(`[API] GET /users/${id} (network request)`);
    // 実際はHTTPリクエスト
    await new Promise(r => setTimeout(r, 100));
    return { id, name: `User-${id}`, email: `user${id}@example.com` };
  }

  async fetchPosts(userId: string): Promise<Post[]> {
    console.log(`[API] GET /users/${userId}/posts (network request)`);
    await new Promise(r => setTimeout(r, 200));
    return [
      { id: "p1", title: "Hello", content: "World" },
    ];
  }
}

// === LRU キャッシュ実装 ===

class LRUCache<T> {
  private cache = new Map<string, { data: T; expiry: number }>();

  constructor(
    private maxSize: number,
    private ttlMs: number,
  ) {}

  get(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;

    if (Date.now() > entry.expiry) {
      this.cache.delete(key);
      console.log(`[Cache] EXPIRED: ${key}`);
      return undefined;
    }

    // LRU: アクセスされたエントリを末尾に移動
    this.cache.delete(key);
    this.cache.set(key, entry);
    console.log(`[Cache] HIT: ${key}`);
    return entry.data;
  }

  set(key: string, data: T): void {
    // 容量超過時は最も古いエントリを削除
    if (this.cache.size >= this.maxSize) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
        console.log(`[Cache] EVICTED: ${oldestKey}`);
      }
    }

    this.cache.set(key, { data, expiry: Date.now() + this.ttlMs });
    console.log(`[Cache] STORED: ${key} (TTL: ${this.ttlMs}ms)`);
  }

  invalidate(key: string): void {
    this.cache.delete(key);
    console.log(`[Cache] INVALIDATED: ${key}`);
  }

  clear(): void {
    this.cache.clear();
    console.log(`[Cache] CLEARED`);
  }

  get size(): number {
    return this.cache.size;
  }
}

// === Cache Proxy ===

class CachingApiProxy implements ApiClient {
  private cache: LRUCache<unknown>;

  constructor(
    private real: ApiClient,
    maxSize = 100,
    ttlMs = 60_000,
  ) {
    this.cache = new LRUCache(maxSize, ttlMs);
  }

  async fetchUser(id: string): Promise<User> {
    const key = `user:${id}`;
    const cached = this.cache.get(key) as User | undefined;
    if (cached) return cached;

    const user = await this.real.fetchUser(id);
    this.cache.set(key, user);
    return user;
  }

  async fetchPosts(userId: string): Promise<Post[]> {
    const key = `posts:${userId}`;
    const cached = this.cache.get(key) as Post[] | undefined;
    if (cached) return cached;

    const posts = await this.real.fetchPosts(userId);
    this.cache.set(key, posts);
    return posts;
  }

  /** キャッシュを手動で無効化 */
  invalidateUser(id: string): void {
    this.cache.invalidate(`user:${id}`);
  }
}

// === 使用例 ===

const api: ApiClient = new CachingApiProxy(new RealApiClient(), 50, 30_000);

// 1回目: キャッシュなし → ネットワークリクエスト
const user1 = await api.fetchUser("42");
// 出力: [Cache] key not found
// 出力: [API] GET /users/42 (network request)
// 出力: [Cache] STORED: user:42

// 2回目: キャッシュヒット → ネットワークリクエストなし
const user2 = await api.fetchUser("42");
// 出力: [Cache] HIT: user:42
```

**ポイント**: LRU（Least Recently Used）とTTL（Time To Live）を組み合わせたキャッシュ Proxy。`invalidateUser` で明示的なキャッシュ無効化も可能。

---

### コード例 4: ES6 Proxy -- メタプログラミング（TypeScript）

```typescript
// === ES6 Proxy を使ったバリデーション ===

interface UserData {
  name: string;
  age: number;
  email: string;
}

type ValidationRule = {
  validate: (value: unknown) => boolean;
  message: string;
};

const RULES: Partial<Record<keyof UserData, ValidationRule[]>> = {
  name: [
    { validate: (v) => typeof v === "string", message: "name must be a string" },
    { validate: (v) => (v as string).length >= 2, message: "name must be at least 2 chars" },
    { validate: (v) => (v as string).length <= 50, message: "name must be at most 50 chars" },
  ],
  age: [
    { validate: (v) => typeof v === "number", message: "age must be a number" },
    { validate: (v) => Number.isInteger(v), message: "age must be an integer" },
    { validate: (v) => (v as number) >= 0, message: "age must be non-negative" },
    { validate: (v) => (v as number) <= 150, message: "age must be at most 150" },
  ],
  email: [
    { validate: (v) => typeof v === "string", message: "email must be a string" },
    { validate: (v) => (v as string).includes("@"), message: "email must contain @" },
    { validate: (v) => (v as string).includes("."), message: "email must contain ." },
  ],
};

function createValidatedUser(initial: UserData): UserData {
  return new Proxy(initial, {
    set(target, prop: string, value: unknown): boolean {
      const rules = RULES[prop as keyof UserData];
      if (rules) {
        for (const rule of rules) {
          if (!rule.validate(value)) {
            throw new Error(`Validation failed: ${rule.message} (got: ${value})`);
          }
        }
      }
      console.log(`[Proxy] Set ${prop} = ${value}`);
      (target as Record<string, unknown>)[prop] = value;
      return true;
    },

    get(target, prop: string): unknown {
      console.log(`[Proxy] Get ${prop}`);
      return (target as Record<string, unknown>)[prop];
    },

    deleteProperty(target, prop: string): boolean {
      throw new Error(`Cannot delete property: ${prop}`);
    },
  });
}

// === 使用例 ===

const user = createValidatedUser({ name: "Taro", age: 30, email: "taro@example.com" });

user.name = "Hanako";   // OK
user.age = 25;           // OK
// user.age = -1;        // Error: age must be non-negative
// user.email = "invalid"; // Error: email must contain @
// delete user.name;     // Error: Cannot delete property: name

console.log(user.name);  // "Hanako"
```

---

### コード例 5: ES6 Proxy -- リアクティブ変更検知（TypeScript）

```typescript
// === Vue.js 風のリアクティブシステム ===

type Listener = () => void;

function reactive<T extends object>(target: T, onChange: Listener): T {
  const handler: ProxyHandler<T> = {
    set(obj: T, prop: string | symbol, value: unknown): boolean {
      const oldValue = (obj as Record<string | symbol, unknown>)[prop];
      if (oldValue !== value) {
        (obj as Record<string | symbol, unknown>)[prop] = value;
        console.log(`[Reactive] ${String(prop)}: ${String(oldValue)} -> ${String(value)}`);
        onChange();
      }
      return true;
    },

    get(obj: T, prop: string | symbol): unknown {
      const value = (obj as Record<string | symbol, unknown>)[prop];
      // ネストされたオブジェクトも再帰的にリアクティブ化
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return reactive(value as object, onChange);
      }
      return value;
    },
  };

  return new Proxy(target, handler);
}

// === 使用例 ===

interface AppState {
  count: number;
  user: {
    name: string;
    settings: {
      theme: string;
    };
  };
}

let renderCount = 0;
const state = reactive<AppState>(
  {
    count: 0,
    user: {
      name: "Taro",
      settings: { theme: "light" },
    },
  },
  () => {
    renderCount++;
    console.log(`[App] Re-render #${renderCount}`);
  },
);

state.count = 1;
// 出力: [Reactive] count: 0 -> 1
// 出力: [App] Re-render #1

state.user.name = "Hanako";
// 出力: [Reactive] name: Taro -> Hanako
// 出力: [App] Re-render #2

state.user.settings.theme = "dark";
// 出力: [Reactive] theme: light -> dark
// 出力: [App] Re-render #3

state.count = 1; // 同じ値 → 変更なし → 再レンダリングなし
```

**ポイント**: Vue.js 3 のリアクティブシステムはこの ES6 Proxy ベースの設計。プロパティアクセスをトラップしてネストされたオブジェクトも再帰的にリアクティブ化する。

---

### コード例 6: Logging Proxy -- メソッド呼び出しの自動トレース（TypeScript）

```typescript
// === 汎用 Logging Proxy ファクトリ ===

interface LogEntry {
  method: string;
  args: unknown[];
  result?: unknown;
  error?: string;
  durationMs: number;
  timestamp: Date;
}

function createLoggingProxy<T extends object>(
  target: T,
  options: {
    name?: string;
    logArgs?: boolean;
    logResult?: boolean;
    onLog?: (entry: LogEntry) => void;
  } = {},
): T {
  const {
    name = target.constructor.name,
    logArgs = true,
    logResult = false,
    onLog = (entry) => {
      const argsStr = logArgs ? `(${entry.args.map(a => JSON.stringify(a)).join(", ")})` : "(...)";
      const resultStr = logResult && entry.result !== undefined ? ` -> ${JSON.stringify(entry.result)}` : "";
      const errorStr = entry.error ? ` ERROR: ${entry.error}` : "";
      console.log(
        `[${name}] ${entry.method}${argsStr}${resultStr}${errorStr} [${entry.durationMs}ms]`
      );
    },
  } = options;

  return new Proxy(target, {
    get(obj, prop) {
      const value = (obj as Record<string | symbol, unknown>)[prop];
      if (typeof value !== "function") return value;

      return function (this: unknown, ...args: unknown[]) {
        const start = performance.now();
        const entry: LogEntry = {
          method: String(prop),
          args,
          durationMs: 0,
          timestamp: new Date(),
        };

        try {
          const result = (value as Function).apply(obj, args);

          // Promise を検知して非同期もトレース
          if (result instanceof Promise) {
            return result
              .then((resolved: unknown) => {
                entry.result = resolved;
                entry.durationMs = Math.round(performance.now() - start);
                onLog(entry);
                return resolved;
              })
              .catch((err: Error) => {
                entry.error = err.message;
                entry.durationMs = Math.round(performance.now() - start);
                onLog(entry);
                throw err;
              });
          }

          entry.result = result;
          entry.durationMs = Math.round(performance.now() - start);
          onLog(entry);
          return result;
        } catch (err) {
          entry.error = (err as Error).message;
          entry.durationMs = Math.round(performance.now() - start);
          onLog(entry);
          throw err;
        }
      };
    },
  });
}

// === 使用例 ===

class UserRepository {
  findById(id: string): User | null {
    return { id, name: `User-${id}`, email: `${id}@example.com` };
  }

  async save(user: User): Promise<void> {
    await new Promise(r => setTimeout(r, 50));
  }

  delete(id: string): void {
    throw new Error("Not implemented");
  }
}

const repo = createLoggingProxy(new UserRepository(), {
  name: "UserRepo",
  logResult: true,
});

repo.findById("42");
// 出力: [UserRepo] findById("42") -> {"id":"42","name":"User-42",...} [0ms]

await repo.save({ id: "1", name: "New", email: "new@example.com" });
// 出力: [UserRepo] save({"id":"1",...}) [51ms]

try {
  repo.delete("1");
} catch (e) {
  // 出力: [UserRepo] delete("1") ERROR: Not implemented [0ms]
}
```

**ポイント**: ES6 Proxy を使って、任意のオブジェクトに対して自動的にログ Proxy を生成。同期・非同期メソッドの両方に対応し、実行時間も計測する。

---

### コード例 7: Python -- 動的 Proxy（`__getattr__`）

```python
from abc import ABC, abstractmethod
import time
import functools
from typing import Any, Callable


# === Subject ===

class Database(ABC):
    @abstractmethod
    def query(self, sql: str) -> list[dict]: ...

    @abstractmethod
    def execute(self, sql: str) -> int: ...


class PostgresDatabase(Database):
    def query(self, sql: str) -> list[dict]:
        time.sleep(0.05)  # ネットワーク遅延のシミュレーション
        print(f"[Postgres] Executing query: {sql}")
        return [{"id": 1, "name": "Taro"}]

    def execute(self, sql: str) -> int:
        time.sleep(0.05)
        print(f"[Postgres] Executing: {sql}")
        return 1  # affected rows


# === Logging Proxy ===

class LoggingProxy:
    """__getattr__ を使った汎用 Logging Proxy"""

    def __init__(self, target: Any, name: str = ""):
        self._target = target
        self._name = name or type(target).__name__

    def __getattr__(self, attr: str) -> Any:
        original = getattr(self._target, attr)

        if not callable(original):
            return original

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            args_str = ", ".join(
                [repr(a) for a in args] +
                [f"{k}={v!r}" for k, v in kwargs.items()]
            )
            print(f"[{self._name}] {attr}({args_str})")

            start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                print(f"[{self._name}] {attr} -> {result!r} [{elapsed:.1f}ms]")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                print(f"[{self._name}] {attr} ERROR: {e} [{elapsed:.1f}ms]")
                raise

        return wrapper


# === Cache Proxy ===

class CachingProxy:
    """TTL 付きキャッシュ Proxy"""

    def __init__(self, target: Any, ttl: float = 60.0):
        self._target = target
        self._ttl = ttl
        self._cache: dict[str, tuple[Any, float]] = {}

    def __getattr__(self, attr: str) -> Any:
        original = getattr(self._target, attr)

        if not callable(original):
            return original

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = f"{attr}:{args!r}:{kwargs!r}"

            # キャッシュチェック
            if key in self._cache:
                result, expiry = self._cache[key]
                if time.time() < expiry:
                    print(f"[Cache] HIT: {key[:50]}")
                    return result
                else:
                    del self._cache[key]
                    print(f"[Cache] EXPIRED: {key[:50]}")

            print(f"[Cache] MISS: {key[:50]}")
            result = original(*args, **kwargs)
            self._cache[key] = (result, time.time() + self._ttl)
            return result

        return wrapper


# === Proxy チェーン ===

db: Database = PostgresDatabase()
db = CachingProxy(db, ttl=30.0)   # キャッシュ層
db = LoggingProxy(db, "DB")        # ログ層

# 1回目: キャッシュミス → DB アクセス
result = db.query("SELECT * FROM users")
# 出力:
# [DB] query('SELECT * FROM users')
# [Cache] MISS: query:('SELECT * FROM users',):{}
# [Postgres] Executing query: SELECT * FROM users
# [DB] query -> [{'id': 1, 'name': 'Taro'}] [52.3ms]

# 2回目: キャッシュヒット
result = db.query("SELECT * FROM users")
# 出力:
# [DB] query('SELECT * FROM users')
# [Cache] HIT: query:('SELECT * FROM users',):{}
# [DB] query -> [{'id': 1, 'name': 'Taro'}] [0.1ms]
```

**ポイント**: Python の `__getattr__` を使うと、GoF Proxy パターンをインタフェースなしで実装できる。CachingProxy と LoggingProxy のチェーンで、Decorator パターン的な合成も可能。

---

### コード例 8: Java -- 動的 Proxy（java.lang.reflect.Proxy）

```java
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.*;

// === Subject インタフェース ===

interface UserService {
    User findById(String id);
    List<User> findAll();
    void delete(String id);
}

record User(String id, String name, String email) {}

// === RealSubject ===

class UserServiceImpl implements UserService {
    @Override
    public User findById(String id) {
        System.out.println("[UserService] findById: " + id);
        return new User(id, "User-" + id, id + "@example.com");
    }

    @Override
    public List<User> findAll() {
        System.out.println("[UserService] findAll");
        return List.of(
            new User("1", "Alice", "alice@example.com"),
            new User("2", "Bob", "bob@example.com")
        );
    }

    @Override
    public void delete(String id) {
        System.out.println("[UserService] delete: " + id);
    }
}

// === 動的 Proxy (InvocationHandler) ===

class LoggingHandler implements InvocationHandler {
    private final Object target;

    LoggingHandler(Object target) {
        this.target = target;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        String argsStr = args != null ? Arrays.toString(args) : "()";
        System.out.printf("[Log] %s.%s%s%n",
            target.getClass().getSimpleName(), method.getName(), argsStr);

        long start = System.nanoTime();
        try {
            Object result = method.invoke(target, args);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            System.out.printf("[Log] %s returned in %dms%n", method.getName(), elapsed);
            return result;
        } catch (Exception e) {
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            System.out.printf("[Log] %s FAILED in %dms: %s%n",
                method.getName(), elapsed, e.getMessage());
            throw e.getCause();
        }
    }
}

class CachingHandler implements InvocationHandler {
    private final Object target;
    private final Map<String, Object> cache = new HashMap<>();

    CachingHandler(Object target) {
        this.target = target;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        // void メソッドはキャッシュしない
        if (method.getReturnType() == void.class) {
            return method.invoke(target, args);
        }

        String key = method.getName() + ":" + Arrays.toString(args);
        if (cache.containsKey(key)) {
            System.out.println("[Cache] HIT: " + key);
            return cache.get(key);
        }

        System.out.println("[Cache] MISS: " + key);
        Object result = method.invoke(target, args);
        cache.put(key, result);
        return result;
    }
}

// === 動的 Proxy ファクトリ ===

class ProxyFactory {
    @SuppressWarnings("unchecked")
    static <T> T createLoggingProxy(T target, Class<T> iface) {
        return (T) Proxy.newProxyInstance(
            iface.getClassLoader(),
            new Class[]{iface},
            new LoggingHandler(target)
        );
    }

    @SuppressWarnings("unchecked")
    static <T> T createCachingProxy(T target, Class<T> iface) {
        return (T) Proxy.newProxyInstance(
            iface.getClassLoader(),
            new Class[]{iface},
            new CachingHandler(target)
        );
    }
}

// === 使用例 ===

public class Main {
    public static void main(String[] args) {
        UserService real = new UserServiceImpl();

        // キャッシュ + ログの Proxy チェーン
        UserService cached = ProxyFactory.createCachingProxy(real, UserService.class);
        UserService logged = ProxyFactory.createLoggingProxy(cached, UserService.class);

        logged.findById("42");
        // [Log] $Proxy.findById[42]
        // [Cache] MISS: findById:[42]
        // [UserService] findById: 42
        // [Log] findById returned in 1ms

        logged.findById("42");
        // [Log] $Proxy.findById[42]
        // [Cache] HIT: findById:[42]
        // [Log] findById returned in 0ms
    }
}
```

**ポイント**: Java の `java.lang.reflect.Proxy` を使うと、インタフェースに対して動的に Proxy を生成できる。Spring AOP やHibernate の遅延ロードも同じ仕組み。

---

### コード例 9: Go -- インタフェースベース Proxy

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// === Subject インタフェース ===

type Storage interface {
    Get(ctx context.Context, key string) (string, error)
    Set(ctx context.Context, key string, value string) error
    Delete(ctx context.Context, key string) error
}

// === RealSubject ===

type RedisStorage struct {
    data map[string]string
    mu   sync.RWMutex
}

func NewRedisStorage() *RedisStorage {
    return &RedisStorage{data: make(map[string]string)}
}

func (r *RedisStorage) Get(ctx context.Context, key string) (string, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    time.Sleep(10 * time.Millisecond) // ネットワーク遅延
    val, ok := r.data[key]
    if !ok {
        return "", fmt.Errorf("key not found: %s", key)
    }
    fmt.Printf("[Redis] GET %s -> %s\n", key, val)
    return val, nil
}

func (r *RedisStorage) Set(ctx context.Context, key string, value string) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    time.Sleep(10 * time.Millisecond)
    r.data[key] = value
    fmt.Printf("[Redis] SET %s = %s\n", key, value)
    return nil
}

func (r *RedisStorage) Delete(ctx context.Context, key string) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    delete(r.data, key)
    fmt.Printf("[Redis] DEL %s\n", key)
    return nil
}

// === Logging Proxy ===

type LoggingStorage struct {
    inner Storage
}

func NewLoggingStorage(inner Storage) *LoggingStorage {
    return &LoggingStorage{inner: inner}
}

func (l *LoggingStorage) Get(ctx context.Context, key string) (string, error) {
    start := time.Now()
    val, err := l.inner.Get(ctx, key)
    elapsed := time.Since(start)
    if err != nil {
        fmt.Printf("[Log] GET %s ERROR: %v [%v]\n", key, err, elapsed)
    } else {
        fmt.Printf("[Log] GET %s -> %s [%v]\n", key, val, elapsed)
    }
    return val, err
}

func (l *LoggingStorage) Set(ctx context.Context, key string, value string) error {
    start := time.Now()
    err := l.inner.Set(ctx, key, value)
    elapsed := time.Since(start)
    fmt.Printf("[Log] SET %s = %s [%v]\n", key, value, elapsed)
    return err
}

func (l *LoggingStorage) Delete(ctx context.Context, key string) error {
    start := time.Now()
    err := l.inner.Delete(ctx, key)
    elapsed := time.Since(start)
    fmt.Printf("[Log] DEL %s [%v]\n", key, elapsed)
    return err
}

// === Circuit Breaker Proxy ===

type CircuitState int

const (
    Closed CircuitState = iota
    Open
    HalfOpen
)

type CircuitBreakerStorage struct {
    inner        Storage
    state        CircuitState
    failures     int
    threshold    int
    resetTimeout time.Duration
    lastFailure  time.Time
    mu           sync.Mutex
}

func NewCircuitBreakerStorage(inner Storage, threshold int, resetTimeout time.Duration) *CircuitBreakerStorage {
    return &CircuitBreakerStorage{
        inner:        inner,
        state:        Closed,
        threshold:    threshold,
        resetTimeout: resetTimeout,
    }
}

func (cb *CircuitBreakerStorage) canExecute() bool {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    switch cb.state {
    case Closed:
        return true
    case Open:
        if time.Since(cb.lastFailure) > cb.resetTimeout {
            cb.state = HalfOpen
            fmt.Println("[CB] State: Open -> HalfOpen")
            return true
        }
        return false
    case HalfOpen:
        return true
    }
    return false
}

func (cb *CircuitBreakerStorage) recordSuccess() {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    cb.failures = 0
    if cb.state == HalfOpen {
        cb.state = Closed
        fmt.Println("[CB] State: HalfOpen -> Closed")
    }
}

func (cb *CircuitBreakerStorage) recordFailure() {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    cb.failures++
    cb.lastFailure = time.Now()
    if cb.failures >= cb.threshold {
        cb.state = Open
        fmt.Printf("[CB] State: -> Open (failures: %d)\n", cb.failures)
    }
}

func (cb *CircuitBreakerStorage) Get(ctx context.Context, key string) (string, error) {
    if !cb.canExecute() {
        return "", fmt.Errorf("circuit breaker is OPEN")
    }
    val, err := cb.inner.Get(ctx, key)
    if err != nil {
        cb.recordFailure()
    } else {
        cb.recordSuccess()
    }
    return val, err
}

func (cb *CircuitBreakerStorage) Set(ctx context.Context, key string, value string) error {
    if !cb.canExecute() {
        return fmt.Errorf("circuit breaker is OPEN")
    }
    err := cb.inner.Set(ctx, key, value)
    if err != nil {
        cb.recordFailure()
    } else {
        cb.recordSuccess()
    }
    return err
}

func (cb *CircuitBreakerStorage) Delete(ctx context.Context, key string) error {
    if !cb.canExecute() {
        return fmt.Errorf("circuit breaker is OPEN")
    }
    err := cb.inner.Delete(ctx, key)
    if err != nil {
        cb.recordFailure()
    } else {
        cb.recordSuccess()
    }
    return err
}

// === 使用例 ===

func main() {
    ctx := context.Background()

    // Proxy チェーン: Redis -> CircuitBreaker -> Logging
    var storage Storage = NewRedisStorage()
    storage = NewCircuitBreakerStorage(storage, 3, 30*time.Second)
    storage = NewLoggingStorage(storage)

    storage.Set(ctx, "user:1", "Alice")
    storage.Get(ctx, "user:1")
}
```

**ポイント**: Go のインタフェースは暗黙的実装のため、Proxy パターンが自然に実装できる。CircuitBreaker を Proxy として実装し、Logging Proxy とチェーンしている。

---

### コード例 10: Kotlin -- Property Delegation as Proxy

```kotlin
import kotlin.properties.ReadWriteProperty
import kotlin.reflect.KProperty

// === Kotlin の Property Delegation は Proxy パターン ===

class LoggedProperty<T>(
    private var value: T,
    private val name: String,
) : ReadWriteProperty<Any?, T> {

    override fun getValue(thisRef: Any?, property: KProperty<*>): T {
        println("[Log] Reading $name = $value")
        return value
    }

    override fun setValue(thisRef: Any?, property: KProperty<*>, value: T) {
        val old = this.value
        this.value = value
        println("[Log] Writing $name: $old -> $value")
    }
}

class ValidatedProperty<T>(
    private var value: T,
    private val validator: (T) -> Boolean,
    private val errorMessage: String,
) : ReadWriteProperty<Any?, T> {

    override fun getValue(thisRef: Any?, property: KProperty<*>): T = value

    override fun setValue(thisRef: Any?, property: KProperty<*>, value: T) {
        if (!validator(value)) {
            throw IllegalArgumentException("$errorMessage (got: $value)")
        }
        this.value = value
    }
}

// === LazyProxy: Virtual Proxy as Property Delegate ===

class LazyProxy<T>(
    private val factory: () -> T,
) : ReadWriteProperty<Any?, T> {
    private var instance: T? = null
    private var overridden = false
    private var overriddenValue: T? = null

    override fun getValue(thisRef: Any?, property: KProperty<*>): T {
        if (overridden) {
            @Suppress("UNCHECKED_CAST")
            return overriddenValue as T
        }
        if (instance == null) {
            println("[LazyProxy] Creating ${property.name}")
            instance = factory()
        }
        return instance!!
    }

    override fun setValue(thisRef: Any?, property: KProperty<*>, value: T) {
        overridden = true
        overriddenValue = value
    }
}

// === 使用例 ===

class UserProfile {
    var name: String by LoggedProperty("", "name")
    var age: Int by ValidatedProperty(0, { it in 0..150 }, "Age must be 0-150")
    var email: String by ValidatedProperty("", { "@" in it }, "Invalid email")

    // 重いリソースの遅延読み込み
    var avatar: ByteArray by LazyProxy {
        println("[LazyProxy] Loading avatar from storage...")
        ByteArray(1_000_000) // 1MB
    }
}

fun main() {
    val profile = UserProfile()

    profile.name = "Taro"
    // [Log] Writing name:  -> Taro

    println(profile.name)
    // [Log] Reading name = Taro
    // Taro

    profile.age = 25  // OK
    // profile.age = -1 // IllegalArgumentException

    profile.email = "taro@example.com" // OK
    // profile.email = "invalid" // IllegalArgumentException

    // avatar はアクセスするまでロードされない
    println("Profile created, avatar not loaded yet")
    val size = profile.avatar.size
    // [LazyProxy] Creating avatar
    // [LazyProxy] Loading avatar from storage...
    println("Avatar size: $size")
}
```

**ポイント**: Kotlin の `by` キーワード（Property Delegation）は Proxy パターンのネイティブサポート。`LoggedProperty`, `ValidatedProperty`, `LazyProxy` はすべて Proxy として機能する。

---

## 4. 比較表

### 比較表 1: Proxy vs Decorator vs Adapter

| 観点 | Proxy | Decorator | Adapter |
|------|-------|-----------|---------|
| **目的** | アクセス制御 | 機能追加 | インタフェース変換 |
| **インタフェース** | 同一 | 同一 | 異なる |
| **ライフサイクル管理** | する（遅延生成等） | しない | しない |
| **クライアントの認識** | 透過的 | 透過的 | 意図的 |
| **典型的な責務** | 認証、キャッシュ、遅延 | ログ、圧縮、暗号化 | API 変換 |
| **RealSubject の生成** | Proxy が管理可能 | 外部から渡される | 外部から渡される |
| **構造** | 1:1 | N:1（スタック） | 1:1 |

### 比較表 2: Proxy の種類と適用場面

| 種類 | ユースケース | パフォーマンス影響 | 複雑度 | 実装コスト |
|------|-------------|:-:|:-:|:-:|
| **Virtual** | 重いリソースの遅延読み込み | 初回のみ遅延 | 低 | 低 |
| **Protection** | RBAC / 認証・認可ガード | 微小 | 中 | 中 |
| **Cache** | API/DB 結果のメモ化 | 大幅に改善 | 高 | 高 |
| **Remote** | RPC/gRPC 呼び出し | ネットワーク依存 | 高 | 高 |
| **Logging** | 監査ログ・メトリクス | 微小 | 低 | 低 |
| **Smart Ref** | 参照カウント・変更通知 | 微小 | 中 | 中 |

### 比較表 3: GoF Proxy vs ES6 Proxy

| 観点 | GoF Proxy（クラスベース） | ES6 Proxy（メタプログラミング） |
|------|--------------------------|-------------------------------|
| **実装方式** | 同一インタフェースを実装 | Handler オブジェクトでトラップ |
| **対象** | メソッド呼び出し | プロパティアクセス、代入、削除、列挙等 |
| **型安全性** | 高い（インタフェース準拠） | 低い（any に近い） |
| **トラップ対象** | メソッドのみ | 13 種のトラップ（get, set, has, etc.） |
| **用途** | 設計パターン | メタプログラミング、リアクティブ |
| **パフォーマンス** | 良好 | プロパティアクセスごとにオーバーヘッド |
| **例** | `ImageProxy implements Image` | `new Proxy(target, handler)` |

---

## 5. アンチパターン

### アンチパターン 1: Proxy と RealSubject の密結合

```typescript
// NG: Proxy が具象クラスに直接依存し、new で生成
class BadProxy {
  private real = new SpecificDatabaseService("localhost", 5432); // 直接 new

  query(sql: string): Result {
    console.log("Logging:", sql);
    return this.real.query(sql); // インタフェースなし
  }
}

// 問題:
// 1. SpecificDatabaseService を差し替えられない
// 2. テストでモックに差し替えられない
// 3. Proxy が RealSubject のコンストラクタ引数を知る必要がある
```

```typescript
// OK: インタフェース経由で依存し、DI で注入

interface DatabaseService {
  query(sql: string): Result;
}

class LoggingProxy implements DatabaseService {
  constructor(private real: DatabaseService) {} // DI

  query(sql: string): Result {
    console.log("Logging:", sql);
    return this.real.query(sql);
  }
}

// テスト
const mockDb: DatabaseService = { query: jest.fn() };
const proxy = new LoggingProxy(mockDb);
```

**判断基準**: Proxy 内で `new RealSubject()` があったら密結合の警告サイン（Virtual Proxy は例外）。

---

### アンチパターン 2: God Proxy（1つの Proxy に複数の責務）

```typescript
// NG: キャッシュ + 認証 + ログ + レート制限を 1 つの Proxy で
class GodProxy implements Service {
  private cache = new Map();
  private requestCount = 0;

  doSomething(args: unknown): Result {
    // 認証チェック
    if (!this.currentUser.isAdmin) throw new Error("Access denied");

    // レート制限
    this.requestCount++;
    if (this.requestCount > 100) throw new Error("Rate limited");

    // キャッシュ
    const key = JSON.stringify(args);
    if (this.cache.has(key)) return this.cache.get(key);

    // ログ
    console.log(`[${new Date().toISOString()}] doSomething called`);

    // 実行
    const result = this.real.doSomething(args);
    this.cache.set(key, result);
    return result;
  }
}

// 問題:
// 1. SRP 違反: 4 つの責務が混在
// 2. テストが困難（すべての責務をテストする必要がある）
// 3. 責務の組み合わせを変更できない
```

```typescript
// OK: 責務ごとに Proxy を分離し、チェーンする

class AuthProxy implements Service {
  constructor(private real: Service, private user: User) {}
  doSomething(args: unknown): Result {
    if (!this.user.isAdmin) throw new Error("Access denied");
    return this.real.doSomething(args);
  }
}

class RateLimitProxy implements Service {
  private count = 0;
  constructor(private real: Service, private limit: number) {}
  doSomething(args: unknown): Result {
    if (++this.count > this.limit) throw new Error("Rate limited");
    return this.real.doSomething(args);
  }
}

class CacheProxy implements Service {
  private cache = new Map();
  constructor(private real: Service) {}
  doSomething(args: unknown): Result {
    const key = JSON.stringify(args);
    if (this.cache.has(key)) return this.cache.get(key);
    const result = this.real.doSomething(args);
    this.cache.set(key, result);
    return result;
  }
}

class LogProxy implements Service {
  constructor(private real: Service) {}
  doSomething(args: unknown): Result {
    console.log(`[${new Date().toISOString()}] doSomething called`);
    return this.real.doSomething(args);
  }
}

// チェーン: Auth -> RateLimit -> Cache -> Log -> Real
const service: Service =
  new AuthProxy(
    new RateLimitProxy(
      new CacheProxy(
        new LogProxy(realService),
      ),
      100,
    ),
    currentUser,
  );
```

**判断基準**: 1 Proxy = 1 責務。複数の責務があればチェーンで合成する。

---

### アンチパターン 3: 不必要な Proxy（YAGNI 違反）

```typescript
// NG: 何の付加価値もない Proxy
class UselessProxy implements Service {
  constructor(private real: Service) {}

  doSomething(args: unknown): Result {
    return this.real.doSomething(args); // ただの委譲
  }
}

// 問題:
// 1. 間接層の追加による複雑さだけが増える
// 2. デバッグ時のスタックトレースが深くなる
// 3. パフォーマンスオーバーヘッド（わずかだが不要）
```

**判断基準**: Proxy を導入する前に「何を制御するのか」を明確にする。制御すべきものがなければ Proxy は不要。

---

## 6. エッジケースと注意点

### 6.1 ES6 Proxy のパフォーマンス

```typescript
// ES6 Proxy はプロパティアクセスごとにトラップが呼ばれるため、
// ホットパスでの使用に注意

// NG: ホットループ内で ES6 Proxy
const proxied = new Proxy(array, handler);
for (let i = 0; i < 1_000_000; i++) {
  proxied[i]; // 毎回 get トラップが呼ばれる
}

// OK: ホットパスでは直接アクセス、Proxy は外部API向け
const raw = array;
for (let i = 0; i < 1_000_000; i++) {
  raw[i]; // 直接アクセス
}
```

### 6.2 Proxy の透過性とデバッグ

```typescript
// ES6 Proxy は typeof, instanceof で RealSubject と区別できない
const target = { x: 1 };
const proxy = new Proxy(target, {});

console.log(typeof proxy);           // "object" (target と同じ)
console.log(proxy instanceof Object); // true
console.log(proxy === target);        // false (参照は異なる)

// デバッグ時は Proxy であることを示すメタデータを追加すると便利
const debugProxy = new Proxy(target, {
  get(obj, prop) {
    if (prop === Symbol.for("isProxy")) return true;
    return Reflect.get(obj, prop);
  },
});
```

### 6.3 キャッシュ Proxy の Invalidation 戦略

```
┌──────────────────────────────────────────────────────────┐
│          キャッシュ Invalidation 戦略の選択               │
│                                                          │
│  ┌─────────────┬─────────────────────────────────────┐  │
│  │ TTL         │ 一定時間後に自動失効                 │  │
│  │             │ 簡単だが鮮度に制約                    │  │
│  ├─────────────┼─────────────────────────────────────┤  │
│  │ Event-based │ データ変更イベントで明示的に無効化   │  │
│  │             │ 正確だが実装が複雑                    │  │
│  ├─────────────┼─────────────────────────────────────┤  │
│  │ LRU         │ 容量超過時に最古のエントリを削除     │  │
│  │             │ メモリ制約に有効                      │  │
│  ├─────────────┼─────────────────────────────────────┤  │
│  │ Write-through│ 書き込み時にキャッシュも更新        │  │
│  │             │ 一貫性は高いが書き込みが遅い          │  │
│  ├─────────────┼─────────────────────────────────────┤  │
│  │ Stale-while │ 期限切れでも古い値を返しつつ         │  │
│  │ -revalidate │ バックグラウンドで更新               │  │
│  │             │ 可用性が高い                          │  │
│  └─────────────┴─────────────────────────────────────┘  │
│                                                          │
│  推奨: 最初は TTL + LRU のシンプルな組み合わせから始め、 │
│  必要に応じて Event-based を追加する。                    │
└──────────────────────────────────────────────────────────┘
```

### 6.4 スレッドセーフな Proxy

```typescript
// ブラウザでは問題ないが、Node.js Worker Threads や
// マルチスレッド環境ではキャッシュ Proxy にロックが必要

class ThreadSafeCacheProxy implements Service {
  private cache = new Map();
  private locks = new Map<string, Promise<void>>();

  async doSomething(key: string): Promise<Result> {
    // 同じキーに対する同時リクエストを防ぐ
    if (this.locks.has(key)) {
      await this.locks.get(key);
      const cached = this.cache.get(key);
      if (cached) return cached;
    }

    let resolve: () => void;
    this.locks.set(key, new Promise(r => { resolve = r; }));

    try {
      const result = await this.real.doSomething(key);
      this.cache.set(key, result);
      return result;
    } finally {
      resolve!();
      this.locks.delete(key);
    }
  }
}
```

---

## 7. トレードオフ分析

### 導入すべき場面

| 場面 | 推奨 Proxy 種類 | 理由 |
|------|----------------|------|
| 画像ギャラリー（大量の画像） | Virtual | メモリ節約、初期化コスト削減 |
| マルチテナント API | Protection | テナントごとのアクセス制御 |
| 外部 API 呼び出し | Cache + Logging | コスト削減、デバッグ容易性 |
| マイクロサービス間通信 | Remote + Circuit Breaker | 障害伝搬の防止 |
| 監査要件のあるシステム | Logging | コンプライアンス対応 |

### 導入すべきでない場面

| 場面 | 理由 |
|------|------|
| 単純な直接呼び出しで十分 | YAGNI: 不要な間接層 |
| ホットパスの内部処理 | パフォーマンスオーバーヘッド |
| 横断的関心事がない | Proxy の付加価値がない |
| AOP フレームワークが利用可能 | Proxy を手動実装する必要がない |

### Proxy のコスト

```
┌──────────────────────────────────────────────────────────┐
│                 Proxy のコスト分析                        │
│                                                          │
│  メリット                    デメリット                   │
│  ┌──────────────────────┐    ┌──────────────────────┐    │
│  │ + 横断的関心事の分離  │    │ - 間接層の追加       │    │
│  │ + OCP 準拠           │    │ - デバッグの複雑化   │    │
│  │ + 遅延初期化         │    │ - God Proxy のリスク │    │
│  │ + テスト容易性       │    │ - ES6 Proxy の       │    │
│  │ + 透過的             │    │   パフォーマンス     │    │
│  │ + 組み合わせ可能     │    │ - キャッシュの       │    │
│  │   （チェーン）       │    │   一貫性管理         │    │
│  └──────────────────────┘    └──────────────────────┘    │
│                                                          │
│  判断: 横断的関心事（認証、キャッシュ、ログ）が           │
│  ビジネスロジックに混入している場合に Proxy を検討する。  │
└──────────────────────────────────────────────────────────┘
```

---

## 8. 演習問題

### 演習 1: 基本 -- Virtual Proxy（難易度: ★☆☆）

以下の `ExpensiveReport` クラスに対する Virtual Proxy を実装してください。

```typescript
interface Report {
  getTitle(): string;
  generate(): string;
}

class ExpensiveReport implements Report {
  private data: string;

  constructor(private title: string) {
    // 重い初期化処理（DB クエリ、集計処理等）
    console.log(`[Report] Generating "${title}" (takes 3 seconds)...`);
    this.data = `Report data for "${title}" with 10000 rows`;
  }

  getTitle(): string {
    return this.title;
  }

  generate(): string {
    return this.data;
  }
}

// TODO: ReportProxy を実装
// - getTitle() は RealSubject を生成せずに返す
// - generate() は初回アクセス時のみ RealSubject を生成
```

**期待される出力**:

```
Creating 3 report proxies...
(no loading happens yet)

Getting title: Monthly Sales
(still no loading)

Generating report:
[Report] Generating "Monthly Sales" (takes 3 seconds)...
Report data for "Monthly Sales" with 10000 rows
```

<details>
<summary>解答例（クリックで展開）</summary>

```typescript
class ReportProxy implements Report {
  private real: ExpensiveReport | null = null;

  constructor(private title: string) {
    console.log(`[Proxy] Created proxy for "${title}"`);
  }

  getTitle(): string {
    // タイトルは Proxy が知っているので RealSubject 不要
    return this.title;
  }

  generate(): string {
    if (!this.real) {
      this.real = new ExpensiveReport(this.title);
    }
    return this.real.generate();
  }
}

// テスト
console.log("Creating 3 report proxies...");
const reports: Report[] = [
  new ReportProxy("Monthly Sales"),
  new ReportProxy("Weekly Users"),
  new ReportProxy("Daily Revenue"),
];
console.log("(no loading happens yet)\n");

console.log(`Getting title: ${reports[0].getTitle()}`);
console.log("(still no loading)\n");

console.log("Generating report:");
console.log(reports[0].generate());
// [Report] Generating "Monthly Sales" (takes 3 seconds)...
// Report data for "Monthly Sales" with 10000 rows
```

</details>

---

### 演習 2: 応用 -- Protection + Logging Proxy チェーン（難易度: ★★☆）

以下の要件を満たす Proxy チェーンを実装してください。

**要件**:
1. `FileService` インタフェースに対して、Protection Proxy と Logging Proxy を作成
2. Protection Proxy: `"admin"` ロール以外は `delete` を禁止
3. Logging Proxy: 全メソッド呼び出しを `[LOG]` プレフィックスで記録
4. チェーン順序: Client -> Logging -> Protection -> RealSubject

```typescript
interface FileService {
  read(path: string): string;
  write(path: string, content: string): void;
  delete(path: string): void;
}

interface User {
  name: string;
  role: "viewer" | "editor" | "admin";
}
```

**期待される出力（admin ユーザー）**:

```
[LOG] read("/data/file.txt")
[File] Reading /data/file.txt
[LOG] delete("/data/old.txt")
[Auth] admin access granted for delete
[File] Deleting /data/old.txt
```

**期待される出力（viewer ユーザー）**:

```
[LOG] read("/data/file.txt")
[File] Reading /data/file.txt
[LOG] delete("/data/old.txt")
[Auth] Access denied: viewer cannot delete
Error: Access denied
```

<details>
<summary>解答例（クリックで展開）</summary>

```typescript
class RealFileService implements FileService {
  read(path: string): string {
    console.log(`[File] Reading ${path}`);
    return `content of ${path}`;
  }
  write(path: string, content: string): void {
    console.log(`[File] Writing ${path}`);
  }
  delete(path: string): void {
    console.log(`[File] Deleting ${path}`);
  }
}

class ProtectionProxy implements FileService {
  constructor(
    private real: FileService,
    private user: User,
  ) {}

  read(path: string): string {
    return this.real.read(path); // 全ロール許可
  }

  write(path: string, content: string): void {
    if (this.user.role === "viewer") {
      console.log(`[Auth] Access denied: ${this.user.role} cannot write`);
      throw new Error("Access denied");
    }
    return this.real.write(path, content);
  }

  delete(path: string): void {
    if (this.user.role !== "admin") {
      console.log(`[Auth] Access denied: ${this.user.role} cannot delete`);
      throw new Error("Access denied");
    }
    console.log(`[Auth] ${this.user.role} access granted for delete`);
    this.real.delete(path);
  }
}

class LoggingProxy implements FileService {
  constructor(private real: FileService) {}

  read(path: string): string {
    console.log(`[LOG] read("${path}")`);
    return this.real.read(path);
  }

  write(path: string, content: string): void {
    console.log(`[LOG] write("${path}", "${content.substring(0, 20)}...")`);
    this.real.write(path, content);
  }

  delete(path: string): void {
    console.log(`[LOG] delete("${path}")`);
    this.real.delete(path);
  }
}

// チェーン構築
function createFileService(user: User): FileService {
  const real = new RealFileService();
  const protected_ = new ProtectionProxy(real, user);
  const logged = new LoggingProxy(protected_);
  return logged;
}

// Admin テスト
const adminFs = createFileService({ name: "Alice", role: "admin" });
adminFs.read("/data/file.txt");
adminFs.delete("/data/old.txt");

// Viewer テスト
const viewerFs = createFileService({ name: "Bob", role: "viewer" });
viewerFs.read("/data/file.txt");
try {
  viewerFs.delete("/data/old.txt");
} catch (e) {
  console.log(`Error: ${(e as Error).message}`);
}
```

</details>

---

### 演習 3: 発展 -- 汎用 Proxy ファクトリ（難易度: ★★★）

ES6 Proxy を使って、任意のオブジェクトに対して以下の機能を自動付加する汎用ファクトリを実装してください。

**要件**:
1. `createSmartProxy<T>(target, options)` ファクトリ関数
2. `options.cache`: true の場合、同じ引数のメソッド呼び出し結果をキャッシュ
3. `options.log`: true の場合、メソッド呼び出しをログ出力
4. `options.readonly`: true の場合、プロパティの set と delete を禁止
5. `options.onAccess`: プロパティアクセス時のコールバック
6. 同期・非同期メソッドの両方に対応

```typescript
interface SmartProxyOptions {
  cache?: boolean;
  log?: boolean;
  readonly?: boolean;
  onAccess?: (prop: string) => void;
}

function createSmartProxy<T extends object>(
  target: T,
  options: SmartProxyOptions,
): T {
  // TODO: 実装
}
```

**期待される出力**:

```typescript
const obj = createSmartProxy(
  {
    greet(name: string) { return `Hello, ${name}!`; },
    value: 42,
  },
  { cache: true, log: true, readonly: true },
);

obj.greet("World");
// [Log] greet("World") -> "Hello, World!" [0ms]
// "Hello, World!"

obj.greet("World"); // 2回目: キャッシュヒット
// [Cache] HIT: greet:["World"]
// "Hello, World!"

obj.value = 100; // Error: Cannot modify readonly proxy
```

<details>
<summary>解答例（クリックで展開）</summary>

```typescript
function createSmartProxy<T extends object>(
  target: T,
  options: SmartProxyOptions = {},
): T {
  const cache = new Map<string, unknown>();

  return new Proxy(target, {
    get(obj, prop, receiver) {
      const value = Reflect.get(obj, prop, receiver);

      // onAccess コールバック
      if (options.onAccess && typeof prop === "string") {
        options.onAccess(prop);
      }

      // メソッドでなければそのまま返す
      if (typeof value !== "function") return value;

      // メソッドをラップ
      return function (this: unknown, ...args: unknown[]) {
        const methodName = String(prop);

        // キャッシュチェック
        if (options.cache) {
          const key = `${methodName}:${JSON.stringify(args)}`;
          if (cache.has(key)) {
            console.log(`[Cache] HIT: ${key}`);
            return cache.get(key);
          }
        }

        // 実行
        const start = performance.now();
        const result = (value as Function).apply(obj, args);

        // Promise の場合
        if (result instanceof Promise) {
          return result.then((resolved: unknown) => {
            const elapsed = Math.round(performance.now() - start);
            if (options.log) {
              console.log(`[Log] ${methodName}(${args.map(a => JSON.stringify(a)).join(", ")}) -> ${JSON.stringify(resolved)} [${elapsed}ms]`);
            }
            if (options.cache) {
              cache.set(`${methodName}:${JSON.stringify(args)}`, resolved);
            }
            return resolved;
          });
        }

        // 同期の場合
        const elapsed = Math.round(performance.now() - start);
        if (options.log) {
          console.log(`[Log] ${methodName}(${args.map(a => JSON.stringify(a)).join(", ")}) -> ${JSON.stringify(result)} [${elapsed}ms]`);
        }
        if (options.cache) {
          cache.set(`${methodName}:${JSON.stringify(args)}`, result);
        }
        return result;
      };
    },

    set(obj, prop, value) {
      if (options.readonly) {
        throw new Error(`Cannot modify readonly proxy (property: ${String(prop)})`);
      }
      return Reflect.set(obj, prop, value);
    },

    deleteProperty(obj, prop) {
      if (options.readonly) {
        throw new Error(`Cannot delete from readonly proxy (property: ${String(prop)})`);
      }
      return Reflect.deleteProperty(obj, prop);
    },
  });
}

// テスト
const calculator = createSmartProxy(
  {
    add(a: number, b: number) { return a + b; },
    async fetchData(id: string) {
      await new Promise(r => setTimeout(r, 50));
      return { id, data: "result" };
    },
    value: 42,
  },
  { cache: true, log: true, readonly: true },
);

console.log(calculator.add(1, 2));
// [Log] add(1, 2) -> 3 [0ms]
// 3

console.log(calculator.add(1, 2)); // キャッシュヒット
// [Cache] HIT: add:[1,2]
// 3

const data = await calculator.fetchData("abc");
// [Log] fetchData("abc") -> {"id":"abc","data":"result"} [51ms]

try {
  calculator.value = 100;
} catch (e) {
  console.log((e as Error).message);
  // Cannot modify readonly proxy (property: value)
}
```

</details>

---

## 9. FAQ

### Q1: JavaScript の Proxy オブジェクトは GoF の Proxy パターンと同じですか？

本質は同じですが、抽象度が異なります。GoF Proxy はクラスベースで「同じインタフェースを持つ代理クラス」を作ります。ES6 Proxy はメタプログラミングの仕組みで、13 種のトラップ（get, set, has, deleteProperty, apply, construct 等）を通じてあらゆるオブジェクト操作をインターセプトできます。GoF Proxy の全パターン（Virtual, Protection, Cache, Logging）を ES6 Proxy で実装できますが、逆は成り立ちません（ES6 Proxy のプロパティトラップは GoF にはない概念）。

### Q2: Proxy はパフォーマンスに影響しますか？

Proxy 層の処理内容に依存します。GoF スタイルのクラス Proxy は単純な委譲なのでオーバーヘッドはほぼゼロです。ES6 Proxy の handler はプロパティアクセスごとに呼ばれるため、ホットパス（1 秒に 100 万回以上のアクセス）では注意が必要です。V8 エンジンのベンチマークでは、ES6 Proxy 経由のプロパティアクセスは直接アクセスの 5-10 倍遅くなることがあります。

### Q3: キャッシュ Proxy の Invalidation はどうすべきですか？

「キャッシュの無効化はコンピュータサイエンスで最も難しい問題の 1 つ」と言われます。以下の順序で段階的に導入してください。

1. **TTL（時間ベース）**: 最もシンプル。60 秒後に自動失効
2. **TTL + LRU**: メモリ制約があれば LRU を追加
3. **Event-based**: データ更新イベントで明示的に無効化（必要になったら）
4. **Stale-while-revalidate**: 高可用性が必要な場合

### Q4: Proxy と AOP（アスペクト指向プログラミング）の関係は？

AOP のウィービング（横断的関心事の織り込み）を手動で実装したものが Proxy パターンです。Spring AOP や AspectJ は Proxy パターンを内部で使用しています。フレームワークが AOP をサポートしていれば、手動 Proxy は不要です。

### Q5: React の Suspense は Virtual Proxy ですか？

はい、概念的に Virtual Proxy の一形態です。`React.lazy()` はコンポーネントの遅延ロードを行い、ロード完了まで Suspense フォールバックを表示します。これは Virtual Proxy の「必要になるまで RealSubject を生成しない」という特性と同じです。

### Q6: Proxy チェーンの順序はどう決めるべきですか？

外側から順に評価されることを意識して、以下の順序を推奨します。

```
Client
  -> Logging（全アクセスを記録）
    -> Rate Limiting（早い段階で制限）
      -> Authentication（認証チェック）
        -> Caching（キャッシュで高速化）
          -> RealSubject
```

理由: ログは全リクエストを記録したいので最外層、レート制限は早い段階で弾きたいので認証の前、キャッシュは認証済みリクエストのみ対象にしたいので最内層。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| **目的** | オブジェクトへのアクセスを制御する代理オブジェクト |
| **本質** | 同一インタフェース + 透過的なアクセス制御 |
| **主な種類** | Virtual, Protection, Cache, Remote, Logging, Smart Reference |
| **GoF vs ES6** | GoF はクラスベース、ES6 はメタプログラミング |
| **利点** | 横断的関心事の分離、遅延初期化、OCP 準拠 |
| **チェーン** | 1 Proxy = 1 責務、チェーンで合成 |
| **注意** | ES6 Proxy のパフォーマンス、キャッシュ Invalidation |
| **テスト** | DI で RealSubject を注入、モックに差し替え可能 |

---

## 次に読むべきガイド

- [Decorator パターン](./01-decorator.md) -- 動的な機能追加（Proxy との違いを理解）
- [Facade パターン](./02-facade.md) -- サブシステムの簡素化
- [Adapter パターン](./00-adapter.md) -- インタフェース変換
- [Composite パターン](./04-composite.md) -- ツリー構造の統一操作
- [Observer パターン](../02-behavioral/00-observer.md) -- リアクティブな変更通知
- [キャッシュ戦略](../../../system-design-guide/docs/01-components/01-caching.md) -- システムレベルのキャッシュ設計

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. MDN Web Docs -- Proxy. https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Proxy
3. MDN Web Docs -- Reflect. https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Reflect
4. Refactoring.Guru -- Proxy. https://refactoring.guru/design-patterns/proxy
5. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
6. Rauschmayer, A. (2015). *Exploring ES6*. Chapter 28: Metaprogramming with Proxies. https://exploringjs.com/es6/ch_proxies.html
