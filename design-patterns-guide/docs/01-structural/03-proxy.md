# Proxy パターン

> 別のオブジェクトへの **アクセスを制御する代理オブジェクト** を提供し、遅延初期化・アクセス制御・キャッシュなどの横断的関心事を実装する構造パターン。

---

## この章で学ぶこと

1. Proxy パターンの種類（Virtual / Protection / Cache / Remote）と各用途
2. JavaScript の Proxy オブジェクトとの関係と実践的活用
3. Proxy と Decorator の違いと正しい使い分け

---

## 1. Proxy の構造

```
+--------+       +-----------+       +--------------+
| Client |------>|   Proxy   |------>| RealSubject  |
+--------+       +-----------+       +--------------+
                 | - real:   |       | + request()  |
    same         |  Subject  |       +--------------+
    interface     | + request()|
                 | (制御ロジック)
+-----------+    +-----------+
|  Subject  |<---+
| (interface)|
| + request()|
+-----------+
```

---

## 2. Proxy の種類

```
┌─────────────────────────────────────────────────────┐
│                   Proxy の分類                       │
├──────────────┬──────────────────────────────────────┤
│ Virtual      │ 重いオブジェクトの遅延初期化          │
│ Proxy        │ 例: 大きな画像、DB接続               │
├──────────────┼──────────────────────────────────────┤
│ Protection   │ アクセス権限のチェック                │
│ Proxy        │ 例: 認証・認可ガード                  │
├──────────────┼──────────────────────────────────────┤
│ Cache        │ 結果のキャッシュ                      │
│ Proxy        │ 例: API レスポンスのメモ化            │
├──────────────┼──────────────────────────────────────┤
│ Remote       │ リモートオブジェクトのローカル代理    │
│ Proxy        │ 例: RPC, gRPC スタブ                  │
├──────────────┼──────────────────────────────────────┤
│ Logging      │ 操作の記録・監査                      │
│ Proxy        │ 例: メソッド呼び出しのトレース        │
└──────────────┴──────────────────────────────────────┘
```

---

## 3. コード例

### コード例 1: Virtual Proxy（遅延初期化）

```typescript
interface Image {
  display(): void;
  getSize(): number;
}

class HighResImage implements Image {
  private data: Buffer;

  constructor(private filename: string) {
    // 重い処理: ファイルを読み込む
    console.log(`Loading ${filename}...`);
    this.data = Buffer.alloc(10_000_000); // 10MB
  }

  display(): void { console.log(`Displaying ${this.filename}`); }
  getSize(): number { return this.data.length; }
}

class ImageProxy implements Image {
  private real: HighResImage | null = null;

  constructor(private filename: string) {}

  private ensureLoaded(): HighResImage {
    if (!this.real) {
      this.real = new HighResImage(this.filename);
    }
    return this.real;
  }

  display(): void { this.ensureLoaded().display(); }
  getSize(): number { return this.ensureLoaded().getSize(); }
}

// 生成時にはロードされない
const img = new ImageProxy("photo.jpg");
// display() 呼び出し時に初めてロード
img.display();
```

### コード例 2: Protection Proxy

```typescript
interface AdminService {
  deleteUser(userId: string): void;
  resetDatabase(): void;
}

class RealAdminService implements AdminService {
  deleteUser(userId: string): void {
    console.log(`User ${userId} deleted`);
  }
  resetDatabase(): void {
    console.log("Database reset");
  }
}

class AdminProxy implements AdminService {
  constructor(
    private real: AdminService,
    private currentUser: { role: string }
  ) {}

  deleteUser(userId: string): void {
    if (this.currentUser.role !== "admin") {
      throw new Error("Access denied: admin role required");
    }
    this.real.deleteUser(userId);
  }

  resetDatabase(): void {
    if (this.currentUser.role !== "superadmin") {
      throw new Error("Access denied: superadmin role required");
    }
    this.real.resetDatabase();
  }
}
```

### コード例 3: Cache Proxy

```typescript
interface ApiClient {
  fetchUser(id: string): Promise<User>;
}

class RealApiClient implements ApiClient {
  async fetchUser(id: string): Promise<User> {
    const res = await fetch(`/api/users/${id}`);
    return res.json();
  }
}

class CachingProxy implements ApiClient {
  private cache = new Map<string, { data: User; expiry: number }>();
  private ttl = 60_000; // 1分

  constructor(private real: ApiClient) {}

  async fetchUser(id: string): Promise<User> {
    const cached = this.cache.get(id);
    if (cached && cached.expiry > Date.now()) {
      console.log(`Cache HIT: ${id}`);
      return cached.data;
    }

    console.log(`Cache MISS: ${id}`);
    const user = await this.real.fetchUser(id);
    this.cache.set(id, { data: user, expiry: Date.now() + this.ttl });
    return user;
  }
}
```

### コード例 4: JavaScript Proxy オブジェクト

```typescript
// ES6 Proxy を使ったバリデーション
interface UserData {
  name: string;
  age: number;
  email: string;
}

function createValidatedUser(data: UserData): UserData {
  return new Proxy(data, {
    set(target, prop: keyof UserData, value) {
      if (prop === "age" && (typeof value !== "number" || value < 0)) {
        throw new Error("age must be a non-negative number");
      }
      if (prop === "email" && !value.includes("@")) {
        throw new Error("Invalid email format");
      }
      target[prop] = value;
      return true;
    },
    get(target, prop: keyof UserData) {
      console.log(`Accessing ${String(prop)}`);
      return target[prop];
    },
  });
}

const user = createValidatedUser({ name: "Taro", age: 30, email: "t@x.com" });
user.age = -1; // Error: age must be a non-negative number
```

### コード例 5: Python — Proxy

```python
from abc import ABC, abstractmethod
import time

class Database(ABC):
    @abstractmethod
    def query(self, sql: str) -> list: ...

class RealDatabase(Database):
    def query(self, sql: str) -> list:
        time.sleep(0.1)  # 重いクエリ
        return [{"id": 1, "name": "Taro"}]

class LoggingProxy(Database):
    def __init__(self, real: Database):
        self._real = real

    def query(self, sql: str) -> list:
        start = time.time()
        result = self._real.query(sql)
        elapsed = time.time() - start
        print(f"[{elapsed:.3f}s] {sql} -> {len(result)} rows")
        return result

db: Database = LoggingProxy(RealDatabase())
db.query("SELECT * FROM users")
```

---

## 4. 比較表

### 比較表 1: Proxy vs Decorator

| 観点 | Proxy | Decorator |
|------|-------|-----------|
| 目的 | アクセス制御 | 機能追加 |
| ライフサイクル管理 | する（遅延生成等） | しない |
| クライアントの認識 | 透過的 | 透過的 |
| 典型的な責務 | 認証、キャッシュ、遅延 | ログ、圧縮、暗号化 |
| 実体の生成 | Proxy が管理 | 外部から渡される |

### 比較表 2: Proxy の種類と適用場面

| 種類 | ユースケース | パフォーマンス影響 |
|------|-------------|:---:|
| Virtual | 重いリソースの遅延読み込み | 初回のみ遅延 |
| Protection | 権限チェック | 微小 |
| Cache | API/DB 結果のメモ化 | 大幅に改善 |
| Remote | RPC/gRPC 呼び出し | ネットワーク依存 |
| Logging | 監査ログ | 微小 |

---

## 5. アンチパターン

### アンチパターン 1: Proxy と RealSubject の密結合

```typescript
// BAD: Proxy が具象クラスに依存
class BadProxy {
  private real = new SpecificService(); // 直接 new
  // インタフェースを介していない
}
```

**改善**: インタフェース経由で依存し、DI で注入する。

### アンチパターン 2: 1つの Proxy に複数の責務

```typescript
// BAD: キャッシュ + 認証 + ログを1つの Proxy で
class GodProxy implements Service {
  doSomething(): void {
    this.checkAuth();      // 認証
    const cached = this.getCache(); // キャッシュ
    this.log("doSomething"); // ログ
    // ...
  }
}
```

**改善**: 責務ごとに Proxy/Decorator を分離し、チェーンする。

---

## 6. FAQ

### Q1: JavaScript の Proxy オブジェクトは GoF の Proxy パターンと同じですか？

メタプログラミングの仕組みという点で異なりますが、「オブジェクトへのアクセスをインターセプトする」という本質は同じです。ES6 Proxy はプロパティアクセス、代入、関数呼び出し等をトラップできるため、より汎用的です。

### Q2: Proxy はパフォーマンスに影響しますか？

Proxy 層の処理内容に依存します。単純な委譲ならほぼ無視できますが、ES6 Proxy の handler はプロパティアクセスごとに呼ばれるため、ホットパスでの使用には注意が必要です。

### Q3: キャッシュ Proxy の invalidation はどうすべきですか？

TTL（時間ベース）、イベント駆動（変更通知で無効化）、LRU（最近使われていないものを破棄）の3つが一般的です。「キャッシュの無効化はコンピュータサイエンスで最も難しい問題の1つ」と言われるため、シンプルな戦略から始めてください。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | オブジェクトへのアクセスを制御する代理 |
| 主な種類 | Virtual, Protection, Cache, Remote, Logging |
| 利点 | 横断的関心事の分離、遅延初期化 |
| JS/TS 固有 | ES6 Proxy で動的にトラップ可能 |
| 注意 | 1 Proxy = 1 責務を守る |

---

## 次に読むべきガイド

- [Composite パターン](./04-composite.md) — ツリー構造
- [Decorator パターン](./01-decorator.md) — 動的な機能追加
- [キャッシュ戦略](../../../system-design-guide/docs/01-components/01-caching.md) — システムレベルのキャッシュ

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. MDN Web Docs — Proxy. https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Proxy
3. Refactoring.Guru — Proxy. https://refactoring.guru/design-patterns/proxy
