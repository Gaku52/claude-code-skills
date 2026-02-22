# 抽象化

> 抽象化は「複雑さを隠し、本質的な特徴だけを公開する」原則。インターフェース設計、抽象クラスの使い方、そして「リーキー抽象化」の回避がポイント。

## この章で学ぶこと

- [ ] 抽象化のレベルと適用方法を理解する
- [ ] インターフェースと抽象クラスの使い分けを把握する
- [ ] リーキー抽象化の問題とその回避を学ぶ
- [ ] 良い抽象化と悪い抽象化の見分け方を身につける
- [ ] 各言語でのインターフェース設計の実践パターンを学ぶ
- [ ] 実務におけるレイヤードアーキテクチャとの関係を理解する

---

## 1. 抽象化のレベル

```
抽象化 = 「不要な詳細を隠し、重要な情報だけを公開する」

レベル1: データ抽象化
  → 内部表現を隠す（カプセル化と重なる）
  → Date クラス: 内部がタイムスタンプか年月日構造体かを隠す

レベル2: 手続き抽象化
  → 処理の詳細を関数/メソッドに閉じ込める
  → array.sort(): ソートアルゴリズムの詳細を隠す

レベル3: 型抽象化（インターフェース）
  → 「何ができるか」だけを定義し、「どうやるか」は隠す
  → Iterable: 反復可能であることだけを約束

レベル4: モジュール抽象化
  → パッケージ/モジュールの公開APIのみを見せる
  → 内部のクラス群の複雑さを隠す

  ┌──────────── 利用者が見る世界 ────────────┐
  │  database.query("SELECT * FROM users")    │
  └────────────────────────────────────────────┘
                     ↓ 隠蔽
  ┌──────────── 内部の複雑さ ──────────────────┐
  │ コネクションプール管理                      │
  │ SQL パース → クエリプラン最適化             │
  │ インデックス検索 → ページ読み込み           │
  │ ロック管理 → トランザクション制御           │
  │ 結果セットのシリアライズ                    │
  └────────────────────────────────────────────┘
```

### 1.1 抽象化の具体例：各レベルの実践

```typescript
// レベル1: データ抽象化
// 内部表現を隠して統一的なインターフェースを提供
class Temperature {
  // 内部はケルビン（K）で保持するが、利用者は意識しない
  private kelvin: number;

  private constructor(kelvin: number) {
    if (kelvin < 0) throw new Error("絶対零度以下の温度は存在しません");
    this.kelvin = kelvin;
  }

  // ファクトリメソッド: 様々な単位から生成
  static fromCelsius(c: number): Temperature {
    return new Temperature(c + 273.15);
  }

  static fromFahrenheit(f: number): Temperature {
    return new Temperature((f - 32) * 5 / 9 + 273.15);
  }

  static fromKelvin(k: number): Temperature {
    return new Temperature(k);
  }

  // 様々な単位で取得
  toCelsius(): number {
    return this.kelvin - 273.15;
  }

  toFahrenheit(): number {
    return (this.kelvin - 273.15) * 9 / 5 + 32;
  }

  toKelvin(): number {
    return this.kelvin;
  }

  // 比較
  isHigherThan(other: Temperature): boolean {
    return this.kelvin > other.kelvin;
  }

  // 人間が読める形式
  toString(): string {
    return `${this.toCelsius().toFixed(1)}°C`;
  }
}

// 利用者は内部表現（ケルビン）を知る必要がない
const boiling = Temperature.fromCelsius(100);
const body = Temperature.fromFahrenheit(98.6);
console.log(boiling.toString());                  // 100.0°C
console.log(body.toFahrenheit());                 // 98.6
console.log(boiling.isHigherThan(body));           // true
```

```python
# レベル2: 手続き抽象化
# 複雑な処理を意味のある名前の関数に閉じ込める
import hashlib
import secrets
import re
from typing import Optional


class PasswordManager:
    """パスワード管理の抽象化"""

    SALT_LENGTH = 32
    HASH_ITERATIONS = 100_000
    MIN_PASSWORD_LENGTH = 8

    def hash_password(self, password: str) -> str:
        """パスワードをハッシュ化（詳細は隠蔽）"""
        # 利用者は以下の詳細を知る必要がない:
        # - ソルトの生成方法
        # - ハッシュアルゴリズム（PBKDF2 + SHA256）
        # - イテレーション回数
        # - エンコード形式
        salt = secrets.token_hex(self.SALT_LENGTH)
        hash_value = self._compute_hash(password, salt)
        return f"{salt}:{hash_value}"

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """パスワードを検証"""
        salt, expected_hash = stored_hash.split(":")
        actual_hash = self._compute_hash(password, salt)
        return secrets.compare_digest(actual_hash, expected_hash)

    def validate_strength(self, password: str) -> list[str]:
        """パスワード強度を検証して問題のリストを返す"""
        errors = []
        if len(password) < self.MIN_PASSWORD_LENGTH:
            errors.append(f"{self.MIN_PASSWORD_LENGTH}文字以上必要です")
        if not re.search(r"[A-Z]", password):
            errors.append("大文字を1文字以上含めてください")
        if not re.search(r"[a-z]", password):
            errors.append("小文字を1文字以上含めてください")
        if not re.search(r"\d", password):
            errors.append("数字を1文字以上含めてください")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            errors.append("特殊文字を1文字以上含めてください")
        return errors

    def _compute_hash(self, password: str, salt: str) -> str:
        """内部: ハッシュ計算（プライベートメソッド）"""
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            self.HASH_ITERATIONS,
        ).hex()


# 利用者は hash/verify を呼ぶだけ
pm = PasswordManager()
hashed = pm.hash_password("MySecureP@ss1")
print(pm.verify_password("MySecureP@ss1", hashed))  # True
print(pm.verify_password("wrong", hashed))            # False
```

```typescript
// レベル3: 型抽象化（インターフェース）
// 「何ができるか」だけを定義

interface Cache<T> {
  get(key: string): Promise<T | null>;
  set(key: string, value: T, ttlSeconds?: number): Promise<void>;
  delete(key: string): Promise<void>;
  has(key: string): Promise<boolean>;
  clear(): Promise<void>;
}

// Redis 実装
class RedisCache<T> implements Cache<T> {
  constructor(private redisClient: any) {}

  async get(key: string): Promise<T | null> {
    const value = await this.redisClient.get(key);
    return value ? JSON.parse(value) : null;
  }

  async set(key: string, value: T, ttlSeconds: number = 3600): Promise<void> {
    await this.redisClient.set(key, JSON.stringify(value), "EX", ttlSeconds);
  }

  async delete(key: string): Promise<void> {
    await this.redisClient.del(key);
  }

  async has(key: string): Promise<boolean> {
    return (await this.redisClient.exists(key)) === 1;
  }

  async clear(): Promise<void> {
    await this.redisClient.flushdb();
  }
}

// インメモリ実装（テスト用）
class InMemoryCache<T> implements Cache<T> {
  private store = new Map<string, { value: T; expiresAt: number }>();

  async get(key: string): Promise<T | null> {
    const entry = this.store.get(key);
    if (!entry) return null;
    if (entry.expiresAt < Date.now()) {
      this.store.delete(key);
      return null;
    }
    return entry.value;
  }

  async set(key: string, value: T, ttlSeconds: number = 3600): Promise<void> {
    this.store.set(key, {
      value,
      expiresAt: Date.now() + ttlSeconds * 1000,
    });
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key);
  }

  async has(key: string): Promise<boolean> {
    return (await this.get(key)) !== null;
  }

  async clear(): Promise<void> {
    this.store.clear();
  }
}

// ファイルシステム実装
class FileCache<T> implements Cache<T> {
  constructor(private cacheDir: string) {}

  async get(key: string): Promise<T | null> {
    // ファイルからJSONを読み込み、TTLをチェック
    try {
      const filePath = `${this.cacheDir}/${this.hashKey(key)}.json`;
      // const content = await fs.readFile(filePath, 'utf-8');
      // const { value, expiresAt } = JSON.parse(content);
      // if (expiresAt < Date.now()) return null;
      // return value;
      return null;
    } catch {
      return null;
    }
  }

  async set(key: string, value: T, ttlSeconds: number = 3600): Promise<void> {
    const filePath = `${this.cacheDir}/${this.hashKey(key)}.json`;
    const content = JSON.stringify({
      value,
      expiresAt: Date.now() + ttlSeconds * 1000,
    });
    // await fs.writeFile(filePath, content, 'utf-8');
  }

  async delete(key: string): Promise<void> {
    // await fs.unlink(`${this.cacheDir}/${this.hashKey(key)}.json`);
  }

  async has(key: string): Promise<boolean> {
    return (await this.get(key)) !== null;
  }

  async clear(): Promise<void> {
    // cacheDir 内の全ファイルを削除
  }

  private hashKey(key: string): string {
    // キーをハッシュ化してファイル名に使える形式にする
    return key.replace(/[^a-zA-Z0-9]/g, "_");
  }
}

// 利用側: Cache<T> のみに依存
class UserService {
  constructor(
    private userRepo: any,
    private cache: Cache<User>,  // 具象を知らない
  ) {}

  async getUser(id: string): Promise<User | null> {
    // キャッシュから取得
    const cached = await this.cache.get(`user:${id}`);
    if (cached) return cached;

    // DBから取得してキャッシュ
    const user = await this.userRepo.findById(id);
    if (user) {
      await this.cache.set(`user:${id}`, user, 600); // 10分
    }
    return user;
  }

  async updateUser(id: string, data: Partial<User>): Promise<User> {
    const user = await this.userRepo.update(id, data);
    await this.cache.delete(`user:${id}`); // キャッシュを無効化
    return user;
  }
}

// 環境に応じて実装を切り替え
const cache = process.env.NODE_ENV === "test"
  ? new InMemoryCache<User>()   // テスト: インメモリ
  : new RedisCache<User>(redis); // 本番: Redis

const userService = new UserService(userRepo, cache);
```

```python
# レベル4: モジュール抽象化
# パッケージの公開APIのみを見せる

# payment/__init__.py
# 公開するものだけを __all__ で定義
# __all__ = ["PaymentService", "PaymentResult", "PaymentError"]

# 利用者は内部構造を知る必要がない:
# payment/
# ├── __init__.py          ← 公開API
# ├── service.py           ← PaymentService
# ├── result.py            ← PaymentResult
# ├── errors.py            ← PaymentError
# ├── providers/           ← 内部の実装詳細
# │   ├── stripe.py
# │   ├── paypay.py
# │   └── bank_transfer.py
# ├── validators/
# │   ├── card_validator.py
# │   └── amount_validator.py
# └── utils/
#     ├── currency.py
#     └── retry.py

# 利用者:
# from payment import PaymentService, PaymentResult
# → 内部の providers/, validators/, utils/ は知らない
```

---

## 2. インターフェース vs 抽象クラス

```
┌──────────────┬─────────────────┬─────────────────┐
│              │ インターフェース │ 抽象クラス       │
├──────────────┼─────────────────┼─────────────────┤
│ 実装         │ なし（契約のみ）│ 部分的に可能     │
├──────────────┼─────────────────┼─────────────────┤
│ フィールド   │ なし            │ あり             │
├──────────────┼─────────────────┼─────────────────┤
│ 多重         │ 複数実装可能    │ 単一継承のみ     │
├──────────────┼─────────────────┼─────────────────┤
│ 関係         │ can-do          │ is-a             │
├──────────────┼─────────────────┼─────────────────┤
│ 用途         │ 能力の定義      │ 共通実装の提供   │
├──────────────┼─────────────────┼─────────────────┤
│ 例           │ Serializable    │ AbstractList     │
│              │ Comparable      │ HttpServlet      │
└──────────────┴─────────────────┴─────────────────┘

選択基準:
  「何ができるか」を定義 → インターフェース
  「どう動くか」の共通部分を提供 → 抽象クラス
  迷ったら → インターフェース（より柔軟）
```

### 2.1 インターフェース設計の実践

```typescript
// TypeScript: インターフェースの実践

// 能力を表すインターフェース（細粒度）
interface Printable {
  print(): string;
}

interface Serializable {
  serialize(): string;
  deserialize(data: string): void;
}

interface Loggable {
  toLogString(): string;
}

interface Validatable {
  validate(): ValidationResult;
}

interface ValidationResult {
  isValid: boolean;
  errors: string[];
}

// 複数のインターフェースを実装
class Invoice implements Printable, Serializable, Loggable, Validatable {
  constructor(
    private id: string,
    private items: { name: string; price: number; quantity: number }[],
    private date: Date,
    private customerName: string,
  ) {}

  print(): string {
    const total = this.getTotal();
    const itemLines = this.items
      .map(item => `  ${item.name}: ¥${item.price} × ${item.quantity} = ¥${item.price * item.quantity}`)
      .join("\n");
    return [
      `═══════════════════════════`,
      `請求書 #${this.id}`,
      `日付: ${this.date.toLocaleDateString("ja-JP")}`,
      `顧客: ${this.customerName}`,
      `───────────────────────────`,
      itemLines,
      `───────────────────────────`,
      `合計: ¥${total.toLocaleString()}`,
      `═══════════════════════════`,
    ].join("\n");
  }

  serialize(): string {
    return JSON.stringify({
      id: this.id,
      items: this.items,
      date: this.date.toISOString(),
      customerName: this.customerName,
    });
  }

  deserialize(data: string): void {
    const parsed = JSON.parse(data);
    this.id = parsed.id;
    this.items = parsed.items;
    this.date = new Date(parsed.date);
    this.customerName = parsed.customerName;
  }

  toLogString(): string {
    return `[Invoice:${this.id}] customer=${this.customerName} items=${this.items.length} total=${this.getTotal()}`;
  }

  validate(): ValidationResult {
    const errors: string[] = [];
    if (!this.id) errors.push("請求書IDが未設定です");
    if (this.items.length === 0) errors.push("明細が空です");
    if (!this.customerName) errors.push("顧客名が未設定です");
    for (const item of this.items) {
      if (item.price < 0) errors.push(`${item.name}: 価格が負です`);
      if (item.quantity <= 0) errors.push(`${item.name}: 数量が0以下です`);
    }
    return { isValid: errors.length === 0, errors };
  }

  private getTotal(): number {
    return this.items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  }
}

// インターフェースを使った汎用関数
function printAll(items: Printable[]): void {
  items.forEach(item => console.log(item.print()));
}

function serializeAll(items: Serializable[]): string[] {
  return items.map(item => item.serialize());
}

function validateAll(items: Validatable[]): ValidationResult[] {
  return items.map(item => item.validate());
}
```

### 2.2 抽象クラスの実践（テンプレートメソッドパターン）

```python
# Python: 抽象クラス（ABC）
from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime


class DataStore(ABC):
    """データストアの抽象基底クラス"""

    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._connected = False
        self._query_count = 0

    # 共通実装
    def ensure_connected(self):
        if not self._connected:
            self.connect()
            self._connected = True

    # テンプレートメソッド（共通フロー）
    def save(self, key: str, value: Any) -> None:
        self.ensure_connected()
        self._validate(key, value)
        start = datetime.now()
        self._do_save(key, value)
        self._query_count += 1
        elapsed = (datetime.now() - start).total_seconds()
        self._log_operation("SAVE", key, elapsed)

    def load(self, key: str) -> Optional[Any]:
        self.ensure_connected()
        start = datetime.now()
        result = self._do_load(key)
        self._query_count += 1
        elapsed = (datetime.now() - start).total_seconds()
        self._log_operation("LOAD", key, elapsed)
        return result

    def delete(self, key: str) -> bool:
        self.ensure_connected()
        start = datetime.now()
        result = self._do_delete(key)
        self._query_count += 1
        elapsed = (datetime.now() - start).total_seconds()
        self._log_operation("DELETE", key, elapsed)
        return result

    def _validate(self, key: str, value: Any) -> None:
        if not key:
            raise ValueError("Key cannot be empty")
        if key.startswith("_"):
            raise ValueError("Key cannot start with underscore")

    def _log_operation(self, operation: str, key: str, elapsed: float) -> None:
        print(f"[{self.__class__.__name__}] {operation} '{key}' ({elapsed:.3f}s) "
              f"[total queries: {self._query_count}]")

    def get_stats(self) -> dict:
        return {
            "connected": self._connected,
            "query_count": self._query_count,
            "store_type": self.__class__.__name__,
        }

    # サブクラスが実装すべき抽象メソッド
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def _do_save(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def _do_load(self, key: str) -> Optional[Any]: ...

    @abstractmethod
    def _do_delete(self, key: str) -> bool: ...


class RedisStore(DataStore):
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self._data: dict[str, Any] = {}  # 簡易的な模擬実装

    def connect(self) -> None:
        print(f"Redis に接続: {self._connection_string}")

    def disconnect(self) -> None:
        print("Redis から切断")
        self._connected = False

    def _do_save(self, key: str, value: Any) -> None:
        self._data[key] = value

    def _do_load(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def _do_delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False


class FileStore(DataStore):
    def __init__(self, base_dir: str):
        super().__init__(base_dir)
        self._base_dir = base_dir

    def connect(self) -> None:
        print(f"ファイルストア初期化: {self._base_dir}")
        # os.makedirs(self._base_dir, exist_ok=True)

    def disconnect(self) -> None:
        print("ファイルストアクローズ")

    def _do_save(self, key: str, value: Any) -> None:
        import json
        # file_path = os.path.join(self._base_dir, f"{key}.json")
        # with open(file_path, "w") as f:
        #     json.dump(value, f)
        pass

    def _do_load(self, key: str) -> Optional[Any]:
        import json
        # file_path = os.path.join(self._base_dir, f"{key}.json")
        # try:
        #     with open(file_path, "r") as f:
        #         return json.load(f)
        # except FileNotFoundError:
        #     return None
        return None

    def _do_delete(self, key: str) -> bool:
        # file_path = os.path.join(self._base_dir, f"{key}.json")
        # try:
        #     os.remove(file_path)
        #     return True
        # except FileNotFoundError:
        #     return False
        return False


class PostgresStore(DataStore):
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self._pool = None

    def connect(self) -> None:
        print(f"PostgreSQL に接続: {self._connection_string}")
        # self._pool = asyncpg.create_pool(self._connection_string)

    def disconnect(self) -> None:
        print("PostgreSQL から切断")
        # await self._pool.close()

    def _do_save(self, key: str, value: Any) -> None:
        # INSERT OR UPDATE
        pass

    def _do_load(self, key: str) -> Optional[Any]:
        # SELECT WHERE key = ?
        return None

    def _do_delete(self, key: str) -> bool:
        # DELETE WHERE key = ?
        return False


# 利用側: DataStore のみに依存
def backup_data(source: DataStore, destination: DataStore, keys: list[str]) -> int:
    """データストア間でデータをコピー"""
    count = 0
    for key in keys:
        value = source.load(key)
        if value is not None:
            destination.save(key, value)
            count += 1
    return count


# 使用例
redis = RedisStore("redis://localhost:6379")
redis.save("user:001", {"name": "田中", "age": 30})
print(redis.load("user:001"))  # {'name': '田中', 'age': 30}
print(redis.get_stats())       # {'connected': True, 'query_count': 2, ...}
```

---

## 3. リーキー抽象化

```
Joel Spolsky の「リーキー抽象化の法則」(2002):
  「すべての重要な抽象化は、ある程度漏れている」

例:
  TCP/IP: 「信頼性のある通信」を抽象化
    → でもネットワーク遅延、パケットロスは隠しきれない
    → タイムアウト設定が必要 = 抽象化が漏れている

  ORM（Object-Relational Mapping）:
    → DBをオブジェクトとして抽象化
    → でも N+1 問題、JOIN の最適化は隠しきれない
    → SQL の知識が結局必要 = 抽象化が漏れている

  ファイルシステム:
    → 「ファイルは連続したバイト列」と抽象化
    → でもシーク時間、フラグメンテーションは存在する

  自動メモリ管理（GC）:
    → 「メモリ管理は不要」と抽象化
    → でも GC停止時間、メモリリーク（参照保持）は存在する
    → パフォーマンスチューニングにはGCの理解が必要

対策:
  1. 抽象化の下のレイヤーも理解しておく
  2. 抽象化が漏れるケースをドキュメント化
  3. エスケープハッチ（生のアクセス手段）を提供
  4. 抽象化のレベルを適切に設定する
```

### 3.1 ORM のリーキー抽象化

```typescript
// ORM のリーキー抽象化の例
class UserRepository {
  // 抽象化: オブジェクトとして操作
  async findUsersWithPosts(): Promise<User[]> {
    // ❌ N+1問題（抽象化が漏れる）
    const users = await User.findAll();
    for (const user of users) {
      user.posts = await Post.findByUserId(user.id); // N回のクエリ
    }
    return users;
    // → ユーザーが100人なら、101回のSQLクエリが発行される
    // → SELECT * FROM users; (1回)
    // → SELECT * FROM posts WHERE user_id = 1; (1回目)
    // → SELECT * FROM posts WHERE user_id = 2; (2回目)
    // → ...
    // → SELECT * FROM posts WHERE user_id = 100; (100回目)
  }

  // ✅ SQLの知識を使って最適化（抽象化の漏れに対処）
  async findUsersWithPostsOptimized(): Promise<User[]> {
    return await User.findAll({
      include: [{ model: Post }], // Eager loading（JOINに変換）
    });
    // → SELECT users.*, posts.* FROM users LEFT JOIN posts ON ...
    // → 1回のSQLクエリで完了
  }

  // ✅ さらに高度な最適化: 必要なカラムだけ取得
  async findUsersWithPostCount(): Promise<UserWithPostCount[]> {
    return await User.findAll({
      attributes: [
        "id",
        "name",
        "email",
        [sequelize.fn("COUNT", sequelize.col("posts.id")), "postCount"],
      ],
      include: [{
        model: Post,
        attributes: [], // Post のカラムは不要
      }],
      group: ["User.id"],
    });
    // → SELECT users.id, users.name, users.email, COUNT(posts.id) as postCount
    //   FROM users LEFT JOIN posts ON ... GROUP BY users.id
  }
}
```

### 3.2 HTTP クライアントのリーキー抽象化

```python
# HTTP クライアントの抽象化が漏れるケース
from abc import ABC, abstractmethod
from typing import Any, Optional
import time


class HttpClient(ABC):
    """HTTP クライアントの抽象化"""

    @abstractmethod
    def get(self, url: str, headers: Optional[dict] = None) -> "HttpResponse": ...

    @abstractmethod
    def post(self, url: str, body: Any, headers: Optional[dict] = None) -> "HttpResponse": ...


class HttpResponse:
    def __init__(self, status_code: int, body: Any, headers: dict):
        self.status_code = status_code
        self.body = body
        self.headers = headers

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300


class SimpleHttpClient(HttpClient):
    """単純な実装: 抽象化が多くを隠す"""

    def get(self, url: str, headers: Optional[dict] = None) -> HttpResponse:
        import urllib.request
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req) as response:
            return HttpResponse(
                status_code=response.status,
                body=response.read().decode(),
                headers=dict(response.headers),
            )

    def post(self, url: str, body: Any, headers: Optional[dict] = None) -> HttpResponse:
        import urllib.request
        import json
        data = json.dumps(body).encode() if isinstance(body, dict) else str(body).encode()
        req = urllib.request.Request(url, data=data, headers=headers or {})
        with urllib.request.urlopen(req) as response:
            return HttpResponse(
                status_code=response.status,
                body=response.read().decode(),
                headers=dict(response.headers),
            )


# ❌ 抽象化が漏れる場面:
# 1. タイムアウト: ネットワーク遅延は抽象化できない
# 2. リトライ: 一時的なエラーへの対処が必要
# 3. 接続プーリング: パフォーマンスのために必要
# 4. SSL/TLS: 証明書の検証方法
# 5. プロキシ: 企業環境での接続


# ✅ 改善版: 漏れを認識した上で適切に対処
class ResilientHttpClient(HttpClient):
    """レジリエントな実装: 抽象化の漏れに対処"""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_on_status: list[int] | None = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_on_status = retry_on_status or [429, 500, 502, 503, 504]

    def get(self, url: str, headers: Optional[dict] = None) -> HttpResponse:
        return self._request_with_retry("GET", url, headers=headers)

    def post(self, url: str, body: Any, headers: Optional[dict] = None) -> HttpResponse:
        return self._request_with_retry("POST", url, body=body, headers=headers)

    def _request_with_retry(
        self, method: str, url: str,
        body: Any = None, headers: Optional[dict] = None,
    ) -> HttpResponse:
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._do_request(method, url, body, headers)

                if response.status_code in self.retry_on_status and attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数バックオフ
                    print(f"[Retry] {method} {url} → {response.status_code}, "
                          f"リトライ {attempt + 1}/{self.max_retries} ({wait_time}s後)")
                    time.sleep(wait_time)
                    continue

                return response

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"[Retry] {method} {url} → エラー: {e}, "
                          f"リトライ {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)

        raise ConnectionError(f"全リトライ失敗: {last_error}")

    def _do_request(self, method: str, url: str,
                    body: Any = None, headers: Optional[dict] = None) -> HttpResponse:
        import urllib.request
        import json

        data = None
        if body is not None:
            data = json.dumps(body).encode() if isinstance(body, dict) else str(body).encode()

        req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return HttpResponse(
                status_code=response.status,
                body=response.read().decode(),
                headers=dict(response.headers),
            )
```

### 3.3 エスケープハッチの設計

```typescript
// エスケープハッチ: 抽象化の下に直接アクセスする手段を提供

interface Database {
  // 抽象化されたAPI
  findById<T>(table: string, id: string): Promise<T | null>;
  findAll<T>(table: string, where?: Record<string, any>): Promise<T[]>;
  insert<T>(table: string, data: T): Promise<string>;
  update<T>(table: string, id: string, data: Partial<T>): Promise<void>;
  delete(table: string, id: string): Promise<void>;

  // エスケープハッチ: 生のSQLを実行する手段
  rawQuery<T>(sql: string, params?: any[]): Promise<T[]>;
  rawExecute(sql: string, params?: any[]): Promise<number>;

  // トランザクション: 抽象化では隠せない操作
  transaction<T>(fn: (tx: Transaction) => Promise<T>): Promise<T>;
}

interface Transaction {
  findById<T>(table: string, id: string): Promise<T | null>;
  insert<T>(table: string, data: T): Promise<string>;
  update<T>(table: string, id: string, data: Partial<T>): Promise<void>;
  delete(table: string, id: string): Promise<void>;
  rawQuery<T>(sql: string, params?: any[]): Promise<T[]>;
}

class PostgresDatabase implements Database {
  // 抽象化API
  async findById<T>(table: string, id: string): Promise<T | null> {
    const results = await this.rawQuery<T>(
      `SELECT * FROM ${table} WHERE id = $1 LIMIT 1`,
      [id]
    );
    return results[0] || null;
  }

  async findAll<T>(table: string, where?: Record<string, any>): Promise<T[]> {
    if (!where || Object.keys(where).length === 0) {
      return this.rawQuery<T>(`SELECT * FROM ${table}`);
    }
    const conditions = Object.keys(where)
      .map((key, i) => `${key} = $${i + 1}`)
      .join(" AND ");
    return this.rawQuery<T>(
      `SELECT * FROM ${table} WHERE ${conditions}`,
      Object.values(where)
    );
  }

  // ... insert, update, delete の実装

  // エスケープハッチ: ORM では表現しにくいクエリに対応
  async rawQuery<T>(sql: string, params?: any[]): Promise<T[]> {
    // pg.query(sql, params)
    console.log(`SQL: ${sql}`, params);
    return [];
  }

  async rawExecute(sql: string, params?: any[]): Promise<number> {
    // pg.query(sql, params).rowCount
    return 0;
  }

  async transaction<T>(fn: (tx: Transaction) => Promise<T>): Promise<T> {
    await this.rawExecute("BEGIN");
    try {
      const result = await fn(this as unknown as Transaction);
      await this.rawExecute("COMMIT");
      return result;
    } catch (error) {
      await this.rawExecute("ROLLBACK");
      throw error;
    }
  }

  async insert<T>(table: string, data: T): Promise<string> { return ""; }
  async update<T>(table: string, id: string, data: Partial<T>): Promise<void> {}
  async delete(table: string, id: string): Promise<void> {}
}

// 利用例: 通常は抽象化APIを使い、必要な時だけエスケープハッチを使う
class ReportService {
  constructor(private db: Database) {}

  // 通常は抽象化APIで十分
  async getUser(id: string): Promise<User | null> {
    return this.db.findById<User>("users", id);
  }

  // 複雑なクエリにはエスケープハッチを使用
  async getMonthlyReport(year: number, month: number): Promise<Report[]> {
    return this.db.rawQuery<Report>(`
      SELECT
        u.name,
        COUNT(o.id) as order_count,
        SUM(o.total) as total_amount,
        AVG(o.total) as avg_amount
      FROM users u
      JOIN orders o ON u.id = o.user_id
      WHERE EXTRACT(YEAR FROM o.created_at) = $1
        AND EXTRACT(MONTH FROM o.created_at) = $2
      GROUP BY u.id, u.name
      HAVING COUNT(o.id) > 0
      ORDER BY total_amount DESC
    `, [year, month]);
  }
}
```

---

## 4. 良い抽象化の設計原則

```
1. 適切な粒度
   → 細かすぎ: 使いにくい（メソッドが多すぎる）
   → 粗すぎ: 柔軟性がない（何もカスタマイズできない）

2. 一貫性
   → 同じレベルの抽象度で統一
   → save() と write_bytes_to_disk() が混在しない

3. 最小驚き原則
   → 名前から想像できる動作をする
   → sort() が元の配列を破壊するのは驚き（Rubyの sort vs sort!）

4. 情報隠蔽
   → 知る必要のないことは隠す
   → ただしエスケープハッチは用意する

5. 単一レベルの抽象度
   → 1つのメソッド内で抽象度を混在させない
   → 高レベルの処理と低レベルの処理を分ける
```

### 4.1 抽象度の一貫性

```python
# ❌ 抽象度が混在している例
class OrderService:
    def process_order(self, order_data: dict) -> dict:
        # 高レベル: ビジネスロジック
        if order_data["total"] > 100000:
            discount = 0.1
        else:
            discount = 0

        # 低レベル: DB操作の詳細
        import psycopg2
        conn = psycopg2.connect("host=localhost dbname=mydb")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO orders (customer_id, total, discount) VALUES (%s, %s, %s)",
            (order_data["customer_id"], order_data["total"], discount),
        )
        conn.commit()

        # 低レベル: メール送信の詳細
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(f"ご注文ありがとうございます。合計: {order_data['total']}")
        msg["Subject"] = "注文確認"
        msg["From"] = "shop@example.com"
        msg["To"] = order_data["email"]
        smtp = smtplib.SMTP("localhost", 587)
        smtp.send_message(msg)
        smtp.quit()

        return {"status": "success"}


# ✅ 抽象度を統一した例
class OrderServiceV2:
    def __init__(
        self,
        discount_calculator: "DiscountCalculator",
        order_repository: "OrderRepository",
        notification_service: "NotificationService",
    ):
        self.discount_calculator = discount_calculator
        self.order_repository = order_repository
        self.notification_service = notification_service

    def process_order(self, order_data: dict) -> dict:
        """全て同じ抽象レベルの操作"""
        # 1. 割引を計算
        discount = self.discount_calculator.calculate(order_data)

        # 2. 注文を保存
        order = self.order_repository.save(
            customer_id=order_data["customer_id"],
            total=order_data["total"],
            discount=discount,
        )

        # 3. 通知を送信
        self.notification_service.send_order_confirmation(
            email=order_data["email"],
            order=order,
        )

        return {"status": "success", "order_id": order.id}


class DiscountCalculator:
    """割引計算のルールを集約"""
    def calculate(self, order_data: dict) -> float:
        if order_data["total"] > 100000:
            return 0.1
        if order_data.get("is_member"):
            return 0.05
        return 0
```

### 4.2 インターフェースの粒度設計

```typescript
// ❌ 粗すぎるインターフェース（God Interface）
interface DataManager {
  // ユーザー操作
  createUser(data: CreateUserDto): Promise<User>;
  updateUser(id: string, data: UpdateUserDto): Promise<User>;
  deleteUser(id: string): Promise<void>;
  findUserById(id: string): Promise<User | null>;

  // 商品操作
  createProduct(data: CreateProductDto): Promise<Product>;
  updateProduct(id: string, data: UpdateProductDto): Promise<Product>;
  deleteProduct(id: string): Promise<void>;
  findProductById(id: string): Promise<Product | null>;

  // 注文操作
  createOrder(data: CreateOrderDto): Promise<Order>;
  cancelOrder(id: string): Promise<void>;

  // レポート
  generateSalesReport(month: number): Promise<Report>;
  generateUserReport(): Promise<Report>;

  // 通知
  sendEmail(to: string, subject: string, body: string): Promise<void>;
  sendSms(to: string, message: string): Promise<void>;
}
// → 全ての機能が1つのインターフェースに集約
// → 利用者は不要な依存を強制される（ISP違反）


// ✅ 適切な粒度のインターフェース
interface UserRepository {
  create(data: CreateUserDto): Promise<User>;
  update(id: string, data: UpdateUserDto): Promise<User>;
  delete(id: string): Promise<void>;
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<User | null>;
}

interface ProductRepository {
  create(data: CreateProductDto): Promise<Product>;
  update(id: string, data: UpdateProductDto): Promise<Product>;
  delete(id: string): Promise<void>;
  findById(id: string): Promise<Product | null>;
  search(criteria: ProductSearchCriteria): Promise<Product[]>;
}

interface OrderService {
  create(data: CreateOrderDto): Promise<Order>;
  cancel(id: string): Promise<void>;
  findById(id: string): Promise<Order | null>;
}

interface ReportGenerator {
  generateSalesReport(period: DateRange): Promise<Report>;
  generateUserReport(filters?: UserReportFilters): Promise<Report>;
}

interface NotificationSender {
  send(notification: Notification): Promise<void>;
}

// 各サービスは必要なインターフェースのみに依存
class UserRegistrationService {
  constructor(
    private userRepo: UserRepository,           // ユーザー操作のみ
    private notifier: NotificationSender,       // 通知のみ
  ) {}

  async register(data: CreateUserDto): Promise<User> {
    const user = await this.userRepo.create(data);
    await this.notifier.send({
      type: "email",
      to: data.email,
      subject: "ようこそ！",
      body: `${data.name}さん、登録ありがとうございます。`,
    });
    return user;
  }
}
```

---

## 5. 抽象化とレイヤードアーキテクチャ

```
クリーンアーキテクチャにおける抽象化の層:

  ┌─────────────────────────────────────────┐
  │            Presentation Layer            │
  │  (Controller, View, API Endpoint)        │
  │  → ユーザー入出力の抽象化                 │
  ├─────────────────────────────────────────┤
  │            Application Layer             │
  │  (UseCase, Service, Command Handler)     │
  │  → ビジネスフローの抽象化                 │
  ├─────────────────────────────────────────┤
  │              Domain Layer                │
  │  (Entity, Value Object, Domain Service)  │
  │  → ビジネスルールの抽象化                 │
  ├─────────────────────────────────────────┤
  │          Infrastructure Layer            │
  │  (Repository Impl, External API Client)  │
  │  → 技術的な詳細の実装                     │
  └─────────────────────────────────────────┘

  依存の方向: 外側 → 内側
  → Infrastructure は Domain の定義したインターフェースを実装
  → 内側のレイヤーは外側を知らない
```

```typescript
// クリーンアーキテクチャの抽象化レイヤー例

// ======= Domain Layer（最内層）=======
// ビジネスルールを表現。技術的な詳細を一切知らない

interface ArticleRepository {
  findById(id: string): Promise<Article | null>;
  findByAuthor(authorId: string): Promise<Article[]>;
  save(article: Article): Promise<void>;
  delete(id: string): Promise<void>;
}

interface EventPublisher {
  publish(event: DomainEvent): Promise<void>;
}

class Article {
  constructor(
    public readonly id: string,
    public title: string,
    public content: string,
    public authorId: string,
    public status: "draft" | "published" | "archived",
    public readonly createdAt: Date,
    public updatedAt: Date,
  ) {}

  publish(): void {
    if (this.status !== "draft") {
      throw new Error("下書き以外は公開できません");
    }
    if (this.title.length === 0 || this.content.length === 0) {
      throw new Error("タイトルと本文は必須です");
    }
    this.status = "published";
    this.updatedAt = new Date();
  }

  archive(): void {
    if (this.status !== "published") {
      throw new Error("公開中の記事のみアーカイブできます");
    }
    this.status = "archived";
    this.updatedAt = new Date();
  }
}

interface DomainEvent {
  type: string;
  occurredAt: Date;
  data: Record<string, any>;
}

// ======= Application Layer =======
// ユースケースを調整。ドメインオブジェクトを使ってフローを実現

class PublishArticleUseCase {
  constructor(
    private articleRepo: ArticleRepository,   // インターフェースに依存
    private eventPublisher: EventPublisher,   // インターフェースに依存
  ) {}

  async execute(articleId: string, userId: string): Promise<void> {
    const article = await this.articleRepo.findById(articleId);
    if (!article) throw new Error("記事が見つかりません");
    if (article.authorId !== userId) throw new Error("権限がありません");

    article.publish(); // ドメインロジック

    await this.articleRepo.save(article);
    await this.eventPublisher.publish({
      type: "article.published",
      occurredAt: new Date(),
      data: { articleId, authorId: userId },
    });
  }
}

// ======= Infrastructure Layer（最外層）=======
// 技術的な詳細を実装

class PostgresArticleRepository implements ArticleRepository {
  constructor(private db: any) {}

  async findById(id: string): Promise<Article | null> {
    const row = await this.db.query("SELECT * FROM articles WHERE id = $1", [id]);
    return row ? this.toArticle(row) : null;
  }

  async findByAuthor(authorId: string): Promise<Article[]> {
    const rows = await this.db.query(
      "SELECT * FROM articles WHERE author_id = $1 ORDER BY created_at DESC",
      [authorId]
    );
    return rows.map(this.toArticle);
  }

  async save(article: Article): Promise<void> {
    await this.db.query(
      `INSERT INTO articles (id, title, content, author_id, status, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       ON CONFLICT (id) DO UPDATE SET
         title = $2, content = $3, status = $5, updated_at = $7`,
      [article.id, article.title, article.content, article.authorId,
       article.status, article.createdAt, article.updatedAt]
    );
  }

  async delete(id: string): Promise<void> {
    await this.db.query("DELETE FROM articles WHERE id = $1", [id]);
  }

  private toArticle(row: any): Article {
    return new Article(
      row.id, row.title, row.content, row.author_id,
      row.status, row.created_at, row.updated_at,
    );
  }
}

class KafkaEventPublisher implements EventPublisher {
  constructor(private producer: any) {}

  async publish(event: DomainEvent): Promise<void> {
    await this.producer.send({
      topic: event.type,
      messages: [{ value: JSON.stringify(event) }],
    });
  }
}

// ======= Presentation Layer =======
// HTTP リクエスト/レスポンスの変換

class ArticleController {
  constructor(private publishUseCase: PublishArticleUseCase) {}

  async publishArticle(req: Request, res: Response): Promise<void> {
    try {
      await this.publishUseCase.execute(req.params.id, req.user.id);
      res.status(200).json({ message: "記事を公開しました" });
    } catch (error) {
      if (error.message === "権限がありません") {
        res.status(403).json({ error: error.message });
      } else {
        res.status(400).json({ error: error.message });
      }
    }
  }
}
```

---

## 6. 抽象化のアンチパターン

```
アンチパターン1: 早すぎる抽象化（Premature Abstraction）
  → まだ1つしか実装がないのにインターフェースを作る
  → 「将来変わるかも」で不要な抽象化を導入
  → 対策: Rule of Three（3回繰り返したら抽象化）

アンチパターン2: 間違った抽象化（Wrong Abstraction）
  → 異なる概念を同じ抽象に無理やり合わせる
  → 条件分岐が増えて複雑化
  → 対策: 重複コードの方が間違った抽象化より良い

アンチパターン3: 抽象化の層が多すぎる（Over-layering）
  → Controller → Service → Manager → Helper → Repository → DAO
  → 各層がほぼパススルー
  → 対策: 実際に責任の分離が必要な層だけ設ける

アンチパターン4: 具象への逆依存
  → インターフェースが具象の実装詳細に引っ張られる
  → 例: ISqlDatabase インターフェースに executeSql() がある
  → 対策: 利用側の視点でインターフェースを設計
```

```python
# アンチパターン: 間違った抽象化（Wrong Abstraction）

# ❌ 異なる概念を1つの抽象に無理やり合わせる
class Notification:
    """全ての通知を1つのクラスで扱おうとする"""
    def __init__(self, type: str, recipient: str, content: str,
                 cc: list[str] = None, channel: str = None,
                 webhook_url: str = None, phone_number: str = None):
        self.type = type
        self.recipient = recipient
        self.content = content
        self.cc = cc                    # メールのみ
        self.channel = channel          # Slackのみ
        self.webhook_url = webhook_url  # Slackのみ
        self.phone_number = phone_number  # SMSのみ

    def send(self):
        if self.type == "email":
            # メール固有の処理
            self._send_email()
        elif self.type == "slack":
            # Slack固有の処理
            self._send_slack()
        elif self.type == "sms":
            # SMS固有の処理
            self._send_sms()
        elif self.type == "push":
            # プッシュ通知固有の処理
            self._send_push()
        # → 新しい通知タイプの追加 = 巨大なif-elseチェーンの追加
        # → 各タイプ固有のフィールドが混在して混乱


# ✅ 正しい抽象化: 共通点と相違点を見極める
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NotificationResult:
    success: bool
    message_id: str
    error: Optional[str] = None


class NotificationSender(ABC):
    """通知送信の共通インターフェース"""
    @abstractmethod
    def send(self, recipient: str, content: str) -> NotificationResult: ...

    @abstractmethod
    def supports(self, channel: str) -> bool: ...


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    sender: str

class EmailSender(NotificationSender):
    def __init__(self, config: EmailConfig):
        self.config = config

    def send(self, recipient: str, content: str,
             subject: str = "", cc: list[str] = None) -> NotificationResult:
        # メール固有のフィールドはこのクラスのみ
        # cc, subject はメール特有
        return NotificationResult(success=True, message_id="email-001")

    def supports(self, channel: str) -> bool:
        return channel == "email"


@dataclass
class SlackConfig:
    webhook_url: str
    default_channel: str

class SlackSender(NotificationSender):
    def __init__(self, config: SlackConfig):
        self.config = config

    def send(self, recipient: str, content: str,
             channel: str = None) -> NotificationResult:
        # Slack固有のフィールドはこのクラスのみ
        target_channel = channel or self.config.default_channel
        return NotificationResult(success=True, message_id="slack-001")

    def supports(self, channel: str) -> bool:
        return channel == "slack"


class SmsSender(NotificationSender):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def send(self, recipient: str, content: str) -> NotificationResult:
        # SMS固有の制約: 160文字制限
        if len(content) > 160:
            content = content[:157] + "..."
        return NotificationResult(success=True, message_id="sms-001")

    def supports(self, channel: str) -> bool:
        return channel == "sms"


# 利用側: NotificationSender のみに依存
class NotificationDispatcher:
    def __init__(self, senders: list[NotificationSender]):
        self.senders = senders

    def dispatch(self, channel: str, recipient: str, content: str) -> NotificationResult:
        for sender in self.senders:
            if sender.supports(channel):
                return sender.send(recipient, content)
        raise ValueError(f"未対応のチャンネル: {channel}")

    def broadcast(self, recipient: str, content: str) -> list[NotificationResult]:
        """全チャンネルに送信"""
        return [sender.send(recipient, content) for sender in self.senders]
```

---

## 7. Dependency Injection と抽象化

```
DI（依存性注入）は抽象化を活かすための仕組み:

  抽象化だけでは不十分:
    interface Logger { log(msg: string): void; }
    class UserService {
      private logger = new ConsoleLogger(); // ← 具象に依存!
    }
    → インターフェースを定義しても、内部で具象を生成していれば意味がない

  DI で解決:
    class UserService {
      constructor(private logger: Logger) {} // ← 外から注入
    }
    → テスト時に MockLogger、本番で CloudWatchLogger を注入可能
```

```typescript
// DI コンテナと抽象化の連携例

// インターフェース定義
interface Logger {
  info(message: string): void;
  error(message: string, error?: Error): void;
  warn(message: string): void;
}

interface UserRepository {
  findById(id: string): Promise<User | null>;
  save(user: User): Promise<void>;
}

interface EmailService {
  send(to: string, subject: string, body: string): Promise<void>;
}

// 本番用の実装
class CloudWatchLogger implements Logger {
  info(message: string): void {
    console.log(`[INFO] ${new Date().toISOString()} ${message}`);
  }
  error(message: string, error?: Error): void {
    console.error(`[ERROR] ${new Date().toISOString()} ${message}`, error);
  }
  warn(message: string): void {
    console.warn(`[WARN] ${new Date().toISOString()} ${message}`);
  }
}

// テスト用の実装
class MockLogger implements Logger {
  public logs: { level: string; message: string }[] = [];

  info(message: string): void {
    this.logs.push({ level: "info", message });
  }
  error(message: string): void {
    this.logs.push({ level: "error", message });
  }
  warn(message: string): void {
    this.logs.push({ level: "warn", message });
  }

  hasLog(level: string, messagePattern: string): boolean {
    return this.logs.some(
      log => log.level === level && log.message.includes(messagePattern)
    );
  }
}

// サービスクラス: 全てインターフェースに依存
class UserRegistrationService {
  constructor(
    private logger: Logger,
    private userRepo: UserRepository,
    private emailService: EmailService,
  ) {}

  async register(name: string, email: string): Promise<User> {
    this.logger.info(`ユーザー登録開始: ${email}`);

    // バリデーション
    const existing = await this.userRepo.findById(email);
    if (existing) {
      this.logger.warn(`既存ユーザー: ${email}`);
      throw new Error("このメールアドレスは既に登録されています");
    }

    // 保存
    const user = new User(crypto.randomUUID(), name, email);
    await this.userRepo.save(user);
    this.logger.info(`ユーザー保存完了: ${user.id}`);

    // 通知
    await this.emailService.send(
      email,
      "ようこそ！",
      `${name}さん、登録ありがとうございます。`
    );

    return user;
  }
}

// テストでの使用
async function testUserRegistration() {
  const mockLogger = new MockLogger();
  const mockRepo = new InMemoryUserRepository();
  const mockEmail = new MockEmailService();

  const service = new UserRegistrationService(mockLogger, mockRepo, mockEmail);

  const user = await service.register("田中", "tanaka@test.com");

  // アサーション
  assert(user.name === "田中");
  assert(mockLogger.hasLog("info", "ユーザー登録開始"));
  assert(mockEmail.sentEmails.length === 1);
  assert(mockEmail.sentEmails[0].to === "tanaka@test.com");
}
```

---

## 8. 関数型プログラミングにおける抽象化

```
OOPとFPの抽象化の違い:

  OOP: インターフェース/クラスで「振る舞い」を抽象化
    → Shape インターフェース → Circle, Rectangle
    → 新しい型の追加が容易

  FP: 関数で「操作」を抽象化
    → map, filter, reduce は型に依存しない
    → 新しい操作の追加が容易
```

```typescript
// 関数型のアプローチによる抽象化

// 高階関数による抽象化
type Predicate<T> = (item: T) => boolean;
type Mapper<T, U> = (item: T) => U;
type Reducer<T, U> = (acc: U, item: T) => U;

// 汎用的なパイプライン関数
function pipe<T>(...fns: ((value: T) => T)[]): (value: T) => T {
  return (value: T) => fns.reduce((acc, fn) => fn(acc), value);
}

// バリデーション: 関数の合成で表現
type Validator<T> = (value: T) => string | null; // null = OK

function createValidator<T>(...rules: Validator<T>[]): (value: T) => string[] {
  return (value: T) => {
    return rules
      .map(rule => rule(value))
      .filter((error): error is string => error !== null);
  };
}

// バリデーションルール（純粋関数）
const minLength = (min: number): Validator<string> =>
  (s) => s.length < min ? `${min}文字以上必要です` : null;

const maxLength = (max: number): Validator<string> =>
  (s) => s.length > max ? `${max}文字以下にしてください` : null;

const containsUpperCase: Validator<string> =
  (s) => /[A-Z]/.test(s) ? null : "大文字を含めてください";

const containsNumber: Validator<string> =
  (s) => /\d/.test(s) ? null : "数字を含めてください";

// バリデータの合成
const validatePassword = createValidator<string>(
  minLength(8),
  maxLength(128),
  containsUpperCase,
  containsNumber,
);

console.log(validatePassword("short"));
// ["8文字以上必要です", "大文字を含めてください", "数字を含めてください"]

console.log(validatePassword("MySecureP4ss"));
// []（エラーなし）

// データ変換パイプライン
interface RawUser {
  first_name: string;
  last_name: string;
  email_address: string;
  age: string;
}

interface ProcessedUser {
  fullName: string;
  email: string;
  age: number;
  isAdult: boolean;
}

// 変換関数の合成
const processUsers = (rawUsers: RawUser[]): ProcessedUser[] =>
  rawUsers
    .map(raw => ({
      fullName: `${raw.last_name} ${raw.first_name}`,
      email: raw.email_address.toLowerCase().trim(),
      age: parseInt(raw.age, 10),
      isAdult: parseInt(raw.age, 10) >= 18,
    }))
    .filter(user => !isNaN(user.age))
    .sort((a, b) => a.fullName.localeCompare(b.fullName, "ja"));
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 抽象化 | 複雑さを隠し本質のみ公開 |
| インターフェース | 能力（can-do）を定義。複数実装可 |
| 抽象クラス | 共通実装を提供。is-a 関係 |
| リーキー抽象化 | 全ての抽象化は漏れる。下層の理解も必要 |
| エスケープハッチ | 抽象化の下に直接アクセスする手段を提供 |
| 設計原則 | 適切な粒度、一貫性、最小驚き |
| DI | 抽象化を活かすための仕組み |
| レイヤー | 各層が適切な抽象レベルを担当 |

---

## 次に読むべきガイド
→ [[../02-design-principles/00-solid-overview.md]] -- SOLID原則

---

## 参考文献
1. Spolsky, J. "The Law of Leaky Abstractions." 2002.
2. Liskov, B. "Data Abstraction and Hierarchy." 1988.
3. Martin, R. "Clean Architecture." Prentice Hall, 2017.
4. Parnas, D. "On the Criteria To Be Used in Decomposing Systems into Modules." 1972.
5. Sandi Metz. "The Wrong Abstraction." blog, 2016.
6. Abelson, H. and Sussman, G. "Structure and Interpretation of Computer Programs." MIT Press, 1996.
