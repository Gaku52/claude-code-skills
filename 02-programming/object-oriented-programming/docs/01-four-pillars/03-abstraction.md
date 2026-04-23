# Abstraction

> Abstraction is the principle of "hiding complexity and exposing only the essential features." The key points are interface design, how to use abstract classes, and avoiding "leaky abstractions."

## What you will learn in this chapter

- [ ] Understand the levels of abstraction and how to apply them
- [ ] Grasp the distinction between interfaces and abstract classes
- [ ] Learn about the leaky abstraction problem and how to avoid it
- [ ] Acquire the ability to distinguish good abstractions from bad ones
- [ ] Learn practical patterns for interface design in each language
- [ ] Understand the relationship with layered architecture in practice


## Prerequisites

Before reading this guide, the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding the content of [Polymorphism](./02-polymorphism.md)

---

## 1. Levels of Abstraction

```
Abstraction = "Hide unnecessary details and expose only the important information"

Level 1: Data abstraction
  -> Hide the internal representation (overlaps with encapsulation)
  -> Date class: hides whether internals are a timestamp or a year/month/day struct

Level 2: Procedural abstraction
  -> Confine processing details inside functions/methods
  -> array.sort(): hides the details of the sort algorithm

Level 3: Type abstraction (interfaces)
  -> Define only "what it can do"; hide "how it does it"
  -> Iterable: promises only that it can be iterated

Level 4: Module abstraction
  -> Expose only the public API of a package/module
  -> Hide the complexity of the internal classes

  +------------ What the user sees ------------+
  |  database.query("SELECT * FROM users")     |
  +--------------------------------------------+
                     | hides
  +------------ Internal complexity -----------+
  | Connection pool management                 |
  | SQL parsing -> query plan optimization     |
  | Index lookup -> page loading               |
  | Lock management -> transaction control     |
  | Serialization of the result set            |
  +--------------------------------------------+
```

### 1.1 Concrete examples of abstraction: practice at each level

```typescript
// Level 1: Data abstraction
// Hide the internal representation and provide a unified interface
class Temperature {
  // Internally stored in Kelvin (K), but users are unaware
  private kelvin: number;

  private constructor(kelvin: number) {
    if (kelvin < 0) throw new Error("Temperatures below absolute zero do not exist");
    this.kelvin = kelvin;
  }

  // Factory methods: construct from various units
  static fromCelsius(c: number): Temperature {
    return new Temperature(c + 273.15);
  }

  static fromFahrenheit(f: number): Temperature {
    return new Temperature((f - 32) * 5 / 9 + 273.15);
  }

  static fromKelvin(k: number): Temperature {
    return new Temperature(k);
  }

  // Retrieve in various units
  toCelsius(): number {
    return this.kelvin - 273.15;
  }

  toFahrenheit(): number {
    return (this.kelvin - 273.15) * 9 / 5 + 32;
  }

  toKelvin(): number {
    return this.kelvin;
  }

  // Comparison
  isHigherThan(other: Temperature): boolean {
    return this.kelvin > other.kelvin;
  }

  // Human-readable format
  toString(): string {
    return `${this.toCelsius().toFixed(1)}°C`;
  }
}

// The user does not need to know the internal representation (Kelvin)
const boiling = Temperature.fromCelsius(100);
const body = Temperature.fromFahrenheit(98.6);
console.log(boiling.toString());                  // 100.0°C
console.log(body.toFahrenheit());                 // 98.6
console.log(boiling.isHigherThan(body));           // true
```

```python
# Level 2: Procedural abstraction
# Confine complex processing inside meaningfully named functions
import hashlib
import secrets
import re
from typing import Optional


class PasswordManager:
    """Abstraction for password management"""

    SALT_LENGTH = 32
    HASH_ITERATIONS = 100_000
    MIN_PASSWORD_LENGTH = 8

    def hash_password(self, password: str) -> str:
        """Hash the password (details are hidden)"""
        # The user does not need to know the following details:
        # - How the salt is generated
        # - The hash algorithm (PBKDF2 + SHA256)
        # - The number of iterations
        # - The encoding format
        salt = secrets.token_hex(self.SALT_LENGTH)
        hash_value = self._compute_hash(password, salt)
        return f"{salt}:{hash_value}"

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password"""
        salt, expected_hash = stored_hash.split(":")
        actual_hash = self._compute_hash(password, salt)
        return secrets.compare_digest(actual_hash, expected_hash)

    def validate_strength(self, password: str) -> list[str]:
        """Validate password strength and return a list of issues"""
        errors = []
        if len(password) < self.MIN_PASSWORD_LENGTH:
            errors.append(f"At least {self.MIN_PASSWORD_LENGTH} characters are required")
        if not re.search(r"[A-Z]", password):
            errors.append("Include at least one uppercase letter")
        if not re.search(r"[a-z]", password):
            errors.append("Include at least one lowercase letter")
        if not re.search(r"\d", password):
            errors.append("Include at least one digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            errors.append("Include at least one special character")
        return errors

    def _compute_hash(self, password: str, salt: str) -> str:
        """Internal: hash computation (private method)"""
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            self.HASH_ITERATIONS,
        ).hex()


# The user simply calls hash/verify
pm = PasswordManager()
hashed = pm.hash_password("MySecureP@ss1")
print(pm.verify_password("MySecureP@ss1", hashed))  # True
print(pm.verify_password("wrong", hashed))            # False
```

```typescript
// Level 3: Type abstraction (interfaces)
// Define only "what it can do"

interface Cache<T> {
  get(key: string): Promise<T | null>;
  set(key: string, value: T, ttlSeconds?: number): Promise<void>;
  delete(key: string): Promise<void>;
  has(key: string): Promise<boolean>;
  clear(): Promise<void>;
}

// Redis implementation
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

// In-memory implementation (for testing)
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

// File system implementation
class FileCache<T> implements Cache<T> {
  constructor(private cacheDir: string) {}

  async get(key: string): Promise<T | null> {
    // Read JSON from file and check TTL
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
    // Delete all files in cacheDir
  }

  private hashKey(key: string): string {
    // Hash the key into a format usable as a filename
    return key.replace(/[^a-zA-Z0-9]/g, "_");
  }
}

// Consumer side: depends only on Cache<T>
class UserService {
  constructor(
    private userRepo: any,
    private cache: Cache<User>,  // Does not know the concrete type
  ) {}

  async getUser(id: string): Promise<User | null> {
    // Retrieve from cache
    const cached = await this.cache.get(`user:${id}`);
    if (cached) return cached;

    // Fetch from DB and cache
    const user = await this.userRepo.findById(id);
    if (user) {
      await this.cache.set(`user:${id}`, user, 600); // 10 minutes
    }
    return user;
  }

  async updateUser(id: string, data: Partial<User>): Promise<User> {
    const user = await this.userRepo.update(id, data);
    await this.cache.delete(`user:${id}`); // Invalidate cache
    return user;
  }
}

// Swap implementations based on environment
const cache = process.env.NODE_ENV === "test"
  ? new InMemoryCache<User>()   // Test: in-memory
  : new RedisCache<User>(redis); // Production: Redis

const userService = new UserService(userRepo, cache);
```

```python
# Level 4: Module abstraction
# Expose only the package's public API

# payment/__init__.py
# Define only what is exposed via __all__
# __all__ = ["PaymentService", "PaymentResult", "PaymentError"]

# The user does not need to know the internal structure:
# payment/
# ├── __init__.py          <- Public API
# ├── service.py           <- PaymentService
# ├── result.py            <- PaymentResult
# ├── errors.py            <- PaymentError
# ├── providers/           <- Internal implementation details
# │   ├── stripe.py
# │   ├── paypay.py
# │   └── bank_transfer.py
# ├── validators/
# │   ├── card_validator.py
# │   └── amount_validator.py
# └── utils/
#     ├── currency.py
#     └── retry.py

# User:
# from payment import PaymentService, PaymentResult
# -> Does not know the internal providers/, validators/, utils/
```

---

## 2. Interface vs. Abstract Class

```
+--------------+-----------------+------------------+
|              | Interface       | Abstract class   |
+--------------+-----------------+------------------+
| Implementation| None (contract)| Partially allowed|
+--------------+-----------------+------------------+
| Fields       | None            | Yes              |
+--------------+-----------------+------------------+
| Multiplicity | Multiple allowed| Single inheritance|
+--------------+-----------------+------------------+
| Relationship | can-do          | is-a             |
+--------------+-----------------+------------------+
| Purpose      | Defining capability | Providing common implementation |
+--------------+-----------------+------------------+
| Example      | Serializable    | AbstractList     |
|              | Comparable      | HttpServlet      |
+--------------+-----------------+------------------+

Selection criteria:
  Define "what it can do" -> Interface
  Provide the common parts of "how it behaves" -> Abstract class
  When in doubt -> Interface (more flexible)
```

### 2.1 Practical interface design

```typescript
// TypeScript: Interfaces in practice

// Interfaces representing capabilities (fine-grained)
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

// Implement multiple interfaces
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
      `===========================`,
      `Invoice #${this.id}`,
      `Date: ${this.date.toLocaleDateString("ja-JP")}`,
      `Customer: ${this.customerName}`,
      `---------------------------`,
      itemLines,
      `---------------------------`,
      `Total: ¥${total.toLocaleString()}`,
      `===========================`,
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
    if (!this.id) errors.push("Invoice ID is not set");
    if (this.items.length === 0) errors.push("Line items are empty");
    if (!this.customerName) errors.push("Customer name is not set");
    for (const item of this.items) {
      if (item.price < 0) errors.push(`${item.name}: price is negative`);
      if (item.quantity <= 0) errors.push(`${item.name}: quantity is zero or less`);
    }
    return { isValid: errors.length === 0, errors };
  }

  private getTotal(): number {
    return this.items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  }
}

// Generic functions using interfaces
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

### 2.2 Practical use of abstract classes (Template Method pattern)

```python
# Python: Abstract classes (ABC)
from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime


class DataStore(ABC):
    """Abstract base class for data stores"""

    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._connected = False
        self._query_count = 0

    # Common implementation
    def ensure_connected(self):
        if not self._connected:
            self.connect()
            self._connected = True

    # Template method (common flow)
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

    # Abstract methods that subclasses must implement
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
        self._data: dict[str, Any] = {}  # Simple mock implementation

    def connect(self) -> None:
        print(f"Connecting to Redis: {self._connection_string}")

    def disconnect(self) -> None:
        print("Disconnecting from Redis")
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
        print(f"Initializing file store: {self._base_dir}")
        # os.makedirs(self._base_dir, exist_ok=True)

    def disconnect(self) -> None:
        print("Closing file store")

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
        print(f"Connecting to PostgreSQL: {self._connection_string}")
        # self._pool = asyncpg.create_pool(self._connection_string)

    def disconnect(self) -> None:
        print("Disconnecting from PostgreSQL")
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


# Consumer side: depends only on DataStore
def backup_data(source: DataStore, destination: DataStore, keys: list[str]) -> int:
    """Copy data between data stores"""
    count = 0
    for key in keys:
        value = source.load(key)
        if value is not None:
            destination.save(key, value)
            count += 1
    return count


# Usage example
redis = RedisStore("redis://localhost:6379")
redis.save("user:001", {"name": "Tanaka", "age": 30})
print(redis.load("user:001"))  # {'name': 'Tanaka', 'age': 30}
print(redis.get_stats())       # {'connected': True, 'query_count': 2, ...}
```

---

## 3. Leaky Abstractions

```
Joel Spolsky's "Law of Leaky Abstractions" (2002):
  "All non-trivial abstractions, to some degree, are leaky."

Examples:
  TCP/IP: Abstracts "reliable communication"
    -> But it cannot fully hide network latency or packet loss
    -> Timeout settings are required = the abstraction is leaking

  ORM (Object-Relational Mapping):
    -> Abstracts the DB as objects
    -> But it cannot fully hide the N+1 problem or JOIN optimization
    -> SQL knowledge is ultimately required = the abstraction is leaking

  File systems:
    -> Abstract "a file is a sequence of bytes"
    -> But seek time and fragmentation exist

  Automatic memory management (GC):
    -> Abstracts away "memory management is unnecessary"
    -> But GC pause times and memory leaks (retained references) exist
    -> Performance tuning requires understanding of the GC

Countermeasures:
  1. Also understand the layers below the abstraction
  2. Document cases where the abstraction leaks
  3. Provide an escape hatch (a means of raw access)
  4. Set the level of abstraction appropriately
```

### 3.1 Leaky abstraction in ORM

```typescript
// Example of a leaky abstraction in ORM
class UserRepository {
  // Abstraction: manipulate as objects
  async findUsersWithPosts(): Promise<User[]> {
    // Bad: N+1 problem (the abstraction leaks)
    const users = await User.findAll();
    for (const user of users) {
      user.posts = await Post.findByUserId(user.id); // N queries
    }
    return users;
    // -> For 100 users, 101 SQL queries are issued
    // -> SELECT * FROM users; (1 time)
    // -> SELECT * FROM posts WHERE user_id = 1; (1st time)
    // -> SELECT * FROM posts WHERE user_id = 2; (2nd time)
    // -> ...
    // -> SELECT * FROM posts WHERE user_id = 100; (100th time)
  }

  // Good: optimize using SQL knowledge (address the leaky abstraction)
  async findUsersWithPostsOptimized(): Promise<User[]> {
    return await User.findAll({
      include: [{ model: Post }], // Eager loading (converted to JOIN)
    });
    // -> SELECT users.*, posts.* FROM users LEFT JOIN posts ON ...
    // -> Done in a single SQL query
  }

  // Good: even more advanced optimization - fetch only the necessary columns
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
        attributes: [], // Post columns are not needed
      }],
      group: ["User.id"],
    });
    // -> SELECT users.id, users.name, users.email, COUNT(posts.id) as postCount
    //   FROM users LEFT JOIN posts ON ... GROUP BY users.id
  }
}
```

### 3.2 Leaky abstraction in an HTTP client

```python
# Cases where the HTTP client abstraction leaks
from abc import ABC, abstractmethod
from typing import Any, Optional
import time


class HttpClient(ABC):
    """HTTP client abstraction"""

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
    """Simple implementation: the abstraction hides much"""

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


# Situations where the abstraction leaks:
# 1. Timeouts: network latency cannot be abstracted away
# 2. Retries: handling transient errors is needed
# 3. Connection pooling: needed for performance
# 4. SSL/TLS: certificate validation methods
# 5. Proxies: connections in enterprise environments


# Improved version: acknowledge the leaks and handle them appropriately
class ResilientHttpClient(HttpClient):
    """Resilient implementation: handle leaks in the abstraction"""

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
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"[Retry] {method} {url} -> {response.status_code}, "
                          f"retry {attempt + 1}/{self.max_retries} (after {wait_time}s)")
                    time.sleep(wait_time)
                    continue

                return response

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"[Retry] {method} {url} -> error: {e}, "
                          f"retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)

        raise ConnectionError(f"All retries failed: {last_error}")

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

### 3.3 Designing escape hatches

```typescript
// Escape hatch: provide a means to access the layer below the abstraction directly

interface Database {
  // Abstracted API
  findById<T>(table: string, id: string): Promise<T | null>;
  findAll<T>(table: string, where?: Record<string, any>): Promise<T[]>;
  insert<T>(table: string, data: T): Promise<string>;
  update<T>(table: string, id: string, data: Partial<T>): Promise<void>;
  delete(table: string, id: string): Promise<void>;

  // Escape hatch: a way to execute raw SQL
  rawQuery<T>(sql: string, params?: any[]): Promise<T[]>;
  rawExecute(sql: string, params?: any[]): Promise<number>;

  // Transactions: an operation the abstraction cannot hide
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
  // Abstracted API
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

  // ... implementations of insert, update, delete

  // Escape hatch: handle queries hard to express in ORM
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

// Usage example: normally use the abstracted API and only use the escape hatch when needed
class ReportService {
  constructor(private db: Database) {}

  // The abstracted API is usually sufficient
  async getUser(id: string): Promise<User | null> {
    return this.db.findById<User>("users", id);
  }

  // Use the escape hatch for complex queries
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

## 4. Design Principles for Good Abstractions

```
1. Appropriate granularity
   -> Too fine: hard to use (too many methods)
   -> Too coarse: inflexible (nothing can be customized)

2. Consistency
   -> Unify at the same level of abstraction
   -> Do not mix save() and write_bytes_to_disk()

3. Principle of least surprise
   -> Behave in a way the name implies
   -> It's surprising when sort() destructively modifies the original array (Ruby's sort vs sort!)

4. Information hiding
   -> Hide what does not need to be known
   -> But provide an escape hatch

5. Single level of abstraction
   -> Do not mix abstraction levels within one method
   -> Separate high-level and low-level operations
```

### 4.1 Consistency of abstraction level

```python
# Bad example: mixed abstraction levels
class OrderService:
    def process_order(self, order_data: dict) -> dict:
        # High level: business logic
        if order_data["total"] > 100000:
            discount = 0.1
        else:
            discount = 0

        # Low level: details of DB operations
        import psycopg2
        conn = psycopg2.connect("host=localhost dbname=mydb")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO orders (customer_id, total, discount) VALUES (%s, %s, %s)",
            (order_data["customer_id"], order_data["total"], discount),
        )
        conn.commit()

        # Low level: details of email sending
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(f"Thank you for your order. Total: {order_data['total']}")
        msg["Subject"] = "Order confirmation"
        msg["From"] = "shop@example.com"
        msg["To"] = order_data["email"]
        smtp = smtplib.SMTP("localhost", 587)
        smtp.send_message(msg)
        smtp.quit()

        return {"status": "success"}


# Good example: unified abstraction level
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
        """All operations at the same abstraction level"""
        # 1. Calculate discount
        discount = self.discount_calculator.calculate(order_data)

        # 2. Save the order
        order = self.order_repository.save(
            customer_id=order_data["customer_id"],
            total=order_data["total"],
            discount=discount,
        )

        # 3. Send notification
        self.notification_service.send_order_confirmation(
            email=order_data["email"],
            order=order,
        )

        return {"status": "success", "order_id": order.id}


class DiscountCalculator:
    """Aggregate discount calculation rules"""
    def calculate(self, order_data: dict) -> float:
        if order_data["total"] > 100000:
            return 0.1
        if order_data.get("is_member"):
            return 0.05
        return 0
```

### 4.2 Interface granularity design

```typescript
// Bad: too coarse an interface (God Interface)
interface DataManager {
  // User operations
  createUser(data: CreateUserDto): Promise<User>;
  updateUser(id: string, data: UpdateUserDto): Promise<User>;
  deleteUser(id: string): Promise<void>;
  findUserById(id: string): Promise<User | null>;

  // Product operations
  createProduct(data: CreateProductDto): Promise<Product>;
  updateProduct(id: string, data: UpdateProductDto): Promise<Product>;
  deleteProduct(id: string): Promise<void>;
  findProductById(id: string): Promise<Product | null>;

  // Order operations
  createOrder(data: CreateOrderDto): Promise<Order>;
  cancelOrder(id: string): Promise<void>;

  // Reports
  generateSalesReport(month: number): Promise<Report>;
  generateUserReport(): Promise<Report>;

  // Notifications
  sendEmail(to: string, subject: string, body: string): Promise<void>;
  sendSms(to: string, message: string): Promise<void>;
}
// -> All features aggregated into a single interface
// -> Consumers are forced into unnecessary dependencies (ISP violation)


// Good: interfaces with appropriate granularity
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

// Each service depends only on the interfaces it needs
class UserRegistrationService {
  constructor(
    private userRepo: UserRepository,           // Only user operations
    private notifier: NotificationSender,       // Only notifications
  ) {}

  async register(data: CreateUserDto): Promise<User> {
    const user = await this.userRepo.create(data);
    await this.notifier.send({
      type: "email",
      to: data.email,
      subject: "Welcome!",
      body: `${data.name}, thank you for registering.`,
    });
    return user;
  }
}
```

---

## 5. Abstraction and Layered Architecture

```
Abstraction layers in Clean Architecture:

  +-----------------------------------------+
  |            Presentation Layer            |
  |  (Controller, View, API Endpoint)        |
  |  -> Abstraction of user I/O              |
  +-----------------------------------------+
  |            Application Layer             |
  |  (UseCase, Service, Command Handler)     |
  |  -> Abstraction of the business flow     |
  +-----------------------------------------+
  |              Domain Layer                |
  |  (Entity, Value Object, Domain Service)  |
  |  -> Abstraction of business rules        |
  +-----------------------------------------+
  |          Infrastructure Layer            |
  |  (Repository Impl, External API Client)  |
  |  -> Implementation of technical details  |
  +-----------------------------------------+

  Direction of dependency: outside -> inside
  -> Infrastructure implements interfaces defined by Domain
  -> Inner layers do not know the outer ones
```

```typescript
// Example of Clean Architecture abstraction layers

// ======= Domain Layer (innermost) =======
// Expresses business rules. Does not know any technical details.

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
      throw new Error("Only drafts can be published");
    }
    if (this.title.length === 0 || this.content.length === 0) {
      throw new Error("Title and body are required");
    }
    this.status = "published";
    this.updatedAt = new Date();
  }

  archive(): void {
    if (this.status !== "published") {
      throw new Error("Only published articles can be archived");
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
// Orchestrates use cases. Realizes flows using domain objects.

class PublishArticleUseCase {
  constructor(
    private articleRepo: ArticleRepository,   // Depends on the interface
    private eventPublisher: EventPublisher,   // Depends on the interface
  ) {}

  async execute(articleId: string, userId: string): Promise<void> {
    const article = await this.articleRepo.findById(articleId);
    if (!article) throw new Error("Article not found");
    if (article.authorId !== userId) throw new Error("Not authorized");

    article.publish(); // Domain logic

    await this.articleRepo.save(article);
    await this.eventPublisher.publish({
      type: "article.published",
      occurredAt: new Date(),
      data: { articleId, authorId: userId },
    });
  }
}

// ======= Infrastructure Layer (outermost) =======
// Implements the technical details.

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
// Converts HTTP requests/responses.

class ArticleController {
  constructor(private publishUseCase: PublishArticleUseCase) {}

  async publishArticle(req: Request, res: Response): Promise<void> {
    try {
      await this.publishUseCase.execute(req.params.id, req.user.id);
      res.status(200).json({ message: "Article published" });
    } catch (error) {
      if (error.message === "Not authorized") {
        res.status(403).json({ error: error.message });
      } else {
        res.status(400).json({ error: error.message });
      }
    }
  }
}
```

---

## 6. Abstraction Anti-patterns

```
Anti-pattern 1: Premature Abstraction
  -> Creating an interface when there is still only one implementation
  -> Introducing unnecessary abstraction on the grounds that "it might change in the future"
  -> Remedy: Rule of Three (abstract after the third repetition)

Anti-pattern 2: Wrong Abstraction
  -> Forcing different concepts into the same abstraction
  -> Conditional branches proliferate and complexity grows
  -> Remedy: duplicated code is better than the wrong abstraction

Anti-pattern 3: Over-layering
  -> Controller -> Service -> Manager -> Helper -> Repository -> DAO
  -> Most layers are pass-through
  -> Remedy: set up only the layers where responsibility separation is actually needed

Anti-pattern 4: Reverse dependency on concretions
  -> The interface is dragged by the concrete implementation details
  -> Example: an ISqlDatabase interface has executeSql()
  -> Remedy: design the interface from the consumer's perspective
```

```python
# Anti-pattern: Wrong Abstraction

# Bad: forcing different concepts into a single abstraction
class Notification:
    """Try to handle all notifications in a single class"""
    def __init__(self, type: str, recipient: str, content: str,
                 cc: list[str] = None, channel: str = None,
                 webhook_url: str = None, phone_number: str = None):
        self.type = type
        self.recipient = recipient
        self.content = content
        self.cc = cc                    # Email only
        self.channel = channel          # Slack only
        self.webhook_url = webhook_url  # Slack only
        self.phone_number = phone_number  # SMS only

    def send(self):
        if self.type == "email":
            # Email-specific processing
            self._send_email()
        elif self.type == "slack":
            # Slack-specific processing
            self._send_slack()
        elif self.type == "sms":
            # SMS-specific processing
            self._send_sms()
        elif self.type == "push":
            # Push notification-specific processing
            self._send_push()
        # -> Adding a new notification type = adding to a huge if-else chain
        # -> Type-specific fields get mixed up and confused


# Good abstraction: identify the common and the different
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NotificationResult:
    success: bool
    message_id: str
    error: Optional[str] = None


class NotificationSender(ABC):
    """Common interface for notification senders"""
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
        # Email-specific fields only in this class
        # cc and subject are email-specific
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
        # Slack-specific fields only in this class
        target_channel = channel or self.config.default_channel
        return NotificationResult(success=True, message_id="slack-001")

    def supports(self, channel: str) -> bool:
        return channel == "slack"


class SmsSender(NotificationSender):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def send(self, recipient: str, content: str) -> NotificationResult:
        # SMS-specific constraint: 160-character limit
        if len(content) > 160:
            content = content[:157] + "..."
        return NotificationResult(success=True, message_id="sms-001")

    def supports(self, channel: str) -> bool:
        return channel == "sms"


# Consumer side: depends only on NotificationSender
class NotificationDispatcher:
    def __init__(self, senders: list[NotificationSender]):
        self.senders = senders

    def dispatch(self, channel: str, recipient: str, content: str) -> NotificationResult:
        for sender in self.senders:
            if sender.supports(channel):
                return sender.send(recipient, content)
        raise ValueError(f"Unsupported channel: {channel}")

    def broadcast(self, recipient: str, content: str) -> list[NotificationResult]:
        """Send to all channels"""
        return [sender.send(recipient, content) for sender in self.senders]
```

---

## 7. Dependency Injection and Abstraction

```
DI (Dependency Injection) is the mechanism that makes abstraction effective:

  Abstraction alone is not enough:
    interface Logger { log(msg: string): void; }
    class UserService {
      private logger = new ConsoleLogger(); // <- depends on the concrete!
    }
    -> Even if you define an interface, it is meaningless if a concrete is created internally

  Solved with DI:
    class UserService {
      constructor(private logger: Logger) {} // <- injected from outside
    }
    -> Can inject MockLogger in tests and CloudWatchLogger in production
```

```typescript
// Example of DI container and abstraction working together

// Interface definitions
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

// Production implementation
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

// Test implementation
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

// Service class: all dependencies are on interfaces
class UserRegistrationService {
  constructor(
    private logger: Logger,
    private userRepo: UserRepository,
    private emailService: EmailService,
  ) {}

  async register(name: string, email: string): Promise<User> {
    this.logger.info(`User registration started: ${email}`);

    // Validation
    const existing = await this.userRepo.findById(email);
    if (existing) {
      this.logger.warn(`Existing user: ${email}`);
      throw new Error("This email address is already registered");
    }

    // Save
    const user = new User(crypto.randomUUID(), name, email);
    await this.userRepo.save(user);
    this.logger.info(`User saved: ${user.id}`);

    // Notification
    await this.emailService.send(
      email,
      "Welcome!",
      `${name}, thank you for registering.`
    );

    return user;
  }
}

// Usage in tests
async function testUserRegistration() {
  const mockLogger = new MockLogger();
  const mockRepo = new InMemoryUserRepository();
  const mockEmail = new MockEmailService();

  const service = new UserRegistrationService(mockLogger, mockRepo, mockEmail);

  const user = await service.register("Tanaka", "tanaka@test.com");

  // Assertions
  assert(user.name === "Tanaka");
  assert(mockLogger.hasLog("info", "User registration started"));
  assert(mockEmail.sentEmails.length === 1);
  assert(mockEmail.sentEmails[0].to === "tanaka@test.com");
}
```

---

## 8. Abstraction in Functional Programming

```
The difference between OOP and FP abstraction:

  OOP: abstract "behavior" with interfaces/classes
    -> Shape interface -> Circle, Rectangle
    -> Easy to add new types

  FP: abstract "operations" with functions
    -> map, filter, reduce do not depend on a specific type
    -> Easy to add new operations
```

```typescript
// Abstraction with a functional approach

// Abstraction via higher-order functions
type Predicate<T> = (item: T) => boolean;
type Mapper<T, U> = (item: T) => U;
type Reducer<T, U> = (acc: U, item: T) => U;

// Generic pipeline function
function pipe<T>(...fns: ((value: T) => T)[]): (value: T) => T {
  return (value: T) => fns.reduce((acc, fn) => fn(acc), value);
}

// Validation: expressed via function composition
type Validator<T> = (value: T) => string | null; // null = OK

function createValidator<T>(...rules: Validator<T>[]): (value: T) => string[] {
  return (value: T) => {
    return rules
      .map(rule => rule(value))
      .filter((error): error is string => error !== null);
  };
}

// Validation rules (pure functions)
const minLength = (min: number): Validator<string> =>
  (s) => s.length < min ? `At least ${min} characters are required` : null;

const maxLength = (max: number): Validator<string> =>
  (s) => s.length > max ? `Please keep it within ${max} characters` : null;

const containsUpperCase: Validator<string> =
  (s) => /[A-Z]/.test(s) ? null : "Please include an uppercase letter";

const containsNumber: Validator<string> =
  (s) => /\d/.test(s) ? null : "Please include a digit";

// Composing validators
const validatePassword = createValidator<string>(
  minLength(8),
  maxLength(128),
  containsUpperCase,
  containsNumber,
);

console.log(validatePassword("short"));
// ["At least 8 characters are required", "Please include an uppercase letter", "Please include a digit"]

console.log(validatePassword("MySecureP4ss"));
// [] (no errors)

// Data transformation pipeline
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

// Composing transformation functions
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


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Writing actual code and observing its behavior, rather than only studying theory, deepens your understanding.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to applications. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architectural design.

---

## Summary

| Concept | Key point |
|------|---------|
| Abstraction | Hide complexity and expose only the essence |
| Interface | Define capability (can-do). Multiple implementations allowed |
| Abstract class | Provide common implementation. An is-a relationship |
| Leaky abstraction | All abstractions leak. Understanding the lower layers is also necessary |
| Escape hatch | Provide a means to access the layer below the abstraction directly |
| Design principles | Appropriate granularity, consistency, least surprise |
| DI | The mechanism for making abstraction effective |
| Layers | Each layer is responsible for the appropriate level of abstraction |

---

## Recommended next guides

---

## References
1. Spolsky, J. "The Law of Leaky Abstractions." 2002.
2. Liskov, B. "Data Abstraction and Hierarchy." 1988.
3. Martin, R. "Clean Architecture." Prentice Hall, 2017.
4. Parnas, D. "On the Criteria To Be Used in Decomposing Systems into Modules." 1972.
5. Sandi Metz. "The Wrong Abstraction." blog, 2016.
6. Abelson, H. and Sussman, G. "Structure and Interpretation of Computer Programs." MIT Press, 1996.
