# パフォーマンスレビューガイド

## 概要

パフォーマンスレビューは、コードの実行速度、メモリ使用量、スケーラビリティを評価します。ボトルネックを特定し、効率的なコードを目指します。

## 目次

1. [時間計算量](#時間計算量)
2. [メモリ使用](#メモリ使用)
3. [データベースクエリ](#データベースクエリ)
4. [キャッシング](#キャッシング)
5. [非同期処理](#非同期処理)
6. [言語別最適化](#言語別最適化)

---

## 時間計算量

### Big O記法

```typescript
// O(1) - 定数時間
function getFirst<T>(array: T[]): T | undefined {
  return array[0];
}

// O(n) - 線形時間
function findMax(numbers: number[]): number {
  let max = numbers[0];
  for (const num of numbers) {
    if (num > max) max = num;
  }
  return max;
}

// ❌ Bad: O(n²) - 二次時間
function hasDuplicates(array: number[]): boolean {
  for (let i = 0; i < array.length; i++) {
    for (let j = i + 1; j < array.length; j++) {
      if (array[i] === array[j]) return true;
    }
  }
  return false;
}

// ✅ Good: O(n) - Set使用
function hasDuplicates(array: number[]): boolean {
  const seen = new Set<number>();
  for (const item of array) {
    if (seen.has(item)) return true;
    seen.add(item);
  }
  return false;
}

// ❌ Bad: O(n) - 配列内検索を繰り返し
function filterUsers(users: User[], validIds: string[]): User[] {
  return users.filter(user => validIds.includes(user.id));
  // includes()がO(n)で、filter()内でn回呼ばれるのでO(n²)
}

// ✅ Good: O(n) - Setで高速化
function filterUsers(users: User[], validIds: string[]): User[] {
  const validIdSet = new Set(validIds);  // O(n)
  return users.filter(user => validIdSet.has(user.id));  // O(n)
  // 合計 O(n)
}
```

### 不要な計算の削減

```python
# ❌ Bad: ループ内で繰り返し計算
def process_items(items):
    results = []
    for item in items:
        # 毎回同じ計算をしている
        discount_rate = calculate_discount_rate()
        price = item.price * (1 - discount_rate)
        results.append(price)
    return results

# ✅ Good: ループ外で計算
def process_items(items):
    discount_rate = calculate_discount_rate()  # 1回だけ
    results = []
    for item in items:
        price = item.price * (1 - discount_rate)
        results.append(price)
    return results

# ✅ Better: リスト内包表記
def process_items(items):
    discount_rate = calculate_discount_rate()
    return [item.price * (1 - discount_rate) for item in items]
```

### Early Return

```swift
// ❌ Bad: すべてをチェック
func validateUser(_ user: User) -> Bool {
    var isValid = true

    if user.email.isEmpty {
        isValid = false
    }

    if user.age < 18 {
        isValid = false
    }

    if !user.hasAcceptedTerms {
        isValid = false
    }

    return isValid
}

// ✅ Good: Early Return
func validateUser(_ user: User) -> Bool {
    if user.email.isEmpty { return false }
    if user.age < 18 { return false }
    if !user.hasAcceptedTerms { return false }
    return true
}

// ✅ Better: Guard文
func validateUser(_ user: User) -> Bool {
    guard !user.email.isEmpty else { return false }
    guard user.age >= 18 else { return false }
    guard user.hasAcceptedTerms else { return false }
    return true
}
```

---

## メモリ使用

### メモリリーク

```swift
// ❌ Bad: 循環参照によるメモリリーク
class ViewController: UIViewController {
    var dataLoader: DataLoader?

    override func viewDidLoad() {
        super.viewDidLoad()

        dataLoader = DataLoader()
        dataLoader?.onComplete = {
            self.updateUI()  // 強参照サイクル
        }
    }
}

class DataLoader {
    var onComplete: (() -> Void)?

    func load() {
        // ...
        onComplete?()
    }
}

// ✅ Good: weak selfで解決
class ViewController: UIViewController {
    var dataLoader: DataLoader?

    override func viewDidLoad() {
        super.viewDidLoad()

        dataLoader = DataLoader()
        dataLoader?.onComplete = { [weak self] in
            self?.updateUI()
        }
    }
}
```

### 大量データの処理

```go
// ❌ Bad: すべてをメモリに読み込む
func ProcessLargeFile(filename string) error {
    data, err := os.ReadFile(filename)  // 10GBのファイルだと...
    if err != nil {
        return err
    }

    lines := strings.Split(string(data), "\n")
    for _, line := range lines {
        process(line)
    }

    return nil
}

// ✅ Good: ストリーム処理
func ProcessLargeFile(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        line := scanner.Text()
        process(line)
    }

    return scanner.Err()
}

// ✅ バッチ処理
func ProcessLargeData(items []Item) error {
    const batchSize = 1000

    for i := 0; i < len(items); i += batchSize {
        end := i + batchSize
        if end > len(items) {
            end = len(items)
        }

        batch := items[i:end]
        if err := processBatch(batch); err != nil {
            return err
        }

        // GCのヒント
        batch = nil
        runtime.GC()
    }

    return nil
}
```

### 文字列結合

```typescript
// ❌ Bad: ループ内での文字列結合
function buildHTML(items: string[]): string {
  let html = '';
  for (const item of items) {
    html += '<li>' + item + '</li>';  // 毎回新しい文字列を作成
  }
  return html;
}

// ✅ Good: 配列でjoin
function buildHTML(items: string[]): string {
  return items.map(item => `<li>${item}</li>`).join('');
}

// ✅ Better: テンプレートリテラル
function buildHTML(items: string[]): string {
  return items.map(item => `<li>${item}</li>`).join('');
}
```

---

## データベースクエリ

### N+1 Problem

```python
# ❌ Bad: N+1問題
def get_users_with_posts():
    users = User.query.all()  # 1クエリ
    result = []

    for user in users:
        posts = Post.query.filter_by(user_id=user.id).all()  # Nクエリ
        result.append({
            'user': user,
            'posts': posts
        })

    return result
# 合計: 1 + N クエリ（N = ユーザー数）

# ✅ Good: Eager Loading
def get_users_with_posts():
    users = User.query.options(
        joinedload(User.posts)
    ).all()  # 1クエリ（JOIN使用）

    return [{
        'user': user,
        'posts': user.posts
    } for user in users]
# 合計: 1クエリ
```

### インデックス

```sql
-- ❌ Bad: インデックスなし
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255),
    created_at TIMESTAMP
);

SELECT * FROM users WHERE email = 'user@example.com';
-- Full table scan

-- ✅ Good: 適切なインデックス
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255),
    created_at TIMESTAMP
);

CREATE UNIQUE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at);

SELECT * FROM users WHERE email = 'user@example.com';
-- Index scan
```

### クエリの最適化

```typescript
// ❌ Bad: 不要なカラムを取得
const users = await db.query(`
  SELECT * FROM users
`);
// すべてのカラムを取得（大きなBLOBフィールドも含む）

// ✅ Good: 必要なカラムのみ取得
const users = await db.query(`
  SELECT id, name, email FROM users
`);

// ❌ Bad: ループ内でクエリ
async function getUsersWithRoles(userIds: string[]) {
  const results = [];
  for (const userId of userIds) {
    const user = await db.query(
      'SELECT * FROM users WHERE id = $1',
      [userId]
    );
    const roles = await db.query(
      'SELECT * FROM roles WHERE user_id = $1',
      [userId]
    );
    results.push({ user, roles });
  }
  return results;
}

// ✅ Good: 1クエリで取得
async function getUsersWithRoles(userIds: string[]) {
  return db.query(`
    SELECT
      u.*,
      json_agg(r.*) as roles
    FROM users u
    LEFT JOIN roles r ON r.user_id = u.id
    WHERE u.id = ANY($1)
    GROUP BY u.id
  `, [userIds]);
}
```

---

## キャッシング

### メモリキャッシュ

```go
// ✅ シンプルなキャッシュ
type Cache struct {
    mu    sync.RWMutex
    items map[string]CacheItem
}

type CacheItem struct {
    Value      interface{}
    Expiration int64
}

func (c *Cache) Set(key string, value interface{}, duration time.Duration) {
    c.mu.Lock()
    defer c.mu.Unlock()

    expiration := time.Now().Add(duration).UnixNano()
    c.items[key] = CacheItem{
        Value:      value,
        Expiration: expiration,
    }
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()

    item, found := c.items[key]
    if !found {
        return nil, false
    }

    if time.Now().UnixNano() > item.Expiration {
        return nil, false
    }

    return item.Value, true
}

// 使用例
var userCache = &Cache{items: make(map[string]CacheItem)}

func GetUser(id string) (*User, error) {
    // キャッシュチェック
    if cached, found := userCache.Get(id); found {
        return cached.(*User), nil
    }

    // DBから取得
    user, err := db.FindUser(id)
    if err != nil {
        return nil, err
    }

    // キャッシュに保存（5分間）
    userCache.Set(id, user, 5*time.Minute)

    return user, nil
}
```

### 計算結果のキャッシュ（メモ化）

```python
from functools import lru_cache

# ❌ Bad: 毎回計算
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# fibonacci(40) は数秒かかる

# ✅ Good: メモ化
@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# fibonacci(40) は瞬時に完了

# 手動でのメモ化
_cache = {}

def expensive_calculation(x, y):
    key = (x, y)
    if key in _cache:
        return _cache[key]

    # 重い計算
    result = complex_computation(x, y)

    _cache[key] = result
    return result
```

### HTTP キャッシュ

```typescript
// ✅ キャッシュヘッダーの設定
app.get('/api/products', (req, res) => {
  const products = getProducts();

  // 5分間キャッシュ
  res.set('Cache-Control', 'public, max-age=300');

  res.json(products);
});

// ETagを使用
app.get('/api/user/:id', (req, res) => {
  const user = getUser(req.params.id);
  const etag = generateETag(user);

  // クライアントのETagと比較
  if (req.headers['if-none-match'] === etag) {
    return res.status(304).send();  // Not Modified
  }

  res.set('ETag', etag);
  res.set('Cache-Control', 'private, max-age=300');
  res.json(user);
});
```

---

## 非同期処理

### 並列処理

```typescript
// ❌ Bad: 順次実行
async function loadUserData(userId: string) {
  const user = await fetchUser(userId);        // 100ms
  const posts = await fetchPosts(userId);      // 100ms
  const comments = await fetchComments(userId); // 100ms
  const likes = await fetchLikes(userId);      // 100ms

  return { user, posts, comments, likes };
}
// 合計: 400ms

// ✅ Good: 並列実行
async function loadUserData(userId: string) {
  const [user, posts, comments, likes] = await Promise.all([
    fetchUser(userId),
    fetchPosts(userId),
    fetchComments(userId),
    fetchLikes(userId),
  ]);

  return { user, posts, comments, likes };
}
// 合計: 100ms（最も遅い処理の時間）
```

### バックグラウンド処理

```python
# ❌ Bad: 同期処理で遅い
def create_user(email, password):
    user = User(email=email, password=hash_password(password))
    db.session.add(user)
    db.session.commit()

    # メール送信（3秒かかる）
    send_welcome_email(user.email)

    # サムネイル生成（5秒かかる）
    generate_thumbnail(user.avatar)

    return user
# 合計: 8秒以上

# ✅ Good: バックグラウンドタスク
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379')

@celery.task
def send_welcome_email_task(email):
    send_welcome_email(email)

@celery.task
def generate_thumbnail_task(avatar):
    generate_thumbnail(avatar)

def create_user(email, password):
    user = User(email=email, password=hash_password(password))
    db.session.add(user)
    db.session.commit()

    # バックグラウンドで実行
    send_welcome_email_task.delay(user.email)
    generate_thumbnail_task.delay(user.avatar)

    return user
# 合計: 数百ms（DB書き込みのみ）
```

---

## 言語別最適化

### TypeScript/JavaScript

```typescript
// ❌ Bad: 配列の再生成
function updateItems(items: Item[]): Item[] {
  let result = items.map(item => ({ ...item, processed: true }));
  result = result.filter(item => item.active);
  result = result.sort((a, b) => a.name.localeCompare(b.name));
  return result;
}

// ✅ Good: メソッドチェーン（中間配列なし）
function updateItems(items: Item[]): Item[] {
  return items
    .map(item => ({ ...item, processed: true }))
    .filter(item => item.active)
    .sort((a, b) => a.name.localeCompare(b.name));
}

// debounce/throttle
import { debounce } from 'lodash';

// ❌ Bad: 毎回実行
input.addEventListener('input', (e) => {
  searchAPI(e.target.value);  // 1文字ごとにAPI呼び出し
});

// ✅ Good: debounce
const debouncedSearch = debounce((value: string) => {
  searchAPI(value);
}, 300);

input.addEventListener('input', (e) => {
  debouncedSearch(e.target.value);  // 300ms後に実行
});
```

### Python

```python
# ジェネレーター
# ❌ Bad: リスト生成
def get_large_dataset():
    results = []
    for i in range(1000000):
        results.append(process(i))
    return results

data = get_large_dataset()  # 全データがメモリに
for item in data:
    use(item)

# ✅ Good: ジェネレーター
def get_large_dataset():
    for i in range(1000000):
        yield process(i)

for item in get_large_dataset():  # 1つずつ処理
    use(item)

# NumPy配列
import numpy as np

# ❌ Bad: Pythonリスト
data = list(range(1000000))
squared = [x ** 2 for x in data]  # 遅い

# ✅ Good: NumPy
data = np.arange(1000000)
squared = data ** 2  # 高速
```

### Swift

```swift
// Lazy evaluation
// ❌ Bad: すぐに評価
let numbers = Array(1...1000000)
let result = numbers
    .map { $0 * 2 }
    .filter { $0 % 3 == 0 }
    .prefix(10)
// すべての要素を処理してから10個取得

// ✅ Good: Lazy
let numbers = Array(1...1000000)
let result = numbers
    .lazy
    .map { $0 * 2 }
    .filter { $0 % 3 == 0 }
    .prefix(10)
// 10個見つかったら終了

// Copy-on-Write最適化
struct LargeData {
    private var storage: [Int]

    // Uniqueness checkでコピーを最小化
    mutating func modify() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = storage  // コピー
        }
        storage[0] = 42
    }
}
```

### Go

```go
// sync.Pool でメモリ再利用
var bufferPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func ProcessData(data []byte) ([]byte, error) {
    buf := bufferPool.Get().(*bytes.Buffer)
    defer func() {
        buf.Reset()
        bufferPool.Put(buf)
    }()

    // bufferを使用
    buf.Write(data)
    // ...

    return buf.Bytes(), nil
}

// Goroutine poolで並列処理
func ProcessItems(items []Item) error {
    const numWorkers = 10

    jobs := make(chan Item, len(items))
    results := make(chan error, len(items))

    // Worker起動
    for w := 0; w < numWorkers; w++ {
        go worker(jobs, results)
    }

    // ジョブ投入
    for _, item := range items {
        jobs <- item
    }
    close(jobs)

    // 結果収集
    for i := 0; i < len(items); i++ {
        if err := <-results; err != nil {
            return err
        }
    }

    return nil
}

func worker(jobs <-chan Item, results chan<- error) {
    for item := range jobs {
        results <- process(item)
    }
}
```

---

## レビューチェックリスト

### パフォーマンスレビュー完全チェックリスト

#### アルゴリズム
- [ ] 時間計算量が適切（O(n²)以下）
- [ ] 不要な計算がない
- [ ] Early Returnを使用

#### メモリ
- [ ] メモリリークがない
- [ ] 大量データを適切に処理
- [ ] 文字列結合が効率的

#### データベース
- [ ] N+1問題がない
- [ ] 適切なインデックス
- [ ] 必要なカラムのみ取得

#### キャッシング
- [ ] 適切にキャッシュされている
- [ ] キャッシュの有効期限が適切
- [ ] キャッシュキーが適切

#### 非同期
- [ ] 並列処理が使われている
- [ ] 重い処理がバックグラウンド化
- [ ] デッドロックの可能性がない

---

## まとめ

パフォーマンスは設計段階から考慮すべきです。

### 重要ポイント

1. **計測してから最適化**
2. **ボトルネックを特定**
3. **アルゴリズムの改善**
4. **適切なキャッシング**
5. **並列処理の活用**

### 次のステップ

- [保守性レビュー](07-maintainability.md)
- [セルフレビュー](08-self-review.md)
