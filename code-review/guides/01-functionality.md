# 機能性レビューガイド

## 概要

機能性レビューは、コードが意図した動作を正しく実装しているかを確認するレビューです。要件を満たしているか、エッジケースを考慮しているか、エラーハンドリングが適切かなど、コードの実際の動作に焦点を当てます。

## 目次

1. [要件の確認](#要件の確認)
2. [エッジケースの考慮](#エッジケースの考慮)
3. [エラーハンドリング](#エラーハンドリング)
4. [境界値テスト](#境界値テスト)
5. [言語別の注意点](#言語別の注意点)
6. [実装例と問題パターン](#実装例と問題パターン)

---

## 要件の確認

### 基本チェックリスト

- [ ] PRの説明に要件が明記されている
- [ ] コードが要件を満たしている
- [ ] 仕様変更がドキュメント化されている
- [ ] ユーザーストーリーが完了している
- [ ] 受け入れ基準を満たしている

### 要件との照合プロセス

#### 1. 要件の理解

```markdown
## 要件チェック

### 元の要件
- ユーザーがログインできる
- メールアドレスとパスワードで認証
- 2要素認証をサポート

### 実装内容
✅ メール/パスワード認証実装済み
✅ 2FA実装済み
❌ パスワードリセット機能（追加要件）
```

#### 2. 機能の動作確認

**TypeScript Example:**

```typescript
// ❌ Bad: 要件を満たしていない
class LoginService {
  async login(email: string, password: string): Promise<User> {
    // 2FA機能が実装されていない
    const user = await this.authenticate(email, password);
    return user;
  }
}

// ✅ Good: 要件を満たしている
class LoginService {
  async login(
    email: string,
    password: string,
    twoFactorCode?: string
  ): Promise<LoginResult> {
    const user = await this.authenticate(email, password);

    if (user.has2FAEnabled) {
      if (!twoFactorCode) {
        return {
          status: 'REQUIRES_2FA',
          userId: user.id
        };
      }

      const isValid = await this.verify2FA(user.id, twoFactorCode);
      if (!isValid) {
        throw new InvalidTwoFactorCodeError();
      }
    }

    return {
      status: 'SUCCESS',
      user,
      token: this.generateToken(user)
    };
  }
}
```

---

## エッジケースの考慮

### 一般的なエッジケース

#### 1. 空の入力

```typescript
// ❌ Bad: 空配列で失敗
function getFirst<T>(items: T[]): T {
  return items[0]; // 空配列でundefined
}

// ✅ Good: 空配列を考慮
function getFirst<T>(items: T[]): T | null {
  if (items.length === 0) {
    return null;
  }
  return items[0];
}

// ✅ Better: Optional型を返す
function getFirst<T>(items: T[]): T | undefined {
  return items[0]; // undefinedが明示的
}
```

#### 2. Null/Undefined

**TypeScript:**

```typescript
// ❌ Bad: Nullチェックなし
function getUserName(user: User): string {
  return user.profile.name; // userやprofileがnullだとエラー
}

// ✅ Good: Optional chainingとnullish coalescing
function getUserName(user: User | null): string {
  return user?.profile?.name ?? 'Unknown';
}
```

**Swift:**

```swift
// ❌ Bad: 強制アンラップ
func getUserName(user: User?) -> String {
    return user!.profile!.name // クラッシュの危険
}

// ✅ Good: Optional binding
func getUserName(user: User?) -> String {
    guard let user = user,
          let profile = user.profile,
          let name = profile.name else {
        return "Unknown"
    }
    return name
}

// ✅ Better: Optional chaining
func getUserName(user: User?) -> String {
    return user?.profile?.name ?? "Unknown"
}
```

#### 3. 境界値

```python
# ❌ Bad: 境界値チェックなし
def get_page(items, page_number, page_size=10):
    start = page_number * page_size
    end = start + page_size
    return items[start:end]  # 範囲外アクセスの可能性

# ✅ Good: 境界値チェック
def get_page(items, page_number, page_size=10):
    if page_number < 0:
        raise ValueError("Page number must be non-negative")

    if page_size <= 0:
        raise ValueError("Page size must be positive")

    start = page_number * page_size
    end = start + page_size

    # Pythonのスライスは範囲外でもエラーにならないが、
    # 明示的にチェックすることで意図を明確にする
    if start >= len(items):
        return []

    return items[start:end]
```

#### 4. 大量データ

```go
// ❌ Bad: メモリ効率が悪い
func ProcessAllUsers(users []User) error {
    results := make([]Result, len(users))
    for i, user := range users {
        results[i] = process(user)
    }
    return saveAll(results) // 大量データで問題
}

// ✅ Good: バッチ処理
func ProcessAllUsers(users []User) error {
    const batchSize = 100

    for i := 0; i < len(users); i += batchSize {
        end := i + batchSize
        if end > len(users) {
            end = len(users)
        }

        batch := users[i:end]
        if err := processBatch(batch); err != nil {
            return fmt.Errorf("failed to process batch %d: %w", i/batchSize, err)
        }
    }

    return nil
}
```

### エッジケースチェックリスト

#### 数値関連
- [ ] ゼロ
- [ ] 負の数
- [ ] 最大値/最小値
- [ ] オーバーフロー
- [ ] 浮動小数点誤差

#### 文字列関連
- [ ] 空文字列
- [ ] 非常に長い文字列
- [ ] 特殊文字（絵文字、制御文字）
- [ ] エンコーディング問題

#### コレクション関連
- [ ] 空のコレクション
- [ ] 単一要素
- [ ] 大量の要素
- [ ] Null要素を含む

#### タイミング関連
- [ ] 同時実行
- [ ] タイムアウト
- [ ] レースコンディション

---

## エラーハンドリング

### エラーハンドリングの原則

1. **適切なエラー型を使用する**
2. **エラーメッセージは具体的に**
3. **エラーを握りつぶさない**
4. **リソースは必ずクリーンアップ**

### 言語別エラーハンドリング

#### TypeScript

```typescript
// ❌ Bad: エラー情報が失われる
async function fetchUser(id: string): Promise<User> {
  try {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
  } catch (error) {
    console.error('Error fetching user'); // エラー詳細が失われる
    return null; // Nullを返すとエラーが隠蔽される
  }
}

// ✅ Good: カスタムエラーで詳細を保持
class UserFetchError extends Error {
  constructor(
    public userId: string,
    public statusCode?: number,
    message?: string
  ) {
    super(message || `Failed to fetch user ${userId}`);
    this.name = 'UserFetchError';
  }
}

async function fetchUser(id: string): Promise<User> {
  try {
    const response = await fetch(`/api/users/${id}`);

    if (!response.ok) {
      throw new UserFetchError(
        id,
        response.status,
        `HTTP ${response.status}: ${response.statusText}`
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof UserFetchError) {
      throw error;
    }
    // ネットワークエラーなど
    throw new UserFetchError(id, undefined, String(error));
  }
}
```

#### Python

```python
# ❌ Bad: 広範囲のtry/exceptとエラー握りつぶし
def process_data(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
            result = transform(data)
            validate(result)
            save(result)
            return result
    except Exception:  # あらゆるエラーをキャッチ
        return None  # エラーを隠蔽

# ✅ Good: 具体的なエラーハンドリング
class DataProcessingError(Exception):
    """データ処理エラーのベースクラス"""
    pass

class DataValidationError(DataProcessingError):
    """データ検証エラー"""
    pass

def process_data(file_path):
    """
    データファイルを処理する

    Raises:
        FileNotFoundError: ファイルが存在しない
        json.JSONDecodeError: JSONが不正
        DataValidationError: データが無効
    """
    try:
        with open(file_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise DataProcessingError(
            f"Invalid JSON in {file_path}: {e}"
        ) from e

    try:
        result = transform(data)
        validate(result)
    except ValueError as e:
        raise DataValidationError(f"Validation failed: {e}") from e

    try:
        save(result)
    except IOError as e:
        raise DataProcessingError(f"Failed to save result: {e}") from e

    return result
```

#### Swift

```swift
// ❌ Bad: try!による強制的なエラー無視
func loadConfig() -> Config {
    let data = try! Data(contentsOf: configURL)  // クラッシュの危険
    return try! JSONDecoder().decode(Config.self, from: data)
}

// ✅ Good: 適切なエラー伝播
enum ConfigError: Error {
    case fileNotFound(URL)
    case invalidFormat(String)
    case decodingFailed(Error)
}

func loadConfig() throws -> Config {
    guard FileManager.default.fileExists(atPath: configURL.path) else {
        throw ConfigError.fileNotFound(configURL)
    }

    let data: Data
    do {
        data = try Data(contentsOf: configURL)
    } catch {
        throw ConfigError.invalidFormat("Failed to read file: \(error)")
    }

    do {
        return try JSONDecoder().decode(Config.self, from: data)
    } catch {
        throw ConfigError.decodingFailed(error)
    }
}

// 使用例
func setupApp() {
    do {
        let config = try loadConfig()
        app.configure(with: config)
    } catch ConfigError.fileNotFound(let url) {
        print("Config file not found at: \(url)")
        app.useDefaultConfig()
    } catch ConfigError.invalidFormat(let message) {
        print("Config file is invalid: \(message)")
        app.useDefaultConfig()
    } catch {
        print("Unexpected error: \(error)")
        app.useDefaultConfig()
    }
}
```

#### Go

```go
// ❌ Bad: エラーを無視
func GetUser(id string) *User {
    user, _ := db.FindUser(id)  // エラー無視
    return user
}

// ✅ Good: エラーを適切に処理
type UserNotFoundError struct {
    UserID string
}

func (e *UserNotFoundError) Error() string {
    return fmt.Sprintf("user not found: %s", e.UserID)
}

func GetUser(id string) (*User, error) {
    if id == "" {
        return nil, errors.New("user ID is required")
    }

    user, err := db.FindUser(id)
    if err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, &UserNotFoundError{UserID: id}
        }
        return nil, fmt.Errorf("failed to get user %s: %w", id, err)
    }

    return user, nil
}

// エラーハンドリングのベストプラクティス
func ProcessUser(id string) error {
    user, err := GetUser(id)
    if err != nil {
        var notFoundErr *UserNotFoundError
        if errors.As(err, &notFoundErr) {
            // ユーザーが見つからない場合の処理
            log.Printf("Creating new user: %s", id)
            return createUser(id)
        }
        // その他のエラー
        return fmt.Errorf("user processing failed: %w", err)
    }

    return updateUser(user)
}
```

---

## 境界値テスト

### 境界値の種類

#### 1. 数値の境界値

```typescript
// 境界値テストのためのヘルパー
function validateAge(age: number): boolean {
  // テストすべき境界値:
  // - 0 (最小値)
  // - 1 (最小値+1)
  // - 17 (未成年/成人の境界-1)
  // - 18 (成人)
  // - 120 (最大値)
  // - 121 (最大値+1)
  // - -1 (不正な値)

  if (age < 0) {
    throw new Error('Age cannot be negative');
  }

  if (age > 120) {
    throw new Error('Age is too large');
  }

  return age >= 18;
}

// テストケース
describe('validateAge', () => {
  test('boundary values', () => {
    expect(() => validateAge(-1)).toThrow(); // 境界外
    expect(validateAge(0)).toBe(false);      // 最小値
    expect(validateAge(1)).toBe(false);      // 最小値+1
    expect(validateAge(17)).toBe(false);     // 境界-1
    expect(validateAge(18)).toBe(true);      // 境界
    expect(validateAge(19)).toBe(true);      // 境界+1
    expect(validateAge(120)).toBe(true);     // 最大値
    expect(() => validateAge(121)).toThrow(); // 境界外
  });
});
```

#### 2. 配列の境界値

```python
def get_items(items: list, start: int, end: int) -> list:
    """
    境界値テストケース:
    - 空配列
    - start=0, end=0 (空範囲)
    - start=0, end=1 (最小範囲)
    - start=len-1, end=len (最後の1要素)
    - start=0, end=len (全要素)
    - start=-1 (不正)
    - end > len (範囲外)
    """
    if start < 0 or end < 0:
        raise ValueError("Indices must be non-negative")

    if start > len(items):
        raise ValueError("Start index out of range")

    if end > len(items):
        raise ValueError("End index out of range")

    if start > end:
        raise ValueError("Start must be <= end")

    return items[start:end]
```

---

## 言語別の注意点

### TypeScript/JavaScript

#### 型の問題

```typescript
// ❌ Bad: 暗黙的なany
function process(data) {  // any型
  return data.value;
}

// ✅ Good: 明示的な型定義
function process(data: { value: string }): string {
  return data.value;
}

// ❌ Bad: 型アサーションの乱用
const user = JSON.parse(jsonString) as User;  // 実際の構造が違う可能性

// ✅ Good: ランタイム検証
function parseUser(jsonString: string): User {
  const data = JSON.parse(jsonString);

  if (!isValidUser(data)) {
    throw new Error('Invalid user data');
  }

  return data as User;
}

function isValidUser(data: any): data is User {
  return (
    typeof data === 'object' &&
    typeof data.id === 'string' &&
    typeof data.email === 'string'
  );
}
```

### Python

#### Mutable Default Arguments

```python
# ❌ Bad: ミュータブルなデフォルト引数
def add_item(item, items=[]):
    items.append(item)
    return items

# 問題: デフォルト引数は1回だけ評価される
add_item(1)  # [1]
add_item(2)  # [1, 2]  期待: [2]

# ✅ Good: Noneをデフォルトに
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### Swift

#### Optional型の扱い

```swift
// ❌ Bad: 暗黙的アンラップ
var user: User!
func setup() {
    user = loadUser()
}
func getName() -> String {
    return user.name  // setupが呼ばれていないとクラッシュ
}

// ✅ Good: Optional型を適切に使用
var user: User?
func setup() {
    user = loadUser()
}
func getName() -> String? {
    return user?.name
}
```

### Go

#### エラーチェック

```go
// ❌ Bad: エラーチェック漏れ
func ProcessFile(path string) {
    data, _ := os.ReadFile(path)
    process(data)  // dataがnilの可能性
}

// ✅ Good: すべてのエラーをチェック
func ProcessFile(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        return fmt.Errorf("failed to read file: %w", err)
    }

    if err := process(data); err != nil {
        return fmt.Errorf("failed to process data: %w", err)
    }

    return nil
}
```

---

## 実装例と問題パターン

### パターン1: 不完全な入力検証

```typescript
// ❌ Bad
class UserService {
  createUser(email: string, password: string): User {
    // メールアドレスの形式チェックのみ
    if (!email.includes('@')) {
      throw new Error('Invalid email');
    }

    return this.db.save({ email, password });
  }
}

// ✅ Good
class UserService {
  createUser(email: string, password: string): User {
    // 包括的な検証
    this.validateEmail(email);
    this.validatePassword(password);
    this.checkDuplicateEmail(email);

    const hashedPassword = this.hashPassword(password);
    return this.db.save({ email, password: hashedPassword });
  }

  private validateEmail(email: string): void {
    if (!email) {
      throw new ValidationError('Email is required');
    }

    // RFC 5322準拠の正規表現（簡略版）
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      throw new ValidationError('Invalid email format');
    }

    if (email.length > 254) {
      throw new ValidationError('Email is too long');
    }
  }

  private validatePassword(password: string): void {
    if (!password) {
      throw new ValidationError('Password is required');
    }

    if (password.length < 8) {
      throw new ValidationError('Password must be at least 8 characters');
    }

    if (!/[A-Z]/.test(password)) {
      throw new ValidationError('Password must contain uppercase letter');
    }

    if (!/[a-z]/.test(password)) {
      throw new ValidationError('Password must contain lowercase letter');
    }

    if (!/[0-9]/.test(password)) {
      throw new ValidationError('Password must contain number');
    }
  }
}
```

### パターン2: レースコンディション

```typescript
// ❌ Bad: レースコンディションの可能性
class Counter {
  private count = 0;

  async increment(): Promise<void> {
    const current = this.count;
    await this.delay(10);  // 非同期処理
    this.count = current + 1;  // 他のincrementと競合
  }
}

// ✅ Good: ロックメカニズム
class Counter {
  private count = 0;
  private locked = false;

  async increment(): Promise<void> {
    await this.acquireLock();
    try {
      this.count++;
    } finally {
      this.releaseLock();
    }
  }

  private async acquireLock(): Promise<void> {
    while (this.locked) {
      await this.delay(1);
    }
    this.locked = true;
  }

  private releaseLock(): void {
    this.locked = false;
  }
}
```

### パターン3: メモリリーク

```swift
// ❌ Bad: 循環参照によるメモリリーク
class ViewController: UIViewController {
    var onDataLoaded: (() -> Void)?

    override func viewDidLoad() {
        super.viewDidLoad()

        dataService.loadData { [self] data in
            self.updateUI(data)
            self.onDataLoaded?()  // selfへの強参照
        }
    }
}

// ✅ Good: weak selfでメモリリーク防止
class ViewController: UIViewController {
    var onDataLoaded: (() -> Void)?

    override func viewDidLoad() {
        super.viewDidLoad()

        dataService.loadData { [weak self] data in
            guard let self = self else { return }
            self.updateUI(data)
            self.onDataLoaded?()
        }
    }
}
```

---

## レビューチェックリスト

### 機能性レビュー完全チェックリスト

#### 要件
- [ ] PR説明に要件が明記されている
- [ ] すべての要件が実装されている
- [ ] 追加された機能が文書化されている
- [ ] 破壊的変更が明示されている

#### エッジケース
- [ ] 空の入力が考慮されている
- [ ] Null/Undefinedが処理されている
- [ ] 境界値がテストされている
- [ ] 大量データが考慮されている

#### エラーハンドリング
- [ ] すべてのエラーケースが処理されている
- [ ] エラーメッセージが具体的
- [ ] エラーが握りつぶされていない
- [ ] リソースが適切にクリーンアップされている

#### データ整合性
- [ ] データの検証が行われている
- [ ] トランザクションが適切に使用されている
- [ ] 同時実行の問題が考慮されている

#### セキュリティ
- [ ] 入力値が検証されている
- [ ] SQLインジェクション対策がある
- [ ] XSS対策がある
- [ ] 認証・認可が適切

---

## 機能性レビューのケーススタディ

### ケース1: ユーザー登録機能のレビュー

#### 問題のあるコード

```typescript
async function registerUser(email: string, password: string) {
  const user = await database.users.create({
    email,
    password, // パスワードが平文で保存されている！
  });
  return user;
}
```

#### レビュー指摘事項

1. **セキュリティ**: パスワードがハッシュ化されていない
2. **バリデーション**: メールアドレスの形式チェックがない
3. **エラーハンドリング**: 重複メールアドレスの処理がない
4. **要件**: パスワード強度チェックがない

#### 改善版

```typescript
async function registerUser(email: string, password: string) {
  // 1. バリデーション
  if (!isValidEmail(email)) {
    throw new ValidationError('Invalid email format');
  }

  if (!isStrongPassword(password)) {
    throw new ValidationError('Password must be at least 8 characters with uppercase, lowercase, and numbers');
  }

  // 2. 重複チェック
  const existingUser = await database.users.findByEmail(email);
  if (existingUser) {
    throw new ConflictError('Email already registered');
  }

  // 3. パスワードハッシュ化
  const hashedPassword = await bcrypt.hash(password, 10);

  // 4. ユーザー作成
  try {
    const user = await database.users.create({
      email,
      password: hashedPassword,
      createdAt: new Date(),
    });

    // 5. ウェルカムメール送信（エラーは握りつぶす）
    await sendWelcomeEmail(email).catch(err =>
      logger.error('Failed to send welcome email', err)
    );

    return user;
  } catch (error) {
    logger.error('User registration failed', error);
    throw new DatabaseError('Failed to create user');
  }
}
```

### ケース2: 決済処理のレビュー

#### 問題のあるコード

```python
def process_payment(user_id, amount):
    user = get_user(user_id)
    user.balance -= amount
    save_user(user)
    return True
```

#### レビュー指摘事項

1. **エラーハンドリング**: 残高不足のチェックがない
2. **データ整合性**: トランザクションがない
3. **境界値**: 負の金額が処理できる
4. **監査ログ**: 決済履歴が記録されない

#### 改善版

```python
def process_payment(user_id: int, amount: Decimal) -> Payment:
    # 1. バリデーション
    if amount <= 0:
        raise ValueError("Amount must be positive")

    if amount > Decimal('1000000'):
        raise ValueError("Amount exceeds maximum limit")

    # 2. トランザクション開始
    with database.transaction():
        # 3. ロック付きでユーザー取得
        user = get_user_for_update(user_id)

        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")

        # 4. 残高チェック
        if user.balance < amount:
            raise InsufficientBalanceError(
                f"Balance {user.balance} is less than {amount}"
            )

        # 5. 残高更新
        user.balance -= amount
        save_user(user)

        # 6. 決済履歴記録
        payment = Payment.objects.create(
            user_id=user_id,
            amount=amount,
            status='completed',
            timestamp=timezone.now()
        )

        # 7. 監査ログ
        audit_log.info(
            'payment_processed',
            user_id=user_id,
            amount=amount,
            new_balance=user.balance
        )

        return payment
```

### ケース3: データ取得APIのレビュー

#### 問題のあるコード

```swift
func fetchUsers(page: Int) -> [User] {
    let url = "https://api.example.com/users?page=\(page)"
    let data = try! Data(contentsOf: URL(string: url)!)
    let users = try! JSONDecoder().decode([User].self, from: data)
    return users
}
```

#### レビュー指摘事項

1. **エラーハンドリング**: `try!`による強制アンラップでクラッシュの危険
2. **同期処理**: メインスレッドでネットワークリクエスト
3. **タイムアウト**: ネットワークタイムアウトの設定がない
4. **ページネーション**: ページ番号のバリデーションがない

#### 改善版

```swift
func fetchUsers(page: Int, completion: @escaping (Result<[User], Error>) -> Void) {
    // 1. バリデーション
    guard page > 0 else {
        completion(.failure(ValidationError.invalidPage))
        return
    }

    // 2. URL構築
    guard let url = URL(string: "https://api.example.com/users?page=\(page)") else {
        completion(.failure(NetworkError.invalidURL))
        return
    }

    // 3. リクエスト設定（タイムアウト付き）
    var request = URLRequest(url: url)
    request.timeoutInterval = 30
    request.httpMethod = "GET"
    request.setValue("application/json", forHTTPHeaderField: "Accept")

    // 4. 非同期リクエスト
    URLSession.shared.dataTask(with: request) { data, response, error in
        // 5. エラーチェック
        if let error = error {
            completion(.failure(error))
            return
        }

        // 6. レスポンスチェック
        guard let httpResponse = response as? HTTPURLResponse else {
            completion(.failure(NetworkError.invalidResponse))
            return
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            completion(.failure(NetworkError.httpError(httpResponse.statusCode)))
            return
        }

        // 7. データチェック
        guard let data = data else {
            completion(.failure(NetworkError.noData))
            return
        }

        // 8. デコード
        do {
            let users = try JSONDecoder().decode([User].self, from: data)
            completion(.success(users))
        } catch {
            completion(.failure(NetworkError.decodingError(error)))
        }
    }.resume()
}
```

---

## まとめ

機能性レビューは、コードが「正しく動く」ことを保証するための最も重要なレビュー観点です。

### 重要ポイント

1. **要件を明確に理解する**
   - PRの説明、仕様書、ユーザーストーリーを確認
   - 不明点は必ず質問して明確化

2. **エッジケースを徹底的に考える**
   - 空配列、null/nil、境界値、異常系を必ずチェック
   - 「こんなことは起きないだろう」という思い込みを捨てる

3. **エラーハンドリングを適切に実装する**
   - すべてのエラーケースを考慮
   - ユーザーフレンドリーなエラーメッセージ
   - 監査ログとエラーログの記録

4. **境界値を必ずテストする**
   - 0, 1, 最大値、最小値をテスト
   - オフバイワンエラーに注意

5. **言語固有の落とし穴を理解する**
   - 各言語の危険な構文を把握
   - ベストプラクティスに従う

### レビュープロセス

1. **要件の理解** (5分)
2. **全体像の把握** (5-10分)
3. **詳細レビュー** (20-30分)
4. **テストコード確認** (10分)
5. **コメント作成** (10分)

### チェックリストの活用

機能性レビューでは、以下のチェックリストを活用してください：

- [セルフレビューチェックリスト](../checklists/self-review.md)
- [レビュー観点チェックリスト](../checklists/review-checklist.md)
- [セキュリティチェックリスト](../checklists/security-checklist.md)

### 次のステップ

- [設計レビュー](02-design.md) - アーキテクチャと設計パターン
- [可読性レビュー](03-readability.md) - 理解しやすいコード
- [テストレビュー](04-testing.md) - テストの品質
- [セキュリティレビュー](05-security.md) - 脆弱性の発見
