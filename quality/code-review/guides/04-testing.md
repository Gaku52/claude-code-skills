# テストレビューガイド

## 概要

テストレビューは、テストコードの品質、カバレッジ、テスタビリティを評価します。適切なテストは、リファクタリングの安全性を保証し、バグの早期発見を可能にします。

## 目次

1. [テストカバレッジ](#テストカバレッジ)
2. [テストケースの品質](#テストケースの品質)
3. [テスタビリティ](#テスタビリティ)
4. [モックとスタブ](#モックとスタブ)
5. [テストの種類](#テストの種類)
6. [言語別テスト戦略](#言語別テスト戦略)

---

## テストカバレッジ

### カバレッジの種類

#### 1. Line Coverage (行カバレッジ)

```typescript
// 実装コード
function calculateDiscount(price: number, userType: string): number {
  let discount = 0;

  if (userType === 'premium') {
    discount = price * 0.2;  // Line 4
  } else if (userType === 'regular') {
    discount = price * 0.1;  // Line 6
  }

  return discount;  // Line 9
}

// ❌ Bad: 不完全なカバレッジ（50%）
test('premium user gets discount', () => {
  expect(calculateDiscount(100, 'premium')).toBe(20);
});
// Line 6, 9のみテスト、Line 4, 7は未テスト

// ✅ Good: 完全なカバレッジ（100%）
describe('calculateDiscount', () => {
  test('premium user gets 20% discount', () => {
    expect(calculateDiscount(100, 'premium')).toBe(20);
  });

  test('regular user gets 10% discount', () => {
    expect(calculateDiscount(100, 'regular')).toBe(10);
  });

  test('guest user gets no discount', () => {
    expect(calculateDiscount(100, 'guest')).toBe(0);
  });
});
```

#### 2. Branch Coverage (分岐カバレッジ)

```python
def process_order(order: Order) -> bool:
    if not order.items:
        return False  # Branch 1

    if order.total > 1000 and not order.is_verified:
        return False  # Branch 2

    return True  # Branch 3

# ❌ Bad: 分岐カバレッジ不足
def test_process_order():
    order = Order(items=[Item()], total=500, is_verified=True)
    assert process_order(order) == True
    # Branch 1, 2が未テスト

# ✅ Good: 全分岐をテスト
def test_process_order_with_empty_items():
    order = Order(items=[], total=500, is_verified=True)
    assert process_order(order) == False

def test_process_order_with_high_total_unverified():
    order = Order(items=[Item()], total=2000, is_verified=False)
    assert process_order(order) == False

def test_process_order_with_high_total_verified():
    order = Order(items=[Item()], total=2000, is_verified=True)
    assert process_order(order) == True

def test_process_order_with_low_total():
    order = Order(items=[Item()], total=500, is_verified=False)
    assert process_order(order) == True
```

### カバレッジ目標

| レイヤー | 目標カバレッジ | 理由 |
|---------|--------------|------|
| ビジネスロジック | 90-100% | クリティカルな処理 |
| API/Controller | 80-90% | 統合テストでカバー可能 |
| ユーティリティ | 90-100% | 再利用が多い |
| UI | 60-80% | E2Eテストでカバー |

---

## テストケースの品質

### AAA Pattern（Arrange-Act-Assert）

```swift
// ✅ Good: AAA Pattern
func testUserRegistration() {
    // Arrange: テストの準備
    let userService = UserService(
        repository: MockUserRepository(),
        validator: UserValidator()
    )
    let email = "test@example.com"
    let password = "SecurePass123"

    // Act: テスト対象の実行
    let result = try? userService.register(
        email: email,
        password: password
    )

    // Assert: 結果の検証
    XCTAssertNotNil(result)
    XCTAssertEqual(result?.email, email)
    XCTAssertNotEqual(result?.password, password) // ハッシュ化されている
}

// ❌ Bad: 混在している
func testUserRegistration() {
    let userService = UserService(repository: MockUserRepository(), validator: UserValidator())
    let result = try? userService.register(email: "test@example.com", password: "pass")
    XCTAssertNotNil(result)
    let email = result?.email
    XCTAssertEqual(email, "test@example.com")
}
```

### テストの独立性

```go
// ❌ Bad: テストが依存し合っている
var globalUser *User

func TestCreateUser(t *testing.T) {
    globalUser = &User{Name: "Test"}
    db.Save(globalUser)

    assert.NotNil(t, globalUser.ID)
}

func TestUpdateUser(t *testing.T) {
    // TestCreateUserに依存している
    globalUser.Name = "Updated"
    db.Save(globalUser)

    assert.Equal(t, "Updated", globalUser.Name)
}

// ✅ Good: 独立したテスト
func TestCreateUser(t *testing.T) {
    // Setup
    db := setupTestDB(t)
    defer db.Close()

    user := &User{Name: "Test"}

    // Execute
    err := db.Save(user)

    // Verify
    assert.NoError(t, err)
    assert.NotEmpty(t, user.ID)
}

func TestUpdateUser(t *testing.T) {
    // Setup
    db := setupTestDB(t)
    defer db.Close()

    // 既存ユーザーを作成
    user := &User{Name: "Original"}
    db.Save(user)

    // Execute
    user.Name = "Updated"
    err := db.Save(user)

    // Verify
    assert.NoError(t, err)
    assert.Equal(t, "Updated", user.Name)
}

func setupTestDB(t *testing.T) *DB {
    db, err := NewTestDB()
    if err != nil {
        t.Fatal(err)
    }
    return db
}
```

### テスト名の明確性

```typescript
// ❌ Bad: 不明確なテスト名
test('test1', () => { /* ... */ });
test('works', () => { /* ... */ });
test('user test', () => { /* ... */ });

// ✅ Good: 明確なテスト名
describe('UserService', () => {
  describe('register', () => {
    test('creates new user with valid email and password', () => {
      // ...
    });

    test('throws ValidationError when email is invalid', () => {
      // ...
    });

    test('throws DuplicateError when email already exists', () => {
      // ...
    });

    test('hashes password before saving', () => {
      // ...
    });
  });

  describe('login', () => {
    test('returns token when credentials are correct', () => {
      // ...
    });

    test('throws AuthenticationError when password is wrong', () => {
      // ...
    });

    test('throws NotFoundError when user does not exist', () => {
      // ...
    });
  });
});

// テスト名のパターン: [メソッド名] [条件] [期待結果]
// - should_ReturnTrue_When_UserIsActive
// - register_ThrowsError_When_EmailIsInvalid
// - calculateTotal_ReturnsZero_When_CartIsEmpty
```

---

## テスタビリティ

### 依存性注入

```python
# ❌ Bad: テストしにくい（依存が隠されている）
class OrderService:
    def __init__(self):
        self.db = DatabaseConnection()  # ハードコーディング
        self.email = EmailService()     # ハードコーディング

    def create_order(self, items: List[Item]) -> Order:
        order = Order(items=items)
        self.db.save(order)
        self.email.send_confirmation(order)
        return order

# テスト時に実際のDBとメールサービスが使われてしまう

# ✅ Good: テストしやすい（依存性注入）
class OrderService:
    def __init__(
        self,
        db: Database,
        email_service: EmailService
    ):
        self.db = db
        self.email_service = email_service

    def create_order(self, items: List[Item]) -> Order:
        order = Order(items=items)
        self.db.save(order)
        self.email_service.send_confirmation(order)
        return order

# テスト
def test_create_order():
    # モックを注入
    mock_db = Mock(spec=Database)
    mock_email = Mock(spec=EmailService)

    service = OrderService(db=mock_db, email_service=mock_email)

    items = [Item(name="Book", price=1000)]
    order = service.create_order(items)

    # モックの呼び出しを検証
    mock_db.save.assert_called_once_with(order)
    mock_email.send_confirmation.assert_called_once_with(order)
```

### Pure Functions

```swift
// ❌ Bad: テストしにくい（副作用がある）
class Calculator {
    var lastResult: Double = 0

    func add(_ a: Double, _ b: Double) -> Double {
        lastResult = a + b  // 副作用
        print("Result: \(lastResult)")  // 副作用
        return lastResult
    }
}

// ✅ Good: テストしやすい（純粋関数）
class Calculator {
    func add(_ a: Double, _ b: Double) -> Double {
        return a + b  // 副作用なし
    }
}

// テスト
func testAdd() {
    let calculator = Calculator()

    // 何度実行しても同じ結果
    XCTAssertEqual(calculator.add(2, 3), 5)
    XCTAssertEqual(calculator.add(2, 3), 5)

    // 順序に依存しない
    XCTAssertEqual(calculator.add(5, 10), 15)
    XCTAssertEqual(calculator.add(2, 3), 5)
}
```

---

## モックとスタブ

### Mock vs Stub

```typescript
// Stub: 事前定義された値を返す
class UserRepositoryStub implements UserRepository {
  async findById(id: string): Promise<User> {
    return {
      id: '123',
      name: 'Test User',
      email: 'test@example.com',
    };
  }
}

// Mock: 呼び出しを記録し、検証できる
class UserRepositoryMock implements UserRepository {
  calls: { method: string; args: any[] }[] = [];

  async findById(id: string): Promise<User> {
    this.calls.push({ method: 'findById', args: [id] });
    return {
      id,
      name: 'Test User',
      email: 'test@example.com',
    };
  }

  assertCalled(method: string, times: number): void {
    const count = this.calls.filter(c => c.method === method).length;
    if (count !== times) {
      throw new Error(`Expected ${method} to be called ${times} times, but was called ${count} times`);
    }
  }
}

// テスト
describe('UserService', () => {
  test('getUserById calls repository once', async () => {
    const mockRepo = new UserRepositoryMock();
    const service = new UserService(mockRepo);

    await service.getUserById('123');

    mockRepo.assertCalled('findById', 1);
  });
});
```

### Jestでのモック

```typescript
// ✅ Good: Jest Mock Functions
import { UserRepository } from './UserRepository';
import { UserService } from './UserService';

jest.mock('./UserRepository');

describe('UserService', () => {
  let mockRepository: jest.Mocked<UserRepository>;
  let userService: UserService;

  beforeEach(() => {
    mockRepository = {
      findById: jest.fn(),
      save: jest.fn(),
      delete: jest.fn(),
    } as any;

    userService = new UserService(mockRepository);
  });

  test('getUserById returns user from repository', async () => {
    const mockUser = {
      id: '123',
      name: 'Test',
      email: 'test@example.com',
    };

    mockRepository.findById.mockResolvedValue(mockUser);

    const result = await userService.getUserById('123');

    expect(result).toEqual(mockUser);
    expect(mockRepository.findById).toHaveBeenCalledWith('123');
    expect(mockRepository.findById).toHaveBeenCalledTimes(1);
  });

  test('deleteUser calls repository delete', async () => {
    const userId = '123';
    mockRepository.findById.mockResolvedValue({
      id: userId,
      name: 'Test',
      email: 'test@example.com',
    });

    await userService.deleteUser(userId);

    expect(mockRepository.delete).toHaveBeenCalledWith(userId);
  });
});
```

---

## テストの種類

### 1. Unit Test（単体テスト）

```go
// 単一の関数やメソッドをテスト
func TestValidateEmail(t *testing.T) {
    tests := []struct {
        name    string
        email   string
        wantErr bool
    }{
        {
            name:    "valid email",
            email:   "user@example.com",
            wantErr: false,
        },
        {
            name:    "invalid email without @",
            email:   "userexample.com",
            wantErr: true,
        },
        {
            name:    "invalid email without domain",
            email:   "user@",
            wantErr: true,
        },
        {
            name:    "empty email",
            email:   "",
            wantErr: true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := ValidateEmail(tt.email)
            if (err != nil) != tt.wantErr {
                t.Errorf("ValidateEmail() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### 2. Integration Test（統合テスト）

```python
# 複数のコンポーネントの連携をテスト
def test_user_registration_flow():
    """ユーザー登録の統合テスト"""
    # 実際のDBを使用（テスト用DB）
    db = create_test_database()

    # 実際のサービスを組み立て
    repo = UserRepository(db)
    validator = UserValidator()
    email_service = MockEmailService()  # メールだけモック
    service = UserService(repo, validator, email_service)

    # 実行
    user = service.register(
        email="newuser@example.com",
        password="SecurePass123"
    )

    # 検証
    assert user.id is not None
    assert user.email == "newuser@example.com"

    # DBに実際に保存されているか確認
    saved_user = db.query(User).filter_by(id=user.id).first()
    assert saved_user is not None
    assert saved_user.email == "newuser@example.com"

    # メール送信が呼ばれたか確認
    assert email_service.sent_emails[0]["to"] == "newuser@example.com"

    # クリーンアップ
    db.close()
```

### 3. E2E Test（エンドツーエンドテスト）

```typescript
// Playwright/Cypressでの例
describe('User Registration E2E', () => {
  test('user can register with valid credentials', async ({ page }) => {
    // ページにアクセス
    await page.goto('http://localhost:3000/register');

    // フォーム入力
    await page.fill('[data-testid="email-input"]', 'newuser@example.com');
    await page.fill('[data-testid="password-input"]', 'SecurePass123');
    await page.fill('[data-testid="confirm-password-input"]', 'SecurePass123');

    // 送信
    await page.click('[data-testid="submit-button"]');

    // 成功メッセージを確認
    await page.waitForSelector('[data-testid="success-message"]');

    const message = await page.textContent('[data-testid="success-message"]');
    expect(message).toContain('Registration successful');

    // ダッシュボードにリダイレクトされることを確認
    await page.waitForURL('http://localhost:3000/dashboard');
  });

  test('shows error when email is already registered', async ({ page }) => {
    await page.goto('http://localhost:3000/register');

    await page.fill('[data-testid="email-input"]', 'existing@example.com');
    await page.fill('[data-testid="password-input"]', 'SecurePass123');
    await page.fill('[data-testid="confirm-password-input"]', 'SecurePass123');

    await page.click('[data-testid="submit-button"]');

    // エラーメッセージを確認
    const error = await page.textContent('[data-testid="error-message"]');
    expect(error).toContain('Email already registered');
  });
});
```

---

## 言語別テスト戦略

### TypeScript/Jest

```typescript
// ✅ ベストプラクティス
describe('UserService', () => {
  // セットアップ・ティアダウン
  let service: UserService;
  let mockRepo: jest.Mocked<UserRepository>;

  beforeEach(() => {
    mockRepo = {
      findById: jest.fn(),
      save: jest.fn(),
    } as any;

    service = new UserService(mockRepo);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  // パラメータ化テスト
  test.each([
    ['valid email', 'user@example.com', false],
    ['invalid email', 'invalid', true],
    ['empty email', '', true],
  ])('validateEmail with %s', (_, email, shouldThrow) => {
    if (shouldThrow) {
      expect(() => service.validateEmail(email)).toThrow();
    } else {
      expect(() => service.validateEmail(email)).not.toThrow();
    }
  });

  // 非同期テスト
  test('getUserById returns user', async () => {
    const mockUser = { id: '1', name: 'Test' };
    mockRepo.findById.mockResolvedValue(mockUser);

    const result = await service.getUserById('1');

    expect(result).toEqual(mockUser);
  });

  // エラーテスト
  test('getUserById throws when user not found', async () => {
    mockRepo.findById.mockRejectedValue(new Error('Not found'));

    await expect(service.getUserById('1')).rejects.toThrow('Not found');
  });
});
```

### Python/pytest

```python
import pytest
from unittest.mock import Mock, patch

class TestUserService:
    @pytest.fixture
    def mock_repo(self):
        """各テストで使用するモックリポジトリ"""
        return Mock(spec=UserRepository)

    @pytest.fixture
    def service(self, mock_repo):
        """各テストで使用するサービス"""
        return UserService(repository=mock_repo)

    # パラメータ化テスト
    @pytest.mark.parametrize("email,expected", [
        ("valid@example.com", True),
        ("invalid", False),
        ("", False),
    ])
    def test_validate_email(self, service, email, expected):
        result = service.validate_email(email)
        assert result == expected

    # 正常系
    def test_get_user_returns_user(self, service, mock_repo):
        mock_user = User(id=1, name="Test")
        mock_repo.find_by_id.return_value = mock_user

        result = service.get_user(1)

        assert result == mock_user
        mock_repo.find_by_id.assert_called_once_with(1)

    # 異常系
    def test_get_user_raises_when_not_found(self, service, mock_repo):
        mock_repo.find_by_id.return_value = None

        with pytest.raises(UserNotFoundError):
            service.get_user(1)
```

### Swift/XCTest

```swift
import XCTest
@testable import MyApp

class UserServiceTests: XCTestCase {
    var service: UserService!
    var mockRepository: MockUserRepository!

    override func setUp() {
        super.setUp()
        mockRepository = MockUserRepository()
        service = UserService(repository: mockRepository)
    }

    override func tearDown() {
        service = nil
        mockRepository = nil
        super.tearDown()
    }

    func testGetUserReturnsUser() {
        // Given
        let expectedUser = User(id: "1", name: "Test")
        mockRepository.userToReturn = expectedUser

        // When
        let result = try? service.getUser(id: "1")

        // Then
        XCTAssertEqual(result, expectedUser)
        XCTAssertEqual(mockRepository.findByIdCallCount, 1)
    }

    func testGetUserThrowsWhenNotFound() {
        // Given
        mockRepository.shouldThrowError = true

        // When/Then
        XCTAssertThrowsError(try service.getUser(id: "1")) { error in
            XCTAssertTrue(error is UserNotFoundError)
        }
    }

    // 非同期テスト
    func testFetchUserAsync() async throws {
        // Given
        let expectedUser = User(id: "1", name: "Test")
        mockRepository.userToReturn = expectedUser

        // When
        let result = try await service.fetchUser(id: "1")

        // Then
        XCTAssertEqual(result, expectedUser)
    }
}

// Mock
class MockUserRepository: UserRepository {
    var userToReturn: User?
    var shouldThrowError = false
    var findByIdCallCount = 0

    func findById(_ id: String) throws -> User {
        findByIdCallCount += 1

        if shouldThrowError {
            throw UserNotFoundError()
        }

        guard let user = userToReturn else {
            throw UserNotFoundError()
        }

        return user
    }
}
```

---

## レビューチェックリスト

### テストレビュー完全チェックリスト

#### カバレッジ
- [ ] 重要な機能がテストされている
- [ ] エッジケースがテストされている
- [ ] エラーケースがテストされている
- [ ] カバレッジ目標を達成している

#### テストケース
- [ ] AAA Patternに従っている
- [ ] テストが独立している
- [ ] テスト名が明確
- [ ] 1テスト1検証

#### テスタビリティ
- [ ] 依存性注入が使われている
- [ ] 副作用が少ない
- [ ] テストしやすい設計

#### モック
- [ ] 適切にモックが使われている
- [ ] 過剰なモックでない
- [ ] モックの検証が適切

---

## まとめ

良いテストは、コードの品質とリファクタリングの安全性を保証します。

### 重要ポイント

1. **十分なカバレッジ**
2. **明確なテストケース**
3. **テストしやすい設計**
4. **適切なモック使用**
5. **独立したテスト**

### 次のステップ

- [セキュリティレビュー](05-security.md)
- [パフォーマンスレビュー](06-performance.md)
