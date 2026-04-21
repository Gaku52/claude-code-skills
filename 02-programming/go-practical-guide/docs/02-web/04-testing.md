# Complete Guide to Go Testing

> Practical testing techniques that guarantee Go code quality through table-driven tests, testify, and httptest

## What You Will Learn in This Chapter

1. How to write comprehensive and maintainable tests using the **table-driven tests** pattern
2. How to leverage the **testify** library for assertions, mocks, and suites
3. Techniques for testing HTTP handlers and clients with the **httptest** package
4. Design patterns for **integration tests** and test helpers
5. Practical use of **test coverage** and benchmarks


## Prerequisites

Before reading this guide, the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content of [gRPC -- Protocol Buffers, Service Definitions, Streaming](./03-grpc.md)

---

## 1. Basic Structure of Go Tests

### 1-1. Naming Conventions for Test Files

```
project/
├── handler.go
├── handler_test.go      ← Same package
├── service.go
├── service_test.go
├── handler_integration_test.go  ← Integration test
└── testdata/                    ← Directory for test data
    ├── golden_response.json
    └── fixtures/
        ├── users.json
        └── config.yaml
```

Go tests are written in files with the `_test.go` suffix. Test functions start with the `Test` prefix and take `*testing.T` as an argument. The `testdata` directory is ignored by the Go build system, making it ideal for storing test fixtures and golden files.

### 1-2. Choosing a Test Package

There are two approaches to naming the package in a test file.

```go
// Approach 1: Same-package test (white-box testing)
// handler_test.go
package myapp

// Can access private functions and fields
func TestInternalLogic(t *testing.T) {
    result := internalHelper("input")  // Directly tests unexported functions
    if result != "expected" {
        t.Errorf("internalHelper() = %q, want %q", result, "expected")
    }
}

// Approach 2: External-package test (black-box testing)
// handler_test.go
package myapp_test

import "myproject/myapp"

// Tests only the public API (user-perspective verification)
func TestPublicAPI(t *testing.T) {
    result := myapp.Process("input")
    if result != "expected" {
        t.Errorf("Process() = %q, want %q", result, "expected")
    }
}
```

### Code Example 1: Minimal Test

```go
package calc

import "testing"

func Add(a, b int) int {
    return a + b
}

func TestAdd(t *testing.T) {
    got := Add(2, 3)
    want := 5
    if got != want {
        t.Errorf("Add(2, 3) = %d, want %d", got, want)
    }
}
```

### Test Execution Flow

```
+------------------+     +------------------+     +------------------+
|  go test ./...   | --> | Compile          | --> | Test binary      |
|  command         |     | including        |     | execution &      |
|                  |     | *_test.go files  |     | result display   |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
  Flag parsing             Discover test          PASS / FAIL decision
  -v, -run, -cover         functions              exit code 0 or 1
                           Test*, Benchmark*
```

### 1-3. Main Flags for Test Execution

```bash
# Basic execution
go test ./...                       # Run tests in all packages
go test -v ./...                    # Verbose output
go test -run TestAdd ./...          # Run only a specific test
go test -run TestDivide/ZeroDivision # Specify a subtest

# Parallel / timeout control
go test -parallel 4 ./...           # Specify number of parallel executions
go test -timeout 60s ./...          # Overall test timeout
go test -count=5 ./...              # Run repeatedly (invalidate cache)

# Coverage
go test -cover ./...                # Run with coverage
go test -coverprofile=coverage.out  # Output profile
go test -covermode=atomic ./...     # Concurrency-safe coverage

# Race detection
go test -race ./...                 # Detect data races
go test -race -count=10 ./...       # Improve race detection accuracy by repeating

# Build tags
go test -tags=integration ./...     # Run tests with specified tag
go test -short ./...                # Short-test mode
```

---

## 2. Table-Driven Tests

The most recommended testing pattern in Go. Test cases are defined as a slice and executed in a loop.

### Code Example 2: Basic Table-Driven Test

```go
func TestDivide(t *testing.T) {
    tests := []struct {
        name      string
        a, b      float64
        want      float64
        wantError bool
    }{
        {name: "normal division", a: 10, b: 2, want: 5, wantError: false},
        {name: "fractional result", a: 7, b: 3, want: 2.3333, wantError: false},
        {name: "division by zero", a: 5, b: 0, want: 0, wantError: true},
        {name: "negative number", a: -10, b: 2, want: -5, wantError: false},
        {name: "both negative", a: -10, b: -2, want: 5, wantError: false},
        {name: "very small number", a: 1, b: 1000000, want: 0.000001, wantError: false},
        {name: "very large number", a: 1e18, b: 1e9, want: 1e9, wantError: false},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := Divide(tt.a, tt.b)
            if (err != nil) != tt.wantError {
                t.Fatalf("Divide(%v, %v) error = %v, wantError %v",
                    tt.a, tt.b, err, tt.wantError)
            }
            if !tt.wantError && math.Abs(got-tt.want) > 0.001 {
                t.Errorf("Divide(%v, %v) = %v, want %v",
                    tt.a, tt.b, got, tt.want)
            }
        })
    }
}
```

### Structure of a Table-Driven Test

```
+------------------------------------------+
|  tests := []struct{ ... }{               |
|    +------------------------------------+|
|    | Case 1: name, input, expected      ||
|    +------------------------------------+|
|    | Case 2: name, input, expected      ||
|    +------------------------------------+|
|    | Case 3: name, input, expected      ||
|    +------------------------------------+|
|  }                                       |
|                                          |
|  for _, tt := range tests {              |
|    t.Run(tt.name, func(t *testing.T){    |
|      // Test logic                       |
|    })                                    |
|  }                                       |
+------------------------------------------+
         |
         v
  $ go test -run TestDivide/ZeroDivision
  --- Individual cases can also be executed
```

### Code Example 3: Parallel Table-Driven Test

```go
func TestSlowOperation(t *testing.T) {
    tests := []struct {
        name  string
        input string
        want  string
    }{
        {"case A", "hello", "HELLO"},
        {"case B", "world", "WORLD"},
        {"case C", "go", "GO"},
    }

    for _, tt := range tests {
        tt := tt // Capture loop variable (required before Go 1.22)
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel() // Enable parallel execution
            got := strings.ToUpper(tt.input)
            if got != tt.want {
                t.Errorf("ToUpper(%q) = %q, want %q", tt.input, got, tt.want)
            }
        })
    }
}
```

### Code Example 4: Table-Driven Test with Setup/Teardown

```go
func TestDatabaseOperations(t *testing.T) {
    tests := []struct {
        name    string
        setup   func(db *sql.DB)    // Setup before the test
        action  func(db *sql.DB) error
        verify  func(t *testing.T, db *sql.DB)
        cleanup func(db *sql.DB)    // Cleanup after the test
    }{
        {
            name: "create user",
            setup: func(db *sql.DB) {
                db.Exec("DELETE FROM users")
            },
            action: func(db *sql.DB) error {
                _, err := db.Exec("INSERT INTO users (name, email) VALUES (?, ?)",
                    "Alice", "alice@example.com")
                return err
            },
            verify: func(t *testing.T, db *sql.DB) {
                var count int
                db.QueryRow("SELECT COUNT(*) FROM users").Scan(&count)
                if count != 1 {
                    t.Errorf("user count = %d, want 1", count)
                }
            },
            cleanup: func(db *sql.DB) {
                db.Exec("DELETE FROM users")
            },
        },
        {
            name: "reject duplicate email",
            setup: func(db *sql.DB) {
                db.Exec("DELETE FROM users")
                db.Exec("INSERT INTO users (name, email) VALUES (?, ?)",
                    "Alice", "alice@example.com")
            },
            action: func(db *sql.DB) error {
                _, err := db.Exec("INSERT INTO users (name, email) VALUES (?, ?)",
                    "Bob", "alice@example.com")
                return err
            },
            verify: func(t *testing.T, db *sql.DB) {
                // Verify that an error occurred (check via the action's return value)
            },
            cleanup: func(db *sql.DB) {
                db.Exec("DELETE FROM users")
            },
        },
    }

    db := setupTestDB(t) // Test DB connection
    defer db.Close()

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            if tt.setup != nil {
                tt.setup(db)
            }
            if tt.cleanup != nil {
                defer tt.cleanup(db)
            }

            err := tt.action(db)
            if tt.verify != nil {
                tt.verify(t, db)
            }
            _ = err // Verify error as needed
        })
    }
}
```

### Code Example 5: Table-Driven Test with Custom Matcher

```go
func TestParseConfig(t *testing.T) {
    tests := []struct {
        name    string
        input   string
        check   func(t *testing.T, cfg *Config, err error) // Custom verification function
    }{
        {
            name:  "complete config",
            input: `{"host": "localhost", "port": 8080, "debug": true}`,
            check: func(t *testing.T, cfg *Config, err error) {
                t.Helper()
                require.NoError(t, err)
                assert.Equal(t, "localhost", cfg.Host)
                assert.Equal(t, 8080, cfg.Port)
                assert.True(t, cfg.Debug)
            },
        },
        {
            name:  "apply default values",
            input: `{}`,
            check: func(t *testing.T, cfg *Config, err error) {
                t.Helper()
                require.NoError(t, err)
                assert.Equal(t, "0.0.0.0", cfg.Host, "default host")
                assert.Equal(t, 3000, cfg.Port, "default port")
                assert.False(t, cfg.Debug, "debug disabled by default")
            },
        },
        {
            name:  "invalid JSON",
            input: `{invalid`,
            check: func(t *testing.T, cfg *Config, err error) {
                t.Helper()
                require.Error(t, err)
                assert.Nil(t, cfg)
                assert.Contains(t, err.Error(), "invalid")
            },
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            cfg, err := ParseConfig([]byte(tt.input))
            tt.check(t, cfg, err)
        })
    }
}
```

---

## 3. The testify Library

### Installation

```bash
go get github.com/stretchr/testify
```

### Code Example 6: testify/assert and testify/require

```go
package user_test

import (
    "testing"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestCreateUser(t *testing.T) {
    // assert: continues the test even on failure
    user, err := CreateUser("Alice", "alice@example.com")
    assert.NoError(t, err, "an error occurred while creating the user")
    assert.Equal(t, "Alice", user.Name)
    assert.NotEmpty(t, user.ID)

    // require: immediately aborts the test on failure
    token, err := user.GenerateToken()
    require.NoError(t, err, "token generation is required")
    require.NotEmpty(t, token)

    // Further test using the token
    claims, err := ParseToken(token)
    assert.NoError(t, err)
    assert.Equal(t, user.ID, claims.UserID)
}
```

### assert vs. require Comparison Table

| Item | `assert` | `require` |
|------|----------|-----------|
| Behavior on failure | Continue test (equivalent to `t.Errorf`) | Abort test immediately (equivalent to `t.Fatalf`) |
| Use case | Run multiple verifications at once | Verify prerequisites for subsequent tests |
| Return value | `bool` (success/failure) | None (calls t.FailNow on failure) |
| Recommended for | Value comparisons, attribute checks | Nil checks, error checks |
| Output | Shows all failures together | Shows only the first failure |

### Main testify Assertions

```go
// Equality comparison
assert.Equal(t, expected, actual)           // DeepEqual comparison
assert.NotEqual(t, unexpected, actual)
assert.EqualValues(t, expected, actual)     // Comparison including type conversion

// nil / empty checks
assert.Nil(t, obj)
assert.NotNil(t, obj)
assert.Empty(t, collection)                // len == 0
assert.NotEmpty(t, collection)

// Boolean
assert.True(t, condition)
assert.False(t, condition)

// Error
assert.NoError(t, err)
assert.Error(t, err)
assert.ErrorIs(t, err, ErrNotFound)        // Equivalent to errors.Is
assert.ErrorAs(t, err, &target)            // Equivalent to errors.As
assert.ErrorContains(t, err, "not found")

// Collection
assert.Contains(t, list, element)
assert.NotContains(t, list, element)
assert.Len(t, list, expectedLen)
assert.ElementsMatch(t, expected, actual)  // Order-independent comparison

// String
assert.Contains(t, str, substring)
assert.Regexp(t, regexp, str)

// Numeric
assert.Greater(t, a, b)
assert.GreaterOrEqual(t, a, b)
assert.InDelta(t, expected, actual, delta)  // Approximate floating-point comparison

// Panic
assert.Panics(t, func() { panicFunc() })
assert.NotPanics(t, func() { safeFunc() })

// JSON
assert.JSONEq(t, expectedJSON, actualJSON)  // Semantic comparison of JSON strings

// Time
assert.WithinDuration(t, expected, actual, delta)
```

### Code Example 7: testify/mock

```go
// Interface definition
type UserRepository interface {
    FindByID(id string) (*User, error)
    FindByEmail(email string) (*User, error)
    Save(user *User) error
    Delete(id string) error
    List(offset, limit int) ([]*User, error)
}

// Mock generation
type MockUserRepo struct {
    mock.Mock
}

func (m *MockUserRepo) FindByID(id string) (*User, error) {
    args := m.Called(id)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).(*User), args.Error(1)
}

func (m *MockUserRepo) FindByEmail(email string) (*User, error) {
    args := m.Called(email)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).(*User), args.Error(1)
}

func (m *MockUserRepo) Save(user *User) error {
    args := m.Called(user)
    return args.Error(0)
}

func (m *MockUserRepo) Delete(id string) error {
    args := m.Called(id)
    return args.Error(0)
}

func (m *MockUserRepo) List(offset, limit int) ([]*User, error) {
    args := m.Called(offset, limit)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).([]*User), args.Error(1)
}

// Usage in tests
func TestUpdateUserName(t *testing.T) {
    mockRepo := new(MockUserRepo)

    existingUser := &User{ID: "123", Name: "Alice"}
    mockRepo.On("FindByID", "123").Return(existingUser, nil)
    mockRepo.On("Save", mock.AnythingOfType("*User")).Return(nil)

    service := NewUserService(mockRepo)
    err := service.UpdateName("123", "Bob")

    assert.NoError(t, err)
    mockRepo.AssertExpectations(t)
    mockRepo.AssertCalled(t, "Save", mock.MatchedBy(func(u *User) bool {
        return u.Name == "Bob"
    }))
}

// More advanced mock patterns
func TestUserServiceEdgeCases(t *testing.T) {
    t.Run("when user is not found", func(t *testing.T) {
        mockRepo := new(MockUserRepo)
        mockRepo.On("FindByID", "999").Return(nil, ErrNotFound)

        service := NewUserService(mockRepo)
        err := service.UpdateName("999", "Bob")

        assert.ErrorIs(t, err, ErrNotFound)
        mockRepo.AssertNotCalled(t, "Save")
    })

    t.Run("rollback on save failure", func(t *testing.T) {
        mockRepo := new(MockUserRepo)
        existingUser := &User{ID: "123", Name: "Alice"}
        mockRepo.On("FindByID", "123").Return(existingUser, nil)
        mockRepo.On("Save", mock.Anything).Return(errors.New("db error"))

        service := NewUserService(mockRepo)
        err := service.UpdateName("123", "Bob")

        assert.Error(t, err)
        assert.Contains(t, err.Error(), "db error")
    })

    t.Run("verify call count", func(t *testing.T) {
        mockRepo := new(MockUserRepo)
        users := []*User{
            {ID: "1", Name: "Alice"},
            {ID: "2", Name: "Bob"},
        }
        mockRepo.On("List", 0, 10).Return(users, nil).Once()

        service := NewUserService(mockRepo)
        result, _ := service.ListUsers(0, 10)

        assert.Len(t, result, 2)
        mockRepo.AssertNumberOfCalls(t, "List", 1)
    })
}
```

### Code Example 8: testify/suite

```go
package user_test

import (
    "database/sql"
    "testing"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/suite"
)

// Test suite definition
type UserServiceSuite struct {
    suite.Suite
    db      *sql.DB
    service *UserService
    repo    *UserRepo
}

// Run once before the suite starts
func (s *UserServiceSuite) SetupSuite() {
    db, err := sql.Open("sqlite3", ":memory:")
    s.Require().NoError(err)
    s.db = db

    // Create table
    _, err = db.Exec(`
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    `)
    s.Require().NoError(err)

    s.repo = NewUserRepo(db)
    s.service = NewUserService(s.repo)
}

// Run before each test
func (s *UserServiceSuite) SetupTest() {
    s.db.Exec("DELETE FROM users")
}

// Run after each test
func (s *UserServiceSuite) TearDownTest() {
    // Cleanup as needed
}

// Run once when the suite ends
func (s *UserServiceSuite) TearDownSuite() {
    s.db.Close()
}

// Test cases
func (s *UserServiceSuite) TestCreateUser() {
    user, err := s.service.Create("Alice", "alice@example.com")
    s.NoError(err)
    s.NotEmpty(user.ID)
    s.Equal("Alice", user.Name)
}

func (s *UserServiceSuite) TestCreateDuplicateEmail() {
    _, err := s.service.Create("Alice", "alice@example.com")
    s.NoError(err)

    _, err = s.service.Create("Bob", "alice@example.com")
    s.Error(err)
    s.ErrorIs(err, ErrDuplicate)
}

func (s *UserServiceSuite) TestFindUser() {
    created, _ := s.service.Create("Alice", "alice@example.com")

    found, err := s.service.FindByID(created.ID)
    s.NoError(err)
    s.Equal(created.ID, found.ID)
    s.Equal("Alice", found.Name)
}

func (s *UserServiceSuite) TestDeleteUser() {
    created, _ := s.service.Create("Alice", "alice@example.com")

    err := s.service.Delete(created.ID)
    s.NoError(err)

    _, err = s.service.FindByID(created.ID)
    s.ErrorIs(err, ErrNotFound)
}

// Entry point for running the suite
func TestUserServiceSuite(t *testing.T) {
    suite.Run(t, new(UserServiceSuite))
}
```

---

## 4. The httptest Package

### Overview of HTTP Testing

```
+----------------------------+
|  Choosing the test target  |
+----------------------------+
        |             |
        v             v
+-------------+ +----------------+
| Server-side | | Client-side    |
| Test        | | Test external  |
| handlers    | | API calls      |
+-------------+ +----------------+
        |             |
        v             v
+-------------+ +----------------+
| httptest.   | | httptest.      |
| NewRecorder | | NewServer      |
| Request →   | | Spin up a mock |
| Response    | | server to verify|
+-------------+ +----------------+
```

### Code Example 9: Handler Testing with httptest.NewRecorder

```go
func TestHealthHandler(t *testing.T) {
    // Handler definition
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
    })

    // Create request
    req := httptest.NewRequest("GET", "/health", nil)
    rec := httptest.NewRecorder()

    // Execute handler
    handler.ServeHTTP(rec, req)

    // Verify
    assert.Equal(t, http.StatusOK, rec.Code)
    assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")

    var body map[string]string
    err := json.Unmarshal(rec.Body.Bytes(), &body)
    require.NoError(t, err)
    assert.Equal(t, "ok", body["status"])
}
```

### Code Example 10: Full Test for JSON Requests and Responses

```go
func TestCreateUserHandler(t *testing.T) {
    tests := []struct {
        name       string
        body       interface{}
        wantStatus int
        wantBody   map[string]interface{}
    }{
        {
            name:       "successful creation",
            body:       map[string]string{"name": "Alice", "email": "alice@example.com"},
            wantStatus: http.StatusCreated,
            wantBody:   map[string]interface{}{"name": "Alice", "email": "alice@example.com"},
        },
        {
            name:       "empty name",
            body:       map[string]string{"name": "", "email": "alice@example.com"},
            wantStatus: http.StatusBadRequest,
            wantBody:   map[string]interface{}{"error": "name is required"},
        },
        {
            name:       "invalid email",
            body:       map[string]string{"name": "Alice", "email": "invalid"},
            wantStatus: http.StatusBadRequest,
            wantBody:   map[string]interface{}{"error": "invalid email format"},
        },
        {
            name:       "invalid JSON",
            body:       "invalid json",
            wantStatus: http.StatusBadRequest,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            var bodyReader io.Reader
            switch v := tt.body.(type) {
            case string:
                bodyReader = strings.NewReader(v)
            default:
                jsonBytes, _ := json.Marshal(v)
                bodyReader = bytes.NewReader(jsonBytes)
            }

            req := httptest.NewRequest("POST", "/api/users", bodyReader)
            req.Header.Set("Content-Type", "application/json")
            rec := httptest.NewRecorder()

            handler := NewRouter() // Router under test
            handler.ServeHTTP(rec, req)

            assert.Equal(t, tt.wantStatus, rec.Code)

            if tt.wantBody != nil {
                var got map[string]interface{}
                err := json.Unmarshal(rec.Body.Bytes(), &got)
                require.NoError(t, err)
                for key, want := range tt.wantBody {
                    assert.Equal(t, want, got[key], "field %s does not match", key)
                }
            }
        })
    }
}
```

### Code Example 11: Mocking an External API with httptest.NewServer

```go
func TestFetchUserFromAPI(t *testing.T) {
    // Create a mock server
    mockServer := httptest.NewServer(http.HandlerFunc(
        func(w http.ResponseWriter, r *http.Request) {
            assert.Equal(t, "/api/users/42", r.URL.Path)
            assert.Equal(t, "Bearer test-token", r.Header.Get("Authorization"))

            w.Header().Set("Content-Type", "application/json")
            json.NewEncoder(w).Encode(map[string]interface{}{
                "id":   42,
                "name": "Alice",
            })
        },
    ))
    defer mockServer.Close()

    // Inject the mock URL into the client under test
    client := NewAPIClient(mockServer.URL, "test-token")
    user, err := client.FetchUser(42)

    require.NoError(t, err)
    assert.Equal(t, 42, user.ID)
    assert.Equal(t, "Alice", user.Name)
}
```

### Code Example 12: Mock Server with Multiple Endpoints

```go
func TestExternalAPIClient(t *testing.T) {
    // Mock server that handles multiple endpoints
    mockServer := httptest.NewServer(http.HandlerFunc(
        func(w http.ResponseWriter, r *http.Request) {
            switch {
            case r.Method == "GET" && r.URL.Path == "/api/users":
                // User list
                w.Header().Set("Content-Type", "application/json")
                json.NewEncoder(w).Encode([]map[string]interface{}{
                    {"id": 1, "name": "Alice"},
                    {"id": 2, "name": "Bob"},
                })

            case r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/api/users/"):
                // Get individual user
                id := strings.TrimPrefix(r.URL.Path, "/api/users/")
                if id == "999" {
                    w.WriteHeader(http.StatusNotFound)
                    json.NewEncoder(w).Encode(map[string]string{"error": "not found"})
                    return
                }
                w.Header().Set("Content-Type", "application/json")
                json.NewEncoder(w).Encode(map[string]interface{}{
                    "id":   1,
                    "name": "Alice",
                })

            case r.Method == "POST" && r.URL.Path == "/api/users":
                // Create user
                var body map[string]string
                json.NewDecoder(r.Body).Decode(&body)
                w.WriteHeader(http.StatusCreated)
                json.NewEncoder(w).Encode(map[string]interface{}{
                    "id":   100,
                    "name": body["name"],
                })

            default:
                w.WriteHeader(http.StatusNotFound)
            }
        },
    ))
    defer mockServer.Close()

    client := NewAPIClient(mockServer.URL, "test-token")

    t.Run("fetch user list", func(t *testing.T) {
        users, err := client.ListUsers()
        require.NoError(t, err)
        assert.Len(t, users, 2)
    })

    t.Run("non-existent user", func(t *testing.T) {
        _, err := client.FetchUser(999)
        assert.ErrorIs(t, err, ErrNotFound)
    })

    t.Run("create user", func(t *testing.T) {
        user, err := client.CreateUser("Charlie")
        require.NoError(t, err)
        assert.Equal(t, "Charlie", user.Name)
    })
}
```

### Code Example 13: Testing Middleware

```go
func TestAuthMiddleware(t *testing.T) {
    tests := []struct {
        name       string
        token      string
        wantStatus int
    }{
        {"valid token", "valid-token", http.StatusOK},
        {"invalid token", "bad-token", http.StatusUnauthorized},
        {"no token", "", http.StatusUnauthorized},
        {"expired token", "expired-token", http.StatusUnauthorized},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                w.WriteHeader(http.StatusOK)
            })

            handler := AuthMiddleware(inner)
            req := httptest.NewRequest("GET", "/protected", nil)
            if tt.token != "" {
                req.Header.Set("Authorization", "Bearer "+tt.token)
            }
            rec := httptest.NewRecorder()

            handler.ServeHTTP(rec, req)
            assert.Equal(t, tt.wantStatus, rec.Code)
        })
    }
}

// Rate-limit middleware test
func TestRateLimitMiddleware(t *testing.T) {
    inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
    })

    // Rate limit of up to 3 requests per second
    handler := RateLimitMiddleware(3, time.Second)(inner)

    // The first 3 requests succeed
    for i := 0; i < 3; i++ {
        req := httptest.NewRequest("GET", "/api/data", nil)
        req.RemoteAddr = "192.168.1.1:12345"
        rec := httptest.NewRecorder()
        handler.ServeHTTP(rec, req)
        assert.Equal(t, http.StatusOK, rec.Code, "request %d", i+1)
    }

    // The 4th request is throttled
    req := httptest.NewRequest("GET", "/api/data", nil)
    req.RemoteAddr = "192.168.1.1:12345"
    rec := httptest.NewRecorder()
    handler.ServeHTTP(rec, req)
    assert.Equal(t, http.StatusTooManyRequests, rec.Code)
}

// CORS middleware test
func TestCORSMiddleware(t *testing.T) {
    inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
    })

    handler := CORSMiddleware(CORSConfig{
        AllowOrigins: []string{"https://example.com"},
        AllowMethods: []string{"GET", "POST"},
        AllowHeaders: []string{"Content-Type", "Authorization"},
    })(inner)

    t.Run("preflight request", func(t *testing.T) {
        req := httptest.NewRequest("OPTIONS", "/api/data", nil)
        req.Header.Set("Origin", "https://example.com")
        req.Header.Set("Access-Control-Request-Method", "POST")
        rec := httptest.NewRecorder()

        handler.ServeHTTP(rec, req)

        assert.Equal(t, http.StatusNoContent, rec.Code)
        assert.Equal(t, "https://example.com",
            rec.Header().Get("Access-Control-Allow-Origin"))
    })

    t.Run("disallowed origin", func(t *testing.T) {
        req := httptest.NewRequest("GET", "/api/data", nil)
        req.Header.Set("Origin", "https://evil.com")
        rec := httptest.NewRecorder()

        handler.ServeHTTP(rec, req)

        assert.Empty(t, rec.Header().Get("Access-Control-Allow-Origin"))
    })
}
```

### Code Example 14: Testing a TLS Server

```go
func TestHTTPSClient(t *testing.T) {
    // Test server with TLS
    tlsServer := httptest.NewTLSServer(http.HandlerFunc(
        func(w http.ResponseWriter, r *http.Request) {
            w.WriteHeader(http.StatusOK)
            w.Write([]byte("secure"))
        },
    ))
    defer tlsServer.Close()

    // Get the client for the TLS server
    client := tlsServer.Client()

    resp, err := client.Get(tlsServer.URL + "/secure")
    require.NoError(t, err)
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)
    assert.Equal(t, "secure", string(body))
}
```

---

## 5. Test Helpers and Utilities

### Code Example 15: Test Helper Functions

```go
// testhelper.go (inside the test package)

// Use t.Helper() to display the caller's line number
func assertJSON(t *testing.T, body []byte, key, want string) {
    t.Helper() // Without this, this function's line number is displayed
    var m map[string]string
    require.NoError(t, json.Unmarshal(body, &m))
    assert.Equal(t, want, m[key])
}

// HTTP request builder for tests
func newJSONRequest(t *testing.T, method, url string, body interface{}) *http.Request {
    t.Helper()
    var reader io.Reader
    if body != nil {
        jsonBytes, err := json.Marshal(body)
        require.NoError(t, err)
        reader = bytes.NewReader(jsonBytes)
    }
    req := httptest.NewRequest(method, url, reader)
    req.Header.Set("Content-Type", "application/json")
    return req
}

// Test DB setup
func setupTestDB(t *testing.T) *sql.DB {
    t.Helper()
    db, err := sql.Open("sqlite3", ":memory:")
    require.NoError(t, err)

    // Run migrations
    _, err = db.Exec(testSchema)
    require.NoError(t, err)

    // Automatic cleanup via t.Cleanup (Go 1.14+)
    t.Cleanup(func() {
        db.Close()
    })

    return db
}

// Temporary file for tests
func createTempFile(t *testing.T, content string) string {
    t.Helper()
    f, err := os.CreateTemp("", "test-*")
    require.NoError(t, err)

    _, err = f.WriteString(content)
    require.NoError(t, err)
    f.Close()

    t.Cleanup(func() {
        os.Remove(f.Name())
    })

    return f.Name()
}

// Environment variable setup for tests
func setEnv(t *testing.T, key, value string) {
    t.Helper()
    original := os.Getenv(key)
    os.Setenv(key, value)
    t.Cleanup(func() {
        if original == "" {
            os.Unsetenv(key)
        } else {
            os.Setenv(key, original)
        }
    })
}
```

### Code Example 16: Golden File Testing

```go
// Golden file pattern: save expected output to a file
var update = flag.Bool("update", false, "update golden files")

func TestRenderTemplate(t *testing.T) {
    tests := []struct {
        name string
        data interface{}
    }{
        {"user profile", User{Name: "Alice", Age: 30}},
        {"empty data", User{}},
        {"Japanese name", User{Name: "Taro", Age: 25}},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := RenderTemplate(tt.data)

            goldenFile := filepath.Join("testdata", t.Name()+".golden")

            if *update {
                // Update the golden file
                os.MkdirAll(filepath.Dir(goldenFile), 0755)
                os.WriteFile(goldenFile, []byte(got), 0644)
                return
            }

            // Compare against the golden file
            want, err := os.ReadFile(goldenFile)
            require.NoError(t, err, "golden file not found. Generate it with the -update flag")
            assert.Equal(t, string(want), got)
        })
    }
}
```

### Code Example 17: Time Control for Tests

```go
// Abstract time via an interface
type Clock interface {
    Now() time.Time
}

// Production implementation
type RealClock struct{}
func (RealClock) Now() time.Time { return time.Now() }

// Test implementation
type FakeClock struct {
    current time.Time
}
func (c *FakeClock) Now() time.Time { return c.current }
func (c *FakeClock) Advance(d time.Duration) { c.current = c.current.Add(d) }

// Example usage
type TokenService struct {
    clock     Clock
    ttl       time.Duration
}

func (s *TokenService) IsExpired(token *Token) bool {
    return s.clock.Now().After(token.ExpiresAt)
}

// Test
func TestTokenExpiry(t *testing.T) {
    fakeClock := &FakeClock{current: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)}
    service := &TokenService{clock: fakeClock, ttl: time.Hour}

    token := &Token{
        ExpiresAt: time.Date(2024, 1, 1, 1, 0, 0, 0, time.UTC),
    }

    // Within the validity period
    assert.False(t, service.IsExpired(token))

    // Advance time
    fakeClock.Advance(2 * time.Hour)

    // Expired
    assert.True(t, service.IsExpired(token))
}
```

---

## 6. Integration Tests and Build Tags

### Code Example 18: Test Separation via Build Tags

```go
//go:build integration

package store_test

import (
    "database/sql"
    "os"
    "testing"

    _ "github.com/lib/pq"
)

func TestPostgresUserStore(t *testing.T) {
    dsn := os.Getenv("TEST_DATABASE_URL")
    if dsn == "" {
        t.Skip("TEST_DATABASE_URL is not set")
    }

    db, err := sql.Open("postgres", dsn)
    require.NoError(t, err)
    defer db.Close()

    store := NewUserStore(db)

    t.Run("CRUD operations", func(t *testing.T) {
        // Create
        user, err := store.Create(&User{Name: "Alice", Email: "alice@test.com"})
        require.NoError(t, err)
        assert.NotEmpty(t, user.ID)

        // Read
        found, err := store.FindByID(user.ID)
        require.NoError(t, err)
        assert.Equal(t, "Alice", found.Name)

        // Update
        found.Name = "Alice Updated"
        err = store.Update(found)
        require.NoError(t, err)

        // Delete
        err = store.Delete(user.ID)
        require.NoError(t, err)

        _, err = store.FindByID(user.ID)
        assert.ErrorIs(t, err, ErrNotFound)
    })
}
```

```bash
# Normal tests (integration tests excluded)
go test ./...

# Including integration tests
go test -tags=integration ./...

# Only integration tests
go test -tags=integration -run TestPostgres ./...
```

### Code Example 19: Short Tests via testing.Short()

```go
func TestHeavyComputation(t *testing.T) {
    if testing.Short() {
        t.Skip("skipping heavy computation test in short mode")
    }

    // Time-consuming test
    result := HeavyComputation(largeDataset)
    assert.Equal(t, expectedResult, result)
}
```

```bash
# Short mode (for fast CI feedback)
go test -short ./...

# Full tests
go test ./...
```

### Code Example 20: Docker-based Testing with testcontainers

```go
package store_test

import (
    "context"
    "testing"

    "github.com/testcontainers/testcontainers-go"
    "github.com/testcontainers/testcontainers-go/modules/postgres"
    "github.com/testcontainers/testcontainers-go/wait"
)

func TestWithPostgresContainer(t *testing.T) {
    if testing.Short() {
        t.Skip("skipping container test in short mode")
    }

    ctx := context.Background()

    // Start a PostgreSQL container
    container, err := postgres.RunContainer(ctx,
        testcontainers.WithImage("postgres:16-alpine"),
        postgres.WithDatabase("testdb"),
        postgres.WithUsername("test"),
        postgres.WithPassword("test"),
        testcontainers.WithWaitStrategy(
            wait.ForLog("database system is ready to accept connections").
                WithOccurrence(2),
        ),
    )
    require.NoError(t, err)
    defer container.Terminate(ctx)

    // Get the connection string
    connStr, err := container.ConnectionString(ctx, "sslmode=disable")
    require.NoError(t, err)

    // Connect to DB and run tests
    db, err := sql.Open("postgres", connStr)
    require.NoError(t, err)
    defer db.Close()

    // Run migrations
    runMigrations(t, db)

    // Run tests
    store := NewUserStore(db)
    user, err := store.Create(&User{Name: "Alice", Email: "alice@test.com"})
    require.NoError(t, err)
    assert.NotEmpty(t, user.ID)
}
```

---

## 7. Test Coverage and Benchmarks

### Coverage

```bash
# Run tests with coverage
go test -cover ./...

# Generate HTML report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html

# Per-function coverage
go tool cover -func=coverage.out

# Coverage for specific packages
go test -coverprofile=coverage.out -coverpkg=./internal/... ./...

# Coverage threshold check (used in CI)
COVERAGE=$(go test -cover ./... | grep -oP '\d+\.\d+%' | head -1 | tr -d '%')
if (( $(echo "$COVERAGE < 70" | bc -l) )); then
    echo "coverage below 70%: $COVERAGE%"
    exit 1
fi
```

### Code Example 21: Benchmark Tests

```go
func BenchmarkJSONMarshal(b *testing.B) {
    user := User{ID: "1", Name: "Alice", Email: "alice@example.com"}

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        json.Marshal(user)
    }
}

func BenchmarkJSONMarshalParallel(b *testing.B) {
    user := User{ID: "1", Name: "Alice", Email: "alice@example.com"}

    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            json.Marshal(user)
        }
    })
}

// Measure memory allocations
func BenchmarkStringConcat(b *testing.B) {
    b.Run("Plus operator", func(b *testing.B) {
        b.ReportAllocs()
        for i := 0; i < b.N; i++ {
            s := ""
            for j := 0; j < 100; j++ {
                s += "a"
            }
        }
    })

    b.Run("strings.Builder", func(b *testing.B) {
        b.ReportAllocs()
        for i := 0; i < b.N; i++ {
            var sb strings.Builder
            for j := 0; j < 100; j++ {
                sb.WriteString("a")
            }
            _ = sb.String()
        }
    })

    b.Run("bytes.Buffer", func(b *testing.B) {
        b.ReportAllocs()
        for i := 0; i < b.N; i++ {
            var buf bytes.Buffer
            for j := 0; j < 100; j++ {
                buf.WriteString("a")
            }
            _ = buf.String()
        }
    })
}
```

```bash
# Run benchmarks
go test -bench=BenchmarkJSON -benchmem ./...

# Example output:
# BenchmarkJSONMarshal-8       5000000    320 ns/op    128 B/op    2 allocs/op
# BenchmarkJSONMarshalParallel-8  20000000  85 ns/op   128 B/op    2 allocs/op

# Benchmark comparison (benchstat)
go test -bench=. -count=10 > old.txt
# After code changes
go test -bench=. -count=10 > new.txt
benchstat old.txt new.txt
```

### Code Example 22: Varying Input Size with Sub-Benchmarks

```go
func BenchmarkSort(b *testing.B) {
    sizes := []int{10, 100, 1000, 10000, 100000}

    for _, size := range sizes {
        b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
            // Prepare test data (outside measurement)
            data := make([]int, size)
            for i := range data {
                data[i] = rand.Intn(size * 10)
            }

            b.ResetTimer()
            for i := 0; i < b.N; i++ {
                // Copy data and sort
                cp := make([]int, len(data))
                copy(cp, data)
                sort.Ints(cp)
            }
        })
    }
}
```

---

## 8. Comparison of Testing Techniques

| Technique | Speed | External dependencies | Reliability | Maintenance cost | Applicable scenarios |
|-----------|-------|----------------------|-------------|------------------|----------------------|
| Unit test (table-driven) | Very fast | None | High | Low | Functions/methods |
| httptest.Recorder | Fast | None | High | Low | HTTP handlers |
| httptest.Server | Fast | None | Medium-High | Medium | HTTP clients |
| testify/mock | Fast | None | Medium | Medium | Components with many dependencies |
| testify/suite | Fast | Optional | High | Medium | Test series needing common setup |
| Golden file | Fast | None | High | Low | Template output, serialization |
| Integration test (with DB) | Slow | Yes | Very high | High | End-to-end verification |
| testcontainers | Slow | Docker | Very high | High | Verification in near-production environments |

---

## 9. Comparison of Test Doubles

```
+--------------------------------------------------------------+
|                    Types of test doubles                       |
+--------------------------------------------------------------+
|                                                              |
|  +----------+   +--------+   +------+   +------+   +------+ |
|  | Dummy    |   | Stub   |   | Spy  |   | Mock |   | Fake | |
|  | Not used |   | Returns|   | Record|  | Verify|  | Simple| |
|  | Fills    |   | fixed  |   | calls |  | calls |  | impl. | |
|  | args     |   | values |   |       |  |       |  |       | |
|  +----------+   +--------+   +------+   +------+   +------+ |
|                                                              |
|  Complexity:  Low ←←←←←←←←←←←←←←←←←←←→→→→→→→→→→→→→→→→→ High|
+--------------------------------------------------------------+
```

### Code Example 23: Implementation of Each Test Double

```go
// Interface under test
type EmailSender interface {
    Send(to, subject, body string) error
}

// Dummy: used solely to fill arguments
type DummyEmailSender struct{}
func (d DummyEmailSender) Send(to, subject, body string) error { return nil }

// Stub: returns a fixed value
type StubEmailSender struct {
    Err error // set the error to return
}
func (s StubEmailSender) Send(to, subject, body string) error { return s.Err }

// Spy: records calls
type SpyEmailSender struct {
    Calls []struct {
        To, Subject, Body string
    }
}
func (s *SpyEmailSender) Send(to, subject, body string) error {
    s.Calls = append(s.Calls, struct{ To, Subject, Body string }{to, subject, body})
    return nil
}

// Fake: simple implementation (stores in memory instead of actually sending email)
type FakeEmailSender struct {
    mu     sync.Mutex
    Inbox  map[string][]Message
}

type Message struct {
    Subject string
    Body    string
}

func NewFakeEmailSender() *FakeEmailSender {
    return &FakeEmailSender{Inbox: make(map[string][]Message)}
}

func (f *FakeEmailSender) Send(to, subject, body string) error {
    f.mu.Lock()
    defer f.mu.Unlock()
    f.Inbox[to] = append(f.Inbox[to], Message{Subject: subject, Body: body})
    return nil
}

// Choosing among them in tests
func TestNotificationService(t *testing.T) {
    t.Run("verify send content with Spy", func(t *testing.T) {
        spy := &SpyEmailSender{}
        service := NewNotificationService(spy)

        service.NotifyUser("alice@example.com", "Welcome!")

        require.Len(t, spy.Calls, 1)
        assert.Equal(t, "alice@example.com", spy.Calls[0].To)
        assert.Contains(t, spy.Calls[0].Subject, "Welcome")
    })

    t.Run("verify error case with Stub", func(t *testing.T) {
        stub := StubEmailSender{Err: errors.New("SMTP error")}
        service := NewNotificationService(stub)

        err := service.NotifyUser("alice@example.com", "Welcome!")
        assert.Error(t, err)
    })

    t.Run("verify multiple email sends with Fake", func(t *testing.T) {
        fake := NewFakeEmailSender()
        service := NewNotificationService(fake)

        service.NotifyUser("alice@example.com", "Welcome!")
        service.NotifyUser("alice@example.com", "Update!")

        assert.Len(t, fake.Inbox["alice@example.com"], 2)
    })
}
```

---

## 10. Optimizing Test Execution

### Improving Test Execution Speed

```go
// Package-level setup/teardown with TestMain
func TestMain(m *testing.M) {
    // Setup before all tests
    db := setupDatabase()
    seedTestData(db)

    // Run tests
    code := m.Run()

    // Cleanup after all tests
    db.Close()
    os.Exit(code)
}

// Parallelization via t.Parallel()
func TestParallelOperations(t *testing.T) {
    // t.Setenv() cannot be used in parallel tests (Go 1.17+)
    // Instead, design to avoid shared resources

    tests := []struct {
        name string
        fn   func(t *testing.T)
    }{
        {"test 1", func(t *testing.T) { /* ... */ }},
        {"test 2", func(t *testing.T) { /* ... */ }},
        {"test 3", func(t *testing.T) { /* ... */ }},
    }

    for _, tt := range tests {
        tt := tt
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel()
            tt.fn(t)
        })
    }
}
```

### Test Configuration in a CI Environment

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'

      - name: Unit tests
        run: go test -race -cover -short ./...

      - name: Integration tests
        env:
          TEST_DATABASE_URL: postgres://test:test@localhost:5432/testdb?sslmode=disable
        run: go test -race -tags=integration ./...

      - name: Coverage report
        run: |
          go test -coverprofile=coverage.out ./...
          go tool cover -func=coverage.out
```

---

## 11. Anti-Patterns

### Anti-Pattern 1: Omitting Test Case Names

```go
// BAD: cannot tell which case failed
func TestParse(t *testing.T) {
    tests := []struct {
        input string
        want  int
    }{
        {"42", 42},
        {"abc", 0},
    }
    for _, tt := range tests {
        // No t.Run → hard to identify which case failed
        got, _ := Parse(tt.input)
        if got != tt.want {
            t.Errorf("Parse(%q) = %d, want %d", tt.input, got, tt.want)
        }
    }
}

// GOOD: name each case with t.Run
func TestParse(t *testing.T) {
    tests := []struct {
        name  string
        input string
        want  int
    }{
        {"numeric string", "42", 42},
        {"non-numeric", "abc", 0},
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, _ := Parse(tt.input)
            assert.Equal(t, tt.want, got)
        })
    }
}
```

### Anti-Pattern 2: Relying on sleep in Tests

```go
// BAD: time-dependent, leading to flaky tests
func TestAsyncProcess(t *testing.T) {
    go StartProcess()
    time.Sleep(2 * time.Second) // unstable depending on the environment
    assert.True(t, IsProcessDone())
}

// GOOD: synchronize with channels or contexts
func TestAsyncProcess(t *testing.T) {
    done := make(chan struct{})
    go func() {
        StartProcess()
        close(done)
    }()

    select {
    case <-done:
        assert.True(t, IsProcessDone())
    case <-time.After(5 * time.Second):
        t.Fatal("timeout: process did not complete")
    }
}
```

### Anti-Pattern 3: Dependencies Between Tests

```go
// BAD: depends on test execution order
var sharedState string

func TestStep1(t *testing.T) {
    sharedState = "initialized"
    // ...
}

func TestStep2(t *testing.T) {
    // Assumes sharedState is "initialized"
    // Fails unless TestStep1 runs first
    require.Equal(t, "initialized", sharedState)
}

// GOOD: each test can run independently
func TestStep1(t *testing.T) {
    state := setup()
    // ...
}

func TestStep2(t *testing.T) {
    state := setup() // Set up independently for each test
    require.Equal(t, "initialized", state)
}
```

### Anti-Pattern 4: Testing Implementation Details

```go
// BAD: test tightly coupled to internal implementation
func TestUserService_Create(t *testing.T) {
    service := NewUserService(repo)
    service.Create("Alice", "alice@example.com")

    // Directly inspect internal cache state
    assert.Len(t, service.cache, 1)                    // accessing internal fields
    assert.Equal(t, "Alice", service.cache["alice"].Name) // depends on implementation details
}

// GOOD: test the behavior of the public API
func TestUserService_Create(t *testing.T) {
    service := NewUserService(repo)
    user, err := service.Create("Alice", "alice@example.com")
    require.NoError(t, err)

    // Verify via the public API
    found, err := service.FindByEmail("alice@example.com")
    require.NoError(t, err)
    assert.Equal(t, user.ID, found.ID)
}
```

---

## FAQ

### Q1. What is the difference between `testing.T`'s `Error` and `Fatal`?

`t.Error` / `t.Errorf` marks the test as failed but continues execution. `t.Fatal` / `t.Fatalf` aborts the test immediately. Use `Fatal` when subsequent assertions depend on a prerequisite, and `Error` for independent checks. In testify, `assert` corresponds to `Error` and `require` corresponds to `Fatal`.

### Q2. Do test helper functions need `t.Helper()`?

Yes. Calling `t.Helper()` causes the error output on test failure to display the caller's line number rather than the helper function's. It dramatically improves debugging efficiency, so always add it to test utility functions.

```go
func assertJSON(t *testing.T, body []byte, key, want string) {
    t.Helper() // Without this, this function's line number is displayed
    var m map[string]string
    require.NoError(t, json.Unmarshal(body, &m))
    assert.Equal(t, want, m[key])
}
```

### Q3. What percentage of test coverage should you aim for?

In general, 70-80% is a realistic target. Aiming for 100% skyrockets maintenance costs and leads to writing tests even for trivial code. What matters is achieving high coverage on critical paths (business logic, error handling).

### Q4. What is the difference between t.Cleanup() and defer?

`t.Cleanup()` was added in Go 1.14 and registers a cleanup function to run at the end of the test function (including at the end of subtests). The difference from `defer` is that it can register resource cleanup from within test helper functions. A `defer` inside a helper function executes when the helper returns, whereas `t.Cleanup()` executes when the entire test ends.

```go
// defer executes when the helper function returns (not the intent)
func setupDB(t *testing.T) *sql.DB {
    db, _ := sql.Open("sqlite3", ":memory:")
    defer db.Close() // BAD: DB closes when this function exits
    return db
}

// t.Cleanup executes at the end of the test (correct)
func setupDB(t *testing.T) *sql.DB {
    t.Helper()
    db, _ := sql.Open("sqlite3", ":memory:")
    t.Cleanup(func() { db.Close() }) // GOOD: close at test end
    return db
}
```

### Q5. How do you handle flaky tests (unstable tests)?

Main causes of flaky tests and their mitigations:
1. **Time dependence** → Control time with the `FakeClock` pattern
2. **Concurrency races** → Detect with the `-race` flag; use synchronization primitives
3. **Dependency on external services** → Isolate with mocks/test containers
4. **Shared state between tests** → Make each test independent (easier to spot with `t.Parallel()`)
5. **Random inputs** → Fix the seed value and log output (make it reproducible)

### Q6. How do you disable the go test cache?

`go test` caches test results. To disable the cache, use one of the following:

```bash
# -count=1 disables the cache (the most common approach)
go test -count=1 ./...

# Clear the entire cache
go clean -testcache

# Can also be controlled via an environment variable
GOFLAGS="-count=1" go test ./...
```

---

## Summary

| Concept | Key point |
|---------|-----------|
| table-driven tests | Define test cases as a slice and loop them with `t.Run` |
| t.Parallel() | Speed up with parallel execution of subtests |
| testify/assert | Flexible assertions that continue on failure |
| testify/require | Immediately-aborting assertions for verifying prerequisites |
| testify/mock | Interface-based mock generation |
| testify/suite | Share setup/teardown |
| httptest.NewRecorder | Unit testing handlers |
| httptest.NewServer | Mocking external API calls |
| httptest.NewTLSServer | Testing TLS communication |
| Golden files | Manage expected output as files |
| Build tags | Separate integration tests with the integration tag |
| testcontainers | Production-like environment testing via Docker containers |
| t.Cleanup() | Register resource cleanup at test end |
| t.Helper() | Correctly display the error line number for helper functions |
| Coverage | Measure with `go test -cover`; 70-80% is a realistic target |
| Benchmarks | Measure performance with the `Benchmark` prefix |

---

## Recommended Next Guides

- **03-tools/00-cli-development.md** — CLI development: cobra, flag, promptui
- **03-tools/02-profiling.md** — Profiling: pprof, trace
- **03-tools/04-best-practices.md** — Best practices: Effective Go

---

## References

1. **Go official — Testing package** https://pkg.go.dev/testing
2. **stretchr/testify GitHub** https://github.com/stretchr/testify
3. **Go Blog — Using Subtests and Sub-benchmarks** https://go.dev/blog/subtests
4. **Dave Cheney — Writing Table Driven Tests in Go** https://dave.cheney.net/2019/05/07/prefer-table-driven-tests
5. **testcontainers-go GitHub** https://github.com/testcontainers/testcontainers-go
6. **Go official — Code Coverage for Go Integration Tests** https://go.dev/blog/integration-test-coverage
