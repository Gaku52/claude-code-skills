# Software Testing

> Code without tests is legacy code. -- Michael Feathers, *Working Effectively with Legacy Code*

Software testing is a systematic activity for verifying that programs behave according to specifications,
enabling early bug detection, quality assurance, and improved maintainability.
This chapter comprehensively covers the classification system from unit tests through integration tests,
E2E tests, and acceptance tests; development methodologies such as TDD and BDD; test techniques
including equivalence partitioning and boundary value analysis; the proper use of mocks and stubs;
automation strategies in CI/CD pipelines; the significance and limitations of code coverage;
property-based testing; and commonly observed anti-patterns in practice.

---

## Learning Objectives

- [ ] Understand the structure and role of each layer in the test pyramid
- [ ] Be able to write unit tests following the AAA pattern
- [ ] Be able to practice the TDD Red-Green-Refactor cycle
- [ ] Be able to explain BDD scenario-driven testing
- [ ] Be able to apply equivalence partitioning, boundary value analysis, and decision tables
- [ ] Be able to determine when to use mocks vs. stubs
- [ ] Know how to integrate tests into a CI/CD pipeline
- [ ] Understand the types and limitations of coverage metrics
- [ ] Know the basics of property-based testing
- [ ] Be able to identify and avoid common testing anti-patterns


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content in [Software Development Process](./00-development-process.md)

---

## 1. The Importance of Testing

### 1.1 Why Write Tests

The reasons why testing is important in software development can be broadly organized into the following four categories.

**1. Early Bug Detection and Cost Reduction**

The cost of fixing software defects increases exponentially the later they are discovered.
If the cost of fixing an error found during the requirements phase is 1,
it becomes 5x in the design phase, 10x in the implementation phase, 20x in the testing phase,
and over 100x after release (based on Barry Boehm's research).
By writing automated tests alongside development, defects can be caught during implementation, minimizing repair costs.

```
Bug fix cost growth curve (conceptual diagram):

Cost
  ^
  |                                              * After release
  |                                         *
  |                                    *
  |                              *
  |                        *
  |                  *
  |            *
  |       * Testing phase
  |    * Implementation phase
  |  * Design phase
  | * Requirements
  +--------------------------------------------> Time
    Req.  Design  Impl.  Testing  Ops  Maint.
```

**2. Prevention of Regression**

When existing functionality breaks due to new changes, it is called a "regression bug."
If an automated test suite exists, tests can be run every time code is changed to immediately
confirm that there are no unintended side effects.
This is especially essential for large codebases and team development.

**3. Design Improvement**

Code that is easy to test is generally loosely coupled and highly cohesive.
By writing tests first (TDD), you are naturally guided toward clean design.
Code that is hard to test almost always has design problems (tight coupling, excessive side effects, mixed responsibilities).

**4. Role as Living Documentation**

Test code concretely demonstrates how the target code should be used and what behavior is expected.
Even if API documentation becomes outdated, as long as tests continue to pass,
the test code itself functions as an up-to-date specification.

### 1.2 Basic Testing Terminology

| Term | Definition |
|------|-----------|
| Test Case | The smallest unit that verifies expected results under specific conditions |
| Test Suite | A group of related test cases |
| Test Runner | A tool that executes tests and reports results (pytest, JUnit, etc.) |
| Test Fixture | Setup and teardown processing before and after test execution |
| Assertion | A verification statement that compares expected and actual values |
| SUT (System Under Test) | The system or component being tested |
| Test Double | A generic term for substitutes used in place of real objects |
| Test Coverage | The proportion of code executed by tests |
| Regression Test | A test that confirms existing functionality is not broken |
| Flaky Test | A test that produces inconsistent results even under the same conditions |

---

## 2. Test Classification

Software tests are classified from multiple perspectives based on test granularity, purpose, execution timing, and more.
Here we focus primarily on "classification by test level," the most fundamental approach.

### 2.1 The Test Pyramid

The test pyramid, proposed by Mike Cohn,
is a visual representation of the fundamental guidelines for test automation strategy.

```
Test Pyramid:

                    /\
                   /  \
                  / E2E \        <- Few / High cost / Slow / Fragile
                 / Tests  \        Browser operations, full API integration
                /----------\
               /            \
              / Integration  \    <- Moderate quantity / cost / speed
             /    Tests       \    DB connections, API integration, inter-service communication
            /------------------\
           /                    \
          /    Unit Tests        \  <- Many / Low cost / Fast / Stable
         /                        \   Function/class level, isolated with mocks
        /--------------------------\

  Recommended ratios:
  +----------------+--------+--------------+------------------+
  | Layer          | Ratio  | Exec Speed   | Maintenance Cost |
  +----------------+--------+--------------+------------------+
  | Unit           | 70%    | milliseconds | Low              |
  | Integration    | 20%    | sec to min   | Medium           |
  | E2E            | 10%    | min to tens  | High             |
  +----------------+--------+--------------+------------------+
```

The meaning of this pyramid is clear.
Write a large number of low-cost, fast unit tests as the foundation,
verify key component interactions with integration tests,
and write only a small number of E2E tests focused on business-critical scenarios.

When the pyramid becomes inverted (ice cream cone shape),
the entire test suite becomes slow, unstable, and expensive to maintain.
This is an anti-pattern commonly seen in many projects.

```
Anti-pattern: Ice Cream Cone

        +----------------------------+
        |    Manual Tests (many)     |  <- Only verified manually
        +----------------------------+
        |    E2E Tests (many)        |  <- Slow / Fragile
        +----------------------------+
        |  Integration Tests (few)   |
        +----------------------------+
        | Unit Tests (very few)      |  <- Almost none written
        +----------------------------+

  Problems:
  - Entire test suite takes tens of minutes to hours to run
  - Flaky tests occur frequently, eroding trust in test results
  - Developers stop running tests -- a vicious cycle
  - Difficult to pinpoint bug causes (too coarse-grained)
```

### 2.2 Unit Tests

Unit tests verify the smallest units, such as functions and methods, in isolation.
They form the foundation of the test pyramid, and it is recommended that they comprise 70% or more of the entire test suite.

#### Characteristics

- **Fast**: Each test completes in milliseconds
- **Isolated**: External dependencies (DB, network, filesystem) are replaced with mocks
- **Deterministic**: Always returns the same result for the same input
- **Independent**: No order dependency between tests

#### AAA Pattern

The structure of unit tests follows the standard AAA (Arrange-Act-Assert) pattern.

```python
# ===== Code Example 1: Unit Testing with pytest and the AAA Pattern =====

import pytest
from decimal import Decimal


# --- Code Under Test ---

class Money:
    """A value object for handling monetary amounts."""

    def __init__(self, amount: int, currency: str = "JPY"):
        if amount < 0:
            raise ValueError("Amount must be 0 or greater")
        if currency not in ("JPY", "USD", "EUR"):
            raise ValueError(f"Unsupported currency: {currency}")
        self.amount = amount
        self.currency = currency

    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot add amounts in different currencies")
        return Money(self.amount + other.amount, self.currency)

    def multiply(self, factor: int) -> "Money":
        if factor < 0:
            raise ValueError("Factor must be 0 or greater")
        return Money(self.amount * factor, self.currency)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount and self.currency == other.currency

    def __repr__(self) -> str:
        return f"Money({self.amount}, '{self.currency}')"


# --- Test Code ---

class TestMoney:
    """Unit tests for the Money class."""

    # === Normal Case Tests ===

    def test_creation_succeeds_with_valid_amount(self):
        # Arrange
        amount = 1000
        currency = "JPY"

        # Act
        money = Money(amount, currency)

        # Assert
        assert money.amount == 1000
        assert money.currency == "JPY"

    def test_addition_adds_same_currency_amounts(self):
        # Arrange
        money1 = Money(1000, "JPY")
        money2 = Money(500, "JPY")

        # Act
        result = money1.add(money2)

        # Assert
        assert result == Money(1500, "JPY")

    def test_multiplication_multiplies_by_positive_factor(self):
        # Arrange
        money = Money(100, "USD")

        # Act
        result = money.multiply(3)

        # Assert
        assert result == Money(300, "USD")

    def test_equality_equal_when_same_amount_and_currency(self):
        assert Money(500, "JPY") == Money(500, "JPY")

    def test_equality_not_equal_when_different_amounts(self):
        assert Money(500, "JPY") != Money(600, "JPY")

    def test_equality_not_equal_when_different_currencies(self):
        assert Money(500, "JPY") != Money(500, "USD")

    # === Error Case Tests ===

    def test_creation_raises_for_negative_amount(self):
        with pytest.raises(ValueError, match="Amount must be 0 or greater"):
            Money(-100, "JPY")

    def test_creation_raises_for_unsupported_currency(self):
        with pytest.raises(ValueError, match="Unsupported currency"):
            Money(100, "GBP")

    def test_addition_raises_for_different_currencies(self):
        money_jpy = Money(1000, "JPY")
        money_usd = Money(10, "USD")
        with pytest.raises(ValueError, match="different currencies"):
            money_jpy.add(money_usd)

    def test_multiplication_raises_for_negative_factor(self):
        money = Money(100, "JPY")
        with pytest.raises(ValueError, match="Factor must be 0 or greater"):
            money.multiply(-1)

    # === Boundary Value Tests ===

    def test_creation_zero_amount_is_valid(self):
        money = Money(0, "JPY")
        assert money.amount == 0

    def test_addition_of_two_zero_amounts(self):
        result = Money(0, "JPY").add(Money(0, "JPY"))
        assert result == Money(0, "JPY")

    def test_multiplication_by_zero_yields_zero(self):
        result = Money(1000, "JPY").multiply(0)
        assert result == Money(0, "JPY")
```

#### Test Naming Conventions

Test names should clearly indicate "what is being tested," "under what conditions," and "what is expected."

| Naming Style | Example | Characteristics |
|---|---|---|
| Descriptive method name | `test_addition_adds_same_currency` | High readability. Works with pytest |
| Given-When-Then | `test_given_same_currency_when_add_then_returns_sum` | BDD-style. Conditions are clear |
| should style | `test_add_should_return_sum_for_same_currency` | Expected behavior is clear |
| method_condition_expected | `test_add_sameCurrency_returnsSum` | Common in Java/JUnit |

### 2.3 Integration Tests

Integration tests verify that multiple components work correctly together
when combined.

#### Differences from Unit Tests

| Aspect | Unit Tests | Integration Tests |
|--------|-----------|------------------|
| Test target | Single function/class | Interaction of multiple components |
| External dependencies | Isolated with mocks | Uses actual resources |
| Execution speed | Milliseconds | Seconds to minutes |
| Test environment | No special setup required | Requires DB, API server, etc. |
| Bugs detected | Logic errors | Connection settings, data conversion, protocol mismatches |
| Stability | High (deterministic) | Somewhat lower (environment-dependent) |

#### Integration Test Targets

- CRUD operations with databases
- Communication with external APIs
- Message queue sending/receiving
- File system read/write operations
- Authentication/authorization flows

```python
# ===== Code Example 2: Integration Tests with pytest + SQLAlchemy =====

import pytest
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


class User(Base):
    """User table model."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)


class UserRepository:
    """Repository responsible for user data access."""

    def __init__(self, session):
        self.session = session

    def add(self, name: str, email: str) -> User:
        user = User(name=name, email=email)
        self.session.add(user)
        self.session.commit()
        return user

    def find_by_email(self, email: str) -> User | None:
        return self.session.query(User).filter_by(email=email).first()

    def find_all(self) -> list[User]:
        return self.session.query(User).all()

    def delete(self, user: User) -> None:
        self.session.delete(user)
        self.session.commit()


# --- Test Fixtures ---

@pytest.fixture
def db_session():
    """Provides an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def user_repo(db_session):
    """Provides a UserRepository instance."""
    return UserRepository(db_session)


# --- Integration Tests ---

class TestUserRepository:
    """Integration tests for UserRepository. Uses an actual DB (in-memory SQLite)."""

    def test_can_add_and_retrieve_user(self, user_repo):
        # Arrange & Act
        user_repo.add("Alice", "alice@example.com")

        # Assert
        found = user_repo.find_by_email("alice@example.com")
        assert found is not None
        assert found.name == "Alice"
        assert found.email == "alice@example.com"

    def test_searching_nonexistent_email_returns_none(self, user_repo):
        result = user_repo.find_by_email("nobody@example.com")
        assert result is None

    def test_can_register_multiple_users_and_retrieve_all(self, user_repo):
        user_repo.add("Alice", "alice@example.com")
        user_repo.add("Bob", "bob@example.com")
        user_repo.add("Charlie", "charlie@example.com")

        users = user_repo.find_all()
        assert len(users) == 3

    def test_can_delete_user(self, user_repo):
        user = user_repo.add("Alice", "alice@example.com")
        user_repo.delete(user)

        assert user_repo.find_by_email("alice@example.com") is None

    def test_duplicate_email_raises_exception(self, user_repo):
        user_repo.add("Alice", "alice@example.com")
        with pytest.raises(Exception):  # IntegrityError
            user_repo.add("Alice2", "alice@example.com")
```

### 2.4 E2E Tests (End-to-End Tests)

E2E tests verify the entire application from the end user's perspective.
They use browser automation tools (Playwright, Cypress, Selenium) to
simulate actual user operations.

#### E2E Test Scope

```
E2E Test Scope:

  Browser/Client         Server              Database
  +--------------+    +--------------+    +--------------+
  | User actions |---->| API/Web      |---->| Data         |
  | Page nav.    |<----| Business     |<----| persistence  |
  | Display      |    | logic        |    | Query exec.  |
  | verification |    | Auth         |    |              |
  +--------------+    +--------------+    +--------------+
       ^                                         |
       |          E2E Test Scope                  |
       +------------------------------------------+
       Verifies across all layers
```

#### When to Write E2E Tests

- Critical user flows like Login -> Product Search -> Add to Cart -> Checkout
- Features directly tied to legal regulations or business requirements (payments, personal data processing)
- Areas where serious incidents have occurred in the past

#### When to Avoid E2E Tests

- Individual validation rules (unit tests are sufficient)
- Exhaustive coverage of all UI patterns (cost is not justified)
- Frequently changing UI elements (tests become fragile)

### 2.5 Acceptance Tests

Acceptance tests confirm that the system meets business requirements.
Written from the perspective of the customer or product owner,
they serve as the criteria for determining "Is this feature done?"

Acceptance tests are often confused with E2E tests, but they are different concepts.

| Aspect | E2E Tests | Acceptance Tests |
|--------|-----------|-----------------|
| Purpose | Technical verification of the entire system | Verification of business requirement fulfillment |
| Perspective | Developers | Customers / Product Owners |
| Authors | QA Engineers / Developers | PO and developers collaborating |
| Implementation tools | Playwright, Cypress, etc. | Cucumber, Behave, etc. (BDD tools) |
| Execution frequency | Automated in CI | Sprint reviews, etc. |

---

## 3. Test-Driven Development (TDD)

### 3.1 The Basic TDD Cycle

TDD (Test-Driven Development) is a methodology systematized by Kent Beck,
with "writing tests first" at its core.

```
TDD Red-Green-Refactor Cycle:

        +-----------------------------------+
        |                                   |
        v                                   |
  +-----------+    +-----------+    +-----------+
  |   RED     |--->|   GREEN   |--->| REFACTOR  |
  |           |    |           |    |           |
  | Write a   |    | Write the |    | Clean up  |
  | failing   |    | minimal   |    | the code  |
  | test      |    | code to   |    | (tests    |
  |           |    | pass      |    |  still    |
  |           |    | the test  |    |  pass)    |
  +-----------+    +-----------+    +-----------+

  Details of each phase:

  RED:
    1. Write a test for functionality that doesn't exist yet
    2. Run the test and confirm it fails
    3. Confirm the failure reason is "the feature is not yet implemented"

  GREEN:
    1. Write the minimal code to pass the test
    2. Don't worry about elegance or efficiency at all
    3. Hard-coded values are fine (as a first step)
    4. Confirm the test passes

  REFACTOR:
    1. Eliminate duplication
    2. Improve naming
    3. Apply design patterns
    4. Always confirm tests continue to pass
```

### 3.2 TDD in Practice: FizzBuzz

Here is a demonstration of the TDD flow using FizzBuzz as the subject.

```python
# ===== Code Example 3: Implementing FizzBuzz with TDD =====

# --- Step 1: RED -- Write the first test ---
# Test file: test_fizzbuzz.py

def test_returns_string_1_when_given_1():
    assert fizzbuzz(1) == "1"

# At this point the fizzbuzz function doesn't exist, so the test fails (NameError)


# --- Step 2: GREEN -- Write the minimal code to pass ---
# Production code: fizzbuzz.py

def fizzbuzz(n: int) -> str:
    return str(n)

# Run tests -> PASSED


# --- Step 3: RED -- Add the next test ---

def test_returns_Fizz_when_given_3():
    assert fizzbuzz(3) == "Fizz"

# Run tests -> FAILED ("3" != "Fizz")


# --- Step 4: GREEN -- Make the test pass ---

def fizzbuzz(n: int) -> str:
    if n % 3 == 0:
        return "Fizz"
    return str(n)

# Run tests -> Both PASSED


# --- Step 5: RED -- Add another test ---

def test_returns_Buzz_when_given_5():
    assert fizzbuzz(5) == "Buzz"

# Run tests -> FAILED


# --- Step 6: GREEN ---

def fizzbuzz(n: int) -> str:
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)

# Run tests -> All 3 PASSED


# --- Step 7: RED -- Test for multiples of 15 ---

def test_returns_FizzBuzz_when_given_15():
    assert fizzbuzz(15) == "FizzBuzz"

# Run tests -> FAILED ("Fizz" != "FizzBuzz")


# --- Step 8: GREEN ---

def fizzbuzz(n: int) -> str:
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)

# Run tests -> All 4 PASSED


# --- Step 9: REFACTOR -- Clean up the code ---

def fizzbuzz(n: int) -> str:
    """Return the FizzBuzz result for n.

    - Multiples of 3 return "Fizz"
    - Multiples of 5 return "Buzz"
    - Multiples of 15 return "FizzBuzz"
    - All others return the string representation of the number
    """
    result = ""
    if n % 3 == 0:
        result += "Fizz"
    if n % 5 == 0:
        result += "Buzz"
    return result or str(n)

# Run tests -> All 4 PASSED (tests still pass after refactoring)


# --- Final Test Suite ---

import pytest

class TestFizzBuzz:
    """FizzBuzz test suite."""

    @pytest.mark.parametrize("input_val, expected", [
        (1, "1"),
        (2, "2"),
        (3, "Fizz"),
        (5, "Buzz"),
        (6, "Fizz"),
        (10, "Buzz"),
        (15, "FizzBuzz"),
        (30, "FizzBuzz"),
        (7, "7"),
    ])
    def test_fizzbuzz(self, input_val, expected):
        assert fizzbuzz(input_val) == expected
```

### 3.3 Benefits and Caveats of TDD

**Benefits:**

1. **Design is guided toward usable APIs** -- Writing tests first forces you to think about the interface from the user's perspective
2. **Regression tests accumulate automatically** -- The test suite grows alongside development
3. **Prevents over-implementation (YAGNI principle)** -- Only functionality required by tests is implemented
4. **Confidence in changes** -- Tests act as a safety net during refactoring
5. **Reduced debugging time** -- Tests fail at the point a bug is introduced, making it easy to pinpoint the cause

**Caveats:**

1. **TDD need not be applied to everything** -- For exploratory prototyping or UI implementation, it may be more efficient to write tests after the design is settled
2. **Test maintenance cost** -- Test code requires maintenance just like production code
3. **Learning cost** -- Effective TDD practice requires skills in test design and refactoring
4. **Excessive mocking** -- Those new to TDD tend to overuse mocks to make tests pass, leading to tests coupled to implementation details

---

## 4. Behavior-Driven Development (BDD)

### 4.1 BDD Overview

BDD (Behavior-Driven Development), proposed by Dan North,
is a reinterpretation of TDD from a business perspective.
Scenarios are written as "behavior specifications" rather than "tests."

The core of BDD is that the "Three Amigos" -- developers, QA, and business stakeholders --
discuss scenarios using a common language (ubiquitous language),
and those scenarios are made directly executable as automated tests.

### 4.2 Gherkin Notation

BDD scenarios are written in Gherkin, a notation close to natural language.

```gherkin
# ===== Example of scenario description in Gherkin =====

Feature: Shopping Cart
  As an online shop customer
  I want to add products to my cart and check the total amount
  I also want to be able to change quantities and remove products before purchase

  Background:
    Given the following products are registered in the catalog
      | Product Name         | Unit Price |
      | Python Introduction  | 3000       |
      | Go in Practice       | 3500       |
      | Rust Introduction    | 4000       |

  Scenario: Add a product to the cart
    Given the cart is empty
    When I add 1 "Python Introduction" to the cart
    Then the number of items in the cart is 1
    And the total amount is 3000 yen

  Scenario: Add multiple products to the cart
    Given the cart is empty
    When I add 2 "Python Introduction" to the cart
    And I add 1 "Go in Practice" to the cart
    Then the number of items in the cart is 3
    And the total amount is 9500 yen

  Scenario: Remove a product from the cart
    Given the cart contains 1 "Python Introduction"
    When I remove "Python Introduction" from the cart
    Then the cart is empty
    And the total amount is 0 yen

  Scenario Outline: Total amount calculation by quantity
    Given the cart is empty
    When I add <quantity> "<product_name>" to the cart
    Then the total amount is <expected_amount> yen

    Examples:
      | product_name         | quantity | expected_amount |
      | Python Introduction  | 1        | 3000            |
      | Python Introduction  | 3        | 9000            |
      | Go in Practice       | 2        | 7000            |
      | Rust Introduction    | 1        | 4000            |
```

### 4.3 Comparison of TDD and BDD

| Aspect | TDD | BDD |
|--------|-----|-----|
| Starting point | Technical tests | Business behavior |
| Authors | Developers | Developers + PO + QA |
| Notation | Programming language | Gherkin (natural language-like) |
| Granularity | Function/method level | User story level |
| Tools | pytest, JUnit, etc. | Cucumber, Behave, SpecFlow, etc. |
| Primary use | Verification of implementation correctness | Requirements agreement and verification |
| Shared understanding | Within the dev team | Entire team including business |

---

## 5. Test Techniques

Test techniques are methodologies for designing test cases efficiently and effectively.
They help select test cases most likely to find bugs from the infinite combinations of inputs.

### 5.1 Equivalence Partitioning

This technique divides the input domain into "groups with the same behavior (equivalence classes)"
and selects one representative value from each class for testing.

```
Equivalence Partitioning Example: Pricing by Age

  Input: Age (assuming integers from 0 to 150)
  Rules:
    - 0-5 years    -> Free
    - 6-12 years   -> Child rate
    - 13-64 years  -> Adult rate
    - 65-150 years -> Senior rate
    - Other        -> Error

  Equivalence Classes:
  +-----------------------------------------------------------+
  | Invalid | Free  | Child | Adult  | Senior | Invalid       |
  | (neg.)  | 0-5   | 6-12  | 13-64  | 65-150 | (exceeds)    |
  |  < 0    |       |       |        |        |  > 150       |
  +-----------------------------------------------------------+

  Representative values:  -1     3      9      30      80       200
```

### 5.2 Boundary Value Analysis

The area near equivalence class boundaries is prone to off-by-one errors.
Boundary value analysis tests the boundary values and their adjacent values.

```
Boundary Value Analysis Example: Pricing by Age

  Boundary values to test:

  Invalid/Free boundary:    -1,  0,  1
  Free/Child boundary:       4,  5,  6,  7
  Child/Adult boundary:     11, 12, 13, 14
  Adult/Senior boundary:    63, 64, 65, 66
  Senior/Invalid boundary: 149, 150, 151

  Test Case Table:
  +------+------------------+---------------------+
  | Input| Expected Category| Test Purpose        |
  +------+------------------+---------------------+
  |  -1  | Error            | Below lower bound   |
  |   0  | Free             | Lower boundary      |
  |   1  | Free             | Lower bound + 1     |
  |   5  | Free             | Upper boundary      |
  |   6  | Child            | Next class lower    |
  |  12  | Child            | Upper boundary      |
  |  13  | Adult            | Next class lower    |
  |  64  | Adult            | Upper boundary      |
  |  65  | Senior           | Next class lower    |
  | 150  | Senior           | Upper boundary      |
  | 151  | Error            | Above upper bound   |
  +------+------------------+---------------------+
```

### 5.3 Decision Tables

When behavior changes based on combinations of multiple conditions,
decision tables are used to exhaustively enumerate all combinations.

```
Decision Table Example: EC Site Shipping Cost Calculation

  Conditions:
    C1: Is member?              (Yes/No)
    C2: Total >= 3000 yen?      (Yes/No)
    C3: Remote island?          (Yes/No)

  +------------------+-----+-----+-----+-----+-----+------+-----+------+
  | Rule Number      |  1  |  2  |  3  |  4  |  5  |  6   |  7  |  8   |
  +------------------+-----+-----+-----+-----+-----+------+-----+------+
  | C1: Member       | Yes | Yes | Yes | Yes | No  | No   | No  | No   |
  | C2: >= 3000 yen  | Yes | Yes | No  | No  | Yes | Yes  | No  | No   |
  | C3: Remote island| No  | Yes | No  | Yes | No  | Yes  | No  | Yes  |
  +------------------+-----+-----+-----+-----+-----+------+-----+------+
  | Shipping cost    | 0   | 500 | 300 | 800 | 500 | 1000 | 800 | 1500 |
  +------------------+-----+-----+-----+-----+-----+------+-----+------+
```

### 5.4 Pairwise Testing

When the number of conditions increases, the total number of combinations explodes.
Pairwise testing dramatically reduces the number of test cases based on the criterion
"cover all value combinations for any two factors at least once."

This is based on research findings that most real bugs are caused by the interaction of two factors.

| Number of factors | All combinations | Pairwise | Reduction rate |
|-------------------|-----------------|----------|---------------|
| 3 factors x 3 values | 27 | 9-12 | 56-67% |
| 4 factors x 3 values | 81 | 9-15 | 81-89% |
| 10 factors x 3 values | 59,049 | 15-20 | 99.97% |
| 13 factors x 3 values | 1,594,323 | 15-20 | 99.999% |

Tools such as PICT (by Microsoft) and AllPairs are used to generate pairwise tests.

---

## 6. Mocks and Stubs

### 6.1 Classification of Test Doubles

Here we organize the types of test doubles based on Martin Fowler's classification.

```
Test Double Classification:

  +---------------------------------------------------------+
  |                   Test Double                            |
  |                                                         |
  |  +---------+  +---------+  +---------+  +------------+  |
  |  | Dummy   |  | Stub    |  | Spy     |  | Mock       |  |
  |  +---------+  +---------+  +---------+  +------------+  |
  |  |Just fill|  |Returns  |  |Records  |  |Verifies    |  |
  |  |argument |  |fixed    |  |calls    |  |expected    |  |
  |  |slots   |  |values   |  |         |  |calls       |  |
  |  +---------+  +---------+  +---------+  +------------+  |
  |                                                         |
  |  +-----------------+                                    |
  |  | Fake            |                                    |
  |  +-----------------+                                    |
  |  |Has a simplified |                                    |
  |  |implementation   |                                    |
  |  |(in-memory DB)   |                                    |
  |  +-----------------+                                    |
  +---------------------------------------------------------+
```

| Type | Purpose | Behavior | Verification |
|------|---------|----------|-------------|
| Dummy | Passed to fill argument slots | Does nothing. Throws exception if called | None |
| Stub | Controls indirect inputs | Returns pre-defined fixed values | None |
| Spy | Records indirect outputs | Retains call history | Post-verification of call history |
| Mock | Verifies expected interactions | Fails on unexpected calls | Verifies interactions |
| Fake | Provides a lightweight alternative implementation | Working simplified implementation (in-memory DB, etc.) | None |

### 6.2 Practical Use of unittest.mock

```python
# ===== Code Example 4: Practical Mocks and Stubs with unittest.mock =====

from unittest.mock import Mock, patch, MagicMock
import pytest
from dataclasses import dataclass
from typing import Protocol


# --- Production Code ---

@dataclass
class WeatherData:
    """Data class representing weather data."""
    city: str
    temperature: float
    humidity: float
    description: str


class WeatherApiClient:
    """Handles communication with the external weather API."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_weather(self, city: str) -> dict:
        """Fetch weather data from the external API (actual HTTP communication)."""
        # In production, this would call the API using requests.get(), etc.
        raise NotImplementedError("Production code performs HTTP communication")


class WeatherService:
    """Handles business logic for weather information."""

    def __init__(self, api_client: WeatherApiClient):
        self.api_client = api_client

    def get_weather(self, city: str) -> WeatherData:
        """Fetch weather for the specified city and convert to WeatherData."""
        raw = self.api_client.fetch_weather(city)
        return WeatherData(
            city=city,
            temperature=raw["main"]["temp"],
            humidity=raw["main"]["humidity"],
            description=raw["weather"][0]["description"],
        )

    def is_hot(self, city: str, threshold: float = 30.0) -> bool:
        """Determine whether the specified city is hot."""
        weather = self.get_weather(city)
        return weather.temperature >= threshold

    def compare_temperature(self, city1: str, city2: str) -> str:
        """Compare the temperature of two cities."""
        w1 = self.get_weather(city1)
        w2 = self.get_weather(city2)
        if w1.temperature > w2.temperature:
            return f"{city1} is hotter"
        elif w1.temperature < w2.temperature:
            return f"{city2} is hotter"
        else:
            return "Same temperature"


# --- Test Code (Stub Example) ---

class TestWeatherService:
    """Tests for WeatherService. External API is replaced with stubs."""

    def _create_service_with_stub(self, stub_response: dict) -> WeatherService:
        """Create a service with a stubbed API client."""
        mock_client = Mock(spec=WeatherApiClient)
        mock_client.fetch_weather.return_value = stub_response
        return WeatherService(mock_client)

    def _sample_response(self, temp: float = 25.0, humidity: float = 60.0,
                         desc: str = "clear sky") -> dict:
        """Generate a dummy response for testing."""
        return {
            "main": {"temp": temp, "humidity": humidity},
            "weather": [{"description": desc}],
        }

    def test_correctly_converts_weather_data(self):
        # Arrange: Configure stub to return a fixed response
        service = self._create_service_with_stub(
            self._sample_response(temp=25.0, humidity=60.0, desc="sunny")
        )

        # Act
        weather = service.get_weather("Tokyo")

        # Assert
        assert weather.city == "Tokyo"
        assert weather.temperature == 25.0
        assert weather.humidity == 60.0
        assert weather.description == "sunny"

    def test_determines_hot_when_above_threshold(self):
        service = self._create_service_with_stub(
            self._sample_response(temp=35.0)
        )
        assert service.is_hot("Tokyo", threshold=30.0) is True

    def test_determines_not_hot_when_below_threshold(self):
        service = self._create_service_with_stub(
            self._sample_response(temp=25.0)
        )
        assert service.is_hot("Tokyo", threshold=30.0) is False


# --- Test Code (Mock Example: Verifying Calls) ---

class TestWeatherServiceInteraction:
    """Verify WeatherService interactions using mocks."""

    def test_get_weather_calls_api_client_correctly(self):
        # Arrange
        mock_client = Mock(spec=WeatherApiClient)
        mock_client.fetch_weather.return_value = {
            "main": {"temp": 25.0, "humidity": 60.0},
            "weather": [{"description": "sunny"}],
        }
        service = WeatherService(mock_client)

        # Act
        service.get_weather("Osaka")

        # Assert: Verify the API client was called with the correct argument
        mock_client.fetch_weather.assert_called_once_with("Osaka")

    def test_compare_temperature_calls_api_twice(self):
        # Arrange
        mock_client = Mock(spec=WeatherApiClient)
        mock_client.fetch_weather.side_effect = [
            {"main": {"temp": 30.0, "humidity": 50.0},
             "weather": [{"description": "sunny"}]},
            {"main": {"temp": 25.0, "humidity": 70.0},
             "weather": [{"description": "cloudy"}]},
        ]
        service = WeatherService(mock_client)

        # Act
        result = service.compare_temperature("Tokyo", "Sapporo")

        # Assert
        assert result == "Tokyo is hotter"
        assert mock_client.fetch_weather.call_count == 2
```

### 6.3 Beware of Over-Mocking

Excessive use of mocks causes the following problems.

1. **Tests become coupled to implementation details** -- Tests break even from internal structural changes
2. **False sense of security** -- Tests pass as long as mocks are correctly configured, but actual integration may be broken
3. **Decreased test readability** -- Complex mock setup code makes it difficult to understand what is being tested

**Rule of thumb**: "Don't mock what you don't own."
Rather than directly mocking external libraries or API clients,
create a thin wrapper (adapter) and mock that adapter.

---

## 7. Test Automation in CI/CD

### 7.1 Overview of Test Automation

```
Test Automation in CI/CD Pipelines:

  Code Change
      |
      v
  +-----------------------------------------------------+
  |                  CI Pipeline                          |
  |                                                     |
  |  +----------+  +----------+  +----------+           |
  |  | Lint /   |->| Unit     |->| Integ.   |           |
  |  | Static   |  | Tests    |  | Tests    |           |
  |  | Analysis |  | (secs)   |  | (mins)   |           |
  |  | (secs)   |  |          |  |          |           |
  |  +----------+  +----------+  +----------+           |
  |       |              |             |                |
  |       v              v             v                |
  |  +----------+  +----------+  +----------+           |
  |  | Security |  | Coverage |  | E2E      |           |
  |  | Scan     |  | Report   |  | Tests    |           |
  |  | (mins)   |  | Gen.     |  | (tens of |           |
  |  |          |  |          |  |  mins)   |           |
  |  +----------+  +----------+  +----------+           |
  |                      |                              |
  |                      v                              |
  |               +----------+                          |
  |               | Build /  |                          |
  |               | Package  |                          |
  |               +----------+                          |
  +-----------------------------------------------------+
      |
      v
  +-----------------------------------------------------+
  |                  CD Pipeline                          |
  |                                                     |
  |  +----------+  +----------+  +----------+           |
  |  | Staging  |->| Smoke    |->| Prod.    |           |
  |  | Deploy   |  | Tests    |  | Deploy   |           |
  |  +----------+  +----------+  +----------+           |
  +-----------------------------------------------------+
```

### 7.2 Test Automation with GitHub Actions

```yaml
# .github/workflows/test.yml

name: Automated Test Execution

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    name: Lint & Static Analysis
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff mypy
      - run: ruff check .
      - run: mypy src/

  unit-test:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: lint
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[test]"
      - run: pytest tests/unit/ -v --cov=src/ --cov-report=xml
      - uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  integration-test:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-test
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: testdb
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[test]"
      - run: pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql://testuser:testpass@localhost:5432/testdb

  e2e-test:
    name: E2E Tests
    runs-on: ubuntu-latest
    needs: integration-test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[test]"
      - run: npx playwright install --with-deps chromium
      - run: pytest tests/e2e/ -v --headed=false
```

### 7.3 Best Practices for Test Automation

1. **Fast feedback** -- Run unit tests on every PR. Ideally, all tests complete within 5 minutes
2. **Parallel test execution** -- Use pytest-xdist or similar to parallelize and reduce execution time
3. **Stop on first failure** -- Use the `pytest -x` (stop at first failure) option to eliminate unnecessary waiting
4. **Test result caching** -- Skip tests for modules that haven't changed
5. **Flaky test management** -- Isolate unstable tests and fix them regularly

---

## 8. Code Coverage

### 8.1 Types of Coverage Metrics

| Metric | Definition | What is measured |
|--------|-----------|-----------------|
| Line Coverage | Percentage of lines executed by tests | Each line of source code |
| Branch Coverage | Percentage of branches traversed by tests | Each branch of if/else, switch |
| Function Coverage | Percentage of functions called by tests | Each defined function |
| Condition Coverage | Percentage of conditions tested for both true and false | Each sub-condition in compound conditions |
| Path Coverage | Percentage of execution paths traversed by tests | All execution paths |
| MC/DC | Verification that each condition independently affects the decision outcome | Aerospace, automotive safety standards |

### 8.2 Coverage Targets

| Coverage Level | Meaning | Application |
|---------------|---------|-------------|
| 80% or higher | Standard target. Recommended for most projects | General web applications |
| 90% or higher | High quality. Recommended for core libraries | Libraries, frameworks |
| 95% or higher | Very high quality. Maintenance cost is also high | Payment processing, healthcare |
| 100% | Ideal but often not cost-effective in practice | Safety-critical systems |

### 8.3 Limitations of Coverage

**Coverage measures "test thoroughness" but cannot measure "test quality."**

```python
# Example of 100% coverage that fails to detect a bug

def divide(a: int, b: int) -> float:
    return a / b  # No handling for b=0

def test_divide():
    assert divide(10, 2) == 5.0
    # This single test achieves 100% line coverage,
    # but misses the b=0 case, so the bug goes undetected
```

High coverage is a necessary condition but not a sufficient one.
The following points must always be kept in mind.

1. **Coverage only shows that code was "executed"** -- Whether correct results are returned is a separate issue
2. **Pursuing coverage numbers alone leads to low-quality tests** -- Tests without assertions or meaningless test cases
3. **Some quality factors cannot be measured by coverage** -- Performance, usability, security, etc.
4. **Absence of edge cases is not reflected in coverage** -- 100% can be achieved with only normal cases

### 8.4 Coverage Measurement with pytest-cov

```bash
# Generate a coverage report
pytest --cov=src/ --cov-report=term-missing --cov-report=html

# Example output:
# Name                     Stmts   Miss  Cover   Missing
# -------------------------------------------------------
# src/money.py                25      2    92%   18, 22
# src/weather_service.py      40      5    88%   35-39
# -------------------------------------------------------
# TOTAL                       65      7    89%
```

---

## 9. Property-Based Testing

### 9.1 What Is Property-Based Testing

In conventional tests (example-based tests),
the test author manually selects specific input values and expected values.
In property-based testing, inputs are automatically generated randomly,
and "properties that should hold for any input" are verified.

| Comparison | Example-Based Testing | Property-Based Testing |
|-----------|----------------------|----------------------|
| Input determination | Manually selected by test author | Automatically generated by framework |
| Number of test cases | Several to dozens | Hundreds to thousands (automatic) |
| Bug discovery power | Misses unforeseen cases | Discovers unexpected input patterns |
| Test description | Specific inputs and expected values | Abstract properties |
| Reproducibility | Always the same | Reproducible via seed |
| Shrinking | None | Automatically searches for minimal counterexample |

### 9.2 Practice with Hypothesis

```python
# ===== Code Example 5: Property-Based Testing with Hypothesis =====

from hypothesis import given, assume, settings, example
from hypothesis import strategies as st
import pytest


# --- Code Under Test ---

def sort_list(lst: list[int]) -> list[int]:
    """Sort and return a list."""
    return sorted(lst)


def reverse_string(s: str) -> str:
    """Reverse a string."""
    return s[::-1]


def encode_decode(text: str) -> str:
    """Encode to UTF-8 and decode back."""
    return text.encode("utf-8").decode("utf-8")


def clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp value to the range [min_val, max_val]."""
    if min_val > max_val:
        raise ValueError("min_val must be less than or equal to max_val")
    return max(min_val, min(value, max_val))


# --- Property-Based Tests ---

class TestSortListProperties:
    """Property-based tests for sort_list."""

    @given(st.lists(st.integers()))
    def test_sorted_result_has_same_length_as_input(self, lst):
        """Property: Sorting does not change the number of elements."""
        result = sort_list(lst)
        assert len(result) == len(lst)

    @given(st.lists(st.integers()))
    def test_sorted_result_is_in_ascending_order(self, lst):
        """Property: Each element in the sorted result is >= the previous element."""
        result = sort_list(lst)
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]

    @given(st.lists(st.integers()))
    def test_sorted_result_contains_same_elements_as_input(self, lst):
        """Property: Sorting does not change elements (only rearranges them)."""
        result = sort_list(lst)
        assert sorted(result) == sorted(lst)

    @given(st.lists(st.integers()))
    def test_sort_is_idempotent(self, lst):
        """Property: Sorting twice produces the same result."""
        once = sort_list(lst)
        twice = sort_list(once)
        assert once == twice


class TestReverseStringProperties:
    """Property-based tests for reverse_string."""

    @given(st.text())
    def test_double_reversal_returns_original(self, s):
        """Property: Reversing twice is an identity operation."""
        assert reverse_string(reverse_string(s)) == s

    @given(st.text())
    def test_reversed_result_has_same_length(self, s):
        """Property: Reversing does not change the length."""
        assert len(reverse_string(s)) == len(s)


class TestClampProperties:
    """Property-based tests for clamp."""

    @given(
        st.integers(min_value=-1000, max_value=1000),
        st.integers(min_value=-1000, max_value=0),
        st.integers(min_value=0, max_value=1000),
    )
    def test_result_is_always_within_range(self, value, min_val, max_val):
        """Property: The result of clamp is always within [min_val, max_val]."""
        assume(min_val <= max_val)
        result = clamp(value, min_val, max_val)
        assert min_val <= result <= max_val

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
    )
    def test_value_within_range_is_returned_unchanged(self, value, bound):
        """Property: A value already within range is not changed."""
        min_val = 0
        max_val = max(value, bound)
        min_val_actual = min(0, value)
        # If value is within [min_val, max_val], the result is value itself
        result = clamp(value, min_val, max_val)
        if min_val <= value <= max_val:
            assert result == value


class TestEncodeDecodeProperties:
    """Property-based tests for encode_decode."""

    @given(st.text())
    def test_round_trip(self, text):
        """Property: Encoding then decoding returns the original."""
        assert encode_decode(text) == text
```

### 9.3 How to Find Properties

When writing property-based tests, knowing which properties to verify
is a common stumbling block for beginners. Here are representative patterns.

| Pattern | Description | Example |
|---------|-----------|---------|
| Round-trip | encode -> decode returns the original | JSON, Base64, encryption |
| Idempotence | Executing twice produces the same result | Sort, normalization, formatting |
| Invariant | Properties preserved before and after an operation | Element count, total sum |
| Monotonicity | Output increases as input increases | Insertion into a sorted list |
| Reference implementation | Compare with a simple but correct implementation | Optimized vs. naive version |
| Inverse function | f(g(x)) == x | push/pop, insert/delete |
| Induction | Induction from small to large inputs | Recursive data structures |

---

## 10. Testing Anti-Patterns

### 10.1 Anti-Pattern Catalog

Here we explain commonly seen anti-patterns in test code.
These are causes of degraded test reliability, maintainability, and readability.

#### Anti-Pattern 1: Test Interdependence (The Order-Dependent Test)

```python
# ===== Anti-pattern: Order dependency between tests =====

# Dangerous: The result of test A affects test B

class SharedState:
    """Globally shared state (anti-pattern)."""
    items = []


class TestBad_OrderDependent:
    """Bad example: Tests depend on execution order."""

    def test_A_add_item(self):
        SharedState.items.append("item1")
        assert len(SharedState.items) == 1

    def test_B_check_item_count(self):
        # Dangerous: Assumes test_A was executed first
        assert len(SharedState.items) == 1  # Depends on test_A

    def test_C_remove_items(self):
        SharedState.items.clear()
        assert len(SharedState.items) == 0
        # May fail if not executed after test_B


# ===== Improved Example: Each test is independent =====

class TestGood_Independent:
    """Good example: Each test is independent."""

    def setup_method(self):
        """Initialize state before each test."""
        self.items = []

    def test_A_add_item(self):
        self.items.append("item1")
        assert len(self.items) == 1

    def test_B_empty_list_has_size_zero(self):
        assert len(self.items) == 0

    def test_C_add_then_clear_results_in_empty(self):
        self.items.append("item1")
        self.items.clear()
        assert len(self.items) == 0
```

**Problems:**
- Results change when test execution order changes
- Cannot be run in parallel
- When one test fails, subsequent tests fail in cascade

**Solutions:**
- Initialize state before each test with fixtures
- Avoid global state; use independent instances for each test
- Use `pytest --randomly` (pytest-randomly plugin) to verify by shuffling order

#### Anti-Pattern 2: The Iceberg Test (The Ice-Cream Cone / The Giant Test)

```python
# ===== Anti-pattern: Too many assertions in one test =====

def test_bad_everything_from_registration_to_purchase():
    """Bad example: Too many verifications packed into one test case."""
    # User registration
    user = register_user("Alice", "alice@example.com")
    assert user.id is not None
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
    assert user.is_active is True

    # Login
    token = login(user.email, "password123")
    assert token is not None
    assert len(token) > 0

    # Product search
    products = search_products("Python")
    assert len(products) > 0
    assert products[0].name == "Python Introduction"

    # Add to cart
    cart = add_to_cart(user.id, products[0].id, quantity=2)
    assert len(cart.items) == 1
    assert cart.total == 6000

    # Checkout
    order = checkout(user.id, cart.id)
    assert order.status == "completed"
    assert order.total == 6000


# ===== Improved Example: Appropriately granular tests =====

class TestUserRegistration:
    def test_register_with_valid_info(self):
        user = register_user("Alice", "alice@example.com")
        assert user.id is not None
        assert user.name == "Alice"

class TestAuthentication:
    def test_login_with_valid_credentials(self):
        # ... (user is prepared via fixture)
        pass

class TestShoppingCart:
    def test_add_product_to_cart(self):
        # ... (user and cart are prepared via fixture)
        pass

class TestCheckout:
    def test_checkout_with_cart_contents(self):
        # ...
        pass
```

**Problems:**
- When the test fails, it is difficult to identify which feature has a problem
- The test name does not indicate what is being tested
- If the first part fails, later tests are skipped

**Solutions:**
- Follow the principle of one concern per test
- Make test names clearly indicate what is being tested
- Prepare preconditions with fixtures and have each test verify only one behavior

#### Anti-Pattern 3: The Flaky Test (The Flickering Test)

A flaky test is a test whose results vary each time it is run against the same code.

**Common causes:**

| Cause | Example | Solution |
|-------|---------|---------|
| Time dependency | Depends on `datetime.now()` | Make time injectable |
| Random dependency | Relies on random values | Fix the seed |
| Concurrency | Race conditions | Add proper synchronization |
| External services | Network latency/failures | Use mocks |
| Shared resources | DB modified by other tests | Reset for each test |
| Timing dependency | Verify after `sleep(1)` | Use polling or event-waiting instead |
| Environment dependency | OS, locale, timezone | Fix the test environment |

#### Anti-Pattern 4: Refactoring Without Tests

Changing code without tests carries a high risk of missing regression bugs.
"Tests enable refactoring" -- refactoring without tests is merely gambling.

**Solutions:**
- Write tests before refactoring (especially for legacy code)
- Use techniques from Michael Feathers' "Working Effectively with Legacy Code"
- Record current behavior with characterization tests before refactoring

#### Anti-Pattern 5: Overly Specific Assertions

```python
# Bad example: Full string comparison of output
def test_bad_report_output():
    result = generate_report(2024, 1)
    assert result == "January 2024 Sales Report\nTotal Sales: $1,234,567\nMonth-over-month: +5.2%\n..."
    # Fails if even one space or newline changes

# Good example: Verify only important information
def test_good_report_contains_total_sales():
    result = generate_report(2024, 1)
    assert "Total Sales" in result
    assert "$1,234,567" in result
```

---

## 11. Test Design Principles

### 11.1 FIRST Principles

Good unit tests follow the FIRST principles.

| Letter | Principle | Description |
|--------|----------|-------------|
| F | Fast | Tests should complete in milliseconds |
| I | Independent | No dependencies between tests |
| R | Repeatable | Returns the same result in any environment |
| S | Self-validating | The test itself determines success/failure |
| T | Timely | Written at the same time as production code |

### 11.2 Test Structure Patterns

#### AAA Pattern (Detailed Review)

```
AAA Pattern Structure:

  def test_when_X_then_Y():
      # --- Arrange (Setup) ----------------
      #   Create the object under test
      #   Prepare data needed for the test
      #   Set up dependency objects
      sut = SystemUnderTest()
      input_data = create_test_data()

      # --- Act (Execute) ------------------
      #   Execute exactly one operation on the SUT
      result = sut.do_something(input_data)

      # --- Assert (Verify) ----------------
      #   Verify the expected result
      #   Ideally one concept per test
      assert result == expected_value
```

#### Given-When-Then Pattern

BDD-oriented notation; essentially the same as AAA but uses business-oriented terms.

| AAA | Given-When-Then | Description |
|-----|----------------|-------------|
| Arrange | Given (precondition) | Setting the initial state |
| Act | When (action) | Operation under test |
| Assert | Then (expected result) | Verification of the result |

### 11.3 Test Fixture Best Practices

```python
# Example of using pytest fixtures

import pytest

@pytest.fixture
def sample_user():
    """Provides a test user object."""
    return User(name="Test User", email="test@example.com")

@pytest.fixture
def authenticated_client(sample_user):
    """Provides an authenticated API client."""
    client = TestClient(app)
    token = create_token(sample_user)
    client.headers["Authorization"] = f"Bearer {token}"
    return client

# Fixture granularity:
#   - Too small: Same setup code duplicated across each test
#   - Too large: Unnecessary setup is included, slowing down tests
#   - Appropriate: Clear what the test is testing, with minimal setup
```

---

## 12. Specialized Testing Techniques

### 12.1 Mutation Testing

Mutation testing is a technique that intentionally introduces mutations (mutants)
into production code and evaluates whether the test suite can detect them.

```
Mutation Testing Flow:

  Original code:
    if age >= 18:
        return "adult"

  Mutant 1:              Mutant 2:
    if age > 18:           if age >= 19:
        return "adult"         return "adult"
    (Changed >= to >)      (Changed 18 to 19)

  If the test suite:
    - Detects (kills) the mutant -> Tests are high quality
    - Misses (lets survive) the mutant -> Tests have gaps
```

In Python, this can be executed with the mutmut tool.

```bash
# Running mutation tests
mutmut run --paths-to-mutate=src/
mutmut results
```

### 12.2 Snapshot Testing

Snapshot testing saves the output of a function as a "snapshot"
and compares it against the previous snapshot on subsequent runs.
It is useful for regression testing of UI components and serialized data.

```python
# Snapshot testing with pytest-snapshot

def test_report_output_snapshot(snapshot):
    result = generate_report(year=2024, month=1)
    snapshot.assert_match(result, "report_2024_01.txt")
    # First run: A snapshot file is generated
    # Subsequent runs: Compared against the saved snapshot
    # On changes: Update with pytest --snapshot-update
```

### 12.3 Contract Testing

In microservice API integration,
contract testing verifies that the "contract" between services is maintained.
Tools like Pact are used.

```
Contract Testing Concept:

  Consumer (Client)            Provider (Server)
  +--------------+            +--------------+
  | Frontend     |  Contract  | Backend      |
  |              |<---------->|              |
  | "When calling|  (Pact)    | API Server   |
  |  GET /users  |            |              |
  |  returns     |            |              |
  |  [{id,name}]"|            |              |
  +--------------+            +--------------+

  Consumer side: Defines expected requests/responses as contracts
  Provider side: Verifies it can respond according to the contracts
  -> Both can be tested independently. Compatibility can be confirmed before deployment
```

---

## 13. Test Frameworks and Tools Comparison

### 13.1 Python Test Framework Comparison

| Framework | Characteristics | Test Writing Style | Fixtures | Parameterization | Plugins |
|---|---|---|---|---|---|
| pytest | Most widely used test framework for Python | Function-based + class-based | `@pytest.fixture` (powerful and flexible) | `@pytest.mark.parametrize` | 1000+ plugins |
| unittest | Included in Python standard library | Class-based (`TestCase` inheritance) | `setUp` / `tearDown` | `subTest` | Limited |
| nose2 | Extension of unittest (successor to nose) | Function-based + class-based | Plugin-based | Parameter plugin | Moderate |
| doctest | Tests written within docstrings | Interactive examples in docstrings | None | None | None |

**Recommendation**: For new projects, **pytest** should be the first choice.
Its rich plugin ecosystem, intuitive fixture mechanism,
and clear assertion failure messages are its strengths.

### 13.2 Test Frameworks by Language

| Language | Framework | Characteristics |
|----------|----------|----------------|
| Python | pytest | Function-based, powerful fixtures, rich plugins |
| JavaScript/TypeScript | Jest | Made by Meta. Snapshot testing, built-in mocking |
| JavaScript/TypeScript | Vitest | Vite-based. Native ESM, Jest-compatible API |
| Java | JUnit 5 | Annotation-driven. Powerful parameterized tests |
| Go | testing (standard) | Self-contained in standard library. `go test` command |
| Rust | cargo test (standard) | `#[test]` attribute. Supports documentation tests |
| C# | xUnit.net | Standard .NET framework. `[Fact]`, `[Theory]` |
| Ruby | RSpec | BDD-style. `describe`, `it` blocks |

### 13.3 Test Support Tools

#### Mocking/Stubbing

| Tool | Language | Characteristics |
|------|----------|----------------|
| unittest.mock | Python | Standard library. `Mock`, `patch`, `MagicMock` |
| pytest-mock | Python | pytest wrapper for unittest.mock. `mocker` fixture |
| responses | Python | HTTP mocking for the requests library |
| Mockito | Java | Representative mock library for Java |
| testdouble.js | JavaScript | Test double library for JavaScript |

#### E2E / Browser Automation

| Tool | Supported Browsers | Characteristics |
|------|-------------------|----------------|
| Playwright | Chromium, Firefox, WebKit | Made by Microsoft. Multi-browser support, auto-waiting |
| Cypress | Chromium-based | Native JavaScript. Time-travel debugging |
| Selenium | All major browsers | Longest history. WebDriver protocol |

#### Coverage

| Tool | Language | Output Formats |
|------|----------|---------------|
| pytest-cov (coverage.py) | Python | HTML, XML, JSON, terminal |
| Istanbul (nyc) | JavaScript | HTML, lcov, text |
| JaCoCo | Java | HTML, XML, CSV |
| gcov / lcov | C/C++ | HTML, text |

#### Property-Based Testing

| Tool | Language | Characteristics |
|------|----------|----------------|
| Hypothesis | Python | Powerful shrinking, stateful test support |
| QuickCheck | Haskell | The originator of property-based testing |
| fast-check | JavaScript/TypeScript | For JS/TS. Inspired by Hypothesis |
| PropTest | Rust | Property-based testing for Rust |
| jqwik | Java | JUnit 5 integrated property-based testing |

#### Mutation Testing

| Tool | Language | Characteristics |
|------|----------|----------------|
| mutmut | Python | Simple and easy to use |
| cosmic-ray | Python | More mutant operators |
| Stryker | JS/TS, C# | Multi-language support. Rich reports |
| PIT (pitest) | Java | Representative mutation testing tool for Java |

### 13.4 Test Execution Optimization Techniques

As the test suite grows, execution time becomes a problem.
Here are the main optimization techniques.

```
Test Execution Optimization Strategy:

  +-----------------------------------------------------+
  |           Test Execution Speedup                      |
  +-----------------------------------------------------+
  |                                                     |
  |  1. Parallel execution                               |
  |     pytest-xdist: pytest -n auto                    |
  |     -> Auto-parallelizes based on CPU core count     |
  |                                                     |
  |  2. Change-detection-based execution                  |
  |     pytest --lf  (re-run only previously failed tests)|
  |     pytest --ff  (prioritize previously failed tests) |
  |     pytest-testmon (only tests related to changed code)|
  |                                                     |
  |  3. Layered execution                                |
  |     pytest -m "not slow"  (skip slow tests)          |
  |     pytest -m "unit"      (unit tests only)          |
  |     pytest -m "smoke"     (smoke tests only)         |
  |                                                     |
  |  4. Fixture optimization                              |
  |     scope="session" -> Run once for entire session    |
  |     scope="module"  -> Once per module               |
  |     scope="class"   -> Once per class                |
  |     scope="function"-> Once per test function (default)|
  |                                                     |
  |  5. Reducing unnecessary I/O                          |
  |     Use in-memory DB (SQLite :memory:)               |
  |     StringIO / BytesIO instead of filesystem         |
  |     Mock HTTP communication                          |
  +-----------------------------------------------------+
```

```python
# Example of layered execution with pytest markers

import pytest

# Assign markers to tests
@pytest.mark.unit
def test_fast_unit_test():
    assert 1 + 1 == 2

@pytest.mark.integration
def test_with_db_connection():
    # Test that connects to DB
    pass

@pytest.mark.slow
def test_slow_running_test():
    # Test that takes tens of seconds
    pass

@pytest.mark.e2e
def test_browser_operation():
    # Test that operates a browser with Playwright
    pass

# pyproject.toml configuration:
# [tool.pytest.ini_options]
# markers = [
#     "unit: Unit tests",
#     "integration: Integration tests",
#     "slow: Slow-running tests",
#     "e2e: E2E tests",
# ]

# Execution examples:
# pytest -m unit           -> Unit tests only
# pytest -m "not slow"     -> All except slow tests
# pytest -m "unit or integration"  -> Unit + Integration
```

### 13.5 Test Data Management

Test data management is directly tied to test reliability and maintainability.

#### Factory Pattern

```python
# Factory pattern for test data

from dataclasses import dataclass, field
import uuid


@dataclass
class UserFactory:
    """Factory for generating test user data."""

    name: str = "Test User"
    email: str = field(default_factory=lambda: f"test-{uuid.uuid4().hex[:8]}@example.com")
    age: int = 30
    is_active: bool = True

    def build(self) -> dict:
        """Return user data in dictionary format."""
        return {
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "is_active": self.is_active,
        }

    @classmethod
    def admin(cls) -> "UserFactory":
        """Admin user preset."""
        return cls(name="Admin", age=40)

    @classmethod
    def child(cls) -> "UserFactory":
        """Child user preset."""
        return cls(name="Test Child", age=10)


# Usage examples
def test_register_with_default_user():
    user_data = UserFactory().build()
    result = register_user(**user_data)
    assert result.name == "Test User"

def test_register_as_admin():
    user_data = UserFactory.admin().build()
    result = register_user(**user_data)
    assert result.name == "Admin"

def test_register_with_custom_data():
    user_data = UserFactory(name="Custom", age=25).build()
    result = register_user(**user_data)
    assert result.name == "Custom"
```

#### Builder Pattern

```python
# Builder pattern for test data

class OrderBuilder:
    """Generates test order data using the builder pattern."""

    def __init__(self):
        self._customer_id = "C001"
        self._items = []
        self._discount = 0
        self._shipping_address = "Tokyo, Chiyoda-ku"

    def with_customer(self, customer_id: str) -> "OrderBuilder":
        self._customer_id = customer_id
        return self

    def with_item(self, name: str, price: int, quantity: int = 1) -> "OrderBuilder":
        self._items.append({"name": name, "price": price, "quantity": quantity})
        return self

    def with_discount(self, discount: int) -> "OrderBuilder":
        self._discount = discount
        return self

    def with_shipping_address(self, address: str) -> "OrderBuilder":
        self._shipping_address = address
        return self

    def build(self) -> dict:
        return {
            "customer_id": self._customer_id,
            "items": self._items,
            "discount": self._discount,
            "shipping_address": self._shipping_address,
        }


# Usage example
def test_order_total_with_multiple_items():
    order = (
        OrderBuilder()
        .with_item("Python Introduction", 3000, quantity=2)
        .with_item("Go in Practice", 3500)
        .with_discount(500)
        .build()
    )
    total = calculate_order_total(order)
    assert total == 9000  # (3000*2 + 3500) - 500
```

---

## 14. The Relationship Between Testing and Design

### 14.1 Testability and Design Quality

Testability is an excellent indicator of design quality.
Code that is hard to test almost certainly has design problems.

```
Relationship Between Testability and Design:

  Characteristics of hard-to-test code      Corresponding design problems
  ------------------------------------      ----------------------------
  - Creates dependencies with new directly  -> Tight Coupling
  - Depends on global variables             -> Hidden dependencies
  - Many static methods                     -> Cannot be replaced with test doubles
  - Methods with hundreds of lines          -> SRP violation
  - Side effects in constructors            -> Mixed creation and usage
  - Direct access to environment variables  -> Implicit dependency on configuration

  Characteristics of easy-to-test code      Corresponding design principles
  ------------------------------------      ------------------------------
  - Dependencies injected via constructor   -> Dependency Inversion Principle (DIP)
  - Depends on interfaces                   -> Open/Closed Principle (OCP)
  - Methods are short with single purpose   -> Single Responsibility Principle (SRP)
  - Pure functions with few side effects    -> Functional Programming
  - Configuration received via arguments    -> Explicit dependencies
```

### 14.2 Dependency Injection and Testability

```python
# ===== Code Example 6: Improving Testability Through Dependency Injection =====

# Bad example: Creating dependencies directly
class NotificationService_Bad:
    def notify(self, user_id: str, message: str) -> bool:
        # Emails are sent even during testing
        import smtplib
        server = smtplib.SMTP("smtp.example.com")
        server.sendmail("noreply@example.com", user_id, message)
        return True


# Good example: Dependency injection
from typing import Protocol

class EmailSender(Protocol):
    def send(self, to: str, subject: str, body: str) -> bool: ...

class NotificationService_Good:
    def __init__(self, email_sender: EmailSender):
        self._email_sender = email_sender

    def notify(self, user_id: str, message: str) -> bool:
        return self._email_sender.send(
            to=user_id,
            subject="Notification",
            body=message,
        )


# Test code
class FakeEmailSender:
    """Fake implementation for testing."""
    def __init__(self):
        self.sent_emails = []

    def send(self, to: str, subject: str, body: str) -> bool:
        self.sent_emails.append({"to": to, "subject": subject, "body": body})
        return True


def test_notification_is_sent():
    fake_sender = FakeEmailSender()
    service = NotificationService_Good(fake_sender)

    result = service.notify("user@example.com", "Test message")

    assert result is True
    assert len(fake_sender.sent_emails) == 1
    assert fake_sender.sent_emails[0]["to"] == "user@example.com"
```

### 14.3 Hexagonal Architecture and Testing

Hexagonal Architecture (Ports & Adapters Architecture)
is a design pattern that excels in testability.

```
Hexagonal Architecture and Test Strategy:

                    +----------------------+
                    |    Test Strategy      |
                    +----------------------+

  +-----------+                               +-----------+
  | Adapter   |    +--------------------+     | Adapter   |
  | (Input)   |--->|   Port (Input)     |     | (Output)  |
  | HTTP API  |    |                    |     | DB        |
  | CLI       |    |  +--------------+  |     | Email     |
  | Message   |    |  | Domain       |  |---> | External  |
  |           |    |  | Logic        |  |     | API       |
  |           |    |  +--------------+  |     |           |
  +-----------+    |   Port (Output)    |     +-----------+
                   +--------------------+

  Test Strategy:
  ==================================================

  Unit Tests -> Domain Logic
    - No external dependencies. Verifies pure business rules
    - Ports are replaced with mocks/stubs

  Integration Tests -> Adapters
    - Verify connections with actual DB and HTTP servers
    - Whether port implementations work correctly

  E2E Tests -> Input Adapter -> Domain -> Output Adapter
    - Penetrates all layers. Only for key flows
```

---

## 15. Exercises

### Exercise 1 (Beginner): Implementing Unit Tests

Create unit tests using pytest for the following `StringCalculator` class.

```python
class StringCalculator:
    """Calculates numbers in string format."""

    def add(self, numbers: str) -> int:
        """Takes a comma-separated number string and returns the sum.

        Rules:
        - Returns 0 for empty string
        - Returns the number if there is only one
        - Returns the sum of comma-separated numbers
        - Newlines are also treated as delimiters ("1\n2,3" -> 6)
        - Raises ValueError if negative numbers are included
          (error message includes the negative numbers)
        - Numbers greater than 1000 are ignored ("2,1001" -> 2)
        """
        if not numbers:
            return 0

        delimiters = [",", "\n"]
        for d in delimiters:
            numbers = numbers.replace(d, ",")

        values = [int(x) for x in numbers.split(",")]

        negatives = [v for v in values if v < 0]
        if negatives:
            raise ValueError(f"Negative numbers are not allowed: {negatives}")

        return sum(v for v in values if v <= 1000)
```

**Requirements:**
- At least 5 normal case tests
- At least 2 error case tests
- At least 3 boundary value tests
- Use `@pytest.mark.parametrize` in at least one place

### Exercise 2 (Intermediate): Implement a Stack Using TDD

Implement a `Stack` class that satisfies the following specification using TDD.

**Specification:**
1. `push(item)` -- Add an item to the top of the stack
2. `pop()` -- Remove and return the top item. Raise `IndexError` if empty
3. `peek()` -- Return the top item without removing it. Raise `IndexError` if empty
4. `is_empty()` -- Return `True` if the stack is empty
5. `size()` -- Return the number of items in the stack
6. `max_size` -- Specified in constructor. Raise `OverflowError` on `push` when size is exceeded

**Procedure:**
1. First write a test for `is_empty` and implement it (Red -> Green)
2. Write tests for `push` and `size` and implement them
3. Write tests for `pop` (normal and error cases) and implement
4. Write tests for `peek` and implement
5. Write tests for `max_size` and implement
6. Refactor while all tests pass

### Exercise 3 (Advanced): Designing Property-Based Tests

Design and implement property-based tests using Hypothesis for the following functions.

```python
import json
from typing import Any


def json_round_trip(data: dict) -> dict:
    """JSON serialize -> deserialize round trip."""
    return json.loads(json.dumps(data))


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dictionary.
    Example: {"a": {"b": 1}} -> {"a.b": 1}
    """
    result = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, new_key))
        else:
            result[new_key] = value
    return result


def compact(lst: list) -> list:
    """Remove None and empty strings from a list."""
    return [x for x in lst if x is not None and x != ""]
```

**Requirements:**
- Define at least 2 properties for each function
- Use advanced strategies such as `st.dictionaries`, `st.recursive`
- Use the `@example` decorator to specify edge cases

---

## 16. FAQ (Frequently Asked Questions)

### Q1: What percentage should code coverage target?

Generally, **80% or higher** is the recommended guideline. However, pursuing coverage numbers
alone is dangerous. The important points are the following three:

1. **Business-critical areas** (payment processing, authentication, data conversion, etc.) should maintain high coverage
2. **Regularly review low-coverage areas** and determine whether tests are truly unnecessary
3. **Prioritize test quality over high coverage** (appropriate assertions, edge case coverage)

Even 100% coverage does not mean zero bugs (as mentioned earlier).
Coverage should be used as "an indicator for discovering untested areas,"
and other techniques (such as mutation testing) should be used in combination for test quality evaluation.

### Q2: Tests are slow and CI feedback is delayed. How should this be improved?

Apply the following approaches incrementally.

**Short-term measures:**
1. **Parallel test execution**: Use `pytest-xdist` with `pytest -n auto`
2. **Identify slow tests**: Use `pytest --durations=20` to check the top 20 slowest tests
3. **Reduce unnecessary E2E tests**: Migrate those that can be replaced by unit tests
4. **Optimize fixtures**: Eliminate duplicate setups

**Medium-term measures:**
1. **Layered test execution**: Run only fast unit tests on PRs, run all tests after merge
2. **Change detection**: Only run tests related to changed files
3. **Optimize test DB**: Use SQLite (for tests) or in-memory DB instead of PostgreSQL

**Long-term measures:**
1. **Reconstruct the test pyramid**: If there are too many E2E tests, migrate to lower layers
2. **Improve test infrastructure**: Caching, parallelization, distributed execution
3. **Review test architecture**: Isolate external dependencies with Hexagonal Architecture

### Q3: How do you add tests to legacy code?

We recommend the approach from Michael Feathers' "Working Effectively with Legacy Code."

**Step 1: Identify the change point**
- Identify the area that needs change and understand the dependency relationships of related code

**Step 2: Establish a test harness**
- Perform minimal refactoring to make the change point testable
- Techniques for breaking dependencies:
  - Extract Interface
  - Parameterize Constructor
  - Wrap Method

**Step 3: Write characterization tests**
- Write tests that record current behavior (regardless of whether it matches the spec)
- This allows detection of unintended changes during refactoring

**Step 4: Make the change**
- Make the necessary changes under the safety net of tests

**Step 5: Improve the tests**
- Gradually replace characterization tests with tests based on the correct specification

### Q4: How much should mocks be used?

The amount of mock usage varies depending on "test type" and "dependencies of the test target."

**Should be mocked:**
- HTTP requests to external APIs
- Non-deterministic elements such as time and random numbers
- Sending operations (email sending, push notifications, etc.)
- External services that don't exist in the test environment

**Should not be mocked:**
- Methods of the test target itself (defeats the purpose of the test)
- Simple value objects
- Basic standard library functions

**Judgment criteria:**
- Ask yourself "If I remove this mock, does the test still have meaning?"
- If mock setup code is longer than production code, reconsider the design

### Q5: Where is the boundary between E2E tests and integration tests?

There is no clear boundary as it varies by organization and project, but general guidelines are as follows.

| Criteria | Integration Tests | E2E Tests |
|----------|------------------|-----------|
| UI involvement | None (up to API layer) | Yes (includes browser operations) |
| Test target | Interaction of 2-3 components | Entire system flow |
| Data | Prepared directly in test DB | Prepared via UI input |
| Execution speed | Seconds to minutes | Minutes to tens of minutes |
| Fragility | Moderate | High |

In practice, "whether or not UI is involved" is often used as the boundary.
Integration tests against APIs are sufficiently stable,
and the recognition that E2E tests involving browser operations are costly has become widespread.

---

## 17. Practical Guidelines for Test Strategy

### 15.1 Order of Writing Tests

Here are guidelines for the order of test creation when developing new features.

```
Recommended Order for Test Creation:

  1. Unit tests for domain logic
     +-- Business rules, validation, calculations
         Most important and easiest to test

  2. Unit tests for the service layer
     +-- Combinations of domain logic, mocking external dependencies

  3. Integration tests for repositories/data access
     +-- Accuracy of DB interactions

  4. Integration tests for API endpoints
     +-- Request/response formats, status codes

  5. E2E tests for critical user flows
     +-- Only business-critical scenarios
```

### 15.2 Test Maintenance

Test code requires maintenance just like production code.

**Things to do regularly:**
1. **Improve slow tests** -- Periodically measure with `pytest --durations`
2. **Fix flaky tests** -- Do not leave unstable tests unaddressed
3. **Remove unnecessary tests** -- Tests made unnecessary by spec changes, duplicate tests
4. **Organize test utilities** -- Optimize factories, builders, shared fixtures
5. **Review coverage reports** -- Identify untested areas

### 15.3 Quality Standards for Test Code

Apply coding standards to test code as well.

| Standard | Content |
|----------|---------|
| Readability | The specification can be understood just by reading the test |
| Naming | Test names clearly indicate what is being verified |
| Structure | Follows the AAA pattern with clearly separated sections |
| DRY | Setup is shared via fixtures (but not at the expense of test readability) |
| Independence | No dependencies between tests |
| Speed | Unit tests complete in milliseconds |
| Stability | Not flaky |

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It is particularly important during code reviews and architecture design.

---

## 18. Summary

### Overview of Test Classification

| Test Level | Target | Speed | Cost | Recommended Ratio |
|-----------|--------|-------|------|-------------------|
| Unit Tests | Functions/Classes | ms | Low | 70% |
| Integration Tests | Component interaction | sec to min | Medium | 20% |
| E2E Tests | Entire system | min to tens | High | 10% |
| Acceptance Tests | Business requirements | min to tens | High | Per sprint |

### Comparison of Development Methodologies

| Methodology | Core | Tools | Application |
|------------|------|-------|-------------|
| TDD | Test-first drives design | pytest, JUnit | Logic implementation |
| BDD | Business scenarios as automated tests | Cucumber, Behave | Requirements agreement |
| Property-Based Testing | Verify invariants with random inputs | Hypothesis, QuickCheck | General-purpose logic |

### When to Use Each Test Technique

| Technique | Use Case | Effect |
|-----------|---------|--------|
| Equivalence Partitioning | Grouping input domains | Reducing test case count |
| Boundary Value Analysis | Detecting off-by-one errors | Early detection of boundary bugs |
| Decision Table | Covering compound conditions | Preventing condition omissions |
| Pairwise Testing | Reducing multi-factor combinations | Maximizing test efficiency |

### Test Design Principles

```
Test Design Checklist:

  [ ] FIRST principles (Fast, Independent, Repeatable, Self-validating, Timely)
  [ ] AAA pattern (Arrange, Act, Assert)
  [ ] One concept per test
  [ ] Test names express the specification
  [ ] Test pyramid is balanced
  [ ] Coverage is 80% or higher
  [ ] No flaky tests
  [ ] Automated execution in CI
```

---

## Recommended Next Reading


---

## References

1. Beck, K. *Test Driven Development: By Example*. Addison-Wesley, 2002.
   -- The definitive work on TDD. Covers the Red-Green-Refactor basic cycle and practical examples.
2. Feathers, M. *Working Effectively with Legacy Code*. Prentice Hall, 2004.
   -- Practical techniques for adding tests to code without tests (legacy code).
3. Freeman, S., Pryce, N. *Growing Object-Oriented Software, Guided by Tests*. Addison-Wesley, 2009.
   -- A seminal work on test-driven design using mocks. Representative of the London School of TDD.
4. Meszaros, G. *xUnit Test Patterns: Refactoring Test Code*. Addison-Wesley, 2007.
   -- Systematized the classification of test doubles (Dummy, Stub, Spy, Mock, Fake).
5. Cohn, M. *Succeeding with Agile*. Addison-Wesley, 2009.
   -- The origin of the test pyramid. Explains testing strategy in agile development.
6. Claessen, K., Hughes, J. "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs." ICFP, 2000.
   -- The original paper on property-based testing. Hypothesis ported this philosophy to Python.
7. MacLeod, D. *Hypothesis documentation*. https://hypothesis.readthedocs.io/
   -- The official documentation for Hypothesis, the property-based testing framework for Python.

---

> Testing does not build quality in. Testing makes quality visible.
> Quality comes from testable design.
