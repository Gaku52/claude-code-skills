# Clean Code

> Code is read 10 times more than it is written. Readable code is the shortest path to correct code.

## What You Will Learn in This Chapter

- [ ] Master the principles of good naming
- [ ] Understand the principles of function design
- [ ] Recognize Code Smells
- [ ] Understand and practice the SOLID principles
- [ ] Correctly apply DRY / KISS / YAGNI
- [ ] Learn how to write comments and how to identify unnecessary comments
- [ ] Master best practices for error handling
- [ ] Be able to write highly testable code
- [ ] Master refactoring techniques and decision criteria
- [ ] Learn how to improve quality through code reviews


## Prerequisites

Having the following knowledge before reading this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the content of [Design Patterns](./02-design-patterns.md)

---

## 1. Naming

### 1.1 Fundamental Naming Principles

```python
# Bad naming
d = 30          # Days of what?
lst = []        # List of what?
def proc(x):    # What does it process?
    pass

# Good naming
trial_period_days = 30
active_users = []
def calculate_monthly_revenue(transactions):
    pass

# Naming Principles:
# 1. Be intention-revealing: is_active, has_permission, should_retry
# 2. Be pronounceable: Bad: genymdhms -> Good: generation_timestamp
# 3. Be searchable: Bad: 7 -> Good: MAX_RETRY_COUNT = 7
# 4. Proportional to scope: Short for loop variables (i), long for globals
# 5. Consistency: Don't mix get/fetch/retrieve
```

### 1.2 Naming Pattern Collection

```python
# --- Boolean variable naming ---
# Use is_, has_, can_, should_, will_ prefixes
is_active = True
has_permission = False
can_edit = True
should_retry = False
will_expire = True

# Bad boolean names
flag = True        # Flag for what?
status = False     # Shouldn't status have a value?
check = True       # Result of check? Or should check?

# Good boolean names
is_email_verified = True
has_admin_role = False
can_access_dashboard = True
should_send_notification = False

# --- Collection type naming ---
# Use plurals
users = [user1, user2, user3]
email_addresses = ["a@example.com", "b@example.com"]
order_items = []

# Bad: Singular collection names
user_list = []     # "list" is type information and redundant
email_array = []   # "array" is similarly redundant

# Good: Map/Dict naming uses "key_to_value" or "value_by_key"
user_by_id = {"u001": user1, "u002": user2}
price_by_product = {"apple": 100, "banana": 200}
email_to_user = {"a@example.com": user1}

# --- Function name patterns ---
# verb + noun
def calculate_total_price(items):
    """Calculate the total price"""
    pass

def validate_email_address(email):
    """Validate an email address"""
    pass

def send_welcome_email(user):
    """Send a welcome email"""
    pass

def parse_csv_file(file_path):
    """Parse a CSV file"""
    pass

# --- Factory method naming ---
def create_user(name, email):
    """Create a new user"""
    pass

def from_json(json_string):
    """Create from JSON"""
    pass

def build_query(params):
    """Build a query"""
    pass
```

### 1.3 Naming Anti-Patterns

```python
# --- 1. Hungarian Notation (unnecessary in modern code) ---
# Bad: Including type as a prefix
str_name = "Alice"
int_age = 30
lst_users = []
dict_config = {}

# Good: Express types with type hints
name: str = "Alice"
age: int = 30
users: list[User] = []
config: dict[str, Any] = {}

# --- 2. Excessive abbreviation ---
# Bad: Cryptic abbreviations
def calc_avg_rev_per_usr(txns):
    pass

# Good: Use complete words
def calculate_average_revenue_per_user(transactions):
    pass

# However, common abbreviations are OK
# id, url, http, api, db, io, cpu, os, ui, html, css
user_id = "u001"
api_url = "https://api.example.com"
db_connection = get_connection()

# --- 3. Double negatives ---
# Bad: Hard to understand
if not is_not_active:
    pass

if not disable_feature:
    pass

# Good: Use positive form
if is_active:
    pass

if enable_feature:
    pass

# --- 4. Overly generic names ---
# Bad: Not specific enough
data = fetch_data()
result = process(data)
temp = calculate(result)
info = get_info()
manager = create_manager()

# Good: Specific names
user_profiles = fetch_user_profiles()
monthly_revenue = calculate_revenue(user_profiles)
formatted_report = format_revenue_report(monthly_revenue)
server_health_info = get_server_health()
connection_pool_manager = create_connection_pool()
```

### 1.4 Naming Consistency Rules

```python
# Unify naming conventions within a project

# --- Use the same verb for the same concept ---
# Bad: Mixed
def get_user(user_id): pass
def fetch_order(order_id): pass
def retrieve_product(product_id): pass
def obtain_payment(payment_id): pass

# Good: Unified
def get_user(user_id): pass
def get_order(order_id): pass
def get_product(product_id): pass
def get_payment(payment_id): pass

# --- Use symmetrical names ---
# open / close
# start / stop
# begin / end
# insert / delete
# add / remove
# create / destroy
# lock / unlock
# source / target
# first / last
# min / max
# next / previous
# up / down
# show / hide
# enable / disable
# increment / decrement

# --- Example of sharing naming conventions with the team ---
"""
Naming Convention Document:
1. Class names: PascalCase (UserProfile, OrderItem)
2. Functions/variables: snake_case (calculate_total, user_name)
3. Constants: UPPER_SNAKE_CASE (MAX_RETRY_COUNT, API_BASE_URL)
4. Private: Leading underscore (_internal_method)
5. Booleans: is_, has_, can_, should_ prefix
6. Collections: Plural (users, items, orders)
7. Dictionaries: value_by_key format (user_by_id)
8. Event handlers: on_ prefix (on_click, on_submit)
9. Callbacks: _callback suffix (success_callback)
10. Tests: test_ prefix (test_create_user)
"""
```

---

## 2. Function Design

### 2.1 Fundamental Principles of Function Design

```python
# Principle: One function, one responsibility. Keep it short. Few arguments.

# Bad: Long function (multiple responsibilities)
def process_order(order):
    # Validation (20 lines)...
    # Inventory check (15 lines)...
    # Payment processing (25 lines)...
    # Email sending (10 lines)...
    # Logging (5 lines)...
    pass  # 75-line giant function

# Good: Split functions
def process_order(order):
    validate_order(order)
    check_inventory(order.items)
    charge_payment(order.payment)
    send_confirmation_email(order.customer)
    log_order(order)

# Argument principles:
# 0 arguments (niladic): Ideal
# 1 argument (monadic): Good
# 2 arguments (dyadic): Acceptable
# 3+ arguments: Consider grouping into an object
```

### 2.2 Abstraction Levels in Functions

```python
# Maintain the same abstraction level within a single function

# Bad: Mixed abstraction levels
def generate_report(data):
    # High level: Data validation
    if not data:
        raise ValueError("Data is empty")

    # Low level: Direct HTML manipulation
    html = "<html><head><title>Report</title></head><body>"
    html += "<table>"
    for row in data:
        html += "<tr>"
        for cell in row:
            html += f"<td>{cell}</td>"
        html += "</tr>"
    html += "</table></body></html>"

    # High level: Send email
    send_email("admin@example.com", "Report", html)
    return html

# Good: Same abstraction level
def generate_report(data):
    validate_report_data(data)
    html = build_report_html(data)
    distribute_report(html)
    return html

def validate_report_data(data):
    if not data:
        raise ValueError("Data is empty")
    if not all(isinstance(row, list) for row in data):
        raise TypeError("Data must be a list of lists")

def build_report_html(data):
    header = create_html_header("Report")
    body = create_html_table(data)
    return wrap_in_html(header, body)

def distribute_report(html):
    send_email("admin@example.com", "Report", html)
```

### 2.3 Best Practices for Argument Design

```python
from dataclasses import dataclass
from typing import Optional

# --- Functions with too many arguments ---
# Bad: Too many arguments
def create_user(name, email, age, address, phone,
                role, department, manager_id,
                start_date, salary):
    pass

# Good: Use a parameter object
@dataclass
class UserCreationParams:
    name: str
    email: str
    age: int
    address: str
    phone: str
    role: str = "member"
    department: str = "general"
    manager_id: Optional[str] = None
    start_date: Optional[str] = None
    salary: Optional[float] = None

def create_user(params: UserCreationParams):
    # Access via params.name, params.email, etc.
    pass

# --- Avoid flag arguments ---
# Bad: Branching on a boolean argument
def get_users(include_inactive: bool):
    if include_inactive:
        return get_all_users()
    else:
        return get_active_users()

# Good: Split the function
def get_active_users():
    return [u for u in all_users if u.is_active]

def get_all_users():
    return all_users

# --- Avoid output arguments ---
# Bad: Modifying an argument
def add_to_list(items, new_item):
    items.append(new_item)  # Side effect

# Good: Return a new value
def add_to_list(items, new_item):
    return [*items, new_item]

# --- Leverage default arguments ---
def connect_to_database(
    host: str = "localhost",
    port: int = 5432,
    database: str = "app",
    timeout_seconds: int = 30,
    max_retries: int = 3
):
    """Only specify the arguments you need"""
    pass

# Usage: When defaults are sufficient
connect_to_database()

# Override only some
connect_to_database(host="production-db.example.com", timeout_seconds=60)
```

### 2.4 Pure Functions and Side Effects

```python
# --- Pure function: Same input always produces same output, no side effects ---
# Good: Pure function examples
def calculate_tax(price: float, tax_rate: float) -> float:
    return price * tax_rate

def format_full_name(first_name: str, last_name: str) -> str:
    return f"{last_name} {first_name}"

def filter_active_users(users: list) -> list:
    return [u for u in users if u.is_active]

# --- Functions with side effects: Make it clear in the name ---
# Good: Side effect is apparent from the name
def save_user_to_database(user):
    """Save to database (side effect)"""
    db.users.insert(user.to_dict())

def send_notification_email(user, message):
    """Send an email (side effect)"""
    email_service.send(user.email, message)

def log_access(user_id, resource):
    """Record an access log (side effect)"""
    logger.info(f"User {user_id} accessed {resource}")

# --- Command Query Separation (CQS) ---
# Bad: Command and query mixed
class UserService:
    def get_and_update_last_login(self, user_id):
        """Get user while also updating last login timestamp"""
        user = self.db.find(user_id)
        user.last_login = datetime.now()
        self.db.save(user)
        return user  # Query (get) and command (update) are mixed

# Good: Separate command and query
class UserService:
    def get_user(self, user_id):
        """Get a user (query)"""
        return self.db.find(user_id)

    def update_last_login(self, user_id):
        """Update last login timestamp (command)"""
        user = self.db.find(user_id)
        user.last_login = datetime.now()
        self.db.save(user)
```

### 2.5 Function Length and Complexity

```python
# --- Keep cyclomatic complexity low ---

# Bad: High complexity (too many branches)
def calculate_discount(user, order):
    discount = 0
    if user.is_premium:
        if order.total > 10000:
            if user.years_of_membership > 5:
                discount = 0.20
            elif user.years_of_membership > 2:
                discount = 0.15
            else:
                discount = 0.10
        elif order.total > 5000:
            if user.years_of_membership > 5:
                discount = 0.15
            else:
                discount = 0.10
        else:
            discount = 0.05
    else:
        if order.total > 10000:
            discount = 0.05
        elif order.total > 5000:
            discount = 0.03
    return discount

# Good: Table-driven to reduce complexity
DISCOUNT_TABLE = {
    # (is_premium, min_total, min_years): discount_rate
    (True,  10000, 5): 0.20,
    (True,  10000, 2): 0.15,
    (True,  10000, 0): 0.10,
    (True,   5000, 5): 0.15,
    (True,   5000, 0): 0.10,
    (True,      0, 0): 0.05,
    (False, 10000, 0): 0.05,
    (False,  5000, 0): 0.03,
}

def calculate_discount(user, order):
    for (premium, min_total, min_years), rate in DISCOUNT_TABLE.items():
        if (user.is_premium == premium and
            order.total >= min_total and
            user.years_of_membership >= min_years):
            return rate
    return 0.0

# --- Eliminate conditional branching with polymorphism ---
# Bad: Branching by type
def calculate_shipping(order):
    if order.shipping_type == "standard":
        return order.weight * 10
    elif order.shipping_type == "express":
        return order.weight * 20 + 500
    elif order.shipping_type == "overnight":
        return order.weight * 30 + 1000
    elif order.shipping_type == "international":
        return order.weight * 50 + 2000
    else:
        raise ValueError(f"Unknown shipping type: {order.shipping_type}")

# Good: Strategy pattern
from abc import ABC, abstractmethod

class ShippingStrategy(ABC):
    @abstractmethod
    def calculate(self, weight: float) -> float:
        pass

class StandardShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 10

class ExpressShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 20 + 500

class OvernightShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 30 + 1000

class InternationalShipping(ShippingStrategy):
    def calculate(self, weight: float) -> float:
        return weight * 50 + 2000

SHIPPING_STRATEGIES = {
    "standard": StandardShipping(),
    "express": ExpressShipping(),
    "overnight": OvernightShipping(),
    "international": InternationalShipping(),
}

def calculate_shipping(order):
    strategy = SHIPPING_STRATEGIES.get(order.shipping_type)
    if not strategy:
        raise ValueError(f"Unknown shipping type: {order.shipping_type}")
    return strategy.calculate(order.weight)
```

---

## 3. Code Smells

### 3.1 Common Code Smells List

```
Code Smells (Signs That Refactoring Is Needed):

  +--------------------+-------------------------------+
  | Smell              | Remedy                        |
  +--------------------+-------------------------------+
  | Long Method        | Extract Method                |
  | Large Class        | Split Class                   |
  | Duplicated Code    | Extract into common function  |
  | Long Parameter List| Parameter Object              |
  | Flag Argument      | Split the function            |
  | Comment Needed     | Make code self-documenting     |
  | Deep Nesting       | Early return, guard clauses   |
  | Magic Number       | Named constants               |
  | Data Clumps        | Extract into data class       |
  | Feature Envy       | Move method to proper class   |
  | Divergent Change   | Separate responsibilities     |
  | Shotgun Surgery    | Consolidate related changes   |
  | Middle Man         | Remove delegation             |
  | Inappropriate      | Strengthen encapsulation      |
  |   Intimacy         |                               |
  | Lazy Class         | Merge classes                 |
  | Speculative        | Remove unnecessary            |
  |   Generality       |   abstraction                 |
  +--------------------+-------------------------------+
```

### 3.2 Early Return and Guard Clauses

```python
# Bad: Deep nesting
def process(user):
    if user:
        if user.is_active:
            if user.has_permission:
                if user.email_verified:
                    # Finally the actual logic
                    result = do_complex_calculation(user)
                    save_result(result)
                    notify_user(user, result)
                    return result
                else:
                    raise ValueError("Email not verified")
            else:
                raise PermissionError("No permission")
        else:
            raise ValueError("User is not active")
    else:
        raise ValueError("User is None")

# Good: Guard clauses with early return
def process(user):
    if not user:
        raise ValueError("User is None")
    if not user.is_active:
        raise ValueError("User is not active")
    if not user.has_permission:
        raise PermissionError("No permission")
    if not user.email_verified:
        raise ValueError("Email not verified")

    # The actual logic can be written flat
    result = do_complex_calculation(user)
    save_result(result)
    notify_user(user, result)
    return result
```

### 3.3 Eliminating Duplicated Code

```python
# Bad: Code duplication
class UserReport:
    def generate_csv(self, users):
        # Validation
        if not users:
            raise ValueError("No users provided")
        if len(users) > 10000:
            raise ValueError("Too many users")

        # Filtering
        active_users = [u for u in users if u.is_active]

        # CSV generation
        lines = ["name,email,role"]
        for user in active_users:
            lines.append(f"{user.name},{user.email},{user.role}")
        return "\n".join(lines)

    def generate_json(self, users):
        # Validation (duplicate!)
        if not users:
            raise ValueError("No users provided")
        if len(users) > 10000:
            raise ValueError("Too many users")

        # Filtering (duplicate!)
        active_users = [u for u in users if u.is_active]

        # JSON generation
        return json.dumps([
            {"name": u.name, "email": u.email, "role": u.role}
            for u in active_users
        ])

# Good: Extract common parts (Template Method pattern)
class UserReport:
    def _validate_and_filter(self, users):
        """Common validation and filtering"""
        if not users:
            raise ValueError("No users provided")
        if len(users) > 10000:
            raise ValueError("Too many users")
        return [u for u in users if u.is_active]

    def generate_csv(self, users):
        active_users = self._validate_and_filter(users)
        lines = ["name,email,role"]
        for user in active_users:
            lines.append(f"{user.name},{user.email},{user.role}")
        return "\n".join(lines)

    def generate_json(self, users):
        active_users = self._validate_and_filter(users)
        return json.dumps([
            {"name": u.name, "email": u.email, "role": u.role}
            for u in active_users
        ])
```

### 3.4 Eliminating Magic Numbers

```python
# Bad: Full of magic numbers
def check_password(password):
    if len(password) < 8:
        return False
    if len(password) > 128:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True

def calculate_shipping(weight):
    if weight <= 1.0:
        return 500
    elif weight <= 5.0:
        return 500 + (weight - 1.0) * 200
    else:
        return 500 + 800 + (weight - 5.0) * 150

# Good: Named constants
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128

def check_password(password):
    if len(password) < MIN_PASSWORD_LENGTH:
        return False
    if len(password) > MAX_PASSWORD_LENGTH:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True

BASE_SHIPPING_FEE = 500
LIGHT_WEIGHT_LIMIT = 1.0    # kg
MEDIUM_WEIGHT_LIMIT = 5.0   # kg
LIGHT_RATE_PER_KG = 200     # yen/kg
MEDIUM_RATE_PER_KG = 200    # yen/kg
HEAVY_RATE_PER_KG = 150     # yen/kg

def calculate_shipping(weight):
    if weight <= LIGHT_WEIGHT_LIMIT:
        return BASE_SHIPPING_FEE
    elif weight <= MEDIUM_WEIGHT_LIMIT:
        extra = (weight - LIGHT_WEIGHT_LIMIT) * LIGHT_RATE_PER_KG
        return BASE_SHIPPING_FEE + extra
    else:
        medium_fee = (MEDIUM_WEIGHT_LIMIT - LIGHT_WEIGHT_LIMIT) * MEDIUM_RATE_PER_KG
        heavy_fee = (weight - MEDIUM_WEIGHT_LIMIT) * HEAVY_RATE_PER_KG
        return BASE_SHIPPING_FEE + medium_fee + heavy_fee
```

### 3.5 Data Clumps

```python
# Bad: The same group of data appears repeatedly
def create_invoice(
    customer_name, customer_email, customer_phone,
    customer_address, customer_city, customer_zip,
    items, tax_rate
):
    pass

def send_invoice(
    customer_name, customer_email, customer_phone,
    customer_address, customer_city, customer_zip,
    invoice_id
):
    pass

def update_customer(
    customer_name, customer_email, customer_phone,
    customer_address, customer_city, customer_zip
):
    pass

# Good: Group into data classes
@dataclass
class Address:
    street: str
    city: str
    zip_code: str

@dataclass
class Customer:
    name: str
    email: str
    phone: str
    address: Address

def create_invoice(customer: Customer, items: list, tax_rate: float):
    pass

def send_invoice(customer: Customer, invoice_id: str):
    pass

def update_customer(customer: Customer):
    pass
```

---

## 4. SOLID Principles

### 4.1 Single Responsibility Principle (SRP)

```python
# Bad: A class with multiple responsibilities
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def save_to_database(self):
        """Save to database (persistence responsibility)"""
        db.execute("INSERT INTO users VALUES (?, ?)", (self.name, self.email))

    def send_welcome_email(self):
        """Send email (notification responsibility)"""
        smtp.send(self.email, "Welcome!", f"Hello {self.name}")

    def generate_report(self):
        """Generate report (presentation responsibility)"""
        return f"User Report: {self.name} ({self.email})"

    def validate(self):
        """Validation (verification responsibility)"""
        if "@" not in self.email:
            raise ValueError("Invalid email")

# Good: Split classes by responsibility
@dataclass
class User:
    """Holds only user data"""
    name: str
    email: str

class UserValidator:
    """Handles user validation"""
    @staticmethod
    def validate(user: User):
        if not user.name:
            raise ValueError("Name is required")
        if "@" not in user.email:
            raise ValueError("Invalid email")

class UserRepository:
    """Handles user persistence"""
    def __init__(self, db_connection):
        self.db = db_connection

    def save(self, user: User):
        self.db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (user.name, user.email)
        )

    def find_by_email(self, email: str) -> Optional[User]:
        row = self.db.execute(
            "SELECT name, email FROM users WHERE email = ?",
            (email,)
        ).fetchone()
        return User(name=row[0], email=row[1]) if row else None

class UserNotifier:
    """Handles user notifications"""
    def __init__(self, email_service):
        self.email_service = email_service

    def send_welcome_email(self, user: User):
        self.email_service.send(
            to=user.email,
            subject="Welcome!",
            body=f"Hello {user.name}, welcome to our platform!"
        )

class UserReportGenerator:
    """Handles user report generation"""
    @staticmethod
    def generate(user: User) -> str:
        return f"User Report: {user.name} ({user.email})"
```

### 4.2 Open/Closed Principle (OCP)

```python
# Bad: Modify existing code every time a new shape is added
class AreaCalculator:
    def calculate(self, shape):
        if shape.type == "circle":
            return 3.14159 * shape.radius ** 2
        elif shape.type == "rectangle":
            return shape.width * shape.height
        elif shape.type == "triangle":
            return 0.5 * shape.base * shape.height
        # Need to modify here every time a new shape is added...

# Good: Open for extension, closed for modification
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height

    def area(self) -> float:
        return 0.5 * self.base * self.height

# Adding a new shape requires no changes to existing code
class Trapezoid(Shape):
    def __init__(self, top: float, bottom: float, height: float):
        self.top = top
        self.bottom = bottom
        self.height = height

    def area(self) -> float:
        return 0.5 * (self.top + self.bottom) * self.height

# The caller depends only on the Shape interface
def total_area(shapes: list[Shape]) -> float:
    return sum(s.area() for s in shapes)
```

### 4.3 Liskov Substitution Principle (LSP)

```python
# Bad: LSP violation: Square is not a proper subtype of Rectangle
class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    def area(self):
        return self._width * self._height

class Square(Rectangle):
    """Square: Always keeps width and height equal"""
    def __init__(self, side):
        super().__init__(side, side)

    @Rectangle.width.setter
    def width(self, value):
        self._width = value
        self._height = value  # Height also changes!

    @Rectangle.height.setter
    def height(self, value):
        self._width = value
        self._height = value

# This becomes problematic
def resize_rectangle(rect: Rectangle):
    rect.width = 10
    rect.height = 5
    assert rect.area() == 50  # Fails with Square!

# Good: Design that respects LSP
class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Square(Shape):
    def __init__(self, side: float):
        self.side = side

    def area(self) -> float:
        return self.side ** 2
```

### 4.4 Interface Segregation Principle (ISP)

```python
# Bad: Interface is too large
class Worker(ABC):
    @abstractmethod
    def work(self): pass

    @abstractmethod
    def eat(self): pass

    @abstractmethod
    def sleep(self): pass

    @abstractmethod
    def code(self): pass

    @abstractmethod
    def test(self): pass

    @abstractmethod
    def deploy(self): pass

class Robot(Worker):
    def work(self):
        print("Working...")

    def eat(self):
        raise NotImplementedError("Robots don't eat!")  # Violation!

    def sleep(self):
        raise NotImplementedError("Robots don't sleep!")  # Violation!

    def code(self):
        print("Coding...")

    def test(self):
        print("Testing...")

    def deploy(self):
        print("Deploying...")

# Good: Segregated interfaces
class Workable(ABC):
    @abstractmethod
    def work(self): pass

class Eatable(ABC):
    @abstractmethod
    def eat(self): pass

class Sleepable(ABC):
    @abstractmethod
    def sleep(self): pass

class Codeable(ABC):
    @abstractmethod
    def code(self): pass

class Testable(ABC):
    @abstractmethod
    def test(self): pass

class Deployable(ABC):
    @abstractmethod
    def deploy(self): pass

class Human(Workable, Eatable, Sleepable, Codeable, Testable, Deployable):
    def work(self): print("Working...")
    def eat(self): print("Eating...")
    def sleep(self): print("Sleeping...")
    def code(self): print("Coding...")
    def test(self): print("Testing...")
    def deploy(self): print("Deploying...")

class Robot(Workable, Codeable, Testable, Deployable):
    def work(self): print("Working...")
    def code(self): print("Coding...")
    def test(self): print("Testing...")
    def deploy(self): print("Deploying...")
```

### 4.5 Dependency Inversion Principle (DIP)

```python
# Bad: High-level module directly depends on low-level module
import mysql.connector

class UserService:
    def __init__(self):
        # Directly depends on MySQL
        self.connection = mysql.connector.connect(
            host="localhost",
            database="myapp"
        )

    def get_user(self, user_id):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        return cursor.fetchone()

# Good: Depend on abstractions
class UserRepository(ABC):
    """Abstract repository (interface)"""
    @abstractmethod
    def find_by_id(self, user_id: str) -> Optional[User]:
        pass

    @abstractmethod
    def save(self, user: User) -> None:
        pass

    @abstractmethod
    def delete(self, user_id: str) -> None:
        pass

class MySQLUserRepository(UserRepository):
    """MySQL implementation"""
    def __init__(self, connection):
        self.connection = connection

    def find_by_id(self, user_id: str) -> Optional[User]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        return User(name=row[1], email=row[2]) if row else None

    def save(self, user: User) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s)",
            (user.name, user.email)
        )
        self.connection.commit()

    def delete(self, user_id: str) -> None:
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        self.connection.commit()

class InMemoryUserRepository(UserRepository):
    """In-memory implementation for testing"""
    def __init__(self):
        self.users: dict[str, User] = {}

    def find_by_id(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    def save(self, user: User) -> None:
        self.users[user.email] = user

    def delete(self, user_id: str) -> None:
        self.users.pop(user_id, None)

class UserService:
    """Depends on abstract repository (does not know the concrete implementation)"""
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def get_user(self, user_id: str) -> Optional[User]:
        return self.repository.find_by_id(user_id)

    def register_user(self, name: str, email: str) -> User:
        user = User(name=name, email=email)
        self.repository.save(user)
        return user

# Production environment
production_service = UserService(MySQLUserRepository(db_connection))

# Test environment
test_service = UserService(InMemoryUserRepository())
```

---

## 5. DRY / KISS / YAGNI

### 5.1 DRY (Don't Repeat Yourself)

```python
# DRY = Avoid duplication of knowledge (not just textual duplication of code)

# Bad: Knowledge duplication: Tax rate calculation logic scattered
class OrderService:
    def calculate_order_total(self, items):
        subtotal = sum(item.price * item.quantity for item in items)
        tax = subtotal * 0.10  # 10% tax rate hard-coded
        return subtotal + tax

class InvoiceService:
    def generate_invoice(self, order):
        subtotal = order.subtotal
        tax = subtotal * 0.10  # Same tax rate in another place
        return {"subtotal": subtotal, "tax": tax, "total": subtotal + tax}

class ReportService:
    def calculate_revenue(self, orders):
        total = 0
        for order in orders:
            total += order.subtotal * 1.10  # Same tax rate again
        return total

# Good: Consolidate tax calculation in one place
TAX_RATE = 0.10

class TaxCalculator:
    @staticmethod
    def calculate_tax(amount: float) -> float:
        return amount * TAX_RATE

    @staticmethod
    def calculate_total_with_tax(amount: float) -> float:
        return amount + TaxCalculator.calculate_tax(amount)

class OrderService:
    def calculate_order_total(self, items):
        subtotal = sum(item.price * item.quantity for item in items)
        return TaxCalculator.calculate_total_with_tax(subtotal)

class InvoiceService:
    def generate_invoice(self, order):
        tax = TaxCalculator.calculate_tax(order.subtotal)
        total = TaxCalculator.calculate_total_with_tax(order.subtotal)
        return {"subtotal": order.subtotal, "tax": tax, "total": total}

# --- Be careful not to over-apply DRY ---
# Do not DRY-ify accidental duplication
# Bad: Forcefully DRY-ifying accidental duplication
def format_address(address):
    return f"{address.street}, {address.city} {address.zip}"

def format_log_entry(entry):
    # Happens to have the same format, but is a different concept
    return f"{entry.message}, {entry.level} {entry.timestamp}"

# Good: Keep separate concepts separate
# format_address and format_log_entry just happen to look similar
# They are likely to diverge in different directions in the future
```

### 5.2 KISS (Keep It Simple, Stupid)

```python
# KISS = Avoid unnecessary complexity

# Bad: Overly complex (abuse of metaprogramming)
class DynamicValidator:
    def __init__(self):
        self._rules = {}

    def register_rule(self, field, rule_type, **kwargs):
        if field not in self._rules:
            self._rules[field] = []
        self._rules[field].append((rule_type, kwargs))

    def validate(self, data):
        errors = []
        for field, rules in self._rules.items():
            value = data.get(field)
            for rule_type, kwargs in rules:
                validator = getattr(self, f"_validate_{rule_type}")
                result = validator(value, **kwargs)
                if not result:
                    errors.append(f"{field}: {rule_type} validation failed")
        return errors

    def _validate_required(self, value):
        return value is not None and value != ""

    def _validate_min_length(self, value, length=0):
        return len(str(value)) >= length

    def _validate_max_length(self, value, length=float("inf")):
        return len(str(value)) <= length

    def _validate_pattern(self, value, regex=""):
        return bool(re.match(regex, str(value)))

# The caller side is also complex
validator = DynamicValidator()
validator.register_rule("email", "required")
validator.register_rule("email", "pattern", regex=r"^[\w.-]+@[\w.-]+\.\w+$")
validator.register_rule("password", "required")
validator.register_rule("password", "min_length", length=8)

# Good: Simple and clear
def validate_registration(data: dict) -> list[str]:
    """Validate registration data"""
    errors = []

    email = data.get("email", "")
    if not email:
        errors.append("Email address is required")
    elif not re.match(r"^[\w.-]+@[\w.-]+\.\w+$", email):
        errors.append("Email address format is invalid")

    password = data.get("password", "")
    if not password:
        errors.append("Password is required")
    elif len(password) < 8:
        errors.append("Password must be at least 8 characters")

    return errors

# Consider abstraction only when it becomes truly necessary
```

### 5.3 YAGNI (You Aren't Gonna Need It)

```python
# YAGNI = Don't implement features you don't need yet

# Bad: Over-engineering based on "might need someday"
class UserService:
    def __init__(self, repository, cache, event_bus, logger,
                 rate_limiter, circuit_breaker, metrics_collector,
                 feature_flags, plugin_manager):
        self.repository = repository
        self.cache = cache
        self.event_bus = event_bus
        self.logger = logger
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.metrics_collector = metrics_collector
        self.feature_flags = feature_flags
        self.plugin_manager = plugin_manager

    def get_user(self, user_id):
        # In practice, cache, event_bus, circuit_breaker are unused
        self.logger.info(f"Getting user {user_id}")
        self.metrics_collector.increment("user.get")
        return self.repository.find_by_id(user_id)

# Good: Implement only what you need now
class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def get_user(self, user_id: str) -> Optional[User]:
        return self.repository.find_by_id(user_id)

# Add caching when it is truly needed
# Add event bus when it is truly needed
# -> Extend incrementally as needs arise
```

---

## 6. Writing Comments

### 6.1 Good Comments and Bad Comments

```python
# --- Bad comments (comments that explain the code) ---
# Bad: Repeating what can be understood from reading the code
i = 0  # Initialize i to 0
users = get_users()  # Get users
total = price * quantity  # amount = unit price x quantity

# Bad: Lying comments (diverged from code)
# Returns the user's age
def get_user_name(user_id):  # Actually returns the name
    pass

# Bad: Closing brace comments (a sign that the structure is too complex)
for user in users:
    for order in user.orders:
        for item in order.items:
            process(item)
        # end for item
    # end for order
# end for user

# Bad: Commented-out code
# def old_calculate_tax(amount):
#     return amount * 0.08
# No one knows when this code was written, and no one dares delete it

# --- Good comments ---
# Good: Explain WHY (why it is done this way)
# Batch size limited to 1000 for performance reasons
# Processing 10000+ at once caused memory exhaustion (Issue #234)
BATCH_SIZE = 1000

# Good: Legal comments
# Copyright (c) 2024 Example Corp. All rights reserved.
# Licensed under the MIT License.

# Good: TODO/FIXME (making technical debt explicit)
# TODO(gaku): Add caching after Redis is introduced (#456)
# FIXME: Bug in timezone handling at month-end (#789)
# HACK: Workaround for API spec bug. Fix planned for v3.0 release

# Good: Warning about consequences
# Warning: This function calls an external API and may take 3+ seconds
def fetch_exchange_rates():
    pass

# Good: Explaining regular expressions
# Matches Japanese phone numbers: 03-1234-5678, 090-1234-5678, etc.
PHONE_PATTERN = re.compile(r"^0\d{1,4}-\d{1,4}-\d{4}$")

# Good: Explaining business rules
# Per Article 29 of the Consumption Tax Act, amounts less than 1 yen are truncated
tax = math.floor(subtotal * TAX_RATE)
```

### 6.2 Docstring Best Practices

```python
def calculate_compound_interest(
    principal: float,
    annual_rate: float,
    years: int,
    compounds_per_year: int = 12
) -> float:
    """Calculate compound interest.

    Calculates the future value based on the specified principal,
    annual interest rate, and compounding frequency.

    Args:
        principal: Principal amount (yen). Must be a positive number.
        annual_rate: Annual interest rate (0.05 = 5%).
        years: Number of years for the investment.
        compounds_per_year: Number of compounding periods per year (default: 12=monthly).

    Returns:
        Future value of principal plus interest (yen).

    Raises:
        ValueError: If principal is negative.
        ValueError: If years is 0 or less.

    Examples:
        >>> calculate_compound_interest(1000000, 0.05, 10)
        1647009.49
        >>> calculate_compound_interest(1000000, 0.03, 5, compounds_per_year=1)
        1159274.07

    Note:
        This calculation does not account for taxes.
        In actual investments, approximately 20% tax applies to interest income.
    """
    if principal < 0:
        raise ValueError("Principal must be a positive number")
    if years <= 0:
        raise ValueError("Years must be 1 or greater")

    return principal * (1 + annual_rate / compounds_per_year) ** (compounds_per_year * years)


class ShoppingCart:
    """A class for managing a shopping cart.

    Provides features for managing items added to the cart,
    calculating totals, and applying coupons.

    Attributes:
        items: List of items in the cart.
        user_id: User ID of the cart owner.
        created_at: Cart creation timestamp.

    Example:
        >>> cart = ShoppingCart(user_id="u001")
        >>> cart.add_item(Item("apple", 100), quantity=3)
        >>> cart.total
        300
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.items: list[CartItem] = []
        self.created_at = datetime.now()
```

---

## 7. Error Handling

### 7.1 Exception Handling Best Practices

```python
# --- 1. Catch specific exceptions ---
# Bad: Catching all exceptions
try:
    result = process_data(data)
except Exception:
    print("Something went wrong")

# Good: Catch specific exceptions
try:
    result = process_data(data)
except ValueError as e:
    logger.warning(f"Invalid data: {e}")
    return default_value
except ConnectionError as e:
    logger.error(f"Database connection failed: {e}")
    raise ServiceUnavailableError("Database is temporarily unavailable") from e
except TimeoutError as e:
    logger.error(f"Operation timed out: {e}")
    raise

# --- 2. Define custom exceptions ---
class AppError(Exception):
    """Application common base exception"""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.error_code = error_code

class ValidationError(AppError):
    """Validation error"""
    def __init__(self, field: str, message: str):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field

class NotFoundError(AppError):
    """Resource not found error"""
    def __init__(self, resource: str, resource_id: str):
        message = f"{resource} with id '{resource_id}' not found"
        super().__init__(message, error_code="NOT_FOUND")
        self.resource = resource
        self.resource_id = resource_id

class AuthenticationError(AppError):
    """Authentication error"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, error_code="AUTHENTICATION_ERROR")

class AuthorizationError(AppError):
    """Authorization error"""
    def __init__(self, required_permission: str):
        message = f"Permission '{required_permission}' is required"
        super().__init__(message, error_code="AUTHORIZATION_ERROR")
        self.required_permission = required_permission

# Usage example
class UserService:
    def get_user(self, user_id: str) -> User:
        user = self.repository.find_by_id(user_id)
        if user is None:
            raise NotFoundError("User", user_id)
        return user

    def create_user(self, name: str, email: str) -> User:
        if not name:
            raise ValidationError("name", "Name is required")
        if not self._is_valid_email(email):
            raise ValidationError("email", "Invalid email format")
        if self.repository.find_by_email(email):
            raise ValidationError("email", "Email already registered")
        return self.repository.save(User(name=name, email=email))

# --- 3. Log exceptions ---
import logging
logger = logging.getLogger(__name__)

def process_payment(order):
    try:
        payment_gateway.charge(order.amount, order.payment_method)
    except PaymentDeclinedError as e:
        # Expected business error: WARNING level
        logger.warning(
            "Payment declined for order %s: %s",
            order.id, e,
            extra={"order_id": order.id, "amount": order.amount}
        )
        raise
    except PaymentGatewayError as e:
        # System error: ERROR level + stack trace
        logger.error(
            "Payment gateway error for order %s: %s",
            order.id, e,
            exc_info=True,
            extra={"order_id": order.id}
        )
        raise ServiceUnavailableError("Payment service is temporarily unavailable") from e

# --- 4. Exception translation (exception chaining) ---
def get_user_profile(user_id: str) -> UserProfile:
    try:
        data = external_api.fetch_user(user_id)
    except requests.ConnectionError as e:
        raise ServiceUnavailableError(
            "External API is unavailable"
        ) from e  # from e preserves the original exception
    except requests.Timeout as e:
        raise ServiceUnavailableError(
            "External API request timed out"
        ) from e

    try:
        return UserProfile.from_dict(data)
    except KeyError as e:
        raise DataIntegrityError(
            f"Missing required field in API response: {e}"
        ) from e
```

### 7.2 Safe Handling of Null/None

```python
# --- 1. None check patterns ---
# Bad: Scattered None checks
def get_user_city(user_id):
    user = repository.find_by_id(user_id)
    if user is not None:
        address = user.address
        if address is not None:
            city = address.city
            if city is not None:
                return city
    return "Unknown"

# Good: Early return
def get_user_city(user_id: str) -> str:
    user = repository.find_by_id(user_id)
    if user is None:
        return "Unknown"
    if user.address is None:
        return "Unknown"
    return user.address.city or "Unknown"

# Good: Use Optional type (Python 3.10+)
from typing import Optional

def find_user(user_id: str) -> Optional[User]:
    """Returns None if user is not found"""
    return repository.find_by_id(user_id)

# --- 2. Null Object pattern ---
class NullUser:
    """Default value when user does not exist"""
    name = "Guest"
    email = ""
    is_active = False
    permissions = frozenset()

    def has_permission(self, permission: str) -> bool:
        return False

NULL_USER = NullUser()

def get_user_or_default(user_id: str) -> User:
    user = repository.find_by_id(user_id)
    return user if user is not None else NULL_USER

# --- 3. Leveraging default values ---
# dict's get method
config = {"debug": True, "log_level": "INFO"}
debug_mode = config.get("debug", False)  # False if key is missing
log_level = config.get("log_level", "WARNING")  # Default WARNING
timeout = config.get("timeout", 30)  # 30 if key is missing
```

### 7.3 Resource Management

```python
# --- Use context managers (with statement) ---

# Bad: Manual close (leaks on exception)
file = open("data.csv", "r")
data = file.read()
file.close()  # close() is not called if an exception occurs

# Good: Automatic close with with statement
with open("data.csv", "r") as file:
    data = file.read()  # close() is called even if an exception occurs

# --- Custom context manager ---
from contextlib import contextmanager

@contextmanager
def database_transaction(connection):
    """Database transaction management"""
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()

# Usage example
with database_transaction(db_connection) as cursor:
    cursor.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    cursor.execute("INSERT INTO logs (action) VALUES (?)", ("user_created",))
    # Automatically rolls back if an exception occurs

# --- Managing multiple resources ---
with open("input.csv") as infile, open("output.csv", "w") as outfile:
    for line in infile:
        processed = process_line(line)
        outfile.write(processed)

# --- Timeout management ---
import signal

@contextmanager
def timeout(seconds):
    """Operation timeout management"""
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Usage example
with timeout(5):
    result = long_running_operation()
```

---

## 8. Highly Testable Code

### 8.1 Characteristics of Testable Code

```python
# --- Hard to test code ---
# Bad: Dependency on global state
import datetime

user_count = 0  # Global variable

class UserService:
    def create_user(self, name, email):
        global user_count
        user = User(
            name=name,
            email=email,
            created_at=datetime.datetime.now()  # Directly depends on current time
        )
        user_count += 1  # Modifies global state
        db.save(user)  # Directly depends on global DB
        return user

# --- Easy to test code ---
# Good: Dependency injection
class UserService:
    def __init__(self, repository: UserRepository, clock: Clock):
        self.repository = repository
        self.clock = clock

    def create_user(self, name: str, email: str) -> User:
        user = User(
            name=name,
            email=email,
            created_at=self.clock.now()
        )
        self.repository.save(user)
        return user

# Test
class FakeClock:
    def __init__(self, fixed_time):
        self._time = fixed_time
    def now(self):
        return self._time

def test_create_user():
    fake_repo = InMemoryUserRepository()
    fake_clock = FakeClock(datetime.datetime(2024, 1, 1, 12, 0, 0))
    service = UserService(repository=fake_repo, clock=fake_clock)

    user = service.create_user("Alice", "alice@example.com")

    assert user.name == "Alice"
    assert user.email == "alice@example.com"
    assert user.created_at == datetime.datetime(2024, 1, 1, 12, 0, 0)
    assert fake_repo.find_by_email("alice@example.com") is not None
```

### 8.2 Test Structure (AAA Pattern)

```python
import pytest

class TestShoppingCart:
    """Shopping cart tests"""

    def test_add_item_increases_total(self):
        # Arrange
        cart = ShoppingCart()
        item = Item(name="Apple", price=100)

        # Act
        cart.add_item(item, quantity=3)

        # Assert
        assert cart.total == 300
        assert cart.item_count == 3

    def test_apply_percentage_coupon(self):
        # Arrange
        cart = ShoppingCart()
        cart.add_item(Item("Apple", 100), quantity=5)
        coupon = PercentageCoupon(rate=0.10)  # 10% off

        # Act
        cart.apply_coupon(coupon)

        # Assert
        assert cart.total == 450  # 500 - 50

    def test_remove_item_decreases_total(self):
        # Arrange
        cart = ShoppingCart()
        apple = Item("Apple", 100)
        cart.add_item(apple, quantity=3)

        # Act
        cart.remove_item(apple, quantity=1)

        # Assert
        assert cart.total == 200
        assert cart.item_count == 2

    def test_empty_cart_has_zero_total(self):
        # Arrange & Act
        cart = ShoppingCart()

        # Assert
        assert cart.total == 0
        assert cart.item_count == 0
        assert cart.is_empty

    def test_cannot_add_negative_quantity(self):
        # Arrange
        cart = ShoppingCart()
        item = Item("Apple", 100)

        # Act & Assert
        with pytest.raises(ValueError, match="Quantity must be positive"):
            cart.add_item(item, quantity=-1)
```

### 8.3 Test Doubles (Mock/Stub/Fake/Spy)

```python
from unittest.mock import Mock, patch, MagicMock

# --- Stub: Returns a fixed value ---
class StubPaymentGateway:
    """Payment gateway that always succeeds"""
    def charge(self, amount, card):
        return PaymentResult(success=True, transaction_id="stub-txn-001")

# --- Fake: Simplified implementation ---
class FakeEmailService:
    """Does not actually send, but retains send history"""
    def __init__(self):
        self.sent_emails = []

    def send(self, to, subject, body):
        self.sent_emails.append({
            "to": to, "subject": subject, "body": body
        })

# --- Mock: Verify invocations ---
def test_order_sends_confirmation_email():
    # Arrange
    mock_email = Mock(spec=EmailService)
    service = OrderService(email_service=mock_email)
    order = Order(customer_email="alice@example.com", items=["item1"])

    # Act
    service.complete_order(order)

    # Assert: Was it called with the correct arguments?
    mock_email.send_confirmation.assert_called_once_with(
        to="alice@example.com",
        order_id=order.id
    )

# --- Spy: Executes actual processing while recording calls ---
class SpyLogger:
    def __init__(self, real_logger):
        self.real_logger = real_logger
        self.logged_messages = []

    def info(self, message):
        self.logged_messages.append(("INFO", message))
        self.real_logger.info(message)

    def error(self, message):
        self.logged_messages.append(("ERROR", message))
        self.real_logger.error(message)

# --- Mocking with patch ---
@patch("app.services.external_api.fetch_user")
def test_get_user_profile(mock_fetch):
    # Arrange
    mock_fetch.return_value = {"name": "Alice", "email": "alice@example.com"}

    # Act
    service = UserProfileService()
    profile = service.get_profile("user-001")

    # Assert
    assert profile.name == "Alice"
    mock_fetch.assert_called_once_with("user-001")
```

---

## 9. Refactoring

### 9.1 When to Refactor

```
When You Should Refactor:

  1. Before adding features: Make the structure amenable to new features
  2. During bug fixes: In the process of removing the root cause of bugs
  3. During code review: Address improvement points raised in review
  4. Boy Scout Rule: Leave code a little better than you found it

When You Should NOT Refactor:

  1. Right before a deadline: Risk is too high
  2. When you don't understand the behavior: Write tests first
  3. Code without tests: Add tests first
  4. Large-scale rewrites: Proceed incrementally
```

### 9.2 Common Refactoring Techniques

```python
# --- 1. Extract Method ---
# Bad: Long function
def print_invoice(invoice):
    print("=" * 50)
    print(f"Invoice #{invoice.id}")
    print(f"Date: {invoice.date}")
    print("-" * 50)
    for item in invoice.items:
        total = item.price * item.quantity
        print(f"  {item.name}: {item.quantity} x {item.price} = {total}")
    print("-" * 50)
    subtotal = sum(i.price * i.quantity for i in invoice.items)
    tax = subtotal * 0.10
    total = subtotal + tax
    print(f"  Subtotal: {subtotal}")
    print(f"  Tax (10%): {tax}")
    print(f"  Total: {total}")
    print("=" * 50)

# Good: Extract methods
def print_invoice(invoice):
    print_header(invoice)
    print_line_items(invoice.items)
    print_totals(invoice.items)

def print_header(invoice):
    print("=" * 50)
    print(f"Invoice #{invoice.id}")
    print(f"Date: {invoice.date}")
    print("-" * 50)

def print_line_items(items):
    for item in items:
        total = item.price * item.quantity
        print(f"  {item.name}: {item.quantity} x {item.price} = {total}")
    print("-" * 50)

def print_totals(items):
    subtotal = calculate_subtotal(items)
    tax = calculate_tax(subtotal)
    total = subtotal + tax
    print(f"  Subtotal: {subtotal}")
    print(f"  Tax (10%): {tax}")
    print(f"  Total: {total}")
    print("=" * 50)

# --- 2. Extract Variable ---
# Bad: Complex expression
def is_eligible_for_premium(user):
    return (user.age >= 18 and
            user.years_of_membership >= 2 and
            user.total_purchases >= 100000 and
            user.is_active and
            not user.has_violations and
            user.last_login_days_ago <= 30)

# Good: Split into meaningful variables
def is_eligible_for_premium(user):
    is_adult = user.age >= 18
    is_long_term_member = user.years_of_membership >= 2
    is_high_value_customer = user.total_purchases >= 100000
    is_recently_active = user.last_login_days_ago <= 30
    has_good_standing = user.is_active and not user.has_violations

    return (is_adult and
            is_long_term_member and
            is_high_value_customer and
            is_recently_active and
            has_good_standing)

# --- 3. Decompose Conditional ---
# Bad: Complex conditional expression
def calculate_charge(date, quantity):
    if (date.month >= 6 and date.month <= 9):
        charge = quantity * SUMMER_RATE + SUMMER_SERVICE_CHARGE
    else:
        charge = quantity * WINTER_RATE + WINTER_SERVICE_CHARGE
    return charge

# Good: Extract condition into a function
def is_summer(date):
    return 6 <= date.month <= 9

def summer_charge(quantity):
    return quantity * SUMMER_RATE + SUMMER_SERVICE_CHARGE

def winter_charge(quantity):
    return quantity * WINTER_RATE + WINTER_SERVICE_CHARGE

def calculate_charge(date, quantity):
    if is_summer(date):
        return summer_charge(quantity)
    return winter_charge(quantity)

# --- 4. Introduce Parameter Object ---
# Bad: Too many arguments
def search_products(
    category, min_price, max_price,
    brand, color, size,
    sort_by, sort_order,
    page, page_size
):
    pass

# Good: Consolidate into parameter objects
@dataclass
class ProductSearchCriteria:
    category: str = ""
    min_price: float = 0
    max_price: float = float("inf")
    brand: str = ""
    color: str = ""
    size: str = ""

@dataclass
class PaginationParams:
    page: int = 1
    page_size: int = 20
    sort_by: str = "created_at"
    sort_order: str = "desc"

def search_products(
    criteria: ProductSearchCriteria,
    pagination: PaginationParams
):
    pass

# --- 5. Replace Nested Conditional with Guard Clauses ---
# Bad: Nested conditional branches
def calculate_pay(employee):
    if employee.is_separated:
        result = calculate_separated_pay(employee)
    else:
        if employee.is_retired:
            result = calculate_retired_pay(employee)
        else:
            result = calculate_normal_pay(employee)
    return result

# Good: Guard clauses
def calculate_pay(employee):
    if employee.is_separated:
        return calculate_separated_pay(employee)
    if employee.is_retired:
        return calculate_retired_pay(employee)
    return calculate_normal_pay(employee)
```

### 9.3 Safe Refactoring Practices

```
Safe Refactoring Steps:

  1. Verify tests pass
  |
  2. Refactor in small steps
  |
  3. Verify tests pass
  |
  4. Commit
  |
  5. Repeat steps 2-4

  Key Points:
  - One type of refactoring per commit
  - Do not mix feature changes and refactoring
  - If there are no tests, add tests first
  - Leverage IDE refactoring features (reduce manual work)
  - Combine with pair programming and code review

  Recommended Tools:
  - Python: ruff, mypy, black, isort
  - JavaScript/TypeScript: ESLint, Prettier
  - Java: IntelliJ IDEA, SpotBugs, PMD
  - Go: gofmt, golint, go vet
```

---

## 10. Code Review

### 10.1 Code Review Perspectives

```
Code Review Checklist:

  [ ] Correctness
    - Is the logic free of errors?
    - Are edge cases considered?
    - Is boundary value handling correct?
    - Is concurrency safety ensured?

  [ ] Readability
    - Are names appropriate?
    - Are functions an appropriate length?
    - Are comments necessary and sufficient?
    - Is the code's intent clear?

  [ ] Maintainability
    - Does it follow SOLID principles?
    - Is the abstraction level appropriate?
    - Are tests sufficient?
    - Is error handling appropriate?

  [ ] Performance
    - Are there unnecessary N+1 queries?
    - Are appropriate indexes being used?
    - Is there a possibility of memory leaks?
    - Is caching being used appropriately?

  [ ] Security
    - Are SQL injection countermeasures sufficient?
    - Is input validation performed?
    - Are authentication/authorization checks appropriate?
    - Is sensitive information not being logged?
```

### 10.2 Writing Effective Review Comments

```
Review Comment Best Practices:

  1. Be specific
    Bad: "This is not good"
    Good: "This function is 30 lines long and has 3 responsibilities:
        validation, calculation, and notification.
        Extracting each into a separate function would improve readability"

  2. Explain the reason
    Bad: "Make this a constant"
    Good: "It's unclear what this 86400 represents.
        Defining it as a named constant like SECONDS_PER_DAY = 86400
        would make the code's intent clear"

  3. Include a suggestion
    Bad: "Error handling is insufficient"
    Good: "A ConnectionError could occur from the external API call.
        I suggest adding try-except to catch it and implement
        appropriate retry or fallback handling"

  4. Indicate severity
    [must]    Must fix (bugs, security issues)
    [should]  Strongly recommended (readability, maintainability improvements)
    [nit]     Minor observation (style, naming fine-tuning)
    [question] Question (for understanding)
    [praise]  Praise for good points

  5. Include positive feedback
    "This test case coverage is excellent!
     The edge case coverage is particularly thorough."
```

---

## 11. Language-Specific Clean Code Practices

### 11.1 TypeScript/JavaScript

```typescript
// --- Code made safe through types ---

// Bad: Abuse of any type
function processData(data: any): any {
    return data.map((item: any) => item.value * 2);
}

// Good: Proper type definitions
interface DataItem {
    id: string;
    value: number;
    label: string;
}

interface ProcessedItem {
    id: string;
    doubledValue: number;
}

function processData(data: DataItem[]): ProcessedItem[] {
    return data.map(item => ({
        id: item.id,
        doubledValue: item.value * 2
    }));
}

// --- Using Union types and Type Guards ---
type Result<T> =
    | { success: true; data: T }
    | { success: false; error: string };

function fetchUser(id: string): Result<User> {
    try {
        const user = database.findUser(id);
        if (!user) {
            return { success: false, error: `User ${id} not found` };
        }
        return { success: true, data: user };
    } catch (e) {
        return { success: false, error: `Failed to fetch user: ${e}` };
    }
}

// Type-safe handling on the caller side
const result = fetchUser("u001");
if (result.success) {
    console.log(result.data.name);  // Type safe
} else {
    console.error(result.error);
}

// --- Ensuring immutability ---
// Bad: Mutable
const cart = { items: [], total: 0 };
cart.items.push(newItem);  // Direct mutation
cart.total = calculateTotal(cart.items);

// Good: Immutable
interface Cart {
    readonly items: readonly CartItem[];
    readonly total: number;
}

function addItem(cart: Cart, item: CartItem): Cart {
    const newItems = [...cart.items, item];
    return {
        items: newItems,
        total: calculateTotal(newItems)
    };
}
```

### 11.2 Go

```go
// --- Error handling patterns ---

// Bad: Ignoring errors
func getUser(id string) *User {
    user, _ := db.FindUser(id)  // Error ignored
    return user
}

// Good: Handle errors properly
func getUser(id string) (*User, error) {
    user, err := db.FindUser(id)
    if err != nil {
        return nil, fmt.Errorf("failed to get user %s: %w", id, err)
    }
    if user == nil {
        return nil, ErrUserNotFound
    }
    return user, nil
}

// --- Using interfaces ---

// Good: Small interfaces (Go style)
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type ReadWriter interface {
    Reader
    Writer
}

// Good: Accept small interfaces, return concrete types
type UserRepository interface {
    FindByID(id string) (*User, error)
    Save(user *User) error
}

func NewUserService(repo UserRepository) *UserService {
    return &UserService{repo: repo}
}

// --- Struct constructor patterns ---
type Server struct {
    host    string
    port    int
    timeout time.Duration
    logger  *log.Logger
}

// Functional Options pattern
type ServerOption func(*Server)

func WithPort(port int) ServerOption {
    return func(s *Server) {
        s.port = port
    }
}

func WithTimeout(timeout time.Duration) ServerOption {
    return func(s *Server) {
        s.timeout = timeout
    }
}

func WithLogger(logger *log.Logger) ServerOption {
    return func(s *Server) {
        s.logger = logger
    }
}

func NewServer(host string, opts ...ServerOption) *Server {
    s := &Server{
        host:    host,
        port:    8080,  // Default value
        timeout: 30 * time.Second,
        logger:  log.Default(),
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// Usage example
server := NewServer("localhost",
    WithPort(3000),
    WithTimeout(60*time.Second),
)
```

### 11.3 Java/Kotlin

```java
// --- Correct usage of Optional (Java) ---

// Bad: Misuse of Optional
public Optional<User> getUser(String id) {
    User user = repository.findById(id);
    return Optional.ofNullable(user);  // This part is OK
}

// Caller-side misuse
Optional<User> optUser = getUser("u001");
if (optUser.isPresent()) {  // Bad: isPresent() before get() = same as null check
    User user = optUser.get();
}

// Good: Functional style using Optional
public String getUserDisplayName(String userId) {
    return repository.findById(userId)
        .map(User::getDisplayName)
        .orElse("Unknown User");
}

public void sendWelcomeEmail(String userId) {
    repository.findById(userId)
        .ifPresent(user -> emailService.sendWelcome(user.getEmail()));
}

public User getActiveUser(String userId) {
    return repository.findById(userId)
        .filter(User::isActive)
        .orElseThrow(() -> new UserNotFoundException(userId));
}
```

```kotlin
// --- Kotlin clean code ---

// Using data classes
data class User(
    val id: String,
    val name: String,
    val email: String,
    val isActive: Boolean = true
)

// Extending existing classes with extension functions
fun String.isValidEmail(): Boolean =
    matches(Regex("^[\\w.-]+@[\\w.-]+\\.\\w+$"))

fun List<User>.activeUsers(): List<User> =
    filter { it.isActive }

// Using scope functions properly
// let: Null-safe transformation
val displayName = user?.let { "${it.name} (${it.email})" } ?: "Guest"

// apply: Object initialization
val config = ServerConfig().apply {
    host = "localhost"
    port = 8080
    timeout = Duration.ofSeconds(30)
}

// also: Side effects (logging, etc.)
val result = repository.findById(id)
    .also { logger.info("Found user: ${it?.name}") }

// run: Computation on an object
val summary = order.run {
    "Order #$id: $itemCount items, total = $total"
}

// Exhaustive pattern matching with sealed classes
sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Failure(val error: String) : Result<Nothing>()
    object Loading : Result<Nothing>()
}

fun handleResult(result: Result<User>) = when (result) {
    is Result.Success -> showUser(result.data)
    is Result.Failure -> showError(result.error)
    Result.Loading -> showLoading()
    // when is exhaustive so else is not needed
}
```

---

## 12. Clean Code Anti-Pattern Collection

### 12.1 Over-Engineering

```python
# Bad: Excessive abstraction for something that does only one thing
class AbstractUserValidatorFactory(ABC):
    @abstractmethod
    def create_validator(self) -> AbstractUserValidator:
        pass

class AbstractUserValidator(ABC):
    @abstractmethod
    def validate(self, user: AbstractUserDTO) -> AbstractValidationResult:
        pass

class AbstractUserDTO(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

class AbstractValidationResult(ABC):
    @abstractmethod
    def is_valid(self) -> bool:
        pass

class ConcreteUserValidatorFactory(AbstractUserValidatorFactory):
    def create_validator(self):
        return ConcreteUserValidator()

class ConcreteUserValidator(AbstractUserValidator):
    def validate(self, user):
        return ConcreteValidationResult(bool(user.get_name()))

# What hundreds of lines of code achieve is...

# Good: This is sufficient
def validate_user(name: str) -> bool:
    return bool(name)
```

### 12.2 Premature Optimization

```python
# Bad: Premature optimization
# Optimizing with bit operations without even profiling first
def is_even(n):
    return not (n & 1)  # Hard to read

# Good: Prioritize readability first
def is_even(n):
    return n % 2 == 0  # Clear

# Good: Optimize only when performance is truly a problem
# And document the reason in a comment
# Per performance profiling results, this function accounted for 30% of total execution time
# Optimized to bit operations (benchmark: 2.1ms -> 0.8ms, Issue #567)
def is_even_optimized(n):
    return not (n & 1)
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

The knowledge from this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Naming | Be intention-revealing. Be searchable. Maintain consistency |
| Functions | Keep small. One responsibility. Few arguments |
| Code Smells | Long functions, duplication, deep nesting -> Refactoring |
| SOLID | Follow the 5 principles: SRP, OCP, LSP, ISP, DIP |
| Principles | DRY, KISS, YAGNI |
| Comments | Write WHY. Don't write what the code already tells |
| Error Handling | Specific exceptions, custom exceptions, resource management |
| Testing | Design for testability. AAA pattern |
| Refactoring | Small steps, safely. Tests come first |
| Review | Be specific, give reasons, include suggestions |

---

## Recommended Next Guides

---

## References
1. Martin, R. C. "Clean Code." Prentice Hall, 2008.
2. Fowler, M. "Refactoring." 2nd Edition, Addison-Wesley, 2018.
3. Martin, R. C. "Clean Architecture." Prentice Hall, 2017.
4. Hunt, A. and Thomas, D. "The Pragmatic Programmer." 20th Anniversary Edition, Addison-Wesley, 2019.
5. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018.
6. Kernighan, B. W. and Pike, R. "The Practice of Programming." Addison-Wesley, 1999.
7. McConnell, S. "Code Complete." 2nd Edition, Microsoft Press, 2004.
8. Beck, K. "Test Driven Development: By Example." Addison-Wesley, 2002.
