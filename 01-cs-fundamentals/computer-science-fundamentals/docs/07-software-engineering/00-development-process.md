# Software Development Processes

> Great software comes from great processes. The transition from waterfall to agile to DevOps is the evolution of adaptability to uncertainty itself.

## What You Will Learn in This Chapter

- [ ] Explain the historical evolution of software development processes
- [ ] Understand the structure, advantages, and limitations of the waterfall model
- [ ] Distinguish and apply major agile development frameworks (Scrum, XP, Kanban)
- [ ] Design and build DevOps and CI/CD pipelines
- [ ] Gain a holistic understanding of the end-to-end flow from requirements definition to deployment
- [ ] Select appropriate project management methodologies
- [ ] Recognize and avoid anti-patterns in development processes


## Prerequisites

Having the following knowledge before reading this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. History and Background of Development Processes

### 1.1 The Software Crisis and the Beginning of Systematization

In the late 1960s, software development faced serious problems. Project delays, budget overruns, and poor quality became the norm, and this situation was called the "Software Crisis." At the 1968 NATO Software Engineering Conference, the need to apply engineering principles to software development was discussed.

The root cause of this crisis was that while software complexity was growing exponentially, the development process remained ad hoc and unsystematic. While hardware manufacturing had established quality control methods, there was no equivalent for software.

```
Structure of the Software Crisis:

  1960s                        1970s                       2000s onward
  +-----------+              +----------------+           +----------------+
  | Ad hoc    |  Crisis      | Systematic     | Demand    | Adaptive       |
  | development| ----------> | processes      | for       | processes      |
  |           |              | introduced     | change    | emerge         |
  |           |              | (Waterfall)    | --------> | (Agile)        |
  | - Craft   |              | - Phased mgmt  |           |                |
  | - No docs |              | - Doc-focused  |           | - Iterative    |
  | - No plans|              | - Quality gates|           | - Customer     |
  +-----------+              +----------------+           |   collaboration|
                                                          | - Continuous   |
                                                          |   improvement  |
                                                          +----------------+
```

### 1.2 Lineage of Development Process Models

Software development process models have evolved along the following lineage.

```
Evolution of Development Process Models:

  1970  Waterfall (Royce, 1970)
    |     +-- Sequential, phased approach
    |
  1981  Spiral Model (Boehm, 1986)
    |     +-- Risk-driven + iterative
    |
  1990  RUP: Rational Unified Process (1998)
    |     +-- Use-case driven + architecture-centric + iterative
    |
  1996  XP: eXtreme Programming (Beck, 1996)
    |     +-- Lightweight process, pair programming, TDD
    |
  2001  Agile Manifesto
    |     +-- 4 values and 12 principles
    |
  2002  Scrum Formalized (Schwaber & Sutherland)
    |     +-- Defined roles, events, and artifacts
    |
  2009  DevOps Movement
    |     +-- Integration of development and operations, automation
    |
  2017  GitOps (Weaveworks)
    |     +-- Git as the Single Source of Truth
    |
  2022  Platform Engineering
        +-- Building Internal Developer Platforms (IDP)
```

### 1.3 Criteria for Choosing a Process Model

Which process model to adopt depends on the characteristics of the project. The following table organizes the major decision criteria.

| Criterion | Waterfall-Leaning | Agile-Leaning |
|-----------|-------------------|---------------|
| Requirements Stability | Clear requirements with few changes | Uncertain requirements with frequent changes |
| Customer Involvement | Initial and final stages only | Continuous involvement possible |
| Team Size | Large (50+ people) | Small to medium (3-9 people) |
| Regulatory Compliance | Strict regulations (medical, aviation) | Relatively lenient regulations |
| Risk Tolerance | Risk-averse | Risk-tolerant |
| Release Frequency | 1-2 times per year | Weekly to daily |
| Technical Uncertainty | Using known technologies | Exploring new technologies |
| Documentation Requirements | Detailed documentation required | Minimal documentation sufficient |

---

## 2. The Waterfall Model

### 2.1 Structure of the Model

The waterfall model originates from a model described by Winston W. Royce in his 1970 paper "Managing the Development of Large Software Systems." Each phase is executed sequentially, and in principle, going back to a previous phase is not permitted.

```
Structure of the Waterfall Model:

  +------------------+
  |  Requirements     |  <- Define what to build
  |  Definition       |
  +--------+---------+
           v
  +------------------+
  |  System Design    |  <- Design how to build it
  |                   |
  +--------+---------+
           v
  +------------------+
  |  Implementation   |  <- Write the code
  |                   |
  +--------+---------+
           v
  +------------------+
  |  Testing          |  <- Verify quality
  |  (Verification)   |
  +--------+---------+
           v
  +------------------+
  |  Maintenance      |  <- Operate and improve
  |                   |
  +------------------+
```

An important historical fact is that Royce himself described this simple sequential model as "risky and inviting failure," and in the latter half of his paper, he proposed an improved version that included iterative elements. However, the industry widely adopted only this simple sequential model.

### 2.2 Details of Each Phase

#### Requirements Definition Phase

Requirements definition is the phase that clarifies "what" the system should do. Both functional requirements (the functions the system provides) and non-functional requirements (performance, security, availability, etc.) are documented.

```python
# Code Example 1: Structuring and managing requirements definitions
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class RequirementPriority(Enum):
    """MoSCoW Priority Classification"""
    MUST = "Must Have"      # Mandatory requirement
    SHOULD = "Should Have"  # Important but not mandatory
    COULD = "Could Have"    # Nice to have
    WONT = "Won't Have"     # Out of scope for this release


class RequirementType(Enum):
    FUNCTIONAL = "Functional"          # Functional requirement
    NON_FUNCTIONAL = "Non-Functional"  # Non-functional requirement
    CONSTRAINT = "Constraint"          # Constraint


@dataclass
class Requirement:
    """Data class representing a requirement"""
    id: str                              # e.g., "REQ-001"
    title: str                           # Requirement title
    description: str                     # Detailed description
    type: RequirementType                # Requirement type
    priority: RequirementPriority        # Priority
    acceptance_criteria: list[str]       # Acceptance criteria
    source: str = ""                     # Source of the requirement (stakeholder name, etc.)
    dependencies: list[str] = field(default_factory=list)  # Dependent requirement IDs
    status: str = "Draft"               # Draft / Approved / Implemented / Verified
    rationale: Optional[str] = None      # Reason why this requirement is needed

    def is_testable(self) -> bool:
        """Verify that acceptance criteria are defined"""
        return len(self.acceptance_criteria) > 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.type.value,
            "priority": self.priority.value,
            "acceptance_criteria": self.acceptance_criteria,
            "dependencies": self.dependencies,
            "status": self.status,
        }


class RequirementsDocument:
    """Class for managing requirements documents"""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.requirements: dict[str, Requirement] = {}
        self._next_id = 1

    def add_requirement(self, title: str, description: str,
                        req_type: RequirementType,
                        priority: RequirementPriority,
                        acceptance_criteria: list[str],
                        **kwargs) -> Requirement:
        req_id = f"REQ-{self._next_id:03d}"
        self._next_id += 1
        req = Requirement(
            id=req_id, title=title, description=description,
            type=req_type, priority=priority,
            acceptance_criteria=acceptance_criteria, **kwargs
        )
        self.requirements[req_id] = req
        return req

    def get_by_priority(self, priority: RequirementPriority) -> list[Requirement]:
        return [r for r in self.requirements.values() if r.priority == priority]

    def validate_all(self) -> list[str]:
        """Validate consistency of all requirements"""
        issues = []
        for req in self.requirements.values():
            if not req.is_testable():
                issues.append(f"{req.id}: Acceptance criteria not defined")
            for dep in req.dependencies:
                if dep not in self.requirements:
                    issues.append(f"{req.id}: Dependency {dep} does not exist")
        return issues

    def coverage_report(self) -> dict:
        """Generate a requirements coverage report"""
        total = len(self.requirements)
        if total == 0:
            return {"total": 0}
        by_status = {}
        for req in self.requirements.values():
            by_status[req.status] = by_status.get(req.status, 0) + 1
        return {
            "total": total,
            "by_status": by_status,
            "must_have_count": len(self.get_by_priority(RequirementPriority.MUST)),
            "verified_ratio": by_status.get("Verified", 0) / total,
        }

    def export_json(self) -> str:
        return json.dumps(
            {
                "project": self.project_name,
                "requirements": [r.to_dict() for r in self.requirements.values()],
            },
            indent=2, ensure_ascii=False
        )


# Usage example
doc = RequirementsDocument("E-Commerce Site Construction Project")

doc.add_requirement(
    title="User Registration Feature",
    description="Users can register with email address and password",
    req_type=RequirementType.FUNCTIONAL,
    priority=RequirementPriority.MUST,
    acceptance_criteria=[
        "Email address format validation is performed",
        "Password must be at least 8 characters with alphanumeric and special characters",
        "A confirmation email is sent upon registration completion",
        "Duplicate registration with an existing email address results in an error",
    ],
    rationale="A fundamental feature that is a prerequisite for using the service"
)

doc.add_requirement(
    title="Response Time",
    description="All pages must load within 2 seconds",
    req_type=RequirementType.NON_FUNCTIONAL,
    priority=RequirementPriority.MUST,
    acceptance_criteria=[
        "Response time at the 95th percentile is within 2 seconds",
        "Achieved under a load of 1,000 concurrent users",
    ]
)

# Validation
issues = doc.validate_all()
report = doc.coverage_report()
print(f"Requirements count: {report['total']}, Must Have: {report['must_have_count']}")
print(f"Verified ratio: {report['verified_ratio']:.0%}")
```

#### Design Phase

In the design phase, technical architecture and detailed design are created to realize the requirements. It is common to divide this into high-level design (external design) and detailed design (internal design).

High-level design covers the overall system architecture, module decomposition, interface definitions, database design, and screen design. Detailed design covers internal logic of each module, class design, and algorithm selection.

#### Implementation Phase

In the implementation phase, code is written based on the design documents. Adherence to coding standards, appropriate commenting, and creation of unit tests are required.

#### Testing Phase

In the testing phase, unit testing, integration testing, system testing, and acceptance testing are conducted in stages. Based on the test plan, each test level verifies that expected quality criteria are met.

#### Maintenance Phase

In the post-release operations and maintenance phase, bug fixes (corrective maintenance), feature additions (adaptive maintenance), performance improvements (perfective maintenance), and future problem prevention (preventive maintenance) are carried out continuously.

### 2.3 Advantages and Limitations of Waterfall

**Advantages:**

- Phases are clear, making progress management easy
- Deliverables (documents) for each phase are comprehensive
- Well-suited for managing large-scale projects
- Facilitates traceability required in regulated industries (medical, aerospace, defense)
- The approach is clear even for newcomers and less experienced teams

**Limitations:**

- High cost of responding to requirement changes (change costs increase exponentially in later phases)
- Working software cannot be verified until the very end
- Major rework if critical issues are discovered during the testing phase
- Customer feedback is delayed
- Not suitable for projects with high uncertainty

### 2.4 V-Model

The V-Model is an extension of the waterfall model that explicitly maps each development phase to a corresponding testing phase.

```
V-Model:

  Requirements Definition  --------------------------------  Acceptance Testing
       \                                                   /
    High-Level Design  ----------------------------  System Testing
         \                                         /
      Detailed Design  --------------------  Integration Testing
           \                               /
         Coding  ----------------  Unit Testing

  <- Development Phases ->     <- Testing Phases ->

  Correspondence:
    Requirements Definition verification  -> Acceptance Testing (are requirements met?)
    High-Level Design verification        -> System Testing (does it work as a whole?)
    Detailed Design verification          -> Integration Testing (is inter-module interaction correct?)
    Coding verification                   -> Unit Testing (are individual functions/classes correct?)
```

---

## 3. Agile Development

### 3.1 The Agile Manifesto

In February 2001, 17 software developers gathered in Snowbird, Utah, and drafted the "Manifesto for Agile Software Development." This manifesto was an antithesis to the traditional heavyweight development processes.

**4 Values:**

1. **Individuals and interactions over processes and tools** -- Tools and processes are important, but communication between team members is the most valuable
2. **Working software over comprehensive documentation** -- Documentation is necessary, but working software is the most reliable measure of progress
3. **Customer collaboration over contract negotiation** -- Contracts are necessary, but collaborating with customers to build the best product is what matters
4. **Responding to change over following a plan** -- Plans are important, but being able to respond flexibly to change is more important

**12 Principles (Summary):**

1. Satisfy the customer through early and continuous delivery of valuable software
2. Welcome changing requirements, even late in development
3. Deliver working software frequently, in short cycles
4. Business people and developers must work together daily
5. Build projects around motivated individuals
6. Face-to-face conversation is the most efficient method of conveying information
7. Working software is the primary measure of progress
8. Maintain a sustainable pace of development
9. Continuously pay attention to technical excellence and good design
10. Simplicity -- the art of maximizing the amount of work not done -- is essential
11. The best architectures, requirements, and designs emerge from self-organizing teams
12. The team regularly reflects on how to become more effective and adjusts its behavior

### 3.2 Scrum

Scrum is the most widely adopted agile framework. It was formalized by Ken Schwaber and Jeff Sutherland.

#### Scrum's 3 Roles

| Role | Responsibility | Specific Activities |
|------|---------------|---------------------|
| Product Owner (PO) | Maximize the value of the product | Prioritizing the backlog, defining acceptance criteria, coordinating with stakeholders |
| Scrum Master (SM) | Support the practice of Scrum | Removing impediments, facilitating process improvement, supporting team self-organization |
| Development Team | Create the increment | Design, coding, testing, deployment (composed of 3-9 members) |

#### Scrum's 5 Events

```
Scrum Sprint Cycle:

  +------------- Sprint (1-4 weeks) -------------+
  |                                               |
  |  Sprint         Daily           Sprint        |
  |  Planning       Scrum           Review        |
  |  (up to 8h)    (up to 15min)   (up to 4h)    |
  |    |              |               |            |
  |    |   +----------+               |            |
  |    |   |  Daily    |              |            |
  |    v   v  repeat   v              v            |
  |  [Plan]->[Develop & Test]-> ... ->[Demo/Review]|
  |                                    |           |
  |                              Sprint            |
  |                              Retrospective     |
  |                              (up to 3h)        |
  +------------------------------------------------+
        |                                  |
        v                                  v
  Product                          Increment
  Backlog                         (Releasable
  (Prioritized)                    deliverable)

  Event List:
    1. Sprint              : Development time-box
    2. Sprint Planning     : Plan what and how to build
    3. Daily Scrum         : 15-minute sync meeting
    4. Sprint Review       : Demo and inspection of deliverables
    5. Sprint Retrospective: Process reflection and improvement
```

#### Scrum's 3 Artifacts

1. **Product Backlog**: A prioritized list of all features and improvements needed for the product
2. **Sprint Backlog**: The list of items to be implemented in the current sprint + the plan for achieving them
3. **Increment**: The "Done" product increment produced as a result of the sprint

```python
# Code Example 2: A simple Scrum board implementation
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional


class StoryStatus(Enum):
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    IN_REVIEW = "In Review"
    DONE = "Done"


class StorySize(Enum):
    """Story Points (Fibonacci sequence)"""
    XS = 1
    S = 2
    M = 3
    L = 5
    XL = 8
    XXL = 13


@dataclass
class UserStory:
    """User Story"""
    id: str
    title: str
    description: str        # "As a [user], I want [goal], so that [benefit]"
    story_points: StorySize
    acceptance_criteria: list[str]
    status: StoryStatus = StoryStatus.TODO
    assignee: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def points(self) -> int:
        return self.story_points.value

    def start(self, assignee: str) -> None:
        self.status = StoryStatus.IN_PROGRESS
        self.assignee = assignee

    def submit_for_review(self) -> None:
        if self.status != StoryStatus.IN_PROGRESS:
            raise ValueError("Only in-progress stories can be submitted for review")
        self.status = StoryStatus.IN_REVIEW

    def complete(self) -> None:
        self.status = StoryStatus.DONE
        self.completed_at = datetime.now()


@dataclass
class Sprint:
    """Sprint"""
    id: str
    goal: str
    start_date: datetime
    duration_weeks: int = 2
    stories: list[UserStory] = field(default_factory=list)

    @property
    def end_date(self) -> datetime:
        return self.start_date + timedelta(weeks=self.duration_weeks)

    @property
    def total_points(self) -> int:
        return sum(s.points for s in self.stories)

    @property
    def completed_points(self) -> int:
        return sum(s.points for s in self.stories if s.status == StoryStatus.DONE)

    @property
    def velocity(self) -> int:
        """This sprint's velocity (completed story points)"""
        return self.completed_points

    def add_story(self, story: UserStory) -> None:
        self.stories.append(story)

    def burndown_data(self) -> list[dict]:
        """Generate burndown chart data"""
        remaining = self.total_points
        data = [{"day": 0, "remaining": remaining, "ideal": remaining}]
        total_days = self.duration_weeks * 5  # Business days
        daily_ideal = remaining / total_days

        completed_stories = sorted(
            [s for s in self.stories if s.completed_at],
            key=lambda s: s.completed_at
        )

        for day in range(1, total_days + 1):
            for story in completed_stories:
                days_elapsed = (story.completed_at - self.start_date).days
                if days_elapsed == day:
                    remaining -= story.points
            data.append({
                "day": day,
                "remaining": remaining,
                "ideal": max(0, self.total_points - daily_ideal * day),
            })
        return data

    def summary(self) -> str:
        status_counts = {}
        for story in self.stories:
            key = story.status.value
            status_counts[key] = status_counts.get(key, 0) + 1
        lines = [
            f"Sprint: {self.id} - {self.goal}",
            f"Duration: {self.start_date.date()} to {self.end_date.date()}",
            f"Total Points: {self.total_points}, Completed: {self.completed_points}",
            f"Velocity: {self.velocity}",
            "By Status:"
        ]
        for status, count in status_counts.items():
            lines.append(f"  {status}: {count}")
        return "\n".join(lines)


class ScrumBoard:
    """Scrum Board (manages multiple sprints)"""

    def __init__(self, team_name: str):
        self.team_name = team_name
        self.sprints: list[Sprint] = []
        self.product_backlog: list[UserStory] = []

    def add_to_backlog(self, story: UserStory) -> None:
        self.product_backlog.append(story)

    def create_sprint(self, sprint_id: str, goal: str,
                      start_date: datetime) -> Sprint:
        sprint = Sprint(id=sprint_id, goal=goal, start_date=start_date)
        self.sprints.append(sprint)
        return sprint

    def plan_sprint(self, sprint: Sprint, story_ids: list[str]) -> None:
        """Sprint Planning: Select stories from the backlog"""
        for sid in story_ids:
            for i, story in enumerate(self.product_backlog):
                if story.id == sid:
                    sprint.add_story(story)
                    self.product_backlog.pop(i)
                    break

    def average_velocity(self, last_n: int = 3) -> float:
        """Average velocity of the last n sprints"""
        recent = self.sprints[-last_n:]
        if not recent:
            return 0.0
        return sum(s.velocity for s in recent) / len(recent)


# Usage example
board = ScrumBoard("Development Team A")

# Add stories to the backlog
stories = [
    UserStory("US-001", "Login Feature",
              "As a user, I want to log in, so that I can access my account",
              StorySize.M,
              ["Can log in with email and password", "Error message is displayed on failure"]),
    UserStory("US-002", "Product Search",
              "As a user, I want to search products, so that I can find what I need",
              StorySize.L,
              ["Can search by keyword", "Can filter by category"]),
    UserStory("US-003", "Cart Feature",
              "As a user, I want to add items to cart, so that I can purchase multiple items",
              StorySize.XL,
              ["Can add items to cart", "Can change quantity", "Can remove from cart"]),
]
for s in stories:
    board.add_to_backlog(s)

# Create and plan a sprint
sprint1 = board.create_sprint("Sprint-1", "Implement Basic Authentication", datetime.now())
board.plan_sprint(sprint1, ["US-001", "US-002"])

# Progress of work
sprint1.stories[0].start("Tanaka")
sprint1.stories[0].submit_for_review()
sprint1.stories[0].complete()

print(sprint1.summary())
```

### 3.3 Extreme Programming (XP)

Extreme Programming (XP) is an agile development methodology proposed by Kent Beck. Its distinguishing feature is its emphasis on engineering practices to improve software quality.

#### XP's 5 Values

1. **Communication**: Emphasis on dialogue within and outside the team
2. **Simplicity**: Don't build anything beyond what is needed
3. **Feedback**: Quick feedback for course correction
4. **Courage**: Making necessary changes without fear
5. **Respect**: Mutual respect among team members

#### XP Practices

| Practice | Description | Effect |
|----------|------------|--------|
| Pair Programming | Two people write code together | Improved code quality, knowledge sharing |
| Test-Driven Development (TDD) | Write tests first | Improved design quality, regression bug prevention |
| Refactoring | Improve the internal structure of code | Suppression of technical debt |
| Continuous Integration | Integrate code frequently | Early detection of integration issues |
| Small Releases | Release frequently in small units | Risk reduction, early feedback |
| Coding Standards | Unified code style | Improved readability, reduced key-person dependency |
| Collective Ownership | Code belongs to everyone | Elimination of bottlenecks |
| Sustainable Pace | Based on a 40-hour work week | Burnout prevention, quality maintenance |
| Metaphor | Share system understanding through metaphors | Promoting common understanding |
| Planning Game | Estimate using story cards | Realistic planning |

#### TDD Cycle

```python
# Code Example 3: TDD's Red-Green-Refactor cycle

# === Step 1: Red (Write a failing test) ===
import unittest


class TestShoppingCart(unittest.TestCase):
    """Shopping cart tests"""

    def test_empty_cart_has_zero_total(self):
        cart = ShoppingCart()
        self.assertEqual(cart.total(), 0)

    def test_add_single_item(self):
        cart = ShoppingCart()
        cart.add_item("Apple", price=150, quantity=1)
        self.assertEqual(cart.total(), 150)

    def test_add_multiple_items(self):
        cart = ShoppingCart()
        cart.add_item("Apple", price=150, quantity=2)
        cart.add_item("Banana", price=100, quantity=3)
        self.assertEqual(cart.total(), 600)  # 150*2 + 100*3

    def test_apply_percentage_discount(self):
        cart = ShoppingCart()
        cart.add_item("Apple", price=1000, quantity=1)
        cart.apply_discount(percent=10)
        self.assertEqual(cart.total(), 900)

    def test_remove_item(self):
        cart = ShoppingCart()
        cart.add_item("Apple", price=150, quantity=1)
        cart.remove_item("Apple")
        self.assertEqual(cart.total(), 0)

    def test_item_count(self):
        cart = ShoppingCart()
        cart.add_item("Apple", price=150, quantity=2)
        cart.add_item("Banana", price=100, quantity=3)
        self.assertEqual(cart.item_count(), 5)


# === Step 2: Green (Minimum implementation to make tests pass) ===

@dataclass
class CartItem:
    name: str
    price: int
    quantity: int

    @property
    def subtotal(self) -> int:
        return self.price * self.quantity


class ShoppingCart:
    def __init__(self):
        self._items: dict[str, CartItem] = {}
        self._discount_percent: int = 0

    def add_item(self, name: str, price: int, quantity: int) -> None:
        if name in self._items:
            self._items[name].quantity += quantity
        else:
            self._items[name] = CartItem(name=name, price=price, quantity=quantity)

    def remove_item(self, name: str) -> None:
        self._items.pop(name, None)

    def apply_discount(self, percent: int) -> None:
        if not 0 <= percent <= 100:
            raise ValueError("Discount rate must be between 0 and 100")
        self._discount_percent = percent

    def item_count(self) -> int:
        return sum(item.quantity for item in self._items.values())

    def subtotal(self) -> int:
        return sum(item.subtotal for item in self._items.values())

    def total(self) -> int:
        sub = self.subtotal()
        discount = sub * self._discount_percent // 100
        return sub - discount


# === Step 3: Refactor (Improve the code while keeping tests green) ===
# The above implementation is already refactored:
# - CartItem separated into an independent data class
# - Subtotal calculation as a property
# - Discount logic clearly separated
```

### 3.4 Kanban

Kanban is a development methodology originating from the Toyota Production System that centers on work visualization and WIP (Work In Progress) limits for flow management. Unlike Scrum, it does not have fixed sprint cycles and emphasizes continuous flow.

#### Kanban Principles

1. **Visualize current work**: Make all work visible on the Kanban board
2. **Limit WIP**: Limit the number of concurrent tasks to improve efficiency
3. **Manage flow**: Optimize the flow of work
4. **Make process policies explicit**: Document definitions like "Definition of Done"
5. **Implement feedback loops**: Conduct regular retrospectives
6. **Improve collaboratively, evolve experimentally**: The whole team works on improvement

#### Kanban Board Example

```
Kanban Board:

  Backlog    | To Do     | In Progress | Review    |   Done
  (No limit) | (WIP: 5)  |  (WIP: 3)   | (WIP: 2)  |
  -----------+----------+-------------+----------+----------
  [Search    | [API     | [Auth       | [DB      | [Initial
   improve]  |  design] |  implement] |  migrate]|  setup]
  [Notif.    | [Tests]  | [UI         |          | [CI
   feature]  |          |  implement] |          |  build]
  [Analytics |          |             |          | [Doc
   feature]  |          |             |          |  prep]
  [i18n]     |          |             |          |
  -----------+----------+-------------+----------+----------

  When WIP limit is exceeded -> Do not start new work, finish existing work first

  Lead Time Measurement:
    Record the period from start date to completion date
    Identify bottlenecks (columns where WIP is at the limit) and improve
```

### 3.5 Comparison of Scrum, XP, and Kanban

| Aspect | Scrum | XP | Kanban |
|--------|-------|-----|--------|
| Iteration | Fixed-length sprints | Fixed-length | None (continuous flow) |
| Roles | PO, SM, Dev Team | Coach, Customer, Developers | Not specifically defined |
| Change Tolerance | No changes during sprint | Flexible | Changes allowed anytime |
| Estimation | Story points | Story points | Optional |
| Metrics | Velocity | Velocity | Lead time, throughput |
| Technical Practices | Not prescribed | Emphasizes TDD, pair programming, etc. | Not prescribed |
| Use Cases | Product development | Technical quality focus | Maintenance, operations, support |
| Ease of Adoption | Moderate | Difficult (discipline required) | Relatively easy |

---

## 4. DevOps and CI/CD

### 4.1 DevOps Overview

DevOps is a collection of culture, practices, and tools that breaks down the wall between Development and Operations to accelerate and improve the quality of software delivery. It emerged from the DevOpsDays conference in 2009.

The core of DevOps is breaking down the "wall" between development teams and operations teams. Traditionally, development teams wanted to "deliver new features as quickly as possible" while operations teams wanted to "keep the system stable" -- conflicting goals. DevOps resolves this conflict and builds a system where both collaborate to continuously deliver business value.

#### The CALMS Framework of DevOps

The CALMS framework is widely used to understand the cultural aspects of DevOps.

- **C - Culture**: Collaboration between teams, learning organization
- **A - Automation**: CI/CD, Infrastructure as Code, test automation
- **L - Lean**: Waste elimination, small batch sizes, WIP limits
- **M - Measurement**: Metrics-driven improvement, DORA metrics
- **S - Sharing**: Knowledge sharing, transparency, postmortems

#### DORA Metrics (Four Keys)

The DORA (DevOps Research and Assessment) metrics are widely adopted as four indicators for measuring DevOps performance.

| Metric | Elite | High | Medium | Low |
|--------|-------|------|--------|-----|
| Deployment Frequency | On-demand (multiple times/day) | Weekly to monthly | Monthly to semi-annually | Less than semi-annually |
| Lead Time (commit to deploy) | Less than 1 hour | 1 day to 1 week | 1 week to 1 month | 1 month to 6 months |
| Change Failure Rate | 0-15% | 16-30% | 16-30% | 46-60% |
| Mean Time to Recovery (MTTR) | Less than 1 hour | Less than 1 day | 1 day to 1 week | 6 months or more |

### 4.2 CI/CD Pipeline

CI (Continuous Integration) and CD (Continuous Delivery/Deployment) are the technical foundations of DevOps.

```
CI/CD Pipeline Overview:

  Developer
    |
    v  git push
  +-------------+
  | Source Code  |
  | Repository   |  (GitHub / GitLab)
  +------+------+
         v  Webhook Trigger
  +-----------------------------------------------+
  |              CI Pipeline                        |
  |                                                 |
  |  +--------+  +--------+  +--------+  +--------+|
  |  | Build  |->| Static |->| Unit   |->| Integ. ||
  |  |        |  | Analy. |  | Tests  |  | Tests  ||
  |  +--------+  +--------+  +--------+  +--------+|
  |                                                 |
  +---------------------+---------------------------+
                        v  On success
  +-----------------------------------------------+
  |              CD Pipeline                        |
  |                                                 |
  |  +--------+  +--------+  +--------+  +--------+|
  |  |Artifact|->|Staging |->|  E2E   |->| Prod   ||
  |  | Create |  | Deploy |  | Tests  |  | Deploy ||
  |  +--------+  +--------+  +--------+  +--------+|
  |                                     ^           |
  |                               Approval Gate     |
  |                             (manual/automatic)  |
  +-----------------------------------------------+
```

```yaml
# Code Example 4: GitHub Actions CI/CD Pipeline Configuration
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ===== CI: Build and Test =====
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install ruff mypy
          pip install -r requirements.txt
      - name: Lint with ruff
        run: ruff check .
      - name: Type check with mypy
        run: mypy src/

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    services:
      postgres:
        image: postgres:16
        env:
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
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml
      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/testdb
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: fs
          scan-ref: .
          severity: CRITICAL,HIGH

  # ===== CD: Build and Deploy =====
  build-and-push:
    needs: [lint, test, security-scan]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying ${{ github.sha }} to staging environment"
          # kubectl set image deployment/app \
          #   app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://app.example.com
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying ${{ github.sha }} to production environment"
          # Production deployment command
```

### 4.3 Infrastructure as Code (IaC)

Infrastructure as Code is a methodology for managing infrastructure configuration as code. It eliminates manual configuration and achieves reproducibility, version control, and automation.

Major tools include Terraform (multi-cloud declarative IaC), AWS CloudFormation (AWS native), Pulumi (writable in general-purpose programming languages), and Ansible (configuration management).

### 4.4 Monitoring and Observability

In DevOps, monitoring and observability for understanding the state of systems are essential.

**Three Pillars of Observability:**

1. **Metrics**: Numerical measurement data (CPU utilization, response time, error rate)
2. **Logs**: Time-series records of events (application logs, access logs)
3. **Traces**: Tracking the processing path of requests (distributed tracing)

Major tools include Prometheus + Grafana (metrics collection and visualization), ELK Stack / Loki (log management), Jaeger / OpenTelemetry (distributed tracing), and Datadog / New Relic (integrated monitoring SaaS).

---

## 5. Flow from Requirements to Deployment

### 5.1 Overall Flow in Modern Development

In modern software development, a development flow that combines agile's iterative approach with DevOps automation has become mainstream.

```
Overall Modern Development Flow:

  +-----------------------------------------------------------+
  |                   Product Discovery                        |
  |  User Research -> Problem Definition -> Hypothesis ->      |
  |  Prototype -> Validation                                   |
  +---------------------------+-------------------------------+
                              v
  +-----------------------------------------------------------+
  |                   Product Backlog                           |
  |  Prioritized user stories / technical tasks                 |
  +---------------------------+-------------------------------+
                              v
  +--------- Sprint (2 weeks) ---------+
  |                                     |
  |  Planning                           |
  |    |                                |
  |  Design -> Implement -> Code Review |
  |    |                                |
  |  Test (automated + manual)          |
  |    |                                |
  |  Review + Retrospective             |
  |                                     |
  +--------------+---------------------+
                 v
  +------------------------+
  |   CI/CD Pipeline        |
  |  Build -> Test ->       |
  |  Deploy (automated)     |
  +--------------+---------+
                 v
  +------------------------+
  |    Production           |
  |  Monitoring             |
  |  Feedback Collection    |
  +--------------+---------+
                 |
                 +---> Feedback to backlog (loop)
```

### 5.2 Requirements Definition Practices

#### User Story Mapping

User Story Mapping, proposed by Jeff Patton, is a technique for organizing user activities chronologically and assigning priorities.

```
User Story Map (E-Commerce Site Example):

  User Activities (chronological from left to right):
  -------------------------------------------------------
  Find Products   ->  Select Products  ->  Purchase     ->  Wait for Delivery
  -------------------------------------------------------
  |                    |                    |                 |
  | [Keyword Search]   | [Product Detail]  | [Add to Cart]  | [Order Status]
  | [Category List]    | [Review Browsing] | [Payment]      | [Delivery Track]
  | [Recommendations]  | [Comparison]      | [Address Input]| [Receipt Confirm]
  | [Filter Feature]   | [Favorites]       | [Coupon Apply] | [Return Request]
  |                    | [Stock Check]     | [Gift Setting] |
  -------------------------------------------------------
  ^ MVP Line (features needed for the first release)
  -------------------------------------------------------
  ^ v2.0 Line
  -------------------------------------------------------
```

#### Event Storming

Event Storming is a workshop technique devised by Alberto Brandolini for exploring system behavior centered on domain events. It has high affinity with Domain-Driven Design (DDD).

### 5.3 Design Phase Practices

#### Architecture Decision Records (ADR)

Architecture Decision Records (ADR) are created to record and track important architectural decisions.

```markdown
# ADR-001: Adopt GraphQL for API Communication

## Status
Approved (2024-01-15)

## Context
The data required by the frontend varies by screen,
and with REST API, over-fetching and under-fetching
occur frequently. A mobile app is also planned,
requiring efficient bandwidth utilization.

## Decision
Adopt GraphQL as the API communication protocol.
Use Apollo Server for the server implementation.

## Rationale
- Clients can retrieve only the data they need
- Schema definitions through the type system make the API contract clear
- Resolver patterns can integrate multiple data sources
- Mobile and web can efficiently use the same API

## Consequences
- The team will need to invest in learning GraphQL
- Countermeasures for the N+1 problem (DataLoader) are needed
- Cache strategy differs from REST and needs redesign
- Performance monitoring tools need to be selected
```

### 5.4 Code Review Best Practices

Code review is an important practice that achieves both quality improvement and knowledge sharing.

**Key Points for Effective Code Reviews:**

1. **Keep review size small**: Target 200-400 lines per pull request. Split large changes into multiple PRs
2. **Clarify review perspectives**: Functional correctness, security, performance, maintainability, test coverage
3. **Provide constructive feedback**: Not just pointing out issues, but also suggesting improvements
4. **Automate what can be automated**: Run formatting, linting, and type checking in CI, and let humans focus on design and logic discussions

---

## 6. Project Management Methods

### 6.1 Estimation Techniques

Estimation in software development is one of the most challenging tasks. Let's compare the major techniques.

#### Planning Poker

A technique where all team members use Fibonacci sequence cards (1, 2, 3, 5, 8, 13, 21) to estimate the relative size of each story. When opinions differ significantly, discussion takes place and re-estimation is done.

#### T-Shirt Sizing

A technique for rough classification using sizes like S, M, L, XL. Suitable for early-stage roadmap creation and quick classification of large backlogs.

#### Three-Point Estimation

A technique that calculates the expected value from three estimates: optimistic (O), most likely (M), and pessimistic (P).

Expected Value = (O + 4M + P) / 6

```python
# Code Example 5: Project management tool implementation
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import math


class TaskStatus(Enum):
    BACKLOG = "Backlog"
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    DONE = "Done"
    BLOCKED = "Blocked"


@dataclass
class ThreePointEstimate:
    """Three-Point Estimation"""
    optimistic: float     # Optimistic value (days)
    most_likely: float    # Most likely value (days)
    pessimistic: float    # Pessimistic value (days)

    @property
    def expected(self) -> float:
        """PERT expected value"""
        return (self.optimistic + 4 * self.most_likely + self.pessimistic) / 6

    @property
    def standard_deviation(self) -> float:
        """Standard deviation"""
        return (self.pessimistic - self.optimistic) / 6

    @property
    def variance(self) -> float:
        """Variance"""
        return self.standard_deviation ** 2

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Calculate confidence interval (normal distribution approximation)"""
        # 95% confidence interval -> z = 1.96
        z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_values.get(confidence, 1.96)
        margin = z * self.standard_deviation
        return (self.expected - margin, self.expected + margin)


@dataclass
class Task:
    """Project Task"""
    id: str
    name: str
    estimate: Optional[ThreePointEstimate] = None
    status: TaskStatus = TaskStatus.BACKLOG
    assignee: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    actual_days: Optional[float] = None

    @property
    def estimated_days(self) -> float:
        if self.estimate:
            return self.estimate.expected
        return 0.0


class ProjectPlanner:
    """Project Planning Tool"""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        self.tasks[task.id] = task

    def total_estimate(self) -> ThreePointEstimate:
        """Total estimate for all tasks"""
        total_o = sum(t.estimate.optimistic for t in self.tasks.values() if t.estimate)
        total_m = sum(t.estimate.most_likely for t in self.tasks.values() if t.estimate)
        total_p = sum(t.estimate.pessimistic for t in self.tasks.values() if t.estimate)
        return ThreePointEstimate(total_o, total_m, total_p)

    def monte_carlo_simulation(self, iterations: int = 10000) -> dict:
        """Completion date prediction via Monte Carlo simulation"""
        import random

        results = []
        tasks_with_estimates = [t for t in self.tasks.values() if t.estimate]

        for _ in range(iterations):
            total = 0.0
            for task in tasks_with_estimates:
                est = task.estimate
                # Approximate PERT distribution with triangular distribution
                sample = random.triangular(
                    est.optimistic, est.pessimistic, est.most_likely
                )
                total += sample
            results.append(total)

        results.sort()
        return {
            "mean": sum(results) / len(results),
            "median": results[len(results) // 2],
            "p50": results[int(len(results) * 0.50)],
            "p75": results[int(len(results) * 0.75)],
            "p85": results[int(len(results) * 0.85)],
            "p95": results[int(len(results) * 0.95)],
            "min": results[0],
            "max": results[-1],
        }

    def velocity_forecast(self, velocity_history: list[int],
                          remaining_points: int) -> dict:
        """Release forecast based on velocity"""
        if not velocity_history:
            return {"error": "Velocity history is required"}

        avg_velocity = sum(velocity_history) / len(velocity_history)
        std_dev = (
            sum((v - avg_velocity) ** 2 for v in velocity_history)
            / len(velocity_history)
        ) ** 0.5

        sprints_needed = math.ceil(remaining_points / avg_velocity) if avg_velocity > 0 else float('inf')
        optimistic = math.ceil(remaining_points / (avg_velocity + std_dev)) if (avg_velocity + std_dev) > 0 else float('inf')
        pessimistic = math.ceil(remaining_points / max(avg_velocity - std_dev, 1))

        return {
            "average_velocity": round(avg_velocity, 1),
            "remaining_points": remaining_points,
            "sprints_needed": {
                "optimistic": optimistic,
                "expected": sprints_needed,
                "pessimistic": pessimistic,
            },
        }

    def progress_report(self) -> dict:
        """Progress Report"""
        total = len(self.tasks)
        done = sum(1 for t in self.tasks.values() if t.status == TaskStatus.DONE)
        in_progress = sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS)
        blocked = sum(1 for t in self.tasks.values() if t.status == TaskStatus.BLOCKED)

        estimated_total = sum(t.estimated_days for t in self.tasks.values())
        actual_total = sum(
            t.actual_days for t in self.tasks.values()
            if t.actual_days is not None
        )
        done_estimated = sum(
            t.estimated_days for t in self.tasks.values()
            if t.status == TaskStatus.DONE
        )

        return {
            "total_tasks": total,
            "done": done,
            "in_progress": in_progress,
            "blocked": blocked,
            "completion_rate": f"{done / total * 100:.1f}%" if total > 0 else "N/A",
            "estimated_total_days": round(estimated_total, 1),
            "actual_total_days": round(actual_total, 1),
            "estimation_accuracy": (
                f"{actual_total / done_estimated * 100:.1f}%"
                if done_estimated > 0 else "N/A"
            ),
        }


# Usage example
planner = ProjectPlanner("E-Commerce Site Renewal")

planner.add_task(Task(
    id="T-001", name="Authentication System",
    estimate=ThreePointEstimate(3, 5, 10),
    status=TaskStatus.DONE, actual_days=6
))
planner.add_task(Task(
    id="T-002", name="Product Search API",
    estimate=ThreePointEstimate(5, 8, 15),
    status=TaskStatus.IN_PROGRESS, assignee="Suzuki"
))
planner.add_task(Task(
    id="T-003", name="Payment Integration",
    estimate=ThreePointEstimate(8, 12, 20),
    status=TaskStatus.TODO, dependencies=["T-001"]
))

# Total estimate
total = planner.total_estimate()
print(f"Total expected value: {total.expected:.1f} days")
lower, upper = total.confidence_interval(0.95)
print(f"95% confidence interval: {lower:.1f} to {upper:.1f} days")

# Monte Carlo simulation
mc = planner.monte_carlo_simulation()
print(f"P85 completion forecast: {mc['p85']:.1f} days")

# Progress report
report = planner.progress_report()
print(f"Progress: {report['completion_rate']}")
```

### 6.2 Risk Management

Software projects entail many risks. A process for identifying, assessing, and mitigating risks is essential.

**Risk Categories:**

1. **Technical Risk**: Unknown technologies, performance issues, integration complexity
2. **Schedule Risk**: Estimation errors, dependency delays
3. **Personnel Risk**: Loss of key members, skill shortages
4. **Requirements Risk**: Unclear requirements, frequent changes
5. **External Risk**: Vendor dependency, regulatory changes, market changes

**Risk Response Strategies:**

- **Avoidance**: Eliminate the cause of the risk
- **Mitigation**: Reduce the probability or impact of the risk
- **Transfer**: Transfer the risk to a third party (insurance, outsourcing)
- **Acceptance**: Acknowledge the risk and choose not to address it (prepare a contingency plan)

### 6.3 Technical Debt Management

Technical Debt is a concept proposed by Ward Cunningham in 1992 that refers to the future costs incurred by making technically suboptimal choices for short-term benefit.

**Types of Technical Debt:**

| Type | Intentional/Unintentional | Example |
|------|--------------------------|---------|
| Prudent and Intentional | Intentional | "Prioritize release, refactor later" |
| Reckless and Intentional | Intentional | "No time for design, just push forward" |
| Prudent and Unintentional | Unintentional | "Now I know a better way" (realization through learning) |
| Reckless and Unintentional | Unintentional | "What is layering?" (lack of knowledge) |

For managing technical debt, continuous measurement of code metrics (cyclomatic complexity, code duplication rate, test coverage) and allocating a fixed percentage (recommended: 15-20%) of each sprint to paying down technical debt are effective.

---

## 7. Branch Strategies

### 7.1 Major Branch Strategies

Git branch strategies in team development have a significant impact on code quality and development efficiency.

#### Git Flow

A branch model proposed by Vincent Driessen in 2010. It uses branches with clearly defined roles.

- **main**: Code that has been released to production
- **develop**: Development branch for the next release
- **feature/\***: Feature development branches (branch from develop, merge into develop)
- **release/\***: Release preparation branches (branch from develop, merge into main and develop)
- **hotfix/\***: Emergency fix branches (branch from main, merge into main and develop)

Applicable situations: Products with clear release cycles, cases requiring parallel maintenance of multiple versions.

#### GitHub Flow

A simple branch model proposed by GitHub. Only uses main branch and feature branches.

1. Create a feature branch from main
2. Add commits
3. Create a pull request
4. Review and discussion
5. Deploy and verify
6. Merge into main

Applicable situations: Web applications with continuous deployment, small teams.

#### Trunk-Based Development

A methodology where all developers commit directly to main (trunk), or use very short-lived branches (within 1 day). Used in combination with feature flags.

Applicable situations: Environments with highly automated CI/CD, cases requiring high-frequency deployment.

### 7.2 Branch Strategy Comparison

| Aspect | Git Flow | GitHub Flow | Trunk-Based Development |
|--------|----------|-------------|------------------------|
| Complexity | High | Low | Lowest |
| Release Frequency | Low to medium | High | Highest |
| Branch Lifespan | Long | Medium | Very short |
| Team Size | Large-scale | Small to medium | Any size |
| Prerequisites | Release management structure | CI/CD | Advanced automated testing + feature flags |
| Merge Conflicts | Many | Moderate | Few |

---

## 8. Anti-Patterns

### 8.1 Anti-Pattern 1: Cargo Cult Agile

**Problem**: Adopting only the forms of agile without understanding its essence. Typical cases include running daily scrums as "progress report meetings" and sprint reviews as "management report sessions."

**Symptoms:**

- All Scrum events are held, but the team is not self-organizing
- The Product Owner has no real decision-making authority
- Improvement actions from Sprint Retrospectives are never executed
- The misconception that "agile means no planning is needed"
- Daily Scrum takes more than 30 minutes
- Velocity has become a reporting number rather than an improvement metric

**Countermeasures:**

- Return to the 4 values and 12 principles of agile
- Ensure the Scrum Master functions as a "servant leader" not a "process guardian"
- Foster a culture where the team proposes and executes improvement measures themselves
- Invite an external agile coach to objectively evaluate the team's state
- Measure process health with quantitative indicators (velocity stability, sprint goal achievement rate)

```
Identifying Cargo Cult Agile:

  Authentic Agile                    Cargo Cult
  ---------------------------------------------------
  Daily Scrum:                       Daily Scrum:
  "Today I plan to pair with         "Yesterday I had meetings.
   Suzuki to fix the auth bug"        Today I'll develop.
                                      No particular issues."

  -> Purpose is collaboration         -> Just a reporting obligation
     and impediment removal

  Retrospective:                     Retrospective:
  "Test execution time is long.       "No particular issues"
   Let's try parallelization          (everyone silent)
   next sprint"

  -> Concrete improvement actions     -> Ceremonial going through the motions
```

### 8.2 Anti-Pattern 2: Big Bang Integration

**Problem**: Each team or developer works independently for an extended period and attempts to integrate all code at once just before release.

**Symptoms:**

- Branches have been diverged from main for weeks to months
- Massive conflicts occur during merging
- Integration testing is not performed until the final stage
- An "integration phase" is scheduled
- Unexpected bugs are discovered in large numbers just before release

**Countermeasures:**

- Introduce Continuous Integration (CI) and integrate code at least daily
- Keep feature branch lifespans short (target within 2-3 days)
- Use feature flags to safely integrate incomplete features into main
- Enhance automated testing to ensure integration safety
- Consider migrating to trunk-based development

### 8.3 Anti-Pattern 3: Golden Hammer

**Problem**: Trying to apply a specific technology or process to every problem. Thinking patterns such as "every project should use Scrum" or "microservices solve everything."

**Countermeasures:**

- Analyze project characteristics (scale, complexity, team, regulations) before selecting a process
- Understand the advantages and limitations of multiple approaches
- Start with small experiments, verify effectiveness, then proceed to full-scale adoption

---

## 9. Exercises

### 9.1 Basic Exercises (Comprehension Check)

**Exercise 1**: For the following project characteristics, select the optimal development process model and explain the rationale.

(a) A government agency tax calculation system. Requirements are strictly defined by law, and changes occur only during annual law revisions. Audit trails are required.

(b) A startup's new SNS app. They want to add and change features based on user feedback. The team has 5 members.

(c) A factory control system. Safety is the top priority, and 24/7/365 operation is required.

**Model Answer Direction:**

(a) Waterfall (or V-Model) is appropriate. Reasons: requirements are clear and stable, traceability is required in a regulated industry, change frequency is low, detailed documentation is mandatory for audit trails.

(b) Scrum is appropriate. Reasons: requirements are uncertain with many feedback-driven changes, small team, want to release in short cycles to gauge market reaction, gradual growth strategy from MVP (Minimum Viable Product).

(c) Waterfall + safety analysis methods (FMEA, FTA) are appropriate. Reasons: safety is paramount and thorough verification is needed, planned testing and staged quality assurance are essential. However, partially adopting an iterative approach for development efficiency can also be effective.

### 9.2 Applied Exercises (Design and Implementation)

**Exercise 2**: Design a Scrum board for a project with the following requirements and plan the first 2 sprints.

- Project: Internal task management tool
- Team: 4 developers, 1 PO, 1 SM
- Sprint duration: 2 weeks
- Past velocity: 25, 30, 28 story points/sprint

Functional requirements:
1. User authentication (login/logout): 5 points
2. Task create/edit/delete: 8 points
3. Task status management (TODO/In Progress/Done): 5 points
4. Task assignee assignment: 3 points
5. Dashboard display: 8 points
6. Deadline setting and reminders: 5 points
7. Task commenting feature: 5 points
8. Search and filtering: 8 points
9. Email notifications: 5 points
10. Team management: 5 points

Average velocity: (25 + 30 + 28) / 3 = approximately 28 points

Sprint 1 (28-point target): 1, 2, 3, 4, 6 = 26 points (prioritize foundational features)
Sprint 2 (28-point target): 5, 7, 8, 9 = 26 points (UX improvement features)
Sprint 3: 10 + technical debt repayment + bug fixes

### 9.3 Advanced Exercises (Practice and Application)

**Exercise 3**: Analyze the development process of your current team (or an imagined team) and create the following.

1. Create a Value Stream Map of the current process
   - List all steps from code commit to production deployment
   - Record the duration and wait time for each step
   - Identify bottlenecks

2. Measure (or estimate) the current DORA metrics values
   - Deployment frequency
   - Lead time
   - Change failure rate
   - Recovery time

3. Develop an improvement plan
   - Propose 3 specific improvement measures for the biggest bottleneck
   - Estimate the cost (time and resources) and expected effect of each improvement measure
   - Determine priorities and create a plan to implement the first one within 2 weeks

---

## 10. FAQ (Frequently Asked Questions)

### Q1: Should I adopt agile or waterfall?

**A**: This is not an "either/or" binary choice; rather, it is important to select based on the characteristics of the project. Waterfall is still effective in regulated industries where requirements are clear and changes are few, while agile is suitable when requirements are highly uncertain and feedback is valued.

In practice, many organizations adopt a "hybrid approach" combining elements of both. For example, overall planning is done in a waterfall style, while iterative development is used within each phase. The key is finding the most effective process for your team and project and continuously improving it.

### Q2: We introduced Scrum but it's not working. What's the problem?

**A**: Typical causes for Scrum introduction failure include:

1. **Lack of management support**: Agile involves organizational culture change. Without management understanding and support, it becomes hollow
2. **Insufficient Product Owner authority**: If the PO lacks authority to decide priorities, interference enters during sprints
3. **Scrum Master not dedicated**: If the SM also serves as a manager, they cannot function as a servant leader
4. **Not allowing team self-organization**: Scrum cannot function in environments where micromanagement prevails
5. **Lack of technical practices**: Without test automation and CI/CD, releasing in short cycles becomes difficult

It is recommended to start with a small team (3-5 people) as a pilot and mature the team over 3-4 sprints (6-8 weeks) before considering organization-wide rollout.

### Q3: What should be done first to introduce DevOps?

**A**: DevOps adoption should be approached incrementally. The recommended steps are:

**Step 1 (1-2 weeks): Understand the Current State**
- Visualize the deployment process (create a Value Stream Map)
- Measure current DORA metrics values
- Identify the team's challenges and pain points

**Step 2 (2-4 weeks): Introduce CI**
- Unify version control (Git)
- Set up automated builds
- Introduce automated testing (start with unit tests)
- Establish a code review process

**Step 3 (1-2 months): Introduce CD**
- Set up a staging environment
- Automate deployment
- Introduce Infrastructure as Code

**Step 4 (Continuous): Foster Culture**
- Conduct postmortems (incident retrospectives)
- Establish improvement cycles based on metrics
- Build a collaboration framework between development and operations teams

Most importantly, start with "culture" and "measurement," not "automation." Automating without understanding what needs improvement yields limited results.

### Q4: How much technical debt should be tolerated?

**A**: It is neither realistic nor desirable to eliminate technical debt entirely. What matters is "managing it consciously."

The following criteria can help guide decisions:

- **Debt to repay immediately**: Items with security risks, items that could cause system outages
- **Debt to repay in a planned manner**: Items slowing down development speed, items blocking new feature additions
- **Acceptable debt**: Items with limited scope of impact and low cost

Allocating 15-20% of each sprint to paying down technical debt is a generally recommended guideline. Additionally, quantitatively measuring technical debt with static analysis tools like SonarQube and monitoring trends is effective.

### Q5: Is agile development possible with remote teams?

**A**: Yes. Agile development with remote teams has rapidly spread since the COVID-19 pandemic, with many organizations achieving success.

However, the following practices are necessary:

1. **Leverage asynchronous communication**: Updates via Slack/Teams, thorough documentation
2. **Tool readiness**: Digital Kanban boards (Jira, Linear), virtual whiteboards (Miro, FigJam)
3. **Time zone considerations**: Concentrate meetings during overlapping time slots
4. **Intentional casual interaction opportunities**: Virtual coffee breaks, team-building events
5. **Documented Working Agreements**: Agreed-upon response time expectations, core hours, meeting rules

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

The knowledge from this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 11. Summary

### Key Concept Overview

| Concept | Key Points |
|---------|-----------|
| Waterfall | Sequential execution model. Suited for regulated industries with stable requirements |
| V-Model | Extension of waterfall. Explicitly maps correspondence between test phases |
| Agile | Iterative and incremental development. Emphasizes adaptation to change |
| Scrum | Most widely adopted agile framework. Clear roles, events, and artifacts |
| XP | Emphasis on engineering practices. Characterized by TDD and pair programming |
| Kanban | Flow-oriented. Visualizes bottlenecks through WIP limits |
| DevOps | Integration of development and operations. CALMS framework |
| CI/CD | Continuous Integration/Delivery. Automation as the foundation |
| DORA Metrics | 4 indicators of DevOps performance |
| Technical Debt | Technical compromises for short-term gains. Manage systematically |
| Branch Strategy | Git Flow / GitHub Flow / Trunk-Based Development |

### Learning Roadmap

1. **Beginner**: Read the Agile Manifesto -> Read the Scrum Guide
2. **Foundation**: Practice Scrum in a small project -> Build a CI/CD pipeline
3. **Applied**: Measure and improve DORA metrics -> Build a technical debt management framework
4. **Advanced**: Organization-wide agile adoption -> Platform Engineering practice

---

## 12. Advanced Topics

### 12.1 Scaling Agile

Once single-team agile practice succeeds, the challenge of scaling to the entire organization arises. When multiple Scrum teams develop the same product, inter-team coordination, dependency management, and ensuring architectural consistency become necessary.

#### SAFe (Scaled Agile Framework)

SAFe is the most widely adopted scaling framework for large organizations. It consists of four tiers: Team Level, Program Level, Large Solution Level, and Portfolio Level.

```
SAFe Hierarchy:

  +---------------------------------------------------+
  |               Portfolio Level                       |
  |  Strategic Themes -> Portfolio Backlog -> Epics     |
  +--------------------------+-------------------------+
                             v
  +---------------------------------------------------+
  |            Large Solution Level                     |
  |  Solution Train -> Integration of multiple ARTs     |
  +--------------------------+-------------------------+
                             v
  +---------------------------------------------------+
  |              Program Level (ART)                    |
  |  Agile Release Train = 5-12 teams synchronized     |
  |  PI Planning (8-12 week plans)                      |
  |  System Demo (every 2 weeks)                        |
  +--------------------------+-------------------------+
                             v
  +---------------------------------------------------+
  |               Team Level                            |
  |  Each team develops using Scrum or Kanban           |
  |  Sprint = 2 weeks                                   |
  +---------------------------------------------------+
```

#### LeSS (Large-Scale Scrum)

LeSS is a scaling framework proposed by Craig Larman and Bas Vodde that aims to scale while keeping Scrum as simple as possible. It offers Basic LeSS for up to 8 teams and LeSS Huge for larger scales.

The characteristic of LeSS lies in its "Less is More" philosophy, minimizing additional framework elements. All teams share a single Product Backlog managed by one Product Owner.

#### Spotify Model

A model adopted by Spotify to structure their development organization. Rather than a strict framework, it is referenced as an organizational design pattern.

- **Squad**: An autonomous small team (equivalent to a Scrum team)
- **Tribe**: A collection of related Squads (40-150 people)
- **Chapter**: A cross-cutting group of members with the same expertise
- **Guild**: A voluntary community of members with shared interests

### 12.2 Feature Flags

Feature Flags (also called Feature Toggles) are a technique for dynamically enabling or disabling features after code has been deployed. They are an essential foundation technology for trunk-based development and canary releases.

```python
# Code Example 6: Feature flag implementation patterns
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hashlib
import time


class FlagType(Enum):
    """Types of Feature Flags"""
    RELEASE = "release"          # Release flag (gradual rollout of new features)
    EXPERIMENT = "experiment"    # Experiment flag (A/B testing)
    OPS = "ops"                  # Ops flag (emergency feature shutdown)
    PERMISSION = "permission"    # Permission flag (release to specific users)


@dataclass
class FeatureFlag:
    """Feature Flag"""
    name: str
    flag_type: FlagType
    enabled: bool = False
    description: str = ""
    rollout_percentage: int = 0        # Gradual rollout percentage (0-100)
    allowed_users: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # Expiration time

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class FeatureFlagManager:
    """Feature Flag Manager"""

    def __init__(self):
        self._flags: dict[str, FeatureFlag] = {}

    def register(self, flag: FeatureFlag) -> None:
        self._flags[flag.name] = flag

    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """Determine if the specified flag is enabled"""
        flag = self._flags.get(flag_name)
        if flag is None:
            return False

        # Expiration check
        if flag.is_expired():
            return False

        # Global disable
        if not flag.enabled:
            return False

        # User-specific determination
        if user_id:
            # Users in the allow list are always enabled
            if user_id in flag.allowed_users:
                return True

            # Gradual rollout (determined by user ID hash)
            if flag.rollout_percentage < 100:
                hash_val = int(hashlib.md5(
                    f"{flag_name}:{user_id}".encode()
                ).hexdigest(), 16)
                return (hash_val % 100) < flag.rollout_percentage

        return True

    def get_all_flags(self) -> dict[str, dict]:
        """Return the state of all flags"""
        return {
            name: {
                "enabled": flag.enabled,
                "type": flag.flag_type.value,
                "rollout": flag.rollout_percentage,
                "expired": flag.is_expired(),
            }
            for name, flag in self._flags.items()
        }

    def cleanup_expired(self) -> list[str]:
        """Remove expired flags"""
        expired = [name for name, f in self._flags.items() if f.is_expired()]
        for name in expired:
            del self._flags[name]
        return expired


# Usage example
manager = FeatureFlagManager()

# Gradual rollout of a new feature
manager.register(FeatureFlag(
    name="new_search_ui",
    flag_type=FlagType.RELEASE,
    enabled=True,
    description="Gradually roll out the new search UI",
    rollout_percentage=20,  # Initially roll out to 20% of users
))

# Emergency shutdown flag
manager.register(FeatureFlag(
    name="payment_processing",
    flag_type=FlagType.OPS,
    enabled=True,
    description="Toggle payment processing on/off",
    rollout_percentage=100,
))

# Usage in application code
def search_handler(user_id: str, query: str):
    if manager.is_enabled("new_search_ui", user_id):
        return new_search(query)  # New search logic
    else:
        return legacy_search(query)  # Existing search logic

def new_search(query: str) -> dict:
    return {"engine": "new", "query": query}

def legacy_search(query: str) -> dict:
    return {"engine": "legacy", "query": query}

# Different results per user
print(manager.is_enabled("new_search_ui", "user_001"))  # True/False depending on hash
print(manager.is_enabled("new_search_ui", "user_002"))
```

### 12.3 Deployment Strategies

Deployment methods to production environments directly affect the magnitude of risk and the ease of rollback.

```
Comparison of Deployment Strategies:

  * Big Bang Deploy (Not Recommended)
    Old version: [============]
    New version:              [============]  <- All-at-once switch
    Risk: Maximum. Affects all users

  * Rolling Update
    Server 1: [OldOldOld][NewNewNewNewNew]
    Server 2: [OldOldOldOld][NewNewNewNew]
    Server 3: [OldOldOldOldOld][NewNewNew]
    -> Servers updated one at a time sequentially

  * Blue/Green Deploy
    Blue (current): [============]--+
    Green (new):    [============]  | <- Routing switch
    Router:         [Blue-------->Green------->]
    -> Two environments prepared, switch via routing

  * Canary Release
    Old version: [===================]  95%
    New version: [==]                    5% <- Validate with a small set of users
    -> If no issues, gradually increase the ratio

  * A/B Test Deploy
    Version A: [============]  50%  <- Control group
    Version B: [============]  50%  <- Test group
    -> Compare metrics and select the winner
```

| Strategy | Risk | Rollback Speed | Infrastructure Cost | Use Case |
|----------|------|----------------|--------------------|---------|
| Big Bang | High | Slow | Low | Recommended for test environments only |
| Rolling | Medium | Medium | Low | Kubernetes standard |
| Blue/Green | Low | Instant | High (2x) | Mission-critical |
| Canary | Low | Instant | Medium | Large-scale web services |
| A/B Test | Low | Instant | Medium | UX improvement validation |

### 12.4 Postmortem Culture

As an important cultural aspect of DevOps, postmortems (post-incident reviews) are conducted after incidents. The "Blameless Postmortem" established by Google's SRE team treats incidents as learning opportunities, focusing on system improvements rather than blaming individuals.

**Items to Include in a Postmortem:**

1. **Incident Summary**: Date/time of occurrence, scope of impact, duration of impact
2. **Timeline**: Chronological sequence from incident detection to resolution
3. **Root Cause Analysis**: Cause analysis using 5 Whys or fishbone diagrams
4. **Response Actions**: Mitigation and fix measures taken
5. **Impact**: Quantitative assessment of user impact and business impact
6. **Lessons Learned**: What went well, what did not go well
7. **Action Items**: Specific improvement measures for prevention (with owners and deadlines)

### 12.5 Platform Engineering

Platform Engineering is an approach that emerged in the 2020s and can be considered an evolution of DevOps. It establishes a specialized team (platform team) that builds and operates a self-service Internal Developer Platform (IDP) to reduce cognitive load on developers.

**Problems Platform Engineering Solves:**

- Cognitive overload from DevOps' "everyone knows everything" approach
- Fragmentation of toolchains and lack of standardization
- Duplicate work from each team independently building infrastructure
- Difficulty in consistently applying security and compliance

**Typical Capabilities Provided by an IDP:**

- Self-service infrastructure provisioning
- Standardized CI/CD templates
- Common monitoring and logging infrastructure
- Automated security scanning
- Documentation and service catalogs

### 12.6 Metrics-Driven Improvement

Development process improvements should be based on quantitative metrics. In addition to DORA metrics, the following metrics are useful.

**Process Metrics:**

- **Cycle Time**: Time from work start to completion
- **Lead Time**: Time from request origination to customer delivery
- **Throughput**: Number of completed items per unit of time
- **WIP (Work In Progress) Count**: Number of items concurrently in progress

**Quality Metrics:**

- **Defect Density**: Number of bugs per 1,000 lines of code
- **Test Coverage**: Percentage of code covered by tests
- **Technical Debt Ratio**: Time needed to resolve technical debt / new feature development time
- **Escape Rate**: Percentage of bugs that reach production

**Team Metrics:**

- **Velocity**: Completed story points per sprint
- **Sprint Goal Achievement Rate**: Percentage of sprints that achieved their goal
- **Planning Accuracy**: Deviation rate between estimates and actuals

These metrics should be used for team improvement and should not be used for individual performance evaluation. When metrics are directly tied to evaluation, incentives arise to manipulate the numbers, undermining the original purpose (process improvement).

---

## 13. Real-World Case Studies

### 13.1 Case: Migration from Waterfall to Agile

Consider a case where a financial institution's IT department (200 developers) migrated from waterfall to agile.

**Background and Challenges:**

- 6-month release cycles could not keep up with market changes
- Requirement changes after confirmation caused major rework
- Large numbers of bugs discovered in the testing phase, causing constant release delays
- A deep divide between development and operations teams

**Migration Approach:**

1. **Pilot Team Selection (Month 1)**: Selected 2 teams (7 members each) centered on volunteers. Piloted on a relatively low-risk internal tool project
2. **Scrum Training and Introduction (Months 2-3)**: Hired an external agile coach. Appointed PO and SM, started development in 2-week sprints
3. **CI/CD Infrastructure Build (Months 3-4)**: Migrated from Jenkins to GitHub Actions. Improved test automation rate from 30% to 70%
4. **Gradual Expansion (Months 5-12)**: Shared pilot success stories internally. 4 additional teams migrated. Established a migration support team (CoE: Center of Excellence)
5. **Organization-wide Adoption (Months 13-24)**: Remaining teams migrated gradually. Introduced SAFe PI Planning at the portfolio level

**Results (After 2 Years):**

- Release cycle: 6 months -> 2 weeks
- Production bugs: 40% decrease
- Customer satisfaction: 25% increase
- Developer satisfaction: 30% increase

**Lessons Learned:**

- Strong management commitment is essential
- Gradual adoption, not all-at-once migration, is the key to success
- Technical automation infrastructure must be prepared in advance
- Cultural transformation takes time (at least 1-2 years)
- The presence of an agile coach makes a significant difference in the early stages

### 13.2 Case: Improving Deployment Frequency Through DevOps Adoption

Consider a case where an e-commerce company adopted DevOps and improved deployment frequency from monthly to daily.

**Initial State:**

- Monthly deployment, manual late-night work (4 hours)
- Testing was primarily manual (regression testing took 3 days)
- Large discrepancies between development and production environments
- Average incident recovery time of 8 hours

**Improvement Measures:**

1. **Containerization**: Containerized applications with Docker to eliminate environment discrepancies
2. **Test Automation**: Automated E2E tests with Playwright (manual 3 days -> automated 30 minutes)
3. **CI Pipeline Build**: Automated test execution per pull request
4. **CD Pipeline Build**: Automated deployment on merge to main branch
5. **Canary Release Introduction**: Shifted from all-at-once releases to gradual releases
6. **Monitoring Enhancement**: Real-time monitoring and automatic alerting with Datadog

**Results:**

| Metric | Before | After | Improvement |
|--------|--------|-------|------------|
| Deployment Frequency | Monthly | Daily (10+ times/day) | 300x+ |
| Lead Time | 2 months | 1 hour | 99% reduction |
| Change Failure Rate | 20% | 3% | 85% reduction |
| Recovery Time | 8 hours | 15 minutes | 97% reduction |

---

## 14. Toolchain Guide

### 14.1 Tools Supporting the Development Process

Let's organize the major tools used at each stage of the development process.

**Project Management:**

| Tool | Features | Use Case |
|------|----------|----------|
| Jira | Feature-rich, highly customizable | Large teams, SAFe |
| Linear | Fast, developer-focused UX | Startups, small to medium scale |
| GitHub Projects | GitHub integration | OSS, GitHub-centric development |
| Notion | Flexible document management | Documentation-focused teams |
| Asana | Intuitive UI | Collaboration with non-technical teams |

**CI/CD:**

| Tool | Features | Use Case |
|------|----------|----------|
| GitHub Actions | GitHub integration, marketplace | GitHub users |
| GitLab CI/CD | GitLab integration, self-hostable | GitLab users |
| CircleCI | Fast, Docker affinity | Container-based development |
| Jenkins | Highly customizable | Legacy environments, special requirements |
| ArgoCD | GitOps native | Kubernetes environments |

**Communication:**

| Tool | Features | Use Case |
|------|----------|----------|
| Slack | Rich integrations | Tech companies |
| Microsoft Teams | Microsoft 365 integration | Enterprise |
| Discord | Rich voice chat | Game development, communities |

---

## Recommended Next Guides


---

## References

1. Beck, K. et al. "Manifesto for Agile Software Development." 2001. https://agilemanifesto.org/
2. Schwaber, K. & Sutherland, J. "The Scrum Guide." 2020. https://scrumguides.org/
3. Humble, J. & Farley, D. "Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation." Addison-Wesley, 2010.
4. Kim, G. et al. "The DevOps Handbook: How to Create World-Class Agility, Reliability, & Security in Technology Organizations." IT Revolution Press, 2016.
5. Forsgren, N., Humble, J. & Kim, G. "Accelerate: The Science of Lean Software and DevOps." IT Revolution Press, 2018.
6. Beck, K. "Extreme Programming Explained: Embrace Change." Addison-Wesley, 2nd Edition, 2004.
7. Royce, W. "Managing the Development of Large Software Systems." Proceedings of IEEE WESCON, 1970.
8. Anderson, D. "Kanban: Successful Evolutionary Change for Your Technology Business." Blue Hole Press, 2010.
9. Patton, J. "User Story Mapping: Discover the Whole Story, Build the Right Product." O'Reilly Media, 2014.
