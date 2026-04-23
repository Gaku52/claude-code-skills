# Creational Patterns

> Design patterns that provide flexibility in how objects are created. A practical guide to the five major patterns—Factory, Abstract Factory, Builder, Singleton, and Prototype—explaining "why you need them" and "when to use them."

## What You Will Learn in This Chapter

- [ ] Understand the purpose of each creational pattern and when to use which
- [ ] Grasp the implementation of each pattern across multiple languages
- [ ] Learn the problems with Singleton as an anti-pattern
- [ ] Know how creational patterns are applied in modern frameworks
- [ ] Be able to design creational patterns that account for testability


## Prerequisite Knowledge

Your understanding will be deeper if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. Factory Method Pattern

### 1.1 Overview and Purpose

```
Purpose: Encapsulate the object creation logic

When to use:
  -> You want to decide the class to create at runtime
  -> The creation logic is complex
  -> You don't want to let users use new directly
  -> You want to swap in mock objects during testing

Structure:
  +----------------+         +-----------------+
  |    Creator     |         |    Product      |
  | (factory)      |-------->| (created item)  |
  +----------------+         +-----------------+
  | factoryMethod()|         | operation()     |
  +-------+--------+         +--------+--------+
          |                           |
  +-------+--------+         +--------+--------+
  |ConcreteCreator |         |ConcreteProduct  |
  |                |-------->|                 |
  +----------------+         +-----------------+
```

### 1.2 Simple Factory (Static Factory Method)

```typescript
// Simple Factory: the most basic form of factory
interface Notification {
  send(message: string): void;
  getChannel(): string;
}

class EmailNotification implements Notification {
  constructor(private to: string) {}
  send(message: string) {
    console.log(`Email to ${this.to}: ${message}`);
  }
  getChannel() { return "email"; }
}

class SmsNotification implements Notification {
  constructor(private phoneNumber: string) {}
  send(message: string) {
    console.log(`SMS to ${this.phoneNumber}: ${message}`);
  }
  getChannel() { return "sms"; }
}

class SlackNotification implements Notification {
  constructor(private channel: string) {}
  send(message: string) {
    console.log(`Slack #${this.channel}: ${message}`);
  }
  getChannel() { return "slack"; }
}

class PushNotification implements Notification {
  constructor(private deviceToken: string) {}
  send(message: string) {
    console.log(`Push to ${this.deviceToken}: ${message}`);
  }
  getChannel() { return "push"; }
}

// Factory: centralizes the creation logic
interface NotificationConfig {
  type: "email" | "sms" | "slack" | "push";
  target: string;
}

class NotificationFactory {
  static create(config: NotificationConfig): Notification {
    switch (config.type) {
      case "email":
        return new EmailNotification(config.target);
      case "sms":
        return new SmsNotification(config.target);
      case "slack":
        return new SlackNotification(config.target);
      case "push":
        return new PushNotification(config.target);
      default:
        throw new Error(`Unknown notification type: ${config.type}`);
    }
  }

  // Create multiple notifications in a batch
  static createBatch(configs: NotificationConfig[]): Notification[] {
    return configs.map(config => this.create(config));
  }
}

// Callers do not need to know the concrete classes
const notification = NotificationFactory.create({
  type: "email",
  target: "tanaka@example.com",
});
notification.send("Hello!");

// Batch send to multiple channels
const notifications = NotificationFactory.createBatch([
  { type: "email", target: "tanaka@example.com" },
  { type: "slack", target: "general" },
  { type: "push", target: "device-token-123" },
]);
notifications.forEach(n => n.send("Important announcement"));
```

### 1.3 Factory Method (Combined with Template Method)

```typescript
// Factory Method: the subclass determines the type of object to create
abstract class DocumentExporter {
  // Template method
  async export(data: ReportData): Promise<Buffer> {
    const formatter = this.createFormatter();
    const header = formatter.formatHeader(data.title);
    const body = formatter.formatBody(data.content);
    const footer = formatter.formatFooter(data.metadata);
    return Buffer.from(header + body + footer);
  }

  // Factory Method: implemented by subclasses
  protected abstract createFormatter(): DocumentFormatter;
}

interface DocumentFormatter {
  formatHeader(title: string): string;
  formatBody(content: string[]): string;
  formatFooter(metadata: Record<string, string>): string;
}

class PdfExporter extends DocumentExporter {
  protected createFormatter(): DocumentFormatter {
    return new PdfFormatter();
  }
}

class HtmlExporter extends DocumentExporter {
  protected createFormatter(): DocumentFormatter {
    return new HtmlFormatter();
  }
}

class MarkdownExporter extends DocumentExporter {
  protected createFormatter(): DocumentFormatter {
    return new MarkdownFormatter();
  }
}

class CsvExporter extends DocumentExporter {
  protected createFormatter(): DocumentFormatter {
    return new CsvFormatter();
  }
}

// Implementations of each formatter
class PdfFormatter implements DocumentFormatter {
  formatHeader(title: string): string {
    return `%PDF-1.4\n/Title (${title})\n`;
  }
  formatBody(content: string[]): string {
    return content.map(line => `  ${line}`).join('\n');
  }
  formatFooter(metadata: Record<string, string>): string {
    return `\n/Author (${metadata.author ?? "Unknown"})`;
  }
}

class HtmlFormatter implements DocumentFormatter {
  formatHeader(title: string): string {
    return `<!DOCTYPE html>\n<html><head><title>${title}</title></head><body>\n`;
  }
  formatBody(content: string[]): string {
    return content.map(line => `<p>${line}</p>`).join('\n');
  }
  formatFooter(metadata: Record<string, string>): string {
    return `\n<footer>Author: ${metadata.author ?? "Unknown"}</footer></body></html>`;
  }
}

class MarkdownFormatter implements DocumentFormatter {
  formatHeader(title: string): string {
    return `# ${title}\n\n`;
  }
  formatBody(content: string[]): string {
    return content.join('\n\n');
  }
  formatFooter(metadata: Record<string, string>): string {
    return `\n\n---\n*Author: ${metadata.author ?? "Unknown"}*`;
  }
}

class CsvFormatter implements DocumentFormatter {
  formatHeader(title: string): string {
    return `"Title","${title}"\n`;
  }
  formatBody(content: string[]): string {
    return content.map((line, i) => `"${i + 1}","${line}"`).join('\n');
  }
  formatFooter(metadata: Record<string, string>): string {
    return `\n"Author","${metadata.author ?? "Unknown"}"`;
  }
}

// Example usage: select export format dynamically
function getExporter(format: string): DocumentExporter {
  switch (format) {
    case "pdf": return new PdfExporter();
    case "html": return new HtmlExporter();
    case "markdown": return new MarkdownExporter();
    case "csv": return new CsvExporter();
    default: throw new Error(`Unsupported format: ${format}`);
  }
}
```

### 1.4 Factory Method in Java

```java
// Java: Factory Method pattern implementation
public interface Logger {
    void log(String level, String message);
    void close();
}

public class ConsoleLogger implements Logger {
    @Override
    public void log(String level, String message) {
        System.out.printf("[%s] %s: %s%n",
            LocalDateTime.now(), level, message);
    }

    @Override
    public void close() {
        // Console does not need to be closed
    }
}

public class FileLogger implements Logger {
    private final PrintWriter writer;

    public FileLogger(String filePath) throws IOException {
        this.writer = new PrintWriter(
            new FileWriter(filePath, true), true);
    }

    @Override
    public void log(String level, String message) {
        writer.printf("[%s] %s: %s%n",
            LocalDateTime.now(), level, message);
    }

    @Override
    public void close() {
        writer.close();
    }
}

public class DatabaseLogger implements Logger {
    private final Connection connection;

    public DatabaseLogger(String jdbcUrl) throws SQLException {
        this.connection = DriverManager.getConnection(jdbcUrl);
    }

    @Override
    public void log(String level, String message) {
        try (PreparedStatement stmt = connection.prepareStatement(
                "INSERT INTO logs (level, message, created_at) VALUES (?, ?, ?)")) {
            stmt.setString(1, level);
            stmt.setString(2, message);
            stmt.setTimestamp(3, Timestamp.valueOf(LocalDateTime.now()));
            stmt.executeUpdate();
        } catch (SQLException e) {
            System.err.println("Failed to log: " + e.getMessage());
        }
    }

    @Override
    public void close() {
        try { connection.close(); } catch (SQLException ignored) {}
    }
}

// Factory class
public class LoggerFactory {
    public static Logger createLogger(String type) {
        return createLogger(type, Map.of());
    }

    public static Logger createLogger(String type, Map<String, String> config) {
        switch (type.toLowerCase()) {
            case "console":
                return new ConsoleLogger();
            case "file":
                String path = config.getOrDefault("path", "app.log");
                try {
                    return new FileLogger(path);
                } catch (IOException e) {
                    throw new RuntimeException("Failed to create file logger", e);
                }
            case "database":
                String url = config.getOrDefault("url", "jdbc:h2:mem:logs");
                try {
                    return new DatabaseLogger(url);
                } catch (SQLException e) {
                    throw new RuntimeException("Failed to create database logger", e);
                }
            default:
                throw new IllegalArgumentException("Unknown logger type: " + type);
        }
    }
}

// Example usage
Logger logger = LoggerFactory.createLogger("file",
    Map.of("path", "/var/log/myapp.log"));
logger.log("INFO", "Application started");
```

### 1.5 Registration-Based Factory (Extensible Factory)

```typescript
// Registration-based: new types can be added dynamically
type Creator<T> = (...args: any[]) => T;

class PluggableFactory<T> {
  private creators = new Map<string, Creator<T>>();

  register(type: string, creator: Creator<T>): void {
    if (this.creators.has(type)) {
      throw new Error(`Type "${type}" is already registered`);
    }
    this.creators.set(type, creator);
  }

  unregister(type: string): boolean {
    return this.creators.delete(type);
  }

  create(type: string, ...args: any[]): T {
    const creator = this.creators.get(type);
    if (!creator) {
      const available = [...this.creators.keys()].join(", ");
      throw new Error(
        `Unknown type: "${type}". Available: ${available}`
      );
    }
    return creator(...args);
  }

  getRegisteredTypes(): string[] {
    return [...this.creators.keys()];
  }

  has(type: string): boolean {
    return this.creators.has(type);
  }
}

// Example usage: pluggable notification system
interface NotificationPlugin {
  send(message: string, target: string): Promise<void>;
  getName(): string;
}

const notificationFactory = new PluggableFactory<NotificationPlugin>();

// Register core plugins
notificationFactory.register("email", (smtpConfig: SmtpConfig) => ({
  async send(message: string, target: string) {
    // SMTP send
    console.log(`Email to ${target}: ${message}`);
  },
  getName() { return "Email"; },
}));

notificationFactory.register("webhook", (url: string) => ({
  async send(message: string, target: string) {
    await fetch(url, {
      method: "POST",
      body: JSON.stringify({ message, target }),
    });
  },
  getName() { return "Webhook"; },
}));

// Third parties can add plugins
notificationFactory.register("teams", (webhookUrl: string) => ({
  async send(message: string, target: string) {
    await fetch(webhookUrl, {
      method: "POST",
      body: JSON.stringify({ text: `${target}: ${message}` }),
    });
  },
  getName() { return "Microsoft Teams"; },
}));

// Usage
const emailPlugin = notificationFactory.create("email", { host: "smtp.example.com" });
const teamsPlugin = notificationFactory.create("teams", "https://outlook.webhook.office.com/...");
```

---

## 2. Abstract Factory Pattern

### 2.1 Overview and Purpose

```
Purpose: Create groups of related objects without specifying their concrete classes

When to use:
  -> You want to create related groups of objects consistently
  -> You want to switch between product families (themes, platforms)
  -> You want to fully decouple clients from concrete classes

Structure:
  +-----------------+
  | AbstractFactory |
  | createProductA()|
  | createProductB()|
  +--------+--------+
           |
  +--------+--------+
  |                 |
  +---------+  +---------+
  |Factory1 |  |Factory2 |
  +---------+  +---------+
```

### 2.2 Switching UI Themes

```typescript
// Abstract Factory: creates a group of related objects
interface Button {
  render(): string;
  onClick(handler: () => void): void;
}

interface Input {
  render(): string;
  getValue(): string;
  setValue(value: string): void;
}

interface Modal {
  render(): string;
  open(): void;
  close(): void;
}

interface Card {
  render(): string;
  setContent(title: string, body: string): void;
}

interface UIFactory {
  createButton(label: string): Button;
  createInput(placeholder: string): Input;
  createModal(title: string): Modal;
  createCard(): Card;
}

// Material Design factory
class MaterialUIFactory implements UIFactory {
  createButton(label: string): Button {
    return {
      render: () => `<button class="mdc-button mdc-button--raised">${label}</button>`,
      onClick: (handler) => { /* Material Ripple + handler */ },
    };
  }

  createInput(placeholder: string): Input {
    let value = "";
    return {
      render: () => `
        <div class="mdc-text-field">
          <input class="mdc-text-field__input" placeholder="${placeholder}">
          <label class="mdc-floating-label">${placeholder}</label>
        </div>`,
      getValue: () => value,
      setValue: (v) => { value = v; },
    };
  }

  createModal(title: string): Modal {
    let isOpen = false;
    return {
      render: () => `
        <div class="mdc-dialog ${isOpen ? "mdc-dialog--open" : ""}">
          <div class="mdc-dialog__title">${title}</div>
        </div>`,
      open: () => { isOpen = true; },
      close: () => { isOpen = false; },
    };
  }

  createCard(): Card {
    let content = { title: "", body: "" };
    return {
      render: () => `
        <div class="mdc-card">
          <div class="mdc-card__title">${content.title}</div>
          <div class="mdc-card__body">${content.body}</div>
        </div>`,
      setContent: (title, body) => { content = { title, body }; },
    };
  }
}

// Ant Design factory
class AntDesignFactory implements UIFactory {
  createButton(label: string): Button {
    return {
      render: () => `<button class="ant-btn ant-btn-primary">${label}</button>`,
      onClick: (handler) => { /* Ant Button handler */ },
    };
  }

  createInput(placeholder: string): Input {
    let value = "";
    return {
      render: () => `
        <span class="ant-input-affix-wrapper">
          <input class="ant-input" placeholder="${placeholder}">
        </span>`,
      getValue: () => value,
      setValue: (v) => { value = v; },
    };
  }

  createModal(title: string): Modal {
    let isOpen = false;
    return {
      render: () => `
        <div class="ant-modal ${isOpen ? "ant-modal-visible" : ""}">
          <div class="ant-modal-header">${title}</div>
        </div>`,
      open: () => { isOpen = true; },
      close: () => { isOpen = false; },
    };
  }

  createCard(): Card {
    let content = { title: "", body: "" };
    return {
      render: () => `
        <div class="ant-card">
          <div class="ant-card-head">${content.title}</div>
          <div class="ant-card-body">${content.body}</div>
        </div>`,
      setContent: (title, body) => { content = { title, body }; },
    };
  }
}

// Just changing the theme swaps all UI components
function buildDashboard(factory: UIFactory) {
  const searchInput = factory.createInput("Search...");
  const submitButton = factory.createButton("Submit");
  const detailModal = factory.createModal("Details");
  const summaryCard = factory.createCard();
  summaryCard.setContent("This month's sales", "$1,234,567");

  return {
    render: () => `
      <div class="dashboard">
        ${searchInput.render()}
        ${submitButton.render()}
        ${summaryCard.render()}
        ${detailModal.render()}
      </div>
    `,
  };
}

// Select the factory based on configuration
const theme = process.env.UI_THEME ?? "material";
const factory: UIFactory = theme === "ant"
  ? new AntDesignFactory()
  : new MaterialUIFactory();

const dashboard = buildDashboard(factory);
```

### 2.3 Database Abstraction Layer

```typescript
// Abstract Factory: abstracting the database
interface DbConnection {
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  isConnected(): boolean;
}

interface DbQueryBuilder {
  select(table: string, columns: string[]): DbQueryBuilder;
  where(condition: string, params: unknown[]): DbQueryBuilder;
  orderBy(column: string, direction: "asc" | "desc"): DbQueryBuilder;
  limit(count: number): DbQueryBuilder;
  build(): { sql: string; params: unknown[] };
}

interface DbMigrationRunner {
  up(sql: string): Promise<void>;
  down(sql: string): Promise<void>;
  getVersion(): Promise<number>;
}

interface DatabaseFactory {
  createConnection(config: DbConfig): DbConnection;
  createQueryBuilder(): DbQueryBuilder;
  createMigrationRunner(connection: DbConnection): DbMigrationRunner;
}

// PostgreSQL implementation
class PostgresFactory implements DatabaseFactory {
  createConnection(config: DbConfig): DbConnection {
    return new PostgresConnection(config);
  }

  createQueryBuilder(): DbQueryBuilder {
    return new PostgresQueryBuilder();
  }

  createMigrationRunner(connection: DbConnection): DbMigrationRunner {
    return new PostgresMigrationRunner(connection);
  }
}

// MySQL implementation
class MySQLFactory implements DatabaseFactory {
  createConnection(config: DbConfig): DbConnection {
    return new MySQLConnection(config);
  }

  createQueryBuilder(): DbQueryBuilder {
    return new MySQLQueryBuilder();
  }

  createMigrationRunner(connection: DbConnection): DbMigrationRunner {
    return new MySQLMigrationRunner(connection);
  }
}

// SQLite implementation (for testing)
class SQLiteFactory implements DatabaseFactory {
  createConnection(config: DbConfig): DbConnection {
    return new SQLiteConnection(config);
  }

  createQueryBuilder(): DbQueryBuilder {
    return new SQLiteQueryBuilder();
  }

  createMigrationRunner(connection: DbConnection): DbMigrationRunner {
    return new SQLiteMigrationRunner(connection);
  }
}

// The application layer does not know the database type
class UserRepository {
  private queryBuilder: DbQueryBuilder;

  constructor(private dbFactory: DatabaseFactory) {
    this.queryBuilder = dbFactory.createQueryBuilder();
  }

  async findActiveUsers(limit: number = 10): Promise<User[]> {
    const query = this.queryBuilder
      .select("users", ["id", "name", "email", "role"])
      .where("active = ?", [true])
      .orderBy("created_at", "desc")
      .limit(limit)
      .build();

    // execute query...
    return [];
  }
}

// Switch factories based on the environment
function createDatabaseFactory(env: string): DatabaseFactory {
  switch (env) {
    case "production": return new PostgresFactory();
    case "development": return new MySQLFactory();
    case "test": return new SQLiteFactory();
    default: return new SQLiteFactory();
  }
}
```

### 2.4 Abstract Factory in Python

```python
# Python: Abstract Factory pattern
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

# Abstract products
class Serializer(Protocol):
    def serialize(self, data: dict) -> str: ...
    def deserialize(self, raw: str) -> dict: ...

class Validator(Protocol):
    def validate(self, data: dict, schema: dict) -> list[str]: ...

class Formatter(Protocol):
    def format(self, data: dict) -> str: ...

# Abstract factory
class DataProcessingFactory(ABC):
    @abstractmethod
    def create_serializer(self) -> Serializer: ...

    @abstractmethod
    def create_validator(self) -> Validator: ...

    @abstractmethod
    def create_formatter(self) -> Formatter: ...

# JSON family
class JsonSerializer:
    def serialize(self, data: dict) -> str:
        import json
        return json.dumps(data, ensure_ascii=False, indent=2)

    def deserialize(self, raw: str) -> dict:
        import json
        return json.loads(raw)

class JsonValidator:
    def validate(self, data: dict, schema: dict) -> list[str]:
        errors = []
        for field, rules in schema.items():
            if rules.get("required") and field not in data:
                errors.append(f"Missing required field: {field}")
        return errors

class JsonFormatter:
    def format(self, data: dict) -> str:
        import json
        return json.dumps(data, ensure_ascii=False, indent=4)

class JsonProcessingFactory(DataProcessingFactory):
    def create_serializer(self) -> Serializer:
        return JsonSerializer()

    def create_validator(self) -> Validator:
        return JsonValidator()

    def create_formatter(self) -> Formatter:
        return JsonFormatter()

# XML family
class XmlSerializer:
    def serialize(self, data: dict) -> str:
        def dict_to_xml(d: dict, root: str = "root") -> str:
            xml = f"<{root}>"
            for key, value in d.items():
                xml += f"<{key}>{value}</{key}>"
            xml += f"</{root}>"
            return xml
        return dict_to_xml(data)

    def deserialize(self, raw: str) -> dict:
        # Simple XML parser
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw)
        return {child.tag: child.text for child in root}

class XmlValidator:
    def validate(self, data: dict, schema: dict) -> list[str]:
        # XSD-based validation
        errors = []
        for field, rules in schema.items():
            if rules.get("required") and field not in data:
                errors.append(f"Missing required element: <{field}>")
        return errors

class XmlFormatter:
    def format(self, data: dict) -> str:
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append("<root>")
        for key, value in data.items():
            lines.append(f"  <{key}>{value}</{key}>")
        lines.append("</root>")
        return "\n".join(lines)

class XmlProcessingFactory(DataProcessingFactory):
    def create_serializer(self) -> Serializer:
        return XmlSerializer()

    def create_validator(self) -> Validator:
        return XmlValidator()

    def create_formatter(self) -> Formatter:
        return XmlFormatter()

# Example usage
def process_data(factory: DataProcessingFactory, data: dict, schema: dict) -> str:
    validator = factory.create_validator()
    errors = validator.validate(data, schema)
    if errors:
        raise ValueError(f"Validation errors: {errors}")

    serializer = factory.create_serializer()
    serialized = serializer.serialize(data)

    formatter = factory.create_formatter()
    return formatter.format(data)
```

---

## 3. Builder Pattern

### 3.1 Overview and Purpose

```
Purpose: Separate the construction process of a complex object

When to use:
  -> The constructor has many arguments (5 or more)
  -> There are many optional parameters
  -> You want to build incrementally
  -> You want to produce different representations from the same construction process

Structure:
  +----------+      +-----------+      +---------+
  | Director |----->|  Builder  |----->| Product |
  | (leader) |      |(constructor)|    |(product)|
  +----------+      +-----------+      +---------+
```

### 3.2 Fluent Builder (Method Chaining)

```typescript
// Builder pattern: constructing an HTTP request
class HttpRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: Record<string, string>;
  readonly body?: string;
  readonly timeout: number;
  readonly retries: number;
  readonly auth?: { type: string; credentials: string };
  readonly queryParams: Record<string, string>;

  private constructor(builder: HttpRequestBuilder) {
    this.method = builder.method;
    this.url = builder.url;
    this.headers = { ...builder.headers };
    this.body = builder.body;
    this.timeout = builder.timeout;
    this.retries = builder.retries;
    this.auth = builder.auth ? { ...builder.auth } : undefined;
    this.queryParams = { ...builder.queryParams };
  }

  static builder(method: string, url: string): HttpRequestBuilder {
    return new HttpRequestBuilder(method, url);
  }
}

class HttpRequestBuilder {
  headers: Record<string, string> = {};
  body?: string;
  timeout: number = 5000;
  retries: number = 0;
  auth?: { type: string; credentials: string };
  queryParams: Record<string, string> = {};

  constructor(
    public readonly method: string,
    public readonly url: string,
  ) {}

  setHeader(key: string, value: string): this {
    this.headers[key] = value;
    return this; // Method chaining
  }

  setBody(body: string): this {
    this.body = body;
    return this;
  }

  setJsonBody(data: unknown): this {
    this.body = JSON.stringify(data);
    this.headers["Content-Type"] = "application/json";
    return this;
  }

  setTimeout(ms: number): this {
    this.timeout = ms;
    return this;
  }

  setRetries(n: number): this {
    this.retries = n;
    return this;
  }

  setBasicAuth(username: string, password: string): this {
    this.auth = {
      type: "basic",
      credentials: Buffer.from(`${username}:${password}`).toString("base64"),
    };
    return this;
  }

  setBearerToken(token: string): this {
    this.auth = { type: "bearer", credentials: token };
    return this;
  }

  addQueryParam(key: string, value: string): this {
    this.queryParams[key] = value;
    return this;
  }

  build(): HttpRequest {
    // Validation
    if (!this.url) throw new Error("URL is required");
    if (!this.method) throw new Error("Method is required");
    if (this.timeout < 0) throw new Error("Timeout must be non-negative");
    if (this.retries < 0) throw new Error("Retries must be non-negative");

    return new (HttpRequest as any)(this);
  }
}

// Highly readable object construction
const request = HttpRequest.builder("POST", "https://api.example.com/users")
  .setHeader("Accept", "application/json")
  .setBearerToken("token123")
  .setJsonBody({ name: "Tanaka", email: "tanaka@example.com" })
  .setTimeout(10000)
  .setRetries(3)
  .addQueryParam("version", "v2")
  .build();
```

### 3.3 Step Builder (Incremental Builder)

```typescript
// Step Builder: enforces required parameters through the type system
interface NeedsRecipient {
  to(recipient: string): NeedsSubject;
}

interface NeedsSubject {
  subject(subject: string): EmailBuilder;
}

interface EmailBuilder {
  body(body: string): EmailBuilder;
  cc(address: string): EmailBuilder;
  bcc(address: string): EmailBuilder;
  attachment(file: Buffer, name: string): EmailBuilder;
  replyTo(address: string): EmailBuilder;
  priority(level: "low" | "normal" | "high"): EmailBuilder;
  build(): Email;
}

class Email {
  constructor(
    public readonly to: string,
    public readonly subject: string,
    public readonly body: string,
    public readonly cc: string[],
    public readonly bcc: string[],
    public readonly attachments: { file: Buffer; name: string }[],
    public readonly replyTo?: string,
    public readonly priority: "low" | "normal" | "high" = "normal",
  ) {}

  static create(): NeedsRecipient {
    return new EmailStepBuilder();
  }
}

class EmailStepBuilder implements NeedsRecipient, NeedsSubject, EmailBuilder {
  private _to = "";
  private _subject = "";
  private _body = "";
  private _cc: string[] = [];
  private _bcc: string[] = [];
  private _attachments: { file: Buffer; name: string }[] = [];
  private _replyTo?: string;
  private _priority: "low" | "normal" | "high" = "normal";

  to(recipient: string): NeedsSubject {
    this._to = recipient;
    return this;
  }

  subject(subject: string): EmailBuilder {
    this._subject = subject;
    return this;
  }

  body(body: string): EmailBuilder {
    this._body = body;
    return this;
  }

  cc(address: string): EmailBuilder {
    this._cc.push(address);
    return this;
  }

  bcc(address: string): EmailBuilder {
    this._bcc.push(address);
    return this;
  }

  attachment(file: Buffer, name: string): EmailBuilder {
    this._attachments.push({ file, name });
    return this;
  }

  replyTo(address: string): EmailBuilder {
    this._replyTo = address;
    return this;
  }

  priority(level: "low" | "normal" | "high"): EmailBuilder {
    this._priority = level;
    return this;
  }

  build(): Email {
    return new Email(
      this._to, this._subject, this._body,
      this._cc, this._bcc, this._attachments,
      this._replyTo, this._priority,
    );
  }}

// Example usage: the type system enforces the order
const email = Email.create()
  .to("tanaka@example.com")     // Required: must be specified first
  .subject("Monthly Report")    // Required: specified next
  .body("Please review the attached report.") // Optional
  .cc("manager@example.com")
  .priority("high")
  .build();

// Email.create().subject("...")  // Compile error: to() must come first
```

### 3.4 Director Pattern (Standardized Construction)

```typescript
// Director: encapsulates standardized construction procedures
class QueryDirector {
  // Paginated list fetch
  static paginatedList<T>(
    builder: QueryBuilder<T>,
    page: number,
    pageSize: number,
  ): Query<T> {
    return builder
      .orderBy("created_at", "desc")
      .limit(pageSize)
      .offset((page - 1) * pageSize)
      .build();
  }

  // Search query
  static search<T>(
    builder: QueryBuilder<T>,
    keyword: string,
    searchFields: string[],
  ): Query<T> {
    const conditions = searchFields.map(
      field => `${field} LIKE '%${keyword}%'`
    );
    return builder
      .whereRaw(conditions.join(" OR "))
      .orderBy("relevance", "desc")
      .limit(50)
      .build();
  }

  // Retrieve only records that have not been soft-deleted
  static activeOnly<T>(
    builder: QueryBuilder<T>,
  ): Query<T> {
    return builder
      .where("deleted_at", "IS NULL")
      .orderBy("updated_at", "desc")
      .build();
  }
}
```

### 3.5 Builder Pattern in Java

```java
// Java: Builder pattern (hand-written, no Lombok)
public class ServerConfig {
    private final String host;
    private final int port;
    private final boolean ssl;
    private final int maxConnections;
    private final int timeoutSeconds;
    private final String certPath;
    private final String keyPath;
    private final List<String> allowedOrigins;
    private final Map<String, String> customHeaders;
    private final boolean gzipEnabled;
    private final int maxRequestSize;

    private ServerConfig(Builder builder) {
        this.host = builder.host;
        this.port = builder.port;
        this.ssl = builder.ssl;
        this.maxConnections = builder.maxConnections;
        this.timeoutSeconds = builder.timeoutSeconds;
        this.certPath = builder.certPath;
        this.keyPath = builder.keyPath;
        this.allowedOrigins = List.copyOf(builder.allowedOrigins);
        this.customHeaders = Map.copyOf(builder.customHeaders);
        this.gzipEnabled = builder.gzipEnabled;
        this.maxRequestSize = builder.maxRequestSize;
    }

    // Getter methods...
    public String getHost() { return host; }
    public int getPort() { return port; }
    public boolean isSsl() { return ssl; }
    // ... omitted

    public static Builder builder(String host, int port) {
        return new Builder(host, port);
    }

    public static class Builder {
        // Required
        private final String host;
        private final int port;
        // Optional (with default values)
        private boolean ssl = false;
        private int maxConnections = 100;
        private int timeoutSeconds = 30;
        private String certPath = "";
        private String keyPath = "";
        private List<String> allowedOrigins = new ArrayList<>();
        private Map<String, String> customHeaders = new HashMap<>();
        private boolean gzipEnabled = true;
        private int maxRequestSize = 10 * 1024 * 1024; // 10MB

        private Builder(String host, int port) {
            this.host = host;
            this.port = port;
        }

        public Builder ssl(boolean ssl) {
            this.ssl = ssl;
            return this;
        }

        public Builder maxConnections(int max) {
            this.maxConnections = max;
            return this;
        }

        public Builder timeout(int seconds) {
            this.timeoutSeconds = seconds;
            return this;
        }

        public Builder cert(String certPath, String keyPath) {
            this.certPath = certPath;
            this.keyPath = keyPath;
            this.ssl = true;
            return this;
        }

        public Builder allowOrigin(String origin) {
            this.allowedOrigins.add(origin);
            return this;
        }

        public Builder header(String key, String value) {
            this.customHeaders.put(key, value);
            return this;
        }

        public Builder gzip(boolean enabled) {
            this.gzipEnabled = enabled;
            return this;
        }

        public Builder maxRequestSize(int bytes) {
            this.maxRequestSize = bytes;
            return this;
        }

        public ServerConfig build() {
            // Validation
            if (ssl && (certPath.isEmpty() || keyPath.isEmpty())) {
                throw new IllegalStateException(
                    "SSL enabled but cert/key paths not provided");
            }
            if (maxConnections <= 0) {
                throw new IllegalStateException("maxConnections must be positive");
            }
            return new ServerConfig(this);
        }
    }
}

// Example usage
ServerConfig config = ServerConfig.builder("0.0.0.0", 8080)
    .ssl(true)
    .cert("/etc/ssl/cert.pem", "/etc/ssl/key.pem")
    .maxConnections(500)
    .timeout(60)
    .allowOrigin("https://example.com")
    .allowOrigin("https://app.example.com")
    .header("X-Powered-By", "MyApp")
    .gzip(true)
    .build();
```

---

## 4. Singleton Pattern

### 4.1 Overview and Caveats

```
Purpose: Ensure that a class has only one instance

Caveat: Singleton is often criticized as an "anti-pattern"

Problems:
  -> Global state = hard to test
  -> Tight coupling = hinders dependency injection
  -> Concurrency = risk of race conditions
  -> Hidden dependencies = code becomes hard to understand

Appropriate uses:
  -> Logger, configuration manager (when you truly need only one)
  -> It is often better to let a DI container control "exactly one"
```

### 4.2 Basic Implementation

```typescript
// Singleton (minimal viable implementation)
class AppConfig {
  private static instance: AppConfig;

  private constructor(
    public readonly dbUrl: string,
    public readonly apiKey: string,
    public readonly debug: boolean,
    public readonly logLevel: "debug" | "info" | "warn" | "error",
    public readonly maxRetries: number,
  ) {}

  static getInstance(): AppConfig {
    if (!AppConfig.instance) {
      AppConfig.instance = new AppConfig(
        process.env.DATABASE_URL ?? "localhost:5432",
        process.env.API_KEY ?? "",
        process.env.NODE_ENV !== "production",
        (process.env.LOG_LEVEL as any) ?? "info",
        parseInt(process.env.MAX_RETRIES ?? "3", 10),
      );
    }
    return AppConfig.instance;
  }

  // Reset for testing
  static resetForTesting(): void {
    AppConfig.instance = undefined as any;
  }

  // For testing: inject a custom instance
  static setInstanceForTesting(config: AppConfig): void {
    AppConfig.instance = config;
  }
}
```

### 4.3 Thread-Safe Singleton (Java)

```java
// Java: various ways to implement a thread-safe Singleton

// Approach 1: Eager Initialization (simplest)
public class EagerSingleton {
    // Created at class-load time (thread-safe)
    private static final EagerSingleton INSTANCE = new EagerSingleton();

    private EagerSingleton() {}

    public static EagerSingleton getInstance() {
        return INSTANCE;
    }
}

// Approach 2: Double-Checked Locking (when lazy initialization is needed)
public class LazyThreadSafeSingleton {
    private static volatile LazyThreadSafeSingleton instance;

    private LazyThreadSafeSingleton() {}

    public static LazyThreadSafeSingleton getInstance() {
        if (instance == null) {
            synchronized (LazyThreadSafeSingleton.class) {
                if (instance == null) {
                    instance = new LazyThreadSafeSingleton();
                }
            }
        }
        return instance;
    }
}

// Approach 3: Holder Pattern (recommended: lazy initialization + thread-safe)
public class HolderSingleton {
    private HolderSingleton() {}

    // The inner class is not loaded until it is first accessed
    private static class Holder {
        private static final HolderSingleton INSTANCE = new HolderSingleton();
    }

    public static HolderSingleton getInstance() {
        return Holder.INSTANCE;
    }
}

// Approach 4: Enum Singleton (safest: resilient against serialization and reflection attacks)
public enum EnumSingleton {
    INSTANCE;

    private final Map<String, String> config = new HashMap<>();

    public void setConfig(String key, String value) {
        config.put(key, value);
    }

    public String getConfig(String key) {
        return config.get(key);
    }
}
```

### 4.4 Singleton Alternative: Scope Management via a DI Container

```typescript
// A better approach: manage scope with a DI container
interface ServiceContainer {
  register<T>(token: string, factory: () => T, scope?: "singleton" | "transient"): void;
  resolve<T>(token: string): T;
}

class SimpleContainer implements ServiceContainer {
  private factories = new Map<string, { factory: () => any; scope: string }>();
  private singletons = new Map<string, any>();

  register<T>(
    token: string,
    factory: () => T,
    scope: "singleton" | "transient" = "transient",
  ): void {
    this.factories.set(token, { factory, scope });
  }

  resolve<T>(token: string): T {
    const entry = this.factories.get(token);
    if (!entry) throw new Error(`Service not registered: ${token}`);

    if (entry.scope === "singleton") {
      if (!this.singletons.has(token)) {
        this.singletons.set(token, entry.factory());
      }
      return this.singletons.get(token);
    }

    return entry.factory();
  }

  // For testing: reset singletons
  resetSingletons(): void {
    this.singletons.clear();
  }
}

// Example usage
const container = new SimpleContainer();

// Register with singleton scope (share a single instance)
container.register("config", () => loadConfig(), "singleton");
container.register("logger", () => createLogger(), "singleton");

// Register with transient scope (new instance every time)
container.register("httpClient", () => new HttpClient(), "transient");

// Gains the benefits of Singleton while preserving testability
const config = container.resolve<AppConfig>("config");
const logger = container.resolve<Logger>("logger");

// During tests
container.register("config", () => createTestConfig(), "singleton");
container.resetSingletons();
```

---

## 5. Prototype Pattern

### 5.1 Overview and Purpose

```
Purpose: Create a new object by copying an existing object

When to use:
  -> The creation cost is high (built from a DB/API)
  -> You want to tweak based on a template object
  -> You want to save/restore object state (combined with Memento)
  -> You want to copy without knowing the concrete type
```

### 5.2 Basic Implementation

```typescript
// Prototype pattern
interface Cloneable<T> {
  clone(): T;
  deepClone(): T;
}

class DocumentTemplate implements Cloneable<DocumentTemplate> {
  constructor(
    public title: string,
    public content: string,
    public styles: Record<string, string>,
    public metadata: Record<string, string>,
    public sections: Array<{ heading: string; body: string }>,
  ) {}

  // Shallow copy
  clone(): DocumentTemplate {
    return new DocumentTemplate(
      this.title,
      this.content,
      { ...this.styles },
      { ...this.metadata },
      [...this.sections],  // Note: inner objects are shared
    );
  }

  // Deep copy
  deepClone(): DocumentTemplate {
    return new DocumentTemplate(
      this.title,
      this.content,
      { ...this.styles },
      { ...this.metadata },
      this.sections.map(s => ({ ...s })),  // Each section is copied as well
    );
  }
}

// Copy from a template and tweak
const template = new DocumentTemplate(
  "Monthly Report",
  "## Overview\n...",
  { fontSize: "14px", fontFamily: "Noto Sans JP" },
  { author: "", department: "" },
  [
    { heading: "Overview", body: "" },
    { heading: "Results", body: "" },
    { heading: "Issues and Countermeasures", body: "" },
    { heading: "Next Month's Plan", body: "" },
  ],
);

const januaryReport = template.deepClone();
januaryReport.title = "January 2025 Monthly Report";
januaryReport.metadata.author = "Taro Tanaka";
januaryReport.metadata.department = "Engineering";
januaryReport.sections[0].body = "Summary of January's development progress";

const februaryReport = template.deepClone();
februaryReport.title = "February 2025 Monthly Report";
februaryReport.metadata.author = "Taro Tanaka";
februaryReport.sections[0].body = "Summary of February's development progress";
```

### 5.3 Prototype Registry (Managing Prototypes)

```typescript
// Prototype Registry: manage templates
class PrototypeRegistry<T extends Cloneable<T>> {
  private prototypes = new Map<string, T>();

  register(name: string, prototype: T): void {
    this.prototypes.set(name, prototype);
  }

  unregister(name: string): boolean {
    return this.prototypes.delete(name);
  }

  create(name: string): T {
    const prototype = this.prototypes.get(name);
    if (!prototype) {
      const available = [...this.prototypes.keys()].join(", ");
      throw new Error(`Unknown prototype: "${name}". Available: ${available}`);
    }
    return prototype.clone();
  }

  list(): string[] {
    return [...this.prototypes.keys()];
  }
}

// Example usage: form templates
interface FormField {
  name: string;
  type: "text" | "number" | "email" | "select" | "textarea";
  label: string;
  required: boolean;
  options?: string[];
}

class FormTemplate implements Cloneable<FormTemplate> {
  constructor(
    public name: string,
    public description: string,
    public fields: FormField[],
  ) {}

  clone(): FormTemplate {
    return new FormTemplate(
      this.name,
      this.description,
      this.fields.map(f => ({ ...f, options: f.options ? [...f.options] : undefined })),
    );
  }

  deepClone(): FormTemplate {
    return this.clone();
  }

  addField(field: FormField): void {
    this.fields.push(field);
  }
}

// Register predefined form templates in the registry
const formRegistry = new PrototypeRegistry<FormTemplate>();

formRegistry.register("contact", new FormTemplate(
  "Contact Us",
  "Contact form",
  [
    { name: "name", type: "text", label: "Full Name", required: true },
    { name: "email", type: "email", label: "Email", required: true },
    { name: "category", type: "select", label: "Category", required: true,
      options: ["General", "Technical", "Sales", "Other"] },
    { name: "message", type: "textarea", label: "Message", required: true },
  ],
));

formRegistry.register("survey", new FormTemplate(
  "Survey",
  "Customer satisfaction survey",
  [
    { name: "satisfaction", type: "select", label: "Satisfaction", required: true,
      options: ["Very satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very dissatisfied"] },
    { name: "feedback", type: "textarea", label: "Comments", required: false },
  ],
));

// Customize from a template
const customContact = formRegistry.create("contact");
customContact.name = "Technical Support Inquiry";
customContact.addField({
  name: "product",
  type: "select",
  label: "Product",
  required: true,
  options: ["Product A", "Product B", "Product C"],
});
```

### 5.4 JavaScript Spread Syntax and Structured Clone

```typescript
// Modern copy techniques in JavaScript/TypeScript

// 1. Spread syntax (shallow copy)
const original = { name: "Taro", tags: ["developer", "manager"] };
const shallow = { ...original };
shallow.tags.push("admin");  // Warning: original.tags is also modified!

// 2. Object.assign (shallow copy)
const assigned = Object.assign({}, original);

// 3. JSON.parse/JSON.stringify (deep copy, with limitations)
const jsonCopy = JSON.parse(JSON.stringify(original));
// Warning: Date, Map, Set, RegExp, functions, undefined are not copied correctly

// 4. structuredClone (modern deep copy, Node.js 17+)
const structuredCopy = structuredClone(original);
// OK: correctly copies Date, Map, Set, ArrayBuffer, RegExp
// NG: cannot copy functions, DOM nodes, Error objects

// 5. A practical deep clone utility
function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== "object") return obj;
  if (obj instanceof Date) return new Date(obj.getTime()) as T;
  if (obj instanceof Map) {
    const map = new Map();
    obj.forEach((value, key) => map.set(deepClone(key), deepClone(value)));
    return map as T;
  }
  if (obj instanceof Set) {
    const set = new Set();
    obj.forEach(value => set.add(deepClone(value)));
    return set as T;
  }
  if (Array.isArray(obj)) return obj.map(item => deepClone(item)) as T;

  const copy = {} as T;
  for (const key of Object.keys(obj as object)) {
    (copy as any)[key] = deepClone((obj as any)[key]);
  }
  return copy;
}
```

---

## 6. Object Pool Pattern

### 6.1 Overview and Purpose

```
Purpose: Reuse objects that are expensive to create

When to use:
  -> Object creation/destruction is expensive (DB connections, threads)
  -> The number of simultaneously required objects is limited
  -> Objects are reusable (their state can be reset)
```

### 6.2 Implementing a Connection Pool

```typescript
// Object Pool: database connection pool
interface Poolable {
  reset(): void;
  isValid(): boolean;
  destroy(): void;
}

class ConnectionPool<T extends Poolable> {
  private available: T[] = [];
  private inUse = new Set<T>();
  private waitQueue: Array<{
    resolve: (conn: T) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
  }> = [];

  constructor(
    private factory: () => T,
    private readonly minSize: number = 2,
    private readonly maxSize: number = 10,
    private readonly acquireTimeoutMs: number = 5000,
  ) {
    // Pre-create the minimum number of objects
    for (let i = 0; i < minSize; i++) {
      this.available.push(factory());
    }
  }

  async acquire(): Promise<T> {
    // Return an available object if one exists
    while (this.available.length > 0) {
      const obj = this.available.pop()!;
      if (obj.isValid()) {
        this.inUse.add(obj);
        return obj;
      }
      obj.destroy(); // Destroy invalid objects
    }

    // Create a new one if we haven't reached the maximum
    if (this.inUse.size < this.maxSize) {
      const obj = this.factory();
      this.inUse.add(obj);
      return obj;
    }

    // Wait for a slot
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        const index = this.waitQueue.findIndex(w => w.timer === timer);
        if (index >= 0) this.waitQueue.splice(index, 1);
        reject(new Error("Connection pool acquire timeout"));
      }, this.acquireTimeoutMs);

      this.waitQueue.push({ resolve, reject, timer });
    });
  }

  release(obj: T): void {
    if (!this.inUse.has(obj)) return;
    this.inUse.delete(obj);
    obj.reset();

    // Hand off to a waiting request if any
    if (this.waitQueue.length > 0 && obj.isValid()) {
      const waiter = this.waitQueue.shift()!;
      clearTimeout(waiter.timer);
      this.inUse.add(obj);
      waiter.resolve(obj);
    } else if (obj.isValid()) {
      this.available.push(obj);
    } else {
      obj.destroy();
    }
  }

  // using pattern (ensures reliable release)
  async withConnection<R>(fn: (conn: T) => Promise<R>): Promise<R> {
    const conn = await this.acquire();
    try {
      return await fn(conn);
    } finally {
      this.release(conn);
    }
  }

  get stats() {
    return {
      available: this.available.length,
      inUse: this.inUse.size,
      waiting: this.waitQueue.length,
      total: this.available.length + this.inUse.size,
      maxSize: this.maxSize,
    };
  }

  async drain(): Promise<void> {
    for (const obj of this.available) obj.destroy();
    for (const obj of this.inUse) obj.destroy();
    this.available = [];
    this.inUse.clear();
    for (const waiter of this.waitQueue) {
      clearTimeout(waiter.timer);
      waiter.reject(new Error("Pool drained"));
    }
    this.waitQueue = [];
  }
}

// Example usage
const dbPool = new ConnectionPool(
  () => new DatabaseConnection("postgres://localhost:5432/mydb"),
  { minSize: 5, maxSize: 20, acquireTimeoutMs: 3000 }
);

// using pattern (recommended)
const users = await dbPool.withConnection(async (conn) => {
  return conn.query("SELECT * FROM users WHERE active = true");
});

console.log(dbPool.stats);
// { available: 5, inUse: 0, waiting: 0, total: 5, maxSize: 20 }
```

---

## 7. Selection Guidelines

### 7.1 Pattern Comparison

```
+------------------+------------------------------------------+
| Pattern          | When to use                              |
+------------------+------------------------------------------+
| Simple Factory   | Decide type at runtime, centralize logic |
| Factory Method   | Subclass decides the product type        |
| Abstract Factory | Consistent creation of related object sets|
| Builder          | Complex construction with many parameters|
| Step Builder     | Enforce order of required params via types|
| Singleton        | Really need just one (use cautiously)    |
| Prototype        | Copy based on an existing object         |
| Object Pool      | Reuse expensive-to-create objects        |
+------------------+------------------------------------------+
```

### 7.2 Decision Flowchart

```
Object creation needed
|-- Type to create is determined at runtime
|   |-- Related groups of objects -> Abstract Factory
|   \-- Single object -> Factory Method / Simple Factory
|-- Many parameters or incremental construction
|   |-- Enforce order of required parameters -> Step Builder
|   \-- Flexible construction -> Fluent Builder
|-- Want to copy an existing object -> Prototype
|-- Want to limit the number of instances
|   |-- Only one -> Singleton (* DI recommended)
|   \-- Up to N -> Object Pool
\-- Simple creation -> Call the constructor directly
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Beyond theory, your understanding deepens when you actually write code and verify its behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and moving on to advanced topics. We recommend solidly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

The knowledge covered in this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Pattern | Purpose | Caveats |
|---------|------|--------|
| Simple Factory | Centralize creation logic | Avoid excessive factory-ization |
| Factory Method | Subclass determines type | Use alongside template method |
| Abstract Factory | Swap product families | Number of interfaces tends to grow |
| Builder | Incremental construction | Unnecessary for simple classes |
| Step Builder | Type-safe incremental construction | Number of interfaces increases |
| Singleton | Guarantee uniqueness | Consider DI as an alternative |
| Prototype | Creation by copying | Be careful with deep/shallow copy |
| Object Pool | Object reuse | Beware of forgetting to release |

---

## Next Guides to Read

---

## References
1. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
2. Bloch, J. "Effective Java." 3rd Ed, Item 2: Consider a builder when faced with many constructor parameters. 2018.
3. Freeman, E. et al. "Head First Design Patterns." O'Reilly, 2020.
4. Martin, R.C. "Clean Code." Chapter 3: Functions. Prentice Hall, 2008.
5. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.
