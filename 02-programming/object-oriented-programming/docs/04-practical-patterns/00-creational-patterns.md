# 生成パターン（Creational Patterns）

> オブジェクトの生成方法を柔軟にする設計パターン。Factory、Abstract Factory、Builder、Singleton、Prototype の5つの主要パターンの「なぜ必要か」「いつ使うか」を実践的に解説。

## この章で学ぶこと

- [ ] 各生成パターンの目的と使い分けを理解する
- [ ] 各パターンの実装方法を複数言語で把握する
- [ ] アンチパターンとしての Singleton の問題を学ぶ
- [ ] 現代のフレームワークでの生成パターンの応用を知る
- [ ] テスタビリティを考慮した生成パターンの設計ができるようになる

---

## 1. Factory Method パターン

### 1.1 概要と目的

```
目的: オブジェクトの生成ロジックをカプセル化する

いつ使うか:
  → 生成するクラスを実行時に決定したい
  → 生成ロジックが複雑
  → new を直接使わせたくない
  → テスト時にモックオブジェクトに差し替えたい

構造:
  ┌────────────────┐         ┌─────────────────┐
  │    Creator     │         │    Product       │
  │ (ファクトリ)   │────────→│ (生成物)         │
  ├────────────────┤         ├─────────────────┤
  │ factoryMethod()│         │ operation()      │
  └───────┬────────┘         └────────┬────────┘
          │                           │
  ┌───────┴────────┐         ┌────────┴────────┐
  │ConcreteCreator │         │ConcreteProduct  │
  │                │────────→│                 │
  └────────────────┘         └─────────────────┘
```

### 1.2 Simple Factory（静的ファクトリメソッド）

```typescript
// Simple Factory: 最も基本的なファクトリ
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

// ファクトリ: 生成ロジックを集約
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

  // 複数の通知を一括生成
  static createBatch(configs: NotificationConfig[]): Notification[] {
    return configs.map(config => this.create(config));
  }
}

// 利用側は具象クラスを知らなくてよい
const notification = NotificationFactory.create({
  type: "email",
  target: "tanaka@example.com",
});
notification.send("Hello!");

// 複数チャネルへの一括送信
const notifications = NotificationFactory.createBatch([
  { type: "email", target: "tanaka@example.com" },
  { type: "slack", target: "general" },
  { type: "push", target: "device-token-123" },
]);
notifications.forEach(n => n.send("重要なお知らせ"));
```

### 1.3 Factory Method（テンプレートメソッドとの組み合わせ）

```typescript
// Factory Method: サブクラスが生成するオブジェクトの型を決定
abstract class DocumentExporter {
  // テンプレートメソッド
  async export(data: ReportData): Promise<Buffer> {
    const formatter = this.createFormatter();
    const header = formatter.formatHeader(data.title);
    const body = formatter.formatBody(data.content);
    const footer = formatter.formatFooter(data.metadata);
    return Buffer.from(header + body + footer);
  }

  // Factory Method: サブクラスが実装
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

// 各フォーマッタの実装
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

// 使用例: エクスポート形式を動的に選択
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

### 1.4 Java での Factory Method

```java
// Java: Factory Method パターンの実装
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
        // コンソールは閉じる必要なし
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

// ファクトリクラス
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

// 使用例
Logger logger = LoggerFactory.createLogger("file",
    Map.of("path", "/var/log/myapp.log"));
logger.log("INFO", "Application started");
```

### 1.5 登録ベースのファクトリ（拡張可能なファクトリ）

```typescript
// 登録ベース: 新しいタイプを動的に追加可能
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

// 使用例: プラグイン可能な通知システム
interface NotificationPlugin {
  send(message: string, target: string): Promise<void>;
  getName(): string;
}

const notificationFactory = new PluggableFactory<NotificationPlugin>();

// コアプラグインの登録
notificationFactory.register("email", (smtpConfig: SmtpConfig) => ({
  async send(message: string, target: string) {
    // SMTP送信
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

// サードパーティがプラグインを追加
notificationFactory.register("teams", (webhookUrl: string) => ({
  async send(message: string, target: string) {
    await fetch(webhookUrl, {
      method: "POST",
      body: JSON.stringify({ text: `${target}: ${message}` }),
    });
  },
  getName() { return "Microsoft Teams"; },
}));

// 利用
const emailPlugin = notificationFactory.create("email", { host: "smtp.example.com" });
const teamsPlugin = notificationFactory.create("teams", "https://outlook.webhook.office.com/...");
```

---

## 2. Abstract Factory パターン

### 2.1 概要と目的

```
目的: 関連するオブジェクト群を、具象クラスを指定せずに生成する

いつ使うか:
  → 関連するオブジェクト群を一貫して生成したい
  → 製品ファミリーを切り替えたい（テーマ、プラットフォーム）
  → 具象クラスからクライアントを完全に分離したい

構造:
  ┌─────────────────┐
  │ AbstractFactory  │
  │ createProductA() │
  │ createProductB() │
  └────────┬────────┘
           │
  ┌────────┴────────┐
  │                 │
  ┌─────────┐  ┌─────────┐
  │Factory1 │  │Factory2 │
  └─────────┘  └─────────┘
```

### 2.2 UIテーマの切り替え

```typescript
// Abstract Factory: 関連するオブジェクト群を生成
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

// Material Design ファクトリ
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

// Ant Design ファクトリ
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

// テーマを変更するだけで全UIコンポーネントが切り替わる
function buildDashboard(factory: UIFactory) {
  const searchInput = factory.createInput("検索...");
  const submitButton = factory.createButton("送信");
  const detailModal = factory.createModal("詳細");
  const summaryCard = factory.createCard();
  summaryCard.setContent("今月の売上", "¥1,234,567");

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

// 設定に応じてファクトリを選択
const theme = process.env.UI_THEME ?? "material";
const factory: UIFactory = theme === "ant"
  ? new AntDesignFactory()
  : new MaterialUIFactory();

const dashboard = buildDashboard(factory);
```

### 2.3 データベース抽象化レイヤー

```typescript
// Abstract Factory: データベースの抽象化
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

// PostgreSQL 実装
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

// MySQL 実装
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

// SQLite 実装（テスト用）
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

// アプリケーション層はデータベースの種類を知らない
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

// 環境に応じてファクトリを切り替え
function createDatabaseFactory(env: string): DatabaseFactory {
  switch (env) {
    case "production": return new PostgresFactory();
    case "development": return new MySQLFactory();
    case "test": return new SQLiteFactory();
    default: return new SQLiteFactory();
  }
}
```

### 2.4 Python での Abstract Factory

```python
# Python: Abstract Factory パターン
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

# 抽象プロダクト
class Serializer(Protocol):
    def serialize(self, data: dict) -> str: ...
    def deserialize(self, raw: str) -> dict: ...

class Validator(Protocol):
    def validate(self, data: dict, schema: dict) -> list[str]: ...

class Formatter(Protocol):
    def format(self, data: dict) -> str: ...

# 抽象ファクトリ
class DataProcessingFactory(ABC):
    @abstractmethod
    def create_serializer(self) -> Serializer: ...

    @abstractmethod
    def create_validator(self) -> Validator: ...

    @abstractmethod
    def create_formatter(self) -> Formatter: ...

# JSON ファミリー
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

# XML ファミリー
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
        # 簡易的なXMLパーサー
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw)
        return {child.tag: child.text for child in root}

class XmlValidator:
    def validate(self, data: dict, schema: dict) -> list[str]:
        # XSD ベースのバリデーション
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

# 使用例
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

## 3. Builder パターン

### 3.1 概要と目的

```
目的: 複雑なオブジェクトの構築過程を分離する

いつ使うか:
  → コンストラクタの引数が多い（5個以上）
  → オプショナルなパラメータが多い
  → 段階的に構築したい
  → 同じ構築プロセスで異なる表現を生成したい

構造:
  ┌──────────┐      ┌───────────┐      ┌─────────┐
  │ Director │─────→│  Builder  │─────→│ Product │
  │ (指揮者) │      │ (構築者)  │      │ (製品)  │
  └──────────┘      └───────────┘      └─────────┘
```

### 3.2 Fluent Builder（メソッドチェーン）

```typescript
// Builder パターン: HTTPリクエストの構築
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
    return this; // メソッドチェーン
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
    // バリデーション
    if (!this.url) throw new Error("URL is required");
    if (!this.method) throw new Error("Method is required");
    if (this.timeout < 0) throw new Error("Timeout must be non-negative");
    if (this.retries < 0) throw new Error("Retries must be non-negative");

    return new (HttpRequest as any)(this);
  }
}

// 可読性の高いオブジェクト構築
const request = HttpRequest.builder("POST", "https://api.example.com/users")
  .setHeader("Accept", "application/json")
  .setBearerToken("token123")
  .setJsonBody({ name: "田中", email: "tanaka@example.com" })
  .setTimeout(10000)
  .setRetries(3)
  .addQueryParam("version", "v2")
  .build();
```

### 3.3 Step Builder（段階的ビルダー）

```typescript
// Step Builder: 必須パラメータを型で強制
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
  }
}

// 使用例: 型システムが順序を強制
const email = Email.create()
  .to("tanaka@example.com")     // 必須: 最初に指定
  .subject("月次レポート")       // 必須: 次に指定
  .body("添付のレポートをご確認ください。") // オプション
  .cc("manager@example.com")
  .priority("high")
  .build();

// Email.create().subject("...")  // コンパイルエラー: to() が先に必要
```

### 3.4 Director パターン（定型構築）

```typescript
// Director: 定型的な構築手順をカプセル化
class QueryDirector {
  // ページネーション付きリスト取得
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

  // 検索クエリ
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

  // ソフトデリート済みを除外した取得
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

### 3.5 Java での Builder パターン

```java
// Java: Builder パターン（Lombokなしの手書き版）
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

    // getter メソッド群...
    public String getHost() { return host; }
    public int getPort() { return port; }
    public boolean isSsl() { return ssl; }
    // ... 省略

    public static Builder builder(String host, int port) {
        return new Builder(host, port);
    }

    public static class Builder {
        // 必須
        private final String host;
        private final int port;
        // オプション（デフォルト値付き）
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
            // バリデーション
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

// 使用例
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

## 4. Singleton パターン

### 4.1 概要と注意点

```
目的: クラスのインスタンスが1つだけであることを保証する

注意: Singleton は「アンチパターン」として批判されることが多い

問題点:
  → グローバル状態 = テスト困難
  → 密結合 = 依存性注入の妨げ
  → 並行処理 = 競合状態のリスク
  → 隠れた依存 = コードの理解が困難

適切な用途:
  → ロガー、設定マネージャ（本当に1つでいい場合）
  → DIコンテナ側で「1つだけ」を制御する方が良い
```

### 4.2 基本実装

```typescript
// Singleton（必要最小限の実装）
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

  // テスト用リセット
  static resetForTesting(): void {
    AppConfig.instance = undefined as any;
  }

  // テスト用: カスタムインスタンスの注入
  static setInstanceForTesting(config: AppConfig): void {
    AppConfig.instance = config;
  }
}
```

### 4.3 スレッドセーフな Singleton（Java）

```java
// Java: スレッドセーフな Singleton の実装方法

// 方法1: Eager Initialization（最もシンプル）
public class EagerSingleton {
    // クラスロード時に生成（スレッドセーフ）
    private static final EagerSingleton INSTANCE = new EagerSingleton();

    private EagerSingleton() {}

    public static EagerSingleton getInstance() {
        return INSTANCE;
    }
}

// 方法2: Double-Checked Locking（遅延初期化が必要な場合）
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

// 方法3: Holder Pattern（推奨: 遅延初期化 + スレッドセーフ）
public class HolderSingleton {
    private HolderSingleton() {}

    // 内部クラスは最初にアクセスされるまでロードされない
    private static class Holder {
        private static final HolderSingleton INSTANCE = new HolderSingleton();
    }

    public static HolderSingleton getInstance() {
        return Holder.INSTANCE;
    }
}

// 方法4: Enum Singleton（最も安全: シリアライズ・リフレクション攻撃に耐える）
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

### 4.4 Singleton の代替: DIコンテナによるスコープ管理

```typescript
// より良いアプローチ: DIコンテナでスコープ管理
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

  // テスト用: シングルトンのリセット
  resetSingletons(): void {
    this.singletons.clear();
  }
}

// 使用例
const container = new SimpleContainer();

// シングルトンスコープで登録（1つのインスタンスを共有）
container.register("config", () => loadConfig(), "singleton");
container.register("logger", () => createLogger(), "singleton");

// トランジェントスコープで登録（毎回新しいインスタンス）
container.register("httpClient", () => new HttpClient(), "transient");

// Singleton の利点を活かしつつ、テスタビリティも確保
const config = container.resolve<AppConfig>("config");
const logger = container.resolve<Logger>("logger");

// テスト時
container.register("config", () => createTestConfig(), "singleton");
container.resetSingletons();
```

---

## 5. Prototype パターン

### 5.1 概要と目的

```
目的: 既存オブジェクトをコピーして新しいオブジェクトを生成する

いつ使うか:
  → 生成コストが高い（DB/APIから構築）
  → テンプレートオブジェクトを元に微調整
  → オブジェクトの状態を保存・復元（Memento との組み合わせ）
  → クラスの具体的な型を知らずにコピーしたい
```

### 5.2 基本実装

```typescript
// Prototype パターン
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

  // シャローコピー
  clone(): DocumentTemplate {
    return new DocumentTemplate(
      this.title,
      this.content,
      { ...this.styles },
      { ...this.metadata },
      [...this.sections],  // 注意: 中のオブジェクトは共有
    );
  }

  // ディープコピー
  deepClone(): DocumentTemplate {
    return new DocumentTemplate(
      this.title,
      this.content,
      { ...this.styles },
      { ...this.metadata },
      this.sections.map(s => ({ ...s })),  // 各セクションもコピー
    );
  }
}

// テンプレートからコピーして微調整
const template = new DocumentTemplate(
  "月次レポート",
  "## 概要\n...",
  { fontSize: "14px", fontFamily: "Noto Sans JP" },
  { author: "", department: "" },
  [
    { heading: "概要", body: "" },
    { heading: "実績", body: "" },
    { heading: "課題と対策", body: "" },
    { heading: "次月の計画", body: "" },
  ],
);

const januaryReport = template.deepClone();
januaryReport.title = "2025年1月 月次レポート";
januaryReport.metadata.author = "田中太郎";
januaryReport.metadata.department = "開発部";
januaryReport.sections[0].body = "1月の開発進捗をまとめる";

const februaryReport = template.deepClone();
februaryReport.title = "2025年2月 月次レポート";
februaryReport.metadata.author = "田中太郎";
februaryReport.sections[0].body = "2月の開発進捗をまとめる";
```

### 5.3 Prototype Registry（プロトタイプの管理）

```typescript
// Prototype Registry: テンプレートを管理
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

// 使用例: フォームテンプレート
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

// レジストリに定義済みフォームテンプレートを登録
const formRegistry = new PrototypeRegistry<FormTemplate>();

formRegistry.register("contact", new FormTemplate(
  "お問い合わせ",
  "お問い合わせフォーム",
  [
    { name: "name", type: "text", label: "氏名", required: true },
    { name: "email", type: "email", label: "メール", required: true },
    { name: "category", type: "select", label: "カテゴリ", required: true,
      options: ["一般", "技術", "営業", "その他"] },
    { name: "message", type: "textarea", label: "メッセージ", required: true },
  ],
));

formRegistry.register("survey", new FormTemplate(
  "アンケート",
  "顧客満足度アンケート",
  [
    { name: "satisfaction", type: "select", label: "満足度", required: true,
      options: ["非常に満足", "満足", "普通", "不満", "非常に不満"] },
    { name: "feedback", type: "textarea", label: "ご意見", required: false },
  ],
));

// テンプレートからカスタマイズ
const customContact = formRegistry.create("contact");
customContact.name = "技術サポート問い合わせ";
customContact.addField({
  name: "product",
  type: "select",
  label: "製品",
  required: true,
  options: ["製品A", "製品B", "製品C"],
});
```

### 5.4 JavaScript のスプレッド構文とStructured Clone

```typescript
// JavaScript/TypeScript での現代的なコピー手法

// 1. スプレッド構文（シャローコピー）
const original = { name: "太郎", tags: ["developer", "manager"] };
const shallow = { ...original };
shallow.tags.push("admin");  // 注意: original.tags も変更される！

// 2. Object.assign（シャローコピー）
const assigned = Object.assign({}, original);

// 3. JSON.parse/JSON.stringify（ディープコピー、制限あり）
const jsonCopy = JSON.parse(JSON.stringify(original));
// ⚠️ 制限: Date, Map, Set, RegExp, 関数, undefined は正しくコピーされない

// 4. structuredClone（モダンなディープコピー、Node.js 17+）
const structuredCopy = structuredClone(original);
// ✅ Date, Map, Set, ArrayBuffer, RegExp を正しくコピー
// ❌ 関数, DOM ノード, Error オブジェクトは不可

// 5. 実践的なディープコピーユーティリティ
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

## 6. Object Pool パターン

### 6.1 概要と目的

```
目的: 生成コストの高いオブジェクトを再利用する

いつ使うか:
  → オブジェクトの生成/破棄コストが高い（DB接続、スレッド）
  → 同時に必要なオブジェクト数が限られている
  → オブジェクトが再利用可能（状態をリセットできる）
```

### 6.2 接続プールの実装

```typescript
// Object Pool: データベース接続プール
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
    // 最小数のオブジェクトを事前生成
    for (let i = 0; i < minSize; i++) {
      this.available.push(factory());
    }
  }

  async acquire(): Promise<T> {
    // 利用可能なオブジェクトがあれば返す
    while (this.available.length > 0) {
      const obj = this.available.pop()!;
      if (obj.isValid()) {
        this.inUse.add(obj);
        return obj;
      }
      obj.destroy(); // 無効なオブジェクトは破棄
    }

    // 最大数に達していなければ新規生成
    if (this.inUse.size < this.maxSize) {
      const obj = this.factory();
      this.inUse.add(obj);
      return obj;
    }

    // 空きを待つ
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

    // 待機中のリクエストがあれば渡す
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

  // using パターン（確実にリリースする）
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

// 使用例
const dbPool = new ConnectionPool(
  () => new DatabaseConnection("postgres://localhost:5432/mydb"),
  { minSize: 5, maxSize: 20, acquireTimeoutMs: 3000 }
);

// using パターン（推奨）
const users = await dbPool.withConnection(async (conn) => {
  return conn.query("SELECT * FROM users WHERE active = true");
});

console.log(dbPool.stats);
// { available: 5, inUse: 0, waiting: 0, total: 5, maxSize: 20 }
```

---

## 7. 選択指針

### 7.1 パターン比較

```
┌─────────────────┬──────────────────────────────────────────┐
│ パターン        │ 使う場面                                  │
├─────────────────┼──────────────────────────────────────────┤
│ Simple Factory  │ 型を実行時に決定、生成ロジック集約       │
│ Factory Method  │ サブクラスが生成物の型を決定             │
│ Abstract Factory│ 関連するオブジェクト群の一貫した生成     │
│ Builder         │ パラメータが多い複雑な構築               │
│ Step Builder    │ 必須パラメータの順序を型で強制           │
│ Singleton       │ 本当に1つだけ必要（慎重に）              │
│ Prototype       │ 既存オブジェクトを元にコピー             │
│ Object Pool     │ 生成コストが高いオブジェクトの再利用     │
└─────────────────┴──────────────────────────────────────────┘
```

### 7.2 判断フローチャート

```
オブジェクト生成が必要
├── 生成する型が実行時に決まる
│   ├── 関連するオブジェクト群 → Abstract Factory
│   └── 単一のオブジェクト → Factory Method / Simple Factory
├── パラメータが多い or 段階的に構築
│   ├── 必須パラメータの順序を強制したい → Step Builder
│   └── 柔軟な構築 → Fluent Builder
├── 既存オブジェクトをコピーしたい → Prototype
├── インスタンス数を制限したい
│   ├── 1つだけ → Singleton（※DI推奨）
│   └── N個まで → Object Pool
└── シンプルな生成 → コンストラクタ直接呼び出し
```

---

## まとめ

| パターン | 目的 | 注意点 |
|---------|------|--------|
| Simple Factory | 生成ロジックの集約 | 過剰なFactory化を避ける |
| Factory Method | サブクラスが型を決定 | テンプレートメソッドと併用 |
| Abstract Factory | 製品ファミリーの切替 | インターフェース数が増大しがち |
| Builder | 段階的構築 | 単純なクラスには不要 |
| Step Builder | 型安全な段階構築 | インターフェース数が増える |
| Singleton | 唯一性の保証 | DIで代替を検討 |
| Prototype | コピー生成 | deep/shallow コピーに注意 |
| Object Pool | オブジェクト再利用 | リリース忘れに注意 |

---

## 次に読むべきガイド
→ [[01-structural-patterns.md]] — 構造パターン

---

## 参考文献
1. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software." Addison-Wesley, 1994.
2. Bloch, J. "Effective Java." 3rd Ed, Item 2: Consider a builder when faced with many constructor parameters. 2018.
3. Freeman, E. et al. "Head First Design Patterns." O'Reilly, 2020.
4. Martin, R.C. "Clean Code." Chapter 3: Functions. Prentice Hall, 2008.
5. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.
