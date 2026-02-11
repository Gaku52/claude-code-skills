# 生成パターン（Creational Patterns）

> オブジェクトの生成方法を柔軟にする設計パターン。Factory、Builder、Singleton、Prototype の4つの主要パターンの「なぜ必要か」「いつ使うか」を実践的に解説。

## この章で学ぶこと

- [ ] 各生成パターンの目的と使い分けを理解する
- [ ] 各パターンの実装方法を把握する
- [ ] アンチパターンとしての Singleton の問題を学ぶ

---

## 1. Factory パターン

```
目的: オブジェクトの生成ロジックをカプセル化する

いつ使うか:
  → 生成するクラスを実行時に決定したい
  → 生成ロジックが複雑
  → new を直接使わせたくない
```

```typescript
// Factory Method
interface Notification {
  send(message: string): void;
}

class EmailNotification implements Notification {
  send(message: string) { console.log(`Email: ${message}`); }
}

class SmsNotification implements Notification {
  send(message: string) { console.log(`SMS: ${message}`); }
}

class SlackNotification implements Notification {
  send(message: string) { console.log(`Slack: ${message}`); }
}

// ファクトリ: 生成ロジックを集約
class NotificationFactory {
  static create(type: "email" | "sms" | "slack"): Notification {
    switch (type) {
      case "email": return new EmailNotification();
      case "sms": return new SmsNotification();
      case "slack": return new SlackNotification();
    }
  }
}

// 利用側は具象クラスを知らなくてよい
const notification = NotificationFactory.create("email");
notification.send("Hello!");
```

```typescript
// Abstract Factory: 関連するオブジェクト群を生成
interface UIFactory {
  createButton(): Button;
  createInput(): Input;
  createModal(): Modal;
}

class MaterialUIFactory implements UIFactory {
  createButton() { return new MaterialButton(); }
  createInput() { return new MaterialInput(); }
  createModal() { return new MaterialModal(); }
}

class AntDesignFactory implements UIFactory {
  createButton() { return new AntButton(); }
  createInput() { return new AntInput(); }
  createModal() { return new AntModal(); }
}

// テーマを変更するだけで全UIコンポーネントが切り替わる
function buildUI(factory: UIFactory) {
  const button = factory.createButton();
  const input = factory.createInput();
  // factory の実装によって Material UI か Ant Design が使われる
}
```

---

## 2. Builder パターン

```
目的: 複雑なオブジェクトの構築過程を分離する

いつ使うか:
  → コンストラクタの引数が多い（5個以上）
  → オプショナルなパラメータが多い
  → 段階的に構築したい
```

```typescript
// Builder パターン
class HttpRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: Record<string, string>;
  readonly body?: string;
  readonly timeout: number;
  readonly retries: number;

  private constructor(builder: HttpRequestBuilder) {
    this.method = builder.method;
    this.url = builder.url;
    this.headers = { ...builder.headers };
    this.body = builder.body;
    this.timeout = builder.timeout;
    this.retries = builder.retries;
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

  setTimeout(ms: number): this {
    this.timeout = ms;
    return this;
  }

  setRetries(n: number): this {
    this.retries = n;
    return this;
  }

  build(): HttpRequest {
    return new (HttpRequest as any)(this);
  }
}

// 可読性の高いオブジェクト構築
const request = HttpRequest.builder("POST", "https://api.example.com/users")
  .setHeader("Content-Type", "application/json")
  .setHeader("Authorization", "Bearer token123")
  .setBody(JSON.stringify({ name: "田中" }))
  .setTimeout(10000)
  .setRetries(3)
  .build();
```

---

## 3. Singleton パターン

```
目的: クラスのインスタンスが1つだけであることを保証する

注意: Singleton は「アンチパターン」として批判されることが多い

問題点:
  → グローバル状態 = テスト困難
  → 密結合 = 依存性注入の妨げ
  → 並行処理 = 競合状態のリスク

適切な用途:
  → ロガー、設定マネージャ（本当に1つでいい場合）
  → DIコンテナ側で「1つだけ」を制御する方が良い
```

```typescript
// Singleton（必要最小限の実装）
class AppConfig {
  private static instance: AppConfig;

  private constructor(
    public readonly dbUrl: string,
    public readonly apiKey: string,
    public readonly debug: boolean,
  ) {}

  static getInstance(): AppConfig {
    if (!AppConfig.instance) {
      AppConfig.instance = new AppConfig(
        process.env.DATABASE_URL ?? "localhost:5432",
        process.env.API_KEY ?? "",
        process.env.NODE_ENV !== "production",
      );
    }
    return AppConfig.instance;
  }

  // テスト用リセット
  static resetForTesting(): void {
    AppConfig.instance = undefined as any;
  }
}

// より良いアプローチ: DIコンテナでスコープ管理
// container.register(AppConfig, { scope: "singleton" });
```

---

## 4. Prototype パターン

```
目的: 既存オブジェクトをコピーして新しいオブジェクトを生成する

いつ使うか:
  → 生成コストが高い（DB/APIから構築）
  → テンプレートオブジェクトを元に微調整
```

```typescript
// Prototype パターン
interface Cloneable<T> {
  clone(): T;
}

class DocumentTemplate implements Cloneable<DocumentTemplate> {
  constructor(
    public title: string,
    public content: string,
    public styles: Record<string, string>,
    public metadata: Record<string, string>,
  ) {}

  clone(): DocumentTemplate {
    return new DocumentTemplate(
      this.title,
      this.content,
      { ...this.styles },      // シャローコピー
      { ...this.metadata },
    );
  }
}

// テンプレートからコピーして微調整
const template = new DocumentTemplate(
  "月次レポート",
  "## 概要\n...",
  { fontSize: "14px", fontFamily: "Noto Sans JP" },
  { author: "", department: "" },
);

const report = template.clone();
report.title = "2025年1月 月次レポート";
report.metadata.author = "田中太郎";
report.metadata.department = "開発部";
```

---

## 5. 選択指針

```
┌─────────────┬──────────────────────────────────┐
│ パターン    │ 使う場面                          │
├─────────────┼──────────────────────────────────┤
│ Factory     │ 型を実行時に決定したい           │
│ Builder     │ パラメータが多い複雑な構築       │
│ Singleton   │ 本当に1つだけ必要（慎重に）      │
│ Prototype   │ 既存オブジェクトを元にコピー     │
└─────────────┴──────────────────────────────────┘
```

---

## まとめ

| パターン | 目的 | 注意点 |
|---------|------|--------|
| Factory | 生成の柔軟性 | 過剰なFactory化を避ける |
| Builder | 段階的構築 | 単純なクラスには不要 |
| Singleton | 唯一性の保証 | DIで代替を検討 |
| Prototype | コピー生成 | deep/shallow コピーに注意 |

---

## 次に読むべきガイド
→ [[01-structural-patterns.md]] — 構造パターン

---

## 参考文献
1. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
