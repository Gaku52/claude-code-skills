# SRP（単一責任の原則）+ OCP（開放閉鎖の原則）

> SRPは「変更する理由を1つに」、OCPは「変更せずに拡張する」。この2つの原則が、保守性の高い設計の土台を作る。

## この章で学ぶこと

- [ ] SRP の「責任」の正しい定義を理解する
- [ ] OCP をポリモーフィズムで実現する方法を把握する
- [ ] 実践的なリファクタリング手法を学ぶ

---

## 1. SRP: 単一責任の原則

```
定義（Robert C. Martin）:
  「クラスを変更する理由は1つだけであるべき」

より正確な定義:
  「クラスは1つのアクター（利害関係者）に対してのみ責任を持つ」

  例:
    Employee クラスが以下を持つ場合:
    - calculatePay()    → CFO（経理部門）の責任
    - reportHours()     → COO（業務部門）の責任
    - save()            → CTO（技術部門）の責任

    → 3つのアクターに依存 = SRP 違反
    → 経理部門の要求変更が業務部門のコードに影響する可能性
```

### SRP リファクタリング

```typescript
// ❌ SRP違反: 複数の責任を持つクラス
class UserService {
  // 責任1: ユーザーの作成ロジック
  createUser(data: CreateUserDto): User {
    // バリデーション
    if (!data.email.includes("@")) throw new Error("Invalid email");
    if (data.password.length < 8) throw new Error("Password too short");

    // パスワードハッシュ化
    const hashedPassword = bcrypt.hashSync(data.password, 10);

    // DB保存
    const user = db.users.create({ ...data, password: hashedPassword });

    // メール送信
    const html = `<h1>Welcome ${data.name}!</h1>`;
    emailClient.send(data.email, "Welcome", html);

    // ログ
    logger.info(`User created: ${user.id}`);

    return user;
  }
}

// ✅ SRP適用: 各クラスが1つの責任を持つ
class UserValidator {
  validate(data: CreateUserDto): void {
    if (!data.email.includes("@")) throw new ValidationError("Invalid email");
    if (data.password.length < 8) throw new ValidationError("Password too short");
  }
}

class PasswordHasher {
  hash(password: string): string {
    return bcrypt.hashSync(password, 10);
  }
}

class UserRepository {
  create(data: CreateUserDto & { password: string }): User {
    return db.users.create(data);
  }
}

class WelcomeEmailSender {
  send(user: User): void {
    const html = `<h1>Welcome ${user.name}!</h1>`;
    emailClient.send(user.email, "Welcome", html);
  }
}

// オーケストレーター
class UserRegistrationService {
  constructor(
    private validator: UserValidator,
    private hasher: PasswordHasher,
    private repo: UserRepository,
    private emailSender: WelcomeEmailSender,
  ) {}

  async register(data: CreateUserDto): Promise<User> {
    this.validator.validate(data);
    const hashedPassword = this.hasher.hash(data.password);
    const user = await this.repo.create({ ...data, password: hashedPassword });
    this.emailSender.send(user);
    return user;
  }
}
```

---

## 2. OCP: 開放閉鎖の原則

```
定義:
  「ソフトウェアの構成要素は、拡張に対して開き（Open）、
   修正に対して閉じている（Closed）べき」

つまり:
  → 新しい機能を追加するとき、既存のコードを変更しない
  → ポリモーフィズム（インターフェース + 実装クラス）で実現

なぜ重要か:
  → 既存コードを変更するとリグレッションのリスク
  → テスト済みのコードに触らずに済む
  → チーム開発でのコンフリクト減少
```

### OCP リファクタリング

```typescript
// ❌ OCP違反: 新しい通知手段を追加するたびに修正が必要
class NotificationService {
  send(type: string, message: string, recipient: string): void {
    if (type === "email") {
      // メール送信処理
      emailClient.send(recipient, message);
    } else if (type === "sms") {
      // SMS送信処理
      smsClient.send(recipient, message);
    } else if (type === "slack") {
      // Slack送信処理（新規追加するたびにここを修正）
      slackClient.post(recipient, message);
    }
    // LINE追加？ Discord追加？ → ここを修正し続ける...
  }
}

// ✅ OCP適用: 新しい通知手段はクラスを追加するだけ
interface NotificationChannel {
  send(message: string, recipient: string): Promise<void>;
}

class EmailChannel implements NotificationChannel {
  async send(message: string, recipient: string): Promise<void> {
    await emailClient.send(recipient, message);
  }
}

class SmsChannel implements NotificationChannel {
  async send(message: string, recipient: string): Promise<void> {
    await smsClient.send(recipient, message);
  }
}

class SlackChannel implements NotificationChannel {
  async send(message: string, recipient: string): Promise<void> {
    await slackClient.post(recipient, message);
  }
}

// LINE追加 → LineChannel クラスを追加するだけ
// NotificationService は一切変更不要

class NotificationService {
  constructor(private channels: NotificationChannel[]) {}

  async sendAll(message: string, recipient: string): Promise<void> {
    await Promise.all(
      this.channels.map(ch => ch.send(message, recipient))
    );
  }
}
```

### OCP のもう一つの実現方法: デコレータ

```python
# Python: デコレータによるOCP
class Logger:
    """既存クラスを変更せずにログ機能を追加"""
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        original = getattr(self._wrapped, name)
        if callable(original):
            def wrapper(*args, **kwargs):
                print(f"[LOG] {name} called with {args}")
                result = original(*args, **kwargs)
                print(f"[LOG] {name} returned {result}")
                return result
            return wrapper
        return original

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

# Calculator を変更せずにログ機能を追加
calc = Logger(Calculator())
calc.add(1, 2)
# [LOG] add called with (1, 2)
# [LOG] add returned 3
```

---

## 3. SRP と OCP の関係

```
SRP → クラスを小さく分割
  ↓
OCP → 小さなクラスをインターフェースで接続
  ↓
結果: 拡張が容易で変更の影響が局所的な設計

実践の流れ:
  1. SRP で責任を分離
  2. 変化しやすい部分を特定
  3. OCP でインターフェースを設計
  4. 新しい要件はクラスを追加して対応
```

---

## 4. アンチパターンと注意点

```
SRP の過剰適用:
  → 1メソッドだけのクラスが大量発生
  → ファイル数が爆発してナビゲーション困難
  → 対策: 「変更する理由」で分割。メソッド数ではない

OCP の過剰適用:
  → 変更されない部分まで抽象化
  → 不要なインターフェースだらけ
  → 対策: 「実際に変更が発生してから」抽象化する

判断基準:
  「このクラスが変更される理由は何か？」
  → 理由が複数ある → SRP で分割
  「この部分は今後変更される可能性があるか？」
  → ある → OCP でインターフェースを導入
  → ない → そのままでよい（YAGNI）
```

---

## まとめ

| 原則 | 核心 | 実現手段 | 注意 |
|------|------|---------|------|
| SRP | 1クラス1責任 | 責任の分離、委譲 | 過剰分割に注意 |
| OCP | 拡張は開、修正は閉 | インターフェース、ポリモーフィズム | 必要になってから抽象化 |

---

## 次に読むべきガイド
→ [[02-lsp-and-isp.md]] — LSP + ISP

---

## 参考文献
1. Martin, R. "Clean Architecture." Chapter 7-8, 2017.
2. Martin, R. "The Single Responsibility Principle." blog, 2014.
