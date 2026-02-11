# Facade パターン

> 複雑なサブシステム群に対する **統一された簡潔なインタフェース** を提供し、利用側の負担を軽減する構造パターン。

---

## この章で学ぶこと

1. Facade パターンの目的と、サブシステムの複雑さを隠蔽する設計手法
2. Facade が適切な場面と、過度に厚い Facade（God Facade）の回避
3. Facade と Adapter・Mediator の違い

---

## 1. Facade の構造

```
Client
  │
  ▼
┌──────────────────────┐
│       Facade         │
│ + simpleOperation()  │
└──────────┬───────────┘
           │ delegates
    ┌──────┼──────┐
    ▼      ▼      ▼
┌──────┐┌──────┐┌──────┐
│Sub A ││Sub B ││Sub C │
│      ││      ││      │
└──────┘└──────┘└──────┘
  サブシステム群（複雑な内部構造）
```

---

## 2. コード例

### コード例 1: ホームシアター Facade

```typescript
// 複雑なサブシステム群
class Projector {
  on(): void { console.log("Projector ON"); }
  setInput(src: string): void { console.log(`Input: ${src}`); }
}
class AudioSystem {
  on(): void { console.log("Audio ON"); }
  setVolume(v: number): void { console.log(`Volume: ${v}`); }
  setSurround(): void { console.log("Surround ON"); }
}
class StreamingPlayer {
  on(): void { console.log("Player ON"); }
  play(movie: string): void { console.log(`Playing: ${movie}`); }
}
class Lights {
  dim(level: number): void { console.log(`Lights: ${level}%`); }
}

// Facade
class HomeTheaterFacade {
  constructor(
    private projector: Projector,
    private audio: AudioSystem,
    private player: StreamingPlayer,
    private lights: Lights,
  ) {}

  watchMovie(movie: string): void {
    this.lights.dim(10);
    this.projector.on();
    this.projector.setInput("HDMI1");
    this.audio.on();
    this.audio.setSurround();
    this.audio.setVolume(50);
    this.player.on();
    this.player.play(movie);
  }

  endMovie(): void {
    console.log("Shutting down...");
  }
}

// クライアントは1メソッドを呼ぶだけ
const theater = new HomeTheaterFacade(
  new Projector(), new AudioSystem(),
  new StreamingPlayer(), new Lights()
);
theater.watchMovie("Inception");
```

### コード例 2: デプロイメント Facade

```typescript
class GitService {
  pull(branch: string): void { /* ... */ }
  tag(version: string): void { /* ... */ }
}
class BuildService {
  install(): void { /* ... */ }
  build(): void { /* ... */ }
  test(): void { /* ... */ }
}
class DeployService {
  upload(artifact: string): void { /* ... */ }
  activate(version: string): void { /* ... */ }
}
class NotifyService {
  sendSlack(msg: string): void { /* ... */ }
}

class DeployFacade {
  constructor(
    private git: GitService,
    private build: BuildService,
    private deploy: DeployService,
    private notify: NotifyService,
  ) {}

  async release(version: string): Promise<void> {
    this.git.pull("main");
    this.build.install();
    this.build.test();
    this.build.build();
    this.deploy.upload(`dist-${version}.tar.gz`);
    this.deploy.activate(version);
    this.git.tag(version);
    this.notify.sendSlack(`v${version} deployed`);
  }
}
```

### コード例 3: Python — API Facade

```python
class UserRepository:
    def find(self, user_id: int) -> dict: ...
    def save(self, user: dict) -> None: ...

class EmailService:
    def send(self, to: str, subject: str, body: str) -> None: ...

class AuditLogger:
    def log(self, action: str, user_id: int) -> None: ...

class UserFacade:
    """ユーザー操作の統一インタフェース"""
    def __init__(self, repo: UserRepository, email: EmailService, audit: AuditLogger):
        self._repo = repo
        self._email = email
        self._audit = audit

    def register(self, name: str, email_addr: str) -> dict:
        user = {"name": name, "email": email_addr}
        self._repo.save(user)
        self._email.send(email_addr, "Welcome!", f"Hello {name}")
        self._audit.log("REGISTER", user.get("id", 0))
        return user
```

### コード例 4: React — カスタムフック as Facade

```typescript
// 複雑な状態管理を Facade で隠蔽
function useCheckout() {
  const cart = useCart();
  const payment = usePayment();
  const shipping = useShipping();
  const [isProcessing, setIsProcessing] = useState(false);

  const checkout = async () => {
    setIsProcessing(true);
    try {
      const shippingCost = await shipping.calculate(cart.items);
      const total = cart.total + shippingCost;
      await payment.charge(total);
      await cart.clear();
    } finally {
      setIsProcessing(false);
    }
  };

  return { checkout, isProcessing, total: cart.total };
}

// コンポーネントはシンプルに
function CheckoutButton() {
  const { checkout, isProcessing } = useCheckout();
  return <button onClick={checkout} disabled={isProcessing}>購入</button>;
}
```

### コード例 5: モジュールの公開 API as Facade

```typescript
// internal/
//   parser.ts, validator.ts, transformer.ts, emitter.ts

// index.ts — モジュールの Facade
import { Parser } from "./internal/parser";
import { Validator } from "./internal/validator";
import { Transformer } from "./internal/transformer";
import { Emitter } from "./internal/emitter";

export function compile(source: string): string {
  const ast = new Parser().parse(source);
  new Validator().validate(ast);
  const ir = new Transformer().transform(ast);
  return new Emitter().emit(ir);
}

// 利用側は内部を知る必要がない
// import { compile } from "my-compiler";
```

---

## 3. Facade 適用の判断フロー

```
サブシステムが複雑？
      |
      Yes
      |
クライアントが複数の
サブシステムを直接操作？
      |
      Yes
      |
操作手順が定型的？
  |            |
  Yes          No（自由な組み合わせが必要）
  |            |
  v            v
Facade       サブシステムを直接公開
を導入       （Facade は不適切）
```

---

## 4. 比較表

### 比較表 1: Facade vs Adapter vs Mediator

| 観点 | Facade | Adapter | Mediator |
|------|--------|---------|----------|
| 目的 | 複雑さの隠蔽 | インタフェース変換 | オブジェクト間の調停 |
| 対象数 | 多数のサブシステム | 1つ | 多数のコンポーネント |
| 方向 | 一方向（Client→Sub） | 一方向 | 双方向 |
| 新しいインタフェース | 作る | 変換する | 作る |

### 比較表 2: Facade のレベル別設計

| レベル | 例 | 粒度 |
|--------|-----|------|
| モジュール | `index.ts` の re-export | 細粒度 |
| サービス | `UserFacade` | 中粒度 |
| アプリケーション | API Gateway | 粗粒度 |
| インフラ | CDK/Terraform wrapper | 最粗粒度 |

---

## 5. アンチパターン

### アンチパターン 1: God Facade

```typescript
// BAD: あらゆる操作を1つの Facade に詰め込む
class AppFacade {
  createUser() { /* ... */ }
  deleteUser() { /* ... */ }
  createOrder() { /* ... */ }
  processPayment() { /* ... */ }
  sendEmail() { /* ... */ }
  generateReport() { /* ... */ }
  // ... 50メソッド
}
```

**問題**: 単一責任原則に違反。Facade 自体が複雑になり本末転倒。

**改善**: ドメインごとに Facade を分割する（UserFacade, OrderFacade, ReportFacade）。

### アンチパターン 2: Facade がサブシステムのアクセスを完全に遮断

```typescript
// BAD: サブシステムへの直接アクセスを禁止
class StrictFacade {
  private subsystem: SubSystem; // private で完全隠蔽
  // サブシステムの全機能を再実装...
}
```

**問題**: Facade はあくまで「便利なショートカット」。高度なユースケースではサブシステムに直接アクセスできるべき。

---

## 6. FAQ

### Q1: Facade は API Gateway と同じですか？

概念は似ていますが、API Gateway はネットワーク境界で動作するインフラコンポーネントです。Facade はアプリケーション内のコード構造パターンです。API Gateway は Facade パターンの大規模な適用と言えます。

### Q2: Facade を使うとテストが難しくなりませんか？

Facade 自体は薄いオーケストレーション層なので、サブシステムをモック化すれば容易にテストできます。DI でサブシステムを注入する設計にしておくことが重要です。

### Q3: React のカスタムフックは Facade ですか？

はい、複数の Hook や状態管理を内部に隠蔽し、コンポーネントにシンプルな API を提供する点で Facade パターンの一形態です。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | 複雑なサブシステムに統一インタフェースを提供 |
| 利点 | クライアントコードの簡素化、疎結合 |
| 注意 | God Facade にしない、アクセスを遮断しない |
| 適用場面 | 定型的な操作手順、モジュール公開 API |
| 粒度 | モジュール〜インフラまで多段階で適用可 |

---

## 次に読むべきガイド

- [Proxy パターン](./03-proxy.md) — アクセス制御
- [Mediator パターン](../02-behavioral/00-observer.md) — オブジェクト間の調停
- [クリーンアーキテクチャ](../../../system-design-guide/docs/02-architecture/01-clean-architecture.md) — 層構造設計

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
3. Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley.
