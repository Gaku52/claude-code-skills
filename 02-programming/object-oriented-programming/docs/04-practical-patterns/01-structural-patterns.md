# 構造パターン（Structural Patterns）

> クラスやオブジェクトの組み合わせ方に関するパターン。Adapter、Decorator、Facade、Proxy、Composite の5つを実践的に解説。

## この章で学ぶこと

- [ ] 各構造パターンの目的と適用場面を理解する
- [ ] 各パターンのコード実装を把握する
- [ ] パターンの組み合わせと使い分けを学ぶ

---

## 1. Adapter パターン

```
目的: 互換性のないインターフェースを変換して接続する

  ┌─────────┐     ┌─────────┐     ┌──────────┐
  │ Client  │────→│ Adapter │────→│ Adaptee  │
  │         │     │ (変換)  │     │ (既存)   │
  └─────────┘     └─────────┘     └──────────┘
```

```typescript
// 既存のライブラリ（変更不可）
class LegacyPaymentGateway {
  processPayment(cardNumber: string, amount: number, currency: string): boolean {
    console.log(`Legacy: ${amount} ${currency} charged to ${cardNumber}`);
    return true;
  }
}

// 新しいインターフェース（自分たちの標準）
interface PaymentProcessor {
  pay(request: PaymentRequest): Promise<PaymentResult>;
}

interface PaymentRequest {
  amount: number;
  currency: string;
  method: { type: "card"; cardNumber: string };
}

// Adapter: 新旧を橋渡し
class LegacyPaymentAdapter implements PaymentProcessor {
  constructor(private legacy: LegacyPaymentGateway) {}

  async pay(request: PaymentRequest): Promise<PaymentResult> {
    const success = this.legacy.processPayment(
      request.method.cardNumber,
      request.amount,
      request.currency,
    );
    return { success, transactionId: crypto.randomUUID() };
  }
}

// 利用側は PaymentProcessor のみに依存
const processor: PaymentProcessor = new LegacyPaymentAdapter(
  new LegacyPaymentGateway()
);
```

---

## 2. Decorator パターン

```
目的: 既存オブジェクトに動的に機能を追加する

  ┌────────┐     ┌───────────┐     ┌───────────┐
  │ Client │────→│ Decorator │────→│ Component │
  │        │     │ (機能追加)│     │ (元)      │
  └────────┘     └───────────┘     └───────────┘

  複数のDecoratorを重ねられる（入れ子）
```

```typescript
// コンポーネントインターフェース
interface HttpClient {
  request(url: string, options?: RequestInit): Promise<Response>;
}

// 基本実装
class BasicHttpClient implements HttpClient {
  async request(url: string, options?: RequestInit): Promise<Response> {
    return fetch(url, options);
  }
}

// Decorator: ログ追加
class LoggingHttpClient implements HttpClient {
  constructor(private wrapped: HttpClient) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    console.log(`[HTTP] ${options?.method ?? "GET"} ${url}`);
    const start = Date.now();
    const response = await this.wrapped.request(url, options);
    console.log(`[HTTP] ${response.status} (${Date.now() - start}ms)`);
    return response;
  }
}

// Decorator: リトライ追加
class RetryHttpClient implements HttpClient {
  constructor(private wrapped: HttpClient, private maxRetries: number = 3) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    for (let i = 0; i <= this.maxRetries; i++) {
      try {
        return await this.wrapped.request(url, options);
      } catch (error) {
        if (i === this.maxRetries) throw error;
        await new Promise(r => setTimeout(r, 1000 * (i + 1)));
      }
    }
    throw new Error("Unreachable");
  }
}

// Decorator: 認証ヘッダー追加
class AuthHttpClient implements HttpClient {
  constructor(private wrapped: HttpClient, private token: string) {}

  async request(url: string, options?: RequestInit): Promise<Response> {
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${this.token}`);
    return this.wrapped.request(url, { ...options, headers });
  }
}

// デコレータを重ねる
const client = new LoggingHttpClient(
  new RetryHttpClient(
    new AuthHttpClient(
      new BasicHttpClient(),
      "my-token"
    ),
    3
  )
);
// リクエスト → Auth → Retry → Logging の順で処理
```

---

## 3. Facade パターン

```
目的: 複雑なサブシステムにシンプルなインターフェースを提供する

  ┌────────┐     ┌──────────┐     ┌─────┐ ┌─────┐ ┌─────┐
  │ Client │────→│  Facade  │────→│ SubA│ │ SubB│ │ SubC│
  │        │     │ (窓口)   │     └─────┘ └─────┘ └─────┘
  └────────┘     └──────────┘
```

```typescript
// 複雑なサブシステム
class VideoDecoder { decode(file: string): Buffer { return Buffer.alloc(0); } }
class AudioDecoder { decode(file: string): Buffer { return Buffer.alloc(0); } }
class SubtitleParser { parse(file: string): string[] { return []; } }
class VideoRenderer { render(video: Buffer, audio: Buffer, subs: string[]): void {} }

// Facade: シンプルなAPI
class MediaPlayer {
  private videoDecoder = new VideoDecoder();
  private audioDecoder = new AudioDecoder();
  private subtitleParser = new SubtitleParser();
  private renderer = new VideoRenderer();

  // 複雑な内部処理を1つのメソッドに
  play(videoFile: string, subtitleFile?: string): void {
    const video = this.videoDecoder.decode(videoFile);
    const audio = this.audioDecoder.decode(videoFile);
    const subs = subtitleFile ? this.subtitleParser.parse(subtitleFile) : [];
    this.renderer.render(video, audio, subs);
  }
}

// 利用側はシンプル
const player = new MediaPlayer();
player.play("movie.mp4", "movie.srt");
```

---

## 4. Proxy パターン

```
目的: オブジェクトへのアクセスを制御する代理を提供する

種類:
  → 仮想Proxy: 遅延初期化（重いオブジェクトを必要時に生成）
  → 保護Proxy: アクセス制御（権限チェック）
  → キャッシュProxy: 結果のキャッシュ
```

```typescript
// キャッシュProxy
interface DataService {
  fetchUser(id: string): Promise<User>;
}

class RealDataService implements DataService {
  async fetchUser(id: string): Promise<User> {
    // 重いDB/APIアクセス
    console.log(`Fetching user ${id} from database...`);
    return { id, name: "User " + id };
  }
}

class CachingProxy implements DataService {
  private cache = new Map<string, { data: User; expiry: number }>();

  constructor(
    private real: DataService,
    private ttlMs: number = 60000,
  ) {}

  async fetchUser(id: string): Promise<User> {
    const cached = this.cache.get(id);
    if (cached && cached.expiry > Date.now()) {
      console.log(`Cache hit for user ${id}`);
      return cached.data;
    }

    const data = await this.real.fetchUser(id);
    this.cache.set(id, { data, expiry: Date.now() + this.ttlMs });
    return data;
  }
}

const service: DataService = new CachingProxy(new RealDataService());
await service.fetchUser("123"); // DB アクセス
await service.fetchUser("123"); // キャッシュから
```

---

## 5. Composite パターン

```
目的: 個別オブジェクトとオブジェクトの集合を同一視して扱う

  Component（共通インターフェース）
  ├── Leaf（葉: 個別要素）
  └── Composite（枝: 子要素を含む）
```

```typescript
// ファイルシステムの例
interface FileSystemEntry {
  name: string;
  size(): number;
  display(indent?: string): string;
}

class File implements FileSystemEntry {
  constructor(public name: string, private bytes: number) {}

  size(): number { return this.bytes; }
  display(indent = ""): string { return `${indent}${this.name} (${this.bytes}B)`; }
}

class Directory implements FileSystemEntry {
  private children: FileSystemEntry[] = [];

  constructor(public name: string) {}

  add(entry: FileSystemEntry): void { this.children.push(entry); }

  size(): number {
    return this.children.reduce((sum, child) => sum + child.size(), 0);
  }

  display(indent = ""): string {
    const lines = [`${indent}${this.name}/`];
    for (const child of this.children) {
      lines.push(child.display(indent + "  "));
    }
    return lines.join("\n");
  }
}

// File と Directory を統一的に扱える
const root = new Directory("src");
const components = new Directory("components");
components.add(new File("Button.tsx", 2048));
components.add(new File("Modal.tsx", 4096));
root.add(components);
root.add(new File("index.ts", 512));

console.log(root.display());
console.log(`Total: ${root.size()}B`);
```

---

## まとめ

| パターン | 目的 | キーワード |
|---------|------|-----------|
| Adapter | インターフェース変換 | 既存コードの統合 |
| Decorator | 動的な機能追加 | 重ね掛け、ミドルウェア |
| Facade | 複雑さの隠蔽 | シンプルなAPI |
| Proxy | アクセス制御 | キャッシュ、遅延、権限 |
| Composite | 個と集合の同一視 | ツリー構造 |

---

## 次に読むべきガイド
→ [[02-behavioral-patterns.md]] — 振る舞いパターン

---

## 参考文献
1. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
