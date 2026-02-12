# Facade パターン

> 複雑なサブシステム群に対する **統一された簡潔なインタフェース** を提供し、利用側の負担を軽減する構造パターン。

---

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| クラスとインタフェース | 基本 | TypeScript / Java / Python の OOP |
| 依存性注入 (DI) | 基本 | [DI パターン](../../../system-design-guide/docs/02-architecture/01-clean-architecture.md) |
| 単一責任原則 (SRP) | 基本 | [SOLID 原則](../../solid-guide/docs/00-srp.md) |
| モジュール設計 | 基本 | ES Modules / Python packages |
| async/await | 基本 | JavaScript / Python の非同期処理 |

---

## この章で学ぶこと

1. Facade パターンが解決する「サブシステム結合の爆発」問題と、その根本原因
2. Facade の構造と 4 つの適用レベル（モジュール / サービス / アプリケーション / インフラ）
3. Facade と Adapter / Mediator / Controller の明確な違い
4. 5 言語（TypeScript, Python, Java, Go, Kotlin）での実装パターン
5. God Facade / Leaky Facade / Rigid Facade の 3 大アンチパターンとその回避策

---

## 1. なぜ Facade が必要なのか（WHY）

### 1.1 Facade なしの世界

複雑なシステムでは、クライアントが複数のサブシステムを直接操作しなければならない。

```
┌──────────────────────────────────────────────────────────┐
│  Facade なし: クライアントがサブシステムを直接操作        │
│                                                          │
│   Client A ──┬──▶ SubSystem1.init()                     │
│              ├──▶ SubSystem2.configure()                 │
│              ├──▶ SubSystem3.connect()                   │
│              ├──▶ SubSystem1.validate()                  │
│              └──▶ SubSystem2.execute()                   │
│                                                          │
│   Client B ──┬──▶ SubSystem1.init()       ← 同じ手順を  │
│              ├──▶ SubSystem2.configure()    繰り返す     │
│              ├──▶ SubSystem3.connect()                   │
│              ├──▶ SubSystem1.validate()                  │
│              └──▶ SubSystem2.execute()                   │
│                                                          │
│   問題:                                                  │
│   - クライアントがサブシステムの内部構造を知る必要がある  │
│   - 操作手順の重複（DRY 違反）                           │
│   - サブシステム変更時に全クライアントを修正              │
│   - テストでモック対象が多すぎる                          │
└──────────────────────────────────────────────────────────┘
```

### 1.2 現実世界のアナロジー

**ホテルのコンシェルジュ** を考えてみよう。宿泊客がレストラン予約、タクシー手配、観光チケット購入をそれぞれ自分で行う代わりに、コンシェルジュに「今晩のディナーと明日の観光を手配してください」と言えば、すべてを調整してくれる。

- コンシェルジュ = **Facade**
- レストラン、タクシー会社、チケット窓口 = **サブシステム**
- 宿泊客 = **クライアント**

宿泊客はレストランの予約システムの使い方を知らなくてもよい。しかし、自分で直接レストランに電話することも可能である（Facade はアクセスを遮断しない）。

### 1.3 Facade ありの世界

```
┌──────────────────────────────────────────────────────────┐
│  Facade あり: クライアントは Facade だけを知る            │
│                                                          │
│   Client A ──▶ Facade.doOperation()                     │
│                    │                                     │
│   Client B ──▶ Facade.doOperation()                     │
│                    │                                     │
│                    ├──▶ SubSystem1.init()                │
│                    ├──▶ SubSystem2.configure()           │
│                    ├──▶ SubSystem3.connect()             │
│                    ├──▶ SubSystem1.validate()            │
│                    └──▶ SubSystem2.execute()             │
│                                                          │
│   利点:                                                  │
│   - クライアントは内部を知る必要がない                    │
│   - 手順の一元管理（DRY）                                │
│   - サブシステム変更の影響が Facade に局所化              │
│   - テストでは Facade のみモック可能                      │
└──────────────────────────────────────────────────────────┘
```

### 1.4 Facade パターンの本質

Facade の本質は「**情報隠蔽**」と「**手順のカプセル化**」である。

1. **情報隠蔽**: クライアントはサブシステムの存在すら知らなくてよい
2. **手順のカプセル化**: 定型的な操作手順を 1 メソッドに集約
3. **疎結合**: クライアントとサブシステムの結合度を下げる
4. **ショートカット**: 便利な入口を提供するが、直接アクセスを禁止しない

> **重要**: Facade は「壁」ではなく「門」である。高度なユースケースではサブシステムへの直接アクセスを許容すべきである。

---

## 2. Facade の構造

### 2.1 クラス図

```
┌─────────────────────────────────────────────────────────┐
│                      UML クラス図                        │
│                                                         │
│  ┌──────────┐         ┌──────────────────┐              │
│  │  Client  │────────▶│     Facade       │              │
│  └──────────┘         │                  │              │
│                       │ - subA: SubA     │              │
│                       │ - subB: SubB     │              │
│                       │ - subC: SubC     │              │
│                       │                  │              │
│                       │ + operation()    │              │
│                       │ + anotherOp()    │              │
│                       └────────┬─────────┘              │
│                                │ delegates              │
│                    ┌───────────┼───────────┐            │
│                    ▼           ▼           ▼            │
│              ┌──────────┐┌──────────┐┌──────────┐      │
│              │  SubA    ││  SubB    ││  SubC    │      │
│              │          ││          ││          │      │
│              │ + a1()   ││ + b1()   ││ + c1()   │      │
│              │ + a2()   ││ + b2()   ││ + c2()   │      │
│              └──────────┘└──────────┘└──────────┘      │
│                                                         │
│  ポイント:                                              │
│  - Client は Facade のみに依存                          │
│  - Facade はサブシステム群を集約（has-a）                │
│  - サブシステムは Facade の存在を知らない                │
│  - サブシステムへの直接アクセスも可能（optional）        │
└─────────────────────────────────────────────────────────┘
```

### 2.2 シーケンス図

```
┌─────────────────────────────────────────────────────────┐
│                    シーケンス図                           │
│                                                         │
│  Client        Facade         SubA    SubB    SubC      │
│    │              │             │       │       │        │
│    │ operation()  │             │       │       │        │
│    │─────────────▶│             │       │       │        │
│    │              │ a1()        │       │       │        │
│    │              │────────────▶│       │       │        │
│    │              │  result_a   │       │       │        │
│    │              │◀────────────│       │       │        │
│    │              │ b1(result_a)│       │       │        │
│    │              │─────────────────────▶       │        │
│    │              │  result_b   │       │       │        │
│    │              │◀─────────────────────       │        │
│    │              │ c1(result_a, result_b)      │        │
│    │              │────────────────────────────▶│        │
│    │              │  result_c   │       │       │        │
│    │              │◀────────────────────────────│        │
│    │  final_result│             │       │       │        │
│    │◀─────────────│             │       │       │        │
│    │              │             │       │       │        │
│                                                         │
│  ポイント: Facade がオーケストレーションを担当            │
└─────────────────────────────────────────────────────────┘
```

### 2.3 Facade 適用の判断フロー

```
┌─────────────────────────────────────────────────────────┐
│                  Facade 適用判断フロー                    │
│                                                         │
│  サブシステムが複数ある？                                │
│       │                                                 │
│       ├── No ──▶ 不要（単一サブシステムなら直接呼ぶ）   │
│       │                                                 │
│       └── Yes                                           │
│            │                                            │
│  クライアントが複数のサブシステムを                      │
│  直接操作している？                                      │
│       │                                                 │
│       ├── No ──▶ 不要（結合度が低い）                   │
│       │                                                 │
│       └── Yes                                           │
│            │                                            │
│  操作手順が定型的？                                      │
│       │                  │                              │
│       ├── Yes            └── No                         │
│       │                      │                          │
│       ▼                      ▼                          │
│   Facade を導入         サブシステムを直接公開           │
│   （手順を集約）        （柔軟性を重視）                 │
│       │                                                 │
│  1つの Facade に                                        │
│  収まる規模？                                            │
│       │                  │                              │
│       ├── Yes            └── No                         │
│       │                      │                          │
│       ▼                      ▼                          │
│   単一 Facade           ドメイン別に                     │
│                         Facade を分割                    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. コード例

### コード例 1: ホームシアター Facade（TypeScript）

```typescript
// === サブシステム群 ===

class Projector {
  on(): void { console.log("Projector: ON"); }
  off(): void { console.log("Projector: OFF"); }
  setInput(src: string): void { console.log(`Projector: Input set to ${src}`); }
  setResolution(res: string): void { console.log(`Projector: Resolution ${res}`); }
}

class AudioSystem {
  on(): void { console.log("Audio: ON"); }
  off(): void { console.log("Audio: OFF"); }
  setVolume(v: number): void { console.log(`Audio: Volume ${v}`); }
  setSurround(): void { console.log("Audio: Surround mode enabled"); }
  setStereo(): void { console.log("Audio: Stereo mode enabled"); }
}

class StreamingPlayer {
  on(): void { console.log("Player: ON"); }
  off(): void { console.log("Player: OFF"); }
  play(movie: string): void { console.log(`Player: Playing "${movie}"`); }
  pause(): void { console.log("Player: Paused"); }
  stop(): void { console.log("Player: Stopped"); }
}

class Lights {
  on(): void { console.log("Lights: ON (100%)"); }
  off(): void { console.log("Lights: OFF"); }
  dim(level: number): void { console.log(`Lights: Dimmed to ${level}%`); }
}

class Screen {
  down(): void { console.log("Screen: Lowered"); }
  up(): void { console.log("Screen: Raised"); }
}

// === Facade ===

class HomeTheaterFacade {
  constructor(
    private projector: Projector,
    private audio: AudioSystem,
    private player: StreamingPlayer,
    private lights: Lights,
    private screen: Screen,
  ) {}

  /** 映画鑑賞モード: 6つのサブシステムを正しい順序で操作 */
  watchMovie(movie: string): void {
    console.log("=== Setting up movie mode ===");
    this.lights.dim(10);
    this.screen.down();
    this.projector.on();
    this.projector.setInput("HDMI1");
    this.projector.setResolution("4K");
    this.audio.on();
    this.audio.setSurround();
    this.audio.setVolume(50);
    this.player.on();
    this.player.play(movie);
    console.log("=== Enjoy your movie! ===");
  }

  /** 映画終了: 全サブシステムを正しい順序でシャットダウン */
  endMovie(): void {
    console.log("=== Shutting down ===");
    this.player.stop();
    this.player.off();
    this.audio.off();
    this.projector.off();
    this.screen.up();
    this.lights.on();
    console.log("=== Done ===");
  }

  /** 音楽モード: 映像不要、ステレオ音声 */
  listenToMusic(): void {
    console.log("=== Setting up music mode ===");
    this.lights.dim(40);
    this.audio.on();
    this.audio.setStereo();
    this.audio.setVolume(30);
  }
}

// === 使用例 ===

const theater = new HomeTheaterFacade(
  new Projector(),
  new AudioSystem(),
  new StreamingPlayer(),
  new Lights(),
  new Screen(),
);

// クライアントは 1 メソッドを呼ぶだけ
theater.watchMovie("Inception");
// 出力:
// === Setting up movie mode ===
// Lights: Dimmed to 10%
// Screen: Lowered
// Projector: ON
// Projector: Input set to HDMI1
// Projector: Resolution 4K
// Audio: ON
// Audio: Surround mode enabled
// Audio: Volume 50
// Player: ON
// Player: Playing "Inception"
// === Enjoy your movie! ===

theater.endMovie();
```

**ポイント**: 6 つのサブシステムの操作順序（ライト暗転 → スクリーン → プロジェクター → オーディオ → プレーヤー）をクライアントが知る必要がない。

---

### コード例 2: デプロイメントパイプライン Facade（TypeScript）

```typescript
// === サブシステム群 ===

interface DeployResult {
  version: string;
  url: string;
  timestamp: Date;
}

class GitService {
  pull(branch: string): void {
    console.log(`Git: Pulling ${branch}`);
  }
  tag(version: string): void {
    console.log(`Git: Tagged ${version}`);
  }
  getLatestCommit(): string {
    return "abc1234";
  }
}

class BuildService {
  install(): void {
    console.log("Build: Installing dependencies");
  }
  lint(): void {
    console.log("Build: Linting code");
  }
  test(): void {
    console.log("Build: Running tests");
  }
  build(): string {
    console.log("Build: Creating production build");
    return "dist/app.tar.gz";
  }
}

class DeployService {
  upload(artifact: string, env: string): string {
    console.log(`Deploy: Uploading ${artifact} to ${env}`);
    return `https://${env}.example.com`;
  }
  activate(version: string): void {
    console.log(`Deploy: Activating version ${version}`);
  }
  healthCheck(url: string): boolean {
    console.log(`Deploy: Health check on ${url}`);
    return true;
  }
  rollback(version: string): void {
    console.log(`Deploy: Rolling back to ${version}`);
  }
}

class NotifyService {
  sendSlack(channel: string, msg: string): void {
    console.log(`Slack [${channel}]: ${msg}`);
  }
  sendEmail(to: string, subject: string): void {
    console.log(`Email to ${to}: ${subject}`);
  }
}

// === Facade ===

class DeployFacade {
  constructor(
    private git: GitService,
    private build: BuildService,
    private deploy: DeployService,
    private notify: NotifyService,
  ) {}

  /** 本番リリース: 全ステップを自動実行 */
  async release(version: string): Promise<DeployResult> {
    try {
      // 1. ソースコード取得
      this.git.pull("main");

      // 2. ビルドパイプライン
      this.build.install();
      this.build.lint();
      this.build.test();
      const artifact = this.build.build();

      // 3. デプロイ
      const url = this.deploy.upload(artifact, "production");
      this.deploy.activate(version);

      // 4. ヘルスチェック
      const healthy = this.deploy.healthCheck(url);
      if (!healthy) {
        this.deploy.rollback(version);
        throw new Error(`Health check failed for ${version}`);
      }

      // 5. 完了処理
      this.git.tag(version);
      this.notify.sendSlack("#deploys", `v${version} deployed to ${url}`);
      this.notify.sendEmail("team@example.com", `Release v${version} complete`);

      return { version, url, timestamp: new Date() };
    } catch (error) {
      this.notify.sendSlack("#alerts", `Deploy v${version} FAILED: ${error}`);
      throw error;
    }
  }

  /** ステージング環境へのデプロイ（テスト・通知なし） */
  async deployToStaging(branch: string): Promise<string> {
    this.git.pull(branch);
    this.build.install();
    const artifact = this.build.build();
    return this.deploy.upload(artifact, "staging");
  }
}

// === 使用例 ===

const pipeline = new DeployFacade(
  new GitService(),
  new BuildService(),
  new DeployService(),
  new NotifyService(),
);

await pipeline.release("1.2.0");
```

**ポイント**: エラーハンドリングを含むオーケストレーション全体が Facade に集約されている。ステージング用の簡易メソッドも提供し、用途に応じて使い分けられる。

---

### コード例 3: Python -- データ分析パイプライン Facade

```python
from dataclasses import dataclass
from typing import Any
import json
import csv
import io


# === サブシステム群 ===

class DataLoader:
    """様々な形式のデータを読み込む"""
    def load_csv(self, path: str) -> list[dict]:
        print(f"DataLoader: Loading CSV from {path}")
        # 実際の実装では csv.DictReader を使用
        return [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]

    def load_json(self, path: str) -> list[dict]:
        print(f"DataLoader: Loading JSON from {path}")
        return [{"name": "Charlie", "score": 85}]

    def load_database(self, query: str) -> list[dict]:
        print(f"DataLoader: Executing query: {query}")
        return [{"id": 1, "value": 100}]


class DataCleaner:
    """データのクレンジング・正規化"""
    def remove_nulls(self, data: list[dict]) -> list[dict]:
        print(f"DataCleaner: Removing nulls from {len(data)} records")
        return [row for row in data if all(v is not None for v in row.values())]

    def normalize(self, data: list[dict], columns: list[str]) -> list[dict]:
        print(f"DataCleaner: Normalizing columns {columns}")
        return data  # 実際は正規化処理

    def deduplicate(self, data: list[dict], key: str) -> list[dict]:
        print(f"DataCleaner: Deduplicating by {key}")
        seen = set()
        result = []
        for row in data:
            if row[key] not in seen:
                seen.add(row[key])
                result.append(row)
        return result


class DataAnalyzer:
    """統計分析・集計"""
    def aggregate(self, data: list[dict], column: str) -> dict:
        print(f"DataAnalyzer: Aggregating column '{column}'")
        values = [row.get(column, 0) for row in data]
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / max(len(values), 1),
            "min": min(values, default=0),
            "max": max(values, default=0),
        }

    def group_by(self, data: list[dict], key: str) -> dict[str, list[dict]]:
        print(f"DataAnalyzer: Grouping by '{key}'")
        groups: dict[str, list[dict]] = {}
        for row in data:
            k = str(row.get(key, "unknown"))
            groups.setdefault(k, []).append(row)
        return groups


class ReportGenerator:
    """レポート生成"""
    def generate_summary(self, stats: dict) -> str:
        print("ReportGenerator: Generating summary report")
        return json.dumps(stats, indent=2)

    def generate_csv(self, data: list[dict]) -> str:
        print("ReportGenerator: Generating CSV report")
        if not data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()


# === Facade ===

@dataclass
class AnalysisResult:
    raw_count: int
    clean_count: int
    stats: dict
    report: str


class DataPipelineFacade:
    """データ分析パイプラインの統一インタフェース"""

    def __init__(
        self,
        loader: DataLoader,
        cleaner: DataCleaner,
        analyzer: DataAnalyzer,
        reporter: ReportGenerator,
    ):
        self._loader = loader
        self._cleaner = cleaner
        self._analyzer = analyzer
        self._reporter = reporter

    def analyze_csv(
        self,
        path: str,
        target_column: str,
        dedup_key: str | None = None,
    ) -> AnalysisResult:
        """CSV ファイルを読み込み、クレンジング・分析・レポート生成を一括実行"""
        # 1. データ読み込み
        data = self._loader.load_csv(path)
        raw_count = len(data)

        # 2. クレンジング
        data = self._cleaner.remove_nulls(data)
        if dedup_key:
            data = self._cleaner.deduplicate(data, dedup_key)
        clean_count = len(data)

        # 3. 分析
        stats = self._analyzer.aggregate(data, target_column)

        # 4. レポート生成
        report = self._reporter.generate_summary(stats)

        return AnalysisResult(
            raw_count=raw_count,
            clean_count=clean_count,
            stats=stats,
            report=report,
        )

    def quick_summary(self, path: str, column: str) -> dict:
        """クレンジングなしの簡易集計"""
        data = self._loader.load_csv(path)
        return self._analyzer.aggregate(data, column)


# === 使用例 ===

pipeline = DataPipelineFacade(
    loader=DataLoader(),
    cleaner=DataCleaner(),
    analyzer=DataAnalyzer(),
    reporter=ReportGenerator(),
)

result = pipeline.analyze_csv("sales.csv", target_column="age", dedup_key="name")
print(f"Raw: {result.raw_count}, Clean: {result.clean_count}")
print(result.report)
# 出力:
# DataLoader: Loading CSV from sales.csv
# DataCleaner: Removing nulls from 2 records
# DataCleaner: Deduplicating by name
# DataAnalyzer: Aggregating column 'age'
# ReportGenerator: Generating summary report
# Raw: 2, Clean: 2
# {
#   "count": 2,
#   "sum": 55,
#   "avg": 27.5,
#   "min": 25,
#   "max": 30
# }
```

**ポイント**: 4 つのサブシステム（Loader, Cleaner, Analyzer, Reporter）の操作手順を `analyze_csv` に集約。簡易版 `quick_summary` も提供して、柔軟性と簡便性を両立している。

---

### コード例 4: Java -- メール送信 Facade

```java
// === サブシステム群 ===

class SmtpClient {
    public void connect(String host, int port) {
        System.out.println("SMTP: Connected to " + host + ":" + port);
    }
    public void authenticate(String user, String pass) {
        System.out.println("SMTP: Authenticated as " + user);
    }
    public void send(String from, String to, String raw) {
        System.out.println("SMTP: Sent from " + from + " to " + to);
    }
    public void disconnect() {
        System.out.println("SMTP: Disconnected");
    }
}

class MimeBuilder {
    private String subject = "";
    private String textBody = "";
    private String htmlBody = "";
    private final List<String> attachments = new ArrayList<>();

    public void setSubject(String subject) { this.subject = subject; }
    public void setTextBody(String body) { this.textBody = body; }
    public void setHtmlBody(String html) { this.htmlBody = html; }
    public void addAttachment(String path) { attachments.add(path); }

    public String build() {
        System.out.println("MIME: Building message (subject=" + subject
            + ", attachments=" + attachments.size() + ")");
        return "MIME-Version: 1.0\nSubject: " + subject + "\n\n" + textBody;
    }
}

class TemplateEngine {
    public String render(String templateName, Map<String, Object> vars) {
        System.out.println("Template: Rendering " + templateName);
        // 実際はテンプレートファイルを読み込み、変数を埋め込む
        return "<html><body>Hello " + vars.getOrDefault("name", "User") + "</body></html>";
    }
}

class AddressValidator {
    public boolean validate(String email) {
        boolean valid = email != null && email.contains("@");
        System.out.println("Validator: " + email + " -> " + (valid ? "OK" : "INVALID"));
        return valid;
    }
}

// === Facade ===

class EmailFacade {
    private final SmtpClient smtp;
    private final MimeBuilder mime;
    private final TemplateEngine templates;
    private final AddressValidator validator;
    private final String smtpHost;
    private final int smtpPort;
    private final String smtpUser;
    private final String smtpPass;

    public EmailFacade(String host, int port, String user, String pass) {
        this.smtp = new SmtpClient();
        this.mime = new MimeBuilder();
        this.templates = new TemplateEngine();
        this.validator = new AddressValidator();
        this.smtpHost = host;
        this.smtpPort = port;
        this.smtpUser = user;
        this.smtpPass = pass;
    }

    /** テンプレートを使ったメール送信（最も一般的なユースケース） */
    public void sendTemplated(
        String from, String to,
        String templateName, Map<String, Object> vars
    ) {
        if (!validator.validate(to)) {
            throw new IllegalArgumentException("Invalid email: " + to);
        }

        String html = templates.render(templateName, vars);

        mime.setSubject((String) vars.getOrDefault("subject", "No Subject"));
        mime.setHtmlBody(html);
        String raw = mime.build();

        smtp.connect(smtpHost, smtpPort);
        try {
            smtp.authenticate(smtpUser, smtpPass);
            smtp.send(from, to, raw);
        } finally {
            smtp.disconnect();
        }
    }

    /** プレーンテキストメールの簡易送信 */
    public void sendPlain(String from, String to, String subject, String body) {
        if (!validator.validate(to)) {
            throw new IllegalArgumentException("Invalid email: " + to);
        }

        mime.setSubject(subject);
        mime.setTextBody(body);
        String raw = mime.build();

        smtp.connect(smtpHost, smtpPort);
        try {
            smtp.authenticate(smtpUser, smtpPass);
            smtp.send(from, to, raw);
        } finally {
            smtp.disconnect();
        }
    }
}

// === 使用例 ===

EmailFacade email = new EmailFacade("smtp.example.com", 587, "user", "pass");

email.sendTemplated(
    "noreply@example.com",
    "user@example.com",
    "welcome",
    Map.of("name", "Taro", "subject", "Welcome!")
);
// 出力:
// Validator: user@example.com -> OK
// Template: Rendering welcome
// MIME: Building message (subject=Welcome!, attachments=0)
// SMTP: Connected to smtp.example.com:587
// SMTP: Authenticated as user
// SMTP: Sent from noreply@example.com to user@example.com
// SMTP: Disconnected
```

**ポイント**: SMTP 接続、MIME 構築、テンプレートレンダリング、アドレスバリデーションという 4 つの複雑なサブシステムを、`sendTemplated` と `sendPlain` の 2 メソッドに集約。

---

### コード例 5: Go -- HTTP サーバー Facade

```go
package main

import (
    "fmt"
    "log"
    "net/http"
    "time"
)

// === サブシステム群 ===

type Router struct {
    routes map[string]http.HandlerFunc
}

func NewRouter() *Router {
    return &Router{routes: make(map[string]http.HandlerFunc)}
}

func (r *Router) AddRoute(path string, handler http.HandlerFunc) {
    r.routes[path] = handler
    fmt.Printf("Router: Added route %s\n", path)
}

func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
    if handler, ok := r.routes[req.URL.Path]; ok {
        handler(w, req)
    } else {
        http.NotFound(w, req)
    }
}

type Middleware struct {
    handlers []func(http.Handler) http.Handler
}

func NewMiddleware() *Middleware {
    return &Middleware{}
}

func (m *Middleware) Use(mw func(http.Handler) http.Handler) {
    m.handlers = append(m.handlers, mw)
    fmt.Println("Middleware: Added middleware")
}

func (m *Middleware) Apply(handler http.Handler) http.Handler {
    for i := len(m.handlers) - 1; i >= 0; i-- {
        handler = m.handlers[i](handler)
    }
    return handler
}

type CORSConfig struct {
    AllowOrigins []string
    AllowMethods []string
}

func (c *CORSConfig) Apply(next http.Handler) http.Handler {
    fmt.Println("CORS: Configured")
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        next.ServeHTTP(w, r)
    })
}

type GracefulShutdown struct {
    server *http.Server
}

func NewGracefulShutdown(addr string, handler http.Handler) *GracefulShutdown {
    return &GracefulShutdown{
        server: &http.Server{
            Addr:         addr,
            Handler:      handler,
            ReadTimeout:  10 * time.Second,
            WriteTimeout: 10 * time.Second,
        },
    }
}

func (g *GracefulShutdown) Start() error {
    fmt.Printf("Server: Listening on %s\n", g.server.Addr)
    return g.server.ListenAndServe()
}

// === Facade ===

type ServerFacade struct {
    router     *Router
    middleware *Middleware
    cors       *CORSConfig
    addr       string
}

func NewServerFacade(addr string) *ServerFacade {
    return &ServerFacade{
        router:     NewRouter(),
        middleware: NewMiddleware(),
        cors:       &CORSConfig{},
        addr:       addr,
    }
}

// GET ルートを追加
func (s *ServerFacade) GET(path string, handler http.HandlerFunc) {
    s.router.AddRoute(path, handler)
}

// ミドルウェアを追加
func (s *ServerFacade) Use(mw func(http.Handler) http.Handler) {
    s.middleware.Use(mw)
}

// サーバー起動（CORS・ミドルウェア・グレースフルシャットダウンを自動設定）
func (s *ServerFacade) Start() error {
    var handler http.Handler = s.router
    handler = s.cors.Apply(handler)
    handler = s.middleware.Apply(handler)

    gs := NewGracefulShutdown(s.addr, handler)
    return gs.Start()
}

// === 使用例 ===

func main() {
    app := NewServerFacade(":8080")

    // ロギングミドルウェア
    app.Use(func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            log.Printf("%s %s", r.Method, r.URL.Path)
            next.ServeHTTP(w, r)
        })
    })

    app.GET("/health", func(w http.ResponseWriter, r *http.Request) {
        w.Write([]byte("OK"))
    })

    // Router, Middleware, CORS, GracefulShutdown を意識せずに起動
    app.Start()
}
```

**ポイント**: Router, Middleware, CORS, GracefulShutdown という 4 つのサブシステムを `ServerFacade` が統合。`app.GET()` と `app.Start()` だけでサーバーが立ち上がる。Express.js の設計思想と同じ。

---

### コード例 6: React -- カスタムフック as Facade（TypeScript）

```typescript
import { useState, useEffect, useCallback, useMemo } from "react";

// === サブシステム（個別の Hook） ===

function useCart() {
  const [items, setItems] = useState<CartItem[]>([]);

  const addItem = useCallback((item: CartItem) => {
    setItems(prev => [...prev, item]);
  }, []);

  const removeItem = useCallback((id: string) => {
    setItems(prev => prev.filter(item => item.id !== id));
  }, []);

  const clear = useCallback(() => setItems([]), []);

  const total = useMemo(
    () => items.reduce((sum, item) => sum + item.price * item.quantity, 0),
    [items],
  );

  return { items, addItem, removeItem, clear, total };
}

function usePayment() {
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const charge = useCallback(async (amount: number, method: string) => {
    setProcessing(true);
    setError(null);
    try {
      const res = await fetch("/api/payment", {
        method: "POST",
        body: JSON.stringify({ amount, method }),
      });
      if (!res.ok) throw new Error("Payment failed");
      return await res.json();
    } catch (e) {
      setError((e as Error).message);
      throw e;
    } finally {
      setProcessing(false);
    }
  }, []);

  return { charge, processing, error };
}

function useShipping() {
  const [address, setAddress] = useState<Address | null>(null);
  const [cost, setCost] = useState(0);

  const calculate = useCallback(async (items: CartItem[], addr: Address) => {
    const res = await fetch("/api/shipping/calculate", {
      method: "POST",
      body: JSON.stringify({ items, address: addr }),
    });
    const data = await res.json();
    setCost(data.cost);
    return data.cost;
  }, []);

  return { address, setAddress, cost, calculate };
}

function useCoupon() {
  const [code, setCode] = useState("");
  const [discount, setDiscount] = useState(0);

  const apply = useCallback(async (couponCode: string) => {
    const res = await fetch(`/api/coupons/${couponCode}`);
    if (!res.ok) throw new Error("Invalid coupon");
    const data = await res.json();
    setCode(couponCode);
    setDiscount(data.discount);
    return data.discount;
  }, []);

  return { code, discount, apply, setCode };
}

// === Facade Hook ===

interface CartItem {
  id: string;
  name: string;
  price: number;
  quantity: number;
}

interface Address {
  zip: string;
  city: string;
  line1: string;
}

interface CheckoutState {
  // カート
  items: CartItem[];
  addItem: (item: CartItem) => void;
  removeItem: (id: string) => void;
  // 金額
  subtotal: number;
  shippingCost: number;
  discount: number;
  total: number;
  // アクション
  applyCoupon: (code: string) => Promise<void>;
  setShippingAddress: (addr: Address) => Promise<void>;
  checkout: (paymentMethod: string) => Promise<void>;
  // 状態
  isProcessing: boolean;
  error: string | null;
  step: "cart" | "shipping" | "payment" | "complete";
}

function useCheckout(): CheckoutState {
  const cart = useCart();
  const payment = usePayment();
  const shipping = useShipping();
  const coupon = useCoupon();
  const [step, setStep] = useState<CheckoutState["step"]>("cart");

  const total = useMemo(
    () => Math.max(0, cart.total + shipping.cost - coupon.discount),
    [cart.total, shipping.cost, coupon.discount],
  );

  const applyCoupon = useCallback(async (code: string) => {
    await coupon.apply(code);
  }, [coupon]);

  const setShippingAddress = useCallback(async (addr: Address) => {
    shipping.setAddress(addr);
    await shipping.calculate(cart.items, addr);
    setStep("shipping");
  }, [cart.items, shipping]);

  const checkout = useCallback(async (paymentMethod: string) => {
    setStep("payment");
    await payment.charge(total, paymentMethod);
    cart.clear();
    setStep("complete");
  }, [total, payment, cart]);

  return {
    items: cart.items,
    addItem: cart.addItem,
    removeItem: cart.removeItem,
    subtotal: cart.total,
    shippingCost: shipping.cost,
    discount: coupon.discount,
    total,
    applyCoupon,
    setShippingAddress,
    checkout,
    isProcessing: payment.processing,
    error: payment.error,
    step,
  };
}

// === コンポーネントはシンプルに ===

function CheckoutPage() {
  const {
    items, subtotal, shippingCost, discount, total,
    checkout, isProcessing, error, step,
  } = useCheckout();

  if (step === "complete") return <div>Thank you!</div>;

  return (
    <div>
      <h2>Checkout ({items.length} items)</h2>
      <p>Subtotal: ${subtotal}</p>
      <p>Shipping: ${shippingCost}</p>
      <p>Discount: -${discount}</p>
      <p>Total: ${total}</p>
      {error && <p style={{ color: "red" }}>{error}</p>}
      <button onClick={() => checkout("credit_card")} disabled={isProcessing}>
        {isProcessing ? "Processing..." : "Pay Now"}
      </button>
    </div>
  );
}
```

**ポイント**: `useCheckout` が Cart, Payment, Shipping, Coupon の 4 つの Hook を統合し、コンポーネントに対して統一された API を提供。コンポーネントは「支払い処理の内部構造」を知らなくてよい。

---

### コード例 7: Kotlin -- Android ネットワーク Facade

```kotlin
// === サブシステム群 ===

class HttpClient {
    fun get(url: String, headers: Map<String, String> = emptyMap()): String {
        println("HTTP: GET $url")
        return """{"status": "ok"}"""
    }

    fun post(url: String, body: String, headers: Map<String, String> = emptyMap()): String {
        println("HTTP: POST $url body=$body")
        return """{"id": 1}"""
    }
}

class JsonParser {
    fun <T> parse(json: String, clazz: Class<T>): T {
        println("JSON: Parsing ${json.take(50)}...")
        @Suppress("UNCHECKED_CAST")
        return mapOf("status" to "ok") as T
    }

    fun toJson(obj: Any): String {
        println("JSON: Serializing $obj")
        return """{"serialized": true}"""
    }
}

class TokenManager {
    private var token: String? = null

    fun getToken(): String {
        return token ?: throw IllegalStateException("Not authenticated")
    }

    fun setToken(newToken: String) {
        token = newToken
        println("Token: Stored new token")
    }

    fun isAuthenticated(): Boolean = token != null

    fun clearToken() {
        token = null
        println("Token: Cleared")
    }
}

class CacheManager {
    private val cache = mutableMapOf<String, Pair<String, Long>>()
    private val ttl = 60_000L // 1分

    fun get(key: String): String? {
        val entry = cache[key] ?: return null
        if (System.currentTimeMillis() - entry.second > ttl) {
            cache.remove(key)
            return null
        }
        println("Cache: HIT for $key")
        return entry.first
    }

    fun put(key: String, value: String) {
        cache[key] = value to System.currentTimeMillis()
        println("Cache: Stored $key")
    }
}

// === Facade ===

class ApiClient(
    private val http: HttpClient = HttpClient(),
    private val json: JsonParser = JsonParser(),
    private val auth: TokenManager = TokenManager(),
    private val cache: CacheManager = CacheManager(),
    private val baseUrl: String = "https://api.example.com",
) {
    /** 認証ヘッダー付き GET（キャッシュあり） */
    fun <T> get(path: String, clazz: Class<T>, useCache: Boolean = true): T {
        val url = "$baseUrl$path"

        if (useCache) {
            cache.get(url)?.let { return json.parse(it, clazz) }
        }

        val headers = mapOf("Authorization" to "Bearer ${auth.getToken()}")
        val response = http.get(url, headers)

        if (useCache) cache.put(url, response)
        return json.parse(response, clazz)
    }

    /** 認証ヘッダー付き POST */
    fun <T> post(path: String, body: Any, clazz: Class<T>): T {
        val url = "$baseUrl$path"
        val headers = mapOf("Authorization" to "Bearer ${auth.getToken()}")
        val jsonBody = json.toJson(body)
        val response = http.post(url, jsonBody, headers)
        return json.parse(response, clazz)
    }

    /** ログイン（トークン取得・保存） */
    fun login(username: String, password: String) {
        val body = json.toJson(mapOf("username" to username, "password" to password))
        val response = http.post("$baseUrl/auth/login", body)
        val result = json.parse(response, Map::class.java)
        auth.setToken(result["token"] as? String ?: "dummy-token")
    }

    /** ログアウト */
    fun logout() {
        auth.clearToken()
    }
}

// === 使用例 ===

fun main() {
    val api = ApiClient()
    api.login("user", "pass")
    val result = api.get("/users/me", Map::class.java)
    println("Result: $result")
}
```

**ポイント**: HTTP 通信、JSON パース、認証トークン管理、キャッシュという 4 つのサブシステムを `ApiClient` が統合。Android の Retrofit ライブラリはこの設計思想を発展させたもの。

---

### コード例 8: モジュール公開 API as Facade（TypeScript）

```typescript
// === 内部モジュール群 ===
// internal/parser.ts
class Parser {
  parse(source: string): ASTNode {
    console.log("Parser: Parsing source code");
    return { type: "program", body: [] };
  }
}

// internal/validator.ts
class Validator {
  validate(ast: ASTNode): ValidationResult {
    console.log("Validator: Validating AST");
    return { valid: true, errors: [] };
  }
}

// internal/optimizer.ts
class Optimizer {
  optimize(ast: ASTNode): ASTNode {
    console.log("Optimizer: Optimizing AST");
    return ast;
  }
}

// internal/transformer.ts
class Transformer {
  transform(ast: ASTNode): IRNode {
    console.log("Transformer: Transforming to IR");
    return { type: "module", instructions: [] };
  }
}

// internal/emitter.ts
class Emitter {
  emit(ir: IRNode): string {
    console.log("Emitter: Generating output");
    return "compiled output";
  }
}

// === 型定義 ===
interface ASTNode {
  type: string;
  body: ASTNode[];
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
}

interface IRNode {
  type: string;
  instructions: unknown[];
}

interface CompileOptions {
  optimize?: boolean;
  validate?: boolean;
  sourceMaps?: boolean;
}

interface CompileResult {
  code: string;
  ast?: ASTNode;
  errors: string[];
}

// === Facade (index.ts) ===

/**
 * コンパイラの公開 API。
 * 内部の Parser, Validator, Optimizer, Transformer, Emitter を隠蔽する。
 */
export function compile(
  source: string,
  options: CompileOptions = {},
): CompileResult {
  const { optimize = true, validate = true } = options;

  // 1. パース
  const parser = new Parser();
  const ast = parser.parse(source);

  // 2. バリデーション（オプション）
  if (validate) {
    const validator = new Validator();
    const result = validator.validate(ast);
    if (!result.valid) {
      return { code: "", errors: result.errors };
    }
  }

  // 3. 最適化（オプション）
  let optimizedAst = ast;
  if (optimize) {
    const optimizer = new Optimizer();
    optimizedAst = optimizer.optimize(ast);
  }

  // 4. 変換
  const transformer = new Transformer();
  const ir = transformer.transform(optimizedAst);

  // 5. 出力
  const emitter = new Emitter();
  const code = emitter.emit(ir);

  return { code, ast, errors: [] };
}

// === 使用例 ===
// 利用側は内部を知る必要がない
// import { compile } from "my-compiler";

const result = compile("const x = 1 + 2;");
console.log(result.code);

const resultNoOptimize = compile("const x = 1 + 2;", { optimize: false });
```

**ポイント**: npm パッケージの `index.ts` は典型的な Facade。内部の 5 つのクラスを `compile` 関数 1 つに集約。TypeScript コンパイラ (`tsc`)、Babel、webpack も同じ設計。

---

### コード例 9: Python -- Django/Flask 風 ORM Facade

```python
from typing import Any, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime


# === サブシステム群 ===

class ConnectionPool:
    """データベース接続プール"""
    _instance = None

    def __init__(self, dsn: str, pool_size: int = 5):
        self.dsn = dsn
        self.pool_size = pool_size
        print(f"ConnectionPool: Created (dsn={dsn}, size={pool_size})")

    def acquire(self) -> "Connection":
        print("ConnectionPool: Connection acquired")
        return Connection()

    def release(self, conn: "Connection") -> None:
        print("ConnectionPool: Connection released")


class Connection:
    """データベース接続"""
    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        print(f"Connection: {sql} params={params}")
        return [{"id": 1}]  # ダミー結果

    def begin(self) -> None:
        print("Connection: BEGIN")

    def commit(self) -> None:
        print("Connection: COMMIT")

    def rollback(self) -> None:
        print("Connection: ROLLBACK")


class QueryBuilder:
    """SQL クエリビルダー"""
    def __init__(self, table: str):
        self._table = table
        self._conditions: list[str] = []
        self._params: list[Any] = []
        self._order: str | None = None
        self._limit: int | None = None

    def where(self, condition: str, *params: Any) -> "QueryBuilder":
        self._conditions.append(condition)
        self._params.extend(params)
        return self

    def order_by(self, column: str, desc: bool = False) -> "QueryBuilder":
        direction = "DESC" if desc else "ASC"
        self._order = f"{column} {direction}"
        return self

    def limit(self, n: int) -> "QueryBuilder":
        self._limit = n
        return self

    def build_select(self) -> tuple[str, tuple]:
        sql = f"SELECT * FROM {self._table}"
        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)
        if self._order:
            sql += f" ORDER BY {self._order}"
        if self._limit:
            sql += f" LIMIT {self._limit}"
        return sql, tuple(self._params)

    def build_insert(self, data: dict) -> tuple[str, tuple]:
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        sql = f"INSERT INTO {self._table} ({cols}) VALUES ({placeholders})"
        return sql, tuple(data.values())

    def build_delete(self) -> tuple[str, tuple]:
        sql = f"DELETE FROM {self._table}"
        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)
        return sql, tuple(self._params)


class Migrator:
    """スキーママイグレーション"""
    def create_table(self, name: str, columns: dict[str, str]) -> str:
        cols = ", ".join(f"{k} {v}" for k, v in columns.items())
        return f"CREATE TABLE IF NOT EXISTS {name} ({cols})"


# === Facade ===

T = TypeVar("T")


class DatabaseFacade:
    """データベース操作の統一インタフェース"""

    def __init__(self, dsn: str):
        self._pool = ConnectionPool(dsn)
        self._migrator = Migrator()

    def find(
        self,
        table: str,
        conditions: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """条件検索（SELECT）"""
        qb = QueryBuilder(table)
        if conditions:
            for col, val in conditions.items():
                qb.where(f"{col} = ?", val)
        if order_by:
            desc = order_by.startswith("-")
            col = order_by.lstrip("-")
            qb.order_by(col, desc=desc)
        if limit:
            qb.limit(limit)

        sql, params = qb.build_select()
        conn = self._pool.acquire()
        try:
            return conn.execute(sql, params)
        finally:
            self._pool.release(conn)

    def insert(self, table: str, data: dict) -> dict:
        """レコード挿入（INSERT）"""
        qb = QueryBuilder(table)
        sql, params = qb.build_insert(data)
        conn = self._pool.acquire()
        try:
            conn.begin()
            result = conn.execute(sql, params)
            conn.commit()
            return result[0] if result else {}
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.release(conn)

    def delete(self, table: str, conditions: dict[str, Any]) -> None:
        """レコード削除（DELETE）"""
        qb = QueryBuilder(table)
        for col, val in conditions.items():
            qb.where(f"{col} = ?", val)
        sql, params = qb.build_delete()
        conn = self._pool.acquire()
        try:
            conn.begin()
            conn.execute(sql, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.release(conn)

    def migrate(self, table: str, columns: dict[str, str]) -> None:
        """テーブル作成（マイグレーション）"""
        sql = self._migrator.create_table(table, columns)
        conn = self._pool.acquire()
        try:
            conn.execute(sql)
        finally:
            self._pool.release(conn)


# === 使用例 ===

db = DatabaseFacade("postgresql://localhost:5432/mydb")

# テーブル作成
db.migrate("users", {
    "id": "SERIAL PRIMARY KEY",
    "name": "VARCHAR(100)",
    "email": "VARCHAR(255)",
    "created_at": "TIMESTAMP DEFAULT NOW()",
})

# レコード挿入
db.insert("users", {"name": "Alice", "email": "alice@example.com"})

# 検索
users = db.find("users", conditions={"name": "Alice"}, order_by="-created_at", limit=10)

# 削除
db.delete("users", conditions={"id": 1})
```

**ポイント**: ConnectionPool, QueryBuilder, Connection, Migrator の 4 つのサブシステムを `DatabaseFacade` が統合。Django ORM や SQLAlchemy の `Session` もこの設計。

---

### コード例 10: Facade + Strategy の組み合わせ（TypeScript）

```typescript
// === Strategy インタフェース ===

interface NotificationChannel {
  send(to: string, message: string): Promise<boolean>;
}

class EmailChannel implements NotificationChannel {
  async send(to: string, message: string): Promise<boolean> {
    console.log(`Email to ${to}: ${message}`);
    return true;
  }
}

class SmsChannel implements NotificationChannel {
  async send(to: string, message: string): Promise<boolean> {
    console.log(`SMS to ${to}: ${message}`);
    return true;
  }
}

class PushChannel implements NotificationChannel {
  async send(to: string, message: string): Promise<boolean> {
    console.log(`Push to ${to}: ${message}`);
    return true;
  }
}

class SlackChannel implements NotificationChannel {
  async send(to: string, message: string): Promise<boolean> {
    console.log(`Slack to ${to}: ${message}`);
    return true;
  }
}

// === サブシステム群 ===

class TemplateRenderer {
  render(template: string, vars: Record<string, string>): string {
    let result = template;
    for (const [key, value] of Object.entries(vars)) {
      result = result.replace(`{{${key}}}`, value);
    }
    return result;
  }
}

class UserPreferences {
  getPreferredChannels(userId: string): string[] {
    // 実際はDBから取得
    return ["email", "push"];
  }

  getContactInfo(userId: string): Record<string, string> {
    return {
      email: "user@example.com",
      phone: "+81-90-1234-5678",
      deviceToken: "abc123",
      slackId: "U12345",
    };
  }
}

class NotificationLog {
  async log(
    userId: string,
    channel: string,
    message: string,
    success: boolean,
  ): Promise<void> {
    console.log(
      `Log: user=${userId} channel=${channel} success=${success}`
    );
  }
}

// === Facade ===

interface NotifyOptions {
  template: string;
  vars: Record<string, string>;
  userId: string;
  channels?: string[];  // 指定しなければユーザー設定を使用
}

class NotificationFacade {
  private channelMap: Map<string, NotificationChannel>;

  constructor(
    private renderer: TemplateRenderer,
    private preferences: UserPreferences,
    private logger: NotificationLog,
    channels: Record<string, NotificationChannel>,
  ) {
    this.channelMap = new Map(Object.entries(channels));
  }

  /** 通知送信: テンプレート展開 → チャネル選択 → 送信 → ログ */
  async notify(options: NotifyOptions): Promise<void> {
    const { template, vars, userId } = options;

    // 1. テンプレート展開
    const message = this.renderer.render(template, vars);

    // 2. チャネル決定（明示指定 or ユーザー設定）
    const channelNames = options.channels
      ?? this.preferences.getPreferredChannels(userId);

    // 3. 連絡先取得
    const contacts = this.preferences.getContactInfo(userId);

    // 4. 全チャネルに送信（並列）
    const results = await Promise.allSettled(
      channelNames.map(async (name) => {
        const channel = this.channelMap.get(name);
        if (!channel) {
          console.warn(`Unknown channel: ${name}`);
          return;
        }

        const to = contacts[name] ?? contacts.email;
        const success = await channel.send(to, message);
        await this.logger.log(userId, name, message, success);
      }),
    );

    const failures = results.filter(r => r.status === "rejected");
    if (failures.length > 0) {
      console.error(`${failures.length} notification(s) failed`);
    }
  }

  /** 緊急通知: 全チャネルに送信 */
  async emergency(userId: string, message: string): Promise<void> {
    const allChannels = Array.from(this.channelMap.keys());
    await this.notify({
      template: "[URGENT] {{message}}",
      vars: { message },
      userId,
      channels: allChannels,
    });
  }
}

// === 使用例 ===

const notifier = new NotificationFacade(
  new TemplateRenderer(),
  new UserPreferences(),
  new NotificationLog(),
  {
    email: new EmailChannel(),
    sms: new SmsChannel(),
    push: new PushChannel(),
    slack: new SlackChannel(),
  },
);

// ユーザー設定に基づいて通知（email + push）
await notifier.notify({
  template: "Hello {{name}}, your order #{{orderId}} has shipped!",
  vars: { name: "Taro", orderId: "12345" },
  userId: "user-1",
});

// 緊急通知: 全チャネル
await notifier.emergency("user-1", "System maintenance in 30 minutes");
```

**ポイント**: Facade（NotificationFacade）が Strategy（NotificationChannel）と組み合わされている。Facade がサブシステム群のオーケストレーションを担当し、Strategy が個々の送信チャネルの切り替えを担当する。

---

## 4. 比較表

### 比較表 1: Facade vs Adapter vs Mediator vs Controller

| 観点 | Facade | Adapter | Mediator | Controller |
|------|--------|---------|----------|------------|
| **目的** | 複雑さの隠蔽 | インタフェース変換 | オブジェクト間の調停 | リクエストのルーティング |
| **対象数** | 多数のサブシステム | 1 つ | 多数のコンポーネント | 多数のサービス |
| **方向** | 一方向（Client -> Sub） | 一方向 | 双方向 | 一方向 |
| **新 IF 作成** | 簡素化された IF を作る | 既存 IF を変換 | 通信 IF を作る | エンドポイント IF を作る |
| **サブシステムの認識** | 知らない | 知らない | 互いを知る | 知らない |
| **状態管理** | なし（ステートレス） | なし | あり（コンポーネント状態） | あり（リクエスト状態） |
| **例** | `compile(source)` | XMLParser -> JSON | ChatRoom | Express Router |

### 比較表 2: Facade のレベル別設計

| レベル | 例 | 粒度 | スコープ |
|--------|-----|------|---------|
| **関数** | `compile(source)` | 最細粒度 | 1 ファイル内 |
| **モジュール** | `index.ts` の re-export | 細粒度 | 1 パッケージ |
| **クラス** | `UserFacade` | 中粒度 | 1 ドメイン |
| **サービス** | API Gateway | 粗粒度 | 複数マイクロサービス |
| **インフラ** | CDK/Terraform wrapper | 最粗粒度 | クラウドリソース群 |

### 比較表 3: Facade パターンの実装方式

| 方式 | 言語/フレームワーク | 特徴 | 適用場面 |
|------|---------------------|------|---------|
| **クラス Facade** | Java, TypeScript, Kotlin | DI 可能、テストしやすい | サービス層 |
| **関数 Facade** | TypeScript, Python, Go | シンプル、状態なし | ユーティリティ |
| **モジュール Facade** | `index.ts`, `__init__.py` | re-export で公開 API 制御 | パッケージ公開 |
| **カスタムフック** | React | Hook の合成 | UI 状態管理 |
| **ゲートウェイ** | API Gateway, BFF | ネットワーク境界 | マイクロサービス |
| **CLI ラッパー** | Makefile, npm scripts | コマンド集約 | 開発ワークフロー |

---

## 5. アンチパターン

### アンチパターン 1: God Facade（肥大化 Facade）

```typescript
// NG: あらゆるドメインの操作を 1 つの Facade に詰め込む
class AppFacade {
  // ユーザー管理
  createUser(name: string): void { /* ... */ }
  deleteUser(id: string): void { /* ... */ }
  updateUserProfile(id: string, data: unknown): void { /* ... */ }

  // 注文管理
  createOrder(userId: string, items: unknown[]): void { /* ... */ }
  cancelOrder(orderId: string): void { /* ... */ }
  refundOrder(orderId: string): void { /* ... */ }

  // 決済
  processPayment(orderId: string, method: string): void { /* ... */ }
  verifyPayment(txId: string): void { /* ... */ }

  // レポート
  generateDailyReport(): void { /* ... */ }
  generateMonthlyReport(): void { /* ... */ }

  // メール
  sendWelcomeEmail(userId: string): void { /* ... */ }
  sendOrderConfirmation(orderId: string): void { /* ... */ }

  // ... 50 メソッド以上
}

// 問題:
// 1. SRP 違反: 1 つのクラスに 5+ ドメインの責務
// 2. 変更の影響範囲が広すぎる
// 3. テストが困難（モック対象が多すぎる）
// 4. Facade 自体が「複雑なサブシステム」になる
```

```typescript
// OK: ドメインごとに Facade を分割

class UserFacade {
  constructor(
    private repo: UserRepository,
    private email: EmailService,
    private audit: AuditService,
  ) {}

  register(name: string, emailAddr: string): User { /* ... */ }
  deactivate(id: string): void { /* ... */ }
}

class OrderFacade {
  constructor(
    private repo: OrderRepository,
    private payment: PaymentService,
    private inventory: InventoryService,
  ) {}

  place(userId: string, items: OrderItem[]): Order { /* ... */ }
  cancel(orderId: string): void { /* ... */ }
}

class ReportFacade {
  constructor(
    private analytics: AnalyticsService,
    private exporter: ExportService,
  ) {}

  daily(): Report { /* ... */ }
  monthly(): Report { /* ... */ }
}

// 各 Facade は 3-5 メソッド、明確な責務境界
```

**判断基準**: 1 つの Facade が **7 メソッド以上** になったら分割を検討する。

---

### アンチパターン 2: Leaky Facade（漏れのある Facade）

```typescript
// NG: Facade がサブシステムの内部詳細を露出している
class LeakyFacade {
  private db: Database;
  private cache: Redis;

  // サブシステムの型がそのまま露出
  getDbConnection(): DatabaseConnection {
    return this.db.getConnection();
  }

  // 内部のキャッシュキー規則がリークしている
  getCachedUser(userId: string): string | null {
    return this.cache.get(`user:v2:${userId}`);  // キー形式がリーク
  }

  // SQL がそのまま露出
  queryUsers(sql: string): User[] {
    return this.db.query(sql);
  }
}

// 問題:
// 1. クライアントがサブシステムの内部を知る必要がある
// 2. キャッシュキー形式の変更が全クライアントに影響
// 3. SQL インジェクションのリスク
```

```typescript
// OK: 内部詳細を完全に隠蔽

class CleanFacade {
  constructor(
    private db: Database,
    private cache: Redis,
  ) {}

  // サブシステムの型を隠蔽し、ドメイン型を返す
  async findUser(userId: string): Promise<User | null> {
    // キャッシュ戦略は内部で管理
    const cached = this.cache.get(`user:v2:${userId}`);
    if (cached) return JSON.parse(cached);

    // SQL は内部で管理
    const user = await this.db.query(
      "SELECT * FROM users WHERE id = ?", [userId]
    );
    if (user) {
      this.cache.set(`user:v2:${userId}`, JSON.stringify(user), 300);
    }
    return user;
  }
}
```

**判断基準**: Facade のメソッドシグネチャにサブシステム固有の型や規則が現れたら、抽象化が不十分。

---

### アンチパターン 3: Rigid Facade（硬直した Facade）

```typescript
// NG: サブシステムへの直接アクセスを完全に遮断
class RigidFacade {
  // サブシステムをすべて private で隠蔽
  private emailService: EmailService;
  private smsService: SmsService;
  private pushService: PushService;

  // 定型操作のみ提供
  notifyUser(userId: string, msg: string): void {
    // 常に email + push で送信（変更不可）
    this.emailService.send(userId, msg);
    this.pushService.send(userId, msg);
  }

  // SMS だけ送りたい場合の手段がない
  // カスタムチャネル組み合わせの手段がない
}

// 問題:
// 1. 高度なユースケースに対応できない
// 2. 新しい要件のたびに Facade にメソッド追加が必要
// 3. 結局 Facade を迂回するハックが生まれる
```

```typescript
// OK: Facade は便利なショートカット、直接アクセスも許容

class FlexibleFacade {
  // サブシステムを公開（高度なユースケース用）
  readonly email: EmailService;
  readonly sms: SmsService;
  readonly push: PushService;

  constructor(
    email: EmailService,
    sms: SmsService,
    push: PushService,
  ) {
    this.email = email;
    this.sms = sms;
    this.push = push;
  }

  // 定型操作のショートカット
  notifyUser(userId: string, msg: string): void {
    this.email.send(userId, msg);
    this.push.send(userId, msg);
  }

  // 高度なユースケースはサブシステムに直接アクセス
  // facade.sms.send(userId, urgentMsg);
}
```

**判断基準**: Facade は「壁」ではなく「門」。80% のユースケースをカバーし、残り 20% はサブシステム直接アクセスを許容する。

---

## 6. エッジケースと注意点

### 6.1 Facade のライフサイクル管理

```typescript
// Facade がサブシステムのリソースを管理する場合、
// 適切なクリーンアップが必要

class ManagedFacade implements Disposable {
  private pool: ConnectionPool;
  private cache: CacheService;

  constructor() {
    this.pool = new ConnectionPool(10);
    this.cache = new CacheService();
  }

  // ES2024+ Disposable パターン
  [Symbol.dispose](): void {
    this.pool.close();
    this.cache.flush();
    console.log("ManagedFacade: Resources cleaned up");
  }
}

// using で自動クリーンアップ
{
  using facade = new ManagedFacade();
  // ... 使用
} // 自動的に dispose される
```

### 6.2 非同期 Facade のエラーハンドリング

```typescript
class AsyncFacade {
  async complexOperation(): Promise<Result> {
    // 複数の非同期操作を順番に実行
    // 途中で失敗した場合の補償処理（compensation）が重要
    const step1 = await this.serviceA.doSomething();

    try {
      const step2 = await this.serviceB.doSomething(step1);
    } catch (error) {
      // step1 を元に戻す（補償トランザクション）
      await this.serviceA.undo(step1);
      throw error;
    }
  }
}
```

### 6.3 Facade のバージョニング

```typescript
// API が進化する場合、古い Facade を残して新しい Facade を追加

/** @deprecated Use CheckoutFacadeV2 */
class CheckoutFacade {
  checkout(cartId: string): void { /* 旧実装 */ }
}

class CheckoutFacadeV2 {
  checkout(cartId: string, options: CheckoutOptions): Promise<Receipt> {
    /* 新実装 */
  }
}
```

### 6.4 テスト戦略

```typescript
// Facade のテスト: サブシステムをモックして統合テスト

describe("DeployFacade", () => {
  it("should execute full deployment pipeline", async () => {
    const git = { pull: jest.fn(), tag: jest.fn() };
    const build = {
      install: jest.fn(),
      lint: jest.fn(),
      test: jest.fn(),
      build: jest.fn().mockReturnValue("artifact.tar.gz"),
    };
    const deploy = {
      upload: jest.fn().mockReturnValue("https://prod.example.com"),
      activate: jest.fn(),
      healthCheck: jest.fn().mockReturnValue(true),
    };
    const notify = { sendSlack: jest.fn(), sendEmail: jest.fn() };

    const facade = new DeployFacade(git, build, deploy, notify);
    const result = await facade.release("1.0.0");

    // 各サブシステムが正しい順序で呼ばれたか検証
    expect(git.pull).toHaveBeenCalledWith("main");
    expect(build.build).toHaveBeenCalledAfter(build.test);
    expect(deploy.healthCheck).toHaveBeenCalled();
    expect(git.tag).toHaveBeenCalledWith("1.0.0");
    expect(notify.sendSlack).toHaveBeenCalled();
  });

  it("should rollback on health check failure", async () => {
    const deploy = {
      upload: jest.fn().mockReturnValue("url"),
      activate: jest.fn(),
      healthCheck: jest.fn().mockReturnValue(false),  // 失敗
      rollback: jest.fn(),
    };
    // ... ロールバックが呼ばれることを検証
  });
});
```

---

## 7. トレードオフ分析

### 導入すべき場面

| 場面 | 理由 |
|------|------|
| 3+ サブシステムを組み合わせる定型操作 | 手順の重複を排除 |
| ライブラリ/フレームワークの公開 API | 内部を隠蔽し安定 API を提供 |
| レガシーシステムのラッピング | 新 API で古い実装を隠蔽 |
| マイクロサービスの BFF | 複数サービスを集約 |
| テスト容易性の向上 | モック対象を Facade に集約 |

### 導入すべきでない場面

| 場面 | 理由 |
|------|------|
| サブシステムが 1 つだけ | Facade の付加価値がない |
| クライアントがサブシステムの柔軟な組み合わせを必要とする | Facade が制約になる |
| Facade のメソッド数が 10+ になりそう | God Facade のリスク |
| パフォーマンスクリティカルなパス | 追加の間接層がオーバーヘッドになる |

### Facade のコスト

```
┌──────────────────────────────────────────────────────────┐
│                 Facade のコスト分析                       │
│                                                         │
│  メリット                    デメリット                  │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │ + クライアントの簡素化│    │ - 間接層の追加       │   │
│  │ + DRY（手順の一元化） │    │ - God Facade のリスク│   │
│  │ + 疎結合             │    │ - 柔軟性の低下       │   │
│  │ + テスト容易性       │    │   （定型操作のみ）    │   │
│  │ + 変更の局所化       │    │ - 全機能の再公開が   │   │
│  │ + 学習コスト低下     │    │   必要になる場合     │   │
│  └──────────────────────┘    └──────────────────────┘   │
│                                                         │
│  判断: サブシステムが 3+ あり、80%+ のユースケースが     │
│  定型的であれば、Facade の導入は合理的。                  │
└──────────────────────────────────────────────────────────┘
```

---

## 8. 演習問題

### 演習 1: 基本 -- ファイル変換 Facade（難易度: ★☆☆）

以下のサブシステムを使って、`FileConverterFacade` を実装してください。

```typescript
// 与えられたサブシステム
class FileReader {
  read(path: string): string {
    console.log(`Reading ${path}`);
    return "file content";
  }
}

class CsvParser {
  parse(content: string): Record<string, string>[] {
    console.log("Parsing CSV");
    return [{ name: "Alice", age: "30" }];
  }
}

class JsonFormatter {
  format(data: unknown): string {
    console.log("Formatting to JSON");
    return JSON.stringify(data, null, 2);
  }
}

class FileWriter {
  write(path: string, content: string): void {
    console.log(`Writing to ${path}`);
  }
}

// TODO: FileConverterFacade を実装
// - convertCsvToJson(inputPath, outputPath) メソッドを提供
// - 4 つのサブシステムを正しい順序で使う
```

**期待される出力**:

```
Reading data.csv
Parsing CSV
Formatting to JSON
Writing to data.json
```

<details>
<summary>解答例（クリックで展開）</summary>

```typescript
class FileConverterFacade {
  constructor(
    private reader: FileReader,
    private parser: CsvParser,
    private formatter: JsonFormatter,
    private writer: FileWriter,
  ) {}

  convertCsvToJson(inputPath: string, outputPath: string): void {
    // 1. ファイル読み込み
    const content = this.reader.read(inputPath);

    // 2. CSV パース
    const data = this.parser.parse(content);

    // 3. JSON フォーマット
    const json = this.formatter.format(data);

    // 4. ファイル書き込み
    this.writer.write(outputPath, json);
  }
}

const converter = new FileConverterFacade(
  new FileReader(),
  new CsvParser(),
  new JsonFormatter(),
  new FileWriter(),
);

converter.convertCsvToJson("data.csv", "data.json");
// 出力:
// Reading data.csv
// Parsing CSV
// Formatting to JSON
// Writing to data.json
```

</details>

---

### 演習 2: 応用 -- 決済処理 Facade（難易度: ★★☆）

以下の要件を満たす `PaymentFacade` を設計・実装してください。

**要件**:
1. `processPayment(orderId, amount, method)` メソッドを提供
2. 以下の手順を実行: 在庫確認 -> 決済処理 -> 在庫減算 -> 領収書生成 -> メール送信
3. 決済失敗時は在庫を元に戻す（補償トランザクション）
4. 各サブシステムはインタフェースで定義し、DI で注入

```typescript
// サブシステムのインタフェース
interface InventoryService {
  check(orderId: string): boolean;
  reserve(orderId: string): void;
  release(orderId: string): void;  // 補償用
}

interface PaymentGateway {
  charge(amount: number, method: string): Promise<string>; // returns txId
}

interface ReceiptService {
  generate(orderId: string, txId: string, amount: number): string;
}

interface EmailService {
  send(to: string, subject: string, body: string): void;
}
```

**期待される出力（正常系）**:

```
Inventory: Checking order-123
Inventory: Reserved order-123
Payment: Charging 5000 via credit_card
Receipt: Generated for order-123 (tx: TX-abc)
Email: Sent to customer
Payment complete: TX-abc
```

**期待される出力（決済失敗）**:

```
Inventory: Checking order-123
Inventory: Reserved order-123
Payment: Charging 5000 via credit_card
Payment failed: insufficient funds
Inventory: Released order-123 (compensation)
Error: Payment failed
```

<details>
<summary>解答例（クリックで展開）</summary>

```typescript
class PaymentFacade {
  constructor(
    private inventory: InventoryService,
    private payment: PaymentGateway,
    private receipt: ReceiptService,
    private email: EmailService,
  ) {}

  async processPayment(
    orderId: string,
    amount: number,
    method: string,
    customerEmail: string = "customer@example.com",
  ): Promise<string> {
    // 1. 在庫確認
    if (!this.inventory.check(orderId)) {
      throw new Error("Out of stock");
    }

    // 2. 在庫予約
    this.inventory.reserve(orderId);

    // 3. 決済（失敗時は在庫を元に戻す）
    let txId: string;
    try {
      txId = await this.payment.charge(amount, method);
    } catch (error) {
      // 補償トランザクション: 在庫を元に戻す
      this.inventory.release(orderId);
      throw error;
    }

    // 4. 領収書生成
    const receiptText = this.receipt.generate(orderId, txId, amount);

    // 5. メール送信
    this.email.send(
      customerEmail,
      `Order ${orderId} confirmed`,
      receiptText,
    );

    return txId;
  }
}

// === テスト用の実装 ===

class MockInventory implements InventoryService {
  check(orderId: string): boolean {
    console.log(`Inventory: Checking ${orderId}`);
    return true;
  }
  reserve(orderId: string): void {
    console.log(`Inventory: Reserved ${orderId}`);
  }
  release(orderId: string): void {
    console.log(`Inventory: Released ${orderId} (compensation)`);
  }
}

class MockPaymentGateway implements PaymentGateway {
  constructor(private shouldFail: boolean = false) {}
  async charge(amount: number, method: string): Promise<string> {
    console.log(`Payment: Charging ${amount} via ${method}`);
    if (this.shouldFail) {
      console.log("Payment failed: insufficient funds");
      throw new Error("Payment failed");
    }
    return "TX-abc";
  }
}

class MockReceiptService implements ReceiptService {
  generate(orderId: string, txId: string, amount: number): string {
    console.log(`Receipt: Generated for ${orderId} (tx: ${txId})`);
    return `Receipt: ${orderId} - ${txId} - $${amount}`;
  }
}

class MockEmailService implements EmailService {
  send(to: string, subject: string, body: string): void {
    console.log("Email: Sent to customer");
  }
}

// 正常系
const facade = new PaymentFacade(
  new MockInventory(),
  new MockPaymentGateway(false),
  new MockReceiptService(),
  new MockEmailService(),
);
const txId = await facade.processPayment("order-123", 5000, "credit_card");
console.log(`Payment complete: ${txId}`);

// 異常系
const facadeFail = new PaymentFacade(
  new MockInventory(),
  new MockPaymentGateway(true),  // 決済失敗
  new MockReceiptService(),
  new MockEmailService(),
);
try {
  await facadeFail.processPayment("order-123", 5000, "credit_card");
} catch (e) {
  console.log(`Error: ${(e as Error).message}`);
}
```

</details>

---

### 演習 3: 発展 -- Facade のリファクタリング（難易度: ★★★）

以下の God Facade を、適切に分割してリファクタリングしてください。

**要件**:
1. God Facade を 3 つ以上の Facade に分割
2. 各 Facade は SRP を守る（1 ドメインのみ担当）
3. Facade 間で共通のサブシステムがある場合は DI で共有
4. 元のクライアントコードが最小限の変更で動作すること

```typescript
// 現状の God Facade（リファクタリング対象）
class ECommerceFacade {
  // ユーザー系
  registerUser(name: string, email: string): User { /* ... */ }
  loginUser(email: string, password: string): string { /* ... */ }
  updateProfile(userId: string, data: Partial<User>): void { /* ... */ }
  deleteUser(userId: string): void { /* ... */ }

  // 商品系
  listProducts(category?: string): Product[] { /* ... */ }
  searchProducts(query: string): Product[] { /* ... */ }
  getProductDetail(productId: string): Product { /* ... */ }
  addProductReview(productId: string, review: Review): void { /* ... */ }

  // 注文系
  createOrder(userId: string, items: CartItem[]): Order { /* ... */ }
  cancelOrder(orderId: string): void { /* ... */ }
  trackOrder(orderId: string): OrderStatus { /* ... */ }
  returnOrder(orderId: string, reason: string): void { /* ... */ }
  processPayment(orderId: string, method: string): void { /* ... */ }
  generateInvoice(orderId: string): string { /* ... */ }

  // 通知系
  sendEmail(to: string, template: string, vars: object): void { /* ... */ }
  sendSms(to: string, message: string): void { /* ... */ }
  sendPushNotification(deviceId: string, message: string): void { /* ... */ }
}
```

**期待する分割結果の構造**:

```
Before: 1 God Facade (16 methods)
After:  4 Domain Facades (3-5 methods each)
        - UserFacade
        - ProductFacade
        - OrderFacade
        - NotificationFacade
```

<details>
<summary>解答例（クリックで展開）</summary>

```typescript
// === 共通サブシステム ===

interface EventBus {
  emit(event: string, data: unknown): void;
}

interface AuditLogger {
  log(action: string, userId: string, detail: string): void;
}

// === 分割された Facade 群 ===

class UserFacade {
  constructor(
    private repo: UserRepository,
    private auth: AuthService,
    private events: EventBus,
    private audit: AuditLogger,
  ) {}

  register(name: string, email: string): User {
    const user = this.repo.create({ name, email });
    this.events.emit("user.registered", user);
    this.audit.log("REGISTER", user.id, `User ${name} registered`);
    return user;
  }

  login(email: string, password: string): string {
    return this.auth.authenticate(email, password);
  }

  updateProfile(userId: string, data: Partial<User>): void {
    this.repo.update(userId, data);
    this.audit.log("UPDATE_PROFILE", userId, JSON.stringify(data));
  }

  delete(userId: string): void {
    this.repo.delete(userId);
    this.events.emit("user.deleted", { userId });
    this.audit.log("DELETE", userId, "User deleted");
  }
}

class ProductFacade {
  constructor(
    private catalog: CatalogService,
    private search: SearchService,
    private reviews: ReviewService,
  ) {}

  list(category?: string): Product[] {
    return this.catalog.list(category);
  }

  search(query: string): Product[] {
    return this.search.search(query);
  }

  getDetail(productId: string): Product {
    return this.catalog.getById(productId);
  }

  addReview(productId: string, review: Review): void {
    this.reviews.add(productId, review);
  }
}

class OrderFacade {
  constructor(
    private orders: OrderRepository,
    private payment: PaymentGateway,
    private inventory: InventoryService,
    private invoicing: InvoiceService,
    private events: EventBus,        // UserFacade と共有
    private audit: AuditLogger,      // UserFacade と共有
  ) {}

  create(userId: string, items: CartItem[]): Order {
    this.inventory.reserve(items);
    const order = this.orders.create(userId, items);
    this.events.emit("order.created", order);
    return order;
  }

  cancel(orderId: string): void {
    const order = this.orders.get(orderId);
    this.inventory.release(order.items);
    this.orders.updateStatus(orderId, "cancelled");
    this.audit.log("CANCEL_ORDER", order.userId, orderId);
  }

  track(orderId: string): OrderStatus {
    return this.orders.getStatus(orderId);
  }

  processPayment(orderId: string, method: string): void {
    const order = this.orders.get(orderId);
    this.payment.charge(order.total, method);
    this.orders.updateStatus(orderId, "paid");
    this.events.emit("order.paid", { orderId });
  }

  return_(orderId: string, reason: string): void {
    this.orders.updateStatus(orderId, "returned");
    this.audit.log("RETURN_ORDER", "", `${orderId}: ${reason}`);
  }

  generateInvoice(orderId: string): string {
    const order = this.orders.get(orderId);
    return this.invoicing.generate(order);
  }
}

class NotificationFacade {
  constructor(
    private email: EmailService,
    private sms: SmsService,
    private push: PushService,
    private templates: TemplateEngine,
  ) {}

  sendEmail(to: string, template: string, vars: object): void {
    const body = this.templates.render(template, vars);
    this.email.send(to, body);
  }

  sendSms(to: string, message: string): void {
    this.sms.send(to, message);
  }

  sendPush(deviceId: string, message: string): void {
    this.push.send(deviceId, message);
  }
}

// === DI コンテナでの構成 ===

function createFacades(container: DIContainer) {
  // 共通サブシステム
  const events = container.get(EventBus);
  const audit = container.get(AuditLogger);

  return {
    users: new UserFacade(
      container.get(UserRepository),
      container.get(AuthService),
      events,  // 共有
      audit,   // 共有
    ),
    products: new ProductFacade(
      container.get(CatalogService),
      container.get(SearchService),
      container.get(ReviewService),
    ),
    orders: new OrderFacade(
      container.get(OrderRepository),
      container.get(PaymentGateway),
      container.get(InventoryService),
      container.get(InvoiceService),
      events,  // 共有
      audit,   // 共有
    ),
    notifications: new NotificationFacade(
      container.get(EmailService),
      container.get(SmsService),
      container.get(PushService),
      container.get(TemplateEngine),
    ),
  };
}

// === 移行用のアダプター（後方互換性） ===

/** @deprecated 個別の Facade を使用してください */
class LegacyECommerceFacade {
  constructor(
    private users: UserFacade,
    private products: ProductFacade,
    private orders: OrderFacade,
    private notifications: NotificationFacade,
  ) {}

  // 旧 API を新 Facade に委譲
  registerUser(name: string, email: string): User {
    return this.users.register(name, email);
  }
  createOrder(userId: string, items: CartItem[]): Order {
    return this.orders.create(userId, items);
  }
  // ... 他のメソッドも同様に委譲
}
```

</details>

---

## 9. FAQ

### Q1: Facade は API Gateway と同じですか？

概念は同じです。API Gateway はネットワーク境界で動作する Facade パターンの大規模適用と言えます。両者の違いは以下の通りです。

| 観点 | Facade（コード内） | API Gateway |
|------|---------------------|-------------|
| スコープ | アプリケーション内 | ネットワーク境界 |
| プロトコル | メソッド呼び出し | HTTP/gRPC |
| 追加責務 | なし | 認証、レート制限、ロードバランシング |
| 例 | `UserFacade` クラス | Kong, AWS API Gateway |

### Q2: Facade を使うとテストが難しくなりませんか？

いいえ、むしろ容易になります。DI でサブシステムを注入する設計にすれば、テスト時にモックを注入できます。Facade がなければ、クライアントが直接操作する全サブシステムをモックする必要があり、テストがより複雑になります。

### Q3: React のカスタムフックは Facade ですか？

はい。複数の Hook（useState, useEffect, useReducer 等）や API 呼び出しを内部に隠蔽し、コンポーネントにシンプルな API を提供する点で、Facade パターンの一形態です。React 公式ドキュメントでも「カスタムフックで複雑さを隠蔽する」設計が推奨されています。

### Q4: Facade とサービス層（Service Layer）の違いは何ですか？

Facade は構造パターンで「既存のサブシステム群を簡素化する入口」を提供します。Service Layer はアーキテクチャパターンで「ビジネスロジックの実行層」を定義します。実際にはサービス層が Facade の役割を兼ねることが多いです。

| 観点 | Facade | Service Layer |
|------|--------|---------------|
| 目的 | 複雑さの隠蔽 | ビジネスロジックの集約 |
| ロジック | 最小限（委譲のみ） | ビジネスルールを含む |
| トランザクション | 通常なし | トランザクション境界 |
| 再利用 | プレゼンテーション層から | 複数のプレゼンテーションから |

### Q5: Facade の中にビジネスロジックを入れてもよいですか？

原則として入れるべきではありません。Facade は「薄いオーケストレーション層」であるべきです。ビジネスロジックはサブシステム（ドメインサービス）に配置し、Facade はそれらの呼び出し順序の管理に専念してください。

```typescript
// NG: Facade にビジネスロジック
class BadFacade {
  placeOrder(items: Item[]) {
    const total = items.reduce((sum, i) => sum + i.price * i.qty, 0);
    if (total > 10000) {
      const discount = total * 0.1;  // ← ビジネスロジック
      // ...
    }
  }
}

// OK: ビジネスロジックはサブシステムに
class GoodFacade {
  placeOrder(items: Item[]) {
    const total = this.pricing.calculate(items);  // ← サブシステムに委譲
    const order = this.orders.create(items, total);
    this.notify.send(order);
  }
}
```

### Q6: Facade パターンはマイクロサービスでどう使われますか？

マイクロサービスアーキテクチャでは、**BFF（Backend for Frontend）** が Facade パターンの典型例です。モバイルアプリ用 BFF は複数のマイクロサービス（User, Product, Order, Payment）を集約して、1 つの API エンドポイントでクライアントに必要なデータをまとめて返します。

```
Mobile App
    │
    ▼
┌─────────────────┐
│  Mobile BFF     │  ← Facade
│  (API Gateway)  │
└────────┬────────┘
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
  User Product Order Payment
  Service Service Service Service
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| **目的** | 複雑なサブシステム群に統一された簡潔なインタフェースを提供 |
| **本質** | 情報隠蔽 + 手順のカプセル化 + ショートカット |
| **利点** | クライアントの簡素化、疎結合、DRY、テスト容易性 |
| **適用レベル** | 関数 / モジュール / クラス / サービス / インフラ |
| **アンチパターン** | God Facade / Leaky Facade / Rigid Facade |
| **注意** | Facade は「壁」ではなく「門」。直接アクセスも許容する |
| **テスト** | DI でサブシステムを注入し、モックでテスト |
| **組み合わせ** | Strategy, Observer, Template Method と併用可能 |

---

## 次に読むべきガイド

- [Proxy パターン](./03-proxy.md) -- アクセス制御の代理オブジェクト
- [Adapter パターン](./00-adapter.md) -- インタフェース変換
- [Decorator パターン](./01-decorator.md) -- 動的な機能追加
- [Composite パターン](./04-composite.md) -- ツリー構造の統一操作
- [Mediator パターン](../02-behavioral/00-observer.md) -- オブジェクト間の調停
- [クリーンアーキテクチャ](../../../system-design-guide/docs/02-architecture/01-clean-architecture.md) -- 層構造設計

---

## 参考文献

1. Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
2. Freeman, E. et al. (2004). *Head First Design Patterns*. O'Reilly Media.
3. Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley.
4. Martin, R. C. (2017). *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall.
5. Richardson, C. (2018). *Microservices Patterns*. Manning Publications.
6. Refactoring.Guru -- Facade. https://refactoring.guru/design-patterns/facade
