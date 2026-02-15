# AIデバッグ -- エラー解析、ログ分析、自動修正

> AIを活用してバグの原因特定を高速化し、エラーログの分析からスタックトレースの解読、根本原因の推定、修正提案までの一連のデバッグプロセスを体系化して、従来数時間かかっていた調査を数分に短縮する手法を示す

## この章で学ぶこと

1. **AI デバッグの基本アプローチ** -- エラーメッセージ解析、スタックトレース読解、コンテキスト理解によるバグ原因の推定手法
2. **ログ分析と異常検知** -- 大量ログからのパターン抽出、異常検知、根本原因分析（RCA）の自動化パイプライン
3. **実践的なデバッグワークフロー** -- IDE統合、CI/CD連携、チーム共有の仕組みとAIデバッグの限界を理解した活用戦略

---

## 1. AI デバッグの全体像

### 1.1 デバッグプロセスにおける AI の役割

```
従来のデバッグ vs AI 支援デバッグ

  従来 (手動)
  +----------+     +----------+     +----------+     +----------+
  | エラー    | --> | ログ確認  | --> | 仮説立案  | --> | コード    |
  | 発生     |     | (手動検索)|     | (経験依存)|     | 修正     |
  +----------+     +----------+     +----------+     +----------+
      所要時間: 30分〜数時間（経験に大きく依存）

  AI 支援
  +----------+     +----------+     +----------+     +----------+
  | エラー    | --> | AI が     | --> | AI が     | --> | AI が    |
  | 発生     |     | ログ解析  |     | 原因推定  |     | 修正提案 |
  +----------+     +----------+     +----------+     +----------+
      所要時間: 1〜10分（AI のコンテキスト理解に依存）

  AI が得意な領域:
  ├── エラーメッセージの意味解説
  ├── スタックトレースの読解
  ├── 類似バグのパターンマッチング
  ├── 大量ログからの異常パターン抽出
  └── 修正コードの提案

  人間が必要な領域:
  ├── ビジネスロジックの正しさの判断
  ├── 再現手順の特定
  ├── 環境固有の問題の理解
  ├── セキュリティ影響の評価
  └── 修正の最終判断
```

### 1.2 技術スタック

```
AI デバッグ 技術マップ

  AI モデル / アシスタント
  ├── Claude Code         --- ターミナル統合型デバッグ支援
  ├── GitHub Copilot Chat --- IDE 統合型デバッグ支援
  ├── ChatGPT             --- 汎用エラー解析
  └── Cursor AI           --- エディタ統合型

  ログ分析
  ├── Datadog AI          --- ログパターン解析、異常検知
  ├── Elastic Observability --- ML ベースの異常検知
  ├── Sentry              --- エラー自動グルーピング、AI 要約
  └── PagerDuty AIOps     --- インシデント相関分析

  静的解析 (バグ予防)
  ├── SonarQube           --- コード品質・バグパターン検出
  ├── Semgrep             --- カスタムルール静的解析
  ├── CodeQL              --- セキュリティ脆弱性検出
  └── Ruff / ESLint       --- リンター

  動的解析
  ├── Sentry              --- エラートラッキング
  ├── OpenTelemetry       --- 分散トレーシング
  └── pdb / debugpy       --- Python デバッガー
```

### 1.3 AI デバッグフロー

```
  [エラー発生]
       |
       v
  [情報収集]
  ├── エラーメッセージ
  ├── スタックトレース
  ├── 関連ログ (前後 100 行)
  ├── 関連コード
  └── 環境情報 (OS, ランタイム, 依存関係)
       |
       v
  [AI に質問] ← プロンプト設計が重要
       |
       v
  [AI の回答]
  ├── 原因の推定 (複数候補)
  ├── 各候補の確率と根拠
  ├── 確認すべきポイント
  └── 修正コードの提案
       |
       v
  [人間が検証]
  ├── 修正の妥当性を確認
  ├── テストで回帰がないか確認
  └── 根本原因の理解を深める
```

---

## 2. エラーメッセージ解析

### 2.1 構造化されたエラー報告

```python
# AI に渡すためのエラー情報の構造化

import traceback
import sys
import platform
import json

class ErrorReporter:
    """デバッグに必要な情報を構造化して収集"""

    def capture_error(self, exception: Exception, context: dict = None) -> dict:
        """エラー情報を AI デバッグ用に構造化"""
        tb = traceback.extract_tb(exception.__traceback__)

        report = {
            "error": {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": self._format_traceback(tb),
            },
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "packages": self._get_relevant_packages(tb),
            },
            "context": context or {},
            "source_code": self._extract_source_context(tb),
            "recent_changes": self._get_recent_git_changes(),
        }
        return report

    def _format_traceback(self, tb) -> list[dict]:
        """スタックトレースを構造化"""
        frames = []
        for frame in tb:
            frames.append({
                "file": frame.filename,
                "line": frame.lineno,
                "function": frame.name,
                "code": frame.line,
            })
        return frames

    def _extract_source_context(self, tb, context_lines=10) -> list[dict]:
        """エラー発生箇所の前後のソースコードを抽出"""
        contexts = []
        for frame in tb[-3:]:  # 直近3フレーム
            try:
                with open(frame.filename) as f:
                    lines = f.readlines()
                start = max(0, frame.lineno - context_lines - 1)
                end = min(len(lines), frame.lineno + context_lines)
                contexts.append({
                    "file": frame.filename,
                    "error_line": frame.lineno,
                    "code": "".join(lines[start:end]),
                    "start_line": start + 1,
                })
            except (FileNotFoundError, PermissionError):
                pass
        return contexts

    def _get_recent_git_changes(self) -> str:
        """直近の Git 変更を取得"""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            return "Git 情報取得失敗"

    def _get_relevant_packages(self, tb) -> dict:
        """スタックトレースに関連するパッケージのバージョンを取得"""
        import importlib.metadata
        packages = {}
        for frame in tb:
            parts = frame.filename.split("/")
            for part in parts:
                if part.endswith(".egg-info") or part == "site-packages":
                    continue
                try:
                    version = importlib.metadata.version(part)
                    packages[part] = version
                except importlib.metadata.PackageNotFoundError:
                    pass
        return packages


# 使用例
reporter = ErrorReporter()

try:
    # バグのあるコード
    result = process_data(None)
except Exception as e:
    error_report = reporter.capture_error(
        e, context={"input": None, "user_id": 123}
    )
    print(json.dumps(error_report, indent=2, ensure_ascii=False))
```

### 2.2 AI へのデバッグプロンプト設計

```python
# 効果的なデバッグプロンプトのテンプレート

DEBUG_PROMPT_TEMPLATE = """
以下のエラーの原因を特定し、修正方法を提案してください。

## エラー情報
エラー種別: {error_type}
メッセージ: {error_message}

## スタックトレース
{traceback}

## 関連コード
{source_code}

## 環境
{environment}

## コンテキスト
{context}

## 直近の変更
{recent_changes}

以下の形式で回答してください:
1. **原因の推定** (確率の高い順に最大3つ)
2. **各原因の根拠**
3. **確認すべきポイント** (原因を絞り込むための追加調査)
4. **修正コード** (最も可能性の高い原因に対する修正)
5. **再発防止策** (テスト追加、バリデーション強化等)
"""

def build_debug_prompt(error_report: dict) -> str:
    """エラーレポートからデバッグプロンプトを構築"""
    return DEBUG_PROMPT_TEMPLATE.format(
        error_type=error_report["error"]["type"],
        error_message=error_report["error"]["message"],
        traceback=format_traceback_text(error_report["error"]["traceback"]),
        source_code=format_source_contexts(error_report["source_code"]),
        environment=json.dumps(error_report["environment"], indent=2),
        context=json.dumps(error_report["context"], indent=2),
        recent_changes=error_report.get("recent_changes", "N/A"),
    )
```

### 2.3 よくあるエラーパターンのナレッジベース

```python
# チーム共有のエラーパターンデータベース

ERROR_PATTERNS = {
    "TypeError: Cannot read properties of undefined": {
        "category": "null_reference",
        "common_causes": [
            "API レスポンスの形式変更 (フィールドが undefined)",
            "非同期処理の完了前にデータアクセス",
            "オプショナルチェーン (?.) の未使用",
        ],
        "fix_patterns": [
            "Optional Chaining: obj?.prop?.nested",
            "Nullish Coalescing: value ?? defaultValue",
            "Guard Clause: if (!obj) return;",
        ],
        "prevention": [
            "TypeScript strict mode の有効化",
            "Zod / Valibot でランタイムバリデーション",
            "API レスポンスの型定義とバリデーション",
        ],
    },
    "ECONNREFUSED": {
        "category": "connection",
        "common_causes": [
            "対象サービスが起動していない",
            "ポート番号の設定ミス",
            "Docker ネットワーク設定の問題",
            "ファイアウォール / セキュリティグループの制限",
        ],
        "diagnosis_steps": [
            "curl / telnet で接続テスト",
            "docker ps でコンテナ状態確認",
            "netstat / lsof でポート使用状況確認",
            "環境変数 (DATABASE_URL 等) の値確認",
        ],
    },
    "OOMKilled": {
        "category": "resource",
        "common_causes": [
            "メモリリーク (イベントリスナーの未解除)",
            "大量データの一括読み込み",
            "コンテナのメモリ制限が不足",
            "N+1 クエリによる大量オブジェクト生成",
        ],
        "fix_patterns": [
            "ストリーミング処理への変更",
            "ページネーション / バッチ処理",
            "メモリプロファイリング (heapdump)",
            "コンテナのリソース制限の見直し",
        ],
    },
}
```

### 2.4 言語別デバッグパターン集

```python
# 言語・フレームワーク固有のデバッグパターン

LANGUAGE_DEBUG_PATTERNS = {
    "python": {
        "ImportError / ModuleNotFoundError": {
            "diagnosis": """
            1. パッケージがインストールされているか確認:
               pip list | grep <package_name>
            2. 仮想環境が正しく有効化されているか確認:
               which python
            3. PYTHONPATH にモジュールのディレクトリが含まれているか確認
            4. パッケージ名とインポート名が異なるケース:
               例: pip install Pillow → import PIL
            """,
            "ai_prompt": """
            以下のImportErrorを解決してください。
            エラー: {error_message}
            Python バージョン: {python_version}
            インストール済みパッケージ: {pip_list}
            仮想環境: {venv_info}
            """,
        },
        "asyncio.TimeoutError": {
            "diagnosis": """
            1. タイムアウト値が適切か確認
            2. 対象サービスのレスポンスタイムを計測
            3. ネットワーク遅延の有無を確認
            4. コネクションプールの枯渇を確認
            5. async/await の使い方が正しいか確認
            """,
            "common_fixes": [
                "タイムアウト値の調整",
                "リトライロジックの追加",
                "コネクションプールサイズの拡大",
                "Circuit Breaker パターンの導入",
            ],
        },
        "SQLAlchemy DetachedInstanceError": {
            "diagnosis": """
            1. セッションのスコープを確認（リクエスト単位か）
            2. Lazy Loading がセッション外で発生していないか
            3. expire_on_commit=False の設定を確認
            4. joinedload / selectinload でイーガーロードに変更
            """,
            "ai_prompt": """
            SQLAlchemy の DetachedInstanceError が発生しています。
            モデル定義: {model_code}
            クエリコード: {query_code}
            セッション設定: {session_config}
            アクセスしようとしたリレーション: {relation_name}
            """,
        },
    },
    "javascript": {
        "Unhandled Promise Rejection": {
            "diagnosis": """
            1. async関数内でtry-catchが適切にされているか
            2. .catch()がPromiseチェーンに付いているか
            3. Promise.allSettled vs Promise.all の使い分け
            4. Node.js のバージョンによる挙動の違い
            """,
            "ai_prompt": """
            Unhandled Promise Rejection が発生しています。
            エラー: {error_message}
            コード: {code}
            Node.js バージョン: {node_version}
            このPromiseがどこでrejectされたか追跡してください。
            """,
        },
        "CORS Error": {
            "diagnosis": """
            1. サーバー側のAccess-Control-Allow-Originヘッダを確認
            2. プリフライトリクエスト（OPTIONS）への応答を確認
            3. クレデンシャル（Cookie）送信時のwithCredentials設定
            4. プロキシ設定（開発時のvite.config.ts / next.config.js）
            """,
            "common_fixes": [
                "cors ミドルウェアの設定追加",
                "開発用プロキシの設定",
                "API GatewayでのCORS設定",
                "credentials: 'include' と対応するサーバー設定",
            ],
        },
    },
    "go": {
        "panic: runtime error: invalid memory address": {
            "diagnosis": """
            1. nil ポインタのデリファレンスを確認
            2. マップの初期化忘れ（var m map[string]int → make必要）
            3. インターフェースの nil チェック漏れ
            4. goroutine 間での共有データの競合
            """,
            "ai_prompt": """
            Go の nil pointer dereference パニックが発生しています。
            スタックトレース: {stacktrace}
            関連コード: {code}
            goroutine の使用有無: {uses_goroutines}
            このパニックの原因と安全な修正方法を提案してください。
            """,
        },
    },
}
```

### 2.5 フロントエンド特有のデバッグ手法

```typescript
// ブラウザ環境のデバッグ情報収集

interface BrowserDebugInfo {
  url: string;
  userAgent: string;
  viewport: { width: number; height: number };
  networkErrors: NetworkError[];
  consoleErrors: ConsoleError[];
  performanceMetrics: PerformanceMetrics;
  reactComponentTree?: ComponentInfo[];
}

class FrontendDebugCollector {
  /**
   * ブラウザ環境のデバッグ情報を包括的に収集する
   * AIデバッグに最適な形式で構造化
   */

  collectDebugInfo(): BrowserDebugInfo {
    return {
      url: window.location.href,
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
      networkErrors: this.getNetworkErrors(),
      consoleErrors: this.getConsoleErrors(),
      performanceMetrics: this.getPerformanceMetrics(),
      reactComponentTree: this.getReactTree(),
    };
  }

  private getNetworkErrors(): NetworkError[] {
    // Performance API からネットワークエラーを抽出
    const entries = performance.getEntriesByType("resource") as PerformanceResourceTiming[];
    return entries
      .filter((entry) => entry.responseStatus >= 400 || entry.responseStatus === 0)
      .map((entry) => ({
        url: entry.name,
        status: entry.responseStatus,
        duration: entry.duration,
        initiatorType: entry.initiatorType,
        timestamp: entry.startTime,
      }));
  }

  private getConsoleErrors(): ConsoleError[] {
    // 事前にオーバーライドされた console.error のログを取得
    return (window as any).__debugConsoleErrors || [];
  }

  private getPerformanceMetrics(): PerformanceMetrics {
    const navigation = performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming;
    const paint = performance.getEntriesByType("paint");

    return {
      domContentLoaded: navigation?.domContentLoadedEventEnd - navigation?.startTime,
      loadComplete: navigation?.loadEventEnd - navigation?.startTime,
      firstPaint: paint.find((p) => p.name === "first-paint")?.startTime,
      firstContentfulPaint: paint.find((p) => p.name === "first-contentful-paint")?.startTime,
      jsHeapSize: (performance as any).memory?.usedJSHeapSize,
      jsHeapLimit: (performance as any).memory?.jsHeapSizeLimit,
    };
  }

  /**
   * 収集した情報をAIデバッグ用のプロンプトに変換
   */
  buildAIPrompt(info: BrowserDebugInfo, userDescription: string): string {
    return `
## フロントエンドバグレポート

### ユーザー報告
${userDescription}

### 環境情報
- URL: ${info.url}
- ブラウザ: ${info.userAgent}
- ビューポート: ${info.viewport.width}x${info.viewport.height}

### ネットワークエラー
${JSON.stringify(info.networkErrors, null, 2)}

### コンソールエラー
${JSON.stringify(info.consoleErrors, null, 2)}

### パフォーマンスメトリクス
${JSON.stringify(info.performanceMetrics, null, 2)}

### 分析依頼
1. エラーの根本原因を推定してください
2. ネットワークエラーとコンソールエラーの関連性を分析してください
3. パフォーマンスに問題がある場合、その原因を指摘してください
4. 修正方法を具体的なコードで示してください
`;
  }
}
```

---

## 3. ログ分析と異常検知

### 3.1 構造化ログの設計

```python
# AI 解析に適した構造化ログの設計
import structlog
import logging
from datetime import datetime

# structlog の設定
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger()

# 構造化ログの出力例
def process_order(order_id: str, user_id: str):
    log = logger.bind(order_id=order_id, user_id=user_id)

    log.info("order_processing_started", step="validation")

    try:
        # バリデーション
        order = validate_order(order_id)
        log.info("order_validated", items_count=len(order.items))

        # 支払い処理
        payment = process_payment(order)
        log.info("payment_processed",
                 amount=payment.amount,
                 method=payment.method,
                 duration_ms=payment.duration_ms)

        # 在庫確認
        inventory = check_inventory(order)
        log.info("inventory_checked",
                 all_available=inventory.all_available,
                 unavailable_items=inventory.unavailable_items)

    except PaymentError as e:
        log.error("payment_failed",
                  error_code=e.code,
                  error_message=str(e),
                  retry_possible=e.retryable)
        raise

    except Exception as e:
        log.error("order_processing_failed",
                  error_type=type(e).__name__,
                  error_message=str(e),
                  exc_info=True)
        raise
```

### 3.2 AI によるログパターン解析

```python
# 大量ログから異常パターンを AI で抽出

class LogAnalyzer:
    """AI を活用したログ分析エンジン"""

    def analyze_error_patterns(self, logs: list[dict],
                                time_window_minutes: int = 60) -> dict:
        """エラーログのパターンを分析"""

        # 1. エラーの集計
        error_groups = {}
        for log in logs:
            if log.get("level") in ("error", "critical"):
                key = f"{log.get('error_type', 'unknown')}:{log.get('error_message', '')[:100]}"
                error_groups.setdefault(key, []).append(log)

        # 2. エラーの時系列分析
        timeline = self._build_error_timeline(logs, time_window_minutes)

        # 3. 相関分析
        correlations = self._find_correlations(logs)

        return {
            "error_summary": {
                key: {
                    "count": len(entries),
                    "first_seen": entries[0].get("timestamp"),
                    "last_seen": entries[-1].get("timestamp"),
                    "sample": entries[0],
                }
                for key, entries in sorted(
                    error_groups.items(), key=lambda x: -len(x[1])
                )[:10]
            },
            "timeline": timeline,
            "correlations": correlations,
            "ai_analysis_prompt": self._build_analysis_prompt(
                error_groups, timeline, correlations
            ),
        }

    def _find_correlations(self, logs: list[dict]) -> list[dict]:
        """エラー間の相関関係を検出"""
        correlations = []

        # 同一リクエスト内のエラーチェーン
        request_logs = {}
        for log in logs:
            req_id = log.get("request_id")
            if req_id:
                request_logs.setdefault(req_id, []).append(log)

        for req_id, req_log_list in request_logs.items():
            errors = [l for l in req_log_list if l.get("level") == "error"]
            if len(errors) >= 2:
                correlations.append({
                    "type": "error_chain",
                    "request_id": req_id,
                    "errors": [e.get("error_message", "")[:100] for e in errors],
                    "root_cause_candidate": errors[0].get("error_message", ""),
                })

        return correlations

    def _build_analysis_prompt(self, error_groups, timeline, correlations) -> str:
        """AI に渡すログ分析プロンプトを構築"""
        return f"""
以下のログ分析結果から、根本原因と対策を提案してください。

## エラーサマリー（発生頻度順）
{json.dumps(list(error_groups.keys())[:10], ensure_ascii=False, indent=2)}

## 時系列パターン
{json.dumps(timeline, ensure_ascii=False, indent=2)}

## エラーの相関関係
{json.dumps(correlations[:5], ensure_ascii=False, indent=2)}

以下の形式で回答してください:
1. 根本原因の推定（最も可能性が高い順に）
2. 影響範囲の評価
3. 即座に実施すべき対策
4. 中長期的な改善策
"""
```

### 3.3 分散トレーシングとの連携

```python
# OpenTelemetry + AI 分析の統合

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

tracer = trace.get_tracer("debug-service")

class DistributedDebugger:
    """分散トレーシングと AI を組み合わせたデバッグ"""

    def analyze_slow_request(self, trace_id: str) -> dict:
        """遅いリクエストのトレースを AI で分析"""
        spans = self._get_trace_spans(trace_id)

        # スパンの所要時間を分析
        bottlenecks = []
        for span in spans:
            duration_ms = (span["end_time"] - span["start_time"]) / 1_000_000
            if duration_ms > 100:  # 100ms 以上のスパン
                bottlenecks.append({
                    "service": span["service_name"],
                    "operation": span["operation_name"],
                    "duration_ms": duration_ms,
                    "attributes": span.get("attributes", {}),
                })

        # AI に分析を依頼
        analysis_prompt = f"""
以下の分散トレースを分析し、パフォーマンスのボトルネックを特定してください。

トレース ID: {trace_id}
総所要時間: {self._get_total_duration(spans)} ms

ボトルネック候補（100ms超のスパン）:
{json.dumps(bottlenecks, indent=2, ensure_ascii=False)}

スパン数: {len(spans)}
サービス数: {len(set(s['service_name'] for s in spans))}
"""
        return {
            "bottlenecks": bottlenecks,
            "analysis_prompt": analysis_prompt,
        }
```

### 3.4 メトリクスベースの異常検知

```python
# アプリケーションメトリクスの異常をAIで検知・分析

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Optional

@dataclass
class MetricPoint:
    """メトリクスデータポイント"""
    timestamp: datetime
    value: float
    labels: dict = field(default_factory=dict)

@dataclass
class AnomalyReport:
    """異常検知レポート"""
    metric_name: str
    anomaly_type: str  # "spike", "drop", "trend_change", "pattern_break"
    severity: str      # "critical", "warning", "info"
    detected_at: datetime
    current_value: float
    expected_range: tuple[float, float]
    context: dict = field(default_factory=dict)

class MetricsAnomalyDetector:
    """メトリクスの異常をAIで検知するシステム"""

    def __init__(self, lookback_window: int = 60):
        """
        Args:
            lookback_window: 異常検知の基準とする過去の分数
        """
        self.lookback_window = lookback_window
        self.metrics_history: dict[str, list[MetricPoint]] = {}

    def detect_anomalies(self, current_metrics: dict[str, float]) -> list[AnomalyReport]:
        """現在のメトリクスから異常を検知"""
        anomalies = []

        for metric_name, current_value in current_metrics.items():
            history = self.metrics_history.get(metric_name, [])
            if len(history) < 10:
                continue  # データ不足

            historical_values = [p.value for p in history]
            avg = mean(historical_values)
            std = stdev(historical_values) if len(historical_values) > 1 else 0

            # Z-score ベースの異常検知
            z_score = (current_value - avg) / std if std > 0 else 0

            if abs(z_score) > 3:
                anomaly_type = "spike" if z_score > 0 else "drop"
                severity = "critical" if abs(z_score) > 5 else "warning"

                anomalies.append(AnomalyReport(
                    metric_name=metric_name,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    detected_at=datetime.now(),
                    current_value=current_value,
                    expected_range=(avg - 2 * std, avg + 2 * std),
                    context={
                        "z_score": z_score,
                        "historical_avg": avg,
                        "historical_std": std,
                        "data_points": len(historical_values),
                    },
                ))

        return anomalies

    def build_ai_analysis_prompt(self, anomalies: list[AnomalyReport]) -> str:
        """異常検知結果をAI分析用プロンプトに変換"""
        if not anomalies:
            return "異常は検知されませんでした。"

        anomaly_details = []
        for a in anomalies:
            anomaly_details.append(f"""
- メトリクス: {a.metric_name}
  種類: {a.anomaly_type} ({a.severity})
  現在値: {a.current_value:.2f}
  期待範囲: {a.expected_range[0]:.2f} - {a.expected_range[1]:.2f}
  Z-score: {a.context['z_score']:.2f}
""")

        return f"""
以下のメトリクス異常を分析し、原因と対策を提案してください。

## 検知された異常（{len(anomalies)}件）
{''.join(anomaly_details)}

## 分析依頼
1. 異常間の相関関係（複数の異常が同一原因に起因している可能性）
2. 推定される根本原因（最も可能性が高い順に）
3. 即座に確認すべき項目
4. 緩和策と恒久対策
"""
```

---

## 4. 自動修正と提案

### 4.1 AI によるバグフィックス提案

```python
# CI で失敗したテストを AI で自動分析・修正提案

class AutoFixSuggester:
    """テスト失敗時の自動修正提案"""

    def analyze_test_failure(self, test_output: str, source_files: dict) -> dict:
        """テスト失敗の原因分析と修正提案"""

        prompt = f"""
以下のテスト失敗を分析し、修正を提案してください。

## テスト出力
```
{test_output}
```

## 関連ソースコード
"""
        for filename, content in source_files.items():
            prompt += f"\n### {filename}\n```python\n{content}\n```\n"

        prompt += """
以下の形式で回答してください:
1. 失敗の原因
2. 修正が必要なファイルと行番号
3. 修正後のコード（差分形式）
4. 修正が他のテストに影響しないことの根拠
"""
        return {"prompt": prompt}

    def suggest_fix_from_sentry(self, sentry_event: dict) -> dict:
        """Sentry のエラーイベントから修正を提案"""
        prompt = f"""
本番環境で以下のエラーが発生しています。修正を提案してください。

エラー: {sentry_event['title']}
発生回数: {sentry_event['count']}
影響ユーザー数: {sentry_event['userCount']}
最初の発生: {sentry_event['firstSeen']}

スタックトレース:
{sentry_event['stacktrace']}

直近のリリース:
{sentry_event.get('release', 'N/A')}
"""
        return {"prompt": prompt, "severity": self._assess_severity(sentry_event)}

    def _assess_severity(self, event: dict) -> str:
        """エラーの深刻度を評価"""
        if event.get("userCount", 0) > 100:
            return "critical"
        elif event.get("count", 0) > 50:
            return "high"
        elif event.get("count", 0) > 10:
            return "medium"
        return "low"
```

### 4.2 デバッグセッションの記録と共有

```python
# デバッグセッションのナレッジベース化

class DebugSessionRecorder:
    """デバッグセッションを記録してチームで共有"""

    def record_session(self, session: dict) -> dict:
        """デバッグセッションを構造化して記録"""
        record = {
            "id": generate_session_id(),
            "timestamp": datetime.now().isoformat(),
            "error": {
                "type": session["error_type"],
                "message": session["error_message"],
                "stacktrace": session["stacktrace"],
            },
            "investigation": {
                "hypotheses": session["hypotheses"],
                "steps_taken": session["investigation_steps"],
                "ai_suggestions": session["ai_suggestions"],
                "time_spent_minutes": session["time_spent"],
            },
            "resolution": {
                "root_cause": session["root_cause"],
                "fix_description": session["fix_description"],
                "fix_commit": session["commit_hash"],
                "tests_added": session["tests_added"],
            },
            "lessons": {
                "what_helped": session.get("what_helped", []),
                "what_hindered": session.get("what_hindered", []),
                "prevention_measures": session.get("prevention", []),
            },
            "tags": session.get("tags", []),
        }
        return record

    def search_similar_issues(self, error_message: str,
                               knowledge_base: list[dict]) -> list[dict]:
        """類似のデバッグ記録を検索"""
        # テキスト類似度に基づく検索
        results = []
        for record in knowledge_base:
            similarity = compute_text_similarity(
                error_message,
                record["error"]["message"]
            )
            if similarity > 0.6:
                results.append({
                    "record": record,
                    "similarity": similarity,
                    "resolution_summary": record["resolution"]["root_cause"],
                })
        return sorted(results, key=lambda x: -x["similarity"])[:5]
```

### 4.3 CI/CD パイプラインとの統合

```python
# GitHub Actions + AI デバッグの統合例

class CIDebugIntegration:
    """CI/CDパイプラインでのAIデバッグ自動化"""

    def analyze_ci_failure(self, workflow_run: dict) -> dict:
        """CIの失敗を自動分析してPRにコメント"""

        # 失敗したジョブの情報収集
        failed_jobs = [
            job for job in workflow_run["jobs"]
            if job["conclusion"] == "failure"
        ]

        analysis_results = []
        for job in failed_jobs:
            # ログの取得と解析
            log_content = self._fetch_job_logs(job["id"])
            error_sections = self._extract_error_sections(log_content)

            # 変更されたファイルとの照合
            changed_files = self._get_pr_changed_files(workflow_run["pr_number"])
            related_changes = self._correlate_errors_with_changes(
                error_sections, changed_files
            )

            analysis_results.append({
                "job_name": job["name"],
                "errors": error_sections,
                "related_changes": related_changes,
                "ai_prompt": self._build_ci_debug_prompt(
                    job, error_sections, related_changes
                ),
            })

        return {
            "workflow_run_id": workflow_run["id"],
            "analyses": analysis_results,
            "pr_comment": self._format_pr_comment(analysis_results),
        }

    def _build_ci_debug_prompt(self, job: dict, errors: list,
                                changes: list) -> str:
        """CI失敗のAI分析用プロンプトを構築"""
        return f"""
以下のCIジョブの失敗を分析してください。

## ジョブ情報
- 名前: {job['name']}
- ステップ: {job.get('failed_step', 'N/A')}

## エラー内容
{chr(10).join(errors[:5])}

## PR で変更されたファイル
{chr(10).join(f'- {c["filename"]} (+{c["additions"]} -{c["deletions"]})' for c in changes[:10])}

## 分析依頼
1. 失敗の原因（変更との関連）
2. 修正方法
3. ローカルでの再現手順
"""

    def _format_pr_comment(self, analyses: list) -> str:
        """PRコメント用のフォーマット"""
        comment = "## AI デバッグ分析結果\n\n"
        for analysis in analyses:
            comment += f"### {analysis['job_name']}\n"
            comment += f"**検出されたエラー**: {len(analysis['errors'])}件\n"
            if analysis['related_changes']:
                comment += f"**関連する変更ファイル**: "
                comment += ", ".join(
                    f"`{c['filename']}`" for c in analysis['related_changes'][:3]
                )
                comment += "\n"
            comment += "\n"
        return comment
```

---

## 5. 比較表

| AI デバッグツール | 対象 | 統合先 | リアルタイム | コスト |
|-----------------|:----:|:-----:|:--------:|:-----:|
| Claude Code | 全般 | ターミナル | 対話型 | API 従量制 |
| GitHub Copilot Chat | 全般 | VS Code, JetBrains | 対話型 | $10-39/月 |
| Cursor AI | 全般 | Cursor エディタ | 対話型 | $20/月 |
| Sentry AI | 本番エラー | Sentry ダッシュボード | 自動 | $26/月〜 |
| Datadog AI | ログ・APM | Datadog | 自動 | $15/ホスト〜 |

| デバッグアプローチ | 速度 | 精度 | コンテキスト理解 | 適用範囲 |
|------------------|:---:|:---:|:-----------:|:------:|
| AI にエラーを貼り付け | 高 | 中 | 低 | 汎用 |
| AI + ソースコード提供 | 中 | 高 | 高 | 汎用 |
| AI + ログ + トレース | 低 | 最高 | 最高 | 本番問題 |
| AI + デバッグ記録検索 | 高 | 高 | 高 | 既知問題 |
| 手動デバッグのみ | 低 | 最高 | 最高 | 全て |

---

## 6. アンチパターン

### アンチパターン 1: AI の回答を鵜呑みにする

```
BAD:
  エラーメッセージを AI に貼り付け → 提案された修正をそのまま適用
  → AI が見当違いな修正を提案（コンテキスト不足による誤診）
  → 別のバグを埋め込む、根本原因が解決されない

GOOD:
  1. AI に十分なコンテキストを提供する
     - エラーメッセージだけでなく、関連コード、ログ、環境情報
  2. AI の回答を「仮説」として扱う
     - 提案された原因を自分で検証する
  3. 修正前にテストで確認する
     - 既存テストが通ること + 新しいテストを追加
  4. 根本原因を理解してから修正する
     - 「なぜそのバグが発生したか」を説明できる状態にする
```

### アンチパターン 2: コンテキスト不足の質問

```
BAD:
  「TypeError: Cannot read properties of undefined が出ます」
  → AI は一般的な回答しかできない
  → 実際の原因と無関係な対策を提案される

GOOD:
  エラー報告テンプレートに沿って情報を提供:

  1. エラーメッセージ + 完全なスタックトレース
  2. エラーが発生するコード（前後20行）
  3. 期待される動作 vs 実際の動作
  4. 再現手順
  5. 環境情報 (ランタイムバージョン、OS、依存関係)
  6. 最近の変更 (git log --oneline -5)
  7. 既に試したこと
```

### アンチパターン 3: デバッグ知識を個人に閉じ込める

```
BAD:
  ベテランエンジニアだけが特定のバグの直し方を知っている
  → 属人化、チームのデバッグ速度にばらつき
  → 同じバグが何度も再発しても同じ調査が繰り返される

GOOD:
  - デバッグセッションの記録を残す（原因、調査過程、修正方法）
  - チームのエラーパターンナレッジベースを構築
  - ポストモーテムを書き、根本原因と再発防止策を共有
  - AI チャットのやり取りで有用だったものをチームに共有
  - 新メンバーのオンボーディングにデバッグ事例集を活用
```

### アンチパターン 4: 機密情報の無配慮な共有

```
BAD:
  本番環境のログをそのまま外部AIサービスに送信
  → API キー、ユーザー個人情報、データベースの接続情報が漏洩
  → コンプライアンス違反、セキュリティインシデント

GOOD:
  1. 送信前にセンシティブ情報をマスキング
     - API キー: sk-****** → [REDACTED_API_KEY]
     - メールアドレス: user@example.com → [EMAIL]
     - IPアドレス: 192.168.1.1 → [IP_ADDRESS]
  2. 社内ホスト型のLLMを使用（AWS Bedrock等）
  3. データ処理規約を確認（GDPR、個人情報保護法）
  4. 監査ログにAIへの送信内容を記録
```

---

## 7. 実践シナリオ: インシデント対応でのAI活用

### 7.1 本番障害対応のタイムライン

```
時刻    イベント                            AI活用ポイント
────────────────────────────────────────────────────────────
00:00   アラート発報（エラーレート急上昇）     → AI: アラート内容の要約
00:02   オンコールエンジニアが確認開始         → AI: 類似インシデントの検索
00:05   ログ調査開始                          → AI: エラーログのパターン分析
00:08   原因仮説の立案                        → AI: 仮説の検証ポイント提案
00:12   原因特定（DB接続プール枯渇）          → AI: 緩和策のコード生成
00:15   緩和策適用（コネクション上限引き上げ）  → AI: 影響範囲の評価
00:20   正常性確認                            → AI: 確認すべきメトリクス一覧
00:30   ポストモーテム作成開始                 → AI: タイムラインの整理
01:00   ポストモーテム完了・根本対策策定       → AI: 再発防止策の提案
```

### 7.2 ポストモーテムのAI支援テンプレート

```python
# インシデント後のポストモーテム作成をAIで支援

POSTMORTEM_AI_PROMPT = """
以下のインシデント情報に基づいて、ポストモーテムのドラフトを作成してください。

## インシデント概要
- 発生日時: {incident_start}
- 復旧日時: {incident_end}
- 影響範囲: {impact}
- 深刻度: {severity}

## タイムライン
{timeline}

## 検知方法
{detection}

## 根本原因
{root_cause}

## 対応アクション
{actions_taken}

## テンプレート
以下の形式でポストモーテムを出力してください:

### 1. サマリー（3行以内）
### 2. インパクト（定量的に）
### 3. タイムライン（時系列）
### 4. 根本原因分析（5 Whys）
### 5. 修正内容
### 6. 再発防止策（短期・中期・長期）
### 7. 学んだこと（Good / Bad / Lucky）
"""
```

---

## 8. FAQ

### Q1. AI デバッグで最も効果が高いのはどのような場面か？

**A.** (1) **エラーメッセージの解読**: 見慣れないライブラリのエラーや暗号的なメッセージの意味を即座に理解できる。(2) **依存関係のバージョン不整合**: パッケージの互換性問題を AI が過去の事例から特定。(3) **大量ログからのパターン抽出**: 人間が目視で確認するのが困難な大量ログから異常パターンを抽出。(4) **類似バグの検索**: 「以前似たエラーを見た」という曖昧な記憶を AI が具体的な事例としてマッチング。逆に、ビジネスロジックのバグや環境固有の問題は AI の苦手領域。

### Q2. 本番環境のデバッグに AI を安全に使うには？

**A.** (1) **機密情報のマスキング**: ログやコードを AI に渡す前に、パスワード、API キー、個人情報をマスクする。(2) **社内 LLM の活用**: 機密性の高いコードは社内でホストした LLM で処理する（AWS Bedrock、Azure OpenAI 等）。(3) **権限の分離**: AI が直接本番環境にアクセスする仕組みは作らない。人間が AI の提案を検証してから適用する。(4) **監査ログ**: AI に渡した情報と受け取った提案のログを残す。

### Q3. チームとして AI デバッグ力を向上させるには？

**A.** (1) **プロンプトテンプレートの標準化**: チーム共通のデバッグプロンプトテンプレートを作成し、必要な情報が漏れなく提供されるようにする。(2) **ナレッジベースの構築**: 過去のデバッグセッション（原因、調査過程、修正方法）を記録し、AI による類似検索を可能にする。(3) **ペアデバッグ**: AI との対話を画面共有しながらペアで行い、効果的なプロンプトの書き方を共有する。(4) **定期的な振り返り**: 月次でデバッグ効率のメトリクス（MTTR: 平均復旧時間）を計測し、改善点を議論する。

### Q4. AIデバッグの精度を上げるためのコツは？

**A.** (1) **段階的に情報を提供**: まずエラーメッセージとスタックトレースで初期分析、その結果に基づいて関連コードや設定を追加提供する。(2) **仮説を複数立てさせる**: 「原因を3つ挙げて、それぞれの確率と根拠を示して」と指示する。(3) **否定的な情報も提供**: 「DBは正常に動作している」「ネットワークに問題はない」など、排除できる原因を伝える。(4) **再現コードを添える**: 最小限の再現コードがあればAIの精度は大幅に向上する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| エラー解析 | エラーメッセージ + スタックトレース + 関連コード + 環境情報を構造化して AI に渡す |
| プロンプト設計 | コンテキスト量が精度を決める。テンプレートで情報の漏れを防止 |
| ログ分析 | 構造化ログ + AI パターン解析で大量ログから異常を検出 |
| 分散トレーシング | OpenTelemetry のトレースデータを AI で分析しボトルネック特定 |
| 自動修正 | AI の提案は「仮説」。テストで検証してから適用 |
| ナレッジ共有 | デバッグセッションを記録し、チームのナレッジベースを構築 |
| インシデント対応 | タイムラインの整理からポストモーテム作成までAIが支援 |
| セキュリティ | 機密情報のマスキング、社内LLMの活用、監査ログの記録 |

---

## 次に読むべきガイド

- [AIドキュメント生成](./02-ai-documentation.md) -- エラーレポートの自動ドキュメント化
- [AIコーディング](./01-ai-coding.md) -- AI によるコード生成とバグ予防
- [AI倫理と開発](../03-team/03-ai-ethics-development.md) -- AI 活用の倫理的配慮

---

## 参考文献

1. **Debugging: The 9 Indispensable Rules** -- David J. Agans (AMACOM, 2002) -- デバッグの基本原則
2. **Sentry Documentation** -- https://docs.sentry.io/ -- エラートラッキングプラットフォーム
3. **OpenTelemetry Documentation** -- https://opentelemetry.io/docs/ -- 分散トレーシングの標準
4. **Structured Logging with structlog** -- https://www.structlog.org/ -- Python 構造化ログライブラリ
