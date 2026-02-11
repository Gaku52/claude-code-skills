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

---

## 7. FAQ

### Q1. AI デバッグで最も効果が高いのはどのような場面か？

**A.** (1) **エラーメッセージの解読**: 見慣れないライブラリのエラーや暗号的なメッセージの意味を即座に理解できる。(2) **依存関係のバージョン不整合**: パッケージの互換性問題を AI が過去の事例から特定。(3) **大量ログからのパターン抽出**: 人間が目視で確認するのが困難な大量ログから異常パターンを抽出。(4) **類似バグの検索**: 「以前似たエラーを見た」という曖昧な記憶を AI が具体的な事例としてマッチング。逆に、ビジネスロジックのバグや環境固有の問題は AI の苦手領域。

### Q2. 本番環境のデバッグに AI を安全に使うには？

**A.** (1) **機密情報のマスキング**: ログやコードを AI に渡す前に、パスワード、API キー、個人情報をマスクする。(2) **社内 LLM の活用**: 機密性の高いコードは社内でホストした LLM で処理する（AWS Bedrock、Azure OpenAI 等）。(3) **権限の分離**: AI が直接本番環境にアクセスする仕組みは作らない。人間が AI の提案を検証してから適用する。(4) **監査ログ**: AI に渡した情報と受け取った提案のログを残す。

### Q3. チームとして AI デバッグ力を向上させるには？

**A.** (1) **プロンプトテンプレートの標準化**: チーム共通のデバッグプロンプトテンプレートを作成し、必要な情報が漏れなく提供されるようにする。(2) **ナレッジベースの構築**: 過去のデバッグセッション（原因、調査過程、修正方法）を記録し、AI による類似検索を可能にする。(3) **ペアデバッグ**: AI との対話を画面共有しながらペアで行い、効果的なプロンプトの書き方を共有する。(4) **定期的な振り返り**: 月次でデバッグ効率のメトリクス（MTTR: 平均復旧時間）を計測し、改善点を議論する。

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
