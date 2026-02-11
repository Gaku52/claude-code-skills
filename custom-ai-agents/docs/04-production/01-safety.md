# AI エージェントの安全性 — ガードレール・人間監視・制限

> 自律的に行動する AI エージェントが暴走せず、安全かつ制御可能に動作するための技術的ガードレール、人間によるオーバーサイト、権限制限の設計パターンを体系的に学ぶ。

---

## この章で学ぶこと

1. **ガードレール設計** — 入力検証、出力フィルタリング、アクション制限による多層防御の実装
2. **人間監視 (Human-in-the-Loop)** — 承認ワークフロー、エスカレーション、介入メカニズムの設計
3. **権限制限とサンドボックス** — 最小権限の原則、リソース制限、実行環境の隔離

---

## 1. エージェント安全性の全体像

### 1.1 安全性の多層防御モデル

```
+------------------------------------------------------------------+
|                    多層防御アーキテクチャ                           |
+------------------------------------------------------------------+
|                                                                    |
|  Layer 1: 入力ガード                                               |
|  +------------------------------------------------------------+  |
|  | プロンプトインジェクション検出 | 入力バリデーション | レート制限 |  |
|  +------------------------------------------------------------+  |
|                              |                                     |
|  Layer 2: エージェントコア                                         |
|  +------------------------------------------------------------+  |
|  | システムプロンプト | ツール権限制御 | コンテキスト制限          |  |
|  +------------------------------------------------------------+  |
|                              |                                     |
|  Layer 3: アクションガード                                         |
|  +------------------------------------------------------------+  |
|  | ツール呼び出し検証 | 破壊的操作の承認 | 実行前確認             |  |
|  +------------------------------------------------------------+  |
|                              |                                     |
|  Layer 4: 出力ガード                                               |
|  +------------------------------------------------------------+  |
|  | 有害性チェック | PII検出 | 品質検証 | 一貫性チェック            |  |
|  +------------------------------------------------------------+  |
|                              |                                     |
|  Layer 5: 実行環境                                                 |
|  +------------------------------------------------------------+  |
|  | サンドボックス | リソース制限 | ネットワーク制限 | タイムアウト  |  |
|  +------------------------------------------------------------+  |
|                                                                    |
+------------------------------------------------------------------+
```

### 1.2 エージェントのリスクマトリクス

```
影響度
  高 |  監視必須     承認必須     禁止
     |  (メール送信)  (課金操作)   (データ削除)
     |
  中 |  ログ記録     監視必須     承認必須
     |  (検索)       (ファイル作成) (外部API)
     |
  低 |  制限なし     ログ記録     監視必須
     |  (計算)       (読み取り)    (設定変更)
     +----------------------------------------
        低           中           高
                   可逆性の低さ (非可逆度)
```

---

## 2. ガードレール設計

### 2.1 ツールコール検証システム

```python
# コード例 1: ツールコールのガードレール実装
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any
import re

class ActionLevel(Enum):
    ALLOW = "allow"          # 自動許可
    LOG = "log"              # ログ記録して許可
    CONFIRM = "confirm"      # 人間の確認が必要
    DENY = "deny"            # 拒否

@dataclass
class ToolPolicy:
    """各ツールのセキュリティポリシー"""
    tool_name: str
    default_level: ActionLevel
    max_calls_per_session: int
    allowed_parameters: dict | None  # パラメータの許容値
    validators: list[Callable]       # カスタムバリデータ

class ToolGuardrail:
    """ツール呼び出しのガードレール"""

    def __init__(self, policies: list[ToolPolicy]):
        self.policies = {p.tool_name: p for p in policies}
        self.call_counts: dict[str, int] = {}

    async def check(self, tool_name: str, parameters: dict,
                     context: dict) -> tuple[ActionLevel, str]:
        """ツール呼び出しを検証し、アクションレベルを返す"""

        policy = self.policies.get(tool_name)
        if policy is None:
            return ActionLevel.DENY, f"未登録のツール: {tool_name}"

        # 1. 呼び出し回数チェック
        count = self.call_counts.get(tool_name, 0)
        if count >= policy.max_calls_per_session:
            return ActionLevel.DENY, (
                f"セッション上限超過: {tool_name} "
                f"({count}/{policy.max_calls_per_session})"
            )

        # 2. パラメータ検証
        if policy.allowed_parameters:
            for param, allowed in policy.allowed_parameters.items():
                value = parameters.get(param)
                if value is not None and value not in allowed:
                    return ActionLevel.DENY, (
                        f"パラメータ '{param}' の値 '{value}' は"
                        f"許可されていません"
                    )

        # 3. カスタムバリデータ
        for validator in policy.validators:
            result = await validator(parameters, context)
            if not result["ok"]:
                return ActionLevel.DENY, result["reason"]

        # 4. 呼び出しカウント更新
        self.call_counts[tool_name] = count + 1

        return policy.default_level, "OK"


# ポリシーの定義例
policies = [
    ToolPolicy(
        tool_name="web_search",
        default_level=ActionLevel.ALLOW,
        max_calls_per_session=50,
        allowed_parameters=None,
        validators=[],
    ),
    ToolPolicy(
        tool_name="send_email",
        default_level=ActionLevel.CONFIRM,
        max_calls_per_session=5,
        allowed_parameters={
            "to_domain": ["@company.com", "@partner.com"],
        },
        validators=[check_email_content_safety],
    ),
    ToolPolicy(
        tool_name="execute_code",
        default_level=ActionLevel.LOG,
        max_calls_per_session=20,
        allowed_parameters=None,
        validators=[check_code_safety, check_no_network_access],
    ),
    ToolPolicy(
        tool_name="delete_file",
        default_level=ActionLevel.CONFIRM,
        max_calls_per_session=3,
        allowed_parameters=None,
        validators=[check_not_system_file, check_backup_exists],
    ),
]
```

### 2.2 出力フィルタリング

```python
# コード例 2: エージェント出力の安全性フィルタリング
class OutputGuardrail:
    """エージェントの出力をフィルタリングする"""

    # PII パターン
    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone_jp": r"0\d{1,4}-?\d{1,4}-?\d{3,4}",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "my_number": r"\b\d{4}\s?\d{4}\s?\d{4}\b",  # マイナンバー
    }

    async def filter_output(self, output: str, context: dict) -> dict:
        """出力を検証し、必要に応じてフィルタリングする"""
        issues = []

        # 1. PII 検出
        pii_found = self._detect_pii(output)
        if pii_found:
            output = self._mask_pii(output, pii_found)
            issues.append({
                "type": "pii_detected",
                "count": len(pii_found),
                "action": "masked"
            })

        # 2. 有害性チェック
        toxicity = await self._check_toxicity(output)
        if toxicity["score"] > 0.8:
            issues.append({
                "type": "toxic_content",
                "score": toxicity["score"],
                "action": "blocked"
            })
            return {
                "output": "[安全性フィルタにより出力がブロックされました]",
                "issues": issues,
                "blocked": True
            }

        # 3. 機密情報の漏洩チェック
        if self._contains_system_prompt(output, context):
            issues.append({
                "type": "system_prompt_leak",
                "action": "blocked"
            })
            return {
                "output": "[システム情報の漏洩を検出しブロックしました]",
                "issues": issues,
                "blocked": True
            }

        return {
            "output": output,
            "issues": issues,
            "blocked": False
        }

    def _detect_pii(self, text: str) -> list[dict]:
        findings = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            for match in matches:
                findings.append({"type": pii_type, "value": match})
        return findings

    def _mask_pii(self, text: str, findings: list[dict]) -> str:
        for finding in findings:
            text = text.replace(
                finding["value"],
                f"[{finding['type'].upper()}_MASKED]"
            )
        return text
```

---

## 3. 人間監視 (Human-in-the-Loop)

### 3.1 承認ワークフロー

```
+------------------------------------------------------------------+
|                    Human-in-the-Loop パターン                      |
+------------------------------------------------------------------+
|                                                                    |
|  パターン A: 事前承認 (Pre-Approval)                               |
|  エージェント → [計画提示] → 人間承認 → 実行                       |
|  用途: 高リスク操作、不可逆な変更                                  |
|                                                                    |
|  パターン B: 事後確認 (Post-Verification)                          |
|  エージェント → 実行 → [結果提示] → 人間確認 → 確定/取消           |
|  用途: 中リスク操作、取り消し可能な変更                            |
|                                                                    |
|  パターン C: 監視 (Monitoring)                                     |
|  エージェント → 実行 → ログ記録 → [異常時のみ] → 人間アラート      |
|  用途: 低リスク操作、大量の定型処理                                |
|                                                                    |
|  パターン D: エスカレーション (Escalation)                         |
|  エージェント → 自信度判定 → [低自信度] → 人間に委譲               |
|                           → [高自信度] → 自動実行                  |
|  用途: 判断の確実性が変動する場面                                  |
+------------------------------------------------------------------+
```

### 3.2 承認ワークフローの実装

```python
# コード例 3: Human-in-the-Loop 承認フロー
import asyncio
from enum import Enum

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"

class HumanApprovalGate:
    """人間の承認を要求するゲート"""

    def __init__(self, notification_service, timeout_seconds=300):
        self.notification_service = notification_service
        self.timeout = timeout_seconds
        self.pending_requests: dict[str, asyncio.Future] = {}

    async def request_approval(self, action_description: str,
                                parameters: dict,
                                risk_level: str,
                                reviewer_id: str) -> ApprovalStatus:
        """人間の承認をリクエストし、結果を待つ"""

        request_id = str(uuid.uuid4())

        # 承認リクエストを通知
        await self.notification_service.send(
            to=reviewer_id,
            message={
                "type": "approval_request",
                "request_id": request_id,
                "action": action_description,
                "parameters": parameters,
                "risk_level": risk_level,
                "expires_at": time.time() + self.timeout,
            }
        )

        # 承認待ちの Future を作成
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[request_id] = future

        try:
            # タイムアウト付きで承認を待つ
            result = await asyncio.wait_for(future, timeout=self.timeout)
            return result
        except asyncio.TimeoutError:
            return ApprovalStatus.TIMEOUT
        finally:
            self.pending_requests.pop(request_id, None)

    async def submit_decision(self, request_id: str,
                               status: ApprovalStatus,
                               comment: str = ""):
        """レビューアーが承認/拒否を提出する"""
        future = self.pending_requests.get(request_id)
        if future and not future.done():
            future.set_result(status)


class SafeAgent:
    """安全なエージェント実行フレームワーク"""

    def __init__(self, llm, tools, guardrail, approval_gate):
        self.llm = llm
        self.tools = tools
        self.guardrail = guardrail
        self.approval_gate = approval_gate

    async def execute_action(self, tool_name: str,
                              parameters: dict, context: dict):
        """ガードレール付きでアクションを実行する"""

        # Step 1: ガードレールチェック
        level, reason = await self.guardrail.check(
            tool_name, parameters, context
        )

        if level == ActionLevel.DENY:
            return {"status": "denied", "reason": reason}

        if level == ActionLevel.CONFIRM:
            # Step 2: 人間の承認を要求
            status = await self.approval_gate.request_approval(
                action_description=f"{tool_name}({parameters})",
                parameters=parameters,
                risk_level="high",
                reviewer_id=context["owner_id"],
            )

            if status != ApprovalStatus.APPROVED:
                return {"status": "rejected", "approval": status.value}

        # Step 3: 実行
        try:
            result = await self.tools[tool_name].execute(parameters)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
```

---

## 4. 権限制限とサンドボックス

### 4.1 最小権限設計

```python
# コード例 4: ロールベースのエージェント権限管理
from dataclasses import dataclass, field

@dataclass
class AgentPermissions:
    """エージェントの権限定義"""

    # ファイルシステム
    allowed_read_paths: list[str] = field(default_factory=list)
    allowed_write_paths: list[str] = field(default_factory=list)
    max_file_size_mb: int = 10

    # ネットワーク
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(
        default_factory=lambda: ["*.internal", "localhost"]
    )
    max_requests_per_minute: int = 30

    # コード実行
    allow_code_execution: bool = False
    allowed_languages: list[str] = field(default_factory=list)
    max_execution_time_seconds: int = 30
    max_memory_mb: int = 256

    # 外部サービス
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)

    # リソース制限
    max_tokens_per_session: int = 100_000
    max_tool_calls_per_session: int = 50
    session_timeout_minutes: int = 60


# ロール別のプリセット
ROLE_PRESETS = {
    "researcher": AgentPermissions(
        allowed_read_paths=["/data/public/", "/data/research/"],
        allowed_write_paths=["/output/research/"],
        allowed_domains=["*.arxiv.org", "*.wikipedia.org", "*.github.com"],
        allow_code_execution=True,
        allowed_languages=["python"],
        allowed_tools=["web_search", "read_file", "write_file",
                       "execute_python"],
        max_tool_calls_per_session=100,
    ),
    "customer_support": AgentPermissions(
        allowed_read_paths=["/data/faq/", "/data/products/"],
        allowed_write_paths=[],
        allowed_domains=["api.company.com"],
        allow_code_execution=False,
        allowed_tools=["search_faq", "lookup_order", "create_ticket"],
        max_tool_calls_per_session=30,
    ),
    "admin": AgentPermissions(
        allowed_read_paths=["/"],
        allowed_write_paths=["/data/", "/config/"],
        allowed_domains=["*"],
        allow_code_execution=True,
        allowed_languages=["python", "bash"],
        allowed_tools=["*"],
        max_tool_calls_per_session=200,
    ),
}
```

### 4.2 サンドボックス実行環境

```python
# コード例 5: Docker ベースのサンドボックスでコードを実行する
import docker
import tempfile
import os

class CodeSandbox:
    """Docker コンテナ内でエージェントのコードを安全に実行する"""

    def __init__(self, permissions: AgentPermissions):
        self.client = docker.from_env()
        self.permissions = permissions

    async def execute(self, code: str, language: str = "python") -> dict:
        """サンドボックス内でコードを実行する"""

        if language not in self.permissions.allowed_languages:
            return {"error": f"言語 '{language}' は許可されていません"}

        # 一時ファイルにコードを書き出す
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            code_path = f.name

        try:
            container = self.client.containers.run(
                image="python:3.11-slim",
                command=f"python /code/script.py",
                volumes={
                    code_path: {"bind": "/code/script.py", "mode": "ro"}
                },
                # セキュリティ制限
                mem_limit=f"{self.permissions.max_memory_mb}m",
                cpu_period=100000,
                cpu_quota=50000,     # CPU 50%制限
                network_disabled=True,  # ネットワーク無効
                read_only=True,      # ファイルシステム読み取り専用
                security_opt=["no-new-privileges"],
                # タイムアウト
                detach=True,
            )

            # 実行完了を待つ（タイムアウト付き）
            result = container.wait(
                timeout=self.permissions.max_execution_time_seconds
            )

            logs = container.logs().decode("utf-8")
            exit_code = result["StatusCode"]

            return {
                "stdout": logs,
                "exit_code": exit_code,
                "success": exit_code == 0,
            }

        except docker.errors.ContainerError as e:
            return {"error": f"実行エラー: {e}"}
        except Exception as e:
            return {"error": f"サンドボックスエラー: {e}"}
        finally:
            os.unlink(code_path)
            try:
                container.remove(force=True)
            except:
                pass
```

---

## 5. 監視とアラート

### 5.1 エージェント行動の監視ダッシュボード

```
+------------------------------------------------------------------+
|                エージェント監視ダッシュボード                       |
+------------------------------------------------------------------+
|                                                                    |
|  リアルタイム指標:                                                 |
|  +------------------+  +------------------+  +------------------+ |
|  | アクティブ       |  | ツール呼び出し   |  | エラー率         | |
|  | セッション: 142  |  | QPS: 2,340      |  | 0.3%             | |
|  +------------------+  +------------------+  +------------------+ |
|                                                                    |
|  +------------------+  +------------------+  +------------------+ |
|  | 承認待ち: 7      |  | 拒否された       |  | 平均セッション   | |
|  |                  |  | アクション: 23   |  | 時間: 4.2min     | |
|  +------------------+  +------------------+  +------------------+ |
|                                                                    |
|  アラート条件:                                                     |
|  [!] 同一ツールの連続呼び出し > 10回/分                            |
|  [!] エラー率 > 5%                                                 |
|  [!] セッション時間 > 30分                                         |
|  [!] 拒否されたアクション率 > 10%                                  |
|  [!] トークン消費量 > 80% of limit                                 |
+------------------------------------------------------------------+
```

---

## 6. パターン比較

### 6.1 安全性パターンの比較

| パターン | 安全性 | ユーザー体験 | 実装コスト | 適用場面 |
|----------|--------|-------------|-----------|---------|
| 全操作事前承認 | 極高 | 低 (遅延大) | 低 | 金融、医療 |
| リスクベース承認 | 高 | 中 | 中 | 一般業務 |
| 事後監視 + 取消 | 中 | 高 | 中 | 社内ツール |
| 自信度ベースエスカレーション | 中〜高 | 高 | 高 | カスタマーサポート |
| サンドボックス + ログ | 中 | 高 | 高 | 開発・研究 |

### 6.2 エージェント権限モデルの比較

| モデル | 粒度 | 柔軟性 | 管理コスト | 適用場面 |
|--------|------|--------|-----------|---------|
| ホワイトリスト | ツール単位 | 低 | 低 | シンプルなエージェント |
| RBAC | ロール単位 | 中 | 中 | 組織内エージェント |
| ABAC | 属性ベース | 高 | 高 | 複雑な権限要件 |
| Capability-based | 能力トークン | 高 | 高 | マルチエージェント |

---

## 7. アンチパターン

### アンチパターン 1: 「全権限を付与した汎用エージェント」

```
[誤り] エージェントに全ツール・全リソースへのアクセスを許可する

  「何でもできるAIアシスタント」
  → ファイル削除、メール送信、API呼び出しが全て自動実行
  → プロンプトインジェクションで全権限が悪用される

[正解] 最小権限の原則を適用する
  1. タスクに必要なツールのみを許可
  2. アクセス可能なファイル/ディレクトリを制限
  3. ネットワークアクセスをホワイトリストで管理
  4. セッションごとに権限を動的に付与
```

### アンチパターン 2: 「エージェントのループを放置」

```
[誤り] エージェントが無限ループや反復的な無意味な操作を
       検出・停止する仕組みがない

実際に起きた問題:
  - 同じ API を数千回呼び出してレート制限に到達
  - ファイルの読み書きを繰り返してディスクを圧迫
  - トークンを大量消費してコストが爆発

[正解] ループ検出とリソース制限を実装する
  - 同一ツールの連続呼び出し検出 (> N回で停止)
  - セッションあたりのトークン上限
  - ツール呼び出し回数の上限
  - タイムアウトの設定
  - コスト上限の設定（$XX/セッション）
```

---

## 8. FAQ

### Q1: エージェントにどの程度の自律性を与えるべきですか？

**A:** リスクと業務効率のバランスで段階的に決定します。

- **Level 1 (推薦)**: エージェントが提案し、人間が全て実行する
- **Level 2 (承認付き実行)**: エージェントが実行計画を立て、人間が承認後に実行する
- **Level 3 (監視付き自律)**: エージェントが自律実行し、人間は監視のみ。異常時に介入
- **Level 4 (完全自律)**: エージェントが完全に自律実行。定期的な事後レビューのみ

本番環境では Level 2〜3 から始め、信頼性が実証された後に Level を上げることを推奨します。

### Q2: マルチエージェント環境での安全性はどう設計しますか？

**A:** 以下の原則を適用します。

1. **エージェント間の権限分離**: 各エージェントは独自の権限セットを持ち、他のエージェントの権限を継承しない
2. **通信チャネルの制限**: エージェント間の通信は定義済みのプロトコルに限定し、直接的なプロンプト注入を防止
3. **オーケストレーターの監視**: 中央のオーケストレーターが全エージェントのアクションを監視し、異常な協調動作を検出
4. **最終決定権の集約**: 重要な意思決定は単一のエージェント（またはオーケストレーター）に集約

### Q3: コスト爆発を防ぐにはどうすればよいですか？

**A:** 多層的なコスト制限を設けます。

- **セッション単位**: 1セッションあたりの最大トークン数と最大ツール呼び出し数を設定
- **ユーザー単位**: 1日/月あたりの利用上限を設定
- **組織単位**: 月間予算の上限を設定し、閾値でアラート（80%, 90%, 95%）
- **リアルタイム監視**: コスト監視ダッシュボードで異常な消費パターンを検出
- **自動停止**: 予算の 100% に達したら全エージェントを自動停止

---

## 9. まとめ

| 安全対策 | 実装方法 | 目的 | 優先度 |
|----------|----------|------|--------|
| ツール権限制御 | ホワイトリスト + ポリシー | 不正操作の防止 | 必須 |
| 入力ガードレール | パターン + ML 検出 | インジェクション防止 | 必須 |
| 出力フィルタリング | PII検出 + 有害性チェック | 情報漏洩・有害出力防止 | 必須 |
| 人間承認ゲート | 非同期ワークフロー | 高リスク操作の制御 | 推奨 |
| サンドボックス | Docker + リソース制限 | 実行環境の隔離 | 推奨 |
| ループ検出 | 呼び出しパターン監視 | リソース浪費の防止 | 推奨 |
| 監査ログ | 全操作の記録 | 追跡可能性の確保 | 必須 |
| コスト制限 | 多層的な予算制御 | コスト爆発の防止 | 推奨 |

---

## 次に読むべきガイド

- [AI セーフティ](../../../llm-and-ai-comparison/docs/04-ethics/00-ai-safety.md) — アライメント・レッドチームの技術的手法
- [AI ガバナンス](../../../llm-and-ai-comparison/docs/04-ethics/01-ai-governance.md) — 規制・ポリシーの動向
- [責任ある AI](../../../ai-analysis-guide/docs/03-applied/03-responsible-ai.md) — 公平性・説明可能性・プライバシーの実装

---

## 参考文献

1. Wunderwuzzi. (2024). "Prompt Injection and AI Agents: Threats, Defenses, and Real-world Scenarios." *Embracethered*. https://embracethered.com/blog/
2. Yao, S. et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*. https://arxiv.org/abs/2210.03629
3. OWASP. (2025). "OWASP Top 10 for Large Language Model Applications." *OWASP Foundation*. https://owasp.org/www-project-top-10-for-large-language-model-applications/
