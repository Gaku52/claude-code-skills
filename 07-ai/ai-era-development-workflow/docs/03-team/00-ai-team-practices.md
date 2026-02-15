# AI時代のチーム開発プラクティス

> AIペアプログラミング、コード共有、レビュー文化、生産性向上など、AI時代に適応したチーム開発の方法論とベストプラクティスを体系的に解説する。

---

## この章で学ぶこと

1. **AIペアプログラミングの効果的な運用方法**を理解し、チーム全体の生産性を最大化できる
2. **AIを活用したコードレビュー・品質管理のプロセス**を設計し、品質と速度の両立を実現できる
3. **チームのAIリテラシー格差を解消**し、全員がAIを効果的に活用できる文化を構築できる

---

## 1. AI時代のチーム開発の全体像

### 1.1 従来型開発とAI時代の開発の変化

```
┌──────────────────────────────────────────────────────┐
│              開発ワークフローの進化                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  従来型                    AI時代                     │
│  ──────                   ──────                     │
│                                                      │
│  要件定義    ──────>   AI支援要件分析                 │
│  (数日)                (数時間 + LLMレビュー)         │
│                                                      │
│  設計       ──────>   AI生成設計案 + 人間レビュー     │
│  (数日)                (数時間)                       │
│                                                      │
│  実装       ──────>   AI補完/生成 + 人間監督         │
│  (数週間)              (数日)                        │
│                                                      │
│  レビュー    ──────>   AI事前レビュー + 人間承認      │
│  (数日)                (数時間)                       │
│                                                      │
│  テスト      ──────>   AIテスト生成 + 人間検証       │
│  (数日)                (数時間)                       │
│                                                      │
│  デバッグ    ──────>   AIエラー分析 + 人間判断       │
│  (不定)                (大幅短縮)                     │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 1.2 チームにおけるAIの役割分類

```
┌───────────────────────────────────────────────┐
│         チーム内AIの役割マトリクス               │
├───────────────────────────────────────────────┤
│                                               │
│  [ペアプログラマ]     [レビュアー]             │
│  ・コード補完         ・静的解析              │
│  ・リファクタリング    ・セキュリティチェック   │
│  ・バグ修正提案       ・コード品質評価        │
│                                               │
│  [ドキュメンター]     [テスター]              │
│  ・API文書自動生成    ・テストケース生成      │
│  ・READMEメンテ       ・エッジケース発見      │
│  ・コメント整理       ・回帰テスト提案        │
│                                               │
│  [アーキテクト補助]   [ナレッジベース]         │
│  ・設計パターン提案    ・コード検索            │
│  ・依存関係分析       ・過去事例の検索        │
│  ・技術選定支援       ・暗黙知の形式知化      │
│                                               │
└───────────────────────────────────────────────┘
```

---

## 2. AIペアプログラミングの実践

### 2.1 効果的なAIペアプロの原則

| 原則 | 説明 | 具体例 |
|------|------|-------|
| 人間が舵を取る | AIは提案者、人間が最終判断 | AIの生成コードを必ずレビュー |
| コンテキストを最大化 | AIに十分な文脈を与える | 関連ファイル、仕様書、テストを提供 |
| 段階的に依頼 | 大きなタスクを分割して依頼 | 関数単位→クラス単位→モジュール単位 |
| 検証可能な粒度 | 人間が検証できる量に制限 | 一度に100行以下のコード生成 |
| 学習機会の確保 | AIに丸投げせず理解する | 生成コードの意図を説明させる |

### 2.2 プロンプトテンプレートのチーム共有

```yaml
# .ai/prompts/code-review.yaml
# チーム共有プロンプトテンプレート
name: コードレビュー依頼
description: PRのコードレビューをAIに依頼する標準テンプレート
template: |
  以下のPull Requestをレビューしてください。

  ## コンテキスト
  - プロジェクト: {{project_name}}
  - 機能概要: {{feature_description}}
  - 対象ファイル: {{changed_files}}

  ## レビュー観点
  1. ロジックの正当性
  2. エッジケースの考慮
  3. パフォーマンスへの影響
  4. セキュリティリスク
  5. コーディング規約の遵守
  6. テストの十分性

  ## コーディング規約
  - {{coding_standards_url}}

  ## 変更差分
  ```diff
  {{diff_content}}
  ```

  ## 期待する出力
  - 重要度(Critical/Major/Minor)付きの指摘リスト
  - 各指摘に対する修正案
  - 全体的な評価コメント

variables:
  - project_name
  - feature_description
  - changed_files
  - coding_standards_url
  - diff_content
```

### 2.3 AIペアプロのセッション管理

```python
# AIペアプロセッション管理ツール
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

@dataclass
class AISession:
    """AIペアプロセッションの記録"""
    session_id: str
    developer: str
    ai_tool: str                # Claude, GPT, Copilot等
    task_type: str              # feature, bugfix, refactor, test
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    prompts_count: int = 0
    accepted_suggestions: int = 0
    rejected_suggestions: int = 0
    files_modified: list[str] = field(default_factory=list)
    notes: str = ""
    effectiveness_rating: int = 0  # 1-5

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_suggestions + self.rejected_suggestions
        return self.accepted_suggestions / total if total > 0 else 0.0

class AISessionTracker:
    """チーム全体のAIセッションを追跡"""

    def __init__(self, log_dir: str = ".ai/sessions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def start_session(self, developer: str, ai_tool: str, task_type: str) -> AISession:
        """新しいセッションを開始"""
        session_id = f"{developer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = AISession(
            session_id=session_id,
            developer=developer,
            ai_tool=ai_tool,
            task_type=task_type,
        )
        return session

    def end_session(self, session: AISession, rating: int = 3):
        """セッションを終了して保存"""
        session.end_time = datetime.now().isoformat()
        session.effectiveness_rating = rating

        path = self.log_dir / f"{session.session_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(session), f, indent=2, ensure_ascii=False)

    def team_stats(self) -> dict:
        """チーム全体の統計を集計"""
        sessions = []
        for path in self.log_dir.glob("*.json"):
            with open(path) as f:
                sessions.append(json.load(f))

        if not sessions:
            return {}

        return {
            "total_sessions": len(sessions),
            "avg_acceptance_rate": sum(
                s["accepted_suggestions"] /
                max(s["accepted_suggestions"] + s["rejected_suggestions"], 1)
                for s in sessions
            ) / len(sessions),
            "avg_effectiveness": sum(
                s["effectiveness_rating"] for s in sessions
            ) / len(sessions),
            "by_tool": self._group_by(sessions, "ai_tool"),
            "by_task_type": self._group_by(sessions, "task_type"),
            "by_developer": self._group_by(sessions, "developer"),
        }

    def _group_by(self, sessions: list, key: str) -> dict:
        groups = {}
        for s in sessions:
            g = s[key]
            if g not in groups:
                groups[g] = {"count": 0, "total_rating": 0}
            groups[g]["count"] += 1
            groups[g]["total_rating"] += s["effectiveness_rating"]
        for g in groups:
            groups[g]["avg_rating"] = groups[g]["total_rating"] / groups[g]["count"]
        return groups
```

---

## 3. AIを活用したコードレビュー

### 3.1 AI+人間の二段階レビューフロー

```
PR作成
  │
  v
┌──────────────────────┐
│  Stage 1: AIレビュー  │  (自動実行、数分)
├──────────────────────┤
│ ・静的解析            │
│ ・セキュリティスキャン │
│ ・コーディング規約     │
│ ・テストカバレッジ     │
│ ・パフォーマンス懸念   │
└──────────────────────┘
  │
  │ AI指摘のうち
  │ Critical → 即ブロック
  │ Major/Minor → コメント
  │
  v
┌──────────────────────┐
│  Stage 2: 人間レビュー │  (指名レビュアー)
├──────────────────────┤
│ ・ビジネスロジック     │
│ ・設計判断             │
│ ・ユーザー体験         │
│ ・チーム方針との整合    │
│ ・AI指摘の妥当性確認   │
└──────────────────────┘
  │
  v
マージ判断
```

### 3.2 GitHub Actions による自動AIレビュー

```yaml
# .github/workflows/ai-review.yml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Get diff
        id: diff
        run: |
          git diff origin/${{ github.base_ref }}...HEAD > /tmp/diff.txt

      - name: AI Review
        uses: actions/github-script@v7
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        with:
          script: |
            const fs = require('fs');
            const diff = fs.readFileSync('/tmp/diff.txt', 'utf8');

            // AIレビュー実行（Anthropic API呼び出し）
            const response = await fetch('https://api.anthropic.com/v1/messages', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'x-api-key': process.env.ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01',
              },
              body: JSON.stringify({
                model: 'claude-sonnet-4-20250514',
                max_tokens: 4096,
                messages: [{
                  role: 'user',
                  content: `以下のdiffをレビューしてください。
                    重要度(Critical/Major/Minor)付きで指摘し、
                    修正案を提示してください。\n\n${diff}`
                }],
              }),
            });

            const result = await response.json();
            const review = result.content[0].text;

            // PRにコメント投稿
            await github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: `## 🤖 AI Code Review\n\n${review}`,
            });
```

---

## 4. チーム生産性メトリクス

### 4.1 AI活用の効果測定

| メトリクス | 測定方法 | AI導入前の目安 | AI導入後の目安 |
|-----------|---------|--------------|--------------|
| PR作成〜マージ時間 | GitHubメトリクス | 2-5日 | 0.5-2日 |
| コードレビュー時間 | レビュー開始〜承認 | 4-8時間 | 1-3時間 |
| バグ検出率(レビュー) | レビュー指摘/全バグ | 30-50% | 50-70% |
| テストカバレッジ | CI計測 | 60-70% | 75-85% |
| 開発者満足度 | 月次サーベイ | ベースライン | +15-30% |
| 1人あたりPR数/週 | GitHub統計 | 3-5件 | 5-10件 |
| ドキュメント更新率 | コミット連動 | 20-30% | 60-80% |

### 4.2 DORA メトリクスへの影響

```
              AI導入の影響
              ────────────

  デプロイ頻度       ▲▲▲  大幅改善
  (週1→日複数回)

  変更リードタイム    ▲▲▲  大幅改善
  (数日→数時間)

  変更失敗率         ▲▲   改善
  (15%→8%)

  復旧時間           ▲    やや改善
  (数時間→1時間)

  ▲▲▲ = 大幅改善  ▲▲ = 改善  ▲ = やや改善
```

---

## 5. AIリテラシー格差の解消

### 5.1 チームAIスキルマトリクス

```python
# チームAIスキル評価・可視化ツール
from dataclasses import dataclass

@dataclass
class AISkillAssessment:
    """開発者のAIスキル評価"""
    developer: str
    prompt_engineering: int    # 1-5: プロンプト設計力
    tool_proficiency: int      # 1-5: AIツール操作力
    output_evaluation: int     # 1-5: AI出力の評価力
    workflow_integration: int  # 1-5: ワークフロー統合力
    teaching_ability: int      # 1-5: 他者への指導力

    @property
    def total_score(self) -> int:
        return (
            self.prompt_engineering
            + self.tool_proficiency
            + self.output_evaluation
            + self.workflow_integration
            + self.teaching_ability
        )

    @property
    def level(self) -> str:
        s = self.total_score
        if s >= 22:
            return "AI Champion"
        elif s >= 17:
            return "AI Practitioner"
        elif s >= 12:
            return "AI Learner"
        else:
            return "AI Beginner"

def generate_skill_matrix(team: list[AISkillAssessment]) -> str:
    """チームスキルマトリクスをテキスト表示"""
    header = (
        f"{'名前':12s} {'Prompt':8s} {'Tool':8s} "
        f"{'Eval':8s} {'Flow':8s} {'Teach':8s} {'Level':16s}"
    )
    lines = [header, "-" * len(header)]

    for member in sorted(team, key=lambda m: m.total_score, reverse=True):
        lines.append(
            f"{member.developer:12s} "
            f"{'*' * member.prompt_engineering:8s} "
            f"{'*' * member.tool_proficiency:8s} "
            f"{'*' * member.output_evaluation:8s} "
            f"{'*' * member.workflow_integration:8s} "
            f"{'*' * member.teaching_ability:8s} "
            f"{member.level:16s}"
        )

    return "\n".join(lines)
```

### 5.2 ペアローテーション制度

```
週次AIペアローテーション
──────────────────────

Week 1: Champion + Beginner
  → 基本操作の伝授、プロンプトの書き方

Week 2: Practitioner + Learner
  → 実タスクでのAI活用、ワークフロー統合

Week 3: Champion + Practitioner
  → 高度なテクニック共有、ツール評価

Week 4: チーム全体ワークショップ
  → 知見共有会、新ツール評価、プロンプト集更新
```

---

## 6. AIを活用した知識管理とナレッジベース

### 6.1 チーム知識の構造化・形式知化

```python
# AIを活用したチーム知識管理システム

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json

class KnowledgeType(Enum):
    DECISION = "decision"           # 設計判断・意思決定
    PATTERN = "pattern"             # コードパターン・慣習
    TROUBLESHOOT = "troubleshoot"   # トラブルシューティング
    DOMAIN = "domain"              # ドメイン知識
    PROCESS = "process"            # プロセス・手順
    TOOLING = "tooling"            # ツール使用法

@dataclass
class KnowledgeEntry:
    """知識ベースのエントリ"""
    id: str
    title: str
    knowledge_type: KnowledgeType
    content: str
    context: str                    # どのような状況で使うか
    created_by: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)
    ai_generated: bool = False      # AIが生成した知識か
    verified_by: str = ""           # 人間の検証者
    usage_count: int = 0            # 参照回数

@dataclass
class TeamKnowledgeBase:
    """チーム知識ベース管理"""
    team_name: str
    entries: list[KnowledgeEntry] = field(default_factory=list)
    kb_dir: Path = field(default_factory=lambda: Path(".ai/knowledge"))

    def add_from_code_review(
        self,
        pr_number: int,
        reviewer: str,
        learning: str,
        related_code: str,
    ) -> KnowledgeEntry:
        """コードレビューから知識を抽出して登録"""
        entry = KnowledgeEntry(
            id=f"review-{pr_number}-{len(self.entries)}",
            title=f"PR #{pr_number} からの学び",
            knowledge_type=KnowledgeType.PATTERN,
            content=learning,
            context=f"PR #{pr_number} のレビュー中に発見",
            created_by=reviewer,
            related_files=[related_code],
            tags=["code-review", f"pr-{pr_number}"],
        )
        self.entries.append(entry)
        return entry

    def add_from_incident(
        self,
        incident_id: str,
        responder: str,
        root_cause: str,
        fix_description: str,
        prevention: str,
    ) -> KnowledgeEntry:
        """インシデント対応から知識を抽出"""
        content = f"""
## 根本原因
{root_cause}

## 修正内容
{fix_description}

## 再発防止策
{prevention}
"""
        entry = KnowledgeEntry(
            id=f"incident-{incident_id}",
            title=f"インシデント {incident_id} の教訓",
            knowledge_type=KnowledgeType.TROUBLESHOOT,
            content=content,
            context=f"インシデント {incident_id} の事後分析",
            created_by=responder,
            tags=["incident", "postmortem"],
        )
        self.entries.append(entry)
        return entry

    def ai_generate_summary(self) -> str:
        """AIに知識ベースの要約を生成させるためのプロンプト"""
        entries_text = "\n".join(
            f"- [{e.knowledge_type.value}] {e.title}: {e.content[:100]}..."
            for e in self.entries[-20:]  # 直近20件
        )

        return f"""
以下のチーム知識ベースの直近エントリを分析し、
チームの技術的傾向と改善提案を生成してください。

チーム: {self.team_name}
エントリ数: {len(self.entries)}

直近のエントリ:
{entries_text}

分析してほしい観点:
1. 繰り返し発生している問題パターン
2. チームの技術的強み・弱み
3. 知識が不足している領域
4. 推奨するアクション（研修、ツール導入等）
"""

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeEntry]:
        """知識ベースをキーワード検索"""
        scored = []
        query_lower = query.lower()
        for entry in self.entries:
            score = 0
            if query_lower in entry.title.lower():
                score += 10
            if query_lower in entry.content.lower():
                score += 5
            for tag in entry.tags:
                if query_lower in tag.lower():
                    score += 3
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def save(self) -> None:
        """知識ベースをファイルに保存"""
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        for entry in self.entries:
            path = self.kb_dir / f"{entry.id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "id": entry.id,
                    "title": entry.title,
                    "type": entry.knowledge_type.value,
                    "content": entry.content,
                    "context": entry.context,
                    "created_by": entry.created_by,
                    "created_at": entry.created_at,
                    "tags": entry.tags,
                    "related_files": entry.related_files,
                    "ai_generated": entry.ai_generated,
                    "verified_by": entry.verified_by,
                    "usage_count": entry.usage_count,
                }, f, indent=2, ensure_ascii=False)
```

### 6.2 ADR（Architecture Decision Record）のAI支援

```python
# AIを活用したADR（アーキテクチャ決定記録）管理

from dataclasses import dataclass, field
from datetime import date

@dataclass
class ADREntry:
    """アーキテクチャ決定記録"""
    number: int
    title: str
    status: str  # "proposed", "accepted", "deprecated", "superseded"
    date: str
    context: str
    decision: str
    consequences: str
    alternatives_considered: list[str] = field(default_factory=list)
    ai_analysis: str = ""  # AIによるトレードオフ分析

class ADRManager:
    """ADR管理ツール"""

    def __init__(self, adr_dir: str = "docs/adr"):
        self.adr_dir = Path(adr_dir)
        self.adr_dir.mkdir(parents=True, exist_ok=True)

    def create_adr_prompt(
        self,
        title: str,
        context: str,
        options: list[str],
    ) -> str:
        """ADR作成のためのAIプロンプトを生成"""
        options_text = "\n".join(f"  {i+1}. {opt}" for i, opt in enumerate(options))

        return f"""
以下のアーキテクチャ決定について、ADRを作成してください。

## タイトル
{title}

## コンテキスト
{context}

## 検討中の選択肢
{options_text}

以下の形式で出力してください:

### 各選択肢のトレードオフ分析
（メリット、デメリット、リスクを具体的に）

### 推奨される決定
（理由と共に）

### 予測される影響
（短期的・長期的な影響）

### 将来の見直しトリガー
（この決定を見直すべき条件）
"""

    def generate_adr_markdown(self, entry: ADREntry) -> str:
        """ADRをMarkdown形式で生成"""
        alternatives = "\n".join(
            f"- {alt}" for alt in entry.alternatives_considered
        )

        return f"""# ADR-{entry.number:04d}: {entry.title}

## ステータス
{entry.status}

## 日付
{entry.date}

## コンテキスト
{entry.context}

## 検討した代替案
{alternatives}

## 決定
{entry.decision}

## AIによるトレードオフ分析
{entry.ai_analysis}

## 結果
{entry.consequences}
"""

    def save_adr(self, entry: ADREntry) -> Path:
        """ADRをファイルとして保存"""
        filename = f"{entry.number:04d}-{entry.title.replace(' ', '-').lower()}.md"
        path = self.adr_dir / filename
        content = self.generate_adr_markdown(entry)
        path.write_text(content, encoding="utf-8")
        return path
```

---

## 7. AI時代のコミュニケーションプラクティス

### 7.1 AI支援による非同期コミュニケーション

```yaml
# .ai/communication/templates.yaml
# チームコミュニケーション用AIテンプレート

templates:
  # PRの説明文を自動生成
  pr_description:
    name: "PR説明文自動生成"
    trigger: "PR作成時に自動実行"
    prompt: |
      以下のgit diffから、PR説明文を生成してください。

      ## 形式
      ### 変更概要
      （1-2文で変更の目的を説明）

      ### 変更内容
      （箇条書きで具体的な変更を列挙）

      ### テスト方法
      （動作確認の手順）

      ### 影響範囲
      （この変更が影響するコンポーネント・機能）

      ### レビュー観点
      （レビュアーに特に見てほしいポイント）

      ## diff
      {{diff}}

  # デイリースタンドアップのサマリー生成
  standup_summary:
    name: "スタンドアップサマリー"
    trigger: "毎朝9:00に自動実行"
    prompt: |
      以下のチームメンバーのGitHub活動データから、
      デイリースタンドアップ用のサマリーを生成してください。

      ## データソース
      - 昨日のコミット: {{commits}}
      - オープンPR: {{open_prs}}
      - マージされたPR: {{merged_prs}}
      - 新規Issue: {{new_issues}}

      ## 出力形式
      各メンバーについて:
      - 昨日やったこと
      - 今日の予定（推測）
      - ブロッカー（あれば）

  # Slackでの技術質問への自動回答
  tech_support:
    name: "技術質問自動回答"
    trigger: "#dev-support チャンネルへの投稿"
    prompt: |
      以下の技術質問に対して、チームの知識ベースと
      プロジェクトのドキュメントを参照して回答してください。

      質問: {{question}}

      参考ドキュメント: {{relevant_docs}}
      過去の類似質問: {{similar_questions}}

      回答後に「この回答は正確ですか？」と確認を求めてください。
```

### 7.2 会議の効率化

```python
# AI支援による会議の効率化ツール

from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MeetingAgenda:
    """AIが生成する会議アジェンダ"""
    title: str
    date: str
    duration_minutes: int
    participants: list[str]
    topics: list[dict] = field(default_factory=list)

@dataclass
class MeetingFacilitator:
    """AI会議ファシリテーター"""

    def generate_sprint_review_agenda(
        self,
        sprint_number: int,
        completed_stories: list[str],
        incomplete_stories: list[str],
        metrics: dict,
    ) -> str:
        """スプリントレビューのアジェンダを自動生成"""
        completed_list = "\n".join(f"  - {s}" for s in completed_stories)
        incomplete_list = "\n".join(f"  - {s}" for s in incomplete_stories)

        return f"""
# Sprint {sprint_number} レビュー アジェンダ

## 1. スプリント概要（5分）
- 期間: {metrics.get('start_date', 'N/A')} 〜 {metrics.get('end_date', 'N/A')}
- 計画ポイント: {metrics.get('planned_points', 0)}
- 完了ポイント: {metrics.get('completed_points', 0)}
- 達成率: {metrics.get('completion_rate', 0):.0%}

## 2. 完了ストーリーのデモ（20分）
{completed_list}

## 3. 未完了ストーリーの状況報告（10分）
{incomplete_list}

## 4. AI活用メトリクス（5分）
- AIペアプロセッション数: {metrics.get('ai_sessions', 0)}
- AI生成コード比率: {metrics.get('ai_code_ratio', 0):.0%}
- AIレビュー指摘の採用率: {metrics.get('ai_review_acceptance', 0):.0%}

## 5. 振り返り・改善提案（10分）
- チームからのフィードバック
- AI活用の改善ポイント

## 6. 次スプリントの優先事項（10分）
"""

    def generate_retro_prompts(self, sprint_metrics: dict) -> str:
        """レトロスペクティブのAI支援プロンプト"""
        return f"""
チームのスプリントメトリクスに基づいて、
レトロスペクティブの議論のたたき台を生成してください。

メトリクス:
- ベロシティ: {sprint_metrics.get('velocity', 'N/A')}
- テストカバレッジ: {sprint_metrics.get('coverage', 'N/A')}%
- バグ発生率: {sprint_metrics.get('bug_rate', 'N/A')}件/スプリント
- PRマージ平均時間: {sprint_metrics.get('pr_merge_time', 'N/A')}時間
- AI活用度: {sprint_metrics.get('ai_usage', 'N/A')}%

以下の形式で出力してください:

### うまくいったこと（Keep）
- メトリクスから読み取れるポジティブな傾向を3つ

### 改善すべきこと（Problem）
- メトリクスから読み取れる課題を3つ

### 試したいこと（Try）
- 具体的な改善アクションを3つ（AI活用の観点含む）
"""
```

---

## 8. チームのAIガバナンスフレームワーク

### 8.1 AIガバナンスポリシーの策定

```yaml
# .ai/governance/ai-usage-policy.yaml
# チームAI利用ポリシー

policy:
  version: "2.0"
  last_updated: "2026-02-01"
  approved_by: "Engineering Manager"

  # データセキュリティ
  data_security:
    allowed_data_types:
      - "オープンソースコード"
      - "社内技術ドキュメント（機密以外）"
      - "テストデータ（匿名化済み）"
    prohibited_data_types:
      - "顧客の個人情報（PII）"
      - "認証情報（APIキー、パスワード、トークン）"
      - "財務データ（未公開）"
      - "医療データ"
      - "契約書・法務文書"
    encryption_requirement: "転送中のデータはTLS 1.3以上"

  # AIツール使用ルール
  tool_usage:
    approved_tools:
      - name: "Claude Code"
        allowed_for: ["コード生成", "レビュー", "テスト", "ドキュメント"]
        restrictions: "機密コードには使用禁止"
      - name: "GitHub Copilot"
        allowed_for: ["コード補完", "テスト生成"]
        restrictions: "Telemetry無効化必須"
    approval_required_for:
      - "新しいAIツールの導入"
      - "AIツールの本番環境への統合"
      - "カスタムAIモデルのデプロイ"

  # コード品質ルール
  code_quality:
    ai_generated_code_rules:
      - "全てのAI生成コードは人間のレビューを通すこと"
      - "セキュリティクリティカルなコードはシニアエンジニア以上がレビュー"
      - "AI生成コードにはテストの追加が必須"
      - "PRの説明にAI支援の範囲を記載"
    minimum_test_coverage: 80
    mandatory_security_scan: true

  # 監査・記録
  audit:
    log_ai_usage: true
    retention_period_days: 365
    quarterly_review: true
    metrics_tracking:
      - "AI生成コード比率"
      - "AI生成コードの欠陥密度"
      - "AIレビュー指摘の精度"
```

### 8.2 コンプライアンスチェッカー

```python
# AIガバナンスポリシーの自動チェックツール

import re
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ComplianceIssue:
    """コンプライアンス違反"""
    severity: str  # "critical", "warning", "info"
    category: str
    message: str
    file: str = ""
    line: int = 0

class AIGovernanceChecker:
    """AIガバナンスポリシーのコンプライアンスチェック"""

    # チェック対象パターン
    SENSITIVE_PATTERNS = {
        "pii_email": {
            "pattern": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "message": "メールアドレスが検出されました",
            "severity": "critical",
            "exclude_files": ["*.test.*", "*.spec.*"],
        },
        "api_key": {
            "pattern": r'(?:api[_-]?key|apikey)\s*[=:]\s*["\'][a-zA-Z0-9]{20,}',
            "message": "APIキーのハードコードが検出されました",
            "severity": "critical",
        },
        "password": {
            "pattern": r'(?:password|passwd|pwd)\s*[=:]\s*["\'][^"\']{4,}',
            "message": "パスワードのハードコードが検出されました",
            "severity": "critical",
        },
        "private_key": {
            "pattern": r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----',
            "message": "秘密鍵が検出されました",
            "severity": "critical",
        },
        "ip_address": {
            "pattern": r'\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b',
            "message": "内部IPアドレスが検出されました",
            "severity": "warning",
        },
    }

    def __init__(self):
        self.issues: list[ComplianceIssue] = []

    def check_file(self, file_path: Path) -> list[ComplianceIssue]:
        """ファイルのコンプライアンスチェック"""
        issues = []
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            return issues

        for name, check in self.SENSITIVE_PATTERNS.items():
            # 除外ファイルの確認
            excludes = check.get("exclude_files", [])
            if any(file_path.match(exc) for exc in excludes):
                continue

            for match in re.finditer(check["pattern"], content, re.IGNORECASE):
                line_num = content[:match.start()].count("\n") + 1
                issue = ComplianceIssue(
                    severity=check["severity"],
                    category="data_security",
                    message=check["message"],
                    file=str(file_path),
                    line=line_num,
                )
                issues.append(issue)

        self.issues.extend(issues)
        return issues

    def check_pr_description(self, description: str) -> list[ComplianceIssue]:
        """PRの説明文にAI支援の記載があるかチェック"""
        issues = []

        # AI支援の記載確認
        ai_keywords = ["AI", "Copilot", "Claude", "GPT", "AI支援", "AI生成"]
        has_ai_mention = any(kw.lower() in description.lower() for kw in ai_keywords)

        if not has_ai_mention:
            issues.append(ComplianceIssue(
                severity="warning",
                category="transparency",
                message="PR説明にAI支援の範囲が記載されていません。"
                        "AIを使用した場合は記載してください。",
            ))

        self.issues.extend(issues)
        return issues

    def generate_report(self) -> str:
        """コンプライアンスレポートを生成"""
        critical = [i for i in self.issues if i.severity == "critical"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        infos = [i for i in self.issues if i.severity == "info"]

        lines = [
            "# AIガバナンス コンプライアンスレポート\n",
            f"チェック日時: {datetime.now().isoformat()}",
            f"検出数: Critical {len(critical)}, Warning {len(warnings)}, Info {len(infos)}\n",
        ]

        if critical:
            lines.append("## Critical Issues（即座に対応が必要）\n")
            for issue in critical:
                lines.append(f"- [{issue.category}] {issue.message}")
                if issue.file:
                    lines.append(f"  ファイル: {issue.file}:{issue.line}")

        if warnings:
            lines.append("\n## Warnings（確認推奨）\n")
            for issue in warnings:
                lines.append(f"- [{issue.category}] {issue.message}")
                if issue.file:
                    lines.append(f"  ファイル: {issue.file}:{issue.line}")

        return "\n".join(lines)
```

---

## 9. アンチパターン

### 6.1 アンチパターン：AI出力を無検証で採用

```
NG: AIが生成したコードをコピペしてそのままコミット
  - セキュリティ脆弱性の混入リスク
  - プロジェクト固有のパターンとの不整合
  - テストなしでのデプロイ

OK: AI出力の検証フロー
  1. AIがコードを生成
  2. 開発者がロジックを理解・検証
  3. 既存テストを実行して回帰確認
  4. 新規テストを追加
  5. コードレビューで他者の目を通す
  6. CI/CDパイプラインで自動チェック
```

**問題点**: AIは自信を持って誤ったコードを生成することがある。特にセキュリティ関連やビジネスロジックでは人間の検証が不可欠。

### 6.2 アンチパターン：チーム内でプロンプトが属人化

```
NG: 各自がバラバラのプロンプトでAIを使用
  - 品質にバラつき
  - 知見が共有されない
  - 新メンバーが効果的に使えない

OK: プロンプトライブラリの共有管理
  .ai/
  ├── prompts/
  │   ├── code-review.yaml
  │   ├── test-generation.yaml
  │   ├── refactoring.yaml
  │   ├── documentation.yaml
  │   └── debugging.yaml
  ├── guidelines/
  │   ├── ai-usage-policy.md
  │   └── prompt-writing-guide.md
  └── templates/
      ├── feature-request.md
      └── bug-report.md
```

**問題点**: プロンプトの品質がチーム生産性に直結する。個人の暗黙知をチームの形式知に変換し、継続的に改善する仕組みが必要。

---

## 7. FAQ

### Q1: AIツールの選定基準は？

**A**: チームのAIツール選定では以下の観点で評価する。

| 観点 | 重要度 | 例 |
|------|--------|-----|
| セキュリティ | 最高 | コードの外部送信ポリシー |
| 精度 | 高 | 使用言語・フレームワークでの性能 |
| 統合性 | 高 | IDE/CI/CDとの連携 |
| コスト | 中 | 人数×単価のROI |
| 学習コスト | 中 | チーム全員が使えるまでの時間 |

### Q2: AI活用のガバナンスルールはどう設計するか？

**A**: 最低限以下のルールを策定する。(1) 機密コード（認証、暗号化、個人情報処理）へのAI使用制限、(2) AI生成コードの必須レビュー基準、(3) 外部APIへのコード送信に関するセキュリティポリシー、(4) AI出力の著作権・ライセンス取り扱い。これらをチームのContributing Guideに明記する。

### Q3: AI導入に抵抗するメンバーへの対応は？

**A**: (1) 強制せず成功体験を見せる（AIで時間短縮できた実例共有）、(2) AIは代替ではなく拡張であることを強調、(3) 小さなタスク（テスト生成、ドキュメント作成）から始めるよう提案、(4) ペアプロでChampionと組ませる。強制的な導入は逆効果になるため、自然な動機付けを重視する。

### Q4: リモートチームでのAI活用のコツは？

**A**: (1) 非同期コミュニケーションでのAIレビュー活用（タイムゾーン差をAIが埋める）、(2) 共有プロンプトリポジトリの整備、(3) AIセッションの録画共有（Screen Recording + AI操作の実演）、(4) SlackボットによるAI支援の民主化。

---

## 8. まとめ

| カテゴリ | ポイント |
|---------|---------|
| ペアプロ | 人間が舵取り、AIは提案者。コンテキスト最大化が鍵 |
| コードレビュー | AI事前レビュー+人間最終判断の二段階フロー |
| 生産性計測 | DORA指標+AI固有メトリクスで効果を可視化 |
| スキル格差 | スキルマトリクス+ペアローテーションで底上げ |
| ガバナンス | セキュリティ・品質・ライセンスのルール明文化 |
| 文化醸成 | 強制より成功体験の共有で自然な導入を促進 |
| プロンプト共有 | リポジトリ管理で属人化を防止 |

---

## 次に読むべきガイド

- [01-ai-onboarding.md](./01-ai-onboarding.md) — AI時代の開発者オンボーディング
- AIコーディングアシスタント徹底比較 — ツール選定の詳細
- プロンプトエンジニアリング for 開発者 — 効果的なプロンプト設計

---

## 参考文献

1. DORA "Accelerate State of DevOps Report" — https://dora.dev/research/
2. GitHub "The Impact of AI on Developer Productivity" — https://github.blog/news-insights/research/
3. Anthropic Claude Documentation — https://docs.anthropic.com/
4. ThoughtWorks Technology Radar — https://www.thoughtworks.com/radar
5. Martin Fowler, "Continuous Integration" — https://martinfowler.com/articles/continuousIntegration.html
