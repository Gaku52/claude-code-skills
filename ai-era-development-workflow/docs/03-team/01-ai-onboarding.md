# AI時代の開発者オンボーディング

> AIツール研修、プロンプト共有、ナレッジベース構築など、新メンバーがAI時代のチーム開発に迅速に適応するための体系的なオンボーディングプログラムを解説する。

---

## この章で学ぶこと

1. **AI時代に対応したオンボーディングプログラム**を設計し、新メンバーの立ち上がり時間を短縮できる
2. **AIツールの研修カリキュラムとプロンプト共有基盤**を構築し、チーム全体のAI活用水準を均一化できる
3. **ナレッジベースの構築・運用手法**を習得し、暗黙知を形式知化して組織の資産にできる

---

## 1. AI時代のオンボーディングの全体像

### 1.1 従来型 vs AI時代のオンボーディング

```
┌──────────────────────────────────────────────────────┐
│         オンボーディングプロセスの進化                   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  従来型 (4-8週間)              AI時代 (2-4週間)       │
│  ─────────────                ──────────────         │
│                                                      │
│  Week 1: 環境構築             Day 1-2: AI支援環境構築 │
│  ・手動でセットアップ          ・AIが手順をガイド      │
│  ・ドキュメント読み込み        ・AIでコード探索        │
│                                                      │
│  Week 2: コード理解           Day 3-5: AI支援コード理解│
│  ・ソースを1つずつ読む        ・AIにアーキテクチャ質問 │
│  ・先輩に都度質問             ・AIで依存関係を分析    │
│                                                      │
│  Week 3-4: 小タスク着手       Week 2: AI活用実装      │
│  ・ペアプロで学習             ・AIペアプロで小タスク   │
│  ・レビューで指摘受ける       ・AIプレレビューで品質確保│
│                                                      │
│  Week 5-8: 独り立ち           Week 3-4: 自律的開発    │
│  ・徐々に難易度UP             ・AIツール使いこなし    │
│  ・暗黙知の習得に時間          ・ナレッジベース活用    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 1.2 オンボーディングの4フェーズ

```
Phase 1          Phase 2          Phase 3          Phase 4
環境構築          コード理解        AI実践           自律
(Day 1-2)        (Day 3-5)        (Week 2)         (Week 3-4)
─────────        ─────────        ─────────        ─────────
・開発環境        ・アーキテクチャ  ・AIペアプロ      ・独立タスク
・AIツール導入    ・AIで探索        ・プロンプト活用  ・メンター不要
・ルール確認      ・ドメイン理解    ・レビュー参加    ・ナレッジ貢献

    目標: 動く環境  目標: 全体把握   目標: AI活用     目標: 独り立ち
```

---

## 2. Phase 1: AI開発環境のセットアップ

### 2.1 自動化されたセットアップスクリプト

```bash
#!/bin/bash
# scripts/onboarding-setup.sh
# 新メンバーの開発環境を自動構築するスクリプト

set -euo pipefail

echo "=== AI時代の開発環境セットアップ ==="
echo ""

# 1. 基本開発ツール
echo "[1/5] 基本開発ツールのインストール..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    command -v brew >/dev/null || /bin/bash -c \
        "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install git node python@3.12 docker
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update && sudo apt install -y git nodejs python3 docker.io
fi

# 2. AIコーディングツール
echo "[2/5] AIコーディングツールの設定..."

# GitHub Copilot (VS Code拡張)
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat

# Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Cursor IDE (オプション)
if [[ "$INSTALL_CURSOR" == "true" ]]; then
    echo "Cursor IDEをインストール..."
    # プラットフォーム別インストール
fi

# 3. プロジェクト固有の設定
echo "[3/5] プロジェクト設定の適用..."
git clone "$PROJECT_REPO" ~/workspace/project
cd ~/workspace/project

# AIツール設定ファイルのコピー
cp .ai/configs/recommended-settings.json ~/.config/ai-tools/

# 4. AIプロンプトライブラリの設定
echo "[4/5] プロンプトライブラリの設定..."
mkdir -p ~/.ai/prompts
ln -sf ~/workspace/project/.ai/prompts ~/.ai/prompts/project

# 5. 検証
echo "[5/5] 環境を検証中..."
python3 -c "print('Python OK')"
node -e "console.log('Node OK')"
git --version
echo ""
echo "=== セットアップ完了 ==="
echo "次のステップ: docs/onboarding/phase1-checklist.md を確認してください"
```

### 2.2 AIツール設定のチーム標準

```json
// .ai/configs/recommended-settings.json
{
  "copilot": {
    "enable": true,
    "inlineSuggest.enable": true,
    "advanced": {
      "length": 500,
      "temperature": 0.1,
      "top_p": 0.95
    }
  },
  "claude_code": {
    "model": "claude-sonnet-4-20250514",
    "allowed_tools": ["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
    "project_context": ".claude/CLAUDE.md"
  },
  "team_rules": {
    "ai_review_required": true,
    "max_ai_generated_without_test": 50,
    "sensitive_dirs_no_ai": ["src/auth/", "src/crypto/", "config/secrets/"],
    "prompt_template_required": true
  }
}
```

---

## 3. Phase 2: AIを使ったコードベース理解

### 3.1 AIによるコードベース探索セッション

```python
# onboarding/codebase_explorer.py
# 新メンバー向けAIコードベース探索ツール

from pathlib import Path
import json

class CodebaseExplorer:
    """新メンバーがAIを使ってコードベースを探索するためのガイド"""

    EXPLORATION_TASKS = [
        {
            "title": "アーキテクチャ概要の把握",
            "prompt": """
このプロジェクトのディレクトリ構成を分析し、
以下を説明してください：
1. 全体のアーキテクチャパターン（MVC, Clean Architecture等）
2. 主要なモジュールとその役割
3. データの流れ（リクエストからレスポンスまで）
4. 外部依存関係
""",
            "target": "プロジェクトルート",
            "estimated_time": "30分",
        },
        {
            "title": "ドメインモデルの理解",
            "prompt": """
このプロジェクトの主要なドメインモデル（エンティティ）を
一覧化し、それぞれの関係を説明してください：
1. 各モデルの属性と責務
2. モデル間のリレーション（1:N, N:M等）
3. ビジネスルールの実装場所
""",
            "target": "src/models/ or src/domain/",
            "estimated_time": "45分",
        },
        {
            "title": "API仕様の把握",
            "prompt": """
このプロジェクトのAPI一覧を分析し、
以下を整理してください：
1. エンドポイント一覧（メソッド, パス, 概要）
2. 認証・認可の仕組み
3. エラーハンドリングのパターン
4. レスポンス形式の規約
""",
            "target": "src/routes/ or src/controllers/",
            "estimated_time": "30分",
        },
        {
            "title": "テスト戦略の理解",
            "prompt": """
このプロジェクトのテスト戦略を分析してください：
1. テストの種類（ユニット/統合/E2E）と配置
2. テストフレームワークとその設定
3. モック/スタブの使い方
4. テストカバレッジの状況
""",
            "target": "tests/ or __tests__/",
            "estimated_time": "30分",
        },
    ]

    @classmethod
    def generate_exploration_plan(cls, project_path: str) -> str:
        """探索プランを生成"""
        plan = "# AIコードベース探索プラン\n\n"
        plan += f"プロジェクト: {project_path}\n\n"

        for i, task in enumerate(cls.EXPLORATION_TASKS, 1):
            plan += f"## タスク {i}: {task['title']}\n\n"
            plan += f"**対象**: `{task['target']}`\n"
            plan += f"**想定時間**: {task['estimated_time']}\n\n"
            plan += f"**AIへのプロンプト**:\n```\n{task['prompt'].strip()}\n```\n\n"
            plan += "**理解度チェック**:\n"
            plan += "- [ ] 自分の言葉で説明できる\n"
            plan += "- [ ] 関連コードの場所を指摘できる\n"
            plan += "- [ ] 改善点を1つ以上挙げられる\n\n"

        return plan
```

### 3.2 理解度確認テンプレート

```markdown
<!-- .ai/onboarding/understanding-check.md -->
# コードベース理解度チェックシート

## 回答者: ___
## 日付: ___

### アーキテクチャ
1. このプロジェクトのアーキテクチャパターンは何ですか？
   - 回答:
   - AIに確認した結果:

2. リクエストがDBに到達するまでの経路を説明してください。
   - 回答:
   - 通過するファイル/クラス:

### ドメイン知識
3. 主要なエンティティを3つ挙げ、関係を説明してください。
   - 回答:

4. 最も複雑なビジネスルールはどこに実装されていますか？
   - 回答:
   - ファイルパス:

### 開発フロー
5. 新機能を追加する場合の手順を説明してください。
   - 回答:

6. AIツールをどのように活用しますか？
   - 回答:

### メンターコメント
- 理解度: ☆☆☆☆☆ (5段階)
- 追加学習が必要な領域:
- 次のステップ:
```

---

## 4. Phase 3: AIツール研修カリキュラム

### 4.1 研修カリキュラム全体像

| 日 | テーマ | 内容 | 演習 |
|---|-------|------|------|
| Day 1 | AI基礎 | LLMの仕組み、できること/できないこと | 簡単なプロンプト演習 |
| Day 2 | プロンプト設計 | 効果的なプロンプトの書き方 | チームテンプレートの活用 |
| Day 3 | コード生成 | AIによるコード生成と検証 | 実タスクでのAI活用 |
| Day 4 | レビュー・テスト | AIレビューとテスト生成 | PR作成とAIレビュー体験 |
| Day 5 | 応用・統合 | ワークフロー統合、ナレッジベース | 自分なりの活用法を発表 |

### 4.2 プロンプトエンジニアリング研修

```python
# onboarding/prompt_training.py
# プロンプト設計の研修モジュール

class PromptTraining:
    """プロンプトエンジニアリングの段階的研修"""

    LEVELS = {
        "level_1_basic": {
            "title": "基本: 明確な指示",
            "principle": "具体的に、明確に、1つずつ依頼する",
            "bad_example": "このコードを直して",
            "good_example": (
                "以下のPython関数にある無限ループのバグを修正してください。\n"
                "修正後のコードと、何を変更したかの説明を含めてください。\n\n"
                "```python\ndef process_items(items):\n"
                "    i = 0\n    while i < len(items):\n"
                "        if items[i].is_valid():\n"
                "            handle(items[i])\n"
                "        # i += 1 が抜けている\n```"
            ),
            "exercise": "チームのバグレポートからAIに修正を依頼するプロンプトを書く",
        },
        "level_2_context": {
            "title": "中級: コンテキストの提供",
            "principle": "背景、制約、期待する形式を明示する",
            "bad_example": "テストを書いて",
            "good_example": (
                "以下の仕様に基づいてユニットテストを書いてください。\n\n"
                "## テスト対象\n"
                "UserService.create_user(name, email) メソッド\n\n"
                "## 仕様\n"
                "- nameは1-50文字の文字列\n"
                "- emailは有効なメールアドレス形式\n"
                "- 重複emailはエラー\n\n"
                "## テストフレームワーク\n"
                "pytest + pytest-mock\n\n"
                "## テストパターン\n"
                "正常系2つ、異常系3つ以上を含めてください"
            ),
            "exercise": "既存のテストファイルを参考に、新機能のテストプロンプトを作成",
        },
        "level_3_advanced": {
            "title": "上級: 思考の誘導",
            "principle": "段階的な推論、制約の明示、自己検証を組み込む",
            "bad_example": "パフォーマンスを改善して",
            "good_example": (
                "以下のAPIエンドポイントのレスポンスタイムが遅い問題を分析してください。\n\n"
                "## 手順\n"
                "1. まず、このコードのボトルネックになりうる箇所を3つ挙げてください\n"
                "2. 各ボトルネックの影響度を高/中/低で評価してください\n"
                "3. 最も影響度の高い箇所から、具体的な改善案を提示してください\n"
                "4. 各改善案のトレードオフ（メモリ使用量、コード複雑度等）も説明してください\n\n"
                "## 制約\n"
                "- 外部ライブラリの追加は不可\n"
                "- APIの互換性を維持すること\n"
                "- テストが通ること"
            ),
            "exercise": "本番障害シナリオでAIにデバッグを依頼するプロンプトを設計",
        },
    }

    @classmethod
    def get_curriculum(cls) -> str:
        """カリキュラムを出力"""
        output = "# プロンプトエンジニアリング研修\n\n"
        for level_id, level in cls.LEVELS.items():
            output += f"## {level['title']}\n\n"
            output += f"**原則**: {level['principle']}\n\n"
            output += f"**悪い例**:\n```\n{level['bad_example']}\n```\n\n"
            output += f"**良い例**:\n```\n{level['good_example']}\n```\n\n"
            output += f"**演習**: {level['exercise']}\n\n---\n\n"
        return output
```

---

## 5. ナレッジベースの構築

### 5.1 ナレッジベースのアーキテクチャ

```
┌──────────────────────────────────────────────────┐
│            チームナレッジベース                      │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐   ┌──────────────────┐        │
│  │  暗黙知       │   │  形式知           │        │
│  │  (個人の頭の中) │──>│  (ドキュメント化)  │        │
│  └──────────────┘   └──────────────────┘        │
│         │                     │                  │
│         v                     v                  │
│  ┌──────────────┐   ┌──────────────────┐        │
│  │  AIが抽出     │   │  検索可能なDB     │        │
│  │  ・質疑応答    │   │  ・ベクトル検索    │        │
│  │  ・パターン化  │   │  ・キーワード検索  │        │
│  └──────────────┘   └──────────────────┘        │
│         │                     │                  │
│         v                     v                  │
│  ┌────────────────────────────────────┐         │
│  │  統合ナレッジインターフェース        │         │
│  │  ・Slack Bot + AI                  │         │
│  │  ・IDE統合 (コード上でQ&A)          │         │
│  │  ・自動ドキュメント更新             │         │
│  └────────────────────────────────────┘         │
└──────────────────────────────────────────────────┘
```

### 5.2 ナレッジ登録・検索システム

```python
# knowledge/knowledge_base.py
# チームナレッジベースの管理システム

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class KnowledgeEntry:
    """ナレッジエントリ"""
    id: str
    title: str
    category: str          # architecture, debugging, workflow, domain, tool
    content: str
    tags: list[str]
    author: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""
    related_files: list[str] = field(default_factory=list)
    ai_generated: bool = False
    verified: bool = False

class TeamKnowledgeBase:
    """チームナレッジベースの管理"""

    CATEGORIES = [
        "architecture",   # アーキテクチャの意思決定
        "debugging",      # デバッグのノウハウ
        "workflow",       # 開発ワークフロー
        "domain",         # ドメイン知識
        "tool",           # ツール活用法
        "ai-prompt",      # 効果的なプロンプト
        "postmortem",     # 障害振り返り
    ]

    def __init__(self, base_dir: str = ".knowledge"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def add_entry(self, entry: KnowledgeEntry) -> str:
        """ナレッジを追加"""
        category_dir = self.base_dir / entry.category
        category_dir.mkdir(exist_ok=True)

        path = category_dir / f"{entry.id}.json"
        with open(path, "w") as f:
            json.dump(asdict(entry), f, indent=2, ensure_ascii=False)

        return str(path)

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[KnowledgeEntry]:
        """ナレッジを検索"""
        results = []
        search_dirs = (
            [self.base_dir / category] if category
            else [self.base_dir / c for c in self.CATEGORIES]
        )

        query_lower = query.lower()
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for path in search_dir.glob("*.json"):
                with open(path) as f:
                    data = json.load(f)

                # テキストマッチ
                text = f"{data['title']} {data['content']} {' '.join(data['tags'])}".lower()
                if query_lower not in text:
                    continue

                # タグフィルタ
                if tags and not set(tags).intersection(set(data["tags"])):
                    continue

                results.append(KnowledgeEntry(**data))

        return sorted(results, key=lambda e: e.created_at, reverse=True)

    def generate_onboarding_digest(self) -> str:
        """新メンバー向けのナレッジダイジェストを生成"""
        digest = "# 新メンバー向けナレッジダイジェスト\n\n"
        digest += f"生成日: {datetime.now().strftime('%Y-%m-%d')}\n\n"

        for category in self.CATEGORIES:
            entries = self.search("", category=category)
            if not entries:
                continue

            category_names = {
                "architecture": "アーキテクチャ",
                "debugging": "デバッグ",
                "workflow": "ワークフロー",
                "domain": "ドメイン知識",
                "tool": "ツール",
                "ai-prompt": "AIプロンプト",
                "postmortem": "障害振り返り",
            }
            digest += f"## {category_names.get(category, category)}\n\n"

            for entry in entries[:5]:  # 各カテゴリ最新5件
                digest += f"### {entry.title}\n"
                digest += f"- 著者: {entry.author}\n"
                digest += f"- タグ: {', '.join(entry.tags)}\n"
                digest += f"- 概要: {entry.content[:200]}...\n\n"

        return digest
```

---

## 6. メンター制度との統合

### 6.1 AI時代のメンター役割

| メンターの役割 | 従来 | AI時代 |
|--------------|------|--------|
| 技術質問への回答 | メンターが直接回答 | AIに聞く方法を教える |
| コードレビュー | 全てメンターが実施 | AI事前レビュー + メンター最終確認 |
| 設計相談 | メンターの経験に依存 | AIに設計案を出させてメンターと議論 |
| ドメイン知識の伝達 | 口頭説明 | ナレッジベース + AI質問 + メンター補足 |
| 暗黙知の共有 | ペアプロ時に断片的に | AI記録 + ナレッジベース登録 |

### 6.2 週次1on1チェックポイント

```yaml
# .ai/onboarding/weekly-checklist.yaml
week_1:
  title: "環境構築とコード理解"
  checkpoints:
    - "開発環境が完全に動作している"
    - "AIツール（Copilot/Claude）を設定済み"
    - "プロジェクトのアーキテクチャを説明できる"
    - "AIを使ってコードベースを探索した"
  mentor_discussion:
    - "AIツールの第一印象は？困っていることは？"
    - "コードベースで最も理解が難しかった部分は？"

week_2:
  title: "AI活用実践"
  checkpoints:
    - "AIペアプロで最初のPRを作成した"
    - "チームのプロンプトテンプレートを使った"
    - "AIレビューを受けてフィードバックに対応した"
    - "ナレッジベースに1件以上登録した"
  mentor_discussion:
    - "AIの提案を受け入れた/拒否した判断基準は？"
    - "プロンプトの書き方で工夫した点は？"

week_3:
  title: "独立開発への移行"
  checkpoints:
    - "メンターなしで1タスク完了した"
    - "AIツールの自分なりの活用パターンを確立した"
    - "他メンバーのPRをAI支援でレビューした"
  mentor_discussion:
    - "自信を持って取り組めるタスクの範囲は？"
    - "まだ不安な領域は？"

week_4:
  title: "自律と貢献"
  checkpoints:
    - "通常のスプリントタスクを独立で完了"
    - "ナレッジベースに3件以上貢献"
    - "オンボーディング改善提案を1つ以上提出"
  mentor_discussion:
    - "オンボーディングプロセス全体の振り返り"
    - "他の新メンバーへのアドバイス"
```

---

## 7. アンチパターン

### 7.1 アンチパターン：AIに依存しすぎるオンボーディング

```
NG: 全てをAIに聞けばいいと教える
  - 「わからないことはAIに聞いて」で放置
  - メンターとの対話がゼロ
  - ドメイン固有の暗黙知が伝わらない
  - AIの誤った回答を鵜呑みにするリスク

OK: AIは補助、人間が主導
  1. まずAIに聞いてみる（効率化）
  2. AIの回答を自分で検証する（批判的思考）
  3. 不明点はメンターに確認（信頼できる情報源）
  4. 学んだことをナレッジベースに記録（組織貢献）
```

### 7.2 アンチパターン：画一的なAI研修

```
NG: 全員同じペースでAI研修を実施
  - 経験5年のシニアと新卒に同じカリキュラム
  - プログラミング言語の経験差を無視
  - 学習スタイルの違いを考慮しない

OK: スキルレベル別の適応型カリキュラム
  ・新卒/ジュニア: AI基礎 → プロンプト基本 → 実践（3日）
  ・ミドル: プロンプト応用 → ワークフロー統合（1日）
  ・シニア: チーム戦略 → ガバナンス設計（半日）
  ・AIネイティブ: ツール評価 → チーム布教（半日）
```

---

## 8. FAQ

### Q1: オンボーディング期間はどの程度短縮できるか？

**A**: AI導入前と比較して平均40-60%の短縮が報告されている。特にコードベース理解フェーズの短縮が顕著で、AIにアーキテクチャの説明を求めることで、従来1-2週間かかっていた全体把握が2-3日に短縮できる。ただし、ドメイン知識やチーム文化の理解にはAIでは代替できない時間が必要。

### Q2: AIツールの研修にどれくらいの時間を割くべきか？

**A**: 初週に集中的に1-2日、その後は実タスクを通じたOJTが効果的。座学だけでは定着しないため、実際のタスクでAIを使う実践を重視する。理想的には、最初の1ヶ月間はすべてのタスクでAI活用を意識させ、月末に振り返りセッションを設ける。

### Q3: ナレッジベースの継続的な更新はどう維持するか？

**A**: (1) PR作成時にナレッジ登録チェックを設ける、(2) スプリント振り返りでナレッジ登録を必須タスクにする、(3) AIが自動でPRからナレッジ候補を抽出する仕組みを導入、(4) ナレッジ貢献を評価指標に含める。最も効果的なのは、ナレッジ登録が自然なワークフローの一部になるような仕組みづくり。

---

## 9. まとめ

| カテゴリ | ポイント |
|---------|---------|
| 全体設計 | 4フェーズ(環境構築→理解→実践→自律)で段階的に |
| 環境構築 | 自動化スクリプト+AIツール設定のチーム標準化 |
| コード理解 | AIで探索→理解度チェックシート→メンター確認 |
| AI研修 | レベル別カリキュラム+実タスクOJT |
| ナレッジ | 暗黙知→形式知の継続的変換+検索可能なDB |
| メンター | AIは補助、人間が主導。週次1on1で進捗確認 |
| 期間短縮 | 従来比40-60%短縮が目安。ドメイン知識は例外 |

---

## 次に読むべきガイド

- [00-ai-team-practices.md](./00-ai-team-practices.md) — AI時代のチーム開発プラクティス
- プロンプトエンジニアリング — 効果的なプロンプト設計の詳細
- AI開発ガバナンス — セキュリティとコンプライアンス

---

## 参考文献

1. Google re:Work, "Guide: Set up a new hire onboarding program" — https://rework.withgoogle.com/guides/hiring/steps/set-up-onboarding/
2. Stripe, "Developer Coefficient Report" — https://stripe.com/reports/developer-coefficient
3. Anthropic, "Prompt Engineering Guide" — https://docs.anthropic.com/claude/docs/prompt-engineering
4. GitHub, "Onboarding developers with GitHub Copilot" — https://github.blog/
5. ThoughtWorks, "Technology Radar: AI-assisted development" — https://www.thoughtworks.com/radar
