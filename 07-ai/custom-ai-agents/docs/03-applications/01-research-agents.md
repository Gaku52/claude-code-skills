# リサーチエージェント

> 情報収集・分析・要約――複数の情報源から自律的にデータを収集し、構造化されたリサーチレポートを生成するエージェントの設計と実装。

## この章で学ぶこと

1. リサーチエージェントの情報収集パイプラインと多段階フィルタリングの設計
2. Web検索・文書解析・データ統合の実装パターン
3. 信頼性の高いリサーチ出力のための検証と引用管理の仕組み
4. 専門分野別のリサーチエージェント実装
5. 大規模情報の要約・統合テクニック

---

## 1. リサーチエージェントの全体像

```
リサーチエージェントのパイプライン

[目標設定]
    |
    v
[クエリ生成] -- 目標を複数の検索クエリに分解
    |
    v
[情報収集] -- Web検索、DB検索、文書読み取り
    |            +-- 検索エンジン
    |            +-- 学術論文DB (Semantic Scholar等)
    |            +-- 社内ドキュメント
    |            +-- API / データベース
    v
[フィルタリング] -- 関連性・信頼性の評価
    |
    v
[分析・統合] -- 情報の構造化、矛盾の解決
    |
    v
[レポート生成] -- 構造化された出力、引用付き
    |
    v
[品質チェック] -- 事実確認、バイアスチェック
```

### 1.1 リサーチの深度レベル

```
リサーチの深度スペクトラム

Level 1: クイック調査（1-2分）
  - 検索1-2回
  - 上位結果のスニペットのみ
  - 用途: 事実確認、定義確認

Level 2: 標準調査（5-10分）
  - 検索3-5回
  - 2-3ページの詳細読み込み
  - クロスチェック1回
  - 用途: 一般的な調査、ブリーフィング

Level 3: 深掘り調査（15-30分）
  - 検索5-10回
  - 5+ページの詳細読み込み
  - 学術論文の調査
  - 複数回のクロスチェック
  - 用途: 意思決定支援、詳細レポート

Level 4: 包括的調査（1-2時間）
  - 多段階のリサーチサイクル
  - 10+ページの詳細読み込み
  - 学術論文の精読
  - データの定量分析
  - 専門家意見の調査
  - 用途: 戦略レポート、市場調査
```

---

## 2. 基本的なリサーチエージェント

### 2.1 完全な実装

```python
# リサーチエージェントの実装
import anthropic
import json
from dataclasses import dataclass, field

@dataclass
class Source:
    title: str
    url: str
    snippet: str
    reliability: float = 0.5  # 0-1

@dataclass
class ResearchResult:
    topic: str
    summary: str
    key_findings: list[str]
    sources: list[Source]
    confidence: float  # 0-1

class ResearchAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.tools = [
            {
                "name": "web_search",
                "description": "Web検索を実行して結果を返す",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {
                            "type": "integer", "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "read_webpage",
                "description": "指定URLのWebページ内容を取得する",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "search_papers",
                "description": "学術論文を検索する（Semantic Scholar）",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "year_from": {"type": "integer"}
                    },
                    "required": ["query"]
                }
            }
        ]

    def research(self, topic: str, depth: str = "standard") -> str:
        """トピックについてリサーチを実行"""
        system_prompt = f"""あなたは優秀なリサーチアナリストです。
以下のルールに従ってリサーチを実施してください:

1. まずトピックを分解し、3-5個の検索クエリを生成
2. 各クエリで検索を実行し、関連性の高い結果を収集
3. 重要なページは詳細に読み込む
4. 複数情報源を照合し、矛盾がないか確認
5. 構造化されたレポートにまとめる

リサーチ深度: {depth}
- light: 概要レベル（検索1-2回）
- standard: 標準（検索3-5回、2-3ページ詳細読み込み）
- deep: 深掘り（検索5-10回、5+ページ詳細読み込み）

出力形式: Markdown形式のレポート（引用付き）
"""
        messages = [
            {"role": "user", "content": f"リサーチトピック: {topic}"}
        ]

        for _ in range(20):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                tools=self.tools,
                messages=messages
            )

            if response.stop_reason == "end_turn":
                return response.content[0].text

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(
                        block.name, block.input
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({
                "role": "assistant",
                "content": response.content
            })
            messages.append({"role": "user", "content": tool_results})

        return "リサーチが完了できませんでした"

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "web_search":
            return self._web_search(
                args["query"], args.get("num_results", 10)
            )
        elif name == "read_webpage":
            return self._read_webpage(args["url"])
        elif name == "search_papers":
            return self._search_papers(
                args["query"], args.get("year_from")
            )
        return "不明なツール"

    def _web_search(self, query: str, num_results: int) -> str:
        # 実際にはSerpAPI、Google Custom Search等を使用
        pass

    def _read_webpage(self, url: str) -> str:
        import requests
        from bs4 import BeautifulSoup
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return text[:5000]
        except Exception as e:
            return f"ページ読み込みエラー: {e}"

    def _search_papers(self, query: str,
                       year_from: int = None) -> str:
        import requests
        params = {"query": query, "limit": 5}
        if year_from:
            params["year"] = f"{year_from}-"
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params
        )
        return json.dumps(
            resp.json().get("data", [])[:5], ensure_ascii=False
        )
```

### 2.2 多段階リサーチ

```python
# 深いリサーチのための多段階パイプライン
class DeepResearchAgent:
    def research(self, topic: str) -> str:
        # Phase 1: 広く浅く調査
        overview = self._broad_search(topic)

        # Phase 2: 重要テーマを特定
        key_themes = self._identify_themes(overview)

        # Phase 3: 各テーマを深掘り
        detailed_findings = {}
        for theme in key_themes:
            detailed_findings[theme] = self._deep_dive(theme)

        # Phase 4: 統合レポート生成
        report = self._synthesize(topic, detailed_findings)

        # Phase 5: 事実確認
        verified_report = self._fact_check(report)

        return verified_report
```

### 2.3 構造化出力付きリサーチ

```python
from pydantic import BaseModel, Field

class ResearchFinding(BaseModel):
    """個別の調査結果"""
    claim: str = Field(description="主張・事実")
    evidence: str = Field(description="根拠")
    source_url: str = Field(description="情報源URL")
    confidence: float = Field(
        description="信頼度 0.0-1.0", ge=0, le=1
    )

class ResearchReport(BaseModel):
    """構造化されたリサーチレポート"""
    title: str
    executive_summary: str = Field(description="要約（300字以内）")
    key_findings: list[ResearchFinding]
    analysis: str = Field(description="分析と考察")
    limitations: list[str] = Field(description="調査の限界")
    recommendations: list[str] = Field(description="推奨事項")

class StructuredResearchAgent:
    """構造化出力を返すリサーチエージェント"""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.base_agent = ResearchAgent()

    async def research_structured(
        self,
        topic: str,
        depth: str = "standard"
    ) -> ResearchReport:
        """構造化されたリサーチレポートを生成"""
        # Phase 1: 通常のリサーチ実行
        raw_report = self.base_agent.research(topic, depth)

        # Phase 2: 構造化されたJSONに変換
        structured = await self._structurize(raw_report)

        return structured

    async def _structurize(self, raw_report: str) -> ResearchReport:
        """生のレポートを構造化形式に変換"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"""以下のリサーチレポートを構造化された
JSON形式に変換してください。

レポート:
{raw_report}

以下のJSON形式で出力してください:
{{
  "title": "...",
  "executive_summary": "...",
  "key_findings": [
    {{
      "claim": "...",
      "evidence": "...",
      "source_url": "...",
      "confidence": 0.0-1.0
    }}
  ],
  "analysis": "...",
  "limitations": ["..."],
  "recommendations": ["..."]
}}"""
            }]
        )

        data = json.loads(response.content[0].text)
        return ResearchReport(**data)
```

---

## 3. 情報の信頼性評価

### 3.1 信頼性ピラミッド

```
情報源の信頼性ピラミッド

         /\
        /  \     一次資料（論文、公式データ）
       /    \    信頼度: 最高
      /------\
     /        \   二次資料（ニュース記事、専門メディア）
    /          \  信頼度: 高
   /------------\
  /              \ 三次資料（ブログ、SNS、Wikipedia）
 /                \ 信頼度: 中-低
/------------------\
```

### 3.2 信頼性スコアリングの実装

```python
# 情報源の信頼性スコアリング
class SourceReliabilityScorer:
    DOMAIN_SCORES = {
        "arxiv.org": 0.9,
        "nature.com": 0.95,
        "science.org": 0.95,
        "acm.org": 0.9,
        "ieee.org": 0.9,
        "github.com": 0.7,
        "stackoverflow.com": 0.7,
        "docs.anthropic.com": 0.85,
        "openai.com": 0.85,
        "wikipedia.org": 0.6,
        "medium.com": 0.4,
        "reddit.com": 0.3,
        "twitter.com": 0.2,
    }

    # ドメインカテゴリ別デフォルトスコア
    CATEGORY_SCORES = {
        ".gov": 0.85,     # 政府機関
        ".edu": 0.8,      # 教育機関
        ".org": 0.6,      # 非営利団体
        ".ac.jp": 0.8,    # 日本の大学
        ".go.jp": 0.85,   # 日本の政府機関
    }

    def score(self, url: str, content: str) -> float:
        """情報源の信頼性をスコアリング"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        # ドメインベースのスコア
        domain_score = self._get_domain_score(domain)

        # 引用の有無
        has_citations = any(
            marker in content
            for marker in ["[1]", "参考文献", "References", "doi:"]
        )
        citation_bonus = 0.1 if has_citations else 0

        # 日付の新しさ
        recency_bonus = self._check_recency(content)

        # コンテンツの質的指標
        quality_bonus = self._assess_content_quality(content)

        return min(
            1.0,
            domain_score + citation_bonus + recency_bonus + quality_bonus
        )

    def _get_domain_score(self, domain: str) -> float:
        """ドメインからスコアを取得"""
        # 完全一致
        if domain in self.DOMAIN_SCORES:
            return self.DOMAIN_SCORES[domain]

        # サブドメインチェック
        for known_domain, score in self.DOMAIN_SCORES.items():
            if domain.endswith(f".{known_domain}"):
                return score

        # カテゴリチェック
        for suffix, score in self.CATEGORY_SCORES.items():
            if domain.endswith(suffix):
                return score

        return 0.5  # デフォルト

    def _check_recency(self, content: str) -> float:
        """コンテンツの新しさを評価"""
        import re
        from datetime import datetime

        # 年号のパターンを検出
        years = re.findall(r'20[12]\d', content)
        if years:
            latest_year = max(int(y) for y in years)
            current_year = datetime.now().year
            diff = current_year - latest_year
            if diff == 0:
                return 0.1
            elif diff <= 1:
                return 0.05
            elif diff <= 3:
                return 0.0
            else:
                return -0.05  # 古い情報はペナルティ
        return 0.0

    def _assess_content_quality(self, content: str) -> float:
        """コンテンツの質的指標を評価"""
        bonus = 0.0

        # データや数値の含有率
        import re
        numbers = re.findall(r'\d+\.?\d*%|\$[\d,]+|\d{4}年', content)
        if len(numbers) > 5:
            bonus += 0.05

        # 構造化されたコンテンツ
        if content.count('\n') > 20:
            bonus += 0.02

        # 長さ（詳細な記事ほど信頼性が高い傾向）
        if len(content) > 3000:
            bonus += 0.03

        return min(bonus, 0.1)
```

### 3.3 クロスチェックの実装

```python
class CrossChecker:
    """複数ソース間のクロスチェックを実施"""

    def __init__(self, llm_client):
        self.client = llm_client

    async def cross_check(
        self,
        claim: str,
        sources: list[dict]
    ) -> dict:
        """主張を複数ソースでクロスチェック"""
        supporting = []
        contradicting = []
        neutral = []

        for source in sources:
            alignment = await self._check_alignment(
                claim, source["content"]
            )
            if alignment["supports"]:
                supporting.append(source)
            elif alignment["contradicts"]:
                contradicting.append(source)
            else:
                neutral.append(source)

        # 信頼度の計算
        total_weight = sum(
            s.get("reliability", 0.5)
            for s in supporting + contradicting
        )
        support_weight = sum(
            s.get("reliability", 0.5) for s in supporting
        )

        confidence = support_weight / total_weight if total_weight > 0 else 0.5

        return {
            "claim": claim,
            "confidence": confidence,
            "supporting_sources": len(supporting),
            "contradicting_sources": len(contradicting),
            "neutral_sources": len(neutral),
            "verdict": self._determine_verdict(confidence),
            "details": {
                "supporting": [s["url"] for s in supporting],
                "contradicting": [s["url"] for s in contradicting],
            }
        }

    def _determine_verdict(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "highly_supported"
        elif confidence >= 0.6:
            return "moderately_supported"
        elif confidence >= 0.4:
            return "inconclusive"
        else:
            return "likely_inaccurate"

    async def _check_alignment(
        self, claim: str, content: str
    ) -> dict:
        """コンテンツが主張を支持するか判定"""
        response = self.client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""以下の主張と文書の関係を判定してください。

主張: {claim}
文書: {content[:2000]}

JSON形式で回答:
{{"supports": true/false, "contradicts": true/false, "reason": "..."}}"""
            }]
        )
        return json.loads(response.content[0].text)
```

---

## 4. 専門分野別リサーチエージェント

### 4.1 市場調査エージェント

```python
class MarketResearchAgent(ResearchAgent):
    """市場調査に特化したリサーチエージェント"""

    def __init__(self):
        super().__init__()
        # 市場調査専用ツールを追加
        self.tools.extend([
            {
                "name": "search_industry_report",
                "description": "業界レポートを検索する",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "industry": {"type": "string"},
                        "aspect": {
                            "type": "string",
                            "enum": [
                                "market_size", "competitors",
                                "trends", "regulations"
                            ]
                        }
                    },
                    "required": ["industry"]
                }
            },
            {
                "name": "get_company_info",
                "description": "企業情報を取得する",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "company_name": {"type": "string"},
                        "info_type": {
                            "type": "string",
                            "enum": [
                                "overview", "financials",
                                "products", "news"
                            ]
                        }
                    },
                    "required": ["company_name"]
                }
            }
        ])

    def market_analysis(
        self,
        industry: str,
        region: str = "global"
    ) -> str:
        """包括的な市場分析を実施"""
        return self.research(
            topic=f"""
{industry} の市場分析（地域: {region}）

以下の観点で調査してください:
1. 市場規模と成長率（過去5年 + 将来予測）
2. 主要プレイヤーと市場シェア
3. トレンドと新技術
4. 規制環境
5. 参入障壁と機会
6. SWOT分析

数値データは必ず出典を明記してください。
""",
            depth="deep"
        )

    def competitive_analysis(
        self,
        target_company: str,
        competitors: list[str]
    ) -> str:
        """競合分析を実施"""
        competitor_list = ", ".join(competitors)
        return self.research(
            topic=f"""
{target_company} の競合分析

比較対象: {competitor_list}

以下の観点で比較分析してください:
1. 製品/サービスラインナップ
2. 価格戦略
3. ターゲット顧客
4. 技術的優位性
5. 市場ポジション
6. 最近の動向（直近1年）

比較表形式でまとめてください。
""",
            depth="deep"
        )
```

### 4.2 学術リサーチエージェント

```python
class AcademicResearchAgent(ResearchAgent):
    """学術論文の調査に特化したエージェント"""

    def literature_review(
        self,
        topic: str,
        year_range: tuple[int, int] = (2020, 2025)
    ) -> str:
        """文献レビューを実施"""
        return self.research(
            topic=f"""
学術文献レビュー: {topic}
期間: {year_range[0]}-{year_range[1]}

以下の構成でレビューを作成してください:

1. 概要
   - 研究分野の背景
   - 主要な研究課題

2. 方法論の分類
   - 主なアプローチの分類と比較
   - 各アプローチの利点と限界

3. 主要な研究成果
   - 代表的な論文の詳細レビュー
   - 実験結果の比較

4. 研究のギャップ
   - 未解決の問題
   - 今後の研究方向

5. 結論
   - 現状のまとめ
   - 推奨される研究方向

各引用は必ず [著者名, 年] 形式で記載してください。
Semantic Scholar での論文検索を活用してください。
""",
            depth="deep"
        )

    def find_related_work(
        self,
        paper_title: str,
        paper_abstract: str
    ) -> str:
        """関連研究を調査"""
        return self.research(
            topic=f"""
以下の論文の関連研究を調査してください。

タイトル: {paper_title}
アブストラクト: {paper_abstract}

調査項目:
1. 先行研究（この論文が引用すべきもの）
2. 後続研究（この論文を発展させたもの）
3. 競合研究（同様の課題に異なるアプローチで取り組んだもの）
4. 応用研究（この論文の手法を応用したもの）

各論文について以下を記載:
- タイトル、著者、年
- 関連の種類と理由
- 主要な貢献
""",
            depth="deep"
        )
```

### 4.3 技術調査エージェント

```python
class TechResearchAgent(ResearchAgent):
    """技術調査に特化したエージェント"""

    def technology_comparison(
        self,
        technologies: list[str],
        criteria: list[str]
    ) -> str:
        """技術比較調査"""
        tech_list = ", ".join(technologies)
        criteria_list = ", ".join(criteria)

        return self.research(
            topic=f"""
技術比較調査: {tech_list}

評価軸: {criteria_list}

以下の構成でレポートを作成してください:

1. 各技術の概要（200字程度ずつ）
2. 比較表（全評価軸 x 全技術）
3. ユースケース別の推奨
4. 移行コストと学習曲線
5. コミュニティの活発さとエコシステム
6. 将来性の評価
7. 総合評価と推奨

GitHubスター数、NPMダウンロード数などの
定量データも含めてください。
""",
            depth="deep"
        )

    def security_audit_research(
        self,
        technology: str,
        version: str
    ) -> str:
        """セキュリティ脆弱性の調査"""
        return self.research(
            topic=f"""
{technology} v{version} のセキュリティ脆弱性調査

調査項目:
1. 既知のCVE（過去2年間）
2. セキュリティアドバイザリ
3. 推奨されるセキュリティ設定
4. よくある誤設定
5. 依存関係の脆弱性
6. セキュリティベストプラクティス

NVD、GitHub Security Advisory等を
情報源として使用してください。
""",
            depth="deep"
        )
```

---

## 5. リサーチパターン比較

### 5.1 リサーチ戦略比較

| 戦略 | 深さ | 時間 | コスト | 適用場面 |
|------|------|------|--------|---------|
| 広浅検索 | 低 | 短 | 低 | 概要把握 |
| 深掘り検索 | 高 | 長 | 中 | 特定トピック調査 |
| 多段階リサーチ | 高 | 最長 | 高 | 包括的レポート |
| 比較調査 | 中 | 中 | 中 | 選択肢の比較 |
| トレンド分析 | 中 | 中-長 | 中 | 時系列変化の把握 |
| 文献レビュー | 最高 | 最長 | 最高 | 学術研究 |

### 5.2 出力形式比較

| 形式 | 用途 | 長さ | 含むもの |
|------|------|------|---------|
| サマリー | 迅速な情報共有 | 100-300字 | 要点3-5個 |
| ブリーフィング | 意思決定支援 | 500-1000字 | 要点+根拠+推奨 |
| レポート | 詳細分析 | 2000-5000字 | 全セクション+引用 |
| データシート | 定量比較 | 表形式 | 数値+比較表 |
| ホワイトペーパー | 深い洞察 | 5000-15000字 | 分析+図表+引用 |

### 5.3 情報源の比較

| 情報源 | 信頼性 | 最新性 | アクセス | 用途 |
|--------|--------|--------|---------|------|
| 学術論文 | 最高 | 中 | Semantic Scholar | 基礎研究 |
| 公式ドキュメント | 高 | 高 | 直接アクセス | 技術調査 |
| ニュースメディア | 中-高 | 最高 | Web検索 | トレンド |
| 業界レポート | 高 | 中 | 有料/Web | 市場調査 |
| ブログ/SNS | 低-中 | 高 | Web検索 | 実践知 |
| 政府統計 | 最高 | 低-中 | 公開データ | 定量分析 |

---

## 6. 引用管理

### 6.1 引用管理システム

```python
# 引用管理システム
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Citation:
    id: int
    title: str
    url: str
    author: str = ""
    date: str = ""
    accessed: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

class CitationManager:
    def __init__(self):
        self.citations: list[Citation] = []
        self._counter = 0

    def add(self, title: str, url: str, **kwargs) -> str:
        """引用を追加し、参照番号を返す"""
        self._counter += 1
        citation = Citation(
            id=self._counter, title=title, url=url, **kwargs
        )
        self.citations.append(citation)
        return f"[{self._counter}]"

    def format_references(self) -> str:
        """参考文献セクションを生成"""
        lines = ["## 参考文献\n"]
        for c in self.citations:
            line = f"[{c.id}] {c.title}"
            if c.author:
                line += f" - {c.author}"
            if c.date:
                line += f" ({c.date})"
            line += f"\n    {c.url}"
            lines.append(line)
        return "\n".join(lines)

    def get_citation(self, citation_id: int) -> Citation | None:
        """IDで引用を取得"""
        for c in self.citations:
            if c.id == citation_id:
                return c
        return None

    def verify_all_cited(self, report: str) -> list[int]:
        """レポート内で引用されていないソースを検出"""
        import re
        cited_ids = set(
            int(m) for m in re.findall(r'\[(\d+)\]', report)
        )
        all_ids = set(c.id for c in self.citations)
        uncited = all_ids - cited_ids
        return sorted(uncited)

    def export_bibtex(self) -> str:
        """BibTeX形式でエクスポート"""
        entries = []
        for c in self.citations:
            entry = f"""@misc{{ref{c.id},
  title = {{{c.title}}},
  author = {{{c.author or 'Unknown'}}},
  year = {{{c.date[:4] if c.date else 'n.d.'}}},
  url = {{{c.url}}},
  note = {{Accessed: {c.accessed}}}
}}"""
            entries.append(entry)
        return "\n\n".join(entries)
```

---

## 7. 並列リサーチ

### 7.1 アーキテクチャ

```
並列リサーチのアーキテクチャ

                    [トピック分解]
                    /     |     \
                   v      v      v
            [サブトピック1] [サブトピック2] [サブトピック3]
                   |      |      |
                   v      v      v
            [検索+読取]  [検索+読取]  [検索+読取]
                   \      |      /
                    v     v     v
                    [結果統合]
                       |
                       v
                    [レポート生成]
```

### 7.2 非同期並列リサーチの実装

```python
# 非同期並列リサーチ
import asyncio

class ParallelResearchAgent:
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = anthropic.AsyncAnthropic()

    async def research_parallel(
        self,
        topic: str,
        sub_topics: list[str]
    ) -> str:
        """サブトピックを並列に調査"""
        tasks = [
            self._research_with_limit(sub)
            for sub in sub_topics
        ]
        results = await asyncio.gather(*tasks)

        # 結果を統合
        combined = "\n\n".join(
            f"## {topic}\n{result}"
            for topic, result in zip(sub_topics, results)
        )

        return await self._synthesize_report(topic, combined)

    async def _research_with_limit(self, subtopic: str) -> str:
        """同時実行数を制限してリサーチ"""
        async with self.semaphore:
            return await self._research_subtopic(subtopic)

    async def _research_subtopic(self, subtopic: str) -> str:
        """個別のサブトピックを調査"""
        search_results = await self._async_search(subtopic)
        relevant_pages = await self._filter_and_read(search_results)
        return await self._summarize(subtopic, relevant_pages)

    async def _synthesize_report(
        self,
        topic: str,
        combined_findings: str
    ) -> str:
        """全サブトピックの結果を統合レポートにまとめる"""
        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"""以下のサブトピック調査結果を統合して、
包括的なレポートを作成してください。

メイントピック: {topic}

調査結果:
{combined_findings}

レポート要件:
1. エグゼクティブサマリー（300字以内）
2. 主要な発見事項（箇条書き）
3. 詳細分析（サブトピック間の関連性も含む）
4. 矛盾する情報があれば指摘
5. 結論と推奨事項
6. 参考文献リスト
"""
            }]
        )
        return response.content[0].text
```

---

## 8. 大規模情報の要約テクニック

### 8.1 Map-Reduce要約パターン

```python
class MapReduceSummarizer:
    """大量テキストのMap-Reduce要約"""

    def __init__(self, llm_client, chunk_size: int = 3000):
        self.client = llm_client
        self.chunk_size = chunk_size

    async def summarize(
        self,
        documents: list[str],
        final_prompt: str
    ) -> str:
        """Map-Reduceパターンで大量文書を要約"""

        # Map: 各文書を個別に要約
        chunk_summaries = []
        for doc in documents:
            chunks = self._split_into_chunks(doc)
            for chunk in chunks:
                summary = await self._summarize_chunk(chunk)
                chunk_summaries.append(summary)

        # Reduce: チャンク要約を統合
        while len(chunk_summaries) > 1:
            # 3-5個ずつ統合
            batches = [
                chunk_summaries[i:i+4]
                for i in range(0, len(chunk_summaries), 4)
            ]
            chunk_summaries = [
                await self._merge_summaries(batch)
                for batch in batches
            ]

        # Final: 最終要約
        final = await self._finalize(
            chunk_summaries[0], final_prompt
        )
        return final

    def _split_into_chunks(self, text: str) -> list[str]:
        """テキストをチャンクに分割"""
        words = text.split()
        chunks = []
        current = []
        current_len = 0

        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= self.chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

        if current:
            chunks.append(" ".join(current))

        return chunks

    async def _summarize_chunk(self, chunk: str) -> str:
        """個別チャンクを要約"""
        response = await self.client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"以下のテキストを200字以内で要約:\n\n{chunk}"
            }]
        )
        return response.content[0].text

    async def _merge_summaries(self, summaries: list[str]) -> str:
        """複数の要約を統合"""
        combined = "\n---\n".join(summaries)
        response = await self.client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"以下の要約を統合して1つにまとめ:\n\n{combined}"
            }]
        )
        return response.content[0].text
```

### 8.2 Refine要約パターン

```python
class RefineSummarizer:
    """逐次的に要約を改善していくパターン"""

    async def summarize(
        self,
        documents: list[str],
        topic: str
    ) -> str:
        """Refineパターンで要約を逐次改善"""
        current_summary = ""

        for i, doc in enumerate(documents):
            if i == 0:
                # 最初の文書から初期要約を作成
                current_summary = await self._initial_summary(
                    doc, topic
                )
            else:
                # 新しい文書の情報で要約を改善
                current_summary = await self._refine_summary(
                    current_summary, doc, topic
                )

        return current_summary

    async def _initial_summary(
        self, doc: str, topic: str
    ) -> str:
        """初期要約の作成"""
        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""トピック「{topic}」について、
以下の文書から重要な情報を要約してください。

文書:
{doc[:3000]}

要約を構造化して出力してください。"""
            }]
        )
        return response.content[0].text

    async def _refine_summary(
        self,
        current: str,
        new_doc: str,
        topic: str
    ) -> str:
        """新しい情報で要約を改善"""
        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""以下の既存の要約を、新しい文書の情報で
改善・拡充してください。

トピック: {topic}

既存の要約:
{current}

新しい文書:
{new_doc[:3000]}

改善された要約を出力してください。
矛盾する情報があれば指摘してください。"""
            }]
        )
        return response.content[0].text
```

---

## 9. アンチパターン

### アンチパターン1: 検索結果の鵜呑み

```python
# NG: 最初の検索結果をそのまま使用
results = search("AI市場規模")
return results[0]["snippet"]  # 古い or 不正確な可能性

# OK: 複数ソースでクロスチェック
results = search("AI市場規模 2025")
page1 = read(results[0]["url"])
page2 = read(results[1]["url"])
page3 = read(results[2]["url"])
# 3つのソースで一致する情報のみ採用
```

### アンチパターン2: バイアスのある検索

```python
# NG: 結論に合う情報だけ探す
search("AIエージェント 問題点")  # ネガティブな情報のみ

# OK: バランスの取れた調査
search("AIエージェント 利点 メリット")
search("AIエージェント 課題 限界")
search("AIエージェント 事例 成功")
search("AIエージェント 失敗 教訓")
```

### アンチパターン3: 引用なしのリサーチ

```
# NG: ソースを記載せずにファクトを記述
"AIエージェント市場は2025年に1000億ドルに達する見込みです。"

# OK: 必ず引用を付ける
"AIエージェント市場は2025年に1000億ドルに達する見込みです [1]。"
[1] Gartner, "AI Agent Market Forecast", 2024
```

### アンチパターン4: 古い情報の無検証な使用

```
# NG: 日付を確認せずに情報を使用
"ChatGPTの月間アクティブユーザーは1億人です"  # 2023年のデータかも

# OK: 日付を明記し、最新性を確認
"2025年1月時点で、ChatGPTの月間アクティブユーザーは
約3億人と報告されている [1]。なお、この数値は急速に
変化する可能性がある。"
```

---

## 10. FAQ

### Q1: リサーチエージェントの精度を上げるには？

- **クエリの多様化**: 同じトピックを異なる角度から検索
- **ソースの多様化**: Web、論文、公式ドキュメント等を組み合わせ
- **クロスチェック**: 重要な事実は3つ以上のソースで確認
- **新しさの重視**: 日付の新しい情報を優先
- **構造化出力**: JSON等の構造化形式で一貫性を確保

### Q2: ハルシネーション（でっち上げ）の防止策は？

- **引用必須**: すべての事実にソースURLを紐付け
- **検索結果の原文引用**: パラフレーズでなく原文を引用
- **「見つからなかった」の許容**: 情報がない場合は正直に報告
- **数値の慎重な扱い**: 統計データは必ず原典を確認
- **自信度の表明**: 不確実な情報には信頼度を付記

### Q3: 大量の情報をどう要約するか？

Map-Reduceパターンが有効:
1. **Map**: 各ページを個別に要約（200-300字）
2. **Reduce**: 個別要約を統合して最終要約を生成
3. **Refine**: 最終要約を目標に照らして改善

これにより、コンテキストウィンドウの制限内で大量情報を処理可能。

### Q4: リアルタイムデータの扱い方は？

- **タイムスタンプの明記**: 全データに取得日時を記録
- **鮮度の警告**: 古いデータには「N日前の情報」と注記
- **更新頻度の設定**: 定期的にリサーチを再実行するスケジュール
- **キャッシュ戦略**: 変化の少ない情報はキャッシュ、頻繁に変化する情報は毎回取得

### Q5: コスト最適化のコツは？

- **深度の適切な設定**: 全てのリサーチをdeepにする必要はない
- **モデルの使い分け**: 要約にはHaiku、分析にはSonnet/Opus
- **キャッシュの活用**: 同じ検索クエリの結果をキャッシュ
- **並列化の制御**: 同時実行数を制限してAPIコストを管理
- **段階的な深掘り**: まず浅く調査し、必要な部分だけ深掘り

---

## まとめ

| 項目 | 内容 |
|------|------|
| パイプライン | クエリ生成->収集->フィルタ->分析->レポート |
| 信頼性 | 情報源のスコアリング + クロスチェック |
| 引用管理 | すべての事実にソースを紐付け |
| 並列化 | サブトピックごとに並列調査 |
| 要約手法 | Map-Reduce / Refine パターン |
| 品質保証 | 複数ソース照合、バイアス排除 |
| 専門分野 | 市場調査、学術研究、技術調査 |
| 核心原則 | 「正確さ」>「網羅性」>「速度」 |

## 次に読むべきガイド

- [02-customer-support.md](./02-customer-support.md) -- カスタマーサポートエージェント
- [03-data-agents.md](./03-data-agents.md) -- データ分析エージェント
- [../01-patterns/01-multi-agent.md](../01-patterns/01-multi-agent.md) -- マルチエージェントでの協調リサーチ

## 参考文献

1. Nakano, R. et al., "WebGPT: Browser-assisted question-answering with human feedback" (2022) -- https://arxiv.org/abs/2112.09332
2. Trivedi, H. et al., "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions" (2023) -- https://arxiv.org/abs/2212.10509
3. Semantic Scholar API -- https://api.semanticscholar.org/
4. Shuster, K. et al., "Retrieval Augmentation Reduces Hallucination in Conversation" (2021) -- https://arxiv.org/abs/2104.07567
5. Lewis, P. et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) -- https://arxiv.org/abs/2005.11401
