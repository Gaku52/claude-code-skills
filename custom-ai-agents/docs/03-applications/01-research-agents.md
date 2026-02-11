# リサーチエージェント

> 情報収集・分析・要約――複数の情報源から自律的にデータを収集し、構造化されたリサーチレポートを生成するエージェントの設計と実装。

## この章で学ぶこと

1. リサーチエージェントの情報収集パイプラインと多段階フィルタリングの設計
2. Web検索・文書解析・データ統合の実装パターン
3. 信頼性の高いリサーチ出力のための検証と引用管理の仕組み

---

## 1. リサーチエージェントの全体像

```
リサーチエージェントのパイプライン

[目標設定]
    |
    v
[クエリ生成] ── 目標を複数の検索クエリに分解
    |
    v
[情報収集] ── Web検索、DB検索、文書読み取り
    |            ├── 検索エンジン
    |            ├── 学術論文DB (Semantic Scholar等)
    |            ├── 社内ドキュメント
    |            └── API / データベース
    v
[フィルタリング] ── 関連性・信頼性の評価
    |
    v
[分析・統合] ── 情報の構造化、矛盾の解決
    |
    v
[レポート生成] ── 構造化された出力、引用付き
    |
    v
[品質チェック] ── 事実確認、バイアスチェック
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
                        "num_results": {"type": "integer", "default": 10}
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
        messages = [{"role": "user", "content": f"リサーチトピック: {topic}"}]

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
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "リサーチが完了できませんでした"

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "web_search":
            return self._web_search(args["query"], args.get("num_results", 10))
        elif name == "read_webpage":
            return self._read_webpage(args["url"])
        elif name == "search_papers":
            return self._search_papers(args["query"], args.get("year_from"))
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

    def _search_papers(self, query: str, year_from: int = None) -> str:
        import requests
        params = {"query": query, "limit": 5}
        if year_from:
            params["year"] = f"{year_from}-"
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params
        )
        return json.dumps(resp.json().get("data", [])[:5], ensure_ascii=False)
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

---

## 3. 情報の信頼性評価

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

```python
# 情報源の信頼性スコアリング
class SourceReliabilityScorer:
    DOMAIN_SCORES = {
        "arxiv.org": 0.9,
        "nature.com": 0.95,
        "github.com": 0.7,
        "stackoverflow.com": 0.7,
        "wikipedia.org": 0.6,
        "medium.com": 0.4,
        "reddit.com": 0.3,
    }

    def score(self, url: str, content: str) -> float:
        """情報源の信頼性をスコアリング"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        # ドメインベースのスコア
        domain_score = self.DOMAIN_SCORES.get(domain, 0.5)

        # 引用の有無
        has_citations = any(
            marker in content
            for marker in ["[1]", "参考文献", "References", "doi:"]
        )
        citation_bonus = 0.1 if has_citations else 0

        # 日付の新しさ
        recency_bonus = self._check_recency(content)

        return min(1.0, domain_score + citation_bonus + recency_bonus)
```

---

## 4. リサーチパターン比較

### 4.1 リサーチ戦略比較

| 戦略 | 深さ | 時間 | コスト | 適用場面 |
|------|------|------|--------|---------|
| 広浅検索 | 低 | 短 | 低 | 概要把握 |
| 深掘り検索 | 高 | 長 | 中 | 特定トピック調査 |
| 多段階リサーチ | 高 | 最長 | 高 | 包括的レポート |
| 比較調査 | 中 | 中 | 中 | 選択肢の比較 |
| トレンド分析 | 中 | 中-長 | 中 | 時系列変化の把握 |

### 4.2 出力形式比較

| 形式 | 用途 | 長さ | 含むもの |
|------|------|------|---------|
| サマリー | 迅速な情報共有 | 100-300字 | 要点3-5個 |
| ブリーフィング | 意思決定支援 | 500-1000字 | 要点+根拠+推奨 |
| レポート | 詳細分析 | 2000-5000字 | 全セクション+引用 |
| データシート | 定量比較 | 表形式 | 数値+比較表 |

---

## 5. 引用管理

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
    accessed: str = field(default_factory=lambda: datetime.now().isoformat())

class CitationManager:
    def __init__(self):
        self.citations: list[Citation] = []
        self._counter = 0

    def add(self, title: str, url: str, **kwargs) -> str:
        """引用を追加し、参照番号を返す"""
        self._counter += 1
        citation = Citation(id=self._counter, title=title, url=url, **kwargs)
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
```

---

## 6. 並列リサーチ

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

```python
# 非同期並列リサーチ
import asyncio

class ParallelResearchAgent:
    async def research_parallel(self, topic: str, sub_topics: list[str]) -> str:
        """サブトピックを並列に調査"""
        tasks = [
            self._research_subtopic(sub)
            for sub in sub_topics
        ]
        results = await asyncio.gather(*tasks)

        # 結果を統合
        combined = "\n\n".join(
            f"## {topic}\n{result}"
            for topic, result in zip(sub_topics, results)
        )

        return await self._synthesize_report(topic, combined)

    async def _research_subtopic(self, subtopic: str) -> str:
        """個別のサブトピックを調査"""
        search_results = await self._async_search(subtopic)
        relevant_pages = await self._filter_and_read(search_results)
        return await self._summarize(subtopic, relevant_pages)
```

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: リサーチエージェントの精度を上げるには？

- **クエリの多様化**: 同じトピックを異なる角度から検索
- **ソースの多様化**: Web、論文、公式ドキュメント等を組み合わせ
- **クロスチェック**: 重要な事実は3つ以上のソースで確認
- **新しさの重視**: 日付の新しい情報を優先

### Q2: ハルシネーション（でっち上げ）の防止策は？

- **引用必須**: すべての事実にソースURLを紐付け
- **検索結果の原文引用**: パラフレーズでなく原文を引用
- **「見つからなかった」の許容**: 情報がない場合は正直に報告
- **数値の慎重な扱い**: 統計データは必ず原典を確認

### Q3: 大量の情報をどう要約するか？

Map-Reduceパターンが有効:
1. **Map**: 各ページを個別に要約（200-300字）
2. **Reduce**: 個別要約を統合して最終要約を生成
3. **Refine**: 最終要約を目標に照らして改善

これにより、コンテキストウィンドウの制限内で大量情報を処理可能。

---

## まとめ

| 項目 | 内容 |
|------|------|
| パイプライン | クエリ生成→収集→フィルタ→分析→レポート |
| 信頼性 | 情報源のスコアリング + クロスチェック |
| 引用管理 | すべての事実にソースを紐付け |
| 並列化 | サブトピックごとに並列調査 |
| 品質保証 | 複数ソース照合、バイアス排除 |
| 核心原則 | 「正確さ」>「網羅性」>「速度」 |

## 次に読むべきガイド

- [02-customer-support.md](./02-customer-support.md) — カスタマーサポートエージェント
- [03-data-agents.md](./03-data-agents.md) — データ分析エージェント
- [../01-patterns/01-multi-agent.md](../01-patterns/01-multi-agent.md) — マルチエージェントでの協調リサーチ

## 参考文献

1. Nakano, R. et al., "WebGPT: Browser-assisted question-answering with human feedback" (2022) — https://arxiv.org/abs/2112.09332
2. Trivedi, H. et al., "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions" (2023) — https://arxiv.org/abs/2212.10509
3. Semantic Scholar API — https://api.semanticscholar.org/
