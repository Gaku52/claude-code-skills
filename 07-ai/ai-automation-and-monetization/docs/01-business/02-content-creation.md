# コンテンツ制作 — ブログ、動画、SNS自動化

> AIを活用したコンテンツ制作の自動化と効率化を体系的に解説し、ブログ記事、動画制作、SNS運用の各領域で実践的な手法とツールチェーンを提供する。

---

## この章で学ぶこと

1. **AIコンテンツ制作パイプライン** — 企画→生成→編集→配信の自動化フロー設計
2. **マルチチャネル最適化** — ブログ、YouTube、Twitter/X、Instagram、LinkedIn向け最適コンテンツ生成
3. **品質管理とブランド一貫性** — AI生成コンテンツの品質保証とトーン・スタイルの統一
4. **収益化とスケーリング** — コンテンツ制作の事業化、KPI管理、チーム構築
5. **法務・倫理ガイドライン** — 著作権、景表法、ステルスマーケティング規制への対応

---

## 1. コンテンツ制作パイプライン

### 1.1 全体アーキテクチャ

```
┌──────────────────────────────────────────────────────────┐
│           AIコンテンツ制作パイプライン                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  企画         生成         編集        配信        分析    │
│  ┌────┐    ┌────┐     ┌────┐     ┌────┐     ┌────┐   │
│  │トレンド│──▶│AI   │──▶│品質  │──▶│各   │──▶│効果 │   │
│  │分析  │   │生成  │   │チェック│  │チャネル│  │測定  │   │
│  │キーワード│ │テキスト│  │ファクト│  │最適化 │  │改善  │   │
│  │競合分析│  │画像  │   │チェック│  │スケジュール││フィード│  │
│  └────┘    │動画  │   │ブランド│  └────┘     │バック │   │
│            └────┘     │チェック│              └────┘   │
│                        └────┘                           │
│                                                          │
│  ツール例:                                                │
│  BuzzSumo  GPT-4     Grammarly  Buffer     Google       │
│  Ahrefs    Claude    人間レビュー Hootsuite  Analytics    │
│  SEMrush   DALL-E              Zapier      PostHog      │
└──────────────────────────────────────────────────────────┘
```

### 1.2 コンテンツ種別と自動化レベル

| コンテンツ種別 | AI自動化率 | 人間介入 | 品質リスク | ROI |
|--------------|-----------|---------|-----------|-----|
| ブログ記事 | 70-80% | 編集・監修 | 中 | 高 |
| SNS投稿 | 80-90% | 承認 | 低 | 最高 |
| メルマガ | 60-70% | 編集・承認 | 中 | 高 |
| 動画スクリプト | 50-60% | 大幅編集 | 中〜高 | 中 |
| 動画編集 | 30-50% | 監修 | 高 | 中 |
| ホワイトペーパー | 40-50% | 大幅編集 | 高 | 中 |
| ポッドキャスト台本 | 60-70% | 編集 | 中 | 中 |
| プレスリリース | 50-60% | 法務確認 | 高 | 中 |
| ケーススタディ | 30-40% | 取材・編集 | 高 | 高 |

### 1.3 コンテンツ管理システム（CMS連携）

```python
import os
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class ContentStatus(Enum):
    IDEA = "idea"
    RESEARCHING = "researching"
    DRAFTING = "drafting"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class ContentType(Enum):
    BLOG = "blog"
    SNS_POST = "sns_post"
    VIDEO_SCRIPT = "video_script"
    NEWSLETTER = "newsletter"
    WHITEPAPER = "whitepaper"
    PODCAST = "podcast"
    PRESS_RELEASE = "press_release"

@dataclass
class ContentItem:
    """コンテンツアイテムの管理単位"""
    id: str
    title: str
    content_type: ContentType
    status: ContentStatus = ContentStatus.IDEA
    topic: str = ""
    keywords: list[str] = field(default_factory=list)
    target_audience: str = ""
    author: str = ""
    ai_model_used: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    platforms: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    version: int = 1
    content_body: str = ""
    meta_description: str = ""
    tags: list[str] = field(default_factory=list)
    internal_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content_type": self.content_type.value,
            "status": self.status.value,
            "topic": self.topic,
            "keywords": self.keywords,
            "target_audience": self.target_audience,
            "author": self.author,
            "ai_model_used": self.ai_model_used,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "platforms": self.platforms,
            "metrics": self.metrics,
            "version": self.version,
            "meta_description": self.meta_description,
            "tags": self.tags,
        }


class ContentPipeline:
    """コンテンツ制作パイプライン管理"""

    def __init__(self, storage_path: str = "./content_db"):
        self.storage_path = storage_path
        self.items: dict[str, ContentItem] = {}
        os.makedirs(storage_path, exist_ok=True)

    def create_content(self, title: str, content_type: ContentType,
                       topic: str, keywords: list[str],
                       target_audience: str = "") -> ContentItem:
        """新規コンテンツアイテムの作成"""
        import uuid
        item = ContentItem(
            id=str(uuid.uuid4())[:8],
            title=title,
            content_type=content_type,
            topic=topic,
            keywords=keywords,
            target_audience=target_audience,
        )
        self.items[item.id] = item
        self._save(item)
        return item

    def advance_status(self, item_id: str) -> ContentItem:
        """ステータスを次の段階に進める"""
        item = self.items[item_id]
        status_flow = [
            ContentStatus.IDEA,
            ContentStatus.RESEARCHING,
            ContentStatus.DRAFTING,
            ContentStatus.REVIEWING,
            ContentStatus.APPROVED,
            ContentStatus.SCHEDULED,
            ContentStatus.PUBLISHED,
        ]
        current_idx = status_flow.index(item.status)
        if current_idx < len(status_flow) - 1:
            item.status = status_flow[current_idx + 1]
            item.updated_at = datetime.now()
            self._save(item)
        return item

    def get_by_status(self, status: ContentStatus) -> list[ContentItem]:
        """ステータス別にコンテンツを取得"""
        return [item for item in self.items.values() if item.status == status]

    def get_overdue(self) -> list[ContentItem]:
        """期限超過のコンテンツを取得"""
        now = datetime.now()
        return [
            item for item in self.items.values()
            if item.scheduled_at and item.scheduled_at < now
            and item.status != ContentStatus.PUBLISHED
        ]

    def get_pipeline_summary(self) -> dict:
        """パイプライン全体のサマリーを取得"""
        summary = {}
        for status in ContentStatus:
            items = self.get_by_status(status)
            summary[status.value] = {
                "count": len(items),
                "items": [i.title for i in items]
            }
        return summary

    def _save(self, item: ContentItem):
        filepath = os.path.join(self.storage_path, f"{item.id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(item.to_dict(), f, ensure_ascii=False, indent=2)
```

### 1.4 コンテンツ戦略フレームワーク

```python
class ContentStrategy:
    """コンテンツ戦略の設計と管理"""

    def __init__(self, client):
        self.client = client

    def generate_content_pillars(self, business_description: str,
                                  target_audience: str,
                                  competitors: list[str]) -> dict:
        """コンテンツピラー（柱）の設計"""
        prompt = f"""
あなたはコンテンツマーケティングの専門家です。
以下のビジネスに対して、コンテンツ戦略のピラー（柱）を設計してください。

ビジネス概要: {business_description}
ターゲット読者: {target_audience}
主要競合: {', '.join(competitors)}

以下の形式で回答:
1. コンテンツピラー（3-5個）
   - ピラー名
   - 概要（1-2文）
   - 対象キーワード群（5-10個）
   - コンテンツタイプ（ブログ、動画、SNS等）
   - 想定記事テーマ例（5個）

2. コンテンツミックス比率
   - 教育コンテンツ: X%
   - エンタメコンテンツ: X%
   - セールスコンテンツ: X%
   - コミュニティコンテンツ: X%

3. 競合との差別化ポイント
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"strategy": response.content[0].text}

    def create_editorial_calendar(self, pillars: list[str],
                                   months: int = 3) -> dict:
        """編集カレンダーの自動生成"""
        prompt = f"""
以下のコンテンツピラーに基づき、{months}ヶ月分の編集カレンダーを作成:

ピラー: {', '.join(pillars)}

要件:
- 週3本のブログ記事
- 毎日のSNS投稿（Twitter/LinkedIn）
- 月2本の動画コンテンツ
- 月1本のメルマガ

各エントリに:
- 公開日
- コンテンツタイプ
- タイトル案
- ターゲットキーワード
- 担当（AI/人間/ハイブリッド）
- 制作所要時間（目安）

季節イベント、業界イベントも考慮すること。
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"calendar": response.content[0].text}
```

---

## 2. ブログ記事AI生成

### 2.1 記事生成エンジン

```python
import anthropic
from dataclasses import dataclass

@dataclass
class ArticlePlan:
    topic: str
    target_keyword: str
    secondary_keywords: list[str]
    target_length: int
    tone: str
    audience: str
    outline: list[str] = None

class BlogGenerator:
    """AIブログ記事生成エンジン"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.brand_voice = ""

    def set_brand_voice(self, examples: list[str]):
        """ブランドボイスを学習"""
        self.brand_voice = f"""
以下は弊社のブログ記事例です。このトーンとスタイルを維持してください:

{chr(10).join(f'例{i+1}: {ex[:500]}' for i, ex in enumerate(examples[:3]))}
"""

    def generate_outline(self, plan: ArticlePlan) -> list[str]:
        """記事構成案を生成"""
        prompt = f"""
以下のブログ記事の構成案（見出しリスト）を生成:
- トピック: {plan.topic}
- メインKW: {plan.target_keyword}
- サブKW: {', '.join(plan.secondary_keywords)}
- 対象読者: {plan.audience}
- 目標文字数: {plan.target_length}文字

SEO最適化された見出し構成を提案。H2は5-7個、H3は各H2に2-3個。
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def generate_article(self, plan: ArticlePlan) -> dict:
        """記事本文を生成"""
        outline = plan.outline or self.generate_outline(plan)

        prompt = f"""
{self.brand_voice}

以下の構成でブログ記事を執筆:

トピック: {plan.topic}
構成: {outline}
トーン: {plan.tone}
メインキーワード: {plan.target_keyword}（自然に5-8回使用）
サブキーワード: {', '.join(plan.secondary_keywords)}
目標文字数: {plan.target_length}文字

ルール:
- 導入部は読者の課題に共感し、記事を読む価値を明示
- 各セクションに具体例やデータを含める
- CTAで次のアクションを促す
- meta descriptionも生成
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        article = response.content[0].text
        return {
            "content": article,
            "word_count": len(article),
            "seo_score": self._calculate_seo_score(
                article, plan.target_keyword, plan.secondary_keywords
            ),
            "readability_score": self._calculate_readability(article)
        }

    def _calculate_seo_score(self, text, main_kw, sub_kws) -> float:
        """簡易SEOスコア計算"""
        score = 0
        main_count = text.lower().count(main_kw.lower())
        if 3 <= main_count <= 10:
            score += 30
        for kw in sub_kws:
            if kw.lower() in text.lower():
                score += 10
        if len(text) >= 2000:
            score += 20
        return min(100, score)

    def _calculate_readability(self, text: str) -> float:
        """可読性スコア（日本語向け簡易版）"""
        sentences = text.split("。")
        avg_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
        if avg_length < 60:
            return 90
        elif avg_length < 80:
            return 70
        else:
            return 50
```

### 2.2 SEO最適化フロー

```
SEO最適化パイプライン:

  キーワードリサーチ
  ┌────────────┐
  │ Ahrefs/    │──▶ 検索ボリューム、難易度、関連KW
  │ SEMrush    │
  └────────────┘
        │
        ▼
  競合分析
  ┌────────────┐
  │ 上位10記事  │──▶ 文字数、構成、カバー範囲
  │ スクレイピング│
  └────────────┘
        │
        ▼
  記事生成
  ┌────────────┐
  │ AI生成     │──▶ 競合を上回る構成・深さ・独自性
  │ (Claude)   │
  └────────────┘
        │
        ▼
  最適化
  ┌────────────┐
  │ タイトル    │──▶ 32文字以内、KW含む、CTR最適化
  │ メタ       │──▶ 120文字、行動喚起
  │ 見出し     │──▶ KW含む、階層構造
  │ 内部リンク  │──▶ 関連記事への自然なリンク
  └────────────┘
```

### 2.3 高度なSEO分析エンジン

```python
import re
from collections import Counter
from dataclasses import dataclass

@dataclass
class SEOAnalysis:
    """SEO分析結果"""
    overall_score: float
    title_score: float
    meta_score: float
    heading_score: float
    keyword_density: float
    readability_score: float
    internal_links: int
    external_links: int
    image_alt_coverage: float
    word_count: int
    recommendations: list[str]

class AdvancedSEOAnalyzer:
    """高度なSEO分析エンジン"""

    def __init__(self):
        self.min_word_count = 1500
        self.max_keyword_density = 0.03
        self.min_keyword_density = 0.005
        self.optimal_title_length = 32  # 日本語文字数
        self.optimal_meta_length = 120  # 日本語文字数

    def analyze(self, content: str, title: str,
                meta_description: str,
                target_keyword: str,
                secondary_keywords: list[str]) -> SEOAnalysis:
        """総合SEO分析を実行"""
        recommendations = []

        # タイトル分析
        title_score = self._analyze_title(title, target_keyword, recommendations)

        # メタディスクリプション分析
        meta_score = self._analyze_meta(meta_description, target_keyword, recommendations)

        # 見出し構造分析
        heading_score = self._analyze_headings(content, target_keyword, recommendations)

        # キーワード密度分析
        keyword_density = self._calculate_keyword_density(
            content, target_keyword
        )
        if keyword_density > self.max_keyword_density:
            recommendations.append(
                f"キーワード密度が高すぎます({keyword_density:.1%})。"
                f"{self.max_keyword_density:.1%}以下に抑えてください。"
            )
        elif keyword_density < self.min_keyword_density:
            recommendations.append(
                f"キーワード密度が低すぎます({keyword_density:.1%})。"
                f"自然にキーワードを追加してください。"
            )

        # 可読性分析
        readability_score = self._analyze_readability(content, recommendations)

        # リンク分析
        internal_links = len(re.findall(r'\[.*?\]\((?!https?://)', content))
        external_links = len(re.findall(r'\[.*?\]\(https?://', content))
        if internal_links < 3:
            recommendations.append("内部リンクを3個以上追加してください。")
        if external_links < 2:
            recommendations.append("信頼できる外部ソースへのリンクを追加してください。")

        # 画像alt分析
        images = re.findall(r'!\[(.*?)\]', content)
        images_with_alt = [img for img in images if img.strip()]
        image_alt_coverage = len(images_with_alt) / max(len(images), 1)

        # 文字数チェック
        word_count = len(content)
        if word_count < self.min_word_count:
            recommendations.append(
                f"文字数が不足しています（{word_count}文字）。"
                f"{self.min_word_count}文字以上を目指してください。"
            )

        # サブキーワードカバレッジ
        covered = sum(1 for kw in secondary_keywords if kw in content)
        if covered < len(secondary_keywords) * 0.7:
            missing = [kw for kw in secondary_keywords if kw not in content]
            recommendations.append(
                f"以下のサブキーワードが未使用です: {', '.join(missing[:5])}"
            )

        # 総合スコア計算
        overall_score = (
            title_score * 0.15 +
            meta_score * 0.1 +
            heading_score * 0.15 +
            min(100, (1 - abs(keyword_density - 0.015) / 0.015) * 100) * 0.15 +
            readability_score * 0.15 +
            min(100, internal_links * 20) * 0.1 +
            image_alt_coverage * 100 * 0.1 +
            min(100, word_count / self.min_word_count * 100) * 0.1
        )

        return SEOAnalysis(
            overall_score=round(overall_score, 1),
            title_score=title_score,
            meta_score=meta_score,
            heading_score=heading_score,
            keyword_density=keyword_density,
            readability_score=readability_score,
            internal_links=internal_links,
            external_links=external_links,
            image_alt_coverage=image_alt_coverage,
            word_count=word_count,
            recommendations=recommendations
        )

    def _analyze_title(self, title: str, keyword: str,
                       recommendations: list) -> float:
        """タイトル分析"""
        score = 0
        if keyword in title:
            score += 40
        else:
            recommendations.append("タイトルにメインキーワードを含めてください。")

        title_len = len(title)
        if 20 <= title_len <= self.optimal_title_length:
            score += 30
        elif title_len > self.optimal_title_length:
            recommendations.append(
                f"タイトルが長すぎます({title_len}文字)。"
                f"{self.optimal_title_length}文字以内に。"
            )
            score += 15

        # パワーワードチェック
        power_words = ["完全ガイド", "入門", "まとめ", "比較",
                       "おすすめ", "方法", "解説", "徹底"]
        if any(pw in title for pw in power_words):
            score += 15
        # 数字を含むか
        if re.search(r'\d+', title):
            score += 15
        return min(100, score)

    def _analyze_meta(self, meta: str, keyword: str,
                      recommendations: list) -> float:
        """メタディスクリプション分析"""
        score = 0
        if keyword in meta:
            score += 40
        if 80 <= len(meta) <= self.optimal_meta_length:
            score += 30
        elif len(meta) > self.optimal_meta_length:
            recommendations.append("メタディスクリプションが長すぎます。")
            score += 15
        # CTA含有チェック
        cta_words = ["詳しく", "チェック", "今すぐ", "無料", "限定"]
        if any(cta in meta for cta in cta_words):
            score += 30
        return min(100, score)

    def _analyze_headings(self, content: str, keyword: str,
                          recommendations: list) -> float:
        """見出し構造分析"""
        score = 0
        h2_matches = re.findall(r'^## (.+)$', content, re.MULTILINE)
        h3_matches = re.findall(r'^### (.+)$', content, re.MULTILINE)

        if 4 <= len(h2_matches) <= 8:
            score += 30
        elif len(h2_matches) < 4:
            recommendations.append("H2見出しを4個以上設置してください。")

        if len(h3_matches) >= len(h2_matches):
            score += 20

        # キーワード含有率
        kw_in_headings = sum(
            1 for h in h2_matches + h3_matches if keyword in h
        )
        if kw_in_headings >= 2:
            score += 30

        # 階層構造の正しさ
        lines = content.split('\n')
        prev_level = 0
        structure_ok = True
        for line in lines:
            if line.startswith('####'):
                level = 4
            elif line.startswith('###'):
                level = 3
            elif line.startswith('##'):
                level = 2
            elif line.startswith('#'):
                level = 1
            else:
                continue
            if level > prev_level + 1 and prev_level > 0:
                structure_ok = False
            prev_level = level

        if structure_ok:
            score += 20
        else:
            recommendations.append("見出しの階層構造が不正です（H2→H4のスキップ等）。")

        return min(100, score)

    def _calculate_keyword_density(self, content: str,
                                    keyword: str) -> float:
        """キーワード密度を計算"""
        total_chars = len(content)
        if total_chars == 0:
            return 0
        keyword_count = content.count(keyword)
        return (keyword_count * len(keyword)) / total_chars

    def _analyze_readability(self, content: str,
                              recommendations: list) -> float:
        """可読性分析"""
        sentences = [s.strip() for s in content.split("。") if s.strip()]
        if not sentences:
            return 50

        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
        score = 0

        if avg_sentence_length < 50:
            score += 40
        elif avg_sentence_length < 80:
            score += 25
        else:
            recommendations.append("文が長すぎます。1文60文字以内を目安に。")
            score += 10

        # 段落チェック
        paragraphs = content.split('\n\n')
        avg_para_length = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        if avg_para_length < 300:
            score += 30
        elif avg_para_length < 500:
            score += 20
        else:
            recommendations.append("段落が長すぎます。適宜改行を入れてください。")

        # リストや箇条書きの使用
        list_items = len(re.findall(r'^[-*]\s', content, re.MULTILINE))
        if list_items >= 3:
            score += 30
        elif list_items >= 1:
            score += 15

        return min(100, score)
```

### 2.4 コンテンツリライトエンジン

```python
class ContentRewriter:
    """既存記事のAIリライト・改善エンジン"""

    def __init__(self, client):
        self.client = client

    def rewrite_for_freshness(self, original: str,
                               new_data: str = "") -> str:
        """既存記事の鮮度を更新"""
        prompt = f"""
以下の既存記事を最新情報で更新してください。

既存記事:
{original[:3000]}

最新情報（ある場合）:
{new_data}

指示:
1. 古い統計やデータを最新のものに更新
2. 新しいトレンドや技術に言及
3. 日付参照を更新
4. 既存の構成や良い点は維持
5. SEO面で改善できる部分があれば対応
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def expand_thin_content(self, original: str,
                             target_length: int = 3000) -> str:
        """薄いコンテンツの拡充"""
        prompt = f"""
以下の記事を{target_length}文字以上に拡充してください。

現在の記事:
{original}

拡充方針:
1. 各セクションに具体例や事例を追加
2. データや統計を引用
3. 実践的なステップバイステップの手順を追加
4. FAQ セクションを追加
5. 関連するサブトピックをカバー
6. 読者の疑問に先回りして回答

元の構成や主張は維持すること。
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def convert_format(self, content: str,
                        from_format: str,
                        to_format: str) -> str:
        """コンテンツ形式の変換"""
        prompt = f"""
以下の{from_format}形式のコンテンツを{to_format}形式に変換してください。

元コンテンツ:
{content[:3000]}

変換ルール:
- ブログ → SNS: 核心メッセージを抽出、簡潔に
- ブログ → 動画スクリプト: 視覚的表現を追加、話し言葉に
- SNS → ブログ: 詳細を追加、SEO要素を含める
- ブログ → メルマガ: パーソナルなトーン、CTA強化
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class ContentRepurposer:
    """1つのコンテンツから複数形式への再利用エンジン"""

    def __init__(self, client):
        self.client = client

    def repurpose(self, source_content: str,
                   source_type: str = "blog") -> dict:
        """1つのコンテンツを複数形式に展開"""
        prompt = f"""
以下の{source_type}コンテンツを、複数の形式に変換してください。

元コンテンツ:
{source_content[:3000]}

以下の形式を生成:
1. Twitter/Xスレッド（5-8ツイート）
2. LinkedInの長文投稿（500-1000文字）
3. Instagram用キャプション（ハッシュタグ10個付き）
4. YouTubeショート用スクリプト（60秒）
5. メルマガの一部として使えるセクション（300文字）
6. ポッドキャスト台本の話題（2分間）

各形式ごとに最適化し、プラットフォーム特性を活かすこと。
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"repurposed": response.content[0].text}
```

---

## 3. SNS自動化

### 3.1 マルチプラットフォーム投稿生成

```python
class SocialMediaGenerator:
    """マルチプラットフォームSNS投稿生成"""

    PLATFORM_SPECS = {
        "twitter": {
            "max_length": 280,
            "tone": "カジュアル、簡潔、インパクト重視",
            "hashtags": 2,
            "emoji": True
        },
        "linkedin": {
            "max_length": 3000,
            "tone": "プロフェッショナル、知見共有、ストーリー",
            "hashtags": 5,
            "emoji": False
        },
        "instagram": {
            "max_length": 2200,
            "tone": "ビジュアル訴求、共感、ライフスタイル",
            "hashtags": 15,
            "emoji": True
        },
        "facebook": {
            "max_length": 500,
            "tone": "親しみやすい、コミュニティ、質問形式",
            "hashtags": 3,
            "emoji": True
        }
    }

    def __init__(self, client):
        self.client = client

    def generate_posts(self, content: str,
                       platforms: list[str]) -> dict:
        """1つのコンテンツから各プラットフォーム用の投稿を生成"""
        prompt = f"""
以下のコンテンツを各SNSプラットフォーム向けに変換:

元コンテンツ:
{content}

各プラットフォームの制約:
{self._format_specs(platforms)}

JSON形式で各プラットフォームの投稿を返す。
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_posts(response.content[0].text)

    def create_content_calendar(self, topics: list[str],
                                 frequency: dict) -> list[dict]:
        """コンテンツカレンダー自動生成"""
        prompt = f"""
以下のトピックから1ヶ月分のSNSコンテンツカレンダーを作成:

トピック: {', '.join(topics)}
投稿頻度:
- Twitter: {frequency.get('twitter', '毎日2回')}
- LinkedIn: {frequency.get('linkedin', '週3回')}
- Instagram: {frequency.get('instagram', '週5回')}

各投稿に: 日時、プラットフォーム、投稿内容、ハッシュタグ、画像のイメージ
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_calendar(response.content[0].text)

    def _format_specs(self, platforms):
        return "\n".join(
            f"- {p}: {self.PLATFORM_SPECS[p]}" for p in platforms
        )

    def _parse_posts(self, text):
        """レスポンスをパース"""
        try:
            import json
            return json.loads(text)
        except Exception:
            return {"raw": text}

    def _parse_calendar(self, text):
        """カレンダーレスポンスをパース"""
        return [{"content": text}]
```

### 3.2 プラットフォーム別最適化比較

| 要素 | Twitter/X | LinkedIn | Instagram | Facebook | TikTok |
|------|----------|----------|-----------|----------|--------|
| 最適投稿長 | 70-100文字 | 500-1500文字 | 500-1000文字 | 100-250文字 | 15-50文字 |
| ベスト投稿時間 | 12:00, 18:00 | 8:00, 12:00 | 11:00, 19:00 | 13:00, 16:00 | 19:00, 21:00 |
| 画像重要度 | 中 | 中 | 最高 | 高 | 動画必須 |
| ハッシュタグ数 | 1-2 | 3-5 | 10-15 | 2-3 | 3-5 |
| エンゲージメント型 | RT/引用 | コメント | いいね/保存 | シェア | いいね/シェア |
| アルゴリズム優遇 | 引用RT | 長文投稿 | リール | ライブ | デュエット |
| コンテンツ寿命 | 数時間 | 1-3日 | 1-2日 | 1日 | 1-7日 |
| B2B向き | 中 | 最高 | 低 | 中 | 低 |
| B2C向き | 高 | 低 | 最高 | 高 | 最高 |

### 3.3 エンゲージメント分析エンジン

```python
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class PostMetrics:
    """投稿メトリクス"""
    post_id: str
    platform: str
    published_at: datetime
    impressions: int = 0
    engagements: int = 0
    clicks: int = 0
    shares: int = 0
    comments: int = 0
    likes: int = 0
    saves: int = 0
    profile_visits: int = 0
    followers_gained: int = 0

    @property
    def engagement_rate(self) -> float:
        return self.engagements / max(self.impressions, 1)

    @property
    def click_through_rate(self) -> float:
        return self.clicks / max(self.impressions, 1)

    @property
    def virality_score(self) -> float:
        return self.shares / max(self.engagements, 1)


class EngagementAnalyzer:
    """SNSエンゲージメント分析"""

    def __init__(self):
        self.metrics_history: list[PostMetrics] = []

    def add_metrics(self, metrics: PostMetrics):
        """メトリクスを追加"""
        self.metrics_history.append(metrics)

    def get_best_performing(self, platform: str = None,
                             metric: str = "engagement_rate",
                             top_n: int = 10) -> list[PostMetrics]:
        """パフォーマンスの高い投稿を取得"""
        filtered = self.metrics_history
        if platform:
            filtered = [m for m in filtered if m.platform == platform]

        return sorted(
            filtered,
            key=lambda m: getattr(m, metric, 0),
            reverse=True
        )[:top_n]

    def get_optimal_posting_times(self, platform: str) -> dict:
        """最適な投稿時間を分析"""
        platform_metrics = [
            m for m in self.metrics_history if m.platform == platform
        ]

        hour_performance = {}
        for m in platform_metrics:
            hour = m.published_at.hour
            if hour not in hour_performance:
                hour_performance[hour] = []
            hour_performance[hour].append(m.engagement_rate)

        avg_by_hour = {
            hour: sum(rates) / len(rates)
            for hour, rates in hour_performance.items()
        }

        sorted_hours = sorted(
            avg_by_hour.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "best_hours": sorted_hours[:3],
            "worst_hours": sorted_hours[-3:],
            "all_hours": dict(sorted_hours)
        }

    def get_content_type_performance(self, platform: str) -> dict:
        """コンテンツタイプ別のパフォーマンス分析"""
        platform_metrics = [
            m for m in self.metrics_history if m.platform == platform
        ]

        total = len(platform_metrics)
        avg_engagement = sum(
            m.engagement_rate for m in platform_metrics
        ) / max(total, 1)

        return {
            "total_posts": total,
            "avg_engagement_rate": f"{avg_engagement:.2%}",
            "total_impressions": sum(m.impressions for m in platform_metrics),
            "total_engagements": sum(m.engagements for m in platform_metrics),
            "total_clicks": sum(m.clicks for m in platform_metrics),
        }

    def generate_weekly_report(self, week_start: datetime) -> dict:
        """週次レポート生成"""
        week_end = week_start + timedelta(days=7)
        week_metrics = [
            m for m in self.metrics_history
            if week_start <= m.published_at < week_end
        ]

        platforms = set(m.platform for m in week_metrics)
        report = {
            "period": f"{week_start.strftime('%Y-%m-%d')} - {week_end.strftime('%Y-%m-%d')}",
            "total_posts": len(week_metrics),
            "platforms": {}
        }

        for platform in platforms:
            p_metrics = [m for m in week_metrics if m.platform == platform]
            report["platforms"][platform] = {
                "posts": len(p_metrics),
                "total_impressions": sum(m.impressions for m in p_metrics),
                "avg_engagement_rate": f"{sum(m.engagement_rate for m in p_metrics) / max(len(p_metrics), 1):.2%}",
                "top_post": max(p_metrics, key=lambda m: m.engagement_rate).post_id
                if p_metrics else None,
                "total_followers_gained": sum(m.followers_gained for m in p_metrics),
            }

        return report
```

### 3.4 自動投稿スケジューラ

```python
import asyncio
from datetime import datetime, timedelta
from typing import Callable

class PostScheduler:
    """SNS自動投稿スケジューラ"""

    def __init__(self):
        self.scheduled_posts: list[dict] = []
        self.posted: list[dict] = []
        self.platform_apis: dict[str, Callable] = {}

    def register_platform(self, platform: str, api_func: Callable):
        """プラットフォームAPIを登録"""
        self.platform_apis[platform] = api_func

    def schedule_post(self, platform: str, content: str,
                       post_at: datetime,
                       media_urls: list[str] = None,
                       hashtags: list[str] = None):
        """投稿をスケジュール"""
        post = {
            "id": f"{platform}_{post_at.strftime('%Y%m%d%H%M')}",
            "platform": platform,
            "content": content,
            "post_at": post_at,
            "media_urls": media_urls or [],
            "hashtags": hashtags or [],
            "status": "scheduled"
        }
        self.scheduled_posts.append(post)
        self.scheduled_posts.sort(key=lambda p: p["post_at"])
        return post

    def schedule_batch(self, posts: list[dict]):
        """バッチスケジュール"""
        for post in posts:
            self.schedule_post(**post)

    def get_upcoming(self, hours: int = 24) -> list[dict]:
        """今後の予定投稿を取得"""
        cutoff = datetime.now() + timedelta(hours=hours)
        return [
            p for p in self.scheduled_posts
            if p["status"] == "scheduled" and p["post_at"] <= cutoff
        ]

    async def execute_scheduled(self):
        """スケジュール済み投稿を実行"""
        now = datetime.now()
        due_posts = [
            p for p in self.scheduled_posts
            if p["status"] == "scheduled" and p["post_at"] <= now
        ]

        results = []
        for post in due_posts:
            try:
                api_func = self.platform_apis.get(post["platform"])
                if api_func:
                    result = await api_func(
                        content=post["content"],
                        media_urls=post["media_urls"]
                    )
                    post["status"] = "posted"
                    post["result"] = result
                    self.posted.append(post)
                    results.append({"id": post["id"], "status": "success"})
                else:
                    post["status"] = "error"
                    post["error"] = f"No API registered for {post['platform']}"
                    results.append({"id": post["id"], "status": "error"})
            except Exception as e:
                post["status"] = "error"
                post["error"] = str(e)
                results.append({"id": post["id"], "status": "error", "error": str(e)})

        return results

    def get_analytics_summary(self) -> dict:
        """スケジューラの分析サマリー"""
        return {
            "total_scheduled": len(self.scheduled_posts),
            "total_posted": len(self.posted),
            "pending": len([p for p in self.scheduled_posts if p["status"] == "scheduled"]),
            "errors": len([p for p in self.scheduled_posts if p["status"] == "error"]),
            "by_platform": {
                platform: len([p for p in self.scheduled_posts if p["platform"] == platform])
                for platform in set(p["platform"] for p in self.scheduled_posts)
            }
        }
```

---

## 4. 動画コンテンツ自動化

### 4.1 動画制作パイプライン

```
動画AI制作パイプライン:

  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
  │企画     │──▶│スクリプト│──▶│素材生成 │──▶│編集     │
  │トレンド │   │AI生成   │   │AI画像   │   │AI編集   │
  │分析     │   │構成案   │   │TTS音声  │   │字幕     │
  └─────────┘   └─────────┘   └─────────┘   └─────────┘
                                                  │
                                           ┌──────▼──────┐
                                           │サムネイル    │
                                           │AI生成       │
                                           │A/Bテスト    │
                                           └──────┬──────┘
                                                  │
                                           ┌──────▼──────┐
                                           │配信・分析    │
                                           │YouTube      │
                                           │TikTok       │
                                           │Instagram    │
                                           └─────────────┘
```

### 4.2 動画スクリプト生成

```python
class VideoScriptGenerator:
    """動画スクリプトAI生成"""

    def __init__(self, client):
        self.client = client

    def generate_youtube_script(self, topic: str,
                                 duration_minutes: int = 10) -> dict:
        """YouTube動画スクリプト生成"""
        prompt = f"""
{duration_minutes}分のYouTube動画スクリプトを作成:

トピック: {topic}
構成:
1. フック（最初の10秒で視聴者を引きつける）
2. 問題提起（なぜこのトピックが重要か）
3. 本編（3-5つのポイントに分割）
4. CTA（チャンネル登録、コメント促進）

各セクションに:
- 話す内容（ナレーション文）
- 画面表示の指示（テロップ、画像、アニメーション）
- タイムスタンプ目安

タイトル案を3つ（CTR最適化）、説明文、タグも作成。
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"script": response.content[0].text}

    def generate_short_script(self, topic: str,
                               platform: str = "tiktok") -> dict:
        """ショート動画スクリプト生成（60秒以内）"""
        prompt = f"""
{platform}向け60秒ショート動画スクリプト:

トピック: {topic}
構成:
- 0-3秒: フック（スクロールを止める一言）
- 3-10秒: 問題提起
- 10-45秒: 解決策/コンテンツ
- 45-60秒: CTA

テンポよく、1文は短く、視覚的な演出指示も含める。
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"script": response.content[0].text}

    def generate_tutorial_script(self, topic: str,
                                  steps: list[str],
                                  duration_minutes: int = 15) -> dict:
        """チュートリアル動画スクリプト生成"""
        prompt = f"""
{duration_minutes}分のチュートリアル動画スクリプトを作成:

トピック: {topic}
ステップ: {chr(10).join(f'{i+1}. {s}' for i, s in enumerate(steps))}

構成:
1. イントロ（完成物を先に見せる）: 30秒
2. 準備（必要な環境・ツール）: 1分
3. 各ステップの実演: メインパート
4. まとめとTips: 1分
5. CTA: 15秒

各ステップに:
- 画面操作の詳細手順
- ナレーション文
- 重要ポイントのテロップ指示
- つまづきやすい箇所の注意事項
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"script": response.content[0].text}
```

### 4.3 サムネイル最適化エンジン

```python
class ThumbnailOptimizer:
    """サムネイル生成・最適化"""

    def __init__(self, client):
        self.client = client

    def generate_thumbnail_concept(self, video_title: str,
                                     video_topic: str) -> dict:
        """サムネイルのコンセプトを生成"""
        prompt = f"""
以下のYouTube動画のサムネイルコンセプトを3パターン提案してください。

タイトル: {video_title}
トピック: {video_topic}

各パターンに:
1. メインビジュアルの説明
2. テキストオーバーレイ（3-5文字の短い言葉）
3. 色使い（背景色、テキスト色）
4. 表情/感情（驚き、疑問、喜びなど）
5. レイアウト（人物の位置、テキストの位置）
6. A/Bテスト時のバリエーション提案

CTR最適化のポイント:
- コントラストの高い色使い
- 人の顔が入っているとCTRが上がる
- テキストは3-5文字で読みやすく
- 画面の1/3ルールを意識
- 競合サムネイルとの差別化
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"concepts": response.content[0].text}

    def analyze_thumbnail_performance(self,
                                        thumbnails: list[dict]) -> dict:
        """サムネイルのA/Bテスト結果を分析"""
        best = max(thumbnails, key=lambda t: t.get("ctr", 0))
        worst = min(thumbnails, key=lambda t: t.get("ctr", 0))

        return {
            "best_performing": best,
            "worst_performing": worst,
            "avg_ctr": sum(t.get("ctr", 0) for t in thumbnails) / len(thumbnails),
            "recommendation": f"パターン'{best.get('name', 'A')}'が最も効果的。"
                             f"CTR: {best.get('ctr', 0):.2%}"
        }


class VideoSEOOptimizer:
    """YouTube SEO最適化"""

    def __init__(self, client):
        self.client = client

    def optimize_metadata(self, video_title: str,
                           video_description: str,
                           target_keyword: str) -> dict:
        """動画メタデータのSEO最適化"""
        prompt = f"""
以下のYouTube動画のメタデータをSEO最適化してください。

現在のタイトル: {video_title}
現在の説明文: {video_description}
ターゲットキーワード: {target_keyword}

以下を生成:
1. 最適化タイトル案（3パターン）
   - 60文字以内
   - キーワードを前半に含む
   - CTRを高める要素を含む

2. 最適化説明文
   - 最初の2行にキーワードと核心内容
   - タイムスタンプ（チャプター）
   - 関連リンク
   - SNSリンク
   - ハッシュタグ（5-10個）

3. タグ候補（15-20個）
   - メインキーワード
   - ロングテールキーワード
   - 関連キーワード
   - 競合が使用しているタグ推測

4. カード・エンドスクリーンの提案
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"optimized_metadata": response.content[0].text}

    def generate_chapters(self, script: str,
                           duration_minutes: int) -> list[dict]:
        """動画チャプター自動生成"""
        prompt = f"""
以下の{duration_minutes}分の動画スクリプトからチャプター（タイムスタンプ）を生成:

スクリプト:
{script[:3000]}

フォーマット:
00:00 イントロ
01:30 セクション名
...

ルール:
- 最初のタイムスタンプは必ず 00:00
- チャプター数は5-10個
- 各チャプター名はキーワードを含む簡潔な表現
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"chapters": response.content[0].text}
```

### 4.4 自動字幕生成と多言語対応

```python
class SubtitleGenerator:
    """字幕自動生成・翻訳"""

    def __init__(self, client):
        self.client = client

    def generate_srt(self, transcript: str,
                      timing_data: list[dict] = None) -> str:
        """SRT形式の字幕ファイルを生成"""
        if timing_data:
            return self._from_timing_data(transcript, timing_data)

        # タイミングデータがない場合はAIで推定
        prompt = f"""
以下のトランスクリプトをSRT字幕形式に変換してください。
各字幕は2行以内、1行35文字以内にしてください。

トランスクリプト:
{transcript[:3000]}

SRTフォーマット:
1
00:00:00,000 --> 00:00:03,000
字幕テキスト

2
00:00:03,500 --> 00:00:07,000
次の字幕テキスト
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def translate_subtitles(self, srt_content: str,
                             target_language: str) -> str:
        """字幕の翻訳"""
        prompt = f"""
以下のSRT字幕を{target_language}に翻訳してください。
タイムスタンプは変更しないでください。
文化的に自然な表現を使い、直訳を避けてください。

{srt_content[:5000]}
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _from_timing_data(self, transcript: str,
                           timing_data: list[dict]) -> str:
        """タイミングデータからSRT生成"""
        srt_lines = []
        for i, segment in enumerate(timing_data, 1):
            start = self._format_time(segment["start"])
            end = self._format_time(segment["end"])
            text = segment.get("text", "")
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(text)
            srt_lines.append("")
        return "\n".join(srt_lines)

    def _format_time(self, seconds: float) -> str:
        """秒数をSRT形式に変換"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
```

---

## 5. メールマーケティング自動化

### 5.1 メルマガ生成エンジン

```python
class NewsletterGenerator:
    """AIメルマガ生成エンジン"""

    def __init__(self, client):
        self.client = client

    def generate_newsletter(self, topic: str,
                              audience: str,
                              sections: list[str] = None,
                              tone: str = "プロフェッショナルかつ親しみやすい") -> dict:
        """メルマガ本文を生成"""
        default_sections = [
            "冒頭の挨拶（パーソナル）",
            "今週のメイントピック",
            "実践Tips（3つ）",
            "おすすめリソース",
            "編集後記",
        ]
        sections = sections or default_sections

        prompt = f"""
以下の要件でメルマガを生成してください。

トピック: {topic}
読者層: {audience}
トーン: {tone}

構成:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(sections))}

要件:
- 件名候補を3つ（開封率を最大化）
- プレヘッダーテキスト（50文字以内）
- 本文はHTML形式でも使える構成
- CTA（行動喚起）を自然に含める
- 配信停止リンクの文言
- 読了時間の目安を冒頭に表示
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"newsletter": response.content[0].text}

    def generate_drip_sequence(self, product: str,
                                 audience: str,
                                 sequence_length: int = 7) -> list[dict]:
        """ドリップメールシーケンスの生成"""
        prompt = f"""
以下の製品/サービスに対するドリップメールシーケンスを
{sequence_length}通分生成してください。

製品: {product}
ターゲット: {audience}

各メールに:
- 件名
- 送信タイミング（登録後X日）
- 目的（教育/信頼構築/セールスなど）
- 本文
- CTA
- 次のメールへの導線

全体の流れ:
1通目: ウェルカム（価値提供）
2通目: 問題の深掘り
3通目: 解決策の提示
4通目: 社会的証明（事例）
5通目: 限定オファー
6通目: Q&A/反論処理
7通目: 最終CTA
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"sequence": response.content[0].text}

    def ab_test_subject_lines(self, topic: str,
                                count: int = 5) -> list[dict]:
        """件名のA/Bテスト候補生成"""
        prompt = f"""
以下のトピックに対するメルマガ件名を{count}パターン生成。
各パターンで異なるアプローチを使用:

トピック: {topic}

アプローチ:
1. 数字を使う（「3つの方法」「7割の人が」）
2. 疑問形（「〜していませんか？」）
3. 緊急性（「見逃すと損する」「今だけ」）
4. パーソナル（「あなたの〜」「[名前]さんへ」）
5. 好奇心（「意外な真実」「知られざる」）

各件名に:
- 件名テキスト
- アプローチ種類
- 想定開封率（高/中/低）
- 使用すべきセグメント
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"subject_lines": response.content[0].text}
```

---

## 6. コンテンツ品質管理

### 6.1 品質チェックエンジン

```python
class ContentQualityChecker:
    """AI生成コンテンツの品質チェック"""

    def __init__(self, client):
        self.client = client

    def full_quality_check(self, content: str,
                            content_type: str = "blog") -> dict:
        """総合品質チェック"""
        checks = {
            "factual_accuracy": self._check_facts(content),
            "brand_consistency": self._check_brand(content),
            "readability": self._check_readability(content),
            "originality": self._check_originality(content),
            "legal_compliance": self._check_legal(content, content_type),
            "seo_quality": self._check_seo(content) if content_type == "blog" else None,
        }

        passed = sum(1 for v in checks.values() if v and v.get("passed", False))
        total = sum(1 for v in checks.values() if v is not None)

        return {
            "overall_pass": passed == total,
            "score": f"{passed}/{total}",
            "checks": checks
        }

    def _check_facts(self, content: str) -> dict:
        """ファクトチェック"""
        prompt = f"""
以下のコンテンツに含まれる事実主張を検証してください。

コンテンツ:
{content[:3000]}

以下を確認:
1. 統計データの正確性
2. 日付や固有名詞の正確性
3. 技術的な記述の正確性
4. 引用元の確認
5. 古い情報がないか

JSON形式で回答:
- claims: 検出された主張のリスト
- issues: 問題のある記述のリスト
- passed: true/false
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"result": response.content[0].text, "passed": True}

    def _check_brand(self, content: str) -> dict:
        """ブランド一貫性チェック"""
        # NG ワードリストによるチェック
        ng_words = ["絶対に", "必ず", "100%", "最高の", "世界一"]
        found_ng = [w for w in ng_words if w in content]
        return {
            "passed": len(found_ng) == 0,
            "ng_words_found": found_ng,
            "recommendation": "誇大表現を避けてください" if found_ng else "OK"
        }

    def _check_readability(self, content: str) -> dict:
        """可読性チェック"""
        sentences = [s.strip() for s in content.split("。") if s.strip()]
        avg_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
        long_sentences = [s for s in sentences if len(s) > 80]

        return {
            "passed": avg_length < 70 and len(long_sentences) < 3,
            "avg_sentence_length": round(avg_length, 1),
            "long_sentences_count": len(long_sentences),
            "total_sentences": len(sentences)
        }

    def _check_originality(self, content: str) -> dict:
        """独自性チェック（簡易版）"""
        # 一般的な定型文パターンの検出
        generic_patterns = [
            "いかがでしたか",
            "最後までお読みいただき",
            "参考になれば幸いです",
            "ぜひ試してみてください",
        ]
        found_generic = [p for p in generic_patterns if p in content]

        return {
            "passed": len(found_generic) <= 1,
            "generic_patterns_found": found_generic,
            "recommendation": "AI臭い定型文を減らしてください" if found_generic else "OK"
        }

    def _check_legal(self, content: str, content_type: str) -> dict:
        """法的コンプライアンスチェック"""
        issues = []

        # 景品表示法チェック
        misleading_terms = [
            "業界No.1", "日本初", "世界初", "最安値",
            "効果抜群", "副作用なし", "返金保証"
        ]
        for term in misleading_terms:
            if term in content:
                issues.append(f"景表法リスク: '{term}'の使用。根拠の明示が必要。")

        # ステルスマーケティング規制
        if content_type in ["blog", "sns_post"]:
            ad_indicators = ["PR", "広告", "提供", "アフィリエイト", "#PR", "#ad"]
            # 広告的内容がある場合は表示が必要
            promotional_words = ["おすすめ", "購入", "お得", "クーポン"]
            has_promotional = any(w in content for w in promotional_words)
            has_disclosure = any(i in content for i in ad_indicators)
            if has_promotional and not has_disclosure:
                issues.append(
                    "ステマ規制リスク: 広告的内容にPR表示がありません。"
                )

        return {
            "passed": len(issues) == 0,
            "issues": issues
        }

    def _check_seo(self, content: str) -> dict:
        """SEO品質チェック"""
        headings = len([l for l in content.split('\n') if l.startswith('#')])
        paragraphs = content.split('\n\n')
        has_lists = any('- ' in p or '* ' in p for p in paragraphs)
        word_count = len(content)

        issues = []
        if headings < 4:
            issues.append("見出しが少なすぎます。")
        if not has_lists:
            issues.append("箇条書きを追加してスキャナビリティを向上させてください。")
        if word_count < 2000:
            issues.append(f"文字数が不足しています（{word_count}文字）。")

        return {
            "passed": len(issues) == 0,
            "headings_count": headings,
            "word_count": word_count,
            "has_lists": has_lists,
            "issues": issues
        }
```

### 6.2 ブランドボイスガード

```python
class BrandVoiceGuard:
    """ブランドボイスの一貫性を保証"""

    def __init__(self, client):
        self.client = client
        self.brand_guidelines = {}

    def set_guidelines(self, tone: str, vocabulary: dict,
                        examples: list[str],
                        forbidden_phrases: list[str]):
        """ブランドガイドラインを設定"""
        self.brand_guidelines = {
            "tone": tone,
            "preferred_words": vocabulary.get("preferred", []),
            "avoid_words": vocabulary.get("avoid", []),
            "examples": examples[:5],
            "forbidden_phrases": forbidden_phrases,
        }

    def validate(self, content: str) -> dict:
        """コンテンツがブランドガイドラインに適合するかチェック"""
        issues = []

        # 禁止フレーズチェック
        for phrase in self.brand_guidelines.get("forbidden_phrases", []):
            if phrase in content:
                issues.append(f"禁止フレーズ検出: '{phrase}'")

        # 回避すべき単語チェック
        for word in self.brand_guidelines.get("avoid_words", []):
            count = content.count(word)
            if count > 0:
                issues.append(f"回避推奨語: '{word}'が{count}回使用されています")

        # AIによるトーンチェック
        prompt = f"""
以下のコンテンツが指定されたブランドトーンに適合しているか評価してください。

ブランドトーン: {self.brand_guidelines.get('tone', '')}
参考例:
{chr(10).join(self.brand_guidelines.get('examples', [])[:3])}

評価対象:
{content[:2000]}

評価基準:
1. トーンの一致度（1-10）
2. 語彙の適切性（1-10）
3. 文体の一貫性（1-10）
4. 具体的な改善提案（箇条書き）
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "ai_evaluation": response.content[0].text
        }

    def rewrite_to_brand(self, content: str) -> str:
        """コンテンツをブランドボイスに合わせてリライト"""
        prompt = f"""
以下のコンテンツをブランドガイドラインに合わせてリライトしてください。

ブランドトーン: {self.brand_guidelines.get('tone', '')}
使うべき語彙: {', '.join(self.brand_guidelines.get('preferred_words', [])[:20])}
避けるべき語彙: {', '.join(self.brand_guidelines.get('avoid_words', [])[:20])}

参考例:
{chr(10).join(self.brand_guidelines.get('examples', [])[:2])}

リライト対象:
{content[:3000]}

内容や主張は変えず、トーンと語彙のみ調整してください。
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
```

---

## 7. コンテンツ分析・KPI管理

### 7.1 KPIダッシュボード

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class ContentKPI:
    """コンテンツマーケティングKPI"""
    period_start: datetime
    period_end: datetime

    # トラフィック
    total_pageviews: int = 0
    unique_visitors: int = 0
    organic_traffic: int = 0
    referral_traffic: int = 0
    social_traffic: int = 0

    # エンゲージメント
    avg_time_on_page: float = 0.0  # 秒
    bounce_rate: float = 0.0  # %
    pages_per_session: float = 0.0
    comments_count: int = 0
    social_shares: int = 0

    # コンバージョン
    email_signups: int = 0
    lead_forms: int = 0
    trial_signups: int = 0
    purchases: int = 0
    conversion_rate: float = 0.0

    # コンテンツ制作
    articles_published: int = 0
    videos_published: int = 0
    sns_posts: int = 0
    newsletters_sent: int = 0

    # コスト
    ai_api_cost: float = 0.0
    tools_cost: float = 0.0
    freelancer_cost: float = 0.0
    total_cost: float = 0.0

    @property
    def cost_per_article(self) -> float:
        if self.articles_published == 0:
            return 0
        return self.total_cost / self.articles_published

    @property
    def cost_per_lead(self) -> float:
        total_leads = self.email_signups + self.lead_forms
        if total_leads == 0:
            return 0
        return self.total_cost / total_leads

    @property
    def roi(self) -> float:
        """コンテンツマーケティングROI"""
        if self.total_cost == 0:
            return 0
        revenue = self.purchases * 10000  # 仮: 1購入=1万円
        return (revenue - self.total_cost) / self.total_cost


class ContentAnalyticsDashboard:
    """コンテンツ分析ダッシュボード"""

    def __init__(self):
        self.kpi_history: list[ContentKPI] = []

    def add_period(self, kpi: ContentKPI):
        self.kpi_history.append(kpi)

    def get_trend(self, metric: str, periods: int = 12) -> list[dict]:
        """メトリクスのトレンドを取得"""
        recent = self.kpi_history[-periods:]
        return [
            {
                "period": f"{kpi.period_start.strftime('%Y-%m')}",
                "value": getattr(kpi, metric, 0)
            }
            for kpi in recent
        ]

    def compare_periods(self, current: ContentKPI,
                         previous: ContentKPI) -> dict:
        """期間比較"""
        metrics = [
            "total_pageviews", "unique_visitors", "organic_traffic",
            "email_signups", "conversion_rate", "social_shares"
        ]
        comparison = {}
        for metric in metrics:
            curr_val = getattr(current, metric, 0)
            prev_val = getattr(previous, metric, 0)
            if prev_val > 0:
                change_pct = (curr_val - prev_val) / prev_val * 100
            else:
                change_pct = 0
            comparison[metric] = {
                "current": curr_val,
                "previous": prev_val,
                "change": curr_val - prev_val,
                "change_pct": f"{change_pct:+.1f}%"
            }
        return comparison

    def get_content_roi_report(self) -> dict:
        """コンテンツROIレポート"""
        if not self.kpi_history:
            return {"error": "No data available"}

        latest = self.kpi_history[-1]
        total_cost = sum(k.total_cost for k in self.kpi_history)
        total_leads = sum(
            k.email_signups + k.lead_forms for k in self.kpi_history
        )
        total_purchases = sum(k.purchases for k in self.kpi_history)

        return {
            "total_investment": f"¥{total_cost:,.0f}",
            "total_leads": total_leads,
            "total_purchases": total_purchases,
            "overall_cpl": f"¥{total_cost / max(total_leads, 1):,.0f}",
            "overall_cpa": f"¥{total_cost / max(total_purchases, 1):,.0f}",
            "latest_roi": f"{latest.roi:.1%}",
            "articles_total": sum(k.articles_published for k in self.kpi_history),
            "avg_cost_per_article": f"¥{sum(k.cost_per_article for k in self.kpi_history) / len(self.kpi_history):,.0f}",
        }
```

### 7.2 コンテンツパフォーマンス予測

```python
class ContentPerformancePredictor:
    """コンテンツパフォーマンス予測"""

    def __init__(self, client):
        self.client = client

    def predict_performance(self, title: str,
                              content_preview: str,
                              historical_data: list[dict]) -> dict:
        """公開前のパフォーマンス予測"""
        avg_views = sum(d.get("views", 0) for d in historical_data) / max(len(historical_data), 1)
        avg_engagement = sum(d.get("engagement_rate", 0) for d in historical_data) / max(len(historical_data), 1)

        prompt = f"""
以下の記事の予想パフォーマンスを分析してください。

タイトル: {title}
内容プレビュー: {content_preview[:1000]}

過去の記事パフォーマンス平均:
- 平均PV: {avg_views:.0f}
- 平均エンゲージメント率: {avg_engagement:.2%}

以下を予測してください:
1. 予想PV範囲（最小-最大）
2. 予想エンゲージメント率
3. SEOポテンシャル（高/中/低）
4. SNSバイラル可能性（高/中/低）
5. 改善提案（タイトル、構成、CTA等）
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"prediction": response.content[0].text}
```

---

## 8. アンチパターン

### アンチパターン1: 大量生成・低品質戦略

```python
# BAD: AIで毎日10記事を無差別に量産
def mass_produce():
    for topic in get_trending_topics(100):
        article = ai.generate(topic)  # ノーチェック
        publish(article)  # 即公開
    # → Googleからスパム判定、信頼性低下

# GOOD: 質を担保した計画的制作
def quality_first():
    topics = research_topics(5)  # 月5記事に厳選
    for topic in topics:
        draft = ai.generate(topic)         # AI生成
        reviewed = human_review(draft)      # 人間編集
        fact_checked = verify_facts(reviewed) # ファクトチェック
        optimized = seo_optimize(fact_checked) # SEO最適化
        schedule_publish(optimized)         # スケジュール公開
```

### アンチパターン2: プラットフォーム無視のコピペ投稿

```python
# BAD: 同じ文章を全SNSにコピペ
def post_everywhere(content):
    for platform in ["twitter", "linkedin", "instagram"]:
        post(platform, content)  # 全部同じ文面

# GOOD: プラットフォーム最適化
def post_optimized(content):
    posts = social_generator.generate_posts(
        content,
        platforms=["twitter", "linkedin", "instagram"]
    )
    for platform, post_content in posts.items():
        schedule_post(
            platform,
            post_content,
            optimal_time=get_best_time(platform)
        )
```

### アンチパターン3: 分析なきコンテンツ制作

```python
# BAD: 効果測定なしの投稿を続ける
def post_blindly():
    for week in range(52):
        content = ai.generate(random_topic())
        publish(content)
        # メトリクスを見ない → 何が効果的か不明

# GOOD: データドリブンなコンテンツ改善
def data_driven_content():
    # 先月のパフォーマンスを分析
    top_content = analytics.get_best_performing(top_n=5)
    worst_content = analytics.get_worst_performing(top_n=5)

    # 成功パターンを抽出
    patterns = analyze_success_patterns(top_content)
    # → 例: 「ハウツー系が強い」「木曜公開がPV高い」

    # パターンに基づいて次月のコンテンツを企画
    next_month_plan = create_plan_from_patterns(patterns)
    execute_plan(next_month_plan)
```

### アンチパターン4: AI依存100%

```python
# BAD: 人間の関与ゼロ
def fully_automated():
    article = ai.generate(topic)
    # ファクトチェックなし
    # ブランドチェックなし
    # 独自知見なし
    publish(article)
    # → 没個性、信頼性低下、法的リスク

# GOOD: AI + 人間のハイブリッド
def hybrid_creation():
    # AIが下書き
    draft = ai.generate(topic, brand_voice=brand_guide)

    # 人間が独自知見を追加
    draft_with_insights = add_original_insights(draft)

    # ファクトチェック
    verified = fact_check(draft_with_insights)

    # ブランドチェック
    brand_approved = brand_voice_guard.validate(verified)

    # 法務チェック
    legal_approved = legal_check(brand_approved)

    # 最終承認後に公開
    if all_checks_passed:
        schedule_publish(legal_approved)
```

### アンチパターン5: パーソナライゼーション無視

```python
# BAD: 全読者に同じメルマガ
def one_size_fits_all():
    newsletter = generate_newsletter(topic)
    send_to_all(newsletter)  # 全員に同じ内容
    # → 開封率低下、配信停止増加

# GOOD: セグメント別パーソナライゼーション
def personalized_content():
    segments = {
        "beginners": {"tone": "やさしい", "depth": "入門"},
        "intermediate": {"tone": "実践的", "depth": "中級"},
        "advanced": {"tone": "テクニカル", "depth": "上級"},
    }
    for segment, config in segments.items():
        newsletter = generate_newsletter(
            topic,
            tone=config["tone"],
            depth=config["depth"]
        )
        send_to_segment(segment, newsletter)
```

---

## 9. FAQ

### Q1: AI生成コンテンツはSEOに不利？

**A:** Googleは「AIで作られたかどうか」ではなく「ユーザーに価値があるか」で判断すると公式に表明（2023年）。ただし (1) 事実誤認を含む記事はペナルティの対象、(2) 大量の低品質記事はスパム判定される、(3) E-E-A-T（経験、専門性、権威性、信頼性）の観点で独自の知見や体験を加えることが重要。AI生成+人間の専門知識という組み合わせが最強。

### Q2: 著作権の問題は？

**A:** 現状の法的見解（2025年時点）。(1) AI生成テキストの著作権は不明確（日本では「創作的寄与」があれば人間に帰属）、(2) 他者の著作物をAIに学習させた場合のリスクあり、(3) 画像生成AIは既存画像との類似性に注意。対策: AI生成物を「素材」として扱い、人間が編集・加工することで著作権を確保する。

### Q3: ブランドの一貫性を保つには？

**A:** 3つの仕組みを構築する。(1) ブランドボイスガイド — トーン、使用語彙、禁止表現を定義しプロンプトに組み込む、(2) テンプレートシステム — 記事構成、SNS投稿、メルマガの型を統一、(3) レビューチェックリスト — AI生成物を公開前にブランドガイドと照合。Custom Instructionsや System Promptを活用すれば80%は自動で一貫性を保てる。

### Q4: コンテンツ制作のコストはどのくらい？

**A:** AI活用時の概算コスト（月間）を以下に示す。ブログ記事（月10本）: AI API費 $30-50 + 人間編集 5-10時間 = 総コスト約5-15万円。SNS投稿（月60本）: AI API費 $10-20 + 承認作業 3-5時間 = 総コスト約3-8万円。動画コンテンツ（月4本）: AI API費 $10-20 + 撮影・編集 20-40時間 = 総コスト約20-50万円。比較: AI未使用の場合は上記の3-5倍のコストが一般的。特にブログ記事は外注で1本3-5万円が相場のため、AI活用で70-80%のコスト削減が可能。

### Q5: ステルスマーケティング規制への対応は？

**A:** 2023年10月施行のステマ規制（景品表示法の指定告示）への対応が必須。(1) 広告主からの依頼で作成したコンテンツには「広告」「PR」「提供」等の表示が必要、(2) アフィリエイトリンクを含む記事も対象、(3) SNS投稿では「#PR」「#広告」等のハッシュタグを付ける、(4) 表示は「一般消費者が認識できる位置・サイズ」で行う。違反した場合は措置命令の対象となり、広告主側に責任が生じる。AI生成コンテンツであっても同様の規制が適用される。

### Q6: どのAIモデルがコンテンツ制作に最適？

**A:** 用途別の推奨モデル。ブログ記事（長文）: Claude（文章品質が高い、日本語が自然）。SNS投稿（短文大量生成）: GPT-4o-mini（コスト効率が良い）。画像生成: DALL-E 3、Midjourney、Stable Diffusion（用途により選択）。動画スクリプト: Claude（構成力が高い）。翻訳: DeepL API（翻訳品質）またはClaude（文脈理解力）。メルマガ: GPT-4o（パーソナライゼーションが得意）。コスト重視の場合はGPT-4o-miniやClaude Haiku、品質重視の場合はGPT-4oやClaude Sonnet/Opusを使い分ける。

### Q7: AIコンテンツの「AI臭さ」を消すには？

**A:** 5つの実践テクニック。(1) 具体的なデータや事例を追加する（「多くの企業が」→「当社の調査では67%の企業が」）、(2) 個人的な経験や見解を混ぜる（「筆者の場合は〜」）、(3) 業界特有の専門用語を自然に使う、(4) 定型文を排除する（「いかがでしたか」「参考になれば幸いです」等を削除）、(5) 文章のリズムに変化をつける（短文と長文を交互に、問いかけを混ぜるなど）。最終的に人間のレビューアが「自分の言葉で書き直す」工程を入れるのが最も効果的。

### Q8: コンテンツカレンダーの運用で失敗しないコツは？

**A:** 4つのポイント。(1) 2週間先までの計画にとどめる（1ヶ月以上先は柔軟に変更できる枠だけ確保）。(2) 70-20-10ルール: 70%は計画通りのコンテンツ、20%はトレンド対応、10%は実験的コンテンツに配分。(3) 「ネタ切れ」を防ぐため、常に30個以上のアイデアストックを維持（Notionやスプレッドシートで管理）。(4) 週次のレトロスペクティブで前週の数値を確認し、翌週の計画を微調整する。完璧な計画を作ることよりも、PDCAサイクルを回すことが重要。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| パイプライン | 企画→生成→編集→配信→分析の5段階 |
| ブログ | AI生成70%+人間編集30%で最高ROI |
| SNS | プラットフォーム別最適化が必須 |
| 動画 | スクリプト生成から字幕まで段階的自動化 |
| メルマガ | ドリップシーケンス+パーソナライゼーション |
| 品質管理 | ファクトチェック + ブランドボイス + 人間レビュー |
| SEO | 量より質、E-E-A-Tの充足が鍵 |
| 分析 | KPIダッシュボードでデータドリブンに改善 |
| 法務 | ステマ規制・景表法・著作権への対応を忘れずに |
| コスト | AI活用で従来の3-5分の1にコスト削減可能 |

---

## 次に読むべきガイド

- [03-ai-marketplace.md](./03-ai-marketplace.md) — AIマーケットプレイス活用
- [../02-monetization/02-scaling-strategy.md](../02-monetization/02-scaling-strategy.md) — スケーリング戦略
- [../03-case-studies/01-solo-developer.md](../03-case-studies/01-solo-developer.md) — 個人開発者の成功事例

---

## 参考文献

1. **Google Search Central: AI-generated content** — https://developers.google.com/search/docs — AI生成コンテンツに対するGoogleの公式見解
2. **"Content Inc." — Joe Pulizzi (2021)** — コンテンツファーストのビジネス構築
3. **Buffer State of Social Media Report (2024)** — SNSマーケティングの最新トレンドとデータ
4. **"AI for Marketers" — Christopher Penn (2024)** — マーケターのためのAI活用実践ガイド
5. **消費者庁: ステルスマーケティング規制** — https://www.caa.go.jp/ — 景品表示法に基づくステマ規制ガイドライン
6. **"They Ask, You Answer" — Marcus Sheridan (2019)** — コンテンツマーケティングの基本戦略
7. **HubSpot: State of Marketing Report (2024)** — マーケティング全般のトレンドとベンチマーク
