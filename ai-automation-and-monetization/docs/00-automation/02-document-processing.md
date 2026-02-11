# ドキュメント処理 — OCR、PDF解析、契約書分析

> AIを活用したドキュメント処理の自動化技術を体系的に解説し、OCR、PDF解析、契約書分析の実装パターンからプロダクション運用まで網羅する。

---

## この章で学ぶこと

1. **OCRとAIの統合アーキテクチャ** — Tesseract、Cloud Vision、GPT-4V を組み合わせた高精度テキスト抽出
2. **PDF解析パイプラインの構築** — 構造化データ抽出、テーブル解析、マルチモーダル処理
3. **契約書AI分析の実践** — リスク検出、条項比較、コンプライアンスチェックの自動化

---

## 1. ドキュメント処理アーキテクチャ

### 1.1 処理パイプライン全体像

```
┌──────────────────────────────────────────────────────────────┐
│              AI ドキュメント処理パイプライン                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  入力               前処理            AI処理          出力    │
│  ┌─────┐      ┌──────────┐      ┌──────────┐    ┌──────┐  │
│  │ PDF │─────▶│画像補正  │─────▶│ OCR      │──▶│構造化│  │
│  │画像 │      │ノイズ除去│      │テキスト抽出   │  │データ│  │
│  │スキャン│    │傾き補正  │      │レイアウト解析│  │JSON │  │
│  └─────┘      └──────────┘      └──────────┘    └──────┘  │
│                     │                  │              │      │
│                     ▼                  ▼              ▼      │
│              ┌──────────┐      ┌──────────┐    ┌──────┐    │
│              │品質検証  │      │LLM分析   │    │DB    │    │
│              │信頼度判定│      │要約・分類 │    │保存  │    │
│              └──────────┘      └──────────┘    └──────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 技術スタック比較

| 技術 | 用途 | 精度 | コスト | 処理速度 |
|------|------|------|--------|---------|
| Tesseract OCR | オンプレOCR | 中 | 無料 | 高速 |
| Google Cloud Vision | クラウドOCR | 高 | $1.50/1000ページ | 高速 |
| AWS Textract | 構造化抽出 | 高 | $1.50/1000ページ | 中 |
| Azure Document Intelligence | テーブル解析 | 高 | $1.00/1000ページ | 中 |
| GPT-4 Vision | マルチモーダル | 最高 | $0.01/画像 | 低速 |
| Claude Vision | マルチモーダル | 最高 | $0.01/画像 | 低速 |

---

## 2. OCR実装

### 2.1 Tesseract + 前処理

```python
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np

class OCRProcessor:
    """高精度OCR処理クラス"""

    def __init__(self, lang: str = "jpn+eng"):
        self.lang = lang
        self.config = "--oem 3 --psm 6"  # LSTMエンジン + ブロック検出

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """画像前処理: ノイズ除去、二値化、傾き補正"""
        img = cv2.imread(image_path)

        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ノイズ除去（ガウシアンブラー）
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # 適応的二値化（照明ムラに対応）
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 傾き補正
        corrected = self._correct_skew(binary)
        return corrected

    def _correct_skew(self, image: np.ndarray) -> np.ndarray:
        """傾き補正"""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def extract_text(self, image_path: str) -> dict:
        """テキスト抽出 + 信頼度情報"""
        processed = self.preprocess_image(image_path)
        data = pytesseract.image_to_data(
            processed, lang=self.lang,
            config=self.config,
            output_type=pytesseract.Output.DICT
        )

        results = []
        for i in range(len(data["text"])):
            if int(data["conf"][i]) > 0:
                results.append({
                    "text": data["text"][i],
                    "confidence": int(data["conf"][i]),
                    "bbox": {
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i]
                    }
                })

        full_text = " ".join(r["text"] for r in results if r["text"].strip())
        avg_confidence = (
            sum(r["confidence"] for r in results) / len(results)
            if results else 0
        )

        return {
            "text": full_text,
            "confidence": avg_confidence,
            "details": results
        }
```

### 2.2 Cloud Vision API連携

```python
from google.cloud import vision
import io

class CloudVisionOCR:
    """Google Cloud Vision OCR"""

    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def extract_text(self, image_path: str) -> dict:
        """Cloud Vision でテキスト抽出"""
        with io.open(image_path, "rb") as f:
            content = f.read()

        image = vision.Image(content=content)
        response = self.client.document_text_detection(image=image)

        if response.error.message:
            raise Exception(f"Vision API Error: {response.error.message}")

        full_text = response.full_text_annotation
        pages = []
        for page in full_text.pages:
            for block in page.blocks:
                block_text = ""
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join(
                            symbol.text for symbol in word.symbols
                        )
                        block_text += word_text
                    block_text += "\n"
                pages.append({
                    "text": block_text,
                    "confidence": block.confidence,
                    "type": block.block_type.name
                })

        return {
            "text": full_text.text,
            "pages": pages,
            "language": full_text.pages[0].property.detected_languages[0].language_code
        }
```

---

## 3. PDF解析

### 3.1 PDF構造解析

```python
import fitz  # PyMuPDF
from dataclasses import dataclass

@dataclass
class PDFPage:
    page_num: int
    text: str
    tables: list
    images: list
    metadata: dict

class PDFAnalyzer:
    """PDF構造解析エンジン"""

    def __init__(self, pdf_path: str):
        self.doc = fitz.open(pdf_path)
        self.metadata = self.doc.metadata

    def extract_all(self) -> list[PDFPage]:
        """全ページの構造化抽出"""
        pages = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            pages.append(PDFPage(
                page_num=page_num,
                text=page.get_text("text"),
                tables=self._extract_tables(page),
                images=self._extract_images(page),
                metadata={
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation
                }
            ))
        return pages

    def _extract_tables(self, page) -> list[list[list[str]]]:
        """テーブル抽出"""
        tables = page.find_tables()
        result = []
        for table in tables:
            rows = []
            for row in table.extract():
                rows.append([cell if cell else "" for cell in row])
            result.append(rows)
        return result

    def _extract_images(self, page) -> list[dict]:
        """画像抽出"""
        images = []
        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            base_image = self.doc.extract_image(xref)
            images.append({
                "index": img_index,
                "size": len(base_image["image"]),
                "format": base_image["ext"],
                "width": base_image.get("width"),
                "height": base_image.get("height")
            })
        return images

    def to_markdown(self) -> str:
        """PDF全体をMarkdown変換"""
        md_parts = []
        for page in self.extract_all():
            md_parts.append(f"## Page {page.page_num + 1}\n")
            md_parts.append(page.text)
            for i, table in enumerate(page.tables):
                md_parts.append(f"\n### Table {i + 1}\n")
                if table:
                    header = "| " + " | ".join(table[0]) + " |"
                    separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
                    md_parts.append(header)
                    md_parts.append(separator)
                    for row in table[1:]:
                        md_parts.append("| " + " | ".join(row) + " |")
        return "\n".join(md_parts)
```

### 3.2 AI連携PDF分析

```
PDF + AI 分析フロー:

  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ PDF     │──▶│ 構造解析  │──▶│ チャンク  │──▶│ LLM      │
  │ Upload  │   │ PyMuPDF  │   │ 分割     │   │ 分析     │
  └─────────┘   └──────────┘   └──────────┘   └────┬─────┘
                                                     │
                     ┌───────────────────────────────┘
                     ▼
              ┌──────────────┐
              │ 結果統合      │
              │ - 要約        │
              │ - キーポイント │
              │ - テーブルデータ│
              │ - アクション   │
              └──────────────┘
```

---

## 4. 契約書AI分析

### 4.1 契約書分析エンジン

```python
import anthropic
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ContractAnalyzer:
    """AI契約書分析エンジン"""

    ANALYSIS_PROMPT = """
あなたは日本の契約法に精通した法務AIアシスタントです。
以下の契約書を分析し、JSON形式で結果を返してください。

分析項目:
1. contract_type: 契約種別
2. parties: 当事者情報
3. key_terms: 重要条項（配列）
4. risks: リスク項目（配列、各項目にlevel付き）
5. missing_clauses: 欠落している一般的条項
6. recommendations: 推奨アクション

契約書本文:
{contract_text}
"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def analyze(self, contract_text: str) -> dict:
        """契約書の包括的分析"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": self.ANALYSIS_PROMPT.format(
                    contract_text=contract_text
                )
            }]
        )
        return self._parse_response(response.content[0].text)

    def compare_contracts(self, contract_a: str, contract_b: str) -> dict:
        """2つの契約書を比較分析"""
        prompt = f"""
2つの契約書を比較し、以下を分析:
1. 共通条項と相違点
2. どちらが契約者に有利か
3. 交渉すべきポイント

契約書A:
{contract_a}

契約書B:
{contract_b}
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_response(response.content[0].text)

    def check_compliance(self, contract_text: str,
                         regulations: list[str]) -> dict:
        """コンプライアンスチェック"""
        regs = "\n".join(f"- {r}" for r in regulations)
        prompt = f"""
以下の契約書が規制に準拠しているか確認:

規制一覧:
{regs}

契約書:
{contract_text}

各規制について: 準拠/非準拠/要確認 を判定し、根拠を説明。
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_response(response.content[0].text)

    def _parse_response(self, text: str) -> dict:
        """レスポンスのJSONパース"""
        import json
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"raw_analysis": text}
```

### 4.2 リスク分類マトリクス

| リスクカテゴリ | 検出対象 | 影響度 | AI精度 |
|--------------|---------|--------|--------|
| 賠償責任 | 無制限賠償条項 | 致命的 | 95% |
| 知的財産 | 権利帰属の曖昧さ | 高 | 90% |
| 解除条件 | 一方的解除権 | 高 | 92% |
| 競業避止 | 過度な制限期間 | 中 | 88% |
| 秘密保持 | 開示範囲の不明確さ | 中 | 85% |
| 支払条件 | 不利な支払条件 | 低〜中 | 90% |

---

## 5. アンチパターン

### アンチパターン1: OCR精度を過信する

```python
# BAD: OCR結果をそのまま使用
def process_invoice(image_path):
    text = ocr.extract_text(image_path)
    amount = extract_amount(text)  # OCRエラーで誤金額の可能性
    charge_customer(amount)  # 即座に課金 — 危険！

# GOOD: 信頼度チェック + 人間レビュー
def process_invoice(image_path):
    result = ocr.extract_text(image_path)  # 信頼度付き

    if result["confidence"] < 85:
        return flag_for_human_review(result)

    amount = extract_amount(result["text"])
    # ダブルチェック: 2つのOCRエンジンで結果を比較
    result2 = cloud_vision.extract_text(image_path)
    amount2 = extract_amount(result2["text"])

    if amount != amount2:
        return flag_for_human_review({"amount1": amount, "amount2": amount2})

    return create_draft_invoice(amount)  # ドラフト作成、承認後に処理
```

### アンチパターン2: 全文をLLMに投げる

```python
# BAD: 100ページのPDFを丸ごとLLMに送信
def analyze_contract(pdf_path):
    full_text = extract_all_text(pdf_path)  # 10万トークン
    result = call_ai(f"分析して: {full_text}")  # コスト高、精度低

# GOOD: 構造化抽出 → 関連部分のみAI分析
def analyze_contract(pdf_path):
    analyzer = PDFAnalyzer(pdf_path)
    pages = analyzer.extract_all()

    # 条項ごとに分割
    clauses = split_into_clauses(pages)

    # リスクの高い条項のみAI分析（コスト1/10）
    risk_clauses = [c for c in clauses
                    if any(kw in c for kw in ["賠償", "解除", "違約"])]

    results = []
    for clause in risk_clauses:
        result = call_ai(f"リスク分析: {clause}")
        results.append(result)

    return results
```

---

## 6. FAQ

### Q1: 日本語OCRの精度を上げるには？

**A:** 3つの対策が有効。(1) 前処理の徹底 — 二値化、ノイズ除去、傾き補正で10-20%精度向上、(2) 日本語特化モデル — Tesseractの`jpn_vert`（縦書き対応）や Google Cloud Vision（日本語精度95%+）を使う、(3) 後処理 — 辞書照合、文脈によるスペルチェック、LLMでの誤字修正。特に手書き文字は Cloud Vision + GPT-4V の組み合わせが最も高精度。

### Q2: 契約書AI分析は法的に有効か？

**A:** AI分析はあくまで「補助ツール」であり、法的判断の代替にはならない。ただし (1) 見落とし防止 — 人間のレビューアが見逃しがちな条項をAIが検出、(2) 初期スクリーニング — 大量の契約書から要注意案件を抽出、(3) 比較分析 — 過去の契約との差分検出。最終判断は必ず弁護士が行うワークフローにすべき。

### Q3: 大量のPDF処理のスケーリング方法は？

**A:** 3段階で対応する。(1) バッチ処理 — Celery/SQSでキュー管理し非同期処理、(2) 並列化 — PDFのページ単位で並列OCR処理（10倍速）、(3) キャッシュ — 同一ドキュメントのハッシュで結果をキャッシュ。月10万ページ規模なら AWS Lambda + SQS + DynamoDB の構成が費用対効果最良。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| OCR選定 | 無料: Tesseract、高精度: Cloud Vision、マルチモーダル: GPT-4V |
| PDF解析 | PyMuPDF で構造抽出 → LLM で意味解析 の2段階 |
| 契約書分析 | 条項分割 → リスク条項抽出 → AI分析 → 人間レビュー |
| 精度管理 | 信頼度スコア + ダブルチェック + Human-in-the-Loop |
| コスト最適化 | 関連部分のみAI分析、キャッシュ、バッチ処理 |
| 法的留意 | AIは補助ツール、最終判断は専門家 |

---

## 次に読むべきガイド

- [03-email-communication.md](./03-email-communication.md) — メール・コミュニケーション自動化
- [../01-business/00-ai-saas.md](../01-business/00-ai-saas.md) — ドキュメント処理SaaSの構築
- [../02-monetization/01-cost-management.md](../02-monetization/01-cost-management.md) — API費用の最適化

---

## 参考文献

1. **"Document AI" — Google Cloud Documentation** — https://cloud.google.com/document-ai — 構造化ドキュメント処理のクラウドサービス
2. **PyMuPDF Documentation** — https://pymupdf.readthedocs.io — PDF操作ライブラリの公式ガイド
3. **Tesseract OCR Documentation** — https://github.com/tesseract-ocr/tesseract — オープンソースOCRエンジン
4. **"AI-Powered Contract Analysis" — Stanford Law Review (2024)** — AI契約分析の法的考察と精度評価
