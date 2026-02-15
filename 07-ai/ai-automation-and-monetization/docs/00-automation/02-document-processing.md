# ドキュメント処理 — OCR、PDF解析、契約書分析

> AIを活用したドキュメント処理の自動化技術を体系的に解説し、OCR、PDF解析、契約書分析の実装パターンからプロダクション運用まで網羅する。

---

## この章で学ぶこと

1. **OCRとAIの統合アーキテクチャ** — Tesseract、Cloud Vision、GPT-4V を組み合わせた高精度テキスト抽出
2. **PDF解析パイプラインの構築** — 構造化データ抽出、テーブル解析、マルチモーダル処理
3. **契約書AI分析の実践** — リスク検出、条項比較、コンプライアンスチェックの自動化
4. **プロダクション運用の設計** — スケーリング、エラーハンドリング、監視体制の構築
5. **業界別ドキュメント処理パターン** — 請求書、医療文書、不動産、金融の実例

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

| 技術 | 用途 | 精度 | コスト | 処理速度 | 日本語対応 |
|------|------|------|--------|---------|-----------|
| Tesseract OCR | オンプレOCR | 中 | 無料 | 高速 | 縦書き/横書き対応 |
| Google Cloud Vision | クラウドOCR | 高 | $1.50/1000ページ | 高速 | 95%+精度 |
| AWS Textract | 構造化抽出 | 高 | $1.50/1000ページ | 中 | フォーム/テーブル特化 |
| Azure Document Intelligence | テーブル解析 | 高 | $1.00/1000ページ | 中 | プリビルトモデル充実 |
| GPT-4 Vision | マルチモーダル | 最高 | $0.01/画像 | 低速 | 手書き文字にも強い |
| Claude Vision | マルチモーダル | 最高 | $0.01/画像 | 低速 | 文脈理解に優れる |

### 1.3 技術選定フローチャート

```
ドキュメント処理の技術選定:

Q1: ドキュメントの種類は？
├─ 印刷テキスト（活字）───▶ Q2へ
├─ 手書きテキスト ────────▶ Cloud Vision or GPT-4V
├─ フォーム/表 ──────────▶ AWS Textract or Azure DI
└─ 混在（写真+テキスト）──▶ マルチモーダルLLM

Q2: 処理量は？
├─ 月1,000ページ未満 ────▶ Cloud Vision（従量課金）
├─ 月10,000ページ程度 ───▶ Tesseract + Cloud Vision ハイブリッド
└─ 月100,000ページ以上 ──▶ Tesseract主体 + AI後処理

Q3: 精度要件は？
├─ 99%+必要（金融/医療）─▶ ダブルOCR + Human-in-the-Loop
├─ 95%+で十分 ──────────▶ Cloud Vision 単体
└─ 90%+で十分 ──────────▶ Tesseract + 後処理
```

### 1.4 コスト試算モデル

```python
from dataclasses import dataclass

@dataclass
class CostEstimate:
    """ドキュメント処理コスト試算"""
    pages_per_month: int
    avg_pages_per_doc: int = 5

    def tesseract_only(self) -> dict:
        """Tesseractのみ: サーバーコストのみ"""
        # t3.medium インスタンス: 1000ページ/時間
        hours_needed = self.pages_per_month / 1000
        server_cost = hours_needed * 0.0464  # t3.medium 料金
        return {
            "ocr_cost": 0,
            "server_cost": round(server_cost, 2),
            "total": round(server_cost, 2),
            "accuracy": "85-90%"
        }

    def cloud_vision(self) -> dict:
        """Google Cloud Vision"""
        ocr_cost = (self.pages_per_month / 1000) * 1.50
        return {
            "ocr_cost": round(ocr_cost, 2),
            "server_cost": 0,
            "total": round(ocr_cost, 2),
            "accuracy": "95-97%"
        }

    def hybrid_with_ai(self) -> dict:
        """Tesseract + Cloud Vision(低信頼度のみ) + LLM分析"""
        # 80%はTesseractで処理、20%をCloud Visionで再処理
        tesseract_pages = int(self.pages_per_month * 0.8)
        cloud_pages = int(self.pages_per_month * 0.2)
        cloud_cost = (cloud_pages / 1000) * 1.50

        # LLM分析: 文書単位で分析（5ページ/文書平均）
        docs = self.pages_per_month / self.avg_pages_per_doc
        # Claude: 約$0.003/文書（要約+分類）
        llm_cost = docs * 0.003

        hours_needed = tesseract_pages / 1000
        server_cost = hours_needed * 0.0464

        total = cloud_cost + llm_cost + server_cost
        return {
            "ocr_cost": round(cloud_cost, 2),
            "llm_cost": round(llm_cost, 2),
            "server_cost": round(server_cost, 2),
            "total": round(total, 2),
            "accuracy": "96-99%"
        }

    def report(self) -> str:
        """コスト比較レポート生成"""
        t = self.tesseract_only()
        cv = self.cloud_vision()
        h = self.hybrid_with_ai()
        return f"""
月間 {self.pages_per_month:,} ページの処理コスト比較:
─────────────────────────────────────────
方式              月額コスト   精度
─────────────────────────────────────────
Tesseract only    ${t['total']:>8}   {t['accuracy']}
Cloud Vision      ${cv['total']:>8}   {cv['accuracy']}
ハイブリッド+AI    ${h['total']:>8}   {h['accuracy']}
─────────────────────────────────────────
"""

# 使用例
estimate = CostEstimate(pages_per_month=50000)
print(estimate.report())
```

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

### 2.3 マルチモーダルLLMによるOCR

```python
import anthropic
import base64

class MultimodalOCR:
    """Claude Vision / GPT-4V によるマルチモーダルOCR"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def extract_with_context(self, image_path: str,
                              document_type: str = "general") -> dict:
        """画像からコンテキストを理解してテキスト抽出"""
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # ドキュメントタイプ別のプロンプト
        prompts = {
            "general": "この画像のテキストを正確に文字起こししてください。",
            "invoice": """この請求書の画像から以下の情報をJSON形式で抽出してください:
{
  "invoice_number": "請求書番号",
  "date": "発行日",
  "due_date": "支払期限",
  "vendor": {"name": "会社名", "address": "住所"},
  "items": [{"description": "品名", "quantity": 数量, "unit_price": 単価, "amount": 金額}],
  "subtotal": 小計,
  "tax": 消費税,
  "total": 合計金額,
  "bank_account": "振込先"
}""",
            "receipt": """このレシート画像から以下を抽出してJSON形式で返してください:
{
  "store_name": "店名",
  "date": "日付",
  "items": [{"name": "商品名", "price": 価格}],
  "total": 合計,
  "payment_method": "支払方法"
}""",
            "business_card": """この名刺画像から以下を抽出してJSON形式で返してください:
{
  "name": "氏名",
  "name_reading": "フリガナ",
  "company": "会社名",
  "department": "部署",
  "title": "役職",
  "phone": "電話番号",
  "mobile": "携帯番号",
  "email": "メールアドレス",
  "address": "住所",
  "url": "URL"
}""",
            "handwritten": """この手書き文書の画像を正確に文字起こししてください。
判読困難な箇所は [不明] と記載し、推測できる場合は (推測: xxx) と併記してください。
"""
        }

        prompt = prompts.get(document_type, prompts["general"])

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        return {
            "text": response.content[0].text,
            "document_type": document_type,
            "model": "claude-sonnet-4-20250514",
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }

    def batch_extract(self, image_paths: list[str],
                       document_type: str = "general") -> list[dict]:
        """複数画像のバッチOCR処理"""
        results = []
        for path in image_paths:
            try:
                result = self.extract_with_context(path, document_type)
                result["file_path"] = path
                result["status"] = "success"
                results.append(result)
            except Exception as e:
                results.append({
                    "file_path": path,
                    "status": "error",
                    "error": str(e)
                })
        return results
```

### 2.4 ダブルOCRによる精度向上

```python
from difflib import SequenceMatcher

class DoubleCheckOCR:
    """2つのOCRエンジンで相互検証し精度向上"""

    def __init__(self):
        self.primary = OCRProcessor(lang="jpn+eng")
        self.secondary = CloudVisionOCR()
        self.confidence_threshold = 85

    def extract_with_verification(self, image_path: str) -> dict:
        """ダブルOCRで精度検証"""
        # Phase 1: Tesseract で抽出
        primary_result = self.primary.extract_text(image_path)

        # 高信頼度ならそのまま返す（コスト節約）
        if primary_result["confidence"] >= 95:
            return {
                "text": primary_result["text"],
                "confidence": primary_result["confidence"],
                "method": "tesseract_only",
                "verified": False
            }

        # Phase 2: Cloud Vision で二重チェック
        secondary_result = self.secondary.extract_text(image_path)

        # テキスト一致度を計算
        similarity = SequenceMatcher(
            None,
            primary_result["text"],
            secondary_result["text"]
        ).ratio()

        if similarity >= 0.95:
            # 高一致: Cloud Vision の結果を採用（通常より高精度）
            return {
                "text": secondary_result["text"],
                "confidence": min(99, primary_result["confidence"] + 10),
                "method": "double_check_agreed",
                "similarity": similarity,
                "verified": True
            }
        else:
            # 不一致: マージして人間レビューをフラグ
            merged = self._merge_results(primary_result, secondary_result)
            return {
                "text": merged,
                "confidence": min(
                    primary_result["confidence"],
                    secondary_result.get("confidence", 80)
                ) * 0.8,
                "method": "double_check_diverged",
                "similarity": similarity,
                "verified": False,
                "needs_review": True,
                "primary_text": primary_result["text"],
                "secondary_text": secondary_result["text"]
            }

    def _merge_results(self, primary: dict, secondary: dict) -> str:
        """2つのOCR結果をインテリジェントにマージ"""
        # 信頼度が高い方を基本テキストとする
        if primary.get("confidence", 0) > 80:
            return primary["text"]
        return secondary.get("text", primary["text"])
```

### 2.5 日本語OCR特有の課題と対策

日本語ドキュメントをOCR処理する際に特有の問題とその解決策を整理する。

```python
class JapaneseOCROptimizer:
    """日本語OCR最適化クラス"""

    def __init__(self):
        self.vertical_config = "--oem 3 --psm 5"  # 縦書き用PSM
        self.horizontal_config = "--oem 3 --psm 6"  # 横書き用PSM

    def detect_text_orientation(self, image: np.ndarray) -> str:
        """テキストの縦横方向を自動検出"""
        # Tesseract OSD（Orientation and Script Detection）
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotation = osd.get("rotate", 0)
        script = osd.get("script", "")

        # 日本語の縦書き判定ヒューリスティック
        if script == "Japanese" and rotation in [90, 270]:
            return "vertical"
        return "horizontal"

    def optimize_for_japanese(self, image_path: str) -> dict:
        """日本語ドキュメント向け最適化パイプライン"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 1: 方向検出
        orientation = self.detect_text_orientation(gray)

        # Step 2: 方向に応じた前処理
        if orientation == "vertical":
            # 縦書き: 90度回転して横書きとして処理後、結果を復元
            rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            config = self.horizontal_config
            processed = rotated
        else:
            config = self.horizontal_config
            processed = gray

        # Step 3: コントラスト強調（日本語の細い線に効果的）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(processed)

        # Step 4: 適応的二値化（ふりがな付き文書対応）
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 4
        )

        # Step 5: OCR実行
        text = pytesseract.image_to_string(
            binary, lang="jpn+eng", config=config
        )

        # Step 6: 日本語後処理
        cleaned = self._postprocess_japanese(text)

        return {
            "text": cleaned,
            "orientation": orientation,
            "preprocessing": "japanese_optimized"
        }

    def _postprocess_japanese(self, text: str) -> str:
        """日本語OCR結果の後処理"""
        import re

        # よくある誤認識の修正
        corrections = {
            "0": "〇",  # 数字のゼロと漢数字の〇
            "l": "1",   # 小文字のLと数字の1
            "rn": "m",  # rnとmの混同
        }

        # 全角半角の統一
        result = text
        # 数字は半角に統一
        result = re.sub(r'[０-９]', lambda m: chr(ord(m.group()) - 0xFEE0), result)

        # 不要な空白の除去（日本語の文字間）
        result = re.sub(r'(?<=[\u3000-\u9FFF])\s+(?=[\u3000-\u9FFF])', '', result)

        # 改行の正規化
        result = re.sub(r'\n{3,}', '\n\n', result)

        return result.strip()
```

**日本語OCRの主な課題と解決策一覧:**

| 課題 | 原因 | 解決策 | 効果 |
|------|------|--------|------|
| 縦書きの認識失敗 | PSM設定の不適切 | 方向自動検出 + PSM切替 | 精度+20% |
| ふりがなの混入 | 小文字の誤認識 | 領域分割でふりがなを除外 | 精度+15% |
| 旧字体の未認識 | 訓練データ不足 | カスタム辞書 + LLM後処理 | 精度+10% |
| 全角半角の混在 | 標準的な誤認識 | 正規化後処理 | データ品質向上 |
| 表中の日本語 | セル境界の誤検出 | テーブル検出 + セル単位OCR | 精度+25% |
| 手書き日本語 | 個人差が大きい | Cloud Vision + GPT-4V | 精度80-90% |

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

### 3.3 インテリジェントPDFチャンキング

大規模PDFを効率的にAI分析するためのチャンキング戦略は重要である。単純なページ分割ではなく、論理的な構造を保持したチャンキングにより精度が大幅に向上する。

```python
import re
from typing import Optional

class IntelligentPDFChunker:
    """論理構造を保持したPDFチャンキング"""

    def __init__(self, max_chunk_tokens: int = 3000):
        self.max_chunk_tokens = max_chunk_tokens

    def chunk_by_structure(self, pages: list[PDFPage]) -> list[dict]:
        """構造ベースのチャンキング"""
        all_text = "\n".join(p.text for p in pages)

        # 見出しパターンで分割
        sections = self._split_by_headings(all_text)

        chunks = []
        current_chunk = ""
        current_heading = ""

        for section in sections:
            heading = section["heading"]
            content = section["content"]
            estimated_tokens = len(content) // 3  # 日本語の概算

            if estimated_tokens > self.max_chunk_tokens:
                # 大きなセクションはさらに段落で分割
                paragraphs = content.split("\n\n")
                for para in paragraphs:
                    if len(current_chunk) // 3 + len(para) // 3 > self.max_chunk_tokens:
                        if current_chunk:
                            chunks.append({
                                "heading": current_heading,
                                "content": current_chunk.strip(),
                                "estimated_tokens": len(current_chunk) // 3
                            })
                        current_chunk = para
                        current_heading = heading
                    else:
                        current_chunk += "\n\n" + para
            else:
                if len(current_chunk) // 3 + estimated_tokens > self.max_chunk_tokens:
                    if current_chunk:
                        chunks.append({
                            "heading": current_heading,
                            "content": current_chunk.strip(),
                            "estimated_tokens": len(current_chunk) // 3
                        })
                    current_chunk = content
                    current_heading = heading
                else:
                    current_chunk += "\n\n" + content
                    if not current_heading:
                        current_heading = heading

        if current_chunk:
            chunks.append({
                "heading": current_heading,
                "content": current_chunk.strip(),
                "estimated_tokens": len(current_chunk) // 3
            })

        return chunks

    def _split_by_headings(self, text: str) -> list[dict]:
        """見出しパターンでテキストを分割"""
        # 日本語文書の見出しパターン
        heading_patterns = [
            r'^第[一二三四五六七八九十\d]+[条章節項]',  # 法律文書
            r'^\d+[\.\)]\s',                           # 番号付き見出し
            r'^[（(]\d+[）)]',                          # 括弧番号
            r'^■|^●|^◆|^▶',                           # 記号見出し
        ]

        combined_pattern = "|".join(f"({p})" for p in heading_patterns)
        sections = []
        current = {"heading": "", "content": ""}

        for line in text.split("\n"):
            if re.match(combined_pattern, line.strip()):
                if current["content"]:
                    sections.append(current)
                current = {"heading": line.strip(), "content": ""}
            else:
                current["content"] += line + "\n"

        if current["content"]:
            sections.append(current)

        return sections

    def chunk_with_overlap(self, text: str,
                            chunk_size: int = 2000,
                            overlap: int = 200) -> list[dict]:
        """オーバーラップ付きチャンキング（コンテキスト保持用）"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # 文の途中で切らない
            if end < text_len:
                # 日本語の文末パターンで区切り位置を調整
                for delimiter in ["。\n", "。", "\n\n", "\n"]:
                    last_delim = text[start:end].rfind(delimiter)
                    if last_delim > chunk_size * 0.7:  # 70%以上の位置
                        end = start + last_delim + len(delimiter)
                        break

            chunk_text = text[start:end]
            chunks.append({
                "content": chunk_text,
                "start_pos": start,
                "end_pos": end,
                "index": len(chunks)
            })

            start = end - overlap  # オーバーラップ分戻る

        return chunks
```

### 3.4 PDF → 構造化データ変換

```python
import anthropic
import json

class PDFToStructuredData:
    """PDFから構造化データへの変換エンジン"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.pdf_analyzer = None  # 遅延初期化

    def extract_invoice_data(self, pdf_path: str) -> dict:
        """請求書PDFから構造化データを抽出"""
        self.pdf_analyzer = PDFAnalyzer(pdf_path)
        pages = self.pdf_analyzer.extract_all()

        # テーブルデータがあればテーブル優先
        tables = []
        for page in pages:
            tables.extend(page.tables)

        full_text = "\n".join(p.text for p in pages)

        prompt = f"""以下の請求書テキストから情報を抽出し、正確なJSON形式で返してください。

{{
  "invoice_number": "請求書番号",
  "issue_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "vendor": {{
    "name": "会社名",
    "address": "住所",
    "registration_number": "登録番号（インボイス制度）"
  }},
  "buyer": {{
    "name": "宛先会社名",
    "address": "住所"
  }},
  "items": [
    {{
      "description": "品名・サービス名",
      "quantity": 数量,
      "unit": "単位",
      "unit_price": 単価,
      "tax_rate": 税率（0.08 or 0.10）,
      "amount": 金額
    }}
  ],
  "subtotal": 小計,
  "tax_8_percent": 8%対象税額,
  "tax_10_percent": 10%対象税額,
  "total": 合計金額,
  "payment_method": "振込先情報",
  "notes": "備考"
}}

テーブルデータ: {json.dumps(tables, ensure_ascii=False) if tables else "なし"}

請求書テキスト:
{full_text}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_json(response.content[0].text)

    def extract_resume_data(self, pdf_path: str) -> dict:
        """履歴書/職務経歴書PDFから構造化データを抽出"""
        self.pdf_analyzer = PDFAnalyzer(pdf_path)
        pages = self.pdf_analyzer.extract_all()
        full_text = "\n".join(p.text for p in pages)

        prompt = f"""以下の履歴書/職務経歴書から情報を抽出し、JSON形式で返してください。

{{
  "personal": {{
    "name": "氏名",
    "name_reading": "フリガナ",
    "birth_date": "生年月日",
    "gender": "性別",
    "address": "住所",
    "phone": "電話番号",
    "email": "メール"
  }},
  "education": [
    {{
      "period": "期間",
      "institution": "学校名",
      "degree": "学位/専攻",
      "status": "卒業/在学中"
    }}
  ],
  "work_experience": [
    {{
      "period": "期間",
      "company": "会社名",
      "position": "役職",
      "responsibilities": "業務内容",
      "achievements": "実績"
    }}
  ],
  "skills": ["スキルリスト"],
  "certifications": [
    {{
      "name": "資格名",
      "date": "取得日"
    }}
  ],
  "self_pr": "自己PR"
}}

テキスト:
{full_text}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_json(response.content[0].text)

    def _parse_json(self, text: str) -> dict:
        """レスポンスからJSONをパース"""
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"raw_text": text, "parse_error": True}
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
| 準拠法 | 不利な裁判管轄 | 中 | 93% |
| 自動更新 | 不利な更新条件 | 低〜中 | 91% |
| 反社排除 | 条項の欠落 | 高 | 94% |

### 4.3 契約書条項自動抽出

```python
class ClauseExtractor:
    """契約書の条項を自動分類・抽出"""

    CLAUSE_TYPES = [
        "定義条項", "契約期間", "報酬・対価",
        "成果物・納品", "知的財産権", "秘密保持",
        "損害賠償", "契約解除", "競業避止",
        "反社会的勢力排除", "準拠法・管轄",
        "通知条項", "不可抗力", "権利義務譲渡禁止"
    ]

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def extract_clauses(self, contract_text: str) -> dict:
        """契約書から各条項を構造化抽出"""
        clause_types_str = "\n".join(
            f"- {ct}" for ct in self.CLAUSE_TYPES
        )

        prompt = f"""以下の契約書から各条項を抽出し、JSON形式で返してください。

対象条項:
{clause_types_str}

出力形式:
{{
  "clauses": [
    {{
      "type": "条項タイプ",
      "article_number": "第X条",
      "title": "条項タイトル",
      "content": "条項全文",
      "key_points": ["重要ポイント"],
      "risk_level": "low/medium/high/critical",
      "risk_reason": "リスク理由（あれば）"
    }}
  ],
  "missing_clauses": ["欠落条項タイプ"],
  "overall_risk": "low/medium/high/critical",
  "summary": "契約書全体の概要（3文以内）"
}}

契約書:
{contract_text}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_json(response.content[0].text)

    def generate_review_report(self, contract_text: str,
                                reviewer_perspective: str = "受注者") -> str:
        """レビューレポートを生成"""
        prompt = f"""以下の契約書を「{reviewer_perspective}」の立場からレビューし、
以下の形式でレポートを作成してください。

# 契約書レビューレポート

## 1. 概要
- 契約種別、当事者、期間

## 2. リスク評価サマリー
- 全体リスクレベルと主要リスク3点

## 3. 条項別レビュー
各条項について:
- 内容要約
- リスク評価（★1-5）
- コメント・推奨修正案

## 4. 欠落条項
一般的に含まれるべきだが欠落している条項

## 5. 交渉ポイント
{reviewer_perspective}として交渉すべき点（優先度順）

## 6. 推奨アクション
具体的な次のステップ

契約書:
{contract_text}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _parse_json(self, text: str) -> dict:
        import json
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"raw_text": text}
```

### 4.4 契約書バージョン比較

```python
from difflib import unified_diff, SequenceMatcher

class ContractVersionComparator:
    """契約書のバージョン間差分を検出し、法的影響を分析"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def compare_versions(self, old_text: str, new_text: str) -> dict:
        """2バージョンの契約書を比較"""
        # Step 1: テキスト差分を検出
        diff = self._compute_diff(old_text, new_text)

        # Step 2: 変更箇所をAIで法的分析
        analysis = self._analyze_changes(diff, old_text, new_text)

        return {
            "diff": diff,
            "analysis": analysis,
            "change_count": len(diff["additions"]) + len(diff["deletions"]),
            "risk_assessment": analysis.get("overall_risk", "unknown")
        }

    def _compute_diff(self, old_text: str, new_text: str) -> dict:
        """差分計算"""
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        diff_lines = list(unified_diff(
            old_lines, new_lines,
            fromfile="旧版", tofile="新版",
            lineterm=""
        ))

        additions = [l[1:] for l in diff_lines if l.startswith("+") and not l.startswith("+++")]
        deletions = [l[1:] for l in diff_lines if l.startswith("-") and not l.startswith("---")]

        similarity = SequenceMatcher(None, old_text, new_text).ratio()

        return {
            "additions": additions,
            "deletions": deletions,
            "similarity": round(similarity, 4),
            "raw_diff": "".join(diff_lines[:100])  # 最初の100行
        }

    def _analyze_changes(self, diff: dict, old_text: str, new_text: str) -> dict:
        """差分の法的影響をAIで分析"""
        changes_summary = []
        for i, (add, rem) in enumerate(
            zip(diff["additions"][:20], diff["deletions"][:20])
        ):
            changes_summary.append(f"変更{i+1}: '{rem.strip()}' → '{add.strip()}'")

        changes_text = "\n".join(changes_summary) if changes_summary else "差分なし"

        prompt = f"""以下の契約書の変更点について法的影響を分析してください。

変更箇所:
{changes_text}

類似度: {diff['similarity']:.1%}

分析観点:
1. 各変更の法的影響（有利/不利/中立）
2. 注意すべきリスクの変化
3. 変更の意図の推測
4. 全体的なリスク評価

JSON形式で返してください:
{{
  "changes": [
    {{
      "description": "変更内容",
      "impact": "有利/不利/中立",
      "risk_change": "増加/減少/変化なし",
      "comment": "コメント"
    }}
  ],
  "overall_risk": "low/medium/high/critical",
  "recommendation": "推奨アクション"
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return {"raw_analysis": response.content[0].text}
```

---

## 5. プロダクション運用

### 5.1 バッチ処理パイプライン

大量のドキュメントを効率的に処理するためのプロダクション向けバッチ処理パイプラインの設計を解説する。

```python
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"

@dataclass
class ProcessingJob:
    job_id: str
    file_path: str
    document_type: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: dict = field(default_factory=dict)
    error: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime = None
    retry_count: int = 0
    file_hash: str = ""

class DocumentProcessingPipeline:
    """プロダクション向けドキュメント処理パイプライン"""

    def __init__(self, config: dict):
        self.config = config
        self.max_retries = config.get("max_retries", 3)
        self.concurrent_limit = config.get("concurrent_limit", 10)
        self.cache = {}  # 本番では Redis 等を使用
        self.jobs: dict[str, ProcessingJob] = {}

    async def submit_job(self, file_path: str,
                          document_type: str = "auto") -> str:
        """処理ジョブを投入"""
        # ファイルハッシュでキャッシュ確認
        file_hash = self._compute_hash(file_path)
        if file_hash in self.cache:
            logger.info(f"Cache hit: {file_path}")
            return self.cache[file_hash]

        job_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file_hash[:8]}"
        job = ProcessingJob(
            job_id=job_id,
            file_path=file_path,
            document_type=document_type,
            file_hash=file_hash
        )
        self.jobs[job_id] = job

        # 非同期処理キューに投入
        asyncio.create_task(self._process_job(job))
        return job_id

    async def _process_job(self, job: ProcessingJob):
        """ジョブ処理（リトライ付き）"""
        job.status = ProcessingStatus.PROCESSING
        logger.info(f"Processing: {job.job_id}")

        for attempt in range(self.max_retries):
            try:
                # ドキュメントタイプ自動判定
                if job.document_type == "auto":
                    job.document_type = await self._detect_type(job.file_path)

                # 処理実行
                result = await self._execute_processing(job)

                # 品質チェック
                quality = self._quality_check(result)
                if quality["passed"]:
                    job.result = result
                    job.status = ProcessingStatus.COMPLETED
                    job.completed_at = datetime.now()
                    self.cache[job.file_hash] = result
                    logger.info(f"Completed: {job.job_id}")
                else:
                    job.result = result
                    job.status = ProcessingStatus.NEEDS_REVIEW
                    job.result["quality_issues"] = quality["issues"]
                    logger.warning(f"Needs review: {job.job_id}")
                return

            except Exception as e:
                job.retry_count = attempt + 1
                job.error = str(e)
                logger.error(f"Attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数バックオフ

        job.status = ProcessingStatus.FAILED
        logger.error(f"Failed after {self.max_retries} attempts: {job.job_id}")

    async def _execute_processing(self, job: ProcessingJob) -> dict:
        """ドキュメントタイプ別の処理実行"""
        processors = {
            "invoice": self._process_invoice,
            "contract": self._process_contract,
            "resume": self._process_resume,
            "receipt": self._process_receipt,
            "general": self._process_general,
        }
        processor = processors.get(job.document_type, self._process_general)
        return await processor(job.file_path)

    async def _detect_type(self, file_path: str) -> str:
        """ドキュメントタイプを自動判定"""
        # 最初のページのテキストから判定
        analyzer = PDFAnalyzer(file_path)
        pages = analyzer.extract_all()
        if not pages:
            return "general"

        first_page_text = pages[0].text[:500]

        # キーワードベースの簡易判定
        type_keywords = {
            "invoice": ["請求書", "御請求", "Invoice", "合計金額", "振込先"],
            "contract": ["契約書", "甲", "乙", "条項", "本契約"],
            "resume": ["履歴書", "職務経歴", "学歴", "職歴"],
            "receipt": ["領収書", "レシート", "Receipt"],
        }

        for doc_type, keywords in type_keywords.items():
            if any(kw in first_page_text for kw in keywords):
                return doc_type

        return "general"

    def _quality_check(self, result: dict) -> dict:
        """処理結果の品質チェック"""
        issues = []

        # OCR信頼度チェック
        if result.get("confidence", 100) < 80:
            issues.append("OCR信頼度が低い")

        # 必須フィールドの存在チェック
        if result.get("document_type") == "invoice":
            required = ["invoice_number", "total", "vendor"]
            for field in required:
                if not result.get(field):
                    issues.append(f"必須フィールド欠落: {field}")

        # テキスト長チェック（極端に短い場合は問題あり）
        text_len = len(result.get("text", ""))
        if text_len < 50:
            issues.append("抽出テキストが極端に短い")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "score": max(0, 100 - len(issues) * 20)
        }

    def _compute_hash(self, file_path: str) -> str:
        """ファイルハッシュ計算"""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def _process_invoice(self, file_path: str) -> dict:
        """請求書処理"""
        converter = PDFToStructuredData(self.config["api_key"])
        return converter.extract_invoice_data(file_path)

    async def _process_contract(self, file_path: str) -> dict:
        """契約書処理"""
        analyzer = ContractAnalyzer(self.config["api_key"])
        pdf = PDFAnalyzer(file_path)
        text = "\n".join(p.text for p in pdf.extract_all())
        return analyzer.analyze(text)

    async def _process_resume(self, file_path: str) -> dict:
        """履歴書処理"""
        converter = PDFToStructuredData(self.config["api_key"])
        return converter.extract_resume_data(file_path)

    async def _process_receipt(self, file_path: str) -> dict:
        """レシート処理"""
        ocr = MultimodalOCR(self.config["api_key"])
        return ocr.extract_with_context(file_path, "receipt")

    async def _process_general(self, file_path: str) -> dict:
        """汎用処理"""
        pdf = PDFAnalyzer(file_path)
        pages = pdf.extract_all()
        return {
            "text": "\n".join(p.text for p in pages),
            "page_count": len(pages),
            "tables": [t for p in pages for t in p.tables],
            "document_type": "general"
        }

    def get_status(self, job_id: str) -> dict:
        """ジョブステータス取得"""
        job = self.jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "document_type": job.document_type,
            "retry_count": job.retry_count,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }
```

### 5.2 監視・メトリクスシステム

```python
from collections import defaultdict
from datetime import datetime, timedelta

class DocumentProcessingMetrics:
    """ドキュメント処理の監視メトリクス"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []

    def record(self, event_type: str, data: dict):
        """メトリクス記録"""
        entry = {
            "timestamp": datetime.now(),
            "type": event_type,
            **data
        }
        self.metrics[event_type].append(entry)

        # アラートチェック
        self._check_alerts(event_type, data)

    def _check_alerts(self, event_type: str, data: dict):
        """アラート条件チェック"""
        if event_type == "ocr_result" and data.get("confidence", 100) < 70:
            self.alerts.append({
                "level": "warning",
                "message": f"OCR信頼度低下: {data.get('confidence')}%",
                "timestamp": datetime.now()
            })

        if event_type == "processing_error":
            # 直近1時間のエラー率チェック
            recent_errors = [
                m for m in self.metrics["processing_error"]
                if m["timestamp"] > datetime.now() - timedelta(hours=1)
            ]
            if len(recent_errors) > 10:
                self.alerts.append({
                    "level": "critical",
                    "message": f"1時間のエラー数: {len(recent_errors)}件",
                    "timestamp": datetime.now()
                })

    def generate_dashboard(self) -> dict:
        """ダッシュボードデータ生成"""
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # 今日の処理統計
        today_jobs = [
            m for m in self.metrics.get("job_complete", [])
            if m["timestamp"] >= today
        ]

        total_pages = sum(m.get("pages", 0) for m in today_jobs)
        avg_time = (
            sum(m.get("processing_time_ms", 0) for m in today_jobs) / len(today_jobs)
            if today_jobs else 0
        )
        avg_confidence = (
            sum(m.get("confidence", 0) for m in today_jobs) / len(today_jobs)
            if today_jobs else 0
        )

        # コスト計算
        api_costs = sum(m.get("api_cost", 0) for m in today_jobs)

        return {
            "date": today.strftime("%Y-%m-%d"),
            "total_documents": len(today_jobs),
            "total_pages": total_pages,
            "avg_processing_time_ms": round(avg_time),
            "avg_confidence": round(avg_confidence, 1),
            "api_cost_usd": round(api_costs, 2),
            "cost_per_page": round(api_costs / total_pages, 4) if total_pages else 0,
            "error_count": len([
                m for m in self.metrics.get("processing_error", [])
                if m["timestamp"] >= today
            ]),
            "pending_reviews": len([
                m for m in self.metrics.get("needs_review", [])
                if m["timestamp"] >= today
            ]),
            "active_alerts": [
                a for a in self.alerts
                if a["timestamp"] > now - timedelta(hours=24)
            ]
        }
```

### 5.3 エラーハンドリング戦略

```python
class DocumentProcessingError(Exception):
    """ドキュメント処理エラーの基底クラス"""
    def __init__(self, message: str, error_code: str, recoverable: bool = True):
        super().__init__(message)
        self.error_code = error_code
        self.recoverable = recoverable

class OCRError(DocumentProcessingError):
    pass

class PDFParseError(DocumentProcessingError):
    pass

class AIAnalysisError(DocumentProcessingError):
    pass

class ErrorHandler:
    """ドキュメント処理のエラーハンドリング"""

    ERROR_STRATEGIES = {
        "ocr_low_confidence": {
            "action": "retry_with_alternative",
            "fallback": "cloud_vision",
            "max_retries": 2
        },
        "pdf_corrupted": {
            "action": "repair_and_retry",
            "fallback": "image_conversion",
            "max_retries": 1
        },
        "api_rate_limit": {
            "action": "exponential_backoff",
            "initial_wait": 1,
            "max_wait": 60,
            "max_retries": 5
        },
        "api_timeout": {
            "action": "retry_with_smaller_chunk",
            "chunk_reduction": 0.5,
            "max_retries": 3
        },
        "parse_error": {
            "action": "manual_review",
            "notify": True,
            "max_retries": 0
        }
    }

    def handle(self, error: DocumentProcessingError,
               context: dict) -> dict:
        """エラーに応じた復旧戦略を実行"""
        strategy = self.ERROR_STRATEGIES.get(
            error.error_code,
            {"action": "manual_review", "max_retries": 0}
        )

        logger.error(
            f"Error [{error.error_code}]: {error} | "
            f"Strategy: {strategy['action']} | "
            f"Recoverable: {error.recoverable}"
        )

        return {
            "error_code": error.error_code,
            "strategy": strategy,
            "recoverable": error.recoverable,
            "context": context
        }
```

---

## 6. 業界別ドキュメント処理パターン

### 6.1 請求書処理（経理部門向け）

```
請求書自動処理フロー:

  メール受信 ──▶ 添付PDF抽出 ──▶ OCR処理 ──▶ データ抽出
                                                    │
      ┌────────────────────────────────────────────┘
      ▼
  インボイス番号照合 ──▶ 仕訳自動生成 ──▶ 承認ワークフロー
      │                       │                  │
      ▼                       ▼                  ▼
  重複チェック            会計ソフト連携      Slack通知
  金額検証               freee / MFクラウド    承認リクエスト
```

**請求書処理の精度向上テクニック:**

| テクニック | 説明 | 精度向上 |
|-----------|------|---------|
| テンプレートマッチング | 取引先ごとの請求書レイアウトを学習 | +15% |
| 金額クロスチェック | 明細合計 = 小計の検証 | エラー検出率99% |
| 過去データ照合 | 過去の請求書と金額レンジを比較 | 異常値検出95% |
| インボイス番号検証 | T+13桁の適格請求書番号を国税庁DBで検証 | 100% |

### 6.2 医療文書処理

```python
class MedicalDocumentProcessor:
    """医療文書専用の処理エンジン"""

    # 医療特有の個人情報マスキングルール
    PII_PATTERNS = {
        "patient_id": r'\d{8,10}',
        "insurance_number": r'[0-9]{8}',
        "phone": r'0\d{2,3}-?\d{2,4}-?\d{4}',
        "name": None,  # NERモデルで検出
    }

    def process_prescription(self, image_path: str) -> dict:
        """処方箋の処理"""
        # 医療文書は特に高精度が求められる
        # ダブルOCR + 薬品名辞書照合
        ocr = DoubleCheckOCR()
        result = ocr.extract_with_verification(image_path)

        # 薬品名の正規化（医薬品マスター照合）
        medications = self._extract_medications(result["text"])
        verified_meds = self._verify_with_drug_master(medications)

        return {
            "text": result["text"],
            "medications": verified_meds,
            "confidence": result["confidence"],
            "warnings": self._check_interactions(verified_meds)
        }

    def _extract_medications(self, text: str) -> list:
        """処方箋テキストから薬品情報を抽出"""
        # 実装: LLMで薬品名、用量、用法を抽出
        pass

    def _verify_with_drug_master(self, medications: list) -> list:
        """医薬品マスターDBで薬品名を検証"""
        # 実装: 薬品名の正規化と存在確認
        pass

    def _check_interactions(self, medications: list) -> list:
        """薬物相互作用チェック"""
        # 実装: 複数薬品の併用リスクを検出
        pass
```

### 6.3 不動産文書処理

```
不動産ドキュメント処理パイプライン:

  物件情報PDF ──▶ テキスト抽出 ──▶ 構造化データ変換
       │                                    │
       ▼                                    ▼
  重要事項説明書 ──▶ 条項分析 ──▶ リスク検出
       │                              │
       ▼                              ▼
  登記簿謄本 ──▶ 権利関係抽出 ──▶ 総合レポート
```

**不動産文書で抽出すべきフィールド:**

| 文書種類 | 抽出フィールド | 精度要件 |
|---------|--------------|---------|
| 重要事項説明書 | 物件情報、法令制限、インフラ | 99%+ |
| 売買契約書 | 価格、引渡条件、違約条項 | 99%+ |
| 登記簿謄本 | 所有者、抵当権、地目 | 100%（人間検証必須） |
| 賃貸借契約書 | 賃料、更新条件、禁止事項 | 95%+ |
| 建物検査報告 | 劣化箇所、修繕見積 | 90%+ |

---

## 7. アンチパターン

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

### アンチパターン3: キャッシュなしで同一文書を再処理

```python
# BAD: 同じ文書を毎回処理
def process_document(file_path):
    return ocr.extract_text(file_path)  # 毎回APIコール

# GOOD: ファイルハッシュでキャッシュ
import hashlib
import json

class CachedProcessor:
    def __init__(self, cache_dir: str = "/tmp/doc_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def process(self, file_path: str) -> dict:
        file_hash = self._hash_file(file_path)
        cache_path = f"{self.cache_dir}/{file_hash}.json"

        # キャッシュ確認
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        # 新規処理
        result = ocr.extract_text(file_path)

        # キャッシュ保存
        with open(cache_path, "w") as f:
            json.dump(result, f, ensure_ascii=False)

        return result

    def _hash_file(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
```

---

## 8. FAQ

### Q1: 日本語OCRの精度を上げるには？

**A:** 3つの対策が有効。(1) 前処理の徹底 — 二値化、ノイズ除去、傾き補正で10-20%精度向上、(2) 日本語特化モデル — Tesseractの`jpn_vert`（縦書き対応）や Google Cloud Vision（日本語精度95%+）を使う、(3) 後処理 — 辞書照合、文脈によるスペルチェック、LLMでの誤字修正。特に手書き文字は Cloud Vision + GPT-4V の組み合わせが最も高精度。

### Q2: 契約書AI分析は法的に有効か？

**A:** AI分析はあくまで「補助ツール」であり、法的判断の代替にはならない。ただし (1) 見落とし防止 — 人間のレビューアが見逃しがちな条項をAIが検出、(2) 初期スクリーニング — 大量の契約書から要注意案件を抽出、(3) 比較分析 — 過去の契約との差分検出。最終判断は必ず弁護士が行うワークフローにすべき。

### Q3: 大量のPDF処理のスケーリング方法は？

**A:** 3段階で対応する。(1) バッチ処理 — Celery/SQSでキュー管理し非同期処理、(2) 並列化 — PDFのページ単位で並列OCR処理（10倍速）、(3) キャッシュ — 同一ドキュメントのハッシュで結果をキャッシュ。月10万ページ規模なら AWS Lambda + SQS + DynamoDB の構成が費用対効果最良。

### Q4: マルチモーダルLLMとOCR専用エンジンはどう使い分ける？

**A:** 基本方針は「OCR専用エンジンをメイン、LLMを補助」とする。(1) OCR専用エンジン（Tesseract / Cloud Vision）はコストが低く高速で、大量の定型文書に適している。(2) マルチモーダルLLM（GPT-4V / Claude Vision）は手書き文字や複雑なレイアウトに強いが、1画像あたり$0.01-0.03のコストがかかる。(3) 推奨構成: OCR専用エンジンで基本処理し、信頼度が低い場合のみLLMでフォールバック。これでコストを80%削減しつつ、精度は最高水準を維持できる。

### Q5: 個人情報を含む文書のAI処理で注意すべき点は？

**A:** 5つの対策が必須。(1) PIIマスキング — AI送信前に個人情報（氏名、住所、電話番号等）を自動マスク。NERモデルまたは正規表現で検出。(2) データ保持ポリシー — API利用規約を確認し、データが学習に使用されないプランを選択（Anthropic APIはデフォルトで不使用）。(3) オンプレミス処理 — 医療情報や金融情報など高機密データはセルフホストモデル（Llama等）で処理。(4) 暗号化 — 処理中のデータは常にTLS暗号化、保存データはAES-256暗号化。(5) アクセスログ — 誰がどの文書にアクセスしたかを完全に記録し、監査に備える。

### Q6: 既存の文書管理システムとの統合方法は？

**A:** 3つの統合パターンがある。(1) Webhook連携 — SharePoint、Google Drive、Box等のファイルアップロードイベントをトリガーに自動処理を起動。最も簡単に始められる。(2) API統合 — 文書管理システムのAPIを使って直接やり取り。処理結果をメタデータとして保存できる。(3) ファイルシステム監視 — 特定フォルダを監視し、新規ファイルを自動処理。オンプレミス環境やレガシーシステムとの統合に有効。いずれの場合も、処理結果を元の文書にメタデータとして紐付けることで、後からの検索・活用が容易になる。

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
| プロダクション | リトライ、監視、アラート、エラーハンドリングの完備 |
| 業界対応 | 請求書、医療、不動産など業界別パイプライン設計 |

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
5. **AWS Textract Developer Guide** — https://docs.aws.amazon.com/textract/ — AWSの構造化テキスト抽出サービス
6. **Azure AI Document Intelligence** — https://learn.microsoft.com/azure/ai-services/document-intelligence/ — Azureのドキュメント解析サービス
