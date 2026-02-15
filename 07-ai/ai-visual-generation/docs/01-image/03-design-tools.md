# デザインツール -- Canva AI、Adobe Firefly、Figma AI

> AI機能を統合した主要デザインプラットフォームの特徴と活用法を、ワークフロー視点で比較し、非デザイナーでもプロフェッショナルな成果物を効率的に制作するための実践手法を解説する

## この章で学ぶこと

1. **各ツールのAI機能** -- Canva Magic Studio、Adobe Firefly、Figma AI の中核機能と得意領域
2. **ワークフローへの統合** -- 企画→デザイン→フィードバック→納品の各段階でのAI活用法
3. **プロンプト設計とカスタマイズ** -- 意図通りの出力を得るためのプロンプト技法とブランド統一手法

---

## 1. 各ツールのAI機能

### 1.1 機能マップ

```
デザインツール AI機能マップ

  Canva Magic Studio          Adobe Firefly              Figma AI
  +------------------+       +------------------+       +------------------+
  | Magic Design     |       | Text to Image    |       | Auto Layout AI   |
  | (テンプレ自動生成)|       | (テキスト画像生成) |       | (レイアウト提案)  |
  +------------------+       +------------------+       +------------------+
  | Magic Edit       |       | Generative Fill  |       | Component Suggest|
  | (AI画像編集)     |       | (生成塗りつぶし)  |       | (コンポーネント提案)|
  +------------------+       +------------------+       +------------------+
  | Magic Write      |       | Text Effects     |       | Content Reel     |
  | (AI文章生成)     |       | (テキスト装飾)    |       | (コンテンツ一括)  |
  +------------------+       +------------------+       +------------------+
  | Magic Eraser     |       | Generative Expand|       | Variable Suggest |
  | (不要物除去)     |       | (画像拡張)       |       | (変数自動設定)    |
  +------------------+       +------------------+       +------------------+
  | Background Remover|       | Structure Ref   |       | Prototype AI     |
  | (背景除去)       |       | (構図参照生成)    |       | (プロトタイプ支援)|
  +------------------+       +------------------+       +------------------+

  対象ユーザー:
  Canva    = 非デザイナー、マーケター、中小企業
  Firefly  = プロデザイナー、フォトグラファー
  Figma AI = UI/UXデザイナー、エンジニア
```

### 1.2 ワークフロー全体像

```
デザインワークフローにおけるAI活用

  1. 企画           2. デザイン         3. レビュー        4. 納品
  +----------+     +----------+       +----------+      +----------+
  | AI で     |     | AI で    |       | AI で    |      | AI で    |
  | ブレスト  | --> | 初稿生成 | --->  | バリエー | ---> | リサイズ |
  | アイデア  |     | テンプレ |       | ション   |      | フォーマ |
  | 出し      |     | 選択     |       | 生成     |      | ット変換 |
  +----------+     +----------+       +----------+      +----------+
  |Canva:     |    |Firefly:   |     |Canva:     |    |Canva:     |
  | Magic     |    | Text to   |     | Magic     |    | Resize    |
  | Write     |    | Image     |     | Design    |    | & Magic   |
  |           |    |Figma AI:  |     |           |    | Switch    |
  |           |    | Auto      |     |           |    |           |
  |           |    | Layout    |     |           |    |           |
  +-----------+    +-----------+     +-----------+    +-----------+
```

### 1.3 技術進化のタイムライン

```
AIデザインツール進化史

2020 ─── Canva: 基本的な背景除去機能を追加
         Adobe: Neural Filters を Photoshop に統合（ベータ）
         Figma: Smart Selection 導入

2021 ─── Canva: Magic Resize リリース
         Adobe: Neural Filters 正式版、Super Resolution
         Figma: Auto Layout v3、Interactive Components

2022 ─── Canva: Magic Write（AI文章生成）、Text to Image
         Adobe: Firefly プロジェクト発表
         Figma: Component Properties、Variable Modes
         Microsoft: Designer（DALL-E統合）リリース

2023 ─── Canva: Magic Studio（統合AIスイート）リリース
         Adobe: Firefly 正式版、Photoshop に Generative Fill
         Figma: AI機能プレビュー、First Draft
         Google: Gemini を Workspace に統合

2024 ─── Canva: Dream Lab（高品質画像生成）、Magic Expand
         Adobe: Firefly 3（Generative Match、Structure Reference）
         Figma: AI機能拡充、Dev Mode 改善
         Penpot: オープンソースにAI機能追加

2025 ─── Canva: Magic Design v3（マルチページ対応）
         Adobe: Firefly 4（ビデオ生成対応）
         Figma: AI Prototyping、Design System Intelligence
         各ツール: マルチモーダルAI統合が標準化
```

### 1.4 AI デザインツールのアーキテクチャ

```
AI デザインツールの内部構造

┌─────────────────────────────────────────────────┐
│                 ユーザーインターフェース              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ テンプレ  │  │ エディタ  │  │プレビュー │      │
│  │ ブラウザ  │  │  キャンバス│  │  パネル  │      │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘      │
│        └────────────┼────────────┘              │
│                     ▼                            │
│  ┌──────────────────────────────────────────┐   │
│  │        AI オーケストレーションレイヤー       │   │
│  │  ┌────────┐ ┌────────┐ ┌────────────┐   │   │
│  │  │プロンプ │ │コンテキ│ │ブランド     │   │   │
│  │  │ト解析  │ │スト理解│ │ガイドライン │   │   │
│  │  │エンジン │ │エンジン│ │エンジン     │   │   │
│  │  └───┬────┘ └───┬────┘ └──────┬─────┘   │   │
│  │      └──────────┼───────────┘           │   │
│  │                 ▼                        │   │
│  │  ┌──────────────────────────────────┐   │   │
│  │  │     AIモデルディスパッチャー        │   │   │
│  │  │  画像生成 | 編集 | テキスト | 提案 │   │   │
│  │  └──────────────────────────────────┘   │   │
│  └──────────────────────────────────────────┘   │
│                     ▼                            │
│  ┌──────────────────────────────────────────┐   │
│  │           バックエンド AI サービス          │   │
│  │  Diffusion Models | LLM | Vision Models  │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 1.5 料金体系の詳細比較

| 項目 | Canva Free | Canva Pro | Adobe CC | Figma Starter | Figma Pro |
|------|-----------|-----------|----------|--------------|-----------|
| 月額 | ¥0 | ¥1,500 | ¥7,780〜 | $0 | $15/人 |
| AI画像生成 | 50回/月 | 500回/月 | 250クレジット/月 | -- | -- |
| AI編集 | 制限あり | 無制限 | 含む | 基本機能のみ | 全機能 |
| ストレージ | 5GB | 1TB | 100GB〜 | 無制限 | 無制限 |
| ブランドキット | 1個 | 100個 | -- | -- | チーム共有 |
| API アクセス | -- | 対応 | 対応 | -- | 対応 |
| チーム機能 | -- | 5人〜 | 含む | 2人 | 無制限 |
| 商用利用 | 制限あり | 全て可 | 全て可 | 可 | 可 |
| IP補償 | -- | -- | あり | -- | -- |

---

## 2. Canva Magic Studio の実践

### 2.1 Magic Design でのテンプレート生成

```
プロンプト例: SNS投稿の自動生成

  入力:
    目的: Instagram 投稿
    トーン: モダン、ミニマル
    テーマ: 新商品発売のお知らせ
    ブランドカラー: #2563EB (青)
    テキスト: 「新コレクション登場 | 2026年春」

  Magic Design が生成:
  +----------------------------------+
  |  [AI生成画像: 商品イメージ]       |
  |                                  |
  |      新コレクション登場            |
  |      2026年春                    |
  |                                  |
  |  #NewCollection #Spring2026      |
  +----------------------------------+
  → 5-10バリエーションから選択 → カスタマイズ
```

### 2.2 Magic Edit の活用

```
Magic Edit ワークフロー

  元画像                  指示              結果
  +------------+         +-----------+    +------------+
  | [商品写真]  |  --->  | 背景を     |    | [商品写真]  |
  | (白背景)   |         | カフェの   | -> | (カフェ背景)|
  |            |         | テーブルに |    |            |
  +------------+         +-----------+    +------------+

  プロンプト例:
  - 「背景を暖かい色調のカフェのテーブルに変更」
  - 「商品の横にコーヒーカップを追加」
  - 「全体をフィルム写真風のトーンに変更」
```

### 2.3 Bulk Create（一括作成）

```python
# Canva API でバリエーションを一括生成 (擬似コード)
import canva_api

template_id = "DAF-xxxxx"  # テンプレートID

# CSV データから一括生成
products = [
    {"name": "プロダクトA", "price": "¥3,980", "image_url": "https://..."},
    {"name": "プロダクトB", "price": "¥5,480", "image_url": "https://..."},
    {"name": "プロダクトC", "price": "¥2,980", "image_url": "https://..."},
]

for product in products:
    design = canva_api.create_from_template(
        template_id=template_id,
        data={
            "product_name": product["name"],
            "price_text": product["price"],
            "product_image": product["image_url"],
        }
    )
    design.export(format="png", quality="high")
    print(f"生成完了: {product['name']}")
```

### 2.4 Brand Kit の統合管理

```python
# Canva Brand Kit の自動設定とバリデーション
class CanvaBrandManager:
    """ブランド一貫性を維持するための管理クラス"""

    def __init__(self, api_key: str, brand_id: str):
        self.client = canva_api.Client(api_key=api_key)
        self.brand_id = brand_id
        self.brand_kit = None

    def load_brand_kit(self) -> dict:
        """ブランドキットを読み込み"""
        self.brand_kit = self.client.get_brand_kit(self.brand_id)
        return {
            "colors": {
                "primary": self.brand_kit.primary_color,
                "secondary": self.brand_kit.secondary_colors,
                "accent": self.brand_kit.accent_color,
            },
            "fonts": {
                "heading": self.brand_kit.heading_font,
                "body": self.brand_kit.body_font,
                "caption": self.brand_kit.caption_font,
            },
            "logos": [logo.url for logo in self.brand_kit.logos],
            "guidelines": self.brand_kit.guidelines_text,
        }

    def validate_design(self, design_id: str) -> dict:
        """デザインがブランドガイドラインに適合しているか検証"""
        design = self.client.get_design(design_id)
        issues = []

        # カラーチェック
        used_colors = design.get_used_colors()
        allowed_colors = set(self.brand_kit.all_colors)
        unauthorized = used_colors - allowed_colors
        if unauthorized:
            issues.append({
                "type": "color_violation",
                "severity": "warning",
                "detail": f"未承認カラー使用: {unauthorized}",
                "suggestion": f"許可カラー: {allowed_colors}"
            })

        # フォントチェック
        used_fonts = design.get_used_fonts()
        allowed_fonts = set(self.brand_kit.all_fonts)
        unauthorized_fonts = used_fonts - allowed_fonts
        if unauthorized_fonts:
            issues.append({
                "type": "font_violation",
                "severity": "error",
                "detail": f"未承認フォント使用: {unauthorized_fonts}",
                "suggestion": f"許可フォント: {allowed_fonts}"
            })

        # ロゴ配置チェック
        logo_placements = design.get_logo_placements()
        for placement in logo_placements:
            if placement.clear_space < self.brand_kit.min_clear_space:
                issues.append({
                    "type": "logo_clearspace",
                    "severity": "error",
                    "detail": f"ロゴの余白不足: {placement.clear_space}px",
                    "suggestion": f"最小余白: {self.brand_kit.min_clear_space}px"
                })

        return {
            "is_compliant": len([i for i in issues if i["severity"] == "error"]) == 0,
            "issues": issues,
            "score": max(0, 100 - len(issues) * 15),
        }

    def batch_generate_with_brand(
        self,
        template_id: str,
        data_list: list[dict],
        output_formats: list[str] = ["png"],
        sizes: list[str] = None,
    ) -> list[dict]:
        """ブランドキット適用済みの一括生成"""
        results = []
        for data in data_list:
            design = self.client.create_from_template(
                template_id=template_id,
                data=data,
                brand_kit_id=self.brand_id,  # ブランドキット自動適用
            )

            # バリデーション
            validation = self.validate_design(design.id)
            if not validation["is_compliant"]:
                # 自動修正を試行
                design = self._auto_fix_brand_issues(design, validation["issues"])

            # 複数サイズで書き出し
            exports = []
            target_sizes = sizes or ["instagram_post", "facebook_post", "twitter_post"]
            for size in target_sizes:
                for fmt in output_formats:
                    export = design.export(
                        format=fmt,
                        size_preset=size,
                        quality="high"
                    )
                    exports.append({
                        "size": size,
                        "format": fmt,
                        "url": export.url,
                    })

            results.append({
                "design_id": design.id,
                "data": data,
                "validation": validation,
                "exports": exports,
            })

        return results

    def _auto_fix_brand_issues(self, design, issues):
        """ブランド違反の自動修正"""
        for issue in issues:
            if issue["type"] == "color_violation":
                # 最も近いブランドカラーに置換
                design.replace_colors_to_nearest_brand()
            elif issue["type"] == "font_violation":
                # ブランドフォントに置換
                design.replace_fonts_to_brand()
        return design


# 使用例
brand_mgr = CanvaBrandManager(
    api_key="your-api-key",
    brand_id="brand-123"
)

brand_mgr.load_brand_kit()

# SNS投稿の一括生成
products_data = [
    {"title": "春の新作コレクション", "subtitle": "2026 Spring", "image": "spring.jpg"},
    {"title": "サマーセール開催中", "subtitle": "最大50%OFF", "image": "summer.jpg"},
    {"title": "秋の限定アイテム", "subtitle": "数量限定", "image": "autumn.jpg"},
]

results = brand_mgr.batch_generate_with_brand(
    template_id="template-456",
    data_list=products_data,
    output_formats=["png", "jpg"],
    sizes=["instagram_post", "instagram_story", "facebook_cover"],
)

for r in results:
    print(f"デザイン {r['design_id']}: スコア={r['validation']['score']}")
    for export in r["exports"]:
        print(f"  {export['size']} ({export['format']}): {export['url']}")
```

### 2.5 Magic Write を活用したコピーライティング

```
Magic Write の効果的なプロンプト構成

┌────────────────────────────────────────────────────┐
│ レベル1: 基本プロンプト                              │
│ 「商品紹介のキャッチコピーを書いて」                   │
│ → 汎用的で個性のないコピーが生成される                  │
├────────────────────────────────────────────────────┤
│ レベル2: コンテキスト付きプロンプト                    │
│ 「20代女性向けの春物ワンピース（¥8,980）の            │
│   Instagram投稿用キャッチコピーを3案書いて」           │
│ → ターゲットに合ったコピーが生成される                  │
├────────────────────────────────────────────────────┤
│ レベル3: ブランドボイス指定プロンプト                  │
│ 「ブランドボイス: 上品だが親しみやすい                  │
│   トーン: 明るく前向き                                │
│   NG表現: 『激安』『爆安』などの安売り表現              │
│   必須要素: 素材の良さ、着心地の良さ                   │
│   CTA: ECサイトへの誘導                               │
│   文字数: 80-120字                                    │
│   ハッシュタグ: 5個」                                 │
│ → ブランドに一致した高品質なコピーが生成される           │
└────────────────────────────────────────────────────┘
```

---

## 3. Adobe Firefly の実践

### 3.1 Text to Image の効果的なプロンプト

```
プロンプト構成テンプレート:

  [被写体] + [スタイル] + [構図] + [照明] + [色調] + [品質修飾子]

  例:
  "minimalist product photography of a ceramic coffee mug,
   on a marble surface, soft natural window light,
   warm neutral tones, studio quality, 4K"

  Structure Reference (構図参照):
  - 参照画像をアップロード → 同じ構図で別の内容を生成
  - ブランド一貫性の維持に有効

  Style Reference (スタイル参照):
  - 参照画像の色調・雰囲気を新しい画像に適用
  - ブランドのビジュアルアイデンティティ統一
```

### 3.2 Generative Fill (生成塗りつぶし)

```
Photoshop + Firefly ワークフロー

  Step 1: 選択範囲を作成
  +------------------+
  | [人物写真]        |
  | [---選択---]     |  ← 背景部分を選択
  +------------------+

  Step 2: プロンプト入力
  「東京の夜景、ネオンライト、雨上がりの反射」

  Step 3: AI生成
  +------------------+
  | [人物写真]        |
  | [東京夜景背景]    |  ← 自然な合成
  +------------------+

  活用例:
  - 商品画像の背景変更
  - 被写体の拡張（画像の端を生成で広げる）
  - 不要なオブジェクトの除去と背景生成
```

### 3.3 Firefly API 連携

```python
# Adobe Firefly API (擬似コード)
import adobe_firefly

client = adobe_firefly.Client(api_key="your-api-key")

# Text to Image
result = client.generate_image(
    prompt="modern office workspace, clean desk, laptop, indoor plant, "
           "natural light from large window, minimalist style",
    style="photo",
    aspect_ratio="16:9",
    content_class="photo",        # photo / art
    visual_intensity=4,            # 1-10
    negative_prompt="cluttered, dark, messy",
    num_variations=4,
)

for i, image in enumerate(result.images):
    image.save(f"workspace_v{i+1}.png")

# Generative Fill
result = client.generative_fill(
    image_path="product.jpg",
    mask_path="mask.png",          # 白=生成領域、黒=保持領域
    prompt="wooden table surface with soft shadows",
)
result.image.save("product_on_wood.jpg")
```

### 3.4 Firefly と Creative Cloud の統合ワークフロー

```python
# Adobe Creative Cloud 統合ワークフロー
class AdobeCreativeWorkflow:
    """Firefly + Photoshop + Illustrator の統合パイプライン"""

    def __init__(self, client_id: str, client_secret: str):
        self.auth = self._authenticate(client_id, client_secret)
        self.firefly = FireflyClient(self.auth)
        self.photoshop = PhotoshopAPIClient(self.auth)
        self.illustrator = IllustratorAPIClient(self.auth)

    def _authenticate(self, client_id, client_secret):
        """Adobe IMS OAuth認証"""
        import requests
        response = requests.post(
            "https://ims-na1.adobelogin.com/ims/token/v3",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "openid,creative_sdk,firefly_api",
            }
        )
        return response.json()["access_token"]

    def generate_product_hero(
        self,
        product_image_path: str,
        scene_prompt: str,
        output_sizes: dict = None,
    ) -> dict:
        """商品ヒーロー画像の生成パイプライン

        1. 商品画像の背景除去
        2. Firefly でシーン生成
        3. Photoshop で合成・レタッチ
        4. 複数サイズで書き出し
        """
        # Step 1: 背景除去
        cutout = self.photoshop.remove_background(
            input_path=product_image_path,
            output_format="png",
            refine_edge=True,      # エッジの精密処理
            edge_feather=1.5,      # 自然なフェザリング
        )

        # Step 2: シーン背景を Firefly で生成
        background = self.firefly.generate_image(
            prompt=scene_prompt,
            style="photo",
            aspect_ratio="16:9",
            content_class="photo",
            visual_intensity=5,
            num_variations=4,       # 4パターン生成
        )

        # Step 3: Photoshop API で合成
        composites = []
        for i, bg in enumerate(background.images):
            composite = self.photoshop.composite_images(
                layers=[
                    {"type": "background", "image": bg.url},
                    {
                        "type": "foreground",
                        "image": cutout.url,
                        "position": "center",
                        "scale": 0.6,
                        "shadow": {
                            "type": "drop",
                            "opacity": 30,
                            "angle": 135,
                            "distance": 15,
                            "blur": 25,
                        },
                    },
                ],
                adjustments=[
                    {"type": "color_match", "reference": "background"},
                    {"type": "lighting_match", "intensity": 0.7},
                ],
            )
            composites.append(composite)

        # Step 4: 複数サイズで書き出し
        sizes = output_sizes or {
            "hero_desktop": {"width": 1920, "height": 1080},
            "hero_mobile": {"width": 750, "height": 1334},
            "thumbnail": {"width": 400, "height": 400},
            "og_image": {"width": 1200, "height": 630},
        }

        final_outputs = {}
        best_composite = composites[0]  # または手動選択

        for name, size in sizes.items():
            output = self.photoshop.resize_and_crop(
                image=best_composite.url,
                width=size["width"],
                height=size["height"],
                crop_mode="smart",   # AI による最適クロップ
                format="jpg",
                quality=92,
            )
            final_outputs[name] = output.url

        return {
            "variations": [c.url for c in composites],
            "final_outputs": final_outputs,
        }

    def batch_style_transfer(
        self,
        source_images: list[str],
        style_reference: str,
        strength: float = 0.7,
    ) -> list[str]:
        """Style Reference を使ったバッチスタイル統一

        ブランドの写真すべてを同じトーンに統一する
        """
        results = []
        for img_path in source_images:
            result = self.firefly.style_transfer(
                source_image=img_path,
                style_reference=style_reference,
                strength=strength,
                preserve_structure=True,    # 構図を保持
                preserve_color_range=0.3,   # 色の変更範囲
            )
            results.append(result.url)
        return results


# 使用例: ECサイトの商品画像制作
workflow = AdobeCreativeWorkflow(
    client_id="your-client-id",
    client_secret="your-client-secret"
)

# 商品ヒーロー画像生成
hero = workflow.generate_product_hero(
    product_image_path="product_sneaker.jpg",
    scene_prompt="urban street at golden hour, wet asphalt reflection, "
                 "cinematic lighting, shallow depth of field, warm tones",
    output_sizes={
        "banner": {"width": 1920, "height": 600},
        "square": {"width": 1080, "height": 1080},
        "story": {"width": 1080, "height": 1920},
    }
)

print(f"バリエーション: {len(hero['variations'])}個生成")
for name, url in hero["final_outputs"].items():
    print(f"  {name}: {url}")
```

### 3.5 Generative Expand（画像拡張）の実践テクニック

```
Generative Expand の活用パターン

パターン1: アスペクト比変換
  ┌──────────┐
  │ 元の画像  │ 1:1 (Instagram)
  │ 1080x1080│
  └──────────┘
       ↓ Generative Expand
  ┌────────────────────────┐
  │ [←拡張] 元画像 [拡張→] │ 16:9 (YouTube サムネイル)
  │    1920 x 1080         │
  └────────────────────────┘

パターン2: 印刷用の余白追加
  ┌──────────┐         ┌────────────────┐
  │ [デザイン]│  →     │    [余白]       │
  │          │         │  ┌──────────┐  │
  │          │         │  │[デザイン] │  │ 裁ち落とし
  │          │         │  └──────────┘  │ 余白を AI で生成
  └──────────┘         └────────────────┘

パターン3: パノラマ化
  ┌──────────┐
  │ [風景写真]│
  │          │
  └──────────┘
       ↓ 左右に Generative Expand
  ┌──────────────────────────────────────┐
  │ [←AI生成拡張] [元の風景写真] [拡張→] │
  │            超ワイドパノラマ            │
  └──────────────────────────────────────┘

プロンプトのコツ:
- 拡張部分に何を生成すべきかを具体的に指示
- 「continue the same style and lighting」を追加
- 構図のバランスを意識した指示
- NG: 「拡張して」だけでは不自然な結果になりやすい
```

---

## 4. Figma AI の実践

### 4.1 Auto Layout AI

```
Figma AI のデザイン支援

  /auto-layout コマンド:
  選択した要素に最適な Auto Layout を自動適用
  ├── パディング推定
  ├── ギャップ推定
  └── アラインメント推定

  /suggest-component コマンド:
  デザインシステムから類似コンポーネントを提案
  ├── 既存のボタンスタイル候補
  ├── カードレイアウト候補
  └── ナビゲーションパターン候補
```

### 4.2 プロトタイプ支援

```
  デザイン画面                  AI 提案
  +------------------+         +------------------+
  | [ログイン画面]    |  --->   | インタラクション: |
  |                  |         | - ボタン→ホーム  |
  | Email: [____]    |         | - エラー表示     |
  | Pass:  [____]    |         | - ローディング    |
  | [ログイン]       |         | - パスワード忘れ  |
  +------------------+         +------------------+
```

### 4.3 Figma API を活用した自動化

```python
# Figma API でデザインデータを取得・操作
import requests
import json

class FigmaDesignAutomation:
    """Figma API を使ったデザインプロセス自動化"""

    BASE_URL = "https://api.figma.com/v1"

    def __init__(self, access_token: str):
        self.headers = {"X-FIGMA-TOKEN": access_token}

    def get_file(self, file_key: str, depth: int = 2) -> dict:
        """Figma ファイルの構造を取得"""
        response = requests.get(
            f"{self.BASE_URL}/files/{file_key}",
            headers=self.headers,
            params={"depth": depth}
        )
        response.raise_for_status()
        return response.json()

    def get_components(self, file_key: str) -> list[dict]:
        """ファイル内のコンポーネント一覧を取得"""
        response = requests.get(
            f"{self.BASE_URL}/files/{file_key}/components",
            headers=self.headers,
        )
        data = response.json()
        return [
            {
                "key": comp["key"],
                "name": comp["name"],
                "description": comp.get("description", ""),
                "containing_frame": comp.get("containing_frame", {}).get("name", ""),
            }
            for comp in data.get("meta", {}).get("components", [])
        ]

    def get_design_tokens(self, file_key: str) -> dict:
        """デザイントークン（Variables）を取得"""
        response = requests.get(
            f"{self.BASE_URL}/files/{file_key}/variables/local",
            headers=self.headers,
        )
        data = response.json()

        tokens = {"colors": {}, "spacing": {}, "typography": {}}
        for var_id, var in data.get("meta", {}).get("variables", {}).items():
            name = var["name"]
            resolved = var.get("resolvedType", "")
            values = var.get("valuesByMode", {})

            if resolved == "COLOR":
                # RGBA値を取得
                for mode_id, value in values.items():
                    if isinstance(value, dict) and "r" in value:
                        hex_color = self._rgba_to_hex(value)
                        tokens["colors"][name] = hex_color
            elif resolved == "FLOAT":
                for mode_id, value in values.items():
                    tokens["spacing"][name] = value

        return tokens

    def export_components_as_svg(
        self,
        file_key: str,
        node_ids: list[str],
        output_dir: str = "./exports",
    ) -> list[str]:
        """コンポーネントをSVGとしてエクスポート"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        ids_param = ",".join(node_ids)
        response = requests.get(
            f"{self.BASE_URL}/images/{file_key}",
            headers=self.headers,
            params={
                "ids": ids_param,
                "format": "svg",
                "svg_include_id": True,
                "svg_simplify_stroke": True,
            }
        )
        data = response.json()

        exported = []
        for node_id, url in data.get("images", {}).items():
            if url:
                svg_response = requests.get(url)
                filename = f"{output_dir}/{node_id.replace(':', '-')}.svg"
                with open(filename, "w") as f:
                    f.write(svg_response.text)
                exported.append(filename)

        return exported

    def generate_design_audit_report(self, file_key: str) -> dict:
        """デザインファイルの品質監査レポートを生成"""
        file_data = self.get_file(file_key, depth=4)
        components = self.get_components(file_key)

        # ページ分析
        pages = file_data.get("document", {}).get("children", [])
        report = {
            "file_name": file_data.get("name", "Unknown"),
            "pages": len(pages),
            "components_count": len(components),
            "issues": [],
            "recommendations": [],
        }

        # コンポーネント使用状況の分析
        detached_instances = 0
        unnamed_layers = 0
        inconsistent_spacing = set()

        for page in pages:
            self._analyze_node(
                page, report, detached_instances,
                unnamed_layers, inconsistent_spacing
            )

        # レポート生成
        if unnamed_layers > 10:
            report["issues"].append({
                "type": "naming",
                "severity": "warning",
                "message": f"{unnamed_layers}個の命名されていないレイヤーがあります",
                "fix": "レイヤーに意味のある名前を付けてください",
            })

        if detached_instances > 0:
            report["issues"].append({
                "type": "consistency",
                "severity": "error",
                "message": f"{detached_instances}個のデタッチされたインスタンスがあります",
                "fix": "メインコンポーネントに再リンクしてください",
            })

        report["score"] = max(0, 100 - len(report["issues"]) * 10)
        return report

    def _rgba_to_hex(self, rgba: dict) -> str:
        """RGBA値を16進カラーコードに変換"""
        r = int(rgba["r"] * 255)
        g = int(rgba["g"] * 255)
        b = int(rgba["b"] * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _analyze_node(self, node, report, detached, unnamed, spacing):
        """ノードを再帰的に分析"""
        if node.get("name", "").startswith("Frame ") or node.get("name", "").startswith("Group "):
            unnamed += 1
        children = node.get("children", [])
        for child in children:
            self._analyze_node(child, report, detached, unnamed, spacing)


# 使用例
figma = FigmaDesignAutomation(access_token="your-figma-token")

# デザイントークンの取得
tokens = figma.get_design_tokens(file_key="abc123xyz")
print("カラートークン:")
for name, color in tokens["colors"].items():
    print(f"  {name}: {color}")

# 品質監査レポート
report = figma.generate_design_audit_report(file_key="abc123xyz")
print(f"\nデザイン監査スコア: {report['score']}/100")
for issue in report["issues"]:
    print(f"  [{issue['severity']}] {issue['message']}")
```

### 4.4 デザインシステムの AI 活用

```
デザインシステムにおける AI の役割

┌──────────────────────────────────────────────────────────┐
│                    デザインシステム                         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Foundation    │  │ Components   │  │ Patterns     │  │
│  │ ─────────    │  │ ──────────   │  │ ────────     │  │
│  │ Color Tokens │  │ Button       │  │ Navigation   │  │
│  │ Typography   │  │ Card         │  │ Form Layout  │  │
│  │ Spacing      │  │ Input        │  │ Dashboard    │  │
│  │ Elevation    │  │ Modal        │  │ Settings     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         └─────────────────┼─────────────────┘           │
│                           ▼                              │
│  ┌────────────────────────────────────────────────────┐  │
│  │              AI Intelligence Layer                  │  │
│  │                                                    │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │  │
│  │  │コンポーネント │ │アクセシビリ │ │一貫性       │ │  │
│  │  │推薦エンジン  │ │ティチェック │ │チェッカー    │ │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ │  │
│  │                                                    │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │  │
│  │  │レスポンシブ  │ │ダークモード │ │多言語対応    │ │  │
│  │  │レイアウト    │ │自動生成     │ │レイアウト    │ │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 5. ツール連携とパイプライン構築

### 5.1 マルチツールワークフロー

```python
# 複数ツールを連携させた制作パイプライン
class DesignPipeline:
    """Canva + Firefly + Figma の統合パイプライン"""

    def __init__(self, config: dict):
        self.canva = CanvaBrandManager(
            api_key=config["canva_api_key"],
            brand_id=config["canva_brand_id"],
        )
        self.adobe = AdobeCreativeWorkflow(
            client_id=config["adobe_client_id"],
            client_secret=config["adobe_client_secret"],
        )
        self.figma = FigmaDesignAutomation(
            access_token=config["figma_token"],
        )

    def product_launch_campaign(
        self,
        product_name: str,
        product_images: list[str],
        brand_guidelines: dict,
    ) -> dict:
        """製品ローンチキャンペーンの全デザイン素材を一括生成

        1. Firefly で高品質商品画像を生成
        2. Canva でSNS投稿素材を一括生成
        3. Figma のデザインシステムから LP コンポーネントを取得
        """
        results = {"hero_images": [], "sns_assets": [], "lp_components": []}

        # Phase 1: 商品ヒーロー画像（Adobe Firefly）
        for img in product_images:
            hero = self.adobe.generate_product_hero(
                product_image_path=img,
                scene_prompt=brand_guidelines.get(
                    "hero_scene",
                    "clean minimalist studio, soft lighting"
                ),
            )
            results["hero_images"].append(hero)

        # Phase 2: SNS 素材（Canva）
        sns_data = []
        for i, hero in enumerate(results["hero_images"]):
            sns_data.append({
                "product_name": product_name,
                "hero_image": hero["final_outputs"]["square"],
                "campaign_text": f"{product_name} 新発売",
                "cta_text": "詳しくはプロフィールのリンクから",
            })

        results["sns_assets"] = self.canva.batch_generate_with_brand(
            template_id=brand_guidelines["sns_template_id"],
            data_list=sns_data,
            sizes=["instagram_post", "instagram_story",
                   "facebook_post", "twitter_post"],
        )

        # Phase 3: LP コンポーネント（Figma）
        components = self.figma.get_components(
            file_key=brand_guidelines["figma_file_key"]
        )
        lp_components = [
            c for c in components
            if "product" in c["name"].lower() or "hero" in c["name"].lower()
        ]
        results["lp_components"] = lp_components

        return results


# パイプライン実行
pipeline = DesignPipeline(config={
    "canva_api_key": "canva-key",
    "canva_brand_id": "brand-123",
    "adobe_client_id": "adobe-id",
    "adobe_client_secret": "adobe-secret",
    "figma_token": "figma-token",
})

campaign = pipeline.product_launch_campaign(
    product_name="EcoBreeze スニーカー",
    product_images=["sneaker_front.jpg", "sneaker_side.jpg"],
    brand_guidelines={
        "hero_scene": "urban rooftop at sunset, concrete and plants",
        "sns_template_id": "template-789",
        "figma_file_key": "xyz789abc",
    },
)

print(f"ヒーロー画像: {len(campaign['hero_images'])}パターン")
print(f"SNS素材: {len(campaign['sns_assets'])}セット")
print(f"LPコンポーネント: {len(campaign['lp_components'])}個")
```

---

## 6. ツール比較表

| 機能 | Canva | Adobe Firefly | Figma AI |
|------|:-----:|:------------:|:--------:|
| テキスト→画像生成 | Magic Media | Text to Image | -- |
| 画像編集 AI | Magic Edit | Generative Fill | -- |
| 背景除去 | 対応 | 対応 | -- |
| テンプレート自動生成 | Magic Design | -- | Auto Layout |
| 文章生成 | Magic Write | -- | -- |
| UI コンポーネント提案 | -- | -- | 対応 |
| プロトタイプ AI | -- | -- | 対応 |
| API 連携 | Canva API | Firefly API | Figma API |
| 料金 | 無料〜月額1,500円 | Creative Cloud に含む | 無料〜月額$15 |
| 対象ユーザー | 全般・非デザイナー | プロデザイナー | UI/UXデザイナー |

| ユースケース | 推奨ツール | 理由 |
|------------|-----------|------|
| SNS 投稿画像 | Canva | テンプレート+一括生成 |
| 商品画像の背景変更 | Adobe Firefly | 高品質な Generative Fill |
| Webサイト UI デザイン | Figma AI | コンポーネント管理+プロトタイプ |
| プレゼン資料 | Canva | テンプレート豊富、操作簡単 |
| 写真加工・合成 | Adobe Firefly | Photoshop 連携、プロ品質 |
| デザインシステム構築 | Figma AI | 変数管理、コンポーネントライブラリ |

---

## 6. アンチパターン

### アンチパターン 1: AI 生成をそのまま使う

```
BAD:
  AI で画像生成 → ブランドガイドライン無視でそのまま使用
  → 色調がバラバラ、フォントが統一されない、ブランドイメージが崩壊

GOOD:
  1. ブランドキット（カラー、フォント、ロゴ）を事前設定
  2. AI 生成をベースとして使用
  3. ブランドガイドラインに合わせて調整
  4. チームレビューを経て公開
```

### アンチパターン 2: 1つのツールで全てをこなそうとする

```
BAD:
  Canva で UI デザイン → レスポンシブ対応できない
  Figma で SNS 画像 → テンプレート機能が弱い

GOOD: ツールを使い分ける
  企画・アイデア出し    → Canva (Magic Write)
  UI/UXデザイン         → Figma AI
  写真加工・商品画像    → Adobe Firefly + Photoshop
  SNS投稿・マーケ素材   → Canva (Magic Design)
```

---

## 7. FAQ

### Q1. AI 生成画像の著作権はどうなる？

**A.** 各ツールで規約が異なる。**Canva**: 商用利用可。生成画像の著作権はユーザーに帰属。**Adobe Firefly**: 商用利用可。学習データは Adobe Stock とライセンス済み素材のみ（著作権侵害リスクが低い）。IP 補償あり。**Figma**: AI 提案のレイアウト自体に著作権は発生しない。いずれも利用規約を定期的に確認すること。

### Q2. デザインの一貫性をAIで維持するには？

**A.** (1) **ブランドキット**を各ツールに事前登録する（カラーパレット、フォント、ロゴ）。(2) Adobe Firefly の **Style Reference** で色調・雰囲気を統一する。(3) Canva の **Brand Kit** 機能でテンプレートを標準化する。(4) Figma の **Design Tokens** でコンポーネントの変数を管理する。AI 生成時にこれらの制約を入力として与えることで一貫性を維持する。

### Q3. 非デザイナーがデザインツールを選ぶ基準は？

**A.** (1) **学習コストが最も低い**のは Canva（テンプレートベースで直感的操作）。(2) SNS やプレゼン資料なら Canva で十分。(3) Web/アプリの UI デザインが必要なら Figma（エンジニアとの連携が容易）。(4) 高品質な画像加工が必要なら Adobe Firefly（Photoshop 連携）。まず Canva から始め、必要に応じて他ツールを追加するのが現実的。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Canva | 非デザイナー向け。テンプレート + AI で高速にデザイン制作 |
| Adobe Firefly | プロ向け。Photoshop 連携、高品質な画像生成・編集 |
| Figma AI | UI/UX 向け。コンポーネント提案、Auto Layout、プロトタイプ支援 |
| ツール選定 | ユースケースと対象ユーザーのスキルレベルで判断 |
| ブランド一貫性 | Brand Kit、Style Reference、Design Tokens で統一 |
| AI 生成の注意点 | そのまま使わず、ブランドガイドラインに合わせて調整 |

---

## 次に読むべきガイド

- [動画編集](../02-video/01-video-editing.md) -- AI を活用した動画編集ツール
- [アニメーション](../02-video/02-animation.md) -- AI アニメーション生成
- [倫理的考慮](../03-3d/03-ethical-considerations.md) -- AI 生成コンテンツの著作権と倫理

---

## 参考文献

1. **Canva Design School** -- https://www.canva.com/designschool/ -- Canva の公式学習リソース
2. **Adobe Firefly Documentation** -- https://www.adobe.com/products/firefly.html -- Firefly の公式ドキュメント
3. **Figma Learn** -- https://help.figma.com/ -- Figma の公式ヘルプセンター
