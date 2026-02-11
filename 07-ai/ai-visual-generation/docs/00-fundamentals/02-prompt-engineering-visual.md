# ビジュアルプロンプト — 構図、スタイル、ネガティブプロンプト

> 画像生成AIから意図通りの出力を得るためのプロンプトエンジニアリング技術を、構図設計からネガティブプロンプトまで体系的に解説する。

---

## この章で学ぶこと

1. **プロンプト構造の設計原則** — 主題、スタイル、品質、構図の4層フレームワーク
2. **ネガティブプロンプトの戦略** — 不要要素の排除と品質向上のテクニック
3. **モデル固有の最適化** — SD系、DALL-E、Midjourney それぞれの効果的なプロンプト手法

---

## 1. プロンプト構造の4層フレームワーク

### コード例1: プロンプトビルダークラス

```python
class VisualPromptBuilder:
    """4層フレームワークに基づくプロンプト構築"""

    def __init__(self):
        self.subject = ""       # 主題
        self.style = ""         # スタイル
        self.quality = ""       # 品質指定
        self.composition = ""   # 構図
        self.negative = ""      # ネガティブプロンプト

    def set_subject(self, subject: str, details: str = ""):
        """第1層: 何を描くか"""
        self.subject = f"{subject}, {details}" if details else subject
        return self

    def set_style(self, style: str, artist: str = "", medium: str = ""):
        """第2層: どのスタイルで描くか"""
        parts = [style]
        if artist:
            parts.append(f"in the style of {artist}")
        if medium:
            parts.append(medium)
        self.style = ", ".join(parts)
        return self

    def set_quality(self, *tags):
        """第3層: 品質と詳細度"""
        self.quality = ", ".join(tags)
        return self

    def set_composition(self, camera: str = "", lighting: str = "",
                        angle: str = ""):
        """第4層: 構図と撮影設定"""
        parts = [p for p in [camera, lighting, angle] if p]
        self.composition = ", ".join(parts)
        return self

    def set_negative(self, *tags):
        """ネガティブプロンプト"""
        self.negative = ", ".join(tags)
        return self

    def build(self) -> dict:
        positive_parts = [
            p for p in [self.subject, self.style,
                        self.quality, self.composition] if p
        ]
        return {
            "prompt": ", ".join(positive_parts),
            "negative_prompt": self.negative
        }

# 使用例
prompt = (
    VisualPromptBuilder()
    .set_subject("古い日本の寺院", "苔むした石段、雨上がり")
    .set_style("フォトリアリスティック", medium="デジタル写真")
    .set_quality("8K", "超高解像度", "シャープ", "高ディテール")
    .set_composition(
        camera="Sony α7R V, 24mm f/1.4",
        lighting="ゴールデンアワー、柔らかい自然光",
        angle="ローアングル"
    )
    .set_negative("低品質", "ぼやけ", "歪み", "人物", "テキスト")
    .build()
)
print(prompt)
```

### ASCII図解1: プロンプト4層フレームワーク

```
┌─────────────────────────────────────────────────────┐
│                  プロンプト構造                       │
│                                                     │
│  第1層: 主題 (Subject)          [最重要・最初に記述]  │
│  ┌─────────────────────────────────────────────┐    │
│  │ "赤い着物を着た女性が桜の下に立っている"      │    │
│  └─────────────────────────────────────────────┘    │
│           │                                         │
│  第2層: スタイル (Style)        [芸術的方向性]       │
│  ┌─────────────────────────────────────────────┐    │
│  │ "浮世絵風, 葛飾北斎のスタイル, 木版画"        │    │
│  └─────────────────────────────────────────────┘    │
│           │                                         │
│  第3層: 品質 (Quality)          [技術的品質]         │
│  ┌─────────────────────────────────────────────┐    │
│  │ "masterpiece, best quality, 8K, 高ディテール" │    │
│  └─────────────────────────────────────────────┘    │
│           │                                         │
│  第4層: 構図 (Composition)      [カメラ・照明]       │
│  ┌─────────────────────────────────────────────┐    │
│  │ "黄金比構図, 自然光, 50mm lens, 浅い被写界深度"│    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ネガティブ: [除外したい要素]                        │
│  ┌─────────────────────────────────────────────┐    │
│  │ "低品質, ぼやけ, 変形した手, 余分な指"        │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 2. スタイル指定のテクニック

### コード例2: スタイルキーワード辞書

```python
STYLE_KEYWORDS = {
    "フォトリアル": {
        "keywords": ["photorealistic", "hyperrealistic", "RAW photo",
                     "8K UHD", "DSLR", "film grain"],
        "camera": ["Canon EOS R5", "Sony α7R V", "Nikon Z9",
                    "Hasselblad X2D"],
        "lens": ["85mm f/1.4", "35mm f/1.8", "50mm f/1.2",
                 "24-70mm f/2.8"],
    },
    "アニメ": {
        "keywords": ["anime style", "cel shading", "vibrant colors",
                     "detailed eyes"],
        "substyles": ["新海誠風", "ジブリ風", "サイバーパンクアニメ",
                      "少女漫画風"],
    },
    "油絵": {
        "keywords": ["oil painting", "canvas texture", "visible brushstrokes",
                     "impasto technique"],
        "artists": ["モネ風", "ゴッホ風", "レンブラント風",
                    "フェルメール風"],
    },
    "水彩画": {
        "keywords": ["watercolor painting", "soft edges", "color bleeding",
                     "wet-on-wet technique"],
        "effects": ["にじみ", "透明感", "紙のテクスチャ"],
    },
    "3Dレンダリング": {
        "keywords": ["3D render", "octane render", "unreal engine 5",
                     "ray tracing"],
        "software": ["Blender", "Cinema 4D", "KeyShot"],
    },
    "コンセプトアート": {
        "keywords": ["concept art", "digital painting", "matte painting",
                     "detailed illustration"],
        "use_cases": ["ゲーム", "映画", "書籍表紙"],
    },
}

def suggest_style_prompt(style_name: str) -> str:
    """スタイルに合ったキーワードを提案"""
    style = STYLE_KEYWORDS.get(style_name, {})
    keywords = style.get("keywords", [])
    return ", ".join(keywords[:4])

# 使用例
print(suggest_style_prompt("フォトリアル"))
# → "photorealistic, hyperrealistic, RAW photo, 8K UHD"
```

### コード例3: 構図キーワードの体系

```python
COMPOSITION_GUIDE = {
    "カメラアングル": {
        "俯瞰 (バーズアイ)": "bird's eye view, top-down perspective",
        "ローアングル": "low angle shot, worm's eye view",
        "アイレベル": "eye level, straight-on view",
        "ダッチアングル": "dutch angle, tilted camera",
        "オーバーショルダー": "over-the-shoulder shot",
    },
    "ショットサイズ": {
        "極端なクローズアップ": "extreme close-up, macro shot",
        "クローズアップ": "close-up portrait, head shot",
        "ミディアム": "medium shot, waist up",
        "フルショット": "full body shot, full length",
        "ワイド": "wide shot, establishing shot",
        "パノラマ": "panoramic view, ultra-wide",
    },
    "照明": {
        "ゴールデンアワー": "golden hour, warm sunlight",
        "ブルーアワー": "blue hour, twilight",
        "レンブラント光": "Rembrandt lighting, dramatic shadow",
        "リムライト": "rim lighting, backlit, silhouette",
        "スタジオ照明": "studio lighting, softbox, professional",
        "ネオン": "neon lighting, cyberpunk glow",
        "キアロスクーロ": "chiaroscuro, dramatic contrast",
    },
    "構図法則": {
        "三分割法": "rule of thirds composition",
        "黄金比": "golden ratio, fibonacci spiral",
        "対称": "symmetrical composition, centered",
        "リーディングライン": "leading lines, depth",
        "フレーム内フレーム": "frame within frame",
    },
}

def build_composition_prompt(angle, shot, light, rule):
    """構図プロンプトを組み立てる"""
    parts = []
    for category, key in [
        ("カメラアングル", angle),
        ("ショットサイズ", shot),
        ("照明", light),
        ("構図法則", rule),
    ]:
        if key and key in COMPOSITION_GUIDE.get(category, {}):
            parts.append(COMPOSITION_GUIDE[category][key])
    return ", ".join(parts)

# 使用例
comp = build_composition_prompt("ローアングル", "ワイド",
                                 "ゴールデンアワー", "黄金比")
print(comp)
# → "low angle shot, worm's eye view, wide shot, establishing shot,
#     golden hour, warm sunlight, golden ratio, fibonacci spiral"
```

### ASCII図解2: 構図法則の視覚的ガイド

```
三分割法:                    黄金比:
┌───┬───┬───┐              ┌──────┬────┐
│   │   │   │              │      │    │
│   │ ● │   │  ← 交点に   │      │ ●  │  ← 螺旋の
├───┼───┼───┤    主題配置  │      │    │    焦点に
│   │   │   │              ├──────┼────┤    主題配置
│   │   │   │              │      │    │
├───┼───┼───┤              └──────┴────┘
│   │   │   │
└───┴───┴───┘

対称構図:                    リーディングライン:
┌─────────────┐              ┌─────────────┐
│      |      │              │ \         / │
│      |      │              │  \       /  │
│    __|__    │              │   \     /   │
│   / | \   │              │    \   /    │
│  /  |  \  │              │     \ /     │
│ /   |   \ │              │      ●      │
└─────────────┘              └─────────────┘
建物、通路 等                道、川、視線誘導
```

---

## 3. ネガティブプロンプト戦略

### コード例4: ネガティブプロンプトテンプレート

```python
NEGATIVE_PROMPT_TEMPLATES = {
    "汎用 (高品質化)": (
        "low quality, worst quality, blurry, out of focus, "
        "jpeg artifacts, compression artifacts, watermark, "
        "text, signature, username, logo"
    ),
    "人物写真": (
        "deformed, ugly, bad anatomy, bad proportions, "
        "extra limbs, extra fingers, mutated hands, "
        "poorly drawn hands, poorly drawn face, "
        "disfigured, gross proportions, long neck, "
        "cross-eyed, malformed limbs"
    ),
    "風景写真": (
        "oversaturated, HDR artifacts, chromatic aberration, "
        "lens flare, overexposed, underexposed, "
        "person, people, human, text, watermark"
    ),
    "アニメ/イラスト": (
        "3d, realistic, photographic, bad anatomy, "
        "bad hands, missing fingers, extra digit, "
        "fewer digits, cropped, worst quality, "
        "low quality, normal quality"
    ),
    "建築/インテリア": (
        "people, furniture out of place, distorted walls, "
        "unrealistic proportions, floating objects, "
        "bad perspective, warped lines"
    ),
}

def get_negative_prompt(category: str, custom_exclusions: list = None):
    """カテゴリに応じたネガティブプロンプトを生成"""
    base = NEGATIVE_PROMPT_TEMPLATES.get(category, "")
    if custom_exclusions:
        base += ", " + ", ".join(custom_exclusions)
    return base

# 使用例
neg = get_negative_prompt("人物写真", ["nsfw", "child", "cartoon"])
print(neg)
```

### コード例5: プロンプト重み付け (SD系)

```python
"""
Stable Diffusion系モデルでのプロンプト重み付け構文

基本構文:
  (keyword)       → 1.1倍の重み
  ((keyword))     → 1.21倍 (1.1²)
  (keyword:1.5)   → 1.5倍の重み
  (keyword:0.5)   → 0.5倍 (弱める)
  [keyword]       → 0.9倍の重み (一部ツール)

DALL-E 3では重み付け構文は使えない
→ 自然言語で強調を表現する
"""

def apply_emphasis(prompt: str, weights: dict) -> str:
    """キーワードに重み付けを適用 (SD系向け)"""
    result = prompt
    for keyword, weight in weights.items():
        if weight == 1.0:
            continue
        weighted = f"({keyword}:{weight})"
        result = result.replace(keyword, weighted)
    return result

# 使用例
prompt = "beautiful landscape, cherry blossoms, mount fuji, sunset"
weights = {
    "cherry blossoms": 1.4,   # 桜を強調
    "mount fuji": 1.2,        # 富士山をやや強調
    "sunset": 0.8,            # 夕日を少し抑える
}

weighted_prompt = apply_emphasis(prompt, weights)
print(weighted_prompt)
# → "beautiful landscape, (cherry blossoms:1.4),
#    (mount fuji:1.2), (sunset:0.8)"
```

### ASCII図解3: モデル別プロンプト最適化マップ

```
┌────────── Stable Diffusion 系 ──────────┐
│ ・重み付け構文 (keyword:1.5) が使える    │
│ ・ネガティブプロンプトが効果大           │
│ ・タグベース + 自然文のハイブリッド       │
│ ・LoRA/テキスト反転でカスタム語彙追加    │
│ ・推奨: 短いタグの羅列 + 品質キーワード  │
└──────────────────────────────────────────┘

┌────────── DALL-E 3 ─────────────────────┐
│ ・自然言語で詳細に記述                   │
│ ・重み付け構文は未サポート               │
│ ・ネガティブプロンプトなし               │
│   → 「～を含まない」と記述              │
│ ・GPT-4がプロンプトを内部で書き換え      │
│ ・推奨: 文章として詳細に状況を説明       │
└──────────────────────────────────────────┘

┌────────── Midjourney ───────────────────┐
│ ・パラメータ: --ar, --v, --s, --c, --q │
│ ・--no でネガティブ指定                 │
│ ・短く印象的なプロンプトが効果的         │
│ ・:: でマルチプロンプト (重み分離)       │
│ ・推奨: キーワード + パラメータ調整      │
└──────────────────────────────────────────┘

┌────────── Flux ─────────────────────────┐
│ ・自然言語記述が得意                     │
│ ・テキスト描画が正確                     │
│ ・ネガティブプロンプトは限定的           │
│ ・T5エンコーダで長文理解が可能           │
│ ・推奨: 詳細な自然文記述                 │
└──────────────────────────────────────────┘
```

---

## 4. 比較表

### 比較表1: プロンプト構成要素の効果

| 構成要素 | 効果の大きさ | 記述例 | 影響範囲 |
|---------|------------|--------|---------|
| **主題** | 最大 | "赤い着物の女性" | 生成内容の核心 |
| **スタイル** | 大 | "油絵風, 印象派" | 全体の雰囲気・質感 |
| **品質タグ** | 中~大 | "masterpiece, 8K" | 細部の品質 |
| **照明** | 中 | "ゴールデンアワー" | 色調・陰影 |
| **構図** | 中 | "三分割法, ローアングル" | レイアウト |
| **カメラ設定** | 小~中 | "85mm f/1.4, bokeh" | 被写界深度・ボケ |
| **ネガティブ** | 中~大 | "blurry, deformed" | 不要要素の排除 |
| **重み付け** | 小~中 | "(cherry:1.4)" | 特定要素の強弱 |

### 比較表2: モデル別プロンプト特性比較

| 特性 | SD 1.5/XL | SD3/Flux | DALL-E 3 | Midjourney |
|------|-----------|----------|----------|-----------|
| **推奨記述** | タグ羅列 | 自然文 | 詳細な自然文 | 短いキーワード |
| **最大長** | 75/150トークン | 256トークン | 4000文字 | ~350語 |
| **重み構文** | (keyword:1.5) | 限定的 | なし | :: |
| **ネガティブ** | 強力 | 限定的 | なし (文中記述) | --no |
| **日本語** | モデル依存 | 部分対応 | 対応 | 限定的 |
| **テキスト描画** | 苦手 | 得意 | 得意 | やや苦手 |

---

## 5. アンチパターン

### アンチパターン1: プロンプトの過剰詰め込み

```
[問題]
「beautiful amazing stunning gorgeous incredible masterpiece
 best quality ultra detailed 8K HDR award winning...」
と品質タグを大量に詰め込む。

[なぜ問題か]
- トークン制限を圧迫し、主題の記述スペースが不足
- 重複するキーワードは効果が薄い (収穫逓減)
- モデルが混乱して焦点がぼやけた画像になる

[正しいアプローチ]
- 品質タグは3-5個に絞る
- 主題と構図の記述に十分なトークンを確保
- テスト生成で効果を確認し、不要なタグを除去
```

### アンチパターン2: 他モデルのプロンプトをそのまま流用

```
[問題]
MidjourneyのプロンプトをそのままStable Diffusionに、
SD用のタグ羅列をDALL-E 3にそのまま入力する。

[なぜ問題か]
- 各モデルのテキストエンコーダが異なる
  (CLIP, T5, GPT-4 等)
- Midjourney固有パラメータ (--v, --ar) は他で無意味
- SD用の重み構文はDALL-E 3で無視される
- 最適なプロンプト形式がモデルごとに異なる

[正しいアプローチ]
- モデルごとにプロンプトを最適化
- SD系: タグ + 重み付け
- DALL-E 3: 自然言語の詳細な文章
- Midjourney: 短く印象的 + パラメータ
```

---

## FAQ

### Q1: 日本語プロンプトと英語プロンプト、どちらが効果的?

**A:** 一般的に**英語プロンプトが推奨**されます:

- **SD系:** 学習データが英語中心のため、英語の方が高精度
- **DALL-E 3:** 日本語対応だが、英語の方がニュアンスが正確
- **Midjourney:** 英語のみ正式対応
- **Flux:** T5エンコーダで多言語対応だが、英語がベスト
- **例外:** 日本語特化モデル (Animagine XL等) は日本語タグが有効

### Q2: プロンプトの長さはどのくらいが最適?

**A:** モデルとタスクによります:

- **SD 1.5:** 20-40トークン (約75トークン上限)
- **SDXL:** 40-80トークン (150トークン上限)
- **DALL-E 3:** 100-300語の詳細な文章が効果的
- **Midjourney:** 20-60語。短い方がスタイルが安定
- **原則:** 必要十分な情報を、無駄なく記述する

### Q3: プロンプトのデバッグ方法は?

**A:** 体系的にテストします:

1. **最小プロンプトから始める:** 主題のみで生成し、基本品質を確認
2. **要素を一つずつ追加:** スタイル → 品質 → 構図 の順に追加
3. **シード固定:** 同一シードで変更の影響を比較
4. **バッチ生成:** 4-8枚同時生成してばらつきを確認
5. **ネガティブの段階テスト:** なし → 基本 → 詳細 と比較

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **4層構造** | 主題 → スタイル → 品質 → 構図 の優先順位で記述 |
| **ネガティブ** | 汎用テンプレート + カテゴリ固有 + カスタム除外 |
| **重み付け** | SD系: (keyword:weight)、MJ: ::weight |
| **モデル別最適化** | SD=タグ、DALL-E=自然文、MJ=短いキーワード |
| **言語** | 英語が基本。日本語特化モデルは例外 |
| **デバッグ** | 最小プロンプト → 段階的追加 → シード固定比較 |

---

## 次に読むべきガイド

- [../01-image/00-image-generation.md](../01-image/00-image-generation.md) — 実際の画像生成ツールで実践
- [../01-image/01-image-editing.md](../01-image/01-image-editing.md) — インペインティングでの部分的プロンプト
- [../02-video/00-video-generation.md](../02-video/00-video-generation.md) — 動画生成向けプロンプト

---

## 参考文献

1. Oppenlaender, J. (2023). "A Taxonomy of Prompt Modifiers for Text-to-Image Generation." *arXiv*. https://arxiv.org/abs/2204.13988
2. Liu, V. & Chilton, L. B. (2022). "Design Guidelines for Prompt Engineering Text-to-Image Generative Models." *CHI 2022*. https://doi.org/10.1145/3491102.3501825
3. Witteveen, S. & Andrews, M. (2022). "Investigating Prompt Engineering in Diffusion Models." *arXiv*. https://arxiv.org/abs/2211.15462
4. Betker, J. et al. (2023). "Improving Image Generation with Better Captions." *OpenAI Technical Report*. https://cdn.openai.com/papers/dall-e-3.pdf
