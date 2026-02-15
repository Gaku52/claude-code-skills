# ビジュアルプロンプト — 構図、スタイル、ネガティブプロンプト

> 画像生成AIから意図通りの出力を得るためのプロンプトエンジニアリング技術を、構図設計からネガティブプロンプトまで体系的に解説する。

---

## この章で学ぶこと

1. **プロンプト構造の設計原則** — 主題、スタイル、品質、構図の4層フレームワーク
2. **ネガティブプロンプトの戦略** — 不要要素の排除と品質向上のテクニック
3. **モデル固有の最適化** — SD系、DALL-E、Midjourney それぞれの効果的なプロンプト手法
4. **プロンプトの体系的デバッグ** — A/Bテスト、シード固定比較、段階的改善の手法
5. **業種別プロンプトテンプレート** — 広告、ゲーム、建築、ファッション向けの実践パターン
6. **プロンプト品質の定量評価** — CLIP Score、人間評価、自動評価パイプラインの構築

---

## 1. プロンプト構造の4層フレームワーク

画像生成AIへの指示は、4つの層を意識して構築することで一貫した高品質な出力が得られる。各層は独立しており、タスクに応じて重み付けを調整できる。

### 1.1 フレームワークの理論的背景

テキストエンコーダ（CLIP、T5等）は入力テキストをトークンに分割し、各トークンの埋め込みベクトルが画像生成を制御する。プロンプト内のトークン位置（先頭ほど影響が大きい）と意味的クラスタリング（関連するキーワードをまとめる）が出力品質に直結する。

```
テキスト入力 → トークン化 → 埋め込みベクトル → Cross-Attention → 画像生成

トークン位置の影響度:
位置  1-10:  ████████████████████ 100% (主題を配置)
位置 11-20:  ████████████████     80% (スタイルを配置)
位置 21-40:  ████████████         60% (品質・構図を配置)
位置 41-75:  ████████             40% (補足情報)
位置 76+:    ████                 20% (SD1.5では切り捨て)
```

### コード例1: プロンプトビルダークラス（拡張版）

```python
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class PromptLayer:
    """プロンプトの各層を表すデータクラス"""
    name: str
    content: str = ""
    priority: int = 0  # 0が最高優先
    tokens_estimate: int = 0

    def estimate_tokens(self) -> int:
        """おおよそのトークン数を推定（英語: ~4文字/トークン）"""
        self.tokens_estimate = len(self.content.split(", "))
        return self.tokens_estimate


class VisualPromptBuilder:
    """
    4層フレームワークに基づくプロンプト構築

    各層の役割:
    - 第1層 (Subject): 何を描くか — 最も重要、先頭に配置
    - 第2層 (Style): どのスタイルで描くか — 芸術的方向性
    - 第3層 (Quality): 技術的品質 — 解像度、ディテール
    - 第4層 (Composition): 構図と撮影設定 — カメラ、照明、アングル
    """

    # モデルごとのトークン上限
    TOKEN_LIMITS = {
        "sd15": 75,
        "sdxl": 150,
        "sd3": 256,
        "flux": 512,
        "dalle3": 4000,  # 文字数
        "midjourney": 350,  # 語数
    }

    def __init__(self, model: str = "sdxl"):
        self.model = model
        self.subject = PromptLayer("subject", priority=0)
        self.style = PromptLayer("style", priority=1)
        self.quality = PromptLayer("quality", priority=2)
        self.composition = PromptLayer("composition", priority=3)
        self.negative = ""
        self._history: list[dict] = []  # プロンプト履歴

    def set_subject(self, subject: str, details: str = "",
                    action: str = "", environment: str = ""):
        """
        第1層: 何を描くか

        Args:
            subject: 主題（人物、物体、風景など）
            details: 主題の詳細（服装、色、素材など）
            action: 動作・ポーズ
            environment: 環境・背景
        """
        parts = [subject]
        if details:
            parts.append(details)
        if action:
            parts.append(action)
        if environment:
            parts.append(environment)
        self.subject.content = ", ".join(parts)
        return self

    def set_style(self, style: str, artist: str = "",
                  medium: str = "", era: str = "",
                  influences: list[str] = None):
        """
        第2層: どのスタイルで描くか

        Args:
            style: 基本スタイル（フォトリアル、アニメ、油絵など）
            artist: 参照アーティスト
            medium: 画材・技法
            era: 時代・年代
            influences: その他の影響・参照
        """
        parts = [style]
        if artist:
            parts.append(f"in the style of {artist}")
        if medium:
            parts.append(medium)
        if era:
            parts.append(era)
        if influences:
            parts.extend(influences)
        self.style.content = ", ".join(parts)
        return self

    def set_quality(self, *tags, resolution: str = "",
                    detail_level: str = ""):
        """
        第3層: 品質と詳細度

        Args:
            *tags: 品質タグ (masterpiece, best quality 等)
            resolution: 解像度指定 (8K, 4K 等)
            detail_level: ディテールレベル
        """
        parts = list(tags)
        if resolution:
            parts.append(resolution)
        if detail_level:
            parts.append(detail_level)
        self.quality.content = ", ".join(parts)
        return self

    def set_composition(self, camera: str = "", lighting: str = "",
                        angle: str = "", depth_of_field: str = "",
                        color_palette: str = ""):
        """
        第4層: 構図と撮影設定

        Args:
            camera: カメラ機種・レンズ
            lighting: 照明設定
            angle: カメラアングル
            depth_of_field: 被写界深度
            color_palette: カラーパレット
        """
        parts = [p for p in [camera, lighting, angle,
                             depth_of_field, color_palette] if p]
        self.composition.content = ", ".join(parts)
        return self

    def set_negative(self, *tags, template: str = None):
        """
        ネガティブプロンプト

        Args:
            *tags: 除外タグ
            template: テンプレート名 (汎用, 人物, 風景 等)
        """
        neg_parts = list(tags)
        if template and template in NEGATIVE_PROMPT_TEMPLATES:
            neg_parts.insert(0, NEGATIVE_PROMPT_TEMPLATES[template])
        self.negative = ", ".join(neg_parts)
        return self

    def estimate_tokens(self) -> dict:
        """各層のトークン数を推定"""
        layers = [self.subject, self.style,
                  self.quality, self.composition]
        total = sum(l.estimate_tokens() for l in layers)
        limit = self.TOKEN_LIMITS.get(self.model, 75)
        return {
            "layers": {l.name: l.tokens_estimate for l in layers},
            "total": total,
            "limit": limit,
            "utilization": f"{total / limit * 100:.1f}%",
            "remaining": max(0, limit - total),
        }

    def optimize(self) -> "VisualPromptBuilder":
        """トークン制限に合わせてプロンプトを最適化"""
        token_info = self.estimate_tokens()
        if token_info["total"] <= token_info["limit"]:
            return self  # 制限内なら何もしない

        # 優先度の低い層から削減
        layers = sorted(
            [self.composition, self.quality, self.style, self.subject],
            key=lambda l: l.priority,
            reverse=True  # 優先度低い順
        )
        excess = token_info["total"] - token_info["limit"]
        for layer in layers:
            if excess <= 0:
                break
            tokens = layer.content.split(", ")
            while len(tokens) > 1 and excess > 0:
                tokens.pop()
                excess -= 1
            layer.content = ", ".join(tokens)
        return self

    def build(self) -> dict:
        """最終プロンプトを構築"""
        positive_parts = [
            l.content for l in [self.subject, self.style,
                                self.quality, self.composition]
            if l.content
        ]
        result = {
            "prompt": ", ".join(positive_parts),
            "negative_prompt": self.negative,
            "model": self.model,
            "token_estimate": self.estimate_tokens(),
        }
        self._history.append(result)
        return result

    def export_history(self, filepath: str):
        """プロンプト履歴をJSONで出力"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2)

    def __repr__(self):
        return (f"VisualPromptBuilder(model={self.model}, "
                f"subject='{self.subject.content[:30]}...')")


# 使用例: 日本の寺院
prompt = (
    VisualPromptBuilder(model="sdxl")
    .set_subject(
        "古い日本の寺院",
        details="苔むした石段、雨上がりの濡れた地面",
        environment="山間の杉林に囲まれた静寂な空間"
    )
    .set_style(
        "フォトリアリスティック",
        medium="デジタル写真",
        era="現代"
    )
    .set_quality(
        "8K", "超高解像度", "シャープ", "高ディテール",
        resolution="8192x4608"
    )
    .set_composition(
        camera="Sony α7R V, 24mm f/1.4",
        lighting="ゴールデンアワー、柔らかい自然光",
        angle="ローアングル",
        depth_of_field="パンフォーカス",
        color_palette="落ち着いた緑と金色"
    )
    .set_negative(
        "低品質", "ぼやけ", "歪み", "人物", "テキスト",
        template="汎用 (高品質化)"
    )
    .build()
)
print(prompt["prompt"])
print(f"トークン使用率: {prompt['token_estimate']['utilization']}")
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

トークン位置と影響度の関係:
┌──────────────────────────────────────────────────┐
│ Position:  1    10   20   30   40   50   60   75 │
│ Impact: ████████████████████░░░░░░░░░░░░░░░░░░░  │
│         ▲ 主題   ▲ スタイル  ▲ 品質   ▲ 構図    │
│         最重要    重要        中程度    補足      │
│                                                  │
│ SD1.5:  |←──── 75トークン上限 ────→|             │
│ SDXL:   |←──── 150トークン上限 ──────────→|      │
│ Flux:   |←──── 512トークン上限 ────────────────→ │
└──────────────────────────────────────────────────┘
```

### 1.2 主題記述のベストプラクティス

主題は「何を（What）」「誰が（Who）」「どこで（Where）」「何をしている（Doing What）」の4Wで構成する。

```python
SUBJECT_TEMPLATES = {
    "人物ポートレート": {
        "who": "{性別}, {年齢層}, {外見の特徴}",
        "what": "{服装}, {アクセサリー}, {表情}",
        "where": "{背景}, {環境}",
        "doing": "{ポーズ}, {動作}",
        "example": "young Japanese woman, long black hair, "
                   "wearing a red silk kimono with golden obi, "
                   "gentle smile, standing under cherry blossoms, "
                   "holding a paper umbrella"
    },
    "風景": {
        "what": "{主要な地形/建造物}",
        "where": "{地理的な場所}, {季節}, {時間帯}",
        "details": "{天候}, {大気の状態}, {植生}",
        "example": "majestic snow-capped mountain reflected in "
                   "a crystal clear alpine lake, surrounded by "
                   "autumn foliage, early morning mist, "
                   "remote wilderness in Hokkaido"
    },
    "静物": {
        "what": "{主要オブジェクト}, {配置}",
        "where": "{テーブル/サーフェス}, {背景}",
        "details": "{素材感}, {色}, {テクスチャ}",
        "example": "antique ceramic teapot with crackle glaze, "
                   "two yunomi cups, bamboo tea whisk, "
                   "on a weathered wooden tray, "
                   "soft window light from the left"
    },
    "建築": {
        "what": "{建物の種類}, {建築様式}",
        "where": "{立地}, {周辺環境}",
        "details": "{素材}, {装飾}, {状態}",
        "example": "traditional Japanese machiya townhouse, "
                   "wooden lattice facade, noren curtains, "
                   "narrow Kyoto alley, stone pavement, "
                   "potted plants along the entrance"
    },
}
```

---

## 2. スタイル指定のテクニック

### コード例2: 拡張スタイルキーワード辞書

```python
STYLE_KEYWORDS = {
    "フォトリアル": {
        "keywords": ["photorealistic", "hyperrealistic", "RAW photo",
                     "8K UHD", "DSLR", "film grain"],
        "camera": ["Canon EOS R5", "Sony α7R V", "Nikon Z9",
                    "Hasselblad X2D", "Fujifilm GFX 100S",
                    "Leica M11"],
        "lens": ["85mm f/1.4", "35mm f/1.8", "50mm f/1.2",
                 "24-70mm f/2.8", "70-200mm f/2.8",
                 "90mm f/2.8 macro"],
        "film_stocks": ["Kodak Portra 400", "Fujifilm Pro 400H",
                        "Ilford HP5 Plus", "Kodak Ektar 100",
                        "CineStill 800T"],
        "techniques": ["shallow depth of field", "bokeh",
                       "lens flare", "chromatic aberration",
                       "motion blur", "long exposure"],
    },
    "アニメ": {
        "keywords": ["anime style", "cel shading", "vibrant colors",
                     "detailed eyes"],
        "substyles": {
            "新海誠風": "Makoto Shinkai style, lens flare, "
                       "vivid sky, detailed clouds, "
                       "photorealistic backgrounds",
            "ジブリ風": "Studio Ghibli style, Hayao Miyazaki, "
                       "hand-painted backgrounds, warm colors, "
                       "whimsical atmosphere",
            "サイバーパンクアニメ": "cyberpunk anime, neon lights, "
                                   "dark atmosphere, futuristic city, "
                                   "holographic displays",
            "少女漫画風": "shoujo manga style, sparkle effects, "
                         "soft pastel colors, floral backgrounds, "
                         "detailed eyes with highlights",
            "90年代アニメ": "90s anime style, retro anime, "
                           "VHS grain, cel animation, "
                           "Yoshiaki Kawajiri style",
        },
        "quality_boosters": ["detailed anime illustration",
                             "key visual", "official art",
                             "anime masterpiece"],
    },
    "油絵": {
        "keywords": ["oil painting", "canvas texture",
                     "visible brushstrokes", "impasto technique"],
        "artists_by_era": {
            "ルネサンス": ["Leonardo da Vinci", "Raphael",
                          "Michelangelo"],
            "バロック": ["Rembrandt", "Vermeer", "Caravaggio"],
            "印象派": ["Claude Monet", "Pierre-Auguste Renoir",
                       "Edgar Degas", "Camille Pissarro"],
            "後期印象派": ["Vincent van Gogh", "Paul Cézanne",
                          "Paul Gauguin", "Georges Seurat"],
            "現代": ["David Hockney", "Gerhard Richter"],
        },
        "techniques": ["alla prima", "glazing", "scumbling",
                       "palette knife", "wet-on-wet"],
    },
    "水彩画": {
        "keywords": ["watercolor painting", "soft edges",
                     "color bleeding", "wet-on-wet technique"],
        "effects": ["にじみ", "透明感", "紙のテクスチャ",
                    "granulation", "backwash"],
        "paper_types": ["cold press", "hot press",
                        "rough texture", "Arches paper"],
        "techniques": ["wet-on-wet", "wet-on-dry", "dry brush",
                       "lifting", "salt texture", "splatter"],
    },
    "3Dレンダリング": {
        "keywords": ["3D render", "octane render",
                     "unreal engine 5", "ray tracing"],
        "software": ["Blender Cycles", "Cinema 4D Redshift",
                     "KeyShot", "V-Ray", "Arnold"],
        "materials": ["subsurface scattering", "PBR materials",
                      "metallic", "glass", "translucent",
                      "iridescent"],
        "lighting_setups": ["HDRI environment", "three-point lighting",
                            "global illumination", "caustics",
                            "volumetric lighting"],
    },
    "コンセプトアート": {
        "keywords": ["concept art", "digital painting",
                     "matte painting", "detailed illustration"],
        "use_cases": {
            "ゲーム": "game concept art, character design sheet, "
                      "environment concept, prop design",
            "映画": "film concept art, VFX pre-visualization, "
                    "production design, storyboard quality",
            "書籍表紙": "book cover illustration, epic composition, "
                        "dramatic lighting, narrative scene",
        },
        "industry_artists": ["Craig Mullins", "Feng Zhu",
                             "Syd Mead", "Ralph McQuarrie",
                             "Sparth", "Maciej Kuciara"],
    },
    "ピクセルアート": {
        "keywords": ["pixel art", "16-bit", "retro game style",
                     "sprite art"],
        "resolutions": {
            "8-bit": "NES style, 8-bit, limited palette",
            "16-bit": "SNES style, 16-bit, detailed sprites",
            "32-bit": "PS1 style, early 3D, low poly",
            "modern": "modern pixel art, high detail, "
                      "smooth animation, HD pixels",
        },
    },
    "イソメトリック": {
        "keywords": ["isometric view", "isometric art",
                     "diorama style", "miniature scene"],
        "applications": ["ゲームアセット", "建築図解",
                         "インフォグラフィック", "都市計画"],
    },
}


def suggest_style_prompt(style_name: str,
                          substyle: str = None) -> str:
    """スタイルに合ったキーワードを提案"""
    style = STYLE_KEYWORDS.get(style_name, {})
    keywords = style.get("keywords", [])

    # サブスタイルがある場合
    if substyle:
        substyles = style.get("substyles", {})
        if isinstance(substyles, dict) and substyle in substyles:
            return substyles[substyle]
        elif isinstance(substyles, list) and substyle in substyles:
            return f"{', '.join(keywords[:3])}, {substyle}"

    return ", ".join(keywords[:4])


def build_style_combination(*styles: str) -> str:
    """複数スタイルの融合プロンプトを構築"""
    combined = []
    for s in styles:
        kw = STYLE_KEYWORDS.get(s, {}).get("keywords", [])
        combined.extend(kw[:2])
    return ", ".join(combined)


# 使用例
print(suggest_style_prompt("フォトリアル"))
# → "photorealistic, hyperrealistic, RAW photo, 8K UHD"

print(suggest_style_prompt("アニメ", "新海誠風"))
# → "Makoto Shinkai style, lens flare, vivid sky, ..."

print(build_style_combination("水彩画", "アニメ"))
# → "watercolor painting, soft edges, anime style, cel shading"
```

### コード例3: 構図キーワードの体系

```python
COMPOSITION_GUIDE = {
    "カメラアングル": {
        "俯瞰 (バーズアイ)": {
            "prompt": "bird's eye view, top-down perspective",
            "use_case": "風景全体、地図的表現、パターン強調",
            "emotion": "客観的、俯瞰的、神の視点",
        },
        "ローアングル": {
            "prompt": "low angle shot, worm's eye view",
            "use_case": "建物の威圧感、人物の力強さ",
            "emotion": "畏怖、尊厳、パワー",
        },
        "アイレベル": {
            "prompt": "eye level, straight-on view",
            "use_case": "自然な視線、ポートレート",
            "emotion": "等身大、親近感、共感",
        },
        "ダッチアングル": {
            "prompt": "dutch angle, tilted camera",
            "use_case": "不安定感、緊張感の演出",
            "emotion": "不安、混乱、ダイナミズム",
        },
        "オーバーショルダー": {
            "prompt": "over-the-shoulder shot",
            "use_case": "対話シーン、主観的視点",
            "emotion": "没入感、会話の臨場感",
        },
        "真正面": {
            "prompt": "frontal view, symmetrical framing",
            "use_case": "建築、ポートレート、プロダクト",
            "emotion": "対峙、直接性、インパクト",
        },
    },
    "ショットサイズ": {
        "極端なクローズアップ": {
            "prompt": "extreme close-up, macro shot",
            "use_case": "テクスチャ、目、素材の詳細",
        },
        "クローズアップ": {
            "prompt": "close-up portrait, head shot",
            "use_case": "表情、感情表現",
        },
        "バストショット": {
            "prompt": "bust shot, chest up, medium close-up",
            "use_case": "SNSプロフィール、ID写真",
        },
        "ミディアム": {
            "prompt": "medium shot, waist up",
            "use_case": "上半身、ジェスチャー",
        },
        "フルショット": {
            "prompt": "full body shot, full length",
            "use_case": "衣装全体、ポーズ",
        },
        "ワイド": {
            "prompt": "wide shot, establishing shot",
            "use_case": "環境説明、スケール感",
        },
        "パノラマ": {
            "prompt": "panoramic view, ultra-wide",
            "use_case": "壮大な風景、都市全景",
        },
    },
    "照明": {
        "ゴールデンアワー": {
            "prompt": "golden hour, warm sunlight, long shadows",
            "color_temp": "3000-4000K",
            "mood": "温かみ、ノスタルジア、ロマンチック",
        },
        "ブルーアワー": {
            "prompt": "blue hour, twilight, cool ambient light",
            "color_temp": "7000-10000K",
            "mood": "静寂、神秘的、メランコリック",
        },
        "レンブラント光": {
            "prompt": "Rembrandt lighting, dramatic shadow, "
                      "triangle of light on cheek",
            "setup": "45度横、やや上方からの単灯",
            "mood": "ドラマチック、深み、古典的",
        },
        "リムライト": {
            "prompt": "rim lighting, backlit, silhouette, "
                      "edge lighting",
            "setup": "被写体の背後から光",
            "mood": "神秘的、輪郭強調、ドラマチック",
        },
        "スタジオ照明": {
            "prompt": "studio lighting, softbox, professional, "
                      "beauty lighting",
            "setup": "3灯式（キー・フィル・バック）",
            "mood": "プロフェッショナル、クリーン",
        },
        "ネオン": {
            "prompt": "neon lighting, cyberpunk glow, "
                      "colorful neon reflections",
            "color_temp": "多色（ピンク、ブルー、パープル）",
            "mood": "未来的、都会的、エネルギッシュ",
        },
        "キアロスクーロ": {
            "prompt": "chiaroscuro, dramatic contrast, "
                      "deep shadows, single light source",
            "setup": "単一の強い指向性光源",
            "mood": "劇的、緊張感、芸術的",
        },
        "フラットライト": {
            "prompt": "flat lighting, even illumination, "
                      "no harsh shadows, overcast daylight",
            "setup": "曇天や大きなディフューザー",
            "mood": "柔らかい、均一、ファッション誌風",
        },
    },
    "構図法則": {
        "三分割法": {
            "prompt": "rule of thirds composition",
            "description": "画面を3x3に分割、交点に主題を配置",
        },
        "黄金比": {
            "prompt": "golden ratio, fibonacci spiral",
            "description": "1:1.618の比率、螺旋の焦点に主題",
        },
        "対称": {
            "prompt": "symmetrical composition, centered",
            "description": "左右対称、安定感と秩序",
        },
        "リーディングライン": {
            "prompt": "leading lines, depth, perspective lines",
            "description": "道・川・手すりなどで視線を誘導",
        },
        "フレーム内フレーム": {
            "prompt": "frame within frame, natural framing",
            "description": "窓、アーチ、木の枝で主題を囲む",
        },
        "対角線構図": {
            "prompt": "diagonal composition, dynamic angle",
            "description": "斜めの線で動きとエネルギーを表現",
        },
        "ネガティブスペース": {
            "prompt": "negative space, minimalist composition, "
                      "vast empty space",
            "description": "余白を活かして主題を際立たせる",
        },
    },
}


def build_composition_prompt(angle: str = None, shot: str = None,
                              light: str = None, rule: str = None,
                              detailed: bool = False) -> str:
    """
    構図プロンプトを組み立てる

    Args:
        angle: カメラアングル名
        shot: ショットサイズ名
        light: 照明名
        rule: 構図法則名
        detailed: Trueの場合、詳細情報も含める
    """
    parts = []
    for category, key in [
        ("カメラアングル", angle),
        ("ショットサイズ", shot),
        ("照明", light),
        ("構図法則", rule),
    ]:
        if key and key in COMPOSITION_GUIDE.get(category, {}):
            entry = COMPOSITION_GUIDE[category][key]
            if isinstance(entry, dict):
                parts.append(entry["prompt"])
            else:
                parts.append(entry)
    return ", ".join(parts)


# 使用例
comp = build_composition_prompt(
    angle="ローアングル",
    shot="ワイド",
    light="ゴールデンアワー",
    rule="黄金比"
)
print(comp)
# → "low angle shot, worm's eye view, wide shot, establishing shot,
#     golden hour, warm sunlight, long shadows,
#     golden ratio, fibonacci spiral"
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

フレーム内フレーム:          ネガティブスペース:
┌─────────────┐              ┌─────────────┐
│  ┌───────┐  │              │             │
│  │       │  │              │             │
│  │   ●   │  │  ← 窓や     │         ●   │  ← 広い余白で
│  │       │  │    アーチで  │             │    主題を
│  └───────┘  │    囲む      │             │    際立たせる
│             │              │             │
└─────────────┘              └─────────────┘

対角線構図:                  三角構図:
┌─────────────┐              ┌─────────────┐
│ ●         / │              │      ●      │
│  \       /  │              │     / \     │
│   \     /   │              │    /   \    │
│    \   /    │              │   /     \   │
│     \ /     │              │  ●───────●  │
│      ●      │              │  安定感     │
└─────────────┘              └─────────────┘
動き、エネルギー             バランス、安定
```

### 2.1 色彩理論とプロンプト

色彩はプロンプトで明示的に指定することで、画像の雰囲気を大きく制御できる。

```python
COLOR_PALETTES = {
    "暖色系": {
        "prompt": "warm color palette, reds, oranges, yellows, "
                  "warm tones, cozy atmosphere",
        "mood": "活力、情熱、温かみ、親密さ",
        "hex_samples": ["#FF6B6B", "#FFA07A", "#FFD700",
                        "#FF8C00", "#DC143C"],
    },
    "寒色系": {
        "prompt": "cool color palette, blues, teals, purples, "
                  "cool tones, serene atmosphere",
        "mood": "静寂、知性、信頼、落ち着き",
        "hex_samples": ["#4169E1", "#00CED1", "#6A5ACD",
                        "#4682B4", "#008B8B"],
    },
    "パステル": {
        "prompt": "pastel color palette, soft colors, muted tones, "
                  "light and airy, gentle hues",
        "mood": "優しさ、繊細、夢幻、軽やかさ",
        "hex_samples": ["#FFB6C1", "#B0E0E6", "#DDA0DD",
                        "#98FB98", "#FFDAB9"],
    },
    "モノクロ": {
        "prompt": "monochrome, black and white, grayscale, "
                  "high contrast, noir",
        "mood": "クラシック、ドラマチック、時代超越",
        "hex_samples": ["#000000", "#333333", "#666666",
                        "#999999", "#FFFFFF"],
    },
    "アースカラー": {
        "prompt": "earth tones, natural colors, browns, greens, "
                  "muted orange, organic palette",
        "mood": "自然、落ち着き、オーガニック、素朴",
        "hex_samples": ["#8B4513", "#556B2F", "#D2691E",
                        "#BDB76B", "#A0522D"],
    },
    "ネオン/サイバー": {
        "prompt": "neon colors, vibrant magenta, electric blue, "
                  "fluorescent green, dark background, glow",
        "mood": "未来的、都会的、活気、テクノロジー",
        "hex_samples": ["#FF00FF", "#00FFFF", "#39FF14",
                        "#FF6EC7", "#7B68EE"],
    },
    "和風配色": {
        "prompt": "traditional Japanese colors, wabi-sabi palette, "
                  "muted indigo, deep vermillion, moss green",
        "mood": "伝統、侘び寂び、優雅、和の美意識",
        "hex_samples": ["#264348", "#C53D43", "#7B8D42",
                        "#F6C555", "#5B4F3E"],
        "named_colors": {
            "藍色": "#264348",
            "紅色": "#C53D43",
            "萌黄": "#7B8D42",
            "山吹": "#F6C555",
            "海松茶": "#5B4F3E",
        },
    },
}


def get_color_prompt(palette: str,
                      accent: str = None) -> str:
    """カラーパレットプロンプトを取得"""
    p = COLOR_PALETTES.get(palette, {})
    prompt = p.get("prompt", "")
    if accent:
        prompt += f", accent color: {accent}"
    return prompt
```

---

## 3. ネガティブプロンプト戦略

### コード例4: ネガティブプロンプトテンプレート（拡張版）

```python
NEGATIVE_PROMPT_TEMPLATES = {
    "汎用 (高品質化)": (
        "low quality, worst quality, blurry, out of focus, "
        "jpeg artifacts, compression artifacts, watermark, "
        "text, signature, username, logo, "
        "poorly rendered, amateur, unprofessional"
    ),
    "人物写真": (
        "deformed, ugly, bad anatomy, bad proportions, "
        "extra limbs, extra fingers, mutated hands, "
        "poorly drawn hands, poorly drawn face, "
        "disfigured, gross proportions, long neck, "
        "cross-eyed, malformed limbs, "
        "missing arms, missing legs, extra arms, extra legs, "
        "fused fingers, too many fingers, "
        "cloned face, duplicate, morbid"
    ),
    "風景写真": (
        "oversaturated, HDR artifacts, chromatic aberration, "
        "lens flare, overexposed, underexposed, "
        "person, people, human, text, watermark, "
        "power lines, trash, litter, construction"
    ),
    "アニメ/イラスト": (
        "3d, realistic, photographic, bad anatomy, "
        "bad hands, missing fingers, extra digit, "
        "fewer digits, cropped, worst quality, "
        "low quality, normal quality, "
        "username, text, error, missing arms"
    ),
    "建築/インテリア": (
        "people, furniture out of place, distorted walls, "
        "unrealistic proportions, floating objects, "
        "bad perspective, warped lines, "
        "construction equipment, debris, clutter"
    ),
    "プロダクト写真": (
        "background clutter, shadows on product, "
        "reflections, fingerprints, dust, scratches, "
        "text overlay, watermark, low resolution, "
        "motion blur, off-center, tilted"
    ),
    "食品写真": (
        "unappetizing, overcooked, burnt, raw, spoiled, "
        "artificial, plastic looking, "
        "dirty plate, messy table, hands, utensils in wrong place, "
        "flash reflection, harsh shadows"
    ),
}

# モデル別ネガティブプロンプトの推奨設定
MODEL_NEGATIVE_DEFAULTS = {
    "sd15": {
        "always_include": (
            "lowres, bad anatomy, bad hands, text, error, "
            "missing fingers, extra digit, fewer digits, "
            "cropped, worst quality, low quality, "
            "normal quality, jpeg artifacts, signature, "
            "watermark, username, blurry"
        ),
        "strength": "強力に作用（必須）",
    },
    "sdxl": {
        "always_include": (
            "low quality, worst quality, blurry, "
            "watermark, text, logo"
        ),
        "strength": "SD1.5より効きが弱い（品質が元々高い）",
        "note": "SDXLはネガティブを減らす方が良い場合も多い",
    },
    "sd3": {
        "always_include": "",
        "strength": "ネガティブプロンプト非対応",
        "note": "SD3はネガティブプロンプトをサポートしない",
    },
    "flux": {
        "always_include": "",
        "strength": "ネガティブプロンプト非対応",
        "note": "Fluxはネガティブプロンプトをサポートしない。"
                "ポジティブプロンプトで品質を指示する",
    },
}


def get_negative_prompt(category: str,
                         model: str = "sdxl",
                         custom_exclusions: list = None) -> str:
    """
    カテゴリとモデルに応じたネガティブプロンプトを生成

    Args:
        category: テンプレートカテゴリ
        model: 使用モデル
        custom_exclusions: カスタム除外キーワード
    """
    # モデルがネガティブ非対応の場合
    model_info = MODEL_NEGATIVE_DEFAULTS.get(model, {})
    if not model_info.get("always_include") and model in ["sd3", "flux"]:
        return f"[{model}はネガティブプロンプト非対応]"

    parts = []
    # モデルデフォルト
    if model_info.get("always_include"):
        parts.append(model_info["always_include"])
    # カテゴリテンプレート
    if category in NEGATIVE_PROMPT_TEMPLATES:
        parts.append(NEGATIVE_PROMPT_TEMPLATES[category])
    # カスタム除外
    if custom_exclusions:
        parts.append(", ".join(custom_exclusions))

    # 重複を除去
    all_tags = ", ".join(parts).split(", ")
    unique_tags = list(dict.fromkeys(all_tags))
    return ", ".join(unique_tags)


# 使用例
neg = get_negative_prompt("人物写真", "sdxl",
                           ["nsfw", "child", "cartoon"])
print(neg)
```

### 3.1 ネガティブプロンプトの高度なテクニック

```python
class NegativePromptOptimizer:
    """
    ネガティブプロンプトの効果を最大化するための最適化ツール

    原則:
    1. 具体的な除外 > 抽象的な除外
    2. モデルの弱点に合わせた除外
    3. 過剰な除外は逆効果
    """

    def __init__(self, model: str = "sdxl"):
        self.model = model
        self.exclusions: list[str] = []
        self.priority_map: dict[str, int] = {}

    def add_anatomical_fix(self, body_part: str = "hands"):
        """人体の解剖学的問題を修正"""
        fixes = {
            "hands": [
                "bad hands", "mutated hands", "extra fingers",
                "fused fingers", "too many fingers",
                "missing fingers", "deformed fingers",
                "poorly drawn hands", "malformed hands",
            ],
            "face": [
                "bad face", "ugly face", "deformed face",
                "cross-eyed", "asymmetric eyes",
                "malformed ears", "double chin",
                "poorly drawn face",
            ],
            "body": [
                "bad anatomy", "bad proportions",
                "extra limbs", "missing limbs",
                "long neck", "short neck",
                "deformed body", "twisted torso",
            ],
            "feet": [
                "bad feet", "extra toes", "missing toes",
                "deformed feet", "poorly drawn feet",
            ],
        }
        self.exclusions.extend(fixes.get(body_part, []))
        return self

    def add_quality_guard(self, level: str = "standard"):
        """品質ガードを追加"""
        guards = {
            "minimal": ["low quality", "blurry"],
            "standard": [
                "low quality", "worst quality", "blurry",
                "jpeg artifacts", "watermark", "text",
            ],
            "aggressive": [
                "low quality", "worst quality", "blurry",
                "jpeg artifacts", "watermark", "text",
                "logo", "signature", "username",
                "normal quality", "amateur",
                "poorly rendered", "bad composition",
            ],
        }
        self.exclusions.extend(guards.get(level, guards["standard"]))
        return self

    def add_style_guard(self, unwanted_style: str):
        """不要なスタイルを除外"""
        style_negatives = {
            "photorealistic": ["3d render", "realistic",
                               "photograph", "DSLR"],
            "anime": ["anime", "cartoon", "manga",
                      "cel shading", "illustration"],
            "3d": ["3d", "3d render", "CGI", "octane render"],
            "painting": ["painting", "oil painting",
                         "watercolor", "canvas texture"],
        }
        self.exclusions.extend(
            style_negatives.get(unwanted_style, [])
        )
        return self

    def build(self) -> str:
        """重複除去して最終ネガティブプロンプトを構築"""
        unique = list(dict.fromkeys(self.exclusions))
        return ", ".join(unique)


# 使用例: 人物ポートレートのネガティブ最適化
neg_optimizer = (
    NegativePromptOptimizer(model="sdxl")
    .add_quality_guard("standard")
    .add_anatomical_fix("hands")
    .add_anatomical_fix("face")
    .add_style_guard("3d")
)
print(neg_optimizer.build())
```

### コード例5: プロンプト重み付け（SD系）

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

Midjourney の重み付け:
  subject:: 2 details:: 1  → subjectに2倍の重み
  --iw 0.5                 → 画像プロンプトの重み
"""

import re
from dataclasses import dataclass


@dataclass
class WeightedToken:
    """重み付きトークン"""
    text: str
    weight: float = 1.0

    def to_sd_syntax(self) -> str:
        """SD系の重み付け構文に変換"""
        if self.weight == 1.0:
            return self.text
        return f"({self.text}:{self.weight})"

    def to_mj_syntax(self) -> str:
        """Midjourney の重み付け構文に変換"""
        if self.weight == 1.0:
            return self.text
        return f"{self.text}::{self.weight}"

    def to_natural_language(self) -> str:
        """DALL-E 3 向けの自然言語強調に変換"""
        if self.weight > 1.3:
            return f"prominently featuring {self.text}"
        elif self.weight > 1.0:
            return f"with emphasis on {self.text}"
        elif self.weight < 0.7:
            return f"with subtle {self.text} in the background"
        elif self.weight < 1.0:
            return f"with slight {self.text}"
        return self.text


class PromptWeightManager:
    """プロンプトの重み付けを管理"""

    def __init__(self, model: str = "sdxl"):
        self.model = model
        self.tokens: list[WeightedToken] = []

    def add(self, text: str, weight: float = 1.0):
        """重み付きトークンを追加"""
        self.tokens.append(WeightedToken(text, weight))
        return self

    def build(self) -> str:
        """モデルに合わせた重み付けプロンプトを構築"""
        if self.model in ["sd15", "sdxl"]:
            return ", ".join(t.to_sd_syntax() for t in self.tokens)
        elif self.model == "midjourney":
            return " ".join(t.to_mj_syntax() for t in self.tokens)
        elif self.model == "dalle3":
            return ". ".join(t.to_natural_language()
                            for t in self.tokens)
        else:
            return ", ".join(t.text for t in self.tokens)

    def analyze_weights(self) -> dict:
        """重み分布を分析"""
        weights = [t.weight for t in self.tokens]
        return {
            "total_tokens": len(weights),
            "max_weight": max(weights) if weights else 0,
            "min_weight": min(weights) if weights else 0,
            "avg_weight": sum(weights) / len(weights) if weights else 0,
            "emphasized": sum(1 for w in weights if w > 1.0),
            "de_emphasized": sum(1 for w in weights if w < 1.0),
            "warning": ("重みの差が大きすぎます"
                       if weights and max(weights) - min(weights) > 1.5
                       else None),
        }


# SD系での使用例
sd_prompt = (
    PromptWeightManager(model="sdxl")
    .add("beautiful landscape", 1.0)
    .add("cherry blossoms", 1.4)   # 桜を強調
    .add("mount fuji", 1.2)        # 富士山をやや強調
    .add("sunset", 0.8)            # 夕日を少し抑える
    .add("dramatic clouds", 1.1)   # 雲をわずかに強調
)
print("SD:", sd_prompt.build())
# → "beautiful landscape, (cherry blossoms:1.4),
#    (mount fuji:1.2), (sunset:0.8), (dramatic clouds:1.1)"

# DALL-E 3での使用例
dalle_prompt = (
    PromptWeightManager(model="dalle3")
    .add("beautiful landscape", 1.0)
    .add("cherry blossoms", 1.4)
    .add("mount fuji", 1.2)
    .add("sunset", 0.8)
)
print("DALL-E:", dalle_prompt.build())
# → "beautiful landscape. prominently featuring cherry blossoms.
#    with emphasis on mount fuji.
#    with slight sunset"

print(sd_prompt.analyze_weights())
```

### 3.2 Compel によるSD系の高精度重み付け

```python
"""
Compel: diffusers公式の重み付けライブラリ

手動の重み構文よりも正確なトークン重み制御を提供。
CLIP埋め込みレベルで直接重みを操作する。
"""

from diffusers import StableDiffusionXLPipeline
# from compel import Compel, ReturnedEmbeddingsType
import torch


def generate_with_compel_weights():
    """Compelを使った高精度プロンプト重み付け"""

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # Compelインスタンスの作成（SDXL用）
    from compel import Compel, ReturnedEmbeddingsType

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=(
            ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
        ),
        requires_pooled=[False, True],
    )

    # Compel構文でプロンプトを記述
    # +/- で重み調整、++ は 1.1^2 = 1.21倍
    prompt = (
        "a beautiful Japanese garden++ with cherry blossoms+++, "
        "koi pond+, stone lantern, "
        "golden hour lighting++, "
        "photorealistic+, 8K resolution"
    )

    negative_prompt = (
        "low quality--, blurry-, watermark-, text-"
    )

    # 埋め込みに変換
    conditioning, pooled = compel(prompt)
    neg_conditioning, neg_pooled = compel(negative_prompt)

    # 生成
    image = pipe(
        prompt_embeds=conditioning,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=neg_conditioning,
        negative_pooled_prompt_embeds=neg_pooled,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    return image


def compel_prompt_blending():
    """
    Compelによるプロンプトブレンディング

    2つのプロンプトを任意の比率で混合できる。
    スタイル転送やコンセプトミックスに有用。
    """
    # ブレンド構文の例
    blended_prompts = {
        # 50:50のブレンド
        "equal_blend": (
            '("oil painting of a castle",'
            ' "watercolor of a castle").blend(0.5, 0.5)'
        ),
        # 70:30 — 油絵寄り
        "oil_dominant": (
            '("oil painting of a castle",'
            ' "watercolor of a castle").blend(0.7, 0.3)'
        ),
        # コンセプトのブレンド
        "concept_blend": (
            '("a cat", "a dog").blend(0.6, 0.4)'
        ),
        # 3つのプロンプトをブレンド
        "triple_blend": (
            '("sunset", "aurora", "starry night")'
            '.blend(0.5, 0.3, 0.2)'
        ),
    }

    # 連結構文 (.and()) — 各プロンプトを独立に処理して結合
    conjunction_prompts = {
        "scene_composition": (
            '("a red sports car", "rainy city street at night")'
            '.and()'
        ),
        "weighted_conjunction": (
            '("a red sports car", 1.5, '
            '"rainy city street at night", 0.8)'
            '.and()'
        ),
    }

    return blended_prompts, conjunction_prompts
```

### ASCII図解3: モデル別プロンプト最適化マップ

```
┌────────── Stable Diffusion 1.5 ────────┐
│ ・重み付け構文 (keyword:1.5) が使える    │
│ ・ネガティブプロンプトが効果大           │
│ ・タグベース + 自然文のハイブリッド       │
│ ・LoRA/テキスト反転でカスタム語彙追加    │
│ ・推奨: 短いタグの羅列 + 品質キーワード  │
│ ・上限: 75トークン (CLIP ViT-L/14)      │
│ ・Compel推奨: 重み精度が高い            │
└──────────────────────────────────────────┘

┌────────── Stable Diffusion XL ──────────┐
│ ・デュアルCLIPエンコーダ (OpenCLIP G/14) │
│ ・150トークンまで対応                    │
│ ・ネガティブは控えめが良い場合あり       │
│ ・解像度条件付け: target_size指定可能    │
│ ・リファイナーとの組み合わせ             │
│ ・推奨: 自然文寄り + 品質タグ少なめ     │
└──────────────────────────────────────────┘

┌────────── SD3 / Flux ──────────────────┐
│ ・T5-XXLエンコーダで長文理解が可能      │
│ ・512トークン以上対応                   │
│ ・ネガティブプロンプト非対応            │
│ ・テキスト描画が正確                    │
│ ・自然言語で詳細に記述するのが最適      │
│ ・推奨: 詳細な自然文記述               │
│ ・重み構文: 非対応（文章で表現）        │
└─────────────────────────────────────────┘

┌────────── DALL-E 3 ─────────────────────┐
│ ・自然言語で詳細に記述                   │
│ ・重み付け構文は未サポート               │
│ ・ネガティブプロンプトなし               │
│   → 「～を含まない」と記述              │
│ ・GPT-4がプロンプトを内部で書き換え      │
│ ・revised_prompt で確認可能             │
│ ・推奨: 文章として詳細に状況を説明       │
│ ・上限: 4,000文字                       │
└──────────────────────────────────────────┘

┌────────── Midjourney ───────────────────┐
│ ・パラメータ: --ar, --v, --s, --c, --q │
│ ・--no でネガティブ指定                 │
│ ・短く印象的なプロンプトが効果的         │
│ ・:: でマルチプロンプト (重み分離)       │
│ ・--sref でスタイル参照                 │
│ ・--cref でキャラクター参照             │
│ ・推奨: キーワード + パラメータ調整      │
│ ・上限: ~350語                          │
└──────────────────────────────────────────┘
```

---

## 4. モデル固有のプロンプト最適化

### コード例6: DALL-E 3 向け自然言語プロンプト構築

```python
"""
DALL-E 3はGPT-4がプロンプトを内部で書き換えるため、
タグ羅列ではなく自然言語での詳細な記述が最も効果的。

ポイント:
1. 具体的なシーン描写（5W1H）
2. 「～しない」形式でネガティブ要素を表現
3. スタイル指定は文章の冒頭か末尾に
4. revised_prompt を確認して意図とのズレを把握
"""

from openai import OpenAI


class DALLE3PromptCrafter:
    """DALL-E 3に最適化されたプロンプト構築"""

    def __init__(self):
        self.client = OpenAI()
        self.scene_elements = {}

    def set_scene(self, description: str,
                   time_of_day: str = "",
                   weather: str = "",
                   season: str = ""):
        """シーンの基本設定"""
        self.scene_elements["scene"] = description
        if time_of_day:
            self.scene_elements["time"] = time_of_day
        if weather:
            self.scene_elements["weather"] = weather
        if season:
            self.scene_elements["season"] = season
        return self

    def set_subject(self, description: str,
                     action: str = "",
                     appearance: str = ""):
        """主題の設定"""
        self.scene_elements["subject"] = description
        if action:
            self.scene_elements["action"] = action
        if appearance:
            self.scene_elements["appearance"] = appearance
        return self

    def set_style(self, style: str,
                   art_direction: str = ""):
        """スタイル設定"""
        self.scene_elements["style"] = style
        if art_direction:
            self.scene_elements["art_direction"] = art_direction
        return self

    def set_exclusions(self, *exclusions: str):
        """除外要素（DALL-E 3は自然言語で記述）"""
        self.scene_elements["exclusions"] = list(exclusions)
        return self

    def build_natural_prompt(self) -> str:
        """自然言語プロンプトを構築"""
        parts = []

        # スタイル指定（冒頭）
        if "style" in self.scene_elements:
            parts.append(
                f"Create a {self.scene_elements['style']} image."
            )
            if "art_direction" in self.scene_elements:
                parts.append(self.scene_elements["art_direction"])

        # シーン描写
        if "scene" in self.scene_elements:
            scene = self.scene_elements["scene"]
            time_info = self.scene_elements.get("time", "")
            weather_info = self.scene_elements.get("weather", "")
            season_info = self.scene_elements.get("season", "")

            scene_desc = f"The scene depicts {scene}"
            if time_info:
                scene_desc += f" during {time_info}"
            if season_info:
                scene_desc += f" in {season_info}"
            if weather_info:
                scene_desc += f", with {weather_info}"
            parts.append(scene_desc + ".")

        # 主題描写
        if "subject" in self.scene_elements:
            subject = self.scene_elements["subject"]
            action = self.scene_elements.get("action", "")
            appearance = self.scene_elements.get("appearance", "")

            subject_desc = f"The main subject is {subject}"
            if appearance:
                subject_desc += f", {appearance}"
            if action:
                subject_desc += f", {action}"
            parts.append(subject_desc + ".")

        # 除外要素
        if "exclusions" in self.scene_elements:
            exclusion_text = ", ".join(
                self.scene_elements["exclusions"]
            )
            parts.append(
                f"The image should not contain {exclusion_text}."
            )

        return " ".join(parts)

    def generate(self, size: str = "1024x1024",
                  quality: str = "hd",
                  style: str = "natural") -> dict:
        """画像を生成"""
        prompt = self.build_natural_prompt()

        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,  # "natural" or "vivid"
            n=1,
        )

        return {
            "url": response.data[0].url,
            "original_prompt": prompt,
            "revised_prompt": response.data[0].revised_prompt,
            "prompt_drift": self._calculate_drift(
                prompt, response.data[0].revised_prompt
            ),
        }

    def _calculate_drift(self, original: str,
                          revised: str) -> dict:
        """元プロンプトと改変プロンプトの差異を分析"""
        orig_words = set(original.lower().split())
        rev_words = set(revised.lower().split())

        added = rev_words - orig_words
        removed = orig_words - rev_words

        return {
            "similarity": len(orig_words & rev_words)
                         / len(orig_words | rev_words),
            "words_added": len(added),
            "words_removed": len(removed),
            "top_additions": list(added)[:10],
        }


# 使用例
crafter = (
    DALLE3PromptCrafter()
    .set_scene(
        "a serene Japanese garden with a koi pond",
        time_of_day="golden hour",
        season="autumn",
        weather="light mist rising from the water"
    )
    .set_subject(
        "an ancient stone lantern",
        appearance="covered in moss, weathered by centuries",
    )
    .set_style(
        "photorealistic",
        art_direction="Shot with a medium format camera, "
                      "shallow depth of field, warm tones, "
                      "the composition follows the golden ratio"
    )
    .set_exclusions("people", "modern objects", "text")
)

print(crafter.build_natural_prompt())
```

### コード例7: Midjourney プロンプト構築

```python
class MidjourneyPromptBuilder:
    """
    Midjourney V6+ 向けプロンプト構築

    V6の特徴:
    - より自然な言語理解
    - --sref でスタイル参照
    - --cref でキャラクター参照
    - --p でパーソナライゼーション
    - --sv でスタイルバリエーション
    """

    ASPECT_RATIOS = {
        "正方形": "1:1",
        "横長": "16:9",
        "横長ワイド": "21:9",
        "縦長": "9:16",
        "ポートレート": "2:3",
        "ランドスケープ": "3:2",
        "映画": "2.39:1",
        "SNSストーリー": "9:16",
        "SNS投稿": "4:5",
    }

    STYLE_PRESETS = {
        "写真": {
            "keywords": ["photograph", "realistic"],
            "recommended_params": {"--s": 100, "--q": 1},
        },
        "イラスト": {
            "keywords": ["illustration", "digital art"],
            "recommended_params": {"--s": 250, "--q": 1},
        },
        "アニメ": {
            "keywords": ["anime style", "--niji 6"],
            "recommended_params": {"--s": 200},
        },
        "油絵": {
            "keywords": ["oil painting", "canvas texture"],
            "recommended_params": {"--s": 300, "--c": 20},
        },
        "ミニマル": {
            "keywords": ["minimalist", "clean", "simple"],
            "recommended_params": {"--s": 50, "--c": 0},
        },
        "ファンタジー": {
            "keywords": ["fantasy art", "magical", "epic"],
            "recommended_params": {"--s": 400, "--c": 30},
        },
    }

    def __init__(self):
        self.subject = ""
        self.style_keywords: list[str] = []
        self.params: dict[str, any] = {}
        self.multi_prompts: list[tuple[str, float]] = []
        self.no_list: list[str] = []

    def set_subject(self, subject: str):
        """主題を設定"""
        self.subject = subject
        return self

    def set_preset(self, preset_name: str):
        """スタイルプリセットを適用"""
        if preset_name in self.STYLE_PRESETS:
            preset = self.STYLE_PRESETS[preset_name]
            self.style_keywords.extend(preset["keywords"])
            self.params.update(preset.get("recommended_params", {}))
        return self

    def set_aspect_ratio(self, ratio_name: str):
        """アスペクト比を設定"""
        if ratio_name in self.ASPECT_RATIOS:
            self.params["--ar"] = self.ASPECT_RATIOS[ratio_name]
        return self

    def set_stylize(self, value: int):
        """スタイライズ値 (0-1000)"""
        self.params["--s"] = max(0, min(1000, value))
        return self

    def set_chaos(self, value: int):
        """カオス値 (0-100) — バリエーションの多様性"""
        self.params["--c"] = max(0, min(100, value))
        return self

    def set_weird(self, value: int):
        """ウィアード値 (0-3000) — 奇抜さ"""
        self.params["--weird"] = max(0, min(3000, value))
        return self

    def add_style_ref(self, url: str, weight: int = 100):
        """スタイル参照画像を追加"""
        self.params["--sref"] = url
        if weight != 100:
            self.params["--sw"] = weight
        return self

    def add_character_ref(self, url: str, weight: int = 100):
        """キャラクター参照画像を追加"""
        self.params["--cref"] = url
        if weight != 100:
            self.params["--cw"] = weight
        return self

    def add_no(self, *exclusions: str):
        """除外要素を追加"""
        self.no_list.extend(exclusions)
        return self

    def add_multi_prompt(self, text: str, weight: float = 1.0):
        """マルチプロンプト（重み付き）を追加"""
        self.multi_prompts.append((text, weight))
        return self

    def build(self) -> str:
        """最終プロンプトを構築"""
        parts = []

        # マルチプロンプトの場合
        if self.multi_prompts:
            mp_parts = []
            for text, weight in self.multi_prompts:
                if weight != 1.0:
                    mp_parts.append(f"{text}::{weight}")
                else:
                    mp_parts.append(text)
            parts.append(" ".join(mp_parts))
        else:
            # 通常プロンプト
            prompt_parts = [self.subject]
            prompt_parts.extend(self.style_keywords)
            parts.append(", ".join(p for p in prompt_parts if p))

        # --no パラメータ
        if self.no_list:
            parts.append(f"--no {', '.join(self.no_list)}")

        # その他パラメータ
        for key, value in self.params.items():
            if key != "--no":
                parts.append(f"{key} {value}")

        return " ".join(parts)


# 使用例: 和風ファンタジー
mj_prompt = (
    MidjourneyPromptBuilder()
    .set_subject("ancient Japanese shrine floating "
                  "among clouds, torii gates, "
                  "mystical atmosphere")
    .set_preset("ファンタジー")
    .set_aspect_ratio("横長")
    .set_stylize(500)
    .set_chaos(15)
    .add_no("people", "modern objects", "text")
)
print(mj_prompt.build())

# マルチプロンプトの例
mj_multi = (
    MidjourneyPromptBuilder()
    .add_multi_prompt("serene Japanese garden", 2.0)
    .add_multi_prompt("cyberpunk neon elements", 0.5)
    .set_aspect_ratio("横長")
    .set_stylize(300)
)
print(mj_multi.build())
# → "serene Japanese garden::2.0 cyberpunk neon elements::0.5
#    --ar 16:9 --s 300"
```

---

## 5. プロンプトの体系的デバッグとA/Bテスト

### コード例8: プロンプトA/Bテストフレームワーク

```python
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class PromptVariant:
    """プロンプトバリアントの記録"""
    variant_id: str
    prompt: str
    negative_prompt: str = ""
    seed: int = 42
    steps: int = 30
    cfg_scale: float = 7.5
    sampler: str = "DPM++ 2M Karras"
    score: Optional[float] = None
    notes: str = ""
    generation_time: float = 0.0


class PromptABTester:
    """
    プロンプトのA/Bテストを体系的に実行

    原則:
    1. 一度に変更する要素は1つだけ
    2. シードを固定して比較
    3. 複数シードで再現性を確認
    4. 評価基準を事前に定義
    """

    def __init__(self, base_prompt: str,
                  base_negative: str = "",
                  output_dir: str = "./ab_test_results"):
        self.base_prompt = base_prompt
        self.base_negative = base_negative
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.variants: list[PromptVariant] = []
        self.test_seeds = [42, 123, 456, 789, 1024]

    def add_variant(self, name: str,
                     prompt_modification: str = None,
                     negative_modification: str = None,
                     **kwargs) -> "PromptABTester":
        """テストバリアントを追加"""
        variant = PromptVariant(
            variant_id=name,
            prompt=prompt_modification or self.base_prompt,
            negative_prompt=(negative_modification
                            or self.base_negative),
            **kwargs
        )
        self.variants.append(variant)
        return self

    def create_comparison_grid(self) -> list[dict]:
        """比較グリッド（バリアント x シード）を生成"""
        grid = []
        for variant in self.variants:
            for seed in self.test_seeds:
                grid.append({
                    "variant": variant.variant_id,
                    "prompt": variant.prompt,
                    "negative": variant.negative_prompt,
                    "seed": seed,
                    "steps": variant.steps,
                    "cfg_scale": variant.cfg_scale,
                })
        return grid

    def analyze_results(self) -> dict:
        """テスト結果を分析"""
        scored = [v for v in self.variants if v.score is not None]
        if not scored:
            return {"error": "No scored variants"}

        ranked = sorted(scored, key=lambda v: v.score, reverse=True)
        return {
            "best_variant": ranked[0].variant_id,
            "best_score": ranked[0].score,
            "ranking": [
                {"id": v.variant_id, "score": v.score,
                 "prompt_preview": v.prompt[:80]}
                for v in ranked
            ],
            "improvement": (
                f"{((ranked[0].score - ranked[-1].score)"
                f" / ranked[-1].score * 100):.1f}%"
                if ranked[-1].score > 0 else "N/A"
            ),
        }

    def export_report(self, filepath: str = None):
        """テスト結果レポートをJSON出力"""
        if filepath is None:
            filepath = str(
                self.output_dir / "ab_test_report.json"
            )
        report = {
            "base_prompt": self.base_prompt,
            "base_negative": self.base_negative,
            "variants": [
                {
                    "id": v.variant_id,
                    "prompt": v.prompt,
                    "negative": v.negative_prompt,
                    "score": v.score,
                    "notes": v.notes,
                }
                for v in self.variants
            ],
            "analysis": self.analyze_results(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return filepath


# 使用例: 照明のA/Bテスト
tester = PromptABTester(
    base_prompt="portrait of a young woman in a garden, "
                "wearing a white dress, masterpiece, best quality",
    base_negative="low quality, blurry, deformed"
)

# 照明バリエーションをテスト
for lighting in ["golden hour sunlight",
                  "studio softbox lighting",
                  "overcast diffused light",
                  "dramatic Rembrandt lighting",
                  "neon colored lighting"]:
    tester.add_variant(
        name=f"lighting_{lighting.split()[0]}",
        prompt_modification=(
            f"portrait of a young woman in a garden, "
            f"wearing a white dress, {lighting}, "
            f"masterpiece, best quality"
        ),
    )

grid = tester.create_comparison_grid()
print(f"生成する画像数: {len(grid)}")
# → 生成する画像数: 25 (5バリアント x 5シード)
```

### 5.1 段階的プロンプト改善メソッド

```python
class IterativePromptRefiner:
    """
    段階的にプロンプトを改善するワークフロー

    ステップ:
    1. 最小プロンプト（主題のみ）で基準画像を生成
    2. スタイルを追加
    3. 品質タグを追加
    4. 構図を追加
    5. ネガティブプロンプトを追加
    6. 重み付けを調整
    """

    def __init__(self, subject: str, seed: int = 42):
        self.subject = subject
        self.seed = seed
        self.iterations: list[dict] = []
        self.current_prompt = subject
        self.current_negative = ""

    def step(self, addition: str,
              category: str = "unknown",
              to_negative: bool = False) -> dict:
        """1ステップ追加して記録"""
        if to_negative:
            if self.current_negative:
                self.current_negative += f", {addition}"
            else:
                self.current_negative = addition
        else:
            self.current_prompt += f", {addition}"

        iteration = {
            "step": len(self.iterations) + 1,
            "category": category,
            "added": addition,
            "to_negative": to_negative,
            "full_prompt": self.current_prompt,
            "full_negative": self.current_negative,
        }
        self.iterations.append(iteration)
        return iteration

    def get_comparison_prompts(self) -> list[dict]:
        """全ステップの比較用プロンプトリストを返す"""
        return [
            {
                "step": it["step"],
                "category": it["category"],
                "prompt": it["full_prompt"],
                "negative": it["full_negative"],
                "seed": self.seed,
            }
            for it in self.iterations
        ]

    def print_evolution(self):
        """プロンプトの進化を表示"""
        for it in self.iterations:
            target = "negative" if it["to_negative"] else "positive"
            print(f"Step {it['step']} [{it['category']}] "
                  f"({target}): +'{it['added']}'")
        print(f"\n最終プロンプト: {self.current_prompt}")
        print(f"最終ネガティブ: {self.current_negative}")


# 使用例
refiner = IterativePromptRefiner(
    "a majestic samurai standing on a cliff",
    seed=42
)

refiner.step("overlooking a misty valley at dawn",
              category="環境")
refiner.step("wearing ornate black and gold armor",
              category="主題詳細")
refiner.step("digital painting, concept art",
              category="スタイル")
refiner.step("masterpiece, highly detailed, 8K",
              category="品質")
refiner.step("dramatic rim lighting, volumetric fog",
              category="照明")
refiner.step("wide shot, epic composition, low angle",
              category="構図")
refiner.step("low quality, blurry, deformed, watermark",
              category="品質ガード", to_negative=True)
refiner.step("bad anatomy, extra limbs, bad proportions",
              category="解剖学", to_negative=True)

refiner.print_evolution()
```

---

## 6. 業種別プロンプトテンプレート

### コード例9: 業種別テンプレートシステム

```python
INDUSTRY_TEMPLATES = {
    "EC/プロダクト写真": {
        "template": (
            "{product}, product photography, "
            "clean white background, studio lighting, "
            "commercial photography, sharp focus, "
            "{angle}, {style_modifier}"
        ),
        "variables": {
            "product": "製品の詳細な記述",
            "angle": ["front view", "45-degree angle",
                      "flat lay", "lifestyle context"],
            "style_modifier": ["minimalist", "luxury",
                                "vibrant", "organic"],
        },
        "negative": (
            "shadows on product, reflections, dust, "
            "fingerprints, background clutter, text, watermark"
        ),
        "best_practices": [
            "製品の素材感を明確に記述する",
            "背景は製品に合わせて選択（白/グレー/コンテキスト）",
            "ライティングは製品の形状を強調するように指定",
        ],
        "examples": [
            {
                "name": "腕時計",
                "prompt": "luxury mechanical watch with silver case "
                          "and dark blue dial, product photography, "
                          "clean dark background, dramatic lighting "
                          "highlighting the case finishing, "
                          "45-degree angle, luxury, "
                          "sharp focus on dial details",
            },
            {
                "name": "スニーカー",
                "prompt": "modern white sneaker with neon accents, "
                          "product photography, clean white background, "
                          "studio lighting, floating in air, "
                          "45-degree angle, dynamic, sharp focus",
            },
        ],
    },
    "不動産/建築": {
        "template": (
            "{property_type}, {architectural_style}, "
            "architectural photography, {time_of_day}, "
            "{composition}, professional real estate photography"
        ),
        "variables": {
            "property_type": "物件タイプの記述",
            "architectural_style": ["modern", "traditional Japanese",
                                     "minimalist", "industrial"],
            "time_of_day": ["twilight exterior", "bright daylight",
                            "blue hour", "golden hour"],
            "composition": ["wide angle", "symmetrical",
                            "leading lines", "aerial view"],
        },
        "negative": (
            "people, cars, clutter, construction equipment, "
            "distorted perspective, unrealistic proportions"
        ),
    },
    "飲食/フード": {
        "template": (
            "{dish}, food photography, {plating_style}, "
            "{background}, {lighting}, appetizing, "
            "shallow depth of field"
        ),
        "variables": {
            "dish": "料理の詳細な記述",
            "plating_style": ["rustic plating", "fine dining",
                               "casual", "minimalist"],
            "background": ["dark wood table", "marble surface",
                            "rustic linen", "clean white plate"],
            "lighting": ["natural window light", "warm ambient",
                         "moody dark", "bright and airy"],
        },
        "negative": (
            "unappetizing, overcooked, artificial, "
            "plastic looking, messy, hands, utensils"
        ),
    },
    "ゲーム開発": {
        "template": (
            "{asset_type}, game art, {art_style}, "
            "{view}, {detail_level}, "
            "concept art quality"
        ),
        "variables": {
            "asset_type": ["character design", "environment concept",
                            "weapon design", "creature design",
                            "UI element", "icon set"],
            "art_style": ["stylized", "realistic", "pixel art",
                          "cel shaded", "painterly"],
            "view": ["front/side/back turnaround",
                     "3/4 view", "isometric", "top-down"],
            "detail_level": ["high poly reference",
                              "low poly style", "texture sheet"],
        },
    },
    "SNS/マーケティング": {
        "template": (
            "{content_type}, {brand_mood}, "
            "{color_scheme}, social media ready, "
            "high impact visual, {format}"
        ),
        "variables": {
            "content_type": ["hero image", "product feature",
                              "lifestyle", "quote background",
                              "story visual", "banner"],
            "brand_mood": ["professional", "playful", "luxury",
                           "eco-friendly", "tech-forward"],
            "color_scheme": "ブランドカラーに合わせて指定",
            "format": ["square 1:1", "portrait 4:5",
                       "landscape 16:9", "story 9:16"],
        },
    },
}


def generate_industry_prompt(industry: str,
                               **variables) -> dict:
    """業種テンプレートからプロンプトを生成"""
    template_data = INDUSTRY_TEMPLATES.get(industry, {})
    if not template_data:
        return {"error": f"業種 '{industry}' は未対応です"}

    template = template_data["template"]

    # 変数を埋め込み
    for key, value in variables.items():
        template = template.replace(f"{{{key}}}", str(value))

    # 未設定の変数を検出
    import re
    missing = re.findall(r'\{(\w+)\}', template)

    return {
        "prompt": template,
        "negative_prompt": template_data.get("negative", ""),
        "missing_variables": missing,
        "best_practices": template_data.get("best_practices", []),
    }


# 使用例: プロダクト写真
result = generate_industry_prompt(
    "EC/プロダクト写真",
    product="handcrafted ceramic coffee mug with speckled glaze",
    angle="45-degree angle",
    style_modifier="organic"
)
print(result["prompt"])
```

---

## 7. 比較表

### 比較表1: プロンプト構成要素の効果

| 構成要素 | 効果の大きさ | 記述例 | 影響範囲 | SD系での推奨重み |
|---------|------------|--------|---------|----------------|
| **主題** | 最大 | "赤い着物の女性" | 生成内容の核心 | 1.0 (デフォルト) |
| **スタイル** | 大 | "油絵風, 印象派" | 全体の雰囲気・質感 | 1.0-1.2 |
| **品質タグ** | 中~大 | "masterpiece, 8K" | 細部の品質 | 1.0 |
| **照明** | 中 | "ゴールデンアワー" | 色調・陰影 | 0.8-1.2 |
| **構図** | 中 | "三分割法, ローアングル" | レイアウト | 0.8-1.0 |
| **カメラ設定** | 小~中 | "85mm f/1.4, bokeh" | 被写界深度・ボケ | 0.8-1.0 |
| **ネガティブ** | 中~大 | "blurry, deformed" | 不要要素の排除 | - |
| **重み付け** | 小~中 | "(cherry:1.4)" | 特定要素の強弱 | 0.5-1.5 |
| **色彩指定** | 中 | "warm tones, golden" | 全体の色調 | 0.8-1.2 |
| **テクスチャ** | 小 | "smooth, rough, grainy" | 表面の質感 | 0.8-1.0 |

### 比較表2: モデル別プロンプト特性比較

| 特性 | SD 1.5 | SDXL | SD3 | Flux | DALL-E 3 | Midjourney V6 |
|------|--------|------|-----|------|----------|---------------|
| **推奨記述** | タグ羅列 | タグ+自然文 | 自然文 | 自然文 | 詳細な自然文 | 短いキーワード |
| **最大長** | 75トークン | 150トークン | 256トークン | 512トークン | 4000文字 | ~350語 |
| **重み構文** | (keyword:1.5) | (keyword:1.5) | 限定的 | なし | なし | :: |
| **ネガティブ** | 強力 | やや強力 | なし | なし | なし (文中記述) | --no |
| **日本語** | モデル依存 | モデル依存 | 部分対応 | 部分対応 | 対応 | 限定的 |
| **テキスト描画** | 苦手 | やや苦手 | 得意 | 得意 | 得意 | やや改善 |
| **Compel対応** | あり | あり | なし | なし | なし | なし |
| **スタイル参照** | LoRA/IP-Adapter | LoRA/IP-Adapter | なし | なし | なし | --sref |
| **品質タグ効果** | 大 | 中 | 小 | 小 | なし | --q |

### 比較表3: 照明タイプ別の効果と適用シーン

| 照明タイプ | 色温度 | ムード | 適用シーン | プロンプト例 |
|-----------|--------|--------|-----------|-------------|
| ゴールデンアワー | 3000-4000K | 温かい、ノスタルジック | ポートレート、風景 | golden hour, warm sunlight |
| ブルーアワー | 7000-10000K | 静寂、神秘的 | 都市、風景 | blue hour, twilight |
| レンブラント光 | - | ドラマチック | ポートレート | Rembrandt lighting |
| リムライト | - | 神秘的、輪郭強調 | シルエット、ドラマ | rim lighting, backlit |
| スタジオ照明 | 5500K | プロフェッショナル | 製品、ファッション | studio lighting, softbox |
| ネオン | 多色 | 未来的、都会的 | サイバーパンク | neon lighting, glow |
| キアロスクーロ | - | 劇的、緊張感 | ドラマ、古典 | chiaroscuro, deep shadows |
| フラットライト | 6500K | 柔らかい、均一 | ファッション、商品 | flat lighting, even |

---

## 8. アンチパターン

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
- SDXL以降のモデルでは品質タグの効果自体が小さい

[正しいアプローチ]
- 品質タグは3-5個に絞る
- 主題と構図の記述に十分なトークンを確保
- テスト生成で効果を確認し、不要なタグを除去
- モデルのベースライン品質を信頼する
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
- SD系: タグ + 重み付け + Compel
- DALL-E 3: 自然言語の詳細な文章
- Midjourney: 短く印象的 + パラメータ
- Flux/SD3: 詳細な自然文 (ネガティブなし)
```

### アンチパターン3: ネガティブプロンプトの過信

```
[問題]
ネガティブプロンプトに100語以上の除外指定を列挙する。

[なぜ問題か]
- ネガティブプロンプトにもトークン制限がある
- 過剰な否定は逆に不要な概念を活性化させる可能性
- SD3/Fluxではネガティブプロンプト自体が非対応
- ネガティブに依存するよりポジティブで品質を上げるべき

[正しいアプローチ]
- ネガティブは最も頻出する問題に絞る（10-20語）
- まずポジティブプロンプトの品質を上げる
- モデルごとのネガティブ対応状況を確認
- 段階的にネガティブを追加して効果を確認
```

### アンチパターン4: シード固定なしの比較

```
[問題]
プロンプトを変更しながらランダムシードで生成し、
「このプロンプトの方が良い」と判断する。

[なぜ問題か]
- シードの違いによる差なのか、プロンプトの差なのか不明
- 同じプロンプトでもシードで大きく結果が変わる
- 再現性がなく、知見として蓄積できない

[正しいアプローチ]
- A/Bテスト時は必ずシードを固定
- 複数シード（最低5つ）で傾向を確認
- 変更する要素は一度に1つだけ
- 結果を記録して比較可能にする
```

### アンチパターン5: コンテキスト無視のスタイル指定

```
[問題]
「oil painting, watercolor, digital art, anime style,
 photorealistic」のように矛盾するスタイルを混在させる。

[なぜ問題か]
- 矛盾するスタイル指定はモデルを混乱させる
- どのスタイルも中途半端な結果になる
- 意図的なスタイルミックスと無秩序な混在は異なる

[正しいアプローチ]
- 1つの主要スタイルを選択する
- スタイルミックスは意図的に比率を制御する
  (Compelのblend、MJの::で重み指定)
- テスト生成でスタイルの一貫性を確認
```

---

## 9. FAQ

### Q1: 日本語プロンプトと英語プロンプト、どちらが効果的?

**A:** 一般的に**英語プロンプトが推奨**されます:

- **SD系:** 学習データが英語中心のため、英語の方が高精度
- **DALL-E 3:** 日本語対応だが、英語の方がニュアンスが正確。日本語で入力してもGPT-4が内部で英語に変換する場合がある
- **Midjourney:** 英語のみ正式対応
- **Flux:** T5エンコーダで多言語対応だが、英語がベスト
- **例外:** 日本語特化モデル (Animagine XL、SDXL-Lightning JP等) は日本語タグが有効
- **ハイブリッド戦略:** 日本語で構想 → 英語に翻訳 → モデルに入力が実践的

### Q2: プロンプトの長さはどのくらいが最適?

**A:** モデルとタスクによります:

- **SD 1.5:** 20-40トークン (約75トークン上限)。短い方がトークンあたりの影響が大きい
- **SDXL:** 40-80トークン (150トークン上限)。SD1.5より長めでも安定
- **SD3/Flux:** 100-200トークン。T5エンコーダにより長文理解が改善
- **DALL-E 3:** 100-300語の詳細な文章が効果的。短すぎるとGPT-4が大幅に書き換える
- **Midjourney:** 20-60語。短い方がスタイルが安定。V6では長めでも改善
- **原則:** 必要十分な情報を、無駄なく記述する。「必要な要素が全て含まれる最短のプロンプト」が理想

### Q3: プロンプトのデバッグ方法は?

**A:** 体系的にテストします:

1. **最小プロンプトから始める:** 主題のみで生成し、基本品質を確認
2. **要素を一つずつ追加:** スタイル → 品質 → 構図 の順に追加
3. **シード固定:** 同一シードで変更の影響を比較
4. **バッチ生成:** 4-8枚同時生成してばらつきを確認
5. **ネガティブの段階テスト:** なし → 基本 → 詳細 と比較
6. **IterativePromptRefinerクラス** を使って進化を記録

### Q4: DALL-E 3でプロンプトが勝手に書き換えられるのを防ぐには?

**A:** 完全に防ぐことはできませんが、以下の対策があります:

- `revised_prompt` を確認して意図とのズレを把握
- 具体的で詳細なプロンプトを書くと書き換えが少なくなる
- "I NEED to test how the tool works with this exact prompt." を冒頭に追加すると、書き換えが抑制される場合がある（非公式テクニック）
- APIの `style` パラメータで `"natural"` を選ぶと書き換えが控えめになる傾向

### Q5: LoRAやテキスト反転のトリガーワードはどこに配置すべき?

**A:** 以下の配置が推奨されます:

- **LoRAトリガーワード:** プロンプトの先頭付近に配置。影響を最大化するため
- **テキスト反転のトークン:** 主題部分に自然に組み込む
- **複数のLoRA併用:** 各トリガーワードの重みを0.5-0.8に下げて干渉を防ぐ
- **注意:** トリガーワードがプロンプトの意味を乱す場合は、BREAK構文で分離する

### Q6: プロンプトの「BREAK」構文とは?

**A:** BREAK構文はSD系で使えるプロンプト分離技法です:

```
[使用例]
beautiful landscape, cherry blossoms, mount fuji BREAK
golden hour lighting, dramatic clouds, warm colors BREAK
8K resolution, masterpiece, highly detailed

[効果]
- BREAK前後で独立したトークングループとして処理される
- 各グループが独立したCLIP埋め込みを持つ
- 長いプロンプトでの概念混合を防ぐ
- 75トークンを超える場合のチャンク分割を制御

[注意]
- DALL-E 3、Midjourney、Fluxでは使用不可
- 過度な使用は不自然な結果につながる
```

### Q7: 同じプロンプトでも結果が毎回違うのはなぜ?

**A:** いくつかの要因があります:

- **シード:** ランダムシードが異なれば結果が変わる（最大の要因）
- **浮動小数点演算:** GPU環境によりわずかな計算誤差が累積
- **モデルバージョン:** マイナーアップデートで結果が変わることがある
- **サンプラー:** DPM++やEulerなど、サンプラーの選択で結果が異なる
- **CFGスケール:** 値の微調整で出力が大きく変わる
- **対策:** シード固定 + 同一環境 + 同一パラメータ で再現性を確保

---

## 10. まとめ表

| 項目 | 要点 |
|------|------|
| **4層構造** | 主題 → スタイル → 品質 → 構図 の優先順位で記述 |
| **トークン位置** | 先頭のトークンほど影響が大きい |
| **ネガティブ** | 汎用テンプレート + カテゴリ固有 + カスタム除外 |
| **重み付け** | SD系: (keyword:weight) / Compel推奨、MJ: ::weight |
| **モデル別最適化** | SD=タグ、DALL-E=自然文、MJ=短いキーワード、Flux=自然文 |
| **色彩制御** | カラーパレットを明示的に指定、和風配色も対応 |
| **言語** | 英語が基本。日本語特化モデルは例外 |
| **デバッグ** | 最小プロンプト → 段階的追加 → シード固定比較 |
| **A/Bテスト** | 1要素ずつ変更、複数シード、結果を記録 |
| **業種テンプレート** | EC/不動産/飲食/ゲーム/SNS向け最適化済みテンプレート |
| **BREAK構文** | SD系でのトークングループ分離（概念混合防止） |
| **Compel** | SD系での高精度重み制御、ブレンド、連結に対応 |

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
5. Rombach, R. et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*. https://arxiv.org/abs/2112.10752
6. Midjourney Documentation. "Prompts." https://docs.midjourney.com/docs/prompts
7. Stability AI. "Stable Diffusion XL Documentation." https://stability.ai/stable-diffusion
8. Black Forest Labs. "Flux.1 Model Documentation." https://blackforestlabs.ai/
