# 画像生成 — Stable Diffusion、DALL-E、Midjourney

> 3大画像生成プラットフォームの特徴と使い方を、実践的なコード例とワークフロー比較で完全ガイドする。

---

## この章で学ぶこと

1. **Stable Diffusion のローカル実行とカスタマイズ** — diffusers、ComfyUI、LoRA の活用
2. **DALL-E 3 API の活用** — API設計、品質制御、ChatGPT連携
3. **Midjourney の効果的な使い方** — パラメータ制御、スタイル一貫性、ワークフロー統合

---

## 1. Stable Diffusion エコシステム

### コード例1: diffusers による基本的な画像生成

```python
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
import torch

# VAE をロード (品質向上のため専用VAEを使用)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

# SDXL パイプラインをロード
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

# サンプラーを高速版に変更
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

# 画像生成
image = pipe(
    prompt="A majestic Japanese castle surrounded by cherry blossoms, "
           "golden hour lighting, photorealistic, 8K",
    negative_prompt="low quality, blurry, distorted",
    num_inference_steps=25,
    guidance_scale=7.0,
    width=1024,
    height=1024,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("castle.png")
```

### コード例2: LoRA の適用

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# LoRA をロードして適用
pipe.load_lora_weights(
    "path/to/lora",
    weight_name="anime_style_v2.safetensors",
    adapter_name="anime_style",
)

# LoRA の影響度を調整 (0.0~1.0)
pipe.set_adapters(["anime_style"], adapter_weights=[0.8])

image = pipe(
    prompt="1girl, cherry blossom, detailed eyes, anime style",
    num_inference_steps=25,
).images[0]

# 複数 LoRA の同時適用
pipe.load_lora_weights(
    "path/to/lora2",
    weight_name="lighting_enhance.safetensors",
    adapter_name="lighting",
)
pipe.set_adapters(
    ["anime_style", "lighting"],
    adapter_weights=[0.7, 0.5],  # 個別に重み調整
)
```

### ASCII図解1: Stable Diffusion パイプラインの全体構成

```
┌─────────────── Stable Diffusion パイプライン ──────────────┐
│                                                           │
│  テキスト入力                                              │
│      │                                                    │
│      v                                                    │
│  ┌──────────────┐                                         │
│  │ Text Encoder │  CLIP (SD1.5) / CLIP+OpenCLIP (SDXL)   │
│  │ (トークン化) │  / T5+CLIP (SD3/Flux)                   │
│  └──────┬───────┘                                         │
│         │ テキスト埋め込み                                 │
│         v                                                 │
│  ┌──────────────────────────────────────┐                  │
│  │         UNet / DiT                   │                  │
│  │  ┌─────────┐  ┌─────────────────┐   │  ← ノイズ       │
│  │  │ Self-   │  │ Cross-Attention │   │    (潜在空間)    │
│  │  │Attention│  │ (テキスト条件)  │   │                  │
│  │  └─────────┘  └─────────────────┘   │                  │
│  │  × N ステップ (逆拡散)              │                  │
│  └──────────────┬───────────────────────┘                  │
│                 │ ノイズ除去された潜在表現                  │
│                 v                                          │
│  ┌──────────────┐                                         │
│  │ VAE Decoder  │  潜在空間 → ピクセル空間                 │
│  │ (4x64x64 →  │                                         │
│  │  3x512x512)  │                                         │
│  └──────┬───────┘                                         │
│         │                                                 │
│         v                                                 │
│     生成画像                                               │
└───────────────────────────────────────────────────────────┘
```

---

## 2. DALL-E 3 API

### コード例3: DALL-E 3 による画像生成

```python
from openai import OpenAI
import base64
import httpx
from pathlib import Path

client = OpenAI()

# 基本的な画像生成
response = client.images.generate(
    model="dall-e-3",
    prompt="日本の伝統的な温泉旅館の露天風呂。紅葉に囲まれ、"
           "湯気が朝霧のように立ち上る。写真のようにリアル。",
    size="1792x1024",       # 横長
    quality="hd",           # 高品質モード
    style="natural",        # "natural" or "vivid"
    n=1,
)

# 生成された画像の情報
image_url = response.data[0].url
revised_prompt = response.data[0].revised_prompt  # GPT-4が書き換えたプロンプト
print(f"修正プロンプト: {revised_prompt}")

# 画像をダウンロード
image_data = httpx.get(image_url).content
Path("onsen.png").write_bytes(image_data)

# バリエーション生成のパターン
prompts_batch = [
    "同じ温泉旅館を春の桜の季節で",
    "同じ温泉旅館を冬の雪景色で",
    "同じ温泉旅館を夏の緑深い季節で",
]

for i, prompt in enumerate(prompts_batch):
    resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
    )
    print(f"Season {i}: {resp.data[0].url}")
```

### コード例4: Midjourney スタイルのパラメータガイド

```python
"""
Midjourney パラメータリファレンス (v6.1)

構文: /imagine prompt: [テキスト] --[パラメータ]
"""

MIDJOURNEY_PARAMS = {
    # アスペクト比
    "--ar": {
        "説明": "画像のアスペクト比",
        "例": "--ar 16:9, --ar 3:2, --ar 1:1",
        "デフォルト": "1:1",
    },
    # スタイライズ
    "--s": {
        "説明": "Midjourneyのスタイル適用度 (0-1000)",
        "例": "--s 50 (控えめ), --s 750 (強い)",
        "デフォルト": "100",
    },
    # カオス
    "--c": {
        "説明": "バリエーションの多様性 (0-100)",
        "例": "--c 0 (安定), --c 50 (多様)",
        "デフォルト": "0",
    },
    # 品質
    "--q": {
        "説明": "生成品質/計算量 (.25, .5, 1)",
        "例": "--q 1 (最高), --q .5 (高速)",
        "デフォルト": "1",
    },
    # ネガティブ
    "--no": {
        "説明": "除外したい要素",
        "例": "--no text, watermark, people",
    },
    # シード
    "--seed": {
        "説明": "再現性のためのシード値",
        "例": "--seed 12345",
    },
    # スタイルロー
    "--sref": {
        "説明": "スタイル参照画像URL",
        "例": "--sref https://example.com/style.png",
    },
}

def build_mj_prompt(text: str, **params) -> str:
    """Midjourney用プロンプトを構築"""
    parts = [f"/imagine prompt: {text}"]
    for param, value in params.items():
        if not param.startswith("--"):
            param = f"--{param}"
        parts.append(f"{param} {value}")
    return " ".join(parts)

# 使用例
prompt = build_mj_prompt(
    "Ancient Japanese temple, moss-covered, misty morning",
    ar="16:9", s="250", c="20", q="1", no="people, text"
)
print(prompt)
```

### ASCII図解2: 3大プラットフォームのワークフロー比較

```
Stable Diffusion (ローカル):
┌──────┐   ┌─────────┐   ┌──────┐   ┌──────┐
│モデル │→  │ComfyUI/ │→  │生成   │→  │後処理 │
│選択   │   │A1111    │   │(LoRA) │   │(拡大) │
└──────┘   └─────────┘   └──────┘   └──────┘
  自由度: ★★★★★  コスト: GPU電気代のみ

DALL-E 3 (API):
┌──────┐   ┌─────────┐   ┌──────┐   ┌──────┐
│プロン │→  │GPT-4    │→  │生成   │→  │URL   │
│プト   │   │書き換え │   │(クラウド)│  │取得  │
└──────┘   └─────────┘   └──────┘   └──────┘
  自由度: ★★★☆☆  コスト: $0.04-0.12/枚

Midjourney (Discord/Web):
┌──────┐   ┌─────────┐   ┌──────┐   ┌──────┐
│/imagine│→ │4枚グリッド│→│選択   │→  │Upscale│
│コマンド│  │生成     │   │(U/V) │   │/Vary │
└──────┘   └─────────┘   └──────┘   └──────┘
  自由度: ★★☆☆☆  コスト: $10-120/月
```

### ASCII図解3: ComfyUI ノードベースワークフロー

```
┌─────────────┐
│ Load        │
│ Checkpoint  │──────────────┐
│ (SDXL)      │              │
└─────────────┘              │
                             v
┌─────────────┐    ┌─────────────────┐    ┌──────────┐
│ CLIP Text   │───>│  KSampler       │───>│ VAE      │
│ Encode      │    │  Steps: 25      │    │ Decode   │
│ (Positive)  │    │  CFG: 7.0       │    │          │──> 画像
└─────────────┘    │  Sampler: dpmpp │    └──────────┘
                   │  Scheduler: karras│
┌─────────────┐    │                 │
│ CLIP Text   │───>│                 │
│ Encode      │    └─────────────────┘
│ (Negative)  │            ↑
└─────────────┘            │
                  ┌────────────────┐
                  │ Empty Latent   │
                  │ Image          │
                  │ 1024x1024      │
                  └────────────────┘
```

---

## 3. 比較表

### 比較表1: 3大プラットフォーム詳細比較

| 項目 | Stable Diffusion | DALL-E 3 | Midjourney |
|------|-----------------|----------|-----------|
| **実行環境** | ローカル/クラウド | クラウドAPI | Discord/Web |
| **カスタマイズ** | 極めて高い | 低い | 低い |
| **LoRA対応** | あり | なし | なし |
| **ControlNet** | あり | なし | なし |
| **日本語入力** | モデル依存 | 対応 | 限定的 |
| **商用利用** | モデル依存 | 可 (有料) | 可 (有料) |
| **テキスト描画** | 苦手 (Fluxは得意) | 得意 | やや苦手 |
| **一貫性** | シード固定可 | 低い | スタイルリファレンス |
| **バッチ生成** | 容易 | API可能 | 手動 |
| **学習曲線** | 急 | 緩やか | 中程度 |

### 比較表2: ユースケース別推奨プラットフォーム

| ユースケース | 推奨 | 理由 |
|-------------|------|------|
| **大量のバリエーション生成** | Stable Diffusion | コスト効率、バッチ処理 |
| **ビジネス資料の挿絵** | DALL-E 3 | 簡単、高品質、日本語対応 |
| **アート作品の制作** | Midjourney | 芸術的品質、独自のスタイル |
| **ゲームアセット** | Stable Diffusion | ControlNet、一貫性制御 |
| **プロトタイプUI** | DALL-E 3 | 自然言語で詳細指定可能 |
| **建築パース** | Stable Diffusion | ControlNet (深度/線画) |
| **SNS投稿素材** | Midjourney / DALL-E 3 | 手軽さと品質のバランス |

---

## 4. アンチパターン

### アンチパターン1: モデルバージョンを固定しない

```
[問題]
「Stable Diffusion」とだけ指定し、具体的なモデルを
決めずにプロジェクトを開始する。

[なぜ問題か]
- SD 1.5, SDXL, SD3, Flux でプロンプト互換性が異なる
- LoRA はモデルバージョンに依存
- プロジェクト途中のモデル変更はスタイルの一貫性を破壊

[正しいアプローチ]
- プロジェクト開始時にベースモデルとバージョンを決定
- テスト生成で品質を確認してから本番投入
- モデルカード(ライセンス)を確認して商用利用可否を確認
```

### アンチパターン2: guidance_scale を極端に設定

```
[問題]
「プロンプトに忠実にしたい」と guidance_scale を
20や30に設定する。

[なぜ問題か]
- 過剰なCFGは画像の彩度過多、コントラスト異常を引き起こす
- 画像が「焼けた」ような不自然な色味になる
- 詳細が潰れてアーティファクトが発生

[正しいアプローチ]
- SDXL: 5.0-8.0 が推奨範囲
- SD3/Flux: 3.0-5.0 が推奨範囲
- DALL-E 3: ユーザー側で調整不可 (最適化済み)
- 高い忠実度が欲しい場合はプロンプト自体を改善する
```

---

## FAQ

### Q1: Stable Diffusion をローカルで動かす最低スペックは?

**A:**

- **SD 1.5:** VRAM 4GB (最低) / 8GB (推奨)
- **SDXL:** VRAM 8GB (最低) / 12GB (推奨)
- **SD3/Flux:** VRAM 12GB (最低) / 16GB+ (推奨)
- **Apple Silicon:** M1 8GB で SD1.5 動作可、SDXL は M2 Pro 16GB~
- **量子化版:** GGUF/NF4形式で必要VRAMを半減可能

### Q2: 生成した画像の品質が低い場合の対処法は?

**A:** 以下の順序でチェックします:

1. **プロンプトの見直し:** 品質タグ追加、ネガティブプロンプト設定
2. **サンプラーの変更:** DPM++ 2M Karras が安定
3. **ステップ数の調整:** 20-30ステップを推奨
4. **CFGの調整:** 5-8の範囲で試行
5. **モデルの変更:** タスクに合ったファインチューンモデルを使用
6. **後処理:** アップスケーリング、img2img による仕上げ

### Q3: 商用利用する際に注意すべきことは?

**A:**

- **Stable Diffusion:** モデルのライセンスを確認 (CreativeML Open RAIL-M 等)
- **DALL-E 3:** 利用規約に基づき商用利用可。ただしコンテンツポリシー遵守
- **Midjourney:** 有料プランで商用利用可。年収$1M超は Pro プラン必須
- **共通:** 他者の著作物に酷似した画像の商用利用は法的リスク
- **推奨:** 生成画像を素材として使い、最終成果物に人間の創作を加える

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **SD系** | カスタマイズ性最高。LoRA、ControlNet で精密制御。学習コスト高 |
| **DALL-E 3** | 最も手軽。自然言語プロンプト。API統合が容易 |
| **Midjourney** | アート品質最高。Discord/Webで手軽。カスタマイズ性は低い |
| **モデル選択** | プロジェクト初期に確定。途中変更はスタイル崩壊リスク |
| **ワークフロー** | ComfyUI (ノード) / Automatic1111 (WebUI) / API統合 |
| **コスト** | ローカル=GPU投資、クラウド=従量/サブスク |

---

## 次に読むべきガイド

- [01-image-editing.md](./01-image-editing.md) — インペインティング、アウトペインティング
- [02-upscaling.md](./02-upscaling.md) — 超解像でさらなる高品質化
- [03-design-tools.md](./03-design-tools.md) — Canva AI、Adobe Firefly 等との統合

---

## 参考文献

1. Rombach, R. et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*. https://arxiv.org/abs/2112.10752
2. Podell, D. et al. (2023). "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." *arXiv*. https://arxiv.org/abs/2307.01952
3. Betker, J. et al. (2023). "Improving Image Generation with Better Captions (DALL-E 3)." *OpenAI*. https://cdn.openai.com/papers/dall-e-3.pdf
4. Esser, P. et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)." https://arxiv.org/abs/2403.03206
