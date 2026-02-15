# 画像生成 — Stable Diffusion、DALL-E、Midjourney

> 3大画像生成プラットフォームの特徴と使い方を、実践的なコード例とワークフロー比較で完全ガイドする。

---

## この章で学ぶこと

1. **Stable Diffusion のローカル実行とカスタマイズ** — diffusers、ComfyUI、LoRA の活用
2. **DALL-E 3 API の活用** — API設計、品質制御、ChatGPT連携
3. **Midjourney の効果的な使い方** — パラメータ制御、スタイル一貫性、ワークフロー統合
4. **Flux の導入と実践** — Rectified Flow Transformer による次世代画像生成
5. **バッチ生成と自動化パイプライン** — 大量画像の効率的な生成ワークフロー
6. **品質最適化テクニック** — スケジューラ選択、CFG制御、後処理の実践

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

### コード例1b: メモリ最適化付きの高度な生成

```python
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
import torch
from compel import Compel, ReturnedEmbeddingsType

# VAE ロード
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

# メモリ最適化の各種オプション
pipe.enable_attention_slicing()     # Attention メモリ削減
pipe.enable_vae_slicing()           # VAE メモリ削減
pipe.enable_vae_tiling()            # 大解像度時のVAE処理
# pipe.enable_model_cpu_offload()   # VRAM不足時にCPUオフロード
# pipe.enable_sequential_cpu_offload()  # 極端にVRAM不足の場合

# Compel によるプロンプトの重み付け制御
compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
)

# プロンプト重み付け: (word)++ で強調、(word)-- で抑制
prompt = "A (majestic)++ Japanese castle, (cherry blossoms)++, golden hour"
negative = "(low quality)--, (blurry)--, distorted, watermark"

conditioning, pooled = compel(prompt)
neg_conditioning, neg_pooled = compel(negative)

image = pipe(
    prompt_embeds=conditioning,
    pooled_prompt_embeds=pooled,
    negative_prompt_embeds=neg_conditioning,
    negative_pooled_prompt_embeds=neg_pooled,
    num_inference_steps=30,
    guidance_scale=7.0,
    width=1024,
    height=1024,
).images[0]

image.save("castle_weighted.png")
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

### コード例2b: LoRA の訓練 (DreamBooth LoRA)

```python
"""
LoRA ファインチューニングの実行例
diffusers の train_dreambooth_lora_sdxl.py スクリプトを使用
"""

# コマンドライン実行の設定
train_config = {
    "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
    "instance_data_dir": "./training_images",     # 学習画像フォルダ (10-30枚推奨)
    "output_dir": "./my_lora_model",
    "instance_prompt": "a photo of sks dog",      # sks はトリガーワード
    "resolution": 1024,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 100,
    "max_train_steps": 1000,
    "rank": 32,                                    # LoRA のランク (4-128)
    "seed": 42,
    "mixed_precision": "fp16",
    "use_8bit_adam": True,                         # メモリ節約
    "gradient_checkpointing": True,                # メモリ節約
    "prior_preservation": True,                    # 過学習防止
    "prior_preservation_class_prompt": "a photo of a dog",
    "num_class_images": 100,
}

# accelerate を使った訓練コマンド生成
def generate_train_command(config):
    cmd = "accelerate launch train_dreambooth_lora_sdxl.py"
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{key}"
        else:
            cmd += f" --{key}={value}"
    return cmd

print(generate_train_command(train_config))

# 訓練後の LoRA 使用
# pipe.load_lora_weights("./my_lora_model", weight_name="pytorch_lora_weights.safetensors")
# image = pipe("a photo of sks dog in a garden", num_inference_steps=25).images[0]
```

### コード例2c: ControlNet による構造制御

```python
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from diffusers.utils import load_image
from controlnet_aux import CannyDetector, OpenposeDetector, MidasDetector
import torch
from PIL import Image

# --- Canny Edge ControlNet ---
controlnet_canny = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet_canny,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()

# エッジ検出
canny = CannyDetector()
source_image = load_image("reference_building.png")
canny_image = canny(source_image, low_threshold=100, high_threshold=200)

# エッジに基づいた画像生成
image = pipe(
    prompt="A futuristic glass skyscraper, same structure as reference, "
           "cyberpunk city, neon lights, night scene, photorealistic",
    negative_prompt="low quality, blurry",
    image=canny_image,
    controlnet_conditioning_scale=0.7,  # ControlNet の影響度
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

image.save("controlnet_building.png")


# --- OpenPose ControlNet ---
controlnet_pose = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16,
)
pipe_pose = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet_pose,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe_pose.to("cuda")

# ポーズ検出
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
pose_image = openpose(load_image("person_reference.png"))

# ポーズに基づいた画像生成
image_pose = pipe_pose(
    prompt="A samurai warrior in traditional armor, dynamic pose, "
           "detailed, concept art, dramatic lighting",
    image=pose_image,
    controlnet_conditioning_scale=0.8,
    num_inference_steps=30,
).images[0]

image_pose.save("controlnet_samurai.png")


# --- Depth ControlNet ---
controlnet_depth = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16,
)
pipe_depth = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet_depth,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe_depth.to("cuda")

# 深度マップ推定
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
depth_image = midas(load_image("room_photo.png"))

# 深度に基づいたインテリアデザイン変換
image_depth = pipe_depth(
    prompt="Modern Japanese interior, tatami room, shoji screens, "
           "natural light, same spatial layout, interior design magazine",
    image=depth_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=30,
).images[0]

image_depth.save("controlnet_interior.png")
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

### コード例2d: スケジューラ(サンプラー)の比較と選択

```python
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DDIMScheduler,
)
import torch
import time

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# スケジューラの一覧と特徴
schedulers = {
    "DPM++ 2M Karras": {
        "class": DPMSolverMultistepScheduler,
        "config": {"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"},
        "推奨ステップ": 25,
        "特徴": "高速・高品質のバランスが最良。最も汎用的",
    },
    "Euler": {
        "class": EulerDiscreteScheduler,
        "config": {},
        "推奨ステップ": 30,
        "特徴": "シンプルで安定。Midjourney のデフォルトに近い",
    },
    "Euler Ancestral": {
        "class": EulerAncestralDiscreteScheduler,
        "config": {},
        "推奨ステップ": 30,
        "特徴": "ランダム性あり。同一シードでもやや異なる結果",
    },
    "UniPC": {
        "class": UniPCMultistepScheduler,
        "config": {},
        "推奨ステップ": 20,
        "特徴": "最小ステップ数で高品質。高速生成に最適",
    },
    "Heun": {
        "class": HeunDiscreteScheduler,
        "config": {},
        "推奨ステップ": 25,
        "特徴": "高品質だが低速 (各ステップで2回評価)。精密な生成向き",
    },
    "DDIM": {
        "class": DDIMScheduler,
        "config": {},
        "推奨ステップ": 50,
        "特徴": "決定論的。画像間の補間に使用可能",
    },
}

# ベンチマーク実行
prompt = "A beautiful landscape, mountains, lake, sunset, photorealistic"
results = {}

for name, sched_info in schedulers.items():
    scheduler = sched_info["class"].from_config(
        pipe.scheduler.config, **sched_info["config"]
    )
    pipe.scheduler = scheduler
    steps = sched_info["推奨ステップ"]

    start = time.time()
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=7.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    elapsed = time.time() - start

    results[name] = {
        "time": round(elapsed, 2),
        "steps": steps,
        "特徴": sched_info["特徴"],
    }
    image.save(f"scheduler_{name.replace(' ', '_').lower()}.png")

# 結果出力
for name, info in results.items():
    print(f"{name}: {info['time']}秒 ({info['steps']}ステップ)")
    print(f"  特徴: {info['特徴']}")
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

### コード例3b: DALL-E 3 の高度な活用パターン

```python
from openai import OpenAI
import httpx
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

client = OpenAI()

class DALLEWorkflow:
    """DALL-E 3 の実務的な活用パターン集"""

    def __init__(self, output_dir="./generated"):
        self.client = OpenAI()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_with_metadata(self, prompt, size="1024x1024",
                                quality="hd", style="natural"):
        """メタデータ付き画像生成"""
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1,
        )

        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt

        # 画像ダウンロード
        image_data = httpx.get(image_url).content

        # ファイル名生成
        import hashlib
        name_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        filename = f"dalle3_{name_hash}.png"
        filepath = self.output_dir / filename

        filepath.write_bytes(image_data)

        # メタデータ保存
        metadata = {
            "original_prompt": prompt,
            "revised_prompt": revised_prompt,
            "size": size,
            "quality": quality,
            "style": style,
            "filename": filename,
        }
        meta_path = self.output_dir / f"dalle3_{name_hash}_meta.json"
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

        return filepath, metadata

    def generate_product_shots(self, product_name, angles=None, styles=None):
        """商品画像の複数アングル生成"""
        if angles is None:
            angles = [
                "front view, centered",
                "45 degree angle, slightly above",
                "close-up detail shot",
                "lifestyle setting, in use",
            ]
        if styles is None:
            styles = ["natural"]

        results = []
        for angle in angles:
            for style in styles:
                prompt = (
                    f"Professional product photography of {product_name}, "
                    f"{angle}, studio lighting, white background, "
                    f"commercial quality, 4K, no text or watermark"
                )
                filepath, meta = self.generate_with_metadata(
                    prompt, size="1024x1024", quality="hd", style=style
                )
                results.append({"file": str(filepath), "angle": angle, **meta})

        return results

    def generate_with_chatgpt_enhancement(self, rough_description):
        """ChatGPT でプロンプトを拡張してから生成"""
        # GPT-4 でプロンプトを改善
        enhancement = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content":
                    "You are an expert at writing prompts for DALL-E 3. "
                    "Given a rough description, create a detailed, vivid prompt "
                    "that will produce a stunning image. Include details about "
                    "lighting, composition, style, and mood."},
                {"role": "user", "content": rough_description},
            ],
            max_tokens=300,
        )
        enhanced_prompt = enhancement.choices[0].message.content

        # 強化されたプロンプトで生成
        filepath, meta = self.generate_with_metadata(enhanced_prompt)
        meta["rough_description"] = rough_description
        meta["enhanced_prompt"] = enhanced_prompt

        return filepath, meta


# 使用例
workflow = DALLEWorkflow(output_dir="./product_shots")

# 商品画像生成
results = workflow.generate_product_shots("minimalist leather wallet")
for r in results:
    print(f"[{r['angle']}] {r['file']}")

# ChatGPT 強化生成
filepath, meta = workflow.generate_with_chatgpt_enhancement(
    "日本の古い温泉街の夜景"
)
print(f"生成: {filepath}")
print(f"強化プロンプト: {meta['enhanced_prompt'][:100]}...")
```

### コード例3c: DALL-E 3 サイズ・スタイル・品質の組み合わせ効果

```python
"""
DALL-E 3 パラメータの効果比較

サイズ:
  1024x1024  — 正方形 (デフォルト、SNS投稿向き)
  1024x1792  — 縦長 (ポスター、スマホ壁紙)
  1792x1024  — 横長 (バナー、プレゼン背景)

品質:
  standard   — 標準品質、高速、$0.040/枚
  hd         — 高品質、精密なディテール、$0.080/枚

スタイル:
  natural    — 写実的、自然な印象
  vivid      — 鮮やか、コントラスト強め、映画的
"""

from openai import OpenAI
client = OpenAI()

# パラメータの全組み合わせで生成比較
def compare_dalle3_params(base_prompt):
    sizes = ["1024x1024", "1024x1792", "1792x1024"]
    qualities = ["standard", "hd"]
    styles = ["natural", "vivid"]

    results = []
    for size in sizes:
        for quality in qualities:
            for style in styles:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=base_prompt,
                    size=size,
                    quality=quality,
                    style=style,
                    n=1,
                )
                cost = 0.040 if quality == "standard" and size == "1024x1024" else \
                       0.080 if quality == "standard" else \
                       0.080 if size == "1024x1024" else 0.120
                results.append({
                    "size": size,
                    "quality": quality,
                    "style": style,
                    "cost_usd": cost,
                    "revised_prompt": response.data[0].revised_prompt[:80],
                    "url": response.data[0].url,
                })
                print(f"Generated: {size} / {quality} / {style} (${cost})")

    return results

# 使用例
# results = compare_dalle3_params("A serene Japanese garden in autumn")
```

---

## 3. Midjourney

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
    # キャラクターリファレンス (v6+)
    "--cref": {
        "説明": "キャラクター参照画像URL",
        "例": "--cref https://example.com/character.png",
        "注意": "同一キャラクターの別ポーズ生成に使用",
    },
    # タイル
    "--tile": {
        "説明": "タイリング可能なパターンを生成",
        "例": "--tile",
        "用途": "テクスチャ、壁紙、テキスタイル",
    },
    # ストップ
    "--stop": {
        "説明": "生成を途中で停止 (10-100)",
        "例": "--stop 80",
        "用途": "抽象的、未完成感のある画像",
    },
    # ウィアード
    "--weird": {
        "説明": "実験的・奇抜な生成 (0-3000)",
        "例": "--weird 500",
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

### コード例4b: Midjourney プロンプトテンプレート集

```python
"""
Midjourney 実践的プロンプトテンプレート
"""

class MidjourneyPromptBuilder:
    """Midjourney 向けプロンプトの体系的な構築"""

    # スタイルプリセット
    STYLE_PRESETS = {
        "photorealistic": "photorealistic, DSLR, 85mm lens, f/1.8, "
                         "natural lighting, 8K resolution",
        "cinematic": "cinematic lighting, film grain, anamorphic, "
                    "movie still, dramatic shadows",
        "anime": "anime style, cel shading, vibrant colors, "
                "detailed, Studio Ghibli inspired",
        "watercolor": "watercolor painting, soft edges, transparent washes, "
                     "artistic, paper texture",
        "3d_render": "3D render, octane render, subsurface scattering, "
                    "volumetric lighting, high detail",
        "concept_art": "concept art, digital painting, matte painting, "
                      "artstation trending, detailed illustration",
        "oil_painting": "oil painting, classical style, rich textures, "
                       "canvas texture, chiaroscuro lighting",
        "flat_design": "flat design, vector art, clean lines, "
                      "minimal, modern graphic design",
    }

    # ライティングプリセット
    LIGHTING_PRESETS = {
        "golden_hour": "golden hour, warm light, long shadows",
        "blue_hour": "blue hour, twilight, cool tones",
        "studio": "studio lighting, softbox, key light and fill light",
        "dramatic": "dramatic lighting, strong contrast, rim lighting",
        "natural": "natural daylight, soft shadows",
        "neon": "neon lighting, cyberpunk, colorful glow",
        "moonlight": "moonlit, silvery light, ethereal atmosphere",
    }

    # 構図プリセット
    COMPOSITION_PRESETS = {
        "rule_of_thirds": "rule of thirds composition",
        "centered": "centered symmetrical composition",
        "wide_angle": "wide angle perspective, expansive view",
        "closeup": "extreme close-up, macro detail",
        "birds_eye": "bird's eye view, top-down perspective",
        "worms_eye": "worm's eye view, looking up, dramatic angle",
    }

    def build(self, subject, style=None, lighting=None,
              composition=None, additional="", **mj_params):
        """構造化されたプロンプトを構築"""
        parts = [subject]

        if style and style in self.STYLE_PRESETS:
            parts.append(self.STYLE_PRESETS[style])
        elif style:
            parts.append(style)

        if lighting and lighting in self.LIGHTING_PRESETS:
            parts.append(self.LIGHTING_PRESETS[lighting])

        if composition and composition in self.COMPOSITION_PRESETS:
            parts.append(self.COMPOSITION_PRESETS[composition])

        if additional:
            parts.append(additional)

        prompt_text = ", ".join(parts)

        # Midjourney パラメータを追加
        param_parts = []
        for param, value in mj_params.items():
            if not param.startswith("--"):
                param = f"--{param}"
            if value is True:
                param_parts.append(param)
            else:
                param_parts.append(f"{param} {value}")

        if param_parts:
            prompt_text += " " + " ".join(param_parts)

        return f"/imagine prompt: {prompt_text}"


# 使用例
builder = MidjourneyPromptBuilder()

# フォトリアルな風景
prompt1 = builder.build(
    subject="Ancient Japanese shrine in a bamboo forest",
    style="photorealistic",
    lighting="golden_hour",
    composition="rule_of_thirds",
    ar="16:9", s="200", q="1"
)
print(f"風景: {prompt1}")

# コンセプトアート
prompt2 = builder.build(
    subject="A cyberpunk samurai standing on a neon-lit rooftop",
    style="concept_art",
    lighting="neon",
    composition="worms_eye",
    additional="rain, reflections, detailed armor",
    ar="2:3", s="500"
)
print(f"コンセプトアート: {prompt2}")

# テキスタイルパターン
prompt3 = builder.build(
    subject="Japanese wave pattern with koi fish",
    style="flat_design",
    additional="seamless pattern, navy blue and gold",
    tile=True, ar="1:1", s="100"
)
print(f"パターン: {prompt3}")
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

Flux (ローカル/API):
┌──────┐   ┌─────────┐   ┌──────┐   ┌──────┐
│プロン │→  │Rectified│→  │生成   │→  │保存   │
│プト   │   │Flow DiT │   │(28step)│  │      │
└──────┘   └─────────┘   └──────┘   └──────┘
  自由度: ★★★★☆  コスト: GPU/API依存
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

### ASCII図解3b: ComfyUI ControlNet + LoRA ワークフロー

```
┌──────────────┐
│ Load         │
│ Checkpoint   │─────────┐
│ (SDXL Base)  │         │
└──────────────┘         │
        │                │
┌───────────────┐        │
│ Load LoRA     │        │
│ (anime_v2)    │────────┤
│ strength: 0.8 │        │
└───────────────┘        │
                         v
┌──────────────┐  ┌──────────────┐  ┌─────────────┐
│ Load Image   │  │ Apply        │  │ KSampler    │
│ (reference)  │─>│ ControlNet   │─>│ Advanced    │──> VAE Decode ──> 画像
└──────────────┘  │ (Canny)      │  │ Steps: 30   │
        │         │ strength:0.7 │  │ CFG: 7.0    │
        v         └──────────────┘  │ denoise:1.0 │
┌──────────────┐                    └─────────────┘
│ Canny Edge   │                          ↑
│ Detector     │                    ┌──────────┐
│ low:100      │                    │ CLIP     │
│ high:200     │                    │ Encode   │
└──────────────┘                    │ (Pos/Neg)│
                                    └──────────┘
```

---

## 4. Flux による次世代画像生成

### コード例5: Flux パイプラインの基本

```python
from diffusers import FluxPipeline
import torch

# Flux.1-dev モデルのロード
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()  # VRAM 節約

# 基本生成 — Flux はテキスト描画が非常に得意
image = pipe(
    prompt='A wooden sign in a forest that reads "Welcome to the '
           'Enchanted Forest" in elegant calligraphy, '
           'moss-covered, sunlight filtering through leaves',
    num_inference_steps=28,
    guidance_scale=3.5,      # Flux は低 CFG が推奨 (2.0-5.0)
    width=1024,
    height=768,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]

image.save("flux_basic.png")
```

### コード例5b: Flux の高度な活用

```python
from diffusers import FluxPipeline, FluxImg2ImgPipeline
import torch
from PIL import Image

# --- Flux テキスト描画の実践例 ---
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

# テキスト描画 (Flux の最大の強み)
text_prompts = [
    'A coffee shop menu board that reads:\n'
    '"Today\'s Special\nMatcha Latte - $5\nEspresso - $4"\n'
    'chalk style on dark background',

    'A vintage book cover with the title "The Last Samurai" '
    'in gold embossed letters, leather texture, ornate border',

    'A neon sign that says "OPEN 24/7" in bright pink and blue, '
    'wet street reflection, night scene, Japanese alley',
]

for i, prompt in enumerate(text_prompts):
    image = pipe(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        width=1024,
        height=1024,
        generator=torch.Generator("cpu").manual_seed(42 + i),
    ).images[0]
    image.save(f"flux_text_{i}.png")
    print(f"Generated: flux_text_{i}.png")


# --- Flux img2img ---
pipe_i2i = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe_i2i.enable_model_cpu_offload()

# 既存画像のスタイル変換
init_image = Image.open("photo.png").resize((1024, 1024))
image = pipe_i2i(
    prompt="Same scene but in Studio Ghibli anime style, "
           "vibrant colors, detailed background",
    image=init_image,
    strength=0.6,           # 元画像の維持度 (0.0=変更なし, 1.0=完全生成)
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("flux_style_transfer.png")
```

### 比較表: SD系モデルの世代間比較

| 特徴 | SD 1.5 | SDXL | SD3 | Flux.1 |
|------|--------|------|-----|--------|
| **アーキテクチャ** | U-Net | U-Net (大型) | MMDiT | DiT (Rectified Flow) |
| **テキストエンコーダ** | CLIP | CLIP + OpenCLIP | CLIP + T5 | CLIP + T5 |
| **ネイティブ解像度** | 512x512 | 1024x1024 | 1024x1024 | 1024x1024 |
| **テキスト描画** | 不可 | 苦手 | 良好 | 非常に得意 |
| **VRAM要件** | 4GB+ | 8GB+ | 12GB+ | 16GB+ |
| **推奨ステップ数** | 20-50 | 25-40 | 28 | 28 |
| **推奨CFG** | 7-12 | 5-8 | 4-7 | 2-5 |
| **LoRA対応** | 豊富 | 豊富 | 少ない | 増加中 |
| **ControlNet** | 豊富 | 豊富 | 限定的 | 限定的 |
| **コミュニティ** | 最大 | 大 | 小 | 増加中 |
| **商用ライセンス** | モデル依存 | モデル依存 | 要確認 | dev:研究, pro:商用 |

---

## 5. バッチ生成と自動化パイプライン

### コード例6: 大量画像の効率的な生成

```python
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class GenerationJob:
    """画像生成ジョブの定義"""
    prompt: str
    negative_prompt: str = "low quality, blurry, distorted, watermark"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 25
    guidance_scale: float = 7.0
    seed: Optional[int] = None
    lora_path: Optional[str] = None
    lora_weight: float = 0.8
    output_name: Optional[str] = None

class BatchGenerator:
    """バッチ画像生成パイプライン"""

    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0",
                 output_dir="./batch_output", device="cuda"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        self.pipe.enable_attention_slicing()

    def generate_single(self, job: GenerationJob, index: int = 0):
        """単一ジョブの実行"""
        # LoRA 適用
        if job.lora_path:
            self.pipe.load_lora_weights(job.lora_path)
            self.pipe.set_adapters(
                ["default"], adapter_weights=[job.lora_weight]
            )

        # シード設定
        generator = None
        if job.seed is not None:
            generator = torch.Generator(self.device).manual_seed(job.seed)

        start_time = time.time()

        image = self.pipe(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            width=job.width,
            height=job.height,
            num_inference_steps=job.num_inference_steps,
            guidance_scale=job.guidance_scale,
            generator=generator,
        ).images[0]

        elapsed = time.time() - start_time

        # ファイル保存
        if job.output_name:
            filename = f"{job.output_name}.png"
        else:
            filename = f"batch_{index:04d}.png"

        filepath = self.output_dir / filename
        image.save(filepath)

        # メタデータ保存
        metadata = {
            **asdict(job),
            "filename": filename,
            "generation_time_sec": round(elapsed, 2),
        }
        meta_path = self.output_dir / f"{filename.replace('.png', '_meta.json')}"
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

        # LoRA をアンロード
        if job.lora_path:
            self.pipe.unload_lora_weights()

        return filepath, metadata

    def run_batch(self, jobs: list[GenerationJob]):
        """バッチ実行"""
        results = []
        total = len(jobs)

        for i, job in enumerate(jobs):
            print(f"[{i+1}/{total}] {job.prompt[:60]}...")
            filepath, meta = self.generate_single(job, i)
            results.append(meta)
            print(f"  -> {filepath} ({meta['generation_time_sec']}秒)")

        # バッチサマリー保存
        summary_path = self.output_dir / "batch_summary.json"
        summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nバッチ完了: {total}枚生成")

        return results


# 使用例
generator = BatchGenerator(output_dir="./product_images")

jobs = [
    GenerationJob(
        prompt="Professional product photo of wireless earbuds, "
               "white background, studio lighting, 4K",
        seed=42,
        output_name="earbuds_front",
    ),
    GenerationJob(
        prompt="Professional product photo of wireless earbuds, "
               "in charging case, white background, studio lighting",
        seed=43,
        output_name="earbuds_case",
    ),
    GenerationJob(
        prompt="Lifestyle photo of person wearing wireless earbuds, "
               "jogging in park, natural lighting",
        seed=44,
        width=1024,
        height=768,
        output_name="earbuds_lifestyle",
    ),
]

results = generator.run_batch(jobs)
```

### コード例7: ComfyUI API によるバッチ生成

```python
import requests
import json
import time
from pathlib import Path
import websocket

class ComfyUIClient:
    """ComfyUI の API クライアント

    ComfyUI をサーバーモードで起動: python main.py --listen
    """

    def __init__(self, host="127.0.0.1", port=8188):
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"

    def queue_prompt(self, workflow: dict) -> str:
        """ワークフローをキューに追加"""
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow},
        )
        return response.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        """生成履歴を取得"""
        response = requests.get(f"{self.base_url}/history/{prompt_id}")
        return response.json()

    def get_image(self, filename: str, subfolder: str = "",
                  folder_type: str = "output") -> bytes:
        """生成画像をダウンロード"""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        }
        response = requests.get(
            f"{self.base_url}/view", params=params
        )
        return response.content

    def wait_for_completion(self, prompt_id: str, timeout: int = 120):
        """生成完了を待機"""
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(1)
        raise TimeoutError(f"Generation timed out after {timeout}s")

    def generate_from_workflow(self, workflow_path: str,
                                prompt_text: str, seed: int = -1):
        """ワークフローJSON からバッチ生成"""
        with open(workflow_path) as f:
            workflow = json.load(f)

        # プロンプトノードを更新
        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                if "positive" in node.get("_meta", {}).get("title", "").lower():
                    node["inputs"]["text"] = prompt_text
            if node.get("class_type") == "KSampler":
                if seed >= 0:
                    node["inputs"]["seed"] = seed

        prompt_id = self.queue_prompt(workflow)
        result = self.wait_for_completion(prompt_id)
        return result


# 使用例
# client = ComfyUIClient()
# result = client.generate_from_workflow(
#     "workflow_sdxl.json",
#     prompt_text="A beautiful sunset over Mount Fuji",
#     seed=42,
# )
```

---

## 6. 品質最適化テクニック

### コード例8: Img2Img による品質改善

```python
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
import torch
from PIL import Image

# 二段階生成: txt2img → img2img で品質向上
base_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

refine_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

prompt = "A detailed fantasy map of a fictional island, "
prompt += "hand-drawn style, parchment texture, labeled locations"
negative = "low quality, blurry, modern"

# Step 1: 基本画像生成
base_image = base_pipe(
    prompt=prompt,
    negative_prompt=negative,
    num_inference_steps=30,
    guidance_scale=7.0,
    width=1024,
    height=1024,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
base_image.save("map_base.png")

# Step 2: img2img でディテール追加
refined_image = refine_pipe(
    prompt=prompt + ", intricate details, fine lines",
    negative_prompt=negative,
    image=base_image,
    strength=0.35,          # 低い strength でディテールのみ追加
    num_inference_steps=25,
    guidance_scale=7.0,
).images[0]
refined_image.save("map_refined.png")

# Step 3: 部分的な Hires Fix (高解像度化)
# 画像を2倍に拡大してから img2img
upscaled = base_image.resize((2048, 2048), Image.LANCZOS)
hires_image = refine_pipe(
    prompt=prompt + ", ultra detailed, sharp lines",
    negative_prompt=negative,
    image=upscaled,
    strength=0.25,          # 構造を維持しつつ高解像度ディテール追加
    num_inference_steps=30,
    guidance_scale=7.0,
).images[0]
hires_image.save("map_hires.png")
```

### コード例9: プロンプトテンプレートと品質タグ管理

```python
class PromptQualityManager:
    """画像生成プロンプトの品質管理システム"""

    # 品質向上タグ (モデル共通)
    QUALITY_TAGS = {
        "high": [
            "masterpiece", "best quality", "highly detailed",
            "sharp focus", "professional",
        ],
        "photo": [
            "photorealistic", "DSLR quality", "natural lighting",
            "8K resolution", "RAW photo",
        ],
        "art": [
            "masterpiece", "artstation trending", "detailed illustration",
            "professional artwork", "concept art",
        ],
    }

    # ネガティブプロンプト (モデル共通)
    NEGATIVE_PRESETS = {
        "standard": (
            "low quality, worst quality, blurry, distorted, "
            "deformed, ugly, watermark, text, signature"
        ),
        "photo": (
            "low quality, worst quality, blurry, out of focus, "
            "overexposed, underexposed, noise, grain, "
            "watermark, text, illustration, painting, cartoon"
        ),
        "art": (
            "low quality, worst quality, blurry, distorted, "
            "deformed, ugly, amateur, bad anatomy, "
            "bad proportions, watermark, text"
        ),
        "product": (
            "low quality, blurry, distorted, watermark, text, "
            "shadow on background, cluttered background, "
            "multiple objects, person"
        ),
    }

    def build_prompt(self, subject, quality_preset="high",
                     additional_tags=None, style_tags=None):
        """構造化されたプロンプト生成"""
        parts = [subject]

        if quality_preset in self.QUALITY_TAGS:
            parts.extend(self.QUALITY_TAGS[quality_preset])

        if style_tags:
            parts.extend(style_tags)

        if additional_tags:
            parts.extend(additional_tags)

        return ", ".join(parts)

    def get_negative(self, preset="standard", additional=None):
        """ネガティブプロンプト取得"""
        neg = self.NEGATIVE_PRESETS.get(preset, self.NEGATIVE_PRESETS["standard"])
        if additional:
            neg += ", " + ", ".join(additional)
        return neg


# 使用例
pm = PromptQualityManager()

# フォトリアルな商品画像
prompt = pm.build_prompt(
    subject="Wireless bluetooth earbuds on marble surface",
    quality_preset="photo",
    additional_tags=["studio lighting", "45 degree angle"],
)
negative = pm.get_negative("product")

print(f"Prompt: {prompt}")
print(f"Negative: {negative}")

# アート作品
prompt_art = pm.build_prompt(
    subject="A dragon flying over a medieval castle",
    quality_preset="art",
    style_tags=["digital painting", "epic fantasy", "volumetric lighting"],
)
negative_art = pm.get_negative("art")
print(f"\nArt Prompt: {prompt_art}")
print(f"Art Negative: {negative_art}")
```

---

## 7. 比較表

### 比較表1: 3大プラットフォーム詳細比較

| 項目 | Stable Diffusion | DALL-E 3 | Midjourney | Flux |
|------|-----------------|----------|-----------|------|
| **実行環境** | ローカル/クラウド | クラウドAPI | Discord/Web | ローカル/API |
| **カスタマイズ** | 極めて高い | 低い | 低い | 高い |
| **LoRA対応** | あり | なし | なし | あり |
| **ControlNet** | あり | なし | なし | 限定的 |
| **日本語入力** | モデル依存 | 対応 | 限定的 | T5で対応 |
| **商用利用** | モデル依存 | 可 (有料) | 可 (有料) | dev:研究のみ |
| **テキスト描画** | 苦手 | 得意 | やや苦手 | 非常に得意 |
| **一貫性** | シード固定可 | 低い | スタイルリファレンス | シード固定可 |
| **バッチ生成** | 容易 | API可能 | 手動 | 容易 |
| **学習曲線** | 急 | 緩やか | 中程度 | 中程度 |
| **推奨CFG** | 5-8 | N/A | N/A | 2-5 |

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
| **テキスト入り画像** | Flux / DALL-E 3 | テキスト描画能力 |
| **ファインチューニング** | Stable Diffusion | LoRA/DreamBooth対応 |
| **ブランド一貫性** | Stable Diffusion + LoRA | 独自スタイルの学習 |

### 比較表3: スケジューラ(サンプラー)選択ガイド

| スケジューラ | 推奨ステップ | 速度 | 品質 | 用途 |
|------------|-----------|------|------|------|
| DPM++ 2M Karras | 25 | 速い | 高い | 汎用・最推奨 |
| Euler | 30 | 速い | 高い | 安定志向 |
| Euler Ancestral | 30 | 速い | 高い | バリエーション重視 |
| UniPC | 20 | 最速 | 良い | 高速生成 |
| Heun | 25 | 遅い | 最高 | 精密生成 |
| DDIM | 50 | 遅い | 良い | 補間・逆変換 |
| LCM | 4-8 | 超高速 | 中程度 | リアルタイム用途 |

---

## 8. アンチパターン

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

### アンチパターン3: ControlNet の conditioning_scale 過大

```
[問題]
ControlNet の conditioning_scale を 1.0 のままで
全ての生成を行う。

[なぜ問題か]
- 制御条件(エッジ、ポーズ等)に過度に拘束される
- 自然な表現やディテールが失われる
- 入力画像のノイズやアーティファクトまで忠実に再現してしまう

[正しいアプローチ]
- Canny Edge: 0.5-0.8 の範囲で調整
- OpenPose: 0.6-0.9 の範囲で調整
- Depth: 0.4-0.7 の範囲で調整
- 用途に応じて実験し、最適値を見つける
- Multi-ControlNet の場合、合計が 1.0-1.5 程度に抑える
```

### アンチパターン4: バッチ生成でエラーハンドリングなし

```
[問題]
100枚のバッチ生成を開始し、途中でエラーが発生すると
全体が停止して最初からやり直しになる。

[なぜ問題か]
- VRAM不足で途中クラッシュ
- API レート制限でタイムアウト
- ネットワーク障害でダウンロード失敗
- 数時間の作業が無駄になる

[正しいアプローチ]
1. try-except で個別ジョブのエラーをキャッチ
2. 進捗をファイルに記録し、途中再開可能にする
3. 生成済みファイルはスキップするロジックを実装
4. VRAM管理: torch.cuda.empty_cache() を定期実行
5. API利用時はレートリミットを考慮した待機を実装
```

---

## 9. FAQ

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
- **Flux:** dev版は研究用途のみ、pro版は商用利用可
- **共通:** 他者の著作物に酷似した画像の商用利用は法的リスク
- **推奨:** 生成画像を素材として使い、最終成果物に人間の創作を加える

### Q4: LoRA の学習に必要な画像枚数は?

**A:** 用途によって異なります:

| 目的 | 推奨枚数 | ステップ数 | 備考 |
|------|---------|----------|------|
| 顔の学習 | 10-20枚 | 500-1000 | 様々な角度・表情を含める |
| スタイル学習 | 20-50枚 | 1000-2000 | 一貫したスタイルの画像 |
| コンセプト学習 | 5-15枚 | 500-1500 | 対象物体のバリエーション |
| テクスチャ | 10-30枚 | 800-1500 | 様々な条件の撮影 |

### Q5: ComfyUI でカスタムノードを追加するには?

**A:**

```bash
# ComfyUI Manager の導入 (推奨)
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# 手動でカスタムノードを追加
git clone https://github.com/author/custom-node-repo.git

# 依存パッケージのインストール
pip install -r custom-node-repo/requirements.txt

# ComfyUI を再起動
```

代表的なカスタムノード:
- **ComfyUI-Manager:** ノード管理UI
- **ComfyUI-Impact-Pack:** 顔検出・修復
- **ComfyUI-AnimateDiff:** アニメーション生成
- **ComfyUI-Advanced-ControlNet:** ControlNet拡張
- **ComfyUI-IPAdapter:** 画像プロンプト

### Q6: DALL-E 3 の revised_prompt を制御する方法は?

**A:** DALL-E 3 は内部でプロンプトを書き換えますが、以下の方法で制御できます:

1. **I NEED to test how the tool works with prompts.** を冒頭に追加すると書き換えが最小限になる
2. プロンプトを可能な限り詳細に記述する (書き換えの余地を減らす)
3. 重要な要素を明示的に指定する (色、構図、スタイル)
4. ChatGPT 経由で使う場合、「プロンプトを変更せずに送って」と指示

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **SD系** | カスタマイズ性最高。LoRA、ControlNet で精密制御。学習コスト高 |
| **DALL-E 3** | 最も手軽。自然言語プロンプト。API統合が容易 |
| **Midjourney** | アート品質最高。Discord/Webで手軽。カスタマイズ性は低い |
| **Flux** | テキスト描画に優れる。Rectified Flow で効率的。急速にエコシステム拡大 |
| **モデル選択** | プロジェクト初期に確定。途中変更はスタイル崩壊リスク |
| **ワークフロー** | ComfyUI (ノード) / Automatic1111 (WebUI) / API統合 |
| **コスト** | ローカル=GPU投資、クラウド=従量/サブスク |
| **品質最適化** | スケジューラ選択 + CFG調整 + img2img仕上げ + アップスケーリング |

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
5. Zhang, L. et al. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." *ICCV 2023*. https://arxiv.org/abs/2302.05543
6. Hu, E.J. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*. https://arxiv.org/abs/2106.09685
7. Ruiz, N. et al. (2023). "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." *CVPR 2023*. https://arxiv.org/abs/2208.12242
