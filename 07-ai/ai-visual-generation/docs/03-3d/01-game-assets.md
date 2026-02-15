# AIによるゲームアセット生成 実践ガイド

> テクスチャ、3Dモデル、アニメーション、レベルデザインなど、ゲーム開発に必要なアセットをAIで効率的に生成・活用するための実践的ワークフローを解説する。

---

## この章で学ぶこと

1. **ゲームアセットの種類別にAI生成手法を使い分け**、プロトタイプから本番品質まで段階的に活用できる
2. **テクスチャ、3Dモデル、アニメーションの自動生成パイプライン**を構築し、ゲームエンジンと統合できる
3. **AI生成アセットの品質管理・ライセンス・パフォーマンス最適化**の実務的な課題に対処できる

---

## 1. ゲームアセット生成の全体像

### 1.1 アセット種別とAI生成手法のマッピング

```
┌─────────────────────────────────────────────────────────┐
│          ゲームアセット × AI生成手法マトリクス              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  テクスチャ ──────> Stable Diffusion / DALL-E            │
│  │  ├ ディフューズマップ    (色・模様)                    │
│  │  ├ ノーマルマップ        (凹凸)                       │
│  │  ├ ラフネスマップ        (粗さ)                       │
│  │  └ 環境マップ/HDRI      (照明)                       │
│  │                                                      │
│  3Dモデル ──────> TripoSR / Meshy / Shap-E              │
│  │  ├ プロップ(小物)        (椅子、武器、アイテム)        │
│  │  ├ 建物・構造物          (城、家、橋)                 │
│  │  └ キャラクター          (要手動調整)                 │
│  │                                                      │
│  アニメーション ──> MDM / MotionDiffuse / Mixamo         │
│  │  ├ 歩行・走行            (モーションキャプチャ代替)    │
│  │  ├ アクション            (攻撃、ジャンプ)             │
│  │  └ フェイシャル          (表情変化)                   │
│  │                                                      │
│  レベルデザイン ──> WaveFunctionCollapse / PCG + LLM     │
│     ├ 地形生成              (高さマップ)                 │
│     ├ ダンジョン配置        (部屋・通路生成)             │
│     └ オブジェクト配置      (自動デコレーション)          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 AI生成アセットの品質段階

```
品質レベル        用途                   AI依存度    手動調整
──────────────────────────────────────────────────────────
Lv.1 プレースホルダ  プロトタイプ           100% AI    なし
Lv.2 ドラフト       内部レビュー用          90% AI     軽微
Lv.3 ゲームレディ   インディーゲーム        60% AI     中程度
Lv.4 AAA品質       商用ゲーム             30% AI     大幅
──────────────────────────────────────────────────────────
```

### 1.3 ジャンル別アセット要件マトリクス

```
ゲームジャンル別 AI 生成アセットの適合度:

ジャンル         テクスチャ  3Dモデル  アニメ  レベル  適合度
──────────────────────────────────────────────────────────
ローグライク      ◎         ◎        ○      ◎      最高
サンドボックス    ◎         ○        △      ◎      高い
モバイルRPG      ◎         ○        ○      ○      高い
インディー2D     ◎         -        △      ○      中程度
AAA オープンワールド ○      △        △      ○      やや低い
格闘ゲーム       ○         △        ✕      -      低い
──────────────────────────────────────────────────────────

◎ = そのまま使える  ○ = 調整すれば使える  △ = 大幅調整必要
✕ = 非推奨          - = 該当なし
```

### 1.4 AI生成パイプラインの全体アーキテクチャ

```
┌──────────────────────────────────────────────────────────────┐
│            AI ゲームアセット生成パイプライン v2.0                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [入力]                                                      │
│  ├── ゲームデザインドキュメント (GDD)                          │
│  ├── スタイルガイド / リファレンスアート                        │
│  ├── 技術仕様 (ポリゴン予算、テクスチャ解像度)                  │
│  └── プロンプトライブラリ                                      │
│       │                                                      │
│       v                                                      │
│  ┌──────────────────────────────────────────┐               │
│  │  Phase 1: AI生成 (バッチ処理)              │               │
│  │  ├── テクスチャ: SD/DALL-E → PBRセット     │               │
│  │  ├── 3Dモデル: Meshy/TripoSR → GLB        │               │
│  │  ├── アニメ: MDM/Mixamo → BVH/FBX         │               │
│  │  └── レベル: WFC/PCG → マップデータ         │               │
│  └──────────────┬───────────────────────────┘               │
│                  │                                           │
│                  v                                           │
│  ┌──────────────────────────────────────────┐               │
│  │  Phase 2: 自動品質チェック                  │               │
│  │  ├── ポリゴン数チェック                     │               │
│  │  ├── テクスチャサイズ検証                   │               │
│  │  ├── UV重なりチェック                       │               │
│  │  ├── マテリアル数検証                       │               │
│  │  └── スタイル一貫性スコア (CLIP類似度)       │               │
│  └──────────────┬───────────────────────────┘               │
│                  │                                           │
│                  v                                           │
│  ┌──────────────────────────────────────────┐               │
│  │  Phase 3: 後処理・最適化                    │               │
│  │  ├── 自動リトポロジー (Instant Meshes)      │               │
│  │  ├── LOD 自動生成                           │               │
│  │  ├── テクスチャアトラス化                    │               │
│  │  ├── コリジョンメッシュ生成                  │               │
│  │  └── メタデータ付与 (タグ、カテゴリ)         │               │
│  └──────────────┬───────────────────────────┘               │
│                  │                                           │
│                  v                                           │
│  [出力] → ゲームエンジン対応アセットバンドル                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. テクスチャ生成

### 2.1 PBRテクスチャセットの自動生成

```python
# AI + 画像処理によるPBRテクスチャセット生成
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2

class PBRTextureGenerator:
    """PBR(Physically Based Rendering)テクスチャセットを生成"""

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")

    def generate_diffuse(
        self,
        prompt: str,
        size: int = 1024,
        seamless: bool = True,
    ) -> Image.Image:
        """ディフューズ（アルベド）マップを生成"""
        full_prompt = (
            f"seamless tileable texture of {prompt}, "
            f"top-down view, flat lighting, no shadows, "
            f"high resolution PBR texture"
        )
        image = self.pipe(
            full_prompt,
            width=size,
            height=size,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]

        if seamless:
            image = self._make_seamless(image)
        return image

    def generate_normal_map(self, diffuse: Image.Image) -> Image.Image:
        """ディフューズマップからノーマルマップを推定"""
        gray = np.array(diffuse.convert("L"), dtype=np.float32) / 255.0

        # Sobelフィルタで勾配計算
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 法線ベクトルの構築
        normal = np.zeros((*gray.shape, 3), dtype=np.float32)
        normal[:, :, 0] = -grad_x  # X成分
        normal[:, :, 1] = -grad_y  # Y成分
        normal[:, :, 2] = 1.0      # Z成分

        # 正規化
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (norm + 1e-8)

        # [−1,1] → [0,255]
        normal_map = ((normal + 1.0) * 0.5 * 255).astype(np.uint8)
        return Image.fromarray(normal_map)

    def generate_roughness_map(self, diffuse: Image.Image) -> Image.Image:
        """ディフューズマップからラフネスマップを推定"""
        gray = diffuse.convert("L")
        # 高周波成分が多い = 粗い表面
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=5))
        diff = np.abs(
            np.array(gray, dtype=np.float32)
            - np.array(blurred, dtype=np.float32)
        )
        # 正規化してラフネスマップに
        roughness = (diff / diff.max() * 200 + 55).clip(0, 255)
        return Image.fromarray(roughness.astype(np.uint8))

    def generate_ao_map(self, diffuse: Image.Image) -> Image.Image:
        """ディフューズマップからアンビエントオクルージョンマップを推定"""
        gray = np.array(diffuse.convert("L"), dtype=np.float32)
        # 複数スケールのブラーで擬似AO
        ao = np.ones_like(gray)
        for radius in [3, 7, 15, 31]:
            blurred = cv2.GaussianBlur(gray, (0, 0), radius)
            local_ao = gray / (blurred + 1e-8)
            local_ao = np.clip(local_ao, 0.5, 1.5)
            ao *= local_ao
        ao = np.clip(ao * 128 + 64, 0, 255)
        return Image.fromarray(ao.astype(np.uint8))

    def generate_height_map(self, diffuse: Image.Image) -> Image.Image:
        """ディフューズマップから高さマップを推定"""
        gray = np.array(diffuse.convert("L"), dtype=np.float32)
        # エッジ検出で凹凸を推定
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        # ガウスブラーで滑らかに
        height = cv2.GaussianBlur(gray, (0, 0), 3)
        # エッジ周辺を強調
        height = height - edges.astype(np.float32) * 0.3
        height = np.clip(height, 0, 255)
        return Image.fromarray(height.astype(np.uint8))

    def generate_full_set(
        self, prompt: str, output_dir: str, size: int = 1024
    ) -> dict[str, str]:
        """完全なPBRテクスチャセットを生成"""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        diffuse = self.generate_diffuse(prompt, size)
        normal = self.generate_normal_map(diffuse)
        roughness = self.generate_roughness_map(diffuse)
        ao = self.generate_ao_map(diffuse)
        height = self.generate_height_map(diffuse)

        paths = {}
        for name, img in [
            ("diffuse", diffuse),
            ("normal", normal),
            ("roughness", roughness),
            ("ao", ao),
            ("height", height),
        ]:
            path = f"{output_dir}/{name}.png"
            img.save(path)
            paths[name] = path

        return paths

    def _make_seamless(self, image: Image.Image) -> Image.Image:
        """タイリング可能なシームレステクスチャに変換"""
        arr = np.array(image, dtype=np.float32)
        h, w = arr.shape[:2]
        blend_size = w // 4

        # 左右のブレンド
        for i in range(blend_size):
            alpha = i / blend_size
            arr[:, i] = arr[:, i] * alpha + arr[:, w - blend_size + i] * (1 - alpha)

        # 上下のブレンド
        for i in range(blend_size):
            alpha = i / blend_size
            arr[i, :] = arr[i, :] * alpha + arr[h - blend_size + i, :] * (1 - alpha)

        return Image.fromarray(arr.astype(np.uint8))
```

### 2.2 テクスチャバリエーション生成パイプライン

```python
class TextureVariationPipeline:
    """同一スタイルのテクスチャバリエーションを一括生成"""

    def __init__(self, generator: PBRTextureGenerator):
        self.generator = generator

    def generate_material_family(
        self,
        base_material: str,
        variations: list[str],
        output_dir: str,
        style_reference: str = "",
    ) -> dict[str, dict[str, str]]:
        """同一マテリアルファミリーのバリエーション生成

        例:
            base_material = "stone wall"
            variations = ["mossy", "cracked", "weathered", "clean"]
        """
        results = {}
        style_suffix = f", {style_reference}" if style_reference else ""

        for variation in variations:
            prompt = f"{variation} {base_material}{style_suffix}"
            var_dir = f"{output_dir}/{variation.replace(' ', '_')}"
            paths = self.generator.generate_full_set(prompt, var_dir)
            results[variation] = paths

        return results

    def generate_tileset(
        self,
        theme: str,
        tile_types: list[str],
        output_dir: str,
        size: int = 512,
    ) -> dict[str, str]:
        """タイルセット（地面、壁、天井等）を統一スタイルで生成

        例:
            theme = "medieval dungeon"
            tile_types = ["floor_stone", "wall_brick",
                         "ceiling_wooden", "pillar_marble"]
        """
        results = {}
        style_prompt = (
            f"{theme} style, consistent art direction, "
            f"game texture, seamless tileable"
        )

        for tile_type in tile_types:
            readable = tile_type.replace("_", " ")
            prompt = f"{readable}, {style_prompt}"
            tile_dir = f"{output_dir}/{tile_type}"
            paths = self.generator.generate_full_set(prompt, tile_dir, size)
            results[tile_type] = paths

        return results

    def generate_damage_progression(
        self,
        material: str,
        damage_levels: int = 4,
        output_dir: str = "./damage_progression",
    ) -> list[dict[str, str]]:
        """ダメージ段階テクスチャ（耐久度に応じた見た目変化）"""
        results = []
        damage_descriptions = [
            "pristine clean undamaged",
            "slightly worn scratched",
            "moderately damaged cracked chipped",
            "heavily damaged broken destroyed",
        ]

        for level in range(min(damage_levels, len(damage_descriptions))):
            desc = damage_descriptions[level]
            prompt = f"{desc} {material}, game texture"
            level_dir = f"{output_dir}/damage_{level}"
            paths = self.generator.generate_full_set(prompt, level_dir)
            results.append(paths)

        return results
```

### 2.3 HDRI環境マップの生成

```python
class HDRIGenerator:
    """ゲーム用 HDRI 環境マップ生成"""

    def __init__(self, pipe):
        self.pipe = pipe

    def generate_panoramic_hdri(
        self,
        environment: str,
        time_of_day: str = "noon",
        weather: str = "clear",
        width: int = 2048,
        height: int = 1024,
    ) -> Image.Image:
        """パノラミック HDRI 環境マップを生成

        Args:
            environment: 環境の説明 ("forest", "city", "desert" 等)
            time_of_day: 時間帯 ("dawn", "noon", "sunset", "night")
            weather: 天候 ("clear", "cloudy", "overcast", "stormy")
        """
        prompt = (
            f"360 degree equirectangular panorama HDRI, "
            f"{environment}, {time_of_day}, {weather} sky, "
            f"seamless panoramic environment map, "
            f"high dynamic range, realistic lighting"
        )
        image = self.pipe(
            prompt,
            width=width,
            height=height,
            num_inference_steps=30,
            guidance_scale=7.0,
        ).images[0]

        return image

    def generate_skybox_faces(
        self,
        environment: str,
        face_size: int = 1024,
    ) -> dict[str, Image.Image]:
        """6面スカイボックステクスチャを生成"""
        faces = {}
        directions = {
            "front": "facing forward",
            "back": "facing backward",
            "left": "facing left",
            "right": "facing right",
            "top": "looking straight up at sky",
            "bottom": "looking straight down at ground",
        }
        for face_name, direction in directions.items():
            prompt = (
                f"skybox texture, {environment}, {direction}, "
                f"seamless edges, game environment"
            )
            faces[face_name] = self.pipe(
                prompt,
                width=face_size,
                height=face_size,
                num_inference_steps=25,
            ).images[0]

        return faces
```

---

## 3. 3Dモデル生成とゲームエンジン統合

### 3.1 ゲームアセット向け3D生成パイプライン

```
テキスト/画像プロンプト
        │
        v
┌──────────────────┐
│  AI 3D生成       │  Meshy API / TripoSR
│  (ハイポリ)      │  100K-500K ポリゴン
└──────────────────┘
        │
        v
┌──────────────────┐
│  自動リトポロジー │  Instant Meshes / Blender
│  (ローポリ化)    │  1K-50K ポリゴン
└──────────────────┘
        │
        v
┌──────────────────┐
│  UV展開          │  xatlas / Blender
│  + テクスチャベイク│  法線/AO/ディフューズ
└──────────────────┘
        │
        v
┌──────────────────┐
│  LOD生成         │  Level of Detail
│  LOD0: 10K faces │  近距離用
│  LOD1: 2K faces  │  中距離用
│  LOD2: 500 faces │  遠距離用
└──────────────────┘
        │
        v
┌──────────────────┐
│  エクスポート     │  glTF 2.0 / FBX
│  + メタデータ     │  コリジョン / タグ
└──────────────────┘
```

### 3.2 Meshy APIを使った商用品質のアセット生成

```python
# Meshy API: テキストから商用品質3Dアセットを生成
import requests
import time
from pathlib import Path

class MeshyAssetGenerator:
    """Meshy APIによるゲームアセット生成"""

    BASE_URL = "https://api.meshy.ai/v2"

    def __init__(self, api_key: str):
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def text_to_3d(
        self,
        prompt: str,
        negative_prompt: str = "",
        art_style: str = "game-asset",
        topology: str = "quad",
        target_polycount: int = 30000,
    ) -> str:
        """テキストから3Dモデルを生成（タスクIDを返す）"""
        payload = {
            "mode": "preview",  # preview → refine の2段階
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "art_style": art_style,
            "topology": topology,
            "target_polycount": target_polycount,
        }
        resp = requests.post(
            f"{self.BASE_URL}/text-to-3d",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["result"]

    def image_to_3d(
        self,
        image_url: str,
        target_polycount: int = 30000,
    ) -> str:
        """参照画像から3Dモデルを生成"""
        payload = {
            "image_url": image_url,
            "target_polycount": target_polycount,
            "enable_pbr": True,
        }
        resp = requests.post(
            f"{self.BASE_URL}/image-to-3d",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["result"]

    def refine_model(self, preview_task_id: str) -> str:
        """プレビューモデルをリファインして高品質化"""
        payload = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
        }
        resp = requests.post(
            f"{self.BASE_URL}/text-to-3d",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["result"]

    def wait_and_download(
        self, task_id: str, output_dir: str, poll_interval: int = 10
    ) -> dict[str, str]:
        """タスク完了を待ってダウンロード"""
        while True:
            resp = requests.get(
                f"{self.BASE_URL}/text-to-3d/{task_id}",
                headers=self.headers,
            )
            data = resp.json()

            if data["status"] == "SUCCEEDED":
                break
            elif data["status"] == "FAILED":
                raise RuntimeError(f"生成失敗: {data.get('error')}")

            print(f"進捗: {data.get('progress', 0)}%")
            time.sleep(poll_interval)

        # ダウンロード
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        paths = {}
        for fmt in ["glb", "fbx", "obj"]:
            url = data.get(f"model_urls", {}).get(fmt)
            if url:
                path = f"{output_dir}/{task_id}.{fmt}"
                content = requests.get(url).content
                with open(path, "wb") as f:
                    f.write(content)
                paths[fmt] = path

        # テクスチャダウンロード
        for tex in data.get("texture_urls", []):
            tex_path = f"{output_dir}/{tex['name']}"
            content = requests.get(tex["url"]).content
            with open(tex_path, "wb") as f:
                f.write(content)
            paths[tex["name"]] = tex_path

        return paths

# 使用例
generator = MeshyAssetGenerator(api_key="your-api-key")
task_id = generator.text_to_3d(
    prompt="medieval wooden barrel, game prop, low poly style",
    art_style="game-asset",
    target_polycount=5000,
)
assets = generator.wait_and_download(task_id, "./assets/barrel")
```

### 3.3 バッチ生成とアセットカタログ管理

```python
import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class AssetMetadata:
    """AI生成アセットのメタデータ"""
    asset_id: str
    name: str
    category: str              # "prop", "character", "environment", "weapon"
    subcategory: str           # "furniture", "nature", "architecture"
    prompt: str
    model_used: str            # "meshy", "triposr", "shap-e"
    poly_count: int = 0
    texture_resolution: int = 0
    lod_levels: int = 0
    file_formats: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    style: str = ""            # "realistic", "stylized", "low-poly"
    license: str = ""
    created_at: str = ""
    quality_score: float = 0.0
    status: str = "generated"  # "generated", "reviewed", "approved", "rejected"

class AssetCatalog:
    """AI生成アセットのカタログ管理"""

    def __init__(self, catalog_path: str = "./asset_catalog.json"):
        self.catalog_path = catalog_path
        self.assets: dict[str, AssetMetadata] = {}
        self._load()

    def _load(self):
        if Path(self.catalog_path).exists():
            with open(self.catalog_path) as f:
                data = json.load(f)
                for asset_id, meta in data.items():
                    self.assets[asset_id] = AssetMetadata(**meta)

    def save(self):
        with open(self.catalog_path, "w") as f:
            json.dump(
                {k: asdict(v) for k, v in self.assets.items()},
                f, indent=2, ensure_ascii=False,
            )

    def register(self, metadata: AssetMetadata) -> str:
        """アセットをカタログに登録"""
        if not metadata.asset_id:
            metadata.asset_id = hashlib.md5(
                f"{metadata.prompt}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
        if not metadata.created_at:
            metadata.created_at = datetime.now().isoformat()
        self.assets[metadata.asset_id] = metadata
        self.save()
        return metadata.asset_id

    def search(
        self,
        category: str = "",
        tags: list[str] = None,
        style: str = "",
        min_quality: float = 0.0,
    ) -> list[AssetMetadata]:
        """カタログからアセットを検索"""
        results = []
        for asset in self.assets.values():
            if category and asset.category != category:
                continue
            if tags and not all(t in asset.tags for t in tags):
                continue
            if style and asset.style != style:
                continue
            if asset.quality_score < min_quality:
                continue
            results.append(asset)
        return results

    def batch_generate(
        self,
        generator: MeshyAssetGenerator,
        asset_list: list[dict],
        output_base: str = "./assets",
    ) -> list[str]:
        """アセットリストから一括生成"""
        generated_ids = []
        for spec in asset_list:
            try:
                task_id = generator.text_to_3d(
                    prompt=spec["prompt"],
                    art_style=spec.get("art_style", "game-asset"),
                    target_polycount=spec.get("polycount", 10000),
                )
                output_dir = f"{output_base}/{spec['name']}"
                paths = generator.wait_and_download(task_id, output_dir)

                metadata = AssetMetadata(
                    asset_id="",
                    name=spec["name"],
                    category=spec.get("category", "prop"),
                    subcategory=spec.get("subcategory", ""),
                    prompt=spec["prompt"],
                    model_used="meshy",
                    file_formats=list(paths.keys()),
                    tags=spec.get("tags", []),
                    style=spec.get("style", "game-asset"),
                )
                asset_id = self.register(metadata)
                generated_ids.append(asset_id)
                print(f"生成完了: {spec['name']} ({asset_id})")

            except Exception as e:
                print(f"生成失敗: {spec['name']}: {e}")

        return generated_ids

# 使用例: バッチ生成
catalog = AssetCatalog()
asset_specs = [
    {
        "name": "wooden_barrel",
        "prompt": "medieval wooden barrel, game prop, low poly",
        "category": "prop",
        "subcategory": "container",
        "tags": ["medieval", "wood", "container"],
        "polycount": 5000,
    },
    {
        "name": "iron_sword",
        "prompt": "iron longsword, fantasy weapon, game asset",
        "category": "weapon",
        "subcategory": "melee",
        "tags": ["fantasy", "metal", "sword"],
        "polycount": 8000,
    },
    {
        "name": "stone_well",
        "prompt": "stone water well, medieval village prop",
        "category": "prop",
        "subcategory": "architecture",
        "tags": ["medieval", "stone", "water"],
        "polycount": 12000,
    },
]
```

### 3.4 Unity向けアセットインポーター

```csharp
// Unity: AI生成アセットの自動インポートと設定
using UnityEngine;
using UnityEditor;
using System.IO;

public class AIAssetImporter : AssetPostprocessor
{
    // AI生成アセットの自動設定
    void OnPreprocessModel()
    {
        // AI生成アセットフォルダを判定
        if (!assetPath.Contains("AI_Generated")) return;

        ModelImporter importer = assetImporter as ModelImporter;

        // メッシュ最適化設定
        importer.optimizeMeshPolygons = true;
        importer.optimizeMeshVertices = true;
        importer.meshCompression = ModelImporterMeshCompression.Medium;

        // コリジョン生成
        importer.addCollider = true;

        // LOD自動設定
        importer.importNormals = ModelImporterNormals.Calculate;
        importer.normalCalculationMode =
            ModelImporterNormalCalculationMode.AreaAndAngleWeighted;

        // スケール調整（AI生成モデルはスケールがバラバラ）
        importer.globalScale = 1.0f;
        importer.useFileScale = false;
    }

    void OnPreprocessTexture()
    {
        if (!assetPath.Contains("AI_Generated")) return;

        TextureImporter importer = assetImporter as TextureImporter;

        // テクスチャ種別の自動判定
        string fileName = Path.GetFileNameWithoutExtension(assetPath).ToLower();

        if (fileName.Contains("normal"))
        {
            importer.textureType = TextureImporterType.NormalMap;
        }
        else if (fileName.Contains("roughness") || fileName.Contains("metallic"))
        {
            importer.textureType = TextureImporterType.Default;
            importer.sRGBTexture = false; // リニア空間
        }
        else if (fileName.Contains("ao") || fileName.Contains("occlusion"))
        {
            importer.textureType = TextureImporterType.Default;
            importer.sRGBTexture = false;
        }
        else if (fileName.Contains("height") || fileName.Contains("displacement"))
        {
            importer.textureType = TextureImporterType.Default;
            importer.sRGBTexture = false;
        }

        // 圧縮設定
        importer.textureCompression = TextureImporterCompression.CompressedHQ;
        importer.maxTextureSize = 2048;
    }
}
```

### 3.5 Unreal Engine向けアセットインポーター

```python
# Unreal Engine: Python Editor Scripting によるAI生成アセットの自動インポート
import unreal

class AIAssetImporterUE:
    """Unreal Engine 向け AI 生成アセットインポーター"""

    def __init__(self, asset_base_path="/Game/AI_Generated"):
        self.asset_base_path = asset_base_path
        self.asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

    def import_fbx_with_settings(
        self,
        source_path: str,
        destination_path: str,
        asset_name: str,
    ):
        """FBXファイルをAI生成アセット向けの設定でインポート"""
        task = unreal.AssetImportTask()
        task.set_editor_property("automated", True)
        task.set_editor_property("filename", source_path)
        task.set_editor_property("destination_path", destination_path)
        task.set_editor_property("destination_name", asset_name)
        task.set_editor_property("replace_existing", True)
        task.set_editor_property("save", True)

        # FBX インポート設定
        options = unreal.FbxImportUI()
        options.set_editor_property("import_mesh", True)
        options.set_editor_property("import_textures", True)
        options.set_editor_property("import_materials", True)
        options.set_editor_property("import_as_skeletal", False)

        # スタティックメッシュ設定
        options.static_mesh_import_data.set_editor_property(
            "combine_meshes", True
        )
        options.static_mesh_import_data.set_editor_property(
            "generate_lightmap_u_vs", True
        )
        options.static_mesh_import_data.set_editor_property(
            "auto_generate_collision", True
        )

        task.set_editor_property("options", options)
        self.asset_tools.import_asset_tasks([task])

        return task.get_editor_property("imported_object_paths")

    def auto_setup_lod(
        self,
        static_mesh_path: str,
        lod_count: int = 3,
    ):
        """LODの自動セットアップ"""
        mesh = unreal.EditorAssetLibrary.load_asset(static_mesh_path)
        if not mesh:
            return

        # LOD グループ設定
        reduction_settings = []
        for i in range(1, lod_count):
            settings = unreal.MeshReductionSettings()
            # LOD レベルに応じたポリゴン削減率
            percent = max(0.1, 1.0 - (i * 0.3))
            settings.set_editor_property("percent_triangles", percent)
            reduction_settings.append(settings)

        unreal.EditorStaticMeshLibrary.set_lod_count(mesh, lod_count)

    def batch_import_directory(
        self,
        source_directory: str,
        destination_path: str = None,
    ):
        """ディレクトリ内の全FBX/GLBファイルを一括インポート"""
        import os
        if destination_path is None:
            destination_path = self.asset_base_path

        supported_extensions = [".fbx", ".glb", ".gltf", ".obj"]
        imported = []

        for root, dirs, files in os.walk(source_directory):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_extensions:
                    source = os.path.join(root, filename)
                    name = os.path.splitext(filename)[0]
                    result = self.import_fbx_with_settings(
                        source, destination_path, name
                    )
                    imported.extend(result)

        return imported
```

---

## 4. AIアニメーション生成

### 4.1 モーション生成手法の比較

| 手法 | 入力 | 品質 | 速度 | 適用場面 |
|------|------|------|------|---------|
| Mixamo | リグ付きモデル | 高 | 即時 | 汎用人型モーション |
| MDM | テキスト | 中~高 | 数秒 | テキスト記述のモーション |
| MotionDiffuse | テキスト | 中~高 | 数秒 | テキスト記述のモーション |
| Motion Matching | データベース | 高 | リアルタイム | ゲーム内遷移 |
| RAG+LLM | テキスト+DB | 高 | 数秒 | カスタムモーション |
| MotionGPT | テキスト | 高 | 数秒 | 自然言語からモーション |
| MoMask | テキスト/マスク | 中~高 | 数秒 | 部分的モーション編集 |

### 4.2 テキストからモーション生成

```python
# Motion Diffusion Model (MDM) を使ったテキストからモーション生成
import torch
from motion_diffusion_model import MDM, HumanML3DDataset

def generate_motion_from_text(
    prompt: str,
    duration: float = 3.0,
    num_samples: int = 1,
    model_path: str = "pretrained/mdm_humanml.pth",
) -> dict:
    """テキストプロンプトからモーションを生成"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルロード
    model = MDM.load_pretrained(model_path).to(device)
    model.eval()

    # フレーム数計算（30fps想定）
    n_frames = int(duration * 30)

    with torch.no_grad():
        motions = model.generate(
            texts=[prompt] * num_samples,
            lengths=[n_frames] * num_samples,
            guidance_scale=2.5,
        )

    # 関節位置データを返す
    return {
        "joints": motions.cpu().numpy(),  # (samples, frames, joints, 3)
        "fps": 30,
        "duration": duration,
        "prompt": prompt,
    }

def export_to_bvh(motion_data: dict, output_path: str):
    """モーションデータをBVH形式でエクスポート"""
    from motion_utils import joints_to_bvh

    joints = motion_data["joints"][0]  # 最初のサンプル
    bvh_data = joints_to_bvh(
        joints,
        fps=motion_data["fps"],
        skeleton="humanml3d",
    )

    with open(output_path, "w") as f:
        f.write(bvh_data)

# 使用例
motion = generate_motion_from_text(
    prompt="a person swings a sword overhead then steps forward",
    duration=2.5,
)
export_to_bvh(motion, "sword_attack.bvh")
```

### 4.3 モーションブレンドと遷移生成

```python
class MotionBlender:
    """AI生成モーション間のスムーズな遷移を生成"""

    def __init__(self, model):
        self.model = model

    def blend_motions(
        self,
        motion_a: dict,
        motion_b: dict,
        blend_frames: int = 15,
        method: str = "slerp",
    ) -> dict:
        """2つのモーション間をブレンドして繋ぐ

        Args:
            motion_a: 前のモーション
            motion_b: 次のモーション
            blend_frames: ブレンドに使うフレーム数
            method: "slerp" (球面線形補間) or "lerp" (線形補間)
        """
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp

        joints_a = motion_a["joints"][0]  # (frames, joints, 3)
        joints_b = motion_b["joints"][0]

        # 最後のblend_framesフレームと最初のblend_framesフレームをブレンド
        end_a = joints_a[-blend_frames:]
        start_b = joints_b[:blend_frames]

        blended = np.zeros_like(end_a)
        for i in range(blend_frames):
            t = i / (blend_frames - 1)  # 0.0 ~ 1.0
            if method == "slerp":
                # 球面線形補間（回転に適した補間）
                t = self._ease_in_out(t)
            blended[i] = end_a[i] * (1 - t) + start_b[i] * t

        # 結合
        result = np.concatenate([
            joints_a[:-blend_frames],
            blended,
            joints_b[blend_frames:],
        ], axis=0)

        return {
            "joints": result[np.newaxis],
            "fps": motion_a["fps"],
            "duration": len(result) / motion_a["fps"],
            "prompt": f"{motion_a['prompt']} -> {motion_b['prompt']}",
        }

    def _ease_in_out(self, t: float) -> float:
        """Ease-in-out 補間カーブ"""
        return t * t * (3 - 2 * t)

    def create_animation_state_machine(
        self,
        states: dict[str, dict],
        transitions: list[tuple[str, str]],
        blend_frames: int = 10,
    ) -> dict:
        """アニメーションステートマシン用の遷移データを生成

        Args:
            states: {"idle": motion_data, "walk": motion_data, ...}
            transitions: [("idle", "walk"), ("walk", "run"), ...]
        """
        transition_clips = {}
        for state_a, state_b in transitions:
            if state_a in states and state_b in states:
                key = f"{state_a}_to_{state_b}"
                transition_clips[key] = self.blend_motions(
                    states[state_a],
                    states[state_b],
                    blend_frames=blend_frames,
                )

        return {
            "states": states,
            "transitions": transition_clips,
        }
```

---

## 5. プロシージャルレベル生成

### 5.1 Wave Function Collapse + AIの組み合わせ

```python
# WFC + LLMによるインテリジェントレベル生成
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Tile:
    """タイルデータ"""
    id: str
    asset_path: str
    connections: dict  # 方向 -> 接続タイプ
    weight: float = 1.0
    tags: list[str] = None

class WFCLevelGenerator:
    """Wave Function Collapseベースのレベル生成"""

    def __init__(self, width: int, height: int, tiles: list[Tile]):
        self.width = width
        self.height = height
        self.tiles = tiles
        # 各セルの可能なタイルセット
        self.grid = [
            [set(range(len(tiles))) for _ in range(width)]
            for _ in range(height)
        ]

    def collapse(self) -> np.ndarray:
        """WFCアルゴリズムでグリッドを確定"""
        while not self._is_fully_collapsed():
            # 最もエントロピーが低いセルを選択
            y, x = self._find_min_entropy_cell()
            if y is None:
                raise RuntimeError("矛盾が発生: バックトラックが必要")

            # 重み付きランダムで1タイルに確定
            possible = list(self.grid[y][x])
            weights = [self.tiles[t].weight for t in possible]
            total = sum(weights)
            probs = [w / total for w in weights]
            chosen = np.random.choice(possible, p=probs)
            self.grid[y][x] = {chosen}

            # 制約伝播
            self._propagate(x, y)

        # 結果をグリッドに変換
        result = np.zeros((self.height, self.width), dtype=int)
        for y in range(self.height):
            for x in range(self.width):
                result[y][x] = list(self.grid[y][x])[0]
        return result

    def _is_fully_collapsed(self) -> bool:
        return all(
            len(self.grid[y][x]) == 1
            for y in range(self.height)
            for x in range(self.width)
        )

    def _find_min_entropy_cell(self) -> tuple:
        min_entropy = float("inf")
        min_pos = (None, None)
        for y in range(self.height):
            for x in range(self.width):
                e = len(self.grid[y][x])
                if 1 < e < min_entropy:
                    min_entropy = e
                    min_pos = (y, x)
        return min_pos

    def _propagate(self, start_x: int, start_y: int):
        """制約伝播: 隣接セルの可能性を絞り込む"""
        stack = [(start_x, start_y)]
        directions = [(0, -1, "up"), (0, 1, "down"), (-1, 0, "left"), (1, 0, "right")]
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}

        while stack:
            x, y = stack.pop()
            for dx, dy, direction in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # 現在のセルから許可される接続
                    allowed = set()
                    for t in self.grid[y][x]:
                        conn_type = self.tiles[t].connections.get(direction)
                        for nt in self.grid[ny][nx]:
                            opp_conn = self.tiles[nt].connections.get(opposite[direction])
                            if conn_type == opp_conn:
                                allowed.add(nt)

                    new_possible = self.grid[ny][nx] & allowed
                    if len(new_possible) < len(self.grid[ny][nx]):
                        self.grid[ny][nx] = new_possible
                        if len(new_possible) == 0:
                            raise RuntimeError(f"矛盾: ({nx},{ny})")
                        stack.append((nx, ny))
```

### 5.2 LLMベースのレベルデザインアシスタント

```python
class LLMLevelDesigner:
    """LLMを活用したインテリジェントレベルデザイン"""

    def __init__(self, llm_client, wfc_generator: WFCLevelGenerator):
        self.llm = llm_client
        self.wfc = wfc_generator

    async def design_level_from_description(
        self,
        description: str,
        width: int = 20,
        height: int = 20,
    ) -> dict:
        """自然言語の説明からレベルを生成

        Args:
            description: "暗い洞窟で、中央に大きな湖がある。
                         北側に宝箱部屋、南側に入口がある。"
        """
        # Step 1: LLMでレベルの構造を計画
        plan_prompt = f"""
ゲームレベルを設計してください。

説明: {description}
サイズ: {width}x{height}

以下のJSON形式で出力してください:
{{
    "zones": [
        {{"name": "入口", "position": [x, y], "size": [w, h], "type": "entrance"}},
        {{"name": "メインエリア", "position": [x, y], "size": [w, h], "type": "open"}},
    ],
    "connections": [
        {{"from": "入口", "to": "メインエリア", "type": "corridor"}},
    ],
    "poi": [
        {{"name": "宝箱", "zone": "北の部屋", "importance": "high"}},
    ],
    "atmosphere": "dark_cave"
}}
"""
        plan = await self.llm.generate(plan_prompt)
        level_plan = json.loads(plan)

        # Step 2: WFCで詳細なタイル配置を生成
        # ゾーン情報をWFCの重みに反映
        for zone in level_plan["zones"]:
            self._apply_zone_weights(zone)

        grid = self.wfc.collapse()

        # Step 3: POI（ポイントオブインタレスト）を配置
        poi_placements = self._place_points_of_interest(
            grid, level_plan["poi"], level_plan["zones"]
        )

        return {
            "grid": grid.tolist(),
            "plan": level_plan,
            "poi": poi_placements,
            "metadata": {
                "width": width,
                "height": height,
                "description": description,
            }
        }

    def _apply_zone_weights(self, zone: dict):
        """ゾーン情報に基づきタイルの重みを調整"""
        zone_type = zone["type"]
        x, y = zone["position"]
        w, h = zone["size"]

        # ゾーンタイプに応じたタイル重みの調整
        type_preferences = {
            "entrance": {"door": 5.0, "floor": 2.0, "wall": 1.0},
            "open": {"floor": 3.0, "wall": 0.5},
            "corridor": {"floor": 2.0, "wall": 2.0},
            "treasure": {"floor": 2.0, "decoration": 3.0},
        }
        # 実装は省略: ゾーン領域のタイル重みを動的に変更

    def _place_points_of_interest(
        self, grid, pois, zones
    ) -> list[dict]:
        """POIをグリッド上に配置"""
        placements = []
        for poi in pois:
            # 対象ゾーン内のフロアタイルからランダムに選択
            zone = next(z for z in zones if z["name"] == poi["zone"])
            zx, zy = zone["position"]
            zw, zh = zone["size"]

            candidates = []
            for dy in range(zh):
                for dx in range(zw):
                    gx, gy = zx + dx, zy + dy
                    if 0 <= gx < len(grid[0]) and 0 <= gy < len(grid):
                        tile_id = grid[gy][gx]
                        # フロアタイルの場合のみ候補
                        if self.wfc.tiles[tile_id].id.startswith("floor"):
                            candidates.append((gx, gy))

            if candidates:
                chosen = candidates[np.random.randint(len(candidates))]
                placements.append({
                    "name": poi["name"],
                    "position": list(chosen),
                    "importance": poi["importance"],
                })

        return placements
```

---

## 6. パフォーマンス最適化

### 6.1 AI生成アセットの最適化チェックリスト

| チェック項目 | 基準値 | ツール |
|------------|--------|-------|
| ポリゴン数 | モバイル: <5K, PC: <50K | Blender Decimate |
| テクスチャサイズ | モバイル: 512px, PC: 2048px | ImageMagick |
| ドローコール | オブジェクトあたり1-3 | Unity Profiler |
| UV重なり | 0% | UV Checker |
| 法線方向 | 全て外向き | Blender Recalculate |
| マテリアル数 | 1-2個/オブジェクト | アトラス化 |
| LOD段階 | 3段階以上 | 自動LOD生成 |
| テクスチャ圧縮 | BC7/ASTC | GPU圧縮 |
| メッシュバウンド | 適切なAABB | エンジン確認 |
| コリジョン | 単純化メッシュ | プリミティブ代用 |

### 6.2 自動最適化パイプライン

```python
class AssetOptimizer:
    """AI生成アセットの自動最適化パイプライン"""

    def __init__(self, target_platform: str = "pc"):
        self.platform = target_platform
        self.budgets = self._get_platform_budgets()

    def _get_platform_budgets(self) -> dict:
        """プラットフォーム別のアセット予算"""
        budgets = {
            "mobile": {
                "max_poly": 5000,
                "max_texture": 512,
                "max_materials": 1,
                "texture_format": "ASTC",
            },
            "pc": {
                "max_poly": 50000,
                "max_texture": 2048,
                "max_materials": 3,
                "texture_format": "BC7",
            },
            "console": {
                "max_poly": 30000,
                "max_texture": 2048,
                "max_materials": 2,
                "texture_format": "BC7",
            },
        }
        return budgets.get(self.platform, budgets["pc"])

    def validate_asset(self, asset_path: str) -> dict:
        """アセットのバリデーション"""
        import trimesh

        mesh = trimesh.load(asset_path)
        results = {
            "poly_count": len(mesh.faces),
            "vertex_count": len(mesh.vertices),
            "is_watertight": mesh.is_watertight,
            "has_degenerate_faces": mesh.is_empty,
            "bounding_box": mesh.bounding_box.extents.tolist(),
        }

        # 予算チェック
        results["poly_over_budget"] = (
            results["poly_count"] > self.budgets["max_poly"]
        )
        results["needs_decimation"] = results["poly_over_budget"]

        # UV チェック
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uv = mesh.visual.uv
            results["has_uv"] = True
            results["uv_range"] = {
                "min": uv.min(axis=0).tolist(),
                "max": uv.max(axis=0).tolist(),
            }
        else:
            results["has_uv"] = False

        return results

    def optimize_mesh(
        self,
        input_path: str,
        output_path: str,
        target_faces: int = None,
    ) -> dict:
        """メッシュの自動最適化"""
        import trimesh

        mesh = trimesh.load(input_path)
        original_faces = len(mesh.faces)

        if target_faces is None:
            target_faces = min(original_faces, self.budgets["max_poly"])

        if original_faces > target_faces:
            # Quadric Edge Collapse Decimation
            ratio = target_faces / original_faces
            mesh = mesh.simplify_quadric_decimation(target_faces)

        # 法線の再計算
        mesh.fix_normals()

        # エクスポート
        mesh.export(output_path)

        return {
            "original_faces": original_faces,
            "optimized_faces": len(mesh.faces),
            "reduction_ratio": 1 - len(mesh.faces) / original_faces,
        }

    def generate_lods(
        self,
        input_path: str,
        output_dir: str,
        lod_ratios: list[float] = None,
    ) -> list[str]:
        """LODメッシュの自動生成"""
        import trimesh
        from pathlib import Path

        if lod_ratios is None:
            lod_ratios = [1.0, 0.5, 0.25, 0.1]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        mesh = trimesh.load(input_path)
        base_faces = len(mesh.faces)
        lod_paths = []

        for i, ratio in enumerate(lod_ratios):
            target = max(100, int(base_faces * ratio))
            if ratio < 1.0:
                lod_mesh = mesh.simplify_quadric_decimation(target)
            else:
                lod_mesh = mesh.copy()

            lod_path = f"{output_dir}/lod{i}.glb"
            lod_mesh.export(lod_path)
            lod_paths.append(lod_path)

        return lod_paths
```

### 6.3 テクスチャアトラス自動化

```python
class TextureAtlasBuilder:
    """複数アセットのテクスチャをアトラスに統合"""

    def __init__(self, atlas_size: int = 2048):
        self.atlas_size = atlas_size

    def build_atlas(
        self,
        textures: list[dict],
        output_path: str,
    ) -> dict:
        """テクスチャアトラスを生成

        Args:
            textures: [{"name": "barrel", "image": PIL.Image, "size": (256, 256)}, ...]
        """
        from PIL import Image
        import numpy as np

        # パッキング（単純なシェルフアルゴリズム）
        atlas = Image.new("RGBA", (self.atlas_size, self.atlas_size), (0, 0, 0, 0))
        uv_mapping = {}

        current_x = 0
        current_y = 0
        row_height = 0

        for tex_info in textures:
            img = tex_info["image"]
            w, h = img.size

            # 行をまたぐ場合
            if current_x + w > self.atlas_size:
                current_x = 0
                current_y += row_height
                row_height = 0

            # アトラスに収まらない場合
            if current_y + h > self.atlas_size:
                raise ValueError(
                    f"アトラスサイズ {self.atlas_size}px に収まりません"
                )

            # テクスチャを配置
            atlas.paste(img, (current_x, current_y))

            # UV マッピング情報を記録
            uv_mapping[tex_info["name"]] = {
                "offset": (
                    current_x / self.atlas_size,
                    current_y / self.atlas_size,
                ),
                "scale": (
                    w / self.atlas_size,
                    h / self.atlas_size,
                ),
            }

            current_x += w
            row_height = max(row_height, h)

        atlas.save(output_path)
        return {
            "atlas_path": output_path,
            "uv_mapping": uv_mapping,
            "utilization": self._calculate_utilization(uv_mapping),
        }

    def _calculate_utilization(self, mapping: dict) -> float:
        """アトラスの利用率を計算"""
        total_used = sum(
            m["scale"][0] * m["scale"][1] for m in mapping.values()
        )
        return total_used  # 0.0 ~ 1.0
```

---

## 7. スタイル統一とアートディレクション

### 7.1 CLIP ベースのスタイル一貫性チェック

```python
class StyleConsistencyChecker:
    """CLIP を使ったアセットのスタイル一貫性チェック"""

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def compute_style_similarity(
        self,
        reference_image: Image.Image,
        target_image: Image.Image,
    ) -> float:
        """リファレンスとターゲットのスタイル類似度を計算"""
        import torch

        inputs = self.processor(
            images=[reference_image, target_image],
            return_tensors="pt",
        )

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        # コサイン類似度
        similarity = torch.nn.functional.cosine_similarity(
            features[0:1], features[1:2]
        ).item()

        return similarity

    def batch_check_consistency(
        self,
        reference: Image.Image,
        assets: list[Image.Image],
        threshold: float = 0.75,
    ) -> list[dict]:
        """複数アセットのスタイル一貫性を一括チェック"""
        results = []
        for i, asset in enumerate(assets):
            sim = self.compute_style_similarity(reference, asset)
            results.append({
                "index": i,
                "similarity": sim,
                "passes": sim >= threshold,
                "grade": (
                    "A" if sim >= 0.9 else
                    "B" if sim >= 0.8 else
                    "C" if sim >= 0.7 else
                    "D" if sim >= 0.6 else "F"
                ),
            })
        return results
```

### 7.2 LoRA / IP-Adapter によるスタイル固定

```python
class StyleFixedGenerator:
    """LoRA や IP-Adapter を使ってスタイルを固定した生成"""

    def __init__(self, base_model: str, lora_path: str = None):
        from diffusers import StableDiffusionXLPipeline
        import torch

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model, torch_dtype=torch.float16
        ).to("cuda")

        if lora_path:
            self.pipe.load_lora_weights(lora_path)

    def generate_with_style_lock(
        self,
        prompts: list[str],
        style_prompt: str = "low poly, hand-painted, stylized game asset",
        color_palette: list[str] = None,
    ) -> list[Image.Image]:
        """統一されたスタイルで複数アセットのテクスチャを生成"""
        results = []

        palette_suffix = ""
        if color_palette:
            palette_suffix = f", color palette: {', '.join(color_palette)}"

        for prompt in prompts:
            full_prompt = f"{prompt}, {style_prompt}{palette_suffix}"
            image = self.pipe(
                full_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
            ).images[0]
            results.append(image)

        return results
```

---

## 8. アンチパターン

### 8.1 アンチパターン：AI生成モデルを無検証で大量投入

```
NG: 1000個のAI生成プロップをそのままシーンに配置
  - ポリゴン数合計: 5000万 → フレームレート激減
  - テクスチャメモリ: 20GB → VRAM不足
  - スタイルの不統一 → 世界観の崩壊

OK: 品質ゲートを設けた段階的投入
  1. AI生成 → 自動品質チェック（ポリ数、テクスチャサイズ）
  2. スタイル一貫性チェック（色パレット、ディテール密度）
  3. パフォーマンスプロファイリング
  4. アートディレクターレビュー
  5. 本番シーンに投入
```

### 8.2 アンチパターン：ライセンス未確認のAI生成アセット

```
NG: AI生成アセットのライセンスを確認せずに商用リリース
  - 学習データの著作権問題
  - APIの利用規約違反
  - 類似性の高い既存作品との権利衝突

OK: ライセンスチェックフローの構築
  1. 使用したAIツール/APIの利用規約を確認
  2. 商用利用可否を明確化
  3. 生成物の権利帰属を確認
  4. 法務チームによるレビュー
  5. 利用規約のスナップショットを保存
```

### 8.3 アンチパターン：スケール不統一の放置

```
NG: AI生成モデル間でスケールがバラバラ
  - 樽が家と同じ大きさ
  - 剣がキャラクターより大きい
  - コリジョンが見た目と合わない

OK: スケール正規化パイプライン
  1. 基準オブジェクト（1m の立方体）を設定
  2. カテゴリ別のスケール基準表を作成
     - 小物: 0.1-0.5m
     - 家具: 0.5-2.0m
     - 建物: 3-20m
  3. インポート時に自動スケーリング
  4. バウンディングボックスの検証
```

### 8.4 アンチパターン：リトポロジーなしの直接使用

```
NG: AI生成のハイポリメッシュをそのままゲームに使用
  - 三角形のみの不規則なトポロジー
  - ポリゴン密度の偏り（平坦な面にも大量のポリゴン）
  - アニメーションが破綻する
  - LOD生成の品質が低い

OK: 適切なリトポロジーワークフロー
  1. AI生成ハイポリ → Instant Meshes で自動リトポ
  2. Quad ベースの均等なトポロジー取得
  3. 重要な形状エッジにループを維持
  4. UV展開 → テクスチャベイク（法線マップでディテール保持）
  5. LOD生成で段階的な簡略化
```

---

## 9. トラブルシューティング

### 9.1 よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| テクスチャがタイリングで目立つ | シームの処理不足 | シームレス化処理を適用、ブレンド幅を増加 |
| 3Dモデルに穴がある | 非多様体メッシュ | メッシュ修復ツール（MeshFix）で自動修正 |
| ノーマルマップが反転 | OpenGL/DirectX規格の違い | Y軸を反転、エンジン設定を確認 |
| テクスチャが伸びる | UV展開の失敗 | xatlasで再展開、ストレッチチェック |
| LOD切り替えがチラつく | LOD間の差が大きすぎる | 中間LODを追加、ブレンド距離を調整 |
| マテリアルが黒く表示 | メタリックマップの不備 | メタリック値を0に、PBR設定を確認 |
| アニメーションが浮く | ルートモーションの不整合 | 足のIKを適用、地面との接地チェック |
| ゲームの起動が遅い | テクスチャのメモリ不足 | ストリーミング有効化、ミップマップ生成 |
| フレームレートが低い | ドローコール過多 | テクスチャアトラス化、インスタンシング |
| モデルのシルエットが不自然 | ポリゴン不足 | 重要なエッジにポリゴンを追加 |

### 9.2 プラットフォーム別の制約対応

```
モバイル向け最適化チェックリスト:
┌─────────────────────────────────────────────┐
│  [ ] ポリゴン数: オブジェクトあたり500-5000   │
│  [ ] テクスチャ: 256-512px、ASTC圧縮          │
│  [ ] マテリアル: オブジェクトあたり1つ         │
│  [ ] シェーダー: Unlit または Simple Lit       │
│  [ ] ドローコール: シーン全体で100以下         │
│  [ ] メモリ: テクスチャ合計 200MB 以下         │
│  [ ] LOD: 2段階以上                           │
│  [ ] アニメーション: ボーン数30以下            │
│  [ ] パーティクル: 同時表示50以下              │
│  [ ] ライトマップ: プリベイク推奨              │
└─────────────────────────────────────────────┘

PC/コンソール向け最適化チェックリスト:
┌─────────────────────────────────────────────┐
│  [ ] ポリゴン数: オブジェクトあたり5K-50K     │
│  [ ] テクスチャ: 1024-4096px、BC7圧縮         │
│  [ ] マテリアル: オブジェクトあたり1-3         │
│  [ ] シェーダー: PBR Standard                  │
│  [ ] ドローコール: シーン全体で2000以下        │
│  [ ] メモリ: テクスチャ合計 2GB 以下           │
│  [ ] LOD: 3-4段階                             │
│  [ ] アニメーション: ボーン数100以下           │
│  [ ] ライティング: リアルタイムGI対応          │
│  [ ] レイトレーシング: メッシュ最適化済み       │
└─────────────────────────────────────────────┘
```

---

## 10. FAQ

### Q1: AI生成アセットだけでゲームを完成できるか？

**A**: インディーゲームやプロトタイプであれば十分可能。特にローポリスタイルやスタイライズドなアートスタイルでは、AI生成アセットの品質がそのまま使えることが多い。ただし、キャラクターのリギングやアニメーション、UI要素は手動調整が必要。AAA品質を目指す場合は、AI生成をベースに専門アーティストが仕上げるハイブリッドワークフローが現実的。2025年現在、インディーゲームの約30%が何らかのAI生成アセットを使用しているとされる。

### Q2: AI生成アセットのスタイル統一はどうするか？

**A**: (1) プロンプトにスタイルガイドを含める（"low poly, pastel colors, hand-painted style"等）、(2) LoRAやControlNetでスタイルを固定、(3) 後処理シェーダーで統一感を出す、(4) カラーパレットを制限する。最も効果的なのは、少数の高品質リファレンス画像を用意し、img2imgやIP-Adapterで一貫したスタイルを適用する手法。CLIP ベースのスタイル類似度チェックを自動化すると、パイプラインに組み込みやすい。

### Q3: ゲームエンジン別の推奨ワークフローは？

**A**:

| エンジン | 推奨フォーマット | インポート方法 | 自動化 |
|---------|---------------|--------------|--------|
| Unity | glTF 2.0 / FBX | AssetPostprocessor | C#スクリプト |
| Unreal Engine | FBX / USD | Python Editor Script | Blueprint/Python |
| Godot | glTF 2.0 | EditorImportPlugin | GDScript |

### Q4: AI生成モーションの品質を向上させるには？

**A**: (1) テキストプロンプトを具体的にする（「歩く」→「ゆっくりと慎重に歩く、周囲を警戒しながら」）、(2) 生成後にモーションエディタで微調整する、(3) 足のIKを適用して接地感を改善する、(4) モーションブレンドで遷移を滑らかにする、(5) 複数サンプルを生成して最良のものを選択する。特に格闘ゲームやアクションゲームでは、手動調整が必要になるケースが多い。

### Q5: テクスチャ生成でシームレス化がうまくいかない場合は？

**A**: (1) プロンプトに "seamless tileable" を必ず含める、(2) 生成後にシームレス化処理を適用する（ブレンド幅を画像の1/4程度に）、(3) 複数回生成して最もシームレスなものを選択する、(4) 後処理で境界部分にクローンスタンプ的な修正を適用する、(5) ControlNet の Tile モードを使うと品質が向上する場合がある。

### Q6: AI生成アセットのバージョン管理はどうすべきか？

**A**: (1) Git LFS でアセットファイルを管理する（テクスチャ、メッシュ）、(2) 生成に使用したプロンプト、パラメータ、モデルバージョンをメタデータとして保存する、(3) アセットカタログ（JSON/DB）で管理し、検索・フィルタを可能にする、(4) プロンプトのバージョン管理をコードと同様に行う、(5) 再生成可能性を担保するためにシード値を記録する。

---

## 11. まとめ

| カテゴリ | ポイント |
|---------|---------|
| テクスチャ | Stable Diffusionでシームレス生成、PBRセット自動作成、バリエーション一括生成 |
| 3Dモデル | Meshy/TripoSRで生成、リトポ+LODで最適化、バッチ生成で効率化 |
| アニメーション | MDMでテキストからモーション、Mixamoで汎用動作、ブレンドで遷移生成 |
| レベル | WFC+LLMでインテリジェントな自動配置、自然言語からレベル設計 |
| 品質管理 | 自動チェックゲート+CLIP一貫性チェック+アートディレクターレビュー |
| パフォーマンス | ポリゴン予算、テクスチャアトラス、LOD必須、プラットフォーム別最適化 |
| ライセンス | 必ず利用規約を確認、法務レビューを挟む、メタデータで記録 |
| スタイル統一 | LoRA/IP-Adapter活用、カラーパレット制限、CLIPスコアチェック |

---

## 次に読むべきガイド

- [00-3d-generation.md](./00-3d-generation.md) -- AI 3Dモデル生成技術の基礎
- ゲームエンジン統合 -- UnityやUEとの詳細な連携方法
- プロシージャル生成 -- 高度なPCG技術の応用

---

## 参考文献

1. Meshy API ドキュメント -- https://docs.meshy.ai/
2. Gumin, "WaveFunctionCollapse" -- https://github.com/mxgmn/WaveFunctionCollapse
3. Tevet et al., "Human Motion Diffusion Model" -- https://guytevet.github.io/mdm-page/
4. Unity Asset Pipeline ドキュメント -- https://docs.unity3d.com/Manual/AssetWorkflow.html
5. KhronosGroup glTF 仕様 -- https://www.khronos.org/gltf/
6. Instant Meshes -- https://github.com/wjakob/instant-meshes
7. xatlas UV パッキング -- https://github.com/jpcy/xatlas
8. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision (CLIP)" -- https://arxiv.org/abs/2103.00020
