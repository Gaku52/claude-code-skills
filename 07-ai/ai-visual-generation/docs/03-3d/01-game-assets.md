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

    def generate_full_set(
        self, prompt: str, output_dir: str, size: int = 1024
    ) -> dict[str, str]:
        """完全なPBRテクスチャセットを生成"""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        diffuse = self.generate_diffuse(prompt, size)
        normal = self.generate_normal_map(diffuse)
        roughness = self.generate_roughness_map(diffuse)

        paths = {}
        for name, img in [
            ("diffuse", diffuse),
            ("normal", normal),
            ("roughness", roughness),
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

### 3.3 Unity向けアセットインポーター

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

        // 圧縮設定
        importer.textureCompression = TextureImporterCompression.CompressedHQ;
        importer.maxTextureSize = 2048;
    }
}
```

---

## 4. AIアニメーション生成

### 4.1 モーション生成手法の比較

| 手法 | 入力 | 品質 | 速度 | 適用場面 |
|------|------|------|------|---------|
| Mixamo | リグ付きモデル | 高 | 即時 | 汎用人型モーション |
| MDM | テキスト | 中〜高 | 数秒 | テキスト記述のモーション |
| MotionDiffuse | テキスト | 中〜高 | 数秒 | テキスト記述のモーション |
| Motion Matching | データベース | 高 | リアルタイム | ゲーム内遷移 |
| RAG+LLM | テキスト+DB | 高 | 数秒 | カスタムモーション |

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

---

## 7. アンチパターン

### 7.1 アンチパターン：AI生成モデルを無検証で大量投入

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

### 7.2 アンチパターン：ライセンス未確認のAI生成アセット

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

---

## 8. FAQ

### Q1: AI生成アセットだけでゲームを完成できるか？

**A**: インディーゲームやプロトタイプであれば十分可能。特にローポリスタイルやスタイライズドなアートスタイルでは、AI生成アセットの品質がそのまま使えることが多い。ただし、キャラクターのリギングやアニメーション、UI要素は手動調整が必要。AAA品質を目指す場合は、AI生成をベースに専門アーティストが仕上げるハイブリッドワークフローが現実的。

### Q2: AI生成アセットのスタイル統一はどうするか？

**A**: (1) プロンプトにスタイルガイドを含める（"low poly, pastel colors, hand-painted style"等）、(2) LoRAやControlNetでスタイルを固定、(3) 後処理シェーダーで統一感を出す、(4) カラーパレットを制限する。最も効果的なのは、少数の高品質リファレンス画像を用意し、img2imgやIP-Adapterで一貫したスタイルを適用する手法。

### Q3: ゲームエンジン別の推奨ワークフローは？

**A**:

| エンジン | 推奨フォーマット | インポート方法 | 自動化 |
|---------|---------------|--------------|--------|
| Unity | glTF 2.0 / FBX | AssetPostprocessor | C#スクリプト |
| Unreal Engine | FBX / USD | Python Editor Script | Blueprint/Python |
| Godot | glTF 2.0 | EditorImportPlugin | GDScript |

---

## 9. まとめ

| カテゴリ | ポイント |
|---------|---------|
| テクスチャ | Stable Diffusionでシームレス生成、PBRセット自動作成 |
| 3Dモデル | Meshy/TripoSRで生成、リトポ+LODで最適化 |
| アニメーション | MDMでテキストからモーション、Mixamoで汎用動作 |
| レベル | WFC+LLMでインテリジェントな自動配置 |
| 品質管理 | 自動チェックゲート+アートディレクターレビュー |
| パフォーマンス | ポリゴン予算、テクスチャアトラス、LOD必須 |
| ライセンス | 必ず利用規約を確認、法務レビューを挟む |

---

## 次に読むべきガイド

- [00-3d-generation.md](./00-3d-generation.md) — AI 3Dモデル生成技術の基礎
- ゲームエンジン統合 — UnityやUEとの詳細な連携方法
- プロシージャル生成 — 高度なPCG技術の応用

---

## 参考文献

1. Meshy API ドキュメント — https://docs.meshy.ai/
2. Gumin, "WaveFunctionCollapse" — https://github.com/mxgmn/WaveFunctionCollapse
3. Tevet et al., "Human Motion Diffusion Model" — https://guytevet.github.io/mdm-page/
4. Unity Asset Pipeline ドキュメント — https://docs.unity3d.com/Manual/AssetWorkflow.html
5. KhronosGroup glTF 仕様 — https://www.khronos.org/gltf/
