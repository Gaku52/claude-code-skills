# AI 3Dモデル生成 技術・ツール・活用ガイド

> NeRF、3D Gaussian Splatting、Point-E、Shap-E など最先端のAI 3D生成技術を体系的に解説し、原理の理解からツール活用、本番ワークフローへの統合までを網羅する。

---

## この章で学ぶこと

1. **主要なAI 3D生成手法**（NeRF、3DGS、拡散モデル系）の原理と特徴を理解し、適材適所で使い分けられる
2. **テキスト/画像から3Dモデルを生成**する実践的なワークフローを構築できる
3. **生成した3Dモデルの後処理・最適化・エクスポート**までの本番パイプラインを設計できる

---

## 1. AI 3D生成技術の全体像

### 1.1 技術マップ

```
┌───────────────────────────────────────────────────────────┐
│                AI 3D生成技術マップ                          │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  入力                   手法              出力             │
│  ─────                 ─────            ─────            │
│                                                           │
│  複数画像 ────> [NeRF]          ────> 暗黙的3D表現         │
│                [3D Gaussian     ────> 点群+ガウシアン      │
│                 Splatting]                                 │
│                                                           │
│  テキスト ────> [Shap-E]        ────> メッシュ/点群        │
│                [Point-E]        ────> 点群→メッシュ        │
│                [DreamFusion]    ────> NeRF→メッシュ        │
│                [Meshy/Tripo]    ────> メッシュ(商用API)     │
│                                                           │
│  単一画像 ────> [Zero-1-to-3]   ────> マルチビュー→3D      │
│                [One-2-3-45]     ────> メッシュ             │
│                [TripoSR]        ────> メッシュ             │
│                [InstantMesh]    ────> メッシュ             │
│                                                           │
│  3Dスキャン ──> [COLMAP+NeRF]   ────> 高品質3Dシーン       │
│                [3DGS]           ────> リアルタイム描画      │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 1.2 手法間の進化系譜

```
2020  NeRF (Mildenhall et al.)
  │     └─ 暗黙的ニューラル表現の革命
  │
2021  Instant-NGP
  │     └─ ハッシュエンコーディングで高速化
  │
2022  DreamFusion (Google)
  │     └─ テキストから3D (SDS Loss)
  │   Point-E / Shap-E (OpenAI)
  │     └─ 拡散モデルで直接3D生成
  │
2023  3D Gaussian Splatting
  │     └─ 明示的表現でリアルタイム描画
  │   TripoSR / InstantMesh
  │     └─ 単一画像から高速3D再構成
  │
2024  大規模3D基盤モデル
  │     └─ LRM, GRM, Trellis等
  │
2025  統合マルチモーダル3D生成
        └─ テキスト・画像・動画から高品質3D
```

---

## 2. NeRF（Neural Radiance Fields）

### 2.1 原理

```
カメラ位置 (x,y,z) + 視線方向 (θ,φ)
            │
            v
   ┌──────────────────┐
   │  MLPネットワーク   │
   │  (多層パーセプトロン) │
   │                  │
   │  位置エンコーディング│
   │       ↓          │
   │  256次元隠れ層×8   │
   │       ↓          │
   │  密度σ + 色(r,g,b)│
   └──────────────────┘
            │
            v
   ボリュームレンダリング
   C(r) = Σ T_i (1-exp(-σ_i δ_i)) c_i
            │
            v
   新規視点画像の生成
```

### 2.2 nerfstudioでのNeRF実装

```python
# nerfstudioを使ったNeRFパイプライン
# インストール: pip install nerfstudio

# 1. データ準備（COLMAPでカメラ推定）
# ns-process-data images --data ./my_images --output-dir ./processed

# 2. 学習実行
# ns-train nerfacto --data ./processed

# 3. Pythonからの制御
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.trainer import TrainerConfig
from pathlib import Path

def train_nerf(
    data_dir: str,
    output_dir: str,
    method: str = "nerfacto",
    max_iterations: int = 30000,
):
    """NeRFモデルを学習する"""
    config = method_configs[method]
    config.data = Path(data_dir)
    config.output_dir = Path(output_dir)
    config.max_num_iterations = max_iterations

    # GPU設定
    config.machine.num_gpus = 1

    # 学習パラメータ調整
    config.pipeline.model.near_plane = 0.01
    config.pipeline.model.far_plane = 1000.0

    trainer = config.setup()
    trainer.train()
    return trainer

# 4. メッシュエクスポート
# ns-export poisson --load-config outputs/.../config.yml
#   --output-dir exports/ --target-num-faces 50000
```

---

## 3. 3D Gaussian Splatting

### 3.1 NeRFとの比較

| 項目 | NeRF | 3D Gaussian Splatting |
|------|------|----------------------|
| 3D表現 | 暗黙的（MLP） | 明示的（ガウシアン点群） |
| レンダリング | レイマーチング | ラスタライゼーション |
| 描画速度 | 遅い（数秒/フレーム） | リアルタイム（100+ FPS） |
| 学習速度 | 数時間 | 数十分 |
| メモリ使用量 | 小（MLPの重み） | 大（数百万ガウシアン） |
| 編集容易性 | 困難 | 容易（点の操作） |
| メッシュ化 | Marching Cubes | 表面再構成が必要 |
| 品質 | 高（特に反射） | 高（特にテクスチャ） |

### 3.2 3DGSの実装

```python
# 3D Gaussian Splatting パイプライン
# リポジトリ: https://github.com/graphdeco-inria/gaussian-splatting

import subprocess
from pathlib import Path

class GaussianSplattingPipeline:
    """3D Gaussian Splatting の学習・レンダリングパイプライン"""

    def __init__(self, repo_path: str = "./gaussian-splatting"):
        self.repo = Path(repo_path)

    def prepare_data(self, images_dir: str, output_dir: str):
        """COLMAPでSfM（Structure from Motion）を実行"""
        cmd = [
            "python", str(self.repo / "convert.py"),
            "-s", images_dir,
            "--output", output_dir,
        ]
        subprocess.run(cmd, check=True)

    def train(
        self,
        data_dir: str,
        output_dir: str,
        iterations: int = 30000,
        densify_until: int = 15000,
        sh_degree: int = 3,
    ):
        """3DGSモデルを学習"""
        cmd = [
            "python", str(self.repo / "train.py"),
            "-s", data_dir,
            "-m", output_dir,
            "--iterations", str(iterations),
            "--densify_until_iter", str(densify_until),
            "--sh_degree", str(sh_degree),
            # 品質調整パラメータ
            "--position_lr_init", "0.00016",
            "--scaling_lr", "0.005",
            "--opacity_lr", "0.05",
        ]
        subprocess.run(cmd, check=True)

    def render(self, model_dir: str, output_dir: str):
        """学習済みモデルからレンダリング"""
        cmd = [
            "python", str(self.repo / "render.py"),
            "-m", model_dir,
            "--output", output_dir,
        ]
        subprocess.run(cmd, check=True)

    def export_ply(self, model_dir: str) -> str:
        """PLY形式で点群をエクスポート"""
        ply_path = Path(model_dir) / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        return str(ply_path)
```

---

## 4. テキスト/画像からの3D生成

### 4.1 主要手法の比較

| 手法 | 入力 | 生成速度 | 品質 | 出力形式 | 商用利用 |
|------|------|---------|------|---------|---------|
| Shap-E | テキスト/画像 | 数秒 | 中 | メッシュ/NeRF | MIT |
| Point-E | テキスト/画像 | 数分 | 中 | 点群 | MIT |
| DreamFusion | テキスト | 数時間 | 高 | NeRF | 研究用 |
| TripoSR | 単一画像 | 数秒 | 高 | メッシュ | MIT |
| InstantMesh | 単一画像 | 数十秒 | 高 | メッシュ | Apache 2.0 |
| Meshy API | テキスト/画像 | 数分 | 高 | メッシュ(テクスチャ付) | 商用API |
| Tripo API | テキスト/画像 | 数十秒 | 高 | メッシュ(テクスチャ付) | 商用API |

### 4.2 Shap-Eによるテキストからの3D生成

```python
# Shap-E: テキストから3Dモデル生成
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

def generate_3d_from_text(
    prompt: str,
    output_path: str = "output.obj",
    batch_size: int = 1,
    guidance_scale: float = 15.0,
    num_steps: int = 64,
) -> str:
    """テキストプロンプトから3Dメッシュを生成"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルロード
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    # 潜在表現のサンプリング
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=num_steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # メッシュにデコード
    for i, latent in enumerate(latents):
        mesh = decode_latent_mesh(xm, latent).tri_mesh()
        with open(output_path, "w") as f:
            mesh.write_obj(f)

    return output_path

# 使用例
generate_3d_from_text(
    prompt="a red sports car, detailed, high quality",
    output_path="car.obj",
)
```

### 4.3 TripoSRによる単一画像からの3D再構成

```python
# TripoSR: 単一画像から高速3D再構成
import torch
from tsr.system import TSR
from PIL import Image

def image_to_3d(
    image_path: str,
    output_path: str = "output.obj",
    chunk_size: int = 8192,
    mc_resolution: int = 256,
) -> str:
    """単一画像から3Dメッシュを生成"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルロード
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model = model.to(device)

    # 画像読み込み・前処理
    image = Image.open(image_path)

    # 3D生成（数秒で完了）
    with torch.no_grad():
        scene_codes = model([image], device=device)

    # メッシュ抽出（Marching Cubes）
    meshes = model.extract_mesh(
        scene_codes,
        resolution=mc_resolution,
    )

    # エクスポート
    meshes[0].export(output_path)
    return output_path
```

---

## 5. 後処理・最適化パイプライン

### 5.1 3Dモデル後処理フロー

```
生成された3Dモデル
        │
        v
┌──────────────────┐
│  メッシュクリーン  │  孤立頂点・重複面の除去
└──────────────────┘
        │
        v
┌──────────────────┐
│  メッシュ簡略化    │  ポリゴン数の最適化
└──────────────────┘
        │
        v
┌──────────────────┐
│  UV展開          │  テクスチャマッピング準備
└──────────────────┘
        │
        v
┌──────────────────┐
│  テクスチャベイク  │  色・法線マップの生成
└──────────────────┘
        │
        v
┌──────────────────┐
│  フォーマット変換  │  glTF/FBX/USD等へ変換
└──────────────────┘
        │
        v
   最終3Dアセット
```

### 5.2 Trimeshによるメッシュ後処理

```python
# Trimeshを使ったメッシュ後処理
import trimesh
import numpy as np

class MeshPostProcessor:
    """生成された3Dメッシュの後処理"""

    def __init__(self, mesh_path: str):
        self.mesh = trimesh.load(mesh_path)

    def clean(self) -> "MeshPostProcessor":
        """メッシュのクリーンアップ"""
        # 重複頂点の統合
        self.mesh.merge_vertices()
        # 退化面（面積ゼロ）の除去
        self.mesh.remove_degenerate_faces()
        # 孤立頂点の除去
        self.mesh.remove_unreferenced_vertices()
        # 法線の再計算
        self.mesh.fix_normals()
        return self

    def simplify(self, target_faces: int = 10000) -> "MeshPostProcessor":
        """メッシュの簡略化（ポリゴン削減）"""
        if len(self.mesh.faces) > target_faces:
            self.mesh = self.mesh.simplify_quadric_decimation(target_faces)
        return self

    def smooth(self, iterations: int = 3) -> "MeshPostProcessor":
        """ラプラシアンスムージング"""
        trimesh.smoothing.filter_laplacian(
            self.mesh, iterations=iterations
        )
        return self

    def center_and_normalize(self) -> "MeshPostProcessor":
        """原点中心に正規化"""
        # 重心を原点に
        self.mesh.vertices -= self.mesh.centroid
        # バウンディングボックスを[-1,1]に正規化
        scale = 2.0 / max(self.mesh.extents)
        self.mesh.vertices *= scale
        return self

    def export(self, output_path: str, file_type: str = None):
        """メッシュをエクスポート"""
        self.mesh.export(output_path, file_type=file_type)

# 使用例
processor = MeshPostProcessor("generated.obj")
processor.clean().simplify(target_faces=50000).smooth().center_and_normalize()
processor.export("optimized.glb", file_type="glb")
```

---

## 6. アンチパターン

### 6.1 アンチパターン：撮影品質を無視したNeRF/3DGS

```
NG: 適当に撮影した数枚の画像でNeRFを学習
  - ブレた画像、露出バラバラ、カバレッジ不足
  - → 浮遊アーティファクト、穴だらけのモデル

OK: 体系的な撮影プロトコル
  - 均一な照明条件（曇天 or スタジオライト）
  - オーバーラップ70%以上の連続撮影
  - 対象物を360度+上下からカバー
  - 最低50-100枚、高解像度、手ブレなし
```

**問題点**: NeRF/3DGSの品質は入力画像の品質に直結する。「ゴミを入れればゴミが出る」原則がそのまま当てはまる。

### 6.2 アンチパターン：後処理なしでの本番利用

```python
# NG: 生成されたメッシュをそのままゲームエンジンに投入
raw_mesh = generate_3d("a medieval castle")
game_engine.load(raw_mesh)  # 100万ポリゴン、UV未設定

# OK: 適切な後処理パイプラインを通す
raw_mesh = generate_3d("a medieval castle")
processor = MeshPostProcessor(raw_mesh)
optimized = (
    processor.clean()
    .simplify(target_faces=10000)  # LODに応じた削減
    .smooth()
    .center_and_normalize()
)
optimized.export("castle_game_ready.glb")
```

**問題点**: AI生成メッシュはポリゴン数過多、トポロジー不整合、UV未設定が一般的。本番利用には必ず後処理が必要。

---

## 7. FAQ

### Q1: NeRFと3D Gaussian Splattingはどちらを選ぶべきか？

**A**: リアルタイム描画が必要ならば3DGS、最高品質の新規視点合成が目的ならばNeRF（nerfacto等）が適している。3DGSは編集が容易で描画が速いが、メモリ使用量が多い。NeRFはコンパクトだがレンダリングが遅い。2024年以降は3DGSが主流になりつつある。

### Q2: テキストから実用品質の3Dモデルを生成できるか？

**A**: 2025年時点で、商用APIサービス（Meshy、Tripo3D等）を使えばプロトタイピングレベルの3Dモデルが生成可能。ゲームのバックグラウンドアセットやコンセプトモデルとしては十分使える。ただし、主要キャラクターや製品ビジュアライゼーションなど高品質が求められる場面では、生成モデルをベースに手動で調整するワークフローが現実的。

### Q3: 3D生成に必要なGPUスペックは？

**A**: 手法別の推奨GPU。

| 手法 | 最低VRAM | 推奨GPU | 備考 |
|------|---------|---------|------|
| NeRF (nerfacto) | 8GB | RTX 3080以上 | 学習に数時間 |
| 3DGS | 12GB | RTX 4080以上 | シーン規模に依存 |
| Shap-E | 6GB | RTX 3060以上 | 数秒で生成 |
| TripoSR | 8GB | RTX 3070以上 | 数秒で生成 |
| DreamFusion | 16GB+ | RTX 4090/A100 | 数時間の最適化 |

---

## 8. まとめ

| カテゴリ | ポイント |
|---------|---------|
| NeRF | 暗黙的ニューラル表現、高品質だが低速 |
| 3DGS | 明示的ガウシアン表現、リアルタイム描画、編集容易 |
| テキスト→3D | Shap-E/DreamFusionが研究、Meshy/Tripoが商用 |
| 画像→3D | TripoSR/InstantMeshが高速・高品質 |
| 後処理 | メッシュクリーン→簡略化→UV展開→テクスチャベイクが必須 |
| 品質の鍵 | 入力データ品質と後処理パイプラインが最終品質を決定 |
| GPU要件 | 推論は8GB〜、学習は12-16GB〜が目安 |

---

## 次に読むべきガイド

- [01-game-assets.md](./01-game-assets.md) — AIによるゲームアセット生成の実践
- AI画像生成基礎 — 2D画像生成技術の基盤理解
- 3Dレンダリングパイプライン — 従来型3DCG技術との統合

---

## 参考文献

1. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" — https://www.matthewtancik.com/nerf
2. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" — https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
3. OpenAI Shap-E — https://github.com/openai/shap-e
4. nerfstudio ドキュメント — https://docs.nerf.studio/
5. TripoSR — https://github.com/VAST-AI-Research/TripoSR
