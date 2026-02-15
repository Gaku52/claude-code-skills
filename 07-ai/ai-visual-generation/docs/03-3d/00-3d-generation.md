# AI 3Dモデル生成 技術・ツール・活用ガイド

> NeRF、3D Gaussian Splatting、Point-E、Shap-E など最先端のAI 3D生成技術を体系的に解説し、原理の理解からツール活用、本番ワークフローへの統合までを網羅する。

---

## この章で学ぶこと

1. **主要なAI 3D生成手法**（NeRF、3DGS、拡散モデル系）の原理と特徴を理解し、適材適所で使い分けられる
2. **テキスト/画像から3Dモデルを生成**する実践的なワークフローを構築できる
3. **生成した3Dモデルの後処理・最適化・エクスポート**までの本番パイプラインを設計できる
4. **商用APIサービス**（Meshy、Tripo3D等）を効果的に活用できる
5. **ゲームエンジン・Webへの統合**まで含めた実務パイプラインを構築できる

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

### 1.3 3D表現形式の比較

```
┌─────────────────── 3D表現形式の比較 ───────────────────┐
│                                                        │
│  メッシュ (Mesh)                                       │
│  ┌────────────┐  頂点 + 面で構成                       │
│  │  △ △ △     │  ゲームエンジン・DCCツール互換          │
│  │ △ △ △ △   │  テクスチャマッピング容易                │
│  │  △ △ △     │  ファイル形式: OBJ, FBX, glTF          │
│  └────────────┘  LOD制御が容易                         │
│                                                        │
│  点群 (Point Cloud)                                    │
│  ┌────────────┐  3D空間の座標点の集合                   │
│  │ . . . .    │  スキャンデータに直結                   │
│  │  . . . .   │  レンダリングにはスプラッティング必要   │
│  │ . . . .    │  ファイル形式: PLY, PCD, LAS            │
│  └────────────┘  メッシュ変換が必要な場合が多い         │
│                                                        │
│  暗黙的表現 (Implicit)                                 │
│  ┌────────────┐  ニューラルネットワークで表現           │
│  │ f(x,y,z)=σ │  連続的な密度場                        │
│  │ + color    │  Marching Cubesでメッシュ化            │
│  └────────────┘  コンパクトだが推論コスト高い           │
│                                                        │
│  ガウシアン (3DGS)                                     │
│  ┌────────────┐  3Dガウシアン関数の集合                 │
│  │ ○ ○ ○     │  位置+共分散+色+不透明度               │
│  │  ○ ○ ○    │  ラスタライゼーションで高速描画         │
│  │ ○ ○ ○     │  編集が直感的                          │
│  └────────────┘  メモリ使用量は大きい                   │
│                                                        │
│  ボクセル (Voxel)                                      │
│  ┌────────────┐  3Dグリッドの各セルに値を格納          │
│  │ ■ ■ □     │  単純だがメモリ効率が悪い               │
│  │ □ ■ ■     │  解像度に制限あり                       │
│  │ ■ □ ■     │  畳み込み演算に適する                   │
│  └────────────┘  ファイル形式: VDB, numpy array         │
└────────────────────────────────────────────────────────┘
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

### 2.2 NeRFの数学的基礎

```python
"""
NeRF のボリュームレンダリング方程式:

レイ r(t) = o + t*d  (o: 原点, d: 方向)

色の積分:
  C(r) = ∫[t_n, t_f] T(t) * σ(r(t)) * c(r(t), d) dt

透過率:
  T(t) = exp(-∫[t_n, t] σ(r(s)) ds)

離散近似 (実装で使用):
  C(r) ≈ Σ_{i=1}^{N} T_i * (1 - exp(-σ_i * δ_i)) * c_i
  T_i = exp(-Σ_{j=1}^{i-1} σ_j * δ_j)
  δ_i = t_{i+1} - t_i  (サンプル間距離)
"""

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """NeRF の位置エンコーディング"""

    def __init__(self, input_dim, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.input_dim = input_dim
        # 出力次元 = 入力次元 * (2 * num_frequencies + 1)
        self.output_dim = input_dim * (2 * num_frequencies + 1)

    def forward(self, x):
        """
        γ(p) = (p, sin(2^0 π p), cos(2^0 π p),
                   sin(2^1 π p), cos(2^1 π p),
                   ...,
                   sin(2^{L-1} π p), cos(2^{L-1} π p))
        """
        encodings = [x]
        for i in range(self.num_frequencies):
            freq = 2.0 ** i * torch.pi
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))
        return torch.cat(encodings, dim=-1)


class NeRFModel(nn.Module):
    """NeRF のコアネットワーク"""

    def __init__(self, pos_dim=63, dir_dim=27, hidden_dim=256):
        super().__init__()

        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(3, num_frequencies=10)
        self.dir_encoder = PositionalEncoding(3, num_frequencies=4)

        # 密度ネットワーク (位置のみに依存)
        self.density_net = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # スキップ接続後の続き
        self.density_net2 = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # 密度出力
        self.sigma_out = nn.Linear(hidden_dim, 1)

        # 色ネットワーク (位置+方向に依存)
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(),
        )

    def forward(self, positions, directions):
        """
        positions: [N, 3] — 3D座標
        directions: [N, 3] — 視線方向
        """
        pos_enc = self.pos_encoder(positions)  # [N, 63]
        dir_enc = self.dir_encoder(directions)  # [N, 27]

        # 密度の計算
        h = self.density_net(pos_enc)
        h = self.density_net2(torch.cat([h, pos_enc], dim=-1))
        sigma = torch.relu(self.sigma_out(h))

        # 色の計算
        color = self.color_net(torch.cat([h, dir_enc], dim=-1))

        return sigma, color


def volume_rendering(sigmas, colors, deltas):
    """
    ボリュームレンダリング (離散近似)

    sigmas: [N_rays, N_samples] — 密度
    colors: [N_rays, N_samples, 3] — 色
    deltas: [N_rays, N_samples] — サンプル間距離
    """
    # αの計算: α_i = 1 - exp(-σ_i * δ_i)
    alphas = 1.0 - torch.exp(-sigmas * deltas)

    # 透過率の計算: T_i = Π_{j<i} (1 - α_j)
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones_like(alphas[:, :1]),
            1.0 - alphas[:, :-1],
        ], dim=1),
        dim=1,
    )

    # 重みの計算: w_i = T_i * α_i
    weights = transmittance * alphas

    # 色の積算: C = Σ w_i * c_i
    rendered_color = torch.sum(weights.unsqueeze(-1) * colors, dim=1)

    # 深度の推定
    depths = torch.sum(weights * deltas.cumsum(dim=1), dim=1)

    return rendered_color, depths, weights
```

### 2.3 nerfstudioでのNeRF実装

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

### 2.4 撮影プロトコル — 高品質NeRF/3DGS のためのデータ収集

```python
class CaptureProtocol:
    """NeRF/3DGS 用の撮影ガイドライン"""

    GUIDELINES = {
        "camera_settings": {
            "iso": "固定 (400以下推奨)",
            "aperture": "f/8〜f/11 (被写界深度確保)",
            "shutter_speed": "手ブレ防止 (1/250以上)",
            "white_balance": "固定 (変動防止)",
            "format": "RAW推奨 (後処理の自由度)",
            "resolution": "4000x3000以上",
        },
        "shooting_pattern": {
            "minimum_images": 50,
            "recommended_images": "100-200",
            "overlap": "70%以上 (隣接画像間)",
            "coverage": "対象物を360度 + 上下から",
            "orbit_levels": "3段階 (低/中/高アングル)",
            "close_ups": "ディテール部分のクローズアップ",
        },
        "environment": {
            "lighting": "均一な拡散光 (曇天 or スタジオ)",
            "avoid": "強い影、反射、透明物体",
            "background": "動かない静的な背景",
            "turntable": "小物体はターンテーブル推奨",
        },
    }

    @staticmethod
    def generate_camera_positions(
        num_cameras=100,
        radius=2.0,
        num_levels=3,
        elevation_range=(15, 75),
    ):
        """理想的なカメラ位置を計算"""
        import numpy as np

        positions = []
        elevations = np.linspace(
            elevation_range[0], elevation_range[1], num_levels
        )

        cameras_per_level = num_cameras // num_levels

        for elev_deg in elevations:
            elev_rad = np.radians(elev_deg)
            for i in range(cameras_per_level):
                azimuth = 2 * np.pi * i / cameras_per_level
                x = radius * np.cos(elev_rad) * np.cos(azimuth)
                y = radius * np.cos(elev_rad) * np.sin(azimuth)
                z = radius * np.sin(elev_rad)
                positions.append({
                    "position": (x, y, z),
                    "elevation_deg": elev_deg,
                    "azimuth_deg": np.degrees(azimuth),
                    "look_at": (0, 0, 0),
                })

        return positions
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

### 3.2 3DGSの原理

```
3D Gaussian Splatting の各ガウシアン:

1つのガウシアン = {
    位置:     μ ∈ R^3          (3D空間の中心座標)
    共分散:   Σ ∈ R^{3×3}      (形状と向き)
    不透明度: α ∈ [0, 1]       (透明度)
    色:       SH係数 ∈ R^{K}   (球面調和関数で視点依存の色)
}

共分散行列の分解:
  Σ = R * S * S^T * R^T
  R: 回転行列 (quaternionで表現)
  S: スケール行列 (対角, 3軸のサイズ)

レンダリング:
  1. カメラへの投影 (3D → 2D ガウシアン)
  2. ソート (深度順)
  3. α-blending (前→後)

  投影: Σ' = J * W * Σ * W^T * J^T
    J: ヤコビアン (投影微分)
    W: ワールド→カメラ変換
```

### 3.3 3DGSの実装

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

    def compact_model(self, model_dir: str, target_points: int = 500000):
        """ガウシアン数を削減してモデルを軽量化"""
        import numpy as np
        from plyfile import PlyData

        ply_path = self.export_ply(model_dir)
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']

        current_count = len(vertices)
        if current_count <= target_points:
            return ply_path

        # 不透明度でソートし、上位を保持
        opacities = vertices['opacity']
        indices = np.argsort(opacities)[::-1][:target_points]

        # サブセットの作成
        new_vertices = vertices[indices]
        new_plydata = PlyData(
            [PlyElement.describe(new_vertices, 'vertex')],
            text=False,
        )

        output_path = ply_path.replace(".ply", f"_compact_{target_points}.ply")
        new_plydata.write(output_path)

        print(f"圧縮: {current_count} → {target_points} ガウシアン")
        return output_path
```

### 3.4 Web ビューアでの 3DGS 表示

```python
"""
3D Gaussian Splatting をWebブラウザで表示する方法

主要なWebビューア:
1. gsplat.js — WebGL ベース
2. splat — Three.js ベース
3. Luma AI WebGL Viewer
"""

def create_web_viewer(ply_path: str, output_dir: str = "viewer"):
    """3DGSモデルのWebビューアを生成"""
    from pathlib import Path
    import shutil

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # HTMLテンプレート
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>3D Gaussian Splatting Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        canvas {{ width: 100vw; height: 100vh; display: block; }}
        #info {{
            position: absolute; top: 10px; left: 10px;
            color: white; font-family: monospace;
            background: rgba(0,0,0,0.5); padding: 10px;
        }}
    </style>
</head>
<body>
    <div id="info">
        Controls: Left-click drag to orbit,
        Right-click drag to pan, Scroll to zoom
    </div>
    <canvas id="canvas"></canvas>
    <script type="module">
        import {{ Viewer }} from './gsplat.js';
        const viewer = new Viewer({{
            canvas: document.getElementById('canvas'),
            url: 'model.splat',
        }});
    </script>
</body>
</html>"""

    with open(out / "index.html", "w") as f:
        f.write(html_content)

    # PLYをsplat形式に変換
    convert_ply_to_splat(ply_path, str(out / "model.splat"))

    print(f"Webビューア生成: {out / 'index.html'}")
    print("ローカルサーバー起動: python -m http.server 8080")
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

### 4.4 商用API (Meshy / Tripo3D) の活用

```python
import requests
import time
from pathlib import Path

class Meshy3DClient:
    """Meshy API による3Dモデル生成"""

    BASE_URL = "https://api.meshy.ai/v2"

    def __init__(self, api_key: str):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def text_to_3d(self, prompt: str, art_style: str = "realistic",
                   negative_prompt: str = "", topology: str = "quad"):
        """テキストから3Dモデルを生成"""
        # Step 1: プレビュー生成
        resp = requests.post(
            f"{self.BASE_URL}/text-to-3d",
            headers=self.headers,
            json={
                "mode": "preview",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "art_style": art_style,
                "topology": topology,
            },
        )
        task_id = resp.json()["result"]

        # ポーリング
        result = self._wait_for_completion(task_id)

        # Step 2: リファイン生成 (高品質化)
        resp = requests.post(
            f"{self.BASE_URL}/text-to-3d",
            headers=self.headers,
            json={
                "mode": "refine",
                "preview_task_id": task_id,
                "texture_richness": "high",
            },
        )
        refine_id = resp.json()["result"]

        return self._wait_for_completion(refine_id)

    def image_to_3d(self, image_url: str):
        """画像から3Dモデルを生成"""
        resp = requests.post(
            f"{self.BASE_URL}/image-to-3d",
            headers=self.headers,
            json={
                "image_url": image_url,
                "topology": "quad",
                "target_polycount": 30000,
            },
        )
        task_id = resp.json()["result"]
        return self._wait_for_completion(task_id)

    def _wait_for_completion(self, task_id, timeout=600):
        """タスク完了を待機"""
        for _ in range(timeout // 5):
            resp = requests.get(
                f"{self.BASE_URL}/text-to-3d/{task_id}",
                headers=self.headers,
            )
            data = resp.json()
            if data["status"] == "SUCCEEDED":
                return {
                    "model_urls": data["model_urls"],
                    "thumbnail_url": data.get("thumbnail_url"),
                    "task_id": task_id,
                }
            elif data["status"] == "FAILED":
                raise Exception(f"生成失敗: {data}")
            time.sleep(5)
        raise TimeoutError("タイムアウト")

    def download_model(self, model_urls: dict, output_dir: str):
        """生成モデルをダウンロード"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for fmt, url in model_urls.items():
            resp = requests.get(url)
            file_path = out / f"model.{fmt}"
            with open(file_path, "wb") as f:
                f.write(resp.content)
            print(f"ダウンロード: {file_path}")


class Tripo3DClient:
    """Tripo3D API による3Dモデル生成"""

    BASE_URL = "https://api.tripo3d.ai/v2/openapi"

    def __init__(self, api_key: str):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def text_to_model(self, prompt: str):
        """テキストから3Dモデルを生成"""
        resp = requests.post(
            f"{self.BASE_URL}/task",
            headers=self.headers,
            json={
                "type": "text_to_model",
                "prompt": prompt,
            },
        )
        task_id = resp.json()["data"]["task_id"]
        return self._wait_for_result(task_id)

    def image_to_model(self, image_token: str):
        """画像から3Dモデルを生成"""
        resp = requests.post(
            f"{self.BASE_URL}/task",
            headers=self.headers,
            json={
                "type": "image_to_model",
                "file": {"type": "jpg", "file_token": image_token},
            },
        )
        task_id = resp.json()["data"]["task_id"]
        return self._wait_for_result(task_id)

    def _wait_for_result(self, task_id, timeout=300):
        """結果を待機"""
        for _ in range(timeout // 3):
            resp = requests.get(
                f"{self.BASE_URL}/task/{task_id}",
                headers=self.headers,
            )
            data = resp.json()["data"]
            if data["status"] == "success":
                return data["output"]
            elif data["status"] == "failed":
                raise Exception(f"失敗: {data}")
            time.sleep(3)
        raise TimeoutError("タイムアウト")
```

### 4.5 DreamFusion — SDS Loss によるテキストから3D

```python
"""
DreamFusion: Score Distillation Sampling (SDS) を用いたテキストから3D生成

原理:
1. ランダムな3D表現 (NeRF) を初期化
2. ランダムな視点からレンダリング
3. 事前学習済みの画像拡散モデル (Imagen / SD) で
   「このレンダリング結果がプロンプトに合っているか」を評価
4. SDS Loss を計算し、3D表現を更新
5. 2-4 を繰り返す (数千イテレーション)

SDS Loss:
  ∇_θ L_SDS = E_{t,ε} [w(t) * (ε_φ(z_t; y, t) - ε) * ∂z/∂θ]

  θ: NeRFのパラメータ
  z: レンダリング結果のVAEエンコーディング
  ε_φ: 拡散モデルの予測ノイズ
  y: テキストプロンプト
  t: ノイズレベル
"""

class DreamFusionConcept:
    """DreamFusion の概念的な実装"""

    def __init__(self, prompt, diffusion_model="stabilityai/stable-diffusion-2-1"):
        self.prompt = prompt
        self.nerf = self._init_nerf()
        self.diffusion = self._load_diffusion(diffusion_model)

    def train_step(self, iteration):
        """1イテレーションの学習"""
        # 1. ランダムなカメラ位置をサンプリング
        camera = self._random_camera()

        # 2. 現在のNeRFからレンダリング
        rendered_image = self.nerf.render(camera)

        # 3. SDS Loss の計算
        # (拡散モデルが「この画像がプロンプトに合うか」を評価)
        t = self._sample_timestep()
        noise = torch.randn_like(rendered_image)
        noisy_image = self._add_noise(rendered_image, noise, t)

        with torch.no_grad():
            predicted_noise = self.diffusion(noisy_image, t, self.prompt)

        # SDS勾配
        gradient = predicted_noise - noise

        # 4. NeRFパラメータの更新
        self.nerf.backward(gradient)
        self.optimizer.step()

    def optimize(self, num_iterations=10000):
        """最適化ループ"""
        for i in range(num_iterations):
            self.train_step(i)
            if i % 1000 == 0:
                print(f"Iteration {i}/{num_iterations}")
                self._save_checkpoint(i)

        # 5. メッシュ抽出
        mesh = self.nerf.extract_mesh(resolution=256)
        return mesh
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

    def fill_holes(self) -> "MeshPostProcessor":
        """穴の修復"""
        trimesh.repair.fill_holes(self.mesh)
        return self

    def remesh(self, target_edge_length: float = 0.02) -> "MeshPostProcessor":
        """リメッシュ (均一なポリゴン分布に)"""
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                self.mesh.vertices, self.mesh.faces
            ))
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.AbsoluteValue(target_edge_length)
            )
            result = ms.current_mesh()
            self.mesh = trimesh.Trimesh(
                vertices=result.vertex_matrix(),
                faces=result.face_matrix(),
            )
        except ImportError:
            print("pymeshlab が必要です: pip install pymeshlab")
        return self

    def generate_lod(self, levels=(50000, 10000, 2000, 500)):
        """LOD (Level of Detail) メッシュを生成"""
        lod_meshes = {}
        for level in levels:
            mesh_copy = self.mesh.copy()
            if len(mesh_copy.faces) > level:
                mesh_copy = mesh_copy.simplify_quadric_decimation(level)
            lod_meshes[f"LOD_{level}"] = mesh_copy
        return lod_meshes

    def get_stats(self) -> dict:
        """メッシュの統計情報"""
        return {
            "vertices": len(self.mesh.vertices),
            "faces": len(self.mesh.faces),
            "edges": len(self.mesh.edges),
            "watertight": self.mesh.is_watertight,
            "volume": self.mesh.volume if self.mesh.is_watertight else None,
            "bounds": self.mesh.bounds.tolist(),
            "extents": self.mesh.extents.tolist(),
            "centroid": self.mesh.centroid.tolist(),
        }

    def export(self, output_path: str, file_type: str = None):
        """メッシュをエクスポート"""
        self.mesh.export(output_path, file_type=file_type)

# 使用例
processor = MeshPostProcessor("generated.obj")
processor.clean().simplify(target_faces=50000).smooth().center_and_normalize()
processor.export("optimized.glb", file_type="glb")
print(processor.get_stats())
```

### 5.3 テクスチャベイキングとUV展開

```python
def auto_uv_and_bake(mesh_path: str, output_dir: str):
    """
    UV展開とテクスチャベイキングの自動化

    Blender Python API を使用
    """
    import subprocess

    blender_script = '''
import bpy
import sys

# 引数の取得
mesh_path = sys.argv[-2]
output_dir = sys.argv[-1]

# メッシュのインポート
bpy.ops.import_scene.obj(filepath=mesh_path)
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# UV展開 (Smart UV Project)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(
    angle_limit=66.0,
    island_margin=0.02,
    area_weight=0.0,
)
bpy.ops.object.mode_set(mode='OBJECT')

# テクスチャベイキング用のイメージ作成
bpy.ops.image.new(
    name='BakedTexture',
    width=2048,
    height=2048,
    color=(0, 0, 0, 1),
)

# マテリアルとテクスチャノードの設定
mat = obj.data.materials[0] if obj.data.materials else bpy.data.materials.new("Material")
if not obj.data.materials:
    obj.data.materials.append(mat)

mat.use_nodes = True
nodes = mat.node_tree.nodes
tex_node = nodes.new('ShaderNodeTexImage')
tex_node.image = bpy.data.images['BakedTexture']
tex_node.select = True
nodes.active = tex_node

# ベイキング実行
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.bake_type = 'DIFFUSE'
bpy.ops.object.bake(type='DIFFUSE')

# テクスチャの保存
bpy.data.images['BakedTexture'].save_render(
    filepath=output_dir + '/texture_diffuse.png'
)

# glTF形式でエクスポート
bpy.ops.export_scene.gltf(
    filepath=output_dir + '/model.glb',
    export_format='GLB',
    export_texcoords=True,
    export_normals=True,
    export_materials='EXPORT',
)
'''

    # Blenderをヘッドレスモードで実行
    script_path = Path(output_dir) / "bake_script.py"
    with open(script_path, "w") as f:
        f.write(blender_script)

    cmd = [
        "blender", "--background", "--python", str(script_path),
        "--", mesh_path, output_dir,
    ]
    subprocess.run(cmd, check=True)
```

---

## 6. ゲームエンジン・Web統合

### 6.1 Unity への3Dモデル統合

```csharp
// Unity での AI生成3Dモデル読み込み (概念コード)
using UnityEngine;
using System.Threading.Tasks;

public class AI3DModelLoader : MonoBehaviour
{
    [Header("API Settings")]
    public string apiEndpoint = "https://api.meshy.ai/v2";
    public string apiKey;

    [Header("Generation Settings")]
    public string prompt = "a medieval sword";
    public string artStyle = "realistic";

    public async Task<GameObject> GenerateAndLoad(string prompt)
    {
        // 1. API で3Dモデルを生成
        string modelUrl = await RequestGeneration(prompt);

        // 2. glTFファイルをダウンロード
        byte[] modelData = await DownloadModel(modelUrl);

        // 3. Unityにインポート (GLTFUtility等を使用)
        GameObject model = GLTFUtility.ImportGLB(modelData);

        // 4. LOD設定
        SetupLOD(model);

        // 5. コライダー追加
        model.AddComponent<MeshCollider>();

        return model;
    }

    private void SetupLOD(GameObject model)
    {
        LODGroup lodGroup = model.AddComponent<LODGroup>();
        // AI生成モデルは通常ハイポリなので
        // ランタイムでLODを設定
    }
}
```

### 6.2 Three.js での Web 表示

```javascript
// Three.js での AI生成3Dモデル表示
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

class AI3DViewer {
    constructor(container) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75, container.clientWidth / container.clientHeight, 0.1, 1000
        );
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        container.appendChild(this.renderer.domElement);

        // コントロール
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        // ライティング
        this.setupLighting();

        // カメラ位置
        this.camera.position.set(2, 2, 2);
        this.camera.lookAt(0, 0, 0);

        this.animate();
    }

    setupLighting() {
        // 環境光
        const ambient = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambient);

        // ディレクショナルライト
        const directional = new THREE.DirectionalLight(0xffffff, 1.0);
        directional.position.set(5, 10, 5);
        directional.castShadow = true;
        this.scene.add(directional);

        // 環境マップ (IBL)
        const pmremGenerator = new THREE.PMREMGenerator(this.renderer);
        // HDR環境マップのロードと設定
    }

    async loadModel(url) {
        const loader = new GLTFLoader();
        return new Promise((resolve, reject) => {
            loader.load(url, (gltf) => {
                const model = gltf.scene;

                // 正規化 (バウンディングボックスを統一)
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 2.0 / maxDim;

                model.position.sub(center);
                model.scale.multiplyScalar(scale);

                this.scene.add(model);
                resolve(model);
            }, undefined, reject);
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
```

---

## 7. アンチパターン

### 7.1 アンチパターン：撮影品質を無視したNeRF/3DGS

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

### 7.2 アンチパターン：後処理なしでの本番利用

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

### 7.3 アンチパターン：単一手法への依存

```
NG: 全てのユースケースで Shap-E を使う
  - プロトタイプには良いが品質が限定的
  - テクスチャの質が不十分
  - 複雑な形状の再現が困難

OK: ユースケースに応じた手法選択
  - コンセプト検討: Shap-E / Point-E (高速、無料)
  - プロトタイプ: Meshy / Tripo API (テクスチャ付き)
  - 本番アセット: 商用API + 手動リタッチ
  - リアルスキャン: 3DGS / NeRF (実物ベース)
```

### 7.4 アンチパターン：メモリ制限を無視した3DGS学習

```
NG: 高解像度画像200枚 + SH degree=3 で一気に学習
  - VRAM不足で学習が停止
  - ガウシアン数が爆発的に増加

OK: リソースに応じた設定
  - 画像解像度を1600x1200程度にリサイズ
  - densify_until を制限 (15000-20000)
  - sh_degree を 2 に下げる (品質は若干低下)
  - 定期的にプルーニング
```

---

## 8. FAQ

### Q1: NeRFと3D Gaussian Splattingはどちらを選ぶべきか?

**A**: リアルタイム描画が必要ならば3DGS、最高品質の新規視点合成が目的ならばNeRF（nerfacto等）が適している。3DGSは編集が容易で描画が速いが、メモリ使用量が多い。NeRFはコンパクトだがレンダリングが遅い。2024年以降は3DGSが主流になりつつある。

### Q2: テキストから実用品質の3Dモデルを生成できるか?

**A**: 2025年時点で、商用APIサービス（Meshy、Tripo3D等）を使えばプロトタイピングレベルの3Dモデルが生成可能。ゲームのバックグラウンドアセットやコンセプトモデルとしては十分使える。ただし、主要キャラクターや製品ビジュアライゼーションなど高品質が求められる場面では、生成モデルをベースに手動で調整するワークフローが現実的。

### Q3: 3D生成に必要なGPUスペックは?

**A**: 手法別の推奨GPU。

| 手法 | 最低VRAM | 推奨GPU | 備考 |
|------|---------|---------|------|
| NeRF (nerfacto) | 8GB | RTX 3080以上 | 学習に数時間 |
| 3DGS | 12GB | RTX 4080以上 | シーン規模に依存 |
| Shap-E | 6GB | RTX 3060以上 | 数秒で生成 |
| TripoSR | 8GB | RTX 3070以上 | 数秒で生成 |
| DreamFusion | 16GB+ | RTX 4090/A100 | 数時間の最適化 |
| InstantMesh | 8GB | RTX 3070以上 | 数十秒で生成 |
| Meshy API | 0 (クラウド) | 不要 | API呼び出しのみ |

### Q4: 3DGSモデルをメッシュに変換するには?

**A**: 3DGSのガウシアン点群からメッシュを生成するには、以下のアプローチがあります:

1. **PoissontReconstructon**: 点群から表面を再構成（Open3Dで実装可能）
2. **SuGaR**: 3DGSに正則化を追加してメッシュ抽出を改善する手法
3. **2DGS**: 2Dガウシアンを使って表面をより正確に表現

```python
import open3d as o3d

def gaussians_to_mesh(ply_path, output_path, depth=9):
    """3DGS点群からPoisson Surface Reconstructionでメッシュ化"""
    pcd = o3d.io.read_point_cloud(ply_path)

    # 法線推定
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson Surface Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False
    )

    # 低密度領域の除去
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    o3d.io.write_triangle_mesh(output_path, mesh)
    return output_path
```

### Q5: 生成した3Dモデルの著作権は?

**A:**

| ツール/API | ライセンス | 商用利用 | 注意点 |
|-----------|----------|---------|--------|
| **Shap-E** | MIT | 可能 | モデルの品質は限定的 |
| **TripoSR** | MIT | 可能 | Stability AI が公開 |
| **InstantMesh** | Apache 2.0 | 可能 | TencentARC が公開 |
| **Meshy** | 利用規約 | 有料プランで可 | 生成物は利用者に帰属 |
| **Tripo3D** | 利用規約 | 有料プランで可 | 利用規約を要確認 |
| **DreamFusion** | 研究用 | 非商用 | Google の研究論文 |

---

## 9. まとめ

| カテゴリ | ポイント |
|---------|---------|
| NeRF | 暗黙的ニューラル表現、高品質だが低速 |
| 3DGS | 明示的ガウシアン表現、リアルタイム描画、編集容易 |
| テキスト→3D | Shap-E/DreamFusionが研究、Meshy/Tripoが商用 |
| 画像→3D | TripoSR/InstantMeshが高速・高品質 |
| 商用API | Meshy/Tripo3D がテクスチャ付き高品質モデルを提供 |
| 後処理 | メッシュクリーン→簡略化→UV展開→テクスチャベイクが必須 |
| ゲーム統合 | glTF形式、LOD設定、コライダー設定が重要 |
| Web統合 | Three.js + GLTFLoader、3DGSはgsplat.js |
| 品質の鍵 | 入力データ品質と後処理パイプラインが最終品質を決定 |
| GPU要件 | 推論は8GB~、学習は12-16GB~が目安 |

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
6. Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion" — https://dreamfusion3d.github.io/
7. Xu et al., "InstantMesh: Efficient 3D Mesh Generation from a Single Image" — https://arxiv.org/abs/2404.07191
8. Guedon & Lepetit, "SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction" — https://arxiv.org/abs/2311.12775
