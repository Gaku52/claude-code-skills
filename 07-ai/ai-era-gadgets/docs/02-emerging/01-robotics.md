# ロボティクスガイド

> Boston Dynamics、Figure、家庭用ロボットなどAI時代のロボット技術を包括的に解説する

## この章で学ぶこと

1. **ロボティクスの基礎** — センサー、アクチュエータ、制御系の構成と役割
2. **主要ロボット企業** — Boston Dynamics、Figure、Tesla Optimus、家庭用ロボットの技術と戦略
3. **AIとロボットの融合** — 基盤モデル（Foundation Model）によるロボット制御の革新
4. **シミュレーションと転移** — Isaac Sim、MuJoCo、Sim-to-Realの実践手法
5. **安全設計** — ロボット安全規格、多層防御、人間-ロボット協調

---

## 1. ロボティクスの基本構成

### ロボットシステムの構造

```
+-----------------------------------------------------------+
|                 ロボットシステム全体像                       |
+-----------------------------------------------------------+
|                                                           |
|  +------------------+     +------------------+            |
|  | 知覚 (Perception)|     | 計画 (Planning)  |            |
|  | カメラ、LiDAR    | --> | 経路計画          |            |
|  | 触覚センサー     |     | タスク計画        |            |
|  | IMU、力覚        |     | モーション計画    |            |
|  +------------------+     +------------------+            |
|                                  |                        |
|                                  v                        |
|  +------------------+     +------------------+            |
|  | 学習 (Learning)  |     | 制御 (Control)   |            |
|  | 強化学習         | <-> | PID制御           |            |
|  | 模倣学習         |     | MPC               |            |
|  | 基盤モデル       |     | トルク制御        |            |
|  +------------------+     +------------------+            |
|                                  |                        |
|                                  v                        |
|                          +------------------+             |
|                          | 行動 (Action)    |             |
|                          | モーター、油圧    |             |
|                          | グリッパー       |             |
|                          +------------------+             |
+-----------------------------------------------------------+
```

### センサーの種類と用途

```
+-----------------------------------------------------------+
|  ロボットセンサー体系                                       |
+-----------------------------------------------------------+
|                                                           |
|  視覚系                                                    |
|  +-- RGB カメラ: 色・形状認識                              |
|  +-- 深度カメラ (ToF/構造化光): 3D空間認識                 |
|  +-- ステレオカメラ: 立体視による距離推定                   |
|  +-- LiDAR: 高精度3Dマッピング                             |
|  +-- イベントカメラ: 超高速変化検出（ダイナミックビジョン） |
|                                                           |
|  力覚系                                                    |
|  +-- 力覚/トルクセンサー: 接触力の検出                     |
|  +-- 触覚センサー: 表面テクスチャ・滑り検出                |
|  +-- 圧力センサー: 把持力の制御                            |
|  +-- 電子皮膚: 全身分布型触覚（柔軟な表面）               |
|                                                           |
|  慣性系                                                    |
|  +-- IMU (加速度+ジャイロ): 姿勢・動き検出                 |
|  +-- エンコーダ: 関節角度の精密計測                         |
|  +-- 磁気エンコーダ: 非接触角度検出                        |
|                                                           |
|  環境系                                                    |
|  +-- 超音波: 近距離障害物検出                              |
|  +-- 赤外線: 熱源検出、人感センサー                        |
|  +-- マイクロフォンアレイ: 音源方向定位                     |
+-----------------------------------------------------------+
```

### アクチュエータの種類と特性

```
+-----------------------------------------------------------+
|  ロボット用アクチュエータの比較                              |
+-----------------------------------------------------------+
|                                                           |
|  電動モーター（DC/BLDC）                                   |
|  +-- 精密制御が容易、応答性が高い                          |
|  +-- 効率: 80-95%                                          |
|  +-- 用途: ロボットアーム、ヒューマノイド関節               |
|  +-- Atlas(電動), Figure 02, Optimus が採用                |
|                                                           |
|  油圧アクチュエータ                                        |
|  +-- 高出力、重量物の操作                                  |
|  +-- 効率: 40-60%                                          |
|  +-- 用途: 建設機械、旧Atlas（油圧版）                     |
|  +-- 油漏れ、メンテナンス性が課題                          |
|                                                           |
|  空圧アクチュエータ                                        |
|  +-- 軽量、安全（低出力）                                  |
|  +-- 効率: 20-30%                                          |
|  +-- 用途: ソフトロボティクス、グリッパー                  |
|                                                           |
|  人工筋肉（SMA/EAP）                                       |
|  +-- 軽量、柔軟、生体に近い動き                            |
|  +-- 効率: 1-10%（現状）                                   |
|  +-- 用途: 研究段階、ソフトロボティクス                    |
|                                                           |
|  準直動アクチュエータ（QDD）                               |
|  +-- 低減速比で高バックドライバビリティ                     |
|  +-- 衝突時に力を逃がせる（安全性↑）                       |
|  +-- Unitree H1/G1、MIT Cheetah が採用                    |
+-----------------------------------------------------------+
```

---

## 2. 主要ロボット企業と製品

### 企業・製品比較表

| 企業 | 代表製品 | カテゴリ | 自由度 | AIアプローチ | 状況(2025) |
|------|---------|---------|--------|------------|-----------|
| Boston Dynamics | Atlas (電動) | ヒューマノイド | 28+ | 強化学習+MPC | 研究・商用デモ |
| Boston Dynamics | Spot | 四足歩行 | 17 | 自律ナビゲーション | 商用展開中 |
| Figure | Figure 02 | ヒューマノイド | 40+ | OpenAIモデル統合 | プロトタイプ |
| Tesla | Optimus Gen 2 | ヒューマノイド | 28 | FSD技術転用 | 開発中 |
| Unitree | H1/G1 | ヒューマノイド | 23-40 | 強化学習 | 商用開始 |
| Agility Robotics | Digit | ヒューマノイド | 16+ | 倉庫作業特化 | Amazon試験導入 |
| 1X Technologies | NEO Beta | ヒューマノイド | 25+ | OpenAI支援 | プロトタイプ |
| Apptronik | Apollo | ヒューマノイド | 30+ | Mercedes-Benz連携 | 工場テスト |
| iRobot | Roomba j9+ | 家庭用掃除 | - | 物体認識AI | 一般販売中 |
| Sony | aibo (ERS-1000) | ペットロボット | 22 | 感情AI | 一般販売中 |

### ヒューマノイドロボットの世代進化

```
+-----------------------------------------------------------+
|  ヒューマノイドロボットの進化                                |
+-----------------------------------------------------------+
|                                                           |
|  第1世代 (2000-2015): ASIMO, HRP                          |
|  |██|                                                     |
|  ZMP歩行、事前プログラム動作、限定環境                      |
|                                                           |
|  第2世代 (2015-2022): Atlas (油圧), Pepper                 |
|  |██████|                                                 |
|  ダイナミック歩行、バク転、基本的な自律性                    |
|                                                           |
|  第3世代 (2022-2025): Atlas (電動), Figure, Optimus        |
|  |████████████|                                           |
|  電動アクチュエータ、AIビジョン、タスク学習                  |
|                                                           |
|  第4世代 (2025-): 基盤モデル統合型                          |
|  |████████████████████|                                   |
|  言語指示で動作、汎用タスク実行、自己学習                    |
+-----------------------------------------------------------+
```

### 各社のAI戦略比較

```
+-----------------------------------------------------------+
|  ヒューマノイドロボット AI戦略比較                           |
+-----------------------------------------------------------+
|                                                           |
|  Boston Dynamics (Atlas)                                   |
|  +-- 制御: MPC + 強化学習のハイブリッド                    |
|  +-- 知覚: 独自ビジョンパイプライン                        |
|  +-- 学習: シミュレーション強化学習 → 実機転移             |
|  +-- 強み: 運動性能、堅牢性                                |
|                                                           |
|  Figure (Figure 02)                                        |
|  +-- 制御: 基盤モデル（OpenAI VLM）で高レベル計画          |
|  +-- 知覚: カメラ + 言語理解                               |
|  +-- 学習: テレオペ + 模倣学習 + 強化学習                  |
|  +-- 強み: 自然言語指示での汎用タスク                      |
|                                                           |
|  Tesla (Optimus)                                           |
|  +-- 制御: FSD (Full Self-Driving) の技術転用              |
|  +-- 知覚: カメラのみ（LiDARなし、FSDと同じ哲学）         |
|  +-- 学習: 大規模データ + ニューラルネット                 |
|  +-- 強み: スケーラビリティ、コスト削減                    |
|                                                           |
|  Unitree (H1/G1)                                           |
|  +-- 制御: 強化学習（Isaac Gym で学習）                    |
|  +-- 知覚: LiDAR + カメラ + IMU                            |
|  +-- 学習: Sim-to-Real 転移                                |
|  +-- 強み: 低コスト（$90,000〜）、俊敏な動作               |
+-----------------------------------------------------------+
```

---

## 3. AI × ロボティクスの融合

### コード例1: ROS 2 でのロボット制御基本

```python
# ROS 2 (Robot Operating System) によるロボットノードの基本
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        # LiDARデータの購読
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # 速度指令の発行
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.min_distance = 0.5  # 最小安全距離 (メートル)
        self.get_logger().info('障害物回避ノード起動')

    def scan_callback(self, msg: LaserScan):
        """LiDARスキャンデータから障害物を検出し回避"""
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)

        # 前方180度のスキャンデータ
        front_ranges = ranges[len(ranges)//4 : 3*len(ranges)//4]

        cmd = Twist()

        if np.min(front_ranges) < self.min_distance:
            # 障害物検出 → 回転
            cmd.angular.z = 0.5  # 左旋回
            cmd.linear.x = 0.0
            self.get_logger().warn(
                f'障害物検出: {np.min(front_ranges):.2f}m — 回避中'
            )
        else:
            # 安全 → 前進
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = ObstacleAvoidanceNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

### コード例2: 模倣学習（Imitation Learning）

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class RobotPolicy(nn.Module):
    """画像観測から行動を予測する方策ネットワーク"""
    def __init__(self, action_dim=7):
        super().__init__()
        # 視覚エンコーダ
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        # 行動予測ヘッド
        self.policy = nn.Sequential(
            nn.Linear(64 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),  # [dx, dy, dz, droll, dpitch, dyaw, gripper]
        )

    def forward(self, image):
        features = self.vision(image)
        action = self.policy(features)
        return action

# 人間のデモンストレーションデータから学習
class DemonstrationDataset(torch.utils.data.Dataset):
    def __init__(self, demo_path):
        self.demos = load_demonstrations(demo_path)
        # demo: [(image, action), (image, action), ...]

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx):
        image, action = self.demos[idx]
        return torch.tensor(image).float(), torch.tensor(action).float()

# 学習ループ
policy = RobotPolicy().to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

for epoch in range(100):
    for images, expert_actions in dataloader:
        predicted_actions = policy(images.to(device))
        loss = nn.MSELoss()(predicted_actions, expert_actions.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### コード例3: Diffusion Policy（拡散モデルベースの行動生成）

```python
import torch
import torch.nn as nn

class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy: 拡散モデルを使ったロボット行動生成
    ノイズから行動軌道を段階的にデノイズして生成する

    利点:
    - マルチモーダルな行動分布を表現可能
    - 複雑な操作タスクで高い成功率
    - 2024-2025年のロボティクスで最も注目される手法
    """
    def __init__(self, obs_dim=512, action_dim=7, action_horizon=16,
                 n_diffusion_steps=100):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_steps = n_diffusion_steps

        # 観測エンコーダ（画像→特徴量）
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, obs_dim),
        )

        # ノイズ予測ネットワーク（1D U-Net的構造）
        self.noise_pred_net = nn.Sequential(
            nn.Linear(action_dim * action_horizon + obs_dim + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * action_horizon),
        )

    def forward(self, obs, noisy_action, timestep):
        """ノイズ予測: 現在の観測と汚染された行動からノイズを推定"""
        obs_feat = self.obs_encoder(obs)
        noisy_flat = noisy_action.flatten(start_dim=1)
        t_embed = timestep.float().unsqueeze(1) / self.n_steps

        x = torch.cat([noisy_flat, obs_feat, t_embed], dim=1)
        noise_pred = self.noise_pred_net(x)
        return noise_pred.view(-1, self.action_horizon, self.action_dim)

    @torch.no_grad()
    def generate_action(self, obs):
        """推論: ノイズから行動軌道を段階的にデノイズ"""
        batch_size = obs.shape[0]
        device = obs.device

        # ランダムノイズから開始
        action = torch.randn(
            batch_size, self.action_horizon, self.action_dim, device=device
        )

        # DDPM デノイジングプロセス
        for t in reversed(range(self.n_steps)):
            timestep = torch.full((batch_size,), t, device=device)
            noise_pred = self.forward(obs, action, timestep)

            # デノイジングステップ（簡略化）
            alpha = 1 - 0.02 * t / self.n_steps
            action = (action - (1 - alpha) * noise_pred) / alpha.sqrt()

            if t > 0:
                action += 0.1 * torch.randn_like(action)

        return action  # (batch, horizon, action_dim)

# 学習
policy = DiffusionPolicy().to(device)
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

for epoch in range(100):
    for obs, expert_actions in dataloader:
        # ランダムタイムステップでノイズ追加
        t = torch.randint(0, policy.n_steps, (obs.shape[0],), device=device)
        noise = torch.randn_like(expert_actions)
        noisy_actions = expert_actions + noise * (t.float() / policy.n_steps).unsqueeze(1).unsqueeze(2)

        # ノイズ予測の学習
        noise_pred = policy(obs.to(device), noisy_actions.to(device), t)
        loss = nn.MSELoss()(noise_pred, noise.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### コード例4: 言語指示によるロボット制御（基盤モデル統合）

```python
# VLM（Vision-Language Model）によるロボット制御の概念コード
class VLMRobotController:
    """
    言語指示を理解し、カメラ画像から状況判断して
    ロボットの行動を生成するコントローラ
    """
    def __init__(self):
        self.vlm = load_vlm("rt-2-x")  # Robotics Transformer
        self.low_level_controller = MotorController()

    def execute_instruction(self, instruction: str, camera_image):
        """
        例: instruction = "テーブルの上の赤いカップを取って、棚に置いて"
        """
        # VLMが画像と指示を理解し、行動トークンを生成
        action_tokens = self.vlm.predict(
            image=camera_image,
            instruction=instruction,
        )
        # action_tokens → [dx, dy, dz, rx, ry, rz, gripper_open]

        # 低レベル制御器でモーター指令に変換
        for action in action_tokens:
            joint_torques = self.low_level_controller.inverse_kinematics(action)
            self.low_level_controller.execute(joint_torques)

        return action_tokens

# Google RT-2-X のアプローチ:
# 1. 大規模言語モデル（PaLM-E）で言語理解
# 2. ビジョンエンコーダで環境認識
# 3. 行動トークンとして離散化した動作を生成
# 4. デモリファイニング(学習済み)モデルで実行
```

---

## 4. ロボット制御手法の比較

### 制御手法の比較表

| 手法 | 適用場面 | 汎用性 | 安全性 | 学習コスト | 代表例 |
|------|---------|--------|--------|-----------|--------|
| PID制御 | 関節角度制御 | 低 | 高 | なし | 産業用ロボットアーム |
| MPC（モデル予測制御） | 歩行・移動 | 中 | 高 | 低 | Boston Dynamics Atlas |
| 強化学習（RL） | 複雑な動作獲得 | 高 | 中 | 高（sim-to-real） | 歩行、マニピュレーション |
| 模倣学習（IL） | タスク固有動作 | 中 | 中 | 中（デモ収集） | 組立作業、調理 |
| Diffusion Policy | 精密操作 | 高 | 中 | 中 | 折り畳み、組立 |
| 基盤モデル（FM） | 汎用タスク | 非常に高 | 開発中 | 高（大規模学習） | RT-2, Figure + OpenAI |

### コード例5: 強化学習によるロボット歩行（Isaac Gym）

```python
# NVIDIA Isaac Gym を使った四足歩行ロボットの強化学習
import isaacgym
from isaacgym import gymapi, gymutil
import torch

class QuadrupedEnv:
    """四足歩行ロボットの並列シミュレーション環境"""

    def __init__(self, num_envs=4096):
        self.num_envs = num_envs
        self.gym = gymapi.acquire_gym()

        # シミュレーション設定
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 200.0  # 200Hz
        sim_params.substeps = 2
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.use_gpu = True

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        # 4096個の環境を並列作成（GPU上で同時シミュレーション）
        self._create_envs()

    def _create_envs(self):
        """並列環境の作成"""
        asset_root = "/path/to/urdf/"
        asset_file = "a1_robot.urdf"

        # ロボットモデルのロード
        asset = self.gym.load_asset(self.sim, asset_root, asset_file)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, ...)
            self.gym.create_actor(env, asset, ...)

    def step(self, actions):
        """1ステップ実行（全環境を同時に進める）"""
        # actions: (num_envs, 12) — 各脚3関節 × 4脚
        self.gym.set_dof_position_target_tensor(self.sim, actions)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        obs = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._check_termination()

        return obs, rewards, dones

    def _compute_rewards(self):
        """報酬関数"""
        # 前進速度への報酬
        forward_reward = self.base_velocity[:, 0] * 2.0

        # エネルギーペナルティ（省エネ歩行を促進）
        energy_penalty = -0.005 * torch.sum(self.torques ** 2, dim=1)

        # 姿勢安定性（転倒ペナルティ）
        orientation_penalty = -1.0 * torch.sum(
            (self.base_orientation[:, :2]) ** 2, dim=1
        )

        # 足の接地パターン（対角歩行を促進）
        gait_reward = self._compute_gait_reward()

        return forward_reward + energy_penalty + orientation_penalty + gait_reward

# PPO (Proximal Policy Optimization) で学習
# 4096環境を並列 → 1時間で数億ステップの学習が可能
# → 実機では歩行を即座に実行できる
```

### コード例6: テレオペレーション（遠隔操作）によるデモ収集

```python
import numpy as np
from dataclasses import dataclass
from typing import List
import h5py

@dataclass
class DemoStep:
    """デモの1ステップ"""
    timestamp: float
    image: np.ndarray       # (H, W, 3) RGB画像
    depth: np.ndarray       # (H, W) 深度画像
    joint_positions: np.ndarray   # 関節角度
    joint_velocities: np.ndarray  # 関節速度
    ee_position: np.ndarray      # エンドエフェクタ位置 [x, y, z]
    ee_orientation: np.ndarray   # エンドエフェクタ姿勢 [qx, qy, qz, qw]
    gripper_state: float         # グリッパー開閉 (0-1)

class TeleopDataCollector:
    """テレオペレーションによるデモンストレーションデータ収集"""

    def __init__(self, robot, camera, save_dir="demos"):
        self.robot = robot
        self.camera = camera
        self.save_dir = save_dir
        self.current_demo: List[DemoStep] = []
        self.demo_count = 0

    def start_recording(self):
        """デモ記録開始"""
        self.current_demo = []
        print("デモ記録を開始しました。")

    def record_step(self):
        """現在のロボット状態を記録"""
        step = DemoStep(
            timestamp=time.time(),
            image=self.camera.get_rgb(),
            depth=self.camera.get_depth(),
            joint_positions=self.robot.get_joint_positions(),
            joint_velocities=self.robot.get_joint_velocities(),
            ee_position=self.robot.get_ee_position(),
            ee_orientation=self.robot.get_ee_orientation(),
            gripper_state=self.robot.get_gripper_state(),
        )
        self.current_demo.append(step)

    def save_demo(self, task_name: str):
        """デモデータをHDF5形式で保存"""
        filename = f"{self.save_dir}/{task_name}_demo_{self.demo_count:04d}.hdf5"

        with h5py.File(filename, 'w') as f:
            n_steps = len(self.current_demo)
            f.attrs['task'] = task_name
            f.attrs['n_steps'] = n_steps

            # 各データフィールドをバッチで保存
            images = np.stack([s.image for s in self.current_demo])
            f.create_dataset('images', data=images, compression='gzip')

            actions = np.stack([
                np.concatenate([s.ee_position, s.ee_orientation, [s.gripper_state]])
                for s in self.current_demo
            ])
            f.create_dataset('actions', data=actions)

            joint_pos = np.stack([s.joint_positions for s in self.current_demo])
            f.create_dataset('joint_positions', data=joint_pos)

        self.demo_count += 1
        print(f"デモ保存完了: {filename} ({n_steps} ステップ)")

# テレオペデバイス
# - VR コントローラー (Meta Quest): 手の位置・姿勢を直接マッピング
# - 3Dマウス (SpaceMouse): 6DoFの入力デバイス
# - リーダーフォロワー: 2台のロボットアームで操作
# - Apple Vision Pro: 手のトラッキングでロボット制御
```

---

## 5. シミュレーションとSim-to-Real転移

### シミュレータの比較

| シミュレータ | 開発元 | GPU並列 | 物理エンジン | 主な用途 | ライセンス |
|------------|--------|--------|------------|---------|----------|
| Isaac Gym/Lab | NVIDIA | 数千環境 | PhysX | 強化学習 | 無料 |
| MuJoCo | Google | 限定的 | 独自 | 研究 | Apache 2.0 |
| PyBullet | Coumans | なし | Bullet | 教育・研究 | MIT |
| Gazebo | Open Robotics | なし | ODE/DART | ROS統合 | Apache 2.0 |
| Isaac Sim | NVIDIA | 対応 | PhysX 5 | フォトリアル | 無料 |
| Genesis | 研究チーム | 数万環境 | 独自 | 超大規模RL | 研究用 |

### Sim-to-Real 転移のフレームワーク

```
+-----------------------------------------------------------+
|  Sim-to-Real 転移の全体像                                   |
+-----------------------------------------------------------+
|                                                           |
|  シミュレーション                                          |
|  +----------------------------------------------+        |
|  |                                              |        |
|  |  1. ドメインランダム化                        |        |
|  |     摩擦: 0.2-1.0                            |        |
|  |     質量: ±20%                               |        |
|  |     重力: 9.6-10.2 m/s^2                     |        |
|  |     センサーノイズ: ±5%                       |        |
|  |     視覚: 色、照明、テクスチャのランダム化     |        |
|  |     遅延: 0-30ms のランダム制御遅延           |        |
|  |                                              |        |
|  |  2. 大規模並列学習                            |        |
|  |     Isaac Gym: 4096環境 → ~1時間で数億ステップ |        |
|  |                                              |        |
|  |  3. 方策の学習                                |        |
|  |     PPO / SAC / TD3                          |        |
|  +----------------------------------------------+        |
|                    |                                      |
|                    v                                      |
|  転移 (Sim-to-Real Gap の縮小)                            |
|  +----------------------------------------------+        |
|  |  システム同定: 実機パラメータの精密計測        |        |
|  |  残差学習: 実機データで微修正                  |        |
|  |  適応制御: 実機環境にオンラインで適応          |        |
|  +----------------------------------------------+        |
|                    |                                      |
|                    v                                      |
|  実機                                                     |
|  +----------------------------------------------+        |
|  |  少量の実機データでファインチューニング        |        |
|  |  安全フィルターの適用                          |        |
|  |  段階的な難易度上昇                            |        |
|  +----------------------------------------------+        |
+-----------------------------------------------------------+
```

---

## 6. 家庭用ロボット

### コード例7: Roomba 的な経路計画アルゴリズム

```python
import numpy as np
from enum import Enum

class CoverageState(Enum):
    SPIRAL = "spiral"
    WALL_FOLLOW = "wall_follow"
    RANDOM = "random"

class CoveragePlanner:
    """家庭用掃除ロボットのカバレッジ計画"""

    def __init__(self, grid_size=(100, 100)):
        self.grid = np.zeros(grid_size, dtype=bool)  # 掃除済みマップ
        self.obstacles = np.zeros(grid_size, dtype=bool)
        self.position = (50, 50)
        self.heading = 0  # 0-359度
        self.state = CoverageState.SPIRAL

    def plan_next_action(self, bumper_hit, cliff_detected):
        """センサー入力に基づいて次の行動を決定"""
        if cliff_detected:
            return self._backup_and_turn(180)

        if bumper_hit:
            self.state = CoverageState.WALL_FOLLOW
            return self._wall_follow()

        if self.state == CoverageState.SPIRAL:
            return self._spiral_outward()

        if self._coverage_percentage() > 0.9:
            return self._move_to_uncovered()

        return self._random_bounce()

    def _coverage_percentage(self):
        """掃除済み面積の割合"""
        cleanable = ~self.obstacles
        return np.sum(self.grid & cleanable) / np.sum(cleanable)

    def _spiral_outward(self):
        """スパイラル移動パターン"""
        # 中心から外側に向かって渦巻き状に移動
        # 障害物に当たったら状態遷移
        pass

    def _wall_follow(self):
        """壁沿い移動（部屋の端を掃除）"""
        pass

    def _move_to_uncovered(self):
        """未掃除エリアへ移動"""
        uncovered = ~self.grid & ~self.obstacles
        if np.any(uncovered):
            target = find_nearest_uncovered(self.position, uncovered)
            return plan_path_to(self.position, target)
```

### コード例8: ロボットアームの逆運動学

```python
import numpy as np

class SimpleRobotArm:
    """2リンクロボットアームの逆運動学"""

    def __init__(self, l1=0.3, l2=0.25):
        self.l1 = l1  # 第1リンク長 (m)
        self.l2 = l2  # 第2リンク長 (m)

    def forward_kinematics(self, theta1, theta2):
        """順運動学: 関節角度 → エンドエフェクタ位置"""
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        return x, y

    def inverse_kinematics(self, x, y):
        """逆運動学: 目標位置 → 関節角度"""
        d = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)

        if abs(d) > 1:
            raise ValueError("目標位置がワークスペース外です")

        theta2 = np.arctan2(np.sqrt(1 - d**2), d)  # 肘上解
        theta1 = np.arctan2(y, x) - np.arctan2(
            self.l2 * np.sin(theta2),
            self.l1 + self.l2 * np.cos(theta2)
        )

        return theta1, theta2

    def plan_trajectory(self, start, end, steps=50):
        """始点から終点への滑らかな軌道計画"""
        trajectory = []
        for t in np.linspace(0, 1, steps):
            # 3次補間で滑らかな動き
            s = 3*t**2 - 2*t**3  # smoothstep
            x = start[0] + s * (end[0] - start[0])
            y = start[1] + s * (end[1] - start[1])
            theta1, theta2 = self.inverse_kinematics(x, y)
            trajectory.append((theta1, theta2))
        return trajectory
```

### コード例9: 6DoF ロボットアームの運動学（DH法）

```python
import numpy as np

def dh_transform(theta, d, a, alpha):
    """
    Denavit-Hartenberg 変換行列
    theta: 関節角度 (回転関節)
    d: リンクオフセット
    a: リンク長
    alpha: リンクのねじり角
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ],
    ])

class Robot6DOF:
    """6自由度ロボットアーム（Puma 560 風 DH パラメータ）"""

    def __init__(self):
        # DH パラメータ [d, a, alpha] (theta は変数)
        self.dh_params = [
            [0.670, 0,     np.pi/2],   # Joint 1
            [0,     0.432, 0],          # Joint 2
            [0,     0.020, np.pi/2],    # Joint 3
            [0.432, 0,    -np.pi/2],    # Joint 4
            [0,     0,     np.pi/2],    # Joint 5
            [0.056, 0,     0],          # Joint 6
        ]

    def forward_kinematics(self, joint_angles):
        """順運動学: 6関節角度 → エンドエフェクタの4x4変換行列"""
        T = np.eye(4)
        for i, (d, a, alpha) in enumerate(self.dh_params):
            T = T @ dh_transform(joint_angles[i], d, a, alpha)
        return T

    def jacobian(self, joint_angles, delta=1e-6):
        """数値ヤコビアン: 関節速度 → エンドエフェクタ速度の変換"""
        J = np.zeros((6, 6))
        T0 = self.forward_kinematics(joint_angles)
        pos0 = T0[:3, 3]

        for i in range(6):
            angles_plus = joint_angles.copy()
            angles_plus[i] += delta
            T_plus = self.forward_kinematics(angles_plus)

            # 位置のヤコビアン
            J[:3, i] = (T_plus[:3, 3] - pos0) / delta

            # 姿勢のヤコビアン（簡略化）
            dR = T_plus[:3, :3] @ T0[:3, :3].T
            J[3:, i] = np.array([dR[2, 1], dR[0, 2], dR[1, 0]]) / delta

        return J

    def inverse_kinematics_numerical(self, target_pos, target_orient=None,
                                      max_iter=100, tol=1e-4):
        """数値的逆運動学（ヤコビアンベース）"""
        q = np.zeros(6)  # 初期姿勢

        for iteration in range(max_iter):
            T_current = self.forward_kinematics(q)
            pos_error = target_pos - T_current[:3, 3]

            if np.linalg.norm(pos_error) < tol:
                return q

            J = self.jacobian(q)
            J_pos = J[:3, :]  # 位置のヤコビアンのみ

            # ダンプ付き擬似逆行列（特異姿勢対策）
            lambda_sq = 0.01
            J_pinv = J_pos.T @ np.linalg.inv(
                J_pos @ J_pos.T + lambda_sq * np.eye(3)
            )

            dq = J_pinv @ pos_error
            q += dq * 0.5  # ステップサイズ

        return q
```

---

## 7. 安全設計と規格

### ロボット安全アーキテクチャ

```
+-----------------------------------------------------------+
|  ロボット安全の多層防御アーキテクチャ                        |
+-----------------------------------------------------------+
|                                                           |
|  Layer 5: 環境設計                                        |
|  +-- 作業ゾーンの物理的分離（フェンス、光カーテン）        |
|  +-- 速度・力の制限ゾーン設定                              |
|  +-- 緊急停止ボタンの配置                                  |
|                                                           |
|  Layer 4: AI安全フィルター                                 |
|  +-- 異常行動検出（学習済みモデルの出力チェック）          |
|  +-- 予測衝突回避（軌道予測 + 回避行動）                  |
|  +-- 不確実性の高い行動の拒否                              |
|                                                           |
|  Layer 3: ソフトウェア安全制約                              |
|  +-- 速度上限: max_velocity = 1.5 m/s（人と共存時）       |
|  +-- 力上限: max_force = 150 N（ISO/TS 15066準拠）        |
|  +-- ワークスペース制限（動作範囲の制限）                  |
|                                                           |
|  Layer 2: ハードウェア安全                                 |
|  +-- 力覚センサーによる衝突検出                            |
|  +-- バックドライバブルアクチュエータ（衝突時に力を逃す）  |
|  +-- 電流制限（トルクの物理的制限）                        |
|                                                           |
|  Layer 1: 緊急停止（E-Stop）                              |
|  +-- ハードウェアE-Stop（物理ボタン）                     |
|  +-- ソフトウェアE-Stop（通信断でも安全側に）              |
|  +-- SIL 3 (Safety Integrity Level 3) 準拠                |
|                                                           |
|  常に下位層が上位層を上書きできる構造にする                 |
+-----------------------------------------------------------+
```

### 主要ロボット安全規格

| 規格 | 内容 | 対象 |
|------|------|------|
| ISO 10218-1/2 | 産業用ロボットの安全要求 | 工場ロボット |
| ISO/TS 15066 | 協働ロボットの安全（力・速度制限） | コボット |
| ISO 13482 | パーソナルケアロボットの安全 | 家庭用ロボット |
| IEC 61508 | 機能安全の一般規格 | 全安全システム |
| ISO 26262 | 自動車の機能安全 | 自動運転車 |
| ISO/DIS 22166 | 自律移動ロボットの安全 | AMR |

---

## 8. ロボティクスの産業応用

### 産業分野別ロボット導入状況

```
+-----------------------------------------------------------+
|  産業ロボティクスの導入分野                                  |
+-----------------------------------------------------------+
|                                                           |
|  製造業                                                    |
|  +-- 自動車組立ライン: 溶接、塗装、組立                   |
|  +-- 電子機器: SMT実装、検査                              |
|  +-- 食品加工: ピッキング、パッケージング                  |
|  +-- BMW/Figure 02: ヒューマノイドの工場試験導入           |
|                                                           |
|  物流・倉庫                                                |
|  +-- Amazon: Proteus（自律移動ロボット）                   |
|  +-- Amazon: Digit（Agility Robotics、箱運搬）             |
|  +-- 仕分け: ピースピッキングロボット                      |
|  +-- 自律搬送車 (AMR): Fetch, Locus                       |
|                                                           |
|  農業                                                      |
|  +-- 収穫ロボット: イチゴ、トマトのピッキング              |
|  +-- 除草ロボット: カメラ+AIで雑草を識別                   |
|  +-- ドローン: 農薬散布、作物モニタリング                  |
|                                                           |
|  医療・介護                                                |
|  +-- 手術ロボット: da Vinci（直感的操作）                  |
|  +-- リハビリ: 外骨格ロボット                              |
|  +-- 搬送: 病院内の物品搬送ロボット                        |
|                                                           |
|  建設                                                      |
|  +-- Spot (Boston Dynamics): 建設現場の巡回・検査          |
|  +-- 3Dプリンティング: コンクリート積層                    |
|  +-- 解体ロボット: 危険区域の遠隔操作                      |
+-----------------------------------------------------------+
```

---

## 9. アンチパターン

### アンチパターン1: シミュレーションと現実のギャップを無視

```
NG: シミュレーション（MuJoCo, Isaac Sim）で完璧に動く
    → そのまま実機に転送
    → 現実では全く動かない（sim-to-real gap）

OK: Sim-to-Real 転移の対策を講じる
    1. ドメインランダム化: 物理パラメータを乱数で変える
       - 摩擦係数: 0.3-0.8
       - 重力: 9.6-10.0 m/s^2
       - 質量: ±10%
       - センサーノイズ: ±5%
       - 制御遅延: 0-30ms
    2. システム同定: 実機のパラメータを正確に計測
    3. 段階的転移: sim → sim-to-real → 実機少量データ
    4. 残差学習: simで基本方策 → 実機データで差分を学習
```

### アンチパターン2: 安全機構なしでのAI制御

```
NG: AIモデルの出力をそのままモーターに送る
    → 予期しない動作で人や物を損傷

OK: 多層安全アーキテクチャ
    Layer 1: AIポリシー（学習済みモデル）
    Layer 2: 安全フィルター（速度・力の上限制限）
    Layer 3: 衝突検出（力覚センサー閾値）
    Layer 4: 緊急停止（ハードウェアE-stop）

    常に Layer 2-4 が Layer 1 を上書きできる構造にする
```

### アンチパターン3: エンドツーエンドへの過信

```
NG: 画像入力 → 行動出力 を単一モデルで学習
    → ブラックボックスで何が起きているか分からない
    → デバッグ不可能、安全性の検証不可能

OK: モジュラーアーキテクチャ + 基盤モデル
    1. 知覚モジュール: 物体認識、環境理解
    2. 計画モジュール: タスク分解、行動計画
    3. 制御モジュール: 低レベルモーター制御
    各モジュールの入出力を検証可能にする
    基盤モデルは計画レベルで活用（制御は従来手法）
```

### アンチパターン4: データ収集の軽視

```
NG: シミュレーションデータだけで十分と考える
    → 実環境の多様性をカバーできない

OK: 体系的なデータ収集パイプライン
    1. テレオペレーション: VR/リーダーフォロワーで人間がデモ
    2. 自律探索: 安全範囲内でロボットが自己データ収集
    3. マルチロボット: 複数台で並列データ収集
    4. データ拡張: 視覚ランダム化、ノイズ追加
    目安: 1タスクあたり50-200デモ（Diffusion Policy基準）
```

---

## FAQ

### Q1. ヒューマノイドロボットはいつ家庭に普及するか？

2025年時点では商用デプロイの初期段階。BMW工場でのFigure 02の試験導入、AmazonでのDigitの導入テストなど産業用途が先行。家庭用は2030年代前半に初期製品、価格が車程度（300-500万円）まで下がるには2030年代後半と予測される。Tesla Optimusが量産を目指しているが、汎用家庭用ロボットの技術的課題（柔軟な物体操作、未知環境への適応）は依然として大きい。

### Q2. ロボット開発に必要なスキルセットは？

ハードウェア（メカニクス、電子回路）、ソフトウェア（C++/Python、ROS 2）、制御工学（PID、MPC）、AI（強化学習、コンピュータビジョン）、数学（線形代数、最適化）が必要。全てを一人でカバーする必要はなく、チーム開発が基本。2025年時点では特に模倣学習/Diffusion Policyの実装経験、Isaac Gym/Labでの強化学習経験が求められている。

### Q3. ROS 2 と Isaac Sim の関係は？

ROS 2 はロボットソフトウェアの標準フレームワーク（通信、センサー統合、経路計画）。Isaac Sim は NVIDIA のロボットシミュレータで、ROS 2 と連携して動作する。Isaac Sim でシミュレーション学習を行い、ROS 2 で実機に展開するのが典型的なワークフロー。Isaac Lab（旧Orbit）は強化学習に特化したフレームワークで、Isaac Sim上で動作する。

### Q4. Diffusion Policyとは何か？なぜ注目されているか？

Diffusion Policyは拡散モデル（Stable Diffusionと同じ原理）をロボットの行動生成に適用した手法。行動をノイズから段階的にデノイズして生成する。従来の模倣学習では表現できなかった「同じ状況で複数の正解がある」マルチモーダルな行動分布を自然に扱える。2024年以降、折り畳み、組立、調理などの精密操作タスクで従来手法を大幅に上回る成功率を示している。

### Q5. 強化学習の学習時間を短縮するには？

Isaac Gym/Lab を使ったGPU並列シミュレーションが最も効果的。1つのGPUで4096個以上の環境を同時にシミュレーションし、数時間で数億ステップの学習が可能。カリキュラム学習（簡単な環境から段階的に難しく）、報酬のシェーピング（段階的な報酬設計）、事前学習済みモデルの転移も有効。

---

## まとめ

| 概念 | 要点 |
|------|------|
| 知覚-計画-行動 | ロボット制御の基本サイクル |
| ROS 2 | ロボットソフトウェアの標準フレームワーク |
| 強化学習 | シミュレーションで動作を試行錯誤で獲得 |
| 模倣学習 | 人間のデモンストレーションから動作を学習 |
| Diffusion Policy | 拡散モデルベースの精密行動生成 |
| 基盤モデル | 言語指示でロボットを汎用的に制御 |
| Sim-to-Real | シミュレーションから実機への転移技術 |
| 逆運動学 | 目標位置から関節角度を逆算 |
| 安全フィルター | AI制御の上位に配置する安全制約 |
| テレオペレーション | 遠隔操作によるデモデータ収集 |
| QDD アクチュエータ | 安全なロボット向け低減速比モーター |

---

## 次に読むべきガイド

- **02-emerging/02-smart-home.md** — スマートホーム：Matter、AI家電
- **02-emerging/00-ar-vr-ai.md** — AR/VR×AI：Vision Pro、Quest
- **01-computing/02-edge-ai.md** — エッジAI：NPU、Coral、Jetson

---

## 参考文献

1. **ROS 2 公式ドキュメント** https://docs.ros.org/en/rolling/
2. **Google DeepMind — RT-2 論文** https://robotics-transformer2.github.io/
3. **Boston Dynamics 公式** https://bostondynamics.com/
4. **NVIDIA Isaac Sim** https://developer.nvidia.com/isaac-sim
5. **Diffusion Policy** https://diffusion-policy.cs.columbia.edu/
6. **Figure AI** https://figure.ai/
7. **Unitree Robotics** https://www.unitree.com/
