# ロボティクスガイド

> Boston Dynamics、Figure、家庭用ロボットなどAI時代のロボット技術を包括的に解説する

## この章で学ぶこと

1. **ロボティクスの基礎** — センサー、アクチュエータ、制御系の構成と役割
2. **主要ロボット企業** — Boston Dynamics、Figure、Tesla Optimus、家庭用ロボットの技術と戦略
3. **AIとロボットの融合** — 基盤モデル（Foundation Model）によるロボット制御の革新

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
|                                                           |
|  力覚系                                                    |
|  +-- 力覚/トルクセンサー: 接触力の検出                     |
|  +-- 触覚センサー: 表面テクスチャ・滑り検出                |
|  +-- 圧力センサー: 把持力の制御                            |
|                                                           |
|  慣性系                                                    |
|  +-- IMU (加速度+ジャイロ): 姿勢・動き検出                 |
|  +-- エンコーダ: 関節角度の精密計測                         |
|                                                           |
|  環境系                                                    |
|  +-- 超音波: 近距離障害物検出                              |
|  +-- 赤外線: 熱源検出、人感センサー                        |
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

### コード例3: 言語指示によるロボット制御（基盤モデル統合）

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
| 基盤モデル（FM） | 汎用タスク | 非常に高 | 開発中 | 高（大規模学習） | RT-2, Figure + OpenAI |

---

## 5. 家庭用ロボット

### コード例4: Roomba 的な経路計画アルゴリズム

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

### コード例5: ロボットアームの逆運動学

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

---

## 6. アンチパターン

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
    2. システム同定: 実機のパラメータを正確に計測
    3. 段階的転移: sim → sim-to-real → 実機少量データ
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

---

## FAQ

### Q1. ヒューマノイドロボットはいつ家庭に普及するか？

2025年時点では商用デプロイの初期段階。BMW工場でのFigure 02の試験導入、AmazonでのDigitの導入テストなど産業用途が先行。家庭用は2030年代前半に初期製品、価格が車程度（300-500万円）まで下がるには2030年代後半と予測される。

### Q2. ロボット開発に必要なスキルセットは？

ハードウェア（メカニクス、電子回路）、ソフトウェア（C++/Python、ROS 2）、制御工学（PID、MPC）、AI（強化学習、コンピュータビジョン）、数学（線形代数、最適化）が必要。全てを一人でカバーする必要はなく、チーム開発が基本。

### Q3. ROS 2 と Isaac Sim の関係は？

ROS 2 はロボットソフトウェアの標準フレームワーク（通信、センサー統合、経路計画）。Isaac Sim は NVIDIA のロボットシミュレータで、ROS 2 と連携して動作する。Isaac Sim でシミュレーション学習を行い、ROS 2 で実機に展開するのが典型的なワークフロー。

---

## まとめ

| 概念 | 要点 |
|------|------|
| 知覚-計画-行動 | ロボット制御の基本サイクル |
| ROS 2 | ロボットソフトウェアの標準フレームワーク |
| 強化学習 | シミュレーションで動作を試行錯誤で獲得 |
| 模倣学習 | 人間のデモンストレーションから動作を学習 |
| 基盤モデル | 言語指示でロボットを汎用的に制御 |
| Sim-to-Real | シミュレーションから実機への転移技術 |
| 逆運動学 | 目標位置から関節角度を逆算 |
| 安全フィルター | AI制御の上位に配置する安全制約 |

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
