# AR/VR × AI ガイド

> Vision Pro、Quest、AI空間コンピューティングの技術と未来を包括的に解説する

## この章で学ぶこと

1. **AR/VRの基礎技術** — ディスプレイ、トラッキング、レンダリングの仕組み
2. **主要プラットフォーム** — Apple Vision Pro、Meta Quest、その他XRデバイスの比較
3. **AI × 空間コンピューティング** — AIがAR/VRにもたらす革新と実践的な開発手法

---

## 1. AR/VR/MR の基本概念

### Reality-Virtuality Continuum（現実-仮想連続体）

```
現実世界                                           仮想世界
   |                                                  |
   v                                                  v
+------+--------+----------------+-----------+--------+
| 現実  |   AR   |      MR       |    VR     | 完全仮想|
|      | 拡張現実| 複合現実       | 仮想現実  |        |
+------+--------+----------------+-----------+--------+
   |       |            |              |          |
   |   スマホ AR    Vision Pro     Quest 3     VRChat
   |   ポケモンGO  HoloLens 2    (パススルー)  メタバース
   |   Google Lens                PSVR 2
   |
   +--- XR (Extended Reality): AR + MR + VR の総称
```

### XRヘッドセットの基本構成

```
+-----------------------------------------------------------+
|              XR ヘッドセット 内部構造                        |
+-----------------------------------------------------------+
|                                                           |
|  +------------------+  +------------------+               |
|  | ディスプレイ      |  | レンズ           |               |
|  | Micro-OLED       |  | パンケーキレンズ  |               |
|  | 片目 2K-4K       |  | 薄型・軽量化     |               |
|  +------------------+  +------------------+               |
|                                                           |
|  +------------------+  +------------------+               |
|  | チップセット      |  | センサー群       |               |
|  | SoC (GPU+CPU+NPU)|  | カメラ(パススルー)|               |
|  | Apple M2/R1      |  | LiDAR/ToF       |               |
|  | Snapdragon XR2   |  | IMU (加速度+ジャイロ)|           |
|  +------------------+  +------------------+               |
|                                                           |
|  +------------------+  +------------------+               |
|  | トラッキング      |  | 入力デバイス     |               |
|  | Inside-Out 6DoF  |  | ハンドトラッキング|               |
|  | アイトラッキング  |  | コントローラ     |               |
|  | SLAM             |  | 音声入力         |               |
|  +------------------+  +------------------+               |
+-----------------------------------------------------------+
```

---

## 2. 主要XRプラットフォーム比較

### デバイス比較表

| 項目 | Apple Vision Pro | Meta Quest 3 | Meta Quest Pro | PSVR 2 |
|------|-----------------|--------------|----------------|--------|
| 発売年 | 2024 | 2023 | 2022 | 2023 |
| 価格 | $3,499 | $499 | $999 | $549 |
| チップ | M2 + R1 | Snapdragon XR2 Gen 2 | Snapdragon XR2+ | 独自チップ(PS5接続) |
| 解像度(片目) | 3,660×3,200 | 2,064×2,208 | 1,800×1,920 | 2,000×2,040 |
| リフレッシュレート | 90-100Hz | 90-120Hz | 90Hz | 90-120Hz |
| トラッキング | 6DoF + Eye + Hand | 6DoF + Hand | 6DoF + Eye + Hand | 6DoF + Eye |
| パススルー | 高品質カラー | カラー | カラー | なし(純VR) |
| OS | visionOS | Android(Meta Horizon) | Android(Meta Horizon) | PS5専用 |
| 重量 | 600-650g | 515g | 722g | 560g |
| 主な用途 | 空間コンピューティング | ゲーム・MR | ビジネス | ゲーム |

### ディスプレイ技術の進化

```
+-----------------------------------------------------------+
|  XR ディスプレイ技術                                        |
+-----------------------------------------------------------+
|                                                           |
|  LCD         |████████████|  低コスト、Quest 2             |
|  解像度: 中   コントラスト: 低   応答速度: 中               |
|                                                           |
|  OLED        |██████████████████|  高コントラスト、PSVR 2  |
|  解像度: 高   コントラスト: 高   応答速度: 高               |
|                                                           |
|  Micro-OLED  |████████████████████████|  Vision Pro        |
|  解像度: 最高  コントラスト: 最高  応答速度: 最高            |
|                                                           |
|  Micro-LED   |██████████████████████████████|  次世代       |
|  解像度: 最高  コントラスト: 最高  輝度: 最高  消費電力: 低 |
+-----------------------------------------------------------+
```

---

## 3. AI × 空間コンピューティング

### AIが変えるXR体験

```
+-----------------------------------------------------------+
|  AI × XR の融合領域                                        |
+-----------------------------------------------------------+
|                                                           |
|  視覚AI                                                   |
|  +-- 空間認識: 部屋の3Dマッピング (SLAM + NeRF)           |
|  +-- 物体認識: 現実世界の物体をリアルタイム認識            |
|  +-- セグメンテーション: 背景と前景の分離                  |
|  +-- オクルージョン: 仮想物体の前後関係を正確に処理        |
|                                                           |
|  自然言語AI                                                |
|  +-- 音声コマンド: 空間UIの音声操作                        |
|  +-- リアルタイム翻訳: AR字幕                              |
|  +-- 空間的会話AI: 仮想アバターとの対話                    |
|                                                           |
|  生成AI                                                    |
|  +-- 3Dアセット生成: テキストから3Dモデル                  |
|  +-- 環境生成: AIによるVR空間の自動生成                    |
|  +-- アバター生成: 写真1枚からリアルアバター                |
|                                                           |
|  予測AI                                                    |
|  +-- フォービエイテッドレンダリング: 視線予測で描画最適化  |
|  +-- モーション予測: 遅延を感じさせない動き補間            |
|  +-- アダプティブ品質: 負荷予測による動的品質調整          |
+-----------------------------------------------------------+
```

### コード例1: ARKit での平面検出と3D配置

```swift
import ARKit
import RealityKit

class ViewController: UIViewController, ARSessionDelegate {
    @IBOutlet var arView: ARView!

    override func viewDidLoad() {
        super.viewDidLoad()

        // AR セッション設定
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        config.sceneReconstruction = .meshWithClassification
        config.environmentTexturing = .automatic

        arView.session.delegate = self
        arView.session.run(config)

        // タップで3Dオブジェクトを配置
        let tapGesture = UITapGestureRecognizer(
            target: self, action: #selector(handleTap)
        )
        arView.addGestureRecognizer(tapGesture)
    }

    @objc func handleTap(_ sender: UITapGestureRecognizer) {
        let location = sender.location(in: arView)

        // レイキャストで平面との交差点を取得
        if let result = arView.raycast(
            from: location,
            allowing: .estimatedPlane,
            alignment: .horizontal
        ).first {
            // 3Dオブジェクトの配置
            let anchor = AnchorEntity(world: result.worldTransform)
            let box = ModelEntity(
                mesh: .generateBox(size: 0.1),
                materials: [SimpleMaterial(color: .blue, isMetallic: true)]
            )
            anchor.addChild(box)
            arView.scene.addAnchor(anchor)
        }
    }
}
```

### コード例2: visionOS での空間コンピューティング

```swift
import SwiftUI
import RealityKit

@main
struct MyVisionApp: App {
    var body: some Scene {
        // ウィンドウ（2D UI）
        WindowGroup {
            ContentView()
        }

        // ボリューム（3D コンテンツ）
        WindowGroup(id: "3d-viewer") {
            VolumetricView()
        }
        .windowStyle(.volumetric)
        .defaultSize(width: 0.5, height: 0.5, depth: 0.5, in: .meters)

        // イマーシブ空間（フル没入体験）
        ImmersiveSpace(id: "immersive") {
            ImmersiveView()
        }
    }
}

struct VolumetricView: View {
    var body: some View {
        RealityView { content in
            // 3Dモデルの読み込みと表示
            if let model = try? await ModelEntity(named: "Globe") {
                model.scale = [0.3, 0.3, 0.3]
                content.add(model)
            }
        }
        .gesture(
            // ハンドジェスチャーで回転
            RotateGesture3D()
                .targetedToAnyEntity()
                .onChanged { value in
                    value.entity.transform.rotation = value.rotation
                }
        )
    }
}
```

### コード例3: AI による空間理解

```python
# Meta Quest の Scene Understanding API（概念コード）
# MR アプリで部屋の構造を理解する

class SpatialAIProcessor:
    def __init__(self):
        self.scene_model = load_model("room_segmentation_v2")
        self.object_detector = load_model("3d_object_detection")

    def process_scene(self, depth_map, rgb_image, imu_data):
        """空間認識パイプライン"""
        # 1. 深度マップから3Dポイントクラウド生成
        point_cloud = depth_to_pointcloud(depth_map, camera_intrinsics)

        # 2. セマンティックセグメンテーション（壁、床、天井、家具）
        segmentation = self.scene_model.predict(point_cloud, rgb_image)
        # → {'wall': [...], 'floor': [...], 'ceiling': [...], 'furniture': [...]}

        # 3. 3D物体検出と分類
        objects = self.object_detector.detect(point_cloud, rgb_image)
        # → [{'class': 'chair', 'bbox_3d': ..., 'confidence': 0.95}, ...]

        # 4. 空間メッシュの構築
        scene_mesh = reconstruct_mesh(point_cloud, segmentation)

        return {
            'mesh': scene_mesh,
            'objects': objects,
            'planes': extract_planes(segmentation),
        }
```

---

## 4. フォービエイテッドレンダリング

### AI視線予測による描画最適化

```
+-----------------------------------------------------------+
|  フォービエイテッド レンダリング                             |
+-----------------------------------------------------------+
|                                                           |
|  従来: 画面全体を高解像度でレンダリング                    |
|  +--------------------------------------------+          |
|  |############################################|          |
|  |############################################|          |
|  |############################################|          |
|  |############################################|          |
|  +--------------------------------------------+          |
|  → GPU負荷: 100%                                         |
|                                                           |
|  フォービエイテッド: 視線の中心のみ高解像度                |
|  +--------------------------------------------+          |
|  |.........:::::::::::::::::::::...............|          |
|  |......::::::::#########::::::::..............|          |
|  |....::::::::##(視線中心)##::::::::............|          |
|  |......::::::::#########::::::::..............|          |
|  |.........:::::::::::::::::::::...............|          |
|  +--------------------------------------------+          |
|  # = 高解像度  : = 中解像度  . = 低解像度                 |
|  → GPU負荷: 30-50%（AI視線予測で遅延補償）                |
+-----------------------------------------------------------+
```

---

## 5. XR開発プラットフォーム比較表

| 項目 | Unity | Unreal Engine | visionOS (RealityKit) | WebXR |
|------|-------|--------------|----------------------|-------|
| 対応デバイス | Quest, Vision Pro, PSVR等 | Quest, PSVR等 | Vision Pro専用 | ブラウザ全般 |
| 言語 | C# | C++/Blueprint | Swift | JavaScript |
| 学習コスト | 中 | 高 | 中 | 低 |
| グラフィック品質 | 高 | 非常に高 | 高 | 中 |
| AI統合 | Barracuda, ONNX | NNE Plugin | Core ML, Create ML | TF.js, ONNX.js |
| 3D物理演算 | PhysX | Chaos Physics | RealityKit Physics | Ammo.js |
| ライセンス | 無料〜有料 | ロイヤリティ制 | 無料 | 無料 |

---

## 6. アンチパターン

### アンチパターン1: VR酔いを無視した設計

```
NG:
- フレームレートが60fps以下に落ちる
- カメラをプログラムで強制移動させる
- 加速度のある移動（急発進・急停止）
- UIを視界の端に固定配置する

OK:
- 常に90fps以上を維持（フォービエイテッドレンダリング活用）
- 移動はテレポート方式 or ビネット効果付き
- ユーザー主導のカメラ制御
- 固定UIの代わりに空間にアンカーされたUI
- 加速度運動を避け、等速直線運動を基本にする
```

### アンチパターン2: エッジケースでのトラッキング喪失

```
NG:
- トラッキング喪失時に何も対策しない
  → ユーザーの位置が突然飛ぶ、VR酔い発生

OK:
- IMU（慣性計測装置）でのフォールバック推定
- トラッキング喪失を検出してユーザーに通知
- 最後の有効な位置にスムーズに戻す
- 暗い部屋や反射面など、喪失しやすい環境の事前検出
```

---

## FAQ

### Q1. Vision Pro は買うべきか？

2024年時点で一般消費者にはまだ高価（$3,499）。空間コンピューティングの開発者、3D映像・デザインのプロフェッショナル、アーリーアダプターには価値がある。一般的なVRゲーム目的なら Quest 3（$499）の方がコスパが圧倒的に良い。

### Q2. WebXR で実用的なAR/VRアプリは作れるか？

シンプルなARフィルターや3Dビューアーは十分実用的。Three.js + WebXR APIで開発でき、アプリストアを介さずURLで配布できるのが強み。ただしネイティブアプリと比べてGPU性能への制約が大きく、複雑なMR体験には向かない。

### Q3. 空間コンピューティングの「キラーアプリ」は何か？

現時点で最も有望なのは、1) リモートコラボレーション（空間を共有した会議）、2) 空間デザイン（建築・インテリアの実寸プレビュー）、3) 教育・トレーニング（外科手術シミュレーション等）、4) エンターテインメント（空間を使った没入体験）。AIアシスタントとの空間的対話も今後の有力候補。

---

## まとめ

| 概念 | 要点 |
|------|------|
| AR/VR/MR/XR | 現実-仮想の連続体上の各技術 |
| Vision Pro | 空間コンピューティングの先駆、Micro-OLED |
| Quest 3 | 最もコスパの良いMRデバイス |
| SLAM | 自己位置推定と環境地図の同時構築 |
| フォービエイテッドレンダリング | AI視線予測でGPU負荷60-70%削減 |
| 空間UI | 3D空間に配置するユーザーインターフェース |
| NeRF / 3D Gaussian Splatting | AIによる3Dシーン再構築 |
| 6DoF | 位置(x,y,z) + 回転(pitch,yaw,roll) の6自由度トラッキング |

---

## 次に読むべきガイド

- **02-emerging/01-robotics.md** — ロボティクス：Boston Dynamics、Figure
- **02-emerging/02-smart-home.md** — スマートホーム：Matter、AI家電
- **01-computing/02-edge-ai.md** — エッジAI：NPU、Coral、Jetson

---

## 参考文献

1. **Apple — visionOS Developer Documentation** https://developer.apple.com/visionos/
2. **Meta Quest Developer Hub** https://developer.oculus.com/
3. **WebXR Device API — W3C** https://www.w3.org/TR/webxr/
4. **NeRF (Neural Radiance Fields) 原論文** https://www.matthewtancik.com/nerf
