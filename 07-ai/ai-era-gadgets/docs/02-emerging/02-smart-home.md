# スマートホームガイド

> Matter、AI家電、音声アシスタントを活用した次世代スマートホームの構築と技術を解説する

## この章で学ぶこと

1. **スマートホームプロトコル** — Matter、Thread、Zigbee、Wi-Fi の違いと選び方
2. **AI家電の進化** — 音声アシスタント、AIカメラ、予測制御の技術基盤
3. **実践的構築** — Home Assistant を中心としたスマートホームシステムの設計と自動化

---

## 1. スマートホームの全体像

### スマートホームのレイヤー構造

```
+-----------------------------------------------------------+
|                  スマートホーム アーキテクチャ               |
+-----------------------------------------------------------+
|                                                           |
|  +----------------------------------------------------+  |
|  | アプリケーション層                                    |  |
|  | Apple Home / Google Home / Alexa / Home Assistant   |  |
|  | 自動化ルール、シーン、音声制御                        |  |
|  +----------------------------------------------------+  |
|                          |                                |
|  +----------------------------------------------------+  |
|  | プロトコル層                                          |  |
|  | Matter / HomeKit / Google Home API / Alexa Skills   |  |
|  +----------------------------------------------------+  |
|                          |                                |
|  +----------------------------------------------------+  |
|  | 通信層                                                |  |
|  | Thread / Wi-Fi / Zigbee / Z-Wave / Bluetooth LE    |  |
|  +----------------------------------------------------+  |
|                          |                                |
|  +----------------------------------------------------+  |
|  | デバイス層                                            |  |
|  | 照明 / サーモスタット / カメラ / ドアロック / センサー  |  |
|  +----------------------------------------------------+  |
+-----------------------------------------------------------+
```

### Matter の登場による統一

```
【Matter 以前】                    【Matter 以後】

+--------+  +--------+  +------+  +----------------------------+
| HomeKit|  | Google |  | Alexa|  |         Matter             |
|  only  |  | only   |  | only |  |  (Apple + Google + Amazon  |
+--------+  +--------+  +------+  |   + Samsung + ...)         |
    |           |           |     +----------------------------+
    v           v           v              |
 一部の      一部の      一部の     ほぼ全ての対応デバイスが
 デバイス    デバイス    デバイス    全プラットフォームで動作
```

---

## 2. 通信プロトコル比較

### プロトコル比較表

| プロトコル | 周波数帯 | 通信距離 | 消費電力 | 速度 | メッシュ | 主な用途 |
|-----------|---------|---------|---------|------|---------|---------|
| Wi-Fi | 2.4/5/6 GHz | 30-50m | 高 | 高速 | 非対応 | カメラ、ディスプレイ |
| Thread | 2.4 GHz | 10-30m | 非常に低 | 中速 | 対応 | センサー、照明 |
| Zigbee | 2.4 GHz | 10-20m | 非常に低 | 低速 | 対応 | センサー、スイッチ |
| Z-Wave | 900 MHz | 30-100m | 低 | 低速 | 対応 | ドアロック、センサー |
| Bluetooth LE | 2.4 GHz | 10-30m | 非常に低 | 低速 | 対応(Mesh) | 近距離小型デバイス |
| Matter | 上記の上位層 | プロトコル依存 | プロトコル依存 | - | Thread経由 | 統一規格 |

### Thread ネットワーク構造

```
+-----------------------------------------------------------+
|  Thread メッシュネットワーク                                 |
+-----------------------------------------------------------+
|                                                           |
|  +------+            +------+            +------+         |
|  |Border|--- Wi-Fi --|Router|--- Wi-Fi --|Border|         |
|  |Router|            |      |            |Router|         |
|  +--+---+            +--+---+            +--+---+         |
|     |                   |                   |             |
|   Thread              Thread              Thread          |
|   mesh                mesh                mesh            |
|     |                   |                   |             |
|  +--+---+  +------+  +--+---+  +------+  +--+---+        |
|  | 照明  |--| 温度 |--| 照明  |--|ドアロック|--| 温度 |       |
|  |      |  |センサー|  |      |  |       |  |センサー|       |
|  +------+  +------+  +------+  +------+  +------+        |
|                                                           |
|  Border Router: Thread ←→ Wi-Fi/Ethernet のブリッジ       |
|  (Apple TV, Google Nest Hub, HomePod mini が対応)         |
+-----------------------------------------------------------+
```

---

## 3. Matter プロトコル

### Matterの技術構造

```
+-----------------------------------------------------------+
|                    Matter プロトコル                        |
+-----------------------------------------------------------+
|                                                           |
|  +----------------------------------------------------+  |
|  | Application Layer                                   |  |
|  | デバイスタイプ定義（照明、サーモスタット、ドアロック）  |  |
|  | クラスタ（On/Off, Level Control, Color Control）     |  |
|  +----------------------------------------------------+  |
|                                                           |
|  +----------------------------------------------------+  |
|  | Interaction Model                                   |  |
|  | Read / Write / Subscribe / Invoke                   |  |
|  +----------------------------------------------------+  |
|                                                           |
|  +----------------------------------------------------+  |
|  | Security Layer                                      |  |
|  | CASE (Certificate Authenticated Session)             |  |
|  | PASE (Passcode Authenticated Session)                |  |
|  +----------------------------------------------------+  |
|                                                           |
|  +----------------------------------------------------+  |
|  | Transport Layer                                     |  |
|  | IPv6 over Wi-Fi / Thread / Ethernet                 |  |
|  +----------------------------------------------------+  |
+-----------------------------------------------------------+
```

### コード例1: Matter デバイスの制御（概念コード）

```python
# Matter デバイスの制御例（Python chip-tool ライクなAPI）
from matter_sdk import MatterController, clusters

async def control_smart_home():
    controller = MatterController()

    # デバイスの検出とペアリング
    devices = await controller.discover()
    print(f"検出されたデバイス: {len(devices)}台")

    for device in devices:
        print(f"  - {device.name} (Type: {device.device_type})")

    # 照明の制御
    light = controller.get_device("living_room_light")

    # On/Off クラスタ
    await light.clusters.on_off.on()

    # Level Control クラスタ（明るさ）
    await light.clusters.level_control.move_to_level(
        level=128,           # 0-254 (50%)
        transition_time=10,  # 1秒 (10 = 1s)
    )

    # Color Control クラスタ（色温度）
    await light.clusters.color_control.move_to_color_temperature(
        color_temperature_mireds=370,  # 2700K (暖白色)
        transition_time=20,
    )

    # サーモスタットの制御
    thermostat = controller.get_device("thermostat")
    await thermostat.clusters.thermostat.set_setpoint(
        mode="heating",
        temperature=22.0,  # 摂氏
    )

    # ドアロックの制御
    lock = controller.get_device("front_door")
    await lock.clusters.door_lock.lock()
    status = await lock.clusters.door_lock.get_lock_state()
    print(f"ドアの状態: {status}")  # "locked"
```

### コード例2: Home Assistant の自動化設定

```yaml
# Home Assistant の automation.yaml
# AI的な条件判断を含む自動化ルール

# 自動化1: 日没時に照明を自動点灯
- alias: "日没時の照明自動化"
  trigger:
    - platform: sun
      event: sunset
      offset: "-00:30:00"  # 日没30分前
  condition:
    - condition: state
      entity_id: binary_sensor.occupancy_living_room
      state: "on"  # 在宅時のみ
  action:
    - service: light.turn_on
      target:
        entity_id: light.living_room
      data:
        brightness_pct: 70
        color_temp_kelvin: 3000
        transition: 30  # 30秒かけてフェードイン

# 自動化2: 外出検知で省エネモード
- alias: "全員外出で省エネモード"
  trigger:
    - platform: state
      entity_id: group.family
      to: "not_home"
      for: "00:10:00"  # 10分間不在
  action:
    - service: climate.set_temperature
      target:
        entity_id: climate.main_thermostat
      data:
        temperature: 18  # 暖房を下げる
    - service: light.turn_off
      target:
        entity_id: all
    - service: switch.turn_off
      target:
        entity_id: switch.entertainment_system

# 自動化3: AIカメラ連携（人物検出）
- alias: "不審者検知アラート"
  trigger:
    - platform: state
      entity_id: image_processing.front_camera_person_detection
      to: "detected"
  condition:
    - condition: state
      entity_id: group.family
      state: "not_home"
  action:
    - service: notify.mobile_app
      data:
        title: "セキュリティ警告"
        message: "玄関カメラで人物を検出しました"
        data:
          image: "/api/camera_proxy/camera.front_door"
    - service: light.turn_on
      target:
        entity_id: light.porch
      data:
        brightness_pct: 100
```

---

## 4. 音声アシスタントとAI

### 音声アシスタント比較表

| 項目 | Amazon Alexa | Google Assistant | Apple Siri | Home Assistant Voice |
|------|-------------|-----------------|------------|---------------------|
| デバイス | Echo シリーズ | Nest シリーズ | HomePod, iPhone | 自作/ESP32 |
| スマートホーム統合 | 非常に広い | 広い | HomeKit中心 | 最も広い(DIY) |
| AI能力 | Alexa LLM | Gemini統合 | Apple Intelligence | ローカルLLM対応 |
| プライバシー | クラウド処理 | クラウド処理 | オンデバイス重視 | 完全ローカル可 |
| Skills/Actions | 10万+ | 数万 | Siri Shortcuts | Home Assistantの全機能 |
| 日本語対応 | 対応 | 対応 | 対応 | コミュニティ対応 |
| 価格帯 | 3,000-30,000円 | 5,000-30,000円 | 15,000-50,000円 | 自作コスト |

### 音声認識パイプライン

```
+-----------------------------------------------------------+
|  音声アシスタント処理フロー                                  |
+-----------------------------------------------------------+
|                                                           |
|  「アレクサ、リビングの照明を暖色に」                       |
|      |                                                    |
|      v                                                    |
|  +--------------------+                                   |
|  | ウェイクワード検出   |  ← デバイス上で常時動作（NPU）   |
|  | "アレクサ" を検知   |                                   |
|  +--------------------+                                   |
|      |                                                    |
|      v                                                    |
|  +--------------------+                                   |
|  | 音声認識 (ASR)     |  ← クラウド or オンデバイス        |
|  | 音声 → テキスト     |    Whisper, Google ASR           |
|  +--------------------+                                   |
|      |                                                    |
|      v                                                    |
|  +--------------------+                                   |
|  | 自然言語理解 (NLU) |  ← LLM ベースが主流に             |
|  | Intent: 照明制御    |    意図と実体の抽出               |
|  | Entity: リビング    |                                   |
|  | Entity: 暖色        |                                   |
|  +--------------------+                                   |
|      |                                                    |
|      v                                                    |
|  +--------------------+                                   |
|  | スキル/アクション   |  ← デバイスAPI呼び出し            |
|  | light.set_color()  |                                   |
|  +--------------------+                                   |
|      |                                                    |
|      v                                                    |
|  +--------------------+                                   |
|  | 音声合成 (TTS)     |  ← 確認応答を生成                 |
|  | 「リビングの照明を   |                                   |
|  |  暖色にしました」   |                                   |
|  +--------------------+                                   |
+-----------------------------------------------------------+
```

---

## 5. AI 家電の技術

### コード例3: エネルギー最適化AI

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class SmartThermostatAI:
    """
    AIによるサーモスタットの予測制御
    ユーザーの行動パターンと外気温から最適な温度を予測
    """
    def __init__(self):
        self.comfort_model = GradientBoostingRegressor()
        self.occupancy_model = GradientBoostingRegressor()

    def train(self, history_data):
        """過去の行動データから学習"""
        features = self._extract_features(history_data)
        # 特徴量: 時刻, 曜日, 外気温, 湿度, 過去の設定温度
        # ターゲット: ユーザーが設定した温度

        self.comfort_model.fit(
            features, history_data['target_temperature']
        )
        self.occupancy_model.fit(
            features, history_data['is_occupied']
        )

    def predict_schedule(self, forecast_weather, day_of_week):
        """24時間の温度スケジュールを予測"""
        schedule = []
        for hour in range(24):
            features = np.array([[
                hour,
                day_of_week,
                forecast_weather[hour]['temperature'],
                forecast_weather[hour]['humidity'],
            ]])

            predicted_temp = self.comfort_model.predict(features)[0]
            occupancy_prob = self.occupancy_model.predict(features)[0]

            # 不在予測時は省エネ温度に
            if occupancy_prob < 0.3:
                target_temp = predicted_temp - 3  # 3度下げる
            else:
                target_temp = predicted_temp

            schedule.append({
                'hour': hour,
                'target': round(target_temp, 1),
                'occupancy': round(occupancy_prob, 2),
            })

        return schedule

    def estimate_energy_savings(self, schedule, baseline=22.0):
        """省エネ効果を推定"""
        # 温度1度下げると約7%の省エネ
        savings = sum(
            max(0, baseline - s['target']) * 0.07
            for s in schedule
        ) / 24
        return f"推定省エネ率: {savings*100:.1f}%"
```

### コード例4: AIカメラの人物検出

```python
# ローカルAIカメラ（Frigate NVR + Home Assistant連携）
# frigate.yml 設定例

# Frigate NVR の設定
mqtt:
  host: 192.168.1.100

detectors:
  coral:
    type: edgetpu
    device: usb  # Google Coral USB Accelerator

cameras:
  front_door:
    ffmpeg:
      inputs:
        - path: rtsp://192.168.1.50:554/stream
          roles:
            - detect
            - record
    detect:
      width: 1280
      height: 720
      fps: 5
    objects:
      track:
        - person
        - car
        - dog
      filters:
        person:
          min_area: 5000
          max_area: 100000
          threshold: 0.7
    zones:
      front_yard:
        coordinates: 0,300,640,300,640,720,0,720
    record:
      enabled: true
      retain:
        days: 7
      events:
        retain:
          default: 30  # イベント映像は30日保持
    snapshots:
      enabled: true
      retain:
        default: 30
```

### コード例5: Home Assistant でのローカルLLM統合

```yaml
# Home Assistant の configuration.yaml
# ローカル LLM (Ollama) との統合

# Ollama 音声アシスタント統合
conversation:
  intents:
    # カスタムインテント定義

# Extended OpenAI Conversation (カスタムコンポーネント)
# ローカルの Ollama にリクエスト
openai_conversation:
  api_key: "sk-not-needed"
  base_url: "http://192.168.1.100:11434/v1"
  model: "llama3.1"
  prompt: |
    あなたはスマートホームアシスタントです。
    以下のデバイスを制御できます:
    - light.living_room: リビング照明
    - light.bedroom: 寝室照明
    - climate.thermostat: エアコン
    - lock.front_door: 玄関ドアロック
    - cover.curtain_living: リビングカーテン

    ユーザーの要望に対して、適切なサービスコールを生成してください。
    応答は日本語で行ってください。
```

---

## 6. セキュリティとプライバシー

### スマートホームセキュリティの層

```
+-----------------------------------------------------------+
|  スマートホーム セキュリティ層                               |
+-----------------------------------------------------------+
|                                                           |
|  Layer 4: アプリケーションセキュリティ                     |
|  +-- 二要素認証 (2FA)                                     |
|  +-- アクセスログの監視                                    |
|  +-- ゲストアクセスの期限管理                              |
|                                                           |
|  Layer 3: プロトコルセキュリティ                            |
|  +-- Matter: CASE (証明書認証)                             |
|  +-- HomeKit: Ed25519 暗号化                               |
|  +-- TLS 1.3 通信暗号化                                    |
|                                                           |
|  Layer 2: ネットワークセキュリティ                          |
|  +-- IoT専用VLAN分離                                       |
|  +-- ファイアウォールルール                                 |
|  +-- DNS over HTTPS (DoH)                                 |
|                                                           |
|  Layer 1: デバイスセキュリティ                              |
|  +-- ファームウェア自動更新                                 |
|  +-- セキュアブート                                        |
|  +-- デフォルトパスワードの変更                             |
+-----------------------------------------------------------+
```

---

## 7. アンチパターン

### アンチパターン1: 全デバイスをWi-Fiに接続

```
NG: 照明、センサー、カメラ全てをWi-Fiで接続
    → Wi-Fiルーターの接続上限超過（通常30-50台）
    → 遅延増大、ネットワーク不安定化

OK: プロトコルを適材適所で使い分け
    Wi-Fi: カメラ（高帯域が必要）、スマートディスプレイ
    Thread: 照明、温度センサー、ドアセンサー（低消費電力メッシュ）
    Zigbee: 既存のIKEAやPhilips Hueデバイス
    Bluetooth: ビーコン、近距離一時接続
```

### アンチパターン2: クラウド依存の過信

```
NG: 全ての自動化をクラウドサービスに依存
    → インターネット切断時に全てが停止
    → サービス終了で全デバイスがブリック化

OK: ローカルファースト設計
    1. Home Assistant などローカルハブを中核に
    2. 照明・ドアロックなど基本制御はローカルで完結
    3. クラウドは付加価値（リモートアクセス、AI機能）のみ
    4. Matter/Thread 対応デバイスを優先（ローカル通信可）
```

---

## FAQ

### Q1. Matter 対応デバイスと非対応デバイス、どちらを買うべき？

今後の購入は原則 Matter 対応を推奨。Matter 対応デバイスはApple Home、Google Home、Amazon Alexa全てで動作し、ベンダーロックインを避けられる。ただし Zigbee の既存エコシステム（IKEA TRADFRI、Philips Hue）は Bridge経由で Matter に対応するため、既存デバイスの買い替えは不要。

### Q2. Home Assistant の導入コストと難易度は？

Raspberry Pi 4/5（1-2万円）にインストールすれば最低限で始められる。専用ハードウェアの Home Assistant Green（約15,000円）やHome Assistant Yellow もある。初期設定はGUIで行え、基本的な自動化はYAML不要。高度なカスタマイズにはYAMLとPythonの知識が必要。

### Q3. 音声アシスタントのプライバシーは大丈夫か？

Amazon Alexa と Google Assistant はデフォルトで音声をクラウドに送信する。プライバシーが重要なら、1) 録音データの自動削除を設定、2) Apple Siri（オンデバイス処理重視）を使う、3) Home Assistant のローカル音声処理（Wyoming protocol + Whisper + Piper）でクラウドを完全排除する。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Matter | Apple/Google/Amazon統一のスマートホーム規格 |
| Thread | 低消費電力メッシュネットワーク（Matter推奨通信層） |
| Home Assistant | オープンソースのローカルスマートホームハブ |
| 音声アシスタント | Alexa/Google/Siri + ローカルLLMの選択肢 |
| Frigate NVR | ローカルAIカメラ（Coral対応） |
| ローカルファースト | インターネット不要で基本機能が動作する設計 |
| VLAN分離 | IoTデバイスをメインネットワークから隔離 |
| エネルギー最適化AI | 行動予測による自動温度制御 |

---

## 次に読むべきガイド

- **02-emerging/03-future-hardware.md** — 未来のハードウェア：量子コンピュータ、ニューロモルフィック
- **02-emerging/01-robotics.md** — ロボティクス：Boston Dynamics、Figure
- **01-computing/02-edge-ai.md** — エッジAI：NPU、Coral、Jetson

---

## 参考文献

1. **CSA — Matter 仕様** https://csa-iot.org/all-solutions/matter/
2. **Home Assistant 公式ドキュメント** https://www.home-assistant.io/docs/
3. **Thread Group 公式** https://www.threadgroup.org/
4. **Frigate NVR** https://docs.frigate.video/
