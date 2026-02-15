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

### スマートホームの市場動向と進化

スマートホーム市場は2025年時点で約1,500億ドル規模に達し、年間成長率20%以上で拡大を続けている。この成長を牽引するのは3つの要因である。

1. **Matter規格の普及**: 2022年に最初のバージョンがリリースされて以来、対応デバイスが急速に増加。2025年末時点で1,000以上の認定デバイスが市場に出ている。
2. **エッジAIの進化**: NPU搭載デバイスの低価格化により、クラウドに依存しないローカルAI処理が可能になった。
3. **エネルギー価格の上昇**: 電力コスト削減のため、AIによるエネルギー最適化への需要が増加。

```
【スマートホーム進化の4段階】

Stage 1: リモコン代替期（2014-2018）
├── スマホから家電を操作するだけ
├── Wi-Fi接続の単体デバイス（Philips Hue、TP-Link）
└── 音声アシスタント初期（Echo第1世代: 2014年）

Stage 2: 自動化期（2018-2022）
├── IF-THEN ルールによる自動化（IFTTT、Alexa Routines）
├── シーン管理（「おやすみ」で全照明OFF + ドアロック）
└── 各エコシステムの囲い込み競争

Stage 3: AI統合期（2022-2025）
├── Matter による相互運用性の実現
├── LLM ベースの自然言語制御
├── 行動予測による先回り自動化
└── ローカルAI処理（NPU、Coral）

Stage 4: アンビエントAI期（2025-）
├── 環境が自動的に居住者に適応
├── マルチモーダルセンシング（映像+音声+環境）
├── デジタルツインによるシミュレーション最適化
└── ヘルスケア統合（睡眠、ストレス、活動量）
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

### Thread プロトコルの詳細技術

Thread は IEEE 802.15.4 上に構築された IPv6 ベースのメッシュネットワーキングプロトコルである。従来の Zigbee と同じ物理層を使用するが、ネットワーク層以上が大きく異なる。

```
【Thread プロトコルスタック】

+-------------------------------------------+
| Application Layer (Matter / CoAP)         |
+-------------------------------------------+
| UDP / TCP                                 |
+-------------------------------------------+
| IPv6 (6LoWPAN 圧縮)                       |
+-------------------------------------------+
| Mesh Link Establishment (MLE)             |
| ルーティング: RLOC16 ベース               |
+-------------------------------------------+
| IEEE 802.15.4 MAC                          |
| AES-CCM-128 暗号化                         |
+-------------------------------------------+
| IEEE 802.15.4 PHY                          |
| 2.4 GHz, 250 kbps                         |
+-------------------------------------------+
```

Thread ネットワークのノードには以下の役割がある。

| ノードタイプ | 役割 | 常時稼働 | 中継能力 |
|-------------|------|---------|---------|
| Leader | ネットワーク管理、パーティション統合 | はい | はい |
| Router | パケット中継、子ノード管理 | はい | はい |
| REED (Router-Eligible End Device) | 必要時にRouterに昇格 | はい | 昇格後 |
| End Device (SED/MED) | 末端デバイス、中継なし | MED:はい / SED:間欠 | なし |
| Border Router | Thread ←→ Wi-Fi/Ethernet ブリッジ | はい | はい |

```python
# OpenThread を使った Thread ネットワーク情報取得例
import openthread

def analyze_thread_network(interface="wpan0"):
    """Thread ネットワークの状態を分析"""
    ot = openthread.OpenThread(interface)

    # ネットワーク情報
    network_info = {
        "network_name": ot.get_network_name(),
        "channel": ot.get_channel(),
        "panid": hex(ot.get_panid()),
        "extended_panid": ot.get_extended_panid().hex(),
        "mesh_local_prefix": str(ot.get_mesh_local_prefix()),
    }

    # ノード情報
    node_info = {
        "role": ot.get_role(),  # leader, router, child, detached
        "rloc16": hex(ot.get_rloc16()),
        "router_id": ot.get_router_id(),
        "partition_id": ot.get_partition_id(),
    }

    # 隣接ノード一覧
    neighbors = ot.get_neighbor_table()
    for neighbor in neighbors:
        print(f"  Neighbor RLOC16: {hex(neighbor.rloc16)}")
        print(f"    Role: {'Router' if neighbor.is_router else 'Child'}")
        print(f"    Link Quality: {neighbor.link_quality_in}/3")
        print(f"    Age: {neighbor.age}s")
        print(f"    RSSI: {neighbor.average_rssi} dBm")

    # ルーティングテーブル
    router_table = ot.get_router_table()
    print(f"\nRouter Table ({len(router_table)} entries):")
    for router in router_table:
        print(f"  Router ID {router.router_id}: "
              f"RLOC16={hex(router.rloc16)}, "
              f"Next Hop={router.next_hop}, "
              f"Path Cost={router.path_cost}")

    return network_info, node_info
```

### Zigbee と Thread の移行戦略

既存の Zigbee デバイスを所有するユーザーにとって、Thread への移行は重要な検討事項である。

```
【Zigbee → Thread 移行パス】

パターン A: Bridge 経由（推奨）
+-------------------+     +------------------+     +--------+
| Zigbee デバイス群   | --> | Zigbee-Matter    | --> | Matter |
| (Hue, IKEA等)     |     | Bridge           |     | 統合   |
+-------------------+     | (Hue Bridge v2)  |     +--------+
                          +------------------+

パターン B: 段階的置換
Phase 1: 新規購入は Thread/Matter 対応のみ
Phase 2: 故障・寿命時に Thread デバイスへ交換
Phase 3: 3-5年で完全移行

パターン C: 併用（コスト重視）
+-------------------+     +------------------+
| Zigbee デバイス     | --> | Zigbee2MQTT      | --> Home Assistant
+-------------------+     +------------------+         ↑
+-------------------+     +------------------+         |
| Thread デバイス     | --> | Thread Border    | -------+
+-------------------+     | Router           |
                          +------------------+
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

### Matter のデバイスタイプとクラスタ

Matter はデバイスの機能を「クラスタ」という単位で標準化している。各デバイスタイプは必須クラスタとオプショナルクラスタの組み合わせで定義される。

| デバイスタイプ | Device Type ID | 必須クラスタ | オプショナルクラスタ |
|--------------|---------------|-------------|-------------------|
| On/Off Light | 0x0100 | On/Off, Level Control | Color Control, Scenes |
| Dimmable Light | 0x0101 | On/Off, Level Control | Color Control |
| Color Temperature Light | 0x010C | On/Off, Level, Color Control | Scenes |
| Thermostat | 0x0301 | Thermostat, Fan Control | Humidity Measurement |
| Door Lock | 0x000A | Door Lock | Alarms, Time Sync |
| Window Covering | 0x0202 | Window Covering | Scenes |
| Occupancy Sensor | 0x0107 | Occupancy Sensing | Illuminance |
| Temperature Sensor | 0x0302 | Temperature Measurement | - |
| Humidity Sensor | 0x0307 | Relative Humidity | - |
| Contact Sensor | 0x0015 | Boolean State | - |

### Matter のコミッショニング（ペアリング）プロセス

```
【Matter デバイスのコミッショニングフロー】

1. QRコードスキャン / NFC タッチ
   ┌─────────────┐     ┌──────────────┐
   │ スマートフォン │ --> │ デバイスQRコード │
   │ (Commissioner)│     │ (Commissionee) │
   └─────────────┘     └──────────────┘

2. PASE（パスコード認証セッション確立）
   Commissioner ←──── SPAKE2+ ────→ Commissionee
   ※ QRコードからセットアップペイロードを抽出
   ※ Discriminator + Passcode で認証

3. 証明書チェーン検証
   Commissionee → DAC (Device Attestation Certificate)
                → PAI (Product Attestation Intermediate)
                → PAA (Product Attestation Authority)
   ※ DCL (Distributed Compliance Ledger) で検証

4. NOC（Network Operating Certificate）発行
   Commissioner → Root CA → NOC → Commissionee
   ※ Fabric ID + Node ID を付与

5. ACL（Access Control List）設定
   Adminが他のコントローラにもアクセス権を付与可能
   ※ Multi-Admin: 最大5つのFabricに同時参加可能

6. CASE セッション確立（運用時）
   Controller ←── Sigma1/Sigma2/Sigma3 ──→ Device
   ※ NOC ベースの相互認証
   ※ AES-CCM-128 暗号化通信
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

### コード例: Matter デバイス開発（ESP32 + ESP-Matter）

```cpp
// ESP32 で Matter 対応照明デバイスを開発する例
// ESP-IDF + ESP-Matter SDK

#include <esp_matter.h>
#include <esp_matter_attribute.h>
#include <esp_matter_endpoint.h>
#include <app/server/Server.h>

using namespace esp_matter;
using namespace chip::app::Clusters;

// GPIO 設定
#define LED_GPIO GPIO_NUM_2

// Attribute コールバック
static esp_err_t app_attribute_update_cb(
    attribute::callback_type_t type,
    uint16_t endpoint_id,
    uint32_t cluster_id,
    uint32_t attribute_id,
    esp_matter_attr_val_t *val,
    void *priv_data)
{
    if (type == attribute::PRE_UPDATE) {
        // On/Off クラスタの処理
        if (cluster_id == OnOff::Id) {
            if (attribute_id == OnOff::Attributes::OnOff::Id) {
                gpio_set_level(LED_GPIO, val->val.b ? 1 : 0);
                ESP_LOGI("APP", "LED %s", val->val.b ? "ON" : "OFF");
            }
        }
        // Level Control クラスタの処理
        if (cluster_id == LevelControl::Id) {
            if (attribute_id == LevelControl::Attributes::CurrentLevel::Id) {
                uint8_t level = val->val.u8;
                // PWM で明るさ制御（0-254 → 0-255 duty）
                ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, level);
                ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
                ESP_LOGI("APP", "Brightness: %d/254", level);
            }
        }
    }
    return ESP_OK;
}

extern "C" void app_main()
{
    // GPIO 初期化
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LED_GPIO),
        .mode = GPIO_MODE_OUTPUT,
    };
    gpio_config(&io_conf);

    // Matter ノード作成
    node::config_t node_config;
    node_t *node = node::create(&node_config, app_attribute_update_cb, NULL);

    // Dimmable Light エンドポイント追加
    endpoint::dimmable_light::config_t light_config;
    light_config.on_off.on_off = false;
    light_config.level_control.current_level = 128;
    endpoint_t *endpoint = endpoint::dimmable_light::create(
        node, &light_config, ENDPOINT_FLAG_NONE, NULL
    );

    // Matter スタート
    esp_matter::start(NULL);

    ESP_LOGI("APP", "Matter Dimmable Light Started");
    ESP_LOGI("APP", "QR Code URL: https://project-chip.github.io/...");
}
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

### コード例: Home Assistant カスタムコンポーネント開発

```python
"""Home Assistant カスタムインテグレーション: スマート環境センサー"""
# custom_components/smart_environment/sensor.py

import logging
from datetime import timedelta
from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.const import UnitOfTemperature, PERCENTAGE
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

_LOGGER = logging.getLogger(__name__)
SCAN_INTERVAL = timedelta(seconds=30)

class SmartEnvironmentCoordinator(DataUpdateCoordinator):
    """環境データを統合管理するコーディネーター"""

    def __init__(self, hass, sensors_config):
        super().__init__(
            hass,
            _LOGGER,
            name="Smart Environment",
            update_interval=SCAN_INTERVAL,
        )
        self._sensors = sensors_config
        self._history = []

    async def _async_update_data(self):
        """センサーデータを収集・分析"""
        data = {}

        # 各センサーからデータ収集
        for sensor_id in self._sensors:
            state = self.hass.states.get(sensor_id)
            if state and state.state not in ("unknown", "unavailable"):
                data[sensor_id] = float(state.state)

        # 快適度スコアを計算
        if "sensor.temperature" in data and "sensor.humidity" in data:
            temp = data["sensor.temperature"]
            humidity = data["sensor.humidity"]
            data["comfort_score"] = self._calculate_comfort(temp, humidity)

        # 換気推奨判定
        if "sensor.co2" in data:
            co2 = data["sensor.co2"]
            data["ventilation_needed"] = co2 > 1000

        # 履歴保存（トレンド分析用）
        self._history.append(data)
        if len(self._history) > 120:  # 1時間分
            self._history.pop(0)

        # トレンド分析
        data["temperature_trend"] = self._analyze_trend("sensor.temperature")

        return data

    def _calculate_comfort(self, temp, humidity):
        """PMV簡易モデルによる快適度計算（0-100）"""
        # 不快指数ベースの簡易計算
        discomfort = 0.81 * temp + 0.01 * humidity * (
            0.99 * temp - 14.3
        ) + 46.3
        # 70-75が快適ゾーン
        if 70 <= discomfort <= 75:
            return 100
        elif discomfort < 70:
            return max(0, 100 - (70 - discomfort) * 10)
        else:
            return max(0, 100 - (discomfort - 75) * 10)

    def _analyze_trend(self, sensor_id):
        """直近30分のトレンドを分析"""
        values = [
            h.get(sensor_id) for h in self._history[-60:]
            if h.get(sensor_id) is not None
        ]
        if len(values) < 10:
            return "stable"
        slope = (values[-1] - values[0]) / len(values)
        if slope > 0.05:
            return "rising"
        elif slope < -0.05:
            return "falling"
        return "stable"


class ComfortScoreSensor(CoordinatorEntity, SensorEntity):
    """快適度スコアセンサー"""

    _attr_name = "快適度スコア"
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:emoticon-happy-outline"

    def __init__(self, coordinator):
        super().__init__(coordinator)
        self._attr_unique_id = "smart_env_comfort_score"

    @property
    def native_value(self):
        if self.coordinator.data:
            return self.coordinator.data.get("comfort_score")
        return None

    @property
    def extra_state_attributes(self):
        """追加属性"""
        data = self.coordinator.data or {}
        return {
            "temperature_trend": data.get("temperature_trend", "unknown"),
            "ventilation_needed": data.get("ventilation_needed", False),
        }
```

### Home Assistant ダッシュボード（Lovelace UI）設計

```yaml
# Home Assistant Lovelace Dashboard 設計例
# ui-lovelace.yaml

title: スマートホーム
views:
  - title: ホーム
    path: home
    icon: mdi:home
    cards:
      # 在宅状態カード
      - type: entities
        title: 家族の在宅状況
        entities:
          - entity: person.taro
            secondary_info: last-changed
          - entity: person.hanako
            secondary_info: last-changed
          - entity: binary_sensor.anyone_home
            name: 在宅者あり

      # 環境モニター
      - type: custom:mini-graph-card
        title: 室内環境
        entities:
          - entity: sensor.living_room_temperature
            name: 温度
            color: "#e74c3c"
          - entity: sensor.living_room_humidity
            name: 湿度
            color: "#3498db"
            y_axis: secondary
        hours_to_show: 24
        points_per_hour: 4
        show:
          labels: true
          average: true
          extrema: true

      # 照明コントロール
      - type: custom:light-entity-card
        entity: light.living_room
        shorten_cards: true
        consolidate_entities: true
        child_card: true
        hide_header: false
        effects_list: true

      # エネルギー使用量
      - type: energy-distribution
        title: エネルギー分配
        link_dashboard: true

      # カメラフィード
      - type: picture-glance
        title: 玄関カメラ
        camera_image: camera.front_door
        entities:
          - binary_sensor.front_door_motion
          - binary_sensor.front_door_person
        camera_view: live

  - title: 自動化
    path: automations
    icon: mdi:robot
    cards:
      # 自動化の一覧と状態
      - type: custom:auto-entities
        card:
          type: entities
          title: アクティブな自動化
        filter:
          include:
            - domain: automation
              state: "on"
          exclude:
            - entity_id: automation.system_*
        sort:
          method: last_triggered
          reverse: true

      # 自動化のトリガー履歴
      - type: logbook
        title: 直近の自動化実行
        hours_to_show: 24
        entities:
          - automation.sunset_lights
          - automation.away_mode
          - automation.security_alert
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

### ローカル音声アシスタント構築（Wyoming Protocol）

Home Assistant のローカル音声処理は Wyoming プロトコルを使用して、各音声処理コンポーネントをマイクロサービスとして接続する。

```
【ローカル音声アシスタント アーキテクチャ】

+-------------------+
| マイク入力         |  ESP32-S3-BOX / USB マイク
+--------+----------+
         |
         v
+--------+----------+
| ウェイクワード検出   |  openWakeWord / Porcupine
| "OK ナブ"          |  ← ESP32上で動作（低遅延）
+--------+----------+
         |
         v (Wyoming Protocol)
+--------+----------+
| 音声認識 (STT)     |  faster-whisper / Whisper.cpp
| 音声 → テキスト     |  ← ローカルGPU or CPU
| モデル: large-v3   |  ← 日本語対応
+--------+----------+
         |
         v
+--------+----------+
| インテント処理      |  Home Assistant Conversation Agent
| LLM or ルールベース |  ← Ollama (llama3) / ルールベース
+--------+----------+
         |
         v
+--------+----------+
| 音声合成 (TTS)     |  Piper TTS
| テキスト → 音声     |  ← ローカル、低遅延
| 声質: ja_JP-takumi |  ← 日本語音声モデル
+--------+----------+
         |
         v
+--------+----------+
| スピーカー出力      |  ESP32-S3-BOX / 外部スピーカー
+-------------------+
```

```yaml
# Home Assistant の Wyoming 音声パイプライン設定
# docker-compose.yml

version: '3.8'
services:
  # Whisper STT サーバー
  whisper:
    image: rhasspy/wyoming-whisper:latest
    ports:
      - "10300:10300"
    volumes:
      - whisper-data:/data
    command: >
      --model large-v3
      --language ja
      --device cuda  # GPU使用（CPU: --device cpu）
      --beam-size 5
      --compute-type float16
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Piper TTS サーバー
  piper:
    image: rhasspy/wyoming-piper:latest
    ports:
      - "10200:10200"
    volumes:
      - piper-data:/data
    command: >
      --voice ja_JP-takumi-medium
      --speaker 0
      --length-scale 1.0
      --noise-scale 0.667
      --noise-w 0.8

  # openWakeWord サーバー
  openwakeword:
    image: rhasspy/wyoming-openwakeword:latest
    ports:
      - "10400:10400"
    command: >
      --preload-model ok_nabu
      --threshold 0.5
      --trigger-level 1

volumes:
  whisper-data:
  piper-data:
```

### ESP32-S3 ベースの音声サテライト構築

```yaml
# ESPHome 設定: ESP32-S3-BOX を音声サテライト化
# esphome/voice-satellite.yaml

esphome:
  name: voice-satellite-living
  friendly_name: "リビング音声アシスタント"

esp32:
  board: esp32-s3-box
  framework:
    type: esp-idf

# Wi-Fi設定
wifi:
  ssid: !secret wifi_ssid
  password: !secret wifi_password

# マイクロフォン（I2S入力）
i2s_audio:
  - id: i2s_input
    i2s_lrclk_pin: GPIO41
    i2s_bclk_pin: GPIO42
  - id: i2s_output
    i2s_lrclk_pin: GPIO46
    i2s_bclk_pin: GPIO17

microphone:
  - platform: i2s_audio
    id: mic
    i2s_audio_id: i2s_input
    adc_type: external
    i2s_din_pin: GPIO2
    pdm: false
    channel: left
    bits_per_sample: 32bit
    sample_rate: 16000

# スピーカー（I2S出力）
speaker:
  - platform: i2s_audio
    id: spk
    i2s_audio_id: i2s_output
    dac_type: external
    i2s_dout_pin: GPIO15
    mode: stereo

# 音声アシスタント
voice_assistant:
  id: va
  microphone: mic
  speaker: spk
  noise_suppression_level: 2
  auto_gain: 31dBFS
  volume_multiplier: 2.0
  use_wake_word: true
  on_wake_word_detected:
    - light.turn_on:
        id: led_ring
        effect: "listening"
  on_stt_end:
    - light.turn_on:
        id: led_ring
        effect: "thinking"
  on_tts_start:
    - light.turn_on:
        id: led_ring
        effect: "speaking"
  on_end:
    - light.turn_off:
        id: led_ring
    - wait_until:
        not:
          voice_assistant.is_running:
    - delay: 500ms
    - voice_assistant.start_continuous:

# LED リング（状態表示）
light:
  - platform: esp32_rmt_led_strip
    id: led_ring
    pin: GPIO39
    num_leds: 12
    chipset: SK6812
    rgb_order: GRB
    effects:
      - addressable_rainbow:
          name: "listening"
          speed: 30
      - pulse:
          name: "thinking"
          min_brightness: 30%
          max_brightness: 100%
      - addressable_scan:
          name: "speaking"
          move_interval: 50ms

# 物理ボタン（ミュート）
binary_sensor:
  - platform: gpio
    pin:
      number: GPIO0
      inverted: true
    name: "ミュートボタン"
    on_press:
      - voice_assistant.stop:
      - light.turn_on:
          id: led_ring
          red: 100%
          green: 0%
          blue: 0%
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

### コード例: 高度なエネルギー管理システム（HEMS）

```python
"""家庭用エネルギー管理システム (HEMS) の実装"""
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

@dataclass
class EnergyDevice:
    """エネルギーデバイスの抽象化"""
    name: str
    entity_id: str
    power_watts: float
    priority: int  # 1=最高, 5=最低
    shiftable: bool  # 稼働時間をずらせるか
    min_runtime_minutes: int = 0
    max_power_watts: Optional[float] = None

class HEMSController:
    """
    HEMS（Home Energy Management System）コントローラー

    太陽光発電、蓄電池、電力料金を考慮して
    家電の稼働スケジュールを最適化する
    """

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.devices: list[EnergyDevice] = []
        self.solar_forecast = []
        self.price_schedule = []
        self.battery_soc = 0.0  # State of Charge (%)

    async def update_solar_forecast(self):
        """太陽光発電予測を更新（Solcast API）"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = "https://api.solcast.com.au/rooftop_sites"
            params = {
                "resource_id": self.config["solcast_resource_id"],
                "api_key": self.config["solcast_api_key"],
            }
            async with session.get(
                f"{url}/{params['resource_id']}/forecasts",
                params={"api_key": params["api_key"]}
            ) as resp:
                data = await resp.json()

        self.solar_forecast = [
            {
                "time": entry["period_end"],
                "power_kw": entry["pv_estimate"],
                "power_kw_10": entry["pv_estimate10"],  # 10%ile
                "power_kw_90": entry["pv_estimate90"],  # 90%ile
            }
            for entry in data["forecasts"][:48]  # 24時間分
        ]

    def get_electricity_price(self, hour: int) -> float:
        """時間帯別電力料金を取得（円/kWh）"""
        # オクトパスエナジー等の動的料金プラン想定
        price_table = {
            range(0, 6): 18.0,    # 深夜料金
            range(6, 8): 28.0,    # 朝方
            range(8, 10): 32.0,   # 午前
            range(10, 17): 35.0,  # 日中ピーク
            range(17, 21): 38.0,  # 夕方ピーク
            range(21, 24): 25.0,  # 夜間
        }
        for time_range, price in price_table.items():
            if hour in time_range:
                return price
        return 30.0

    async def optimize_schedule(self):
        """デバイス稼働スケジュールを最適化"""
        schedule = {}
        current_hour = datetime.now().hour

        for device in sorted(self.devices, key=lambda d: d.priority):
            if not device.shiftable:
                # シフト不可デバイスはそのまま
                schedule[device.entity_id] = "always_on"
                continue

            # 最安時間帯を探索
            best_hour = current_hour
            best_cost = float('inf')
            runtime_hours = max(1, device.min_runtime_minutes // 60)

            for start_hour in range(24):
                total_cost = 0
                for h in range(runtime_hours):
                    hour = (start_hour + h) % 24
                    price = self.get_electricity_price(hour)

                    # 太陽光発電がある時間帯はコストを下げる
                    solar_offset = self._get_solar_power(hour)
                    net_price = price * max(
                        0, 1 - solar_offset / device.power_watts
                    )
                    total_cost += net_price * (device.power_watts / 1000)

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_hour = start_hour

            schedule[device.entity_id] = {
                "start_hour": best_hour,
                "duration_hours": runtime_hours,
                "estimated_cost_yen": round(best_cost, 1),
            }

        return schedule

    def _get_solar_power(self, hour: int) -> float:
        """指定時刻の太陽光発電量（W）を取得"""
        for forecast in self.solar_forecast:
            forecast_hour = datetime.fromisoformat(
                forecast["time"]
            ).hour
            if forecast_hour == hour:
                return forecast["power_kw"] * 1000
        return 0

    async def battery_strategy(self):
        """蓄電池の充放電戦略を決定"""
        strategies = []
        for hour in range(24):
            price = self.get_electricity_price(hour)
            solar = self._get_solar_power(hour)

            if solar > 2000 and self.battery_soc < 80:
                # 太陽光余剰で充電
                strategies.append({
                    "hour": hour, "action": "charge",
                    "reason": "太陽光余剰", "power_w": min(solar - 1500, 3000)
                })
            elif price >= 35 and self.battery_soc > 30:
                # 高料金時に放電
                strategies.append({
                    "hour": hour, "action": "discharge",
                    "reason": "ピーク料金回避", "power_w": 2000
                })
            elif price <= 20 and self.battery_soc < 50:
                # 深夜料金で充電
                strategies.append({
                    "hour": hour, "action": "charge",
                    "reason": "深夜料金充電", "power_w": 3000
                })
            else:
                strategies.append({
                    "hour": hour, "action": "standby",
                    "reason": "待機", "power_w": 0
                })

        return strategies

    def daily_report(self, schedule, battery_plan):
        """日次エネルギーレポート生成"""
        total_cost = sum(
            s.get("estimated_cost_yen", 0)
            for s in schedule.values()
            if isinstance(s, dict)
        )
        solar_total = sum(
            self._get_solar_power(h) / 1000 for h in range(24)
        )
        charge_hours = sum(
            1 for s in battery_plan if s["action"] == "charge"
        )
        discharge_hours = sum(
            1 for s in battery_plan if s["action"] == "discharge"
        )

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "estimated_cost_yen": round(total_cost, 0),
            "solar_generation_kwh": round(solar_total, 1),
            "battery_charge_hours": charge_hours,
            "battery_discharge_hours": discharge_hours,
            "self_consumption_rate": round(
                solar_total / max(1, solar_total + total_cost / 30) * 100, 1
            ),
        }
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

### Frigate NVR の高度な設定と最適化

```yaml
# frigate.yml - 高度な設定
# 複数カメラ + カスタムモデル + 通知設定

# グローバル設定
mqtt:
  host: 192.168.1.100
  port: 1883
  user: frigate
  password: "{FRIGATE_MQTT_PASSWORD}"
  topic_prefix: frigate

database:
  path: /media/frigate/frigate.db

# 検出器設定（複数対応）
detectors:
  coral_usb:
    type: edgetpu
    device: usb:0
  # 2つ目のCoral（高負荷環境）
  coral_pcie:
    type: edgetpu
    device: pci:0

# モデル設定
model:
  path: /config/model_cache/yolov8n_320.tflite
  input_tensor: nhwc
  input_pixel_format: rgb
  width: 320
  height: 320
  labelmap_path: /config/labelmap.txt

# 録画設定
record:
  enabled: true
  retain:
    days: 7
    mode: motion  # motion/all
  events:
    retain:
      default: 30
      mode: active_objects
    pre_capture: 5   # イベント前5秒
    post_capture: 10  # イベント後10秒

# スナップショット設定
snapshots:
  enabled: true
  timestamp: true
  bounding_box: true
  crop: true
  quality: 85
  retain:
    default: 30

# 複数カメラ設定
cameras:
  # 玄関カメラ
  front_door:
    ffmpeg:
      inputs:
        - path: rtsp://192.168.1.50:554/h264
          roles: [detect]
        - path: rtsp://192.168.1.50:554/h265_main
          roles: [record]
      output_args:
        record: -f segment -segment_time 60 -segment_format mp4 -reset_timestamps 1 -strftime 1 -c:v copy -c:a aac
    detect:
      width: 1280
      height: 720
      fps: 5
      enabled: true
    motion:
      threshold: 25
      contour_area: 30
      delta_alpha: 0.2
      frame_alpha: 0.2
      improve_contrast: true
    objects:
      track: [person, car, dog, cat, package]
      filters:
        person:
          min_area: 5000
          min_score: 0.6
          threshold: 0.75
        car:
          min_area: 10000
          min_score: 0.5
        package:
          min_area: 2000
          min_score: 0.5
    zones:
      porch:
        coordinates: 100,400,500,400,500,720,100,720
        objects: [person, package]
      driveway:
        coordinates: 500,300,1280,300,1280,720,500,720
        objects: [person, car]
    review:
      alerts:
        required_zones: [porch]
        labels: [person]
      detections:
        labels: [car, dog, cat]

  # 裏庭カメラ
  backyard:
    ffmpeg:
      inputs:
        - path: rtsp://192.168.1.51:554/stream
          roles: [detect, record]
    detect:
      width: 1920
      height: 1080
      fps: 5
    objects:
      track: [person, dog, cat]
      filters:
        person:
          min_area: 3000
          threshold: 0.7
    motion:
      mask:
        # 木の揺れを除外
        - 0,0,200,0,200,300,0,300

  # 室内カメラ（ペットモニター）
  living_room:
    ffmpeg:
      inputs:
        - path: rtsp://192.168.1.52:554/stream
          roles: [detect]
    detect:
      width: 640
      height: 480
      fps: 3  # 室内は低FPSで十分
    objects:
      track: [dog, cat]
      filters:
        dog:
          min_area: 2000
    record:
      enabled: false  # プライバシー考慮で録画なし
    snapshots:
      enabled: true

# 通知設定（Home Assistant連携）
# automation.yaml 側で設定
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

### LLM ベースのスマートホーム制御（Function Calling）

```python
"""LLM Function Calling によるスマートホーム制御"""
import json
from openai import OpenAI

# ローカル Ollama に接続
client = OpenAI(
    base_url="http://192.168.1.100:11434/v1",
    api_key="not-needed",
)

# Home Assistant のサービスを関数として定義
tools = [
    {
        "type": "function",
        "function": {
            "name": "control_light",
            "description": "照明を制御する（点灯/消灯/明るさ/色温度の変更）",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "照明のエンティティID",
                        "enum": [
                            "light.living_room",
                            "light.bedroom",
                            "light.kitchen",
                            "light.bathroom",
                        ],
                    },
                    "action": {
                        "type": "string",
                        "enum": ["turn_on", "turn_off", "toggle"],
                    },
                    "brightness_pct": {
                        "type": "integer",
                        "description": "明るさ（0-100%）",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "color_temp_kelvin": {
                        "type": "integer",
                        "description": "色温度（2000-6500K）",
                        "minimum": 2000,
                        "maximum": 6500,
                    },
                },
                "required": ["entity_id", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_climate",
            "description": "エアコン/暖房を制御する",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "enum": ["climate.living_room", "climate.bedroom"],
                    },
                    "action": {
                        "type": "string",
                        "enum": ["set_temperature", "turn_off", "set_mode"],
                    },
                    "temperature": {
                        "type": "number",
                        "description": "目標温度（摂氏）",
                    },
                    "hvac_mode": {
                        "type": "string",
                        "enum": ["heat", "cool", "auto", "off"],
                    },
                },
                "required": ["entity_id", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sensor_value",
            "description": "センサーの値を取得する",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "センサーのエンティティID",
                        "enum": [
                            "sensor.temperature_living_room",
                            "sensor.humidity_living_room",
                            "sensor.co2_living_room",
                            "sensor.power_consumption",
                        ],
                    },
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_scene",
            "description": "シーン（プリセット）を実行する",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_name": {
                        "type": "string",
                        "enum": [
                            "scene.movie_time",
                            "scene.good_morning",
                            "scene.good_night",
                            "scene.away_mode",
                            "scene.party",
                        ],
                    },
                },
                "required": ["scene_name"],
            },
        },
    },
]

def process_user_command(user_input: str):
    """ユーザーの自然言語コマンドを処理"""
    messages = [
        {
            "role": "system",
            "content": (
                "あなたはスマートホームアシスタントです。"
                "ユーザーの要望を理解し、適切な関数を呼び出してください。"
                "曖昧な要望は確認してください。"
            ),
        },
        {"role": "user", "content": user_input},
    ]

    response = client.chat.completions.create(
        model="llama3.1:8b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if message.tool_calls:
        results = []
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            # Home Assistant API を呼び出し
            result = execute_ha_service(func_name, func_args)
            results.append({
                "function": func_name,
                "args": func_args,
                "result": result,
            })

        return results
    else:
        return {"response": message.content}


def execute_ha_service(func_name: str, args: dict) -> str:
    """Home Assistant のサービスを実行"""
    import requests

    ha_url = "http://192.168.1.100:8123"
    ha_token = "YOUR_LONG_LIVED_TOKEN"
    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }

    if func_name == "control_light":
        service = f"light/{args['action']}"
        data = {"entity_id": args["entity_id"]}
        if "brightness_pct" in args:
            data["brightness_pct"] = args["brightness_pct"]
        if "color_temp_kelvin" in args:
            data["color_temp_kelvin"] = args["color_temp_kelvin"]

    elif func_name == "control_climate":
        if args["action"] == "set_temperature":
            service = "climate/set_temperature"
            data = {
                "entity_id": args["entity_id"],
                "temperature": args.get("temperature", 22),
            }
        elif args["action"] == "set_mode":
            service = "climate/set_hvac_mode"
            data = {
                "entity_id": args["entity_id"],
                "hvac_mode": args.get("hvac_mode", "auto"),
            }
        else:
            service = "climate/turn_off"
            data = {"entity_id": args["entity_id"]}

    elif func_name == "get_sensor_value":
        resp = requests.get(
            f"{ha_url}/api/states/{args['entity_id']}",
            headers=headers,
        )
        state = resp.json()
        return f"{state['attributes'].get('friendly_name')}: {state['state']} {state['attributes'].get('unit_of_measurement', '')}"

    elif func_name == "activate_scene":
        service = "scene/turn_on"
        data = {"entity_id": args["scene_name"]}

    else:
        return f"Unknown function: {func_name}"

    resp = requests.post(
        f"{ha_url}/api/services/{service}",
        headers=headers,
        json=data,
    )
    return f"OK (status: {resp.status_code})"


# 使用例
if __name__ == "__main__":
    commands = [
        "リビングの照明を暖かい色で50%にして",
        "今の室温を教えて",
        "映画モードにして",
        "寝室のエアコンを25度に設定して",
        "外出モードをオンにして",
    ]
    for cmd in commands:
        print(f"\nユーザー: {cmd}")
        result = process_user_command(cmd)
        print(f"結果: {json.dumps(result, ensure_ascii=False, indent=2)}")
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

### IoT VLAN 分離の具体的設定

```
# UniFi / pfSense でのIoT VLAN設定例

【ネットワーク設計】
┌─────────────────────────────────────────────────────┐
│ VLAN 1 (Default): 管理用ネットワーク                   │
│   192.168.1.0/24                                     │
│   デバイス: PC, スマホ, NAS                           │
│   → フルアクセス                                      │
├─────────────────────────────────────────────────────┤
│ VLAN 10: IoTデバイス                                   │
│   192.168.10.0/24                                    │
│   デバイス: 照明, センサー, スマートプラグ              │
│   → インターネット制限、VLAN1へアクセス不可            │
├─────────────────────────────────────────────────────┤
│ VLAN 20: カメラ専用                                    │
│   192.168.20.0/24                                    │
│   デバイス: IPカメラ, NVR                              │
│   → インターネットアクセス完全禁止                     │
│   → NVR (192.168.1.x) のみアクセス可                  │
├─────────────────────────────────────────────────────┤
│ VLAN 30: ゲスト用                                      │
│   192.168.30.0/24                                    │
│   デバイス: ゲストのスマホ                              │
│   → インターネットのみ、LAN内アクセス不可              │
└─────────────────────────────────────────────────────┘
```

```bash
# pfSense ファイアウォールルール（概念）
# /etc/pf.conf

# IoT VLAN → メインLAN: ブロック
block in on $iot_vlan from 192.168.10.0/24 to 192.168.1.0/24

# IoT VLAN → Home Assistant のみ許可
pass in on $iot_vlan from 192.168.10.0/24 to 192.168.1.100 port 8123

# IoT VLAN → MQTT Broker のみ許可
pass in on $iot_vlan from 192.168.10.0/24 to 192.168.1.100 port 1883

# IoT VLAN → DNS のみ許可（Pi-hole/AdGuard）
pass in on $iot_vlan from 192.168.10.0/24 to 192.168.1.53 port 53

# カメラ VLAN → 完全隔離
block in on $camera_vlan from 192.168.20.0/24 to any
pass in on $camera_vlan from 192.168.20.0/24 to 192.168.1.100  # NVRのみ

# ゲスト VLAN → インターネットのみ
block in on $guest_vlan from 192.168.30.0/24 to 192.168.0.0/16
pass in on $guest_vlan from 192.168.30.0/24 to any
```

### DNS フィルタリングによるIoTセキュリティ

```yaml
# AdGuard Home 設定 - IoTデバイス用フィルタリング
# /opt/adguardhome/conf/AdGuardHome.yaml (抜粋)

dns:
  bind_hosts:
    - 192.168.1.53
  port: 53
  upstream_dns:
    - https://dns.cloudflare.com/dns-query  # DoH
    - https://dns.google/dns-query

filtering:
  # IoTデバイスのテレメトリをブロック
  rewrites:
    # 中華IoTデバイスの電話帰り通信をブロック
    - domain: "*.tuya.com"
      answer: "0.0.0.0"
    - domain: "*.tuyaus.com"
      answer: "0.0.0.0"
    # カメラのクラウドアップロードをブロック
    - domain: "*.xiongmaitech.com"
      answer: "0.0.0.0"
    # スマートTVの追跡をブロック
    - domain: "*.samsungacr.com"
      answer: "0.0.0.0"
    - domain: "*.lgtvsdp.com"
      answer: "0.0.0.0"

clients:
  # IoTデバイスグループ
  runtime_sources:
    - name: "IoT Devices"
      ids:
        - "192.168.10.0/24"
      tags:
        - "device_iot"
      use_global_blocked_services: false
      blocked_services:
        - facebook
        - tiktok
      filtering_enabled: true
      parental_enabled: false
      safesearch_enabled: false
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

### アンチパターン3: セキュリティ無視のデバイス導入

```
NG: 安価な中華IoTデバイスを無対策で導入
    → デフォルトパスワード（admin/admin）のまま運用
    → クラウドサーバーへの常時通信（データ漏洩リスク）
    → ファームウェア更新なし（既知の脆弱性放置）
    → フラットなネットワーク上に全デバイスが同居

OK: 段階的なセキュリティ対策
    1. VLAN分離: IoTデバイスを専用ネットワークに隔離
    2. DNSフィルタリング: 不要な外部通信をブロック
    3. ファームウェア: 自動更新が可能なデバイスを選択
    4. 認証: 全デバイスのパスワードを変更、可能なら2FA
    5. 監査: ネットワーク監視で異常な通信を検出
```

### アンチパターン4: 過度な自動化による混乱

```
NG: 条件が複雑すぎる自動化ルールを大量に作成
    → ルール同士が競合（照明をONにする自動化とOFFにする自動化が同時発火）
    → デバッグ困難（なぜ照明が点いたのかわからない）
    → 家族が手動で操作できなくなる

OK: 自動化の設計原則
    1. シンプルに: 1つの自動化に条件は3つ以下
    2. 優先度管理: 手動操作 > 自動化（手動操作後は一定時間自動化を抑制）
    3. フィードバック: 自動化実行時に通知（LED、音、アプリ通知）
    4. キルスイッチ: 全自動化を一括停止できるスイッチを用意
    5. 段階的導入: 1つずつ追加して動作確認
    6. ドキュメント: 自動化の意図と条件をコメントに記述
```

### アンチパターン5: バックアップ戦略の欠如

```
NG: Home Assistant の設定をバックアップしていない
    → SDカード故障で全設定が消失
    → 数十時間かけた自動化ルールが一瞬でゼロに
    → デバイスの再ペアリングが必要

OK: 多層バックアップ戦略
    1. 自動スナップショット: 毎日深夜にHA自動バックアップ
    2. 外部保存: Google Drive / NAS にバックアップを同期
    3. Git管理: YAML設定ファイルをGitリポジトリで管理
    4. SSD化: Raspberry PiのSDカードをSSD/NVMeに換装
    5. HA OS: 専用OS使用でスナップショット復元が容易

設定例（Home Assistant バックアップ自動化）:
```

```yaml
# Home Assistant バックアップ自動化
- alias: "毎日自動バックアップ"
  trigger:
    - platform: time
      at: "03:00:00"
  action:
    - service: backup.create
      data:
        name: "auto_backup_{{ now().strftime('%Y%m%d') }}"
    # Google Drive にアップロード（Google Drive Backup アドオン）
    - delay: "00:05:00"
    - service: hassio.addon_stdin
      data:
        addon: cebe7a76_hassio_google_drive_backup
        input:
          command: "backup"
```

---

## 8. スマートホーム構築実践ガイド

### 予算別構成例

```
【ミニマル構成】予算: 2-3万円
├── Raspberry Pi 5 (4GB): 12,000円
├── Zigbee USBドングル (SONOFF ZBDongle-E): 2,500円
├── スマート照明 x3 (IKEA TRADFRI): 6,000円
├── 温度・湿度センサー x2 (Aqara): 4,000円
└── スマートプラグ x2 (TP-Link): 3,000円

【スタンダード構成】予算: 5-8万円
├── Home Assistant Green: 15,000円
├── Thread Border Router (Apple TV 4K): 22,000円
├── Matter対応照明 x5 (Nanoleaf/Eve): 15,000円
├── スマートロック (SwitchBot Lock Pro): 12,000円
├── 温度・湿度・CO2センサー (Aqara): 8,000円
├── スマートカーテン (SwitchBot): 8,000円
└── スマートプラグ x3: 5,000円

【フル構成】予算: 15-25万円
├── Home Assistant Yellow (PoE): 25,000円
├── UniFi Dream Machine SE: 50,000円
├── Thread/Matter照明システム: 30,000円
├── Frigate NVR + Coral USB: 15,000円
├── IPカメラ x3 (Reolink PoE): 30,000円
├── ESP32-S3-BOX x2 (音声サテライト): 8,000円
├── スマートロック + キーパッド: 20,000円
├── 各種センサー群: 15,000円
├── スマートカーテン x3: 24,000円
└── UPS (停電対策): 15,000円
```

### 段階的導入ロードマップ

```
【Phase 1: 基盤構築（1-2週間）】
Day 1-2: ハードウェア設置
  └── Home Assistant インストール、ネットワーク設定

Day 3-5: 基本デバイス接続
  └── 照明、プラグ、センサーのペアリング

Day 6-7: 基本自動化
  └── 日没照明、外出モード、温度アラート

Day 8-14: 安定性確認
  └── 1週間運用して問題を洗い出し

【Phase 2: 拡張（3-4週目）】
Week 3: セキュリティ強化
  └── VLAN分離、DNSフィルタリング、バックアップ設定

Week 4: AI機能追加
  └── Frigate NVR、音声アシスタント（Wyoming）

【Phase 3: 最適化（2ヶ月目以降）】
Month 2: 高度な自動化
  └── 行動パターン学習、エネルギー最適化

Month 3+: 継続改善
  └── 新デバイス追加、自動化の調整
```

---

## FAQ

### Q1. Matter 対応デバイスと非対応デバイス、どちらを買うべき？

今後の購入は原則 Matter 対応を推奨。Matter 対応デバイスはApple Home、Google Home、Amazon Alexa全てで動作し、ベンダーロックインを避けられる。ただし Zigbee の既存エコシステム（IKEA TRADFRI、Philips Hue）は Bridge経由で Matter に対応するため、既存デバイスの買い替えは不要。

### Q2. Home Assistant の導入コストと難易度は？

Raspberry Pi 4/5（1-2万円）にインストールすれば最低限で始められる。専用ハードウェアの Home Assistant Green（約15,000円）やHome Assistant Yellow もある。初期設定はGUIで行え、基本的な自動化はYAML不要。高度なカスタマイズにはYAMLとPythonの知識が必要。

### Q3. 音声アシスタントのプライバシーは大丈夫か？

Amazon Alexa と Google Assistant はデフォルトで音声をクラウドに送信する。プライバシーが重要なら、1) 録音データの自動削除を設定、2) Apple Siri（オンデバイス処理重視）を使う、3) Home Assistant のローカル音声処理（Wyoming protocol + Whisper + Piper）でクラウドを完全排除する。

### Q4. Thread と Zigbee はどちらを選ぶべきか？

新規購入なら Thread を推奨する。Thread は IPv6 ネイティブであり、Matter の推奨トランスポート層として位置づけられている。ただし現時点では Zigbee のデバイス種類が圧倒的に多い。既存の Zigbee デバイスがあるなら Zigbee2MQTT で Home Assistant に統合しつつ、新規購入は Thread/Matter 対応を選ぶ「併用戦略」が現実的。Thread Border Router は Apple TV 4K、HomePod mini、Google Nest Hub (2nd gen) が対応しており、いずれかを所有していれば追加コストなしで Thread ネットワークを構築できる。

### Q5. スマートホームのデバイスが100台を超えるとどうなる？

デバイス数が増えると以下の問題が発生しやすい。(1) Wi-Fi のみの構成では帯域不足でレスポンスが悪化する。Thread/Zigbee メッシュで負荷を分散すること。(2) Home Assistant の自動化が複雑化し、起動時間が長くなる。YAML を分割し、パッケージ構成（packages ディレクトリ）で管理する。(3) ダッシュボードが煩雑になる。部屋別・機能別のビューを作成し、custom:auto-entities カードで動的にフィルタリングする。(4) mDNS/DNS-SD のブロードキャストが増えるため、ネットワーク機器の処理能力に注意する。エンタープライズグレードのルーター（UniFi Dream Machine 等）を推奨する。

### Q6. 停電時にスマートホームはどうなる？

UPS（無停電電源装置）を Home Assistant サーバーとネットワーク機器に接続することで、停電後も数十分から数時間の稼働が可能。スマートロックは電池駆動のため停電の影響を受けない。照明やエアコンは物理的に停止するが、復電後に自動的に前の状態に復帰するよう自動化を設定できる。蓄電池（テスラ Powerwall 等）があれば、太陽光発電と組み合わせて完全な停電対策が可能。Home Assistant の NUT（Network UPS Tools）統合で UPS の状態を監視し、バッテリー残量が低下したら安全にシャットダウンする自動化も推奨する。

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
| HEMS | 太陽光+蓄電池+動的料金の統合エネルギー管理 |
| Wyoming Protocol | ローカル音声処理のためのマイクロサービス接続規格 |

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
5. **ESPHome 音声アシスタント** https://esphome.io/components/voice_assistant.html
6. **Wyoming Protocol** https://github.com/rhasspy/wyoming
7. **Matter SDK (connectedhomeip)** https://github.com/project-chip/connectedhomeip
8. **OpenThread** https://openthread.io/
