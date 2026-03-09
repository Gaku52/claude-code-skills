# ライブ配信（OBS）

リアルタイムでファンと繋がる。OBS Studio と DDJ-FLX4 でプロ級のライブ配信を実現する完全ガイドです。

## この章で学ぶこと

- OBS Studio 基礎
- DDJ-FLX4 + OBS 設定
- Twitch / YouTube Live 配信
- シーン構成
- オーバーレイ作成
- チャット管理
- トラブルシューティング


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [コンテンツ戦略](./content-strategy.md) の内容を理解していること

---

## OBS Studio とは

**無料の配信ソフト:**

```
OBS Studio:
Open Broadcaster Software

機能:
- ライブ配信
- 録画
- シーン切り替え
- オーバーレイ
- 複数ソース（音声、映像）

対応プラットフォーム:
- Twitch
- YouTube Live
- Facebook Live
- その他（RTMP対応全て）

費用:
完全無料

プロも使用:
業界標準
```

---

## OBS Studio インストール

### ダウンロード

**公式サイトから:**

```
Step 1: ダウンロード

ブラウザで:
https://obsproject.com

[Download OBS Studio]

OS選択:
- Windows
- macOS
- Linux

最新版:
28.x以降推奨

Step 2: インストール

ダウンロードファイル:
実行

Mac:
OBS.dmg をマウント
→ Applications にドラッグ

Windows:
OBS-Studio-XX-Installer.exe
→ ウィザードに従う

Step 3: 初回起動

OBS Studio 起動

Auto-Configuration Wizard:
表示される

[Yes] クリック:
自動設定開始

Use Case:
[Optimize for streaming]

Video Settings:
Base Resolution: 1920×1080
FPS: 30

完了:
基本設定完了
```

---

## DDJ-FLX4 音声設定

### 音声ルーティング

**Rekordbox → OBS:**

```
方法1: PC内蔵オーディオ（簡単）

Rekordbox:
Master Out → PC スピーカー/ヘッドフォン

OBS:
Desktop Audio（PC の音を拾う）

メリット:
設定簡単

デメリット:
- 他の音も拾う（通知音等）
- 音質やや劣化

方法2: 仮想オーディオデバイス（推奨）

Windows:
VB-Audio Virtual Cable（無料）

Mac:
Blackhole（無料）

仕組み:
Rekordbox → 仮想デバイス → OBS

メリット:
- Rekordbox の音だけ
- 音質良い

デメリット:
- 設定やや複雑

推奨:
方法2（仮想オーディオデバイス）
```

### 仮想オーディオデバイス設定（Mac）

**Blackhole インストール:**

```
Step 1: ダウンロード

ブラウザで:
https://existential.audio/blackhole/

[Download Blackhole 2ch]
無料

Step 2: インストール

BlackHole2ch.pkg 実行

ウィザードに従う:
→ インストール完了

Step 3: Audio MIDI 設定

Spotlight 検索:
「Audio MIDI Setup」起動

左下 [+] クリック:
[Create Multi-Output Device]

名前:
「DJ Output」

設定:
☑ Built-in Output（スピーカー）
☑ BlackHole 2ch

Master Device:
Built-in Output

完了:
DJ Output デバイス作成

Step 4: Rekordbox 設定

Rekordbox:
[Preferences] > [Audio]

Audio Output:
「DJ Output」選択

適用:
OK

テスト:
曲を再生
→ スピーカーから音が出る

Step 5: OBS 設定

OBS:
[Settings] > [Audio]

Mic/Auxiliary Audio:
None（または使うマイク）

Desktop Audio:
None

追加オーディオソース:
後で追加（次のセクション）

完了:
OBS から Rekordbox の音を拾う準備完了
```

---

## OBS シーン構成

### 基本シーン作成

**3シーン推奨:**

```
Scene 1: Starting Soon（配信開始前）

ソース:
- 背景画像
  「Starting Soon」テキスト
  開始時間表示

- BGM（音楽ファイル）
  配信開始までのBGM

用途:
配信開始5-10分前から表示

Scene 2: DJ Set（メイン）

ソース:
- カメラ（DDJ-FLX4 + 手元）
  Webカメラ or スマホカメラ

- Rekordbox Audio
  BlackHole からの音声

- オーバーレイ
  DJ名、SNS、曲名（オプション）

用途:
DJプレイ中のメイン画面

Scene 3: BRB（休憩中）

ソース:
- 背景画像
  「Be Right Back」テキスト

- BGM

用途:
トイレ休憩等

実装:

OBS 下部 [Scenes]:

[+] クリック:
「1. Starting Soon」追加

[+] クリック:
「2. DJ Set」追加

[+] クリック:
「3. BRB」追加

切り替え:
クリックで即座に切り替え
```

### ソース追加（DJ Set シーン）

**ステップバイステップ:**

```
Step 1: カメラ追加

Scene:
「2. DJ Set」選択

Sources（ソース）:
[+] クリック

[Video Capture Device]（Mac）
[Video Capture Device]（Windows）

名前:
「Camera」

デバイス選択:
Webカメラ or スマホ（接続済み）

解像度:
1920×1080（可能なら）

配置:
プレビュー画面でドラッグ
フルスクリーン

Step 2: オーディオ追加

Sources:
[+] クリック

[Audio Input Capture]（Mac）
[Audio Input Capture]（Windows）

名前:
「Rekordbox Audio」

Device:
BlackHole 2ch

完了:
Audio Mixer に表示される

レベル確認:
Rekordbox で曲再生
→ メーターが動く

Step 3: オーバーレイ追加（オプション）

Sources:
[+] クリック

[Image]

名前:
「Logo」

ファイル選択:
あなたのロゴ（PNG、透過）

配置:
右下または左上

サイズ調整:
Alt + ドラッグで縮小

Step 4: テキスト追加（オプション）

Sources:
[+] クリック

[Text (GDI+)]（Windows）
[Text (FreeType 2)]（Mac）

名前:
「DJ Name」

テキスト:
「DJ GAKU - LIVE」

フォント:
選択（読みやすく）

カラー:
白 or ブランドカラー

配置:
下部中央

完了:
DJ Set シーン完成
```

---

## 配信設定

### Twitch

**ステップバイステップ:**

```
Step 1: Twitch アカウント

https://www.twitch.tv

[Sign Up]

Username:
djgaku（例）

Email, Password:
設定

Step 2: プロフィール設定

Profile:
クリック

Profile Picture:
アップロード

Banner:
横長画像

Bio:
簡潔に

Step 3: Stream Key 取得

右上アイコン:
[Creator Dashboard]

左メニュー:
[Settings] > [Stream]

Primary Stream Key:
[Copy]

⚠️ 絶対に公開しない

Step 4: OBS 設定

OBS:
[Settings] > [Stream]

Service:
Twitch

Server:
Auto（自動選択）

Stream Key:
ペースト（先ほどコピーした）

[Apply] > [OK]

完了:
Twitch に配信可能
```

### YouTube Live

**ステップバイステップ:**

```
Step 1: YouTube チャンネル

YouTube Studio:
https://studio.youtube.com

[Go Live]

初回:
24時間待つ必要あり
（ライブ配信有効化）

Step 2: Stream Key 取得

[Create] > [Go Live]

[Stream]タブ

Stream Settings:
[Stream key] をコピー

⚠️ 絶対に公開しない

Step 3: OBS 設定

OBS:
[Settings] > [Stream]

Service:
YouTube - RTMPS

Server:
Primary YouTube ingest server

Stream Key:
ペースト

[Apply] > [OK]

完了:
YouTube Live に配信可能
```

---

## 配信開始

### テスト配信

**本番前に必須:**

```
Step 1: プライベート配信

Twitch:
配信タイトル入力
カテゴリ: Music

YouTube:
配信タイトル入力
公開設定: 限定公開

Step 2: OBS で配信開始

OBS:
[Start Streaming] クリック

ステータス:
緑色 → 配信中

Step 3: 確認

Twitch/YouTube:
自分のチャンネル確認

映像:
正しく映っているか

音声:
聞こえるか
レベルは適切か（-6〜0 dB）

遅延:
10-30秒程度

Step 4: 調整

問題あれば:
[Stop Streaming]
→ 設定調整
→ 再度テスト

満足したら:
本番配信へ
```

### 本番配信

**チェックリスト:**

```
30分前:

□ プレイリスト準備
□ Hot Cue 設定
□ カメラ位置確認
□ 照明調整
□ 部屋を片付け

15分前:

□ OBS 起動
□ Rekordbox 起動
□ テスト音声確認
□ Scene「Starting Soon」選択

10分前:

□ [Start Streaming] クリック
□ 配信開始
□ Twitch/YouTube で確認

5分前:

□ チャット挨拶
  「もうすぐ始めます！」

開始:

□ Scene「DJ Set」に切り替え
□ DJプレイ開始
□ チャット見ながら

終了:

□ 「ありがとうございました」
□ Scene「BRB」に切り替え
□ [Stop Streaming]

配信後:

□ アーカイブ確認
□ SNS で感謝投稿
□ 次回予告
```

---

## チャット管理

### エンゲージメント

**視聴者と交流:**

```
配信中:

定期的にチャット確認:
5-10分ごと

読み上げ:
「〇〇さん、こんにちは！」

質問に答える:
「この曲は△△です」

リクエスト:
可能なら対応

謝辞:
フォロー、サブスクに感謝

注意:

DJ に集中:
チャット最優先ではない

バランス:
音楽70% / チャット30%

モデレーター:
友人に依頼（荒らし対策）
```

---

## よくある質問

### Q1: スペックは足りる？

**A:** 中程度のPCで OK

```
最低スペック:

CPU: Intel i5 以上
RAM: 8GB 以上
GPU: 統合GPU で OK

推奨:

CPU: Intel i7 以上
RAM: 16GB
GPU: 専用GPU（GTX 1650等）

DDJ-FLX4 + OBS:
負荷は中程度

最近のPC:
ほぼ問題なし

心配なら:
テスト配信で確認
```

### Q2: どれくらいのビットレート？

**A:** 3000-6000 kbps

```
Twitch:
最大 6000 kbps

YouTube:
最大 51000 kbps（実質6000推奨）

設定:

OBS:
[Settings] > [Output]

Video Bitrate:
3000 kbps（低スペック）
4500 kbps（推奨）
6000 kbps（高スペック）

Audio Bitrate:
160 kbps（音楽配信は高め）

回線速度:
上り10 Mbps 以上推奨

テスト:
speedtest.net で確認
```

### Q3: 顔出しは必要？

**A:** 推奨だが必須ではない

```
顔出しあり:
- エンゲージメント高い
- 親しみやすい

顔出しなし:
- 手元のみ
- 機材メイン

選択:
あなた次第

最初は:
手元のみでも OK
慣れたら顔出し検討
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する
---

## まとめ

### 配信開始チェックリスト

```
準備:
□ OBS Studio インストール
□ Blackhole インストール（Mac）
□ Rekordbox 音声設定
□ Twitch or YouTube アカウント作成
□ Stream Key 取得・設定

OBS設定:
□ シーン作成（3シーン）
□ ソース追加（カメラ、音声、オーバーレイ）
□ ビットレート設定
□ テスト配信

本番:
□ プレイリスト準備
□ カメラ・照明確認
□ 配信開始
□ DJプレイ
□ チャット管理
□ 配信終了
```

### 今日からできること

```
□ OBS Studio ダウンロード・インストール
□ Blackhole インストール（Mac）
□ Twitch or YouTube アカウント作成
□ テストシーン作成
□ 5分間テスト配信
```

---

## OBS Studio 詳細設定ガイド

OBS Studio はデフォルト設定でもある程度の配信が可能だが、DJ配信に最適化するためには各種設定の調整が不可欠である。ここでは OBS の全般設定から出力設定、映像設定までを詳細に解説する。

### 全般設定（General Settings）

**OBS の基本動作を制御する設定:**

```
OBS メニュー:
[Settings] > [General]

基本設定:

Language:
日本語（必要に応じて変更可能）

Theme:
Dark（推奨 - 暗い部屋での配信に最適）
Acri, Rachni なども選択可能

Output:
☑ Show confirmation dialog when starting streams
  配信開始時に確認ダイアログを表示
  誤操作防止に有効

☑ Show confirmation dialog when stopping streams
  配信停止時にも確認
  誤って配信を止めない

☑ Automatically record when streaming
  配信時に自動録画開始
  アーカイブ用にON推奨

Source Alignment Snapping:
☑ Enable
  ソースの位置合わせを簡単に

Snap Sensitivity: 10.0
  スナップ感度（数値が大きいほど吸い付く）

System Tray:
☑ Enable System Tray
  最小化時にシステムトレイに格納

☑ Minimize to System Tray when started
  起動時にトレイに最小化

Projectors:
☑ Make projectors always on top
  プロジェクター（別ウィンドウ）を常に最前面に
```

### 出力設定（Output Settings）詳細

**DJ配信向けの出力品質設定:**

```
OBS メニュー:
[Settings] > [Output]

Output Mode: Advanced（詳細モードに切替）

=== Streaming タブ ===

Encoder（エンコーダー選択）:

GPU搭載の場合:
  NVIDIA → NVENC H.264
  AMD → AMF H.264/AVC
  Apple Silicon → Apple VT H264 Hardware Encoder

GPU非搭載/統合GPUの場合:
  x264（CPU エンコード）

エンコーダー優先順:
1. NVENC（最も軽量）
2. Apple VT Hardware（Mac推奨）
3. AMF（AMD GPU）
4. x264（最終手段 - CPU負荷高い）

Rate Control:
CBR（Constant Bitrate）推奨
→ 一定のビットレートで安定配信

Bitrate:
Twitch向け: 4500-6000 kbps
YouTube向け: 4500-8000 kbps
回線速度に応じて調整

Keyframe Interval:
2（秒）
Twitch の推奨値
YouTube も 2秒で OK

Preset:
NVENC: Quality
x264: veryfast（CPU負荷を抑える）
Apple VT: なし（自動）

Profile:
high（高品質プロファイル）

B-frames:
2（NVENC/x264の場合）

=== Recording タブ ===

Type:
Standard

Recording Path:
任意のフォルダ（例: ~/Movies/DJ_Recordings/）

Recording Format:
mkv（推奨 - クラッシュ耐性あり）
※ 配信後に Remux で mp4 に変換可能

Recording Quality:
High Quality, Medium File Size
またはカスタム設定

Encoder:
配信と同じエンコーダーでOK
または別のエンコーダーを指定可能

=== Audio タブ ===

Track 1:
配信用メインオーディオ

Audio Bitrate: 320 kbps
（DJ配信は音質重視、320推奨）

Sample Rate:
48 kHz（音楽配信推奨）
44.1 kHz でも可

注意:
Rekordbox のサンプルレートと合わせること
不一致だと音声が乱れる場合あり
```

### 映像設定（Video Settings）詳細

**解像度とフレームレートの最適化:**

```
OBS メニュー:
[Settings] > [Video]

Base (Canvas) Resolution:
1920×1080（フルHD）

Output (Scaled) Resolution:

回線・PCスペック別推奨:
高スペック + 高回線 → 1920×1080（配信もフルHD）
中スペック + 中回線 → 1280×720（HD）
低スペック + 低回線 → 854×480（SD）

推奨:
1280×720 から始めて問題なければ 1080 に上げる

Downscale Filter:
Lanczos（36 samples）推奨
→ 最も高品質なスケーリング

Bicubic でも OK（やや軽量）

Common FPS Values:
30 fps（推奨 - DJ配信は映像よりも音声が重要）
60 fps（ゲーム配信向け、DJ配信には不要）

DJ配信でのFPS考察:
30fps で十分な理由:
- DJ の手元動作は激しくない
- 映像より音質優先
- CPU/GPU 負荷を抑えられる
- ビットレートを音声に回せる

60fps が有効な場面:
- VJ映像を同時に流す場合
- カメラ複数台でダイナミックに切り替える場合
```

### 音声設定（Audio Settings）詳細

**DJ配信の音声品質を最大化する設定:**

```
OBS メニュー:
[Settings] > [Audio]

=== General ===

Sample Rate:
48 kHz（推奨）
Rekordbox と統一すること

Channels:
Stereo（ステレオ必須）
DJ の音はステレオが前提

=== Global Audio Devices ===

Desktop Audio:
Disabled（無効化推奨）
→ PC の全ての音を拾うのを防ぐ

Desktop Audio 2:
Disabled

Mic/Auxiliary Audio:
マイク使用時のみ設定
→ 使わない場合は Disabled

Mic/Auxiliary Audio 2:
Disabled

注意:
DJ の音声は Sources から個別に追加する
Global Audio は使わない方が安全

=== Monitoring ===

Advanced Audio Properties:
Audio Mixer 右の歯車アイコン
→ [Advanced Audio Properties]

各ソースごとに設定:

Rekordbox Audio:
  Volume: 0 dB（基本）
  Balance: Center
  Sync Offset: 0 ms
  Audio Monitoring: Monitor Off
  → 配信にだけ音声を送る

Microphone（使用時）:
  Volume: 調整（-6 dB 程度から開始）
  Audio Monitoring: Monitor Off
  Sync Offset: 0 ms（必要に応じて調整）

BGM（Starting Soon 用）:
  Volume: -10 dB（控えめ）
  Audio Monitoring: Monitor and Output
  → 自分でも聞ける
```

### 音声ルーティング詳細（Windows編）

**VB-Audio Virtual Cable の設定:**

```
Step 1: VB-Audio Virtual Cable ダウンロード

ブラウザで:
https://vb-audio.com/Cable/

[Download] クリック
（Donationware - 無料で使用可能）

VBCABLE_Driver_Pack43.zip をダウンロード

Step 2: インストール

ZIP を解凍

VBCABLE_Setup_x64.exe を右クリック
→ 「管理者として実行」

[Install Driver] クリック

完了 → PC再起動

Step 3: サウンド設定確認

Windows:
[設定] > [システム] > [サウンド]

出力デバイスに「CABLE Input (VB-Audio Virtual Cable)」
入力デバイスに「CABLE Output (VB-Audio Virtual Cable)」
が表示されていることを確認

Step 4: Rekordbox 設定

Rekordbox:
[Preferences] > [Audio]

Audio: DDJ-FLX4（コントローラー使用時）

Master Output:
CABLE Input (VB-Audio Virtual Cable)

注意:
DDJ-FLX4 使用時は DDJ-FLX4 の Master Out が優先
PC のスピーカーでモニタリングしたい場合は
別途設定が必要

Step 5: OBS 設定

OBS Sources:
[+] > [Audio Input Capture]

名前: 「Rekordbox Audio」

Device:
CABLE Output (VB-Audio Virtual Cable)

完了:
Rekordbox → Virtual Cable → OBS の
ルーティングが完成

Step 6: 自分でも音を聞く方法

Windowsサウンド設定:
[サウンド] > [録音] タブ

CABLE Output:
右クリック → [プロパティ]

[聴く] タブ:
☑ このデバイスを聴く

再生デバイス:
スピーカー/ヘッドフォン選択

適用 → OK

これで Rekordbox の音が:
1. Virtual Cable → OBS（配信）
2. スピーカー（自分のモニター）
に同時に出力される
```

### 音声フィルター設定

**OBS の音声フィルターでプロ品質に:**

```
OBS Audio Mixer:
Rekordbox Audio ソースの歯車アイコン
→ [Filters]

=== 推奨フィルター ===

1. Compressor（コンプレッサー）

[+] > [Compressor]

Ratio: 4:1
Threshold: -18 dB
Attack: 6 ms
Release: 60 ms
Output Gain: 0 dB
Sidechain: None

効果:
音量の差を圧縮
急な音量変化を防ぐ
配信の音量を安定化

2. Limiter（リミッター）

[+] > [Limiter]

Threshold: -1.0 dB
Release: 60 ms

効果:
音量がピークを超えないよう制限
クリッピング（音割れ）防止

3. Noise Gate（ノイズゲート）- マイク用

マイクソースの歯車 → [Filters]

[+] > [Noise Gate]

Close Threshold: -32 dB
Open Threshold: -26 dB
Attack Time: 25 ms
Hold Time: 200 ms
Release Time: 150 ms

効果:
話していない時のノイズをカット
キーボード音、環境音の除去

4. Noise Suppression（ノイズ抑制）- マイク用

[+] > [Noise Suppression]

Method: RNNoise（AI ベース、推奨）
Suppression Level: -30 dB

効果:
AI がリアルタイムでノイズを除去
エアコン音、ファン音などに有効

=== フィルター適用順序 ===

推奨順（上から下に処理される）:

Rekordbox Audio:
1. Compressor
2. Limiter

Microphone:
1. Noise Suppression
2. Noise Gate
3. Compressor
4. Limiter

注意:
順序が重要
Noise系 → Compressor → Limiter の順で適用
```

### ホットキー設定

**配信中のシーン切り替えをキーボードで:**

```
OBS メニュー:
[Settings] > [Hotkeys]

=== シーン切り替え ===

Switch to Scene「1. Starting Soon」:
F1 キーに設定

Switch to Scene「2. DJ Set」:
F2 キーに設定

Switch to Scene「3. BRB」:
F3 キーに設定

=== 配信制御 ===

Start Streaming:
Ctrl + Shift + S（Windows）
Cmd + Shift + S（Mac）

Stop Streaming:
Ctrl + Shift + E（Windows）
Cmd + Shift + E（Mac）

Start Recording:
Ctrl + Shift + R（Windows）
Cmd + Shift + R（Mac）

Stop Recording:
Ctrl + Shift + T（Windows）
Cmd + Shift + T（Mac）

=== ソース制御 ===

Mute/Unmute Rekordbox Audio:
F5

Mute/Unmute Microphone:
F6

=== トランジション ===

Quick Transition (Fade):
F9

Quick Transition (Cut):
F10

注意:
ホットキーは他のアプリと被らないよう注意
Rekordbox のショートカットとも確認

DJ中にキーを押しやすい配置を考える:
ファンクションキー（F1-F12）が便利
```

### OBS のパフォーマンス最適化

**配信がカクつかないための設定:**

```
=== CPU 負荷軽減 ===

1. エンコーダー選択
GPU エンコード（NVENC/AMF/Apple VT）を使う
x264 は CPU に大きな負荷

2. Output Resolution を下げる
1920×1080 → 1280×720
これだけで大幅に負荷軽減

3. FPS を 30 に
60fps → 30fps で負荷半減
DJ 配信なら 30fps で十分

4. プレビュー無効化
Studio Mode ON の場合
プレビューの解像度を下げる

OBS 右下:
[Stats] で確認

=== GPU 負荷軽減 ===

1. ソース数を減らす
不要なオーバーレイを削除
ブラウザソースは特に重い

2. ゲームキャプチャより画面キャプチャ
必要最小限のキャプチャ

3. フィルター数を減らす
映像フィルターは GPU 負荷が高い

=== メモリ使用量 ===

推奨:
OBS 単体: 300-500 MB
OBS + Rekordbox: 2-4 GB
合計システム: 8 GB 以上必要

確認:
タスクマネージャー（Windows）
アクティビティモニター（Mac）

=== ネットワーク最適化 ===

有線LAN推奨:
Wi-Fi は不安定になりがち

帯域確認:
配信ビットレート × 1.5 以上の上り速度

例:
6000 kbps 配信 → 上り 9 Mbps 以上必要

バッファリング対策:
OBS [Settings] > [Advanced]
Network:
☑ Enable new networking code
☑ Low latency mode（Twitch 向け）
```

---

## OBS オーディオミキサー詳細操作

DJ配信において、オーディオミキサーの操作は配信品質を左右する最も重要な要素の一つである。ここでは Audio Mixer の各機能と最適な設定値について詳しく解説する。

### Audio Mixer パネルの構成

**OBS 下部に表示されるミキサー:**

```
Audio Mixer パネル:

各ソースが横並びで表示される:

[Rekordbox Audio] [Microphone] [BGM] [Desktop Audio]
     ■■■■□□□       ■■□□□□□    ■□□□□□□    （無効）
     -6 dB          -18 dB       -24 dB

各ソースの要素:

1. フェーダー（スライダー）
   上下で音量調整
   dB 表示

2. メーター（レベルメーター）
   緑: 安全な音量（-60〜-20 dB）
   黄: 適正〜やや大きい（-20〜-6 dB）
   赤: 危険、クリッピング（-6〜0 dB）

3. スピーカーアイコン
   クリックでミュート/ミュート解除

4. 歯車アイコン
   詳細設定、フィルター

5. ロックアイコン
   誤操作防止
```

### DJ配信の理想的な音量バランス

**各ソースの推奨レベル:**

```
=== 基本的な考え方 ===

配信の総合音量:
ピーク（最大値）: -3 dB 〜 -1 dB
平均: -12 dB 〜 -6 dB

クリッピング（0 dB超え）は絶対に避ける
→ 音割れして視聴者が離脱する原因

=== 各ソースの設定 ===

Rekordbox Audio（メインDJ音声）:
フェーダー: 0 dB（基本位置）
ピーク目標: -6 dB 〜 -3 dB
Rekordbox 側の Master Level で調整

Microphone（MCマイク）:
フェーダー: -6 dB 〜 -12 dB
DJ音声より小さくならない程度
ただし音楽を邪魔しない

BGM（Starting Soon用）:
フェーダー: -18 dB 〜 -12 dB
小さめで環境音程度

=== 音量チェック方法 ===

1. Rekordbox で音楽再生
2. OBS のメーターを確認
3. 黄色の範囲内が理想
4. 赤に到達しないよう調整
5. テスト配信で実際に確認

Loudness Meter プラグイン:
LUFS 測定が可能
配信の推奨: -14 LUFS（YouTube）/ -16 LUFS（一般）
```

---

## 配信プラットフォーム別 詳細設定

各プラットフォームにはそれぞれ固有の仕様や推奨設定がある。DJ配信で最大限のリーチとクオリティを得るために、プラットフォームごとの最適化を理解しておくことが重要である。

### Twitch 詳細設定

**Twitch DJ配信のための完全ガイド:**

```
=== Twitch DJ カテゴリ設定 ===

カテゴリ選択:
Music → DJ（推奨）
Music & Performing Arts でも可

タグ設定（重要）:
- DJ
- EDM（ジャンルに応じて）
- House Music
- Techno
- Japanese（日本語配信の場合）
- Vinyl DJ（レコード使用の場合）
- Live Music

タグは最大5個まで設定可能
検索されやすいタグを選ぶ

タイトル例:
「[DJ GAKU] Deep House & Techno | Weekend Vibes 🎧」
「【DJ配信】90s Hip Hop Mix | リクエスト受付中」

=== Twitch アフィリエイト条件 ===

条件（全て30日以内に達成）:
1. フォロワー 50人以上
2. 配信日数 7日以上
3. 配信時間 500分以上
4. 平均視聴者数 3人以上

達成すると:
- サブスクリプション収益
- ビッツ（投げ銭）
- エモート設定
- 広告収益

DJ配信での達成コツ:
- 定期的なスケジュール配信
- SNS で告知
- コラボ配信
- 他のDJ配信を Raid する

=== Twitch 固有の OBS 設定 ===

ビットレート:
最大 6000 kbps（アフィリエイト/パートナー以外）
パートナー: 最大 8500 kbps

解像度:
1280×720 @ 60fps または
1920×1080 @ 30fps

Keyframe Interval:
2秒（必須）

Low Latency Mode:
OBS [Settings] > [Advanced]
☑ Low latency mode

Twitch Dashboard:
[Settings] > [Stream]
Latency: Low Latency（推奨）

VOD保存:
[Settings] > [Stream]
☑ Store past broadcasts
14日間保存（アフィリエイト）
60日間保存（パートナー）

=== Twitch DMCA 対策 ===

重要:
Twitch は著作権に厳しい
DMCA ストライクで配信停止リスク

対策:
1. 著作権フリー音源を使用
2. DJ ミックスは「演奏」として配信
3. VOD でのミュートに注意
4. Soundtrack by Twitch は DJ配信には不向き

推奨:
配信中はOK（フェアユース的な運用）
VOD は非公開 or 削除を検討
クリップも DMCA 対象になりうる
```

### YouTube Live 詳細設定

**YouTube Live DJ配信の最適化:**

```
=== YouTube Live の利点 ===

Twitch と比較:
- アーカイブが永続保存
- 検索エンジンに強い（Google）
- 収益化条件がTwitchと異なる
- Super Chat（投げ銭）
- 広告収益

=== YouTube Live 設定 ===

配信タイプ:
1. ストリームキー方式（推奨）
   OBS から直接配信

2. ウェブカメラ方式
   ブラウザから配信
   DJ配信には不向き

公開設定:
- 公開: 誰でも視聴可能
- 限定公開: URL を知っている人のみ
- 非公開: テスト用

カテゴリ:
音楽

タグ:
DJ, Live DJ Set, House Music, Techno, Mix
（半角カンマ区切り）

サムネイル:
カスタムサムネイル推奨
1280×720 以上の画像
DJ 機材の写真 + テキストが効果的

=== YouTube Live のビットレート設定 ===

解像度別推奨ビットレート:

1080p 30fps:
映像: 4500-9000 kbps
音声: 128-256 kbps

720p 30fps:
映像: 2500-6500 kbps
音声: 128-256 kbps

720p 60fps:
映像: 3500-9000 kbps
音声: 128-256 kbps

DJ配信推奨:
720p 30fps / 4500 kbps / Audio 256 kbps
安定性と音質のバランスが良い

=== YouTube Live の遅延設定 ===

Ultra Low Latency:
遅延 約2-4秒
チャットとの連携が快適
ただし一部機能制限あり

Low Latency:
遅延 約5-10秒
推奨設定

Normal Latency:
遅延 約15-30秒
最も安定

DJ配信推奨:
Low Latency
チャットとの対話もスムーズ

=== YouTube ライブ配信の収益化条件 ===

YouTube パートナープログラム:
- チャンネル登録者 1000人以上
- 過去12ヶ月の総再生時間 4000時間以上
  または
- 過去90日のShorts再生回数 1000万回以上

Super Chat / Super Stickers:
- ライブ配信中の投げ銭機能
- 収益化承認後に利用可能

メンバーシップ:
- 月額課金型のファンコミュニティ
- チャンネル登録者 1000人以上で利用可能
```

### Facebook Live / Instagram Live

**その他プラットフォームの設定:**

```
=== Facebook Live ===

OBS 設定:
[Settings] > [Stream]

Service: Facebook Live

Stream Key:
Facebook Creator Studio から取得

ビットレート:
最大 4000 kbps
推奨: 2500-4000 kbps

解像度:
1280×720（推奨）

特徴:
- Facebook ページやグループで配信可能
- リアクション機能
- コメントが活発
- アーカイブは Facebook に保存

DJ配信での注意:
- 著作権検知が自動で働く
- ミュートされる場合あり
- オリジナル音源推奨

=== Instagram Live ===

注意:
OBS からの直接配信は不可
スマホアプリからのみ

DJ配信方法:
1. スマホをDJ機材に向ける
2. Instagram アプリで配信開始
3. OBS は使えない

別の方法:
Yellow Duck 等のサードパーティツール
Instagram Live に OBS から配信可能にする
ただし利用規約違反のリスクあり

推奨:
Instagram はクリップや告知に使い
メイン配信は Twitch/YouTube で行う

=== Kick ===

新興プラットフォーム:
DJ配信に寛容な傾向

OBS 設定:
[Settings] > [Stream]
Service: Custom
Server: kick.com のRTMP URL
Stream Key: Kick Dashboard から取得

ビットレート:
最大 8000 kbps

特徴:
- Twitch より DMCA に寛容
- 成長中のプラットフォーム
- 70/30 の収益分配（配信者有利）
```

---

## カメラと照明のセットアップ

DJ配信の映像クオリティは視聴者の第一印象を決定する。適切なカメラ配置と照明は、プロフェッショナルな印象を与える最も効果的な方法である。

### カメラの選択と配置

**DJ配信に適したカメラ:**

```
=== カメラの種類 ===

1. Webカメラ（入門）

推奨機種:
- Logicool C920/C922（定番、1080p）
- Logicool StreamCam（高画質）
- Razer Kiyo（ライト付き）

価格帯: 5,000-15,000円
画質: 1080p 30fps
設置: モニター上部やクリップマウント

メリット:
- 安価で始められる
- USB 接続で簡単
- 自動フォーカス

デメリット:
- 暗所に弱い
- 画角が限られる

2. スマホカメラ（コスパ最高）

接続方法:
- Camo（Mac/Windows、高品質）
- DroidCam（Android）
- EpocCam（iPhone）
- OBS Camera（OBS プラグイン）

設定:
スマホにアプリインストール
→ USB または Wi-Fi で接続
→ OBS で Video Capture Device として認識

メリット:
- 手持ちのスマホで OK
- 高画質（スマホカメラは進化している）
- 無料アプリあり

デメリット:
- バッテリー消耗が早い
- 充電しながら推奨

3. デジタルカメラ / ミラーレス（上級）

推奨機種:
- Sony α6400（AF 優秀）
- Canon EOS R50
- Panasonic GH6

接続:
HDMI キャプチャボード経由
- Elgato Cam Link 4K
- AVerMedia Live Gamer MINI

メリット:
- 圧倒的に高画質
- ボケ味が出る
- 暗所に強い

デメリット:
- 高価（カメラ + キャプチャボード）
- 設定が複雑

=== カメラ配置パターン ===

パターン1: 正面アングル（初心者推奨）
カメラ: モニター上部
撮影対象: 顔 + DDJ-FLX4 手元
視聴者: DJ の表情と操作が両方見える

パターン2: 手元メイン（顔出しなし）
カメラ: DDJ-FLX4 の真上
撮影対象: コントローラー操作のみ
マウント: アームスタンドやゴリラポッド

パターン3: 複数カメラ（上級）
カメラ1: 正面（顔）
カメラ2: 手元（俯瞰）
OBS で Scene 内にカメラ2つ配置
Picture in Picture（PinP）構成

PinP 設定:
メイン画面: カメラ1（正面）
小窓（右下）: カメラ2（手元）
小窓サイズ: 画面の 25-30%
```

### 照明の基本

**配信映像を劇的に改善する照明テクニック:**

```
=== 照明が重要な理由 ===

高価なカメラでも照明が悪いと:
- 暗くてノイズだらけ
- 顔が見えない
- プロ感がゼロ

安いカメラでも照明が良いと:
- 明るくクリア
- 色が正確
- プロフェッショナルな印象

結論:
カメラより照明に投資すべき

=== 照明機材 ===

1. リングライト（入門、推奨）

推奨サイズ: 10-18インチ
価格帯: 3,000-10,000円
設置: カメラの背後、顔の正面

メリット:
- 均一な光
- 瞳にリングのキャッチライト
- 角度調整可能

設定:
色温度: 5000K-5500K（昼白色）
明るさ: 70-80%（まぶしくない程度）

2. LEDパネルライト（中級）

推奨: Elgato Key Light / Key Light Air
価格帯: 10,000-20,000円

設置: 45度の角度で顔を照らす
左右に2灯が理想

3. RGB LEDストリップ（雰囲気作り）

設置: 壁面、デスク裏、天井
色: ブランドカラーに合わせる

効果:
- 背景に色をつける
- DJ らしい雰囲気
- 視聴者の没入感向上

推奨:
Philips Hue / Govee LEDストリップ

4. スマートライト

Philips Hue:
スマホから色と明るさを制御
配信中にリアルタイムで変更可能

Nanoleaf:
壁に貼るパネル型 LED
映像映えする背景に

=== 三点照明の基本 ===

1. キーライト（メイン照明）
位置: 顔の斜め前方 45度
強さ: 最も明るい

2. フィルライト（補助照明）
位置: キーライトの反対側
強さ: キーライトの 50-70%
影を和らげる

3. バックライト（背景照明）
位置: 被写体の後方
強さ: 適宜
背景と被写体を分離する

DJ配信では:
キーライト: リングライト or LEDパネル（正面）
バックライト: RGB LEDストリップ（壁面）
フィルライト: なくても可（リングライトで十分）
```

---

## オーバーレイ設計の詳細

OBS のオーバーレイは配信画面の見た目を大きく左右する要素である。プロフェッショナルなオーバーレイは視聴者の滞在時間を延ばし、チャンネルのブランディングに貢献する。

### オーバーレイの種類と作成方法

**DJ配信向けオーバーレイ一覧:**

```
=== オーバーレイの構成要素 ===

1. フレーム / ボーダー
画面の縁を装飾するフレーム
カメラ映像の周りを囲む

2. ローワーサード（Lower Third）
画面下部 1/3 のバー
DJ名、楽曲名、SNS 情報を表示

3. ロゴ / ウォーターマーク
DJ のロゴを隅に配置
透過 PNG で半透明に

4. チャットボックス
配信画面内にチャットを表示
視聴者の発言がリアルタイムで表示

5. アラート表示エリア
フォロー、サブスク等の通知
画面上部や中央にポップアップ

6. 曲名表示（Now Playing）
現在再生中の曲情報
自動更新が理想

=== 作成ツール ===

無料:
- Canva（テンプレート豊富）
- GIMP（Photoshop 代替）
- Figma（デザインツール）
- StreamElements（Web ベース）
- StreamLabs（テーマ付き）

有料:
- Adobe Photoshop
- Adobe After Effects（アニメーション付き）
- Nerd or Die（テンプレート販売）
- Own3d.pro（テンプレート販売）

=== Canva でオーバーレイ作成 ===

Step 1: Canva にアクセス
https://www.canva.com

Step 2: カスタムサイズ
1920 × 1080 px（フルHD）

Step 3: 背景を透明に
背景色: 透明（ダウンロード時に PNG で透過保存）

Step 4: デザイン要素を配置

ローワーサード:
- 長方形を画面下部に配置
- 不透明度: 60-80%
- カラー: ブランドカラー
- テキスト: DJ名、SNS アカウント

ロゴ:
- 右下に配置
- 不透明度: 50-70%
- サイズ: 小さめ（邪魔にならない）

Step 5: ダウンロード
ファイル形式: PNG（透過背景）

Step 6: OBS に追加
Sources > [+] > [Image]
作成した PNG を選択
```

### Now Playing（曲名表示）の設定

**再生中の曲名を自動表示する方法:**

```
=== 方法1: Rekordbox + テキストファイル連携 ===

Rekordbox の Now Playing 機能:
[Preferences] > [Advanced]

Now Playing:
☑ Enable

出力ファイル:
テキストファイルのパスを指定
例: ~/Documents/now_playing.txt

OBS 設定:
Sources > [+] > [Text (GDI+/FreeType 2)]

名前: 「Now Playing」

テキストソース設定:
☑ Read from file
ファイルパス: ~/Documents/now_playing.txt

フォント: 読みやすいもの（Noto Sans JP 等）
サイズ: 24-36pt
カラー: 白（黒い背景上）

自動更新:
OBS がファイルの変更を自動検知
曲が変わると自動でテキスト更新

=== 方法2: OBS WebSocket + カスタムスクリプト ===

上級者向け:
Python/Node.js スクリプトで
Rekordbox の情報を取得し OBS に送信

メリット:
- カスタマイズ自由
- アートワーク表示も可能
- アニメーション付き

デメリット:
- プログラミング知識必要
- セットアップが複雑

=== 方法3: SNIP（Windows のみ）===

SNIP:
音楽プレーヤーの再生情報を取得する無料ツール

対応:
Spotify, iTunes, foobar2000 等
（Rekordbox は非対応の場合あり）

OBS 連携:
テキストファイル出力 → OBS の Text ソースで読込
```

### チャット連携とアラート設定

**StreamElements / StreamLabs を使った連携:**

```
=== StreamElements 設定（推奨）===

Step 1: アカウント作成
https://streamelements.com
Twitch or YouTube アカウントでログイン

Step 2: オーバーレイ作成
[My Overlays] > [New Overlay]

解像度: 1920×1080

Widget 追加:

1. Alert Box（アラート）
フォロー、サブスク、レイド通知
サウンド: カスタム SE（DJ向けの音）
アニメーション: フェードイン
表示時間: 5秒
位置: 画面上部中央

2. Chat Widget（チャット表示）
チャットを画面内に表示
フォントサイズ: 18-24px
背景: 半透明黒
位置: 右側

3. Goal Widget（目標表示）
フォロワー目標
サブスク目標

Step 3: OBS に追加
StreamElements Overlay URL をコピー

OBS Sources:
[+] > [Browser]

名前: 「StreamElements Overlay」

URL: コピーした URL をペースト
Width: 1920
Height: 1080

完了:
アラートが OBS に表示される

=== StreamLabs 設定 ===

同様の手順:
https://streamlabs.com

Alert Box URL をコピー
OBS > Browser Source に追加

StreamLabs の利点:
- テーマが豊富
- 設定が簡単
- モバイルアプリあり

=== チャットボット設定 ===

Nightbot（推奨）:
https://nightbot.tv

設定:
Twitch/YouTube でログイン
[Commands] で自動応答設定

DJ 配信向けコマンド例:

!song
応答: 「現在の曲は OBS の Now Playing をチェック！」

!request
応答: 「リクエストはチャットに曲名を書いてね！」

!socials
応答: 「Twitter: @djgaku / Instagram: @djgaku」

!schedule
応答: 「毎週金曜 21:00-23:00 配信中！」

!gear
応答: 「DDJ-FLX4 + Rekordbox を使っています！」

タイマーメッセージ:
15分ごとに自動投稿
「フォローありがとう！次回配信もお楽しみに」

Spam フィルター:
☑ Spam Protection
☑ Caps Protection
☑ Link Protection
☑ Excessive Emotes
```

---

## ビットレート最適化ガイド

配信のビットレートは映像品質とネットワーク安定性のバランスを決定する重要な要素である。DJ配信では映像よりも音声品質が重視されるため、一般的なゲーム配信とは異なるビットレート配分が推奨される。

### ビットレートの基礎知識

**ビットレートとは何か:**

```
=== ビットレートの定義 ===

ビットレート:
1秒あたりに転送されるデータ量
単位: kbps（キロビット毎秒）または Mbps（メガビット毎秒）

映像ビットレート:
映像の品質を決定
高い → 高画質だが回線に負担
低い → 低画質だが安定

音声ビットレート:
音声の品質を決定
DJ配信では特に重要

総ビットレート:
映像ビットレート + 音声ビットレート
= 必要な上り回線速度

=== DJ配信の推奨ビットレート ===

プラン1: 低帯域（上り5-10 Mbps）
映像: 2500 kbps
音声: 192 kbps
解像度: 720p 30fps
合計: 約 2.7 Mbps

プラン2: 中帯域（上り10-20 Mbps）
映像: 4500 kbps
音声: 256 kbps
解像度: 720p 30fps
合計: 約 4.8 Mbps
★ 推奨

プラン3: 高帯域（上り20 Mbps+）
映像: 6000 kbps
音声: 320 kbps
解像度: 1080p 30fps
合計: 約 6.3 Mbps

=== 音声ビットレートの重要性 ===

DJ配信での音声品質:
128 kbps: 最低限（非推奨）
160 kbps: 標準的なゲーム配信
192 kbps: DJ配信の最低ライン
256 kbps: 推奨
320 kbps: 最高品質

比較:
ゲーム配信: 音声 128-160 kbps で十分
DJ配信: 音声 256-320 kbps 推奨
→ 音楽が主役だから

設定方法:
OBS [Settings] > [Output] > [Audio]タブ
Audio Bitrate: 256 または 320

注意:
Twitch は Audio Bitrate 最大 320 kbps
YouTube は制限なし（ただし 320 で十分）
```

### 回線速度テストと安定性確認

**配信前の回線チェック:**

```
=== 回線速度テスト ===

テストサイト:
1. https://speedtest.net
2. https://fast.com
3. OBS 内蔵テスト（Auto-Configuration）

確認すべき値:
上り速度（Upload Speed）:
配信ビットレートの 1.5-2倍が必要

例:
4500 kbps 配信 → 上り 7-9 Mbps 以上
6000 kbps 配信 → 上り 9-12 Mbps 以上

Ping:
50ms 以下が理想
100ms 超えると遅延やドロップフレームの原因

Jitter:
低いほど良い
10ms 以下推奨

=== 安定性テスト ===

方法:
1. OBS でテスト配信開始
2. 20-30分間放置
3. OBS 右下の Stats を確認

確認項目:

Dropped Frames:
0% が理想
1% 以下: 許容範囲
5% 以上: ビットレート下げるべき

Encoding Overload:
表示されない: OK
表示される: CPU/GPU 過負荷

Bitrate:
安定しているか
大きく変動する → 回線不安定

=== 回線が不安定な場合の対策 ===

1. 有線LAN に切り替え
Wi-Fi → 有線 で劇的に安定

2. ビットレートを下げる
6000 → 4500 → 3000 kbps

3. 解像度を下げる
1080p → 720p → 480p

4. 不要な通信を止める
配信PC でのダウンロード停止
他のデバイスの通信を制限

5. VPN を使用しない
VPN は遅延と不安定の原因

6. ISP に相談
配信用に帯域確保
夜間の速度低下が酷い場合
```

---

## トラブルシューティング完全ガイド

DJ配信中に発生しがちな問題とその解決策を網羅的にまとめる。ライブ配信はリアルタイムであるため、迅速な対応が視聴者の離脱を防ぐ鍵となる。

### 音声トラブル

**最も多い問題とその対処法:**

```
=== 問題1: OBS に音が入らない ===

症状:
Audio Mixer のメーターが動かない
配信に音声が乗らない

原因と対策:

確認1: ソースの設定
OBS Sources で Audio Input Capture の設定を確認
Device: BlackHole 2ch（Mac）/ CABLE Output（Windows）
が正しく選択されているか

確認2: Rekordbox の出力先
[Preferences] > [Audio]
Output が DJ Output（Mac）/ CABLE Input（Windows）になっているか

確認3: ミュートされていないか
Audio Mixer でスピーカーアイコンを確認
赤い × が付いていたらクリックで解除

確認4: 音量フェーダー
フェーダーが一番下（-inf）になっていないか
0 dB 付近に設定

確認5: Advanced Audio Properties
歯車 > Advanced Audio Properties
Audio Monitoring が「Monitor Only (mute output)」になっていないか
「Monitor Off」に設定

確認6: デバイスが認識されているか
Mac: Audio MIDI Setup で BlackHole が表示されているか
Windows: サウンド設定で Virtual Cable が表示されているか

=== 問題2: 音が二重に聞こえる（エコー） ===

症状:
同じ音が重なって聞こえる
反響のような音

原因:
Desktop Audio と Audio Input Capture の両方が有効

対策:
1. Desktop Audio を Disabled にする
   OBS [Settings] > [Audio] > Desktop Audio: Disabled
2. Audio Input Capture のみ使用
3. Advanced Audio Properties で不要なソースをミュート

=== 問題3: 音声が途切れる / ノイズ ===

症状:
プツプツとした音
音声が一瞬途切れる

原因:
サンプルレートの不一致 or CPU 過負荷

対策:
1. サンプルレート統一
   Rekordbox: 48 kHz
   OBS: 48 kHz
   OS のサウンド設定: 48 kHz
   全て同じ値にする

2. バッファサイズ調整
   Rekordbox [Preferences] > [Audio]
   Buffer Size: 512 以上に設定

3. CPU 負荷軽減
   不要なアプリを終了
   エンコーダーを GPU に変更

=== 問題4: 音声と映像がずれる（リップシンク） ===

症状:
手の動きと音がずれている

対策:
OBS Advanced Audio Properties
Sync Offset を調整

手順:
1. テスト配信で確認
2. 音が映像より早い → プラスの値（50-100 ms）
3. 音が映像より遅い → マイナスの値（-50〜-100 ms）
4. 10ms 刻みで微調整
```

### 映像トラブル

**映像に関する問題の解決:**

```
=== 問題1: 映像がカクつく（ドロップフレーム） ===

症状:
配信映像がカクカクする
OBS Stats に Dropped Frames 表示

原因と対策:

ネットワーク原因の場合:
Stats > Dropped Frames (Network)
1. ビットレートを下げる（6000→4500→3000）
2. 有線LANに切り替え
3. 他のネットワーク使用を停止

エンコード原因の場合:
Stats > Frames missed due to rendering lag
1. エンコーダーを GPU に変更
2. Output Resolution を下げる（1080p→720p）
3. FPS を 30 に下げる
4. x264 の Preset を superfast に変更
5. 不要なソースやフィルターを削除

=== 問題2: カメラが認識されない ===

症状:
Video Capture Device に何も表示されない

対策:
1. USB を差し直す
2. 別の USB ポートに接続
3. OBS を再起動
4. カメラのドライバーを更新
5. 他のアプリがカメラを使用していないか確認
   （Zoom、Teams 等を終了）
6. Mac: システム環境設定 > セキュリティ > カメラ で OBS を許可

=== 問題3: 画面が真っ暗 ===

症状:
Video Capture Device が黒い画面

対策:
1. カメラのレンズキャップを外す（意外と多い）
2. ソースのプロパティを再設定
3. 解像度を変更して元に戻す
4. OBS を管理者権限で実行（Windows）
5. ソースを削除して再追加

=== 問題4: 画面がチラつく / ティアリング ===

対策:
1. OBS [Settings] > [Video]
   FPS を整数値に（30 or 60）
2. GPU ドライバーを更新
3. Display Capture の場合
   → Window Capture に変更
```

### 配信接続トラブル

**配信が開始できない / 途中で切れる:**

```
=== 問題1: 配信が開始できない ===

症状:
[Start Streaming] 押しても接続失敗

対策:
1. Stream Key が正しいか確認
   OBS [Settings] > [Stream] > Stream Key
   Twitch/YouTube Dashboard から再コピー

2. サーバー設定
   Twitch: Auto を試す / 東京サーバーを手動選択
   YouTube: Primary ingest server

3. ファイアウォール
   OBS の通信を許可
   Windows: ファイアウォール設定で OBS を許可
   Mac: セキュリティ設定でOBS を許可

4. ポート開放
   RTMP: ポート 1935
   RTMPS: ポート 443

=== 問題2: 配信中に切断される ===

症状:
配信中に突然切断
OBS ステータスが赤に

対策:
1. 有線LAN に切り替え
2. ビットレートを下げる
3. OBS [Settings] > [Advanced]
   ☑ Automatically reconnect
   Retry Delay: 10秒
   Maximum Retries: 20

4. ISP の上り帯域を確認
   ピーク時間帯は速度低下する場合あり

5. ルーターの再起動
   長時間使用でメモリリーク等

=== 問題3: OBS がクラッシュする ===

対策:
1. OBS を最新版にアップデート
2. プラグインを無効化して確認
3. ログファイルを確認
   [Help] > [Log Files] > [Upload Current Log File]
4. ブラウザソースが原因の場合が多い
   → ブラウザソースを一つずつ無効化して特定
5. GPU ドライバーを更新
```

---

## マルチプラットフォーム同時配信

一つの OBS 配信を複数のプラットフォームに同時配信する方法を解説する。Twitch と YouTube Live に同時に配信することで、より広い視聴者にリーチできる。

### 同時配信の方法

**3つのアプローチ:**

```
=== 方法1: Restream.io（推奨）===

Restream:
https://restream.io

仕組み:
OBS → Restream サーバー → 各プラットフォームに分配

設定手順:

Step 1: Restream アカウント作成
無料プランあり（2プラットフォームまで）

Step 2: プラットフォーム接続
[Channels] > [Add Channel]
- Twitch 接続
- YouTube 接続
- Facebook 接続（必要に応じて）

Step 3: OBS 設定
[Settings] > [Stream]
Service: Restream.io
Stream Key: Restream Dashboard からコピー

Step 4: 配信開始
OBS で [Start Streaming]
→ Restream が自動的に各プラットフォームに配信

無料プランの制限:
- 2プラットフォームまで
- Restream のロゴが入る場合あり
- チャット統合は有料

有料プラン（Professional）:
月額 $16
- 無制限プラットフォーム
- ロゴなし
- チャット統合
- 録画機能

=== 方法2: OBS Multiple RTMP Output プラグイン ===

プラグイン:
obs-multi-rtmp

インストール:
GitHub からダウンロード
OBS プラグインフォルダにコピー

設定:
OBS 右側 [Docks] > [Multi-RTMP]

[Add New Target]:
1. Twitch
   RTMP URL: rtmp://live-tyo.twitch.tv/app
   Stream Key: Twitch のキー

2. YouTube
   RTMP URL: rtmp://a.rtmp.youtube.com/live2
   Stream Key: YouTube のキー

配信開始:
各ターゲットの [Start] を個別にクリック

メリット:
- 完全無料
- サードパーティサーバー不要
- 遅延が少ない

デメリット:
- 上り回線の帯域が2倍必要
  （各プラットフォームに個別に送信するため）
- PC 負荷が増加

回線要件:
1プラットフォーム 4500 kbps の場合
2プラットフォーム同時: 上り 14 Mbps 以上必要

=== 方法3: nginx + RTMP（上級者向け）===

自前のRTMPサーバーを立てる方法
技術的知識が必要

メリット:
- 完全制御可能
- 無料

デメリット:
- サーバー構築の知識必要
- メンテナンスが必要

DJ 配信では方法1（Restream）を推奨
```

### マルチプラットフォーム チャット管理

**複数プラットフォームのチャットを一元管理:**

```
=== Restream Chat ===

Restream の機能:
全プラットフォームのチャットを統合

設定:
Restream Dashboard > [Chat]
OBS にブラウザソースで追加可能

表示:
[Twitch] user1: こんにちは！
[YouTube] user2: 最高のミックス！

=== チャット管理のコツ ===

DJ配信中のチャット優先度:

1. 自分の名前が呼ばれたとき
2. リクエスト
3. 質問（機材、曲名等）
4. 一般的な挨拶
5. スパム（無視/BAN）

各プラットフォーム均等に:
Twitch だけ反応して YouTube を無視しない
交互に確認するのが理想

モデレーターを配置:
各プラットフォームに1人ずつ
荒らし対策、チャット盛り上げ
```

---

## DMCA 対策と著作権ガイド

DJ配信における著作権問題は避けて通れない重要な課題である。適切な理解と対策により、安心して配信活動を続けることができる。

### 著作権の基本

**DJ配信と著作権の関係:**

```
=== DJ配信の法的位置づけ ===

ライブ配信での音楽使用:
各国の著作権法によって扱いが異なる

日本の場合:
JASRACが管理する楽曲の配信:
- Twitch: JASRACと包括契約あり（2020年〜）
- YouTube: JASRACと包括契約あり
- ただしレコード原盤権は別問題

配信での注意点:
1. 作曲者の権利（演奏権）→ JASRAC包括契約でカバー
2. 原盤権（レコード会社の権利）→ 個別許諾が必要な場合あり
3. 海外楽曲は各国の権利団体による

=== Twitch での著作権リスク ===

自動検知システム:
Audible Magic による音声認識
VOD（アーカイブ）が自動ミュートされる場合あり

DMCA ストライク:
3回 → アカウント永久BAN

対策:
1. VOD を自動削除に設定
   Creator Dashboard > [Settings] > [Stream]
   Store past broadcasts: OFF

2. クリップ削除
   著作権のある音楽が含まれるクリップを削除

3. 配信中のリスク:
   ライブ配信自体での DMCA は少ないが
   完全に安全ではない

=== YouTube での著作権リスク ===

Content ID システム:
自動で楽曲を検知
アーカイブに広告が付く場合あり
収益は権利者に

著作権侵害の警告:
3回 → チャンネル削除

ただし Content ID 一致 ≠ 著作権侵害警告
多くの場合は広告付与のみ

=== 安全に配信するための選択肢 ===

選択肢1: 著作権フリー音源のみ使用
完全に安全
ただし DJ としてのパフォーマンスが限定的

選択肢2: 許諾済み音源を使用
レーベルから配信許可を得る
一部のレーベルは DJ 配信を歓迎

選択肢3: リスクを理解して通常の楽曲を使用
多くの DJ がこの方法
VOD 管理を徹底する
ストライクが来たら即対応

選択肢4: Beatport LINK / SoundCloud Go+
サブスクリプションに配信権が含まれる場合あり
利用規約を確認

=== 著作権フリー音源ソース ===

1. Free Music Archive（FMA）
https://freemusicarchive.org
CC ライセンスの音楽

2. Epidemic Sound（有料）
月額制で配信での使用OK

3. Artlist（有料）
年額制で全音楽使用可能

4. NoCopyrightSounds（NCS）
YouTube で有名
EDM 系が多い

5. Monstercat Gold
月額制で Twitch 配信向け
EDM ジャンルに特化
```

---

## 配信スケジュールと成長戦略

定期的な配信スケジュールは視聴者を定着させ、チャンネルを成長させる最も効果的な方法である。

### 配信スケジュールの設計

**効果的なスケジュール作り:**

```
=== スケジュールの重要性 ===

定期配信のメリット:
1. 視聴者が予定を立てやすい
2. プラットフォームのアルゴリズムに有利
3. 自分自身のルーティン化
4. コミュニティの形成

=== 推奨スケジュールパターン ===

パターン1: 週1回（初心者向け）
毎週金曜 21:00-23:00
2時間の DJ セット

パターン2: 週2回（成長期）
水曜 21:00-22:30（平日セット）
土曜 20:00-23:00（週末ロングセット）

パターン3: 週3回以上（本格的）
月・水・金 21:00-23:00
各日異なるジャンルテーマ

=== 配信時間帯の考察 ===

日本のゴールデンタイム:
20:00-23:00 → 視聴者が最も多い

深夜帯:
23:00-02:00 → クラブタイムに合わせる
海外視聴者を狙える

週末:
金曜夜、土曜夜が最も人が集まる

=== テーマ別配信 ===

毎回テーマを設けると:
- 視聴者が期待して来る
- SNS での告知がしやすい
- セットの準備がしやすい

テーマ例:
月曜: Deep House Monday
水曜: Throwback Wednesday（90s-2000s）
金曜: Club Vibes Friday
土曜: All Genre Party
```

### 視聴者エンゲージメント戦略

**コミュニティを育てる具体的な方法:**

```
=== エンゲージメント向上テクニック ===

1. 挨拶を丁寧に
新しい視聴者: 「ようこそ！初めて？」
常連: 「おー、〇〇さんまた来てくれた！」

2. リクエストタイム
セットの途中で:
「次の曲リクエスト受け付けます！」
全てに応えられなくても OK
対応した時の盛り上がりが大きい

3. 投票システム
「次のジャンル、どっちがいい？」
A: Deep House
B: Techno
チャットで投票

4. Q&A セッション
配信の合間に:
「DJ 機材について質問ある？」
「この曲の探し方教えるよ」

5. コラボ配信
他の DJ と共同配信
Back to Back（B2B）スタイル
両方の視聴者がクロスオーバー

6. 限定コンテンツ
サブスク限定の配信
リクエスト優先権
限定ミックステープ

=== 視聴者維持のコツ ===

離脱を防ぐポイント:

1. 最初の5分が勝負
配信開始直後は盛り上がる曲で
ゆっくりスタートすると離脱される

2. 30分ルール
30分に1回は何か変化をつける
ジャンル変更、MC、エフェクト等

3. セットの流れ
起承転結を意識
オープニング → ビルドアップ → ピーク → クールダウン

4. 配信の長さ
1-2時間: 初心者向け
2-3時間: 標準的
4時間以上: 長時間配信イベント

=== Discord コミュニティ ===

Discord サーバー作成:
配信の常連が集まる場

チャンネル構成:
#一般チャット: 雑談
#リクエスト: 次回配信のリクエスト
#配信告知: スケジュール通知
#おすすめ曲: メンバーの曲シェア
#機材相談: DJ 機材の話題
```

---

## 実践テンプレート集

DJ配信の各シーンで使える実践的なテンプレートとセットアップ例をまとめる。

### OBS シーンコレクション テンプレート

**完成形のシーン構成例:**

```
=== テンプレート: Standard DJ Stream ===

シーン一覧:

Scene 1: Pre-Stream（配信準備中）
Sources:
- Image: pre_stream_bg.png（1920x1080 背景画像）
- Text: 「配信まもなく開始」
- Text: 開始時間（手動更新）
- Audio: BGM（小さめの音量）
- Browser: StreamElements チャット

Scene 2: Intro（オープニング）
Sources:
- Media Source: intro_video.mp4（10秒のイントロ動画）
- Audio: イントロ SE

Scene 3: DJ Main（メイン画面）
Sources:
- Video Capture: Camera（正面）
- Audio Input: Rekordbox Audio（BlackHole）
- Image: overlay_frame.png（フレーム）
- Image: logo.png（右下ロゴ）
- Text: DJ Name（下部中央）
- Browser: Now Playing Widget
- Browser: StreamElements Alerts
- Browser: StreamElements Chat（右側）

Scene 4: Close-Up（手元クローズアップ）
Sources:
- Video Capture: Camera 2（手元カメラ）
- Audio Input: Rekordbox Audio
- Image: overlay_frame.png
- Browser: StreamElements Alerts

Scene 5: BRB（休憩中）
Sources:
- Image: brb_bg.png
- Text: 「少々お待ちください」
- Audio: BGM（環境音楽）

Scene 6: Ending（エンディング）
Sources:
- Image: ending_bg.png
- Text: 「ご視聴ありがとうございました」
- Text: 次回配信予定
- Text: SNS アカウント

=== シーン切り替えのタイミング ===

配信の流れ:

-10分: Scene 1（Pre-Stream）
-1分: Scene 2（Intro）
0分: Scene 3（DJ Main）
30分: Scene 4（Close-Up、適宜切り替え）
60分: Scene 5（BRB、休憩時）
再開: Scene 3（DJ Main）
終了5分前: MC で感謝
終了: Scene 6（Ending）
+2分: 配信停止
```

### 配信前チェックリスト完全版

**本番で失敗しないための確認リスト:**

```
=== 1日前 ===

□ プレイリスト作成（20-30曲以上）
□ Hot Cue / Memory Cue 設定
□ 新曲の BPM・Key 確認
□ SNS で告知投稿
□ PC のアップデート適用（配信直前は避ける）

=== 3時間前 ===

□ 部屋の片付け（カメラに映る範囲）
□ 照明チェック・位置調整
□ カメラ角度確認
□ PC 再起動（メモリクリア）
□ 不要なアプリを全て終了

=== 1時間前 ===

□ OBS 起動・設定確認
□ Rekordbox 起動・オーディオ確認
□ 音声ルーティング テスト
□ カメラ映像テスト
□ Stream Key 有効性確認

=== 30分前 ===

□ テスト配信（1-2分）で映像・音声確認
□ Audio Mixer レベル確認（ピーク -6〜-3 dB）
□ オーバーレイ表示確認
□ チャットボット起動（Nightbot等）
□ Scene「Pre-Stream」選択
□ ドリンク準備（水は必須）

=== 10分前 ===

□ [Start Streaming] クリック
□ 配信開始を確認（Twitch/YouTube ダッシュボード）
□ 映像が正常か最終確認
□ チャットに挨拶投稿

=== 配信中 ===

□ 定期的に Stats 確認（ドロップフレーム）
□ 5-10分ごとにチャット確認
□ 30分ごとにシーン変化（アングル変更等）
□ 飲み物補給

=== 配信終了時 ===

□ 感謝の挨拶
□ 次回配信の告知
□ Scene「Ending」に切り替え
□ 1-2分待って [Stop Streaming]
□ Rekordbox 終了
□ OBS 終了

=== 配信後 ===

□ アーカイブ確認
□ VOD のDMCA チェック（必要に応じて削除）
□ SNS に感謝投稿
□ 良かった曲・反応をメモ
□ 次回のセットリスト構想
□ 配信の Stats を確認（最大視聴者数、平均視聴者数）
```

### OBS 設定エクスポートとバックアップ

**設定を保存して万が一に備える:**

```
=== シーンコレクションのエクスポート ===

OBS メニュー:
[Scene Collection] > [Export]

保存先:
~/Documents/OBS_Backup/

ファイル:
.json 形式

定期的にバックアップ:
月に1回以上

=== プロファイルのエクスポート ===

OBS メニュー:
[Profile] > [Export]

含まれる設定:
- Stream 設定（キー以外）
- Output 設定
- Audio 設定
- Video 設定
- Hotkey 設定

=== 全設定のバックアップ ===

OBS 設定フォルダ:

Mac:
~/Library/Application Support/obs-studio/

Windows:
%APPDATA%/obs-studio/

フォルダごとコピーで完全バックアップ

復元:
同じ場所にコピーバック

=== バックアップのタイミング ===

必ずバックアップすべき時:
1. 配信設定が完成した時
2. OBS アップデート前
3. OS アップデート前
4. 新しいオーバーレイ追加後
5. 月次定期バックアップ
```

---

## 上級テクニック: Studio Mode と トランジション

OBS の Studio Mode を使うことで、配信中にプレビューを確認しながらスムーズなシーン切り替えが可能になる。

### Studio Mode の活用

**プロフェッショナルな配信制御:**

```
=== Studio Mode とは ===

通常モード:
シーン切り替え → 即座に切り替わる

Studio Mode:
左: プレビュー（次のシーン）
右: プログラム（現在の配信映像）
中央: [Transition] ボタンで切り替え

有効化:
OBS 右下の [Studio Mode] ボタンをクリック

メリット:
- 切り替え前にプレビュー確認
- トランジション効果が使える
- 誤操作を防げる

=== DJ配信での活用シーン ===

1. カメラアングル切り替え
DJ Main → Close-Up へ
プレビューで画角を確認してから切り替え

2. オーバーレイ変更
テキストや画像を変更してから表示

3. 緊急時
問題が起きてもプログラムは影響なし
プレビューで修正してから切り替え

=== トランジション設定 ===

OBS Scene Transitions:

1. Cut（カット）
瞬時に切り替え
DJ配信で最もよく使う

2. Fade（フェード）
Duration: 300-500ms
滑らかに切り替え
シーン間の移行に最適

3. Slide（スライド）
横方向にスライド
アクセントに使える

4. Stinger（スティンガー）
動画ファイルをトランジションに使用
プロ感が出る
動画制作が必要

DJ配信推奨:
基本: Fade 300ms
アクセント: Cut（曲のドロップに合わせて）
```

---

## 配信成長のためのデータ分析

配信のパフォーマンスを数値で把握し、改善につなげることが長期的な成長の鍵となる。

### 分析すべき指標

**主要 KPI（重要業績指標）:**

```
=== 視聴者指標 ===

平均視聴者数（Average Viewers）:
最も重要な指標
Twitch アフィリエイトの条件にも関係

最大同時視聴者数（Peak Viewers）:
配信のピーク時の視聴者数

新規視聴者数:
初めて来た人の数

フォロワー/チャンネル登録者増加数:
配信ごとの増加を追跡

=== エンゲージメント指標 ===

チャット数:
活発なチャット = 高いエンゲージメント

平均視聴時間:
長いほど良い
15分以上が理想

=== 成長目標設定 ===

1ヶ月目:
平均視聴者: 3-5人
フォロワー: 30人
配信回数: 4回以上

3ヶ月目:
平均視聴者: 10-20人
フォロワー: 100人
配信回数: 12回以上

6ヶ月目:
平均視聴者: 30-50人
フォロワー: 300人
Twitch アフィリエイト達成

1年目:
平均視聴者: 50-100人
フォロワー: 1000人
収益化達成
```

---

**次は:** [動画コンテンツ](./video-content.md) - YouTube、TikTok用の動画作成

---

## 次に読むべきガイド

- [Mixcloud・SoundCloud投稿](./mixcloud-soundcloud.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
