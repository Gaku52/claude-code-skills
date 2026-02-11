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

**次は:** [動画コンテンツ](./video-content.md) - YouTube、TikTok用の動画作成
