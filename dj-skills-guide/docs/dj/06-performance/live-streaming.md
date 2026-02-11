# ライブストリーミング

オンラインでDJセットを配信し、世界中のファンにリーチする完全ガイド。

## この章で学ぶこと

- ライブストリーミングの重要性
- OBS Studioセットアップ
- Twitch配信
- YouTube Live配信
- 機材構成と接続
- 音質と画質設定
- チャットとの交流
- 配信スケジュール
- 著作権の注意点
- 収益化

## なぜライブストリーミングが重要か

**配信しない vs 配信する:**
```
配信しない:
- クラブだけでプレイ
- 限られた観客（50-200人）
- ローカルのみ
- 成長遅い

配信する:
- 世界中に配信
- 無限の観客（100-10,000人+）
- グローバル
- ファンベース拡大

ストリーミング = 現代DJの必須スキル
```

---

## 1. ライブストリーミングの重要性

### 3つのメリット

**メリット1: ファンベース構築**
```
定期配信:
- 毎週金曜20:00等
- ファンが習慣化
- コミュニティ形成

例: Carl Coxも配信
→ 数千人が視聴
→ ファン獲得
```

**メリット2: 練習の場**
```
配信 = 本番:
- 緊張感
- 観客の反応
- 実践経験

クラブより気軽:
- 失敗してもOK
- 何度でもできる
```

**メリット3: 収益化**
```
Twitch:
- サブスクリプション
- ドネーション（投げ銭）
- 広告収益

副収入の可能性
```

---

## 2. OBS Studioセットアップ

### OBS Studioとは

```
OBS Studio = 無料の配信ソフト

特徴:
✓ 完全無料
✓ Windows、macOS対応
✓ Twitch、YouTube対応
✓ プロも使用

ダウンロード:
https://obsproject.com/
```

### インストール手順

**Step 1: ダウンロードとインストール**
```
1. https://obsproject.com/ にアクセス
2. 「Download OBS Studio」
3. macOS / Windowsを選択
4. ダウンロード
5. インストール
   - macOS: .dmgを開いてアプリケーションにドラッグ
   - Windows: .exeを実行

5分で完了
```

### 基本設定

**Step 2: 初回起動と設定**
```
1. OBS Studioを起動

2. Auto-Configuration Wizard（自動設定）:
   - 「Optimize for streaming」選択
   - Next

3. プラットフォーム選択:
   - Twitch / YouTube選択
   - Next

4. 画質設定:
   - 1920x1080（フルHD）推奨
   - 30 FPS
   - Next

5. テスト実行:
   - OBSが自動で最適設定
   - Apply Settings

基本設定完了
```

---

## 3. DDJ-FLX4とOBSの接続

### 機材構成

**必要な機材:**
```
必須:
✓ DDJ-FLX4
✓ ノートPC（Core i5以上、8GB RAM以上）
✓ Webカメラ（720p以上）
✓ インターネット（アップロード 5Mbps以上）

推奨:
✓ リングライト（照明）
✓ グリーンバック（背景）
✓ 外付けマイク（音質向上）
```

### オーディオ設定

**OBSのオーディオ設定:**
```
OBSで音声をキャプチャ:

1. Settings（設定）> Audio

2. Desktop Audio:
   - Device: DDJ-FLX4（またはシステム音声）
   - これでRekordboxの音をキャプチャ

3. Mic/Auxiliary Audio:
   - Device: Built-in Microphone
   - または外付けマイク
   - MC用

4. Sample Rate:
   - 44.1kHz（音楽用）

5. OK

音声キャプチャ完了
```

---

## 4. OBSシーン構成

### シーンとは

```
シーン = 配信画面の構成

例:
- シーン1: DJ全体（ワイド）
- シーン2: 機材アップ
- シーン3: 手元アップ
- シーン4: 「休憩中」画面

切り替え可能
```

### シーン作成

**基本的な3シーン:**
```
シーン1: DJ全体

Sources（ソース）:
1. Video Capture Device（Webカメラ）
   - Webカメラを追加
   - 自分が映る
   - サイズ: フルスクリーン

2. Audio Input Capture（DDJ-FLX4）
   - DDJ-FLX4の音
   - Rekordbox音声

3. Text（テキスト）
   - DJ名
   - SNSハンドル
   - 配置: 下部

---

シーン2: 機材アップ

Sources:
1. Video Capture Device（別アングル）
   - 機材を映す
   - DDJ-FLX4中心

2. Audio Input Capture
   - 同じ

---

シーン3: 休憩中

Sources:
1. Image（画像）
   - 「休憩中」画像
   - DJ名、SNS

2. Audio Input Capture
   - 音楽継続

切り替えて使う
```

---

## 5. Twitch配信

### Twitchとは

```
Twitch = ゲーム・音楽配信プラットフォーム

特徴:
✓ DJカテゴリーあり
✓ チャット活発
✓ 収益化可能
✓ コミュニティ形成

URL: https://www.twitch.tv/
```

### Twitchアカウント作成と設定

**Step 1: アカウント作成**
```
1. https://www.twitch.tv/ にアクセス
2. Sign Up（登録）
3. ユーザー名（DJ名推奨）
4. メールアドレス、パスワード
5. 登録完了
```

**Step 2: Stream Key取得**
```
1. Twitchにログイン
2. 右上のアイコン > Creator Dashboard
3. Settings > Stream
4. Primary Stream Key
   - 「Copy」をクリック
   - この鍵が必要

絶対に他人に教えない
```

**Step 3: OBSにStream Key設定**
```
1. OBS > Settings > Stream
2. Service: Twitch
3. Server: Auto（自動）
4. Stream Key: [コピーした鍵を貼り付け]
5. OK

配信準備完了
```

### 配信開始

**Step 4: 配信開始**
```
1. OBSで「Start Streaming」ボタン
2. Twitchで自分のチャンネルを開く
3. 配信が開始されている

Twitch側で設定:
- Title: 「Tech House Mix - Friday Night」
- Category: Music & Performing Arts > DJ
- Tags: Tech House, DJ, Live等

Go Live!
```

---

## 6. YouTube Live配信

### YouTube Liveとは

```
YouTube Live = YouTubeのライブ配信

特徴:
✓ 圧倒的なリーチ
✓ アーカイブ自動保存
✓ 収益化（条件あり）
✓ SEOに強い

URL: https://www.youtube.com/
```

### YouTube Live設定

**Step 1: YouTube Studioへ**
```
1. YouTubeにログイン
2. 右上のカメラアイコン > Go Live
3. 初回は24時間待つ必要あり
   （YouTubeの仕様）
4. 承認後、配信可能
```

**Step 2: Stream Key取得**
```
1. YouTube Studio > Go Live
2. Stream（ストリーム）タブ
3. Stream Key
   - 「Copy」をクリック
4. Stream URL
   - 「Copy」をクリック

両方必要
```

**Step 3: OBSに設定**
```
1. OBS > Settings > Stream
2. Service: YouTube / YouTube - RTMPS
3. Stream Key: [コピーした鍵を貼り付け]
4. OK

配信準備完了
```

### 配信開始

**Step 4: 配信開始**
```
1. YouTube Studio > Go Live
2. タイトル、説明入力
3. カテゴリ: Music
4. Visibility: Public（公開）
5. OBSで「Start Streaming」
6. YouTube側で「Go Live」

配信開始
```

---

## 7. 音質と画質設定

### 画質設定（OBS）

**推奨設定:**
```
Settings > Video:

Base Resolution: 1920x1080（フルHD）
Output Resolution: 1920x1080
FPS: 30（または60）

Settings > Output:

Output Mode: Advanced
Encoder: x264（CPU）または NVENC（GPU）
Bitrate: 4500 kbps（フルHDの場合）

理由:
- フルHD = 高画質
- 30 FPS = 十分滑らか
- 4500 kbps = Twitch推奨
```

### 音質設定

**推奨設定:**
```
Settings > Audio:

Sample Rate: 44.1kHz
Audio Bitrate: 160 kbps（音楽用）

Settings > Advanced:

Audio Monitoring: Monitor and Output
理由: 自分でも音を確認

音質最優先
DJは音が命
```

---

## 8. チャットとの交流

### チャット確認方法

**OBSでチャット表示:**
```
方法1: ブラウザソース

1. OBS > Sources > Browser
2. URL: Twitchのチャット埋め込みURL
3. Width: 400, Height: 600
4. 画面右側に配置

配信中にチャットが見える
```

**方法2: 別モニター**
```
2台目モニターに:
- Twitchページ
- チャット確認
- コメントに反応

理想的
```

### チャットとの交流

**やり方:**
```
定期的にチャット確認:

- 曲の合間に
- 「Hi everyone!」
- 「Thanks for watching!」
- 質問に答える

例:
視聴者: 「この曲名は？」
DJ: 「It's [曲名] by [アーティスト]!」

コミュニケーション = ファン獲得
```

---

## 9. 配信スケジュール

### 定期配信の重要性

**習慣化:**
```
おすすめ:
- 毎週金曜 20:00-22:00
- 毎週同じ時間
- 2時間

ファンが習慣化:
→ 定期視聴
→ コミュニティ形成

不定期配信 = ファン定着しない
```

### 配信告知

**SNSで事前告知:**
```
配信3日前:
「今週金曜20:00から配信します！
Tech Houseをプレイします。

Twitch: [リンク]

お楽しみに！🎧

#Twitch #DJLife #TechHouse」

当日3時間前にもリマインド

告知 = 視聴者増加
```

---

## 10. 著作権の注意点

### Twitch

**Twitch DMCA:**
```
注意:
- 商用音楽は著作権侵害リスク
- DMCAテイクダウン（削除）
- アカウント停止の可能性

対処法:
1. Twitch Soundtrack使用（無料音楽）
2. 自分の曲のみ
3. リスク承知でプレイ

多くのDJはリスク承知
```

### YouTube

**YouTube Content ID:**
```
注意:
- Content IDで自動検出
- 収益化停止
- 動画削除
- アカウント停止

対処法:
1. オリジナル曲のみ
2. ロイヤリティフリー音楽
3. リスク承知

著作権が最も厳しい
```

### 安全な方法

**著作権クリア:**
```
完全に安全:
1. 自分のオリジナル曲
2. ロイヤリティフリー音楽
3. 許可を得た曲

現実的:
- 多くのDJはリスク承知
- DMCAが来たら対応
- バックアップアカウント準備

プロも同じリスク
```

---

## 11. 収益化

### Twitch収益化

**Affiliate / Partner:**
```
Twitch Affiliate（準パートナー）:
条件:
- 50フォロワー
- 7日間配信
- 平均3人視聴

収益:
✓ サブスクリプション（月額課金）
✓ ビッツ（投げ銭）
✓ 広告収益

月5,000-50,000円可能
```

### YouTube収益化

**YouTube Partner Program:**
```
条件:
- チャンネル登録者 1,000人
- 年間視聴時間 4,000時間

収益:
✓ 広告収益
✓ スーパーチャット（投げ銭）
✓ チャンネルメンバーシップ

達成難しいが可能
```

---

## 12. よくある質問

### Q1: PCスペックはどれくらい必要？

**A: Core i5、8GB RAM以上**

```
推奨スペック:
- CPU: Intel Core i5以上（または AMD Ryzen 5）
- RAM: 8GB以上（16GB推奨）
- GPU: 内蔵グラフィックスでOK（NVENC使うならGTX 1650以上）
- ネット: アップロード 5Mbps以上

これで1080p 30fps配信可能
```

### Q2: Webカメラは必要？

**A: あった方が良い**

```
理由:
✓ 視聴者がDJの顔を見れる
✓ コネクションが強くなる
✓ エンゲージメント向上

安いWebカメラでOK:
- Logicool C920（5,000円程度）
- 720p以上

ない場合:
- 機材だけ映す
- それもあり
```

### Q3: 照明は必要？

**A: リングライトがおすすめ**

```
理由:
✓ 顔が明るく映る
✓ プロっぽい
✓ 視聴者体験向上

リングライト:
- 3,000-10,000円
- Amazon等で購入

なくてもOK:
- 部屋の照明で
- ただし明るめに
```

---

## まとめ

- **重要性**: ファンベース構築、練習の場、収益化の可能性
- **OBS**: 無料配信ソフト、https://obsproject.com/ からダウンロード
- **機材**: DDJ-FLX4、PC、Webカメラ、ネット5Mbps以上
- **Twitch**: Stream Key取得 > OBSに設定 > Start Streaming
- **YouTube**: 24時間待つ > Stream Key取得 > 配信開始
- **音質**: 44.1kHz、160kbps、音質最優先
- **画質**: 1920x1080、30fps、4500kbps
- **チャット**: ブラウザソースで表示、定期的にコミュニケーション
- **スケジュール**: 定期配信（毎週金曜等）、SNSで事前告知
- **著作権**: TwitchとYouTubeはDMCAリスク、多くのDJはリスク承知
- **収益化**: Twitch Affiliate（50フォロワー）、YouTube Partner（1,000登録者）

**次のステップ:** [クラブプロトコル](./club-protocol.md) でプロのマナーを学ぶ

---

## 参考リンク

- [クラブプロトコル](./club-protocol.md)
- [セットのレコーディング](./recording-sets.md)
- [OBS Studio](https://obsproject.com/)
- [Twitch](https://www.twitch.tv/)
- [YouTube](https://www.youtube.com/)
