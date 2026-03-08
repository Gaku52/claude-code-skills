# ファイル管理

プロジェクトとサンプルを完璧に整理する。データ消失を防ぎ、制作効率を最大化します。

## この章で学ぶこと

- プロジェクトフォルダ構造の最適化
- サンプル管理システム構築
- バックアップ戦略（3-2-1ルール）
- クラウド同期活用
- ライブラリ整理術
- ディスク容量管理

---

## なぜファイル管理が重要なのか

**データ消失は悲劇:**

```
よくある失敗:

「プロジェクトが開かない」
→ サンプルのリンク切れ

「HDDクラッシュ」
→ 1年分の曲が全消失

「どこに保存したか不明」
→ 作りかけの曲が見つからない

「ディスク満杯」
→ 録音できない

正しいファイル管理:

整理:
どこに何があるか明確

バックアップ:
3箇所に保存

高速:
SSDで快適

結果:
安心して制作に集中
```

---

## プロジェクトフォルダ構造

**標準構造を理解:**

### Abletonの自動生成フォルダ

```
Techno Track 001/
├── Techno Track 001.als ← プロジェクトファイル本体
├── Techno Track 001 v2.als ← バージョン
│
├── Samples/ ← 録音/追加サンプル
│   ├── Recorded/ ← 自分で録音したAudio
│   │   ├── 0001 Audio.wav
│   │   └── 0002 Audio.wav
│   └── Imported/ ← 外部から追加したサンプル
│
├── Ableton Project Info/ ← メタデータ
│   ├── AProject-8_1.cfg
│   └── Project8_1.cfg
│
└── Backup/ ← 自動バックアップ
    ├── Techno Track 001 [2025-12-08 140523].als
    └── Techno Track 001 [2025-12-08 153012].als

自動生成:
Ableton が勝手に作る
触らなくてOK

重要:

.als ファイル:
プロジェクト本体

Samples/:
使用中のサンプル

Backup/:
自動バックアップ
(最大25個まで保存)
```

### 推奨フォルダ構造

```
~/Music/
├── Ableton/
│   ├── Projects/ ← 全プロジェクト
│   │   ├── 2025-12/
│   │   │   ├── Techno Track 001/
│   │   │   ├── House Track 002/
│   │   │   └── Minimal 003/
│   │   └── 2026-01/
│   │       └── ...
│   │
│   ├── Templates/ ← テンプレート
│   │   ├── Techno Template.als
│   │   └── House Template.als
│   │
│   ├── User Library/ ← Abletonライブラリ
│   │   ├── Presets/
│   │   ├── Samples/
│   │   └── Clips/
│   │
│   └── Samples/ ← 共有サンプル集
│       ├── Kicks/
│       ├── Basslines/
│       ├── Synths/
│       └── Vocals/
│
└── Exports/ ← 書き出したファイル
    ├── Masters/
    ├── Stems/
    └── Demos/

メリット:

月別整理:
古いプロジェクトも見つけやすい

集中管理:
全て ~/Music/Ableton/ 配下

バックアップ簡単:
このフォルダごとバックアップ
```

---

## サンプル管理

**膨大なサンプルを整理:**

### サンプルの種類

```
1. Abletonパック:
   公式/サードパーティ
   → User Library に自動保存

2. 購入サンプル:
   Splice, Loopmasters等
   → ~/Music/Ableton/Samples/

3. 自作サンプル:
   録音した音
   → プロジェクト内Samples/

4. フリーサンプル:
   ネットからDL
   → ~/Music/Ableton/Samples/Free/

管理方針:

集中管理:
全サンプルを1箇所に
~/Music/Ableton/Samples/

カテゴリ分け:
Kicks, Snares, Bass...

命名規則:
Kick_Dark_001.wav
Bass_Deep_Techno.wav
```

### サンプル整理術

```
フォルダ構造:

~/Music/Ableton/Samples/
├── Drums/
│   ├── Kicks/
│   │   ├── Techno/
│   │   ├── House/
│   │   └── 808/
│   ├── Snares/
│   ├── Claps/
│   ├── Hats/
│   └── Percussion/
│
├── Bass/
│   ├── Sub Bass/
│   ├── Acid Bass/
│   └── Reese Bass/
│
├── Synths/
│   ├── Leads/
│   ├── Pads/
│   └── Plucks/
│
├── FX/
│   ├── Risers/
│   ├── Impacts/
│   └── Transitions/
│
└── Vocals/
    ├── Loops/
    └── One Shots/

命名規則:

[カテゴリ]_[特徴]_[番号].wav

例:
Kick_Dark_001.wav
Bass_Deep_Techno_001.wav
Vocal_Female_Aah_Cm.wav

メタデータ:
BPM, Key を含める
Kick_125BPM_Gm.wav
```

### Splice / Loopmasters サンプル管理

```
Splice:

自動DL先:
~/Music/Splice/

整理:
定期的に Ableton/Samples/ へ移動

Loopmasters:

購入後:
ZIP解凍
→ Ableton/Samples/Loopmasters/[パック名]/

タグ付け:
Ableton Browser で検索しやすく

重複回避:
同じサンプルを複数箇所に保存しない
→ ディスク容量の無駄
```

---

## バックアップ戦略

**3-2-1 ルール:**

### 3-2-1 ルールとは

```
3: 3つのコピー
2: 2種類のメディア
1: 1つはオフサイト (外部)

具体例:

コピー1 (オリジナル):
内蔵SSD
~/Music/Ableton/

コピー2 (ローカルバックアップ):
外付けSSD
/Volumes/Backup SSD/Ableton/

コピー3 (クラウド):
Google Drive / Dropbox / iCloud
クラウド/Ableton/

頻度:

コピー1: 常時
コピー2: 毎日 or 毎週
コピー3: 毎週 or 毎月

自動化:
Mac: Time Machine
Win: File History
クラウド: 同期アプリ
```

### Time Machine (Mac)

```
設定:

1. 外付けHDD接続 (1TB以上)

2. システム設定 > Time Machine

3. バックアップディスク追加:
   外付けHDD選択

4. 自動バックアップ: On

バックアップ頻度:
1時間ごと

復元:

アプリ: Time Machine
→ 過去に遡る
→ ファイル選択
→ 復元

利点:
完全自動
追加作業不要
```

### File History (Windows)

```
設定:

1. 外付けHDD接続

2. 設定 > 更新とセキュリティ
   > バックアップ

3. ドライブの追加:
   外付けHDD選択

4. 自動的にファイルをバックアップ: On

バックアップ頻度:
1時間ごと

復元:
設定 > バックアップ > その他のオプション
→ 現在のバックアップからファイルを復元
```

### クラウドバックアップ

```
選択肢:

Google Drive: 15GB無料、100GB¥250/月
Dropbox: 2GB無料、2TB¥1,200/月
iCloud (Mac): 5GB無料、50GB¥130/月
OneDrive (Win): 5GB無料、100GB¥229/月

推奨:

完成プロジェクトのみ:
クラウドに保存

作業中:
ローカルのみ
(同期が遅い)

設定:

1. クラウドフォルダ作成:
   ~/Google Drive/Ableton Backup/

2. 完成プロジェクトをコピー:
   週1回手動
   または rsync で自動

3. 確認:
   クラウド上にアップロード済みか
```

---

## Collect All and Save

**ポータブル化:**

### 使い方

```
目的:

外部サンプルをプロジェクトフォルダにコピー
→ どこでも開けるようにする

タイミング:

完成時:
必ず実行

他PCに移す前:
必須

定期的:
月1回

手順:

1. File > Collect All and Save

2. 確認ダイアログ:
   「Collecting files...」

3. 完了:
   全サンプルがSamples/フォルダにコピーされる

4. サイズ確認:
   プロジェクトフォルダが大きくなる
   (数百MB〜数GB)

Minimal化:

Collect Files > Minimal

効果:
未使用サンプルは除外
→ サイズ削減
```

---

## ディスク容量管理

**容量不足を防ぐ:**

### 容量チェック

```
Mac:
Finder > ファイル > 情報を見る
または
Cmd+I

Windows:
右クリック > プロパティ

推奨容量:

最低: 50GB空き
推奨: 100GB空き
快適: 200GB以上

不足時:

症状:
録音できない
プロジェクトが保存できない

対処:
後述の削減テクニック
```

### 容量削減テクニック

```
1. 古いBackup削除:

各プロジェクト/Backup/
→ 古いバックアップ削除
(最新5個だけ残す)

2. 未使用プロジェクト削除:

1年以上開いていない
→ 外付けHDDに移動
または削除

3. 書き出したAudioを圧縮:

WAV → MP3
(Demo用)

Master以外はMP3でOK

4. サンプルパック整理:

未使用パック:
アンインストール

Splice:
ダウンロードしすぎない

5. Cache削除:

~/Library/Caches/Ableton/
→ 古いキャッシュ削除

6. 外付けSSD活用:

作業中プロジェクト: 内蔵SSD
完成プロジェクト: 外付けSSD
アーカイブ: 外付けHDD
```

---

## 実践: ファイル管理システム構築

**60分の作業:**

### Step 1: フォルダ構造作成 (15分)

```
1. Finder / Explorer 開く

2. ~/Music/Ableton/ 作成

3. サブフォルダ作成:
   - Projects/
   - Projects/2025-12/
   - Templates/
   - Samples/
   - Samples/Drums/
   - Samples/Bass/
   - Samples/Synths/
   - Samples/FX/

4. ~/Music/Exports/ 作成

5. 確認:
   全フォルダが作成された
```

### Step 2: 既存プロジェクト移動 (15分)

```
1. 既存プロジェクト検索:
   Spotlight/検索 で .als

2. Projects/2025-12/ に移動:
   フォルダごとドラッグ

3. Ableton で開く:
   正常に開くか確認

4. 古いプロジェクト削除:
   元の場所から削除
```

### Step 3: バックアップ設定 (20分)

```
Mac:

1. 外付けHDD接続

2. Time Machine設定:
   システム設定 > Time Machine
   ディスク追加

3. 自動バックアップ開始

Windows:

1. 外付けHDD接続

2. File History設定:
   設定 > バックアップ
   ドライブ追加

3. 自動バックアップ開始

クラウド:

1. Google Drive アプリインストール

2. ~/Google Drive/Ableton Backup/ 作成

3. 完成プロジェクトをコピー
```

### Step 4: 整理ルール決定 (10分)

```
命名規則:

プロジェクト:
YYYYMMDD_Genre_Description
例: 20251208_Techno_Dark

サンプル:
Category_Feature_Number
例: Kick_Heavy_001

バージョン:
プロジェクト名_v2, v3...

メモ:
ルールをテキストファイルに保存
~/Music/Ableton/FILE_NAMING_RULES.txt
```

---

## よくある質問

### Q1: プロジェクトがサンプル見つからないエラー

**A:** Collect All and Save

```
症状:
「サンプルが見つかりません」

原因:
外部サンプルのパスが変わった
元の場所が削除された

解決:

修復:
サンプル再リンク
(面倒)

予防:
File > Collect All and Save
→ プロジェクトフォルダにコピー

以降:
定期的にCollect実行
```

### Q2: ディスクがすぐ満杯になる

**A:** 定期的な整理

```
週1回:
未使用プロジェクト確認
→ 外付けに移動

月1回:
Backup フォルダ整理
→ 古いバックアップ削除

3ヶ月ごと:
完成プロジェクト:
内蔵SSD → 外付けSSD

未完成だが放置:
削除またはアーカイブ

サンプル:
未使用パック削除

目標:
常に100GB以上空き
```

### Q3: クラウド同期が遅い

**A:** 完成品のみ同期

```
問題:
作業中プロジェクトをクラウド同期
→ 常にアップロード
→ 制作が遅くなる

解決:

作業中:
ローカルのみ
~/Music/Ableton/Projects/

完成時:
手動でクラウドにコピー
~/Google Drive/Ableton Backup/

自動化:
週末に rsync スクリプト実行
(上級者向け)
```

---

## プロジェクト整理の高度なテクニック

**大規模ライブラリの効率的管理:**

### プロジェクトアーカイブ戦略

```
3段階管理:

段階1: 作業中 (内蔵SSD)
~/Music/Ableton/Projects/Current/
- 現在制作中のプロジェクト
- 毎日アクセス
- 最速SSDに配置

段階2: 完成品 (外付けSSD)
/Volumes/Projects SSD/Ableton/Finished/
- 完成したがすぐアクセスしたい
- 月1-2回アクセス
- 高速アクセス可能

段階3: アーカイブ (外付けHDD)
/Volumes/Archive HDD/Ableton/Archive/
- 1年以上未使用
- 念のため保管
- 低速だが大容量・安価

年別整理:

Archive/2023/
Archive/2024/
Archive/2025/

ジャンル別サブフォルダ:
Archive/2025/Techno/
Archive/2025/House/
Archive/2025/Experimental/

メリット:
- 内蔵SSDを常に軽量化
- 過去作品も保管
- 必要時に復元可能
```

### プロジェクトメタデータ管理

```
プロジェクトフォルダに INFO.txt を配置:

---
Project: Dark Techno 001
Date: 2025-12-08
Genre: Techno
BPM: 135
Key: G minor
Status: Finished
Release: SoundCloud 2025-12-15
---

Notes:
Heavy kick with sidechain compression
Used Serum for main bass
Mastered with Ozone 11

Samples Used:
- Kick: Vengeance EDM Vol 2
- Hat: Splice - Underground Techno
- Vocal: Self-recorded

Plugins:
- Serum
- FabFilter Pro-Q 3
- Ozone 11

---

メリット:

1年後に見ても:
何を使ったか一目瞭然

コラボ時:
情報共有が簡単

プラグイン確認:
別PCで開く前に必要プラグイン確認
```

### バージョン管理ベストプラクティス

```
命名規則:

Track Name.als ← 最新版
Track Name v2.als ← 過去バージョン
Track Name v3.als
Track Name Final.als ← 最終版
Track Name Final Master.als ← マスタリング済み

バージョンメモ:

Track Name v2.als.txt:
「キック音色変更、ベース追加」

Track Name v3.als.txt:
「ブレイクダウン追加、全体構成変更」

定期保存:

大きな変更の前:
必ず新バージョン保存

実験時:
Track Name Experiment.als

元に戻せる:
失敗しても大丈夫

Git管理 (上級者):

git init (プロジェクトフォルダ内)
git add Track Name.als
git commit -m "Initial version"

変更後:
git commit -m "Added breakdown"

過去に戻る:
git checkout <commit-id>

注意:
.als はバイナリファイル
→ diffは取れない
→ コミットメッセージが重要
```

---

## サンプルライブラリの最適化

**検索速度を最大化:**

### Abletonブラウザのタグ活用

```
サンプルにタグ付け:

1. Browser > サンプル右クリック
2. Edit Info Text
3. タグ追加:
   kick, dark, techno, 135bpm, gm

検索:
Browser検索窓に "dark kick"
→ 即座に該当サンプル表示

カラータグ:

赤: よく使う
オレンジ: お気に入り
黄: 実験的
緑: クリーン
青: ダーク

右クリック > Color:
色選択

ブラウザでフィルタ:
色で絞り込み

お気に入り登録:

右クリック > Add to Favorites
または
ドラッグ > User Library/Favorites/

次回から:
Favorites フォルダから即アクセス
```

### サンプルパックの効率的インストール

```
インストール先統一:

全パック:
~/Music/Ableton/Packs/

サブフォルダ:
~/Music/Ableton/Packs/Vengeance/
~/Music/Ableton/Packs/Splice/
~/Music/Ableton/Packs/Loopmasters/

Abletonに認識させる:

Preferences > Library
> Add Folder
> ~/Music/Ableton/Packs/ を追加

→ Browser に全パック表示

パックカタログ管理:

PACKS_CATALOG.txt:

---
Pack: Vengeance EDM Vol 2
Installed: 2025-10-01
Size: 15GB
Categories: Kicks, Snares, Synths
Notes: Heavy techno sounds
---

Pack: Splice - Underground Techno
Installed: 2025-11-15
Size: 2GB
Categories: Loops, Percussion
Notes: Dark minimal loops
---

使用状況記録:
よく使うパック → 保持
未使用パック → アンインストール候補
```

### 重複サンプル検出と削除

```
問題:
同じサンプルを複数箇所に保存
→ ディスク容量の無駄

ツール:

Mac: dupeGuru
Windows: Duplicate Cleaner

使い方:

1. dupeGuru起動

2. フォルダ追加:
   ~/Music/Ableton/Samples/

3. スキャン開始

4. 重複検出:
   同一ファイル一覧表示

5. 削除:
   1つ残して削除

注意:

削除前確認:
Ableton プロジェクトで使用中か

安全策:
削除前に外付けHDDにバックアップ

サイズ節約:
数十GB削減も可能
```

---

## クラウドストレージ高度活用

**複数デバイス間での同期:**

### Dropbox / Google Drive 選択的同期

```
問題:
全プロジェクトをクラウド同期
→ ストレージ容量不足
→ 同期が遅い

解決: 選択的同期

Dropbox:

1. Dropbox設定
2. Preferences > Sync > Selective Sync
3. 同期しないフォルダ選択:
   - Archive/
   - Old Projects/

→ 最新プロジェクトのみ同期

Google Drive:

1. Google Drive設定
2. Preferences > Google Drive
3. "Stream files" (ストリーミング)
   または
   "Mirror files" (ミラー)

Stream files:
クラウドにのみ保存
必要時にDL

Mirror files:
ローカルにもコピー

推奨:
作業中: Mirror
完成品: Stream
```

### プロジェクト共有とコラボレーション

```
他のプロデューサーと共同制作:

方法1: Zipで共有

1. Collect All and Save 実行
2. プロジェクトフォルダをZip圧縮
3. WeTransfer / Dropbox で送信

メリット:
確実に全サンプル含まれる

デメリット:
大容量 (数GB)

方法2: クラウド共有フォルダ

1. Dropbox/Google Drive に共有フォルダ作成
2. プロジェクトフォルダをアップロード
3. 共有リンク送信

リアルタイム共同編集:
Splice Studio (有料)
Ableton Note (iOS)

注意点:

バージョン管理:
誰が何を変更したか明記

コミュニケーション:
変更内容をSlack/Discord で共有

競合回避:
同時編集禁止
交代で作業
```

### クラウドバックアップ自動化

```
rsync スクリプト (Mac/Linux):

backup.sh:

#!/bin/bash
SOURCE="$HOME/Music/Ableton/Projects/"
DEST="$HOME/Google Drive/Ableton Backup/"
rsync -av --delete "$SOURCE" "$DEST"
echo "Backup completed: $(date)"

実行権限:
chmod +x backup.sh

手動実行:
./backup.sh

自動実行 (cron):

crontab -e

追加:
0 2 * * 0 /path/to/backup.sh
(毎週日曜 2:00AM に実行)

Windows (PowerShell):

backup.ps1:

$SOURCE = "$env:USERPROFILE\Music\Ableton\Projects\"
$DEST = "$env:USERPROFILE\Google Drive\Ableton Backup\"
robocopy $SOURCE $DEST /MIR
Write-Host "Backup completed: $(Get-Date)"

タスクスケジューラ:
毎週日曜 2:00AM に実行

確認:
定期的にバックアップ成功を確認
```

---

## ディスク性能最適化

**SSD/HDDを使い分ける:**

### ストレージタイプ別活用法

```
内蔵SSD (NVMe/SATA):

用途:
- 作業中プロジェクト
- よく使うサンプル
- Ableton本体

容量:
500GB - 1TB

速度:
読込: 3000MB/s (NVMe)
読込: 550MB/s (SATA)

メリット:
超高速
録音時のレイテンシ最小

外付けSSD (Thunderbolt/USB-C):

用途:
- 完成プロジェクト
- 大容量サンプルパック

容量:
1TB - 2TB

速度:
読込: 2800MB/s (TB3)
読込: 1000MB/s (USB-C)

メリット:
持ち運び可能
複数PC間で共有

外付けHDD (USB 3.0):

用途:
- アーカイブ
- Time Machine バックアップ

容量:
2TB - 8TB

速度:
読込: 150MB/s

メリット:
大容量で安価
長期保存に最適

推奨構成:

作業用: 内蔵SSD 1TB
バックアップ: 外付けSSD 1TB
アーカイブ: 外付けHDD 4TB
```

### ディスクパフォーマンス監視

```
Mac:

アクティビティモニタ:
ディスクタブ
→ 読込/書込速度確認

異常に遅い:
ディスク容量不足
断片化
故障の可能性

Windows:

タスクマネージャー:
パフォーマンスタブ > ディスク
→ 使用率確認

100%使用率:
ディスクがボトルネック
SSD交換を検討

ベンチマーク:

Mac: Blackmagic Disk Speed Test
Win: CrystalDiskMark

定期確認:
月1回実行
速度低下を早期発見
```

### キャッシュとTemp管理

```
Abletonキャッシュ:

場所:
Mac: ~/Library/Caches/Ableton/
Win: %AppData%\Local\Ableton\

容量:
数GB〜数十GB

削除:
Ableton終了後
フォルダごと削除

再生成:
次回起動時に自動作成

効果:
ディスク容量回復
動作軽快化

システムTemp:

Mac: /tmp/, /var/folders/
Win: %TEMP%

自動削除:
Mac: 定期的に自動削除
Win: ディスククリーンアップ

手動削除:

Mac:
sudo rm -rf /tmp/*
(注意: システムファイル削除しない)

Win:
設定 > ストレージ > 一時ファイル
→ 削除
```

---

## 大規模サンプルライブラリ管理

**数十万ファイルを効率管理:**

### サンプルデータベース構築

```
問題:
サンプル数万個
→ どれを使ったか忘れる
→ 重複ダウンロード

解決: スプレッドシート管理

Google Sheets / Excel:

列:
- ファイル名
- カテゴリ
- パック名
- BPM
- Key
- 評価 (1-5)
- 使用回数
- パス

例:

Kick_Dark_001.wav | Kick | Vengeance | - | - | 5 | 10 | ~/Samples/Kicks/
Bass_Techno.wav | Bass | Splice | 128 | Am | 4 | 3 | ~/Samples/Bass/

フィルタ:
評価5のみ表示
→ お気に入りサンプル一覧

集計:
最も使用するパック
→ 投資対効果確認

自動化 (上級者):

Python スクリプト:
全サンプルをスキャン
→ メタデータ抽出
→ CSV出力
```

### サンプル試聴システム

```
問題:
サンプル多すぎ
→ 探すのに時間かかる

解決: プレイリスト作成

Ableton Browser:

Favoritesフォルダに分類:

User Library/Favorites/
├── Best Kicks/
├── Best Snares/
├── Top Bass/
└── Go-To Vocals/

使用頻度高いサンプル:
即座にアクセス

iTunes / Music.app:

サンプルをインポート:
全サンプルをライブラリに追加

プレイリスト作成:
Techno Kicks
House Vocals
Minimal Percussion

通勤中試聴:
スマホで聴いて覚える

制作時:
すぐ思い出せる
```

### 外部サンプルリンク管理

```
リンク切れ防止:

相対パス vs 絶対パス:

絶対パス:
/Users/username/Music/Samples/Kick.wav
→ フォルダ移動でリンク切れ

相対パス:
../Samples/Kick.wav
→ プロジェクトと一緒に移動OK

Abletonデフォルト:
絶対パス

対策:

Collect All and Save:
全サンプルをプロジェクトフォルダに

定期実行:
月1回
完成プロジェクトに必須

メリット:
他PCで開ける
バックアップが完全
```

---

## プロフェッショナル向けバックアップ

**データ損失ゼロを目指す:**

### RAID構成

```
RAID 1 (ミラーリング):

構成:
2台のHDD
同じデータを両方に書込

メリット:
1台故障しても大丈夫

デメリット:
容量が半分

使用例:
2TB × 2台 = 実質2TB

設定:

Mac: Disk Utility > RAID
Win: Storage Spaces

RAID 5 (分散パリティ):

構成:
3台以上のHDD
1台故障してもデータ復旧可能

メリット:
容量効率良い

デメリット:
複雑、コスト高

推奨:

個人制作: RAID 1で十分
プロスタジオ: RAID 5 or 6
```

### NAS (Network Attached Storage)

```
NASとは:
ネットワーク接続ストレージ
複数PCから同時アクセス

メリット:

集中管理:
全プロジェクトをNASに

自動バックアップ:
定期的にクラウドへ

複数PC:
デスクトップとラップトップで共有

推奨機種:

Synology DS220+: 2-bay, ¥30,000
QNAP TS-251D: 2-bay, ¥35,000

構成例:

NAS: 4TB × 2 (RAID 1)
内蔵SSD: 作業中プロジェクト
NAS: 完成/アーカイブ

ワークフロー:

制作中:
内蔵SSDで作業

完成:
NASに移動

NAS:
毎晩自動でクラウドバックアップ
```

### オフサイトバックアップ

```
火災・盗難対策:

問題:
自宅で全バックアップ
→ 火災で全消失

解決: オフサイト保管

方法1: クラウドストレージ

Google Drive
Dropbox
Backblaze (無制限バックアップ ¥700/月)

方法2: 物理的別拠点

外付けHDDを親の家に保管
月1回更新

方法3: 銀行貸金庫

重要プロジェクトのHDD
年1回更新

推奨:

日常: クラウド自動バックアップ
重要: 月1で物理HDD更新
```

---

## トラブルシューティング

**問題解決ガイド:**

### プロジェクトが開かない

```
症状:
ダブルクリックしても開かない

原因と対処:

1. サンプルリンク切れ:

対処:
Locate (サンプル再リンク)
または
Collect All and Save済みバージョンを開く

2. Abletonバージョン不一致:

症状:
「新しいバージョンで作成されました」

対処:
Abletonをアップデート

3. プラグイン不足:

症状:
「プラグインが見つかりません」

対処:
必要プラグインをインストール
または
プラグインをFreeze & Flatten

4. ファイル破損:

対処:
Backup フォルダから古いバージョン開く

予防:
定期的にCollect & Save
```

### ディスクが読み込めない

```
症状:
外付けHDD/SSDがマウントしない

対処:

Mac:

1. Disk Utility 起動
2. View > Show All Devices
3. ディスク選択
4. First Aid
5. Run

修復不可:
データ復旧ソフト
(Disk Drill, TestDisk)

Windows:

1. ディスクの管理
2. ドライブ右クリック
3. プロパティ > ツール > チェック

修復不可:
データ復旧ソフト
(Recuva, EaseUS)

最終手段:

プロのデータ復旧業者
¥50,000〜
重要データのみ
```

### バックアップが失敗する

```
Time Machine エラー:

原因1: ディスク容量不足

対処:
古いバックアップ削除
または
大容量ディスクに交換

原因2: ディスク故障

対処:
Disk Utility > First Aid
修復不可 → ディスク交換

File History エラー:

原因: ドライブ接続切れ

対処:
設定 > バックアップ
ドライブ再接続

クラウド同期エラー:

原因: ネットワーク不安定

対処:
同期一時停止
→ 安定した回線で再開

ファイルサイズ制限:
Google Drive 5TB/ファイル
→ 分割アップロード
```

---

## 移行とアップグレード

**新PCへの移行:**

### Mac to Mac 移行

```
移行アシスタント使用:

手順:

1. 新Mac起動
2. 移行アシスタント起動
3. 古Macから転送 選択
4. 両Mac同じWi-Fi接続
5. 転送開始 (数時間)

転送内容:
全ファイル
全アプリケーション
全設定

注意:

プラグイン:
再認証必要な場合あり

Ableton:
ライセンス再認証

外部サンプル:
パスが変わる可能性
→ Collect All and Save推奨

手動移行:

1. 外付けSSDに全コピー
2. 新Macに接続
3. ~/Music/Ableton/ にコピー
4. Abletonインストール
5. Preferences > Library で確認
```

### Windows to Mac 移行

```
互換性:

Abletonプロジェクト:
完全互換

サンプル:
WAV/AIFF → 互換
Apple Loops → Macのみ

プラグイン:
VST (Win) ≠ AU/VST3 (Mac)
→ 再購入必要な場合あり

手順:

1. Windows:
   Collect All and Save 全実行

2. 外付けHDDにコピー:
   ~/Music/Ableton/

3. Mac接続:
   コピー

4. プラグイン再インストール:
   Mac版購入/DL

5. プロジェクト開く:
   プラグイン不足警告
   → 代替プラグイン使用

推奨:

完成プロジェクト:
Freeze & Flatten
→ プラグイン不要

制作中:
プラグイン再購入検討
```

### SSDアップグレード

```
容量不足時:

症状:
ディスク空き50GB以下
常に容量警告

解決:

1. SSD増設 (デスクトップ):
   内蔵SSD追加
   数万円

2. SSD交換 (ノートPC):
   大容量SSDに交換
   データ移行必要

3. 外付けSSD追加:
   最も簡単
   1TB ¥15,000〜

推奨構成:

内蔵SSD 500GB:
OS + Ableton + 作業中

外付けSSD 1TB:
完成プロジェクト

外付けHDD 4TB:
アーカイブ + バックアップ

移行手順:

1. 新SSD購入
2. Carbon Copy Cloner (Mac) / Macrium Reflect (Win)
3. クローン作成
4. 旧SSD → 新SSD交換
5. 起動確認
```

---

## まとめ

### フォルダ構造

```
~/Music/Ableton/
├── Projects/ (月別整理)
├── Templates/
├── Samples/ (カテゴリ別)
└── User Library/

~/Music/Exports/
├── Masters/
└── Demos/
```

### バックアップ 3-2-1ルール

```
3つのコピー:
1. 内蔵SSD
2. 外付けSSD
3. クラウド

2種類のメディア:
SSD + HDD + クラウド

1つはオフサイト:
クラウドストレージ
```

### 定期メンテナンス

```
毎日: 手動保存 (Cmd+S)
毎週: Collect All and Save (完成時)
毎月: Backup フォルダ整理
3ヶ月: 古いプロジェクト移動
```

### チェックリスト

```
□ フォルダ構造作成
□ 命名規則決定
□ Time Machine / File History 設定
□ クラウドバックアップ設定
□ Collect All and Save 実行
□ ディスク容量確認 (100GB以上空き)
```

---

## エクスポート管理とアーカイブ

**完成品の適切な保存:**

### エクスポート設定とファイル形式

```
用途別エクスポート:

マスター (WAV 24bit):
~/Music/Exports/Masters/
YYYYMMDD_TrackName_Master.wav

配信用 (WAV 16bit 44.1kHz):
~/Music/Exports/Distribution/
TrackName_Distribution.wav

SoundCloud (MP3 320kbps):
~/Music/Exports/Streaming/
TrackName_SC.mp3

DJプール (AIFF):
~/Music/Exports/DJ Pool/
TrackName_DJPool.aiff

Stems (個別トラック):
~/Music/Exports/Stems/TrackName/
01_Kick.wav
02_Bass.wav
03_Lead.wav
...

エクスポート設定:

マスター:
Sample Rate: 48kHz
Bit Depth: 24bit
Format: WAV
Dither: なし

配信:
Sample Rate: 44.1kHz
Bit Depth: 16bit
Format: WAV
Dither: POW-r 2

MP3:
Bitrate: 320kbps
Quality: Highest
Constant Bitrate

メタデータ埋め込み:

Title: トラック名
Artist: アーティスト名
Album: アルバム名
Year: 2025
Genre: Techno
BPM: 135
Key: Gm
ISRC: 国際標準レコーディングコード
```

### Stem管理

```
Stemエクスポートの重要性:

用途:
- リミックス提供
- マスタリングエンジニアへ
- ライブパフォーマンス
- 将来の再編集

グループ分け:

Drums:
Kicks, Snares, Hats, Percussion

Bass:
Sub Bass, Mid Bass

Synths:
Lead, Pad, Arp

FX:
Risers, Impacts, Atmos

Vocals:
Main Vocal, Backing

命名規則:

01_Drums_Kick.wav
02_Drums_Snares.wav
03_Drums_Hats.wav
04_Bass_Sub.wav
05_Bass_Mid.wav
06_Synth_Lead.wav
...

エクスポート手順:

1. 各トラックをSolo
2. Export Audio/Video
3. Rendered Track選択
4. Stem フォルダに保存

一括エクスポート:

全トラック選択
Export Audio/Video
Render as Loop
Individual Clips: On
→ 全Stemを一度にエクスポート

保存場所:

プロジェクトフォルダ内:
TrackName/Stems/

または

専用フォルダ:
~/Music/Exports/Stems/TrackName/
```

### プロジェクトアーカイブ手順

```
完成プロジェクトのアーカイブ:

Step 1: Collect All and Save

File > Collect All and Save
→ 全サンプル集約

Step 2: 不要ファイル削除

Backup フォルダ:
古いバックアップ削除
最新3つだけ残す

未使用Clips:
削除

Step 3: Freeze & Flatten

全Instrument/MIDI トラック:
Freeze
→ Flatten

メリット:
プラグイン不要
容量削減

Step 4: 最終保存

TrackName_FINAL.als
として保存

Step 5: エクスポート

Master WAV
Stems
MP3

Step 6: メタデータ作成

INFO.txt:
全情報記録

Step 7: 圧縮

ZIP圧縮:
TrackName_FINAL.zip

サイズ:
通常1-3GB

Step 8: アーカイブ保存

外付けHDD:
/Volumes/Archive/2025/Techno/

クラウド:
~/Google Drive/Archive/

確認:
両方に保存完了

Step 9: ローカル削除

内蔵SSDから削除
または
外付けSSDに移動

内蔵SSD空き容量確保
```

---

## ライセンスとプラグイン管理

**プラグイン情報の整理:**

### プラグインインベントリ

```
所有プラグインリスト:

PLUGINS_INVENTORY.txt:

---
Name: Serum
Type: Synth
Manufacturer: Xfer Records
Version: 1.36
License: Serial Number
Key: XXXX-XXXX-XXXX
Purchase Date: 2023-05-10
Price: $189
---

Name: FabFilter Pro-Q 3
Type: EQ
Manufacturer: FabFilter
Version: 3.21
License: Activation Code
Key: YYYY-YYYY-YYYY
Purchase Date: 2024-01-15
Price: €149
---

メリット:

新PC移行時:
必要プラグイン確認

再インストール:
ライセンスキー即座に

税務:
経費計上の証拠

バックアップ:
このファイルも必ずバックアップ
```

### プラグインプリセット管理

```
プリセット保存場所:

Mac:
~/Library/Audio/Presets/

Win:
C:\Users\[User]\Documents\[Plugin]/

バックアップ:

定期的にコピー:
外付けSSD
クラウド

カスタムプリセット:

命名規則:
[ジャンル]_[用途]_[特徴]

例:
Techno_Bass_Deep
House_Lead_Bright

整理:

フォルダ分け:
My Presets/Techno/
My Presets/House/

エクスポート:

他PCに移行:
プリセットフォルダごとコピー
```

---

## ファイル管理自動化

**スクリプトとツールで効率化:**

### 自動整理スクリプト

```
Mac/Linux (Bash):

organize_projects.sh:

#!/bin/bash

# 古いプロジェクトを自動アーカイブ

SOURCE="$HOME/Music/Ableton/Projects/"
ARCHIVE="$HOME/Music/Ableton/Archive/"

# 1年以上未更新のプロジェクトを検索
find "$SOURCE" -type d -mtime +365 -maxdepth 1 | while read dir; do
  if [ -d "$dir" ]; then
    echo "Archiving: $dir"
    mv "$dir" "$ARCHIVE"
  fi
done

echo "Archive completed: $(date)"

実行:
chmod +x organize_projects.sh
./organize_projects.sh

自動実行 (cron):
0 3 1 * * /path/to/organize_projects.sh
(毎月1日 3:00AM に実行)

Windows (PowerShell):

organize_projects.ps1:

$SOURCE = "$env:USERPROFILE\Music\Ableton\Projects\"
$ARCHIVE = "$env:USERPROFILE\Music\Ableton\Archive\"
$CUTOFF = (Get-Date).AddDays(-365)

Get-ChildItem -Path $SOURCE -Directory | Where-Object {
  $_.LastWriteTime -lt $CUTOFF
} | ForEach-Object {
  Write-Host "Archiving: $($_.Name)"
  Move-Item $_.FullName -Destination $ARCHIVE
}

Write-Host "Archive completed: $(Get-Date)"

タスクスケジューラ:
毎月1日 3:00AM に実行
```

### サンプル整理ツール

```
重複検出:

dupeGuru:
同一サンプル検出・削除

使用頻度分析:

Python スクリプト:

import os
from collections import Counter

sample_dir = os.path.expanduser("~/Music/Ableton/Samples/")
projects_dir = os.path.expanduser("~/Music/Ableton/Projects/")

# 全プロジェクトから使用サンプルを抽出
used_samples = []

for root, dirs, files in os.walk(projects_dir):
  for file in files:
    if file.endswith(".als"):
      # .alsファイルを解析 (簡略版)
      # 実際はXMLパース必要
      pass

# 使用頻度集計
sample_count = Counter(used_samples)

# 結果出力
for sample, count in sample_count.most_common(50):
  print(f"{sample}: {count} times")

未使用サンプル削除:

全サンプルリスト作成
使用済みサンプルと比較
未使用のみ削除候補として表示
```

---

## チームコラボレーション

**複数人での制作環境:**

### 共有フォルダルール

```
Dropbox/Google Drive 共有:

構造:

Shared Projects/
├── TrackName_001/
│   ├── TrackName_001.als
│   ├── Samples/
│   └── NOTES.txt
│
└── README.txt

README.txt:

---
Collaboration Rules:

1. 作業前:
   必ず他のメンバーに確認
   同時編集禁止

2. 保存時:
   新バージョンとして保存
   TrackName_v2_YourName.als

3. 変更内容:
   NOTES.txt に記載

4. Collect All and Save:
   週1回実行

5. 完成時:
   Freeze & Flatten
   全プラグインをAudio化
---

バージョン命名:

TrackName_v1_Taro.als
TrackName_v2_Hanako.als
TrackName_v3_Taro_Final.als

NOTES.txt更新:

2025-12-08 Taro:
- キック音色変更
- ベースライン追加

2025-12-09 Hanako:
- ブレイクダウン作成
- ボーカルサンプル追加

2025-12-10 Taro:
- ミキシング調整
- マスタリング
```

### プロジェクトハンドオフ

```
他のプロデューサーに引き継ぎ:

チェックリスト:

□ Collect All and Save 実行済み
□ 全サンプルが含まれる
□ 使用プラグインリスト作成
□ BPM, Key 記載
□ INFO.txt 添付
□ テンポマップ確認
□ タイムシグネチャ確認

INFO.txt 内容:

---
Project: Dark Techno 001
BPM: 135
Key: G minor
Time Signature: 4/4

Plugins Used:
- Serum (Lead)
- FabFilter Pro-Q 3 (Master EQ)
- Valhalla VintageVerb (Reverb)

Notes:
Main drop starts at Bar 33
Breakdown at Bar 65
Outro at Bar 97

Please install all plugins before opening.
All samples are included in Samples/ folder.
---

ZIP圧縮:
TrackName_Handoff.zip

送信:
WeTransfer / Dropbox リンク

確認:
受取側が正常に開けることを確認
```

---

## 年次メンテナンス

**年1回の大掃除:**

### 年末アーカイブ作業

```
12月実施:

Step 1: 全プロジェクト確認 (1時間)

開いて確認:
完成? 未完成? 削除?

分類:
- Finished (完成)
- WIP (制作中)
- Archive (保管)
- Delete (削除)

Step 2: アーカイブ移動 (30分)

Archive/2025/ へ移動:
完成プロジェクト
未完成だが保管したい

削除:
明らかに不要
品質低い

Step 3: サンプル整理 (1時間)

重複検出:
dupeGuru 実行

未使用パック削除:
1年間未使用のパック

お気に入り選定:
よく使うサンプルをFavoritesへ

Step 4: バックアップ検証 (30分)

Time Machine:
正常動作確認
復元テスト

クラウド:
アップロード完了確認

外付けHDD:
ディスクエラーチェック

Step 5: ディスク容量最適化 (30分)

Cache削除
Temp削除
古いBackup削除

目標:
200GB以上空き確保

Step 6: プラグインアップデート (30分)

全プラグイン:
最新版に更新

Ableton:
最新版に更新

Step 7: ドキュメント更新 (15分)

PLUGINS_INVENTORY.txt 更新
PACKS_CATALOG.txt 更新
FILE_NAMING_RULES.txt 確認

Step 8: 統計記録 (15分)

STATS_2025.txt:

---
Year: 2025
Tracks Completed: 23
Total Projects: 47
Disk Used: 450GB
Samples: 15,342
Plugins: 37
Most Used Plugin: Serum
Most Used Sample Pack: Vengeance EDM Vol 2
---

振り返り:
何を達成したか
何を改善するか
```

---

## ファイル管理のプロフェッショナルティップス

**上級者向けテクニック:**

### プロジェクトテンプレート活用

```
ジャンル別テンプレート作成:

Techno Template:
- 8トラック構成
- よく使うプラグイン配置
- ルーティング設定済み
- 標準エフェクトチェーン

保存:
~/Music/Ableton/Templates/Techno_Template.als

使用:
File > Open Recent > Templates
→ 即座に制作開始

時間短縮:
セットアップ10分 → 30秒
```

### ディスク暗号化

```
重要データ保護:

Mac (FileVault):
システム設定 > セキュリティ > FileVault
→ ディスク全体暗号化

Win (BitLocker):
コントロールパネル > BitLocker
→ ドライブ暗号化

外付けHDD暗号化:
VeraCrypt (無料)
→ パスワード保護

メリット:
盗難時もデータ安全
未発表曲の保護
```

### クラウド同期の高度設定

```
帯域幅制限:

Dropbox:
Preferences > Bandwidth
Upload: 制限設定
Download: 制限なし

メリット:
制作中も同期が邪魔しない

夜間同期:

スケジュール設定:
22:00-06:00 のみ同期

メリット:
日中の帯域確保
```

---

## 最終チェックリスト

### 日次

```
□ 制作終了時 Cmd+S
□ 重要な変更後 新バージョン保存
□ Time Machine 動作確認
```

### 週次

```
□ 完成プロジェクト Collect All and Save
□ Backup フォルダ確認 (5個まで)
□ ディスク容量確認 (100GB以上空き)
□ クラウド同期確認
```

### 月次

```
□ 未使用プロジェクト整理
□ サンプル整理
□ プラグインアップデート確認
□ バックアップ検証
```

### 年次

```
□ 全プロジェクト分類
□ アーカイブ移動
□ 統計記録
□ ドキュメント更新
□ ディスク最適化
```

---

## 推奨ツール一覧

### ファイル管理

```
Mac:
- Finder (標準)
- Path Finder (有料 ¥4,000)
- Forklift (有料 ¥3,000)

Win:
- Explorer (標準)
- Total Commander (有料 €40)
- Directory Opus (有料 €90)
```

### バックアップ

```
Mac:
- Time Machine (標準)
- Carbon Copy Cloner (有料 ¥4,500)
- SuperDuper! (有料 ¥3,000)

Win:
- File History (標準)
- Macrium Reflect (無料/有料)
- Acronis True Image (有料 ¥5,000/年)
```

### クラウドストレージ

```
- Google Drive (¥250/月 100GB)
- Dropbox (¥1,200/月 2TB)
- iCloud (¥130/月 50GB)
- OneDrive (¥229/月 100GB)
- Backblaze (¥700/月 無制限)
```

### 重複検出

```
Mac:
- dupeGuru (無料)
- Gemini 2 (有料 ¥2,000)

Win:
- Duplicate Cleaner (無料/有料)
- CloneSpy (無料)
```

### ディスク分析

```
Mac:
- DaisyDisk (有料 ¥1,200)
- Disk Inventory X (無料)

Win:
- WinDirStat (無料)
- TreeSize (無料/有料)
```

---

## まとめ

### フォルダ構造

```
~/Music/Ableton/
├── Projects/ (月別整理)
├── Templates/
├── Samples/ (カテゴリ別)
└── User Library/

~/Music/Exports/
├── Masters/
└── Demos/
```

### バックアップ 3-2-1ルール

```
3つのコピー:
1. 内蔵SSD
2. 外付けSSD
3. クラウド

2種類のメディア:
SSD + HDD + クラウド

1つはオフサイト:
クラウドストレージ
```

### 定期メンテナンス

```
毎日: 手動保存 (Cmd+S)
毎週: Collect All and Save (完成時)
毎月: Backup フォルダ整理
3ヶ月: 古いプロジェクト移動
1年: 大掃除・アーカイブ
```

### チェックリスト

```
□ フォルダ構造作成
□ 命名規則決定
□ Time Machine / File History 設定
□ クラウドバックアップ設定
□ Collect All and Save 実行
□ ディスク容量確認 (100GB以上空き)
□ プラグインインベントリ作成
□ 年次メンテナンス計画
```

---

**次は:** [ワークフロー基礎](./workflow-basics.md) - 効率的な作業の流れ
