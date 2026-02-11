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

**次は:** [ワークフロー基礎](./workflow-basics.md) - 効率的な作業の流れ
