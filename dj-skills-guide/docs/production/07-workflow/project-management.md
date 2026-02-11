# Project Management（プロジェクト管理）

複数プロジェクトの効率的な管理、バージョン管理、ファイル整理システムを完全マスターします。

## この章で学ぶこと

- 複数プロジェクト管理
- フォルダ構造最適化
- バージョン管理システム
- ファイル命名規則
- バックアップ戦略
- プロジェクト進捗管理
- データ整理テクニック

---

## なぜProject Managementが重要なのか

**カオスとの戦い:**

```
管理なし:

状況:
ファイルどこ？
どのバージョン？
バックアップは？

結果:
探す時間 30分/日
ファイル消失
作業停止

管理あり:

状況:
全て決まった場所
バージョン明確
バックアップ自動

結果:
探す時間 0分
安心
効率的

プロとアマの差:

アマ:
Desktop散らかる
ファイル名「無題1」
バックアップなし

プロ:
整理された構造
明確な命名
自動バックアップ

統計:

プロデューサー調査:

ファイル探す時間:
管理なし: 20-30分/日
管理あり: 0-2分/日

差: 10-15倍

1年で:
管理なし: 120時間浪費
管理あり: 10時間

差: 110時間 = 5曲以上制作可能

データ消失:

管理なし: 50%が経験
管理あり: 5%が経験

差: 10倍
```

---

## フォルダ構造

**完璧なシステム:**

### 推奨構造

```
~/Music/Production/
│
├── 00-Templates/
│   ├── Mixing-Template.als
│   ├── Drum-Rack-Template.als
│   ├── Mastering-Chain.als
│   └── Buildup-Template.als
│
├── 01-Active/（進行中 最大3曲）
│   ├── 2025-01-DarkTechno/
│   └── 2025-02-HouseVocal/
│
├── 02-Paused/（一時停止）
│   └── 2024-12-ExperimentalIDM/
│
├── 03-Mixing/（Mix中）
│   └── 2025-01-MelodicTechno/
│
├── 04-Finished/（完成・未リリース）
│   ├── 2024-11-NightDrive/
│   └── 2024-12-DeepHouse/
│
├── 05-Released/（リリース済み）
│   ├── 2024-01-FirstTrack/
│   └── 2024-06-SummerVibes/
│
├── 06-Archive/（アーカイブ）
│   └── 2023-XX-OldProjects/
│
├── 99-Samples/
│   ├── Kicks/
│   ├── Hi-Hats/
│   ├── Vocals/
│   └── FX/
│
└── 99-Renders/
    ├── Mixdowns/
    └── Stems/

理由:

番号prefix:
ソート順固定
重要順に表示

Active 3曲制限:
集中力維持
完成率向上

明確な状態分け:
進捗明確
次に何するか明確
```

---

## プロジェクトフォルダ構造

**個別プロジェクト:**

### 完璧な構造

```
2025-01-DarkTechno/
│
├── Project Files/
│   ├── Versions/
│   │   ├── v1.0_idea_250101.als
│   │   ├── v2.0_drums_250102.als
│   │   ├── v3.0_harmony_250103.als
│   │   ├── v4.0_melody_250104.als
│   │   ├── v5.0_arrangement_250105.als
│   │   ├── v6.0_mixing_250106.als
│   │   └── v7.0_final_250107.als
│   │
│   └── Working/
│       └── DarkTechno.als（現在の作業ファイル）
│
├── Samples/
│   ├── Drums/
│   ├── Bass/
│   ├── Synths/
│   └── FX/
│
├── Renders/
│   ├── Mixdowns/
│   │   ├── DarkTechno_v5.0_250105.wav
│   │   ├── DarkTechno_v6.0_250106.wav
│   │   └── DarkTechno_FINAL_250107.wav
│   │
│   ├── Stems/
│   │   └── DarkTechno_Stems_250107/
│   │       ├── Kick.wav
│   │       ├── Bass.wav
│   │       ├── Drums.wav
│   │       ├── Synths.wav
│   │       └── FX.wav
│   │
│   └── Demo/
│       └── DarkTechno_Demo_128kbps.mp3
│
├── Notes/
│   ├── Ideas.txt
│   ├── Feedback.txt
│   └── Changelog.txt
│
└── Reference/
    ├── Reference-Track-1.mp3
    └── Reference-Track-2.mp3

メリット:

明確な分類:
探す時間ゼロ

バージョン管理:
全履歴保存
いつでも戻れる

Samples整理:
プロジェクト専用
削除しない

Renders保存:
全バージョン比較可能
成長確認
```

---

## ファイル命名規則

**一貫性が命:**

### Ableton Liveプロジェクト

```
形式:
vX.Y_description_YYMMDD.als

例:

v1.0_idea_250101.als
v2.0_drums_250102.als
v3.0_harmony_250103.als
v4.0_melody_250104.als
v5.0_arrangement_250105.als
v6.0_mixing_250106.als
v7.0_final_250107.als

ルール:

v1.0: Idea完成
v2.0: Drums & Bass完成
v3.0: Harmony追加
v4.0: Melody追加
v5.0: Arrangement完成
v6.0: Mixing完成
v7.0: Mastering完成・最終版

日付:
YYMMDD形式
250101 = 2025年1月1日

description:
短く
内容示す
```

### Render・Export

```
形式:
TrackName_vX.Y_YYMMDD.wav

例:

DarkTechno_v5.0_250105.wav
DarkTechno_v6.0_250106.wav
DarkTechno_FINAL_250107.wav

Stems:
TrackName_Stems_YYMMDD/
└── ElementName.wav

例:
DarkTechno_Stems_250107/
├── Kick.wav
├── Bass.wav
├── Drums.wav
├── Synths.wav
└── FX.wav

Demo用:
TrackName_Demo_128kbps.mp3
軽量
フィードバック用
```

### Sample Files

```
形式:
Category_Description_Key_BPM.wav

例:

Kick_Sub_C_128.wav
Kick_Punch_G_128.wav
Bass_Reese_Am_128.wav
Synth_Pad_Dm_128.wav
Vocal_Chop_C_120.wav

メリット:

検索容易:
Key・BPM即座に分かる

整理簡単:
Category別に分類

互換性:
プロジェクト間で使い回し
```

---

## バージョン管理

**いつでも戻れる:**

### バージョニング戦略

```
メジャーバージョン（X.0）:

v1.0: Idea
v2.0: Drums & Bass
v3.0: Harmony
v4.0: Melody
v5.0: Arrangement
v6.0: Mixing
v7.0: Mastering

変更:
大きな変更
Phase完成

マイナーバージョン（X.Y）:

v2.1: Kickを変更
v2.2: Hi-Hat追加
v2.3: Bass Pattern変更

変更:
小さな調整
Phase内の改善

保存タイミング:

Phase完了時:
必ず保存
新バージョン作成

大きな変更前:
実験する前
失敗しても戻れる

1日の終わり:
その日の最終状態
翌日続きから

保存方法:

Ableton Live:

File → Save As...
または
Cmd + Shift + S

新しいバージョン番号:
v2.0 → v2.1

保存先:
Versions/フォルダ

重要:

Working/にも保存:
現在の作業ファイル
常に最新版
```

### Changelog（変更履歴）

```
Notes/Changelog.txt:

v1.0 (2025-01-01):
- Initial idea
- BPM: 128, Key: Am
- Reference: Track X by Artist Y

v2.0 (2025-01-02):
- Drums complete
- Kick: 3 layers
- Hi-Hat: 16th notes
- Bass: simple root notes

v3.0 (2025-01-03):
- Harmony added
- Chords: Am - F - C - G
- Pad: 2 layers, heavy reverb

v4.0 (2025-01-04):
- Melody created
- Lead: Wavetable,saw wave
- Hook: catchy, repeats 4 times

v5.0 (2025-01-05):
- Arrangement complete
- Structure: Intro-Verse-Build-Drop-Break-Build-Drop-Outro
- Length: 6:00

v6.0 (2025-01-06):
- Mixing complete
- Reference: Track Z
- LUFS: -8.5

v7.0 (2025-01-07):
- Mastering complete
- Final LUFS: -8.2
- Ready for release

メリット:

記憶補助:
何をしたか明確

判断材料:
どのバージョンが良かったか

学習:
次のプロジェクトで改善
```

---

## プロジェクト進捗管理

**可視化で効率化:**

### Notion・Trelloでの管理

```
Notion Database:

カラム:

1. Track Name: 曲名
2. Status: Active/Paused/Mixing/Finished
3. Phase: Idea/Drums/Harmony/Melody/Arr/Mix/Master
4. Progress: 0-100%
5. Deadline: 締め切り
6. BPM: テンポ
7. Key: キー
8. Genre: ジャンル
9. Started: 開始日
10. Notes: メモ

例:

| Track Name    | Status | Phase   | Progress | Deadline   |
|---------------|--------|---------|----------|------------|
| DarkTechno    | Active | Mixing  | 80%      | 2025-01-15 |
| HouseVocal    | Active | Melody  | 50%      | 2025-02-01 |
| ExperimentalIDM| Paused | Harmony | 30%      | -          |

フィルター:

Active:
進行中のみ表示

By Phase:
Phaseごとに並べ替え

By Deadline:
締め切り順

メリット:

一目で把握:
全プロジェクト状況

優先順位:
何を先にやるか明確

モチベーション:
進捗可視化
```

### 週次レビュー

```
毎週日曜日:

Step 1: 進捗確認（15分）

今週完了:
□ DarkTechno: Mixing完了
□ HouseVocal: Melody 50%

来週予定:
□ DarkTechno: Mastering完成
□ HouseVocal: Melody完成

Step 2: Statusupdate（10分）

Notion:
Progress更新
Status変更

Paused確認:
1ヶ月停滞:
諦めてArchive

Step 3: 計画（10分）

来週目標:
1曲完成
または
大きな進捗

時間配分:
各プロジェクト何時間

Step 4: 整理（10分）

Folder整理:
完成した曲 → Finished
諦めた曲 → Archive

Backup確認:
最新バックアップ確認

合計: 45分/週

効果:
完成率 2倍
迷走防止
```

---

## バックアップ戦略

**データ消失防止:**

### 3-2-1ルール

```
3つのコピー:

1. Original:
Mac本体
作業用

2. Local Backup:
外付けHDD
Time Machine

3. Cloud Backup:
Google Drive
または
Dropbox

2つの異なるメディア:

HDD:
外付け
Time Machine

Cloud:
Google Drive
遠隔地

1つのオフサイト:

Cloud:
火事・盗難でも安全

実装:

Time Machine:

System Preferences → Time Machine:
On

外付けHDD:
1TB以上
自動バックアップ

頻度:
1時間ごと

Google Drive:

重要プロジェクト:
Finished・Released

Sync:
手動
完成時のみ

理由:
自動だと容量使いすぎ

手動バックアップ:

週1回:
Active・Pausedフォルダ
外付けHDDにコピー

月1回:
全プロジェクト
Google Driveにアップ
```

### Ableton Live自動バックアップ

```
Preferences:

File/Folder:

Create Backup Folder: On

場所:
Project Folder内
「Backup」フォルダ

頻度:
5分ごと（推奨）

保存数:
最新10個

メリット:

クラッシュ対策:
最新5分に戻れる

誤操作:
すぐ戻せる

重要:

定期的にVersions保存:
Backupは一時的
Versionsは永久保存
```

---

## データ整理

**定期的な整理:**

### 月次整理（60分）

```
第1週:

Step 1: Active整理（15分）

完成した曲:
Active → Finished

停滞している曲（1ヶ月以上）:
Active → Paused
または
Archive

新しいプロジェクト:
開始
Active 3曲まで

Step 2: Paused整理（10分）

2ヶ月以上停滞:
Paused → Archive

理由:
もう戻らない
容量の無駄

Step 3: Samples整理（20分）

使わないSample:
削除
または
外付けHDDへ

使うSample:
Categoryごとに整理
命名規則統一

重複削除:
同じSample複数ある
1つだけ残す

Step 4: Renders整理（15分）

古いRender:
中間バージョン削除
最終版のみ残す

理由:
容量節約

保存:
重要なバージョン
すべて残す

合計: 60分/月

効果:
ディスク空間 30-50% 節約
探す時間 ゼロ
```

---

## 複数プロジェクト管理

**同時進行のコツ:**

### 最大3曲ルール

```
理由:

1曲のみ:
飽きる
行き詰まる

3曲まで:
気分転換可能
行き詰まったら別の曲

4曲以上:
集中力分散
完成しない

運用:

曲A（最優先）:
Phase 5 Arrangement 80%
締め切り: 今週末

曲B（次優先）:
Phase 3 Harmony 50%
締め切り: 来週

曲C（実験）:
Phase 1 Idea 20%
締め切りなし

1日の配分:

曲A: 2時間（最優先）
曲B: 1時間
曲C: 30分（気分転換）

完成したら:

曲A完成:
Active → Finished

曲B:
最優先へ昇格

新曲D:
Active追加
```

### プロジェクトローテーション

```
週次ローテーション:

月曜:
曲A 3時間

火曜:
曲B 3時間

水曜:
曲A 3時間

木曜:
曲C 2時間 + 曲A 1時間

金曜:
曲A 3時間（追い込み）

土日:
休み
または
新しいIdea

メリット:

新鮮な耳:
1日空けると客観的

集中:
1日1曲に集中

完成:
最優先を確実に進める
```

---

## よくある失敗

### 1. フォルダ構造なし

```
問題:
ファイルどこ？
探す時間 30分

解決:
推奨フォルダ構造導入
一度整理したら維持
```

### 2. バージョン管理なし

```
問題:
前のバージョンに戻れない
失敗したら最初から

解決:
Phase完了ごとに保存
vX.Y命名規則
```

### 3. バックアップなし

```
問題:
HDD故障
全データ消失

解決:
3-2-1ルール
Time Machine + Cloud
```

### 4. 命名規則バラバラ

```
問題:
「無題1」「test」「final_final_REAL」

解決:
統一した命名規則
vX.Y_description_YYMMDD
```

### 5. 整理しない

```
問題:
古いファイル溜まる
容量不足
遅くなる

解決:
月次整理60分
不要ファイル削除
Archive移動
```

---

## まとめ

### Project Management

```
□ フォルダ構造最適化
□ ファイル命名規則統一
□ バージョン管理システム
□ バックアップ 3-2-1ルール
□ 進捗管理（Notion・Trello）
□ 月次整理
□ Active 3曲まで
```

### 重要原則

```
□ 一貫性が命（命名・構造）
□ バックアップ必須（消失防止）
□ 定期的整理（月1回60分）
□ 進捗可視化（モチベーション）
□ シンプルに保つ（複雑化防止）
```

### 次のステップ

1. **フォルダ構造構築** - 今日中に
2. **既存プロジェクト整理** - 今週末
3. **バックアップ設定** - 今日中に

---

**次は:** [Time Management](./time-management.md) - 時間配分の黄金比と効率化テクニック
