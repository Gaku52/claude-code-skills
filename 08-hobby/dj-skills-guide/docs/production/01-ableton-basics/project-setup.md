# プロジェクト設定

新規プロジェクトの作成から保存まで。正しいプロジェクト管理で制作効率を最大化します。

## この章で学ぶこと

- 新規プロジェクト作成の完全手順
- テンポ、拍子、キー設定
- テンプレートの作成と活用
- プロジェクト保存とバージョン管理
- フォルダ構造の最適化
- バックアップ戦略

---

## なぜプロジェクト設定が重要なのか

**最初が肝心:**

```
正しい設定:

スムーズな制作:
途中で設定変更不要

ファイル整理:
どこに何があるか明確

バックアップ:
データ消失防止

間違った設定:

後で困る:
テンポ変更が面倒
ファイルが散乱

データ消失:
保存場所が分からない
バックアップなし

プロの習慣:

新規プロジェクト:
必ず同じ手順
テンプレート活用

結果:
制作に集中できる
```

---

## 新規プロジェクト作成

**Step by Step:**

### Step 1: Ableton Live起動

```
起動画面:

┌────────────────────────────┐
│  New Live Set              │ ← これを選択
│  Open Live Set             │
│  Recent Projects           │
│  Templates                 │
│  Lessons                   │
└────────────────────────────┘

New Live Set:
完全な空プロジェクト

Templates:
後で説明

Open Live Set:
既存プロジェクト開く
```

### Step 2: テンポ設定

```
Transport セクション:

BPM設定:
┌──────────┐
│ 128.00   │ ← ここをクリック
└──────────┘

ジャンル別推奨BPM:

Techno: 125-135
House: 120-128
Trance: 138-142
Drum & Bass: 170-180
Dubstep: 140
Hip Hop: 85-95

Techno制作の場合:
128 BPM から始める
(後で変更可能)

設定方法:

1. BPM数字をクリック
2. 128 と入力
3. Enter

または:
↑↓キーで微調整
```

### Step 3: 拍子設定

```
Transport セクション:

拍子設定:
┌──────┐
│ 4/4  │ ← ここをクリック
└──────┘

4/4 拍子:
99%の電子音楽

他の拍子:
3/4 (ワルツ)
6/8 (トライバル)
5/4 (実験的)

変更方法:
クリック → プルダウン → 選択

通常:
4/4 のまま変更不要
```

### Step 4: プロジェクト保存

```
重要: すぐ保存！

Cmd+S (Mac) / Ctrl+S (Win)

保存ダイアログ:

プロジェクト名:
例: "Techno Track 001"

保存場所:
~/Music/Ableton/Projects/
(デフォルト)

または:
外付けSSD
（推奨）

Collect All and Save:
☑ チェック推奨
→ 全サンプルをプロジェクトフォルダにコピー

保存:
クリック

結果:
"Techno Track 001.als" 作成
+ "Techno Track 001 Project" フォルダ
```

---

## テンプレートの作成

**毎回の手間を省く:**

### 基本テンプレート作成

```
目的:

新規プロジェクトごとに:
トラック作成
エフェクト設定
→ 面倒

テンプレート:
よく使う構成を保存
→ 毎回1クリック

作成手順:

1. 新規プロジェクト作成

2. よく使うトラック配置:

Track 1: Kick
└─ EQ Three
└─ Compressor

Track 2: Bass
└─ Auto Filter
└─ Saturation

Track 3: Synth
└─ Reverb

Track 4-8: 空トラック

Return A: Reverb
Return B: Delay

Master: Limiter

3. 保存:
File > Save Live Set as Default Set
→ "Techno Template"

4. 場所:
~/Music/Ableton/Templates/

次回から:
File > New Live Set from Template
→ "Techno Template"
→ すぐ使える！
```

### ジャンル別テンプレート

```
Techno Template:

BPM: 128
Tracks:
- Kick (Compressor)
- Bass (Filter)
- Percussion
- Synth (Reverb)
- FX
- Vocal (空)
Return: Reverb, Delay
Master: Limiter

House Template:

BPM: 124
Tracks:
- Kick
- Clap/Snare
- Hi-Hat
- Bass
- Chords
- Vocal
Return: Reverb, Delay, Filter
Master: Limiter, EQ

Minimal Template:

BPM: 125
Tracks:
- Kick
- Perc 1-4 (複数)
- Bass
- Texture
Return: Reverb, Delay
Master: Limiter

複数テンプレート:
使い分けて効率化
```

---

## プロジェクト保存戦略

**データ消失を防ぐ:**

### 保存のタイミング

```
自動保存: なし

Ableton Live:
自動保存機能なし
→ 手動保存必須

保存頻度:

5分ごと:
Cmd+S

大きな変更後:
すぐ Cmd+S

休憩前:
必ず Cmd+S

習慣化:
無意識にCmd+S

バージョン保存:

Save As (Cmd+Shift+S):
"Techno Track 001 v2"
"Techno Track 001 v3"

いつ:
大きく方向性変える前
完成版の前

理由:
元に戻れる
```

### プロジェクトフォルダ構造

```
Ableton Projects/
├── Techno Track 001/
│   ├── Techno Track 001.als ← プロジェクトファイル
│   ├── Samples/
│   │   └── Recorded/ ← 録音したAudio
│   ├── Ableton Project Info/ ← メタデータ
│   └── Backup/ ← バックアップ(自動)
│
├── Techno Track 002/
│   ├── Techno Track 002.als
│   ├── Techno Track 002 v2.als ← バージョン
│   ├── Techno Track 002 v3.als
│   └── Samples/
│
└── Templates/
    ├── Techno Template.als
    └── House Template.als

自動生成:
Samples/, Backup/
→ Ableton が自動作成

重要:
全て1つのフォルダに
散らばらない
```

---

## プロジェクト設定の詳細

**Preferences から:**

### Collect All and Save

```
File > Collect All and Save

効果:

使用中のサンプル:
全てプロジェクトフォルダにコピー

外部サンプル:
~/Music/Samples/kick.wav
↓
Projects/Techno001/Samples/kick.wav

メリット:

ポータブル:
プロジェクトフォルダごと移動
→ 他のPCでも開ける

安全:
元のサンプル削除しても大丈夫

デメリット:

容量増える:
1サンプル = 数MB
100サンプル = 数百MB

いつ使う:

完成時:
最終版で必ず実行

途中:
定期的に（週1回）
```

### 情報の記録

```
プロジェクト情報:

File > Info Text

記入内容:

Started: 2025-12-08
BPM: 128
Key: A minor
Genre: Techno
Notes: Dark, minimal

後で見返す:
何を作っていたか分かる

バージョン管理:

v1: 2025-12-08
初回アイデア

v2: 2025-12-10
ドロップ追加

v3: 2025-12-15
マスタリング完了

メモ:
変更内容を記録
```

---

## プロジェクト命名規則

**一貫性が大事:**

### 推奨命名法

```
パターン1: 日付+通し番号

例:
20251208_Track_001
20251208_Track_002
20251215_Track_003

メリット:
時系列で並ぶ
一目で分かる

パターン2: ジャンル+通し番号

例:
Techno_001_Dark
House_001_Groovy
Techno_002_Minimal

メリット:
ジャンル別に整理

パターン3: アーティスト名+タイトル

例:
YourName_Eclipse
YourName_Midnight
YourName_Dawn

メリット:
リリース用
プロっぽい

選ぶ:
自分に合ったパターン
一貫性が重要
```

### NGな命名

```
ダメな例:

"New Project"
"Untitled"
"asdf"
"test"
"aaaa"

理由:
後で何か分からない

良い例:

"Techno_128BPM_Dark_v1"
"20251208_Minimal_Groove"
"Eclipse_Final_Master"

理由:
内容が分かる
```

---

## キー設定（オプション）

**ハーモニックミキシング用:**

```
キー設定:

Transport付近:
Key表示なし
→ 手動で記録

記録方法:

Info Text に記入:
Key: A minor
または
Key: 8A (Camelot)

なぜ重要:

DJセットで使う:
自作曲のキーを知る
→ 他の曲とハーモニックミキシング

コラボ:
「このトラックはCメジャー」
→ 相手が合わせやすい

設定方法:

1. プロジェクト開始時に決める
   "Aminorで作る"

2. 途中で分析
   Mixed In Key等のソフト

3. Info Textに記録
```

---

## 実践: 初めてのプロジェクト作成

**30分の演習:**

### Step 1: 新規プロジェクト (5分)

```
1. Ableton Live起動
   → New Live Set

2. BPM設定: 128

3. 拍子確認: 4/4

4. すぐ保存:
   Cmd+S
   名前: "My_First_Project"
   場所: デフォルト
   Collect All: ☑

5. 確認:
   Finderで確認
   フォルダ作成されている
```

### Step 2: トラック作成 (10分)

```
1. Track 1作成:
   Browser > Drums > Kick
   → Session Viewにドラッグ

2. Track 2作成:
   Browser > Sounds > Bass
   → ドラッグ

3. Track 3作成:
   Browser > Sounds > Synth
   → ドラッグ

4. 保存:
   Cmd+S
```

### Step 3: テンプレート保存 (10分)

```
1. File > Save Live Set as Template

2. 名前: "My Template"

3. 次回確認:
   File > New from Template
   → "My Template" があるか
```

### Step 4: バージョン保存 (5分)

```
1. Bass削除

2. 別のBass追加

3. Save As:
   Cmd+Shift+S
   → "My_First_Project_v2"

4. 確認:
   v1, v2 両方存在
```

---

## よくある質問

### Q1: プロジェクトファイルが見つからない

**A:** 検索機能を使う

```
Finder (Mac):
Cmd+Space
→ "Techno Track 001"

Windows:
Windowsキー
→ "Techno Track 001"

Ableton内:
File > Open Recent
→ 最近のプロジェクト一覧

習慣:
決まった場所に保存
~/Music/Ableton/Projects/
```

### Q2: 途中でBPM変更できる？

**A:** 可能、影響範囲に注意

```
変更方法:
Transport のBPM変更
128 → 130

影響:

Audioクリップ:
Warpされていれば追従
→ 問題なし

MIDIクリップ:
自動で追従
→ 問題なし

エフェクト:
Delay等のタイミング変わる
→ 要調整

いつ変更:

早めに:
プロジェクト初期なら気軽に

完成間近:
避ける（面倒）
```

### Q3: Collect All and Saveはいつやる？

**A:** 完成時、または定期的

```
必須:

完成時:
最終版で必ず

タイミング:

週1回:
金曜日に全プロジェクト

大きな変更後:
新しいサンプル多数追加

他PCに移す前:
必ず実行

理由:
外部サンプルのリンク切れ防止
```

---

## まとめ

### プロジェクト作成手順

```
1. New Live Set
2. BPM設定 (128)
3. 拍子確認 (4/4)
4. すぐ保存 (Cmd+S)
5. 作業開始
6. 5分ごとに保存
7. バージョン保存 (Cmd+Shift+S)
8. 完成時: Collect All and Save
```

### テンプレート活用

```
よく使う構成:
テンプレートに保存
→ 毎回の手間削減
```

### 命名規則

```
一貫性:
自分のルールを決める
例: "Genre_001_Description"
```

### チェックリスト

```
□ 新規プロジェクト作成
□ BPM設定
□ すぐ保存
□ トラック作成
□ テンプレート保存
□ バージョン保存を試す
□ Collect All and Save実行
□ 命名規則を決める
```

---

**次は:** [環境設定](./preferences-settings.md) - Preferences完全ガイド
