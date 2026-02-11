# インターフェイス概要

Ableton Liveの画面構成を完全理解する。初めて開いたときの「これ何？」を全て解決します。

## この章で学ぶこと

- Ableton Liveの全画面構成
- 各セクションの役割と使い方
- Browser（サウンド検索）の活用
- Clip View / Device Viewの違い
- Mixer の使い方
- Transport（再生コントロール）
- 効率的な画面レイアウト

---

## なぜインターフェイス理解が重要なのか

**迷子にならないために:**

```
初心者がつまずく理由:

画面が複雑:
情報量が多すぎる

どこに何があるか分からない:
音を探せない
エフェクトが見つからない

結果:
挫折

解決:

体系的に理解:
1つずつ覚えていく

毎日触る:
1週間で慣れる

実践:
実際に使いながら

ゴール:
目をつぶっても操作できる
= Rekordboxと同じレベルに
```

---

## Ableton Liveの画面構成

**5つの主要セクション:**

```
┌─────────────────────────────────────────┐
│ 1. Browser (左)                          │
├──────────┬──────────────────────────────┤
│          │ 2. Session View              │
│          │    または                     │
│          │    Arrangement View (中央)   │
│          ├──────────────────────────────┤
│          │ 3. Clip View                 │
│          │    または                     │
│          │    Device View (下)          │
├──────────┴──────────────────────────────┤
│ 4. Mixer (右)                            │
├─────────────────────────────────────────┤
│ 5. Transport (一番下)                    │
└─────────────────────────────────────────┘

役割分担:

Browser:
サウンド、プラグイン検索

Session/Arrangement View:
曲を作る場所

Clip/Device View:
詳細編集

Mixer:
音量、エフェクト

Transport:
再生、録音、テンポ
```

---

## 1. Browser（ブラウザ）

**サウンドとプラグインの図書館:**

### Browserの構成

```
Browser (左サイド):

├─ Places
│  ├─ User Library (自分のサウンド)
│  ├─ Packs (Abletonパック)
│  └─ Plugins (VSTプラグイン)
│
├─ Categories
│  ├─ Sounds (音色)
│  ├─ Drums (ドラム)
│  ├─ Instruments (音源)
│  ├─ Audio Effects (エフェクト)
│  ├─ MIDI Effects (MIDIエフェクト)
│  └─ Max for Live (Suite版のみ)
│
└─ Search (検索窓)

使い方:

音を探す:
Categories > Sounds > Bass
→ ベース音色一覧

ドラムを探す:
Categories > Drums > Kicks
→ キック一覧

エフェクトを探す:
Categories > Audio Effects > Reverb
→ リバーブ一覧

プラグインを探す:
Places > Plugins > VSTi
→ サードパーティ音源
```

### Browser操作

```
基本:

開く/閉じる:
Cmd+Opt+B (Mac)
Ctrl+Alt+B (Win)

検索:
検索窓に「kick」入力
→ キック音色のみ表示

お気に入り:
右クリック > Add to Favorites
→ Favorites フォルダに追加

プレビュー:
音色クリック
→ 自動再生（ヘッドフォンで聴ける）

ドラッグ&ドロップ:
音色を Session View にドラッグ
→ 新しいトラック作成

Tips:

検索の活用:
「techno kick」
「dark bass」
など具体的に

タグ:
音色には自動タグ
(Dark, Warm, Analog等)
→ タグでフィルター可能

コレクション:
Place タブで
Collections > 自分のコレクション作成
```

---

## 2. Session View / Arrangement View

**2つのビュー:**

### Session View (セッションビュー)

```
Session View とは:

グリッド型:
┌───┬───┬───┬───┐
│Cl │Cl │Cl │Cl │ ← Track 1
├───┼───┼───┼───┤
│Cl │Cl │Cl │Cl │ ← Track 2
├───┼───┼───┼───┤
│Cl │Cl │Cl │Cl │ ← Track 3
└───┴───┴───┴───┘
 Scene 1-4

特徴:

ライブ演奏向き:
クリップを自由に再生

DJプレイに似ている:
即興性高い

ループベース:
8小節、16小節のループ

縦軸:
トラック（楽器）

横軸:
シーン（展開）

使い方:

クリップ再生:
クリップをクリック
→ 再生開始

Scene再生:
右の▶ボタン
→ その行全て再生

Stop:
■ボタン
→ トラック停止

DJとの類似:

DJのチャンネル = Abletonのトラック
DJの曲 = Abletonのクリップ
```

### Arrangement View (アレンジメントビュー)

```
Arrangement View とは:

タイムライン型:
┌──────────────────────────┐
│ Track 1 ████████░░░░████│
│ Track 2 ░░████████████░░│
│ Track 3 ████░░░░██████░░│
└──────────────────────────┘
  0:00 → 4:00

特徴:

曲作り向き:
最初から最後まで

直線的:
時間軸で編集

DAWらしい:
従来のDAWと同じ

使い方:

録音:
赤●ボタン → 録音開始

編集:
クリップを切る、貼る、移動

アレンジ:
イントロ → ビルドアップ → ドロップ

切り替え:

Tab キー:
Session ⇔ Arrangement
即座に切り替え

用途:

Session:
アイデア出し、ライブ演奏

Arrangement:
完成させる、書き出し
```

---

## 3. Clip View / Device View

**下部のビュー:**

### Clip View (クリップビュー)

```
Clip View とは:

クリップの詳細編集:

Audio Clip:
波形表示
Warp（タイミング調整）
ループ設定

MIDI Clip:
ピアノロール
ノート編集
ベロシティ

表示方法:

クリップをダブルクリック
→ Clip View表示

内容:

┌────────────────────────┐
│ ■■ Clip [Kick]        │ ← クリップ名
├────────────────────────┤
│ 波形 or ピアノロール    │ ← 編集エリア
├────────────────────────┤
│ Loop設定               │
│ Start/End             │
│ Transpose             │
└────────────────────────┘

操作:

ループ範囲:
ドラッグで設定

Transpose:
半音単位で移調

Warp (Audio):
タイミング修正
```

### Device View (デバイスビュー)

```
Device View とは:

エフェクト・音源の詳細:

┌────────────────────────┐
│ Reverb [リバーブ]      │
├────────────────────────┤
│ Dry/Wet  [75%]        │
│ Decay    [2.5s]       │
│ Size     [Large]      │
└────────────────────────┘

表示方法:

トラックのデバイスをクリック
→ Device View表示

切り替え:

Clip View ⇔ Device View:
Shift+Tab

使い分け:

Clip View:
クリップの編集
（音階、タイミング）

Device View:
音色の編集
（エフェクト、音源パラメーター）
```

---

## 4. Mixer（ミキサー）

**DJミキサーと同じ概念:**

### Mixer構成

```
各トラックごと:

┌─────┐
│ Pan │ ← パン（左右）
├─────┤
│ A B │ ← Send A/B（エフェクト送り）
├─────┤
│ Vol │ ← 音量フェーダー
├─────┤
│ Mtr │ ← メーター（音量表示）
└─────┘

DJミキサーとの対応:

DJミキサー → Ableton Mixer
EQ Hi/Mid/Low → EQ Three (Device)
Trim → Gain
Volume Fader → Volume Fader
Crossfader → なし（不要）

操作:

音量調整:
フェーダーをドラッグ

Pan:
左右の定位

Mute:
トラック無音化

Solo:
そのトラックのみ再生

Send:
リバーブ、ディレイ等に送る
```

### Mixer の便利機能

```
グループ化:

複数トラック選択:
Cmd/Ctrl+G
→ Group Track作成
→ まとめて音量調整

Return Tracks:

Send A/B の行き先:
リバーブ、ディレイ等
共通エフェクト

Master Track:

最終出力:
全トラックの合計
Master でマスタリング

表示/非表示:

Mixer表示:
Cmd+Opt+M (Mac)
Ctrl+Alt+M (Win)

幅調整:
境界線ドラッグ
→ 広く/狭く
```

---

## 5. Transport（トランスポート）

**再生コントロール:**

### Transport構成

```
┌────────────────────────────────────┐
│ ●Rec  ▶Play  ■Stop  ⏺Loop  ⏮⏭  │
│ 128.00 BPM   4/4   ♩= Quarter     │
│ 1.1.1 (Bar.Beat.16th)             │
└────────────────────────────────────┘

各ボタン:

● Record:
録音開始

▶ Play:
再生/停止（Space）

■ Stop:
完全停止

⏺ Loop:
ループ再生 ON/OFF

⏮ Previous / Next ⏭:
前/次のマーカー

BPM:
テンポ設定
(Techno: 125-135)

拍子:
4/4（普通）
3/4、6/8等も可

Position:
現在位置
1.1.1 = 1小節目の1拍目
```

### ショートカット

```
再生/停止:
Space

録音:
F9

最初に戻る:
Home (Win)
Fn+← (Mac)

BPM変更:
数字入力可能
または↑↓キー

Metronome (メトロノーム):
Cmd+U (Mac)
Ctrl+U (Win)
→ クリック音
```

---

## 画面レイアウトのカスタマイズ

**自分好みに:**

### 各セクションの表示/非表示

```
Browser:
Cmd+Opt+B

Info View (下部左):
Cmd+Opt+I
→ ヘルプテキスト表示

Clip/Device View:
Shift+Tab
→ 切り替え

Mixer:
Cmd+Opt+M

In/Out (Session View):
Cmd+Opt+I/O
→ 入出力表示

全画面:
F11 (Win)
Cmd+Ctrl+F (Mac)
```

### ワークスペース設定

```
Session View中心:
Browser: 表示
Session View: 広く
Arrangement View: 隠す

Arrangement View中心:
Arrangement View: 広く
Session View: 隠す
Mixer: 表示

作曲時:
Device View: 表示
→ 音源操作

ミックス時:
Mixer: 広く表示
→ 音量バランス調整

保存:
ウィンドウ配置は
プロジェクトごとに保存される
```

---

## Info View（インフォビュー）

**ヘルプ機能:**

```
Info View とは:

画面左下:
マウスカーソルを当てた箇所の
説明が表示される

表示方法:
Cmd+Opt+I (Mac)
Ctrl+Alt+I (Win)

使い方:

分からないボタン:
マウスを当てる
→ 説明が出る

例:
「Warp」にマウス当てる
→ 「オーディオのタイミングを
   プロジェクトのテンポに合わせます」

初心者:
常時表示推奨

慣れたら:
非表示でOK
```

---

## 実践: 画面を触ってみる

**30分の探索:**

### Step 1: Ableton Live起動 (5分)

```
1. Ableton Live 起動

2. 画面確認:
   左: Browser
   中央: Session View
   下: Clip View
   右: Mixer
   一番下: Transport

3. 各セクションをクリック:
   どこが反応するか確認
```

### Step 2: Browser探索 (10分)

```
1. Browser開く:
   Cmd+Opt+B

2. Categories > Sounds:
   クリック

3. Bass カテゴリ:
   音色をクリック
   → プレビュー再生

4. Drums > Kicks:
   キック試聴

5. 気に入った音:
   ドラッグ&ドロップ
   → Session Viewに配置
```

### Step 3: Session Viewで遊ぶ (10分)

```
1. クリップ再生:
   配置したクリップをクリック

2. Stop:
   ■ボタン

3. 複数クリップ:
   別のトラックにも音色配置
   → 同時再生

4. Scene:
   右の▶ボタン
   → 全トラック同時再生
```

### Step 4: Transport操作 (5分)

```
1. BPM変更:
   128 → 125に変更

2. 再生:
   Space

3. 停止:
   Space

4. Metronome:
   Cmd+U
   → クリック音確認
```

---

## よくある質問

### Q1: 画面が複雑すぎて覚えられない

**A:** 1週間で慣れる

```
初日:
何が何だか分からない

3日目:
Browserの場所は覚えた

7日目:
だいたい分かってきた

14日目:
普通に使える

30日目:
目をつぶっても操作可能

コツ:

毎日触る:
30分でOK

1つずつ:
今日はBrowserだけ
明日はSession Viewだけ

実践:
読むだけではダメ
触って覚える
```

### Q2: Session ViewとArrangement View、どっちを使うべき？

**A:** 最初はSession View

```
Session View:
アイデア出しに最適
8小節ループ作る

Arrangement View:
曲として完成させる
イントロ〜アウトロ

流れ:

Week 1-4:
Session Viewのみ

Week 5-:
Arrangement Viewも

曲作り:
1. Session Viewでループ作成
2. Arrangement Viewでアレンジ
3. 書き出し
```

### Q3: Rekordboxと似ている部分は？

**A:** 多い

```
対応表:

Rekordbox → Ableton Live

Deck A/B → Track 1/2
Waveform → Clip View
Hot Cue → Locator
Loop → Clip Loop
EQ → EQ Three (Device)
Effects → Audio Effects
Playlist → Session View Scene

慣れやすい:
DJ経験者は有利
```

---

## まとめ

### インターフェイス5つのセクション

```
1. Browser: サウンド検索
2. Session/Arrangement View: 曲作り
3. Clip/Device View: 詳細編集
4. Mixer: 音量調整
5. Transport: 再生コントロール
```

### 重要なショートカット

```
Browser: Cmd+Opt+B
Info View: Cmd+Opt+I
Mixer: Cmd+Opt+M
Session ⇔ Arrangement: Tab
Clip ⇔ Device: Shift+Tab
再生/停止: Space
```

### チェックリスト

```
□ 5つのセクションを確認
□ Browser で音色検索
□ Session View でクリップ再生
□ Mixer で音量調整
□ Transport でBPM変更
□ Info View を表示
□ 30分間自由に探索
```

---

**次は:** [Session View vs Arrangement View](./session-vs-arrangement.md) - 2つのビューを深掘り
