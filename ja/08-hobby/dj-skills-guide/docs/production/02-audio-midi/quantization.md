# Quantize（クオンタイズ）

タイミングを完璧に。グリッド補正からSwing、Grooveまで、リズムの全てをマスターします。

## この章で学ぶこと

- Quantizeとは何か
- Quantize設定（1/4, 1/8, 1/16）
- パーシャルQuantize（50%, 75%）
- Swing（グルーヴ感）
- Groove Pool完全活用
- ヒューマナイズテクニック
- Audio Quantize


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [MIDI基礎](./midi-basics.md) の内容を理解していること

---

## なぜQuantizeが重要なのか

**プロとアマの差:**

```
初心者の打ち込み:

ズレまくり:
手打ちMIDI
タイミングバラバラ

結果:
素人っぽい
グルーヴがない

プロの打ち込み:

Quantize使用:
完璧なタイミング

適度なズレ:
50-75% Quantize
人間らしさ

結果:
プロの音
グルーヴ感

統計:

Quantize使用:
プロの100%

100% Quantize:
Kick, Bass, Snare

50-75% Quantize:
Hi-Hat, Percussion
```

---

## Quantizeとは

**グリッド補正:**

### 基本概念

```
Quantizeの仕組み:

ズレたノート:
■  ■ ■   ■
↓  ↓ ↓   ↓

グリッド:
|  |  |  |

Quantize後:
■  ■  ■  ■
完璧に整列

効果:

タイミング:
ズレを修正

グリッド:
拍・小節に合わせる

音楽的:
正確なリズム

用途:

MIDI入力後:
手弾きの補正

打ち込み後:
クリックミス修正
```

### Quantize対象

```
MIDIノート:
Note On の位置
Note Off は通常そのまま

Audio:
Warp Marker の位置
トランジェント

Automation:
ブレイクポイント
```

---

## Quantize設定

**Grid値の選択:**

### Grid設定

```
Transport > Quantization:

┌──────────────┐
│ None         │ ← Quantizeなし
│ 1 Bar        │ ← 1小節
│ 1/2          │ ← 2分音符
│ 1/4          │ ← 4分音符（推奨）
│ 1/8          │ ← 8分音符
│ 1/16         │ ← 16分音符
│ 1/32         │ ← 32分音符
└──────────────┘

Techno/House推奨:

Kick: 1/4
Bass: 1/4
Snare: 1/4
Hi-Hat: 1/16
Percussion: 1/16

細かすぎる設定:

1/32:
細かすぎ
通常不要

1/64:
ほぼ使わない

粗すぎる設定:

1 Bar:
粗すぎ
ほぼ使わない
```

### Record Quantization

```
Preferences > Record Warp Launch:

MIDI Record Quantization:

┌──────────────┐
│ None         │
│ 1/16         │ ← 推奨
└──────────────┘

効果:
録音中にリアルタイムQuantize

メリット:
録音しながら補正
ズレなし

デメリット:
機械的になる可能性

推奨:
1/16 または None

Noneの場合:
後で手動Quantize
```

---

## Quantize実行

**操作方法:**

### MIDIノートQuantize

```
方法1: ショートカット (推奨)

1. MIDIノート選択
   Cmd+A (全選択)

2. Cmd+U (Quantize)

3. 即座にQuantize

方法2: メニュー

1. ノート選択

2. Edit > Quantize Settings

3. Grid設定

4. OK

方法3: 右クリック

1. ノート選択

2. 右クリック > Quantize

3. 設定ダイアログ

4. Apply

Live録音後:

1. 録音完了

2. Clip選択

3. Cmd+U

4. 完璧なタイミング
```

### Quantize Settings詳細

```
ダイアログ:

┌────────────────────────┐
│ Quantize Settings      │
├────────────────────────┤
│ Quantize To: 1/16      │ ← Grid
│                        │
│ Amount: 100%           │ ← 補正量
│                        │
│ Quantize:              │
│ ☑ Start               │ ← Note On
│ ☐ End                 │ ← Note Off
│                        │
│ ☐ Triplets            │ ← 3連符
└────────────────────────┘

Amount (重要):

100%:
完全にグリッドに合わせる

75%:
75%だけ補正
25%は元のまま

50%:
半分だけ補正
人間らしさ残る

25%:
わずかに補正

推奨:

Kick: 100%
Bass: 100%
Hi-Hat: 75%
Percussion: 50%
```

---

## パーシャルQuantize

**人間らしさを残す:**

### Amount設定

```
100% Quantize:

完璧:
グリッドに完全一致

機械的:
生命感がない

用途:
Kick, Bass, Snare

50-75% Quantize:

適度なズレ:
元の25-50%残る

人間的:
グルーヴ感

用途:
Hi-Hat, Shaker, Melody

実験:

同じパターン:
100% vs 75% vs 50%

聴き比べ:
違いを体感

75%が最適:
多くの場合
```

### ヒューマナイズ

```
ランダム化:

1. ノート選択

2. 右クリック
   → Randomize

3. Velocity: 20%
   Position: 10%

効果:
機械的 → 人間的

Velocity:
強弱のバラつき

Position:
タイミングのバラつき

注意:
やりすぎ注意
20-30%が限度
```

---

## Swing（スウィング）

**グルーヴ感の秘密:**

### Swingとは

```
ストレート (Swing 0%):

|  ■  |  ■  |  ■  |  ■  |
均等

Swing 50%:

|  ■   |■  ■   |■  ■   |■
後ろにズレる

Swing 66%:

|  ■    | ■ ■    | ■
さらにズレる

効果:
跳ねる感じ
グルーヴ

ジャンル別:

Techno: 0-10% (ほぼストレート)
House: 10-20%
Hip Hop: 30-50%
Jazz: 50-66%
```

### Swing設定

```
Groove Pool使用:

1. Browser > Groove Pool

2. Swing 項目選択:
   MPC Swing 16
   Swing 62
   等

3. ドラッグ:
   Clip にドロップ

4. Clip View > Groove:
   Amount調整

手動Swing:

Quantize Settings:
☑ Triplets
→ 3連符ベース

または:

偶数番目ノートを手動で後ろにずらす
```

---

## Groove Pool

**Abletonの秘密兵器:**

### Groove Poolとは

```
定義:
タイミングとベロシティのテンプレート

用途:
グルーヴをコピー
別のClipに適用

場所:
Browser > Groove Pool

種類:

MPC Grooves:
AKAI MPC由来
Hip Hop的

Swing Grooves:
様々なSwing量

Logic Grooves:
Logic Pro互換

カスタム:
自分で作成可能
```

### Groove適用

```
基本操作:

1. Browser > Groove Pool

2. Groove選択:
   例: "MPC 16 Swing-60"

3. Clipにドロップ:
   ドラッグ&ドロップ

4. Clip View > Groove:

   Amount: 100%
   Random: 0%
   Velocity: 0%
   Timing: 0%

5. 再生:
   グルーヴ確認

パラメーター:

Amount:
Groove適用量
0% = なし
100% = 完全適用

Random:
ランダム化
0-100%

Velocity:
ベロシティ変化
-100 〜 +100%

Timing:
タイミング調整
-100 〜 +100%
```

### 複数Clipに適用

```
統一されたグルーヴ:

1. Groove選択

2. 全ドラムClipにドロップ:
   Kick, Snare, Hi-Hat

3. Amount統一:
   全て 75%

効果:
全体が一体化
グルーヴ感UP

Master Groove:

1つのGrooveを:
プロジェクト全体に

統一感:
全トラックが同じグルーヴ
```

---

## Audio Quantize

**Audioのタイミング修正:**

### Warp Markerで補正

```
準備:

Audio Clip:
ドラムループ等

Warp: On

手順:

1. Clip View

2. Warp Marker確認:
   自動配置されている

3. Quantize to Grid:
   右クリック > Quantize Warp Markers

4. Grid選択:
   1/16

5. 適用:
   全Warp Markerが移動

効果:
タイミングが完璧に

注意:
音質変化の可能性
```

---

## 実践: Hi-Hatパターン作成

**Quantize活用:**

### Step 1: 打ち込み (5分)

```
1. MIDI Track

2. Hi-Hat音源

3. 16分音符で打ち込み:
   1小節分
   手動またはMIDIキーボード

4. タイミング:
   適当でOK（後でQuantize）

5. ベロシティ:
   バラバラでOK
```

### Step 2: Quantize (2分)

```
1. 全ノート選択:
   Cmd+A

2. Quantize:
   Cmd+U

3. Settings:
   1/16
   Amount: 75%

4. 確認:
   Space で再生
```

### Step 3: Swing追加 (3分)

```
1. Groove Pool:
   Browser > Grooves

2. Swing選択:
   "Swing 62"

3. Clipにドロップ

4. Amount調整:
   50%

5. 再生:
   グルーヴ確認
```

### Step 4: ヒューマナイズ (5分)

```
1. ノート選択

2. 右クリック:
   Randomize

3. Velocity: 25%
   Position: 0% (Groove使用のため)

4. 確認:
   自然な強弱

5. 保存:
   Cmd+S
```

---

## よくある質問

### Q1: 100% Quantizeは悪い？

**A:** 楽器による

```
100%推奨:

Kick: 絶対100%
Bass: 100%
Snare: 100%

理由:
土台は完璧に

75%推奨:

Hi-Hat: 75%
Percussion: 75%
Shaker: 50%

理由:
グルーヴ感

メロディ:

Lead: 50-75%
Chord: 75-100%

ケースバイケース
```

### Q2: Swingの適量は？

**A:** ジャンル次第

```
Techno: 0-10%
ほぼストレート
機械的でOK

House: 10-20%
わずかなSwing
グルーヴ

Hip Hop: 30-50%
明確なSwing
跳ねる

Jazz: 50-66%
強いSwing
3連符的

実験:
色々試す
耳で判断
```

### Q3: Groove Poolが分からない

**A:** まず無視でOK

```
優先順位:

初心者:
基本Quantize (Cmd+U)
Groove Pool 不要

中級者:
Swing 試す
Groove Pool 使い始める

上級者:
カスタムGroove作成

最初の1ヶ月:
Cmd+U だけで十分

2ヶ月目以降:
Groove Pool 試す
```

---

## まとめ

### Quantize基本

```
□ Cmd+U で即Quantize
□ Grid: 1/16 (推奨)
□ Amount: 75% (装飾音)
□ Amount: 100% (土台)
□ Swing: ジャンル次第
```

### 楽器別設定

```
Kick: 100%, 1/4
Bass: 100%, 1/4
Snare: 100%, 1/4
Hi-Hat: 75%, 1/16
Percussion: 50-75%, 1/16
Lead: 50-75%, 1/16
```

### Groove活用

```
□ Groove Pool試す
□ Swing追加
□ Amount調整
□ 複数Clipに統一
□ カスタムGroove作成（上級）
```

---

**次は:** [録音テクニック](./recording-techniques.md) - Audio/MIDI完璧な録音

---

## 上級編: Quantizeの理論的基礎

### 音楽理論とQuantizeの関係

Quantize（クオンタイズ）は単なるタイミング補正ツールではなく、西洋音楽理論における「拍節構造」（Metrical Structure）をデジタル上で具現化したものです。音楽には階層的な拍の構造があり、Quantizeはその構造にノートを整列させる行為そのものです。

```
拍節構造の階層:

Level 4 (最上位): 小節 (Bar)
|                               |
Level 3:         拍 (Beat)
|       |       |       |
Level 2:     8分音符 (Eighth)
| . | . | . | . | . | . | . | . |
Level 1 (最下位): 16分音符 (Sixteenth)
|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|

各Level と Quantize Grid の対応:

Level 4  →  1 Bar
Level 3  →  1/4 (四分音符)
Level 2  →  1/8 (八分音符)
Level 1  →  1/16 (十六分音符)

音楽理論的意味:
- 強拍 (Strong Beat): Level 3 の 1拍目・3拍目
- 弱拍 (Weak Beat): Level 3 の 2拍目・4拍目
- 裏拍 (Off-Beat): Level 2 の偶数位置
- ゴーストノート領域: Level 1 の細分化された位置
```

### 拍子記号とQuantize Gridの関係

```
4/4拍子 (最も一般的):
┌─────────────────────────────────┐
│ 1小節 = 4拍                     │
│ 1/4 Grid → 4つのポイント        │
│ 1/8 Grid → 8つのポイント        │
│ 1/16 Grid → 16のポイント        │
│ 1/32 Grid → 32のポイント        │
└─────────────────────────────────┘

3/4拍子 (ワルツ):
┌─────────────────────────────────┐
│ 1小節 = 3拍                     │
│ 1/4 Grid → 3つのポイント        │
│ 1/8 Grid → 6つのポイント        │
│ 1/16 Grid → 12のポイント        │
└─────────────────────────────────┘

6/8拍子 (複合拍子):
┌─────────────────────────────────┐
│ 1小節 = 6つの8分音符            │
│ 実質2拍 (付点四分音符×2)        │
│ 1/8 Grid → 6つのポイント        │
│ 1/16 Grid → 12のポイント        │
│                                 │
│ 注意: Triplet Gridが有効        │
└─────────────────────────────────┘

5/4拍子 (変拍子):
┌─────────────────────────────────┐
│ 1小節 = 5拍                     │
│ DAWでの設定:                    │
│   Time Signature を 5/4 に変更  │
│   Grid は通常通り 1/16 使用     │
│   Quantize ポイント = 20個/小節 │
└─────────────────────────────────┘
```

### 3連符 (Triplet) Quantize の理論

```
通常の2分割系:
|   ■   ■   |   ■   ■   |
  1/8  1/8    1/8  1/8

3連符 (Triplet):
|  ■  ■  ■  |  ■  ■  ■  |
  1/8T 1/8T 1/8T

数学的関係:
- 通常の1/8: 1拍を2等分 → 各 = 拍の50%
- 1/8 Triplet: 1拍を3等分 → 各 = 拍の33.3%
- 通常の1/16: 1拍を4等分 → 各 = 拍の25%
- 1/16 Triplet: 1拍を6等分 → 各 = 拍の16.7%

Ableton での Triplet 設定:
┌────────────────────────┐
│ Quantize Settings      │
│                        │
│ ☑ Triplets            │ ← チェックON
│                        │
│ Quantize To: 1/8      │
│ → 実質 1/8T に変換     │
└────────────────────────┘

Triplet Grid が必要なジャンル:
- Shuffle系 (6/8的なフィール)
- ブルース
- Jazz
- Swing House
- Future Bass (部分的に)
```

---

## 上級編: Grid設定の高度な使い分け

### パート別Grid設定の完全マトリクス

```
楽器/パート別 最適Grid設定:

┌──────────────┬────────┬────────┬──────────┬────────────┐
│ パート       │ Grid   │ Amount │ Swing    │ 備考       │
├──────────────┼────────┼────────┼──────────┼────────────┤
│ Kick         │ 1/4    │ 100%   │ 0%       │ 完全固定   │
│ Snare/Clap   │ 1/4    │ 100%   │ 0%       │ 完全固定   │
│ Closed HH    │ 1/16   │ 70-80% │ 10-30%   │ 跳ね具合   │
│ Open HH      │ 1/8    │ 85%    │ 10-20%   │ アクセント │
│ Ride         │ 1/8    │ 75%    │ 15-25%   │ ジャズ寄り │
│ Percussion   │ 1/16   │ 50-70% │ 20-40%   │ ルーズに   │
│ Shaker       │ 1/16   │ 60%    │ 15-30%   │ 自然に     │
│ Bass (Sub)   │ 1/4    │ 100%   │ 0%       │ Kickと一致 │
│ Bass (Mid)   │ 1/8    │ 90%    │ 5-15%    │ やや自由   │
│ Chord/Pad    │ 1/4    │ 85%    │ 10-20%   │ ゆったり   │
│ Lead Melody  │ 1/16   │ 60-75% │ 10-25%   │ 表情豊か   │
│ Arp          │ 1/16   │ 95%    │ 0-10%    │ 正確に     │
│ Vocal Chop   │ 1/8    │ 80%    │ 15-25%   │ 雰囲気重視 │
│ FX/Riser     │ None   │ -      │ -        │ 自由配置   │
└──────────────┴────────┴────────┴──────────┴────────────┘
```

### 複合Grid テクニック

```
同一パートに複数のGridを使い分ける手法:

例: Hi-Hat パターンの高度な構成

Step 1 - メインパターン (1/16, 100%)
|■ ■ ■ ■|■ ■ ■ ■|■ ■ ■ ■|■ ■ ■ ■|
基本の16分音符パターン → 完全Quantize

Step 2 - ゴーストノート追加 (1/32, 40%)
|■.■.■.■.|■.■.■.■.|■.■.■.■.|■.■.■.■.|
32分音符のゴーストノート → 低Amount、低Velocity

Step 3 - アクセント追加 (1/4, 100%)
|●   ●   |●   ●   |●   ●   |●   ●   |
4分音符のアクセント → 高Velocity

結合結果:
|●.■.■.■.|●.■.■.■.|●.■.■.■.|●.■.■.■.|
多層的なHi-Hatパターン完成

実装手順 (Ableton):
1. 1つのClip内で各レイヤーを別々にQuantize
2. まずアクセントノートを選択 → 1/4, 100%
3. 次にメインノートを選択 → 1/16, 100%
4. 最後にゴーストノートを選択 → 1/32, 40%
5. Velocity で各レイヤーを差別化
   アクセント: 110-127
   メイン: 70-90
   ゴースト: 20-45
```

### Grid切り替えの判断基準

```
曲の展開に応じたGrid変更:

イントロ (1-16小節):
├── Grid: 1/4 or 1/8
├── Amount: 100%
├── 理由: シンプルで正確
└── 要素: Kick + Bass のみ

ビルドアップ (17-32小節):
├── Grid: 1/8 → 1/16 に遷移
├── Amount: 90%
├── 理由: 徐々に密度UP
└── 要素: HH, Perc 追加

ドロップ (33-48小節):
├── Grid: 1/16
├── Amount: 75-85%
├── 理由: 最大の躍動感
└── 要素: フルキット稼働

ブレイクダウン (49-64小節):
├── Grid: 1/4 or None
├── Amount: 50-60%
├── 理由: ルーズで有機的
└── 要素: Percussion 中心
```

---

## 上級編: パーシャルQuantizeの専門テクニック

### Amount値の音楽的意味

```
Amount値と聴覚的印象の関係:

┌────────┬─────────────────────────────────────────┐
│ Amount │ 聴覚的印象                               │
├────────┼─────────────────────────────────────────┤
│ 100%   │ 完全に機械的、正確無比                   │
│        │ → EDM Kick, Bassline に最適              │
│ 90%    │ ほぼ正確だが微かな揺らぎ                 │
│        │ → Snare, 正確なArpeggio                  │
│ 80%    │ タイトだが人間の演奏に近い                │
│        │ → ファンク系ギター、キーボード            │
│ 75%    │ 黄金比率: 正確さとグルーヴの最適バランス  │
│        │ → Hi-Hat, Percussion 全般                 │
│ 60%    │ 明確に揺らぎを感じる                     │
│        │ → Conga, Bongo, Shaker                   │
│ 50%    │ 元の演奏とグリッドの中間                  │
│        │ → ジャズ系フレーズ、ブルース               │
│ 30%    │ ほぼ元の演奏、わずかに補正                │
│        │ → ライブ録音のニュアンス保持               │
│ 10%    │ ほぼ効果なし、微調整用                    │
│        │ → ルバート的フレーズの微修正               │
│ 0%     │ 完全に元のまま (Quantize無効)             │
└────────┴─────────────────────────────────────────┘
```

### 段階的Quantizeテクニック

```
一度に100%ではなく段階的にQuantize:

手法: Iterative Quantization

Round 1: Amount 30%
├── 大きなズレだけ修正
├── 全体の雰囲気を確認
└── Cmd+U (30%) → 再生 → 確認

Round 2: Amount 50%
├── さらにタイトに
├── まだ揺らぎが残る
└── Cmd+U (50%) → 再生 → 確認

Round 3: Amount 75%
├── 十分にタイト
├── 人間らしさも残存
└── Cmd+U (75%) → 再生 → 確認

※注意:
Quantizeは累積する
30% → 50% ではなく、30% → 追加50%
元の位置からの割合ではなく、現在位置からの割合

計算例:
元のズレ: 100 tick
Round 1 (30%): 100 × 0.7 = 70 tick 残り
Round 2 (50%): 70 × 0.5 = 35 tick 残り
Round 3 (75%): 35 × 0.25 = 8.75 tick 残り
→ 最終的に約91%の補正と同等

利点:
- 各段階で確認できる
- やりすぎを防止
- Cmd+Z で1段階戻れる
```

### ノート別Amount設定

```
同一Clip内でノートごとに異なるAmountを適用:

手法: Selective Quantization

Step 1: 全選択 → 100% Quantize
→ 一旦全てをグリッドに合わせる

Step 2: Cmd+Z で全体を戻す

Step 3: パート分離して個別Quantize

Kick ノート選択 → 100%
Snare ノート選択 → 100%
HH ノート選択 → 75%
Perc ノート選択 → 50%

効率的な選択方法 (Ableton):
1. Piano Roll で特定ピッチのノートをクリック
   → 同じピッチの全ノートが選択される
2. Cmd+U で Quantize
3. 次のピッチに移動して繰り返し

Tips:
Drum Rack使用時、各パッドが異なるピッチ:
C1 = Kick → 選択 → 100%
D1 = Snare → 選択 → 100%
F#1 = HH → 選択 → 75%
A#1 = Perc → 選択 → 50%
```

---

## 上級編: Swing/Grooveの音楽理論

### Swingの数学的定義

```
Swing比率 (Swing Ratio) の理論:

ストレートの8分音符:
|  ■     ■  |  ■     ■  |
  50%   50%    50%   50%
1拍を均等に2分割

Swing 比率 2:1 (66%):
|  ■       ■|  ■       ■|
  66.7%  33.3%
1拍を2:1に分割
→ 最も一般的なSwing

Swing 比率 3:1 (75%):
|  ■         ■|  ■         ■|
  75%      25%
1拍を3:1に分割
→ 非常に強いSwing (ハードバップ的)

Swing比率とDAW設定の対応:

┌─────────────┬─────────┬──────────────────┐
│ Swing比率   │ DAW値   │ 音楽スタイル     │
├─────────────┼─────────┼──────────────────┤
│ 1:1 (50%)   │ 0%      │ ストレート       │
│ 1.2:1 (54%) │ 15-20%  │ 軽いバウンス     │
│ 1.5:1 (60%) │ 35-40%  │ シャッフル       │
│ 2:1 (66.7%) │ 55-60%  │ 標準的Swing      │
│ 2.5:1 (71%) │ 65-70%  │ 重いSwing        │
│ 3:1 (75%)   │ 80%     │ ハードSwing      │
│ 4:1 (80%)   │ 90%     │ 極端なSwing      │
└─────────────┴─────────┴──────────────────┘

注: DAW設定値はソフトウェアにより異なる
Ableton, FL Studio, Logic 等で微妙に計算が違う
```

### Swingの歴史と文化的背景

```
Swingの起源:

1920-30年代:
├── ニューオーリンズ・ジャズ
├── スウィング・ジャズ時代
├── 均等な8分音符を「跳ねさせる」演奏法
└── Duke Ellington: "It Don't Mean a Thing
    (If It Ain't Got That Swing)"

1940-50年代:
├── ビバップ
├── より複雑なSwing解釈
├── Charlie Parker: 不均等なSwing比率を自在に操作
└── アップテンポではSwingが緩和される傾向

1970-80年代 (電子音楽):
├── ドラムマシンの登場 (Roland TR-808, Linn LM-1)
├── Roger Linn がShuffle機能を発明
├── MPC のSwing = Hip Hopのグルーヴの基盤
└── "MPC Swing" が業界標準に

1990-2000年代:
├── J Dilla のドランクビート
├── 意図的な「ズレ」の美学
├── グリッドからの逸脱をアートとして確立
└── 現代ビートメイキングの基礎

2010年代以降:
├── DAW内蔵Groove機能の充実
├── Ableton Groove Pool
├── FL Studio Swing ノブ
├── プリセットGrooveの普及
└── ジャンルを超えたSwing活用
```

### Swing適用の実践的判断フロー

```
Swing量の決定フローチャート:

Start: ジャンルは何か？
│
├── Techno / Industrial
│   └── Swing: 0-5%
│       ├── 機械的な正確さが美学
│       ├── Kick は絶対ストレート
│       └── HH にわずかな Swing で微動感
│
├── House / Disco
│   └── Swing: 10-25%
│       ├── 4つ打ちはストレート
│       ├── HH, Percussion に Swing
│       └── ファンク的グルーヴを意識
│
├── Hip Hop / R&B
│   └── Swing: 25-55%
│       ├── MPC Swing が定番
│       ├── ドラム全体に適用
│       └── "跳ねる" ビート
│
├── UK Garage / 2-Step
│   └── Swing: 20-40%
│       ├── シャッフル的HH
│       ├── シンコペーション多用
│       └── 不規則なキックパターン
│
├── Drum & Bass / Jungle
│   └── Swing: 5-15%
│       ├── ブレイクビーツ由来
│       ├── 高速BPMのため控えめ
│       └── 元のBreaks のGroove を活かす
│
├── Lo-Fi / Chillhop
│   └── Swing: 40-60%
│       ├── J Dilla 的ドランクビート
│       ├── 大きなSwing + 低Quantize Amount
│       └── 意図的な「酔い」感
│
└── Jazz / Fusion
    └── Swing: 50-66%
        ├── 3連符ベースの伝統的Swing
        ├── テンポに応じて変動
        └── 速いテンポ → Swing 弱め
            遅いテンポ → Swing 強め
```

---

## 上級編: Groove Poolの完全マスターガイド

### Groove Pool の内部構造

```
Grooveファイル (.agr) の構成:

┌─────────────────────────────┐
│ Groove Template              │
├─────────────────────────────┤
│ Timing Map:                  │
│   各グリッドポイントの       │
│   タイミングオフセット値     │
│                              │
│ Velocity Map:                │
│   各グリッドポイントの       │
│   ベロシティオフセット値     │
│                              │
│ Resolution:                  │
│   16th / 8th / 32nd         │
│                              │
│ Metadata:                    │
│   名前、ソース情報           │
└─────────────────────────────┘

Timing Map の例 (16ステップ):

ステップ:  1   2   3   4   5   6   7   8
Offset:   +0  -5  +0  +12 +0  -3  +0  +8
(tick)

ステップ:  9  10  11  12  13  14  15  16
Offset:   +0  -5  +0  +15 +0  -2  +0  +10
(tick)

正の値 = 遅らせる (Behind the beat)
負の値 = 前に出す (Ahead of the beat)
0     = グリッド上 (On the grid)
```

### Ableton付属Groove一覧と特徴

```
MPC Grooves:
┌────────────────────────┬──────────────────────────────┐
│ 名前                   │ 特徴                         │
├────────────────────────┼──────────────────────────────┤
│ MPC 8 Swing-50         │ 8分Swing 50%, 軽いバウンス   │
│ MPC 8 Swing-54         │ 8分Swing 54%, 軽めHip Hop   │
│ MPC 8 Swing-58         │ 8分Swing 58%, 中程度         │
│ MPC 8 Swing-62         │ 8分Swing 62%, 定番Hip Hop   │
│ MPC 8 Swing-66         │ 8分Swing 66%, 強いSwing     │
│ MPC 8 Swing-70         │ 8分Swing 70%, 非常に強い    │
│ MPC 16 Swing-50        │ 16分Swing 50%, 軽いバウンス │
│ MPC 16 Swing-54        │ 16分Swing 54%, 軽め         │
│ MPC 16 Swing-58        │ 16分Swing 58%, House向き    │
│ MPC 16 Swing-62        │ 16分Swing 62%, 万能         │
│ MPC 16 Swing-66        │ 16分Swing 66%, Funk向き     │
│ MPC 16 Swing-70        │ 16分Swing 70%, 強烈         │
└────────────────────────┴──────────────────────────────┘

Swing Grooves (Ableton内蔵):
┌────────────────────────┬──────────────────────────────┐
│ Swing 8-XX             │ 8分音符ベースのSwing         │
│ Swing 16-XX            │ 16分音符ベースのSwing        │
└────────────────────────┴──────────────────────────────┘

Logic Grooves:
├── Logic系DAWとの互換性
├── Pop / Rock 向きのGroove
└── より控えめなSwing量

推奨Groove (ジャンル別):
Techno     → Swing 16-52 (Amount 30-50%)
Deep House → MPC 16 Swing-58 (Amount 60-80%)
Hip Hop    → MPC 16 Swing-62 (Amount 80-100%)
Lo-Fi      → MPC 8 Swing-66 (Amount 70-90%)
Funk       → MPC 16 Swing-66 (Amount 90-100%)
```

### Groove Pool パラメーター詳細解説

```
4つのパラメーター相互作用:

┌─────────────────────────────────────────┐
│         Groove Pool Parameters           │
├──────────┬──────────────────────────────┤
│ Base     │ Groove のグリッド解像度       │
│          │ 1/8 or 1/16                  │
│          │ → Groove 内のどのグリッドに   │
│          │   ノートを吸着させるか       │
├──────────┼──────────────────────────────┤
│ Quantize │ グリッド吸着量               │
│          │ 0% = Groove のみ適用         │
│          │ 100% = まずグリッドに合わせ  │
│          │   てからGroove 適用          │
│          │ → Quantize ≠ Amount          │
│          │   Quantize は前処理          │
│          │   Amount は Groove の強度    │
├──────────┼──────────────────────────────┤
│ Timing   │ Groove のタイミング適用量     │
│          │ 0% = タイミング変更なし       │
│          │ 100% = テンプレート通り       │
│          │ 負の値 = 逆方向に適用        │
├──────────┼──────────────────────────────┤
│ Random   │ ランダムなタイミングの揺らぎ │
│          │ 0% = 揺らぎなし              │
│          │ 100% = 最大の揺らぎ          │
│          │ → 毎回再生ごとに変わる       │
├──────────┼──────────────────────────────┤
│ Velocity │ ベロシティへの影響量          │
│          │ 0% = ベロシティ変更なし       │
│          │ 100% = テンプレートのVel適用  │
│          │ 負の値 = 強弱を反転          │
├──────────┼──────────────────────────────┤
│ Amount   │ Groove 全体の適用量           │
│          │ 0% = Groove 無効             │
│          │ 100% = 完全適用              │
│          │ → Timing と Velocity の      │
│          │   マスターコントロール       │
└──────────┴──────────────────────────────┘

パラメーター設定例:

タイトだが人間的:
├── Quantize: 80%
├── Timing: 60%
├── Random: 10%
├── Velocity: 40%
└── Amount: 75%

ルーズでドランク:
├── Quantize: 20%
├── Timing: 90%
├── Random: 25%
├── Velocity: 70%
└── Amount: 100%

微かなバウンス:
├── Quantize: 100%
├── Timing: 30%
├── Random: 5%
├── Velocity: 20%
└── Amount: 50%
```

---

## 上級編: カスタムGroove作成法

### MIDIクリップからGrooveを抽出

```
手順: Extract Groove

1. お気に入りのドラムパターンを用意
   (MIDIクリップまたはAudioクリップ)

2. Clip を右クリック

3. "Extract Groove(s)" を選択

4. Groove Pool に自動追加される

5. 他の Clip にドラッグ&ドロップで適用

活用例:

例1: ヴィンテージレコードからGrooveを抽出
├── 好きな曲のドラムBreakをサンプリング
├── Audio Clip として配置
├── Warp Marker が自動検出
├── Extract Groove
├── → そのレコード固有のGrooveが抽出
└── → 自分の打ち込みに適用

例2: 自分の演奏からGrooveを抽出
├── MIDIキーボードでドラムを演奏
├── 録音 (Quantize OFF)
├── 人間らしいタイミングが記録される
├── Extract Groove
├── → 自分の「クセ」がGrooveテンプレートに
└── → 他の楽器にも同じクセを適用

例3: 有名ドラマーのGrooveを再現
├── ライブ演奏のドラム録音を入手
├── Audio Clip として読み込み
├── Warp + Extract Groove
├── → そのドラマーのタイム感を抽出
└── → プログラミングしたドラムに適用
```

### カスタムGrooveの編集

```
抽出後の Groove 微調整:

1. Groove Pool 内の Groove をダブルクリック
   → MIDI Clip として展開される

2. Piano Roll で編集:
   ├── ノートの位置 = タイミングオフセット
   ├── ノートのVelocity = ベロシティマップ
   └── ノートの長さ = (通常影響なし)

3. 編集例:
   ├── 特定ステップのSwingを強調
   ├── ゴーストノートのVelocityを調整
   └── 不要なオフセットを0に戻す

4. Ctrl+S で保存
   → User Library に保存される

保存場所:
User Library > Grooves > [カスタム名].agr

命名規則の推奨:
[ジャンル]_[BPM]_[特徴]_[バージョン]
例: HipHop_90_MPC_v2.agr
例: House_124_Shuffle_v1.agr
例: Jungle_170_Break_v3.agr
```

### レイヤードGrooveテクニック

```
複数のGrooveを組み合わせる高度な手法:

手法: 各楽器に異なるGrooveを適用

Kick    → MPC 16 Swing-54 (Amount 40%)
Snare   → MPC 16 Swing-54 (Amount 40%)
HH      → MPC 16 Swing-62 (Amount 80%)
Perc    → カスタムGroove (Amount 60%)
Shaker  → Swing 16-58 (Amount 70%)

効果:
├── 各パートが微妙に異なるGrooveを持つ
├── 有機的な「ズレ」が生まれる
├── 人間のバンド演奏に近い
└── 一体感がありつつも生命感がある

注意点:
├── Kick と Snare は同じGrooveにする
│   → 土台の一体感を保つ
├── HH と Perc は異なるGrooveでもOK
│   → 表面のテクスチャーを豊かに
└── 全てのAmountを揃えすぎない
    → 差をつけることで立体感
```

---

## 上級編: Audio Quantizeの高度な手法

### トランジェント検出の最適化

```
Audio Quantize の精度を上げるための前処理:

Step 1: Warp Mode の選択
┌──────────────┬──────────────────────────────┐
│ Warp Mode    │ Audio Quantize 適性           │
├──────────────┼──────────────────────────────┤
│ Beats        │ ★★★★★ ドラム/パーカッション │
│ Tones        │ ★★★☆☆ メロディ楽器          │
│ Texture      │ ★★☆☆☆ パッド/アンビエント   │
│ Re-Pitch     │ ★★★★☆ 短いサンプル          │
│ Complex      │ ★★★☆☆ 複合素材             │
│ Complex Pro  │ ★★★★☆ 高品質だがCPU重い    │
└──────────────┴──────────────────────────────┘

Step 2: トランジェント感度の調整
Clip View > Sample Box:
├── Transient Sensitivity スライダー
├── 低い値: 大きなトランジェントのみ検出
│   → Kick, Snare 等の主要ヒットのみ
├── 高い値: 細かなトランジェントも検出
│   → ゴーストノート、HH のディテール
└── 推奨: まず中間値から始めて調整

Step 3: 手動 Warp Marker の配置
├── 自動検出が不正確な場合
├── ダブルクリックで Warp Marker を手動追加
├── 正確なトランジェント位置に配置
└── 不要な Warp Marker はダブルクリックで削除
```

### Audio Quantize のワークフロー

```
プロのAudio Quantizeワークフロー:

Phase 1: 準備
├── Audio Clip を選択
├── Warp: ON 確認
├── Warp Mode: Beats (ドラムの場合)
├── Transient 感度を調整
└── Warp Marker の自動配置を確認

Phase 2: 確認と修正
├── 全 Warp Marker を目視確認
├── 誤検出を削除
├── 見落としを手動追加
└── 特に重要: Kick, Snare の位置を正確に

Phase 3: Quantize 実行
├── 全 Warp Marker 選択 (Cmd+A)
├── 右クリック > Quantize
├── Grid: 1/16 (通常)
├── Amount: 開始は 50% から
└── 再生して確認

Phase 4: 微調整
├── 不自然な箇所を個別に修正
├── 特定の Warp Marker をドラッグ
├── Amount を上下して最適値を探す
└── A/B比較 (Warp ON/OFF で切り替え)

Phase 5: 仕上げ
├── 音質劣化がないか確認
├── 必要なら Warp Mode 変更
├── Consolidate (Cmd+J) で確定
└── 元の Audio は保持 (非破壊)
```

### マルチトラック Audio Quantize

```
複数のオーディオトラックを同時にQuantize:

シナリオ: ドラムの生録音
├── Track 1: Kick (マイク)
├── Track 2: Snare (マイク)
├── Track 3: OH Left (オーバーヘッド)
├── Track 4: OH Right (オーバーヘッド)
└── Track 5: Room (アンビエント)

問題:
各トラックを個別にQuantizeすると
位相がズレてサウンドが破綻する

解決法 1: マスタートラック基準
├── Kick トラック (最も明確) を Quantize
├── その Warp Marker 情報を他のトラックにコピー
├── 全トラックが同じタイミングで移動
└── 位相関係が保持される

解決法 2: グループ化
├── 全ドラムトラックを Group 化
├── Group の Audio を Bounce
├── Bounce した Audio を Quantize
├── 元のマルチトラックは参考として残す

解決法 3: Elastic Audio (Pro Tools) / Audio Quantize (Logic)
├── 他のDAWではマルチトラック対応機能あり
├── Ableton では手動コピーが必要
└── 将来のアップデートに期待
```

---

## 上級編: ジャンル別Quantize設定の完全ガイド

### Techno

```
Techno Quantize 設定:

BPM: 125-140

基本思想:
機械的な正確さが美学
人間的揺らぎは最小限
グリッドへの忠実さが重要

┌──────────────┬────────┬────────┬─────────┐
│ パート       │ Grid   │ Amount │ Swing   │
├──────────────┼────────┼────────┼─────────┤
│ Kick         │ 1/4    │ 100%   │ 0%      │
│ Clap/Snare   │ 1/4    │ 100%   │ 0%      │
│ Closed HH    │ 1/16   │ 100%   │ 0-5%    │
│ Open HH      │ 1/8    │ 100%   │ 0%      │
│ Ride         │ 1/8    │ 95%    │ 0%      │
│ Perc (Low)   │ 1/16   │ 90%    │ 0-5%    │
│ Perc (High)  │ 1/16   │ 85%    │ 5-10%   │
│ Bassline     │ 1/16   │ 100%   │ 0%      │
│ Synth Lead   │ 1/16   │ 95%    │ 0%      │
│ Pad          │ 1/4    │ 100%   │ 0%      │
│ FX/Noise     │ None   │ -      │ -       │
└──────────────┴────────┴────────┴─────────┘

サブジャンル別:

Minimal Techno:
├── 全パート 100% Quantize
├── Swing: 完全に 0%
└── 究極の機械美

Dub Techno:
├── Kick, Bass: 100%
├── コード/パッド: 90%
├── ディレイ成分: None (自然に)
└── Swing: 0-5%

Industrial Techno:
├── 全パート 100%
├── 意図的なズレは Automation で
└── ノイズ系: Quantize 不要

Melodic Techno:
├── リズム隊: 100%
├── メロディ: 85-95%
├── アルペジオ: 100%
└── 微かな人間味をメロディに
```

### House

```
House Quantize 設定:

BPM: 118-130

基本思想:
4つ打ちの正確さ + グルーヴ感
ファンク/ソウルのDNA
適度なSwingでバウンス感

┌──────────────┬────────┬────────┬──────────┐
│ パート       │ Grid   │ Amount │ Swing    │
├──────────────┼────────┼────────┼──────────┤
│ Kick         │ 1/4    │ 100%   │ 0%       │
│ Clap         │ 1/4    │ 100%   │ 0%       │
│ Closed HH    │ 1/16   │ 80%    │ 15-25%   │
│ Open HH      │ 1/8    │ 90%    │ 10-20%   │
│ Shaker       │ 1/16   │ 70%    │ 20-30%   │
│ Conga/Bongo  │ 1/16   │ 60%    │ 25-35%   │
│ Bassline     │ 1/8    │ 95%    │ 5-15%    │
│ Chord Stab   │ 1/8    │ 85%    │ 10-20%   │
│ Vocal        │ 1/8    │ 70%    │ 15-25%   │
│ Piano/Keys   │ 1/16   │ 75%    │ 15-25%   │
└──────────────┴────────┴────────┴──────────┘

Groove推奨:
MPC 16 Swing-58 (Amount 50-70%)

サブジャンル別:

Deep House:
├── よりルーズ (Amount低め)
├── Swing: 15-25%
├── Perc: 50-70% Amount
└── 有機的でウォーム

Tech House:
├── Techno寄りにタイト
├── Swing: 5-15%
├── HH: 85-95% Amount
└── 機能的かつグルーヴィ

Afro House:
├── Percussionが主役
├── Swing: 20-35%
├── Conga/Djembe: 55-70%
├── ポリリズム的要素
└── MPC Groove がフィット

Soulful House:
├── ファンク/ソウル的グルーヴ
├── Swing: 20-30%
├── Rhodes/Wurli: 70% Amount
├── 生演奏のニュアンス重視
└── Groove Pool + Humanize 併用
```

### Hip Hop / Trap

```
Hip Hop Quantize 設定:

BPM: 75-100 (Hip Hop), 130-170 (Trap, half-time感)

基本思想:
MPC由来のSwingフィール
ルーズさと正確さの共存
ベース/キックの重さが重要

Hip Hop (Boom Bap):
┌──────────────┬────────┬────────┬──────────┐
│ パート       │ Grid   │ Amount │ Swing    │
├──────────────┼────────┼────────┼──────────┤
│ Kick         │ 1/8    │ 85%    │ 30-50%   │
│ Snare        │ 1/4    │ 90%    │ 25-40%   │
│ HH           │ 1/16   │ 70%    │ 35-55%   │
│ Perc         │ 1/16   │ 55%    │ 40-55%   │
│ Bass         │ 1/8    │ 90%    │ 25-40%   │
│ Sample Chop  │ 1/8    │ 75%    │ 30-45%   │
│ Keys         │ 1/8    │ 65%    │ 25-40%   │
└──────────────┴────────┴────────┴──────────┘

Groove推奨: MPC 16 Swing-62 (Amount 80-100%)

Trap:
┌──────────────┬────────┬────────┬──────────┐
│ パート       │ Grid   │ Amount │ Swing    │
├──────────────┼────────┼────────┼──────────┤
│ Kick         │ 1/8    │ 95%    │ 0-10%    │
│ Snare/Clap   │ 1/4    │ 100%   │ 0%       │
│ HH (Main)    │ 1/32   │ 90%    │ 0-5%     │
│ HH (Roll)    │ 1/64   │ 100%   │ 0%       │
│ 808 Bass     │ 1/8    │ 100%   │ 0%       │
│ Perc         │ 1/16   │ 80%    │ 5-15%    │
└──────────────┴────────┴────────┴──────────┘

注: Trap は比較的ストレート
Hi-Hat Roll は機械的に正確
Triplet Grid (1/8T, 1/16T) を多用

Lo-Fi Hip Hop:
├── 全パート Amount を 50-70% に下げる
├── Swing: 45-60%
├── Random: 15-25% 追加
├── MPC 8 Swing-66 が定番
├── J Dilla 的ドランクビート
└── 「完璧でないこと」が美学
```

### Drum & Bass / Jungle

```
Drum & Bass Quantize 設定:

BPM: 160-180

基本思想:
高速BPMでの微細なグルーヴ
ブレイクビーツの有機性
Amen Break 由来のタイム感

┌──────────────┬────────┬────────┬──────────┐
│ パート       │ Grid   │ Amount │ Swing    │
├──────────────┼────────┼────────┼──────────┤
│ Kick         │ 1/8    │ 90%    │ 5-10%    │
│ Snare        │ 1/4    │ 95%    │ 0-5%     │
│ Ghost Snare  │ 1/16   │ 60%    │ 10-20%   │
│ HH           │ 1/16   │ 75%    │ 5-15%    │
│ Ride         │ 1/8    │ 80%    │ 5-10%    │
│ Sub Bass     │ 1/8    │ 100%   │ 0%       │
│ Reese Bass   │ 1/8    │ 95%    │ 0-5%     │
│ Pad          │ 1/2    │ 85%    │ 0%       │
│ Vocal Chop   │ 1/16   │ 70%    │ 10-15%   │
└──────────────┴────────┴────────┴──────────┘

特殊テクニック:
├── Amen Break のGrooveを抽出
│   → Extract Groove from classic Amen sample
│   → 他のドラムパターンに適用
│
├── Ghost Note の活用
│   → Velocity: 20-40
│   → Amount: 50-65%
│   → Swing: 10-20%
│   → 有機的な隙間を埋める
│
└── Half-Time セクション
    → BPM そのままで半分のスピード感
    → Grid: 1/4 に切り替え
    → Swing: 0% (ストレートに)
```

### Future Bass / Chillwave

```
Future Bass Quantize 設定:

BPM: 130-160 (half-time感: 65-80)

基本思想:
ポップ的な明瞭さ
シンセコードの揺らぎ
Sidechain と Quantize の連携

┌──────────────┬────────┬────────┬──────────┐
│ パート       │ Grid   │ Amount │ Swing    │
├──────────────┼────────┼────────┼──────────┤
│ Kick         │ 1/4    │ 100%   │ 0%       │
│ Snare        │ 1/2    │ 100%   │ 0%       │
│ HH           │ 1/16   │ 85%    │ 10-20%   │
│ Perc         │ 1/16   │ 70%    │ 15-25%   │
│ Super Saw    │ 1/8    │ 90%    │ 5-15%    │
│ Chord Stab   │ 1/8    │ 85%    │ 10-20%   │
│ Vocal        │ 1/8    │ 75%    │ 10-20%   │
│ 808 Bass     │ 1/8    │ 100%   │ 0%       │
│ Arp          │ 1/16   │ 95%    │ 5-10%    │
│ Pluck        │ 1/16   │ 90%    │ 10-15%   │
└──────────────┴────────┴────────┴──────────┘

テクニック:
├── コードスタブの「遅れ」感
│   → Amount: 85% で微かに遅れる
│   → サイドチェインと相まってポンピング
│
├── Vocal Chop の Groove
│   → 手動配置が多い
│   → Quantize後にわずかに手動調整
│
└── Drop と Verse で Amount を変える
    Verse: 全体的にルーズ (70-85%)
    Drop: タイトに (90-100%)
```

---

## 上級編: MPC/SP-404等のハードウェアGroove

### MPC (AKAI) のSwingの秘密

```
MPC Swing が特別な理由:

歴史的背景:
├── 1988年: MPC60 発売 (Roger Linn 設計)
├── Swing 機能を初搭載
├── Hip Hop プロデューサーに普及
├── J Dilla, DJ Premier, Pete Rock 等が愛用
└── 「MPC Swing」= Hip Hop グルーヴの代名詞

MPC Swing の仕組み:
┌─────────────────────────────────────────┐
│ 通常の16分音符:                          │
│ |1 e & a|2 e & a|3 e & a|4 e & a|      │
│  ↕ ↕ ↕ ↕                                │
│ 均等間隔                                 │
│                                          │
│ MPC Swing 適用後:                        │
│ |1 e  &a|2 e  &a|3 e  &a|4 e  &a|      │
│         ↑                                │
│     偶数番目(e, a)が後ろにズレる         │
│     奇数番目(1, &)はそのまま             │
│                                          │
│ Swing値の意味:                           │
│ 50% = ストレート (均等)                  │
│ 66% = 2:1 比率 (標準的Swing)            │
│ 75% = 3:1 比率 (強いSwing)             │
└─────────────────────────────────────────┘

MPC モデル別の Swing 特性:

MPC60 (1988):
├── 12bit サンプリング
├── Swing の「温かみ」
├── 低解像度による自然な丸み
└── ヴィンテージ Hip Hop の定番

MPC3000 (1994):
├── 16bit サンプリング
├── より正確な Swing 計算
├── 90年代 Hip Hop の標準
└── Pete Rock, Large Professor

MPC2000/2000XL (1997/2000):
├── コストパフォーマンス重視
├── Swing は MPC60 系を継承
├── 多くのビートメイカーが使用
└── 現在も中古市場で人気

MPC Live/One/X (現行):
├── 全モデルの Swing を再現
├── ソフトウェアベースの Swing
├── プリセット: MPC60, MPC3000 等
└── USB 経由で DAW 連携可能
```

### SP-404 の Groove

```
Roland SP-404 シリーズの特徴:

SP-404 の Swing:
├── MPC とは異なるSwing アルゴリズム
├── より「ルーズ」で「酔った」感覚
├── Lo-Fi ビートの定番
├── J Dilla の影響を受けたスタイル

SP-404 特有のグルーヴ要素:
├── パッド演奏のタイミング
│   → 指で叩くため自然なズレ
│   → これ自体がGrooveになる
│
├── 内蔵エフェクト (Vinyl Sim, Compressor)
│   → 音質がGroove感に影響
│   → ローファイな質感 = グルーヴ感UP
│
└── リサンプリング文化
    → 演奏をリサンプル
    → さらに上から演奏
    → 「ズレのズレ」が重なる
    → 唯一無二の有機的Groove

DAW で SP-404 的 Groove を再現:
1. MIDI キーボードで手弾き (Quantize OFF)
2. Amount 40-60% で軽く Quantize
3. Vinyl Sim 的プラグイン適用 (RC-20, Goodhertz Vinyl)
4. テープサチュレーション追加
5. 微かなピッチ揺れ (Chorus 0.1Hz, Depth 極小)
```

### その他のハードウェアGroove

```
各ハードウェアの Groove 特性:

Roland TR-808:
├── 完全に機械的 (Swing なし)
├── しかしステップ入力の独特なフィール
├── アクセント機能でダイナミクス
├── Shuffle 機能: 偶数ステップを遅らせる
└── DAW再現: 100% Quantize + Accent パターン

Roland TR-909:
├── Shuffle ノブ搭載
├── House / Techno の基盤
├── 909 Shuffle = 独特の跳ね感
├── 0-100% のシャッフル量
└── DAW再現: 1/16 Grid + Swing 10-20%

Elektron Digitakt/Analog Rytm:
├── Micro Timing 機能
├── 各ステップを ±23 tick 調整可能
├── パラメーターロックで動的Swing
├── Conditional Trig でランダム性
└── DAW再現: ノート個別のオフセット

E-mu SP-1200:
├── 12bit / 26.04kHz サンプリング
├── 独特のザラついた質感
├── Hip Hop 黎明期の名機
├── Swing は MPC 系に近い
└── DAW再現: Bitcrusher + MPC Groove

Dave Smith Tempest:
├── アナログドラムシンセ
├── 内蔵 Swing + Humanize
├── Beat Repeat 機能
└── DAW再現: Analog Drum + Swing + Random
```

---

## 上級編: ヒューマナイズの科学

### 人間の演奏のタイミング特性

```
科学的研究から分かっていること:

人間のドラマーのタイミングズレ:

プロドラマー:
├── 平均ズレ: ±5-15 ms
├── 標準偏差: 8-12 ms
├── 特徴: 安定したズレ (consistent deviation)
└── クセ: 個人固有のパターンがある

アマチュアドラマー:
├── 平均ズレ: ±15-40 ms
├── 標準偏差: 20-35 ms
├── 特徴: 不安定 (inconsistent)
└── ランダムなズレ (プロとの決定的違い)

知覚閾値 (Perception Threshold):
├── 人間が「ズレ」と感じる最小値: 約 20-30 ms
├── 「快い揺らぎ」と感じる範囲: 5-20 ms
├── 「不快なズレ」: 30 ms 以上
└── 「同時」と感じる: 5 ms 以内

DAW での tick 換算 (120 BPM, 480 ppq):
├── 1 tick = 約 1.04 ms
├── 5 ms ≈ 5 ticks
├── 15 ms ≈ 14 ticks
├── 30 ms ≈ 29 ticks
└── 人間的揺らぎ = ±5-15 ticks が理想
```

### Ahead/Behind the Beat

```
ビートに対する位置関係:

Behind the Beat (レイドバック):
├── ノートがグリッドより後ろ
├── 「溜める」「遅れる」感覚
├── リラックス、ルーズ、クール
├── 代表: J Dilla, D'Angelo, Erykah Badu
└── DAW設定: Timing Offset +3〜+10 ms

On the Beat:
├── ノートがグリッドぴったり
├── 正確、タイト、ソリッド
├── ダンスミュージックの基本
├── 代表: Techno, EDM全般
└── DAW設定: Offset 0 ms

Ahead of the Beat (プッシュ):
├── ノートがグリッドより前
├── 「前のめり」「攻撃的」感覚
├── エネルギー、ドライブ感
├── 代表: Punk Rock, Speed Metal
└── DAW設定: Timing Offset -3〜-8 ms

パート別の位置設定例 (Deep House):

Kick:    On the Beat (0 ms)
Snare:   Behind +5 ms (ゆったり)
HH:      Behind +3 ms (軽く遅れ)
Bass:    On the Beat (Kickと合わせる)
Rhodes:  Behind +8 ms (最もレイドバック)
Vocal:   Behind +5 ms (自然な遅れ)

Ableton での実装:
1. MIDI Track の Track Delay を使用
   → ミキサー表示の "D" 欄
   → 正の値 = 遅らせる
   → 負の値 = 前に出す
   → 単位: ms

2. または MIDI Clip 内でノートを手動シフト
   → 全選択 → 矢印キーで微調整
```

### ベロシティのヒューマナイズ

```
人間のベロシティパターン:

ドラマーの自然な強弱パターン:

4/4 の Hi-Hat パターン (16分音符):
|強 弱 中 弱|強 弱 中 弱|強 弱 中 弱|強 弱 中 弱|
 ↓ ↓ ↓ ↓  ↓ ↓ ↓ ↓  ↓ ↓ ↓ ↓  ↓ ↓ ↓ ↓
110 50 80 50 110 50 80 50 ...

数値パターン:
Beat 1:  110, 50, 80, 50
Beat 2:  100, 45, 75, 48
Beat 3:  108, 52, 78, 50
Beat 4:  105, 48, 82, 52

特徴:
├── ダウンビート (1,3拍) が最も強い
├── アップビート (2,4拍) がやや弱い
├── 裏拍は常に弱い
├── 完全に同じ値にはならない (±5-10 の揺らぎ)
└── パターン全体が徐々に変化

Ableton での Velocity ヒューマナイズ手順:

方法1: Velocity MIDI Effect
1. MIDI Track に "Velocity" を追加
2. Random: 15-25
3. Out Hi: 127
4. Out Low: 1
→ 自動的にランダムなVelocity揺らぎ

方法2: 手動パターン設定
1. Piano Roll で Velocity レーンを表示
2. ペンツールで描画
3. 強拍 > 弱拍 > 裏拍 の階層を意識
4. 各ノートに ±5-10 のランダム揺らぎを加える

方法3: Groove Pool の Velocity パラメーター
1. Groove 適用後
2. Velocity パラメーター: 30-60%
→ Groove テンプレートの強弱パターンが適用
```

---

## 上級編: マイクロタイミングの概念

### マイクロタイミングとは

```
定義:
グリッドよりも細かい時間単位でのタイミング操作
通常のQuantize Grid (1/16, 1/32) では捉えきれない
数 ms 〜 数十 ms レベルの微調整

マイクロタイミングの単位:
├── Tick: DAWの最小時間単位
│   └── 通常 480 ppq (Pulses Per Quarter note)
│       = 1拍を480分割
│       BPM 120 で 1 tick ≈ 1.04 ms
│
├── ms (ミリ秒): 絶対時間
│   └── BPMに依存しない
│       人間の知覚に直接対応
│
└── % of Grid: グリッド相対値
    └── 1/16 グリッドの 10% = 約 3 ms (BPM 120)

マイクロタイミングが重要な場面:

1. Kick と Bass の関係
   ├── 完全一致: パンチ力最大
   ├── Bass が 5ms 遅れ: 太さ UP
   ├── Bass が 10ms 遅れ: フランジング発生の危険
   └── 推奨: ±3 ms 以内

2. Snare のゴーストノート
   ├── メインSnare: On the Beat
   ├── Ghost: ±5-15 ms のバラつき
   └── グルーヴの「息づかい」を生む

3. Hi-Hat の裏拍
   ├── Straight: グリッド上
   ├── Behind: +3-8 ms でレイドバック
   ├── Ahead: -3-5 ms で前のめり
   └── ジャンルとフィールに応じて選択
```

### マイクロタイミングの実装方法

```
方法1: Track Delay (トラックディレイ)

Ableton Live:
├── ミキサーセクションの "D" ボタン
├── 各トラックに ms 単位でオフセット設定
├── 利点: 非破壊的、いつでも変更可能
└── 欠点: トラック全体に適用 (ノート個別不可)

設定例 (Deep House, 124 BPM):
Track 1 (Kick):     0 ms
Track 2 (Snare):   +4 ms
Track 3 (HH):      +2 ms
Track 4 (Perc):    +6 ms
Track 5 (Bass):     0 ms
Track 6 (Keys):    +8 ms
Track 7 (Vocal):   +5 ms

方法2: ノート個別のオフセット

Ableton Live:
├── Piano Roll でノートを選択
├── 矢印キー (← →) で微調整
│   └── Cmd + ← → で細かく移動
├── Grid を細かく設定 (1/32, Triplet)
└── または Grid OFF (Cmd+4) で自由配置

方法3: MIDI Clip の Nudge

1. Clip 全体を微量シフト
2. Clip の Start Marker を微調整
3. 数 tick 単位の調整が可能

方法4: Max for Live デバイス

"Humanizer" 系デバイス:
├── Timing Randomization
├── Velocity Randomization
├── Note Length Randomization
└── リアルタイムでランダム変動
```

### フラム (Flam) テクニック

```
Flam とは:
2つの打撃をわずかな時間差で鳴らすドラム技法
Quantize の逆 = 意図的なズレの作成

基本的な Flam:

ストレートな打撃:
| ■ |
 ↑ 1つの打撃

Flam:
|■■ |
 ↑↑ 2つの打撃 (10-30 ms 差)
 Grace Note + Main Note

DAW での Flam 実装:

1. メインノートを配置 (例: Snare, C1)
2. 同じピッチのノートをコピー
3. コピーを 10-30 ms 前に配置
4. コピーの Velocity を 40-60% に下げる

Velocity:
Grace Note: 50-70
Main Note: 100-120

Timing差:
タイト Flam: 10-15 ms
通常 Flam: 15-25 ms
ワイド Flam: 25-40 ms

活用シーン:
├── Snare のアクセント
├── Tom Fill のダイナミクス
├── パーカッション の表情付け
└── ドロップの直前 (インパクト)
```

---

## 上級編: プロの制作事例でのQuantize活用

### Daft Punk の制作手法

```
"Random Access Memories" (2013):

アプローチ:
├── 生演奏のレコーディング中心
├── Quantize は最小限
├── むしろ「Quantize しない」ことが重要
├── ハードウェアシンセ + 生ドラム
└── デジタル的完璧さよりアナログの温かみ

"Get Lucky" の分析:
├── ドラム: Omar Hakim (生演奏)
│   └── Quantize: 最小限 (5-10% 程度の微補正)
├── ギター: Nile Rodgers (生演奏)
│   └── Quantize: なし (完全に生演奏のまま)
├── ベース: Nathan East (生演奏)
│   └── Quantize: なし
├── シンセ: DAW での打ち込み
│   └── Quantize: 100% (機械的なシーケンスとして)
└── 生演奏 vs 機械的シーケンスの対比が魅力

教訓:
全てを Quantize する必要はない
生演奏とプログラミングのコントラスト
意図を持った使い分けが重要
```

### J Dilla の革命的アプローチ

```
"Donuts" (2006):

アプローチ:
├── MPC3000 使用
├── 意図的に「ズレた」タイミング
├── Quantize をあえて使わない
├── 「ドランクビート」の確立
└── 完璧なタイミングの否定

特徴的なテクニック:
├── Kick が微妙に遅れる (Behind the Beat)
├── Snare が微妙に前に出る (Ahead of the Beat)
├── HH はさらに不規則
├── 結果: 「酔った」ようなグルーヴ
└── しかし全体としてはグルーヴが成立

DAW で J Dilla 的ビートを作る:
1. MIDIキーボードでリアルタイム録音 (Quantize OFF)
2. そのままの状態をベースにする
3. 極端にズレた箇所のみ手動修正
4. Amount: 20-30% の軽い Quantize
5. MPC 8 Swing-66 (Amount 50%)
6. Random: 15-20%
7. Velocity: 大きなバラつき (Range: 40-120)
8. ローファイ処理 (Bitcrusher, Tape Saturation)

重要な美学:
「完璧にしないことが完璧」
Quantize の量 = アーティストの個性
ルールを知った上で壊す
```

### Skrillex / EDM プロダクション

```
モダン EDM の Quantize 戦略:

アプローチ:
├── 基本: 100% Quantize (全パート)
├── グリッドへの完全一致が前提
├── 人間的揺らぎよりも正確さ
├── グルーヴは別の手法で付加
└── サウンドデザインで個性を出す

Dubstep/Brostep のQuantize:
├── 全リズムパート: 100% Quantize
├── Grid: 1/16 or 1/32
├── Swing: 0% (完全ストレート)
├── ワブルベース: Quantize + Automation
└── ドロップ: 完璧なタイミングが重要

グルーヴの代替手法:
├── Sidechain Compression
│   └── ポンピング効果でグルーヴ感
├── Automation
│   └── フィルター、ボリュームの動的変化
├── サウンドデザイン
│   └── 音色の変化でリズム感
├── Polyrhythm
│   └── 異なる周期のパターンを重ねる
└── FX (Delay, Reverb)
    └── 空間的な揺らぎ

教訓:
Quantize 100% でもグルーヴは作れる
Quantize ≠ 機械的
タイミング以外の要素でグルーヴを生む
```

### Disclosure の制作手法

```
UK Garage / House 的 Quantize:

"Latch" 分析:
├── Kick: 100% Quantize, 1/4 Grid
├── Snare: 100%, Off-beat 配置
├── HH: 80%, 1/16, Swing 15-20%
├── Percussion: 70%, Swing 20%
├── Bass: 95%, 1/8
├── Synth Stab: 85%, 1/8, Swing 15%
└── Vocal: 手動タイミング調整

特徴:
├── UK Garage の Shuffle 的 HH
├── 2-Step 的キックパターン
├── Groove Pool: MPC 16 Swing-58 相当
└── 全体的にバウンシーなフィール

テクニック:
├── HH の偶数番目ノートを手動で後ろに
├── コードスタブを少し遅らせる (+5 ms)
├── ベースは Kick とタイト (0 ms)
└── Vocal は自然なタイミングを活かす
```

---

## 上級編: Quantize トラブルシューティング

### よくある問題と解決策

```
問題1: Quantize後に音が不自然
┌─────────────────────────────────────┐
│ 原因: Amount が高すぎる              │
│ 解決: Amount を 75% に下げる         │
│ または: Cmd+Z → 段階的Quantize      │
└─────────────────────────────────────┘

問題2: Tripletフレーズが崩れる
┌─────────────────────────────────────┐
│ 原因: 通常Grid (非Triplet) を使用  │
│ 解決: Quantize Settings で           │
│       ☑ Triplets をON               │
│ または: Grid を 1/8T, 1/16T に       │
└─────────────────────────────────────┘

問題3: Swing が効かない
┌─────────────────────────────────────┐
│ 原因1: Groove の Amount が 0%       │
│ 解決: Amount を 50-100% に          │
│                                      │
│ 原因2: ノートが既に100% Quantize済  │
│ 解決: Groove の Quantize = 100% に  │
│   → まずグリッドに合わせてから      │
│     Swing を適用                    │
└─────────────────────────────────────┘

問題4: Audio Quantize で音質劣化
┌─────────────────────────────────────┐
│ 原因: Warp Mode が不適切            │
│ 解決:                                │
│   ドラム → Beats mode               │
│   メロディ → Complex Pro            │
│   低品質 → Complex Pro に変更       │
│   Amount を下げる (80% → 60%)       │
└─────────────────────────────────────┘

問題5: Quantize がリセットされる
┌─────────────────────────────────────┐
│ 原因: Groove Pool と Quantize の競合│
│ 解決:                                │
│   1. Groove を Commit               │
│      (右クリック > Commit Groove)   │
│   2. その後 Quantize を実行         │
│   3. Groove は解除済み → 干渉なし   │
└─────────────────────────────────────┘

問題6: MIDI録音時にQuantizeが勝手にかかる
┌─────────────────────────────────────┐
│ 原因: Record Quantize が有効        │
│ 解決:                                │
│   Preferences > Record Warp Launch  │
│   MIDI Record Quantization: None    │
└─────────────────────────────────────┘
```

### Quantize前後の比較方法

```
A/B比較テクニック:

方法1: Undo/Redo
├── Quantize 適用 → 再生して確認
├── Cmd+Z (Undo) → 元に戻して確認
├── Cmd+Shift+Z (Redo) → 再度適用
└── 繰り返して比較

方法2: Clipの複製
├── Clip を Cmd+D で複製
├── 複製にのみ Quantize 適用
├── 交互にミュート/ソロで比較
└── 好みの方を採用

方法3: Groove の Amount でフェード
├── Groove を適用
├── Amount を 0% → 100% まで動かす
├── リアルタイムで変化を確認
├── 最適値を見つける
└── Commit して確定
```

---

## 実践演習10問

### 演習1: 基本的なKickパターンのQuantize

```
課題:
4つ打ち Kick パターンを作成し、完璧に Quantize する

手順:
1. 新規 MIDI Track 作成
2. Drum Rack または Kick サンプル読み込み
3. 4小節の Kick パターンを手弾きで録音
   (意図的にズレを作る)
4. Cmd+A で全選択
5. Cmd+U で Quantize
6. Settings: 1/4, Amount 100%
7. 再生して確認

チェックポイント:
□ 全ての Kick がグリッド上にあるか
□ BPM に合っているか
□ クリックと完全同期しているか

目標達成基準:
全ノートが 1/4 グリッドに完全一致していること
```

### 演習2: パーシャルQuantizeの比較実験

```
課題:
同じ Hi-Hat パターンを異なる Amount で Quantize し、
違いを聴き比べる

手順:
1. 16分音符の Hi-Hat パターンを2小節手弾き録音
2. Clip を4つ複製 (合計5つ)
3. 各 Clip に異なる Amount を適用:
   Clip A: Amount 100%
   Clip B: Amount 75%
   Clip C: Amount 50%
   Clip D: Amount 25%
   Clip E: Amount 0% (元のまま)
4. 順番に再生して比較
5. 最も心地よい Amount をメモ

チェックポイント:
□ 100% と 0% の違いが明確に分かるか
□ 75% がバランス良く感じるか
□ 50% でもグルーヴが感じられるか

考察ポイント:
- どの Amount が最も「プロっぽい」か？
- ジャンルによって好みが変わるか？
```

### 演習3: Swing量の最適値を見つける

```
課題:
House ビートに最適な Swing 量を見つける

手順:
1. 基本的な House ビートを作成:
   Kick: 4つ打ち (1/4, 100%)
   Clap: 2, 4拍 (1/4, 100%)
   HH: 16分音符 (1/16, 90%)

2. HH に Groove Pool から Swing を適用:
   最初: Swing 16-50 (Amount 0%)

3. Amount を段階的に上げる:
   0% → 25% → 50% → 75% → 100%

4. 各設定で再生して体感する

5. 最も「踊れる」設定を見つける

チェックポイント:
□ Swing なし (0%) のサウンドを記憶したか
□ 段階的な変化を体感できたか
□ 自分の好みの値を特定できたか

典型的な回答: House では Amount 40-60% が最適なことが多い
```

### 演習4: MPC Groove の適用

```
課題:
Hip Hop ビートに MPC Groove を適用する

手順:
1. Hip Hop ビートを打ち込み (BPM 90):
   Kick: Boom Bap パターン
   Snare: 2, 4拍
   HH: 16分音符パターン

2. まず 100% Quantize (1/16) を適用

3. Browser > Groove Pool
   "MPC 16 Swing-62" を選択

4. 全 Clip にドラッグ&ドロップ

5. パラメーター調整:
   Amount: 80%
   Timing: 70%
   Random: 10%
   Velocity: 40%

6. 再生して Hip Hop 的グルーヴを確認

チェックポイント:
□ Groove 適用前後で明確な違いがあるか
□ MPC 的な「跳ね」を感じるか
□ Velocity の変化がダイナミクスに寄与しているか
```

### 演習5: カスタムGrooveの作成

```
課題:
お気に入りの楽曲からGrooveを抽出し、自分のビートに適用する

手順:
1. リファレンス曲の Audio を読み込み
   (ドラムがクリアに聴こえる部分)

2. Warp を ON にして正確に合わせる

3. 2-4小節の範囲を選択

4. 右クリック > Extract Groove(s)

5. Groove Pool に追加されたGrooveを確認

6. 自分のビートに適用

7. Amount, Timing, Velocity を微調整

チェックポイント:
□ Groove が正しく抽出されたか
□ 元の曲のフィールが再現されているか
□ パラメーター調整で微調整できたか

応用:
異なるジャンルの Groove を抽出して
ジャンル横断的な組み合わせを試す
(例: Jazz のGroove を Techno に適用)
```

### 演習6: マルチレイヤーQuantize

```
課題:
1つのドラムパターン内で各パートに異なるQuantize設定を適用する

手順:
1. Drum Rack で以下を打ち込み (手弾き):
   C1: Kick
   D1: Snare
   F#1: Closed HH
   A#1: Open HH
   D#1: Conga

2. パート別に選択してQuantize:
   C1 (Kick): 1/4, 100%
   D1 (Snare): 1/4, 100%
   F#1 (CHH): 1/16, 75%
   A#1 (OHH): 1/8, 85%
   D#1 (Conga): 1/16, 55%

3. 再生して全体のバランスを確認

4. 必要に応じて各パートの Amount を微調整

チェックポイント:
□ Kick と Snare は完璧にグリッド上か
□ HH に自然な揺らぎがあるか
□ Conga が最もルーズに感じるか
□ 全体として一体感があるか
```

### 演習7: Audio Quantize 実践

```
課題:
Audio ドラムループの タイミングを Quantize で補正する

手順:
1. Audio ドラムループを読み込み
   (意図的にズレがあるもの、または生ドラム録音)

2. Warp: ON
   Warp Mode: Beats

3. Transient 感度を調整してマーカーを確認

4. 不要な Warp Marker を削除、
   不足分を手動追加

5. 全 Warp Marker を選択

6. 右クリック > Quantize
   Grid: 1/16
   Amount: 60%

7. 再生して確認

8. Amount を調整して最適値を探す

チェックポイント:
□ Warp Marker が正確なトランジェントに配置されているか
□ Quantize 後に音質劣化がないか
□ タイミングが改善されたか
□ 過度な補正で不自然になっていないか
```

### 演習8: ヒューマナイズ専門演習

```
課題:
完全に機械的なパターンを「人間らしく」する

手順:
1. ステップ入力で完璧な16分音符 HH パターン作成
   (4小節, 全ノート Velocity 100, 完璧なタイミング)

2. 再生して「機械的」であることを確認

3. Velocity ヒューマナイズ:
   強拍 (1,3): Velocity 100-120
   弱拍 (2,4): Velocity 80-95
   裏拍全般: Velocity 50-75
   各値に ±10 のランダム揺らぎ

4. タイミング ヒューマナイズ:
   右クリック > Randomize
   Position: 8-12%

5. Groove Pool 適用:
   Swing 16-56, Amount 40%

6. 最終確認:
   「人間が叩いている」ように聴こえるか

チェックポイント:
□ 機械的 → 有機的 に変化したか
□ Velocity のパターンが自然か
□ タイミングの揺らぎが心地よいか
□ やりすぎていないか (ズレすぎ注意)
```

### 演習9: ジャンル横断 Quantize チャレンジ

```
課題:
同じドラムパターンを5つのジャンルに変換する
(Quantize と Swing の設定のみで)

手順:
1. 基本ドラムパターンを作成:
   Kick + Snare + HH + Perc (4小節)
   全て 1/16, 100% Quantize (ストレート)

2. Clip を5つ複製

3. 各 Clip に以下の設定を適用:

   Version A - Techno:
   全パート 100%, Swing 0%

   Version B - Deep House:
   HH 80%, Perc 65%, Swing 20%
   Groove: MPC 16 Swing-58

   Version C - Hip Hop:
   HH 70%, Perc 55%, Kick 85%
   Groove: MPC 16 Swing-62, Amount 85%

   Version D - Lo-Fi:
   全パート 50-65%, Random 20%
   Groove: MPC 8 Swing-66, Amount 90%

   Version E - DnB (170 BPM):
   Kick 90%, Snare 95%, HH 75%
   Swing 10%, Ghost Note 追加

4. 順番に再生して比較

チェックポイント:
□ 各バージョンが異なるジャンルに聴こえるか
□ Quantize/Swing だけでこれほど変わることを体感できたか
□ 自分の好みのジャンル設定を発見できたか
```

### 演習10: 総合マスタリー課題

```
課題:
1曲分のドラムプログラミングを完成させる
(全てのQuantizeテクニックを駆使)

構成 (16小節 × 4セクション = 64小節):

Section A: イントロ (16小節)
├── Kick のみ (1/4, 100%)
├── 4小節目から HH 追加 (1/16, 90%, Swing 10%)
├── 8小節目から Perc 追加 (1/16, 70%, Swing 15%)
└── シンプル → 徐々に密度UP

Section B: ビルドアップ (16小節)
├── 全パート稼働
├── HH の Velocity を徐々に UP
├── Swing を 10% → 20% に徐々に増加
├── 12小節目から Fill 追加
└── エネルギー上昇

Section C: ドロップ (16小節)
├── フルキット (Kick+Snare+HH+Perc+Shaker)
├── 全パートにGroove Pool適用
│   Groove: MPC 16 Swing-58, Amount 60%
├── Velocity ヒューマナイズ適用
├── Track Delay で micro timing 設定
└── 最もグルーヴィなセクション

Section D: ブレイクダウン (16小節)
├── Kick OFF, Perc + HH のみ
├── Amount を下げる (60-70%)
├── Random を上げる (15%)
├── ルーズで有機的
└── Section C との対比

完成基準:
□ 全64小節が一貫したグルーヴを持つ
□ 各セクションに適切なQuantize設定がある
□ パート別に異なるAmount/Swing が設定されている
□ Groove Pool が効果的に活用されている
□ ヒューマナイズが自然に適用されている
□ A/B比較で明確な改善が確認できる
□ 楽曲全体として「踊れる」グルーヴである
```

---

## 付録: Quantize クイックリファレンス

### ショートカット一覧

```
Ableton Live:

Cmd+U          : Quantize (現在の設定で即実行)
Cmd+Shift+U    : Quantize Settings ダイアログ
Cmd+A          : 全ノート選択
Cmd+Z          : Undo (Quantize取り消し)
Cmd+Shift+Z    : Redo (Quantize再適用)
Cmd+D          : Clip 複製
Cmd+J          : Consolidate (Audio Clip確定)
Cmd+4          : Grid ON/OFF トグル
Cmd+1          : Grid を細かく
Cmd+2          : Grid を粗く
Cmd+3          : Triplet Grid トグル
```

### ジャンル別 Quantize 早見表

```
┌──────────────┬───────┬────────┬─────────┬─────────────┐
│ ジャンル     │ BPM   │ Amount │ Swing   │ Groove      │
├──────────────┼───────┼────────┼─────────┼─────────────┤
│ Techno       │125-140│ 95-100%│ 0-5%    │ なし        │
│ Deep House   │118-125│ 70-85% │ 15-25%  │ MPC 16-58   │
│ Tech House   │124-130│ 85-95% │ 5-15%   │ Swing 16-54 │
│ Hip Hop      │ 80-100│ 65-85% │ 30-55%  │ MPC 16-62   │
│ Trap         │130-170│ 90-100%│ 0-10%   │ なし        │
│ Lo-Fi        │ 70-90 │ 45-65% │ 45-60%  │ MPC 8-66    │
│ DnB          │160-180│ 75-90% │ 5-15%   │ カスタム    │
│ Garage       │128-135│ 70-85% │ 20-35%  │ MPC 16-60   │
│ Future Bass  │130-160│ 80-95% │ 10-20%  │ Swing 16-56 │
│ Ambient      │ 60-100│ 50-70% │ 0-10%   │ なし        │
│ Jazz         │ 80-180│ 30-60% │ 50-66%  │ カスタム    │
│ Funk         │100-130│ 60-80% │ 25-40%  │ MPC 16-66   │
└──────────────┴───────┴────────┴─────────┴─────────────┘
```

### Quantize 判断フローチャート

```
START: ノートを打ち込んだ/録音した
│
├── リズムの土台 (Kick, Bass, Snare)?
│   └── YES → 100% Quantize, 適切なGrid
│
├── 装飾的リズム (HH, Perc, Shaker)?
│   └── YES → 70-85% Quantize + Swing 検討
│
├── メロディ/和音?
│   └── YES → 50-85% Quantize (表現による)
│
├── エフェクト/ノイズ?
│   └── YES → Quantize 不要 (自由配置)
│
├── 生演奏の録音?
│   └── YES → まず 50% から段階的に
│             人間らしさを残しつつ補正
│
└── 完全にプログラミング?
    └── YES → 100% 後に Groove Pool で味付け

Swing の判断:
├── ストレートなジャンル? → 0-10%
├── バウンシーなジャンル? → 15-30%
├── ヒップホップ系? → 30-55%
└── ジャズ系? → 50-66%

最終チェック:
□ クリック/メトロノームと合っているか
□ 他のパートと一体感があるか
□ 「踊れる」グルーヴになっているか
□ 不自然な箇所はないか
□ A/B比較で改善を確認したか
```

---

**次は:** [録音テクニック](./recording-techniques.md) - Audio/MIDI完璧な録音

---

## 次に読むべきガイド

- [録音テクニック](./recording-techniques.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
