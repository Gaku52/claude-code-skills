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
