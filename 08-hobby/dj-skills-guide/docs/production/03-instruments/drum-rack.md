# Drum Rack

ドラムプログラミングの全て。Techno/House制作に必須のDrum Rackを完全マスターします。

## この章で学ぶこと

- Drum Rackの構造
- パッド配置（4×4グリッド）
- サンプル読み込み
- Chain（チェイン）活用
- Send/Return活用
- Technoキット構築
- Hi-Hatプログラミング
- ベロシティ・ランダム化

---

## なぜDrum Rackが最重要なのか

**Techno/House の心臓:**

```
使用頻度:

Wavetable: 70%
Drum Rack: 90%
最も使う

理由:

ドラムなし:
曲にならない

リズム:
全ての土台

Techno/House:
ドラム中心

プロの制作:

ドラム作成時間:
全体の40-50%

Drum Rack:
100%使用

他の方法:

個別トラック:
Kick → Track 1
Snare → Track 2
...

問題:
トラック数多い
管理困難

Drum Rack:

全ドラム:
1トラック

メリット:
管理簡単
エフェクト共有
CPU効率的
```

---

## Drum Rackの構造

**4×4グリッド:**

### 全体像

```
┌───────────────────────────┐
│ Drum Rack                 │
├───────────────────────────┤
│ ┌──┬──┬──┬──┐            │
│ │C │C#│D │D#│ ← Row 1   │
│ ├──┼──┼──┼──┤            │
│ │E │F │F#│G │ ← Row 2   │
│ ├──┼──┼──┼──┤            │
│ │G#│A │A#│B │ ← Row 3   │
│ ├──┼──┼──┼──┤            │
│ │C │C#│D │D#│ ← Row 4   │
│ └──┴──┴──┴──┘            │
│                           │
│ 16パッド (4×4)           │
└───────────────────────────┘

各パッド:

Empty:
空

Simpler:
サンプル1つ

Instrument:
Wavetable等も可

Chain:
複数デバイス

MIDI Note対応:

C1 = パッド1
C#1 = パッド2
...
D#2 = パッド16

DDJ-FLX4 Performance Pads:
直接トリガー可能
```

### インターフェイス

```
上部 (パッド表示):
┌───────────────────────────┐
│ ┌───┬───┬───┬───┐        │
│ │Kck│Snr│Ht │Ht │        │
│ ├───┼───┼───┼───┤        │
│ │Cla│Cla│Rim│Rim│        │
│ ├───┼───┼───┼───┤        │
│ │Prc│Prc│Fx │Fx │        │
│ └───┴───┴───┴───┘        │
└───────────────────────────┘

下部 (Chain View):
┌───────────────────────────┐
│ パッド選択時:              │
│ Simpler / Instrumentチェイン│
│ エフェクト追加可能         │
└───────────────────────────┘

右側 (Return):
┌───────────────────────────┐
│ Send A, B, C, D           │
│ Reverb, Delay等           │
└───────────────────────────┘
```

---

## パッド配置

**標準レイアウト:**

### Techno/House 推奨配置

```
┌──────┬──────┬──────┬──────┐
│ Kick │ Snare│ CH   │ OH   │ Row 1
│ C1   │ C#1  │ D1   │ D#1  │ 基本
├──────┼──────┼──────┼──────┤
│ Clap │ Clap2│ Rim  │ Rim2 │ Row 2
│ E1   │ F1   │ F#1  │ G1   │ スネア系
├──────┼──────┼──────┼──────┤
│ Perc │ Perc2│ FX   │ FX2  │ Row 3
│ G#1  │ A1   │ A#1  │ B1   │ パーカッション
├──────┼──────┼──────┼──────┤
│ Kick2│ Tom  │ Cymb │ Crash│ Row 4
│ C2   │ C#2  │ D2   │ D#2  │ 補助
└──────┴──────┴──────┴──────┘

略語:

CH = Closed Hi-Hat
OH = Open Hi-Hat
Perc = Percussion
Cymb = Cymbal

理由:

Row 1:
最も重要
Kick, HH

Row 2:
スネア系
バリエーション

Row 3:
装飾
パーカッション

Row 4:
補助・FX
```

### MIDIノート割り当て

```
C1 (36):
Kick
最重要

D1 (38):
Closed Hi-Hat
2番目に重要

C#1 (37):
Snare / Clap

D#1 (39):
Open Hi-Hat

推奨:

Kick = C1:
標準
変更しない

理由:
互換性
他の人と共有
```

---

## サンプル配置

**ドラッグ&ドロップ:**

### 基本操作

```
方法1: 直接ドロップ

1. Browser > Drums

2. サンプル選択:
   例: Kick.wav

3. ドラッグ:
   パッドにドロップ

4. Simpler自動挿入

方法2: 右クリック

1. パッド右クリック

2. "Load Sample"

3. ファイル選択

方法3: Hot-Swap

1. パッドの🔍アイコン

2. Browser表示

3. サンプル選択

4. 入れ替え

推奨:

初心者:
ドラッグ&ドロップ

大量:
Browser活用
```

### サンプル選択

```
Kick:

特徴:
太い
40-60 Hz
アタック明確

推奨パック:
Vengeance
Splice Techno Kicks

Snare:

特徴:
明るい
200 Hz + 1-3 kHz
スナップ

Clap:
リバーブ付き
ハウス向き

Hi-Hat:

Closed:
短い
明るい

Open:
長い
リリース

Percussion:

Shaker:
テクスチャ

Conga:
低域

効果音:

Riser:
ビルドアップ

Impact:
ドロップ
```

---

## Chain（チェイン）

**パッドの内部:**

### Chain構造

```
1つのパッド内:

Simpler (デフォルト):
サンプル再生

または:

Wavetable:
シンセキック

または:

Chain:
Simpler + エフェクト

例:

Kick Chain:

Simpler (Kick.wav)
   ↓
EQ Eight (Low Boost)
   ↓
Compressor (Punch)
   ↓
Output

効果:
パッド専用エフェクト
```

### Chain View

```
表示:

パッド選択:
下部にChain表示

デバイス追加:

Browser > Audio Effects:
ドラッグ&ドロップ

例 (Hi-Hat):

Simpler
   ↓
Auto Filter (Cutoff 8000 Hz)
   ↓
Reverb (Room, Size 30%)
   ↓
Output

効果:
明るく、空間広い
```

---

## Send/Return活用

**共有エフェクト:**

### Return Track設定

```
通常トラック:

各パッドに:
個別エフェクト

問題:
CPU負荷
管理困難

Return Track:

1つのエフェクト:
全パッドで共有

Send量:
パッドごと調整

設定:

Return A:

Reverb:
Type: Hall
Size: 60%
Decay: 2.5s

Return B:

Delay:
Type: Ping Pong
Time: 1/8
Feedback: 40%

Return C:

Reverb (Room):
Size: 30%
短い残響

推奨:

Kick: Send なし
Bass: Send なし
理由: 低域濁る

Snare: Send A 20%
HH: Send A 10%, Send B 5%
Perc: Send A 30%, Send C 20%
```

### Send量調整

```
各パッド:

Kick:
Send A: 0%
Send B: 0%

Snare:
Send A: 20% (Reverb)
Send B: 0%

Closed HH:
Send A: 8%
Send B: 5% (Delay)

Open HH:
Send A: 15%
Send B: 10%

Percussion:
Send A: 25%
Send B: 15%

FX:
Send A: 40%
Send B: 30%

ルール:

低域 (Kick, Bass):
Sendなし

中域 (Snare, Clap):
適度に

高域 (HH, Perc):
多めOK
```

---

## 実践: Technoキット構築

**30分で完成:**

### Step 1: Drum Rack作成 (2分)

```
1. 新規MIDI Track:
   Cmd+T

2. Drum Rack挿入:
   Browser > Instruments > Drum Rack

3. 確認:
   空のパッド表示
```

### Step 2: Kick配置 (5分)

```
パッド: C1

1. Browser > Drums > Kicks

2. Kick選択:
   太く、アタック明確

3. C1パッドにドロップ

4. Simpler設定:
   Loop: Off
   Transpose: 0 st

5. 確認:
   C1ノート入力
   4 on the floor

Chain追加:

EQ Eight:
Low: +3 dB (60 Hz)

Saturator:
Drive: 5 dB
わずかに歪み

Compressor:
Ratio: 4:1
Attack: 10 ms
Release: 80 ms
```

### Step 3: Snare/Clap (5分)

```
パッド: C#1

1. Clap サンプル

2. ドロップ

3. Simpler:
   Transpose: +2 st
   わずかに高く

Chain:

EQ Eight:
High Shelf: +2 dB (6 kHz)

Reverb:
Type: Plate
Size: 40%
Dry/Wet: 30%

Send:
Send A: 25%
```

### Step 4: Hi-Hat (10分)

```
Closed HH (D1):

1. Closed HH サンプル

2. ドロップ

3. Chain:

Auto Filter:
Cutoff: 10000 Hz
Resonance: 10%

Compressor:
Ratio: 3:1
速いアタック

Send:
Send A: 10%
Send B: 8%

Open HH (D#1):

1. Open HH サンプル

2. Choke Group設定:
   Closed と同じグループ
   → 排他的

3. Chain:

Filter Cutoff: 12000 Hz

Send:
Send A: 18%
Send B: 12%

パターン:

16分音符:
CH × 16

4拍目:
OH

効果:
グルーヴ
```

### Step 5: Percussion (5分)

```
Shaker (G#1):

1. Shaker サンプル

2. Chain:

Transpose: +5 st
高く

Reverb:
わずかに

Send:
Send A: 30%

Conga (A1):

1. Conga サンプル

2. ベロシティ調整:
   80-100 範囲
   バラつき

Rim (F#1):

1. Rimshot

2. Transpose: +3 st

3. Send A: 15%
```

### Step 6: 仕上げ (3分)

```
Return Track:

Return A:
Valhalla VintageVerb
または
Ableton Reverb (Hall)

Return B:
Filter Delay
Ping Pong

Master:

Volume: -6 dB
ヘッドルーム

確認:

4小節パターン:
Kick, Clap, HH, Perc

再生:
グルーヴ確認

保存:
Cmd+S
```

---

## ベロシティとランダム化

**人間らしさ:**

### Velocity設定

```
Simpler内:

Velocity → Volume:
強弱で音量変化

Velocity → Filter:
強弱で明るさ変化

推奨:

Kick:
Velocity: 127固定
安定

Snare:
Velocity: 100-120
わずかなバラつき

Hi-Hat:
Velocity: 70-110
大きくバラつき
グルーヴ

Percussion:
Velocity: 60-100
自然

MIDI Clip:

Randomize:
右クリック → Randomize

Velocity: 20-30%
Position: 0%

効果:
機械的 → 人間的
```

### Choke Group

```
機能:
排他的発音

用途:

Hi-Hat:
Closed と Open
同時に鳴らない

設定:

1. Closed HH パッド選択

2. Choke: Group A

3. Open HH パッド選択

4. Choke: Group A

効果:
リアルなHH動作

他の例:

Conga High / Low:
同じグループ

Tom:
複数トムを排他
```

---

## よくある質問

### Q1: サンプルが多すぎて選べない

**A:** プリセットキット使用

```
Drum Rack:

Browser > Drum Rack:
プリセットキット

推奨:

Techno Kit
House Kit
等

方法:

1. プリセット選択

2. 各サンプル入れ替え:
   Hot-Swap

3. 調整

利点:
構成済み
すぐ使える

時間:
ゼロから: 30分
プリセット: 5分
```

### Q2: CPU負荷が高い

**A:** Freeze使用

```
問題:
エフェクト多い
CPU重い

解決:

右クリック → Flatten:
Drum Rack → Audio化

または:

各パッド:
必要最小限エフェクト

Return Track:
共有エフェクト活用

推奨:

制作中:
そのまま

完成後:
Flatten
```

### Q3: 音が重なりすぎ

**A:** EQ で整理

```
問題:
全ての音が濁る

解決:

各パッドにEQ:

Kick:
High Cut: 8000 Hz
低域のみ

Snare:
High Pass: 200 Hz
Low Cut: 8000 Hz
中域

Hi-Hat:
High Pass: 6000 Hz
高域のみ

効果:
分離
クリア

または:

Drum Bus:
EQ Eight
全体調整
```

---

## まとめ

### Drum Rack基礎

```
□ 4×4 グリッド (16パッド)
□ C1 = Kick (標準)
□ Chain でエフェクト
□ Send/Return 活用
□ Choke Group で排他
```

### キット構築

```
Row 1: Kick, Snare, CH, OH
Row 2: Clap, Rim, Perc
Row 3: FX, 補助音
Row 4: バリエーション
```

### 重要ポイント

```
□ プリセットキット活用
□ Return Track設定
□ Velocity でグルーヴ
□ EQ で分離
□ CPU管理に Freeze
```

---

**次は:** [External Instruments](./external-instruments.md) - 外部音源連携
