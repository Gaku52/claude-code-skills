# Audio Effect Rack

複雑なエフェクトChainを構築。Parallel処理・Macro Knobを完全マスターし、プロの高度なエフェクト技術を実現します。

## この章で学ぶこと

- Audio Effect Rack基礎
- Parallel Processing（NY Compression）
- Chain構造とルーティング
- Macro Knob作成
- Multi-band Split
- プリセット保存
- 実践的テンプレート

---

## なぜEffect Rackが重要なのか

**プロの秘密兵器:**

```
通常エフェクト:

Individual:
1つずつ挿入

問題:
柔軟性低い
複雑な処理困難

Audio Effect Rack:

Chain:
複数ルート

Parallel:
Dry/Wet自由

Macro:
1つまみで複数制御

使用頻度:

初心者: 5%
知らない

中級者: 20%
一部使用

上級者: 60%
多用

プロ: 80%+
必須技術

プロとアマの差:

アマ:
Individual効果のみ

プロ:
Rack・Parallel
複雑な処理

結果:
プロ: 柔軟、高度
アマ: 限定的
```

---

## Audio Effect Rack 基礎

**構造理解:**

### Rackとは

```
定義:

Container:
複数エフェクト収納

Chain:
並列・直列ルーティング

Macro:
パラメーター統合制御

メリット:

柔軟:
複雑なルーティング

効率:
1つまみで複数制御

整理:
見やすい

プリセット:
保存・共有
```

### 作成方法

```
方法1: エフェクトからGroup

1. エフェクト選択 (複数可)
2. 右クリック
3. "Group" (Cmd+G)
4. Audio Effect Rack作成

方法2: 直接挿入

1. Browser > Audio Effects
2. "Audio Effect Rack"
3. ドラッグ&ドロップ

方法3: 空Rackから

1. 空Rack挿入
2. Chainにエフェクト追加
```

### インターフェイス

```
┌─────────────────────────────────┐
│  Audio Effect Rack              │
├─────────────────────────────────┤
│  [Show/Hide Chains] [Macros]    │
│                                 │
│  Chain 1 (100%):                │
│    [EQ Eight] [Compressor]      │
│                                 │
│  Chain 2 (50%):                 │
│    [Saturator] [Reverb]         │
│                                 │
│  Macro Knobs (1-8):             │
│    Macro 1: Drive               │
│    Macro 2: Mix                 │
│    ...                          │
└─────────────────────────────────┘

3つのView:
- Chain List (左)
- Chain View (中央)
- Macro (右)
```

---

## Chain構造

**ルーティング基礎:**

### Single Chain (直列)

```
構造:

Input
  ↓
EQ Eight
  ↓
Compressor
  ↓
Saturator
  ↓
Output

用途:
通常のエフェクトChain
整理目的

特徴:
1本道
Rackなしと同じ
```

### Parallel Chains (並列)

```
構造:

Input
  ├→ Chain 1 (Dry): 70%
  │    何もなし
  │
  └→ Chain 2 (Wet): 30%
       Saturator (Drive 12 dB)

Output (Mix)

用途:
Parallel処理
NY Compression

特徴:
Chain Mix調整
柔軟
```

### Multi-band Split

```
構造:

Input
  ├→ Chain 1 (Low): 20-500 Hz
  │    Compressor (強め)
  │
  ├→ Chain 2 (Mid): 500-5k Hz
  │    EQ Eight
  │
  └→ Chain 3 (High): 5k-20k Hz
       Saturator

Output (Mix)

用途:
帯域別処理
高度

特徴:
Frequency Split使用
```

---

## Parallel Processing

**NY Compression:**

### 基本概念

```
NY (New York) Compression:

通常Compression:

Input → Compressor → Output

問題:
ダイナミクス失われる

NY Compression:

Input
  ├→ Dry (70%): そのまま
  └→ Wet (30%): Compressor (強烈)

Output: Mix

メリット:

ダイナミクス:
Dryで維持

パンチ:
Wetで圧縮

結果:
最良のバランス

使用:
Drums・Vocal・Bass
プロ標準
```

### 設定方法

```
Step 1: Rack作成

1. トラック選択 (Drums)
2. Audio Effect Rack挿入
3. Chain List表示

Step 2: Chain設定

Chain 1 (Dry):

名前: "Dry"
Chain Volume: 0 dB (100%)
エフェクト: なし

Chain 2 (Wet - Compressed):

名前: "Compressed"
Chain Volume: -6 dB (50%)

エフェクト:
Compressor:
  Threshold: -30 dB (低い)
  Ratio: 8:1 (強烈)
  Attack: 1 ms (最速)
  Release: 50 ms
  Make-Up: +20 dB

EQ Eight (Post):
  High Pass: 100 Hz
  (低域は圧縮しない)

Step 3: バランス調整

Chain 1: 70%
Chain 2: 30%

A/B比較:
Bypass Rackで確認

結果:
ダイナミクス + パンチ
```

---

## Parallel Saturation

**温かみを並列処理:**

### 設定例

```
目的:
自然な温かみ

Chain構成:

Chain 1 (Dry): 60%
何もなし

Chain 2 (Saturated): 40%
Saturator:
  Curve: Analog Clip
  Drive: 15 dB (強烈)
  Output: -15 dB

EQ Eight (Post):
  High Pass: 200 Hz
  Low Pass: 8 kHz

用途:
Bass・Vocal・Drums

メリット:

Dry:
クリーン維持

Wet:
倍音追加

Mix:
自然なブレンド
```

---

## Macro Knob 完全ガイド

**1つまみで複数制御:**

### 基本概念

```
通常:

Parameter 1: 個別調整
Parameter 2: 個別調整
Parameter 3: 個別調整

問題:
時間かかる

Macro Knob:

Macro 1:
- Parameter 1 (Map)
- Parameter 2 (Map)
- Parameter 3 (Map)

1つまみ:
3つ同時制御

メリット:

効率:
速い

ライブ:
即座に変化

Automation:
1つのオートメーション
```

### Map方法

```
Step 1: Macro表示

1. Rack選択
2. "Show Macro" (右上)
3. 8つのMacro Knob表示

Step 2: Map Mode

1. Macro 1選択
2. "Map" ボタン: On
3. パラメーター有効化

Step 3: パラメーターMap

1. マップしたいパラメータークリック
   例: Saturator > Drive

2. 自動的にMap

3. Range調整:
   Min: 0 dB
   Max: 10 dB

4. 他のパラメーターも追加
   例: Compressor > Threshold

Step 4: 名前変更

1. Macro 1右クリック
2. "Rename"
3. 名前: "Drive"

完成:

Macro 1 (Drive):
- Saturator Drive: 0-10 dB
- Compressor Threshold: -20 〜 -10 dB

1つまみで2つ同時制御
```

---

## 実践例: Drum Bus Rack

**完全なDrum処理:**

### Chain構成

```
目的:
全ドラム処理
Parallel Compression
Saturation
EQ

Rack名: "Drum Bus Master"

Chain 1 (Dry - 70%):

EQ Eight:
  High Pass: 30 Hz
  Peak: -2 dB @ 300 Hz

Chain 2 (Compressed - 30%):

Compressor:
  Threshold: -25 dB
  Ratio: 8:1
  Attack: 3 ms
  Release: 60 ms
  Make-Up: +18 dB

EQ Eight:
  High Pass: 150 Hz

Chain 3 (Saturated - 20%):

Saturator:
  Curve: Analog Clip
  Drive: 12 dB
  Output: -12 dB

Auto Filter:
  Lowpass 6 kHz

Output Chain:

Glue Compressor:
  Threshold: -15 dB
  Ratio: 2:1
  GR: -3 dB

EQ Eight (Final):
  High Shelf: +1 dB @ 10 kHz
```

### Macro設定

```
Macro 1 (Compression):

Map:
- Chain 2 Volume: 0-100%
- Compressor Threshold: -30 〜 -15 dB

用途:
圧縮量調整

Macro 2 (Saturation):

Map:
- Chain 3 Volume: 0-100%
- Saturator Drive: 5-15 dB

用途:
歪み量調整

Macro 3 (High End):

Map:
- Final EQ High Shelf: 0 〜 +3 dB

用途:
明るさ調整

Macro 4 (Width):

Map:
- Utility Width: 100-120%

用途:
ステレオ幅

結果:

4つのつまみ:
完全なDrum制御

プリセット保存:
再利用可能
```

---

## Multi-band Rack

**帯域別処理:**

### Frequency Split使用

```
目的:
Low・Mid・High
それぞれ独立処理

設定方法:

Step 1: 3 Chains作成

Chain 1: Low
Chain 2: Mid
Chain 3: High

Step 2: Frequency Split

Chain 1 (Low):
- Chain選択
- Frequency範囲設定
  Low: 20-500 Hz

Chain 2 (Mid):
- Frequency: 500-5000 Hz

Chain 3 (High):
- Frequency: 5000-20000 Hz

Step 3: 各Chain処理

Chain 1 (Low):
Compressor:
  Ratio: 4:1
  GR: -4 dB

Saturator:
  Drive: 6 dB

Chain 2 (Mid):
EQ Eight:
  Peak: -2 dB @ 500 Hz

Compressor:
  Ratio: 3:1

Chain 3 (High):
Saturator:
  Drive: 3 dB

De-esser (EQ):
  -3 dB @ 8 kHz

結果:
各帯域最適化
```

---

## 実践例: Vocal Rack

**完全なVocal処理:**

### Chain構成

```
Rack名: "Vocal Master"

Main Chain (Serial):

1. EQ Eight (Cut):
   High Pass: 80 Hz
   Peak: -3 dB @ 300 Hz (こもり)

2. Compressor 1 (Leveling):
   Threshold: -20 dB
   Ratio: 3:1
   Attack: 5 ms
   Release: 50 ms

3. De-esser (EQ):
   Peak: -4 dB @ 7 kHz
   Q: 3.0

4. Compressor 2 (Color):
   Threshold: -15 dB
   Ratio: 2:1

5. EQ Eight (Boost):
   Peak: +3 dB @ 3 kHz (明瞭度)
   High Shelf: +1.5 dB @ 10 kHz

6. Saturator:
   Curve: Warm
   Drive: 4 dB
   Dry/Wet: 60%

Parallel Chain (追加):

Chain 2 (Harmony - 15%):
Pitch Shifter:
  +7 semitones

Reverb:
  Plate, Decay 2.0s

Macro設定:

Macro 1 (Compression):
- Comp 1 Threshold
- Comp 2 Threshold

Macro 2 (Brightness):
- EQ Boost @ 3 kHz
- High Shelf @ 10 kHz

Macro 3 (De-ess):
- De-esser Gain

Macro 4 (Harmony):
- Chain 2 Volume
```

---

## プリセット保存

**再利用:**

### 保存方法

```
Step 1: Rack完成

全設定完了
Macro設定完了

Step 2: 保存

1. Rackタイトルバー右クリック
2. "Save Preset"
3. 名前: "My Drum Bus Master"
4. カテゴリ: Drums
5. タグ: Compression, NY, Parallel

Step 3: 確認

Browser > Audio Effects
→ User Library
→ "My Drum Bus Master"

使用:

新規プロジェクト:
ドラッグ&ドロップ

即座に使用可能
```

---

## 高度なテクニック

**プロの技:**

### Macro Range調整

```
目的:
安全範囲のみ

例:

Saturator Drive:
- 通常Range: 0-36 dB
- 安全Range: 3-10 dB

設定:

Map後:
Min: 3 dB
Max: 10 dB

Macroつまみ:
0%: 3 dB
100%: 10 dB

メリット:
過剰防止
```

### Chain Key Mapping

```
機能:

MIDI Note:
Chain切り替え

用途:

Key C1: Chain 1
Key C#1: Chain 2
Key D1: Chain 3

ライブ:
即座に切り替え

設定:

Chain Key:
On

Zone:
C1-C2
```

### Macro Automation

```
メリット:

1つのAutomation:
複数パラメーター変化

例:

Macro 1 (Buildup):

Map:
- Filter Cutoff: 200 → 5000 Hz
- Resonance: 10 → 60%
- Reverb Send: 10 → 50%

Automation:

Macro 1: 0% → 100%
8小節

結果:
3つ同時変化
ビルドアップ
```

---

## よくある失敗

### 1. Chain Mixバランス悪い

```
問題:
Wetが大きすぎ
Dryが埋もれる

原因:
Chain 2: 80%
Chain 1: 20%

解決:

標準:
Dry: 70%
Wet: 30%

A/B比較:
必須
```

### 2. Macroマップしすぎ

```
問題:
1つのMacroに
10個パラメーター

制御困難

解決:

1つのMacro:
2-4パラメーター

関連するもののみ
```

### 3. Multi-bandで位相問題

```
問題:
Frequency Split
位相ずれ

解決:

Linear Phase:
使用

Ableton:
自動的にLinear

確認:
A/B比較
```

---

## テンプレート集

**すぐ使える:**

### 1. NY Compression (Drums)

```
Chain 1 (Dry): 65%
Chain 2 (Comp): 35%
  Compressor: -30 dB, 8:1

Macro 1: Compression Amount
```

### 2. Parallel Saturation (Bass)

```
Chain 1 (Dry): 60%
Chain 2 (Sat): 40%
  Saturator: Analog, 12 dB

Macro 1: Saturation Amount
```

### 3. Multi-band (Master)

```
Chain 1 (Low): 20-200 Hz, Comp 4:1
Chain 2 (Mid): 200-5k Hz, EQ
Chain 3 (High): 5k-20k Hz, Saturator

Macro 1-3: 各帯域Volume
```

---

## 実践ワークフロー

**30分練習:**

### Week 1: Parallel Compression

```
Day 1 (15分):

1. Drum Track選択
2. Audio Effect Rack挿入
3. 2 Chains作成
   Dry + Compressed
4. バランス調整

Day 2-3:

Macro作成
Compression Amount

Day 4-7:

他のトラック適用
Bass・Vocal
```

### Week 2: 複雑なRack

```
Day 1-3:

Multi-band Rack
3 Chains
帯域別処理

Day 4-5:

Macro 4つ作成
各帯域制御

Day 6-7:

プリセット保存
再利用
```

---

## まとめ

### Audio Effect Rack

```
□ Parallel処理の基本
□ Chain構造理解
□ Macro Knob作成
□ プリセット保存
□ プロ必須技術
```

### NY Compression

```
□ Dry: 70%
□ Wet: 30% (強烈圧縮)
□ Drums・Vocal・Bassに
□ ダイナミクス維持
```

### Macro

```
□ 1つで2-4パラメーター
□ 名前わかりやすく
□ Range調整
□ Automation活用
```

### 重要原則

```
□ 整理整頓
□ プリセット活用
□ A/B比較
□ 過剰注意
□ プロの技術
```

---

**セクション完了！** これでproduction/04-effectsセクションの全8ファイルが完成しました。

次のステップ: コミット・プッシュ、そしてPROGRESS.md更新
