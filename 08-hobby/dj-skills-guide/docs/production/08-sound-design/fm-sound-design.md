# FM Sound Design（FM合成サウンドデザイン）

Ableton Live 12のOperatorを使ったFM合成を完全マスター。ベル、ブラス、エレピ、金属音など、Wavetableでは作れない特殊なサウンドを作成します。

## この章で学ぶこと

- FM合成の基本原理
- Operator 11種類のAlgorithm
- ベル音作成（Step-by-Step）
- ブラス音作成
- エレクトリックピアノ作成
- Ratio（倍音比率）の理解
- Fine Tuning活用
- 実践的な練習方法


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## なぜFM Sound Designが重要なのか

**サウンドの差別化:**

```
Watetableのみ:

できること:
ベース 90%
リード 80%
パッド 85%

できないこと:
ベル音
金属音
複雑なブラス
エレピ

結果:
70%の音は作れる
30%は作れない

Operator追加:

できること:
Wavetableの70%
+ FM合成の30%
= 100%

結果:
全ての音を作れる
完全な自由

プロの使用率:

Wavetable: 70%（メイン）
Operator: 20%（特殊音）
その他: 10%

結論:
両方使えば無敵
```

**FM合成の独自性:**

```
減算式（Wavetable）:
波形を削る
→ 丸い音、温かい音

FM合成（Operator）:
波形を変調する
→ 金属音、ベル音、複雑な倍音

できる音:
- ベル、チャイム
- エレクトリックピアノ
- ブラス（トランペット、サックス）
- 金属音、工業音
- パーカッシブ音

ジャンル:
Techno: 工業音、金属音
IDM: 複雑な倍音
Ambient: ベル音
Lo-Fi Hip Hop: エレピ
```

---

## FM合成の基本原理

### 減算式 vs FM合成

```
減算式合成（Wavetable）:

OSC（Saw波）
   ↓
Filter（削る）
   ↓
音色

特徴:
シンプル
直感的
丸い音

FM合成（Operator）:

Carrier OSC（音を出す）
   ↑
Modulator OSC（音色を変える）
   ↓
複雑な倍音

特徴:
複雑
非直感的
金属的な音
```

### Carrier と Modulator

**Carrier（キャリア）:**

```
定義: 実際に聞こえる音を出すOSC

役割:
基音を生成
Modulatorの影響を受ける

例:
Carrier周波数: 440 Hz（A4）
→ A4の音が聞こえる

重要:
Carrierがないと音が出ない
```

**Modulator（モジュレーター）:**

```
定義: Carrierの音色を変えるOSC

役割:
Carrierの周波数を変調
倍音を追加

例:
Modulator周波数: 440 Hz
Modulatorの強さ: 大
→ 複雑な倍音が追加される

重要:
Modulatorの周波数比率（Ratio）で音色が決まる
```

### Ratio（レシオ、倍音比率）

```
定義: Carrierに対するModulatorの周波数比

計算:
Carrier: 440 Hz
Modulator Ratio: 2
→ Modulator周波数: 880 Hz（2倍）

音色への影響:

Ratio 整数（1, 2, 3, 4...）:
→ ハーモニックな音（倍音）
→ 楽器的な音

Ratio 非整数（1.5, 2.7, 3.14...）:
→ 非ハーモニックな音
→ ベル音、金属音

例:
Ratio 1: 基音と同じ
Ratio 2: 1オクターブ上
Ratio 2.7: ベル音（非整数）
Ratio 3.5: 金属音
```

---

## Operatorインターフェース完全理解

### 基本構成

```
上部:
┌─────────────────────────────────────┐
│  A    B    C    D  ←4つのOSC      │
│ [OSC] [OSC] [OSC] [OSC]           │
└─────────────────────────────────────┘

中央:
┌─────────────────────────────────────┐
│     Algorithm                       │
│   [1] [2] [3]...[11]               │
│   接続パターン選択                   │
└─────────────────────────────────────┘

下部:
┌─────────────────────────────────────┐
│  Filter    LFO    Pitch Envelope   │
└─────────────────────────────────────┘
```

### 4つのOSC（A、B、C、D）

```
各OSCパラメータ:

1. Waveform:
   - Sine（推奨、FM合成の基本）
   - Triangle
   - Square
   - Saw

2. Coarse（Ratio整数部）:
   - 範囲: 0.5, 1, 2, 3...16
   - 倍音比率の整数部

3. Fine（微調整）:
   - 範囲: -50 cent 〜 +50 cent
   - Ratioの微調整

4. Fixed（固定周波数）:
   - On: Hz固定（ドラム音）
   - Off: ピッチに追従（楽器音）

5. Level:
   - 0-100%
   - 音量

6. Envelope (A D S R):
   - 各OSCごとに設定可能
```

### 11種類のAlgorithm

```
Algorithm = OSCの接続パターン

Algorithm 1:
A → B → C → D → 出力
直列接続、最も複雑な変調

Algorithm 2:
A → B → D → 出力
     C → D
分岐、中程度の複雑さ

Algorithm 3:
A → D → 出力
B → D
C → D
並列、シンプル

Algorithm 4:
A → B → 出力
C → D → 出力
2つのペア

...（全11種類）

選び方:
ベル音: Algorithm 11
ブラス: Algorithm 8
エレピ: Algorithm 4
実験: 全部試す
```

---

## 実践1: ベル音作成

**目標:** 寺院の鐘のような澄んだベル音を作る

### Step 1: 初期設定（3分）

```
新規MIDIトラック:
Operatorを追加

初期化:
全パラメータをリセット
または
Init Presetを選択

BPM: 120
Key: C Major
```

### Step 2: Algorithm選択（5分）

```
Algorithm: 11

構造:
A → 出力（Carrier）
B → A（Modulator 1）
C → A（Modulator 2）
D → A（Modulator 3）

理由:
複数のModulatorがCarrierを変調
→ 複雑な倍音
→ ベル音の特徴
```

### Step 3: OSC設定（15分）

```
OSC A（Carrier）:
Waveform: Sine
Coarse: 1.00
Fine: 0 cent
Fixed: Off
Level: 100%

Envelope:
A: 0 ms
D: 2500 ms（長い）
S: 0%
R: 3000 ms（長い）

理由:
Carrier = 実際に聞こえる音
長いDecay/Release = ベルの余韻

OSC B（Modulator 1）:
Waveform: Sine
Coarse: 2.00
Fine: +40 cent（重要）
Fixed: Off
Level: 85%

Envelope:
A: 0 ms
D: 1800 ms
S: 0%
R: 2000 ms

理由:
Ratio 2.00 + 40 cent ≈ 2.7
→ 非整数比率
→ ベル音の特徴

OSC C（Modulator 2）:
Waveform: Sine
Coarse: 3.00
Fine: +20 cent
Fixed: Off
Level: 60%

Envelope:
A: 0 ms
D: 1200 ms
S: 0%
R: 1500 ms

理由:
さらに複雑な倍音追加

OSC D（Modulator 3）:
Waveform: Sine
Coarse: 4.00
Fine: +10 cent
Fixed: Off
Level: 40%

Envelope:
A: 0 ms
D: 800 ms
S: 0%
R: 1000 ms

理由:
高域の倍音追加
```

### Step 4: Global設定（5分）

```
Filter:
Type: Off
理由: FM合成はFilter不要が多い

LFO:
Off
理由: ベル音は静的

Pitch Envelope:
Amount: 0
理由: ピッチ変化なし

Spread:
0%
理由: ベル音はMono推奨

Time:
100%
理由: 標準速度
```

### Step 5: エフェクト追加（10分）

```
エフェクトチェイン:

1. EQ Eight:
   High Pass: 120 Hz
   Peak: 800 Hz、+2 dB、Q 1.5
   理由: ベルの「チーン」音強調

2. Reverb:
   Type: Large Hall
   Decay: 6.5s（非常に長い）
   Dry/Wet: 40%
   理由: 寺院の空間感

3. Delay:
   Time: 1/4 Dotted
   Feedback: 25%
   Dry/Wet: 15%
   Filter: LP 5000 Hz
   理由: 微妙な反復

4. Utility:
   Width: 30%
   Gain: -1 dB
   理由: ややMono、ベルは中央定位
```

### Step 6: テスト演奏（10分）

```
テストMIDI:

パターン（8小節）:
Bar 1: C4（1拍）
Bar 2: E4（1拍）
Bar 3: G4（1拍）
Bar 4: C5（1拍）
Bar 5: C4 + E4 + G4（同時、2拍）
Bar 6: 休符
Bar 7: C3（1拍）
Bar 8: 休符

ベロシティ: 100-127（強く）
Length: 1拍（短いノート）

確認:
□ ベルの「チーン」音
□ 長い余韻（3秒以上）
□ 澄んだ音色
□ 非ハーモニック（わずかに不協和）
```

### Step 7: 調整（12分）

```
調整ポイント:

1. ベル感を強める:
   - OSC B Fine: +30 〜 +50 cent
   - Ratio非整数化

2. 余韻を長くする:
   - OSC A Decay/Release: 3000-5000 ms
   - Reverb Decay: 8s

3. 明るさ調整:
   - OSC C、D Level: 増減
   - 高域Modulator調整

4. アタック調整:
   - 全OSC Attack: 0-10 ms
   - 10 msで柔らかく

最終確認:
参考曲のベル音と比較
50%似たら成功
```

### 完成基準

```
□ Algorithm 11使用
□ Ratio 2.7付近（非整数）
□ Decay/Release 2000 ms以上
□ Reverb 6s以上
□ ベルの「チーン」音
□ 長い余韻
□ 澄んだ音色

所要時間: 60分
```

---

## 実践2: ブラス音作成

**目標:** トランペット風のブラス音を作る

### Step 1: 初期設定（3分）

```
新規MIDIトラック:
Operator追加、初期化

BPM: 115
Key: Bb Major（ブラス向き）
```

### Step 2: Algorithm選択（5分）

```
Algorithm: 8

構造:
A → 出力（Carrier 1）
B → 出力（Carrier 2）
C → A + B（Modulator）
D → 出力（Carrier 3）

理由:
複数のCarrier
→ 豊かな音色
→ ブラスの特徴
```

### Step 3: OSC設定（20分）

```
OSC A（Carrier 1）:
Waveform: Sine
Coarse: 1.00
Fine: 0 cent
Level: 100%

Envelope:
A: 50 ms
D: 100 ms
S: 75%
R: 150 ms

理由:
Attack 50 ms = ブラスの立ち上がり
Sustain 75% = 吹き続ける感じ

OSC B（Carrier 2）:
Waveform: Sine
Coarse: 2.00（1オクターブ上）
Fine: 0 cent
Level: 40%

Envelope:
A: 60 ms
D: 120 ms
S: 65%
R: 140 ms

理由:
2倍音を追加
→ 明るさ

OSC C（Modulator）:
Waveform: Sine
Coarse: 1.00
Fine: +5 cent
Level: 70%

Envelope:
A: 20 ms
D: 200 ms
S: 50%
R: 100 ms

理由:
Ratio 1 + 5 cent
→ わずかな変調
→ ブラスの「ぼわー」感

OSC D（Carrier 3）:
Waveform: Sine
Coarse: 3.00
Fine: 0 cent
Level: 20%

Envelope:
A: 70 ms
D: 80 ms
S: 55%
R: 120 ms

理由:
3倍音追加
→ さらなる明るさ
```

### Step 4: Filter設定（8分）

```
Filter:
Type: Low Pass 12 dB
Cutoff: 3500 Hz
Resonance: 15%
Envelope Amount: +25

Filter Envelope:
A: 100 ms
D: 300 ms
S: 60%
R: 200 ms

理由:
ブラスは息を吹き込む
→ Filter Cutoffが徐々に開く
→ リアルな演奏感
```

### Step 5: LFO設定（8分）

```
LFO 1（ビブラート）:
Destination: Pitch
Waveform: Sine
Rate: 5 Hz
Depth: 3%
Retrigger: Off

理由:
ブラス奏者のビブラート再現

LFO 2（音色変化）:
Destination: OSC C Level
Waveform: Triangle
Rate: 0.8 Hz
Depth: 15%
Retrigger: Off

理由:
息の強弱
→ Modulator Levelが変化
→ 音色が揺れる
```

### Step 6: エフェクト追加（12分）

```
エフェクトチェイン:

1. EQ Eight:
   High Pass: 150 Hz
   Peak: 1200 Hz、+3 dB、Q 1.2
   Peak: 3000 Hz、+2 dB、Q 0.8
   理由: ブラスの存在感

2. Saturator:
   Drive: 4 dB
   Curve: Warm
   Dry/Wet: 70%
   理由: 音圧、ブラスの温かみ

3. Compressor:
   Ratio: 3:1
   Threshold: -12 dB
   Attack: 10 ms
   Release: 100 ms
   Gain: +3 dB
   理由: ダイナミクス制御

4. Reverb:
   Type: Medium Room
   Decay: 1.8s
   Dry/Wet: 18%
   理由: スタジオ録音感

5. Utility:
   Width: 20%
   理由: ブラスは中央定位
```

### Step 7: テスト演奏（15分）

```
テストMIDI:

メロディ（8小節）:
Bar 1: Bb3 ──── D4 ── F4 ──
Bar 2: Bb4 ──── ──── ──── ────
Bar 3: G4 ──── F4 ── Eb4 ─
Bar 4: D4 ──── ──── ──── ────
Bar 5-8: 繰り返し

ベロシティ: 90-120
Length: 1/2 〜 1拍

演奏技法:
強い音: Velocity 120
→ Filter開く、音色明るい

弱い音: Velocity 90
→ Filter閉じる、音色暗い

確認:
□ ブラスの「ぼわー」音
□ ビブラートがある
□ Velocity感度が高い
□ リアルな演奏感
```

### 完成基準

```
□ Algorithm 8使用
□ 複数Carrier
□ Filter Envelope設定
□ LFO（Pitch、Level）
□ Attack 50-70 ms
□ ビブラート3-5%
□ ブラス音

所要時間: 70分
```

---

## 実践3: エレクトリックピアノ作成

**目標:** Rhodes風のエレピ音を作る

### Step 1: 初期設定（3分）

```
新規MIDIトラック:
Operator追加

BPM: 90（Lo-Fi向き）
Key: F Major
```

### Step 2: Algorithm選択（5分）

```
Algorithm: 4

構造:
A → B → 出力（ペア1）
C → D → 出力（ペア2）

理由:
2つのペア
→ 豊かな音色
→ エレピの特徴
```

### Step 3: OSC設定（25分）

```
OSC A（Modulator 1）:
Waveform: Sine
Coarse: 1.00
Fine: 0 cent
Level: 65%

Envelope:
A: 0 ms
D: 800 ms
S: 0%
R: 600 ms

理由:
Pluck音の減衰

OSC B（Carrier 1）:
Waveform: Sine
Coarse: 1.00
Fine: 0 cent
Level: 100%

Envelope:
A: 0 ms
D: 1200 ms
S: 0%
R: 800 ms

理由:
Rhodes特有の長い減衰

OSC C（Modulator 2）:
Waveform: Sine
Coarse: 14.00（高い倍音）
Fine: +15 cent
Level: 25%

Envelope:
A: 0 ms
D: 200 ms
S: 0%
R: 150 ms

理由:
Ratio 14 + 15 cent
→ Rhodesの「チーン」音

OSC D（Carrier 2）:
Waveform: Triangle
Coarse: 1.00
Fine: -7 cent
Level: 60%

Envelope:
A: 0 ms
D: 1000 ms
S: 0%
R: 700 ms

理由:
Triangle波でさらなる倍音
-7 cent = Detuneでコーラス効果
```

### Step 4: Filter設定（8分）

```
Filter:
Type: Low Pass 12 dB
Cutoff: 4500 Hz
Resonance: 8%
Envelope Amount: -15（閉じる方向）

Filter Envelope:
A: 0 ms
D: 400 ms
S: 0%
R: 300 ms

理由:
鍵盤を押す → Cutoff開いている
時間経過 → 徐々に閉じる
→ Rhodesの音色変化
```

### Step 5: Pitch Envelope設定（5分）

```
Pitch Envelope:
A: 0 ms
D: 30 ms
S: 0%
R: 0 ms
Amount: +8 semitones

理由:
鍵盤を押す瞬間にピッチ上昇
→ すぐ下降
→ Rhodesのハンマー音再現
```

### Step 6: エフェクト追加（18分）

```
エフェクトチェイン:

1. Auto Pan:
   Rate: 1/8
   Amount: 15%
   Phase: 180°
   理由: Rhodesのトレモロ再現

2. Phaser:
   Rate: 0.15 Hz
   Amount: 30%
   Feedback: 40%
   Dry/Wet: 25%
   理由: 70年代Rhodes感

3. Chorus:
   Rate: 0.6 Hz
   Amount: 25%
   Dry/Wet: 30%
   理由: コーラス効果

4. EQ Eight:
   High Pass: 80 Hz
   Bell: 250 Hz、+2 dB、Q 0.8
   Bell: 2500 Hz、+1.5 dB、Q 1.0
   理由: Rhodes特有の周波数強調

5. Compressor:
   Ratio: 4:1
   Threshold: -18 dB
   Attack: 3 ms
   Release: 150 ms
   Gain: +4 dB
   理由: Pluck音の制御

6. Reverb:
   Type: Plate
   Decay: 2.2s
   Dry/Wet: 25%
   理由: スタジオ感

7. Saturator:
   Drive: 2 dB
   Curve: Warm
   Dry/Wet: 50%
   理由: ビンテージ感

8. Utility:
   Width: 110%
   理由: わずかなステレオ拡張
```

### Step 7: テスト演奏（16分）

```
テストMIDI:

コードパターン（16小節）:
Bar 1-2: Fmaj7（F3、A3、C4、E4）
Bar 3-4: Dm7（D3、F3、A3、C4）
Bar 5-6: Bb maj7（Bb2、D3、F3、A3）
Bar 7-8: C7（C3、E3、G3、Bb3）
Bar 9-16: 繰り返し

演奏技法:
左手: ルート音（1拍前に）
右手: コード（4分音符）

ベロシティ: 80-110
Length: 1/4拍（短く）

確認:
□ Rhodes特有の「チーン」音
□ 減衰が自然
□ トレモロ効果
□ Lo-Fi感
□ ビンテージ感
```

### 完成基準

```
□ Algorithm 4使用
□ Pitch Envelope設定
□ Decay 1000 ms以上
□ Auto Pan設定
□ Phaser/Chorus設定
□ Rhodes音

所要時間: 80分
```

---

## Ratio（倍音比率）深掘り

### 整数Ratio vs 非整数Ratio

```
整数Ratio（1, 2, 3, 4...）:

結果:
ハーモニックな音
楽器的
協和音

例:
Ratio 1: ユニゾン
Ratio 2: 1オクターブ上
Ratio 3: 1オクターブ+5度上
Ratio 4: 2オクターブ上

用途:
ブラス、木管、弦楽器

非整数Ratio（1.5, 2.7, 3.14...）:

結果:
非ハーモニックな音
金属的
不協和音

例:
Ratio 2.7: ベル音
Ratio 3.5: 金属音
Ratio 4.2: ガムラン風

用途:
ベル、チャイム、パーカッション
```

### 推奨Ratio設定

```
ベル音:
Carrier: 1.00
Modulator 1: 2.7
Modulator 2: 3.2
Modulator 3: 4.5

ブラス:
Carrier 1: 1.00
Carrier 2: 2.00
Modulator: 1.00
Carrier 3: 3.00

エレピ:
Carrier 1: 1.00
Modulator 1: 1.00
Carrier 2: 1.00
Modulator 2: 14.0

金属音:
Carrier: 1.00
Modulator 1: 5.8
Modulator 2: 7.3
Modulator 3: 11.2
```

### Fine Tuning活用

```
用途:

1. 非整数Ratio作成:
   Coarse: 2
   Fine: +40 cent
   → Ratio ≈ 2.7

2. Detune効果:
   OSC A Fine: 0 cent
   OSC B Fine: +8 cent
   → コーラス効果

3. 微妙な不協和:
   Fine: +5 〜 +15 cent
   → リアルな楽器感

推奨:
ベル音: +30 〜 +50 cent
ブラス: +5 〜 +10 cent
エレピ: ±7 cent
```

---

## Algorithm完全ガイド

### Algorithm選択フローチャート

```
目的:
ベル音、金属音
↓
Algorithm 11
複数ModulatorがCarrierを変調

目的:
ブラス、オルガン
↓
Algorithm 8
複数Carrier + Modulator

目的:
エレピ、Pluck
↓
Algorithm 4
2つのペア

目的:
ベース
↓
Algorithm 1
直列接続、最も変調

実験:
わからない
↓
全部試す
5分/Algorithm
```

### よく使うAlgorithm Top 5

```
1位: Algorithm 4（使用率 30%）
用途: エレピ、Pluck、ギター
理由: 2ペア、バランス良い

2位: Algorithm 11（使用率 25%）
用途: ベル、金属音、パーカッション
理由: 複雑な倍音

3位: Algorithm 8（使用率 20%）
用途: ブラス、オルガン
理由: 複数Carrier

4位: Algorithm 1（使用率 15%）
用途: ベース、Pad
理由: 直列、強い変調

5位: Algorithm 7（使用率 10%）
用途: 実験、特殊音
理由: 柔軟性
```

---

## よくある質問（FAQ）

**Q1: OperatorとWavetable、どちらを先に学ぶべきですか？**

```
A: Wavetableを先に学ぶ

理由:

Wavetable:
- シンプル
- 直感的
- 70%の音を作れる
- 初心者向き

Operator:
- 複雑
- 非直感的
- 30%の特殊音
- 中級者向き

推奨順序:
1. Wavetable（0-6ヶ月）
2. Operator（6-12ヶ月）

結果:
基礎を固めてから応用
```

**Q2: Sine波以外のWaveformを使うべきですか？**

```
A: 基本はSine波、実験で他を試す

Sine波:
使用率: 90%
理由: FM合成の基本、純粋な変調

Triangle波:
使用率: 7%
理由: わずかに倍音、柔らかい

Square/Saw波:
使用率: 3%
理由: 実験、特殊音

推奨:
最初の3ヶ月はSine波のみ
慣れたら他も試す
```

**Q3: Fixed Frequencyはいつ使いますか？**

```
A: ドラム音、効果音のみ

Fixed On:
用途: キック、スネア、ハイハット
理由: ピッチ固定、周波数指定

例:
キック: 60 Hz
スネア: 200 Hz
ハイハット: 8000 Hz

Fixed Off:
用途: 楽器音（ベル、ブラス、エレピ）
理由: ピッチに追従

推奨:
楽器音 = Fixed Off（95%）
ドラム音 = Fixed On（5%）
```

**Q4: Level（音量）をどう設定すればいいですか？**

```
A: Carrierは100%、Modulatorは実験

設定例:

ベル音:
Carrier: 100%
Modulator 1: 80%
Modulator 2: 60%
Modulator 3: 40%

ブラス:
Carrier 1: 100%
Carrier 2: 40%
Modulator: 70%
Carrier 3: 20%

一般則:
Carrier: 100%（基本）
Modulator: 50-90%（変調の深さ）

調整:
Modulator Level ↑ = 変調強い = 複雑な音
Modulator Level ↓ = 変調弱い = シンプルな音
```

**Q5: OperatorでベースやPadは作れますか？**

```
A: 作れるが、Wavetableの方が効率的

Operator:
できること:
ベース: 可能
Pad: 可能
リード: 可能

効率:
ベース: Wavetable 5分 vs Operator 20分
Pad: Wavetable 10分 vs Operator 40分

推奨:
ベース・Pad: Wavetable
ベル・ブラス・エレピ: Operator

理由:
適材適所
効率重視
```

**Q6: プリセットを改変するコツは？**

```
A: Ratioを変更する

手順:
1. 近いプリセット選択
2. Algorithm確認
3. 各OSC Ratio変更（±0.5）
4. Fine調整（±10 cent）
5. Envelope調整
6. 保存

変更箇所:
最低3箇所
→ オリジナリティ

時間:
20分/プリセット
```

---

## 練習方法

### Week 1: FM合成理論理解

```
Day 1-2: Carrier vs Modulator
1. Algorithm 1選択
2. OSC A（Carrier）のみOn
3. OSC B（Modulator）を徐々にLevel Up
4. 音色変化を聴く

Day 3-4: Ratio実験
1. Modulator Ratio 1、2、3、4
2. 整数Ratioの音を聴く
3. Ratio 2.7、3.5、4.8
4. 非整数Ratioの音を聴く

Day 5-7: Algorithm全探索
1. 11種類のAlgorithm
2. 各Algorithmで同じ設定
3. 違いを聴き比べ
4. メモを取る

目標:
FM合成の原理を体感
```

### Week 2: ベル音完全マスター

```
Day 1-2: 基本ベル音作成
1. 実践1を完全再現
2. Algorithm 11、Ratio 2.7
3. Decay 2500 ms

Day 3-4: バリエーション
1. Ratio変更（2.5、2.7、3.0、3.5）
2. 4種類のベル音作成
3. 違いを聴き比べ

Day 5-6: エフェクト実験
1. Reverb Decay（4s、6s、8s）
2. Delay追加/削除
3. EQ調整

Day 7: 実戦投入
1. Ambient楽曲作成
2. ベル音をメインに
3. 完成

目標:
ベル音を30分で作れる
```

### Week 3: ブラス音完全マスター

```
Day 1-2: 基本ブラス音作成
1. 実践2を完全再現
2. Algorithm 8
3. Filter Envelope設定

Day 3-4: ビブラート練習
1. LFO Depth（0%、3%、5%、8%）
2. Rate（4 Hz、5 Hz、6 Hz）
3. 自然なビブラート探し

Day 5-6: Velocity感度
1. Velocity → Filter Cutoff
2. Velocity → OSC Level
3. 演奏性向上

Day 7: メロディ作成
1. ブラスメロディ3個作成
2. Jazz、Funk、Pop
3. 完成

目標:
ブラス音を45分で作れる
```

### Week 4: エレピ完全マスター

```
Day 1-2: 基本エレピ作成
1. 実践3を完全再現
2. Algorithm 4
3. Pitch Envelope設定

Day 3-4: エフェクト深掘り
1. Phaser調整
2. Chorus調整
3. Auto Pan調整

Day 5-6: コード進行
1. Jazz風コード進行
2. Lo-Fi風コード進行
3. 5種類作成

Day 7: 楽曲制作
1. Lo-Fi Hip Hopトラック作成
2. エレピをメインに
3. ドラム、ベース追加
4. 完成

目標:
エレピ音を60分で作れる
```

---

## まとめ

FM合成は、Watetableでは作れない特殊な音を作る強力なツールです。

**重要ポイント:**

1. **Carrier**: 実際に聞こえる音を出す
2. **Modulator**: Carrierの音色を変える
3. **Ratio**: 整数=楽器音、非整数=金属音
4. **Algorithm**: OSCの接続パターン、音色を決定

**学習順序:**
1. FM合成理論理解（Week 1）
2. ベル音作成（Week 2）
3. ブラス音作成（Week 3）
4. エレピ作成（Week 4）

**推奨使用率:**
- Wavetable: 70%（ベース、リード、Pad）
- Operator: 20%（ベル、ブラス、エレピ）
- その他: 10%

**次のステップ:** [Sampling Techniques（サンプリング技術）](./sampling-techniques.md) へ進む

---

## 関連ファイル

- **[Wavetable Sound Design](./wavetable-sound-design.md)** - Wavetable活用
- **[Synthesis Basics](./synthesis-basics.md)** - シンセシス基礎理論
- **[Genre Sounds](./genre-sounds.md)** - ジャンル別サウンド
- **[03-instruments/operator.md](../03-instruments/)** - Operator詳細リファレンス

---

**OperatorでWavetableでは作れない音を作りましょう！**

---

## FM合成の数学的基礎

### 周波数変調の数式

```
FM合成の基本式:

y(t) = A × sin(2π × fc × t + I × sin(2π × fm × t))

各パラメータ:
A  = 振幅（Amplitude）
fc = キャリア周波数（Carrier Frequency）
fm = モジュレータ周波数（Modulator Frequency）
I  = 変調指数（Modulation Index）
t  = 時間（Time）

解説:
sin(2π × fc × t) がキャリア信号
sin(2π × fm × t) がモジュレータ信号
I がモジュレータの影響の強さを決定

変調指数 I の効果:
I = 0   → 純粋なサイン波（変調なし）
I = 0.5 → わずかな倍音追加
I = 1   → 明確な倍音構造
I = 2   → 豊かな倍音
I = 5   → 非常に複雑な倍音
I = 10  → 金属的・ノイジーな音
```

### ベッセル関数と倍音スペクトル

```
FM合成で生成される倍音はベッセル関数で記述される:

各倍音の振幅 = Jn(I)

Jn = n次のベッセル関数
I  = 変調指数
n  = 倍音次数

実用的な理解:

変調指数 I = 0:
  基音のみ（J0 = 1, 他は0）
  → 純粋なサイン波

変調指数 I = 1:
  基音: J0(1) ≈ 0.77
  第1倍音: J1(1) ≈ 0.44
  第2倍音: J2(1) ≈ 0.11
  → 少数の倍音、穏やかな音色

変調指数 I = 2:
  基音: J0(2) ≈ 0.22
  第1倍音: J1(2) ≈ 0.58
  第2倍音: J2(2) ≈ 0.35
  第3倍音: J3(2) ≈ 0.13
  → 基音より第1倍音が大きい、明るい音色

変調指数 I = 5:
  多数の倍音がほぼ均等に分布
  → 複雑でメタリックな音色

Operatorでの対応:
変調指数 ≈ Modulator Level × Modulator Ratio
Modulator Level 50% + Ratio 2 ≈ I = 1
Modulator Level 100% + Ratio 2 ≈ I = 2
```

### サイドバンド周波数の計算

```
FM合成で生成されるサイドバンド:

上側サイドバンド: fc + n × fm
下側サイドバンド: fc - n × fm

n = 1, 2, 3, 4...（倍音次数）

例: fc = 440 Hz, fm = 880 Hz（Ratio = 2）

上側:
440 + 1×880 = 1320 Hz（3倍音）
440 + 2×880 = 2200 Hz（5倍音）
440 + 3×880 = 3080 Hz（7倍音）

下側:
440 - 1×880 = -440 Hz → 折り返し → 440 Hz（基音に加算）
440 - 2×880 = -1320 Hz → 折り返し → 1320 Hz

結果:
整数Ratio → ハーモニックなサイドバンド
→ 音楽的に協和する倍音列

非整数Ratio例: fc = 440 Hz, fm = 1188 Hz（Ratio = 2.7）

上側:
440 + 1188 = 1628 Hz
440 + 2376 = 2816 Hz

下側:
440 - 1188 = -748 Hz → 折り返し → 748 Hz
440 - 2376 = -1936 Hz → 折り返し → 1936 Hz

結果:
非整数 → 非ハーモニックなサイドバンド
→ ベル音、金属音の原因
→ 倍音列が整数倍にならない
```

### C:M比率と音色の関係

```
C:M比率（Carrier対Modulator比率）の体系的理解:

C:M = 1:1
  サイドバンド: f, 2f, 3f, 4f...
  音色: 全倍音列（鋸歯状波に近い）
  用途: ブラス、ストリングス

C:M = 1:2
  サイドバンド: f, 3f, 5f, 7f...
  音色: 奇数倍音列（矩形波に近い）
  用途: クラリネット、木管楽器

C:M = 1:3
  サイドバンド: f, 2f, 4f, 5f, 7f, 8f...
  音色: 複雑な倍音パターン
  用途: オルガン、特殊音

C:M = 1:4
  サイドバンド: f, 3f, 5f, 7f, 9f...
  音色: 高域が豊富
  用途: ブライトなリード

C:M = 1:1.41（√2）
  サイドバンド: 非ハーモニック
  音色: ベル音
  用途: チャイム、ガムラン

C:M = 1:π
  サイドバンド: 完全に非ハーモニック
  音色: 金属的
  用途: 工業音、効果音
```

---

## Operatorの高度パッチ設計テクニック

### フィードバック（Self-Modulation）の活用

```
フィードバックとは:
OSCが自分自身を変調すること
Operatorでは各OSCにFeedbackパラメータがある

効果:
Feedback 0%: 純粋なサイン波
Feedback 20%: わずかに鋸歯状波化
Feedback 50%: 明確な鋸歯状波
Feedback 75%: 複雑なスペクトル
Feedback 100%: ノイズに近い

実践的な使用法:

1. SubBassの倍音追加:
   OSC A: Carrier, Sine, Feedback 15%
   → サイン波に微妙なエッジを追加
   → Filter不要の倍音コントロール

2. ノイズ生成:
   OSC D: Feedback 95-100%
   → ほぼホワイトノイズ
   → ハイハット、シンバル作成に利用

3. 有機的なテクスチャ:
   OSC C: Modulator, Feedback 30%
   → 変調元自体が複雑化
   → 予測不能な倍音変化

注意点:
Feedback量の変更は音量にも影響する
Feedback増加 → 音量増加 → Levelで補正必要
```

### Velocity Mappingの高度設定

```
Velocity（弾く強さ）でFM合成パラメータを制御:

設定場所:
Operator → 各OSC → Velocity Sensitivity

推奨マッピング:

1. Modulator Level × Velocity:
   弱く弾く → Modulator Level低い → シンプルな音
   強く弾く → Modulator Level高い → 複雑な倍音
   → アコースティック楽器の自然な挙動を再現
   設定: Vel → OSC B Level = 50-80%

2. Filter Cutoff × Velocity:
   弱く弾く → Filter閉じる → 暗い音
   強く弾く → Filter開く → 明るい音
   設定: Vel → Filter Cutoff = +30-50

3. Envelope Decay × Velocity:
   弱く弾く → 短いDecay → 短い音
   強く弾く → 長いDecay → 長い余韻
   設定: Vel → Time = 20-40%

4. 複合マッピング（上級）:
   Velocity → Modulator Level + Filter + Decay
   → 3つ同時制御
   → プロフェッショナルなパッチの必須テクニック

エレピの推奨Velocityマッピング:
  Vel → OSC A Level: 60%
  Vel → OSC C Level: 80%（高倍音を強く制御）
  Vel → Filter Cutoff: +35
  Vel → Pitch Env Amount: 40%
  → 弱打は柔らかく暗い音、強打は明るくアタック強い音
```

### エンベロープの高度デザイン

```
FM合成におけるエンベロープの重要性:
各OSCのエンベロープが独立 → 時間的な音色変化を精密制御

テクニック1: Modulatorの短いDecay
  Modulator Envelope:
  A: 0ms, D: 200ms, S: 0%, R: 100ms
  Carrier Envelope:
  A: 0ms, D: 2000ms, S: 30%, R: 500ms

  効果:
  アタック時のみ倍音豊か → 徐々にシンプルに
  → ピアノ、マリンバ、ビブラフォンに最適

テクニック2: Modulatorの遅いAttack
  Modulator Envelope:
  A: 500ms, D: 1000ms, S: 70%, R: 300ms

  効果:
  音の立ち上がりはシンプル → 徐々に複雑に
  → ストリングス、パッドに最適
  → 「ゆっくり開く」音色変化

テクニック3: 異なるRelease時間
  OSC A（Carrier）: Release 2000ms
  OSC B（Modulator 1）: Release 500ms
  OSC C（Modulator 2）: Release 1500ms

  効果:
  ノートオフ後の音色が時間とともに変化
  → リリース中に倍音構造が変わる
  → 自然な減衰感

テクニック4: Loop Envelope（Operator固有）
  Envelope Mode: Loop
  Loop Start: Decay開始点
  Loop End: Sustain到達点

  効果:
  エンベロープが繰り返される
  → リズミカルな音色変化
  → LFOよりも複雑なモジュレーション
```

---

## FM8/DX7との比較

### Yamaha DX7の歴史と影響

```
DX7（1983年発売）:

歴史的重要性:
- 世界初の本格的デジタルFMシンセサイザー
- 20万台以上販売（当時のシンセ史上最多）
- 80年代ポップスのサウンドを定義
- エレピ、ベル、ブラスが象徴的

特徴:
- 6オペレーター（OSC）
- 32種類のアルゴリズム
- 各オペレーターに独立エンベロープ（8段）
- 演奏用パラメータ（Breath Controller対応）

代表的な音色:
- E.Piano 1: 80年代を代表するエレピ音
- Bass 1: パンチのあるFMベース
- Brass 1: シンセブラス
- Marimba: クリアなマリンバ
- Tubular Bells: チューブラーベル

DX7の限界:
- 操作が非常に難しい（LCD 2行表示）
- パラメータアクセスが複雑
- リアルタイム操作に不向き
- プリセットに頼るユーザーが大多数
```

### Native Instruments FM8の特徴

```
FM8（2007年発売）:

DX7からの進化:
- GUIによる視覚的操作
- モジュレーションマトリクス
- 内蔵エフェクト
- DX7パッチの読み込み対応
- アルペジエーター内蔵

構成:
- 6オペレーター + 1ノイズジェネレーター
- フリーモジュレーションマトリクス
- 各オペレーターに波形選択（Sine以外も可能）
- エフェクトセクション（Reverb, Delay, EQ等）

FM8の強み:
- DX7の音色を完全再現可能
- マトリクスで任意の接続パターン作成
- スペクトルディスプレイで倍音を視覚確認
- モーフィング機能（音色間をスムーズ遷移）

FM8の弱み:
- 単体プラグイン（DAW統合度低い）
- CPU負荷がやや高い
- UIがやや古い
- Ableton Liveとの統合なし
```

### Ableton Operator vs DX7 vs FM8 比較表

```
┌─────────────────┬──────────┬──────────┬──────────┐
│ 項目            │ Operator │ DX7      │ FM8      │
├─────────────────┼──────────┼──────────┼──────────┤
│ オペレーター数  │ 4        │ 6        │ 6+1      │
│ アルゴリズム    │ 11種類   │ 32種類   │ 自由     │
│ 波形選択        │ 4種類    │ Sineのみ │ 多数     │
│ フィードバック  │ あり     │ あり     │ あり     │
│ フィルター      │ 内蔵     │ なし     │ 内蔵     │
│ LFO             │ 1基      │ 1基      │ 多数     │
│ エフェクト      │ 別途追加 │ なし     │ 内蔵     │
│ DAW統合         │ 完全     │ N/A      │ 低い     │
│ 操作性          │ 良好     │ 困難     │ 中程度   │
│ CPU負荷         │ 低い     │ N/A      │ 中程度   │
│ 価格            │ Live付属 │ 中古市場 │ 有料     │
│ 学習コスト      │ 中程度   │ 高い     │ 中〜高   │
└─────────────────┴──────────┴──────────┴──────────┘

結論:
Operator = DAW統合と操作性で最良
DX7 = ビンテージサウンドの本物感
FM8 = 最も柔軟だが環境依存

推奨:
初心者 → Operator一択
中級者 → Operator + FM8プリセット参考
上級者 → 目的に応じて使い分け
```

### DX7パッチのOperator再現テクニック

```
DX7の有名パッチをOperatorで再現する方法:

制約:
DX7 = 6 OSC, Operator = 4 OSC
→ 完全再現は不可能、近似再現を目指す

E.Piano 1（最も有名なDX7音色）:

DX7オリジナル:
Algorithm 5（6 OSC）
OSC 1-2: Carrier ペア1
OSC 3-4: Modulator → Carrier ペア2
OSC 5-6: 高域変調ペア

Operator再現:
Algorithm 4（2ペア構成）
OSC A: Modulator, Ratio 1.00, Level 65%
  Env: A:0, D:800, S:0, R:600
OSC B: Carrier, Ratio 1.00, Level 100%
  Env: A:0, D:1200, S:0, R:800
OSC C: Modulator, Ratio 14.00, Level 25%
  Env: A:0, D:200, S:0, R:150
OSC D: Carrier, Ratio 1.00, Level 60%
  Env: A:0, D:1000, S:0, R:700

ポイント:
- OSC CのRatio 14が「チーン」音の鍵
- OSC Cの短いDecayでアタック時のみ高倍音
- 2ペア構成で厚みを出す

Bass 1:

Operator再現:
Algorithm 1（直列）
OSC A: Carrier, Ratio 1.00, Level 100%
  Env: A:0, D:500, S:60%, R:200
OSC B: Modulator, Ratio 1.00, Level 80%, Feedback 20%
  Env: A:0, D:300, S:40%, R:150
OSC C: Off
OSC D: Off

ポイント:
- Feedbackで鋸歯状波化
- Algorithm 1の直列で強い変調
- シンプルな2 OSC構成で十分
```

---

## アルゴリズム別サウンドレシピ集

### Algorithm 1: 直列接続レシピ

```
構造: A → B → C → D → 出力

特徴:
最も変調が深い
4段階の直列変調
複雑で予測不能な音色

レシピ1: アグレッシブFMベース
  OSC A: Ratio 1.00, Level 90%, Feedback 25%
    Env: A:0, D:300, S:50%, R:150
  OSC B: Ratio 2.00, Level 70%
    Env: A:0, D:200, S:30%, R:100
  OSC C: Ratio 1.00, Level 60%
    Env: A:0, D:400, S:40%, R:200
  OSC D: Ratio 1.00, Level 100%（Carrier）
    Env: A:0, D:800, S:60%, R:300
  Filter: LP 24dB, Cutoff 1200Hz, Res 20%
  用途: Techno, Drum & Bass

レシピ2: グロウルベース
  OSC A: Ratio 1.00, Level 85%, Feedback 40%
    Env: A:0, D:500, S:70%, R:200
  OSC B: Ratio 3.00, Level 75%
    Env: A:0, D:350, S:50%, R:150
  OSC C: Ratio 1.00, Level 65%
    Env: A:0, D:600, S:55%, R:250
  OSC D: Ratio 1.00, Level 100%（Carrier）
    Env: A:5, D:1000, S:65%, R:400
  LFO → OSC A Level: Rate 4Hz, Depth 30%
  用途: Dubstep, Riddim
  → LFOで変調量を周期的に変化させるとグロウル感
```

### Algorithm 4: 2ペア構成レシピ

```
構造: A→B→出力, C→D→出力

特徴:
2つの独立したFMペア
レイヤーサウンドに最適
最も汎用性が高い

レシピ1: ヴィンテージエレピ（前述の発展版）
  ペア1（本体）:
  OSC A: Ratio 1.00, Level 55%
    Env: A:0, D:900, S:0%, R:700
  OSC B: Ratio 1.00, Level 100%
    Env: A:0, D:1500, S:0%, R:1000

  ペア2（アタック音）:
  OSC C: Ratio 13.00, Level 20%
    Env: A:0, D:150, S:0%, R:100
  OSC D: Ratio 1.00, Level 70%
    Env: A:0, D:1200, S:0%, R:800

  Pitch Env: Amount +6st, D:25ms
  用途: Lo-Fi, Neo Soul, Jazz

レシピ2: クリスタルパッド
  ペア1（低域パッド）:
  OSC A: Ratio 2.00, Level 40%
    Env: A:800, D:2000, S:70%, R:3000
  OSC B: Ratio 1.00, Level 100%
    Env: A:1000, D:3000, S:80%, R:4000

  ペア2（高域きらめき）:
  OSC C: Ratio 5.00, Fine +20cent, Level 30%
    Env: A:1200, D:2500, S:50%, R:3500
  OSC D: Ratio 3.00, Level 60%
    Env: A:1500, D:4000, S:60%, R:5000

  Reverb: Decay 8s, Wet 45%
  Chorus: Rate 0.3Hz, Amount 35%
  用途: Ambient, Cinematic, Chillout

レシピ3: プラックシンセ
  ペア1:
  OSC A: Ratio 3.00, Level 70%
    Env: A:0, D:150, S:0%, R:100
  OSC B: Ratio 1.00, Level 100%
    Env: A:0, D:300, S:0%, R:200

  ペア2:
  OSC C: Ratio 7.00, Level 45%
    Env: A:0, D:80, S:0%, R:60
  OSC D: Ratio 2.00, Level 80%
    Env: A:0, D:250, S:0%, R:180

  用途: Future Bass, Pop EDM
```

### Algorithm 7: 柔軟構成レシピ

```
構造: A→D→出力, B→D, C→D

特徴:
3つのModulatorが1つのCarrierを変調
各Modulatorが独立した倍音を追加
倍音の足し算的な設計が可能

レシピ1: ガムラン風ベル
  OSC A: Ratio 2.41, Level 55%
    Env: A:0, D:1200, S:0%, R:1500
  OSC B: Ratio 3.89, Level 45%
    Env: A:0, D:900, S:0%, R:1200
  OSC C: Ratio 6.73, Level 30%
    Env: A:0, D:600, S:0%, R:800
  OSC D: Ratio 1.00, Level 100%（Carrier）
    Env: A:0, D:3000, S:0%, R:4000

  Reverb: Large Hall, Decay 7s, Wet 35%
  用途: Ambient, World Music, IDM

レシピ2: 工業ノイズヒット
  OSC A: Ratio 7.13, Level 90%, Feedback 60%
    Env: A:0, D:100, S:0%, R:50
  OSC B: Ratio 11.37, Level 80%
    Env: A:0, D:80, S:0%, R:40
  OSC C: Ratio 4.71, Level 70%
    Env: A:0, D:120, S:0%, R:60
  OSC D: Ratio 1.00, Level 100%（Carrier）
    Env: A:0, D:200, S:0%, R:100

  Distortion: Drive 8dB
  用途: Industrial Techno, Noise
```

### Algorithm 11: 複数Modulator→Carrierレシピ

```
構造: B→A→出力, C→A, D→A

レシピ1: チューブラーベル
  OSC A: Ratio 1.00, Level 100%（Carrier）
    Env: A:0, D:4000, S:0%, R:5000
  OSC B: Ratio 3.50, Level 70%
    Env: A:0, D:2500, S:0%, R:3000
  OSC C: Ratio 5.19, Level 50%
    Env: A:0, D:1800, S:0%, R:2200
  OSC D: Ratio 8.27, Level 35%
    Env: A:0, D:1000, S:0%, R:1500

  Reverb: Cathedral, Decay 10s, Wet 50%
  EQ: HP 200Hz, Peak 1.5kHz +3dB
  用途: Cinematic, Orchestral, Ambient

レシピ2: ウィンドチャイム
  OSC A: Ratio 1.00, Level 100%
    Env: A:0, D:1500, S:0%, R:2000
  OSC B: Ratio 2.76, Level 60%
    Env: A:0, D:800, S:0%, R:1000
  OSC C: Ratio 4.13, Level 40%
    Env: A:0, D:500, S:0%, R:700
  OSC D: Ratio 6.85, Level 25%
    Env: A:0, D:300, S:0%, R:400

  Delay: Ping Pong, 1/8D, Feedback 35%, Wet 30%
  Reverb: Plate, Decay 3s, Wet 25%
  用途: Ambient, New Age, Film Score
```

---

## メタリック/ベル系サウンド設計の極意

### 非ハーモニック倍音の制御

```
メタリックサウンドの本質:
非整数Ratio → 非ハーモニック倍音 → 金属的な質感

非整数Ratioの選び方:

√2 ≈ 1.414:
  わずかに不協和
  穏やかなベル、ビブラフォン

π/2 ≈ 1.571:
  やや不協和
  教会の鐘、チャイム

√5 ≈ 2.236:
  明確な不協和
  ガムラン、東洋的ベル

e ≈ 2.718:
  強い不協和
  工業的なベル

π ≈ 3.14159:
  非常に強い不協和
  メタリックパーカッション

黄金比 φ ≈ 1.618:
  独特の響き
  不思議なベル音

実験手順:
1. 基本設定（Algorithm 11, Carrier Ratio 1.00）
2. Modulator 1のRatioを上記の値に設定
3. Modulator Levelを50%から徐々に上げる
4. 音色の変化を注意深く聴く
5. 好みのポイントでLevel固定
6. 他のModulatorも同様に設定
```

### ベルサウンドの種類別レシピ

```
1. 教会の鐘（Church Bell）:
   特徴: 重厚、長い余韻、低域豊か
   Carrier Ratio: 1.00
   Mod 1 Ratio: 2.00, Fine +35cent（≈2.5）
   Mod 2 Ratio: 3.00, Fine +45cent（≈3.8）
   Mod 3 Ratio: 5.00, Fine +25cent（≈5.4）
   Decay: 5000ms以上
   Reverb: Cathedral, 12s

2. 風鈴（Wind Chime）:
   特徴: 高域、短い余韻、軽やか
   Carrier Ratio: 2.00
   Mod 1 Ratio: 5.76, Level 40%
   Mod 2 Ratio: 8.23, Level 25%
   Mod 3 Ratio: 12.41, Level 15%
   Decay: 1200ms
   HP Filter: 500Hz

3. チベタンボウル（Singing Bowl）:
   特徴: 持続音、うなり、瞑想的
   Carrier Ratio: 1.00
   Mod 1 Ratio: 2.24（≈√5）, Level 55%
   Mod 2 Ratio: 3.00, Fine +8cent, Level 45%
   Mod 3 Ratio: 1.00, Fine +3cent, Level 30%
   Sustain: 80%（長い持続）
   LFO → Pitch: Rate 0.5Hz, Depth 1%（うなり）

4. ゲームラン（Gamelan）:
   特徴: 明るい打撃音、独特の倍音
   Carrier Ratio: 1.00
   Mod 1 Ratio: 2.41, Level 70%
   Mod 2 Ratio: 3.89, Level 50%
   Mod 3 Ratio: 5.93, Level 35%
   Decay: 2000ms
   Pitch Env: +2st, D:15ms

5. スチールドラム（Steel Pan）:
   特徴: カリブ的、明るい、リズミカル
   Carrier Ratio: 1.00
   Mod 1 Ratio: 2.00, Level 60%
   Mod 2 Ratio: 3.00, Level 45%
   Mod 3 Ratio: 4.50, Level 30%
   Decay: 800ms
   Pitch Env: +4st, D:10ms
   Filter: LP 6000Hz
```

---

## FMベース設計ガイド

### FMベースの基本構造

```
FMベースが減算式ベースと異なる点:

減算式ベース:
Saw/Square波 → Filter → 太いベース
→ フィルターでスペクトルを削る

FMベース:
Sine波 + 変調 → 独特のエッジ
→ 変調でスペクトルを追加する

FMベースの利点:
1. CPU負荷が低い（Sine波ベース）
2. 独特のアタック感
3. クリーンな低域
4. 変調量でキャラクター変化

FMベースの種類:
- サブベース（Sub Bass）
- アシッドベース（Acid-style）
- リーシーベース（Reese Bass）
- プラックベース（Pluck Bass）
- ウォブルベース（Wobble Bass）
```

### FMベース レシピ集

```
1. クリーンサブベース:
   Algorithm: 1
   OSC A: Ratio 0.5, Level 100%（1オクターブ下）
     Env: A:5, D:200, S:90%, R:100
   OSC B: Ratio 1.00, Level 30%, Feedback 10%
     Env: A:0, D:100, S:0%, R:50
   OSC C: Off
   OSC D: Off
   Filter: LP 200Hz
   用途: 低域の土台、どんなジャンルでも

2. パンチFMベース:
   Algorithm: 4
   ペア1:
   OSC A: Ratio 1.00, Level 70%
     Env: A:0, D:200, S:0%, R:100
   OSC B: Ratio 1.00, Level 100%
     Env: A:0, D:500, S:50%, R:200
   ペア2:
   OSC C: Ratio 3.00, Level 50%
     Env: A:0, D:80, S:0%, R:40
   OSC D: Ratio 0.5, Level 80%
     Env: A:0, D:400, S:60%, R:150
   Pitch Env: +12st, D:20ms
   用途: House, Tech House

3. アシッドFMベース:
   Algorithm: 1（直列）
   OSC A: Ratio 1.00, Level 80%, Feedback 30%
     Env: A:0, D:400, S:40%, R:200
   OSC B: Ratio 2.00, Level 70%
     Env: A:0, D:300, S:30%, R:150
   OSC C: Ratio 1.00, Level 50%
     Env: A:0, D:500, S:50%, R:250
   OSC D: Ratio 1.00, Level 100%（Carrier）
     Env: A:0, D:600, S:60%, R:300
   Filter: LP 24dB, Cutoff 800Hz, Res 50%
   Filter Env: A:0, D:400, S:20%, R:200, Amount +60
   用途: Acid Techno, Acid House

4. リーシーFMベース:
   Algorithm: 4
   ペア1:
   OSC A: Ratio 1.00, Level 60%, Fine +7cent
     Env: A:10, D:800, S:70%, R:400
   OSC B: Ratio 1.00, Level 100%, Fine 0cent
     Env: A:10, D:1000, S:75%, R:500
   ペア2:
   OSC C: Ratio 1.00, Level 55%, Fine -7cent
     Env: A:10, D:800, S:70%, R:400
   OSC D: Ratio 1.00, Level 90%, Fine +3cent
     Env: A:10, D:1000, S:75%, R:500
   Detune効果で厚みを出す
   用途: Drum & Bass, Jungle

5. ウォブルFMベース:
   Algorithm: 1
   OSC A-D: 上記アグレッシブベース設定
   LFO → Filter Cutoff:
     Rate: 1/4（BPM同期）
     Depth: 60%
   LFO → OSC B Level:
     Rate: 1/4
     Depth: 40%
   用途: Dubstep, Bass Music
```

---

## FMパッド作成テクニック

### パッドに適したFM設計原則

```
パッド作成の基本原則:

1. 長いAttack（500ms-3000ms）
   → ゆっくり立ち上がる音

2. 高いSustain（60%-90%）
   → 長く持続する音

3. 長いRelease（2000ms-8000ms）
   → ゆっくり消える音

4. 穏やかな変調量
   → Modulator Level 20-50%
   → 激しすぎない倍音

5. Detuneの活用
   → Fine ±5-15cent
   → コーラス効果でステレオ感

6. エフェクト重視
   → Reverb大量
   → Chorus/Ensemble
   → Delay（空間の奥行き）
```

### FMパッド レシピ集

```
1. シネマティック・ストリングスパッド:
   Algorithm: 8
   OSC A: Ratio 1.00, Level 100%（Carrier 1）
     Env: A:1500, D:3000, S:80%, R:4000
   OSC B: Ratio 2.00, Level 80%（Carrier 2）
     Env: A:2000, D:3500, S:75%, R:4500
   OSC C: Ratio 1.00, Level 35%（Modulator）
     Env: A:1000, D:2000, S:50%, R:3000
   OSC D: Ratio 3.00, Fine +5cent, Level 50%（Carrier 3）
     Env: A:2500, D:4000, S:70%, R:5000
   Chorus: Rate 0.4Hz, Amount 30%
   Reverb: Hall, Decay 6s, Wet 40%
   用途: Film Score, Cinematic, Ambient

2. エーテルパッド（Ethereal Pad）:
   Algorithm: 11
   OSC A: Ratio 1.00, Level 100%（Carrier）
     Env: A:3000, D:5000, S:85%, R:8000
   OSC B: Ratio 2.00, Fine +12cent, Level 30%
     Env: A:2000, D:4000, S:60%, R:6000
   OSC C: Ratio 3.00, Fine -8cent, Level 20%
     Env: A:2500, D:4500, S:50%, R:7000
   OSC D: Ratio 5.00, Fine +15cent, Level 15%
     Env: A:3500, D:6000, S:40%, R:9000
   Reverb: Shimmer, Decay 12s, Wet 55%
   Delay: 1/4D, Feedback 40%, Wet 20%
   用途: Ambient, Meditation, Drone

3. ダークドローン:
   Algorithm: 1（直列）
   OSC A: Ratio 0.5, Level 70%, Feedback 35%
     Env: A:4000, D:8000, S:90%, R:10000
   OSC B: Ratio 1.00, Level 55%
     Env: A:3000, D:6000, S:80%, R:8000
   OSC C: Ratio 0.5, Fine +3cent, Level 40%
     Env: A:5000, D:10000, S:85%, R:12000
   OSC D: Ratio 1.00, Level 100%（Carrier）
     Env: A:4000, D:8000, S:90%, R:10000
   LFO → OSC A Level: Rate 0.1Hz, Depth 20%
   Filter: LP 800Hz
   Reverb: Dark Room, Decay 15s, Wet 60%
   用途: Dark Ambient, Horror, Experimental
```

---

## モジュレーションマトリクス活用法

### Operatorでのマクロコントロール設計

```
Ableton LiveのMacroコントロールとOperatorの連携:

Macro 1: 「Brightness（明るさ）」
  → Modulator Level（全Modulator）
  → Filter Cutoff
  効果: 1つのノブで音色の明るさを制御

Macro 2: 「Decay（減衰）」
  → 全OSC Decay Time
  → Reverb Decay
  効果: 音の長さを一括制御

Macro 3: 「Metallic（金属感）」
  → Modulator Fine Tuning
  → Modulator Ratio微調整
  効果: 倍音の協和/不協和を制御

Macro 4: 「Movement（動き）」
  → LFO Rate
  → LFO Depth
  効果: 音色の揺れを制御

Macro 5: 「Space（空間）」
  → Reverb Dry/Wet
  → Delay Dry/Wet
  → Chorus Amount
  効果: 空間の広がりを制御

Macro 6: 「Attack（アタック）」
  → 全OSC Attack Time
  → Pitch Envelope Amount
  効果: 音の立ち上がりを制御

Macro 7: 「Character（キャラクター）」
  → Feedback Amount
  → Saturator Drive
  効果: 音の荒さを制御

Macro 8: 「Detune（デチューン）」
  → 各OSC Fine値
  効果: ステレオ感とコーラス効果

設定手順:
1. Operatorを含むInstrument Rackを作成
2. Macroノブにパラメータをマッピング
3. 各マッピングのMin/Max範囲を調整
4. プリセットとして保存
```

### LFOとエンベロープの高度なルーティング

```
Operator内蔵LFOの高度な活用:

LFO → OSC Level（Tremolo効果）:
  Rate: 3-8Hz
  → 音量の周期的変化
  → ビブラフォン風のトレモロ

LFO → OSC Frequency（Vibrato効果）:
  Rate: 4-7Hz, Depth: 1-5%
  → ピッチの周期的変化
  → 楽器のビブラート再現

LFO → Filter Cutoff（Wah効果）:
  Rate: BPM同期 1/8 or 1/4
  → フィルターの周期的開閉
  → リズミカルな音色変化

外部LFO（Ableton LFO デバイス）の活用:

利点:
- 複数のLFOを使用可能
- より多くの波形選択
- BPM同期の精密制御
- Envelopeフォロワーとの組み合わせ

推奨構成:
LFO 1 → Modulator 1 Level（音色変化）
LFO 2 → Filter Cutoff（フィルター変化）
LFO 3 → Panning（定位変化）
Envelope Follower → Modulator 2 Level（ダイナミック変調）

実践例 - ライブパフォーマンス用パッチ:
LFO（Rate: 1/2, Sine）→ OSC B Level
  → 2拍ごとの音色変化
LFO（Rate: 1/16, Square）→ OSC C Level
  → 16分音符のリズミックな倍音追加
Macro → LFO Depth
  → ライブ中にノブで変化量制御
```

---

## ジャンル別FMサウンド設計

### Techno向けFMサウンド

```
Technoでよく使われるFMサウンド:

1. メタリックパーカッション:
   Algorithm: 11
   全OSC: Fixed On（ピッチ固定）
   OSC A: 200Hz, Level 100%
   OSC B: 547Hz, Level 60%
   OSC C: 1123Hz, Level 40%
   OSC D: 2789Hz, Level 25%
   全Decay: 200-500ms
   Distortion追加
   用途: ハイハット代替、パーカッション

2. インダストリアルヒット:
   Algorithm: 1
   非整数Ratio多用
   短いDecay（50-200ms）
   高いFeedback（50-80%）
   Distortion + Compressor
   用途: ワンショット、アクセント

3. テクノリード:
   Algorithm: 4
   Ratio: 整数（1, 2, 3）
   Filter: BP 2000Hz, Res 40%
   LFO → Filter: Rate 1/16
   用途: メインリード、アルペジオ

4. アトモスフェリックFX:
   Algorithm: 7
   非整数Ratio（π, e, φ）
   長いAttack/Release
   Reverb大量
   Delay: Ping Pong
   用途: バックグラウンドテクスチャ
```

### IDM/Experimental向けFMサウンド

```
IDM/Experimentalでの高度な活用:

1. グリッチベル:
   Algorithm: 11
   基本はベル音設定
   追加: LFO → Modulator Ratio
     Rate: Random
     Depth: 0.5
   効果: ピッチが微妙に不安定なベル
   → Autechre風のサウンド

2. モーフィングパッド:
   Algorithm: 8
   Macro 1 → Algorithm切り替え（自動化）
   Macro 2 → 全Ratio同時変更
   Macro 3 → Envelope Time同時変更
   効果: 時間とともに音色が劇的に変化
   → Aphex Twin風の音響実験

3. ランダムFMパーカッション:
   Random LFO → 各OSC Ratio
   Random LFO → 各OSC Level
   Fixed周波数使用
   短いDecay
   効果: 毎回異なるパーカッション音
   → Autechre/Squarepusher風

4. マイクロサウンド:
   非常に短いエンベロープ（1-50ms）
   高いRatio（8以上）
   グラニュラー的テクスチャ
   Delay: 短いフィードバック（20-50ms）
   → Curtis Roads風のマイクロサウンド
```

### Lo-Fi / Neo Soul向けFMサウンド

```
Lo-Fi / Neo Soulでの定番FM音色:

1. ウォーム・エレピ:
   前述のRhodesパッチ + 以下の加工
   Saturator: Drive 3dB, Warm
   Bit Reduction: 14bit
   Sample Rate Reduction: 32000Hz
   Auto Filter: LP, Cutoff 4000Hz
   用途: Lo-Fi Hip Hop定番

2. ベルキーボード:
   Algorithm: 11
   穏やかな非整数Ratio
   Decay: 1500ms
   Chorus + Phaser
   Vinyl Distortion: Wet 15%
   用途: チルアウト、BGM

3. ソフトFMベース:
   Algorithm: 4
   低い変調量（Modulator Level 30%以下）
   Filter: LP 1500Hz
   Saturator: Soft Clip
   用途: Neo Soul, R&B

4. FMクラビネット:
   Algorithm: 1
   OSC A: Ratio 1, Feedback 20%
   OSC B: Ratio 3, Level 55%
   短いDecay（300ms）
   Auto Wah: Envelope Follow
   用途: Funk, Neo Soul
```

### Ambient / Cinematic向けFMサウンド

```
Ambient/Cinematicでの活用:

1. エボルビングテクスチャ:
   Algorithm: 11
   超長いエンベロープ（Attack 5-20秒）
   LFO → Modulator Level: 極低速（0.02-0.1Hz）
   LFO → Fine Tuning: 極低速
   Reverb: 20秒以上
   Delay: 長いフィードバック（70%+）
   → Brian Eno風のジェネラティブ音響

2. クリスタルアルペジオ:
   Algorithm: 4
   ベル系設定 + アルペジエーター
   Arp Rate: 1/8 or 1/16
   Reverb: Shimmer, 10s
   Delay: Dotted 1/8
   → Steve Reich風のミニマリスト音響

3. サブハーモニックドローン:
   Algorithm: 1
   Carrier Ratio: 0.5（オクターブ下）
   Modulator Ratio: 0.25
   極長エンベロープ
   LFO → 全パラメータ微量変調
   → 瞑想的なドローン

4. スペースFX:
   Algorithm: 7
   非整数Ratio（宇宙的な響き）
   Pitch LFO: 極低速（0.01Hz）
   Reverb: Infinite
   Granulator連携
   → SF映画風サウンドスケープ
```

---

## 実践演習プログラム

### 演習1: Ratioの聴き比べ（30分）

```
目的: 各Ratioが音色にどう影響するか体感する

準備:
1. Operator追加、Algorithm 11
2. OSC A（Carrier）: Ratio 1.00, Level 100%
   Env: A:0, D:2000, S:0%, R:2000
3. OSC B（Modulator）: Ratio 1.00, Level 70%
   Env: A:0, D:1500, S:0%, R:1500
4. OSC C, D: Off

手順:
Step 1: 整数Ratioの聴き比べ（15分）
  OSC B Ratio → 1.00 → 鳴らす → メモ
  OSC B Ratio → 2.00 → 鳴らす → メモ
  OSC B Ratio → 3.00 → 鳴らす → メモ
  OSC B Ratio → 4.00 → 鳴らす → メモ
  OSC B Ratio → 6.00 → 鳴らす → メモ
  OSC B Ratio → 8.00 → 鳴らす → メモ

Step 2: 非整数Ratioの聴き比べ（15分）
  OSC B Ratio → 1.41 → 鳴らす → メモ
  OSC B Ratio → 1.62 → 鳴らす → メモ
  OSC B Ratio → 2.24 → 鳴らす → メモ
  OSC B Ratio → 2.72 → 鳴らす → メモ
  OSC B Ratio → 3.14 → 鳴らす → メモ
  OSC B Ratio → 5.83 → 鳴らす → メモ

記録テンプレート:
Ratio [___]: 音色の印象 [________________]
            似ている楽器 [________________]
            使えそうなジャンル [________________]
```

### 演習2: Algorithm全探索（45分）

```
目的: 11種類のAlgorithmの音色の違いを理解する

準備:
全OSC同一設定で固定:
  Waveform: Sine
  Level: 70%
  Coarse: 2.00
  Env: A:0, D:1000, S:30%, R:500

手順:
Algorithm 1 → 鳴らす → 印象メモ → 5分
Algorithm 2 → 鳴らす → 印象メモ → 5分
...
Algorithm 11 → 鳴らす → 印象メモ → 5分

（途中休憩5分）

振り返り（10分）:
最も気に入ったAlgorithm: [___]
最も複雑だったAlgorithm: [___]
最もシンプルだったAlgorithm: [___]
ベル向きのAlgorithm: [___]
ベース向きのAlgorithm: [___]
パッド向きのAlgorithm: [___]
```

### 演習3: 10分チャレンジ（各音色10分で作成）

```
目的: 時間制限で直感的な音作りを練習

チャレンジ1: ベル音を10分で作る
  制限: Algorithm 11固定
  目標: ベルの「チーン」音が鳴る
  評価: □鳴った □鳴らなかった

チャレンジ2: エレピ音を10分で作る
  制限: Algorithm 4固定
  目標: Rhodes風の音が鳴る
  評価: □鳴った □鳴らなかった

チャレンジ3: ブラス音を10分で作る
  制限: Algorithm 8固定
  目標: トランペット風の音が鳴る
  評価: □鳴った □鳴らなかった

チャレンジ4: 金属パーカッションを10分で作る
  制限: Fixed On使用
  目標: ハイハット風の音が鳴る
  評価: □鳴った □鳴らなかった

チャレンジ5: パッド音を10分で作る
  制限: Attack 1000ms以上
  目標: ゆっくり立ち上がるパッド
  評価: □鳴った □鳴らなかった

合格基準:
5つ中3つ以上「鳴った」→ 合格
5つ中4つ以上 → 優秀
5つ全部 → FM合成マスター
```

### 演習4: リファレンス再現チャレンジ（60分）

```
目的: 既存楽曲のFMサウンドを再現する

推奨リファレンス曲:

1. Brian Eno - "An Ending (Ascent)"
   目標音色: FMパッド
   ヒント: 長いAttack、穏やかな変調、大量Reverb
   難易度: ★★☆☆☆

2. Aphex Twin - "Xtal"
   目標音色: アンビエントベル
   ヒント: 非整数Ratio、Delay多用
   難易度: ★★★☆☆

3. Herbie Hancock - "Rockit"
   目標音色: DX7エレピ
   ヒント: Algorithm 4、Ratio 14のアタック音
   難易度: ★★★★☆

4. Autechre - "Eutow"
   目標音色: グリッチFMテクスチャ
   ヒント: ランダムLFO、非整数Ratio、短いEnvelope
   難易度: ★★★★★

手順:
1. リファレンス曲を繰り返し聴く（10分）
2. 目標音色の特徴を言語化する（5分）
3. Operatorで再現を試みる（35分）
4. リファレンスと比較（10分）

評価基準:
30%似ている → 良い出発点
50%似ている → 成功
70%以上 → 素晴らしい
```

### 演習5: ジャンル別ミニトラック作成（120分）

```
目的: FM合成を使った楽曲制作の実践

課題: 以下の4ジャンルからひとつ選び、
OperatorのFM音色を最低2種類使ったミニトラック（16小節）を完成させる

選択肢A: Ambient（推奨所要時間120分）
  必須FM音色: パッド + ベル
  テンポ: 70-90 BPM
  構成: パッド持続 + ベルアルペジオ
  追加要素: Reverb、Delay

選択肢B: Lo-Fi Hip Hop（推奨所要時間90分）
  必須FM音色: エレピ + サブベース
  テンポ: 80-90 BPM
  構成: エレピコード + ベースライン
  追加要素: ドラムループ、Vinyl FX

選択肢C: Techno（推奨所要時間100分）
  必須FM音色: メタリックパーカッション + FMベース
  テンポ: 128-135 BPM
  構成: キック + FMパーカッション + ベースライン
  追加要素: ハイハット、FX

選択肢D: IDM/Experimental（推奨所要時間120分）
  必須FM音色: グリッチベル + モーフィングパッド
  テンポ: 自由
  構成: 実験的、自由形式
  追加要素: 自動化、ランダマイゼーション

完成チェックリスト:
□ FM音色が最低2種類使われている
□ 16小節以上ある
□ ミックスバランスが取れている
□ エフェクトが適切に使われている
□ オリジナリティがある
```

---

## トラブルシューティング

### FM合成でよくある問題と解決策

```
問題1: 音が歪む/割れる
原因: Modulator Levelが高すぎる
解決: Modulator Level を 50% 以下に下げる
     または Operator全体のVolumeを下げる

問題2: 音がノイズのように聞こえる
原因: 変調指数が大きすぎる/非整数Ratioが極端
解決: Modulator Levelを下げる
     Ratioを整数に近づける
     Feedbackを下げる

問題3: 音程が不安定
原因: Modulatorが Fixed Off で非整数Ratio
解決: 意図的でなければRatioを整数にする
     Fixed Onの場合は周波数を確認

問題4: 音が薄い/弱い
原因: Carrierが少ない/Modulator Levelが低い
解決: Algorithm変更で複数Carrier使用
     Modulator Levelを徐々に上げる
     Saturator/Compressorで音圧追加

問題5: CPUが高い
原因: 一般的にOperatorはCPU効率が良い
解決: Oversampling設定を確認
     不要なOSCをOffに
     エフェクトチェインの見直し

問題6: 音がこもる
原因: 高域の倍音不足
解決: Modulator Ratioを上げる（3以上）
     Modulator Levelを上げる
     EQで高域ブースト
     Filter Cutoffを上げる

問題7: ベル音がベルに聞こえない
原因: Ratioが整数のまま
解決: Fine Tuningで非整数化（+30-50cent）
     Decay/Releaseを長くする（2000ms以上）
     Reverbを追加する
```

### パラメータ早見表

```
目的別パラメータ設定ガイド:

明るい音にしたい:
→ Modulator Ratio ↑（2以上）
→ Modulator Level ↑（60%以上）
→ Filter Cutoff ↑

暗い音にしたい:
→ Modulator Level ↓（30%以下）
→ Filter Cutoff ↓
→ 高域OSCをOff

金属的にしたい:
→ 非整数Ratio使用
→ Modulator Level ↑
→ Feedback ↑

柔らかくしたい:
→ 整数Ratio使用
→ Modulator Level ↓
→ Attack ↑（10-100ms）

パーカッシブにしたい:
→ Decay短く（100-300ms）
→ Sustain: 0%
→ Pitch Envelope追加

持続音にしたい:
→ Sustain: 60-90%
→ Release: 2000ms以上
→ Attack: 500ms以上（パッド）

太くしたい:
→ Ratio 0.5使用（サブオクターブ）
→ 複数Carrier使用
→ Saturator追加
→ Detune ±5-10cent
```

---

## 次に読むべきガイド

- [Genre Sounds（ジャンル別サウンド）](./genre-sounds.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
