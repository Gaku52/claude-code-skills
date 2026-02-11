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
