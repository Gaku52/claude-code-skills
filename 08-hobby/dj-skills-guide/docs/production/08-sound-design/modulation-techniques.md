# Modulation Techniques（モジュレーション技術）

LFO、Envelope、Macro、MIDI CCを駆使した高度なモジュレーション技術を完全マスター。動きのあるサウンド、演奏性の高いパッチで、プロレベルの表現力を実現します。

## この章で学ぶこと

- LFOモジュレーション深掘り
- Envelope Follower活用
- Macroノブ高度な使い方
- MIDI CCマッピング
- パフォーマンス志向のモジュレーション
- 複数パラメータ同時制御
- 実践的な練習方法

---

## なぜModulation Techniquesが重要なのか

**静的 vs 動的:**

```
モジュレーションなし:

状況:
静的なサウンド
変化なし
単調

結果:
つまらない
飽きる
素人っぽい

モジュレーションあり:

状況:
動的なサウンド
時間的変化
周期的変化

結果:
興味深い
飽きない
プロっぽい

プロの使用率:

LFO使用率: 85%
Envelope Follower: 40%
Macro Knob: 95%
MIDI CC: 70%

理由:
動きが必須
演奏性重要
```

**演奏性の重要性:**

```
演奏性なし:

状況:
固定パラメータ
オートメーションのみ
リアルタイム制御不可

結果:
時間かかる
創造性低い
ライブ不可

演奏性あり:

状況:
Macroノブ
MIDI CC
リアルタイム制御

結果:
即座に変化
創造性高い
ライブ可能
```

---

## LFOモジュレーション深掘り

### LFOの基本復習

```
LFO（Low Frequency Oscillator）:

定義:
周期的な変化を生成
0.01-40 Hz（音として聞こえない）

パラメータ:
1. Destination（どこを変調するか）
2. Waveform（波形）
3. Rate（速度）
4. Depth（深さ）

用途:
ビブラート、トレモロ、ワウワウ、オートパン
```

### LFO Destination（変調先）

**Pitch（ピッチ）:**

```
効果: ビブラート

設定:
Destination: OSC Pitch
Waveform: Sine
Rate: 5 Hz
Depth: 4%

用途:
リード、パッド、ボーカル

バリエーション:

控えめビブラート:
Rate: 5 Hz
Depth: 2-3%
→ 自然

強いビブラート:
Rate: 6 Hz
Depth: 8-10%
→ 極端、EDM

ランダムピッチ:
Waveform: Random S&H
Rate: 0.5 Hz
Depth: 5%
→ 予測不可能、実験的
```

**Filter Cutoff:**

```
効果: ワウワウ

設定:
Destination: Filter Cutoff
Waveform: Triangle
Rate: 1/4（テンポ同期）
Depth: 40%

用途:
Techno Bass、Funkベース、リード

バリエーション:

ゆっくりワウワウ:
Rate: 1/2
Depth: 30%
→ Ambient

速いワウワウ:
Rate: 1/16
Depth: 50%
→ Techno

ランダムワウワウ:
Waveform: Random S&H
Rate: 1/8
Depth: 35%
→ IDM
```

**Volume（音量）:**

```
効果: トレモロ

設定:
Destination: Amp Volume
Waveform: Sine
Rate: 8 Hz
Depth: 25%

用途:
パッド、FX、Ambient

バリエーション:

ゲート効果:
Waveform: Square
Rate: 1/16
Depth: 100%
→ オン/オフ、Techno

呼吸効果:
Waveform: Sine
Rate: 0.2 Hz
Depth: 15%
→ Ambient、Drone

リズミック:
Rate: 1/8 Dotted
Depth: 40%
→ テンポに同期
```

**Pan（定位）:**

```
効果: オートパン

設定:
Destination: Pan
Waveform: Sine
Rate: 1/8
Depth: 70%

用途:
パッド、FX、Hi-Hat

バリエーション:

ゆっくり移動:
Rate: 1/2
Depth: 100%
→ 左右にゆっくり

速い移動:
Rate: 1/16
Depth: 80%
→ 細かく動く

ランダム定位:
Waveform: Random S&H
Rate: 1/4
Depth: 60%
→ 予測不可能
```

### LFO Waveform（波形）選択

```
Sine（サイン波）:

特徴: 滑らか
用途: ビブラート、ワウワウ
推奨: 自然な変化

Triangle（三角波）:

特徴: やや角ばった
用途: ワウワウ、オートパン
推奨: Sineより少し鋭い変化

Square（矩形波）:

特徴: カクカク、オン/オフ
用途: ゲート効果、リズミック
推奨: 極端な変化

Saw Up（上昇のこぎり波）:

特徴: ゆっくり上昇、急降下
用途: ビルドアップ、Riser
推奨: 方向性のある変化

Saw Down（下降のこぎり波）:

特徴: 急上昇、ゆっくり下降
用途: Drop、インパクト
推奨: 逆方向の変化

Random S&H（ランダム）:

特徴: ランダム、予測不可能
用途: 実験的、IDM
推奨: 有機的な変化
```

### 複数LFO使用

```
3 LFO同時使用例:

パッチ: Ambient Pad

LFO 1:
Destination: Filter Cutoff
Waveform: Sine
Rate: 0.3 Hz
Depth: 20%
→ 音色がゆっくり変化

LFO 2:
Destination: Pan
Waveform: Triangle
Rate: 0.15 Hz
Depth: 60%
→ 左右に動く

LFO 3:
Destination: Amp Volume
Waveform: Sine
Rate: 0.12 Hz
Depth: 10%
→ 呼吸感

結果:
音色、定位、音量が独立して変化
→ 非常に動的なパッド
```

---

## Envelope Follower（エンベロープフォロワー）

### 基本原理

```
Envelope Follower:

定義:
オーディオ信号の音量変化を検出
→ 他のパラメータを制御

仕組み:
Input Audio → Envelope検出 → Destination

用途:
サイドチェイン的効果
リズミックな変調
ダッキング
```

### 実践: キックでBassをダッキング

```
設定（Ableton Live 12）:

手順:

1. Bassトラックに Auto Filter追加

2. Sidechain設定:
   Sidechain Input: Kickトラック
   Envelope Follower: On

3. パラメータ:
   Destination: Filter Cutoff
   Gain: -40 dB
   Attack: 5 ms
   Release: 150 ms

動作:
Kick鳴る → Bass Filter閉じる
Kick消える → Bass Filter開く

効果:
Kickが鳴る時、Bassが引っ込む
→ Kickとの住み分け
→ リズム感

用途:
House、Techno、EDM全般
```

### 実践: ドラムでSynthを揺らす

```
パッチ: Pad

設定:

1. Padトラックに Wavetable

2. Macro設定:
   Macro 1 → OSC 1 Volume
   範囲: 100% - 70%

3. Envelope Follower:
   Input: ドラムループ
   Destination: Macro 1
   Gain: -15 dB
   Attack: 10 ms
   Release: 200 ms

動作:
ドラム鳴る → Pad音量下がる
ドラム消える → Pad音量戻る

効果:
Padがドラムに反応して揺れる
→ リズム感
→ 一体感
```

### 実践: ボーカルでReverbを制御

```
設定:

1. ボーカルトラックに Reverb追加

2. Macro設定:
   Macro 1 → Reverb Dry/Wet
   範囲: 15% - 45%

3. Envelope Follower:
   Input: ボーカル自身
   Destination: Macro 1（逆）
   Gain: +20 dB
   Attack: 50 ms
   Release: 300 ms

動作:
ボーカル強い → Reverb減る
ボーカル弱い → Reverb増える

効果:
強い音: クリア（Reverb少ない）
弱い音: 空間的（Reverb多い）
→ ダイナミックな空間感
```

---

## Macroノブ高度な使い方

### 複数パラメータマッピング

```
1つのMacroで複数制御:

例: "Brightness" Macro

Macro 1 → 以下を同時制御:

1. Filter Cutoff: 500 Hz - 3500 Hz
2. Resonance: 5% - 30%
3. Drive: 0 dB - 8 dB
4. High Shelf EQ: 0 dB - +3 dB

効果:
1つのノブで「明るさ」を総合的に制御
→ 演奏性向上
→ 創造的
```

### Macro範囲設定のコツ

```
良い範囲設定:

原則:
実際に使う範囲のみ

例:
Filter Cutoff:
Min: 200 Hz（使用最低値）
Max: 2500 Hz（使用最高値）

NG:
Min: 20 Hz
Max: 20000 Hz
→ 範囲広すぎ、微調整困難

推奨:
実用範囲の±20%
```

### 逆マッピング

```
Macroノブを上げる → パラメータ下がる

用途:

例: "Darkness" Macro

Macro 1 → Filter Cutoff（逆）
Macro 0% = Cutoff 3000 Hz（明るい）
Macro 100% = Cutoff 500 Hz（暗い）

設定方法:
Macroマッピング時に Min/Max を逆にする
```

### Macroチェイン

```
Macro 1 → Macro 2 → パラメータ

用途:
複雑な制御
非線形な変化

例:
Macro 1（演奏用）
 ↓
Macro 2（内部）
 ↓
Filter Cutoff、Resonance、Drive

Macro 1を動かす
→ Macro 2が動く
→ 複数パラメータが連動
```

---

## MIDI CCマッピング

### MIDI CC基本

```
MIDI CC（Control Change）:

定義:
MIDIコントローラー（ノブ、スライダー）からの制御信号
CC 1-127

代表的なCC:

CC 1: Mod Wheel（モジュレーションホイール）
CC 7: Volume
CC 10: Pan
CC 11: Expression
CC 64: Sustain Pedal
CC 74: Filter Cutoff（多用）
```

### Mod Wheel（CC 1）活用

```
Mod Wheelでビブラート深さ制御:

設定:

1. Wavetableに LFO 1設定:
   Destination: Pitch
   Waveform: Sine
   Rate: 5.5 Hz
   Depth: 0%（初期値）

2. MIDI CC Mapping:
   CC 1 → LFO 1 Depth
   Min: 0%
   Max: 8%

動作:
Mod Wheel 0 → ビブラートなし
Mod Wheel 上げる → ビブラート増加

用途:
リード演奏
ライブパフォーマンス
```

### Mod WheelでFilter制御

```
設定:

1. Filter設定:
   Cutoff: 1500 Hz（初期値）

2. MIDI CC Mapping:
   CC 1 → Filter Cutoff
   Min: 500 Hz
   Max: 4000 Hz

動作:
Mod Wheel 0 → 暗い音
Mod Wheel 上げる → 明るい音

用途:
ワウワウ効果
演奏表現
```

### Velocity Sensitivity（ベロシティ感度）

```
Velocityで複数制御:

設定:

1. Velocity → Filter Cutoff:
   Min: 800 Hz（弱い音）
   Max: 2500 Hz（強い音）

2. Velocity → Filter Envelope Amount:
   Min: +10
   Max: +50

3. Velocity → Amp Volume:
   Min: -6 dB
   Max: 0 dB

動作:
弱く弾く → 暗い、柔らかい音
強く弾く → 明るい、強い音

効果:
リアルな演奏感
表現力大
```

### Aftertouch活用

```
Aftertouch（鍵盤を押した後の圧力）:

用途:
ビブラート
Filter開く
音量増加

設定例:

1. Aftertouch → LFO Depth:
   Min: 0%
   Max: 10%

2. Aftertouch → Filter Cutoff:
   Min: 0 Hz（変化なし）
   Max: +1000 Hz

動作:
鍵盤を押す → 通常の音
さらに押し込む → ビブラート+明るくなる

用途:
プロの演奏
表現力
```

---

## パフォーマンス志向のモジュレーション

### ライブ演奏用Macro設定

```
推奨Macro配置（ライブ用）:

Macro 1: Filter Cutoff
→ 最重要、最も使う

Macro 2: Resonance
→ 音色変化

Macro 3: Reverb/Delay Mix
→ 空間感

Macro 4: Drive/Saturation
→ 音圧、歪み

Macro 5: LFO Depth
→ 動きの深さ

Macro 6: Attack Time
→ アタック調整

Macro 7: Release Time
→ リリース調整

Macro 8: Volume
→ 全体音量

理由:
左から右に重要度順
筋肉記憶
高速操作
```

### MIDI Controller推奨マッピング

```
Push 2、LaunchpadでのMapping:

ノブ1-8: Macro 1-8
パッド: クリップ起動
フェーダー: トラック音量

APC40でのMapping:

ノブ上段: Macro 1-4（頻繁に使用）
ノブ下段: Macro 5-8（たまに使用）
フェーダー: トラック音量
ボタン: クリップ起動

理由:
物理的配置 = 重要度
エルゴノミクス
```

---

## 実践: 動きのあるTechno Bass

**目標:** LFO、Envelope、Macroを駆使した動的なベース

### Step 1: 基本パッチ作成（15分）

```
楽器: Wavetable

OSC 1:
Wavetable: Basic Shapes > Saw
Octave: 0

Filter:
Type: Low Pass 24 dB
Cutoff: 600 Hz（開始点）
Resonance: 65%

Filter Envelope:
A: 0 ms
D: 180 ms
S: 20%
R: 50 ms
Amount: +40

Amp Envelope:
A: 0 ms
D: 100 ms
S: 80%
R: 20 ms
```

### Step 2: LFO設定（15分）

```
LFO 1（Filter変調）:
Destination: Filter Cutoff
Waveform: Triangle
Rate: 1/16（テンポ同期）
Depth: 30%
Retrigger: On
→ 16分音符でワウワウ

LFO 2（Resonance変調）:
Destination: Resonance
Waveform: Sine
Rate: 1/4
Depth: 15%
Retrigger: Off
→ ゆっくりResonance変化

LFO 3（Pan変調）:
Destination: Pan
Waveform: Random S&H
Rate: 1/8
Depth: 20%
Retrigger: On
→ ランダムに左右移動（控えめ）
```

### Step 3: Macro設定（15分）

```
Macro 1 - Filter Sweep:
- Filter Cutoff: 200 Hz - 2500 Hz
- LFO 1 Depth: 0% - 50%
→ Cutoff範囲とLFO深さを同時制御

Macro 2 - Resonance:
- Resonance: 40% - 85%
- Drive: 0 dB - 10 dB
→ Resonanceと歪みを連動

Macro 3 - Movement:
- LFO 1 Rate: 1/32 - 1/8
- LFO 2 Depth: 0% - 30%
→ 動きの速さと深さ

Macro 4 - Chaos:
- LFO 3 Depth: 0% - 40%（Pan）
- Random Amount: 変化
→ ランダム性制御
```

### Step 4: エフェクト（12分）

```
エフェクトチェイン:

1. EQ Eight:
   High Pass: 35 Hz
   Peak: 80 Hz、+3 dB、Q 1.5

2. Saturator:
   Drive: 6 dB
   Curve: Warm
   Dry/Wet: 80%

3. Auto Filter（2個目）:
   LFO → Cutoff
   さらなる動き

4. Delay:
   Time: 1/8 Dotted
   Feedback: 20%
   Dry/Wet: 15%

5. Utility:
   Width: 10%（ほぼMono）
```

### Step 5: MIDI CCマッピング（10分）

```
MIDI Controller Mapping:

Mod Wheel（CC 1）:
→ Macro 1（Filter Sweep）

Expression（CC 11）:
→ Macro 2（Resonance）

Knob 1:
→ Macro 3（Movement）

Knob 2:
→ Macro 4（Chaos）

Velocity:
→ Filter Envelope Amount（+20 〜 +60）
```

### Step 6: 演奏（13分）

```
MIDIパターン（16小節）:

Bar 1-4:
A1 ──── ──── C2 ── E2 ──
Velocity: 100-127（変化）

Bar 5-8:
A1 ── C2 ── E2 ── G2 ──
Velocity: 80-120

Bar 9-12:
D2 ──── F2 ── A2 ── C3 ──
Velocity: 90-127

Bar 13-16:
A1 ── C2 ── E2 ── A1 ──
Velocity: 100-110

演奏技法:
Mod Wheelを徐々に上げる（Bar 1-8）
→ Filter開く、明るくなる

Mod Wheelを下げる（Bar 9-16）
→ Filter閉じる、暗くなる

確認:
□ 常に動いている
□ Mod Wheelで変化
□ Velocityで表情
□ ライブ演奏的
```

### 完成基準

```
□ LFO 3個設定
□ Macro 4個設定
□ MIDI CC Mapping
□ Velocity Sensitivity
□ 動きのある音
□ 演奏性高い

所要時間: 80分
```

---

## 実践: 表現力のあるリード

**目標:** Mod Wheel、Aftertouch、Velocityで演奏性の高いリード

### Step 1: 基本パッチ（20分）

```
楽器: Wavetable

OSC 1:
Wavetable: Modern Shapes > Formant Square
Octave: 0

OSC 2:
Wavetable: Basic Shapes > Saw
Detune: +9 cent
Volume: 65%

UNISON: 5 voices、Detune 18%

Filter:
Type: Low Pass 12 dB
Cutoff: 2200 Hz
Resonance: 20%

Amp Envelope:
A: 12 ms
D: 220 ms
S: 75%
R: 320 ms
```

### Step 2: ビブラートLFO（10分）

```
LFO 1（ビブラート）:
Destination: OSC 1 Pitch + OSC 2 Pitch
Waveform: Sine
Rate: 5.5 Hz
Depth: 0%（初期値、Mod Wheelで制御）
Retrigger: Off

LFO 2（Filter揺らぎ）:
Destination: Filter Cutoff
Waveform: Triangle
Rate: 0.3 Hz
Depth: 8%
Retrigger: Off
→ 微妙な音色変化
```

### Step 3: MIDI CC設定（20分）

```
Mod Wheel（CC 1）マッピング:

1. LFO 1 Depth:
   Min: 0%（ビブラートなし）
   Max: 8%（強いビブラート）

2. LFO 1 Rate:
   Min: 5 Hz
   Max: 6.5 Hz
   → 微妙に速くなる

Velocity マッピング:

1. Filter Cutoff:
   Min: 1800 Hz（弱い音）
   Max: 2800 Hz（強い音）

2. Amp Volume:
   Min: -4 dB
   Max: 0 dB

3. Filter Resonance:
   Min: 15%
   Max: 25%

Aftertouch マッピング:

1. Filter Cutoff:
   Min: 0 Hz（変化なし）
   Max: +800 Hz

2. Unison Detune:
   Min: 18%（変化なし）
   Max: 28%（広がる）
```

### Step 4: Macro設定（15分）

```
Macro 1 - Brightness:
- Filter Cutoff: 1000 Hz - 4000 Hz
- High Shelf EQ: 0 dB - +3 dB

Macro 2 - Vibrato:
- LFO 1 Depth: 0% - 10%
- LFO 1 Rate: 4 Hz - 7 Hz

Macro 3 - Width:
- Unison Detune: 10% - 30%
- Stereo Width: 100% - 150%

Macro 4 - Air:
- Reverb Dry/Wet: 10% - 40%
- Delay Dry/Wet: 5% - 25%
```

### Step 5: エフェクト（15分）

```
エフェクトチェイン:

1. EQ Eight:
   High Pass: 200 Hz

2. Chorus:
   Rate: 0.5 Hz
   Amount: 30%
   Dry/Wet: 35%

3. Reverb:
   Decay: 2.0s
   Dry/Wet: 22%

4. Delay:
   Time: 1/4
   Feedback: 18%
   Dry/Wet: 12%

5. EQ Eight（2個目）:
   Peak: 2500 Hz、+2 dB、Q 1.2

6. Utility:
   Width: 120%
```

### Step 6: 演奏練習（20分）

```
メロディ（8小節）:

Bar 1-2:
C4 ──── Eb4 ─ G4 ── Bb4 ─

演奏技法:
最初: Mod Wheel 0
→ ビブラートなし

G4: Mod Wheel 50%
→ ビブラート開始

Bb4: Mod Wheel 80% + Aftertouch
→ 強いビブラート + 明るくなる

Bar 3-4:
C5 ──── G4 ── Eb4 ─ C4 ───

C5: Velocity 120（強く）
→ 明るい音

Eb4-C4: Velocity 80-100（弱め）
→ 暗めの音

確認:
□ Mod Wheelで表情
□ Velocityで強弱
□ Aftertouchで変化
□ プロ級の演奏感
```

### 完成基準

```
□ Mod Wheel Mapping
□ Velocity Mapping
□ Aftertouch Mapping
□ Macro 4個
□ LFO 2個
□ 演奏性最高
□ 表現力大

所要時間: 100分
```

---

## よくある質問（FAQ）

**Q1: LFOのRateをどう設定すべきですか？**

```
A: 用途で変える

ビブラート:
Rate: 4-6 Hz
理由: 人間の自然なビブラート

ワウワウ:
Rate: 1/4 - 1/8（テンポ同期）
理由: リズムに合わせる

トレモロ:
Rate: 6-10 Hz
理由: 明瞭な音量変化

Ambient:
Rate: 0.1-0.5 Hz（ゆっくり）
理由: 微妙な変化

推奨:
最初は Syncモード
テンポに合う
```

**Q2: Macroに何をマッピングすべきですか？**

```
A: 演奏中に変えたいパラメータ

優先順位:

1位: Filter Cutoff（必須）
2位: Resonance
3位: Reverb/Delay Mix
4位: Drive/Saturation
5位: LFO Depth
6位: Attack/Release
7位: Unison Detune
8位: Volume

理由:
頻繁に使うパラメータ
演奏性重視

NG:
細かいパラメータ
めったに変えないもの
```

**Q3: Mod Wheelが使いにくいです**

```
A: 範囲設定を見直す

問題例:
Mod Wheel → Filter Cutoff
Min: 20 Hz
Max: 20000 Hz
→ 範囲広すぎ

解決:
Min: 500 Hz
Max: 2500 Hz
→ 実用範囲のみ

コツ:
実際に使う範囲を事前に確認
その範囲のみをMap
```

**Q4: Envelope FollowerとLFOの違いは？**

```
A: トリガーが違う

Envelope Follower:
トリガー: オーディオ信号
変化: 不規則（音に依存）
用途: サイドチェイン、リアクティブ

例:
Kick → Bass Filter閉じる

LFO:
トリガー: 時間
変化: 周期的
用途: ビブラート、ワウワウ

例:
1/4ごとにFilter開閉

使い分け:
反応的 = Envelope Follower
周期的 = LFO
```

**Q5: Velocity感度が低すぎます**

```
A: Velocity Curve調整

Ableton Live:

Preferences → Link/MIDI
→ MIDI Port
→ Input: Your Controller
→ Track: On
→ Remote: On

Velocity Curve:
Linear（線形）
Soft（柔らかい、感度高い）
Hard（硬い、感度低い）

推奨:
Soft Curve
弱い音も拾える

または:
パラメータマッピング範囲を広げる
Min: -12 dB
Max: 0 dB
→ 大きな変化
```

**Q6: ライブ演奏中にパラメータ変更したいです**

```
A: Macroと MIDIコントローラー

推奨セットアップ:

1. 重要パラメータを Macro 1-4にMap

2. MIDIコントローラー:
   ノブ1-4 → Macro 1-4

3. 筋肉記憶:
   Macro 1 = Filter（左端ノブ）
   Macro 2 = Resonance（左から2番目）
   常に同じ配置

4. ライブ練習:
   30分/日
   パラメータ変更しながら演奏

結果:
見なくても操作可能
プロのライブパフォーマンス
```

---

## 練習方法

### Week 1: LFOマスター

```
Day 1-2: ビブラート
1. 5種類のリードにLFO追加
2. Depth 2-10%で実験
3. 自然なビブラート探し

Day 3-4: ワウワウ
1. 5種類のベースにLFO追加
2. Rate 1/4、1/8、1/16
3. テンポに同期

Day 5-6: 複数LFO
1. 1つのパッチに LFO 3個
2. Pitch、Filter、Pan
3. Ambient Pad作成

Day 7: 実戦投入
1. 自分の楽曲でLFO使用
2. 動きのあるサウンド作成
3. 完成

目標:
LFOを自在に使える
```

### Week 2: Macroマスター

```
Day 1-2: 基本Macro
1. 10個のパッチ作成
2. 各パッチにMacro 4個設定
3. Filter、Resonance、Reverb、Drive

Day 3-4: 複数パラメータMapping
1. 1つのMacroに3-4パラメータMap
2. "Brightness"、"Darkness"等
3. 総合的な変化

Day 5-6: ライブ演奏
1. MIDIコントローラー接続
2. Macroにマッピング
3. 演奏しながらパラメータ変更

Day 7: パフォーマンス
1. 1曲をライブ演奏
2. Macro多用
3. 録画、確認

目標:
Macroを演奏の一部にする
```

### Week 3: MIDI CCマスター

```
Day 1-2: Mod Wheel
1. 5種類のリード作成
2. Mod Wheel → LFO Depth
3. 演奏表現

Day 3-4: Velocity
1. Velocity → Filter Cutoff
2. Velocity → Volume
3. 5種類のパッチ

Day 5-6: Aftertouch
1. Aftertouch対応コントローラー準備
2. Aftertouch → Filter、LFO
3. 演奏練習

Day 7: 総合演奏
1. Mod Wheel + Velocity + Aftertouch
2. 複合的な表現
3. 録音、確認

目標:
MIDI CCを演奏に活用
```

### Week 4: 総合練習

```
Day 1-2: 動的なベース
1. 実践1を完全再現
2. LFO、Macro、CC全使用
3. ライブ演奏

Day 3-4: 表現力のあるリード
1. 実践2を完全再現
2. Mod Wheel、Velocity、Aftertouch
3. メロディ演奏

Day 5-6: 楽曲制作
1. モジュレーション多用の楽曲
2. 全ての音が動く
3. ライブ演奏可能

Day 7: パフォーマンス動画
1. 楽曲をライブ演奏
2. パラメータ変更を見せる
3. SNSにアップ

目標:
モジュレーションを完全マスター
```

---

## まとめ

モジュレーション技術は、動きと演奏性を生む重要なスキルです。

**重要ポイント:**

1. **LFO**: 周期的変化、ビブラート・ワウワウ
2. **Envelope Follower**: オーディオに反応、サイドチェイン
3. **Macro**: 複数パラメータ同時制御、演奏性
4. **MIDI CC**: Mod Wheel、Velocity、Aftertouch
5. **演奏性**: ライブパフォーマンス重視

**学習順序:**
1. LFOマスター（Week 1）
2. Macroマスター（Week 2）
3. MIDI CCマスター（Week 3）
4. 総合練習（Week 4）

**推奨設定:**
- LFO: 2-3個/パッチ
- Macro: 4-8個/パッチ
- Mod Wheel: LFO Depth、Filter Cutoff
- Velocity: Filter Cutoff、Volume

**次のステップ:** [Genre Sounds（ジャンル別サウンド）](./genre-sounds.md) へ進む

---

## 関連ファイル

- **[Wavetable Sound Design](./wavetable-sound-design.md)** - Wavetableシンセ
- **[Synthesis Basics](./synthesis-basics.md)** - シンセシス基礎
- **[Genre Sounds](./genre-sounds.md)** - ジャンル別サウンド
- **[07-workflow/performance.md](../07-workflow/)** - ライブパフォーマンス

---

## LFOの高度活用テクニック

基本的なLFO操作をマスターした後は、より複雑で表現力のあるLFO活用法を学びましょう。ここでは、プロの現場で使われる高度なLFOテクニックを解説します。

### 複雑な波形（カスタムLFOシェイプ）

```
カスタムLFO波形の活用:

標準波形以外の選択肢:

1. Fade In / Fade Out波形:
   特徴: 徐々に強くなり、徐々に弱くなる
   用途: ビルドアップ、ブレイクダウン
   設定例:
   Ableton Wavetable → LFO Shape: Custom
   ポイントを手動で描画
   → ゆっくり立ち上がり、急に落ちる形
   効果: 自然なスウェル感

2. ステップ波形（階段状）:
   特徴: 離散的なステップで変化
   用途: アルペジオ的フィルター変化、グリッチ
   設定例:
   LFO Shape: Custom Steps
   4ステップ: 0%, 25%, 75%, 100%
   Rate: 1/4（テンポ同期）
   → 4拍ごとに異なるCutoff値
   効果: リズミックかつ予測可能な変化

3. Exponential Curve（指数関数的曲線）:
   特徴: 急激に上昇、ゆっくり下降（またはその逆）
   用途: パーカッシブなフィルター変化
   設定例:
   カスタムカーブ描画
   急上昇部: 10%の時間で0→100%
   緩下降部: 90%の時間で100→0%
   → キックのようなフィルター動き
   効果: インパクトのあるアタック

4. S字カーブ:
   特徴: 中央付近でゆっくり、両端で速い変化
   用途: 滑らかなフィルタースイープ
   設定例:
   中央50%の範囲: 変化ゆっくり
   両端25%の範囲: 変化速い
   → 中間値に留まりやすい
   効果: 自然な聴覚的印象

Ableton Live 12でのカスタムLFO作成手順:

1. Wavetable → LFO → Envelope表示
2. Shape欄のドロップダウンを開く
3. 「Draw」モードを選択
4. マウスで自由にポイントを配置
5. スナップ設定で精度調整
6. Rate / Depthを調整して確認
```

### LFOレートモジュレーション（LFO on LFO）

```
LFOの速度を別のLFOで変調する:

概念:
LFO 1 → パラメータ変調（Filter等）
LFO 2 → LFO 1のRate変調

効果:
LFO 1の速度が一定でなく、揺れる
→ より有機的な動き
→ 機械的でない変化

設定例 1: 揺れるワウワウ

LFO 1:
Destination: Filter Cutoff
Waveform: Triangle
Rate: 1/8（基準値）
Depth: 35%

LFO 2:
Destination: LFO 1 Rate
Waveform: Sine
Rate: 0.2 Hz
Depth: 20%

動作:
LFO 1のRateが 1/8 を中心に揺れる
→ 速くなったり遅くなったり
→ ワウワウの速度が有機的に変化
→ 人間が手動でノブを回しているような感覚

設定例 2: 加速するビブラート

LFO 1:
Destination: OSC Pitch
Waveform: Sine
Rate: 4 Hz（基準値）
Depth: 5%

LFO 2:
Destination: LFO 1 Rate
Waveform: Saw Up
Rate: 0.1 Hz（非常にゆっくり）
Depth: 40%

動作:
ビブラート速度が徐々に加速
→ 最大値に達したら急にリセット
→ 感情的な表現
→ クラシック音楽のビブラート手法に近い

設定例 3: ランダムなLFOスピード

LFO 1:
Destination: Filter Cutoff
Waveform: Sine
Rate: 1/4
Depth: 30%

LFO 2:
Destination: LFO 1 Rate
Waveform: Random S&H
Rate: 1/2
Depth: 50%

動作:
2拍ごとにLFO 1の速度がランダムに変わる
→ 予測不可能なフィルター動き
→ IDM、Glitch系に最適
```

### LFO Phase Offset（位相オフセット）

```
複数LFOの位相をずらす技法:

概念:
同じRate、同じWaveformのLFOを
位相をずらして使用
→ ステレオ効果、複雑なテクスチャ

設定例: ステレオワイドLFO

LFO 1（Left Channel）:
Destination: Left Pan / Left Volume
Waveform: Sine
Rate: 0.25 Hz
Depth: 15%
Phase: 0度

LFO 2（Right Channel）:
Destination: Right Pan / Right Volume
Waveform: Sine
Rate: 0.25 Hz
Depth: 15%
Phase: 180度（反転）

動作:
左チャンネルが上がる時、右チャンネルが下がる
→ 自然なステレオ揺れ
→ モノラルでは打ち消し合わない

応用: 3相LFO（120度間隔）

LFO 1: Phase 0度 → Filter Cutoff
LFO 2: Phase 120度 → Resonance
LFO 3: Phase 240度 → Drive

効果:
3つのパラメータが120度ずれて変化
→ 常にどれかが最大値、どれかが最小値
→ 複雑で有機的な音色変化
→ パッド、テクスチャに最適
```

---

## サイドチェインモジュレーション詳解

Envelope Followerの基本を発展させた、より高度なサイドチェインモジュレーション手法を解説します。

### マルチバンドサイドチェイン

```
周波数帯域ごとに異なるサイドチェイン反応:

概念:
低域、中域、高域それぞれに
異なるEnvelope Followerを適用
→ 帯域ごとに独立した反応

設定例: Padトラック

低域サイドチェイン（20-200 Hz）:
Input: Kick
Destination: Low Band Volume
Attack: 2 ms
Release: 100 ms
Gain: -20 dB
→ Kickの低域に即座に反応、素早く復帰

中域サイドチェイン（200-2000 Hz）:
Input: Snare
Destination: Mid Band Volume
Attack: 5 ms
Release: 200 ms
Gain: -10 dB
→ Snareの中域に反応、やや遅い復帰

高域サイドチェイン（2000-20000 Hz）:
Input: Hi-Hat
Destination: High Band Volume
Attack: 1 ms
Release: 50 ms
Gain: -8 dB
→ Hi-Hatに素早く反応

効果:
各帯域がドラムの異なる要素に反応
→ Kickで低域が引っ込む（通常のサイドチェイン）
→ Snareで中域が引っ込む（スネアの抜けが良くなる）
→ Hi-Hatで高域が引っ込む（ハイハットとの干渉回避）
→ 非常にタイトなミックス

Ableton Live 12での実装:
1. Multiband Dynamicsを使用
2. 各バンドにSidechain Inputを設定
3. 各バンドのAttack/Releaseを個別調整
4. Gain Reductionを確認しながら微調整
```

### リバースサイドチェイン

```
通常とは逆の動作:
音が鳴ると他のパラメータが増加する

設定例 1: Kickで Reverb増加

Envelope Follower:
Input: Kick
Destination: Reverb Dry/Wet（順方向）
Attack: 50 ms
Release: 500 ms
Gain: +15 dB

動作:
Kick鳴る → Reverb増える
Kick消える → Reverb減る
→ Kickの残響だけが強調される
→ 空間的なKick

設定例 2: Snareで Delay増加

Envelope Follower:
Input: Snare
Destination: Delay Feedback（順方向）
Attack: 10 ms
Release: 800 ms
Gain: +20 dB

動作:
Snare鳴る → Delay Feedback増加
→ Snareの残響がどんどんフィードバック
→ Dub Technoスタイルのエフェクト

設定例 3: ボーカルで Distortion増加

Envelope Follower:
Input: ボーカル
Destination: Distortion Drive
Attack: 20 ms
Release: 300 ms
Gain: +10 dB

動作:
歌声が強い → Distortion増加
歌声が弱い → Distortion減少
→ 感情的な表現に歪みが連動
→ ロック、メタル的ボーカル処理
```

### クロスモジュレーション（トラック間変調）

```
異なるトラック同士でパラメータを相互制御:

設定例: Bass ↔ Pad クロスモジュレーション

Bass → Pad:
Envelope Follower on Pad
Input: Bass
Destination: Pad Filter Cutoff
方向: 逆（Bass鳴る → Pad暗くなる）

Pad → Bass:
Envelope Follower on Bass
Input: Pad
Destination: Bass Resonance
方向: 順（Pad鳴る → Bass Resonance上がる）

動作:
Bassが鳴る → Padが引っ込む
Padが鳴る → Bassの音色が変わる
→ 2つのトラックが相互に影響
→ 有機的な一体感

注意点:
1. フィードバックループに注意
2. 変調深さは控えめに（10-20%）
3. Attack/Releaseを適切に設定
4. モニタリングしながら微調整
```

---

## マクロコントロール設計の極意

単純なマッピングを超えた、プロレベルのマクロコントロール設計手法を学びます。

### コンセプト・ベース・マクロ設計

```
パラメータベースではなく「概念」ベースで設計:

従来のアプローチ（NG）:
Macro 1: Filter Cutoff
Macro 2: Resonance
Macro 3: LFO Rate
Macro 4: Reverb
→ パラメータ名をそのまま = 演奏時に考えすぎる

コンセプト・ベース設計（推奨）:
Macro 1: "Energy"（エネルギー）
Macro 2: "Space"（空間）
Macro 3: "Texture"（テクスチャ）
Macro 4: "Chaos"（カオス）
→ 抽象的概念 = 直感的に操作可能

"Energy" マクロの設計:

0%（Low Energy）:
- Filter Cutoff: 400 Hz
- Resonance: 10%
- Drive: 0 dB
- Volume: -6 dB
- Attack: 50 ms
- LFO Depth: 5%

50%（Mid Energy）:
- Filter Cutoff: 1500 Hz
- Resonance: 30%
- Drive: 4 dB
- Volume: -3 dB
- Attack: 15 ms
- LFO Depth: 20%

100%（High Energy）:
- Filter Cutoff: 4000 Hz
- Resonance: 55%
- Drive: 12 dB
- Volume: 0 dB
- Attack: 0 ms
- LFO Depth: 45%

効果:
1つのノブで「エネルギー」という概念を制御
→ 7つのパラメータが連動
→ 演奏中は「エネルギーを上げる」だけ考えれば良い

"Space" マクロの設計:

0%（Dry / Close）:
- Reverb Dry/Wet: 5%
- Reverb Decay: 0.5s
- Delay Dry/Wet: 0%
- Stereo Width: 80%
- Pre-Delay: 5 ms
- High Damp: 8000 Hz

100%（Wet / Far）:
- Reverb Dry/Wet: 45%
- Reverb Decay: 4.0s
- Delay Dry/Wet: 20%
- Stereo Width: 150%
- Pre-Delay: 40 ms
- High Damp: 3000 Hz

効果:
「近い/遠い」を直感的に制御
→ ミックス中の空間配置が即座に変更可能
```

### 非線形マクロカーブ

```
Macroの動きを直線ではなく曲線にする:

問題:
Macroのリニア（直線）マッピング
→ 中間値での変化が知覚的に不均一

例:
Filter Cutoff 200Hz - 5000Hz をリニアにMap
→ 50%の位置 = 2600Hz
→ 低域の変化が鈍く、高域の変化が急激
→ 聴覚的に不自然

解決: 対数カーブ

Ableton Live 12での設定:
1. Macro → Map → Filter Cutoff
2. Mapping Editor を開く
3. Curve を Logarithmic に変更

結果:
Macro 25% = 600Hz（低域でも変化を感じる）
Macro 50% = 1200Hz（中域）
Macro 75% = 2800Hz（高域）
Macro 100% = 5000Hz

→ 聴覚的に均一な変化

他の推奨カーブ:

Resonance: S字カーブ
→ 低値と高値で変化が急、中間で緩やか
→ 共振のスイートスポットを探しやすい

Volume: リニア
→ dBスケールは元々対数
→ リニアで十分

LFO Depth: 指数カーブ
→ 小さい値での微調整が可能
→ 大きい値は急激に変化
```

### マクロ・スナップショット

```
Macroの複数設定を瞬時に切り替える:

概念:
あらかじめMacro値のプリセットを用意
→ ボタン一発で切り替え

設定例: 4つのシーン

Scene A: "Intro"（導入）
Macro 1 (Energy): 20%
Macro 2 (Space): 60%
Macro 3 (Texture): 10%
Macro 4 (Chaos): 5%
→ 静かで空間的

Scene B: "Build"（ビルドアップ）
Macro 1 (Energy): 60%
Macro 2 (Space): 30%
Macro 3 (Texture): 50%
Macro 4 (Chaos): 30%
→ エネルギー上昇中

Scene C: "Drop"（ドロップ）
Macro 1 (Energy): 100%
Macro 2 (Space): 10%
Macro 3 (Texture): 80%
Macro 4 (Chaos): 60%
→ 最大エネルギー

Scene D: "Breakdown"（ブレイクダウン）
Macro 1 (Energy): 10%
Macro 2 (Space): 80%
Macro 3 (Texture): 20%
Macro 4 (Chaos): 10%
→ 静寂と空間

実装方法（Ableton Live 12）:
1. Clip Envelope にMacro値を記録
2. Scene起動 → 全Macroが設定値に変化
3. Follow Action で自動遷移も可能

ライブでの運用:
パッド上段 = Scene A, B, C, D
パッド下段 = クリップ起動
ノブ = Macro微調整
→ 構成の大きな変化はパッド
→ 細かい変化はノブ
```

---

## Max for Liveモジュレーター活用

Ableton Live 12のMax for Liveデバイスを使った高度なモジュレーション手法を解説します。

### LFO（Max for Live版）

```
Max for Live LFO の利点:

標準LFOとの違い:
1. 任意のパラメータにマッピング可能
2. カスタム波形描画
3. より多くの波形選択
4. ステップシーケンサー内蔵
5. マルチアウト対応

設定手順:
1. Audio Effects → Max for Live → LFO
2. Map ボタンをクリック
3. 変調したいパラメータをクリック
4. Rate、Depth、Offset を調整

推奨設定例:

パッド用 Evolving Texture:
Wave: Custom（手描き）
Rate: 0.08 Hz（非常にゆっくり）
Depth: 35%
Destination: Wavetable Position
Jitter: 10%（ランダム揺れ追加）

→ Wavetable Positionがゆっくり変化
→ 音色が常に微妙に変わる
→ 飽きないパッドサウンド

ベース用 Rhythmic Filter:
Wave: Step Sequence
Steps: 16
Rate: 1/16（テンポ同期）
Depth: 45%
Destination: Auto Filter Cutoff

ステップ値:
Step 1: 100%
Step 2: 30%
Step 3: 70%
Step 4: 10%
Step 5: 90%
Step 6: 20%
Step 7: 80%
Step 8: 50%
Step 9: 100%
Step 10: 0%
Step 11: 60%
Step 12: 40%
Step 13: 85%
Step 14: 15%
Step 15: 95%
Step 16: 5%

→ 16ステップの複雑なフィルターパターン
→ パターンが繰り返すが複雑
→ Techno、Minimal に最適
```

### Envelope Follower（Max for Live版）

```
Max for Live Envelope Follower の利点:

1. ビジュアルフィードバック
2. より精密なAttack/Release制御
3. Gain / Map Amount 独立制御
4. Smoothing 機能
5. 複数Destinationへの同時出力

高度な設定:

Multi-Destination Envelope Follower:

Input: ドラムバス
Smoothing: 15%

Output 1:
Destination: Pad Volume（逆）
Amount: -40%
→ ドラムでPadをダッキング

Output 2:
Destination: Synth Filter Cutoff
Amount: +25%
→ ドラムでSynthのFilterが開く

Output 3:
Destination: Reverb Dry/Wet（逆）
Amount: -20%
→ ドラム時にReverbが減少

効果:
1つのEnvelope Followerで3つの異なる反応
→ ドラムがミックス全体をコントロール
→ 統一感のあるグルーヴ
```

### Shaper（Max for Live）

```
Shaperデバイスの活用:

機能:
カスタムカーブでMIDIやオーディオを変形
→ 入力値と出力値の関係をカスタマイズ

設定例 1: Velocity Shaper

Input: MIDI Velocity
Output: Filter Cutoff Amount

カーブ設定:
Velocity 0-60: ほぼ変化なし（フラット）
Velocity 60-100: 急激に上昇
Velocity 100-127: 緩やかに最大値へ

効果:
弱く弾く → ほとんど変化なし
中程度 → 急に明るくなる
強く弾く → 最大の明るさ
→ ドラマティックなベロシティ応答

設定例 2: LFO出力のShaping

Input: LFO 出力
Output: Filter Cutoff

カーブ設定:
LFO値 0-30%: 出力 0%（無変化ゾーン）
LFO値 30-70%: リニア変化
LFO値 70-100%: 出力 100%（飽和ゾーン）

効果:
LFOの中央部分のみが反映される
→ デッドゾーンと飽和ゾーンが存在
→ より「スイートスポット」に留まる変化
→ 音楽的に意味のある範囲での動き
```

---

## モジュレーションマトリクスの活用

複数のモジュレーションソースとデスティネーションを体系的に管理する方法を学びます。

### モジュレーションマトリクスとは

```
定義:
モジュレーションソース（LFO、Envelope等）と
デスティネーション（パラメータ）の接続を
格子状（マトリクス）で管理するシステム

利点:
1. 全モジュレーション接続を一覧で把握
2. 複雑なルーティングも視覚的に管理
3. 各接続のAmount（量）を個別設定
4. ソースとデスティネーションの追加/削除が容易

対応シンセサイザー:
- Serum（Xfer Records）: 標準装備
- Vital（Matt Tytel）: 標準装備
- Massive X（Native Instruments）: 標準装備
- Pigments（Arturia）: 標準装備
- Phase Plant（Kilohearts）: 標準装備
```

### Serumでのモジュレーションマトリクス

```
Serumのマトリクス構造:

ソース（16スロット）:
LFO 1-4
Envelope 1-4
Velocity
Aftertouch
Mod Wheel
Note
Noise
Macro 1-4

デスティネーション（任意）:
OSC A/B の全パラメータ
Filter の全パラメータ
FX の全パラメータ
Amplifier の全パラメータ
LFO Rate/Depth
他のマトリクススロット

実践パッチ例: "Morphing Pad"

マトリクス設定:
Slot 1: LFO 1 → OSC A Wavetable Position
   Amount: 45%
   Curve: Linear

Slot 2: LFO 2 → OSC B Wavetable Position
   Amount: 30%
   Curve: Linear

Slot 3: LFO 3 → Filter Cutoff
   Amount: 25%
   Curve: Bipolar

Slot 4: Envelope 2 → OSC A Level
   Amount: -15%
   Curve: Exponential

Slot 5: Mod Wheel → LFO 1 Rate
   Amount: 50%
   Curve: Linear

Slot 6: Velocity → Filter Resonance
   Amount: 20%
   Curve: Linear

Slot 7: Aftertouch → Reverb Mix
   Amount: 30%
   Curve: S-Curve

Slot 8: Macro 1 → LFO 3 Depth
   Amount: 80%
   Curve: Linear

LFO設定:
LFO 1: Sine, 0.15 Hz, Free Running
LFO 2: Triangle, 0.22 Hz, Free Running
LFO 3: Custom Shape, 1/8 Sync

結果:
8つのモジュレーション接続が同時動作
→ 常に変化し続ける複雑なパッドサウンド
→ Mod WheelでLFO速度を制御
→ Aftertouchで空間を制御
→ Macro 1でフィルター動きの深さを制御
```

### Vitalでのモジュレーションマトリクス

```
Vitalの特徴的なモジュレーション機能:

ドラッグ&ドロップ方式:
1. モジュレーションソースの上でマウスをドラッグ
2. デスティネーションパラメータにドロップ
3. Amount をドラッグで調整
→ 直感的な操作

ビジュアルフィードバック:
各パラメータの横にモジュレーション量が表示
→ リアルタイムで変調の動きが見える

Random LFO:
Vital固有の機能
ランダムな値を滑らかに補間
→ S&Hよりも滑らかなランダム変化
→ パーリンノイズ的な動き

実践例: "Glitch Bass"

モジュレーション設定:

LFO 1（テンポ同期）:
Shape: Random
Rate: 1/16
Smooth: 20%
→ OSC 1 Wavetable Frame: 60%
→ Filter 1 Cutoff: 40%

LFO 2（フリーラン）:
Shape: Sine
Rate: 0.3 Hz
→ OSC 1 Unison Detune: 25%
→ Filter 1 Resonance: 15%

Envelope 2:
A: 0ms, D: 150ms, S: 30%, R: 50ms
→ Filter 1 Cutoff: 50%
→ OSC 1 Level: -10%

Mod Wheel:
→ LFO 1 Depth: 80%
→ Filter 1 Drive: 30%

効果:
ランダムなWavetable変化 + フィルター動き
→ グリッチ感のある動的なベース
→ Mod Wheelで「グリッチ度」を制御
```

---

## ジャンル別モジュレーション戦略

各ジャンルに最適なモジュレーション手法を具体的に解説します。

### Techno/Minimal

```
Technoのモジュレーション哲学:
「繰り返しの中の微妙な変化」

重要なモジュレーション:

1. フィルタースイープ（最重要）:
   LFO → Filter Cutoff
   Rate: 1/4 - 4bars（長い周期）
   Depth: 20-60%
   波形: Triangle or Custom
   → 長い時間をかけてフィルターが開閉
   → Technoの「うねり」を生む

2. Resonance揺れ:
   LFO → Resonance
   Rate: 別のLFOとは異なるRate
   Depth: 10-25%
   → フィルタースイープとずれて動く
   → 予測不可能な酸っぱさ

3. ディレイフィードバック変調:
   Macro → Delay Feedback
   範囲: 15-65%
   ライブ中に手動操作
   → Dub Technoの空間演出

4. グレインポジション変調:
   LFO → Granular Position
   Rate: 非常にゆっくり（0.05 Hz）
   Depth: 100%
   → テクスチャが常に変化
   → Ambient Techno に必須

Techno Bass モジュレーション設定:
LFO 1: Triangle, 2bars, Filter Cutoff 40%
LFO 2: Sine, 0.07Hz, Resonance 15%
Macro 1: Filter Sweep（200Hz-3000Hz）
Macro 2: Delay Feedback（10%-55%）

Techno Pad モジュレーション設定:
LFO 1: Sine, 0.1Hz, Wavetable Position 50%
LFO 2: Triangle, 0.15Hz, Pan 40%
LFO 3: Custom, 4bars, Filter Cutoff 30%
Envelope Follower: Kick → Volume（逆）-15dB
```

### House/Deep House

```
Houseのモジュレーション哲学:
「グルーヴとノリを生むモジュレーション」

重要なモジュレーション:

1. サイドチェインポンピング:
   Envelope Follower: Kick → Bass Volume
   Attack: 3ms
   Release: 120-200ms
   Amount: -10dB to -20dB
   → 4つ打ちに合わせたポンピング
   → Houseの基本中の基本

2. ベースラインのFilter動き:
   LFO → Filter Cutoff
   Rate: 1/8 or 1/16
   Depth: 25-40%
   波形: Triangle
   → ファンキーなベースライン
   → Disco House に必須

3. コード・スタブのResonance:
   LFO → Resonance
   Rate: 1/4
   Depth: 15%
   → コードにアクセント
   → リズミックな音色変化

4. パッドのステレオ動き:
   LFO → Pan
   Rate: 1/2 or 1bar
   Depth: 30-50%
   → 広がりのあるパッド
   → Deep House の空間感

House Organ モジュレーション:
LFO 1: Sine, 5.5Hz, Pitch 3%（ビブラート）
LFO 2: Triangle, 1/8, Filter 20%
Velocity → Volume + Filter Cutoff
→ クラシックなHouseオルガンサウンド
```

### Drum & Bass / Neurofunk

```
DnBのモジュレーション哲学:
「高速かつアグレッシブな変調」

重要なモジュレーション:

1. ワブルベース（Wobble Bass）:
   LFO → Filter Cutoff
   Rate: 1/4 - 1/2（170BPMでの体感速度）
   Depth: 60-80%
   波形: Sine or Triangle
   → ワブワブするベース
   → DnB の象徴的サウンド

2. リーセ（Reese Bass）変調:
   LFO → Detune Amount
   Rate: 0.2-0.5 Hz
   Depth: 20-40%
   + LFO → Filter Cutoff
   Rate: 0.15 Hz
   Depth: 30%
   → うなりのある太いベース
   → Neurofunkの基本

3. FM変調ベース:
   LFO → FM Amount
   Rate: 1/8
   Depth: 50%
   → 金属的な音色変化
   → Neurofunkの攻撃的サウンド

4. グレインベース:
   LFO → Grain Position + Grain Size
   Rate: Random
   Depth: 60%
   → 壊れたようなテクスチャ
   → 実験的DnB

Neuro Bass パッチ:
LFO 1: Sine, 1/4, Filter Cutoff 70%
LFO 2: Random, 1/8, FM Amount 40%
LFO 3: Triangle, 0.3Hz, Detune 25%
Macro 1: Wobble Rate（1/8 - 1/1）
Macro 2: Aggression（Drive + FM + Resonance）
```

### Ambient/Experimental

```
Ambientのモジュレーション哲学:
「ゆっくり、繊細、有機的な変化」

重要なモジュレーション:

1. 超低速LFO:
   LFO → Wavetable Position
   Rate: 0.02-0.1 Hz（10秒-50秒周期）
   Depth: 40-80%
   → 音色が非常にゆっくり変化
   → 聴いていて飽きない

2. ランダムLFO（Perlin Noise的）:
   LFO → Multiple Destinations
   Wave: Smooth Random
   Rate: 0.05 Hz
   Depth: 15-30%
   → 予測不可能だが滑らかな変化
   → 自然界の音に近い

3. グラニュラー変調:
   LFO → Grain Position
   Rate: 0.03 Hz
   Depth: 100%
   + LFO → Grain Size
   Rate: 0.07 Hz
   Depth: 40%
   → テクスチャが常に変化
   → Brian Eno的サウンドスケープ

4. リバーブ変調:
   LFO → Reverb Decay
   Rate: 0.01 Hz（100秒周期）
   Depth: 30%
   + LFO → Reverb Damping
   Rate: 0.015 Hz
   Depth: 25%
   → 空間自体が変化する
   → 没入感

Ambient Drone パッチ:
LFO 1: Sine, 0.03Hz, WT Position 60%
LFO 2: Sine, 0.02Hz, Filter Cutoff 20%
LFO 3: Smooth Random, 0.05Hz, Pan 40%
LFO 4: Triangle, 0.01Hz, Reverb Decay 25%
→ 4つのLFOがすべて異なる超低速
→ 数分間聴いても同じ瞬間がない
```

---

## 実践パッチ演習（上級編）

ここまで学んだ技術を組み合わせた、上級者向けの実践パッチ演習を行います。

### 演習1: セルフ・エボルビング・パッド（自己進化パッド）

```
目標: 何もしなくても常に変化し続けるパッド

所要時間: 45分

Step 1: オシレーター設定

OSC A:
Wavetable: Complex Shapes
Position: 30%（初期値）
Unison: 4 voices, Detune 15%

OSC B:
Wavetable: Vocal/Formant
Position: 50%（初期値）
Unison: 2 voices, Detune 8%
Volume: -6dB

Step 2: フィルター設定

Filter 1: Low Pass 12dB
Cutoff: 1800Hz
Resonance: 20%
Key Tracking: 50%

Filter 2: Band Pass 12dB
Cutoff: 3500Hz
Resonance: 35%
Key Tracking: 30%
→ Filter 2 は OSC B のみに適用

Step 3: LFO設定（5つのLFO）

LFO 1:
Destination: OSC A Wavetable Position
Waveform: Sine
Rate: 0.08 Hz（12.5秒周期）
Depth: 45%
→ メインの音色変化

LFO 2:
Destination: OSC B Wavetable Position
Waveform: Triangle
Rate: 0.12 Hz（8.3秒周期）
Depth: 35%
→ サブの音色変化（異なる速度）

LFO 3:
Destination: Filter 1 Cutoff
Waveform: Custom（S字カーブ）
Rate: 0.05 Hz（20秒周期）
Depth: 25%
→ 非常にゆっくりなフィルター変化

LFO 4:
Destination: Pan
Waveform: Sine
Rate: 0.1 Hz（10秒周期）
Depth: 40%
Phase: 0度（Left）と180度（Right）
→ ステレオフィールドで揺れる

LFO 5:
Destination: OSC A/B Mix Balance
Waveform: Smooth Random
Rate: 0.03 Hz（33秒周期）
Depth: 20%
→ 2つのOSCのバランスがランダムに変化

Step 4: エフェクト設定

1. Chorus:
   Rate: 0.3 Hz
   Depth: 40%
   Mix: 30%

2. Reverb:
   Decay: 5.0s
   Damping: 60%
   Mix: 35%
   + LFO → Decay: Rate 0.02Hz, Depth 15%

3. Delay:
   Time: 3/8（付点4分）
   Feedback: 25%
   Mix: 15%
   Filter: High Pass 200Hz, Low Pass 3000Hz

4. Utility:
   Width: 130%

Step 5: Macro設定

Macro 1 "Brightness":
→ Filter 1 Cutoff: 800-4000Hz
→ Filter 2 Cutoff: 2000-6000Hz
→ High Shelf: 0 - +4dB

Macro 2 "Depth":
→ Reverb Mix: 15-55%
→ Delay Mix: 5-30%
→ Chorus Depth: 20-60%

Macro 3 "Evolution Speed":
→ LFO 1 Rate: 0.02-0.2Hz
→ LFO 2 Rate: 0.03-0.3Hz
→ LFO 3 Rate: 0.01-0.15Hz
→ 全LFOの速度を一括制御

Macro 4 "Complexity":
→ LFO 5 Depth: 0-50%
→ OSC B Volume: -12dB - 0dB
→ Filter 2 Resonance: 10-50%

完成基準:
□ 5つのLFOが独立して動作
□ 音色が常に変化している
□ 1分間聴いても同じ瞬間がない
□ Macroで全体的なキャラクターを制御可能
□ 演奏なしでも音楽的に成立
```

### 演習2: インタラクティブ・ベースライン

```
目標: 演奏入力に反応して音色が動的に変化するベース

所要時間: 60分

Step 1: 基本サウンド

OSC 1: Saw Wave
OSC 2: Square Wave, -1 Octave, Volume 70%
Sub OSC: Sine, -2 Octave, Volume 40%

Filter: Low Pass 24dB
Cutoff: 800Hz
Resonance: 45%
Drive: 4dB

Step 2: Velocity反応設計

Velocity → Filter Cutoff:
弱（0-60）: 600-1000Hz
中（60-100）: 1000-2000Hz
強（100-127）: 2000-3500Hz
カーブ: 指数関数

Velocity → Filter Envelope Amount:
弱: +10（控えめなアタック）
強: +55（鋭いアタック）

Velocity → Distortion Drive:
弱: 0dB（クリーン）
強: +8dB（歪む）

Velocity → OSC 2 Volume:
弱: 40%（Squareが控えめ）
強: 90%（Square強調）

効果:
弱く弾く → 丸い、柔らかいベース
強く弾く → 鋭い、攻撃的なベース
→ 演奏のダイナミクスが直接音色に反映

Step 3: Mod Wheel設計

Mod Wheel → LFO Rate:
0%: 1/4（ゆっくりワウワウ）
100%: 1/32（高速ワウワウ）

Mod Wheel → LFO Depth:
0%: 10%（控えめ）
100%: 60%（深い）

Mod Wheel → Resonance:
0%: 30%
100%: 70%

効果:
Mod Wheel上げる → 速く深いワウワウ + 高Resonance
→ アシッドベース的サウンド
→ TB-303エミュレーション

Step 4: ノート位置による変化

Note（音程）→ Filter Cutoff:
Key Tracking: 80%
低い音 → Cutoff低い（暗い）
高い音 → Cutoff高い（明るい）

Note → Unison Detune:
低い音: Detune 5%（タイトに）
高い音: Detune 20%（広がる）

Note → Reverb Mix:
低い音: 5%（ドライに）
高い音: 20%（空間的に）

効果:
低音域はタイトで明瞭
高音域は広がりのある音
→ 音域全体で自然なバランス

Step 5: 演奏パターンとMacro操作

16小節演奏シナリオ:

Bar 1-4: 基本パターン
A1を繰り返し、Velocity 80-100
Mod Wheel: 0%
→ シンプルで安定

Bar 5-8: エネルギー上昇
A1-C2-E2-G2、Velocity上昇（90-127）
Mod Wheel: 0% → 40%
→ 徐々にアシッド感

Bar 9-12: ピーク
C2-E2-G2-C3、Velocity 110-127
Mod Wheel: 60-80%
→ アグレッシブ、アシッド全開

Bar 13-16: クールダウン
G1-A1、Velocity 70-90
Mod Wheel: 80% → 0%
→ 徐々に落ち着く

完成基準:
□ Velocityで音色が明確に変化
□ Mod Wheelでワウワウ深さが変化
□ ノート位置で自然な音色変化
□ 16小節の演奏がストーリー性を持つ
□ ライブ演奏で即興可能
```

### 演習3: モジュレーション・オーケストレーション

```
目標: 複数トラックのモジュレーションを連携させた
     統一感のあるアレンジメント

所要時間: 90分

コンセプト:
4つのトラックが1つのマスターマクロで連動

トラック構成:
Track 1: Bass
Track 2: Pad
Track 3: Lead
Track 4: FX/Texture

Master Macro設定（Macro Variations装置）:

Master Macro 1: "Intensity"（強度）
各トラックへの影響:

Bass:
→ Filter Cutoff: 400Hz → 2500Hz
→ Drive: 0dB → 10dB
→ LFO Depth: 10% → 50%

Pad:
→ Volume: -8dB → 0dB
→ Reverb Mix: 40% → 15%
→ Filter Cutoff: 1000Hz → 5000Hz

Lead:
→ LFO Vibrato Depth: 0% → 8%
→ Delay Mix: 5% → 20%
→ Unison Voices: 2 → 6

FX/Texture:
→ Volume: -20dB → -3dB
→ Grain Density: 20% → 80%
→ Distortion: 0% → 40%

Master Macro 2: "Space"（空間）
各トラックへの影響:

Bass:
→ Reverb Mix: 0% → 15%
→ Stereo Width: Mono → 30%

Pad:
→ Reverb Decay: 1.5s → 6.0s
→ Stereo Width: 100% → 160%
→ Pre-Delay: 10ms → 60ms

Lead:
→ Delay Feedback: 10% → 45%
→ Reverb Mix: 15% → 40%

FX/Texture:
→ Reverb Mix: 30% → 70%
→ Delay Time: 1/8 → 1/2

実装手順:

1. 4トラック分のインストゥルメントを作成
2. 各トラックにRack（グループ）を構築
3. Rack内のMacroに個別パラメータをMap
4. Master Track に MIDI CC → 各トラックMacroを接続
5. MIDIコントローラーの2つのノブに割り当て

ライブ演奏シナリオ:

Intro（16bars）:
Intensity: 10%, Space: 60%
→ 静かで広い空間

Build（16bars）:
Intensity: 10% → 70%, Space: 60% → 30%
→ 徐々にエネルギー上昇、空間引き締め

Drop（16bars）:
Intensity: 100%, Space: 10%
→ 最大エネルギー、タイト

Breakdown（16bars）:
Intensity: 100% → 20%, Space: 10% → 80%
→ エネルギー低下、空間拡大

完成基準:
□ 4トラックが Master Macro に連動
□ 2つのノブで楽曲全体のキャラクターが変化
□ Intro → Build → Drop → Breakdown が演奏可能
□ 各トラックの変化が音楽的に自然
□ ライブパフォーマンスとして成立
```

---

## トラブルシューティング（上級編）

### モジュレーションが音楽的でない場合

```
問題: モジュレーションが機械的、不自然

原因と解決策:

原因1: LFO Rate が単純すぎる
解決: 非整数比のRateを使用
悪い例: LFO 1 = 1/4, LFO 2 = 1/8（整数比 2:1）
良い例: LFO 1 = 0.33Hz, LFO 2 = 0.21Hz（非整数比）
→ パターンが繰り返しにくくなる

原因2: Depth が深すぎる
解決: 控えめなDepthから始める
悪い例: LFO → Filter Cutoff 80%
良い例: LFO → Filter Cutoff 15-25%
→ 聴き取れるが押し付けがましくない

原因3: 波形が角ばりすぎ
解決: SineまたはTriangleから始める
悪い例: Square Wave → Filter Cutoff
良い例: Sine Wave → Filter Cutoff
→ 滑らかな変化の方が音楽的

原因4: 位相が揃いすぎ
解決: 各LFOの開始位相をランダムに
Retrigger: Off にする
→ ノートごとに異なる変化
→ 毎回新鮮なサウンド
```

### CPU負荷が高い場合

```
問題: モジュレーション追加でCPU負荷増大

対策:

1. LFO数の最適化:
   不要なLFOを削除
   似た動きのLFOは統合
   → 3-5個のLFOで十分

2. Envelope Followerの軽量化:
   Smoothing を上げる（CPU軽減）
   不要な周波数帯のフィルタリング
   → 処理負荷低減

3. Max for Liveの代替:
   可能ならネイティブLFOを使用
   Max for Live は CPU負荷が高め
   → 必要な場面のみ使用

4. フリーズ＆フラッテン:
   モジュレーション完成後
   トラックをFreeze → Flatten
   → CPU解放、オーディオ化

5. バッファサイズ:
   制作中: 512-1024 samples
   ミックス中: 1024-2048 samples
   → レイテンシーとのトレードオフ
```

### モジュレーション間の干渉

```
問題: 複数モジュレーションが互いに打ち消し合う

診断方法:

1. 1つずつモジュレーションを有効にする
2. 各モジュレーション単独の効果を確認
3. 2つずつ組み合わせて干渉チェック
4. 問題のある組み合わせを特定

よくある干渉パターン:

パターン1: 同じDestinationへの二重変調
LFO 1 → Filter Cutoff +30%
LFO 2 → Filter Cutoff -25%
→ 打ち消し合い、効果が弱い
解決: 1つに統合するか、範囲を分ける

パターン2: 逆方向の変調
LFO → Filter Cutoff（上昇）
Envelope → Filter Cutoff（下降）
→ 矛盾する動き
解決: 意図的な場合はOK、そうでなければ方向を揃える

パターン3: レート干渉
LFO 1: Rate 1/4
LFO 2: Rate 1/8（倍の関係）
→ 4拍ごとにLFOが揃い、予測可能な繰り返し
解決: 非整数比にする（例: 1/4 と 3/16）

確認手順:
□ 各モジュレーションの効果を単独で確認
□ Destination重複がないか確認
□ LFO Rateの比率を確認
□ 意図しない打ち消しがないか確認
□ 全体を聴いて音楽的かどうか判断
```

---

## 上級者向けリファレンス

### モジュレーション設計チェックリスト

```
パッチ作成時のチェックリスト:

基本設計:
□ モジュレーションの目的が明確か
□ 必要最小限のモジュレーション数か
□ 各モジュレーションの役割が重複していないか

LFO設計:
□ Rate は音楽的か（テンポ同期 or 意図的なフリーラン）
□ Depth は控えめから始めているか
□ 波形は用途に合っているか
□ 複数LFOのRate比は非整数比か
□ Retrigger設定は適切か

Macro設計:
□ コンセプト・ベースの命名になっているか
□ 範囲設定は実用的か
□ カーブは聴覚的に自然か
□ ライブ演奏で直感的に操作可能か

MIDI CC設計:
□ Mod Wheel に最重要パラメータを割り当てたか
□ Velocity感度は適切か
□ Aftertouch対応なら活用しているか
□ 物理的操作性を考慮しているか

最終チェック:
□ 音楽的に自然か
□ CPU負荷は許容範囲内か
□ 他のトラックとの干渉はないか
□ ライブ演奏で破綻しないか
□ 30秒間聴いて飽きないか
```

### モジュレーションのベストプラクティス集

```
プロから学ぶ10のルール:

1. 「Less is More（少ないほど良い）」
   モジュレーション数は最小限に
   各モジュレーションの効果を最大化

2. 「意図を持つ」
   なぜこのモジュレーションが必要か
   目的なき変調は雑音と同じ

3. 「耳で判断する」
   数値ではなく音で判断
   15%でも聴こえなければ意味がない

4. 「レイヤーで考える」
   速い変化（1/16）+ 遅い変化（4bars）
   マイクロ変化 + マクロ変化の二層構造

5. 「演奏性を最優先」
   どんなに複雑でも1-2ノブで制御可能に
   ライブで使えないモジュレーションは無価値

6. 「非整数比を意識する」
   LFO Rate同士を倍数にしない
   予測不可能性 = 有機的

7. 「コンテキストで考える」
   ソロで良くてもミックスで機能するか
   他のトラックとの関係を意識

8. 「保存と再利用」
   良いモジュレーション設定はプリセット保存
   テンプレート化で効率向上

9. 「参照曲を分析する」
   好きな曲のモジュレーションを聴き取る
   LFO Rate、Depth、Destinationを推測

10. 「実験を恐れない」
    意図しない接続が最高の結果を生むことも
    Undo があるのだから自由に試す
```

---

**モジュレーションで、動きと表現力のある音を作りましょう！**
