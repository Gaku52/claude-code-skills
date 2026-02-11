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

**モジュレーションで、動きと表現力のある音を作りましょう！**
