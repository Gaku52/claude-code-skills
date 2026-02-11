# Synthesis Basics（シンセシス基礎）

シンセサイザーで音を作る基本原理を完全理解します。Oscillator、Filter、Envelope、LFOの4つの要素をマスターすれば、70%の音は作れるようになります。

## この章で学ぶこと

- シンセシスの基本原理（減算式合成）
- Oscillator（オシレーター）と波形の種類
- Filter（フィルター）と音色の削り方
- Envelope（エンベロープ）と時間的変化
- LFO（エルエフオー）と周期的変化
- 4つの要素を組み合わせた音作り
- Wavetableでの実践
- 基本パッチ10個の作成

---

## なぜSynthesis Basicsが重要なのか

**理解度の差:**

```
理解なし:
行動:
プリセットをランダムに選ぶ
パラメータの意味が分からない
調整できない

結果:
運任せ
時間浪費（2時間）
完成しない

理解あり:
行動:
目的の音をイメージ
必要なパラメータを調整
10分で完成

結果:
確実に作れる
時間効率
クリエイティブ

プロとアマの差:
アマ: パラメータ理解度 20%
プロ: パラメータ理解度 90%
```

**シンセシス理解の重要性:**

- **効率**: 10分で目的の音を作れる
- **創造性**: 無限の可能性を理解できる
- **応用**: どのシンセでも通用する知識
- **修正**: 問題を即座に解決できる

---

## シンセサイザーの基本構造

### 信号の流れ（Signal Flow）

```
1. Oscillator（音の源）
   ↓
   音を生成
   ↓
2. Filter（音色を削る）
   ↓
   倍音を削って音色を作る
   ↓
3. Amplifier（音量制御）
   ↓
   音量を調整
   ↓
4. Output（出力）

モジュレーション:
- Envelope → Filter、Amplifier
- LFO → Oscillator、Filter
```

### 減算式合成（Subtractive Synthesis）

**基本原理:**

```
原理:
倍音豊かな波形（Saw、Square）
   ↓
Filterで削る
   ↓
目的の音色

例:
Saw波（倍音多い）
   ↓
Low Pass Filter 500 Hz
   ↓
丸い音色
   ↓
ベース音

なぜ "減算式" か:
倍音を減らす（削る）ことで音を作る
→ 最も一般的なシンセシス方式
→ Wavetable、Analogはこの方式
```

**他のシンセシス方式（参考）:**

```
FM合成（Frequency Modulation）:
倍音を足す
→ Operator

加算合成（Additive Synthesis）:
Sine波を重ねる
→ 複雑

ウェーブテーブル合成:
波形を切り替える
→ Wavetable

サンプリング:
録音した音を加工
→ Sampler
```

---

## 1. Oscillator（オシレーター）

**定義**: 音の源。波形を生成する。

### 波形の種類と特徴

**1. Sine波（サイン波）**

```
形状:
     /\
    /  \
   /    \
  /      \___

特徴:
- 最も単純な波形
- 倍音なし（基音のみ）
- 柔らかい音
- 丸い音

用途:
- Sub Bass（50-100 Hz）
- ベル音（FM合成）
- テスト信号

倍音構造:
基音（1倍音）: 100%
2倍音: 0%
3倍音: 0%
→ 倍音なし
```

**2. Saw波（ノコギリ波）**

```
形状:
 /|  /|  /|
/ | / | / |
  |/  |/  |

特徴:
- 全倍音を含む
- 明るい音
- 鋭い音
- 倍音最多

用途:
- ベース（Techno）
- リード（全ジャンル）
- ブラス
- パッド

倍音構造:
基音: 100%
2倍音: 50%
3倍音: 33%
4倍音: 25%
5倍音: 20%
...
→ 全倍音を含む（最も明るい）
```

**3. Square波（矩形波）**

```
形状:
 ___   ___
|   | |   |
|   |_|   |_

特徴:
- 奇数倍音のみ
- Sawより柔らかい
- 中空的な音

用途:
- リード（8bit風）
- ベース（Deep House）
- パッド
- チップチューン

倍音構造:
基音: 100%
2倍音: 0%
3倍音: 33%
4倍音: 0%
5倍音: 20%
...
→ 奇数倍音のみ
```

**4. Triangle波（三角波）**

```
形状:
   /\
  /  \
 /    \
/      \

特徴:
- 奇数倍音のみ（Squareより少ない）
- 非常に柔らかい
- Sineに近い

用途:
- Sub Bass（Sineの代替）
- パッド
- ビンテージ音

倍音構造:
基音: 100%
2倍音: 0%
3倍音: 11%（Squareの1/3）
5倍音: 4%
→ 奇数倍音のみ（非常に少ない）
```

### 波形の選び方

```
目的の音 → 推奨波形

Sub Bass:
- Sine 80%
- Triangle 20%

Techno Bass:
- Saw 90%
- Square 10%

Deep House Bass:
- Square 60%
- Triangle 40%

リード:
- Saw 70%
- Square 30%

パッド:
- Triangle 40%
- Square 30%
- Saw 30%
```

### Oscillatorパラメータ

**1. Octave（オクターブ）**

```
定義: 音の高さ（1オクターブ = 周波数2倍）

範囲:
-3 octave（超低音）
-2 octave（低音）
-1 octave（低め）
0 octave（標準）
+1 octave（高め）
+2 octave（高音）
+3 octave（超高音）

用途:
Sub Bass: -1 octave
ベース: 0 octave
リード: 0 octave
パッド: 0 octave
ハイハット: +3 octave
```

**2. Detune（デチューン）**

```
定義: 微妙なピッチのズレ（±50 cent = ±半音）

範囲:
0 cent（ズレなし）
±5 cent（わずかに太い）
±10 cent（コーラス効果）
±20 cent（ハモリ感）
±50 cent（半音）

用途:
2つのOSCをDetune:
OSC 1: Detune 0
OSC 2: Detune +7 cent
→ 音が太くなる、コーラス効果

理由:
微妙にズレると波が干渉
→ 音が揺れる（ビート）
→ アナログ感、温かみ
```

**3. Volume（音量）**

```
定義: Oscillatorの音量バランス

使用例:
OSC 1（Saw）: Volume 100%（メイン）
OSC 2（Square）: Volume 50%（ブレンド）
Sub（Sine）: Volume 30%（低域補強）

バランス:
Main OSC: 100%
2nd OSC: 50-80%
Sub: 20-40%
```

---

## 2. Filter（フィルター）

**定義**: 特定の周波数を削る（減衰させる）

### Filterの種類

**1. Low Pass Filter（ローパスフィルター）**

```
動作:
Cutoff周波数より高い周波数を削る
→ 低い周波数だけ通す（Pass）

図:
|
|    ←音量
|___
|    \
|     \___
|__________|_________
    ↑      ←周波数
  Cutoff

用途:
- ベース（200-500 Hz以下を通す）
- パッド（1500 Hz以下を通す）
- 温かい音色

最も一般的なFilter（使用頻度80%）
```

**2. High Pass Filter（ハイパスフィルター）**

```
動作:
Cutoff周波数より低い周波数を削る
→ 高い周波数だけ通す

図:
|           ___
|          /
|        _/
|       /
|______/
|___|________
    ↑
  Cutoff

用途:
- 低域削除（30-50 Hz以下を削る）
- リード（200 Hz以下を削る）
- 音の整理

ミキシングで必須（使用頻度70%）
```

**3. Band Pass Filter（バンドパスフィルター）**

```
動作:
Cutoff周波数の周辺だけ通す
→ 帯域を通す

図:
|
|      /\
|     /  \
|    /    \
|___/      \___
|___|_____|___
    ↓    ↑
   低   高

用途:
- 電話音（300-3000 Hz）
- ラジオ音
- エフェクト

特殊用途（使用頻度10%）
```

**4. Notch Filter（ノッチフィルター）**

```
動作:
Cutoff周波数だけ削る
→ バンドパスの逆

用途:
- 特定周波数の削除
- ハウリング除去

特殊用途（使用頻度5%）
```

### Filterパラメータ

**1. Cutoff（カットオフ周波数）**

```
定義: どこから削るか

範囲: 20 Hz - 20000 Hz

設定例:
Sub Bass:
Cutoff 150 Hz
→ 150 Hz以上を削る
→ 深い低音のみ

Techno Bass:
Cutoff 500 Hz（開始）
Cutoff 2000 Hz（ピーク）
→ オートメーションで動かす

リード:
Cutoff 3000 Hz
→ 3000 Hz以上を削る
→ 柔らかめのリード

パッド:
Cutoff 1500 Hz
→ 非常に柔らかい

最も重要なパラメータ（サウンドの80%を決める）
```

**2. Resonance（レゾナンス）**

```
定義: Cutoff周波数を強調

範囲: 0-100%

動作:
Cutoff周波数を山型に強調
→ ピーキング

図:
|
|      /\  ←Resonance 70%
|     /  \
|    /    \___
|___/
|___|________
    ↑
  Cutoff

効果:
0%: 強調なし（滑らか）
20%: わずかに強調
50%: 中程度の強調（「ピコピコ」）
70%: 強い強調（TB-303風）
90%+: 自己発振（ホイッスル音）

用途:
Acid Bass: Resonance 60-80%
クリーンベース: Resonance 0-20%
リード: Resonance 10-30%
```

**3. Slope（スロープ、dB/oct）**

```
定義: 削る急峻さ

種類:
12 dB/oct: 緩やか（柔らかい）
24 dB/oct: 急峻（鋭い）
48 dB/oct: 非常に急峻

選び方:
ベース: 24 dB/oct（低域をしっかり削る）
リード: 12 dB/oct（柔らかく削る）
パッド: 12 dB/oct

Wavetableでは:
デフォルト 24 dB/oct（十分）
```

---

## 3. Envelope（エンベロープ）

**定義**: 時間的な変化を制御する

### ADSR Envelope

```
構成要素:
A = Attack（立ち上がり時間）
D = Decay（減衰時間）
S = Sustain（持続レベル）
R = Release（余韻時間）

図:
    ___Peak
   /|   \
  / |    \_____Sustain
 /  |          \
/   |           \___
A  D     S      R

タイムライン:
1. 鍵盤を押す
2. Attack: 0 → Peak（100%）
3. Decay: Peak → Sustain レベル
4. Sustain: 鍵盤を押している間
5. 鍵盤を離す
6. Release: Sustain → 0
```

### パラメータ詳細

**1. Attack（アタック）**

```
定義: 鍵盤を押してから最大音量に達するまでの時間

範囲: 0 ms - 5000 ms

設定例:
0 ms: 即座に最大音量（ベース、ドラム）
10 ms: わずかに柔らかい（リード）
100 ms: ゆっくり立ち上がる（パッド）
500 ms: 非常にゆっくり（Ambient）
2000 ms: 超ゆっくり（Drone）

用途別:
Sub Bass: 0 ms（即座に）
Techno Bass: 0-5 ms
リード: 10-50 ms
パッド: 500-2000 ms
```

**2. Decay（ディケイ）**

```
定義: Peak から Sustain レベルまで減衰する時間

範囲: 0 ms - 5000 ms

設定例:
0 ms: 即座にSustainレベル
100 ms: 短い減衰
500 ms: 中程度の減衰
2000 ms: 長い減衰

用途別:
Pluck音（ピアノ、ギター）: Decay 200-500 ms
パッド: Decay 1000 ms
持続音（オルガン）: Decay 0 ms（すぐSustain）
```

**3. Sustain（サステイン）**

```
定義: 鍵盤を押している間の音量レベル

範囲: 0% - 100%

設定例:
100%: 鍵盤を押している間ずっと最大音量（オルガン）
70%: わずかに減衰（リード）
50%: 中程度の減衰
0%: すぐに消える（Pluck、ドラム）

用途別:
オルガン、パッド: 100%
リード: 70-90%
ベース: 80-100%
Pluck、ドラム: 0%
```

**4. Release（リリース）**

```
定義: 鍵盤を離してから音が消えるまでの時間

範囲: 0 ms - 10000 ms

設定例:
10 ms: 即座に消える（パーカッシブ）
100 ms: 短い余韻（ベース）
500 ms: 中程度の余韻（リード）
2000 ms: 長い余韻（パッド）
5000 ms: 超長い余韻（Ambient）

用途別:
ベース: 10-100 ms（短い）
リード: 200-500 ms
パッド: 2000-5000 ms（長い）
```

### Envelopeの種類

**1. Amp Envelope（音量エンベロープ）**

```
対象: 音量

必須: Yes（全ての音に必須）

設定例:
Sub Bass:
A: 0 ms
D: 50 ms
S: 100%
R: 10 ms

パッド:
A: 800 ms
D: 1000 ms
S: 80%
R: 3000 ms
```

**2. Filter Envelope（フィルターエンベロープ）**

```
対象: Filter Cutoff

効果: 音色の時間的変化

設定例:
Pluck音:
A: 0 ms
D: 300 ms
S: 0%
R: 100 ms
Envelope Amount: +50

動作:
鍵盤を押す → Cutoff 開く（明るい）
300 ms後 → Cutoff 閉じる（暗い）
→ ギター、ピアノのような減衰
```

**3. Pitch Envelope（ピッチエンベロープ）**

```
対象: Pitch

効果: ピッチ変化

設定例:
キック:
A: 0 ms
D: 50 ms
S: 0%
R: 0 ms
Envelope Amount: +12 semitones

動作:
鍵盤を押す → Pitch +12 semitones（1オクターブ上）
50 ms後 → Pitch 0（元に戻る）
→ ドゥーン（キック音）
```

---

## 4. LFO（Low Frequency Oscillator）

**定義**: 周期的な変化を作るオシレーター

### LFOの基本

```
特徴:
- 非常に低い周波数（0.1-20 Hz）
- 音として聞こえない
- パラメータを周期的に変化させる

用途:
- ビブラート（Pitch変化）
- トレモロ（Volume変化）
- ワウワウ（Filter変化）
- オートパン（Pan変化）
```

### LFOパラメータ

**1. Rate（レート、速度）**

```
定義: LFOの周期の速さ

範囲: 0.01 Hz - 40 Hz

設定例:
0.2 Hz: 非常にゆっくり（5秒/周期）
1 Hz: ゆっくり（1秒/周期）
5 Hz: 中速（ビブラート）
10 Hz: 速い（トレモロ）
20 Hz: 非常に速い

Sync to Tempo:
1/4: 4分音符ごと
1/8: 8分音符ごと
1/16: 16分音符ごと
→ テンポに同期（重要）
```

**2. Depth（デプス、深さ）**

```
定義: 変化の大きさ

範囲: 0% - 100%

設定例:
0%: 変化なし
5%: わずかな変化（ビブラート）
20%: 中程度の変化
50%: 大きな変化
100%: 最大変化

バランス:
ビブラート: Depth 5-10%
ワウワウ: Depth 30-60%
トレモロ: Depth 20-40%
```

**3. Waveform（波形）**

```
種類:
1. Sine: 滑らか
2. Triangle: やや角ばった
3. Square: カクカク（オン/オフ）
4. Saw Up: 上昇のこぎり
5. Saw Down: 下降のこぎり
6. Random (S&H): ランダム

用途:
ビブラート: Sine
ワウワウ: Triangle、Sine
トレモロ: Square
ランダム効果: Random
```

### LFO応用例

**1. ビブラート（Pitch変化）**

```
設定:
Destination: OSC Pitch
Waveform: Sine
Rate: 5 Hz
Depth: 5%

効果:
ピッチが上下に揺れる
→ 歌声のビブラート
→ リード、パッドに使用
```

**2. ワウワウ（Filter変化）**

```
設定:
Destination: Filter Cutoff
Waveform: Triangle
Rate: 1/4（テンポ同期）
Depth: 40%

効果:
Filter Cutoffが周期的に開閉
→ ワウワウ音
→ Funkベース、Technoリード
```

**3. トレモロ（Volume変化）**

```
設定:
Destination: Amp Volume
Waveform: Sine
Rate: 8 Hz
Depth: 30%

効果:
音量が周期的に変化
→ トレモロ効果
→ パッド、FX
```

**4. オートパン（Pan変化）**

```
設定:
Destination: Pan
Waveform: Sine
Rate: 1/8（テンポ同期）
Depth: 80%

効果:
音が左右に動く
→ ステレオ効果
→ パッド、FX、リード
```

---

## 4つの要素を組み合わせた音作り

### 例1: Sub Bass

```
目標: 深く太い Sub Bass

Oscillator:
- Waveform: Sine
- Octave: 0
- Detune: 0

Filter:
- Type: Low Pass 24 dB
- Cutoff: 150 Hz
- Resonance: 0%

Amp Envelope:
- Attack: 0 ms
- Decay: 50 ms
- Sustain: 100%
- Release: 10 ms

LFO:
- なし

理由:
- Sine波: 倍音なし、純粋な低音
- LP 150 Hz: 150 Hz以上を完全カット
- Attack 0 ms: 即座に鳴る
- LFOなし: 安定した低音
```

### 例2: Techno Acid Bass

```
目標: TB-303風のアシッドベース

Oscillator:
- Waveform: Saw
- Octave: 0
- Detune: 0

Filter:
- Type: Low Pass 24 dB
- Cutoff: 500 Hz（開始点）
- Resonance: 70%

Filter Envelope:
- Attack: 0 ms
- Decay: 200 ms
- Sustain: 20%
- Release: 50 ms
- Envelope Amount: +50

Amp Envelope:
- Attack: 0 ms
- Decay: 100 ms
- Sustain: 80%
- Release: 20 ms

LFO:
- なし（Filter EnvelopeとCutoffオートメーションで動き）

理由:
- Saw波: 明るい音色
- Resonance 70%: ピコピコ音
- Filter Envelope: 音色が時間的に変化
- Cutoffオートメーション: 手動で動かす
```

### 例3: Ambient Pad

```
目標: 柔らかく広がりのあるパッド

Oscillator 1:
- Waveform: Triangle
- Octave: 0
- Volume: 100%

Oscillator 2:
- Waveform: Saw
- Octave: +1
- Detune: +12 cent
- Volume: 60%

Filter:
- Type: Low Pass 12 dB
- Cutoff: 1500 Hz
- Resonance: 10%

Amp Envelope:
- Attack: 1000 ms
- Decay: 1500 ms
- Sustain: 70%
- Release: 3000 ms

LFO 1:
- Destination: Filter Cutoff
- Waveform: Sine
- Rate: 0.3 Hz
- Depth: 15%

理由:
- 2 OSC Detune: 広がり
- Attack 1000 ms: ゆっくり立ち上がる
- Release 3000 ms: 長い余韻
- LFO → Filter: 音色がゆっくり変化
```

---

## Wavetableでの実践

### 基本パッチ1: Sub Bass

```
手順:
1. Wavetableを追加
2. OSC 1: Basic Shapes > Sine
3. OSC 2: Off
4. Filter: Low Pass 24 dB
5. Cutoff: 150 Hz
6. Amp Envelope:
   - A: 0, D: 50, S: 100%, R: 10
7. テスト: C1, F1, G1 を演奏

確認:
スペクトラムで 50-100 Hz が強い
150 Hz 以上はほぼなし
```

### 基本パッチ2: Techno Bass

```
手順:
1. Wavetableを追加
2. OSC 1: Basic Shapes > Saw
3. Filter: Low Pass 24 dB
4. Cutoff: 500 Hz
5. Resonance: 60%
6. Filter Envelope:
   - A: 0, D: 200, S: 20%, R: 50
   - Amount: +40
7. Amp Envelope:
   - A: 0, D: 100, S: 80%, R: 20
8. Macro 1 に Filter Cutoff をマッピング
9. テスト: C2 で16分音符パターン

確認:
Filter Cutoffを動かすと「ピコピコ」
Resonanceで独特の音色
```

### 基本パッチ3: Simple Lead

```
手順:
1. Wavetableを追加
2. OSC 1: Basic Shapes > Saw
3. OSC 2: Basic Shapes > Square
   - Detune: +7 cent
   - Volume: 60%
4. Unison: 4 voices, Detune 10%
5. Filter: Low Pass 12 dB
6. Cutoff: 2500 Hz
7. Resonance: 15%
8. Amp Envelope:
   - A: 10, D: 200, S: 70%, R: 300
9. LFO 1 → OSC 1 Pitch:
   - Waveform: Sine
   - Rate: 5 Hz
   - Depth: 5%

確認:
2 OSC Detuneで太い音
Unisonで広がり
LFOでビブラート
```

---

## よくある質問（FAQ）

**Q1: どの波形を使えばいいですか？**

```
A: 用途で選ぶ

迷ったら:
ベース: Saw（80%）、Square（20%）
リード: Saw（70%）、Square（30%）
パッド: Triangle（50%）、Saw（50%）
Sub: Sine（100%）

理由:
Saw: 倍音が多い → 明るい、削りやすい
Sine: 倍音なし → 純粋な低音
Square: 中空的 → Deep House風
Triangle: 柔らかい → パッド
```

**Q2: Filter Cutoffをどこに設定すればいいですか？**

```
A: 音域で決める

目安:
Sub Bass: 100-200 Hz
ベース: 300-800 Hz
リード: 1500-4000 Hz
パッド: 1000-2000 Hz

調整方法:
1. Cutoffを最大（20000 Hz）にする
2. ゆっくり下げる
3. 「これだ！」という点で止める
4. 微調整
```

**Q3: Envelopeの設定がうまくいきません**

```
A: 音の種類で設定を変える

短い音（ドラム、Pluck）:
A: 0 ms
D: 100-300 ms
S: 0%
R: 50 ms

持続音（ベース、リード）:
A: 0-50 ms
D: 100-500 ms
S: 70-100%
R: 100-500 ms

ゆっくりした音（パッド）:
A: 500-2000 ms
D: 1000 ms
S: 70%
R: 2000-5000 ms
```

**Q4: LFOはいつ使いますか？**

```
A: 動きが欲しい時

用途:
静的な音 → LFOなし
動きのある音 → LFO使用

例:
Sub Bass: LFOなし（安定が重要）
リード: LFO → Pitch（ビブラート）
パッド: LFO → Filter（ゆっくり変化）
FX: LFO → Pan（左右に動く）

初心者:
最初はLFOなしでOK
慣れたら追加
```

---

## 練習方法

### Week 1: 波形の理解

```
Day 1-2: 4つの波形を聴き比べ
1. Wavetable追加
2. 各波形でC3を演奏
3. 違いを体感

Day 3-4: Filterで削る
1. Saw波を選択
2. Cutoffを20000 Hz → 200 Hz
3. 音色の変化を聴く

Day 5-7: Resonanceの効果
1. Cutoff 500 Hz
2. Resonance 0% → 80%
3. 「ピコピコ」音を体感
```

### Week 2: Envelope練習

```
Day 1-2: Attack変化
1. Sub Bass作成
2. Attack 0 ms → 2000 ms
3. 立ち上がりの変化を聴く

Day 3-4: Release変化
1. Pluck音作成
2. Release 10 ms → 5000 ms
3. 余韻の変化を聴く

Day 5-7: Filter Envelope
1. Pluck音にFilter Envelope追加
2. Envelope Amount 0 → +60
3. 音色の時間的変化を聴く
```

### Week 3: LFO練習

```
Day 1-2: ビブラート
1. リード作成
2. LFO → Pitch
3. Rate、Depth調整

Day 3-4: ワウワウ
1. ベース作成
2. LFO → Filter Cutoff
3. Waveform変更（Sine、Triangle、Square）

Day 5-7: オートパン
1. パッド作成
2. LFO → Pan
3. Rate 1/8（テンポ同期）
```

### Week 4: 総合練習

```
Day 1-2: 基本パッチ10個作成
1. Sub Bass
2. Techno Bass
3. Deep House Bass
4. Simple Lead
5. Pluck
6. Simple Pad
7. FX (Riser)
8. FX (Impact)
9. Bass with LFO
10. Pad with LFO

Day 3-7: ジャンル別音作り
参考曲を選び、特定の音を再現
50%似たら成功
```

---

## まとめ

シンセシスの基礎は、4つの要素で構成されます。

**4つの要素:**

1. **Oscillator**: 音の源、波形選択
2. **Filter**: 音色を削る、Cutoff/Resonance
3. **Envelope**: 時間的変化、ADSR
4. **LFO**: 周期的変化、Rate/Depth

**重要ポイント:**

- **Saw波**: 最も汎用的、ベース・リードの基本
- **Filter Cutoff**: サウンドの80%を決める最重要パラメータ
- **Amp Envelope**: 全ての音に必須
- **LFO**: 動きのある音を作る

**学習順序:**
1. 波形の違いを聴き比べ（Sine、Saw、Square、Triangle）
2. Filter Cutoffで音色を削る練習
3. Amp Envelopeで時間的変化を作る
4. LFOで周期的変化を追加

**次のステップ:** [Wavetable Sound Design](./wavetable-sound-design.md) へ進む

---

## 関連ファイル

- **[Wavetable Sound Design](./wavetable-sound-design.md)** - Wavetableでの実践的音作り
- **[03-instruments/wavetable.md](../03-instruments/wavetable.md)** - Wavetable詳細
- **[00-fundamentals/audio-basics.md](../../00-fundamentals/audio-basics.md)** - 音響基礎

---

**シンセシスの基礎をマスターして、自分だけの音を作りましょう！** 🎹
