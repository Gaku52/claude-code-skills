# Sound Design（サウンドデザイン）

オリジナルの音を作るサウンドデザイン技術を完全マスターします。シンセシス、サンプリング、レイヤリング、モジュレーションを駆使して、個性的なサウンドを生み出します。

## この章で学ぶこと

- シンセシスの基礎（減算式、FM、ウェーブテーブル）
- Wavetableでの実践的な音作り
- FM合成（Operator）での音作り
- サンプリング技術とサンプル加工
- レイヤリング（複数の音を重ねる技術）
- モジュレーション（LFO、エンベロープ）
- ジャンル別サウンド作成
- プロのサウンドデザインワークフロー

---

## なぜSound Designが重要なのか

**個性の差:**

```
プリセットのみ:

結果:
誰でも同じ音
個性なし
埋もれる楽曲

リスナー反応:
「どこかで聴いた音」
記憶に残らない
シェアされない

サウンドデザインあり:

結果:
オリジナルの音
強い個性
目立つ楽曲

リスナー反応:
「この音すごい！」
記憶に残る
シェアされる

プロとアマの差:

アマ:
プリセット使用率 90%
サウンド編集 10%
個性なし

プロ:
プリセット使用率 30%
サウンド編集 70%
強い個性

結果の違い:
アマ: 埋もれる
プロ: 目立つ、記憶に残る、契約獲得
```

**制作における重要性:**

- **個性**: 他と差別化できる唯一の音
- **ブランディング**: 「あの人の音だ」と認識される
- **創造性**: 無限の可能性、制約なし
- **プロへの道**: レーベル契約の決め手

**音楽ジャンルとサウンドデザイン:**

```
Techno:
必須度: ★★★★★（100%必須）
理由: アシッドベース、工業的サウンド、独自性が命

Dubstep:
必須度: ★★★★★（100%必須）
理由: ワブルベース、グロウルは完全なサウンドデザイン

House:
必須度: ★★★☆☆（60%）
理由: ベースライン、リードは必須、ドラムはサンプルでOK

Hip Hop:
必須度: ★★☆☆☆（30%）
理由: サンプリングメイン、加工技術は必須

Trance:
必須度: ★★★★☆（80%）
理由: プラック、パッド、リードは独自性重要
```

**サウンドデザインのレベル:**

```
レベル1（初心者・0-3ヶ月）:
- プリセット選択と微調整
- Filter Cutoff、Resonanceのみ
- 基本的なエンベロープ
→ 結果: 70%の音は作れる

レベル2（中級者・3-12ヶ月）:
- Oscillator設定
- LFOによるモジュレーション
- エフェクトチェイン構築
→ 結果: 85%の音は作れる

レベル3（上級者・12-24ヶ月）:
- 複雑なFM合成
- 高度なレイヤリング
- Macroによる演奏性
→ 結果: 95%の音は作れる

レベル4（プロ・24ヶ月以上）:
- 全パラメータ理解
- ジャンル特化音作り
- オリジナルサンプル作成
→ 結果: 100%、完全オリジナル
```

---

## 学習の順序

### ステップ1: シンセシスの基礎理解（Week 1-2）

**目標**: 音がどのように作られるかを理解する

```
学習内容:
1. Oscillator（音の源）
   - Waveform（波形）の種類
   - Sine、Saw、Square、Triangle
   - 倍音構造の違い

2. Filter（音色を削る）
   - Low Pass、High Pass、Band Pass
   - Cutoff（カットオフ周波数）
   - Resonance（レゾナンス）

3. Envelope（時間的変化）
   - ADSR（Attack、Decay、Sustain、Release）
   - Filter Envelope、Amp Envelope

4. LFO（周期的変化）
   - Rate（速度）、Depth（深さ）
   - Waveform（波形）

実践:
- synthesis-basics.md を完全理解
- Wavetableで基本パッチ10個作成
- 各要素の役割を体感
```

### ステップ2: Wavetableでの音作り（Week 3-4）

**目標**: Abletonの主力シンセでベース、リード、パッドを作れる

```
学習内容:
1. Sub Bass作成
   - Sine波1つ
   - Filter LP 200 Hz
   - Amp Envelope短め

2. Techno Bass作成
   - Saw波
   - Filter LP + Resonance
   - Filter Envelope

3. Lead作成
   - 2 OSC（Saw + Square）
   - Unison + Detune
   - Filter動き

4. Pad作成
   - 2 OSC（Triangle + Saw）
   - Long Attack
   - Reverb

実践:
- wavetable-sound-design.md で4種類作成
- 各音色の用途理解
- MIDIで演奏テスト
```

### ステップ3: FM合成（Operator）（Week 5-6）

**目標**: Operatorで金属音、ベル、ブラスを作れる

```
学習内容:
1. FM合成の原理
   - Carrier（音を出す）
   - Modulator（音色を変える）
   - Ratio（倍音比率）

2. Algorithm理解
   - 11種類のAlgorithm
   - 用途別使い分け

3. ベル音作成
   - Algorithm 11
   - Ratio設定（1:2.7）

4. ブラス音作成
   - Algorithm 8
   - Filter + Envelope

実践:
- fm-sound-design.md で4種類作成
- Algorithmの違いを体感
- Ratioによる音色変化理解
```

### ステップ4: サンプリング技術（Week 7-8）

**目標**: サンプルを加工して新しい音を作る

```
学習内容:
1. Samplerの使い方
   - Zone設定
   - Loop Point
   - Pitch Envelope

2. ワンショット加工
   - ドラムサンプル加工
   - Filter、Amp、Pitch

3. ボーカルチョップ
   - スライス
   - キーマッピング
   - Grain Delay

4. テクスチャサウンド
   - 環境音加工
   - Reverse、Stretch
   - Grain効果

実践:
- sampling-techniques.md で4種類作成
- 既存サンプルを完全変化させる
- オリジナルサンプル録音
```

### ステップ5: レイヤリング（Week 9-10）

**目標**: 複数の音を重ねて厚みのあるサウンドを作る

```
学習内容:
1. レイヤリング基礎
   - 周波数帯域の住み分け
   - Low（60-200 Hz）、Mid（200-2k）、High（2k-10k）
   - EQで調整

2. ベースのレイヤリング
   - Sub（Sine）+ Mid（Saw）
   - 各トラックにEQ

3. リードのレイヤリング
   - Main Lead + Octave Lead
   - Detune + Stereo Width

4. ドラムのレイヤリング
   - Kick（Sub + Punch + Click）
   - Snare（Body + Snap）

実践:
- layering.md で4種類作成
- 3層レイヤリング構築
- 周波数スペクトラム確認
```

### ステップ6: モジュレーション（Week 11-12）

**目標**: LFO、Envelopeで動きのある音を作る

```
学習内容:
1. LFOモジュレーション
   - Pitch（ビブラート）
   - Filter（ワウワウ）
   - Volume（トレモロ）
   - Pan（オートパン）

2. Envelope Follower
   - オーディオに反応
   - サイドチェイン的効果

3. Macro Knob
   - 複数パラメータ同時コントロール
   - 演奏性向上

4. MIDI CC
   - Mod Wheel
   - Aftertouch

実践:
- modulation-techniques.md で4種類作成
- 動きのある音作成
- Macro設定
```

### ステップ7: ジャンル別サウンド（Week 13-16）

**目標**: Techno、House、Dubstepの代表的サウンドを作れる

```
学習内容:
1. Techno
   - アシッドベース（TB-303風）
   - 工業的リード
   - パーカッシブシンセ

2. House
   - Deep Houseベース
   - Pluckシンセ
   - Soulfulパッド

3. Dubstep
   - ワブルベース
   - グロウル
   - Riser

4. その他ジャンル
   - Trance: Pluck、Supersaw
   - Hip Hop: サンプル加工
   - D&B: リースベース

実践:
- genre-sounds.md で各ジャンル2-3種類作成
- ジャンル特性理解
- 代表曲と比較
```

---

## Ableton Live 12 でのサウンドデザイン

### 使用する主要楽器

**1. Wavetable（使用頻度: 70%）**

```
用途:
- ベース（Sub、Techno、Dubstep）
- リード（Main、Pluck）
- パッド（Ambient、Wide）

特徴:
- 2 Oscillator
- 強力なFilter
- 豊富なWavetable
- 初心者に優しい

推奨プリセット改変率: 50-70%
```

**2. Operator（使用頻度: 20%）**

```
用途:
- ベル音
- ブラス
- 金属音
- エレピ

特徴:
- 4 Operator FM合成
- 11種類のAlgorithm
- 複雑な倍音

推奨プリセット改変率: 30-50%
```

**3. Sampler（使用頻度: 15%）**

```
用途:
- ボーカルチョップ
- テクスチャ
- ワンショット加工
- オリジナルサンプル

特徴:
- Zone設定
- Loop Point
- Modulationマトリクス

推奨プリセット改変率: 70-100%（完全加工）
```

**4. Analog（使用頻度: 10%）**

```
用途:
- ビンテージサウンド
- Warm Pad
- Classic Lead

特徴:
- アナログモデリング
- Moogスタイル
- 太い音

推奨プリセット改変率: 40-60%
```

**5. Drift（Live 12新機能、使用頻度: 5%）**

```
用途:
- アナログサウンド
- LoFiテクスチャ
- Drift特有の揺らぎ

特徴:
- ビンテージシンセ再現
- Noise、Drift機能
- Warm音質

推奨プリセット改変率: 50-70%
```

### エフェクトチェイン構築

**基本的なエフェクト順序:**

```
シンセ音源
   ↓
1. EQ Eight（不要な周波数削除）
   - High Pass: 30-50 Hz
   - Low Cut: 役割に応じて
   ↓
2. Saturator（倍音追加）
   - Drive: 2-8 dB
   - Curve: Warm、Analog Clip
   ↓
3. Filter（音色調整）
   - Auto Filter
   - Cutoff、Resonance
   ↓
4. Modulation（動き）
   - Chorus、Phaser、Flanger
   ↓
5. Reverb/Delay（空間）
   - Dry/Wet 10-30%
   ↓
6. Limiter（音量制御）
   - Ceiling -0.3 dB
```

**ジャンル別エフェクトチェイン:**

```
Technoベース:
Wavetable → Saturator → Auto Filter → Delay (1/8)

Dubstep Wobble:
Wavetable → Auto Filter (LFO) → Erosion → Limiter

Deep Houseパッド:
Wavetable → Chorus → Reverb (Plate) → EQ Eight
```

---

## サウンドデザインのワークフロー

### フェーズ1: コンセプト決定（5分）

```
質問:
1. どんな音を作りたいか？
   - ベース、リード、パッド、FX

2. ジャンルは？
   - Techno、House、Dubstep

3. 参考曲はあるか？
   - Spotifyで探す
   - 0:30-1:00の音を分析

4. 音の特徴は？
   - 太い、鋭い、柔らかい
   - 動きがある、静的
```

### フェーズ2: 音源選択（5分）

```
選択基準:
- ベース: Wavetable 80%、Operator 20%
- リード: Wavetable 70%、Analog 30%
- パッド: Wavetable 60%、Analog 40%
- FX: Sampler 80%、Wavetable 20%

手順:
1. 楽器をトラックに追加
2. プリセット検索（必要なら）
3. 初期化（Cmd+Shift+I）またはプリセット読み込み
```

### フェーズ3: 基本音作り（15-30分）

```
ステップ1: Oscillator設定（5分）
- Waveform選択
- Octave設定
- 2nd OSCデチューン

ステップ2: Filter設定（5分）
- Filter Type選択
- Cutoff調整
- Resonance調整
- Filter Envelope設定

ステップ3: Amp Envelope設定（5分）
- Attack（0-500 ms）
- Decay（0-1000 ms）
- Sustain（50-100%）
- Release（10-2000 ms）

ステップ4: テスト演奏（5分）
- MIDI入力
- オクターブ確認
- ベロシティ確認

ステップ5: LFO追加（10分）
- LFO → Pitch（ビブラート）
- LFO → Filter（ワウワウ）
- Rate、Depth調整
```

### フェーズ4: エフェクト追加（10-20分）

```
ステップ1: EQ Eight（3分）
- High Pass 30-50 Hz
- 不要な周波数カット

ステップ2: Saturator（5分）
- Drive 2-8 dB
- 倍音追加

ステップ3: 空間系（5分）
- Reverb or Delay
- Dry/Wet 10-30%

ステップ4: 最終調整（7分）
- Volume Balance
- Limiter
```

### フェーズ5: Macro設定（10分）

```
推奨Macro:
Macro 1: Filter Cutoff
Macro 2: Resonance
Macro 3: Dry/Wet
Macro 4: Drive

手順:
1. Audio Effect Rackで囲む
2. Macroに各パラメータをマッピング
3. 範囲設定（Min、Max）
4. Macro名変更
```

### フェーズ6: 保存（5分）

```
手順:
1. プリセット保存
   - 名前: "TB303 Bass v1"
   - カテゴリ: Bass、Lead、Pad

2. Audio Effect Rack保存
   - FXチェインごと保存

3. プロジェクト保存
   - Cmd+S

4. バックアップ
   - Collect All and Save
```

**合計時間: 50-80分/音色**

---

## 実践: 4つの必須サウンド作成

### プロジェクト1: Sub Bass（Techno、House）

**目標**: 深く太い Sub Bass を作る

```
使用楽器: Wavetable

設定:
OSC 1:
- Wavetable: Basic Shapes > Sine
- Octave: 0
- Volume: 100%

OSC 2:
- Off

Filter:
- Type: Low Pass (24 dB)
- Cutoff: 200 Hz
- Resonance: 0%

Amp Envelope:
- Attack: 0 ms
- Decay: 50 ms
- Sustain: 100%
- Release: 10 ms

エフェクト:
- Saturator: Drive 3 dB, Curve Warm
- EQ Eight: High Pass 30 Hz

Macro:
Macro 1: Filter Cutoff (100-300 Hz)
Macro 2: Saturator Drive (0-8 dB)

テストMIDI:
- Note: C1, F1, G1（低音域）
- Velocity: 100-127
- Length: 1拍

完成基準:
- スペクトラム: 50-100 Hz が最大
- 200 Hz 以上はほぼなし
- クラブで腹に響く音
```

### プロジェクト2: Techno Acid Bass（Techno）

**目標**: TB-303風のアシッドベースを作る

```
使用楽器: Wavetable

設定:
OSC 1:
- Wavetable: Basic Shapes > Saw
- Octave: 0
- Volume: 100%

OSC 2:
- Off

Filter:
- Type: Low Pass (24 dB)
- Cutoff: 500 Hz（オートメーション）
- Resonance: 60%

Filter Envelope:
- Attack: 0 ms
- Decay: 200 ms
- Sustain: 20%
- Release: 50 ms
- Envelope Amount: +40

Amp Envelope:
- Attack: 0 ms
- Decay: 100 ms
- Sustain: 80%
- Release: 20 ms

エフェクト:
- Saturator: Drive 6 dB, Curve A Bit Warmer
- Delay: 1/8 Dotted, Dry/Wet 15%

Macro:
Macro 1: Filter Cutoff (200-2000 Hz)
Macro 2: Resonance (30-80%)
Macro 3: Envelope Amount (0-60)
Macro 4: Delay Dry/Wet (0-30%)

テストMIDI:
- Note: C2-C3（16分音符パターン）
- Velocity: 80-127（ランダム）
- Accent（高ベロシティ）でFilter開く

完成基準:
- Filter Cutoffをオートメーション
- Resonanceで「ピコピコ」音
- クラブで踊れるグルーヴ
```

### プロジェクト3: Lead Synth（House、Trance）

**目標**: メインメロディを引き立てるリードを作る

```
使用楽器: Wavetable

設定:
OSC 1:
- Wavetable: Modern Shapes > Formant Square
- Octave: 0
- Volume: 100%

OSC 2:
- Wavetable: Basic Shapes > Saw
- Octave: 0
- Detune: +7 cent
- Volume: 80%

Sub:
- Volume: 20%

Unison:
- Amount: 4 voices
- Detune: 15%

Filter:
- Type: Low Pass (12 dB)
- Cutoff: 2000 Hz
- Resonance: 20%
- Envelope Amount: +20

Filter Envelope:
- Attack: 50 ms
- Decay: 500 ms
- Sustain: 50%
- Release: 200 ms

Amp Envelope:
- Attack: 10 ms
- Decay: 200 ms
- Sustain: 70%
- Release: 300 ms

LFO 1:
- Destination: OSC 1 Pitch
- Waveform: Sine
- Rate: 5 Hz
- Depth: 5%（ビブラート）

エフェクト:
- Chorus: Rate 0.5 Hz, Amount 30%
- Reverb: Decay 1.5s, Dry/Wet 25%
- EQ Eight: High Pass 200 Hz

Macro:
Macro 1: Filter Cutoff (500-4000 Hz)
Macro 2: Reverb Dry/Wet (0-40%)
Macro 3: Unison Detune (0-30%)
Macro 4: LFO Depth (0-10%)

テストMIDI:
- Note: C3-C5（メロディ）
- Velocity: 90-127
- Length: 1/4-1拍

完成基準:
- メロディが明瞭に聞こえる
- 広がりがある（Stereo Width 80%）
- 存在感がある
```

### プロジェクト4: Ambient Pad（全ジャンル）

**目標**: 空間を埋める柔らかいパッドを作る

```
使用楽器: Wavetable

設定:
OSC 1:
- Wavetable: FM > Harmonic Flight
- Octave: 0
- Volume: 100%

OSC 2:
- Wavetable: Additive Resonant > Formant Vowels
- Octave: +1
- Detune: +12 cent
- Volume: 70%

Sub:
- Volume: 0%

Unison:
- Amount: 6 voices
- Detune: 20%

Filter:
- Type: Low Pass (12 dB)
- Cutoff: 1500 Hz
- Resonance: 10%

Amp Envelope:
- Attack: 800 ms（ゆっくり）
- Decay: 1000 ms
- Sustain: 80%
- Release: 2000 ms（長い）

LFO 1:
- Destination: Filter Cutoff
- Waveform: Sine
- Rate: 0.2 Hz（ゆっくり）
- Depth: 15%

エフェクト:
- Chorus: Rate 0.3 Hz, Amount 40%
- Reverb: Decay 4s, Dry/Wet 40%
- Delay: 1/4, Dry/Wet 20%, Feedback 30%
- EQ Eight: High Pass 300 Hz、Low Pass 8000 Hz

Macro:
Macro 1: Reverb Dry/Wet (20-60%)
Macro 2: Filter Cutoff (800-3000 Hz)
Macro 3: LFO Rate (0.1-1 Hz)
Macro 4: Attack Time (200-2000 ms)

テストMIDI:
- Note: C3、E3、G3（コード）
- Velocity: 80-100
- Length: 4-8拍（長い）

完成基準:
- アタックがゆっくり（800 ms）
- 広がりがある（Stereo Width 100%）
- 柔らかい音色
- 空間を埋める
```

---

## よくある質問（FAQ）

**Q1: プリセットを使うのは悪いことですか？**

```
A: 全く悪くありません。

プロの使用率:
- プリセット使用: 30-50%
- プリセット改変: 30-40%
- ゼロから作成: 20-30%

推奨ワークフロー:
1. プリセット検索（5分）
2. 近いものを選択
3. 50%以上改変
4. 自分のプリセットとして保存

理由:
- 時間効率
- 学習効果（プロの設定を見れる）
- 完成が第一優先
```

**Q2: サウンドデザインに何時間かけるべきですか？**

```
A: 1音色あたり30-60分が目安

時間配分:
- ベース: 30-45分
- リード: 45-60分
- パッド: 30-45分
- FX: 15-30分

注意:
- 2時間以上は危険（完璧主義の罠）
- 60%の完成度でOK
- 後で改善できる

プロの習慣:
- 1音色 = 30分
- 5音色/日 = 2.5時間
- 完成を優先
```

**Q3: どのシンセを学ぶべきですか？**

```
A: Wavetableを最優先

推奨順序:
1. Wavetable（最初の6ヶ月）
   - 70%の音はこれで作れる
   - 初心者に優しい
   - Techno、House、Dubstep全対応

2. Operator（6-12ヶ月後）
   - FM合成の理解
   - ベル、ブラス
   - 30%の音

3. Sampler（12ヶ月後）
   - サンプル加工
   - ボーカルチョップ
   - オリジナリティ

4. その他（2年後）
   - Analog、Drift
   - 特殊用途
```

**Q4: オリジナルサウンドを作るコツは？**

```
A: レイヤリング + エフェクト + 偶然

コツ:
1. 2-3音を重ねる
   - Sub + Mid + High

2. 予想外のエフェクト
   - Erosion、Redux
   - Grain Delay

3. 偶然を活かす
   - LFOランダム
   - サンプルReverse

4. 制約を設ける
   - "Sine波のみ"
   - "1 OSCのみ"

5. 参考曲を徹底分析
   - スペクトラム確認
   - 再現を試みる
   - 50%似たら成功
```

**Q5: サウンドデザインが上達しません。どうすれば？**

```
A: 毎日1音色作成を30日継続

30 Day Sound Design Challenge:
Week 1-2: Sub Bass 14種類
Week 3-4: Techno Bass 14種類
Week 5-6: Lead 14種類
Week 7-8: Pad 14種類

結果:
- 56音色作成
- パターン理解
- 速度向上（60分→20分）

学習方法:
1. 参考曲選択
2. 特定の音を再現
3. 50%似たら成功
4. 保存してライブラリ化

継続のコツ:
- 毎日同じ時間（朝9時等）
- 1音色=30分制限
- SNSでシェア（#30DaySoundDesign）
```

**Q6: プロのサウンドに近づけるには？**

```
A: リファレンストラック + スペクトラム分析

手順:
1. Spotifyで参考曲選択
   - 同じジャンル
   - プロのリリース曲

2. Abletonにインポート
   - Audio Track

3. スペクトラム表示
   - EQ Eight（Analyzer On）

4. 周波数分布確認
   - どこが強調されているか
   - どこが削られているか

5. 自分の音と比較
   - 差分を埋める
   - EQ、Saturatorで調整

6. A/B切り替え
   - 10秒ごとに切り替え
   - 違いを聞き取る

7. 80%似たら成功
   - 100%は不可能
   - 80%で十分プロレベル
```

---

## まとめ

サウンドデザインは、楽曲制作における**最も創造的で個性的なスキル**です。

**重要ポイント:**

1. **Wavetableを最優先**: 70%の音はこれで作れる
2. **プリセット改変OK**: 50%以上改変すれば自分の音
3. **時間制限**: 1音色30-60分、完璧主義は禁物
4. **レイヤリング**: 2-3音を重ねて厚みを出す
5. **リファレンス**: プロの音を徹底分析
6. **毎日継続**: 30日で56音色作成チャレンジ
7. **保存**: 自分のライブラリを構築

**学習順序:**
1. synthesis-basics.md - シンセシス基礎理解
2. wavetable-sound-design.md - Wavetableで4種類作成
3. fm-sound-design.md - Operatorで特殊音作成
4. sampling-techniques.md - サンプル加工技術
5. layering.md - レイヤリングで厚み
6. modulation-techniques.md - 動きのある音
7. genre-sounds.md - ジャンル特化サウンド

**次のステップ:** [Synthesis Basics（シンセシス基礎）](./synthesis-basics.md) へ進む

---

## 関連ファイル

- **[03-instruments](../03-instruments/)** - 音源・楽器の使い方
- **[04-effects](../04-effects/)** - エフェクト活用
- **[05-mixing](../05-mixing/)** - ミキシング技術
- **[07-workflow](../07-workflow/)** - 効率的ワークフロー
- **[00-fundamentals/音楽理論](../../00-fundamentals/music-theory.md)** - 音楽理論基礎

---

**サウンドデザインで、あなただけの音を作りましょう！** 🎹🔊
