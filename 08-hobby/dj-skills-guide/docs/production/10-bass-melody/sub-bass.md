# サブベース（Sub Bass）

**20-60 Hz 超低域を完全マスター**

サブベースは、人間の耳ではほとんど聞こえない20-60 Hzの超低域ですが、クラブのサブウーファーで「体感」する重低音です。Dubstep、Techno、Trapなど、現代のダンスミュージックにおいて、サブベースは楽曲の**物理的な存在感**を決定づける最重要要素です。

---

## この章で学ぶこと

- ✅ 20-60 Hz領域の科学的理解
- ✅ サイン波サブベースの作成（Wavetable）
- ✅ Mono処理の必須性
- ✅ EQ/フィルター処理
- ✅ サブベース + Mid Bassのレイヤリング
- ✅ クラブシステムでの最適化
- ✅ ジャンル別サブベース設定

**学習時間**: 2-4時間
**難易度**: ★★★☆☆ 中級

---

## なぜサブベースが重要なのか

### DJの視点から

**DJとして**:
- クラブで**床が振動する**瞬間を体感している
- サブベースの有無で**フロアのエネルギーが変わる**
- 50 Hzと60 Hzの違いを**体で感じている**

**プロデューサーとして**:
- その「振動」を**自分で作り出す**ことができる
- 周波数を1 Hzレベルで**正確にコントロール**
- ヘッドフォンでは聞こえなくても**クラブでは破壊力**

### プロの意見

> "サブベースがなければDubstepじゃない。50 Hzの純粋なサイン波が全ての基礎。"
> — **Skrillex**

> "Technoでは55-60 Hzのサブベースがキックと融合して、クラブのコンクリート床を揺らす。"
> — **Richie Hawtin**

> "808のサブベースは44 Hzが最適。これより低いと音楽的じゃない、高いと迫力が足りない。"
> — **Metro Boomin**

### 数字で見る重要性

| ジャンル | サブベース使用率 | 推奨周波数 | クラブでの体感 |
|---------|--------------|---------|-------------|
| **Dubstep** | 100% | 50 Hz | ★★★★★ |
| **Trap** | 95% | 44 Hz (F1) | ★★★★★ |
| **Techno** | 90% | 55-60 Hz | ★★★★☆ |
| **House** | 80% | 60 Hz | ★★★☆☆ |
| **Drum & Bass** | 70% | 60-65 Hz | ★★★☆☆ |
| **Trance** | 50% | 65 Hz | ★★☆☆☆ |

---

## 1. 周波数の科学

### 1.1 可聴域と体感域

**人間の聴覚**:
```
可聴域: 20 Hz - 20,000 Hz

実際の聞こえ方:
20-60 Hz:   ほとんど聞こえない（体感のみ）
60-250 Hz:  低音として聞こえる
250 Hz以上: 明瞭に聞こえる
```

**サブベース領域**:
```
20-30 Hz:  体感のみ（サブウーファー18インチ以上必須）
30-40 Hz:  かすかに聞こえる + 強い振動
40-50 Hz:  低音として聞こえ始める
50-60 Hz:  低音として明瞭に聞こえる
```

### 1.2 音楽的な周波数

**主要な音とその周波数**:
```
C1:  32.70 Hz  ← 体感重視
C#1: 34.65 Hz
D1:  36.71 Hz
D#1: 38.89 Hz
E1:  41.20 Hz
F1:  43.65 Hz  ← 808サブベース（Trap/Hip Hop）
F#1: 46.25 Hz
G1:  49.00 Hz
G#1: 51.91 Hz
A1:  55.00 Hz
A#1: 58.27 Hz
B1:  61.74 Hz
C2:  65.41 Hz  ← Techno/Houseの境界
```

**ジャンル別最適周波数**:
```
Dubstep:  50 Hz (G#1) - 最も体感が強い
Trap:     44 Hz (F1)  - 808スタイル
Techno:   55 Hz (A1)  - クリアで力強い
House:    60 Hz (B1)  - 温かく安定
D&B:      65 Hz (C2)  - 高速に適合
```

### 1.3 スピーカーシステムの限界

**家庭用機器**:
```
ヘッドフォン:     50 Hz以下はほぼ出ない
PCスピーカー:     80 Hz以下はほぼ出ない
モニタースピーカー: 40-50 Hz（6-8インチ）
                  35 Hz以下は出ない

→ 制作時はスペクトラムアナライザー必須
```

**クラブシステム**:
```
サブウーファー:
18インチ: 30 Hz以下まで対応
21インチ: 25 Hz以下まで対応（映画館級）

→ 真の体感は現場でのみ
```

---

## 2. サブベースの種類

### 2.1 純粋サイン波（Pure Sine Wave）

**特徴**:
- 単一周波数のみ
- 倍音ゼロ
- 最もクリーン
- **最も力強い体感**

**使用ジャンル**:
- Dubstep 90%
- Minimal Techno 80%
- Deep House 70%

**Wavetableでの作成**:
```
Oscillator 1: Basic Shapes → Sine
Oscillator 2: Off
Filter: Off（またはLow Pass 100 Hz）

→ 最もシンプル、最も効果的
```

### 2.2 倍音付きサブベース

**特徴**:
- わずかな倍音（2倍音、3倍音）
- 60-200 Hz領域にも存在感
- ヘッドフォンでも聞こえやすい

**使用ジャンル**:
- Tech House 70%
- Progressive House 80%

**Wavetableでの作成**:
```
Oscillator 1: Sine (Root)
Oscillator 2: Sine (Coarse +12) - 1オクターブ上
              Level 20-30%

→ わずかな倍音で聞こえやすくなる
```

### 2.3 808スタイル（Pitch Envelope）

**特徴**:
- ピッチが下降する
- F1 (44 Hz) から始まる
- 長いリリースタイム

**使用ジャンル**:
- Trap 95%
- Hip Hop 90%
- Future Bass 70%

**Operatorでの作成**:
```
Algorithm: 1 (単一Operator)
Operator A: Sine
Pitch Envelope:
  - Attack 0 ms
  - Decay 100 ms
  - Initial +12 semitones
  - Sustain 0 semitones
Amp Envelope:
  - Release 800 ms

→ 808独特の「ボ〜ン」という下降音
```

---

## 3. Wavetableでの作成（ステップバイステップ）

### Step 1: 新規トラック作成

```
1. Cmd+Shift+T (新規MIDIトラック)
2. Browser → Instruments → Wavetable
3. ドラッグ&ドロップ
```

### Step 2: Oscillator設定

```
Oscillator 1:
  - Category: Basic Shapes
  - Wavetable: Sine
  - Position: 0.00 (完全なサイン波)
  - Level: 0.00 dB

Oscillator 2:
  - Off（または Sub Oscillator）

Sub:
  - Level: +3 dB
  - Transpose: 0 (同じ音程)
  - Wave: Sine
```

### Step 3: Filter設定

**オプション1: Filter Off**
```
Filter 1: Off
→ 最もクリーンなサブベース
```

**オプション2: Low Pass 100 Hz**
```
Filter 1: Low Pass (Clean)
Cutoff: 100 Hz
Resonance: 0%
→ 100 Hz以上の倍音を完全カット
```

### Step 4: Envelope設定

```
Amp Envelope:
  - Attack: 0 ms（即座に鳴る）
  - Decay: 200 ms
  - Sustain: 100%（一定の音量）
  - Release: 100 ms（短く切れる）

Techno/House用:
  - Release: 100 ms（歯切れ良い）

Dubstep用:
  - Release: 300 ms（余韻）
```

### Step 5: Unison Off

```
Unison:
  - Amount: 1（Unison無効）

理由: Unisonはステレオ幅を広げるが、
      サブベースはMono必須
```

### Step 6: MIDI打ち込み

```
1. ダブルクリックで空のMIDI Clip作成
2. ピアノロールで音程選択:
   - Dubstep: G#1 (50 Hz)
   - Techno: A1 (55 Hz)
   - House: B1 (60 Hz)
3. 4つ打ちまたはベースラインに合わせて配置
```

---

## 4. Mono処理（最重要）

### 4.1 なぜMonoが必須か

**物理的理由**:
```
低域（< 120 Hz）:
  - 波長が長い（3-10メートル）
  - ステレオ効果がない
  - 位相ずれで打ち消し合う可能性

Mono化の効果:
  - 位相の問題を完全回避
  - サブウーファーで最大の出力
  - クラブシステムで確実に鳴る
```

**クラブシステム**:
```
サブウーファー配置:
  - 通常、中央に1台または左右に2台
  - 両方から同じ信号を出力（Mono）
  - ステレオサブベースは位相問題を起こす

→ プロの楽曲は100% Mono化されている
```

### 4.2 Utilityでの Mono化

**方法1: Bass Mono機能**
```
1. Wavetable の後に Utility 追加
2. Width: 100% (デフォルト)
3. Bass Mono: On
4. Frequency: 120 Hz

→ 120 Hz以下が自動的にMonoに
```

**方法2: Width 0%**
```
Utility:
  - Width: 0%
  - すべての周波数がMono

→ サブベース専用トラックで使用
```

### 4.3 確認方法

**Correlation Meter**:
```
Utility → Correlation表示:
+1.0: 完全Mono（理想）
 0.0: 無相関
-1.0: 逆位相（最悪）

サブベース:
→ 常に +1.0 を維持
```

**Spectrum Analyzer**:
```
Utility → Spectrum: On
Mid/Side表示:
  - Mid（中央）: 大きいピーク
  - Side（左右）: ほぼゼロ

→ Sideが大きい = Stereo（NG）
```

---

## 5. EQ/フィルター処理

### 5.1 High Pass Filter（必須）

**30 Hz以下をカット**:
```
理由:
  - 20-30 Hzは音楽的でない（ランブル）
  - スピーカーに負担
  - ヘッドルーム圧迫

EQ Eight:
  - High Pass 30 Hz
  - Slope: 24 dB/oct（急峻）
```

### 5.2 ターゲット周波数のブースト

**ジャンル別最適化**:

**Dubstep（50 Hz）**:
```
EQ Eight:
1. High Pass 30 Hz (24 dB/oct)
2. Bell 50 Hz +3 dB, Q=2.0（鋭く）
3. Low Pass 80 Hz (12 dB/oct)（倍音カット）

→ 50 Hz に集中した強力なサブベース
```

**Techno（55 Hz）**:
```
EQ Eight:
1. High Pass 35 Hz (24 dB/oct)
2. Bell 55 Hz +2 dB, Q=1.5
3. Low Pass 100 Hz (12 dB/oct)

→ クリアで力強い
```

**Trap（44 Hz）**:
```
EQ Eight:
1. High Pass 30 Hz (24 dB/oct)
2. Bell 44 Hz +4 dB, Q=2.5（非常に鋭く）
3. Low Pass 70 Hz (24 dB/oct)（808スタイル）

→ 808独特の「パンチ」
```

### 5.3 Low Pass Filter

**不要な倍音除去**:
```
サブベース目標:
  - 20-60 Hz のみ存在
  - 60 Hz以上は Mid Bass の領域

Low Pass:
  - Cutoff: 80-100 Hz
  - Slope: 12 dB/oct（緩やか）

→ 60 Hz以上を徐々にカット
```

---

## 6. レイヤリング（Sub + Mid Bass）

### 6.1 なぜレイヤリングが必要か

**問題**:
```
Sub Bass単体（20-60 Hz）:
  - ヘッドフォンでほぼ聞こえない
  - 小さいスピーカーで聞こえない
  - 音楽的な存在感が弱い

解決:
  - Sub Bass: 20-60 Hz（体感）
  - Mid Bass: 60-250 Hz（聞こえる）
  - 2つを重ねる
```

### 6.2 2トラック構成

**Track 1: Sub Bass**
```
音源: Wavetable Sine
音域: C1-C2 (33-65 Hz)
処理:
  - Utility: Bass Mono On
  - EQ: High Pass 30 Hz、Low Pass 80 Hz
  - 音量: -6 dB（控えめ）

役割: クラブでの体感
```

**Track 2: Mid Bass**
```
音源: Wavetable Saw/Square
音域: C2-C3 (65-130 Hz)
処理:
  - EQ: High Pass 80 Hz、Low Pass 300 Hz
  - Filter: Low Pass 500 Hz、Resonance 20%
  - Utility: Bass Mono On (120 Hz)
  - 音量: -3 dB

役割: ヘッドフォン/家で聞こえる
```

**同じMIDI**:
```
両トラックに同じベースラインを打ち込む:
  - Sub: C1で演奏
  - Mid: C2で演奏（1オクターブ上）

または:
  - 同じC2で演奏
  - EQで周波数帯域を分ける
```

### 6.3 周波数分離

**EQでの完全分離**:

**Sub Bass EQ**:
```
EQ Eight:
1. High Pass 30 Hz
2. Bell 50 Hz +3 dB
3. Low Pass 80 Hz (24 dB/oct)

→ 30-80 Hz のみ
```

**Mid Bass EQ**:
```
EQ Eight:
1. High Pass 80 Hz (24 dB/oct)
2. Bell 120 Hz +2 dB
3. Low Pass 300 Hz (12 dB/oct)

→ 80-300 Hz のみ
```

**確認**:
```
Spectrum Analyzer:
  - Subとの重複がほぼゼロ
  - きれいに分離されている

→ マスキングなし、クリアなサウンド
```

---

## 7. ジャンル別サブベース設定

### 7.1 Dubstep

**目標**: 床を破壊する50 Hzサブベース

**設定**:
```
Wavetable:
  - Osc 1: Sine
  - Sub: +3 dB
  - Filter: Low Pass 70 Hz

MIDI: G#1 (50 Hz)
リズム: ハーフタイム
  |C-------|--------|C-------|--------|
   Kickと同期

EQ Eight:
  1. High Pass 28 Hz
  2. Bell 50 Hz +4 dB, Q=2.5
  3. Low Pass 70 Hz (24 dB/oct)

Utility:
  - Width: 0% (完全Mono)

音量: -3 dB (Masterで十分なヘッドルーム)
```

**Wobble Bass（別トラック）と組み合わせ**:
```
Sub Bass: 50 Hz（変化なし）
Wobble:   100-1000 Hz（フィルターLFO）

→ Subは安定、Wobbleが動く
```

### 7.2 Techno

**目標**: クリアで力強い55 Hzサブベース

**設定**:
```
Wavetable:
  - Osc 1: Sine
  - Filter: Low Pass 100 Hz

MIDI: A1 (55 Hz)
リズム: 4つ打ち
  |C---|C---|C---|C---|
   Kickと完全同期

EQ Eight:
  1. High Pass 35 Hz
  2. Bell 55 Hz +2 dB, Q=1.5
  3. Low Pass 100 Hz (12 dB/oct)

Utility:
  - Bass Mono: On (120 Hz)

Compressor (Sidechain: Kick):
  - Ratio: 6:1
  - Attack: 0 ms
  - Release: 100 ms
  - Threshold: -20 dB

→ Kickが鳴る瞬間だけSubが下がる
```

### 7.3 Trap / Hip Hop

**目標**: 808スタイル44 Hz、ピッチ下降

**Operator設定**:
```
Algorithm: 1
Operator A:
  - Waveform: Sine
  - Coarse: 1.00
  - Fine: 0.00
  - Level: 0.00 dB

Pitch Envelope:
  - Attack: 0 ms
  - Decay: 100 ms
  - Initial: +12 semitones (1オクターブ上から)
  - Sustain: 0 semitones (F1に戻る)
  - Release: 0 ms

Amp Envelope:
  - Attack: 0 ms
  - Decay: 0 ms
  - Sustain: 100%
  - Release: 800-1200 ms (長い余韻)

→ 「ボ〜ン」という下降808サウンド
```

**MIDI**:
```
F1 (44 Hz)
配置: シンコペーション
  |F-------|--F-----|F-F-----|--------|
   808独特のリズム

Velocity: 100-127（強く）
Length: 1/4 - 1/2（長く、リリースで減衰）
```

**EQ**:
```
EQ Eight:
  1. High Pass 30 Hz
  2. Bell 44 Hz +4 dB, Q=2.0
  3. Low Pass 80 Hz (12 dB/oct)
```

### 7.4 House

**目標**: 温かい60 Hzサブベース

**設定**:
```
Wavetable:
  - Osc 1: Sine
  - Osc 2: Sine (Coarse +12, Level 20%)
    → わずかな倍音で温かみ

MIDI: B1 (60 Hz)
リズム: Offbeat
  |-C--|-C--|-C--|-C--|
   Kickの直後

EQ Eight:
  1. High Pass 40 Hz
  2. Bell 60 Hz +2 dB, Q=1.0
  3. High Shelf 150 Hz +1 dB (温かみ)
  4. Low Pass 200 Hz (12 dB/oct)

Saturator:
  - Drive: 3 dB
  - Curve: Warm
  - Dry/Wet: 30%

→ アナログ的な温かさ
```

---

## 8. ミキシング

### 8.1 音量バランス

**サブベースの適切な音量**:
```
Kickとの関係:
  - Kick: -6 dB (Peak)
  - Sub Bass: -9 dB (Peak)

→ Kickより 3 dB 小さい
```

**確認方法**:
```
1. Kickソロ → Peak -6 dB
2. Sub Bassソロ → Peak -9 dB
3. 両方再生 → Peak -3 dB (理想)

Utility → Output Level で調整
```

### 8.2 サイドチェインコンプレッション

**Kickとの共存**:
```
Sub Bass Track:
→ Compressor
→ Audio From: 1-Kick

設定:
  - Ratio: 8:1（強め）
  - Threshold: -20 dB
  - Attack: 0 ms（即座に）
  - Release: 100-150 ms
  - Makeup Gain: 0 dB

効果:
  - Kickが鳴る瞬間、Subが -6 dB 下がる
  - Kickのパンチを維持
  - 周波数の衝突回避
```

**視覚的確認**:
```
Compressor → GR (Gain Reduction) メーター:
  - Kickのタイミングで -6 dB
  - すぐに 0 dB に戻る

→ 正しく動作している
```

### 8.3 Saturation（倍音追加）

**ヘッドフォンでも聞こえるようにする**:

```
Saturator:
  - Drive: 3-6 dB
  - Curve: Warm または A-Shape
  - Dry/Wet: 30-50%

効果:
  - 2倍音、3倍音が追加される
  - 60 Hz → 120 Hz、180 Hz にも成分
  - ヘッドフォンで聞こえやすくなる
```

**注意**:
```
過度なSaturation:
  - サブベースのクリーンさが失われる
  - Mid Bassとの分離が曖昧に

→ Dry/Wet 50%以下推奨
```

---

## 9. クラブシステム最適化

### 9.1 サウンドチェック

**クラブでのテスト**:
```
1. 自分の楽曲をUSBで持参
2. サウンドチェック時に再生
3. フロアで聞く（DJブースではない）
4. サブベースの体感を確認
```

**確認ポイント**:
```
✓ 床が振動しているか
✓ 胸に響くか
✓ 他の楽曲と比較して弱すぎないか
✓ 強すぎて歪んでいないか
```

### 9.2 リファレンストラック比較

**クラブで人気の楽曲と比較**:
```
1. 同じジャンルのプロトラック
2. 自分の楽曲
3. 交互に再生

サブベース比較:
  - 音量レベル
  - 周波数（50 Hz vs 60 Hz）
  - クリアさ
  - 体感の強さ
```

**調整**:
```
自分の楽曲が弱い:
  → Sub Bass +2 dB

自分の楽曲が強すぎる:
  → Sub Bass -2 dB
  → EQ: 50 Hz を -1 dB

→ 家に戻って調整、再テスト
```

### 9.3 Master EQ調整

**クラブシステムに合わせる**:
```
家でのマスタリング:
  - 40-60 Hz を控えめに

クラブでテスト後:
  - 必要に応じて +1〜2 dB ブースト

Master Track EQ:
  - Bell 50 Hz +1 dB, Q=1.0
  - Low Pass 20 Hz (ランブル除去)
```

---

## 10. 練習方法

### 初級（Day 1-3）

**Day 1: サイン波サブベース作成**
```
1. Wavetable: Sine
2. C2 (65 Hz) で4つ打ち
3. Utility: Bass Mono On
4. Spectrum Analyzerで確認
```

**Day 2: EQ処理**
```
1. EQ Eight追加
2. High Pass 30 Hz
3. Bell 65 Hz +3 dB
4. Low Pass 100 Hz
5. Before/After比較
```

**Day 3: サイドチェイン**
```
1. Kickトラックを作成
2. Subにサイドチェインコンプレッサー
3. GR -6 dB を確認
4. Kickとの共存を体感
```

---

### 中級（Week 1-2）

**Week 1: ジャンル別サブベース**
```
Day 1-2: Dubstep 50 Hz
Day 3-4: Techno 55 Hz
Day 5-6: Trap 44 Hz (808)
Day 7: House 60 Hz
```

**Week 2: レイヤリング**
```
Day 1-3: Sub + Mid Bass作成
Day 4-5: 周波数分離EQ
Day 6-7: ミキシング完成
```

---

### 上級（Week 3-4）

**Week 3: リファレンストラック分析**
```
1. プロの楽曲をSpectrum Analyzerで分析
2. サブベースの周波数を特定
3. 同じ周波数で再現
4. 音量レベルを合わせる
```

**Week 4: クラブテスト**
```
1. 楽曲をUSBで持参
2. サウンドチェックで再生
3. フロアで体感
4. 調整して再テスト
```

---

## 11. よくある失敗と対処法

### 失敗1: サブベースが聞こえない

**原因**:
- ヘッドフォンで確認している
- 周波数が低すぎる（30 Hz以下）

**対処法**:
```
1. Spectrum Analyzerで視覚的に確認
2. Mid Bassをレイヤリング
3. クラブシステムで確認
4. Saturationで倍音追加
```

---

### 失敗2: Kickとサブベースがぶつかる

**原因**:
- 同じ周波数帯域
- サイドチェインがない

**対処法**:
```
サイドチェインコンプレッション:
  - Ratio 8:1
  - Attack 0 ms
  - Release 100 ms

周波数分離:
  - Kick: 60 Hz +3 dB
  - Sub: 50 Hz +3 dB
```

---

### 失敗3: サブベースがステレオになっている

**原因**:
- Unison有効
- Utility Mono化なし
- エフェクトでステレオ化

**対処法**:
```
1. Wavetable: Unison 1
2. Utility: Width 0% または Bass Mono On
3. Correlation Meter: +1.0 確認
4. ステレオエフェクト（Chorus等）を削除
```

---

### 失敗4: サブベースが歪む

**原因**:
- 音量が大きすぎる
- Master Limiterで歪み

**対処法**:
```
1. Sub Bass: -9 dB (Peak)
2. Master: -6 dB (ヘッドルーム確保)
3. Limiter Threshold: -0.3 dB
4. サブベースをソロで確認
```

---

## まとめ

### サブベースの核心

1. **20-60 Hzの超低域**: 耳で聞こえない、体で感じる
2. **Mono必須**: 120 Hz以下は必ずMono化
3. **サイン波が基本**: 最もクリーン、最も力強い
4. **ジャンル別周波数**: Dubstep 50 Hz、Techno 55 Hz、Trap 44 Hz
5. **レイヤリング**: Sub (20-60 Hz) + Mid (60-250 Hz)
6. **サイドチェイン**: Kickとの共存に必須

### DJから制作者へ

**DJスキル**:
- クラブでの体感を理解
- ジャンル別の最適周波数を知っている

**プロデューサーとして**:
- その「振動」を作り出せる
- 周波数を正確にコントロール
- クラブで確実に鳴るサブベース

### 次のステップ

1. **[ベースライン作成](./bassline-creation.md)**: サブベースと組み合わせる
2. **[ミキシング](../05-mixing/)**: サブベースを完璧に仕上げる
3. **[ドラム・リズム](../09-drums-rhythm/)**: Kickとの完璧な融合

---

**次のステップ**: [メロディ作成](./melody-creation.md) へ進む

---

**🎵 20-60 Hzの超低域を完全マスターして、クラブの床を揺らしましょう！**
