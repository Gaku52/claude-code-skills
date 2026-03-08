# Depth & Space

Reverb・Delayで奥行きを作ります。前後配置・Pre-Delay・Return Track戦略を完全マスターします。

## この章で学ぶこと

- 奥行きの作り方
- Return Track活用
- Pre-Delay重要性
- Send量調整
- 前中後配置戦略
- 低域Reverb問題
- EQとReverbの組み合わせ

---

## なぜDepth & Spaceが重要なのか

**立体感の実現:**

```
奥行きなし:

特徴:
平坦
2次元
単調

奥行きあり:

特徴:
立体的
3次元
没入感

プロとアマの差:

アマ:
全て同じ距離
平坦

プロ:
前中後明確
立体的

真実:

「プロの音」=
奥行きが完璧

方法:
Reverb・Delay
Pre-Delay設定
```

---

## 奥行きの原理

**音響心理学:**

### 距離感の要因

```
要因1: 音量

大きい:
近い

小さい:
遠い

要因2: 明るさ

明るい (高域多い):
近い

暗い (高域少ない):
遠い

理由:
空気吸収
高域減衰

要因3: Reverb量

Dry (Reverb少ない):
近い

Wet (Reverb多い):
遠い

要因4: Pre-Delay

短い (0-10 ms):
近い
密着

長い (20-40 ms):
適度な距離
分離

結論:

近く:
大きい、明るい、Dry、Pre-Delay短い

遠く:
小さい、暗い、Wet、Pre-Delay長い
```

---

## 前中後配置

**3層構造:**

### 最前列

```
楽器:

Kick:
最も前

Vocal:
Kick並び

Bass:
やや後ろ

Snare:
前

設定:

Reverb Send: 0-10%
Pre-Delay: 15-20 ms
EQ: 明るい (+2 dB @ 3 kHz)
Volume: 大きい (-6 dB)

理由:
パワー
存在感
```

### 中間

```
楽器:

Lead:
前寄り

Keys:
中間

Percussion:
中間

設定:

Reverb Send: 20-35%
Pre-Delay: 20-30 ms
EQ: 適度
Volume: 中程度 (-12 dB)

理由:
メロディー
主要要素
```

### 後列

```
楽器:

Pad:
後ろ

FX:
最も後ろ

Ambience:
背景

設定:

Reverb Send: 40-60%
Pre-Delay: 10-20 ms (短め)
EQ: 暗い (-3 dB @ 6 kHz)
Volume: 小さい (-18 dB)

理由:
背景
空間演出
```

---

## Return Track戦略

**効率的な空間処理:**

### 標準4 Returns

```
Return A: Main Reverb (Hall)

用途:
メイン空間
全般

Reverb:
Type: Hall
Decay: 2.5 s
Size: 70%
Pre-Delay: 25 ms

EQ (Post):
High Pass: 300 Hz
High Cut: 10 kHz

推奨Send:
Snare: 30%
Lead: 25%
Vocal: 25%

Return B: Room Reverb

用途:
近い空間
Drums

Reverb:
Type: Room
Decay: 1.0 s
Size: 40%
Pre-Delay: 15 ms

EQ (Post):
High Pass: 400 Hz

推奨Send:
Drums: 15%

Return C: Delay (1/8)

用途:
リズミカル装飾

Simple Delay:
Time: 1/8
Feedback: 40%
Ping Pong: On

Filter:
High Pass: 500 Hz
Low Pass: 6 kHz

推奨Send:
Vocal: 20%
Lead: 25%

Return D: Delay (1/4 Dotted)

用途:
創造的

Filter Delay:
Time: 1/4 Dotted
Feedback: 50%

推奨Send:
Lead: 15%
FX: 30%
```

---

## Pre-Delay完全ガイド

**分離の鍵:**

### なぜ重要？

```
Pre-Delay なし (0 ms):

Dry音:
すぐReverb

結果:
密着
分離悪い
Dry埋もれる

Pre-Delay あり (20-30 ms):

Dry音:
明確

その後:
Reverb

結果:
分離良い
Dry音明確

推奨値:

Kick: 20 ms
Bass: 25 ms (少ないSend)
Vocal: 25-30 ms (最重要)
Lead: 25 ms
Snare: 20 ms
Pad: 10-15 ms

ルール:
前に出したい = 長い Pre-Delay
後ろに = 短い Pre-Delay
```

---

## Send量調整

**トラック別推奨:**

### 詳細設定

```
Kick:

Reverb: 0%
Delay: 0%

理由:
タイト維持
パワー

Bass:

Reverb: 0-5%
Delay: 0%

理由:
低域濁る
わずかに許容

Snare/Clap:

Reverb A: 25-35%
Reverb B: 10-15%
Delay C: 15-20%

理由:
広がり
Techno/House必須

Vocal:

Reverb A: 20-30%
Delay C: 15-20%
Delay D: 10% (装飾)

理由:
深み
自然

Lead:

Reverb A: 20-30%
Delay C: 25-35%

理由:
空間
動き

Pad:

Reverb A: 40-60%
Delay: 10-20%

理由:
壮大
後ろに

Hi-Hat:

Reverb B: 8-12% (Room)
Delay: 5-10%

理由:
わずかな空間

Percussion:

Reverb A: 25-35%
Delay: 20-30%

理由:
装飾
```

---

## 低域Reverb問題

**濁り防止:**

### 原因

```
問題:

Kick・Bass:
Reverbあり

結果:
低域濁る
パワーダウン

理由:

Reverb:
全周波数に残響

低域残響:
濁りの原因
```

### 解決法

```
方法1: Sendなし

Kick・Bass:
Send: 0%

最も確実

方法2: Return Track EQ

Return A-D全て:

EQ Eight挿入
High Pass: 300-500 Hz

効果:
低域Reverbなし
高域のみ残響

推奨:
High Pass 400 Hz

方法3: Pre-EQ

Kick・Bassトラック:

Send前:
High Pass 200 Hz (Send専用Chain)

理由:
低域送らない

推奨:
方法2 (Return EQ)
最も実用的
```

---

## Decay Time設定

**テンポ連動:**

### BPM別推奨

```
BPM 128 (Techno/House):

Hall:
Decay: 2.0-2.5 s

Room:
Decay: 1.0-1.2 s

理由:
ちょうど良い

BPM 140 (Dubstep):

Hall:
Decay: 1.5-2.0 s

理由:
速いテンポ
短めが良い

BPM 90 (Hip Hop):

Hall:
Decay: 2.5-3.5 s

理由:
遅いテンポ
長めOK

計算式:

Decay (s) = 60 / BPM × 4
目安

例:
BPM 128
60 / 128 × 4 = 1.875 s
→ 約 2.0 s
```

---

## Delay Timeテンポ連動

**リズミカル設定:**

### 音符単位

```
1/4 (4分音符):

BPM 128:
468.75 ms

用途:
ゆっくり
Dub

1/8 (8分音符):

BPM 128:
234.375 ms

用途:
標準
Techno/House

1/16 (16分音符):

BPM 128:
117.1875 ms

用途:
速い
Hi-Hat

1/4 Dotted (付点4分):

BPM 128:
703.125 ms

用途:
The Edge風
創造的

推奨:

Techno/House:
1/8 または 1/16

Ambient:
1/4 Dotted

Sync: On
必須
```

---

## Parallel Reverb

**より自然:**

### 概念

```
通常Reverb:

Individual Track:
Dry/Wet 30%

問題:
Dry減る

Parallel (Return):

Dry: 100% (そのまま)
Wet: Send量調整

メリット:
Dry維持
Reverb追加

推奨:
必ずReturn Track使用
Individual Reverb不使用
```

---

## よくある失敗

### 1. 低域にReverb

```
問題:
濁る

原因:
Kick・BassにSend

解決:

Kick・Bass:
Send: 0%

Return Track:
High Pass 400 Hz

効果:
劇的にクリア
```

### 2. Pre-Delayなし

```
問題:
Dry埋もれる

原因:
Pre-Delay: 0 ms

解決:

Pre-Delay:
20-30 ms設定

特にVocal:
必須

効果:
分離良い
```

### 3. Decay長すぎ

```
問題:
濁る
音が遠い

原因:
Decay: 4.0 s+

解決:

Decay:
1.5-2.5 s (標準)

テンポ連動:
BPM高い → 短く

効果:
クリア
```

### 4. Send過剰

```
問題:
全体が遠い

原因:
全トラック Send 50%+

解決:

推奨Send:
Vocal・Lead: 20-30%
Pad: 40-60%
Kick・Bass: 0%

ルール:
「少し足りない」
```

---

## Reverb種類使い分け

**Type選択:**

### 状況別

```
Hall:

特徴:
大きい空間
長い残響

用途:
Pad・Lead・Snare
壮大

Decay: 2.0-3.0 s

Room:

特徴:
小さい空間
自然

用途:
Drums全般
リアル

Decay: 0.8-1.5 s

Plate:

特徴:
金属板
明るい
密度高い

用途:
Snare・Vocal
クラシック

Decay: 1.5-2.5 s

Chamber:

特徴:
反射多い
ビンテージ

用途:
Vocal (60年代風)

Decay: 1.0-2.0 s
```

---

## Convolution Reverb

**リアルな空間:**

### IR (Impulse Response)

```
機能:

実際の空間:
録音

再現:
その空間のReverb

メリット:

リアル:
本物の空間

高品質:
自然

用途:

Orchestra Hall:
クラシック

Abbey Road:
ビンテージ

Club:
Techno/House

Ableton:

なし (標準)

推奨:
付属Reverbで十分

上級:
Altiverb, Space等
```

---

## 実践ワークフロー

**30分で完成:**

### Step-by-Step

```
0-10分: Return Track設定

1. Return A: Hall, Decay 2.5s
2. EQ High Pass 400 Hz
3. Return B: Room, Decay 1.0s
4. Return C: Delay 1/8
5. Return D: Delay 1/4D

10-20分: Send量調整

1. Kick・Bass: 0%
2. Snare: 30%
3. Vocal: 25%
4. Lead: 25%
5. Pad: 50%
6. Hi-Hat: 10%

20-25分: Pre-Delay

全Return:
Pre-Delay 20-30 ms設定

25-30分: 確認

1. 全体聴き直し
2. 奥行き確認
3. 低域濁りチェック
4. 微調整
```

---

## まとめ

### Depth & Space

```
□ Return Track活用 (最低2つ)
□ Pre-Delay 20-30 ms必須
□ 低域Reverb厳禁 (High Pass 400 Hz)
□ Decay Time テンポ連動
□ Send量控えめ
```

### 配置

```
最前列: Kick・Vocal (Send 0-10%)
中間: Lead・Keys (Send 20-35%)
後列: Pad・FX (Send 40-60%)
```

### 重要原則

```
□ Kick・Bass Send 0%
□ Return Track EQ必須
□ Pre-Delay で分離
□ Individual Reverb不使用
□ 「少し足りない」が正解
```

---

## 奥行きと空間の基本概念 — 深掘り

### 人間の空間知覚メカニズム

人間が音の距離感や空間を認識する仕組みは、視覚以上に複雑な処理によって成り立っています。音響心理学（Psychoacoustics）の研究によれば、人間は以下の手がかりを総合的に判断して音の位置を知覚します。

```
空間知覚の6つの手がかり:

1. 直接音と反射音の比率（D/R比）
   - Direct-to-Reverberant Ratio
   - 直接音が多い → 近い
   - 反射音が多い → 遠い
   - 最も重要な距離感の手がかり
   - ミキシングではSend量で制御

2. 初期反射（Early Reflections）
   - 直接音の後、約5-80 msの間に届く反射音
   - 空間のサイズと形状を知覚する手がかり
   - 壁・天井・床からの反射
   - Pre-Delayの設定がこれに影響

3. スペクトル変化（Spectral Cues）
   - 距離が遠くなると高域が減衰
   - 空気の吸収による自然現象
   - 6 kHz以上が最も影響を受ける
   - EQのハイカットで距離感を演出

4. 両耳間時間差（ITD: Interaural Time Difference）
   - 左右の耳に到達する時間の差
   - 水平方向の位置知覚
   - パンニングで制御
   - 最大約0.7 ms（頭の幅による）

5. 両耳間レベル差（ILD: Interaural Level Difference）
   - 左右の耳の音量差
   - 高域ほど差が大きい（頭部の遮蔽効果）
   - パンニングの補助的手がかり

6. 頭部伝達関数（HRTF: Head-Related Transfer Function）
   - 耳の形状による周波数フィルタリング
   - 上下・前後の位置知覚に影響
   - バイノーラル処理の基盤
   - 個人差が大きい
```

### 音楽制作における空間の3次元

```
ミキシングの3次元空間:

X軸（左右）: パンニング
- ステレオフィールドの配置
- 左 (-100%) ← Center (0%) → 右 (+100%)
- 幅を作る主要手段

Y軸（上下）: 周波数帯域
- 低音 = 下方向の知覚
- 高音 = 上方向の知覚
- EQで制御
- 人間の聴覚特性に基づく

Z軸（前後）: 奥行き ★この章のメインテーマ★
- Reverb/Delay量
- 音量レベル
- 高域の有無
- Pre-Delay設定
- トランジェントの明瞭さ

完成度の指標:
X軸のみ → 2Dミックス（アマチュア）
X+Y軸 → 2.5Dミックス（中級）
X+Y+Z軸 → 3Dミックス（プロフェッショナル）
```

### ドライ/ウェットの概念を深く理解する

```
Dry信号とWet信号の関係:

Dry（原音）:
- 音源そのまま
- 空間処理なし
- 明瞭、直接的
- 近い印象

Wet（加工音）:
- Reverb/Delayを通した音
- 空間情報を含む
- 拡散的、間接的
- 遠い印象

Dry/Wet比率の効果:

100% Dry / 0% Wet:
- 完全に近い
- 空間感なし
- 人工的に感じる場合あり

80% Dry / 20% Wet:
- 近い（最前列）
- わずかな空間感
- Kick, Vocal向き

60% Dry / 40% Wet:
- 中間距離
- 自然な空間感
- Lead, Keys向き

40% Dry / 60% Wet:
- やや遠い
- 明確な空間
- Pad, Strings向き

20% Dry / 80% Wet:
- 遠い
- 没入的
- Ambient, FX向き

重要:
Insert方式 → Dry/Wet比率で制御
Send/Return方式 → Send量で制御（推奨）
Send方式の方がDry信号を完全に保持できる
```

---

## 前後配置の高度なテクニック

### マスキングと距離感の関係

```
マスキング問題と空間配置:

問題の例:
VocalとLeadが同じ帯域で衝突
→ 両方が不明瞭

空間配置による解決:

方法1: 前後の分離
- Vocal: 前（Send少、Pre-Delay長）
- Lead: 中（Send多め、Pre-Delay中）
→ Vocalが前に出て明確

方法2: 左右 + 前後の分離
- Vocal: Center + 前
- Lead: やや左右に広げ + 中間
→ 3次元的に分離

方法3: 周波数 + 距離
- Vocal: 2-5 kHzブースト + 前
- Lead: 1-3 kHz中心 + やや後ろ
→ EQと空間の両方で分離

ルール:
重要な要素ほど前に配置
→ リスナーの注意を集める
```

### ダイナミックな前後移動

```
オートメーションによる距離変化:

テクニック1: ビルドアップでの接近
- イントロ: Vocal Send 40%（遠い）
- ビルドアップ: 40% → 10%（近づく）
- ドロップ: Send 10%（最前列）
→ Vocalがリスナーに向かってくる効果

テクニック2: ブレイクでの空間拡張
- ドロップ中: Pad Send 30%
- ブレイク: Pad Send 60% + Decay延長
→ 空間が広がる印象
→ ドロップへの対比を作る

テクニック3: トランジションでの距離操作
- セクション終わり: Lead Sendを増加
- 新セクション開始: Lead Sendを急減
→ シーンチェンジの明確化

パラメータの連動:
Send量 ↑ = 遠くなる
Volume ↓ = 遠くなる
High Cut ↓ = 遠くなる
Pre-Delay ↓ = 空間に埋もれる

全パラメータを同時にオートメーション
→ 説得力のある距離変化
```

### 音像のフォーカスと背景

```
フォーカス（焦点）の制御:

映画のカメラワークとの比較:
- 主役にフォーカス → 背景ボケ
- 背景にフォーカス → 前景ボケ
- ミキシングでも同じ概念

ミキシングでの実現:

フォーカスを当てたい要素:
- Dry多め（Send少）
- トランジェント明確
- 高域明るい
- 適度なコンプレッション

背景に回す要素:
- Wet多め（Send多）
- トランジェントソフト
- 高域カット
- ソフトなコンプレッション

実践例 — Techno:
- フォーカス: Kick, Hi-Hat（リズムセクション）
- 中間: Lead, Percussion
- 背景: Pad, Atmosphere, FX

実践例 — Vocal House:
- フォーカス: Vocal, Kick
- 中間: Bass, Keys
- 背景: Pad, Strings, FX
```

---

## リバーブによる空間表現 — 徹底解説

### リバーブの構成要素

```
リバーブの時間構造:

直接音（Direct Sound）:
- 0 ms
- 音源からの直接到達
- 最も大きい

初期反射（Early Reflections）:
- 5-80 ms
- 壁・天井からの最初の反射
- 空間のサイズを知覚
- Pre-Delayが制御

Late Reflections / Diffuse Field:
- 80 ms以降
- 多数の反射が混合
- 「残響」として知覚
- Decay Timeが制御

時間軸:
0 ms → Direct → Pre-Delay → Early Ref → Late/Diffuse → Silence

各パラメータの対応:
Direct → Dry Level
Pre-Delay → Early ReflectionsまでのGap
Early Ref → ER Level, Room Size
Late/Diffuse → Decay Time, Diffusion
```

### Reverbパラメータ詳細解説

```
Decay Time（残響時間）:
- 残響が60 dB減衰するまでの時間（RT60）
- 短い（0.5-1.0 s）→ 小さい部屋
- 中間（1.5-2.5 s）→ ホール
- 長い（3.0 s+）→ 大聖堂、洞窟
- テンポと連動させること

Size（サイズ）:
- 仮想空間の大きさ
- 小（20-40%）→ 部屋、ブース
- 中（40-70%）→ ホール、スタジオ
- 大（70-100%）→ 教会、アリーナ
- 大きすぎると不自然

Diffusion（拡散）:
- 反射音の密度
- 低い → 個別の反射が聴こえる（フラッター）
- 高い → 滑らかな残響
- 通常70-90%推奨
- Percussionは低めで独特の効果

Damping（ダンピング）:
- 高域の減衰速度
- 高い → 暗い残響（カーペット敷きの部屋）
- 低い → 明るい残響（タイル張りの浴室）
- ジャンルに応じて調整
- Techno: やや高め（暗い空間）
- Pop: やや低め（明るい空間）

Pre-Delay:
- 直接音から初期反射までの時間差
- 0 ms → 音源と空間が一体化
- 10-20 ms → 自然な距離感
- 30-50 ms → 明確な分離
- 前述のセクションで詳述

Early Reflection Level:
- 初期反射の音量
- 高い → 空間のサイズ明確
- 低い → よりスムーズ
- Drum向けは高め

Density（密度）:
- 反射音の密度（Diffusionと類似だが異なる）
- 高い → 豊かな響き
- 低い → スパースな響き
- テクスチャーに影響
```

### Reverbの種類と特性 — 完全版

```
1. Hall Reverb（ホール）:

特性:
- コンサートホールのシミュレーション
- 豊かで壮大な残響
- 初期反射が豊富
- 長いDecay（2.0-4.0 s）

最適な用途:
- オーケストラ的パッド
- 壮大なリードシンセ
- スネアのテール
- ボーカルの深み

パラメータ例:
- Decay: 2.5 s
- Size: 70%
- Diffusion: 80%
- Damping: 40%
- Pre-Delay: 25 ms

2. Room Reverb（ルーム）:

特性:
- 小〜中規模の部屋
- 自然で控えめ
- 短いDecay（0.3-1.5 s）
- 初期反射が明確

最適な用途:
- ドラムキット全般
- パーカッション
- リアルな楽器
- 「生」感を出したい要素

パラメータ例:
- Decay: 0.8 s
- Size: 35%
- Diffusion: 70%
- Damping: 50%
- Pre-Delay: 10 ms

3. Plate Reverb（プレート）:

特性:
- 金属板の共振をシミュレーション
- 密度が高くスムーズ
- 明るいトーン
- 中程度のDecay（1.0-3.0 s）

最適な用途:
- スネアドラム
- ボーカル
- クラシックな音楽制作
- 「80年代的」サウンド

パラメータ例:
- Decay: 1.8 s
- Damping: 30%
- Pre-Delay: 20 ms

4. Spring Reverb（スプリング）:

特性:
- バネの共振
- 独特の「ビョーン」という質感
- ローファイ的
- 短〜中程度のDecay

最適な用途:
- ギター（特にサーフロック）
- ダブ/レゲエ
- Lo-Fi Hip Hop
- 実験的なエフェクト

パラメータ例:
- Decay: 1.2 s
- Tension: 中
- Mix: 20-30%

5. Shimmer Reverb（シマー）:

特性:
- オクターブ上のピッチシフト + リバーブ
- 天空的、天使的な響き
- 非常に長いDecay
- 幻想的

最適な用途:
- アンビエントパッド
- チルアウト
- ブレイクダウンのFX
- 特殊演出

パラメータ例:
- Decay: 4.0 s+
- Pitch: +12 semitones
- Mix: 30-50%
- Shimmer Amount: 40-60%

6. Gated Reverb（ゲートリバーブ）:

特性:
- リバーブ残響を急にカット
- 80年代ドラムサウンド
- パンチのある空間感
- Phil Collins的サウンド

最適な用途:
- スネアドラム
- タム
- 特定のボーカルスタイル
- レトロ演出

パラメータ例:
- Decay: 1.5 s
- Gate Threshold: -20 dB
- Gate Release: 100 ms
```

---

## ディレイによる奥行き — 徹底解説

### ディレイの基本タイプ

```
1. Simple Delay（シンプルディレイ）:

構造:
入力 → 遅延 → 出力
      ↑ フィードバック ↓

パラメータ:
- Time: 遅延時間（msまたはBPM同期）
- Feedback: 繰り返し量（0-100%）
- Dry/Wet: 原音と遅延音の比率

特徴:
- 基本的で使いやすい
- Left/Right独立設定可能
- モノラルまたはステレオ

用途:
- ボーカルのダブリング（短いディレイ）
- リズミカルなリピート
- 基本的な奥行き

2. Ping Pong Delay:

構造:
入力 → Left → Right → Left → ...
      交互に左右にバウンス

パラメータ:
- Time: バウンス間隔
- Feedback: 繰り返し回数
- Width: ステレオ幅

特徴:
- 左右に交互にバウンス
- ステレオフィールドを活用
- 広がり感が強い

用途:
- リード/シンセ
- ボーカル
- パーカッション
- 空間を広げたい要素

3. Filter Delay:

構造:
入力 → フィルター → 遅延 → 出力
各チャンネル独立フィルター

パラメータ:
- Time: 各チャンネル個別設定
- Filter: HP/LP/BP各チャンネル
- Feedback: 各チャンネル個別

特徴:
- 周波数帯域ごとに異なるディレイ
- 非常にクリエイティブ
- 複雑な空間表現

用途:
- ダブエフェクト
- クリエイティブFX
- テクノのパーカッション

4. Tape Delay:

構造:
アナログテープレコーダーのシミュレーション

パラメータ:
- Time: テープ速度
- Feedback: テープループ回数
- Saturation: テープの飽和
- Wobble: ピッチの揺れ

特徴:
- 暖かみのある音
- 繰り返すごとに劣化（高域減衰）
- ピッチが微妙に揺れる
- ビンテージ感

用途:
- ダブ/レゲエ
- ボーカル（暖かみ）
- アナログ的な雰囲気
- Lo-Fi系

5. Grain Delay:

構造:
入力をグレイン（粒子）に分割して遅延

パラメータ:
- Delay Time: グレイン遅延
- Frequency: グレイン分割率
- Pitch: ピッチシフト量
- Random: ランダム化

特徴:
- グラニュラー合成的
- 実験的サウンド
- テクスチャー生成

用途:
- アンビエント
- 実験的テクノ
- サウンドデザイン
- FX素材作成
```

### ディレイによる距離感の制御

```
ディレイと距離感の関係:

短いディレイ（1-30 ms）:
- ダブリング効果
- 音が「太く」なる
- 距離感: 変化なし〜わずかに近い
- Haas効果（先行音効果）の活用

中間ディレイ（30-100 ms）:
- スラップバック
- 小さな空間の反射
- 距離感: 小さな部屋にいる印象
- ロカビリー、50年代風

長いディレイ（100-500 ms）:
- エコー
- リズミカルなリピート
- 距離感: 大きな空間
- ダブ、テクノでの定番

非常に長いディレイ（500 ms+）:
- 明確なリピート
- カノン的効果
- 距離感: 広大な空間、渓谷
- アンビエント

フィードバックと距離感:
- 低い（10-30%）→ 1-2回のリピートで消える → 近い
- 中間（30-60%）→ 数回リピート → 中間距離
- 高い（60-90%）→ 多数リピート → 遠い、広大
- 100%付近 → 自己発振 → 無限の空間（注意！）
```

### ディレイの周波数フィルタリング

```
ディレイにフィルターをかける理由:

問題:
フルレンジのディレイ音 → ミックスを濁す

解決:
ディレイ音にEQ/Filterをかける

推奨フィルター設定:

High Pass: 400-800 Hz
→ 低域のリピートを除去
→ ミックスの低域をクリアに保つ

Low Pass: 4-8 kHz
→ 高域のリピートを除去
→ 自然な距離感（空気吸収のシミュレーション）
→ リピートごとに暗くなる = よりリアル

帯域制限の例:

ボーカル用ディレイ:
HP: 500 Hz / LP: 6 kHz
→ ミッドレンジのみリピート
→ 原音の明瞭さを損なわない

シンセリード用ディレイ:
HP: 300 Hz / LP: 8 kHz
→ やや広いレンジ
→ キャラクターを保持

パーカッション用ディレイ:
HP: 800 Hz / LP: 4 kHz
→ 狭いレンジ
→ トランジェントを乱さない

テクニック — フィードバック内フィルター:
リピートごとにフィルターが適用される
→ 1回目: フルレンジに近い
→ 2回目: やや暗い
→ 3回目: さらに暗い
→ 自然な距離感の演出
```

---

## EQによる距離感の演出

### 周波数と距離の関係

音が遠くなるほど高域が減衰する現象は自然界で普遍的です。これは空気による高域吸収が原因であり、ミキシングでこの現象をシミュレートすることで説得力のある距離感を作り出せます。

```
空気吸収による高域減衰の法則:

距離と周波数減衰の関係:

1m (近い):
- ほぼ全帯域フラット
- 高域減衰なし
- 明瞭なトランジェント

10m (中間):
- 10 kHz以上: -2 dB程度
- わずかな高域減衰
- 自然な距離感

50m (遠い):
- 8 kHz以上: -6 dB程度
- 4 kHz以上: -3 dB程度
- 明らかな暗さ

100m+ (非常に遠い):
- 6 kHz以上: -10 dB以上
- 全体的に暗い
- 低域が支配的
```

### EQで距離感を制御するテクニック

```
テクニック1: ハイシェルフでの距離制御

近く配置したい要素:
- High Shelf: +2〜+3 dB @ 8 kHz
- Air Band: +1 dB @ 15 kHz
- 明るく輝くサウンド
- 例: Vocal, Kick, Snare

中間に配置したい要素:
- High Shelf: 0 dB（フラット）
- 自然な状態を維持
- 例: Lead, Keys, Guitar

遠くに配置したい要素:
- High Shelf: -3〜-6 dB @ 6 kHz
- Low Pass: 8-10 kHz
- 暗くソフトなサウンド
- 例: Pad, Ambience, FX

テクニック2: ローカットと距離感

近い要素:
- Low Cut: 最小限（30-60 Hz）
- 低域のパワー感を維持
- Kick: 30 Hz / Bass: 40 Hz

遠い要素:
- Low Cut: やや深め（80-150 Hz）
- 低域を整理してリバーブの濁りを防止
- Pad: 100 Hz / FX: 150 Hz

テクニック3: プレゼンスの制御

近い = プレゼンス強い:
- 2-5 kHz ブースト（+2〜+4 dB）
- 人間の耳が最も敏感な帯域
- 音が「前に出る」

遠い = プレゼンス弱い:
- 2-5 kHz カット（-2〜-4 dB）
- 存在感が薄れる
- 背景に溶け込む
```

### EQとリバーブの連携

```
EQ → Reverb（Pre-EQ）:

目的:
リバーブに送る前に不要な帯域をカット

設定例:
Send前のEQ Eight:
- High Pass: 200-400 Hz（低域を送らない）
- Low Pass: 10-12 kHz（超高域を送らない）
- 問題帯域のカット

効果:
- リバーブがクリーンに
- 低域の濁りを防止
- CPU負荷の軽減

Reverb → EQ（Post-EQ）:

目的:
リバーブの出力を整形

設定例:
Return TrackのEQ Eight:
- High Pass: 300-500 Hz（必須！）
- High Shelf: -3 dB @ 8 kHz（暗めに）
- Notch: 問題帯域をカット

効果:
- ミックスとの馴染みが良い
- 低域がクリア
- リバーブが「座る」

実践的なチェーン例:

Return A（Hall Reverb）:
EQ Eight (Pre) → Reverb → EQ Eight (Post) → Compressor

Pre-EQ設定:
HP: 300 Hz / LP: 10 kHz
→ リバーブに送る帯域を制限

Post-EQ設定:
HP: 400 Hz / HS: -2 dB @ 10 kHz
→ リバーブ出力を整形

Compressor設定:
Ratio: 2:1 / Attack: 30 ms / Release: 200 ms
→ リバーブのダイナミクスを制御
```

---

## パンニングと空間配置

### ステレオフィールドの基本

```
パンニングの基本ルール:

Center（0%）:
- Kick
- Bass
- Vocal（メイン）
- Snare
- 最も重要な要素

やや左右（15-30%）:
- Hi-Hat（やや右が一般的）
- Lead Synth
- Guitar
- 主要なメロディー要素

中程度の左右（30-60%）:
- Percussion
- Keys
- Backing Vocal
- パッドの一部

広い左右（60-100%）:
- Stereo Pad（L+R）
- Atmosphere
- FX
- ワイドなシンセ
- Overhead（ドラム）
```

### パンニングと奥行きの組み合わせ

```
3次元配置マトリクス:

          左         Center       右
前列:     ---        Kick         ---
          ---        Vocal        ---
          ---        Snare        ---

中列:     Keys(L)    Lead         Keys(R)
          Perc(L)    ---          Perc(R)

後列:     Pad(L)     ---          Pad(R)
          FX(L)      Ambience     FX(R)
          Atmos(L)   ---          Atmos(R)

ルール:
1. 重要な要素はCenter + 前
2. 前列の要素は狭いパンニング
3. 後列の要素は広いパンニング
4. 前 + 広い = 不自然（避ける）
5. 後ろ + 狭い = OK（焦点の背景）
```

### ステレオ幅と距離感

```
ステレオ幅が距離感に与える影響:

狭いステレオ（Mono〜30%）:
- 近く感じる
- フォーカスが明確
- 音源の位置が明確
- 例: 目の前の演奏者

広いステレオ（60-100%）:
- 遠く / 包み込む感じ
- 空間的
- 音源の位置が曖昧
- 例: コンサートホールの残響

テクニック — Utility（Ableton）:
- Width: 0%（Mono）→ 100%（Full Stereo）→ 200%（Super Wide）
- 前列要素: Width 80-100%
- 後列要素: Width 120-150%
- Mid/Side処理との併用で効果的

テクニック — Mid/Side EQ:
- Mid（中央）: 2-5 kHzブースト → 前に出る
- Side（左右）: 2-5 kHzカット → 背景に
- Mid: 低域維持 → パワー
- Side: 低域カット → クリーンさ
```

---

## 3Dオーディオとイマーシブサウンド

### バイノーラルオーディオの基礎

```
バイノーラルオーディオとは:

原理:
- 人間の両耳の聴こえ方の違いを再現
- HRTF（頭部伝達関数）を使用
- ヘッドフォンで3D空間を表現

仕組み:
1. 音源の位置を設定（上下左右前後）
2. HRTFフィルターを適用
3. 左右の耳に異なる信号を送る
4. 脳が立体的に知覚

ミキシングでの応用:

用途:
- ヘッドフォンリスニング向け
- VR/AR音楽コンテンツ
- イマーシブな体験型ライブ
- ポッドキャスト/ASMR

ツール:
- dearVR（プラグイン）
- Waves NX
- Dolby Atmos Renderer
- Apple Spatial Audio

注意点:
- スピーカー再生との互換性
- モノ互換性のチェック必須
- 全てのリスナーがヘッドフォンではない
```

### Dolby Atmosとサラウンドミキシング

```
Dolby Atmos概要:

チャンネル構成:
- 7.1.4（スピーカー: 7ch + サブ + 天井4ch）
- オブジェクトベースオーディオ
- 最大128オーディオトラック
- 最大7.1.2のベッド

音楽制作での応用:

ベッド（Bed）:
- 固定位置のチャンネル
- メインの楽器配置
- 7.1.2チャンネル

オブジェクト（Object）:
- 自由に動かせる音源
- 最大118個
- XYZ座標で配置
- オートメーション可能

DJ/電子音楽での活用:

Kick: Center（ベッド）
Bass: Center（ベッド）
Vocal: Center上方（オブジェクト）
Lead: 動的配置（オブジェクト）→ リスナーの周りを回る
Pad: サラウンド全体（ベッド）
FX: 天井含む全方位（オブジェクト）
Percussion: 左右+やや後方（オブジェクト）

Apple Spatial Audio:
- AirPodsでのAtmos再生
- ヘッドトラッキング対応
- Apple Music配信
- Logic ProでのAtmosミキシング

実用的なアプローチ:
1. まずステレオミックスを完成させる
2. Atmosバージョンを別途作成
3. ベッドに主要要素を配置
4. オブジェクトで動的要素を追加
5. 高さ方向のFXを配置
```

---

## ジャンル別空間設計

### Techno

```
Techno の空間設計:

コンセプト:
- クラブの空間を再現
- ダークでディープ
- 没入感重視
- 機能的（踊れること）

空間構造:
前列: Kick（超前面、Dry、パワフル）
      Hi-Hat（前、タイト）
中列: リードシンセ（Reverb適度）
      パーカッション（Room Reverb）
後列: パッド（大きなHall）
      アトモスフィア（非常にWet）

Reverb設定:
- メインHall: Decay 2.0-2.5s, Dark
- Room: Decay 0.8-1.0s
- Damping: 高め（暗い空間）
- 低域Reverb: 厳禁

Delay設定:
- 1/8 Ping Pong: Hi-Hat, Percussion
- 1/16 Mono: リードのアクセント
- Feedback: 30-50%
- フィルター: 暗め（LP 4-6 kHz）

特徴的テクニック:
- Send Automationでブレイク時に空間拡大
- Delayフィードバック増加でテンション
- Reverb Freeze（特殊エフェクト）
- ダブテクノ: 長いDelay + 高Feedback
```

### House / Deep House

```
House / Deep House の空間設計:

コンセプト:
- 暖かく心地よい空間
- ソウルフル
- 適度な開放感
- Vocalの存在感

空間構造:
前列: Kick（前、パンチ）
      Vocal（最前面、明瞭）
      Bass（前、グルーヴ）
中列: Keys/Piano（Plate Reverb）
      Guitar（Room + Delay）
後列: Pad（Hall, 壮大）
      Strings（Hall + Delay）

Reverb設定:
- Plate: Vocal用、Decay 1.5-2.0s
- Hall: パッド用、Decay 2.5-3.0s
- Room: ドラム用、Decay 1.0s
- Damping: 低め（明るい空間）

Delay設定:
- 1/4 Dotted: Vocal装飾
- 1/8 Ping Pong: Keys
- Tape Delay: ギター（暖かみ）
- Feedback: 30-40%

特徴的テクニック:
- Vocal Throw: 特定フレーズだけDelayを強くSend
- Piano Space: Plate + 短いDelay
- Bass Space: ほぼDry（わずかなRoom）
- コーラスパート: 広いStereo + Hall
```

### Trance / Progressive

```
Trance / Progressive の空間設計:

コンセプト:
- 壮大で広大
- エモーショナル
- 展開が重要
- ビルドアップ→ブレイク→ドロップ

空間構造:
前列: Kick（前、パワー）
      Lead（ドロップ時は前面）
中列: Arpeggios（Delay重要）
      Plucks（Ping Pong Delay）
後列: Super Saw Pad（巨大なHall）
      Atmosphere（Shimmer Reverb）
      FX Sweeps（Delay + Reverb）

Reverb設定:
- Hall: 壮大、Decay 3.0-4.0s
- Shimmer: ブレイク用、Decay 5.0s+
- Plate: リード用、Decay 2.0s
- Size: 大きめ（70-90%）

Delay設定:
- 1/8 Ping Pong: アルペジオ（必須！）
- 1/4 Dotted: リード装飾
- Multi-Tap: FX
- Feedback: 40-60%

特徴的テクニック:
- ビルドアップ: Reverb Decay延長 + Send増加
- ブレイク: 全要素をWetに → 巨大空間
- ドロップ: 急にDryに → インパクト
- Reverse Reverb: ボーカル/リードの導入
```

### Ambient / Downtempo

```
Ambient / Downtempo の空間設計:

コンセプト:
- 空間そのものが主役
- テクスチャー重視
- 非常にWetなミックス
- 瞑想的

空間構造:
（前後の概念が曖昧）
全体: 広大な空間に溶け合う
焦点: わずかに前に出る要素
背景: ほぼ全ての要素

Reverb設定:
- Hall: Decay 4.0-8.0s（非常に長い）
- Shimmer: 幻想的なテクスチャー
- Convolution: リアルな空間（教会など）
- Size: 最大（90-100%）
- Damping: 低い（明るく開放的）

Delay設定:
- 1/4 Dotted: メインテクスチャー
- Grain Delay: テクスチャー生成
- Multi-Tap: 複雑な空間
- Feedback: 60-80%（自己発振手前）

特徴的テクニック:
- Reverb Freeze: 無限残響
- Delay自己発振: テクスチャー生成
- Granular Processing: 空間の粒子化
- Layer Multiple Reverbs: 複数空間の重ね合わせ
- Pitch Shifted Delay: オクターブ違いのリピート
```

### Drum & Bass

```
Drum & Bass の空間設計:

コンセプト:
- 高速（170-180 BPM）
- ドラムの精度が命
- Bassの存在感
- 空間は控えめかつ効果的

空間構造:
前列: Kick（超前面、Dry）
      Snare（前、タイトRoom）
      Bass（前面、Dry）
中列: パッド（控えめなHall）
      リード（Delay中心）
後列: Atmosphere（Hall）
      FX（Delay + Reverb）

Reverb設定:
- 速いBPMのためDecayは短め
- Hall: Decay 1.2-1.8s
- Room: Decay 0.5-0.8s
- Pre-Delay: 必須（分離のため）
- Damping: 中程度

Delay設定:
- 1/16: Hi-Hat装飾
- 1/8: リード
- Feedback: 20-40%（短め）
- フィルター: 重要（混濁防止）

重要ポイント:
- BPMが速いので空間はタイト
- 低域Reverbは絶対禁止
- Snare Reverbが空間の主役
- Breakbeatsの処理が鍵
```

---

## センド/リターン構成の高度な設計

### 6 Return構成（拡張版）

```
基本4 Returnを超えた拡張構成:

Return A: Main Hall Reverb
- Type: Hall
- Decay: 2.5 s
- Pre-Delay: 25 ms
- Post-EQ: HP 400 Hz / LP 10 kHz
- 用途: メインの空間、Snare/Lead/Vocal

Return B: Tight Room
- Type: Room
- Decay: 0.8 s
- Pre-Delay: 10 ms
- Post-EQ: HP 500 Hz
- 用途: ドラム全般、タイトな空間

Return C: Short Delay (1/8)
- Type: Ping Pong Delay
- Time: 1/8 Sync
- Feedback: 35%
- Filter: HP 500 Hz / LP 6 kHz
- 用途: リズミカルな装飾

Return D: Long Delay (1/4 Dotted)
- Type: Filter Delay
- Time: 1/4 Dotted
- Feedback: 45%
- Filter: HP 400 Hz / LP 5 kHz
- 用途: クリエイティブ、メロディック

Return E: Special Plate ★追加★
- Type: Plate Reverb
- Decay: 1.5 s
- Pre-Delay: 15 ms
- Post-EQ: HP 600 Hz
- 用途: Snare専用、ボーカル専用

Return F: Creative FX ★追加★
- Type: Shimmer / Grain Delay / 特殊
- 曲ごとに変更
- 用途: ブレイクダウン、特殊演出
```

### Return Trackのルーティング

```
高度なルーティング技法:

テクニック1: Returnの直列接続
Return A (Reverb) → Return C (Delay)
→ リバーブ音にディレイをかける
→ 「リバーブのエコー」効果
→ 壮大な空間表現

設定方法:
1. Return Aの出力をReturn Cにも送る
2. Send量: 10-20%（控えめに）
3. フィードバック注意（ループ防止）

テクニック2: グループ経由のReturn
Drum Group → Return B (Room)のみ
Synth Group → Return A (Hall) + Return C (Delay)
Vocal Group → Return E (Plate) + Return D (Long Delay)

メリット:
- グループ単位で空間を管理
- 各グループの空間特性を独立制御
- セクション変更時に一括調整可能

テクニック3: Sidechain Compressor on Return
Return A (Hall) にSidechain Compressor:
- Key: Kick
- Ratio: 4:1
- Attack: 1 ms
- Release: 150 ms

効果:
- Kickが鳴るたびにReverbがダック
- 低域のクリアさ維持
- パンプ効果（クラブサウンド）
- Kickのパワーを損なわない
```

---

## パラレルプロセッシングの深掘り

### パラレルリバーブの高度な手法

```
パラレルリバーブの発展形:

基本（おさらい）:
- Dry: 100%（元トラック）
- Wet: Send量で調整（Return Track）
- Dry信号を完全に保持

発展1: Compressed Reverb
Return Track内:
Reverb → Compressor → EQ

Compressor設定:
- Ratio: 4:1
- Attack: 30 ms
- Release: 200 ms
- Threshold: -20 dB

効果:
- リバーブの初期反射を強調
- テールを均一に
- より存在感のある空間

発展2: Saturated Reverb
Return Track内:
Reverb → Saturator → EQ

Saturator設定:
- Drive: 3-6 dB
- Type: Analog Clip / Soft Sine
- Dry/Wet: 100%

効果:
- 暖かみのあるリバーブ
- ビンテージ感
- 存在感の向上

発展3: Gated Parallel Reverb
Return Track内:
Reverb → Gate → EQ

Gate設定:
- Threshold: -30 dB
- Attack: 0.1 ms
- Hold: 100 ms
- Release: 50 ms

効果:
- Gated Reverbの音
- パンチのある空間
- 80年代ドラムサウンド
```

### パラレルディレイのテクニック

```
パラレルディレイの応用:

テクニック1: Ducked Delay
Return Track内:
Delay → Compressor (Sidechain: 元トラック)

Compressor設定:
- Sidechain: On
- Key Input: 元のDryトラック
- Ratio: ∞:1
- Attack: 0.1 ms
- Release: 100 ms

効果:
- Dry音が鳴っている間はDelay無音
- Dry音が止まるとDelayが聞こえる
- 非常にクリーンなミックス
- ボーカルに最適

テクニック2: Modulated Delay
Return Track内:
Delay → Chorus/Flanger → EQ

効果:
- ディレイ音が揺れる
- より有機的な空間
- シンセパッドに効果的

テクニック3: Pitch Shifted Delay
Return Track内:
Pitch Shifter (+7 semitones) → Delay → EQ

効果:
- 5度上のディレイ音
- ハーモニー効果
- リード/ボーカルに魔法のような効果
```

---

## 空間系エフェクトチェーンの設計

### 標準エフェクトチェーンの順序

```
Insert Chain（トラック内）:

推奨順序:
1. EQ (Pre) — 不要帯域カット
2. Compressor — ダイナミクス制御
3. EQ (Post) — トーンシェイプ
4. Saturation — 色付け（オプション）
5. ※空間系はInsertに入れない（Return使用）

Return Chain（空間処理）:

推奨順序:
1. EQ (Pre) — 入力整形
2. Reverb または Delay — メイン空間処理
3. EQ (Post) — 出力整形
4. Compressor — ダイナミクス均一化
5. Utility — 幅/レベル調整

禁止事項:
- Insert Reverbの使用（特別な場合を除く）
- Reverb前のCompressor（不自然な結果）
- フィルター後のSaturation（予測不能な倍音）
```

### クリエイティブな空間チェーン

```
チェーン1: "Cathedral"（大聖堂）
EQ (HP 300Hz) → Hall Reverb (Decay 4s) →
Shimmer (+12st, 30%) → EQ (LP 8kHz) → Utility (-3dB)

用途: ブレイクダウン、パッド
効果: 巨大な教会のような残響

チェーン2: "Dub Space"（ダブスペース）
EQ (HP 400Hz) → Tape Delay (1/4, FB 60%) →
Filter (LP Sweep) → Spring Reverb (Decay 1.5s)

用途: パーカッション、ボーカル
効果: クラシックなダブ/レゲエ空間

チェーン3: "Frozen Texture"（凍結テクスチャー）
Grain Delay (Pitch +5st, Random 40%) →
Hall Reverb (Decay 6s, Freeze可) →
EQ (HP 500Hz, LP 6kHz) → Compressor

用途: アンビエント、トランジション
効果: 音の粒子が凍結したような空間

チェーン4: "Radio Space"（ラジオスペース）
EQ (BP 400Hz-4kHz) → Room Reverb (Decay 0.5s) →
Saturation (Warm) → EQ (LP 4kHz)

用途: ボーカル、Lo-Fi効果
効果: 古いラジオのような親密な空間

チェーン5: "Infinite Corridor"（無限回廊）
Ping Pong Delay (1/8, FB 70%) →
Plate Reverb (Decay 2s) →
Chorus (Rate 0.5Hz) → EQ (HP 300Hz)

用途: リード、アルペジオ
効果: 無限に続く廊下のような空間
```

---

## 実践テクニック集

### テクニック1: Reverb Throw

```
Reverb Throw（スロー）:

概念:
特定のフレーズやワードだけにリバーブを強くかける

方法（Ableton Live）:
1. Return TrackのSend量をオートメーション
2. 通常: Send 0%
3. 強調したい瞬間: Send 80-100%
4. その後すぐに0%に戻す

用途:
- ボーカルの最後のワード
- スネアのフィルイン
- ブレイク前の最後のビート
- セクション終わりの余韻

設定のコツ:
- Send量: 80-100%（一瞬だけ）
- 持続時間: 1/4〜1小節
- Decay: 通常より長め
- フェードアウト: 自然に消えるように

効果:
- ドラマチックな演出
- セクション間のつなぎ
- リスナーの注意を引く
```

### テクニック2: Reverse Reverb

```
Reverse Reverb:

概念:
リバーブ音を逆再生して「フワッ」と近づく効果

制作方法:
1. 音素材を選択（Vocal、Snare等）
2. 音素材を反転（Reverse）
3. Reverb 100% Wetで処理
4. 書き出し（Flatten/Freeze）
5. 再度反転（Reverse）
6. 元の音の直前に配置

パラメータ:
- Reverb: Hall, Decay 2-3s
- Dry/Wet: 100% Wet
- Pre-Delay: 0 ms
- Size: 70-80%

用途:
- ボーカル導入前のエフェクト
- ドロップ前のビルドアップ
- スネアの「スワッ」効果
- シンセの導入

注意:
- 元の音と正確にタイミングを合わせる
- Reverse Reverbのテールと元音の頭が重なるように
- EQで低域をカットすること
```

### テクニック3: Sidechain Space

```
Sidechain Reverbの活用:

概念:
Kickに同期してReverb/Delayをダッキング

設定方法:
1. Return Track（Reverb）にCompressorを追加
2. Sidechain Input: Kickトラック
3. Ratio: 4:1〜∞:1
4. Attack: 0.5-1 ms
5. Release: 100-200 ms（BPMに合わせる）

効果:
- Kick部分のReverbがダック
- 低域がクリア
- グルーヴ感が向上
- パンプ効果

BPM別Release目安:
BPM 128: Release 150 ms
BPM 140: Release 130 ms
BPM 90: Release 200 ms

応用:
- Kickだけでなく、Bassでもsidechain可
- Delay Returnにも適用可能
- グループバス全体に適用も効果的
```

### テクニック4: Space Automation

```
空間パラメータのオートメーション戦略:

セクション別空間設計:

イントロ:
- Reverb Send: 中程度（40%）
- Decay: やや長め
- 空間: 広い
- 目的: 雰囲気作り

ビルドアップ:
- Reverb Send: 徐々に増加（40%→70%）
- Decay: 延長
- Delay Feedback: 増加
- 目的: テンション構築

ドロップ:
- Reverb Send: 急減（10-20%）
- Decay: 短め
- 空間: タイト
- 目的: インパクト、パワー

ブレイク:
- Reverb Send: 最大（60-80%）
- Decay: 最長
- Shimmer: 追加
- 目的: 開放感、エモーション

アウトロ:
- Reverb Send: 徐々に増加
- Decay: 延長
- Volume: フェードアウト
- 目的: 余韻、終結感
```

---

## Depth & Space 完全チェックリスト

### 設定前チェック

```
□ Return Trackを最低4つ作成したか
  - Hall Reverb
  - Room Reverb
  - Short Delay
  - Long Delay

□ 全Return TrackにPost-EQ（HP 400 Hz）を挿入したか

□ テンポに合ったDecay Timeを計算したか
  - 計算式: 60 / BPM × 4 = 基準Decay

□ Pre-Delayを全Reverbに設定したか
  - 推奨: 20-30 ms
```

### ミキシング中チェック

```
□ Kick/BassのReverb Sendが0%か
□ VocalのPre-Delayが25-30 msか
□ Padが後列に配置されているか（Send 40%+）
□ LeadにDelay Sendがあるか
□ Snare Reverbが適切か（25-35%）
□ Hi-HatはRoom Reverbのみか（8-12%）
□ 全体的にSendが控えめか（「少し足りない」が正解）
```

### 最終確認チェック

```
□ モノラルで確認 — 空間が崩壊していないか
□ 低音量で確認 — 奥行きが維持されているか
□ ヘッドフォンで確認 — ステレオ空間が自然か
□ リファレンストラックと比較 — 奥行き感が近いか
□ 低域の濁りがないか — Kick/Bassがクリアか
□ 前列要素が埋もれていないか — Vocal/Kickが明確か
□ セクション間の空間変化があるか — 単調でないか
□ Delay Timingがテンポに同期しているか
```

### トラブルシューティング

```
問題: ミックスが濁る
→ Return TrackのHP EQを確認（400 Hz以上）
→ Send量を全体的に10%下げる
→ Decay Timeを0.5s短くする

問題: Vocalが埋もれる
→ Pre-Delayを30 msに上げる
→ Send量を20%以下にする
→ Return TrackのEQで2-5 kHzをカット

問題: 空間感がない
→ Send量を全体的に10%上げる
→ 後列要素（Pad/FX）のSendを増やす
→ ステレオ幅を確認

問題: 位相の問題
→ モノラルチェックを実行
→ Reverb Sizeを調整
→ Pre-Delayを変更してフェーズ衝突を回避

問題: CPUが重い
→ Convolution Reverbをアルゴリズミックに変更
→ Freeze/Flattenで負荷軽減
→ 不要なReturn Trackをミュート
```

---

**次は:** [Automation](./automation.md) - 動的ミックスで時間軸の変化を作る
