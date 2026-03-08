# Clip編集

Clipを自在に操る。カット、コピー、ペーストから高度なテクニックまで、編集スキルを完全マスターします。

## この章で学ぶこと

- Clip基本操作(選択、移動、複製)
- カット、コピー、ペースト
- ループ設定完全理解
- トランスポーズ(音程変更)
- リバース(逆再生)
- Fade In/Out
- Consolidate(統合)
- Split/Join(分割・連結)
- オーディオクリップ詳細編集
- MIDIクリップ高度編集
- クリップエンベロープ
- マルチクリップ編集
- クリエイティブ操作

---

## なぜClip編集が重要なのか

**制作スピードを10倍に:**

```
初心者:
1つずつ作る
同じパターンを何度も入力

プロ:
コピー&ペースト
編集テクニック駆使

差:

8小節のドラムパターン:
初心者: 30分
プロ: 3分

理由:
編集スキル

このスキル習得:
制作時間 1/10
= 10倍速く曲を作れる
```

---

## Clip基本操作

**選択・移動・複製:**

### Clipの選択

```
1つのClip:
クリック

複数Clip:
Shift+クリック
または
ドラッグで範囲選択

全Clip:
Cmd+A

トラック内全Clip:
トラック名クリック → Cmd+A

Scene内全Clip:
Scene番号クリック

Arrangement View:
同様の操作
```

### Clipの移動

```
ドラッグ:
クリップをドラッグ
→ 別の場所へ移動

矢印キー:
選択して ←→
→ グリッド単位で移動

Cmd+矢印:
小節単位で移動

Session View:
縦横自由に移動

Arrangement View:
時間軸上を移動

スナップ:

Grid: On (デフォルト)
→ グリッドに吸着

Grid: Off (Cmd押しながら)
→ 自由配置
```

### Clipの複製

```
Cmd+D (Duplicate):
最も重要なショートカット

使い方:

1. Clip選択

2. Cmd+D

3. すぐ右に複製される

連続複製:

Cmd+D × 3回
→ 4つのClipができる

応用:

4小節パターン作成
→ Cmd+D × 7回
→ 32小節完成

Session View:
下に複製

Arrangement View:
右に複製
```

---

## カット・コピー・ペースト

**基本中の基本:**

### カット (Cmd+X)

```
機能:
選択Clipを切り取り
クリップボードに保存

用途:

移動:
カット → 別の場所にペースト

削除しつつ保管:
後で使うかも

操作:

1. Clip選択
2. Cmd+X
3. 元のClipが消える
4. クリップボードに保存
```

### コピー (Cmd+C)

```
機能:
選択Clipをコピー
クリップボードに保存
元は残る

用途:

複製:
同じパターンを別の場所へ

バックアップ:
編集前にコピー

操作:

1. Clip選択
2. Cmd+C
3. 元のClipは残る
4. クリップボードに保存
```

### ペースト (Cmd+V)

```
機能:
クリップボードの内容を貼り付け

用途:

配置:
カット/コピーしたClipを配置

操作:

1. 配置先をクリック
2. Cmd+V
3. Clipが貼り付けられる

複数回:
Cmd+V 連打
→ 何度でも貼り付け可能

Session View:
選択したSlotに貼り付け

Arrangement View:
再生ヘッド位置に貼り付け
```

---

## ループ設定

**完璧なループを作る:**

### Loop Brace(ループブレース)

```
表示:

Clip View > Sample/Notes:

Loop: ☑ On

┌────────────────────────────┐
│ [========== Loop ==========] │ ← 黄色いバー
│ Start: 1.1.1               │
│ End: 5.1.1                 │
│ Length: 4 Bars             │
└────────────────────────────┘

操作:

左端ドラッグ:
開始位置変更

右端ドラッグ:
終了位置変更

中央ドラッグ:
範囲ごと移動

ダブルクリック:
全体を選択

数値入力:

Start: 1.1.1 (1小節1拍1/16)
End: 5.1.1 (5小節1拍1/16)
→ 4小節ループ
```

### ループの長さ

```
一般的な長さ:

4 Bars (4小節):
最も一般的
Techno, House

8 Bars (8小節):
展開付き

16 Bars (16小節):
長い展開

1 Bar (1小節):
ドラムパターン

2 Bars (2小節):
シンプルなフレーズ

細かい設定:

1/4 Note (4分音符):
短いフレーズ

1 Beat (1拍):
Hi-Hatパターン

自由な長さ:
3.2.1 - 7.4.3 等
不規則なループも可能
```

### Loop vs Clip Length

```
Loop:
再生される範囲

Clip Length:
Clipの実際の長さ

例:

Clip Length: 8 Bars
Loop: 1.1.1 - 5.1.1 (4 Bars)

→ 最初の4小節だけループ再生

用途:

イントロ部分をスキップ:
Loopを途中から開始

複数バリエーション:
Loopを切り替え
```

---

## トランスポーズ(音程変更)

**半音単位で移調:**

### Transpose設定

```
Clip View > Transpose:

┌──────────────┐
│ Transpose    │
│ 0 st         │ ← 半音単位
└──────────────┘

範囲:
-48 〜 +48 (±4オクターブ)

使い方:

+1 st: 半音上げ
-1 st: 半音下げ
+12 st: 1オクターブ上げ
-12 st: 1オクターブ下げ

Audio Clip:
Warp On で音程変更可能
音質やや劣化

MIDI Clip:
完全に音程変更
劣化なし
```

### Detune(微調整)

```
Clip View > Detune:

┌──────────────┐
│ Detune       │
│ 0 ct         │ ← セント単位
└──────────────┘

範囲:
-50 〜 +50 cent

1 semitone = 100 cent

用途:

微妙なデチューン:
+7 ct
わずかに高く

アナログ感:
-3 ct
不安定な感じ

2つのシンセを重ねる:
Synth 1: 0 ct
Synth 2: +10 ct
→ 厚みが出る
```

---

## リバース(逆再生)

**逆再生エフェクト:**

### Reverse設定

```
Clip View > Sample > Reverse:

☑ Reverse

効果:
波形が左右反転
逆再生される

用途:

リバースシンバル:
通常のシンバル逆再生
→ ビルドアップ効果

リバースボーカル:
不思議な効果

リバーススネア:
EDMでよく使う

操作:

1. Audio Clip選択
2. Clip View > Sample
3. Reverse: ☑
4. 再生確認

元に戻す:
Reverse: ☐
```

### リバーステクニック

```
シンバルリバース:

1. シンバルサンプル配置

2. Reverse: ☑

3. ドロップの直前に配置

4. 盛り上がり効果

ボーカルチョップリバース:

1. ボーカルの一部を切り出し

2. Reverse: ☑

3. 元のボーカルの前に配置

4. Call & Response効果

スネアリバース:

1. スネアClip複製

2. Reverse: ☑

3. 通常スネアの直前

4. 独特のグルーヴ
```

---

## Fade In/Out

**クリック音防止:**

### Fade設定

```
Clip View > Sample > Fade:

Fade In: 0.00 ms
Fade Out: 0.00 ms

推奨値:

短いサンプル (Kick, Snare):
Fade In: 2-5 ms
Fade Out: 5-10 ms

長いサンプル (ボーカル, Pad):
Fade In: 10-50 ms
Fade Out: 50-200 ms

クリック音防止:
Fadeなし → プツッ
Fade付き → 滑らか

自動Fade:

Preferences > Record Warp Launch:
Create Fades on Clip Edges: ☑

新規Clipに自動適用:
手動設定不要
```

### Arrangement ViewでのFade

```
マニュアルFade:

1. Clip選択

2. 左上/右上にマウス

3. 小さい三角マーク表示

4. ドラッグ

5. Fade曲線作成

便利:
視覚的に調整
長さ自由

Cmd+Opt+F:
自動Fade作成
デフォルト長さ適用
```

---

## Consolidate(統合)

**複数Clipを1つに:**

### Consolidate機能

```
Cmd+J (Consolidate):

複数Clip → 1つのClipに統合

用途:

複数Clipをまとめる:
8個の Kick Clip
→ 1つの長いKick Clipに

エフェクト確定:
Reverb等を焼き込み

CPU負荷軽減:
重いエフェクトを統合

操作:

1. 複数Clip選択
   (Shift+クリック)

2. Cmd+J

3. 1つのClipに統合
   新しいファイル作成

Arrangement View:
よく使う
複雑な編集を確定

Session View:
あまり使わない
```

### Freeze & Flatten

```
似た機能:

Freeze Track:
トラック全体を一時的にAudio化
後で編集可能

Flatten:
Freezeを確定
Audio Clipに変換

Consolidate:
選択範囲のみ
すぐ確定

使い分け:

部分的統合: Consolidate
トラック全体: Freeze → Flatten
```

---

## Split(分割)

**Clipを切る:**

### Split操作

```
Cmd+E (Split):

再生ヘッド位置でClipを分割

操作:

1. 再生ヘッドを分割位置に

2. Clip選択

3. Cmd+E

4. 2つのClipに分割

応用:

不要部分削除:

1. 必要な部分の前後でSplit

2. 不要部分を削除

3. 必要部分のみ残る

複数箇所分割:

1. 分割位置1でCmd+E

2. 再生ヘッド移動

3. 分割位置2でCmd+E

4. 何度でも可能
```

### Grid Quantization

```
Split時の吸着:

Grid: On
→ 小節・拍に吸着

Grid: Off (Cmd押しながら)
→ 自由な位置で分割

Grid設定:

Adaptive (推奨):
自動調整

1/16:
16分音符単位

1/4:
4分音符単位

Off:
グリッドなし
```

---

## 実践: 8小節ループを作る

**総合演習:**

### Step 1: Kickパターン (10分)

```
1. Audio Track作成

2. Kick サンプル配置
   Browser > Drums > Kicks

3. Clip View:
   Loop: 1 Bar

4. Cmd+D × 7回
   → 8小節

5. 確認:
   Space で再生
```

### Step 2: ベースライン (15分)

```
1. MIDI Track作成

2. Wavetable 挿入

3. MIDI Clip作成:
   4 Bars

4. ベースライン入力:
   A2, C3, F2, G2

5. Cmd+D
   → 8小節に拡張

6. 後半を編集:
   +12 st (1オクターブ上げ)
```

### Step 3: ハイハット (10分)

```
1. Audio Track作成

2. Hi-Hat サンプル × 2種類

3. パターン作成:
   1/8音符で配置

4. Velocity調整:
   強弱つける

5. Cmd+D で展開
```

### Step 4: 統合 (5分)

```
1. 全Clip選択
   (Arrangement Viewの場合)

2. Cmd+J (Consolidate)

3. 1つのループClipに

4. テンプレート保存:
   次回から使える
```

---

## よくある質問

### Q1: Cmd+Dが効かない

**A:** Clipが選択されているか確認

```
チェック:

Clip選択:
クリックして選択

Session View:
Clip Slotが選択されている

Arrangement View:
Clipが選択されている

複数選択:
Shift+クリック

再試行:
選択 → Cmd+D
```

### Q2: Consolidateとの違いは？

**A:** 用途が違う

```
Cmd+D (Duplicate):
コピー
元のClipも残る

Cmd+J (Consolidate):
統合
新しいClipに置き換わる

使い分け:

展開: Cmd+D
確定: Cmd+J
```

### Q3: Loopがズレる

**A:** Loop長を確認

```
原因:

Loop長が不正確:
4.1.1 になっている
→ 5.1.1 に修正

Start位置がズレ:
1.1.2 から開始
→ 1.1.1 に修正

解決:

数値で指定:
Start: 1.1.1
End: 5.1.1
Length: 4 Bars

確認:
再生して耳で確認
```

---

## オーディオクリップの詳細編集

**波形レベルでの精密操作:**

### Sample Editor(サンプルエディタ)

```
Clip View > Sample:

波形表示:
┌─────────────────────────────┐
│   ╱\    ╱\    ╱\    ╱\     │
│  ╱  \  ╱  \  ╱  \  ╱  \    │
│ ╱    \/    \/    \/    \   │
└─────────────────────────────┘

主要パラメータ:

Start/End Marker:
再生範囲の設定
ドラッグで調整

Loop Position:
ループ開始/終了点
黄色いブレース

Clip Gain:
クリップの音量
0dB基準で±35dB

Sample Offset:
サンプル開始位置
タイミング微調整
```

### Start/End Markerの活用

```
用途:

サンプルのトリミング:

1. 不要な無音部分カット

2. Start Markerを右へ

3. End Markerを左へ

4. 必要部分だけ再生

アタック調整:

Kickのアタック強調:
Start → わずかに右へ
→ アタック前の無音除去

Padの柔らかさ:
Start → アタック部分カット
→ 滑らかな立ち上がり

実践例:

Snare Sample:
元: [無音][アタック][ボディ][リリース][無音]

調整後:
Start → アタック直前
End → リリース終了直後
→ タイトなサウンド
```

### Warp Markerによる精密タイミング調整

```
Warp Mode: Complex Pro

Warp Marker配置:

1. 波形のトランジェント部分をクリック

2. 自動でWarp Marker作成

3. ドラッグして位置調整

4. タイミング完璧に

応用テクニック:

ドラムループの修正:

1. Kickの位置にWarp Marker

2. Snareの位置にWarp Marker

3. Hi-Hatの位置にWarp Marker

4. 各マーカー調整
   → グリッドに完全一致

ボーカルの修正:

1. 単語の頭にMarker

2. タイミングを微調整

3. 自然な歌唱に

サンプル例:

Before:
Kick: 1.1.1
Snare: 1.2.1 (少し遅れ)

After:
Kick: 1.1.1 (そのまま)
Snare: 1.2.1 (Marker調整)
→ 完璧なタイミング
```

### Clip Gain vs Track Volume

```
違い:

Clip Gain:
- Clip単位の音量
- エフェクト前に適用
- ヘッドルーム調整

Track Volume:
- トラック全体の音量
- エフェクト後に適用
- ミックス調整

使い分け:

Clip Gain使用ケース:

サンプル音量バラバラ:
各Clipを揃える

クリッピング防止:
-6dB に下げる

音質改善:
適切なレベルで処理

Track Volume使用ケース:

全体バランス:
トラック間のバランス

オートメーション:
音量変化

フェーダー操作:
ライブミックス

実践:

1. Clip Gain:
   各サンプルを-3dB前後に

2. Compressor等のエフェクト適用

3. Track Volume:
   全体バランス調整
```

---

## MIDIクリップの高度な編集

**ノートの完全制御:**

### MIDI Note Editor詳細

```
Piano Roll表示:

┌─────────────────────────────┐
│ C4  ■■    ■   ■■         │
│ B3      ■   ■             │
│ A3  ■       ■             │
│ G3    ■   ■   ■           │
└─────────────────────────────┘

編集モード:

Draw Mode (B):
ノート描画
クリックで配置

Select Mode (A):
ノート選択
移動・編集

Split Mode:
ノート分割

基本操作:

ノート配置:
Draw Mode → クリック

ノート削除:
選択 → Delete

ノート移動:
ドラッグ

ノート長さ:
右端ドラッグ
```

### Velocity編集

```
Velocity = 打鍵の強さ:

範囲: 0-127
127 = 最強
1 = 最弱
0 = 無音(削除)

表示:

ノートの色で表示:
赤 = 強い (100-127)
オレンジ = 中 (64-99)
黄色 = 弱い (1-63)

編集方法:

個別調整:
ノート選択 → 下部バーで調整

複数調整:
Shift+選択 → まとめて調整

グラデーション:
範囲選択 → Opt+ドラッグ
→ 徐々に変化

ランダム化:
範囲選択 → 右クリック
→ Randomize Velocities

実践例:

Hi-Hatパターン:

強拍: Velocity 100
弱拍: Velocity 70
→ グルーヴ感

ピアノフレーズ:

最初: 90
途中: 70, 65, 60
最後: 50
→ 自然な減衰

Bassline:
全て: 110-120
→ 安定した音圧
```

### Note Length(ノート長)の調整

```
重要性:

長すぎ:
音が重なる
濁る

短すぎ:
スタッカート
途切れる

適切:
楽器特性に合わせる

楽器別推奨:

Bass:
1/8 Note (8分音符)
タイトなグルーヴ

Pad:
1 Bar 以上
持続的な響き

Lead:
フレーズに合わせる
レガート/スタッカート

Piano:
少し短め
自然な減衰

Strings:
長め
滑らかな繋がり

調整方法:

個別:
右端ドラッグ

一括:
全選択 → ドラッグ

数値指定:
Length: 1/16

Legato効果:
次のノートまで延長
→ 隙間なし
```

### Note Nudge(微調整)

```
タイミング微調整:

Opt + ← →:
グリッド無視で微移動

用途:

人間的なタイミング:
完璧なグリッドから少しズラす
→ 機械的でない感じ

グルーヴ調整:

Hi-Hat: わずかに前
Snare: わずかに後ろ
→ スウィング感

シンセベース:
Kickの直後に配置
→ タイトな低音

実践テクニック:

ジャズ風:
ノートをランダムに前後
-10ms 〜 +10ms

Funk:
16分のノートを微妙に前
→ 走る感じ

Ambient:
わずかに遅らせる
→ リラックス感

操作:

1. ノート選択

2. Opt + → を数回
   (約5ms ずつズレる)

3. 耳で確認

4. 調整繰り返し
```

---

## クリップエンベロープの活用

**オートメーションをClip内に:**

### Clip Envelopeとは

```
概念:

Track Automation:
トラック全体に適用
Arrangement Viewで表示

Clip Envelope:
Clip単位で適用
Clip内で完結

メリット:

独立性:
各Clipに異なる設定

再利用:
Clipコピーで設定も複製

柔軟性:
Session Viewでも使える

表示:

Clip View > Envelopes:

┌──────────────────────────┐
│ Device Chooser           │
│ ├─ Mixer                 │
│ │  ├─ Volume             │
│ │  ├─ Pan                │
│ │  └─ Send A             │
│ └─ [使用エフェクト]      │
└──────────────────────────┘
```

### Volume Envelopeによる音量変化

```
基本設定:

1. Clip選択

2. Clip View > Envelopes

3. Device: Mixer

4. Control: Volume

5. 波形上にライン表示

ブレークポイント配置:

クリック:
ラインにポイント追加

ドラッグ:
音量変化作成

削除:
ポイント選択 → Delete

応用例:

フィルターサウンド:

開始: -∞ dB (無音)
1拍目: 0 dB
→ フェードイン

ビルドアップ:

4小節かけて:
0 dB → +3 dB
→ 徐々に盛り上がり

Pad 音量調整:

Intro: -6 dB
Chorus: 0 dB
Outro: -12 dB
→ セクション毎に変化

サイドチェイン効果:

Kickに合わせて:
1拍: -12 dB
1.5拍: 0 dB
→ ポンピング
```

### Pan Envelopeでステレオ展開

```
設定:

Device: Mixer
Control: Pan

範囲:
-100 (Left) 〜 +100 (Right)

テクニック:

左右振り:

開始: -50 (Left)
2拍: +50 (Right)
4拍: -50 (Left)
→ ステレオ動き

回転効果:

4小節で1周:
-100 → 0 → +100 → 0 → -100
→ 音が回る

Hi-Hat Pan:

奇数拍: -30
偶数拍: +30
→ ワイドな空間

実践:

Synth Lead:

Verse: Center (0)
Chorus: 動的 (-50 〜 +50)
→ Chorusで広がり

Vocal Doubles:

Main: Center
Double 1: -40
Double 2: +40
→ 厚み

Percussion:

Shaker: -60 → +60 (連続)
Conga: +40
Bongo: -40
→ 空間配置
```

### エフェクトパラメータのエンベロープ

```
Filter Cutoff:

Device: Auto Filter
Control: Frequency

時間変化:

開始: 200 Hz (暗い)
8拍: 8000 Hz (明るい)
→ フィルタースイープ

Reverb Send:

Device: Send A (Reverb)
Control: Send Level

ダイナミクス:

通常: 0% (ドライ)
最後の1拍: 80% (リバーブ)
→ テール効果

Delay Feedback:

Device: Echo
Control: Feedback

クリエイティブ:

開始: 0%
終盤: 75%
→ 無限ディレイ効果

実践例 - Techno Bass:

1. Filter Cutoff Envelope:
   1小節: 300 Hz → 1200 Hz
   → ウォブル

2. Resonance Envelope:
   2拍毎: 10% → 70%
   → アクセント

3. Volume Envelope:
   Kickと逆相
   → サイドチェイン風
```

---

## マルチクリップ編集テクニック

**複数Clipを一気に操作:**

### 複数選択の応用

```
選択方法:

矩形選択:
ドラッグで範囲選択
複数トラック・Clip同時選択

Shift+クリック:
個別に追加選択

Cmd+A:
全Clip選択

一括編集:

Transpose:
全選択 → +2 st
→ 全部半音2つ上げ

Loop Length:
全選択 → 4 Bars
→ 全部4小節に

Fade:
全選択 → Fade 10ms
→ 全部にフェード

Color:
全選択 → 色変更
→ セクション整理
```

### Clip Grooveの適用

```
Groove Pool:

Clip View > Groove:

プリセット選択:
- MPC Swing
- Logic Swing
- Ableton Grooves

効果:
タイミングを微調整
→ 人間的なグルーヴ

使い方:

1. Groove Pool開く
   (Cmd+Opt+G)

2. Grooveをドラッグ
   → Clipにドロップ

3. Amountで調整
   0-130%

4. 適用確認

応用:

全ドラムに適用:

1. ドラムClip全選択

2. "MPC 60-16" 適用

3. Amount: 40%

4. 統一感のあるグルーヴ

異なるGroove:

Drums: MPC Swing
Bass: Logic Swing 1/16
→ 微妙なズレで深み
```

### Follow Actionによる自動展開

```
設定:

Clip View > Launch:

Follow Action:
┌─────────────────┐
│ Follow Action A │
│ ├─ Next          │
│ ├─ Previous      │
│ ├─ Random        │
│ └─ Other         │
│                 │
│ Follow Time:    │
│ 4 Bars          │
└─────────────────┘

動作:

Next:
次のClipへ自動移動

Random:
ランダムにClip選択

Other:
指定Clipへジャンプ

実践:

ライブセット:

Drum Variation 1-4:
Follow Action: Random
Time: 4 Bars
→ 自動で変化

Bass Pattern:
Clip 1: Follow → Clip 2
Clip 2: Follow → Clip 1
Time: 8 Bars
→ A-Bパターン繰り返し

Ambient Pad:
Random + 16 Bars
→ 予測不可能な展開
```

---

## クリエイティブなクリップ操作

**型破りなテクニック:**

### Clip Stretch(クリップ引き伸ばし)

```
Arrangement Viewでの操作:

Cmd+E で分割後:

端をドラッグ:
Warp On → ピッチ維持
Warp Off → ピッチ変化

効果:

2倍に伸ばす:
Warp On: 半分のテンポ
Warp Off: 1オクターブ下

1/2に縮める:
Warp On: 2倍速
Warp Off: 1オクターブ上

クリエイティブ用途:

Vocal Chop引き伸ばし:
2倍 → 深いボイス

Drum Loop圧縮:
1/2 → ハイスピード感

Pad スロー:
4倍 → ドローン化
```

### Clip Reversal Tricks(逆再生トリック)

```
高度な使い方:

リバーブ → リバース:

1. Clipに強いReverbかける

2. Consolidate (Cmd+J)

3. Reverse: ☑

4. 神秘的なビルドアップ

ディレイ → リバース:

1. Delay Feedbackを70%に

2. Consolidate

3. Reverse

4. 時間逆行サウンド

ボーカルチョップ応用:

1. ボーカル4分割

2. 1つだけReverse

3. 並べ替え

4. グリッチ効果

実践例:

Transition作成:

Before Drop:
リバーブPad → Reverse
→ 吸い込まれる感じ

Drop直前:
ドラムループ → Reverse
→ 期待感
```

### Micro-Editing(マイクロ編集)

```
極小Clipの活用:

1/32 Note Clip:

用途:
グリッチサウンド
テクスチャ

作成:

1. サンプルを細かくSplit

2. ランダム配置

3. 独特のリズム

0.1秒サンプル:

トランジェント部分だけ抽出:

1. Kickのアタック部分

2. 0.05秒だけ切り出し

3. 連打配置

4. ハードなビート

Glitch Technique:

1. ボーカル1単語選択

2. 0.1秒ずつ分割

3. ランダム並べ替え

4. Reverse一部適用

5. グリッチボーカル完成
```

### Polyrhythm Clipping(ポリリズム)

```
複数テンポの共存:

Setup:

Track 1: 4 Bars Loop
Track 2: 3 Bars Loop
→ 12小節で同期

Track 1: 4 Bars
Track 2: 5 Bars
→ 20小節で同期

効果:
予測不可能なパターン
複雑なグルーヴ

実践:

Techno Polyrhythm:

Kick: 4 Bars (4/4)
Percussion: 3 Bars (実質3/4)
Hi-Hat: 5 Bars
→ 20小節周期の複雑さ

Ambient:

Pad 1: 8 Bars
Pad 2: 7 Bars
Pad 3: 11 Bars
→ 長大な周期
```

---

## ワーピングとタイムストレッチ

**時間軸の自在な操作:**

### Warp Modeの使い分け

```
Beats Mode:

用途:
ドラムループ
パーカッション

特徴:
トランジェント保持
リズム維持

設定:
Preserve: Transients

推奨:
BPM変更時

Tones Mode:

用途:
メロディ楽器
ボーカル

特徴:
ピッチ保持
音質優先

設定:
Grain Size: 50-100

推奨:
ボーカル、ギター

Texture Mode:

用途:
Pad
Ambient
ドローン

特徴:
滑らかな変化
アーティファクト少

設定:
Grain Size: 100-200

推奨:
長い持続音

Complex Mode:

用途:
マスタートラック
フルミックス

特徴:
高品質
CPU負荷大

推奨:
完成品のBPM変更

Complex Pro Mode:

用途:
ボーカル
ポリフォニック

特徴:
最高品質
Formant補正

設定:
Envelope: 128

推奨:
プロフェッショナル用途

Re-Pitch Mode:

用途:
ターンテーブル効果
クリエイティブ

特徴:
ピッチ連動
ビニール感

推奨:
ヒップホップ、DJ
```

### タイムストレッチの実践

```
BPM変更なしでClip延長:

方法:

1. Warp: On

2. Clip端をドラッグ

3. 長さ変更

4. ピッチそのまま

応用:

ボーカルフレーズ調整:

Original: 4 Bars
調整後: 5 Bars
→ ゆったり

ドラムループ圧縮:

Original: 8 Bars
調整後: 7 Bars
→ タイト

ダブルタイム効果:

1. Clipを半分に圧縮

2. Warp: On

3. ピッチ維持

4. 2倍速

ハーフタイム効果:

1. Clipを2倍に延長

2. Warp: On

3. ピッチ維持

4. 半分速

クリエイティブストレッチ:

不均等な伸縮:

1. 複数Warp Marker配置

2. 部分的に延長

3. 部分的に圧縮

4. 独特のタイム感
```

### Transpose vs Warp Speed

```
Transpose(半音単位):

変更:
音程のみ

タイミング:
そのまま

用途:
キー変更
ハーモニー調整

Warp(テンポ連動):

変更:
速度のみ

音程:
そのまま(Warp On時)

用途:
BPM同期
タイム調整

Re-Pitch Mode:

変更:
速度+音程

連動:
テープ/ビニール風

用途:
DJ的効果
ヒップホップ

実践例:

キー変更(ピッチのみ):
Transpose: +2 st
Warp: On

BPM同期(速度のみ):
Warp: On
Clip Tempo調整

ビニール風(両方):
Re-Pitch Mode
速度変更 = 音程変更
```

---

## 高度なクリップ管理

**プロジェクト全体の効率化:**

### Clip Color Coding(色分け)

```
色による整理:

ドラム系:
赤系統
Kick, Snare, Hi-Hat

ベース系:
青系統
Bass, Sub Bass

メロディ系:
緑系統
Lead, Chord

効果音系:
黄色系統
FX, Transition

ボーカル系:
紫系統
Vocal, Chop

設定方法:

1. Clip右クリック

2. Color選択

3. 色指定

一括変更:

1. 複数Clip選択

2. 右クリック

3. 同じ色適用

視覚的メリット:

Session View:
セクション識別容易

Arrangement View:
トラック判別容易

ライブパフォーマンス:
瞬時に識別
```

### Clip Naming Convention(命名規則)

```
推奨命名:

[楽器]-[パート]-[バリエーション]

例:

Kick-Main-01
Kick-Main-02
Bass-Intro-A
Bass-Chorus-B
Vocal-Verse-1
Vocal-Verse-2
FX-Riser-Short
FX-Impact-Heavy

メリット:

検索容易:
"Kick" で全Kick検出

整理明確:
機能一目瞭然

共同作業:
他者も理解容易

ショートカット:

F2キー:
選択Clipをリネーム

Cmd+R:
クイックリネーム

実践:

制作開始時:
命名規則決定

制作中:
随時リネーム

完成時:
全Clip確認
```

### Clip Grouping(グループ化)

```
Group Tracks活用:

ドラムグループ:

├─ Kick
├─ Snare
├─ Hi-Hat
├─ Percussion
└─ [Drum Group]

メリット:

一括処理:
グループにエフェクト

ミックス簡単:
1つのフェーダー

CPU軽減:
共有エフェクト

作成方法:

1. 複数Track選択

2. Cmd+G

3. Group作成

4. グループ名設定

グループ内Clip:

独立編集:
個別に操作可能

共通処理:
グループエフェクト

柔軟性:
バランス調整容易

実践例:

ボーカルグループ:

Main Vocal
Double 1
Double 2
Ad-libs
→ 1つのReverbで処理

シンセグループ:

Lead
Pad
Arp
Bass
→ 1つのCompで処理
```

---

## Clip Launch設定

**ライブパフォーマンス最適化:**

### Launch Modeの種類

```
Trigger:
クリックで即座に再生
停止は別操作必要

Gate:
押してる間だけ再生
離すと停止

Toggle:
1回目: 再生
2回目: 停止

Repeat:
ホールドでリピート
リズムに合わせて連打

用途別推奨:

Drums:
Trigger
確実な再生

FX:
Gate
短い効果音

Vocal Chop:
Toggle
オン/オフ切替

Percussion Loop:
Repeat
リズミカルな挿入
```

### Launch Quantization

```
タイミング設定:

None:
即座に再生

1 Bar:
次の小節頭

1/4:
次の拍

Global:
グローバル設定に従う

用途:

Kick Pattern:
1 Bar
小節頭で切替

Hi-Hat:
1/4
拍で切替

FX:
None
即座に

Bass:
1 Bar
タイミング重視

設定方法:

Clip View > Launch:
Quantization選択

または

Global Quantization:
全Clipに適用
```

### Legato Mode

```
機能:

再生位置継続:
Clip切替時もタイミング維持

用途:

ドラムパターン切替:

Pattern A → Pattern B
途中から切替
リズム継続

ベースライン変化:

Bass A → Bass B
タイミングそのまま
音程だけ変化

設定:

Clip View > Launch:
Legato Mode: ☑

メリット:

グルーヴ維持:
リズム途切れない

滑らか:
自然な切替

ライブ向き:
即興演奏に最適
```

---

## まとめ

### 必須ショートカット

```
Cmd+D: 複製(最重要)
Cmd+C/V: コピー/ペースト
Cmd+X: カット
Cmd+E: 分割
Cmd+J: 統合
Cmd+Opt+F: Fade
Cmd+G: Group作成
F2: リネーム
```

### Clip編集フロー

```
1. パターン作成 (4 Bars)
2. Cmd+D で複製
3. バリエーション追加
4. Consolidate で確定
5. 色分け・命名
6. Group化
7. Launch設定
8. 完成
```

### チェックリスト

```
□ Cmd+D で複製マスター
□ Loop設定理解
□ Transpose活用
□ Reverse試す
□ Fade設定
□ Consolidate実行
□ Warp Mode使い分け
□ Clip Envelope活用
□ Warp Marker調整
□ Follow Action設定
□ 色分け実施
□ 命名規則確立
□ Group化完了
□ Launch設定最適化
```

### レベル別目標

```
初級(0-3ヶ月):
□ 基本操作マスター
□ Cmd+D使いこなし
□ Loop設定できる
□ Fade理解

中級(3-12ヶ月):
□ Warp完全理解
□ Envelope使える
□ Groove適用できる
□ Micro-Editing実践

上級(1年以上):
□ Follow Action活用
□ Polyrhythm作成
□ クリエイティブ編集
□ 独自ワークフロー確立
```

---

## ジャンル別Clip編集テクニック

**スタイルに合わせた応用:**

### Techno編集

```
特徴:
反復・ミニマル・緻密

必須テクニック:

ループ長:
4 Bars固定
全Track統一

Micro-Edit:
Hi-Hat細かく調整
1/16 Note単位

サイドチェイン:
Volume Envelope
Kickに同期

フィルタースイープ:
Cutoff Envelope
8 Bars展開

実践フロー:

1. Kick: 4 Bars

2. Bass: 4 Bars
   Volume Envelopeでサイド

3. Hi-Hat: 4 Bars
   Velocity微調整

4. Perc: 3 Bars
   ポリリズム効果

5. 全て同期して完成
```

### House編集

```
特徴:
グルーヴ重視・4つ打ち

必須テクニック:

Groove Pool:
MPC Swing 適用
Amount: 20-40%

Velocity調整:
Hi-Hat強弱
自然なグルーヴ

Swing:
Hi-Hat微妙に前
スウィング感

シンコペーション:
Bass配置ズラす
グルーヴ強調

実践フロー:

1. Kick: 4つ打ち

2. Groove適用
   全ドラムに

3. Hi-Hat Velocity
   強拍100, 弱拍70

4. Bass Note Nudge
   Kickの直後

5. グルーヴ完成
```

### Hip-Hop編集

```
特徴:
サンプリング・ブレイクビーツ

必須テクニック:

Chop:
ボーカル細分化
Cmd+E連打

Re-Pitch:
ビニール風
サンプラー感

Time Stretch:
ループを極端に延長/圧縮
LoFi効果

Reverse:
一部逆再生
アクセント

実践フロー:

1. サンプル選択

2. Cmd+E で細分化

3. ランダム並べ替え

4. 一部Reverse

5. Re-Pitch Mode
   ±3 st

6. チョップ完成
```

### Ambient編集

```
特徴:
長尺・空間・発展

必須テクニック:

長いLoop:
16-32 Bars
ゆっくり展開

Envelope複雑:
Volume/Pan/Filter
時間かけて変化

Polyrhythm:
異なる周期
予測不能性

Reverse Reverb:
Pad → Reverb → Reverse
神秘的

実践フロー:

1. Pad: 16 Bars

2. Volume Envelope
   徐々に上昇

3. Pan Envelope
   ゆっくり回転

4. Filter Envelope
   16小節スイープ

5. 他Padと異なる周期
   7 Bars, 11 Bars

6. 長大な展開完成
```

### Drum & Bass編集

```
特徴:
高速BPM・複雑リズム

必須テクニック:

Time Stretch:
Breakbeat 170BPM化
Warp: Beats Mode

Micro-Chop:
ブレイク細分化
1/32単位編集

Resampling:
エフェクト → Consolidate
レイヤー構築

ポリリズム:
ドラム 3/4 周期
Bass 4/4 周期

実践フロー:

1. Breakbeat取り込み

2. Warp: Beats
   170 BPM化

3. Cmd+E 細分化

4. 並べ替え

5. Bass: 1/16刻み

6. 複雑なリズム完成
```

### Trance編集

```
特徴:
ビルドアップ・リリース

必須テクニック:

長いEnvelope:
16-32 Bars展開
徐々に変化

Filter Sweep:
Cutoff: 200 Hz → 8 kHz
32小節かけて

Volume Build:
-∞ dB → 0 dB
ビルドアップ

Reverse Cymbal:
ドロップ直前
盛り上がり

実践フロー:

1. Lead: 16 Bars

2. Filter Envelope
   32小節スイープ

3. Volume Envelope
   徐々に上昇

4. Reverse Cymbal
   ドロップ前

5. ドロップで全開放

6. ビルドアップ完成
```

---

## トラブルシューティング

**よくある問題と解決策:**

### Clipが再生されない

```
原因1: Mute状態

確認:
Clip/Trackがミュートされていないか

解決:
M ボタン確認
解除

原因2: Launch設定

確認:
Launch Mode設定

解決:
Trigger に変更

原因3: Loop範囲

確認:
Loop長が0になっていないか

解決:
Loop: 4 Bars に設定

原因4: Clip無効

確認:
Clip名グレーアウト

解決:
右クリック → Enable
```

### 音がズレる/遅延

```
原因1: Warp Off

確認:
Warp: Off状態
BPM同期しない

解決:
Warp: On

原因2: Buffer Size大

確認:
Preferences > Audio
Buffer Size

解決:
256 samples 以下に

原因3: CPU負荷

確認:
CPU使用率

解決:
不要Track Freeze
エフェクト削減

原因4: Driver問題

確認:
オーディオドライバ

解決:
ASIO使用
最新版更新
```

### Clipが途切れる

```
原因1: Loop設定ミス

確認:
Loop End位置

解決:
正確な小節区切りに

原因2: Fade Out短すぎ

確認:
Fade Out設定

解決:
10-50 ms に延長

原因3: CPU過負荷

確認:
音飛び発生

解決:
Buffer Size増加
Freeze実行

原因4: Sample Rate不一致

確認:
Project vs File

解決:
同じSample Rateに統一
```

### Consolidateできない

```
原因1: MIDI Clip

確認:
MIDI Clipは不可

解決:
Freeze → Flatten

原因2: 複数Track

確認:
異なるTrackのClip

解決:
1 Track内で実行

原因3: 空白含む

確認:
Clip間に空白

解決:
空白部分も選択

原因4: Looping Clip

確認:
Session View Looping

解決:
一度停止してから実行
```

### Warpがうまくいかない

```
原因1: Mode選択ミス

確認:
Warp Mode不適切

解決:
Drums: Beats
Vocal: Complex Pro
に変更

原因2: Marker不足

確認:
Warp Marker少ない

解決:
トランジェント位置に追加

原因3: 元BPM不明

確認:
Seg. BPM不正確

解決:
手動で正しいBPM設定

原因4: 音質劣化

確認:
極端なストレッチ

解決:
Warp Mode変更
Complex/Complex Pro使用
```

---

## パフォーマンス最適化

**CPU負荷軽減:**

### Freeze活用

```
重いシンセ:

1. Freeze Track

2. 一時的にAudio化

3. CPU負荷激減

4. 編集時はUnfreeze

複数エフェクト:

1. エフェクトチェーン完成

2. Freeze

3. CPU開放

4. 他Track制作

利点:

CPU: 80%削減可能
編集: Unfreezeで復帰
音質: 完全維持
```

### Consolidate戦略

```
タイミング:

編集確定時:
これ以上変更しない

CPU限界時:
音飛び発生

共同作業:
ファイル共有前

手順:

1. 完成部分選択

2. Cmd+J

3. 元Clip削除

4. CPU軽減

注意:

元に戻せない
バックアップ推奨
```

### Buffer Size調整

```
制作時:
512-1024 samples
CPU余裕
遅延許容

演奏時:
128-256 samples
低遅延
リアルタイム

録音時:
256 samples
バランス重視

ミックス時:
512 samples
CPU優先
```

### Sample Rate最適化

```
制作開始時設定:

44.1 kHz:
CD品質
軽い

48 kHz:
映像用
標準

96 kHz:
ハイレゾ
重い

推奨:

Techno/House: 44.1 kHz
Film/Game: 48 kHz
Mastering: 96 kHz
```

---

## ワークフロー改善

**効率的な制作:**

### テンプレート作成

```
目的:
毎回の設定を自動化

作成手順:

1. 基本Track配置
   Kick, Bass, Lead等

2. Group設定

3. Send設定
   Reverb, Delay

4. 色分け完了

5. Save as Default Set

次回から:
即座に制作開始
設定不要
```

### Clip Library構築

```
整理方法:

フォルダ構成:

/Clips
  /Drums
    /Kick
    /Snare
    /Hi-Hat
  /Bass
  /Synth
  /FX
  /Vocals

命名規則:

[BPM]-[楽器]-[特徴]

例:
128-Kick-Deep
140-Bass-Wobble
174-Drums-Break

検索:
BPMで絞り込み
即座に発見
```

### Keyboard Shortcuts カスタマイズ

```
よく使う操作:

Consolidate:
デフォルト: Cmd+J
カスタム: Cmd+Shift+C

Reverse:
デフォルト: なし
カスタム: Cmd+R

Normalize:
デフォルト: なし
カスタム: Cmd+N

設定:

Preferences > Keys
Edit押下
機能選択
キー割当

効果:
作業速度3倍
```

### Macro活用

```
複数操作を1つに:

例: Drum準備

1. Track作成
2. Drum Rack挿入
3. Samples配置
4. Groove適用
5. Color設定

→ 1つのMacroに

作成:

Max for Live
Macro Mapping
保存

実行:
1クリックで完了
```

---

## 上級Tips集

**プロの秘訣:**

### Clip抽出テクニック

```
長い録音から抽出:

1. 全体録音

2. Arrangement View配置

3. 良い部分だけSplit

4. 個別Clip化

5. Session Viewへドラッグ

6. ライブ素材完成

メリット:
即興録音活用
ライブセット構築
```

### Clip Chain作成

```
複数Clipを連続再生:

方法:

1. Clip A: Follow Action → Next

2. Clip B: Follow Action → Next

3. Clip C: Follow Action → Stop

4. 自動連続再生

用途:

曲構成:
Intro → Verse → Chorus
自動展開

ライブ:
Scene連続再生
手が空く
```

### MIDI → Audio変換活用

```
目的:
柔軟な編集

手順:

1. MIDI Clip完成

2. Freeze Track

3. Flatten

4. Audio化

5. Warp, Reverse等適用

利点:

逆再生可能
Stretch可能
サンプリング的操作
```

### Clip Probability設定

```
Max for Live機能:

Clip Probability Device:

発生確率: 50%
→ ランダム再生

用途:

Hi-Hat Variation:
50% で別パターン
予測不能

FX:
25% でサウンド挿入
スパイス

実装:

Max for Live必須
Probability Device配置
確率設定
```

### Audio to MIDI変換

```
機能:

右クリック:
Convert Harmony to MIDI
Convert Drums to MIDI
Convert Melody to MIDI

用途:

サンプル分析:
コード抽出

リズム抽出:
ドラムパターン

再構築:
別音源で再生

精度:

Harmony: 80%
Drums: 90%
Melody: 70%

調整:
抽出後編集必須
```

---

## 最終チェックリスト

### 制作完了時

```
音質確認:
□ クリッピングなし
□ 全Clip Gain適切
□ Fade設定完了
□ ノイズなし

タイミング確認:
□ Warp設定適切
□ グリッド整合
□ Groove統一
□ タイミングズレなし

整理:
□ 全Clip命名済
□ 色分け完了
□ Group整理
□ 不要Clip削除

最適化:
□ Freeze実行
□ Consolidate完了
□ CPU使用率50%以下
□ 再生安定
```

### ライブ準備時

```
Launch設定:
□ 全Clip Mode設定
□ Quantization確認
□ Follow Action設定
□ Legato適用

信頼性:
□ 全Clip再生確認
□ Scene動作確認
□ CPU余裕あり
□ バックアップ作成

視認性:
□ 色分け明確
□ 命名統一
□ Scene整理
□ 演奏順序確認
```

### 共同作業時

```
ファイル整理:
□ 全Sample収集
□ Clip命名明確
□ Track整理
□ Tempo記載

互換性:
□ Freeze重いTrack
□ 外部Plugin確認
□ Sample Rate統一
□ Version記載

ドキュメント:
□ README作成
□ 構成説明
□ 使用音源リスト
□ 変更履歴
```

---

## 学習リソース

**さらなる上達のために:**

### 公式リソース

```
Ableton Manual:
Clip詳細説明
全機能網羅

Ableton YouTube:
公式チュートリアル
Tips & Tricks

Ableton Forum:
Q&A
ユーザー交流
```

### 推奨学習順序

```
Week 1-2:
基本操作
Cmd+D, Loop

Week 3-4:
Warp, Transpose
Consolidate

Week 5-6:
Envelope
Groove

Week 7-8:
Follow Action
高度な編集

Month 3-6:
ジャンル別応用
独自ワークフロー

Month 6-12:
Max for Live
完全マスター
```

### 練習プロジェクト

```
初級:
8小節ドラムループ
4 Trackシンプル曲

中級:
16小節展開
8 Track フル構成

上級:
ライブセット構築
Follow Action活用
32 Scene以上
```

---

**次は:** [Warp機能](./warp-modes.md) - BPM同期の魔法
