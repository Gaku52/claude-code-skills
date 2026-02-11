# Clip編集

Clipを自在に操る。カット、コピー、ペーストから高度なテクニックまで、編集スキルを完全マスターします。

## この章で学ぶこと

- Clip基本操作（選択、移動、複製）
- カット、コピー、ペースト
- ループ設定完全理解
- トランスポーズ（音程変更）
- リバース（逆再生）
- Fade In/Out
- Consolidate（統合）
- Split/Join（分割・連結）

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

### Loop Brace（ループブレース）

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

## トランスポーズ（音程変更）

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

### Detune（微調整）

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

## リバース（逆再生）

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

## Consolidate（統合）

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

## Split（分割）

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

## まとめ

### 必須ショートカット

```
Cmd+D: 複製（最重要）
Cmd+C/V: コピー/ペースト
Cmd+X: カット
Cmd+E: 分割
Cmd+J: 統合
Cmd+Opt+F: Fade
```

### Clip編集フロー

```
1. パターン作成 (4 Bars)
2. Cmd+D で複製
3. バリエーション追加
4. Consolidate で確定
5. 8-16小節完成
```

### チェックリスト

```
□ Cmd+D で複製
□ Loop設定
□ Transpose理解
□ Reverse試す
□ Fade設定
□ Consolidate実行
□ 8小節ループ完成
```

---

**次は:** [Warp機能](./warp-modes.md) - BPM同期の魔法
