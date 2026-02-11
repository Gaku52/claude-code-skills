# Session View vs Arrangement View

Ableton Live最大の特徴。2つのビューを理解し、使い分けることで制作スピードが劇的に上がります。

## この章で学ぶこと

- Session Viewの仕組みと活用法
- Arrangement Viewの仕組みと活用法
- 2つのビューの使い分け
- DJプレイとの類似点
- Session → Arrangement ワークフロー
- ライブパフォーマンス活用

---

## なぜ2つのビューがあるのか

**それぞれに最適な用途:**

```
他のDAW:
Arrangement Viewのみ
（Logic, FL Studio, Cubase等）

Ableton Live:
Session View + Arrangement View
両方使える

理由:

Session View:
即興性、ライブ演奏
アイデア出し

Arrangement View:
完成させる
従来のDAWと同じ

強み:
2つを行き来できる
= 最強のワークフロー

DJ的に言うと:

Session View = DJプレイ
自由に曲を繋げる即興性

Arrangement View = ミックス録音
最初から最後まで完成形
```

---

## Session View（セッションビュー）

**グリッド型の即興ツール:**

### 基本構造

```
Session View:

┌───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ ← Clip Slots
├───┼───┼───┼───┼───┤
│ 6 │ 7 │ 8 │ 9 │10 │
├───┼───┼───┼───┼───┤
│11 │12 │13 │14 │15 │
└───┴───┴───┴───┴───┘
 Track 1-5

縦軸:
Track（トラック）
= 楽器1つ

横軸:
Scene（シーン）
= 展開1つ

クリップ:
各マス = 1つのループ
（8小節、16小節等）

用語:

Clip Slot:
クリップを入れる枠

Empty Slot:
空のスロット

Playing Clip:
再生中のクリップ（緑色）

Stopped Clip:
停止中（灰色）
```

### Session Viewの操作

```
クリップ再生:

クリックで再生:
クリップをクリック
→ 次の小節頭から再生開始

すぐ再生:
Shift+クリック
→ 即座に再生

停止:
もう一度クリック
または■ボタン

Scene再生:

Scene▶ボタン:
その行の全クリップを同時再生

例:
Scene 1▶
→ Track 1-5の Scene 1 を全て再生

ショートカット:
数字キー 1-9
→ Scene 1-9 再生

Stop All:
Shift+Space
→ 全トラック停止

レコーディング:

Clip Slot選択:
空のスロットをクリック

●Record:
F9 または Transport の●
→ 録音開始

停止:
Space
→ クリップ作成完了
```

### Session ViewのDJ的活用

```
DJプレイとの類似:

DJデッキ = Abletonトラック
Deck A → Track 1
Deck B → Track 2

曲 = クリップ
Song A → Clip 1
Song B → Clip 2

Crossfader = ボリュームフェーダー
音量で切り替え

実践例:

Track 1: Kick
├─ Clip 1: Kick A
├─ Clip 2: Kick B
└─ Clip 3: Kick C

Track 2: Bass
├─ Clip 1: Bass A
├─ Clip 2: Bass B
└─ Clip 3: Bass C

Track 3: Synth
├─ Clip 1: Synth A
├─ Clip 2: Synth B
└─ Clip 3: Synth C

プレイ:
Scene 1: Kick A + Bass A + Synth A
Scene 2: Kick B + Bass B + Synth B
Scene 3: Kick A + Bass C + Synth B

自由に組み合わせ可能！
```

### Session Viewの強み

```
即興性:

その場で判断:
次にどのクリップを鳴らすか
リアルタイム決定

DJセットのように:
観客の反応を見て変更

アイデア出し:

複数バージョン:
Kick を3種類試す
どれが良いか聴き比べ

実験:
Bass + Synth A
Bass + Synth B
組み合わせ試行錯誤

ライブパフォーマンス:

クラブで演奏:
ノートPCとMIDIコントローラー
即興で曲作り

Richie Hawtin:
Session Viewで有名

制約なし:

曲の長さ不定:
Scene 1を5分
Scene 2を10分
自由に調整
```

---

## Arrangement View（アレンジメントビュー）

**タイムライン型の完成ツール:**

### 基本構造

```
Arrangement View:

┌──────────────────────────────────┐
│ Track 1 ████████░░░░████░░░░████│ ← Kick
│ Track 2 ░░░░████████████░░░░████│ ← Bass
│ Track 3 ░░░░░░░░████████████░░░░│ ← Synth
│ Track 4 ████░░░░░░░░░░░░████████│ ← Vocal
└──────────────────────────────────┘
  0:00    1:00    2:00    3:00  4:00
  Intro   Build   Drop    Break Outro

横軸:
時間（左→右）

縦軸:
トラック（上→下）

クリップ:
時間軸上に配置

特徴:

直線的:
最初から最後まで

固定:
2:30でドロップ等決まっている

従来のDAW:
Logic, FL Studio と同じ
```

### Arrangement Viewの操作

```
クリップ配置:

Browserからドラッグ:
音色 → タイムラインに配置

Session Viewから:
クリップをドラッグ
→ Arrangement Viewに移動

コピー:
Cmd+C / Cmd+V

編集:

切る:
Cmd+E
→ 分割

移動:
ドラッグ

伸ばす:
端をドラッグ
→ ループ延長

削除:
選択してDelete

録音:

リアルタイム録音:
●Record → 演奏
→ タイムラインに記録

オーバーダビング:
2回目の録音
→ 重ねて録音

マーカー:

Locator:
Cmd+クリック
→ マーカー設定

名前:
「Intro」「Drop」等

ジャンプ:
クリックで移動
```

### Arrangement Viewの強み

```
完成させる:

曲の構成:
0:00-0:30 Intro
0:30-1:30 Build
1:30-3:00 Drop
3:00-4:00 Outro

明確:
どこで何が起こるか一目瞭然

編集:

細かい調整:
2:45.3 でシンバル
正確な位置

コピペ:
Drop をコピー
→ 2回目のDropに貼る

書き出し:

Master出力:
Cmd+Shift+R
→ WAV/MP3書き出し

範囲指定:
Loop Bracesで範囲
→ その部分のみ書き出し

従来のDAWユーザー:

Logic Pro等の経験者:
Arrangement Viewなら慣れている
すぐ使える
```

---

## 2つのビューの使い分け

**適材適所:**

### 制作フェーズ別

```
Phase 1: アイデア出し
→ Session View

8小節ループ作成:
Kick + Bass + Synth
ひたすらループ

複数バージョン:
Scene 1-10 作成
一番良いもの選ぶ

Phase 2: 構成決定
→ Session View

Scene順序:
1 → 3 → 2 → 4
展開を決める

録音:
Session Viewの演奏を
Arrangement Viewに録音

Phase 3: 完成
→ Arrangement View

細かい編集:
イントロ追加
ブレイク作成
オートメーション

書き出し:
WAV/MP3

流れ:
Session (アイデア)
→ Arrangement (完成)
```

### タスク別

```
Task: 8小節ループ作り
→ Session View

理由:
繰り返し聴ける
すぐ変更できる

Task: イントロ/アウトロ作成
→ Arrangement View

理由:
時間軸で配置
徐々にフェードイン

Task: ライブ演奏
→ Session View

理由:
即興性
観客の反応で変更

Task: リリース用完成
→ Arrangement View

理由:
正確な長さ
プロの仕上がり

Task: DJセットで使う曲
→ どちらでもOK

Session:
ライブリミックス

Arrangement:
完成品として
```

---

## Session → Arrangement ワークフロー

**プロの制作フロー:**

### Step 1: Session Viewでアイデア (Day 1-2)

```
1. 新規プロジェクト:
   128 BPM、4/4

2. Track 1: Kick
   Browser > Drums > Kick
   → 8小節ループ作成

3. Track 2: Bass
   Browser > Sounds > Bass
   → 8小節ループ作成

4. Track 3: Synth
   Browser > Sounds > Synth
   → 8小節ループ作成

5. 聴き返し:
   Scene 1 再生
   → ループが完成

6. バリエーション:
   Scene 2-4 も作成
   → 異なるシンセ、ベース

結果:
4つのSceneができる
```

### Step 2: Session Viewで展開決定 (Day 3)

```
1. Scene順序決定:
   Scene 1 → Intro
   Scene 2 → Build
   Scene 3 → Drop
   Scene 4 → Outro

2. 演奏:
   1 → 2 → 3 → 4
   順番に再生

3. 録音:
   Arrangement View の●Record
   → Session View演奏を録音

4. 確認:
   Tabキーで Arrangement View
   → 録音されている
```

### Step 3: Arrangement Viewで完成 (Day 4-7)

```
1. イントロ追加:
   最初の16小節
   → Kickのみ

2. ビルドアップ:
   32小節かけて
   → 徐々にシンセ追加

3. ドロップ:
   全要素フル

4. ブレイク:
   Bassカット
   → 静かに

5. アウトロ:
   徐々にフェードアウト

6. オートメーション:
   フィルター開閉
   リバーブ追加

7. マスタリング:
   Master トラック
   → Limiter

8. 書き出し:
   Cmd+Shift+R
   → 完成！
```

---

## ライブパフォーマンス活用

**Session Viewの真骨頂:**

### ライブセットの構築

```
準備:

20-30 Scene作成:
それぞれ異なる展開

MIDI Controller:
Akai APC40
Push 2
または DDJ-FLX4（MIDI化）

本番:

Scene起動:
MIDIパッドで Scene 1-9
即座に切り替え

クリップ起動:
個別に on/off

エフェクト:
リアルタイムで Reverb, Delay

即興:
観客の反応で展開変更

有名アーティスト:

Richie Hawtin (Plastikman):
Session View のみでライブ

Deadmau5:
Arrangement Viewベース
一部 Session View

Nina Kraviz:
Session View + DJ

あなたも:

DJ + Session View
= ハイブリッドセット
```

---

## 両方を同時に使う

**禁断のテクニック:**

### Session Viewオーバーライド

```
問題:

Session ViewとArrangement View
同時再生すると衝突

解決:

Arrangement Viewを再生中:
Session Viewのクリップ起動
→ そのトラックだけSession優先

Back to Arrangement:
Arrangement Record Enableボタン
→ 元に戻る

活用:

Arrangement Viewで曲再生:
基本はアレンジ通り

Session Viewで即興:
一部トラックだけ変更

ライブ感:
毎回違う演奏
```

---

## 実践: 両方のビューで作る

**60分の演習:**

### 演習1: Session Viewでループ (30分)

```
1. Session View起動

2. Track 1 (Kick):
   Browser > Drums > Kick
   → 8小節ループ

3. Track 2 (Bass):
   Browser > Sounds > Bass
   → 8小節ループ

4. Track 3 (Synth):
   Browser > Sounds > Synth
   → 8小節ループ

5. Scene 1 再生:
   3トラック同時再生

6. 調整:
   音量バランス
   EQ

7. Scene 2 作成:
   異なるBass、Synth

8. Scene 3 作成:
   さらに別パターン
```

### 演習2: Arrangement Viewで完成 (30分)

```
1. Tab → Arrangement View

2. Session Viewクリップをドラッグ:
   Scene 1 → 0:00-0:32
   Scene 2 → 0:32-1:04
   Scene 3 → 1:04-1:36

3. イントロ追加:
   最初の8小節
   → Kickのみ

4. アウトロ追加:
   最後の8小節
   → 徐々にフェードアウト

5. 再生:
   最初から最後まで

6. 調整:
   不要な部分削除

7. 保存:
   Cmd+S
```

---

## よくある質問

### Q1: どっちのビューをメインに使うべき？

**A:** Session Viewから始める

```
初心者:
Session Viewの方が楽しい
すぐ音が出る

中級者:
両方使い分け

上級者:
ワークフロー確立済み

推奨:

Week 1-4:
Session Viewのみ

Week 5-:
Arrangement Viewも

理由:
Session Viewでアイデア出しが楽
Arrangement Viewは後から学べる
```

### Q2: Session Viewだけで完成させられる？

**A:** 可能、ただし限定的

```
可能:

ライブセット:
Session Viewのみで十分

ループ音楽:
Techno、Minimal

不向き:

複雑な曲:
イントロ、ブレイク、展開多い

正確な長さ:
3:45ちょうど等

リリース:
プロの仕上がり

結論:
ライブ用 → Session View
リリース用 → Arrangement View
```

### Q3: DJとライブの違いは？

**A:** Session Viewライブは制作寄り

```
DJ:
既存曲をプレイ
Rekordbox + CDJ

ライブ:
自作ループを即興組み合わせ
Ableton Live Session View

ハイブリッド:
DJ 50% + ライブ 50%
= 最強

例:
Richie Hawtin
DJもライブもやる
```

---

## まとめ

### 2つのビューの特徴

```
Session View:
- グリッド型
- 即興性高い
- アイデア出し
- ライブ演奏向き
- DJに似ている

Arrangement View:
- タイムライン型
- 完成させる
- 細かい編集
- 書き出し
- 従来DAWと同じ
```

### 使い分け

```
アイデア出し → Session View
完成 → Arrangement View
ライブ → Session View
リリース → Arrangement View
```

### ワークフロー

```
1. Session Viewでループ作成
2. 複数Scene作成
3. Arrangement Viewに録音
4. 細かい編集
5. 書き出し
```

### チェックリスト

```
□ Session Viewでクリップ再生
□ Scene再生を試す
□ Arrangement Viewでクリップ配置
□ TabキーでView切り替え
□ Session → Arrangementワークフロー実践
□ 8小節ループを完成させる
```

---

**次は:** [プロジェクト設定](./project-setup.md) - 新規プロジェクトの作り方
