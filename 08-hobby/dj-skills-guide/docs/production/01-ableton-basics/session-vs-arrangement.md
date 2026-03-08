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

## Session ViewとArrangement Viewの詳細比較

**機能別の徹底比較:**

### 再生方式の違い

```
Session View:
━━━━━━━━━━━━━━━━━━━━━━━━━

ノンリニア再生:
順番が決まっていない
Scene 1 → 3 → 1 → 5
自由に行き来できる

ループベース:
各クリップが独立ループ
8小節、16小節等
終わったら最初に戻る

クオンタイズ:
Global Quantization設定
→ 1 Bar、2 Bar、4 Bar
次の小節頭から起動

リアルタイム性:
演奏中に変更可能
クリップ追加・削除
エフェクト調整

同期:
全トラックが同期
BPM 128なら全て128
テンポ変更は全体に影響

Arrangement View:
━━━━━━━━━━━━━━━━━━━━━━━━━

リニア再生:
左から右へ時間進行
0:00 → 4:00
一直線

固定配置:
2:30でDropと決まっている
毎回同じ位置で再生

正確な時間:
ミリ秒単位で配置可能
2:45.347 等
プロレベルの精度

事前構成:
全て配置済み
変更は編集モード

テンポオートメーション:
BPM変化可能
128 → 140 → 100
曲中でテンポチェンジ

比較表:
━━━━━━━━━━━━━━━━━━━━━━━━━

                Session    Arrangement
再生順序        自由       固定
ループ          標準       オプション
即興性          高         低
編集精度        低         高
ライブ向き      ◎         ×
完成度          △         ◎
初心者          易         中
プロ仕上げ      △         ◎
```

### クリップ管理の違い

```
Session View:
━━━━━━━━━━━━━━━━━━━━━━━━━

クリップスロット:
無制限に追加可能
Track 1に10個、20個
いくらでもバリエーション

色分け:
クリップに色設定
Kick = 赤
Bass = 青
Synth = 緑

Follow Action:
クリップ再生後の動作設定
→ Next、Previous、Random
自動で次のクリップへ

Clip Launch Mode:
Trigger: 起動して再生
Gate: 押している間だけ
Toggle: on/off切り替え
Repeat: ループ回数指定

Clip Length:
各クリップ独立した長さ
Clip 1 = 8小節
Clip 2 = 16小節
Clip 3 = 4小節

グループ化:
Track Group作成
→ まとめて管理
Drums Group = 8トラック

Arrangement View:
━━━━━━━━━━━━━━━━━━━━━━━━━

タイムライン配置:
時間軸上に配置
0:00-0:32 = Intro
0:32-1:04 = Build

クリップ連結:
Consolidate機能
複数クリップ → 1クリップ
Cmd+J

Fade In/Out:
クリップ端にフェード
ドラッグで調整
自然な繋ぎ

Warp:
テンポ同期
オーディオを BPM に合わせる
ピッチ変えずに速度変更

Stretch:
クリップ長さ変更
伸ばす・縮める
タイムストレッチ

Automation:
詳細なオートメーション
Volume、Pan、Effect等
時間軸で自動変化
```

### ワークスペースの違い

```
Session View:
━━━━━━━━━━━━━━━━━━━━━━━━━

画面構成:
左: Browser
中: Clip Grid
右: Device/Clip

縦スクロール:
Scene数が増えると下へ
Scene 1-100等

横スクロール:
Track数が増えると右へ
Track 1-50等

Master Scene:
Scene再生ボタン
全Scene一括管理

Return Tracks:
Send/Return
Reverb、Delay等
全トラック共有

拡張性:
無限にScene追加可能
制限なし

Arrangement View:
━━━━━━━━━━━━━━━━━━━━━━━━━

画面構成:
左: Track List
中: Timeline
右: Device

横スクロール:
時間軸が長いと右へ
0:00-10:00等

縦スクロール:
Track数が増えると下へ

Locator:
時間位置マーカー
Intro、Drop等
ジャンプ可能

Loop Brace:
ループ範囲指定
[ ]で囲む
その部分だけループ再生

Arrangement Overdub:
録音中に重ねる
2回目、3回目の録音
レイヤー追加
```

---

## Session Viewでのライブパフォーマンス活用法

**クラブ・フェスでの実践テクニック:**

### ライブセットの構築方法

```
基本構成:
━━━━━━━━━━━━━━━━━━━━━━━━━

Track構成:
Track 1-4: Drums
  Kick、Snare、Hi-hat、Perc

Track 5-8: Bass
  Sub Bass、Mid Bass、Bass Fill

Track 9-12: Synth
  Lead、Pad、Arp、FX

Track 13-16: Vocal/Effects
  Vocal、Riser、Impact、White Noise

Scene構成:
Scene 1-5: Intro Variations
Scene 6-10: Build Variations
Scene 11-15: Drop Variations
Scene 16-20: Break Variations
Scene 21-25: Outro Variations

合計:
16 Tracks × 25 Scenes
= 400 Clips
約60分のライブセット

色分け:
Intro = 青
Build = 緑
Drop = 赤
Break = 黄
Outro = 灰
```

### MIDIコントローラー設定

```
推奨コントローラー:
━━━━━━━━━━━━━━━━━━━━━━━━━

Ableton Push 2:
完全統合
64パッド
Scene起動、Clip起動
エフェクトコントロール
ディスプレイ付き

Akai APC40 MKII:
40パッド
Scene起動に最適
Crossfader付き

Novation Launchpad Pro:
64パッド RGB
カスタマイズ性高い

DDJ-FLX4 (MIDI化):
DJコントローラー活用
Jog Wheelでフィルター
Crossfaderでミックス

マッピング例:
━━━━━━━━━━━━━━━━━━━━━━━━━

Push 2:
8×8 Pad = Clip起動
Scene Button = Scene起動
Knob 1-8 = Track Volume
Knob 9-12 = Send A-D
Touch Strip = Master Filter

APC40:
5×8 Pad = Clip起動 (Track 1-5)
Scene Launch = Scene起動
Fader = Track Volume
Crossfader = Track A/B切り替え

カスタムマッピング:
Pad 1 = Scene 1
Pad 2 = Scene 2
Knob 1 = Reverb Send
Knob 2 = Delay Send
Knob 3 = Filter Cutoff
Knob 4 = Filter Resonance
```

### Follow Action活用

```
Follow Actionとは:
━━━━━━━━━━━━━━━━━━━━━━━━━

自動Clip切り替え:
Clip再生終了後
→ 次の動作を自動実行

設定項目:
Action A: 第一動作
Action B: 第二動作
Chance A: 確率 (0-100%)
Chance B: 確率 (0-100%)
Time: 実行タイミング (小節数)

動作オプション:
Stop: 停止
Play Again: もう一度再生
Previous: 前のClip
Next: 次のClip
First: 最初のClip
Last: 最後のClip
Any: ランダム
Other: 他のClip (現在以外)

実践例1: Hi-hatバリエーション
━━━━━━━━━━━━━━━━━━━━━━━━━

Track 3: Hi-hat
Clip 1: Pattern A (8小節)
Clip 2: Pattern B (8小節)
Clip 3: Pattern C (8小節)

Clip 1設定:
Action A: Next (60%)
Action B: Other (40%)
Time: 8 Bars

結果:
8小節後
→ 60%でClip 2へ
→ 40%でClip 3へ
ランダム性のあるHi-hat

実践例2: ドロップランダム化
━━━━━━━━━━━━━━━━━━━━━━━━━

Scene 11-15: Drop Variations

各Dropクリップ:
Action A: Other (100%)
Time: 32 Bars

結果:
32小節 (約1分) ごと
→ 別のDropに切り替わる
毎回違う展開

実践例3: ビルドアップ自動化
━━━━━━━━━━━━━━━━━━━━━━━━━

Track 9: Synth Build

Clip 1: Build Start (16小節)
  Action A: Next (100%)
  Time: 16 Bars

Clip 2: Build Mid (16小節)
  Action A: Next (100%)
  Time: 16 Bars

Clip 3: Build Peak (16小節)
  Action A: Play Scene 11 (100%)
  Time: 16 Bars

結果:
自動でビルドアップ
→ 48小節後にDrop (Scene 11) へ
```

### ライブエフェクトテクニック

```
必須エフェクト:
━━━━━━━━━━━━━━━━━━━━━━━━━

Return Track A: Reverb
Algorithm: Large Hall
Decay: 4.0s
Dry/Wet: 100% (Send量で調整)
マッピング: Knob 1

Return Track B: Delay
Time: 1/4 (BPM同期)
Feedback: 60%
Dry/Wet: 100%
マッピング: Knob 2

Return Track C: Filter
Type: Low Pass
Cutoff: 20,000 Hz
Resonance: 0.3
マッピング: Knob 3 + Touch Strip

Return Track D: Sidechain
Compressor + Kick trigger
Attack: 1ms
Release: 150ms
Ratio: 4:1
マッピング: Knob 4

ライブエフェクトルーティング:
━━━━━━━━━━━━━━━━━━━━━━━━━

全トラック:
Send A = Reverb量
Send B = Delay量
Send C = Filter Send
Send D = Sidechain量

演奏中操作:
ビルドアップ時:
  Send A (Reverb) 0% → 50%
  Send C (Filter) 20,000Hz → 500Hz
  徐々に閉じる

ドロップ時:
  Send C (Filter) 500Hz → 20,000Hz
  一気に開く
  Send D (Sidechain) 0% → 80%

ブレイク時:
  Send A (Reverb) 50% → 80%
  Send B (Delay) 0% → 40%
  空間系エフェクト増加
```

---

## Arrangement Viewでの楽曲制作ワークフロー

**プロレベルの完成度を目指す:**

### 楽曲構成の設計

```
標準的なEDM構成:
━━━━━━━━━━━━━━━━━━━━━━━━━

0:00-0:32 (32小節)
Intro:
  Kickのみ
  徐々にHi-hat追加
  8小節ごとにレイヤー追加

0:32-1:04 (32小節)
Build 1:
  Bass追加
  Synth Pad追加
  徐々に盛り上げる
  最後8小節でRiser

1:04-1:36 (32小節)
Drop 1:
  全要素フル
  Lead Synth
  Vocal (あれば)
  最高潮

1:36-2:08 (32小節)
Break:
  Kickカット
  Bassカット
  Pad + Arpだけ
  静かな展開

2:08-2:40 (32小節)
Build 2:
  再びビルドアップ
  Build 1より激しく
  Filter Sweep
  Riser + White Noise

2:40-3:44 (64小節)
Drop 2:
  Drop 1より長い
  ピーク
  最も盛り上がる部分
  32小節 × 2回繰り返し

3:44-4:16 (32小節)
Outro:
  徐々に引き算
  8小節ごとに要素削除
  最後はKickのみ
  Fade Out

合計: 4:16 (256小節)

ジャンル別構成:
━━━━━━━━━━━━━━━━━━━━━━━━━

Techno (Minimal):
Intro: 64小節 (長い)
Build: 32小節
Drop: 64小節 (シンプル)
Break: 32小節
Drop 2: 64小節
Outro: 64小節
合計: 6:00-8:00

House:
Intro: 32小節
Build: 16小節
Drop: 32小節
Break: 16小節
Build 2: 16小節
Drop 2: 32小節
Outro: 32小節
合計: 3:30-4:00

Trance:
Intro: 32小節
Build: 32小節 (長いビルド)
Drop: 64小節
Break: 32小節 (ブレイクダウン)
Build 2: 32小節
Drop 2: 64小節
Outro: 32小節
合計: 5:30-6:30

Dubstep:
Intro: 16小節
Build: 16小節
Drop: 32小節 (重低音)
Break: 16小節
Build 2: 16小節
Drop 2: 32小節
Outro: 16小節
合計: 3:00-3:30
```

### Locator（マーカー）設定

```
Locator追加方法:
━━━━━━━━━━━━━━━━━━━━━━━━━

Cmd+クリック:
タイムライン上部
→ Locator作成

右クリック:
Locatorを右クリック
→ 「Edit」で名前変更

ショートカット:
Cmd+1-9
→ Locator 1-9 にジャンプ

推奨Locator設定:
━━━━━━━━━━━━━━━━━━━━━━━━━

Locator 1: 0:00 (Intro)
Locator 2: 0:32 (Build 1)
Locator 3: 1:04 (Drop 1)
Locator 4: 1:36 (Break)
Locator 5: 2:08 (Build 2)
Locator 6: 2:40 (Drop 2)
Locator 7: 3:44 (Outro)
Locator 8: 1:20 (Drop 1 Peak)
Locator 9: 3:12 (Drop 2 Peak)

活用:
編集中ジャンプ:
  Cmd+3 → Drop 1へ即移動
  確認したい箇所に素早く

クライアント確認:
  「Drop部分聴きたい」
  → Cmd+3で即再生

書き出し範囲:
  Drop 1のみ書き出し
  → Locator 3-4 をLoop Brace
```

---

**次は:** [プロジェクト設定](./project-setup.md) - 新規プロジェクトの作り方
