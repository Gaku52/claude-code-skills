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

### Consolidate（統合）機能

```
Consolidateとは:
━━━━━━━━━━━━━━━━━━━━━━━━━

複数クリップ → 1クリップ:
ばらばらのクリップを
→ 1つのオーディオファイルに

ショートカット:
Cmd+J (Mac)
Ctrl+J (Windows)

使用例:
━━━━━━━━━━━━━━━━━━━━━━━━━

例1: ドラム統合

Track 1: Kick (4クリップ)
Track 2: Snare (3クリップ)
Track 3: Hi-hat (5クリップ)

全選択 → Cmd+J
→ 各トラック1クリップに

利点:
CPU負荷軽減
エフェクト処理が確定
編集しやすい

例2: ビルドアップ統合

Build部分:
16クリップ × 8トラック
= 128クリップ

Consolidate実行:
→ 1クリップに
管理が簡単

注意点:
取り消し不可
→ 事前に保存推奨
```

### オートメーション制作

```
オートメーションとは:
━━━━━━━━━━━━━━━━━━━━━━━━━

パラメータ自動変化:
Volume、Pan、Effect等
時間軸で自動的に変化

描画方法:
━━━━━━━━━━━━━━━━━━━━━━━━━

1. パラメータ選択:
   例: Filter Cutoff

2. 録音モード:
   A (Automation) ボタン ON

3. 再生しながら操作:
   Knobを回す
   → 動きが記録される

4. 確認:
   オートメーションレーン表示
   赤い線 = 動き

手動描画:
━━━━━━━━━━━━━━━━━━━━━━━━━

1. Show Automation:
   トラック右クリック
   → Show Automation Lane

2. パラメータ選択:
   プルダウンメニュー
   → Volume、Pan等

3. ブレークポイント追加:
   ダブルクリック
   → ポイント作成

4. ドラッグ:
   上下に動かす
   → 値変更

実践例:
━━━━━━━━━━━━━━━━━━━━━━━━━

ビルドアップフィルター:

0:32 (Build開始)
Filter Cutoff: 20,000 Hz (全開)

0:48 (16小節後)
Filter Cutoff: 1,000 Hz (閉じる)

0:56 (24小節後)
Filter Cutoff: 500 Hz (さらに閉じる)

1:04 (Drop)
Filter Cutoff: 20,000 Hz (一気に開く)

結果:
徐々に音が曇る
→ ドロップで一気に明るく

ボリュームフェードアウト:

3:44 (Outro開始)
Volume: 0 dB (フル)

4:16 (終了)
Volume: -∞ dB (無音)

直線描画:
自然なフェードアウト

パンニング自動化:

Synth Arp:
L 100% ← → R 100%
2小節ごとに左右移動

設定:
0.0: L 100%
2.0: R 100%
4.0: L 100%
6.0: R 100%

効果:
音が左右に揺れる
空間の広がり
```

---

## 両ビューの連携テクニック

**シームレスな統合ワークフロー:**

### Session録音 → Arrangement編集

```
最強ワークフロー:
━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Session Viewで即興演奏

準備:
Scene 1-10 作成済み
各Scene異なる展開

演奏:
Scene 1 → 2 → 5 → 3 → 10
即興で順番変更

録音開始:
Arrangement Record ON (●)
→ Session View演奏を録音

演奏時間:
約5分

Step 2: Arrangement Viewで編集

確認:
Tab → Arrangement View
→ 演奏が全て録音されている

不要部分削除:
最初の1分削除
最後の30秒削除

良い部分抽出:
2:00-2:32 のDropが最高
→ コピーして別の場所に

並び替え:
Drop 1 → Break → Drop 2
構成を整える

Step 3: 細かい調整

トランジション追加:
Scene切り替わり部分
→ Riser、Impact追加

オートメーション:
Filter Sweep
Volume Fade

完成:
Session即興のライブ感
+ Arrangement編集の完成度
= 最高の仕上がり
```

### Arrangement → Session 逆ワークフロー

```
完成曲をSessionに戻す:
━━━━━━━━━━━━━━━━━━━━━━━━━

目的:
完成した曲を
Session Viewでライブ演奏

手順:
━━━━━━━━━━━━━━━━━━━━━━━━━

1. Arrangement View確認:
   完成曲あり
   例: 4分間のトラック

2. セクション分割:
   Intro: 0:00-0:32
   Build: 0:32-1:04
   Drop: 1:04-1:36
   Break: 1:36-2:08
   Build 2: 2:08-2:40
   Drop 2: 2:40-3:44
   Outro: 3:44-4:16

3. 各セクション選択:
   Introを選択 (0:00-0:32)

4. Session Viewにドラッグ:
   Session View Track 1
   → Clip Slot 1

5. 繰り返し:
   Build → Clip Slot 2
   Drop → Clip Slot 3
   Break → Clip Slot 4
   Build 2 → Clip Slot 5
   Drop 2 → Clip Slot 6
   Outro → Clip Slot 7

結果:
Session View:
Scene 1: Intro
Scene 2: Build
Scene 3: Drop
Scene 4: Break
Scene 5: Build 2
Scene 6: Drop 2
Scene 7: Outro

活用:
ライブで演奏可能
Scene順序を即興変更
例: Drop 2回連続
```

### ハイブリッド活用

```
両方同時使用:
━━━━━━━━━━━━━━━━━━━━━━━━━

基本:
Arrangement View再生
→ 曲の土台

Session View起動:
個別トラックだけ変更
→ ライブ感追加

実践例:
━━━━━━━━━━━━━━━━━━━━━━━━━

ライブDJセット:

Track 1-8: Arrangement View
Kick、Bass、Synth等
→ 完成した曲を再生

Track 9-12: Session View
Vocal、FX、Riser等
→ 即興で追加

演奏:
Arrangement再生中
→ Scene 1 起動 (Vocal追加)
→ Scene 2 起動 (FX変更)

毎回違う:
土台は同じ
上モノは即興
→ ライブ感

リハーサル活用:
━━━━━━━━━━━━━━━━━━━━━━━━━

練習:
Session Viewで演奏練習
色々試す

録音:
良かった演奏を
Arrangement録音

確認:
Arrangement再生
→ 客観的に聴く

修正:
良い部分は残す
悪い部分は再録音

繰り返し:
完璧になるまで
Session → Arrangement
```

---

## ジャンル別の推奨ワークフロー

**音楽スタイルに合わせた最適手法:**

### Techno / Minimal

```
特徴:
━━━━━━━━━━━━━━━━━━━━━━━━━

長いループ:
8-16分の曲
ミニマルな変化

徐々に展開:
急激な変化なし
じわじわ盛り上げる

推奨ワークフロー:
━━━━━━━━━━━━━━━━━━━━━━━━━

90% Session View:

理由:
ループ中心
即興性が重要
ライブ演奏向き

制作手順:

1. Session Viewでループ作成:
   Track 1-8
   各8小節ループ

2. 複数バージョン作成:
   Scene 1: Kick + Hi-hat
   Scene 2: +Bass
   Scene 3: +Synth 1
   Scene 4: +Synth 2
   Scene 5: +Perc
   ...
   Scene 20まで

3. Follow Action設定:
   自動でScene切り替え
   ランダム性追加

4. ライブ録音:
   60分Session演奏
   → Arrangement録音

5. 軽い編集:
   不要部分削除
   書き出し

Arrangement使用:
最小限
書き出しのみ

有名アーティスト:
Richie Hawtin、Adam Beyer
→ Session View中心
```

### House / Dance Pop

```
特徴:
━━━━━━━━━━━━━━━━━━━━━━━━━

明確な構成:
Intro → Build → Drop
わかりやすい展開

3-4分:
短め
ラジオフレンドリー

推奨ワークフロー:
━━━━━━━━━━━━━━━━━━━━━━━━━

50% Session / 50% Arrangement:

制作手順:

1. Session Viewでアイデア (Day 1-2):
   8小節ループ
   Kick、Bass、Synth、Vocal

2. Scene作成 (Day 3):
   Scene 1: Intro
   Scene 2: Verse
   Scene 3: Pre-Chorus
   Scene 4: Chorus (Drop)
   Scene 5: Break
   Scene 6: Chorus 2

3. Arrangement移行 (Day 4):
   Scene → Arrangement
   ドラッグで配置

4. 構成整理 (Day 5):
   0:00-0:16 Intro (16小節)
   0:16-0:48 Verse (32小節)
   0:48-1:04 Pre-Chorus (16小節)
   1:04-1:36 Chorus (32小節)
   1:36-2:08 Break (32小節)
   2:08-3:12 Chorus 2 (64小節)
   3:12-3:28 Outro (16小節)

5. 細かい編集 (Day 6-7):
   トランジション
   オートメーション
   ボーカル編集

バランス:
Session = アイデア
Arrangement = 完成
```

### Trance / Progressive

```
特徴:
━━━━━━━━━━━━━━━━━━━━━━━━━

長いビルドアップ:
32-64小節かけて盛り上げる

壮大な展開:
5-8分
感動的

推奨ワークフロー:
━━━━━━━━━━━━━━━━━━━━━━━━━

30% Session / 70% Arrangement:

制作手順:

1. Session Viewで土台 (Day 1):
   基本8小節ループ
   Kick、Bass

2. 早めにArrangement移行 (Day 2):
   構成が重要
   タイムラインで考える

3. Arrangement構成 (Day 3-4):
   0:00-1:00 Intro (64小節)
   1:00-2:04 Build (64小節)
   2:04-3:40 Drop (96小節)
   3:40-4:44 Break (64小節)
   4:44-6:20 Build 2 (96小節)
   6:20-8:28 Drop 2 (128小節)
   8:28-9:00 Outro (32小節)

4. オートメーション重視 (Day 5-6):
   Filter Sweep多用
   Volume変化
   Reverb増減

5. トランジション作り込み (Day 7):
   Riser、Impact
   White Noise
   Reverse Cymbal

Arrangement中心:
緻密な構成が必要
Session不向き

有名アーティスト:
Armin van Buuren、Above & Beyond
→ Arrangement中心
```

### Dubstep / Bass Music

```
特徴:
━━━━━━━━━━━━━━━━━━━━━━━━━

激しいDrop:
重低音
複雑な音作り

短い構成:
3-4分
コンパクト

推奨ワークフロー:
━━━━━━━━━━━━━━━━━━━━━━━━━

40% Session / 60% Arrangement:

制作手順:

1. Session ViewでBass作り (Day 1-2):
   複数Bassバリエーション
   Scene 1-10
   Drop候補

2. 一番良いDrop選択 (Day 3):
   聴き比べ
   Scene 5が最高 → 採用

3. Arrangement構成 (Day 4):
   0:00-0:16 Intro (16小節)
   0:16-0:32 Build (16小節)
   0:32-1:04 Drop 1 (32小節)
   1:04-1:20 Break (16小節)
   1:20-1:36 Build 2 (16小節)
   1:36-2:40 Drop 2 (64小節)
   2:40-2:56 Outro (16小節)

4. Drop作り込み (Day 5-6):
   Bass編集
   Wobble調整
   Sidechain

5. ビルドアップ (Day 7):
   Riser
   Snare Roll
   Filter

Session活用:
Bass実験に最適
Dropバリエーション

有名アーティスト:
Skrillex、Excision
→ ハイブリッド型
```

### Ambient / Downtempo

```
特徴:
━━━━━━━━━━━━━━━━━━━━━━━━━

自由な構成:
明確な展開なし
雰囲気重視

長い:
5-10分以上
ゆったり

推奨ワークフロー:
━━━━━━━━━━━━━━━━━━━━━━━━━

80% Session View:

理由:
即興性が重要
雰囲気を大事に

制作手順:

1. Session Viewで実験 (Day 1-3):
   色々な音色
   Scene 1-30
   雰囲気探る

2. Follow Action多用 (Day 4):
   自動でランダム変化
   予測不可能な展開

3. 長時間録音 (Day 5):
   Session演奏30-60分
   → Arrangement録音

4. 良い部分抽出 (Day 6):
   20分録音
   → 8分に編集

5. 軽い編集 (Day 7):
   フェードイン/アウト
   不要部分削除

Sessionメイン:
作り込みすぎない
自然な流れ

有名アーティスト:
Boards of Canada、Jon Hopkins
→ 実験的アプローチ
```

---

## 効率的な制作のためのビュー切り替え戦略

**時間を無駄にしない賢い使い方:**

### 制作段階別の最適ビュー

```
Day 1: アイデア収集
━━━━━━━━━━━━━━━━━━━━━━━━━

使用: Session View 100%

やること:
- とにかく音を出す
- Kickを5種類試す
- Bassを3種類試す
- Synthを10種類試す

禁止:
- Arrangement開かない
- 完成度考えない
- 構成考えない

理由:
アイデア段階で構成考えると
クリエイティビティが下がる

時間配分:
8時間Session View

Day 2-3: バリエーション作成
━━━━━━━━━━━━━━━━━━━━━━━━━

使用: Session View 100%

やること:
- Scene 1-10 作成
- 各Scene異なる雰囲気
- 聴き比べ

切り替えタイミング:
まだArrangement不要

時間配分:
各日 6時間Session View

Day 4: 構成決定
━━━━━━━━━━━━━━━━━━━━━━━━━

使用: Session 50% / Arrangement 50%

午前 (Session):
- Scene順序決定
- 演奏練習
- 録音

午後 (Arrangement):
- 録音確認
- 不要部分削除
- 構成整理

切り替え:
Tab頻繁に使用
両方行き来

時間配分:
各3時間

Day 5-6: 細かい編集
━━━━━━━━━━━━━━━━━━━━━━━━━

使用: Arrangement View 90%

やること:
- トランジション追加
- オートメーション
- EQ調整
- ミキシング

Session使用:
新しい音追加時のみ

時間配分:
各日 8時間Arrangement
1時間Session (必要時)

Day 7: 仕上げ
━━━━━━━━━━━━━━━━━━━━━━━━━

使用: Arrangement View 100%

やること:
- マスタリング
- 書き出し
- 最終確認

Session不使用:
完成段階

時間配分:
6時間Arrangement
```

### ショートカット活用

```
ビュー切り替え:
━━━━━━━━━━━━━━━━━━━━━━━━━

Tab:
Session ⇄ Arrangement
瞬時に切り替え

使用頻度:
初心者: 1日5回
中級者: 1日20回
上級者: 1日50回以上

練習:
今すぐTab押す
→ また押す
→ また押す
癖にする

作業効率:
━━━━━━━━━━━━━━━━━━━━━━━━━

マルチタスク禁止:
Session作業中
→ Arrangement開かない

集中:
今Session作業中
→ Sessionだけ

切り替えルール:
1時間単位で切り替え
細かく切り替えない

例:
9:00-10:00 Session
10:00-11:00 Arrangement
11:00-12:00 Session

理由:
頻繁な切り替えは
集中力低下

画面レイアウト:
━━━━━━━━━━━━━━━━━━━━━━━━━

Session作業時:
Browser開く
Device開く
Clip View開く

Arrangement作業時:
Track View最大化
Timeline拡大
Device開く

Cmd+Option+L:
View最大化

活用:
作業に応じて画面変更
```

### 実践的な時間配分

```
初心者 (1-3ヶ月目):
━━━━━━━━━━━━━━━━━━━━━━━━━

Session: 80%
Arrangement: 20%

理由:
まずSessionで遊ぶ
楽しさ優先

1曲制作時間:
2-3週間

中級者 (4-12ヶ月目):
━━━━━━━━━━━━━━━━━━━━━━━━━

Session: 50%
Arrangement: 50%

理由:
両方使いこなす段階
バランス重視

1曲制作時間:
1-2週間

上級者 (1年以上):
━━━━━━━━━━━━━━━━━━━━━━━━━

Session: 30%
Arrangement: 70%

理由:
効率化
完成度重視

1曲制作時間:
3-7日

プロ:
━━━━━━━━━━━━━━━━━━━━━━━━━

ジャンルによる:
Techno → Session 70%
House → Session 40%
Trance → Session 20%

1曲制作時間:
1-3日 (集中時)
1-2週間 (通常)
```

---

## 最終チェックリスト：両ビューをマスターする

**自己評価用:**

### Session View理解度

```
基礎:
□ Clip Slotの概念理解
□ Sceneの仕組み理解
□ クリップ再生できる
□ Scene再生できる
□ 複数Sceneを作成できる

応用:
□ Follow Action設定できる
□ MIDIコントローラーで操作
□ 20Scene以上作成経験
□ ライブセットを構築できる
□ 即興演奏できる

マスター:
□ 60分のライブセット作成
□ Follow Actionで自動化
□ Return Track活用
□ Send/Return理解
□ プロレベルのセット構築
```

### Arrangement View理解度

```
基礎:
□ タイムライン操作
□ クリップ配置
□ Locator設定
□ 基本的な編集 (Cut, Copy, Paste)
□ 書き出し (Export)

応用:
□ Consolidate活用
□ Automation描画
□ Fade In/Out設定
□ Loop Brace活用
□ 構成設計できる

マスター:
□ 複雑なAutomation
□ プロレベルの構成
□ ジャンル別構成理解
□ マスタリング
□ リリース品質の完成
```

### 両ビュー連携理解度

```
基礎:
□ Tabで切り替え
□ Session → Arrangement録音
□ Arrangement → Sessionドラッグ

応用:
□ ハイブリッド活用
□ ワークフロー確立
□ 効率的な時間配分
□ ジャンル別使い分け

マスター:
□ プロレベルのワークフロー
□ 1週間で1曲完成
□ ライブとリリース両対応
□ オリジナルワークフロー確立
```

---

## トラブルシューティング

**よくある問題と解決策:**

### Q: Session Viewのクリップが同期しない

```
原因:
Global Quantization設定

解決:
1. 画面下部のQuantization確認
2. 「1 Bar」に設定推奨
3. または「None」で即座に起動

確認方法:
Options > Preferences > Record Warp Launch
→ Default Launch Quantisation: 1 Bar
```

### Q: Arrangement録音がうまくいかない

```
原因:
Arrangement Record Enableオフ

解決:
1. Transport右の●確認
2. Cmd+F9 で録音開始
3. Session View演奏開始
4. 全て録音される

注意:
既存Arrangementは上書きされる
→ 事前に保存推奨
```

### Q: 両ビュー同時再生で音が重なる

```
原因:
Session ViewとArrangementが衝突

解決:
1. Back to Arrangementボタン押す
2. またはSession Viewクリップ停止
3. どちらか一方を使用

活用:
意図的に重ねることも可能
→ ハイブリッド活用
```

---

**次は:** [プロジェクト設定](./project-setup.md) - 新規プロジェクトの作り方
