# Reference Mixing

リファレンストラックでプロレベル到達。LUFS比較・周波数バランス確認を完全マスターします。

## この章で学ぶこと

- Reference Track選び方
- LUFS比較方法
- Spectrum比較
- A/B切り替え
- Volume Matching
- 具体的な調整方法
- よくある間違い

---

## なぜReference Mixingが重要なのか

**プロレベル到達:**

```
Reference なし:

結果:
主観的
バランス不明

Reference あり:

結果:
客観的
プロと比較

プロとアマの差:

アマ:
Reference使わない
主観100%

プロ:
Reference必須
10分ごと比較

真実:

「プロの音」=
Referenceと同じレベル

方法:
LUFS・Spectrum・聴感
全て比較

使用頻度:
100%
全プロジェクト
```

---

## Reference Track選び方

**適切な選択:**

### 条件

```
必須条件:

1. 同じジャンル:
   Techno → Techno
   House → House

理由:
周波数バランス違う

2. プロ制作:
   商業リリース
   Major Label

3. 音質良い:
   WAV/FLAC推奨
   最低 320kbps MP3

4. 好きな曲:
   目標とする音

5. 最近の曲:
   2020年代
   現代的

推奨:

Techno:
Adam Beyer - Your Mind
Amelie Lens - In My Mind

House:
CamelPhat - Panic Room
Fisher - Losing It

入手先:

Beatport: WAV (最良)
Bandcamp: FLAC/WAV
Spotify: 320kbps (可)

NG:

YouTube: 圧縮
SoundCloud Free: 128kbps
古い曲: 1990年代
```

---

## セットアップ

**Ableton Live:**

### Reference Track配置

```
方法1: Audio Track

1. 新規Audio Track作成
2. 名前: "Reference"
3. Color: 赤 (目立つ)
4. Reference WAV配置
5. Arrangement View

方法2: Return Track使用

NG:
混乱する

推奨:
方法1 (Audio Track)

配置位置:

Track順序:
最上部
または
Master直下

理由:
見やすい
すぐMute可能

重要設定:

Output: Master
Monitor: Off
Solo: 可能に
```

---

## Volume Matching

**音量合わせ:**

### なぜ必要？

```
問題:

Reference: -14 LUFS (大きい)
自分: -20 LUFS (小さい)

比較:
Reference大きく聴こえる
→ 良く聴こえる (錯覚)

解決:

Volume Matching:
同じ音量で比較

方法:

Reference Track:
Utility挿入
Gain: 調整

目標:
自分と同じ LUFS
```

### 実践手順

```
Step 1: LUFS測定

LUFS Meter挿入:

1. Master Track
2. Youlean Loudness Meter

再生:

自分の曲: -20 LUFS
Reference: -14 LUFS

差: 6 dB

Step 2: Reference調整

Reference Track:

Utility挿入
Gain: -6 dB

結果:
両方 -20 LUFS

Step 3: 確認

切り替え:
同じ音量感

これで:
客観的比較可能
```

---

## LUFS比較

**ラウドネス:**

### 測定

```
ツール:

Youlean Loudness Meter:
無料
Master Trackに

測定値:

Integrated LUFS:
全体平均

Short-term LUFS:
3秒平均

Momentary LUFS:
0.4秒

推奨:
Integrated LUFS使用

目標値:

Reference: -14 LUFS
自分: -14 LUFS (Mix段階では-16〜-20 LUFS)

理由:
マスタリング後 -14 LUFS

Mix段階:

-18 LUFS程度OK
マスタリングで+4 dB

確認頻度:

Mix中: 30分ごと
完成前: 必須
```

---

## Spectrum比較

**周波数バランス:**

### 視覚的確認

```
設定:

Master Track:
Spectrum挿入

表示:
常時表示

比較方法:

1. Reference再生
   Spectrum確認
   スクリーンショット

2. 自分の曲再生
   Spectrum確認

3. 比較:
   並べて見る

確認ポイント:

Low (20-120 Hz):

Reference: 太い？
自分: 同じくらい？

調整:
Kick・Bass +2〜+3 dB @ 60-80 Hz

Low-Mid (250-500 Hz):

Reference: 控えめ？
自分: 盛り上がりすぎ？

調整:
全トラック -2〜-4 dB

Mid (1-3 kHz):

Reference: 適度？
自分: 少ない？

調整:
Lead・Vocal +2〜+3 dB

High (10-20 kHz):

Reference: 適度にある？
自分: 少ない？

調整:
Master High Shelf +1 dB

全体形状:

理想:
やや右下がり

Reference同じ:
OK
```

---

## A/B切り替え

**瞬時比較:**

### 方法

```
推奨方法:

1. Reference Track Mute

2. 自分の曲再生

3. 同じ箇所でReference Un-Mute
   自分の曲 Mute

4. 瞬時切り替え

5. 繰り返し

比較箇所:

Drop:
最も情報量多い

Verse:
バランス確認

Intro:
スペース確認

推奨:
同じセクション比較

頻度:

Mix中: 10分ごと
完成前: 最低5回
```

---

## 聴感比較

**耳で確認:**

### チェックポイント

```
1. 低域:

Question:
Reference同じくらい太い？

調整:
Kick・Bass EQ
Low Shelf +2 dB

2. 明瞭度:

Question:
Vocal・Lead明確？

調整:
Peak +3 dB @ 3 kHz

3. 広がり:

Question:
Stereo幅同じ？

調整:
Width 100-120%

4. 奥行き:

Question:
前後関係明確？

調整:
Reverb Send

5. 音圧:

Question:
パワフル？

調整:
Compression
Mastering

6. バランス:

Question:
全楽器聴こえる？

調整:
Volume Balance
```

---

## 具体的な調整例

**問題別解決:**

### 問題1: 低域弱い

```
Reference:
Kick太い、パワフル

自分:
Kick薄い

診断:

Spectrum確認:
20-120 Hz 少ない

解決:

Kick Track:

EQ Eight:
Low Shelf +3 dB @ 60 Hz

Compressor:
Ratio 4:1
GR -4 dB

Saturator:
Drive 3 dB

確認:
Reference並び？
```

### 問題2: 濁る

```
Reference:
クリア、分離良い

自分:
濁る

診断:

Spectrum確認:
250-500 Hz 盛り上がり

解決:

全トラック:

EQ Eight:
Peak -2〜-4 dB @ 300-500 Hz

確認:
劇的改善
```

### 問題3: Vocal埋もれる

```
Reference:
Vocal明確、前

自分:
Vocal埋もれる

診断:

Volume不足？
EQ不足？

解決:

Vocal Track:

Volume: +2 dB

EQ Eight:
Peak +3 dB @ 3 kHz (明瞭度)
High Shelf +1.5 dB @ 10 kHz

Compressor:
Ratio 3:1
GR -4 dB

確認:
前に出た？
```

### 問題4: 狭い

```
Reference:
Wide、広い

自分:
狭い

診断:

Stereo幅不足

解決:

Pad Track:
Width 80-100%

FX Track:
Width 120%

Master:
Width 105-110%
Bass Mono: On (120 Hz)

確認:
広がり？
```

---

## Multiple References

**複数比較:**

### 推奨

```
Reference 3曲:

1. 最も近い目標
2. 異なるアプローチ
3. 自分の前作 (ある場合)

メリット:

客観的:
1曲のみ: 偏る
3曲: バランス見える

多様:
様々なアプローチ

方法:

Track 1: Reference A
Track 2: Reference B
Track 3: Reference C

切り替え:
それぞれMute/Un-Mute

推奨:
最低2曲
```

---

## よくある間違い

### 1. Volume Matchingなし

```
問題:
大きい方が良く聴こえる

解決:

必須:
Volume Matching

LUFS:
同じにする

理由:
客観的比較
```

### 2. 違うジャンル

```
問題:
バランス全く違う

例:

Rock Reference
→ Techno制作
NG

解決:

同じジャンル:
必須

理由:
周波数バランス
ミックスアプローチ
全く違う
```

### 3. Reference聴きすぎ

```
問題:
自分の個性失う
完全コピー

解決:

Reference:
参考のみ
目標レベル確認

自分:
個性維持

バランス:
70% 自分
30% Reference
```

### 4. 比較頻度少ない

```
問題:
完成後に気づく
手遅れ

解決:

頻度:
10-15分ごと

タイミング:
各セクション完了時

理由:
早期発見
修正簡単
```

---

## Reference Mixing Workflow

**30分で完成:**

### Step-by-Step

```
0-5分: Setup

1. Reference Track配置
2. Volume Matching
3. Spectrum表示

5-15分: LUFS・Spectrum比較

1. Integrated LUFS確認
2. Spectrum形状比較
3. 問題特定

15-25分: 調整

1. 低域: EQ調整
2. Mid: 明瞭度
3. High: 空気感
4. Width: ステレオ

25-30分: 確認

1. A/B切り替え
2. 各セクション比較
3. 最終チェック
```

---

## まとめ

### Reference Mixing

```
□ 同じジャンル必須
□ Volume Matching必須
□ 10分ごと比較
□ LUFS・Spectrum確認
□ A/B切り替え
```

### 選び方

```
条件:
- 同じジャンル
- プロ制作
- WAV/FLAC (320kbps MP3可)
- 好きな曲
- 最近の曲
```

### 重要原則

```
□ 客観的比較
□ 主観 70% + Reference 30%
□ 個性維持
□ 早期・頻繁に比較
□ 複数Reference推奨
```

---

**次は:** [Mixing Workflow](./mixing-workflow.md) - 完全なミキシング手順の集大成
