# Frequency Balance

周波数分離でクリアなミックスを実現。EQの実践、Spectrum確認を完全マスターします。

## この章で学ぶこと

- 周波数帯域の役割
- 全トラックHigh Pass必須
- Low-Mid (250-500 Hz) 処理
- Spectrum活用法
- トラック別EQ設定
- マスキング問題の解決
- よくある失敗と対処法

---

## なぜFrequency Balanceが重要なのか

**明瞭度の核心:**

```
周波数バランス悪い:

特徴:
濁る
埋もれる
分離悪い

原因:
周波数衝突
特に Low-Mid

周波数バランス良い:

特徴:
クリア
分離良い
明瞭

方法:
EQ Eight
全トラック処理

プロとアマの差:

アマ:
EQ: 一部のみ
適当

プロ:
EQ: 全トラック
100%

結果:
プロ: クリア、分離
アマ: 濁る

真実:
「良いミックス」=
周波数分離が完璧
```

---

## 周波数帯域の役割

**20 Hz - 20 kHz:**

### 帯域別特徴

```
Sub Bass (20-60 Hz):

特徴:
体で感じる
聴こえにくい

担当:
Kick・Bass

処理:
他は全てカット
High Pass 30-40 Hz

Low (60-250 Hz):

特徴:
パワー
温かみ
濁りやすい

担当:
Kick・Bass・Tom

処理:
注意深く
+2〜+3 dB (Kick・Bass)
他はカット

Low-Mid (250-500 Hz):

特徴:
最も濁りやすい
問題多い

担当:
全楽器が集中

処理:
-2〜-4 dB (ほぼ全トラック)
最重要

Mid (500 Hz-2 kHz):

特徴:
存在感
前に出る

担当:
Vocal・Snare・Guitar・Lead

処理:
+2〜+3 dB (主要楽器)
0〜-2 dB (その他)

Upper-Mid (2-5 kHz):

特徴:
明瞭度
アタック

担当:
Vocal・Snare・Lead

処理:
+2〜+3 dB (明瞭度)
De-ess (Vocal 6-8 kHz)

Presence (5-10 kHz):

特徴:
空気感
明るさ

担当:
Hi-Hat・Cymbals

処理:
+1〜+2 dB
刺さり注意

Air (10-20 kHz):

特徴:
高級感
開放感

担当:
全体

処理:
High Shelf +0.5〜+1.5 dB
控えめ
```

---

## High Pass必須

**最重要処理:**

### 全トラックに適用

```
原則:

Kick・Bass以外:
全てHigh Pass

理由:

不要低域:
濁りの原因

低域スペース:
Kick・Bass専用

推奨設定:

Kick:
High Pass 30 Hz (24 dB/oct)
不要な超低域のみ

Bass:
High Pass 40 Hz (24 dB/oct)

Snare:
High Pass 200 Hz (12 dB/oct)

Hi-Hat:
High Pass 6000 Hz (12 dB/oct)
極端

Lead:
High Pass 200 Hz (12 dB/oct)

Vocal:
High Pass 80 Hz (18 dB/oct)

Pad:
High Pass 300 Hz (12 dB/oct)
大胆に

FX:
High Pass 500 Hz (12 dB/oct)

効果:
劇的にクリア
最も効果的な処理
```

---

## Low-Mid処理

**濁り除去:**

### 250-500 Hz問題

```
なぜ濁る？

全楽器:
この帯域に倍音

結果:
混雑
濁り

解決:

全トラック:
Peak EQ -2〜-4 dB

頻度:
Q: 1.5-2.0 (やや広い)

例外:

Kick:
この帯域重要
触らない

Bass:
-2 dB程度 (軽く)

推奨設定:

Snare:
-2 dB @ 400 Hz
Q: 2.0

Lead:
-3 dB @ 300 Hz
Q: 1.5

Vocal:
-3 dB @ 300 Hz
Q: 2.0

Pad:
-4 dB @ 500 Hz
Q: 1.5

効果:
驚くほどクリア
```

---

## Spectrum活用

**視覚的確認:**

### Spectrumデバイス

```
挿入:

Browser > Audio Effects > Spectrum
Master Trackに

表示:

横軸: 周波数 (Hz)
縦軸: 音量 (dB)

色:
明るい = 大きい
暗い = 小さい

確認ポイント:

低域 (20-120 Hz):
Kick・Bass明確？
他の楽器入ってない？

Low-Mid (250-500 Hz):
盛り上がりすぎ？
濁りの原因

Mid (1-3 kHz):
Lead・Vocal明確？

High (10-20 kHz):
適度にある？

理想的な形:

全体:
やや右下がり

低域:
太い

Mid:
適度

High:
控えめだが存在
```

---

## マスキング問題

**楽器が埋もれる:**

### 原理

```
マスキング:

定義:
同じ周波数の楽器
大きい方が小さい方を隠す

例:

Kick (60 Hz):
大きい

Bass (60 Hz):
隠れる

結果:
Bassが聴こえない

解決法:

1. 周波数ずらし:
   Bass: 80 Hz中心
   Kick: 60 Hz中心

2. EQ:
   Bass: -2 dB @ 60 Hz
   Kick: +3 dB @ 60 Hz

3. サイドチェイン:
   Kick鳴る時 → Bass下がる

推奨:
3つ全て使用
```

---

## トラック別EQ設定

**実践例:**

### Kick

```
目標:
太い、明確、低域専用

EQ設定:

Band 1 (High Pass):
Freq: 30 Hz
Slope: 24 dB/oct
理由: 不要超低域カット

Band 2 (Low Shelf):
Freq: 60 Hz
Gain: +3 dB
理由: パワー

Band 3 (Peak):
Freq: 250 Hz
Gain: -3 dB
Q: 2.0
理由: 濁り除去

Band 4 (Peak):
Freq: 4 kHz
Gain: +2 dB
Q: 1.5
理由: アタック明確

Band 5 (High Cut):
Freq: 10 kHz
Slope: 12 dB/oct
理由: 不要高域カット
```

### Bass

```
目標:
太い、クリア、Kickと分離

EQ設定:

High Pass:
40 Hz (24 dB/oct)

Low Shelf:
80 Hz, +2 dB

Peak (分離):
250 Hz, -2 dB, Q: 2.0
Kickと分離

Peak (存在感):
1 kHz, +2 dB, Q: 1.5

High Cut:
5 kHz (18 dB/oct)
```

### Vocal

```
目標:
明瞭、前に出る、自然

EQ設定:

High Pass:
80 Hz (18 dB/oct)

Peak (こもり除去):
300 Hz, -3 dB, Q: 2.0

Peak (明瞭度):
3 kHz, +3 dB, Q: 1.5

Peak (De-ess):
7 kHz, -4 dB, Q: 3.0
歯擦音除去

High Shelf:
10 kHz, +1.5 dB
空気感
```

### Pad

```
目標:
広い、後ろに、濁らない

EQ設定:

High Pass:
300 Hz (12 dB/oct)
大胆に

Peak:
500 Hz, -4 dB, Q: 1.5
濁り除去

High Cut:
8 kHz (12 dB/oct)
暗く、後ろに
```

---

## 周波数分離戦略

**スペース作り:**

### 帯域別担当

```
20-120 Hz:

専用:
Kick・Bass

他:
全てカット

120-250 Hz:

主:
Bass・Tom

副:
わずかに他楽器

処理:
他は-2 dB程度

250-500 Hz:

全楽器:
倍音あり

処理:
全て-2〜-4 dB
最重要

500 Hz-2 kHz:

主:
Lead・Vocal・Snare

処理:
主 +2〜+3 dB
副 0〜-2 dB

2-5 kHz:

主:
Vocal・Snare

処理:
明瞭度 +2〜+3 dB

5-10 kHz:

主:
Hi-Hat・Cymbals

処理:
適度に +1〜+2 dB

10-20 kHz:

全体:
High Shelf

処理:
+0.5〜+1.5 dB
```

---

## Dynamic EQ

**発展技術:**

### 概念

```
通常EQ:

固定:
常に同じ

Dynamic EQ:

変化:
音量により変化

用途:

De-ess:
大きい時だけカット

Bass制御:
過剰な時だけ

Ableton:

Multiband Dynamics:
帯域別Compressor
Dynamic EQ的

推奨:
上級者向け
```

---

## よくある失敗

### 1. High Passなし

```
問題:
濁る

原因:
不要低域残る

解決:

全トラック:
High Pass必須

Kick・Bass:
30-40 Hz

他:
200-300 Hz以上

効果:
劇的改善
```

### 2. Low-Mid放置

```
問題:
最も濁る

原因:
250-500 Hz混雑

解決:

全トラック:
-2〜-4 dB @ 300-500 Hz

例外:
Kick・Bass (軽く)

効果:
クリア
```

### 3. EQブースト過剰

```
問題:
不自然
刺さる

原因:
+6 dB以上

解決:

ブースト:
最大 +3〜+4 dB

推奨:
カット優先
他をカット → 相対的ブースト
```

### 4. Spectrumで確認しない

```
問題:
感覚頼り
バランス悪い

解決:

Spectrum:
常に表示

確認:
全帯域バランス

理想:
やや右下がり
```

---

## EQ順序

**Chain内配置:**

### 推奨順序

```
Chain:

1. Utility (Gain In)
2. EQ Eight (Cut) ← カット専用
3. Compressor
4. EQ Eight (Boost) ← ブースト専用
5. Saturator
6. その他
7. Utility (Gain Out)

理由:

カット先:
不要除去

Comp中:
クリーンな信号

ブースト後:
補正・調整

効果:
効率的
クリーン
```

---

## 実践ワークフロー

**30分で完成:**

### Step-by-Step

```
0-10分: 全トラックHigh Pass

1. Kick: 30 Hz
2. Bass: 40 Hz
3. Snare: 200 Hz
4. Lead: 200 Hz
5. Pad: 300 Hz
6. 他全て

10-20分: Low-Mid処理

1. Spectrum確認
2. 全トラック -2〜-4 dB @ 300-500 Hz
3. 再度Spectrum確認

20-25分: Mid・High

1. Lead・Vocal: +3 dB @ 3 kHz
2. Snare: +2 dB @ 3 kHz
3. 全体 High Shelf: +1 dB @ 10 kHz

25-30分: 確認

1. Spectrum全体確認
2. A/B比較
3. 各トラック Solo確認
```

---

## まとめ

### Frequency Balance

```
□ 全トラック High Pass必須
□ Low-Mid -2〜-4 dB (全トラック)
□ ブースト最大 +3〜+4 dB
□ Spectrum常時確認
□ カット優先
```

### 帯域別処理

```
20-120 Hz: Kick・Bass専用
250-500 Hz: 全て-2〜-4 dB (最重要)
2-5 kHz: 明瞭度 +2〜+3 dB
10-20 kHz: High Shelf +1 dB
```

### 重要原則

```
□ 視覚的確認 (Spectrum)
□ カット > ブースト
□ マスキング回避
□ 帯域分離
□ 全トラック処理
```

---

**次は:** [Stereo Imaging](./stereo-imaging.md) - ステレオ空間で広がりを作る
