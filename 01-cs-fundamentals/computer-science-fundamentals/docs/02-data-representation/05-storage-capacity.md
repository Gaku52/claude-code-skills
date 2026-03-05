# ストレージ容量 --- データ量の感覚とシステム設計のための容量計算

> 「ギガバイト」「テラバイト」と聞いて具体的なイメージが浮かぶようになれば、システム設計の見積もりは格段に正確になる。
> ストレージ容量を正しく理解することは、コスト効率の良いシステムを構築する第一歩である。

## この章で学ぶこと

- [ ] データ量の単位（SI接頭辞と2進接頭辞）の違いを正確に理解する
- [ ] 各種データ（テキスト・画像・音声・動画）の典型的なサイズを知る
- [ ] ストレージ技術の歴史的進化と原理を理解する
- [ ] ストレージ/帯域幅の見積もりができる
- [ ] システム設計における容量計算（Back-of-the-envelope Estimation）ができる
- [ ] 階層型ストレージとコスト最適化の設計ができる

## 前提知識

- データサイズの単位 → 参照: [[00-binary-and-number-systems.md]]
- 圧縮の基礎 → 参照: [[04-compression-algorithms.md]]

---

## 1. データ量の単位体系

### 1.1 SI接頭辞と2進接頭辞の区別

コンピュータの世界でデータ量を扱う際に最も混乱を招くのが、「キロ」「メガ」「ギガ」といった接頭辞の解釈の二重性である。歴史的にコンピュータ業界では2の累乗（1024）を「キロ」と呼んでいたが、SI（国際単位系）の定義では「キロ」は厳密に1000を意味する。この曖昧さを解消するために、IEC（国際電気標準会議）が1998年に2進接頭辞（kibi, mebi, gibi, ...）を制定した。

```
SI接頭辞と2進接頭辞の対比:

  SI接頭辞（10の累乗）          2進接頭辞（2の累乗）
  ─────────────────────────    ─────────────────────────
  1 kB  = 10^3  = 1,000 B     1 KiB = 2^10 = 1,024 B        差: 2.4%
  1 MB  = 10^6  = 1,000,000   1 MiB = 2^20 = 1,048,576      差: 4.9%
  1 GB  = 10^9  = 10^9        1 GiB = 2^30 = 1,073,741,824  差: 7.4%
  1 TB  = 10^12 = 10^12       1 TiB = 2^40 ≈ 1.100×10^12    差: 10.0%
  1 PB  = 10^15 = 10^15       1 PiB = 2^50 ≈ 1.126×10^15    差: 12.6%
  1 EB  = 10^18               1 EiB = 2^60 ≈ 1.153×10^18    差: 15.3%
  1 ZB  = 10^21               1 ZiB = 2^70 ≈ 1.181×10^21    差: 18.1%
  1 YB  = 10^24               1 YiB = 2^80 ≈ 1.209×10^24    差: 20.9%

  注目ポイント: 単位が大きくなるほど差が拡大する。
  TBレベルで約10%の差は実務上無視できない。
```

なぜこの区別が重要なのか。ハードディスクメーカーはSI単位（10の累乗）でラベルを付けるが、OSは2進単位（2の累乗）で表示する。結果として「500GBのHDDを買ったのに、OSでは465GBしか認識されない」という現象が起きる。これは詐欺ではなく、単位系の違いに起因する。

```
HDD容量表記のギャップ:

  メーカー表記    OS表示（2進）    差分
  ──────────    ────────────    ─────
   250 GB  →     232.83 GiB    -6.9%
   500 GB  →     465.66 GiB    -6.9%
   1 TB    →     931.32 GiB    -6.9%
   2 TB    →    1862.65 GiB    -6.9%
   4 TB    →    3725.29 GiB    -6.9%
  16 TB    →   14901.16 GiB    -6.9%

  計算式: OS表示 = メーカー表記 × 10^9 / 2^30
         = メーカー表記 × 1,000,000,000 / 1,073,741,824
         = メーカー表記 × 0.93132...
```

### 1.2 ビット・バイト・ワードの関係

```
データ量の基本単位の階層:

  ┌─────────────────────────────────────────────────┐
  │  1 ビット (bit, b)                                │
  │  = 0 または 1 の2値                               │
  │  情報理論における最小単位                           │
  ├─────────────────────────────────────────────────┤
  │  1 ニブル (nibble)                                │
  │  = 4 ビット                                      │
  │  16進数1桁に対応 (0x0〜0xF)                       │
  ├─────────────────────────────────────────────────┤
  │  1 バイト (byte, B)                               │
  │  = 8 ビット                                      │
  │  ASCII 1文字に対応。アドレス可能な最小単位          │
  ├─────────────────────────────────────────────────┤
  │  1 ワード (word)                                  │
  │  = CPUアーキテクチャ依存                           │
  │  32ビットCPU: 4バイト / 64ビットCPU: 8バイト        │
  ├─────────────────────────────────────────────────┤
  │  1 ダブルワード (dword) = 2ワード                  │
  │  1 クワッドワード (qword) = 4ワード                │
  └─────────────────────────────────────────────────┘

  なぜバイトは8ビットなのか:
  - 歴史的には6ビットバイトや9ビットバイトのマシンも存在した
  - IBM System/360 (1964年) が8ビットバイトを採用し事実上の標準に
  - 8ビット = 256通り → ASCII文字セット(128) + 拡張(128) に十分
  - 8 = 2^3 なのでビット操作が自然に行える

  ネットワーク速度の注意点:
  - ネットワーク速度: ビット毎秒 (bps) で表記
  - ファイルサイズ: バイト (B) で表記
  - 1 Gbps = 1,000,000,000 bits/s = 125,000,000 B/s ≈ 125 MB/s
  - つまり Gbps の値を 8 で割るとバイト毎秒になる
```

---

## 2. データサイズの直感

システム設計における容量計算の基礎は、各種データの典型的なサイズを「直感的に」把握していることにある。以下では主要なデータ型ごとに、生のサイズからフォーマット別サイズまで体系的に整理する。

### 2.1 テキストデータ

```
テキストの容量:

  文字エンコーディング別:
  ─────────────────────
  ASCII            1文字 = 1バイト（英数字・記号）
  UTF-8 (英語)     1文字 = 1バイト（ASCII互換）
  UTF-8 (日本語)   1文字 = 3バイト（ひらがな・カタカナ・漢字）
  UTF-8 (絵文字)   1文字 = 4バイト
  UTF-16 (BMP)     1文字 = 2バイト
  UTF-16 (補助面)  1文字 = 4バイト（サロゲートペア）
  UTF-32           1文字 = 4バイト（固定長）

  なぜUTF-8が主流か:
  - ASCII互換のため既存システムとの親和性が高い
  - 英語圏では最もコンパクト（1バイト/文字）
  - 可変長だが、バイト列から文字境界を一意に特定可能
  - Webの97%以上がUTF-8を使用（2024年時点）

  具体的なデータ量の目安:
  ──────────────────────
  1ツイート (280文字):       約560B〜840B
  1ページの文章:             約2KB (英語) / 約4KB (日本語)
  小説1冊 (10万字):         約300KB (UTF-8日本語)
  Wikipedia英語版全文:       約22GB (非圧縮)
  青空文庫全作品:            約2GB

  実務での目安:
  ─────────────
  JSON APIレスポンス:         1KB - 100KB
  ログ1行:                   100B - 1KB
  1日のアプリログ:            100MB - 10GB
  RDBの1レコード:             100B - 10KB
  CSVファイル (100万行):     50MB - 500MB
  ElasticSearchの1ドキュメント: 1KB - 50KB
```

### 2.2 画像データ

画像データのサイズを理解するためには、まず非圧縮画像のサイズ計算方法を理解する必要がある。非圧縮画像のサイズは「幅 x 高さ x 1ピクセルあたりのバイト数」で求まる。

```
画像サイズの計算式:

  非圧縮サイズ = 幅 × 高さ × 色深度(バイト)

  色深度の種類:
  ─────────────
  モノクロ (1bit):        1ピクセル = 0.125 バイト
  グレースケール (8bit):  1ピクセル = 1 バイト
  RGB (24bit):            1ピクセル = 3 バイト
  RGBA (32bit):           1ピクセル = 4 バイト (アルファチャネル付き)
  HDR (48bit RGB):        1ピクセル = 6 バイト

  計算例 (フルHD, RGB):
  1920 × 1080 × 3 = 6,220,800 B ≈ 5.93 MiB ≈ 6.22 MB

画像の容量（フォーマット別）:

  アイコン (32×32, PNG):          約2KB
  サムネイル (200×200, JPEG):     約10KB
  Web画像 (800×600, JPEG75):      約80KB
  フルHD写真 (1920×1080):
    非圧縮 BMP:                    約6MB
    PNG (可逆):                    約2-4MB
    JPEG (品質85):                 約500KB
    WebP (非可逆):                 約300KB
    AVIF:                          約200KB
  4K写真 (3840×2160, JPEG):       約3MB
  RAW写真 (6000×4000):            約25-50MB

  スマホ1枚の写真:                約3-5MB (HEIF/JPEG)
  1000枚の写真:                   約3-5GB

  フォーマット別圧縮率比較 (フルHD写真を基準):
  ┌──────────┬──────────┬─────────┬────────────────┐
  │ 形式      │ サイズ    │ 圧縮率   │ 特性            │
  ├──────────┼──────────┼─────────┼────────────────┤
  │ BMP      │ 6.0 MB   │ 1.0x    │ 非圧縮          │
  │ PNG      │ 3.0 MB   │ 2.0x    │ 可逆圧縮        │
  │ JPEG 95  │ 1.5 MB   │ 4.0x    │ ほぼ劣化なし     │
  │ JPEG 85  │ 500 KB   │ 12.0x   │ 一般的品質       │
  │ JPEG 50  │ 150 KB   │ 40.0x   │ 明らかに劣化     │
  │ WebP     │ 300 KB   │ 20.0x   │ JPEG後継        │
  │ AVIF     │ 200 KB   │ 30.0x   │ 最新、高圧縮     │
  │ HEIF     │ 250 KB   │ 24.0x   │ Apple標準       │
  └──────────┴──────────┴─────────┴────────────────┘

  実務での目安:
  ─────────────
  ユーザーアバター:         50KB - 200KB
  ECサイト商品画像:         100KB - 500KB
  LP/バナー画像:            200KB - 2MB
  Webページ画像合計:        500KB - 5MB (目標: 1MB以下)
  OGP画像:                 100KB - 300KB (1200×630推奨)
```

### 2.3 音声データ

```
音声データの計算式:

  非圧縮サイズ/秒 = サンプリングレート × ビット深度 / 8 × チャネル数

  例 (CD品質):
  44,100 × 16 / 8 × 2 = 176,400 B/秒 ≈ 172 KiB/秒 ≈ 10.1 MiB/分

  サンプリングレートの意味:
  - 8 kHz:   電話品質（人の声を伝える最低限）
  - 22 kHz:  AM ラジオ品質
  - 44.1 kHz: CD品質（ナイキスト定理: 人の可聴域 20kHz の2倍以上）
  - 48 kHz:  DVD / Blu-ray / プロ用
  - 96 kHz:  ハイレゾ
  - 192 kHz: ハイレゾ最高品質

  なぜ 44.1kHz なのか:
  人間の可聴域は約 20Hz〜20kHz。ナイキスト・シャノンの標本化定理により、
  元の信号を完全に復元するには最高周波数の2倍以上でサンプリングする必要がある。
  20kHz × 2 = 40kHz 以上が必要。44.1kHz はこの条件を満たす最小の標準値として
  CDに採用された（歴史的にはビデオテープ録音の制約にも由来する）。

音声の容量:

  電話品質 (8kHz, 8bit, mono):      8 KB/秒
  CD品質 (44.1kHz, 16bit, stereo):  176 KB/秒 = 10.6 MB/分

  MP3 (128kbps):     約1MB/分 → 1曲(4分) ≈ 4MB
  MP3 (320kbps):     約2.4MB/分
  AAC (256kbps):     約2MB/分
  Opus (128kbps):    約1MB/分（品質はMP3 192kbps相当）
  FLAC (可逆):       約5MB/分 → 1曲 ≈ 20MB
  WAV (非圧縮):      約10MB/分

  Podcast 1時間 (MP3 128kbps):      約60MB
  音楽ライブラリ 1000曲:            約4GB (MP3) / 20GB (FLAC)
  Apple Music / Spotify の1曲:      約10MB (AAC 256kbps)

  音声コーデック比較:
  ┌───────────┬──────────┬─────────┬──────────────────┐
  │ コーデック  │ ビットレート│ 品質     │ 用途              │
  ├───────────┼──────────┼─────────┼──────────────────┤
  │ Opus      │ 32kbps   │ 良      │ VoIP、通話        │
  │ AAC       │ 128kbps  │ 優      │ ストリーミング     │
  │ MP3       │ 192kbps  │ 優      │ ダウンロード       │
  │ Vorbis    │ 160kbps  │ 良      │ ゲーム、Web       │
  │ FLAC      │ ~900kbps │ 完全    │ アーカイブ         │
  └───────────┴──────────┴─────────┴──────────────────┘
```

### 2.4 動画データ

```
動画データの計算式:

  非圧縮サイズ/秒 = 幅 × 高さ × 色深度(B) × フレームレート

  例 (1080p, 24bit, 30fps):
  1920 × 1080 × 3 × 30 = 186,624,000 B/秒 ≈ 178 MiB/秒

  1分 ≈ 10.4 GiB（非圧縮）
  → H.264圧縮後: 約35MB（3000倍以上の圧縮！）

  解像度の階層:
  ──────────────
  SD   (720×480):     約 1.0M ピクセル
  HD   (1280×720):    約 0.9M ピクセル
  FHD  (1920×1080):   約 2.1M ピクセル
  QHD  (2560×1440):   約 3.7M ピクセル
  4K   (3840×2160):   約 8.3M ピクセル
  8K   (7680×4320):   約 33.2M ピクセル

動画の容量:

  非圧縮 1080p 30fps:           約186 MB/秒

  YouTube品質別 (1分あたり):
    360p:     約5MB
    720p:     約15MB
    1080p:    約35MB
    4K:       約100MB

  Netflix (1時間):
    SD:      約0.7GB
    HD:      約3GB
    4K HDR:  約7GB

  Zoom通話 (1時間):
    音声のみ:  約40MB
    ビデオ720p: 約800MB

  動画コーデックの世代比較 (同画質での1080p 1分あたり):
  ┌──────────┬──────────┬──────────┬─────────────────┐
  │ コーデック │ 登場年    │ サイズ    │ 備考             │
  ├──────────┼──────────┼──────────┼─────────────────┤
  │ MPEG-2   │ 1995     │ 80MB     │ DVD時代          │
  │ H.264    │ 2003     │ 35MB     │ 現在の主流        │
  │ H.265    │ 2013     │ 18MB     │ 4K配信向け        │
  │ VP9      │ 2013     │ 20MB     │ YouTube使用      │
  │ AV1      │ 2018     │ 15MB     │ ロイヤリティフリー │
  └──────────┴──────────┴──────────┴─────────────────┘

  実務での目安:
  ─────────────
  30秒のプロモ動画:       5-20MB (Web用)
  YouTube 10分動画:       100-500MB (元ファイル)
  映画1本 (4K):           20-50GB (Blu-ray)
  監視カメラ (1080p, 24h): 約15-30GB/日 (H.265)
```

---

## 3. ストレージ技術の進化

### 3.1 ストレージ技術の歴史的変遷

```
ストレージ密度の進化（1956年〜現在）:

  年代    技術                   容量/単価       容量密度
  ────────────────────────────────────────────────────────
  1956    IBM 350 (磁気ドラム)   5 MB/$500,000   3.75 MB/m^3
  1971    IBM 3330 (磁気ディスク) 200 MB/$1M     洗濯機サイズ
  1980    Seagate ST-506 (HDD)   5 MB/$1,500     5.25インチ
  1984    CD-ROM                 650 MB          直径12cm
  1995    DVD                    4.7 GB          直径12cm
  1998    IBM Microdrive         170 MB          1インチ
  2003    Blu-ray               25 GB            直径12cm
  2006    SSD 商用化             32 GB           2.5インチ
  2015    Samsung 850 EVO       2 TB             2.5インチ
  2020    Seagate HAMR HDD      20 TB           3.5インチ
  2023    Samsung PM1743 SSD    30.72 TB        2.5インチ
  2024    100TB+ HDD開発中       100 TB          3.5インチ

  容量単価の劇的低下:
  ┌──────┬────────────────┬──────────────┐
  │ 年    │ $/GB (HDD)     │ $/GB (SSD)   │
  ├──────┼────────────────┼──────────────┤
  │ 1980 │ $300,000       │ (未登場)      │
  │ 1990 │ $10,000        │ (未登場)      │
  │ 2000 │ $10            │ (未登場)      │
  │ 2005 │ $1             │ $50          │
  │ 2010 │ $0.10          │ $3           │
  │ 2015 │ $0.035         │ $0.50        │
  │ 2020 │ $0.020         │ $0.12        │
  │ 2024 │ $0.015         │ $0.06        │
  └──────┴────────────────┴──────────────┘

  40年間でHDDの容量単価は約2000万分の1に低下した。
  これはムーアの法則をさらに上回るペースである（Kryder's Law）。
```

### 3.2 現代のストレージ技術比較

```
主要ストレージ技術の比較:

  ┌──────────┬───────┬─────────┬────────────┬───────────┬────────────┐
  │ 技術      │ 速度   │ 耐久性   │ 容量/筐体   │ $/TB     │ 用途        │
  ├──────────┼───────┼─────────┼────────────┼───────────┼────────────┤
  │ DRAM     │ 超高速 │ 揮発性   │ 〜512GB    │ $3,000+  │ メインメモリ │
  │ Optane   │ 極高速 │ 不揮発性  │ 〜512GB    │ $1,500   │ 永続メモリ  │
  │ NVMe SSD │ 高速   │ 不揮発性  │ 〜30TB     │ $60-150  │ ホットデータ │
  │ SATA SSD │ 中速   │ 不揮発性  │ 〜8TB      │ $40-80   │ ウォーム    │
  │ HDD      │ 低速   │ 不揮発性  │ 〜24TB     │ $15-25   │ 大容量保存  │
  │ テープ    │ 極低速 │ 不揮発性  │ 〜45TB/巻  │ $3-5     │ アーカイブ  │
  │ 光ディスク │ 低速   │ 不揮発性  │ 〜128GB    │ $10-30   │ 長期保存    │
  └──────────┴───────┴─────────┴────────────┴───────────┴────────────┘

  IOPS (Input/Output Operations Per Second) の比較:
  ┌──────────┬────────────┬────────────┬────────────┐
  │ 技術      │ ランダム読取 │ ランダム書込 │ レイテンシ   │
  ├──────────┼────────────┼────────────┼────────────┤
  │ DRAM     │ 無制限に近い │ 無制限に近い │ 〜100ns     │
  │ NVMe SSD │ 500K-1M    │ 200K-500K  │ 〜20μs     │
  │ SATA SSD │ 75K-100K   │ 50K-75K    │ 〜100μs    │
  │ HDD      │ 100-200    │ 100-200    │ 〜5-10ms   │
  │ テープ    │ N/A (順次)  │ N/A (順次)  │ 数秒〜分    │
  └──────────┴────────────┴────────────┴────────────┘

  注目: HDDとNVMe SSDのIOPS差は約5000倍。
  これはランダムアクセスが支配的なワークロードで圧倒的な差を生む。
```

### 3.3 SSD の内部構造と寿命

SSD（Solid State Drive）はNANDフラッシュメモリを使用する。NANDセルには1セルあたりの記録ビット数によって種類がある。

```
NANDフラッシュの種類:

  ┌──────┬──────────┬─────────┬───────────┬───────────────────┐
  │ 種類  │ bit/cell │ 書換寿命 │ 速度      │ 用途               │
  ├──────┼──────────┼─────────┼───────────┼───────────────────┤
  │ SLC  │ 1        │ 100K回  │ 最速      │ エンタープライズ     │
  │ MLC  │ 2        │ 10K回   │ 速い      │ サーバー             │
  │ TLC  │ 3        │ 3K回    │ 普通      │ コンシューマ/サーバー │
  │ QLC  │ 4        │ 1K回    │ やや遅い  │ 読取重視ワークロード │
  │ PLC  │ 5        │ <1K回   │ 遅い      │ アーカイブ（開発中）  │
  └──────┴──────────┴─────────┴───────────┴───────────────────┘

  なぜビット数が増えると寿命が短くなるのか:
  - SLCは電圧の閾値が1つ（0/1の2状態）→ マージンが大きい
  - QLCは電圧の閾値が15個（16状態）→ マージンが極めて小さい
  - 書き込みのたびに酸化膜が劣化し、閾値のマージンがさらに減少
  - マージンが消失すると正しいビット値を読めなくなる

  SSDの寿命指標:
  ──────────────
  TBW (Terabytes Written): 生涯書き込み量
  DWPD (Drive Writes Per Day): 1日あたり何回全容量を書き換えられるか

  例: Samsung 870 EVO 1TB → TBW 600TB
  → 1日100GB書き込みなら: 600TB / 100GB = 6000日 ≈ 16.4年
  → 1日1TB書き込みなら: 600日 ≈ 1.6年

  Write Amplification (書き込み増幅):
  SSDではガベージコレクションやウェアレベリングにより
  実際の書き込み量が論理書き込み量の数倍になることがある。
  WAF = 物理書き込み量 / 論理書き込み量
  一般的なWAFは2〜5倍。これを考慮して寿命を見積もる必要がある。
```

---

## 4. コード例: ストレージ容量計算の実践

### コード例1: データサイズの単位変換ツール (Python)

```python
#!/usr/bin/env python3
"""
データサイズの単位変換ツール
SI接頭辞と2進接頭辞の両方に対応し、相互変換を行う。
"""


def bytes_to_human(size_bytes: int, binary: bool = True) -> str:
    """
    バイト数を人間が読みやすい形式に変換する。

    Args:
        size_bytes: 変換するバイト数
        binary: Trueなら2進接頭辞(KiB,MiB,...), Falseなら10進(KB,MB,...)

    Returns:
        人間が読みやすい文字列表現
    """
    if size_bytes < 0:
        raise ValueError("サイズは0以上である必要があります")
    if size_bytes == 0:
        return "0 B"

    if binary:
        base = 1024
        suffixes = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
    else:
        base = 1000
        suffixes = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]

    # 適切な単位を見つける
    unit_index = 0
    size_float = float(size_bytes)
    while size_float >= base and unit_index < len(suffixes) - 1:
        size_float /= base
        unit_index += 1

    # 小数点以下の桁数を調整（大きい値は整数で表示、小さい値は小数で）
    if size_float >= 100:
        return f"{size_float:.0f} {suffixes[unit_index]}"
    elif size_float >= 10:
        return f"{size_float:.1f} {suffixes[unit_index]}"
    else:
        return f"{size_float:.2f} {suffixes[unit_index]}"


def human_to_bytes(size_str: str) -> int:
    """
    人間が読みやすい形式からバイト数に変換する。

    Args:
        size_str: "10GB", "1.5 TiB", "500 MB" などの文字列

    Returns:
        バイト数（整数）
    """
    import re

    size_str = size_str.strip()
    match = re.match(r"^([\d.]+)\s*([A-Za-z]+)$", size_str)
    if not match:
        raise ValueError(f"パースできない形式: {size_str}")

    value = float(match.group(1))
    unit = match.group(2).upper()

    # 2進接頭辞のマッピング
    binary_units = {
        "B": 1,
        "KIB": 2**10, "MIB": 2**20, "GIB": 2**30,
        "TIB": 2**40, "PIB": 2**50, "EIB": 2**60,
    }
    # SI接頭辞のマッピング
    si_units = {
        "B": 1,
        "KB": 10**3, "MB": 10**6, "GB": 10**9,
        "TB": 10**12, "PB": 10**15, "EB": 10**18,
    }

    if unit in binary_units:
        return int(value * binary_units[unit])
    elif unit in si_units:
        return int(value * si_units[unit])
    else:
        raise ValueError(f"未知の単位: {unit}")


def show_size_comparison(size_bytes: int) -> None:
    """SI接頭辞と2進接頭辞の両方でサイズを表示し、差分を可視化する。"""
    print(f"  元の値: {size_bytes:,} バイト")
    print(f"  SI表記:     {bytes_to_human(size_bytes, binary=False)}")
    print(f"  2進表記:    {bytes_to_human(size_bytes, binary=True)}")

    # TB以上の場合、差分のパーセンテージを表示
    if size_bytes >= 10**12:
        si_tb = size_bytes / 10**12
        bin_tib = size_bytes / 2**40
        diff_pct = (si_tb - bin_tib) / bin_tib * 100
        print(f"  差分: SI表記はバイナリ表記より {diff_pct:.1f}% 大きく見える")


if __name__ == "__main__":
    print("=== データサイズ変換デモ ===\n")

    # 典型的なストレージサイズでの比較
    test_sizes = [
        ("USBメモリ", 16 * 10**9),          # 16 GB (メーカー表記)
        ("SSD", 500 * 10**9),               # 500 GB
        ("HDD", 4 * 10**12),                # 4 TB
        ("エンタープライズ", 100 * 10**12),   # 100 TB
    ]

    for name, size in test_sizes:
        print(f"--- {name} ({bytes_to_human(size, binary=False)} 表記) ---")
        show_size_comparison(size)
        print()

    # 文字列からの変換テスト
    print("=== 文字列→バイト変換 ===\n")
    test_strings = ["1.5 GiB", "500 MB", "2 TB", "100 KiB"]
    for s in test_strings:
        result = human_to_bytes(s)
        print(f"  {s:>10} = {result:>20,} バイト = {bytes_to_human(result)}")
```

### コード例2: 画像サイズ計算と圧縮率分析 (Python)

```python
#!/usr/bin/env python3
"""
画像サイズ計算と圧縮率分析ツール
非圧縮サイズの計算と、各フォーマットでの想定サイズを算出する。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageSpec:
    """画像仕様を表すデータクラス"""
    width: int
    height: int
    bits_per_pixel: int = 24  # デフォルトRGB
    name: str = ""

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    @property
    def raw_size_bytes(self) -> int:
        """非圧縮サイズ（バイト）"""
        return self.total_pixels * self.bits_per_pixel // 8

    @property
    def megapixels(self) -> float:
        return self.total_pixels / 1_000_000


# よく使われる解像度の定義
COMMON_RESOLUTIONS = {
    "VGA":      ImageSpec(640, 480, name="VGA"),
    "HD":       ImageSpec(1280, 720, name="HD 720p"),
    "FHD":      ImageSpec(1920, 1080, name="Full HD 1080p"),
    "QHD":      ImageSpec(2560, 1440, name="QHD 1440p"),
    "4K":       ImageSpec(3840, 2160, name="4K UHD"),
    "8K":       ImageSpec(7680, 4320, name="8K UHD"),
    "iPhone15": ImageSpec(4032, 3024, name="iPhone 15 Pro (12MP)"),
    "DSLR":     ImageSpec(6000, 4000, name="DSLR (24MP)"),
}

# 圧縮フォーマットの想定圧縮率（非圧縮に対する比率）
COMPRESSION_RATIOS = {
    "BMP":      1.0,        # 非圧縮
    "PNG":      0.4,        # 可逆圧縮（写真の場合）
    "JPEG 95":  0.25,       # 高品質
    "JPEG 85":  0.083,      # 一般的な品質
    "JPEG 50":  0.025,      # 低品質
    "WebP":     0.05,       # 非可逆
    "AVIF":     0.033,      # 高効率
    "HEIF":     0.042,      # Apple標準
}


def format_size(size_bytes: int) -> str:
    """バイト数を読みやすい形式に変換"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def analyze_image(spec: ImageSpec) -> None:
    """画像仕様を分析し、各フォーマットでの想定サイズを表示"""
    print(f"\n{'='*60}")
    print(f"  画像: {spec.name}")
    print(f"  解像度: {spec.width} x {spec.height}")
    print(f"  画素数: {spec.megapixels:.1f} MP")
    print(f"  色深度: {spec.bits_per_pixel} bit")
    print(f"  非圧縮サイズ: {format_size(spec.raw_size_bytes)}")
    print(f"{'='*60}")
    print(f"  {'フォーマット':<12} {'想定サイズ':>12} {'圧縮率':>8}")
    print(f"  {'-'*12:<12} {'-'*12:>12} {'-'*8:>8}")

    for fmt, ratio in COMPRESSION_RATIOS.items():
        compressed_size = int(spec.raw_size_bytes * ratio)
        print(f"  {fmt:<12} {format_size(compressed_size):>12} {ratio*100:>6.1f}%")


def calculate_storage_for_service(
    daily_uploads: int,
    avg_size_kb: float,
    thumbnail_count: int = 3,
    thumbnail_avg_kb: float = 20.0,
    retention_years: int = 5,
    replication_factor: int = 3,
) -> dict:
    """
    写真共有サービスのストレージ要件を計算する。

    Args:
        daily_uploads: 1日あたりのアップロード数
        avg_size_kb: 元画像の平均サイズ (KB)
        thumbnail_count: 1画像あたりのサムネイル数
        thumbnail_avg_kb: サムネイル1枚の平均サイズ (KB)
        retention_years: データ保持年数
        replication_factor: レプリケーション係数

    Returns:
        各種の計算結果を格納した辞書
    """
    daily_original_gb = daily_uploads * avg_size_kb / (1024 * 1024)
    daily_thumbnail_gb = daily_uploads * thumbnail_count * thumbnail_avg_kb / (1024 * 1024)
    daily_total_gb = daily_original_gb + daily_thumbnail_gb

    yearly_tb = daily_total_gb * 365 / 1024
    total_with_replication_tb = yearly_tb * replication_factor
    total_retention_tb = total_with_replication_tb * retention_years

    return {
        "daily_uploads": daily_uploads,
        "daily_original_gb": daily_original_gb,
        "daily_thumbnail_gb": daily_thumbnail_gb,
        "daily_total_gb": daily_total_gb,
        "yearly_tb": yearly_tb,
        "with_replication_tb": total_with_replication_tb,
        "total_retention_tb": total_retention_tb,
        "total_retention_pb": total_retention_tb / 1024,
    }


if __name__ == "__main__":
    print("=== 画像サイズ分析 ===")

    # 各解像度の分析
    for key in ["FHD", "4K", "iPhone15"]:
        analyze_image(COMMON_RESOLUTIONS[key])

    # サービスレベルのストレージ計算
    print("\n\n=== 写真共有サービスのストレージ見積もり ===")

    # 中規模サービス: DAU 1000万, 1人平均3枚/日
    result = calculate_storage_for_service(
        daily_uploads=10_000_000 * 3,
        avg_size_kb=2048,  # 2MB
        thumbnail_count=3,
        thumbnail_avg_kb=20,
        retention_years=5,
        replication_factor=3,
    )

    print(f"\n  前提条件:")
    print(f"    DAU: 1000万人、1人平均3枚/日")
    print(f"    元画像: 平均2MB、サムネイル: 3種 x 20KB")
    print(f"    保持期間: 5年、レプリケーション: 3倍")
    print(f"\n  計算結果:")
    print(f"    1日のアップロード数:  {result['daily_uploads']:>15,}")
    print(f"    1日の元画像:          {result['daily_original_gb']:>12,.1f} GB")
    print(f"    1日のサムネイル:      {result['daily_thumbnail_gb']:>12,.1f} GB")
    print(f"    1日の合計:            {result['daily_total_gb']:>12,.1f} GB")
    print(f"    年間:                 {result['yearly_tb']:>12,.1f} TB")
    print(f"    レプリケーション込:   {result['with_replication_tb']:>12,.1f} TB")
    print(f"    5年間合計:            {result['total_retention_tb']:>12,.1f} TB")
    print(f"                          {result['total_retention_pb']:>12,.2f} PB")
```

### コード例3: ストレージ階層コスト最適化 (Python)

```python
#!/usr/bin/env python3
"""
階層型ストレージのコスト最適化シミュレーター

データのアクセス頻度に応じてホット/ウォーム/コールドに分類し、
それぞれのストレージ階層に割り当てた場合のコスト比較を行う。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class StorageTier:
    """ストレージ階層の定義"""
    name: str
    cost_per_tb_month: float  # $/TB/月
    read_cost_per_gb: float   # $/GB（読み取り）
    write_cost_per_gb: float  # $/GB（書き込み）
    retrieval_time: str       # 取得にかかる時間の目安

    def monthly_cost(self, data_tb: float, read_gb: float = 0, write_gb: float = 0) -> float:
        """月額コストを計算"""
        storage = data_tb * self.cost_per_tb_month
        read = read_gb * self.read_cost_per_gb
        write = write_gb * self.write_cost_per_gb
        return storage + read + write


# AWS S3 ベースの階層定義（想定される価格帯、2024年時点の参考値）
TIERS = {
    "hot": StorageTier(
        name="S3 Standard",
        cost_per_tb_month=23.0,
        read_cost_per_gb=0.0004,
        write_cost_per_gb=0.005,
        retrieval_time="ミリ秒",
    ),
    "warm": StorageTier(
        name="S3 IA (Infrequent Access)",
        cost_per_tb_month=12.5,
        read_cost_per_gb=0.01,
        write_cost_per_gb=0.01,
        retrieval_time="ミリ秒",
    ),
    "cold": StorageTier(
        name="S3 Glacier Instant",
        cost_per_tb_month=4.0,
        read_cost_per_gb=0.03,
        write_cost_per_gb=0.02,
        retrieval_time="ミリ秒",
    ),
    "archive": StorageTier(
        name="S3 Glacier Deep Archive",
        cost_per_tb_month=1.0,
        read_cost_per_gb=0.02,
        write_cost_per_gb=0.05,
        retrieval_time="12時間以内",
    ),
}


@dataclass
class DataCategory:
    """データカテゴリの定義"""
    name: str
    total_tb: float
    monthly_read_gb: float
    monthly_write_gb: float
    recommended_tier: str


def compare_storage_strategies(categories: List[DataCategory]) -> None:
    """
    全データをホットに置く場合と階層化する場合のコスト比較を行う。
    """
    print("=" * 70)
    print("  階層型ストレージ コスト分析")
    print("=" * 70)

    # 戦略1: 全てホット
    all_hot_cost = 0.0
    total_tb = sum(c.total_tb for c in categories)
    total_read_gb = sum(c.monthly_read_gb for c in categories)
    total_write_gb = sum(c.monthly_write_gb for c in categories)
    hot_tier = TIERS["hot"]
    all_hot_cost = hot_tier.monthly_cost(total_tb, total_read_gb, total_write_gb)

    print(f"\n  【戦略1】全データをS3 Standardに保存")
    print(f"  総データ量: {total_tb:.1f} TB")
    print(f"  月額コスト: ${all_hot_cost:,.2f}")
    print(f"  年間コスト: ${all_hot_cost * 12:,.2f}")

    # 戦略2: 階層化
    print(f"\n  【戦略2】アクセス頻度に応じた階層化")
    print(f"  {'カテゴリ':<20} {'階層':<25} {'データ量':>8} {'月額':>12}")
    print(f"  {'-'*20:<20} {'-'*25:<25} {'-'*8:>8} {'-'*12:>12}")

    tiered_cost = 0.0
    for cat in categories:
        tier = TIERS[cat.recommended_tier]
        cost = tier.monthly_cost(cat.total_tb, cat.monthly_read_gb, cat.monthly_write_gb)
        tiered_cost += cost
        print(f"  {cat.name:<20} {tier.name:<25} {cat.total_tb:>6.1f}TB ${cost:>10,.2f}")

    print(f"\n  階層化後の月額コスト: ${tiered_cost:,.2f}")
    print(f"  階層化後の年間コスト: ${tiered_cost * 12:,.2f}")

    # 節約額
    savings = all_hot_cost - tiered_cost
    savings_pct = (savings / all_hot_cost) * 100 if all_hot_cost > 0 else 0
    print(f"\n  月間節約額: ${savings:,.2f} ({savings_pct:.1f}%)")
    print(f"  年間節約額: ${savings * 12:,.2f}")


if __name__ == "__main__":
    # 写真共有サービスの例
    categories = [
        DataCategory(
            name="直近30日の画像",
            total_tb=60.0,
            monthly_read_gb=50000,   # 頻繁にアクセスされる
            monthly_write_gb=2000,
            recommended_tier="hot",
        ),
        DataCategory(
            name="30日〜1年の画像",
            total_tb=660.0,
            monthly_read_gb=5000,    # たまにアクセス
            monthly_write_gb=0,
            recommended_tier="warm",
        ),
        DataCategory(
            name="1年〜3年の画像",
            total_tb=1320.0,
            monthly_read_gb=500,     # まれにアクセス
            monthly_write_gb=0,
            recommended_tier="cold",
        ),
        DataCategory(
            name="3年以上の画像",
            total_tb=660.0,
            monthly_read_gb=10,      # ほぼアクセスなし
            monthly_write_gb=0,
            recommended_tier="archive",
        ),
    ]

    compare_storage_strategies(categories)
```

---

## 5. システム設計の容量見積もり

### 5.1 Back-of-the-envelope Estimation の基本フレームワーク

システム設計面接や実際のインフラ設計で容量見積もりを行う際の体系的なフレームワークを示す。

```
容量見積もりの5ステップ:

  Step 1: ユーザー規模を定義する
  ─────────────────────────────
  - 総ユーザー数 (Total Users)
  - DAU (Daily Active Users): 通常は総ユーザーの 20-30%
  - ピーク時DAU: 通常DAUの 2-3倍

  Step 2: ユーザー行動を定量化する
  ─────────────────────────────
  - 1ユーザーあたりの操作数/日
  - 各操作で生成/消費するデータ量
  - 読み書き比率 (Read:Write ratio)

  Step 3: データ量を算出する
  ─────────────────────────
  日次データ = DAU × 操作数/ユーザー × データ/操作
  年間データ = 日次 × 365
  合計 = 年間 × 保持年数

  Step 4: 冗長性とオーバーヘッドを加算する
  ─────────────────────────────────────
  - レプリケーション: ×3 (一般的)
  - インデックス: 元データの 10-30%
  - ログ・メタデータ: 元データの 5-20%
  - バッファ: 20-30% の余裕

  Step 5: コストを見積もる
  ────────────────────────
  - ストレージコスト ($/TB/月)
  - 転送コスト ($/GB)
  - コンピューティングコスト ($/時間)
```

### 5.2 Twitter風サービスの容量計算例

```
Twitter風サービスの容量見積もり:

  前提:
  ─────
  - 総ユーザー: 500M (5億)
  - DAU: 300M (3億) = 60%
  - 1人あたり平均2ツイート/日（書き込み）
  - 1人あたり平均100ツイート/日（読み取り＝タイムライン表示）
  - テキスト平均200文字 (UTF-8: 400バイト)
  - 20%のツイートに画像添付 (平均200KB)
  - 2%のツイートに動画添付 (平均2MB)
  - メタデータ: 200バイト/ツイート（タイムスタンプ、ユーザーID、位置情報等）

  1日のデータ量:
  ──────────────
  ツイート数:   300M × 2 = 600M ツイート/日

  テキスト:     600M × (400 + 200)B = 360 GB/日
  画像:         600M × 0.20 × 200KB = 24 TB/日
  動画:         600M × 0.02 × 2MB   = 24 TB/日
  ────────────────────────────────────────────
  合計:         約48 TB/日

  年間:         48 TB × 365 = 約17.5 PB/年

  ストレージ要件（5年運用）:
  ──────────────────────────
  基本:         17.5 PB × 5 = 87.5 PB
  レプリカ3倍:  87.5 × 3 = 262.5 PB
  インデックス:  87.5 × 0.2 = 17.5 PB
  バッファ30%:  (262.5 + 17.5) × 1.3 = 364 PB
  圧縮50%適用:  364 × 0.5 = 182 PB

  帯域幅:
  ────────
  書き込みQPS: 600M / 86400 ≈ 6,944 QPS
  ピーク (×3):  6,944 × 3 ≈ 20,833 QPS
  読み取り比:   100:1
  読み取りQPS:  20,833 × 100 ≈ 2,083,300 QPS

  必要帯域幅:
  - 書き込み: 48TB / 86400 ≈ 555 MB/s ≈ 4.4 Gbps
  - ピーク:   4.4 × 3 ≈ 13.3 Gbps
  - 読み取り: 13.3 × 100 ≈ 1.33 Tbps（CDN分散前）
```

### 5.3 よくあるシステムの容量

```
サービス規模別の目安:

  ┌─────────────────┬──────────┬──────────────┬───────────┬──────────┐
  │ 規模            │ DB       │ ストレージ    │ 帯域幅    │ 月額概算  │
  ├─────────────────┼──────────┼──────────────┼───────────┼──────────┤
  │ 個人ブログ      │ 100MB    │ 10GB         │ 10Mbps   │ $5-20    │
  │ スタートアップ   │ 10GB     │ 1TB          │ 100Mbps  │ $500-2K  │
  │ 中規模サービス   │ 1TB      │ 100TB        │ 10Gbps   │ $50K-200K│
  │ 大規模サービス   │ 100TB    │ 10PB         │ 1Tbps    │ $1M-10M  │
  │ FAANG          │ 10PB+    │ 1EB+         │ 100Tbps+ │ $100M+   │
  └─────────────────┴──────────┴──────────────┴───────────┴──────────┘

  各規模の典型的なアーキテクチャ:
  ──────────────────────────────
  個人ブログ:      単一サーバー、SQLite/MySQL
  スタートアップ:   マネージドDB (RDS)、S3、CloudFront
  中規模:          DBクラスタ、Redis、複数AZ
  大規模:          シャーディング、専用CDN、マルチリージョン
  FAANG:          自前DC、カスタムストレージ、エッジコンピューティング
```

### 5.4 レイテンシの数字（Jeff Dean's Numbers 2024版）

システム設計においてデータの「どこにあるか」は、必要な容量だけでなくアクセス速度にも直結する。ストレージ階層選択の判断にはレイテンシの理解が不可欠である。

```
プログラマが知るべきレイテンシ:

  操作                          時間           比喩
  ──────────────────────────────────────────────────
  L1キャッシュ参照               1 ns          瞬き
  分岐予測ミス                   3 ns
  L2キャッシュ参照               4 ns
  ミューテックスのロック/アンロック 17 ns
  L3キャッシュ参照               20 ns
  メインメモリ参照               100 ns         まばたき
  1KB zippy圧縮                 2,000 ns
  SSD ランダム読み取り           16,000 ns      くしゃみ
  1MB メモリから連続読み取り      3,000 ns
  1MB SSD から連続読み取り       49,000 ns
  1MB HDD から連続読み取り       825,000 ns
  同一DC内ラウンドトリップ        500,000 ns     深呼吸
  HDD シーク                    2,000,000 ns
  パケット: CA→NL→CA            150,000,000 ns  昼寝

  覚えるべき比率:
  ──────────────
  - メモリ vs SSD:    100倍
  - SSD vs HDD:      50-100倍
  - DC内 vs 大陸間:   300倍
  - メモリ vs ネット:  5,000倍

  ストレージ階層選択への応用:
  ┌─────────────┬──────────┬──────────────────────────────┐
  │ レイテンシ要件│ 選択肢    │ ユースケース                  │
  ├─────────────┼──────────┼──────────────────────────────┤
  │ < 1ms       │ メモリ/SSD│ リアルタイム検索、セッション    │
  │ 1-10ms      │ SSD      │ DB読取、API応答               │
  │ 10-100ms    │ HDD/SSD  │ ログ検索、バッチ処理           │
  │ > 100ms     │ HDD/テープ│ アーカイブ、コンプライアンス    │
  └─────────────┴──────────┴──────────────────────────────┘
```

---

## 6. 帯域幅の計算

### 6.1 ネットワーク帯域幅の基礎

```
帯域幅の実務計算:

  基本変換:
  ─────────
  ネットワーク速度は bit/s、ファイルサイズは byte で表記される。
  → 1 Gbps = 1,000,000,000 bits/s = 125,000,000 bytes/s ≈ 125 MB/s

  なぜ bit/s を使うのか:
  歴史的にネットワークはシリアル通信（1ビットずつ送信）から発展した。
  そのため速度の基本単位がビット毎秒になった。
  一方、ストレージはバイト単位でアドレスされるためバイトが基本単位。

  実効帯域幅（オーバーヘッド考慮）:
  ──────────────────────────────
  物理層のオーバーヘッド、プロトコルヘッダ、再送等により、
  理論値の80-90%が実効帯域幅となる。

  1 Gbps → 実効 約100-110 MB/s
  10 Gbps → 実効 約1.1 GB/s
  100 Gbps → 実効 約11 GB/s

  ダウンロード時間の目安:
  ┌──────────────┬─────────┬──────────┬──────────┐
  │ ファイル      │ 10Mbps  │ 100Mbps  │ 1Gbps   │
  ├──────────────┼─────────┼──────────┼──────────┤
  │ Webページ(2MB)│ 1.6秒   │ 0.16秒   │ 0.016秒 │
  │ 曲1曲(4MB)   │ 3.2秒   │ 0.32秒   │ 0.032秒 │
  │ 映画(5GB)    │ 67分    │ 6.7分    │ 40秒    │
  │ ゲーム(50GB) │ 11時間  │ 67分     │ 6.7分   │
  │ バックアップ1TB│ 9.3日  │ 22時間   │ 2.2時間 │
  └──────────────┴─────────┴──────────┴──────────┘
```

### 6.2 Sneakernet: 物理メディアの帯域幅

```
Sneakernet（物理メディア輸送）:

  「トラック一杯のハードディスクの帯域幅を舐めてはいけない」
  — Andrew Tanenbaum

  物理配送の帯域幅計算:
  ─────────────────────
  10TB HDD × 100台 = 1PB
  翌日配送 (86400秒) → 1PB / 86400 ≈ 11.6 GB/s ≈ 93 Gbps

  → 一般的な企業のインターネット回線 (1-10 Gbps) より
    物理輸送の方が帯域幅が大きい場合がある

  AWS の物理データ転送サービス:
  ┌──────────────┬──────────┬──────────────────────────┐
  │ サービス      │ 容量      │ 用途                      │
  ├──────────────┼──────────┼──────────────────────────┤
  │ Snowcone    │ 8-14 TB   │ エッジコンピューティング    │
  │ Snowball    │ 80 TB     │ データセンター移行          │
  │ Snowball Edge│ 80-210TB │ エッジでの処理+転送        │
  │ Snowmobile  │ 100 PB    │ エクサバイト級の移行        │
  └──────────────┴──────────┴──────────────────────────┘

  コスト比較 (100TB転送):
  - インターネット経由 (1Gbps): 約9.3日 + 転送料 $9,000+
  - Snowball: 約1週間 (配送含む) + $300 (固定料金)
  → 大量データでは物理輸送が時間的にもコスト的にも有利
```

### 6.3 Webパフォーマンス予算

```
Webパフォーマンス予算の考え方:

  なぜパフォーマンス予算が必要なのか:
  - ページ読み込みが1秒遅れるとCVRが7%低下する（Amazon調査）
  - 3秒以上かかるとモバイルユーザーの53%が離脱（Google調査）
  - Core Web Vitals はSEOランキング要因の一つ

  目標: 3Gモバイル (1.5Mbps) で3秒以内にLCP

  3秒 × 1.5Mbps = 4.5Mbit = 562KB の予算

  配分例:
  ┌──────────────┬───────────┬──────────────────────────┐
  │ リソース      │ 予算      │ 説明                      │
  ├──────────────┼───────────┼──────────────────────────┤
  │ HTML         │ 30KB      │ gzip後                    │
  │ CSS          │ 50KB      │ gzip後、Critical CSS抽出  │
  │ JavaScript   │ 200KB     │ gzip後、コード分割適用     │
  │ フォント      │ 80KB      │ woff2、サブセット化       │
  │ 画像         │ 200KB     │ WebP/AVIF、遅延読み込み   │
  │ 合計         │ 560KB     │ ≦ 予算                   │
  └──────────────┴───────────┴──────────────────────────┘

  JavaScript 200KB (gzip) ≈ 800KB (非圧縮)
  → パース+コンパイル時間もボトルネックになる
  → モバイルCPUでは1MBのJSパースに 2-5秒
  → Tree Shaking、Dynamic Import で不要なコードを排除すべき
```

---

## 7. データセンター規模のスケール

### 7.1 主要サービスのデータ量

```
世界のデータ量（2024年時点の想定される規模）:

  Google:
  ──────
  - 検索インデックス: 約100PB以上
  - Gmail: 約15EB以上（30億ユーザー × 5GBの割当）
  - YouTube: 1分あたり500時間のアップロード → 1日約720,000時間
  - Google Photos: 1日40億枚の写真がアップロード
  - Google全体の保存データ: 推定 10-15 EB

  Meta:
  ─────
  - Instagram: 1日1億枚以上の写真
  - WhatsApp: 1日1,000億通のメッセージ
  - Facebook: 推定 数EB のユーザーデータ
  - AI学習用データ: 推定 数十PB

  Netflix:
  ────────
  - カタログ: 約10PB（全タイトル×全解像度×多言語）
  - CDN配信: ピーク時インターネット帯域の15%
  - Open Connect CDN: 世界中に数千のサーバー設置

  世界全体:
  ─────────
  - 2025年の年間データ生成量: 約180ZB（ゼタバイト）
  - 1ZB = 10^21 バイト = 1,000 EB = 1,000,000 PB
  - データの90%は過去2年間に生成された
  - IoTデバイスからのデータが急増中
```

### 7.2 ストレージコスト詳細

```
ストレージコスト比較 (2024年時点の想定される価格帯):

  ┌─────────────────┬──────────────┬──────────────────┬──────────────┐
  │ メディア         │ $/TB/月      │ 用途              │ 取出し時間    │
  ├─────────────────┼──────────────┼──────────────────┼──────────────┤
  │ RAMメモリ        │ $3,000       │ キャッシュ、DB     │ 即座 (ns)    │
  │ NVMe SSD        │ $50          │ ホットデータ       │ 即座 (μs)    │
  │ SATA SSD        │ $20          │ ウォームデータ     │ 即座 (μs)    │
  │ HDD             │ $5           │ コールドデータ     │ 即座 (ms)    │
  │ S3 Standard     │ $23          │ クラウドホット     │ 即座 (ms)    │
  │ S3 IA           │ $12.5        │ クラウドウォーム   │ 即座 (ms)    │
  │ S3 Glacier      │ $4           │ クラウドコールド   │ 分〜時間     │
  │ S3 Deep Archive │ $1           │ アーカイブ        │ 12時間以内   │
  │ テープ (LTO-9)  │ $0.5         │ 長期保存          │ 数分〜時間   │
  └─────────────────┴──────────────┴──────────────────┴──────────────┘

  階層型ストレージの設計指針:
  ──────────────────────────
  ホット (10-20% のデータ):
    → SSD / S3 Standard
    → 頻繁にアクセスされるデータ
    → ユーザープロフィール、直近の投稿、セッションデータ

  ウォーム (20-30% のデータ):
    → HDD / S3 IA
    → たまにアクセスされるデータ
    → 30日〜1年前の投稿、古い注文履歴

  コールド (30-40% のデータ):
    → Glacier / テープ
    → ほぼアクセスされないデータ
    → 1年以上前のデータ、バックアップ

  アーカイブ (10-20% のデータ):
    → Deep Archive
    → 法令遵守のためだけに保持するデータ
    → 監査ログ、削除済みアカウントのデータ
```

---

## 8. コード例: 帯域幅とスループット分析

### コード例4: ネットワーク帯域幅計算ツール (Python)

```python
#!/usr/bin/env python3
"""
ネットワーク帯域幅と転送時間の計算ツール
ビットとバイトの変換、実効帯域幅、転送時間を正確に計算する。
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class BandwidthSpec:
    """帯域幅仕様"""
    name: str
    bits_per_second: float
    efficiency: float = 0.85  # プロトコルオーバーヘッド等で85%が想定値

    @property
    def effective_bytes_per_second(self) -> float:
        """実効帯域幅 (バイト/秒)"""
        return self.bits_per_second * self.efficiency / 8

    @property
    def effective_mbps(self) -> float:
        """実効帯域幅 (Mbps)"""
        return self.bits_per_second * self.efficiency / 1_000_000


def format_duration(seconds: float) -> str:
    """秒数を人間が読みやすい時間表現に変換"""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.1f} 秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} 分"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} 時間"
    else:
        days = seconds / 86400
        return f"{days:.1f} 日"


def transfer_time(file_size_bytes: float, bandwidth: BandwidthSpec) -> float:
    """ファイル転送にかかる時間（秒）を計算"""
    if bandwidth.effective_bytes_per_second <= 0:
        raise ValueError("帯域幅は正の値である必要があります")
    return file_size_bytes / bandwidth.effective_bytes_per_second


def calculate_daily_bandwidth_requirement(
    daily_requests: int,
    avg_response_size_kb: float,
    peak_factor: float = 3.0,
) -> dict:
    """
    1日のリクエスト数から必要な帯域幅を計算する。

    Args:
        daily_requests: 1日のリクエスト数
        avg_response_size_kb: 平均レスポンスサイズ (KB)
        peak_factor: ピーク倍率（デフォルト3倍）

    Returns:
        計算結果の辞書
    """
    avg_qps = daily_requests / 86400
    peak_qps = avg_qps * peak_factor

    avg_throughput_mbps = avg_qps * avg_response_size_kb * 8 / 1000
    peak_throughput_mbps = peak_qps * avg_response_size_kb * 8 / 1000

    daily_transfer_gb = daily_requests * avg_response_size_kb / (1024 * 1024)

    return {
        "avg_qps": avg_qps,
        "peak_qps": peak_qps,
        "avg_throughput_mbps": avg_throughput_mbps,
        "peak_throughput_mbps": peak_throughput_mbps,
        "daily_transfer_gb": daily_transfer_gb,
        "monthly_transfer_tb": daily_transfer_gb * 30 / 1024,
    }


if __name__ == "__main__":
    # 一般的な回線速度の定義
    bandwidths = [
        BandwidthSpec("3G Mobile", 1_500_000),          # 1.5 Mbps
        BandwidthSpec("4G LTE", 50_000_000),            # 50 Mbps
        BandwidthSpec("5G", 1_000_000_000),             # 1 Gbps
        BandwidthSpec("家庭用光回線", 1_000_000_000),     # 1 Gbps
        BandwidthSpec("企業用10G", 10_000_000_000),      # 10 Gbps
        BandwidthSpec("DC間100G", 100_000_000_000),      # 100 Gbps
    ]

    # 一般的なファイルサイズ
    files = [
        ("Webページ", 2 * 1024 * 1024),               # 2 MB
        ("写真1枚", 5 * 1024 * 1024),                  # 5 MB
        ("音楽1曲", 4 * 1024 * 1024),                  # 4 MB
        ("動画10分", 200 * 1024 * 1024),               # 200 MB
        ("映画1本", 5 * 1024 * 1024 * 1024),           # 5 GB
        ("ゲーム", 50 * 1024 * 1024 * 1024),           # 50 GB
        ("バックアップ", 1024 * 1024 * 1024 * 1024),    # 1 TB
    ]

    print("=== 転送時間一覧表 ===\n")
    header = f"  {'ファイル':<14}"
    for bw in bandwidths:
        header += f" {bw.name:>12}"
    print(header)
    print("  " + "-" * (14 + 13 * len(bandwidths)))

    for file_name, file_size in files:
        row = f"  {file_name:<14}"
        for bw in bandwidths:
            t = transfer_time(file_size, bw)
            row += f" {format_duration(t):>12}"
        print(row)

    # 帯域幅要件計算の例
    print("\n\n=== サービスの帯域幅要件計算 ===\n")
    scenarios = [
        ("ブログサイト", 100_000, 200),          # 10万PV/日, 200KB
        ("ECサイト", 5_000_000, 500),             # 500万PV/日, 500KB
        ("動画配信", 50_000_000, 5_000),          # 5000万再生/日, 5MB
    ]

    for name, requests, avg_kb in scenarios:
        result = calculate_daily_bandwidth_requirement(requests, avg_kb)
        print(f"  [{name}] {requests:,} req/日, 平均{avg_kb}KB")
        print(f"    平均QPS:        {result['avg_qps']:>12,.1f}")
        print(f"    ピークQPS:      {result['peak_qps']:>12,.1f}")
        print(f"    平均帯域:       {result['avg_throughput_mbps']:>12,.1f} Mbps")
        print(f"    ピーク帯域:     {result['peak_throughput_mbps']:>12,.1f} Mbps")
        print(f"    日次転送量:     {result['daily_transfer_gb']:>12,.1f} GB")
        print(f"    月次転送量:     {result['monthly_transfer_tb']:>12,.2f} TB")
        print()
```

### コード例5: ディスク容量モニタリングとアラート (C)

```c
/*
 * disk_monitor.c - ディスク容量監視ツール
 *
 * ファイルシステムの使用状況を取得し、
 * 閾値を超えた場合にアラートを出力する。
 *
 * コンパイル: gcc -o disk_monitor disk_monitor.c -Wall -Wextra
 * 実行: ./disk_monitor
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/statvfs.h>
#endif

/* アラート閾値 */
#define WARN_THRESHOLD  80.0   /* 警告: 80%以上 */
#define CRITICAL_THRESHOLD 90.0 /* 危険: 90%以上 */

/* サイズ単位の定義 */
typedef enum {
    UNIT_B,
    UNIT_KIB,
    UNIT_MIB,
    UNIT_GIB,
    UNIT_TIB
} SizeUnit;

/* ファイルシステム情報 */
typedef struct {
    char mount_point[256];
    unsigned long long total_bytes;
    unsigned long long used_bytes;
    unsigned long long avail_bytes;
    double usage_percent;
} FsInfo;

/*
 * バイト数を適切な単位に変換して文字列にする。
 * なぜ1024で割るのか: ファイルシステムは伝統的に2進接頭辞を使用するため。
 * OSの表示に合わせる場合は1024ベースが適切。
 */
void format_bytes(unsigned long long bytes, char *buf, size_t buf_size) {
    const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    int unit_idx = 0;
    double size = (double)bytes;

    while (size >= 1024.0 && unit_idx < 5) {
        size /= 1024.0;
        unit_idx++;
    }

    if (size >= 100.0) {
        snprintf(buf, buf_size, "%.0f %s", size, units[unit_idx]);
    } else if (size >= 10.0) {
        snprintf(buf, buf_size, "%.1f %s", size, units[unit_idx]);
    } else {
        snprintf(buf, buf_size, "%.2f %s", size, units[unit_idx]);
    }
}

/*
 * 使用率に応じたアラートレベルの判定。
 *
 * 閾値の考え方:
 * - 80%未満: 正常。通常運用。
 * - 80-90%: 警告。増加傾向の場合は容量拡張を計画すべき。
 * - 90%以上: 危険。即座に対応が必要。
 *   ファイルシステムの残容量が少ないとパフォーマンスも低下する
 *   （断片化の増加、ジャーナリングの圧迫など）。
 */
const char* get_alert_level(double usage_percent) {
    if (usage_percent >= CRITICAL_THRESHOLD) {
        return "[CRITICAL]";
    } else if (usage_percent >= WARN_THRESHOLD) {
        return "[WARNING] ";
    } else {
        return "[OK]      ";
    }
}

#ifndef _WIN32
/*
 * POSIXシステムでファイルシステム情報を取得する。
 * statvfs() を使用する理由:
 * - POSIX標準で移植性が高い
 * - ブロックサイズ、総ブロック数、空きブロック数を取得可能
 * - f_bavail は一般ユーザーが使用可能な空きブロック数
 *   (f_bfree はrootのみの予約領域を含む)
 */
int get_fs_info(const char *path, FsInfo *info) {
    struct statvfs stat;

    if (statvfs(path, &stat) != 0) {
        return -1;
    }

    strncpy(info->mount_point, path, sizeof(info->mount_point) - 1);
    info->mount_point[sizeof(info->mount_point) - 1] = '\0';

    info->total_bytes = (unsigned long long)stat.f_blocks * stat.f_frsize;
    info->avail_bytes = (unsigned long long)stat.f_bavail * stat.f_frsize;
    info->used_bytes = info->total_bytes -
                       (unsigned long long)stat.f_bfree * stat.f_frsize;

    if (info->total_bytes > 0) {
        info->usage_percent =
            (double)info->used_bytes / (double)info->total_bytes * 100.0;
    } else {
        info->usage_percent = 0.0;
    }

    return 0;
}
#endif

/*
 * 使用率をASCIIバーグラフとして表示する。
 * バーの長さは最大40文字とし、使用率に比例して塗りつぶす。
 */
void print_usage_bar(double usage_percent) {
    const int bar_width = 40;
    int filled = (int)(usage_percent * bar_width / 100.0);

    printf("[");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) {
            printf("#");
        } else {
            printf("-");
        }
    }
    printf("]");
}

int main(void) {
    printf("=== ディスク容量モニター ===\n\n");

#ifndef _WIN32
    const char *paths[] = {"/", "/home", "/tmp", "/var"};
    int num_paths = sizeof(paths) / sizeof(paths[0]);

    printf("  %-10s %10s %10s %10s %6s  %s\n",
           "マウント", "合計", "使用済", "空き", "使用率", "状態");
    printf("  %-10s %10s %10s %10s %6s  %s\n",
           "----------", "----------", "----------",
           "----------", "------", "----------");

    for (int i = 0; i < num_paths; i++) {
        FsInfo info;
        if (get_fs_info(paths[i], &info) == 0) {
            char total_str[32], used_str[32], avail_str[32];
            format_bytes(info.total_bytes, total_str, sizeof(total_str));
            format_bytes(info.used_bytes, used_str, sizeof(used_str));
            format_bytes(info.avail_bytes, avail_str, sizeof(avail_str));

            printf("  %-10s %10s %10s %10s %5.1f%%  %s\n",
                   info.mount_point,
                   total_str, used_str, avail_str,
                   info.usage_percent,
                   get_alert_level(info.usage_percent));

            printf("  ");
            print_usage_bar(info.usage_percent);
            printf("\n\n");
        }
    }
#else
    printf("  Windows環境ではGetDiskFreeSpaceEx()を使用します。\n");
    printf("  このサンプルはPOSIX環境向けです。\n");
#endif

    /* 容量予測シミュレーション */
    printf("\n=== 容量予測シミュレーション ===\n\n");
    double current_used_tb = 8.5;
    double total_tb = 12.0;
    double daily_growth_gb = 50.0;

    printf("  現在の使用量: %.1f TB / %.1f TB (%.1f%%)\n",
           current_used_tb, total_tb,
           current_used_tb / total_tb * 100.0);
    printf("  日次増加量:   %.0f GB/日\n\n", daily_growth_gb);

    double remaining_gb = (total_tb - current_used_tb) * 1024.0;
    double days_to_full = remaining_gb / daily_growth_gb;
    double days_to_80 = (total_tb * 0.8 - current_used_tb) * 1024.0 / daily_growth_gb;
    double days_to_90 = (total_tb * 0.9 - current_used_tb) * 1024.0 / daily_growth_gb;

    printf("  80%%到達まで: %.0f 日\n", days_to_80 > 0 ? days_to_80 : 0);
    printf("  90%%到達まで: %.0f 日\n", days_to_90 > 0 ? days_to_90 : 0);
    printf("  100%%到達まで: %.0f 日\n", days_to_full);
    printf("\n  推奨: %.0f日以内に容量拡張を計画してください。\n",
           days_to_80 > 0 ? days_to_80 : 0);

    return 0;
}
```

---

## 9. アンチパターン

### アンチパターン1: 単位の混同による容量不足

```
問題のシナリオ:

  「1TBのHDDを10台で10TBのストレージプールを構成する」と計画。
  しかし、以下の複数の要因で想定より少ない実効容量となる。

  ┌─────────────────────────────────┬──────────────────┐
  │ 要因                            │ 影響              │
  ├─────────────────────────────────┼──────────────────┤
  │ SI vs 2進の差 (7%)              │ 10TB → 9.31TiB   │
  │ ファイルシステムのオーバーヘッド (3-5%) │ 9.31 → 8.84 TiB │
  │ RAID 6のパリティ (2台分)         │ 8.84 → 7.07 TiB  │
  │ ファイルシステム予約 (5%)        │ 7.07 → 6.72 TiB  │
  └─────────────────────────────────┴──────────────────┘

  結果: メーカー表記の10TBに対して実効容量は約6.72TiB（67%）

  なぜ起こるのか:
  - 容量計画時に「額面上の容量」だけで見積もる
  - 各層のオーバーヘッドを考慮していない
  - RAIDレベルによるパリティ消費を忘れる

  正しいアプローチ:
  - 必要な実効容量から逆算して物理容量を決める
  - 安全係数として 1.4〜1.5倍 の物理容量を確保する
  - 実効容量 = 物理容量 × (1 - SI差) × (1 - FS OH) × RAID効率 × (1 - 予約)
```

### アンチパターン2: 全データを同一ストレージ階層に保存する

```
問題のシナリオ:

  写真共有サービスで、全ての画像（3年分、合計2700TB）を
  S3 Standardに保存している。

  コスト計算:
  2700 TB × $23/TB/月 = $62,100/月 = $745,200/年

  実際のアクセスパターン分析:
  ┌──────────────┬────────────┬───────────────┐
  │ 期間          │ データ量    │ アクセス頻度    │
  ├──────────────┼────────────┼───────────────┤
  │ 直近30日      │ 75 TB      │ 全体の80%      │
  │ 30日〜6ヶ月   │ 375 TB     │ 全体の15%      │
  │ 6ヶ月〜1年    │ 450 TB     │ 全体の4%       │
  │ 1年〜3年      │ 1800 TB    │ 全体の1%       │
  └──────────────┴────────────┴───────────────┘

  階層化後のコスト:
  75TB × $23    =  $1,725
  375TB × $12.5 =  $4,688
  450TB × $4    =  $1,800
  1800TB × $1   =  $1,800
  ────────────────────────
  合計:           $10,013/月 = $120,156/年

  節約額: $745,200 - $120,156 = $625,044/年（84%削減）

  なぜ全て同一階層になるのか:
  - サービスの初期段階では全データがホット
  - データが増えてもストレージ戦略を見直さない
  - ライフサイクルポリシーの自動化を怠る

  正しいアプローチ:
  - S3 Lifecycle Policy でアクセス頻度に応じた自動移行を設定
  - S3 Intelligent-Tiering で自動分類も可能
  - 定期的にアクセスパターンを分析し、閾値を調整する
```

---

## 10. エッジケース分析

### エッジケース1: ストレージの「見えない消費」

```
問題: ディスク使用量がデータ量と合わない

  ファイルシステム上の「見えない消費」の要因:

  1. ファイルシステムのメタデータ
     ────────────────────────────
     - inode テーブル: ext4ではデフォルトで16KBごとに1 inode
     - 1億ファイル → inodeだけで約25GB消費
     - ジャーナル: ext4ではデフォルトで128MB

  2. ブロックサイズの無駄（内部断片化）
     ──────────────────────────────
     - ext4のデフォルトブロックサイズ: 4KB
     - 1バイトのファイルでも4KB消費する
     - 大量の小ファイルがある場合、この無駄が膨大になる

     例: 1000万個の1KBファイル
     → 論理サイズ: 10GB
     → 実際のディスク使用量: 40GB (ブロックサイズ4KB × 1000万)
     → 無駄: 30GB (75%)

  3. 削除済みファイルのハンドル保持
     ──────────────────────────────
     - プロセスがファイルを開いたまま削除すると、
       プロセスが終了するまでディスク領域は解放されない
     - lsof +L1 で検出可能

  4. スパースファイル
     ────────────────
     - 論理サイズは大きいが物理的にはゼロブロックを省略
     - du と ls -l で異なるサイズが表示される
     - バックアップツールによってはスパース性が失われ、
       バックアップ先で実サイズに膨張することがある

  対策:
  - df と du の差分を定期的にチェック
  - 小ファイルが大量にある場合はパック化やDB化を検討
  - スパースファイルを扱うツールの挙動を事前に確認する
```

### エッジケース2: 容量見積もりにおけるスパイクの罠

```
問題: 平均値ベースの見積もりで容量不足に陥る

  シナリオ: 動画配信サービスの帯域幅設計

  平均値での計算:
  ──────────────
  DAU: 1000万人
  平均視聴時間: 30分/日
  平均ビットレート: 5Mbps
  平均同時接続: 1000万 × 30/1440 ≈ 208,333

  必要帯域: 208,333 × 5Mbps ≈ 1.04 Tbps

  しかし実際には:
  ──────────────
  - 夜間ピーク (20:00-23:00): 平均の3-5倍
  - 人気作品のリリース日: 平均の10倍
  - ワールドカップ決勝: 平均の20倍以上

  ピーク時:
  208,333 × 5 × 5Mbps ≈ 5.2 Tbps （通常ピーク）
  208,333 × 20 × 5Mbps ≈ 20.8 Tbps（イベント時）

  ストレージへの影響:
  ─────────────────
  ピーク時にバッファリング用のストレージI/Oも急増する。
  - SSD の IOPS が枯渇する
  - カーネルのページキャッシュが溢れてスワップが発生
  - ログの書き込みが追いつかずにログロスト

  対策:
  ┌──────────────────┬──────────────────────────────┐
  │ 対策              │ 効果                          │
  ├──────────────────┼──────────────────────────────┤
  │ CDNの地理分散     │ エッジでの配信で中央負荷軽減   │
  │ Adaptive Bitrate │ 輻輳時に自動的に画質を下げる   │
  │ Pre-warming      │ 人気コンテンツを事前にキャッシュ │
  │ Auto-scaling     │ クラウドインスタンスの自動拡張  │
  │ Circuit Breaker  │ 過負荷時に新規接続を制限       │
  └──────────────────┴──────────────────────────────┘

  見積もりの鉄則:
  - ピーク係数は最低3倍、メディア系は10倍以上を想定
  - 年間最大イベントの想定トラフィックで設計する
  - ただし全てをピークに合わせると過剰投資になるため、
    CDNとオートスケーリングで弾力的に対応する設計が現実的
```

### エッジケース3: ファイルシステムの制限に起因する問題

```
問題: 容量が余っているのにファイルを作成できない

  ファイルシステムには容量以外にも制限がある。

  1. inode枯渇
     ─────────
     ext4のデフォルト: 16KBごとに1 inode
     1TB のパーティション → 約6400万 inode

     小さなファイルが大量にある場合:
     - メール(1通≈4KB) × 6400万 = 約256GB でinode枯渇
     - ディスク残容量が700GB以上あるのに「No space left on device」

     確認方法: df -i
     対策: inode数を多く設定してフォーマット、またはXFSを使用

  2. 単一ディレクトリ内のファイル数
     ─────────────────────────────
     - ext4: 理論上は無制限だが、10万ファイル超でls等が遅延
     - dir_index (htree) が有効でも100万超で顕著に遅延
     - 対策: 日付やハッシュでサブディレクトリに分散

     例:
     /uploads/2024/03/15/abc123.jpg    ← 日付階層
     /uploads/a/b/c/abc123.jpg          ← ハッシュ先頭3文字

  3. 最大ファイルサイズ
     ────────────────
     ┌──────────┬──────────────────────┐
     │ FS        │ 最大ファイルサイズ     │
     ├──────────┼──────────────────────┤
     │ FAT32    │ 4 GiB - 1            │
     │ NTFS     │ 16 TiB               │
     │ ext4     │ 16 TiB               │
     │ XFS      │ 8 EiB                │
     │ ZFS      │ 16 EiB               │
     │ Btrfs    │ 16 EiB               │
     └──────────┴──────────────────────┘

     FAT32の4GB制限:
     - 動画ファイルや大きなディスクイメージで問題になる
     - USBメモリがFAT32フォーマットの場合に遭遇しやすい
     - exFATへのフォーマット変更で解決
```

---

## 11. コード例: RAID容量計算シミュレーター

### コード例6: RAID構成の実効容量計算 (Python)

```python
#!/usr/bin/env python3
"""
RAID構成の実効容量とコスト効率を計算するシミュレーター。

RAIDレベルごとの実効容量、冗長性、パフォーマンス特性を比較し、
ワークロードに応じた最適なRAID構成を提案する。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RAIDConfig:
    """RAID構成の定義"""
    level: str
    num_disks: int
    disk_size_tb: float
    description: str

    @property
    def raw_capacity_tb(self) -> float:
        """物理容量合計"""
        return self.num_disks * self.disk_size_tb

    @property
    def usable_capacity_tb(self) -> float:
        """実効容量（RAIDレベルに応じたパリティ/ミラーの消費を考慮）"""
        if self.level == "RAID 0":
            # ストライピングのみ: 全容量が使用可能
            return self.raw_capacity_tb
        elif self.level == "RAID 1":
            # ミラーリング: 半分の容量
            return self.raw_capacity_tb / 2
        elif self.level == "RAID 5":
            # シングルパリティ: ディスク1台分がパリティ
            return (self.num_disks - 1) * self.disk_size_tb
        elif self.level == "RAID 6":
            # ダブルパリティ: ディスク2台分がパリティ
            return (self.num_disks - 2) * self.disk_size_tb
        elif self.level == "RAID 10":
            # ミラー+ストライプ: 半分の容量
            return self.raw_capacity_tb / 2
        elif self.level == "RAID 50":
            # RAID 5のストライプ (グループ数2と仮定)
            groups = 2
            disks_per_group = self.num_disks // groups
            return groups * (disks_per_group - 1) * self.disk_size_tb
        elif self.level == "RAID 60":
            # RAID 6のストライプ (グループ数2と仮定)
            groups = 2
            disks_per_group = self.num_disks // groups
            return groups * (disks_per_group - 2) * self.disk_size_tb
        else:
            raise ValueError(f"未対応のRAIDレベル: {self.level}")

    @property
    def efficiency(self) -> float:
        """容量効率（実効容量/物理容量）"""
        return self.usable_capacity_tb / self.raw_capacity_tb * 100

    @property
    def fault_tolerance(self) -> int:
        """同時に故障しても耐えられるディスク数"""
        tolerance_map = {
            "RAID 0": 0,
            "RAID 1": self.num_disks - 1,  # 1台残れば復旧可能
            "RAID 5": 1,
            "RAID 6": 2,
            "RAID 10": 1,  # 各ミラーペアで1台まで
            "RAID 50": 1,  # 各グループで1台まで
            "RAID 60": 2,  # 各グループで2台まで
        }
        return tolerance_map.get(self.level, 0)

    @property
    def read_performance(self) -> str:
        """読み取り性能の傾向"""
        perf_map = {
            "RAID 0": f"非常に高速 (×{self.num_disks})",
            "RAID 1": f"高速 (×{self.num_disks})",
            "RAID 5": f"高速 (×{self.num_disks - 1})",
            "RAID 6": f"高速 (×{self.num_disks - 2})",
            "RAID 10": f"高速 (×{self.num_disks})",
            "RAID 50": "非常に高速",
            "RAID 60": "非常に高速",
        }
        return perf_map.get(self.level, "不明")

    @property
    def write_performance(self) -> str:
        """書き込み性能の傾向"""
        perf_map = {
            "RAID 0": f"非常に高速 (×{self.num_disks})",
            "RAID 1": "標準 (ミラー書込)",
            "RAID 5": "やや遅い (パリティ計算)",
            "RAID 6": "遅い (二重パリティ計算)",
            "RAID 10": f"中速 (×{self.num_disks // 2})",
            "RAID 50": "中速",
            "RAID 60": "やや遅い",
        }
        return perf_map.get(self.level, "不明")


def apply_real_world_overhead(usable_tb: float) -> dict:
    """
    実効容量に実際のオーバーヘッドを適用する。

    なぜこれらのオーバーヘッドが存在するのか:
    - SI/2進変換: メーカーはSI単位、OSは2進単位で表示
    - FS予約: rootが緊急時に使えるよう5%が予約される
    - メタデータ: inode, ジャーナル, スーパーブロック等
    """
    si_to_binary = usable_tb * 0.9313      # SI → 2進変換 (約7%減)
    after_fs_reserve = si_to_binary * 0.95  # FS予約 5%
    after_metadata = after_fs_reserve * 0.97  # メタデータ 3%

    return {
        "raw_usable_tb": usable_tb,
        "after_si_conversion_tib": si_to_binary,
        "after_fs_reserve_tib": after_fs_reserve,
        "after_metadata_tib": after_metadata,
        "total_overhead_pct": (1 - after_metadata / usable_tb) * 100,
    }


def compare_raid_configs() -> None:
    """様々なRAID構成を比較する"""
    configs = [
        RAIDConfig("RAID 0", 8, 4.0, "高速だが冗長性なし"),
        RAIDConfig("RAID 1", 2, 4.0, "完全ミラー"),
        RAIDConfig("RAID 5", 8, 4.0, "シングルパリティ"),
        RAIDConfig("RAID 6", 8, 4.0, "ダブルパリティ"),
        RAIDConfig("RAID 10", 8, 4.0, "ミラー+ストライプ"),
        RAIDConfig("RAID 50", 8, 4.0, "RAID5のストライプ"),
        RAIDConfig("RAID 60", 8, 4.0, "RAID6のストライプ"),
    ]

    print("=" * 80)
    print("  RAID構成比較 (4TB HDD × 8台)")
    print("=" * 80)
    print(f"  {'RAID':<9} {'物理':>6} {'実効':>6} {'効率':>6} "
          f"{'耐障害':>6} {'読取性能':<20} {'書込性能':<20}")
    print(f"  {'-'*9:<9} {'-'*6:>6} {'-'*6:>6} {'-'*6:>6} "
          f"{'-'*6:>6} {'-'*20:<20} {'-'*20:<20}")

    for cfg in configs:
        print(f"  {cfg.level:<9} {cfg.raw_capacity_tb:>5.0f}T "
              f"{cfg.usable_capacity_tb:>5.0f}T "
              f"{cfg.efficiency:>5.1f}% "
              f"{cfg.fault_tolerance:>4}台  "
              f"{cfg.read_performance:<20} {cfg.write_performance:<20}")

    # 実際のオーバーヘッド適用例
    print(f"\n\n  --- RAID 6 (8×4TB) の実効容量詳細 ---")
    raid6 = configs[3]  # RAID 6
    overhead = apply_real_world_overhead(raid6.usable_capacity_tb)

    print(f"  RAID実効容量:        {overhead['raw_usable_tb']:>8.1f} TB  (メーカー表記)")
    print(f"  SI→2進変換後:        {overhead['after_si_conversion_tib']:>8.2f} TiB")
    print(f"  FS予約(5%)控除後:    {overhead['after_fs_reserve_tib']:>8.2f} TiB")
    print(f"  メタデータ控除後:    {overhead['after_metadata_tib']:>8.2f} TiB")
    print(f"  合計オーバーヘッド:  {overhead['total_overhead_pct']:>8.1f}%")
    print(f"\n  物理32TB → 実効 {overhead['after_metadata_tib']:.2f} TiB "
          f"({overhead['after_metadata_tib']/32*100:.1f}%)")


if __name__ == "__main__":
    compare_raid_configs()
```

---

## 12. 実践演習

### 演習1: 容量見積もり（基礎）

**問題**: Instagram風の写真共有サービスを設計する。以下の前提条件で、1年間に必要なストレージ容量を算出せよ。

- DAU: 1000万人
- 1人平均3枚/日アップロード
- 1枚平均2MB (JPEG圧縮後)
- サムネイル: 1枚につき3サイズ生成（50KB, 150KB, 300KB）
- レプリカ: 3倍
- メタデータ: 1レコード500B

```
解答の指針:

  Step 1: 1日のデータ量
  ─────────────────────
  アップロード数: 10M × 3 = 30M 枚/日

  元画像: 30M × 2MB = 60TB/日
  サムネイル: 30M × (50KB + 150KB + 300KB) = 30M × 500KB = 15TB/日
  メタデータ: 30M × 500B = 15GB/日（無視できるレベル）

  合計: ≈ 75TB/日

  Step 2: 年間データ量
  ────────────────────
  75TB × 365 = 27,375TB ≈ 27.4PB/年

  Step 3: レプリカ考慮
  ────────────────────
  27.4PB × 3 = 82.2PB

  Step 4: オーバーヘッド考慮 (20%)
  ─────────────────────────────
  82.2PB × 1.2 = 98.6PB

  最終回答: 約100PB/年 のストレージが必要

  追加の考慮点:
  - CDN用のキャッシュストレージ
  - バックアップ用ストレージ
  - データ増加のバッファ (通常20-30%)
```

### 演習2: コスト計算（応用）

**問題**: 演習1のサービスのストレージを、ホット/ウォーム/コールドに階層化して設計し、年間コストを見積もれ。前提として、30日以内の画像は全アクセスの80%、30日〜6ヶ月の画像は15%、6ヶ月以上の画像は5%を占める。

```
解答の指針:

  Step 1: データの分類（1年運用後の状態）
  ────────────────────────────────────
  直近30日:     75TB/日 × 30 = 2,250 TB
  30日〜6ヶ月:  75TB/日 × 150 = 11,250 TB
  6ヶ月〜1年:   75TB/日 × 185 = 13,875 TB

  Step 2: レプリカ込みの容量
  ────────────────────────
  ホット (直近30日):    2,250 × 3 = 6,750 TB
  ウォーム (30d〜6m):   11,250 × 3 = 33,750 TB
  コールド (6m〜1y):    13,875 × 3 = 41,625 TB

  Step 3: 月額コスト計算
  ─────────────────────
  ホット (S3 Standard):  6,750 × $23 = $155,250
  ウォーム (S3 IA):      33,750 × $12.5 = $421,875
  コールド (Glacier):    41,625 × $4 = $166,500

  月額合計: $743,625
  年間: $8,923,500

  比較: 全て S3 Standard の場合
  82,125 TB × $23 = $1,888,875/月 = $22,666,500/年

  節約額: $22,666,500 - $8,923,500 = $13,743,000/年（61%削減）

  追加考慮:
  - APIリクエスト料金（GET/PUT）
  - データ転送料金（リージョン外）
  - コールドストレージからの復元料金
```

### 演習3: 帯域幅設計（発展）

**問題**: 演習1のサービスについて、ピーク時のリクエストQPS、必要な帯域幅、CDNの設計を含めたインフラ構成を提案せよ。読み書き比率は100:1、ピーク係数は3倍とする。

```
解答の指針:

  Step 1: QPS計算
  ────────────────
  書き込み（アップロード）:
  - 平均QPS: 30M / 86400 = 347 QPS
  - ピークQPS: 347 × 3 = 1,041 QPS

  読み取り（画像表示）:
  - 平均QPS: 347 × 100 = 34,700 QPS
  - ピークQPS: 34,700 × 3 = 104,100 QPS

  Step 2: 帯域幅計算
  ──────────────────
  書き込み帯域:
  - ピーク: 1,041 × 2MB = 2,082 MB/s ≈ 16.7 Gbps

  読み取り帯域（平均画像サイズ500KBと想定: サムネイル多め）:
  - ピーク: 104,100 × 500KB = 52,050 MB/s ≈ 416 Gbps

  Step 3: CDN設計
  ────────────────
  CDN無しでは416 Gbpsをオリジンサーバーで処理するのは非現実的。

  CDNキャッシュヒット率を95%と想定:
  オリジンへのリクエスト: 104,100 × 0.05 = 5,205 QPS
  オリジン帯域: 416 × 0.05 = 20.8 Gbps

  CDN構成:
  ┌─────────────┬────────────────────────────────┐
  │ レイヤー     │ 構成                            │
  ├─────────────┼────────────────────────────────┤
  │ エッジPOP   │ 世界50+拠点、SSDキャッシュ        │
  │ リージョナル │ 主要5リージョンにシールドサーバー  │
  │ オリジン     │ S3 + CloudFront / 自前CDN        │
  └─────────────┴────────────────────────────────┘

  Step 4: インフラ構成
  ────────────────────

  ┌─────────────────────────────────────────────────────┐
  │                    クライアント                       │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │              CDN (50+ エッジPOP)                     │
  │         キャッシュヒット率: 95%                       │
  └──────────┬────────────────────────┬─────────────────┘
             │ ミス時                  │ アップロード
  ┌──────────▼──────────┐   ┌────────▼────────────────┐
  │  画像配信サーバー     │   │  アップロードサーバー     │
  │  (読み取り最適化)     │   │  (書き込みパイプライン)   │
  └──────────┬──────────┘   └────────┬────────────────┘
             │                       │
  ┌──────────▼───────────────────────▼─────────────────┐
  │              オブジェクトストレージ (S3)              │
  │         ホット / ウォーム / コールド 階層             │
  │              合計: 約100PB/年                        │
  └─────────────────────────────────────────────────────┘
```

---

## 13. データ保持ポリシーとコンプライアンス

### 13.1 データライフサイクル管理

```
データ保持ポリシーの設計フレームワーク:

  なぜデータ保持ポリシーが必要か:
  ────────────────────────────
  1. コスト: 不要なデータを保持し続けると
     ストレージコストが際限なく増加する
  2. コンプライアンス: GDPR、個人情報保護法等により
     保持期間の制限が課される
  3. セキュリティ: データが存在する限りリスクが存在する
  4. パフォーマンス: データ量が増えるとクエリ性能が低下する

  典型的なデータ保持ポリシー:
  ┌──────────────────┬───────────┬────────────────────┐
  │ データ種別        │ 保持期間   │ 根拠               │
  ├──────────────────┼───────────┼────────────────────┤
  │ アプリログ        │ 90日      │ デバッグに十分      │
  │ アクセスログ      │ 1年       │ セキュリティ監査    │
  │ トランザクション  │ 7年       │ 税法・会計基準      │
  │ ユーザーデータ    │ 退会+30日 │ GDPR削除権         │
  │ バックアップ      │ 世代管理   │ 障害復旧            │
  │ 監査ログ         │ 10年      │ 金融規制            │
  │ 医療記録         │ 永久      │ 医療法              │
  └──────────────────┴───────────┴────────────────────┘

  バックアップの世代管理:
  ──────────────────────
  GFS (Grandfather-Father-Son) 方式:
  - 日次 (Son):     7世代保持
  - 週次 (Father):  4世代保持
  - 月次 (Grand):   12世代保持
  - 年次:           5世代保持

  必要なバックアップストレージ:
  = データ量 × (7 + 4 + 12 + 5) = データ量 × 28世代
  ただし増分バックアップを使用すれば大幅に削減可能。
  増分バックアップの場合、変更差分のみ ≈ フルの5-20%

  日次変更率10%の場合:
  フル1回 + 増分6回 = 1.0 + 0.6 = 1.6 (日次7世代分)
  全体: ≈ データ量 × 5〜8倍 が想定される範囲
```

### 13.2 GDPR/個人情報保護における容量への影響

```
GDPR準拠のストレージ設計:

  データ最小化の原則 (Article 5(1)(c)):
  ────────────────────────────────────
  「目的に関連し、必要かつ最小限のデータのみを処理する」

  ストレージへの影響:
  ┌──────────────────────┬──────────────────────────────┐
  │ GDPR要件             │ ストレージ設計への影響         │
  ├──────────────────────┼──────────────────────────────┤
  │ 削除権(Right to be   │ 論理削除ではなく物理削除が      │
  │ forgotten)           │ 必要な場合がある。バックアップ  │
  │                      │ からの削除も考慮が必要          │
  ├──────────────────────┼──────────────────────────────┤
  │ データポータビリティ  │ エクスポート機能のための一時的  │
  │                      │ なストレージが必要              │
  ├──────────────────────┼──────────────────────────────┤
  │ 保存期間制限          │ TTL (Time-To-Live) による      │
  │                      │ 自動削除の仕組みが必要          │
  ├──────────────────────┼──────────────────────────────┤
  │ 暗号化               │ 暗号化によるストレージ          │
  │                      │ オーバーヘッド(通常1-5%)        │
  └──────────────────────┴──────────────────────────────┘

  Crypto Shredding (暗号学的破砕):
  ──────────────────────────────
  個別データの物理削除が困難な場合（バックアップテープ等）、
  データを暗号化しておき、暗号鍵を削除することで
  事実上データを復元不可能にする技術。
  - 各ユーザーに固有の暗号鍵を割り当て
  - 削除要求時に暗号鍵のみを削除
  - バックアップ内のデータは暗号化されたまま残るが復号不可能
```

---

## 14. ストレージ性能のベンチマーク手法

### 14.1 主要なベンチマーク指標

```
ストレージ性能の4つの主要指標:

  1. スループット (Throughput)
     ─────────────────────────
     - 単位時間あたりのデータ転送量 (MB/s, GB/s)
     - 大きなファイルの連続読み書きで重要
     - 動画編集、バックアップ、データ分析に影響

  2. IOPS (Input/Output Operations Per Second)
     ─────────────────────────────────────────
     - 1秒あたりの読み書き操作数
     - 小さなファイルのランダムアクセスで重要
     - DB操作、メール、Webサービスに影響

  3. レイテンシ (Latency)
     ────────────────────
     - 1回のI/O操作にかかる時間 (μs, ms)
     - リアルタイムシステムで重要
     - IOPS = 1 / レイテンシ × 同時実行数

  4. 帯域幅利用率 (Bandwidth Utilization)
     ────────────────────────────────────
     - 理論最大値に対する実際の利用率
     - インターフェースのボトルネック検出に使用

  インターフェース別の理論最大帯域幅:
  ┌──────────────┬────────────────┬─────────────────┐
  │ インターフェース│ 最大帯域幅      │ 主な用途         │
  ├──────────────┼────────────────┼─────────────────┤
  │ SATA III     │ 6 Gbps (600MB/s)│ コンシューマSSD  │
  │ SAS-3        │ 12 Gbps        │ エンタープライズ  │
  │ NVMe (PCIe 3)│ 32 Gbps (3.5GB/s)│ 高速SSD        │
  │ NVMe (PCIe 4)│ 64 Gbps (7GB/s) │ 最新SSD         │
  │ NVMe (PCIe 5)│ 128 Gbps (14GB/s)│ 次世代SSD       │
  │ U.2/U.3     │ PCIe 4相当     │ データセンター    │
  │ NVMe-oF     │ 100 Gbps+      │ ネットワーク接続  │
  └──────────────┴────────────────┴─────────────────┘
```

### コード例7: 簡易ストレージベンチマーク (Python)

```python
#!/usr/bin/env python3
"""
簡易ストレージベンチマークツール。

シーケンシャル読み書きとランダム読み書きの性能を測定する。
本番環境のベンチマークにはfio等の専用ツールを推奨するが、
ここではストレージ性能の概念を理解するための教育用実装を示す。
"""

import os
import time
import tempfile
import random
from typing import Tuple


def benchmark_sequential_write(
    file_path: str,
    total_mb: int = 256,
    block_size_kb: int = 1024,
) -> Tuple[float, float]:
    """
    シーケンシャル書き込みベンチマーク。

    なぜブロックサイズが重要なのか:
    - 小さいブロック: システムコールのオーバーヘッドが支配的
    - 大きいブロック: I/Oのバルク転送が活きてスループット向上
    - 一般的なベンチマークでは1MBブロックを使用

    Args:
        file_path: 書き込み先ファイルパス
        total_mb: 書き込む合計サイズ (MB)
        block_size_kb: ブロックサイズ (KB)

    Returns:
        (経過時間(秒), スループット(MB/s))
    """
    block_size = block_size_kb * 1024
    num_blocks = (total_mb * 1024 * 1024) // block_size
    data = os.urandom(block_size)  # ランダムデータ（圧縮が効かないように）

    start_time = time.perf_counter()

    with open(file_path, "wb") as f:
        for _ in range(num_blocks):
            f.write(data)
        f.flush()
        os.fsync(f.fileno())  # ディスクへの書き込みを保証

    elapsed = time.perf_counter() - start_time
    throughput = total_mb / elapsed

    return elapsed, throughput


def benchmark_sequential_read(
    file_path: str,
    block_size_kb: int = 1024,
) -> Tuple[float, float]:
    """
    シーケンシャル読み取りベンチマーク。

    Returns:
        (経過時間(秒), スループット(MB/s))
    """
    file_size = os.path.getsize(file_path)
    block_size = block_size_kb * 1024

    # OSのキャッシュをクリアするのは特権操作のため、
    # ここでは注記にとどめる。
    # 正確な測定には: echo 3 > /proc/sys/vm/drop_caches (Linux, root)

    start_time = time.perf_counter()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(block_size)
            if not data:
                break

    elapsed = time.perf_counter() - start_time
    throughput = (file_size / (1024 * 1024)) / elapsed

    return elapsed, throughput


def benchmark_random_read(
    file_path: str,
    num_operations: int = 10000,
    block_size: int = 4096,
) -> Tuple[float, float, float]:
    """
    ランダム読み取りベンチマーク（4KBブロック）。

    なぜ4KBなのか:
    - ファイルシステムのブロックサイズが一般的に4KB
    - SSDの最小読み取り単位（ページサイズ）が4-16KB
    - DB操作の多くが4KB単位のランダムアクセス

    Returns:
        (経過時間(秒), IOPS, 平均レイテンシ(μs))
    """
    file_size = os.path.getsize(file_path)
    max_offset = file_size - block_size

    if max_offset <= 0:
        raise ValueError("ファイルサイズがブロックサイズより小さいです")

    # ランダムなオフセットを事前に生成
    offsets = [random.randint(0, max_offset) for _ in range(num_operations)]

    start_time = time.perf_counter()

    with open(file_path, "rb") as f:
        for offset in offsets:
            f.seek(offset)
            f.read(block_size)

    elapsed = time.perf_counter() - start_time
    iops = num_operations / elapsed
    avg_latency_us = (elapsed / num_operations) * 1_000_000

    return elapsed, iops, avg_latency_us


def run_benchmark() -> None:
    """ベンチマークを実行して結果を表示"""
    total_mb = 256
    print("=" * 60)
    print("  簡易ストレージベンチマーク")
    print("=" * 60)
    print(f"  テストサイズ: {total_mb} MB")
    print(f"  注: OSキャッシュの影響を受ける場合があります。")
    print(f"  正確な計測にはfio等の専用ツールを使用してください。")
    print()

    # 一時ファイルでベンチマーク実行
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bench") as tmp:
        file_path = tmp.name

    try:
        # シーケンシャル書き込み
        print("  [1/3] シーケンシャル書き込み...")
        elapsed, throughput = benchmark_sequential_write(file_path, total_mb)
        print(f"        経過時間: {elapsed:.2f}秒")
        print(f"        スループット: {throughput:.1f} MB/s")
        print()

        # シーケンシャル読み取り
        print("  [2/3] シーケンシャル読み取り...")
        elapsed, throughput = benchmark_sequential_read(file_path)
        print(f"        経過時間: {elapsed:.2f}秒")
        print(f"        スループット: {throughput:.1f} MB/s")
        print()

        # ランダム読み取り
        print("  [3/3] ランダム読み取り (4KB ×10000)...")
        elapsed, iops, latency = benchmark_random_read(file_path)
        print(f"        経過時間: {elapsed:.2f}秒")
        print(f"        IOPS: {iops:,.0f}")
        print(f"        平均レイテンシ: {latency:.1f} μs")

    finally:
        os.unlink(file_path)  # テストファイルを削除

    print()
    print("  参考値 (NVMe SSD の想定される性能帯):")
    print("  - シーケンシャル読取: 3,000-7,000 MB/s")
    print("  - シーケンシャル書込: 2,000-5,000 MB/s")
    print("  - ランダム4K IOPS:   500,000-1,000,000")
    print()
    print("  参考値 (SATA HDD の想定される性能帯):")
    print("  - シーケンシャル読取: 100-200 MB/s")
    print("  - シーケンシャル書込: 100-200 MB/s")
    print("  - ランダム4K IOPS:   100-200")


if __name__ == "__main__":
    run_benchmark()
```

---

## 15. 将来のストレージ技術

### 15.1 次世代ストレージ技術のロードマップ

```
次世代ストレージ技術 (2024〜2030年代):

  1. DNA ストレージ
     ──────────────
     - 密度: 1グラムのDNAに 215PB のデータを格納可能
     - 耐久性: 適切な環境で数千年保存可能
     - 課題: 書き込み/読み取りが極めて遅い（時間〜日単位）
     - 現状: コスト $3,500/MB (2024) → 目標 $1/GB (2030年代)
     - 用途: 超長期アーカイブ（人類遺産の保存等）
     - MicrosoftとUniversity of Washingtonが研究をリード

  2. ガラスストレージ (Project Silica)
     ─────────────────────────────
     - Microsoftの研究プロジェクト
     - フェムト秒レーザーでガラスにナノスケールの構造を刻む
     - 耐久性: 理論上数万年
     - 密度: 石英ガラス1cm^3に数百GB
     - 用途: エンタープライズアーカイブ

  3. HAMR / MAMR HDD
     ────────────────
     - Heat-Assisted Magnetic Recording
     - Microwave-Assisted Magnetic Recording
     - 従来のHDDの記録密度限界を突破
     - 目標: 1プラッタあたり5-6TB (現在は約2.5TB)
     - 2025年以降に50TB+ HDDが想定される

  4. CXL (Compute Express Link) メモリ
     ──────────────────────────────
     - メモリとストレージの境界を曖昧にする技術
     - CPUからTB単位のメモリプールにアクセス可能
     - レイテンシ: DRAMに近い低遅延
     - 用途: インメモリデータベース、AI推論

  5. Computational Storage
     ─────────────────────
     - ストレージデバイス内にプロセッサを搭載
     - データの移動なしにフィルタリング・前処理を実行
     - 帯域幅のボトルネックを解消
     - 用途: ビッグデータ分析、ログ検索

  容量単価の予測:
  ┌──────┬────────────────┬──────────────┐
  │ 年    │ $/GB (HDD)     │ $/GB (SSD)   │
  ├──────┼────────────────┼──────────────┤
  │ 2024 │ $0.015         │ $0.06        │
  │ 2026 │ $0.010         │ $0.04        │
  │ 2028 │ $0.007         │ $0.025       │
  │ 2030 │ $0.005         │ $0.015       │
  └──────┴────────────────┴──────────────┘
  ※ 上記は業界トレンドから想定される方向性
```

---

## 16. FAQ

### Q1: 「1GBのRAMで何件のレコードが保持できますか？」の見積もり方は？

**A**: レコードサイズを見積もり、オーバーヘッドを含めて計算する。

例えばユーザーレコード:
- ID: 8B（int64）
- 名前: 100B（可変長文字列 + ヘッダ）
- メール: 100B
- メタデータ: 100B
- **レコード合計: 約 300B**

1GiB / 300B ≈ 3,580,000 レコード（約360万件）

ただしこれは理論値。実際には以下のオーバーヘッドを考慮する:
- メモリアロケータのオーバーヘッド: 10-20%
- ハッシュマップ/B-Treeのインデックス構造: 元データの2-3倍
- ガベージコレクション用のヘッドルーム（Java/Go等）: 30-50%

安全な見積もり: **理論値の 30-50% = 100万〜150万レコード**

### Q2: クラウドとオンプレのストレージコスト、どちらが安い？

**A**: 規模と運用体制によって大きく異なる。

| 規模 | 推奨 | 理由 |
|------|------|------|
| 〜数TB | クラウド | 管理コスト削減、初期投資不要 |
| 数TB〜数百TB | ハイブリッド | ホットはクラウド、コールドはオンプレ |
| PB以上 | オンプレ中心 | 容量単価でクラウドを大幅に下回る |

具体的な分岐点の計算例 (1PBのストレージ):
- **S3 Standard**: 1024TB × $23 = $23,552/月 = $282,624/年
- **オンプレHDD**: 100台 × 12TB × $300/台 = $30,000 (初期投資) + 電力・冷却・人件費
- 3年TCO（Total Cost of Ownership）で比較すると、PB級ではオンプレが安くなることが多い

Netflix、Dropboxは自社CDN/ストレージへの移行で年間数百万ドルのコスト削減を達成した事例がある。ただしオンプレには人件費、データセンター賃料、電気代、ハードウェア更新費用が加わることに注意。

### Q3: 全てのデータを永遠に保持すべきですか？

**A**: いいえ。データ保持ポリシーを定めるべきである。

理由:
1. **コスト**: データは増え続ける。保持コストも増え続ける。
2. **法令遵守**: GDPRでは「目的を達成したら削除する」義務がある。
3. **リスク**: 保持するデータが多いほど漏洩時の被害が大きい。

推奨される保持期間の目安:
- **操作ログ**: 90日〜1年（デバッグ・監視用）
- **ユーザーデータ**: 退会後30〜90日で物理削除
- **取引データ**: 7年（税法上の保存義務）
- **バックアップ**: GFS方式で世代管理

バックアップの世代管理例:
- 日次: 7世代
- 週次: 4世代
- 月次: 12世代
- 年次: 法令に応じて（通常5〜10年）

コストとコンプライアンスのバランスで保持期間を決定する。迷ったらデータ保護責任者（DPO）やリーガルチームに相談する。

### Q4: ストレージのRAIDレベルはどう選べばよいですか？

**A**: ワークロードの特性と要件に応じて選択する。

| 用途 | 推奨RAID | 理由 |
|------|----------|------|
| 一時データ/キャッシュ | RAID 0 | 速度重視、冗長性不要 |
| OS/ブート | RAID 1 | シンプルで高信頼 |
| 一般的なファイルサーバー | RAID 5 | 容量効率と冗長性のバランス |
| 大容量アーカイブ | RAID 6 | 大容量HDDのリビルド中の二重障害対策 |
| DB (OLTP) | RAID 10 | 高IOPS + 冗長性 |

注意: 大容量HDD（8TB以上）ではRAID 5を避けるべきである。リビルド中（数十時間）にもう1台故障する確率が無視できないほど高くなるため、RAID 6またはRAID 10を推奨する。

### Q5: SSDの寿命はどの程度心配すべきですか？

**A**: 一般的な用途では心配する必要はほぼない。ただし書き込みが極端に多いワークロードでは注意が必要。

計算例 (Samsung 870 EVO 1TB, TBW 600TB):
- 1日10GB書き込み → 600,000GB / 10GB = 60,000日 ≈ 164年
- 1日100GB書き込み → 6,000日 ≈ 16年
- 1日1TB書き込み → 600日 ≈ 1.6年（要注意）

Write Amplification Factor (WAF) を2-3倍考慮しても、一般的なワークロード（1日数十GB）では10年以上持つ。ただし以下の用途では監視が必要:
- 高頻度ログ書き込み
- データベースのWAL (Write-Ahead Log)
- 仮想化環境でのスワップ

S.M.A.R.T. 情報で残りTBWを定期的に確認することを推奨する。

---

## 17. まとめ

| 概念 | ポイント |
|------|---------|
| 単位系 | SI接頭辞（10の累乗）と2進接頭辞（2の累乗）の混同に注意。TBレベルで約10%の差 |
| テキスト | 日本語1文字≈3B (UTF-8)。小説1冊≈300KB。ログ1行≈100B-1KB |
| 画像 | 非圧縮サイズ = 幅×高×色深度。JPEG圧縮で10-40倍。Web向けはWebP/AVIFを検討 |
| 音声 | CD品質非圧縮≈10MB/分。MP3 128kbps≈1MB/分。Opus/AACが効率的 |
| 動画 | 1080p非圧縮≈186MB/秒。H.264圧縮後≈35MB/分。コーデック世代で2倍の差 |
| ストレージ技術 | SSD vs HDD: IOPSで5000倍の差。容量単価はHDDが安い。適材適所 |
| 見積もり手法 | DAU×行動×データサイズ×レプリカ×保持期間。ピーク係数3-10倍 |
| 階層化 | ホット/ウォーム/コールドの分類でストレージコストを60-80%削減可能 |
| コスト | RAM: $3,000/TB/月 → SSD: $50 → HDD: $5 → テープ: $0.5 |
| 帯域幅 | Gbps÷8=GB/s。実効は理論値の80-90%。大量データは物理輸送も選択肢 |
| データ保持 | 無限保持は悪手。コスト・コンプライアンス・リスクのバランスで決定 |

---

## 次に読むべきガイド

→ [[06-brain-vs-computer.md]] --- 脳とコンピュータの比較

---

## 参考文献

1. Dean, J. & Barroso, L. "The Tail at Scale." Communications of the ACM, 56(2), 2013. -- レイテンシの数値とシステム設計への応用の原典。Jeff Deanの「Numbers Every Programmer Should Know」はGoogleの内部講演から広まったもの。

2. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly Media, 2017. -- 分散システムにおけるストレージ、レプリケーション、パーティショニングの包括的な解説。特に第2章「データモデルとクエリ言語」と第3章「ストレージと検索」がストレージ容量設計に直結する。

3. Xu, A. "System Design Interview - An Insider's Guide." Byte Code LLC, 2020. -- 第2章「Back-of-the-envelope Estimation」がシステム設計面接での容量見積もり手法を詳述。本章の見積もりフレームワークはこの書籍に基づいている。

4. Patterson, D. & Hennessy, J. "Computer Organization and Design: RISC-V Edition." Morgan Kaufmann, 6th Edition, 2021. -- メモリ階層（キャッシュ、メインメモリ、ディスク）の設計原理を理論と実装の両面から解説する教科書。ストレージ技術の物理的原理についてはAppendix Cが詳しい。

5. AWS Documentation. "Amazon S3 Storage Classes." https://aws.amazon.com/s3/storage-classes/ -- S3の各ストレージクラス（Standard, IA, Glacier, Deep Archive）の仕様と料金体系。階層型ストレージ設計の基盤となる情報源。

6. Kryder, M. "Kryder's Law." Scientific American, 2005. -- ストレージ密度の指数関数的成長（ムーアの法則のストレージ版）を提唱。ただし2010年代以降は成長率が鈍化しており、HAMRなどの新技術による打開が期待されている。

7. Church, G., Gao, Y., & Kosuri, S. "Next-Generation Digital Information Storage in DNA." Science, 337(6102), 2012. -- DNAストレージの先駆的研究。1グラムのDNAに理論上 455EB を格納できることを示した論文。

---

## 用語集

| 用語 | 定義 |
|------|------|
| SI接頭辞 | 国際単位系の接頭辞。kilo=10^3, mega=10^6, giga=10^9 |
| 2進接頭辞 | IEC 60027-2で定義。kibi=2^10, mebi=2^20, gibi=2^30 |
| IOPS | Input/Output Operations Per Second。ストレージの操作速度 |
| TBW | Terabytes Written。SSDの生涯書き込み量 |
| DWPD | Drive Writes Per Day。SSDの1日あたりの全容量書き換え回数 |
| WAF | Write Amplification Factor。論理書込に対する物理書込の比率 |
| RAID | Redundant Array of Independent Disks。複数ディスクの冗長化 |
| HAMR | Heat-Assisted Magnetic Recording。熱アシスト磁気記録 |
| GFS | Grandfather-Father-Son。バックアップ世代管理方式 |
| Sneakernet | 物理メディアを運んでデータを転送する方法の俗称 |
| Crypto Shredding | 暗号鍵を削除してデータを復元不可能にする技術 |
| LCP | Largest Contentful Paint。Webページの主要コンテンツの表示時間 |
| TCO | Total Cost of Ownership。所有にかかる総コスト |
