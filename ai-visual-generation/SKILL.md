# AI ビジュアル生成

> AI が画像・映像制作を革新する。Stable Diffusion、DALL-E、Midjourney から動画生成（Sora）、3D モデリングまで、AI ビジュアル生成の全てを解説する。

## このSkillの対象者

- AI 画像・映像生成技術を学びたいクリエイター
- プロダクトに AI ビジュアル生成を組み込みたいエンジニア
- AI アート・デザインに興味がある方

## 前提知識

- AI/ML の基礎概念
- 画像処理の基礎知識

## 学習ガイド

### 00-fundamentals — 画像生成 AI の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-fundamentals/00-generative-ai-overview.md]] | 生成 AI の歴史、Diffusion Model、GAN、VAE |
| 01 | [[docs/00-fundamentals/01-prompt-engineering-visual.md]] | ビジュアルプロンプト技法、スタイル指定、ネガティブプロンプト |
| 02 | [[docs/00-fundamentals/02-ethical-considerations.md]] | 著作権、ディープフェイク、バイアス、規制動向 |

### 01-image — 画像生成

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-image/00-stable-diffusion.md]] | Stable Diffusion（SDXL/SD3/Flux）、ComfyUI、LoRA |
| 01 | [[docs/01-image/01-commercial-services.md]] | DALL-E 3、Midjourney、Adobe Firefly、Ideogram |
| 02 | [[docs/01-image/02-image-editing-ai.md]] | Inpainting、Outpainting、Image-to-Image、ControlNet |
| 03 | [[docs/01-image/03-api-integration.md]] | 画像生成 API（OpenAI/Stability AI/Replicate）統合 |

### 02-video — 動画生成

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-video/00-ai-video-generation.md]] | Sora、Runway Gen-3、Pika、テキストから動画 |
| 01 | [[docs/02-video/01-video-editing-ai.md]] | AI 動画編集、自動字幕、シーン検出、スタイル変換 |
| 02 | [[docs/02-video/02-real-time-generation.md]] | リアルタイム生成、ライブストリーミング AI、AR フィルター |

### 03-3d — 3D 生成

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-3d/00-3d-generation.md]] | テキスト→3D、画像→3D、NeRF、3D Gaussian Splatting |
| 01 | [[docs/03-3d/01-game-and-metaverse.md]] | ゲームアセット生成、仮想空間、アバター生成 |

## クイックリファレンス

```
AI 画像生成サービス比較:
  Midjourney:       最高品質、Discord ベース
  DALL-E 3:         API 統合容易、ChatGPT 連携
  Stable Diffusion: オープンソース、カスタマイズ自由
  Adobe Firefly:    商用利用安全、Adobe 統合
  Flux:             最新オープンモデル、高品質
```

## 参考文献

1. Rombach, R. et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR, 2022.
2. OpenAI. "DALL-E 3." openai.com, 2024.
3. Stability AI. "Stable Diffusion." stability.ai, 2024.
