# AI 音声・音楽生成

> AI が音の創造を民主化する。テキスト音声合成、音声クローニング、AI 作曲、サウンドデザインまで、AI 音声・音楽生成の全てを解説する。

## このSkillの対象者

- AI 音声・音楽生成技術を学びたいクリエイター
- 音声合成をアプリに組み込みたいエンジニア
- AI 音楽制作に興味がある方

## 前提知識

- 音声・音楽の基礎概念
- Python の基礎知識

## 学習ガイド

### 00-fundamentals — 音声 AI の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-fundamentals/00-audio-ai-overview.md]] | 音声 AI の全体像、歴史、現在のトレンド |
| 01 | [[docs/00-fundamentals/01-audio-basics.md]] | 音声のデジタル表現、サンプリング、スペクトログラム |
| 02 | [[docs/00-fundamentals/02-speech-recognition.md]] | 音声認識（Whisper/Deepgram）、文字起こし、多言語対応 |

### 01-music — AI 音楽生成

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-music/00-ai-music-generation.md]] | Suno、Udio、MusicLM、テキストから音楽生成 |
| 01 | [[docs/01-music/01-music-production-ai.md]] | AI 作曲支援、メロディ生成、アレンジ、マスタリング |
| 02 | [[docs/01-music/02-stem-separation.md]] | ステム分離（Demucs）、リミックス、サンプリング |

### 02-voice — AI 音声合成

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-voice/00-tts.md]] | テキスト音声合成（ElevenLabs/OpenAI TTS/VOICEVOX） |
| 01 | [[docs/02-voice/01-voice-cloning.md]] | 音声クローニング、リアルタイム変換、倫理的考慮 |
| 02 | [[docs/02-voice/02-voice-api-integration.md]] | 音声 API 統合、ストリーミング、多言語、コスト比較 |

### 03-tools — ツールとワークフロー

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-tools/00-daw-and-ai.md]] | DAW × AI プラグイン、Ableton/Logic + AI ツール |
| 01 | [[docs/03-tools/01-podcast-and-audiobook.md]] | ポッドキャスト制作 AI、オーディオブック生成 |
| 02 | [[docs/03-tools/02-sound-design-ai.md]] | AI サウンドデザイン、効果音生成、環境音 |

## クイックリファレンス

```
AI 音声サービス比較:
  TTS:     ElevenLabs（高品質）/ OpenAI TTS（API統合）/ VOICEVOX（無料・日本語）
  音楽:    Suno（歌詞→楽曲）/ Udio（高品質）/ Stable Audio
  認識:    Whisper（オープン）/ Deepgram（API）/ Google STT
  分離:    Demucs / Spleeter
```

## 参考文献

1. Radford, A. et al. "Robust Speech Recognition via Large-Scale Weak Supervision." OpenAI, 2023.
2. ElevenLabs. "Documentation." elevenlabs.io/docs, 2024.
3. Suno. "Documentation." suno.com, 2024.
