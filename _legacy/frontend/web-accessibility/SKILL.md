---
name: web-accessibility
description: Webアクセシビリティ対応ガイド。WCAG 2.1準拠、セマンティックHTML、ARIA属性、キーボード操作、スクリーンリーダー対応など、誰もが使えるWebアプリケーション構築のベストプラクティス。
---

# Web Accessibility Skill

## 📋 目次

1. [概要](#概要)
2. [詳細ガイド](#詳細ガイド)
3. [いつ使うか](#いつ使うか)
4. [WCAG 2.1基準](#wcag-21基準)
5. [セマンティックHTML](#セマンティックhtml)
6. [ARIA属性](#aria属性)
7. [キーボード操作](#キーボード操作)
8. [カラーコントラスト](#カラーコントラスト)
9. [実践例](#実践例)
10. [テストツール](#テストツール)
11. [Agent連携](#agent連携)

---

## 概要

このSkillは、Webアクセシビリティをカバーします：

- **WCAG 2.1** - アクセシビリティ基準（A, AA, AAA）
- **セマンティックHTML** - 適切なHTML要素の使用
- **ARIA** - スクリーンリーダー対応
- **キーボード操作** - マウスなしで操作可能
- **カラーコントラスト** - 視認性確保
- **フォーム** - アクセシブルなフォーム設計

## 📚 公式ドキュメント・参考リソース

**このガイドで学べること**: WCAG準拠方法、セマンティックHTML、ARIA属性、キーボード操作、アクセシビリティテスト
**公式で確認すべきこと**: 最新のWCAG基準、ブラウザサポート、支援技術の動向

### 主要な公式ドキュメント

- **[WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)** - W3Cアクセシビリティガイドライン
  - [Understanding WCAG 2.1](https://www.w3.org/WAI/WCAG21/Understanding/)
  - [Techniques](https://www.w3.org/WAI/WCAG21/Techniques/)

- **[WAI-ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/)** - ARIAパターンガイド
  - [Patterns](https://www.w3.org/WAI/ARIA/apg/patterns/)

- **[MDN Accessibility](https://developer.mozilla.org/en-US/docs/Web/Accessibility)** - MDNアクセシビリティガイド
  - [ARIA](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA)

- **[Apple Accessibility](https://developer.apple.com/accessibility/)** - iOSアクセシビリティ

### 関連リソース

- **[a11y Project](https://www.a11yproject.com/)** - アクセシビリティチェックリスト
- **[WebAIM](https://webaim.org/)** - アクセシビリティリソース
- **[axe DevTools](https://www.deque.com/axe/)** - 自動アクセシビリティ検査

---

## 詳細ガイド

以下の完全ガイドで、実践的なアクセシビリティ実装を学べます：

### 1. [WCAG 2.1準拠完全ガイド](./guides/wcag/wcag-compliance-complete.md)

**26,000文字 | React 18.2.0+ | Next.js 14.0+**

- WCAG 2.1の4原則（POUR）完全解説
- Level A, AA, AAA の全基準と実装例
- 実際のコード例（Modal, Form, Image, Video, Table等）
- **10のトラブルシューティング事例**
- 実測データ（Lighthouse 68→95点、タスク完了率 45%→92%）

### 2. [ARIAパターン完全ガイド](./guides/aria/aria-patterns-complete.md)

**27,500文字 | WAI-ARIA 1.2 | React 18.2.0+**

- **20の主要ARIAパターン**（Accordion, Tabs, Modal, Dropdown, Combobox, Tooltip, Alert, Live Region, Breadcrumb, Progress Bar, Slider, Checkbox, Radio, Switch, Tree View, Listbox, Disclosure, Link, Carousel, Form Validation）
- 完全な動作コード（フォーカス管理、キーボード操作、ARIA属性）
- スクリーンリーダーテスト結果（NVDA, JAWS, VoiceOver, TalkBack）
- **12のトラブルシューティング事例**
- 実測データ（axe-coreエラー -97.6%、キーボード操作完了率 +128%）

### 3. [アクセシビリティテスト完全ガイド](./guides/testing/accessibility-testing-complete.md)

**28,000文字 | axe-core 4.8.0+ | Playwright 1.40.0+**

- **自動テストツール5種**（axe-core, Lighthouse CI, Playwright + axe, pa11y, eslint-plugin-jsx-a11y）
- 手動テスト手順（キーボード、色覚多様性、コントラスト比、ズーム）
- **スクリーンリーダーテスト詳細**（NVDA, JAWS, VoiceOver, TalkBackの操作方法）
- CI/CD統合（GitHub Actions設定、pre-commit hook）
- **12のトラブルシューティング事例**
- 実測データ（手動テスト工数 -75%、バグ報告 -91.7%、ユーザー満足度 +229%）

**合計: 81,500文字 | 3ガイド**

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] 新規UIコンポーネント作成時
- [ ] フォーム実装時
- [ ] モーダル・ダイアログ実装時
- [ ] ナビゲーション実装時

### 🔄 定期的に

- [ ] リリース前（アクセシビリティ監査）
- [ ] 月次（自動テスト実行）

---

## WCAG 2.1基準

### レベル

| レベル | 説明 | 要件 |
|-------|------|------|
| **A** | 最低限 | 基本的なアクセシビリティ |
| **AA** | 推奨（法的要件の場合多い） | 中程度のアクセシビリティ |
| **AAA** | 最高レベル | 高度なアクセシビリティ |

### 主要原則（POUR）

1. **Perceivable（知覚可能）** - 情報が認識できる
2. **Operable（操作可能）** - UIコンポーネントが操作できる
3. **Understandable（理解可能）** - 情報と操作が理解できる
4. **Robust（堅牢）** - 様々な技術で解釈できる

---

## セマンティックHTML

### ✅ 適切なHTML要素を使用

```html
<!-- ❌ 悪い例 -->
<div onclick="navigate()">Home</div>
<div onclick="submit()">Submit</div>

<!-- ✅ 良い例 -->
<nav>
  <a href="/">Home</a>
</nav>
<button type="submit">Submit</button>
```

### ランドマーク要素

```html
<header>
  <nav aria-label="Main navigation">
    <!-- ナビゲーション -->
  </nav>
</header>

<main>
  <article>
    <h1>Article Title</h1>
    <section>
      <h2>Section Title</h2>
      <!-- コンテンツ -->
    </section>
  </article>

  <aside>
    <!-- サイドバー -->
  </aside>
</main>

<footer>
  <!-- フッター -->
</footer>
```

### 見出し階層

```html
<!-- ✅ 良い例（階層的） -->
<h1>Page Title</h1>
  <h2>Section 1</h2>
    <h3>Subsection 1.1</h3>
  <h2>Section 2</h2>

<!-- ❌ 悪い例（階層がスキップ） -->
<h1>Page Title</h1>
  <h3>Section 1</h3> <!-- h2をスキップ -->
```

---

## ARIA属性

### aria-label / aria-labelledby

```tsx
// アイコンボタン
<button aria-label="Close">
  <XIcon />
</button>

// 複数ナビゲーション
<nav aria-label="Main navigation">
  {/* ... */}
</nav>
<nav aria-label="Footer navigation">
  {/* ... */}
</nav>

// aria-labelledby（既存の要素を参照）
<div role="dialog" aria-labelledby="dialog-title">
  <h2 id="dialog-title">Confirmation</h2>
  {/* ... */}
</div>
```

### aria-describedby

```tsx
<input
  type="email"
  aria-describedby="email-help"
/>
<span id="email-help">
  We'll never share your email.
</span>
```

### aria-live（動的コンテンツ）

```tsx
// リアルタイム通知
<div aria-live="polite" aria-atomic="true">
  {status}
</div>

// 緊急通知
<div aria-live="assertive">
  {error}
</div>
```

### role属性

```tsx
// カスタムコンポーネント
<div role="button" tabIndex={0} onClick={handleClick} onKeyDown={handleKeyDown}>
  Click me
</div>

// リスト
<ul role="list">
  <li role="listitem">Item 1</li>
  <li role="listitem">Item 2</li>
</ul>
```

---

## キーボード操作

### フォーカス可能要素

```tsx
// ✅ ネイティブ要素（自動的にフォーカス可能）
<button>Click</button>
<a href="/">Link</a>
<input type="text" />

// ✅ カスタム要素（tabIndexを追加）
<div
  role="button"
  tabIndex={0}
  onClick={handleClick}
  onKeyDown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      handleClick()
    }
  }}
>
  Custom Button
</div>
```

### フォーカストラップ（モーダル）

```tsx
import { useEffect, useRef } from 'react'

function Modal({ onClose }: { onClose: () => void }) {
  const modalRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }

      // Tab キーでフォーカストラップ
      if (e.key === 'Tab') {
        const focusableElements = modalRef.current?.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        )
        if (!focusableElements) return

        const firstElement = focusableElements[0] as HTMLElement
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement

        if (e.shiftKey && document.activeElement === firstElement) {
          lastElement.focus()
          e.preventDefault()
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          firstElement.focus()
          e.preventDefault()
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  return (
    <div
      ref={modalRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
    >
      <h2 id="modal-title">Modal Title</h2>
      <button onClick={onClose}>Close</button>
    </div>
  )
}
```

### スキップリンク

```tsx
// app/layout.tsx
export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <a href="#main" className="skip-link">
          Skip to main content
        </a>
        <nav>{/* ナビゲーション */}</nav>
        <main id="main">{children}</main>
      </body>
    </html>
  )
}

// globals.css
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: #000;
  color: #fff;
  padding: 8px;
  z-index: 100;
}

.skip-link:focus {
  top: 0;
}
```

---

## カラーコントラスト

### WCAG AA基準

- **通常テキスト**: 最低 4.5:1
- **大きなテキスト**: 最低 3:1

### 例

```css
/* ❌ 悪い例（コントラスト不足） */
.text {
  color: #999; /* 3:1 */
  background: #fff;
}

/* ✅ 良い例 */
.text {
  color: #666; /* 5.74:1 */
  background: #fff;
}
```

### フォーカスインジケーター

```css
/* ✅ 視認性の高いフォーカススタイル */
button:focus-visible {
  outline: 3px solid #0066cc;
  outline-offset: 2px;
}

/* ❌ 悪い例 */
button:focus {
  outline: none; /* フォーカスが見えない */
}
```

---

## 実践例

### Example 1: アクセシブルなボタン

```tsx
interface ButtonProps {
  children: React.ReactNode
  onClick: () => void
  disabled?: boolean
  'aria-label'?: string
}

export function Button({ children, onClick, disabled, 'aria-label': ariaLabel }: ButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {children}
    </button>
  )
}
```

### Example 2: アクセシブルなフォーム

```tsx
export function ContactForm() {
  return (
    <form>
      <div>
        <label htmlFor="name">Name *</label>
        <input
          id="name"
          type="text"
          required
          aria-required="true"
        />
      </div>

      <div>
        <label htmlFor="email">Email *</label>
        <input
          id="email"
          type="email"
          required
          aria-required="true"
          aria-describedby="email-help"
        />
        <span id="email-help" className="text-sm text-gray-600">
          We'll never share your email.
        </span>
      </div>

      <div role="group" aria-labelledby="contact-method">
        <span id="contact-method">Preferred contact method</span>
        <label>
          <input type="radio" name="contact" value="email" />
          Email
        </label>
        <label>
          <input type="radio" name="contact" value="phone" />
          Phone
        </label>
      </div>

      <button type="submit">Submit</button>
    </form>
  )
}
```

### Example 3: アクセシブルなモーダル

```tsx
'use client'

import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'

interface ModalProps {
  isOpen: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
}

export function Modal({ isOpen, onClose, title, children }: ModalProps) {
  const modalRef = useRef<HTMLDivElement>(null)
  const previousActiveElement = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (isOpen) {
      previousActiveElement.current = document.activeElement as HTMLElement
      modalRef.current?.focus()
    } else {
      previousActiveElement.current?.focus()
    }
  }, [isOpen])

  if (!isOpen) return null

  return createPortal(
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center"
      onClick={onClose}
    >
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        tabIndex={-1}
        className="bg-white p-6 rounded-lg max-w-md w-full"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="modal-title" className="text-2xl font-bold mb-4">
          {title}
        </h2>
        {children}
        <button
          onClick={onClose}
          aria-label="Close modal"
          className="mt-4 px-4 py-2 bg-gray-200 rounded"
        >
          Close
        </button>
      </div>
    </div>,
    document.body
  )
}
```

---

## テストツール

### 自動テスト

```bash
pnpm add -D @axe-core/react
```

```tsx
// app/layout.tsx（開発環境のみ）
if (process.env.NODE_ENV !== 'production') {
  const axe = require('@axe-core/react')
  const React = require('react')
  const ReactDOM = require('react-dom')
  axe(React, ReactDOM, 1000)
}
```

### 手動テスト

1. **キーボード操作**: Tabキーで全要素に到達できるか
2. **スクリーンリーダー**: VoiceOver (Mac), NVDA (Windows)
3. **カラーコントラスト**: Chrome DevTools → Lighthouse
4. **Lighthouse**: Accessibility スコア90+

---

## Agent連携

### 📖 Agentへの指示例

**アクセシビリティ監査**
```
このページのアクセシビリティを確認してください。
WCAG AA基準に準拠しているか確認してください。
```

**ARIA属性追加**
```
このコンポーネントに適切なARIA属性を追加してください。
スクリーンリーダーで読み上げられるようにしてください。
```

---

## まとめ

### アクセシビリティチェックリスト

- [ ] セマンティックHTMLを使用
- [ ] 適切なARIA属性を追加
- [ ] キーボード操作可能
- [ ] カラーコントラスト4.5:1以上
- [ ] Lighthouse Accessibilityスコア90+

---

_Last updated: 2025-12-26_
