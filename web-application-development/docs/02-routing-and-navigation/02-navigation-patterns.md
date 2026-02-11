# ナビゲーション設計

> ナビゲーションはユーザーがアプリ内を移動する道標。ヘッダー、サイドバー、ブレッドクラム、タブ、コマンドパレットまで、直感的で効率的なナビゲーションUIの設計パターンを習得する。

## この章で学ぶこと

- [ ] ナビゲーション構造の設計原則を理解する
- [ ] 主要なナビゲーションパターンの実装を把握する
- [ ] レスポンシブナビゲーションの設計を学ぶ

---

## 1. ナビゲーション構造

```
階層型ナビゲーション:

  第1階層: グローバルナビゲーション
  → ヘッダー/サイドバーに常時表示
  → Dashboard, Users, Orders, Settings

  第2階層: セクションナビゲーション
  → タブ、サブメニュー
  → Users: List, Create, Import

  第3階層: コンテキストナビゲーション
  → ブレッドクラム、ページ内リンク
  → Dashboard > Users > Taro Yamada

ナビゲーションパターン:
  ① トップナビゲーション: Webサイト、ランディングページ
  ② サイドバーナビゲーション: 管理画面、ダッシュボード
  ③ ボトムナビゲーション: モバイルアプリ
  ④ コマンドパレット: パワーユーザー向け
  ⑤ ブレッドクラム: 階層の可視化
```

---

## 2. サイドバーナビゲーション

```typescript
// サイドバーの実装
'use client';
import { usePathname } from 'next/navigation';
import Link from 'next/link';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Users', href: '/users', icon: UsersIcon },
  { name: 'Orders', href: '/orders', icon: ShoppingCartIcon },
  {
    name: 'Settings',
    href: '/settings',
    icon: CogIcon,
    children: [
      { name: 'Profile', href: '/settings/profile' },
      { name: 'Billing', href: '/settings/billing' },
      { name: 'Team', href: '/settings/team' },
    ],
  },
];

function Sidebar() {
  const pathname = usePathname();

  return (
    <nav className="w-64 bg-gray-900 text-white h-screen p-4">
      <div className="text-xl font-bold mb-8">MyApp</div>
      <ul className="space-y-1">
        {navigation.map((item) => {
          const isActive = pathname.startsWith(item.href);
          return (
            <li key={item.name}>
              <Link
                href={item.href}
                className={`flex items-center gap-3 px-3 py-2 rounded-md
                  ${isActive ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-white'}`}
              >
                <item.icon className="w-5 h-5" />
                {item.name}
              </Link>
              {item.children && isActive && (
                <ul className="ml-8 mt-1 space-y-1">
                  {item.children.map((child) => (
                    <li key={child.name}>
                      <Link
                        href={child.href}
                        className={`block px-3 py-1 rounded-md text-sm
                          ${pathname === child.href ? 'text-white' : 'text-gray-400'}`}
                      >
                        {child.name}
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
```

---

## 3. ブレッドクラム

```typescript
// 動的ブレッドクラム
'use client';
import { usePathname } from 'next/navigation';

const breadcrumbMap: Record<string, string> = {
  '/dashboard': 'Dashboard',
  '/users': 'Users',
  '/orders': 'Orders',
  '/settings': 'Settings',
  '/settings/profile': 'Profile',
};

function Breadcrumbs() {
  const pathname = usePathname();
  const segments = pathname.split('/').filter(Boolean);

  const crumbs = segments.map((_, index) => {
    const href = '/' + segments.slice(0, index + 1).join('/');
    const label = breadcrumbMap[href] ?? segments[index];
    return { href, label };
  });

  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-2 text-sm text-gray-500">
      <Link href="/" className="hover:text-gray-700">Home</Link>
      {crumbs.map((crumb, i) => (
        <span key={crumb.href} className="flex items-center gap-2">
          <span>/</span>
          {i === crumbs.length - 1 ? (
            <span className="text-gray-900 font-medium">{crumb.label}</span>
          ) : (
            <Link href={crumb.href} className="hover:text-gray-700">{crumb.label}</Link>
          )}
        </span>
      ))}
    </nav>
  );
}
```

---

## 4. コマンドパレット

```typescript
// Cmd+K コマンドパレット（cmdk ライブラリ）
'use client';
import { Command } from 'cmdk';
import { useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';

function CommandPalette() {
  const [open, setOpen] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen(o => !o);
      }
    };
    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, []);

  return (
    <Command.Dialog open={open} onOpenChange={setOpen} label="Command Menu">
      <Command.Input placeholder="Search..." />
      <Command.List>
        <Command.Empty>No results found.</Command.Empty>

        <Command.Group heading="Pages">
          <Command.Item onSelect={() => { router.push('/dashboard'); setOpen(false); }}>
            Dashboard
          </Command.Item>
          <Command.Item onSelect={() => { router.push('/users'); setOpen(false); }}>
            Users
          </Command.Item>
          <Command.Item onSelect={() => { router.push('/settings'); setOpen(false); }}>
            Settings
          </Command.Item>
        </Command.Group>

        <Command.Group heading="Actions">
          <Command.Item onSelect={() => { router.push('/users/new'); setOpen(false); }}>
            Create User
          </Command.Item>
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}
```

---

## 5. レスポンシブナビゲーション

```
レスポンシブ戦略:

  デスクトップ（≥ 1024px）:
  → サイドバー（常時表示）

  タブレット（768px - 1023px）:
  → サイドバー（折りたたみ可能、アイコンのみ）

  モバイル（< 768px）:
  → ハンバーガーメニュー + ドロワー
  → または ボトムナビゲーション

実装:
  → Tailwind の responsive prefix を活用
  → lg:block, md:hidden 等
  → Sheet コンポーネント（shadcn/ui）でモバイルドロワー

アクセシビリティ:
  ✓ nav 要素に aria-label
  ✓ 現在のページに aria-current="page"
  ✓ キーボードナビゲーション対応
  ✓ フォーカストラップ（モバイルメニュー）
  ✓ Escape キーでメニューを閉じる
```

---

## まとめ

| パターン | 用途 |
|---------|------|
| サイドバー | 管理画面、ダッシュボード |
| トップナビ | マーケティングサイト |
| ブレッドクラム | 階層の可視化 |
| コマンドパレット | パワーユーザー向け（Cmd+K） |
| ボトムナビ | モバイルアプリ |

---

## 次に読むべきガイド
→ [[03-auth-and-guards.md]] — 認証ガード

---

## 参考文献
1. shadcn/ui. "Sidebar." ui.shadcn.com, 2024.
2. cmdk. "Command Menu." cmdk.paco.me, 2024.
3. Nielsen Norman Group. "Navigation Design." nngroup.com, 2024.
