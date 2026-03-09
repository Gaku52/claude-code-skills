# ナビゲーション設計

> ナビゲーションはユーザーがアプリ内を移動する道標。ヘッダー、サイドバー、ブレッドクラム、タブ、コマンドパレットまで、直感的で効率的なナビゲーションUIの設計パターンを習得する。

## この章で学ぶこと

- [ ] ナビゲーション構造の設計原則を理解する
- [ ] 主要なナビゲーションパターンの実装を把握する
- [ ] レスポンシブナビゲーションの設計を学ぶ
- [ ] アクセシビリティに配慮したナビゲーション実装を習得する
- [ ] コマンドパレットによるキーボード駆動ナビゲーションを構築する
- [ ] ナビゲーション状態管理のベストプラクティスを理解する
- [ ] パフォーマンスを考慮したナビゲーション最適化を実践する

---

## 1. ナビゲーション構造の設計原則

### 1.1 ナビゲーションの役割と重要性

ナビゲーションはWebアプリケーションにおいて最も重要なUI要素の一つである。ユーザーが目的のコンテンツや機能に到達するための道筋を提供し、アプリケーション全体の構造を視覚的に表現する。優れたナビゲーション設計は、ユーザーの生産性を大幅に向上させ、アプリケーションの学習コストを低減させる。

ナビゲーション設計において考慮すべき基本原則は以下の通りである。

```
ナビゲーション設計の基本原則:

  1. 発見可能性（Discoverability）:
     → ユーザーが利用可能な機能やコンテンツを容易に見つけられること
     → 主要なナビゲーション項目は常に視認可能であること
     → 隠れたナビゲーション（ハンバーガーメニュー等）は補助的に使用

  2. 一貫性（Consistency）:
     → アプリ全体でナビゲーションの位置・スタイル・動作を統一
     → ユーザーの学習コストを最小化
     → プラットフォームの慣習に従った配置

  3. 文脈の保持（Context Preservation）:
     → ユーザーが「今どこにいるか」を常に明示
     → アクティブ状態のハイライト、ブレッドクラムの表示
     → 戻る操作の容易さ

  4. 効率性（Efficiency）:
     → 頻繁に使用する機能へのショートカット
     → キーボードナビゲーションの対応
     → コマンドパレットによる高速アクセス

  5. スケーラビリティ（Scalability）:
     → 機能追加時にナビゲーション構造が破綻しないこと
     → 階層の深さを3段階以内に抑える
     → 項目数の増加に対応できるグループ化の仕組み
```

### 1.2 階層型ナビゲーション構造

Webアプリケーションのナビゲーションは、一般的に3つの階層に分けて設計される。各階層がそれぞれの役割を持ち、ユーザーを適切な粒度で案内する。

```
階層型ナビゲーション:

  第1階層: グローバルナビゲーション
  → ヘッダー/サイドバーに常時表示
  → アプリケーション全体のメインセクション
  → Dashboard, Users, Orders, Settings, Reports
  → 通常5〜8項目に制限する（ミラーの法則：7±2）

  第2階層: セクションナビゲーション
  → タブ、サブメニュー、セカンダリサイドバー
  → メインセクション内のサブカテゴリ
  → Users: List, Create, Import, Export, Analytics
  → 親セクションとの関連性を視覚的に表現

  第3階層: コンテキストナビゲーション
  → ブレッドクラム、ページ内リンク、関連コンテンツ
  → 特定のページ内での移動手段
  → Dashboard > Users > Taro Yamada > Edit Profile
  → ユーザーの現在位置と移動履歴の可視化
```

### 1.3 ナビゲーションパターンの選択基準

```
ナビゲーションパターンの選択マトリクス:

  パターン           適用場面                     メリット                  デメリット
  ─────────────────────────────────────────────────────────────────────────────────
  ① トップナビ       Webサイト                   馴染みがある              項目数に制限
                     ランディングページ           水平スペース活用          サブメニュー難
                     コーポレートサイト            SEO親和性高い            モバイル対応要

  ② サイドバー       管理画面                     多数の項目対応            画面幅消費
                     ダッシュボード               階層表現が容易            モバイル非表示
                     SaaSアプリケーション          折りたたみ対応            実装が複雑

  ③ ボトムナビ       モバイルアプリ               親指操作に最適            項目数制限(5個)
                     PWA                          直感的                    デスクトップ不適
                     モバイルファースト            プラットフォーム慣習      階層表現困難

  ④ コマンドパレット パワーユーザー向け           高速アクセス              発見性が低い
                     開発ツール                   検索可能                  学習コスト高い
                     複雑なアプリ                  拡張性高い               補助的使用に限定

  ⑤ ブレッドクラム   ECサイト                     階層の可視化              スペース消費
                     コンテンツサイト              戻り操作容易              複雑な階層で冗長
                     ファイル管理系                SEO効果                  単独では不十分

  ⑥ タブナビ         設定画面                     直感的な切替              項目数制限
                     詳細ページ                   関連コンテンツ整理        レスポンシブ難
                     フォーム分割                  状態が明確               ネスト非推奨

  ⑦ メガメニュー     ECサイト                     大量カテゴリ表示          モバイル不適
                     ニュースサイト                視覚的に整理可能          実装複雑
                     ポータルサイト                プレビュー表示            パフォーマンス
```

### 1.4 情報アーキテクチャとナビゲーション

ナビゲーション設計の前提として、情報アーキテクチャ（IA）の設計が不可欠である。IAはコンテンツの構造化・ラベリング・組織化を扱い、ナビゲーションの基盤となる。

```typescript
// 情報アーキテクチャに基づくナビゲーション構造の定義
interface NavigationItem {
  /** 表示ラベル */
  label: string;
  /** 遷移先パス */
  href: string;
  /** アイコンコンポーネント */
  icon?: React.ComponentType<{ className?: string }>;
  /** 子ナビゲーション項目 */
  children?: NavigationItem[];
  /** バッジ表示（通知数等） */
  badge?: number | string;
  /** アクセス権限 */
  permission?: string;
  /** セクション分類 */
  section?: 'main' | 'secondary' | 'footer';
  /** ショートカットキー */
  shortcut?: string;
  /** 外部リンクかどうか */
  external?: boolean;
  /** 表示条件 */
  visible?: boolean | (() => boolean);
}

// ナビゲーション構造の型定義
interface NavigationConfig {
  /** メインナビゲーション項目 */
  main: NavigationItem[];
  /** セカンダリナビゲーション項目 */
  secondary?: NavigationItem[];
  /** フッターナビゲーション項目 */
  footer?: NavigationItem[];
  /** ユーザーメニュー項目 */
  userMenu?: NavigationItem[];
}

// 実際のナビゲーション設定例
const navigationConfig: NavigationConfig = {
  main: [
    {
      label: 'Dashboard',
      href: '/dashboard',
      icon: HomeIcon,
      shortcut: 'g d',
      section: 'main',
    },
    {
      label: 'Projects',
      href: '/projects',
      icon: FolderIcon,
      shortcut: 'g p',
      section: 'main',
      children: [
        { label: 'All Projects', href: '/projects' },
        { label: 'Starred', href: '/projects/starred' },
        { label: 'Archived', href: '/projects/archived' },
      ],
    },
    {
      label: 'Team',
      href: '/team',
      icon: UsersIcon,
      shortcut: 'g t',
      section: 'main',
      badge: 3,
      children: [
        { label: 'Members', href: '/team/members' },
        { label: 'Roles', href: '/team/roles' },
        { label: 'Invitations', href: '/team/invitations', badge: 3 },
      ],
    },
    {
      label: 'Analytics',
      href: '/analytics',
      icon: ChartBarIcon,
      shortcut: 'g a',
      section: 'main',
      permission: 'analytics:read',
    },
  ],
  secondary: [
    {
      label: 'Documentation',
      href: 'https://docs.example.com',
      icon: BookOpenIcon,
      external: true,
    },
    {
      label: 'Support',
      href: '/support',
      icon: LifebuoyIcon,
    },
  ],
  footer: [
    {
      label: 'Settings',
      href: '/settings',
      icon: CogIcon,
      shortcut: 'g s',
      children: [
        { label: 'General', href: '/settings/general' },
        { label: 'Security', href: '/settings/security' },
        { label: 'Billing', href: '/settings/billing' },
        { label: 'Integrations', href: '/settings/integrations' },
        { label: 'API Keys', href: '/settings/api-keys' },
      ],
    },
  ],
};
```

### 1.5 ナビゲーション状態管理の設計

ナビゲーションの状態はアプリケーション全体で共有される必要がある。サイドバーの開閉状態、アクティブな項目、展開されたサブメニューなどの状態を効率的に管理する方法を理解する。

```typescript
// ナビゲーション状態の型定義
interface NavigationState {
  /** サイドバーの開閉状態 */
  sidebarOpen: boolean;
  /** サイドバーの折りたたみ状態（アイコンのみ表示） */
  sidebarCollapsed: boolean;
  /** 展開されているサブメニューのパス */
  expandedItems: Set<string>;
  /** モバイルメニューの開閉状態 */
  mobileMenuOpen: boolean;
  /** コマンドパレットの開閉状態 */
  commandPaletteOpen: boolean;
  /** 直近の訪問履歴 */
  recentPages: string[];
}

// React Context を使ったナビゲーション状態管理
import { createContext, useContext, useReducer, useCallback, ReactNode } from 'react';

type NavigationAction =
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'TOGGLE_SIDEBAR_COLLAPSE' }
  | { type: 'TOGGLE_EXPAND'; path: string }
  | { type: 'TOGGLE_MOBILE_MENU' }
  | { type: 'CLOSE_MOBILE_MENU' }
  | { type: 'TOGGLE_COMMAND_PALETTE' }
  | { type: 'ADD_RECENT_PAGE'; path: string }
  | { type: 'SET_SIDEBAR_OPEN'; open: boolean };

const initialState: NavigationState = {
  sidebarOpen: true,
  sidebarCollapsed: false,
  expandedItems: new Set(),
  mobileMenuOpen: false,
  commandPaletteOpen: false,
  recentPages: [],
};

function navigationReducer(
  state: NavigationState,
  action: NavigationAction
): NavigationState {
  switch (action.type) {
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarOpen: !state.sidebarOpen };

    case 'TOGGLE_SIDEBAR_COLLAPSE':
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed };

    case 'TOGGLE_EXPAND': {
      const newExpanded = new Set(state.expandedItems);
      if (newExpanded.has(action.path)) {
        newExpanded.delete(action.path);
      } else {
        newExpanded.add(action.path);
      }
      return { ...state, expandedItems: newExpanded };
    }

    case 'TOGGLE_MOBILE_MENU':
      return { ...state, mobileMenuOpen: !state.mobileMenuOpen };

    case 'CLOSE_MOBILE_MENU':
      return { ...state, mobileMenuOpen: false };

    case 'TOGGLE_COMMAND_PALETTE':
      return { ...state, commandPaletteOpen: !state.commandPaletteOpen };

    case 'ADD_RECENT_PAGE': {
      const recent = [
        action.path,
        ...state.recentPages.filter(p => p !== action.path),
      ].slice(0, 10);
      return { ...state, recentPages: recent };
    }

    case 'SET_SIDEBAR_OPEN':
      return { ...state, sidebarOpen: action.open };

    default:
      return state;
  }
}

const NavigationContext = createContext<{
  state: NavigationState;
  dispatch: React.Dispatch<NavigationAction>;
} | null>(null);

export function NavigationProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(navigationReducer, initialState);

  return (
    <NavigationContext.Provider value={{ state, dispatch }}>
      {children}
    </NavigationContext.Provider>
  );
}

export function useNavigation() {
  const context = useContext(NavigationContext);
  if (!context) {
    throw new Error('useNavigation must be used within NavigationProvider');
  }
  return context;
}

// カスタムフック: ナビゲーションアクションを提供
export function useNavigationActions() {
  const { dispatch } = useNavigation();

  return {
    toggleSidebar: useCallback(() => dispatch({ type: 'TOGGLE_SIDEBAR' }), [dispatch]),
    toggleSidebarCollapse: useCallback(
      () => dispatch({ type: 'TOGGLE_SIDEBAR_COLLAPSE' }),
      [dispatch]
    ),
    toggleExpand: useCallback(
      (path: string) => dispatch({ type: 'TOGGLE_EXPAND', path }),
      [dispatch]
    ),
    toggleMobileMenu: useCallback(
      () => dispatch({ type: 'TOGGLE_MOBILE_MENU' }),
      [dispatch]
    ),
    closeMobileMenu: useCallback(
      () => dispatch({ type: 'CLOSE_MOBILE_MENU' }),
      [dispatch]
    ),
    toggleCommandPalette: useCallback(
      () => dispatch({ type: 'TOGGLE_COMMAND_PALETTE' }),
      [dispatch]
    ),
    addRecentPage: useCallback(
      (path: string) => dispatch({ type: 'ADD_RECENT_PAGE', path }),
      [dispatch]
    ),
  };
}
```

---

## 2. サイドバーナビゲーション

### 2.1 基本的なサイドバー実装

サイドバーナビゲーションは管理画面やSaaSアプリケーションで最も広く採用されているパターンである。垂直方向に項目を配置することで、多数のナビゲーション項目を効率的に表示できる。

```typescript
// 本格的なサイドバーの実装
'use client';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { useState, useCallback, useEffect } from 'react';
import { cn } from '@/lib/utils';
import {
  HomeIcon,
  UsersIcon,
  ShoppingCartIcon,
  CogIcon,
  ChartBarIcon,
  FolderIcon,
  BellIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  MenuIcon,
  XIcon,
} from 'lucide-react';

// ナビゲーション項目の型定義
interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: number;
  children?: Omit<NavItem, 'icon' | 'children'>[];
}

const navigation: NavItem[] = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  {
    name: 'Users',
    href: '/users',
    icon: UsersIcon,
    badge: 12,
    children: [
      { name: 'All Users', href: '/users' },
      { name: 'Create User', href: '/users/new' },
      { name: 'Import', href: '/users/import' },
      { name: 'User Groups', href: '/users/groups' },
    ],
  },
  {
    name: 'Orders',
    href: '/orders',
    icon: ShoppingCartIcon,
    badge: 5,
    children: [
      { name: 'All Orders', href: '/orders' },
      { name: 'Pending', href: '/orders/pending' },
      { name: 'Completed', href: '/orders/completed' },
      { name: 'Refunds', href: '/orders/refunds' },
    ],
  },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
  { name: 'Projects', href: '/projects', icon: FolderIcon },
  { name: 'Notifications', href: '/notifications', icon: BellIcon, badge: 3 },
  {
    name: 'Settings',
    href: '/settings',
    icon: CogIcon,
    children: [
      { name: 'General', href: '/settings/general' },
      { name: 'Profile', href: '/settings/profile' },
      { name: 'Billing', href: '/settings/billing' },
      { name: 'Team', href: '/settings/team' },
      { name: 'API Keys', href: '/settings/api-keys' },
    ],
  },
];

function Sidebar() {
  const pathname = usePathname();
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [collapsed, setCollapsed] = useState(false);

  // パス変更時に該当するメニューを自動展開
  useEffect(() => {
    const parentItem = navigation.find(
      (item) =>
        item.children?.some((child) => pathname.startsWith(child.href))
    );
    if (parentItem) {
      setExpandedItems((prev) => new Set([...prev, parentItem.href]));
    }
  }, [pathname]);

  const toggleExpand = useCallback((href: string) => {
    setExpandedItems((prev) => {
      const next = new Set(prev);
      if (next.has(href)) {
        next.delete(href);
      } else {
        next.add(href);
      }
      return next;
    });
  }, []);

  const isActive = (href: string) => pathname === href;
  const isParentActive = (item: NavItem) =>
    pathname.startsWith(item.href) ||
    item.children?.some((child) => pathname.startsWith(child.href));

  return (
    <aside
      className={cn(
        'flex flex-col bg-gray-900 text-white h-screen transition-all duration-300',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* ロゴ & 折りたたみボタン */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        {!collapsed && (
          <span className="text-xl font-bold tracking-tight">MyApp</span>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 rounded-md hover:bg-gray-800 transition-colors"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <MenuIcon className="w-5 h-5" />
        </button>
      </div>

      {/* ナビゲーション本体 */}
      <nav className="flex-1 overflow-y-auto p-3 space-y-1">
        {navigation.map((item) => {
          const active = isParentActive(item);
          const expanded = expandedItems.has(item.href);
          const hasChildren = item.children && item.children.length > 0;

          return (
            <div key={item.name}>
              {/* メインナビゲーション項目 */}
              <div className="flex items-center">
                <Link
                  href={hasChildren ? '#' : item.href}
                  onClick={(e) => {
                    if (hasChildren) {
                      e.preventDefault();
                      toggleExpand(item.href);
                    }
                  }}
                  className={cn(
                    'flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200',
                    active
                      ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800'
                  )}
                  title={collapsed ? item.name : undefined}
                >
                  <item.icon className="w-5 h-5 flex-shrink-0" />
                  {!collapsed && (
                    <>
                      <span className="flex-1">{item.name}</span>
                      {item.badge && (
                        <span className="px-2 py-0.5 text-xs font-medium bg-red-500 text-white rounded-full">
                          {item.badge}
                        </span>
                      )}
                      {hasChildren && (
                        <ChevronDownIcon
                          className={cn(
                            'w-4 h-4 transition-transform duration-200',
                            expanded ? 'rotate-180' : ''
                          )}
                        />
                      )}
                    </>
                  )}
                </Link>
              </div>

              {/* サブナビゲーション */}
              {hasChildren && expanded && !collapsed && (
                <div className="mt-1 ml-4 pl-4 border-l border-gray-700 space-y-1">
                  {item.children!.map((child) => (
                    <Link
                      key={child.href}
                      href={child.href}
                      className={cn(
                        'block px-3 py-2 rounded-md text-sm transition-colors duration-200',
                        isActive(child.href)
                          ? 'text-blue-400 bg-gray-800 font-medium'
                          : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                      )}
                    >
                      {child.name}
                      {child.badge && (
                        <span className="ml-2 px-1.5 py-0.5 text-xs bg-red-500 text-white rounded-full">
                          {child.badge}
                        </span>
                      )}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </nav>

      {/* フッター: ユーザー情報 */}
      {!collapsed && (
        <div className="p-4 border-t border-gray-800">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-sm font-medium">
              TY
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">Taro Yamada</p>
              <p className="text-xs text-gray-400 truncate">taro@example.com</p>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
```

### 2.2 折りたたみ可能なサイドバー（アニメーション対応）

サイドバーの折りたたみは、画面領域を有効活用するために重要な機能である。CSS transition と Framer Motion を使ったスムーズなアニメーション実装を示す。

```typescript
// Framer Motion を使ったアニメーション付きサイドバー
'use client';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigation, useNavigationActions } from '@/contexts/NavigationContext';

function AnimatedSidebar() {
  const { state } = useNavigation();
  const { toggleSidebarCollapse, toggleExpand } = useNavigationActions();
  const { sidebarCollapsed, expandedItems } = state;

  const sidebarVariants = {
    expanded: { width: 256 },
    collapsed: { width: 64 },
  };

  const labelVariants = {
    expanded: { opacity: 1, display: 'block' },
    collapsed: { opacity: 0, display: 'none' },
  };

  return (
    <motion.aside
      initial={false}
      animate={sidebarCollapsed ? 'collapsed' : 'expanded'}
      variants={sidebarVariants}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="flex flex-col bg-gray-900 text-white h-screen overflow-hidden"
    >
      {/* ロゴエリア */}
      <div className="flex items-center h-16 px-4 border-b border-gray-800">
        <motion.span
          variants={labelVariants}
          className="text-xl font-bold whitespace-nowrap"
        >
          MyApp
        </motion.span>
      </div>

      {/* ナビゲーション */}
      <nav className="flex-1 overflow-y-auto p-2">
        {navigation.map((item) => (
          <NavItemComponent
            key={item.href}
            item={item}
            collapsed={sidebarCollapsed}
            expanded={expandedItems.has(item.href)}
            onToggle={() => toggleExpand(item.href)}
          />
        ))}
      </nav>

      {/* 折りたたみトグル */}
      <div className="p-2 border-t border-gray-800">
        <button
          onClick={toggleSidebarCollapse}
          className="w-full flex items-center justify-center p-2 rounded-lg hover:bg-gray-800 transition-colors"
        >
          <motion.div
            animate={{ rotate: sidebarCollapsed ? 180 : 0 }}
            transition={{ duration: 0.3 }}
          >
            <ChevronLeftIcon className="w-5 h-5" />
          </motion.div>
        </button>
      </div>
    </motion.aside>
  );
}

// ナビゲーション項目コンポーネント（アニメーション対応）
function NavItemComponent({
  item,
  collapsed,
  expanded,
  onToggle,
}: {
  item: NavItem;
  collapsed: boolean;
  expanded: boolean;
  onToggle: () => void;
}) {
  const pathname = usePathname();
  const isActive = pathname === item.href || pathname.startsWith(item.href + '/');

  return (
    <div className="mb-1">
      <Link
        href={item.children ? '#' : item.href}
        onClick={(e) => {
          if (item.children) {
            e.preventDefault();
            onToggle();
          }
        }}
        className={cn(
          'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200',
          isActive
            ? 'bg-blue-600 text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-800'
        )}
      >
        <item.icon className="w-5 h-5 flex-shrink-0" />
        <AnimatePresence>
          {!collapsed && (
            <motion.span
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              className="flex-1 whitespace-nowrap overflow-hidden"
            >
              {item.name}
            </motion.span>
          )}
        </AnimatePresence>
      </Link>

      {/* サブメニュー（アニメーション付き展開） */}
      <AnimatePresence>
        {item.children && expanded && !collapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="mt-1 ml-6 pl-3 border-l border-gray-700 space-y-0.5">
              {item.children.map((child) => (
                <Link
                  key={child.href}
                  href={child.href}
                  className={cn(
                    'block px-3 py-2 rounded-md text-sm transition-colors',
                    pathname === child.href
                      ? 'text-blue-400 bg-gray-800'
                      : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
                  )}
                >
                  {child.name}
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
```

### 2.3 ツールチップ付き折りたたみサイドバー

サイドバーが折りたたまれた状態では、各項目にツールチップを表示してラベルを補完する。

```typescript
// Radix UI のツールチップを使ったナビゲーション項目
import * as Tooltip from '@radix-ui/react-tooltip';

function CollapsedNavItem({ item, isActive }: { item: NavItem; isActive: boolean }) {
  return (
    <Tooltip.Provider delayDuration={0}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <Link
            href={item.href}
            className={cn(
              'flex items-center justify-center w-10 h-10 rounded-lg transition-colors mx-auto',
              isActive
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            )}
          >
            <item.icon className="w-5 h-5" />
            {item.badge && (
              <span className="absolute -top-1 -right-1 w-4 h-4 text-[10px] flex items-center justify-center bg-red-500 text-white rounded-full">
                {item.badge}
              </span>
            )}
          </Link>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            side="right"
            sideOffset={8}
            className="px-3 py-1.5 bg-gray-800 text-white text-sm rounded-md shadow-lg z-50"
          >
            {item.name}
            <Tooltip.Arrow className="fill-gray-800" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}
```

### 2.4 サイドバーのベストプラクティスとアンチパターン

```
ベストプラクティス:

  ✅ アクティブ項目を視覚的に明確にハイライトする
     → 背景色変更 + 左ボーダー or 左マーカー
     → aria-current="page" を設定

  ✅ 現在のパスに基づいてサブメニューを自動展開する
     → useEffect で pathname 変更を監視
     → 初期表示時に該当するサブメニューを展開

  ✅ キーボードナビゲーション対応
     → Tab / Shift+Tab でフォーカス移動
     → Enter / Space でリンク遷移 / サブメニュー展開
     → 矢印キーでサブメニュー内移動

  ✅ 折りたたみ状態の永続化
     → localStorage にサイドバー状態を保存
     → ページリロード後も状態を復元

  ✅ 適切なスクロール処理
     → ナビゲーション項目が多い場合のスクロール対応
     → overflow-y-auto + スクロールバーのカスタマイズ

アンチパターン:

  ❌ サイドバーに10個以上のトップレベル項目を配置
     → グルーピングやセクション分けで整理する

  ❌ 3段階以上のネストされたサブメニュー
     → 深い階層は別ページやモーダルで処理する

  ❌ アイコンなしのテキストのみナビゲーション
     → アイコンは視覚的な手がかりとして重要

  ❌ ページ遷移時にサイドバー全体が再レンダリング
     → React.memo や useMemo で最適化
     → ルートレイアウトにサイドバーを配置

  ❌ モバイルでサイドバーを常時表示
     → ハンバーガーメニュー + ドロワーに切り替え
```

### 2.5 サイドバーの永続化とローカルストレージ

```typescript
// サイドバー状態の永続化カスタムフック
import { useState, useEffect, useCallback } from 'react';

interface SidebarPersistState {
  collapsed: boolean;
  expandedItems: string[];
  pinnedItems: string[];
}

const STORAGE_KEY = 'sidebar-state';

function useSidebarPersistence() {
  const [state, setState] = useState<SidebarPersistState>(() => {
    if (typeof window === 'undefined') {
      return { collapsed: false, expandedItems: [], pinnedItems: [] };
    }
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch {
      // localStorage が利用できない場合のフォールバック
    }
    return { collapsed: false, expandedItems: [], pinnedItems: [] };
  });

  // 状態変更時に自動保存
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // localStorage 書き込みエラーを無視
    }
  }, [state]);

  const toggleCollapsed = useCallback(() => {
    setState((prev) => ({ ...prev, collapsed: !prev.collapsed }));
  }, []);

  const toggleExpanded = useCallback((href: string) => {
    setState((prev) => {
      const items = prev.expandedItems.includes(href)
        ? prev.expandedItems.filter((item) => item !== href)
        : [...prev.expandedItems, href];
      return { ...prev, expandedItems: items };
    });
  }, []);

  const togglePinned = useCallback((href: string) => {
    setState((prev) => {
      const items = prev.pinnedItems.includes(href)
        ? prev.pinnedItems.filter((item) => item !== href)
        : [...prev.pinnedItems, href];
      return { ...prev, pinnedItems: items };
    });
  }, []);

  return { ...state, toggleCollapsed, toggleExpanded, togglePinned };
}
```

---

## 3. トップナビゲーション

### 3.1 基本的なトップナビゲーション

トップナビゲーションはWebサイトやランディングページで最も一般的なパターンである。ヘッダー領域に水平に配置され、サイトのブランドとメインナビゲーションを提供する。

```typescript
// レスポンシブ対応トップナビゲーション
'use client';
import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { MenuIcon, XIcon, ChevronDownIcon } from 'lucide-react';

interface TopNavItem {
  label: string;
  href: string;
  children?: { label: string; href: string; description?: string }[];
}

const topNavItems: TopNavItem[] = [
  { label: 'Home', href: '/' },
  {
    label: 'Products',
    href: '/products',
    children: [
      { label: 'All Products', href: '/products', description: '全商品を閲覧' },
      { label: 'Categories', href: '/products/categories', description: 'カテゴリ別に探す' },
      { label: 'New Arrivals', href: '/products/new', description: '新着商品' },
      { label: 'Best Sellers', href: '/products/popular', description: '人気商品' },
    ],
  },
  { label: 'Pricing', href: '/pricing' },
  {
    label: 'Resources',
    href: '/resources',
    children: [
      { label: 'Blog', href: '/blog', description: '技術記事・お知らせ' },
      { label: 'Documentation', href: '/docs', description: '開発者向けドキュメント' },
      { label: 'Community', href: '/community', description: 'コミュニティフォーラム' },
      { label: 'Support', href: '/support', description: 'サポートセンター' },
    ],
  },
  { label: 'About', href: '/about' },
];

function TopNavigation() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // ドロップダウン外クリックで閉じる
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setActiveDropdown(null);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // ページ遷移時にモバイルメニューを閉じる
  useEffect(() => {
    setMobileMenuOpen(false);
    setActiveDropdown(null);
  }, [pathname]);

  // ドロップダウンのマウスイベントハンドラ（遅延付き）
  const handleMouseEnter = (label: string) => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setActiveDropdown(label);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = setTimeout(() => {
      setActiveDropdown(null);
    }, 150);
  };

  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* ロゴ */}
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg" />
            <span className="text-xl font-bold text-gray-900">MyApp</span>
          </Link>

          {/* デスクトップナビゲーション */}
          <nav ref={dropdownRef} className="hidden md:flex items-center gap-1">
            {topNavItems.map((item) => (
              <div
                key={item.label}
                className="relative"
                onMouseEnter={() => item.children && handleMouseEnter(item.label)}
                onMouseLeave={handleMouseLeave}
              >
                <Link
                  href={item.href}
                  className={cn(
                    'flex items-center gap-1 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                    pathname.startsWith(item.href) && item.href !== '/'
                      ? 'text-blue-600 bg-blue-50'
                      : pathname === item.href
                      ? 'text-blue-600 bg-blue-50'
                      : 'text-gray-700 hover:text-gray-900 hover:bg-gray-100'
                  )}
                >
                  {item.label}
                  {item.children && <ChevronDownIcon className="w-4 h-4" />}
                </Link>

                {/* ドロップダウンメニュー */}
                {item.children && activeDropdown === item.label && (
                  <div className="absolute top-full left-0 mt-1 w-64 bg-white rounded-lg shadow-lg border border-gray-200 py-2 z-50">
                    {item.children.map((child) => (
                      <Link
                        key={child.href}
                        href={child.href}
                        className="block px-4 py-2.5 hover:bg-gray-50 transition-colors"
                      >
                        <div className="text-sm font-medium text-gray-900">
                          {child.label}
                        </div>
                        {child.description && (
                          <div className="text-xs text-gray-500 mt-0.5">
                            {child.description}
                          </div>
                        )}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </nav>

          {/* CTA ボタン */}
          <div className="hidden md:flex items-center gap-3">
            <Link
              href="/login"
              className="text-sm font-medium text-gray-700 hover:text-gray-900"
            >
              Log in
            </Link>
            <Link
              href="/signup"
              className="text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors"
            >
              Sign up
            </Link>
          </div>

          {/* モバイルメニューボタン */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-md hover:bg-gray-100"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? (
              <XIcon className="w-6 h-6" />
            ) : (
              <MenuIcon className="w-6 h-6" />
            )}
          </button>
        </div>
      </div>

      {/* モバイルメニュー */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-gray-200 bg-white">
          <nav className="px-4 py-3 space-y-1">
            {topNavItems.map((item) => (
              <div key={item.label}>
                <Link
                  href={item.href}
                  className={cn(
                    'block px-3 py-2 rounded-md text-base font-medium',
                    pathname.startsWith(item.href)
                      ? 'text-blue-600 bg-blue-50'
                      : 'text-gray-700 hover:bg-gray-100'
                  )}
                >
                  {item.label}
                </Link>
                {item.children && (
                  <div className="ml-4 mt-1 space-y-1">
                    {item.children.map((child) => (
                      <Link
                        key={child.href}
                        href={child.href}
                        className="block px-3 py-1.5 text-sm text-gray-500 hover:text-gray-700"
                      >
                        {child.label}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </nav>
          <div className="px-4 py-3 border-t border-gray-200 space-y-2">
            <Link
              href="/login"
              className="block text-center px-4 py-2 text-sm font-medium text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Log in
            </Link>
            <Link
              href="/signup"
              className="block text-center px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
            >
              Sign up
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
```

### 3.2 スティッキーヘッダーとスクロール対応

```typescript
// スクロール時にスタイルが変化するスティッキーヘッダー
'use client';
import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';

function StickyHeader() {
  const [scrolled, setScrolled] = useState(false);
  const [hidden, setHidden] = useState(false);
  const [lastScrollY, setLastScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;

      // スクロール量でスタイル変更
      setScrolled(currentScrollY > 10);

      // スクロール方向で表示/非表示
      if (currentScrollY > lastScrollY && currentScrollY > 100) {
        setHidden(true); // 下スクロールで非表示
      } else {
        setHidden(false); // 上スクロールで表示
      }

      setLastScrollY(currentScrollY);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [lastScrollY]);

  return (
    <header
      className={cn(
        'fixed top-0 left-0 right-0 z-50 transition-all duration-300',
        scrolled
          ? 'bg-white/80 backdrop-blur-md shadow-sm border-b border-gray-200/50'
          : 'bg-transparent',
        hidden ? '-translate-y-full' : 'translate-y-0'
      )}
    >
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* ナビゲーション内容 */}
      </div>
    </header>
  );
}
```

---

## 4. ブレッドクラム

### 4.1 基本的なブレッドクラム実装

ブレッドクラムは、ユーザーが現在のページの位置を階層構造の中で把握するためのナビゲーション補助要素である。特にECサイトやコンテンツ管理システムで重要な役割を果たす。

```typescript
// 動的ブレッドクラム（Next.js App Router対応）
'use client';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { ChevronRightIcon, HomeIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

// ブレッドクラムのラベルマッピング
const breadcrumbLabels: Record<string, string> = {
  dashboard: 'Dashboard',
  users: 'Users',
  orders: 'Orders',
  settings: 'Settings',
  profile: 'Profile',
  billing: 'Billing',
  team: 'Team',
  new: 'New',
  edit: 'Edit',
  analytics: 'Analytics',
  projects: 'Projects',
  reports: 'Reports',
  import: 'Import',
  export: 'Export',
  general: 'General',
  security: 'Security',
  'api-keys': 'API Keys',
  integrations: 'Integrations',
};

// 動的セグメント（ID等）を検知する関数
function isDynamicSegment(segment: string): boolean {
  // UUID パターン
  if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(segment)) {
    return true;
  }
  // 数値ID
  if (/^\d+$/.test(segment)) {
    return true;
  }
  return false;
}

interface BreadcrumbItem {
  label: string;
  href: string;
  current: boolean;
}

function useBreadcrumbs(): BreadcrumbItem[] {
  const pathname = usePathname();
  const segments = pathname.split('/').filter(Boolean);

  return segments.map((segment, index) => {
    const href = '/' + segments.slice(0, index + 1).join('/');
    const isCurrent = index === segments.length - 1;

    let label: string;
    if (isDynamicSegment(segment)) {
      label = '...'; // 動的セグメントはプレースホルダー
    } else {
      label = breadcrumbLabels[segment] ?? segment.charAt(0).toUpperCase() + segment.slice(1);
    }

    return { label, href, current: isCurrent };
  });
}

function Breadcrumbs() {
  const crumbs = useBreadcrumbs();

  if (crumbs.length === 0) return null;

  return (
    <nav aria-label="Breadcrumb" className="flex items-center text-sm">
      <ol className="flex items-center gap-1.5">
        {/* ホームリンク */}
        <li>
          <Link
            href="/"
            className="text-gray-400 hover:text-gray-600 transition-colors"
            aria-label="Home"
          >
            <HomeIcon className="w-4 h-4" />
          </Link>
        </li>

        {crumbs.map((crumb) => (
          <li key={crumb.href} className="flex items-center gap-1.5">
            <ChevronRightIcon className="w-3.5 h-3.5 text-gray-300 flex-shrink-0" />
            {crumb.current ? (
              <span
                className="text-gray-900 font-medium"
                aria-current="page"
              >
                {crumb.label}
              </span>
            ) : (
              <Link
                href={crumb.href}
                className="text-gray-500 hover:text-gray-700 transition-colors"
              >
                {crumb.label}
              </Link>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}
```

### 4.2 動的エンティティ名を解決するブレッドクラム

実際のアプリケーションでは、URLに含まれるIDを実際のエンティティ名に解決する必要がある。

```typescript
// エンティティ名を動的に解決するブレッドクラム
'use client';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import useSWR from 'swr';

interface ResolvedBreadcrumb {
  label: string;
  href: string;
  current: boolean;
  loading?: boolean;
}

// エンティティ名を解決するためのフェッチャー
const fetcher = (url: string) => fetch(url).then((res) => res.json());

function useResolvedBreadcrumbs(): ResolvedBreadcrumb[] {
  const pathname = usePathname();
  const segments = pathname.split('/').filter(Boolean);

  // 動的セグメントの解決
  // 例: /users/123 → 123 をユーザー名に解決
  const resolvers: Record<string, (id: string) => string> = {
    users: '/api/users/',
    orders: '/api/orders/',
    projects: '/api/projects/',
  };

  const [resolvedLabels, setResolvedLabels] = useState<Record<string, string>>({});

  useEffect(() => {
    const resolveLabels = async () => {
      const newLabels: Record<string, string> = {};

      for (let i = 0; i < segments.length; i++) {
        const segment = segments[i];
        const prevSegment = segments[i - 1];

        if (isDynamicSegment(segment) && prevSegment && resolvers[prevSegment]) {
          try {
            const response = await fetch(`${resolvers[prevSegment]}${segment}`);
            const data = await response.json();
            newLabels[segment] = data.name || data.title || segment;
          } catch {
            newLabels[segment] = segment;
          }
        }
      }

      setResolvedLabels(newLabels);
    };

    resolveLabels();
  }, [pathname]);

  return segments.map((segment, index) => {
    const href = '/' + segments.slice(0, index + 1).join('/');
    const isCurrent = index === segments.length - 1;

    let label: string;
    if (resolvedLabels[segment]) {
      label = resolvedLabels[segment];
    } else if (isDynamicSegment(segment)) {
      label = '...';
    } else {
      label = breadcrumbLabels[segment] ?? segment;
    }

    return { label, href, current: isCurrent };
  });
}

// JSON-LD 構造化データ対応ブレッドクラム
function BreadcrumbsWithStructuredData() {
  const crumbs = useResolvedBreadcrumbs();

  // JSON-LD 構造化データ
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: [
      {
        '@type': 'ListItem',
        position: 1,
        name: 'Home',
        item: typeof window !== 'undefined' ? window.location.origin : '',
      },
      ...crumbs.map((crumb, index) => ({
        '@type': 'ListItem',
        position: index + 2,
        name: crumb.label,
        item: typeof window !== 'undefined'
          ? `${window.location.origin}${crumb.href}`
          : crumb.href,
      })),
    ],
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
      />
      <nav aria-label="Breadcrumb" className="flex items-center text-sm">
        <ol className="flex items-center gap-1.5">
          <li>
            <Link href="/" className="text-gray-400 hover:text-gray-600">
              <HomeIcon className="w-4 h-4" />
            </Link>
          </li>
          {crumbs.map((crumb) => (
            <li key={crumb.href} className="flex items-center gap-1.5">
              <ChevronRightIcon className="w-3.5 h-3.5 text-gray-300" />
              {crumb.current ? (
                <span className="text-gray-900 font-medium" aria-current="page">
                  {crumb.loading ? (
                    <span className="inline-block w-16 h-4 bg-gray-200 rounded animate-pulse" />
                  ) : (
                    crumb.label
                  )}
                </span>
              ) : (
                <Link href={crumb.href} className="text-gray-500 hover:text-gray-700">
                  {crumb.label}
                </Link>
              )}
            </li>
          ))}
        </ol>
      </nav>
    </>
  );
}
```

### 4.3 ブレッドクラムのベストプラクティス

```
ブレッドクラム設計のベストプラクティス:

  ✅ 階層構造を正確に反映する
     → URLパスと一致させる
     → 動的セグメントは実際のエンティティ名に解決する

  ✅ SEO対策として構造化データを出力する
     → JSON-LD 形式の BreadcrumbList
     → Google 検索結果にブレッドクラムが表示される

  ✅ アクセシビリティ対応
     → nav 要素に aria-label="Breadcrumb"
     → 現在のページに aria-current="page"
     → ol/li でマークアップ（セマンティック）

  ✅ モバイルでは省略表示を検討する
     → 中間の階層を「...」で省略
     → 最後の2〜3項目のみ表示
     → スクロール可能なブレッドクラム

  ❌ ブレッドクラムをメインナビゲーションの代替にしない
     → あくまで補助的なナビゲーション要素
     → サイドバーやトップナビと併用する

  ❌ クリック可能な最後の項目
     → 現在のページはテキストのみ（リンクにしない）
     → 視覚的にも非クリックであることを示す
```

---

## 5. タブナビゲーション

### 5.1 基本的なタブ実装

タブナビゲーションは、関連するコンテンツを同一ページ内で切り替える場合に使用される。設定画面やユーザー詳細ページなどで広く採用されている。

```typescript
// URLベースのタブナビゲーション（Next.js対応）
'use client';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { cn } from '@/lib/utils';
import { ReactNode } from 'react';

interface Tab {
  id: string;
  label: string;
  icon?: React.ComponentType<{ className?: string }>;
  badge?: number;
  disabled?: boolean;
}

interface TabNavigationProps {
  tabs: Tab[];
  basePath: string;
  children: ReactNode;
}

// パスベースのタブ（各タブが独立したURLを持つ）
function PathBasedTabs({ tabs, basePath, children }: TabNavigationProps) {
  const pathname = usePathname();

  const activeTab = tabs.find((tab) => {
    const tabPath = tab.id === 'index' ? basePath : `${basePath}/${tab.id}`;
    return pathname === tabPath;
  }) || tabs[0];

  return (
    <div>
      {/* タブヘッダー */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex gap-x-6" aria-label="Tabs">
          {tabs.map((tab) => {
            const tabPath = tab.id === 'index' ? basePath : `${basePath}/${tab.id}`;
            const isActive = activeTab.id === tab.id;

            return (
              <Link
                key={tab.id}
                href={tabPath}
                className={cn(
                  'group inline-flex items-center gap-2 border-b-2 px-1 py-3 text-sm font-medium transition-colors',
                  isActive
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700',
                  tab.disabled && 'opacity-50 pointer-events-none'
                )}
                aria-current={isActive ? 'page' : undefined}
              >
                {tab.icon && (
                  <tab.icon
                    className={cn(
                      'w-4 h-4',
                      isActive ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-500'
                    )}
                  />
                )}
                {tab.label}
                {tab.badge !== undefined && (
                  <span
                    className={cn(
                      'ml-1 rounded-full px-2 py-0.5 text-xs font-medium',
                      isActive
                        ? 'bg-blue-100 text-blue-600'
                        : 'bg-gray-100 text-gray-600'
                    )}
                  >
                    {tab.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* タブコンテンツ */}
      <div className="mt-4">{children}</div>
    </div>
  );
}

// クエリパラメータベースのタブ
function QueryBasedTabs({ tabs, children }: { tabs: Tab[]; children: ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const activeTabId = searchParams.get('tab') || tabs[0].id;

  const setActiveTab = (tabId: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set('tab', tabId);
    router.push(`?${params.toString()}`);
  };

  return (
    <div>
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex gap-x-6" aria-label="Tabs" role="tablist">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              role="tab"
              aria-selected={activeTabId === tab.id}
              aria-controls={`tabpanel-${tab.id}`}
              className={cn(
                'border-b-2 px-1 py-3 text-sm font-medium transition-colors',
                activeTabId === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
              )}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
      <div
        role="tabpanel"
        id={`tabpanel-${activeTabId}`}
        className="mt-4"
      >
        {children}
      </div>
    </div>
  );
}

// 使用例: ユーザー設定ページ
const settingsTabs: Tab[] = [
  { id: 'general', label: 'General', icon: CogIcon },
  { id: 'security', label: 'Security', icon: ShieldIcon },
  { id: 'notifications', label: 'Notifications', icon: BellIcon, badge: 5 },
  { id: 'billing', label: 'Billing', icon: CreditCardIcon },
  { id: 'integrations', label: 'Integrations', icon: PuzzleIcon },
  { id: 'api-keys', label: 'API Keys', icon: KeyIcon },
];

function SettingsPage() {
  return (
    <PathBasedTabs tabs={settingsTabs} basePath="/settings">
      {/* 各タブのコンテンツはルーティングで切り替え */}
    </PathBasedTabs>
  );
}
```

### 5.2 レスポンシブタブ（モバイル対応）

```typescript
// モバイルではドロップダウンに変換するレスポンシブタブ
function ResponsiveTabs({ tabs, activeTab, onChange }: {
  tabs: Tab[];
  activeTab: string;
  onChange: (tabId: string) => void;
}) {
  const active = tabs.find((t) => t.id === activeTab) || tabs[0];

  return (
    <>
      {/* モバイル: ドロップダウン */}
      <div className="sm:hidden">
        <label htmlFor="tab-select" className="sr-only">
          Select a tab
        </label>
        <select
          id="tab-select"
          value={activeTab}
          onChange={(e) => onChange(e.target.value)}
          className="block w-full rounded-md border-gray-300 py-2 pl-3 pr-10 text-base focus:border-blue-500 focus:outline-none focus:ring-blue-500"
        >
          {tabs.map((tab) => (
            <option key={tab.id} value={tab.id} disabled={tab.disabled}>
              {tab.label}
              {tab.badge ? ` (${tab.badge})` : ''}
            </option>
          ))}
        </select>
      </div>

      {/* デスクトップ: タブ */}
      <div className="hidden sm:block">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex gap-x-6" aria-label="Tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => onChange(tab.id)}
                className={cn(
                  'border-b-2 px-1 py-3 text-sm font-medium whitespace-nowrap transition-colors',
                  tab.id === activeTab
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700',
                  tab.disabled && 'opacity-50 cursor-not-allowed'
                )}
                disabled={tab.disabled}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>
    </>
  );
}
```

---

## 6. コマンドパレット

### 6.1 コマンドパレットの概要と設計思想

コマンドパレットは、VS Code、Figma、Slack、Linear、Notion などのモダンアプリケーションで広く採用されているナビゲーションパターンである。`Cmd+K`（macOS）または `Ctrl+K`（Windows/Linux）のキーボードショートカットで呼び出し、テキスト入力による検索・ナビゲーション・アクション実行を可能にする。

パワーユーザーの生産性を大幅に向上させるが、初心者には発見しにくいため、他のナビゲーションパターンの補助として使用する。

```
コマンドパレットの設計原則:

  1. 高速な起動と応答
     → キーストロークから表示まで100ms以内
     → 検索結果のフィルタリングはデバウンス付きで即座に反映
     → 仮想スクロールで大量の結果も高速に表示

  2. インクリメンタル検索
     → 1文字入力するごとに結果を絞り込む
     → ファジーマッチング対応（typo許容）
     → ラベル・説明・キーワードを横断検索

  3. カテゴリ分類
     → Pages / Actions / Settings / Users 等のグループ分け
     → 最近使った項目を優先表示
     → コンテキストに応じた提案

  4. キーボードファースト
     → 矢印キーで項目移動、Enter で実行
     → Escape で閉じる
     → マウス操作も併用可能

  5. 拡張性
     → 新しいコマンドの追加が容易
     → プラグイン的な拡張対応
     → APIからの動的コマンド読み込み
```

### 6.2 cmdk ライブラリを使った本格実装

```typescript
// 本格的なコマンドパレット実装（cmdk + Next.js）
'use client';
import { Command } from 'cmdk';
import { useRouter } from 'next/navigation';
import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  HomeIcon,
  UsersIcon,
  CogIcon,
  SearchIcon,
  FileTextIcon,
  PlusIcon,
  LogOutIcon,
  MoonIcon,
  SunIcon,
  ExternalLinkIcon,
  ClockIcon,
  StarIcon,
  HashIcon,
} from 'lucide-react';

// コマンド項目の型定義
interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon?: React.ComponentType<{ className?: string }>;
  shortcut?: string[];
  category: 'navigation' | 'action' | 'settings' | 'recent' | 'search';
  keywords?: string[];
  onSelect: () => void;
  priority?: number;
}

// 検索結果の型定義
interface SearchResult {
  id: string;
  title: string;
  type: 'page' | 'user' | 'order' | 'project';
  url: string;
  highlight?: string;
}

function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [recentPages, setRecentPages] = useState<string[]>([]);
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  // Cmd+K / Ctrl+K でトグル
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
      // Escape で閉じる
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  // 開いた時に入力フィールドにフォーカス
  useEffect(() => {
    if (open) {
      setSearch('');
      setSearchResults([]);
      // 少し遅延してフォーカス
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  // 最近のページを localStorage から読み込み
  useEffect(() => {
    const stored = localStorage.getItem('recent-pages');
    if (stored) {
      setRecentPages(JSON.parse(stored));
    }
  }, []);

  // ナビゲーション実行
  const navigate = useCallback(
    (path: string) => {
      // 最近のページに追加
      const updated = [path, ...recentPages.filter((p) => p !== path)].slice(0, 5);
      setRecentPages(updated);
      localStorage.setItem('recent-pages', JSON.stringify(updated));

      router.push(path);
      setOpen(false);
    },
    [router, recentPages]
  );

  // 検索API呼び出し（デバウンス付き）
  useEffect(() => {
    if (search.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    const timeoutId = setTimeout(async () => {
      try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(search)}`);
        const data = await response.json();
        setSearchResults(data.results || []);
      } catch {
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [search]);

  // 静的コマンドの定義
  const commands: CommandItem[] = useMemo(
    () => [
      // ナビゲーション
      {
        id: 'nav-dashboard',
        label: 'Dashboard',
        description: 'メインダッシュボードに移動',
        icon: HomeIcon,
        shortcut: ['G', 'D'],
        category: 'navigation',
        keywords: ['home', 'top', 'main', 'ホーム'],
        onSelect: () => navigate('/dashboard'),
        priority: 10,
      },
      {
        id: 'nav-users',
        label: 'Users',
        description: 'ユーザー管理ページに移動',
        icon: UsersIcon,
        shortcut: ['G', 'U'],
        category: 'navigation',
        keywords: ['members', 'people', 'ユーザー', 'メンバー'],
        onSelect: () => navigate('/users'),
        priority: 9,
      },
      {
        id: 'nav-settings',
        label: 'Settings',
        description: 'アプリケーション設定に移動',
        icon: CogIcon,
        shortcut: ['G', 'S'],
        category: 'navigation',
        keywords: ['config', 'preferences', '設定', '環境設定'],
        onSelect: () => navigate('/settings'),
        priority: 8,
      },
      // アクション
      {
        id: 'action-create-user',
        label: 'Create New User',
        description: '新しいユーザーを作成',
        icon: PlusIcon,
        category: 'action',
        keywords: ['add', 'new', 'user', '追加', '作成'],
        onSelect: () => navigate('/users/new'),
        priority: 7,
      },
      {
        id: 'action-create-project',
        label: 'Create New Project',
        description: '新しいプロジェクトを作成',
        icon: PlusIcon,
        category: 'action',
        keywords: ['add', 'new', 'project', 'プロジェクト'],
        onSelect: () => navigate('/projects/new'),
        priority: 6,
      },
      // 設定
      {
        id: 'settings-theme-toggle',
        label: 'Toggle Theme',
        description: 'ダークモード/ライトモードを切り替え',
        icon: MoonIcon,
        shortcut: ['T'],
        category: 'settings',
        keywords: ['dark', 'light', 'theme', 'テーマ', 'ダーク'],
        onSelect: () => {
          document.documentElement.classList.toggle('dark');
          setOpen(false);
        },
        priority: 5,
      },
      {
        id: 'settings-logout',
        label: 'Log Out',
        description: 'アカウントからログアウト',
        icon: LogOutIcon,
        category: 'settings',
        keywords: ['signout', 'exit', 'ログアウト'],
        onSelect: () => {
          // ログアウト処理
          navigate('/login');
        },
        priority: 1,
      },
    ],
    [navigate]
  );

  if (!open) return null;

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* オーバーレイ */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            onClick={() => setOpen(false)}
          />

          {/* コマンドパレット本体 */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
            className="fixed top-[20%] left-1/2 -translate-x-1/2 w-full max-w-xl z-50"
          >
            <Command
              className="bg-white rounded-xl shadow-2xl border border-gray-200 overflow-hidden"
              label="Command Menu"
              shouldFilter={true}
            >
              {/* 検索入力 */}
              <div className="flex items-center gap-2 px-4 border-b border-gray-200">
                <SearchIcon className="w-4 h-4 text-gray-400 flex-shrink-0" />
                <Command.Input
                  ref={inputRef}
                  value={search}
                  onValueChange={setSearch}
                  placeholder="Search pages, actions, settings..."
                  className="flex-1 py-3 text-sm bg-transparent outline-none placeholder:text-gray-400"
                />
                <kbd className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-medium text-gray-400 bg-gray-100 rounded border border-gray-200">
                  ESC
                </kbd>
              </div>

              {/* コマンドリスト */}
              <Command.List className="max-h-80 overflow-y-auto p-2">
                <Command.Empty className="py-6 text-center text-sm text-gray-500">
                  {isSearching ? 'Searching...' : 'No results found.'}
                </Command.Empty>

                {/* 最近のページ */}
                {recentPages.length > 0 && !search && (
                  <Command.Group heading="Recent">
                    {recentPages.map((path) => (
                      <Command.Item
                        key={`recent-${path}`}
                        value={`recent ${path}`}
                        onSelect={() => navigate(path)}
                        className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm cursor-pointer data-[selected=true]:bg-blue-50 data-[selected=true]:text-blue-900"
                      >
                        <ClockIcon className="w-4 h-4 text-gray-400" />
                        <span>{path}</span>
                      </Command.Item>
                    ))}
                  </Command.Group>
                )}

                {/* ナビゲーション */}
                <Command.Group heading="Pages">
                  {commands
                    .filter((cmd) => cmd.category === 'navigation')
                    .map((cmd) => (
                      <Command.Item
                        key={cmd.id}
                        value={`${cmd.label} ${cmd.keywords?.join(' ') || ''}`}
                        onSelect={cmd.onSelect}
                        className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm cursor-pointer data-[selected=true]:bg-blue-50 data-[selected=true]:text-blue-900"
                      >
                        {cmd.icon && <cmd.icon className="w-4 h-4 text-gray-400" />}
                        <div className="flex-1">
                          <div className="font-medium">{cmd.label}</div>
                          {cmd.description && (
                            <div className="text-xs text-gray-500">{cmd.description}</div>
                          )}
                        </div>
                        {cmd.shortcut && (
                          <div className="flex items-center gap-1">
                            {cmd.shortcut.map((key) => (
                              <kbd
                                key={key}
                                className="px-1.5 py-0.5 text-[10px] font-medium bg-gray-100 text-gray-500 rounded border border-gray-200"
                              >
                                {key}
                              </kbd>
                            ))}
                          </div>
                        )}
                      </Command.Item>
                    ))}
                </Command.Group>

                {/* アクション */}
                <Command.Group heading="Actions">
                  {commands
                    .filter((cmd) => cmd.category === 'action')
                    .map((cmd) => (
                      <Command.Item
                        key={cmd.id}
                        value={`${cmd.label} ${cmd.keywords?.join(' ') || ''}`}
                        onSelect={cmd.onSelect}
                        className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm cursor-pointer data-[selected=true]:bg-blue-50 data-[selected=true]:text-blue-900"
                      >
                        {cmd.icon && <cmd.icon className="w-4 h-4 text-gray-400" />}
                        <div className="flex-1">
                          <div className="font-medium">{cmd.label}</div>
                          {cmd.description && (
                            <div className="text-xs text-gray-500">{cmd.description}</div>
                          )}
                        </div>
                      </Command.Item>
                    ))}
                </Command.Group>

                {/* API検索結果 */}
                {searchResults.length > 0 && (
                  <Command.Group heading="Search Results">
                    {searchResults.map((result) => (
                      <Command.Item
                        key={result.id}
                        value={result.title}
                        onSelect={() => navigate(result.url)}
                        className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm cursor-pointer data-[selected=true]:bg-blue-50 data-[selected=true]:text-blue-900"
                      >
                        <HashIcon className="w-4 h-4 text-gray-400" />
                        <div className="flex-1">
                          <div className="font-medium">{result.title}</div>
                          <div className="text-xs text-gray-500">{result.type}</div>
                        </div>
                      </Command.Item>
                    ))}
                  </Command.Group>
                )}

                {/* 設定 */}
                <Command.Group heading="Settings">
                  {commands
                    .filter((cmd) => cmd.category === 'settings')
                    .map((cmd) => (
                      <Command.Item
                        key={cmd.id}
                        value={`${cmd.label} ${cmd.keywords?.join(' ') || ''}`}
                        onSelect={cmd.onSelect}
                        className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm cursor-pointer data-[selected=true]:bg-blue-50 data-[selected=true]:text-blue-900"
                      >
                        {cmd.icon && <cmd.icon className="w-4 h-4 text-gray-400" />}
                        <div className="flex-1">
                          <div className="font-medium">{cmd.label}</div>
                          {cmd.description && (
                            <div className="text-xs text-gray-500">{cmd.description}</div>
                          )}
                        </div>
                        {cmd.shortcut && (
                          <div className="flex items-center gap-1">
                            {cmd.shortcut.map((key) => (
                              <kbd
                                key={key}
                                className="px-1.5 py-0.5 text-[10px] font-medium bg-gray-100 text-gray-500 rounded border border-gray-200"
                              >
                                {key}
                              </kbd>
                            ))}
                          </div>
                        )}
                      </Command.Item>
                    ))}
                </Command.Group>
              </Command.List>

              {/* フッター */}
              <div className="flex items-center justify-between px-4 py-2 border-t border-gray-200 bg-gray-50 text-xs text-gray-400">
                <div className="flex items-center gap-2">
                  <span>Navigate</span>
                  <kbd className="px-1 py-0.5 bg-white rounded border">↑↓</kbd>
                  <span>Select</span>
                  <kbd className="px-1 py-0.5 bg-white rounded border">↵</kbd>
                  <span>Close</span>
                  <kbd className="px-1 py-0.5 bg-white rounded border">Esc</kbd>
                </div>
              </div>
            </Command>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
```

### 6.3 グローバルキーボードショートカットの実装

コマンドパレットと連携するグローバルキーボードショートカットシステムを実装する。

```typescript
// グローバルキーボードショートカット管理
'use client';
import { useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';

interface ShortcutDefinition {
  key: string;
  description: string;
  handler: () => void;
  category?: string;
}

// Vim風シーケンシャルショートカット（g + キー）
function useSequentialShortcuts(shortcuts: Record<string, () => void>) {
  const sequenceRef = useRef<string[]>([]);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 入力フィールドにフォーカスしている場合はスキップ
      const target = e.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
      ) {
        return;
      }

      // 修飾キーが押されている場合はスキップ
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      // タイムアウトをリセット
      if (timeoutRef.current) clearTimeout(timeoutRef.current);

      // シーケンスに追加
      sequenceRef.current.push(e.key.toLowerCase());

      // シーケンスをチェック
      const sequence = sequenceRef.current.join(' ');
      if (shortcuts[sequence]) {
        e.preventDefault();
        shortcuts[sequence]();
        sequenceRef.current = [];
        return;
      }

      // 1秒後にシーケンスをリセット
      timeoutRef.current = setTimeout(() => {
        sequenceRef.current = [];
      }, 1000);

      // シーケンスが長すぎる場合はリセット
      if (sequenceRef.current.length > 3) {
        sequenceRef.current = [];
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [shortcuts]);
}

// 使用例
function GlobalShortcuts() {
  const router = useRouter();

  useSequentialShortcuts({
    'g d': () => router.push('/dashboard'),
    'g u': () => router.push('/users'),
    'g s': () => router.push('/settings'),
    'g p': () => router.push('/projects'),
    'g a': () => router.push('/analytics'),
    'g n': () => router.push('/notifications'),
    'g h': () => router.push('/'),
  });

  return null; // レンダリングなし
}
```

### 6.4 コマンドパレットのベストプラクティス

```
コマンドパレット設計のベストプラクティス:

  ✅ ファジーマッチングを実装する
     → typo を許容する検索アルゴリズム
     → fuse.js などのライブラリを活用
     → 部分一致・前方一致の両方をサポート

  ✅ 最近使った項目を優先表示する
     → localStorage に履歴を保存
     → 検索なしの初期表示で履歴を表示
     → 使用頻度でソート

  ✅ ショートカットキーを表示する
     → 各コマンド横にキーバインドを表示
     → ユーザーが自然に学習できる

  ✅ カテゴリ分けで整理する
     → Pages / Actions / Settings のグループ
     → セパレータで視覚的に区切る
     → グループ内で優先度順にソート

  ✅ スムーズなアニメーション
     → 表示/非表示のトランジション
     → 検索結果の切り替えアニメーション
     → フォーカスリングの移動

  ❌ コマンドパレットを唯一のナビゲーション手段にしない
     → サイドバーやトップナビと併用
     → 発見性の低さを補完する

  ❌ 過度に多いコマンドを登録する
     → 100件以上は仮想スクロールで対応
     → 使用頻度の低いコマンドはフィルタリング
```

---

## 7. ボトムナビゲーション

### 7.1 モバイル向けボトムナビゲーション

ボトムナビゲーションは、モバイルアプリやPWAで最も直感的なナビゲーションパターンである。親指で操作しやすい画面下部に配置され、3〜5項目のメイン機能に素早くアクセスできる。

```typescript
// モバイル向けボトムナビゲーション
'use client';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { cn } from '@/lib/utils';
import {
  HomeIcon,
  SearchIcon,
  PlusCircleIcon,
  BellIcon,
  UserIcon,
} from 'lucide-react';
import { motion } from 'framer-motion';

interface BottomNavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: number;
  /** 中央の強調ボタンかどうか */
  primary?: boolean;
}

const bottomNavItems: BottomNavItem[] = [
  { label: 'Home', href: '/', icon: HomeIcon },
  { label: 'Search', href: '/search', icon: SearchIcon },
  { label: 'Create', href: '/create', icon: PlusCircleIcon, primary: true },
  { label: 'Notifications', href: '/notifications', icon: BellIcon, badge: 3 },
  { label: 'Profile', href: '/profile', icon: UserIcon },
];

function BottomNavigation() {
  const pathname = usePathname();

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-40 bg-white border-t border-gray-200 safe-area-bottom md:hidden"
      aria-label="Bottom navigation"
    >
      <div className="flex items-center justify-around h-16 px-2">
        {bottomNavItems.map((item) => {
          const isActive = pathname === item.href;

          // 中央の強調ボタン
          if (item.primary) {
            return (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center justify-center w-14 h-14 -mt-5 bg-blue-600 rounded-full shadow-lg shadow-blue-600/30 text-white hover:bg-blue-700 transition-colors"
                aria-label={item.label}
              >
                <item.icon className="w-6 h-6" />
              </Link>
            );
          }

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'relative flex flex-col items-center justify-center gap-1 w-16 h-full transition-colors',
                isActive ? 'text-blue-600' : 'text-gray-400'
              )}
              aria-current={isActive ? 'page' : undefined}
            >
              <div className="relative">
                <item.icon className="w-5 h-5" />
                {item.badge && (
                  <span className="absolute -top-1.5 -right-1.5 min-w-[16px] h-4 flex items-center justify-center px-1 text-[10px] font-bold bg-red-500 text-white rounded-full">
                    {item.badge > 99 ? '99+' : item.badge}
                  </span>
                )}
              </div>
              <span className="text-[10px] font-medium">{item.label}</span>
              {/* アクティブインジケーター */}
              {isActive && (
                <motion.div
                  layoutId="bottom-nav-indicator"
                  className="absolute top-0 left-1/2 -translate-x-1/2 w-8 h-0.5 bg-blue-600 rounded-full"
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                />
              )}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
```

### 7.2 セーフエリア対応（iOS ノッチ/ホームインジケーター）

```typescript
// セーフエリア対応のCSS
// globals.css に追加
const safeAreaCSS = `
/* iOS セーフエリア対応 */
.safe-area-bottom {
  padding-bottom: env(safe-area-inset-bottom, 0px);
}

.safe-area-top {
  padding-top: env(safe-area-inset-top, 0px);
}

/* ボトムナビゲーション分のスペース確保 */
.has-bottom-nav {
  padding-bottom: calc(64px + env(safe-area-inset-bottom, 0px));
}

/* PWA スタンドアロンモードでの調整 */
@media (display-mode: standalone) {
  .safe-area-bottom {
    padding-bottom: env(safe-area-inset-bottom, 20px);
  }
}
`;

// セーフエリア検出カスタムフック
function useSafeArea() {
  const [safeArea, setSafeArea] = useState({
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
  });

  useEffect(() => {
    const computeStyles = () => {
      const style = getComputedStyle(document.documentElement);
      setSafeArea({
        top: parseInt(style.getPropertyValue('--sat') || '0', 10),
        bottom: parseInt(style.getPropertyValue('--sab') || '0', 10),
        left: parseInt(style.getPropertyValue('--sal') || '0', 10),
        right: parseInt(style.getPropertyValue('--sar') || '0', 10),
      });
    };

    // CSS カスタムプロパティでセーフエリアを設定
    document.documentElement.style.setProperty(
      '--sat',
      'env(safe-area-inset-top, 0px)'
    );
    document.documentElement.style.setProperty(
      '--sab',
      'env(safe-area-inset-bottom, 0px)'
    );
    document.documentElement.style.setProperty(
      '--sal',
      'env(safe-area-inset-left, 0px)'
    );
    document.documentElement.style.setProperty(
      '--sar',
      'env(safe-area-inset-right, 0px)'
    );

    computeStyles();
    window.addEventListener('resize', computeStyles);
    return () => window.removeEventListener('resize', computeStyles);
  }, []);

  return safeArea;
}
```

### 7.3 ボトムナビゲーションのガイドライン

```
ボトムナビゲーション設計のガイドライン:

  項目数:
    → 3〜5項目に制限する（Material Design推奨）
    → 2項目以下はタブバーの方が適切
    → 6項目以上はハンバーガーメニューやMore タブで対応

  ラベル:
    → 全項目にテキストラベルを付与する
    → アイコンのみは認識性が低い
    → 短いラベル（1-2単語）を使用

  アイコン:
    → 一目で理解できるシンプルなアイコン
    → アクティブ状態は塗りつぶし、非アクティブはアウトライン
    → サイズは24-28dp程度

  フィードバック:
    → タップ時のリップルエフェクト
    → アクティブ項目の明確なハイライト
    → スムーズなトランジションアニメーション

  スクロール動作:
    → 下スクロールでボトムナビを非表示にすることも検討
    → コンテンツの閲覧性を優先する場合に有効
    → ただし再アクセス性が低下するトレードオフ

  アンチパターン:
    ❌ デスクトップでもボトムナビを表示する
    ❌ ラベルなしのアイコンのみ表示
    ❌ スクロール連動で常に非表示にする
    ❌ ボトムナビ内にサブメニューを配置する
```

---

## 8. メガメニュー

### 8.1 ECサイト向けメガメニュー

メガメニューは、ECサイトやポータルサイトで大量のカテゴリやコンテンツを整理して表示するためのナビゲーションパターンである。ドロップダウンメニューの拡張版として、グリッドレイアウトや画像を含むリッチなコンテンツを表示できる。

```typescript
// メガメニュー実装
'use client';
import { useState, useRef, useEffect, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { cn } from '@/lib/utils';
import { ChevronDownIcon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface MegaMenuCategory {
  name: string;
  href: string;
  subcategories: {
    name: string;
    href: string;
    items?: { name: string; href: string }[];
  }[];
  featured?: {
    title: string;
    description: string;
    href: string;
    image: string;
  };
}

const megaMenuData: MegaMenuCategory[] = [
  {
    name: 'Electronics',
    href: '/categories/electronics',
    subcategories: [
      {
        name: 'Smartphones',
        href: '/categories/electronics/smartphones',
        items: [
          { name: 'iPhone', href: '/categories/electronics/smartphones/iphone' },
          { name: 'Samsung Galaxy', href: '/categories/electronics/smartphones/samsung' },
          { name: 'Google Pixel', href: '/categories/electronics/smartphones/pixel' },
        ],
      },
      {
        name: 'Laptops',
        href: '/categories/electronics/laptops',
        items: [
          { name: 'MacBook', href: '/categories/electronics/laptops/macbook' },
          { name: 'ThinkPad', href: '/categories/electronics/laptops/thinkpad' },
          { name: 'Surface', href: '/categories/electronics/laptops/surface' },
        ],
      },
      {
        name: 'Audio',
        href: '/categories/electronics/audio',
        items: [
          { name: 'Headphones', href: '/categories/electronics/audio/headphones' },
          { name: 'Speakers', href: '/categories/electronics/audio/speakers' },
          { name: 'Earbuds', href: '/categories/electronics/audio/earbuds' },
        ],
      },
    ],
    featured: {
      title: 'New iPhone 16 Pro',
      description: '最新のA18 Proチップ搭載',
      href: '/products/iphone-16-pro',
      image: '/images/featured/iphone-16.jpg',
    },
  },
  {
    name: 'Fashion',
    href: '/categories/fashion',
    subcategories: [
      {
        name: "Men's",
        href: '/categories/fashion/mens',
        items: [
          { name: 'T-Shirts', href: '/categories/fashion/mens/tshirts' },
          { name: 'Jackets', href: '/categories/fashion/mens/jackets' },
          { name: 'Shoes', href: '/categories/fashion/mens/shoes' },
        ],
      },
      {
        name: "Women's",
        href: '/categories/fashion/womens',
        items: [
          { name: 'Dresses', href: '/categories/fashion/womens/dresses' },
          { name: 'Tops', href: '/categories/fashion/womens/tops' },
          { name: 'Accessories', href: '/categories/fashion/womens/accessories' },
        ],
      },
    ],
    featured: {
      title: 'Spring Collection 2026',
      description: '春の新作コレクション',
      href: '/collections/spring-2026',
      image: '/images/featured/spring-collection.jpg',
    },
  },
];

function MegaMenu() {
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleMouseEnter = useCallback((categoryName: string) => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setActiveCategory(categoryName);
  }, []);

  const handleMouseLeave = useCallback(() => {
    timeoutRef.current = setTimeout(() => {
      setActiveCategory(null);
    }, 200);
  }, []);

  // メニュー外クリックで閉じる
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setActiveCategory(null);
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const activeData = megaMenuData.find((cat) => cat.name === activeCategory);

  return (
    <div ref={menuRef} className="relative">
      {/* トリガーボタン */}
      <div className="flex items-center gap-6">
        {megaMenuData.map((category) => (
          <button
            key={category.name}
            onMouseEnter={() => handleMouseEnter(category.name)}
            onMouseLeave={handleMouseLeave}
            className={cn(
              'flex items-center gap-1 px-3 py-2 text-sm font-medium transition-colors',
              activeCategory === category.name
                ? 'text-blue-600'
                : 'text-gray-700 hover:text-gray-900'
            )}
          >
            {category.name}
            <ChevronDownIcon
              className={cn(
                'w-4 h-4 transition-transform duration-200',
                activeCategory === category.name ? 'rotate-180' : ''
              )}
            />
          </button>
        ))}
      </div>

      {/* メガメニューパネル */}
      <AnimatePresence>
        {activeData && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            onMouseEnter={() => handleMouseEnter(activeData.name)}
            onMouseLeave={handleMouseLeave}
            className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl shadow-xl border border-gray-200 p-6 z-50"
            style={{ minWidth: '700px' }}
          >
            <div className="flex gap-8">
              {/* カテゴリグリッド */}
              <div className="flex-1 grid grid-cols-3 gap-8">
                {activeData.subcategories.map((subcat) => (
                  <div key={subcat.name}>
                    <Link
                      href={subcat.href}
                      className="text-sm font-semibold text-gray-900 hover:text-blue-600 transition-colors"
                    >
                      {subcat.name}
                    </Link>
                    {subcat.items && (
                      <ul className="mt-2 space-y-1.5">
                        {subcat.items.map((item) => (
                          <li key={item.name}>
                            <Link
                              href={item.href}
                              className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
                            >
                              {item.name}
                            </Link>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>

              {/* フィーチャードコンテンツ */}
              {activeData.featured && (
                <div className="w-64 flex-shrink-0">
                  <Link
                    href={activeData.featured.href}
                    className="group block rounded-lg overflow-hidden"
                  >
                    <div className="relative h-40 bg-gray-100 rounded-lg overflow-hidden">
                      <Image
                        src={activeData.featured.image}
                        alt={activeData.featured.title}
                        fill
                        className="object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                    </div>
                    <div className="mt-3">
                      <h4 className="text-sm font-semibold text-gray-900 group-hover:text-blue-600">
                        {activeData.featured.title}
                      </h4>
                      <p className="text-xs text-gray-500 mt-1">
                        {activeData.featured.description}
                      </p>
                    </div>
                  </Link>
                </div>
              )}
            </div>

            {/* フッターリンク */}
            <div className="mt-6 pt-4 border-t border-gray-100">
              <Link
                href={activeData.href}
                className="text-sm font-medium text-blue-600 hover:text-blue-700"
              >
                View all {activeData.name} →
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
```

### 8.2 メガメニューのアクセシビリティ

```typescript
// アクセシビリティ対応メガメニューの主要パターン
function AccessibleMegaMenu() {
  const [activeIndex, setActiveIndex] = useState(-1);
  const menuItemsRef = useRef<(HTMLButtonElement | null)[]>([]);

  const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
    switch (e.key) {
      case 'ArrowRight':
        e.preventDefault();
        const nextIndex = (index + 1) % megaMenuData.length;
        setActiveIndex(nextIndex);
        menuItemsRef.current[nextIndex]?.focus();
        break;

      case 'ArrowLeft':
        e.preventDefault();
        const prevIndex = (index - 1 + megaMenuData.length) % megaMenuData.length;
        setActiveIndex(prevIndex);
        menuItemsRef.current[prevIndex]?.focus();
        break;

      case 'ArrowDown':
        e.preventDefault();
        // メガメニューパネル内の最初のリンクにフォーカス
        const panel = document.getElementById(`mega-panel-${index}`);
        const firstLink = panel?.querySelector('a');
        firstLink?.focus();
        break;

      case 'Escape':
        setActiveIndex(-1);
        menuItemsRef.current[index]?.focus();
        break;
    }
  };

  return (
    <nav aria-label="Main navigation">
      <ul role="menubar" className="flex items-center gap-4">
        {megaMenuData.map((category, index) => (
          <li key={category.name} role="none">
            <button
              ref={(el) => { menuItemsRef.current[index] = el; }}
              role="menuitem"
              aria-haspopup="true"
              aria-expanded={activeIndex === index}
              aria-controls={`mega-panel-${index}`}
              onKeyDown={(e) => handleKeyDown(e, index)}
              onClick={() => setActiveIndex(activeIndex === index ? -1 : index)}
              onMouseEnter={() => setActiveIndex(index)}
              className="px-3 py-2 text-sm font-medium"
            >
              {category.name}
            </button>
            {activeIndex === index && (
              <div
                id={`mega-panel-${index}`}
                role="menu"
                aria-label={`${category.name} submenu`}
              >
                {/* メガメニューパネルの内容 */}
              </div>
            )}
          </li>
        ))}
      </ul>
    </nav>
  );
}
```

---

## 9. レスポンシブナビゲーション戦略

### 9.1 ブレークポイント別ナビゲーション切り替え

異なるデバイスサイズに応じて最適なナビゲーションパターンを自動的に切り替える戦略を実装する。

```
レスポンシブナビゲーション戦略:

  デスクトップ（≥ 1280px）:
  ┌──────────┬──────────────────────────────────────┐
  │          │                                      │
  │ サイドバー │       メインコンテンツ                  │
  │ (常時表示) │                                      │
  │ 256px    │       ブレッドクラム                    │
  │          │       タブナビゲーション                 │
  │          │       コンテンツ                       │
  │          │                                      │
  └──────────┴──────────────────────────────────────┘

  タブレット（768px - 1279px）:
  ┌────┬────────────────────────────────────────────┐
  │    │                                            │
  │icons│       メインコンテンツ                       │
  │64px │                                            │
  │    │       ブレッドクラム                          │
  │    │       タブナビゲーション                      │
  │    │       コンテンツ                             │
  │    │                                            │
  └────┴────────────────────────────────────────────┘
  → サイドバー折りたたみ（アイコンのみ）
  → ホバーで一時的に展開

  モバイル（< 768px）:
  ┌──────────────────────────────────────────────────┐
  │ [≡]  MyApp                          [通知] [人]  │
  ├──────────────────────────────────────────────────┤
  │                                                  │
  │       メインコンテンツ                             │
  │                                                  │
  │       コンテンツ                                  │
  │                                                  │
  ├──────────────────────────────────────────────────┤
  │  Home  Search Create Notif Profile               │
  └──────────────────────────────────────────────────┘
  → ハンバーガーメニュー + ドロワー
  → ボトムナビゲーション
  → スティッキーヘッダー
```

### 9.2 レスポンシブレイアウトの実装

```typescript
// レスポンシブナビゲーションレイアウト
'use client';
import { useState, useEffect, ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

// ブレークポイントの定義
const breakpoints = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
} as const;

// 画面サイズ検出カスタムフック
function useBreakpoint() {
  const [breakpoint, setBreakpoint] = useState<'mobile' | 'tablet' | 'desktop'>('desktop');

  useEffect(() => {
    const checkBreakpoint = () => {
      const width = window.innerWidth;
      if (width < breakpoints.md) {
        setBreakpoint('mobile');
      } else if (width < breakpoints.xl) {
        setBreakpoint('tablet');
      } else {
        setBreakpoint('desktop');
      }
    };

    checkBreakpoint();
    window.addEventListener('resize', checkBreakpoint);
    return () => window.removeEventListener('resize', checkBreakpoint);
  }, []);

  return breakpoint;
}

// メディアクエリベースのカスタムフック（SSR対応）
function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia(query);
    setMatches(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => setMatches(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, [query]);

  return matches;
}

// レスポンシブナビゲーションレイアウト
function ResponsiveLayout({ children }: { children: ReactNode }) {
  const breakpoint = useBreakpoint();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // ブレークポイント変更時にモバイルメニューを閉じる
  useEffect(() => {
    if (breakpoint !== 'mobile') {
      setMobileMenuOpen(false);
    }
    if (breakpoint === 'tablet') {
      setSidebarCollapsed(true);
    } else if (breakpoint === 'desktop') {
      setSidebarCollapsed(false);
    }
  }, [breakpoint]);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* デスクトップ/タブレット: サイドバー */}
      {breakpoint !== 'mobile' && (
        <aside
          className={cn(
            'flex-shrink-0 bg-gray-900 text-white transition-all duration-300',
            sidebarCollapsed ? 'w-16' : 'w-64'
          )}
        >
          <Sidebar collapsed={sidebarCollapsed} />
        </aside>
      )}

      {/* メインコンテンツエリア */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* モバイル: トップヘッダー */}
        {breakpoint === 'mobile' && (
          <header className="flex items-center justify-between h-14 px-4 bg-white border-b border-gray-200">
            <button
              onClick={() => setMobileMenuOpen(true)}
              className="p-2 rounded-md hover:bg-gray-100"
              aria-label="Open menu"
            >
              <MenuIcon className="w-5 h-5" />
            </button>
            <span className="text-lg font-bold">MyApp</span>
            <div className="flex items-center gap-2">
              <button className="p-2 rounded-md hover:bg-gray-100">
                <BellIcon className="w-5 h-5" />
              </button>
            </div>
          </header>
        )}

        {/* メインコンテンツ */}
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>

        {/* モバイル: ボトムナビゲーション */}
        {breakpoint === 'mobile' && <BottomNavigation />}
      </div>

      {/* モバイル: ドロワーメニュー */}
      <AnimatePresence>
        {mobileMenuOpen && breakpoint === 'mobile' && (
          <>
            {/* オーバーレイ */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 z-40"
              onClick={() => setMobileMenuOpen(false)}
            />
            {/* ドロワー */}
            <motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed top-0 left-0 bottom-0 w-[280px] bg-gray-900 text-white z-50 overflow-y-auto"
            >
              <div className="flex items-center justify-between p-4 border-b border-gray-800">
                <span className="text-lg font-bold">MyApp</span>
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-1 rounded-md hover:bg-gray-800"
                  aria-label="Close menu"
                >
                  <XIcon className="w-5 h-5" />
                </button>
              </div>
              <Sidebar collapsed={false} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
```

### 9.3 モバイルドロワーナビゲーション（shadcn/ui Sheet）

```typescript
// shadcn/ui Sheet を使ったモバイルドロワー
'use client';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
  SheetClose,
} from '@/components/ui/sheet';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

function MobileDrawerNav() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  // ページ遷移時にドロワーを閉じる
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <button className="p-2 rounded-md hover:bg-gray-100 md:hidden">
          <MenuIcon className="w-5 h-5" />
        </button>
      </SheetTrigger>
      <SheetContent side="left" className="w-[280px] p-0 bg-gray-900">
        <SheetHeader className="p-4 border-b border-gray-800">
          <SheetTitle className="text-white text-lg">MyApp</SheetTitle>
        </SheetHeader>
        <nav className="p-3">
          {navigation.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setOpen(false)}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                pathname.startsWith(item.href)
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              )}
            >
              <item.icon className="w-5 h-5" />
              {item.name}
              {item.badge && (
                <span className="ml-auto px-2 py-0.5 text-xs bg-red-500 text-white rounded-full">
                  {item.badge}
                </span>
              )}
            </Link>
          ))}
        </nav>
      </SheetContent>
    </Sheet>
  );
}
```

---

## 10. ナビゲーションのアクセシビリティ

### 10.1 WAI-ARIA によるナビゲーションマークアップ

```
ナビゲーションのアクセシビリティ要件:

  1. ランドマーク
     → <nav> 要素を使用する
     → 複数の <nav> がある場合は aria-label で区別
     → <nav aria-label="Main navigation">
     → <nav aria-label="Breadcrumb">

  2. 現在地の表示
     → 現在のページに aria-current="page" を設定
     → 現在のセクションに aria-current="true" を設定
     → スクリーンリーダーが現在位置を読み上げ可能

  3. キーボード操作
     → Tab / Shift+Tab でフォーカス移動
     → Enter / Space でリンク遷移・ボタン操作
     → 矢印キーでメニュー内移動（メニューバーの場合）
     → Escape でサブメニュー閉じる
     → Home / End で最初/最後の項目に移動

  4. フォーカス管理
     → フォーカスリングを常に表示（:focus-visible）
     → フォーカストラップ（モーダル/ドロワー内）
     → フォーカス復元（メニュー閉じた後に元の要素に戻す）

  5. スクリーンリーダー対応
     → 装飾的アイコンに aria-hidden="true"
     → アイコンのみのボタンに aria-label を設定
     → メニュー展開状態に aria-expanded を設定
     → サブメニューに aria-haspopup を設定

  6. カラーコントラスト
     → テキストと背景のコントラスト比 4.5:1 以上
     → アクティブ状態の識別が色のみに依存しない
     → フォーカスリングは十分に視認可能
```

### 10.2 フォーカストラップの実装

```typescript
// モーダル/ドロワー用フォーカストラップ
import { useEffect, useRef, useCallback } from 'react';

function useFocusTrap(isActive: boolean) {
  const containerRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  // フォーカス可能な要素を取得
  const getFocusableElements = useCallback(() => {
    if (!containerRef.current) return [];
    const elements = containerRef.current.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), textarea:not([disabled]), ' +
      'input:not([disabled]):not([type="hidden"]), select:not([disabled]), ' +
      '[tabindex]:not([tabindex="-1"])'
    );
    return Array.from(elements);
  }, []);

  useEffect(() => {
    if (!isActive) return;

    // 現在のフォーカスを保存
    previousFocusRef.current = document.activeElement as HTMLElement;

    // コンテナ内の最初の要素にフォーカス
    const focusableElements = getFocusableElements();
    if (focusableElements.length > 0) {
      focusableElements[0].focus();
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      const focusable = getFocusableElements();
      if (focusable.length === 0) return;

      const firstElement = focusable[0];
      const lastElement = focusable[focusable.length - 1];

      if (e.shiftKey) {
        // Shift+Tab: 最初の要素から最後にループ
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        // Tab: 最後の要素から最初にループ
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      // フォーカスを復元
      previousFocusRef.current?.focus();
    };
  }, [isActive, getFocusableElements]);

  return containerRef;
}

// 使用例: ドロワーメニュー
function AccessibleDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const containerRef = useFocusTrap(open);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (open) {
      document.addEventListener('keydown', handleEscape);
      // スクロール禁止
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      ref={containerRef}
      role="dialog"
      aria-modal="true"
      aria-label="Navigation menu"
      className="fixed inset-0 z-50"
    >
      {/* オーバーレイ */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
        aria-hidden="true"
      />
      {/* ドロワー内容 */}
      <div className="absolute top-0 left-0 bottom-0 w-[280px] bg-white shadow-xl">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 rounded-md hover:bg-gray-100"
          aria-label="Close navigation menu"
        >
          <XIcon className="w-5 h-5" />
        </button>
        <nav aria-label="Main navigation" className="p-4 mt-12">
          {/* ナビゲーション項目 */}
        </nav>
      </div>
    </div>
  );
}
```

### 10.3 スキップリンクの実装

```typescript
// スキップリンク（ページの先頭に配置）
function SkipLinks() {
  return (
    <div className="sr-only focus-within:not-sr-only">
      <a
        href="#main-content"
        className="fixed top-2 left-2 z-[100] px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
      >
        Skip to main content
      </a>
      <a
        href="#main-navigation"
        className="fixed top-2 left-48 z-[100] px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
      >
        Skip to navigation
      </a>
    </div>
  );
}

// レイアウトでの使用
function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ja">
      <body>
        <SkipLinks />
        <nav id="main-navigation" aria-label="Main navigation">
          <Sidebar />
        </nav>
        <main id="main-content">
          {children}
        </main>
      </body>
    </html>
  );
}
```

### 10.4 アクセシビリティチェックリスト

```
ナビゲーション アクセシビリティ チェックリスト:

  構造:
    [ ] <nav> 要素を使用している
    [ ] 複数の <nav> に aria-label で区別している
    [ ] リスト（<ul>/<ol>）でナビゲーション項目を構造化している
    [ ] スキップリンクを実装している

  状態:
    [ ] aria-current="page" で現在のページを表示
    [ ] aria-expanded で展開/折りたたみ状態を表示
    [ ] aria-haspopup でサブメニューの存在を表示
    [ ] aria-selected でタブの選択状態を表示

  キーボード:
    [ ] Tab で全ナビゲーション項目にアクセス可能
    [ ] Enter / Space でアクション実行
    [ ] Escape でサブメニュー/ドロワーを閉じる
    [ ] フォーカスリングが常に表示される
    [ ] フォーカストラップが適切に実装されている

  視覚:
    [ ] テキストのコントラスト比 4.5:1 以上
    [ ] アクティブ状態が色以外でも識別可能
    [ ] フォーカスリングのコントラスト比 3:1 以上
    [ ] ターゲットサイズ 44x44px 以上（モバイル）

  スクリーンリーダー:
    [ ] 装飾的要素に aria-hidden="true"
    [ ] アイコンボタンに aria-label
    [ ] バッジ数にスクリーンリーダーテキスト
    [ ] 動的コンテンツ変更を aria-live で通知
```

---

## 11. パフォーマンス最適化

### 11.1 ナビゲーションの再レンダリング最適化

```typescript
// React.memo によるナビゲーション項目の最適化
import { memo, useMemo } from 'react';

// ナビゲーション項目を memo 化
const MemoizedNavItem = memo(function MemoizedNavItem({
  item,
  isActive,
  isExpanded,
  onToggle,
}: {
  item: NavItem;
  isActive: boolean;
  isExpanded: boolean;
  onToggle: (href: string) => void;
}) {
  return (
    <div>
      <Link
        href={item.href}
        className={cn(
          'flex items-center gap-3 px-3 py-2 rounded-lg text-sm',
          isActive ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
        )}
        aria-current={isActive ? 'page' : undefined}
      >
        <item.icon className="w-5 h-5" />
        <span>{item.name}</span>
      </Link>
    </div>
  );
});

// サイドバー全体の最適化
function OptimizedSidebar() {
  const pathname = usePathname();

  // ナビゲーション設定を memo 化
  const navItems = useMemo(() => navigation, []);

  // アクティブ状態の計算を memo 化
  const activeStates = useMemo(() => {
    return new Map(
      navItems.map((item) => [item.href, pathname.startsWith(item.href)])
    );
  }, [navItems, pathname]);

  return (
    <nav>
      {navItems.map((item) => (
        <MemoizedNavItem
          key={item.href}
          item={item}
          isActive={activeStates.get(item.href) || false}
          isExpanded={false}
          onToggle={() => {}}
        />
      ))}
    </nav>
  );
}
```

### 11.2 ルートプリフェッチと遅延読み込み

```typescript
// Next.js のプリフェッチ最適化
import Link from 'next/link';

// デフォルト: ビューポート内のリンクを自動プリフェッチ
function NavigationWithPrefetch() {
  return (
    <nav>
      {/* 頻繁に使うページ: プリフェッチ有効（デフォルト） */}
      <Link href="/dashboard" prefetch={true}>
        Dashboard
      </Link>

      {/* まれに使うページ: プリフェッチ無効 */}
      <Link href="/settings/api-keys" prefetch={false}>
        API Keys
      </Link>

      {/* ホバー時にプリフェッチ（カスタム実装） */}
      <HoverPrefetchLink href="/analytics">
        Analytics
      </HoverPrefetchLink>
    </nav>
  );
}

// ホバー時プリフェッチのカスタムリンク
function HoverPrefetchLink({
  href,
  children,
  ...props
}: {
  href: string;
  children: ReactNode;
}) {
  const router = useRouter();
  const [prefetched, setPrefetched] = useState(false);

  const handleMouseEnter = useCallback(() => {
    if (!prefetched) {
      router.prefetch(href);
      setPrefetched(true);
    }
  }, [href, prefetched, router]);

  return (
    <Link
      href={href}
      onMouseEnter={handleMouseEnter}
      prefetch={false}
      {...props}
    >
      {children}
    </Link>
  );
}

// ナビゲーションコンポーネントの遅延読み込み
import dynamic from 'next/dynamic';

const LazyCommandPalette = dynamic(() => import('@/components/CommandPalette'), {
  ssr: false,
  loading: () => null, // 非表示のため loading 不要
});

const LazyMegaMenu = dynamic(() => import('@/components/MegaMenu'), {
  ssr: false,
  loading: () => <div className="h-10 bg-gray-100 animate-pulse rounded" />,
});
```

### 11.3 ナビゲーションのバンドルサイズ最適化

```
バンドルサイズ最適化のポイント:

  1. アイコンのツリーシェイキング
     → import { HomeIcon } from 'lucide-react'  (個別インポート ✅)
     → import * as Icons from 'lucide-react'     (全インポート ❌)
     → アイコンライブラリの選定（lucide-react は軽量）

  2. コマンドパレットの遅延読み込み
     → dynamic import で初期バンドルに含めない
     → ユーザーが Cmd+K を押した時に読み込み
     → ssr: false で SSR を無効化

  3. アニメーションライブラリの選定
     → framer-motion: 高機能だがバンドルサイズ大（約30KB gzipped）
     → CSS transitions: ゼロコスト
     → @formkit/auto-animate: 軽量（約2KB）
     → 軽微なアニメーションは CSS で十分

  4. ナビゲーション設定の分離
     → ナビゲーション定義を別ファイルに分離
     → 権限に基づくフィルタリングをサーバーサイドで実行
     → 不要な項目をクライアントに送らない

  5. 画像の最適化（メガメニュー）
     → next/image で自動最適化
     → lazy loading でビューポート外は遅延読み込み
     → WebP/AVIF フォーマットの使用
```

---

## 12. トラブルシューティング

### 12.1 よくある問題と解決策

```
問題1: サイドバーのアクティブ状態が正しく反映されない

  症状:
    → /users/123/edit にいるのに Users がハイライトされない
    → 複数の項目がアクティブになる

  原因:
    → パスの完全一致で判定している
    → パスの前方一致が広すぎる（/ が全てにマッチ）

  解決策:
    // ❌ 完全一致のみ
    const isActive = pathname === item.href;

    // ❌ 広すぎる前方一致
    const isActive = pathname.startsWith(item.href);

    // ✅ 適切な判定
    const isActive = pathname === item.href ||
      (item.href !== '/' && pathname.startsWith(item.href + '/'));

    // ✅ 子パスも考慮した判定
    const isActive = item.children
      ? item.children.some(child =>
          pathname === child.href || pathname.startsWith(child.href + '/'))
      : pathname === item.href || pathname.startsWith(item.href + '/');


問題2: モバイルメニューがページ遷移後も開いたまま

  症状:
    → リンクをクリックしてもドロワーが閉じない
    → Next.js のクライアントサイドナビゲーションで発生

  原因:
    → pathname の変更を監視していない
    → Link コンポーネントの onClick で閉じていない

  解決策:
    // pathname 変更で自動的に閉じる
    const pathname = usePathname();
    useEffect(() => {
      setMobileMenuOpen(false);
    }, [pathname]);


問題3: スクロール時にサイドバーがちらつく

  症状:
    → スクロールするたびにサイドバーが再レンダリングされる
    → パフォーマンスが低下する

  原因:
    → サイドバーがメインコンテンツと同じスクロールコンテナにある
    → スクロールイベントで不要な状態更新が発生

  解決策:
    // ✅ サイドバーを固定、コンテンツのみスクロール
    <div className="flex h-screen overflow-hidden">
      <aside className="w-64 h-screen overflow-y-auto flex-shrink-0">
        <Sidebar />
      </aside>
      <main className="flex-1 overflow-y-auto">
        {children}
      </main>
    </div>


問題4: コマンドパレットが他の要素の下に表示される

  症状:
    → コマンドパレットがモーダルやドロワーの下に隠れる
    → z-index の競合

  原因:
    → z-index の管理が不適切
    → stacking context の理解不足

  解決策:
    // z-index の統一管理
    // tailwind.config.js
    module.exports = {
      theme: {
        extend: {
          zIndex: {
            'dropdown': '10',
            'sticky': '20',
            'fixed': '30',
            'drawer-overlay': '40',
            'drawer': '41',
            'modal-overlay': '50',
            'modal': '51',
            'command-palette': '60',
            'toast': '70',
            'tooltip': '80',
          },
        },
      },
    };


問題5: iOS Safari でボトムナビがホームインジケーターと重なる

  症状:
    → iPhone X以降のホームインジケーター領域にボトムナビが被る
    → タップしにくい

  原因:
    → env(safe-area-inset-bottom) を使用していない
    → viewport meta タグの設定不足

  解決策:
    <!-- viewport meta タグ -->
    <meta name="viewport"
      content="width=device-width, initial-scale=1, viewport-fit=cover" />

    /* CSS セーフエリア対応 */
    .bottom-nav {
      padding-bottom: env(safe-area-inset-bottom, 0px);
    }


問題6: SSR/SSGでナビゲーションのアクティブ状態が一瞬ずれる

  症状:
    → ページ初期表示時にアクティブ状態がちらつく
    → Hydration mismatch の警告が出る

  原因:
    → サーバーとクライアントでの pathname の不一致
    → useState の初期値がサーバーとクライアントで異なる

  解決策:
    // 'use client' コンポーネントで usePathname を使用
    // サーバーコンポーネントでは pathname を props で渡す

    // Server Component
    import { headers } from 'next/headers';

    async function Layout({ children }) {
      const headersList = headers();
      const pathname = headersList.get('x-pathname') || '/';

      return (
        <div>
          <Sidebar initialPathname={pathname} />
          {children}
        </div>
      );
    }
```

### 12.2 デバッグテクニック

```typescript
// ナビゲーション状態のデバッグコンポーネント
function NavigationDebugger() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const { state } = useNavigation();

  if (process.env.NODE_ENV !== 'development') return null;

  return (
    <div className="fixed bottom-4 right-4 z-[999] p-4 bg-black/90 text-white text-xs rounded-lg max-w-sm font-mono">
      <h3 className="font-bold mb-2 text-yellow-400">Nav Debug</h3>
      <div className="space-y-1">
        <div>
          <span className="text-gray-400">pathname:</span> {pathname}
        </div>
        <div>
          <span className="text-gray-400">params:</span>{' '}
          {searchParams.toString() || '(none)'}
        </div>
        <div>
          <span className="text-gray-400">sidebar:</span>{' '}
          {state.sidebarOpen ? 'open' : 'closed'}
          {state.sidebarCollapsed ? ' (collapsed)' : ''}
        </div>
        <div>
          <span className="text-gray-400">mobile:</span>{' '}
          {state.mobileMenuOpen ? 'open' : 'closed'}
        </div>
        <div>
          <span className="text-gray-400">expanded:</span>{' '}
          {Array.from(state.expandedItems).join(', ') || '(none)'}
        </div>
        <div>
          <span className="text-gray-400">recent:</span>{' '}
          {state.recentPages.slice(0, 3).join(', ') || '(none)'}
        </div>
      </div>
    </div>
  );
}
```

---

## 13. ナビゲーションパターンの比較とまとめ

### 13.1 パターン別比較表

| パターン | 用途 | 項目数 | モバイル対応 | アクセシビリティ | 実装難易度 |
|---------|------|--------|------------|----------------|-----------|
| サイドバー | 管理画面、SaaS | 5〜20+ | ドロワー切替 | 高 | 中〜高 |
| トップナビ | マーケティング、LP | 3〜7 | ハンバーガー | 高 | 低〜中 |
| ブレッドクラム | 階層の可視化 | 動的 | 省略表示 | 高 | 低 |
| タブナビ | 設定、詳細ページ | 3〜8 | ドロップダウン | 中 | 低 |
| コマンドパレット | パワーユーザー | 無制限 | 対応可能 | 中 | 高 |
| ボトムナビ | モバイルアプリ | 3〜5 | ネイティブ | 中 | 低 |
| メガメニュー | ECサイト、ポータル | 大量 | 別パターン | 中〜高 | 高 |

### 13.2 組み合わせパターン

```
推奨される組み合わせ:

  SaaS管理画面:
    → サイドバー + ブレッドクラム + コマンドパレット + タブナビ
    → モバイル: ドロワー + ボトムナビ

  ECサイト:
    → トップナビ + メガメニュー + ブレッドクラム
    → モバイル: ハンバーガー + ボトムナビ

  コーポレートサイト:
    → トップナビ + ブレッドクラム
    → モバイル: ハンバーガー

  ブログ/メディア:
    → トップナビ + サイドバー（カテゴリ）+ ブレッドクラム
    → モバイル: ハンバーガー

  開発者ツール:
    → サイドバー + コマンドパレット + タブナビ
    → モバイル: ドロワー
```

### 13.3 設計判断のフローチャート

```
ナビゲーションパターン選定フロー:

  アプリの種類は？
  ├── Webアプリケーション（SaaS/管理画面）
  │   ├── ナビ項目 <= 7 → トップナビ + タブ
  │   └── ナビ項目 > 7 → サイドバー + タブ
  │       ├── パワーユーザー多い → + コマンドパレット
  │       └── 階層が深い → + ブレッドクラム
  │
  ├── Webサイト（マーケティング/コーポレート）
  │   ├── カテゴリ少ない → トップナビ
  │   └── カテゴリ多い → トップナビ + メガメニュー
  │
  ├── ECサイト
  │   └── メガメニュー + ブレッドクラム + (検索)
  │
  └── モバイルアプリ/PWA
      ├── メイン機能 <= 5 → ボトムナビ
      └── メイン機能 > 5 → ボトムナビ + ドロワー
```

### 13.4 ナビゲーション設計の原則（まとめ）

```
ナビゲーション設計で最も重要な7つの原則:

  1. ユーザーは常に自分の位置を把握できること
     → アクティブ状態のハイライト
     → ブレッドクラムによる階層表示
     → URL の意味のある構造

  2. 最も重要な機能に最短でアクセスできること
     → トップレベルに主要機能を配置
     → ショートカットキーの提供
     → コマンドパレットによる検索

  3. 一貫性を保つこと
     → 全ページでナビゲーション位置を統一
     → スタイルとインタラクションの統一
     → プラットフォーム慣習への準拠

  4. デバイスに最適化すること
     → レスポンシブ設計の徹底
     → タッチ操作への対応
     → セーフエリアへの配慮

  5. アクセシビリティを確保すること
     → キーボード操作の完全サポート
     → スクリーンリーダー対応
     → 十分なコントラスト比

  6. パフォーマンスを維持すること
     → 不要な再レンダリングの防止
     → 遅延読み込みの活用
     → バンドルサイズの最適化

  7. スケーラビリティを考慮すること
     → 機能追加に耐える構造
     → グルーピングと整理の仕組み
     → 設定の外部化
```

---

## 前提知識

この章を最大限に活用するために、以下の知識を事前に習得しておくことを推奨する。

- **ファイルベースルーティング**: Next.js App Router のファイル規約とルート構造 → `./01-file-based-routing.md`
- **ブラウザのナビゲーション**: ブラウザがページ遷移をどのように処理するか → `../../browser-and-web-platform/docs/00-browser-engine/01-navigation-and-loading.md`
- **React の基本**: コンポーネント設計、状態管理、フックの使い方

これらの概念を理解することで、効果的なナビゲーションUIを設計・実装できる。

---

## FAQ

### Q1: プリフェッチの最適化戦略は?

**A:** プリフェッチは帯域幅とメモリを消費するため、ユーザーの行動パターンとデバイスの状態に応じて最適化する。

```typescript
// =========================================
// Next.js のプリフェッチ制御
// =========================================

// デフォルト: ビューポート内の Link は自動プリフェッチ
<Link href="/dashboard">Dashboard</Link>

// プリフェッチを無効化
<Link href="/settings" prefetch={false}>Settings</Link>

// 動的にプリフェッチ
import { useRouter } from 'next/navigation';

function Navigation() {
  const router = useRouter();

  return (
    <nav
      onMouseEnter={() => {
        // ホバー時にプリフェッチ
        router.prefetch('/dashboard');
      }}
    >
      <a href="/dashboard">Dashboard</a>
    </nav>
  );
}

// =========================================
// 条件付きプリフェッチ
// =========================================

function SmartLink({ href, children }: { href: string; children: React.ReactNode }) {
  const [shouldPrefetch, setShouldPrefetch] = useState(false);

  useEffect(() => {
    // ネットワーク状態を確認
    const connection = (navigator as any).connection;
    if (connection) {
      // Wi-Fi または高速接続時のみプリフェッチ
      const isFastConnection =
        connection.effectiveType === '4g' ||
        connection.type === 'wifi';
      setShouldPrefetch(isFastConnection);
    }
  }, []);

  return (
    <Link href={href} prefetch={shouldPrefetch}>
      {children}
    </Link>
  );
}

// =========================================
// 優先度付きプリフェッチ
// =========================================

function PriorityNavigation() {
  const router = useRouter();

  useEffect(() => {
    // 最優先: ユーザーが頻繁にアクセスするページ
    router.prefetch('/dashboard');

    // 低優先: アクセス頻度が低いページは遅延
    setTimeout(() => {
      router.prefetch('/settings');
      router.prefetch('/profile');
    }, 3000);
  }, [router]);

  return <nav>{/* ... */}</nav>;
}
```

**推奨事項**:
- モバイル・低速回線では prefetch を制限
- ログイン後の最初のページのみ積極的にプリフェッチ
- Intersection Observer でビューポート近くの Link のみプリフェッチ

### Q2: ナビゲーション中のローディングUI実装は?

**A:** Next.js App Router では `loading.tsx` と `useTransition` を組み合わせて、段階的なローディング体験を提供する。

```typescript
// =========================================
// 1. Route レベルのローディング
// =========================================
// app/dashboard/loading.tsx
export default function DashboardLoading() {
  return <DashboardSkeleton />;
}

// app/dashboard/page.tsx
export default async function DashboardPage() {
  const data = await fetchDashboardData(); // 自動的に Suspense でラップ
  return <Dashboard data={data} />;
}

// =========================================
// 2. コンポーネントレベルのローディング
// =========================================
import { Suspense } from 'react';

export default function DashboardPage() {
  return (
    <div>
      <h1>Dashboard</h1>
      <Suspense fallback={<ChartSkeleton />}>
        <Chart />
      </Suspense>
      <Suspense fallback={<TableSkeleton />}>
        <DataTable />
      </Suspense>
    </div>
  );
}

// =========================================
// 3. Navigation ローディング（useTransition）
// =========================================
'use client';

import { useTransition } from 'react';
import { useRouter } from 'next/navigation';

function Navigation() {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();

  const navigate = (path: string) => {
    startTransition(() => {
      router.push(path);
    });
  };

  return (
    <nav>
      {isPending && <LoadingSpinner />}
      <button onClick={() => navigate('/dashboard')}>
        Dashboard
      </button>
    </nav>
  );
}
```

### Q3: Parallel Routes と Intercepting Routes の使い分けは?

**A:** Parallel Routes は同一ページ内で複数の独立したコンテンツを並列表示する場合、Intercepting Routes はモーダル表示など、ナビゲーションの「横取り」をする場合に使用する。

```
Parallel Routes (@slot):

app/
├── layout.tsx
├── @main/
│   └── page.tsx         # メインコンテンツ
└── @sidebar/
    └── page.tsx         # サイドバー

// layout.tsx
export default function Layout({
  main,
  sidebar,
}: {
  main: React.ReactNode;
  sidebar: React.ReactNode;
}) {
  return (
    <div className="flex">
      <aside>{sidebar}</aside>
      <main>{main}</main>
    </div>
  );
}

ユースケース:
  - ダッシュボード（メイン + サイドバー + 通知パネル）
  - 分割ビュー（リスト + 詳細）
  - 各スロットで独立した loading.tsx / error.tsx

Intercepting Routes ((.)path):

app/
├── photos/
│   ├── page.tsx              # 写真一覧
│   ├── [id]/
│   │   └── page.tsx          # 写真詳細（全画面）
│   └── (.)[id]/
│       └── page.tsx          # モーダル表示

使い方:
  - 一覧から写真クリック → モーダルで開く（(.)[id]/page.tsx）
  - URL直接アクセス → 全画面表示（[id]/page.tsx）
  - 戻るボタン → 一覧に戻る（モーダルを閉じる）

インターセプトパターン:
  (.)  → 同じディレクトリ
  (..) → 1つ上のディレクトリ
  (..)(..) → 2つ上
  (...) → app/ ルート
```

---

## まとめ

### ナビゲーションパターンの比較表

| パターン | 用途 | 実装コスト | アクセシビリティ | パフォーマンス |
|---------|------|-----------|----------------|--------------|
| サイドバー | 管理画面・ダッシュボード | 中 | 高（aria 対応必須） | 高 |
| トップナビ | Webサイト・LP | 低 | 高（自然な構造） | 高 |
| ブレッドクラム | 階層的コンテンツ | 低 | 高（aria-label 必須） | 高 |
| タブ | セクション切り替え | 低 | 中（キーボード対応） | 高 |
| コマンドパレット | パワーユーザー向け | 高 | 中（発見性低い） | 中 |
| ボトムナビ | モバイルアプリ・PWA | 中 | 高（タッチ最適化） | 高 |
| メガメニュー | ECサイト・大規模サイト | 高 | 低（複雑な構造） | 中 |

### ナビゲーション設計の3つのキーポイント

1. **ユーザーの位置を常に明示する**
   - アクティブ状態のハイライト（`aria-current="page"`）
   - ブレッドクラムによる階層表示
   - 意味のある URL 構造

2. **デバイスに最適化する**
   - デスクトップ: サイドバー or トップナビ
   - モバイル: ボトムナビ + ハンバーガーメニュー
   - タブレット: レスポンシブな折りたたみサイドバー

3. **パフォーマンスとアクセシビリティを両立する**
   - プリフェッチの最適化（ネットワーク状態を考慮）
   - キーボードナビゲーションの完全サポート
   - スクリーンリーダー対応（ARIA 属性の適切な使用）

---

## 次に読むべきガイド
- [[03-auth-and-guards.md]] -- 認証ガード

---

## 参考文献
1. shadcn/ui. "Sidebar." ui.shadcn.com, 2024.
2. cmdk. "Command Menu." cmdk.paco.me, 2024.
3. Nielsen Norman Group. "Navigation Design." nngroup.com, 2024.
4. Material Design. "Navigation." material.io, 2024.
5. WAI-ARIA Authoring Practices. "Menu and Menubar Pattern." w3.org, 2024.
6. Web Content Accessibility Guidelines (WCAG) 2.2. w3.org, 2023.
7. Apple Human Interface Guidelines. "Navigation." developer.apple.com, 2024.
8. Radix UI. "Navigation Menu." radix-ui.com, 2024.
9. Next.js Documentation. "Linking and Navigating." nextjs.org, 2024.
10. Framer Motion. "Animation." framer.com/motion, 2024.
