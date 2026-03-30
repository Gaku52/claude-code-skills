[日本語版](../../ja/05-infrastructure/windows-application-development/SKILL.md)

# Windows Application Development

> Desktop app development using web technologies has become the mainstream approach. This guide covers the characteristics and selection criteria for Electron, Tauri, and WPF/WinUI, along with cross-platform support, native feature integration, distribution, and updates -- providing a complete picture of Windows desktop application development.

## Target Audience

- Engineers who want to build desktop apps using web technologies (React/TypeScript)
- Developers evaluating and implementing Electron or Tauri
- Those looking to leverage Windows-native features (notifications, system tray, file system, etc.)

## Prerequisites

- HTML/CSS/JavaScript fundamentals
- Basic React/TypeScript development experience
- Foundational Node.js knowledge

## Study Guide

### 00-fundamentals — Desktop App Fundamentals

| # | File | Description |
|---|------|-------------|

### 01-wpf-and-winui — Windows Native

| # | File | Description |
|---|------|-------------|

### 02-electron-and-tauri — Web Technology-Based

| # | File | Description |
|---|------|-------------|

### 03-distribution — Distribution and Updates

| # | File | Description |
|---|------|-------------|

## Quick Reference

```
Technology Selection Guide:

  Lightweight + security-focused → Tauri (recommended)
  Rich ecosystem + proven track record → Electron
  Windows-only + native feel → WinUI 3
  Cross-platform + .NET → MAUI

  Bundle Size Comparison:
    Electron: ~150MB (includes Chromium)
    Tauri:    ~5MB (uses OS WebView)
    WinUI 3:  ~20MB (.NET runtime)

  Memory Usage:
    Electron: ~200MB+
    Tauri:    ~50MB
    WinUI 3:  ~100MB
```

## References

1. Electron. "Documentation." electronjs.org, 2024.
2. Tauri. "Documentation." tauri.app, 2024.
3. Microsoft. "WinUI 3 Documentation." learn.microsoft.com, 2024.
