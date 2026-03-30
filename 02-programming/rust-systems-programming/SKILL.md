[日本語版](../../ja/02-programming/rust-systems-programming/SKILL.md)

# Rust Systems Programming

> Rust is a systems programming language that combines safety, speed, and concurrency. This guide systematically covers everything in Rust -- from the ownership system, lifetimes, and traits to async programming and unsafe code.

## Target Audience

- Engineers who want to learn Rust from the ground up
- Developers doing systems programming (CLI tools, servers, WebAssembly)
- Those who want to achieve both memory safety and high performance

## Prerequisites

- Experience with any programming language
- Basic understanding of memory management concepts

## Study Guide

### 00-basics -- Rust Basics

| # | File | Content |
|---|------|---------|

### 01-ownership -- The Ownership System

| # | File | Content |
|---|------|---------|

### 02-advanced -- Advanced Features

| # | File | Content |
|---|------|---------|

### 03-systems -- Systems Applications

| # | File | Content |
|---|------|---------|

### 04-async -- Async Programming

| # | File | Content |
|---|------|---------|

## Quick Reference

```
Rust Cheat Sheet:
  cargo new myapp          -- Create a new project
  cargo build --release    -- Release build
  cargo test               -- Run tests
  cargo clippy             -- Lint
  cargo fmt                -- Format

  Ownership Rules:
    1. Each value has exactly one owner
    2. When the owner goes out of scope, the value is dropped
    3. References must not outlive the value they point to
```

## References

1. Klabnik, S. & Nichols, C. "The Rust Programming Language." doc.rust-lang.org/book, 2024.
2. Rust. "Rust by Example." doc.rust-lang.org/rust-by-example, 2024.
3. Blandy, J. et al. "Programming Rust." O'Reilly, 2021.
