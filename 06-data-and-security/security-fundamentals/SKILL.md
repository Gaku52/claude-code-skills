[日本語版](../../ja/06-data-and-security/security-fundamentals/SKILL.md)

# Security Fundamentals

> Security is the foundation of software development. This skill systematically covers the essential security knowledge every engineer needs — from the OWASP Top 10 and cryptography to network security, application security, cloud security, and security operations.

## Target Audience

- Engineers who want to build a systematic understanding of security fundamentals
- Developers aiming to build secure applications
- Those responsible for security audits and incident response

## Prerequisites

- Basic understanding of web application architecture
- Foundational networking knowledge (TCP/IP, HTTP)
- Basic Linux command-line skills

## Study Guide

### 00-basics — Security Fundamentals

| # | File | Description |
|---|------|-------------|

### 01-web-security — Web Security

| # | File | Description |
|---|------|-------------|

### 02-cryptography — Cryptography

| # | File | Description |
|---|------|-------------|

### 03-network-security — Network Security

| # | File | Description |
|---|------|-------------|

### 04-application-security — Application Security

| # | File | Description |
|---|------|-------------|

### 05-cloud-security — Cloud Security

| # | File | Description |
|---|------|-------------|

### 06-operations — Security Operations

| # | File | Description |
|---|------|-------------|

## Quick Reference

```
Security Checklist:

  Web Applications:
    ✓ Input validation (server-side is mandatory)
    ✓ Parameterized queries (prevent SQL Injection)
    ✓ CSP header configuration (prevent XSS)
    ✓ CSRF tokens or SameSite=Lax
    ✓ HttpOnly + Secure cookies
    ✓ Enforce HTTPS (HSTS)

  Authentication & Authorization:
    ✓ Password hashing with bcrypt/Argon2
    ✓ MFA (TOTP or WebAuthn)
    ✓ JWT signature verification (ES256 recommended)
    ✓ Principle of least privilege

  Infrastructure:
    ✓ Dependency vulnerability scanning
    ✓ Container image scanning
    ✓ Secret management (never commit .env files)
    ✓ Log retention and audit trails

  OWASP Top 10 (2021):
    A01: Broken Access Control
    A02: Cryptographic Failures
    A03: Injection
    A04: Insecure Design
    A05: Security Misconfiguration
    A06: Vulnerable Components
    A07: Auth Failures
    A08: Software/Data Integrity
    A09: Logging Failures
    A10: SSRF
```

## References

1. OWASP. "Top 10 Web Application Security Risks." owasp.org, 2021.
2. NIST. "Cybersecurity Framework." nist.gov, 2024.
3. Mozilla. "Web Security Guidelines." infosec.mozilla.org, 2024.
