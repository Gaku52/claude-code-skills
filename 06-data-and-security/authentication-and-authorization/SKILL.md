[日本語版](../../ja/06-data-and-security/authentication-and-authorization/SKILL.md)

# Authentication and Authorization

> Authentication and authorization are the cornerstones of web application security. This skill systematically covers everything about secure access control — from password management, sessions, JWT, OAuth 2.0, OpenID Connect, RBAC/ABAC, and multi-factor authentication to practical NextAuth.js implementation.

## Target Audience

- Engineers implementing authentication in web applications
- Developers who want to learn security-conscious design and implementation
- Those seeking a deep understanding of OAuth 2.0 / OIDC
- Those designing permission management with RBAC/ABAC

## Prerequisites

- HTTP fundamentals (headers, cookies, status codes)
- Basic JavaScript / TypeScript knowledge
- Understanding of web application architecture (frontend / backend)

## Study Guide

### 00-fundamentals — Authentication and Authorization Basics

| # | File | Description |
|---|------|-------------|

### 01-session-auth — Session-Based Authentication

| # | File | Description |
|---|------|-------------|

### 02-token-auth — Token-Based Authentication

| # | File | Description |
|---|------|-------------|

### 03-authorization — Authorization Design

| # | File | Description |
|---|------|-------------|

### 04-implementation — Implementation Patterns

| # | File | Description |
|---|------|-------------|

## Quick Reference

```
Choosing an Authentication Strategy:
  Personal projects / Small-scale → NextAuth.js + social login
  B2C services → OAuth 2.0 + PKCE + email verification
  B2B SaaS → OIDC + SAML SSO + RBAC
  API services → API Key + OAuth 2.0 Client Credentials
  Mobile apps → OAuth 2.0 + PKCE + Refresh Token Rotation

Security Checklist:
  ✓ Hash passwords with bcrypt/Argon2
  ✓ Sign JWTs with RS256/ES256
  ✓ Set cookies to HttpOnly + Secure + SameSite=Lax
  ✓ Implement CSRF tokens
  ✓ Use Refresh Token rotation with revocation detection
  ✓ Apply rate limiting to login endpoints
  ✓ Require MFA for sensitive operations
```

## References

1. OWASP. "Authentication Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. RFC 6749. "The OAuth 2.0 Authorization Framework." IETF, 2012.
3. RFC 7519. "JSON Web Token (JWT)." IETF, 2015.
4. OpenID Foundation. "OpenID Connect Core 1.0." openid.net, 2014.
5. Auth.js. "Documentation." authjs.dev, 2024.
