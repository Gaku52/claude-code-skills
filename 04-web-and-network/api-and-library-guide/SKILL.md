[日本語版](../../ja/04-web-and-network/api-and-library-guide/SKILL.md)

# API and Library Design — Complete Guide

> A systematic guide to designing, implementing, and operating APIs and libraries. Covers REST/GraphQL design, SDK development, versioning, security, documentation, and monitoring — everything you need to know about APIs.

## Target Audience

- Backend engineers involved in API design and development
- Engineers developing SDKs and libraries
- Developers looking to improve API quality and security

## Prerequisites

- HTTP basics — Reference: [Network Fundamentals](../network-fundamentals/)
- Programming basics — Reference: Programming Language Fundamentals

## Guide Index

### 00-api-design-principles (API Design Principles)
| File | Topic | Overview |
|------|-------|----------|
| [00-api-first-design.md](docs/00-api-design-principles/00-api-first-design.md) | API-First Design | API design philosophy, contract-first development, OpenAPI |
| [01-naming-and-conventions.md](docs/00-api-design-principles/01-naming-and-conventions.md) | Naming and Conventions | Endpoint naming, response formats, error design |
| [02-versioning-strategy.md](docs/00-api-design-principles/02-versioning-strategy.md) | Versioning Strategy | URI/header approaches, managing breaking changes |
| [03-pagination-and-filtering.md](docs/00-api-design-principles/03-pagination-and-filtering.md) | Pagination and Filtering | Cursor/offset, sorting, search |

### 01-rest-and-graphql (REST and GraphQL)
| File | Topic | Overview |
|------|-------|----------|
| [00-rest-best-practices.md](docs/01-rest-and-graphql/00-rest-best-practices.md) | REST Best Practices | HATEOAS, idempotency, content negotiation |
| [01-graphql-fundamentals.md](docs/01-rest-and-graphql/01-graphql-fundamentals.md) | GraphQL Fundamentals | Schema, Query/Mutation, resolvers |
| [02-graphql-advanced.md](docs/01-rest-and-graphql/02-graphql-advanced.md) | Advanced GraphQL | Subscriptions, DataLoader, caching |
| [03-rest-vs-graphql.md](docs/01-rest-and-graphql/03-rest-vs-graphql.md) | REST vs GraphQL | Selection criteria, hybrid approaches |

### 02-sdk-and-libraries (SDKs and Libraries)
| File | Topic | Overview |
|------|-------|----------|
| [00-sdk-design.md](docs/02-sdk-and-libraries/00-sdk-design.md) | SDK Design | Client libraries, DX, type safety |
| [01-npm-package-development.md](docs/02-sdk-and-libraries/01-npm-package-development.md) | npm Package Development | package.json, building, publishing |
| [02-api-documentation.md](docs/02-sdk-and-libraries/02-api-documentation.md) | API Documentation | OpenAPI/Swagger, auto-generation, Storybook |

### 03-api-security (API Security)
| File | Topic | Overview |
|------|-------|----------|
| [00-authentication-patterns.md](docs/03-api-security/00-authentication-patterns.md) | Authentication Patterns | OAuth 2.0, API Key, JWT, mTLS |
| [01-rate-limiting.md](docs/03-api-security/01-rate-limiting.md) | Rate Limiting | Token Bucket, Sliding Window, distributed rate limiting |
| [02-input-validation.md](docs/03-api-security/02-input-validation.md) | Input Validation | Zod, JSON Schema, sanitization |

### 04-api-operations (API Operations)
| File | Topic | Overview |
|------|-------|----------|
| [00-api-testing.md](docs/04-api-operations/00-api-testing.md) | API Testing | Integration testing, contract testing, load testing |
| [01-monitoring-and-logging.md](docs/04-api-operations/01-monitoring-and-logging.md) | Monitoring and Logging | Error rates, latency, distributed tracing |
| [02-api-gateway.md](docs/04-api-operations/02-api-gateway.md) | API Gateway | Kong, AWS API Gateway, centralized auth/rate limiting |

## Learning Path

```
Design:       00-api-design-principles
Implementation: 01-rest-and-graphql → 02-sdk-and-libraries
Security:     03-api-security
Operations:   04-api-operations
```

## FAQ

### Q1: Should I choose REST API or GraphQL?
REST API is well-suited for resource-based CRUD operations, with well-established caching strategies as a key strength. GraphQL excels at complex data fetching and frontend-driven development, solving the over-fetching and under-fetching problems. A hybrid approach — REST for public APIs (external partners) and GraphQL for internal BFF (Backend for Frontend) — works well for many projects.

### Q2: When should I start planning API versioning?
From the very first design phase. Retrofitting a versioning strategy is difficult and disruptive to existing clients. The URI path approach (/api/v1/) is the simplest and most widely adopted. When upgrading versions, plan for a minimum of 12 months of parallel operation, and communicate deprecation notices proactively.

### Q3: What is the most important aspect of SDK development?
Developer experience (DX) should be the top priority. Specifically, four things matter most: type safety (providing TypeScript type definitions), an intuitive API interface (resource-based patterns), informative error messages (actionable information that tells the developer what to do next), and comprehensive documentation (with code examples). The Stripe and Twilio SDKs serve as excellent design references.

## Summary

This guide covers the following topics:

- The philosophy of API-First design and contract-first development using the OpenAPI specification
- REST API best practices (HATEOAS, idempotency, error handling) and GraphQL from fundamentals to advanced usage
- SDK and library design principles (DX-first, type safety, retry strategies) and the npm package publishing workflow
- API security implementation patterns (OAuth 2.0, rate limiting, input validation)
- Test strategies, monitoring and logging, and API gateway usage for API operations

## Related Skills

- [Network Fundamentals](../network-fundamentals/) — Network Fundamentals
- [Security Fundamentals](../../06-data-and-security/security-fundamentals/) — Security Fundamentals
- [Web Application Development](../web-application-development/) — Web Application Development

## References

- [OpenAPI Specification](https://spec.openapis.org/oas/latest.html) - The industry standard for API specification. The foundation for contract-first development and code generation
- [Stripe API Reference](https://docs.stripe.com/api) - Widely referenced as the industry benchmark for REST API design and SDK design
- [Google API Design Guide](https://cloud.google.com/apis/design) - A distillation of design principles and best practices from Google's large-scale API ecosystem
