[日本語版](../../ja/04-web-and-network/web-application-development/SKILL.md)

# Web Application Development — Complete Guide

> From design to production deployment. A systematic guide to modern web application development covering architecture selection, state management, routing, form design, authentication integration, and deployment strategies.

## Target Audience

- Full-stack engineers involved in web application design and development
- Developers evaluating frontend framework options
- Developers building web applications with production readiness in mind

## Prerequisites

- HTML/CSS/JavaScript basics
- React basics — Reference: Programming Language Fundamentals

## Guide Index

### 00-architecture (Architecture)
| File | Topic | Overview |
|------|-------|----------|
| [00-spa-mpa-ssr.md](docs/00-architecture/00-spa-mpa-ssr.md) | SPA/MPA/SSR | Comparison of rendering strategies and selection criteria |
| [01-project-structure.md](docs/00-architecture/01-project-structure.md) | Project Structure | Directory design, module organization |
| [02-component-architecture.md](docs/00-architecture/02-component-architecture.md) | Component Architecture | Atomic Design, Container/Presentational |
| [03-data-fetching-patterns.md](docs/00-architecture/03-data-fetching-patterns.md) | Data Fetching | SWR, TanStack Query, Server Components |

### 01-state-management (State Management)
| File | Topic | Overview |
|------|-------|----------|
| [00-state-management-overview.md](docs/01-state-management/00-state-management-overview.md) | State Management Overview | Categorizing local, global, and server state |
| [01-zustand-and-jotai.md](docs/01-state-management/01-zustand-and-jotai.md) | Zustand / Jotai | Choosing between lightweight state management libraries |
| [02-server-state.md](docs/01-state-management/02-server-state.md) | Server State | Caching strategies with TanStack Query and SWR |
| [03-url-state.md](docs/01-state-management/03-url-state.md) | URL State | Search parameters, deep linking |

### 02-routing-and-navigation (Routing)
| File | Topic | Overview |
|------|-------|----------|
| [00-client-side-routing.md](docs/02-routing-and-navigation/00-client-side-routing.md) | Client-Side Routing | React Router, TanStack Router |
| [01-file-based-routing.md](docs/02-routing-and-navigation/01-file-based-routing.md) | File-Based Routing | Next.js App Router, Remix |
| [02-navigation-patterns.md](docs/02-routing-and-navigation/02-navigation-patterns.md) | Navigation Design | Breadcrumbs, tabs, sidebars |
| [03-auth-and-guards.md](docs/02-routing-and-navigation/03-auth-and-guards.md) | Auth Guards | Route protection, redirects |

### 03-forms-and-validation (Forms)
| File | Topic | Overview |
|------|-------|----------|
| [00-form-design.md](docs/03-forms-and-validation/00-form-design.md) | Form Design | React Hook Form, controlled/uncontrolled components |
| [01-validation-patterns.md](docs/03-forms-and-validation/01-validation-patterns.md) | Validation | Zod integration, real-time validation |
| [02-file-upload.md](docs/03-forms-and-validation/02-file-upload.md) | File Upload | Drag & drop, progress indicators, direct S3 upload |
| [03-complex-forms.md](docs/03-forms-and-validation/03-complex-forms.md) | Complex Forms | Multi-step forms, dynamic forms, conditional logic |

### 04-deployment (Deployment)
| File | Topic | Overview |
|------|-------|----------|
| [00-deployment-platforms.md](docs/04-deployment/00-deployment-platforms.md) | Deployment Platforms | Vercel, Cloudflare, AWS, Docker |
| [01-environment-and-config.md](docs/04-deployment/01-environment-and-config.md) | Environment Configuration | Environment variables, feature flags |
| [02-performance-optimization.md](docs/04-deployment/02-performance-optimization.md) | Performance | Bundle optimization, image optimization, CDN |
| [03-monitoring-and-error-tracking.md](docs/04-deployment/03-monitoring-and-error-tracking.md) | Monitoring | Sentry, Web Vitals, logging |

## Learning Path

```
Architecture:        00-architecture
State management:    01-state-management
Routing:             02-routing-and-navigation
Forms:               03-forms-and-validation
Deployment:          04-deployment
```

## FAQ

### Q1: Is Next.js the only choice? Are there other options?
Next.js is the most popular React framework, but it is far from the only option. Remix takes an approach closer to web standards and is well-suited for projects that prioritize progressive enhancement. Astro is ideal for content-centric sites, using Islands Architecture to build fast pages with minimal JavaScript. In the Vue.js ecosystem, Nuxt.js is the corresponding framework, while SvelteKit serves the same role for Svelte. Choose based on your project's requirements — SSR/SSG needs, SEO requirements, and your team's skill set.

### Q2: Should I adopt a feature-based structure even for small projects?
For small projects with fewer than 50 files, a technical-based (type-based) structure works fine. However, if the project is expected to grow, adopting a feature-based structure from the start is recommended. Migrating later requires changing import paths and updating tests, making the cost significantly higher. A practical approach is to start with just two top-level directories — `features/` and `shared/` — and add features as needed.

### Q3: How should I choose a state management library?
Start by asking whether you truly need global state management. In many cases, managing server state with TanStack Query or SWR and URL state with useSearchParams eliminates the need for a global state management library altogether. If global state is still required, choose Zustand for simplicity or Jotai when you need fine-grained re-render control. Redux has a strong track record in large teams but is declining in adoption for new projects due to its boilerplate overhead.

## Summary

This guide covers the following topics:

- Characteristics and selection criteria for rendering strategies including SPA, MPA, and SSR
- Feature-based project structure and component design patterns
- Client state management with Zustand/Jotai and server state management with TanStack Query
- File-based routing centered on Next.js App Router, along with navigation design
- Form design and validation patterns using React Hook Form and Zod
- Deployment strategies to Vercel, Cloudflare, and AWS, plus performance optimization

## References

1. Next.js. "Documentation." nextjs.org/docs, 2024.
2. React. "React Documentation." react.dev, 2024.
3. TanStack. "TanStack Query Documentation." tanstack.com, 2024.
4. Zustand. "Bear necessities for state management." github.com/pmndrs/zustand, 2024.
5. Vercel. "Deployment Documentation." vercel.com/docs, 2024.

## Related Skills

- [Browser and Web Platform](../browser-and-web-platform/) — Browser and Web Platform
- [Network Fundamentals](../network-fundamentals/) — Network Fundamentals
- [API and Library Guide](../api-and-library-guide/) — API and Library Design
