[日本語版](../../ja/05-infrastructure/docker-container-guide/SKILL.md)

# Docker Container Guide

> Docker is a foundational piece of modern development infrastructure. This guide systematically covers container fundamentals, Dockerfile best practices, multi-service orchestration with Docker Compose, networking, production operations, orchestration, and security.

## Target Audience

- Engineers looking to learn Docker-based development and operations
- Developers deploying containerized applications to production
- Teams building multi-service environments with Docker Compose

## Prerequisites

- Basic Linux commands
- Fundamental understanding of web application architecture
- Basic networking knowledge

## Study Guide

### 00-fundamentals — Container Fundamentals

| # | File | Description |
|---|------|-------------|

### 01-dockerfile — Dockerfile Best Practices

| # | File | Description |
|---|------|-------------|

### 02-compose — Docker Compose

| # | File | Description |
|---|------|-------------|

### 03-networking — Networking

| # | File | Description |
|---|------|-------------|

### 04-production — Production Operations

| # | File | Description |
|---|------|-------------|

### 05-orchestration — Orchestration

| # | File | Description |
|---|------|-------------|

### 06-security — Security

| # | File | Description |
|---|------|-------------|

## Quick Reference

```
Docker Command Cheat Sheet:
  docker build -t app:latest .        — Build an image
  docker run -d -p 3000:3000 app      — Start a container
  docker compose up -d                — Start Compose services
  docker compose down -v              — Stop Compose and remove volumes
  docker logs -f <container>          — Follow container logs
  docker exec -it <container> sh      — Open a shell inside a container
  docker system prune -a              — Remove unused resources

Dockerfile Best Practices:
  ✓ Use multi-stage builds to reduce image size
  ✓ Run as a non-root user
  ✓ Optimize build context with .dockerignore
  ✓ Set file ownership with COPY --chown
  ✓ Configure health checks (HEALTHCHECK)
  ✓ Pin base image versions
```

## References

1. Docker. "Documentation." docs.docker.com, 2024.
2. Docker. "Dockerfile Best Practices." docs.docker.com, 2024.
3. Kubernetes. "Documentation." kubernetes.io/docs, 2024.
