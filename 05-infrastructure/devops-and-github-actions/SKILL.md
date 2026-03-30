[日本語版](../../ja/05-infrastructure/devops-and-github-actions/SKILL.md)

# DevOps and GitHub Actions

> DevOps automates the build, test, and deploy lifecycle, breaking down the wall between development and operations. This guide covers CI/CD fundamentals, hands-on GitHub Actions, deployment strategies, and monitoring/alerting -- providing a complete picture of modern DevOps.

## Target Audience

- Engineers looking to build CI/CD pipelines
- Developers who want to fully leverage GitHub Actions
- Teams aiming to establish deployment automation and monitoring infrastructure

## Prerequisites

- Basic Git operations
- Fundamental Docker knowledge
- YAML syntax

## Study Guide

### 00-devops-basics — DevOps Fundamentals

| # | File | Description |
|---|------|-------------|

### 01-github-actions — GitHub Actions

| # | File | Description |
|---|------|-------------|

### 02-deployment — Deployment

| # | File | Description |
|---|------|-------------|

### 03-monitoring — Monitoring and Observability

| # | File | Description |
|---|------|-------------|

## Quick Reference

```
GitHub Actions Syntax Cheat Sheet:

  Triggers:
    on: push / pull_request / schedule / workflow_dispatch
    branches: [main] / paths: ['src/**']

  Jobs:
    runs-on: ubuntu-latest
    strategy: matrix (node-version: [18, 20, 22])
    needs: [build, test]

  Commonly Used Actions:
    actions/checkout@v4
    actions/setup-node@v4
    actions/cache@v4
    docker/build-push-action@v5
    aws-actions/configure-aws-credentials@v4

  Deployment Strategy Selection:
    Low risk, small scale → Rolling Update
    Medium risk → Blue-Green
    High risk, large scale → Canary
    Experimental features → Feature Flag
```

## References

1. GitHub. "Actions Documentation." docs.github.com/actions, 2024.
2. Forsgren, N. et al. "Accelerate." IT Revolution Press, 2018.
3. HashiCorp. "Terraform Documentation." terraform.io/docs, 2024.
