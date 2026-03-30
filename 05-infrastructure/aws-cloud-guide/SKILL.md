[日本語版](../../ja/05-infrastructure/aws-cloud-guide/SKILL.md)

# AWS Cloud Guide

> AWS is the de facto standard in cloud computing. This guide systematically covers everything from the basics of EC2, S3, and Lambda to network design, database selection, serverless architecture, container operations, security, and cost optimization.

## Target Audience

- Engineers looking to learn AWS infrastructure
- Professionals pursuing AWS certifications (SAA/SAP)
- Teams planning migrations from on-premises to the cloud

## Prerequisites

- Basic Linux operations
- Networking fundamentals (TCP/IP, DNS, HTTP)
- Basic container knowledge (Docker)

## Study Guide

### 00-fundamentals — AWS Fundamentals

| # | File | Description |
|---|------|-------------|

### 01-compute — Compute

| # | File | Description |
|---|------|-------------|

### 02-storage — Storage

| # | File | Description |
|---|------|-------------|

### 03-database — Database

| # | File | Description |
|---|------|-------------|

### 04-networking — Networking

| # | File | Description |
|---|------|-------------|

### 05-serverless — Serverless

| # | File | Description |
|---|------|-------------|

### 06-containers — Container Services

| # | File | Description |
|---|------|-------------|

### 07-devops — DevOps Services

| # | File | Description |
|---|------|-------------|

### 08-security — Security Services

| # | File | Description |
|---|------|-------------|

### 09-cost-management — Cost Management

| # | File | Description |
|---|------|-------------|

## Quick Reference

```
AWS Service Selection Chart:

  Compute:
    Containers → ECS Fargate (recommended) or EKS
    Serverless → Lambda + API Gateway
    VMs → EC2 + Auto Scaling
    PaaS → Elastic Beanstalk

  Database:
    Relational → Aurora (recommended) or RDS
    NoSQL → DynamoDB
    Cache → ElastiCache Redis
    Full-text search → OpenSearch

  Storage:
    Object → S3
    File → EFS
    Block → EBS

  Networking:
    DNS → Route 53
    CDN → CloudFront
    Load Balancer → ALB

  Cost Reduction:
    ✓ Savings Plans / Reserved Instances
    ✓ Spot Instances (for fault-tolerant workloads)
    ✓ S3 Lifecycle policies (transition to Glacier)
    ✓ Lambda (for low-traffic periods)
```

## References

1. AWS. "Documentation." docs.aws.amazon.com, 2024.
2. AWS. "Well-Architected Framework." aws.amazon.com/architecture, 2024.
3. AWS. "Pricing Calculator." calculator.aws, 2024.
