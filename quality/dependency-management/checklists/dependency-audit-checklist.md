# Dependency Audit Checklist

## Before Adding a New Dependency

### Research Phase
- [ ] Check if functionality can be implemented natively
- [ ] Search for existing similar dependencies in project
- [ ] Compare alternative packages (bundlephobia.com)
- [ ] Review package documentation quality
- [ ] Check TypeScript support availability

### Package Evaluation
- [ ] Verify active maintenance (commits within 6 months)
- [ ] Check GitHub stars and weekly downloads
- [ ] Review open issues count and response time
- [ ] Verify package has tests and CI/CD
- [ ] Check for security advisories (Snyk, npm audit)

### License Compliance
- [ ] Verify license is compatible with project
- [ ] Confirm commercial use is allowed
- [ ] Check if attribution is required
- [ ] Review dependencies' licenses recursively

### Bundle Impact
- [ ] Check package size on bundlephobia.com
- [ ] Verify tree-shaking support (ES modules)
- [ ] Compare minified + gzipped size
- [ ] Check for optional peer dependencies

### Security Assessment
- [ ] No known CVEs in current version
- [ ] Package published from verified account
- [ ] No suspicious install scripts
- [ ] Review recent version history for anomalies

### Team Review
- [ ] Discuss with team if significant dependency
- [ ] Document decision in ADR (Architecture Decision Record)
- [ ] Add to approved dependencies list

## Weekly Dependency Maintenance

### Security
- [ ] Review Dependabot/Renovate security alerts
- [ ] Run `npm audit` (or equivalent)
- [ ] Check for zero-day vulnerabilities
- [ ] Review dependency-track dashboard

### Updates
- [ ] Check for patch version updates
- [ ] Review and merge automated update PRs
- [ ] Test critical paths after updates
- [ ] Update lockfile if needed

### Health Monitoring
- [ ] Check for deprecated package warnings
- [ ] Review dependency age report
- [ ] Monitor bundle size changes
- [ ] Check build time metrics

## Monthly Dependency Review

### Analysis
- [ ] Run `npm outdated` (or equivalent)
- [ ] Generate dependency report
- [ ] Identify packages > 2 minor versions behind
- [ ] List packages with major version updates available

### Cleanup
- [ ] Run `npx depcheck` to find unused dependencies
- [ ] Remove confirmed unused packages
- [ ] Consolidate duplicate dependencies (`npm dedupe`)
- [ ] Review and remove deprecated packages

### Documentation
- [ ] Update DEPENDENCIES.md if exists
- [ ] Document any pinned versions with reasons
- [ ] Update team wiki with dependency decisions
- [ ] Generate SBOM (Software Bill of Materials)

### Metrics
- [ ] Record total dependency count
- [ ] Track bundle size over time
- [ ] Monitor build time trends
- [ ] Count security vulnerabilities fixed

## Quarterly Dependency Strategy

### Major Updates
- [ ] List all available major version updates
- [ ] Read migration guides for each
- [ ] Prioritize updates by impact/benefit
- [ ] Create update roadmap

### Technical Debt
- [ ] Review dependency debt dashboard
- [ ] Identify packages to replace or remove
- [ ] Plan monorepo migration if beneficial
- [ ] Evaluate build tool modernization

### Tooling
- [ ] Review and update Dependabot/Renovate config
- [ ] Evaluate new security scanning tools
- [ ] Test beta features of package managers
- [ ] Update CI/CD dependency caching

### Team
- [ ] Conduct dependency management training
- [ ] Review and update dependency policy
- [ ] Share lessons learned from incidents
- [ ] Update runbooks and documentation

## Pre-Release Checklist

### Dependency Lock
- [ ] All dependencies at stable versions
- [ ] No pre-release versions in production deps
- [ ] Lock file is up-to-date
- [ ] No security vulnerabilities

### Documentation
- [ ] SBOM generated and reviewed
- [ ] License compliance verified
- [ ] Third-party attributions updated
- [ ] Breaking changes documented

### Testing
- [ ] All tests pass with current dependencies
- [ ] Production build succeeds
- [ ] Bundle size within budget
- [ ] Performance benchmarks pass

### Compliance
- [ ] Security scan completed
- [ ] License scan completed
- [ ] No GPL/AGPL dependencies (if proprietary)
- [ ] Export compliance reviewed (if applicable)

## Incident Response Checklist

### Immediate (0-1 hour)
- [ ] Identify affected dependency and versions
- [ ] Check if vulnerability is exploitable in our usage
- [ ] Determine severity (P0-P3)
- [ ] Notify security team and stakeholders
- [ ] Create incident ticket

### Containment (1-4 hours)
- [ ] Update to patched version if available
- [ ] Apply temporary workaround if no patch
- [ ] Deploy hotfix to staging
- [ ] Verify fix effectiveness
- [ ] Prepare production deployment

### Remediation (4-24 hours)
- [ ] Deploy fix to production
- [ ] Monitor for exploitation attempts
- [ ] Verify no residual issues
- [ ] Update dependency policies if needed

### Post-Incident (24+ hours)
- [ ] Write incident report
- [ ] Conduct post-mortem
- [ ] Document lessons learned
- [ ] Update security procedures
- [ ] Share knowledge with team

## Migration Checklist (Major Version Updates)

### Preparation
- [ ] Read official migration guide
- [ ] Review changelog for breaking changes
- [ ] Check GitHub issues for known problems
- [ ] Create migration branch
- [ ] Notify team of upcoming changes

### Analysis
- [ ] Identify all affected code (grep/search)
- [ ] List required code changes
- [ ] Estimate effort and timeline
- [ ] Plan rollback strategy

### Implementation
- [ ] Update package.json
- [ ] Fix TypeScript errors
- [ ] Fix linting errors
- [ ] Update tests
- [ ] Update documentation

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E tests pass
- [ ] Manual testing completed
- [ ] Performance testing done
- [ ] Accessibility testing done

### Deployment
- [ ] Deploy to dev environment
- [ ] Deploy to staging environment
- [ ] Smoke tests on staging
- [ ] Monitor staging for 24h
- [ ] Deploy to production (canary/blue-green)
- [ ] Monitor production for 48h

### Finalization
- [ ] Update internal documentation
- [ ] Share learnings with team
- [ ] Close migration ticket
- [ ] Archive old version documentation

## Dependency Health Score

Calculate your project's dependency health:

### Scoring Criteria
- **Security**: No critical/high vulnerabilities (25 points)
- **Freshness**: < 5% packages outdated (20 points)
- **Size**: Bundle size < 500KB (15 points)
- **Count**: < 200 total dependencies (10 points)
- **Build Time**: < 2 minutes (10 points)
- **Documentation**: All decisions documented (10 points)
- **Automation**: Auto-updates configured (10 points)

### Grade Scale
- 90-100: Excellent (🟢)
- 75-89: Good (🟡)
- 60-74: Fair (🟠)
- < 60: Poor (🔴)

**Target: Maintain 80+ score**

## Emergency Contact List

Update with your team's information:

- **Security Team**: security@example.com
- **Tech Lead**: tech-lead@example.com
- **DevOps On-Call**: oncall@example.com
- **Vendor Support**: Snyk, GitHub, etc.

## Useful Commands Reference

```bash
# npm
npm audit
npm outdated
npm update
npx depcheck

# Yarn
yarn audit
yarn outdated
yarn upgrade

# pnpm
pnpm audit
pnpm outdated
pnpm update

# Security scanning
npx snyk test
npx socket-cli audit

# Bundle analysis
npx webpack-bundle-analyzer dist/stats.json
npx source-map-explorer dist/*.js

# SBOM generation
npx @cyclonedx/cyclonedx-npm --output-file sbom.json
```
