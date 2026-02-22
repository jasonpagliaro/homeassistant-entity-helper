# Changelog

All notable changes to this project are documented in this file.

## 2026-02-22

### Changed
- Cleaned up primary page switching in the shared header using accessible tab-style navigation (semantic nav/list links with a clear active state and `aria-current="page"`).
- Added a single source-of-truth primary navigation model in app context and active-path matching for both top-level and detail routes.
- Moved the API Docs link out of the primary header tabs into a global footer utility link.
- Added scoped navigation/footer styling, including visible keyboard focus and mobile horizontal scrolling for tabs.
- Added reusable custom hover/focus tooltips for high-impact mutating and long-running actions using `data-tooltip` across settings, sync/suggestion workflows, and draft/proposal review actions.

### Tests
- Added API tests covering active-tab behavior across all six top-level pages.
- Added assertions that API Docs appears in the footer (and not in primary navigation).
- Added a detail-route active-tab assertion for entity detail pages.
- Added template contract coverage to enforce `data-tooltip` presence on selected high-impact buttons, including duplicate-label button count checks.

### Reference
- Commit: `85f5e5e` ("Clean up header navigation tabs and move docs link to footer")
- Commit: `79a096d` ("Add custom tooltips for high-impact actions (#7)")
