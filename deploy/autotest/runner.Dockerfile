FROM node:20-bookworm-slim

ARG PLAYWRIGHT_VERSION=1.52.0
ENV PLAYWRIGHT_VERSION=${PLAYWRIGHT_VERSION}
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV CI=1

WORKDIR /opt/ha-entity-vault-autotest

COPY deploy/autotest/package.json ./package.json
COPY deploy/autotest/scripts ./scripts

RUN node -e "const fs=require('node:fs'); const pkg=JSON.parse(fs.readFileSync('package.json','utf8')); const pinned=pkg.dependencies.playwright; const desired=process.env.PLAYWRIGHT_VERSION; if (pinned !== desired) { throw new Error('PLAYWRIGHT_VERSION (' + desired + ') must match package.json dependency (' + pinned + ')'); }" && \
    npm install --omit=dev && \
    npx -y playwright@${PLAYWRIGHT_VERSION} install --with-deps chromium && \
    node ./scripts/playwright-preflight.mjs

ENTRYPOINT ["node", "scripts/playwright-preflight.mjs"]
