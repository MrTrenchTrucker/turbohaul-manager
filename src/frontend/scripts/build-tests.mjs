// Bundles every *.test.tsx under src/ into .test-run/*.test.mjs so
// `npm test` (node --test .test-run/*.test.mjs) can run them with Node's
// built-in test runner. No new dependency: esbuild is already vendored (vite
// uses it internally); react/react-dom are marked external and resolved via
// this project's own node_modules at run time, same as any other Node import.
// Output goes to .test-run/ (gitignored) and is removed by the "posttest"
// script — never touches dist/.
import { build } from 'esbuild';
import { readdirSync, statSync, mkdirSync } from 'node:fs';
import { join, relative } from 'node:path';

const SRC = new URL('../src', import.meta.url).pathname;
const OUT_DIR = new URL('../.test-run', import.meta.url).pathname;

function findTestFiles(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) {
      out.push(...findTestFiles(full));
    } else if (/\.test\.tsx?$/.test(name)) {
      out.push(full);
    }
  }
  return out;
}

const testFiles = findTestFiles(SRC);
if (testFiles.length === 0) {
  console.log('No *.test.tsx files found under src/ — nothing to build.');
  process.exit(0);
}

mkdirSync(OUT_DIR, { recursive: true });

for (const file of testFiles) {
  const rel = relative(SRC, file).replace(/\.test\.tsx?$/, '.test.mjs');
  const outfile = join(OUT_DIR, rel.replace(/[\\/]/g, '__'));
  await build({
    entryPoints: [file],
    bundle: true,
    platform: 'node',
    format: 'esm',
    jsx: 'automatic',
    // react-router-dom is external for a specific reason: bundling it pulls in its UMD
    // dev build (dist/umd/react-router-dom.development.js), which does a RUNTIME
    // require('react'). react is external, so that require survives into the ESM output
    // and throws `Dynamic require of "react" is not supported` at module load -- taking
    // the whole test file down before a single assertion runs. Externalising it lets Node
    // resolve the package natively. Without this, no component using a react-router hook or
    // <Link> can be rendered in a test at all.
    external: ['react', 'react-dom', 'react-dom/*', 'react-router-dom'],
    outfile,
  });
  console.log(`built ${relative(SRC, file)} -> ${relative(process.cwd(), outfile)}`);
}
