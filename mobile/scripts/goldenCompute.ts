/**
 * CLI: read fused numeric rows JSON (array of objects with base columns), write window features JSON.
 * Used by `scripts/golden_row_window_parity.py` for Python vs TypeScript parity.
 *
 * Usage: npx tsx scripts/goldenCompute.ts <fused_rows.json> <out_features.json>
 */
import * as fs from 'node:fs';

import { BASE_NUMERIC_COLUMNS, type FusedNumericRow } from '../src/baseNumericColumns';
import { buildWindowFeatureRecord } from '../src/rowWindows';

function rowFromJson(o: Record<string, unknown>): FusedNumericRow {
  const out = {} as FusedNumericRow;
  for (const k of BASE_NUMERIC_COLUMNS) {
    const v = o[k];
    if (v === null || v === undefined) {
      out[k] = Number.NaN;
    } else {
      out[k] = Number(v);
    }
  }
  return out;
}

function main(): void {
  const [, , inPath, outPath] = process.argv;
  if (!inPath || !outPath) {
    console.error('Usage: tsx scripts/goldenCompute.ts <fused_rows.json> <out_features.json>');
    process.exit(1);
  }
  const raw = JSON.parse(fs.readFileSync(inPath, 'utf8')) as Record<string, unknown>[];
  const rows: FusedNumericRow[] = raw.map((r) => rowFromJson(r));
  const ws = rows.length;
  const feats = buildWindowFeatureRecord(rows, ws);
  if (!feats) {
    console.error('buildWindowFeatureRecord returned null');
    process.exit(1);
  }
  fs.writeFileSync(outPath, `${JSON.stringify(feats, null, 2)}\n`, 'utf8');
}

main();
