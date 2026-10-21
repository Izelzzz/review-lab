// Small text utilities.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const TEMPLATE_DIR = fileURLToPath(new URL("../templates/", import.meta.url));

/** Return the number of whitespace-separated words in text. */
export function wordCount(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

/** Shorten text to at most `limit` characters, ending with '...' when cut. */
export function truncate(text: string, limit: number): string {
  if (limit < 3) {
    throw new RangeError("limit must be at least 3");
  }
  if (text.length <= limit) {
    return text;
  }
  return text.slice(0, limit - 3) + "...";
}

/**
 * Load templates/<name> and replace each {key} with values[key].
 * Placeholders without a matching key are left as they are.
 */
export function renderTemplate(name: string, values: Record<string, string>): string {
  const template = readFileSync(join(TEMPLATE_DIR, name), "utf8");
  return template.replace(/\{(\w+)\}/g, (match, key) =>
    Object.hasOwn(values, key) ? values[key] : match,
  );
}
