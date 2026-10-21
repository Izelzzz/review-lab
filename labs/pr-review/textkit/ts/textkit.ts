// Small text utilities.

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

/** Return the average length of the words in text. */
export function meanWordLength(text: string): number {
  const words = text.split(/\s+/).filter(Boolean);
  const total = words.reduce((sum, w) => sum + [...w].length, 0);
  return total / words.length;
}
