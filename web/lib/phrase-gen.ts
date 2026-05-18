// Phrase-of-the-day fetched from a private GitHub repo.
//
// The schedule lives in a private repo as phrases.md — a simple date-keyed
// list that's easy to edit in the GitHub UI. We fetch it server-side using a
// PAT so the phrase list never appears in the public source or client bundle.
//
// Required env vars:
//   GITHUB_PAT          — fine-grained PAT with read access to the private repo
//   GITHUB_PHRASES_URL  — GitHub API URL to phrases.md, e.g.
//                         https://api.github.com/repos/{owner}/{repo}/contents/phrases.md

import "server-only";

// Parse "YYYY-MM-DD: phrase text" lines, ignoring comments and blank lines.
function parseSchedule(md: string): Map<string, string> {
  const map = new Map<string, string>();
  for (const line of md.split("\n")) {
    const match = line.match(/^(\d{4}-\d{2}-\d{2}):\s*(.+)$/);
    if (match) map.set(match[1].trim(), match[2].trim());
  }
  return map;
}

async function fetchSchedule(): Promise<Map<string, string>> {
  const url = process.env.GITHUB_PHRASES_URL;
  const pat = process.env.GITHUB_PAT;

  if (!url || !pat) {
    throw new Error(
      "Missing GITHUB_PHRASES_URL or GITHUB_PAT environment variables"
    );
  }

  const res = await fetch(url, {
    headers: {
      Authorization: `Bearer ${pat}`,
      Accept: "application/vnd.github.raw+json",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    // Cache for 1 hour — phrase only changes once a day.
    next: { revalidate: 3600 },
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch phrase schedule: ${res.status}`);
  }

  return parseSchedule(await res.text());
}

/**
 * Returns the phrase for a given day offset from today.
 *
 * offset=0 → today, offset=1 → yesterday, offset=2 → two days ago, etc.
 * Falls back to cycling through all listed phrases when the date isn't scheduled.
 */
export async function getDailyPhrase(offset = 0): Promise<string> {
  const schedule = await fetchSchedule();

  const targetMs = Date.now() - offset * 86_400_000;
  const dateStr = new Date(targetMs).toISOString().slice(0, 10); // YYYY-MM-DD UTC
  if (schedule.has(dateStr)) return schedule.get(dateStr)!;

  // Fallback: deterministic cycle through all scheduled phrases.
  const all = [...schedule.values()];
  if (all.length === 0) throw new Error("Phrase schedule is empty");
  const dayIndex = Math.floor(targetMs / 86_400_000);
  return all[dayIndex % all.length];
}
