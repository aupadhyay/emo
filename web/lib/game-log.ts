// Append finished-game records to the Modal logging endpoint. This is the
// signal pipeline that the next round of RL training will consume — every
// completed game is one (phrase, emoji_sequence, human_guesses, verdict) row.

import type { GameState, Verdict } from "@/lib/game-state";

const DEFAULT_ENDPOINT = "https://2c-nyc--emo-serve-web.modal.run";

export async function logGame(
  state: GameState,
  outcome: Verdict,
  score: number,
): Promise<void> {
  const base = process.env.MODAL_EMOJI_ENDPOINT ?? DEFAULT_ENDPOINT;
  const url = `${base.replace(/\/$/, "")}/log`;

  const body = {
    phrase: state.phrase,
    outcome,
    score,
    turns: state.turns.map((t) => ({
      emoji: t.emoji,
      guess: t.guess,
      verdict: t.verdict,
    })),
  };

  try {
    // Fire-and-forget: we don't want a slow log call to delay the response.
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      console.error(`[game-log] ${res.status} from ${url}`);
    }
  } catch (err) {
    console.error("[game-log] fetch failed:", err);
  }
}
