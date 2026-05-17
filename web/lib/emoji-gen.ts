// Emoji generation, served by the trained Qwen2.5-3B LoRA policy on Modal.
// See scripts/serve_model.py for the FastAPI app.
//
// Override the URL in production via the MODAL_EMOJI_ENDPOINT env var.

const DEFAULT_ENDPOINT = "https://2c-nyc--emo-serve-web.modal.run";

export type HistoryTurn = { emoji: string; guess: string };

export type EmojiResponse = { emoji: string; checkpoint: string };

const FALLBACK_EMOJI = "<UNK>";

export async function generateEmoji(
  phrase: string,
  history: HistoryTurn[],
  signal?: AbortSignal,
): Promise<string> {
  const base = process.env.MODAL_EMOJI_ENDPOINT ?? DEFAULT_ENDPOINT;
  const url = `${base.replace(/\/$/, "")}/emoji`;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phrase, history }),
      signal,
    });
    if (!res.ok) {
      console.error(`[emoji-gen] ${res.status} from ${url}`);
      return FALLBACK_EMOJI;
    }
    const data = (await res.json()) as EmojiResponse;
    return data.emoji?.trim() || FALLBACK_EMOJI;
  } catch (err) {
    console.error("[emoji-gen] fetch failed:", err);
    return FALLBACK_EMOJI;
  }
}
