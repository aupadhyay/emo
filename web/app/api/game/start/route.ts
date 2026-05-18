import { encryptState } from "@/lib/game-state";
import { generateEmoji } from "@/lib/emoji-gen";
import { getDailyPhrase } from "@/lib/phrase-gen";

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => ({}));
    const offset = typeof body.offset === "number" ? Math.max(0, Math.floor(body.offset)) : 0;

    const phrase = await getDailyPhrase(offset);
    const emoji = await generateEmoji(phrase, []);

    const token = encryptState({
      phrase,
      turns: [{ emoji, guess: null, verdict: null }],
    });

    return Response.json({ token, emoji, turn: 1 });
  } catch (err) {
    console.error("[/api/game/start]", err);
    return Response.json({ error: "Failed to start game" }, { status: 500 });
  }
}
