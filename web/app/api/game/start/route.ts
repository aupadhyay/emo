import Anthropic from "@anthropic-ai/sdk";
import { randomPhrase } from "@/lib/phrases";
import { encryptState } from "@/lib/game-state";

const client = new Anthropic();

const SYSTEM = `You are playing emoji Charades. You are given a secret phrase and must communicate it using only emoji.

Turn 1: Give emoji representing the most guessable aspect of the phrase. Don't try to convey everything — pick the strongest, most recognizable clue.

Rules:
- Output ONLY emoji characters. No text, no punctuation, no spaces.
- Be concrete. Pick emoji a typical person would associate with the concept.
- Prioritize what's guessable over what's literally accurate.`;

export async function POST() {
  try {
    const phrase = randomPhrase();

    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 50,
      system: SYSTEM,
      messages: [{ role: "user", content: `The phrase is: ${phrase.text}` }],
    });

    const emoji =
      response.content[0].type === "text"
        ? response.content[0].text.trim()
        : "❓";

    const token = encryptState({
      phrase: phrase.text,
      tier: phrase.tier,
      turns: [{ emoji, guess: null, verdict: null }],
    });

    return Response.json({ token, emoji, turn: 1, tier: phrase.tier });
  } catch (err) {
    console.error("[/api/game/start]", err);
    return Response.json({ error: "Failed to start game" }, { status: 500 });
  }
}
