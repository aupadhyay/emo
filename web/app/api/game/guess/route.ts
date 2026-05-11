import Anthropic from "@anthropic-ai/sdk";
import {
  decryptState,
  encryptState,
  currentTurnIndex,
  type GameState,
  type Verdict,
} from "@/lib/game-state";
import { calcScore } from "@/lib/phrases";

const client = new Anthropic();

const MAX_TURNS = 3;

const EMOJI_SYSTEM = `You are playing emoji Charades. You are given a secret phrase and must communicate it using only emoji across up to 3 turns.

When given a wrong guess, approach the concept from a completely different angle. Think about what the player's wrong guess reveals about their thinking, then choose emoji that highlight what they're missing — don't just repeat or extend your previous emoji.

Rules:
- Output ONLY emoji characters. No text, no punctuation, no spaces.
- Avoid simply repeating or extending your previous emoji — give a fresh perspective unless there's a strong reason to reuse one.
- Be concrete. Pick emoji a typical person would associate with the concept.
- Prioritize what's guessable over what's literally accurate.`;

async function judge(phrase: string, guess: string): Promise<Verdict> {
  const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    system: `You are a judge in an emoji guessing game. Respond with JSON only: {"verdict":"correct"|"partial"|"wrong"}`,
    messages: [
      {
        role: "user",
        content: `Secret phrase: "${phrase}"
Player guessed: "${guess}"

- "correct": matches the phrase in meaning (exact wording not required)
- "partial": captures part of the meaning or clearly heading in the right direction
- "wrong": doesn't match`,
      },
    ],
  });

  try {
    const text =
      response.content[0].type === "text" ? response.content[0].text : "{}";
    const parsed = JSON.parse(text);
    if (["correct", "partial", "wrong"].includes(parsed.verdict)) {
      return parsed.verdict as Verdict;
    }
  } catch {
    // fall through
  }
  return "wrong";
}

function buildEmojiMessages(
  state: GameState
): Anthropic.MessageParam[] {
  // Reconstruct the conversation history so Claude can generate contextual follow-up emoji
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: `The phrase is: ${state.phrase}` },
  ];

  const completedTurns = state.turns.filter((t) => t.guess !== null);

  for (let i = 0; i < completedTurns.length; i++) {
    const turn = completedTurns[i];
    messages.push({ role: "assistant", content: turn.emoji });

    const isLast = i === completedTurns.length - 1;
    const nextTurnNum = completedTurns.length + 1;
    const finalSuffix =
      nextTurnNum >= MAX_TURNS ? " This is your final turn." : "";

    messages.push({
      role: "user",
      content:
        turn.verdict === "partial"
          ? `The player guessed: '${turn.guess}'. Getting closer, but not quite right. Give 1-3 more emoji.${isLast ? finalSuffix : ""}`
          : `The player guessed: '${turn.guess}'. Wrong. Give 1-3 more emoji to help narrow it down.${isLast ? finalSuffix : ""}`,
    });
  }

  return messages;
}

async function nextEmoji(state: GameState): Promise<string> {
  const messages = buildEmojiMessages(state);
  const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 50,
    system: EMOJI_SYSTEM,
    messages,
  });
  return response.content[0].type === "text"
    ? response.content[0].text.trim()
    : "❓";
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { token?: string; guess?: string };
    const { token, guess } = body;

    if (!token || !guess?.trim()) {
      return Response.json(
        { error: "Missing token or guess" },
        { status: 400 }
      );
    }

    const state = decryptState(token);
    if (!state) {
      return Response.json({ error: "Invalid game token" }, { status: 400 });
    }

    const turnIdx = currentTurnIndex(state);
    if (turnIdx === -1) {
      return Response.json({ error: "Game already complete" }, { status: 400 });
    }

    // Judge the guess
    const verdict = await judge(state.phrase, guess.trim());

    // Record the guess
    state.turns[turnIdx].guess = guess.trim();
    state.turns[turnIdx].verdict = verdict;

    const isLastTurn = turnIdx === MAX_TURNS - 1;
    const isDone = verdict === "correct" || isLastTurn;

    if (isDone) {
      const score = calcScore(state.tier, turnIdx, verdict);
      return Response.json({
        verdict,
        done: true,
        phrase: state.phrase,
        tier: state.tier,
        score,
      });
    }

    // Generate next emoji and add a new turn slot
    const emoji = await nextEmoji(state);
    state.turns.push({ emoji, guess: null, verdict: null });

    const newToken = encryptState(state);

    return Response.json({
      verdict,
      done: false,
      nextEmoji: emoji,
      token: newToken,
      turn: turnIdx + 2, // 1-indexed turn number for the new turn
    });
  } catch (err) {
    console.error("[/api/game/guess]", err);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
