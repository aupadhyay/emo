import Anthropic from "@anthropic-ai/sdk";
import {
  decryptState,
  encryptState,
  currentTurnIndex,
  type GameState,
  type Verdict,
} from "@/lib/game-state";
import { generateEmoji, type HistoryTurn } from "@/lib/emoji-gen";
import { logGame } from "@/lib/game-log";

function scoreFor(verdict: Verdict): number {
  if (verdict === "correct") return 3;
  if (verdict === "partial") return 1;
  return 0;
}

const client = new Anthropic();

const MAX_TURNS = 3;

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

function buildHistory(state: GameState): HistoryTurn[] {
  return state.turns
    .filter((t) => t.guess !== null)
    .map((t) => ({ emoji: t.emoji, guess: t.guess as string }));
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

    const verdict = await judge(state.phrase, guess.trim());

    state.turns[turnIdx].guess = guess.trim();
    state.turns[turnIdx].verdict = verdict;

    const isLastTurn = turnIdx === MAX_TURNS - 1;
    const isDone = verdict === "correct" || isLastTurn;

    if (isDone) {
      const score = scoreFor(verdict);
      // Fire-and-forget logging; don't block the response.
      logGame(state, verdict, score).catch((err) =>
        console.error("[/api/game/guess] logGame failed:", err),
      );
      return Response.json({
        verdict,
        done: true,
        phrase: state.phrase,
        score,
      });
    }

    const emoji = await generateEmoji(state.phrase, buildHistory(state));
    state.turns.push({ emoji, guess: null, verdict: null });

    const newToken = encryptState(state);

    return Response.json({
      verdict,
      done: false,
      nextEmoji: emoji,
      token: newToken,
      turn: turnIdx + 2,
    });
  } catch (err) {
    console.error("[/api/game/guess]", err);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
