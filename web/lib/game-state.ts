import { createCipheriv, createDecipheriv, randomBytes } from "crypto";

export type Tier = "easy" | "medium" | "hard" | "expert";
export type Verdict = "correct" | "partial" | "wrong";

export interface Turn {
  emoji: string;
  guess: string | null;
  verdict: Verdict | null;
}

export interface GameState {
  phrase: string;
  tier: Tier;
  turns: Turn[]; // emoji pre-filled per turn; guess/verdict filled as player guesses
}

function getKey(): Buffer {
  const secret =
    process.env.GAME_SECRET ?? "dev-secret-please-change-in-prod!!";
  const buf = Buffer.alloc(32);
  Buffer.from(secret, "utf8").copy(buf);
  return buf;
}

export function encryptState(state: GameState): string {
  const key = getKey();
  const iv = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", key, iv);
  const json = JSON.stringify(state);
  const encrypted = Buffer.concat([cipher.update(json, "utf8"), cipher.final()]);
  const tag = cipher.getAuthTag();
  return Buffer.concat([iv, tag, encrypted]).toString("base64url");
}

export function decryptState(token: string): GameState | null {
  try {
    const key = getKey();
    const buf = Buffer.from(token, "base64url");
    if (buf.length < 29) return null;
    const iv = buf.subarray(0, 12);
    const tag = buf.subarray(12, 28);
    const encrypted = buf.subarray(28);
    const decipher = createDecipheriv("aes-256-gcm", key, iv);
    decipher.setAuthTag(tag);
    const decrypted = Buffer.concat([
      decipher.update(encrypted),
      decipher.final(),
    ]);
    return JSON.parse(decrypted.toString("utf8")) as GameState;
  } catch {
    return null;
  }
}

// Current turn index = first turn with no guess yet
export function currentTurnIndex(state: GameState): number {
  return state.turns.findIndex((t) => t.guess === null);
}
