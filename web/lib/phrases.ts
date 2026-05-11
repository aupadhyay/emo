import type { Tier, Verdict } from "./game-state";

export interface Phrase {
  text: string;
  tier: Tier;
}

export const phrases: Phrase[] = [
  // Easy
  { text: "pizza", tier: "easy" },
  { text: "birthday party", tier: "easy" },
  { text: "rainy day", tier: "easy" },
  { text: "sunrise", tier: "easy" },
  { text: "beach vacation", tier: "easy" },
  { text: "snowfall", tier: "easy" },
  { text: "coffee shop", tier: "easy" },
  { text: "rainbow", tier: "easy" },
  { text: "campfire", tier: "easy" },
  { text: "thunderstorm", tier: "easy" },

  // Medium
  { text: "road trip", tier: "medium" },
  { text: "first date", tier: "medium" },
  { text: "gym workout", tier: "medium" },
  { text: "movie night", tier: "medium" },
  { text: "cooking dinner", tier: "medium" },
  { text: "job interview", tier: "medium" },
  { text: "graduation day", tier: "medium" },
  { text: "wedding day", tier: "medium" },
  { text: "camping trip", tier: "medium" },
  { text: "traffic jam", tier: "medium" },

  // Hard
  { text: "Star Wars", tier: "hard" },
  { text: "The Lion King", tier: "hard" },
  { text: "Breaking Bad", tier: "hard" },
  { text: "Rolling in the Deep", tier: "hard" },
  { text: "Finding Nemo", tier: "hard" },
  { text: "Game of Thrones", tier: "hard" },
  { text: "Saving Private Ryan", tier: "hard" },
  { text: "Don't Stop Believin'", tier: "hard" },
  { text: "Titanic", tier: "hard" },
  { text: "the early bird catches the worm", tier: "hard" },

  // Expert
  { text: "climate change", tier: "expert" },
  { text: "artificial intelligence", tier: "expert" },
  { text: "supply and demand", tier: "expert" },
  { text: "the American dream", tier: "expert" },
  { text: "survival of the fittest", tier: "expert" },
  { text: "the butterfly effect", tier: "expert" },
  { text: "the Cold War", tier: "expert" },
  { text: "quantum entanglement", tier: "expert" },
  { text: "the Big Bang", tier: "expert" },
  { text: "free will", tier: "expert" },
];

export function randomPhrase(): Phrase {
  return phrases[Math.floor(Math.random() * phrases.length)];
}

const SCORES: Record<Tier, [number, number, number]> = {
  easy:   [100,  60,  30],
  medium: [200, 120,  60],
  hard:   [400, 240, 120],
  expert: [800, 480, 240],
};

export function calcScore(
  tier: Tier,
  turnIndex: number, // 0-based
  verdict: Verdict
): number {
  if (verdict === "wrong") return 0;
  const base = SCORES[tier][Math.min(turnIndex, 2)];
  return verdict === "partial" ? Math.floor(base / 2) : base;
}
