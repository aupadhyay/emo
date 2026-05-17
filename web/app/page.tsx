"use client";

import { useState, useRef, useEffect, useCallback } from "react";

type Verdict = "correct" | "partial" | "wrong";

type Message =
  | { kind: "ai"; emoji: string; turn: number }
  | { kind: "user"; text: string; verdict: Verdict | null }
  | { kind: "system"; text: string; isCorrect?: boolean };

type Phase = "idle" | "playing" | "result";

interface RoundResult {
  phrase: string;
  score: number;
  won: boolean;
}

async function apiStartGame(): Promise<{ token: string; emoji: string }> {
  const res = await fetch("/api/game/start", { method: "POST" });
  if (!res.ok) throw new Error("Failed to start game");
  return res.json();
}

async function apiGuess(
  token: string,
  guess: string
): Promise<
  | { verdict: Verdict; done: false; nextEmoji: string; token: string; turn: number }
  | { verdict: Verdict; done: true; phrase: string; score: number }
> {
  const res = await fetch("/api/game/guess", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, guess }),
  });
  if (!res.ok) throw new Error("Failed to submit guess");
  return res.json();
}

export default function Home() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [result, setResult] = useState<RoundResult | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  const inputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme === "light" ? "light" : "");
  }, [theme]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const startGame = useCallback(async () => {
    if (loading) return;
    setLoading(true);
    setMessages([]);
    setResult(null);
    setInput("");

    try {
      const { token: tok, emoji } = await apiStartGame();
      setToken(tok);
      setMessages([{ kind: "ai", emoji, turn: 1 }]);
      setPhase("playing");
    } catch {
      setMessages([{ kind: "system", text: "[ error ] failed to start. try again." }]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [loading]);

  // Enter to start/next round from idle or result
  useEffect(() => {
    if (phase !== "idle" && phase !== "result") return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Enter") startGame();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [phase, startGame]);

  async function handleGuess(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading || phase !== "playing" || !token) return;

    const guess = input.trim();
    setInput("");
    setLoading(true);

    // Show the guess immediately, verdict pending
    setMessages((prev) => [...prev, { kind: "user", text: guess, verdict: null }]);

    try {
      const res = await apiGuess(token, guess);

      if (res.done) {
        // Fill in verdict on the pending message, then add system message
        setMessages((prev) => {
          const updated = [...prev];
          const lastUser = updated.findLastIndex((m) => m.kind === "user");
          if (lastUser !== -1) updated[lastUser] = { kind: "user", text: guess, verdict: res.verdict };
          return [
            ...updated,
            {
              kind: "system",
              text: res.score > 0
                ? `[ ${res.verdict} ]  "${res.phrase}"  +${res.score} pts`
                : `[ failed ]  the phrase was "${res.phrase}"`,
              isCorrect: res.verdict === "correct" || res.verdict === "partial",
            },
          ];
        });
        setResult({ phrase: res.phrase, score: res.score, won: res.verdict !== "wrong" });
        setToken(null);
        setPhase("result");
      } else {
        // Fill in verdict, then add next AI turn
        setMessages((prev) => {
          const updated = [...prev];
          const lastUser = updated.findLastIndex((m) => m.kind === "user");
          if (lastUser !== -1) updated[lastUser] = { kind: "user", text: guess, verdict: res.verdict };
          return [...updated, { kind: "ai", emoji: res.nextEmoji, turn: res.turn }];
        });
        setToken(res.token);
        setTimeout(() => inputRef.current?.focus(), 50);
      }
    } catch {
      setMessages((prev) => {
        const updated = [...prev];
        const lastUser = updated.findLastIndex((m) => m.kind === "user");
        if (lastUser !== -1) updated[lastUser] = { kind: "user", text: guess, verdict: "wrong" };
        return [...updated, { kind: "system", text: "[ error ] something went wrong. try again." }];
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="game-wrap">
      <header className="game-header">
        <span className="game-title">emo</span>
        <span className="game-meta">{"// emoji charades · qwen2.5-3b"}</span>
        <div className="header-links">
          <a
            href="https://en.wikipedia.org/wiki/Pantheon_(TV_series)"
            target="_blank"
            rel="noopener noreferrer"
            className="header-link header-link-pantheon"
          >
            inspired by pantheon
          </a>
          <a
            href="https://github.com/aupadhyay/emo"
            target="_blank"
            rel="noopener noreferrer"
            className="header-link"
          >
            github
          </a>
          <button
            className="theme-btn"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? "light" : "dark"}
          </button>
        </div>
      </header>

      <div className={`chat-area${phase === "idle" ? " chat-idle" : ""}`}>
        {phase === "idle" && (
          <div className="idle-screen">
            <p className="idle-line title">guess the phrase from emoji clues.</p>
            <p className="idle-line">the ai gets up to 3 turns to communicate.</p>
            <p className="idle-line">the model gets better as you play.</p>
            <button className="play-btn" onClick={startGame}>
              {loading ? "loading..." : "[ press enter to play ]"}
            </button>
          </div>
        )}

        {messages.map((msg, i) => {
          if (msg.kind === "ai") {
            return (
              <div key={i} className="msg msg-ai">
                <span className="label label-ai">&lt;emo&gt;</span>
                <span className="sep"> : </span>
                <span className="emoji-out">{msg.emoji}</span>
              </div>
            );
          }
          if (msg.kind === "user") {
            return (
              <div key={i} className="msg msg-user">
                {msg.verdict && (
                  <>
                    <span className={`verdict verdict-${msg.verdict}`}>
                      [ {msg.verdict} ]
                    </span>
                    <span className="sep"> </span>
                  </>
                )}
                <span className="guess-text">{msg.text}</span>
                <span className="sep"> : </span>
                <span className="label label-user">&lt;you&gt;</span>
              </div>
            );
          }
          if (msg.kind === "system") {
            return (
              <div key={i} className="msg msg-system">
                <span className={`sys-text${msg.isCorrect ? " correct" : ""}`}>
                  {msg.text}
                </span>
              </div>
            );
          }
        })}

        {phase === "result" && (
          <div className="round-result">
            <button className="next-btn" onClick={startGame}>
              {loading ? "loading..." : "[ enter — next round ]"}
            </button>
          </div>
        )}

        {loading && phase === "playing" && (
          <div className="msg msg-ai">
            <span className="label label-ai">&lt;emo&gt;</span>
            <span className="sep"> : </span>
            <span className="sys-text">...</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {phase === "playing" && (
        <form onSubmit={handleGuess} className="input-row">
          <span className="input-prompt">&gt;</span>
          <input
            ref={inputRef}
            className="input-field"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="your guess"
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
            disabled={loading}
          />
        </form>
      )}
    </div>
  );
}
