import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format seconds as mm:ss.t */
export function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

/** Deterministic HSL color from a player ID string. */
export function playerColor(playerId: string): string {
  const hues: Record<string, number> = {
    P1: 200,
    P2: 30,
    P3: 120,
    P4: 270,
  };
  const hue = hues[playerId] ?? ((playerId.charCodeAt(0) * 137) % 360);
  return `hsl(${hue}, 65%, 62%)`;
}
