import type { Action } from "./types";

const BASE = "";  // relative — proxied by Vite dev server to http://localhost:8080

/** List available video stems (data/<stem>/ directories). */
export async function listVideos(): Promise<string[]> {
  const res = await fetch(`${BASE}/api/videos`);
  if (!res.ok) throw new Error(`GET /api/videos failed: ${res.status}`);
  return res.json();
}

/** List action JSON filenames for a video stem. */
export async function listActions(stem: string): Promise<string[]> {
  const res = await fetch(`${BASE}/api/videos/${encodeURIComponent(stem)}/actions`);
  if (!res.ok) throw new Error(`GET /api/videos/${stem}/actions failed: ${res.status}`);
  return res.json();
}

/** Load a specific action JSON file. */
export async function loadActions(stem: string, filename: string): Promise<Action[]> {
  const res = await fetch(
    `${BASE}/api/videos/${encodeURIComponent(stem)}/actions/${encodeURIComponent(filename)}`
  );
  if (!res.ok) throw new Error(`GET action file failed: ${res.status}`);
  const data = await res.json();
  if (!Array.isArray(data)) throw new Error("Expected a JSON array of actions");
  return data as Action[];
}

/** Save edited actions back to disk. */
export async function saveActions(
  stem: string,
  filename: string,
  actions: Action[]
): Promise<void> {
  const res = await fetch(
    `${BASE}/api/videos/${encodeURIComponent(stem)}/actions/${encodeURIComponent(filename)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(actions),
    }
  );
  if (!res.ok) throw new Error(`Save failed: ${res.status}`);
}

/** URL to stream a data file (video, etc.) for a given stem. */
export function dataFileUrl(stem: string, filename: string): string {
  return `${BASE}/data/${encodeURIComponent(stem)}/${encodeURIComponent(filename)}`;
}
