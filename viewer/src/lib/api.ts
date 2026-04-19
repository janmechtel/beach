import type { Action } from "./types";

/** List available video stems from the static manifest. */
export async function listVideos(): Promise<string[]> {
  const res = await fetch("/api/manifest.json");
  if (!res.ok) throw new Error(`Failed to load manifest: ${res.status}`);
  const data = await res.json();
  return data.stems as string[];
}

/**
 * List all filenames published for a video stem (JSON + mp4).
 * ActionViewer filters these client-side with pickVideoFile + actionJsonFiles,
 * so both JSON and mp4 names must be present here.
 */
export async function listActions(stem: string): Promise<string[]> {
  const res = await fetch(`/api/${encodeURIComponent(stem)}/actions.json`);
  if (!res.ok) throw new Error(`Failed to load file list for ${stem}: ${res.status}`);
  return res.json();
}

/** Load a specific action JSON file via the /data/ path (proxied to R2 in production). */
export async function loadActions(stem: string, filename: string): Promise<Action[]> {
  const res = await fetch(
    `/data/${encodeURIComponent(stem)}/${encodeURIComponent(filename)}`
  );
  if (!res.ok) throw new Error(`Failed to load ${filename}: ${res.status}`);
  const data = await res.json();
  if (!Array.isArray(data)) throw new Error("Expected a JSON array of actions");
  return data as Action[];
}

/** URL to stream a data file (video, etc.) for a given stem. */
export function dataFileUrl(stem: string, filename: string): string {
  return `/data/${encodeURIComponent(stem)}/${encodeURIComponent(filename)}`;
}
