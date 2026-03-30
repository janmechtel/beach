import { useEffect, useState } from "react";
import { listVideos } from "../lib/api";

interface Props {
  selected: string;
  onChange: (stem: string) => void;
}

export default function VideoSelector({ selected, onChange }: Props) {
  const [stems, setStems] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listVideos()
      .then((list) => {
        setStems(list);
        setLoading(false);
        // Auto-select first if nothing selected yet
        if (!selected && list.length > 0) onChange(list[0]);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) {
    return (
      <span className="text-xs text-muted-foreground">Loading videos…</span>
    );
  }

  if (error) {
    return (
      <span className="text-xs text-destructive" title={error}>
        Failed to load videos. Is the server running?
      </span>
    );
  }

  if (stems.length === 0) {
    return (
      <span className="text-xs text-muted-foreground">
        No video data found in data/.
      </span>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <label className="text-xs text-muted-foreground uppercase tracking-wider">
        Video
      </label>
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        className="bg-background border border-border rounded px-2 py-1 text-sm text-foreground"
      >
        {stems.map((stem) => (
          <option key={stem} value={stem}>
            {stem}
          </option>
        ))}
      </select>
    </div>
  );
}
