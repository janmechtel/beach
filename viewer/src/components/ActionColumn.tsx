import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ActionRow from "./ActionRow";
import { useActions } from "../hooks/useActions";
import type { Action } from "../lib/types";
import { ACTION_TYPES, PLAYER_IDS } from "../lib/types";

interface Props {
  stem: string;
  availableFiles: string[];
  initialFile?: string;
  currentTime: number;
  /** Called to seek+play a clip: same signature as seekAndPlay */
  onPlayClip: (start: number, durationSec?: number | null, onEnd?: () => void) => void;
  canClose: boolean;
  secondsBefore: number;
  secondsAfter: number;
  repeatMode: boolean;
  onClose: () => void;
}

// Clip timing and repeat mode are controlled globally by ActionViewer.

export default function ActionColumn({
  stem,
  availableFiles,
  initialFile,
  currentTime,
  onPlayClip,
  secondsBefore,
  secondsAfter,
  repeatMode,
  canClose,
  onClose,
}: Props) {
  const [selectedFile, setSelectedFile] = useState(initialFile ?? availableFiles[0] ?? "");
  const [filterPlayer, setFilterPlayer] = useState("");
  const [filterAction, setFilterAction] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

  const [actionsState, actionsControls] = useActions();
  const { actions, error } = actionsState;

  // Load on file change
  useEffect(() => {
    if (selectedFile && stem) {
      actionsControls.load(stem, selectedFile);
    }
  }, [stem, selectedFile]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Filtered actions
  const filtered = useMemo(
    () =>
      actions.filter((a) => {
        if (filterPlayer && a.player_id !== filterPlayer) return false;
        if (filterAction && a.action !== filterAction) return false;
        return true;
      }),
    [actions, filterPlayer, filterAction]
  );

  // Highlight nearest row to currentTime
  const nearestIdx = useMemo(() => {
    if (!filtered.length) return 0;
    let best = 0;
    for (let i = 0; i < filtered.length; i++) {
      if (filtered[i].timestamp_sec <= currentTime + 0.15) best = i;
    }
    return best;
  }, [filtered, currentTime]);

  // Auto-scroll the highlighted row into view during playback
  useEffect(() => {
    if (listRef.current) {
      const rows = listRef.current.querySelectorAll("[data-row]");
      const row = rows[nearestIdx] as Element | undefined;
      if (row && "scrollIntoView" in row && typeof row.scrollIntoView === "function") {
        row.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }
  }, [nearestIdx]);


  /**
   * Seek to filtered[idx] with the configured before/after offsets.
   * onEnd chains to the next filtered action, or stops at the last one.
   * repeatMode re-plays the same clip instead of advancing.
   */
  const playClip = useCallback(
    (idx: number) => {
      if (!filtered.length) return;
      const ts = filtered[idx].timestamp_sec;
      const start = Math.max(0, ts - secondsBefore);
      const duration = secondsBefore + secondsAfter;
      const onEnd = () => {
        const nextIdx = idx + 1;
        if (repeatMode) {
          // Loop same clip.
          playClip(idx);
        } else if (nextIdx < filtered.length) {
          // Advance to next filtered action.
          playClip(nextIdx);
        }
        // At last action with no repeat — stop (video is already paused by timer).
      };
      onPlayClip(start, duration, onEnd);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [filtered, secondsBefore, secondsAfter, repeatMode, onPlayClip]
  );

  const handleRowClick = useCallback(
    (filteredIdx: number) => {
      // Selecting an action starts clip playback from this filtered row and then
      // chains through the remaining filtered actions.
      playClip(filteredIdx);
    },
    [playClip]
  );

  // Unique player IDs and action types actually present in this file
  const presentPlayers = useMemo(
    () => [...new Set(actions.map((a) => a.player_id))].sort() as typeof PLAYER_IDS,
    [actions]
  );
  const presentActions = useMemo(
    () => ACTION_TYPES.filter((t) => actions.some((a: Action) => a.action === t)),
    [actions]
  );

  return (
    <div className="flex flex-col border border-border rounded-lg overflow-hidden min-w-0 flex-1 bg-card">
      {/* Column header */}
      <div className="flex-shrink-0 border-b border-border p-2 space-y-2">
        <div className="flex items-center gap-2">
          <select
            value={selectedFile}
            onChange={(e) => setSelectedFile(e.target.value)}
            className="flex-1 min-w-0 bg-background border border-border rounded px-2 py-1 text-xs text-foreground truncate"
            title="Select action file"
          >
            {availableFiles.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>

          {canClose && (
            <button
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-accent flex-shrink-0"
              title="Close column"
            >
              ✕
            </button>
          )}
        </div>

        {/* Filters */}
        <div className="flex gap-2">
          <select
            value={filterPlayer}
            onChange={(e) => {
              setFilterPlayer(e.target.value);
            }}
            className="flex-1 min-w-0 bg-background border border-border rounded px-1 py-0.5 text-xs text-foreground"
          >
            <option value="">All players</option>
            {presentPlayers.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
          <select
            value={filterAction}
            onChange={(e) => {
              setFilterAction(e.target.value);
            }}
            className="flex-1 min-w-0 bg-background border border-border rounded px-1 py-0.5 text-xs text-foreground"
          >
            <option value="">All actions</option>
            {presentActions.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>

        {/* Clip controls */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground flex-shrink-0">
            {filtered.length === actions.length
              ? `${actions.length} actions`
              : `${filtered.length} / ${actions.length}`}
          </span>

          <span className="text-muted-foreground flex-shrink-0">
            clip: −{secondsBefore.toFixed(1)}s / +{secondsAfter.toFixed(1)}s
          </span>

        </div>

        {error && (
          <div className="text-xs text-destructive bg-destructive/10 rounded px-2 py-1 truncate">
            {error}
          </div>
        )}
      </div>

      {/* Action list */}
      <div ref={listRef} className="flex-1 overflow-y-auto p-1 space-y-0.5">
        {filtered.map((action, filteredIdx) => (
          <div key={`${filteredIdx}-${action.timestamp_sec}`} data-row>
            <ActionRow
              action={action}
              isActive={filteredIdx === nearestIdx}
              onClick={() => handleRowClick(filteredIdx)}
            />
          </div>
        ))}
        {filtered.length === 0 && actions.length > 0 && (
          <p className="text-xs text-muted-foreground text-center py-4">
            No actions match the current filter.
          </p>
        )}
        {actions.length === 0 && !error && (
          <p className="text-xs text-muted-foreground text-center py-4">
            Select a file above to load actions.
          </p>
        )}
      </div>
    </div>
  );
}
