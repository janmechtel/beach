import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ActionRow from "./ActionRow";
import { useActions } from "../hooks/useActions";
import type { Action } from "../lib/types";
import { ACTION_TYPES, PLAYER_IDS } from "../lib/types";
import { cn } from "../lib/utils";

interface Props {
  stem: string;
  availableFiles: string[];
  initialFile?: string;
  currentTime: number;
  /** Called when user clicks a row — seekTo timestamp */
  onSeek: (t: number) => void;
  /** Called on play-after-seek (e.g. Next button) */
  onPlayAt: (t: number) => void;
  canClose: boolean;
  onClose: () => void;
}

// Clip playback constants (matching viewer.html behavior)
const BEFORE_OFFSET = 1.5;   // seconds before action to start clip

export default function ActionColumn({
  stem,
  availableFiles,
  initialFile,
  currentTime,
  onSeek,
  onPlayAt,
  canClose,
  onClose,
}: Props) {
  const [selectedFile, setSelectedFile] = useState(initialFile ?? availableFiles[0] ?? "");
  const [filterPlayer, setFilterPlayer] = useState("");
  const [filterAction, setFilterAction] = useState("");
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [activeRowIndex, setActiveRowIndex] = useState<number | null>(null);
  const [repeatMode, setRepeatMode] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  const [actionsState, actionsControls] = useActions();
  const { actions, dirty, saveState, error } = actionsState;

  // Load on file change
  useEffect(() => {
    if (selectedFile && stem) {
      actionsControls.load(stem, selectedFile);
      setEditingIndex(null);
      setActiveRowIndex(null);
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
      rows[nearestIdx]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [nearestIdx]);

  const handleRowClick = useCallback(
    (filteredIdx: number) => {
      setActiveRowIndex(filteredIdx);
      const ts = filtered[filteredIdx].timestamp_sec;
      // Seek to exact timestamp on click; nav buttons (◀▶) trigger play-from-before.
      onSeek(ts);
    },
    [filtered, onSeek]
  );


  const handleSaveEdit = useCallback(
    (globalIndex: number, patch: Partial<Action>) => {
      actionsControls.update(globalIndex, patch);
      setEditingIndex(null);
    },
    [actionsControls]
  );

  const handleDelete = useCallback(
    (globalIndex: number) => {
      if (!confirm("Delete this action?")) return;
      actionsControls.remove(globalIndex);
      setEditingIndex(null);
    },
    [actionsControls]
  );

  const handleSave = useCallback(async () => {
    if (!dirty) return;
    await actionsControls.save(stem, selectedFile);
  }, [dirty, actionsControls, stem, selectedFile]);

  // Unique player IDs and action types actually present in this file
  const presentPlayers = useMemo(
    () => [...new Set(actions.map((a) => a.player_id))].sort() as typeof PLAYER_IDS,
    [actions]
  );
  const presentActions = useMemo(
    () => ACTION_TYPES.filter((t) => actions.some((a) => a.action === t)),
    [actions]
  );

  // Map filtered index back to global index
  const globalIndexOf = useCallback(
    (filteredIdx: number): number => {
      const target = filtered[filteredIdx];
      return actions.indexOf(target);
    },
    [filtered, actions]
  );

  const saveLabel =
    saveState === "saving"
      ? "Saving…"
      : saveState === "saved"
      ? "Saved"
      : "Save";

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

          <button
            onClick={handleSave}
            disabled={!dirty || saveState === "saving"}
            className={cn(
              "text-xs px-2 py-1 rounded border transition-colors flex-shrink-0",
              dirty && saveState !== "saving"
                ? "border-blue-500 text-blue-400 hover:bg-blue-500/10"
                : saveState === "saved"
                ? "border-green-600 text-green-400"
                : "border-border text-muted-foreground opacity-50"
            )}
          >
            {saveLabel}
          </button>

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
              setActiveRowIndex(null);
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
              setActiveRowIndex(null);
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
          <span className="text-muted-foreground">
            {filtered.length === actions.length
              ? `${actions.length} actions`
              : `${filtered.length} / ${actions.length}`}
          </span>
          <div className="ml-auto flex gap-1">
            <button
              onClick={() => {
                if (activeRowIndex === null || !filtered.length) return;
                const prevIdx = Math.max(0, activeRowIndex - 1);
                setActiveRowIndex(prevIdx);
                const ts = filtered[prevIdx].timestamp_sec;
                onPlayAt(Math.max(0, ts - BEFORE_OFFSET));
              }}
              className="px-2 py-0.5 rounded border border-border text-muted-foreground hover:bg-accent"
              title="Previous action"
            >
              ◀
            </button>
            <button
              onClick={() => {
                if (activeRowIndex === null || !filtered.length) return;
                const nextIdx = Math.min(filtered.length - 1, activeRowIndex + 1);
                setActiveRowIndex(nextIdx);
                const ts = filtered[nextIdx].timestamp_sec;
                onPlayAt(Math.max(0, ts - BEFORE_OFFSET));
              }}
              className="px-2 py-0.5 rounded border border-border text-muted-foreground hover:bg-accent"
              title="Next action"
            >
              ▶
            </button>
            <button
              onClick={() => setRepeatMode((r) => !r)}
              className={cn(
                "px-2 py-0.5 rounded border",
                repeatMode
                  ? "border-blue-500 text-blue-400"
                  : "border-border text-muted-foreground hover:bg-accent"
              )}
              title="Toggle repeat clip"
            >
              ↺
            </button>
          </div>
        </div>

        {error && (
          <div className="text-xs text-destructive bg-destructive/10 rounded px-2 py-1 truncate">
            {error}
          </div>
        )}
      </div>

      {/* Action list */}
      <div ref={listRef} className="flex-1 overflow-y-auto p-1 space-y-0.5">
        {filtered.map((action, filteredIdx) => {
          const gIdx = globalIndexOf(filteredIdx);
          return (
            <div key={`${gIdx}-${action.timestamp_sec}`} data-row>
              <ActionRow
                action={action}
                index={gIdx}
                isActive={filteredIdx === nearestIdx}
                isEditing={editingIndex === gIdx}
                onClick={() => handleRowClick(filteredIdx)}
                onEdit={() => setEditingIndex(gIdx)}
                onSave={(patch) => handleSaveEdit(gIdx, patch)}
                onDelete={() => handleDelete(gIdx)}
                onCancelEdit={() => setEditingIndex(null)}
              />
            </div>
          );
        })}
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
