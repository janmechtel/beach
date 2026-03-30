import { useState } from "react";
import type { Action } from "../lib/types";
import { ACTION_TYPES, PLAYER_IDS } from "../lib/types";
import { cn, fmtTime, playerColor } from "../lib/utils";

interface Props {
  action: Action;
  index: number;
  isActive: boolean;
  isEditing: boolean;
  onClick: () => void;
  onEdit: () => void;
  onSave: (patch: Partial<Action>) => void;
  onDelete: () => void;
  onCancelEdit: () => void;
}

export default function ActionRow({
  action,
  isActive,
  isEditing,
  onClick,
  onEdit,
  onSave,
  onDelete,
  onCancelEdit,
}: Props) {
  const [editTs, setEditTs] = useState(action.timestamp_sec.toFixed(1));
  const [editPlayer, setEditPlayer] = useState(action.player_id);
  const [editAction, setEditAction] = useState(action.action);

  const color = playerColor(action.player_id);

  if (isEditing) {
    return (
      <div className="border border-border rounded px-2 py-2 space-y-2 bg-card">
        <div className="flex gap-2 items-center">
          <input
            type="number"
            step="0.1"
            min="0"
            value={editTs}
            onChange={(e) => setEditTs(e.target.value)}
            className="w-20 bg-background border border-border rounded px-2 py-1 text-xs font-mono text-foreground"
            title="Timestamp (seconds)"
          />
          <select
            value={editPlayer}
            onChange={(e) => setEditPlayer(e.target.value as Action["player_id"])}
            className="bg-background border border-border rounded px-1 py-1 text-xs text-foreground flex-1"
          >
            {PLAYER_IDS.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
          <select
            value={editAction}
            onChange={(e) => setEditAction(e.target.value as Action["action"])}
            className="bg-background border border-border rounded px-1 py-1 text-xs text-foreground flex-1"
          >
            {ACTION_TYPES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() =>
              onSave({
                timestamp_sec: parseFloat(editTs),
                player_id: editPlayer as Action["player_id"],
                action: editAction as Action["action"],
              })
            }
            className="text-xs px-2 py-1 rounded bg-primary text-primary-foreground hover:opacity-90"
          >
            Save
          </button>
          <button
            onClick={onDelete}
            className="text-xs px-2 py-1 rounded bg-destructive text-destructive-foreground hover:opacity-90"
          >
            Delete
          </button>
          <button
            onClick={onCancelEdit}
            className="text-xs px-2 py-1 rounded border border-border text-muted-foreground hover:bg-accent"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer select-none group",
        "hover:bg-accent/50",
        isActive && "bg-accent"
      )}
      onClick={onClick}
      onDoubleClick={onEdit}
      title="Click to seek · Double-click to edit"
    >
      {/* Player color strip */}
      <span
        className="w-1 h-4 rounded-full flex-shrink-0"
        style={{ backgroundColor: color }}
      />

      {/* Timestamp */}
      <span className="font-mono text-xs text-muted-foreground w-12 text-right flex-shrink-0">
        {fmtTime(action.timestamp_sec)}
      </span>

      {/* Player ID */}
      <span
        className="text-xs font-semibold w-6 flex-shrink-0"
        style={{ color }}
      >
        {action.player_id}
      </span>

      {/* Action type */}
      <span className="text-xs text-foreground truncate flex-1">
        {action.action}
      </span>

      {/* Edit button — visible on hover */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onEdit();
        }}
        className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground text-xs px-1 py-0.5 rounded hover:bg-accent transition-opacity"
        title="Edit"
      >
        ✎
      </button>
    </div>
  );
}
