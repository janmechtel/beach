import type { Action } from "../lib/types";
import { cn, fmtTime, playerColor } from "../lib/utils";

interface Props {
  action: Action;
  isActive: boolean;
  onClick: () => void;
}

export default function ActionRow({ action, isActive, onClick }: Props) {
  const color = playerColor(action.player_id);

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer select-none",
        "hover:bg-accent/50",
        isActive && "bg-accent"
      )}
      onClick={onClick}
      title="Click to play clip"
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
    </div>
  );
}
