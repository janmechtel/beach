import { useEffect, useRef } from "react";
import type { VideoPlayerControls, VideoPlayerState } from "../hooks/useVideoPlayer";
import { fmtTime } from "../lib/utils";

interface Props {
  videoRef: VideoPlayerControls["videoRef"];
  src: string;
  state: VideoPlayerState;
  onTogglePlay: () => void;
  onSeek: (t: number) => void;
  onRateChange: (rate: number) => void;
}

const RATES = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

export default function VideoPlayer({
  videoRef,
  src,
  state,
  onTogglePlay,
  onSeek,
  onRateChange,
}: Props) {
  const { currentTime, duration, paused, playbackRate } = state;
  const seekBarRef = useRef<HTMLInputElement>(null);

  // Keep seek bar in sync with playback without causing re-renders.
  useEffect(() => {
    if (seekBarRef.current && duration > 0) {
      seekBarRef.current.value = String(currentTime / duration);
    }
  }, [currentTime, duration]);

  // Spacebar play/pause — guard against triggering when focus is on an input.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      if (
        e.code === "Space" &&
        target.tagName !== "INPUT" &&
        target.tagName !== "SELECT" &&
        target.tagName !== "TEXTAREA"
      ) {
        e.preventDefault();
        onTogglePlay();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onTogglePlay]);


  return (
    <div className="flex flex-col bg-black rounded-lg overflow-hidden">
      {/* Video element */}
      <div className="flex items-center justify-center bg-black min-h-0 flex-1">
        {src ? (
          <video
            ref={videoRef}
            src={src}
            className="max-w-full max-h-full"
            onClick={onTogglePlay}
            preload="metadata"
          />
        ) : (
          <div className="text-muted-foreground text-sm p-8 text-center">
            No video loaded.
            <br />
            Select a video stem to begin.
          </div>
        )}
      </div>

      {/* Custom control bar */}
      <div className="flex items-center gap-2 px-3 py-2 bg-card border-t border-border text-xs select-none">
        {/* Play/Pause */}
        <button
          onClick={onTogglePlay}
          className="w-7 h-7 flex items-center justify-center rounded text-foreground hover:bg-accent transition-colors"
          title={paused ? "Play (Space)" : "Pause (Space)"}
        >
          {paused ? (
            <svg viewBox="0 0 10 10" className="w-3 h-3 fill-current">
              <polygon points="2,1 9,5 2,9" />
            </svg>
          ) : (
            <svg viewBox="0 0 10 10" className="w-3 h-3 fill-current">
              <rect x="1" y="1" width="3" height="8" />
              <rect x="6" y="1" width="3" height="8" />
            </svg>
          )}
        </button>

        {/* Current time */}
        <span className="font-mono text-muted-foreground min-w-[52px]">
          {fmtTime(currentTime)}
        </span>

        {/* Seek bar */}
        <input
          ref={seekBarRef}
          type="range"
          min={0}
          max={1}
          step={0.0001}
          defaultValue={0}
          className="flex-1 h-1 accent-primary cursor-pointer"
          onChange={(e) => onSeek(parseFloat(e.target.value) * duration)}
        />

        {/* Duration */}
        <span className="font-mono text-muted-foreground min-w-[52px] text-right">
          {fmtTime(duration)}
        </span>

        {/* Playback speed */}
        <select
          value={playbackRate}
          onChange={(e) => onRateChange(parseFloat(e.target.value))}
          className="bg-background border border-border rounded px-1 py-0.5 text-foreground cursor-pointer"
          title="Playback speed"
        >
          {RATES.map((r) => (
            <option key={r} value={r}>
              {r}×
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
