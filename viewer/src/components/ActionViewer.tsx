import { useCallback, useEffect, useState } from "react";
import VideoPlayer from "./VideoPlayer";
import ActionColumn from "./ActionColumn";
import { listActions, dataFileUrl } from "../lib/api";
import { useVideoPlayer } from "../hooks/useVideoPlayer";

interface Props {
  stem: string;
}

/** Detect the video file in the data/<stem>/ directory.
 *  Priority: exact <stem>.mp4 > any non-annotated mp4 > annotated > identified > any.
 *  Annotated files are useful but large; raw is preferred when present.
 */
function pickVideoFile(stem: string, files: string[]): string | null {
  const mp4s = files.filter((f) => f.endsWith(".mp4") && !f.endsWith("_identified.mp4"));
  // 1. Exact stem match (e.g. first30.mp4)
  const exact = mp4s.find((f) => f === `${stem}.mp4`);
  if (exact) return exact;
  // 2. Any non-annotated mp4 (e.g. a trimmed court clip)
  const plain = mp4s.find((f) => !f.includes("annotated"));
  if (plain) return plain;
  // 3. Annotated version (has P1-P4 bounding boxes drawn)
  const annotated = mp4s.find((f) => f.includes("annotated"));
  if (annotated) return annotated;
  // 4. Anything else (including identified)
  return files.find((f) => f.endsWith(".mp4")) ?? null;
}

/** Filter a file list to action JSONs (exclude detections/identified large files). */
function actionJsonFiles(files: string[]): string[] {
  return files.filter(
    (f) =>
      f.endsWith(".json") &&
      !f.startsWith(".") &&
      !f.endsWith("_detections.json") &&
      !f.endsWith("_identified.json") &&
      !f.endsWith("players.json") &&
      !f.endsWith(".gemini_file_cache.json")
  );
}

interface ColumnDef {
  id: number;
  initialFile: string;
}

let _nextColId = 1;

export default function ActionViewer({ stem }: Props) {
  const [actionFiles, setActionFiles] = useState<string[]>([]);
  const [videoFile, setVideoFile] = useState<string | null>(null);
  const [columns, setColumns] = useState<ColumnDef[]>([]);

  const [playerState, playerControls] = useVideoPlayer();
  const { setVideoEl, seekTo, seekAndPlay, togglePlay, setPlaybackRate } = playerControls;

  // Reload file list when stem changes
  useEffect(() => {
    if (!stem) return;
    listActions(stem)
      .then((files) => {
        const af = actionJsonFiles(files);
        setActionFiles(af);
        const vf = pickVideoFile(stem, files);
        setVideoFile(vf);
        // Initialise with one column per available action file (up to 3).
        const initial = af.slice(0, Math.min(af.length, 1));
        setColumns(initial.map((f) => ({ id: _nextColId++, initialFile: f })));
      })
      .catch(console.error);
  }, [stem]);

  const videoSrc = videoFile ? dataFileUrl(stem, videoFile) : "";

  const addColumn = useCallback(() => {
    const last = actionFiles[actionFiles.length - 1] ?? "";
    setColumns((cols) => [...cols, { id: _nextColId++, initialFile: last }]);
  }, [actionFiles]);

  const removeColumn = useCallback((id: number) => {
    setColumns((cols) => cols.filter((c) => c.id !== id));
  }, []);

  // Seek + play at a given timestamp (used by Before-clip controls)
  const handlePlayAt = useCallback(
    (t: number) => seekAndPlay(t),
    [seekAndPlay]
  );

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Video + column layout */}
      <div className="flex gap-3 flex-1 min-h-0">
        {/* Left: video player */}
        <div className="flex flex-col" style={{ width: "40%", minWidth: 300 }}>
          <VideoPlayer
            setVideoEl={setVideoEl}
            src={videoSrc}
            state={playerState}
            onTogglePlay={togglePlay}
            onSeek={(t) => seekTo(t)}
            onRateChange={setPlaybackRate}
          />
        </div>

        {/* Right: action columns */}
        <div className="flex gap-2 flex-1 min-w-0 overflow-x-auto">
          {columns.map((col) => (
            <ActionColumn
              key={col.id}
              stem={stem}
              availableFiles={actionFiles}
              initialFile={col.initialFile}
              currentTime={playerState.currentTime}
              onSeek={seekTo}
              onPlayAt={handlePlayAt}
              canClose={columns.length > 1}
              onClose={() => removeColumn(col.id)}
            />
          ))}

          {/* Add column button */}
          <button
            onClick={addColumn}
            disabled={actionFiles.length === 0}
            className="flex-shrink-0 w-10 self-stretch flex items-center justify-center rounded-lg border border-dashed border-border text-muted-foreground hover:border-primary hover:text-primary transition-colors disabled:opacity-30"
            title="Add comparison column"
          >
            +
          </button>
        </div>
      </div>
    </div>
  );
}
