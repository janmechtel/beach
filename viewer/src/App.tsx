import { useState } from "react";
import VideoSelector from "./components/VideoSelector";
import ActionViewer from "./components/ActionViewer";

export default function App() {
  const [selectedStem, setSelectedStem] = useState("");

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 flex items-center gap-4 px-4 py-2 border-b border-border bg-card">
        <h1 className="text-sm font-semibold tracking-wide text-foreground">
          Beach Volleyball Action Viewer
        </h1>
        <VideoSelector selected={selectedStem} onChange={setSelectedStem} />
      </header>

      {/* Main content */}
      <main className="flex-1 min-h-0 overflow-hidden p-3">
        {selectedStem ? (
          <ActionViewer stem={selectedStem} />
        ) : (
          <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
            Select a video above to begin.
          </div>
        )}
      </main>
    </div>
  );
}
