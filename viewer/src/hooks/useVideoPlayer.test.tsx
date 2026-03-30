import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useVideoPlayer } from "./useVideoPlayer";

class FakeVideo extends EventTarget {
  private _currentTime = 0;
  duration = 120;
  playbackRate = 1;
  paused = true;

  readonly play = vi.fn(async () => {
    this.paused = false;
    this.dispatchEvent(new Event("play"));
  });

  readonly pause = vi.fn(() => {
    this.paused = true;
    // Fire pause later so it can race with the next seekAndPlay timer setup.
    setTimeout(() => this.dispatchEvent(new Event("pause")), 1);
  });

  get currentTime() {
    return this._currentTime;
  }

  set currentTime(value: number) {
    this._currentTime = value;
    queueMicrotask(() => this.dispatchEvent(new Event("seeked")));
  }
}

describe("useVideoPlayer", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("keeps next clip timer when prior clip pause event arrives late", async () => {
    vi.useFakeTimers();

    const { result } = renderHook(() => useVideoPlayer());
    const controls = result.current[1];
    const video = new FakeVideo();

    act(() => {
      controls.setVideoEl(video as unknown as HTMLVideoElement);
    });

    act(() => {
      controls.seekAndPlay(10, 1, () => {
        controls.seekAndPlay(20, 1);
      });
    });

    // First seek completes and arms timer for clip #1.
    await act(async () => {
      await Promise.resolve();
    });
    expect(video.play).toHaveBeenCalledTimes(1);

    // Clip #1 timer fires: pauses and chains seekAndPlay for clip #2.
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(video.pause).toHaveBeenCalledTimes(1);

    // Clip #2 seek completes and arms its own timer.
    await act(async () => {
      await Promise.resolve();
    });
    expect(video.play).toHaveBeenCalledTimes(2);

    // Delayed pause event from clip #1 lands now.
    act(() => {
      vi.advanceTimersByTime(1);
    });

    // Clip #2 timer must still fire (regression guard for pause/timer race).
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(video.pause).toHaveBeenCalledTimes(2);
  });
});
