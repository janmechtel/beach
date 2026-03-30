import { useCallback, useRef, useState } from "react";

export interface VideoPlayerState {
  currentTime: number;
  duration: number;
  paused: boolean;
  playbackRate: number;
}

export interface VideoPlayerControls {
  /** Ref callback — pass directly to <video ref={...}>. */
  setVideoEl: (el: HTMLVideoElement | null) => void;
  seekTo: (t: number) => void;
  /** Seek to t, then play once the seek completes. */
  seekAndPlay: (t: number) => void;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  setPlaybackRate: (rate: number) => void;
}

/** Manages a shared video element. Returns live state and imperative controls.
 *
 * Uses a ref callback (`setVideoEl`) instead of useRef so that event listeners
 * are attached exactly when the <video> element mounts and removed when it
 * unmounts. A plain useRef + useEffect([]) misses the element because it is
 * conditionally rendered (src guard) and videoRef.current is null at mount.
 *
 * Handler functions are stored in a stable ref so that add/removeEventListener
 * always operate on the exact same function objects.
 */
export function useVideoPlayer(): [VideoPlayerState, VideoPlayerControls] {
  // Internal mutable ref so imperative controls always reach the live element.
  const videoElRef = useRef<HTMLVideoElement | null>(null);

  const [state, setState] = useState<VideoPlayerState>({
    currentTime: 0,
    duration: 0,
    paused: true,
    playbackRate: 1,
  });

  // Stable event handler refs — identity must not change across renders so that
  // removeEventListener can match the same function that was passed to add.
  const handlers = useRef({
    onTimeUpdate(this: HTMLVideoElement) {
      setState((s) => ({ ...s, currentTime: this.currentTime }));
    },
    onDurationChange(this: HTMLVideoElement) {
      setState((s) => ({ ...s, duration: this.duration || 0 }));
    },
    onPlay() {
      setState((s) => ({ ...s, paused: false }));
    },
    onPause() {
      setState((s) => ({ ...s, paused: true }));
    },
    onRateChange(this: HTMLVideoElement) {
      setState((s) => ({ ...s, playbackRate: this.playbackRate }));
    },
  });

  // Ref callback: called by React with the element on mount and null on unmount.
  // This is the only reliable way to attach listeners to a conditionally-rendered
  // element — useEffect([]) runs before the element exists in the DOM.
  const setVideoEl = useCallback((el: HTMLVideoElement | null) => {
    const prev = videoElRef.current;
    const h = handlers.current;
    if (prev) {
      prev.removeEventListener("timeupdate", h.onTimeUpdate);
      prev.removeEventListener("durationchange", h.onDurationChange);
      prev.removeEventListener("play", h.onPlay);
      prev.removeEventListener("pause", h.onPause);
      prev.removeEventListener("ratechange", h.onRateChange);
    }
    videoElRef.current = el;
    if (el) {
      el.addEventListener("timeupdate", h.onTimeUpdate);
      el.addEventListener("durationchange", h.onDurationChange);
      el.addEventListener("play", h.onPlay);
      el.addEventListener("pause", h.onPause);
      el.addEventListener("ratechange", h.onRateChange);
      // Sync initial state in case the element already has data (e.g. src swap).
      setState({
        currentTime: el.currentTime,
        duration: el.duration || 0,
        paused: el.paused,
        playbackRate: el.playbackRate,
      });
    } else {
      setState({ currentTime: 0, duration: 0, paused: true, playbackRate: 1 });
    }
  }, []); // stable — never recreated, so <video ref={setVideoEl}> won't re-fire

  const seekTo = useCallback((t: number) => {
    const video = videoElRef.current;
    if (!video) return;
    video.currentTime = t;
  }, []);

  // Seek to t, then play once the seek completes (avoids the race where play()
  // is called before the browser has moved to the new position).
  const seekAndPlay = useCallback((t: number) => {
    const video = videoElRef.current;
    if (!video) return;
    video.addEventListener("seeked", () => video.play().catch(() => {}), {
      once: true,
    });
    video.currentTime = t;
  }, []);

  const play = useCallback(() => {
    videoElRef.current?.play().catch(() => {});
  }, []);

  const pause = useCallback(() => {
    videoElRef.current?.pause();
  }, []);

  const togglePlay = useCallback(() => {
    const video = videoElRef.current;
    if (!video) return;
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  }, []);

  const setPlaybackRate = useCallback((rate: number) => {
    if (videoElRef.current) videoElRef.current.playbackRate = rate;
  }, []);

  return [
    state,
    { setVideoEl, seekTo, seekAndPlay, play, pause, togglePlay, setPlaybackRate },
  ];
}