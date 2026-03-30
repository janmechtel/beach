import { useCallback, useEffect, useRef, useState } from "react";

export interface VideoPlayerState {
  currentTime: number;
  duration: number;
  paused: boolean;
  playbackRate: number;
}

export interface VideoPlayerControls {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  seekTo: (t: number) => void;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  setPlaybackRate: (rate: number) => void;
}

/** Manages a shared video element. Returns live state and imperative controls.
 *
 * seekTo attaches a `seeked` listener BEFORE setting currentTime to avoid
 * the race condition where seeked fires before the listener is registered
 * (documented in memory: "attach seeked listener before currentTime").
 */
export function useVideoPlayer(): [VideoPlayerState, VideoPlayerControls] {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [state, setState] = useState<VideoPlayerState>({
    currentTime: 0,
    duration: 0,
    paused: true,
    playbackRate: 1,
  });

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onTimeUpdate = () =>
      setState((s) => ({ ...s, currentTime: video.currentTime }));
    const onDurationChange = () =>
      setState((s) => ({ ...s, duration: video.duration || 0 }));
    const onPlay = () => setState((s) => ({ ...s, paused: false }));
    const onPause = () => setState((s) => ({ ...s, paused: true }));
    const onRateChange = () =>
      setState((s) => ({ ...s, playbackRate: video.playbackRate }));

    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("durationchange", onDurationChange);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("ratechange", onRateChange);

    return () => {
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("durationchange", onDurationChange);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("ratechange", onRateChange);
    };
  }, []);

  const seekTo = useCallback((t: number) => {
    const video = videoRef.current;
    if (!video) return;
    // Attach seeked listener FIRST, then set currentTime (avoids race).
    video.addEventListener(
      "seeked",
      () => {
        // Caller decides whether to play after seek; this just notifies.
      },
      { once: true }
    );
    video.currentTime = t;
  }, []);

  const play = useCallback(() => {
    videoRef.current?.play().catch(() => {});
  }, []);

  const pause = useCallback(() => {
    videoRef.current?.pause();
  }, []);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  }, []);

  const setPlaybackRate = useCallback((rate: number) => {
    if (videoRef.current) videoRef.current.playbackRate = rate;
  }, []);

  return [
    state,
    { videoRef, seekTo, play, pause, togglePlay, setPlaybackRate },
  ];
}
