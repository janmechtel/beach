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
  /**
   * Seek to `start`, then play. If `durationSec` is provided, pause after
   * that many seconds of playback and call `onEnd`. Uses setTimeout (not
   * timeupdate polling) so the stop is accurate regardless of event rate.
   *
   * The timer is cancelled automatically if the user manually pauses or scrubs
   * before it fires. Playback-rate changes reschedule it proportionally.
   */
  seekAndPlay: (start: number, durationSec?: number | null, onEnd?: () => void) => void;
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

  // --- Clip-stop timer state (all mutable, never triggers re-render) ---
  // The setTimeout handle for the pending auto-stop.
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Wall-clock time (ms) at which the timer was set.
  const timerSetAtRef = useRef<number>(0);
  // Total timer duration (ms) as originally scheduled.
  const timerDurationRef = useRef<number>(0);
  // Remaining duration (ms) after a rate-change reschedule.
  const timerRemainingRef = useRef<number>(0);
  // Callback to invoke when the stop fires.
  const onEndRef = useRef<(() => void) | null>(null);
  // Guards onSeeked: true while seekAndPlay is performing its own seek so the
  // stable onSeeked handler doesn't cancel the timer we're about to arm.
  const programmaticSeekRef = useRef(false);
  // Ignore exactly one pause event triggered by our timer callback. Without this,
  // that pause can arrive after the next clip arms its timer and cancel it.
  const ignoreNextPauseRef = useRef(false);

  const [state, setState] = useState<VideoPlayerState>({
    currentTime: 0,
    duration: 0,
    paused: true,
    playbackRate: 1,
  });

  /** Cancel any pending auto-stop timer and clear associated refs.
   * Only wipes onEndRef when a live timer is actually cancelled — if the timer
   * already fired (stopTimerRef is null), onEnd may belong to a new clip.
   */
  function clearStopTimer() {
    if (stopTimerRef.current !== null) {
      clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
      onEndRef.current = null;
      timerRemainingRef.current = 0;
    }
  }

  /** Arm a new stop timer for `remainingMs` milliseconds from now. */
  function armTimer(remainingMs: number) {
    if (stopTimerRef.current !== null) clearTimeout(stopTimerRef.current);
    timerSetAtRef.current = Date.now();
    timerDurationRef.current = remainingMs;
    timerRemainingRef.current = remainingMs;
    stopTimerRef.current = setTimeout(() => {
      stopTimerRef.current = null;
      const video = videoElRef.current;
      if (video) {
        // Pause ends the current clip; the resulting pause event should not clear
        // a timer that the onEnd callback may arm for the next clip.
        ignoreNextPauseRef.current = true;
        video.pause();
      }
      const cb = onEndRef.current;
      onEndRef.current = null;
      cb?.();
    }, remainingMs);
  }

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
      // Manual pauses should cancel pending clip timers so stale callbacks do not
      // fire later. Ignore the pause event emitted by our own timer callback.
      if (ignoreNextPauseRef.current) {
        ignoreNextPauseRef.current = false;
      } else {
        clearStopTimer();
      }
      setState((s) => ({ ...s, paused: true }));
    },
    onSeeked() {
      // User scrubbed — cancel pending stop timer (target time is now stale).
      // Skip when the seek is from seekAndPlay itself.
      if (!programmaticSeekRef.current) clearStopTimer();
    },
    onRateChange(this: HTMLVideoElement) {
      // If a stop timer is pending, reschedule it for the new rate so the
      // remaining clip wall-time is correct.
      if (stopTimerRef.current !== null) {
        const elapsed = Date.now() - timerSetAtRef.current;
        const remainingReal = Math.max(0, timerRemainingRef.current - elapsed);
        // remainingReal is in "real ms at the old rate" — convert to video-time
        // then back to real-time at the new rate. But timerRemainingRef already
        // tracks real-ms, so we just re-arm with the unconsumed remainder.
        armTimer(remainingReal);
      }
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
      prev.removeEventListener("seeked", h.onSeeked);
      prev.removeEventListener("ratechange", h.onRateChange);
    }
    clearStopTimer();
    videoElRef.current = el;
    if (el) {
      el.addEventListener("timeupdate", h.onTimeUpdate);
      el.addEventListener("durationchange", h.onDurationChange);
      el.addEventListener("play", h.onPlay);
      el.addEventListener("pause", h.onPause);
      el.addEventListener("seeked", h.onSeeked);
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

  const seekAndPlay = useCallback(
    (start: number, durationSec?: number | null, onEnd?: () => void) => {
      const video = videoElRef.current;
      if (!video) return;

      // Cancel any prior clip that was still pending.
      clearStopTimer();

      // Store the callback before seeking so the seeked handler (which runs
      // immediately on some browsers) doesn't clear it.
      if (durationSec != null && durationSec > 0) {
        onEndRef.current = onEnd ?? null;
        // Timer is armed after seek completes (see seeked handler below), using
        // the video's actual playbackRate at that moment.
      }

      // One-shot seeked listener: start playing once the browser has repositioned,
      // then arm the stop timer (if a duration was requested).
      video.addEventListener(
        "seeked",
        () => {
          programmaticSeekRef.current = false;
          video.play().catch(() => {});
          if (durationSec != null && durationSec > 0) {
            // Convert video-seconds to real-ms at the current playback rate.
            armTimer((durationSec / video.playbackRate) * 1000);
          }
        },
        { once: true }
      );

      programmaticSeekRef.current = true;
      video.currentTime = start;
    },
    [] // stable: only uses refs, never closes over state
  );

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
