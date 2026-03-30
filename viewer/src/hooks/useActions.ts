import { useCallback, useState } from "react";
import { loadActions, saveActions } from "../lib/api";
import type { Action } from "../lib/types";

export type SaveState = "idle" | "saving" | "saved" | "error";

export interface ActionsState {
  actions: Action[];
  dirty: boolean;
  saveState: SaveState;
  error: string | null;
}

export interface ActionsControls {
  load: (stem: string, filename: string) => Promise<void>;
  update: (index: number, patch: Partial<Action>) => void;
  remove: (index: number) => void;
  save: (stem: string, filename: string) => Promise<void>;
  reset: () => void;
}

/** Load, edit, and save a single action JSON file. */
export function useActions(): [ActionsState, ActionsControls] {
  const [state, setState] = useState<ActionsState>({
    actions: [],
    dirty: false,
    saveState: "idle",
    error: null,
  });

  const load = useCallback(async (stem: string, filename: string) => {
    setState((s) => ({ ...s, error: null }));
    try {
      const actions = await loadActions(stem, filename);
      setState({ actions, dirty: false, saveState: "idle", error: null });
    } catch (e) {
      setState((s) => ({
        ...s,
        error: e instanceof Error ? e.message : String(e),
      }));
    }
  }, []);

  const update = useCallback((index: number, patch: Partial<Action>) => {
    setState((s) => {
      const actions = [...s.actions];
      actions[index] = { ...actions[index], ...patch };
      return { ...s, actions, dirty: true };
    });
  }, []);

  const remove = useCallback((index: number) => {
    setState((s) => {
      const actions = s.actions.filter((_, i) => i !== index);
      return { ...s, actions, dirty: true };
    });
  }, []);

  const save = useCallback(async (stem: string, filename: string) => {
    // Read current actions inside setState to avoid stale closure —
    // useCallback deps can't track state that changes between renders.
    let currentActions: Action[] = [];
    setState((s) => { currentActions = s.actions; return { ...s, saveState: "saving" }; });
    try {
      const sorted = [...currentActions].sort(
        (a, b) => a.timestamp_sec - b.timestamp_sec
      );
      await saveActions(stem, filename, sorted);
      setState((s) => ({
        ...s,
        actions: sorted,
        dirty: false,
        saveState: "saved",
      }));
      setTimeout(() => setState((s) => ({ ...s, saveState: "idle" })), 1500);
    } catch (e) {
      setState((s) => ({
        ...s,
        saveState: "error",
        error: e instanceof Error ? e.message : String(e),
      }));
    }
  }, []);

  const reset = useCallback(() => {
    setState({ actions: [], dirty: false, saveState: "idle", error: null });
  }, []);

  return [state, { load, update, remove, save, reset }];
}
