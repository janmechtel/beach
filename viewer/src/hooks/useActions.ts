import { useCallback, useState } from "react";
import { loadActions } from "../lib/api";
import type { Action } from "../lib/types";

export interface ActionsState {
  actions: Action[];
  error: string | null;
}

export interface ActionsControls {
  load: (stem: string, filename: string) => Promise<void>;
  reset: () => void;
}

/** Load a single action JSON file. Read-only — saving is not supported in the hosted viewer. */
export function useActions(): [ActionsState, ActionsControls] {
  const [state, setState] = useState<ActionsState>({
    actions: [],
    error: null,
  });

  const load = useCallback(async (stem: string, filename: string) => {
    setState((s) => ({ ...s, error: null }));
    try {
      const actions = await loadActions(stem, filename);
      setState({ actions, error: null });
    } catch (e) {
      setState((s) => ({
        ...s,
        error: e instanceof Error ? e.message : String(e),
      }));
    }
  }, []);

  const reset = useCallback(() => {
    setState({ actions: [], error: null });
  }, []);

  return [state, { load, reset }];
}
