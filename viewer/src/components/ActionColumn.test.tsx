import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => {
  const actions = [
    { timestamp_sec: 10, player_id: "P1", action: "Serve" },
    { timestamp_sec: 16, player_id: "P2", action: "Reception" },
  ] as const;

  return {
    actions,
    controls: {
      load: vi.fn(async () => {}),
      update: vi.fn(),
      remove: vi.fn(),
      save: vi.fn(async () => {}),
      reset: vi.fn(),
    },
  };
});

vi.mock("../hooks/useActions", () => ({
  useActions: () => [
    {
      actions: [...mocks.actions],
      dirty: false,
      saveState: "idle",
      error: null,
    },
    mocks.controls,
  ],
}));

import ActionColumn from "./ActionColumn";

describe("ActionColumn", () => {
  it("starts clip playback when clicking a filtered action row", () => {
    const onPlayClip = vi.fn();

    render(
      <ActionColumn
        stem="clip"
        availableFiles={["actions.json"]}
        initialFile="actions.json"
        currentTime={0}
        onPlayClip={onPlayClip}
        secondsBefore={0}
        secondsAfter={2}
        repeatMode={false}
        canClose={false}
        onClose={() => {}}
      />
    );

    fireEvent.click(screen.getAllByTitle("Click to play clip · Double-click to edit")[0]);

    expect(onPlayClip).toHaveBeenCalledTimes(1);
    expect(onPlayClip).toHaveBeenCalledWith(10, 2, expect.any(Function));
  });
});
