// TypeScript types matching beach/models.py Action model and server API responses.

export type PlayerID = "P1" | "P2" | "P3" | "P4";

export type ActionType =
  | "Serve"
  | "Reception"
  | "Set"
  | "Attack"
  | "Dig"
  | "Block"
  | "Free Ball Sent"
  | "Free Ball Received";

export const ACTION_TYPES: ActionType[] = [
  "Serve",
  "Reception",
  "Set",
  "Attack",
  "Dig",
  "Block",
  "Free Ball Sent",
  "Free Ball Received",
];

export const PLAYER_IDS: PlayerID[] = ["P1", "P2", "P3", "P4"];

export interface Action {
  timestamp_sec: number;
  player_id: PlayerID;
  action: ActionType;
  player_description?: string | null;
}

export interface PlayersJson {
  [playerId: string]: {
    name: string;
    description: string;
    team?: string;
    color?: string;
  };
}
