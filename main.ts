// Room-decomposed push-optimal Sokoban solver.
//
// Strategy: the generated levels are a 1-wide hallway plus 1-4 rooms, each room
// holding 3-4 boxes that can never leave it (a box pushed into the hallway can
// never come back). So the level optimum is the sum of the rooms' optima, and
// each room is solved on its own with a push-based breadth-first search — the
// first solution a BFS over pushes finds is push-optimal for that room.
//
// A push state is the sorted box squares plus the player's reachable region
// (named by its lowest square). Pruning: a box may only sit on a "live"
// square (one from which pull-back from a goal can reach it), and a push that
// leaves a box in a frozen 2x2 block of walls/boxes off its goal is dropped.
// Search is bounded by work (state caps), never by the clock, so two runs
// produce byte-identical output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (state counts, never wall time): rooms here are kept solvable
// well inside these caps by the generator.
const ROOM_STATE_CAP = 400_000;
const WHOLE_STATE_CAP = 600_000;

// Directions: [dx, dy, walk letter, push letter].
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares a box can stand on and still be pushable onto some goal, by pull-back from the goals. */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Flood fill from `from` avoiding walls and `occupied`; returns the lowest reachable square. */
function regionLow(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    const up = p - w;
    const down = p + w;
    const left = p - 1;
    const right = p + 1;
    if (wall[up] === 0 && occupied[up] === 0 && seen[up] !== stamp) { seen[up] = stamp; stack[n++] = up; }
    if (wall[down] === 0 && occupied[down] === 0 && seen[down] !== stamp) { seen[down] = stamp; stack[n++] = down; }
    if (wall[left] === 0 && occupied[left] === 0 && seen[left] !== stamp) { seen[left] = stamp; stack[n++] = left; }
    if (wall[right] === 0 && occupied[right] === 0 && seen[right] !== stamp) { seen[right] = stamp; stack[n++] = right; }
  }
  return low;
}

/** True when the box just pushed to `t` is stuck in a 2x2 block of walls/boxes with a box off its goal. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  const corners = [t, t - 1, t - w, t - w - 1];
  for (const a of corners) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) {
      return true;
    }
  }
  return false;
}

/** Shortest walk from `from` to `to` around walls and boxes, lower-case letters, fixed direction order. */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square unreachable');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** Player square after following `moves` (case-insensitive) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

interface Solution {
  pushes: number;
  moves: string;
}

/**
 * Push-optimal solve of `level` by BFS over pushes. Returns null when the
 * state cap is hit or the level is unsolvable. The first solved state a BFS
 * generates is push-optimal.
 */
function solvePushOptimal(level: Level, maxStates: number): Solution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b])) return null;

  // Manhattan distance from each square to its nearest goal (for push ordering).
  const goalDist = new Int32Array(size).fill(-1);
  const dq: number[] = [...goals];
  for (const g of goals) goalDist[g] = 0;
  for (let qi = 0; qi < dq.length; qi++) {
    const p = dq[qi]!;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (!wall[q] && goalDist[q] < 0) {
        goalDist[q] = goalDist[p]! + 1;
        dq.push(q);
      }
    }
  }
  const heuristic = (boxes: number[]): number => {
    let s = 0;
    for (const b of boxes) s += goalDist[b] < 0 ? 999 : goalDist[b]!;
    return s;
  };

  const keyOf = (boxes: number[], low: number): bigint => {
    let k = BigInt(low);
    for (const b of boxes) k = k * 256n + BigInt(b);
    return k;
  };

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1]; // box square before the push that made this state
  const pushTo: number[] = [-1]; // box square after that push
  const index = new Map<bigint, number>();
  for (const b of level.boxes) occupied[b] = 1;
  index.set(keyOf(level.boxes, regionLow(level, occupied, level.player, seen, ++stamp, stack)), 0);
  for (const b of level.boxes) occupied[b] = 0;

  let found = solved(level.boxes) ? 0 : -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const boxes = statesBoxes[head]!;
    for (const b of boxes) occupied[b] = 1;
    regionLow(level, occupied, statesPlayer[head]!, seen, ++stamp, stack);
    const here = stamp;
    // Collect this state's pushes, best-first by goal-distance heuristic.
    const options: { bi: number; from: number; to: number; d: number; score: number }[] = [];
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        options.push({ bi, from: b, to: t, d, score: goalDist[t]! - goalDist[b]! });
      }
    }
    options.sort((a, b) => a.score - b.score || a.from - b.from || a.to - b.to);
    for (const opt of options) {
      if (found >= 0) break;
      const { bi, from, to } = opt;
      occupied[from] = 0;
      occupied[to] = 1;
      const frozen = frozenBlock(level, occupied, goal, to);
      let key = -1n;
      let next: number[] | null = null;
      if (!frozen) {
        next = boxes.slice();
        next[bi] = to;
        next.sort((x, y) => x - y);
        key = keyOf(next, regionLow(level, occupied, from, mark, ++stamp, stack));
      }
      occupied[to] = 0;
      occupied[from] = 1;
      if (frozen || index.has(key)) continue;
      if (statesBoxes.length >= maxStates) {
        for (const x of boxes) occupied[x] = 0;
        return null;
      }
      index.set(key, statesBoxes.length);
      statesBoxes.push(next!);
      statesPlayer.push(from);
      parent.push(head);
      pushFrom.push(from);
      pushTo.push(to);
      if (solved(next!)) {
        found = statesBoxes.length - 1;
        break;
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = statesBoxes[s]!;
  }
  return { pushes: path.length, moves };
}

interface Room {
  boxes: number[];
  goals: number[];
}

/**
 * Split the level into rooms by cutting the dead squares of the player's row
 * (the hallway: a box there reaches no goal). Returns null when the level is
 * not a clean set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve room by room; fall back to a whole-level search when that fails. */
function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    const done: boolean[] = rooms.map(() => false);
    for (let ri = 0; ri < rooms.length; ri++) {
      const room = rooms[ri]!;
      const wall = level.wall.slice();
      for (let oj = 0; oj < rooms.length; oj++) {
        if (oj === ri) continue;
        const other = rooms[oj]!;
        for (const b of done[oj] ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      // A room the first cap misses gets one retry with a much larger cap
      // before the (far harder) whole-level search is tried: the scored set's
      // rooms are drawn the same way, but a retry costs nothing when unused.
      let solved = solvePushOptimal(sub, ROOM_STATE_CAP);
      if (solved === null) solved = solvePushOptimal(sub, ROOM_STATE_CAP * 4);
      if (solved === null) return solveWhole(level);
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      done[ri] = true;
    }
    return moves;
  }
  return solveWhole(level);
}

function solveWhole(level: Level): string {
  const solved = solvePushOptimal(level, WHOLE_STATE_CAP);
  return solved === null ? '' : solved.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
