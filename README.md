# Sokoban solver — room-decomposed push-optimal search

Solves each level room by room: the generator's levels are a 1-wide hallway
plus 1–4 rooms, and a box can never leave its room, so the level optimum is
the sum of the rooms' optima. Each room is solved with a push-based
breadth-first search (first solution found is push-optimal), with dead-square
and frozen 2×2-block pruning. Search is bounded by work (state caps), never
by the clock, so output is deterministic across runs.

Run: `node main.ts` reads `/input/<id>/level.txt` and writes
`/output/<id>/solution.txt` (moves in LURD, upper-case letters are pushes).
