# AO* Implementation

Among other things: our specific implementation
- has "layers" that are "homogeneous" in the sense that, for any given depth $n$, nodes of depth $n$ from the root are either all `AND` nodes or all `OR` nodes
- keeps one tactic "step" (literally one line of Lean 3 code) in each `AND` node; from the root down, along each path, if we accumulate the `AND` node steps in a string, these string piece together to form a partial proof that can be run on Lean 3. Lean 3 then outputs a few goals; each goal is made a child node of the `AND` node; hence all children of the `AND` node must be satisfied for the `AND` node to be satisfied
- keeps "goal" information (see `terminology.md` for my definition of "goal") in each `OR` node; for this goal, multiple possible next tactic steps can be proposed; each such tactic is made a child node of the `OR` node; hence any child being solved results in the `OR` node being solved
  - The information kept in `OR` nodes is not essential: we can always accumulate the `AND` node tactics to form the partial proof and run that partial proof on Lean to obtain the goals again.
- makes the root node an `AND` node. This node holds the definition
- always creates `AND` nodes together with the `OR` children of these `AND` nodes, so `find()` will always `expand()` `OR` nodes and never `AND` nodes.
- does not `backtrack()` leaf nodes; rather, always `backtrack()`s nodes exactly two levels above leaves
- every `AND` node holds a tactic, except the root `AND` which holds the "beginning" of a theorem that looks like
    ```
    theorem ... :=
    ```
    `expand()` furnishes a line `begin\n`. Then the LLM is prompted for tactics. Each `AND` node's tactic, as a string, will NOT have indentation; `find()` will take care of the indentation. `expand()` also furnishes the `end` at the end.

TODO: explain a bit more