# AO* Implementation

Among other things: our specific implementation
- always creates `AND` nodes together with the `OR` childrens of these `AND` nodes, so `find()` will always `expand()` `OR` nodes and never `AND` nodes.
- does not `backtrack()` leaf nodes; rather, always `backtrack()`s nodes exactly two levels above leaves

TODO: explain a bit more