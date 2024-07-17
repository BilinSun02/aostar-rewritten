# Glossary

Many words come from the Lean community and are labelled "(Lean)". All block quotes are from https://leanprover-community.github.io/glossary.html. I avoid words like "obligation": "obligation" is not a real word in the Lean community, and it will be puzzling as to how it would be defined in relation to "goal" which *is* a real word there.


## term
In "type theory per se", a *term* is an "instance" of a type. As per the Curry-Howard correspondence, propositions can be treated as types, and if we can find a term of the type, we can say we have found a proof to the proposition. Therefore, a term simultaneously serves as a proof to a proposition.


## declaration (Lean)
A declaration is
> A single Lean runtime object within a Lean environment.
> Or, ambiguously, any of a number of Lean commands which may define or declare such objects.
> Examples of such commands are the def, theorem, constant or example commands (and in Lean 3, the lemma command), amongst others.

Or we could just say a "declared object".


## working environment (Lean)
The *working environment* (or just *environment*) is the collection of declarations.


## goal (Lean)
A *goal* is,
> Within the context of interactively proving theorems in Lean, each targeted statement whose proof is in-progress.
> Or more broadly, type theoretically, an individual type for which a term is to be exhibited.

More concretely, I define a *goal* to be something that looks like the following in the Lean infoview, or any object that carries equivalent information:
```
a b c : ℕ
h₀: a ≤ b
h₁: b ≤ c
⊢ a ≤ c
```
as in
```
1 goal
a b c : ℕ
h₀: a ≤ b
h₁: b ≤ c
⊢ a ≤ c
```

Note that this includes the working environment and the hypotheses.


## hypothesis
Something like
```
h₀: a ≤ b
```
as in
```
theorem trans (a b c : ℕ) (h₀: a ≤ b) (h₁: b ≤ c) : a ≤ c
```

Pedantically, a *hypothesis* is a proof $p$ to the premise $P$ of an implication-styled proposition $P\rightarrow Q$; interpreted the other way, an argument `p` to the parameter `P` in a function prototype `P→Q`.


## and–or tree
https://en.wikipedia.org/wiki/And%E2%80%93or_tree
See `aostar.md` for more details on how our and-or tree is implemented.


## AND nodes
In an and-or tree, a node that's not solved until *all* children are solved.
See `aostar.md` for more details on how our and-or tree is implemented.

## OR nodes
In an and-or tree, a node that's solved when *any* child is solved.
See `aostar.md` for more details on how our and-or tree is implemented.