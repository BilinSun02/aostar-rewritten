# Mostly adapted from the DeepSeek-Prover codebase

# This module post-processes Lean 4 verifier / REPL JSON output (command ASTs, tactics,
# premises) into Python-friendly dictionaries with extracted source spans (line/column
# and character offsets) and string slices from the original Lean file.

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Iterable

line_break_regex = re.compile(r'(?<=\n)')

# --- Typing helpers -----------------------------------------------------------

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
AST = JSON  # Lean 4 AST nodes are represented as JSON-like dict/list structures.

# A (pos, endPos) span as emitted by Lean, typically byte offsets into UTF-8 text.
ByteSpan = Tuple[Optional[int], Optional[int]]

# A source span in *character* offsets + line/column (1-indexed line/column).
# We keep it as a tuple when returning from low-level helpers, and project into dicts
# in higher-level parsers.
CharSpan = Tuple[
    Optional[str],  # extracted substring (or None)
    Optional[int],  # start line
    Optional[int],  # start column
    Optional[int],  # end line
    Optional[int],  # end column
    Optional[int],  # start char index (0-based)
    Optional[int],  # end char index (0-based)
]

def process_lean_file(file_contents: str, byte_idx_1: int, byte_idx_2: int) -> Tuple[str, int, int, int, int, int, int]:
    """
    Slice the original Lean source text using **Lean byte offsets** (UTF-8) and return
    the extracted substring plus (line, column) and (character-index) endpoints.

    Lean's JSON AST uses `pos` / `endPos` as byte offsets into the source file.
    This helper converts those byte offsets into:
      - 1-indexed line/column pairs, and
      - 0-based character indices in the Python `str`.

    Lean example (conceptual):
    ```lean
    def add1 (n : Nat) : Nat := n + 1
    -- Suppose Lean reports pos/endPos covering "n + 1".
    ```
    Calling `process_lean_file(file_contents, pos, endPos)` returns the substring
    `"n + 1"` along with its precise span.
    """
    def get_line(lines: Sequence[str], line_number: int) -> str:
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]
        else:
            raise IndexError("Line number out of range")
    def convert_pos(lines: Sequence[str], byte_idx: int) -> Tuple[int, int]:
        num_bytes = [len(line.encode('utf-8')) for line in lines]
        n = 0
        for i, num_bytes_in_line in enumerate(num_bytes, start=1):
            n += num_bytes_in_line
            if n > byte_idx:
                line_byte_idx = byte_idx - (n - num_bytes_in_line)
                if line_byte_idx == 0:
                    return i, 1

                line = get_line(lines, i)
                m = 0

                for j, c in enumerate(line, start=1):
                    m += len(c.encode("utf-8"))
                    if m >= line_byte_idx:
                        return i, j + 1
        return len(lines), len(lines[-1])
    def extract_string_between_positions(lines: Sequence[str], byte_idx_1: int, byte_idx_2: int) -> str:
        line_1, column_1 = convert_pos(lines, byte_idx_1)
        line_2, column_2 = convert_pos(lines, byte_idx_2)

        extracted_string = []

        if line_1 == line_2:
            # If both positions are in the same line
            substring = lines[line_1 - 1][column_1 - 1:column_2 - 1]
            extracted_string.append(substring)
        else:
            # If positions span multiple lines
            extracted_string.append(lines[line_1 - 1][column_1 - 1:])
            for line in range(line_1 + 1, line_2):
                extracted_string.append(lines[line - 1])
            extracted_string.append(lines[line_2 - 1][:column_2 - 1])

        return ''.join(extracted_string)
    def convert_line_col_to_char_idx(lines: Sequence[str], line: int, col: int) -> int:
        char_idx = 0
        for i in range(line - 1):
            char_idx += len(lines[i])
        char_idx += col - 1
        return char_idx

    lines = line_break_regex.split(file_contents)

    line_1, column_1 = convert_pos(lines, byte_idx_1)
    line_2, column_2 = convert_pos(lines, byte_idx_2)

    extracted_string = extract_string_between_positions(lines, byte_idx_1, byte_idx_2)

    char_idx_1 = convert_line_col_to_char_idx(lines, line_1, column_1)
    char_idx_2 = convert_line_col_to_char_idx(lines, line_2, column_2)

    return extracted_string, line_1, column_1, line_2, column_2, char_idx_1, char_idx_2

def extract_positions(node: AST) -> List[ByteSpan]:
    """
    Recursively collect all `(pos, endPos)` spans in a Lean AST subtree.

    Lean example (conceptual):
    ```lean
    theorem t : True := by trivial
    ```
    If `node` is the JSON AST for the theorem command, this returns a list of
    byte spans for tokens/subnodes inside that command, which higher-level code
    uses to compute the overall start/end of the declaration.
    """
    positions = []
    if isinstance(node, dict):
        if 'info' in node and 'original' in node['info']:
            info = node['info']['original']
            positions.append((info.get('pos'), info.get('endPos')))
        for key, value in node.items():
            positions.extend(extract_positions(value))
    elif isinstance(node, list):
        for item in node:
            positions.extend(extract_positions(item))
    return positions

def extract_vals(data: AST) -> List[str]:
    """
    Recursively collect and strip all `"val"` fields in a Lean AST subtree.

    Lean example (conceptual):
    ```lean
    theorem t (n : Nat) : n = n := by rfl
    ```
    On the subtree corresponding to the binder `(n : Nat)`, this returns pieces
    like `["n", ":", "Nat"]` (exact tokenization depends on Lean), which callers
    join into a readable snippet.
    """
    vals = []
    if isinstance(data, dict):
        if "val" in data:
            vals.append(data["val"].strip())
        for value in data.values():
            vals.extend(extract_vals(value))
    elif isinstance(data, list):
        for item in data:
            vals.extend(extract_vals(item))
    return vals

def find_doccomment_vals(data: AST) -> Tuple[List[str], List[ByteSpan]]:
    """
    Find doc-comment nodes (`Lean.Parser.Command.docComment`) and return:
      1) the raw doc-comment text fragments (`vals`)
      2) their `(pos, endPos)` byte spans

    Lean example:
    ```lean
    /-- Adds one to a natural number. -/
    def add1 (n : Nat) : Nat := n + 1
    ```
    When given the AST for the `def` command, this returns the doc comment text
    (the `/-- ... -/` content) and spans covering that comment.
    """
    vals = []
    positions = []

    if isinstance(data, dict):
        if data.get("kind") == "Lean.Parser.Command.docComment":
            args = data.get("args", [])
            for arg in args:
                if "atom" in arg:
                    vals.append(arg["atom"]["val"])
                    positions.append((arg["atom"]["info"]["original"]["pos"], arg["atom"]["info"]["original"]["endPos"]))
        for value in data.values():
            v, p = find_doccomment_vals(value)
            vals.extend(v)
            positions.extend(p)
    elif isinstance(data, list):
        for item in data:
            v, p = find_doccomment_vals(item)
            vals.extend(v)
            positions.extend(p)

    return vals, positions

def find_attributes_vals(data: AST) -> Tuple[List[str], List[ByteSpan]]:
    """
    Extract attribute syntax (e.g. `@[simp]`) from a declaration AST.

    Lean example:
    ```lean
    @[simp] theorem t (n : Nat) : n = n := by rfl
    ```
    Returns tokens/spans covering `@[simp]`.
    """
    vals = []
    positions = []

    if isinstance(data, dict):
        if data.get("kind") in ["Lean.Parser.Term.attributes"]:
            args = data.get("args", [])
            for arg in args:
                vals.extend(extract_vals(arg))
                positions.extend(extract_positions(arg))
        for value in data.values():
            v, p = find_attributes_vals(value)
            vals.extend(v)
            positions.extend(p)
    elif isinstance(data, list):
        for item in data:
            v, p = find_attributes_vals(item)
            vals.extend(v)
            positions.extend(p)

    return vals, positions

def find_pripro_vals(data: AST) -> Tuple[List[str], List[ByteSpan]]:
    """
    Extract `private` / `protected` modifiers from a declaration AST.

    Lean example:
    ```lean
    private theorem secret : True := by trivial
    ```
    Returns tokens/spans covering the `private` keyword.
    """
    vals = []
    positions = []

    if isinstance(data, dict):
        if data.get("kind") in ["Lean.Parser.Command.private", "Lean.Parser.Command.protected"]:
            args = data.get("args", [])
            for arg in args:
                vals.extend(extract_vals(arg))
                positions.extend(extract_positions(arg))
        for value in data.values():
            v, p = find_pripro_vals(value)
            vals.extend(v)
            positions.extend(p)
    elif isinstance(data, list):
        for item in data:
            v, p = find_pripro_vals(item)
            vals.extend(v)
            positions.extend(p)

    return vals, positions

def extract_other_vals(data: AST) -> List[str]:
    """
    Like `extract_vals`, but preserves `"val"` exactly (no `.strip()`).

    Lean example (conceptual):
    ```lean
    theorem t : True := by trivial
    ```
    Useful when you care about exact token text (including whitespace) as emitted
    in the AST.
    """
    vals = []
    if isinstance(data, dict):
        if "val" in data:
            vals.append(data["val"])
        for value in data.values():
            vals.extend(extract_other_vals(value))
    elif isinstance(data, list):
        for item in data:
            vals.extend(extract_other_vals(item))
    return vals

def find_kind_name_theorem_lemma_abbrev_def_instance_inductive(file_content: str, data: AST) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    From a declaration command AST, extract the *kind* keyword and declared *name*.

    Supported kinds include: `theorem`, `lemma`, `abbrev`, `def`, `instance`, `inductive`.

    Lean examples:
    ```lean
    theorem T : True := by trivial
    def f (n : Nat) := n + 1
    instance : Inhabited Nat := ⟨0⟩
    inductive MyNat | zero | succ (n : MyNat)
    ```
    For `theorem T ...`, returns kind `"theorem"` and name `"T"` plus spans.
    For anonymous `instance : ...`, `name` may be `None` depending on syntax.
    """
    kind = None
    name = None
    kind_pos = None
    kind_end = None
    name_pos = None
    name_end = None

    if isinstance(data, dict):
        if len(data.get("node", {}).get("args", [])) > 1:
            second_node=data["node"]["args"][1]["node"]
            if second_node["kind"]== "Lean.Parser.Command.instance":
                atom = second_node.get("args", [])[1].get("atom", {})
                kind = atom.get("val")
                kind_pos = atom.get("info", {}).get("original", {}).get("pos")
                kind_end = atom.get("info", {}).get("original", {}).get("endPos")
                for arg in second_node.get("args", []):
                    if arg.get("node"):
                        if arg.get("node")["args"]:
                            if arg.get("node")["args"][0].get("node",{}).get("kind") == "Lean.Parser.Command.declId":
                                ident = arg.get("node")["args"][0]["node"]["args"][0]["ident"]
                                name = ident.get("val")
                                # print(name)
                                name_pos = ident.get("info", {}).get("original", {}).get("pos")
                                # print(name_pos)
                                name_end = ident.get("info", {}).get("original", {}).get("endPos")
                                break
            else:
                atom = second_node.get("args", [])[0].get("atom", {})
                kind = atom.get("val")
                kind_pos = atom.get("info", {}).get("original", {}).get("pos")
                kind_end = atom.get("info", {}).get("original", {}).get("endPos")
                for arg in second_node.get("args", []):
                    if arg.get("node", {}).get("kind") == "Lean.Parser.Command.declId":
                        ident = arg.get("node", {}).get("args", [])[0].get("ident", {})
                        name = ident.get("val")
                        name_pos = ident.get("info", {}).get("original", {}).get("pos")
                        name_end = ident.get("info", {}).get("original", {}).get("endPos")
                        break

    if kind_pos and kind_end:
        kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2 = process_lean_file(file_content, kind_pos, kind_end)
    else:
        kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2 = None,None,None, None,None,None,None

    if name_pos and name_end:
        name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2 = process_lean_file(file_content, name_pos, name_end)
    else:
        name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2 = None,None,None, None,None,None,None
    return kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2, name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2

def find_statement_theorem_lemma_abbrev(file_content: str, data: AST) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[ByteSpan]]:
    """
    Extract binders/parameters and type-spec (the `: ...`) for theorems/lemmas/abbrevs.

    Lean example:
    ```lean
    theorem add_comm (a b : Nat) : a + b = b + a := by
      simpa [Nat.add_comm]
    ```
    Returns:
      - `parameters`: spans for `(a b : Nat)`
      - `Type`: span for `: a + b = b + a`
      - `statement_positions`: all byte spans inside the signature node.
    """
    explicitBinder_list = []
    type_list=[]
    second_node=data["node"]["args"][1]["node"]

    for arg in second_node.get("args", []):
        if arg.get("node", {}).get("kind") in ["Lean.Parser.Command.declSig", "Lean.Parser.Command.optDeclSig"]:
            node_declSig = arg.get("node")
    statement_positions= extract_positions(node_declSig)
    node_1=node_declSig["args"][0]["node"]
    node_2=node_declSig["args"][1]["node"] if len(node_declSig["args"]) > 1 else None
    for arg in node_1.get("args", []):
        if arg.get("node", {}).get("kind") in[ "Lean.Parser.Term.explicitBinder","Lean.Parser.Term.implicitBinder","Lean.Parser.Term.instBinder"]:
            node=arg.get("node", {})
            val=extract_vals(node)
            val=' '.join(val)
            positions=extract_positions(node)
            start_pos = min(pos[0] for pos in positions)
            end_pos = max(pos[1] for pos in positions)
            explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
            explicitBinder_list.append({
                "explicitBinder":{
                    "content":explicit,
                    "start_pos": {
                        "position" :explicit_char_idx_1,
                        "line" : explicit_line_1,
                        "column" : explicit_column_1
                    },
                    "end_pos": {
                        "position" : explicit_char_idx_2,
                        "line" : explicit_line_2,
                        "column" : explicit_column_2
                    }
                }
            })
    if node_2 is not None and second_node["kind"] in ["Lean.Parser.Command.theorem", "group", "Lean.Parser.Command.abbrev"]:
        if node_2.get("kind") in ["Lean.Parser.Term.typeSpec"]:
            node=node_2
            val=extract_vals(node)
            val=' '.join(val)
            positions=extract_positions(node)
            start_pos = min(pos[0] for pos in positions)
            end_pos = max(pos[1] for pos in positions)
            explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
            type_list.append({
                "typeSpec":{
                    "content":explicit,
                    "start_pos": {
                        "position" :explicit_char_idx_1,
                        "line" : explicit_line_1,
                        "column" : explicit_column_1
                    },
                    "end_pos": {
                        "position" : explicit_char_idx_2,
                        "line" : explicit_line_2,
                        "column" : explicit_column_2
                    }
                }
            })

    # print(explicitBinder_list,statement_positions)
    return explicitBinder_list, type_list, statement_positions

def find_proof(file_content: str, data: AST) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Extract the proof/value part of a declaration (`:= ...` or `where ...` forms
    represented by `declValSimple` / `declValEqns`).

    Lean example:
    ```lean
    theorem t : True := by trivial
    ```
    Returns the substring for `:= by trivial` (or the equivalent proof node) and spans.
    """
    vals = []
    positions = []
    if isinstance(data, dict):
        second_node = data["node"]["args"][1]["node"]
        for arg in second_node.get("args", []):
            if arg.get("node", {}).get("kind") in ["Lean.Parser.Command.declValSimple", "Lean.Parser.Command.declValEqns"]:
                vals.extend(extract_vals(arg))
                positions.extend(extract_positions(arg))
                break  # 找到后即可退出
    if positions:
        proof_start_pos = min(pos[0] for pos in positions)
        proof_end_pos = max(pos[1] for pos in positions)
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = process_lean_file(file_content, proof_start_pos, proof_end_pos)
    else:
        proof_start_pos = None
        proof_end_pos = None
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = None,None,None, None,None,None, None


    return proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2

def process_modifier(file_content: str, declaration: AST, tactics: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Given a declaration AST, extract common "modifier" components:
      - doc comment
      - attributes (`@[...]`)
      - `private` / `protected`
      - the full declaration span
      - and tactics that lie within the declaration span (if provided)

    Lean example:
    ```lean
    /-- doc -/
    @[simp] private theorem t : True := by
      trivial
    ```
    Returns the spans for the doc comment, `@[simp]`, `private`, the whole command,
    plus any tactic spans that fall inside the command.
    """
    doccomment_vals, doccomment_positions = find_doccomment_vals(declaration)
    if doccomment_vals:
        comment_start_pos = min(pos[0] for pos in doccomment_positions)
        comment_end_pos = max(pos[1] for pos in doccomment_positions)
        comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2 = process_lean_file(file_content,comment_start_pos,comment_end_pos)
    else:
        comment = None
        comment_start_pos = None
        comment_end_pos = None
        comment= None
        comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2 = None, None, None, None, None, None

    attributes_vals, attributes_positions = find_attributes_vals(declaration)
    if attributes_vals:
        attributes_start_pos = min(pos[0] for pos in attributes_positions)
        attributes_end_pos = max(pos[1] for pos in attributes_positions)
        attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2 = process_lean_file(file_content,attributes_start_pos,attributes_end_pos)

    else:
        attributes = None
        attributes_start_pos = None
        attributes_end_pos = None
        attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2 = None, None, None, None, None, None

    pripro_vals, pripro_positions = find_pripro_vals(declaration)
    if pripro_vals:
        pripro_start_pos = min(pos[0] for pos in pripro_positions)
        pripro_end_pos = max(pos[1] for pos in pripro_positions)
        pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2 = process_lean_file(file_content,pripro_start_pos,pripro_end_pos)
    else:
        pripro = None
        pripro_start_pos = None
        pripro_end_pos = None
        pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2 = None, None, None, None, None, None

    positions = extract_positions(declaration)
    positions = [(start, end) for start, end in positions if start is not None and end is not None]

    start_pos = min(pos[0] for pos in positions)
    end_pos = max(pos[1] for pos in positions)
    whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2 = process_lean_file(file_content, start_pos, end_pos)

    tactics_list = []
    for tactic in tactics:
        if start_pos <= tactic["pos"] <= end_pos:
            _, A,B,c,d, tactic["pos"], tactic["endPos"] =  process_lean_file(file_content, tactic["pos"], tactic["endPos"])
            tactics_list.append(tactic)

    return tactics_list, comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2, attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2, pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2, whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2

def theorem_lemma_abbrev(file_content: str, declaration: AST, tactics: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Parse a theorem/lemma/example/abbrev declaration into a Python-friendly dict.

    Lean example:
    ```lean
    @[simp] theorem t (n : Nat) : n = n := by
      rfl
    ```
    Produces a dict containing:
      - `kind`: "theorem"
      - `name`: "t" + span
      - `parameters`: binder spans
      - `Type`: type-spec span
      - `statement`: full signature span
      - `proof`: proof span
      - plus comment/attributes/private-protected/whole spans.
    """
    tactics_list, comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2, attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2, pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2, whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2= process_modifier(file_content, declaration,tactics)

    kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2,name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2  = find_kind_name_theorem_lemma_abbrev_def_instance_inductive(file_content, declaration)

    explicitBinder_list, type_list, statement_positions = find_statement_theorem_lemma_abbrev(file_content, declaration)
    positions = extract_positions(declaration)
    positions = [(start, end) for start, end in positions if start is not None and end is not None]
    if not positions:
        return {"kind" : kind}

    whole_start_pos = min(pos[0] for pos in positions)
    whole_end_pos = max(pos[1] for pos in positions)
    if statement_positions:
        statement_start_pos = whole_start_pos
        statement_end_pos = max(pos[1] for pos in statement_positions)
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 = process_lean_file(file_content, statement_start_pos, statement_end_pos)
    else:
        statement_start_pos = None
        statement_end_pos = None
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 =  None, None, None, None, None, None,None

    proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = find_proof(file_content, declaration)
    declaration_info = {
        "kind" : kind,
        "whole_start_pos" : {
            "position" : whole_char_idx_1,
            "line" : whole_line_1,
            "column" : whole_column_1
        },

        "whole_end_pos" :{
            "position" : whole_char_idx_2,
            "line" : whole_line_2,
            "column" : whole_column_2
        },

        "attributes": attributes,
        "comment": {
            "content": comment,
            "start_pos": {
                "position" : comment_char_idx_1,
                "line" : comment_line_1,
                "column" : comment_column_1
            },
            "end_pos": {
                "position" : comment_char_idx_2,
                "line" : comment_line_2,
                "column" : comment_column_2
            }
        },
        "private_protected": pripro,
        "name": {
            "content": name,
            "start_pos": {
                "position" : name_char_idx_1,
                "line" : name_line_1,
                "column" : name_column_1
            },
            "end_pos": {
                "position" : name_char_idx_2,
                "line" : name_line_2,
                "column" : name_column_2
            }
        },
        "parameters": explicitBinder_list,
        "Type":type_list,
        "statement":{
            "content": statement_val_str,
            "start_pos": {
                "position" :  statement_char_idx_1,
                "line" :  statement_line_1,
                "column" :  statement_column_1
            },
            "end_pos": {
                "position" :  statement_char_idx_2,
                "line" :  statement_line_2,
                "column" :  statement_column_2
            }
        },
        "proof" :{
            "content": proof,
            "start_pos": {
                "position" :  proof_char_idx_1,
                "line" :  proof_line_1,
                "column" :  proof_column_1
            },
            "end_pos": {
                "position" :  proof_char_idx_2,
                "line" :  proof_line_2,
                "column" :  proof_column_2
            }
        }
    }
    return declaration_info

def find_statement_def(file_content: str, data: AST) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[ByteSpan]]:
    """
    Extract binders/parameters and type-spec for `def` / `instance` declarations.

    Lean examples:
    ```lean
    def add1 (n : Nat) : Nat := n + 1
    instance : Inhabited Nat := ⟨0⟩
    ```
    For `def add1 ...`, returns binder spans `(n : Nat)` and type-spec `: Nat`.
    For an `instance`, treats the post-`:` type as the type-spec.
    """
    explicitBinder_list = []
    type_list=[]
    second_node=data["node"]["args"][1]["node"]

    for arg in second_node.get("args", []):
        if arg.get("node", {}).get("kind") in ["Lean.Parser.Command.declSig", "Lean.Parser.Command.optDeclSig"]:
            node_declSig = arg.get("node")
    statement_positions= extract_positions(node_declSig)
    node_1=node_declSig["args"][0]["node"]
    node_2=node_declSig["args"][1]["node"]
    for arg in node_1.get("args", []):
        if arg.get("node", {}).get("kind") in[ "Lean.Parser.Term.explicitBinder","Lean.Parser.Term.implicitBinder","Lean.Parser.Term.instBinder"]:
            node=arg.get("node", {})
            val=extract_vals(node)
            val=' '.join(val)
            positions=extract_positions(node)
            start_pos = min(pos[0] for pos in positions)
            end_pos = max(pos[1] for pos in positions)
            explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
            # explicitBinder_list.append({"explicitBinder": [explicit, explicit_char_idx_1,  explicit_line_1,  explicit_column_1, explicit_char_idx_2, explicit_line_2,  explicit_column_2]})
            explicitBinder_list.append({
                "explicitBinder":{
                    "content":explicit,
                    "start_pos": {
                        "position" :explicit_char_idx_1,
                        "line" : explicit_line_1,
                        "column" : explicit_column_1
                    },
                    "end_pos": {
                        "position" : explicit_char_idx_2,
                        "line" : explicit_line_2,
                        "column" : explicit_column_2
                    }
                }
            })

    if second_node["kind"] in[ "Lean.Parser.Command.definition"] :
        for arg in node_2.get("args", []):
            if arg.get("node", {}).get("kind") in ["Lean.Parser.Term.typeSpec"]:
                node=arg.get("node", {})
                val=extract_vals(node)
                val=' '.join(val)
                positions=extract_positions(node)
                start_pos = min(pos[0] for pos in positions)
                end_pos = max(pos[1] for pos in positions)
                explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
                type_list.append({
                "typeSpec":{
                    "content":explicit,
                    "start_pos": {
                        "position" :explicit_char_idx_1,
                        "line" : explicit_line_1,
                        "column" : explicit_column_1
                    },
                    "end_pos": {
                        "position" : explicit_char_idx_2,
                        "line" : explicit_line_2,
                        "column" : explicit_column_2
                    }
                }
            })

    elif second_node["kind"] in[ "Lean.Parser.Command.instance"]:
        val=extract_vals(node_2)
        val=' '.join(val)
        positions=extract_positions(node_2)
        start_pos = min(pos[0] for pos in positions)
        end_pos = max(pos[1] for pos in positions)
        explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
        type_list.append({
                "typeSpec":{
                    "content":explicit,
                    "start_pos": {
                        "position" :explicit_char_idx_1,
                        "line" : explicit_line_1,
                        "column" : explicit_column_1
                    },
                    "end_pos": {
                        "position" : explicit_char_idx_2,
                        "line" : explicit_line_2,
                        "column" : explicit_column_2
                    }
                }
            })
    return explicitBinder_list,type_list, statement_positions

def definition_instance(file_content: str, declaration: AST, tactics: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Parse a `def` or `instance` declaration into a Python-friendly dict.

    Lean example:
    ```lean
    def add1 (n : Nat) : Nat := n + 1
    ```
    Produces a dict with the same overall shape as `theorem_lemma_abbrev`, but with
    `kind` "def"/"instance" and the appropriate signature/proof extraction.
    """
    tactics_list, comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2, attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2, pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2, whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2= process_modifier(file_content, declaration,tactics)

    kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2,name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2  = find_kind_name_theorem_lemma_abbrev_def_instance_inductive(file_content, declaration)

    explicitBinder_list,type_list, statement_positions = find_statement_def(file_content,declaration)
    positions = extract_positions(declaration)
    positions = [(start, end) for start, end in positions if start is not None and end is not None]
    if not positions:
        return {"kind" : kind}

    whole_start_pos = min(pos[0] for pos in positions)
    whole_end_pos = max(pos[1] for pos in positions)
    if statement_positions:
        statement_start_pos = whole_start_pos
        statement_end_pos = max(pos[1] for pos in statement_positions)
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 = process_lean_file(file_content, statement_start_pos, statement_end_pos)

    else:
        statement_start_pos = None
        statement_end_pos = None
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 =  None, None, None, None, None, None,None

    proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = find_proof(file_content, declaration)
    declaration_info = {
        "kind" : kind,
        "whole_start_pos" : {
            "position" : whole_char_idx_1,
            "line" : whole_line_1,
            "column" : whole_column_1
        },

        "whole_end_pos" :{
            "position" : whole_char_idx_2,
            "line" : whole_line_2,
            "column" : whole_column_2
        },

        "attributes": attributes,
        "comment": {
            "content": comment,
            "start_pos": {
                "position" : comment_char_idx_1,
                "line" : comment_line_1,
                "column" : comment_column_1
            },
            "end_pos": {
                "position" : comment_char_idx_2,
                "line" : comment_line_2,
                "column" : comment_column_2
            }
        },
        "private_protected": pripro,
        "name": {
            "content": name,
            "start_pos": {
                "position" : name_char_idx_1,
                "line" : name_line_1,
                "column" : name_column_1
            },
            "end_pos": {
                "position" : name_char_idx_2,
                "line" : name_line_2,
                "column" : name_column_2
            }
        },
        "parameters": explicitBinder_list,
        "Type":type_list,
        "statement":{
            "content": statement_val_str,
            "start_pos": {
                "position" :  statement_char_idx_1,
                "line" :  statement_line_1,
                "column" :  statement_column_1
            },
            "end_pos": {
                "position" :  statement_char_idx_2,
                "line" :  statement_line_2,
                "column" :  statement_column_2
            }
        },
        "proof" :{
            "content": proof,
            "start_pos": {
                "position" :  proof_char_idx_1,
                "line" :  proof_line_1,
                "column" :  proof_column_1
            },
            "end_pos": {
                "position" :  proof_char_idx_2,
                "line" :  proof_line_2,
                "column" :  proof_column_2
            }
        }
    }
    return declaration_info

def find_kind_name_structure(file_content: str, data: AST) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Extract kind/name for a `structure` declaration.

    Lean example:
    ```lean
    structure Point where
      x : Nat
      y : Nat
    ```
    Returns kind `"structure"` and name `"Point"` plus spans.
    """
    kind = None
    name = None
    kind_pos = None
    kind_end = None
    name_pos = None
    name_end = None

    if isinstance(data, dict):
        if len(data.get("node", {}).get("args", [])) > 1:
            second_node=data["node"]["args"][1]["node"]
            if second_node["kind"]==  "Lean.Parser.Command.structure":
                node_structureTk=second_node["args"][0]["node"]
                atom = node_structureTk["args"][0]["atom"]
                kind = atom.get("val")
                kind_pos = atom.get("info", {}).get("original", {}).get("pos")
                kind_end = atom.get("info", {}).get("original", {}).get("endPos")

                for arg in second_node.get("args", []):
                    if arg.get("node"):
                        if arg.get("node").get("kind") == "Lean.Parser.Command.declId":
                            node_declId = second_node["args"][1]["node"]
                            val=extract_vals(node_declId)
                            val=' '.join(val)
                            positions=extract_positions(node_declId)
                            name_pos = min(pos[0] for pos in positions)
                            name_end = max(pos[1] for pos in positions)

    if kind_pos and kind_end:
        kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2 = process_lean_file(file_content, kind_pos, kind_end)
    else:
        kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2 = None,None,None, None,None,None,None

    if name_pos and name_end:
        name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2 = process_lean_file(file_content, name_pos, name_end)
    else:
        name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2 = None,None,None, None,None,None,None
    return kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2, name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2

def find_statement_structure(file_content: str, data: AST) -> Tuple[List[Dict[str, Any]], Optional[List[ByteSpan]]]:
    """
    Extract binder-like parameters appearing in a `structure` header.

    Lean example:
    ```lean
    structure Wrap (α : Type) where
      val : α
    ```
    Returns spans for `(α : Type)` and overall header positions when available.
    """
    explicitBinder_list = []
    second_node=data["node"]["args"][1]["node"]
    statement_positions=None
    for arg in second_node.get("args", []):
        if arg.get("node"):
            if arg.get("node")["args"]:
                if arg.get("node")["args"][0].get("node"):
                    if arg.get("node")["args"][0].get("node")["kind"] in [ "Lean.Parser.Term.explicitBinder","Lean.Parser.Term.implicitBinder","Lean.Parser.Term.instBinder"]:
                        node_all= arg.get("node")
                        statement_positions= extract_positions(node_all)
                        # print(statement_positions)
                        for arg in node_all.get("args", []):
                            if arg.get("node", {}).get("kind") in[ "Lean.Parser.Term.explicitBinder","Lean.Parser.Term.implicitBinder","Lean.Parser.Term.instBinder"]:
                                node=arg.get("node", {})
                                val=extract_vals(node)
                                val=' '.join(val)
                                positions=extract_positions(node)
                                start_pos = min(pos[0] for pos in positions)
                                end_pos = max(pos[1] for pos in positions)
                                explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
                                # explicitBinder_list.append({"explicitBinder": [explicit, explicit_char_idx_1,  explicit_line_1,  explicit_column_1, explicit_char_idx_2, explicit_line_2,  explicit_column_2]})
                                explicitBinder_list.append({
                                    "explicitBinder":{
                                        "content":explicit,
                                        "start_pos": {
                                            "position" :explicit_char_idx_1,
                                            "line" : explicit_line_1,
                                            "column" : explicit_column_1
                                        },
                                        "end_pos": {
                                            "position" : explicit_char_idx_2,
                                            "line" : explicit_line_2,
                                            "column" : explicit_column_2
                                        }
                                    }
                                })

    return explicitBinder_list, statement_positions

def find_proof_structure(file_content: str, declaration: AST) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Extract the `where ...` block from a `structure` declaration, if present.

    Lean example:
    ```lean
    structure Point where
      x : Nat
      y : Nat
    ```
    Returns the substring starting at `where` through the end of the fields block.
    """
    vals = []
    positions = []
    second_node=declaration["node"]["args"][1]["node"]
    for arg in second_node.get("args", []):
        if arg.get("node"):
            if arg.get("node")["args"]:
                if arg.get("node")["args"][0].get("atom"):
                    if arg.get("node")["args"][0].get("atom").get("val")== "where":
                        vals.extend(extract_vals(arg))
                        positions.extend(extract_positions(arg))
                        break
    if positions:
        proof_start_pos = min(pos[0] for pos in positions)
        proof_end_pos = max(pos[1] for pos in positions)
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = process_lean_file(file_content, proof_start_pos, proof_end_pos)
    else:
        proof_start_pos = None
        proof_end_pos = None
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = None,None,None, None,None,None, None


    return proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2

def structure(file_content: str, declaration: AST, tactics: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Parse a `structure` declaration into a Python-friendly dict.

    Lean example:
    ```lean
    structure Wrap (α : Type) where
      val : α
    ```
    Produces a dict containing kind/name, parameters, statement span, and the `where`
    block as the "proof" field (mirroring other declaration parsers).
    """
    tactics_list, comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2, attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2, pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2, whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2= process_modifier(file_content, declaration,tactics)

    kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2,name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2  = find_kind_name_structure(file_content, declaration)

    explicitBinder_list, statement_positions = find_statement_structure(file_content, declaration)
    positions = extract_positions(declaration)
    positions = [(start, end) for start, end in positions if start is not None and end is not None]
    if not positions:
        return {"kind" : kind}

    whole_start_pos = min(pos[0] for pos in positions)
    whole_end_pos = max(pos[1] for pos in positions)
    if statement_positions:
        statement_start_pos = whole_start_pos
        statement_end_pos = max(pos[1] for pos in statement_positions)
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 = process_lean_file(file_content, statement_start_pos, statement_end_pos)

    else:
        statement_start_pos = None
        statement_end_pos = None
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 =  None, None, None, None, None, None,None

    if statement_positions:
        statement_start_pos = min(pos[0] for pos in statement_positions)
        statement_end_pos = max(pos[1] for pos in statement_positions)
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 = process_lean_file(file_content, statement_start_pos, statement_end_pos)
    else:
        statement_start_pos = None
        statement_end_pos = None
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 =  None, None, None, None, None, None,None

    parts = [attributes, pripro, kind, name, statement_val_str]
    non_none_parts = [part for part in parts if part is not None]
    statement = ' '.join(non_none_parts).strip()

    proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = find_proof_structure(file_content, declaration)
    declaration_info = {
        "kind" : kind,
        "whole_start_pos" : {
            "position" : whole_char_idx_1,
            "line" : whole_line_1,
            "column" : whole_column_1
        },

        "whole_end_pos" :{
            "position" : whole_char_idx_2,
            "line" : whole_line_2,
            "column" : whole_column_2
        },

        "attributes": attributes,
        "comment": {
            "content": comment,
            "start_pos": {
                "position" : comment_char_idx_1,
                "line" : comment_line_1,
                "column" : comment_column_1
            },
            "end_pos": {
                "position" : comment_char_idx_2,
                "line" : comment_line_2,
                "column" : comment_column_2
            }
        },
        "private_protected": pripro,
        "name": {
            "content": name,
            "start_pos": {
                "position" : name_char_idx_1,
                "line" : name_line_1,
                "column" : name_column_1
            },
            "end_pos": {
                "position" : name_char_idx_2,
                "line" : name_line_2,
                "column" : name_column_2
            }
        },
        "parameters": explicitBinder_list,
        "statement":{
            "content": statement_val_str,
            "start_pos": {
                "position" :  statement_char_idx_1,
                "line" :  statement_line_1,
                "column" :  statement_column_1
            },
            "end_pos": {
                "position" :  statement_char_idx_2,
                "line" :  statement_line_2,
                "column" :  statement_column_2
            }
        },
        "proof" :{
            "content": proof,
            "start_pos": {
                "position" :  proof_char_idx_1,
                "line" :  proof_line_1,
                "column" :  proof_column_1
            },
            "end_pos": {
                "position" :  proof_char_idx_2,
                "line" :  proof_line_2,
                "column" :  proof_column_2
            }
        }

    }
    return declaration_info

def find_proof_inductive(file_content: str, declaration: AST) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Extract constructor blocks (`| ...`) from an `inductive` declaration.

    Lean example:
    ```lean
    inductive MyNat where
      | zero : MyNat
      | succ : MyNat → MyNat
    ```
    Returns the substring covering the first constructor node encountered and spans.
    """
    vals = []
    positions = []
    second_node=declaration["node"]["args"][1]["node"]
    for arg in second_node.get("args", []):
        if arg.get("node"):
            if arg.get("node")["args"]:
                if arg.get("node")["args"][0].get("node",{}).get("kind",{}) == "Lean.Parser.Command.ctor":
                    vals.extend(extract_vals(arg))
                    positions.extend(extract_positions(arg))
                    break
    if positions:
        proof_start_pos = min(pos[0] for pos in positions)
        proof_end_pos = max(pos[1] for pos in positions)
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = process_lean_file(file_content, proof_start_pos, proof_end_pos)
    else:
        proof_start_pos = None
        proof_end_pos = None
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = None,None,None, None,None,None, None
    return proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2

def inductive(file_content: str, declaration: AST, tactics: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Parse an `inductive` declaration into a Python-friendly dict.

    Lean example:
    ```lean
    inductive MyNat where
      | zero : MyNat
      | succ : MyNat → MyNat
    ```
    Produces a dict with kind/name, parameters/type (if any), statement span, and
    constructors captured as the "proof" field.
    """
    tactics_list, comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2, attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2, pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2, whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2= process_modifier(file_content, declaration,tactics)

    kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2,name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2  = find_kind_name_theorem_lemma_abbrev_def_instance_inductive(file_content, declaration)

    explicitBinder_list,type_list, statement_positions = find_statement_theorem_lemma_abbrev(file_content, declaration)
    positions = extract_positions(declaration)
    positions = [(start, end) for start, end in positions if start is not None and end is not None]
    if not positions:
        return {"kind" : kind}

    whole_start_pos = min(pos[0] for pos in positions)
    whole_end_pos = max(pos[1] for pos in positions)
    # statement_val_str = ' '.join(statement_vals)
    if statement_positions:
        statement_start_pos = whole_start_pos
        statement_end_pos = max(pos[1] for pos in statement_positions)
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 = process_lean_file(file_content, statement_start_pos, statement_end_pos)
        # print(statement_val_str)
    else:
        statement_start_pos = None
        statement_end_pos = None
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 =  None, None, None, None, None, None,None

    parts = [attributes, pripro, kind, name, statement_val_str]
    non_none_parts = [part for part in parts if part is not None]
    statement = ' '.join(non_none_parts).strip()

    proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = find_proof_inductive(file_content, declaration)
    declaration_info = {
        "kind" : kind,
        "whole_start_pos" : {
            "position" : whole_char_idx_1,
            "line" : whole_line_1,
            "column" : whole_column_1
        },

        "whole_end_pos" :{
            "position" : whole_char_idx_2,
            "line" : whole_line_2,
            "column" : whole_column_2
        },

        "attributes": attributes,
        "comment": {
            "content": comment,
            "start_pos": {
                "position" : comment_char_idx_1,
                "line" : comment_line_1,
                "column" : comment_column_1
            },
            "end_pos": {
                "position" : comment_char_idx_2,
                "line" : comment_line_2,
                "column" : comment_column_2
            }
        },
        "private_protected": pripro,
        "name": {
            "content": name,
            "start_pos": {
                "position" : name_char_idx_1,
                "line" : name_line_1,
                "column" : name_column_1
            },
            "end_pos": {
                "position" : name_char_idx_2,
                "line" : name_line_2,
                "column" : name_column_2
            }
        },
        "parameters": explicitBinder_list,
        "Type":type_list,
        "statement":{
            "content": statement_val_str,
            "start_pos": {
                "position" :  statement_char_idx_1,
                "line" :  statement_line_1,
                "column" :  statement_column_1
            },
            "end_pos": {
                "position" :  statement_char_idx_2,
                "line" :  statement_line_2,
                "column" :  statement_column_2
            }
        },
        "proof" :{
            "content": proof,
            "start_pos": {
                "position" :  proof_char_idx_1,
                "line" :  proof_line_1,
                "column" :  proof_column_1
            },
            "end_pos": {
                "position" :  proof_char_idx_2,
                "line" :  proof_line_2,
                "column" :  proof_column_2
            }
        }
    }
    return declaration_info

def lean4_parser(file_content: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point: convert Lean REPL/verifier JSON output into a structured dict.

    Expected `data` keys:
      - `"tactics"`: list of tactic spans (byte offsets) emitted by the verifier
      - `"premises"`: dependency/premise info (passed through)
      - `"commandASTs"`: list of command ASTs for the file

    Lean example file:
    ```lean
    /-- doc -/
    @[simp] theorem t (n : Nat) : n = n := by rfl
    def add1 (n : Nat) : Nat := n + 1
    structure Wrap (α : Type) where val : α
    inductive MyNat | zero | succ (n : MyNat)
    ```
    Returns a dict with `declarations` containing one parsed entry per command AST,
    each annotated with extracted substrings and precise spans.
    """
    tactics = data.get("tactics")
    premises = data.get("premises")
    command_asts = data.get("commandASTs")

    declarations_info = []
    for i, declaration in enumerate(command_asts):
        kind = declaration.get("node", {}).get("kind")
        if kind in ["Lean.Parser.Command.declaration", "lemma"]:
            if len(declaration.get("node", {}).get("args", [])) > 1 and declaration["node"]["args"][1]["node"]["kind"] in ["Lean.Parser.Command.theorem","Lean.Parser.Command.example", "group", "Lean.Parser.Command.abbrev"]:
                declaration_info = theorem_lemma_abbrev(file_content, declaration,tactics)
                declarations_info.append(declaration_info)
            elif len(declaration.get("node", {}).get("args", [])) > 1 and declaration["node"]["args"][1]["node"]["kind"] in [ "Lean.Parser.Command.definition","Lean.Parser.Command.instance"]:
                declaration_info = definition_instance(file_content, declaration,tactics)
                declarations_info.append(declaration_info)

            elif len(declaration.get("node", {}).get("args", [])) > 1 and declaration["node"]["args"][1]["node"]["kind"] in [ "Lean.Parser.Command.structure"]:
                declaration_info = structure(file_content, declaration,tactics)
                declarations_info.append(declaration_info)

            elif len(declaration.get("node", {}).get("args", [])) > 1 and declaration["node"]["args"][1]["node"]["kind"] in [ "Lean.Parser.Command.inductive"]:
                declaration_info = inductive(file_content, declaration,tactics)
                declarations_info.append(declaration_info)

        else:
            positions = extract_positions(declaration)
            positions = [(start, end) for start, end in positions if start is not None and end is not None]
            if positions:
                start_pos = min(pos[0] for pos in positions)
                end_pos = max(pos[1] for pos in positions)
                whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2 = process_lean_file(file_content, start_pos, end_pos)
            else:
                start_pos=None
                end_pos =None
                whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2=None,None,None,None,None,None,None
            declaration_info={
                "kind": kind,
                "content":whole,
                "whole_start_pos" : {
                    "position" : whole_char_idx_1,
                    "line" : whole_line_1,
                    "column" : whole_column_1
                },
                "whole_end_pos" :{
                    "position" : whole_char_idx_2,
                    "line" : whole_line_2,
                    "column" : whole_column_2
                }
            }
            declarations_info.append(declaration_info)

    return dict(
        tactics=tactics,
        premises=premises,
        declarations=declarations_info,
    )
