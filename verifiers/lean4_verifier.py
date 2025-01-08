import os
import time
import json
import tempfile
import traceback
import threading
import subprocess
import re
from typing import Any, List

from .verifier import Verifier, VerificationResult, Message, ProofState, EmptyProofState, Goal
from .lean4_repl_ast_parser import lean4_parser

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = '/share/data/mathzero/billion/2dsmodel/DeepSeek-Prover-V1.5/mathlib4' # !!TODO: move mathlib4 to the copra directory, and change this string
#LEAN_PATH = ':'.join([os.path.join(DEFAULT_LEAN_WORKSPACE, package, '.lake/build/lib') for package in [
#    "", # mathlib itself
#    "./.lake/packages/batteries",
#    "./.lake/packages/aesop",
#    "./.lake/packages/Qq",
#    "./.lake/packages/importGraph",
#    "./.lake/packages/proofwidgets"
#]])

lean4_proof_state_separator = "⊢" # Separates hypotheses from inference
lean4_proof_state_boundary = "\n\n" # Separates states
lean4_proof_state_regex = r"((\d+) goals)*([\s|\S]*?)\n\n"
lean4_has_state_message = 'unsolved goals\n'
lean4_goal_regex = rf"([\s|\S]*?){lean4_proof_state_separator}([\s|\S]*)"

class Lean4Verifier(Verifier):
    def verify(self, proof: str) -> VerificationResult:
        response = self.run_lean4_file(proof)
        return VerificationResult(state=response['state'], messages=response['messages'])
    # !!!!TODO: implement
    # !!TODO: perhaps better to give Goal-PartialProofArrivingAtGoal pairs

    # TODO: attribute
    # Also: message here is diff from other places: explain
    def run_lean4_file(
        self,
        code,
        lake_path = DEFAULT_LAKE_PATH,
        lean_workspace = DEFAULT_LEAN_WORKSPACE,
        last_env = None,
        verbose = False,
        timeout = 300,
        allTactics = False,
        ast = False,
        premises = False,
        tactics = False
    ) -> dict[str, Any]: # !!!!!TODO: clarify this Any
        #os.environ['LEAN_PATH'] = LEAN_PATH
        command = dict(
            cmd = code,
            allTactics = allTactics,
            ast = ast,
            tactics = tactics,
            premises = premises
        )
        if last_env is not None:
            command.update(env=last_env)
        message_str = json.dumps(command, ensure_ascii=False)
        if verbose:
            print(message_str)
        #start_time = time.time()
        #system_messages = ''
        try:
            with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_file:
                temp_file.write(message_str + "\r\n\r\n")
                temp_file.seek(0)
                outputs = subprocess.run([lake_path, "exe", 'repl'], stdin=temp_file, capture_output=True, text=True, cwd=lean_workspace, timeout=timeout)
            result = json.loads(outputs.stdout)
            ast_results = lean4_parser(code, result['ast']) if 'ast' in result and result['ast'] else {}
            result = {
                "sorries" : result.get('sorries', []), 
                "tactics" : result.get('tactics', []),
                #"errors" : [m for m in result.get('messages', []) if m['severity'] == 'error'],
                #"warnings" : [m for m in result.get('messages', []) if m['severity'] == 'warning'],
                #"infos" : [m for m in result.get('messages', []) if m['severity'] == 'info'],
                "messages" : result.get('messages', []),
                #"system_messages" : system_messages,
                #"system_errors" : None,
                #"ast" : ast_results,
                #"verified_code" : code,
            }
            #result['pass'] = not any(map(lambda m: m['severity'] == 'error', result['messages']))
            #result['complete'] = result['pass'] and not result['sorries'] and not any("declaration uses 'sorry'" in warning['data'] or 'failed' in warning['data'] for warning in result['warnings'])
        except:
            #result = {
            #    #"pass": False,
            #    #"complete": False,
            #    "system_errors": traceback.format_exc(),
            #    "system_messages": system_messages
            #}
            pass
            # In our codebase, we don't handle this.
        #result['verify_time'] = time.time() - start_time
        return result

    def parse_repl_result(self, result: dict[str, Any]) -> VerificationResult:
        messages : List[Message] = []
        state = EmptyProofState
        for m in result[messages]:
            if m['severity'] == 'error' and m['data'].startswith(lean4_has_state_message):
                unparsed_state = m['data'][len(lean4_has_state_message):]
                # I don't expect multiple proof state messages to occur
                # but should they do, avoid overwriting and let us investigate why this happens
                if not state == EmptyProofState:
                    print(f"Found multiple proof state messages: {state=}, {unparsed_state=}")
                    assert False
                state = self.parse_proof_state(unparsed_state)
            else:
                messages.append(Message(
                    m['severity'],
                    m['data'],
                    m['pos']['line'],
                    m['pos']['column'],
                    m['endPos']['line'],
                    m['endPos']['column'],
                ))

        return VerificationResult(state, messages)
                
    def parse_proof_state( # !!!!! TODO: adapt
        self,
        proof_state_str: str
    ) -> ProofState:
        assert proof_state_str
        if lean4_proof_state_separator not in proof_state_str:
            raise ValueError(f"Invalid {proof_state_str=}")
        goal_strs = proof_state_str.split(lean4_proof_state_boundary)
        goals = map(self.parse_goal, goal_strs)
        return ProofState(proof_state_str, goals)

    def parse_goal(self, goal_str: str): # !!!!! TODO: adapt
        goal_str = goal_str.strip()
        inference = ""
        hyps_infs = re.findall(lean4_goal_regex, goal_str, re.MULTILINE)
        assert len(hyps_infs) == 1, f"Found zero or more than one goal in the goal string: {goal_str}"
        hypotheses_str, inference = hyps_infs[0]
        hypotheses_str = hypotheses_str.strip()
        inference = inference.strip()
        hypotheses = [hyp.rstrip(',') for hyp in hypotheses_str.split("\n")]
        # Get rid of all the empty hypotheses
        hypotheses = [hyp for hyp in hypotheses if len(hyp) > 0]
        goal = Goal(hypotheses, inference)
        return goal


if __name__ == "__main__":
    if False:
        code = """
-- Definitions about natural numbers and primes
import Mathlib.Data.Nat.Prime

-- Mathlib's tactics library
import Mathlib.Tactic

-- We want to refer to some theorems about Natural numbers
open Nat


-- Define theorem or goal to prove
theorem infinitude_of_primes: ∀ N : ℕ, ∃ p ≥ N, Nat.Prime p := by
  -- After `by` we write our "tactics" to prove the theorem...

  -- let N be a natural number
  intro N

  -- Continue with proof as mentioned in link provided in header
  -- let M be N! + 1
  let M := factorial N + 1

  -- let p be smallest prime factor of M which is not 1
  let p := M.minFac


  -- define supporting hypothesis pp, p is prime
  have pp : Nat.Prime p := by
  -- begin proof for supporting p being prime
    -- minimum factor of a number is prime, but what about if M = 1
    apply minFac_prime
    -- so here we prove M != 1 or M > 1
    have : factorial N > 0 := factorial_pos N
    -- this just automatically takes care of linear arithmatic required for proof
    linarith

  -- before this we had existenial statement but now we have condition in p
  use p

  -- split our goal in  2 subgoals
  constructor

  -- proof by contradiction so it should output False
  · by_contra h
    /- hypothesis h1, p divides N! + 1 proved by
    min_fac_dvd : ∀ (n : ℕ), n.min_fac ∣ n
    -/
    have h₁ : p ∣ factorial N + 1 := minFac_dvd M

    -- hypothesis h2, p divides N!
    have h₂ : p ∣  factorial N := by
      apply pp.dvd_factorial.mpr _
      -- proved p <= N, using hypothsis h
      exact le_of_not_ge h
    /-
    proved using dvd_add_right with support from local hypothesis h₂ and h₁
    -/
    have h : p ∣ 1 := (Nat.dvd_add_right h₂).mp h₁
   -- prime not dividing one using local hypothesis pp and h
    exact Nat.Prime.not_dvd_one pp h
   -- second part of proof is just our hypothesis pp that we already proved
  · exact pp

"""
        code = """
-- Definitions about natural numbers and primes
import Mathlib.Data.Nat.Prime

-- Mathlib's tactics library
import Mathlib.Tactic

-- We want to refer to some theorems about Natural numbers
open Nat


-- Define theorem or goal to prove
theorem infinitude_of_primes: ∀ N : ℕ, ∃ p ≥ N, Nat.Prime p := by
  sorry
"""

    code = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))
    (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by
  sorry
"""
    verifier = Lean4Verifier()
    print(verifier.run_lean4_file(code))