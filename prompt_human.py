from typing import List

# TODO: needs refactorization updating the procotols to match those used by prompt_gpt
def get_yes_no(prompt):
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'yes' for yes, 'n' or 'no' for no.")

def prompt_for_tactics(goals:str, avoid_steps:str="[AVOID STEPS]", n_tactics:int=5) -> List[str]:
    tactics = []

    print(f"Tactics to avoid:\n{avoid_steps}")
    print(f"Enter {n_tactics} tactics for goals (one line each; no indentation needed):")
    print(goals)
    while n_tactics:
        one_tactic = input(">")
        if one_tactic:
            tactics.append(one_tactic)
            n_tactics -= 1
    # Separate 
    return tactics