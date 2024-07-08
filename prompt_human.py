def get_yes_no(prompt):
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'yes' for yes, 'n' or 'no' for no.")

def prompt_for_triviality(obligation: str):
    # Dummy code for now
    return get_yes_no(f"Should the tactic for {obligation=} be trivial? (y/n): ")

def prompt_for_tactics(obligation: str):
    # Dummy code for now
    tactics = []
    print(f"Enter the tactics for {obligation=}: ")
    while True:
        one_tactic = input(">")
        if one_tactic:
            tactics.append(one_tactic)
        else:
            break
    # Separate 
    return tactics