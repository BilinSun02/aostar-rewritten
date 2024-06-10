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
    return get_yes_no("Should the tactic be trivial? (y/n): ")

def prompt_for_tactics(obligation: str):
    # Dummy code for now
    tactics = []
    print("Enter your tactics: ")
    one_tactic = "non_empty string"
    while one_tactic:
        tactics.append(input(">"))
    # Separate 
    return tactics