from .common import LLMAccess

class DummyLLMAccess(LLMAccess):
    incurs_cost: bool = False

    def __init__(self, 
        model_name: str,
        budget_in_cents: int = 100,
        dummy_output: str = "Dummy LLM output"
    ) -> None:
        super().__init__(model_name, budget_in_cents)
        self.dummy_output = dummy_output

    def complete(self,
        prompt: str,
        max_tokens: int = 100
    ) -> str:
        return self.dummy_output

class HumanAccess(LLMAccess):
    incurs_cost: bool = False

    # [[maybe_unused]]
    @staticmethod
    def get_yes_no(prompt):
        while True:
            response = input(prompt).strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'yes' for yes, 'n' or 'no' for no.")

    def complete(self,
        prompt: str,
        max_tokens: int = 100
    ) -> str:
        print("Enter your completion given prompt " +
              "(two consecutive empty lines terminates input):")
        print(prompt)
        one_empty_line = False
        all_input = ""
        while True:
            line = input('>')
            all_input += line + "\n"
            if line == "":
                if one_empty_line:
                    return all_input
                one_empty_line = True
            else:
                one_empty_line = False