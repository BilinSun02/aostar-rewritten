import inspect
from functools import wraps
from enum import Enum, auto
import warnings

class Response(Enum):
    ALWAYS_IGNORE = auto()
    ALWAYS_REPORT = auto()
    ASK_NEXT_TIME = auto()

class ExceptionHandler:
    def __init__(self, func, suppress_warning=False):
        self.func = func
        self.exception_preferences = {}
        self.suppress_warning = suppress_warning

        # Check if the function already has exception handling
        self.check_for_existing_exception_handling()

    def check_for_existing_exception_handling(self):
        try:
            source_code = inspect.getsource(self.func)
            if "try:" in source_code or "except" in source_code:
                if not self.suppress_warning:
                    warnings.warn(f"The function '{self.func.__name__}' already has exception handling. The decorator may not work as expected.")
        except (OSError, TypeError):
            # If we can't get the source code, we can't check for existing exception handling
            pass

    def get_response(self):
        while True:
            user_input = input("Enter your response (ignore/report/ask): ").lower()
            if user_input in ["ignore", "i"]:
                return Response.ALWAYS_IGNORE
            elif user_input in ["report", "r"]:
                return Response.ALWAYS_REPORT
            elif user_input in ["ask", "a"]:
                return Response.ASK_NEXT_TIME
            else:
                print("Invalid input. Please try again.")

    def handle_exception(self, exception_type):
        if exception_type in self.exception_preferences:
            return self.exception_preferences[exception_type]

        response = self.get_response()
        self.exception_preferences[exception_type] = response

        if response == Response.ALWAYS_IGNORE:
            return response
        elif response == Response.ALWAYS_REPORT:
            raise exception_type
        else:  # Response.ASK_NEXT_TIME
            return response

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            exception_type = type(e)
            response = self.handle_exception(exception_type)
            if response == Response.ALWAYS_IGNORE:
                return
            elif response == Response.ALWAYS_REPORT:
                raise e
            else:  # Response.ASK_NEXT_TIME
                return self(*args, **kwargs)

def handle_exceptions(func, suppress_warning=False):
    return ExceptionHandler(func, suppress_warning)

if __name__ == "__main__":
    def dec(func, arg=None):
        def wrapper(*args, **kwargs):
            if arg is not None:
                # Do something with the optional argument
                print(f"Decorator argument: {arg}")
            return func(*args, **kwargs)
        return wrapper

    @dec(arg="some_value")
    def my_function(x, y):
        print(f"Function input: {x}, {y}")

    my_function(1, 2)

    @handle_exceptions(suppress_warning=True)
    def divide(a, b):
        return a / b