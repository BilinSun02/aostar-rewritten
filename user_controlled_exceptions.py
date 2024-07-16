import inspect
from functools import wraps
from enum import Enum, auto
import warnings

class Response(Enum):
    ALWAYS_IGNORE = auto()
    ALWAYS_REPORT = auto()
    ASK_NEXT_TIME = auto()

class UserControlledExceptionHandler:
    def __init__(self, func, error_val=None, suppress_warning=False):
        self.func = func
        self.exception_preferences = {}
        self.error_val = error_val
        self.suppress_warning = suppress_warning

        # Check if the function already has exception handling
        self._check_for_existing_exception_handling()

    def _check_for_existing_exception_handling(self):
        if not self.suppress_warning:
            try:
                source_code = inspect.getsource(self.func)
                if "try:" in source_code and "except" in source_code:
                    warnings.warn(f"The function '{self.func.__name__}' may already have exception handling. The decorator may not work as expected.", stacklevel=4)
                if "return" in source_code:
                    warnings.warn(f"UserControlledExceptionHandler may result in '{self.func.__name__}' None return values. Make sure this is admissible.", stacklevel=4)
            except (OSError, TypeError):
                # If we can't get the source code, we can't check for existing exception handling
                warnings.warn(f"With no access to original source code, we can't check if user_controlled_exceptions is compatible.", stacklevel=4)

    def _prompt_for_response(self):
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

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            exception_type = type(e)
            try:
                # When a previous response has been recorded
                response = self.exception_preferences[exception_type]
            except KeyError:
                # Exception encountered the first time; need to
                # prompt the user, which is the same as the intended behavior
                # when ASK_NEXT_TIME was previously recorded
                response = Response.ASK_NEXT_TIME

            match response:
                case Response.ALWAYS_IGNORE:
                    pass
                case Response.ASK_NEXT_TIME:
                    print(repr(e))
                    self.exception_preferences[exception_type] = self._prompt_for_response()
                case Response.ALWAYS_REPORT:
                    print(repr(e))

            return self.error_val
    
    def __get__(self, obj, objtype=None):
        if obj is not None: # If called on an instance
            return lambda *args, **kwargs: self.__call__(obj, *args, **kwargs)
        elif objtype is not None: # If called on a class
            return lambda *args, **kwargs: self.__call__(*args, **kwargs)
        else:
            return self

def user_controls_exceptions_of(*args, **kwargs):
    # Called without args
    if len(args) == 1 and not kwargs and callable(args[0]):
        return UserControlledExceptionHandler(args[0])
    # Called with args
    else:
        return lambda func: UserControlledExceptionHandler(func, *args, **kwargs)

if __name__ == "__main__":
    @user_controls_exceptions_of
    def divide(a, b):
        return a / b

    @user_controls_exceptions_of(suppress_warning=False)
    def divide_with_exception_handling(a, b):
        try:
            res = a / b
        except:
            res = 0
        return res

    @user_controls_exceptions_of(suppress_warning=True)
    def divide_with_exception_handling_2(a, b):
        try:
            res = a / b
        except:
            res = 0
        return res

    while True:
        divide(1, 0)