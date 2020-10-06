import functools
import argparse
import inspect


def func_to_cmd(func):
    """
    Quick way to make any function with optional keyword arguments parsable from the command line.
    """
    @functools.wraps(func)
    def parser(**kwargs):
        # Get default kwargs
        signature_kwargs = {k:v.default for k, v in inspect.signature(func).parameters.items()}
        # Update default values with values of caller
        signature_kwargs.update(kwargs)
        
        # Parse kwargs
        parser = argparse.ArgumentParser()
        if func.__doc__:
            parser.format_help = help_formatter(func)
        
        for key, value in signature_kwargs.items():
            value_type = type(value)
            if isinstance(value, bool):
                value_type = bool_parse
            if isinstance(value, list):
                if len(value) > 0:
                    list_type = type(value[0])
                else:
                    list_type = str
                value_type = list_parse(list_type)
            parser.add_argument(f'--{key}', dest=key, default=value, type=value_type)
        kwargs = vars(parser.parse_args())
        # Returns the original func with new kwargs
        return func(**kwargs)
    return parser


def bool_parse(arg):
    if arg.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif arg.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def list_parse(list_type):
    def _list_parse(arg):
        arg = arg.replace('[', '').replace(']','')
        arg = arg.split(',')
        return [list_type(value) for value in arg]
    return _list_parse


def help_formatter(func):
    def format_help():
        return inspect.cleandoc(
f"""
This help message was automatically generated from the function docstring. To set a parameter, simply type 

\t--<variable_name>=<value>

In the case of lists, use the syntax

\t--<list_variable_name>=[<value1>,<value2>,...]

Docstring of function '{func.__name__}':
""" + func.__doc__)
    return format_help