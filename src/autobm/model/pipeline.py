import os
import importlib

_EXPECTED_FUNCTIONS = [
    "train_model",
    "test_model",
    "visualize_test",
    "test_model_integrity",
]

from dotenv import load_dotenv
load_dotenv()
PIPELINE_TASK_NAME = os.environ.get("CURRENT_TASK") # Make sure CURRENT_TASK is what you intend here, not CURRENT_PIPELINE_TASK

if not PIPELINE_TASK_NAME:
    # The error message below still mentions CURRENT_PIPELINE_TASK, you might want to make it consistent
    # with the variable you are actually checking (PIPELINE_TASK_NAME from os.environ.get("CURRENT_TASK"))
    raise ImportError(
        "The 'CURRENT_TASK' environment variable (used to determine pipeline) is not set. "
        "Ensure it is loaded (e.g., from a .env file via python-dotenv in your application entry point)."
    )

try:
    # 1. Construct the module name RELATIVE to the current package.
    #    The leading dot '.' is crucial for a relative import.
    relative_module_name = f".implementations.{PIPELINE_TASK_NAME}_pipeline"
    
    # 2. Use importlib.import_module with the 'package' argument.
    #    __package__ refers to the package the current module (pipeline.py) belongs to.
    #    If pipeline.py is in src/autobm/model/, then __package__ should be 'src.autobm.model'.
    print(f"Attempting to import '{relative_module_name}' from package '{__package__}'") # Debug print
    task_pipeline_module = importlib.import_module(relative_module_name, package=__package__)

    for func_name in _EXPECTED_FUNCTIONS:
        if hasattr(task_pipeline_module, func_name):
            globals()[func_name] = getattr(task_pipeline_module, func_name)
        else:
            raise AttributeError(
                f"Function '{func_name}' not found in module '{task_pipeline_module.__name__}'. " # Use the actual module name
                f"Ensure all expected pipeline functions ({_EXPECTED_FUNCTIONS}) "
                f"are defined for task '{PIPELINE_TASK_NAME}'."
            )
            
except ImportError as e:
    resolved_package_context = __package__ if __package__ else "None (is this script run as part of a package?)"
    raise ImportError(
        f"Could not import pipeline module for task '{PIPELINE_TASK_NAME}' using relative path '{relative_module_name}' from package context '{resolved_package_context}'. "
        f"Original error: {e}. \nMake sure the implementation file "
        f"(e.g., src/autobm/model/implementations/{PIPELINE_TASK_NAME}_pipeline.py) exists and "
        f"the 'CURRENT_TASK' environment variable ('{PIPELINE_TASK_NAME}') is correct."
    ) from e 
except AttributeError: # This will catch the AttributeError raised above
    raise

__all__ = _EXPECTED_FUNCTIONS