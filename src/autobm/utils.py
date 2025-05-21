from .client import LLMClient 
from .prompt.manager import PromptManager 
import re
from typing import Union, Literal
from .model.pipeline import test_model_integrity 
from types import ModuleType
import torch

def find_match(s: str, pattern: Union[str, re.Pattern]) -> str:
    try:
        compiled_pattern = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)

        matches = list(compiled_pattern.finditer(s))

        if not matches:
            return ''

        # Get the last match
        last_matcher = matches[-1]

        # Use the first capture group if available, otherwise the whole match
        if last_matcher.lastindex:  # If capture groups exist
            raw_content = last_matcher.group(1).strip()
        else:  # No capture group mode
            raw_content = last_matcher.group(0)

        return raw_content
    except re.error as e:
        print(f"Regex error: {e}")
        return ''

def execute_code_and_get_model(code_str: str) -> ModuleType:
    isolated_module = ModuleType("strategy_module")
    # Provide common modules for the executed code
    isolated_module.__dict__.update({
        'torch': torch,
        'F': torch.nn.functional,
        '__name__': '__main__'
    })

    try:
        exec(code_str, isolated_module.__dict__)
    except Exception as e:
        raise RuntimeError(f"Code execution failed: {e}")

    if 'GameModel' not in isolated_module.__dict__:
        raise AttributeError("GameModel not found in executed code.")
    return isolated_module.__dict__['GameModel']


def direct_construct(prompt: str, pattern: Union[str, re.Pattern], client: LLMClient, max_retries: int = 3) -> str:
    """
    Generates content using an LLM client and extracts a match using a regex pattern, with retries.
    """
    retries = 0
    while retries < max_retries:
        response = client.generate(prompt, mode='quest') # Assuming 'quest' is a valid mode
        match = find_match(response, pattern)
        if match:
            return match
        retries += 1
    raise ValueError(f"Max retries ({max_retries}) exceeded without finding a match.")

def code_impl(policy: str,
             pattern: Union[str, re.Pattern],
             client: LLMClient,
             pm: PromptManager,
             train_loader, 
             max_retries_correction: int = 10,
             max_retries_client: int = 3
    ) -> str:


    def _generate_with_retry(current_prompt: str, mode: Literal['code', 'quest', 'reason'], current_max_retries: int = max_retries_client) -> str:
        for _ in range(current_max_retries):
            llm_response = client.generate(current_prompt, mode=mode)
            matched_content = find_match(llm_response, pattern)
            if matched_content:
                return matched_content
        raise ValueError(f"Pattern matching failed after {current_max_retries} retries for mode '{mode}'.")

    raw_prompt = pm.render_prompt('code.j2', policy_desc=policy)
    # Initial code generation
    generated_code = _generate_with_retry(raw_prompt, 'code')

    # Iterative correction loop
    for _ in range(max_retries_correction):
        try:
            GameModel = execute_code_and_get_model(generated_code)
        except Exception as e:
            debug_prompt = pm.render_prompt('debug.j2',
                                        policy=policy,
                                        code=generated_code,
                                        error_message=str(e))
            generated_code = _generate_with_retry(debug_prompt, 'code')
            continue


        model_instance = GameModel()
        is_valid, error_message = test_model_integrity(
            model=model_instance, # Pass the instantiated model
            train_loader=train_loader
        )

        if is_valid:
            return generated_code # Code is valid


        debug_prompt = pm.render_prompt('debug.j2',
                                       code=generated_code,
                                       policy=policy,
                                       error_message=error_message)
        generated_code = _generate_with_retry(debug_prompt, 'code')

    raise RuntimeError(f"Code validation and correction failed after {max_retries_correction} retries.")