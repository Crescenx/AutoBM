# AutoBM

(Add a brief project description here)

---

## Setup

### 1. Configure Environment Variables

Create a `.env` file in the root directory of the project. Add the following environment variables to this file:

```env
ALIYUN_QWEN_BASE_URL="[https://dashscope.aliyuncs.com/compatible-mode/v1](https://dashscope.aliyuncs.com/compatible-mode/v1)"
ALIYUN_QWEN_API_KEY="Your Aliyun API Key"

ANTHROPIC_CLAUDE_BASE_URL=""
ANTHROPIC_CLAUDE_API_KEY="Your ANTHROPIC API Key"

CURRENT_TASK="ulti/rps/cda"
```

**Variable Descriptions:**

- `ALIYUN_QWEN_BASE_URL`: Base URL for the Aliyun Qwen API.
- `ALIYUN_QWEN_API_KEY`: Your API key for the Aliyun Qwen service.
- `ANTHROPIC_CLAUDE_BASE_URL`: Base URL for the Anthropic Claude API.
- `ANTHROPIC_CLAUDE_API_KEY`: Your API key for the Anthropic Claude service.
- `CURRENT_TASK`: Specifies the current task for the application. Choose from `ulti`/`rps`/`cda`


### 2. Install Dependencies

1. **Navigate to the project root directory** (where `pyproject.toml` is located) in your terminal.
2. **Run `pip install .`**:
    
    ```bash
    pip install .
    ```
    
    This command tells `pip` to build the project in the current directory (`.`) and install it along with the dependencies listed in `pyproject.toml`.   
3. **For an editable install (recommended for development):**
    
    ```bash
    pip install -e .
    ```

---

## Running the Application

Once the environment variables are set up and dependencies are installed, you can run the program using the following command:

```bash
python -m src.autobm.main
```
