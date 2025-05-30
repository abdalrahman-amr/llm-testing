# Install dependencies
# pip install datasets
# pip install together
# pip install pynguin
# pip install pytest
# pip install pytest-cov




import os
import json
import ast
import re
import subprocess
from datasets import load_dataset
from together import Together

# ========== CONFIGURATION ==========
TOGETHER_API_KEYS = [
    # Add more API keys as needed, e.g. "key2", "key3"
]
LLM_MODELS = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    # Add more model names as needed
]
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(OUTPUT_DIR, "test_generation_log2.json")
START_INDEX = 2901  # inclusive
END_INDEX = 5000  # exclusive, change as needed

# ========== DATASET LOADING ==========
dataset = load_dataset("KAKA22/CodeRM-UnitTest")
#
def extract_example(dataset, i):
    code = dataset["train"][i]["code_ground_truth"]
    tests = dataset["train"][i]["unit_tests"]
    tests_list = ast.literal_eval(tests)
    return code, tests_list


# ========== LOAD RESULTS FROM JSON ==========
with open(RESULTS_JSON, "r") as f:
    results = json.load(f)

# ========== LLM TEST GENERATION WITH API KEY SWITCHING ==========
def generate_llm_tests(function_code, model):
    last_exception = None
    for key in TOGETHER_API_KEYS:
        try:
            client = Together(api_key=key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": '''
                        You are a highly skilled test case generator, specializing in creating thorough and comprehensive Python unit tests for software functions. 
                        The code to be tested will be in a module called `test_code.py` (import as `from test_code import ...`).
                        Your output must be Python code for pytest, and you must place all generated test code between the lines:
                        # BEGIN TESTS
                        (your test code here)
                        # END TESTS

                        Guidelines:
                        1. Normal Cases: Create tests for typical, valid inputs.
                        2. Edge Cases: Develop tests for boundary values and limits.
                        3. Invalid Inputs: Design tests for inputs that are invalid or out-of-range.
                        4. Special Conditions: Include tests for specific rules or constraints (e.g., null values, empty strings).
                        5. Error Handling: Ensure tests cover proper error handling and exceptions.
                        6. Function Behavior: Verify expected outputs and return values for various inputs.

                        Principles for Test Cases:
                        * Generate multiple, independent test cases for each scenario.
                        * Use clear, descriptive names for each test case.
                        * Include both positive and negative tests.
                        * Utilize boundary values for edge cases.
                        * Clearly define expected results and relevant input data.
                        Your task is to generate a comprehensive suite of test cases that cover a wide range of scenarios, including edge cases, invalid inputs, and performance considerations. Ensure that each test case is well-documented and follows best practices for unit testing.
                    '''},
                    {"role": "user", "content": f"Understand the following Python function and generate unit tests for it using pytest:\n{function_code}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API key failed: {key[:8]}... - {e}")
            last_exception = e
    raise RuntimeError(f"All Together API keys failed or are depleted. Last error: {last_exception}")

def extract_test_cases(generated_test_cases):
    pattern = r"# BEGIN TESTS(.*?)# END TESTS"
    matches = re.findall(pattern, generated_test_cases, re.DOTALL)
    return matches if matches else [generated_test_cases]

# ========== PYNGUIN TEST GENERATION ==========
def generate_pynguin_tests(code):
    # Save code to file
    code_path = os.path.join(OUTPUT_DIR, "test_code.py")
    with open(code_path, "w") as f:
        f.write(code)
    # Run Pynguin
    subprocess.run(
        ["pynguin",
         "--project-path", OUTPUT_DIR,
         "--output-path", OUTPUT_DIR,
         "--module-name", "test_code"],
        check=True,
        env={**os.environ, "PYNGUIN_DANGER_AWARE": "1"}
    )
    # Find the generated test file
    for fname in os.listdir(OUTPUT_DIR):
        if fname.startswith("test") and fname.endswith(".py"):
            with open(os.path.join(OUTPUT_DIR, fname), "r") as f:
                return f.read()
    return ""

# ========== LLM LOOP ==========
# results = []
for i in range(START_INDEX, END_INDEX):
    code, _ = extract_example(dataset, i)
    # original_tests = "\n\n".join(test["code"] for test in tests_list)
    llm_tests = {}
    for model in LLM_MODELS:
        llm_output = generate_llm_tests(code, model)
        llm_tests[model] = "\n\n".join(extract_test_cases(llm_output))
    results.append({
        "sample_id": i,
        "code": code,
        # "test_code": original_tests,
        "llm_tests": llm_tests,
        "pynguin_tests": None
    })
    print(f"Processed LLM sample {i}")
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

# ========== PYNGUIN LOOP ==========
for i, entry in enumerate(results):
    code = entry["code"]
    print(f"Processing Pynguin sample {i}")
    try:
        pynguin_tests = generate_pynguin_tests(code)
        results[i]["pynguin_tests"] = pynguin_tests
        print(f"Processed Pynguin sample {i}")
        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        pynguin_tests = f"Pynguin failed: {e}"

print(f"Results saved to {RESULTS_JSON}") 
