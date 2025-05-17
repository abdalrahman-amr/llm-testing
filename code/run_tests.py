import os
import json
import subprocess
import re

OUTPUT_DIR = "./outputs"
INPUT_JSON = os.path.join(OUTPUT_DIR, "test_generation_logs22.json")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "test_generation_logs_with_results_outs.json")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "test_stats_summary.json")

# Helper to run pytest and parse output, including coverage
def run_pytest( output_dir):
    result = subprocess.run(
        [

            "pytest", "/mnt/d/Masters/Software/Project/outputs/testing.py", "--tb=short", "--maxfail=10",
            f"--cov=/mnt/d/Masters/Software/Project/outputs/test_code.py", "--cov-report=term-missing:skip-covered"
        ],
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    output = result.stdout
    last_line = output.split("\n")[-2]
 
    # Parse pytest output for stats
    passed = failed = 0
    coverage = None
    # match = re.search(r"=+ (\d+) passed", last_line)
    # if match:
    try:
        passed = int(last_line.split(" passed")[0][-1])
    except:
        passed = 0
    # match = re.search(r"=+ (\d+) failed", last_line)
    # if match:
    try:
        failed = int(last_line.split(" failed")[0][-1])
    except:
        failed = 0
    try:
        xfailed = int(last_line.split(" xfailed")[0][-1])
    except:
        xfailed = 0
    # Parse coverage percentage (look for 'TOTAL ... XX%')
    cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+(\d+)%", output)
    if cov_match:
        coverage = int(cov_match.group(1))
    return output, {"passed": passed, "failed": failed, "xfailed": xfailed, "coverage": coverage}


# Load input data
with open(INPUT_JSON, "r") as f:
    input_data = json.load(f)

# Remove loading of OUTPUT_JSON and output_data
# output_data = ... (delete this logic)

summary_stats = {}
total_passed = 0
total_failed = 0
total_xfailed = 0
total_coverage = {}

output_data = {}  # Always start fresh

for idx, entry in enumerate(input_data):
    sample_id = entry.get("sample_id")
    code = entry.get("code")
    llm_tests = entry.get("llm_tests", {})
    pynguin_test = entry.get("pynguin_tests")
    results = {"llms": {}, "pynguin": {}}

    print(f"\nProcessing sample {idx+1}/{len(input_data)} (sample_id: {sample_id})...")

    # # Skip if already processed
    # already_done = False
    # if sample_id in output_data and "results" in output_data[sample_id]:
    #     pynguin_result = output_data[sample_id]["results"].get("pynguin", {})
    #     pynguin_output = pynguin_result.get("output")
    #     if pynguin_output not in [None, "Pynguin generation timed out after 10 minutes."]:
    #         already_done = True

    # if already_done:
    #     print(f"Skipping sample {sample_id} (already processed)")
    #     continue

    # Write code to test_code.py
    code_path = os.path.join(OUTPUT_DIR, "test_code.py")
    with open(code_path, "w") as f:
        f.write(code)

    # # LLM tests
    # for model, test_code in llm_tests.items():
    #     print(f"  Running LLM test for model: {model} ...")
    #     test_file = os.path.join(OUTPUT_DIR, "testing.py")
    #     with open(test_file, "w") as f:
    #         f.write(test_code)
    #     output, stats = run_pytest(OUTPUT_DIR)
    #     results["llms"][model] = {"output": output, "stats": stats}
    #     print(f"    Finished LLM test for model: {model} (Passed: {stats['passed']}, Failed: {stats['failed']}, Coverage: {stats['coverage']})")
    #     # Update summary
    #     summary_stats.setdefault(model, {"passed": 0, "failed": 0, "coverage": []})
    #     summary_stats[model]["passed"] = stats["passed"]
    #     summary_stats[model]["failed"] = stats["failed"]
    #     if stats["coverage"] is not None:
    #         summary_stats[model]["coverage"].append(stats["coverage"])
    #     total_passed += stats["passed"]
    #     total_failed += stats["failed"]
    #     os.remove(test_file)

    # Pynguin test
    if pynguin_test is None or pynguin_test == "Pynguin generation timed out after 10 minutes.":
        print("  Skipping Pynguin test (None or timed out previously)...")
        results["pynguin"] = {"output": "No test", "stats": {"passed": 0, "failed": 0, "xfailed": 0, "coverage": None}}
    else:
        try:
            print("  Running Pynguin test ...")
            test_file = os.path.join(OUTPUT_DIR, "testing.py")
            # print(pynguin_test)
            with open(test_file, "w") as f:
                f.write(pynguin_test)

            output, stats = run_pytest(OUTPUT_DIR)
            results["pynguin"] = {"output": output, "stats": stats}
            print(f"    Finished Pynguin test (Passed: {stats['passed']}, Failed: {stats['failed']}, Coverage: {stats['coverage']})")
            summary_stats.setdefault("pynguin", {"passed": 0, "failed": 0, "xfailed": 0, "coverage": []})
            summary_stats["pynguin"]["passed"] += stats["passed"]
            summary_stats["pynguin"]["failed"] += stats["failed"]
            summary_stats["pynguin"]["xfailed"] += stats["xfailed"]
            if stats["coverage"] is not None:
                summary_stats["pynguin"]["coverage"].append(stats["coverage"])
            total_passed += stats["passed"]
            total_failed += stats["failed"]
            total_xfailed += stats["xfailed"]
            # os.remove(test_file)
        except Exception as e:
            print(f"  Error running Pynguin test: {e}")
            results["pynguin"] = {"output": f"Error running Pynguin test: {e}", "stats": {"passed": 0, "failed": 0, "xfailed": 0, "coverage": None}}

    # Clean up code file
    # os.remove(code_path)
    entry["results"] = results
    print(f"Finished processing sample {idx+1}/{len(input_data)} (sample_id: {sample_id})")

    # Save results in the output_data dict
    output_data[sample_id] = entry

    # Optionally, save progress after each sample
    with open(OUTPUT_JSON, "w") as f:
        json.dump(list(output_data.values()), f, indent=2)

# Final save (in case you want to save once at the end)
with open(OUTPUT_JSON, "w") as f:
    json.dump(list(output_data.values()), f, indent=2)

# Print and save summary stats
print("\n===== TEST SUMMARY =====")
for model, stats in summary_stats.items():
    avg_cov = (
        sum(stats["coverage"]) / len(stats["coverage"]) if stats["coverage"] else 0
    )
    print(f"{model}: Passed: {stats['passed']}, Failed: {stats['failed']}, Avg Coverage: {avg_cov:.2f}%")
print(f"TOTAL: Passed: {total_passed}, Failed: {total_failed}")

# Add average coverage to summary
for model, stats in summary_stats.items():
    stats["avg_coverage"] = (
        sum(stats["coverage"]) / len(stats["coverage"]) if stats["coverage"] else 0
    )
    # Remove the list, keep only the average
    del stats["coverage"]

summary_stats["TOTAL"] = {"passed": total_passed, "failed": total_failed}
with open(SUMMARY_JSON, "w") as f:
    json.dump(summary_stats, f, indent=2) 