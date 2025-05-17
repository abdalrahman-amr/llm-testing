import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
import os
import logging
import traceback
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def load_data(file_path):
    """Load the JSON data from the given file path with error handling."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        raise

def validate_sample(sample):
    """Validate if a sample has the necessary fields."""
    required_fields = ['sample_id', 'code']
    for field in required_fields:
        if field not in sample:
            return False, f"Missing required field: {field}"
    return True, ""

def process_sample(sample):
    """Process a single sample from the dataset with enhanced error handling."""
    try:
        valid, message = validate_sample(sample)
        if not valid:
            logger.warning(f"Skipping invalid sample: {message}")
            return None
        
        sample_id = sample.get('sample_id')
        code = sample.get('code')
        
        # Extract code properties
        code_lines = len(code.split('\n'))
        code_chars = len(code)
        code_complexity = estimate_code_complexity(code)
        code_imports = extract_imports(code)
        cyclomatic_complexity = estimate_cyclomatic_complexity(code)
        functions_count = count_functions(code)
        classes_count = count_classes(code)
        
        # Process LLM test results
        llm_results = {}
        if sample.get('results') and sample.get('results').get('llms'):
            for model_name, result in sample.get('results').get('llms').items():
                if not result.get('stats'):
                    continue
                    
                passed = result.get('stats').get('passed', 0)
                failed = result.get('stats').get('failed', 0)
                total = passed + failed
                
                # Calculate pass rate
                pass_rate = passed / total if total > 0 else 0
                
                # Extract error types and test execution data
                error_types = extract_error_types(result.get('output', ''))
                test_execution_time = extract_execution_time(result.get('output', ''))
                
                # Extract test case details
                test_cases = extract_test_cases(result.get('output', ''))
                
                llm_results[model_name] = {
                    'passed': passed,
                    'failed': failed,
                    'total': total,
                    'pass_rate': pass_rate,
                    'error_types': error_types,
                    'execution_time': test_execution_time,
                    'test_cases': test_cases
                }
        
        # Process Pynguin results if available
        pynguin_results = None
        if sample.get('results') and sample.get('results').get('pynguin'):
            pynguin_stats = sample.get('results').get('pynguin').get('stats', {})
            pynguin_output = sample.get('results').get('pynguin').get('output', '')
            
            passed = pynguin_stats.get('passed', 0)
            failed = pynguin_stats.get('failed', 0)
            total = passed + failed
            
            pynguin_results = {
                'passed': passed,
                'failed': failed,
                'total': total,
                'pass_rate': passed / total if total > 0 else 0,
                'error_types': extract_error_types(pynguin_output),
                'execution_time': extract_execution_time(pynguin_output)
            }
        
        return {
            'sample_id': sample_id,
            'code': code,
            'code_lines': code_lines,
            'code_chars': code_chars,
            'code_complexity': code_complexity,
            'cyclomatic_complexity': cyclomatic_complexity,
            'code_imports': code_imports,
            'functions_count': functions_count,
            'classes_count': classes_count,
            'llm_results': llm_results,
            'pynguin_results': pynguin_results
        }
    except Exception as e:
        logger.error(f"Error processing sample {sample.get('sample_id', 'unknown')}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def extract_imports(code):
    """Extract import statements from the code."""
    import_lines = re.findall(r'^\s*import\s+(\w+).*$|^\s*from\s+(\w+).*$', code, re.MULTILINE)
    # Flatten the list of tuples and remove empty strings
    imports = [imp[0] if imp[0] else imp[1] for imp in import_lines]
    return Counter(imports)

def estimate_code_complexity(code):
    """Estimate the complexity of a code snippet using various metrics."""
    # Count control structures
    control_structures = len(re.findall(r'\b(if|else|elif|for|while|try|except|finally)\b', code))
    
    # Count function definitions 
    functions = len(re.findall(r'\bdef\s+\w+\s*\(', code))
    
    # Count variables
    variables = len(set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code)))
    
    # Count nested structures (approximation)
    indentation_levels = set(len(line) - len(line.lstrip()) for line in code.split('\n') if line.strip())
    max_nesting = len(indentation_levels)
    
    # Count class definitions
    classes = len(re.findall(r'\bclass\s+\w+', code))
    
    # Simple complexity score
    complexity = control_structures + functions + variables + max_nesting + classes
    return complexity

def estimate_cyclomatic_complexity(code):
    """Estimate the cyclomatic complexity of the code."""
    # Count decision points
    decision_points = len(re.findall(r'\b(if|elif|for|while|and|or|not)\b', code))
    
    # Count function definitions
    functions = len(re.findall(r'\bdef\s+\w+\s*\(', code))
    
    # Basic cyclomatic complexity: decision points + 1
    # We'll calculate this per function on average
    if functions > 0:
        return (decision_points / functions) + 1
    else:
        return decision_points + 1

def count_functions(code):
    """Count the number of function definitions in the code."""
    return len(re.findall(r'\bdef\s+\w+\s*\(', code))

def count_classes(code):
    """Count the number of class definitions in the code."""
    return len(re.findall(r'\bclass\s+\w+', code))

def extract_error_types(output):
    """Extract error types from test output with enhanced patterns."""
    # Common error patterns
    error_patterns = [
        (r'AssertionError', 'AssertionError'),
        (r'TypeError', 'TypeError'),
        (r'ValueError', 'ValueError'),
        (r'IndexError', 'IndexError'),
        (r'KeyError', 'KeyError'),
        (r'AttributeError', 'AttributeError'),
        (r'NameError', 'NameError'),
        (r'ZeroDivisionError', 'ZeroDivisionError'),
        (r'SyntaxError', 'SyntaxError'),
        (r'ImportError', 'ImportError'),
        (r'Failed\:\s*DID\s*NOT\s*RAISE', 'Expected Exception Not Raised'),
        (r'RecursionError', 'RecursionError'),
        (r'MemoryError', 'MemoryError'),
        (r'RuntimeError', 'RuntimeError'),
        (r'IndentationError', 'IndentationError'),
        (r'FileNotFoundError', 'FileNotFoundError'),
        (r'PermissionError', 'PermissionError'),
        (r'StopIteration', 'StopIteration'),
        (r'TimeoutError', 'TimeoutError'),
        (r'OSError', 'OSError'),
        (r'expected\s+([^ ]+)\s+to\s+be', 'Value Mismatch'),
        (r'unexpected\s+EOF', 'Unexpected EOF'),
        (r'unittest\.case\.SkipTest', 'Test Skipped')
    ]
    
    errors = []
    for pattern, error_type in error_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            errors.extend([error_type] * len(matches))
    
    return Counter(errors)

def extract_execution_time(output):
    """Extract test execution time from output."""
    time_match = re.search(r'Ran\s+\d+\s+tests?\s+in\s+([\d\.]+)s', output)
    if time_match:
        return float(time_match.group(1))
    return None

def extract_test_cases(output):
    """Extract individual test case results from output."""
    test_cases = []
    
    # Find all test case names (assuming Python unittest format)
    test_lines = re.findall(r'(test_\w+)\s+\(([\w\.]+)\)\s+\.\.\.\s+(ok|FAIL|ERROR|skipped)', output)
    
    for test_name, test_class, result in test_lines:
        test_cases.append({
            'name': test_name,
            'class': test_class,
            'result': result.lower(),
            'passed': result.lower() == 'ok'
        })
    
    return test_cases

def analyze_data(processed_data):
    """Analyze the processed data and extract enhanced insights."""
    # Filter out None values (failed processing)
    processed_data = [data for data in processed_data if data is not None]
    
    if not processed_data:
        logger.error("No valid data to analyze")
        return None
    
    analysis = {
        'overall_stats': {},
        'model_comparison': {},
        'complexity_correlation': {},
        'error_analysis': {},
        'test_quality': {},
        'clustering_analysis': {},  # New section for clustering analysis
        'time_efficiency': {},      # New section for time efficiency analysis
        'test_coverage': {},        # New section for test coverage analysis
        'code_characteristics': {}  # New section for code characteristics analysis
    }
    
    models = set()
    for sample in processed_data:
        for model in sample['llm_results'].keys():
            models.add(model)
    
    # Overall statistics
    total_samples = len(processed_data)
    models_data = {model: {'pass_rates': [], 'total_tests': 0, 'passed_tests': 0, 'failed_tests': 0, 
                          'execution_times': [], 'test_cases': []} 
                  for model in models}
    
    # Collect data for each model across all samples
    for sample in processed_data:
        for model, results in sample['llm_results'].items():
            if model in models_data:
                models_data[model]['pass_rates'].append(results['pass_rate'])
                models_data[model]['total_tests'] += results['total']
                models_data[model]['passed_tests'] += results['passed']
                models_data[model]['failed_tests'] += results['failed']
                
                if results.get('execution_time'):
                    models_data[model]['execution_times'].append(results['execution_time'])
                
                if results.get('test_cases'):
                    models_data[model]['test_cases'].extend(results['test_cases'])
    
    # Calculate overall metrics for each model
    for model, data in models_data.items():
        avg_pass_rate = np.mean(data['pass_rates']) if data['pass_rates'] else 0
        median_pass_rate = np.median(data['pass_rates']) if data['pass_rates'] else 0
        overall_pass_rate = data['passed_tests'] / data['total_tests'] if data['total_tests'] > 0 else 0
        
        # Calculate consistency metrics
        pass_rate_std = np.std(data['pass_rates']) if len(data['pass_rates']) > 1 else 0
        pass_rate_range = (max(data['pass_rates']) - min(data['pass_rates'])) if data['pass_rates'] else 0
        
        # Calculate time efficiency metrics
        avg_execution_time = np.mean(data['execution_times']) if data['execution_times'] else None
        time_per_test = avg_execution_time / data['total_tests'] if avg_execution_time and data['total_tests'] > 0 else None
        
        analysis['model_comparison'][model] = {
            'avg_pass_rate': avg_pass_rate,
            'median_pass_rate': median_pass_rate,
            'overall_pass_rate': overall_pass_rate,
            'total_tests': data['total_tests'],
            'passed_tests': data['passed_tests'],
            'failed_tests': data['failed_tests'],
            'pass_rate_std': pass_rate_std,
            'pass_rate_range': pass_rate_range,
            'avg_execution_time': avg_execution_time,
            'time_per_test': time_per_test,
            'consistency_score': 1 - (pass_rate_std / avg_pass_rate) if avg_pass_rate > 0 else 0
        }
    
    # Time efficiency analysis
    time_efficiency = {}
    for model in models:
        execution_times = [sample['llm_results'][model].get('execution_time') 
                          for sample in processed_data 
                          if model in sample['llm_results'] and sample['llm_results'][model].get('execution_time')]
        
        if execution_times:
            time_efficiency[model] = {
                'avg_execution_time': np.mean(execution_times),
                'median_execution_time': np.median(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'std_execution_time': np.std(execution_times) if len(execution_times) > 1 else 0
            }
    
    analysis['time_efficiency'] = time_efficiency
    
    # Analyze error types
    error_analysis = {}
    for model in models:
        error_counts = Counter()
        for sample in processed_data:
            if model in sample['llm_results']:
                for error_type, count in sample['llm_results'][model].get('error_types', {}).items():
                    error_counts[error_type] += count
        
        total_errors = sum(error_counts.values())
        
        error_analysis[model] = {
            'error_counts': dict(error_counts),
            'total_errors': total_errors,
            'error_rate': total_errors / models_data[model]['total_tests'] if models_data[model]['total_tests'] > 0 else 0,
            'error_percentages': {error: (count/total_errors)*100 if total_errors > 0 else 0 
                                 for error, count in error_counts.items()},
            'most_common_error': error_counts.most_common(1)[0][0] if error_counts else None,
            'error_diversity': len(error_counts) / total_errors if total_errors > 0 else 0
        }
    
    analysis['error_analysis'] = error_analysis
    
    # Correlate code complexity with test success
    complexity_data = []
    for sample in processed_data:
        for model, results in sample['llm_results'].items():
            complexity_data.append({
                'model': model,
                'sample_id': sample['sample_id'],
                'complexity': sample['code_complexity'],
                'cyclomatic_complexity': sample['cyclomatic_complexity'],
                'code_lines': sample['code_lines'],
                'code_chars': sample['code_chars'],
                'functions_count': sample['functions_count'],
                'classes_count': sample['classes_count'],
                'pass_rate': results['pass_rate'],
                'total_tests': results['total'],
                'execution_time': results.get('execution_time')
            })
    
    complexity_df = pd.DataFrame(complexity_data)
    
    # Calculate correlation coefficients between complexity and pass rate for each model
    correlation_results = {}
    for model in models:
        model_data = complexity_df[complexity_df['model'] == model]
        if len(model_data) > 1:  # Need at least 2 points for correlation
            correlations = {}
            for metric in ['complexity', 'cyclomatic_complexity', 'code_lines', 'code_chars', 
                          'functions_count', 'classes_count']:
                correlations[f'{metric}_correlation'] = model_data[metric].corr(model_data['pass_rate'])
                
                # Add p-value for significance testing
                if not model_data[metric].isna().any() and not model_data['pass_rate'].isna().any():
                    r, p_value = stats.pearsonr(model_data[metric], model_data['pass_rate'])
                    correlations[f'{metric}_p_value'] = p_value
                else:
                    correlations[f'{metric}_p_value'] = None
            
            correlation_results[model] = correlations
    
    analysis['complexity_correlation'] = correlation_results
    
    # Clustering analysis to identify patterns
    if len(complexity_df) > 5:  # Only perform clustering with enough data points
        try:
            # Prepare data for clustering
            clustering_features = complexity_df[['complexity', 'code_lines', 'pass_rate']].dropna()
            
            if len(clustering_features) > 5:  # Still have enough data points after dropping NA
                # Normalize features
                normalized_features = (clustering_features - clustering_features.mean()) / clustering_features.std()
                
                # Determine optimal number of clusters (simple method)
                max_clusters = min(5, len(normalized_features) // 2)
                inertias = []
                
                for n_clusters in range(1, max_clusters + 1):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    kmeans.fit(normalized_features)
                    inertias.append(kmeans.inertia_)
                
                # Find elbow point (simple approximation)
                elbow_point = 2  # Default to 2 clusters
                if len(inertias) > 2:
                    differences = np.diff(inertias)
                    elbow_point = np.argmin(differences) + 2
                
                # Cluster with optimal number of clusters
                kmeans = KMeans(n_clusters=elbow_point, random_state=42)
                complexity_df.loc[clustering_features.index, 'cluster'] = kmeans.fit_predict(normalized_features)
                
                # Analyze clusters
                cluster_analysis = {}
                for cluster in range(elbow_point):
                    cluster_data = complexity_df[complexity_df['cluster'] == cluster]
                    
                    cluster_analysis[f'cluster_{cluster}'] = {
                        'size': len(cluster_data),
                        'avg_complexity': cluster_data['complexity'].mean(),
                        'avg_code_lines': cluster_data['code_lines'].mean(),
                        'avg_pass_rate': cluster_data['pass_rate'].mean(),
                        'models': dict(cluster_data['model'].value_counts()),
                        'sample_ids': cluster_data['sample_id'].tolist()
                    }
                
                analysis['clustering_analysis'] = {
                    'num_clusters': elbow_point,
                    'clusters': cluster_analysis,
                    'feature_importance': {
                        'complexity': abs(np.corrcoef(normalized_features['complexity'], 
                                                     normalized_features['pass_rate'])[0, 1]),
                        'code_lines': abs(np.corrcoef(normalized_features['code_lines'], 
                                                     normalized_features['pass_rate'])[0, 1])
                    }
                }
        except Exception as e:
            logger.warning(f"Clustering analysis failed: {str(e)}")
            analysis['clustering_analysis'] = {
                'error': f"Clustering failed: {str(e)}"
            }
    
    # Code characteristics analysis
    code_characteristics = {}
    for metric in ['code_lines', 'code_chars', 'code_complexity', 'cyclomatic_complexity', 
                  'functions_count', 'classes_count']:
        values = [sample[metric] for sample in processed_data]
        code_characteristics[metric] = {
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'quartiles': np.percentile(values, [25, 50, 75]).tolist()
        }
    
    # Import usage analysis
    all_imports = Counter()
    for sample in processed_data:
        for imp, count in sample.get('code_imports', {}).items():
            all_imports[imp] += count
    
    code_characteristics['imports'] = {
        'total_unique_imports': len(all_imports),
        'most_common_imports': dict(all_imports.most_common(10)),
        'import_distribution': dict(all_imports)
    }
    
    analysis['code_characteristics'] = code_characteristics
    
    # Compare with Pynguin
    pynguin_comparison = {}
    pynguin_samples = [s for s in processed_data if s['pynguin_results'] is not None]
    
    if pynguin_samples:
        for model in models:
            model_vs_pynguin = []
            for sample in pynguin_samples:
                if model in sample['llm_results']:
                    llm_result = sample['llm_results'][model]
                    pynguin_result = sample['pynguin_results']
                    
                    model_vs_pynguin.append({
                        'sample_id': sample['sample_id'],
                        'llm_pass_rate': llm_result['pass_rate'],
                        'pynguin_pass_rate': pynguin_result['pass_rate'],
                        'llm_tests': llm_result['total'],
                        'pynguin_tests': pynguin_result['total'],
                        'llm_execution_time': llm_result.get('execution_time'),
                        'pynguin_execution_time': pynguin_result.get('execution_time')
                    })
            
            if model_vs_pynguin:
                # Calculate correlation and agreement metrics
                llm_rates = [item['llm_pass_rate'] for item in model_vs_pynguin]
                pynguin_rates = [item['pynguin_pass_rate'] for item in model_vs_pynguin]
                
                correlation = np.corrcoef(llm_rates, pynguin_rates)[0, 1] if len(model_vs_pynguin) > 1 else None
                
                # Calculate agreement rate (within 10% difference)
                agreement_count = sum(1 for i in range(len(model_vs_pynguin)) 
                                    if abs(llm_rates[i] - pynguin_rates[i]) <= 0.1)
                agreement_rate = agreement_count / len(model_vs_pynguin)
                
                pynguin_comparison[model] = {
                    'samples': model_vs_pynguin,
                    'correlation': correlation,
                    'agreement_rate': agreement_rate,
                    'avg_llm_pass_rate': np.mean(llm_rates),
                    'avg_pynguin_pass_rate': np.mean(pynguin_rates),
                    'avg_test_count_ratio': np.mean([item['llm_tests'] / item['pynguin_tests'] 
                                                  if item['pynguin_tests'] > 0 else float('inf')
                                                  for item in model_vs_pynguin])
                }
    
    analysis['pynguin_comparison'] = pynguin_comparison
    
    # Test coverage analysis
    test_coverage = {}
    for model in models:
        model_test_data = []
        for sample in processed_data:
            if model in sample['llm_results']:
                test_count = sample['llm_results'][model]['total']
                code_lines = sample['code_lines']
                functions_count = sample['functions_count']
                
                model_test_data.append({
                    'sample_id': sample['sample_id'],
                    'test_count': test_count,
                    'code_lines': code_lines,
                    'functions_count': functions_count,
                    'tests_per_line': test_count / code_lines if code_lines > 0 else 0,
                    'tests_per_function': test_count / functions_count if functions_count > 0 else 0
                })
        
        if model_test_data:
            coverage_df = pd.DataFrame(model_test_data)
            
            test_coverage[model] = {
                'avg_tests_per_sample': coverage_df['test_count'].mean(),
                'median_tests_per_sample': coverage_df['test_count'].median(),
                'avg_tests_per_line': coverage_df['tests_per_line'].mean(),
                'avg_tests_per_function': coverage_df['tests_per_function'].mean(),
                'test_count_std': coverage_df['test_count'].std(),
                'line_coverage_ratio': coverage_df['test_count'].sum() / coverage_df['code_lines'].sum()
                  if coverage_df['code_lines'].sum() > 0 else 0
            }
    
    analysis['test_coverage'] = test_coverage
    
    # Overall summary stats
    best_model = max(models_data.keys(), key=lambda m: models_data[m]['passed_tests'] / models_data[m]['total_tests'] 
                     if models_data[m]['total_tests'] > 0 else 0)
    
    most_consistent_model = max(models, key=lambda m: analysis['model_comparison'][m].get('consistency_score', 0) 
                              if m in analysis['model_comparison'] else 0)
    
    fastest_model = min(models, key=lambda m: analysis['model_comparison'][m].get('avg_execution_time', float('inf')) 
                       if m in analysis['model_comparison'] and analysis['model_comparison'][m].get('avg_execution_time') else float('inf'))
    
    analysis['overall_stats'] = {
        'total_samples': total_samples,
        'models_evaluated': list(models),
        'best_model': best_model,
        'most_consistent_model': most_consistent_model,
        'fastest_model': fastest_model,
        'avg_code_complexity': np.mean([s['code_complexity'] for s in processed_data]),
        'avg_code_lines': np.mean([s['code_lines'] for s in processed_data]),
        'avg_functions_per_sample': np.mean([s['functions_count'] for s in processed_data]),
        'avg_classes_per_sample': np.mean([s['classes_count'] for s in processed_data])
    }
    
    return analysis


def generate_enhanced_visualizations(analysis, processed_data):
    """Generate enhanced visualizations from the analysis."""
    if not analysis:
        logger.error("No analysis data to visualize")
        return {}
    
    visualizations = {}
    
    # 1. Enhanced model comparison bar chart with error bars
    fig1 = plt.figure(figsize=(12, 7))
    model_names = list(analysis['model_comparison'].keys())
    pass_rates = [analysis['model_comparison'][model]['overall_pass_rate'] for model in model_names]
    pass_rate_stds = [analysis['model_comparison'][model].get('pass_rate_std', 0) for model in model_names]
    
    bars = plt.bar(model_names, pass_rates, color='skyblue', yerr=pass_rate_stds, capsize=5, 
                  alpha=0.8, edgecolor='black', linewidth=1)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Overall Pass Rate', fontsize=12)
    plt.title('Pass Rate Comparison Across Models (with Standard Deviation)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.ylim(0, 1.1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    visualizations['pass_rate_comparison'] = fig1
    
    # 2. Enhanced heatmap for error type analysis
    error_data = []
    for model, errors in analysis['error_analysis'].items():
        for error_type, count in errors.get('error_counts', {}).items():
            error_data.append({
                'model': model,
                'error_type': error_type,
                'count': count
            })
    
    if error_data:
        error_df = pd.DataFrame(error_data)
        
        fig2 = plt.figure(figsize=(14, 10))
        error_pivot = error_df.pivot_table(index='model', columns='error_type', values='count', fill_value=0)
        
        # Sort columns by total frequency for better visualization
        col_order = error_df.groupby('error_type')['count'].sum().sort_values(ascending=False).index.tolist()
        error_pivot = error_pivot[col_order]
        
        # Use a better colormap
        ax = sns.heatmap(error_pivot, annot=True, cmap='YlOrRd', fmt='.0f', 
                         linewidths=0.5, linecolor='white')
        plt.title('Error Types by Model', fontsize=14)
        plt.ylabel('Model', fontsize=12)
        plt.xlabel('Error Type', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        visualizations['error_types'] = fig2
    
    # 3. Enhanced scatter plot for complexity vs. pass rate
    complexity_data = []
    for sample in processed_data:
        for model, results in sample['llm_results'].items():
            complexity_data.append({
                'model': model,
                'complexity': sample['code_complexity'],
                'pass_rate': results['pass_rate'],
                'code_lines': sample['code_lines'],
                'functions_count': sample.get('functions_count', 0),
                'test_count': results.get('total', 0)
            })
    
    complexity_df = pd.DataFrame(complexity_data)
    
    fig3 = plt.figure(figsize=(12, 8))
    models = complexity_df['model'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        model_data = complexity_df[complexity_df['model'] == model]
        
        # Use test count to determine marker size
        sizes = np.array(model_data['test_count']) * 5 + 20  # Scale for better visibility
        
        plt.scatter(model_data['complexity'], model_data['pass_rate'], 
                   label=model, marker=markers[i % len(markers)], alpha=0.7,
                   color=colors[i], s=sizes, edgecolors='black', linewidth=0.5)
        
        # Add trend line with confidence interval
        if len(model_data) > 1:
            x = model_data['complexity']
            y = model_data['pass_rate']
            
            # Simple linear regression with confidence band
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # Sort x values for plotting
            x_sorted = np.sort(x)
            y_pred = p(x_sorted)
            
            # Plot trend line
            plt.plot(x_sorted, y_pred, linestyle='--', color=colors[i], alpha=0.8, linewidth=2)
            
            # Add annotation for correlation
            if model in analysis['complexity_correlation']:
                corr = analysis['complexity_correlation'][model].get('complexity_correlation')
                if corr is not None:
                    plt.annotate(f"r = {corr:.2f}", 
                                xy=(x.max(), p(x.max())),
                                xytext=(10, 0), 
                                textcoords="offset points",
                                color=colors[i], fontsize=9)
    
    plt.xlabel('Code Complexity', fontsize=12)
    plt.ylabel('Pass Rate', fontsize=12)
    plt.title('Code Complexity vs. Pass Rate (Marker Size = Test Count)', fontsize=14)
    
    # Create custom legend with consistent marker sizes
    legend_elements = [plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', 
                                 label=model, markerfacecolor=colors[i], markersize=10,
                                 markeredgecolor='black', markeredgewidth=0.5) 
                      for i, model in enumerate(models)]
    
    plt.legend(handles=legend_elements, loc='best', title='Models')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    visualizations['complexity_vs_pass_rate'] = fig3
    
    # 4. Enhanced distribution of code complexity by model
    fig4 = plt.figure(figsize=(12, 7))
    sns.boxplot(data=complexity_df, x='model', y='complexity', palette='Set3',
               linewidth=1, width=0.5)
    
    # Add a swarm plot over the box plot for individual data points
    sns.swarmplot(data=complexity_df, x='model', y='complexity', color='black', alpha=0.5, size=4)
    
    plt.title('Distribution of Code Complexity Across Models', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Code Complexity', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    visualizations['complexity_distribution'] = fig4
    
    # 5. New visualization: Radar chart for model performance metrics
    if len(models) > 0:
        fig5 = plt.figure(figsize=(10, 10))
        
        # Prepare radar chart data
        metrics = ['pass_rate', 'consistency', 'speed', 'test_coverage', 'error_diversity']
        N = len(metrics)
        
        # Compute angles for radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection
        ax = plt.subplot(111, polar=True)
        
        # For each model, calculate normalized metrics
        for i, model in enumerate(models):
            model_metrics = []
            
            # Pass rate (normalized 0-1)
            pass_rate = analysis['model_comparison'][model]['overall_pass_rate']
            model_metrics.append(pass_rate)
            
            # Consistency (1 - std/mean, normalized 0-1)
            consistency = analysis['model_comparison'][model].get('consistency_score', 0)
            model_metrics.append(consistency)
            
            # Speed (inverted and normalized execution time)
            exec_times = [m['avg_execution_time'] for m in analysis['model_comparison'].values() 
                         if m.get('avg_execution_time') is not None]
            if exec_times and model in analysis['model_comparison'] and analysis['model_comparison'][model].get('avg_execution_time') is not None:
                # Normalize and invert (faster is better)
                max_time = max(exec_times)
                min_time = min(exec_times)
                if max_time > min_time:
                    norm_speed = 1 - ((analysis['model_comparison'][model]['avg_execution_time'] - min_time) / (max_time - min_time))
                else:
                    norm_speed = 1
            else:
                norm_speed = 0.5  # Default when data is missing
            model_metrics.append(norm_speed)
            
            # Test coverage (normalized tests per line ratio)
            if model in analysis['test_coverage']:
                coverage_values = [m['line_coverage_ratio'] for m in analysis['test_coverage'].values()]
                max_coverage = max(coverage_values) if coverage_values else 1
                coverage = analysis['test_coverage'][model]['line_coverage_ratio'] / max_coverage if max_coverage > 0 else 0
            else:
                coverage = 0.5  # Default
            model_metrics.append(coverage)
            
            # Error diversity (normalized 0-1, higher diversity is good for testing)
            if model in analysis['error_analysis'] and analysis['error_analysis'][model]['total_errors'] > 0:
                diversity = min(1.0, analysis['error_analysis'][model]['error_diversity'] * 2)  # Scale for visibility
            else:
                diversity = 0.5  # Default
            model_metrics.append(diversity)
            
            # Close the loop for plotting
            model_metrics += model_metrics[:1]
            
            # Plot the model metrics
            ax.plot(angles, model_metrics, linewidth=2, linestyle='solid', label=model, color=colors[i])
            ax.fill(angles, model_metrics, color=colors[i], alpha=0.25)
        
        # Set radar chart labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Configure chart
        ax.set_ylim(0, 1)
        plt.title('Model Performance Radar Chart', size=14)
        plt.grid(True, alpha=0.3)
        
        visualizations['model_radar_chart'] = fig5
    
    # 6. New visualization: Time efficiency by model
    if analysis['time_efficiency']:
        fig6 = plt.figure(figsize=(12, 6))
        
        # Prepare data
        models_with_time = list(analysis['time_efficiency'].keys())
        avg_times = [analysis['time_efficiency'][model]['avg_execution_time'] for model in models_with_time]
        std_times = [analysis['time_efficiency'][model]['std_execution_time'] for model in models_with_time]
        
        # Sort by time for better visualization
        time_data = list(zip(models_with_time, avg_times, std_times))
        time_data.sort(key=lambda x: x[1])
        
        models_sorted = [x[0] for x in time_data]
        times_sorted = [x[1] for x in time_data]
        stds_sorted = [x[2] for x in time_data]
        
        # Plot horizontal bar chart
        bars = plt.barh(models_sorted, times_sorted, xerr=stds_sorted, alpha=0.7, 
                       capsize=5, color='lightgreen', edgecolor='darkgreen', linewidth=1)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}s', va='center', fontsize=10)
        
        plt.xlabel('Average Execution Time (seconds)', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title('Test Execution Time Comparison by Model', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        plt.tight_layout()
        
        visualizations['time_efficiency'] = fig6
    
    # 7. New visualization: Complexity components stacked bar chart
    fig7 = plt.figure(figsize=(12, 7))
    
    # Prepare data from processed_data
    code_metrics = {}
    for sample in processed_data:
        for model in sample['llm_results'].keys():
            if model not in code_metrics:
                code_metrics[model] = {
                    'functions': [],
                    'control_structures': [],
                    'classes': []
                }
            
            # Extract complexity components (approximation)
            functions = sample['functions_count']
            classes = sample.get('classes_count', 0)
            # Approximate control structures from total complexity
            control_structures = max(0, sample['code_complexity'] - functions - classes)
            
            code_metrics[model]['functions'].append(functions)
            code_metrics[model]['control_structures'].append(control_structures)
            code_metrics[model]['classes'].append(classes)
    
    # Calculate averages
    models_list = list(code_metrics.keys())
    functions_avg = [np.mean(code_metrics[model]['functions']) for model in models_list]
    control_avg = [np.mean(code_metrics[model]['control_structures']) for model in models_list]
    classes_avg = [np.mean(code_metrics[model]['classes']) for model in models_list]
    
    # Create stacked bar chart
    bar_width = 0.6
    indices = np.arange(len(models_list))
    
    p1 = plt.bar(indices, functions_avg, bar_width, label='Functions', color='#5DA5DA', edgecolor='white')
    p2 = plt.bar(indices, control_avg, bar_width, bottom=functions_avg, 
               label='Control Structures', color='#FAA43A', edgecolor='white')
    
    bottom_values = [functions_avg[i] + control_avg[i] for i in range(len(functions_avg))]
    p3 = plt.bar(indices, classes_avg, bar_width, bottom=bottom_values, 
               label='Classes', color='#60BD68', edgecolor='white')
    
    # Customize the chart
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Component Count', fontsize=12)
    plt.title('Code Complexity Components by Model', fontsize=14)
    plt.xticks(indices, models_list, rotation=45, ha='right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    visualizations['complexity_components'] = fig7
    
    # 8. New visualization: Pass rate by complexity segment
    fig8 = plt.figure(figsize=(12, 7))
    
    # Group the data by complexity levels
    complexity_df['complexity_level'] = pd.qcut(complexity_df['complexity'], 4, 
                                               labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    
    # Create grouped bar chart
    sns.barplot(data=complexity_df, x='complexity_level', y='pass_rate', hue='model', 
               palette=dict(zip(models, colors)), alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.xlabel('Code Complexity Level', fontsize=12)
    plt.ylabel('Average Pass Rate', fontsize=12)
    plt.title('Pass Rate by Code Complexity Level', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(title='Model')
    plt.tight_layout()
    
    visualizations['pass_rate_by_complexity'] = fig8
    
    return visualizations

def perform_statistical_tests(processed_data):
    """Perform statistical tests to validate findings."""
    results = {}
    
    # Extract data frame of all samples and models
    data_points = []
    for sample in processed_data:
        for model, results in sample['llm_results'].items():
            data_points.append({
                'sample_id': sample['sample_id'],
                'model': model,
                'complexity': sample['code_complexity'],
                'code_lines': sample['code_lines'],
                'functions_count': sample.get('functions_count', 0),
                'classes_count': sample.get('classes_count', 0),
                'pass_rate': results['pass_rate'],
                'total_tests': results.get('total', 0),
                'execution_time': results.get('execution_time', None)
            })
    
    df = pd.DataFrame(data_points)
    
    # 1. ANOVA: Do models differ significantly in pass rates?
    try:
        models = df['model'].unique()
        if len(models) > 1:
            model_groups = [df[df['model'] == model]['pass_rate'].values for model in models]
            f_val, p_val = stats.f_oneway(*model_groups)
            
            results['model_pass_rate_anova'] = {
                'test': 'ANOVA',
                'statistic': f_val,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'interpretation': "There is a statistically significant difference between models' pass rates."
                                if p_val < 0.05 else
                                "There is no statistically significant difference between models' pass rates."
            }
    except Exception as e:
        logger.warning(f"ANOVA test failed: {str(e)}")
    
    # 2. Correlation tests
    try:
        # Complexity vs. Pass Rate
        complexity_corr, complexity_p = stats.pearsonr(df['complexity'], df['pass_rate'])
        
        # Code Lines vs. Pass Rate
        lines_corr, lines_p = stats.pearsonr(df['code_lines'], df['pass_rate'])
        
        results['complexity_correlation_test'] = {
            'test': 'Pearson Correlation',
            'complexity_correlation': complexity_corr,
            'complexity_p_value': complexity_p,
            'complexity_significant': complexity_p < 0.05,
            'lines_correlation': lines_corr,
            'lines_p_value': lines_p,
            'lines_significant': lines_p < 0.05,
            'interpretation': f"Code complexity has a {'significant' if complexity_p < 0.05 else 'non-significant'} "
                            f"{'positive' if complexity_corr > 0 else 'negative'} correlation with pass rate."
        }
    except Exception as e:
        logger.warning(f"Correlation test failed: {str(e)}")
    
    # 3. Model pairwise t-tests
    try:
        if len(models) > 1:
            pairwise_tests = {}
            
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    group1 = df[df['model'] == model1]['pass_rate'].values
                    group2 = df[df['model'] == model2]['pass_rate'].values
                    
                    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                    
                    pairwise_tests[f"{model1}_vs_{model2}"] = {
                        'test': 'Independent t-test',
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05,
                        'better_model': model1 if np.mean(group1) > np.mean(group2) else model2
                                       if p_val < 0.05 else "No significant difference",
                        'mean_difference': abs(np.mean(group1) - np.mean(group2))
                    }
            
            results['pairwise_model_comparisons'] = pairwise_tests
    except Exception as e:
        logger.warning(f"Pairwise t-tests failed: {str(e)}")
    
    # 4. Chi-squared test for error type distributions
    try:
        # Group by model and count error types
        error_data = []
        for sample in processed_data:
            for model, result in sample['llm_results'].items():
                for error_type, count in result.get('error_types', {}).items():
                    for _ in range(count):  # Add one row per error occurrence
                        error_data.append({
                            'model': model,
                            'error_type': error_type
                        })
        
        if error_data:
            error_df = pd.DataFrame(error_data)
            
            # Create contingency table
            contingency = pd.crosstab(error_df['model'], error_df['error_type'])
            
            # Calculate chi-squared test
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            results['error_distribution_test'] = {
                'test': 'Chi-squared',
                'chi2_statistic': chi2,
                'p_value': p,
                'degrees_freedom': dof,
                'significant': p < 0.05,
                'interpretation': "Models differ significantly in their error type distributions."
                                if p < 0.05 else
                                "No significant difference in error type distributions across models."
            }
    except Exception as e:
        logger.warning(f"Chi-squared test failed: {str(e)}")
    
    return results

def generate_enhanced_report(analysis, processed_data, statistical_tests, visualization_files=None):
    """Generate a comprehensive enhanced report from the analysis."""
    if not analysis:
        logger.error("No analysis data to generate report")
        return None
    
    report = {
        'summary': {},
        'model_comparison': {},
        'error_analysis': {},
        'complexity_insights': {},
        'pynguin_comparison': {},
        'code_characteristics': {},
        'test_quality': {},
        'time_efficiency': {},
        'advanced_insights': {},
        'statistical_validation': {},
        'recommendations': {},
        'visualization_files': visualization_files or {}
    }
    
    # Enhanced summary
    report['summary'] = {
        'total_samples': analysis['overall_stats']['total_samples'],
        'models_evaluated': analysis['overall_stats']['models_evaluated'],
        'best_performing_model': analysis['overall_stats']['best_model'],
        'most_consistent_model': analysis.get('overall_stats', {}).get('most_consistent_model'),
        'fastest_model': analysis.get('overall_stats', {}).get('fastest_model'),
        'avg_code_complexity': analysis['overall_stats']['avg_code_complexity'],
        'avg_code_lines': analysis['overall_stats']['avg_code_lines'],
        'avg_functions_per_sample': analysis.get('overall_stats', {}).get('avg_functions_per_sample', 0),
        'avg_classes_per_sample': analysis.get('overall_stats', {}).get('avg_classes_per_sample', 0),
        'dataset_size': len(processed_data),
        'dataset_quality': 'High' if len(processed_data) > 50 else 'Medium' if len(processed_data) > 20 else 'Low'
    }
    
    # Enhanced model comparison
    model_comparison = {}
    for model, stats in analysis['model_comparison'].items():
        model_comparison[model] = {
            'overall_pass_rate': stats['overall_pass_rate'],
            'total_tests': stats['total_tests'],
            'passed_tests': stats['passed_tests'],
            'failed_tests': stats['failed_tests'],
            'efficiency_ratio': stats['passed_tests'] / stats['total_tests'] if stats['total_tests'] > 0 else 0,
            'consistency_score': stats.get('consistency_score', 0),
            'pass_rate_std': stats.get('pass_rate_std', 0),
            'pass_rate_range': stats.get('pass_rate_range', 0),
            'avg_execution_time': stats.get('avg_execution_time'),
            'time_per_test': stats.get('time_per_test'),
            'performance_score': stats['overall_pass_rate'] * (1 + stats.get('consistency_score', 0)) / 2,
            'ranking_score': stats['overall_pass_rate'] * 0.6 + 
                           (stats.get('consistency_score', 0) * 0.3) + 
                           (0.1 if stats.get('avg_execution_time') and 
                           stats.get('avg_execution_time') < 1.0 else 0)
        }
    
    # Sort models by ranking score
    sorted_models = sorted(model_comparison.items(), 
                          key=lambda x: x[1]['ranking_score'], reverse=True)
    
    for i, (model, data) in enumerate(sorted_models):
        model_comparison[model]['overall_rank'] = i + 1
    
    report['model_comparison'] = model_comparison
    
    # Enhanced error analysis with percentages and patterns
    error_analysis = {}
    for model, errors in analysis['error_analysis'].items():
        total_errors = errors['total_errors']
        
        # Calculate error diversity metrics
        unique_errors = len(errors['error_counts'])
        error_entropy = -sum((count/total_errors) * np.log2(count/total_errors) 
                           for count in errors['error_counts'].values()) if total_errors > 0 else 0
        
        error_analysis[model] = {
            'error_counts': errors['error_counts'],
            'total_errors': total_errors,
            'error_rate': errors['error_rate'],
            'error_percentages': errors['error_percentages'],
            'most_common_error': errors['most_common_error'],
            'unique_error_types': unique_errors,
            'error_diversity': errors['error_diversity'],
            'error_entropy': error_entropy,
            'error_patterns': {
                'assertion_errors_dominant': errors['error_percentages'].get('AssertionError', 0) > 50,
                'type_errors_significant': errors['error_percentages'].get('TypeError', 0) > 20,
                'syntax_issues_present': any(pct > 5 for err, pct in errors['error_percentages'].items() 
                                         if 'Syntax' in err or 'Indentation' in err),
                'logic_errors_dominant': (errors['error_percentages'].get('AssertionError', 0) +
                                       errors['error_percentages'].get('Value Mismatch', 0)) > 70
            }
        }
    
    report['error_analysis'] = error_analysis
    
    # Enhanced complexity insights
    complexity_data = []
    for sample in processed_data:
        for model, results in sample['llm_results'].items():
            complexity_data.append({
                'model': model,
                'complexity': sample['code_complexity'],
                'cyclomatic_complexity': sample.get('cyclomatic_complexity'),
                'code_lines': sample['code_lines'],
                'code_chars': sample.get('code_chars'),
                'functions_count': sample.get('functions_count', 0),
                'classes_count': sample.get('classes_count', 0),
                'pass_rate': results['pass_rate'],
                'total_tests': results.get('total', 0)
            })
    
    complexity_df = pd.DataFrame(complexity_data)
    
    complexity_insights = {}
    for model in complexity_df['model'].unique():
        model_data = complexity_df[complexity_df['model'] == model]
        
        # Create complexity segments
        model_data['complexity_segment'] = pd.qcut(model_data['complexity'], 
                                                 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        # Calculate average pass rate by segment
        pass_rates_by_segment = model_data.groupby('complexity_segment')['pass_rate'].mean().to_dict()
        
        # Calculate best performing segment
        best_segment = max(pass_rates_by_segment.items(), key=lambda x: x[1])
        
        # Calculate complexity correlation metrics
        complexity_correlations = {}
        for metric in ['complexity', 'cyclomatic_complexity', 'code_lines', 'code_chars', 
                     'functions_count', 'classes_count']:
            if not model_data[metric].isna().any():
                complexity_correlations[metric] = model_data[metric].corr(model_data['pass_rate'])
        
        # Find optimal complexity range (where pass rate >= 90% of max)
        max_pass_rate = model_data['pass_rate'].max()
        optimal_range = model_data[model_data['pass_rate'] >= 0.9 * max_pass_rate]
        
        complexity_insights[model] = {
            'avg_complexity': model_data['complexity'].mean(),
            'median_complexity': model_data['complexity'].median(),
            'complexity_std': model_data['complexity'].std(),
            'complexity_correlations': complexity_correlations,
            'pass_rates_by_segment': pass_rates_by_segment,
            'best_performing_segment': best_segment[0],
            'performance_drop_in_high_complexity': pass_rates_by_segment.get('High', 0) < 
                                                 pass_rates_by_segment.get('Medium-Low', 0),
            'optimal_complexity_range': {
                'min': optimal_range['complexity'].min() if not optimal_range.empty else None,
                'max': optimal_range['complexity'].max() if not optimal_range.empty else None,
                'samples_in_range': len(optimal_range) if not optimal_range.empty else 0
            }
        }
    
    report['complexity_insights'] = complexity_insights
    
    # Enhanced code characteristics analysis
    report['code_characteristics'] = analysis.get('code_characteristics', {})
    
    # Include import analysis if available
    if 'code_characteristics' in analysis and 'imports' in analysis['code_characteristics']:
        report['code_characteristics']['import_analysis'] = {
            'most_common_imports': analysis['code_characteristics']['imports']['most_common_imports'],
            'total_unique_imports': analysis['code_characteristics']['imports']['total_unique_imports'],
            'import_diversity': len(analysis['code_characteristics']['imports']['import_distribution']) / 
                             sum(analysis['code_characteristics']['imports']['import_distribution'].values())
                             if sum(analysis['code_characteristics']['imports']['import_distribution'].values()) > 0 else 0
        }
    
    # Enhanced time efficiency analysis
    time_efficiency = {}
    
    if 'time_efficiency' in analysis:
        for model, metrics in analysis['time_efficiency'].items():
            time_efficiency[model] = metrics.copy()
            
            # Add efficiency score (normalized inverse of execution time)
            max_time = max(m['avg_execution_time'] for m in analysis['time_efficiency'].values())
            if max_time > 0:
                time_efficiency[model]['efficiency_score'] = 1 - (metrics['avg_execution_time'] / max_time)
            else:
                time_efficiency[model]['efficiency_score'] = 1
            
            # Add time consistency metric
            if metrics['avg_execution_time'] > 0:
                time_efficiency[model]['time_consistency'] = 1 - (metrics['std_execution_time'] / metrics['avg_execution_time'])
            else:
                time_efficiency[model]['time_consistency'] = 0
    
    report['time_efficiency'] = time_efficiency
    
    # Enhanced test coverage analysis
    if 'test_coverage' in analysis:
        report['test_quality'] = {
            'test_coverage': analysis['test_coverage'],
            'coverage_quality': {}
        }
        
        # Analyze coverage quality
        for model, metrics in analysis['test_coverage'].items():
            coverage_quality = {
                'coverage_score': min(1.0, metrics['line_coverage_ratio'] * 2),  # Normalize to 0-1
                'test_density': metrics['avg_tests_per_sample'] / 10 if metrics['avg_tests_per_sample'] <= 10 else 1.0,
                'function_coverage': min(1.0, metrics['avg_tests_per_function']),
                'coverage_consistency': 1.0 - (metrics['test_count_std'] / metrics['avg_tests_per_sample']) 
                                    if metrics['avg_tests_per_sample'] > 0 else 0
            }
            
            # Calculate overall quality score
            coverage_quality['overall_score'] = (coverage_quality['coverage_score'] * 0.4 + 
                                             coverage_quality['test_density'] * 0.3 + 
                                             coverage_quality['function_coverage'] * 0.2 +
                                             coverage_quality['coverage_consistency'] * 0.1)
            
            report['test_quality']['coverage_quality'][model] = coverage_quality
    
    # Include clustering analysis if available
    if 'clustering_analysis' in analysis and 'clusters' in analysis['clustering_analysis']:
        clusters = analysis['clustering_analysis']['clusters']
        
        cluster_report = {
            'num_clusters': analysis['clustering_analysis']['num_clusters'],
            'clusters': {}
        }
        
        for cluster_name, cluster_data in clusters.items():
            # Find the defining characteristics of this cluster
            characteristics = []
            
            if cluster_data['avg_complexity'] > 1.2 * np.mean([c['avg_complexity'] for c in clusters.values()]):
                characteristics.append("High complexity code")
            elif cluster_data['avg_complexity'] < 0.8 * np.mean([c['avg_complexity'] for c in clusters.values()]):
                characteristics.append("Low complexity code")
                
            if cluster_data['avg_pass_rate'] > 0.8:
                characteristics.append("High pass rate")
            elif cluster_data['avg_pass_rate'] < 0.5:
                characteristics.append("Low pass rate")
                
            if cluster_data['avg_code_lines'] > 1.2 * np.mean([c['avg_code_lines'] for c in clusters.values()]):
                characteristics.append("Longer code")
            elif cluster_data['avg_code_lines'] < 0.8 * np.mean([c['avg_code_lines'] for c in clusters.values()]):
                characteristics.append("Shorter code")
            
            # Analyze model distribution in cluster
            total_samples = sum(cluster_data['models'].values())
            model_percentages = {model: (count/total_samples)*100 
                                for model, count in cluster_data['models'].items()}
            
            dominant_model = max(model_percentages.items(), key=lambda x: x[1])
            if dominant_model[1] > 50:
                characteristics.append(f"Dominated by {dominant_model[0]}")
            
            cluster_report['clusters'][cluster_name] = {
                'size': cluster_data['size'],
                'characteristics': characteristics,
                'avg_complexity': cluster_data['avg_complexity'],
                'avg_code_lines': cluster_data['avg_code_lines'],
                'avg_pass_rate': cluster_data['avg_pass_rate'],
                'model_distribution': model_percentages
            }
        
        report['advanced_insights']['clustering'] = cluster_report
    
    # Add statistical validation
    report['statistical_validation'] = statistical_tests
    
    # Generate enhanced recommendations
    recommendations = []
    insights = []
    warnings = []
    
    # Best model recommendation
    best_model = sorted_models[0][0]
    best_model_score = sorted_models[0][1]['ranking_score']
    recommendations.append(f"The best overall performer is {best_model} (score: {best_model_score:.2f}) with a pass rate of {model_comparison[best_model]['overall_pass_rate']:.2f} and consistency score of {model_comparison[best_model]['consistency_score']:.2f}.")
    
    # Model combination recommendation
    if len(sorted_models) > 1:
        second_best = sorted_models[1][0]
        # Check error patterns for complementarity
        if (report['error_analysis'][best_model].get('most_common_error') != 
            report['error_analysis'][second_best].get('most_common_error')):
            recommendations.append(f"Consider using both {best_model} and {second_best} together for better test coverage, as they tend to catch different types of errors.")
    
    # Complexity-based recommendations
    for model, insights_data in complexity_insights.items():
        if insights_data.get('performance_drop_in_high_complexity'):
            warnings.append(f"{model} shows significant performance degradation with high complexity code (correlation: {insights_data['complexity_correlations'].get('complexity', 0):.2f}). Consider using a different model for complex code.")
        
        optimal_range = insights_data.get('optimal_complexity_range', {})
        if optimal_range.get('min') is not None and optimal_range.get('max') is not None:
            insights.append(f"{model} performs optimally with code complexity between {optimal_range['min']:.1f} and {optimal_range['max']:.1f}.")
    
    # Consistency-based recommendations
    for model, data in model_comparison.items():
        if data['consistency_score'] < 0.5:
            warnings.append(f"{model} shows high variability in performance (consistency score: {data['consistency_score']:.2f}). Results may be unpredictable.")
        elif data['consistency_score'] > 0.8:
            insights.append(f"{model} shows very consistent test results (score: {data['consistency_score']:.2f}).")
    
    # Error pattern recommendations
    for model, patterns in error_analysis.items():
        error_patterns = patterns.get('error_patterns', {})
        
        if error_patterns.get('assertion_errors_dominant'):
            insights.append(f"{model} primarily produces assertion errors, suggesting issues with the logic or expected outputs rather than syntax.")
        
        if error_patterns.get('syntax_issues_present'):
            warnings.append(f"{model} shows a significant rate of syntax-related errors, which should be addressed.")
    
    # Time efficiency recommendations
    if time_efficiency:
        fastest_model = min(time_efficiency.items(), key=lambda x: x[1]['avg_execution_time'])
        slowest_model = max(time_efficiency.items(), key=lambda x: x[1]['avg_execution_time'])
        
        time_difference = slowest_model[1]['avg_execution_time'] / fastest_model[1]['avg_execution_time']
        
        if time_difference > 2:
            insights.append(f"{fastest_model[0]} executes tests {time_difference:.1f}x faster than {slowest_model[0]}. Consider using {fastest_model[0]} for time-sensitive testing.")
    
    # Test coverage recommendations
    if 'test_quality' in report and 'coverage_quality' in report['test_quality']:
        for model, quality in report['test_quality']['coverage_quality'].items():
            if quality['coverage_score'] < 0.5:
                warnings.append(f"{model} has low test coverage (score: {quality['coverage_score']:.2f}). Consider improving test generation to cover more code.")
    
    # Pynguin comparison recommendations
    if 'pynguin_comparison' in analysis:
        for model, comparison in analysis['pynguin_comparison'].items():
            if comparison.get('correlation') and comparison['correlation'] < 0.5:
                warnings.append(f"{model} shows low correlation with Pynguin results ({comparison['correlation']:.2f}), suggesting potential quality issues.")
            
            if comparison.get('avg_test_count_ratio') and comparison['avg_test_count_ratio'] < 0.7:
                recommendations.append(f"{model} generates significantly fewer tests than Pynguin. Consider improving test quantity.")
    
    # Add insights and warnings to recommendations
    report['recommendations'] = {
        'primary_recommendations': recommendations,
        'insights': insights,
        'warnings': warnings
    }
    
    return report

def save_visualizations(visualizations, output_dir):
    """Save generated visualizations to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_files = {}
    for name, fig in visualizations.items():
        try:
            if fig is not None:
                fig_path = os.path.join(output_dir, f"{name}.png")
                fig.savefig(fig_path, bbox_inches='tight', dpi=300)
                visualization_files[name] = fig_path
                logger.info(f"Saved visualization: {fig_path}")
                plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to save visualization {name}: {str(e)}")
    
    plt.close('all')  # Final cleanup
    return visualization_files

def generate_html_report(report, output_dir):
    """Generate an HTML report from the analysis report."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Fixed HTML template with proper indentation and CSS
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>LLM Testing Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333366; }}
        .section {{ margin-bottom: 30px; border: 1px solid #eee; padding: 15px; border-radius: 5px; }}
        .visualizations {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .visualization {{ max-width: 100%; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .highlight {{ background-color: #ffffcc; }}
        .warning {{ color: #cc3300; }}
        .insight {{ color: #006600; }}
        .recommendation {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>LLM Testing Analysis Report</h1>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>Dataset: {total_samples} samples analyzed across {num_models} models</p>
        <p>Best performing model: <strong>{best_model}</strong> (pass rate: {best_pass_rate})</p>
        <p>Most consistent model: <strong>{most_consistent}</strong> (consistency score: {consistency_score})</p>
        <p>Fastest model: <strong>{fastest_model}</strong> (avg. execution time: {fastest_time}s)</p>
        <p>Average code complexity: {avg_complexity}</p>
        <p>Average code lines: {avg_lines}</p>
    </div>
    
    <div class="section">
        <h2>Model Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Pass Rate</th>
                <th>Consistency</th>
                <th>Execution Time</th>
                <th>Total Tests</th>
                <th>Overall Rank</th>
            </tr>
            {model_comparison_rows}
        </table>
    </div>
    
    <div class="section">
        <h2>Key Recommendations</h2>
        <ul>{recommendations}</ul>
        <h3>Additional Insights</h3>
        <ul>{insights}</ul>
        <h3>Warnings</h3>
        <ul>{warnings}</ul>
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        <div class="visualizations">{visualization_images}</div>
    </div>
    
    <div class="section">
        <h2>Error Analysis</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Error Rate</th>
                <th>Most Common Error</th>
                <th>Error Diversity</th>
            </tr>
            {error_analysis_rows}
        </table>
    </div>
    
    <div class="section">
        <h2>Complexity Insights</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Best Performing Complexity</th>
                <th>Complexity Correlation</th>
                <th>Optimal Range</th>
            </tr>
            {complexity_rows}
        </table>
    </div>
    
    <div class="section">
        <h2>Statistical Validation</h2>
        {statistical_results}
    </div>
    
    <footer>
        <p>Report generated on {timestamp}</p>
    </footer>
</body>
</html>"""
        
        # Get values with safe defaults
        total_samples = report['summary'].get('total_samples', 'N/A')
        models_evaluated = report['summary'].get('models_evaluated', [])
        best_model = report['summary'].get('best_performing_model', 'N/A')
        most_consistent = report['summary'].get('most_consistent_model', 'N/A')
        fastest_model = report['summary'].get('fastest_model', 'N/A')
        
        # Generate HTML components
        model_comparison_rows = ""
        for model, data in sorted(report['model_comparison'].items(), key=lambda x: x[1].get('overall_rank', 999)):
            # Handle possible None values for overall_pass_rate and consistency_score
            pass_rate = data.get('overall_pass_rate')
            pass_rate_display = f"{pass_rate:.2f}" if pass_rate is not None else "N/A"
            
            consistency = data.get('consistency_score')
            consistency_display = consistency if consistency is not None else "N/A"
            
            avg_time = data.get('avg_execution_time')
            if avg_time is not None:
                avg_time_display = f"{avg_time:.2f}s"
            else:
                avg_time_display = "N/A"
                
            total_tests = data.get('total_tests', 0)
            overall_rank = data.get('overall_rank', 'N/A')
            
            model_comparison_rows += f"<tr><td>{model}</td><td>{pass_rate_display}</td><td>{consistency_display}</td><td>{avg_time_display}</td><td>{total_tests}</td><td>{overall_rank}</td></tr>"

        error_analysis_rows = ""
        for model, data in report.get('error_analysis', {}).items():
            # Handle possible None values for error_rate and error_diversity
            error_rate = data.get('error_rate')
            error_rate_display = f"{error_rate:.2f}" if error_rate is not None else "N/A"
            
            most_common_error = data.get('most_common_error', 'N/A')
            
            error_diversity = data.get('error_diversity')
            error_diversity_display = f"{error_diversity:.2f}" if error_diversity is not None else "N/A"
            
            error_analysis_rows += f"<tr><td>{model}</td><td>{error_rate_display}</td><td>{most_common_error}</td><td>{error_diversity_display}</td></tr>"

        complexity_rows = ""
        for model, data in report.get('complexity_insights', {}).items():
            best_segment = data.get('best_performing_segment', 'N/A')
            
            # Safely handle possibly nested dictionaries with None values
            complexity_corr = data.get('complexity_correlations', {})
            complexity_value = complexity_corr.get('complexity') if isinstance(complexity_corr, dict) else None
            complexity_display = complexity_value if complexity_value is not None else "N/A"
            
            # Safely get min and max from optimal_complexity_range
            optimal_range = data.get('optimal_complexity_range', {})
            min_val = optimal_range.get('min') if isinstance(optimal_range, dict) else "N/A"
            max_val = optimal_range.get('max') if isinstance(optimal_range, dict) else "N/A"
            
            complexity_rows += f"<tr><td>{model}</td><td>{best_segment}</td><td>{complexity_display}</td><td>{min_val} - {max_val}</td></tr>"

        # Get recommendations, insights, and warnings with safe handling
        recommendations = report.get('recommendations', {})
        primary_recs = recommendations.get('primary_recommendations', []) if isinstance(recommendations, dict) else []
        insights = recommendations.get('insights', []) if isinstance(recommendations, dict) else []
        warnings = recommendations.get('warnings', []) if isinstance(recommendations, dict) else []
        
        recommendations_html = "".join(f"<li class='recommendation'>{rec}</li>" for rec in primary_recs)
        insights_html = "".join(f"<li class='insight'>{insight}</li>" for insight in insights)
        warnings_html = "".join(f"<li class='warning'>{warning}</li>" for warning in warnings)
        
        visualization_images = ""
        for name, path in report.get('visualization_files', {}).items():
            if path is not None:
                rel_path = os.path.relpath(path, output_dir)
                visualization_images += f"<div class='visualization'><h3>{name.replace('_', ' ').title()}</h3><img src='{rel_path}' alt='{name}' style='max-width:100%; height:auto;'></div>"

        # Safe handling for statistical results
        statistical_validation = report.get('statistical_validation', {})
        statistical_html_items = []
        
        for test_name, results in statistical_validation.items():
            if isinstance(results, dict) and 'interpretation' in results:
                is_significant = results.get('significant', False)
                highlight_class = 'highlight' if is_significant else ''
                interpretation = results.get('interpretation', '')
                if interpretation:
                    statistical_html_items.append(f"<li class='{highlight_class}'>{interpretation}</li>")
        
        statistical_results = "<h3>Key Statistical Findings</h3><ul>" + "".join(statistical_html_items) + "</ul>"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Safely get values for template with appropriate defaults
        best_pass_rate = "N/A"
        if best_model != "N/A":
            model_data = report['model_comparison'].get(best_model, {})
            pass_rate = model_data.get('overall_pass_rate')
            if pass_rate is not None:
                best_pass_rate = f"{pass_rate:.2f}"
        
        consistency_score_value = "N/A"
        if most_consistent != "N/A":
            model_data = report['model_comparison'].get(most_consistent, {})
            consistency = model_data.get('consistency_score')
            if consistency is not None:
                consistency_score_value = consistency
        
        fastest_time_value = "N/A"
        if fastest_model != "N/A":
            time_data = report.get('time_efficiency', {}).get(fastest_model, {})
            avg_time = time_data.get('avg_execution_time')
            if avg_time is not None:
                fastest_time_value = f"{avg_time:.2f}"
        
        avg_complexity = report['summary'].get('avg_code_complexity')
        avg_complexity_display = f"{avg_complexity:.2f}" if avg_complexity is not None else "N/A"
        
        avg_lines = report['summary'].get('avg_code_lines')
        avg_lines_display = f"{avg_lines:.1f}" if avg_lines is not None else "N/A"

        # Fill the template with safe formatting
        html_content = html_template.format(
            total_samples=total_samples,
            num_models=len(models_evaluated),
            best_model=best_model,
            best_pass_rate=best_pass_rate,
            most_consistent=most_consistent,
            consistency_score=consistency_score_value,
            fastest_model=fastest_model,
            fastest_time=fastest_time_value,
            avg_complexity=avg_complexity_display,
            avg_lines=avg_lines_display,
            model_comparison_rows=model_comparison_rows or '',
            error_analysis_rows=error_analysis_rows or '',
            complexity_rows=complexity_rows or '',
            recommendations=recommendations_html or '',
            insights=insights_html or '',
            warnings=warnings_html or '',
            visualization_images=visualization_images or '',
            statistical_results=statistical_results or '',
            timestamp=timestamp)
        
        # Write HTML file
        html_path = os.path.join(output_dir, "llm_testing_analysis_report.html")
        with open(html_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_path}")
        return html_path
    
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.bool_)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def main(file_path, output_dir="llm_analysis_results"):
    """Main function to analyze the data and generate the report."""
    try:
        logger.info(f"Starting LLM testing analysis for {file_path}")
        
        # Load data
        data = load_data(file_path)
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        
        # Process each sample
        processed_data = [process_sample(sample) for sample in data]
        processed_data = [d for d in processed_data if d is not None]
        logger.info(f"Successfully processed {len(processed_data)} samples")
        
        # Analyze the processed data
        analysis = analyze_data(processed_data)
        logger.info("Completed data analysis")
        
        # Perform statistical tests
        statistical_tests = perform_statistical_tests(processed_data)
        logger.info("Completed statistical validation")
        
        # Generate enhanced visualizations
        visualizations = generate_enhanced_visualizations(analysis, processed_data)
        logger.info(f"Generated {len(visualizations)} visualizations")
        
        # Save visualizations
        visualization_files = save_visualizations(visualizations, output_dir)
        logger.info(f"Saved visualizations to {output_dir}")
        
        # Generate enhanced report
        report = generate_enhanced_report(analysis, processed_data, statistical_tests, visualization_files)
        logger.info("Generated comprehensive report")
        
        # Convert numpy types to Python native types for JSON serialization
        report = convert_to_serializable(report)
        
        # Save detailed report as JSON
        report_path = os.path.join(output_dir, "llm_testing_analysis_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved JSON report to {report_path}")
        
        # Generate HTML report
        html_path = generate_html_report(report, output_dir)
        if html_path:
            logger.info(f"Generated HTML report at {html_path}")
        
        # Print summary report
        print("\n===== LLM TESTING ANALYSIS REPORT =====\n")
        
        print("SUMMARY:")
        for key, value in report['summary'].items():
            print(f"  {key}: {value}")
        
        print("\nMODEL COMPARISON:")
        for model, stats in sorted(report['model_comparison'].items(), 
                                 key=lambda x: x[1].get('overall_rank', 999)):
            print(f"  {model} (Rank {stats.get('overall_rank', 'N/A')}):")
            for stat_key in ['overall_pass_rate', 'consistency_score', 'performance_score']:
                if stat_key in stats:
                    print(f"    {stat_key}: {stats[stat_key]:.2f}")
        
        print("\nTOP RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations']['primary_recommendations']):
            print(f"  {i+1}. {rec}")
        
        print("\nKEY INSIGHTS:")
        for i, insight in enumerate(report['recommendations']['insights'][:3]):  # Show top 3 insights
            print(f"  {i+1}. {insight}")
        
        print("\nIMPORTANT WARNINGS:")
        for i, warning in enumerate(report['recommendations']['warnings'][:3]):  # Show top 3 warnings
            print(f"  {i+1}. {warning}")
        
        print(f"\nAnalysis results saved in '{output_dir}' directory:")
        print(f"- Visualizations saved as PNG files")
        print(f"- Detailed report saved as llm_testing_analysis_report.json")
        print(f"- Interactive HTML report saved as llm_testing_analysis_report.html")
        
        return {
            'processed_data': processed_data,
            'analysis': analysis,
            'report': report,
            'visualizations': visualizations,
            'output_dir': output_dir
        }
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Replace with your file path or use command line argument
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "test_generation_logs_with_results 1.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "llm_analysis_results"
    
    try:
        results = main(file_path, output_dir)
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")