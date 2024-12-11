import pytest
import nbformat
import os
import re
from nbconvert.preprocessors import ExecutePreprocessor
from typing import Dict, Any, List, Tuple

def load_notebook(path: str) -> nbformat.NotebookNode:
    with open(path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def execute_notebook(nb: nbformat.NotebookNode, timeout: int = 600) -> Tuple[nbformat.NotebookNode, List[Dict[str, Any]]]:
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    resources = {'metadata': {'path': '.'}}
    results = []
    
    try:
        # Execute the entire notebook
        ep.preprocess(nb, resources)
        
        # Collect results from executed cells
        for cell in nb.cells:
            if cell.cell_type == 'code':
                results.append({
                    'source': cell.source,
                    'outputs': cell.outputs if hasattr(cell, 'outputs') else [],
                    'execution_count': cell.execution_count if hasattr(cell, 'execution_count') else None,
                    'status': 'success'
                })
    except Exception as e:
        print(f"Error executing notebook: {str(e)}")
        # In case of error, collect what we have so far
        for cell in nb.cells:
            if cell.cell_type == 'code':
                results.append({
                    'source': cell.source,
                    'outputs': cell.outputs if hasattr(cell, 'outputs') else [],
                    'execution_count': cell.execution_count if hasattr(cell, 'execution_count') else None,
                    'status': 'error',
                    'error': str(e)
                })
    
    return nb, results

def compare_outputs(original_output: List[Dict], new_output: List[Dict]) -> List[Dict[str, Any]]:
    differences = []
    
    for idx, (orig, new) in enumerate(zip(original_output, new_output)):
        if orig['status'] != new['status']:
            differences.append({
                'cell_index': idx,
                'type': 'status_mismatch',
                'original': orig['status'],
                'new': new['status']
            })
            continue
            
        if orig['status'] == 'error' or new['status'] == 'error':
            differences.append({
                'cell_index': idx,
                'type': 'execution_error',
                'original_error': orig.get('error'),
                'new_error': new.get('error')
            })
            continue
            
        # Compare outputs if both cells executed successfully
        if 'outputs' in orig and 'outputs' in new:
            orig_text = ''.join(str(out) for out in orig['outputs'])
            new_text = ''.join(str(out) for out in new['outputs'])
            
            if orig_text != new_text:
                differences.append({
                    'cell_index': idx,
                    'type': 'output_mismatch',
                    'original': orig_text,
                    'new': new_text
                })
    
    return differences

def test_chapter_1():
    # Load both original and Python 3.12 versions
    original_nb = load_notebook('Chapter_01.ipynb')
    py312_nb = load_notebook('Chapter_01_py312.ipynb')
    
    # Execute both notebooks
    _, original_results = execute_notebook(original_nb)
    _, py312_results = execute_notebook(py312_nb)
    
    # Compare outputs
    differences = compare_outputs(original_results, py312_results)
    
    # Assert no differences or log them for review
    if differences:
        print("\nDifferences found in Chapter 1 notebook:")
        for diff in differences:
            print(f"\nCell {diff['cell_index']}:")
            print(f"Type: {diff['type']}")
            if diff['type'] == 'output_mismatch':
                print("Original output:", diff['original'])
                print("New output:", diff['new'])
            elif diff['type'] == 'execution_error':
                print("Original error:", diff['original_error'])
                print("New error:", diff['new_error'])
    
    assert not differences, f"Found {len(differences)} differences between original and Python 3.12 versions"