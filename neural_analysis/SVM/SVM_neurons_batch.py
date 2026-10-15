import subprocess
import sys

# Define configurations to run
areas_to_run = [
    ['lALM'],
    ['rALM'],
    ['lALM', 'rALM']
]

alphas_to_run = [0.01, 0.1, 1, 10]

# Run all combinations
for area_list in areas_to_run:
    for alpha in alphas_to_run:
        print(f"Running analysis for areas {area_list} with alpha={alpha}")
        subprocess.run([sys.executable, 'SVM_neurons.py',
                       '--area'] + area_list +
                       ['--alpha', str(alpha)])