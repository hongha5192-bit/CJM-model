"""
Fix relative paths in all step scripts to work from parent directory
"""
import os

# Files to fix
files_to_fix = [
    'src/step0_prepare_vnindex.py',
    'src/step1_fit_hmm.py',
    'src/step2_simulate_hmm.py',
    'src/step3_lambda_scan.py',
    'src/step4_fit_jumpmodel_realdata.py'
]

# Replacements to make
replacements = [
    ('../configs/daily.yaml', 'configs/daily.yaml'),
    ('../outputs/', 'outputs/'),
    ('sys.path.append(\'..\')', 'sys.path.append(\'.\')'),
]

for filepath in files_to_fix:
    with open(filepath, 'r') as f:
        content = f.read()

    # Apply replacements
    for old, new in replacements:
        content = content.replace(old, new)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Fixed paths in {filepath}")

print("All paths fixed!")