import subprocess

subprocess.run([
    "python", "-m", "semantic.eval.eval_instance", "../../evaluation/eval_instance.yml"
], cwd="maskclustering/scannetpp-toolkit")