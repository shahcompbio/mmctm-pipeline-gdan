import os
import subprocess

if not os.path.exists(config['tmp_dir']): subprocess.run(f'mkdir -p {config["tmp_dir"]}')
if not os.path.exists(config['log_dir']): subprocess.run(f'mkdir -p {config["log_dir"]}')


