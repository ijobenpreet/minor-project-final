import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_status(msg):
    print(f"[INFO] {msg}")
