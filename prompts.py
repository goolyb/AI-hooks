import os

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompts(name, n=None):
    path = os.path.join(PROMPTS_DIR, name)
    lines = [l.strip() for l in open(path) if l.strip()]
    return lines[:n] if n else lines


harmful = load_prompts("harmful.txt", 100)
harmless = load_prompts("harmless.txt", 100)
