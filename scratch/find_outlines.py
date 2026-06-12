import outlines
import inspect

def print_mod(mod_name):
    print(f"--- {mod_name} ---")
    try:
        mod = __import__(f"outlines.{mod_name}", fromlist=[""])
        print(dir(mod))
    except Exception as e:
        print("Error:", e)

for m in ["models", "processors", "fsm", "json_schema", "grammars"]:
    print_mod(m)
