import sys, tomllib

t = tomllib.load(open("pyproject.toml","rb"))
pepx = set(t.get("project",{}).get("optional-dependencies",{}).keys())
pox  = set(t.get("tool",{}).get("poetry",{}).get("extras",{}).keys())
# ignore differences only if intentional; here we require perfect parity
diff = pepx ^ pox
print("extras parity diff:", diff)
sys.exit(1 if diff else 0)

