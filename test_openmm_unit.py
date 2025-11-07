import sys
print("Python version:", sys.version, file=sys.stderr)

try:
    import openmm
    print("OpenMM imported successfully", file=sys.stderr)
    print("openmm.__version__:", getattr(openmm, '__version__', 'unknown'), file=sys.stderr)

    from openmm import unit
    print("unit imported successfully", file=sys.stderr)
    print("type(unit):", type(unit), file=sys.stderr)

    print("has nanometer:", hasattr(unit, 'nanometer'), file=sys.stderr)
    print("has nanometers:", hasattr(unit, 'nanometers'), file=sys.stderr)

    if hasattr(unit, 'nanometer'):
        print("unit.nanometer works!", file=sys.stderr)
        x = 1.0 * unit.nanometer
        print("Multiplication works:", x, file=sys.stderr)

except Exception as e:
    print("ERROR:", e, file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)

