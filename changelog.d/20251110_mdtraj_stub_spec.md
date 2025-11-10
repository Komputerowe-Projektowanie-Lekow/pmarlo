## fixed

- Give each mdtraj stub created in tests a ModuleSpec so `importlib.util.find_spec("mdtraj")` can run without error and dependent API tests can reliably assert the presence of mdtraj instead of crashing.
