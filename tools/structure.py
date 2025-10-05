import os
import sys


def list_subfolders(folder):
    """Zwraca listę podfolderów w danym folderze."""
    try:
        return [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    except Exception as e:
        print(f"Błąd podczas listowania podfolderów: {e}")
        return []


def recursive_select(folder):
    """
    Rekurencyjnie pozwala użytkownikowi wybierać podfoldery.
    Zwraca listę końcowych folderów (bez dalszych podfolderów lub gdy użytkownik zatrzyma wybór).
    """
    subfolders = list_subfolders(folder)
    # Jeśli brak podfolderów, zwróć bieżący folder jako koniec ścieżki.
    if not subfolders:
        return [folder]

    print(f"\nAktualny folder: {folder}")
    print("Dostępne podfoldery:")
    for idx, sub in enumerate(subfolders, start=1):
        print(f"{idx}. {sub}")
    print(
        "Wpisz numery folderów do wyboru, 'wszystkie' aby wybrać wszystkie, lub 'stop' aby nie wybierać dalej."
    )

    wybor = input("Twój wybór: ").strip().lower()
    if wybor == "stop":
        return [folder]

    chosen = []
    if wybor == "wszystkie":
        for sub in subfolders:
            chosen.extend(recursive_select(os.path.join(folder, sub)))
    else:
        try:
            indices = [int(i.strip()) for i in wybor.split(",")]
            for i in indices:
                if 1 <= i <= len(subfolders):
                    subfolder = os.path.join(folder, subfolders[i - 1])
                    chosen.extend(recursive_select(subfolder))
                else:
                    print(f"Numer {i} poza zakresem, pomijam.")
        except ValueError:
            print("Niepoprawny format wejścia, zatrzymuję dalszy wybór.")
            return [folder]
    return chosen


def gather_files(root_dirs):
    """
    Przeszukuje wszystkie foldery z listy root_dirs rekurencyjnie
    i zwraca listę wszystkich znalezionych plików z ich pełnymi ścieżkami.
    Wyklucza katalogi i pliki '__pycache__'.
    """
    all_files = []
    for root_dir in root_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if "__pycache__" in dirnames:
                dirnames.remove("__pycache__")
            for file in filenames:
                fullpath = os.path.join(dirpath, file)
                # pomijaj pliki w '__pycache__' bez względu na rozszerzenie
                if "__pycache__" in fullpath.split(os.path.sep):
                    continue
                all_files.append(fullpath)
    return all_files


def print_tree(all_files, main_folder):
    """
    Generuje i wypisuje strukturę drzewa na podstawie listy plików.
    Wyklucza również pliki znajdujące się w '__pycache__'.
    """
    tree = {}
    for file in all_files:
        if "__pycache__" in file.split(os.path.sep):
            continue
        rel_path = os.path.relpath(file, main_folder)
        parts = rel_path.split(os.path.sep)
        current = tree
        for part in parts:
            current = current.setdefault(part, {})

    def print_dict(d, prefix=""):
        items = sorted(d.items())
        for i, (key, subtree) in enumerate(items):
            connector = "└── " if i == len(items) - 1 else "├── "
            print(prefix + connector + key)
            if isinstance(subtree, dict) and subtree:
                extension = "    " if i == len(items) - 1 else "│   "
                print_dict(subtree, prefix + extension)

    print_dict(tree)


def process_files(all_files, main_folder):
    """
    Dla każdego pliku .py w /src/pmarlo generuje strukturę drzewa i zwraca ją jako tekst.
    """
    # Filtruj tylko pliki Pythona, znajdujące się w /src/pmarlo
    py_files = [
        f
        for f in all_files
        if f.endswith(".py")
        and os.path.sep + "src" + os.path.sep + "pmarlo" + os.path.sep in f
    ]

    # Buduj drzewo z hierarchią ścieżek względem main_folder
    tree = {}
    for py_file in py_files:
        rel_path = os.path.relpath(py_file, main_folder)
        parts = rel_path.split(os.path.sep)
        current = tree
        for part in parts:
            current = current.setdefault(part, {})

    # Funkcja formatująca drzewo do tekstu
    def format_tree(node, prefix=""):
        lines = []
        items = sorted(node.items())
        for idx, (name, subtree) in enumerate(items):
            connector = "└── " if idx == len(items) - 1 else "├── "
            lines.append(prefix + connector + name)
            if isinstance(subtree, dict) and subtree:
                extension = "    " if idx == len(items) - 1 else "│   "
                lines.extend(format_tree(subtree, prefix + extension))
        return lines

    tree_lines = format_tree(tree, "")
    return "\n".join(tree_lines)


def main():
    # Hardcoded seed path to the /src/pmarlo directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    main_folder = os.path.join(root_dir, "src", "pmarlo")
    output_file = os.path.join(root_dir, "pmarlo_python_tree.txt")

    if not os.path.isdir(main_folder):
        print(f"Podana ścieżka '{main_folder}' nie jest folderem.")
        sys.exit(1)

    # Non-interactive: use the whole /src/pmarlo
    chosen_folders = [main_folder]
    if not chosen_folders:
        print("Nie wybrano żadnych folderów.")
        sys.exit(0)

    # Zbierz wszystkie pliki z wybranych folderów
    all_files = gather_files(chosen_folders)

    print("\nStruktura drzewa wybranych folderów:")
    print_tree(all_files, main_folder)

    # Przetwarzaj pliki wg zasad: szukaj .h dla każdego .cpp
    result = process_files(all_files, main_folder)

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"\nWynik zapisany do pliku: {output_file}")
        except Exception as e:
            print(f"Błąd zapisu do pliku {output_file}: {e}")
    else:
        print("\nPrzetworzony wynik:")
        print(result)


if __name__ == "__main__":
    main()
