import json
import subprocess
import sys
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple

# -----------------------
# Konfiguracja użytkownika
# -----------------------
FOLDERS_TO_SCAN: List[str] = [
    r"C:\Users\konrad_guest\Documents\GitHub\pmarlo\utilities",
    r"C:\Users\konrad_guest\Documents\GitHub\pmarlo\tests",
    r"C:\Users\konrad_guest\Documents\GitHub\pmarlo\src",
    r"C:\Users\konrad_guest\Documents\GitHub\pmarlo\example_programs",
]

USE_SUFFIX_FILTER = True
SUFFIXES = ["py", "ipynb"]

LangTotals = namedtuple("LangTotals", "files code comment source")


def run_pygount_json(path: str, suffixes: List[str] = None) -> dict:
    """
    Uruchamia pygount w formacie JSON i zwraca zdekodowany obiekt dict.
    """
    cmd = ["pygount", "--format=json", "--out=STDOUT", path]
    if suffixes:
        cmd.insert(1, f"--suffix={','.join(suffixes)}")

    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "Błąd: nie znaleziono 'pygount'. Zainstaluj: pip install pygount",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas analizy '{path}': {e}", file=sys.stderr)
        return {}

    stdout = res.stdout.strip()
    if not stdout:
        return {}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        # Pomoc diagnostyczna: pokaż początek wyjścia, aby łatwiej namierzyć problem.
        snippet = stdout[:500].replace("\n", " ")
        print(
            f"Nie udało się zdekodować JSON dla '{path}': {e}. Początek wyjścia: {snippet}",
            file=sys.stderr,
        )
        return {}


def format_table(rows: List[Tuple[str, int, int, int]]) -> str:
    """
    Proste formatowanie tabeli ASCII: (Language, Files, Code, Comment)
    """
    headers = ("Language", "Files", "Code", "Comment")
    col_widths = [
        max(len(headers[0]), max((len(r[0]) for r in rows), default=0)),
        max(len(headers[1]), max((len(str(r[1])) for r in rows), default=0)),
        max(len(headers[2]), max((len(str(r[2])) for r in rows), default=0)),
        max(len(headers[3]), max((len(str(r[3])) for r in rows), default=0)),
    ]

    def sep(ch="─", cross="┼", left="├", right="┤", top=False, bottom=False):
        if top:
            left, right, cross = "┌", "┐", "┬"
        if bottom:
            left, right, cross = "└", "┘", "┴"
        parts = [ch * (w + 2) for w in col_widths]
        return left + cross.join(parts) + right

    def fmt_row(values, left="│", sep="│", right="│"):
        cells = [" " + str(v).ljust(w) + " " for v, w in zip(values, col_widths)]
        return left + sep.join(cells) + right

    out = [sep(top=True), fmt_row(headers), sep()]
    for r in rows:
        out.append(fmt_row(r))
    out.append(sep(bottom=True))
    return "\n".join(out)


def main() -> None:
    # Akumulator per język dla wszystkich folderów
    aggregated: Dict[str, LangTotals] = {}

    # Krótkie sumy per folder
    per_folder_summary: List[Tuple[str, int, int, int]] = []

    suffixes = SUFFIXES if USE_SUFFIX_FILTER else None

    for path in FOLDERS_TO_SCAN:
        print(f"Analiza folderu: {path}")
        data = run_pygount_json(path, suffixes=suffixes)
        if not data:
            per_folder_summary.append((path, 0, 0, 0))
            continue

        # Podsumowanie per folder (z sekcji 'summary')
        summary = data.get("summary", {})
        f_files = int(summary.get("totalFileCount", 0))
        f_code = int(summary.get("totalCodeCount", 0))
        f_comment = int(summary.get("totalDocumentationCount", 0))
        per_folder_summary.append((path, f_files, f_code, f_comment))

        # Sumowanie per język (z sekcji 'languages')
        for lang in data.get("languages", []):
            name = lang.get("language", "__unknown__")
            lt = LangTotals(
                files=int(lang.get("fileCount", 0)),
                code=int(lang.get("codeCount", 0)),
                comment=int(lang.get("documentationCount", 0)),
                source=int(lang.get("sourceCount", 0)),
            )
            if name in aggregated:
                prev = aggregated[name]
                aggregated[name] = LangTotals(
                    files=prev.files + lt.files,
                    code=prev.code + lt.code,
                    comment=prev.comment + lt.comment,
                    source=prev.source + lt.source,
                )
            else:
                aggregated[name] = lt

    # Budowa tabeli zbiorczej per język, sortowane malejąco po "Code"
    rows = []
    for lang, lt in sorted(aggregated.items(), key=lambda kv: kv[1].code, reverse=True):
        rows.append((lang, lt.files, lt.code, lt.comment))

    total_files = sum(lt.files for lt in aggregated.values())
    total_code = sum(lt.code for lt in aggregated.values())
    total_comment = sum(lt.comment for lt in aggregated.values())
    rows.append(("Sum", total_files, total_code, total_comment))

    print("\n=== ZBIORCZY RAPORT (wszystkie folderdly razem) ===\n")
    print(format_table(rows))

    print("\n=== KRÓTKIE PODSUMOWANIE PER FOLDER ===\n")
    for path, f_files, f_code, f_comment in per_folder_summary:
        print(f"- {path}\n  Files: {f_files} | Code: {f_code} | Comment: {f_comment}")

    print(f"\nŁĄCZNA liczba linii kodu (Code) we wszystkich folderach: {total_code}")


if __name__ == "__main__":
    main()
