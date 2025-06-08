import os
import sys
import subprocess
from pathlib import Path

def quickstart():
    """
    Skrypt szybkiego startu do inicjalizacji projektu:
    1. Tworzy niezbędne katalogi.
    2. Uruchamia przetwarzanie danych (process_data.py).
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    print(f"Korzeń projektu: {project_root}")

    # Dodaj src do sys.path, aby umożliwić importy, jeśli skrypt jest uruchamiany z innego miejsca
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"Dodano {src_path} do sys.path")

    # 1. Utwórz niezbędne katalogi
    print("\n--- Krok 1: Tworzenie niezbędnych katalogów ---")
    dirs_to_create = [
        project_root / "data" / "feature_analysis",
        project_root / "data" / "model_analysis",
        project_root / "output" # Ogólny katalog wyjściowy, jeśli potrzebny
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Utworzono lub potwierdzono istnienie katalogu: {dir_path}")

    # Ścieżki do skryptów
    process_data_script = src_path / "data" / "process_data.py"
    select_features_script = src_path / "data" / "select_the_most_important_features.py"

    python_executable = sys.executable # Użyj bieżącego interpretera Python

    # 2. Uruchom process_data.py
    print(f"\n--- Krok 2: Uruchamianie {process_data_script.name} ---")
    if process_data_script.exists():
        # Uruchom z --no-renew, aby nie przetwarzać ponownie, jeśli pliki istnieją
        # Możesz usunąć '--no-renew', aby zawsze przetwarzać od nowa
        process_data_cmd = [python_executable, str(process_data_script), "--no-renew"]
        print(f"Polecenie: {' '.join(process_data_cmd)}")
        result = subprocess.run(process_data_cmd, capture_output=True, text=True, check=False)
        print("--- Wyjście process_data.py ---")
        print(result.stdout)
        if result.stderr:
            print("--- Błędy process_data.py ---")
            print(result.stderr)
        if result.returncode == 0:
            print(f"{process_data_script.name} zakończony pomyślnie.")
        else:
            print(f"BŁĄD: {process_data_script.name} zakończony z kodem {result.returncode}.")
            # Można zdecydować o przerwaniu, jeśli ten krok jest krytyczny
            # sys.exit(result.returncode)
    else:
        print(f"BŁĄD: Nie znaleziono skryptu {process_data_script}")
        sys.exit(1)

    print("\n--- Quickstart zakończony ---")
    print(f"Sprawdź katalog '{project_root / 'data'}' pod kątem wygenerowanych plików, np. 'processed_features.csv'.")
    print(f"Katalogi analizy: '{project_root / 'data' / 'feature_analysis'}' oraz '{project_root / 'data' / 'model_analysis'}'.")

if __name__ == "__main__":
    quickstart()
