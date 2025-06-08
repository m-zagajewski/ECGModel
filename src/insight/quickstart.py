import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import platform
import shutil

# Dodaj ścieżkę projektu do sys.path, aby umożliwić importy z innych modułów
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Importy z istniejących modułów
    from src.insight.analyze_features import main as run_feature_analysis
    from src.insight.analyze_features import CONFIG as analyze_features_config
    from src.insight.analyze_features import load_data as load_processed_data_for_selection
    
    from src.insight.analyze_models import main as run_model_analysis
    from src.insight.analyze_models import CONFIG as analyze_models_config
except ImportError as e:
    print(f"Błąd importu: {e}")
    print("Upewnij się, że znajdujesz się w głównym katalogu projektu lub że ścieżki są poprawnie skonfigurowane.")
    print("Upewnij się również, że wszystkie zależności są zainstalowane.")
    sys.exit(1)

# Konfiguracja dla Quickstart
QUICKSTART_CONFIG = {
    'base_output_dir': os.path.join(project_root, 'data/'), # Pozostawiam zgodnie z ostatnią zmianą użytkownika
    'feature_analysis_subdir': 'feature_analysis', # Dedykowany podkatalog
    'model_analysis_subdir': 'model_analysis',   # Dedykowany podkatalog
    'selected_features_log_filename': 'selected_features.csv', # Nazwa pliku logu
    'top_n_features_for_selection': 80,
    'verbose': True,
    # Parametry do przyspieszenia analizy (opcjonalnie, można zakomentować dla pełnej analizy)
    #'feature_analysis_permutation_repeats': 5, # Domyślnie w analyze_features jest 10
    #'model_analysis_cv_folds': 3, # Domyślnie w analyze_models jest 5
}

def create_selected_features_file(feature_summary_path: str,
                                  original_processed_data_path: str,
                                  output_selected_features_path: str,
                                  top_n: int,
                                  quickstart_log_path: str):
    """
    Tworzy plik selected_features.csv na podstawie wyników analizy cech.
    """
    if QUICKSTART_CONFIG['verbose']:
        print(f"\n--- Tworzenie pliku wybranych cech (top {top_n}) ---")
        print(f"Korzystam z podsumowania ważności: {feature_summary_path}")
        print(f"Korzystam z oryginalnych przetworzonych danych: {original_processed_data_path}")

    try:
        # Wczytaj podsumowanie ważności cech
        summary_df = pd.read_csv(feature_summary_path, index_col=0) # Pierwsza kolumna to nazwy cech
        
        # Wybierz kolumnę do sortowania (preferuj 'mean_rank')
        if 'mean_rank' in summary_df.columns:
            sorted_features = summary_df.sort_values('mean_rank', ascending=True)
        elif not summary_df.empty:
            # Jeśli nie ma 'mean_rank', użyj pierwszej dostępnej kolumny z wynikami (zakładając, że wyższa wartość jest lepsza)
            # lub ostatniej, jeśli to ranking (niższa lepsza) - to wymagałoby więcej logiki.
            # Dla prostoty, jeśli jest tylko jedna kolumna z wynikami (poza *_rank), użyj jej.
            # Bezpieczniej jest polegać na 'mean_rank' lub pierwszej kolumnie, jeśli jest to wynik ważności.
            # Dla tego quickstartu, jeśli 'mean_rank' nie ma, spróbujemy użyć pierwszej kolumny zakładając, że to ważność.
            print("Ostrzeżenie: Kolumna 'mean_rank' nie znaleziona w podsumowaniu ważności. Używam pierwszej kolumny do sortowania.")
            # Sprawdź, czy pierwsza kolumna to nie ranking
            potential_importance_col = summary_df.columns[0]
            if 'rank' in potential_importance_col.lower():
                 print(f"Ostrzeżenie: Pierwsza kolumna '{potential_importance_col}' wygląda na ranking. Sortowanie może być niepoprawne.")
                 # W takim przypadku lepiej byłoby wybrać inną kolumnę lub zgłosić błąd.
                 # Na potrzeby quickstartu, kontynuujemy z ostrożnością.
                 sorted_features = summary_df.sort_values(potential_importance_col, ascending=True) # Zakładamy, że to ranking
            else:
                sorted_features = summary_df.sort_values(potential_importance_col, ascending=False) # Zakładamy, że to ważność

        else:
            print("BŁĄD: Plik podsumowania ważności cech jest pusty.")
            return False
            
        selected_feature_names = list(sorted_features.head(top_n).index)
        if QUICKSTART_CONFIG['verbose']:
            print(f"Wybrano następujące cechy (top {len(selected_feature_names)}): {selected_feature_names}")

        if not selected_feature_names:
            print("BŁĄD: Nie udało się wybrać żadnych cech.")
            return False

        # Wczytaj oryginalne przetworzone dane, aby uzyskać wartości cech i kolumnę 'ZGON'
        _, _, full_original_df = load_processed_data_for_selection() # Ta funkcja wczytuje 'processed_features.csv'
        
        if 'ZGON' not in full_original_df.columns:
            print("BŁĄD: Kolumna 'ZGON' nie znaleziona w oryginalnych przetworzonych danych.")
            return False
            
        # Sprawdź, czy wszystkie wybrane cechy istnieją w oryginalnym DataFrame
        missing_features = [f for f in selected_feature_names if f not in full_original_df.columns]
        if missing_features:
            print(f"BŁĄD: Następujące wybrane cechy nie istnieją w oryginalnym zbiorze danych: {missing_features}")
            return False

        # Utwórz DataFrame z wybranymi cechami i kolumną 'ZGON'
        final_selected_columns = selected_feature_names + ['ZGON']
        selected_df = full_original_df[final_selected_columns]
        
        # Zapisz plik selected_features.csv
        selected_df.to_csv(output_selected_features_path, index=True) # Zachowaj indeks, jeśli istnieje (to już nadpisuje)
        if QUICKSTART_CONFIG['verbose']:
            print(f"Zapisano plik wybranych cech do: {output_selected_features_path}")
        
        # Zapisz kopię do katalogu quickstart dla referencji, jeśli ścieżki są różne
        # lub jeśli nazwa pliku logu jest inna niż pliku głównego
        # Normalizuj ścieżki przed porównaniem
        norm_output_path = os.path.normpath(os.path.abspath(output_selected_features_path))
        norm_log_path = os.path.normpath(os.path.abspath(quickstart_log_path))

        if norm_output_path != norm_log_path:
            # Upewnij się, że katalog docelowy dla logu istnieje
            os.makedirs(os.path.dirname(quickstart_log_path), exist_ok=True)
            shutil.copy(output_selected_features_path, quickstart_log_path)
            if QUICKSTART_CONFIG['verbose']:
                print(f"Kopia pliku wybranych cech zapisana również w: {quickstart_log_path}")
        elif QUICKSTART_CONFIG['verbose']:
            print(f"Plik {output_selected_features_path} jest już w docelowej lokalizacji logu quickstart (ta sama ścieżka lub nazwa). Kopiowanie pominięte.")
            
        return True

    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono pliku: {e}")
        return False
    except Exception as e:
        print(f"BŁĄD podczas tworzenia pliku wybranych cech: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_pipeline_quickstart():
    """
    Uruchamia pełny potok: analiza cech, selekcja cech, analiza modeli.
    """
    print("====== Rozpoczęcie pełnego potoku Quickstart ======")
    
    # Główny katalog wyjściowy dla quickstartu (np. data/)
    base_dir = QUICKSTART_CONFIG['base_output_dir']
    os.makedirs(base_dir, exist_ok=True) # Upewnij się, że istnieje
    print(f"Wszystkie wyniki Quickstart będą zapisywane w podkatalogach: {base_dir}")

    # --- Krok 1: Analiza Cech ---
    print("\n====== Krok 1: Uruchamianie analizy cech ======")
    feature_analysis_output_dir = os.path.join(base_dir, QUICKSTART_CONFIG['feature_analysis_subdir'])
    
    # Zmodyfikuj konfigurację analyze_features w locie
    original_af_output_dir = analyze_features_config['output_dir']
    original_af_verbose = analyze_features_config['verbose']
    original_af_perm_repeats = analyze_features_config['permutation_repeats']
    
    analyze_features_config['output_dir'] = feature_analysis_output_dir
    analyze_features_config['verbose'] = QUICKSTART_CONFIG['verbose']
    if 'feature_analysis_permutation_repeats' in QUICKSTART_CONFIG:
        analyze_features_config['permutation_repeats'] = QUICKSTART_CONFIG['feature_analysis_permutation_repeats']
    
    try:
        run_feature_analysis()
        print("====== Analiza cech zakończona ======")
        print(f"Wyniki analizy cech zapisane w: {feature_analysis_output_dir}")
    except FileNotFoundError as e:
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono pliku wymaganego przez analizę cech (prawdopodobnie 'processed_features.csv'): {e}")
        print("Upewnij się, że plik 'data/processed_features.csv' istnieje.")
        return # Zakończ quickstart, jeśli brakuje podstawowych danych
    except Exception as e:
        print(f"BŁĄD podczas analizy cech: {e}")
        # Przywróć oryginalną konfigurację w razie błędu
        analyze_features_config['output_dir'] = original_af_output_dir
        analyze_features_config['verbose'] = original_af_verbose
        analyze_features_config['permutation_repeats'] = original_af_perm_repeats
        return # Zakończ quickstart
    finally:
        # Przywróć oryginalną konfigurację analyze_features
        analyze_features_config['output_dir'] = original_af_output_dir
        analyze_features_config['verbose'] = original_af_verbose
        analyze_features_config['permutation_repeats'] = original_af_perm_repeats

    # --- Krok 2: Selekcja Cech na podstawie wyników analizy ---
    print("\n====== Krok 2: Selekcja cech dla modeli ======")
    feature_summary_file = os.path.join(feature_analysis_output_dir, 'feature_importance_summary.csv')
    original_processed_data_file = os.path.join(project_root, 'data/processed_features.csv') # Używane przez load_processed_data_for_selection
    
    # Ścieżka, gdzie analyze_models oczekuje pliku selected_features.csv (zawsze w data/)
    selected_features_for_models_path = os.path.join(project_root, 'data/selected_features.csv')
    
    # Ścieżka do kopii w katalogu quickstart (np. data/quickstart_selected_features.csv)
    quickstart_selected_features_copy_path = os.path.join(base_dir, QUICKSTART_CONFIG['selected_features_log_filename'])

    if not os.path.exists(feature_summary_file):
        print(f"BŁĄD KRYTYCZNY: Plik podsumowania ważności cech '{feature_summary_file}' nie został znaleziony po analizie cech.")
        print("Nie można kontynuować z selekcją cech i analizą modeli.")
        return

    selection_successful = create_selected_features_file(
        feature_summary_path=feature_summary_file,
        original_processed_data_path=original_processed_data_file, # load_processed_data_for_selection wie, gdzie to jest
        output_selected_features_path=selected_features_for_models_path,
        top_n=QUICKSTART_CONFIG['top_n_features_for_selection'],
        quickstart_log_path=quickstart_selected_features_copy_path
    )

    if not selection_successful:
        print("BŁĄD KRYTYCZNY: Nie udało się utworzyć pliku wybranych cech. Analiza modeli nie zostanie uruchomiona.")
        return
    print("====== Selekcja cech zakończona ======")

    # --- Krok 3: Analiza Modeli ---
    print("\n====== Krok 3: Uruchamianie analizy modeli ======")
    model_analysis_output_dir = os.path.join(base_dir, QUICKSTART_CONFIG['model_analysis_subdir'])

    # Zmodyfikuj konfigurację analyze_models w locie
    original_am_output_dir = analyze_models_config['output_dir']
    original_am_verbose = analyze_models_config['verbose']
    original_am_cv_folds = analyze_models_config['cv_folds']

    analyze_models_config['output_dir'] = model_analysis_output_dir
    analyze_models_config['verbose'] = QUICKSTART_CONFIG['verbose']
    if 'model_analysis_cv_folds' in QUICKSTART_CONFIG:
        analyze_models_config['cv_folds'] = QUICKSTART_CONFIG['model_analysis_cv_folds']

    try:
        run_model_analysis()
        print("====== Analiza modeli zakończona ======")
        print(f"Wyniki analizy modeli zapisane w: {model_analysis_output_dir}")
    except FileNotFoundError as e:
        # Ten błąd powinien być już obsłużony przez selection_successful, ale dla pewności
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono pliku wymaganego przez analizę modeli (prawdopodobnie 'selected_features.csv'): {e}")
        print(f"Oczekiwano pliku w: {selected_features_for_models_path}")
    except Exception as e:
        print(f"BŁĄD podczas analizy modeli: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Przywróć oryginalną konfigurację analyze_models
        analyze_models_config['output_dir'] = original_am_output_dir
        analyze_models_config['verbose'] = original_am_verbose
        analyze_models_config['cv_folds'] = original_am_cv_folds
        
    print("\n====== Pełny potok Quickstart zakończony ======")
    print(f"Sprawdź wyniki w katalogu: {QUICKSTART_CONFIG['base_output_dir']}")

if __name__ == "__main__":
    # Ustawienie backendu matplotlib, aby uniknąć problemów w środowiskach bez GUI
    # (szczególnie ważne, jeśli skrypty pod spodem generują wykresy)
    current_os = platform.system()
    if QUICKSTART_CONFIG['verbose']:
        print(f"System operacyjny: {current_os}")
    
    # Na macOS, analyze_features.py już ustawia 'Agg'. Unikajmy konfliktów.
    # Dla innych systemów, jeśli nie jest to 'Agg', spróbujmy ustawić.
    if current_os != 'Darwin':
        try:
            current_backend = plt.get_backend()
            if QUICKSTART_CONFIG['verbose']:
                print(f"Aktualny backend matplotlib: {current_backend}")
            if not current_backend.lower() == 'agg':
                plt.switch_backend('Agg')
                if QUICKSTART_CONFIG['verbose']:
                    print("Zmieniono backend matplotlib na 'Agg' dla działania w tle.")
        except Exception as e:
            if QUICKSTART_CONFIG['verbose']:
                print(f"Nie udało się zmienić backendu matplotlib: {e}")
    elif QUICKSTART_CONFIG['verbose']:
        print(f"Na macOS ('Darwin'), zakładamy, że backend matplotlib jest zarządzany przez analyze_features.py lub jest odpowiedni.")

    run_full_pipeline_quickstart()