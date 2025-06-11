# ECGModel: Predykcja Śmiertelności Wewnątrzszpitalnej na Podstawie Danych Klinicznych i Sygnałów EKG

## Cel Projektu

Głównym celem projektu jest opracowanie i ocena modeli uczenia maszynowego zdolnych do predykcji ryzyka śmiertelności wewnątrzszpitalnej pacjentów. Modele te wykorzystują połączone dane kliniczne oraz cechy wyekstrahowane z wieloodprowadzeniowych sygnałów elektrokardiograficznych (EKG).

## Źródła Danych

Projekt bazuje na dwóch głównych typach danych:
1.  **Dane Kliniczne**: Pochodzące z plików Excel (np. `data/clinical_data.xlsx`), zawierające informacje demograficzne, wyniki badań laboratoryjnych, parametry życiowe, historię chorób oraz stosowane leczenie.
2.  **Sygnały EKG**: Zapisane w formacie `.json` (np. w katalogu `data/ecg/`), zawierające surowe dane z 12 standardowych odprowadzeń EKG.

## Ekstrakcja Cech

Kluczowym elementem projektu jest zaawansowana ekstrakcja cech z obu źródeł danych.

### Cechy Kliniczne (z danych tabelarycznych)
Przetwarzane przez skrypt `src/data/process_excel_data.py`. Obejmują m.in.:
-   **Dane demograficzne**: Wiek, płeć.
-   **Parametry życiowe**: Ciśnienie tętnicze (MAP), temperatura, skala Glasgow (GCS).
-   **Wyniki badań laboratoryjnych**: Poziomy bilirubiny, kreatyniny, hemoglobiny (HGB), PaO2.
-   **Informacje o leczeniu**: Stosowanie wentylacji mechanicznej, dawek leków (np. dopamina, noradrenalina).
-   **Choroby współistniejące**: Cukrzyca, nadciśnienie tętnicze, POChP, marskość wątroby.

### Cechy z Sygnałów EKG
Ekstrahowane przez skrypt `src/data/process_time_series.py` przy użyciu biblioteki **NeuroKit2**. Dla każdego z 12 odprowadzeń EKG wyliczane są m.in.:
-   **Statystyki podstawowe**: Średnia, odchylenie standardowe, skośność, kurtoza, RMS.
-   **Analiza rytmu serca i zmienności rytmu serca (HRV)**:
    -   Częstość akcji serca (Heart Rate).
    -   RMSSD (Root Mean Square of Successive Differences).
    -   SDNN (Standard Deviation of NN intervals).
    -   Stosunek mocy pasma niskiej częstotliwości do wysokiej częstotliwości (LF/HF ratio).
-   **Analiza morfologii załamków EKG**:
    -   **Załamek P**: Czas trwania, amplituda, zmienność amplitudy.
    -   **Zespół QRS**: Czas trwania, zmienność czasu trwania, amplituda załamka R.
    -   **Odcinek ST i załamek T**: Poziom odcinka ST, nachylenie odcinka ST, amplituda załamka T.
    -   **Odstęp QT**: Średni skorygowany odstęp QT (QTc), odchylenie standardowe QTc.
-   **Zaawansowane cechy**:
    -   Alternans załamka T (TWA).
    -   Entropia sygnału (np. entropia Shannona).
    -   Złożoność sygnału.
    -   Stosunek sygnału do szumu (SNR).
-   **Jakość sygnału**: Ocena jakości każdego odprowadzenia.

## Potok Przetwarzania Danych i Modelowania

Projekt realizuje kompleksowy potok analizy danych:

1.  **Przetwarzanie Danych Surowych**:
    *   `src/data/process_excel_data.py`: Czyści, transformuje i przygotowuje dane kliniczne, zapisując je do `data/processed_tabular.csv`.
    *   `src/data/process_time_series.py`: Przetwarza pliki EKG `.json`, ekstrahuje cechy i zapisuje je do `data/processed_ecg_features.csv`.
    *   Dane z obu źródeł są następnie łączone (np. przez `src/data/process_every_feature_to_csv.py` lub podobny skrypt) w jeden plik `data/processed_features.csv`.

2.  **Analiza i Selekcja Cech**:
    *   Skrypt: `src/insight/analyze_features.py`.
    *   **Metody analizy ważności**:
        -   Statystyczne (ANOVA F-value, Mutual Information).
        -   Oparte na modelach (Random Forest, XGBoost, Gradient Boosting, CatBoost, SVM, Logistic Regression).
        -   Wartości SHAP (jeśli biblioteka `shap` jest dostępna).
    *   **Analiza dodatkowa**:
        -   Analiza głównych składowych (PCA).
        -   Analiza korelacji cech (z celem i między sobą).
        -   Klastrowanie cech o podobnej ważności.
        -   Analiza interakcji między cechami.
    *   **Wyniki**: Wykresy ważności cech, podsumowania w plikach `.csv` (np. `feature_importance_summary.csv`) zapisywane w `data/feature_analysis/`.
    *   **Selekcja**: Na podstawie zagregowanej ważności cech, wybierane jest `N` najważniejszych cech, które są zapisywane do `data/selected_features.csv`.

3.  **Optymalizacja Hiperparametrów Modeli**:
    *   Skrypt: `src/models/optimize_model.py`.
    *   **Metoda**: Optymalizacja Bayesowska (biblioteka `scikit-optimize`).
    *   **Modele poddawane optymalizacji**:
        -   Random Forest (`RandomForestModel`)
        -   Gradient Boosting (`GradientBoostingModel`)
        -   Support Vector Machine (`SVMModel`)
        -   Logistic Regression (`LogisticRegressionModel`)
        -   CatBoost (`CatBoostModel`)
        -   XGBoost (`XGBoostModel`)
    *   **Metryka oceny**: F1-score (lub inna zdefiniowana).
    *   **Wyniki**: Najlepsze zestawy hiperparametrów zapisywane są do plików `.json` (np. `xgboost_best_params.json`) w katalogu `model_optimization_results/`. Zapisywane są również wytrenowane modele (`.joblib`).

4.  **Trening i Ewaluacja Modeli**:
    *   Skrypt: `src/models/train_models.py`.
    *   Wczytuje zoptymalizowane hiperparametry (jeśli dostępne) lub używa domyślnych.
    *   Trenuje i ocenia modele na zbiorze `data/selected_features.csv` (lub `data/quickstart_input_data.csv`).
    *   **Ewaluacja**: Walidacja krzyżowa oraz ocena na wydzielonym zbiorze testowym.
    *   **Metryki**: Accuracy, F1-score, Precision, Recall, ROC AUC.
    *   **Wyniki**: Porównanie modeli zapisywane do `model_comparison_results.csv`.

5.  **Zaawansowana Analiza Modeli**:
    *   Skrypt: `src/insight/analyze_models.py`.
    *   Przeprowadza szczegółową analizę każdego modelu, wykorzystując zoptymalizowane parametry.
    *   Generuje:
        -   Krzywe uczenia.
        -   Macierze konfuzji.
        -   Krzywe ROC i PR.
        -   Raporty klasyfikacji.
    *   **Wyniki**: Zapisywane w podkatalogach w `data/model_analysis/` oraz zbiorcze podsumowanie `all_models_metrics_summary.csv`.

## Skrypt `quickstart.py`

Skrypt `src/insight/quickstart.py` automatyzuje część potoku:
1.  Uruchamia `analyze_features.py` na `data/processed_features.csv`.
2.  Na podstawie wyników analizy cech tworzy `data/selected_features.csv` (wybierając `N` najlepszych cech). Kopia tego pliku jest zapisywana jako `data/quickstart_input_data.csv`.
3.  Uruchamia `analyze_models.py` na `data/selected_features.csv`.

## Kluczowe Technologie i Biblioteki

-   **Python 3.x**
-   **Pandas**: Manipulacja danymi.
-   **NumPy**: Obliczenia numeryczne.
-   **Scikit-learn**: Implementacje modeli uczenia maszynowego, metryki, walidacja krzyżowa, PCA.
-   **NeuroKit2**: Zaawansowana analiza sygnałów EKG.
-   **SHAP**: Interpretowalność modeli (wyjaśnianie predykcji).
-   **CatBoost**: Implementacja modelu gradient boosting.
-   **XGBoost**: Implementacja modelu gradient boosting.
-   **Matplotlib**: Wizualizacja danych i wyników.
-   **Scikit-optimize (`skopt`)**: Optymalizacja Bayesowska hiperparametrów.
-   **TQDM**: Paski postępu.
-   **Joblib**: Zapisywanie i wczytywanie modeli.

## Struktura Katalogów i Plików (Kluczowe Elementy)

-   `data/`:
    -   `clinical_data.xlsx`: Surowe dane kliniczne (wejście).
    -   `ecg/`: Katalog z plikami EKG `.json` (wejście).
    -   `processed_tabular.csv`: Przetworzone dane kliniczne.
    -   `processed_ecg_features.csv`: Wyekstrahowane cechy EKG.
    -   `processed_features.csv`: Połączone i w pełni przetworzone cechy (kluczowe wejście dla analiz).
    -   `selected_features.csv`: Cechy wybrane do modelowania.
    -   `quickstart_input_data.csv`: Kopia `selected_features.csv` generowana przez `quickstart.py`.
    -   `feature_analysis/`: Wyniki analizy ważności cech.
    -   `model_analysis/`: Wyniki szczegółowej analizy modeli.
-   `model_optimization_results/`:
    -   `*_best_params.json`: Zoptymalizowane hiperparametry dla modeli.
    -   `*_best_model.joblib`: Zapisane najlepsze modele.
-   `src/`: Kod źródłowy projektu.
    -   `data/`: Skrypty do przetwarzania danych surowych.
    -   `models/`: Implementacje modeli, skrypty do optymalizacji i treningu.
    -   `insight/`: Skrypty do analizy cech, analizy modeli oraz `quickstart.py`.
-   `model_comparison_results.csv`: Zbiorcze wyniki ewaluacji modeli z `train_models.py`.

## Jak Uruchomić

1.  **Przygotowanie Danych Wejściowych**:
    *   Upewnij się, że plik `data/clinical_data.xlsx` oraz katalog `data/ecg/` z plikami `.json` istnieją.
    *   Uruchom skrypty z `src/data/` (np. `process_excel_data.py`, `process_time_series.py`) oraz skrypt łączący, aby wygenerować `data/processed_features.csv`.

2.  **Analiza Cech, Selekcja i Analiza Modeli (Quickstart)**:
    *   Uruchom: `python src/insight/quickstart.py`
    *   Ten skrypt przeprowadzi analizę cech na `data/processed_features.csv`, wybierze najważniejsze cechy do `data/selected_features.csv`, a następnie przeprowadzi analizę modeli na tych wybranych cechach.

3.  **Optymalizacja Modeli (Opcjonalnie, przed `quickstart` lub `train_models`)**:
    *   Uruchom: `python src/models/optimize_model.py`
    *   Ten skrypt przeprowadzi optymalizację hiperparametrów dla zdefiniowanych modeli, korzystając z `data/selected_features.csv` (lub innego zbioru, jeśli zmodyfikowano skrypt). Wyniki zostaną zapisane w `model_optimization_results/`.

4.  **Trening i Ewaluacja Modeli (Opcjonalnie, jeśli chcesz tylko wytrenować modele z najlepszymi parametrami)**:
    *   Uruchom: `python src/models/train_models.py`
    *   Skrypt ten wczyta `data/selected_features.csv` (lub `data/quickstart_input_data.csv`, jeśli istnieje) oraz zoptymalizowane parametry (jeśli istnieją) i przeprowadzi ewaluację modeli.

## Wymagania
Zainstaluj wymagane biblioteki, np. używając `pip install -r requirements.txt` (jeśli plik `requirements.txt` jest dostępny i aktualny). Kluczowe biblioteki zostały wymienione w sekcji "Kluczowe Technologie i Biblioteki".

