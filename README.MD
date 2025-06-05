# ECGModel - Przetwarzanie danych klinicznych i sygnałów EKG

Projekt ten ma na celu kompleksowe przetwarzanie danych klinicznych oraz sygnałów EKG w celu przygotowania ich do dalszych analiz i modelowania, np. predykcji śmiertelności wewnątrzszpitalnej.

## Szybki start

Aby przetworzyć wszystkie dane, wygenerować plik z kompletem cech oraz wybrać najważniejsze z nich, uruchom skrypt `quickstart.py` znajdujący się w katalogu `src/data/`:

```bash
python src/data/quickstart.py
```

Skrypt ten wykonuje następujące kroki:
1.  Tworzy niezbędne katalogi (`data/feature_analysis`, `data/model_analysis`, `output`).
2.  Uruchamia `src/data/process_data.py` w celu przetworzenia surowych danych z plików Excel (`clinical_data.xlsx`) oraz JSON (sygnały EKG z folderu `ecg/`). Wynikiem jest plik `data/processed_features.csv`.
3.  Uruchamia `src/data/select_the_most_important_features.py`, który na podstawie analizy ważności cech (generowanej, jeśli nie istnieje) wybiera `k` najważniejszych cech i zapisuje je do `data/selected_features.csv`.

## Wynikowe pliki

Główne pliki generowane przez proces przetwarzania:
-   `data/processed_tabular.csv`: Przetworzone dane kliniczne z pliku Excel.
-   `data/processed_ecg_features.csv`: Cechy wyekstrahowane z sygnałów EKG.
-   `data/processed_features.csv`: Połączony zbiór danych, zawierający zarówno cechy kliniczne, jak i cechy EKG dla wspólnych pacjentów. **To jest główny plik wynikowy z kompletem cech.**
-   `data/selected_features.csv`: Plik zawierający najważniejsze cechy wybrane przez skrypt `select_the_most_important_features.py` wraz ze zmienną docelową `ZGON`.

## Opis cech

Proces ekstrakcji cech jest realizowany przez skrypty w katalogu `src/data/`. Poniżej znajduje się bardziej szczegółowy opis.

### Cechy z sygnałów EKG (time series)

Ekstrakcja cech z sygnałów EKG odbywa się w skrypcie `src/data/process_time_series.py`. Dla każdego z 12 standardowych odprowadzeń EKG (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`) wyznaczane są liczne cechy kliniczne i statystyczne. Wykorzystywana jest do tego głównie biblioteka **NeuroKit2**.

Proces dla pojedynczego odprowadzenia obejmuje:
1.  **Wstępne przetwarzanie sygnału:**
    *   Czyszczenie sygnału (np. filtrowanie, usuwanie trendu) za pomocą `nk.ecg_clean()`.
    *   Ocena jakości sygnału (`is_valid_signal()`); cechy nie są ekstrahowane dla sygnałów o niskiej jakości.
2.  **Podstawowe statystyki sygnału:**
    *   Średnia, odchylenie standardowe, skośność, kurtoza, RMS, liczba przejść przez zero.
3.  **Analiza rytmu serca i detekcja R-peaków:**
    *   Detekcja załamków R (`nk.ecg_peaks()`) w celu identyfikacji poszczególnych uderzeń serca.
    *   Obliczenie interwałów RR.
4.  **Analiza zmienności rytmu serca (HRV):**
    *   **Dziedzina czasu:** np. RMSSD, SDNN (`nk.hrv_time()`).
    *   **Dziedzina częstotliwości:** np. moc w pasmach LF (Low Frequency), HF (High Frequency), stosunek LF/HF (`nk.hrv_frequency()`).
5.  **Delineacja załamków EKG i analiza morfologii:**
    *   Identyfikacja granic (początków i końców) oraz szczytów załamków P, QRS i T (`nk.ecg_delineate()`).
    *   **Załamek P:** czas trwania, amplituda, zmienność amplitudy.
    *   **Zespół QRS:** czas trwania, zmienność czasu trwania, amplituda załamka R.
    *   **Odcinek ST i załamek T:** średni poziom ST, nachylenie ST, amplituda załamka T.
6.  **Analiza interwału QT:**
    *   Obliczenie czasu trwania interwału QT.
    *   Korekcja interwału QT względem częstości rytmu serca (QTc), np. przy użyciu formuły Bazetta.
7.  **Cechy zaawansowane:**
    *   **Alternans załamka T (TWA):** Zmienność amplitudy załamka T w kolejnych pobudzeniach.
    *   **Entropia sygnału:** np. entropia Shannona jako miara złożoności/regularności sygnału.
    *   **Stosunek sygnału do szumu (SNR):** Jako miara jakości sygnału.
    *   Złożoność sygnału.

Nazwy cech w wynikowym pliku CSV są tworzone w formacie `NAZWAODPROWADZENIA_NAZWACECHY` (np. `V1_hrv_rmssd`).

### Cechy z Excela (dane kliniczne)

Przetwarzanie danych klinicznych z pliku `data/clinical_data.xlsx` odbywa się w skrypcie `src/data/process_excel_data.py`.
Wykorzystywane są wybrane kolumny, zdefiniowane na liście `admissible_features` w tym skrypcie. Obejmują one szeroki zakres informacji:
-   **Dane demograficzne:** np. `WIEK`, `PLEC` (konwertowane na `male sex`).
-   **Parametry fizjologiczne przy przyjęciu i w pierwszej dobie:** np. `GLASGOW`, `MAP 1sza doba`, `pao2/fio2 1sza doba`, parametry z TISS.
-   **Wyniki badań laboratoryjnych:** np. `pH`, `pCO2`, `Hb`, `K`, `Na`, `Lac`, `Crea`, `Bil`.
-   **Choroby współistniejące i historia medyczna:** np. `cukrzyca`, `nadciśnienie`, `niewydolność nerek przewlekła`, `COVID-19`.
-   **Interwencje i leczenie:** np. `Operowany przed przyjęciem (0/1)`, `Wentylacja mechaniczna`, stosowane leki (np. `dopamina`, `noradrenalina`).
-   **Zmienna docelowa:** `ZGON wewnątrzszpitalnie` (przekształcana na binarną kolumnę `ZGON`).

Indeks danych jest ustawiany na numer historii choroby (`KG`), a niektóre kolumny są czyszczone lub przekształcane (np. `PLEC` na `male sex`).

### Łączenie danych

Skrypt `src/data/process_every_feature_to_csv.py` (wywoływany przez `process_data.py`) odpowiada za:
1.  Wywołanie `process_excel_data.py` w celu przetworzenia danych tabelarycznych.
2.  Wywołanie `process_time_series.py` w celu ekstrakcji cech z EKG.
3.  **Normalizację identyfikatorów pacjentów (ID):** Identyfikatory z danych Excel (np. `123/45`) i danych EKG (np. `123_45`) są normalizowane do wspólnego formatu (np. `123_45`) w celu umożliwienia poprawnego połączenia.
4.  Połączenie obu zbiorów danych (cech klinicznych i cech EKG) na podstawie wspólnych, znormalizowanych identyfikatorów pacjentów. Wynikiem jest plik `data/processed_features.csv`.

## Uwaga:
-   Plik `clinical_data.xlsx` oraz folder `ecg/` z plikami `.json` muszą być obecne w katalogu `data/` przed uruchomieniem przetwarzania.
-   Pozostałe pliki CSV oraz katalogi  `data/feature_analysis`, `data/model_analysis` są generowane automatycznie przez skrypty.
