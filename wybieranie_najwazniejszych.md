### Szczegółowa Analiza Metod Selekcji Cech (na podstawie `analyze_features.py`)

Skrypt nie polega na jednej metodzie, ale stosuje **zdywersyfikowany portfel technik**, aby uniknąć stronniczości pojedynczego podejścia. Wyniki z tych wszystkich metod są na końcu agregowane, aby uzyskać ostateczny ranking.

#### Metoda 1: Analiza Ważności w Modelach (Model-Based Importance)

To rdzeń analizy. Skrypt trenuje kilka różnych typów modeli uczenia maszynowego i pyta każdy z nich: "które cechy były dla ciebie najważniejsze przy podejmowaniu decyzji?".

*   **Modele Drzewiaste (`RandomForest`, `XGBoost`, `GradientBoosting`, `CatBoost`):**
    *   **Jak to działa:** Te modele naturalnie obliczają ważność cech (`feature_importances_`) na podstawie tego, jak często dana cecha była używana do podziału danych w drzewach decyzyjnych i jak bardzo ten podział poprawiał "czystość" węzłów.
    *   **Funkcja w kodzie:** `_generate_feature_importance_worker` sprawdza, czy model ma atrybut `feature_importances_` lub metodę `get_feature_importance` (dla CatBoost).

*   **Modele Liniowe (`LogisticRegression`, `SVM` z jądrem liniowym):**
    *   **Jak to działa:** W modelach liniowych ważność cechy jest odzwierciedlona przez wartość bezwzględną jej współczynnika (`coef_`). Im większy współczynnik (dodatni lub ujemny), tym większy wpływ ma zmiana tej cechy na wynik predykcji.
    *   **Funkcja w kodzie:** `_generate_feature_importance_worker` sprawdza atrybut `coef_`.

*   **Modele "Czarnej Skrzynki" (np. `SVM` z jądrem nieliniowym RBF):**
    *   **Jak to działa:** Ponieważ te modele nie mają prostych współczynników ani drzew, skrypt stosuje potężną technikę **ważności permutacji (`permutation_importance`)**. Polega ona na losowym "przetasowaniu" wartości w jednej kolumnie (cesze) i zmierzeniu, o ile pogorszyło to wydajność modelu. Duży spadek wydajności oznacza, że cecha jest bardzo ważna. Proces jest powtarzany wielokrotnie (`permutation_repeats: 10`), aby uzyskać stabilny wynik.
    *   **Funkcja w kodzie:** Jest to domyślne (fallback) zachowanie w `_generate_feature_importance_worker`.

*   **Analiza SHAP (SHapley Additive exPlanations):**
    *   **Jak to działa:** To najbardziej zaawansowana metoda w tym skrypcie. SHAP nie tylko mówi, **które** cechy są ważne, ale także **jak** wpływają na predykcję (czy podnoszą, czy obniżają ryzyko `ZGON`) dla każdego pojedynczego pacjenta. Skrypt oblicza średnią wartość bezwzględną SHAP dla każdej cechy, co daje globalny ranking ważności.
    *   **Kod:** Sprawdzenie `if SHAP_AVAILABLE` i użycie `shap.KernelExplainer`.

#### Metoda 2: Analiza Statystyczna (Filter Methods)

Te metody oceniają cechy niezależnie od jakiegokolwiek modelu, bazując tylko na ich statystycznych właściwościach w odniesieniu do zmiennej docelowej (`ZGON`).

*   **ANOVA F-value (`f_classif`):**
    *   **Jak to działa:** Testuje, czy średnie wartości danej cechy są znacząco różne dla różnych klas wynikowych (np. czy średnie tętno jest inne dla pacjentów, którzy przeżyli, a inne dla tych, którzy zmarli). Wysoka wartość F oznacza dużą różnicę.
    *   **Funkcja w kodzie:** `analyze_statistical_parallel` i `statistical_worker_function`.

*   **Informacja Wzajemna (`mutual_info_classif`):**
    *   **Jak to działa:** Mierzy, jak dużo informacji o zmiennej `ZGON` dostarcza znajomość wartości danej cechy. Jest to potężna metryka, ponieważ potrafi wychwycić dowolne zależności, nie tylko liniowe.
    *   **Funkcja w kodzie:** `analyze_statistical_parallel` i `statistical_worker_function`.

#### Metoda 3: Analiza Korelacji i Struktury Danych

Skrypt przeprowadza również głębszą analizę relacji między cechami.

*   **Korelacja z celem:** Oblicza prosty współczynnik korelacji każdej cechy ze zmienną `ZGON`, aby znaleźć cechy o najsilniejszym liniowym związku z wynikiem.
*   **Wielowspółliniowość:** Wykrywa pary cech, które są silnie skorelowane **ze sobą**. Jest to ważne, ponieważ bardzo podobne cechy dostarczają tych samych informacji i mogą destabilizować niektóre modele.
*   **PCA (Analiza głównych składowych):** Redukuje wymiarowość danych, aby zobaczyć, które oryginalne cechy mają największy wkład w główne "osie" zmienności w całym zbiorze danych.

---

### Krok Kluczowy: Agregacja Wyników i Stworzenie Ostatecznego Rankingu

To jest najważniejszy moment całego procesu selekcji, który odbywa się na końcu funkcji `main()` w `analyze_features.py`:

1.  **Zebranie wszystkich wyników:** Skrypt tworzy jedną dużą tabelę (`summary`), gdzie kolumny to różne metody oceny (RandomForest_importance, XGBoost_importance, ANOVA_F, SHAP_values itd.), a wiersze to wszystkie cechy.
2.  **Ranking dla każdej metody:** Dla każdej kolumny (każdej metody) skrypt oblicza ranking cech od najważniejszej do najmniej ważnej (`summary[f'{col}_rank'] = ...`).
3.  **Obliczenie Średniego Rankingu:** Następnie skrypt oblicza **średni ranking (`mean_rank`)** dla każdej cechy, uśredniając jej pozycję we wszystkich przeprowadzonych analizach.
4.  **Sortowanie:** Finalna tabela jest sortowana według tej zagregowanej metryki `mean_rank`.

**Dlaczego to takie skuteczne?** Cecha, która jest uznawana za ważną przez wiele różnych modeli (drzewiastych, liniowych) ORAZ ma silny związek statystyczny z wynikiem, otrzyma niski (czyli dobry) średni ranking. To niezwykle solidne podejście, które minimalizuje ryzyko, że jakaś cecha została uznana za ważną przez przypadek lub z powodu specyfiki jednego algorytmu.

**Wynik:** Powstaje plik `feature_importance_summary.csv` zawierający ostateczny, zagregowany i bardzo wiarygodny ranking wszystkich cech.

### Powrót do `quickstart.py`: Ostateczna Selekcja "Top N"

Teraz skrypt `quickstart.py` (ten "dyrygent") wchodzi do akcji, wykonując finalny, prosty krok:

1.  Wczytuje plik `feature_importance_summary.csv`.
2.  Sortuje go po kolumnie `mean_rank`.
3.  **Wybiera `N` pierwszych cech** z tej posortowanej listy (gdzie `N` jest zdefiniowane jako `top_n_features_for_selection: 80`).
4.  Tworzy ostateczny plik `selected_features.csv`, który zawiera już tylko te 80 najlepszych cech i jest gotowy do użycia przez skrypt `analyze_models.py`.

