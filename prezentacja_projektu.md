# Analiza Danych Klinicznych i Sygnałów EKG dla Predykcji Śmiertelności Wewnątrzszpitalnej
## Prezentacja Projektu

### Spis Treści
1. [Wprowadzenie i Cele Projektu](#1-wprowadzenie-i-cele-projektu)
2. [Metodologia](#2-metodologia)
   - [Przetwarzanie Danych](#21-przetwarzanie-danych)
   - [Ekstrakcja Cech](#22-ekstrakcja-cech)
   - [Analiza Cech](#23-analiza-cech)
   - [Selekcja Cech](#24-selekcja-cech)
3. [Modelowanie](#3-modelowanie)
   - [Implementowane Modele](#31-implementowane-modele)
   - [Proces Uczenia](#32-proces-uczenia)
   - [Ewaluacja Modeli](#33-ewaluacja-modeli)
4. [Wyniki](#4-wyniki)
   - [Porównanie Modeli](#41-porównanie-modeli)
   - [Najważniejsze Cechy](#42-najważniejsze-cechy)
5. [Wnioski i Przyszłe Kierunki](#5-wnioski-i-przyszłe-kierunki)

## 1. Wprowadzenie i Cele Projektu

Projekt "ECGModel" został stworzony w celu kompleksowego przetwarzania i analizy danych klinicznych oraz sygnałów EKG pacjentów, aby przewidzieć śmiertelność wewnątrzszpitalną. Główne cele projektu to:

- Opracowanie efektywnego potoku przetwarzania danych klinicznych i sygnałów EKG
- Ekstrakcja istotnych cech z sygnałów EKG przy użyciu zaawansowanych technik analizy sygnałów
- Identyfikacja najważniejszych czynników wpływających na śmiertelność wewnątrzszpitalną
- Stworzenie i porównanie różnych modeli uczenia maszynowego do predykcji śmiertelności
- Dostarczenie narzędzi do szybkiej analizy i wizualizacji wyników

Projekt ma potencjał do zastosowania w praktyce klinicznej, wspierając lekarzy w podejmowaniu decyzji dotyczących leczenia pacjentów wysokiego ryzyka.

## 2. Metodologia

### 2.1 Przetwarzanie Danych

Proces przetwarzania danych obejmuje kilka kluczowych etapów:

1. **Wczytywanie danych źródłowych**:
   - Dane kliniczne z pliku Excel (`clinical_data.xlsx`)
   - Sygnały EKG z plików JSON w katalogu `ecg/`

2. **Normalizacja identyfikatorów pacjentów**:
   - Ujednolicenie formatów ID między danymi klinicznymi a sygnałami EKG
   - Zapewnienie poprawnego łączenia danych z różnych źródeł

3. **Czyszczenie danych**:
   - Usuwanie nieprawidłowych wartości
   - Obsługa brakujących danych
   - Konwersja typów danych (np. zmiana płci na zmienną binarną)

4. **Łączenie danych**:
   - Połączenie przetworzonych danych klinicznych z cechami wyekstrahowanymi z EKG
   - Tworzenie kompletnego zbioru danych dla dalszej analizy

Wynikiem tego procesu są pliki:
- `processed_tabular.csv` - przetworzone dane kliniczne
- `processed_ecg_features.csv` - cechy wyekstrahowane z EKG
- `processed_features.csv` - połączony zbiór danych

### 2.2 Ekstrakcja Cech

#### Cechy z Sygnałów EKG

Ekstrakcja cech z sygnałów EKG jest realizowana przy użyciu biblioteki NeuroKit2 i obejmuje:

1. **Wstępne przetwarzanie sygnału**:
   - Czyszczenie sygnału (filtrowanie, usuwanie trendu)
   - Ocena jakości sygnału

2. **Podstawowe statystyki sygnału**:
   - Średnia, odchylenie standardowe, skośność, kurtoza
   - RMS (Root Mean Square), liczba przejść przez zero

3. **Analiza rytmu serca**:
   - Detekcja załamków R
   - Obliczenie interwałów RR

4. **Analiza zmienności rytmu serca (HRV)**:
   - Parametry w dziedzinie czasu (RMSSD, SDNN)
   - Parametry w dziedzinie częstotliwości (moc w pasmach LF, HF, stosunek LF/HF)

5. **Delineacja załamków EKG**:
   - Identyfikacja granic i szczytów załamków P, QRS i T
   - Analiza morfologii załamków

6. **Analiza interwału QT**:
   - Obliczenie czasu trwania interwału QT
   - Korekcja interwału QT względem częstości rytmu serca (QTc)

7. **Cechy zaawansowane**:
   - Alternans załamka T
   - Entropia sygnału
   - Stosunek sygnału do szumu (SNR)
   - Złożoność sygnału

Cechy są ekstrahowane dla każdego z 12 standardowych odprowadzeń EKG (I, II, III, aVR, aVL, aVF, V1-V6).

#### Cechy Kliniczne

Z danych klinicznych ekstrahowane są następujące kategorie cech:

1. **Dane demograficzne**:
   - Wiek, płeć

2. **Parametry fizjologiczne**:
   - Skala Glasgow
   - Średnie ciśnienie tętnicze (MAP)
   - Stosunek PaO2/FiO2
   - Parametry z TISS

3. **Wyniki badań laboratoryjnych**:
   - pH, pCO2
   - Hemoglobina, elektrolity (K, Na)
   - Mleczany, kreatynina, bilirubina

4. **Choroby współistniejące**:
   - Cukrzyca, nadciśnienie
   - Niewydolność nerek przewlekła
   - COVID-19

5. **Interwencje i leczenie**:
   - Operacje przed przyjęciem
   - Wentylacja mechaniczna
   - Stosowane leki (dopamina, noradrenalina)

6. **Zmienna docelowa**:
   - Zgon wewnątrzszpitalny (kolumna `ZGON`)

### 2.3 Analiza Cech

Po ekstrakcji cech przeprowadzana jest szczegółowa analiza ich znaczenia:

1. **Metody statystyczne**:
   - ANOVA (f_classif)
   - Informacja wzajemna (mutual_info_classif)

2. **Metody oparte na modelach**:
   - Ważność cech z modelu Random Forest
   - Ważność cech z modelu Gradient Boosting
   - Ważność permutacyjna

3. **Wizualizacje**:
   - Wykresy ważności cech
   - Mapy cieplne korelacji
   - Wykresy SHAP dla interpretacji modeli

Analiza ta pozwala zidentyfikować cechy najbardziej istotne dla predykcji śmiertelności wewnątrzszpitalnej.

### 2.4 Selekcja Cech

Na podstawie analizy ważności cech przeprowadzana jest selekcja najistotniejszych zmiennych:

1. **Ranking cech**:
   - Sortowanie cech według ich ważności
   - Tworzenie rankingu na podstawie różnych metod

2. **Wybór top-k cech**:
   - Selekcja 80 najważniejszych cech (parametr konfigurowalny)
   - Zapisanie wybranych cech wraz z zmienną docelową do pliku `selected_features.csv`

3. **Walidacja wyboru**:
   - Sprawdzenie, czy wybrane cechy istnieją w oryginalnym zbiorze danych
   - Weryfikacja kompletności danych dla wybranych cech

Selekcja cech pozwala na redukcję wymiarowości danych, co poprawia wydajność modeli i zmniejsza ryzyko przeuczenia.

## 3. Modelowanie

### 3.1 Implementowane Modele

W projekcie zaimplementowano kilka modeli uczenia maszynowego, każdy z nich ma swoje zalety i ograniczenia:

1. **Random Forest**:
   - Odporność na przeuczenie
   - Dobra wydajność dla danych o wysokiej wymiarowości
   - Wbudowana ocena ważności cech

2. **Gradient Boosting**:
   - Wysoka dokładność predykcji
   - Sekwencyjne uczenie się na błędach poprzednich drzew
   - Możliwość dostosowania tempa uczenia

3. **Support Vector Machine (SVM)**:
   - Efektywność w przestrzeniach o wysokiej wymiarowości
   - Różne jądra (liniowe, RBF) dla różnych typów danych
   - Dobra generalizacja przy odpowiedniej regularyzacji

4. **Logistic Regression**:
   - Prostota i interpretowalność
   - Efektywność obliczeniowa
   - Dobre wyniki dla liniowo separowalnych danych

5. **CatBoost**:
   - Obsługa zmiennych kategorycznych
   - Odporność na przeuczenie
   - Wysoka wydajność

6. **XGBoost**:
   - Wysoka wydajność obliczeniowa
   - Zaawansowana regularyzacja
   - Obsługa brakujących danych

Wszystkie modele dziedziczą po klasie bazowej `ECGBaseModel`, która zapewnia wspólny interfejs i funkcjonalności, takie jak skalowanie danych i ewaluacja.

### 3.2 Proces Uczenia

Proces uczenia modeli obejmuje następujące kroki:

1. **Przygotowanie danych**:
   - Wczytanie wybranych cech z pliku `selected_features.csv`
   - Podział na zbiory treningowy i testowy (80% / 20%)
   - Skalowanie cech przy użyciu StandardScaler

2. **Trenowanie modeli**:
   - Dopasowanie każdego modelu do danych treningowych
   - Zastosowanie walidacji krzyżowej (domyślnie 5-krotna)
   - Zapisanie wytrenowanych modeli

3. **Optymalizacja hiperparametrów**:
   - Dla wybranych modeli przeprowadzana jest optymalizacja hiperparametrów
   - Wykorzystanie przeszukiwania siatki lub losowego przeszukiwania
   - Wybór najlepszych parametrów na podstawie wyników walidacji krzyżowej

### 3.3 Ewaluacja Modeli

Ewaluacja modeli jest przeprowadzana przy użyciu różnych metryk i technik:

1. **Metryki wydajności**:
   - Dokładność (Accuracy)
   - Precyzja (Precision)
   - Czułość (Recall)
   - F1-Score
   - AUC-ROC
   - Average Precision (AP)

2. **Techniki wizualizacji**:
   - Krzywe ROC
   - Krzywe Precision-Recall
   - Macierze konfuzji
   - Krzywe uczenia

3. **Analiza błędów**:
   - Identyfikacja przypadków fałszywie pozytywnych i fałszywie negatywnych
   - Analiza trudnych przypadków

Wyniki ewaluacji są zapisywane w katalogu `data/model_analysis/` i obejmują zarówno metryki liczbowe, jak i wizualizacje.

## 4. Wyniki

### 4.1 Porównanie Modeli

Na podstawie przeprowadzonych eksperymentów, modele można porównać pod względem różnych metryk:

| Model | Accuracy | F1-Score | Precision | Recall | AUC-ROC | Average Precision |
|-------|----------|----------|-----------|--------|---------|---------|
| RandomForest | 0.7089 | 0.7810 | 0.7069 | 0.8723 | 0.7886 | 0.8551 |
| GradientBoosting | 0.7342 | 0.7921 | 0.7407 | 0.8511 | 0.7939 | 0.8681 |
| SVM_RBF | 0.7848 | 0.8247 | 0.8000 | 0.8511 | 0.8457 | 0.8946 |
| LogisticRegression | 0.7342 | 0.7742 | 0.7826 | 0.7660 | 0.8491 | 0.8967 |
| CatBoost | 0.6962 | 0.7551 | 0.7255 | 0.7872 | 0.7766 | 0.8651 |
| XGBoost | 0.6709 | 0.7400 | 0.6981 | 0.7872 | 0.7600 | 0.8312 |

Najlepsze wyniki osiągnął model **SVM_RBF**, który uzyskał najwyższe wartości dla dokładności (Accuracy: 0.7848), F1-Score (0.8247) i precyzji (Precision: 0.8000). Model **LogisticRegression** osiągnął najwyższy wynik dla AUC-ROC (0.8491) i Average Precision (0.8967), co wskazuje na jego dobrą zdolność do rozróżniania klas.

### 4.2 Najważniejsze Cechy

Analiza ważności cech wykazała, że następujące cechy mają największy wpływ na predykcję śmiertelności wewnątrzszpitalnej:

1. **sodium chloride difference tiss 3** - różnica w poziomie chlorku sodu, parametr związany z równowagą elektrolitową pacjenta
2. **FIO2 (TISS nr 1)** - frakcja wdychanego tlenu, parametr wentylacji mechanicznej
3. **akcja serca przyjęcie (TISS nr 1)** - częstość akcji serca przy przyjęciu pacjenta
4. **BILI TISS 1** - poziom bilirubiny, marker funkcji wątroby
5. **Lac (1. gaz. 1sza doba)** - poziom mleczanów, marker niedotlenienia tkankowego

Wśród najważniejszych cech dominują parametry kliniczne związane z funkcjami życiowymi i biochemicznymi markerami, co sugeruje, że stan fizjologiczny pacjenta przy przyjęciu ma kluczowe znaczenie dla przewidywania śmiertelności wewnątrzszpitalnej. Szczególnie istotne są parametry związane z równowagą elektrolitową, wentylacją, funkcją serca, wątroby oraz metabolizmem tlenowym.

## 5. Wnioski i Przyszłe Kierunki

### Główne Wnioski

1. **Efektywność modeli**:
   - Model SVM z jądrem RBF osiągnął najlepsze wyniki w predykcji śmiertelności wewnątrzszpitalnej pod względem dokładności (78.5%) i F1-Score (82.5%).
   - Model Regresji Logistycznej wykazał najlepszą zdolność do rozróżniania klas z najwyższym AUC-ROC (84.9%).
   - Modele oparte na drzewach decyzyjnych (Random Forest, Gradient Boosting) również osiągnęły dobre wyniki, szczególnie pod względem czułości (Recall).

2. **Istotność cech**:
   - Cechy związane z równowagą elektrolitową, wentylacją mechaniczną i funkcjami życiowymi mają największy wpływ na predykcję śmiertelności.
   - Cechy z sygnałów EKG dostarczają wartościowych informacji, które uzupełniają standardowe dane kliniczne, szczególnie parametry związane ze zmiennością rytmu serca (HRV) i morfologią załamków.

3. **Potencjał kliniczny**:
   - Opracowany system może wspierać decyzje kliniczne poprzez identyfikację pacjentów wysokiego ryzyka.
   - Wczesna identyfikacja zagrożeń może prowadzić do bardziej intensywnego monitorowania i leczenia.

### Przyszłe Kierunki

1. **Rozszerzenie zbioru danych**:
   - Włączenie większej liczby pacjentów do analizy
   - Dodanie danych z innych ośrodków w celu zwiększenia generalizacji

2. **Zaawansowane techniki modelowania**:
   - Zastosowanie głębokich sieci neuronowych do bezpośredniej analizy surowych sygnałów EKG
   - Eksploracja modeli sekwencyjnych (LSTM, GRU) do analizy czasowych aspektów danych

3. **Interpretacja kliniczna**:
   - Głębsza analiza zidentyfikowanych cech we współpracy z ekspertami klinicznymi
   - Opracowanie wytycznych klinicznych na podstawie wyników modeli

4. **Wdrożenie w praktyce**:
   - Stworzenie interfejsu użytkownika dla lekarzy
   - Integracja z systemami szpitalnymi
   - Przeprowadzenie badań prospektywnych w celu walidacji systemu w praktyce klinicznej

Projekt "ECGModel" stanowi solidną podstawę do dalszych badań nad wykorzystaniem uczenia maszynowego w predykcji wyników klinicznych na podstawie danych EKG i parametrów klinicznych.
