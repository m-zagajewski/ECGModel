# ECGModel - Przetwarzanie danych klinicznych i sygnałów EKG

Aby przetworzyć wszystkie dane i wygenerować gotowy plik CSV, uruchom skrypt `process_data.py`:

Wynikowy plik z kompletem cech to:
`data/processed_features.csv`

## Opis cech

- **Cechy z sygnałów EKG (time series):**
  Wyznaczane są zaawansowane cechy kliniczne z każdego odprowadzenia, m.in.:
  - statystyki sygnału (średnia, odchylenie, skośność, kurtoza, RMS)
  - parametry rytmu serca (częstość, HRV, LF/HF)
  - cechy załamków (P, QRS, T): amplitudy, czasy trwania, zmienność
  - poziom ST, alternans T, entropia, SNR i inne wskaźniki jakości sygnału

- **Cechy z Excela (dane kliniczne):**
  Wykorzystywane są wybrane kolumny kliniczne, m.in.:
  - wiek, płeć, skala Glasgow
  - obecność chorób przewlekłych
  - operacje przed przyjęciem
  - wynik śmiertelności szpitalnej (`ZGON`)

Szczegóły wyboru cech znajdują się w plikach `process_excel_data.py` oraz `process_time_series.py`.