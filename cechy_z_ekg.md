Kod przetwarza standardowe, 12-odprowadzeniowe EKG. Oznacza to, że dla każdego pacjenta analizuje sygnały z 12 różnych punktów "widzenia" na serce (odprowadzenia: I, II, III, aVR, aVL, aVF, V1-V6). **Wszystkie poniższe cechy są wyciągane dla każdego z 12 odprowadzeń osobno**, co daje bardzo bogaty zestaw danych (np. `V1_heart_rate`, `II_heart_rate` itd.).

---

### Kategoria 1: Jakość i ogólne statystyki sygnału

Te cechy opisują ogólną charakterystykę i jakość sygnału EKG, co jest kluczowe do oceny, czy dane nadają się do dalszej analizy.

*   `signal_quality`: **Jakość sygnału.** Prosta metryka (0 lub 1) oceniająca, czy sygnał nie jest zbyt zaszumiony, płaski lub z artefaktami, które uniemożliwiają wiarygodną analizę.
*   `mean`: **Średnia wartość sygnału.** Średnia arytmetyczna wszystkich próbek sygnału po jego oczyszczeniu. Może wskazywać na przesunięcie linii izoelektrycznej.
*   `std`: **Odchylenie standardowe.** Mierzy, jak bardzo wartości sygnału odchylają się od średniej. Wysoka wartość oznacza dużą zmienność (amplitudę) sygnału.
*   `skewness`: **Skośność.** Mierzy asymetrię rozkładu wartości sygnału. Wartość 0 oznacza idealną symetrię.
*   `kurtosis`: **Kurtoza.** Mierzy, jak "szpiczaste" lub "spłaszczone" są piki w sygnale w porównaniu do rozkładu normalnego.
*   `rms` (Root Mean Square): **Wartość skuteczna.** Podobna do odchylenia standardowego, jest miarą "mocy" lub "energii" sygnału.
*   `zero_crossings`: **Liczba przejść przez zero.** Liczba, ile razy sygnał EKG przeciął linię zerową. Może być pośrednią miarą dominującej częstotliwości w sygnale.
*   `snr_db` (Signal-to-Noise Ratio): **Stosunek sygnału do szumu.** Mierzy, jak silny jest właściwy sygnał EKG w porównaniu do tła szumów. Im wyższa wartość, tym czystszy sygnał.

---

### Kategoria 2: Rytm serca i jego zmienność (HRV - Heart Rate Variability)

Te cechy koncentrują się na rytmie serca i jego subtelnych fluktuacjach, które są ważnym wskaźnikiem stanu autonomicznego układu nerwowego i ogólnego zdrowia.

*   `heart_rate`: **Tętno.** Średnia częstotliwość bicia serca na minutę, obliczona na podstawie odstępów między kolejnymi załamkami R (najwyższymi pikami w EKG).
*   `hrv_rmssd` (Root Mean Square of Successive Differences): **Pierwiastek ze średniej kwadratów różnic między kolejnymi odstępami RR.** Jest to kluczowy wskaźnik HRV, odzwierciedlający krótkoterminową zmienność rytmu. Wysoka wartość jest generalnie oznaką dobrej adaptacji serca i dominacji układu przywspółczulnego (odpowiedzialnego za odpoczynek).
*   `hrv_sdnn` (Standard Deviation of NN intervals): **Odchylenie standardowe wszystkich odstępów RR.** Mierzy ogólną zmienność rytmu serca w całym zapisie.
*   `lf_hf_ratio`: **Stosunek mocy niskich częstotliwości (LF) do wysokich (HF).** Cechy z analizy częstotliwościowej HRV. Często interpretowany jako wskaźnik równowagi między układem współczulnym (stres, walka-ucieczka) a przywspółczulnym (odpoczynek).

---

### Kategoria 3: Morfologia fal EKG (analiza kształtu)

Te cechy opisują kształt, czas trwania i wysokość poszczególnych fragmentów cyklu pracy serca, co ma ogromne znaczenie diagnostyczne.

#### Załamek P (Depolaryzacja przedsionków)
*   `p_duration`: **Czas trwania załamka P.** Czas potrzebny na rozprzestrzenienie się impulsu elektrycznego w przedsionkach serca, co inicjuje ich skurcz.
*   `p_amplitude`: **Amplituda (wysokość) załamka P.** Odzwierciedla siłę sygnału elektrycznego generowanego przez przedsionki. Nieprawidłowe wartości mogą wskazywać na ich powiększenie.
*   `p_amplitude_var`: **Zmienność amplitudy załamka P.** Mierzy, czy wysokość załamków P jest stała w całym zapisie.

#### Zespół QRS (Depolaryzacja komór)
*   `qrs_duration`: **Czas trwania zespołu QRS.** Czas potrzebny na rozprzestrzenienie się impulsu elektrycznego w komorach serca. Poszerzony QRS jest kluczową cechą zaburzeń przewodzenia śródkomorowego (np. blok odnogi pęczka Hisa).
*   `qrs_duration_var`: **Zmienność czasu trwania QRS.**
*   `r_amplitude`: **Amplituda załamka R.** Wysokość głównego, dodatniego załamka w zespole QRS. Może korelować z masą mięśnia lewej komory serca.

#### Odcinek ST i załamek T (Repolaryzacja komór)
*   `st_level`: **Poziom odcinka ST.** Średnia wartość sygnału w odcinku ST (fragment między końcem zespołu QRS a początkiem załamka T). Jest to **kluczowy wskaźnik niedokrwienia mięśnia sercowego**. Jego uniesienie lub obniżenie może świadczyć o zawale serca.
*   `st_slope`: **Nachylenie odcinka ST.** Kształt (nachylenie) tego odcinka również dostarcza ważnych informacji diagnostycznych, zwłaszcza w kontekście choroby wieńcowej.
*   `t_amplitude`: **Amplituda załamka T.** Załamek T odzwierciedla "elektryczny odpoczynek" komór (repolaryzację). Nieprawidłowości w jego wysokości lub kształcie mogą wskazywać na niedokrwienie, zaburzenia elektrolitowe lub inne problemy.

#### Odstęp QT (Całkowity czas aktywności elektrycznej komór)
*   `qtc_mean`: **Średni skorygowany odstęp QT.** Odstęp QT to czas od początku zespołu QRS do końca załamka T. Ponieważ zależy on od tętna, koryguje się go (stąd "c" w QTc). **Wydłużony odstęp QTc jest poważnym czynnikiem ryzyka groźnych dla życia arytmii komorowych.**
*   `qtc_sd`: **Odchylenie standardowe odstępu QTc.** Mierzy zmienność tego krytycznego interwału w czasie.

---

### Kategoria 4: Cechy zaawansowane i złożoność

Są to bardziej subtelne metryki, często wykorzystywane w badaniach naukowych i zaawansowanej diagnostyce.

*   `twa` (T-wave alternans): **Alternans załamka T.** Mierzy subtelne, naprzemienne zmiany w wysokości załamka T w kolejnych uderzeniach serca. Jest to silny predyktor ryzyka nagłego zgonu sercowego.
*   `shannon_entropy`: **Entropia Shannona.** Miara złożoności i nieprzewidywalności sygnału EKG. Zdrowe systemy biologiczne często charakteryzują się większą złożonością.
*   `signal_complexity`: **Złożoność sygnału.** Inna miara nieregularności sygnału. Podobnie jak entropia, może odzwierciedlać zdolność adaptacyjną serca.

