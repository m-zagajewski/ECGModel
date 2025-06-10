import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
import re
import concurrent.futures
import multiprocessing
import warnings
from functools import partial
import time
import platform  # Dodajemy import do wykrywania systemu operacyjnego

# Wykrywanie systemu operacyjnego
IS_MACOS = platform.system() == 'Darwin'

# Konfiguracja matplotlib dla macOS - ustaw backend nieinteraktywny dla bezpiecznego wielowątkowego działania
if IS_MACOS:
    mpl.use('Agg')  # Używamy nieinteraktywnego backendu na macOS
    print(f"Wykryto macOS. Używam nieinteraktywnego backendu matplotlib.")

# Wyłącz zbędne ostrzeżenia
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Spróbuj zaimportować SHAP - zaawansowaną bibliotekę do interpretacji modeli
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Biblioteka SHAP nie jest zainstalowana. Aby włączyć analizę SHAP, zainstaluj: pip install shap")

# Dodaj ścieżkę projektu do sys.path, jeśli to konieczne
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.model_implementations import (
    RandomForestModel, 
    XGBoostModel, 
    GradientBoostingModel,
    CatBoostModel, 
    SVMModel, 
    LogisticRegressionModel
)

# Globalna konfiguracja
CONFIG = {
    'n_jobs': max(1, multiprocessing.cpu_count() - 1),  # Domyślnie używa wszystkich dostępnych rdzeni - 1
    'random_state': 34,
    'permutation_repeats': 10,
    'top_n_features': 50,  # Ile najważniejszych cech pokazywać na wykresach
    'output_dir': os.path.join(project_root, 'data/feature_analysis'),
    'verbose': True
}

# Mapowanie nazw modeli na klasy modeli dla spójności
MODEL_CLASSES = {
    'random_forest': RandomForestModel,
    'gradient_boosting': GradientBoostingModel,
    'svm': SVMModel,
    'logistic_regression': LogisticRegressionModel, 
    'catboost': CatBoostModel,
    'xgboost': XGBoostModel
}

def load_model_params(model_name, analysis_name):
    """
    Wczytaj zoptymalizowane parametry dla modelu z pliku JSON
    lub zwróć domyślne parametry do analizy, jeśli plik nie istnieje.
    """
    # Ścieżka do pliku z parametrami
    params_file = os.path.join(project_root, f'model_optimization_results/{model_name}_best_params.json')
    
    # Domyślne parametry dla modeli używanych w analizie cech - prostsze niż dla treningu
    analysis_default_params = {
        'random_forest': {'random_state': CONFIG['random_state']},
        'gradient_boosting': {'random_state': CONFIG['random_state']},
        'svm': {'random_state': CONFIG['random_state'], 'probability': True},
        'logistic_regression': {'random_state': CONFIG['random_state'], 'max_iter': 1000},
        'catboost': {'random_seed': CONFIG['random_state'], 'verbose': 0},
        'xgboost': {'random_state': CONFIG['random_state']}
    }
    
    try:
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            # Upewnij się, że parametry zawierają random_state/random_seed dla reprodukowalności
            if 'random_state' not in params and model_name != 'catboost':
                params['random_state'] = CONFIG['random_state']
            elif model_name == 'catboost' and 'random_seed' not in params:
                params['random_seed'] = CONFIG['random_state']
            
            if model_name == 'catboost':  # Zawsze ustaw verbose=0 dla CatBoost w analizie
                params['verbose'] = 0
                
            print(f"Wczytano zoptymalizowane parametry dla {analysis_name} ({model_name}): {params}")
            return params
        else:
            print(f"Plik zoptymalizowanych parametrów {params_file} nie znaleziony dla {analysis_name} ({model_name}). Przechodzę do domyślnych/fabryki.")
            return analysis_default_params.get(model_name, {'random_state': CONFIG['random_state']})
    except Exception as e:
        print(f"Błąd ładowania/użycia zopt. parametrów dla {analysis_name} ({model_name}): {e}. Przechodzę do domyślnych/fabryki.")
        return analysis_default_params.get(model_name, {'random_state': CONFIG['random_state']})

# Funkcje pomocnicze dla serializacji w przetwarzaniu równoległym
def create_model_instance(model_name, model_class, analysis_name):
    """Tworzy instancję modelu z odpowiednimi parametrami."""
    try:
        params = load_model_params(model_name, analysis_name)
        instance = model_class(**params)
        print(f"Użyto domyślnych parametrów analizy dla {analysis_name} ({model_class.__name__}): {params}")
        return instance
    except Exception as e:
        print(f"Błąd tworzenia instancji dla {analysis_name} ({model_class.__name__}): {e}")
        print(f"Użyto fabryki do stworzenia instancji dla {analysis_name} ({model_class.__name__})")
        return model_class()

def create_random_forest_for_analysis():
    """Tworzy instancję RandomForest do analizy cech."""
    return create_model_instance('random_forest', RandomForestModel, "Random Forest Feature Importance")

def create_xgboost_for_analysis():
    """Tworzy instancję XGBoost do analizy cech."""
    return create_model_instance('xgboost', XGBoostModel, "XGBoost Feature Importance")

def create_gradient_boosting_for_analysis():
    """Tworzy instancję Gradient Boosting do analizy cech."""
    return create_model_instance('gradient_boosting', GradientBoostingModel, "Gradient Boosting Feature Importance")

def create_catboost_for_analysis():
    """Tworzy instancję CatBoost do analizy cech."""
    return create_model_instance('catboost', CatBoostModel, "CatBoost Feature Importance")

def create_svm_for_analysis():
    """Tworzy instancję SVM (RBF) do analizy cech."""
    return create_model_instance('svm', SVMModel, "SVM (Optimized Kernel) Feature Importance")

def create_linear_svm():
    """Tworzy instancję SVM z kernelem liniowym."""
    return SVMModel(kernel='linear', random_state=CONFIG['random_state'], probability=True)

def create_logistic_regression_for_analysis():
    """Tworzy instancję Logistic Regression do analizy cech."""
    return create_model_instance('logistic_regression', LogisticRegressionModel, "Logistic Regression Coefficients")

def statistical_worker_function(task):
    """
    Funkcja robocza do analizy statystycznej - przeniesiona na poziom modułu
    dla umożliwienia serializacji w procesach równoległych.
    """
    name, method, X, y, feature_names, title, filename = task
    try:
        selector = SelectKBest(method, k='all')
        selector.fit(X, y)
        scores = selector.scores_
        # Tworzenie podfolderu dla analiz statystycznych
        output_subdir = os.path.join(CONFIG['output_dir'], "StatisticalAnalysis")
        plot_feature_importance(scores, feature_names, title, CONFIG['top_n_features'], filename, subfolder="StatisticalAnalysis")
        return name, pd.Series(scores, index=feature_names).sort_values(ascending=False)
    except Exception as e:
        print(f"Błąd w analizie {name}: {str(e)}")
        return name, None

def load_data():
    """Wczytaj dane z pliku CSV, automatycznie wykrywając kolumnę indeksu"""
    data_path = os.path.join(project_root, 'data/processed_features.csv')
    # Najpierw wczytaj CSV bez określania indeksu, aby sprawdzić dostępne kolumny
    df = pd.read_csv(data_path)
    
    if CONFIG['verbose']:
        print(f"Wczytano dane z {data_path}")
        print(f"Wymiary danych: {df.shape}")
        print(f"Pierwsze kolumny: {', '.join(df.columns[:5])}")
    
    # Sprawdź czy pierwsza kolumna może być indeksem (zwykle nazywana 'Unnamed: 0' lub podobnie)
    if df.columns[0].startswith('Unnamed') or df.columns[0].lower() in ['id', 'case_id', 'kg']:
        if CONFIG['verbose']:
            print(f"Ustawiam kolumnę '{df.columns[0]}' jako indeks")
        df.set_index(df.columns[0], inplace=True)
    
    # Upewnij się, że kolumna 'ZGON' istnieje
    if 'ZGON' not in df.columns:
        raise ValueError("Kolumna 'ZGON' nie istnieje w danych")
        
    X = df.drop(columns=['ZGON'])
    y = df['ZGON']
    return X, y, df

def categorize_features(feature_names):
    """Kategoryzuj cechy w grupy (kliniczne, EKG) z zaawansowaną klasyfikacją"""
    categories = defaultdict(list)
    
    patterns = {
        "ECG_Basic": [r'^[IV]_mean', r'^aV[FLR]_mean', r'^(I|II|III|aVR|aVL|aVF|V\d)_std'],
        "ECG_Rhythm": [r'heart_rate', r'hrv', r'rmssd', r'sdnn'],
        "ECG_Waves": [r'_duration', r'_amplitude', r'qrs', r'qtc', r'p_', r't_'],
        "ECG_Segments": [r'st_', r'pr_'],
        "ECG_Quality": [r'signal_quality', r'snr', r'complexity', r'entropy'],
        "Demographics": [r'WIEK', r'PLEC', r'male sex', r'female'],
        "Vital_Signs": [r'GLASGOW', r'MAP', r'pao2', r'temp', r'Wentylacja'],
        "Lab_Results": [r'BILI', r'KREA', r'PARAMETR', r'poziom', r'HGB'],
        "Medications": [r'dopamina', r'noradrenalina', r'adrena', r'dawki'],
        "Comorbidity": [r'marskość', r'cukrzyca', r'nadciśnienie', r'niewydolność', r'POCHP', r'astma']
    }
    
    # Pierwszy przebieg - próba dopasowania wzorców
    for feature in feature_names:
        categorized = False
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, feature, re.IGNORECASE):
                    categories[category].append(feature)
                    categorized = True
                    break
            if categorized:
                break
        
        # Dodatkowe dopasowanie na podstawie prefixu dla odprowadzeń EKG
        if not categorized:
            ecg_lead_pattern = r'^(I|II|III|aVR|aVL|aVF|V\d)_'
            if re.match(ecg_lead_pattern, feature):
                categories["ECG_Other"].append(feature)
                categorized = True
        
        if not categorized:
            categories["Other"].append(feature)
    
    return categories

def create_colormap(n_colors):
    """Twórz przyjemną dla oka kolorową paletę dla wizualizacji"""
    return plt.cm.viridis(np.linspace(0, 0.85, n_colors))

def plot_feature_importance(importances, feature_names, title, top_n=20, filename=None, style='bar', subfolder=None):
    """Twórz atrakcyjne wykresy ważności cech z wieloma stylami wizualizacji"""
    # Sortuj według ważności
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names)[idx]
    top_importances = importances[idx]
    
    plt.figure(figsize=(14, 10))
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    if style == 'bar':
        colors = create_colormap(len(top_features))
        bars = plt.barh(range(len(top_features)), top_importances, color=colors, alpha=0.8, 
                       edgecolor='gray', linewidth=0.5)
        
        # Dodaj wartości na końcach słupków
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{top_importances[i]:.4f}', 
                    va='center', fontsize=9, alpha=0.8)
        
    elif style == 'lollipop':
        colors = create_colormap(len(top_features))
        plt.hlines(y=range(len(top_features)), xmin=0, xmax=top_importances, 
                  color='skyblue', alpha=0.7, linewidth=2)
        plt.scatter(top_importances, range(len(top_features)), s=80, 
                   color='navy', alpha=0.8, zorder=2)
    
    elif style == 'dot':
        # Gradient dots with increasing radius
        sizes = 50 + (top_importances / max(top_importances) * 150)
        colors = create_colormap(len(top_features))
        plt.scatter(top_importances, range(len(top_features)), s=sizes, color=colors, alpha=0.8)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        # Add importance values next to the dots
        for i, imp in enumerate(top_importances):
            plt.text(imp + 0.002, i, f'{imp:.4f}', va='center', fontsize=9)
    
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Zapisz wykres
    if filename:
        current_output_dir = CONFIG['output_dir']
        if subfolder:
            current_output_dir = os.path.join(CONFIG['output_dir'], subfolder)
        
        os.makedirs(current_output_dir, exist_ok=True)
        plt.savefig(os.path.join(current_output_dir, filename), dpi=300, bbox_inches='tight')
    
    # Na macOS nie wyświetlaj wykresów interaktywnie
    if IS_MACOS:
        plt.close()
    else:
        plt.close()  # Zamykamy wykres zamiast pokazywać - lepsze dla równoległego przetwarzania

def _generate_feature_importance_worker(model_info, X, y, feature_names):
    """
    Funkcja robocza do analizy ważności cech dla jednego modelu - optymalizowana do uruchomienia równoległego
    """
    model = None # Inicjalizacja na wypadek błędu przed przypisaniem
    model_name_for_path = "UnknownModel" # Domyślna nazwa folderu
    try:
        if isinstance(model_info, tuple):
            # Pobierz informacje o modelu z krotki
            if len(model_info) == 4:
                model_class, title, filename, style = model_info
                
                # Tworzenie instancji modelu
                if callable(model_class) and not isinstance(model_class, type):
                    # Jeśli to funkcja (nie klasa), wywołaj ją aby uzyskać instancję
                    model = model_class()
                else:
                    # Jeśli to klasa, utwórz jej instancję
                    model = model_class()
                model_name_for_path = model.__class__.__name__
            else:
                raise ValueError(f"Nieprawidłowy format danych modelu: {model_info}")
        else:
            # Jeśli model_info to już instancja modelu
            model = model_info
            title = model.__class__.__name__ + " Feature Importance"
            filename = model.__class__.__name__.lower() + "_importance.png"
            style = 'bar'
            model_name_for_path = model.__class__.__name__
        
        model_name = model.__class__.__name__
        
        # Trenowanie modelu - ulepszony preprocessing
        X_scaled = StandardScaler().fit_transform(X)
        model._fit_model(X_scaled, y)
        
        # Determine approach for getting feature importances
        if hasattr(model.model, "feature_importances_"):
            # Tree-based models
            importances = model.model.feature_importances_
            importance_method = "feature_importances_"
            
        elif hasattr(model.model, "coef_") and model.model.coef_.ndim <= 2:
            # Linear models
            coefs = model.model.coef_[0] if model.model.coef_.ndim == 2 else model.model.coef_
            importances = np.abs(coefs)
            importance_method = "coef_"
            
        elif hasattr(model.model, "get_feature_importance"):
            # CatBoost
            importances = model.model.get_feature_importance()
            importance_method = "get_feature_importance"
            
        else:
            # Permutation importance for other models - dodaj scoring
            perm_imp = permutation_importance(
                model.model, X_scaled, y, n_repeats=CONFIG['permutation_repeats'], 
                random_state=CONFIG['random_state'], n_jobs=1, scoring='f1'
            )
            importances = perm_imp.importances_mean
            importance_method = "permutation"
            
        # Visualize the importances 
        plot_feature_importance(
            importances, feature_names, title, 
            top_n=CONFIG['top_n_features'], filename=filename, style=style,
            subfolder=model_name_for_path # Przekazanie nazwy modelu jako subfolderu
        )
        
        # Generate SHAP plots if appropriate
        shap_values = None
        if SHAP_AVAILABLE:
            try:
                # Użyj już przeskalowanych danych dla SHAP (zamiast oryginalnych X)
                background_data = shap.sample(pd.DataFrame(X_scaled, columns=feature_names), 100)
                explainer = shap.KernelExplainer(model.predict_proba, background_data)
                shap_values = explainer.shap_values(background_data)

                plt.figure(figsize=(14, 10))
                shap.summary_plot(shap_values[1], background_data,
                                 feature_names=feature_names,
                                 plot_type="bar", show=False)
                plt.title(f"SHAP Values - {model_name}", fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Zapis SHAP do podfolderu modelu
                shap_output_dir = os.path.join(CONFIG['output_dir'], model_name_for_path)
                os.makedirs(shap_output_dir, exist_ok=True)
                plt.savefig(os.path.join(shap_output_dir, f"shap_{model_name}.png"),
                           dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"SHAP analysis failed for {model_name}: {str(e)}")
                shap_values = None

        # Return the results
        results = {
            'model_name': model_name,
            'importance_method': importance_method,
            'importance_values': pd.Series(importances, index=feature_names).sort_values(ascending=False),
        }

        if shap_values is not None:
            # Dla modeli binarnych, shap_values[1] to wartości dla klasy pozytywnej
            if isinstance(shap_values, list) and len(shap_values) > 1:
                pos_class_shap = shap_values[1]
            else:
                pos_class_shap = shap_values
                
            shap_importance = pd.Series(
                np.abs(pos_class_shap).mean(0),
                index=feature_names
            ).sort_values(ascending=False)
            results['shap_values'] = shap_importance

        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        model_name_for_error = model_name_for_path if model else getattr(model_info, '__name__', str(model_info))
        print(f"ERROR w {model_name_for_error}: {str(e)}")
        return {
            'model_name': model_name_for_error,
            'error': str(e),
        }

def analyze_models_parallel(models_to_analyze, X, y, feature_names):
    """
    Równoległa analiza ważności cech dla wielu modeli z wykorzystaniem wielu rdzeni CPU
    """
    all_results = []

    # Określ liczbę procesów
    n_jobs = min(CONFIG['n_jobs'], len(models_to_analyze))

    if CONFIG['verbose']:
        print(f"\nRozpoczynam równoległą analizę modeli używając {n_jobs} wątków...")
        start_time = time.time()

    # Utwórz częściową funkcję z stałymi argumentami
    worker_func = partial(_generate_feature_importance_worker, X=X, y=y, feature_names=feature_names)

    # Na macOS używamy ThreadPoolExecutor zamiast ProcessPoolExecutor aby uniknąć problemów z GUI
    executor_class = concurrent.futures.ThreadPoolExecutor if IS_MACOS else concurrent.futures.ProcessPoolExecutor
    
    # Wykonaj równoległą analizę
    with executor_class(max_workers=n_jobs) as executor:
        futures = {executor.submit(worker_func, model_info): model_info for model_info in models_to_analyze}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                if 'error' not in result:
                    model_name = result['model_name']
                    if CONFIG['verbose']:
                        print(f"✓ {i+1}/{len(models_to_analyze)} Zakończono analizę {model_name}")
                    all_results.append(result)
                else:
                    model_info = futures[future]
                    model_name = getattr(model_info[0], '__name__', str(model_info[0])) if isinstance(model_info, tuple) else str(model_info)
                    print(f"✗ {i+1}/{len(models_to_analyze)} Błąd analizy {model_name}: {result['error']}")
            except Exception as e:
                print(f"✗ Błąd w procesie równoległym: {str(e)}")

    if CONFIG['verbose']:
        end_time = time.time()
        print(f"Zakończono równoległą analizę w {end_time - start_time:.2f} sekund")

    return all_results

def analyze_statistical_parallel(X, y, feature_names, top_n=20):
    """Równoległa analiza statystyczna ważności cech"""
    results = {}
    output_subdir_name = "StatisticalAnalysis" # Nazwa podfolderu dla analiz statystycznych

    # Różne metody statystyczne do analizy w równoległych procesach
    methods = [
        ('ANOVA_F', f_classif, "ANOVA F-value Feature Importance", "anova_importance.png"),
        ('MutualInfo', mutual_info_classif, "Mutual Information Feature Importance", "mi_importance.png")
    ]
    
    tasks = []
    
    for name, method, title, filename in methods:
        # Utwórz zadanie
        tasks.append((name, method, X, y, feature_names, title, filename))
    
    # Wykonaj zadania równolegle
    if CONFIG['verbose']:
        print(f"\nRównolegle wykonuję analizę statystyczną (wyniki w: {output_subdir_name})...")
    
    # Na macOS używamy ThreadPoolExecutor zamiast ProcessPoolExecutor
    executor_class = concurrent.futures.ThreadPoolExecutor if IS_MACOS else concurrent.futures.ThreadPoolExecutor
    
    # Używamy mapowania zamiast submit/result_completed dla prostszego kodu
    with executor_class(max_workers=len(tasks)) as executor:
        for name, result in executor.map(statistical_worker_function, tasks):
            if result is not None:
                results[name] = result
                
    return results

def analyze_correlation(X, y, feature_names, corr_threshold=0.7):
    """Zaawansowana analiza korelacji między cechami"""
    # Przygotuj DataFrame z cechami i wynikiem
    df = pd.DataFrame(X, columns=feature_names)
    df['TARGET'] = y
    
    correlation_output_dir = os.path.join(CONFIG['output_dir'], "CorrelationAnalysis")
    os.makedirs(correlation_output_dir, exist_ok=True)
    
    # Korelacja z wynikiem
    corr_with_target = df.corr()['TARGET'].sort_values(ascending=False).drop('TARGET')
    
    # Zapisz korelację z wynikiem
    corr_with_target.to_csv(os.path.join(correlation_output_dir, 'correlation_with_target.csv'))
    
    # Top cechy korelowane z wynikiem (pozytywnie i negatywnie)
    plt.figure(figsize=(14, 12))
    top_corr = pd.concat([
        corr_with_target.head(15),  # Top 15 positive correlations
        corr_with_target.tail(15)  # Top 15 negative correlations
    ])
    
    # Twórz gradient kolorów na podstawie wartości korelacji
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(top_corr)))
    bars = plt.barh(top_corr.index, top_corr.values, color=colors)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.title('Top cechy najsilniej skorelowane z wynikiem', fontsize=14, fontweight='bold')
    plt.xlabel('Współczynnik korelacji')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Dodaj wartości korelacji
    for i, bar in enumerate(bars):
        xval = bar.get_width()
        plt.text(xval + 0.01 if xval >= 0 else xval - 0.07, 
                i, f'{xval:.3f}', 
                va='center', fontsize=9, 
                color='black' if xval >= 0 else 'white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(correlation_output_dir, 'top_corr_with_target.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Mapa korelacji dla top cech
    top_features = list(corr_with_target.abs().sort_values(ascending=False).head(20).index)
    top_features.append('TARGET')  # Dodaj wynik
    corr_matrix = df[top_features].corr()
    
    # Utwórz zaawansowaną mapę ciepła
    plt.figure(figsize=(16, 14))
    
    # Generowanie maski dla górnego trójkąta
    mask = np.triu(np.ones_like(corr_matrix))
    
    # Tworzenie ciepłej mapy korelacji z przyjemną kolorystyką
    cmap = plt.cm.RdBu_r
    
    # Tworzę własną mapę ciepła z matplotlib
    im = plt.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im)
    
    # Dodaję etykiety
    plt.xticks(np.arange(len(top_features)), labels=top_features, rotation=90)
    plt.yticks(np.arange(len(top_features)), labels=top_features)
    
    # Dodaję wartości tekstowe
    if len(top_features) <= 30:
        for i in range(len(top_features)):
            for j in range(len(top_features)):
                if i != j:  # Aby uniknąć zbędnego tekstu na przekątnej
                    plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                         ha="center", va="center", 
                         color="white" if abs(corr_matrix.iloc[i, j]) > 0.6 else "black",
                         fontsize=8)
    
    plt.title('Mapa korelacji dla najważniejszych cech', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(correlation_output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Wykryj wysoko skorelowane cechy (potencjalna wielowspółliniowość)
    print("\nWykrywanie wielowspółliniowości (cechy wysoko skorelowane):")
    corr_matrix = df.drop(columns=['TARGET']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(upper.index[i], upper.columns[j], upper.iloc[i,j])
                      for i in range(len(upper.index))
                      for j in range(len(upper.columns))
                      if upper.iloc[i,j] > corr_threshold]
    
    # Wyświetl i zapisz wysoko skorelowane pary
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
        high_corr_df = high_corr_df.sort_values('Correlation', ascending=False)
        high_corr_df.to_csv(os.path.join(correlation_output_dir, 'high_correlation_pairs.csv'), index=False)
        print(f"Znaleziono {len(high_corr_df)} par wysoko skorelowanych cech (>= {corr_threshold})")
        print(high_corr_df.head(10))
    else:
        print("Nie znaleziono wysoko skorelowanych cech.")
        
    # Analizuj stabilność korelacji z wykorzystaniem techniki bootstrap
    print("\nAnaliza stabilności korelacji poprzez bootstrap...")
    bootstrap_results = defaultdict(list)
    
    for _ in range(20):  # 20 próbek bootstrap
        # Losujemy indeksy z powtórzeniami
        n_samples = len(df)
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = df.iloc[sample_idx]
        
        # Obliczamy korelacje w próbce bootstrap
        bootstrap_corr = bootstrap_sample.corr()['TARGET'].drop('TARGET')
        
        # Zapisujemy wyniki dla każdej cechy
        for feature in feature_names:
            if feature in bootstrap_corr.index:
                bootstrap_results[feature].append(bootstrap_corr[feature])
    
    # Obliczamy średnią i odchylenie standardowe korelacji
    corr_stability = pd.DataFrame({
        'mean_corr': {f: np.mean(values) for f, values in bootstrap_results.items()},
        'std_corr': {f: np.std(values) for f, values in bootstrap_results.items()}
    })
    
    # Sortujemy według wartości bezwzględnej korelacji
    corr_stability = corr_stability.join(corr_with_target.rename('corr')).sort_values(by='mean_corr', key=abs, ascending=False)
    corr_stability['cv'] = corr_stability['std_corr'] / corr_stability['mean_corr'].abs()
    
    # Zapisujemy wyniki
    corr_stability.to_csv(os.path.join(correlation_output_dir, 'correlation_stability.csv'))
    
    # Wizualizujemy stabilność dla top 15 cech
    top_stability = corr_stability.head(15)
    
    plt.figure(figsize=(14, 10))
    
    # Zamiast używać tablicy kolorów, używamy pojedynczego koloru dla wszystkich punktów
    # i dodajemy kolorowy scatter plot
    plt.errorbar(
        top_stability['mean_corr'], range(len(top_stability)), 
        xerr=top_stability['std_corr'], fmt='none',  # Bez markerów
        ecolor='gray', capsize=5, alpha=0.7
    )
    
    # Dodajemy punkty z kolorowym gradientem
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_stability)))
    for i, (idx, row) in enumerate(top_stability.iterrows()):
        plt.scatter(
            row['mean_corr'], i,
            s=80, color=colors[i], zorder=2
        )
    
    plt.yticks(range(len(top_stability)), top_stability.index)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.title('Stabilność korelacji dla top 15 cech - analiza bootstrap', fontsize=14, fontweight='bold')
    plt.xlabel('Współczynnik korelacji z celem (+/- odch. std.)')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(correlation_output_dir, 'correlation_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_with_target, corr_stability

def perform_pca_analysis(X, feature_names, n_components=10):
    """Przeprowadź zaawansowaną analizę PCA z wizualizacją wkładu cech"""
    print("\nPrzeprowadzam analizę PCA...")
    
    pca_output_dir = os.path.join(CONFIG['output_dir'], "PCA_Analysis")
    os.makedirs(pca_output_dir, exist_ok=True)

    # Standaryzacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=min(n_components, X_scaled.shape[1], X_scaled.shape[0]))
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    
    # Wariancja wyjaśniona
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Wykres wariancji wyjaśnionej z ulepszonym stylem
    plt.figure(figsize=(12, 8))
    
    # Słupki dla pojedynczych PC
    bars = plt.bar(
        range(1, len(explained_variance_ratio)+1), 
        explained_variance_ratio, 
        alpha=0.7, 
        color=plt.cm.viridis(np.linspace(0, 0.8, len(explained_variance_ratio)))
    )
    
    # Linia skumulowanej wariancji
    plt.plot(
        range(1, len(explained_variance_ratio)+1), 
        cumulative_variance, 
        'r-', 
        marker='o', 
        linewidth=3, 
        markersize=8
    )
    
    # Dodaj linię 80% wariancji
    plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.7)
    
    # Dodaj etykiety dla liczby PC potrzebnych do wyjaśnienia 80% wariancji
    pc_80 = np.argmax(cumulative_variance >= 0.8) + 1  # +1 bo indeks od 0
    plt.annotate(
        f'PC = {pc_80} ({cumulative_variance[pc_80-1]*100:.1f}%)', 
        xy=(pc_80, 0.8), 
        xytext=(pc_80+1, 0.85),
        arrowprops=dict(arrowstyle='->', lw=1.5)
    )
    
    # Ustawienia osi i etykiet
    plt.xlabel('Główne składowe', fontsize=12)
    plt.ylabel('Proporcja wariancji wyjaśnionej', fontsize=12)
    plt.title('PCA: Wariancja wyjaśniona przez główne składowe', fontsize=14, fontweight='bold')
    plt.legend(['Skumulowana wariancja', 'Wariancja pojedynczej składowej', '80% wariancji'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Zapisz wykres
    plt.savefig(os.path.join(pca_output_dir, 'pca_explained_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analiza i wizualizacja dla każdego z najważniejszych komponentów
    component_df = pd.DataFrame()
    max_components = min(5, len(pca.components_))
    
    for i in range(max_components):
        component = pd.Series(pca.components_[i], index=feature_names)
        component_df[f'PC{i+1}'] = component
        sorted_component = component.abs().sort_values(ascending=False)
        
        # Wizualizacja najważniejszych cech w każdym komponencie
        plt.figure(figsize=(14, 8))
        top_features = sorted_component.head(15).index
        
        # Pobierz wartości dla top cech (uwzględniając znak!)
        values = component[top_features].values
        
        # Gradient kolorów bazujący na wartościach (nie na wartościach bezwzględnych)
        colors = plt.cm.RdBu_r(np.interp(values, (-1, 1), (0, 1)))
        
        bars = plt.barh(top_features, values, color=colors, alpha=0.8)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        plt.title(f'Najważniejsze cechy w komponencie PCA {i+1} ({explained_variance_ratio[i]*100:.2f}% wariancji)', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.xlabel('Wartość współczynnika (kierunek ma znaczenie)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(pca_output_dir, f'pca_component_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Zapisz współczynniki komponentów
    component_df.to_csv(os.path.join(pca_output_dir, 'pca_components.csv'))
    
    # Biplot dla pierwszych dwóch komponentów
    if X_pca.shape[1] >= 2:
        plt.figure(figsize=(14, 12))
        
        # Losowo wybierz 1000 próbek jeśli danych jest za dużo
        sample_size = min(1000, X_pca.shape[0])
        if X_pca.shape[0] > sample_size:
            indices = np.random.choice(X_pca.shape[0], sample_size, replace=False)
            X_pca_sample = X_pca[indices, :]
        else:
            X_pca_sample = X_pca
        
        # Narysuj punkty projekcji
        plt.scatter(
            X_pca_sample[:, 0], X_pca_sample[:, 1], 
            alpha=0.7, s=50, 
            c=plt.cm.viridis(np.random.rand(len(X_pca_sample)))
        )
        
        # Dodajemy strzałki dla najbardziej znaczących cech
        top_features_idx = np.abs(pca.components_[0]).argsort()[-10:]  # Top 10 cech w PC1
        top_features_idx = np.union1d(top_features_idx, np.abs(pca.components_[1]).argsort()[-10:])  # Dodaj top 10 z PC2
        
        for i in top_features_idx:
            plt.arrow(
                0, 0,  # Od początku układu
                pca.components_[0, i] * 5,  # Skalujemy dla lepszej widoczności
                pca.components_[1, i] * 5, 
                head_width=0.2, head_length=0.2, 
                fc='red', ec='red', 
                alpha=0.8
            )
            plt.text(
                pca.components_[0, i] * 5.2,  # Tekst tuż za strzałką
                pca.components_[1, i] * 5.2, 
                feature_names[i], 
                color='red', 
                fontsize=10, 
                ha='center', va='center'
            )
        
        plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)', fontsize=12)
        plt.title('Projekcja danych na pierwsze dwa komponenty PCA z wektorami cech', 
                 fontsize=14, fontweight='bold')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pca_output_dir, 'pca_biplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return pca, component_df

def analyze_feature_clusters(feature_importances, feature_names, n_clusters=5):
    """Grupowanie cech o podobnej ważności z ulepszoną wizualizacją"""
    if len(feature_importances) == 0:
        print("Brak danych o ważności cech do klastrowania")
        return
        
    clusters_output_dir = os.path.join(CONFIG['output_dir'], "FeatureClusters")
    os.makedirs(clusters_output_dir, exist_ok=True)
    
    # Przygotuj dane
    X_imp = feature_importances.values.reshape(-1, 1)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=CONFIG['random_state'])
    clusters = kmeans.fit_predict(X_imp)
    
    # Stwórz DataFrame z wynikami
    cluster_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances,
        'cluster': clusters
    }).sort_values('importance', ascending=False)
    
    # Wizualizacja klastrów
    plt.figure(figsize=(14, 9))
    
    # Utwórz kolorową paletę dla klastrów
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Dla każdego klastra:
    for i in range(n_clusters):
        subset = cluster_df[cluster_df['cluster'] == i]
        
        # Posortuj według ważności w ramach klastra
        subset = subset.sort_values('importance', ascending=False)
        
        # Generuj losowe przesunięcia x dla rozproszenia punktów
        x_jitter = np.random.normal(0, 0.1, size=len(subset))
        
        # Rysuj punkty z jitter dla lepszej widoczności
        plt.scatter(
            range(len(subset)) + x_jitter, 
            subset['importance'], 
            s=80, 
            color=cluster_colors[i], 
            alpha=0.7,
            label=f'Klaster {i+1} ({len(subset)} cech)'
        )
    
    plt.xlabel('Indeks cechy (posortowany wg ważności)', fontsize=12)
    plt.ylabel('Ważność cechy', fontsize=12)
    plt.title('Klastry cech o podobnej ważności', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Zapisz wykres
    plt.savefig(os.path.join(clusters_output_dir, 'feature_importance_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Drugi wykres: Srednia ważność klastrów z rozrzutem
    plt.figure(figsize=(12, 8))
    
    cluster_stats = cluster_df.groupby('cluster')['importance'].agg(['mean', 'min', 'max', 'count'])
    cluster_stats = cluster_stats.sort_values('mean', ascending=False)
    
    # Dodaj nazwy klastrów opisujące ich naturę
    cluster_names = [f"Klaster {i+1} ({count} cech)" for i, count in enumerate(cluster_stats['count'])]
    
    # Naprawiony kod dla wykresu errorbar - używamy pojedynczego koloru dla słupków błędu
    # i później dodajemy kolorowe markery
    plt.errorbar(
        cluster_stats['mean'],
        range(len(cluster_stats)),
        xerr=[cluster_stats['mean'] - cluster_stats['min'], cluster_stats['max'] - cluster_stats['mean']],
        fmt='none',
        capsize=5,
        ecolor='gray',
        alpha=0.7
    )
    
    # Dodajemy kolorowe punkty
    for i, (idx, row) in enumerate(cluster_stats.iterrows()):
        plt.scatter(
            row['mean'], i,
            s=100, 
            color=cluster_colors[idx],  # Używamy indeksu klastra jako indeksu koloru
            zorder=2  # Upewniamy się, że punkty są na wierzchu
        )
    
    plt.yticks(range(len(cluster_stats)), cluster_names)
    plt.xlabel('Średnia ważność (z zakresem min-max)', fontsize=12)
    plt.title('Charakterystyka klastrów cech', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(clusters_output_dir, 'feature_importance_cluster_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Zapisz wyniki klastrowania
    cluster_df.to_csv(os.path.join(clusters_output_dir, 'feature_importance_clusters.csv'), index=False)
    
    # Podsumowanie klastrów
    print("\nPodsumowanie klastrów cech:")
    for i, (_, row) in enumerate(cluster_stats.iterrows()):
        cluster_id = row.name
        subset = cluster_df[cluster_df['cluster'] == cluster_id]
        print(f"\nKlaster {i+1} - {len(subset)} cech, średnia ważność: {subset['importance'].mean():.4f}")
        print(f"Zakres ważności: {subset['importance'].min():.4f} - {subset['importance'].max():.4f}")
        print(f"Top 5 cech: {', '.join(subset.head(5)['feature'].tolist())}")
    
    return cluster_df

def create_feature_interactions(X, feature_names, top_k=20):
    """Tworzenie interakcji między cechami dla najważniejszych cech"""
    interactions = []
    
    # Wybierz top_k cech do interakcji
    selected_features = feature_names[:top_k]
    
    # Tworzenie interakcji cech
    for i in range(len(selected_features)):
        for j in range(i+1, len(selected_features)):
            feat1 = selected_features[i]
            feat2 = selected_features[j]
            
            # Tworzymy nowe cechy (mnożenie i stosunek)
            product_name = f"{feat1}_x_{feat2}"
            ratio1_name = f"{feat1}_div_{feat2}"
            ratio2_name = f"{feat2}_div_{feat1}"
            
            # Pobierz odpowiednie kolumny z X
            x1 = X.iloc[:, feature_names.index(feat1)]
            x2 = X.iloc[:, feature_names.index(feat2)]
            
            # Unikamy dzielenia przez zero
            nonzero_mask1 = x2 != 0
            nonzero_mask2 = x1 != 0
            
            # Tworzenie nowych cech
            product = x1 * x2
            ratio1 = np.zeros(len(x1))
            ratio2 = np.zeros(len(x1))
            
            ratio1[nonzero_mask1] = x1[nonzero_mask1] / x2[nonzero_mask1]
            ratio2[nonzero_mask2] = x2[nonzero_mask2] / x1[nonzero_mask2]
            
            interactions.append({
                'name': product_name,
                'feature': product,
                'type': 'product'
            })
            
            interactions.append({
                'name': ratio1_name,
                'feature': ratio1,
                'type': 'ratio'
            })
            
            interactions.append({
                'name': ratio2_name,
                'feature': ratio2,
                'type': 'ratio'
            })
    
    return interactions

def analyze_feature_interactions(X, y, feature_names, top_importance_features):
    """Zaawansowana analiza interakcji cech"""
    print("\nAnalizuję interakcje między najważniejszymi cechami...")
    
    interactions_output_dir = os.path.join(CONFIG['output_dir'], "FeatureInteractions")
    os.makedirs(interactions_output_dir, exist_ok=True)

    # Wybierz top 15 najważniejszych cech do analizy interakcji
    top_k = min(15, len(top_importance_features))
    top_features = list(top_importance_features.head(top_k).index)
    
    # Stwórz pary interakcji
    pairs = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            pairs.append((top_features[i], top_features[j]))
    
    interaction_results = []
    
    # Analizuj każdą parę
    for feat1, feat2 in pairs:
        # Pobierz indeksy cech
        idx1 = feature_names.index(feat1)
        idx2 = feature_names.index(feat2)
        
        # Pobierz wartości cech
        x1 = X[:, idx1]
        x2 = X[:, idx2]
        
        # Opcja 1: Korelacja cech z wynikiem
        corr1 = np.corrcoef(x1, y)[0, 1]
        corr2 = np.corrcoef(x2, y)[0, 1]
        
        # Opcja 2: Korelacja między cechami
        feat_corr = np.corrcoef(x1, x2)[0, 1]
        
        # Opcja 3: Interakcja mnożenia
        product = x1 * x2
        product_corr = np.corrcoef(product, y)[0, 1]
        
        # Opcja 4: Stosunek cech (unikając dzielenia przez zero)
        ratio1 = np.zeros_like(x1)
        nonzero = x2 != 0
        ratio1[nonzero] = x1[nonzero] / x2[nonzero]
        ratio1_corr = np.corrcoef(ratio1, y)[0, 1]
        
        interaction_results.append({
            'feature1': feat1,
            'feature2': feat2,
            'corr_feat1_target': corr1,
            'corr_feat2_target': corr2,
            'corr_between_feats': feat_corr,
            'product_corr_target': product_corr,
            'ratio_corr_target': ratio1_corr,
            'interaction_strength': abs(product_corr) - max(abs(corr1), abs(corr2))
        })
    
    # Konwertuj na DataFrame
    interactions_df = pd.DataFrame(interaction_results)
    
    # Posortuj według siły interakcji
    interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)
    
    # Zapisz wyniki
    interactions_df.to_csv(os.path.join(interactions_output_dir, 'feature_interactions.csv'), index=False)
    
    # Wizualizuj najsilniejsze interakcje
    top_interactions = interactions_df.head(10)
    
    plt.figure(figsize=(14, 10))
    
    # Tworzę etykiety par cech
    feature_pairs = [f"{row['feature1']}\n×\n{row['feature2']}" for _, row in top_interactions.iterrows()]
    
    # Tworzę słupki dla interakcji
    bars = plt.barh(
        feature_pairs, 
        top_interactions['interaction_strength'], 
        color=plt.cm.viridis(np.linspace(0, 0.8, len(top_interactions))), 
        alpha=0.8
    )
    
    # Dodaję wartości na końcach słupków
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.001, 
            bar.get_y() + bar.get_height()/2, 
            f"{top_interactions['interaction_strength'].iloc[i]:.3f}", 
            va='center', 
            fontsize=9
        )
    
    plt.xlabel('Siła interakcji (wzrost korelacji)', fontsize=12)
    plt.title('Top 10 najsilniejszych interakcji między cechami', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(interactions_output_dir, 'top_feature_interactions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Zidentyfikowano {len(interactions_df)} interakcji między cechami.")
    print(f"Top 5 najsilniejszych interakcji:")
    for i, (_, row) in enumerate(top_interactions.head(5).iterrows()):
        print(f"{i+1}. {row['feature1']} × {row['feature2']}: "
              f"siła interakcji = {row['interaction_strength']:.3f}, "
              f"korelacja iloczynu z celem = {row['product_corr_target']:.3f}")
    
    return interactions_df

def main():
    # Start mierzenia czasu
    start_time = time.time()
    
    # Upewnienie się, że folder analizy istnieje
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    print("\n====== Analiza najważniejszych cech ======")
    print(f"System operacyjny: {platform.system()}")
    print(f"Używam {CONFIG['n_jobs']} rdzeni CPU do analizy")
    print(f"Wyniki będą zapisane w {CONFIG['output_dir']}")
    
    # Wczytaj dane
    X, y, full_df = load_data()
    feature_names = X.columns.tolist()
    print(f"Wczytano {len(X)} próbek z {len(feature_names)} cechami.")
    
    # Kategoryzacja cech
    feature_categories = categorize_features(feature_names)
    print("\nKategorie cech:")
    for category, features in feature_categories.items():
        print(f"{category}: {len(features)} cech")
    
    # Zapisz kategorie cech
    categories_df = pd.DataFrame([(category, feature) 
                               for category, features in feature_categories.items() 
                               for feature in features], 
                              columns=['Category', 'Feature'])
    categories_df.to_csv(os.path.join(CONFIG['output_dir'], 'feature_categories.csv'), index=False)
    
    # Standaryzacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Równoległa analiza korelacji
    print("\n=== Analiza korelacji cech ===")
    corr_with_target, corr_stability = analyze_correlation(X_scaled, y, feature_names)
    
    # 2. Równoległa analiza PCA dla struktury danych
    print("\n=== Analiza PCA dla zrozumienia struktury danych ===")
    pca, pca_components = perform_pca_analysis(X_scaled, feature_names)
    
    # 3. Równoległa analiza ważności cech w modelach z funkcjami tworzącymi instancje
    print("\n=== Analiza ważności cech w modelach uczenia maszynowego ===")
    # Lista wszystkich modeli do analizy - używa nowych funkcji do tworzenia instancji
    models_to_analyze = [
        (create_random_forest_for_analysis, "Random Forest Feature Importance", "rf_importance.png", 'lollipop'),
        (create_xgboost_for_analysis, "XGBoost Feature Importance", "xgb_importance.png", 'bar'),
        (create_gradient_boosting_for_analysis, "Gradient Boosting Feature Importance", "gb_importance.png", 'bar'),
        (create_catboost_for_analysis, "CatBoost Feature Importance", "catboost_importance.png", 'bar'),
        (create_svm_for_analysis, "SVM (Optimized Kernel) Feature Importance", "svm_rbf_importance.png", 'bar'),
        (create_linear_svm, "SVM (Linear - Hardcoded Defaults) Feature Importance", "svm_linear_importance.png", 'bar'),
        (create_logistic_regression_for_analysis, "Logistic Regression Coefficients", "lr_coefficients.png", 'bar')
    ]

    # Uruchom równoległą analizę modeli - przekazując DataFrame z cechami
    model_results = analyze_models_parallel(models_to_analyze, X.values, y.values, feature_names)
    
    # Zbierz wyniki z równoległej analizy
    importance_results = {}
    for result in model_results:
        if 'importance_values' in result:
            importance_results[result['model_name']] = result['importance_values']
        if 'shap_values' in result:
            importance_results[f"{result['model_name']}_SHAP"] = result['shap_values']

    # 4. Równoległa analiza statystyczna
    print("\n=== Statystyczna analiza cech ===")
    statistical_results = analyze_statistical_parallel(X_scaled, y, feature_names)
    importance_results.update(statistical_results)

    # 5. Zbiorcze podsumowanie
    print("\n=== Zbiorcze podsumowanie ===")
    if importance_results:
        summary = pd.DataFrame(importance_results)
        
        # Dodaj średni ranking jeśli mamy wyniki z więcej niż jednego źródła
        if len(summary.columns) > 1:
            # Ranking dla każdej metody
            for col in summary.columns:
                summary[f'{col}_rank'] = summary[col].rank(ascending=False)
            
            # Średni ranking
            rank_cols = [col for col in summary.columns if col.endswith('_rank')]
            summary['mean_rank'] = summary[rank_cols].mean(axis=1)
            summary = summary.sort_values('mean_rank')
        
        # Zapisz pełne podsumowanie
        summary.to_csv(os.path.join(CONFIG['output_dir'], 'feature_importance_summary.csv'))
        print(f"Podsumowanie zapisane do feature_importance_summary.csv")
        
        # 6. Klastrowanie cech według ważności
        print("\n=== Klastrowanie cech według ważności ===")
        if 'mean_rank' in summary:
            cluster_results = analyze_feature_clusters(summary['mean_rank'], summary.index)
        
        # 7. Analiza interakcji cech
        print("\n=== Analiza interakcji cech ===")
        if len(summary) > 5:  # Potrzebujemy co najmniej kilku cech do analizy interakcji
            top_importance = summary['mean_rank'] if 'mean_rank' in summary else summary.iloc[:, 0]
            interaction_results = analyze_feature_interactions(X_scaled, y, feature_names, top_importance)
        
        # Wydrukuj najważniejsze cechy
        print("\nNajważniejsze cechy:")
        if 'mean_rank' in summary:
            top_features = summary.sort_values('mean_rank').head(20)
            selected_cols = ['mean_rank'] + [c for c in summary.columns if not c.endswith('_rank')][:3]
            print(top_features[selected_cols])
        else:
            # Jeśli nie ma mean_rank, weź pierwszy dostępny wynik
            first_col = summary.columns[0]
            top_features = summary.sort_values(first_col, ascending=False).head(20)
            print(top_features)
    else:
        print("Nie udało się uzyskać wyników ważności cech z żadnego modelu.")
    
    # Podsumowanie czasu wykonania
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nAnaliza zakończona w {minutes} min {seconds} s")

if __name__ == "__main__":
    main()