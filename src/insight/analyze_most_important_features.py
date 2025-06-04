import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Usuwam import seaborn, który powoduje błąd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
# Poprawiony import dla permutation_importance z właściwego modułu
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
import re

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

def load_data():
    data_path = os.path.join(project_root, 'data/processed_features.csv')
    # Najpierw wczytaj CSV bez określania indeksu, aby sprawdzić dostępne kolumny
    df = pd.read_csv(data_path)
    print(f"Dostępne kolumny w pliku: {df.columns[0:5]}...")
    
    # Sprawdź czy pierwsza kolumna może być indeksem (zwykle nazywana 'Unnamed: 0' lub podobnie)
    if df.columns[0].startswith('Unnamed') or df.columns[0].lower() in ['id', 'case_id', 'kg']:
        print(f"Ustawiam kolumnę '{df.columns[0]}' jako indeks")
        df.set_index(df.columns[0], inplace=True)
    
    # Upewnij się, że kolumna 'ZGON' istnieje
    if 'ZGON' not in df.columns:
        raise ValueError("Kolumna 'ZGON' nie istnieje w danych")
        
    X = df.drop(columns=['ZGON'])
    y = df['ZGON']
    return X, y, df

def categorize_features(feature_names):
    """Kategoryzuj cechy w grupy (kliniczne, EKG)"""
    categories = defaultdict(list)
    
    patterns = {
        "ECG": [r'^[IV]_', r'^aV[FLR]_', r'^(I|II|III|aVR|aVL|aVF|V\d)_'],
        "Clinical": [r'WIEK', r'PLEC', r'GLASGOW', r'MAP', r'pao2', r'BILI', r'Wentylacja', r'dopamina', r'noradrenalina'],
        "Demographics": [r'male sex', r'WIEK'],
        "Comorbidity": [r'marskość', r'cukrzyca', r'nadciśnienie', r'niewydolność', r'POCHP', r'astma']
    }
    
    for feature in feature_names:
        categorized = False
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, feature):
                    categories[category].append(feature)
                    categorized = True
                    break
            if categorized:
                break
        
        if not categorized:
            categories["Other"].append(feature)
    
    return categories

def plot_feature_importance(importances, feature_names, title, top_n=20, filename=None, style='bar'):
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names)[idx]
    top_importances = importances[idx]
    
    plt.figure(figsize=(12, 8))
    
    if style == 'bar':
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_importances, color=colors)
        plt.yticks(range(len(top_features)), top_features)
        
        # Dodaj wartości na końcach słupków
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{top_importances[i]:.4f}', 
                    va='center', fontsize=8)
        
    elif style == 'lollipop':
        plt.hlines(y=range(len(top_features)), xmin=0, xmax=top_importances, color='skyblue')
        plt.plot(top_importances, range(len(top_features)), "o", color='navy')
        plt.yticks(range(len(top_features)), top_features)
    
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    
    # Zapisz wykres
    if filename:
        output_dir = os.path.join(project_root, 'data/feature_analysis')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_with_model(model_class, X, y, feature_names, title, top_n=20, filename=None):
    print(f"Analizowanie z modelem {model_class.__name__}...")
    model = model_class()
    model.fit(X, y)
    
    # Wynik standardowej ważności cech
    importance_result = None
    
    # Sprawdź, czy model ma atrybut feature_importances_
    if hasattr(model.model, "feature_importances_"):
        importances = model.model.feature_importances_
        plot_feature_importance(importances, feature_names, title, top_n, filename)
        importance_result = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    # Dla modeli LogisticRegression, używaj współczynników jako wagi cech
    elif hasattr(model.model, "coef_") and model.model.coef_.ndim <= 2:
        coefs = model.model.coef_[0] if model.model.coef_.ndim == 2 else model.model.coef_
        importances = np.abs(coefs)  # Bierzemy wartość bezwzględną
        plot_feature_importance(importances, feature_names, title, top_n, filename)
        importance_result = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    # Dla modeli CatBoost używaj feature_importances_
    elif hasattr(model.model, "get_feature_importance"):
        try:
            importances = model.model.get_feature_importance()
            plot_feature_importance(importances, feature_names, title, top_n, filename)
            importance_result = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        except Exception as e:
            print(f"Nie udało się uzyskać ważności cech dla {model_class.__name__}: {str(e)}")
    
    # Jeśli standardowe metody zawodzą, spróbuj permutacji
    if importance_result is None:
        print(f"Model {model_class.__name__} nie udostępnia standardowej informacji o ważności cech.")
        print("Próbuję analizę permutacyjną...")
        
        # Analiza permutacyjna ważności cech
        try:
            X_scaled = X if isinstance(X, np.ndarray) else X.values
            perm_imp = permutation_importance(model.model, X_scaled, y, n_repeats=10, 
                                          scoring='roc_auc', n_jobs=-1, random_state=42)
            
            importances = perm_imp.importances_mean
            plot_feature_importance(importances, feature_names, 
                                f"{title} (Permutation)", top_n, f"perm_{filename}")
            importance_result = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        except Exception as e:
            print(f"Analiza permutacyjna nie powiodła się: {str(e)}")
            return None
    
    # Analiza SHAP dla wybranego modelu (opcjonalnie)
    if SHAP_AVAILABLE and any(x in model_class.__name__.lower() for x in ["forest", "xgboost", "gradient", "catboost"]):
        try:
            print(f"Wykonuję analizę SHAP dla modelu {model_class.__name__}...")
            
            # Przygotuj objaśniacz SHAP
            X_array = X if isinstance(X, np.ndarray) else X.values
            explainer = shap.Explainer(model.model, X_array)
            shap_values = explainer(X_array)
            
            # Zapisz podsumowanie SHAP
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_array, feature_names=feature_names, show=False)
            plt.tight_layout()
            
            # Zapisz wykres
            output_dir = os.path.join(project_root, 'data/feature_analysis')
            plt.savefig(os.path.join(output_dir, f"shap_{model_class.__name__}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.show()
            
            # Zapisz również wartości SHAP
            shap_importance = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': np.abs(shap_values.values).mean(0)
            }).sort_values('shap_importance', ascending=False)
            
            shap_importance.to_csv(os.path.join(output_dir, f"shap_values_{model_class.__name__}.csv"), index=False)
            
        except Exception as e:
            print(f"Analiza SHAP nie powiodła się: {str(e)}")
    
    return importance_result

def analyze_statistical(X, y, feature_names, top_n=20):
    """Przeprowadź statystyczną analizę ważności cech"""
    results = {}
    
    # 1. ANOVA F-test
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    f_scores = selector.scores_
    plot_feature_importance(f_scores, feature_names, "ANOVA F-value Feature Importance", top_n, "anova_importance.png")
    results['ANOVA_F'] = pd.Series(f_scores, index=feature_names).sort_values(ascending=False)
    
    # 2. Mutual Information
    mi = mutual_info_classif(X, y)
    plot_feature_importance(mi, feature_names, "Mutual Information Feature Importance", top_n, "mi_importance.png")
    results['MutualInfo'] = pd.Series(mi, index=feature_names).sort_values(ascending=False)
    
    return results

def analyze_correlation(X, y, feature_names, corr_threshold=0.7):
    """Analizuj korelacje między cechami"""
    # Przygotuj DataFrame z cechami i wynikiem
    df = pd.DataFrame(X, columns=feature_names)
    df['ZGON'] = y
    
    # Korelacja z wynikiem
    corr_with_target = df.corr()['ZGON'].sort_values(ascending=False).drop('ZGON')
    
    # Zapisz korelację z wynikiem
    output_dir = os.path.join(project_root, 'data/feature_analysis')
    corr_with_target.to_csv(os.path.join(output_dir, 'correlation_with_target.csv'))
    
    # Top 20 korelacji z wynikiem (pozytywnych i negatywnych)
    plt.figure(figsize=(12, 10))
    top_corr = pd.concat([
        corr_with_target.head(10),
        corr_with_target.tail(10)
    ])
    
    colors = ['red' if c > 0 else 'blue' for c in top_corr]
    top_corr.plot(kind='barh', color=colors)
    plt.title('Top cechy najsilniej skorelowane z wynikiem')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_corr_with_target.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Mapa korelacji dla top cech - używam matplotlib zamiast seaborn
    top_features = list(corr_with_target.abs().sort_values(ascending=False).head(20).index)
    top_features.append('ZGON')  # Dodaj wynik
    
    plt.figure(figsize=(14, 12))
    correlation_matrix = df[top_features].corr()
    
    # Tworzę własną mapę ciepła za pomocą matplotlib
    im = plt.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    # Dodaję etykiety
    plt.xticks(np.arange(len(top_features)), labels=top_features, rotation=90)
    plt.yticks(np.arange(len(top_features)), labels=top_features)
    
    # Dodaję wartości tekstowe, jeśli nie ma za dużo cech
    if len(top_features) <= 30:
        for i in range(len(top_features)):
            for j in range(len(top_features)):
                text = plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.title('Mapa korelacji dla najważniejszych cech')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Wykryj wysoko skorelowane cechy (potencjalna wielowspółliniowość)
    print("\nWykrywanie wielowspółliniowości (cechy wysoko skorelowane):")
    corr_matrix = df.drop(columns=['ZGON']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(upper.index[i], upper.columns[j], upper.iloc[i,j])
                      for i in range(len(upper.index))
                      for j in range(len(upper.columns))
                      if upper.iloc[i,j] > corr_threshold]
    
    # Wyświetl i zapisz wysoko skorelowane pary
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
        high_corr_df = high_corr_df.sort_values('Correlation', ascending=False)
        high_corr_df.to_csv(os.path.join(output_dir, 'high_correlation_pairs.csv'), index=False)
        print(high_corr_df.head(20))
    else:
        print("Nie znaleziono wysoko skorelowanych cech.")
        
    return corr_with_target

def perform_pca_analysis(X, feature_names, n_components=10):
    """Przeprowadź analizę PCA aby zrozumieć strukturę danych"""
    print("\nPrzeprowadzam analizę PCA...")
    
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
    
    # Wykres wariancji wyjaśnionej
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, alpha=0.6, color='skyblue')
    plt.plot(range(1, len(explained_variance_ratio)+1), cumulative_variance, 'r-', marker='o')
    plt.axhline(y=0.8, color='g', linestyle='--')
    plt.xlabel('Główne składowe')
    plt.ylabel('Proporcja wariancji wyjaśnionej')
    plt.title('PCA: Wariancja wyjaśniona przez główne składowe')
    plt.legend(['Skumulowana wariancja', 'Wariancja pojedynczej składowej', '80% wariancji'])
    plt.tight_layout()
    
    # Zapisz wykres
    output_dir = os.path.join(project_root, 'data/feature_analysis')
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analiza i wizualizacja komponentów
    component_df = pd.DataFrame()
    for i in range(min(5, len(pca.components_))):
        component = pd.Series(pca.components_[i], index=feature_names)
        sorted_component = component.abs().sort_values(ascending=False)
        component_df[f'PCA{i+1}'] = component
        
        # Wizualizacja najważniejszych cech w każdym komponencie
        plt.figure(figsize=(12, 6))
        top_features = sorted_component.head(15).index
        plt.bar(top_features, component[top_features], color='skyblue')
        plt.xticks(rotation=90)
        plt.title(f'Najważniejsze cechy w komponencie PCA {i+1} ({explained_variance_ratio[i]*100:.2f}% wariancji)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pca_component_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Zapisz współczynniki komponentów
    component_df.to_csv(os.path.join(output_dir, 'pca_components.csv'))
    
    # Wizualizacja dwóch pierwszych komponentów
    if X_pca.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)')
        plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)')
        plt.title('Projekcja danych na pierwsze dwa komponenty PCA')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_scatter.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    return pca

def analyze_feature_clusters(feature_importances, feature_names, n_clusters=5):
    """Grupowanie cech o podobnej ważności"""
    if len(feature_importances) == 0:
        print("Brak danych o ważności cech do klastrowania")
        return
    
    # Przygotuj dane
    X_imp = feature_importances.values.reshape(-1, 1)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_imp)
    
    # Stwórz DataFrame z wynikami
    cluster_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances,
        'cluster': clusters
    }).sort_values('importance', ascending=False)
    
    # Wizualizacja klastrów
    plt.figure(figsize=(12, 8))
    for i in range(n_clusters):
        subset = cluster_df[cluster_df['cluster'] == i]
        plt.scatter(range(len(subset)), subset['importance'], label=f'Klaster {i+1}')
    
    plt.xlabel('Indeks cechy (posortowany wg ważności)')
    plt.ylabel('Ważność cechy')
    plt.title('Klastry cech o podobnej ważności')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Zapisz wykres
    output_dir = os.path.join(project_root, 'data/feature_analysis')
    plt.savefig(os.path.join(output_dir, 'feature_importance_clusters.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Zapisz wyniki klastrowania
    cluster_df.to_csv(os.path.join(output_dir, 'feature_importance_clusters.csv'), index=False)
    
    # Podsumowanie klastrów
    print("\nPodsumowanie klastrów cech:")
    for i in range(n_clusters):
        subset = cluster_df[cluster_df['cluster'] == i]
        print(f"\nKlaster {i+1} - {len(subset)} cech, średnia ważność: {subset['importance'].mean():.4f}")
        print(f"Top 5 cech: {', '.join(subset.head(5)['feature'].tolist())}")
    
    return cluster_df

def main():
    # Upewnienie się, że folder analizy istnieje
    feature_analysis_dir = os.path.join(project_root, 'data/feature_analysis')
    os.makedirs(feature_analysis_dir, exist_ok=True)
    
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
    categories_df.to_csv(os.path.join(feature_analysis_dir, 'feature_categories.csv'), index=False)
    
    # Standaryzacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Analiza korelacji
    print("\n=== Analiza korelacji cech ===")
    corr_with_target = analyze_correlation(X, y, feature_names)
    
    # 2. Analiza PCA dla struktury danych
    print("\n=== Analiza PCA dla zrozumienia struktury danych ===")
    pca = perform_pca_analysis(X, feature_names)
    
    # 3. Analiza ważności cech w modelach
    print("\n=== Analiza ważności cech w modelach uczenia maszynowego ===")
    # Lista wszystkich modeli do analizy
    models_to_analyze = [
        (RandomForestModel, "Random Forest Feature Importance", "rf_importance.png", 'lollipop'),
        (XGBoostModel, "XGBoost Feature Importance", "xgb_importance.png", 'bar'),
        (GradientBoostingModel, "Gradient Boosting Feature Importance", "gb_importance.png", 'bar'),
        (CatBoostModel, "CatBoost Feature Importance", "catboost_importance.png", 'bar'),
        (LogisticRegressionModel, "Logistic Regression Coefficients", "lr_coefficients.png", 'bar')
    ]

    # Słownik na wyniki ważności cech
    importance_results = {}
    
    # Analizuj każdy model
    for model_class, title, filename, style in models_to_analyze:
        print(f"\nAnaliza ważności cech: {title}")
        importance = analyze_with_model(model_class, X_scaled, y, feature_names, title, filename=filename)
        if importance is not None:
            importance_results[model_class.__name__] = importance

    # 4. Analiza statystyczna
    print("\n=== Statystyczna analiza cech ===")
    statistical_results = analyze_statistical(X_scaled, y, feature_names)
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
        summary.to_csv(os.path.join(feature_analysis_dir, 'feature_importance_summary.csv'))
        print(f"Podsumowanie zapisane do data/feature_analysis/feature_importance_summary.csv")
        
        # 6. Klastrowanie cech według ważności
        print("\n=== Klastrowanie cech według ważności ===")
        if 'mean_rank' in summary:
            cluster_results = analyze_feature_clusters(summary['mean_rank'], summary.index)
        
        # Wydrukuj najważniejsze cechy
        print("\nNajważniejsze cechy:")
        if 'mean_rank' in summary:
            top_features = summary.sort_values('mean_rank').head(20)
            print(top_features[['mean_rank'] + [c for c in summary.columns if not c.endswith('_rank')]])
        else:
            # Jeśli nie ma mean_rank, weź pierwszy dostępny wynik
            first_col = summary.columns[0]
            top_features = summary.sort_values(first_col, ascending=False).head(20)
            print(top_features)
    else:
        print("Nie udało się uzyskać wyników ważności cech z żadnego modelu.")

if __name__ == "__main__":
    main()