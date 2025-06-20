import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score, f1_score,
    precision_score, recall_score
)
import itertools

# Dodaj ścieżkę projektu do sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.model_implementations import (
    RandomForestModel,
    GradientBoostingModel,
    SVMModel,
    LogisticRegressionModel,
    CatBoostModel,
    XGBoostModel
)
from src.models.base_model import ECGBaseModel

# Globalna konfiguracja
CONFIG = {
    'output_dir': os.path.join(project_root, 'data/model_analysis'),
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'verbose': True
}

def load_selected_data():
    """Wczytaj dane z pliku selected_features.csv."""
    data_path = os.path.join(project_root, 'data/selected_features.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Plik {data_path} nie został znaleziony. "
            "Uruchom najpierw skrypt select_the_most_important_features.py."
        )
    
    df = pd.read_csv(data_path)
    
    # Sprawdź, czy pierwsza kolumna to indeks i ustaw ją
    if df.columns[0].startswith('Unnamed') or df.columns[0].lower() in ['id', 'case_id', 'kg']:
        df.set_index(df.columns[0], inplace=True)

    if 'ZGON' not in df.columns:
        raise ValueError("Kolumna 'ZGON' nie istnieje w danych.")
        
    X = df.drop(columns=['ZGON'])
    y = df['ZGON']
    
    if CONFIG['verbose']:
        print(f"Wczytano dane z {data_path}. Kształt X: {X.shape}, Kształt y: {y.shape}")
    return X, y

def plot_learning_curve_custom(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy', output_path=None):
    """
    Generuje i zapisuje krzywą uczenia.
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Próbki treningowe")
    axes.set_ylabel(f"Wynik ({scoring})")

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, scoring=scoring, shuffle=True, random_state=CONFIG['random_state'])
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes.grid(True, linestyle='--', alpha=0.7)
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Wynik treningowy")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Wynik walidacji krzyżowej")
    axes.legend(loc="best")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if CONFIG['verbose']:
            print(f"Zapisano krzywą uczenia do: {output_path}")
    plt.close()


def plot_confusion_matrix_custom(y_true, y_pred, classes, model_name,
                          normalize=False, title='Macierz konfuzji',
                          cmap=plt.cm.Blues, output_path=None):
    """
    Drukuje i rysuje macierz konfuzji.
    Normalizację można zastosować ustawiając `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f"{title} - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if CONFIG['verbose']:
            print("Znormalizowana macierz konfuzji")
    else:
        if CONFIG['verbose']:
            print('Macierz konfuzji, bez normalizacji')
    
    if CONFIG['verbose']:
        print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:,}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidziana etykieta')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if CONFIG['verbose']:
            print(f"Zapisano macierz konfuzji do: {output_path}")
    plt.close()


def plot_roc_curve_custom(y_true, y_probs, model_name, output_path=None):
    """Generuje i zapisuje krzywą ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Odsetek fałszywie pozytywnych (FPR)')
    plt.ylabel('Odsetek prawdziwie pozytywnych (TPR)')
    plt.title(f'Krzywa ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if CONFIG['verbose']:
            print(f"Zapisano krzywą ROC do: {output_path}")
    plt.close()
    return roc_auc


def plot_precision_recall_curve_custom(y_true, y_probs, model_name, output_path=None):
    """Generuje i zapisuje krzywą precyzji-odwołania."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(8,6))
    plt.step(recall, precision, color='b', alpha=0.7, where='post',
             label=f'Krzywa Precyzja-Odwołanie (AP = {avg_precision:.2f})')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Odwołanie (Recall)')
    plt.ylabel('Precyzja (Precision)')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Krzywa Precyzja-Odwołanie - {model_name}')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.7)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if CONFIG['verbose']:
            print(f"Zapisano krzywą Precyzja-Odwołanie do: {output_path}")
    plt.close()
    return avg_precision

def analyze_single_model(model_instance: ECGBaseModel, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series, model_name: str, base_output_dir: str):
    """Przeprowadza analizę dla pojedynczego modelu i zapisuje wyniki."""
    
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    if CONFIG['verbose']:
        print(f"\n--- Analiza modelu: {model_name} ---")
        print(f"Wyniki zostaną zapisane w: {model_output_dir}")

    # Trenowanie modelu
    model_instance.fit(X_train, y_train)

    # Predykcje
    y_pred_test = model_instance.predict(X_test)
    y_probs_test = None
    try:
        y_probs_test = model_instance.predict_proba(X_test)[:, 1] # Prawdopodobieństwo klasy pozytywnej
    except AttributeError:
        if CONFIG['verbose']:
            print(f"Model {model_name} nie wspiera predict_proba. Pomijam analizy ROC i PR.")
    
    # Metryki
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'f1_score': f1_score(y_test, y_pred_test, zero_division=0),
        'precision': precision_score(y_test, y_pred_test, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, zero_division=0),
        'roc_auc': None,
        'average_precision': None
    }

    # Raport klasyfikacji
    report = classification_report(y_test, y_pred_test, zero_division=0)
    report_path = os.path.join(model_output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    if CONFIG['verbose']:
        print(f"Zapisano raport klasyfikacji do: {report_path}")
        print("Raport klasyfikacji:\n", report)

    # Krzywa uczenia
    lc_path = os.path.join(model_output_dir, 'learning_curve.png')
    plot_learning_curve_custom(model_instance, f'Krzywa uczenia - {model_name}',
                        X_train, y_train, cv=CONFIG['cv_folds'], n_jobs=-1, scoring='f1', output_path=lc_path)

    # Macierz konfuzji
    cm_path = os.path.join(model_output_dir, 'confusion_matrix.png')
    plot_confusion_matrix_custom(y_test, y_pred_test, classes=np.unique(y_test), model_name=model_name, output_path=cm_path)

    if y_probs_test is not None:
        # Krzywa ROC
        roc_path = os.path.join(model_output_dir, 'roc_curve.png')
        metrics['roc_auc'] = plot_roc_curve_custom(y_test, y_probs_test, model_name, output_path=roc_path)
        
        # Krzywa Precyzji-Odwołania
        pr_path = os.path.join(model_output_dir, 'precision_recall_curve.png')
        metrics['average_precision'] = plot_precision_recall_curve_custom(y_test, y_probs_test, model_name, output_path=pr_path)

    # Zapis metryk
    metrics_path = os.path.join(model_output_dir, 'metrics_summary.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    if CONFIG['verbose']:
        print(f"Zapisano podsumowanie metryk do: {metrics_path}")
        print("Podsumowanie metryk:", metrics)
    
    return metrics


def main():
    """Główna funkcja do uruchamiania analizy modeli."""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    if CONFIG['verbose']:
        print("====== Rozpoczęcie analizy modeli ======")
        print(f"Wyniki będą zapisywane w katalogu: {CONFIG['output_dir']}")

    X, y = load_selected_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=y
    )
    
    if CONFIG['verbose']:
        print(f"Dane podzielone na zbiory treningowe ({X_train.shape[0]} próbek) i testowe ({X_test.shape[0]} próbek).")

    models_to_analyze = [
        (RandomForestModel(random_state=CONFIG['random_state']), "RandomForest"),
        (GradientBoostingModel(random_state=CONFIG['random_state']), "GradientBoosting"),
        (SVMModel(random_state=CONFIG['random_state'], probability=True), "SVM_RBF"), # probability=True dla ROC/PR
        (LogisticRegressionModel(random_state=CONFIG['random_state']), "LogisticRegression"), # usunięto max_iter=1000
        (CatBoostModel(random_seed=CONFIG['random_state'], verbose=0), "CatBoost"), # verbose=0 by uniknąć logów
        (XGBoostModel(random_state=CONFIG['random_state']), "XGBoost") # usunięto parametry niezdefiniowane w konstruktorze
    ]
    
    all_model_metrics = {}

    for model_instance, model_name in models_to_analyze:
        try:
            metrics = analyze_single_model(model_instance, X_train, y_train, X_test, y_test,
                                 model_name, CONFIG['output_dir'])
            all_model_metrics[model_name] = metrics
        except Exception as e:
            print(f"BŁĄD podczas analizy modelu {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Zapis zbiorczego podsumowania metryk
    summary_metrics_path = os.path.join(CONFIG['output_dir'], 'all_models_metrics_summary.json')
    with open(summary_metrics_path, 'w') as f:
        json.dump(all_model_metrics, f, indent=4)
    if CONFIG['verbose']:
        print(f"\nZapisano zbiorcze podsumowanie metryk wszystkich modeli do: {summary_metrics_path}")
        
    # Opcjonalnie: Konwersja do DataFrame i zapis jako CSV
    summary_df = pd.DataFrame.from_dict(all_model_metrics, orient='index')
    summary_csv_path = os.path.join(CONFIG['output_dir'], 'all_models_metrics_summary.csv')
    summary_df.to_csv(summary_csv_path)
    if CONFIG['verbose']:
         print(f"Zbiorcze podsumowanie metryk (CSV) zapisano do: {summary_csv_path}")
         print("\nPodsumowanie wszystkich modeli:")
         print(summary_df)

    if CONFIG['verbose']:
        print("\n====== Analiza modeli zakończona ======")

if __name__ == "__main__":
    # Zmiana backendu matplotlib, jeśli skrypt jest uruchamiany bezpośrednio i nie jest na macOS
    # (na macOS backend 'Agg' jest już ustawiany w analyze_features.py, co może być problematyczne, jeśli importujemy stamtąd)
    import platform
    if platform.system() != 'Darwin':
        try:
            current_backend = plt.get_backend()
            if CONFIG['verbose']:
                print(f"Aktualny backend matplotlib: {current_backend}")
            if not current_backend.lower() == 'agg': # Unikaj zmiany, jeśli już jest 'Agg'
                plt.switch_backend('Agg')
                if CONFIG['verbose']:
                    print("Zmieniono backend matplotlib na 'Agg' dla działania w tle.")
        except Exception as e:
            if CONFIG['verbose']:
                print(f"Nie udało się zmienić backendu matplotlib: {e}")
    main()