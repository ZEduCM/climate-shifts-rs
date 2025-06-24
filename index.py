from pathlib import Path
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

MAP_METADATA = {
    'Nome'            : 'name',
    'Codigo Estacao'  : 'station_code',
    'Latitude'        : 'latitude',
    'Longitude'       : 'longitude',
    'Altitude'        : 'altitude_m',
    'Situacao'        : 'status',
    'Data Inicial'    : 'date_start',
    'Data Final'      : 'date_end',
    'Periodicidade da Medicao' : 'frequency'
}

MAP_COLS = {
    'Data Medicao'                                               : 'date',
    'PRECIPITACAO TOTAL, DIARIO (AUT)(mm)'                       : 'precip_mm',
    'PRESSAO ATMOSFERICA MEDIA DIARIA (AUT)(mB)'                 : 'pressure_mb',
    'TEMPERATURA DO PONTO DE ORVALHO MEDIA DIARIA (AUT)(°C)'     : 'dewpoint_c',
    'TEMPERATURA MAXIMA, DIARIA (AUT)(°C)'                       : 'temp_max_c',
    'TEMPERATURA MEDIA, DIARIA (AUT)(°C)'                        : 'temp_mean_c',
    'TEMPERATURA MINIMA, DIARIA (AUT)(°C)'                       : 'temp_min_c',
    'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)'              : 'rh_mean_pct',
    'UMIDADE RELATIVA DO AR, MINIMA DIARIA (AUT)(%)'             : 'rh_min_pct',
    'VENTO, RAJADA MAXIMA DIARIA (AUT)(m/s)'                     : 'wind_gust_ms',
    'VENTO, VELOCIDADE MEDIA DIARIA (AUT)(m/s)'                  : 'wind_mean_ms'
}

def analyze_multiple_regression(df, target_column, feature_columns=None, test_size=0.3, scale_features=False, save_plots=True, output_dir='regression_plots'):
    """
    Realiza análise completa de regressão linear múltipla
    
    Parameters:
    -----------
    df : DataFrame
        Dados para análise
    target_column : str
        Nome da coluna que será a variável dependente (Y)
    feature_columns : list, optional
        Lista de colunas para usar como variáveis independentes (X)
        Se None, usa todas as colunas numéricas exceto target
    test_size : float
        Proporção dos dados para teste (padrão: 0.2)
    scale_features : bool
        Se True, padroniza as variáveis independentes
    save_plots : bool
        Se True, salva os gráficos como PNG (padrão: True)
    output_dir : str
        Diretório para salvar os gráficos (padrão: 'regression_plots')
    
    Returns:
    --------
    dict : Resultados da análise
    """
    
    # Preparar diretório para salvar gráficos
    if save_plots:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{target_column}_regression_{timestamp}"
    
    # Preparar os dados
    df_clean = df.dropna()
    
    if feature_columns is None:
        # Usar todas as colunas numéricas exceto target e date
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_cols if col != target_column]
    
    # Verificar se as colunas existem
    missing_cols = [col for col in feature_columns + [target_column] if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Colunas não encontradas: {missing_cols}")
    
    X = df_clean[feature_columns]
    y = df_clean[target_column]
    
    print(f"=== ANÁLISE DE REGRESSÃO LINEAR MÚLTIPLA ===")
    print(f"Variável dependente (Y): {target_column}")
    print(f"Variáveis independentes (X): {feature_columns}")
    print(f"Total de observações: {len(df_clean)}")
    print(f"Dados de treino: {int(len(df_clean) * (1-test_size))}")
    print(f"Dados de teste: {int(len(df_clean) * test_size)}")
    print("=" * 50)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # Padronizar se solicitado
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
    
    # Criar e treinar o modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Fazer predições
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calcular métricas
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Imprimir resultados do modelo
    print(f"\n=== EQUAÇÃO DO MODELO ===")
    equation = f"{target_column} = {model.intercept_:.4f}"
    for i, (feature, coef) in enumerate(zip(feature_columns, model.coef_)):
        sign = "+" if coef >= 0 else ""
        equation += f" {sign}{coef:.4f}*{feature}"
    print(equation)
    
    print(f"\n=== COEFICIENTES ===")
    print(f"Intercepto: {model.intercept_:.4f}")
    coef_df = pd.DataFrame({
        'Variável': feature_columns,
        'Coeficiente': model.coef_,
        'Coef_Abs': np.abs(model.coef_)
    }).sort_values('Coef_Abs', ascending=False)
    
    print(coef_df.to_string(index=False))
    
    print(f"\n=== MÉTRICAS DE DESEMPENHO ===")
    print(f"R² (Treino): {train_r2:.4f}")
    print(f"R² (Teste):  {test_r2:.4f}")
    print(f"RMSE (Treino): {train_rmse:.4f}")
    print(f"RMSE (Teste):  {test_rmse:.4f}")
    print(f"MAE (Treino): {train_mae:.4f}")
    print(f"MAE (Teste):  {test_mae:.4f}")
    
    # Análise de resíduos
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    
    print(f"\n=== ANÁLISE DE RESÍDUOS ===")
    print(f"Média dos resíduos (treino): {np.mean(residuals_train):.6f}")
    print(f"Desvio padrão dos resíduos (treino): {np.std(residuals_train):.4f}")
    
    # Teste de normalidade dos resíduos
    _, p_value_shapiro = stats.shapiro(residuals_train[:5000] if len(residuals_train) > 5000 else residuals_train)
    print(f"Teste de normalidade (Shapiro-Wilk) p-value: {p_value_shapiro:.6f}")
    print(f"Resíduos são normais: {'Sim' if p_value_shapiro > 0.05 else 'Não'}")
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Valores reais vs preditos
    axes[0,0].scatter(y_test, y_test_pred, alpha=0.6)
    axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Valores Reais')
    axes[0,0].set_ylabel('Valores Preditos')
    axes[0,0].set_title(f'Reais vs Preditos (R² = {test_r2:.3f})')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Resíduos vs valores preditos
    axes[0,1].scatter(y_test_pred, residuals_test, alpha=0.6)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Valores Preditos')
    axes[0,1].set_ylabel('Resíduos')
    axes[0,1].set_title('Resíduos vs Valores Preditos')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Histograma dos resíduos
    axes[1,0].hist(residuals_test, bins=30, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Resíduos')
    axes[1,0].set_ylabel('Frequência')
    axes[1,0].set_title('Distribuição dos Resíduos')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot dos resíduos
    stats.probplot(residuals_test, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot dos Resíduos')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar o primeiro gráfico
    if save_plots:
        plt.savefig(f"{output_dir}/{base_filename}_diagnostics.png", dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {output_dir}/{base_filename}_diagnostics.png")
    
    plt.show()
    
    # Importância das variáveis
    plt.figure(figsize=(10, 6))
    coef_abs_sorted = coef_df.sort_values('Coef_Abs', ascending=True)
    bars = plt.barh(range(len(coef_abs_sorted)), coef_abs_sorted['Coef_Abs'])
    plt.yticks(range(len(coef_abs_sorted)), coef_abs_sorted['Variável'])
    plt.xlabel('Valor Absoluto do Coeficiente')
    plt.title('Importância das Variáveis (Valor Absoluto dos Coeficientes)')
    plt.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    
    # Salvar o segundo gráfico
    if save_plots:
        plt.savefig(f"{output_dir}/{base_filename}_importance.png", dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {output_dir}/{base_filename}_importance.png")
    
    plt.show()
    
    # Gráfico adicional: Matriz de correlação das variáveis
    plt.figure(figsize=(10, 8))
    correlation_data = df_clean[feature_columns + [target_column]].corr()
    mask = np.triu(np.ones_like(correlation_data, dtype=bool))
    sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    plt.title('Matriz de Correlação das Variáveis')
    plt.tight_layout()
    
    # Salvar o terceiro gráfico
    if save_plots:
        plt.savefig(f"{output_dir}/{base_filename}_correlation.png", dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {output_dir}/{base_filename}_correlation.png")
    
    plt.show()
    
    # Retornar resultados
    results = {
        'model': model,
        'scaler': scaler,
        'coefficients': coef_df,
        'metrics': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        },
        'residuals': {
            'train': residuals_train,
            'test': residuals_test,
            'shapiro_p_value': p_value_shapiro
        },
        'predictions': {
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'y_train': y_train,
            'y_test': y_test
        },
        'feature_columns': feature_columns,
        'target_column': target_column,
        'saved_plots': [
            f"{output_dir}/{base_filename}_diagnostics.png",
            f"{output_dir}/{base_filename}_importance.png", 
            f"{output_dir}/{base_filename}_correlation.png"
        ] if save_plots else []
    }
    
    return results

def save_regression_report(results, filename=None):
    """
    Salva um relatório detalhado da regressão em arquivo de texto
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"regression_report_{results['target_column']}_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DE REGRESSÃO LINEAR MÚLTIPLA ===\n\n")
        f.write(f"Variável dependente: {results['target_column']}\n")
        f.write(f"Variáveis independentes: {', '.join(results['feature_columns'])}\n\n")
        
        f.write("=== COEFICIENTES ===\n")
        f.write(f"Intercepto: {results['model'].intercept_:.4f}\n")
        for _, row in results['coefficients'].iterrows():
            f.write(f"{row['Variável']}: {row['Coeficiente']:.4f}\n")
        
        f.write("\n=== MÉTRICAS ===\n")
        for metric, value in results['metrics'].items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write(f"\n=== EQUAÇÃO DO MODELO ===\n")
        equation = f"{results['target_column']} = {results['model'].intercept_:.4f}"
        for feature, coef in zip(results['feature_columns'], results['model'].coef_):
            sign = "+" if coef >= 0 else ""
            equation += f" {sign}{coef:.4f}*{feature}"
        f.write(equation + "\n")
        
        if results['saved_plots']:
            f.write(f"\n=== GRÁFICOS SALVOS ===\n")
            for plot in results['saved_plots']:
                f.write(f"- {plot}\n")
    
    print(f"Relatório salvo: {filename}")
    return filename

def read_station(file: Path) -> tuple[pd.DataFrame, dict]:
    """
    Lê um arquivo da estação INMET e devolve um DataFrame
    já com as colunas de metadados anexadas e dados tratados.
    """
    metadata = {}
    with file.open(encoding='utf-8') as f:
        for _ in range(8):
            row = next(f).strip()
            if not row:
                continue
            key, value = row.split(':', 1)
            metadata[MAP_METADATA.get(key).strip()] = value.strip()
    
    df = pd.read_csv(
        file,
        sep=';',
        decimal=',',
        na_values=['null', ''],
        skiprows=9,
        engine='python'
    )
    
    df = df.rename(columns=MAP_COLS)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # === TRATAMENTO DOS DADOS ===
    
    # Converter coluna de data para datetime e definir como índice
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Remove linhas sem data válida
        df = df.sort_values('date').set_index('date')
    
    # Remover colunas completamente vazias
    df = df.dropna(axis=1, how='all')
    
    # Variáveis meteorológicas contínuas que podem ser interpoladas
    continuous_variables = ['pressure_mb', 'dewpoint_c', 'temp_max_c', 
                          'temp_mean_c', 'temp_min_c', 'rh_mean_pct', 'rh_min_pct']
    
    # Interpolar variáveis contínuas (apenas gaps pequenos de até 3 dias)
    for col in continuous_variables:
        if col in df.columns:
            df[col] = df[col].interpolate(method='time', limit=3, limit_direction='both')
    
    # Precipitação: NaN geralmente significa "sem chuva"
    if 'precip_mm' in df.columns:
        df['precip_mm'] = df['precip_mm'].fillna(0.0)
    
    # Variáveis de vento: interpolar gaps pequenos
    wind_variables = ['wind_gust_ms', 'wind_mean_ms']
    for col in wind_variables:
        if col in df.columns:
            df[col] = df[col].interpolate(method='time', limit=2, limit_direction='both')
    
    # Resetar o índice para manter 'date' como coluna
    if df.index.name == 'date':
        df = df.reset_index()

    return df, metadata

folder = Path('BDMEP-RS')

stations = {}

for file in folder.glob('*.csv'):
    df, metadata = read_station(file)
    stations[metadata.get('station_code')] = df
    
combined_df = pd.concat(
    [df.assign(station_code=code)
     .loc[:, ['station_code', *df.columns]]
     for code, df in stations.items()],
    ignore_index=True
)

combined_df = pd.get_dummies(
    combined_df,
    columns=['station_code'],
    prefix='station',
    drop_first=True
)

# 2. Define as colunas de entrada:
base_features = [
    'precip_mm',
    'pressure_mb',
    'rh_mean_pct',
    'rh_min_pct',
    'wind_gust_ms',
    'wind_mean_ms'
]

# pega todas as colunas que começam com 'station_' geradas pelo get_dummies
dummy_features = [col for col in combined_df.columns if col.startswith('station_')]

feature_columns = base_features + dummy_features

print(combined_df)
    
# print(stations['A810'].head(10))

# df, metadata = read_station(Path(f'{folder}/dados_A801_D_2000-09-21_2025-01-01.csv'))
# print(df)

results = analyze_multiple_regression(
    combined_df,
    target_column='temp_mean_c',
    feature_columns=feature_columns,
    save_plots=True,
    output_dir='charts_regression'
)

print(results)