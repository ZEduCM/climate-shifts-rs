from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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

# Converter coluna date e extrair mês
combined_df['date'] = pd.to_datetime(combined_df['date'])
combined_df['month'] = combined_df['date'].dt.to_period('M')

# Mapeamento station_code → mesorregião
region_map = {
    'A801': 'Metropolitana de Porto Alegre',
    'B807': 'Metropolitana de Porto Alegre',
    'A884': 'Metropolitana de Porto Alegre',
    'A879': 'Metropolitana de Porto Alegre',
    'A878': 'Metropolitana de Porto Alegre',
    'A808': 'Metropolitana de Porto Alegre',
    'A834': 'Metropolitana de Porto Alegre',
    'A840': 'Nordeste Rio-grandense',
    'A897': 'Nordeste Rio-grandense',
    'A844': 'Nordeste Rio-grandense',
    'A894': 'Nordeste Rio-grandense',
    'A829': 'Nordeste Rio-grandense',
    'A880': 'Nordeste Rio-grandense',
    'A853': 'Noroeste Rio-grandense',
    'A828': 'Noroeste Rio-grandense',
    'A854': 'Noroeste Rio-grandense',
    'A883': 'Noroeste Rio-grandense',
    'A856': 'Noroeste Rio-grandense',
    'A839': 'Noroeste Rio-grandense',
    'A810': 'Noroeste Rio-grandense',
    'A805': 'Noroeste Rio-grandense',
    'A852': 'Noroeste Rio-grandense',
    'A837': 'Noroeste Rio-grandense',
    'A813': 'Centro Oriental Rio-grandense',
    'A882': 'Centro Oriental Rio-grandense',
    'A803': 'Centro Ocidental Rio-grandense',
    'A833': 'Centro Ocidental Rio-grandense',
    'A889': 'Centro Ocidental Rio-grandense',
    'A886': 'Centro Ocidental Rio-grandense',
    'A812': 'Sudeste Rio-grandense',
    'A811': 'Sudeste Rio-grandense',
    'A887': 'Sudeste Rio-grandense',
    'A893': 'Sudeste Rio-grandense',
    'A836': 'Sudeste Rio-grandense',
    'A802': 'Sudeste Rio-grandense',
    'A899': 'Sudeste Rio-grandense',
    'A826': 'Sudoeste Rio-grandense',
    'A827': 'Sudoeste Rio-grandense',
    'A881': 'Sudoeste Rio-grandense',
    'A831': 'Sudoeste Rio-grandense',
    'A804': 'Sudoeste Rio-grandense',
    'A832': 'Sudoeste Rio-grandense',
    'A830': 'Sudoeste Rio-grandense',
    'A809': 'Sudoeste Rio-grandense'
}

combined_df['region'] = combined_df['station_code'].map(region_map)
combined_df = combined_df.dropna(subset=['region'])

# last_date = combined_df['date'].max()
# one_year_ago = last_date - pd.DateOffset(years=1)
# df_last = combined_df[combined_df['date'] >= one_year_ago]

# Cálculo da temperatura média mensal por região
monthly = (
    combined_df
    .groupby(['region', 'month'])
    .agg(monthly_temp_mean=('temp_mean_c', 'mean'))
    .reset_index()
)

# Converter month para timestamp para plot
monthly['month_start'] = monthly['month'].dt.to_timestamp()

# Cria a pasta de saída
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Gera e salva um gráfico PNG por região
for region in monthly['region'].unique():
    data = monthly[monthly['region'] == region]
    
    fig, ax = plt.subplots()
    ax.plot(data['month_start'], data['monthly_temp_mean'])
    ax.set_title(f'Temperatura Média Mensal – {region}')
    ax.set_xlabel('Mês')
    ax.set_ylabel('Temp. Média (°C)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salva o PNG
    filename = f"{region.replace(' ', '_').replace('–','').lower()}.png"
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    
wide = monthly.pivot(index='month_start', columns='region', values='monthly_temp_mean')

plt.figure(figsize=(10, 6))
for region in wide.columns:
    plt.plot(wide.index, wide[region], label=region)
plt.title('Comparativo de Temperatura Média Mensal por Mesorregião')
plt.xlabel('Mês')
plt.ylabel('Temp. Média (°C)')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('comparativo_temp_media_mensal.png')
plt.show()

regional_averages = monthly.groupby('region')['monthly_temp_mean'].mean().sort_values()

# Identificar região com menor e maior temperatura média
coldest_region = regional_averages.index[0]
warmest_region = regional_averages.index[-1]

print(f"Região mais fria: {coldest_region} ({regional_averages.iloc[0]:.1f}°C)")
print(f"Região mais quente: {warmest_region} ({regional_averages.iloc[-1]:.1f}°C)")

# Filtrar dados apenas para essas duas regiões
comparison_data = monthly[monthly['region'].isin([coldest_region, warmest_region])]

# Criar o gráfico de comparação
plt.figure(figsize=(12, 7))

# Cores para as regiões
colors = {'coldest': '#87CEEB', 'warmest': '#FFB6C1'}  # Azul claro e rosa claro
trend_colors = {'coldest': '#1E90FF', 'warmest': '#DC143C'}  # Azul forte e vermelho forte

# Plotar as duas regiões com linhas leves
for i, region in enumerate([coldest_region, warmest_region]):
    data = comparison_data[comparison_data['region'] == region].sort_values('month_start')
    color_key = 'coldest' if region == coldest_region else 'warmest'
    
    # Linha original com cor leve
    plt.plot(data['month_start'], data['monthly_temp_mean'], 
             marker='o', linewidth=1.5, markersize=4, 
             color=colors[color_key], alpha=0.7, label=f'{region} (dados)')
    
    # Linha de tendência com cor forte
    x_numeric = pd.to_numeric(data['month_start'])
    z = np.polyfit(x_numeric, data['monthly_temp_mean'], 1)
    p = np.poly1d(z)
    plt.plot(data['month_start'], p(x_numeric), 
             linewidth=3, color=trend_colors[color_key], 
             linestyle='--', label=f'{region} (tendência)')

# Configurar o gráfico
plt.title('Comparação: Mesorregião Mais Fria vs Mais Quente\nTemperatura Média Mensal', 
          fontsize=14, fontweight='bold')
plt.xlabel('Mês', fontsize=12)
plt.ylabel('Temperatura Média (°C)', fontsize=12)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Adicionar anotações com diferenças (versão mais robusta)
for month in comparison_data['month_start'].unique():
    cold_data = comparison_data[(comparison_data['region'] == coldest_region) & 
                               (comparison_data['month_start'] == month)]['monthly_temp_mean']
    warm_data = comparison_data[(comparison_data['region'] == warmest_region) & 
                               (comparison_data['month_start'] == month)]['monthly_temp_mean']
    
    # Verificar se ambos os dados existem para este mês
    if not cold_data.empty and not warm_data.empty:
        cold_temp = cold_data.iloc[0]
        warm_temp = warm_data.iloc[0]
        diff = warm_temp - cold_temp
        
        # Adicionar texto da diferença no meio das duas linhas
        mid_temp = (cold_temp + warm_temp) / 2

plt.tight_layout()

# Salvar o gráfico
comparison_filename = 'comparacao_menor_vs_maior_temp.png'
plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
plt.close()  # Fechar a figura para liberar memória

print(f"Gráfico de comparação salvo como: {comparison_filename}")

# Estatísticas adicionais
print("\n=== ESTATÍSTICAS COMPARATIVAS ===")
print(f"Diferença média anual: {regional_averages.iloc[-1] - regional_averages.iloc[0]:.1f}°C")
print(f"Maior diferença mensal: {comparison_data.groupby('month_start').apply(lambda x: x['monthly_temp_mean'].max() - x['monthly_temp_mean'].min()).max():.1f}°C")
print(f"Menor diferença mensal: {comparison_data.groupby('month_start').apply(lambda x: x['monthly_temp_mean'].max() - x['monthly_temp_mean'].min()).min():.1f}°C")

print(f'Todos os gráficos foram salvos em ./{output_dir}/')  