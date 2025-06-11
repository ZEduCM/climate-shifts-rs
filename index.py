from pathlib import Path;
import pandas as pd;

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
};

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
};

def read_station(file: Path) -> tuple[pd.DataFrame, dict]:
    """
    Lê um arquivo da estação INMET e devolve um DataFrame
    já com as colunas de metadados anexadas.
    """
    metadata = {};
    with file.open(encoding='utf-8') as f:
        for _ in range(8):
            row = next(f).strip();
            if not row:
                continue;
            key, value = row.split(':', 1);
            metadata[MAP_METADATA.get(key).strip()] = value.strip();
    
    df = pd.read_csv(
        file,
        sep=';',
        decimal=',',
        na_values=['null', ''],
        skiprows=9,
        engine='python'
    );
    
    df = df.rename(columns=MAP_COLS);
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')];
    
    return df, metadata;

folder = Path('BDMEP-RS');

stations = {};

for file in folder.glob('*.csv'):
    df, metadata = read_station(file);
    stations[metadata.get('station_code')] = df;
    
print(stations['A810'].head(10));