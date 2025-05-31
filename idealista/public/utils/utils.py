# %%
import joblib
import plotly.express as px
import numpy as np
import pandas as pd
from prophet import Prophet
from datetime import datetime
import folium
from folium import Choropleth
import geopandas as gpd
import os
from xgboost import XGBRegressor
from models.models import State, HousingType, HousingPrice, EconomicFactor, EconomicFactorValue, HPI
import time

def get_month_from_quartile(quartile):
        return {1: '01', 2: '04', 3: '07', 4: '10'}.get(quartile)

def get_statal_hpi():
    statal_hpi = pd.read_csv('../public/datasets/indices.csv', delimiter=';')
    statal_hpi = statal_hpi.rename(columns={'place_name': 'State','yr': 'Year', 'period': 'Quartile', 'index_nsa': 'HPI'})
    statal_hpi = statal_hpi[statal_hpi['Year'] >= 2000]
    
    states = statal_hpi['State'].unique()
    # Añadir datos de 2024 para el tercer trimestre
    for state in states:
        nuevos_datos = (
        statal_hpi[(statal_hpi['Year'] == 2024) & (statal_hpi['Quartile'] == 3) & (statal_hpi['State'] == state)]
        .assign(Quartile=4)
        )
        statal_hpi = pd.concat([statal_hpi, nuevos_datos], ignore_index=True)

    expanded_rows = []
    for index, row in statal_hpi.iterrows():
        start_date = pd.to_datetime(f"{row['Year']}-{get_month_from_quartile(row['Quartile'])}-01")
        
        for month in range(3):
            month_date = start_date + pd.DateOffset(months=month)
            expanded_rows.append({
                'Date': month_date,
                'state': row['State'],
                'HPI': row['HPI']
            })
    
    statal_hpi = pd.DataFrame(expanded_rows)
    statal_hpi['Date'] = statal_hpi['Date'] + pd.offsets.MonthEnd(0)
    statal_hpi = statal_hpi.set_index('Date')
    return statal_hpi

# %%
# ==============================================================
# EXTRACCIÓN DE DATOS
# ==============================================================

def get_CPI():
    cpi_df = pd.read_csv('../public/datasets/cpi.csv', delimiter=';')
    cpi_df = cpi_df[['Date', 'CPI']]
    cpi_df['CPI'] = cpi_df['CPI'].apply(lambda x: float(x.replace(',', '.')))
    
    cpi_df['Date'] = cpi_df.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d') + pd.offsets.MonthEnd(0))
    cpi_df = cpi_df.set_index('Date')
    return cpi_df


def get_usa_hpi():
    
    usa_df = pd.read_csv('../public/datasets/indices_usa.csv', delimiter=';')
    usa_df = usa_df[usa_df['yr'] >= 2000]
    usa_df = usa_df.rename(columns={'yr': 'Year', 'period': 'Quartile', 'index_nsa': 'HPI'})
    
    nuevos_datos = (
        usa_df[(usa_df['Year'] == 2024) & (usa_df['Quartile'] == 3)]
        .assign(Quartile=4)
    )    
    usa_df = pd.concat([usa_df, nuevos_datos], ignore_index=True)
    
    expanded_rows = []
    for index, row in usa_df.iterrows():
        start_date = pd.to_datetime(f"{row['Year']}-{get_month_from_quartile(row['Quartile'])}-01")
        
        for month in range(3):
            month_date = start_date + pd.DateOffset(months=month)
            expanded_rows.append({
                'Date': month_date,
                'HPI': row['HPI']
            })
    
    usa_df = pd.DataFrame(expanded_rows)
    usa_df['Date'] = usa_df['Date'] + pd.offsets.MonthEnd(0)
    usa_df = usa_df.set_index('Date')
    return usa_df


def get_gdp():
    gdp = pd.read_csv('../public/datasets/gdp.csv', delimiter=';')
    gdp = gdp.melt()
    gdp = gdp.rename(columns={'variable': 'Year', 'value': 'GDP'})
    gdp['Year'] = gdp['Year'].astype(int)
    gdp['GDP'] = gdp['GDP'].apply(lambda x: int(x)/pow(10,12)).astype(float)
    return gdp

# ==============================================================
# PREPROCESAMIENTO DE DATOS
# ==============================================================

def average_bedrooms():
    
    avg_bdrms = pd.read_csv('../public/datasets/bedrooms_single_family.csv', delimiter=';', index_col='RegionName')
    avg_bdrms = avg_bdrms.iloc[:, 4:]

    avg_bdrms = avg_bdrms.loc[avg_bdrms.isnull().sum(axis=1) <= 5]
    indices_to_remove = set()

    for i in range(1, 6):
        bdrms = pd.read_csv(f'../public/datasets/bedrooms_{i}.csv', delimiter=';', index_col='RegionName')
        bdrms = bdrms.iloc[:, 4:]

        indices_with_many_nulls = bdrms.index[bdrms.isnull().sum(axis=1) > 5]
        indices_to_remove.update(indices_with_many_nulls)

    avg_bdrms = avg_bdrms.drop(index=[indice for indice in indices_to_remove if indice in avg_bdrms.index])

    for i in range(1, 6):
        bdrms = pd.read_csv(f'../public/datasets/bedrooms_{i}.csv', delimiter=';', index_col='RegionName')
        bdrms = bdrms.iloc[:, 4:]
        bdrms = bdrms.drop(index=indices_to_remove)
        avg_bdrms = avg_bdrms.add(bdrms)

    avg_bdrms = avg_bdrms.interpolate(method='linear', axis=1, limit_direction="both")
    avg_bdrms = avg_bdrms.div(6)
    avg_bdrms = avg_bdrms.round(3)

    return avg_bdrms


def add_lags_and_features(data, cols, lags=[1, 3, 6, 12]):
    data = data.copy()
    
    # Añadir diferencias
    for col in cols:
        data[f'{col}_diff'] = data[col].diff()

    # Añadir lags de las diferencias
    for col in cols:
        for lag in lags:
            data[f'{col}_diff_lag_{lag}'] = data[f'{col}_diff'].shift(lag)
    
    return data.dropna()

def setup_model():
    avg_bdrms = pd.read_csv('../public/datasets/bedrooms_single_family.csv', delimiter=';', index_col='RegionName')
    avg_bdrms = avg_bdrms.iloc[:, 4:]

    avg_bdrms = avg_bdrms.loc[avg_bdrms.isnull().sum(axis=1) <= 5]
    indices_to_remove = set()

    for i in range(1, 6):
        bdrms = pd.read_csv(f'../public/datasets/bedrooms_{i}.csv', delimiter=';', index_col='RegionName')
        bdrms = bdrms.iloc[:, 4:]

        indices_with_many_nulls = bdrms.index[bdrms.isnull().sum(axis=1) > 5]
        indices_to_remove.update(indices_with_many_nulls)

    avg_bdrms = avg_bdrms.drop(index=[indice for indice in indices_to_remove if indice in avg_bdrms.index]).T
    avg_bdrms = avg_bdrms.rename(columns={col: f"{col}_single_family" for col in avg_bdrms.columns})
    for i in range(1, 6):
        bdrms = pd.read_csv(f'../public/datasets/bedrooms_{i}.csv', delimiter=';', index_col='RegionName')
        bdrms = bdrms.iloc[:, 4:]
        bdrms = bdrms.drop(index=[indice for indice in indices_to_remove if indice in bdrms.index]).T
        bdrms = bdrms.rename(columns={col: f"{col}_{i}" for col in bdrms.columns})
        avg_bdrms = pd.concat([avg_bdrms, bdrms], axis=1)
    
    data = avg_bdrms.interpolate(method='linear', axis=0, limit_direction="both")

    data.index = pd.to_datetime(data.index)
    data.index.name = 'Date'
    usa_hpi = get_usa_hpi()
    gdp = get_gdp()
    gdp.set_index('Year', inplace=True)
    gdp.index = pd.to_datetime(gdp.index.astype(str) + "-01-01") + pd.offsets.MonthEnd(0)

    monthly_index = pd.date_range(start="2000-01-31", end="2024-01-31", freq='ME')
    gdp = gdp.reindex(monthly_index).interpolate(method='linear', limit_direction='forward', axis=0)
    gdp_monthly = gdp.reindex(monthly_index).interpolate()

    m = Prophet()
    m.fit(gdp_monthly.reset_index().rename(columns={'index':'ds', 'GDP':'y'}))

    future =m.make_future_dataframe(periods=23, freq='ME')
    forecast = m.predict(future)

    gdp = pd.DataFrame(forecast[['ds', 'yhat']]).rename(columns={'ds':'Date', 'yhat':'GDP'}).set_index('Date')
    
    data.index = pd.to_datetime(data.index)
    data.index.name = 'Date'
    result = data.join(usa_hpi)
    result = result.join(gdp)
    cpi = get_CPI()
    
    result = result.join(cpi)

    return result


# ==============================================================
# VISUALIZACIÓN DE DATOS EN MAPA
# ==============================================================

def get_boundaries():
    state_boundaries = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
    state_boundaries = state_boundaries.to_crs(epsg=4326)[['name', 'geometry']]
    state_boundaries = state_boundaries.rename(columns={'name': 'RegionName'}).set_index('RegionName')
    return state_boundaries

def plot_interactive_map(year, bedrooms=-1):
    year = str(year)
    exogenous = EconomicFactor.objects.values_list('name', flat=True)
    
    states = State.objects.values_list('name', flat=True)

    state_boundaries = get_boundaries()
    real_prices_map = folium.Map(location=[45.8283, -110.5795], zoom_start=3)
    predicted_prices_map = folium.Map(location=[45.8283, -110.5795], zoom_start=3)

    # Cargar y mostrar los datos de precios por estado y año reales
    if bedrooms == -1:
        house_type = HousingType.objects.get(name='average')
        bdrms = HousingPrice.objects.filter(housing_type=house_type, date__year=year).values('state__name', 'price', 'date')

        bdrms = pd.DataFrame(list(bdrms))
        if not bdrms.empty:
            bdrms = bdrms.rename(columns={'state__name': 'RegionName'})
            bdrms = bdrms.pivot(index='RegionName', columns='date', values='price')
            
    else:
        house_type = HousingType.objects.get(name=str(bedrooms).split('_')[0])
        bdrms = HousingPrice.objects.filter(housing_type=house_type, date__year=year).values('state__name', 'price', 'date')
        
        bdrms = pd.DataFrame(list(bdrms))
        if not bdrms.empty:
            bdrms = bdrms.rename(columns={'state__name': 'RegionName'})
            bdrms = bdrms.pivot(index='RegionName', columns='date', values='price')


    months = bdrms.copy().columns.map(lambda x: str(x).split('-')[0]).get_indexer_for([year])
    bdrms = bdrms.iloc[:, months]
    for idx, row in bdrms.iterrows():
        bdrms.at[idx, year] = row.mean()
    prices = pd.Series(bdrms[year])
    
    prices.name = f'price_{year}{("-" + str(bedrooms)) if bedrooms != -1 else "_1-5_"} bedrooms'
    
    geo_data = state_boundaries.__geo_interface__
    for feature in geo_data['features']:
        state_id = feature['id']
        feature['properties']['name'] = state_id 
        feature['properties'][prices.name] = prices.get(state_id, 'N/A')
    
    choropleth = Choropleth(
        geo_data=geo_data,
        data=prices,
        key_on='feature.id',
        fill_color='YlGnBu',
        legend_name=f'''Prices in {year} (USD) {
            (" - " + str(bedrooms)) + " bedrooms" if bedrooms != -1 else " 1 - 5 bedrooms and single family"}'''  ,
    ).add_to(real_prices_map)
    
    
    folium.GeoJson(
        geo_data,
        style_function=lambda feature: {
            'fillColor': '#ffffff',
            'color': 'black',
            'weight': 0.1,
            'fillOpacity': 0.0
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['name', prices.name],
            aliases=['State:', 'Average Price:'],
            localize=True,
            sticky=True,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 2px solid black;
                border-radius: 3px;
                box-shadow: 3px;
            """,
            max_width=800,
        )
    ).add_to(real_prices_map)

    # Predecir precios por estado y año

    predicted_prices = pd.Series(index=bdrms.index)
    predicted_prices.name = f'price_{year}{("-" + str(bedrooms)) if bedrooms != -1 else "_1-5_"} bedrooms'

    init_date = datetime.strptime(f'{year}-01-31', '%Y-%m-%d')
    end_date = datetime.strptime(f'{year}-12-31', '%Y-%m-%d')

    for state in states:
        house_type = HousingType.objects.get(name='average')
        if bedrooms == -1:
            model = joblib.load(f'models/state_models/{state}/average_model.pkl')
        else: 
            house_type = HousingType.objects.get(name=str(bedrooms).split('_')[0])
            model = joblib.load(f'models/state_models/{state}/{bedrooms}_model.pkl')

        state = State.objects.get(name=state)

        data = HousingPrice.objects.filter(state=state, housing_type=house_type).order_by('date')
        data =pd.DataFrame(data.values_list('price'), columns=['Price'], index=pd.to_datetime([d.date for d in data]))

        # Add economic factors
        gdp_data = EconomicFactorValue.objects.filter(factor__name='GDP').order_by('date')
        gdp_data = pd.DataFrame(gdp_data.values_list('value'), columns=['GDP'], index=pd.to_datetime([d.date for d in gdp_data]))
        hpi_data = HPI.objects.filter(state=state).order_by('date')
        hpi_data = pd.DataFrame(hpi_data.values_list('value'), columns=['HPI'], index=pd.to_datetime([d.date for d in hpi_data]))
        cpi_data = EconomicFactorValue.objects.filter(factor__name='CPI').order_by('date')
        cpi_data = pd.DataFrame(cpi_data.values_list('value'), columns=['CPI'], index=pd.to_datetime([d.date for d in cpi_data]))
        
        data = data.join(gdp_data).join(hpi_data).join(cpi_data)
        
        data_diff = add_lags_and_features(data.copy(), ['Price', 'GDP', 'HPI', 'CPI'], lags=[1, 3, 6, 12, 24])

        data_diff = data_diff[[col for col in data_diff.columns if '_diff' in col]]


        test_data = data_diff[(data_diff.index >= init_date) & (data_diff.index <= end_date)]        
        X_test = test_data.copy().drop('Price_diff', axis=1)


        # Test predictions
        y_pred_diff_test = model.predict(X_test)
        y_pred_test = np.cumsum(y_pred_diff_test) + data['Price'].loc[init_date]

        predicted_prices[state.name] = np.mean(y_pred_test)

    geo_data_predicted = state_boundaries.__geo_interface__
    for feature in geo_data_predicted['features']:
        state_id = feature['id']
        feature['properties']['name'] = state_id 
        feature['properties'][predicted_prices.name] = predicted_prices.get(state_id, 'N/A')
    
    real_prices_choropleth = Choropleth(
        geo_data=geo_data_predicted,
        data=predicted_prices,
        key_on='feature.id',
        fill_color='YlGnBu',
        legend_name=f'''Predicted prices in {year} (USD) {
            (" - " + str(bedrooms)) + " bedrooms" if bedrooms != -1 else " 1 - 5 bedrooms and single family"}'''  ,
    ).add_to(predicted_prices_map)
    
    folium.GeoJson(
        geo_data_predicted,
        style_function=lambda feature: {
            'fillColor': '#ffffff',
            'color': 'black',
            'weight': 0.1,
            'fillOpacity': 0.0
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['name', predicted_prices.name],
            aliases=['State:', 'Average Price:'],
            localize=True,
            sticky=True,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 2px solid black;
                border-radius: 3px;
                box-shadow: 3px;
            """,
            max_width=800,
        )
    ).add_to(predicted_prices_map)
    
    return real_prices_map, predicted_prices_map, predicted_prices, prices


# ==============================================================
# IMPORTACIÓN A LA BASE DE DATOS
# ==============================================================


def import_to_db():
    data = setup_model()
    housing_types = data.columns[:-3].map(lambda x: x.split('_')[1]).unique()
    states = data.columns[:-3].map(lambda x: x.split('_')[0]).unique()

    statal_hpi = get_statal_hpi()

    State.objects.all().delete()
    HousingType.objects.all().delete()
    HousingPrice.objects.all().delete()
    EconomicFactor.objects.all().delete()
    EconomicFactorValue.objects.all().delete()
    HPI.objects.all().delete()


    for state_name in states:
        state_obj, _ = State.objects.get_or_create(name=state_name)
        state_hpi_data = statal_hpi.copy()[statal_hpi['state'] == state_name]
        for date, row in state_hpi_data.iterrows():
            if not pd.isna(row['HPI']):
                HPI.objects.get_or_create(
                    state=state_obj,
                    factor=EconomicFactor.objects.get_or_create(name='HPI')[0],
                    date=date,
                    value=row['HPI']
                )

        for housing_type_name in housing_types:
            housing_type_obj, _ = HousingType.objects.get_or_create(name=housing_type_name)
            state_housing_data = data.copy()[[col for col in data.columns 
                                           if (state_name in col) and (housing_type_name in col)]]
            
            for date, row in state_housing_data.iterrows():
                if not pd.isna(row.values[0]):
                    HousingPrice.objects.get_or_create(
                        housing_type=housing_type_obj,
                        state=state_obj,
                        date=date,
                        price=row.values[0],
                        currency='USD'
                    )
    economic_factors = ['GDP', 'CPI']
    for factor_name in economic_factors:
        factor_obj, _ = EconomicFactor.objects.get_or_create(name=factor_name)
        for date, value in data[factor_name].items():
            if not pd.isna(value):
                EconomicFactorValue.objects.get_or_create(
                    factor=factor_obj,
                    date=date,
                    value=value
                )

    avg_data = average_bedrooms()
    for state, row in avg_data.iterrows():
        state_obj, _ = State.objects.get_or_create(name=state)
        for date, value in row.items():
            if not pd.isna(value):
                HousingPrice.objects.get_or_create(
                    housing_type=HousingType.objects.get_or_create(name='average')[0],
                    state=state_obj,
                    date=date,
                    price=value,
                    currency='USD'
                )
                    

# ==============================================================
# ENTRENAMIENTO Y GENERACIÓN DE MODELOS
# ==============================================================

def train_model(df):
    params = {'colsample_bytree': np.float64(0.998166958028371),
    'gamma': np.float64(9.747931621467831),
    'learning_rate': np.float64(0.10754885295204053),
    'max_depth': 4,
    'n_estimators': 292,
    'reg_alpha': np.float64(6.802282424312914),
    'reg_lambda': np.float64(0.7219840897917584),
    'subsample': np.float64(0.5153262511029031)}
    data = add_lags_and_features(df, ['Price', 'GDP', 'HPI', 'CPI'], lags=[1, 3, 6, 12, 24])
    data = data.dropna()
    data_diff = data.copy()[[col for col in data.columns if '_diff' in col]]
    
    end_date = data.index.max()
    test_start = end_date - pd.DateOffset(years=4)
    val_start = test_start - pd.DateOffset(years=4)
    train_data = data_diff[data_diff.index <= val_start]
    val_data = data_diff[(data_diff.index > val_start) & (data_diff.index <= test_start)]
    
    X_train = train_data.drop('Price_diff', axis=1)
    y_train_diff = train_data['Price_diff']
    X_val = val_data.drop('Price_diff', axis=1)
    y_val_diff = val_data['Price_diff']
    

    xgb = XGBRegressor(**params)
    xgb.fit(pd.concat([X_train, X_val]), pd.concat([y_train_diff, y_val_diff]))

    return xgb


def train_state_models():
    states = State.objects.values_list('name', flat=True)

    # Estadísticas de entrenamiento
    start_time = time.time()
    models_trained = 0
    models_failed = 0
    errors = []
    
    routes = [
        'single_family',
        '1',
        '2',
        '3',
        '4',
        '5',
        'average'
    ]

    for state_name in states:
        try:
            state_obj = State.objects.get(name=state_name)
            os.makedirs(f'models/state_models/{state_name}', exist_ok=True)
            
            for route in routes:
                try:
                    if route == 'single_family':
                        house_type = HousingType.objects.get(name='single')
                    else:
                        house_type = HousingType.objects.get(name=route)
                    
                    data = HousingPrice.objects.filter(state=state_obj, housing_type=house_type).order_by('date')
                    
                    data = pd.DataFrame(data.values_list('price'), columns=['Price'], index=pd.to_datetime([d.date for d in data]))
                    
                    # Factores económicos
                    gdp_data = EconomicFactorValue.objects.filter(factor__name='GDP').order_by('date')
                    gdp_data = pd.DataFrame(gdp_data.values_list('value'), columns=['GDP'], index=pd.to_datetime([d.date for d in gdp_data]))
                    
                    hpi_data = HPI.objects.filter(state=state_obj).order_by('date')
                    hpi_data = pd.DataFrame(hpi_data.values_list('value'), columns=['HPI'], index=pd.to_datetime([d.date for d in hpi_data]))
                    
                    cpi_data = EconomicFactorValue.objects.filter(factor__name='CPI').order_by('date')
                    cpi_data = pd.DataFrame(cpi_data.values_list('value'), columns=['CPI'], index=pd.to_datetime([d.date for d in cpi_data]))
                                        
                    data = data.join(gdp_data).join(hpi_data).join(cpi_data)
                    
                    model = train_model(data)
                    
                    # Salida
                    output_dir = f'models/state_models/{state_name}'
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = f'{output_dir}/{route}_model.pkl'
                    
                    # Guardar modelo
                    joblib.dump(model, output_path)
                    print(f"Model for {state_name} - {route} saved.")
                    models_trained += 1
                    
                except Exception as e:
                    error_msg = f"Error training model for {state_name} - {route}: {str(e)}"
                    print(error_msg)
                    errors.append(error_msg)
                    models_failed += 1
        except Exception as e:
            error_msg = f"Error processing state {state_name}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    end_time = time.time()
    
    results = {
        'time_taken': end_time - start_time,
        'states_trained': len(states),
        'housing_types_trained': len(routes),
        'models_trained_successfully': models_trained,
        'models_failed': models_failed,
        'errors': errors[:10]
    }
    
    return results