from django.shortcuts import render
from django.urls import path
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from public.utils.utils import setup_model, add_lags_and_features, plot_interactive_map, import_to_db, train_state_models
import os
from plotly.offline import plot
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
from datetime import datetime
import json
from plotly.io import to_html
from models.models import State, HousingType, HousingPrice, EconomicFactor, EconomicFactorValue, HPI
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from django.contrib import messages


def predict(request):
    """
    View function for the prediction page.
    """
    if not request.user.is_authenticated:
        return render(request, 'account/login.html')
    init_year = request.GET.get('init_year')
    end_year = request.GET.get('end_year')
    house_type = request.GET.get('house_type')
    state = request.GET.get('state', 'default')
    print(f'init_year: {init_year}, end_year: {end_year}, house_type: {house_type}, state: {state}')
    states_list = os.listdir(os.path.join(os.path.dirname(__file__), 'state_models'))
    route = f'{os.path.dirname(__file__)}/state_models/{state}'
    
    if state != 'default' and init_year and end_year and init_year <= end_year and house_type:

        house_type_name = house_type

        init_date = datetime.strptime(f'{init_year}-01-31', '%Y-%m-%d')
        end_date = datetime.strptime(f'{end_year}-12-31', '%Y-%m-%d')
        print(f'init_date: {init_date}')
        print(f'end_date: {end_date}')

        switcher = {
            'Single family': 'single',
            '1 room': '1',
            '2 rooms': '2',
            '3 rooms': '3',
            '4 rooms': '4',
            '5+ rooms': '5',
            'Average': 'average',
        }

        if switcher[house_type] == 'single':
            model = joblib.load(f'{route}/{switcher[house_type_name]}_family_model.pkl')
        else:
            model = joblib.load(f'{route}/{switcher[house_type_name]}_model.pkl')

        state = State.objects.get(name=state)
        house_type = HousingType.objects.get(name=switcher[house_type])
        data = HousingPrice.objects.filter(state=state, housing_type=house_type).order_by('date')
        data =pd.DataFrame(data.values_list('price'), columns=['Price'], index=pd.to_datetime([d.date for d in data]))

        # Add economic factors
        economic_factors = EconomicFactor.objects.values_list('name', flat=True)
     
        for factor in economic_factors:
            
            if factor == 'HPI':
                hpi_data = HPI.objects.filter(state=state).order_by('date')
                hpi_data = pd.DataFrame(hpi_data.values_list('value'), columns=['HPI'], index=pd.to_datetime([d.date for d in hpi_data]))
                data = data.join(hpi_data)
            else:
                factor_data = EconomicFactorValue.objects.filter(factor__name=factor).order_by('date')
                factor_data = pd.DataFrame(factor_data.values_list('value'), columns=[factor], index=pd.to_datetime([d.date for d in factor_data]))
                data = data.join(factor_data)

        
        data_diff = add_lags_and_features(data.copy(), ['Price', 'GDP', 'HPI', 'CPI'], lags=[1, 3, 6, 12, 24])

        data_diff = data_diff[[col for col in data_diff.columns if '_diff' in col]]


        test_data = data_diff[(data_diff.index >= init_date) & (data_diff.index <= end_date)]        
        X_test = test_data.copy().drop('Price_diff', axis=1)


        # Test predictions
        y_pred_diff_test = model.predict(X_test)
        y_pred_test = np.cumsum(y_pred_diff_test) + data['Price'].loc[init_date]


        plot_data_test = pd.DataFrame({
            'Real': data['Price'].loc[init_date:end_date].values,
            'Predicted': y_pred_test
        }, index=X_test.index)


        # Create test plot
        fig_test = px.line(
            plot_data_test,
            x=plot_data_test.index,
            y=['Real', 'Predicted'],
            labels={'x': 'Date', 'y': 'Price'},
            title=f'Predictions vs Real Prices in {state}',
        )
        fig_test.update_layout(
            legend_title_text='Leyend',
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(
            autorange=True,
            title='Date',
            ),
            yaxis=dict(
            autorange=True,
            title='Price',
        ),
        )

        # Convert Plotly figures to HTML divs
        plot_div_test = to_html(
            fig_test,
            include_plotlyjs=True,
            full_html=False,
            config={'responsive': True}
            )
        
        fig_importances = px.bar(
            x=model.feature_importances_,
            y=X_test.columns,
            orientation='h',
            title='Feature Importances',
            labels={'x': 'Importance', 'y': 'Features'}
        )

        fig_importances.update_layout(
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(
                autorange=True,
                title='Importance',
            ),
            yaxis=dict(
                autorange=True, 
                title='Features',
            ),
        )
        plot_div_importances = to_html(
            fig_importances,
            include_plotlyjs=True,
            full_html=False,
            config={'responsive': True} 
        )

        context = {
            'state': state.name,
            'states_list': states_list,
            'plot_div_test': plot_div_test,
            'plot_div_importances': plot_div_importances,

            'metrics': {
                'MAE': mean_absolute_error(data['Price'].loc[init_date:end_date].values, y_pred_test),
                'RMSE': root_mean_squared_error(data['Price'].loc[init_date:end_date].values, y_pred_test),
                'MAPE': round(mean_absolute_percentage_error(data['Price'].loc[init_date:end_date].values, y_pred_test) * 100, 2),
                'MSE': mean_squared_error(data['Price'].loc[init_date:end_date].values, y_pred_test),            
            },
            'init_year': init_year,
            'end_year': end_year,
            'year_range': range(2010, 2025),
            'types': ['Single family', '1 room', '2 rooms', '3 rooms', '4 rooms', '5+ rooms', 'Average'],
            'house_type': house_type_name,
        }
    else:
        plot_div_test = ''
        plot_div_importances = ''
        context = {
        'state': state,
        'states_list': states_list,
        'plot_div_test': plot_div_test,
        'plot_div_importances': plot_div_importances,
        'init_year': init_year,
        'end_year': end_year,
        'year_range': range(2010, 2025),
        'types': ['Single family', '1 room', '2 rooms', '3 rooms', '4 rooms', '5+ rooms', 'Average'],
        'house_type': house_type,
    }
    
    return render(request, 'predict.html', context)



def plot_map(request):
    """
    View function for the map page.
    Shows interactive maps with real and predicted housing prices by state.
    """
    year = request.GET.get('year', '2024')
    bedrooms = request.GET.get('bedrooms', '-1')

    try:
        year = int(year)

        if year < 2000 or year > 2024:
            year = 2024
    except (ValueError, TypeError):
        year = 2024
    
    try:
        bedrooms_mapping = {
            'single_family': 'single_family',
            '1': '1',
            '2': '2', 
            '3': '3',
            '4': '4',
            '5+': '5'
        }

        if bedrooms.isdigit():
            bedrooms = bedrooms
        elif bedrooms in bedrooms_mapping:
            bedrooms_name = bedrooms_mapping[bedrooms]
        else:
            bedrooms = -1
    except (ValueError, TypeError):
        bedrooms = -1
    
    real_map, predicted_map, predicted_prices, real_prices = plot_interactive_map(year, bedrooms)
    
    # Calculate statistics

    def calculate_percentage_difference(real_prices, predicted_prices):
        percentage_diff = ((predicted_prices - real_prices) / real_prices) * 100
        return percentage_diff.round(2)

    percentage_diff = calculate_percentage_difference(real_prices, predicted_prices)
    avg_percentage_diff = round(percentage_diff.abs().mean(), 2)

    # errors
    mae = mean_absolute_error(real_prices, predicted_prices)
    mse = mean_squared_error(real_prices, predicted_prices)
    mape = mean_absolute_percentage_error(real_prices, predicted_prices) * 100
    rmse = root_mean_squared_error(real_prices, predicted_prices)

    # Estados con mayor diferencia porcentual
    top_states_diff = percentage_diff.abs().nlargest(5)
    bottom_states_diff = percentage_diff.abs().nsmallest(5)

    # Convertir mapas a HTML
    real_map_html = real_map._repr_html_()
    predicted_map_html = predicted_map._repr_html_()
    
    # Preparar datos para la vista
    context = {
        'year': year,
        'bedrooms': bedrooms,
        'real_map_html': real_map_html,
        'predicted_map_html': predicted_map_html,
        'year_range': range(2000, 2025),
        'bedroom_options': [
            {'value': 'single_family', 'label': 'Single family'},
            {'value': '1', 'label': '1 bedroom'},
            {'value': '2', 'label': '2 bedrooms'},
            {'value': '3', 'label': '3 bedrooms'},
            {'value': '4', 'label': '4 bedrooms'},
            {'value': '5', 'label': '5+ bedrooms'},
            {'value': '-1', 'label': 'All types (average)'}
        ],
        'statistics': {
            'avg_percentage_diff': avg_percentage_diff,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mse': mse,
            'top_states_diff': top_states_diff,
            'bottom_states_diff': bottom_states_diff,
        }
    }
    
    return render(request, 'map.html', context)

@login_required
def options_view(request):
    """
    View for the admin options page with database and model management tools.
    """
    # Procesar mensajes de estado desde la redirección
    status = request.GET.get('status')
    message = request.GET.get('message')
    
    if status and message:
        if status == 'success':
            messages.success(request, message)
        elif status == 'error':
            messages.error(request, message)
        elif status == 'warning':
            messages.warning(request, message)
        else:
            messages.info(request, message)
    
    return render(request, 'options.html')

@login_required
@require_POST
def retrain_models_view(request):
    """
    View function to retrain all models with new data.
    """
    try:
        results = train_state_models()
        return JsonResponse({
            'status': 'success',
            'message': 'Models have been successfully retrained.',
            'details': results
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Error retraining models: {str(e)}',
        }, status=500)

@login_required
@require_POST
def populate_db_view(request):
    """
    View function for populating the database.
    """
    try:
        results = import_to_db()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Database has been successfully repopulated.',
            'details': results
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Error repopulating database: {str(e)}',
        }, status=500)