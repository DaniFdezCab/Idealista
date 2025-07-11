"""
URL configuration for idealista project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from .views import predict, plot_map, options_view, populate_db_view, retrain_models_view

app_name = 'models'

urlpatterns = [
    path('state', predict, name='predict'),
    path('country', plot_map, name='plot_map'),
    path('options', options_view, name='options'),
    path('populate', populate_db_view, name='populate'),
    path('retrain', retrain_models_view, name='retrain'),
]
