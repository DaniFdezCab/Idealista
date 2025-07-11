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
from django.contrib import admin
from django.urls import path
from django.urls import include  # Import include to include other URL configurations
from models.views import predict, plot_map
from .views import CustomLoginView, CustomSignupView

namespace = 'idealista'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', include('models.urls', namespace='models')),  # Include the models app URLs
    path('', predict, name='home'),
    path('accounts/signup/', CustomSignupView.as_view(), name='account_signup'),
    path('accounts/login/', CustomLoginView.as_view(), name='account_login'),
    path('accounts/', include('allauth.urls')), 

]
