from django.shortcuts import render
from allauth.account.views import SignupView
from allauth.account.views import LoginView
from django.urls import reverse_lazy

def home(request):
    """
    View function for the home page of the site.
    """
    return render(request, 'base.html')

class CustomSignupView(SignupView):
    success_url = reverse_lazy('home')

    def get_success_url(self):
        return self.success_url
    

class CustomLoginView(LoginView):
    success_url = reverse_lazy('home')

    def get_success_url(self):
        return self.success_url