
from django.db import models

class State(models.Model):
    name = models.CharField(max_length=100, unique=True)
    
    def __str__(self):
        return self.name
    

class HousingType(models.Model):
    name = models.CharField(max_length=100, unique=True)
    
    def __str__(self):
        return self.name
    
class HousingPrice(models.Model):
    housing_type = models.ForeignKey(HousingType, on_delete=models.CASCADE)
    state = models.ForeignKey(State, on_delete=models.CASCADE)
    price = models.FloatField(default=0.0)
    currency = models.CharField(max_length=10, default='USD')
    date = models.DateField(default=None)

    def __str__(self):
        return f"{self.housing_type.name} in {self.state.name} on {self.date}: {self.price} {self.currency}"
    
class EconomicFactor(models.Model):
    name = models.CharField(max_length=100, unique=True)
    
    def __str__(self):
        return self.name

class EconomicFactorValue(models.Model):
    factor = models.ForeignKey(EconomicFactor, on_delete=models.CASCADE)
    value = models.FloatField(default=0.0)
    date = models.DateField(default=None)


class HPI(models.Model):
    state = models.ForeignKey(State, on_delete=models.CASCADE)
    factor = models.ForeignKey(EconomicFactor, on_delete=models.CASCADE)
    value = models.FloatField(default=0.0)
    date = models.DateField(default=None)

    def __str__(self):
        return f"{self.factor.name}: {self.value} on {self.date}"