# Generated by Django 5.2 on 2025-05-29 14:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('models', '0004_alter_hpi_date'),
    ]

    operations = [
        migrations.AlterField(
            model_name='economicfactorvalue',
            name='value',
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name='housingprice',
            name='price',
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name='hpi',
            name='value',
            field=models.FloatField(default=0.0),
        ),
    ]
