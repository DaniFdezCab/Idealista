# 🏠 Housing Price Predictor - Estados Unidos : IDEALISTA

<div align="center">


*Análisis y predicción avanzada de precios inmobiliarios en Estados Unidos usando Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2+-green.svg)](https://djangoproject.com)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-red.svg)](https://plotly.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-FF6600.svg)](https://xgboost.readthedocs.io)
[![Machine Learning](https://img.shields.io/badge/ML-Time_Series-orange.svg)](https://scikit-learn.org)

</div>

## 📋 Tabla de Contenidos

- [🎯 Características](#-características)
- [📊 Tecnologías](#-tecnologías)
- [🚀 Instalación](#-instalación)
- [💻 Uso](#-uso)
- [📈 Funcionalidades](#-funcionalidades)
- [🗂️ Estructura del Proyecto](#️-estructura-del-proyecto)
- [📊 Análisis de Datos](#-análisis-de-datos)
- [🤝 Contribución](#-contribución)

## 🎯 Características

### **Predicciones Avanzadas**
- **Modelos de Machine Learning** entrenados con datos históricos desde 2000
- **Predicciones por estado** con diferentes tipos de vivienda (1-5 habitaciones)
- **Análisis de importancia** de características económicas

### **Visualización Interactiva**
- **Mapas interactivos** con distribución geográfica de precios
- **Gráficos dinámicos** con Plotly para análisis temporal
- **Comparativas en tiempo real** entre predicciones y datos reales

### **Dashboard Profesional**
- **Interfaz moderna** y responsive
- **Métricas de rendimiento** (RMSE, MAE, R²)
- **Exportación de gráficos** en alta calidad

### **Panel de Administración**
- **Reentrenamiento automático** de modelos
- **Gestión de base de datos** con un clic
- **Sistema de notificaciones** en tiempo real

## 📊 Tecnologías

### **Backend & ML**
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) **Python 3.8+**
- ![Django](https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white) **Django 4.2+**
- ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) **Scikit-learn**
- ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white) **XGBoost**
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) **Pandas & NumPy**

### **Frontend & Visualización**
- ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) **HTML5 & CSS3**
- ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) **JavaScript**
- ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) **Plotly.js**
- ![Leaflet](https://img.shields.io/badge/Leaflet-199900?style=flat&logo=leaflet&logoColor=white) **Leaflet Maps**

### **Base de Datos**
- ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white) **SQLite** (desarrollo)


## 🚀 Instalación

### **Prerrequisitos**
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

### **1. Clonar el Repositorio**
```bash
git clone https://github.com/DaniFdezCab/Idealista.git
cd Idealista
```

### **2. Crear Entorno Virtual**
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### **3. Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### **4. Configurar la Base de Datos**
```bash
cd idealista
python manage.py makemigrations
python manage.py migrate
```

### **5. Crear Superusuario (Opcional)**
```bash
python manage.py createsuperuser
```

### **6. Poblar Base de Datos**
```bash
# Cargar datos iniciales y entrenar modelos
python manage.py runserver
# Navega a http://localhost:8000/admin/options/ y usa "Repopulate Database"
```

## 💻 Uso

### **Iniciar el Servidor**
```bash
cd idealista
python manage.py runserver
```

La aplicación estará disponible en: **http://localhost:8000**

### **Usuarios de Prueba**
- **Admin**: Acceso completo al panel de administración
- **Usuario estándar**: Acceso a predicciones y visualizaciones

### **Navegación Principal**
1. **🏠 Predicciones** - `/` : Formulario de predicción principal
2. **🗺️ Mapas** - `/map/` : Visualización geográfica interactiva  
3. **⚙️ Admin** - `/admin/options/` : Panel de administración

## 📈 Funcionalidades

### **Sistema de Predicciones**


- **Selección de Estado**: 50 estados disponibles
- **Tipo de Vivienda**: 1-5 habitaciones + promedio
- **Rango Temporal**: Desde 2000 hasta 2024
- **Métricas de Precisión**: RMSE, MAE, MSE, MAPE

### **Mapas Interactivos**

- **Visualización en Tiempo Real**: Precios actuales vs predicciones
- **Tooltips Informativos**: Detalles al hacer hover
- **Exportación**: Descargar mapas en alta calidad

### **Análisis Económico**

La aplicación incorpora múltiples indicadores económicos:

- **GDP**: Producto Interior Bruto
- **Tasa de Desempleo**: Indicador de estabilidad laboral
- **HPI**: House Price Index (Índice de Precios de Vivienda)
- **CPI**: Consumer Price Index (Inflación)

## 🗂️ Estructura del Proyecto

```
tfg-daniel-fernandez-idealista/
├── 📁 idealista/                       # Proyecto Django principal
│   ├── 📁 idealista/                   # Configuración del proyecto
│   │   ├── settings.py                 # Configuración Django
│   │   ├── urls.py                     # URLs principales
│   │   └── views.py                    # Vistas principales
│   ├── 📁 models/                      # Aplicación de modelos ML
│   │   ├── 📁 templates/               # Templates HTML
│   │   ├── views.py                    # Lógica de predicciones
│   │   ├── models.py                   # Modelos para la base de datos
│   │   └── urls.py                     # URLs de la app
│   ├── 📁 public/                      # Archivos estáticos y utils
│   │   ├── 📁 datasets/                # Datos originales
│   │   └── 📁 utils/                   # Funciones de ML
│   ├── manage.py                       # Archivo de ejecución principal
│   └── 📁 static/                      # CSS, JS, imágenes
├── requirements.txt                    # Dependencias Python
├── LICENSE                             # Archivo de la licencia MIT
└── README.md                           # Este archivo
```

## 📊 Análisis de Datos

### **Conjuntos de Datos**

El proyecto utiliza datos oficiales de:
- **Federal Housing Finance Agency (FHFA)**
- **World Bank Group**
- **U.S. Bureau of Labor Statistics**
- **Zillow Research Data**

### **Período de Análisis**
- **Datos Históricos**: 2000-2024 (300+ meses)
- **Frecuencia**: Mensual
- **Cobertura**: 50 estados + DC

## 🚨 Solución de Problemas

### **Error de Dependencias**
```bash
# Si hay conflictos de dependencias
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### **Error de Base de Datos**
```bash
# Resetear migraciones
python manage.py migrate --fake-initial
python manage.py migrate
```

### **Error de Modelos ML**
```bash
# Reentrenar modelos desde el panel options
```

## 📄 Licencia

Este proyecto está licenciado bajo la **MIT License** - ver el archivo [LICENSE](LICENSE.txt) para más detalles.

## 👨‍💻 Autor

**Daniel Fernández Caballero**
- Email: [fercabdaniel@gmail.com](mailto:fercabdaniel@gmail.com)
- LinkedIn: [Daniel Fernández](https://www.linkedin.com/in/daniel-fern%C3%A1ndez-caballero-7017a9341/)
- **GitHub**: [@DaniFdezCab](https://github.com/DaniFdezCab) ![Profile](https://github.com/DaniFdezCab.png?size=20) 

## 🙏 Agradecimientos

- **Universidad de Sevilla**: Por el apoyo durante el desarrollo del TFG
- **FHFA & World Bank**: Por proporcionar datos públicos de calidad
- **Comunidad Open Source**: Por las librerías utilizadas
- **Plotly & Leaflet**: Por las herramientas de visualización

---

<div align="center">

**⭐ Si este proyecto te ha sido útil, ¡dale una estrella! ⭐**

*Desarrollado para el análisis del mercado inmobiliario estadounidense*

</div>
