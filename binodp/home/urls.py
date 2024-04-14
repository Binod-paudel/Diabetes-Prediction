from django.urls import path
from home import views
urlpatterns = [
    path('', views.index, name='home'),
    path('about', views.about, name='about'),
    path('prediction', views.prediction, name='prediction'),
    path('faq', views.faq, name='FAQs'),
    path('contact', views.contact, name='contact'),
    path('accuracy', views.accuracy, name='accuracy'),
    path('prediction/', views.prediction_view, name='prediction'),
    
]