from django.urls import path
from . import views

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),
    path('demo/', views.index, name='demo_index'),  # Keep existing demo URL
    
    # Patient management and testing
    path('register/', views.patient_registration, name='patient_registration'),
    path('dashboard/<str:patient_id>/', views.test_dashboard, name='test_dashboard'),
    path('upload/', views.upload_data, name='upload_data'),
    
    # Test control APIs
    path('start-test/<uuid:session_id>/', views.start_test, name='start_test'),
    path('stop-test/<uuid:session_id>/', views.stop_test, name='stop_test'),
    path('upload-eeg-data/<uuid:session_id>/', views.upload_eeg_data, name='upload_eeg_data'),
    
    # Results and reports
    path('results/<int:result_id>/', views.results_page, name='results_page'),
    path('history/<str:patient_id>/', views.patient_history, name='patient_history'),
    path('reports/', views.reports_list, name='reports_list'),
    
    # Analysis and technical pages
    path('analysis/', views.eeg_analysis, name='eeg_analysis'),
    path('hardware/', views.hardware_prototype, name='hardware_prototype'),
    path('system/', views.system_details, name='system_details'),
    path('design/', views.hardware_design, name='hardware_design'),
    
    # Information and guidelines
    path('guidelines/', views.user_guidelines, name='user_guidelines'),
    path('impact/', views.impact_benefits, name='impact_benefits'),
    path('financial/', views.financial_projection, name='financial_projection'),
]