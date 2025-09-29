from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q
import json
import uuid
from .models import Patient, TestSession, EEGData, TestResults, Report

def index(request):
    """Landing page with tool description and main navigation"""
    return render(request, 'demo/index.html')

def patient_registration(request):
    """Patient registration form"""
    if request.method == 'POST':
        try:
            patient = Patient(
                name=request.POST['name'],
                age=int(request.POST['age']),
                gender=request.POST['gender'],
                risk_factors=request.POST.get('risk_factors', ''),
                hearing_condition=request.POST['hearing_condition']
            )
            patient.save()
            messages.success(request, f'Patient registered successfully! Patient ID: {patient.patient_id}')
            return redirect('test_dashboard', patient_id=patient.patient_id)
        except Exception as e:
            messages.error(request, f'Error registering patient: {str(e)}')
    
    return render(request, 'demo/patient_registration.html')

def test_dashboard(request, patient_id):
    """Live test dashboard with stimulus control and EEG visualization"""
    patient = get_object_or_404(Patient, patient_id=patient_id)
    
    # Get or create active test session
    test_session, created = TestSession.objects.get_or_create(
        patient=patient,
        status='in_progress',
        defaults={'status': 'pending'}
    )
    
    context = {
        'patient': patient,
        'test_session': test_session,
        'session_created': created
    }
    return render(request, 'demo/test_dashboard.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def start_test(request, session_id):
    """Start the 40Hz audio test"""
    try:
        test_session = get_object_or_404(TestSession, session_id=session_id)
        test_session.status = 'in_progress'
        test_session.save()
        return JsonResponse({'status': 'success', 'message': 'Test started'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

@csrf_exempt
@require_http_methods(["POST"])
def stop_test(request, session_id):
    """Stop the test and process results"""
    try:
        test_session = get_object_or_404(TestSession, session_id=session_id)
        test_session.status = 'completed'
        test_session.save()
        
        # Generate mock AI results for demonstration
        test_results = TestResults.objects.create(
            test_session=test_session,
            result='normal',  # This would be from actual AI analysis
            confidence_score=0.85,
            power_spectrum_40hz=12.5,
            phase_locking_index=0.78,
            latency_ms=150.0,
            ai_analysis={'analysis': 'Mock AI analysis results'},
            recommendations='Continue regular monitoring.'
        )
        
        return JsonResponse({
            'status': 'success', 
            'message': 'Test completed',
            'results_url': f'/results/{test_results.id}/'
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

@csrf_exempt
def upload_eeg_data(request, session_id):
    """Receive real-time EEG data"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            test_session = get_object_or_404(TestSession, session_id=session_id)
            
            eeg_data = EEGData.objects.create(
                test_session=test_session,
                timestamp=data.get('timestamp', 0),
                channel_data=data.get('channel_data', {}),
                filtered_40hz=data.get('filtered_40hz', {}),
                signal_quality_score=data.get('signal_quality_score', 0.5)
            )
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def results_page(request, result_id):
    """Display test results with AI analysis"""
    test_results = get_object_or_404(TestResults, id=result_id)
    
    context = {
        'test_results': test_results,
        'patient': test_results.test_session.patient,
        'test_session': test_results.test_session
    }
    return render(request, 'demo/results.html', context)

def patient_history(request, patient_id):
    """Display patient test history"""
    patient = get_object_or_404(Patient, patient_id=patient_id)
    test_sessions = TestSession.objects.filter(patient=patient).order_by('-test_date')
    
    paginator = Paginator(test_sessions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'patient': patient,
        'page_obj': page_obj
    }
    return render(request, 'demo/patient_history.html', context)

def reports_list(request):
    """List all generated reports"""
    reports = Report.objects.all().order_by('-generated_at')
    
    # Search functionality
    search_query = request.GET.get('search')
    if search_query:
        reports = reports.filter(
            Q(test_results__test_session__patient__name__icontains=search_query) |
            Q(test_results__test_session__patient__patient_id__icontains=search_query)
        )
    
    paginator = Paginator(reports, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query
    }
    return render(request, 'demo/reports_list.html', context)

# Analysis Views
def eeg_analysis(request):
    """EEG signal analysis page"""
    return render(request, 'demo/eeg_analysis.html')

def hardware_prototype(request):
    """Hardware prototype showcase"""
    return render(request, 'demo/hardware_prototype.html')

def system_details(request):
    """System flow chart and detailed analysis"""
    return render(request, 'demo/system_details.html')

def hardware_design(request):
    """Full hardware design analysis"""
    return render(request, 'demo/hardware_design.html')

def user_guidelines(request):
    """User guidelines and instructions"""
    return render(request, 'demo/user_guidelines.html')

def impact_benefits(request):
    """Impact and benefits page"""
    return render(request, 'demo/impact_benefits.html')

def financial_projection(request):
    """Financial projection analysis"""
    return render(request, 'demo/financial_projection.html')

def upload_data(request):
    """Upload data page for existing EEG files"""
    if request.method == 'POST':
        # Handle file upload logic here
        messages.success(request, 'Data uploaded successfully!')
        return redirect('index')
    
    return render(request, 'demo/upload_data.html')
