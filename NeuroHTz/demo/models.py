from django.db import models
from django.utils import timezone
import uuid

class Patient(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    
    HEARING_CHOICES = [
        ('normal', 'Normal'),
        ('mild_loss', 'Mild Hearing Loss'),
        ('moderate_loss', 'Moderate Hearing Loss'),
        ('severe_loss', 'Severe Hearing Loss'),
        ('profound_loss', 'Profound Hearing Loss'),
    ]
    
    patient_id = models.CharField(max_length=20, unique=True, primary_key=True)
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    risk_factors = models.TextField(blank=True, help_text="Enter risk factors separated by commas")
    hearing_condition = models.CharField(max_length=20, choices=HEARING_CHOICES, default='normal')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    def save(self, *args, **kwargs):
        if not self.patient_id:
            # Auto-generate patient ID
            self.patient_id = f"NH{str(uuid.uuid4())[:8].upper()}"
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.patient_id} - {self.name}"

class TestSession(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    session_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    test_date = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    duration_seconds = models.IntegerField(null=True, blank=True)
    signal_quality = models.CharField(max_length=10, default='unknown')  # good, poor, artifact
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"Session {self.session_id} - {self.patient.name}"

class EEGData(models.Model):
    test_session = models.ForeignKey(TestSession, on_delete=models.CASCADE)
    timestamp = models.FloatField()  # Time in seconds from test start
    channel_data = models.JSONField()  # Store channel data as JSON
    filtered_40hz = models.JSONField()  # Filtered 40Hz activity
    signal_quality_score = models.FloatField()  # Quality score 0-1
    
    class Meta:
        ordering = ['timestamp']

class TestResults(models.Model):
    RESULT_CHOICES = [
        ('normal', 'Normal'),
        ('abnormal', 'Abnormal'),
        ('inconclusive', 'Inconclusive'),
    ]
    
    test_session = models.OneToOneField(TestSession, on_delete=models.CASCADE)
    result = models.CharField(max_length=20, choices=RESULT_CHOICES)
    confidence_score = models.FloatField()  # 0-1 confidence
    power_spectrum_40hz = models.FloatField()
    phase_locking_index = models.FloatField()
    latency_ms = models.FloatField()
    ai_analysis = models.JSONField()  # Store detailed AI analysis
    recommendations = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"Results for {self.test_session.patient.name} - {self.result}"

class Report(models.Model):
    test_results = models.OneToOneField(TestResults, on_delete=models.CASCADE)
    pdf_file = models.FileField(upload_to='reports/', null=True, blank=True)
    generated_at = models.DateTimeField(default=timezone.now)
    shared_with = models.TextField(blank=True, help_text="List of healthcare providers/institutions shared with")
    
    def __str__(self):
        return f"Report for {self.test_results.test_session.patient.name}"
