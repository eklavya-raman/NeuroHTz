import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Set the Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NeuroHTz.settings')

import django
from django.core.wsgi import get_wsgi_application

# Initialize Django
django.setup()

# This is the WSGI application for Vercel
app = get_wsgi_application()