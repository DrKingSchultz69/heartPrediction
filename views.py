from django.shortcuts import render
import joblib
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# Load the trained SVM model
model = joblib.load('api/best_svm_model.pkl')
@api_view(['POST'])
def predict(request):
    # Check if request data is provided
    if request.method == 'POST':
        try:
            # Extract features from the request data
            features = request.data['features']
            # Make predictions
            prediction = model.predict([features])
            return Response({'prediction': int(prediction[0])}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
