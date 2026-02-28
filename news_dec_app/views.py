from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render

from .serializers import NewsInputSerializer, NewsPredictionSerializer
from .models import NewsPrediction
from ml_model.predict import predict_news

@api_view(["GET"])
def prediction_history(request):
    predictions = NewsPrediction.objects.all().order_by("-created_at")
    serializer = NewsPredictionSerializer(predictions, many=True)
    return Response(serializer.data)


@api_view(["POST"])
def predict_news_view(request):

    input_serializer = NewsInputSerializer(data=request.data)

    if input_serializer.is_valid():

        text = input_serializer.validated_data["text"]

        label, confidence = predict_news(text)

        prediction_obj = NewsPrediction.objects.create(
            text=text,
            prediction=label,
            confidence=confidence
        )

        output_serializer = NewsPredictionSerializer(prediction_obj)

        return Response({
            "data": output_serializer.data,
            "confidence": f"{confidence}%"
        })

    return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# For web page render
def predict_page(request):
    return render(request, "predict.html")
