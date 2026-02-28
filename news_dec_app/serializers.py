from rest_framework import serializers
from .models import NewsPrediction


# 🔹 For validating incoming request
class NewsInputSerializer(serializers.Serializer):
    text = serializers.CharField()


# 🔹 For returning database object
class NewsPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = NewsPrediction
        fields = "__all__"
