import uuid
from io import BytesIO

import boto3
from PIL import Image
from django.http import JsonResponse
from requests import Response
from rest_framework import status
from rest_framework.views import APIView

from media.PhotoSerializer import PhotoSerializer


class ProfileUploadView(APIView):

    def get(self, request):
        return JsonResponse({'a': 'a'})

    def post(self, request):
        if len(request.FILES) != 0:
            s3_client = boto3.client(
                's3',
                aws_access_key_id="AKIAR26SC7FYQ4OZSQUM",
                aws_secret_access_key="'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",
            )
            file = request.FILES['photo']
            s3_client.upload_fileobj(
                file,
                "bucket-name",
                "file-name",
                ExtraArgs={
                    "ContentType": file.content_type,
                }
            )
            return JsonResponse({'message': 'success'})
        else:
            return JsonResponse({'message': 'file_none'})
