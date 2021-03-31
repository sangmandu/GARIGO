import uuid
from io import BytesIO

import boto3
from PIL import Image
from django.http import JsonResponse
from requests import Response
from rest_framework import status
from rest_framework.views import APIView
from .mosaic import mosaic_video

from media.PhotoSerializer import PhotoSerializer
from util.MongoManager import createMedia, getImageByPid


class ProfileUploadView(APIView):

    def get(self, request):
        root_url = "https://garigo.s3.ap-northeast-2.amazonaws.com/"
        raw_recog_image_set = getImageByPid('c4a63351-f8b1-45bf-9c7e-ac2934b33d95')
        recog_list = list(raw_recog_image_set.keys())
        recog_images = []
        for name in recog_list:
            img_urls = []
            for img_list in raw_recog_image_set[name]:
                img_url = root_url + img_list["fileUid"]
                img_urls.append(img_url)
            recog_images.append(img_urls)
        input_video = ""
        output_video = mosaic_video(recog_list, recog_images, input_video)
        # print()

        return JsonResponse({'a': 'a'})

    def post(self, request):
        if len(request.FILES) != 0:
            s3_client = boto3.client(
                's3',
                aws_access_key_id="AKIAR26SC7FY7ESCGFCB",
                aws_secret_access_key="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",
            )

            pid = str(uuid.uuid4())

            for file in request.FILES.getlist('photos'):
                name, extend = file.name.split(".")
                fileUid = f'{str(uuid.uuid4())}.{extend}'
                s3_client.put_object(
                    Body=file,
                    Bucket="garigo",
                    Key=fileUid,
                )

                createMedia(pid, name.split("_")[0], name.split("_")[1], fileUid)
            return JsonResponse({'message': pid})
        else:
            return JsonResponse({'message': 'file_none'})
            
