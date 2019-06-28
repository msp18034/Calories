from django.shortcuts import render
from django.conf import settings
from django.http import HttpRequest,HttpResponse
import os
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseForbidden
from Colorful import test
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:17:55 2019

@author: Administrator
"""
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.io import imsave
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage import transform
import numpy as np

# def hello(request):
#     return render(request, 'myImg/hello.html',{'before': "001.jpg",'after':"res001.jpg"})



# Create your views here.
def uploadPic(request):
    return render(request,'myImg/uploadPic.html')
def hello(request):
    return render(request,'myImg/index.html')

def uploadHandle(request):
    print("activated!")
    if request.method == "POST":
        f1 = request.FILES['pic1']
        fname = os.path.join(settings.MEDIA_ROOT,f1.name)

        with open(fname,'wb+') as pic:
            for c in f1.chunks():
                pic.write(c)
        pic.close()
        img = load_img("./static/media/"+f1.name)
        test.model(f1.name)
        return render(request, 'myImg/hello.html',{'before': f1.name,'after':"res"+f1.name})
    else:
        return HttpResponse("error")

    #下面这个是直接显示路径
    # pic1 = request.FILES['pic1']
    # picName=os.path.join(settings.MEDIA_ROOT,pic1.name)
    # return HttpResponse(picName)#/Users/liuwei/myImg/static/media/6.png
    #file:///Users/liuwei/myImg/static/media/6.png

