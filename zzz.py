'''
from flask import Flask
from flask import send_file

from skimage import data
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.image as matplotimg

import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

from skimage import io

import io as ogIO

import re
import base64

from PIL import Image

from flask import jsonify, request
from io import BytesIO
import base64
import re
import json


app = Flask(__name__)

def saveAndShowImg(name, image):  
  plt.plot(image)   
  plt.savefig('C:/pyth/flasky/images/'+name+'.png')
  return '<img src="/images/'+name+'.png">'

def image_show(image, name, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    
    plt.savefig('C:/pyth/flasky/images/'+name+'.png')
    
    encoded = base64.b64encode(open('C:/pyth/flasky/images/'+name+'.png', "rb").read()).decode()
    return "data:image/png;base64,"+encoded

@app.route('/')
def index():
  return 'Index Page'

@app.route('/blob')
def blob():  
  return saveAndShowImg("blob",data.binary_blobs())


@app.route('/name')
def name():  
  text = matplotimg.imread('images/test3.png')
  text_segmented = text > 50
  return image_show(text_segmented, "text")

@app.route('/images/<image>')
def showImg(image):  
  return saveAndShowImg(image,io.imread('images/'+image))

@app.route('/begin')
def begin():
  with open('img.txt', 'r') as fileread:
    url = fileread.read().replace('\n', '')
    imgstr = re.search(r'base64,(.*)', url).group(1)
    image_bytes = ogIO.BytesIO(base64.b64decode(imgstr))
    im = Image.open(image_bytes)
    npArrayIm = np.array(im)

    shapeX,shapeY,shapeDim  = npArrayIm.shape

    arrReshaped0 = npArrayIm[int((30/100)*shapeX):int((75/100)*shapeX),  int((5/100)*shapeY):int((30/100)*shapeY), 0:int(shapeDim)]
    arr0 = np.dot(arrReshaped0, [0.299, 0.587, 0.114])

    arrReshaped1 = npArrayIm[int((20/100)*shapeX):int((100/100)*shapeX),  int((30/100)*shapeY):int((80/100)*shapeY), 0:int(shapeDim)]
    arr1 = np.dot(arrReshaped1, [0.299, 0.387, 0.314])
    arr1[arr1 >= 115] = 255
    arr1[arr1 < 100] = 0
    
    
    print(arr1)
    print("\n\n\n")
    arr1 = [npArrayIm[:,:,1] , np.zeros(npArrayIm.shape), np.zeros(npArrayIm.shape)]
    print(arr1)
    print("\n\n\n")
    arr2 = [npArrayIm[:,:,2] , np.zeros(npArrayIm.shape), np.zeros(npArrayIm.shape)]
    print(arr2)
    print("\n\n\n")
    print(np.zeros(npArrayIm.shape))
    img1 = Image.fromarray(arr1)
    image_show(img1, "arr1")
    
    img2 = Image.fromarray(arr2)
    image_show(img2, "arr2")
'''
'''
    img0 = Image.fromarray(arr0)
    image_show(img0, "arr0")
    
    img1 = Image.fromarray(arr1)
    image_show(img1, "arr1")

    return 'good'


@app.route('/process_image', methods=['post'])
def process_image():
  
    image_data = re.sub('^data:image/.+;base64,', '', request.form['data'])
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    
    npArrayIm = np.array(im)

    shapeX,shapeY,shapeDim  = npArrayIm.shape

    arrReshaped0 = npArrayIm[int((30/100)*shapeX):int((75/100)*shapeX),  int((5/100)*shapeY):int((30/100)*shapeY), 0:int(shapeDim)]
    arr0 = np.dot(arrReshaped0, [0.299, 0.587, 0.114])

    arrReshaped1 = npArrayIm[int((20/100)*shapeX):int((100/100)*shapeX),  int((30/100)*shapeY):int((80/100)*shapeY), 0:int(shapeDim)]
    arr1 = np.dot(arrReshaped1, [0.299, 0.387, 0.314])
    arr1[arr1 >= 115] = 255
    arr1[arr1 < 100] = 0
    
    
    print(arr1)
    print("\n\n\n")
      '''
    '''
    arr1 = [npArrayIm[:,:,1] , np.zeros(npArrayIm.shape), np.zeros(npArrayIm.shape)]
    print(arr1)
    print("\n\n\n")
    arr2 = [npArrayIm[:,:,2] , np.zeros(npArrayIm.shape), np.zeros(npArrayIm.shape)]
    print(arr2)
    print("\n\n\n")
    print(np.zeros(npArrayIm.shape))
    img1 = Image.fromarray(arr1)
    image_show(img1, "arr1")
    
    img2 = Image.fromarray(arr2)
    image_show(img2, "arr2")
    '''
    '''
    img0 = Image.fromarray(arr0)
    encoded0 = image_show(img0, "arr0")
    
    img1 = Image.fromarray(arr1)
    encoded1 = image_show(img1, "arr1")

    return json.dumps({'result': 'success','photo':encoded0,'text':encoded1}), 200, {'ContentType': 'application/json'}

    

def main():
  
  name = "arr0"
  path = (folderPath+'/images/'+name+'.png').strip()
  #on lit l'image et on le stocke dans une variable
  image = cv2.imread(path)
  print("###############################################")
  print("###############################################")
  print("###############################################")
  print("image uploadée")
  print("reconnaissance en cours ...")
  #on trasnforme l'image en image niveau de gris
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # on stocke l'image dans un fichier temporaire afin d'appliquer l'ocr
  filename = "{}.png".format("temp")
  cv2.imwrite(filename, gray)

  # load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
  #on applique la methode de reconnaissance afin d'obtenir les caractères
  #et on met le résultat dans text
  pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
  #text = pytesseract.image_to_data(Image.open(filename), output_type=Output.STRING, lang="fra")
  #text = pytesseract.image_to_string(Image.open(filename), output_type=Output.STRING, lang="fra")


  txt = pytesseract.image_to_string(image, lang="fra")

  txtArray = txt.split("\n")  

  print(txtArray)
  print("###############################################")
  print("###############################################")

  infoArray = []
  for info in txtArray:
    if ((len(info) == 0)
    or (info.find("dent") != -1) or (info.find("arte") != -1) or (info.find("de la") != -1)
    or (info.find("éno") != -1) or (info.find("rén") != -1)
    or (info.find("Date") != -1) or (info.find("nais") != -1) or (info.find("exe") != -1) or (info.find("Lie") != -1)
    or (info.find("livr") != -1) or (info.find("xpir") != -1) or (info.find("enre") != -1) or (info.find("istr") != -1)
    or (info.find("dres") != -1) or (info.find("res") != -1)):
      print("info ignored")
    else:
      if(info.find(" cm") != -1):
        infos = [ info[0:10], info[10:12], info[12:]]
        infoArray.extend(infos)

      elif(info.find("/") != -1 and len(info) > 12):
        infos = info.split(" ")
        infoArray.extend(infos)

      else:
        infoArray.append(info)


  infoDict = {
    'numero':infoArray[0],
    'prenom':infoArray[1],
    'nom':infoArray[2],
    'dateDeNaissance':infoArray[3],
    'sexe':infoArray[4],
    'taille':infoArray[5],
    'lieuDeNaissance':infoArray[6],
    'dateDeDelivrance':infoArray[7],
    'dateDExpiration':infoArray[8],
    'centreDEnregistrement':infoArray[9],
    'adresseDeDomicile':infoArray[10]
  }
  
  print(infoDict)
  return infoDict


@app.route('/ocr')
def ocr():
  try:
      main()
  except Exception as e:
      print(e.args)
      print(e.__cause__)
  
  return 'DONE'

  
    '''