from flask import Flask, request, jsonify

import numpy as np
import matplotlib.pyplot as plt
import io as io
from PIL import Image
import base64
import re
import json
from flask_cors import CORS

import os

from cv2 import cv2

import pytesseract
from pytesseract import Output


app = Flask(__name__)

# We use flask_cors to avoid CORS problems maybe we don't need it
CORS(app)


#We need the foldername to save the images in the folder
folderPath = os.path.dirname(os.path.realpath(__file__))
print('--- '+folderPath)


def get_infos(image, filename):


  path = (folderPath+'/images/'+filename+'.png').strip()
  #on lit l'image et on le stocke dans une variable
  image = cv2.imread(path)
  print("###############################################")
  print("image uploadée")
  print("reconnaissance en cours ...")
  
  arrReshaped1 = np.array(image)
  gray = np.dot(arrReshaped1, [0.299, 0.387, 0.314])
  gray[gray >= 90] = 255
  gray[gray < 90] = 0

    # on stocke l'image dans un fichier temporaire afin d'appliquer l'ocr
  filename = "{}.png".format("images/temp")
  cv2.imwrite(filename, gray)

  # load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
  #on applique la methode de reconnaissance afin d'obtenir les caractères
  #et on met le résultat dans text
  pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
  #text = pytesseract.image_to_data(Image.open(filename), output_type=Output.STRING, lang="fra")
  #text = pytesseract.image_to_string(Image.open(filename), output_type=Output.STRING, lang="fra")


  txt = pytesseract.image_to_string(gray, lang="fra")

  txtArray = txt.split("\n")  

  print(txtArray)
  print("###############################################")
  print("###############################################")

  infoArray = []
  for info in txtArray:
    if ((len(info) == 0) or (info.find("N ") != -1)or (info.find("Ps") != -1) or (info.find("om") != -1)or (info.isspace()) 
    or (info.find("-") != -1 and len(info) <3) or (info.find("_") != -1 and len(info) <3)
    or (info.find("dent") != -1) or (info.find("arte") != -1) or (info.find("de la") != -1)
    or (info.find("éno") != -1) or (info.find("rén") != -1)
    or (info.find("Date") != -1) or (info.find("nais") != -1) or (info.find("sanc") != -1) or (info.find("exe") != -1) or (info.find("Lie") != -1)
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

#
# We need a function to save an image and return the data URI
#
def image_show(image, name, nrows=1, ncols=1, cmap='gray'):
  fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
  ax.imshow(image, cmap='gray')
  ax.axis('off')
  # We store the image in the server, it will be useful to get the data uri+
  plt.savefig(folderPath+'/images/'+name+'.png')
  
  encoded = base64.b64encode(open(folderPath+'/images/'+name+'.png', "rb").read()).decode()
  return "data:image/png;base64,"+encoded

#
# We need a function to save an image, crop it, save the cropped version and return the data URI of the cropped version of the image
#
def image_show_cropped(image, name, nrows=1, ncols=1, cmap='gray'):
  fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
  ax.imshow(image, cmap='gray')
  ax.axis('off')
  # We store the image in the server, it will be useful to get the data uri+
  plt.savefig(folderPath+'/images/'+name+'.png')
  
  img = cv2.imread(folderPath+'/images/'+name+'.png') 

  encoded0 = crop_photo(img)

  return encoded0


#
# We need a function to crop an image automatically by removing the unnecessary background
#
def crop_photo(image):
  rsz_img = cv2.resize(np.float32(image), None, fx=0.25, fy=0.25) # resize since the image may be huge

  # We get the resized version in a array
  npArrayIm = np.array(rsz_img)

  # We test if the image is already in 1D (grayscale) or still in 3D
  # We make sure to get a 1D grayscale image 
  shapeLength  = len(npArrayIm.shape)
  if shapeLength > 2:
    gray = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY) # convert to grayscale
  else:
    gray = rsz_img
    
  # We are sure that we got a 1D grayscale image so we can crop it now

  # We use opencv threshold to get just the face of the person
  retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

  # find where the face is and make a cropped region
  points = np.argwhere(thresh_gray< 200) # find where the pixels in the face of the person are
  points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
  x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
  x, y, w, h = x-10, y-10, w+20, h+20 # make the box a little bigger
  crop = gray[y:y+h, x:x+w] # create a cropped region of the gray image
  return image_show(crop, "croppedPhoto")

@app.route('/')
def index():
  return 'API'

#
# Test with a post request to http://[[IP ADDRESS]]:5000/process_image
#   with body = { 'data' : [[DATA URI]] }
#
#  for the app we will have    image_data = re.sub('^data:image/.+;base64,', '', request.form['data'])
# and 
#  for Postman we will have    image_data = re.sub('^data:image/.+;base64,', '', request.json['data'])
#
@app.route('/process_image', methods=['post'])
def process_image():

    print("\n")
    print("IMAGE RECEIVED")
    print("\n")
    print(request.json['data'][:20])

    # We collect the data uri from the post request and remove the heading to keep the data only
    # That will be our image
    image_data = re.sub('^data:image/.+;base64,', '', request.json['data'])
    
    #We change the data to an image
    im = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    #We create a numpy array from the image
    npArrayIm = np.array(im)

    # We need to rotate the array(image) by 90' 
    # npArrayIm = np.rot90(npArrayIm)

    # We get the shape from the array ( nbrDeLignes, nbrDeColonnes, nbrDeDimmensions)
    shapeX,shapeY,shapeDim  = npArrayIm.shape

    #We begin the partitionning (segmentation) using intervals of int(n/100):int(m/100)

    # First for the photo but strangely we have Y then X then Dim
    arrReshaped0 = npArrayIm[int((30/100)*shapeX):int((75/100)*shapeX),  int((0/100)*shapeY):int((30/100)*shapeY), 0:int(shapeDim)]

    #We turn this new array image into grayscale (niveau de gris)
    arr0 = np.dot(arrReshaped0, [0.299, 0.587, 0.114])

    # Same for the text part
    arrReshaped1 = npArrayIm[int((20/100)*shapeX):int((100/100)*shapeX),  int((30/100)*shapeY):int((80/100)*shapeY), 0:int(shapeDim)]
    arr1 = np.dot(arrReshaped1, [0.299, 0.387, 0.314])
    
    
    img0 = Image.fromarray(arr0)
    encoded0 = image_show_cropped(img0, "photoCroppedTemp")
    
    # Same for the text
    img1 = Image.fromarray(arrReshaped1)
    image_show(img1, "textWork")
    infosDict = get_infos(img1, "textWork")

    infosDict.update( {'result' : 'success'} )
    infosDict.update( {'photo' : encoded0 } )

    # We send the results back to the app in a json with 'photo' and 'text'
    return json.dumps(infosDict), 200, {'ContentType': 'application/json'}

#
# Test with a get request to http://[[IP ADDRESS]]:5000/process_image_saved
#
@app.route('/process_image_saved', methods=['get'])
def process_image_saved():

    print("\n")
    print("IMAGE RECEIVED")
    print("\n")

    # We collect the data uri from the textImage file and remove the heading to keep the data only
    # That will be our image
    nameOfImage = "textImage.txt"
    
    with open(nameOfImage, 'r') as fileread:
      dataUri = fileread.read().replace('\n', '')
      imgstr = re.search(r'base64,(.*)', dataUri).group(1)
      image_bytes = io.BytesIO(base64.b64decode(imgstr))
      im = Image.open(image_bytes)
    
      #We create a numpy array from the image
      npArrayIm = np.array(im)

      #No need to partion here

      #We turn this new array image into grayscale (niveau de gris)
      arr0 = np.dot(npArrayIm, [0.299, 0.387, 0.314])
      
      # We get the resulting data uri for the photo
      img0 = Image.fromarray(arr0)
      encoded0 = image_show(img0, "textImage")

      # We send the results back to the app in a json with 'photo' and 'text'
      return json.dumps({'result': 'success','textImage':encoded0}), 200, {'ContentType': 'application/json'}


#
# Test with a get request to http://[[IP ADDRESS]]:5000/process_image_saved
#
@app.route('/clean_photo', methods=['get'])
def clean_photo():

    print("\n")
    print("IMAGE RECEIVED")
    print("\n")

    # We collect the data uri from the textImage file and remove the heading to keep the data only
    # That will be our image
    nameOfImage = "photo.txt"
    
    with open(nameOfImage, 'r') as fileread:
      dataUri = fileread.read().replace('\n', '')
      print(dataUri)
      imgstr = re.search(r'base64,(.*)', dataUri).group(1)
      image_bytes = io.BytesIO(base64.b64decode(imgstr))
      im = Image.open(image_bytes)

      encoded0 = crop_photo(im)

      # We send the results back to the app in a json with 'photo' and 'text'
      return json.dumps({'result': 'success'}), 200, {'ContentType': 'application/json'}

    

def main():
  
  name = "pure"
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
  filename = "{}.png".format("tempOcr")
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
