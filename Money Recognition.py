# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os.path
from firebase.firebase import FirebaseApplication
from pyrebase import pyrebase
import firebase
import datetime
import time
start = time.time()

config = {

  "apiKey": "AIzaSyCPEpMI3LnhGQmCRhiF50cPwYdVAwjpMT4",
  "authDomain": "sddfb-bb4d9.firebaseapp.com",
  "databaseURL": "https://sddfb-bb4d9.firebaseio.com",
  "storageBucket": "sddfb-bb4d9.appspot.com"

}
firebase1 = pyrebase.initialize_app(config)
storage = firebase1.storage()
app = FirebaseApplication('https://sddfb-bb4d9.firebaseio.com', authentication=None)

st = storage.child("serialImages/serial1.jpg")  #download the image from storage
st.download("sohanadi.png")




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args(['-mNumbersModel']))

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

image_path_list = []


#IF READING FROM DISK
# imageDir = "Money/Done_Papers/Experiment_Papers"  # specify your path here
# valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]  # specify your vald extensions here
# valid_image_extensions = [item.lower() for item in valid_image_extensions]
# for file in os.listdir(imageDir):
#     extension = os.path.splitext(file)[1]
#     if extension.lower() not in valid_image_extensions:
#         continue
#     image_path_list.append(os.path.join(imageDir, file))

image_path_list.append("sohanadi.png")
print image_path_list

#Accuracy Function
NumberOfCharacters=image_path_list.__len__()*7
WrongCharacters=0

# load the image
for MoneyBill in range(image_path_list.__len__()):
    image = cv2.imread(image_path_list[MoneyBill])
    orig = image.copy()
    x = 15
    y = 0
    h = 200
    w = 45
    MoneySerial = []
    for counter in range(7):
        imgToCrop = orig.copy()
        image = imgToCrop[y:y + h, x:x + w]
        cv2.imwrite(str(counter)+'.jpg',image)
        # pre-process the image for classification
        image = cv2.resize(image, (28, 28))
        image = image.astype("float") / 255.0
        cv2.imshow("astype",image)
        cv2.waitKey(0)

        image = img_to_array(image)
        cv2.imshow("imgtoarray",image)
        cv2.waitKey(0)

        image = np.expand_dims(image, axis=0)
        # cv2.imshow("expand",image)
        # cv2.waitKey(0)

        # classify the input image
        (n0, n1, n2, n3, n4, n5, n6, n7, n8, n9) = model.predict(image)[0]




        # build the label
        if (max(n0,n1,n2,n3,n4,n5,n6,n7,n8,n9)==n0):
            MoneySerial.append("0")
        elif (max(n0,n1,n2,n3,n4,n5,n6,n7,n8,n9)==n1):
            MoneySerial.append("1")
        elif (max(n0, n1, n2, n3, n4, n5, n6, n7, n8, n9) == n2):
            MoneySerial.append("2")
        elif (max(n0, n1, n2, n3, n4, n5, n6, n7, n8, n9) == n3):
            MoneySerial.append("3")
        elif (max(n0, n1, n2, n3, n4, n5, n6, n7, n8, n9) == n4):
            MoneySerial.append("4")
        elif (max(n0, n1, n2, n3, n4, n5, n6, n7, n8, n9) == n5):
            MoneySerial.append("5")
        elif (max(n0, n1, n2, n3, n4, n5, n6, n7, n8, n9) == n6):
            MoneySerial.append("6")
        elif (max(n0, n1, n2, n3, n4, n5, n6, n7, n8, n9) == n7):
            MoneySerial.append("7")
        elif (max(n0, n1, n2, n3, n4, n5, n6, n7, n8, n9) == n8):
            MoneySerial.append("8")
        else:
            MoneySerial.append("9")

        x+=48



    # show the output image
    output = imutils.resize(orig, width=400)
    # WrittenText=''.join(MoneySerial)
    # cv2.putText(output, WrittenText, (20, 85),  cv2.FONT_HERSHEY_SIMPLEX,
    # 	0.7, (0, 255, 0), 2)
    print MoneySerial
    ToBeWritten = ''.join(MoneySerial)

    result = app.patch('/PaperSerial/', {'Serial Number': ToBeWritten})
    end = time.time()
    print("Time Taken: "+str(datetime.timedelta(seconds=(end-start))))
    print("Time Taken: "+str(end - start))
    cv2.imshow("Output", output)
    cv2.waitKey(0)




    # cv2.waitKey(25)
    # entry = raw_input("Enter number")
    # WrongCharacters+=int(entry)

# print "numbers of Papers: "+str(NumberOfCharacters/7)
# print "numbers of Characters: "+str(NumberOfCharacters)
# print "numbers of Right Characters: "+str(NumberOfCharacters-WrongCharacters)
# print "numbers of Wrong Characters: "+str(WrongCharacters)
# print "Wrong Percentage: "+str((float(WrongCharacters)/NumberOfCharacters)*100)+" %"
# print "Right Percentage: "+str(100-((float(WrongCharacters)/NumberOfCharacters)*100))+" %"