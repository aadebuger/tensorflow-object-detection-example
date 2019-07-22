from queue import Queue
from threading import Thread
import redis
import json
import pandas as pd
import arrow
import os
videoport=os.getenv("video")
if videoport is None:
    print('videoport none')
    exit(-1)

boxv=[]
detectworking=True
import arrow
def worker(input_q, output_q):
        global detectworking
        print("worker")
        while True:
            if input_q.full():
                print("full break1")
#                break
            frame,frametime = input_q.get()
            print("frametime=",frametime)
            if detectworking==False:
                print('detectworking')
                continue;
            start1 = time.time()
            box,newframe=detect_objectsbyframe(frame)
            end1= time.time()
            print("detect_objectsbyframe",(end1-start1))
            if box is not None:
                x,y=box
                boxv.append(box)
                print("box1=",x)
                if  x<400.0:
                    if detectworking:
                        print("capture photo")
                        capturephoto(frame,9999)
#                        capturephoto(newframe,10000)
                    print("send to redis")

                    df=pd.DataFrame({
    "x" :x,
    'y':y,
    'start':start1,
    'span':(end1-start1),
    'frametime':frametime
    },index=[0])
                    if os.path.exists('visiondetect.csv'):

                      df.to_csv('visiondetect.csv', mode='a', header=False)
                    else:
                       df.to_csv('visiondetect.csv') 
                    redissendtime = time.time()
                    r.rpush("apriltag", json.dumps({"x":x,"y":y,"span":(end1-start1),"start":start1,"redistime":redissendtime,"frametime":frametime}))
                    detectworking=False
                
#            if bFind:
#                capturephoto(frame,1000)
            output_q.put("hello")
            print('shape =',frame.shape)
            
        print("worker exit")
input_q = Queue(2)  # fps is better if queue is higher but then more lags
output_q = Queue()
for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()
print("test")
import subprocess
import redis
positionv=[]
r = redis.Redis(host='localhost', port=6379, db=0)
def addposition(x,y):
        positionv.append((x,y))
    
def isDetect(line):
    return line.startswith("detection=")
def toPosition(line):
      plotline=line[10:]
      plotv=plotline.split(",")
      return plotv
def apriltag(imagename):
#    imagename='/Users/aadebuger/GEXT/cloud2019/machinevision/data/tz_belt_output20190417_5/40.jpg'
    
#    lines=subprocess.getoutput('/Users/aadebuger/GEXT/github2019/apriltags/xcodebuild/Debug/apriltag_demo {0}'.format(imagename))
    lines=subprocess.getoutput('/home/aadebugergf/Ai/cloud2019/apriltags/build/apriltag_demo {0}'.format(imagename))
    linev=lines.split("\n")
    detectionv=filter(lambda item:isDetect(item),linev)
    positionv=map(lambda item:toPosition(item),detectionv)
    return list(positionv)
import numpy as np 
def getBox(plotv):
            arr2=np.asarray(plotv[1:],float)
            rect=(arr2[6],arr2[7],arr2[4]-arr2[0],arr2[1]-arr2[5])
            return (rect[0]+rect[2]/2,rect[1]+rect[3]/2)

def computeposition(x,y):
    yuandian=(853.5124299065,706.912679109)
    kx = -0.8937383178 
    ky = -0.8729680915
    scarax = yuandian[0] + kx*x 
    scaray = yuandian[1] + ky*y
    return (scarax,scaray)
r = redis.Redis(host='localhost', port=6379, db=0)
import base64
#import cStringIO
import sys
import tempfile
import time
MODEL_BASE = '/Users/aadebuger/GEXT/github2019/models/research'
MODEL_BASE = '/home/aadebugergf/Ai/github2019/models/research'


sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '/object_detection')
sys.path.append(MODEL_BASE + '/slim')


import numpy as np
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf
from utils import label_map_util




PATH_TO_CKPT = '/Users/aadebuger/GEXT/model/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
PATH_TO_LABELS = MODEL_BASE + '/object_detection/data/mscoco_label_map.pbtxt'

content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())


def is_image():
  def _is_image(form, field):
    if not field.data:
      raise ValidationError()
    elif field.data.filename.split('.')[-1].lower() not in extensions:
      raise ValidationError()

  return _is_image





class ObjectDetector(object):

  def __init__(self):
    self.detection_graph = self._build_graph()
    self.sess = tf.Session(graph=self.detection_graph)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)

  def _build_graph(self):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    return detection_graph

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def detect(self, image):
    image_np = self._load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    graph = self.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = self.sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    boxes, scores, classes, num_detections = map(
        np.squeeze, [boxes, scores, classes, num_detections])

    return boxes, scores, classes.astype(int), num_detections
 

def draw_bounding_box_on_image(image, box, color='red', thickness=4):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  ymin, xmin, ymax, xmax = box
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  if (right-left)<90:
        print("too lower",right-left)
        return None
  print("rect,width="+str(right-left)+"height="+str(bottom-top))
  if abs(right-left)<80 or abs(right-left)>120:
          print("width error")
          return None
  if abs(top-bottom)<80 or abs(top-bottom)>120:
          print("hight error")
          return None
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  print('(left, right, top, bottom)',(left, right, top, bottom))
  centerx=left+(right-left)/2
  centery=top+ (bottom-top)/2
  print("center=",centerx,"y=",centery)
  scaraxy=(centerx*4/3,centery*4/3)
  print("scara xy",scaraxy)
  return scaraxy
    
from io import BytesIO
def encode_image(image):
  image_buffer = BytesIO()
  image.save(image_buffer, format='PNG')
  imagevalue = image_buffer.getvalue()
  base64str=base64.b64encode(imagevalue).decode()

  imgstr = 'data:image/png;base64,{:s}'.format(
     base64str )
  return imgstr


def detect_objects(image_path):
  start = time.time()
  
  image = Image.open(image_path).convert('RGB')
  boxes, scores, classes, num_detections = client.detect(image)
  end = time.time()
  print("execute1 time",(end-start)) 
    
  image.thumbnail((480, 480), Image.ANTIALIAS)
  print(num_detections)
  new_images = {}
  for i in range(int(num_detections)):
    
    if scores[i] < 0.9: continue
    print("scres[i",scores[i])
    cls = classes[i]
    if cls not in new_images.keys():
      new_images[cls] = image.copy()
    print("boxes",boxes[i])
    draw_bounding_box_on_image(new_images[cls], boxes[i],
                               thickness=int(scores[i]*10)-4)

  result = {}
#  result['original'] = encode_image(image.copy())
#  print(new_images)
  count = 0 
  for cls, new_image in new_images.items():
    category = client.category_index[cls]['name']
    print(category)
    result[category] = encode_image(new_image)
    new_image.save("{0}.jpg".format(count))
    count=count+1
  end = time.time()
  print("execute2 time",(end-start)) 
  return result

def detect_objectsbyframe(frame):
  start = time.time()
  
#  image = Image.open(image_path).convert('RGB')
  height, width, channels = frame.shape
  print("height",height)
  image= Image.frombytes("RGB", (width,height), frame.tostring())
  boxes, scores, classes, num_detections = client.detect(image)
  end = time.time()
#  print("execute1 time",(end-start)) 
    
  image.thumbnail((480, 480), Image.ANTIALIAS)
#  print(num_detections)
  new_images = {}
  print("test1")
  for i in range(int(num_detections)):
    if scores[i]>0.8:
        print("scres[i",scores[i])
    if scores[i] < 0.90: continue

    cls = classes[i]
    if cls not in new_images.keys():
      new_images[cls] = image.copy()
    print("boxes",boxes[i])
    box=draw_bounding_box_on_image(new_images[cls], boxes[i],
                               thickness=int(scores[i]*10)-4)
    if box is not None:
        print("scarxbox",box)
        return (box,new_images[cls])
    return (None,None)
#  result['original'] = encode_image(image.copy())
#  print(new_images)
  return (None,None)


PATH_TO_CKPT='/Users/aadebuger/GEXT/model/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb'
PATH_TO_CKPT='/home/aadebugergf/Ai/models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb'


PATH_TO_CKPT='/home/aadebugergf/Ai/models/fastrcnnstbmodelrtx2/frozen_inference_graph.pb'
PATH_TO_CKPT='/home/aadebugergf/Ai/models/fastrcnnstbmodelrtx220190703/frozen_inference_graph.pb'

client = ObjectDetector()
import cv2
#cap = cv2.VideoCapture(1)
#/home/aadebugergf/aixgf/tzimagedata/beltvideo/belt_output2019062101.mp4
videoname='/home/aadebugergf/aixgf/tzimagedata/beltvideo/'+'belt_output2019062101.mp4'
cap = cv2.VideoCapture(int(videoport))

ret, frame = cap.read()
img_str = cv2.imencode('.jpg', frame)[1].tostring()
import time
bExit=False
ivalue=0
detectworking=True
def capturephoto(frame,count):
    
    cv2.imwrite("/home/aadebugergf/aixgf/tzimagedata/beltvision/object{0}.jpg".format(count),frame)

def on_button_clicked(b):
    global bExit
    global ivalue
    print("Button clicked.",bExit)
    bExit=True
    ivalue=10
    print(time.time())
def isExit1():
    global bExit
    return bExit
boxv=[]
input_q.queue.clear()
def videooutput():
    global ivalue
    global detectworking
    bFirst=False
    count = 0 
    start = time.time()
    while(True):
        ret, frame = cap.read()
        if frame is None:
            print("frame none")
            break
        if bFirst is False:
            print("put")
            start10=time.time()
            input_q.put((frame,start10))
            bFirst = True
        if output_q.empty():
                pass
        else:
                data=output_q.get()
                end10=time.time()
                print("detect time",(end10-start10))
                bFirst=False
        if detectworking!=False:
              pass

        end = time.time()
        if ( (end-start)>10):
            print("time ok")
            break
while True:
  videooutput()
  detectworking=True
cap.release()
print("capture end")
