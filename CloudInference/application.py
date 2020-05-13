from flask import Flask, request, Response, jsonify
import mxnet as mx
from os import path

import cv2
import numpy as np
import pickle
import json
from gluoncv import model_zoo, data
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from multiprocessing.pool import ThreadPool
import boto3

class AutoEncoderModel:
    """
    Class used for creating autoencoders with the needed architecture .

    Attributes
    ----------
    autoencoder = the full autoencoder model, containing both the encoder and the decoder
    encoder = the encoder part of the autoencoder, sharing the weights with the autoencoder
    """

    def __init__(self, name, s3_client):
        self.s3_client = s3_client
        self.autoencoder, self.encoder = self.__generate_autoencoder()
        self.__load_autoencoder(name)

    def __generate_autoencoder(self):
        input_img = Input(shape=(64, 64, 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), strides=2, padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        # compiling the models using Adam optimizer and mean squared error as loss
        optimizer = Adam(lr=10 ** -3)
        encoder.compile(optimizer=optimizer, loss='mse')
        autoencoder.compile(optimizer=optimizer, loss='mse')
        print(autoencoder.summary())
        return autoencoder, encoder

    def __load_autoencoder(self,name):
        """Method used for loading the weights from the S3 bucket"""
        # set_session(sess)
        result = self.s3_client.download_file("moldovan.anomalydetectionmodels", name + ".hdf5", "/tmp/"+name + ".hdf5")
        self.autoencoder.load_weights("/tmp/"+name+".hdf5")
        self.autoencoder._make_predict_function()
        self.encoder._make_predict_function()

    def get_encoded_state(self, image):
        """
        Parameters
        ----------
        images - np.array containing the image that need to be encoded

        Returns
        -------
        np.array containing the encoded images, predicted by the encoder.
        """
        global sess
        global graph
        # with graph.as_default():
        #     set_session(sess)
        input = np.expand_dims(image,axis = 0)
        encodings = self.encoder.predict(input)
        return encodings[0]


class ObjectDetector:
    """
    Class used for detecting objects inside a given image

     Parameters
    ----------
    image = mxnet.NDarray - the image for which we want to extract the detections

    Attributes
    ----------
    net : pretrained-model from gluoncv, using ssd architecture trained on the coco dataset.
    threshold : int - the threshold for the detections to be considered positive.
    """

    def __init__(self,image):
        self.net = model_zoo.get_model('ssd_512_mobilenet1.0_coco', pretrained=True, root='/tmp/')
        self.image = image
        self.threshold = 0.5
        self.x_transformed_image, self.img_transformed_image = data.transforms.presets.ssd.transform_test(image, short=512)
        self.class_IDs, self.scores, self.bounding_boxes = self.net(self.x_transformed_image)
        self.bounding_boxes,self.scores,self.class_IDs = self.__clean_bounding_boxes_and_scores(self.bounding_boxes[0].asnumpy(), self.scores[0].asnumpy(),self.class_IDs[0].asnumpy())

    def get_bounding_box_coordinates(self, bounding_boxes, index):
        """
        Parameters
        ----------
        bounding_boxes = np.array(Nx4)list containing all the bounding_boxes coordinates
        index = int - the index of the wanted bounding-box

        Returns
        ----
        A pair of 4 items(c1,l1,c2,l2)describing the top left and bottom right corners of the bounding box
        """
        c1 = int(bounding_boxes[index][0])
        l1 = int(bounding_boxes[index][1])
        c2 = int(bounding_boxes[index][2])
        l2 = int(bounding_boxes[index][3])
        return c1, l1, c2, l2

    def __get_cropped_detections(self,frame):
        cropped_images = []
        for idx,score in enumerate(self.scores):
            try:
                c1, l1, c2, l2 = self.get_bounding_box_coordinates(self.bounding_boxes, idx)
                image = frame[l1:l2, c1:c2]
                image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                cropped_images.append(image)
            except cv2.error as e:
                print('Possible invalid bounding box :')
                print(l1, l2, c1, c2)
                print('Invalid detection! Shape of the invalid image :')
                print(image.shape)

        return np.array(cropped_images)

    def get_object_detections(self):
        """
        Method used for cropping the detected objects from the given image and returning the images
        reshaped to (64x64) and converted to grayscale for further processing by the autoencoder.

        Returns
        ----
        np.array of size (NxWixHix1) where :
        N = number of detections.
        Wi = 64
        Hi = 64
        """
        detections = self.__get_cropped_detections(self.img_transformed_image)
        return detections


    def get_detections_and_cropped_sections(self,frame_d3,frame_p3):
        """
        Method that will return the detections for the image allready present in the ObjectDetector, and using the
        existent bounding-boxes, will also cropp the frames given as parameters.
        :param frame_d3: mxnet.NDArray
        :param frame_p3: mxnet.NDArray
        :return: A pair formed of :
                        - np.array containg detected object appearence of the t frame.
                        - np.array containg cropped image of the t-3 frame of the corresponding detected object
                        - np.array containg cropped image of the t+3 frame of the corresponding detected object
        """
        z, img_d3 = data.transforms.presets.ssd.transform_test(frame_d3, short=512)
        v, img_p3 = data.transforms.presets.ssd.transform_test(frame_p3, short=512)
        detections = self.__get_cropped_detections(self.img_transformed_image)
        cropped_d3 = self.__get_cropped_detections(img_d3)
        cropped_p3 = self.__get_cropped_detections(img_p3)

        return detections,cropped_d3,cropped_p3

    def __clean_bounding_boxes_and_scores(self, bounding_boxes, scores, class_ids):
        """
        Method used for removing the bounding boxes that have a score under the set threshold
        :param bounding_boxes: bounding_boxes np.array with size(Nx4)
        :param scores: scores np.array with size(Nx1)
        :param class_ids: class_ids np.array with size(Nx1)
        :return: Trimmed bounding_boxes as np.array, and trimmed scoresas np.array
        """
        bboxes = []
        new_scores = []
        new_class_ids=[]
        counter = 0
        while scores[counter] > self.threshold:
            c1, l1, c2, l2 = self.get_bounding_box_coordinates(bounding_boxes, counter)
            if c1 < 0 or c2 > self.img_transformed_image.shape[1] or l1 < 0 or l2 > self.img_transformed_image.shape[0]:
                print("Invalid bounding box:", c1,l1,c2,l2)
                counter = counter + 1
                continue
            new_class_ids.append(int(class_ids[counter]))
            bboxes.append(bounding_boxes[counter])
            new_scores.append(scores[counter])
            counter = counter + 1
        return np.array(bboxes),np.array(new_scores),np.array(new_class_ids)

class GradientCalculator:

    def __init__(self) -> None:
        super().__init__()

    def calculate_gradient(self,image):
        # Get x-gradient in "sx"
        sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        # Get y-gradient in "sy"
        sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        # Get square root of sum of squares
        sobel = np.hypot(sx, sy)
        sobel = sobel.astype(np.float32)
        sobel = cv2.normalize(src=sobel, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)
        return sobel
    def calculate_gradient_bulk(self,images):
        gradients = []
        for image in images:
            gradient = self.calculate_gradient(image)
            gradients.append(gradient)
        return np.array(gradients)

class FramePredictor:
    def __init__(self, s3_client) -> None:
        super().__init__()
        self.s3_client = s3_client
        self.autoencoder_images = AutoEncoderModel("image_autoencoder",self.s3_client)
        self.autoencoder_gradients = AutoEncoderModel("gradient_autoencoder",self.s3_client)
        self.num_clusters = 10
        self.svm_models = self.__load_models()
        self.threshold = 2.5

    def __load_models(self):
        models = []

        for cluster in range(self.num_clusters):
            file_name = str(cluster)+".sav"
            result = self.s3_client.download_file("moldovan.anomalydetectionmodels", file_name, path.join("/tmp/",file_name))
            model = pickle.load(open(path.join("/tmp/",file_name), 'rb'))
            models.append(model)
        return models

    def get_inference_score(self,feature_vector):
        scores = [model.decision_function([feature_vector])[0] for model in self.svm_models]
        return max(scores)

def load_frame(arg):
    frame_name = arg[0] 
    s3_client = arg[1]
    frame_path = path.join("/tmp/",frame_name)
    print("Loading frame with name : ", frame_name)
    result = s3_client.download_file("moldovan.inferenceframes", frame_name, frame_path)
    s3 = boto3.resource('s3')
    s3.Object("moldovan.inferenceframes", frame_name).delete()
    # Read the frame
    frame = mx.nd.load(frame_path)[0]
    return frame


def prepare_data_for_CNN( array):
    transformed = []
    for i in range(array.shape[0]):
        transformed.append(array[i] / 255)
    return np.array(transformed)

def get_feature_vectors_and_bounding_boxes(frame_predictor,frame,frame_d3,frame_p3):
    object_detector = ObjectDetector(frame)
    cropped_detections, cropped_d3, cropped_p3 = object_detector.get_detections_and_cropped_sections(frame_d3, frame_p3)
    gradient_calculator = GradientCalculator()
    gradients_d3 = prepare_data_for_CNN(gradient_calculator.calculate_gradient_bulk(cropped_d3))
    gradients_p3 = prepare_data_for_CNN(gradient_calculator.calculate_gradient_bulk(cropped_p3))
    cropped_detections = prepare_data_for_CNN(cropped_detections)
    list_of_feature_vectors = []
    for i in range(cropped_detections.shape[0]):
        apperance_features = frame_predictor.autoencoder_images.get_encoded_state(
            np.resize(cropped_detections[i], (64, 64, 1)))
        motion_features_d3 = frame_predictor.autoencoder_gradients.get_encoded_state(
            np.resize(gradients_d3[i], (64, 64, 1)))
        motion_features_p3 = frame_predictor.autoencoder_gradients.get_encoded_state(
            np.resize(gradients_p3[i], (64, 64, 1)))
        feature_vector = np.concatenate((motion_features_d3.flatten(), apperance_features.flatten(),
                                         motion_features_p3.flatten()))
        list_of_feature_vectors.append(feature_vector)
    return np.array(list_of_feature_vectors), object_detector.bounding_boxes

s3_client = boto3.client("s3")
frame_predictor = FramePredictor(s3_client)
# EB looks for an 'app' callable by default.
app = Flask(__name__)

@app.route('/upload/<file_key>', methods=['POST'])
def lambda_handler(file_key):
    
    # Write request body data into file
    frame_name = file_key
    arguments_tuples = [(frame_name,s3_client),(frame_name+"_d3", s3_client),(frame_name+"_p3", s3_client)]
    pool = ThreadPool(processes=4)
    results = pool.map(load_frame, arguments_tuples)
    frame = results[0]
    frame_d3 = results[1]
    frame_p3 = results[2]

    feature_vectors, bounding_boxes = get_feature_vectors_and_bounding_boxes(frame_predictor,frame,frame_d3,frame_p3)
    frame_score = 0
    boxes = []
    x, img = data.transforms.presets.ssd.transform_test(frame, short=512)
    ratio1 = img.shape[0]/frame.shape[0]
    ratio2 = img.shape[1]/frame.shape[1]

    for idx, vector in enumerate(feature_vectors):
        score = frame_predictor.get_inference_score(vector)
        if score > frame_predictor.threshold:
            c1,l1,c2,l2 = bounding_boxes[idx]
            c1 = int(c1/ratio2)-1
            c2 = int(c2/ratio2)-1
            l1 = int(l1/ratio1)-1
            l2 = int(l2/ratio2)-1
            boxes.append([c1,l1,c2,l2])
        if  score > frame_score:
            frame_score = score
    response = {"statusCode": 200,
            "body" : frame_score,
            "boxes" : boxes}
    return Response(json.dumps(response),  mimetype='application/json')
    
@app.route('/')
def hello_world():
    return '***API test screen***'


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.run(host='0.0.0.0', debug = False)