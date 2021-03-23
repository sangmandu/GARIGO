from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
import face_recognition

from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

# Create arrays of known face encodings and their names
# person name : {[file name, encoding]}
known = {"somin": ["somin.jpg", ], 
#"Jong Kook": ["jongkook.jpg", ]
}
known_face_names, known_face_encodings = [], []
for person_name in known.keys():
    print(known[person_name][0])
    img = face_recognition.load_image_file(known[person_name][0])
    img_encoding = face_recognition.face_encodings(img)[0]
    known_face_names.append(person_name)
    known_face_encodings.append(img_encoding)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
ex_known_face_locations = []
ex_known_face_names = []
ex_unknown_face_locations = []

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/RBF_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', default=640, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--input', default="input.mp4", type = str, help = "input video name")
args = parser.parse_args()

def isClose(loc1, loc2):
    top1, right1, bottom1, left1 = loc1
    top2, right2, bottom2, left2 = loc2
    #print("loc1", top1, right1, bottom1, left1)
    #print("loc2", top2, right2, bottom2, left2)
    cx1, cy1, cx2, cy2 = (right1 + left1)/2, (top1 + bottom1)/2, (right2 + left2)/2, (top2 + bottom2)/2
    x1, x2, y1, y2 = right1 - left1, right2 - left2, bottom1 - top1, bottom2 - top2
    #print(x1, y1, x2, y2)
    distance = (cx1 - cx2)**2 + (cy1 - cy2)**2
    distance = distance**0.5
    size_diff = abs((x1-x2) * (y1-y2))
    print(distance, size_diff)
    if distance < 10 and size_diff < 50:
        return True
    else:
        return False


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg = cfg, phase = 'test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg = cfg, phase = 'test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg = cfg, phase = 'test')
    else:
        print("Don't support network!")
        exit(0)

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    #print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    #input_video = "clip.mp4"
    input_video = args.input
    output_video = "output_"+str(args.vis_thres)+"_"+input_video

    cap = cv2.VideoCapture(input_video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter(output_video, fourcc, 29.97, (width, height))
    frame_number = 0
    ex_hist = []
    methods = {'CORREL' :cv2.HISTCMP_CORREL, 'CHISQR':cv2.HISTCMP_CHISQR, 
           'INTERSECT':cv2.HISTCMP_INTERSECT,
           'BHATTACHARYYA':cv2.HISTCMP_BHATTACHARYYA}

    # testing begin
    while cap.isOpened() and frame_number < 800:
        success, image = cap.read()
        frame_number += 1
        if not success:
            break
        if frame_number < 700:
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [180, 256], [0,180,0,256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        if len(ex_hist) == 0:
            ex_hist = hist
            print("renew")

        similarity = cv2.compareHist(ex_hist, hist, methods['CORREL'])
        print(similarity)
        ex_hist = hist
        #img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_raw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = np.float32(img_raw)

        # testing scale
        target_size = args.long_side
        max_size = args.long_side
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            print("resize!")
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape


        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        #print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        #landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        #landms = landms * scale1 / resize
        #landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        #landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        #landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        #landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        #landms = landms[:args.keep_top_k, :]

        #dets = np.concatenate((dets, landms), axis=1)
        face_locations = []
        face_distances = []
        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                top, bottom, left, right = b[1], b[3], b[0], b[2]

                face_locations.append((top, right, bottom, left))
            if len(face_locations) != 0:
                face_encodings = face_recognition.face_encodings(img_raw, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                    name = "Unknown"

                    
                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                        #print(face_distances)
                        best_match_index = np.argmin(face_distance)
                        name = known_face_names[best_match_index]
                    #else:
                    #    face_distances.append(1)

                    face_names.append(name)
                if len(ex_known_face_locations) != 0:
                    print("------")
                    for i, face_location in enumerate(face_locations):
                        if face_names[i] == "Unknown":
                            for j, ex_known_face_location in enumerate(ex_known_face_locations):
                                if isClose(face_location, ex_known_face_location):
                                    face_names[i] = ex_known_face_names[j]

                ex_known_face_locations = []
                ex_known_face_names = []
                tmp_unknown_face = ex_unknown_face_locations.copy()
                ex_unknown_face_locations = []
                Unknown_face_count = face_names.count('Unknown')

                if similarity < 0.9 or (similarity > 0.9 and (Unknown_face_count == len(tmp_unknown_face) or len(tmp_unknown_face) == 0)):
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # mosaic
                        if name == "Unknown":
                            ex_unknown_face_locations.append((top, right, bottom , left))
                            roi = img_raw[top:bottom, left:right]
                            ry, rx, _ = roi.shape
                            if ry <= 30 or rx <= 30:
                                continue
                            roi = cv2.resize(roi, (rx // 30, ry // 30))
                            roi = cv2.resize(roi, (rx, ry), interpolation=cv2.INTER_AREA)
                            img_raw[top:bottom, left:right] = roi

                        else:
                            ex_known_face_locations.append((top, right, bottom , left))
                            ex_known_face_names.append(name)
                            # Draw a box around the face
                            cv2.rectangle(img_raw, (left, top), (right, bottom), (0, 0, 255), 2)
                            # Draw a label with a name below the face
                            cv2.rectangle(img_raw, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img_raw, name, (left + 6, bottom + 6), font, 1.0, (255, 255, 255), 1)

                else:
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # mosaic
                        if name == "Unknown":
                            continue

                        else:
                            ex_known_face_locations.append((top, right, bottom , left))
                            ex_known_face_names.append(name)
                            # Draw a box around the face
                            cv2.rectangle(img_raw, (left, top), (right, bottom), (0, 0, 255), 2)
                            # Draw a label with a name below the face
                            cv2.rectangle(img_raw, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img_raw, name, (left + 6, bottom + 6), font, 1.0, (255, 255, 255), 1)

                    for (top, right, bottom, left) in tmp_unknown_face:
                        roi = img_raw[top:bottom, left:right]
                        ry, rx, _ = roi.shape
                        if ry <= 30 or rx <= 30:
                            continue
                        roi = cv2.resize(roi, (rx // 30, ry // 30))
                        roi = cv2.resize(roi, (rx, ry), interpolation=cv2.INTER_AREA)
                        img_raw[top:bottom, left:right] = roi
                    ex_unknown_face_locations = tmp_unknown_face.copy()

            cv2.putText(img_raw, str(frame_number), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255,255,255), 1)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
            #cv2.imshow('Video', image)
            print("Writing frame {} / {}".format(frame_number, length))
            output_movie.write(img_raw)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()

