import cv2
import numpy as np
import onnxruntime
import argparse

class Photo2Cartoon:
    def __init__(self):
        self.cartoon_net = onnxruntime.InferenceSession('photo2cartoon_models/minivision_female_photo2cartoon.onnx')
        self.head_seg_net = onnxruntime.InferenceSession('photo2cartoon_models/minivision_head_seg.onnx')
    def inference(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        image = cv2.resize(img, (384, 384))
        blob = np.expand_dims(image, axis=0).astype(np.float32) / 255.0
        output = self.head_seg_net.run(None, {self.head_seg_net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

        output = cv2.resize(output, (srcimg.shape[1], srcimg.shape[0]))
        mask = (output * 255).astype(np.uint8)

        # face_rgba = np.dstack((img, mask))
        # face_rgba = cv2.resize(face_rgba, (256, 256))
        # face = face_rgba[:, :, :3].copy()
        # mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.

        face = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))[:, :, np.newaxis].copy() / 255.

        face = (face * mask + (1 - mask) * 255) / 127.5 - 1
        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)

        # inference
        cartoon = self.cartoon_net.run(['output'], input_feed={'input': face})

        # post-process
        cartoon = np.transpose(cartoon[0][0], (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        return cartoon

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='testimgs/1.jpg', help='input photo path')
    args = parser.parse_args()

    srcimg = cv2.imread(args.imgpath)
    c2p = Photo2Cartoon()
    cartoon = c2p.inference(srcimg)

    cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
    cv2.imshow('srcimg', srcimg)
    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()