import numpy as np

import cv2


def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


def ConvertYUVtoRGB(yuv_planes):
    plane_y = yuv_planes[0]
    plane_u = yuv_planes[1]
    plane_v = yuv_planes[2]
    height = plane_y.shape[0]
    width = plane_y.shape[1]

    # upsample if YV12
    # plane_u = ndimage.zoom(plane_u, 2, order=0)
    # plane_v = ndimage.zoom(plane_v, 2, order=0)
    # alternativelly can perform upsampling with numpy.repeat()
    plane_u = plane_u.repeat(2, axis=1)
    plane_v = plane_v.repeat(2, axis=1)

    # reshape
    plane_y = plane_y.reshape((plane_y.shape[0], plane_y.shape[1], 1))
    plane_u = plane_u.reshape((plane_u.shape[0], plane_u.shape[1], 1))
    plane_v = plane_v.reshape((plane_v.shape[0], plane_v.shape[1], 1))

    # make YUV of shape [height, width, color_plane]
    yuv = np.concatenate((plane_y, plane_u, plane_v), axis=2)

    # according to ITU-R BT.709
    yuv[:, :, 0] = yuv[:, :, 0].clip(16, 235).astype(yuv.dtype) - 16
    yuv[:, :, 1:] = yuv[:, :, 1:].clip(16, 240).astype(yuv.dtype) - 128
    # print(yuv)
    A = np.array([[1.164, 0.000, 1.793], [1.164, -0.213, -0.533], [1.164, 2.112, 0.000]])

    # our result
    rgb = np.dot(yuv, A.T).clip(0, 255).astype('uint8')
    # print(np.dot(yuv, A.T))
    rgb2 = np.copy(rgb)
    # rgb2[:, :, 0] = rgb[:, :, 2]
    # rgb2[:, :, 2] = rgb[:, :, 0]
    # print(rgb2)
    return rgb2


cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)

while True:
    # Grab a single frame of video
    retval, img = cam.read()
    fourcc = decode_fourcc(cam.get(cv2.CAP_PROP_FOURCC))
    if not bool(cam.get(cv2.CAP_PROP_CONVERT_RGB)):
        if fourcc == "MJPG":
            img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        elif fourcc == "YUYV":
            img = img.astype(np.int16)
            img = ConvertYUVtoRGB((img[:, :, 0], img[:, 0::2, 1], img[:, 1::2, 1]))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img = img[:, :, ::-1]
            # y = img[:, :, 0]  #Y
            # u = img[:, ::2, 0]  #U
            # v = img[:, ::2, 1]  #V
            # img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
        else:
            print("unsupported format")
            break
    cv2.imshow('Video', img)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
