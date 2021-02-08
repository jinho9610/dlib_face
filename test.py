import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(
    'models/shape_predictor_68_face_landmarks.dat')  # 얼굴 랜드마크 탐지 모델
facerec = dlib.face_recognition_model_v1(
    'models/dlib_face_recognition_resnet_model_v1.dat')


def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []  # shape은 68개의 점

    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)

    for k, d in enumerate(dets):  # 얼굴마다 루프를 돈다 k: 몇번째 얼굴인지, d: 바운더리 박스?
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)  # 68개의 점 반환(landmark detection)

        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)

    return rects, shapes, shapes_np


def encode_faces(img, shapes):  # 128개의 벡터 반환
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)


def np_load(npy_file):
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    loaded_npy = np.load(npy_file)
    loaded_npy = dict(enumerate(loaded_npy.flatten()))[0]
    np.load = np_load_old

    return loaded_npy


if __name__ == '__main__':
    img_paths = {
        'neo': 'img/neo.jpg',
        'trinity': 'img/trinity.jpg',
        'morpheus': 'img/morpheus.jpg',
        'smith': 'img/smith.jpg',
        'jinho': 'img/jinho3.jpg'
    }

    # descs 만드는 코드임 하단은
    # descs = {
    #     'neo': None,
    #     'trinity': None,
    #     'morpheus': None,
    #     'smith': None,
    #     'jinho': None
    # }
    # print(type(descs))

    # for name, img_path in img_paths.items():
    #     img_bgr = cv2.imread(img_path)
    #     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    #     _, img_shapes, _ = find_faces(img_rgb)

    #     descs[name] = encode_faces(img_rgb, img_shapes)[0]  # 전체이미지와 shape을 넣는다
    # print(descs)

    # np.save('img/descs.npy', descs)

    # 얘는 기존에 존재하던 descs.npy 불러오기
    descs = np_load('img/descs.npy')

    img_bgr = cv2.imread('img/yoosung2.jpg')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    rects, shapes, _ = find_faces(img_rgb)
    descrtipors = encode_faces(img_rgb, shapes)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_rgb)

    for i, desc in enumerate(descrtipors):

        found = False

        for name, saved_desc in descs.items():
            # a, b 벡터 사이의 유클리드 거리 구함
            dist = np.linalg.norm([desc] - saved_desc, axis=1)

            if dist < 0.6:
                found = True

                text = ax.text(rects[i][0][0], rects[i][0][1], name,
                               color='b', fontsize=40, fontweight='bold')
                text.set_path_effects([path_effects.Stroke(
                    linewidth=10, foreground='white'), path_effects.Normal()])
                rect = patches.Rectangle(rects[i][0],
                                         rects[i][1][1] - rects[i][0][1],
                                         rects[i][1][0] - rects[i][0][0],
                                         linewidth=2, edgecolor='w', facecolor='none')
                ax.add_patch(rect)

                break

        if not found:
            ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                    color='r', fontsize=20, fontweight='bold')
            rect = patches.Rectangle(rects[i][0],
                                     rects[i][1][1] - rects[i][0][1],
                                     rects[i][1][0] - rects[i][0][0],
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')
    plt.savefig('result/output.png')
    plt.show()
