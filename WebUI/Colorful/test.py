from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import keras

def model(filename):
    img = load_img("./static/media/" + filename)
    print("coloring:",filename)
    size = img.size
    ori_img_l = rgb2lab(img_to_array(img))[:, :, 0] * 1.0 / 255
    img = img.resize((256, 256))
    x = img_to_array(img)
    test_lab = rgb2lab(1.0 / 255 * x)
    X_test = test_lab[:, :, 0]
    X_test = X_test.reshape((1,) + X_test.shape + (1,))
    model = load_model('./static/media/model_original.h5')
    output = model.predict(X_test)
    keras.backend.clear_session()
    # for i in range(len(output)):
    cur = np.zeros((256, 256, 3))

    cur[:, :, 0] = test_lab[:, :, 0]
    cur[:, :, 1:] = output[0]

    col = array_to_img(lab2rgb(cur))

    col = col.resize(size)

    col_resize = img_to_array(col)
    col_r_l = rgb2lab(1.0 / 255 * col_resize)
    col_r_l[:, :, 0] = ori_img_l
    cor_r_rgb = array_to_img(lab2rgb(col_r_l))

    cor_r_rgb.save("./static/media/res" +filename)
#model("001.jpg")
#model("001.jpg")
# for i in range(1,10):
#     imageName="00"+str(i)+".jpg"
#     model(imageName)