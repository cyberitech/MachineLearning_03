import csv
import random
import sys

from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize


def balance_aus_rain():
    f_in = "data/Australian_Rain/australian_rain.csv"
    f_out = "data/Australian_Rain/balanced_australian_rain.csv"
    data = [line for line in csv.reader(open(f_in))]
    headers=data.pop(0)

    labels = {f[-1] for f in data}

    smallest_label = None
    smallest_occurence = sys.maxsize

    for label in labels:
        count=len([i[-1] for i in data if i[-1]==label])
        if count < smallest_occurence:
            smallest_label = label
            smallest_occurence = count

    balanced_data = [i for i in data if i[-1]==smallest_label]

    for label in labels:
        if label==smallest_label:
            continue
        count=0
        for row in data:
            if count>smallest_occurence:
                break
            if row[-1]==label:
                balanced_data.append(row)
                count+=1

    random.shuffle(balanced_data)
    out_data = [headers]+balanced_data

    with open(f_out,"w",newline="\n")  as f:
        w=csv.writer(f)
        w.writerows(out_data)

def Resize_Images():
    def _proc_image(fpath, xsz, ysz):
        image = imread(fpath)
        image = rgb2gray(image)
        image = resize(image, (xsz, ysz), anti_aliasing=True)
        image = image.flatten()
        return image

    import csv
    x_sz = 30
    y_sz = 40
    dataset_path = "data/Dataset_Handwritten_English/english.csv"
    data = [line for line in csv.reader(open(dataset_path))]
    headers = data.pop(0)
    im_data = []
    label_data = []
    for fpath, label in data:
        im = _proc_image(f"data/Dataset_Handwritten_English/{fpath}", x_sz, y_sz)
        im_data.append(im)
        label_data.append(label)

    with open(f"data/Dataset_Handwritten_English/flattened_images_{x_sz}x{y_sz}.csv", "w", newline="\n") as f:
        writer = csv.writer(f)
        header = [str(i) for i in range(x_sz * y_sz)] + ['label']
        writer.writerow(header)
        for array, label in zip(im_data, label_data):
            row = array.tolist() + [label]
            writer.writerow(row)


Resize_Images()