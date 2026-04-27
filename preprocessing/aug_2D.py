'''
Used for 2D data augmentation; performs synchronised processing on pre-processed 2D images and labels (.xml)
Generates new augmented images and corresponding .xml labels
You can use xml2txt.py and txt_conver.py to generate the corresponding .txt, .xml and .json files
You can also use xml2json.py to convert them to .json files
'''
import os
import cv2
import imageio
import numpy as np
from xml import etree
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
import imgaug as ia
from imgaug import augmenters as iaa 
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

base_dir = "./augmentation_test/"

image_2d = base_dir + "9_9_14-2d/"
label_2d = base_dir + "2dlabel_xml/test/"

class CreateAnnotations:
    def __init__(self, flodername, filename, path):
        self.root = Element("annotation")
        self.root.set('verified', 'no')

        child1 = SubElement(self.root, "folder")
        child1.text = flodername

        child2 = SubElement(self.root, "filename")
        child2.text = filename

        child3 = SubElement(self.root, "path")
        child3.text = path
        
        child4 = SubElement(self.root, "checked")
        child4.text = "NO"

        child5 = SubElement(self.root, "source")

        child6 = SubElement(child5, "database")
        child6.text = "Unknown"
    
    def indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def set_size(self, imgshape):
        (height, witdh, channel) = imgshape
        size = SubElement(self.root, "size")
        widthn = SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = SubElement(size, "height")
        heightn.text = str(height)
        channeln = SubElement(size, "depth")
        channeln.text = str(channel)
        
        segmented = SubElement(self.root, "segmented")
        segmented.text = "0"


    def savefile(self, output_dir):
        tree = ET.ElementTree(self.root)
        self.indent(self.root)
        tree.write(output_dir, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self, label, xmin, ymin, xmax, ymax):
        objectn = SubElement(self.root, "object")
        namen = SubElement(objectn, "name")
        namen.text = label
        flag = SubElement(objectn, "flag")
        flag.text = "rectangle"
        pose = SubElement(objectn, "pose")
        pose.text = "Unspecified"
        truncated = SubElement(objectn, "truncated")
        truncated.text = "0"
        difficult = SubElement(objectn, "difficult")
        difficult.text = "0"
        bndbox = SubElement(objectn, "bndbox")
        xminn = SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)


def augmentation2d(output_2d, output_2d_label):
    seq = iaa.Sequential([
        iaa.Flipud(1.0),  # Flip
        iaa.Fliplr(1.0),  # mirror
        iaa.Multiply((0.7, 1.3)),  # Grayscale variation
        iaa.GaussianBlur(sigma=(0, 3.0)),  # Gaussian noise
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
            rotate=(-30, 30)) # Affine transformation
    ])
    seq = seq.to_deterministic()
    
    #2d
    file_names = next(os.walk(label_2d))[2]
    file_names.sort(key=lambda x:(x.split('.')[0]))
    image_num = 0
    for file_name in file_names:
        file_suffix = os.path.splitext(file_name)[-1]
        if file_suffix != ".xml":
            continue
        xml_dir = os.path.join(label_2d, file_name)
        img_dir = os.path.join(image_2d, file_name.replace("xml", "jpg").strip())
        image_num = image_num + 1
        
        tree = ET.parse(xml_dir)
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        image = img[:,:,np.newaxis]

        bbsOnImg = []
        object_num = 0
        for objects in tree.findall("object"):
            object_num = object_num + 1
            object_name = str(objects.find("name").text)
            for bndbox in objects.findall("bndbox"):
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
            bbsOnImg.append(BoundingBox(x1=xmin, x2=xmax, y1=ymin, y2=ymax,label=object_name))
        bbs = BoundingBoxesOnImage(bbsOnImg, shape=image.shape)
        

        for i in range(len(seq)):
            image_aug = seq[i].augment_image(image)
            bb_aug = seq[i].augment_bounding_boxes(bbs)
            for j in range(object_num):
                n_x1 = int(max(1, min(image_aug.shape[1], bb_aug[j].x1)))
                n_y1 = int(max(1, min(image_aug.shape[0], bb_aug[j].y1)))
                n_x2 = int(max(1, min(image_aug.shape[1], bb_aug[j].x2)))
                n_y2 = int(max(1, min(image_aug.shape[0], bb_aug[j].y2)))
                if n_x1 == 1 and n_x1 == n_x2:
                    n_x2 += 1
                if n_y1 == 1 and n_y2 == n_y1:
                    n_y2 += 1
                if n_x1 >= n_x2 or n_y1 >= n_y2:
                    print("error: ", file_name, "-bbox", j)
                bb_aug[j].x1 = n_x1
                bb_aug[j].y1 = n_y1
                bb_aug[j].x2 = n_x2
                bb_aug[j].y2 = n_y2
                bb_aug = bb_aug.clip_out_of_image()
            

            image_aug_show = cv2.cvtColor(image_aug, cv2.COLOR_GRAY2RGB)
            ia.imshow(bb_aug.draw_on_image(image_aug_show, size=1))

            foldername = output_2d
            filename = str(image_num*(i+1))+".jpg"
            path = os.path.join(foldername, filename)
            anno = CreateAnnotations(foldername, filename, path)
            anno.set_size(image_aug.shape)

            for index,bb in enumerate(bb_aug):
                xmin = int(bb.x1)
                ymin = int(bb.y1)
                xmax = int(bb.x2)
                ymax = int(bb.y2)
                label = str(bb.label)
                anno.add_pic_attr(label, xmin, ymin, xmax, ymax)

            anno.savefile(output_2d_label + "{}.xml".format(filename.split(".")[0]))

           # imageio.imsave(path, image_aug)
            cv2.imwrite(path, image_aug)


if __name__ == "__main__": 
    #2d Data augmentation
    output_2d = image_2d + 'augmentation/'
    output_2d_label = label_2d + 'augmentation/'
    if 'augmentation' not in os.listdir(image_2d):   # 文件夹名称不存在才创建
        os.mkdir(output_2d)
    if 'augmentation' not in os.listdir(label_2d): 
        os.mkdir(output_2d_label)
    augmentation2d(output_2d, output_2d_label)

