'''
Convert the labels in the 2d.txt file to .xml and .json formats
Convert the files according to the selected directories for the training, validation and test sets
'''

import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
from PIL import Image
import json
import sys
from shutil import copyfile

txt_base_dir = "./2D_data/label/"
im_base_dir = "./2D_data/image/"
PRE_DEFINE_CATEGORIES = {"diffuse":1, "core":2, "CAA":3}



class Xml_make(object):
    def __init__(self):
        super().__init__()
    def __indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.__indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _imageinfo(self, list_top):
        annotation_root = ET.Element("annotation")
        annotation_root.set("verified", "no")
        tree = ET.ElementTree(annotation_root)

        '''
        0:xml_savepath 1:folder,2:filename,3:path
        4:checked,5:width,6:height,7:depth
        '''
        
        folder_element = ET.Element("folder")
        folder_element.text = list_top[1]
        annotation_root.append(folder_element)
        filename_element = ET.Element("filename")
        filename_element.text = list_top[2]
        annotation_root.append(filename_element)
        path_element = ET.Element("path")
        path_element.text = list_top[3]
        annotation_root.append(path_element)
        checked_element = ET.Element("checked")
        checked_element.text = list_top[4]
        annotation_root.append(checked_element)
        source_element = ET.Element("source")
        database_element = SubElement(source_element, "database")
        database_element.text = "Unknown"
        annotation_root.append(source_element)
        size_element = ET.Element("size")
        width_element = SubElement(size_element, "width")
        width_element.text = str(list_top[5])
        height_element = SubElement(size_element, "height")
        height_element.text = str(list_top[6])
        depth_element = SubElement(size_element, "depth")
        depth_element.text = str(list_top[7])
        annotation_root.append(size_element)
        segmented_person_element = ET.Element("segmented")
        segmented_person_element.text = "0"
        annotation_root.append(segmented_person_element)
        return tree, annotation_root

    def _bndbox(self, annotation_root, list_bndbox):
        for i in range(0, len(list_bndbox), 9):
            object_element = ET.Element("object")
            name_element = SubElement(object_element, "name")
            name_element.text = list_bndbox[i]
            flag_element = SubElement(object_element, "flag")
            flag_element.text = list_bndbox[i + 1]
            pose_element = SubElement(object_element, "pose")
            pose_element.text = list_bndbox[i + 2]
            truncated_element = SubElement(object_element, "truncated")
            truncated_element.text = list_bndbox[i + 3]
            difficult_element = SubElement(object_element, "difficult")
            difficult_element.text = list_bndbox[i + 4]
            bndbox_element = SubElement(object_element, "bndbox")
            xmin_element = SubElement(bndbox_element, "xmin")
            xmin_element.text = str(list_bndbox[i + 5])
            ymin_element = SubElement(bndbox_element, "ymin")
            ymin_element.text = str(list_bndbox[i + 6])
            xmax_element = SubElement(bndbox_element, "xmax")
            xmax_element.text = str(list_bndbox[i + 7])
            ymax_element = SubElement(bndbox_element, "ymax")
            ymax_element.text = str(list_bndbox[i + 8])
            annotation_root.append(object_element)
        return annotation_root

    def txt_to_xml(self, list_top, list_bndbox):
        tree, annotation_root = self._imageinfo(list_top)
        annotation_root = self._bndbox(annotation_root, list_bndbox)
        self.__indent(annotation_root)      
        tree.write(list_top[0], encoding="utf-8", xml_declaration=True)


def txt_conver(source_path, im_dir, xml_dir, json_dir):
    COUNT = 0
    json_dict = {"images":[],"categories":[]}
    for folder_path_tuple, folder_name_list, file_name_list in os.walk(source_path):
        # file_name_list.sort(key=lambda x:int(x.split('.')[0]))
        for file_name in file_name_list:
            file_suffix = os.path.splitext(file_name)[-1]
            #if file_suffix != ".jpg":
            #    continue
            if file_suffix != ".txt":
                continue
            list_top = []
            list_bndbox = []
            json_list_1=[]
            txt_path = os.path.join(folder_path_tuple, file_name)
            xml_save_path = os.path.join(xml_dir, file_name.replace(file_suffix, ".xml")) 
            filename = os.path.splitext(file_name)[0] + ".jpg"
            # im_id = int(os.path.splitext(file_name)[0])
            im_id = os.path.splitext(file_name)[0]
            im_path = im_dir + "/" + filename
            checked = "NO"
            #im = Image.open(im_path)
            #width = str(im.size[0])
            #height = str(im.size[1])
            width = str(512)
            height = str(512)
            depth = "1"
            flag = "rectangle"
            pose = "Unspecified"
            truncated = "0"
            difficult = "0"
            list_top.extend([xml_save_path, folder_path_tuple, filename, im_path, checked, width, height, depth])
            
            bnd_id = 1
            for line in open(txt_path, "r"):
                line = line.strip()
                info = line.split(" ")
                cls_name = info[7]
                w = int(info[4])
                h = int(info[5])
                xmin = int(info[0])
                ymin = int(info[1])
                xmax = int(info[2])
                ymax = int(info[3])
                if cls_name not in categories:
                    new_id = len(categories)
                    categories[cls_name] = new_id+1
                category_id = categories[cls_name]
                #x_center = xmin + w/2
                #y_center = ymin + h/2
                list_bndbox.extend([cls_name, flag, pose, truncated, difficult,
                                    str(xmin), str(ymin), str(xmax), str(ymax)])
                ann = {"area": w*h, "image_id":im_id, "bbox":[xmin, ymin, w, h], 
                       "category_id": category_id, "bbox_id": bnd_id}
                json_list_1.append(ann)
                bnd_id = bnd_id + 1
                
            Xml_make().txt_to_xml(list_top, list_bndbox)  #生成xml文件
            
            image = {"file_name":filename, "width":width, "height":height, "image_id":im_id, "annotations":json_list_1 }
            json_dict["images"].append(image)
            
            COUNT += 1
            print(COUNT, xml_save_path)

    for c_name,c_id in categories.items():
        cat = { "id": c_id, "name": c_name}
        json_dict["categories"].append(cat)
    json_fp = open(json_dir, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print("json file: ",json_dir)



if __name__ == "__main__":
    categories = PRE_DEFINE_CATEGORIES
    folder_list= ["train","val","test"]
    for i in range(3):
        folderName = folder_list[i]
        source_path = txt_base_dir + folderName

        # if folderName not in os.listdir(txt_base_dir):
        #     os.mkdir(source_path)

        im_dir = im_base_dir + folderName

        xml_dir = source_path + "/xml" 
        if "xml" not in os.listdir(source_path):
            os.mkdir(xml_dir)

        json_dir = source_path + "/instances_" + folderName + ".json"
        print("process dir: ",source_path) 
        # build_data(folderName)
        txt_conver(source_path, im_dir, xml_dir, json_dir)

