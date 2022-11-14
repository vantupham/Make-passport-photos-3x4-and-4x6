import cv2, dlib
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image_4x6, crop_image_3x4
from PIL import Image
import os
import glob
from rembg import remove

def remove_bg(filename, outfilename):
    input_f = Image.open(filename)
    output = remove(input_f, alpha_matting=False)
    fill_color = (255, 255, 255)  # your new background color
    output = output.convert("RGBA") 
    if output.mode in ('RGBA', 'LA'):
        background = Image.new(output.mode[:-1], output.size, fill_color)
        background.paste(output, output.split()[-1])  # omit transparency
        output = background
    output.convert("RGB").save(outfilename)

if __name__ == "__main__":

    input_folder = 'inpic' 
    output_folder = 'outpic'
    rmbg_folder = 'rmbg'
    scale = 1 # scale
    margin = 1.8 # margin
    if not os.path.exists(output_folder):
        print("Đã tạo thư mục output")
        os.makedirs(output_folder)
    #remove background
    bg_files = glob.glob(input_folder + "/*")
    for bg_file in bg_files:
        rmbg_tem = 'rmbg/' + os.path.basename(bg_file)
        print(rmbg_tem)
        remove_bg(bg_file, rmbg_tem)    
    # crop
    img_files = glob.glob(rmbg_folder + "/*")
    for img_file in img_files:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        s_height, s_width = height // scale, width // scale
        img = cv2.resize(img, (s_width, s_height))

        dets = detector(img, 1)

        for i, det in enumerate(dets):
            shape = predictor(img, det)
            left_eye = extract_left_eye_center(shape)
            right_eye = extract_right_eye_center(shape)

            M = get_rotation_matrix(left_eye, right_eye)
            rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

            cropped = crop_image_3x4(rotated, det,margin)
            dir1 = '3x4_' + os.path.basename(img_file)
            print(os.path.join(output_folder,  os.path.basename(dir1)))
            cropped.save(os.path.join(output_folder, os.path.basename(dir1)))

            cropped2 = crop_image_4x6(rotated, det,margin)
            dir2 = '4x6_' + os.path.basename(img_file)
            print(os.path.join(output_folder, os.path.basename(dir2)))
            cropped2.save(os.path.join(output_folder, os.path.basename(dir2)))