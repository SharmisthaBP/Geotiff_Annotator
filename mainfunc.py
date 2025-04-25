import cv2
import os
from PIL import Image
import tifffile
import numpy as np

''' ************************************
The following function opens up the image folder
containing tif / tiff files and allows the user to
select objects for each label. The labels generated
can be used in YOLO models for training.
*****************************************'''
# --- CONFIGURATION TO BE MODIFIED ---
image_folder = 'drivename\\foldername\\tiff_imgs'      # folder containing your images
output_folder = 'drivename\\foldername\\tiff_imgs_labels'  # folder to save YOLO labels
#### also change output folder for labels for diff class_id else overwriten
class_id = 0                 # your object class (change if multiple classes)
image_exts = ['.tiff', '.tif']
###############  END OF CONFIGURATION ############

# --- START OF FUNCTION FOR ANNOTATION ------
if not os.path.exists(output_folder):         ##### Generate output folder
    os.makedirs(output_folder)

drawing = False
ix, iy = -1, -1
zoom_level = 1.0
offset_x, offset_y = 0, 0
scroll_speed = 50
boxes = []
current_image = None

### For zooming during object selection ####
def get_scaled_image(img):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * zoom_level), int(h * zoom_level)))
    return resized

#### Mapping original coordinates to zoom ####
def to_original_coords(x, y):
    return int((x + offset_x) / zoom_level), int((y + offset_y) / zoom_level)

##### For drawing shape on the selected object ####
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, boxes, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = to_original_coords(x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = to_original_coords(x, y)
        x_min, y_min = min(ix, x1), min(iy, y1)
        x_max, y_max = max(ix, x1), max(iy, y1)
        boxes.append((x_min, y_min, x_max, y_max))
        cv2.rectangle(current_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

##### For saving the annotated image for direct use by YOLO model ####
def save_yolo_format(image_name, boxes, width, height):
    file_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + ".txt")
    with open(file_path, 'w') as f:
        for (x_min, y_min, x_max, y_max) in boxes:
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# --- MAIN LOOP for annotator ---
image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in image_exts]
for img_name in image_files:
    boxes = []
    img_path = os.path.join(image_folder, img_name)
    ######## Convert to PNG as openCV cannot handle 32-bit tiffs --------
    '''
    # Works for single band tiffs ###############
    img = Image.open(img_path)
    # Normalize to 8-bit if it's 16 or 32-bit
    if img.mode != "RGB":
        img = img.convert("RGB")
    output_path = os.path.join(image_folder, img_name.replace('.tif', '.png'))
    img.save(output_path)
    # End of single band tiffs #############
    '''
    ##############    OR  ###################################
    # Works for multiband tiffs  #########
    img = tifffile.imread(img_path)
    # Normalize to 0-255 and convert to 8-bit
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    if img_norm.dtype != np.uint8:
        img_norm = img_norm.astype(np.uint8)
    # Handle grayscale, single channel
    if len(img_norm.shape) == 2:
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    # Save as PNG
    output_path = os.path.join(image_folder, os.path.splitext(img_name)[0] + ".png")
    cv2.imwrite(output_path, img_norm)
    # End of multiband tiffs
    ############  End of Conversion -------------
    image = cv2.imread(output_path)
    img_name = img_name.replace('.tif', '.png')
    if image is None:
        continue

    h, w = image.shape[:2]
    boxes = []
    zoom_level = 1.0
    offset_x, offset_y = 0, 0

    print(f"\nLabeling: {img_name}")
    while True:
        display_image = get_scaled_image(image.copy())
        # Step 1: Resize image according to zoom
        resized = cv2.resize(image, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)
        h_resized, w_resized = resized.shape[:2]

        # Step 2: Ensure offset stays within bounds
        viewport_w, viewport_h = 800, 600  # set your display window size
        offset_x = min(max(offset_x, 0), w_resized - viewport_w)
        offset_y = min(max(offset_y, 0), h_resized - viewport_h)

        # Step 3: Crop to current viewport
        view = resized[offset_y:offset_y + viewport_h, offset_x:offset_x + viewport_w].copy()

        # Step 4: Draw all boxes shifted by offset
        for (x1, y1, x2, y2) in boxes:
            x1_view = int(x1 * zoom_level) - offset_x
            y1_view = int(y1 * zoom_level) - offset_y
            x2_view = int(x2 * zoom_level) - offset_x
            y2_view = int(y2 * zoom_level) - offset_y

            if 0 <= x1_view < viewport_w and 0 <= y1_view < viewport_h:
                cv2.rectangle(view, (x1_view, y1_view), (x2_view, y2_view), (0, 255, 0), 2)


        cv2.imshow('TIF_Annotator', view)
        cv2.setMouseCallback('TIF_Annotator', draw_rectangle)
        key = cv2.waitKey(30) & 0xFF
        print("key =",key)
        if key == ord('r'):  # reset boxes
            boxes = []
        elif key == ord('v'):  # save the labels
            save_yolo_format(img_name, boxes, w, h)
            print("Saved:", img_name)
            break
        elif key == ord('+') or key == ord('='): ### zoom in
            print('zoom +')
            zoom_level = min(5.0, zoom_level + 0.1)
        elif key == ord('-') or key == ord('_'):  #### zoom out
            print('zoom -')
            zoom_level = max(0.2, zoom_level - 0.1)
        elif key == 81 or key == ord('w'):  # Left / West movemnet
            print('pan left')
            offset_x = max(0, offset_x - scroll_speed)
        elif key == 82 or key == ord('n'):  # Up / North movement
            offset_y = max(0, offset_y - scroll_speed)
        elif key == 83 or key == ord('e'):  # Right / East movement
            offset_x += scroll_speed
        elif key == 84 or key == ord('s'):  # Down / South movement
            offset_y += scroll_speed
        elif key == ord('q'):  # quit
            cv2.destroyAllWindows()
            exit()
        elif key == ord('x'):  # next image without saving
            break

cv2.destroyAllWindows()
