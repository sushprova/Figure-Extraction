# This file is modified from Sush's test.py (version July 8 or earlier)
# This file has fixed a number of bugs with respect white space/margin trimming,
# and figure area detections.  
#  
# TODO:  1) some more bugs to be fixed - see notes
#        2) Currently, column histograms are looked at first, then with row histograms.  Todo including one more
#           histogram analysis in the opposite direction before setting on the final region.
#        3) (see 2 above) Potentially both orders - column-first-then-row-then-column-again vs
#           row-first-then-column-then-row-again - should be considered, with results compared to each
#           other to select the "better" one.
#        4) after regions are analyzed, potentially find the caption block of any figure/table if not
#           already included.
#        5) manage constants when things are working and tested more
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from skimage.feature import local_binary_pattern
from scipy import stats
import os
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 
def read_and_preprocess_image(image_path):
######passed in the original image path      does color thresholding and returns img
    img = cv2.imread(image_path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# TODO hard coded 230, 255(white) 
# anything higher than 230 will turn into white and lower than 230 will be black.
    ret, thresh = cv2.threshold(imgray, 230, 255, cv2.THRESH_BINARY)

#     plt.figure()
#     plt.imshow(thresh, cmap='gray')
#     plt.show()
    return thresh, img


#not being used right now
def remove_header_footer(image_path):
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # Sort contours based on their vertical positions (top to bottom)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])
    #
    header_boxes = []
    footer_boxes = []
    #
    # Determine the spacing threshold for grouping lines
    spacing_threshold = 10  # Adjust this value based on the typical line spacing in your images
    #
    # Group top bounding boxes for header
    for box in bounding_boxes:
        if not header_boxes or box[1] <= header_boxes[-1][1] + header_boxes[-1][3] + spacing_threshold:
            header_boxes.append(box)
        else:
            break
    #
    # Group bottom bounding boxes for footer
    for box in reversed(bounding_boxes):
        if not footer_boxes or box[1] >= footer_boxes[-1][1] - spacing_threshold:
            footer_boxes.append(box)
        else:
            break
    #
    # Calculate header height
    header_height = max([box[1] + box[3] for box in header_boxes]) if header_boxes else 0
    #
    # Calculate footer height
    footer_height = min([box[1] for box in footer_boxes]) if footer_boxes else img.shape[0]
    #
    # Crop the image
    cropped_image = img[header_height:footer_height, :]
    #
    # Convert the image from BGR to RGB (for matplotlib)
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    #
    # Display the image using matplotlib
    plt.imshow(cropped_image_rgb)
    plt.axis('off')  # Hide the axes
    plt.show()
    #
    # cv2.imwrite('cropped_image.png', cropped_image_rgb)
    return cropped_image_rgb

def trim_white_margins_lr(pixel_array):
    #####passed on "thresh"             takes out only the left and right margin
    col_sums = np.sum(pixel_array, axis=0)
    num_rows = pixel_array.shape[0]
    white_col_sum = 255 * num_rows
    #
    start_index = 0
    # The condition below is too strict - we need to give some slack for the
    # columns to be non-perfect and still "white" largely
    # while start_index < len(col_sums) and col_sums[start_index] == white_col_sum:
    roughly = 0.99
    while start_index < len(col_sums) and col_sums[start_index] >= white_col_sum*roughly:
        start_index += 1
    #
    end_index = len(col_sums) - 1
    # while end_index >= 0 and col_sums[end_index] == white_col_sum:
    while end_index >= 0 and col_sums[end_index] >= white_col_sum * roughly:
        end_index -= 1
    #
    col_trimmed_array = pixel_array[:, start_index:end_index+1]
    #
    plt.figure()
    plt.imshow(col_trimmed_array, cmap='gray')
    #plt.show()
    #
    return col_trimmed_array, start_index


def col_sum(array):
    return np.sum(array, axis=0), 255 * array.shape[0]

#
def identify_col_gaps(col_sum, white_col_sum,array):
    min_gap_col = 0.03* array.shape[1]
    gap_indices = []
    in_gap = False
    current_gap_start = None
    #
    for col_index, value in enumerate(col_sum):
        # HERE TODO: replace 0.96 with a parameter
        print(f"mmmaaaaaaaaaaaaaaaaaaaaa")
        if value >= white_col_sum*0.97:
            if not in_gap:
                in_gap = True
                current_gap_start = col_index
        else:
            if in_gap:
                print(f"alalalaaaaaaaaaaaaaaaaaaaa")
                in_gap = False
                print(col_index-1 - current_gap_start)
                if (col_index-1 - current_gap_start) >= min_gap_col:
                  gap_indices.append((current_gap_start, col_index - 1))
                  current_gap_start = None
    #
    if in_gap and (len(col_sum) - 1 - current_gap_start) >= min_gap_col:
        gap_indices.append((current_gap_start, len(col_sum) - 1))
    #
    return gap_indices


# This function returns a list of "gaps", where the left edge is inclusive, and the right
# range is NOT inclusive, to be consistent with Python convention
#
# TODO: this code may be used for both row and column, with an (optional) parameter controlling
# any row or column specific parameters. Alternatively, the identify_line_gaps() code needs to be
# updated to take care of the edge cases
# 
# takes in side_marginless_array 
def identify_col_gaps2(col_sum, white_col_sum, array):
    min_gap_col = 0.03* array.shape[1]
    gap_indices = []
    # gap_lengths = []
    in_gap = False
    # curr_gap_length is sused to keep track and create the "else" for the sub "if"
    cur_gap_length = 0
    current_gap_start = None
    #
    for col_index, value in enumerate(col_sum):
        # HERE TODO: replace 0.93,0.96 with a parameter
        # less than 0.97 detects too many columns
        if value >= white_col_sum*0.97:
            if not in_gap:
                in_gap = True
                current_gap_start = col_index
                cur_gap_length = 1
            else:

                cur_gap_length = cur_gap_length + 1
        else:
            if in_gap:
                in_gap = False
                # if (col_index-1 - current_gap_start) > min_gap_col:
                if (cur_gap_length > min_gap_col):
                    # remember, the left edge is inclusive, and the right range is NOT inclusive
                    gap_indices.append((current_gap_start, current_gap_start+cur_gap_length))
                    current_gap_start = None
                    cur_gap_length = 0
    # catch the last segment if it is a "in-gap segment"
    if in_gap and (len(col_sum) - 1 - current_gap_start) > min_gap_col:
        gap_indices.append((current_gap_start, current_gap_start+cur_gap_length))    
                    
    # We don't need this bit
    # if in_gap:
    #     gap_indices.append((current_gap_start, len(col_sum) - 1))
    #
    return gap_indices




# Based on the histogram calculated, we divide the images
# into segments.
# Inputs:
# image - the numpy array that contains the intensity values of an image
# gap_indices - list of tuples that delineate the begining and the end (non-inclusive) of all
#               the gaps detected
# orientation_flag: 0 - based on column histogram,
#                   1 - based on row histogram
# Outputs:
# parts:            the list of parts by the starting index
#                   and the ending index (inclusive?)
#
# BUG: this function returns one extra part of pixel 1
# wide/tall if the edge/end is a "white".  Use
# paper12381735 - image8 you can see pixels (1530, 1530) being
# being declared as a part.
# 

#improved version of divide_image
def divide_to_columns(image, gap_indices, orientation_flag):
    parts = []
    start = 0
    for gap_start, gap_end in gap_indices:
        if (gap_start - start) >= 0.1 * image.shape[1]:
            parts.append((start, gap_start))
            start = gap_end
    if orientation_flag == 0:
        # based on column histogram
        end = image.shape[1]
    else:
        # based on row histogram
        end = image.shape[0]
    
    # Check if the last segment meets the size condition before appending
    if (end - start) >= 0.1 * image.shape[1]:
        parts.append((start, end))
    return parts


#
#
def trim_white_margins(pixel_array):
    #####passed in leftright marginless array           removes the margin and returns the top margin  
    row_sums = np.sum(pixel_array, axis=1)
    num_columns = pixel_array.shape[1]
    white_row_sum = 255 * num_columns
    #
    start_index = 0
    roughly = 0.99
    while start_index < len(row_sums) and row_sums[start_index] >= white_row_sum*roughly:
        start_index += 1
    #
    end_index = len(row_sums) - 1
    while end_index >= 0 and row_sums[end_index] >= white_row_sum*roughly:
        end_index -= 1
    #
    row_trimmed_array = pixel_array[start_index:end_index+1, :]
    return row_trimmed_array, start_index




# divide the whole img into columns
def divide_image(image, gap_indices):
    parts = []
    start = 0
    for gap_start, gap_end in gap_indices:
        parts.append((start, gap_start))
        start = gap_end
    # for the last part append (start, image.shape[0])
    parts.append((start, image.shape[0]))
    return parts

# This function 
def convert_non_255_to_zero(marginless_array_processed):
    # summing across axis=1, so it is a row sum
    marginless_array_row_sums = np.sum(marginless_array_processed, axis=1)
    #
    #
    new_arr = np.where(marginless_array_row_sums != 255*marginless_array_processed.shape[1], 0, marginless_array_row_sums)
    return new_arr

def calculate_row_sums(image, threshold=255):
    num_rows , num_columns = image.shape
    #calculates the total intensity if a row has all white(255) pixels
    white_row_sum = threshold * num_columns
    #calculates the total intensity for each row
    row_sums = np.sum(image, axis=1)
    return row_sums, white_row_sum

def identify_line_gaps(row_sums, white_row_sum):
    white_space_lengths = []
    current_length = 0
    #
    for value in row_sums:
        if value == white_row_sum:
            current_length += 1
        else:
            if current_length > 0:
                white_space_lengths.append(current_length)
                current_length = 0
    #
    if current_length > 0:
        white_space_lengths.append(current_length)
    # print(white_space_lengths)
    if white_space_lengths:
        # if (white_space_lengths == None):
        #     print("here, NONE of the whitespaces")
        # else:
        mode_result = stats.mode(white_space_lengths)
        print(mode_result)
        #
        if mode_result.mode.size > 0:
            average_line_gap = mode_result.mode
    else:
        average_line_gap = 0
    #
    return average_line_gap, white_space_lengths

# Note: gap_multiplier is what?  There are issues in this routine.... similarly to the
# other places I have fixed so far.  See identify_col_gaps2()
#
def find_significant_gaps2(row_sums, white_row_sum, average_line_gap, gap_multiplier=2.9):
    significant_gaps = []
    current_length = 0
    significant_threshold = average_line_gap * gap_multiplier
    #
    roughly=0.975
    for i, value in enumerate(row_sums):
        if value >= white_row_sum*roughly:
            current_length += 1
        else:
            if current_length >= significant_threshold:
                significant_gaps.append((i - current_length, i))
            current_length = 0
    #
    if current_length > significant_threshold:
        significant_gaps.append((len(row_sums) - current_length, len(row_sums)))
    #
    return significant_gaps



def find_significant_gaps(row_sums, white_row_sum, average_line_gap, gap_multiplier=2.9):
    significant_gaps = []
    current_length = 0
    significant_threshold = average_line_gap * gap_multiplier
    #should add roughly
    for i, value in enumerate(row_sums):
        if value == white_row_sum:
            current_length += 1
        else:
            if current_length >= significant_threshold:
                significant_gaps.append((i - current_length, i))
            current_length = 0
    #
    if current_length > significant_threshold:
        significant_gaps.append((len(row_sums) - current_length, len(row_sums)))
    #
    return significant_gaps

def display_image_parts_separately(image, parts):
    for i, (start, end) in enumerate(parts):
        part = image[start:end, 0:image.shape[1]]
        plt.figure(figsize=(10, 5))
        plt.imshow(part, cmap='gray')
        plt.axis('off')
        plt.title(f'Part {i + 1}')
        plt.show()

def analyze_texture(image_part, P=8, R=1):
    lbp = local_binary_pattern(image_part, P, R, method='uniform')
    return lbp

def has_significant_texture(lbp, threshold=0.9):
    P = 8
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist.max() > threshold



def highlight_word(img, words_to_find):
    #image = cv2.imread(img)
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform OCR using pytesseract with bounding box information
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    # print(data)

    boxes = []
    n_boxes = len(data['text'])

    for word_to_find in words_to_find:
      for i in range(n_boxes):
        if int(data['conf'][i]) > 80:  # Confidence threshold
            if word_to_find.lower() in data['text'][i].lower():
                # print(data['text'][i])
                # print(data['conf'][i])
            #if word_to_find in data['text'][i]:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                boxes.append((x, y, w, h))

                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.putText(image, word_to_find, (x, y -4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the image with highlighted word
    # cv2.imshow(img)
    return boxes

#this method won't get the actual parts cause when it calculates, the margins
# are already trimmed off
def extract_text_blocks(i, text_blocks, parts, all_cols):
    col = all_cols[i]
    col_start, col_end = col  # Unpack the start and end of the column

    # Iterate over each text block in the column
    for y_start, y_end in parts:
        x = col_start
        y = y_start
        w = col_end - col_start
        h = y_end - y_start

        # Append the box in (x, y, w, h) format
        text_blocks.append((x, y, w, h))


#  Finds the closest aligned word box (least y-distance) for each image box, 
#  where the word box must be aligned in the x-axis and have a greater y-value.
    
def aligned_boxes(image_boxes, word_boxes):

#    Returns:A dictionary where the key is the image box and the value is the closest aligned word box.
#    If no aligned word box is found, the value will be None.

    result = {}

    for img in image_boxes:
        closest_word = None
        # Initialize with infinity
        min_y_distance = float('inf')  

        for word in word_boxes:
            # if img[0] == word[0] and word[1] > img[1]:
            if abs(img[0] - word[0]) <= 5  and word[1] > img[1]:
                y_distance = word[1] - img[1]

                if y_distance < min_y_distance:
                    min_y_distance = y_distance
                    closest_word = word

        #result[tuple(img.values())] = closest_word  
        # img_key = (img['x'], img['y'], img['w'], img['h'])
        result[img] = closest_word
    return result

def resides_in(box1, box2, tolerance=5):
    return (
        box2[0] - tolerance <= box1[0] and  
        box2[0] + box2[2] + tolerance >= box1[0] + box1[2] and  
        box2[1] - tolerance <= box1[1] and  
        box2[1] + box2[3] + tolerance >= box1[1] + box1[3]  
    )


def merge_boxes(box1, box2):
    """
    Returns:
    dict: A dictionary representing the merged box with keys 'x', 'y', 'w','h'.
    """
    min_x = min(box1[0], box2[0])
    min_y = min(box1[1], box2[1])
    
    # max_x = max(box1['x'] + box1['w'], box2['x'] + box2['w'])
    max_x = max(box1[0] + box1[2], box2[0] + box2[2])
    # max_y = max(box1['y'] + box1['h'], box2['y'] + box2['h'])
    max_y = max(box1[1] + box1[3], box2[1] + box2[3])

####if we do the consider boxes for same columns 
    merged_width = max_x - min_x
    merged_height = max_y - min_y

    # Return a merged box as
    return (min_x, min_y, merged_width, merged_height)
    

def find_blocks_to_merge(im_dict, transformed_part_box):
    merged_boxes = []
    #merged_box = None
    for key,value in im_dict.items():
        #if key is None or value is None:
        if key is None or value is None:
            continue
        box2 = None
        for block in transformed_part_box:
            # if resides_in(key,block):
            #     box1 = block
            if resides_in(value, block):
                box2 = block
        #if box1 and box2:
        if box2:
            merged_box = merge_boxes(key, box2)
            merged_boxes.append(merged_box)

    return merged_boxes 


def transform_box_in_uniform(part_box):
    transformed_box = []
    for (x1, y1), (x2, y2) in part_box:
        transformed_box.append((x1, y1, x2 - x1, y2 - y1))
    return transformed_box


# This is the main function that takes an article page as a PNG image, and
# attempts to extract "regions" that are NON-text - figures, tables - and
# bound the detected region with a bounding box.
# 
# TODO: how to link the associated capture block with the image, and include
# those text in the bounding box. 
# 
def process_image2(image_path, out_prefix):
    # read, and threshold the image (thresh), as well as retrieving the initial image (img)
    thresh, img = read_and_preprocess_image(image_path)
    #
    # print(f" ......  thresh={thresh}")
    word_boxes = highlight_word(img, ["Fig.", "Figure", "Table"])
    # print("word boxes are")
    # print(word_boxes)


    side_marginless_array, left_margin = trim_white_margins_lr(thresh)
    #
    col_sum_values, white_col_sum_value = col_sum(side_marginless_array)

    #
    dim=thresh.shape[0]
    # TODO: hardcoded 255
    # col_sum_values/dim is for one column, what the avg intensity is for one pixel & then normalize
    avg_col_sum_values = col_sum_values/dim/255.
    plt.figure(figsize=(10, 5))
    x = range(len(col_sum_values))
    # plt.plot(x, col_sum_values)
    plt.plot(x, avg_col_sum_values)
    #plt.show()


    #    
    #    plt.figure(figsize=(10, 5))
    #    plt.imshow(thresh, cmap='gray')
    #    plt.show()
    #    

    gap_indices = identify_col_gaps2(col_sum_values, white_col_sum_value, side_marginless_array)
    # print (gap_indices)
    #print (f" lalalala laaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


    #a list of all the columns in the page
    all_cols = divide_to_columns(side_marginless_array, gap_indices, 0)
    print("all_cols are:")
    print(all_cols)


    marginless_arrays_processed = []
    top_margin_array = []


    box = []
    imagebox =[]
# text_blocks are to hold the text blocks in the uniform box format
    text_blocks = []
# part_box is to hold the actual chunks of text and transformed_part_box will turn it into
# uniform box format (x,y,w,h)    
    part_box = []
    transformed_part_box =[]
#desired_box is the final box with image and cation
    desired_box = []

    for col in all_cols:
        marginless_array, top_margin = trim_white_margins(side_marginless_array[:, col[0]:col[1]])
        #marginless_arrays_processed stores all the marginless columns
        # #top_margin stores all the margins for all the columns 
        marginless_arrays_processed.append(marginless_array)
        top_margin_array.append(top_margin)
        # print("marginless_arrays_processed is:")
        # print(marginless_arrays_processed)
        
        # print(f"columns are, ({col[0]}, {col[1]})")

    



    for i, marginless_array_processed in enumerate(marginless_arrays_processed):
        # marginless_array_improved_row_sum = convert_non_255_to_zero(marginless_array_processed)
        #
        row_sums, white_row_sum = calculate_row_sums(marginless_array_processed)
        # convert the histogram into a percentage for easy of reasoning

        #dim2 is the number of columns 
        dim2 = marginless_array_processed.shape[1]
        #convert each row sum into a percentage of the total possible white pixels in that row
        srow_sums = row_sums/255/dim2

        # average line spacing is about 30 pixels, without scaling (from observation)
        # the following looks wrong for the 8th page, right handside part

        print(f" =================== here, after for, inside f, i={i}")

        
        average_line_gap, white_space_lengths = identify_line_gaps(row_sums, white_row_sum)

        print(f" =================== here, after for, inside f2, i={i}")
        
        significant_gaps = find_significant_gaps2(row_sums, white_row_sum, average_line_gap)

        print(f" =================== here, after for, inside f3, i={i}")
        # Note: this is missing the last segment for some reason     
        # significant_gaps = identify_col_gaps2(row_sums, white_row_sum)
        #
        parts = divide_to_columns(marginless_array_processed, significant_gaps, 1)

        print(f" =================== here, after for, inside f4, i={i}")

        
        print(parts)





        #quit()


        for start, end in parts:
            part = marginless_array_processed[start:end, 0:marginless_array_processed.shape[1]]
           # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #let's visualize all the parts to see the image positions in parts
            if i == 0:     
                top_left = (left_margin, start + top_margin_array[i])
                bottom_right = (marginless_array_processed.shape[1] + left_margin, end + top_margin_array[i])
                # cv2.rectangle(img, start+top_margin_array[i], marginless_array_processed.shape[1]+left_margin, (0, 0, 255), 2)                          #
            else:
                x = all_cols[i][0] - all_cols[i-1][1]
                last_col_width = all_cols[i-1][1]
                # final_part = img[start+top_margin_array[i]:end+ top_margin_array[i], left_margin+last_col_width+x:marginless_array_processed.shape[1]+left_margin+last_col_width+x]
                #
                top_left = (left_margin + last_col_width + x, start + top_margin_array[i])
                bottom_right = (left_margin + last_col_width + x + marginless_array_processed.shape[1], end + top_margin_array[i])
            #
            part_box.append((top_left, bottom_right))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

            transformed_part_box = transform_box_in_uniform(part_box)


            # Let us expect any part to be at least 36 pixels in size
            # if part.size >=36:
            # TODO: this number should come from resolution/proportion of the paper
            #if part.size >=60:

            # considering 5-10% (average 7.5%) 0f the paper height would be good amount for meaningful chunk.
            if part.size >= dim2*0.075:
                lbp = analyze_texture(part)
                if has_significant_texture(lbp):
                    # i == 0 is for the first columnn with no space on the left.
                    if i == 0:
                        
                    # final_part = img[start+top_margin_array[i]:end+ top_margin_array[i], left_margin:marginless_array_processed.shape[1]+left_margin]
                        #start is the start point of any part and top_margin_array[i] 
                        top_left = (left_margin, start + top_margin_array[i])
                        bottom_right = (marginless_array_processed.shape[1] + left_margin, end + top_margin_array[i])
                        #
                        # cv2.rectangle(img, start+top_margin_array[i], marginless_array_processed.shape[1]+left_margin, (0, 0, 255), 2)                          #
                    else:
                        x = all_cols[i][0] - all_cols[i-1][1]
                        last_col_width = all_cols[i-1][1]
                        # final_part = img[start+top_margin_array[i]:end+ top_margin_array[i], left_margin+last_col_width+x:marginless_array_processed.shape[1]+left_margin+last_col_width+x]
                        #
                        top_left = (left_margin + last_col_width + x, start + top_margin_array[i])
                        bottom_right = (left_margin + last_col_width + x + marginless_array_processed.shape[1], end + top_margin_array[i])
                    #
                    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
                    box.append((top_left, bottom_right))
                    
                    #imagebox is just box but in a box object format (x ,y,w,h)
                    imagebox.append((top_left[0], top_left[1], bottom_right[0]-top_left[0], bottom_right[1]-top_left[1]))
                    # cv2.imshow('Image with Rectangle', img)
                    #
                    # plt.figure()
                    # plt.imshow(final_part, cmap='gray')
                    # plt.title(f'Part: Rows {start+top_margin_array[i]} to {end+top_margin_array[i]}')
                    # plt.show()
                    #
                    # final_result.append(final_part)
            # else:
            #     print(f"skip {image_path} {i}")
            #     break        
        # cv2.imwrite(image_path, img)

    print(len(imagebox))
    if len(imagebox) > 0:
    #extract all the text box in the uniform box format
        extract_text_blocks(i, text_blocks, parts, all_cols)
    
    outpath = out_prefix+image_path
    cv2.imwrite(outpath, img)

    print(" imagebox and word_box printing")
    print ( imagebox, word_boxes)

    # print("text_blocks are:")
    # print(text_blocks)
    print("transformed_text_blocks are:")
    print(transformed_part_box)


#image_and_word is the dictionary to hold the image: "Fig" pair
    image_and_word = aligned_boxes(imagebox, word_boxes)
    print ("image_and_word:",image_and_word)

    desired_box = find_blocks_to_merge(image_and_word, transformed_part_box)
    
    if desired_box:
        print("desired_box:", desired_box)
        for i in desired_box:
            (x, y, w, h) = i
            # print(f"Drawing box at: ({x}, {y}, {w}, {h})")
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 8)
    else:
        print("No desired box found.")
    

    cv2.imshow("Image with Boxes", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_pdfs_to_images(pdf_folder, image_folder):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    #
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
    #convert_from_path function converts each page of the PDF into an image.
            images = convert_from_path(pdf_path)
    #extracts the base name of the PDF file (without the extension)
    # and stores it in the pdf_name variable.
            pdf_name = os.path.splitext(pdf_file)[0]
            #
            for i, image in enumerate(images):
                image_path = os.path.join(image_folder, f"{pdf_name}_{i + 1}.png")
                image.save(image_path, 'PNG')






# Start of the main process, currently a script for debugging
# def main():
pdf_folder = "C:\summer2024\Figure-Extraction\paper_pdf"
# pdf_folder = 'physics_papers'
#image_folder = 'test1999_inversion_xia_image'
# image_folder = 'paper_image_full'
# image_folder = 'paper_image_full'
# image_folder='biochemj00498_0024_pdf_image'
#image_folder = 'paper12381735_image'
image_folder = 'imageFromPaper'


# use the prefix to make sure output images are not overwriting the input images
#########this is the last folder being used.
out_prefix = 'results_30_'
out_folder=out_prefix+image_folder


# create the output image folder 
if not os.path.exists(out_folder):
    os.makedirs(out_folder)


# if not os.path.exists(figures_folder):
if not os.path.exists(image_folder):
    # print("going to make figures_dir")
    # os.makedirs(figures_folder)
    print(f"beging making images of the document and put them into {image_folder}")
    #once the image folder is made, any manual changes 
    # to that folder won't be seen while running the code
    convert_pdfs_to_images(pdf_folder, image_folder)
    print("after making the images")
else:
    print("skipping making figures_dir, or converting any files")


# total number of paper-pages to process
ficounter = 0


# Note: the order seems to be rather random
all_file_images = os.listdir(image_folder)

# sorting this list, please, and let us use the sorted version
all_file_images.sort()

print(f"all file images after sorting are  {all_file_images} .... " )

for image_file in all_file_images:
    ficounter = ficounter + 1
    print(f"\n\n*** ** * Processing No. {ficounter} image file: {image_file} .... " )
    #
    #
    if image_file.endswith('.png'):
        image_path = os.path.join(image_folder, image_file)
        #
        if (ficounter >-1): 
            #
            print(f"-------- got an image ...., processing image in  {image_path}" )
            #
            # figures= process_image(image_path)
            #
            process_image2(image_path, out_prefix)
            print(f"-------- after processing for figures from the path  {image_path} \n" )
        else:
            print("only focusing on page 3")
            continue
        #
    #
    else:
        print("something is not right: please check why we are not looking at a PNG file. ")
        quit()

