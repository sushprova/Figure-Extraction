import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from skimage.feature import local_binary_pattern
from scipy import stats
import os
from pdf2image import convert_from_path

def read_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 230, 255, cv2.THRESH_BINARY)
    return thresh, img

def trim_white_margins_lr(pixel_array):
    col_sums = np.sum(pixel_array, axis=0)
    num_rows = pixel_array.shape[0]
    white_col_sum = 255 * num_rows

    start_index = 0
    while start_index < len(col_sums) and col_sums[start_index] == white_col_sum:
        start_index += 1

    end_index = len(col_sums) - 1
    while end_index >= 0 and col_sums[end_index] == white_col_sum:
        end_index -= 1

    col_trimmed_array = pixel_array[:, start_index:end_index+1]
    return col_trimmed_array, start_index

def trim_white_margins(pixel_array):
    row_sums = np.sum(pixel_array, axis=1)
    num_columns = pixel_array.shape[1]
    white_row_sum = 255 * num_columns

    start_index = 0
    while start_index < len(row_sums) and row_sums[start_index] == white_row_sum:
        start_index += 1

    end_index = len(row_sums) - 1
    while end_index >= 0 and row_sums[end_index] == white_row_sum:
        end_index -= 1

    row_trimmed_array = pixel_array[start_index:end_index+1, :]
    return row_trimmed_array, start_index

def col_sum(array):
    return np.sum(array, axis=0), 255 * array.shape[0]

def identify_col_gaps(col_sum, white_col_sum):
    min_gap_col = 3
    gap_indices = []
    in_gap = False
    current_gap_start = None

    for col_index, value in enumerate(col_sum):
        if value == white_col_sum:
            if not in_gap:
                in_gap = True
                current_gap_start = col_index
        else:
            if in_gap:
                in_gap = False
                if (col_index-1 - current_gap_start) > min_gap_col:
                  gap_indices.append((current_gap_start, col_index - 1))
                  current_gap_start = None

    if in_gap:
        gap_indices.append((current_gap_start, len(col_sum) - 1))

    return gap_indices

def divide_image(image, gap_indices):
    parts = []
    start = 0
    for gap_start, gap_end in gap_indices:
        parts.append((start, gap_start))
        start = gap_end
    # parts.append((start, image.shape[1]))
    parts.append((start, image.shape[0]))
    return parts

def convert_non_255_to_zero(marginless_array_processed):
    marginless_array_row_sums = np.sum(marginless_array_processed, axis=1)
    new_arr = np.where(marginless_array_row_sums != 255*marginless_array_processed.shape[1], 0, marginless_array_row_sums)
    return new_arr

def calculate_row_sums(image, threshold=255):
    num_rows, num_columns = image.shape
    white_row_sum = threshold * num_columns
    row_sums = np.sum(image, axis=1)
    return row_sums, white_row_sum

def identify_line_gaps(row_sums, white_row_sum):
    white_space_lengths = []
    current_length = 0

    for value in row_sums:
        if value == white_row_sum:
            current_length += 1
        else:
            if current_length > 0:
                white_space_lengths.append(current_length)
                current_length = 0

    if current_length > 0:
        white_space_lengths.append(current_length)
    # print(white_space_lengths)
    if white_space_lengths:
        mode_result = stats.mode(white_space_lengths)
        # print(mode_result)
        if mode_result.mode.size > 0:
            average_line_gap = mode_result.mode
    else:
        average_line_gap = 0

    return average_line_gap, white_space_lengths

def find_significant_gaps(row_sums, white_row_sum, average_line_gap, gap_multiplier=2.9):
    significant_gaps = []
    current_length = 0
    significant_threshold = average_line_gap * gap_multiplier

    for i, value in enumerate(row_sums):
        if value == white_row_sum:
            current_length += 1
        else:
            if current_length >= significant_threshold:
                significant_gaps.append((i - current_length, i))
            current_length = 0

    if current_length > significant_threshold:
        significant_gaps.append((len(row_sums) - current_length, len(row_sums)))

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


def process_image(image_path):
    thresh, img = read_and_preprocess_image(image_path)
    side_marginless_array, left_margin = trim_white_margins_lr(thresh)

    col_sum_values, white_col_sum_value = col_sum(side_marginless_array)
    gap_indices = identify_col_gaps(col_sum_values, white_col_sum_value)
    all_cols = divide_image(side_marginless_array, gap_indices)

    marginless_arrays_processed = []
    top_margin_array = []

    final_result = []
    for col in all_cols:
        marginless_array, top_margin = trim_white_margins(side_marginless_array[:, col[0]:col[1]])
        marginless_arrays_processed.append(marginless_array)
        top_margin_array.append(top_margin)

    for i, marginless_array_processed in enumerate(marginless_arrays_processed):
        marginless_array_improved_row_sum = convert_non_255_to_zero(marginless_array_processed)

        row_sums, white_row_sum = calculate_row_sums(marginless_array_processed)
        average_line_gap, white_space_lengths = identify_line_gaps(row_sums, white_row_sum)
        significant_gaps = find_significant_gaps(row_sums, white_row_sum, average_line_gap)

        parts = divide_image(marginless_array_processed, significant_gaps)
        print(parts)
        for start, end in parts:
            part = marginless_array_processed[start:end, 0:marginless_array_processed.shape[1]]
            # print(part)
            if part.size >1:
                lbp = analyze_texture(part)
                if has_significant_texture(lbp):
                    if i == 0:
                        # final_part = img[start+top_margin_array[i]:end+ top_margin_array[i], left_margin:marginless_array_processed.shape[1]+left_margin]
                        #####ekek col er top margin er size diff hoite pare tai#####
                        top_left = (left_margin, start + top_margin_array[i])
                        bottom_right = (marginless_array_processed.shape[1] + left_margin, end + top_margin_array[i])
                   
                        # cv2.rectangle(img, start+top_margin_array[i], marginless_array_processed.shape[1]+left_margin, (0, 0, 255), 2)  
  
                        # cv2.imshow('Image with Rectangle', img)
                    else:
                        x = all_cols[i][0] - all_cols[i-1][1]
                        last_col_width = all_cols[i-1][1]
                        # final_part = img[start+top_margin_array[i]:end+ top_margin_array[i], left_margin+last_col_width+x:marginless_array_processed.shape[1]+left_margin+last_col_width+x]
                        
                        top_left = (left_margin + last_col_width + x, start + top_margin_array[i])
                        bottom_right = (left_margin + last_col_width + x + marginless_array_processed.shape[1], end + top_margin_array[i])

                    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

                    # cv2.imshow('Image with Rectangle', img)

                    # plt.figure()
                    # plt.imshow(final_part, cmap='gray')
                    # plt.title(f'Part: Rows {start+top_margin_array[i]} to {end+top_margin_array[i]}')
                    # plt.show()

                    # final_result.append(final_part)
            # else:
            #     print(f"skip {image_path} {i}")
            #     break        
    # return final_result
    cv2.imwrite(image_path, img)


def convert_pdfs_to_images(pdf_folder, image_folder):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            images = convert_from_path(pdf_path)
            pdf_name = os.path.splitext(pdf_file)[0]

            for i, image in enumerate(images):
                image_path = os.path.join(image_folder, f"{pdf_name}_{i + 1}.png")
                image.save(image_path, 'PNG')

def main():
    pdf_folder = 'paper_pdf'
    image_folder = 'paper_image'
    # figures_folder = 'figures'


    # if not os.path.exists(figures_folder):
    #     os.makedirs(figures_folder)

    convert_pdfs_to_images(pdf_folder, image_folder)

    for image_file in os.listdir(image_folder):
        if image_file.endswith('.png'):
            image_path = os.path.join(image_folder, image_file)
            # figures= process_image(image_path)
            process_image(image_path)

            # for idx, figure in enumerate(figures):
            #     # figure_path = os.path.join(figures_folder, f"processed_{os.path.splitext(image_file)[0]}_{idx + 1}.png")
            #     # matplotlib.image.imsave('name.png', figure)
            #     figure_path = os.path.join(figures_folder, f"processed_{os.path.splitext(os.path.basename(image_path))[0]}_{idx + 1}.png")
            #     plt.imsave(figure_path, figure)

if __name__ == "__main__":
    main()
