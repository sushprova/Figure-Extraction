# def resides_in(box1, box2):
# ##true if box1 is inside box2
#     return (
#         box2[0] <= box1[0] and  
#         box2[0] + box2[2] >= box1[0] + box1[2] and  
#         box2[1] <= box1[1] and  
#         box2[1] + box2[3] >= box1[1] + box1[3]  
#     )


def resides_in(box1, box2, tolerance=5):
    """Returns True if box1 is inside box2, allowing for a tolerance gap."""
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
    

def find_blocks_to_merge(im_dict, text_block):
    merged_box = None 
    i = 0
    for key,value in im_dict.items():
        print("i", i)
        i+= 1
        if key is None or value is None:
            continue
        print("key-value", key, value)

        box1 = box2 = None
        for block in text_block:
            # if resides_in(key,block):
            #     box1 = block
            print("box1", box1)
            if resides_in(value, block):
                box2 = block
            print("box2",box2)
#maybe make 2 different for loops
        # if box1 and box2:
            merged_box = merge_boxes(key, box2)
            # merged_boxes.append(merged_box)

    return merged_box


def find_blocks_to_merge1(im_dict, text_block):
    merged_boxes = []  # Initialize list to store merged boxes

    for key, value in im_dict.items():
        if key is None or value is None:
            continue
        
        print("Key-Value:", key, value)

        # Variables to store the boxes that we might merge
        box_to_merge = None

        # Loop through each block to check if `value` resides in any `block`
        for block in text_block:
            if resides_in(value, block):
                box_to_merge = block
                break  # Exit loop once a match is found

        # If a matching block is found, merge `key` and `box_to_merge`
        if box_to_merge:
            merged_box = merge_boxes(key, box_to_merge)
            merged_boxes.append(merged_box)

    return merged_boxes  # Return list of merged boxes
                
# im_dict = {(166, 149, 630, 243): (163, 716, 38, 27),  (166, 464, 630, 164): (163, 716, 38, 27), (864, 924, 630, 245): (865, 1633, 34, 19)}
# text_block= [(160, 140, 640, 260) , (163,700, 300, 60 ) ]

text_block= [(166, 149, 630, 243), (166, 464, 630, 164), (166, 670, 630, 343), (166, 1075, 630, 849), (864, 97, 630, 69), (864, 196, 630, 650), (864, 924, 630, 245), (864, 1201, 630, 725)]
im_dict = {(166, 149, 630, 243): (163, 716, 38, 27), (166, 464, 630, 164): (163, 716, 38, 27), (864, 924, 630, 245): (865, 1633, 34, 19)}

##Has None so that is counted as the key-value
# im_dict = {(166, 149, 630, 243): (163, 716, 38, 27),  (166, 464, 630, 164): (163, 716, 38, 27), (166, 149, 630, 243): None , 
# (864, 924, 630, 245): (865, 1633, 34, 19)}
# text_block= [(160, 140, 640, 260) , (163,700, 300, 60 ) ]


# im_dict =  {(166, 149, 630, 243): (163, 716, 38, 27), (166, 464, 630, 164): (163, 716, 38, 27), (864, 924, 630, 245): (865, 1633, 34, 19)}
# text_block= [(698, 0, 630, 69), (698, 99, 630, 650), (698, 827, 630, 245), (698, 1104, 630, 725)]

# a = resides_in((166, 464, 630, 164), text_block[0])
# print(a)

# b = resides_in((163, 716, 38, 27), text_block[1])
# print(b)

# c = merge_boxes(text_block[0], text_block[1])
# print(c)

gotit = find_blocks_to_merge1(im_dict, text_block)
print("final:", gotit)
