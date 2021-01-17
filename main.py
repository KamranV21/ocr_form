import cv2
import numpy
import pytesseract
import os
import json

region_of_interests = [
    [(353, 1653), (1119, 1798), 'text', 'item_one'],
    [(353, 1798), (1119, 1943), 'text', 'item_two'],
    [(353, 1943), (1119, 2088), 'text', 'item_three'],
    [(353, 2088), (1119, 2233), 'text', 'item_four'],
    [(353, 2233), (1119, 2378), 'text', 'item_five'],
    [(400, 2700), (1044, 2860), 'text', 'person_one'],
    [(1548, 2700), (2174, 2860), 'text', 'person_two']
]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

original_form = cv2.imread('original_forms/form.png')

orb = cv2.ORB_create(5000)
original_keypoint, original_descriptor = orb.detectAndCompute(original_form, None)

path = 'test_forms'
image_list = os.listdir(path)
for index, file_name in enumerate(image_list):
    image = cv2.imread(f'{path}/{file_name}')
    image_keypoint, image_descriptor = orb.detectAndCompute(image, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(image_descriptor, original_descriptor)
    matches.sort(key=lambda x: x.distance)
    good_matches = matches[0:int(len(matches)*0.05)]
    match_image = cv2.drawMatches(image, image_keypoint, original_form, original_keypoint, good_matches, None, flags=2)

    source_points = numpy.float32([image_keypoint[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    destination_points = numpy.float32([original_keypoint[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
    height, width, channel = original_form.shape
    scan_image = cv2.warpPerspective(image, matrix, (width, height))

    image_show = scan_image.copy()
    image_mask = numpy.zeros_like(image_show)

    data = []

    for i, region in enumerate(region_of_interests):
        cv2.rectangle(image_mask, (region[0][0], region[0][1]), (region[1][0], region[1][1]), (0, 255, 0), cv2.FILLED)
        image_show = cv2.addWeighted(image_show, 0.99, image_mask, 0.1, 0)

        cropped_image = scan_image[region[0][1]:region[1][1], region[0][0]:region[1][0]]
        cv2.imshow(f'Cropped {i + 1}', cropped_image)

        if region[2] == 'text':
            text = pytesseract.image_to_string(cropped_image)
            print(f'{region[3]}: {text}')
            data.append({region[3]: text})

    height_final, width_final, channel_final = image_show.shape
    image_show = cv2.resize(image_show, (width_final // 5, height_final // 5))
    cv2.imshow(f'Final image {index + 1}', image_show)

    json_data = json.dumps(data)
    print(json_data)

# cv2.imshow('Original form', original_form)
cv2.waitKey(0)