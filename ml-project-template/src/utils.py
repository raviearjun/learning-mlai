## Directory: src/utils.py
import cv2

def draw_text(image, label, score, pos=(0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, font_thickness=2, text_color=(0, 0, 0), text_color_bg=(0, 255, 0)):
    x, y = pos
    score_text = f'Score: {score:.2f}'
    (w1, h1), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(image, (x, y-h1-h2-25), (x + max(w1, w2)+20, y), text_color_bg, -1)
    cv2.putText(image, label, (x+10, y-10), font, font_scale, text_color, font_thickness)
    cv2.putText(image, score_text, (x+10, y-h2-15), font, font_scale, text_color, font_thickness)

def draw_result(image, scores, labels, coords):
    result_image = image.copy()
    for (x, y, w, h), label, score in zip(coords, labels, scores):
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_text(result_image, label, score, pos=(x, y))
    return result_image

