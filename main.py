import numpy as np
import mediapipe as mp
import model as m
import torch
import cv2
import utilities as u

def main():
    model = m.Model()
    model.getModel()
    model.model.eval()
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.6, min_tracking_confidence = 0.6)
    color = [255, 0, 127]
    opacity = 11
    cv2.namedWindow("Main Window")
    cv2.createTrackbar("R", "Main Window", color[2], 255, lambda x: None)
    cv2.createTrackbar("G", "Main Window", color[1], 255, lambda x: None)
    cv2.createTrackbar("B", "Main Window", color[0], 255, lambda x: None)
    cv2.createTrackbar("Opacity", "Main Window", opacity, 100, lambda x: None)
    slider_panel_height = 60
    slider_height = 20
    slider_width = width // 4
    label_font_scale = 0.5
    label_thickness = 1

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            color[2] = cv2.getTrackbarPos("R", "Main Window")
            color[1] = cv2.getTrackbarPos("G", "Main Window")
            color[0] = cv2.getTrackbarPos("B", "Main Window")
            opacity = cv2.getTrackbarPos("Opacity", "Main Window") / 100
            for landmarks in results.multi_face_landmarks:
                left_eye_area, left_eye_box = u.getEyeArea(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), landmarks.landmark, u.left_boundaries, image_width, image_height, margin = 0.2)
                right_eye_area, right_eye_box = u.getEyeArea(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), landmarks.landmark, u.right_boundaries, image_width, image_height, margin = 0.2)
                left_eye_area_tensor = u.preprocessImage(left_eye_area, model).to(model.device)
                right_eye_area_tensor = u.preprocessImage(right_eye_area, model).to(model.device)
                with torch.no_grad():
                    left_eye_pred = model.model(left_eye_area_tensor).cpu().squeeze(0).numpy()
                    right_eye_pred = model.model(right_eye_area_tensor).cpu().squeeze(0).numpy()
                left_eye_pred_image = u.postprocessPrediction(left_eye_pred[0], left_eye_area.shape[:2])
                right_eye_pred_image = u.postprocessPrediction(right_eye_pred[0], right_eye_area.shape[:2])
                mask_left = left_eye_pred_image > 0
                mask_right = right_eye_pred_image > 0
                blended_color_left = np.where(mask_left[:, :, None], np.array(color, dtype = np.uint8) * opacity + image[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2]] * (1 - opacity), image[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2]])
                blended_color_right = np.where(mask_right[:, :, None], np.array(color, dtype = np.uint8) * opacity + image[right_eye_box[1]:right_eye_box[3],right_eye_box[0]:right_eye_box[2]] * (1 - opacity),image[right_eye_box[1]:right_eye_box[3], right_eye_box[0]:right_eye_box[2]])
                image[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2]] = blended_color_left
                image[right_eye_box[1]:right_eye_box[3], right_eye_box[0]:right_eye_box[2]] = blended_color_right
        slider_panel = np.zeros((slider_panel_height, width, 3), dtype = np.uint8)
        cv2.putText(slider_panel, "R", (10, slider_panel_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale,(255, 255, 255), label_thickness)
        cv2.putText(slider_panel, "G", (slider_width + 20, slider_panel_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,label_font_scale, (255, 255, 255), label_thickness)
        cv2.putText(slider_panel, "B", (2 * slider_width + 30, slider_panel_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,label_font_scale, (255, 255, 255), label_thickness)
        cv2.putText(slider_panel, "Opacity", (3 * slider_width + 40, slider_panel_height // 2 + 5),cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255), label_thickness)
        cv2.line(slider_panel, (10, slider_panel_height - 20), (10 + slider_width, slider_panel_height - 20),(255, 255, 255), slider_height)
        cv2.line(slider_panel, (10 + slider_width + 20, slider_panel_height - 20),(10 + 2 * slider_width + 20, slider_panel_height - 20), (255, 255, 255), slider_height)
        cv2.line(slider_panel, (10 + 2 * slider_width + 30, slider_panel_height - 20),(10 + 3 * slider_width + 30, slider_panel_height - 20), (255, 255, 255), slider_height)
        cv2.line(slider_panel, (10 + 3 * slider_width + 40, slider_panel_height - 20),(10 + 4 * slider_width + 40, slider_panel_height - 20), (255, 255, 255), slider_height)
        combined_image = np.vstack((image, slider_panel))
        cv2.imshow("Main Window", cv2.flip(combined_image, 1))
        if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty("Main Window", cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
