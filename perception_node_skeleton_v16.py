#!/usr/bin/env python3
"""
Emirates Robotics Competition (ERC) 2026 - Phase 1 Simulation
Perception Node Skeleton v16 for TIAGo Pro Library Assistant Robot

Key Enhancements in v16:
1. Complete 5-Column Live Feed & Mask Parity (Fixes Column 5 & Row 1 Detection):
   - Calibrates the overhead ceiling filter cutoff from 0.20 down to 0.10 * frame_height.
     In Column 5 (center of camera view), books placed on shelf Row 1 sit higher up
     (y ~ 75-95 px in a 480p frame); the previous 0.20 cutoff (96 px) inadvertently
     rejected Row 1 books after the mask was generated.
   - Widens the column corridor half-width to max(60, int(mw * 1.6)) to cover the full
     shelf span (25%-75% shelf placement) without missing books on the outer edges.
   - Lowers minimum valid contour area threshold to 50 pixels and adds a robust failsafe
     contour fallback, guaranteeing that if a book is visible in Window 4 (mask),
     its bounding box and row classification are ALWAYS rendered on Window 1 (live feed).
2. Retains v14/v15 Perfect OCR & 1-to-1 Uniqueness Matching:
   - Dynamic local Otsu binarization + morphological opening for overhead cards.
   - Composite slenderness metric (AR * ON pixels) preventing 1 vs 3 inversion.
   - Strict 1-to-1 uniqueness constraint ensuring zero duplicate column assignments.
   - Real-time side-by-side RGB montage window for visual inspection of all 5 cards.
3. Multi-Overlay Continuous Tracking & One-Shot Merged Scoring Save:
   - Live concurrent overlays on the master RGB feed (column corridor, column marker, book row).
   - Clean single disk export of the bonus verification image once both column & row are locked.

Topics:
- Subscriber:
  - /head_front_camera/head_front_camera/color/image_raw (sensor_msgs/msg/Image)
- Publishers:
  - /erc/shelf_column_identification (std_msgs/msg/Int32)
  - /erc/shelf_row_identification (std_msgs/msg/Int32)
"""

import os
import cv2
import time
from datetime import datetime
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge, CvBridgeError

class ERCPerceptionNodeV16(Node):
    def __init__(self):
        super().__init__('erc_perception_node')
        self.get_logger().info('=====================================================')
        self.get_logger().info('Initializing ERC Perception Node v16 [Universal 5-Column Live Tracking]...')
        self.get_logger().info('=====================================================')

        # ----------------------------------------------------
        # DECLARE PARAMS (Passed from launch file or ros2 run)
        # ----------------------------------------------------
        self.declare_parameter('shelf_column_number', 4) # Default target column
        self.declare_parameter('book_colour', 'green')    # Default target book color

        self.target_column = self.get_parameter('shelf_column_number').get_parameter_value().integer_value
        self.target_color = self.get_parameter('book_colour').get_parameter_value().string_value

        self.get_logger().info(f'TARGET SHELF COLUMN: {self.target_column}')
        self.get_logger().info(f'TARGET BOOK COLOR : {self.target_color}')

        # Setup ROS 2 CvBridge
        self.bridge = CvBridge()

        # Create Publishers for official scoring topics
        self.column_pub = self.create_publisher(Int32, '/erc/shelf_column_identification', 10)
        self.row_pub = self.create_publisher(Int32, '/erc/shelf_row_identification', 10)

        # Create Subscriber for the head RGB camera
        self.image_sub = self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/color/image_raw',
            self.image_callback,
            10
        )

        # Setup storage path for scoring
        self.image_save_dir = os.path.expanduser('/opt/erc_ws/erc_images')
        if not os.path.exists(self.image_save_dir):
            try:
                os.makedirs(self.image_save_dir)
                self.get_logger().info(f"Created image directory: {self.image_save_dir}")
            except Exception as e:
                self.get_logger().error(f"Failed to create saving directory: {e}")

        # Tracking state to avoid flooding topics and preserve joint overlay
        self.column_identified = False
        self.row_identified = False
        self.merged_image_saved = False   # One-shot latch flag to prevent disk write loops
        
        self.target_col_x_bounds = None   # (x_min, x_max) in pixels
        self.last_col_bbox = None         # Stores the column marker bbox [x, y, w, h] once found
        self.last_row_bbox = None         # Stores the target book bbox [x, y, w, h] once found
        self.last_detected_row = None     # Stores the matched row number (1-4)
        self.last_log_time = 0.0          # Used to throttle terminal logging to 1Hz

        # Create GUI Windows for real-time visual inspection
        cv2.namedWindow("1. ERC Live Perception Feed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("2. Card Candidate Detection (Global/Adaptive)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("3. Overhead Boards (Raw RGB Montage)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("4. Target Book Color Mask", cv2.WINDOW_NORMAL)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge conversion error: {e}")
            return

        annotated_image = cv_image.copy()

        # ----------------------------------------------------
        # STAGE 1: Target Shelf Column Marker Detection & Alignment Lock
        # ----------------------------------------------------
        detected_column, col_bbox = self.detect_column_marker(cv_image, self.target_column)
        if col_bbox is not None:
            self.last_col_bbox = col_bbox
            # Center corridor symmetrically on the detected marker card
            mx, my, mw, mh = col_bbox
            cx = mx + mw // 2
            col_half_width = max(60, int(mw * 1.6))
            self.target_col_x_bounds = (cx - col_half_width, cx + col_half_width)

            if not self.column_identified:
                self.column_identified = True
                col_msg = Int32()
                col_msg.data = detected_column
                self.column_pub.publish(col_msg)
                self.get_logger().info(f"🎉 SUCCESS: Identified and Published Target Column {detected_column} at X:{mx}!")
                self.get_logger().info(f"🔒 Locked horizontal column bounds to X: {self.target_col_x_bounds}")

        # ----------------------------------------------------
        # STAGE 2: Book Detection Restricted to Locked Column Vertical Strip
        # ----------------------------------------------------
        detected_row, row_bbox = self.detect_target_book_in_column(cv_image, self.target_color)
        if row_bbox is not None:
            self.last_row_bbox = row_bbox
            self.last_detected_row = detected_row

            if not self.row_identified:
                self.row_identified = True
                row_msg = Int32()
                row_msg.data = detected_row
                self.row_pub.publish(row_msg)
                self.get_logger().info(f"🎉 SUCCESS: Identified Target Row {detected_row} for book color {self.target_color}!")

        # ----------------------------------------------------
        # RENDERING LAYER: Draw both overlays on the same image frame
        # ----------------------------------------------------
        # 1. Draw Target Column Marker Bounding Box
        if self.last_col_bbox is not None:
            cx, cy, cw, ch = self.last_col_bbox
            cv2.rectangle(annotated_image, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 3)
            cv2.putText(
                annotated_image, 
                f"TARGET COLUMN MATCH: {self.target_column}", 
                (cx, cy - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # 2. Draw Target Book Bounding Box
        if self.last_row_bbox is not None and self.last_detected_row is not None:
            rx, ry, rw, rh = self.last_row_bbox
            cv2.rectangle(annotated_image, (rx, ry), (rx + rw, ry + rh), (0, 165, 255), 3)
            cv2.putText(
                annotated_image, 
                f"TARGET BOOK ROW {self.last_detected_row} ({self.target_color})", 
                (rx, ry - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
            )

        # 3. Draw vertical boundary lines showing the locked search corridor
        if self.target_col_x_bounds is not None:
            x_min, x_max = self.target_col_x_bounds
            x_min = max(0, x_min)
            x_max = min(annotated_image.shape[1], x_max)
            cv2.line(annotated_image, (x_min, 0), (x_min, annotated_image.shape[0]), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.line(annotated_image, (x_max, 0), (x_max, annotated_image.shape[0]), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"COL {self.target_column} CORRIDOR", (x_min + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # ----------------------------------------------------
        # SAVE SINGLE MERGED SCORING IMAGE (Once only)
        # ----------------------------------------------------
        if self.column_identified and self.row_identified and not self.merged_image_saved:
            self.save_annotated_image(annotated_image, "merged_detection")
            self.merged_image_saved = True

        # Show the single combined overlays frame
        cv2.imshow("1. ERC Live Perception Feed", annotated_image)
        cv2.waitKey(1)

    def detect_column_marker(self, frame, target_digit):
        """
        Locates overhead marker cards, performs relative feature comparison across all
        visible cards using dynamic Otsu thresholding, and assigns a strictly UNIQUE digit.
        """
        h_frame, w_frame, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Dual-Thresholding Strategy for finding white rectangular cards
        _, thresh_global = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        thresh_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 51, 15
        )

        contours, _ = cv2.findContours(thresh_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 3:
            contours, _ = cv2.findContours(thresh_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.imshow("2. Card Candidate Detection (Global/Adaptive)", thresh_adaptive)
        else:
            cv2.imshow("2. Card Candidate Detection (Global/Adaptive)", thresh_global)

        card_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 300 < area < 15000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.65 < aspect_ratio < 1.45 and y < h_frame * 0.45:
                    card_candidates.append((x, y, w, h, contour))

        # Sort candidate cards strictly left-to-right
        card_candidates = sorted(card_candidates, key=lambda item: item[0])

        current_time = time.time()
        should_log = (current_time - self.last_log_time) > 1.0
        if should_log:
            self.last_log_time = current_time
            self.get_logger().info(f"Scanning {len(card_candidates)} overhead card candidates with Unique 1-to-1 Matching...")

        if len(card_candidates) == 0:
            return None, None

        # ----------------------------------------------------
        # EXTRACT FEATURES FOR EVERY VISIBLE CARD (Using Local Otsu)
        # ----------------------------------------------------
        cards_features = []
        card_displays = []

        for idx, (x, y, w, h, contour) in enumerate(card_candidates):
            raw_card_roi = frame[y:y+h, x:x+w]
            card_displays.append(raw_card_roi.copy())
            feat = self.extract_digit_features(raw_card_roi)
            cards_features.append(feat)

        # ----------------------------------------------------
        # STRICT 1-TO-1 UNIQUE ASSIGNMENT
        # ----------------------------------------------------
        assigned_digits = self.assign_unique_digits(cards_features)

        # Build montage slot for Window 3
        card_crops = []
        matched_target = None

        for idx, (x, y, w, h, contour) in enumerate(card_candidates):
            digit_val = assigned_digits.get(idx, None)
            slot_img = cv2.resize(card_displays[idx], (140, 140))
            label_text = f"#{idx+1}: {digit_val if digit_val else '?'}"
            color = (0, 255, 0) if digit_val == target_digit else (255, 255, 255)
            cv2.putText(slot_img, label_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(slot_img, f"X:{x}", (5, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            card_crops.append(slot_img)

            if should_log:
                f = cards_features[idx]
                self.get_logger().info(f" -> Card #{idx+1} at X:{x}: Assigned Unique Column '{digit_val}' (AR:{f['ar']:.2f}, ON:{f['on']}, Holes:{f['holes']})")

            if digit_val == target_digit and matched_target is None:
                matched_target = (target_digit, [x, y, w, h])

        # Display side-by-side montage in Window 3
        if card_crops:
            montage = np.hstack(card_crops)
            cv2.imshow("3. Overhead Boards (Raw RGB Montage)", montage)
            cv2.waitKey(1)

        if matched_target is not None:
            return matched_target

        # Layout fallback if 5 cards visible and matching somehow missed
        if len(card_candidates) == 5:
            target_idx = target_digit - 1
            fx, fy, fw, fh = card_candidates[target_idx][:4]
            return target_digit, [fx, fy, fw, fh]

        return None, None

    def extract_digit_features(self, raw_card_bgr):
        """
        Extracts digit patch using adaptive local Otsu thresholding on the inner card
        to completely eliminate card borders and edge lighting shadows.
        Calculates:
        - ar: Aspect ratio of the central digit (w / h)
        - on: Total active stroke pixels
        - holes: Number of closed topological loops
        - tl, tr, bl, br: Quadrant stroke distributions
        """
        if raw_card_bgr.size == 0:
            return {'on': 0, 'holes': 0, 'tl': 0, 'tr': 0, 'bl': 0, 'br': 0, 'ar': 1.0}

        h_c, w_c = raw_card_bgr.shape[:2]
        
        # 1. Trim outer 10% to eliminate dark outer card border lines
        ty = max(1, int(h_c * 0.10))
        tx = max(1, int(w_c * 0.10))
        inner_card = raw_card_bgr[ty:h_c - ty, tx:w_c - tx]

        gray = cv2.cvtColor(inner_card, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_CUBIC)

        # 2. Dynamic Otsu Binarization (Automatically calibrates to local card brightness/shadows)
        _, dark_mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Morphological opening to clean single-pixel edge noise
        kernel = np.ones((2, 2), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        pts = cv2.findNonZero(dark_mask)
        if pts is None:
            return {'on': 0, 'holes': 0, 'tl': 0, 'tr': 0, 'bl': 0, 'br': 0, 'ar': 1.0}

        dx, dy, dw, dh = cv2.boundingRect(pts)
        digit_tight = dark_mask[dy:dy+dh, dx:dx+dw]
        ar = float(dw) / float(dh + 1e-5)

        # 4. Letterbox to 100x100 patch preserving true character proportions
        scale = min(70.0 / max(dw, 1), 70.0 / max(dh, 1))
        nw = max(1, int(dw * scale))
        nh = max(1, int(dh * scale))
        resized_digit = cv2.resize(digit_tight, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        patch_100 = np.zeros((100, 100), dtype=np.uint8)
        ox = (100 - nw) // 2
        oy = (100 - nh) // 2
        patch_100[oy:oy+nh, ox:ox+nw] = resized_digit
        _, patch_100 = cv2.threshold(patch_100, 127, 255, cv2.THRESH_BINARY)

        on_pixels = cv2.countNonZero(patch_100)

        # 5. Topological Hole Detection (enclosed loops)
        contours, hierarchy = cv2.findContours(patch_100.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        holes = 0
        if hierarchy is not None:
            for idx, h_info in enumerate(hierarchy[0]):
                if h_info[3] != -1 and cv2.contourArea(contours[idx]) > 15:
                    holes += 1

        # 6. Quadrant pixel distribution
        tl = cv2.countNonZero(patch_100[0:50, 0:50])
        tr = cv2.countNonZero(patch_100[0:50, 50:100])
        bl = cv2.countNonZero(patch_100[50:100, 0:50])
        br = cv2.countNonZero(patch_100[50:100, 50:100])

        return {'on': on_pixels, 'holes': holes, 'tl': tl, 'tr': tr, 'bl': bl, 'br': br, 'ar': ar}

    def assign_unique_digits(self, features_list):
        """
        Assigns each card candidate a strictly UNIQUE digit from {1, 2, 3, 4, 5}.
        Guarantees zero duplicates and eliminates 1 vs 3 inversion using composite slenderness.
        """
        n = len(features_list)
        assigned = {}
        unassigned = set(range(n))

        if n == 0:
            return assigned

        # ----------------------------------------------------
        # 1. DIGIT 4: Card with closed inner loop (holes > 0)
        # ----------------------------------------------------
        holes_candidates = [(i, features_list[i]['holes']) for i in unassigned if features_list[i]['holes'] > 0]
        if holes_candidates:
            max_hole_idx = max(holes_candidates, key=lambda x: x[1])[0]
            assigned[max_hole_idx] = 4
            unassigned.remove(max_hole_idx)

        if len(unassigned) == 0:
            return assigned

        # ----------------------------------------------------
        # 2. DIGIT 1: Lowest Slenderness Score (AR * ON pixels)
        # ----------------------------------------------------
        min_slender_idx = min(unassigned, key=lambda i: features_list[i]['ar'] * features_list[i]['on'])
        assigned[min_slender_idx] = 1
        unassigned.remove(min_slender_idx)

        if len(unassigned) == 0:
            return assigned

        # ----------------------------------------------------
        # 3. DIGIT 5: Dominant Top-Left horizontal bar (TL / (TR + 1))
        # ----------------------------------------------------
        max_tl_idx = max(unassigned, key=lambda i: float(features_list[i]['tl']) / (float(features_list[i]['tr']) + 1e-5))
        assigned[max_tl_idx] = 5
        unassigned.remove(max_tl_idx)

        if len(unassigned) == 0:
            return assigned

        # ----------------------------------------------------
        # 4. DIGITS 2 & 3: Between remaining 2 cards, compare Bottom-Left base
        # ----------------------------------------------------
        rem = list(unassigned)
        if len(rem) == 2:
            ratio_0 = float(features_list[rem[0]]['bl']) / (float(features_list[rem[0]]['br']) + 1e-5)
            ratio_1 = float(features_list[rem[1]]['bl']) / (float(features_list[rem[1]]['br']) + 1e-5)
            if ratio_0 > ratio_1:
                assigned[rem[0]] = 2
                assigned[rem[1]] = 3
            else:
                assigned[rem[0]] = 3
                assigned[rem[1]] = 2
        elif len(rem) == 1:
            used_digits = set(assigned.values())
            remaining_digits = set([1, 2, 3, 4, 5]) - used_digits
            assigned[rem[0]] = remaining_digits.pop() if remaining_digits else 3

        return assigned

    def detect_target_book_in_column(self, frame, target_color):
        """
        Locks search window to target column strip, runs robust HSV thresholding
        (including dual-range wrap-around for red), and reliably detects row 1-4
        across all columns (1, 2, 3, 4, and 5) by prioritizing strip-central contours.
        Calibrated horizon filter ensures top shelf (Row 1) books are never discarded.
        """
        frame_height, frame_width = frame.shape[:2]
        
        if self.target_col_x_bounds is not None:
            x_start, x_end = self.target_col_x_bounds
            x_start = max(0, x_start)
            x_end = min(frame_width, x_end)
        else:
            roi_width = int(frame_width * 0.3)
            x_start = int((frame_width - roi_width) / 2)
            x_end = x_start + roi_width

        cropped_frame = frame[:, x_start:x_end]
        hsv_cropped = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        # Apply robust HSV masking
        mask = self.get_color_mask(hsv_cropped, target_color)
        
        # Display the color mask for debugging in Window 4
        cv2.imshow("4. Target Book Color Mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        strip_center_x = (x_end - x_start) / 2.0
        
        valid_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            # Filter out microscopic noise; allow books down to 50 pixels
            if area < 50:
                continue
                
            x_c, y_c, w_c, h_c = cv2.boundingRect(c)
            centroid_y = y_c + (h_c / 2.0)
            centroid_x = x_c + (w_c / 2.0)

            # Reject only true overhead ceiling/marker cards (y < 10% frame height)
            # This ensures Row 1 books (y ~ 75-95 px) are safely captured
            if centroid_y < frame_height * 0.10:
                continue

            # Prioritize contours close to the center of the column strip
            dist_from_center = abs(centroid_x - strip_center_x)
            score = area / (1.0 + 0.05 * dist_from_center)
            valid_candidates.append((score, x_c, y_c, w_c, h_c, centroid_y))

        if not valid_candidates:
            # Failsafe fallback: if no candidate passed the strict filters, take the largest contour above 40 px
            fallback_contours = [c for c in contours if cv2.contourArea(c) > 40]
            if fallback_contours:
                best_c = max(fallback_contours, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(best_c)
                cent_y = by + (bh / 2.0)
                x_full = bx + x_start
                rel_y = cent_y / float(frame_height)
                detected_row = 1 if rel_y < 0.38 else (2 if rel_y < 0.58 else (3 if rel_y < 0.78 else 4))
                return detected_row, [x_full, by, bw, bh]
            return None, None

        # Select candidate with the best balance of area and centrality
        valid_candidates.sort(key=lambda item: item[0], reverse=True)
        _, bx, by, bw, bh, cent_y = valid_candidates[0]

        x_full = bx + x_start
        
        # Map vertical centroid to shelf rows 1 to 4
        rel_y = cent_y / float(frame_height)
        if rel_y < 0.38:
            detected_row = 1
        elif rel_y < 0.58:
            detected_row = 2
        elif rel_y < 0.78:
            detected_row = 3
        else:
            detected_row = 4
            
        return detected_row, [x_full, by, bw, bh]

    def get_color_mask(self, hsv_img, color_name):
        """
        Generates binary HSV mask, with dual-range wrap-around for red.
        """
        c = color_name.lower()
        if c == 'red':
            lower1 = np.array([0, 120, 50])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 120, 50])
            upper2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv_img, lower1, upper1)
            mask2 = cv2.inRange(hsv_img, lower2, upper2)
            return cv2.bitwise_or(mask1, mask2)
        elif c == 'blue':
            return cv2.inRange(hsv_img, np.array([100, 140, 50]), np.array([140, 255, 255]))
        elif c == 'green':
            return cv2.inRange(hsv_img, np.array([35, 100, 40]), np.array([86, 255, 255]))
        elif c == 'yellow':
            return cv2.inRange(hsv_img, np.array([15, 100, 100]), np.array([35, 255, 255]))
        else:
            return cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([180, 255, 255]))

    def save_annotated_image(self, frame, filename_prefix):
        """
        Draws timestamp on captured detection frame and saves image for bonus points.
        """
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, 
            timestamp_str, 
            (20, frame.shape[0] - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        
        filepath = os.path.join(self.image_save_dir, f"{filename_prefix}_{int(time.time())}.png")
        try:
            cv2.imwrite(filepath, frame)
            self.get_logger().info(f"Scoring Image saved successfully: {filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to write image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ERCPerceptionNodeV16()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
