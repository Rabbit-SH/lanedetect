import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Parser example')

parser.add_argument('--source', type=str, default='0', help='source')

args=parser.parse_args()

source=args.source



def region_of_interest(img, vertices):

    mask = np.zeros_like(img)

    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def get_fitline(img, f_lines):  # 대표선 구하기
    try:
        lines = np.squeeze(f_lines)

        if len(lines.shape) != 1:
            lines = lines.reshape(lines.shape[0] * 2, 2)
            rows, cols = img.shape[:2]
            output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = output[0], output[1], output[2], output[3]
            # 차선변경 에러

            x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
            x2, y2 = int(((img.shape[0] / 2 + 70) - y) / vy * vx + x), int(img.shape[0] / 2 + 70)

            result = [x1, y1, x2, y2]

            return result
    except:
        return None

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # 대표선 그리기
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

def offset(left, mid, right):

    LANEWIDTH = 3.7
    a = mid - left
    b = right - mid
    width = right - left

    if a >= b:  # driving right off
        offset = a / width * LANEWIDTH - LANEWIDTH / 2.0
    else:  # driving left off
        offset = LANEWIDTH / 2.0 - b / width * LANEWIDTH

    return offset
def image_process(image,vertice):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                                       np.array([vertice], np.int32))
    return cropped_image

def birdeye_view(frame, width, height):
    left_bottom = [0,height]
    right_bottom = [width,height]
    left_top = [int(width*0.332),int(height*0.7)]
    right_top = [int(width*0.65), int(height*0.7)]
    pts1 = np.float32([[left_top,left_bottom,right_top,right_bottom]])
    # 좌표의 이동점
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
    M = cv2.getPerspectiveTransform(pts1, pts2)
    b_v = cv2.warpPerspective(frame, M, (width, height))
    return b_v
def process(image):

        height = image.shape[0]
        width = image.shape[1]
        region_of_interest_vertices = [
            (0, height),
            (width*0.4, height/2),
            (width*0.8,height/2),
            (width, height)
        ]
        region_of_interest_vertices1 = [
            (0, height),
            (0, 0),
            (width, 0),
            (width, height)
        ]

        b_v=birdeye_view(image,width,height)

        cropped_image = image_process(image,region_of_interest_vertices)
        cropped_b_v=image_process(b_v,region_of_interest_vertices1)


        lines = cv2.HoughLinesP(cropped_b_v,
                                rho=2,
                                theta=np.pi/180,
                                threshold=50,
                                lines=np.array([]),
                                minLineLength=40,
                                maxLineGap=100)


        line_arr = np.squeeze(lines)


        # 기울기 구하기
        slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

        # 수평 기울기 제한
        line_arr = line_arr[np.abs(slope_degree) < 160]
        slope_degree = slope_degree[np.abs(slope_degree) < 160]
        # 수직 기울기 제한
        line_arr = line_arr[np.abs(slope_degree) > 95]
        slope_degree = slope_degree[np.abs(slope_degree) > 95]
        # 필터링된 직선 버리기
        L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
        temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        L_lines, R_lines = L_lines[:, None], R_lines[:, None]

        # 대표선 만들기
        left_fit_line = get_fitline(temp, L_lines)

        right_fit_line = get_fitline(temp, R_lines)


        if left_fit_line != None and right_fit_line != None:
            print(right_fit_line[0] - left_fit_line[0])




        #if left_fit_line != None:
            #draw_fit_line(temp, left_fit_line, color)

       # if right_fit_line != None:
           # draw_fit_line(temp, right_fit_line, color)
        if left_fit_line != None and right_fit_line != None:
             draw_poly(b_v,left_fit_line,right_fit_line)
        #draw_poly(image, left_fit_line, right_fit_line)

        cv2.imshow("aabb", b_v)
        image_with_lines = cv2.addWeighted(temp, 0.8, image, 1, 0.0)

        return image_with_lines

def draw_poly(img, left_lines,right_lines):
    try:
        left_lines=np.squeeze(left_lines)
        print(left_lines)
        right_lines=np.squeeze(right_lines)
        print(right_lines)

        l_bottom = [left_lines[0] , left_lines[1]]
        l_top = [left_lines[2], left_lines[3]]
        r_bottom = [right_lines[0] , right_lines[1]]
        r_top = [right_lines[2], right_lines[3]]

        cv2.fillConvexPoly(img,np.array([l_bottom,r_bottom,r_top,l_top],np.int32),(0,255,0))
    except:
        pass

#시작 함수
def Start():

    cap = cv2.VideoCapture("project_video.mp4")

    while cap.isOpened():

        ret, frame = cap.read()
        if ret is None:
            break

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = round(1000 / fps)
        try:
            frame = process(frame)
        except:
            pass
        cv2.imshow('frame', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


Start()

