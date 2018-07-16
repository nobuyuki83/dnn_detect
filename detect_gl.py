import cv2
import math

from OpenGL.GL import *

import detect

def loadTexture(img_bgr):
    #    img_bgr = img
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    #    img_rgb = img_bgr.copy()
    width_org = img_rgb.shape[1]
    height_org = img_rgb.shape[0]
    width_po2 = int(math.pow(2, math.ceil(math.log(width_org, 2))))
    height_po2 = int(math.pow(2, math.ceil(math.log(height_org, 2))))

#    print(height_org, width_org)
    img_rgb = cv2.copyMakeBorder(img_rgb,
                                 0, height_po2 - height_org, 0, width_po2 - width_org,
                                 cv2.BORDER_CONSTANT, (0, 0, 0))

    rw = width_org / width_po2
    rh = height_org / height_po2

    glEnable(GL_TEXTURE_2D)
    texid = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texid)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_po2, height_po2, 0, GL_RGB, GL_UNSIGNED_BYTE, img_rgb)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    return (width_org, height_org, rw, rh, texid)

def getPoint(name,dict_data):
    p = None
    if name in dict_data:
        s = dict_data[name]
        p = list( map(float,s.split(",")) )
    return p

def drawPoint(pos,color,point_size=5):
    if pos is None:
        return
    glPointSize(point_size)
    glColor3d(color[0], color[1], color[2])
    glBegin(GL_POINTS)
    glVertex2d(pos[0],-pos[1])
    glEnd()

def drawRect(rect,color=(1,0,0),width=1):
    plt = (rect[0], -rect[1])
    size_rect = rect[2]
    glColor3d(color[0], color[1], color[2])
    glLineWidth(width)
    glBegin(GL_LINE_LOOP)
    glVertex2f(plt[0], plt[1])
    glVertex2f(plt[0] + size_rect, plt[1])
    glVertex2f(plt[0] + size_rect, plt[1] - size_rect)
    glVertex2f(plt[0], plt[1] - size_rect)
    glEnd()


def drawListRect(list_rect: list, color, width):
    for rect in list_rect:
        assert isinstance(rect,list)
        assert len(rect) == 4
        drawRect(rect,color=color,width=width)

def drawLine(pos0,pos1,color=(1,1,1)):
    if pos0 is None:
        return
    if pos1 is None:
        return

    glLineWidth(1)
    glColor3d(color[0],color[1],color[2])
    glBegin(GL_LINES)
    glVertex2d(pos0[0],-pos0[1])
    glVertex2d(pos1[0],-pos1[1])
    glEnd()

def draw_img_annotation(img_size_info, dict_info):
    img_w = img_size_info[0]
    img_h = img_size_info[1]
    imgtex_w = img_size_info[2]
    imgtex_h = img_size_info[3]
    ####
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    id_tex_org = img_size_info[4]
    if id_tex_org is not None and glIsTexture(id_tex_org):
        glBindTexture(GL_TEXTURE_2D, id_tex_org)
    glColor3d(1, 1, 1)
    glBegin(GL_QUADS)
    ## left bottom
    glTexCoord2f(0.0, imgtex_h)
    glVertex2f(0, -img_h)
    ## right bottom
    glTexCoord2f(imgtex_w, imgtex_h)
    glVertex2f(img_w, -img_h)
    ### right top
    glTexCoord2f(imgtex_w, 0.0)
    glVertex2f(img_w, 0)
    ## left top
    glTexCoord2f(0.0, 0.0)
    glVertex2f(0, 0)
    glEnd()

    #####
    glDisable(GL_TEXTURE_2D)
    if "rect_face_cv" in dict_info:
        s = dict_info["rect_face_cv"]
        list_rect = list( map(float,s.split(",")) )
        list_rect = [list_rect[i:i + 4] for i in range(0, len(list_rect), 4)]
        if "no_face" in dict_info or "rect_face_manual" in dict_info or "rect_face_dnn" in dict_info:
            drawListRect(list_rect,color=(0,0,1),width=1)
        else:
            drawListRect(list_rect,color=(0,0,1),width=3)
    if "rect_face_manual" in dict_info:
        s = dict_info["rect_face_manual"]
        list_rect = list( map(float,s.split(",")) )
        list_rect = [list_rect[i:i + 4] for i in range(0, len(list_rect), 4)]
        if "no_face" in dict_info:
            drawListRect(list_rect,color=(1,0,0),width=1)
        else:
            drawListRect(list_rect,color=(1,0,0),width=3)
    if "rect_face_dnn" in dict_info:
        s = dict_info["rect_face_dnn"]
        list_rect = list( map(float,s.split(",")) )
        list_rect = [list_rect[i:i + 4] for i in range(0, len(list_rect), 4)]
        if "no_face" in dict_info or "rect_face_manual" in dict_info:
            drawListRect(list_rect,color=(0,1,0),width=1)
        else:
            drawListRect(list_rect,color=(0,1,0),width=3)

    p_rect_tl = None
    list_rect = detect.get_list_rect_from_dict_info(dict_info)
    if len(list_rect) > 0:
        p_rect_tl = [list_rect[0][0],list_rect[0][1]]

    p_nose_tip         = getPoint("nose_tip",         dict_info)
    p_nose_top         = getPoint("nose_top",         dict_info)
    p_lip_top          = getPoint("lip_top",          dict_info)
    p_lip_down         = getPoint("lip_down",         dict_info)
    p_right_eye_corner = getPoint("right_eye_corner", dict_info)
    p_right_eye_tail   = getPoint("right_eye_tail",   dict_info)
    p_left_eye_corner  = getPoint("left_eye_corner",  dict_info)
    p_left_eye_tail    = getPoint("left_eye_tail",    dict_info)

    drawPoint(p_nose_tip,         (1,0,1))
    drawPoint(p_nose_top,         (0,1,0))
    drawPoint(p_lip_top,          (0,1,1))
    drawPoint(p_lip_down,         (1,1,0))
    drawPoint(p_right_eye_corner, (1,1,0))
    drawPoint(p_right_eye_tail,   (1,0,1))
    drawPoint(p_left_eye_corner,  (1,1,0))
    drawPoint(p_left_eye_tail,    (1,0,1))

    drawLine(p_nose_tip,        p_nose_top      )
    drawLine(p_lip_top,         p_lip_down      )
    drawLine(p_right_eye_corner,p_right_eye_tail)
    drawLine(p_left_eye_corner, p_left_eye_tail )
    drawLine(p_right_eye_tail,   p_rect_tl)


# same function in face
def get_rect_train(dict_data):
    rect = None
    if "no_face" in dict_data:
        return None
    elif "rect_face_manual" in dict_data:
        rect = dict_data["rect_face_manual"]
    elif "rect_face_dnn" in dict_data:
        rect = dict_data["rect_face_dnn"]
    elif "rect_face_cv" in dict_data:
        rect = dict_data["rect_face_cv"]
    else:
        return None
    rect = list(map(float,rect.split(",")))
    rect = [rect[i:i + 4] for i in range(0, len(rect), 4)]
    print(rect)
    return rect


def get_img_coord(xy, img_size_info, dict_train_new, view_mode, list_rect):
    ####
    viewport = glGetIntegerv(GL_VIEWPORT)
    win_h = viewport[3]
    win_w = viewport[2]
    img_w = img_size_info[0]
    img_h = img_size_info[1]
    imgtex_w = img_size_info[2]
    imgtex_h = img_size_info[3]
    ####
    if view_mode == "ORIGINAL":
        scale_imgwin = max(img_h / win_h, img_w / win_w)
        x1 = xy[0]*scale_imgwin
        y1 = xy[1]*scale_imgwin
        return (x1, y1)
    elif view_mode.startswith("CENTER"):
        ir = int(view_mode.split("_")[1])
        assert ir < len(list_rect)
        rect = list_rect[ir]
        scale_imgwin = rect[3] / min(win_w, win_h) * 2
        trans_w = rect[0] + 0.5 * rect[2] - 0.5 * win_w * scale_imgwin
        trans_h = rect[1] + 0.5 * rect[3] - 0.5 * win_h * scale_imgwin
        return (xy[0]*scale_imgwin+trans_w, xy[1]*scale_imgwin+trans_h)
    ####
    assert False


def set_view_trans(img_size_info, dict_train_new, view_mode):
    viewport = glGetIntegerv(GL_VIEWPORT)
    win_h = viewport[3]
    win_w = viewport[2]
    img_w = img_size_info[0]
    img_h = img_size_info[1]
    imgtex_w = img_size_info[2]
    imgtex_h = img_size_info[3]
    #####
    if view_mode == "ORIGINAL":
        scale_imgwin = max(img_h / win_h, img_w / win_w)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, win_w * scale_imgwin, -win_h * scale_imgwin, 0, -1000, 1000)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    elif view_mode.startswith("CENTER"):
        list_rect = get_rect_train(dict_train_new)
        ir = int(view_mode.split("_")[1])
        assert ir < len(list_rect)
        rect = list_rect[ir]
        ####
        scale_imgwin = rect[3] / min(win_w, win_h) * 2
        trans_w = rect[0] + 0.5 * rect[2] - 0.5 * win_w * scale_imgwin
        trans_h = rect[1] + 0.5 * rect[3] - 0.5 * win_h * scale_imgwin
        ####
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #    glOrtho(0,win_w*scale_imgwin,-win_h*scale_imgwin,0,-10,10)
        glOrtho(trans_w,
                (win_w * scale_imgwin + trans_w),
                -(trans_h + win_h * scale_imgwin),
                -trans_h,
                -1000, 1000)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_LIGHTING)




