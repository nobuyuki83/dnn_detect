# import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *

import glob,random,pickle,datetime
import detect, detect_gl

# utility glut
def glut_print( x,  y,  font,  text, color):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glColor3f(color[0],color[1],color[2])
    glRasterPos2f(x,y)
    for ch in text :
        glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )

# -----------
# VARIABLES
# -----------

dict_path_csv_datetime = {}
path_csv_current = 0
img_size_info = (250, 250, 1, 1, -1)
dict_info = {}
rect_drag = None

edit_mode = "RECT_FACE"
view_mode = "ORIGINAL"


def set_id_path():
    global img_size_info, dict_info, view_mode
    ###
    img = detect.get_imgage_train(path_csv_current)
    dict_info = detect.get_csv_data(path_csv_current)
    ####
    id_tex_org = img_size_info[4]
    if id_tex_org is not None and glIsTexture(id_tex_org):
        glDeleteTextures(id_tex_org)
    img_size_info = detect_gl.loadTexture(img)
    if view_mode == "ORIGINAL":
        glutReshapeWindow(img_size_info[0], img_size_info[1])
    ####
    if "last_looked" in dict_info:
        print(path_csv_current,"  ",dict_info["last_looked"])
    else:
        print(path_csv_current)
    ####
    list_rect = detect.get_list_rect_from_dict_info(dict_info)
    if view_mode.startswith("CENTER"):
        ir = int(view_mode.split("_")[1])
        if ir >= len(list_rect):
            view_mode = "ORIGINAL"


def save(dt0: datetime):
    global dict_path_csv_datetime, dict_info
    dict_info["last_looked"] = str(dt0)
    dict_info["shape_img"] = str(img_size_info[1]) + "," + str(img_size_info[0])
    detect.save_csv_data(path_csv_current, dict_info)
    dict_path_csv_datetime[path_csv_current] = dt0


def get_oldest_path(dict_path_csv_datetime):
    while 1:
        path_csv_current = min(dict_path_csv_datetime, key=dict_path_csv_datetime.get)
        if os.path.isfile(path_csv_current):
            break
        else:
            del dict_path_csv_datetime[path_csv_current]
    return path_csv_current


def get_newest_path(dict_path_csv_datetime):
    while 1:
        path_csv_current = max(dict_path_csv_datetime, key=dict_path_csv_datetime.get)
        if os.path.isfile(path_csv_current):
            break
        else:
            del dict_path_csv_datetime[path_csv_current]
    return path_csv_current


def load_file(path_dir):
    global dict_path_csv_datetime

    list_path_img = []
    list_path_img = glob.glob(path_dir + '/*.jpg', recursive=True) + list_path_img
    list_path_img = glob.glob(path_dir + '/*.png', recursive=True) + list_path_img
    print(list_path_img)
    for path_img in list_path_img:
        path_without_ext = path_img.rsplit(".")[0]
        if os.path.isfile(path_without_ext+".csv"):
            continue
        file = open(path_without_ext+".csv","w")

    ###
    dict_path_csv_datetime = {}
    if os.path.isfile(path_dir+"/dict_path_csv__datetime.p"):
        dict_path_csv_datetime = pickle.load(open(path_dir + "/dict_path_csv__datetime.p", "rb"))

    list_path_csv = glob.glob(path_dir + "/**/*.csv", recursive=True)
    print("number of csv files:",len(list_path_csv))

    for path_csv in list_path_csv:
        if path_csv in dict_path_csv_datetime:
            continue
        dict_data = detect.get_csv_data(path_csv)
        if "last_looked" in dict_data:
            dt0 = datetime.datetime.strptime(dict_data["last_looked"],'%Y-%m-%d %H:%M:%S.%f')
        else:
            dt0 = datetime.datetime.now()
            dt0 -= datetime.timedelta(days=random.randint(10, 20))
            dt0 += datetime.timedelta(seconds=random.randint(0, 60))
        dict_path_csv_datetime[path_csv] = dt0
    pickle.dump(dict_path_csv_datetime, open(path_dir + "/dict_path_csv__datetime.p", "wb"))
    print("load dict finished")


# -------------------
# SCENE CONSTRUCTOR
# -------------------


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    detect_gl.set_view_trans(img_size_info, dict_info, view_mode)
    detect_gl.draw_img_annotation(img_size_info, dict_info)
    if rect_drag is not None:
        detect_gl.drawRect(rect_drag, color=(1, 0, 0))

    #    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glColor3d(1,1,1)
    glBegin(GL_QUADS)
    glVertex3d(-1.0,-1.0,-0.2)
    glVertex3d(   0,-1.0,-0.2)
    glVertex3d(   0,-0.8,-0.2)
    glVertex3d(-1.0,-0.8,-0.2)
    glEnd()
    glut_print(-0.9,-1.0, GLUT_BITMAP_9_BY_15, edit_mode, (1,0,0) )
    if len(detect.get_list_rect_from_dict_info(dict_info)) == 0:
        glut_print(-0.9,-0.9, GLUT_BITMAP_9_BY_15, "no face", (0,0,1) )
#    glEnable(GL_DEPTH_TEST)

    glutSwapBuffers()

def reshape(width, height):
    glViewport(0, 0,width, height)

def keyboard(bkey, x, y):
    global view_mode, dict_info, edit_mode, path_csv_current, dict_path_csv_datetime
    key = bkey.decode("utf-8")
    dt0 = datetime.datetime.now()
    if key == 'w':
        pickle.dump(dict_path_csv_datetime, open(path_dir + "/dict_path_csv__datetime.p", "wb"))
    if key == "W":
        os.remove(path_dir + "/dict_path_csv__datetime.p")
        load_file(path_dir)
        path_csv_current = get_oldest_path(dict_path_csv_datetime)
        set_id_path()
    if key == 'd':
        save(dt0)
        ###
        path_csv_current = get_oldest_path(dict_path_csv_datetime)
        set_id_path()
    if key == 'D':
        dt1 = dt0 - datetime.timedelta(days=random.randint(40,100))
        save(dt1)
        ###
        path_csv_current = get_newest_path(dict_path_csv_datetime)
        set_id_path()
    if key == 'c':
        if edit_mode == "LIP_TOP-DOWN":
            dict_info.pop("lip_top",None)
            dict_info.pop("lip_down",None)
        if edit_mode == "NOSE_TIP":
            dict_info.pop("nose_tip",None)
        if edit_mode == "NOSE_TOP":
            dict_info.pop("nose_top",None)
        if edit_mode == "RIGHT_EYE_CORNER-TAIL":
            dict_info.pop("right_eye_corner")
            dict_info.pop("right_eye_tail",None)
        if edit_mode == "LEFT_EYE_CORNER-TAIL":
            dict_info.pop("left_eye_corner",None)
            dict_info.pop("left_eye_tail",None)
        if edit_mode == "RECT_FACE":
            dict_info.pop("rect_face_manual",None)
    if key == 'X': # move to trash box
        print(path_csv_current)
        dir_dist = os.path.abspath(os.path.dirname(path_csv_current)+"/../xtrash")
        csv_data = detect.get_csv_data(path_csv_current)
        base_md5 = path_csv_current.rsplit(".",1)[0]
        ext = ".png"
        if not os.path.isfile(base_md5+ext):
            ext = ".jpg"
        assert os.path.isfile(base_md5+ext)
        if "url_name" in csv_data:
            url_name = csv_data["url_name"]
            print(ext,base_md5, url_name)
            os.remove(path_csv_current)
            os.rename(base_md5+ext,dir_dist+"/"+url_name)
            print("move to trash", dir_dist+"/"+url_name)
        else:
            os.remove(path_csv_current)
            os.remove(base_md5+ext)
        del dict_path_csv_datetime[path_csv_current]
        path_csv_current = min(dict_path_csv_datetime, key=dict_path_csv_datetime.get)
        set_id_path()
    if key == '`':
        if not "no_face" in dict_info:
            dict_info["no_face"] = "True"
        else:
            del dict_info["no_face"]
    if key == '1':
        edit_mode = "LIP_TOP-DOWN"
    if key == '2':
        edit_mode = "NOSE_TIP"
    if key == '3':
        edit_mode = "RIGHT_EYE_CORNER-TAIL"
    if key == '4':
        edit_mode = "LEFT_EYE_CORNER-TAIL"
    if key == '5':
        edit_mode = "NOSE_TOP"
    if key == '0':
        edit_mode = "RECT_FACE"
    if key == ' ':
        list_rect = detect.get_list_rect_from_dict_info(dict_info)
        print(len(list_rect))
        if view_mode == "ORIGINAL":
            if len(list_rect) > 0 :
                view_mode = "CENTER_0"
                glutReshapeWindow(600, 600)
        else:
            assert view_mode.startswith("CENTER")
            ir = int(view_mode.split("_")[1])
            if ir+1 >= len(list_rect):
                view_mode = "ORIGINAL"
                glutReshapeWindow(img_size_info[0], img_size_info[1])
            else:
                view_mode = "CENTER_"+str(ir+1)
                glutReshapeWindow(600, 600)
        print(view_mode)
    if key == 'q':
        exit()
    glutPostRedisplay()


def mouse(button, state, x, y):
    global rect_drag
    list_rect = detect.get_list_rect_from_dict_info(dict_info)
    xy1 = detect_gl.get_img_coord((x, y), img_size_info, dict_info, view_mode, list_rect)
#    print("mouse",x,y,state,xy1)
    if edit_mode == "NOSE_TIP":
        if state == GLUT_DOWN:
            dict_info["nose_tip"] = str(xy1[0]) + "," + str(xy1[1])
    if edit_mode == "NOSE_TOP":
        if state == GLUT_DOWN:
            dict_info["nose_top"] = str(xy1[0]) + "," + str(xy1[1])
    ####
    if edit_mode == "LIP_TOP-DOWN":
        if state == GLUT_DOWN:
            dict_info["lip_top"] = str(xy1[0]) + "," + str(xy1[1])
        if state == GLUT_UP:
            dict_info["lip_down"] = str(xy1[0]) + "," + str(xy1[1])
    ####
    if edit_mode == "RIGHT_EYE_CORNER-TAIL":
        if state == GLUT_DOWN:
            dict_info["right_eye_corner"] = str(xy1[0]) + "," + str(xy1[1])
        if state == GLUT_UP:
            dict_info["right_eye_tail"] = str(xy1[0]) + "," + str(xy1[1])
    if edit_mode == "LEFT_EYE_CORNER-TAIL":
        if state == GLUT_DOWN:
            dict_info["left_eye_corner"] = str(xy1[0]) + "," + str(xy1[1])
        if state == GLUT_UP:
            dict_info["left_eye_tail"] = str(xy1[0]) + "," + str(xy1[1])
    if edit_mode == "RECT_FACE":
        if state == GLUT_DOWN:
            rect_drag = (xy1[0],xy1[1],5,5)
        if state == GLUT_UP:
            list_rect_manual = []
            if "rect_face_manual" in dict_info:
                list_rect_manual = detect.get_list_rect_from_string(dict_info["rect_face_manual"])
            ir_overlap = None
            for ir, rect in enumerate(list_rect_manual):
                iou = detect.iou_rect(rect,rect_drag)
                if iou > 0.25:
                    ir_overlap = ir
                    break
            if ir_overlap is not None:
                list_rect_manual[ir_overlap] = rect_drag
            else:
                list_rect_manual.append(rect_drag)
            print(list_rect_manual)
            dict_info["rect_face_manual"] = detect.get_str_csv_list_rect(list_rect_manual)
            rect_drag = None
    glutPostRedisplay()


def motion(x, y):
    global rect_drag
    list_rect = detect.get_list_rect_from_dict_info(dict_info)
    xy1 = detect_gl.get_img_coord((x, y), img_size_info, dict_info, view_mode, list_rect)
#    print("move",edit_mode,rect_drag,x,y,xy1)
    if edit_mode == "RECT_FACE" and rect_drag is not None:
        size_x = abs(xy1[0] - rect_drag[0])
        size_y = abs(xy1[1] - rect_drag[1])
        size = max(size_x,size_y)
        rect_drag = (rect_drag[0],rect_drag[1],size,size)
#        print("rect_drag",rect_drag)
    glutPostRedisplay()


# ------
# MAIN
# ------
if __name__ == "__main__":
    # GLUT Window Initialization
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)  # zBuffer
    glutInitWindowSize(600, 600)
    glutInitWindowPosition(100, 100)
    glutCreateWindow("Anotator")
    # Register callbacks
    glutReshapeFunc(reshape)
    glutDisplayFunc(display)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutKeyboardFunc(keyboard)
    # Turn the flow of control over to GLUT

    if len(sys.argv) != 2:
        print("missing argument")
    path_dir = sys.argv[1]
    assert os.path.isdir(path_dir)

    load_file(path_dir)

    path_csv_current = get_oldest_path(dict_path_csv_datetime)
    print(dict_path_csv_datetime[path_csv_current])


    set_id_path()

    glutMainLoop()
