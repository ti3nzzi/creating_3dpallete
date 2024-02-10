import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import open3d as o3d
import copy
from shapely.geometry.polygon import Polygon
import math
import random
import matplotlib.colors
import os

from database import DataBaseManager

def preprocesimg(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    # hsv[:,:,1] = hsv[:,:,1]*1.25
    hsv[:, :, 1] = hsv[:, :, 1] * 1.75
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    # hsv[:, :, 2] = hsv[:,:,2]*1.25
    hsv[:, :, 2] = hsv[:, :, 2] * 1.75
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

#'depth_pic/0/0_Depth.raw'
def get_depth_data(dfilename, csvfilename, contour, kscale):
    depth_dt = np.fromfile(dfilename, dtype=np.uint16)

    csv_file = open(csvfilename, "r")
    csv_data = csv_file.read()
    #Intrinsic:,
    #Fx, 296.547302
    #Fy, 296.547302
    #PPx, 156.458435
    #PPy, 119.243950
    #Resolution x, 320
    #Resolution y, 240

    Fx = 296.547302
    Fy = 296.547302
    PPx = 156.458435
    PPy = 119.243950
    Resolution_x = 320
    Resolution_y = 240

    calc_dep = 0
    count_c_points = 0
    #rs_contour = []

    for i in range(Resolution_x):
        for j in range(Resolution_y):
            scx = j*kscale
            scy = i*kscale
            if cv2.pointPolygonTest(contour, (scx, scy), False) >= 0:
                depth = depth_dt[i*Resolution_x+j]
                X = depth * (j - PPx) / Fx
                Y = depth * (i - PPy) / Fy
                Z = depth

                #rs_contour.append((X,Y))
                calc_dep = calc_dep + Z
                count_c_points = count_c_points + 1

    #rect = cv2.minAreaRect(np.array(rs_contour))

    if count_c_points > 0:
        calc_dep = calc_dep / count_c_points

    #return [calc_dep]
    return calc_dep

def find_box(img_filename1, img_filename2, index):
    # Use a breakpoint in the code line below to debug your script.
    # Load images as grayscale
    #image1 = cv2.imread(filename1, 0)
    #image2 = cv2.imread(filename2, 0)

    image1 = cv2.imread(img_filename1)
    image2 = cv2.imread(img_filename2)

    image1 = preprocesimg(image1)
    image2 = preprocesimg(image2)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(image1, image2, full=True)
    #diff = cv2.subtract(image1, image2)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] image1 we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    #print("Image Similarity: {:.4f}%".format(score * 100))

    #cv2.imshow('diff', diff)
    #cv2.waitKey()

    tfilename = os.path.join('diff', 'diff' + str(index) + '.png')
    cv2.imwrite(tfilename, diff)

    image = cv2.imread(tfilename)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Output2", img_gray)
    ret, im = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("Output3", im)
    #    RETR_EXTERNAL: int
    #    RETR_LIST: int
    #    RETR_CCOMP: int
    #    RETR_TREE: int
    #    RETR_FLOODFILL: int

    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height = diff.shape[0]
    width = diff.shape[1]
    #blank_image = np.zeros((height, width, 3), np.uint8)
    #img2 = cv2.drawContours(blank_image, contours, -1, (255, 0, 0), 1)
    # cv2.imshow("Output5", img2)
    # show_image(img2)
    # print(contours)

    ############################################

    cnts = contours
    #image2 = np.zeros((height, width, 3), np.uint8)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    rarea = -1
    rrect = []

    for c in cnts:
        # x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #print(c)
        # compute rotated rectangle (minimum area)
        rect = cv2.minAreaRect(c)
        #(x, y), (width, height), angle = rect
        #area = width * height
        #height = diff.shape[0]
        #width = diff.shape[1]
        #print(diff.shape[0])
        #print(diff.shape[1])
        box = cv2.boxPoints(rect)
        rect_contour = numpy.array([[box[0][0], box[0][1]], [box[1][0], box[1][1]], [box[2][0], box[2][1]], [box[3][0], box[3][1]]], dtype=numpy.int32)
        marea = 0
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                if cv2.pointPolygonTest(rect_contour, (j, i), False) >= 0:
                    marea =  marea + 255 - diff[i,j]

        if marea > rarea:
            rarea = marea
            rrect = copy.deepcopy(rect)

        #box = cv2.boxPoints(rect)
        #box = np.int0(box)
        # draw minimum area rectangle (rotated rectangle)
        # img = cv2.drawContours(image, [box], 0, (0, 255, 255), 2)
        #img = cv2.drawContours(image2, [box], 0, (0, 255, 255), 2)

    box = cv2.boxPoints(rrect)
    box = np.int0(box)
    img = cv2.drawContours(image2, [box], 0, (0, 255, 255), 2)
    #print(rrect)
    cv2.imwrite(os.path.join("cs", "cs" + str(index) + '.png'), img)

    return copy.deepcopy(rrect)

# posz, pts, h
def draw_cubes(cubedata):
    meshes = []
    count_box = len(cubedata)
    for i in range(count_box):
        mesh_box = o3d.geometry.TriangleMesh()
        posz = cubedata[i][0]
        pts = cubedata[i][1]
        h = cubedata[i][2]
        np_vertices = np.array([
            [pts[0][0], pts[0][1], posz],
            [pts[1][0], pts[1][1], posz],
            [pts[2][0], pts[2][1], posz],
            [pts[3][0], pts[3][1], posz],
            [pts[0][0], pts[0][1], posz+h],
            [pts[1][0], pts[1][1], posz+h],
            [pts[2][0], pts[2][1], posz+h],
            [pts[3][0], pts[3][1], posz+h]
        ])
        np_triangles = np.array([[2, 1, 0], [2, 0, 3],
                                 [4, 5, 6], [4, 6, 7],
                                 [0, 4, 3], [3, 4, 7],
                                 [2, 6, 5], [2, 5, 1],
                                 [1, 5, 4], [1, 4, 0],
                                 [3, 7, 2], [7, 6, 2],
                                 ]).astype(np.int32)
        mesh_box.vertices = o3d.utility.Vector3dVector(np_vertices)
        mesh_box.triangles = o3d.utility.Vector3iVector(np_triangles)
        #open3d.visualization.draw_geometries([mesh])
        mesh_box.compute_vertex_normals()
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        #mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
        mesh_box.paint_uniform_color(matplotlib.colors.to_rgb(color))
        #mesh_tx = copy.deepcopy(mesh).translate((1.3,0,0))
        #mesh_box.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, cubedata[i][6], 0]))
        #mesh_box.translate((cubedata[i][3], cubedata[i][4], cubedata[i][5]))
        meshes.append(mesh_box)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    meshes.append(mesh_frame)

    #print("We draw a few primitives using collection.")
    o3d.visualization.draw_geometries(meshes)

    #print("We draw a few primitives using + operator of mesh.")
    #o3d.visualization.draw_geometries(
    #    [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])

class ResultProcess:
    result_box_arr = []
    zero_depth = [0.0]

    @classmethod
    def process_pair(cls, filename1, filename2, filedepth_raw, filedepth_csv, index):
        rrect = find_box(filename1,filename2, index)
        box = cv2.boxPoints(rrect)
        contour = numpy.array(
            [[box[0][0], box[0][1]], [box[1][0], box[1][1]], [box[2][0], box[2][1]], [box[3][0], box[3][1]]],
            dtype=numpy.int32)
        dep_obj_data = get_depth_data(filedepth_raw, filedepth_csv, contour, 2)
        p1 = Polygon([(box[0][0], box[0][1]), (box[1][0], box[1][1]), (box[2][0], box[2][1]), (box[3][0], box[3][1])])
        prev_dep = cls.zero_depth[0]
        for item in cls.result_box_arr:
            itr = p1.intersection(item[0])
            if (itr.area/item[0].area) > 0.5:
                if item[1] < prev_dep:
                    prev_dep = item[1]
        box_height = prev_dep - dep_obj_data
        cls.result_box_arr.append([p1, dep_obj_data, box_height, cls.zero_depth[0] - prev_dep])

def euclidean_distance( l1,l2):
    return math.sqrt((l2[0]-l1[0])**2 + (l2[1]-l1[1])**2)

def get_angle(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    l = math.sqrt(x*x + y*y)
    angle = math.asin(x/l)
    return angle

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(o3d.__version__)

    contour = numpy.array(
        [[210,160], [420,160], [420,320], [210,320]],
        dtype=numpy.int32)
    dep_obj_data = get_depth_data('depth_pic/0/0_Depth.raw', 'depth_pic/0/0_Depth_metadata.csv', contour, 2)
    print("base depth: " + str(dep_obj_data))
    zero_depth = dep_obj_data

    hvr_photo_filenames = ['pic/0_Color.png','pic/1/1_Color.png','pic/1/2_Color.png','pic/1/3_Color.png',
                            'pic/1/4_Color.png','pic/1/5_Color.png','pic/1/6_Color.png','pic/1/7_Color.png',
                            'pic/1/8_Color.png','pic/1/9_Color.png']

    depth_raw_filenames = ['depth_pic/0/0_Depth.raw', 'depth_pic/1/1_Depth.raw', 'depth_pic/2/2_Depth.raw',
                           'depth_pic/3/3_Depth.raw', 'depth_pic/4/4_Depth.raw', 'depth_pic/5/5_Depth.raw',
                           'depth_pic/6/6_Depth.raw', 'depth_pic/7/7_Depth.raw', 'depth_pic/8/8_Depth.raw',
                           'depth_pic/9/9_Depth.raw']

    depth_csv_filenames = ['depth_pic/0/0_Depth_metadata.csv', 'depth_pic/1/1_Depth_metadata.csv', 'depth_pic/2/2_Depth_metadata.csv',
                           'depth_pic/3/3_Depth_metadata.csv', 'depth_pic/4/4_Depth_metadata.csv', 'depth_pic/5/5_Depth_metadata.csv',
                           'depth_pic/6/6_Depth_metadata.csv', 'depth_pic/7/7_Depth_metadata.csv', 'depth_pic/8/8_Depth_metadata.csv',
                           'depth_pic/9/9_Depth_metadata.csv']

    rp = ResultProcess()
    rp.zero_depth[0] = zero_depth
    for i in range(9):
        rp.process_pair(hvr_photo_filenames[i], hvr_photo_filenames[i+1],
                        depth_raw_filenames[i+1], depth_csv_filenames[i+1], i)

    for i in range(9):
        print(rp.result_box_arr[i])

    cubes_data = []

    #[p1, dep_obj_data, box_height, cls.zero_depth[0] - dep_obj_data])
    for i in range(9):
        pts = rp.result_box_arr[i][0].exterior.coords
        h = rp.result_box_arr[i][2]
        posz = rp.result_box_arr[i][3]
        vec1 = [ posz, pts, h ]
        print(vec1)
        cubes_data.append(vec1)

    db = DataBaseManager()
    for i in range(9):
        pts = rp.result_box_arr[i][0].exterior.coords
        w1 = euclidean_distance(pts[0], pts[1])
        w2 = euclidean_distance(pts[0], pts[3])
        h = rp.result_box_arr[i][2]
        db.insert_results(w1, h, w2, w1 * h * w2, str(rp.result_box_arr[i][0]))

    draw_cubes(cubes_data)

    #draw_cubes([[2, 0.5, 1, 0, 0, 1, 0]
    #            ])

    cv2.waitKey()
