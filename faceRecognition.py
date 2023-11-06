# ==============================Importing Libraries================================================= #

import face_recognition
import numpy as np
from datetime import datetime
import os
import cv2
import keyboard
import pyautogui
import customtkinter as cstk
import tkinter as tk
from tkinter import filedialog, PhotoImage

# =======================================Paths===================================================== #

img_dir_path = r"ImagesAttendance"
only_name = r"only_name"
attend_csv_path = r"Attendance.csv"

# ========================================Variables=============================================== #

images = []  #  img to numpy array
global image_names
global filesz
global encodeList
encodeList=[]
filesz=tuple()
image_names = []  # stores people's namesz
mylist = os.listdir(img_dir_path)  # lists all the images in dir
savedImg = []
global attend_dict
attend_dict={}
print(mylist)
global del_names,del_ind
del_names=[]
del_ind=[]

# =======================================Accessing Image========================================== #

def access():
    """
    Loads images and their respective names from the specified directory (path).
    Reads images from the 'ImagesAttendance' directory, converting them to numpy arrays for facial recognition purposes.
    Extracts the base names of the images and stores them in the 'image_names' variable.

    Returns:
    - images: List of loaded images converted to numpy arrays.
    - image_names: List containing the base names of the loaded images for reference.

    Note: This function relies on the 'path' variable pointing to the 'ImagesAttendance' directory.
    """
    global images,image_names
    for cl in mylist:
        curImg = cv2.imread(f'{img_dir_path}/{cl}')
        images.append(curImg)
        image_names.append(os.path.splitext(cl)[0]) #root path of name [0] ext path [1]
    print(image_names)
    image_names2 = image_names[:]

def clean():
    """
    Clears the contents of the 'only_name' directory.

    This function is intended to empty the directory used to store individual names extracted from images for facial recognition.

    Note: It assumes the existence of the 'only_name' directory in the current working environment.
    """
    for f in os.listdir(only_name):
        os.remove(fr"{only_name}\{f}")

# ================================Save the Captured Image========================================= #

def save_img(imagesz,nami):
    """
    Saves the captured image represented by the 'imagesz' numpy array with the given name 'nami'.

    Parameters:
    - imagesz: Numpy array representing the captured image.
    - nami: Name assigned to the captured image.

    Checks if the provided name 'nami' is not already in the 'only_name' directory. If the name is not present,
    it saves the image as a file with the provided name in the 'only_name' directory.

    Note: The function relies on the existence of the 'only_name' directory for storing the captured images.
    """
    savedImg=os.listdir(only_name)
    if nami not in savedImg:
        cv2.imwrite(rf"{only_name}+\{nami}.jpg", imagesz)

# =========================================Image Encoding========================================== #

def find_encodings(images):
    """
    Encodes facial features from a list of images.

    Parameters:
    - images: List containing images in the form of numpy arrays.

    Utilizes the 'face_recognition' library to detect facial locations and encode facial features in the provided images.
    It extracts the facial encodings for all detected faces in the images and constructs a list of encodings.

    Returns:
    - encodeList: List of facial encodings for the detected faces in the input images.

    Note: This function depends on the 'face_recognition' library for facial detection and encoding.
    """
    encodeList = []
    for img in images:
        face_locations = face_recognition.face_locations(img)  # Find face locations
        if len(face_locations) > 0:
            encode = face_recognition.face_encodings(img, face_locations)[0]
            encodeList.append(encode)
        else:
            # Handle the case where no face is detected in the image
            print("No face detected in the image.")
    return encodeList

# =======================================Marking Attendance======================================= #

def markAttendance(name):
    """
    Records the attendance of a person by marking their entry time or updating their exit time in the 'Attendance.csv' file.

    Parameters:
    - name: Name of the person whose attendance is being recorded.

    This function updates the 'Attendance.csv' file by marking the entry time if the person's name is not already
    present in the attendance record. If the person's name already exists in the record, it updates their exit time.
    The time stamps are recorded in a specific format within the CSV file.

    Note: The function relies on the 'Attendance.csv' file for attendance record keeping.
    """
    print(name, "attended")

    with open("Attendance.csv", 'r+') as f:
        myDataList = f.readlines()  # reads every line in attendance list

        for line in myDataList:
            line = line.strip()
            entry = line.split(',')
            attend_dict[entry[0]] = entry[1:]

        if name not in attend_dict.keys():
            now = datetime.now()
            dtString = now.strftime("%I:%M %p")  # I - 12 hr format() , minute , pm or am
            attend_dict[name] = [dtString,""]  # writes time

        elif name in attend_dict.keys():
            now = datetime.now()
            dtString = now.strftime("%I:%M %p")  # I - 12 hr format() , minute , pm or am
            attend_dict[name][1]=dtString

# ===============================================Camera Analysis================================== #

def webcam_scan():
    """
    Initiates the webcam for real-time facial recognition and attendance marking.

    Utilizes the system's webcam to capture live video feed for facial detection and recognition.
    It detects faces in the video feed, matches them with known faces from loaded images, and marks the attendance
    for recognized individuals in real-time. It displays the video feed with detected faces and respective names.

    The function continues to capture and analyze the webcam feed until the 'q' key is pressed.

    Note: This function relies on the 'face_recognition' library, OpenCV, and other related libraries for facial detection
    and marking attendance based on the recognized faces in the video feed.
    """
    cap = cv2.VideoCapture(0) # starts video capture through webcam
    while True:
        # img = numpy array  ,  succces= if loaded or not
        success,img = cap.read()
        # we resizze to 1/4th of size of ease of calculation and faster read time
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # no of faces in an frame
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

        # displays a text below  no               co ordi where tot          font                colour    size
        cv2.putText(img,f'Number of faces detected: {len(facesCurFrame)}', (100, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        # main 
        for encodeFace,FaceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace,tolerance=0.5) # lower is more strict
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            matchIndex = np.argmin(faceDis) # gives matchIndex of match name out of all images
            if matches[matchIndex]:
                name = image_names[matchIndex].upper() # Capitalizes each word
                # print(name)
                # FaceLoc = up right down left
                y1,x2,y2,x1=FaceLoc
                # multiply by 4 cuz we decresed the size by 4
                # were drwaing on regular image not of reduced size one
                y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4

                # draw's rectangle img , loc , colour , size
                cv2.rectangle(img, (x1,y1),(x2,y2) ,(255, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2),(255, 255, 0), cv2.FILLED)
                # displays name
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                # save img
                save_img(img, name)

                # call's attendace to add name
                markAttendance(name)

            else:
                    name = "UNKNOWN"
                    # FaceLoc = up right down left
                    y1, x2, y2, x1 = FaceLoc
                    # multiply by 4 cuz we decresed the size by 4
                    # were drwaing on regular image not of reduced size one
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    # draw's rectangle img , loc , colour , size
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 0), cv2.FILLED)
                    # displays name
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

        # continouly displays the image
        cv2.imshow('webcam',img)
        cv2.waitKey(1)

        if keyboard.is_pressed('q'):
            print("i quit!!")
            cv2.destroyWindow('webcam')
            break


# ============================================Attendance on File============================================== #

def attendance():
    """
    Manages and updates the attendance record in the 'Attendance.csv' file.

    This function prepares and manages the content of the 'Attendance.csv' file by formatting the entries
    for each individual present. It calculates the time spent based on entry and exit times.

    Returns:
    - The 'Attendance.csv' file is updated with individual attendance entries and time spent.

    Note: The function relies on the 'Attendance.csv' file for attendance record management.
    """
    ff = open("Attendance.csv", 'w+')
    ss = ""
    try:
        ff.writelines("NAME,ENTRY,EXIT,TIME_SPENT_IN_MIN")
        ff.writelines("\n")
        del attend_dict['NAME']
        del attend_dict['UNKNOWN']
    except KeyError:
        print()

    for i in (attend_dict.keys()):
        ss += i
        entryy=attend_dict[i][0]
        exitt=attend_dict[i][1]
        try:
            ts=(int(exitt[3:-3]) - int(entryy[3:-3])) + (60*(int(exitt[:2]) - int(entryy[:2]) ))
            ss += "," + entryy + "," + exitt + "," + str(ts)
            ff.writelines(ss)
            ff.writelines("\n")
        except ValueError:
            print()

        ss = ""

    ff.close()
    os.startfile(r"Attendance.csv")

# ===================================================Delete Picture=========================================== #


def open_images_to_delete():
    """
    Allows the user to select and delete image files from the 'ImagesAttendance' directory.

    Utilizes a file dialog to enable the selection of image files for deletion. The function presents a file dialog window,
    allowing the user to choose image files. Once selected, the chosen image files are deleted from the 'ImagesAttendance'
    directory. The function also updates the 'image_names' list by marking the deleted images as 'unknown'.

    Note: This function depends on the 'ImagesAttendance' directory for managing image files.
    """
    L1 = image_names
    L2 = []
    li2 = os.listdir(r"ImagesAttendance")
    filesz = filedialog.askopenfilenames(title = "Select image files", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print("Selected files:", filesz)
    for xx in filesz:
        os.remove(xx)
        xx = os.path.splitext(xx[xx.find('nce') + 4:])[0]
        #set_dif.append(os.path.splitext(xx)[0])
        del_ind.append(L1.index(xx))
        del_names.append(image_names[L1.index(xx)])
        image_names[L1.index(xx)] = "unknown"
        print("removed : ", xx)

    set_dif = []
    for x in li2:
        L2.append(os.path.splitext(x)[0])
    set_dif = list(set(L1).symmetric_difference(set(L2)))
    set_dif = list(filter(lambda t: t != "unknown", set_dif))
    removed_names = ""
    for j in set_dif:
        removed_names += j + " , "
    tk.messagebox.showinfo("showinfo", f"Faces removed = {len(set_dif)}\n{removed_names}\nClose the Window")

# =========================================================GUI=================================================#

def delete_a_face():
    """
    Opens a separate window to manage and delete image files associated with known faces.

    Initiates a new GUI window allowing the user to interact with the 'open_images_to_delete()' function.
    The function provides an interface for selecting and deleting specific image files linked to known faces.
    Upon selection, the chosen image files are deleted, and their corresponding names in the 'image_names' list are updated to 'unknown'.

    Note: This function is dependent on 'open_images_to_delete()' for the actual deletion and name updating process.
    """
    root1 = tk.Toplevel()
    root1.geometry("600x600")
    root1.title("delete")
    image2 = PhotoImage(file=r'other_files\delete.png')
    bg1label = tk.Label(root1, image=image2, width=300, height=180)
    bg1label.pack()
    button9 = tk.Button(root1, text="Select the images", command=open_images_to_delete, width=300,pady=5)
    button9.pack()
    root1.mainloop()

def show():
    """
    Opens the directory where images of known faces are stored.

    This function opens the directory location where images associated with known faces are stored.
    It enables the user to view the images related to recognized individuals for reference or management purposes.

    Note: The function's behavior depends on the system's default file viewer to display the contents of the 'ImagesAttendance' directory.
    """
    os.startfile(r"only_name")

def know_faces():
    """
    Opens the directory containing images of known faces for reference.

    This function initiates the opening of the directory location where images of known faces are stored.
    It allows users to access and view images associated with recognized individuals for reference purposes.

    Note: The function depends on the system's default file viewer to display the contents of the 'ImagesAttendance' directory.
    """
    os.startfile(r"ImagesAttendance")

def about():
    """
    Opens and displays an 'about' image or file for reference.

    This function initiates the opening and display of an 'about' image or file for reference or informational purposes.
    It allows users to access specific content related to the system or application.

    Note: The function's behavior relies on the system's default application for handling the file type specified in the path.
    """
    os.startfile(r"other_files\about.png")

clean() # empty the known images folder
access() # get the names of images
encodeListKnown = find_encodings(images) # encode all the images
print("Encoding Completed..")

cstk.set_appearance_mode("dark")
cstk.set_default_color_theme("green")
root = cstk.CTk()
root.geometry("1320x720")
root.title("Facial Recognition System")

imag = tk.PhotoImage(file=r"other_files\bg4.png")

frame = cstk.CTkFrame(master=root)
frame.pack(padx=60,pady=20,fill="both",expand=True)

label = cstk.CTkLabel(master=frame,text="Facial Recognition System",font=("Roboto",24),compound="left")
label.pack(pady=12,padx=10)

bglabel = cstk.CTkLabel(master=frame,image=imag,text="", width=1080,height=1080)
bglabel.pack()

button1 = cstk.CTkButton(master=frame, text="Scan face (Webcam)", command=webcam_scan, height=80, width=250, font=("Arial", 24))
button1.place(relx=0.3, rely=0.3, anchor="e")

button2 = cstk.CTkButton(master=frame,text="Known Images",command=know_faces,height=80,width=250,font=("Arial",24))
button2.place(relx=0.75,rely=0.3,anchor="w")

button3 = cstk.CTkButton(master=frame,text="Delete a face",command=delete_a_face,height=80,width=250,font=("Arial",24))
button3.place(relx=0.75,rely=0.85,anchor="w")

button4 = cstk.CTkButton(master=frame,text="About",command=about,height=80,width=250,font=("Arial",24))
button4.place(relx=0.3,rely=0.85,anchor="e")

button5 = cstk.CTkButton(master=frame,text="Open Attendance",command=attendance,height=80,width=250,font=("Arial",24))
button5.place(relx=0.52,rely=0.5,anchor="center")

root.mainloop()
