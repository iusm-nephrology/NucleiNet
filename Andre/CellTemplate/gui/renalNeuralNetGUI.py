################################################################################
#
# This is the top level of the command and the GUI.
# You execute this script to run the program:
#    python3 renalNeuralNetGUI.py
#
# All other modules are imported by this one.
#
################################################################################
# 
# Window Class Member Variables
# =============================
# These are the valiables that hold the runtime state and also display it in the window.
#
# GUI Controls 
# These are the GUI elements to display data. Note some things like the menus are
# not stored as member variables because they are only used once, to initialize the
# GUI and register a vallback.
#   self.imageCanvas - display one image file
#   self.StatusLabel - show whether we are training/validating/testing
#   self.TrainDirLabel - display the 
#   self.ValidateDirLabel
#   self.TestDirLabel
#   self.CurrentFileLabel
# 
# File Iteration State
#   self.CurrentAction - This is ACTION_IDLE, ACTION_TRAIN, ACTION_VALIDATE, ACTION_TEST
#   self.TrainDirPathName - The pathname of the directory that holds training files
#   self.ValidateDirPathName - The pathname of the directory that holds validation files
#   self.TestDirPathName - The pathname of the directory that holds test files
#   self.SrcDirName - The directory we are reading from. This is used for all states, 
#                       so it may be set to the training, validation or test directory.
#   self.SrcDirIter - A runtime iterator to look at all files in the src directory.
#                       This is used for train, validate, and test.
#   self.CurrentFilePath - The file we are currently reading.
#                       This is used for train, validate, and test.
#   self.neuralNet - The neural net loaded from another module.
#
#
# Window Class Methods
# ====================
#   __init__(self, master=None)
#   init_window(self)
#   ExitApplication(self)
#   SetProgramStateToIdle(self)
#   SetTrainingDir(self)
#   SetValidationDir(self)
#   SetTestingDir(self)
#   TrainFromAllFiles(self)
#   ValidateFromAllFiles(self)
#   TestFromAllFiles(self)
#   TrainOneFile(self)
#   ProcessAllFiles(self)
#   ProcessOneFile(self, filePathName)
#   OnShowImageCommand(self)
#   OnShowImageInWindowCommand(self)
#
################################################################################
import os.path
from os import scandir
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from jsonHandler import JSON_GUI
 
# This is the module we write.
import basicNeuralNet

# This is an enum for the current action state. It is stored in self.CurrentAction
ACTION_IDLE = 0
ACTION_TRAIN = 1
ACTION_VALIDATE = 2
ACTION_TEST = 3

# You can either use .place() or .grid() or .pack() to arrange items in the window.
# Do not mix these, pick one and use it everywhere.
# 1. pack() is the most common, and lets the window manager do all layout, but the
#   can provide some constraints.
# 2. place() is the least common and is completely general, but then you control all layout.
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
WINDOW_WIDTH_STRING = "800x800"

TEXT_STATUS_POSITION_TOP = 1
TEXT_STATUS_POSITION_LEFT = 10
TEXT_STATUS_HEIGHT = 20

BUTTON_POSITION_TOP = 120
BUTTON_POSITION_LEFT = 10
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 30
BUTTON_PADDING = 20

CANVAS_POSITION_TOP = 200
CANVAS_POSITION_LEFT = 10
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 600



################################################################################
# Application window class. This inherits from the tkinter Frame class.
################################################################################
class Window(Frame):
    #######################################
    # Constructor
    #######################################
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        #reference to the master widget, which is the tk window                 
        self.master = master
        self.init_window()
    # End of __init__


    #######################################
    #
    #######################################
    def init_window(self):
        self.master.title("GUI")

        # Allow the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # Initialize some member variables which will be used below.
        # TODO #1: Change these hardcoded paths to something you prefer.
        self.TrainDirPathName = "/home/ddean/testImages"
        self.ValidateDirPathName = "/home/ddean/testImages"
        self.TestDirPathName = "/home/ddean/testImages"
        self.ConfigPathName = "CellTemplate/config.json"
        self.ImgDirName = ""
        self.TestFileName = ""
        self.csvPath = ""
        self.dataSplit = 0.1

        # Create the neural net runtime
        self.neuralNet = basicNeuralNet.BasicNeuralNet()
        self.neuralNet.initNewNeuralNet()

        # Create menus
        menuBar = Menu(self.master)
        self.master.config(menu = menuBar)

        # Create the file and edit menu
        fileMenu = Menu(menuBar)
        editMenu = Menu(menuBar)
        andreMenu = Menu(menuBar)

        # Add commands to the menus
        fileMenu.add_command(label="Set Training Directory...", command=self.SetTrainingDir)
        fileMenu.add_command(label="Set Validation Directory...", command=self.SetValidationDir)
        fileMenu.add_command(label="Set Test Directory...", command=self.SetTestingDir)
        andreMenu.add_command(label="Set Image Directory...", command= self.SetImgDir)
        andreMenu.add_command(label="Set Config File...", command= self.SetConfig)
        andreMenu.add_command(label="Set Test .pth File...", command= self.SetTestFile)
        andreMenu.add_command(label="Set image labels csv  File...", command= self.SetCsvFile)
        fileMenu.add_command(label="Exit", command=self.ExitApplication)
        editMenu.add_command(label="Show Img", command=self.OnShowImageCommand)

        # Add the menus to the menubar
        menuBar.add_cascade(label="File", menu=fileMenu)
        menuBar.add_cascade(label="Edit", menu=editMenu)
        menuBar.add_cascade(label="Andre", menu=andreMenu)

        # Create the image canvas
        self.imageCanvas = Canvas(self, width = CANVAS_WIDTH, height = CANVAS_HEIGHT)
        self.imageCanvas.place(x = CANVAS_POSITION_LEFT, y = CANVAS_POSITION_TOP, 
                                width = CANVAS_WIDTH, height = CANVAS_HEIGHT)

        # Create buttons and place them in the window.
        xPos = BUTTON_POSITION_LEFT
        yPos = BUTTON_POSITION_TOP + BUTTON_HEIGHT
        currentButton = Button(self, text="Train", command = self.TrainFromAllFiles)
        currentButton.place(x = xPos, y = yPos)

        xPos = xPos + BUTTON_WIDTH + BUTTON_PADDING;
        currentButton = Button(self, text="Validate", command = self.ValidateFromAllFiles)
        currentButton.place(x = xPos, y = yPos)

        xPos = xPos + BUTTON_WIDTH + BUTTON_PADDING;
        currentButton = Button(self, text="Test", command = self.TestFromAllFiles)
        currentButton.place(x = xPos, y = yPos)

        xPos = BUTTON_POSITION_LEFT
        yPos = BUTTON_POSITION_TOP + BUTTON_HEIGHT
        currentButton = Button(self, text="Train One File", command = self.TrainOneFile)
        currentButton.place(x = xPos, y = yPos)

        xPos = xPos + BUTTON_WIDTH + BUTTON_PADDING;
        showButton = Button(self, text="Show Image", command = self.OnShowImageCommand)
        showButton.place(x = xPos, y = yPos)

        xPos = BUTTON_POSITION_LEFT
        yPos = yPos + BUTTON_HEIGHT
        showButton = Button(self, text="Edit config.json", command = self.runJsonGUI)
        showButton.place(x = xPos, y = yPos)

        xPos = xPos + BUTTON_WIDTH + BUTTON_PADDING;
        showButton = Button(self, text="train", command = lambda: self.trainAndre())
        showButton.place(x = xPos, y = yPos)

        xPos = xPos + BUTTON_WIDTH + BUTTON_PADDING;
        showButton = Button(self, text="test", command = lambda: self.testAndre())
        showButton.place(x = xPos, y = yPos)

        xPos = xPos + BUTTON_WIDTH + BUTTON_PADDING;
        showButton = Button(self, text="split data", command = lambda: self.splitData())
        showButton.place(x = xPos, y = yPos)
        
        # Create the text fields for the status strings        
        yPos = TEXT_STATUS_POSITION_TOP
        self.StatusLabel = Label(self, text="Idle")
        self.StatusLabel.place(x = TEXT_STATUS_POSITION_LEFT, y = yPos)

        yPos = yPos + TEXT_STATUS_HEIGHT
        self.TrainDirLabel = Label(self, text="Image directory Directory: " + self.ImgDirName)
        self.TrainDirLabel.place(x = TEXT_STATUS_POSITION_LEFT, y = yPos)

        yPos = yPos + TEXT_STATUS_HEIGHT
        self.ValidateDirLabel = Label(self, text="CSV file with all filepaths and labels: " + self.csvPath)
        self.ValidateDirLabel.place(x = TEXT_STATUS_POSITION_LEFT, y = yPos)

        yPos = yPos + TEXT_STATUS_HEIGHT
        self.TestDirLabel = Label(self, text="Test model.pth file: " + self.TestFileName)
        self.TestDirLabel.place(x = TEXT_STATUS_POSITION_LEFT, y = yPos)

        yPos = yPos + TEXT_STATUS_HEIGHT
        self.ConfigLabel = Label(self, text="Config file: " + self.ConfigPathName)
        self.ConfigLabel.place(x = TEXT_STATUS_POSITION_LEFT, y = yPos)
        

        yPos = yPos + TEXT_STATUS_HEIGHT
        self.CurrentFileLabel = Label(self, text="Current File: ")
        self.CurrentFileLabel.place(x = TEXT_STATUS_POSITION_LEFT, y = yPos)

        # Initialize the source Directory iterator
        self.SetProgramStateToIdle()
    # End of init_window


    #######################################
    #######################################
    def ExitApplication(self):
        exit()
    # End - ExitApplication


    #######################################
    #######################################
    def SetProgramStateToIdle(self):
        self.StatusLabel.config(text = "IDLE")
        self.CurrentAction = ACTION_IDLE

        self.CurrentFilePath = ""
        self.CurrentFileLabel.config(text = "Current File: " + self.CurrentFilePath)
    # End - SetProgramStateToIdle


    #######################################
    # Called by a menu command to change a directory pathname.
    #######################################
    def SetTrainingDir(self):
        dirName = filedialog.askdirectory()
        if dirName:
            self.TrainDirPathName = dirName
            self.TrainDirLabel.config(text = "Image directory Directory: " + self.ImgDirName)
    # End - SetTrainingDir


    #######################################
    # Called by a menu command to change a directory pathname.
    #######################################
    def SetValidationDir(self):
        dirName = filedialog.askdirectory()
        if dirName:
            self.ValidateDirPathName = dirName
            self.ValidateDirLabel.config(text = "CSV file with all filepaths and labels: " + self.csvPath)
    # End - SetValidationDir


    #######################################
    # Called by a menu command to change a directory pathname.
    #######################################
    def SetTestingDir(self):
        dirName = filedialog.askdirectory()
        if dirName:
            self.TestDirPathName = dirName
            self.TestDirLabel.config(text = "Test model.pth file: " + self.TestFileName)
    # End - SetTestDir

    #######################################
    # Called by a menu command to change a config file pathname
    #######################################
    def SetConfig(self):
        fileName = filedialog.askopenfilename()
        if fileName:
            self.ConfigPathName = fileName
            self.ConfigLabel.config(text = "Config path: " + self.ConfigPathName)
    # End - SetTestingDir

    #######################################
    # Called by a menu command to change a test file pathname to config
    #######################################
    def SetTestFile(self):
        fileName = filedialog.askopenfilename()
        if fileName:
            self.TestFileName = fileName
            self.TestDirLabel.config(text = "Test model.pth file: " + self.TestFileName)
    # End - SetTestFile

    #######################################
    # Called by a menu command to change the pathname to the image directory
    #######################################
    
    def SetImgDir(self):
      dirName = filedialog.askdirectory()
      if dirName:
            self.ImgDirName = dirName
            self.TrainDirLabel.config(text = "Image directory Directory: " + self.ImgDirName)
            
    def SetCsvFile(self):
        fileName = filedialog.askopenfilename()
        if fileName:
            self.csvPath = fileName
            self.ValidateDirLabel.config(text = "CSV file with all filepaths and labels: " + self.csvPath)
    #######################################
    #######################################
    def TrainFromAllFiles(self):
        self.StatusLabel.config(text = "TRAINING")
        self.CurrentAction = ACTION_TRAIN

        self.SrcDirName = self.TrainDirPathName
        self.ProcessAllFiles()
    # End - TrainFromAllFiles


    #######################################
    #######################################
    def ValidateFromAllFiles(self):
        self.StatusLabel.config(text = "VALIDATING")
        self.CurrentAction = ACTION_VALIDATE

        self.SrcDirName = self.ValidateDirPathName
        self.ProcessAllFiles()
    # End - ValidateFromAllFiles


    #######################################
    #######################################
    def TestFromAllFiles(self):
        self.StatusLabel.config(text = "TESTING")
        self.CurrentAction = ACTION_TEST

        self.SrcDirName = self.TestDirPathName
        self.ProcessAllFiles()
    # End - TestFromAllFiles


    #######################################
    #######################################
    def TrainOneFile(self):
        # If we are idle, then start a new training sequence, but just single-step.
        if ACTION_IDLE == self.CurrentAction:
            self.StatusLabel.config(text = "TRAINING")
            self.CurrentAction = ACTION_TRAIN
            self.SrcDirName = self.TrainDirPathName
            self.SrcDirIter = scandir(self.SrcDirName)

        # We are either being called on the 2nd or later file, or else we
        # just set up the state.
        for fileEntry in self.SrcDirIter:
            if not fileEntry.name.startswith('.') and fileEntry.is_file():
                self.CurrentFilePath = os.path.join(self.SrcDirName, fileEntry.name)
                self.CurrentFileLabel.config(text = "Current File: " + self.CurrentFilePath)
                self.ProcessOneFile(self.CurrentFilePath)
                return

        # If we didn't find a file, then we are done.
        self.SetProgramStateToIdle()
    # End - TrainOneFile


    #######################################
    #######################################
    def ProcessAllFiles(self):
        # Restart the iterator at the beginning.
        self.SrcDirIter = scandir(self.SrcDirName)

        # Enumerate each file
        for fileEntry in self.SrcDirIter:
            if not fileEntry.name.startswith('.') and fileEntry.is_file():
                self.CurrentFilePath = os.path.join(self.SrcDirName, fileEntry.name)
                self.CurrentFileLabel.config(text = "Current File: " + self.CurrentFilePath)
                self.ProcessOneFile(self.CurrentFilePath)

        # Restart the iterator at the beginning.
        self.SetProgramStateToIdle()
    # End - ProcessAllFiles


    #######################################
    #######################################
    def ProcessOneFile(self, filePathName):
        #print("ProcessOneFile: filePathName=" + filePathName)
        if ACTION_TRAIN == self.CurrentAction:
            self.neuralNet.trainOneFile(filePathName)
        elif ACTION_VALIDATE == self.CurrentAction:
            self.neuralNet.validateOneFile(filePathName)
        elif ACTION_TEST == self.CurrentAction:
            self.neuralNet.testOneFile(filePathName)
     # End - ProcessOneFile


    #######################################
    # Draw the current image in the canvas
    #######################################
    def OnShowImageCommand(self):
        # These must be member variables, or else the objects will be garbage collected 
        # when this procedure returns.
        self.loadedImage = Image.open(self.CurrentFilePath)

        # Scale the image to fit the picture
        imageWidth, imageHeight = self.loadedImage.size
        heightScale = float(CANVAS_HEIGHT / imageHeight)
        widthScale = float(CANVAS_WIDTH / imageWidth)
        newHeight = int((float(imageWidth) * float(heightScale)))
        newWidth = int((float(imageHeight) * float(widthScale)))
        #self.loadedImage = self.loadedImage.resize((newHeight, newWidth), Image.ANTIALIAS)

        self.renderedImage = ImageTk.PhotoImage(self.loadedImage)
        self.imageCanvas.create_image((0, 0), anchor = CENTER, image = self.renderedImage)
        #self.imageCanvas.pack()
    # End - OnShowImageCommand



    #######################################
    #######################################
    def OnShowImageInWindowCommand(self):
        imagePopUp = Image.open(self.CurrentFilePath)
        imagePopUp.show();
    # End - OnShowImageInWindowCommand

    def runJsonGUI(self):
        root_json = Toplevel()
        json_gui = JSON_GUI(root_json, renalGUI = self)
        root_json.mainloop()

    def trainAndre(self):
        runString = "python3 ../train.py -c " + self.ConfigPathName
        os.system(runString)

    def testAndre(self):
        runString = "python3 ../test.py -r " + self.TestFileName 
        os.system(runString)
    def splitData(self):
        runString = "python3 ../utils/splitDataFolder.py -r " + self.ImgDirName + \
                    " -c " +  self.csvPath + \
                    " -s " + str(self.dataSplit)
        print(runString)
        os.system(runString)
        
# End - class Window



# Create the window system root window, and set its size
# This will be the top-level window, or analogous to a "desktop" 
# for all windows that belong to this app.
root = Tk()
root.geometry(WINDOW_WIDTH_STRING)

# Create the main app window.
app = Window(root)

# The main event loop. This function returns when the application exits
root.mainloop()  

