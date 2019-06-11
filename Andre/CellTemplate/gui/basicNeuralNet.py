################################################################################
#
# Basic Neural Net
#
# This is a generic neural net module. 
# It manages all state for a single neural network.
#
# class BasicNeuralNet Methods
# ============================
# __init__(self):
#   Constructor
#
# initNewNeuralNet(self):
#   Initialize the runtime state from a saved file.
#   This is used when we create a new neural net from scratch.
#
# loadNeuralNetFromFile(self, filePathName):
#   Read the saved state of a neural net from a file.
#   This is used to resume the state from where we left off on a previous execution.
#
# loadNeuralNetFromFile(self, filePathName):
#   Save the state of a neural net to a file.
#
# trainOneFile(self, filePathName):
#   Train the neural net using data from a single file.
#   The format of the file is specific to the net, and may be a JPG image
#   file or else a csv Excel file or something else.
#
# validateOneFile(self, filePathName):
#   Validate the neural net using data from a single file.
#
# testOneFile(self, filePathName):
#   Test the neural net using data from a single file.
#
# class BasicNeuralNet variables
# ==============================
#   self.stateFile
#
################################################################################


################################################################################
# BasicNeuralNet class. This manages all state for a single neural network.
################################################################################
class BasicNeuralNet():
    #######################################
    # Constructor
    #######################################
    def __init__(self):
        self.stateFile = ""
    # End of __init__


    #######################################
    # Initialize the runtime state from a saved file.
    # This is used when we create a new neural net from scratch.
    #######################################
    def initNewNeuralNet(self):
        print("initNeuralNet")
    # End of initNewNeuralNet


    #######################################
    # Read the saved state of a neural net from a file.
    # This is used to resume the state from where we left off on a previous execution.
    #######################################
    def loadNeuralNetFromFile(self, filePathName):
        print("loadNeuralNetFromFile: filePathName=" + filePathName)
     # End - loadNeuralNetFromFile


    #######################################
    # Save the state of a neural net to a file.
    #######################################
    def loadNeuralNetFromFile(self, filePathName):
        print("loadNeuralNetFromFile: filePathName=" + filePathName)
     # End - loadNeuralNetFromFile





    #######################################
    # Train the neural net using data from a single file.
    # The format of the file is specific to the net, and may be a JPG image
    # file or else a csv Excel file or something else.
    #######################################
    def trainOneFile(self, filePathName):
        print("trainOneFile: filePathName=" + filePathName)
     # End - trainOneFile


    #######################################
    # Validate the neural net using data from a single file.
    #######################################
    def validateOneFile(self, filePathName):
        print("validateOneFile: filePathName=" + filePathName)
     # End - validateOneFile


    #######################################
    # Test the neural net using data from a single file.
    #######################################
    def testOneFile(self, filePathName):
        print("testOneFile: filePathName=" + filePathName)
     # End - testOneFile


# End - class BasicNeuralNet


