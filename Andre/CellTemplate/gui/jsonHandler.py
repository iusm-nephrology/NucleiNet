import json
from tkinter import *
import os

## TODO: improve readability plus add "help" description for each variable

class JSON_GUI(Frame):
    def __init__(self, master = None, renalGUI = None):
        Frame.__init__(self, master)
        self.master = master
        self.renalGUI = renalGUI
        self.init_window()

    def init_window(self):
        self.master.title("Config editor")
        self.pack(fill=BOTH, expand=1)
        self.createDefaultJson()
        self.fields = []
        self.FILEPATH_OUT = "myconfig.json"

        ## Setup Window
        self.master.geometry('1000x1000')
        myframe = Frame(self.master, width="800",height="400")
        myframe.place(x = 0, y = 0)
        fileFrame = Frame(self.master, width="800",height="50")
        fileFrame.place(x = 0, y = 350)
        self.l1 = Label(myframe, text = "Config file variables").pack()
        canvas=Canvas(myframe)
        frame=Frame(canvas)
        myscrollbar=Scrollbar(myframe,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=myscrollbar.set)
        myscrollbar.pack(side="right",fill="y")
        canvas.pack() 
        canvas.create_window((0,100),window=frame,anchor='nw')
        frame.bind("<Configure>", self.configureCanvas)
        self.canvas = canvas
        self.frame = frame
        
        ## Process inputs
        self.grabInputs()
        self.ents = self.makeform(self.frame, self.fields)
        self.master.bind('<Return>', (lambda event, e=self.ents: self.fetch(e)))

        # Show and save buttons
        self.b1 = Button(self.master, text='Print', command=(lambda e=self.ents: self.fetch(e)))
        self.b1.pack(side=LEFT, padx=5, pady=5)
        self.b2 = Button(self.master, text='Quit', command=self.master.quit)
        self.b2.pack(side=LEFT, padx=5, pady=5)
        self.b3 = Button(self.master, text='Save .json', command= lambda: self.processEntries(self.ents))
        self.b3.pack(side=LEFT, padx=5, pady=5)

        #Save filepath entry
        self.l2 = Label(fileFrame, text = "Filename to save: ", anchor = 'w').pack(side=LEFT)
        self.e1 = Entry(fileFrame)
        self.e1.insert(END, self.FILEPATH_OUT)
        self.e1.pack(side=LEFT)
        
    def grabInputs(self):
        for item in self.data:
            if type(self.data[item]) is dict:
                for subitem in self.data[item]:
                    if type(self.data[item][subitem]) is dict:
                        for subitem2 in self.data[item][subitem]:
                           self.fields.append([item + " " + subitem2 + " " + "arg", self.data[item][subitem][subitem2]])
                    else:
                        self.fields.append([item + " " + subitem, self.data[item][subitem]])
            else:
               self.fields.append([item, self.data[item]])
               
    def configureCanvas(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"),width=800,height=300)
        
    def fetch(self, entries):
        print(self.data)
        print("=========")
        for entry in entries:
            field = entry[0]
            text  = entry[1].get()
            print('%s: "%s"' % (field, text))

    def makeform(self, root, fields):
        entries = []
        i = 0
        for field in fields:
            lab = Label(root, width=30, text=field[0], anchor='w').grid(row=i,column=0, padx = 3)
            ent = Entry(root, width = 100)
            default_text = field[1]
            ent.insert(END, default_text)
            ent.grid(row=i,column=1)
            entries.append((field[0], ent))
            self.frame.grid_columnconfigure(1, weight=1)
            i = i + 1
        return entries
    
    def processEntries(self, entries):
        ## TODO: error when using only one metric
        self.FILEPATH_OUT = self.e1.get()
        for entry in entries:
            field = entry[0].split(' ')
            text = entry[1].get()
            if text[0].isdigit():
                if '.' in text:
                    text = float(text)
                else:
                    text = int(text)
            elif len(text.split(' ')) > 1:
                tmp = text.split(' ')
                text = []
                for item in tmp:
                    text.append(item)
                
            
            if type(self.data[field[0]]) is dict:
                if len(field) == 3:
                    self.data[field[0]]['args'][field[1]] = text 
                else:
                    self.data[field[0]][field[1]] = text                       
            else:
                self.data[field[0]] = text

        self.writeJson(self.FILEPATH_OUT)
        print(".json file saved")
        
    def createDefaultJson(self):
        imgdir = str(self.renalGUI.ImgDirName)
        if imgdir == "": imgdir = "Path/to/images"
        traincsv = os.path.join(imgdir, "Train.csv")
        testcsv = os.path.join(imgdir, "Test.csv")
        
        self.data = {}
        self.data['name'] = 'Default_fromGUI'
        self.data['n_gpu']  = 0
        self.data['arch']  = {'type': 'groundTruthModel', 'args': {}}
        self.data['data_loader']  = {'type': 'groundTruth_DataLoader', 'args': {'data_dir': imgdir, 'csv_path': traincsv, 'batch_size': 32, 'shuffle': True, 'validation_split': 0.1, 'num_workers': 2, 'training': True}}
        self.data['data_loader_test']  = {'type': 'groundTruth_DataLoader', 'args': {'data_dir': imgdir, 'csv_path': testcsv, 'batch_size': 1, 'shuffle': False, 'validation_split': 0.0, 'num_workers': 2, 'training': False}}
        self.data['optimizer']  = {'type': 'Adam', 'args': {'lr': 0.001, 'weight_decay': 0, 'amsgrad': True}}
        self.data['loss']  = 'nll_loss'
        self.data['metrics']  = ['my_metric', 'my_metric2']
        self.data['lr_scheduler']  = {'type': 'StepLR', 'args': {'step_size': 50, 'gamma': 0.1}}
        self.data['trainer']  = {'epochs': 75, 'save_dir': '../saved/', 'save_period': 1, 'verbosity': 2, 'monitor': 'min val_loss', 'early_stop': 10, 'tensorboardX': True, 'log_dir': '../saved/runs'}
        self.data['augmenter'] = {'type': 'augmentation_handler', 'args': {'rotation': 0, 'scale': 0.0, 'crop': 0, 'translate': 0.0, 'random_flip': 0.0}}
        
    def writeJson(self, path):
        with open(path, "w") as write_file:
            json.dump(self.data, write_file, indent=4)
            print("data written to " + path)

        



