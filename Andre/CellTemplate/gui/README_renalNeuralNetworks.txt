This program contains 2 separate python modules, Copy both into the same 
working directory, and then run them there.

1. renalNeuralNetGUI.py
This is the top level of the command and the GUI. It should be used for debugging, performance tuning, and other development tasks. I plan to add a lot to this, specifically space to track loss vs Epochs when we need to tune the hyperparameters.

All other modules are imported by this one. You execute this script to run the program:
    python3 renalNeuralNetGUI.py

2. basicNeuralNet.py
This is the framework for the neural network. It is stored in a separate file so we can use this in different user interfaces (like a simpler command line or batch system) or publish the code.



Installation
============
I am not sure you need to do ANY of this. A lot of the basic libraries are already installed with Python 3.

tkinter is part of Python3, but on my system it defaults to a python2. 
I also had to do this:
  sudo -i
  dnf install python3-imaging-tk

It seems that PIL is also part of Python3, but PIL has been forked, and the active branch is called Pillow.

Running
============
    python3 renalNeuralNetGUI.py


To Do:
- Put in a copyright and license. I suggest that we use something friendly, like the MIT Open Source license (which basically says "you can use this in any way you want, including commercial use, just don't sue us if there is a bug"). Seth, I leave this up to you as this will be part of IU research, but please make it open source.

- Create a GitHub depot. Seth, you should probably be the one to do this, so that you and/or IU remain the official owner. 

