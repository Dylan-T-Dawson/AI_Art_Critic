from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image 
# getting methods from the ML model for use with interface buttons
# from clarity_model_single_tune import *

# function to browse files on a machine (can only select PNG files with this implementation)
def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = ([("Images", "*.PNG*")]))
    labelFileExplorer.configure(text="File Opened: "+filename)
    return filename
    
# An interface for users to select an image file (.PNG) and have the ML algorithm return a response
# Takes in an image file, parses the image data, and returns a relevant response
window = Tk()
window.title("AI Art Critic")
window.geometry("500x500")
mainframe = ttk.Frame(window)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)

# Setting the style of the Tkinter window
style = ttk.Style()
style.theme_use('clam')

# Create a critique button that runs the ML algorithm 
# TODO: display the chosen image
# TODO: add functionality of running the ML algorithm upon clicking (on Critique button)
ttk.Button(mainframe, text="Critique!").grid(column=3, row=3, sticky=W)
ttk.Label(mainframe, text="Select an image to critique:").grid(column=3, row=1, sticky=W)
ttk.Label(mainframe, text="Image Selected:").grid(column=3, row=2, sticky=W)

# Creates a button to browse the files on a machine
browseFilesButton = Button(window, text="Browse Files", command = browseFiles)
browseFilesButton.grid(column=1, row=1)
chosenImageWindow = Frame(mainframe).grid(column=3, row=3)
labelFileExplorer = Label(window, text = "File Explorer")

for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

window.mainloop()