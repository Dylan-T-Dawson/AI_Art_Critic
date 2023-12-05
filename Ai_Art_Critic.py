from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from image_normalizer import resize_image
from Classifier import load_and_classify
from PIL import ImageTk, Image

globalImage = None

def browseFiles():
    global globalImage
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File", filetypes=[("Images", "*.PNG*")])
    labelFileExplorer.configure(text="File Opened: " + filename)

    # Display the selected image
    image = Image.open(filename)
    image = resize_image(image)
    globalImage = image
    img = ImageTk.PhotoImage(image)
    

    # Update the label to show the selected image
    label_chosen_image.img = img  # To prevent garbage collection
    label_chosen_image.configure(image=img)
    label_chosen_image.image = img
    return image

def generate_random_text():
    text_widget.delete(1.0, END)  # Clear existing text
    global globalImage
    classification = load_and_classify(globalImage)
    text_widget.insert(END, classification + "\n")

# Interface for users to select an image file (.PNG) and have the ML algorithm return a response
window = Tk()
window.title("AI Art Critic")
window.geometry("900x600")  # Adjusted window size for a better layout
window.configure(bg='#f0f0f0')  # Light gray background color

mainframe = ttk.Frame(window, padding="20")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)

# Setting a new style for the Tkinter window
style = ttk.Style()
style.theme_use('clam')
style.configure('TButton', foreground='#ffffff', background='#4CAF50')  # Green button style
style.configure('TLabel', foreground='#333333', font=('Helvetica', 12))  # Label style

# Labels with a new style
label_instruction = ttk.Label(mainframe, text="Select an image to critique:", style='TLabel')
label_instruction.grid(column=3, row=1, sticky=W)
label_selected_image = ttk.Label(mainframe, text="Image Selected:", style='TLabel')
label_selected_image.grid(column=3, row=2, sticky=W)

# Create a label to display the selected image
label_chosen_image = Label(mainframe)
label_chosen_image.grid(column=3, row=3, sticky=W, pady=10)
label_chosen_image.configure(bg='#f0f0f0')  # Set background color

# Create a critique button that runs the ML algorithm
critique_button = ttk.Button(mainframe, text="Critique!", style='TButton', command=generate_random_text)
critique_button.grid(column=3, row=4, sticky=W, pady=10)

# Create a Text widget to display random text
text_widget = Text(mainframe, height=20, width=50, wrap=WORD)
text_widget.grid(column=4, row=3, rowspan=2, pady=10, padx=10, sticky=NW)

# Creates a button to browse the files on a machine
browse_files_button = Button(window, text="Browse Files", command=browseFiles, bg='#2196F3', fg='#ffffff')  # Blue button style
browse_files_button.grid(column=1, row=1, pady=10)

labelFileExplorer = Label(window, text="File Explorer", font=('Helvetica', 10), fg='#666666', bg='#f0f0f0')

for child in mainframe.winfo_children():
    child.grid_configure(padx=10, pady=10)

window.mainloop()