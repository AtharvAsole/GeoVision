import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk
import keras.backend as K

# Jaccard coefficient and total loss functions
def jaccard_coef(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred))
    total = K.sum(K.abs(y_true) + K.abs(y_pred))
    return K.mean((intersection + smooth) / (total + smooth))

def total_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + (1 - jaccard_coef(y_true, y_pred))

# Load your model
saved_model = load_model('models/Model_satellite_segmentation.h5', custom_objects={
    'dice_loss_plus_1focal_loss': total_loss,
    'jaccard_coef': jaccard_coef
})

# Define class labels
class_labels = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']
num_classes = len(class_labels)

def load_and_predict(image_path):
    image = Image.open(image_path).resize((256, 256)).convert('RGB')  # image to RGB
    image = np.expand_dims(np.array(image), 0)
    prediction = saved_model.predict(image)
    predicted_mask = np.argmax(prediction, axis=3)[0, :, :]
    return predicted_mask

def calculate_percentage_change(mask1, mask2):
    counts1 = np.bincount(mask1.flatten(), minlength=num_classes)
    counts2 = np.bincount(mask2.flatten(), minlength=num_classes)

    total_pixels1 = np.sum(counts1)
    total_pixels2 = np.sum(counts2)

    percentage_changes = []
    for i in range(num_classes):
        percentage_change = ((counts2[i] / total_pixels2) - (counts1[i] / total_pixels1)) * 100
        percentage_changes.append(percentage_change)
    return percentage_changes

def browse_image1():
    global image_path_1
    image_path_1 = filedialog.askopenfilename(title="Select Image 1", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    entry1.delete(0, tk.END)
    entry1.insert(0, image_path_1)
    show_image(image_path_1, image_label1)

def browse_image2():
    global image_path_2
    image_path_2 = filedialog.askopenfilename(title="Select Image 2", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    entry2.delete(0, tk.END)
    entry2.insert(0, image_path_2)
    show_image(image_path_2, image_label2)

def show_image(image_path, label):
    image = Image.open(image_path).resize((100, 100))
    photo = ImageTk.PhotoImage(image)
    label.config(image=photo)
    label.image = photo 

def display_results():
    if 'image_path_1' not in globals() or 'image_path_2' not in globals():
        messagebox.showerror("Error", "Please select both images.")
        return

    mask1 = load_and_predict(image_path_1)
    mask2 = load_and_predict(image_path_2)

    percentage_changes = calculate_percentage_change(mask1, mask2)

    # results window
    result_window = Toplevel(root)
    result_window.title("Change Detection Results")
    result_window.geometry("800x600")
    result_window.configure(bg="#f0f8ff")

    # Create title label
    title_label = tk.Label(result_window, text="Change Detection Results", bg="#f0f8ff", font=("Helvetica", 20, "bold"))
    title_label.pack(pady=20)

    # Frame for images
    image_frame = tk.Frame(result_window, bg="#f0f8ff")
    image_frame.pack(pady=20)

    # first image with year
    img1_display = tk.Label(image_frame, text=f"Image 1 - Year: {entry_year1.get()}", bg="#f0f8ff", font=("Helvetica", 14, "bold"))
    img1_display.pack(side=tk.LEFT, padx=20)
    img1 = Image.open(image_path_1).resize((300, 300))
    img1_tk = ImageTk.PhotoImage(img1)
    img1_label = tk.Label(image_frame, image=img1_tk, bg="#f0f8ff")
    img1_label.image = img1_tk 
    img1_label.pack(side=tk.LEFT)

    # Display second image with year
    img2_display = tk.Label(image_frame, text=f"Image 2 - Year: {entry_year2.get()}", bg="#f0f8ff", font=("Helvetica", 14, "bold"))
    img2_display.pack(side=tk.LEFT, padx=20)
    img2 = Image.open(image_path_2).resize((300, 300))
    img2_tk = ImageTk.PhotoImage(img2)
    img2_label = tk.Label(image_frame, image=img2_tk, bg="#f0f8ff")
    img2_label.image = img2_tk
    img2_label.pack(side=tk.LEFT)

    # Frame for results
    result_frame = tk.Frame(result_window, bg="#f0f8ff")
    result_frame.pack(pady=20)

    # Display result percentages
    for i, label in enumerate(class_labels):
        change = percentage_changes[i]
        if change > 0:
            result_text = f"{label}: Increase by {change:.2f}%"
            color = "green"
        elif change < 0:
            result_text = f"{label}: Decrease by {-change:.2f}%"
            color = "red"
        else:
            result_text = f"{label}: No change"
            color = "blue"
        
        result_label = tk.Label(result_frame, text=result_text, bg="#f0f8ff", fg=color, font=("Helvetica", 14))
        result_label.pack()

# GUI setup
root = tk.Tk()
root.title("Urban Change Detection GUI")
root.geometry("600x400")
root.configure(bg="#eaeaea")

frame = tk.Frame(root, bg="#f0f8ff", bd=5)
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Image 1
label1 = tk.Label(frame, text="Select Image 1:", bg="#f0f8ff", font=("Helvetica", 12))
label1.grid(row=0, column=0, padx=5, pady=5, sticky="w")

entry1 = tk.Entry(frame, width=50)
entry1.grid(row=0, column=1, padx=5, pady=5)

button1 = tk.Button(frame, text="Browse", command=browse_image1, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
button1.grid(row=0, column=2, padx=5, pady=5)

image_label1 = tk.Label(frame, bg="#f0f8ff")
image_label1.grid(row=1, column=0, columnspan=3, pady=5)

# Image 2
label2 = tk.Label(frame, text="Select Image 2:", bg="#f0f8ff", font=("Helvetica", 12))
label2.grid(row=2, column=0, padx=5, pady=5, sticky="w")

entry2 = tk.Entry(frame, width=50)
entry2.grid(row=2, column=1, padx=5, pady=5)

button2 = tk.Button(frame, text="Browse", command=browse_image2, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
button2.grid(row=2, column=2, padx=5, pady=5)

image_label2 = tk.Label(frame, bg="#f0f8ff")
image_label2.grid(row=3, column=0, columnspan=3, pady=5)

# Year Inputs
label_year1 = tk.Label(frame, text="Year for Image 1:", bg="#f0f8ff", font=("Helvetica", 12))
label_year1.grid(row=4, column=0, padx=5, pady=5, sticky="w")

entry_year1 = tk.Entry(frame, width=20)
entry_year1.grid(row=4, column=1, padx=5, pady=5)

label_year2 = tk.Label(frame, text="Year for Image 2:", bg="#f0f8ff", font=("Helvetica", 12))
label_year2.grid(row=5, column=0, padx=5, pady=5, sticky="w")

entry_year2 = tk.Entry(frame, width=20)
entry_year2.grid(row=5, column=1, padx=5, pady=5)

# Predict Button
button_predict = tk.Button(frame, text="Predict Change", command=display_results, bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"))
button_predict.grid(row=6, column=0, columnspan=3, pady=10)

root.mainloop()
