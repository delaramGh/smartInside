import os
import json
import pandas as pd
import streamlit as st
from PIL import Image
import random

import utils

# Constants
IMAGE_DIR = "./images"
ORIGINAL_IMAGE_DIR = "../../../original_dataset"
STATE_FILE = "state.json"

LABELS = ["Ok", "Lost", "Hallucinations"]
labels_dict = {
    "Ok": {
        "emoji": "✅",
        "button_text": "OK",
        "help": "The modified image is a realistic representation of the original image \
            transformed in accordance with the chosen transformation.",
    },
    "Lost": {
        "emoji": "🚫",
        "button_text": "Loss",
        "help": "Objects in the modified image appear distorted, blurred, \
            or significantly altered, rendering them unrecognizable.",
    },
    "Hallucinations": {
        "emoji": "💭",
        "button_text": "Hallucination",
        "help": "The modified image includes objects that are not present in the original image \
                       (e.g. humans, cars, trees which are not related to the context of electric poles)",
    },
}


def get_random_pun():
    puns = [
        "You've got the eye of the tiger 🐯 - 10 images tamed!",
        "Keep it up! You've just outshined your computer's image recognition! 💻✨",
        "10 down, infinity to go! You're on a pixel-perfect streak! 🎯",
        "Labeling these 10 images was a snapshot for you! 📸",
        "You're 10 images smarter now, or is it 10 images artier? 🎨",
        "A picture's worth a thousand words, but your labels are priceless! 💰🖼️",
        "In a flash, 10 images labeled! You're a true image maestro! 🌟",
        "10 images down! You're painting a picture of success! 🎨🏆",
        "Keep clicking! Your labels are developing beautifully! 📷💐",
        "Your labeling skills are in high resolution - 10 more masterpieces done! 🖼️🌟"
    ]
    return random.choice(puns)

def initialize_state(state):
    state["selected_index"] = 0


def load_state(state):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            loaded_state = json.load(f)
            state["selected_index"] = loaded_state["selected_index"]
    else:
        initialize_state(state)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump({"selected_index": state["selected_index"]}, f)


def display_images(selected_index, image_files):
    # Display the images
    image = Image.open(f"{IMAGE_DIR}/{image_files[selected_index]}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Modified Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Original Image")
        # Check if corresponding image in ORIGINAL_IMAGE_DIR exists and display it
        try:
            original_image_name = image_files[selected_index].replace(
                "_torchvision", ""
            )
            original_image = Image.open(f"{ORIGINAL_IMAGE_DIR}/{original_image_name}")
            st.image(original_image, use_column_width=True)
        except FileNotFoundError:
            st.error("No original image found.")


def prev_image(state):
    if state["selected_index"] > 0:
        state["selected_index"] -= 1


def next_image(state, image_files):
    if state["selected_index"] < len(image_files) - 1:
        state["selected_index"] += 1
    else:
        st.balloons()
    
    if state["selected_index"] % 10 == 0:
        st.success(get_random_pun())


def main():
    # Create CSV and load image files
    utils.create_csv_if_not_exist()
    image_files = utils.get_image_files(IMAGE_DIR)

    # Load labels from CSV file
    df_labels = utils.load_labels()

    # Session state
    if not st.session_state:
        load_state(st.session_state)
    state = st.session_state

    st.set_page_config(layout="wide")
    st.title("Image Labeling App")

    # Layout for navigation buttons and label selection
    col1, col2 = st.sidebar.columns([1, 1])
    if col1.button("⬅️ Previous Image", use_container_width=True):
        prev_image(state)
    if col2.button("Next Image ➡️", use_container_width=True):
        next_image(state, image_files)

    st.sidebar.markdown("""---""")
    # Label buttons
    for label in LABELS:
        label_button = st.sidebar.button(
            label=f"{labels_dict[label]['button_text']} {labels_dict[label]['emoji']}",
            key=label,
            use_container_width=True,
            help=labels_dict[label]["help"],
        )  # Add your preferred emojis here
        if label == "Ok":
            st.sidebar.markdown(""" """)
            st.sidebar.markdown(""" """)
        
        if label_button:
            save_state(state)

            utils.update_labels(df_labels, image_files[state["selected_index"]], label)

            st.toast(
                f"Image {image_files[state['selected_index']]} was labeled as {label}"
            )
            next_image(state, image_files)

    st.sidebar.markdown("""---""")
    # Save buttons
    if st.sidebar.button("💾 Save State", use_container_width=True):
        save_state(state)
        st.sidebar.success("State saved.")

    st.sidebar.markdown(""" """)
    st.sidebar.markdown(""" """)
    number = st.sidebar.number_input('Image Index', min_value=1, 
                                     max_value=len(image_files), value=state["selected_index"]+1)
    if st.sidebar.button("Go to Image Index", use_container_width=True):
        state["selected_index"] = number - 1

    # Display image number
    st.write(f"Image {state['selected_index'] + 1} of {len(image_files)}")

    display_images(state["selected_index"], image_files)

    # Find the row for the current image
    row = df_labels[df_labels["Image"] == image_files[state["selected_index"]]]

    # If the row exists, display the previous label
    with st.expander("Previous Label and Predictions", expanded=True):
        if not row.empty:
            previous_label = row.iloc[0]["Label"]
            st.markdown(f"**Previous label:** {previous_label}")


if __name__ == "__main__":
    main()
