# Purpose: Visulaize a conversation from the given input

from PIL import Image, ImageDraw, ImageFont, ImageOps
import os

# Paths and settings for text font
FONT_PATH = '/usr/share/fonts/Urbanist-fonts/fonts/ttf/Urbanist-SemiBold.ttf'
FONT_SIZE = 60
IMAGE_WIDTH = 2400  # Width of the whole image
TEXT_WIDTH = IMAGE_WIDTH // 2  # Width of the text area
IMAGE_MAX_HEIGHT = 600  # Maximum height of an image message
PADDING = 20  # Padding around text
BORDER_WIDTH = 10  # Width of the border around text messages
BORDER_RADIUS = 10  # Radius of the rounded corners
PERSON_A_COLOR = (248, 179, 144)  # Red
PERSON_B_COLOR = (144, 194, 248)  # Blue
TEXT_COLOR = (0, 0, 0)  # Black
# Path to Person A's avatar
PERSON_A_AVATAR_PATH = '/fsx/users/haboutal/home/datasets/vis/avatar5.png'
# Path to Person B's avatar
PERSON_B_AVATAR_PATH = '/fsx/users/haboutal/home/datasets/vis/avatar6.png'
AVATAR_SIZE = (100, 100)  # Size of the avatar in pixels

# Function to lighten the color


def lighten_color(color, ratio=1.6):
    # Unpack the color into red, green, and blue components
    r, g, b = color
    # Lighten each component by the ratio
    r = min(int(r * ratio), 255)
    g = min(int(g * ratio), 255)
    b = min(int(b * ratio), 255)
    # Return the new color
    return r, g, b

# Function to draw a rounded rectangle


def draw_rounded_rect(draw, rect, radius, fill=None):
    # Lighten the fill color
    fill = lighten_color(fill or (255, 255, 255))
    # Unpack the rectangle bounds
    x1, y1, x2, y2 = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
    # Draw the rounded rectangle
    draw.rounded_rectangle([(x1, y1), (x2, y2)],
                           radius=radius, fill=fill, outline="black")

# Function to create a bordered image


def add_image_border(input_image, border, color=0):
    border_image = ImageOps.expand(input_image, border=border, fill=color)
    return border_image

# Function to wrap text to fit within a specified width


def wrap_text(text, font, max_width):
    lines = []
    # If the whole text can fit on one line, just add it as one line
    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        # Otherwise, we need to split the text into words and add words to the line until it's full
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            lines.append(line.strip())
    return lines

# Function to create a circular avatar from an image


def round_avatar(image):
    # Create a new image for the mask
    mask = Image.new('L', image.size)
    # Create a new Draw object to draw on the mask
    mask_draw = ImageDraw.Draw(mask)
    # Draw a filled circle on the mask
    mask_draw.ellipse((0, 0, *image.size), fill=255)
    # Convert the image to RGBA if necessary
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    # Apply the mask to the image
    image.putalpha(mask)
    # Return the image
    return image

# Function to create a gradient background


def gradient_background(width, height, top_color, bottom_color):
    # Create a new image for the base
    base = Image.new('RGB', (width, height), top_color)
    # Create a new image for the top
    top = Image.new('RGB', (width, height), top_color)
    # Create a new image for the bottom
    bottom = Image.new('RGB', (width, height), bottom_color)
    # Create a new image for the mask
    mask = Image.new('L', (width, height))
    # Create the gradient for the mask
    mask_data = []
    for y in range(height):
        mask_data.extend([int(255 * (y / height))] * width)
    mask.putdata(mask_data)
    # Paste the bottom color onto the base using the mask
    base.paste(bottom, (0, 0), mask)
    # Return the base
    return base


def draw_image_counter(draw, position, text, font, color=(0, 0, 0)):
    # Calculate text size to determine the circle size
    text_width, text_height = draw.textsize(text, font=font)

    # Define a circle diameter based on the text size
    circle_diameter = max(text_width, text_height) + \
        20  # Additional padding for the circle

    # Calculate circle top-left and bottom-right positions
    circle_top_left = (position[0] + text_width, position[1] +
                       text_height - circle_diameter - 20)  # moved 20 pixels upper
    circle_bottom_right = (position[0] + text_width + circle_diameter,
                           position[1] + text_height - 30)  # moved 20 pixels upper

    # Draw the circle with yellow fill
    draw.ellipse([circle_top_left, circle_bottom_right],
                 fill="#f5d0ba", outline=color)

    # Adjust text position to be centered within the circle
    text_position = (circle_top_left[0] + (circle_diameter - text_width) / 2,
                     circle_top_left[1] + (circle_diameter - text_height) / 2 - 10)  # moved 10 pixels upper

    # Draw the text
    draw.text(text_position, text, font=font, fill=color)

# Main function to visualize the conversation


def visualize_conversation(conversation, output_path):
    # Create a new image for the conversation
    images = []
    img_counter = 1
    # Load the font
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    # Load and process the avatars
    avatar_a = round_avatar(Image.open(
        PERSON_A_AVATAR_PATH).resize(AVATAR_SIZE))
    avatar_b = round_avatar(Image.open(
        PERSON_B_AVATAR_PATH).resize(AVATAR_SIZE))
    # Initialize the current height
    current_height = 0
    # Process each message in the conversation
    i = 0
    for message in conversation:
        msg_type, content = message
        # If the message is text
        if msg_type == 'txt':
            # Wrap the text
            lines = wrap_text(content, font, TEXT_WIDTH - 2 *
                              (BORDER_WIDTH + PADDING) - AVATAR_SIZE[0])
            # Calculate the height of the text box
            text_height = max(len(lines) * (FONT_SIZE + PADDING) + 2 *
                              (BORDER_WIDTH + PADDING), AVATAR_SIZE[1] + 2 * BORDER_WIDTH)
            # Create a new image for the text box
            text_image = Image.new(
                'RGB', (IMAGE_WIDTH, text_height), color=(255, 255, 255))
            # Create a new Draw object to draw on the text box
            draw = ImageDraw.Draw(text_image)
            # Calculate the start of the text box
            box_start = IMAGE_WIDTH - TEXT_WIDTH - BORDER_WIDTH - \
                PADDING if i % 2 == 0 else BORDER_WIDTH + PADDING
            # Draw the text box
            draw_rounded_rect(draw, [(box_start, BORDER_WIDTH),
                                     (box_start + TEXT_WIDTH, text_height - BORDER_WIDTH)],
                              BORDER_RADIUS, fill=[PERSON_A_COLOR, PERSON_B_COLOR][i % 2])
            # Calculate the position of the avatar
            avatar_position = (box_start + BORDER_WIDTH, 0)
            # Paste the avatar onto the text box
            text_image.paste([avatar_a, avatar_b][i % 2], avatar_position, mask=[
                             avatar_a, avatar_b][i % 2])
            # Calculate the position of the text
            text_position = box_start + BORDER_WIDTH + PADDING + AVATAR_SIZE[0]
            # Draw each line of text
            for j, line in enumerate(lines):
                draw.text((text_position, j * (FONT_SIZE + PADDING) +
                          BORDER_WIDTH + PADDING), line, fill=TEXT_COLOR, font=font)
            # Add the text box to the list of images
            images.append(text_image)
            # Update the current height
            current_height += text_height + PADDING
            i += 1
        # If the message is an image
        # If the message is an image
        # If the message is an image
        elif msg_type == 'img':
            i -= 1
            # Load the image
            img = Image.open(content)
            # Resize the image if necessary
            w, h = img.size
            if h > IMAGE_MAX_HEIGHT:
                img = img.resize(
                    (int(w * IMAGE_MAX_HEIGHT / h), IMAGE_MAX_HEIGHT))
            if h < IMAGE_MAX_HEIGHT*3/4:
                fixed_height = int(IMAGE_MAX_HEIGHT*3/4)
                img = img.resize((int(w * fixed_height / h), fixed_height))

            # Calculate the position of the avatar
            avatar_position = (box_start + BORDER_WIDTH, 0)

            # Create a new image for the img box
            # Set the width of the image box to be the same as the text box
            img_box_width = TEXT_WIDTH
            img_box_height = max(
                img.height + 2 * BORDER_WIDTH + 2 * PADDING, AVATAR_SIZE[1] + 2 * BORDER_WIDTH)
            img_box = Image.new(
                'RGB', (IMAGE_WIDTH, img_box_height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img_box)

            # Calculate the start of the img box
            box_start = IMAGE_WIDTH - img_box_width - BORDER_WIDTH - \
                PADDING if i % 2 == 0 else BORDER_WIDTH + PADDING
            # Draw the img box
            draw_rounded_rect(draw, [(box_start, BORDER_WIDTH),
                                     (box_start + img_box_width, img_box_height - BORDER_WIDTH)],
                              BORDER_RADIUS, fill=[PERSON_A_COLOR, PERSON_B_COLOR][i % 2])
            # Paste the avatar onto the img box
            img_box.paste([avatar_a, avatar_b][i % 2],
                          avatar_position, mask=[avatar_a, avatar_b][i % 2])

            # Paste the image onto the img box with padding
            img_position = box_start + BORDER_WIDTH + PADDING + AVATAR_SIZE[0]

            # Check if it's the first image and adjust the pasting position
            if img_counter == 1:
                img_position = box_start + BORDER_WIDTH + PADDING + \
                    AVATAR_SIZE[0] + 20  # 20 pixels to the right
            else:
                img_position = box_start + BORDER_WIDTH + \
                    PADDING + AVATAR_SIZE[0]

            img_box.paste(img, (img_position, BORDER_WIDTH + PADDING))

            # Add the image counter to the bottom right of the img box
            img_counter_font = ImageFont.truetype(
                FONT_PATH, FONT_SIZE)  # Making font smaller for counter
            draw_image_counter(draw, (img_position + img.width, BORDER_WIDTH +
                               img.height), str(img_counter), img_counter_font)

            img_counter += 1  # Increment the counter

            # Add the img box to the list of images
            images.append(img_box)

            # Update the current height
            current_height += img_box_height + PADDING
            i += 1

    # Create a new image for the final conversation
    final_image = Image.new(
        'RGB', (IMAGE_WIDTH, current_height), color=(255, 255, 255))
    # Paste each image onto the final conversation
    current_height = 0
    for img in images:
        final_image.paste(img, (0, current_height))
        current_height += img.height + PADDING
    # Save the final conversation
    final_image.save(output_path)


def combined_vis(image1_path, image2_path, label_1, label_2):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Get dimensions
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Define the width of the space
    space_width = 50

    # Create a new image with the combined width and the maximum height
    # Add some space for the labels
    label_height = 150
    new_img = Image.new('RGB', (width1 + width2 + space_width,
                        max(height1, height2) + label_height), (230, 230, 255))

    # Paste the images
    new_img.paste(img1, (0, label_height))
    new_img.paste(img2, (width1 + space_width, label_height))

    # Prepare to draw labels
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Draw labels
    draw.text((width1//2, 10), label_1, font=font, fill="black")
    draw.text((width1 + space_width + width2//2, 10),
              label_2, font=font, fill="black")

    # Get the directory path of image1
    dir_path = os.path.dirname(os.path.abspath(image1_path))

    # Construct the output path
    output_path = os.path.join(dir_path, 'combined.png')

    # Save the new image
    new_img.save(output_path)
